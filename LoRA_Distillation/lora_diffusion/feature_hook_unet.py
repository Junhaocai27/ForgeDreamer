import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

class FeatureHook:
    """用于捕获中间层特征的Hook类"""
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach().clone()
        return fn
    
    def register_hooks(self, model, layer_names):
        """为指定层注册hook"""
        self.clear_hooks()
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """清除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def get_features(self):
        return self.features

class UNetFeatureExtractor:
    def __init__(self, target_layers, mixed_precision_config=None):
        """
        初始化 UNetFeatureExtractor.
        Args:
            target_layers (list of str): 需要提取特征的目标层名称列表。
            mixed_precision_config (str or object, optional): 混合精度配置。
                可以是 "fp16", "bf16", "no"，或者一个 Accelerate 的 accelerator 对象，
                或者任何具有 'mixed_precision' 属性的对象。
        """
        if not isinstance(target_layers, list):
            # 在给 self.target_layers 赋值前进行检查
            raise ValueError(f"target_layers should be a list of layer names, got {type(target_layers)}")
        self.target_layers = target_layers
        self.features = {}  # 用于存储提取到的特征
        self.hooks = []     # 用于存储注册的hook句柄

        # 根据 mixed_precision_config 设置 self.mixed_precision_active
        self.mixed_precision_active = False
        if isinstance(mixed_precision_config, str):
            self.mixed_precision_active = (mixed_precision_config.lower() in ["fp16", "bf16"])
        elif hasattr(mixed_precision_config, 'mixed_precision') and mixed_precision_config.mixed_precision is not None:
            # 假设 mixed_precision_config 是一个 accelerator 对象或类似结构
            if isinstance(mixed_precision_config.mixed_precision, str):
                 self.mixed_precision_active = (mixed_precision_config.mixed_precision.lower() in ["fp16", "bf16"])
        # 你可以根据实际的配置方式调整这里的逻辑
        # print(f"UNetFeatureExtractor initialized with mixed_precision_active: {self.mixed_precision_active}")


    def _hook_fn(self, module, input_tuple, output, layer_name):
        """
        实际的hook函数，当被hook的层执行完毕后被调用。
        """
        feature_to_store = output
        if isinstance(output, tuple):
            # 如果是元组，通常我们取第一个元素作为主要特征。
            # 你可能需要根据具体模型的输出来调整这里的逻辑。
            # 确保元组的第一个元素确实是张量。
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                feature_to_store = output[0]
                # print(f"Info: Layer {layer_name} output was a tuple, taking first element.")
            else:
                # 如果元组为空或第一个元素不是张量，这可能是一个问题
                # print(f"Warning: Layer {layer_name} output is a tuple, but first element is not a Tensor or tuple is empty. Storing as is (type: {type(output[0]) if len(output)>0 else 'empty tuple'}).")
                # 保持原样，让下游处理或报错
                pass # feature_to_store 仍然是原始元组 output

        # 确保最终存储的是张量，或者下游能处理的类型
        # 如果 feature_to_store 仍然可能是元组，那么 FeatureAlignmentLoss 中仍需处理
        self.features[layer_name] = feature_to_store

    def register_hooks(self, model):
        """
        在指定模型的目标层上注册forward hooks。
        Args:
            model (torch.nn.Module): 需要注册hooks的UNet模型。
        """
        self.clear_hooks()  # 先清除可能存在的旧hooks
        self.features.clear() # 清除之前提取的特征

        if not self.target_layers:
            # print("Warning: No target layers specified for feature extraction.")
            return

        for name, module in model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(
                    # 使用 lambda n=name 来确保在hook触发时 layer_name 是正确的
                    lambda m, inp, outp, current_layer_name=name: self._hook_fn(m, inp, outp, current_layer_name)
                )
                self.hooks.append(hook)

    def get_features(self):
        """
        获取所有通过hooks捕获到的特征。
        Returns:
            dict: 一个字典，键是层名称，值是对应的特征张量。
        """
        return {k: v for k, v in self.features.items()} # 返回一个副本

    def clear_hooks(self):
        """
        移除所有已注册的hooks并清空列表。
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_features(self,
                         unet_model: torch.nn.Module,
                         sample: torch.Tensor,
                         timestep: torch.Tensor,
                         encoder_hidden_states: torch.Tensor = None,
                         cross_attention_kwargs=None,
                         return_dict: bool = True,
                         use_grad: bool = False, # 新增：控制是否在 grad 上下文中运行
                         **other_unet_specific_kwargs):
        """
        执行UNet的前向传播并提取指定层的中间特征。
        Args:
            use_grad (bool): 如果为 True，则在 torch.enable_grad() 上下文中运行UNet，
                             允许为学生模型提取可微分的特征。默认为 False (在 no_grad() 中运行)。
        """
        # 1. 注册hooks
        self.register_hooks(unet_model)

        # 2. 确定UNet的实际参数数据类型
        try:
            unet_internal_dtype = next(unet_model.parameters()).dtype
        except StopIteration:
            try:
                unet_internal_dtype = next(unet_model.buffers()).dtype
            except StopIteration:
                unet_internal_dtype = sample.dtype

        # 3. 准备输入张量的数据类型
        input_sample = sample.to(dtype=unet_internal_dtype)
        input_encoder_hidden_states = None
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                input_encoder_hidden_states = [
                    ehs.to(dtype=unet_internal_dtype) if ehs is not None else None
                    for ehs in encoder_hidden_states
                ]
            elif isinstance(encoder_hidden_states, torch.Tensor):
                input_encoder_hidden_states = encoder_hidden_states.to(dtype=unet_internal_dtype)
            else:
                input_encoder_hidden_states = encoder_hidden_states

        processed_cross_attention_kwargs = None
        if cross_attention_kwargs:
            processed_cross_attention_kwargs = {}
            for key, value in cross_attention_kwargs.items():
                if isinstance(value, torch.Tensor):
                    processed_cross_attention_kwargs[key] = value.to(dtype=unet_internal_dtype)
                else:
                    processed_cross_attention_kwargs[key] = value

        # 4. 执行前向传播
        main_output = None
        # 根据 use_grad 选择上下文
        context_manager = torch.enable_grad() if use_grad else torch.no_grad()

        with context_manager:
            # autocast 仅在 CUDA 上且 mixed_precision_active 为 True 时启用
            autocast_enabled = self.mixed_precision_active and unet_model.device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output_data = unet_model(
                    input_sample,
                    timestep,
                    encoder_hidden_states=input_encoder_hidden_states,
                    cross_attention_kwargs=processed_cross_attention_kwargs,
                    return_dict=return_dict,
                    **other_unet_specific_kwargs
                )

                # 5. 从UNet的输出中获取主要的预测
                if return_dict:
                    if not hasattr(output_data, "sample"):
                        raise ValueError("UNet returned a dictionary but it does not have a 'sample' attribute.")
                    main_output = output_data.sample
                elif isinstance(output_data, tuple) and len(output_data) > 0:
                    main_output = output_data[0]
                else:
                    main_output = output_data
        
        # 6. 获取通过hooks捕获的特征
        # self.features 是在 _hook_fn 中填充的。
        # 如果 use_grad 为 True，这些特征张量将连接到计算图。
        intermediate_features = self.get_features()

        # 7. 清除hooks (在获取特征之后清除)
        self.clear_hooks()

        return main_output, intermediate_features

    def __enter__(self):
        # print("Warning: UNetFeatureExtractor context manager usage is not fully implemented for automatic model hooking.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()

class FeatureAlignmentLoss(nn.Module): # 最好继承自 nn.Module
    def __init__(self, alignment_layers: List[str], loss_weights: Dict[str, float] = None, temperature: float = 1.0, loss_type: str = "mse"):
        """
        初始化特征对齐损失模块。
        Args:
            alignment_layers (List[str]): 需要对齐特征的层名称列表。
            loss_weights (Dict[str, float], optional): 每层损失的权重。默认为None (所有权重为1.0)。
            temperature (float, optional): 用于缩放特征的温度（如果需要）。默认为1.0。
            loss_type (str, optional): 使用的损失类型，可以是 "mse" 或 "l1"。默认为 "mse"。
        """
        super().__init__() # 调用父类的 __init__
        self.alignment_layers = alignment_layers
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.temperature = temperature # 虽然目前没在forward中使用，但保留定义

        # ----> 关键修改点：初始化 self.criterion <----
        if loss_type.lower() == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss_type.lower() == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'mse' or 'l1'.")
        # -------------------------------------------
        print(f"FeatureAlignmentLoss initialized with loss_type: {loss_type}")


    def _adaptive_pool_features(self, feat1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                  feat2: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        # ... (你之前的 _adaptive_pool_features 实现，确保它处理元组并返回张量)
        # print(f"\nDEBUG: _adaptive_pool_features entered.")
        # print(f"DEBUG:   Initial feat1 type: {type(feat1)}" + (f", Shape: {feat1.shape}" if isinstance(feat1, torch.Tensor) else ""))
        # print(f"DEBUG:   Initial feat2 type: {type(feat2)}" + (f", Shape: {feat2.shape}" if isinstance(feat2, torch.Tensor) else ""))

        # Attempt to process feat1 if it's a tuple
        if isinstance(feat1, tuple):
            # print(f"DEBUG:   feat1 is a tuple. Attempting to extract tensor.")
            original_feat1_tuple = feat1
            found_tensor_in_feat1 = False
            if len(feat1) > 0:
                for i, el in enumerate(feat1):
                    if isinstance(el, torch.Tensor):
                        feat1 = el
                        found_tensor_in_feat1 = True
                        # print(f"DEBUG:     Extracted tensor from feat1 tuple at index {i}. New feat1 type: {type(feat1)}, Shape: {feat1.shape}")
                        break
            if not found_tensor_in_feat1:
                # print(f"DEBUG:     Could not extract tensor from feat1 tuple: {original_feat1_tuple}. feat1 remains a tuple.")
                raise TypeError(f"CRITICAL: feat1 is a tuple and no tensor found within it. Value: {str(original_feat1_tuple)[:200]}")


        # Attempt to process feat2 if it's a tuple
        if isinstance(feat2, tuple):
            # print(f"DEBUG:   feat2 is a tuple. Attempting to extract tensor.")
            original_feat2_tuple = feat2
            found_tensor_in_feat2 = False
            if len(feat2) > 0:
                for i, el in enumerate(feat2):
                    if isinstance(el, torch.Tensor):
                        feat2 = el
                        found_tensor_in_feat2 = True
                        # print(f"DEBUG:     Extracted tensor from feat2 tuple at index {i}. New feat2 type: {type(feat2)}, Shape: {feat2.shape}")
                        break
            if not found_tensor_in_feat2:
                # print(f"DEBUG:     Could not extract tensor from feat2 tuple: {original_feat2_tuple}. feat2 remains a tuple.")
                raise TypeError(f"CRITICAL: feat2 is a tuple and no tensor found within it. Value: {str(original_feat2_tuple)[:200]}")


        if not isinstance(feat1, torch.Tensor):
            raise TypeError(f"CRITICAL: feat1 is not a torch.Tensor after tuple processing. Type: {type(feat1)}. Value: {str(feat1)[:200]}")
        if not isinstance(feat2, torch.Tensor):
            raise TypeError(f"CRITICAL: feat2 is not a torch.Tensor after tuple processing. Type: {type(feat2)}. Value: {str(feat2)[:200]}")

        # print(f"DEBUG:   After tuple processing, feat1 type: {type(feat1)}, Shape: {feat1.shape}")
        # print(f"DEBUG:   After tuple processing, feat2 type: {type(feat2)}, Shape: {feat2.shape}")
        
        # Pooling logic (simplified for brevity, use your more complete one)
        if feat1.ndim >=3 and feat2.ndim >=3 and feat1.shape[2:] != feat2.shape[2:]: # Check spatial/sequence dims
            if feat1.ndim == 4 and feat2.ndim == 4: # (B, C, H, W)
                pool_h, pool_w = feat2.shape[2], feat2.shape[3]
                # print(f"DEBUG:     Pooling feat1 (4D) from {feat1.shape[2:]} to {(pool_h, pool_w)}")
                feat1_pooled = F.adaptive_avg_pool2d(feat1, (pool_h, pool_w))
                return feat1_pooled, feat2
            elif feat1.ndim == 3 and feat2.ndim == 3: # (B, L, D) -> (B, D, L) for pool1d
                 if feat1.shape[1] != feat2.shape[1]: # Compare L
                    target_seq_len = feat2.shape[1]
                    # print(f"DEBUG:     Pooling feat1 (3D) from seq_len {feat1.shape[1]} to {target_seq_len}")
                    feat1_permuted = feat1.permute(0, 2, 1)
                    feat1_pooled_permuted = F.adaptive_avg_pool1d(feat1_permuted, target_seq_len)
                    feat1_pooled = feat1_pooled_permuted.permute(0, 2, 1)
                    return feat1_pooled, feat2
        return feat1, feat2


    def forward(self, teacher_features: Dict[str, torch.Tensor], student_features: Dict[str, torch.Tensor]):
        total_loss = torch.tensor(0.0, dtype=torch.float32) # Initialize as float32
        # Move total_loss to the device of the features if features are available
        if teacher_features and isinstance(list(teacher_features.values())[0], torch.Tensor):
            total_loss = total_loss.to(list(teacher_features.values())[0].device)
        elif student_features and isinstance(list(student_features.values())[0], torch.Tensor):
             total_loss = total_loss.to(list(student_features.values())[0].device)
        # else: it remains on CPU, which might be an issue if features are on GPU later

        loss_dict = {}
        active_layers_count = 0

        for layer_name in self.alignment_layers:
            if layer_name not in teacher_features or teacher_features[layer_name] is None:
                continue
            if layer_name not in student_features or student_features[layer_name] is None:
                continue

            t_feat_orig = teacher_features[layer_name]
            s_feat_orig = student_features[layer_name]

            if not (isinstance(t_feat_orig, torch.Tensor) or isinstance(t_feat_orig, tuple)) or \
               not (isinstance(s_feat_orig, torch.Tensor) or isinstance(s_feat_orig, tuple)):
                print(f"Warning: Features for layer {layer_name} are not Tensors or Tuples. Skipping. T:{type(t_feat_orig)}, S:{type(s_feat_orig)}")
                continue
            
            try:
                teacher_feat_processed, student_feat_processed = self._adaptive_pool_features(
                    t_feat_orig, s_feat_orig
                )
            except TypeError as e: # Catch specific error from _adaptive_pool_features if tensor extraction fails
                print(f"Error in _adaptive_pool_features for layer {layer_name}: {e}. Skipping layer.")
                continue

            if teacher_feat_processed is None or student_feat_processed is None:
                continue

            s_feat_for_loss = student_feat_processed.float()
            t_feat_for_loss = teacher_feat_processed.float()
            
            # print(f"DEBUG FeatureAlignmentLoss: Layer: {layer_name}, S_dtype: {s_feat_for_loss.dtype}, T_dtype: {t_feat_for_loss.dtype}, S_shape: {s_feat_for_loss.shape}, T_shape: {t_feat_for_loss.shape}")

            # 这是第 286 行
            layer_loss = self.criterion(s_feat_for_loss, t_feat_for_loss)
            
            weight = self.loss_weights.get(layer_name, 1.0)
            weighted_loss = weight * layer_loss
            
            total_loss += weighted_loss
            loss_dict[layer_name] = weighted_loss.item()
            active_layers_count += 1
        
        if active_layers_count == 0:
            # print("Warning: No active layers for feature alignment loss calculation.")
            # Return a zero tensor that requires_grad=False, or handle as error
            return torch.tensor(0.0, device=total_loss.device, dtype=torch.float32), loss_dict

        return total_loss, loss_dict

def modified_forward_with_features(unet, sample, timestep, encoder_hidden_states, **kwargs):
    """修改后的UNet前向传播，同时返回中间特征"""
    features = {}
    
    # 原始的UNet前向传播逻辑，但在关键点保存特征
    # 这里需要根据具体的UNet实现来修改
    
    # 示例：假设我们可以访问UNet的内部结构
    sample = unet.conv_in(sample)
    features['conv_in'] = sample.clone()
    
    # 下采样
    down_block_res_samples = []
    for i, downsample_block in enumerate(unet.down_blocks):
        sample, res_samples = downsample_block(sample, timestep, encoder_hidden_states)
        down_block_res_samples.extend(res_samples)
        features[f'down_blocks.{i}'] = sample.clone()
    
    # 中间块
    sample = unet.mid_block(sample, timestep, encoder_hidden_states)
    features['mid_block'] = sample.clone()
    
    # 上采样
    for i, upsample_block in enumerate(unet.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
        sample = upsample_block(sample, res_samples, timestep, encoder_hidden_states)
        features[f'up_blocks.{i}'] = sample.clone()
    
    # 输出
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)
    
    return sample, features

def compute_distillation_loss_with_features(
    teacher1_unet,
    teacher2_unet,
    student_unet,
    latents,
    timesteps,
    encoder_hidden_states1,
    encoder_hidden_states2,
    noise,
    feature_alignment_loss_fn,
    noise_pred_weight=1.0,
    feature_align_weight=0.5
):
    """计算包含特征对齐的蒸馏损失"""
    
    # Teacher 1 前向传播
    with torch.no_grad():
        teacher1_noise_pred, teacher1_features = modified_forward_with_features(
            teacher1_unet, latents, timesteps, encoder_hidden_states1
        )
    
    # Teacher 2 前向传播
    with torch.no_grad():
        teacher2_noise_pred, teacher2_features = modified_forward_with_features(
            teacher2_unet, latents, timesteps, encoder_hidden_states2
        )
    
    # Student 前向传播
    student_noise_pred, student_features = modified_forward_with_features(
        student_unet, latents, timesteps, encoder_hidden_states1  # 或者混合两个编码器的输出
    )
    
    # 计算噪声预测损失
    target_noise = (teacher1_noise_pred + teacher2_noise_pred) / 2  # 平均两个teacher的输出
    noise_pred_loss = F.mse_loss(student_noise_pred, target_noise)
    
    # 计算特征对齐损失
    # 与teacher1的特征对齐
    feature_loss1, feature_loss_dict1 = feature_alignment_loss_fn(
        teacher1_features, student_features
    )
    
    # 与teacher2的特征对齐
    feature_loss2, feature_loss_dict2 = feature_alignment_loss_fn(
        teacher2_features, student_features
    )
    
    # 平均特征对齐损失
    feature_align_loss = (feature_loss1 + feature_loss2) / 2
    
    # 总损失
    total_loss = (
        noise_pred_weight * noise_pred_loss + 
        feature_align_weight * feature_align_loss
    )
    
    loss_dict = {
        'noise_pred_loss': noise_pred_loss.item(),
        'feature_align_loss': feature_align_loss.item(),
        'total_loss': total_loss.item(),
        **{f"t1_{k}": v for k, v in feature_loss_dict1.items()},
        **{f"t2_{k}": v for k, v in feature_loss_dict2.items()}
    }
    
    return total_loss, loss_dict

class EnhancedFeatureAlignmentLoss(nn.Module):
    def __init__(self, 
                 alignment_layers: List[str], 
                 loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, 
                 loss_type: str = "mse",
                 feature_selection_strategy: str = "adaptive",  # 新增
                 normalize_features: bool = True,  # 新增
                 channel_alignment: str = "projection"):  # 新增
        """
        增强的特征对齐损失模块
        
        Args:
            feature_selection_strategy: "first", "last", "adaptive", "attention"
            normalize_features: 是否对特征进行归一化
            channel_alignment: "projection", "interpolation", "none"
        """
        super().__init__()
        self.alignment_layers = alignment_layers
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.temperature = temperature
        self.feature_selection_strategy = feature_selection_strategy
        self.normalize_features = normalize_features
        self.channel_alignment = channel_alignment
        
        # 损失函数
        if loss_type.lower() == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss_type.lower() == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        elif loss_type.lower() == "cosine":
            self.criterion = nn.CosineEmbeddingLoss(reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        self.loss_type = loss_type.lower()
        
        # 用于通道对齐的投影层
        self.channel_projectors = nn.ModuleDict()
        
        print(f"EnhancedFeatureAlignmentLoss initialized:")
        print(f"  • Loss type: {loss_type}")
        print(f"  • Feature selection: {feature_selection_strategy}")
        print(f"  • Normalize features: {normalize_features}")
        print(f"  • Channel alignment: {channel_alignment}")

    def _extract_tensor_from_tuple(self, 
                                   feature: Union[torch.Tensor, Tuple], 
                                   layer_name: str) -> torch.Tensor:
        """
        从元组中智能提取tensor特征
        """
        if isinstance(feature, torch.Tensor):
            return feature
        
        if not isinstance(feature, tuple) or len(feature) == 0:
            raise TypeError(f"Layer {layer_name}: Feature is not a tensor or tuple, got {type(feature)}")
        
        tensors = [f for f in feature if isinstance(f, torch.Tensor)]
        
        if len(tensors) == 0:
            raise TypeError(f"Layer {layer_name}: No tensors found in tuple of length {len(feature)}")
        
        if self.feature_selection_strategy == "first":
            return tensors[0]
        elif self.feature_selection_strategy == "last":
            return tensors[-1]
        elif self.feature_selection_strategy == "adaptive":
            # 选择具有最多信息的tensor（通常是最大的）
            return max(tensors, key=lambda x: x.numel())
        elif self.feature_selection_strategy == "attention":
            # 使用注意力权重组合多个tensor
            return self._attention_combine_tensors(tensors, layer_name)
        else:
            return tensors[0]  # 默认取第一个

    def _attention_combine_tensors(self, 
                                   tensors: List[torch.Tensor], 
                                   layer_name: str) -> torch.Tensor:
        """
        使用注意力机制组合多个tensor
        """
        if len(tensors) == 1:
            return tensors[0]
        
        # 将所有tensor reshape到相同的空间维度
        reference_shape = tensors[0].shape
        aligned_tensors = []
        
        for tensor in tensors:
            if tensor.shape != reference_shape:
                # 简单的自适应池化对齐
                if tensor.ndim == 4:  # (B, C, H, W)
                    aligned = F.adaptive_avg_pool2d(tensor, reference_shape[2:])
                elif tensor.ndim == 3:  # (B, L, D)
                    aligned = F.adaptive_avg_pool1d(
                        tensor.transpose(1, 2), reference_shape[1]
                    ).transpose(1, 2)
                else:
                    aligned = tensor
                aligned_tensors.append(aligned)
            else:
                aligned_tensors.append(tensor)
        
        # 计算注意力权重
        attention_weights = []
        for tensor in aligned_tensors:
            # 使用tensor的方差作为重要性权重
            weight = torch.var(tensor, dim=list(range(1, tensor.ndim)), keepdim=True)
            attention_weights.append(weight.mean())
        
        # 归一化权重
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            attention_weights = [1.0 / len(tensors)] * len(tensors)
        
        # 加权组合
        combined = sum(w * t for w, t in zip(attention_weights, aligned_tensors))
        return combined

    def _align_channel_dimensions(self, 
                                  teacher_feat: torch.Tensor, 
                                  student_feat: torch.Tensor, 
                                  layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐通道维度
        """
        if teacher_feat.shape[1] == student_feat.shape[1]:
            return teacher_feat, student_feat
        
        if self.channel_alignment == "none":
            # 截断到较小的通道数
            min_channels = min(teacher_feat.shape[1], student_feat.shape[1])
            return teacher_feat[:, :min_channels], student_feat[:, :min_channels]
        
        elif self.channel_alignment == "interpolation":
            # 使用插值调整通道数
            target_channels = teacher_feat.shape[1]
            if student_feat.shape[1] != target_channels:
                # 通过1x1卷积调整通道数
                if layer_name not in self.channel_projectors:
                    self.channel_projectors[layer_name] = nn.Conv2d(
                        student_feat.shape[1], target_channels, 1, bias=False
                    ).to(student_feat.device)
                
                if student_feat.ndim == 4:
                    student_feat = self.channel_projectors[layer_name](student_feat)
                else:
                    # 对于3D tensor，需要添加空间维度
                    original_shape = student_feat.shape
                    reshaped = student_feat.view(original_shape[0], original_shape[1], 1, -1)
                    projected = self.channel_projectors[layer_name](reshaped)
                    student_feat = projected.view(original_shape[0], target_channels, -1)
        
        elif self.channel_alignment == "projection":
            # 使用线性投影对齐
            target_channels = teacher_feat.shape[1]
            if student_feat.shape[1] != target_channels:
                projector_key = f"{layer_name}_proj"
                if projector_key not in self.channel_projectors:
                    self.channel_projectors[projector_key] = nn.Linear(
                        student_feat.shape[1], target_channels, bias=False
                    ).to(student_feat.device)
                
                # 重塑和投影
                original_shape = student_feat.shape
                flattened = student_feat.view(-1, original_shape[1])
                projected = self.channel_projectors[projector_key](flattened)
                student_feat = projected.view(original_shape[0], target_channels, *original_shape[2:])
        
        return teacher_feat, student_feat

    def _adaptive_spatial_alignment(self, 
                                    teacher_feat: torch.Tensor, 
                                    student_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的空间维度对齐
        """
        if teacher_feat.shape[2:] == student_feat.shape[2:]:
            return teacher_feat, student_feat
        
        # 对齐到teacher的空间维度
        target_shape = teacher_feat.shape[2:]
        
        if student_feat.ndim == 4:  # (B, C, H, W)
            student_feat_aligned = F.adaptive_avg_pool2d(student_feat, target_shape)
        elif student_feat.ndim == 3:  # (B, L, D) 或 (B, D, L)
            if len(target_shape) == 1:  # 序列长度对齐
                student_feat_permuted = student_feat.transpose(1, 2)
                aligned_permuted = F.adaptive_avg_pool1d(student_feat_permuted, target_shape[0])
                student_feat_aligned = aligned_permuted.transpose(1, 2)
            else:
                student_feat_aligned = student_feat
        else:
            student_feat_aligned = student_feat
        
        return teacher_feat, student_feat_aligned

    def _normalize_features(self, 
                           teacher_feat: torch.Tensor, 
                           student_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        特征归一化
        """
        if not self.normalize_features:
            return teacher_feat, student_feat
        
        # L2归一化
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        student_norm = F.normalize(student_feat, p=2, dim=1)
        
        return teacher_norm, student_norm

    def _compute_layer_loss(self, 
                           teacher_feat: torch.Tensor, 
                           student_feat: torch.Tensor) -> torch.Tensor:
        """
        计算单层的对齐损失
        """
        # 确保数据类型一致
        teacher_feat = teacher_feat.float()
        student_feat = student_feat.float()
        
        # 检查数值稳定性
        if torch.isnan(teacher_feat).any() or torch.isnan(student_feat).any():
            print("Warning: NaN detected in features")
            return torch.tensor(0.0, device=teacher_feat.device)
        
        if torch.isinf(teacher_feat).any() or torch.isinf(student_feat).any():
            print("Warning: Inf detected in features")
            return torch.tensor(0.0, device=teacher_feat.device)
        
        # 根据损失类型计算
        if self.loss_type == "cosine":
            # 余弦嵌入损失需要target参数
            target = torch.ones(teacher_feat.shape[0], device=teacher_feat.device)
            
            # 展平特征
            teacher_flat = teacher_feat.view(teacher_feat.shape[0], -1)
            student_flat = student_feat.view(student_feat.shape[0], -1)
            
            loss = self.criterion(student_flat, teacher_flat, target)
        else:
            loss = self.criterion(student_feat, teacher_feat)
        
        return loss

    def forward(self, 
                teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        增强的前向传播
        """
        device = None
        # 确定设备
        for features in [teacher_features, student_features]:
            if features:
                first_feat = next(iter(features.values()))
                if isinstance(first_feat, torch.Tensor):
                    device = first_feat.device
                    break
        
        if device is None:
            device = torch.device('cpu')
        
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_dict = {}
        active_layers_count = 0
        
        for layer_name in self.alignment_layers:
            try:
                # 检查特征是否存在
                if (layer_name not in teacher_features or 
                    layer_name not in student_features or
                    teacher_features[layer_name] is None or 
                    student_features[layer_name] is None):
                    continue
                
                # 提取tensor特征
                teacher_feat = self._extract_tensor_from_tuple(
                    teacher_features[layer_name], layer_name
                )
                student_feat = self._extract_tensor_from_tuple(
                    student_features[layer_name], layer_name
                )
                
                # 特征对齐流程
                # 1. 通道维度对齐
                teacher_feat, student_feat = self._align_channel_dimensions(
                    teacher_feat, student_feat, layer_name
                )
                
                # 2. 空间维度对齐
                teacher_feat, student_feat = self._adaptive_spatial_alignment(
                    teacher_feat, student_feat
                )
                
                # 3. 特征归一化
                teacher_feat, student_feat = self._normalize_features(
                    teacher_feat, student_feat
                )
                
                # 4. 计算损失
                layer_loss = self._compute_layer_loss(teacher_feat, student_feat)
                
                # 5. 应用权重
                weight = self.loss_weights.get(layer_name, 1.0)
                weighted_loss = weight * layer_loss
                
                total_loss += weighted_loss
                loss_dict[layer_name] = weighted_loss.item()
                active_layers_count += 1
                
            except Exception as e:
                print(f"Error processing layer {layer_name}: {e}")
                continue
        
        # 处理没有活跃层的情况
        if active_layers_count == 0:
            print("Warning: No active layers for feature alignment loss calculation.")
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True), loss_dict
        
        # 平均损失
        if active_layers_count > 1:
            total_loss = total_loss / active_layers_count
        
        return total_loss, loss_dict
    
# === 使用示例 ===
def create_enhanced_feature_alignment_loss(
    alignment_layers: List[str],
    loss_weights: Dict[str, float] = None,
    **kwargs
) -> EnhancedFeatureAlignmentLoss:
    """
    创建增强的特征对齐损失函数
    """
    default_kwargs = {
        'temperature': 1.0,
        'loss_type': 'mse',
        'feature_selection_strategy': 'adaptive',
        'normalize_features': True,
        'channel_alignment': 'projection'
    }
    default_kwargs.update(kwargs)
    
    return EnhancedFeatureAlignmentLoss(
        alignment_layers=alignment_layers,
        loss_weights=loss_weights,
        **default_kwargs
    )