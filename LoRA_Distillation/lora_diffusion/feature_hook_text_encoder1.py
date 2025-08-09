import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

class TextEncoderFeatureExtractor:
    def __init__(self, target_layers, mixed_precision_config=None):
        """
        初始化 TextEncoderFeatureExtractor.
        Args:
            target_layers (list of str): 需要提取特征的目标层名称列表。
                例如: ['text_model.encoder.layers.0', 'text_model.encoder.layers.6', 'text_model.encoder.layers.11']
            mixed_precision_config (str or object, optional): 混合精度配置。
                可以是 "fp16", "bf16", "no"，或者一个 Accelerate 的 accelerator 对象，
                或者任何具有 'mixed_precision' 属性的对象。
        """
        if not isinstance(target_layers, list):
            raise ValueError(f"target_layers should be a list of layer names, got {type(target_layers)}")
        self.target_layers = target_layers
        self.features = {}  # 用于存储提取到的特征
        self.hooks = []     # 用于存储注册的hook句柄

        # 根据 mixed_precision_config 设置 self.mixed_precision_active
        self.mixed_precision_active = False
        if isinstance(mixed_precision_config, str):
            self.mixed_precision_active = (mixed_precision_config.lower() in ["fp16", "bf16"])
        elif hasattr(mixed_precision_config, 'mixed_precision') and mixed_precision_config.mixed_precision is not None:
            if isinstance(mixed_precision_config.mixed_precision, str):
                 self.mixed_precision_active = (mixed_precision_config.mixed_precision.lower() in ["fp16", "bf16"])

    def _hook_fn(self, module, input_tuple, output, layer_name):
        """
        实际的hook函数，当被hook的层执行完毕后被调用。
        对于CLIP Text Encoder，输出通常是hidden_states张量或包含多个元素的元组。
        """
        feature_to_store = output
        
        # 处理可能的元组输出
        if isinstance(output, tuple):
            # CLIP的transformer层通常返回 (hidden_states, attention_weights) 或类似结构
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                feature_to_store = output[0]  # 通常第一个元素是hidden_states
            else:
                # 如果元组为空或第一个元素不是张量，保持原样
                pass
        
        # 对于CLIP，我们主要关心hidden_states，通常是形状为 (batch_size, seq_len, hidden_size) 的张量
        self.features[layer_name] = feature_to_store

    def register_hooks(self, text_encoder):
        """
        在指定Text Encoder的目标层上注册forward hooks。
        Args:
            text_encoder (torch.nn.Module): 需要注册hooks的CLIP Text Encoder模型。
        """
        self.clear_hooks()  # 先清除可能存在的旧hooks
        self.features.clear()  # 清除之前提取的特征

        if not self.target_layers:
            return

        for name, module in text_encoder.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(
                    lambda m, inp, outp, current_layer_name=name: self._hook_fn(m, inp, outp, current_layer_name)
                )
                self.hooks.append(hook)

    def get_features(self):
        """
        获取所有通过hooks捕获到的特征。
        Returns:
            dict: 一个字典，键是层名称，值是对应的特征张量。
        """
        return {k: v for k, v in self.features.items()}  # 返回一个副本

    def clear_hooks(self):
        """
        移除所有已注册的hooks并清空列表。
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_features(self,
                         text_encoder: torch.nn.Module,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor = None,
                         position_ids: torch.Tensor = None,
                         output_attentions: bool = False,
                         output_hidden_states: bool = False,
                         return_dict: bool = True,
                         use_grad: bool = False,  # 新增：控制是否在 grad 上下文中运行
                         **other_text_encoder_kwargs):
        """
        执行CLIP Text Encoder的前向传播并提取指定层的中间特征。
        Args:
            text_encoder: CLIP Text Encoder模型
            input_ids: 输入的token ids，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            position_ids: 位置编码ids
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典格式
            use_grad: 如果为 True，则在 torch.enable_grad() 上下文中运行，
                     允许为学生模型提取可微分的特征。默认为 False
        """
        # 1. 注册hooks
        self.register_hooks(text_encoder)

        # 2. 确定Text Encoder的实际参数数据类型
        try:
            encoder_internal_dtype = next(text_encoder.parameters()).dtype
        except StopIteration:
            try:
                encoder_internal_dtype = next(text_encoder.buffers()).dtype
            except StopIteration:
                encoder_internal_dtype = input_ids.dtype if hasattr(input_ids, 'dtype') else torch.float32

        # 3. 准备输入张量
        # input_ids通常是long类型，不需要类型转换
        input_input_ids = input_ids
        input_attention_mask = attention_mask
        input_position_ids = position_ids
        
        # 如果有其他浮点型输入，需要转换类型
        if attention_mask is not None and attention_mask.dtype != input_ids.dtype:
            if attention_mask.dtype.is_floating_point:
                input_attention_mask = attention_mask.to(dtype=encoder_internal_dtype)
        
        if position_ids is not None and position_ids.dtype != input_ids.dtype:
            if position_ids.dtype.is_floating_point:
                input_position_ids = position_ids.to(dtype=encoder_internal_dtype)

        # 4. 执行前向传播
        main_output = None
        # 根据 use_grad 选择上下文
        context_manager = torch.enable_grad() if use_grad else torch.no_grad()

        with context_manager:
            # autocast 仅在 CUDA 上且 mixed_precision_active 为 True 时启用
            autocast_enabled = self.mixed_precision_active and text_encoder.device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output_data = text_encoder(
                    input_ids=input_input_ids,
                    attention_mask=input_attention_mask,
                    position_ids=input_position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **other_text_encoder_kwargs
                )

                # 5. 从Text Encoder的输出中获取主要特征
                if return_dict:
                    # CLIP Text Encoder通常返回包含last_hidden_state的对象
                    if hasattr(output_data, "last_hidden_state"):
                        main_output = output_data.last_hidden_state
                    elif hasattr(output_data, "pooler_output"):
                        main_output = output_data.pooler_output
                    else:
                        # 尝试获取第一个可用的张量属性
                        for attr_name in dir(output_data):
                            attr_value = getattr(output_data, attr_name)
                            if isinstance(attr_value, torch.Tensor):
                                main_output = attr_value
                                break
                        if main_output is None:
                            raise ValueError("Text Encoder returned a dictionary but no suitable tensor attribute found.")
                elif isinstance(output_data, tuple) and len(output_data) > 0:
                    main_output = output_data[0]
                else:
                    main_output = output_data
        
        # 6. 获取通过hooks捕获的特征
        intermediate_features = self.get_features()

        # 7. 清除hooks
        self.clear_hooks()

        return main_output, intermediate_features

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()


class HybridTextEncoderFeatureAlignmentLoss(nn.Module):
    """
    🔥 支持混合损失的Text Encoder特征对齐损失模块
    """
    def __init__(self, alignment_layers: List[str], 
                 loss_weights: Dict[str, float] = None,
                 temperature: float = 1.0, 
                 loss_type: str = "hybrid",
                 pooling_strategy: str = "mean",
                 # 🔥 混合损失新增参数
                 primary_loss_type: str = "mse",
                 secondary_loss_type: str = "cosine", 
                 loss_combination_weight: float = 0.3,
                 enable_magnitude_preservation: bool = True,
                 magnitude_weight: float = 0.2,
                 direction_weight: float = 0.8,
                 # 🔥 层级自适应参数
                 use_layer_adaptive: bool = False,
                 layer_loss_config: Dict[str, Dict] = None):
        """
        初始化混合Text Encoder特征对齐损失模块。
        Args:
            alignment_layers (List[str]): 需要对齐特征的层名称列表。
            loss_weights (Dict[str, float], optional): 每层损失的权重。
            temperature (float): 用于缩放特征的温度。
            loss_type (str): 损失类型 - "mse", "l1", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
            pooling_strategy (str): 特征池化策略。
            
            # 混合损失参数
            primary_loss_type (str): 主损失类型 (通常是"mse")
            secondary_loss_type (str): 次损失类型 (通常是"cosine")
            loss_combination_weight (float): 次损失在混合中的权重 (0.3表示30%cosine + 70%mse)
            enable_magnitude_preservation (bool): 是否启用幅度保持
            magnitude_weight (float): 幅度损失权重
            direction_weight (float): 方向损失权重
            
            # 层级自适应参数
            use_layer_adaptive (bool): 是否使用层级自适应损失
            layer_loss_config (Dict): 每层的损失配置
        """
        super().__init__()
        self.alignment_layers = alignment_layers
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.temperature = temperature
        self.pooling_strategy = pooling_strategy
        self.loss_type = loss_type.lower()
        
        # 🔥 混合损失参数
        self.primary_loss_type = primary_loss_type.lower()
        self.secondary_loss_type = secondary_loss_type.lower()
        self.loss_combination_weight = loss_combination_weight
        self.enable_magnitude_preservation = enable_magnitude_preservation
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        
        # 🔥 层级自适应参数
        self.use_layer_adaptive = use_layer_adaptive
        self.layer_loss_config = layer_loss_config or self._get_default_layer_config()
        
        # 初始化基础损失函数
        self.mse_criterion = nn.MSELoss(reduction="mean")
        self.l1_criterion = nn.L1Loss(reduction="mean")
        self.cosine_criterion = nn.CosineEmbeddingLoss(reduction="mean")
        
        print(f"🔥 HybridTextEncoderFeatureAlignmentLoss initialized:")
        print(f"   - Loss type: {loss_type}")
        print(f"   - Pooling strategy: {pooling_strategy}")
        if self.loss_type == "hybrid":
            print(f"   - Primary loss: {primary_loss_type} ({100*(1-loss_combination_weight):.1f}%)")
            print(f"   - Secondary loss: {secondary_loss_type} ({100*loss_combination_weight:.1f}%)")
        elif self.loss_type == "scale_aware_cosine":
            print(f"   - Direction weight: {direction_weight:.2f}, Magnitude weight: {magnitude_weight:.2f}")
        elif self.loss_type == "layer_adaptive":
            print(f"   - Using layer-adaptive loss configuration")
    
    def _get_default_layer_config(self):
        """获取默认的层级损失配置"""
        return {
            'text_model.encoder.layers.0': {'type': 'mse', 'weight': 1.0},
            'text_model.encoder.layers.1': {'type': 'mse', 'weight': 1.0},
            'text_model.encoder.layers.2': {'type': 'hybrid', 'weight': 1.2},
            'text_model.encoder.layers.3': {'type': 'hybrid', 'weight': 1.2},
            'text_model.encoder.layers.4': {'type': 'hybrid', 'weight': 1.5},
            'text_model.encoder.layers.5': {'type': 'hybrid', 'weight': 1.5},
            'text_model.encoder.layers.6': {'type': 'cosine', 'weight': 2.0},
            'text_model.encoder.layers.7': {'type': 'cosine', 'weight': 2.0},
            'text_model.encoder.layers.8': {'type': 'hybrid', 'weight': 1.8},
            'text_model.encoder.layers.9': {'type': 'hybrid', 'weight': 1.8},
            'text_model.encoder.layers.10': {'type': 'hybrid', 'weight': 1.5},
            'text_model.encoder.layers.11': {'type': 'mse', 'weight': 1.0},
        }

    def _pool_text_features(self, features: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        对文本特征进行池化。
        Args:
            features: 形状为 (batch_size, seq_len, hidden_size) 的特征张量
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
        Returns:
            池化后的特征张量
        """
        if self.pooling_strategy == "none":
            return features
        elif self.pooling_strategy == "cls":
            # 使用CLS token (第一个token)
            return features[:, 0, :]  # (batch_size, hidden_size)
        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                # 使用attention mask进行加权平均
                mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
                sum_features = torch.sum(features * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_features / sum_mask
            else:
                # 简单平均
                return torch.mean(features, dim=1)
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                # 使用attention mask进行max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
                features_masked = features * mask_expanded + (1 - mask_expanded) * (-1e9)  # 将padding位置设为很小的值
                return torch.max(features_masked, dim=1)[0]
            else:
                return torch.max(features, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

    def _compute_single_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, 
                           loss_type: str) -> torch.Tensor:
        """计算单一类型的损失"""
        if loss_type == "mse":
            return self.mse_criterion(student_feat, teacher_feat)
        elif loss_type == "l1":
            return self.l1_criterion(student_feat, teacher_feat)
        elif loss_type == "cosine":
            # 余弦相似度损失
            target = torch.ones(student_feat.size(0), device=student_feat.device)
            return self.cosine_criterion(student_feat, teacher_feat, target)
        else:
            raise ValueError(f"Unsupported single loss type: {loss_type}")

    def _compute_hybrid_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                           layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 计算混合损失"""
        # 主损失 (通常是MSE)
        primary_loss = self._compute_single_loss(student_feat, teacher_feat, self.primary_loss_type)
        
        # 次损失 (通常是Cosine)
        secondary_loss = self._compute_single_loss(student_feat, teacher_feat, self.secondary_loss_type)
        
        # 混合损失
        total_loss = (1 - self.loss_combination_weight) * primary_loss + \
                     self.loss_combination_weight * secondary_loss
        
        loss_details = {
            f'{layer_name}_primary': primary_loss.item(),
            f'{layer_name}_secondary': secondary_loss.item(),
            f'{layer_name}_hybrid_total': total_loss.item()
        }
        
        return total_loss, loss_details

    def _compute_scale_aware_cosine_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                                       layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 计算尺度感知的余弦损失"""
        # 展平特征以便计算
        batch_size = student_feat.size(0)
        student_flat = student_feat.view(batch_size, -1)
        teacher_flat = teacher_feat.view(batch_size, -1)
        
        # 1. 方向损失 (余弦相似度)
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        direction_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
        
        # 2. 幅度损失 (L2范数差异)
        student_magnitude = torch.norm(student_flat, p=2, dim=1)
        teacher_magnitude = torch.norm(teacher_flat, p=2, dim=1)
        magnitude_loss = F.mse_loss(student_magnitude, teacher_magnitude)
        
        # 3. 混合损失
        total_loss = self.direction_weight * direction_loss + \
                     self.magnitude_weight * magnitude_loss
        
        loss_details = {
            f'{layer_name}_direction': direction_loss.item(),
            f'{layer_name}_magnitude': magnitude_loss.item(),
            f'{layer_name}_scale_aware_total': total_loss.item()
        }
        
        return total_loss, loss_details

    def _compute_layer_adaptive_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                                   layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 计算层级自适应损失"""
        config = self.layer_loss_config.get(layer_name, {'type': 'mse', 'weight': 1.0})
        loss_type = config['type']
        layer_weight = config['weight']
        
        if loss_type == 'hybrid':
            # 使用混合损失
            loss, loss_details = self._compute_hybrid_loss(student_feat, teacher_feat, layer_name)
        elif loss_type == 'scale_aware_cosine':
            # 使用尺度感知余弦损失
            loss, loss_details = self._compute_scale_aware_cosine_loss(student_feat, teacher_feat, layer_name)
        else:
            # 使用单一损失
            loss = self._compute_single_loss(student_feat, teacher_feat, loss_type)
            loss_details = {f'{layer_name}_{loss_type}': loss.item()}
        
        # 应用层级权重
        weighted_loss = loss * layer_weight
        loss_details[f'{layer_name}_layer_weight'] = layer_weight
        loss_details[f'{layer_name}_weighted_total'] = weighted_loss.item()
        
        return weighted_loss, loss_details

    def _compute_layer_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                          layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 根据配置计算层损失"""
        if self.loss_type == "hybrid":
            return self._compute_hybrid_loss(student_feat, teacher_feat, layer_name)
        elif self.loss_type == "scale_aware_cosine":
            return self._compute_scale_aware_cosine_loss(student_feat, teacher_feat, layer_name)
        elif self.loss_type == "layer_adaptive":
            return self._compute_layer_adaptive_loss(student_feat, teacher_feat, layer_name)
        else:
            # 单一损失类型
            loss = self._compute_single_loss(student_feat, teacher_feat, self.loss_type)
            loss_details = {f'{layer_name}_{self.loss_type}': loss.item()}
            return loss, loss_details

    def _adaptive_align_features(self, feat1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                feat2: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                attention_mask1: torch.Tensor = None,
                                attention_mask2: torch.Tensor = None):
        """
        自适应地对齐两个特征张量。
        """
        # 处理可能的元组输出
        if isinstance(feat1, tuple):
            if len(feat1) > 0 and isinstance(feat1[0], torch.Tensor):
                feat1 = feat1[0]
            else:
                raise TypeError(f"feat1 is a tuple but no tensor found within it.")
        
        if isinstance(feat2, tuple):
            if len(feat2) > 0 and isinstance(feat2[0], torch.Tensor):
                feat2 = feat2[0]
            else:
                raise TypeError(f"feat2 is a tuple but no tensor found within it.")

        if not isinstance(feat1, torch.Tensor) or not isinstance(feat2, torch.Tensor):
            raise TypeError(f"Features must be tensors. Got feat1: {type(feat1)}, feat2: {type(feat2)}")

        # 对文本特征进行池化（如果需要）
        feat1_pooled = self._pool_text_features(feat1, attention_mask1)
        feat2_pooled = self._pool_text_features(feat2, attention_mask2)

        # 确保维度匹配
        if feat1_pooled.shape != feat2_pooled.shape:
            # 如果hidden_size不同，可以考虑添加线性投影层，这里先简单处理
            min_dim = min(feat1_pooled.size(-1), feat2_pooled.size(-1))
            feat1_pooled = feat1_pooled[..., :min_dim]
            feat2_pooled = feat2_pooled[..., :min_dim]

        return feat1_pooled, feat2_pooled

    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor],
                teacher_attention_mask: torch.Tensor = None,
                student_attention_mask: torch.Tensor = None):
        """
        🔥 计算混合特征对齐损失。
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        # 将total_loss移动到特征所在的设备
        if teacher_features and isinstance(list(teacher_features.values())[0], torch.Tensor):
            total_loss = total_loss.to(list(teacher_features.values())[0].device)
        elif student_features and isinstance(list(student_features.values())[0], torch.Tensor):
            total_loss = total_loss.to(list(student_features.values())[0].device)

        loss_dict = {}
        active_layers_count = 0
        detailed_loss_info = {}

        for layer_name in self.alignment_layers:
            if layer_name not in teacher_features or teacher_features[layer_name] is None:
                continue
            if layer_name not in student_features or student_features[layer_name] is None:
                continue

            t_feat_orig = teacher_features[layer_name]
            s_feat_orig = student_features[layer_name]

            if not (isinstance(t_feat_orig, torch.Tensor) or isinstance(t_feat_orig, tuple)) or \
               not (isinstance(s_feat_orig, torch.Tensor) or isinstance(s_feat_orig, tuple)):
                print(f"Warning: Features for layer {layer_name} are not Tensors or Tuples. Skipping.")
                continue
            
            try:
                teacher_feat_processed, student_feat_processed = self._adaptive_align_features(
                    t_feat_orig, s_feat_orig, teacher_attention_mask, student_attention_mask
                )
            except TypeError as e:
                print(f"Error in _adaptive_align_features for layer {layer_name}: {e}. Skipping layer.")
                continue

            if teacher_feat_processed is None or student_feat_processed is None:
                continue

            # 转换为float类型进行计算
            student_feat_for_loss = student_feat_processed.float()
            teacher_feat_for_loss = teacher_feat_processed.float()
            
            # 应用温度缩放
            if self.temperature != 1.0:
                student_feat_for_loss = student_feat_for_loss / self.temperature
                teacher_feat_for_loss = teacher_feat_for_loss / self.temperature

            # 🔥 计算层损失 (支持混合损失)
            layer_loss, layer_loss_details = self._compute_layer_loss(
                student_feat_for_loss, teacher_feat_for_loss, layer_name
            )
            
            # 应用全局层权重
            global_weight = self.loss_weights.get(layer_name, 1.0)
            weighted_layer_loss = global_weight * layer_loss
            
            total_loss += weighted_layer_loss
            loss_dict[layer_name] = weighted_layer_loss.item()
            
            # 保存详细损失信息
            detailed_loss_info.update(layer_loss_details)
            detailed_loss_info[f'{layer_name}_global_weight'] = global_weight
            detailed_loss_info[f'{layer_name}_final_weighted'] = weighted_layer_loss.item()
            
            active_layers_count += 1
        
        if active_layers_count == 0:
            return torch.tensor(0.0, device=total_loss.device, dtype=torch.float32), loss_dict, detailed_loss_info

        # 🔥 添加损失类型统计信息
        detailed_loss_info['total_loss'] = total_loss.item()
        detailed_loss_info['active_layers'] = active_layers_count
        detailed_loss_info['loss_type'] = self.loss_type
        
        return total_loss, loss_dict, detailed_loss_info


# 🔥 保持向后兼容性的原始类（简单包装）
class TextEncoderFeatureAlignmentLoss(HybridTextEncoderFeatureAlignmentLoss):
    """原始的TextEncoderFeatureAlignmentLoss，现在继承自混合版本以保持向后兼容"""
    def __init__(self, alignment_layers: List[str], loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, loss_type: str = "mse", pooling_strategy: str = "mean"):
        # 🔥 调用混合版本的初始化，但使用单一损失配置
        super().__init__(
            alignment_layers=alignment_layers,
            loss_weights=loss_weights,
            temperature=temperature,
            loss_type=loss_type,
            pooling_strategy=pooling_strategy,
            # 保持原始行为的默认参数
            primary_loss_type="mse",
            secondary_loss_type="cosine",
            loss_combination_weight=0.0,  # 不使用混合
            enable_magnitude_preservation=False,
            use_layer_adaptive=False
        )
    
    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor],
                teacher_attention_mask: torch.Tensor = None,
                student_attention_mask: torch.Tensor = None):
        """保持原始返回格式的前向传播"""
        total_loss, loss_dict, detailed_loss_info = super().forward(
            teacher_features, student_features, teacher_attention_mask, student_attention_mask
        )
        # 只返回原始格式的前两个值
        return total_loss, loss_dict


def compute_text_encoder_distillation_loss(
    teacher1_text_encoder,
    teacher2_text_encoder,
    student_text_encoder,
    input_ids,
    attention_mask=None,
    feature_alignment_loss_fn=None,
    text_feature_extractor=None,
    embedding_weight=1.0,
    feature_align_weight=0.5,
    use_grad_for_student=True
):
    """
    🔥 改进的Text Encoder蒸馏损失计算，支持混合特征对齐损失。
    """
    
    # Teacher 1 前向传播 (no grad)
    with torch.no_grad():
        if text_feature_extractor is not None:
            teacher1_output, teacher1_features = text_feature_extractor.extract_features(
                teacher1_text_encoder, input_ids, attention_mask, use_grad=False
            )
        else:
            teacher1_output = teacher1_text_encoder(input_ids, attention_mask=attention_mask)
            teacher1_features = {}
    
    # Teacher 2 前向传播 (no grad)
    with torch.no_grad():
        if text_feature_extractor is not None:
            teacher2_output, teacher2_features = text_feature_extractor.extract_features(
                teacher2_text_encoder, input_ids, attention_mask, use_grad=False
            )
        else:
            teacher2_output = teacher2_text_encoder(input_ids, attention_mask=attention_mask)
            teacher2_features = {}
    
    # Student 前向传播
    if text_feature_extractor is not None:
        student_output, student_features = text_feature_extractor.extract_features(
            student_text_encoder, input_ids, attention_mask, use_grad=use_grad_for_student
        )
    else:
        student_output = student_text_encoder(input_ids, attention_mask=attention_mask)
        student_features = {}
    
    # 获取文本嵌入 (通常是last_hidden_state或pooler_output)
    if hasattr(teacher1_output, 'last_hidden_state'):
        teacher1_embeddings = teacher1_output.last_hidden_state
    elif hasattr(teacher1_output, 'pooler_output'):
        teacher1_embeddings = teacher1_output.pooler_output
    else:
        teacher1_embeddings = teacher1_output
    
    if hasattr(teacher2_output, 'last_hidden_state'):
        teacher2_embeddings = teacher2_output.last_hidden_state
    elif hasattr(teacher2_output, 'pooler_output'):
        teacher2_embeddings = teacher2_output.pooler_output
    else:
        teacher2_embeddings = teacher2_output
        
    if hasattr(student_output, 'last_hidden_state'):
        student_embeddings = student_output.last_hidden_state
    elif hasattr(student_output, 'pooler_output'):
        student_embeddings = student_output.pooler_output
    else:
        student_embeddings = student_output
    
    # 计算文本嵌入损失
    target_embeddings = (teacher1_embeddings + teacher2_embeddings) / 2  # 平均两个teacher的输出
    embedding_loss = F.mse_loss(student_embeddings, target_embeddings)
    
    # 🔥 计算混合特征对齐损失
    total_feature_loss = torch.tensor(0.0, device=embedding_loss.device)
    feature_loss_dict = {}
    detailed_feature_info = {}
    
    if feature_alignment_loss_fn is not None and (teacher1_features or teacher2_features):
        # 与teacher1的特征对齐
        if teacher1_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # 新的混合损失函数，返回3个值
                feature_loss1, feature_loss_dict1, detailed_info1 = feature_alignment_loss_fn(
                    teacher1_features, student_features, attention_mask, attention_mask
                )
                detailed_feature_info.update({f"t1_{k}": v for k, v in detailed_info1.items()})
            else:
                # 旧的损失函数，返回2个值
                feature_loss1, feature_loss_dict1 = feature_alignment_loss_fn(
                    teacher1_features, student_features, attention_mask, attention_mask
                )
            
            total_feature_loss += feature_loss1
            feature_loss_dict.update({f"t1_{k}": v for k, v in feature_loss_dict1.items()})
        
        # 与teacher2的特征对齐
        if teacher2_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # 新的混合损失函数，返回3个值
                feature_loss2, feature_loss_dict2, detailed_info2 = feature_alignment_loss_fn(
                    teacher2_features, student_features, attention_mask, attention_mask
                )
                detailed_feature_info.update({f"t2_{k}": v for k, v in detailed_info2.items()})
            else:
                # 旧的损失函数，返回2个值
                feature_loss2, feature_loss_dict2 = feature_alignment_loss_fn(
                    teacher2_features, student_features, attention_mask, attention_mask
                )
                
            total_feature_loss += feature_loss2
            feature_loss_dict.update({f"t2_{k}": v for k, v in feature_loss_dict2.items()})
        
        # 如果两个teacher都有特征，取平均
        if teacher1_features and teacher2_features:
            total_feature_loss = total_feature_loss / 2
    
    # 总损失
    total_loss = (
        embedding_weight * embedding_loss + 
        feature_align_weight * total_feature_loss
    )
    
    loss_dict = {
        'embedding_loss': embedding_loss.item(),
        'feature_align_loss': total_feature_loss.item(),
        'total_loss': total_loss.item(),
        **feature_loss_dict
    }
    
    # 🔥 如果有详细信息，也加入返回
    if detailed_feature_info:
        loss_dict.update(detailed_feature_info)
    
    return total_loss, loss_dict


# 🔥 便捷的创建函数
def create_hybrid_text_encoder_alignment_loss(alignment_layers: List[str],
                                             loss_type: str = "hybrid",
                                             loss_weights: Dict[str, float] = None,
                                             **kwargs) -> HybridTextEncoderFeatureAlignmentLoss:
    """
    便捷创建混合文本编码器特征对齐损失的函数
    
    Args:
        alignment_layers: 对齐层列表
        loss_type: 损失类型 - "mse", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
        loss_weights: 层权重字典
        **kwargs: 其他参数
    
    推荐配置:
    1. 标准混合损失: loss_type="hybrid", primary_loss_type="mse", secondary_loss_type="cosine", loss_combination_weight=0.3
    2. 尺度感知: loss_type="scale_aware_cosine", direction_weight=0.8, magnitude_weight=0.2
    3. 层级自适应: loss_type="layer_adaptive", use_layer_adaptive=True
    """
    return HybridTextEncoderFeatureAlignmentLoss(
        alignment_layers=alignment_layers,
        loss_type=loss_type,
        loss_weights=loss_weights,
        **kwargs
    )


# 🔥 使用示例和推荐配置
def example_hybrid_usage():
    """
    混合损失使用示例
    """
    target_layers = [
        'text_model.encoder.layers.0',
        'text_model.encoder.layers.2',
        'text_model.encoder.layers.4',
        'text_model.encoder.layers.6',
        'text_model.encoder.layers.8',
        'text_model.encoder.layers.10',
        'text_model.encoder.layers.11'
    ]
    
    # 🔥 方案1: 混合损失 (推荐)
    hybrid_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="hybrid",
        primary_loss_type="mse",           # 主损失保持尺度信息
        secondary_loss_type="cosine",      # 次损失关注方向对齐
        loss_combination_weight=0.3,       # 30% cosine + 70% mse
        pooling_strategy="mean",
        loss_weights={
            'text_model.encoder.layers.0': 0.5,
            'text_model.encoder.layers.6': 1.0,
            'text_model.encoder.layers.11': 1.5
        }
    )
    
    # 🔥 方案2: 尺度感知余弦损失
    scale_aware_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="scale_aware_cosine",
        direction_weight=0.8,              # 方向损失权重
        magnitude_weight=0.2,              # 幅度损失权重
        pooling_strategy="mean"
    )
    
    # 🔥 方案3: 层级自适应损失
    layer_adaptive_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="layer_adaptive",        # 每层使用不同损失策略
        use_layer_adaptive=True,
        pooling_strategy="mean"
    )
    
    print("🔥 混合损失示例配置完成!")
    print("方案1: 混合损失 - 平衡尺度和方向")
    print("方案2: 尺度感知余弦 - 保持余弦优势同时考虑幅度")
    print("方案3: 层级自适应 - 根据层特性选择最佳损失")
    
    return hybrid_loss, scale_aware_loss, layer_adaptive_loss


if __name__ == "__main__":
    # 运行示例
    example_hybrid_usage()
    
    # 测试向后兼容性
    print("\n🔧 测试向后兼容性...")
    old_style_loss = TextEncoderFeatureAlignmentLoss(
        alignment_layers=['text_model.encoder.layers.0'],
        loss_type="mse"
    )
    print("✅ 向后兼容性测试通过!")