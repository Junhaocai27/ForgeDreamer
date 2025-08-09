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


class TextEncoderFeatureAlignmentLoss(nn.Module):
    def __init__(self, alignment_layers: List[str], loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, loss_type: str = "mse", pooling_strategy: str = "mean"):
        """
        初始化Text Encoder特征对齐损失模块。
        Args:
            alignment_layers (List[str]): 需要对齐特征的层名称列表。
            loss_weights (Dict[str, float], optional): 每层损失的权重。默认为None (所有权重为1.0)。
            temperature (float, optional): 用于缩放特征的温度。默认为1.0。
            loss_type (str, optional): 使用的损失类型，可以是 "mse", "l1", "cosine"。默认为 "mse"。
            pooling_strategy (str, optional): 特征池化策略，可以是 "mean", "max", "cls", "none"。默认为 "mean"。
        """
        super().__init__()
        self.alignment_layers = alignment_layers
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.temperature = temperature
        self.pooling_strategy = pooling_strategy

        # 初始化损失函数
        if loss_type.lower() == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss_type.lower() == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        elif loss_type.lower() == "cosine":
            self.criterion = nn.CosineEmbeddingLoss(reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'mse', 'l1', or 'cosine'.")
        
        self.loss_type = loss_type.lower()
        print(f"TextEncoderFeatureAlignmentLoss initialized with loss_type: {loss_type}, pooling_strategy: {pooling_strategy}")

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
        计算特征对齐损失。
        Args:
            teacher_features: 教师模型的特征字典
            student_features: 学生模型的特征字典  
            teacher_attention_mask: 教师模型的attention mask
            student_attention_mask: 学生模型的attention mask
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        # 将total_loss移动到特征所在的设备
        if teacher_features and isinstance(list(teacher_features.values())[0], torch.Tensor):
            total_loss = total_loss.to(list(teacher_features.values())[0].device)
        elif student_features and isinstance(list(student_features.values())[0], torch.Tensor):
            total_loss = total_loss.to(list(student_features.values())[0].device)

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
            s_feat_for_loss = student_feat_processed.float()
            t_feat_for_loss = teacher_feat_processed.float()
            
            # 应用温度缩放
            if self.temperature != 1.0:
                s_feat_for_loss = s_feat_for_loss / self.temperature
                t_feat_for_loss = t_feat_for_loss / self.temperature

            # 计算损失
            if self.loss_type == "cosine":
                # 余弦相似度损失需要target标签
                target = torch.ones(s_feat_for_loss.size(0), device=s_feat_for_loss.device)
                layer_loss = self.criterion(s_feat_for_loss, t_feat_for_loss, target)
            else:
                layer_loss = self.criterion(s_feat_for_loss, t_feat_for_loss)
            
            weight = self.loss_weights.get(layer_name, 1.0)
            weighted_loss = weight * layer_loss
            
            total_loss += weighted_loss
            loss_dict[layer_name] = weighted_loss.item()
            active_layers_count += 1
        
        if active_layers_count == 0:
            return torch.tensor(0.0, device=total_loss.device, dtype=torch.float32), loss_dict

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
    计算包含特征对齐的Text Encoder蒸馏损失。
    
    Args:
        teacher1_text_encoder: 第一个教师Text Encoder
        teacher2_text_encoder: 第二个教师Text Encoder
        student_text_encoder: 学生Text Encoder
        input_ids: 输入的token ids
        attention_mask: 注意力掩码
        feature_alignment_loss_fn: 特征对齐损失函数
        text_feature_extractor: Text Encoder特征提取器
        embedding_weight: 文本嵌入损失权重
        feature_align_weight: 特征对齐损失权重
        use_grad_for_student: 学生模型是否需要梯度
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
    
    # 计算特征对齐损失
    total_feature_loss = torch.tensor(0.0, device=embedding_loss.device)
    feature_loss_dict = {}
    
    if feature_alignment_loss_fn is not None and (teacher1_features or teacher2_features):
        # 与teacher1的特征对齐
        if teacher1_features:
            feature_loss1, feature_loss_dict1 = feature_alignment_loss_fn(
                teacher1_features, student_features, attention_mask, attention_mask
            )
            total_feature_loss += feature_loss1
            feature_loss_dict.update({f"t1_{k}": v for k, v in feature_loss_dict1.items()})
        
        # 与teacher2的特征对齐
        if teacher2_features:
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
    
    return total_loss, loss_dict


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 假设我们有CLIP Text Encoder
    # teacher1_text_encoder = ...
    # teacher2_text_encoder = ...  
    # student_text_encoder = ...
    
    # 定义要提取特征的层
    target_layers = [
        'text_model.encoder.layers.0',
        'text_model.encoder.layers.6',
        'text_model.encoder.layers.11'
    ]
    
    # 创建特征提取器
    text_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=target_layers,
        mixed_precision_config="fp16"
    )
    
    # 创建特征对齐损失函数
    feature_alignment_loss = TextEncoderFeatureAlignmentLoss(
        alignment_layers=target_layers,
        loss_weights={
            'text_model.encoder.layers.0': 0.5,
            'text_model.encoder.layers.6': 1.0,
            'text_model.encoder.layers.11': 1.5
        },
        loss_type="mse",
        pooling_strategy="mean"
    )
    
    # 准备输入数据
    batch_size = 4
    seq_len = 77
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 计算蒸馏损失
    # total_loss, loss_dict = compute_text_encoder_distillation_loss(
    #     teacher1_text_encoder=teacher1_text_encoder,
    #     teacher2_text_encoder=teacher2_text_encoder,
    #     student_text_encoder=student_text_encoder,
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     feature_alignment_loss_fn=feature_alignment_loss,
    #     text_feature_extractor=text_feature_extractor,
    #     embedding_weight=1.0,
    #     feature_align_weight=0.5
    # )
    
    print("Example setup completed!")

if __name__ == "__main__":
    example_usage()