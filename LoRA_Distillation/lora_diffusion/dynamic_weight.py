import torch
import torch.nn.functional as F
from collections import deque
import torch.nn as nn

# 通用的多损失动态权重调整器
class MultiLossDynamicWeightAdjuster:
    def __init__(self, 
                 loss_config,
                 history_length=10,
                 temperature=1.0):
        """
        通用多损失动态权重调整器
        
        Args:
            loss_config: 损失配置字典，格式如下:
                {
                    'loss_name': {
                        'initial_weight': float,
                        'min_weight': float,
                        'max_weight': float,
                        'priority': float  # 权重分配优先级
                    }
                }
            history_length: 保留的历史损失长度
            temperature: softmax温度参数，控制调整的敏感度
        """
        self.loss_config = loss_config
        self.history_length = history_length
        self.temperature = temperature
        
        # 存储历史损失 - 每个损失类型一个队列
        self.loss_histories = {}
        self.current_weights = {}
        
        for loss_name, config in loss_config.items():
            self.loss_histories[loss_name] = deque(maxlen=history_length)
            self.current_weights[loss_name] = config['initial_weight']
    
    def update_weights(self, current_losses):
        """
        根据当前损失更新权重
        
        Args:
            current_losses: 当前损失字典 {'loss_name': loss_value}
            
        Returns:
            dict: 更新后的权重字典 {'loss_name': weight_value}
        """
        # 添加当前损失到历史记录
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.loss_histories:
                self.loss_histories[loss_name].append(loss_value)
        
        # 如果历史记录不足，使用当前权重
        min_history_required = 2
        if any(len(history) < min_history_required for history in self.loss_histories.values()):
            return dict(self.current_weights)
        
        # 计算每个损失的变化趋势
        loss_trends = {}
        for loss_name in self.loss_config.keys():
            if loss_name in self.loss_histories:
                loss_trends[loss_name] = self._compute_loss_trend(self.loss_histories[loss_name])
        
        # 使用softmax计算新权重
        new_weights = self._compute_softmax_weights(loss_trends)
        
        # 更新当前权重
        self.current_weights.update(new_weights)
        
        return dict(new_weights)
    
    def _compute_loss_trend(self, loss_history):
        """计算损失变化趋势"""
        if len(loss_history) < 2:
            return 0.0
        
        # 计算最近几步的平均变化率
        recent_losses = list(loss_history)[-min(5, len(loss_history)):]
        
        if len(recent_losses) < 2:
            return 0.0
        
        # 计算斜率（变化趋势）
        changes = []
        for i in range(1, len(recent_losses)):
            change = (recent_losses[i] - recent_losses[i-1]) / (recent_losses[i-1] + 1e-8)
            changes.append(change)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def _compute_softmax_weights(self, loss_trends):
        """使用softmax计算新权重"""
        loss_names = list(self.loss_config.keys())
        
        # 构建权重调整的logits
        logits = []
        for loss_name in loss_names:
            trend = loss_trends.get(loss_name, 0.0)
            priority = self.loss_config[loss_name].get('priority', 1.0)
            
            # 反向调整：损失上升时增加对应权重，同时考虑优先级
            adjustment = (-trend + priority) * self.temperature
            logits.append(adjustment)
        
        # 应用softmax获得分配比例
        logits_tensor = torch.tensor(logits)
        weights_ratio = F.softmax(logits_tensor, dim=0)
        
        # 计算总权重预算
        total_budget = sum(config['initial_weight'] for config in self.loss_config.values())
        
        # 分配新权重
        new_weights = {}
        for i, loss_name in enumerate(loss_names):
            config = self.loss_config[loss_name]
            
            # 基于softmax比例和优先级分配权重
            base_weight = weights_ratio[i] * total_budget * config.get('priority', 1.0)
            
            # 应用最小最大限制
            new_weight = torch.clamp(
                base_weight, 
                config['min_weight'], 
                config['max_weight']
            ).item()
            
            new_weights[loss_name] = new_weight
        
        return new_weights

# ============ 使用示例 ============

# 1. 为 perform_tuning 阶段创建权重调整器
def create_tuning_weight_adjuster(noise_pred_weight=1.0, feature_align_weight=0.001):
    tuning_config = {
        'noise_pred': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 3.0,
            'priority': 2.0  # 噪声预测优先级较高
        },
        'feature_align': {
            'initial_weight': feature_align_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.0  # 特征对齐优先级相对较低
        }
    }
    return MultiLossDynamicWeightAdjuster(tuning_config, temperature=2.0)

# 2. 为 train_inversion 阶段创建权重调整器
def create_inversion_weight_adjuster(noise_pred_weight=1.0, 
                                   unet_feature_weight=0.001, 
                                   text_encoder_feature_weight=0.001):
    inversion_config = {
        'ti_loss': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 3.0,
            'priority': 2.5  # TI损失优先级最高
        },
        'unet_feature_align': {
            'initial_weight': unet_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.5  # UNet特征对齐中等优先级
        },
        'text_encoder_feature_align': {
            'initial_weight': text_encoder_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.0  # 文本编码器特征对齐优先级较低
        }
    }
    return MultiLossDynamicWeightAdjuster(inversion_config, temperature=1.5)

# ============ 在训练代码中的使用方式 ============

# --- 在 perform_tuning 阶段 ---
"""
# 初始化权重调整器
tuning_weight_adjuster = create_tuning_weight_adjuster(
    noise_pred_weight=noise_pred_weight,
    feature_align_weight=feature_align_weight
)

# 在训练循环中使用
current_losses = {
    'noise_pred': current_noise_loss_item,
    'feature_align': current_feature_loss_item
}

# 更新权重
updated_weights = tuning_weight_adjuster.update_weights(current_losses)
noise_pred_weight = updated_weights['noise_pred']
feature_align_weight = updated_weights['feature_align']

# 使用动态权重计算总损失
total_loss = (
    noise_pred_weight * noise_pred_loss +
    feature_align_weight * feature_align_loss
)
"""

# --- 在 train_inversion 阶段 ---
"""
# 初始化权重调整器
inversion_weight_adjuster = create_inversion_weight_adjuster(
    noise_pred_weight=noise_pred_weight,
    unet_feature_weight=unet_feature_align_weight,
    text_encoder_feature_weight=text_encoder_feature_align_weight
)

# 在训练循环中使用
current_losses = {
    'ti_loss': ti_loss_current_step.item(),
    'unet_feature_align': unet_feature_align_loss_val.item() if isinstance(unet_feature_align_loss_val, torch.Tensor) else unet_feature_align_loss_val,
    'text_encoder_feature_align': text_encoder_feature_align_loss_val.item() if isinstance(text_encoder_feature_align_loss_val, torch.Tensor) else text_encoder_feature_align_loss_val
}

# 更新权重
updated_weights = inversion_weight_adjuster.update_weights(current_losses)
noise_pred_weight = updated_weights['ti_loss']
unet_feature_align_weight = updated_weights['unet_feature_align']
text_encoder_feature_align_weight = updated_weights['text_encoder_feature_align']

# 使用动态权重计算总损失
current_step_total_loss = (
    noise_pred_weight * ti_loss_current_step +
    unet_feature_align_weight * unet_feature_align_loss_val +
    text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
)

# 记录权重变化到TensorBoard
writer.add_scalar('Weights/TI_Loss_Weight', noise_pred_weight, global_step)
writer.add_scalar('Weights/UNet_Feature_Weight', unet_feature_align_weight, global_step)
writer.add_scalar('Weights/TextEncoder_Feature_Weight', text_encoder_feature_align_weight, global_step)
"""

import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
from typing import Dict, List, Optional

class EnhancedMultiTeacherWeightAdjuster:
    def __init__(self, 
                 teacher_info: List[dict],
                 loss_config: dict,
                 history_length=20,
                 temperature=1.0,
                 teacher_balance_weight=0.3,  # 新增：teacher平衡权重
                 convergence_threshold=0.01,  # 新增：收敛阈值
                 performance_window=50):      # 新增：性能窗口
        """
        增强的多Teacher动态权重调整器
        
        Args:
            teacher_info: Teacher信息列表
            loss_config: 损失配置
            teacher_balance_weight: Teacher平衡的重要性权重
            convergence_threshold: 判断损失收敛的阈值
            performance_window: 性能评估窗口大小
        """
        self.teacher_info = teacher_info
        self.num_teachers = len(teacher_info)
        self.loss_config = loss_config
        self.history_length = history_length
        self.temperature = temperature
        self.teacher_balance_weight = teacher_balance_weight
        self.convergence_threshold = convergence_threshold
        self.performance_window = performance_window
        
        # 原有的损失历史记录
        self.loss_histories = {}
        self.current_weights = {}
        
        # 新增：每个teacher的性能跟踪
        self.teacher_performance = {i: {
            'loss_history': deque(maxlen=performance_window),
            'convergence_rate': 0.0,
            'relative_performance': 1.0,  # 相对于最佳teacher的性能比率
            'training_difficulty': 1.0,   # 训练难度估计
            'sample_count': 0,            # 训练样本计数
            'last_improvement': 0         # 上次改善的步数
        } for i in range(self.num_teachers)}
        
        # 新增：全局统计信息
        self.global_step = 0
        self.teacher_selection_counts = [0] * self.num_teachers
        self.performance_gap_threshold = 0.2  # 性能差距阈值
        
        for loss_name, config in loss_config.items():
            self.loss_histories[loss_name] = deque(maxlen=history_length)
            self.current_weights[loss_name] = config['initial_weight']
    
    def update_teacher_performance(self, teacher_idx: int, loss_value: float, global_step: int):
        """更新指定teacher的性能记录"""
        self.global_step = global_step
        self.teacher_selection_counts[teacher_idx] += 1
        
        perf = self.teacher_performance[teacher_idx]
        perf['loss_history'].append(loss_value)
        perf['sample_count'] += 1
        
        # 计算收敛率
        if len(perf['loss_history']) >= 5:
            recent_losses = list(perf['loss_history'])[-5:]
            perf['convergence_rate'] = self._compute_convergence_rate(recent_losses)
            
            # 检查是否有改善
            if len(perf['loss_history']) >= 2:
                if perf['loss_history'][-1] < perf['loss_history'][-2]:
                    perf['last_improvement'] = global_step
        
        # 更新相对性能和训练难度
        self._update_relative_performance()
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """计算损失的收敛速度"""
        if len(losses) < 3:
            return 0.0
        
        # 计算损失的变化率
        changes = []
        for i in range(1, len(losses)):
            if losses[i-1] != 0:
                change_rate = abs(losses[i] - losses[i-1]) / losses[i-1]
                changes.append(change_rate)
        
        if not changes:
            return 0.0
        
        # 收敛率 = 1 - 平均变化率（值越大表示越稳定）
        avg_change_rate = sum(changes) / len(changes)
        return max(0.0, 1.0 - avg_change_rate)
    
    def _update_relative_performance(self):
        """更新每个teacher的相对性能"""
        # 计算每个teacher的平均损失
        teacher_avg_losses = []
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            if perf['loss_history']:
                avg_loss = sum(perf['loss_history']) / len(perf['loss_history'])
                teacher_avg_losses.append(avg_loss)
            else:
                teacher_avg_losses.append(float('inf'))
        
        # 找到最佳性能
        best_loss = min(loss for loss in teacher_avg_losses if loss != float('inf'))
        if best_loss == float('inf'):
            return
        
        # 更新相对性能和训练难度
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            if teacher_avg_losses[i] != float('inf'):
                # 相对性能：最佳性能 / 当前性能
                perf['relative_performance'] = best_loss / teacher_avg_losses[i]
                
                # 训练难度：基于损失水平和收敛率
                difficulty_factor = teacher_avg_losses[i] / best_loss
                convergence_factor = 1.0 / (perf['convergence_rate'] + 0.1)
                perf['training_difficulty'] = difficulty_factor * convergence_factor
    
    def get_teacher_adaptive_weights(self) -> Dict[int, float]:
        """计算每个teacher的自适应权重"""
        if self.global_step < 10:  # 初期使用均匀权重
            return {i: 1.0 for i in range(self.num_teachers)}
        
        teacher_weights = {}
        total_weight = 0.0
        
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            
            # 基础权重因子
            base_weight = 1.0
            
            # 1. 性能因子：性能越差，权重越高（需要更多关注）
            performance_factor = 2.0 - perf['relative_performance']
            performance_factor = max(0.5, min(2.0, performance_factor))
            
            # 2. 训练难度因子：难度越高，权重越高
            difficulty_factor = perf['training_difficulty']
            difficulty_factor = max(0.8, min(1.5, difficulty_factor))
            
            # 3. 收敛因子：收敛慢的需要更多训练
            convergence_factor = 1.0 + (1.0 - perf['convergence_rate'])
            convergence_factor = max(1.0, min(1.8, convergence_factor))
            
            # 4. 样本平衡因子：样本数少的teacher需要更多训练
            min_samples = min(p['sample_count'] for p in self.teacher_performance.values())
            max_samples = max(p['sample_count'] for p in self.teacher_performance.values())
            if max_samples > min_samples:
                sample_ratio = perf['sample_count'] / max_samples
                balance_factor = 1.0 + (1.0 - sample_ratio) * 0.5
            else:
                balance_factor = 1.0
            
            # 5. 长期无改善惩罚
            steps_since_improvement = self.global_step - perf['last_improvement']
            if steps_since_improvement > 100:
                stagnation_factor = 1.0 + min(0.5, steps_since_improvement / 1000)
            else:
                stagnation_factor = 1.0
            
            # 综合权重计算
            teacher_weight = (base_weight * 
                            performance_factor * 
                            difficulty_factor * 
                            convergence_factor * 
                            balance_factor * 
                            stagnation_factor)
            
            teacher_weights[i] = teacher_weight
            total_weight += teacher_weight
        
        # 归一化权重
        if total_weight > 0:
            for i in teacher_weights:
                teacher_weights[i] /= total_weight
        
        return teacher_weights
    
    def update_weights(self, current_losses: dict, current_teacher_idx: int = None) -> dict:
        """更新权重，考虑teacher平衡"""
        # 更新损失历史
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.loss_histories:
                self.loss_histories[loss_name].append(loss_value)
        
        # 如果提供了teacher信息，更新teacher性能
        if current_teacher_idx is not None:
            total_loss = sum(current_losses.values())
            self.update_teacher_performance(current_teacher_idx, total_loss, self.global_step)
        
        # 计算基本的损失权重
        base_weights = self._compute_base_loss_weights(current_losses)
        
        # 获取teacher自适应权重
        teacher_weights = self.get_teacher_adaptive_weights()
        
        # 如果当前有指定teacher，应用teacher特定的调整
        if current_teacher_idx is not None:
            teacher_factor = teacher_weights.get(current_teacher_idx, 1.0)
            
            # 对困难teacher增加权重
            adjusted_weights = {}
            for loss_name, weight in base_weights.items():
                # 应用teacher特定的权重调整
                adjusted_weight = weight * teacher_factor
                
                # 确保在合理范围内
                config = self.loss_config[loss_name]
                adjusted_weight = max(config['min_weight'], 
                                    min(config['max_weight'], adjusted_weight))
                
                adjusted_weights[loss_name] = adjusted_weight
            
            return adjusted_weights
        
        return base_weights
    
    def _compute_base_loss_weights(self, current_losses: dict) -> dict:
        """计算基础损失权重（原有逻辑）"""
        if not any(len(history) >= 2 for history in self.loss_histories.values()):
            return dict(self.current_weights)
        
        loss_trends = {}
        for loss_name in self.loss_config.keys():
            if loss_name in self.loss_histories:
                loss_trends[loss_name] = self._compute_loss_trend(self.loss_histories[loss_name])
        
        new_weights = self._compute_softmax_weights(loss_trends)
        self.current_weights.update(new_weights)
        
        return dict(new_weights)
    
    def _compute_loss_trend(self, loss_history):
        """计算损失变化趋势"""
        if len(loss_history) < 2:
            return 0.0
        
        recent_losses = list(loss_history)[-min(8, len(loss_history)):]
        
        if len(recent_losses) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(recent_losses)):
            change = (recent_losses[i] - recent_losses[i-1]) / (recent_losses[i-1] + 1e-8)
            changes.append(change)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def _compute_softmax_weights(self, loss_trends):
        """使用softmax计算新权重"""
        loss_names = list(self.loss_config.keys())
        
        logits = []
        for loss_name in loss_names:
            trend = loss_trends.get(loss_name, 0.0)
            priority = self.loss_config[loss_name].get('priority', 1.0)
            
            adjustment = (-trend + priority) * self.temperature
            logits.append(adjustment)
        
        logits_tensor = torch.tensor(logits)
        weights_ratio = F.softmax(logits_tensor, dim=0)
        
        total_budget = sum(config['initial_weight'] for config in self.loss_config.values())
        
        new_weights = {}
        for i, loss_name in enumerate(loss_names):
            config = self.loss_config[loss_name]
            
            base_weight = weights_ratio[i] * total_budget * config.get('priority', 1.0)
            
            new_weight = torch.clamp(
                base_weight, 
                config['min_weight'], 
                config['max_weight']
            ).item()
            
            new_weights[loss_name] = new_weight
        
        return new_weights
    
    def get_teacher_statistics(self) -> dict:
        """获取teacher统计信息"""
        stats = {}
        for i, teacher_info in enumerate(self.teacher_info):
            perf = self.teacher_performance[i]
            teacher_name = teacher_info.get('name', f'teacher_{i+1}')
            
            stats[teacher_name] = {
                'avg_loss': sum(perf['loss_history']) / len(perf['loss_history']) if perf['loss_history'] else 0,
                'convergence_rate': perf['convergence_rate'],
                'relative_performance': perf['relative_performance'],
                'training_difficulty': perf['training_difficulty'],
                'sample_count': perf['sample_count'],
                'selection_count': self.teacher_selection_counts[i],
                'steps_since_improvement': self.global_step - perf['last_improvement']
            }
        
        return stats
    
    def should_focus_on_teacher(self, teacher_idx: int) -> bool:
        """判断是否需要重点关注某个teacher"""
        perf = self.teacher_performance[teacher_idx]
        
        # 检查性能是否明显落后
        avg_performance = sum(p['relative_performance'] for p in self.teacher_performance.values()) / self.num_teachers
        
        return (perf['relative_performance'] < avg_performance - self.performance_gap_threshold or
                perf['convergence_rate'] < 0.3 or
                self.global_step - perf['last_improvement'] > 200)


# ============ 创建增强版权重调整器的工厂函数 ============

def create_enhanced_tuning_weight_adjuster(teacher_info: List[dict], 
                                         noise_pred_weight=1.0, 
                                         feature_align_weight=0.001):
    """创建增强版的tuning权重调整器"""
    tuning_config = {
        'noise_pred': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 5.0,  # 增加上限
            'priority': 2.0
        },
        'feature_align': {
            'initial_weight': feature_align_weight,
            'min_weight': 0.0001,
            'max_weight': 0.05,  # 增加上限
            'priority': 1.0
        }
    }
    return EnhancedMultiTeacherWeightAdjuster(
        teacher_info=teacher_info,
        loss_config=tuning_config, 
        temperature=2.0,
        teacher_balance_weight=0.4,
        performance_window=100
    )

def create_enhanced_inversion_weight_adjuster(teacher_info: List[dict],
                                            noise_pred_weight=1.0, 
                                            unet_feature_weight=0.001, 
                                            text_encoder_feature_weight=0.001):
    """创建增强版的inversion权重调整器"""
    inversion_config = {
        'ti_loss': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 5.0,
            'priority': 2.5
        },
        'unet_feature_align': {
            'initial_weight': unet_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.05,
            'priority': 1.5
        },
        'text_encoder_feature_align': {
            'initial_weight': text_encoder_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.0
        }
    }
    return EnhancedMultiTeacherWeightAdjuster(
        teacher_info=teacher_info,
        loss_config=inversion_config, 
        temperature=1.5,
        teacher_balance_weight=0.35,
        performance_window=80
    )

class MultiTeacherLossAggregator(nn.Module):
    """多Teacher损失聚合器，使用注意力机制动态分配权重"""
    
    def __init__(self, num_teachers, feature_dim=768, temperature=1.0):
        super().__init__()
        self.num_teachers = num_teachers
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # 注意：这里先不初始化网络，在第一次forward时根据实际输入维度初始化
        self.teacher_attention = None
        self.similarity_net = None
        self.weight_adjustment = None
        self._initialized = False
        
    def _initialize_networks(self, actual_feature_dim):
        """根据实际特征维度初始化网络"""
        self.feature_dim = actual_feature_dim
        
        # 用于计算teacher重要性的注意力网络
        self.teacher_attention = nn.Sequential(
            nn.Linear(actual_feature_dim, max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), max(actual_feature_dim // 4, 32)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 4, 32), 1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # 学生-teacher相似度计算网络
        self.similarity_net = nn.Sequential(
            nn.Linear(actual_feature_dim * 2, max(actual_feature_dim, 128)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim, 128), max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # 动态权重调整网络
        self.weight_adjustment = nn.Sequential(
            nn.Linear(self.num_teachers + 1, max(self.num_teachers * 2, 8)),  # +1 for current loss
            nn.ReLU(),
            nn.Linear(max(self.num_teachers * 2, 8), self.num_teachers),
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        self._initialized = True
        
    def forward(self, student_features, teacher_features_list, noise_losses, current_step=0):
        """
        Args:
            student_features: [B, feature_dim] - 学生模型的全局特征表示
            teacher_features_list: List of [B, feature_dim] - 各teacher的特征表示
            noise_losses: List of scalars - 各teacher对应的噪声预测损失
            current_step: int - 当前训练步数
        """
        batch_size = student_features.size(0)
        device = student_features.device
        
        # 确保所有输入特征都转换为float32，解决数据类型不匹配问题
        student_features = student_features.float()
        teacher_features_list = [feat.float() for feat in teacher_features_list]
        
        # 获取实际的特征维度并初始化网络（如果还没初始化）
        actual_feature_dim = student_features.size(-1)
        if not self._initialized:
            self._initialize_networks(actual_feature_dim)
        
        # 检查特征维度是否与初始化时一致
        if actual_feature_dim != self.feature_dim:
            print(f"Warning: Feature dimension mismatch. Expected {self.feature_dim}, got {actual_feature_dim}. Re-initializing networks...")
            self._initialize_networks(actual_feature_dim)
        
        # 1. 计算每个teacher的重要性分数
        teacher_importance_scores = []
        for teacher_feat in teacher_features_list:
            teacher_feat = teacher_feat.to(device=device)
            # 确保teacher特征是float32类型再传入网络
            importance = self.teacher_attention(teacher_feat.float())  # [B, 1]
            teacher_importance_scores.append(importance)
        
        teacher_importance = torch.cat(teacher_importance_scores, dim=-1)  # [B, num_teachers]
        
        # 2. 计算学生与各teacher的相似度
        student_teacher_similarities = []
        for teacher_feat in teacher_features_list:
            teacher_feat = teacher_feat.to(device=device)
            combined_feat = torch.cat([student_features, teacher_feat.float()], dim=-1)  # 添加.float()
            similarity = self.similarity_net(combined_feat)  # [B, 1]
            student_teacher_similarities.append(similarity)
        
        similarities = torch.cat(student_teacher_similarities, dim=-1)  # [B, num_teachers]
        
        # 3. 计算基于损失的teacher选择倾向
        loss_tensor = torch.tensor(noise_losses, device=device, dtype=torch.float32)
        loss_weights = F.softmax(loss_tensor / self.temperature, dim=0)
        loss_weights = loss_weights.unsqueeze(0).expand(batch_size, -1)  # [B, num_teachers]
        
        # 4. 综合考虑多个因素计算最终权重
        step_factor = torch.full((batch_size, 1), current_step / 10000.0, device=device, dtype=torch.float32)
        
        combined_input = torch.cat([
            teacher_importance,
            similarities, 
            loss_weights,
            step_factor
        ], dim=-1)  # [B, num_teachers * 3 + 1]
        
        # 动态权重计算
        dynamic_weights = self.weight_adjustment(combined_input)  # [B, num_teachers]
        
        # 5. 应用温度缩放和批次平均
        final_weights = F.softmax(dynamic_weights / self.temperature, dim=-1)
        final_weights = final_weights.mean(dim=0)  # [num_teachers] - 批次平均
        
        # 6. 计算加权损失
        weighted_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for w, loss in zip(final_weights, noise_losses):
            loss_tensor = torch.tensor(loss, device=device, dtype=torch.float32)
            weighted_loss += w * loss_tensor
        
        return weighted_loss, final_weights, {
            'teacher_importance': teacher_importance.mean(dim=0),
            'similarities': similarities.mean(dim=0),
            'loss_weights': loss_weights.mean(dim=0),
            'final_weights': final_weights
        }

class MultiTeacherFeatureAlignmentLoss(nn.Module):
    """多Teacher特征对齐损失"""
    
    def __init__(self, num_teachers, feature_dim=768):
        super().__init__()
        self.num_teachers = num_teachers
        self.feature_dim = feature_dim
        
        # 同样延迟初始化
        self.feature_fusion = None
        self.weight_predictor = None
        self._initialized = False
        
    def _initialize_networks(self, actual_feature_dim):
        """根据实际特征维度初始化网络"""
        self.feature_dim = actual_feature_dim
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(actual_feature_dim * self.num_teachers, max(actual_feature_dim * 2, 256)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim * 2, 256), actual_feature_dim),
            nn.LayerNorm(actual_feature_dim)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(actual_feature_dim, max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), self.num_teachers),
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        self._initialized = True
    
    def forward(self, student_features, teacher_features_list):
        """
        计算学生特征与多个teacher特征的对齐损失
        """
        # 确保数据类型一致
        student_features = student_features.float()
        teacher_features_list = [feat.float() for feat in teacher_features_list]
        
        # 获取实际特征维度并初始化网络（如果还没初始化）
        if len(student_features.shape) == 3:  # [B, seq_len, feature_dim]
            actual_feature_dim = student_features.size(-1)
        else:  # [B, feature_dim]
            actual_feature_dim = student_features.size(-1)
            
        if not self._initialized:
            self._initialize_networks(actual_feature_dim)
        
        # 检查特征维度是否与初始化时一致
        if actual_feature_dim != self.feature_dim:
            print(f"Warning: Feature dimension mismatch in alignment loss. Expected {self.feature_dim}, got {actual_feature_dim}. Re-initializing networks...")
            self._initialize_networks(actual_feature_dim)
        
        # 处理不同形状的特征
        if len(student_features.shape) == 3:  # [B, seq_len, feature_dim]
            # 融合所有teacher特征
            teacher_concat = torch.cat(teacher_features_list, dim=-1)  # [B, seq_len, feature_dim * num_teachers]
            fused_teacher = self.feature_fusion(teacher_concat)  # [B, seq_len, feature_dim]
            
            # 预测权重
            weights = self.weight_predictor(student_features.mean(dim=1))  # [B, num_teachers]
            
            # 加权teacher特征
            weighted_teacher = torch.zeros_like(student_features)
            for i, teacher_feat in enumerate(teacher_features_list):
                weight = weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
                weighted_teacher += weight * teacher_feat
        else:  # [B, feature_dim]
            # 融合所有teacher特征
            teacher_concat = torch.cat(teacher_features_list, dim=-1)  # [B, feature_dim * num_teachers]
            fused_teacher = self.feature_fusion(teacher_concat)  # [B, feature_dim]
            
            # 预测权重
            weights = self.weight_predictor(student_features)  # [B, num_teachers]
            
            # 加权teacher特征
            weighted_teacher = torch.zeros_like(student_features)
            for i, teacher_feat in enumerate(teacher_features_list):
                weight = weights[:, i:i+1]  # [B, 1]
                weighted_teacher += weight * teacher_feat
        
        # 计算对齐损失
        alignment_loss = F.mse_loss(student_features, weighted_teacher)
        fusion_loss = F.mse_loss(student_features, fused_teacher)
        
        total_loss = 0.7 * alignment_loss + 0.3 * fusion_loss
        
        return total_loss, weights.mean(dim=0)