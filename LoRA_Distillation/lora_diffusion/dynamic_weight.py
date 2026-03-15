import torch
import torch.nn.functional as F
from collections import deque
import torch.nn as nn

# General-purpose multi-loss dynamic weight adjuster
class MultiLossDynamicWeightAdjuster:
    def __init__(self, 
                 loss_config,
                 history_length=10,
                 temperature=1.0):
        """
        General-purpose multi-loss dynamic weight adjuster
        
        Args:
            loss_config: Loss configuration dict with the following format:
                {
                    'loss_name': {
                        'initial_weight': float,
                        'min_weight': float,
                        'max_weight': float,
                        'priority': float  # Weight allocation priority
                    }
                }
            history_length: Number of historical loss values to retain
            temperature: Softmax temperature parameter controlling adjustment sensitivity
        """
        self.loss_config = loss_config
        self.history_length = history_length
        self.temperature = temperature
        
        # Store historical losses - one queue per loss type
        self.loss_histories = {}
        self.current_weights = {}
        
        for loss_name, config in loss_config.items():
            self.loss_histories[loss_name] = deque(maxlen=history_length)
            self.current_weights[loss_name] = config['initial_weight']
    
    def update_weights(self, current_losses):
        """
        Update weights based on current losses
        
        Args:
            current_losses: Current loss dict {'loss_name': loss_value}
            
        Returns:
            dict: Updated weight dict {'loss_name': weight_value}
        """
        # Add current losses to history
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.loss_histories:
                self.loss_histories[loss_name].append(loss_value)
        
        # If insufficient history, use current weights
        min_history_required = 2
        if any(len(history) < min_history_required for history in self.loss_histories.values()):
            return dict(self.current_weights)
        
        # Compute the trend of change for each loss
        loss_trends = {}
        for loss_name in self.loss_config.keys():
            if loss_name in self.loss_histories:
                loss_trends[loss_name] = self._compute_loss_trend(self.loss_histories[loss_name])
        
        # Use softmax to compute new weights
        new_weights = self._compute_softmax_weights(loss_trends)
        
        # Update current weights
        self.current_weights.update(new_weights)
        
        return dict(new_weights)
    
    def _compute_loss_trend(self, loss_history):
        """Compute the loss change trend."""
        if len(loss_history) < 2:
            return 0.0
        
        # Compute average rate of change over recent steps
        recent_losses = list(loss_history)[-min(5, len(loss_history)):]
        
        if len(recent_losses) < 2:
            return 0.0
        
        # Compute slope (trend)
        changes = []
        for i in range(1, len(recent_losses)):
            change = (recent_losses[i] - recent_losses[i-1]) / (recent_losses[i-1] + 1e-8)
            changes.append(change)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def _compute_softmax_weights(self, loss_trends):
        """Compute new weights using softmax."""
        loss_names = list(self.loss_config.keys())
        
        # Build logits for weight adjustment
        logits = []
        for loss_name in loss_names:
            trend = loss_trends.get(loss_name, 0.0)
            priority = self.loss_config[loss_name].get('priority', 1.0)
            
            # Inverse adjustment: increase weight when loss rises, also considering priority
            adjustment = (-trend + priority) * self.temperature
            logits.append(adjustment)
        
        # Apply softmax to get allocation ratios
        logits_tensor = torch.tensor(logits)
        weights_ratio = F.softmax(logits_tensor, dim=0)
        
        # Compute total weight budget
        total_budget = sum(config['initial_weight'] for config in self.loss_config.values())
        
        # Allocate new weights
        new_weights = {}
        for i, loss_name in enumerate(loss_names):
            config = self.loss_config[loss_name]
            
            # Allocate weights based on softmax ratios and priority
            base_weight = weights_ratio[i] * total_budget * config.get('priority', 1.0)
            
            # Apply min/max clipping
            new_weight = torch.clamp(
                base_weight, 
                config['min_weight'], 
                config['max_weight']
            ).item()
            
            new_weights[loss_name] = new_weight
        
        return new_weights

# ============ Usage examples ============

# 1. Create weight adjuster for perform_tuning stage
def create_tuning_weight_adjuster(noise_pred_weight=1.0, feature_align_weight=0.001):
    tuning_config = {
        'noise_pred': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 3.0,
            'priority': 2.0  # Noise prediction has higher priority
        },
        'feature_align': {
            'initial_weight': feature_align_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.0  # Feature alignment has relatively lower priority
        }
    }
    return MultiLossDynamicWeightAdjuster(tuning_config, temperature=2.0)

# 2. Create weight adjuster for train_inversion stage
def create_inversion_weight_adjuster(noise_pred_weight=1.0, 
                                   unet_feature_weight=0.001, 
                                   text_encoder_feature_weight=0.001):
    inversion_config = {
        'ti_loss': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 3.0,
            'priority': 2.5  # TI loss has highest priority
        },
        'unet_feature_align': {
            'initial_weight': unet_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.5  # UNet feature alignment has medium priority
        },
        'text_encoder_feature_align': {
            'initial_weight': text_encoder_feature_weight,
            'min_weight': 0.0001,
            'max_weight': 0.02,
            'priority': 1.0  # Text encoder feature alignment has lower priority
        }
    }
    return MultiLossDynamicWeightAdjuster(inversion_config, temperature=1.5)

# ============ How to use in training code ============

# --- In perform_tuning stage ---
"""
# Initialize weight adjuster
tuning_weight_adjuster = create_tuning_weight_adjuster(
    noise_pred_weight=noise_pred_weight,
    feature_align_weight=feature_align_weight
)

# Use in training loop
current_losses = {
    'noise_pred': current_noise_loss_item,
    'feature_align': current_feature_loss_item
}

# Update weights
updated_weights = tuning_weight_adjuster.update_weights(current_losses)
noise_pred_weight = updated_weights['noise_pred']
feature_align_weight = updated_weights['feature_align']

# Compute total loss using dynamic weights
total_loss = (
    noise_pred_weight * noise_pred_loss +
    feature_align_weight * feature_align_loss
)
"""

# --- In train_inversion stage ---
"""
# Initialize weight adjuster
inversion_weight_adjuster = create_inversion_weight_adjuster(
    noise_pred_weight=noise_pred_weight,
    unet_feature_weight=unet_feature_align_weight,
    text_encoder_feature_weight=text_encoder_feature_align_weight
)

# Use in training loop
current_losses = {
    'ti_loss': ti_loss_current_step.item(),
    'unet_feature_align': unet_feature_align_loss_val.item() if isinstance(unet_feature_align_loss_val, torch.Tensor) else unet_feature_align_loss_val,
    'text_encoder_feature_align': text_encoder_feature_align_loss_val.item() if isinstance(text_encoder_feature_align_loss_val, torch.Tensor) else text_encoder_feature_align_loss_val
}

# Update weights
updated_weights = inversion_weight_adjuster.update_weights(current_losses)
noise_pred_weight = updated_weights['ti_loss']
unet_feature_align_weight = updated_weights['unet_feature_align']
text_encoder_feature_align_weight = updated_weights['text_encoder_feature_align']

# Compute total loss using dynamic weights
current_step_total_loss = (
    noise_pred_weight * ti_loss_current_step +
    unet_feature_align_weight * unet_feature_align_loss_val +
    text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
)

# Log weight changes to TensorBoard
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
                 teacher_balance_weight=0.3,  # New: teacher balance weight
                 convergence_threshold=0.01,  # New: convergence threshold
                 performance_window=50):      # New: performance window
        """
        Enhanced multi-teacher dynamic weight adjuster
        
        Args:
            teacher_info: List of teacher info entries
            loss_config: Loss configuration
            teacher_balance_weight: Importance weight for teacher balancing
            convergence_threshold: Threshold for determining loss convergence
            performance_window: Window size for performance evaluation
        """
        self.teacher_info = teacher_info
        self.num_teachers = len(teacher_info)
        self.loss_config = loss_config
        self.history_length = history_length
        self.temperature = temperature
        self.teacher_balance_weight = teacher_balance_weight
        self.convergence_threshold = convergence_threshold
        self.performance_window = performance_window
        
        # Original loss history records
        self.loss_histories = {}
        self.current_weights = {}
        
        # New: performance tracking per teacher
        self.teacher_performance = {i: {
            'loss_history': deque(maxlen=performance_window),
            'convergence_rate': 0.0,
            'relative_performance': 1.0,  # Performance ratio relative to best teacher
            'training_difficulty': 1.0,   # Estimated training difficulty
            'sample_count': 0,            # Training sample count
            'last_improvement': 0         # Step number of last improvement
        } for i in range(self.num_teachers)}
        
        # New: global statistics
        self.global_step = 0
        self.teacher_selection_counts = [0] * self.num_teachers
        self.performance_gap_threshold = 0.2  # Performance gap threshold
        
        for loss_name, config in loss_config.items():
            self.loss_histories[loss_name] = deque(maxlen=history_length)
            self.current_weights[loss_name] = config['initial_weight']
    
    def update_teacher_performance(self, teacher_idx: int, loss_value: float, global_step: int):
        """Update the performance record for the specified teacher."""
        self.global_step = global_step
        self.teacher_selection_counts[teacher_idx] += 1
        
        perf = self.teacher_performance[teacher_idx]
        perf['loss_history'].append(loss_value)
        perf['sample_count'] += 1
        
        # Compute convergence rate
        if len(perf['loss_history']) >= 5:
            recent_losses = list(perf['loss_history'])[-5:]
            perf['convergence_rate'] = self._compute_convergence_rate(recent_losses)
            
            # Check for improvement
            if len(perf['loss_history']) >= 2:
                if perf['loss_history'][-1] < perf['loss_history'][-2]:
                    perf['last_improvement'] = global_step
        
        # Update relative performance and training difficulty
        self._update_relative_performance()
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute the convergence speed of the loss."""
        if len(losses) < 3:
            return 0.0
        
        # Compute rate of change of loss
        changes = []
        for i in range(1, len(losses)):
            if losses[i-1] != 0:
                change_rate = abs(losses[i] - losses[i-1]) / losses[i-1]
                changes.append(change_rate)
        
        if not changes:
            return 0.0
        
        # Convergence rate = 1 - average rate of change (higher = more stable)
        avg_change_rate = sum(changes) / len(changes)
        return max(0.0, 1.0 - avg_change_rate)
    
    def _update_relative_performance(self):
        """Update relative performance for each teacher."""
        # Compute average loss for each teacher
        teacher_avg_losses = []
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            if perf['loss_history']:
                avg_loss = sum(perf['loss_history']) / len(perf['loss_history'])
                teacher_avg_losses.append(avg_loss)
            else:
                teacher_avg_losses.append(float('inf'))
        
        # Find best performance
        best_loss = min(loss for loss in teacher_avg_losses if loss != float('inf'))
        if best_loss == float('inf'):
            return
        
        # Update relative performance and training difficulty
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            if teacher_avg_losses[i] != float('inf'):
                # Relative performance: best performance / current performance
                perf['relative_performance'] = best_loss / teacher_avg_losses[i]
                
                # Training difficulty: based on loss level and convergence rate
                difficulty_factor = teacher_avg_losses[i] / best_loss
                convergence_factor = 1.0 / (perf['convergence_rate'] + 0.1)
                perf['training_difficulty'] = difficulty_factor * convergence_factor
    
    def get_teacher_adaptive_weights(self) -> Dict[int, float]:
        """Compute adaptive weights for each teacher."""
        if self.global_step < 10:  # Use uniform weights early on
            return {i: 1.0 for i in range(self.num_teachers)}
        
        teacher_weights = {}
        total_weight = 0.0
        
        for i in range(self.num_teachers):
            perf = self.teacher_performance[i]
            
            # Base weight factor
            base_weight = 1.0
            
            # 1. Performance factor: worse performance → higher weight (needs more attention)
            performance_factor = 2.0 - perf['relative_performance']
            performance_factor = max(0.5, min(2.0, performance_factor))
            
            # 2. Training difficulty factor: higher difficulty → higher weight
            difficulty_factor = perf['training_difficulty']
            difficulty_factor = max(0.8, min(1.5, difficulty_factor))
            
            # 3. Convergence factor: slow convergence needs more training
            convergence_factor = 1.0 + (1.0 - perf['convergence_rate'])
            convergence_factor = max(1.0, min(1.8, convergence_factor))
            
            # 4. Sample balance factor: teachers with fewer samples need more training
            min_samples = min(p['sample_count'] for p in self.teacher_performance.values())
            max_samples = max(p['sample_count'] for p in self.teacher_performance.values())
            if max_samples > min_samples:
                sample_ratio = perf['sample_count'] / max_samples
                balance_factor = 1.0 + (1.0 - sample_ratio) * 0.5
            else:
                balance_factor = 1.0
            
            # 5. Long-term no-improvement penalty
            steps_since_improvement = self.global_step - perf['last_improvement']
            if steps_since_improvement > 100:
                stagnation_factor = 1.0 + min(0.5, steps_since_improvement / 1000)
            else:
                stagnation_factor = 1.0
            
            # Combined weight computation
            teacher_weight = (base_weight * 
                            performance_factor * 
                            difficulty_factor * 
                            convergence_factor * 
                            balance_factor * 
                            stagnation_factor)
            
            teacher_weights[i] = teacher_weight
            total_weight += teacher_weight
        
        # Normalize weights
        if total_weight > 0:
            for i in teacher_weights:
                teacher_weights[i] /= total_weight
        
        return teacher_weights
    
    def update_weights(self, current_losses: dict, current_teacher_idx: int = None) -> dict:
        """Update weights, taking teacher balance into account."""
        # Update loss history
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.loss_histories:
                self.loss_histories[loss_name].append(loss_value)
        
        # If teacher info is provided, update teacher performance
        if current_teacher_idx is not None:
            total_loss = sum(current_losses.values())
            self.update_teacher_performance(current_teacher_idx, total_loss, self.global_step)
        
        # Compute basic loss weights
        base_weights = self._compute_base_loss_weights(current_losses)
        
        # Get adaptive weights per teacher
        teacher_weights = self.get_teacher_adaptive_weights()
        
        # If a teacher is currently specified, apply teacher-specific adjustments
        if current_teacher_idx is not None:
            teacher_factor = teacher_weights.get(current_teacher_idx, 1.0)
            
            # Increase weight for difficult teachers
            adjusted_weights = {}
            for loss_name, weight in base_weights.items():
                # Apply teacher-specific weight adjustment
                adjusted_weight = weight * teacher_factor
                
                # Ensure within reasonable range
                config = self.loss_config[loss_name]
                adjusted_weight = max(config['min_weight'], 
                                    min(config['max_weight'], adjusted_weight))
                
                adjusted_weights[loss_name] = adjusted_weight
            
            return adjusted_weights
        
        return base_weights
    
    def _compute_base_loss_weights(self, current_losses: dict) -> dict:
        """Compute base loss weights (original logic)."""
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
        """Compute loss change trend."""
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
        """Compute new weights using softmax."""
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
        """Get teacher statistics."""
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
        """Determine whether a teacher needs special focus."""
        perf = self.teacher_performance[teacher_idx]
        
        # Check whether performance is significantly behind
        avg_performance = sum(p['relative_performance'] for p in self.teacher_performance.values()) / self.num_teachers
        
        return (perf['relative_performance'] < avg_performance - self.performance_gap_threshold or
                perf['convergence_rate'] < 0.3 or
                self.global_step - perf['last_improvement'] > 200)


# ============ Factory functions for creating enhanced weight adjusters ============

def create_enhanced_tuning_weight_adjuster(teacher_info: List[dict], 
                                         noise_pred_weight=1.0, 
                                         feature_align_weight=0.001):
    """Create an enhanced tuning weight adjuster."""
    tuning_config = {
        'noise_pred': {
            'initial_weight': noise_pred_weight,
            'min_weight': 0.1,
            'max_weight': 5.0,  # Increased upper bound
            'priority': 2.0
        },
        'feature_align': {
            'initial_weight': feature_align_weight,
            'min_weight': 0.0001,
            'max_weight': 0.05,  # Increased upper bound
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
    """Create an enhanced inversion weight adjuster."""
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
    """Multi-teacher loss aggregator using attention to dynamically allocate weights."""
    
    def __init__(self, num_teachers, feature_dim=768, temperature=1.0):
        super().__init__()
        self.num_teachers = num_teachers
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Note: network is not initialized here; initialization happens on first forward pass based on actual input dimensions
        self.teacher_attention = None
        self.similarity_net = None
        self.weight_adjustment = None
        self._initialized = False
        
    def _initialize_networks(self, actual_feature_dim):
        """Initialize network based on actual feature dimensions."""
        self.feature_dim = actual_feature_dim
        
        # Attention network for computing teacher importance
        self.teacher_attention = nn.Sequential(
            nn.Linear(actual_feature_dim, max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), max(actual_feature_dim // 4, 32)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 4, 32), 1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Student-teacher similarity computation network
        self.similarity_net = nn.Sequential(
            nn.Linear(actual_feature_dim * 2, max(actual_feature_dim, 128)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim, 128), max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Dynamic weight adjustment network
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
            student_features: [B, feature_dim] - global feature representation of student model
            teacher_features_list: List of [B, feature_dim] - feature representations of each teacher
            noise_losses: List of scalars - noise prediction loss for each teacher
            current_step: int - current training step
        """
        batch_size = student_features.size(0)
        device = student_features.device
        
        # Ensure all input features are converted to float32 to avoid dtype mismatch
        student_features = student_features.float()
        teacher_features_list = [feat.float() for feat in teacher_features_list]
        
        # Get actual feature dimension and initialize network (if not yet initialized)
        actual_feature_dim = student_features.size(-1)
        if not self._initialized:
            self._initialize_networks(actual_feature_dim)
        
        # Check that feature dimension matches initialization
        if actual_feature_dim != self.feature_dim:
            print(f"Warning: Feature dimension mismatch. Expected {self.feature_dim}, got {actual_feature_dim}. Re-initializing networks...")
            self._initialize_networks(actual_feature_dim)
        
        # 1. Compute importance score for each teacher
        teacher_importance_scores = []
        for teacher_feat in teacher_features_list:
            teacher_feat = teacher_feat.to(device=device)
            # Ensure teacher features are float32 before passing to network
            importance = self.teacher_attention(teacher_feat.float())  # [B, 1]
            teacher_importance_scores.append(importance)
        
        teacher_importance = torch.cat(teacher_importance_scores, dim=-1)  # [B, num_teachers]
        
        # 2. Compute similarity between student and each teacher
        student_teacher_similarities = []
        for teacher_feat in teacher_features_list:
            teacher_feat = teacher_feat.to(device=device)
            combined_feat = torch.cat([student_features, teacher_feat.float()], dim=-1)  # Added .float()
            similarity = self.similarity_net(combined_feat)  # [B, 1]
            student_teacher_similarities.append(similarity)
        
        similarities = torch.cat(student_teacher_similarities, dim=-1)  # [B, num_teachers]
        
        # 3. Compute teacher selection preference based on loss
        loss_tensor = torch.tensor(noise_losses, device=device, dtype=torch.float32)
        loss_weights = F.softmax(loss_tensor / self.temperature, dim=0)
        loss_weights = loss_weights.unsqueeze(0).expand(batch_size, -1)  # [B, num_teachers]
        
        # 4. Compute final weights by combining multiple factors
        step_factor = torch.full((batch_size, 1), current_step / 10000.0, device=device, dtype=torch.float32)
        
        combined_input = torch.cat([
            teacher_importance,
            similarities, 
            loss_weights,
            step_factor
        ], dim=-1)  # [B, num_teachers * 3 + 1]
        
        # Dynamic weight computation
        dynamic_weights = self.weight_adjustment(combined_input)  # [B, num_teachers]
        
        # 5. Apply temperature scaling and batch averaging
        final_weights = F.softmax(dynamic_weights / self.temperature, dim=-1)
        final_weights = final_weights.mean(dim=0)  # [num_teachers] - batch average
        
        # 6. Compute weighted loss
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
    """Multi-teacher feature alignment loss."""
    
    def __init__(self, num_teachers, feature_dim=768):
        super().__init__()
        self.num_teachers = num_teachers
        self.feature_dim = feature_dim
        
        # Also use deferred initialization
        self.feature_fusion = None
        self.weight_predictor = None
        self._initialized = False
        
    def _initialize_networks(self, actual_feature_dim):
        """Initialize network based on actual feature dimensions."""
        self.feature_dim = actual_feature_dim
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(actual_feature_dim * self.num_teachers, max(actual_feature_dim * 2, 256)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim * 2, 256), actual_feature_dim),
            nn.LayerNorm(actual_feature_dim)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Weight prediction network
        self.weight_predictor = nn.Sequential(
            nn.Linear(actual_feature_dim, max(actual_feature_dim // 2, 64)),
            nn.ReLU(),
            nn.Linear(max(actual_feature_dim // 2, 64), self.num_teachers),
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        self._initialized = True
    
    def forward(self, student_features, teacher_features_list):
        """
        Compute alignment loss between student features and multiple teacher features
        """
        # Ensure consistent data types
        student_features = student_features.float()
        teacher_features_list = [feat.float() for feat in teacher_features_list]
        
        # Get actual feature dimension and initialize network (if not yet initialized)
        if len(student_features.shape) == 3:  # [B, seq_len, feature_dim]
            actual_feature_dim = student_features.size(-1)
        else:  # [B, feature_dim]
            actual_feature_dim = student_features.size(-1)
            
        if not self._initialized:
            self._initialize_networks(actual_feature_dim)
        
        # Check that feature dimension matches initialization
        if actual_feature_dim != self.feature_dim:
            print(f"Warning: Feature dimension mismatch in alignment loss. Expected {self.feature_dim}, got {actual_feature_dim}. Re-initializing networks...")
            self._initialize_networks(actual_feature_dim)
        
        # Handle features of different shapes
        if len(student_features.shape) == 3:  # [B, seq_len, feature_dim]
            # Fuse all teacher features
            teacher_concat = torch.cat(teacher_features_list, dim=-1)  # [B, seq_len, feature_dim * num_teachers]
            fused_teacher = self.feature_fusion(teacher_concat)  # [B, seq_len, feature_dim]
            
            # Predict weights
            weights = self.weight_predictor(student_features.mean(dim=1))  # [B, num_teachers]
            
            # Weighted teacher features
            weighted_teacher = torch.zeros_like(student_features)
            for i, teacher_feat in enumerate(teacher_features_list):
                weight = weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
                weighted_teacher += weight * teacher_feat
        else:  # [B, feature_dim]
            # Fuse all teacher features
            teacher_concat = torch.cat(teacher_features_list, dim=-1)  # [B, feature_dim * num_teachers]
            fused_teacher = self.feature_fusion(teacher_concat)  # [B, feature_dim]
            
            # Predict weights
            weights = self.weight_predictor(student_features)  # [B, num_teachers]
            
            # Weighted teacher features
            weighted_teacher = torch.zeros_like(student_features)
            for i, teacher_feat in enumerate(teacher_features_list):
                weight = weights[:, i:i+1]  # [B, 1]
                weighted_teacher += weight * teacher_feat
        
        # Compute alignment loss
        alignment_loss = F.mse_loss(student_features, weighted_teacher)
        fusion_loss = F.mse_loss(student_features, fused_teacher)
        
        total_loss = 0.7 * alignment_loss + 0.3 * fusion_loss
        
        return total_loss, weights.mean(dim=0)