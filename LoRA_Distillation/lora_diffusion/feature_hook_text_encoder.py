import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

class TextEncoderFeatureExtractor:
    def __init__(self, target_layers, mixed_precision_config=None):
        """
        Initialize TextEncoderFeatureExtractor.
        Args:
            target_layers (list of str): List of target layer names from which to extract features.
                e.g.: ['text_model.encoder.layers.0', 'text_model.encoder.layers.6', 'text_model.encoder.layers.11']
            mixed_precision_config (str or object, optional): Mixed precision configuration.
                Can be "fp16", "bf16", "no", or an Accelerate accelerator object,
                or any object with a 'mixed_precision' attribute.
        """
        if not isinstance(target_layers, list):
            raise ValueError(f"target_layers should be a list of layer names, got {type(target_layers)}")
        self.target_layers = target_layers
        self.features = {}  # Stores extracted features
        self.hooks = []     # Stores registered hook handles

        # Set self.mixed_precision_active based on mixed_precision_config
        self.mixed_precision_active = False
        if isinstance(mixed_precision_config, str):
            self.mixed_precision_active = (mixed_precision_config.lower() in ["fp16", "bf16"])
        elif hasattr(mixed_precision_config, 'mixed_precision') and mixed_precision_config.mixed_precision is not None:
            if isinstance(mixed_precision_config.mixed_precision, str):
                 self.mixed_precision_active = (mixed_precision_config.mixed_precision.lower() in ["fp16", "bf16"])

    def _hook_fn(self, module, input_tuple, output, layer_name):
        """
        The actual hook function, called after the hooked layer completes its forward pass.
        For CLIP Text Encoder, the output is typically a hidden_states tensor or a tuple with multiple elements.
        """
        feature_to_store = output
        
        # Handle possible tuple output
        if isinstance(output, tuple):
            # CLIP transformer layers typically return (hidden_states, attention_weights) or similar
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                feature_to_store = output[0]  # Usually the first element is hidden_states
            else:
                # If the tuple is empty or the first element is not a tensor, keep as is
                pass
        
        # For CLIP, we mainly care about hidden_states, typically a tensor of shape (batch_size, seq_len, hidden_size)
        self.features[layer_name] = feature_to_store

    def register_hooks(self, text_encoder):
        """
        Register forward hooks on target layers of the specified Text Encoder.
        Args:
            text_encoder (torch.nn.Module): The CLIP Text Encoder model on which to register hooks.
        """
        self.clear_hooks()  # Clear any existing hooks first
        self.features.clear()  # Clear previously extracted features

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
        Get all features captured through hooks.
        Returns:
            dict: A dictionary with layer names as keys and corresponding feature tensors as values.
        """
        return {k: v for k, v in self.features.items()}  # Return a copy

    def clear_hooks(self):
        """
        Remove all registered hooks and clear the list.
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
                         use_grad: bool = False,  # New: controls whether to run in grad context
                         **other_text_encoder_kwargs):
        """
        Run the CLIP Text Encoder forward pass and extract intermediate features from specified layers.
        Args:
            text_encoder: CLIP Text Encoder model
            input_ids: Input token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position encoding ids
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict format
            use_grad: If True, run within torch.enable_grad() context,
                     allowing differentiable feature extraction for the student model. Default is False
        """
        # 1. Register hooks
        self.register_hooks(text_encoder)

        # 2. Determine the actual parameter dtype of the Text Encoder
        try:
            encoder_internal_dtype = next(text_encoder.parameters()).dtype
        except StopIteration:
            try:
                encoder_internal_dtype = next(text_encoder.buffers()).dtype
            except StopIteration:
                encoder_internal_dtype = input_ids.dtype if hasattr(input_ids, 'dtype') else torch.float32

        # 3. Prepare input tensors
        # input_ids are usually long type; no dtype conversion needed
        input_input_ids = input_ids
        input_attention_mask = attention_mask
        input_position_ids = position_ids
        
        # If there are other float-type inputs, convert their dtype
        if attention_mask is not None and attention_mask.dtype != input_ids.dtype:
            if attention_mask.dtype.is_floating_point:
                input_attention_mask = attention_mask.to(dtype=encoder_internal_dtype)
        
        if position_ids is not None and position_ids.dtype != input_ids.dtype:
            if position_ids.dtype.is_floating_point:
                input_position_ids = position_ids.to(dtype=encoder_internal_dtype)

        # 4. Run forward pass
        main_output = None
        # Choose context based on use_grad
        context_manager = torch.enable_grad() if use_grad else torch.no_grad()

        with context_manager:
            # autocast is only enabled on CUDA when mixed_precision_active is True
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

                # 5. Get the main feature from the Text Encoder output
                if return_dict:
                    # CLIP Text Encoder typically returns an object containing last_hidden_state
                    if hasattr(output_data, "last_hidden_state"):
                        main_output = output_data.last_hidden_state
                    elif hasattr(output_data, "pooler_output"):
                        main_output = output_data.pooler_output
                    else:
                        # Try to get the first available tensor attribute
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
        
        # 6. Get features captured through hooks
        intermediate_features = self.get_features()

        # 7. Clear hooks
        self.clear_hooks()

        return main_output, intermediate_features

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()


class HybridTextEncoderFeatureAlignmentLoss(nn.Module):
    """
    🔥 Text Encoder feature alignment loss module with mixed loss support
    """
    def __init__(self, alignment_layers: List[str], 
                 loss_weights: Dict[str, float] = None,
                 temperature: float = 1.0, 
                 loss_type: str = "hybrid",
                 pooling_strategy: str = "mean",
                 # 🔥 New parameters for mixed loss
                 primary_loss_type: str = "mse",
                 secondary_loss_type: str = "cosine", 
                 loss_combination_weight: float = 0.3,
                 enable_magnitude_preservation: bool = True,
                 magnitude_weight: float = 0.2,
                 direction_weight: float = 0.8,
                 # 🔥 Layer-adaptive parameters
                 use_layer_adaptive: bool = False,
                 layer_loss_config: Dict[str, Dict] = None):
        """
        Initialize the mixed Text Encoder feature alignment loss module.
        Args:
            alignment_layers (List[str]): List of layer names whose features should be aligned.
            loss_weights (Dict[str, float], optional): Per-layer loss weights.
            temperature (float): Temperature used for scaling features.
            loss_type (str): Loss type - "mse", "l1", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
            pooling_strategy (str): Feature pooling strategy.
            
            # Mixed loss parameters
            primary_loss_type (str): Primary loss type (usually "mse")
            secondary_loss_type (str): Secondary loss type (usually "cosine")
            loss_combination_weight (float): Weight of secondary loss in the mixture (0.3 means 30% cosine + 70% mse)
            enable_magnitude_preservation (bool): Whether to enable magnitude preservation
            magnitude_weight (float): Magnitude loss weight
            direction_weight (float): Direction loss weight
            
            # Layer-adaptive parameters
            use_layer_adaptive (bool): Whether to use layer-adaptive loss
            layer_loss_config (Dict): Per-layer loss configuration
        """
        super().__init__()
        self.alignment_layers = alignment_layers
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.temperature = temperature
        self.pooling_strategy = pooling_strategy
        self.loss_type = loss_type.lower()
        
        # 🔥 Mixed loss parameters
        self.primary_loss_type = primary_loss_type.lower()
        self.secondary_loss_type = secondary_loss_type.lower()
        self.loss_combination_weight = loss_combination_weight
        self.enable_magnitude_preservation = enable_magnitude_preservation
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        
        # 🔥 Layer-adaptive parameters
        self.use_layer_adaptive = use_layer_adaptive
        self.layer_loss_config = layer_loss_config or self._get_default_layer_config()
        
        # Initialize base loss functions
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
        """Get the default layer-level loss configuration."""
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
        Pool text features.
        Args:
            features: Feature tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
        Returns:
            Pooled feature tensor
        """
        if self.pooling_strategy == "none":
            return features
        elif self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return features[:, 0, :]  # (batch_size, hidden_size)
        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Use attention mask for weighted average
                mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
                sum_features = torch.sum(features * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_features / sum_mask
            else:
                # Simple average
                return torch.mean(features, dim=1)
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                # Use attention mask for max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
                features_masked = features * mask_expanded + (1 - mask_expanded) * (-1e9)  # Set padding positions to a very small value
                return torch.max(features_masked, dim=1)[0]
            else:
                return torch.max(features, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

    def _compute_single_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, 
                           loss_type: str) -> torch.Tensor:
        """Compute a single loss type."""
        if loss_type == "mse":
            return self.mse_criterion(student_feat, teacher_feat)
        elif loss_type == "l1":
            return self.l1_criterion(student_feat, teacher_feat)
        elif loss_type == "cosine":
            # Cosine similarity loss
            target = torch.ones(student_feat.size(0), device=student_feat.device)
            return self.cosine_criterion(student_feat, teacher_feat, target)
        else:
            raise ValueError(f"Unsupported single loss type: {loss_type}")

    def _compute_hybrid_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                           layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 Compute mixed loss."""
        # Primary loss (usually MSE)
        primary_loss = self._compute_single_loss(student_feat, teacher_feat, self.primary_loss_type)
        
        # Secondary loss (usually Cosine)
        secondary_loss = self._compute_single_loss(student_feat, teacher_feat, self.secondary_loss_type)
        
        # Mixed loss
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
        """🔥 Compute scale-aware cosine loss."""
        # Flatten features for computation
        batch_size = student_feat.size(0)
        student_flat = student_feat.view(batch_size, -1)
        teacher_flat = teacher_feat.view(batch_size, -1)
        
        # 1. Direction loss (cosine similarity)
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        direction_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
        
        # 2. Magnitude loss (L2 norm difference)
        student_magnitude = torch.norm(student_flat, p=2, dim=1)
        teacher_magnitude = torch.norm(teacher_flat, p=2, dim=1)
        magnitude_loss = F.mse_loss(student_magnitude, teacher_magnitude)
        
        # 3. Mixed loss
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
        """🔥 Compute layer-adaptive loss."""
        config = self.layer_loss_config.get(layer_name, {'type': 'mse', 'weight': 1.0})
        loss_type = config['type']
        layer_weight = config['weight']
        
        if loss_type == 'hybrid':
            # Use mixed loss
            loss, loss_details = self._compute_hybrid_loss(student_feat, teacher_feat, layer_name)
        elif loss_type == 'scale_aware_cosine':
            # Use scale-aware cosine loss
            loss, loss_details = self._compute_scale_aware_cosine_loss(student_feat, teacher_feat, layer_name)
        else:
            # Use single loss
            loss = self._compute_single_loss(student_feat, teacher_feat, loss_type)
            loss_details = {f'{layer_name}_{loss_type}': loss.item()}
        
        # Apply layer-level weight
        weighted_loss = loss * layer_weight
        loss_details[f'{layer_name}_layer_weight'] = layer_weight
        loss_details[f'{layer_name}_weighted_total'] = weighted_loss.item()
        
        return weighted_loss, loss_details

    def _compute_layer_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                          layer_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """🔥 Compute layer loss according to configuration."""
        if self.loss_type == "hybrid":
            return self._compute_hybrid_loss(student_feat, teacher_feat, layer_name)
        elif self.loss_type == "scale_aware_cosine":
            return self._compute_scale_aware_cosine_loss(student_feat, teacher_feat, layer_name)
        elif self.loss_type == "layer_adaptive":
            return self._compute_layer_adaptive_loss(student_feat, teacher_feat, layer_name)
        else:
            # Single loss type
            loss = self._compute_single_loss(student_feat, teacher_feat, self.loss_type)
            loss_details = {f'{layer_name}_{self.loss_type}': loss.item()}
            return loss, loss_details

    def _adaptive_align_features(self, feat1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                feat2: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                attention_mask1: torch.Tensor = None,
                                attention_mask2: torch.Tensor = None):
        """
        Adaptively align two feature tensors.
        """
        # Handle possible tuple output
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

        # Pool text features (if needed)
        feat1_pooled = self._pool_text_features(feat1, attention_mask1)
        feat2_pooled = self._pool_text_features(feat2, attention_mask2)

        # Ensure dimensions match
        if feat1_pooled.shape != feat2_pooled.shape:
            # If hidden_size differs, consider adding a linear projection layer; simplified here
            min_dim = min(feat1_pooled.size(-1), feat2_pooled.size(-1))
            feat1_pooled = feat1_pooled[..., :min_dim]
            feat2_pooled = feat2_pooled[..., :min_dim]

        return feat1_pooled, feat2_pooled

    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor],
                teacher_attention_mask: torch.Tensor = None,
                student_attention_mask: torch.Tensor = None):
        """
        🔥 Compute mixed feature alignment loss.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        # Move total_loss to the device of the features
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

            # Convert to float for computation
            student_feat_for_loss = student_feat_processed.float()
            teacher_feat_for_loss = teacher_feat_processed.float()
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                student_feat_for_loss = student_feat_for_loss / self.temperature
                teacher_feat_for_loss = teacher_feat_for_loss / self.temperature

            # 🔥 Compute layer loss (supports mixed loss)
            layer_loss, layer_loss_details = self._compute_layer_loss(
                student_feat_for_loss, teacher_feat_for_loss, layer_name
            )
            
            # Apply global layer weight
            global_weight = self.loss_weights.get(layer_name, 1.0)
            weighted_layer_loss = global_weight * layer_loss
            
            total_loss += weighted_layer_loss
            loss_dict[layer_name] = weighted_layer_loss.item()
            
            # Save detailed loss info
            detailed_loss_info.update(layer_loss_details)
            detailed_loss_info[f'{layer_name}_global_weight'] = global_weight
            detailed_loss_info[f'{layer_name}_final_weighted'] = weighted_layer_loss.item()
            
            active_layers_count += 1
        
        if active_layers_count == 0:
            return torch.tensor(0.0, device=total_loss.device, dtype=torch.float32), loss_dict, detailed_loss_info

        # 🔥 Add loss type statistics
        detailed_loss_info['total_loss'] = total_loss.item()
        detailed_loss_info['active_layers'] = active_layers_count
        detailed_loss_info['loss_type'] = self.loss_type
        
        return total_loss, loss_dict, detailed_loss_info


# 🔥 Original class kept for backward compatibility (simple wrapper)
class TextEncoderFeatureAlignmentLoss(HybridTextEncoderFeatureAlignmentLoss):
    """Original TextEncoderFeatureAlignmentLoss, now inherits from mixed version for backward compatibility."""
    def __init__(self, alignment_layers: List[str], loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, loss_type: str = "mse", pooling_strategy: str = "mean"):
        # 🔥 Call mixed version initializer with single-loss configuration
        super().__init__(
            alignment_layers=alignment_layers,
            loss_weights=loss_weights,
            temperature=temperature,
            loss_type=loss_type,
            pooling_strategy=pooling_strategy,
            # Default params to preserve original behavior
            primary_loss_type="mse",
            secondary_loss_type="cosine",
            loss_combination_weight=0.0,  # Do not use mixing
            enable_magnitude_preservation=False,
            use_layer_adaptive=False
        )
    
    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor],
                teacher_attention_mask: torch.Tensor = None,
                student_attention_mask: torch.Tensor = None):
        """Forward pass preserving original return format."""
        total_loss, loss_dict, detailed_loss_info = super().forward(
            teacher_features, student_features, teacher_attention_mask, student_attention_mask
        )
        # Return only the first two values in original format
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
    🔥 Improved Text Encoder distillation loss computation with mixed feature alignment loss support.
    """
    
    # Teacher 1 forward pass (no grad)
    with torch.no_grad():
        if text_feature_extractor is not None:
            teacher1_output, teacher1_features = text_feature_extractor.extract_features(
                teacher1_text_encoder, input_ids, attention_mask, use_grad=False
            )
        else:
            teacher1_output = teacher1_text_encoder(input_ids, attention_mask=attention_mask)
            teacher1_features = {}
    
    # Teacher 2 forward pass (no grad)
    with torch.no_grad():
        if text_feature_extractor is not None:
            teacher2_output, teacher2_features = text_feature_extractor.extract_features(
                teacher2_text_encoder, input_ids, attention_mask, use_grad=False
            )
        else:
            teacher2_output = teacher2_text_encoder(input_ids, attention_mask=attention_mask)
            teacher2_features = {}
    
    # Student forward pass
    if text_feature_extractor is not None:
        student_output, student_features = text_feature_extractor.extract_features(
            student_text_encoder, input_ids, attention_mask, use_grad=use_grad_for_student
        )
    else:
        student_output = student_text_encoder(input_ids, attention_mask=attention_mask)
        student_features = {}
    
    # Get text embeddings (usually last_hidden_state or pooler_output)
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
    
    # Compute text embedding loss
    target_embeddings = (teacher1_embeddings + teacher2_embeddings) / 2  # Average the two teacher outputs
    embedding_loss = F.mse_loss(student_embeddings, target_embeddings)
    
    # 🔥 Compute mixed feature alignment loss
    total_feature_loss = torch.tensor(0.0, device=embedding_loss.device)
    feature_loss_dict = {}
    detailed_feature_info = {}
    
    if feature_alignment_loss_fn is not None and (teacher1_features or teacher2_features):
        # Align features with teacher1
        if teacher1_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # New mixed loss function, returns 3 values
                feature_loss1, feature_loss_dict1, detailed_info1 = feature_alignment_loss_fn(
                    teacher1_features, student_features, attention_mask, attention_mask
                )
                detailed_feature_info.update({f"t1_{k}": v for k, v in detailed_info1.items()})
            else:
                # Old loss function, returns 2 values
                feature_loss1, feature_loss_dict1 = feature_alignment_loss_fn(
                    teacher1_features, student_features, attention_mask, attention_mask
                )
            
            total_feature_loss += feature_loss1
            feature_loss_dict.update({f"t1_{k}": v for k, v in feature_loss_dict1.items()})
        
        # Align features with teacher2
        if teacher2_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # New mixed loss function, returns 3 values
                feature_loss2, feature_loss_dict2, detailed_info2 = feature_alignment_loss_fn(
                    teacher2_features, student_features, attention_mask, attention_mask
                )
                detailed_feature_info.update({f"t2_{k}": v for k, v in detailed_info2.items()})
            else:
                # Old loss function, returns 2 values
                feature_loss2, feature_loss_dict2 = feature_alignment_loss_fn(
                    teacher2_features, student_features, attention_mask, attention_mask
                )
                
            total_feature_loss += feature_loss2
            feature_loss_dict.update({f"t2_{k}": v for k, v in feature_loss_dict2.items()})
        
        # If both teachers have features, take average
        if teacher1_features and teacher2_features:
            total_feature_loss = total_feature_loss / 2
    
    # Total loss
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
    
    # 🔥 Include detailed info in return if available
    if detailed_feature_info:
        loss_dict.update(detailed_feature_info)
    
    return total_loss, loss_dict


# 🔥 Convenience creation function
def create_hybrid_text_encoder_alignment_loss(alignment_layers: List[str],
                                             loss_type: str = "hybrid",
                                             loss_weights: Dict[str, float] = None,
                                             **kwargs) -> HybridTextEncoderFeatureAlignmentLoss:
    """
    Convenience function for creating a mixed text encoder feature alignment loss.
    
    Args:
        alignment_layers: List of alignment layers
        loss_type: Loss type - "mse", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
        loss_weights: Dict of layer weights
        **kwargs: Additional arguments
    
    Recommended configurations:
    1. Standard mixed loss: loss_type="hybrid", primary_loss_type="mse", secondary_loss_type="cosine", loss_combination_weight=0.3
    2. Scale-aware: loss_type="scale_aware_cosine", direction_weight=0.8, magnitude_weight=0.2
    3. Layer-adaptive: loss_type="layer_adaptive", use_layer_adaptive=True
    """
    return HybridTextEncoderFeatureAlignmentLoss(
        alignment_layers=alignment_layers,
        loss_type=loss_type,
        loss_weights=loss_weights,
        **kwargs
    )


# 🔥 Usage examples and recommended configurations
def example_hybrid_usage():
    """
    Mixed loss usage examples
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
    
    # 🔥 Option 1: Mixed loss (recommended)
    hybrid_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="hybrid",
        primary_loss_type="mse",           # Primary loss preserves scale information
        secondary_loss_type="cosine",      # Secondary loss focuses on direction alignment
        loss_combination_weight=0.3,       # 30% cosine + 70% mse
        pooling_strategy="mean",
        loss_weights={
            'text_model.encoder.layers.0': 0.5,
            'text_model.encoder.layers.6': 1.0,
            'text_model.encoder.layers.11': 1.5
        }
    )
    
    # 🔥 Option 2: Scale-aware cosine loss
    scale_aware_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="scale_aware_cosine",
        direction_weight=0.8,              # Direction loss weight
        magnitude_weight=0.2,              # Magnitude loss weight
        pooling_strategy="mean"
    )
    
    # 🔥 Option 3: Layer-adaptive loss
    layer_adaptive_loss = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=target_layers,
        loss_type="layer_adaptive",        # Each layer uses a different loss strategy
        use_layer_adaptive=True,
        pooling_strategy="mean"
    )
    
    print("🔥 Mixed loss example configuration complete!")
    print("Option 1: Mixed loss - balances scale and direction")
    print("Option 2: Scale-aware cosine - preserves cosine advantage while considering magnitude")
    print("Option 3: Layer-adaptive - selects the best loss based on layer characteristics")
    
    return hybrid_loss, scale_aware_loss, layer_adaptive_loss


if __name__ == "__main__":
    # Run examples
    example_hybrid_usage()
    
    # Test backward compatibility
    print("\n🔧 Testing backward compatibility...")
    old_style_loss = TextEncoderFeatureAlignmentLoss(
        alignment_layers=['text_model.encoder.layers.0'],
        loss_type="mse"
    )
    print("✅ Backward compatibility test passed!")