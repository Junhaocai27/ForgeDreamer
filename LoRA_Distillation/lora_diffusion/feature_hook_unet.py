import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import inspect

class FeatureHook:
    """Hook class for capturing intermediate layer features."""
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach().clone()
        return fn
    
    def register_hooks(self, model, layer_names):
        """Register a hook for the specified layer."""
        self.clear_hooks()
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """Clear all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def get_features(self):
        return self.features

class UNetFeatureExtractor:
    def __init__(self, target_layers, mixed_precision_config=None):
        """
        Initialize UNetFeatureExtractor.
        Args:
            target_layers (list of str): List of target layer names from which to extract features.
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
        """
        feature_to_store = output
        if isinstance(output, tuple):
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                feature_to_store = output[0]
        self.features[layer_name] = feature_to_store

    def register_hooks(self, model):
        """
        Register forward hooks on target layers of the specified model.
        Args:
            model (torch.nn.Module): The UNet model on which to register hooks.
        """
        self.clear_hooks()
        self.features.clear()

        if not self.target_layers:
            return

        for name, module in model.named_modules():
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
        return {k: v for k, v in self.features.items()}

    def clear_hooks(self):
        """
        Remove all registered hooks and clear the list.
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
                         use_grad: bool = False,
                         **other_unet_specific_kwargs):
        """
        Run the UNet forward pass and extract intermediate features from specified layers.
        Args:
            use_grad (bool): If True, run the UNet within torch.enable_grad() context,
                             allowing differentiable feature extraction for the student model. Default is False (runs in no_grad()).
        """
        # 1. Register hooks
        self.register_hooks(unet_model)

        # 2. Determine the actual parameter dtype of the UNet
        try:
            unet_internal_dtype = next(unet_model.parameters()).dtype
        except StopIteration:
            try:
                unet_internal_dtype = next(unet_model.buffers()).dtype
            except StopIteration:
                unet_internal_dtype = sample.dtype

        # 3. Prepare input tensor dtypes
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

        # 4. Run forward pass
        main_output = None
        context_manager = torch.enable_grad() if use_grad else torch.no_grad()

        with context_manager:
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

                # 5. Get the main prediction from the UNet output
                if return_dict:
                    if not hasattr(output_data, "sample"):
                        raise ValueError("UNet returned a dictionary but it does not have a 'sample' attribute.")
                    main_output = output_data.sample
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


class HybridUNetFeatureAlignmentLoss(nn.Module):
    """
    🔥 UNet feature alignment loss module with mixed loss support
    """
    def __init__(self, 
                 alignment_layers: List[str], 
                 loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, 
                 loss_type: str = "hybrid",
                 feature_selection_strategy: str = "adaptive",
                 normalize_features: bool = True,
                 channel_alignment: str = "projection",
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
        Initialize the mixed UNet feature alignment loss module.
        Args:
            alignment_layers (List[str]): List of layer names whose features should be aligned.
            loss_weights (Dict[str, float], optional): Per-layer loss weights.
            temperature (float): Temperature used for scaling features.
            loss_type (str): Loss type - "mse", "l1", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
            feature_selection_strategy (str): Feature selection strategy
            normalize_features (bool): Whether to normalize features
            channel_alignment (str): Channel alignment method
            
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
        self.loss_type = loss_type.lower()
        self.feature_selection_strategy = feature_selection_strategy
        self.normalize_features = normalize_features
        self.channel_alignment = channel_alignment
        
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
        
        # Projection layers for channel alignment
        self.channel_projectors = nn.ModuleDict()
        
        print(f"🔥 HybridUNetFeatureAlignmentLoss initialized:")
        print(f"   - Loss type: {loss_type}")
        print(f"   - Feature selection: {feature_selection_strategy}")
        print(f"   - Normalize features: {normalize_features}")
        print(f"   - Channel alignment: {channel_alignment}")
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
            'conv_in': {'type': 'mse', 'weight': 0.8},
            'down_blocks.0': {'type': 'mse', 'weight': 1.0},
            'down_blocks.1': {'type': 'hybrid', 'weight': 1.2},
            'down_blocks.2': {'type': 'hybrid', 'weight': 1.5},
            'down_blocks.3': {'type': 'cosine', 'weight': 2.0},
            'mid_block': {'type': 'cosine', 'weight': 2.5},
            'up_blocks.0': {'type': 'cosine', 'weight': 2.0},
            'up_blocks.1': {'type': 'hybrid', 'weight': 1.5},
            'up_blocks.2': {'type': 'hybrid', 'weight': 1.2},
            'up_blocks.3': {'type': 'mse', 'weight': 1.0},
            'conv_out': {'type': 'mse', 'weight': 0.8},
        }

    def _extract_tensor_from_tuple(self, 
                                   feature: Union[torch.Tensor, Tuple], 
                                   layer_name: str) -> torch.Tensor:
        """
        Intelligently extract tensor features from a tuple.
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
            # Select the tensor with the most information (usually the largest)
            return max(tensors, key=lambda x: x.numel())
        elif self.feature_selection_strategy == "attention":
            # Use attention weights to combine multiple tensors
            return self._attention_combine_tensors(tensors, layer_name)
        else:
            return tensors[0]  # Default: take the first one

    def _attention_combine_tensors(self, 
                                   tensors: List[torch.Tensor], 
                                   layer_name: str) -> torch.Tensor:
        """
        Combine multiple tensors using an attention mechanism.
        """
        if len(tensors) == 1:
            return tensors[0]
        
        # Reshape all tensors to the same spatial dimensions
        reference_shape = tensors[0].shape
        aligned_tensors = []
        
        for tensor in tensors:
            if tensor.shape != reference_shape:
                # Simple adaptive pooling alignment
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
        
        # Compute attention weights
        attention_weights = []
        for tensor in aligned_tensors:
            # Use tensor variance as importance weight
            weight = torch.var(tensor, dim=list(range(1, tensor.ndim)), keepdim=True)
            attention_weights.append(weight.mean())
        
        # Normalize weights
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            attention_weights = [1.0 / len(tensors)] * len(tensors)
        
        # Weighted combination
        combined = sum(w * t for w, t in zip(attention_weights, aligned_tensors))
        return combined

    def _align_channel_dimensions(self, 
                                  teacher_feat: torch.Tensor, 
                                  student_feat: torch.Tensor, 
                                  layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align channel dimensions.
        """
        if teacher_feat.shape[1] == student_feat.shape[1]:
            return teacher_feat, student_feat
        
        if self.channel_alignment == "none":
            # Truncate to the smaller channel count
            min_channels = min(teacher_feat.shape[1], student_feat.shape[1])
            return teacher_feat[:, :min_channels], student_feat[:, :min_channels]
        
        elif self.channel_alignment == "interpolation":
            # Use interpolation to adjust channel count
            target_channels = teacher_feat.shape[1]
            if student_feat.shape[1] != target_channels:
                # Adjust channel count via 1x1 convolution
                projector_key = f"{layer_name}_conv"
                if projector_key not in self.channel_projectors:
                    self.channel_projectors[projector_key] = nn.Conv2d(
                        student_feat.shape[1], target_channels, 1, bias=False
                    ).to(student_feat.device)
                
                if student_feat.ndim == 4:
                    student_feat = self.channel_projectors[projector_key](student_feat)
                else:
                    # For 3D tensors, spatial dimensions need to be added
                    original_shape = student_feat.shape
                    reshaped = student_feat.view(original_shape[0], original_shape[1], 1, -1)
                    projected = self.channel_projectors[projector_key](reshaped)
                    student_feat = projected.view(original_shape[0], target_channels, -1)
        
        elif self.channel_alignment == "projection":
            # Use linear projection for alignment
            target_channels = teacher_feat.shape[1]
            if student_feat.shape[1] != target_channels:
                projector_key = f"{layer_name}_proj"
                if projector_key not in self.channel_projectors:
                    self.channel_projectors[projector_key] = nn.Linear(
                        student_feat.shape[1], target_channels, bias=False
                    ).to(student_feat.device)
                
                # Reshape and project
                original_shape = student_feat.shape
                flattened = student_feat.view(-1, original_shape[1])
                projected = self.channel_projectors[projector_key](flattened)
                student_feat = projected.view(original_shape[0], target_channels, *original_shape[2:])
        
        return teacher_feat, student_feat

    def _adaptive_spatial_alignment(self, 
                                    teacher_feat: torch.Tensor, 
                                    student_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Improved spatial dimension alignment.
        """
        if teacher_feat.shape[2:] == student_feat.shape[2:]:
            return teacher_feat, student_feat
        
        # Align to teacher's spatial dimensions
        target_shape = teacher_feat.shape[2:]
        
        if student_feat.ndim == 4:  # (B, C, H, W)
            student_feat_aligned = F.adaptive_avg_pool2d(student_feat, target_shape)
        elif student_feat.ndim == 3:  # (B, L, D) or (B, D, L)
            if len(target_shape) == 1:  # Sequence length alignment
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
        Feature normalization.
        """
        if not self.normalize_features:
            return teacher_feat, student_feat
        
        # L2 normalization
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        student_norm = F.normalize(student_feat, p=2, dim=1)
        
        return teacher_norm, student_norm

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
            
            # Flatten features
            teacher_flat = teacher_feat.view(teacher_feat.shape[0], -1)
            student_flat = student_feat.view(student_feat.shape[0], -1)
            
            return self.cosine_criterion(student_flat, teacher_flat, target)
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
        """�� Compute layer-adaptive loss."""
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
        # Ensure consistent data types
        teacher_feat = teacher_feat.float()
        student_feat = student_feat.float()
        
        # Check numerical stability
        if torch.isnan(teacher_feat).any() or torch.isnan(student_feat).any():
            print(f"Warning: NaN detected in features for layer {layer_name}")
            return torch.tensor(0.0, device=teacher_feat.device), {}
        
        if torch.isinf(teacher_feat).any() or torch.isinf(student_feat).any():
            print(f"Warning: Inf detected in features for layer {layer_name}")
            return torch.tensor(0.0, device=teacher_feat.device), {}
        
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

    def _adaptive_pool_features(self, feat1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                  feat2: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        """
        Adaptively align two feature tensors (preserving original logic).
        """
        # Handle possible tuple output
        if isinstance(feat1, tuple):
            original_feat1_tuple = feat1
            found_tensor_in_feat1 = False
            if len(feat1) > 0:
                for i, el in enumerate(feat1):
                    if isinstance(el, torch.Tensor):
                        feat1 = el
                        found_tensor_in_feat1 = True
                        break
            if not found_tensor_in_feat1:
                raise TypeError(f"CRITICAL: feat1 is a tuple and no tensor found within it. Value: {str(original_feat1_tuple)[:200]}")

        if isinstance(feat2, tuple):
            original_feat2_tuple = feat2
            found_tensor_in_feat2 = False
            if len(feat2) > 0:
                for i, el in enumerate(feat2):
                    if isinstance(el, torch.Tensor):
                        feat2 = el
                        found_tensor_in_feat2 = True
                        break
            if not found_tensor_in_feat2:
                raise TypeError(f"CRITICAL: feat2 is a tuple and no tensor found within it. Value: {str(original_feat2_tuple)[:200]}")

        if not isinstance(feat1, torch.Tensor):
            raise TypeError(f"CRITICAL: feat1 is not a torch.Tensor after tuple processing. Type: {type(feat1)}. Value: {str(feat1)[:200]}")
        if not isinstance(feat2, torch.Tensor):
            raise TypeError(f"CRITICAL: feat2 is not a torch.Tensor after tuple processing. Type: {type(feat2)}. Value: {str(feat2)[:200]}")
        
        # Pooling logic
        if feat1.ndim >=3 and feat2.ndim >=3 and feat1.shape[2:] != feat2.shape[2:]:
            if feat1.ndim == 4 and feat2.ndim == 4: # (B, C, H, W)
                pool_h, pool_w = feat2.shape[2], feat2.shape[3]
                feat1_pooled = F.adaptive_avg_pool2d(feat1, (pool_h, pool_w))
                return feat1_pooled, feat2
            elif feat1.ndim == 3 and feat2.ndim == 3: # (B, L, D) -> (B, D, L) for pool1d
                 if feat1.shape[1] != feat2.shape[1]: # Compare L
                    target_seq_len = feat2.shape[1]
                    feat1_permuted = feat1.permute(0, 2, 1)
                    feat1_pooled_permuted = F.adaptive_avg_pool1d(feat1_permuted, target_seq_len)
                    feat1_pooled = feat1_pooled_permuted.permute(0, 2, 1)
                    return feat1_pooled, feat2
        return feat1, feat2

    def forward(self, 
                teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        🔥 Compute mixed feature alignment loss.
        """
        device = None
        # Determine device
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
        detailed_loss_info = {}
        active_layers_count = 0
        
        for layer_name in self.alignment_layers:
            try:
                # Check whether features exist
                if (layer_name not in teacher_features or 
                    layer_name not in student_features or
                    teacher_features[layer_name] is None or 
                    student_features[layer_name] is None):
                    continue
                
                t_feat_orig = teacher_features[layer_name]
                s_feat_orig = student_features[layer_name]

                if not (isinstance(t_feat_orig, torch.Tensor) or isinstance(t_feat_orig, tuple)) or \
                   not (isinstance(s_feat_orig, torch.Tensor) or isinstance(s_feat_orig, tuple)):
                    print(f"Warning: Features for layer {layer_name} are not Tensors or Tuples. Skipping.")
                    continue
                
                # Use original feature alignment logic
                try:
                    teacher_feat_processed, student_feat_processed = self._adaptive_pool_features(
                        t_feat_orig, s_feat_orig
                    )
                except TypeError as e:
                    print(f"Error in _adaptive_pool_features for layer {layer_name}: {e}. Skipping layer.")
                    continue

                if teacher_feat_processed is None or student_feat_processed is None:
                    continue

                # 🔥 Apply enhanced feature alignment pipeline
                # 1. Channel dimension alignment
                teacher_feat_processed, student_feat_processed = self._align_channel_dimensions(
                    teacher_feat_processed, student_feat_processed, layer_name
                )
                
                # 2. Spatial dimension alignment
                teacher_feat_processed, student_feat_processed = self._adaptive_spatial_alignment(
                    teacher_feat_processed, student_feat_processed
                )
                
                # 3. Feature normalization
                teacher_feat_processed, student_feat_processed = self._normalize_features(
                    teacher_feat_processed, student_feat_processed
                )
                
                # 4. Apply temperature scaling
                if self.temperature != 1.0:
                    student_feat_processed = student_feat_processed / self.temperature
                    teacher_feat_processed = teacher_feat_processed / self.temperature

                # 🔥 5. Compute mixed layer loss
                layer_loss, layer_loss_details = self._compute_layer_loss(
                    student_feat_processed, teacher_feat_processed, layer_name
                )
                
                # 6. Apply global layer weight
                global_weight = self.loss_weights.get(layer_name, 1.0)
                weighted_layer_loss = global_weight * layer_loss
                
                total_loss += weighted_layer_loss
                loss_dict[layer_name] = weighted_layer_loss.item()
                
                # Save detailed loss info
                detailed_loss_info.update(layer_loss_details)
                detailed_loss_info[f'{layer_name}_global_weight'] = global_weight
                detailed_loss_info[f'{layer_name}_final_weighted'] = weighted_layer_loss.item()
                
                active_layers_count += 1
                
            except Exception as e:
                print(f"Error processing layer {layer_name}: {e}")
                continue
        
        # Handle the case where no active layers are present
        if active_layers_count == 0:
            print("Warning: No active layers for feature alignment loss calculation.")
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True), loss_dict, detailed_loss_info
        
        # 🔥 Add loss type statistics
        detailed_loss_info['total_loss'] = total_loss.item()
        detailed_loss_info['active_layers'] = active_layers_count
        detailed_loss_info['loss_type'] = self.loss_type
        
        return total_loss, loss_dict, detailed_loss_info


# �� Original classes kept for backward compatibility (simple wrappers)
class FeatureAlignmentLoss(HybridUNetFeatureAlignmentLoss):
    """Original FeatureAlignmentLoss, now inherits from mixed version for backward compatibility."""
    def __init__(self, alignment_layers: List[str], loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, loss_type: str = "mse"):
        # 🔥 Call mixed version initializer with single-loss configuration
        super().__init__(
            alignment_layers=alignment_layers,
            loss_weights=loss_weights,
            temperature=temperature,
            loss_type=loss_type,
            feature_selection_strategy="adaptive",
            normalize_features=True,
            channel_alignment="projection",
            # Default params to preserve original behavior
            primary_loss_type="mse",
            secondary_loss_type="cosine",
            loss_combination_weight=0.0,  # Do not use mixing
            enable_magnitude_preservation=False,
            use_layer_adaptive=False
        )
    
    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor]):
        """Forward pass preserving original return format."""
        total_loss, loss_dict, detailed_loss_info = super().forward(
            teacher_features, student_features
        )
        # Return only the first two values in original format
        return total_loss, loss_dict


class EnhancedFeatureAlignmentLoss(HybridUNetFeatureAlignmentLoss):
    """Enhanced feature alignment loss, now inherits from mixed version."""
    def __init__(self, 
                 alignment_layers: List[str], 
                 loss_weights: Dict[str, float] = None, 
                 temperature: float = 1.0, 
                 loss_type: str = "mse",
                 feature_selection_strategy: str = "adaptive",
                 normalize_features: bool = True,
                 channel_alignment: str = "projection"):
        # Call mixed version initializer
        super().__init__(
            alignment_layers=alignment_layers,
            loss_weights=loss_weights,
            temperature=temperature,
            loss_type=loss_type,
            feature_selection_strategy=feature_selection_strategy,
            normalize_features=normalize_features,
            channel_alignment=channel_alignment,
            # Default mixed parameters for enhanced version
            primary_loss_type="mse",
            secondary_loss_type="cosine",
            loss_combination_weight=0.0 if loss_type != "hybrid" else 0.3,
            enable_magnitude_preservation=True,
            use_layer_adaptive=False
        )
    
    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor]):
        """Maintain the enhanced version return format."""
        total_loss, loss_dict, detailed_loss_info = super().forward(
            teacher_features, student_features
        )
        # Return first two values to maintain compatibility
        return total_loss, loss_dict


def modified_forward_with_features(unet, sample, timestep, encoder_hidden_states, **kwargs):
    """Modified UNet forward pass that also returns intermediate features."""
    features = {}
    
    # Original UNet forward pass logic, but saves features at key points
    # This needs to be adapted based on the specific UNet implementation
    
    # Example: assuming we can access the internal UNet structure
    sample = unet.conv_in(sample)
    features['conv_in'] = sample.clone()
    
    # Downsampling
    down_block_res_samples = []
    for i, downsample_block in enumerate(unet.down_blocks):
        sample, res_samples = downsample_block(sample, timestep, encoder_hidden_states)
        down_block_res_samples.extend(res_samples)
        features[f'down_blocks.{i}'] = sample.clone()
    
    # Middle block
    sample = unet.mid_block(sample, timestep, encoder_hidden_states)
    features['mid_block'] = sample.clone()
    
    # Upsampling
    for i, upsample_block in enumerate(unet.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
        sample = upsample_block(sample, res_samples, timestep, encoder_hidden_states)
        features[f'up_blocks.{i}'] = sample.clone()
    
    # Output
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)
    
    return sample, features


def compute_distillation_loss_with_hybrid_features(
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
    """🔥 Compute distillation loss with mixed feature alignment."""
    
    # Teacher 1 forward pass
    with torch.no_grad():
        teacher1_noise_pred, teacher1_features = modified_forward_with_features(
            teacher1_unet, latents, timesteps, encoder_hidden_states1
        )
    
    # Teacher 2 forward pass
    with torch.no_grad():
        teacher2_noise_pred, teacher2_features = modified_forward_with_features(
            teacher2_unet, latents, timesteps, encoder_hidden_states2
        )
    
    # Student forward pass
    student_noise_pred, student_features = modified_forward_with_features(
        student_unet, latents, timesteps, encoder_hidden_states1
    )
    
    # Compute noise prediction loss
    target_noise = (teacher1_noise_pred + teacher2_noise_pred) / 2
    noise_pred_loss = F.mse_loss(student_noise_pred, target_noise)
    
    # 🔥 Compute mixed feature alignment loss
    total_feature_loss = torch.tensor(0.0, device=noise_pred_loss.device)
    feature_loss_dict = {}
    detailed_feature_info = {}
    
    if feature_alignment_loss_fn is not None and (teacher1_features or teacher2_features):
        # Align features with teacher1
        if teacher1_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # New mixed loss function, returns 3 values
                feature_loss1, feature_loss_dict1, detailed_info1 = feature_alignment_loss_fn(
                    teacher1_features, student_features
                )
                detailed_feature_info.update({f"t1_{k}": v for k, v in detailed_info1.items()})
            else:
                # Old loss function, returns 2 values
                feature_loss1, feature_loss_dict1 = feature_alignment_loss_fn(
                    teacher1_features, student_features
                )
            
            total_feature_loss += feature_loss1
            feature_loss_dict.update({f"t1_{k}": v for k, v in feature_loss_dict1.items()})
        
        # Align features with teacher2
        if teacher2_features:
            if hasattr(feature_alignment_loss_fn, 'forward') and len(inspect.signature(feature_alignment_loss_fn.forward).parameters) > 4:
                # New mixed loss function, returns 3 values
                feature_loss2, feature_loss_dict2, detailed_info2 = feature_alignment_loss_fn(
                    teacher2_features, student_features
                )
                detailed_feature_info.update({f"t2_{k}": v for k, v in detailed_info2.items()})
            else:
                # Old loss function, returns 2 values
                feature_loss2, feature_loss_dict2 = feature_alignment_loss_fn(
                    teacher2_features, student_features
                )
                
            total_feature_loss += feature_loss2
            feature_loss_dict.update({f"t2_{k}": v for k, v in feature_loss_dict2.items()})
        
        # If both teachers have features, take average
        if teacher1_features and teacher2_features:
            total_feature_loss = total_feature_loss / 2
    
    # Total loss
    total_loss = (
        noise_pred_weight * noise_pred_loss + 
        feature_align_weight * total_feature_loss
    )
    
    loss_dict = {
        'noise_pred_loss': noise_pred_loss.item(),
        'feature_align_loss': total_feature_loss.item(),
        'total_loss': total_loss.item(),
        **feature_loss_dict
    }
    
    # 🔥 Include detailed info in return if available
    if detailed_feature_info:
        loss_dict.update(detailed_feature_info)
    
    return total_loss, loss_dict


# 🔥 Convenience creation functions
def create_hybrid_unet_alignment_loss(alignment_layers: List[str],
                                     loss_type: str = "hybrid",
                                     loss_weights: Dict[str, float] = None,
                                     **kwargs) -> HybridUNetFeatureAlignmentLoss:
    """
    Convenience function for creating a mixed UNet feature alignment loss.
    
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
    return HybridUNetFeatureAlignmentLoss(
        alignment_layers=alignment_layers,
        loss_type=loss_type,
        loss_weights=loss_weights,
        **kwargs
    )


def create_enhanced_feature_alignment_loss(
    alignment_layers: List[str],
    loss_weights: Dict[str, float] = None,
    **kwargs
) -> EnhancedFeatureAlignmentLoss:
    """
    Create an enhanced feature alignment loss function (maintains compatibility).
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


# 🔥 Usage examples and recommended configurations
def example_hybrid_unet_usage():
    """
    Mixed UNet loss usage examples.
    """
    target_layers = [
        'conv_in',
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3',
        'conv_out'
    ]
    
    # 🔥 Option 1: Mixed loss (recommended)
    hybrid_loss = create_hybrid_unet_alignment_loss(
        alignment_layers=target_layers,
        loss_type="hybrid",
        primary_loss_type="mse",           # Primary loss preserves scale information
        secondary_loss_type="cosine",      # Secondary loss focuses on direction alignment
        loss_combination_weight=0.3,       # 30% cosine + 70% mse
        feature_selection_strategy="adaptive",
        normalize_features=True,
        channel_alignment="projection",
        loss_weights={
            'conv_in': 0.5,
            'down_blocks.0': 1.0,
            'down_blocks.1': 1.2,
            'mid_block': 2.0,
            'up_blocks.0': 1.8,
            'up_blocks.3': 1.0,
            'conv_out': 0.8
        }
    )
    
    # 🔥 Option 2: Scale-aware cosine loss
    scale_aware_loss = create_hybrid_unet_alignment_loss(
        alignment_layers=target_layers,
        loss_type="scale_aware_cosine",
        direction_weight=0.8,              # Direction loss weight
        magnitude_weight=0.2,              # Magnitude loss weight
        feature_selection_strategy="adaptive",
        normalize_features=True,
        channel_alignment="projection"
    )
    
    # 🔥 Option 3: Layer-adaptive loss
    layer_adaptive_loss = create_hybrid_unet_alignment_loss(
        alignment_layers=target_layers,
        loss_type="layer_adaptive",        # Each layer uses a different loss strategy
        use_layer_adaptive=True,
        feature_selection_strategy="adaptive",
        normalize_features=True,
        channel_alignment="projection"
    )
    
    print("🔥 Mixed UNet loss example configuration complete!")
    print("Option 1: Mixed loss - balances scale and direction")
    print("Option 2: Scale-aware cosine - preserves cosine advantage while considering magnitude")
    print("Option 3: Layer-adaptive - selects the best loss based on layer characteristics")
    
    return hybrid_loss, scale_aware_loss, layer_adaptive_loss


if __name__ == "__main__":
    # Run examples
    example_hybrid_unet_usage()
    
    # Test backward compatibility
    print("\n🔧 Testing backward compatibility...")
    old_style_loss = FeatureAlignmentLoss(
        alignment_layers=['down_blocks.0'],
        loss_type="mse"
    )
    print("✅ Backward compatibility test passed!")