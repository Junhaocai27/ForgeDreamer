from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

from typing import Union
import cv2
import numpy as np
import torch

# Forcibly disable PyTorch native FlashAttention and MemEfficientAttention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Force the most basic, most accurate Math computation mode (implicit fallback in older environments)
torch.backends.cuda.enable_math_sdp(True)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
from torch.utils.tensorboard import SummaryWriter  # add TensorBoard import
from .hypergraph_enhancer import StaticGradientHypergraphEnhancer, ImprovedStaticDHGLatentHypergraph, SimplifiedDHGLatentHypergraph, DirectSimilarityDHGLatentHypergraph
from .mask_utils import SubjectMaskGenerator

from .sd_step import *

def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    """
    Modified version: segment objects on white-background images, returns a 0/1 binary mask (no largest-connected-component filtering).
    """
    if verbose:
        print("--- [generate_advanced_mask_hsv] Using HSV to segment objects on white background ---")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("Error: input image must be a 3-channel BGR format.")
        return None

    height, width = original_image.shape[:2]

    # Convert to HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # Step 1: Remove white background (low saturation + high brightness)
    white_bg_mask = cv2.inRange(hsv, (0, 0, 240), (180, 10, 255))
    fg_mask = cv2.bitwise_not(white_bg_mask)  # subject area = non-white region (0/255)

    if verbose:
        white_ratio = np.mean(white_bg_mask > 0)
        print(f"[HSV] White background ratio: {white_ratio*100:.2f}%")

    # Step 2: Edge enhancement (optional)
    edges = cv2.Canny(fg_mask, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

    # Step 3: Morphological processing (noise removal + gap filling)
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if verbose:
        print(f"Morphological processing complete, kernel size: {kernel_size}")

    # Step 4: Directly return the processed foreground mask (no longest connected component)
    final_mask_01 = (fg_mask / 255).astype(np.uint8)

    if verbose: print("0/1 binary mask generation complete (no connected-component filtering).")
    return final_mask_01


def generate_advanced_mask(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    """
    [Core Mask Generator] Segment an object on a solid-color background from a NumPy image array.
    This function performs no file I/O, making it easy to reuse in other code.

    Args:
    original_image (np.ndarray): Input BGR image array (shape H, W, 3), dtype=uint8.
    verbose (bool): Whether to print detailed processing step information.

    Returns:
    np.ndarray | None: 
        - Final single-channel grayscale mask (shape H, W), dtype=uint8 (values 0-255).
        - None if processing fails.
    """
    if verbose:
        print("--- [generate_advanced_mask] Starting advanced segmentation ---")
    
    # Ensure image has 3 channels
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("Error: input image must be a 3-channel BGR format.")
        return None

    height, width = original_image.shape[:2]

    # --- Step 2: Multi-channel analysis to select the best channel ---
    b, g, r = cv2.split(original_image)
    channels = [b, g, r]
    channel_names = ['Blue', 'Green', 'Red']
    variances = [np.var(channel) for channel in channels]
    best_channel_idx = np.argmax(variances)
    best_channel = channels[best_channel_idx]
    if verbose: print(f"Selected {channel_names[best_channel_idx]} channel for segmentation (variance: {variances[best_channel_idx]:.2f})")

    # --- Step 3: Edge detection to assist segmentation ---
    median_val = np.median(best_channel)
    lower_thresh = max(0, int(0.5 * median_val))
    upper_thresh = min(255, int(1.5 * median_val))
    edges = cv2.Canny(best_channel, lower_thresh, upper_thresh)
    if verbose: print(f"Edge detection thresholds: {lower_thresh}-{upper_thresh}")

    # --- Step 4: Improved Otsu binarization ---
    blurred = cv2.GaussianBlur(best_channel, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Step 5: Combine edge information to improve mask ---
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(otsu_mask, dilated_edges)
    
    # --- Step 6: Intelligent morphological processing ---
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if verbose: print(f"Morphological processing with {kernel_size}x{kernel_size} elliptical kernel...")

    # --- Step 7: Connected-component analysis to keep the largest region ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    if num_labels <= 1:
        if verbose: print("No valid connected components found; segmentation may have failed.")
        # If no object found, return an all-black mask or the current result
        return cleaned_mask
        
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(cleaned_mask)
    final_mask[labels == largest_component] = 255
    if verbose: print(f"Keeping largest connected component, area: {stats[largest_component, cv2.CC_STAT_AREA]} pixels")

    # --- Steps 8 & 10 (merged): Boundary refinement and smoothing ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose: print("No contours found in final mask.")
        return final_mask  # return previous mask

    largest_contour = max(contours, key=cv2.contourArea)
    refined_mask = np.zeros_like(final_mask)
    cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Slight dilation to ensure edge completeness, then blur to smooth edges
    kernel_dilate = np.ones((3, 3), np.uint8)  # use a smaller kernel for dilation
    refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)  # final smoothing
    
    if verbose: print("Object boundary refined and smoothed...")
    
    # Return the final high-quality single-channel mask (subject=white 255, background=black 0)
    return refined_mask

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None, use_subject_mask=True, sam_checkpoint_path='/root/LucidDreamer/guidance/sam_vit_b_01ec64.pth',
                 mask_strategy='advanced',
                 mask_on_subject=True):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        # ====================  Hypergraph module initialization  ====================
        self.use_hypergraph = getattr(guidance_opt, 'use_hypergraph', False)  # default: use hypergraph enhancement
        if self.use_hypergraph:
            self.hypergraph_enhancer = StaticGradientHypergraphEnhancer(
                patch_size=getattr(guidance_opt, 'hg_patch_size', 4),
                alpha=getattr(guidance_opt, 'hg_alpha', 0.6),
                similarity_threshold=getattr(guidance_opt, 'hg_sim_thresh', 0.8),
                device=self.device  # pass device info
            )
        else:
            self.hypergraph_enhancer = None
        # ===================================================================

        # ====================  DHG Latent Hypergraph module initialization  ====================
        self.use_dhg_latent_hypergraph = getattr(guidance_opt, 'use_dhg_latent_hypergraph', True)
        if self.use_dhg_latent_hypergraph:
            
            reconstruction_interval = getattr(guidance_opt, 'dhg_reconstruction_interval', 10)
            
            # self.dhg_latent_hypergraph = SimplifiedDHGLatentHypergraph(
            #     device=device,
            #     reconstruction_interval=reconstruction_interval,
            #     top_k=16
            # )

            self.dhg_latent_hypergraph = DirectSimilarityDHGLatentHypergraph(
                device=self.device,
                reconstruction_interval=10,
                similarity_threshold=0.65,
            )

            print(f'[INFO] Improved Static DHG Latent Hypergraph initialized (reconstruction every {reconstruction_interval} steps)')
        else:
            self.dhg_latent_hypergraph = None
        # ===================================================================

        print(f'[INFO] loading stable diffusion...')

        # Add curriculum learning parameters
        self.curriculum_schedule = getattr(guidance_opt, 'curriculum_schedule', 'linear')
        self.curriculum_phases = getattr(guidance_opt, 'curriculum_phases', ['coarse', 'medium', 'fine'])
        self.phase_transitions = getattr(guidance_opt, 'phase_transitions', [1500, 2500, 5000])
        
        # Timestep range definitions
        self.timestep_ranges = {
            'coarse': [0.7, 0.98],     # high noise, learn coarse structure
            'medium': [0.3, 0.8],      # medium noise, learn medium detail
            'fine': [0.02, 0.5],       # low noise, learn fine detail
            'adaptive': [0.02, 0.98]   # adaptive range
        }

        # Add TensorBoard writer
        self.writer = SummaryWriter(log_dir=guidance_opt.tensorboard_log_dir if hasattr(guidance_opt, 'tensorboard_log_dir') else './runs/sd_training_new')

        model_key = guidance_opt.model_key

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # --- Force fallback to the base Attention processor ---
        from diffusers.models.attention_processor import AttnProcessor
        self.unet.set_attn_processor(AttnProcessor())
        
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        

        print(f'[INFO] loaded stable diffusion!')

        # ==================== Mask setup (updated) ====================
        self.use_subject_mask = use_subject_mask
        self.mask_strategy = mask_strategy
        self.mask_on_subject = mask_on_subject
        # self.subject_mask_generator = None  # this generator object is no longer needed
        self.mask_cache = {}

        if self.use_subject_mask:
            # We now default to the 'advanced' strategy; no need to load complex models here
            H = getattr(guidance_opt, 'H', 512)
            W = getattr(guidance_opt, 'W', 512)
            self.latent_height = H // 8
            self.latent_width = W // 8
            print(f"[SD Init] Mask feature enabled, strategy: '{self.mask_strategy}'.")
            print(f"[SD Init] Latent space size for mask: ({self.latent_height}, {self.latent_width}).")
            print(f"[SD Init] Apply mask on subject: {self.mask_on_subject}.")
        else:
            print("[SD Init] Mask feature disabled.")

        print("[INFO] Initializing Three-Phase Annealing Curriculum for timesteps and guidance.")
        
        # --- New curriculum learning parameters ---
        self.total_iterations = getattr(guidance_opt, 'total_iterations', 5000)
        
        transitions_percent = getattr(guidance_opt, 'phase_transitions', [0.2, 0.6])
        self.phase_transitions_iter = [int(p * self.total_iterations) for p in transitions_percent]
        
        self.t_ranges = {
            'coarse': getattr(guidance_opt, 't_range_coarse', [0.7, 0.98]),
            'refine': getattr(guidance_opt, 't_range_refine', [0.4, 0.8]),
            'fine':   getattr(guidance_opt, 't_range_fine', [0.02, 0.5]),
        }
        
        self.guidance_scales = {
            'coarse': getattr(guidance_opt, 'guidance_scale_coarse', [100, 100]),
            'refine': getattr(guidance_opt, 'guidance_scale_refine', [100, 50]),
            'fine':   getattr(guidance_opt, 'guidance_scale_fine', [50, 20]),
        }
        
        # (Optional) loss weights
        self.loss_weights = {
            'coarse': getattr(guidance_opt, 'loss_weight_coarse', [1.0, 1.0]),
            'refine': getattr(guidance_opt, 'loss_weight_refine', [1.0, 0.8]),
            'fine':   getattr(guidance_opt, 'loss_weight_fine', [0.8, 0.5]),
        }

    def get_biased_time_step(self, warm_up_rate):
        """
        Sample ind_t from a non-uniform distribution based on training progress warm_up_rate.
        The proportion of high-noise timesteps increases gradually with warm_up_rate.
        """

        # Timestep range
        min_step = self.min_step
        max_step = self.max_step + int(self.warmup_step * warm_up_rate)
        total_steps = max_step - min_step + 1

        # Introduce more high-noise proportion as warm_up_rate increases (up to 40%)
        high_ratio = min(0.1 + 0.3 * warm_up_rate, 0.4)
        low_ratio = 0.3
        mid_ratio = 1.0 - low_ratio - high_ratio

        # Range for each interval
        low_range = (min_step, min_step + int(total_steps * low_ratio))
        mid_range = (low_range[1], low_range[1] + int(total_steps * mid_ratio))
        high_range = (mid_range[1], max_step + 1)

        # Sample based on ratios
        rand_val = torch.rand(1).item()
        if rand_val < low_ratio:
            chosen_range = low_range
        elif rand_val < low_ratio + mid_ratio:
            chosen_range = mid_range
        else:
            chosen_range = high_range

        # Finally, randomly pick one timestep
        ind_t = torch.randint(
            chosen_range[0], chosen_range[1],
            (1,), dtype=torch.long,
            generator=self.noise_gen,
            device=self.device
        )[0]

        return ind_t
    
    # <<< New helper function >>>
    def get_annealed_params(self, iteration):
        """
        Compute the annealed timestep range and guidance strength based on the current iteration.
        """
        # 1. Determine current phase
        if iteration < self.phase_transitions_iter[0]:
            phase = 'coarse'
            phase_start_iter = 0
            phase_end_iter = self.phase_transitions_iter[0]
        elif iteration < self.phase_transitions_iter[1]:
            phase = 'refine'
            phase_start_iter = self.phase_transitions_iter[0]
            phase_end_iter = self.phase_transitions_iter[1]
        else:
            phase = 'fine'
            phase_start_iter = self.phase_transitions_iter[1]
            phase_end_iter = self.total_iterations

        # 2. Compute progress within the current phase (0.0 to 1.0)
        # Prevent division by zero
        if phase_end_iter == phase_start_iter:
            progress = 1.0
        else:
            progress = (iteration - phase_start_iter) / (phase_end_iter - phase_start_iter)

        # 3. Linear interpolation (lerp) to compute current parameters
        def lerp(start, end, progress):
            return start + progress * (end - start)

        # Get the parameter range for the current phase
        t_range_start, t_range_end = self.t_ranges[phase]
        guidance_start, guidance_end = self.guidance_scales[phase]
        loss_weight_start, loss_weight_end = self.loss_weights[phase]

        # Compute current values
        # Note: for t_range we typically keep it fixed within the phase rather than interpolating.
        # Annealing t_range is also possible, but starting from a fixed range is simpler.
        current_t_range = [t_range_start, t_range_end]
        current_guidance_scale = lerp(guidance_start, guidance_end, progress)
        current_loss_weight = lerp(loss_weight_start, loss_weight_end, progress)

        # 4. Sample a timestep from t_range
        min_step = int(self.num_train_timesteps * current_t_range[0])
        max_step = int(self.num_train_timesteps * current_t_range[1])
        
        # Ensure min_step and max_step are valid
        max_step = max(min_step + 1, max_step)
        
        ind_t = torch.randint(min_step, max_step, (1,), dtype=torch.long, device=self.device)[0]

        return ind_t, current_guidance_scale, current_loss_weight, phase


    def _get_or_generate_latent_mask(self, image_tensor, image_index=None, **mask_kwargs):
        """
        Internal helper method: generate and return a latent-space mask from the input image tensor.
        *** This method has been modified to use our new advanced segmenter ***
        
        Args:
            image_tensor (torch.Tensor): Input image tensor, shape [C, H, W], values in [0, 1].
            image_index (int, optional): Image index (used for caching; unused here).
            **mask_kwargs: No longer used by the new method, but retained for API compatibility.
        """
        # Dynamically compute the target latent size
        target_height, target_width = image_tensor.shape[1], image_tensor.shape[2]
        latent_height = target_height // 8
        latent_width = target_width // 8

        # print(f"[Mask gen] Generating new mask with '{self.mask_strategy}' strategy...")
        
        # 1. Convert tensor to an OpenCV-compatible NumPy array
        # PyTorch Tensor: [C, H, W], RGB, 0-1
        # OpenCV aumPy:   [H, W, C], BGR, 0-255
        image_np_rgb = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR because our function is based on OpenCV
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # 2. <<< Core replacement point >>>
        #    Call our new advanced mask generator instead of FastSAM.
        full_res_mask_np = generate_advanced_mask_hsv(image_np_bgr, verbose=False)  # verbose usually off during integration
        
        # 3. Handle mask generation failure
        if full_res_mask_np is None:
            # print(f"[Mask gen] Warning: advanced mask generation failed. Using all-ones mask.")
            latent_mask = torch.ones((1, 1, latent_height, latent_width), device=self.device, dtype=self.precision_t)
            return latent_mask

        # 4. Convert mask back to tensor and downsample to the correct latent size
        # full_res_mask_np is single-channel (H, W); add batch and channel dims -> (1, 1, H, W)
        mask_tensor = torch.from_numpy(full_res_mask_np).to(self.device).float().unsqueeze(0).unsqueeze(0) / 255.0
        # Downsample to latent space size using bilinear interpolation
        latent_mask = F.interpolate(mask_tensor, size=(latent_height, latent_width), mode='bilinear', align_corners=False)

        # 5. Handle mask inversion logic (e.g., if you want to apply guidance on background)
        if not self.mask_on_subject:
            latent_mask = 1.0 - latent_mask

        # 6. Return the final latent mask
        return latent_mask.to(self.precision_t)
        

    def get_curriculum_timestep(self, iteration, warm_up_rate=0):
        """
        Return an appropriate timestep based on the training phase
        """
        current_phase = self.get_current_phase(iteration)
        
        if current_phase == 'adaptive':
            # Adaptive timestep selection
            return self.get_adaptive_timestep(iteration, warm_up_rate)
        else:
            # Fixed-phase timestep
            t_range = self.timestep_ranges[current_phase]
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            
            ind_t = torch.randint(min_step, max_step + 1, (1,), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
            return ind_t
    
    def get_current_phase(self, iteration):
        """Determine the current training phase"""
        for i, transition in enumerate(self.phase_transitions):
            if iteration < transition:
                return self.curriculum_phases[i]
        return self.curriculum_phases[-1]
    
    def get_adaptive_timestep(self, iteration, warm_up_rate):
        """
        Adaptive timestep selection - based on training progress and gradient history
        """
        # Base progress schedule
        progress = min(iteration / 5000, 1.0)  # assume 5000 steps for full training
        
        # Dynamically adjust timestep range
        if progress < 0.3:
            # Early stage: focus on learning coarse structure
            t_min, t_max = 0.6, 0.98
            weight_high = 0.8  # more high-noise timesteps
        elif progress < 0.7:
            # Mid stage: balanced learning
            t_min, t_max = 0.2, 0.9
            weight_high = 0.5
        else:
            # Late stage: focus on learning fine details
            t_min, t_max = 0.02, 0.6
            weight_high = 0.2  # more low-noise timesteps
        
        # Adjust based on gradient history
        if hasattr(self, 'grad_history') and len(self.grad_history) > 10:
            recent_grad_volatility = np.std(self.grad_history[-10:])
            if recent_grad_volatility > 0.1:  # gradient unstable
                # Use higher noise level to stabilize training
                t_min = max(t_min, 0.4)
                weight_high = min(weight_high + 0.2, 0.9)
        
        # Sample timestep
        min_step = int(self.num_train_timesteps * t_min)
        max_step = int(self.num_train_timesteps * t_max)
        
        # Weighted sampling: adjust distribution based on weight_high
        if torch.rand(1) < weight_high:
            # Sample high-noise timestep
            high_min = int((min_step + max_step) * 0.6)
            ind_t = torch.randint(high_min, max_step + 1, (1,), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            # Sample low-noise timestep
            low_max = int((min_step + max_step) * 0.4)
            ind_t = torch.randint(min_step, low_max + 1, (1,), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        
        return ind_t

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]


    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           image_indices=None,  # <--- receive batch image indices
                           grad_scale=1, use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate=0, weights=0,
                           resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None, opt=None):


        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        if self.use_dhg_latent_hypergraph and self.dhg_latent_hypergraph is not None:
            if iteration < opt.warmup_iter:
                # Phase 1: during warmup, randomly sample in [min_step, max_step + warmup portion]
                upper_bound = self.max_step + int(self.warmup_step * warm_up_rate)
                ind_t = torch.randint(self.min_step, upper_bound + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
            else:
                # Phase 2: linearly decrease max sample range, randomly sample in [min_step, current max]
                decay_iter = iteration - opt.warmup_iter
                decay_total = opt.iterations - opt.warmup_iter

                # Compute current max value (linear decay)
                current_max_step = int(self.min_step + (1.0 - decay_iter / decay_total) * (self.max_step - self.min_step))
                current_max_step = max(self.min_step, current_max_step)

                # Randomly sample in [min_step, current max]
                ind_t = torch.randint(self.min_step, current_max_step + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                           guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]


        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds,\
                                                            weights,\
                                                            B)

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
        # pred_noise = noise_pred_uncond + current_guidance_scale * delta_DSD
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        latent_mask_input = None
        latent_mask_pred = None

        # if self.use_subject_mask and self.subject_mask_generator is not None:
        if self.use_subject_mask is not None:

            # =================== 1. Generate mask for pred_rgb (batch_size=4) ===================
            # Since we always regenerate, image_indices and caching logic are no longer needed
            mask_list_input = []
            # print("[Masking] Generating masks for input images (pred_rgb)...")
            for i in range(pred_rgb.shape[0]):
                # Generate directly without passing image_index
                mask = self._get_or_generate_latent_mask(pred_rgb[i], image_index=None)
                mask_list_input.append(mask)
            latent_mask_input = torch.cat(mask_list_input, dim=0)


            # =================== 2. Generate mask for pred_x0_pos (batch_size=4) ===================
            # This part of the logic remains unchanged as it was already generated in real time
            with torch.no_grad():
                noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy) 
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))

            mask_list_pred = []
            # print("[Masking] Generating masks for SD predicted images (pred_x0_pos)...")
            for i in range(pred_x0_pos.shape[0]):
                mask = self._get_or_generate_latent_mask(pred_x0_pos[i], image_index=None)
                mask_list_pred.append(mask)
            latent_mask_pred = torch.cat(mask_list_pred, dim=0)

        # print(f"[DEBUG] latent_mask_input.shape: {latent_mask_input.shape if latent_mask_input is not None else 'None'}")
        # print(f"[DEBUG] latent_mask_pred.shape: {latent_mask_pred.shape if latent_mask_pred is not None else 'None'}")

        grad = w(self.alphas[t]) * (pred_noise - target)

        # Print the shape of grad here
        # print(f"[DEBUG] grad.shape: {grad.shape}")
        # print(f"[DEBUG] grad.dtype: {grad.dtype}")
        # print(f"[DEBUG] grad.device: {grad.device}")
        # print(f"[DEBUG] grad requires_grad: {grad.requires_grad}")

        # ==================== Apply hypergraph enhancement here ====================
        if self.use_hypergraph and self.hypergraph_enhancer is not None:
            original_grad = grad.clone()  # save original gradient for monitoring
            
            # Call directly, no need to worry about backpropagation
            grad = self.hypergraph_enhancer(grad) 
            
            # Monitoring
            with torch.no_grad():
                original_grad_norm = torch.norm(original_grad).item()
                enhanced_grad_norm = torch.norm(grad).item()
                self.writer.add_scalar('Hypergraph/Enhancement_Ratio', enhanced_grad_norm / (original_grad_norm + 1e-9), iteration)
                
                # Visualize the difference between original and enhanced gradients
                if iteration % (guidance_opt.vis_interval * 5) == 0:
                    grad_diff = torch.abs(grad - original_grad).mean(dim=1, keepdim=True)
                    grad_diff_vis = F.interpolate(grad_diff, size=(512, 512), mode='bilinear')
                    grad_diff_vis = grad_diff_vis / (grad_diff_vis.max() + 1e-8)
                    self.writer.add_images('Hypergraph/Grad_Difference', grad_diff_vis.repeat(1, 3, 1, 1), iteration)
        # ==========================================================

        # ==================== Apply DHG Latent Hypergraph enhancement ====================
        dhg_latent_loss = 0
        if self.use_dhg_latent_hypergraph and self.dhg_latent_hypergraph is not None:
            # print(f"[DEBUG] Applying DHG latent hypergraph enhancement...")
            
            try:
                # Use the improved DHG latent hypergraph processor (pass iteration for reconstruction decision)
                grad_enhancement, dhg_latent_loss = self.dhg_latent_hypergraph(
                    latents,          # original latents [B, 4, 64, 64]
                    (prev_latents_noisy-pred_noise),    # noisy latents [B, 4, 64, 64]
                    iteration,        # current iteration, used to decide whether to reconstruct\
                    mask1=latent_mask_pred,
                    mask2=latent_mask_input,
                )
                
                # print(f"[DEBUG] DHG Latent grad_enhancement.shape: {grad_enhancement.shape}")
                # print(f"[DEBUG] DHG Latent loss: {dhg_latent_loss.item()}")
                
                # ==================== Gradient balancing and fusion ====================
                with torch.no_grad():
                    # 1. Compute L2 norm (magnitude) of both gradients
                    original_grad_norm = torch.norm(grad)
                    enhancement_norm = torch.norm(grad_enhancement)
                    
                    # 2. Compute scale factor to align enhanced gradient norm with original gradient norm
                    # Add a small epsilon to prevent division by zero
                    scale_factor = original_grad_norm / (enhancement_norm + 1e-8)
                    
                    # Get hyperparameter weight
                    dhg_enhancement_weight = getattr(guidance_opt, 'dhg_latent_weight', 0.3)  # recommend a small default

                # 3. Apply scaling and weighting
                # dhg_enhancement_weight controls the strength ratio of the enhanced gradient relative to the original
                # e.g., weight=0.1 means the enhanced gradient contributes 10% of the original gradient's strength
                scaled_enhancement = grad_enhancement * scale_factor * dhg_enhancement_weight

                original_grad = grad.clone()  # for monitoring
                grad = grad + scaled_enhancement
                
                # Fixed-weight fusion
                # dhg_enhancement_weight = getattr(guidance_opt, 'dhg_latent_weight', 0.5)
                # original_grad = grad.clone()
                # grad = grad + dhg_enhancement_weight * grad_enhancement
                
                # Record monitoring information
                with torch.no_grad():
                    enhancement_norm = torch.norm(scaled_enhancement).item()
                    original_norm = torch.norm(original_grad).item()
                    enhanced_norm = torch.norm(grad).item()
                    
                    self.writer.add_scalar('DHG_Latent/EnhancementNorm', enhancement_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/OriginalGradNorm', original_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/EnhancedGradNorm', enhanced_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/Loss', dhg_latent_loss.item(), iteration)
                    self.writer.add_scalar('DHG_Latent/EnhancementWeight', dhg_enhancement_weight, iteration)
                    
                    # Record hypergraph reconstruction information
                    if hasattr(self.dhg_latent_hypergraph, 'last_reconstruction_iter'):
                        steps_since_reconstruction = iteration - self.dhg_latent_hypergraph.last_reconstruction_iter
                        self.writer.add_scalar('DHG_Latent/StepsSinceReconstruction', steps_since_reconstruction, iteration)
                    
                    # Compute enhancement effect statistics
                    enhancement_ratio = enhanced_norm / (original_norm + 1e-9)
                    self.writer.add_scalar('DHG_Latent/EnhancementRatio', enhancement_ratio, iteration)
                    
                    # Visualize enhancement effect
                    if iteration % 100 == 0:
                        # Enhancement visualization
                        enhancement_vis = scaled_enhancement.abs().mean(dim=1, keepdim=True)
                        enhancement_vis = F.interpolate(enhancement_vis, size=(512, 512), mode='bilinear')
                        enhancement_vis = enhancement_vis / (enhancement_vis.max() + 1e-8)
                        self.writer.add_images('DHG_Latent/Enhancement_Visualization', 
                                            enhancement_vis.repeat(1, 3, 1, 1), iteration)
                        
                        # Gradient difference visualization
                        grad_diff = torch.abs(grad - original_grad).mean(dim=1, keepdim=True)
                        grad_diff_vis = F.interpolate(grad_diff, size=(512, 512), mode='bilinear')
                        grad_diff_vis = grad_diff_vis / (grad_diff_vis.max() + 1e-8)
                        self.writer.add_images('DHG_Latent/Grad_Difference', 
                                            grad_diff_vis.repeat(1, 3, 1, 1), iteration)
                        
                        # Comparison of original and enhanced gradients
                        original_grad_vis = original_grad.abs().mean(dim=1, keepdim=True)
                        original_grad_vis = F.interpolate(original_grad_vis, size=(512, 512), mode='bilinear')
                        original_grad_vis = original_grad_vis / (original_grad_vis.max() + 1e-8)
                        
                        enhanced_grad_vis = grad.abs().mean(dim=1, keepdim=True)
                        enhanced_grad_vis = F.interpolate(enhanced_grad_vis, size=(512, 512), mode='bilinear')
                        enhanced_grad_vis = enhanced_grad_vis / (enhanced_grad_vis.max() + 1e-8)
                        
                        self.writer.add_images('DHG_Latent/Original_vs_Enhanced', 
                                            torch.cat([original_grad_vis.repeat(1, 3, 1, 1), 
                                                    enhanced_grad_vis.repeat(1, 3, 1, 1)], dim=0), iteration)
                    
            except Exception as e:
                print(f"[ERROR] DHG Latent enhancement failed: {str(e)}")
                import traceback
                traceback.print_exc()
                dhg_latent_loss = 0

        # More comprehensive gradient monitoring strategy
        with torch.no_grad():
            # === Basic statistics ===
            grad_norm = torch.norm(grad).item()
            grad_mean = torch.mean(grad).item()
            grad_std = torch.std(grad).item()
            grad_max = torch.max(grad).item()
            grad_min = torch.min(grad).item()
            grad_abs_mean = torch.mean(torch.abs(grad)).item()
            
            # === Channel-level analysis ===
            for i in range(grad.shape[1]):  # 4 latent channels
                channel_grad = grad[:, i, :, :]
                channel_norm = torch.norm(channel_grad).item()
                channel_mean = torch.mean(channel_grad).item()
                channel_std = torch.std(channel_grad).item()
                
                self.writer.add_scalar(f'Gradient/Channel_{i}_Norm', channel_norm, iteration)
                self.writer.add_scalar(f'Gradient/Channel_{i}_Mean', channel_mean, iteration)
                self.writer.add_scalar(f'Gradient/Channel_{i}_Std', channel_std, iteration)
            
            # === Spatial analysis ===
            # Compute gradient intensity in different regions
            h, w = grad.shape[2], grad.shape[3]
            center_grad = grad[:, :, h//4:3*h//4, w//4:3*w//4]
            edge_grad = grad.clone()
            edge_grad[:, :, h//4:3*h//4, w//4:3*w//4] = 0
            
            center_norm = torch.norm(center_grad).item()
            edge_norm = torch.norm(edge_grad).item()
            
            self.writer.add_scalar('Gradient/Center_Norm', center_norm, iteration)
            self.writer.add_scalar('Gradient/Edge_Norm', edge_norm, iteration)
            self.writer.add_scalar('Gradient/Center_Edge_Ratio', center_norm / (edge_norm + 1e-8), iteration)
            
            # === Gradient change trend ===
            # Save historical gradient norm for trend analysis
            if not hasattr(self, 'grad_history'):
                self.grad_history = []
            self.grad_history.append(grad_norm)
            
            # Keep the most recent 100 steps of history
            if len(self.grad_history) > 100:
                self.grad_history.pop(0)
            
            # Compute gradient change trend
            if len(self.grad_history) >= 10:
                recent_grads = self.grad_history[-10:]
                grad_trend = (recent_grads[-1] - recent_grads[0]) / 10
                grad_volatility = np.std(recent_grads)
                
                self.writer.add_scalar('Gradient/Trend', grad_trend, iteration)
                self.writer.add_scalar('Gradient/Volatility', grad_volatility, iteration)
            
            # === Analysis relative to different baselines ===
            # Comparison with initial noise
            noise_norm = torch.norm(noise).item() if 'noise' in locals() else 0
            pred_noise_norm = torch.norm(pred_noise).item()
            target_norm = torch.norm(target).item()
            
            self.writer.add_scalar('Gradient/Grad_vs_Noise_Ratio', grad_norm / (noise_norm + 1e-8), iteration)
            self.writer.add_scalar('Gradient/Grad_vs_PredNoise_Ratio', grad_norm / (pred_noise_norm + 1e-8), iteration)
            self.writer.add_scalar('Gradient/Grad_vs_Target_Ratio', grad_norm / (target_norm + 1e-8), iteration)
            
            # === Gradient health indicators ===
            # Detect gradient anomalies
            grad_has_nan = torch.isnan(grad).any().item()
            grad_has_inf = torch.isinf(grad).any().item()
            grad_zero_ratio = (grad == 0).float().mean().item()
            grad_positive_ratio = (grad > 0).float().mean().item()
            
            self.writer.add_scalar('Gradient/Has_NaN', float(grad_has_nan), iteration)
            self.writer.add_scalar('Gradient/Has_Inf', float(grad_has_inf), iteration)
            self.writer.add_scalar('Gradient/Zero_Ratio', grad_zero_ratio, iteration)
            self.writer.add_scalar('Gradient/Positive_Ratio', grad_positive_ratio, iteration)
            
            # === Multi-scale analysis ===
            # Gradient intensity at different scales
            grad_pooled_2x2 = F.avg_pool2d(grad.abs(), 2)
            grad_pooled_4x4 = F.avg_pool2d(grad.abs(), 4)
            
            fine_scale_norm = torch.norm(grad).item()
            medium_scale_norm = torch.norm(grad_pooled_2x2).item()
            coarse_scale_norm = torch.norm(grad_pooled_4x4).item()
            
            self.writer.add_scalar('Gradient/Fine_Scale_Norm', fine_scale_norm, iteration)
            self.writer.add_scalar('Gradient/Medium_Scale_Norm', medium_scale_norm, iteration)
            self.writer.add_scalar('Gradient/Coarse_Scale_Norm', coarse_scale_norm, iteration)
            
            # === Basic logging ===
            self.writer.add_scalar('Gradient/Norm', grad_norm, iteration)
            self.writer.add_scalar('Gradient/Mean', grad_mean, iteration)
            self.writer.add_scalar('Gradient/Std', grad_std, iteration)
            self.writer.add_scalar('Gradient/Max', grad_max, iteration)
            self.writer.add_scalar('Gradient/Min', grad_min, iteration)
            self.writer.add_scalar('Gradient/AbsMean', grad_abs_mean, iteration)
            
            # === Distribution logging ===
            self.writer.add_histogram('Gradient/Distribution', grad.flatten(), iteration)
            
            # Log distribution per channel
            for i in range(grad.shape[1]):
                self.writer.add_histogram(f'Gradient/Channel_{i}_Distribution', 
                                        grad[:, i, :, :].flatten(), iteration)
            
            # === Training-related information ===
            self.writer.add_scalar('Training/Timestep', t.item(), iteration)
            self.writer.add_scalar('Training/GuidanceScale', guidance_opt.guidance_scale, iteration)
            self.writer.add_scalar('Training/WarmupRate', warm_up_rate, iteration)
            self.writer.add_scalar('Training/CurrentDeltaT', current_delta_t, iteration)
            
            # === Periodically save gradient visualizations ===
            if iteration % (guidance_opt.vis_interval * 5) == 0:
                # Save spatial distribution map of gradients
                grad_vis = grad.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
                grad_vis = F.interpolate(grad_vis, size=(512, 512), mode='bilinear')
                grad_vis = grad_vis / (grad_vis.max() + 1e-8)  # normalize
                
                self.writer.add_images('Gradient/Spatial_Distribution', 
                                     grad_vis.repeat(1, 3, 1, 1), iteration)
        
        grad = torch.nan_to_num(grad_scale * grad)
        # grad = torch.nan_to_num(current_loss_weight * grad)
        loss = SpecifyGradient.apply(latents, grad)

        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb, 
                                        norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)

        return loss


    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0,
                    resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        
        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                           guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]


        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond
        
        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)     

        grad = w(self.alphas[t]) * (pred_noise - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + 7.5* delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                # pred_x0_uncond = pred_x0_sp[:1, ...]

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb, norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)

        return loss

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence
