import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional, List, Literal

import warnings
warnings.filterwarnings("ignore")  # Disable all warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
import fire
import sys
from torchvision.utils import save_image
from diffusers import EulerAncestralDiscreteScheduler
from torch.utils.tensorboard import SummaryWriter
from feature_hook_unet import FeatureAlignmentLoss, UNetFeatureExtractor, create_enhanced_feature_alignment_loss, create_hybrid_unet_alignment_loss
from feature_hook_text_encoder import TextEncoderFeatureExtractor, TextEncoderFeatureAlignmentLoss, create_hybrid_text_encoder_alignment_loss
from dynamic_weight import create_inversion_weight_adjuster, create_tuning_weight_adjuster

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from lora_diffusion import (
    PivotalTuningDatasetCapation,
    PivotalTuningDatasetCapationPromptOnly,
    PivotalTuningDatasetCapationLoraGenerated,
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    inspect_lora,
    save_lora_weight,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
    UNET_EXTENDED_TARGET_REPLACE,
    tune_lora_scale, 
    patch_pipe,
)

OBJECT_TEMPLATE = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

STYLE_TEMPLATE = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def text2img_dataloader_combined_with_latent_caching(
    train_dataset,
    train_batch_size,
    main_tokenizer,
    vae,                # VAE model for encoding images to latents if cached_latents is True
    aux_tokenizer1=None,
    aux_tokenizer2=None,
    cached_latents: bool = False,
    num_workers: int = 0,
    # vae_scaling_factor: float = 0.18215 # Can be passed as a parameter, or obtained from vae.config
):
    """Creates a DataLoader with optional latent caching."""

    dataset_to_load = train_dataset

    if cached_latents:
        if vae is None:
            raise ValueError("VAE must be provided if cached_latents is True.")
        
        print(f"Caching latents for {len(train_dataset)} images (direct imitation)...")
        
        # Use the device where the VAE resides for operations
        current_vae_device = vae.device 
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

        cached_dataset_items = [] # Store modified dataset items

        for i in tqdm(range(len(train_dataset)), desc="Caching latents (direct imitation)"):
            item_from_dataset = train_dataset[i] 
            
            if "instance_images" not in item_from_dataset:
                raise ValueError(f"Dataset item at index {i} did not return 'instance_images'.")

            pixel_values = item_from_dataset["instance_images"] 

            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError(
                    f"Expected 'instance_images' to be a torch.Tensor for direct imitation, got {type(pixel_values)}. "
                    "If dataset returns PIL, pre-processing is needed or use the 'fixed' version."
                )
            
            input_pixels_for_vae = pixel_values.unsqueeze(0).to(device=current_vae_device, dtype=vae.dtype)

            with torch.no_grad():
                latents_on_vae_device = vae.encode(input_pixels_for_vae).latent_dist.sample()
            
            latents_on_vae_device = latents_on_vae_device * scaling_factor
            
            latents_on_cpu = latents_on_vae_device.squeeze(0).cpu()
            
            item_from_dataset["instance_images"] = latents_on_cpu
            
            cached_dataset_items.append(item_from_dataset)
        
        dataset_to_load = cached_dataset_items
        print("Latent caching (direct imitation) complete.")
    else:
        print("Using pixel values directly from the dataset (no latent caching).")

    def collate_fn(examples):
        batch = {}
        if not examples:
            return batch

        if "instance_images" not in examples[0]:
            raise ValueError("Processed examples lack 'instance_images'.")
        
        image_or_latent_list = [example["instance_images"] for example in examples]
        batch["pixel_values"] = torch.stack(image_or_latent_list).to(memory_format=torch.contiguous_format).float()

        if "instance_prompt_ids" not in examples[0]:
            raise ValueError("Dataset's __getitem__ did not return 'instance_prompt_ids'.")

        main_ids_list = [example["instance_prompt_ids"] for example in examples]
        main_padded = main_tokenizer.pad(
            {"input_ids": main_ids_list},
            padding="max_length",
            max_length=main_tokenizer.model_max_length,
            return_tensors="pt",
        )
        batch["input_ids"] = main_padded.input_ids
        if "attention_mask" in main_padded:
            batch["attention_mask"] = main_padded.attention_mask

        if aux_tokenizer1 and "aux1_prompt_ids" in examples[0] and examples[0]["aux1_prompt_ids"] is not None:
            aux1_ids_list = [example.get("aux1_prompt_ids", []) for example in examples]
            if any(bool(ids) for ids in aux1_ids_list):
                aux1_padded = aux_tokenizer1.pad(
                    {"input_ids": aux1_ids_list},
                    padding="max_length",
                    max_length=aux_tokenizer1.model_max_length,
                    return_tensors="pt",
                )
                batch["aux1_input_ids"] = aux1_padded.input_ids
                if "attention_mask" in aux1_padded:
                    batch["aux1_attention_mask"] = aux1_padded.attention_mask
        elif aux_tokenizer1 and any(("aux1_prompt_ids" not in ex or ex.get("aux1_prompt_ids") is None) for ex in examples):
                 print("Warning: aux_tokenizer1 provided, but 'aux1_prompt_ids' not found or is None in some dataset samples.")

        if aux_tokenizer2 and "aux2_prompt_ids" in examples[0] and examples[0]["aux2_prompt_ids"] is not None:
            aux2_ids_list = [example.get("aux2_prompt_ids", []) for example in examples]
            if any(bool(ids) for ids in aux2_ids_list):
                aux2_padded = aux_tokenizer2.pad(
                    {"input_ids": aux2_ids_list},
                    padding="max_length",
                    max_length=aux_tokenizer2.model_max_length,
                    return_tensors="pt",
                )
                batch["aux2_input_ids"] = aux2_padded.input_ids
                if "attention_mask" in aux2_padded:
                    batch["aux2_attention_mask"] = aux2_padded.attention_mask
        elif aux_tokenizer2 and any(("aux2_prompt_ids" not in ex or ex.get("aux2_prompt_ids") is None) for ex in examples):
            print("Warning: aux_tokenizer2 provided, but 'aux2_prompt_ids' not found or is None in some dataset samples.")

        if "mask" in examples[0] and examples[0]["mask"] is not None:
            try:
                masks_list = [example["mask"] for example in examples]
                if all(isinstance(m, torch.Tensor) for m in masks_list):
                    batch["mask"] = torch.stack(masks_list)
                else:
                    non_tensor_masks_types = {type(m) for m in masks_list if not isinstance(m, torch.Tensor)}
                    print(f"Warning: Not all 'mask' items are tensors (found types: {non_tensor_masks_types}), skipping mask stacking for this batch.")
            except Exception as e:
                print(f"Warning: Could not stack masks. Error: {e}")
        
        if "raw_text" in examples[0]:
            batch["raw_text"] = [example.get("raw_text") for example in examples]

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        dataset_to_load,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 and torch.cuda.is_available() else False
    )

    data_type_in_pixel_values = "latents (cached)" if cached_latents else "pixel values (direct)"
    print(f"DataLoader configured. Batches will contain '{data_type_in_pixel_values}' in 'pixel_values' key.")
    return train_dataloader

def _save_debug_images(student_pred_noise, latents_gt_x0, noisy_latents, timesteps, vae,
                       scheduler, global_step, save_image_every_n_steps, output_dir_for_loss_step,
                       current_teacher_name, current_teacher_idx, base_save_dir,
                       save_comparison_grid, loss_val, batch):
    if global_step % save_image_every_n_steps != 0 or save_image_every_n_steps <= 0:
        return
    vae_decode_input_dtype = torch.float32
    with torch.no_grad():
        latents_gt_x0_viz = latents_gt_x0[0:1].detach()
        student_pred_noise_viz = student_pred_noise[0:1].detach()
        noisy_latents_viz = noisy_latents[0:1].detach()
        timestep_viz = timesteps[0:1].detach()
        current_timestep_int = timestep_viz.item() if timestep_viz.numel() == 1 else timestep_viz[0].item()
        scheduler_output_student = scheduler.step(
            model_output=student_pred_noise_viz.to(dtype=noisy_latents_viz.dtype),
            timestep=torch.tensor([current_timestep_int], device=noisy_latents_viz.device) if not isinstance(current_timestep_int, int) else current_timestep_int,
            sample=noisy_latents_viz
        )
        pred_x0_latent_student = scheduler_output_student.pred_original_sample
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        pred_x0_latent_student_for_vae = pred_x0_latent_student.to(dtype=vae_decode_input_dtype) / scaling_factor
        latent_x0_gt_viz_for_vae = latents_gt_x0_viz.to(dtype=vae_decode_input_dtype) / scaling_factor
        noisy_input_latent_for_vae = noisy_latents_viz.to(dtype=vae_decode_input_dtype) / scaling_factor
        vae_internal_dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float32
        with torch.cuda.amp.autocast(enabled=False):
            pred_image_student = vae.decode(pred_x0_latent_student_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
            gt_image = vae.decode(latent_x0_gt_viz_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
            noisy_input_image = vae.decode(noisy_input_latent_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
        pred_image_student = (pred_image_student / 2 + 0.5).clamp(0, 1)
        gt_image = (gt_image / 2 + 0.5).clamp(0, 1)
        noisy_input_image = (noisy_input_image / 2 + 0.5).clamp(0, 1)
        clean_teacher_name = re.sub(r'[^\w\-_.]', '_', current_teacher_name)
        if base_save_dir:
            teacher_save_dir = os.path.join(base_save_dir, f"teacher_{current_teacher_idx+1:02d}_{clean_teacher_name}")
        else:
            teacher_save_dir = os.path.join(output_dir_for_loss_step, f"teacher_{current_teacher_idx+1:02d}_{clean_teacher_name}")
        subdirs = {
            'student_pred': os.path.join(teacher_save_dir, 'student_predictions'),
            'ground_truth': os.path.join(teacher_save_dir, 'ground_truth'),
            'noisy_input': os.path.join(teacher_save_dir, 'noisy_inputs'),
            'comparisons': os.path.join(teacher_save_dir, 'comparisons')
        }
        for subdir in subdirs.values():
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
        t_val = current_timestep_int
        timestamp_suffix = f"step_{global_step:06d}_t{t_val:03d}"
        text_info = ""
        if "raw_text" in batch and batch["raw_text"]:
            raw_text = batch["raw_text"][0] if isinstance(batch["raw_text"], list) else str(batch["raw_text"])
            clean_text = re.sub(r'[^\w\s\-_.]', '', raw_text)[:20]
            text_info = f"_{clean_text.replace(' ', '_')}" if clean_text else ""
        student_pred_filename = f"{timestamp_suffix}_from_{clean_teacher_name}{text_info}.png"
        student_pred_path = os.path.join(subdirs['student_pred'], student_pred_filename)
        save_image(pred_image_student, student_pred_path)
        gt_filename = f"{timestamp_suffix}_gt{text_info}.png"
        gt_path = os.path.join(subdirs['ground_truth'], gt_filename)
        save_image(gt_image, gt_path)
        noisy_filename = f"{timestamp_suffix}_noisy{text_info}.png"
        noisy_path = os.path.join(subdirs['noisy_input'], noisy_filename)
        save_image(noisy_input_image, noisy_path)
        if save_comparison_grid:
            try:
                import torchvision.utils as vutils
                comparison_images = torch.cat([noisy_input_image, pred_image_student, gt_image], dim=0)
                grid = vutils.make_grid(comparison_images, nrow=3, padding=2, normalize=False, pad_value=1.0)
                comparison_filename = f"{timestamp_suffix}_comparison_from_{clean_teacher_name}{text_info}.png"
                comparison_path = os.path.join(subdirs['comparisons'], comparison_filename)
                save_image(grid.unsqueeze(0), comparison_path)
            except ImportError:
                print("Warning: torchvision.utils not available, skipping comparison grid")
            except Exception as e:
                print(f"Warning: Error saving comparison grid: {e}")
        print(f"[Step {global_step:06d}] Saved images for Teacher {current_teacher_idx+1} ({current_teacher_name}) dir={teacher_save_dir} t={t_val} loss={loss_val:.6f}")
        index_file_path = os.path.join(teacher_save_dir, "save_index.txt")
        with open(index_file_path, "a", encoding='utf-8') as f:
            f.write(f"{global_step:06d},{t_val:03d},{loss_val:.6f},{clean_teacher_name},{timestamp_suffix}\n")

def loss_step_gaussian_noise(
    batch,
    student_unet,
    student_text_encoder,
    scheduler,
    vae,
    global_step,
    save_image_every_n_steps=101,
    output_dir_for_loss_step="/root/lora_train/pic_train",
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    # Additional parameters: for identifying the current teacher and save directory
    current_teacher_name: str = "teacher",
    current_teacher_idx: int = 0,
    base_save_dir: str = None,
    save_comparison_grid: bool = False,  # Whether to save comparison grid image
):
    """Compute Gaussian noise prediction loss and optionally save debug images."""
    weight_dtype = torch.float32
    if mixed_precision:
        vae_decode_input_dtype = torch.float32
    else:
        vae_decode_input_dtype = torch.float32

    if batch["pixel_values"].ndim == 4 and batch["pixel_values"].shape[1] in [1, 3, 4]:
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)
    else:
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)

    bsz = latents_gt_x0.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents_gt_x0.device,
    )
    timesteps = timesteps.long()

    noise = torch.randn_like(latents_gt_x0)
    noisy_latents = scheduler.add_noise(latents_gt_x0, noise, timesteps)

    if mixed_precision:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float16)
    else:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float32)

    student_input_ids = batch.get("aux1_input_ids")
    if student_input_ids is None:
        student_input_ids = batch["input_ids"]
    student_input_ids = student_input_ids.to(student_text_encoder.device)

    student_attention_mask = batch.get("aux1_attention_mask")
    if student_attention_mask is not None:
        student_attention_mask = student_attention_mask.to(student_text_encoder.device)

    student_text_encoder_output_dtype_for_autocast = torch.float32
    if hasattr(student_text_encoder, 'dtype') and student_text_encoder.dtype == torch.float16:
         student_text_encoder_output_dtype_for_autocast = torch.float16

    if mixed_precision and student_text_encoder_output_dtype_for_autocast == torch.float16:
        with torch.cuda.amp.autocast(enabled=True):
            student_encoder_hidden_states = student_text_encoder(
                input_ids=student_input_ids, attention_mask=student_attention_mask
            )[0]
    else:
        _temp_states = student_text_encoder(
            input_ids=student_input_ids, attention_mask=student_attention_mask
        )[0]
        unet_expected_dtype = student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype
        student_encoder_hidden_states = _temp_states.to(dtype=unet_expected_dtype)

    student_unet_internal_dtype = student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype
    student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(dtype=student_unet_internal_dtype)

    if mixed_precision:
        with torch.cuda.amp.autocast(enabled=True):
            student_pred_noise = student_unet(
                student_unet_input_latents.to(student_unet_internal_dtype),
                timesteps,
                student_encoder_hidden_states_for_unet
            ).sample
    else:
        student_pred_noise = student_unet(
            student_unet_input_latents.to(student_unet_internal_dtype),
            timesteps,
            student_encoder_hidden_states_for_unet
        ).sample

    target_noise = noise

    if batch.get("mask", None) is not None:
        mask = batch["mask"].to(student_pred_noise.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Invalid mask shape: {mask.shape}. Expected shape is [B, 1, H, W] or [B, H, W]")

        mask = (mask + 0.01).pow(mask_temperature)
        mask = mask / mask.max()
        pred_dtype = student_pred_noise.dtype
        mask = mask.to(dtype=pred_dtype)
        student_pred_noise = student_pred_noise * mask
        target_noise = target_noise * mask

    loss = F.mse_loss(student_pred_noise.float(), target_noise.float(), reduction="none").mean([1, 2, 3]).mean()

    _save_debug_images(student_pred_noise, latents_gt_x0, noisy_latents, timesteps, vae,
                       scheduler, global_step, save_image_every_n_steps,
                       output_dir_for_loss_step, current_teacher_name, current_teacher_idx,
                       base_save_dir, save_comparison_grid, loss.item(), batch)
    return loss

def select_current_teacher(
    global_step: int,
    num_teachers: int,
    strategy: str = "round_robin",
    weights: Optional[List[float]] = None,
    selection_index: int = 0
) -> int:
    """Select the teacher index for the current step."""
    if strategy == "round_robin":
        return global_step % num_teachers
    
    elif strategy == "weighted_random":
        if weights is None:
            weights = [1.0] * num_teachers
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize
        return torch.multinomial(weights, 1).item()
    
    elif strategy == "adaptive":
        return global_step % num_teachers
    
    else:
        raise ValueError(f"Unknown teacher selection strategy: {strategy}")

def _log_scalar_safe(writer, tag, value, step):
    """Log a scalar to TensorBoard, safely handling int/float/Tensor/str types."""
    if writer is None:
        return
    if isinstance(value, (int, float)):
        writer.add_scalar(tag, value, step)
    elif isinstance(value, torch.Tensor):
        if value.numel() == 1:
            writer.add_scalar(tag, value.item(), step)
    elif isinstance(value, str):
        try:
            writer.add_scalar(tag, float(value), step)
        except (ValueError, TypeError):
            pass

def _log_layer_losses(writer, unet_prefix, text_prefix, teacher_unet_prefix,
                      teacher_text_prefix, unet_info, text_info, teacher_name, step):
    """Log per-layer hybrid feature losses to TensorBoard."""
    _CONFIG_STRINGS = {'hybrid', 'mse', 'cosine', 'scale_aware_cosine', 'layer_adaptive'}
    if isinstance(unet_info, dict):
        for layer_name, layer_loss in unet_info.items():
            clean = layer_name.replace(".", "_")
            if isinstance(layer_loss, (int, float)):
                writer.add_scalar(f'{unet_prefix}/{clean}', layer_loss, step)
                writer.add_scalar(f'{teacher_unet_prefix}/{teacher_name}/{clean}', layer_loss, step)
            elif isinstance(layer_loss, torch.Tensor) and layer_loss.numel() == 1:
                v = layer_loss.item()
                writer.add_scalar(f'{unet_prefix}/{clean}', v, step)
                writer.add_scalar(f'{teacher_unet_prefix}/{teacher_name}/{clean}', v, step)
            elif isinstance(layer_loss, str):
                try:
                    v = float(layer_loss)
                    writer.add_scalar(f'{unet_prefix}/{clean}', v, step)
                    writer.add_scalar(f'{teacher_unet_prefix}/{teacher_name}/{clean}', v, step)
                except (ValueError, TypeError):
                    if layer_loss not in _CONFIG_STRINGS:
                        print(f"Warning: skipping non-numeric loss: {layer_name}={layer_loss}")
            elif isinstance(layer_loss, torch.Tensor):
                print(f"Warning: skipping non-scalar tensor loss: {layer_name} shape={layer_loss.shape}")
    if isinstance(text_info, dict):
        for layer_name, layer_loss in text_info.items():
            clean = layer_name.replace(".", "_")
            if isinstance(layer_loss, (int, float)):
                writer.add_scalar(f'{text_prefix}/{clean}', layer_loss, step)
                writer.add_scalar(f'{teacher_text_prefix}/{teacher_name}/{clean}', layer_loss, step)
            elif isinstance(layer_loss, torch.Tensor) and layer_loss.numel() == 1:
                v = layer_loss.item()
                writer.add_scalar(f'{text_prefix}/{clean}', v, step)
                writer.add_scalar(f'{teacher_text_prefix}/{teacher_name}/{clean}', v, step)
            elif isinstance(layer_loss, str):
                try:
                    v = float(layer_loss)
                    writer.add_scalar(f'{text_prefix}/{clean}', v, step)
                    writer.add_scalar(f'{teacher_text_prefix}/{teacher_name}/{clean}', v, step)
                except (ValueError, TypeError):
                    if layer_loss not in _CONFIG_STRINGS:
                        print(f"Warning: skipping non-numeric loss: {layer_name}={layer_loss}")
            elif isinstance(layer_loss, torch.Tensor):
                print(f"Warning: skipping non-scalar tensor loss: {layer_name} shape={layer_loss.shape}")

def train_inversion_with_multi_feature_alignment(
    # --- "Teacher" model parameters (changed to list form) ---
    teacher_unets: List,  # Changed from teacher1_unet, teacher2_unet to list
    teacher_text_encoders: List,  # Changed from teacher1_text_encoder, teacher2_text_encoder to list
    # --- "Student" model parameters ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader parameters (changed to list form) ---
    dataloaders: List,  # Changed from dataloader1, dataloader2 to list
    # --- Core training parameters ---
    num_steps: int,
    scheduler,
    index_no_updates, # Indicates which token embeddings should not be updated
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    # --- Learning rate scheduler ---
    lr_scheduler_main, # Learning rate scheduler for the text embedding optimizer
    # --- LoRA related parameters (not usually used directly in TI, but save_all may need) ---
    lora_unet_target_modules, 
    lora_clip_target_modules, 
    # --- Output and logging related ---
    out_name: str,
    tokenizer,
    test_image_path: str,
    cached_latents: bool,
    # --- Loss function specific parameters ---
    mask_temperature: float = 1.0,
    t_multiplier_loss: float = 1.0, # Used for loss_step_gaussian_noise
    save_image_every_n_steps_loss: int = 200, # Used for loss_step_gaussian_noise
    # --- UNet Feature Alignment parameters ---
    unet_feature_align_weight: float = 0.01, # Weight for UNet feature alignment loss
    unet_feature_alignment_layers=[ # Layers used for UNet feature alignment
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    # --- Text Encoder Feature Alignment parameters ---
    text_encoder_feature_align_weight: float = 0.1, # Weight for Text Encoder feature alignment loss
    text_encoder_alignment_layers=[ # Layers used for Text Encoder feature alignment - added more layers
        'text_model.encoder.layers.0',   # First layer - captures basic language features
        'text_model.encoder.layers.2',   # Early layer - vocabulary understanding
        'text_model.encoder.layers.4',   # Early-middle layer - syntactic structure
        'text_model.encoder.layers.6',   # Middle layer - semantic understanding
        'text_model.encoder.layers.8',   # Late-middle layer - complex semantics
        'text_model.encoder.layers.10',  # Late layer - high-level semantics
        'text_model.encoder.layers.11'   # Last layer - final representation
    ],
    text_encoder_pooling_strategy: str = "mean", # "mean", "cls", "max", "none"
    text_encoder_loss_type: str = "mse", # 🔥 Changed to hybrid loss
    # --- 🔥 New: hybrid loss parameters ---
    text_encoder_primary_loss_type: str = "mse",  # Primary loss type
    text_encoder_secondary_loss_type: str = "cosine",  # Secondary loss type
    text_encoder_loss_combination_weight: float = 0.3,  # Secondary loss weight
    text_encoder_use_layer_adaptive: bool = True,  # Whether to use layer-adaptive
    # --- UNet hybrid loss parameters ---
    unet_feature_loss_type: str = "mse",  # 🔥 UNet also uses hybrid loss
    unet_primary_loss_type: str = "mse",
    unet_secondary_loss_type: str = "cosine", 
    unet_loss_combination_weight: float = 0.3,
    unet_use_layer_adaptive: bool = True,
    # --- Main loss weights ---
    noise_pred_weight: float = 1.0, # Weight for the main TI loss (noise prediction)
    # --- Other parameters ---
    unet_return_dict: bool = True, # Whether UNetFeatureExtractor returns a dictionary
    accum_iter: int = 1, # Gradient accumulation steps
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False, # Usually False in TI
    mixed_precision: str = "no", # "no", "fp16", "bf16"
    clip_ti_decay: bool = True, # Whether to apply decay/regularization to TI embeddings
    tensorboard_log_dir: str = "runs_new",
    teacher_selection_strategy: str = "round_robin",  # New: teacher selection strategy
    teacher_weights: Optional[List[float]] = None,  # New: teacher weights
):
    """Multi-teacher text inversion training with hybrid UNet and text encoder feature alignment."""

    # Validate input parameters
    assert len(teacher_unets) == len(teacher_text_encoders) == len(dataloaders), \
        "teacher_unets, teacher_text_encoders, and dataloaders must have the same length"
    
    num_teachers = len(teacher_unets)
    print(f"Training with {num_teachers} teacher models")

    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard setup ---
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_path = os.path.join(tensorboard_log_dir, f"{out_name}_inversion_multi_teacher_hybrid_feat_align_{timestamp}")

    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logs: {tb_log_path}")

    # --- 🔥 UNet hybrid feature alignment component setup ---
    unet_alignment_layers_for_loss = unet_feature_alignment_layers
    unet_layer_weights_for_fa_loss = {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }
    unet_layer_weights_for_fa_loss = {
        k: v for k, v in unet_layer_weights_for_fa_loss.items() 
        if k in unet_alignment_layers_for_loss
    }

    # 🔥 Create hybrid UNet feature alignment loss function
    from feature_hook_unet import create_hybrid_unet_alignment_loss
    
    unet_feature_alignment_loss_fn = create_hybrid_unet_alignment_loss(
        alignment_layers=unet_alignment_layers_for_loss,
        loss_type=unet_feature_loss_type,
        loss_weights=unet_layer_weights_for_fa_loss,
        primary_loss_type=unet_primary_loss_type,
        secondary_loss_type=unet_secondary_loss_type,
        loss_combination_weight=unet_loss_combination_weight,
        use_layer_adaptive=unet_use_layer_adaptive,
        feature_selection_strategy="adaptive",
        normalize_features=True,
        channel_alignment="projection"
    )
    
    print(f"UNet feature alignment: type={unet_feature_loss_type}, layers={len(unet_alignment_layers_for_loss)}")

    # Create UNet feature extractor for each teacher
    teacher_unet_extractors = []
    for i, teacher_unet in enumerate(teacher_unets):
        extractor = UNetFeatureExtractor(
            target_layers=unet_alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        )
        teacher_unet_extractors.append(extractor)
    
    # Create student UNet feature extractor
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )
    
    # Initialize dataloader iterators
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0  # Used for round_robin strategy

    # 🔥 Text Encoder hybrid feature extractor
    text_encoder_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )

    # 🔥 Text Encoder hybrid feature alignment loss function
    from feature_hook_text_encoder import create_hybrid_text_encoder_alignment_loss
    
    text_encoder_feature_alignment_loss_fn = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=text_encoder_alignment_layers,
        loss_type=text_encoder_loss_type,
        pooling_strategy=text_encoder_pooling_strategy,
        primary_loss_type=text_encoder_primary_loss_type,
        secondary_loss_type=text_encoder_secondary_loss_type,
        loss_combination_weight=text_encoder_loss_combination_weight,
        use_layer_adaptive=text_encoder_use_layer_adaptive
    )
    
    print(f"🔥 Text Encoder hybrid feature alignment loss configuration:")
    print(f"   - Loss type: {text_encoder_loss_type}")
    print(f"   - Pooling strategy: {text_encoder_pooling_strategy}")
    if text_encoder_loss_type == "hybrid":
        print(f"   - Primary loss: {text_encoder_primary_loss_type} ({100*(1-text_encoder_loss_combination_weight):.1f}%)")
        print(f"   - Secondary loss: {text_encoder_secondary_loss_type} ({100*text_encoder_loss_combination_weight:.1f}%)")

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Training steps (multi-teacher hybrid feature alignment text inversion)")
    global_step = 0

    # Backup original non-placeholder token embeddings
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        if wandb.run is None:
            wandb.init(
                project=f"textual_inversion_project_{out_name}", 
                name=f"{out_name}_inversion_multi_teacher_hybrid_feat_align_run", 
                reinit=True
            )
        wandb.config.update({
            k:v for k,v in locals().items() 
            if isinstance(v, (int, float, str, bool, list, dict))
        })
        preped_clip = prepare_clip_model_sets() if 'prepare_clip_model_sets' in globals() else None

    index_updates = ~index_no_updates

    # Accumulated losses for interval logging
    accumulated_total_loss = 0.0
    accumulated_ti_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    # === New: loss statistics for each teacher model ===
    # Create accumulated loss tracker for each teacher
    teacher_loss_trackers = []
    for i in range(num_teachers):
        teacher_loss_trackers.append({
            'ti_loss': 0.0,
            'unet_feature_loss': 0.0,
            'text_encoder_feature_loss': 0.0,
            'total_loss': 0.0,
            'step_count': 0,
            # 🔥 New hybrid loss tracking
            'unet_primary_loss': 0.0,
            'unet_secondary_loss': 0.0,
            'text_primary_loss': 0.0,
            'text_secondary_loss': 0.0,
        })

    # Determine device from model
    unet_device = next(student_unet.parameters()).device
    text_encoder_device = next(student_text_encoder.parameters()).device
    vae_device = vae.device

    # Main training loop
    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # --- Select current teacher model and data loader ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # Get currently selected model and data
        current_teacher_unet = teacher_unets[current_teacher_idx]
        current_teacher_text_encoder = teacher_text_encoders[current_teacher_idx]
        current_teacher_unet_extractor = teacher_unet_extractors[current_teacher_idx]
        current_dataloader_iter = dataloader_iters[current_teacher_idx]
        current_dataloader_obj = dataloaders[current_teacher_idx]
        teacher_name_log = f"T{current_teacher_idx + 1}"
        
        # Update round_robin index
        if teacher_selection_strategy == "round_robin":
            teacher_selection_index = (teacher_selection_index + 1) % num_teachers
        
        # Get batch data
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            dataloader_iters[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(dataloader_iters[current_teacher_idx])

        # --- Latent variable preparation ---
        main_student_unet = student_unet.module if hasattr(student_unet, 'module') else student_unet
        expected_latents_dtype = main_student_unet.dtype if hasattr(main_student_unet, 'dtype') else torch.float32

        if cached_latents:
            latents = batch["pixel_values"].to(device=unet_device, dtype=expected_latents_dtype)
        else:
            pixel_values_for_vae = batch["pixel_values"].to(device=vae_device, dtype=vae.dtype)
            with torch.no_grad():
                latents_dist = vae.encode(pixel_values_for_vae).latent_dist
                latents = latents_dist.sample() * vae.config.scaling_factor
            latents = latents.to(device=unet_device, dtype=expected_latents_dtype)

        # --- Prepare input data ---
        input_ids = batch["input_ids"].to(device=text_encoder_device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=text_encoder_device)

        # --- Text encoding and feature extraction ---
        main_teacher_unet = current_teacher_unet.module if hasattr(current_teacher_unet, 'module') else current_teacher_unet
        teacher_unet_internal_dtype = next(main_teacher_unet.parameters()).dtype
        student_unet_internal_dtype = next(main_student_unet.parameters()).dtype

        # Teacher Text Encoder encoding and feature extraction (no gradient needed)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and text_encoder_device.type == 'cuda')):
                teacher_text_output, teacher_text_features = text_encoder_feature_extractor.extract_features(
                    text_encoder=current_teacher_text_encoder,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_grad=False
                )
                teacher_encoder_hidden_states = teacher_text_output.to(dtype=teacher_unet_internal_dtype)

        # Student Text Encoder encoding and feature extraction (gradient needed for TI)
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and text_encoder_device.type == 'cuda')):
            student_text_output, student_text_features = text_encoder_feature_extractor.extract_features(
                text_encoder=student_text_encoder,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_grad=True  # Allow gradient flow to text embeddings
            )
            student_encoder_hidden_states = student_text_output.to(dtype=student_unet_internal_dtype)

        # --- Noise and timesteps ---
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)
        
        noisy_latents_for_teacher = noisy_latents.to(
            device=next(current_teacher_unet.parameters()).device, 
            dtype=teacher_unet_internal_dtype
        )
        timesteps_for_teacher = timesteps.to(device=next(current_teacher_unet.parameters()).device)
        
        noisy_latents_for_student = noisy_latents.to(device=unet_device, dtype=student_unet_internal_dtype)
        timesteps_for_student = timesteps.to(device=unet_device)

        # --- Forward propagation and loss computation ---
        with torch.set_grad_enabled(True):
            # 1. Teacher UNet forward pass and feature extraction (no gradient needed)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(use_mixed_precision and noisy_latents_for_teacher.device.type == 'cuda')):
                    _, teacher_unet_features = current_teacher_unet_extractor.extract_features(
                        unet_model=current_teacher_unet,
                        sample=noisy_latents_for_teacher,
                        timestep=timesteps_for_teacher,
                        encoder_hidden_states=teacher_encoder_hidden_states,
                        return_dict=unet_return_dict,
                        use_grad=False
                    )

            # 2. Student UNet forward pass and feature extraction (gradient needs to flow to text embeddings)
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and noisy_latents_for_student.device.type == 'cuda')):
                _, student_unet_features = student_unet_extractor.extract_features(
                    unet_model=student_unet, 
                    sample=noisy_latents_for_student,
                    timestep=timesteps_for_student,
                    encoder_hidden_states=student_encoder_hidden_states,
                    return_dict=unet_return_dict,
                    use_grad=True
                )
            
            # 3. Main TI loss (noise prediction loss)
            ti_loss_current_step = loss_step_gaussian_noise(
                batch=batch,
                student_unet=student_unet,
                student_text_encoder=student_text_encoder,
                vae=vae,
                global_step=global_step,
                scheduler=scheduler,
                t_mutliplier=t_multiplier_loss,
                mixed_precision=(mixed_precision != "no"),
                mask_temperature=mask_temperature,
            )

            # 🔥 4. UNet hybrid feature alignment loss
            valid_teacher_unet_features = isinstance(teacher_unet_features, dict) and teacher_unet_features
            valid_student_unet_features = isinstance(student_unet_features, dict) and student_unet_features

            if valid_teacher_unet_features and valid_student_unet_features:
                # Use new hybrid loss function, returns 3 values
                unet_feature_align_loss_val, unet_feature_loss_dict_current, unet_detailed_loss_info = unet_feature_alignment_loss_fn(
                    teacher_unet_features, student_unet_features
                )
            else:
                unet_feature_align_loss_val = torch.tensor(0.0, device=ti_loss_current_step.device, dtype=ti_loss_current_step.dtype)
                unet_feature_loss_dict_current = {}
                unet_detailed_loss_info = {}

            # 🔥 5. Text Encoder hybrid feature alignment loss
            valid_teacher_text_features = isinstance(teacher_text_features, dict) and teacher_text_features
            valid_student_text_features = isinstance(student_text_features, dict) and student_text_features

            if valid_teacher_text_features and valid_student_text_features:
                # Use new hybrid loss function, returns 3 values
                text_encoder_feature_align_loss_val, text_encoder_feature_loss_dict_current, text_detailed_loss_info = text_encoder_feature_alignment_loss_fn(
                    teacher_features=teacher_text_features,
                    student_features=student_text_features,
                    teacher_attention_mask=attention_mask,
                    student_attention_mask=attention_mask
                )
            else:
                text_encoder_feature_align_loss_val = torch.tensor(0.0, device=ti_loss_current_step.device, dtype=ti_loss_current_step.dtype)
                text_encoder_feature_loss_dict_current = {}
                text_detailed_loss_info = {}

            # 6. Dynamic weight adjustment
            inversion_weight_adjuster = create_inversion_weight_adjuster(
                noise_pred_weight=noise_pred_weight,
                unet_feature_weight=unet_feature_align_weight,
                text_encoder_feature_weight=text_encoder_feature_align_weight
            )

            current_losses = {
                'ti_loss': ti_loss_current_step.item(),
                'unet_feature_align': unet_feature_align_loss_val.item() if isinstance(unet_feature_align_loss_val, torch.Tensor) else unet_feature_align_loss_val,
                'text_encoder_feature_align': text_encoder_feature_align_loss_val.item() if isinstance(text_encoder_feature_align_loss_val, torch.Tensor) else text_encoder_feature_align_loss_val
            }

            updated_weights = inversion_weight_adjuster.update_weights(current_losses)
            noise_pred_weight = updated_weights['ti_loss']
            unet_feature_align_weight = updated_weights['unet_feature_align']
            text_encoder_feature_align_weight = updated_weights['text_encoder_feature_align']

            # 7. Total loss
            current_step_total_loss = (
                noise_pred_weight * ti_loss_current_step +
                # unet_feature_align_weight * unet_feature_align_loss_val +  # 🔥 Temporarily commented out UNet feature alignment
                text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
            )
            
            loss_for_backward = current_step_total_loss / accum_iter
            loss_for_backward.backward()

            # === New: record current teacher's hybrid loss to tracker ===
            current_ti_loss_item = ti_loss_current_step.detach().item()
            current_unet_feature_loss_item = unet_feature_align_loss_val.detach().item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val)
            current_text_encoder_feature_loss_item = text_encoder_feature_align_loss_val.detach().item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val)
            current_total_loss_item = current_step_total_loss.detach().item()

            # 🔥 Extract hybrid loss detailed information
            unet_primary_loss = 0.0
            unet_secondary_loss = 0.0
            text_primary_loss = 0.0
            text_secondary_loss = 0.0
            
            # Extract primary and secondary losses from detailed loss info
            for layer_name, loss_value in unet_detailed_loss_info.items():
                if 'primary' in layer_name:
                    unet_primary_loss += loss_value
                elif 'secondary' in layer_name:
                    unet_secondary_loss += loss_value
            
            for layer_name, loss_value in text_detailed_loss_info.items():
                if 'primary' in layer_name:
                    text_primary_loss += loss_value
                elif 'secondary' in layer_name:
                    text_secondary_loss += loss_value

            # Update current teacher's hybrid loss tracker
            teacher_loss_trackers[current_teacher_idx]['ti_loss'] += current_ti_loss_item
            teacher_loss_trackers[current_teacher_idx]['unet_feature_loss'] += current_unet_feature_loss_item
            teacher_loss_trackers[current_teacher_idx]['text_encoder_feature_loss'] += current_text_encoder_feature_loss_item
            teacher_loss_trackers[current_teacher_idx]['total_loss'] += current_total_loss_item
            teacher_loss_trackers[current_teacher_idx]['step_count'] += 1
            # 🔥 New hybrid loss tracking
            teacher_loss_trackers[current_teacher_idx]['unet_primary_loss'] += unet_primary_loss
            teacher_loss_trackers[current_teacher_idx]['unet_secondary_loss'] += unet_secondary_loss
            teacher_loss_trackers[current_teacher_idx]['text_primary_loss'] += text_primary_loss
            teacher_loss_trackers[current_teacher_idx]['text_secondary_loss'] += text_secondary_loss

            # Record global accumulated losses
            accumulated_total_loss += current_total_loss_item
            accumulated_ti_loss += current_ti_loss_item
            accumulated_unet_feature_loss += current_unet_feature_loss_item
            accumulated_text_encoder_feature_loss += current_text_encoder_feature_loss_item
            step_count_in_interval += 1

        # --- Optimizer step and embedding regularization ---
        if (global_step + 1) % accum_iter == 0:
            if student_text_encoder.get_input_embeddings().weight.grad is not None:
                grad_slice = student_text_encoder.get_input_embeddings().weight.grad[index_updates, :]
                if grad_slice.numel() > 0:
                    grad_norm = grad_slice.norm(dim=-1).mean()
                    if writer:
                        writer.add_scalar('gradient/text_embedding_norm_multi_teacher_hybrid_FA', grad_norm.item(), global_step)
            else:
                print(f"Step {global_step}: Warning: No gradient found for text embeddings during multi-teacher hybrid FA update.")

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Text embedding regularization
            with torch.no_grad():
                embed_weights = student_text_encoder.get_input_embeddings().weight
                if clip_ti_decay:
                    updated_embeds = embed_weights[index_updates, :]
                    pre_norm = updated_embeds.norm(dim=-1, keepdim=True)
                    lambda_ = min(1.0, 100 * lr_scheduler_main.get_last_lr()[0]) 
                    
                    normalized_embeds = F.normalize(updated_embeds, dim=-1)
                    target_norm = pre_norm + lambda_ * (0.4 - pre_norm) 
                    
                    embed_weights.data[index_updates] = normalized_embeds * target_norm

                current_norm_val = embed_weights[index_updates, :].norm(dim=-1).mean().item()
                if writer:
                    writer.add_scalar('embedding/current_norm_multi_teacher_hybrid_FA', current_norm_val, global_step)

                # Restore original embeddings that were not updated
                embed_weights.data[index_no_updates] = orig_embeds_params[index_no_updates]
        
        lr_scheduler_main.step()
        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0]
        
        # 🔥 Display hybrid loss information
        log_dict_postfix = {
            "Total loss": f"{current_step_total_loss.item():.4f}",
            "TI loss": f"{ti_loss_current_step.item():.4f}",
            "UNet features": f"{unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val):.4f}",
            "Text features": f"{text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val):.4f}",
            "LR": f"{current_lr_val:.2e}",
            "Teacher": teacher_name_log
        }
        progress_bar.set_postfix(**log_dict_postfix)

        # === New: detailed TensorBoard logging ===
        if writer:
            # Global loss logging
            writer.add_scalar('loss_multi_teacher_hybrid_FA/total_loss_per_step', current_step_total_loss.item(), global_step)
            writer.add_scalar('loss_multi_teacher_hybrid_FA/TI_loss_per_step', ti_loss_current_step.item(), global_step)
            writer.add_scalar('loss_multi_teacher_hybrid_FA/UNet_feature_alignment_loss_per_step', unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val), global_step)
            writer.add_scalar('loss_multi_teacher_hybrid_FA/TextEncoder_feature_alignment_loss_per_step', text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val), global_step)
            writer.add_scalar('learning_rate_multi_teacher_hybrid_FA/text_embedding', current_lr_val, global_step)
            writer.add_scalar('teacher_model_multi_teacher_hybrid_FA/current_selection', current_teacher_idx + 1, global_step)
            writer.add_scalar('weight_multi_teacher_hybrid/TI_loss_weight', noise_pred_weight, global_step)
            writer.add_scalar('weight_multi_teacher_hybrid/UNet_feature_weight', unet_feature_align_weight, global_step)
            writer.add_scalar('weight_multi_teacher_hybrid/TextEncoder_feature_weight', text_encoder_feature_align_weight, global_step)
            
            # 🔥 New: hybrid loss decomposition logging
            writer.add_scalar('hybrid_loss_UNet/primary_loss(MSE)', unet_primary_loss, global_step)
            writer.add_scalar('hybrid_loss_UNet/secondary_loss(Cosine)', unet_secondary_loss, global_step)
            writer.add_scalar('hybrid_loss_Text/primary_loss(MSE)', text_primary_loss, global_step)
            writer.add_scalar('hybrid_loss_Text/secondary_loss(Cosine)', text_secondary_loss, global_step)
            
            # === New: hybrid loss logging categorized by teacher ===
            teacher_name_for_tb = f"Teacher_{current_teacher_idx+1:02d}"
            
            # Current teacher's per-step loss
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/TI_loss_per_step', current_ti_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/UNet_feature_loss_per_step', current_unet_feature_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/TextEncoder_feature_loss_per_step', current_text_encoder_feature_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/total_loss_per_step', current_total_loss_item, global_step)
            
            # 🔥 Current teacher's hybrid loss decomposition
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/UNet_primary_loss_per_step', unet_primary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/UNet_secondary_loss_per_step', unet_secondary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/Text_primary_loss_per_step', text_primary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/Text_secondary_loss_per_step', text_secondary_loss, global_step)
            
            # Teacher selection frequency statistics
            teacher_selection_counts = [tracker['step_count'] for tracker in teacher_loss_trackers]
            for i, count in enumerate(teacher_selection_counts):
                writer.add_scalar(f'teacher_stats/Teacher_{i+1:02d}_selection_count', count, global_step)
            
            _log_layer_losses(
                writer,
                'UNet_hybrid_feature_loss_layer', 'TextEncoder_hybrid_feature_loss_layer',
                'teacher_UNet_hybrid_layer_loss', 'teacher_TextEncoder_hybrid_layer_loss',
                unet_detailed_loss_info, text_detailed_loss_info, teacher_name_for_tb, global_step
            )

    print(f"\nMulti-teacher hybrid FA inversion complete: steps={global_step}, "
          f"lr={current_lr_val:.2e}, teachers={num_teachers}, strategy={teacher_selection_strategy}")
    if writer:
        for i, tracker in enumerate(teacher_loss_trackers):
            teacher_name = f"Teacher_{i+1:02d}"
            step_count = tracker['step_count']
            if step_count > 0:
                sc = step_count
                final_avg_ti_loss = tracker['ti_loss'] / sc
                final_avg_unet_loss = tracker['unet_feature_loss'] / sc
                final_avg_text_loss = tracker['text_encoder_feature_loss'] / sc
                final_avg_total_loss = tracker['total_loss'] / sc
                final_usage_freq = sc / global_step
                final_avg_unet_primary = tracker['unet_primary_loss'] / sc
                final_avg_unet_secondary = tracker['unet_secondary_loss'] / sc
                final_avg_text_primary = tracker['text_primary_loss'] / sc
                final_avg_text_secondary = tracker['text_secondary_loss'] / sc
                writer.add_scalar(f'final_stats/{teacher_name}/final_avg_TI_loss', final_avg_ti_loss, global_step)
                writer.add_scalar(f'final_stats/{teacher_name}/final_avg_UNet_loss', final_avg_unet_loss, global_step)
                writer.add_scalar(f'final_stats/{teacher_name}/final_avg_Text_loss', final_avg_text_loss, global_step)
                writer.add_scalar(f'final_stats/{teacher_name}/final_avg_total_loss', final_avg_total_loss, global_step)
                writer.add_scalar(f'final_stats/{teacher_name}/final_usage_frequency', final_usage_freq, global_step)
                writer.add_scalar(f'final_hybrid_stats/{teacher_name}/UNet_primary_loss_mean', final_avg_unet_primary, global_step)
                writer.add_scalar(f'final_hybrid_stats/{teacher_name}/UNet_secondary_loss_mean', final_avg_unet_secondary, global_step)
                writer.add_scalar(f'final_hybrid_stats/{teacher_name}/Text_primary_loss_mean', final_avg_text_primary, global_step)
                writer.add_scalar(f'final_hybrid_stats/{teacher_name}/Text_secondary_loss_mean', final_avg_text_secondary, global_step)
                print(f"  {teacher_name}: steps={sc}/{global_step} ({final_usage_freq:.2%}), "
                      f"ti={final_avg_ti_loss:.4f}, total={final_avg_total_loss:.4f}")
    writer.close()
    
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_step_inv_multi_teacher_hybrid_{global_step}.safetensors")
    save_all(
        unet=student_unet, text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path,
        save_lora=False,
    )
    
    print(f"✅ Final hybrid loss model saved to: {final_model_path}")
    print(f"✅ TensorBoard logs saved to: {tb_log_path}")
    print(f"🔥 Multi-teacher hybrid feature alignment training complete!")
    print(f"="*80)

def perform_tuning_multi_teacher(
    teacher_unets: List,
    teacher_text_encoders: List,
    student_unet,
    vae,
    student_text_encoder,
    dataloaders: List,
    num_steps,
    cached_latents,
    scheduler,
    optimizer,
    save_steps,
    placeholder_tokens,
    placeholder_token_ids,
    save_path,
    lr_scheduler_lora,
    lora_unet_target_modules,
    lora_clip_target_modules,
    mask_temperature,
    tokenizer,
    out_name,
    mixed_precision="no",
    log_wandb=False,
    wandb_log_prompt_cnt=10,
    class_token="person",
    train_inpainting=False,
    feature_align_weight=0.005,
    noise_pred_weight=1.0,
    feature_alignment_unet_layers=[
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    unet_return_dict=True,
    tensorboard_log_dir: str = "runs_new",
    teacher_selection_strategy: str = "round_robin",
    teacher_weights: Optional[List[float]] = None,
    
    # 🔥 New: hybrid loss parameters
    unet_feature_loss_type: str = "mse",  # "mse", "cosine", "hybrid", "scale_aware_cosine", "layer_adaptive"
    unet_primary_loss_type: str = "mse",
    unet_secondary_loss_type: str = "cosine",
    unet_loss_combination_weight: float = 0.3,
    unet_use_layer_adaptive: bool = True,
    
    text_encoder_feature_align_weight: float = 0.1,
    text_encoder_alignment_layers: List[str] = [
        'text_model.encoder.layers.0', 'text_model.encoder.layers.2', 'text_model.encoder.layers.4',
        'text_model.encoder.layers.6', 'text_model.encoder.layers.8', 'text_model.encoder.layers.10',
        'text_model.encoder.layers.11'
    ],
    text_encoder_pooling_strategy: str = "mean",
    text_encoder_loss_type: str = "mse",
    text_encoder_primary_loss_type: str = "mse",
    text_encoder_secondary_loss_type: str = "cosine",
    text_encoder_loss_combination_weight: float = 0.3,
    text_encoder_use_layer_adaptive: bool = True,
    
    # Alternating optimization parameters
    use_alternating_optimization: bool = False,
    alternating_interval: int = 5,
    alternating_schedule: str = "fixed",
    noise_only_steps: int = 1000,
    feature_only_steps: int = 0,
):
    """Multi-teacher LoRA fine-tuning with hybrid feature alignment loss."""
    
    assert len(teacher_unets) == len(teacher_text_encoders) == len(dataloaders), \
        "teacher_unets, teacher_text_encoders, and dataloaders must have the same length"
    
    num_teachers = len(teacher_unets)
    print(f"Hybrid-loss LoRA fine-tuning with {num_teachers} teacher models")

    class AlternatingOptimizationController:
        def __init__(self):
            self.current_mode = "noise"  # "noise" or "feature"
            self.mode_switch_count = 0
            self.last_switch_step = 0
            self.noise_loss_history = []
            self.feature_loss_history = []
            
        def should_switch_mode(self, global_step):
            """Determine whether to switch optimization mode"""
            if alternating_schedule == "fixed":
                # Switch at fixed intervals
                if global_step > 0 and (global_step - self.last_switch_step) >= alternating_interval:
                    return True
            elif alternating_schedule == "adaptive":
                # Simple adaptive strategy: based on loss trend
                if len(self.noise_loss_history) >= 5 and len(self.feature_loss_history) >= 5:
                    # If the currently optimized loss hasn't decreased significantly in recent steps, switch
                    if self.current_mode == "noise":
                        recent_trend = (self.noise_loss_history[-1] - self.noise_loss_history[-3]) / (self.noise_loss_history[-3] + 1e-8)
                        if recent_trend > -0.01:  # No significant decrease
                            return True
                    else:
                        recent_trend = (self.feature_loss_history[-1] - self.feature_loss_history[-3]) / (self.feature_loss_history[-3] + 1e-8)
                        if recent_trend > -0.01:  # No significant decrease
                            return True
            return False
            
        def switch_mode(self, global_step):
            """Switch optimization mode"""
            old_mode = self.current_mode
            self.current_mode = "feature" if self.current_mode == "noise" else "noise"
            self.mode_switch_count += 1
            self.last_switch_step = global_step
            
            print(f"🔄 [Step {global_step}] Switched optimization mode: {old_mode} -> {self.current_mode} (switch #{self.mode_switch_count})")
            
        def get_current_mode(self, global_step):
            """Get optimization mode for the current step"""
            # Handle special phases
            if global_step < noise_only_steps:
                return "noise"
            elif feature_only_steps > 0 and global_step >= (num_steps - feature_only_steps):
                return "feature"
            
            # Normal alternating phase
            return self.current_mode
            
        def update_loss_history(self, noise_loss, feature_loss):
            """Update loss history"""
            self.noise_loss_history.append(noise_loss)
            self.feature_loss_history.append(feature_loss)
            
            # Maintain history length
            if len(self.noise_loss_history) > 20:
                self.noise_loss_history.pop(0)
            if len(self.feature_loss_history) > 20:
                self.feature_loss_history.pop(0)

    if use_alternating_optimization:
        alt_controller = AlternatingOptimizationController()
        print(f"Alternating optimization: interval={alternating_interval}, schedule={alternating_schedule}, "
              f"noise_only={noise_only_steps}, feature_only={feature_only_steps}")
    else:
        alt_controller = None
        print("Joint loss optimization mode")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_mode = "alternating" if use_alternating_optimization else "joint"
    tb_log_path = os.path.join(tensorboard_log_dir, f"{out_name}_{opt_mode}_multi_teacher_hybrid_lora_{timestamp}")
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logs: {tb_log_path}")

    alignment_layers_for_loss = feature_alignment_unet_layers
    layer_weights_for_loss = {k: v for k, v in {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }.items() if k in alignment_layers_for_loss}

    from feature_hook_unet import create_hybrid_unet_alignment_loss
    
    unet_feature_alignment_loss_fn = create_hybrid_unet_alignment_loss(
        alignment_layers=alignment_layers_for_loss,
        loss_type=unet_feature_loss_type,
        loss_weights=layer_weights_for_loss,
        primary_loss_type=unet_primary_loss_type,
        secondary_loss_type=unet_secondary_loss_type,
        loss_combination_weight=unet_loss_combination_weight,
        use_layer_adaptive=unet_use_layer_adaptive,
        feature_selection_strategy="adaptive",
        normalize_features=True,
        channel_alignment="projection"
    )
    
    print(f"UNet feature alignment: type={unet_feature_loss_type}")
    print(f"Text Encoder feature alignment: type={text_encoder_loss_type}, pooling={text_encoder_pooling_strategy}")

    from feature_hook_text_encoder import (
        TextEncoderFeatureExtractor, 
        create_hybrid_text_encoder_alignment_loss
    )
    text_encoder_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )
    text_encoder_feature_alignment_loss_fn = create_hybrid_text_encoder_alignment_loss(
        alignment_layers=text_encoder_alignment_layers,
        loss_type=text_encoder_loss_type,
        pooling_strategy=text_encoder_pooling_strategy,
        primary_loss_type=text_encoder_primary_loss_type,
        secondary_loss_type=text_encoder_secondary_loss_type,
        loss_combination_weight=text_encoder_loss_combination_weight,
        use_layer_adaptive=text_encoder_use_layer_adaptive
    )

    from feature_hook_unet import UNetFeatureExtractor
    teacher_extractors = [
        UNetFeatureExtractor(target_layers=feature_alignment_unet_layers, mixed_precision_config=mixed_precision)
        for _ in teacher_unets
    ]
    student_extractor = UNetFeatureExtractor(
        target_layers=feature_alignment_unet_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps), desc=f"Multi-Teacher Hybrid LoRA ({'Alternating' if use_alternating_optimization else 'Joint'}) Training")
    global_step = 0
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0

    teacher_loss_trackers = [{
        'noise_pred_loss': 0.0, 'unet_feature_align_loss': 0.0,
        'text_encoder_feature_align_loss': 0.0, 'total_loss': 0.0, 'step_count': 0,
        'unet_primary_loss': 0.0, 'unet_secondary_loss': 0.0,
        'text_primary_loss': 0.0, 'text_secondary_loss': 0.0,
    } for _ in range(num_teachers)]

    accumulated_total_loss = 0.0
    accumulated_noise_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    student_unet.train()
    student_text_encoder.train()

    # ===== Main training loop =====
    for step_idx in range(num_steps):
        # --- Select current teacher model ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # Get currently selected model and data
        current_teacher_unet = teacher_unets[current_teacher_idx]
        current_teacher_text_encoder = teacher_text_encoders[current_teacher_idx]
        current_teacher_extractor = teacher_extractors[current_teacher_idx]
        current_dataloader_iter = dataloader_iters[current_teacher_idx]
        current_dataloader_obj = dataloaders[current_teacher_idx]
        teacher_name_log = f"T{current_teacher_idx + 1}"
        
        # Update round_robin index
        if teacher_selection_strategy == "round_robin":
            teacher_selection_index = (teacher_selection_index + 1) % num_teachers

        # Get batch data
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            dataloader_iters[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(dataloader_iters[current_teacher_idx])

        # --- 🔥 Forward pass to compute all losses (for monitoring and decision making) ---
        
        # 1. Compute noise prediction loss
        noise_pred_loss = loss_step_gaussian_noise(
            batch=batch,
            student_unet=student_unet,
            student_text_encoder=student_text_encoder,
            vae=vae,
            global_step=global_step,
            scheduler=scheduler,
            t_mutliplier=0.8,
            mixed_precision=(mixed_precision != "no"),
            mask_temperature=mask_temperature,
            current_teacher_name=f"teacher_{current_teacher_idx+1}",
            current_teacher_idx=current_teacher_idx,
            save_comparison_grid=True,
        )

        # 2. Compute feature alignment losses
        
        # Latent processing
        main_module = student_unet.module if hasattr(student_unet, 'module') else student_unet
        expected_latents_dtype = main_module.dtype if hasattr(main_module, 'dtype') else torch.float32

        if cached_latents:
            latents = batch["pixel_values"].to(device=main_module.device, dtype=expected_latents_dtype)
        else:
            vae_device = vae.device
            input_pixels_for_vae = batch["pixel_values"].to(device=vae_device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(input_pixels_for_vae).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(device=main_module.device, dtype=expected_latents_dtype)

        # Text encoding
        text_encoder_device = student_text_encoder.device if hasattr(student_text_encoder, 'device') else main_module.device
        input_ids = batch["input_ids"].to(device=text_encoder_device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=text_encoder_device)
        
        student_input_ids = batch.get('aux1_input_ids', input_ids).to(device=text_encoder_device)
        student_attention_mask = batch.get('aux1_attention_mask', attention_mask)
        if student_attention_mask is not None:
            student_attention_mask = student_attention_mask.to(device=text_encoder_device)

        # 🔥 Teacher Text Encoder encoding and feature extraction (no gradient needed)
        teacher_unet_internal_dtype = next(current_teacher_unet.parameters()).dtype
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
                teacher_text_output, teacher_text_features = text_encoder_feature_extractor.extract_features(
                    text_encoder=current_teacher_text_encoder,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_grad=False
                )
                teacher_encoder_hidden_states = teacher_text_output.to(dtype=teacher_unet_internal_dtype)

        # 🔥 Student Text Encoder encoding and feature extraction (gradient needed)
        student_unet_internal_dtype = next(student_unet.parameters()).dtype
        with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
            student_text_output, student_text_features = text_encoder_feature_extractor.extract_features(
                text_encoder=student_text_encoder,
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                return_dict=True,
                use_grad=True
            )
            student_encoder_hidden_states = student_text_output.to(dtype=student_unet_internal_dtype)

        # Noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

        # 🔥 Teacher UNet forward pass (extract features)
        with torch.no_grad():
            teacher_noise_pred, teacher_unet_features = current_teacher_extractor.extract_features(
                unet_model=current_teacher_unet,
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=teacher_encoder_hidden_states,
                return_dict=unet_return_dict,
                use_grad=False
            )

        # 🔥 Student UNet forward pass (extract features)
        _, student_unet_features = student_extractor.extract_features(
            unet_model=student_unet,
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=student_encoder_hidden_states,
            return_dict=unet_return_dict,
            use_grad=True
        )

        # 🔥 Compute UNet hybrid feature alignment loss
        valid_teacher_unet_features = isinstance(teacher_unet_features, dict) and teacher_unet_features
        valid_student_unet_features = isinstance(student_unet_features, dict) and student_unet_features

        if valid_teacher_unet_features and valid_student_unet_features:
            # Use new hybrid loss function, returns 3 values
            unet_feature_align_loss, unet_feature_loss_dict, unet_detailed_loss_info = unet_feature_alignment_loss_fn(
                teacher_unet_features, student_unet_features
            )
        else:
            unet_feature_align_loss = torch.tensor(0.0, device=noise_pred_loss.device, dtype=noise_pred_loss.dtype)
            unet_feature_loss_dict = {}
            unet_detailed_loss_info = {}

        # 🔥 Compute Text Encoder hybrid feature alignment loss
        valid_teacher_text_features = isinstance(teacher_text_features, dict) and teacher_text_features
        valid_student_text_features = isinstance(student_text_features, dict) and student_text_features

        if valid_teacher_text_features and valid_student_text_features:
            # Use new hybrid loss function, returns 3 values
            text_encoder_feature_align_loss, text_encoder_feature_loss_dict, text_detailed_loss_info = text_encoder_feature_alignment_loss_fn(
                teacher_features=teacher_text_features,
                student_features=student_text_features,
                teacher_attention_mask=attention_mask,
                student_attention_mask=student_attention_mask
            )
        else:
            text_encoder_feature_align_loss = torch.tensor(0.0, device=noise_pred_loss.device, dtype=noise_pred_loss.dtype)
            text_encoder_feature_loss_dict = {}
            text_detailed_loss_info = {}

        # Record loss values for monitoring
        current_noise_loss_item = noise_pred_loss.item()
        current_unet_feature_loss_item = unet_feature_align_loss.item() if isinstance(unet_feature_align_loss, torch.Tensor) else unet_feature_align_loss
        current_text_encoder_feature_loss_item = text_encoder_feature_align_loss.item() if isinstance(text_encoder_feature_align_loss, torch.Tensor) else text_encoder_feature_align_loss

        # 🔥 Extract hybrid loss detailed information
        unet_primary_loss = 0.0
        unet_secondary_loss = 0.0
        text_primary_loss = 0.0
        text_secondary_loss = 0.0
        
        # Extract primary and secondary losses from detailed loss info
        for layer_name, loss_value in unet_detailed_loss_info.items():
            if 'primary' in layer_name and isinstance(loss_value, (int, float)):
                unet_primary_loss += loss_value
            elif 'secondary' in layer_name and isinstance(loss_value, (int, float)):
                unet_secondary_loss += loss_value
        
        for layer_name, loss_value in text_detailed_loss_info.items():
            if 'primary' in layer_name and isinstance(loss_value, (int, float)):
                text_primary_loss += loss_value
            elif 'secondary' in layer_name and isinstance(loss_value, (int, float)):
                text_secondary_loss += loss_value

        # === 🔥 Alternating optimization logic ===
        if use_alternating_optimization:
            # Update loss history
            total_feature_loss = current_unet_feature_loss_item + current_text_encoder_feature_loss_item
            alt_controller.update_loss_history(current_noise_loss_item, total_feature_loss)
            
            # Check if mode switch is needed
            if alt_controller.should_switch_mode(global_step):
                alt_controller.switch_mode(global_step)
            
            # Get current optimization mode
            current_mode = alt_controller.get_current_mode(global_step)
            
            # Select loss to optimize based on current mode
            optimizer.zero_grad()
            
            if current_mode == "noise":
                # Only optimize noise prediction loss
                actual_loss = noise_pred_loss
                loss_for_display = {
                    'optimizing': 'noise',
                    'active_loss': current_noise_loss_item,
                    'monitoring_loss': total_feature_loss
                }
            else:  # feature mode
                # Optimize feature alignment losses
                actual_loss = (
                    feature_align_weight * unet_feature_align_loss +
                    text_encoder_feature_align_weight * text_encoder_feature_align_loss
                )
                loss_for_display = {
                    'optimizing': 'feature',
                    'active_loss': actual_loss.item(),
                    'monitoring_loss': current_noise_loss_item
                }
            
            # Backpropagation and optimization
            actual_loss.backward()
            optimizer.step()
            lr_scheduler_lora.step()
            
            # Record the actually optimized loss as total_loss
            current_total_loss_item = actual_loss.item()
            
        else:
            # Traditional joint optimization
            from dynamic_weight import create_tuning_weight_adjuster
            
            tuning_weight_adjuster = create_tuning_weight_adjuster(
                noise_pred_weight=noise_pred_weight,
                unet_feature_align_weight=feature_align_weight,
                text_encoder_feature_align_weight=text_encoder_feature_align_weight
            )

            current_losses = {
                'noise_pred': current_noise_loss_item,
                'unet_feature_align': current_unet_feature_loss_item,
                'text_encoder_feature_align': current_text_encoder_feature_loss_item,
            }

            updated_weights = tuning_weight_adjuster.update_weights(current_losses)
            noise_pred_weight = updated_weights['noise_pred']
            feature_align_weight = updated_weights.get('unet_feature_align', feature_align_weight)
            text_encoder_feature_align_weight = updated_weights.get('text_encoder_feature_align', text_encoder_feature_align_weight)

            # Total loss
            total_loss = (
                noise_pred_weight * noise_pred_loss +
                feature_align_weight * unet_feature_align_loss +
                text_encoder_feature_align_weight * text_encoder_feature_align_loss
            )

            # Backpropagation and optimizer steps
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler_lora.step()
            
            current_total_loss_item = total_loss.item()
            current_mode = "joint"
            loss_for_display = {
                'optimizing': 'joint',
                'active_loss': current_total_loss_item,
                'monitoring_loss': 0.0
            }

        # === 🔥 Record statistics ===
        # Update current teacher's hybrid loss tracker
        teacher_loss_trackers[current_teacher_idx]['noise_pred_loss'] += current_noise_loss_item
        teacher_loss_trackers[current_teacher_idx]['unet_feature_align_loss'] += current_unet_feature_loss_item
        teacher_loss_trackers[current_teacher_idx]['text_encoder_feature_align_loss'] += current_text_encoder_feature_loss_item
        teacher_loss_trackers[current_teacher_idx]['total_loss'] += current_total_loss_item
        teacher_loss_trackers[current_teacher_idx]['step_count'] += 1
        # 🔥 New hybrid loss tracking
        teacher_loss_trackers[current_teacher_idx]['unet_primary_loss'] += unet_primary_loss
        teacher_loss_trackers[current_teacher_idx]['unet_secondary_loss'] += unet_secondary_loss
        teacher_loss_trackers[current_teacher_idx]['text_primary_loss'] += text_primary_loss
        teacher_loss_trackers[current_teacher_idx]['text_secondary_loss'] += text_secondary_loss

        # Record global accumulated losses
        accumulated_total_loss += current_total_loss_item
        accumulated_noise_loss += current_noise_loss_item
        accumulated_unet_feature_loss += current_unet_feature_loss_item
        accumulated_text_encoder_feature_loss += current_text_encoder_feature_loss_item
        step_count_in_interval += 1

        current_lr = lr_scheduler_lora.get_last_lr()[0]
        
        # 🔥 Display hybrid loss progress information
        mode_display = current_mode.upper()
        progress_bar.set_postfix({
            'mode': mode_display,
            'active_loss': f'{loss_for_display["active_loss"]:.4f}',
            'noise_loss': f'{current_noise_loss_item:.4f}',
            'unet_feat': f'{current_unet_feature_loss_item:.4f}',
            'text_feat': f'{current_text_encoder_feature_loss_item:.4f}',
            'lr': f'{current_lr:.2e}',
            'teacher': teacher_name_log
        })
        progress_bar.update(1)

        # === 🔥 Detailed TensorBoard logging ===
        if writer:
            # Basic loss logging
            writer.add_scalar('loss/noise_prediction_loss_per_step', current_noise_loss_item, global_step)
            writer.add_scalar('loss/UNet_feature_alignment_loss_per_step', current_unet_feature_loss_item, global_step)
            writer.add_scalar('loss/TextEncoder_feature_alignment_loss_per_step', current_text_encoder_feature_loss_item, global_step)
            writer.add_scalar('loss/actual_training_loss_per_step', current_total_loss_item, global_step)
            writer.add_scalar('learning_rate/LoRA_learning_rate', current_lr, global_step)
            writer.add_scalar('teacher_model/current_selection', current_teacher_idx + 1, global_step)
            
            # 🔥 Hybrid loss decomposition logging
            writer.add_scalar('hybrid_loss_UNet/primary_loss(MSE)', unet_primary_loss, global_step)
            writer.add_scalar('hybrid_loss_UNet/secondary_loss(Cosine)', unet_secondary_loss, global_step)
            writer.add_scalar('hybrid_loss_Text/primary_loss(MSE)', text_primary_loss, global_step)
            writer.add_scalar('hybrid_loss_Text/secondary_loss(Cosine)', text_secondary_loss, global_step)
            
            # Alternating optimization specific logging
            if use_alternating_optimization:
                writer.add_scalar('alternating_optimization/current_mode', 1 if current_mode == "noise" else 2, global_step)  # 1=noise, 2=feature
                writer.add_scalar('alternating_optimization/mode_switch_count', alt_controller.mode_switch_count, global_step)
                writer.add_scalar('alternating_optimization/steps_since_last_switch', global_step - alt_controller.last_switch_step, global_step)
                
                # Record optimization effect trends
                if len(alt_controller.noise_loss_history) >= 5:
                    recent_noise_trend = (alt_controller.noise_loss_history[-1] - alt_controller.noise_loss_history[-5]) / (alt_controller.noise_loss_history[-5] + 1e-8)
                    writer.add_scalar('alternating_optimization/noise_loss_trend', recent_noise_trend, global_step)
                
                if len(alt_controller.feature_loss_history) >= 5:
                    recent_feature_trend = (alt_controller.feature_loss_history[-1] - alt_controller.feature_loss_history[-5]) / (alt_controller.feature_loss_history[-5] + 1e-8)
                    writer.add_scalar('alternating_optimization/feature_loss_trend', recent_feature_trend, global_step)
            else:
                # Weight logging for traditional mode
                writer.add_scalar('weight/noise_prediction_weight', noise_pred_weight, global_step)
                writer.add_scalar('weight/UNet_feature_alignment_weight', feature_align_weight, global_step)
                writer.add_scalar('weight/TextEncoder_feature_alignment_weight', text_encoder_feature_align_weight, global_step)

            # Hybrid loss logging categorized by teacher
            teacher_name_for_tb = f"Teacher_{current_teacher_idx+1:02d}"
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/noise_prediction_loss_per_step', current_noise_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/UNet_feature_alignment_loss_per_step', current_unet_feature_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/TextEncoder_feature_alignment_loss_per_step', current_text_encoder_feature_loss_item, global_step)
            writer.add_scalar(f'teacher_loss/{teacher_name_for_tb}/total_loss_per_step', current_total_loss_item, global_step)
            
            # 🔥 Current teacher's hybrid loss decomposition
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/UNet_primary_loss_per_step', unet_primary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/UNet_secondary_loss_per_step', unet_secondary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/Text_primary_loss_per_step', text_primary_loss, global_step)
            writer.add_scalar(f'teacher_hybrid_loss/{teacher_name_for_tb}/Text_secondary_loss_per_step', text_secondary_loss, global_step)
            
            # 🔥 Log per-layer hybrid feature loss detailed information
            if isinstance(unet_detailed_loss_info, dict):
                for layer_name, layer_loss in unet_detailed_loss_info.items():
                    clean_layer_name = layer_name.replace(".", "_")
                    
                    # 🔥 Safe type conversion function
                    def safe_convert_to_float(value, layer_name):
                        """Safely convert a value to float for TensorBoard logging"""
                        if isinstance(value, (int, float)):
                            return float(value)
                        elif isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                return value.item()
                            else:
                                print(f"⚠️ Skipping multi-element tensor: {layer_name} (shape: {value.shape})")
                                return None
                        elif isinstance(value, str):
                            # Check if it's a numeric string
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                # If it is a config string (e.g., "hybrid"), silently skip
                                if value not in ['hybrid', 'mse', 'cosine', 'scale_aware_cosine', 'layer_adaptive']:
                                    print(f"⚠️ Cannot convert string to float: {layer_name}={value}")
                                return None
                        else:
                            print(f"⚠️ Unknown type cannot be converted: {layer_name}={value} (type: {type(value)})")
                            return None
                    
                    # Try to convert and log
                    converted_value = safe_convert_to_float(layer_loss, layer_name)
                    if converted_value is not None:
                        writer.add_scalar(f'UNet_hybrid_feature_loss_layer/{clean_layer_name}', 
                                        converted_value, global_step)
                        # Log categorized by teacher
                        writer.add_scalar(f'teacher_UNet_hybrid_layer_loss/{teacher_name_for_tb}/{clean_layer_name}', 
                                        converted_value, global_step)
            
            if isinstance(text_detailed_loss_info, dict):
                for layer_name, layer_loss in text_detailed_loss_info.items():
                    clean_layer_name = layer_name.replace(".", "_")
                    
                    # Use the same safe conversion function
                    converted_value = safe_convert_to_float(layer_loss, layer_name)
                    if converted_value is not None:
                        writer.add_scalar(f'TextEncoder_hybrid_feature_loss_layer/{clean_layer_name}', 
                                        converted_value, global_step)
                        # Log categorized by teacher
                        writer.add_scalar(f'teacher_TextEncoder_hybrid_layer_loss/{teacher_name_for_tb}/{clean_layer_name}', 
                                        converted_value, global_step)

        global_step += 1

        # --- 🔥 Save checkpoint (including hybrid loss information) ---
        if global_step > 0 and global_step % save_steps == 0:
            # Compute interval average loss
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_noise_loss_interval = accumulated_noise_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_unet_feature_loss_interval = accumulated_unet_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_text_encoder_feature_loss_interval = accumulated_text_encoder_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0

            print(f"\n" + "="*60)
            print(f"🔥 MULTI-TEACHER hybrid loss LoRA CHECKPOINT - Step {global_step}/{num_steps}")
            if use_alternating_optimization:
                print(f"Alternating optimization mode - currently optimizing: {current_mode.upper()}")
                print(f"Mode switch count: {alt_controller.mode_switch_count}")
            else:
                print(f"Joint hybrid loss optimization mode")
            print(f"="*60)
            
            print(f"Average Losses (last {step_count_in_interval} steps):")
            print(f"  • Actual Training Loss:      {avg_total_loss_interval:.6f}")
            print(f"  • Noise Prediction:          {avg_noise_loss_interval:.6f}")
            print(f"  • UNet Feature Alignment:    {avg_unet_feature_loss_interval:.6f}")
            print(f"  • TextEncoder Feature Align: {avg_text_encoder_feature_loss_interval:.6f}")
            print(f"  • Learning Rate:             {current_lr:.2e}")
            print(f"  • Current Teacher:           Teacher {current_teacher_idx + 1}/{num_teachers}")

            # 🔥 Hybrid loss configuration display
            print(f"\n🔥 Hybrid Loss Configuration Summary:")
            print(f"  • UNet hybrid loss type: {unet_feature_loss_type}")
            if unet_feature_loss_type == "hybrid":
                print(f"    - Primary loss ({unet_primary_loss_type}): {100*(1-unet_loss_combination_weight):.1f}%")
                print(f"    - Secondary loss ({unet_secondary_loss_type}): {100*unet_loss_combination_weight:.1f}%")
            print(f"  • TextEncoder hybrid loss type: {text_encoder_loss_type}")
            if text_encoder_loss_type == "hybrid":
                print(f"    - Primary loss ({text_encoder_primary_loss_type}): {100*(1-text_encoder_loss_combination_weight):.1f}%")
                print(f"    - Secondary loss ({text_encoder_secondary_loss_type}): {100*text_encoder_loss_combination_weight:.1f}%")

            # Alternating optimization statistics
            if use_alternating_optimization:
                noise_steps = sum(1 for i in range(max(0, global_step - step_count_in_interval), global_step) 
                                if alt_controller.get_current_mode(i) == "noise")
                feature_steps = step_count_in_interval - noise_steps
                print(f"  • Optimization Distribution:")
                print(f"    - Noise Steps:             {noise_steps} ({noise_steps/step_count_in_interval*100:.1f}%)")
                print(f"    - Feature Steps:           {feature_steps} ({feature_steps/step_count_in_interval*100:.1f}%)")

            # 🔥 Teacher hybrid loss statistics
            print(f"\n📊 Teacher Mixed Loss Performance Summary:")
            for i, tracker in enumerate(teacher_loss_trackers):
                teacher_name = f"Teacher {i+1}"
                step_count = tracker['step_count']
                
                if step_count > 0:
                    avg_noise_loss_teacher = tracker['noise_pred_loss'] / step_count
                    avg_unet_feature_loss_teacher = tracker['unet_feature_align_loss'] / step_count
                    avg_text_encoder_feature_loss_teacher = tracker['text_encoder_feature_align_loss'] / step_count
                    avg_total_loss_teacher = tracker['total_loss'] / step_count
                    teacher_usage_frequency = step_count / global_step
                    
                    # 🔥 Hybrid loss statistics
                    avg_unet_primary = tracker['unet_primary_loss'] / step_count
                    avg_unet_secondary = tracker['unet_secondary_loss'] / step_count
                    avg_text_primary = tracker['text_primary_loss'] / step_count
                    avg_text_secondary = tracker['text_secondary_loss'] / step_count
                    
                    print(f"  📊 {teacher_name} Hybrid Loss Statistics:")
                    print(f"    • Usage count: {step_count}/{global_step} ({teacher_usage_frequency:.2%})")
                    print(f"    • Average noise prediction loss: {avg_noise_loss_teacher:.6f}")
                    print(f"    • 🔥 UNet hybrid loss: primary={avg_unet_primary:.6f}, secondary={avg_unet_secondary:.6f}")
                    print(f"    • 🔥 Text hybrid loss: primary={avg_text_primary:.6f}, secondary={avg_text_secondary:.6f}")
                    print(f"    • Average total loss: {avg_total_loss_teacher:.6f}")
                else:
                    print(f"  📊 {teacher_name}: Not used")

            # Save model
            current_save_dir = os.path.join(save_path, out_name)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir, exist_ok=True)

            checkpoint_path = os.path.join(current_save_dir, f"multi_teacher_hybrid_lora_step_{global_step}.safetensors")
            save_all(
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=checkpoint_path,
                target_replace_module_unet=lora_unet_target_modules,
                target_replace_module_text=lora_clip_target_modules
            )
            print(f"✓ Multi-Teacher Hybrid LoRA Checkpoint saved to: {checkpoint_path}\n")

            # Reset accumulated losses
            accumulated_total_loss = 0.0
            accumulated_noise_loss = 0.0
            accumulated_unet_feature_loss = 0.0
            accumulated_text_encoder_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    
    # === 🔥 Final hybrid loss statistics report ===
    print(f"\n" + "="*80)
    print(f"Multi-TEACHER {'alternating' if use_alternating_optimization else 'joint'} hybrid loss LoRA training completed")
    print(f"="*80)
    
    if use_alternating_optimization:
        total_noise_steps = sum(1 for i in range(global_step) if alt_controller.get_current_mode(i) == "noise")
        total_feature_steps = global_step - total_noise_steps
        print(f"🔄 Optimization Statistics:")
        print(f"  • Total steps: {global_step}")
        print(f"  • Noise optimization steps: {total_noise_steps} ({total_noise_steps/global_step*100:.1f}%)")
        print(f"  • Feature optimization steps: {total_feature_steps} ({total_feature_steps/global_step*100:.1f}%)")
        print(f"  • Mode switch count: {alt_controller.mode_switch_count}")
        print(f"  • Final optimization mode: {alt_controller.current_mode.upper()}")

    # 🔥 Log final teacher hybrid loss statistics to TensorBoard
    if writer:
        for i, tracker in enumerate(teacher_loss_trackers):
            teacher_name = f"Teacher_{i+1:02d}"
            step_count = tracker['step_count']
            
            if step_count > 0:
                # Original statistics
                final_avg_noise_loss = tracker['noise_pred_loss'] / step_count
                final_avg_unet_loss = tracker['unet_feature_align_loss'] / step_count
                final_avg_text_loss = tracker['text_encoder_feature_align_loss'] / step_count
                final_avg_total_loss = tracker['total_loss'] / step_count
                final_usage_freq = step_count / global_step
                
                # 🔥 New hybrid loss statistics
                final_avg_unet_primary = tracker['unet_primary_loss'] / step_count
                final_avg_unet_secondary = tracker['unet_secondary_loss'] / step_count
                final_avg_text_primary = tracker['text_primary_loss'] / step_count
                final_avg_text_secondary = tracker['text_secondary_loss'] / step_count
                
                # Log final statistics
                writer.add_scalar(f'FinalStats/{teacher_name}/FinalAvgNoiseLoss', final_avg_noise_loss, global_step)
                writer.add_scalar(f'FinalStats/{teacher_name}/FinalAvgUNetLoss', final_avg_unet_loss, global_step)
                writer.add_scalar(f'FinalStats/{teacher_name}/FinalAvgTextLoss', final_avg_text_loss, global_step)
                writer.add_scalar(f'FinalStats/{teacher_name}/FinalAvgTotalLoss', final_avg_total_loss, global_step)
                writer.add_scalar(f'FinalStats/{teacher_name}/FinalUsageFrequency', final_usage_freq, global_step)
                
                # 🔥 Final hybrid loss statistics
                writer.add_scalar(f'FinalHybridStats/{teacher_name}/UNetPrimaryLossMean', final_avg_unet_primary, global_step)
                writer.add_scalar(f'FinalHybridStats/{teacher_name}/UNetSecondaryLossMean', final_avg_unet_secondary, global_step)
                writer.add_scalar(f'FinalHybridStats/{teacher_name}/TextPrimaryLossMean', final_avg_text_primary, global_step)
                writer.add_scalar(f'FinalHybridStats/{teacher_name}/TextSecondaryLossMean', final_avg_text_secondary, global_step)
                
                print(f"\n📈 {teacher_name} Final Hybrid Loss Statistics:")
                print(f"   Total usage count: {step_count}/{global_step} ({final_usage_freq:.2%})")
                print(f"   Final average noise loss: {final_avg_noise_loss:.6f}")
                print(f"   🔥 UNet hybrid loss: primary={final_avg_unet_primary:.6f}, secondary={final_avg_unet_secondary:.6f}")
                print(f"   🔥 Text hybrid loss: primary={final_avg_text_primary:.6f}, secondary={final_avg_text_secondary:.6f}")
                print(f"   Final average total loss: {final_avg_total_loss:.6f}")

    writer.close()

    # --- Save final model ---
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_multi_teacher_hybrid_lora_step_{global_step}.safetensors")
    save_all(
        unet=student_unet,
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path,
        target_replace_module_unet=lora_unet_target_modules,
        target_replace_module_text=lora_clip_target_modules
    )
    
    print(f"✓ Final multi-teacher hybrid LoRA model saved to: {final_model_path}")
    print(f"✓ TensorBoard logs saved to: {tb_log_path}")
    print(f"🔥 Multi-teacher hybrid loss training completed with {'alternating' if use_alternating_optimization else 'joint'} optimization!")
    print(f"🔥 Hybrid loss config: UNet({unet_feature_loss_type}), Text({text_encoder_loss_type})")
    print(f"="*80)

def generate_placeholder_tokens_from_lora_names(teacher_info: List[dict]):
    """
    Generate placeholder tokens from LoRA filenames
    
    Args:
        teacher_info: List containing LoRA information
    
    Returns:
        all_placeholder_tokens: List of all placeholder tokens
        all_initializer_tokens: List of all initializer tokens
    """
    all_placeholder_tokens = []
    all_initializer_tokens = []
    
    for info in teacher_info:
        model_name = info["name"]
        # Clean filename to generate a valid token
        clean_name = clean_filename_for_token(model_name)
        placeholder_token = f"<{clean_name}>"
        
        all_placeholder_tokens.append(placeholder_token)
        all_initializer_tokens.append("<rand-0.017>")  # Random initialization
    
    return all_placeholder_tokens, all_initializer_tokens

def clean_filename_for_token(filename: str) -> str:
    """
    Clean filename to generate a valid token name
    """
    import re
    # Remove special characters, keep only letters, digits, and underscores
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    # Remove consecutive underscores
    clean = re.sub(r'_+', '_', clean)
    # Remove leading and trailing underscores
    clean = clean.strip('_')
    # Ensure not empty
    if not clean:
        clean = "model"
    return clean

def parse_manual_tokens(placeholder_tokens1: str, placeholder_tokens2: str, initializer_tokens: str):
    """
    Parse manually specified tokens (backward compatibility)
    """
    all_placeholder_tokens = []
    all_initializer_tokens = []
    
    if placeholder_tokens1:
        all_placeholder_tokens.extend(placeholder_tokens1.split("|"))
    if placeholder_tokens2:
        all_placeholder_tokens.extend(placeholder_tokens2.split("|"))
    
    if initializer_tokens:
        all_initializer_tokens = initializer_tokens.split("|")
    else:
        all_initializer_tokens = ["<rand-0.017>"] * len(all_placeholder_tokens)
    
    return all_placeholder_tokens, all_initializer_tokens

def create_student_model(pretrained_model_name_or_path, pretrained_vae_name_or_path, revision, 
                        placeholder_tokens, initializer_tokens, device):
    """
    Create student model
    """
    # Use modified get_models function to support arbitrary number of tokens
    return get_models_multi_tokens(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pretrained_vae_name_or_path=pretrained_vae_name_or_path,
        revision=revision,
        placeholder_tokens=placeholder_tokens,
        initializer_tokens=initializer_tokens,
        device=device
    )

def get_models_multi_tokens(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device="cuda:0",
):
    """
    Model creation function supporting arbitrary number of tokens
    """
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)
        placeholder_token_ids.append(placeholder_token_id)

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        
        if init_tok.startswith("<rand"):
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])
            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), "
                f"empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- "
                f"{token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")
            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
    )

def create_multiple_teacher_models(teacher_lora_paths: List[str], pretrained_model_name_or_path: str, device: str):
    """
    Create multiple teacher models
    """
    teacher_models = []
    
    for i, lora_path in enumerate(teacher_lora_paths):
        print(f"Loading teacher model {i+1}/{len(teacher_lora_paths)}: {lora_path}")
        
        # Create pipeline
        teacher_pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.float16
        ).to(device)
        
        teacher_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            teacher_pipe.scheduler.config
        )
        
        # Load LoRA weights
        patch_pipe(
            teacher_pipe,
            lora_path,
            patch_text=True,
            patch_ti=True,
            patch_unet=True,
        )
        
        tune_lora_scale(teacher_pipe.unet, 1.0)
        tune_lora_scale(teacher_pipe.text_encoder, 1.0)
        
        teacher_models.append({
            "unet": teacher_pipe.unet,
            "text_encoder": teacher_pipe.text_encoder,
            "vae": teacher_pipe.vae,
            "tokenizer": teacher_pipe.tokenizer,
            "pipeline": teacher_pipe
        })
        
        print(f"✓ Teacher model {i+1} loaded successfully")
    
    return teacher_models

def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
                               gradient_checkpointing, enable_xformers_memory_efficient_attention,
                               placeholder_token_ids, enable_adaptive_gradient_clipping=True,
                               enable_mixed_precision=True, train_vae_decoder=False):
    """
    Set up model training configuration - enhanced version
    """
    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
        # Enable gradient checkpointing for text encoder to save GPU memory
        if hasattr(student_text_encoder, 'gradient_checkpointing_enable'):
            student_text_encoder.gradient_checkpointing_enable()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            student_unet.enable_xformers_memory_efficient_attention()
            # If text encoder supports xformers, enable it too
            if hasattr(student_text_encoder, 'enable_xformers_memory_efficient_attention'):
                try:
                    student_text_encoder.enable_xformers_memory_efficient_attention()
                except:
                    print("Text encoder xformers attention failed, continuing...")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Basic setup
    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    # Optional: train partial VAE decoder layers to improve reconstruction quality
    if train_vae_decoder:
        # Only train the last few decoder layers
        for name, param in student_vae.named_parameters():
            if 'decoder' in name and ('conv_out' in name or 'norm_out' in name):
                param.requires_grad = True
                print(f"Enabled VAE parameter training: {name}")

    # More fine-grained text encoder freezing strategy
    # Fully freeze position embeddings and most transformer layers, but retain flexibility for some layers
    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
        # Freeze most front encoder layers, keep the last few layers trainable
        student_text_encoder.text_model.encoder.layers[:-2].parameters() if hasattr(student_text_encoder.text_model.encoder, 'layers') else [],
    )
    
    for param in params_to_freeze:
        param.requires_grad = False
    
    # Keep final_layer_norm trainable to adapt to new representations
    student_text_encoder.text_model.final_layer_norm.requires_grad_(True)

def calculate_learning_rates(learning_rate_unet, learning_rate_text, learning_rate_ti,
                           scale_lr, gradient_accumulation_steps, train_batch_size):
    """
    Calculate learning rates
    """
    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = learning_rate_text * gradient_accumulation_steps * train_batch_size
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti
    
    return unet_lr, text_encoder_lr, ti_lr

def setup_lora_parameters(student_unet, student_text_encoder, train_text_encoder,
                         continue_inversion, use_extended_lora, lora_rank,
                         lora_unet_target_modules, lora_clip_target_modules,
                         lora_dropout_p, lora_scale, unet_lr, text_encoder_lr,
                         ti_lr, continue_inversion_lr):
    """
    Set up LoRA parameters
    """
    # UNet LoRA setup
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("Using extended UNet LoRA")
        lora_unet_target_modules = lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        unet_lora_params, _ = inject_trainable_lora_extended(
            student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    # Text encoder setup
    student_text_encoder.requires_grad_(False)
    
    if continue_inversion:
        params_to_optimize += [
            {
                "params": student_text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr if continue_inversion_lr is not None else ti_lr,
            }
        ]
        student_text_encoder.requires_grad_(True)
        params_to_freeze = itertools.chain(
            student_text_encoder.text_model.encoder.parameters(),
            student_text_encoder.text_model.final_layer_norm.parameters(),
            student_text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False

    if train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            student_text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank,
        )
        params_to_optimize += [
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": text_encoder_lr,
            }
        ]

    return params_to_optimize, None

def extract_placeholder_tokens_from_lora_file(lora_file_path: str) -> List[str]:
    """
    Extract placeholder tokens from LoRA weight file
    Uses the official parse_safeloras_embeds method, which is the most reliable approach
    """
    import re
    from safetensors import safe_open
    
    try:
        print(f"Extracting placeholder tokens from {os.path.basename(lora_file_path)}...")
        
        placeholder_tokens = []
        
        if lora_file_path.endswith('.safetensors'):
            with safe_open(lora_file_path, framework="pt", device="cpu") as f:
                # Use the official parse_safeloras_embeds method
                try:
                    from lora_diffusion.lora import parse_safeloras_embeds
                    
                    # Extract all embeddings
                    embeds_dict = parse_safeloras_embeds(f)
                    
                    if embeds_dict:
                        # Get all token names
                        placeholder_tokens = list(embeds_dict.keys())
                        print(f"  ✅ Extracted {len(placeholder_tokens)} tokens via official method: {placeholder_tokens}")
                    else:
                        print("  ⚠ Official method found no embeddings")
                        
                except ImportError as e:
                    print(f"  ❌ Failed to import official parsing function: {e}")
                except Exception as e:
                    print(f"  ❌ Official method execution error: {e}")
                
                # Fallback: extract from metadata
                if not placeholder_tokens:
                    print("  🔍 Attempting to extract from metadata...")
                    metadata = f.metadata()
                    if metadata:
                        # Extract from the value of the <embed> key
                        embed_value = metadata.get('<embed>')
                        if embed_value and isinstance(embed_value, str):
                            # Handle multiple tokens case
                            separators = ['|', ',', ';', ' ', '\n']
                            for sep in separators:
                                if sep in embed_value:
                                    tokens = [t.strip() for t in embed_value.split(sep)]
                                    valid_tokens = []
                                    for token in tokens:
                                        if token.strip():
                                            if token.startswith('<') and token.endswith('>'):
                                                valid_tokens.append(token)
                                            else:
                                                clean_token = re.sub(r'[^\w\-_.]', '_', token)
                                                if clean_token:
                                                    valid_tokens.append(f"<{clean_token}>")
                                    
                                    if valid_tokens:
                                        placeholder_tokens.extend(valid_tokens)
                                        print(f"  🔍 Extracted {len(valid_tokens)} tokens from metadata: {valid_tokens}")
                                        break
                            else:
                                # Single token
                                if embed_value.startswith('<') and embed_value.endswith('>'):
                                    placeholder_tokens.append(embed_value)
                                else:
                                    clean_embed = re.sub(r'[^\w\-_.]', '_', embed_value.strip())
                                    if clean_embed:
                                        placeholder_tokens.append(f"<{clean_embed}>")
        
        else:
            # Handle .pt/.pth/.bin files
            import torch
            data = torch.load(lora_file_path, map_location="cpu")
            
            if isinstance(data, dict):
                embed_value = data.get('<embed>')
                if embed_value and isinstance(embed_value, str):
                    if embed_value.startswith('<') and embed_value.endswith('>'):
                        placeholder_tokens.append(embed_value)
                    else:
                        for sep in ['|', ',', ';', ' ']:
                            if sep in embed_value:
                                tokens = [t.strip() for t in embed_value.split(sep)]
                                valid_tokens = []
                                for token in tokens:
                                    if token.strip():
                                        if token.startswith('<') and token.endswith('>'):
                                            valid_tokens.append(token)
                                        else:
                                            clean_token = re.sub(r'[^\w\-_.]', '_', token)
                                            if clean_token:
                                                valid_tokens.append(f"<{clean_token}>")
                                placeholder_tokens.extend(valid_tokens)
                                break
                        else:
                            clean_embed = re.sub(r'[^\w\-_.]', '_', embed_value)
                            if clean_embed:
                                placeholder_tokens.append(f"<{clean_embed}>")
        
        # If none found, generate from filename
        if not placeholder_tokens:
            print(f"  ⚠ Could not extract placeholder tokens from file, generating from filename")
            filename = os.path.splitext(os.path.basename(lora_file_path))[0]
            clean_name = clean_filename_for_token(filename)
            placeholder_tokens = [f"<{clean_name}>"]
        
        # Deduplicate and validate format
        unique_tokens = []
        for token in placeholder_tokens:
            if token not in unique_tokens:
                if not (token.startswith('<') and token.endswith('>')):
                    token = f"<{token.strip('<>')}>"
                unique_tokens.append(token)
        
        print(f"  ✅ Successfully extracted {len(unique_tokens)} placeholder tokens: {unique_tokens}")
        return unique_tokens
        
    except Exception as e:
        print(f"  ❌ Error extracting placeholder tokens from {lora_file_path}: {e}")
        # Fall back to filename-based approach
        filename = os.path.splitext(os.path.basename(lora_file_path))[0]
        clean_name = clean_filename_for_token(filename)
        fallback_token = f"<{clean_name}>"
        print(f"  🔄 Using fallback to generate token: {fallback_token}")
        return [fallback_token]

def generate_placeholder_tokens_from_lora_weights(teacher_lora_paths: List[str], teacher_info: List[dict]):
    """
    Automatically extract placeholder tokens from LoRA weight files
    Uses the official parse_safeloras_embeds method
    """
    all_placeholder_tokens = []
    all_initializer_tokens = []
    teacher_token_mapping = {}
    
    print("🔍 Extracting placeholder tokens from LoRA weight files...")
    
    for i, (lora_path, info) in enumerate(zip(teacher_lora_paths, teacher_info)):
        print(f"\n📁 Processing LoRA file {i+1}/{len(teacher_lora_paths)}: {info['filename']}")
        
        # Extract tokens from file
        extracted_tokens = extract_placeholder_tokens_from_lora_file(lora_path)
        
        if extracted_tokens:
            teacher_token_mapping[i] = extracted_tokens
            all_placeholder_tokens.extend(extracted_tokens)
            all_initializer_tokens.extend(["<rand-0.017>"] * len(extracted_tokens))
            print(f"  ✓ Teacher {i+1} extracted {len(extracted_tokens)} tokens: {extracted_tokens}")
        else:
            # Fallback approach
            model_name = info["name"]
            clean_name = clean_filename_for_token(model_name)
            fallback_token = f"<{clean_name}>"
            teacher_token_mapping[i] = [fallback_token]
            
            print(f"  ⚠ No valid tokens found, generating from filename: {fallback_token}")
            all_placeholder_tokens.append(fallback_token)
            all_initializer_tokens.append("<rand-0.017>")
    
    # Deduplication
    seen = set()
    unique_placeholder_tokens = []
    unique_initializer_tokens = []
    
    for token, init_token in zip(all_placeholder_tokens, all_initializer_tokens):
        if token not in seen:
            seen.add(token)
            unique_placeholder_tokens.append(token)
            unique_initializer_tokens.append(init_token)
        else:
            print(f"  ⚠ Duplicate token skipped: {token}")
    
    print(f"\n✅ Token extraction complete!")
    print(f"📊 Statistics:")
    print(f"  • Total files: {len(teacher_lora_paths)}")
    print(f"  • Deduplicated token count: {len(unique_placeholder_tokens)}")
    
    print(f"\n🗺️ Teacher-Token Mapping:")
    for teacher_idx, tokens in teacher_token_mapping.items():
        teacher_name = teacher_info[teacher_idx]['name']
        print(f"  • Teacher {teacher_idx+1} ({teacher_name}): {tokens}")
    
    print(f"\n🏷️ Final Tokens list: {unique_placeholder_tokens}")
    
    return unique_placeholder_tokens, unique_initializer_tokens, teacher_token_mapping

def create_token_based_teachers_and_dataloaders(teacher_lora_paths: List[str], teacher_info: List[dict], 
                                               teacher_token_mapping: dict, student_tokenizer, 
                                               train_batch_size: int, cached_latents: bool,
                                               use_template, device: str, pretrained_model_name_or_path: str):
    """
    Create independent teacher models and dataloaders based on tokens
    Each token corresponds to an independent teacher, rather than each LoRA file corresponding to a teacher
    
    Returns:
        token_based_teachers: List[dict] - Teacher info for each token
        token_dataloaders: List[DataLoader] - Dataloader for each token
        token_teacher_mapping: dict - Mapping from token to teacher index
    """
    
    print(f"\n🔄 Converting to Token-based Teacher mode")
    print(f"Original mode: {len(teacher_lora_paths)} LoRA files -> New mode: one Teacher per Token")
    
    token_based_teachers = []
    token_dataloaders = []
    token_teacher_mapping = {}
    
    # Fixed number of images per token
    IMAGES_PER_TOKEN = 10
    
    # Count all tokens
    all_tokens_info = []
    token_to_source_lora = {}  # Track which original LoRA file each token comes from
    
    for lora_idx, (lora_path, info) in enumerate(zip(teacher_lora_paths, teacher_info)):
        tokens = teacher_token_mapping.get(lora_idx, [])
        for token in tokens:
            all_tokens_info.append({
                'token': token,
                'source_lora_path': lora_path,
                'source_lora_info': info,
                'source_lora_idx': lora_idx
            })
            token_to_source_lora[token] = {
                'lora_path': lora_path,
                'lora_info': info,
                'lora_idx': lora_idx
            }
    
    print(f"📊 Found {len(all_tokens_info)} independent tokens in total")
    
    # Create an independent teacher for each token
    for token_idx, token_info in enumerate(all_tokens_info):
        token = token_info['token']
        source_lora_path = token_info['source_lora_path']
        source_info = token_info['source_lora_info']
        
        print(f"\n🎯 Creating Teacher for Token {token_idx+1}/{len(all_tokens_info)}")
        print(f"  🏷️ Token: {token}")
        print(f"  📁 Source LoRA: {source_info['filename']}")
        
        # Create the teacher model for this token (reuse pipeline of the same LoRA file)
        # Check if a pipeline has already been created for this LoRA file
        existing_pipeline = None
        for existing_teacher in token_based_teachers:
            if existing_teacher.get('source_lora_path') == source_lora_path:
                existing_pipeline = existing_teacher['pipeline']
                break
        
        if existing_pipeline is not None:
            print(f"  ♻️ Reusing existing pipeline")
            teacher_pipeline = existing_pipeline
        else:
            print(f"  🔧 Creating new pipeline")
            # Create new pipeline
            teacher_pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, 
                torch_dtype=torch.float16
            ).to(device)
            
            teacher_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                teacher_pipeline.scheduler.config
            )
            
            # Load LoRA weights
            patch_pipe(
                teacher_pipeline,
                source_lora_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            
            tune_lora_scale(teacher_pipeline.unet, 1.0)
            tune_lora_scale(teacher_pipeline.text_encoder, 1.0)
        
        # Create token-specific teacher info
        token_teacher = {
            "unet": teacher_pipeline.unet,
            "text_encoder": teacher_pipeline.text_encoder,
            "vae": teacher_pipeline.vae,
            "tokenizer": teacher_pipeline.tokenizer,
            "pipeline": teacher_pipeline,
            "token": token,
            "source_lora_path": source_lora_path,
            "source_lora_info": source_info,
            "teacher_name": f"token_{token.replace('<', '').replace('>', '')}"
        }
        
        token_based_teachers.append(token_teacher)
        token_teacher_mapping[token] = token_idx
        
        print(f"  ✅ Teacher {token_idx+1} created successfully")
        
        # Create dedicated dataloader for this token
        print(f"  📊 Creating dedicated dataloader...")
        
        # Create token-specific token_map with specific dataset size
        token_map = {}
        for i in range(IMAGES_PER_TOKEN):
            token_map[f"SAMPLE_{i}"] = token  # Map each sample to the target token
        
        # Use the existing PivotalTuningDatasetCapationLoraGenerated class
        dataset = PivotalTuningDatasetCapationLoraGenerated(
            sd_pipeline=teacher_pipeline,
            device=device,
            main_tokenizer=teacher_pipeline.tokenizer,
            use_template=use_template,
            token_map=token_map,
            dataset_size=IMAGES_PER_TOKEN,
            transform_size=512,
            h_flip=True,
            aux_tokenizer1=student_tokenizer,
            save_generated_images_path="/root/lora_train/pic_token_based",
            save_image_prefix=f"token_{token_idx+1:02d}_{token.replace('<', '').replace('>', '')}"
        )
        
        # Create dataloader
        dataloader = text2img_dataloader_combined_with_latent_caching(
            train_dataset=dataset,
            train_batch_size=train_batch_size,
            main_tokenizer=teacher_pipeline.tokenizer,
            aux_tokenizer1=student_tokenizer,
            vae=teacher_pipeline.vae,
            cached_latents=cached_latents
        )
        
        token_dataloaders.append(dataloader)
        
        print(f"  ✅ Token {token_idx+1} dataloader created successfully")
        print(f"    • Dataset size: {IMAGES_PER_TOKEN}")
        print(f"    • Target Token: {token}")
        print(f"    • Save path: /root/lora_train/pic_token_based")
        
        # Test whether the dataset correctly generates content for the target token
        print(f"  🧪 Validating dataset Token mapping:")
        try:
            # Get a sample to verify
            test_sample = dataset[0]
            raw_text = test_sample.get('raw_text', 'unknown')
            print(f"    • Sample 0 text: '{raw_text}'")
            if token in raw_text:
                print(f"    ✅ Confirmed target token present: {token}")
            else:
                print(f"    ⚠️ Sample text does not contain the target token")
        except Exception as e:
            print(f"    ⚠️ Error during dataset validation: {e}")
    
    # Create new mapping from token to teacher index
    new_token_teacher_mapping = {}
    for token_idx, teacher in enumerate(token_based_teachers):
        token = teacher['token']
        new_token_teacher_mapping[token_idx] = [token]  # Each teacher corresponds to one token
    
    print(f"\n📋 Token-based Teacher creation completed:")
    print(f"  🎯 Token count: {len(all_tokens_info)}")
    print(f"  🏫 Teacher count: {len(token_based_teachers)}")
    print(f"  📊 Dataloader count: {len(token_dataloaders)}")
    
    print(f"\n🗺️ Token-Teacher Mapping:")
    for token_idx, teacher in enumerate(token_based_teachers):
        token = teacher['token']
        source_file = teacher['source_lora_info']['filename']
        print(f"  • Teacher {token_idx+1}: {token} (source: {source_file})")
    
    print(f"\n🔍 Data generation verification:")
    print(f"  • {IMAGES_PER_TOKEN} images generated per token")
    print(f"  • Images saved to: /root/lora_train/pic_token_based")
    print(f"  • Template type used: {use_template}")
    
    return token_based_teachers, token_dataloaders, new_token_teacher_mapping

def create_multiple_dataloaders_enhanced(teacher_models: List[dict], teacher_info: List[dict], 
                                       teacher_token_mapping: dict, student_tokenizer, 
                                       train_batch_size: int, cached_latents: bool,
                                       use_template, device: str,
                                       # Additional parameters
                                       use_token_based_teachers: bool = True,
                                       teacher_lora_paths: List[str] = None,
                                       pretrained_model_name_or_path: str = None):
    """
    Enhanced dataloader creation function
    Supports two modes:
    1. Original mode: one teacher per LoRA file (may contain multiple tokens)
    2. New mode: one teacher per token
    
    Uses the existing PivotalTuningDatasetCapationLoraGenerated class
    """
    
    if use_token_based_teachers:
        print("🎯 Using Token-based Teacher mode")
        if teacher_lora_paths is None or pretrained_model_name_or_path is None:
            raise ValueError("Token-based mode requires teacher_lora_paths and pretrained_model_name_or_path")
        
        # Use new token-based creation method
        return create_token_based_teachers_and_dataloaders(
            teacher_lora_paths=teacher_lora_paths,
            teacher_info=teacher_info,
            teacher_token_mapping=teacher_token_mapping,
            student_tokenizer=student_tokenizer,
            train_batch_size=train_batch_size,
            cached_latents=cached_latents,
            use_template=use_template,
            device=device,
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
    else:
        print("📁 Using LoRA file-based Teacher mode (original mode)")
        # Use original creation method, also based on the existing PivotalTuningDatasetCapationLoraGenerated class
        dataloaders = []
        
        IMAGES_PER_TOKEN = 10
        
        for i, (teacher_model, info) in enumerate(zip(teacher_models, teacher_info)):
            print(f"\n🔧 Creating dataloader for Teacher {i+1}:")
            print(f"  📁 Model: {info['name']}")
            
            teacher_tokens = teacher_token_mapping.get(i, [])
            
            if not teacher_tokens:
                teacher_tokenizer = teacher_model["tokenizer"]
                base_vocab_size = 49408
                current_vocab_size = len(teacher_tokenizer)
                
                if current_vocab_size > base_vocab_size:
                    for token_id in range(base_vocab_size, current_vocab_size):
                        token = teacher_tokenizer.decode([token_id])
                        if token.startswith('<') and token.endswith('>'):
                            teacher_tokens.append(token)
                
                if not teacher_tokens:
                    model_name = info["name"]
                    clean_name = clean_filename_for_token(model_name)
                    teacher_tokens = [f"<{clean_name}>"]
                    print(f"  ⚠ No teacher tokens found, using generated token: {teacher_tokens}")
            
            print(f"  🏷️ Teacher {i+1} Tokens: {teacher_tokens}")
            print(f"  📊 Token count: {len(teacher_tokens)}")
            
            total_tokens = len(teacher_tokens)
            dataset_size = total_tokens * IMAGES_PER_TOKEN
            
            print(f"  📸 Dataset configuration:")
            print(f"    • Images per token: {IMAGES_PER_TOKEN}")
            print(f"    • Total tokens: {total_tokens}")
            print(f"    • Total dataset size: {dataset_size}")
            
            # Create token_map - assign corresponding token to each sample
            token_map = {}
            
            for sample_idx in range(dataset_size):
                token_index = sample_idx // IMAGES_PER_TOKEN
                if token_index < len(teacher_tokens):
                    current_token = teacher_tokens[token_index]
                else:
                    current_token = teacher_tokens[sample_idx % len(teacher_tokens)]
                token_map[f"SAMPLE_{sample_idx}"] = current_token
            
            # Also add some general mappings for compatibility
            for j, token in enumerate(teacher_tokens):
                token_map[f"TOKEN_{j}"] = token
            
            print(f"  🗺️ Token mapping created with {len(token_map)} mapping entries")
            
            # Use the existing PivotalTuningDatasetCapationLoraGenerated class
            dataset = PivotalTuningDatasetCapationLoraGenerated(
                sd_pipeline=teacher_model["pipeline"],
                device=device,
                main_tokenizer=teacher_model["tokenizer"],
                use_template=use_template,
                token_map=token_map,
                dataset_size=dataset_size,
                transform_size=512,
                h_flip=True,
                aux_tokenizer1=student_tokenizer,
                save_generated_images_path="/root/lora_train/pic_lora_based",
                save_image_prefix=f"teacher_{i+1}_{info['name']}"
            )
            
            dataloader = text2img_dataloader_combined_with_latent_caching(
                train_dataset=dataset,
                train_batch_size=train_batch_size,
                main_tokenizer=teacher_model["tokenizer"],
                aux_tokenizer1=student_tokenizer,
                vae=teacher_model["vae"],
                cached_latents=cached_latents
            )
            
            dataloaders.append(dataloader)
            print(f"  ✅ Teacher {i+1} dataloader created successfully")
            
            print(f"  🔍 Expected token distribution:")
            for j, token in enumerate(teacher_tokens):
                start_idx = j * IMAGES_PER_TOKEN
                end_idx = start_idx + IMAGES_PER_TOKEN
                print(f"    • {token}: samples {start_idx}-{end_idx-1} ({IMAGES_PER_TOKEN} images)")
            
            # Validate dataset
            print(f"  🧪 Validating dataset Token distribution:")
            try:
                for test_idx in [0, dataset_size//2, dataset_size-1]:
                    if test_idx < len(dataset):
                        test_sample = dataset[test_idx]
                        raw_text = test_sample.get('raw_text', 'unknown')
                        print(f"    • Sample {test_idx}: '{raw_text[:50]}...'")
            except Exception as e:
                print(f"    ⚠️ Error during dataset validation: {e}")
        
        return dataloaders

def discover_lora_models(lora_models_dir: str, lora_path1: str = None, lora_path2: str = None):
    """
    Discover all LoRA model files in the directory
    """
    teacher_lora_paths = []
    teacher_info = []
    
    # Backward compatibility: use individual paths if provided
    if lora_path1 and lora_path2:
        teacher_lora_paths = [lora_path1, lora_path2]
        teacher_info = [
            {"name": "teacher1", "filename": os.path.basename(lora_path1)},
            {"name": "teacher2", "filename": os.path.basename(lora_path2)}
        ]
        return teacher_lora_paths, teacher_info
    
    # New approach: discover all LoRA models from directory
    if not os.path.exists(lora_models_dir):
        raise ValueError(f"LoRA model directory does not exist: {lora_models_dir}")
    
    # Supported LoRA file extensions
    supported_extensions = ['.safetensors', '.pt', '.pth', '.bin']
    
    print(f"🔍 Scanning directory: {lora_models_dir}")
    
    for filename in sorted(os.listdir(lora_models_dir)):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_path = os.path.join(lora_models_dir, filename)
            if os.path.isfile(file_path):
                # Extract model name from filename (remove extension)
                model_name = os.path.splitext(filename)[0]
                teacher_lora_paths.append(file_path)
                teacher_info.append({
                    "name": model_name,
                    "filename": filename,
                    "path": file_path
                })
                print(f"  📄 Found LoRA file: {filename}")
    
    print(f"📊 Found {len(teacher_lora_paths)} LoRA files in total")
    
    if len(teacher_lora_paths) == 0:
        raise ValueError(f"No supported LoRA files found in {lora_models_dir}")
    
    return teacher_lora_paths, teacher_info

# Modified main training function with new parameters
def train(
    # --- Main parameters ---
    lora_models_dir: str,
    lora_path1: str = None,
    lora_path2: str = None,
    auto_extract_tokens_from_weights: bool = True,
    auto_generate_placeholder_tokens: bool = False,
    use_token_based_teachers: bool = True,  # Default to token-based teacher mode
    
    # --- Base model configuration ---
    instance_data_dir: str = "",
    pretrained_model_name_or_path: str = "",
    output_dir: str = "",
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    
    # --- Training mode configuration ---
    perform_inversion: bool = False,
    use_template: Literal[None, "object", "style"] = None,
    train_inpainting: bool = False,
    
    # --- Token configuration ---
    placeholder_tokens: str = "",
    placeholder_tokens1: str = "",
    placeholder_tokens2: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    
    # --- Teacher selection strategy ---
    teacher_selection_strategy: str = "round_robin",
    teacher_weights: Optional[List[float]] = None,
    
    # --- Basic training parameters ---
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 1000,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    
    # --- LoRA configuration ---
    lora_rank: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPSdpaAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    
    # --- Learning rate configuration ---
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    scale_lr: bool = False,
    
    # --- Learning rate scheduler ---
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    
    # --- Data processing configuration ---
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    
    # --- Optimizer configuration ---
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    
    # --- Device and system configuration ---
    device: str = "cuda:0",
    extra_args: Optional[dict] = None,
    enable_xformers_memory_efficient_attention: bool = False,
    
    # --- Logging configuration ---
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    out_name: str = "final_lora_log2",
    tensorboard_log_dir: str = "runs_new",
    
    # --- Loss weight configuration ---
    feature_align_weight: float = 0.01,
    noise_pred_weight: float = 1.0,
    
    # --- Alternating optimization parameters ---
    use_alternating_optimization: bool = True,
    alternating_interval: int = 5,
    alternating_schedule: str = "fixed",
    noise_only_steps: int = 100,
    feature_only_steps: int = 0,
    
    # --- Data visualization parameters ---
    enable_dataloader_visualization: bool = False,
    visualization_samples_per_teacher: int = 2,
):
    """
    Training function supporting multi-teacher distillation
    
    Features:
    1. Supports Token-based Teacher mode (one teacher per token) and LoRA file-based Teacher mode
    2. Automatic token extraction from LoRA weight files
    3. Supports multiple teacher selection strategies
    4. Integrated feature alignment loss
    5. Supports alternating optimization strategy
    6. Optional dataloader visualization
    
    Args:
        use_token_based_teachers: Whether to use token-based teacher mode
        auto_extract_tokens_from_weights: Whether to automatically extract tokens from weight files
        teacher_selection_strategy: Teacher selection strategy ("round_robin", "weighted_random", "adaptive")
        use_alternating_optimization: Whether to use alternating optimization (alternating noise loss and feature loss)
        enable_dataloader_visualization: Whether to enable dataloader visualization
    """
    
    # --- 1. Initialization and basic setup ---
    print(f"\n{'='*80}")
    print(f"Multi-Teacher LoRA Distillation Training Started")
    print(f"Mode: {'Token-based Teacher' if use_token_based_teachers else 'LoRA file-based Teacher'}")
    print(f"Alternating optimization: {'enabled' if use_alternating_optimization else 'disabled'}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"multi_teacher_distill_{out_name}",
            reinit=True,
            config={
                "use_token_based_teachers": use_token_based_teachers,
                "teacher_selection_strategy": teacher_selection_strategy,
                "use_alternating_optimization": use_alternating_optimization,
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # --- 2. Discover and load LoRA models ---
    teacher_lora_paths, teacher_info = discover_lora_models(lora_models_dir, lora_path1, lora_path2)
    num_lora_files = len(teacher_lora_paths)
    
    print(f"\n📋 Found {num_lora_files} LoRA files:")
    for i, (path, info) in enumerate(zip(teacher_lora_paths, teacher_info)):
        print(f"  📁 LoRA {i+1}: {info['filename']}")

    if num_lora_files == 0:
        raise ValueError(f"No valid LoRA model files found at the specified location")

    # --- 3. Extract placeholder tokens ---
    if auto_extract_tokens_from_weights:
        print("🔍 Auto-extraction of placeholder tokens from LoRA weight files enabled")
        all_placeholder_tokens, all_initializer_tokens, original_teacher_token_mapping = generate_placeholder_tokens_from_lora_weights(
            teacher_lora_paths, teacher_info
        )
    elif auto_generate_placeholder_tokens:
        print("📝 Filename-based placeholder tokens generation enabled")
        all_placeholder_tokens, all_initializer_tokens = generate_placeholder_tokens_from_lora_names(teacher_info)
        original_teacher_token_mapping = {i: [all_placeholder_tokens[i]] for i in range(len(all_placeholder_tokens))}
    else:
        print("✋ Using manually specified placeholder tokens")
        all_placeholder_tokens, all_initializer_tokens = parse_manual_tokens(
            placeholder_tokens1, placeholder_tokens2, initializer_tokens
        )
        original_teacher_token_mapping = {i: [all_placeholder_tokens[i]] for i in range(len(all_placeholder_tokens))}

    # Validate token count
    print(f"\n📋 Token Extraction Results:")
    print(f"  🏷️ All Tokens: {all_placeholder_tokens}")
    print(f"  📊 LoRA file count: {num_lora_files}, Total token count: {len(all_placeholder_tokens)}")

    # --- 4. Create student model ---
    print(f"\n🎓 Creating student model...")
    student_text_encoder, student_vae, student_unet, student_tokenizer, placeholder_token_ids = create_student_model(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pretrained_vae_name_or_path=pretrained_vae_name_or_path,
        revision=revision,
        placeholder_tokens=all_placeholder_tokens,
        initializer_tokens=all_initializer_tokens,
        device=device
    )
    print(f"✅ Student model created with {len(all_placeholder_tokens)} new tokens")

    # --- 5. Create teacher models and dataloaders based on mode ---
    if use_token_based_teachers:
        print(f"\n🎯 Using Token-based Teacher mode")
        print(f"Each Token will become an independent Teacher")
        
        # Directly create token-based teachers and dataloaders
        teacher_models, dataloaders, teacher_token_mapping = create_multiple_dataloaders_enhanced(
            teacher_models=None,  # No pre-created teacher_models needed
            teacher_info=teacher_info,
            teacher_token_mapping=original_teacher_token_mapping,
            student_tokenizer=student_tokenizer,
            train_batch_size=train_batch_size,
            cached_latents=cached_latents,
            use_template=use_template,
            device=device,
            use_token_based_teachers=True,
            teacher_lora_paths=teacher_lora_paths,
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        
        num_teachers = len(teacher_models)
        print(f"📊 Final result: {num_teachers} Token-based Teachers")
        
    else:
        print(f"\n📁 Using LoRA file-based Teacher mode (original mode)")
        print(f"Each LoRA file corresponds to one Teacher (may contain multiple tokens)")
        
        # Original mode: create teacher models first, then dataloaders
        teacher_models = create_multiple_teacher_models(
            teacher_lora_paths=teacher_lora_paths,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device
        )
        
        dataloaders = create_multiple_dataloaders_enhanced(
            teacher_models=teacher_models,
            teacher_info=teacher_info,
            teacher_token_mapping=original_teacher_token_mapping,
            student_tokenizer=student_tokenizer,
            train_batch_size=train_batch_size,
            cached_latents=cached_latents,
            use_template=use_template,
            device=device,
            use_token_based_teachers=False
        )
        
        teacher_token_mapping = original_teacher_token_mapping
        num_teachers = len(teacher_models)
        print(f"📊 Final result: {num_teachers} LoRA file-based Teachers")

    # Extract UNet and text encoder lists
    teacher_unets = [model["unet"] for model in teacher_models]
    teacher_text_encoders = [model["text_encoder"] for model in teacher_models]

    # --- 7. Display final configuration ---
    print(f"\n📋 Final Training Configuration:")
    print(f"  🎯 Teacher mode: {'Token-based' if use_token_based_teachers else 'LoRA file-based'}")
    print(f"  🏫 Teacher count: {num_teachers}")
    print(f"  🏷️ Token count: {len(all_placeholder_tokens)}")
    print(f"  📊 Dataloader count: {len(dataloaders)}")
    print(f"  🔄 Teacher selection strategy: {teacher_selection_strategy}")
    print(f"  ⚡ Alternating optimization: {'enabled' if use_alternating_optimization else 'disabled'}")
    
    if use_token_based_teachers:
        print(f"  🔗 Token-Teacher correspondence:")
        for teacher_idx, teacher in enumerate(teacher_models):
            token = teacher['token']
            source_file = teacher['source_lora_info']['filename']
            print(f"    • Teacher {teacher_idx+1}: {token} (source: {source_file})")
    else:
        print(f"  🔗 LoRA-Teacher correspondence:")
        for teacher_idx, tokens in teacher_token_mapping.items():
            teacher_name = teacher_info[teacher_idx]['name'] if teacher_idx < len(teacher_info) else f"teacher_{teacher_idx}"
            print(f"    • Teacher {teacher_idx+1} ({teacher_name}): {tokens}")

    # --- 8. Training configuration setup ---
    
    # Set up noise scheduler
    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Configure model training settings
    setup_model_training_config(
        student_unet=student_unet,
        student_text_encoder=student_text_encoder,
        student_vae=student_vae,
        gradient_checkpointing=gradient_checkpointing,
        enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
        placeholder_token_ids=placeholder_token_ids,
    )

    # Calculate learning rates
    unet_lr, text_encoder_lr, ti_lr = calculate_learning_rates(
        learning_rate_unet=learning_rate_unet,
        learning_rate_text=learning_rate_text,
        learning_rate_ti=learning_rate_ti,
        scale_lr=scale_lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size
    )

    # --- 9. Textual inversion training (if enabled) ---
    if perform_inversion:
        print(f"\n{'='*80}")
        print(f"Starting {'Token-based' if use_token_based_teachers else 'LoRA file-based'} textual inversion training")
        print(f"Teacher count: {num_teachers}, Token count: {len(all_placeholder_tokens)}")
        print(f"{'='*80}")
        
        ti_optimizer = optim.AdamW(
            student_text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        lr_scheduler_ti = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        index_no_updates = torch.tensor([tid in placeholder_token_ids for tid in range(len(student_tokenizer))], 
                                 device=student_text_encoder.device)
        true_count = index_no_updates.sum().item()
        print(f"🔍 Number of True values in index_no_updates: {true_count}")

        train_inversion_with_multi_feature_alignment(
            teacher_unets=teacher_unets,
            teacher_text_encoders=teacher_text_encoders,
            student_unet=student_unet,
            vae=student_vae,
            student_text_encoder=student_text_encoder,
            dataloaders=dataloaders,
            num_steps=max_train_steps_ti,
            scheduler=noise_scheduler,
            index_no_updates=~torch.tensor([tid in placeholder_token_ids for tid in range(len(student_tokenizer))], 
                                         device=student_text_encoder.device),
            optimizer=ti_optimizer,
            save_steps=save_steps,
            placeholder_token_ids=placeholder_token_ids,
            placeholder_tokens=all_placeholder_tokens,
            save_path=output_dir,
            lr_scheduler_main=lr_scheduler_ti,
            lora_unet_target_modules=lora_unet_target_modules,
            lora_clip_target_modules=lora_clip_target_modules,
            out_name=out_name,
            tokenizer=student_tokenizer,
            test_image_path=instance_data_dir,
            cached_latents=cached_latents,
            mask_temperature=mask_temperature,
            accum_iter=gradient_accumulation_steps,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=proxy_token,
            train_inpainting=train_inpainting,
            clip_ti_decay=clip_ti_decay,
            teacher_selection_strategy=teacher_selection_strategy,
            teacher_weights=teacher_weights,
        )

        del ti_optimizer

    # --- 10. LoRA fine-tuning training ---
    print(f"\n{'='*80}")
    print(f"Starting {'Token-based' if use_token_based_teachers else 'LoRA file-based'} LoRA fine-tuning training")
    print(f"Number of Teachers: {num_teachers}, Number of tokens: {len(all_placeholder_tokens)}")
    if use_alternating_optimization:
        print(f"Optimization mode: alternating (interval: {alternating_interval}, strategy: {alternating_schedule})")
    else:
        print(f"Optimization mode: joint optimization")
    print(f"{'='*80}")

    # Set up LoRA parameters
    unet_lora_params, text_encoder_lora_params = setup_lora_parameters(
        student_unet=student_unet,
        student_text_encoder=student_text_encoder,
        train_text_encoder=train_text_encoder,
        continue_inversion=continue_inversion,
        use_extended_lora=True,
        lora_rank=lora_rank,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        lora_dropout_p=lora_dropout_p,
        lora_scale=lora_scale,
        unet_lr=unet_lr,
        text_encoder_lr=text_encoder_lr,
        ti_lr=ti_lr,
        continue_inversion_lr=continue_inversion_lr,
    )

    lora_optimizers = optim.AdamW(unet_lora_params, weight_decay=weight_decay_lora)

    student_unet.train()
    if train_text_encoder:
        student_text_encoder.train()

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )

    # Feature alignment layer configuration
    feature_alignment_unet_layers = [
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ]

    # Call multi-teacher LoRA fine-tuning function
    perform_tuning_multi_teacher(
        teacher_unets=teacher_unets,
        teacher_text_encoders=teacher_text_encoders,
        student_unet=student_unet,
        vae=student_vae,
        student_text_encoder=student_text_encoder,
        dataloaders=dataloaders,
        num_steps=max_train_steps_tuning,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=all_placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=student_tokenizer,
        out_name=out_name,
        feature_alignment_unet_layers=feature_alignment_unet_layers,
        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=proxy_token,
        train_inpainting=train_inpainting,
        feature_align_weight=feature_align_weight,
        noise_pred_weight=noise_pred_weight,
        teacher_selection_strategy=teacher_selection_strategy,
        teacher_weights=teacher_weights,
        use_alternating_optimization=use_alternating_optimization,
        alternating_interval=alternating_interval,
        alternating_schedule=alternating_schedule,
        noise_only_steps=noise_only_steps,
        feature_only_steps=feature_only_steps,
    )

    # --- 11. Training completion summary ---
    print(f"\n{'='*80}")
    print(f"Multi-Teacher distillation training completed!")
    print(f"✅ Training mode: {'Token-based Teacher' if use_token_based_teachers else 'LoRA file-based Teacher'}")
    print(f"✅ Processed {num_lora_files} LoRA files")
    print(f"✅ Created {num_teachers} Teachers")
    print(f"✅ Learned {len(all_placeholder_tokens)} tokens")
    print(f"✅ Optimization mode: {'alternating' if use_alternating_optimization else 'joint'}")
    if use_token_based_teachers:
        print(f"✅ One-to-one Token-Teacher mapping, fine-grained training")
    else:
        print(f"✅ LoRA-Teacher mapping, batch token training")
    print(f"✅ Final model saved to: {output_dir}")
    
    # Generate training summary report
    summary_report_path = os.path.join(output_dir, "training_summary.txt")
    try:
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("Multi-Teacher LoRA Distillation Training Summary Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training mode: {'Token-based Teacher' if use_token_based_teachers else 'LoRA file-based Teacher'}\n")
            f.write(f"Number of LoRA files: {num_lora_files}\n")
            f.write(f"Number of Teachers: {num_teachers}\n")
            f.write(f"Number of tokens: {len(all_placeholder_tokens)}\n")
            f.write(f"Teacher selection strategy: {teacher_selection_strategy}\n")
            f.write(f"Optimization mode: {'alternating' if use_alternating_optimization else 'joint'}\n")
            f.write(f"Textual inversion training: {'yes' if perform_inversion else 'no'}\n")
            f.write(f"TI training steps: {max_train_steps_ti}\n")
            f.write(f"LoRA training steps: {max_train_steps_tuning}\n")
            f.write(f"Feature alignment weight: {feature_align_weight}\n")
            f.write(f"Noise prediction weight: {noise_pred_weight}\n")
            f.write("\nProcessed Tokens:\n")
            for i, token in enumerate(all_placeholder_tokens):
                f.write(f"  {i+1}. {token}\n")
            f.write(f"\nTraining device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        print(f"✅ Training summary report saved: {summary_report_path}")
    except Exception as e:
        print(f"⚠️ Error saving training summary report: {e}")
    
    print(f"{'='*80}")

    # Clean up resources
    if log_wandb:
        wandb.finish()
    
    return {
        'output_dir': output_dir,
        'num_teachers': num_teachers,
        'num_tokens': len(all_placeholder_tokens),
        'training_mode': 'token_based' if use_token_based_teachers else 'lora_based',
        'optimization_mode': 'alternating' if use_alternating_optimization else 'joint'
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Teacher LoRA Distillation Training')
    
    # --- Main parameters - support folder input and backward compatibility ---
    parser.add_argument('--lora_models_dir', type=str, default=None, 
                       help='Directory containing multiple LoRA model files (new multi-teacher mode)')
    
    # Backward compatibility parameters
    parser.add_argument('--lora_path1', type=str, default=None, 
                       help='Path to first LoRA model (for backward compatibility)')
    parser.add_argument('--lora_path2', type=str, default=None, 
                       help='Path to second LoRA model (for backward compatibility)')
    
    # Core required parameters
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                       help='Path to pretrained model or HuggingFace model name')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for saving models')
    
    # --- New multi-teacher related parameters ---
    parser.add_argument('--teacher_selection_strategy', type=str, default='round_robin',
                       choices=['round_robin', 'weighted_random', 'adaptive'],
                       help='Strategy for selecting teachers during training')
    parser.add_argument('--teacher_weights', type=str, default=None,
                       help='Comma-separated weights for teachers (for weighted_random strategy)')
    parser.add_argument('--auto_generate_placeholder_tokens', action='store_true', default=True,
                       help='Automatically generate placeholder tokens from LoRA filenames')
    
    # --- Data and model configuration ---
    parser.add_argument('--instance_data_dir', type=str, default="", 
                       help='Directory containing instance data')
    parser.add_argument('--train_text_encoder', action='store_true', default=True, 
                       help='Whether to train text encoder')
    parser.add_argument('--pretrained_vae_name_or_path', type=str, default=None, 
                       help='Path to pretrained VAE')
    parser.add_argument('--revision', type=str, default=None, 
                       help='Revision of pretrained model')
    
    # --- Training mode configuration ---
    parser.add_argument('--perform_inversion', action='store_true', default=False, 
                       help='Perform textual inversion training')
    parser.add_argument('--use_template', type=str, choices=[None, 'object', 'style'], default=None, 
                       help='Template to use for data generation')
    parser.add_argument('--train_inpainting', action='store_true', default=False, 
                       help='Train for inpainting tasks')
    
    # --- Placeholder Token configuration ---
    parser.add_argument('--placeholder_tokens', type=str, default='', 
                       help='Manual placeholder tokens (fallback)')
    parser.add_argument('--placeholder_tokens1', type=str, default='', 
                       help='First set of placeholder tokens (backward compatibility)')
    parser.add_argument('--placeholder_tokens2', type=str, default='', 
                       help='Second set of placeholder tokens (backward compatibility)')
    parser.add_argument('--placeholder_token_at_data', type=str, default=None, 
                       help='Placeholder token at data')
    parser.add_argument('--initializer_tokens', type=str, default=None, 
                       help='Initializer tokens')
    
    # --- Basic training parameters ---
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--resolution', type=int, default=512, 
                       help='Resolution for training images')
    parser.add_argument('--color_jitter', action='store_true', default=True, 
                       help='Apply color jitter augmentation')
    parser.add_argument('--train_batch_size', type=int, default=1, 
                       help='Training batch size')
    parser.add_argument('--sample_batch_size', type=int, default=1, 
                       help='Sampling batch size')
    
    # --- Training steps configuration ---
    parser.add_argument('--max_train_steps_tuning', type=int, default=1000, 
                       help='Maximum number of LoRA tuning steps')
    parser.add_argument('--max_train_steps_ti', type=int, default=1000, 
                       help='Maximum number of textual inversion steps')
    parser.add_argument('--save_steps', type=int, default=100, 
                       help='Steps between saving checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                       help='Number of gradient accumulation steps')
    
    # --- Model optimization configuration ---
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, 
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', default=False, 
                       help='Enable xformers memory efficient attention')
    
    # --- LoRA configuration ---
    parser.add_argument('--lora_rank', type=int, default=4, 
                       help='LoRA rank (dimensionality)')
    parser.add_argument('--lora_unet_target_modules', type=str, 
                       default="CrossAttention,Attention,GEGLU", 
                       help='Target modules for UNet LoRA (comma-separated)')
    parser.add_argument('--lora_clip_target_modules', type=str, 
                       default="CLIPSdpaAttention", 
                       help='Target modules for CLIP LoRA (comma-separated)')
    parser.add_argument('--lora_dropout_p', type=float, default=0.0, 
                       help='LoRA dropout probability')
    parser.add_argument('--lora_scale', type=float, default=1.0, 
                       help='LoRA scale factor')
    parser.add_argument('--use_extended_lora', action='store_true', default=False, 
                       help='Use extended LoRA modules')
    
    # --- Text embedding configuration ---
    parser.add_argument('--clip_ti_decay', action='store_true', default=True, 
                       help='Enable CLIP textual inversion decay/regularization')
    
    # --- Learning rate configuration ---
    parser.add_argument('--learning_rate_unet', type=float, default=1e-4, 
                       help='Learning rate for UNet training')
    parser.add_argument('--learning_rate_text', type=float, default=1e-5, 
                       help='Learning rate for text encoder training')
    parser.add_argument('--learning_rate_ti', type=float, default=5e-4, 
                       help='Learning rate for textual inversion')
    parser.add_argument('--scale_lr', action='store_true', default=False, 
                       help='Scale learning rate by batch size and accumulation steps')
    
    # --- Learning rate scheduler configuration ---
    parser.add_argument('--lr_scheduler', type=str, default='linear', 
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
                       help='Learning rate scheduler for textual inversion')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, 
                       help='Learning rate warmup steps for textual inversion')
    parser.add_argument('--lr_scheduler_lora', type=str, default='linear', 
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
                       help='Learning rate scheduler for LoRA training')
    parser.add_argument('--lr_warmup_steps_lora', type=int, default=0, 
                       help='Learning rate warmup steps for LoRA training')
    
    # --- Continue training configuration ---
    parser.add_argument('--continue_inversion', action='store_true', default=False, 
                       help='Continue textual inversion during LoRA training')
    parser.add_argument('--continue_inversion_lr', type=float, default=None, 
                       help='Learning rate for continued textual inversion (if different from --learning_rate_ti)')
    
    # --- Data processing configuration ---
    parser.add_argument('--use_face_segmentation_condition', action='store_true', default=False, 
                       help='Use face segmentation as conditioning')
    parser.add_argument('--cached_latents', action='store_true', default=True, 
                       help='Use cached latents for faster training')
    parser.add_argument('--use_mask_captioned_data', action='store_true', default=False, 
                       help='Use mask captioned data')
    parser.add_argument('--mask_temperature', type=float, default=1.0, 
                       help='Temperature for mask processing')
    
    # --- Optimizer configuration ---
    parser.add_argument('--weight_decay_ti', type=float, default=0.00, 
                       help='Weight decay for textual inversion optimizer')
    parser.add_argument('--weight_decay_lora', type=float, default=0.001, 
                       help='Weight decay for LoRA optimizer')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False, 
                       help='Use 8-bit Adam optimizer to save memory')
    
    # --- Device and precision configuration ---
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device to use for training (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--mixed_precision', type=str, default='no', 
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training mode')
    
    # --- Loss weight configuration ---
    parser.add_argument('--feature_align_weight', type=float, default=0.01, 
                       help='Weight for feature alignment loss')
    parser.add_argument('--noise_pred_weight', type=float, default=1.0, 
                       help='Weight for noise prediction loss')
    
    # --- Logging and monitoring configuration ---
    parser.add_argument('--log_wandb', action='store_true', default=False, 
                       help='Log training metrics to Weights & Biases')
    parser.add_argument('--wandb_log_prompt_cnt', type=int, default=10, 
                       help='Number of prompts to log to W&B for evaluation')
    parser.add_argument('--wandb_project_name', type=str, default='multi_teacher_lora_distillation', 
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='your_wandb_entity', 
                       help='W&B entity/username')
    parser.add_argument('--proxy_token', type=str, default='person', 
                       help='Proxy token for class-based training')
    parser.add_argument('--out_name', type=str, default='multi_teacher_lora_distilled', 
                       help='Name prefix for output model files')
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs_new', 
                       help='Directory for TensorBoard logs')
    
    # --- Feature alignment configuration ---
    parser.add_argument('--unet_feature_align_weight', type=float, default=0.01, 
                       help='Weight for UNet feature alignment loss')
    parser.add_argument('--text_encoder_feature_align_weight', type=float, default=0.02, 
                       help='Weight for text encoder feature alignment loss')
    parser.add_argument('--unet_feature_alignment_layers', type=str,
                       default="down_blocks.0,down_blocks.1,down_blocks.2,down_blocks.3,mid_block,up_blocks.0,up_blocks.1,up_blocks.2,up_blocks.3",
                       help='UNet layers for feature alignment (comma-separated)')
    parser.add_argument('--text_encoder_alignment_layers', type=str,
                       default="text_model.encoder.layers.0,text_model.encoder.layers.6,text_model.encoder.layers.11",
                       help='Text encoder layers for feature alignment (comma-separated)')
    parser.add_argument('--text_encoder_pooling_strategy', type=str, default='mean',
                       choices=['mean', 'cls', 'max', 'none'],
                       help='Pooling strategy for text encoder features')
    parser.add_argument('--text_encoder_loss_type', type=str, default='mse',
                       choices=['mse', 'l1', 'cosine'],
                       help='Loss type for text encoder feature alignment')
    
    # --- Other advanced configuration ---
    parser.add_argument('--t_multiplier_loss', type=float, default=1.0, 
                       help='Multiplier for timestep sampling in loss calculation')
    parser.add_argument('--save_image_every_n_steps_loss', type=int, default=200, 
                       help='Save sample images every N steps during loss calculation')
    parser.add_argument('--unet_return_dict', action='store_true', default=True, 
                       help='Whether UNet feature extractor returns dictionary')
    
    args = parser.parse_args()
    
    # --- Parameter validation and processing ---
    
    # Validate required input parameters
    if not args.lora_models_dir and not (args.lora_path1 and args.lora_path2):
        parser.error("Must provide --lora_models_dir or both --lora_path1 and --lora_path2")
    
    if args.lora_models_dir and (args.lora_path1 or args.lora_path2):
        print("Warning: Both --lora_models_dir and individual lora_path arguments provided. --lora_models_dir will take priority")
        args.lora_path1 = None
        args.lora_path2 = None
    
    # Process teacher_weights parameter
    teacher_weights = None
    if args.teacher_weights:
        try:
            teacher_weights = [float(w.strip()) for w in args.teacher_weights.split(',')]
            print(f"Using teacher weights: {teacher_weights}")
        except ValueError:
            parser.error("teacher_weights must be comma-separated floats, e.g.: '0.3,0.4,0.3'")
    
    # Process LoRA target module parameters
    if isinstance(args.lora_unet_target_modules, str):
        lora_unet_target_modules = set(args.lora_unet_target_modules.split(','))
    else:
        lora_unet_target_modules = args.lora_unet_target_modules
    
    if isinstance(args.lora_clip_target_modules, str):
        lora_clip_target_modules = set(args.lora_clip_target_modules.split(','))
    else:
        lora_clip_target_modules = args.lora_clip_target_modules
    
    # Process feature alignment layer parameters
    if isinstance(args.unet_feature_alignment_layers, str):
        unet_feature_alignment_layers = args.unet_feature_alignment_layers.split(',')
    else:
        unet_feature_alignment_layers = args.unet_feature_alignment_layers
    
    if isinstance(args.text_encoder_alignment_layers, str):
        text_encoder_alignment_layers = args.text_encoder_alignment_layers.split(',')
    else:
        text_encoder_alignment_layers = args.text_encoder_alignment_layers
    
    # Validate teacher selection strategy
    if args.teacher_selection_strategy == 'weighted_random' and teacher_weights is None:
        print("Warning: Using weighted_random strategy without teacher_weights, uniform weights will be used")
    
    # Create complete parameter dictionary
    train_args = {
        # Main parameters
        'lora_models_dir': args.lora_models_dir,
        'lora_path1': args.lora_path1,
        'lora_path2': args.lora_path2,
        'instance_data_dir': args.instance_data_dir,
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'output_dir': args.output_dir,
        
        # Multi-teacher configuration
        'teacher_selection_strategy': args.teacher_selection_strategy,
        'teacher_weights': teacher_weights,
        'auto_generate_placeholder_tokens': args.auto_generate_placeholder_tokens,
        
        # Model and training configuration
        'train_text_encoder': args.train_text_encoder,
        'pretrained_vae_name_or_path': args.pretrained_vae_name_or_path,
        'revision': args.revision,
        'perform_inversion': args.perform_inversion,
        'use_template': args.use_template,
        'train_inpainting': args.train_inpainting,
        
        # Token configuration
        'placeholder_tokens': args.placeholder_tokens,
        'placeholder_tokens1': args.placeholder_tokens1,
        'placeholder_tokens2': args.placeholder_tokens2,
        'placeholder_token_at_data': args.placeholder_token_at_data,
        'initializer_tokens': args.initializer_tokens,
        
        # Basic training parameters
        'seed': args.seed,
        'resolution': args.resolution,
        'color_jitter': args.color_jitter,
        'train_batch_size': args.train_batch_size,
        'sample_batch_size': args.sample_batch_size,
        'max_train_steps_tuning': args.max_train_steps_tuning,
        'max_train_steps_ti': args.max_train_steps_ti,
        'save_steps': args.save_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'gradient_checkpointing': args.gradient_checkpointing,
        
        # LoRA configuration
        'lora_rank': args.lora_rank,
        'lora_unet_target_modules': lora_unet_target_modules,
        'lora_clip_target_modules': lora_clip_target_modules,
        'lora_dropout_p': args.lora_dropout_p,
        'lora_scale': args.lora_scale,
        'use_extended_lora': args.use_extended_lora,
        'clip_ti_decay': args.clip_ti_decay,
        
        # Learning rate configuration
        'learning_rate_unet': args.learning_rate_unet,
        'learning_rate_text': args.learning_rate_text,
        'learning_rate_ti': args.learning_rate_ti,
        'continue_inversion': args.continue_inversion,
        'continue_inversion_lr': args.continue_inversion_lr,
        'scale_lr': args.scale_lr,
        'lr_scheduler': args.lr_scheduler,
        'lr_warmup_steps': args.lr_warmup_steps,
        'lr_scheduler_lora': args.lr_scheduler_lora,
        'lr_warmup_steps_lora': args.lr_warmup_steps_lora,
        
        # Data processing configuration
        'use_face_segmentation_condition': args.use_face_segmentation_condition,
        'cached_latents': args.cached_latents,
        'use_mask_captioned_data': args.use_mask_captioned_data,
        'mask_temperature': args.mask_temperature,
        
        # Optimizer configuration
        'weight_decay_ti': args.weight_decay_ti,
        'weight_decay_lora': args.weight_decay_lora,
        'use_8bit_adam': args.use_8bit_adam,
        
        # Device and system configuration
        'device': args.device,
        'extra_args': None,  # Additional parameters can be added here
        'enable_xformers_memory_efficient_attention': args.enable_xformers_memory_efficient_attention,
        
        # Loss weights
        'feature_align_weight': args.feature_align_weight,
        'noise_pred_weight': args.noise_pred_weight,
        
        # Logging configuration
        'log_wandb': args.log_wandb,
        'wandb_log_prompt_cnt': args.wandb_log_prompt_cnt,
        'wandb_project_name': args.wandb_project_name,
        'wandb_entity': args.wandb_entity,
        'proxy_token': args.proxy_token,
        'out_name': args.out_name,
    }
    
    # Display configuration summary
    print("\n" + "="*80)
    print("Multi-Teacher LoRA Distillation Training Configuration Summary")
    print("="*80)
    
    if args.lora_models_dir:
        print(f"Mode: Multi-Teacher folder mode")
        print(f"LoRA model directory: {args.lora_models_dir}")
    else:
        print(f"Mode: Dual-Teacher compatibility mode")
        print(f"LoRA path 1: {args.lora_path1}")
        print(f"LoRA path 2: {args.lora_path2}")
    
    print(f"Base model: {args.pretrained_model_name_or_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Teacher selection strategy: {args.teacher_selection_strategy}")
    if teacher_weights:
        print(f"Teacher weights: {teacher_weights}")
    print(f"Auto-generate tokens: {args.auto_generate_placeholder_tokens}")
    print(f"Perform textual inversion: {args.perform_inversion}")
    print(f"Training batch size: {args.train_batch_size}")
    print(f"TI training steps: {args.max_train_steps_ti}")
    print(f"LoRA training steps: {args.max_train_steps_tuning}")
    print(f"Device: {args.device}")
    print(f"Feature alignment weight: {args.feature_align_weight}")
    print("="*80)
    
    # Run training
    try:
        print("Starting multi-Teacher distillation training...")
        train(**train_args)
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
