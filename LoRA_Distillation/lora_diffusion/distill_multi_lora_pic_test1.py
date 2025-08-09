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
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torch.utils.tensorboard import SummaryWriter
from feature_hook_unet import FeatureAlignmentLoss, UNetFeatureExtractor
from feature_hook_text_encoder import TextEncoderFeatureExtractor, TextEncoderFeatureAlignmentLoss
from dynamic_weight import create_inversion_weight_adjuster, create_tuning_weight_adjuster
import warnings

sys.path.append('/root/lora')
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

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens1: List[str],
    placeholder_tokens2: List[str],
    initializer_tokens1: List[str],
    initializer_tokens2: List[str],
    device="cuda:0",
):

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

    placeholder_token_ids1 = []
    placeholder_token_ids2 = []

    for token, init_tok in zip(placeholder_tokens1, initializer_tokens1):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids1.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
            
    for token, init_tok in zip(placeholder_tokens2, initializer_tokens2):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids2.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
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
        placeholder_token_ids1,
        placeholder_token_ids2,
    )

def text2img_dataloader_combined_with_latent_caching(
    train_dataset,
    train_batch_size,
    main_tokenizer,
    vae,                # VAE model for encoding images to latents if cached_latents is True
    aux_tokenizer1=None,
    aux_tokenizer2=None,
    cached_latents: bool = False,
    num_workers: int = 0,
    # vae_scaling_factor: float = 0.18215 # 可以作为参数传入，或从vae.config获取
):
    """
    Creates a DataLoader, with latent caching directly imitating the non-leaking snippet.
    - Modifies the original dataset item directly when caching latents.
    """

    dataset_to_load = train_dataset

    if cached_latents:
        if vae is None:
            raise ValueError("VAE must be provided if cached_latents is True.")
        
        print(f"Caching latents for {len(train_dataset)} images (direct imitation)...")
        
        # 使用 VAE 所在的设备进行操作
        current_vae_device = vae.device 
        # 获取缩放因子
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215) # 与您的"不泄漏"代码一致

        cached_dataset_items = [] # 存储修改后的数据集项

        for i in tqdm(range(len(train_dataset)), desc="Caching latents (direct imitation)"):
            # 1. Get the original item from the dataset
            # 'item_from_dataset' is a dictionary.
            item_from_dataset = train_dataset[i] 
            
            if "instance_images" not in item_from_dataset:
                raise ValueError(f"Dataset item at index {i} did not return 'instance_images'.")

            pixel_values = item_from_dataset["instance_images"] 

            # 确保 pixel_values 是张量 (这里不做PIL转换，假设dataset返回的是张量)
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError(
                    f"Expected 'instance_images' to be a torch.Tensor for direct imitation, got {type(pixel_values)}. "
                    "If dataset returns PIL, pre-processing is needed or use the 'fixed' version."
                )
            
            # 准备 VAE 输入: 添加批次维度，移动到 VAE 设备，转换数据类型
            input_pixels_for_vae = pixel_values.unsqueeze(0).to(device=current_vae_device, dtype=vae.dtype)

            # 2. Encode to latents using VAE (在 no_grad 上下文中)
            with torch.no_grad():
                latents_on_vae_device = vae.encode(input_pixels_for_vae).latent_dist.sample()
            
            latents_on_vae_device = latents_on_vae_device * scaling_factor
            
            # 3. Move latents to CPU and remove batch dimension
            latents_on_cpu = latents_on_vae_device.squeeze(0).cpu()
            
            # 4. DIRECTLY MODIFY the item from dataset
            item_from_dataset["instance_images"] = latents_on_cpu
            
            # 5. Add the MODIFIED item to the list
            cached_dataset_items.append(item_from_dataset)

            # 注意：这里不添加显式的 del 或 torch.cuda.empty_cache()，严格模仿
        
        dataset_to_load = cached_dataset_items
        print("Latent caching (direct imitation) complete.")
    else:
        print("Using pixel values directly from the dataset (no latent caching).")

    # --- Collate function (与之前版本相同) ---
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

def loss_step_gaussian_noise(
    batch,
    student_unet,
    student_text_encoder,
    scheduler,
    vae,
    global_step,
    save_image_every_n_steps=100,
    output_dir_for_loss_step="/root/lora_train/pic_train",
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    # 新增参数：用于标识当前teacher和保存目录
    current_teacher_name: str = "teacher",
    current_teacher_idx: int = 0,
    base_save_dir: str = None,
    save_comparison_grid: bool = False,  # 是否保存对比网格图
):
    """
    计算高斯噪声预测损失并保存学生模型的预测结果
    
    新增功能：
    - 按teacher分类保存预测结果
    - 为每个teacher创建独立的保存目录
    - 在文件名中包含teacher信息
    - 可选的对比网格图保存
    
    Args:
        current_teacher_name: 当前teacher的名称
        current_teacher_idx: 当前teacher的索引
        base_save_dir: 基础保存目录（如果为None，使用output_dir_for_loss_step）
        save_comparison_grid: 是否保存包含所有对比的网格图
    """
    weight_dtype = torch.float32
    if mixed_precision:
        vae_decode_input_dtype = torch.float32
    else:
        vae_decode_input_dtype = torch.float32

    # 处理潜变量
    if batch["pixel_values"].ndim == 4 and batch["pixel_values"].shape[1] in [1, 3, 4]:
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)
    else:
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)

    bsz = latents_gt_x0.shape[0]

    # 为每张图片采样一个随机的时间步
    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents_gt_x0.device,
    )
    timesteps = timesteps.long()

    # 采样噪声
    noise = torch.randn_like(latents_gt_x0)
    noisy_latents = scheduler.add_noise(latents_gt_x0, noise, timesteps)

    # 准备UNet输入
    if mixed_precision:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float16)
    else:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float32)

    # --- 学生模型文本编码部分 ---
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

    # 文本编码（支持混合精度）
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

    # --- 学生模型UNet预测噪声 ---
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

    # 处理mask（如果存在）
    if batch.get("mask", None) is not None:
        mask = batch["mask"].to(student_pred_noise.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Mask 形状异常: {mask.shape}. 期望 [B, 1, H, W] 或 [B, H, W]")

        mask = (mask + 0.01).pow(mask_temperature)
        mask = mask / mask.max()
        pred_dtype = student_pred_noise.dtype
        mask = mask.to(dtype=pred_dtype)
        student_pred_noise = student_pred_noise * mask
        target_noise = target_noise * mask

    # 计算损失
    loss = F.mse_loss(student_pred_noise.float(), target_noise.float(), reduction="none").mean([1, 2, 3]).mean()

    # --- 优化的保存逻辑 ---
    if global_step % save_image_every_n_steps == 0 and save_image_every_n_steps > 0:
        with torch.no_grad():
            latents_gt_x0_viz = latents_gt_x0[0:1].detach()
            student_pred_noise_viz = student_pred_noise[0:1].detach()
            noisy_latents_viz = noisy_latents[0:1].detach()
            timestep_viz = timesteps[0:1].detach()

            current_timestep_int = timestep_viz.item() if timestep_viz.numel() == 1 else timestep_viz[0].item()

            # 从学生预测的噪声重构x0
            scheduler_output_student = scheduler.step(
                model_output=student_pred_noise_viz.to(dtype=noisy_latents_viz.dtype),
                timestep=torch.tensor([current_timestep_int], device=noisy_latents_viz.device) if not isinstance(current_timestep_int, int) else current_timestep_int,
                sample=noisy_latents_viz
            )
            pred_x0_latent_student = scheduler_output_student.pred_original_sample

            # VAE解码准备
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            pred_x0_latent_student_for_vae = pred_x0_latent_student.to(dtype=vae_decode_input_dtype) / scaling_factor
            latent_x0_gt_viz_for_vae = latents_gt_x0_viz.to(dtype=vae_decode_input_dtype) / scaling_factor
            noisy_input_latent_for_vae = noisy_latents_viz.to(dtype=vae_decode_input_dtype) / scaling_factor

            vae_internal_dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float32
            
            # VAE解码
            with torch.cuda.amp.autocast(enabled=False):
                pred_image_student = vae.decode(pred_x0_latent_student_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
                gt_image = vae.decode(latent_x0_gt_viz_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
                noisy_input_image = vae.decode(noisy_input_latent_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample

            # 后处理图像
            pred_image_student = (pred_image_student / 2 + 0.5).clamp(0, 1)
            gt_image = (gt_image / 2 + 0.5).clamp(0, 1)
            noisy_input_image = (noisy_input_image / 2 + 0.5).clamp(0, 1)

            # --- 创建按teacher分类的保存目录结构 ---
            # 清理teacher名称，移除可能引起文件系统问题的字符
            clean_teacher_name = re.sub(r'[^\w\-_.]', '_', current_teacher_name)
            
            if base_save_dir:
                # 使用传入的基础目录
                teacher_save_dir = os.path.join(base_save_dir, f"teacher_{current_teacher_idx+1:02d}_{clean_teacher_name}")
            else:
                # 使用默认目录
                teacher_save_dir = os.path.join(output_dir_for_loss_step, f"teacher_{current_teacher_idx+1:02d}_{clean_teacher_name}")
            
            # 创建子目录用于不同类型的图像
            subdirs = {
                'student_pred': os.path.join(teacher_save_dir, 'student_predictions'),
                'ground_truth': os.path.join(teacher_save_dir, 'ground_truth'),
                'noisy_input': os.path.join(teacher_save_dir, 'noisy_inputs'),
                'comparisons': os.path.join(teacher_save_dir, 'comparisons')
            }
            
            for subdir in subdirs.values():
                if not os.path.exists(subdir):
                    os.makedirs(subdir, exist_ok=True)

            # --- 生成带时间戳和详细信息的文件名 ---
            t_val = current_timestep_int
            timestamp_suffix = f"step_{global_step:06d}_t{t_val:03d}"
            
            # 获取当前批次的文本信息（如果可用）
            text_info = ""
            if "raw_text" in batch and batch["raw_text"]:
                # 清理文本用于文件名
                raw_text = batch["raw_text"][0] if isinstance(batch["raw_text"], list) else str(batch["raw_text"])
                clean_text = re.sub(r'[^\w\s\-_.]', '', raw_text)[:20]  # 限制长度
                text_info = f"_{clean_text.replace(' ', '_')}" if clean_text else ""
            
            # --- 保存各类图像 ---
            # 1. 学生预测结果
            student_pred_filename = f"{timestamp_suffix}_from_{clean_teacher_name}{text_info}.png"
            student_pred_path = os.path.join(subdirs['student_pred'], student_pred_filename)
            save_image(pred_image_student, student_pred_path)
            
            # 2. 真实图像
            gt_filename = f"{timestamp_suffix}_gt{text_info}.png"
            gt_path = os.path.join(subdirs['ground_truth'], gt_filename)
            save_image(gt_image, gt_path)
            
            # 3. 噪声输入
            noisy_filename = f"{timestamp_suffix}_noisy{text_info}.png"
            noisy_path = os.path.join(subdirs['noisy_input'], noisy_filename)
            save_image(noisy_input_image, noisy_path)
            
            # --- 4. 可选：创建对比网格图 ---
            if save_comparison_grid:
                try:
                    import torchvision.utils as vutils
                    
                    # 创建对比网格：[噪声输入, 学生预测, 真实图像]
                    comparison_images = torch.cat([noisy_input_image, pred_image_student, gt_image], dim=0)
                    
                    # 创建网格图
                    grid = vutils.make_grid(
                        comparison_images, 
                        nrow=3, 
                        padding=2, 
                        normalize=False,
                        pad_value=1.0  # 白色边框
                    )
                    
                    comparison_filename = f"{timestamp_suffix}_comparison_from_{clean_teacher_name}{text_info}.png"
                    comparison_path = os.path.join(subdirs['comparisons'], comparison_filename)
                    save_image(grid.unsqueeze(0), comparison_path)
                    
                except ImportError:
                    print("警告: torchvision.utils 不可用，跳过对比网格图保存")
                except Exception as e:
                    print(f"警告: 保存对比网格图时出错: {e}")
            
            # --- 记录详细的保存信息 ---
            print(f"[Step {global_step:06d}] 已保存 Teacher {current_teacher_idx+1} ({current_teacher_name}) 的预测结果:")
            print(f"  📁 保存目录: {teacher_save_dir}")
            print(f"  🎯 学生预测: {os.path.basename(student_pred_path)}")
            print(f"  ✅ 真实图像: {os.path.basename(gt_path)}")
            print(f"  🔀 噪声输入: {os.path.basename(noisy_path)}")
            if save_comparison_grid:
                print(f"  📊 对比网格: {os.path.basename(comparison_path)}")
            print(f"  ⏰ 时间步: {t_val}, 损失: {loss.item():.6f}")
            
            # --- 可选：创建索引文件，便于后续分析 ---
            index_file_path = os.path.join(teacher_save_dir, "save_index.txt")
            with open(index_file_path, "a", encoding='utf-8') as f:
                f.write(f"{global_step:06d},{t_val:03d},{loss.item():.6f},{clean_teacher_name},{timestamp_suffix}\n")
    
    return loss

def select_current_teacher(
    global_step: int,
    num_teachers: int,
    strategy: str = "round_robin",
    weights: Optional[List[float]] = None,
    selection_index: int = 0
) -> int:
    """
    选择当前要使用的teacher模型
    
    Args:
        global_step: 当前全局步数
        num_teachers: teacher数量
        strategy: 选择策略
        weights: teacher权重（用于weighted_random）
        selection_index: 当前选择索引（用于round_robin）
    
    Returns:
        选择的teacher索引
    """
    if strategy == "round_robin":
        return global_step % num_teachers
    
    elif strategy == "weighted_random":
        if weights is None:
            weights = [1.0] * num_teachers
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # 归一化
        return torch.multinomial(weights, 1).item()
    
    elif strategy == "adaptive":
        # 可以根据损失历史等信息自适应选择
        # 这里先实现一个简单版本，后续可以扩展
        return global_step % num_teachers
    
    else:
        raise ValueError(f"Unknown teacher selection strategy: {strategy}")

def compute_multi_teacher_kl_distillation_loss(
    teacher_text_encoders: List,
    student_text_encoder,
    input_ids,
    attention_mask,
    temperature: float = 4.0,
    teacher_weights: Optional[List[float]] = None,
    aggregation_strategy: str = "weighted_average",  # "weighted_average", "ensemble", "competitive"
    use_layer_wise_distillation: bool = True
):
    """
    多Teacher知识蒸馏损失计算
    
    Args:
        teacher_text_encoders: 教师模型列表
        student_text_encoder: 学生模型
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        temperature: 蒸馏温度
        teacher_weights: 教师权重
        aggregation_strategy: 聚合策略
        use_layer_wise_distillation: 是否使用逐层蒸馏
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # 如果没有提供权重，使用均匀权重
    if teacher_weights is None:
        teacher_weights = [1.0 / len(teacher_text_encoders)] * len(teacher_text_encoders)
    
    # 归一化权重
    teacher_weights = torch.tensor(teacher_weights, device=device, dtype=torch.float32)
    teacher_weights = teacher_weights / teacher_weights.sum()
    
    # 获取学生模型输出
    with torch.cuda.amp.autocast(enabled=True):
        student_outputs = student_text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        student_last_hidden = student_outputs.last_hidden_state
        student_hidden_states = student_outputs.hidden_states if use_layer_wise_distillation else None
    
    # 收集所有教师模型的输出
    teacher_last_hiddens = []
    teacher_hidden_states_list = [] if use_layer_wise_distillation else None
    
    with torch.no_grad():
        for teacher_text_encoder in teacher_text_encoders:
            teacher_outputs = teacher_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=use_layer_wise_distillation,
                return_dict=True
            )
            teacher_last_hiddens.append(teacher_outputs.last_hidden_state)
            
            if use_layer_wise_distillation:
                teacher_hidden_states_list.append(teacher_outputs.hidden_states)
    
    # 1. 最终层的知识蒸馏
    final_layer_loss = compute_aggregated_kl_loss(
        teacher_outputs=teacher_last_hiddens,
        student_output=student_last_hidden,
        teacher_weights=teacher_weights,
        temperature=temperature,
        aggregation_strategy=aggregation_strategy,
        attention_mask=attention_mask
    )
    
    # 2. 逐层知识蒸馏（如果启用）
    layer_wise_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if use_layer_wise_distillation and teacher_hidden_states_list is not None:
        layer_wise_loss = compute_layer_wise_distillation_loss(
            teacher_hidden_states_list=teacher_hidden_states_list,
            student_hidden_states=student_hidden_states,
            teacher_weights=teacher_weights,
            temperature=temperature,
            aggregation_strategy=aggregation_strategy,
            attention_mask=attention_mask
        )
    
    # 3. 注意力蒸馏（可选）
    attention_distillation_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    total_kl_loss = final_layer_loss + 0.5 * layer_wise_loss + 0.3 * attention_distillation_loss
    
    return total_kl_loss.float(), {
        'final_layer_kl': final_layer_loss.item(),
        'layer_wise_kl': layer_wise_loss.item(),
        'attention_kl': attention_distillation_loss.item(),
        'total_kl': total_kl_loss.item()
    }

def compute_aggregated_kl_loss(
    teacher_outputs: List[torch.Tensor],
    student_output: torch.Tensor,
    teacher_weights: torch.Tensor,
    temperature: float,
    aggregation_strategy: str,
    attention_mask: Optional[torch.Tensor] = None
):
    """
    计算聚合的KL散度损失
    """
    if aggregation_strategy == "weighted_average":
        # 策略1: 加权平均教师输出
        weighted_teacher_output = torch.zeros_like(teacher_outputs[0])
        for i, teacher_output in enumerate(teacher_outputs):
            weighted_teacher_output += teacher_weights[i] * teacher_output
        
        return compute_single_kl_loss(
            teacher_output=weighted_teacher_output,
            student_output=student_output,
            temperature=temperature,
            attention_mask=attention_mask
        )
    
    elif aggregation_strategy == "ensemble":
        # 策略2: 集成蒸馏 - 分别计算每个teacher的KL损失然后加权
        total_loss = torch.tensor(0.0, device=student_output.device, dtype=torch.float32)
        for i, teacher_output in enumerate(teacher_outputs):
            teacher_kl_loss = compute_single_kl_loss(
                teacher_output=teacher_output,
                student_output=student_output,
                temperature=temperature,
                attention_mask=attention_mask
            )
            total_loss += teacher_weights[i] * teacher_kl_loss
        
        return total_loss
    
    elif aggregation_strategy == "competitive":
        # 策略3: 竞争性蒸馏 - 选择最好的teacher进行蒸馏
        min_loss = float('inf')
        best_loss = None
        
        for teacher_output in teacher_outputs:
            teacher_kl_loss = compute_single_kl_loss(
                teacher_output=teacher_output,
                student_output=student_output,
                temperature=temperature,
                attention_mask=attention_mask
            )
            if teacher_kl_loss.item() < min_loss:
                min_loss = teacher_kl_loss.item()
                best_loss = teacher_kl_loss
        
        return best_loss
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")

def compute_single_kl_loss(
    teacher_output: torch.Tensor,
    student_output: torch.Tensor,
    temperature: float,
    attention_mask: Optional[torch.Tensor] = None
):
    """
    计算单个teacher和student之间的KL散度损失
    """
    # 应用温度缩放
    teacher_logits = teacher_output / temperature
    student_logits = student_output / temperature
    
    # 计算概率分布
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # 计算KL散度
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none')
    
    # 应用注意力掩码（如果提供）
    if attention_mask is not None:
        # 扩展掩码维度以匹配hidden_dim
        mask = attention_mask.unsqueeze(-1).expand_as(kl_loss)
        kl_loss = kl_loss * mask
        
        # 计算平均损失，考虑有效token数量
        valid_tokens = attention_mask.sum(dim=1, keepdim=True).float()
        kl_loss = kl_loss.sum(dim=[1, 2]) / (valid_tokens * kl_loss.shape[-1])
        kl_loss = kl_loss.mean()
    else:
        kl_loss = kl_loss.mean()
    
    # 应用温度平方缩放（标准KL蒸馏做法）
    kl_loss = kl_loss * (temperature ** 2)
    
    return kl_loss

def compute_layer_wise_distillation_loss(
    teacher_hidden_states_list: List[List[torch.Tensor]],
    student_hidden_states: List[torch.Tensor],
    teacher_weights: torch.Tensor,
    temperature: float,
    aggregation_strategy: str,
    attention_mask: Optional[torch.Tensor] = None,
    layer_weights: Optional[List[float]] = None
):
    """
    计算逐层蒸馏损失
    """
    num_layers = len(student_hidden_states)
    total_loss = torch.tensor(0.0, device=student_hidden_states[0].device, dtype=torch.float32)
    
    # 默认层权重（更重视后面的层）
    if layer_weights is None:
        layer_weights = [i / num_layers for i in range(1, num_layers + 1)]
    
    layer_weights = torch.tensor(layer_weights, device=student_hidden_states[0].device, dtype=torch.float32)
    layer_weights = layer_weights / layer_weights.sum()
    
    for layer_idx in range(num_layers):
        # 收集该层的所有teacher输出
        teacher_layer_outputs = []
        for teacher_hidden_states in teacher_hidden_states_list:
            if layer_idx < len(teacher_hidden_states):
                teacher_layer_outputs.append(teacher_hidden_states[layer_idx])
        
        if teacher_layer_outputs:
            student_layer_output = student_hidden_states[layer_idx]
            
            layer_loss = compute_aggregated_kl_loss(
                teacher_outputs=teacher_layer_outputs,
                student_output=student_layer_output,
                teacher_weights=teacher_weights,
                temperature=temperature,
                aggregation_strategy=aggregation_strategy,
                attention_mask=attention_mask
            )
            
            total_loss += layer_weights[layer_idx] * layer_loss
    
    return total_loss

def compute_multi_teacher_attention_distillation_loss(
    teacher_text_encoders: List,
    student_text_encoder,
    input_ids,
    attention_mask,
    teacher_weights: Optional[List[float]] = None,
    attention_temperature: float = 4.0
):
    """
    多Teacher注意力蒸馏损失
    """
    device = input_ids.device
    
    if teacher_weights is None:
        teacher_weights = [1.0 / len(teacher_text_encoders)] * len(teacher_text_encoders)
    teacher_weights = torch.tensor(teacher_weights, device=device, dtype=torch.float32)
    teacher_weights = teacher_weights / teacher_weights.sum()
    
    # 获取学生模型的注意力权重
    student_outputs = student_text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True
    )
    student_attentions = student_outputs.attentions
    
    # 收集教师模型的注意力权重
    teacher_attentions_list = []
    with torch.no_grad():
        for teacher_text_encoder in teacher_text_encoders:
            teacher_outputs = teacher_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            teacher_attentions_list.append(teacher_outputs.attentions)
    
    attention_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    num_layers = len(student_attentions)
    
    for layer_idx in range(num_layers):
        # 聚合该层的teacher注意力权重
        teacher_layer_attentions = []
        for teacher_attentions in teacher_attentions_list:
            if layer_idx < len(teacher_attentions):
                teacher_layer_attentions.append(teacher_attentions[layer_idx])
        
        if teacher_layer_attentions:
            # 加权平均teacher注意力
            weighted_teacher_attention = torch.zeros_like(teacher_layer_attentions[0])
            for i, teacher_attention in enumerate(teacher_layer_attentions):
                weighted_teacher_attention += teacher_weights[i] * teacher_attention
            
            # 计算注意力蒸馏损失
            student_attention = student_attentions[layer_idx]
            
            # 应用温度缩放
            teacher_attention_scaled = weighted_teacher_attention / attention_temperature
            student_attention_scaled = student_attention / attention_temperature
            
            # 计算KL散度
            teacher_attention_probs = F.softmax(teacher_attention_scaled, dim=-1)
            student_attention_log_probs = F.log_softmax(student_attention_scaled, dim=-1)
            
            layer_attention_loss = F.kl_div(
                student_attention_log_probs, 
                teacher_attention_probs, 
                reduction='batchmean'
            ) * (attention_temperature ** 2)
            
            attention_loss += layer_attention_loss / num_layers
    
    return attention_loss

def train_inversion_with_multi_feature_alignment(
    # --- "Teacher" 模型参数 (修改为列表形式) ---
    teacher_unets: List,  # 从 teacher1_unet, teacher2_unet 改为列表
    teacher_text_encoders: List,  # 从 teacher1_text_encoder, teacher2_text_encoder 改为列表
    # --- "Student" 模型参数 ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader 参数 (修改为列表形式) ---
    dataloaders: List,  # 从 dataloader1, dataloader2 改为列表
    # --- 核心训练参数 ---
    num_steps: int,
    scheduler,
    index_no_updates, # 指示哪些词元嵌入不应被更新
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    # --- 学习率调度器 ---
    lr_scheduler_main, # 对应文本嵌入优化器的学习率调度器
    # --- LoRA 相关参数 (TI 中通常不直接使用，但 save_all 可能需要) ---
    lora_unet_target_modules, 
    lora_clip_target_modules, 
    # --- 输出和日志相关 ---
    out_name: str,
    tokenizer,
    test_image_path: str,
    cached_latents: bool,
    # --- 损失函数特定参数 ---
    mask_temperature: float = 1.0,
    t_multiplier_loss: float = 1.0, # 用于 loss_step_gaussian_noise
    save_image_every_n_steps_loss: int = 200, # 用于 loss_step_gaussian_noise
    # --- UNet Feature Alignment 参数 ---
    unet_feature_align_weight: float = 0.01, # UNet特征对齐损失的权重
    unet_feature_alignment_layers=[ # 用于UNet特征对齐的层
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    # --- Text Encoder Feature Alignment 参数 ---
    text_encoder_feature_align_weight: float = 0.02, # Text Encoder特征对齐损失的权重
    text_encoder_alignment_layers=[ # 用于Text Encoder特征对齐的层 - 增加更多层
        'text_model.encoder.layers.0',   # 第一层 - 捕获基础语言特征
        'text_model.encoder.layers.2',   # 早期层 - 词汇理解
        'text_model.encoder.layers.4',   # 中早期层 - 语法结构
        'text_model.encoder.layers.6',   # 中期层 - 语义理解
        'text_model.encoder.layers.8',   # 中后期层 - 复杂语义
        'text_model.encoder.layers.10',  # 后期层 - 高级语义
        'text_model.encoder.layers.11'   # 最后一层 - 最终表示
    ],
    text_encoder_pooling_strategy: str = "mean", # "mean", "cls", "max", "none"
    text_encoder_loss_type: str = "mse", # "mse", "l1", "cosine"
    # --- 主要损失权重 ---
    noise_pred_weight: float = 1.0, # 主要TI损失（噪声预测）的权重
    # --- 其他参数 ---
    unet_return_dict: bool = True, # UNetFeatureExtractor是否返回字典
    accum_iter: int = 1, # 梯度累积步数
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False, # TI中通常为False
    mixed_precision: str = "no", # "no", "fp16", "bf16"
    clip_ti_decay: bool = True, # 是否对TI嵌入应用衰减/正则化
    tensorboard_log_dir: str = "runs",
    teacher_selection_strategy: str = "round_robin",  # 新增：teacher选择策略
    teacher_weights: Optional[List[float]] = None,  # 新增：teacher权重
    # 新增知识蒸馏参数
    use_knowledge_distillation: bool = True,
    kd_temperature: float = 4.0,
    kd_weight: float = 0.5,
    kd_aggregation_strategy: str = "weighted_average",
    use_layer_wise_kd: bool = True,
    use_attention_kd: bool = True,
    kd_attention_weight: float = 0.2,
):
    """
    train_inversion_with_dual_feature_alignment 函数：
    支持多teacher的增强版文本逆向训练，同时使用UNet和Text Encoder的特征对齐。
    
    新增功能：
    1. 支持任意数量的teacher模型
    2. UNet层间特征对齐 (原有功能)
    3. Text Encoder层间特征对齐 (新增功能)
    4. 可以独立控制两种特征对齐的权重
    5. 支持多种teacher选择策略
    """

    # 验证输入参数
    assert len(teacher_unets) == len(teacher_text_encoders) == len(dataloaders), \
        "teacher_unets, teacher_text_encoders, and dataloaders must have the same length"
    
    num_teachers = len(teacher_unets)
    print(f"训练使用 {num_teachers} 个教师模型")

    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_inversion_multi_teacher_dual_feat_align")
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard 日志 (多teacher双重特征对齐的文本逆向) 将保存到: {tb_log_path}")

    # --- UNet特征对齐组件设置 ---
    unet_alignment_layers_for_loss = unet_feature_alignment_layers
    unet_layer_weights_for_fa_loss = {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }
    unet_layer_weights_for_fa_loss = {
        k: v for k, v in unet_layer_weights_for_fa_loss.items() 
        if k in unet_alignment_layers_for_loss
    }

    unet_feature_alignment_loss_fn = FeatureAlignmentLoss(
        alignment_layers=unet_alignment_layers_for_loss,
        loss_weights=unet_layer_weights_for_fa_loss,
        loss_type="mse"
    )

    # 为每个teacher创建UNet特征提取器
    teacher_unet_extractors = []
    for i, teacher_unet in enumerate(teacher_unets):
        extractor = UNetFeatureExtractor(
            target_layers=unet_alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        )
        teacher_unet_extractors.append(extractor)
    
    # 创建学生UNet特征提取器
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )
    
    # 初始化dataloader迭代器
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0  # 用于round_robin策略

    # Text Encoder特征提取器
    text_encoder_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )

    # Text Encoder特征对齐损失函数
    text_encoder_feature_alignment_loss_fn = TextEncoderFeatureAlignmentLoss(
        alignment_layers=text_encoder_alignment_layers,
        pooling_strategy=text_encoder_pooling_strategy,
        loss_type=text_encoder_loss_type
    )

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("训练步数 (多teacher文本逆向 + 双重特征对齐)")
    global_step = 0

    # 备份原始的、非占位符的词元嵌入
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        if wandb.run is None:
            wandb.init(
                project=f"textual_inversion_project_{out_name}", 
                name=f"{out_name}_inversion_multi_teacher_dual_feat_align_run", 
                reinit=True
            )
        wandb.config.update({
            k:v for k,v in locals().items() 
            if isinstance(v, (int, float, str, bool, list, dict))
        })
        preped_clip = prepare_clip_model_sets() if 'prepare_clip_model_sets' in globals() else None

    index_updates = ~index_no_updates

    # 用于区间日志记录的累积损失
    accumulated_total_loss = 0.0
    accumulated_ti_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    # 从模型确定设备
    unet_device = next(student_unet.parameters()).device
    text_encoder_device = next(student_text_encoder.parameters()).device
    vae_device = vae.device

    # 主训练循环
    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # --- 选择当前教师模型和数据加载器 ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # 获取当前选择的模型和数据
        current_teacher_unet = teacher_unets[current_teacher_idx]
        current_teacher_text_encoder = teacher_text_encoders[current_teacher_idx]
        current_teacher_unet_extractor = teacher_unet_extractors[current_teacher_idx]
        current_dataloader_iter = dataloader_iters[current_teacher_idx]
        current_dataloader_obj = dataloaders[current_teacher_idx]
        teacher_name_log = f"T{current_teacher_idx + 1}"
        
        # 更新round_robin索引
        if teacher_selection_strategy == "round_robin":
            teacher_selection_index = (teacher_selection_index + 1) % num_teachers
        
        # 获取批次数据
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            dataloader_iters[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(dataloader_iters[current_teacher_idx])

        # --- 潜变量准备 ---
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

        # --- 准备输入数据 ---
        input_ids = batch["input_ids"].to(device=text_encoder_device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=text_encoder_device)

        # --- 文本编码和特征提取 ---
        main_teacher_unet = current_teacher_unet.module if hasattr(current_teacher_unet, 'module') else current_teacher_unet
        teacher_unet_internal_dtype = next(main_teacher_unet.parameters()).dtype
        student_unet_internal_dtype = next(main_student_unet.parameters()).dtype

        # 教师Text Encoder编码和特征提取 (不需要梯度)
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

        # 学生Text Encoder编码和特征提取 (需要梯度用于TI)
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and text_encoder_device.type == 'cuda')):
            student_text_output, student_text_features = text_encoder_feature_extractor.extract_features(
                text_encoder=student_text_encoder,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_grad=True  # 允许梯度流向文本嵌入
            )
            student_encoder_hidden_states = student_text_output.to(dtype=student_unet_internal_dtype)

        # --- 噪声和时间步 ---
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

        # --- 前向传播和损失计算 ---
        with torch.set_grad_enabled(True):
            # 1. 教师UNet前向传播和特征提取 (不需要梯度)
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

            # 2. 学生UNet前向传播和特征提取 (梯度需要流向文本嵌入)
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and noisy_latents_for_student.device.type == 'cuda')):
                _, student_unet_features = student_unet_extractor.extract_features(
                    unet_model=student_unet, 
                    sample=noisy_latents_for_student,
                    timestep=timesteps_for_student,
                    encoder_hidden_states=student_encoder_hidden_states,
                    return_dict=unet_return_dict,
                    use_grad=True
                )
            
            # 3. 主要TI损失 (噪声预测损失)
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

            # 4. UNet特征对齐损失
            valid_teacher_unet_features = isinstance(teacher_unet_features, dict) and teacher_unet_features
            valid_student_unet_features = isinstance(student_unet_features, dict) and student_unet_features

            if valid_teacher_unet_features and valid_student_unet_features:
                unet_feature_align_loss_val, unet_feature_loss_dict_current = unet_feature_alignment_loss_fn(
                    teacher_unet_features, student_unet_features
                )
            else:
                unet_feature_align_loss_val = torch.tensor(0.0, device=ti_loss_current_step.device, dtype=ti_loss_current_step.dtype)
                unet_feature_loss_dict_current = {}

            # 5. Text Encoder特征对齐损失
            valid_teacher_text_features = isinstance(teacher_text_features, dict) and teacher_text_features
            valid_student_text_features = isinstance(student_text_features, dict) and student_text_features

            if valid_teacher_text_features and valid_student_text_features:
                text_encoder_feature_align_loss_val, text_encoder_feature_loss_dict_current = text_encoder_feature_alignment_loss_fn(
                    teacher_features=teacher_text_features,
                    student_features=student_text_features,
                    teacher_attention_mask=attention_mask,
                    student_attention_mask=attention_mask
                )
            else:
                text_encoder_feature_align_loss_val = torch.tensor(0.0, device=ti_loss_current_step.device, dtype=ti_loss_current_step.dtype)
                text_encoder_feature_loss_dict_current = {}

            # === 新增：多Teacher知识蒸馏损失 ===
            kd_loss = torch.tensor(0.0, device=ti_loss_current_step.device, dtype=ti_loss_current_step.dtype)
            kd_loss_details = {}

            if use_knowledge_distillation:
                # 1. 文本编码器知识蒸馏
                text_kd_loss, text_kd_details = compute_multi_teacher_kl_distillation_loss(
                    teacher_text_encoders=teacher_text_encoders,
                    student_text_encoder=student_text_encoder,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=kd_temperature,
                    teacher_weights=teacher_weights,
                    aggregation_strategy=kd_aggregation_strategy,
                    use_layer_wise_distillation=use_layer_wise_kd
                )
                
                kd_loss += kd_weight * text_kd_loss.float()
                kd_loss_details.update(text_kd_details)
                
                # 2. 注意力蒸馏（可选）
                if use_attention_kd:
                    attention_kd_loss = compute_multi_teacher_attention_distillation_loss(
                        teacher_text_encoders=teacher_text_encoders,
                        student_text_encoder=student_text_encoder,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        teacher_weights=teacher_weights,
                        attention_temperature=kd_temperature
                    )
                    
                    kd_loss += kd_attention_weight * attention_kd_loss.float()
                    kd_loss_details['attention_kd'] = attention_kd_loss.item()

            # 6. 动态权重调整
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

            # 7. 总损失
            current_step_total_loss = (
                noise_pred_weight * ti_loss_current_step +
                unet_feature_align_weight * unet_feature_align_loss_val +
                text_encoder_feature_align_weight * text_encoder_feature_align_loss_val +
                1.0 * kd_loss  # 知识蒸馏损失
            )
            
            loss_for_backward = current_step_total_loss / accum_iter
            loss_for_backward.backward()

            # 记录单步损失
            accumulated_total_loss += current_step_total_loss.detach().item()
            accumulated_ti_loss += ti_loss_current_step.detach().item()
            accumulated_unet_feature_loss += unet_feature_align_loss_val.detach().item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val)
            accumulated_text_encoder_feature_loss += text_encoder_feature_align_loss_val.detach().item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val)
            step_count_in_interval += 1

        # --- 优化器步骤和嵌入正则化 ---
        if (global_step + 1) % accum_iter == 0:
            if student_text_encoder.get_input_embeddings().weight.grad is not None:
                grad_slice = student_text_encoder.get_input_embeddings().weight.grad[index_updates, :]
                if grad_slice.numel() > 0:
                    grad_norm = grad_slice.norm(dim=-1).mean()
                    if writer:
                        writer.add_scalar('梯度/文本嵌入范数_多teacher双重FA', grad_norm.item(), global_step)
            else:
                print(f"步骤 {global_step}: 警告：在多teacher双重FA更新期间未找到文本嵌入的梯度。")

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 文本嵌入正则化
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
                    writer.add_scalar('嵌入/当前范数_多teacher双重FA', current_norm_val, global_step)

                # 恢复未被更新的原始嵌入
                embed_weights.data[index_no_updates] = orig_embeds_params[index_no_updates]
        
        lr_scheduler_main.step()
        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0]
        
        log_dict_postfix = {
            "总损失": f"{current_step_total_loss.item():.4f}",
            "TI损失": f"{ti_loss_current_step.item():.4f}",
            "UNet特征": f"{unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val):.4f}",
            "Text特征": f"{text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val):.4f}",
            "LR": f"{current_lr_val:.2e}",
            "教师": teacher_name_log
        }
        progress_bar.set_postfix(**log_dict_postfix)

        # TensorBoard 日志记录
        if writer:
            writer.add_scalar('损失_多teacher双重FA/总损失_单步', current_step_total_loss.item(), global_step)
            writer.add_scalar('损失_多teacher双重FA/TI损失_单步', ti_loss_current_step.item(), global_step)
            writer.add_scalar('损失_多teacher双重FA/UNet特征对齐损失_单步', unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val), global_step)
            writer.add_scalar('损失_多teacher双重FA/TextEncoder特征对齐损失_单步', text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val), global_step)
            writer.add_scalar('学习率_多teacher双重FA/文本嵌入', current_lr_val, global_step)
            writer.add_scalar('教师模型_多teacher双重FA/当前选择', current_teacher_idx + 1, global_step)
            writer.add_scalar('权重_多teacher/TI损失权重', noise_pred_weight, global_step)
            writer.add_scalar('权重_多teacher/UNet特征权重', unet_feature_align_weight, global_step)
            writer.add_scalar('权重_多teacher/TextEncoder特征权重', text_encoder_feature_align_weight, global_step)
            writer.add_scalar('损失_知识蒸馏/总KD损失', kd_loss.item(), global_step)
            
            for kd_type, kd_value in kd_loss_details.items():
                writer.add_scalar(f'损失_知识蒸馏/{kd_type}', kd_value, global_step)
            
            # 记录各层的UNet特征损失
            if isinstance(unet_feature_loss_dict_current, dict):
                for layer_name, layer_loss in unet_feature_loss_dict_current.items():
                    writer.add_scalar(f'UNet特征损失_多teacher双重FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
            
            # 记录各层的Text Encoder特征损失
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                for layer_name, layer_loss in text_encoder_feature_loss_dict_current.items():
                    writer.add_scalar(f'TextEncoder特征损失_多teacher双重FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
        
        # wandb 日志记录
        wandb_step_logs = {}
        if log_wandb and global_step % 10 == 0:
            wandb_step_logs.update({
                'step': global_step,
                'total_loss_step_multi_teacher_dual_fa': current_step_total_loss.item(),
                'ti_loss_step_multi_teacher_dual_fa': ti_loss_current_step.item(),
                'unet_feature_align_loss_step_multi_teacher_dual_fa': unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val),
                'text_encoder_feature_align_loss_step_multi_teacher_dual_fa': text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val),
                'learning_rate_multi_teacher_dual_fa': current_lr_val,
                'teacher_model_used_multi_teacher_dual_fa': current_teacher_idx + 1,
            })
            
            # 添加各层特征损失
            if isinstance(unet_feature_loss_dict_current, dict):
                wandb_step_logs.update({
                    f"unet_feat_align_layer_{k.replace('.', '_')}_multi_teacher_dual_fa": v_loss.item() if torch.is_tensor(v_loss) else float(v_loss)
                    for k, v_loss in unet_feature_loss_dict_current.items()
                })
            
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                wandb_step_logs.update({
                    f"text_encoder_feat_align_layer_{k.replace('.', '_')}_multi_teacher_dual_fa": v_loss.item() if torch.is_tensor(v_loss) else float(v_loss)
                    for k, v_loss in text_encoder_feature_loss_dict_current.items()
                })

        # --- 保存检查点和评估 ---
        if global_step > 0 and global_step % save_steps == 0:
            current_save_dir = os.path.join(save_path, out_name)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir, exist_ok=True)
            
            save_all( 
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=os.path.join(current_save_dir, f"step_inv_multi_teacher_dual_fa_{global_step}.safetensors"),
                save_lora=False,
            )
            print(f"\n ✓ 多teacher双重FA检查点已保存: step_inv_multi_teacher_dual_fa_{global_step}.safetensors")

            # 计算区间平均损失
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_ti_loss_interval = accumulated_ti_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_unet_feature_loss_interval = accumulated_unet_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_text_encoder_feature_loss_interval = accumulated_text_encoder_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0

            if writer:
                writer.add_scalar('损失_多teacher双重FA/总损失_平均区间', avg_total_loss_interval, global_step)
                writer.add_scalar('损失_多teacher双重FA/TI损失_平均区间', avg_ti_loss_interval, global_step)
                writer.add_scalar('损失_多teacher双重FA/UNet特征对齐损失_平均区间', avg_unet_feature_loss_interval, global_step)
                writer.add_scalar('损失_多teacher双重FA/TextEncoder特征对齐损失_平均区间', avg_text_encoder_feature_loss_interval, global_step)
            
            wandb_interval_logs = {
                "loss_total_avg_interval_multi_teacher_dual_fa": avg_total_loss_interval,
                "loss_ti_avg_interval_multi_teacher_dual_fa": avg_ti_loss_interval,
                "loss_unet_feature_align_avg_interval_multi_teacher_dual_fa": avg_unet_feature_loss_interval,
                "loss_text_encoder_feature_align_avg_interval_multi_teacher_dual_fa": avg_text_encoder_feature_loss_interval,
            }
            if wandb_step_logs:
                wandb_interval_logs.update(wandb_step_logs)

            # 重置累积损失
            accumulated_total_loss = 0.0
            accumulated_ti_loss = 0.0
            accumulated_unet_feature_loss = 0.0
            accumulated_text_encoder_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    writer.close()

    # --- 保存最终模型 ---
    print(f"\n" + "="*80)
    print(f"多TEACHER训练完成!")
    print(f"="*80)
    print(f"总完成步数: {global_step}")
    print(f"使用的teacher模型数量: {num_teachers}")
    print(f"最终学习率: {current_lr_val:.2e}")
    
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_step_inv_multi_teacher_{global_step}.safetensors")
    save_all(
        unet=student_unet, text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path,
        save_lora=False,
    )
    
    print(f"✓ 最终模型已保存至: {final_model_path}")
    print(f"✓ TensorBoard日志已保存至: {tb_log_path}")
    print(f"多teacher特征对齐训练完成!")
    print(f"="*80)

def perform_tuning_multi_teacher(
    # --- "Teacher" 模型参数 (修改为列表形式) ---
    teacher_unets: List,  # 从 teacher1_unet, teacher2_unet 改为列表
    teacher_text_encoders: List,  # 从 teacher1_text_encoder, teacher2_text_encoder 改为列表
    # --- "Student" 模型参数 ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader 参数 (修改为列表形式) ---
    dataloaders: List,  # 从 dataloader1, dataloader2 改为列表
    # --- 核心训练参数 ---
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
    # --- 可选参数 ---
    mixed_precision="no",
    log_wandb=False,
    wandb_log_prompt_cnt=10,
    class_token="person",
    train_inpainting=False,
    feature_align_weight=0.001,
    noise_pred_weight=1.0,
    feature_alignment_unet_layers=[
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    unet_return_dict=True,
    tensorboard_log_dir: str = "runs",
    # --- 新增多teacher参数 ---
    teacher_selection_strategy: str = "round_robin",  # 新增：teacher选择策略
    teacher_weights: Optional[List[float]] = None,  # 新增：teacher权重
):
    """
    执行支持多teacher的LoRA微调，包含特征对齐
    
    Args:
        teacher_unets: List of teacher UNet models
        teacher_text_encoders: List of teacher text encoders
        dataloaders: List of dataloaders corresponding to each teacher
        teacher_selection_strategy: "round_robin", "weighted_random", "adaptive"
        teacher_weights: Weights for each teacher (used with weighted_random)
    """
    
    # 验证输入参数
    assert len(teacher_unets) == len(teacher_text_encoders) == len(dataloaders), \
        "teacher_unets, teacher_text_encoders, and dataloaders must have the same length"
    
    num_teachers = len(teacher_unets)
    print(f"LoRA调优使用 {num_teachers} 个教师模型")
    
    import os
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    
    print(f"=" * 80)
    print(f"Starting Multi-Teacher LoRA Tuning with Feature Alignment")
    print(f"Number of teachers: {num_teachers}")
    print(f"Total steps: {num_steps}")
    print(f"Save steps: {save_steps}")
    print(f"Teacher selection strategy: {teacher_selection_strategy}")
    print(f"Feature alignment weight: {feature_align_weight}")
    print(f"Noise prediction weight: {noise_pred_weight}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Output name: {out_name}")
    print(f"=" * 80)

    # --- 初始化TensorBoard SummaryWriter ---
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_multi_teacher_lora_tuning")
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging to: {tb_log_path}")

    # 特征对齐损失函数组件
    alignment_layers_for_loss = feature_alignment_unet_layers
    layer_weights_for_loss = {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }
    layer_weights_for_loss = {k: v for k, v in layer_weights_for_loss.items() if k in alignment_layers_for_loss}

    feature_alignment_loss_fn = FeatureAlignmentLoss(
        alignment_layers=alignment_layers_for_loss,
        loss_weights=layer_weights_for_loss,
        loss_type="mse"
    )

    # 为每个teacher创建UNet特征提取器
    teacher_extractors = []
    for i, teacher_unet in enumerate(teacher_unets):
        extractor = UNetFeatureExtractor(
            target_layers=feature_alignment_unet_layers,
            mixed_precision_config=mixed_precision
        )
        teacher_extractors.append(extractor)
    
    # 创建学生UNet特征提取器
    student_extractor = UNetFeatureExtractor(
        target_layers=feature_alignment_unet_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps), desc="Multi-Teacher LoRA Tuning with Feature Alignment")
    global_step = 0

    # 初始化dataloader迭代器
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0  # 用于round_robin策略

    # 累积损失用于日志记录 (在每个 save_steps 周期内)
    accumulated_total_loss = 0.0
    accumulated_noise_loss = 0.0
    accumulated_feature_loss = 0.0
    step_count_in_interval = 0

    student_unet.train()
    student_text_encoder.train()

    for step_idx in range(num_steps):
        # --- 选择当前教师模型 ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # 获取当前选择的模型和数据
        current_teacher_unet = teacher_unets[current_teacher_idx]
        current_teacher_text_encoder = teacher_text_encoders[current_teacher_idx]
        current_teacher_extractor = teacher_extractors[current_teacher_idx]
        current_dataloader_iter = dataloader_iters[current_teacher_idx]
        current_dataloader_obj = dataloaders[current_teacher_idx]
        teacher_name_log = f"T{current_teacher_idx + 1}"
        
        # 更新round_robin索引
        if teacher_selection_strategy == "round_robin":
            teacher_selection_index = (teacher_selection_index + 1) % num_teachers

        # 获取批次数据
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            print(f"Info: dataloader for teacher {current_teacher_idx + 1} exhausted at step {global_step}. Re-initializing.")
            dataloader_iters[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(dataloader_iters[current_teacher_idx])

        optimizer.zero_grad()

        # --- Latent 处理 ---
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

        # --- 文本编码 ---
        text_encoder_device = student_text_encoder.device if hasattr(student_text_encoder, 'device') else main_module.device
        input_ids = batch["input_ids"].to(device=text_encoder_device)
        student_input_ids = batch.get('aux1_input_ids', input_ids).to(device=text_encoder_device)

        teacher_unet_internal_dtype = next(current_teacher_unet.parameters()).dtype
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
                teacher_raw_hidden_states = current_teacher_text_encoder(
                    input_ids, output_hidden_states=True
                ).hidden_states[-1]
                teacher_encoder_hidden_states = teacher_raw_hidden_states.to(dtype=teacher_unet_internal_dtype)

        student_unet_internal_dtype = next(student_unet.parameters()).dtype
        with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
            student_raw_hidden_states = student_text_encoder(
                student_input_ids, output_hidden_states=True
            ).hidden_states[-1]
            student_encoder_hidden_states = student_raw_hidden_states.to(dtype=student_unet_internal_dtype)

        # --- 噪声和时间步 ---
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

        # --- 教师模型前向传播 ---
        with torch.no_grad():
            teacher_noise_pred, teacher_features = current_teacher_extractor.extract_features(
                unet_model=current_teacher_unet,
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=teacher_encoder_hidden_states,
                return_dict=unet_return_dict,
                use_grad=False
            )

        # --- 学生模型前向传播（仅提取特征用于对齐损失）---
        _, student_features = student_extractor.extract_features(
            unet_model=student_unet,
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=student_encoder_hidden_states,
            return_dict=unet_return_dict,
            use_grad=True
        )

        # --- 计算损失 ---
        # 噪声预测损失：使用原有的 loss_step_gaussian_noise 函数
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
        )

        # 特征对齐损失：学生模型特征 vs 教师模型特征
        feature_align_loss, feature_loss_dict = feature_alignment_loss_fn(
            teacher_features, student_features
        )

        tuning_weight_adjuster = create_tuning_weight_adjuster(
            noise_pred_weight=noise_pred_weight,
            feature_align_weight=feature_align_weight
        )

        current_losses = {
            'noise_pred': noise_pred_loss,
            'feature_align': feature_align_loss,
        }

        updated_weights = tuning_weight_adjuster.update_weights(current_losses)
        noise_pred_weight = updated_weights['noise_pred']
        feature_align_weight = updated_weights['feature_align']

        # 总损失
        total_loss = (
            noise_pred_weight * noise_pred_loss +
            feature_align_weight * feature_align_loss
        )

        # --- 反向传播和优化器步骤 ---
        total_loss.backward()
        
        # 可选：梯度裁剪
        # torch.nn.utils.clip_grad_norm_(
        #     itertools.chain(student_unet.parameters(), student_text_encoder.parameters()), 1.0
        # )
        
        optimizer.step()
        lr_scheduler_lora.step()

        # --- 记录和日志 ---
        current_total_loss_item = total_loss.item()
        current_noise_loss_item = noise_pred_loss.item()
        current_feature_loss_item = feature_align_loss.item() if isinstance(feature_align_loss, torch.Tensor) else feature_align_loss

        accumulated_total_loss += current_total_loss_item
        accumulated_noise_loss += current_noise_loss_item
        accumulated_feature_loss += current_feature_loss_item
        step_count_in_interval += 1

        current_lr = lr_scheduler_lora.get_last_lr()[0]
        progress_bar.set_postfix({
            'total_loss': f'{current_total_loss_item:.4f}',
            'noise_loss': f'{current_noise_loss_item:.4f}',
            'feature_loss': f'{current_feature_loss_item:.4f}',
            'lr': f'{current_lr:.2e}',
            'teacher': teacher_name_log  # 显示使用的是哪个教师模型
        })
        progress_bar.update(1)

        # TensorBoard 记录每步
        writer.add_scalar('Loss_多teacher/Total_Step', current_total_loss_item, global_step)
        writer.add_scalar('Loss_多teacher/Noise_Pred_Step', current_noise_loss_item, global_step)
        writer.add_scalar('Loss_多teacher/Feature_Align_Step', current_feature_loss_item, global_step)
        writer.add_scalar('LearningRate_多teacher', current_lr, global_step)
        writer.add_scalar('Teacher_Model_多teacher/当前选择', current_teacher_idx + 1, global_step)
        writer.add_scalar('Weights_多teacher/Noise_Pred_Weight', noise_pred_weight, global_step)
        writer.add_scalar('Weights_多teacher/Feature_Align_Weight', feature_align_weight, global_step)
        
        # 记录各层特征损失
        if isinstance(feature_loss_dict, dict):
            for layer_name, layer_loss in feature_loss_dict.items():
                layer_loss_value = layer_loss.item() if isinstance(layer_loss, torch.Tensor) else layer_loss
                writer.add_scalar(f'FeatureLoss_多teacher/{layer_name.replace(".", "_")}', layer_loss_value, global_step)

        # wandb 记录
        if log_wandb and global_step % 10 == 0:
            log_data_wandb = {
                'step': global_step,
                'total_loss_step_multi_teacher': current_total_loss_item,
                'noise_pred_loss_step_multi_teacher': current_noise_loss_item,
                'feature_align_loss_step_multi_teacher': current_feature_loss_item,
                'learning_rate_multi_teacher': current_lr,
                'teacher_model_used_multi_teacher': current_teacher_idx + 1,
            }
            if isinstance(feature_loss_dict, dict):
                log_data_wandb.update({
                    f"feature_loss_layer_{k.replace('.', '_')}_multi_teacher": 
                    v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss 
                    for k, v_loss in feature_loss_dict.items()
                })
            # wandb.log(log_data_wandb)

        global_step += 1

        # --- 保存检查点 ---
        if global_step > 0 and global_step % save_steps == 0:
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval
            avg_noise_loss_interval = accumulated_noise_loss / step_count_in_interval
            avg_feature_loss_interval = accumulated_feature_loss / step_count_in_interval

            print(f"\n" + "="*60)
            print(f"MULTI-TEACHER CHECKPOINT - Step {global_step}/{num_steps}")
            print(f"="*60)
            print(f"Average Losses (last {step_count_in_interval} steps):")
            print(f"  • Total Loss:           {avg_total_loss_interval:.6f}")
            print(f"  • Noise Prediction:     {avg_noise_loss_interval:.6f} (weight: {noise_pred_weight})")
            print(f"  • Feature Alignment:    {avg_feature_loss_interval:.6f} (weight: {feature_align_weight})")
            print(f"  • Learning Rate:        {current_lr:.2e}")
            print(f"  • Progress:             {(global_step/num_steps)*100:.1f}%")
            print(f"  • Current Teacher:      Teacher {current_teacher_idx + 1}/{num_teachers}")
            print(f"  • Teacher Strategy:     {teacher_selection_strategy}")
            
            # 显示各层特征损失详情
            if isinstance(feature_loss_dict, dict) and feature_loss_dict:
                print(f"Feature Loss Details:")
                for layer_name, layer_loss in feature_loss_dict.items():
                    layer_loss_value = layer_loss.item() if isinstance(layer_loss, torch.Tensor) else layer_loss
                    print(f"  • {layer_name:<20}: {layer_loss_value:.6f}")
            print(f"="*60)

            # TensorBoard 记录区间平均值
            writer.add_scalar('Loss_多teacher/Total_Avg_Interval', avg_total_loss_interval, global_step)
            writer.add_scalar('Loss_多teacher/Noise_Pred_Avg_Interval', avg_noise_loss_interval, global_step)
            writer.add_scalar('Loss_多teacher/Feature_Align_Avg_Interval', avg_feature_loss_interval, global_step)

            # 重置累积损失
            accumulated_total_loss = 0.0
            accumulated_noise_loss = 0.0
            accumulated_feature_loss = 0.0
            step_count_in_interval = 0

            # 保存模型
            current_save_dir = os.path.join(save_path, out_name)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir, exist_ok=True)

            checkpoint_path = os.path.join(current_save_dir, f"multi_teacher_step_{global_step}.safetensors")
            save_all(
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=checkpoint_path,
                target_replace_module_unet=lora_unet_target_modules,
                target_replace_module_text=lora_clip_target_modules
            )
            print(f"✓ Multi-Teacher Checkpoint saved to: {checkpoint_path}\n")

        if global_step >= num_steps:
            break

    progress_bar.close()
    writer.close()

    # --- 保存最终模型 ---
    print(f"\n" + "="*80)
    print(f"MULTI-TEACHER TRAINING COMPLETED!")
    print(f"="*80)
    print(f"Total steps completed: {global_step}")
    print(f"Number of teachers used: {num_teachers}")
    print(f"Teacher selection strategy: {teacher_selection_strategy}")
    print(f"Final learning rate: {current_lr:.2e}")
    
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_multi_teacher_step_{global_step}.safetensors")
    save_all(
        unet=student_unet,
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path,
        target_replace_module_unet=lora_unet_target_modules,
        target_replace_module_text=lora_clip_target_modules
    )
    
    print(f"✓ Final multi-teacher model saved to: {final_model_path}")
    print(f"✓ TensorBoard logs saved to: {tb_log_path}")
    print(f"Multi-teacher training completed with feature alignment!")
    print(f"="*80)

def discover_lora_models(lora_models_dir: str, lora_path1: str = None, lora_path2: str = None):
    """
    发现LoRA模型文件
    
    Args:
        lora_models_dir: LoRA模型文件夹路径
        lora_path1, lora_path2: 向后兼容的单个路径参数
    
    Returns:
        teacher_lora_paths: LoRA文件路径列表
        teacher_info: 包含文件名等信息的列表
    """
    teacher_lora_paths = []
    teacher_info = []
    
    # 向后兼容：如果提供了单个路径，使用它们
    if lora_path1 and lora_path2:
        teacher_lora_paths = [lora_path1, lora_path2]
        teacher_info = [
            {"name": "teacher1", "filename": os.path.basename(lora_path1)},
            {"name": "teacher2", "filename": os.path.basename(lora_path2)}
        ]
        return teacher_lora_paths, teacher_info
    
    # 新方式：从文件夹中发现LoRA模型
    if not os.path.exists(lora_models_dir):
        raise ValueError(f"LoRA模型文件夹不存在: {lora_models_dir}")
    
    # 支持的LoRA文件扩展名
    supported_extensions = ['.safetensors', '.pt', '.pth', '.bin']
    
    for filename in sorted(os.listdir(lora_models_dir)):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_path = os.path.join(lora_models_dir, filename)
            if os.path.isfile(file_path):
                # 从文件名提取模型名称（去除扩展名）
                model_name = os.path.splitext(filename)[0]
                teacher_lora_paths.append(file_path)
                teacher_info.append({
                    "name": model_name,
                    "filename": filename,
                    "path": file_path
                })
    
    return teacher_lora_paths, teacher_info

def generate_placeholder_tokens_from_lora_names(teacher_info: List[dict]):
    """
    根据LoRA文件名生成placeholder tokens
    
    Args:
        teacher_info: 包含LoRA信息的列表
    
    Returns:
        all_placeholder_tokens: 所有placeholder tokens的列表
        all_initializer_tokens: 所有initializer tokens的列表
    """
    all_placeholder_tokens = []
    all_initializer_tokens = []
    
    for info in teacher_info:
        model_name = info["name"]
        # 清理文件名，生成有效的token
        clean_name = clean_filename_for_token(model_name)
        placeholder_token = f"<{clean_name}>"
        
        all_placeholder_tokens.append(placeholder_token)
        all_initializer_tokens.append("<rand-0.017>")  # 随机初始化
    
    return all_placeholder_tokens, all_initializer_tokens

def clean_filename_for_token(filename: str) -> str:
    """
    清理文件名以生成有效的token名称
    """
    import re
    # 移除特殊字符，只保留字母、数字和下划线
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    # 移除连续的下划线
    clean = re.sub(r'_+', '_', clean)
    # 移除开头和结尾的下划线
    clean = clean.strip('_')
    # 确保不为空
    if not clean:
        clean = "model"
    return clean

def parse_manual_tokens(placeholder_tokens1: str, placeholder_tokens2: str, initializer_tokens: str):
    """
    解析手动指定的tokens（向后兼容）
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
    创建学生模型
    """
    # 使用修改后的get_models函数来支持任意数量的tokens
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
    支持任意数量tokens的模型创建函数
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
    创建多个教师模型
    """
    teacher_models = []
    
    for i, lora_path in enumerate(teacher_lora_paths):
        print(f"正在加载教师模型 {i+1}/{len(teacher_lora_paths)}: {lora_path}")
        
        # 创建pipeline
        teacher_pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.float16
        ).to(device)
        
        teacher_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            teacher_pipe.scheduler.config
        )
        
        # 加载LoRA权重
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
        
        print(f"✓ 教师模型 {i+1} 加载完成")
    
    return teacher_models

def create_multiple_dataloaders(teacher_models: List[dict], teacher_info: List[dict], 
                               student_tokenizer, train_batch_size: int, cached_latents: bool,
                               use_template, device: str):
    """
    为每个教师模型创建对应的数据加载器
    """
    dataloaders = []
    token_maps = []
    
    for i, (teacher_model, info) in enumerate(zip(teacher_models, teacher_info)):
        model_name = info["name"]
        
        # 为每个模型创建token_map
        token_map = {"DUMMY": f"<{clean_filename_for_token(model_name)}>"}
        token_maps.append(token_map)
        
        # 创建数据集
        dataset = PivotalTuningDatasetCapationLoraGenerated(
            sd_pipeline=teacher_model["pipeline"],
            device=device,
            main_tokenizer=teacher_model["tokenizer"],
            use_template=use_template,
            token_map=token_map,
            dataset_size=10,  # 可以根据需要调整
            transform_size=512,
            h_flip=True,
            aux_tokenizer1=student_tokenizer,
            save_generated_images_path="/root/lora_train/pic",
            save_image_prefix=f"teacher_{i+1}_{model_name}"
        )
        
        # 创建数据加载器
        dataloader = text2img_dataloader_combined_with_latent_caching(
            train_dataset=dataset,
            train_batch_size=train_batch_size,
            main_tokenizer=teacher_model["tokenizer"],
            aux_tokenizer1=student_tokenizer,
            vae=teacher_model["vae"],
            cached_latents=cached_latents
        )
        
        dataloaders.append(dataloader)
        print(f"✓ 教师模型 {i+1} 数据加载器创建完成")
    
    return dataloaders, token_maps

def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
                               gradient_checkpointing, enable_xformers_memory_efficient_attention,
                               placeholder_token_ids):
    """
    设置模型训练配置
    """
    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            student_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    # 冻结文本编码器的大部分参数
    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.encoder.parameters(),
        student_text_encoder.text_model.final_layer_norm.parameters(),
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

def calculate_learning_rates(learning_rate_unet, learning_rate_text, learning_rate_ti,
                           scale_lr, gradient_accumulation_steps, train_batch_size):
    """
    计算学习率
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
    设置LoRA参数
    """
    # UNet LoRA设置
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("使用扩展UNet LoRA")
        lora_unet_target_modules = lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        unet_lora_params, _ = inject_trainable_lora_extended(
            student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    # 文本编码器设置
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

def train(
    # 修改主要参数以支持文件夹输入
    lora_models_dir: str,  # 新增：包含多个LoRA模型的文件夹路径
    lora_path1: str = None,  # 保持向后兼容，设为可选
    lora_path2: str = None,  # 保持向后兼容，设为可选
    instance_data_dir: str = "",
    pretrained_model_name_or_path: str = "",
    output_dir: str = "",
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = False,
    use_template: Literal[None, "object", "style"] = None,
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    placeholder_tokens1: str = "",
    placeholder_tokens2: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    # 新增多teacher相关参数
    teacher_selection_strategy: str = "round_robin",
    teacher_weights: Optional[List[float]] = None,
    auto_generate_placeholder_tokens: bool = True,  # 是否根据文件名自动生成placeholder tokens
    # 其他参数保持不变...
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
    lora_rank: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPSdpaAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_lora_log2",
    feature_align_weight: float = 0.01,
    noise_pred_weight: float = 1.0,
):
    """
    支持多teacher蒸馏的训练函数
    
    Args:
        lora_models_dir: 包含多个LoRA模型文件的文件夹路径
        auto_generate_placeholder_tokens: 是否根据LoRA文件名自动生成placeholder tokens
        teacher_selection_strategy: teacher选择策略 ("round_robin", "weighted_random", "adaptive")
        teacher_weights: teacher权重列表（用于weighted_random策略）
    """
    
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"multi_teacher_steps_{max_train_steps_ti}_lr_{learning_rate_ti}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. 发现和加载多个LoRA模型 ---
    teacher_lora_paths, teacher_info = discover_lora_models(lora_models_dir, lora_path1, lora_path2)
    num_teachers = len(teacher_lora_paths)
    
    print(f"发现 {num_teachers} 个教师LoRA模型:")
    for i, (path, info) in enumerate(zip(teacher_lora_paths, teacher_info)):
        print(f"  教师 {i+1}: {path} -> {info}")

    if num_teachers == 0:
        raise ValueError(f"在 {lora_models_dir} 中未找到有效的LoRA模型文件")

    # --- 2. 根据LoRA文件名生成placeholder tokens ---
    if auto_generate_placeholder_tokens:
        all_placeholder_tokens, all_initializer_tokens = generate_placeholder_tokens_from_lora_names(teacher_info)
        print(f"自动生成的placeholder tokens: {all_placeholder_tokens}")
        print(f"自动生成的initializer tokens: {all_initializer_tokens}")
    else:
        # 使用手动指定的tokens（保持向后兼容）
        all_placeholder_tokens, all_initializer_tokens = parse_manual_tokens(
            placeholder_tokens1, placeholder_tokens2, initializer_tokens
        )

    # --- 3. 创建学生模型 ---
    student_text_encoder, student_vae, student_unet, student_tokenizer, placeholder_token_ids = create_student_model(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pretrained_vae_name_or_path=pretrained_vae_name_or_path,
        revision=revision,
        placeholder_tokens=all_placeholder_tokens,
        initializer_tokens=all_initializer_tokens,
        device=device
    )

    # --- 4. 创建多个教师模型 ---
    teacher_models = create_multiple_teacher_models(
        teacher_lora_paths=teacher_lora_paths,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device
    )

    teacher_unets = [model["unet"] for model in teacher_models]
    teacher_text_encoders = [model["text_encoder"] for model in teacher_models]
    teacher_vaes = [model["vae"] for model in teacher_models]
    teacher_tokenizers = [model["tokenizer"] for model in teacher_models]

    # --- 5. 创建多个数据加载器 ---
    dataloaders, token_maps = create_multiple_dataloaders(
        teacher_models=teacher_models,
        teacher_info=teacher_info,
        student_tokenizer=student_tokenizer,
        train_batch_size=train_batch_size,
        cached_latents=cached_latents,
        use_template=use_template,
        device=device
    )

    # --- 6. 设置噪声调度器 ---
    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    # --- 7. 配置模型训练设置 ---
    setup_model_training_config(
        student_unet=student_unet,
        student_text_encoder=student_text_encoder,
        student_vae=student_vae,
        gradient_checkpointing=gradient_checkpointing,
        enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
        placeholder_token_ids=placeholder_token_ids
    )

    # --- 8. 计算学习率 ---
    unet_lr, text_encoder_lr, ti_lr = calculate_learning_rates(
        learning_rate_unet=learning_rate_unet,
        learning_rate_text=learning_rate_text,
        learning_rate_ti=learning_rate_ti,
        scale_lr=scale_lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size
    )

    # --- 9. 文本逆向训练（如果启用） ---
    if perform_inversion:
        print(f"\n{'='*80}")
        print(f"开始多Teacher文本逆向训练（共 {num_teachers} 个教师模型）")
        print(f"{'='*80}")
        
        ti_optimizer = optim.AdamW(
            student_text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        # 调用多teacher文本逆向训练函数
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
            lr_scheduler_main=lr_scheduler,
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

    # --- 10. LoRA微调训练 ---
    print(f"\n{'='*80}")
    print(f"开始多Teacher LoRA微调训练（共 {num_teachers} 个教师模型）")
    print(f"{'='*80}")

    # 设置LoRA参数
    unet_lora_params, text_encoder_lora_params = setup_lora_parameters(
        student_unet=student_unet,
        student_text_encoder=student_text_encoder,
        train_text_encoder=train_text_encoder,
        continue_inversion=continue_inversion,
        use_extended_lora=use_extended_lora,
        lora_rank=lora_rank,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        lora_dropout_p=lora_dropout_p,
        lora_scale=lora_scale,
        unet_lr=unet_lr,
        text_encoder_lr=text_encoder_lr,
        ti_lr=ti_lr,
        continue_inversion_lr=continue_inversion_lr
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

    # 特征对齐层配置
    feature_alignment_unet_layers = [
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ]

    # 调用多teacher LoRA微调函数
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
    )

    print(f"\n{'='*80}")
    print(f"多Teacher蒸馏训练完成！共使用了 {num_teachers} 个教师模型")
    print(f"最终模型已保存到: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Teacher LoRA Distillation Training')
    
    # --- 主要参数 - 支持文件夹输入和向后兼容 ---
    parser.add_argument('--lora_models_dir', type=str, default=None, 
                       help='Directory containing multiple LoRA model files (new multi-teacher mode)')
    
    # 向后兼容参数
    parser.add_argument('--lora_path1', type=str, default=None, 
                       help='Path to first LoRA model (for backward compatibility)')
    parser.add_argument('--lora_path2', type=str, default=None, 
                       help='Path to second LoRA model (for backward compatibility)')
    
    # 核心必需参数
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                       help='Path to pretrained model or HuggingFace model name')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for saving models')
    
    # --- 新增多teacher相关参数 ---
    parser.add_argument('--teacher_selection_strategy', type=str, default='round_robin',
                       choices=['round_robin', 'weighted_random', 'adaptive'],
                       help='Strategy for selecting teachers during training')
    parser.add_argument('--teacher_weights', type=str, default=None,
                       help='Comma-separated weights for teachers (for weighted_random strategy)')
    parser.add_argument('--auto_generate_placeholder_tokens', action='store_true', default=True,
                       help='Automatically generate placeholder tokens from LoRA filenames')
    
    # --- 数据和模型配置 ---
    parser.add_argument('--instance_data_dir', type=str, default="", 
                       help='Directory containing instance data')
    parser.add_argument('--train_text_encoder', action='store_true', default=True, 
                       help='Whether to train text encoder')
    parser.add_argument('--pretrained_vae_name_or_path', type=str, default=None, 
                       help='Path to pretrained VAE')
    parser.add_argument('--revision', type=str, default=None, 
                       help='Revision of pretrained model')
    
    # --- 训练模式配置 ---
    parser.add_argument('--perform_inversion', action='store_true', default=False, 
                       help='Perform textual inversion training')
    parser.add_argument('--use_template', type=str, choices=[None, 'object', 'style'], default=None, 
                       help='Template to use for data generation')
    parser.add_argument('--train_inpainting', action='store_true', default=False, 
                       help='Train for inpainting tasks')
    
    # --- Placeholder Token 配置 ---
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
    
    # --- 基础训练参数 ---
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
    
    # --- 训练步数配置 ---
    parser.add_argument('--max_train_steps_tuning', type=int, default=1000, 
                       help='Maximum number of LoRA tuning steps')
    parser.add_argument('--max_train_steps_ti', type=int, default=1000, 
                       help='Maximum number of textual inversion steps')
    parser.add_argument('--save_steps', type=int, default=100, 
                       help='Steps between saving checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                       help='Number of gradient accumulation steps')
    
    # --- 模型优化配置 ---
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, 
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', default=False, 
                       help='Enable xformers memory efficient attention')
    
    # --- LoRA 配置 ---
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
    
    # --- 文本嵌入配置 ---
    parser.add_argument('--clip_ti_decay', action='store_true', default=True, 
                       help='Enable CLIP textual inversion decay/regularization')
    
    # --- 学习率配置 ---
    parser.add_argument('--learning_rate_unet', type=float, default=1e-4, 
                       help='Learning rate for UNet training')
    parser.add_argument('--learning_rate_text', type=float, default=1e-5, 
                       help='Learning rate for text encoder training')
    parser.add_argument('--learning_rate_ti', type=float, default=5e-4, 
                       help='Learning rate for textual inversion')
    parser.add_argument('--scale_lr', action='store_true', default=False, 
                       help='Scale learning rate by batch size and accumulation steps')
    
    # --- 学习率调度器配置 ---
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
    
    # --- 继续训练配置 ---
    parser.add_argument('--continue_inversion', action='store_true', default=False, 
                       help='Continue textual inversion during LoRA training')
    parser.add_argument('--continue_inversion_lr', type=float, default=None, 
                       help='Learning rate for continued textual inversion (if different from --learning_rate_ti)')
    
    # --- 数据处理配置 ---
    parser.add_argument('--use_face_segmentation_condition', action='store_true', default=False, 
                       help='Use face segmentation as conditioning')
    parser.add_argument('--cached_latents', action='store_true', default=True, 
                       help='Use cached latents for faster training')
    parser.add_argument('--use_mask_captioned_data', action='store_true', default=False, 
                       help='Use mask captioned data')
    parser.add_argument('--mask_temperature', type=float, default=1.0, 
                       help='Temperature for mask processing')
    
    # --- 优化器配置 ---
    parser.add_argument('--weight_decay_ti', type=float, default=0.00, 
                       help='Weight decay for textual inversion optimizer')
    parser.add_argument('--weight_decay_lora', type=float, default=0.001, 
                       help='Weight decay for LoRA optimizer')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False, 
                       help='Use 8-bit Adam optimizer to save memory')
    
    # --- 设备和精度配置 ---
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device to use for training (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--mixed_precision', type=str, default='no', 
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training mode')
    
    # --- 损失权重配置 ---
    parser.add_argument('--feature_align_weight', type=float, default=0.01, 
                       help='Weight for feature alignment loss')
    parser.add_argument('--noise_pred_weight', type=float, default=1.0, 
                       help='Weight for noise prediction loss')
    
    # --- 日志和监控配置 ---
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
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs', 
                       help='Directory for TensorBoard logs')
    
    # --- 特征对齐配置 ---
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
    
    # --- 其他高级配置 ---
    parser.add_argument('--t_multiplier_loss', type=float, default=1.0, 
                       help='Multiplier for timestep sampling in loss calculation')
    parser.add_argument('--save_image_every_n_steps_loss', type=int, default=200, 
                       help='Save sample images every N steps during loss calculation')
    parser.add_argument('--unet_return_dict', action='store_true', default=True, 
                       help='Whether UNet feature extractor returns dictionary')
    
    args = parser.parse_args()
    
    # --- 参数验证和处理 ---
    
    # 验证必需的输入参数
    if not args.lora_models_dir and not (args.lora_path1 and args.lora_path2):
        parser.error("必须提供 --lora_models_dir 或者同时提供 --lora_path1 和 --lora_path2")
    
    if args.lora_models_dir and (args.lora_path1 or args.lora_path2):
        print("警告: 同时提供了 --lora_models_dir 和单独的 lora_path 参数。将优先使用 --lora_models_dir")
        args.lora_path1 = None
        args.lora_path2 = None
    
    # 处理 teacher_weights 参数
    teacher_weights = None
    if args.teacher_weights:
        try:
            teacher_weights = [float(w.strip()) for w in args.teacher_weights.split(',')]
            print(f"使用教师权重: {teacher_weights}")
        except ValueError:
            parser.error("teacher_weights 必须是逗号分隔的浮点数，例如: '0.3,0.4,0.3'")
    
    # 处理 LoRA 目标模块参数
    if isinstance(args.lora_unet_target_modules, str):
        lora_unet_target_modules = set(args.lora_unet_target_modules.split(','))
    else:
        lora_unet_target_modules = args.lora_unet_target_modules
    
    if isinstance(args.lora_clip_target_modules, str):
        lora_clip_target_modules = set(args.lora_clip_target_modules.split(','))
    else:
        lora_clip_target_modules = args.lora_clip_target_modules
    
    # 处理特征对齐层参数
    if isinstance(args.unet_feature_alignment_layers, str):
        unet_feature_alignment_layers = args.unet_feature_alignment_layers.split(',')
    else:
        unet_feature_alignment_layers = args.unet_feature_alignment_layers
    
    if isinstance(args.text_encoder_alignment_layers, str):
        text_encoder_alignment_layers = args.text_encoder_alignment_layers.split(',')
    else:
        text_encoder_alignment_layers = args.text_encoder_alignment_layers
    
    # 验证teacher选择策略
    if args.teacher_selection_strategy == 'weighted_random' and teacher_weights is None:
        print("警告: 使用 weighted_random 策略但未提供 teacher_weights，将使用均匀权重")
    
    # 创建完整的参数字典
    train_args = {
        # 主要参数
        'lora_models_dir': args.lora_models_dir,
        'lora_path1': args.lora_path1,
        'lora_path2': args.lora_path2,
        'instance_data_dir': args.instance_data_dir,
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'output_dir': args.output_dir,
        
        # 多teacher配置
        'teacher_selection_strategy': args.teacher_selection_strategy,
        'teacher_weights': teacher_weights,
        'auto_generate_placeholder_tokens': args.auto_generate_placeholder_tokens,
        
        # 模型和训练配置
        'train_text_encoder': args.train_text_encoder,
        'pretrained_vae_name_or_path': args.pretrained_vae_name_or_path,
        'revision': args.revision,
        'perform_inversion': args.perform_inversion,
        'use_template': args.use_template,
        'train_inpainting': args.train_inpainting,
        
        # Token配置
        'placeholder_tokens': args.placeholder_tokens,
        'placeholder_tokens1': args.placeholder_tokens1,
        'placeholder_tokens2': args.placeholder_tokens2,
        'placeholder_token_at_data': args.placeholder_token_at_data,
        'initializer_tokens': args.initializer_tokens,
        
        # 基础训练参数
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
        
        # LoRA配置
        'lora_rank': args.lora_rank,
        'lora_unet_target_modules': lora_unet_target_modules,
        'lora_clip_target_modules': lora_clip_target_modules,
        'lora_dropout_p': args.lora_dropout_p,
        'lora_scale': args.lora_scale,
        'use_extended_lora': args.use_extended_lora,
        'clip_ti_decay': args.clip_ti_decay,
        
        # 学习率配置
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
        
        # 数据处理配置
        'use_face_segmentation_condition': args.use_face_segmentation_condition,
        'cached_latents': args.cached_latents,
        'use_mask_captioned_data': args.use_mask_captioned_data,
        'mask_temperature': args.mask_temperature,
        
        # 优化器配置
        'weight_decay_ti': args.weight_decay_ti,
        'weight_decay_lora': args.weight_decay_lora,
        'use_8bit_adam': args.use_8bit_adam,
        
        # 设备和系统配置
        'device': args.device,
        'extra_args': None,  # 可以在这里添加额外参数
        'enable_xformers_memory_efficient_attention': args.enable_xformers_memory_efficient_attention,
        
        # 损失权重
        'feature_align_weight': args.feature_align_weight,
        'noise_pred_weight': args.noise_pred_weight,
        
        # 日志配置
        'log_wandb': args.log_wandb,
        'wandb_log_prompt_cnt': args.wandb_log_prompt_cnt,
        'wandb_project_name': args.wandb_project_name,
        'wandb_entity': args.wandb_entity,
        'proxy_token': args.proxy_token,
        'out_name': args.out_name,
    }
    
    # 显示配置摘要
    print("\n" + "="*80)
    print("多Teacher LoRA蒸馏训练配置摘要")
    print("="*80)
    
    if args.lora_models_dir:
        print(f"模式: 多Teacher文件夹模式")
        print(f"LoRA模型目录: {args.lora_models_dir}")
    else:
        print(f"模式: 双Teacher兼容模式")
        print(f"LoRA路径1: {args.lora_path1}")
        print(f"LoRA路径2: {args.lora_path2}")
    
    print(f"基础模型: {args.pretrained_model_name_or_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"Teacher选择策略: {args.teacher_selection_strategy}")
    if teacher_weights:
        print(f"Teacher权重: {teacher_weights}")
    print(f"自动生成tokens: {args.auto_generate_placeholder_tokens}")
    print(f"执行文本逆向: {args.perform_inversion}")
    print(f"训练批次大小: {args.train_batch_size}")
    print(f"TI训练步数: {args.max_train_steps_ti}")
    print(f"LoRA训练步数: {args.max_train_steps_tuning}")
    print(f"设备: {args.device}")
    print(f"特征对齐权重: {args.feature_align_weight}")
    print("="*80)
    
    # 运行训练
    try:
        print("开始多Teacher蒸馏训练...")
        train(**train_args)
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
