import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
import glob

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


@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    if cached_latents:

        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print("PTI : Using cached latent.")

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader

def text2img_dataloader_noise(
    train_dataset, # 应该是 PivotalTuningDatasetMultiTokenizer 的实例
    train_batch_size,
    student_tokenizer, # Student's tokenizer, 用于 padding student_prompt_ids
    teacher_tokenizer1, # Teacher's tokenizer 1, 用于 padding teacher1_prompt_ids
    teacher_tokenizer2, # Teacher's tokenizer 2, 用于 padding teacher2_prompt_ids
    # vae 和 text_encoder 参数在这里似乎没有直接用于 collate_fn，
    # 但如果它们影响 latent_shape 或其他方面，请保留
    # vae,
    # text_encoder,
    cached_latents: bool = False, # 这个参数在这里似乎不适用，因为总是生成噪声
    latent_shape=(4, 64, 64), # 注意：通常 latent shape 是 (C, H, W)，对于SD1.x, VAE输出是4通道
                               # 原始代码是 (4, 64, 64)，这里我先保持一致
):
    # latent_shape: 通常 Stable Diffusion 的 VAE 输出是 4 通道 (e.g., (4, 64, 64))
    # 原始代码中是 (4, 64, 64)，这看起来是正确的。之前的 (3, 64, 64) 可能是笔误。
    # SDXL 的 latent shape 会更大。

    def collate_fn(examples):
        batch = {}

        # 1. 处理 Student Prompt IDs
        student_ids_list = [example["student_prompt_ids"] for example in examples]
        student_padded = student_tokenizer.pad(
            {"input_ids": student_ids_list},
            padding="max_length", # Pad to the tokenizer's max_length
            max_length=student_tokenizer.model_max_length,
            return_tensors="pt",
        )
        batch["student_input_ids"] = student_padded.input_ids
        if "attention_mask" in student_padded:
            batch["student_attention_mask"] = student_padded.attention_mask

        # 2. 处理 Teacher 1 Prompt IDs
        teacher1_ids_list = [example["teacher1_prompt_ids"] for example in examples]
        teacher1_padded = teacher_tokenizer1.pad(
            {"input_ids": teacher1_ids_list},
            padding="max_length",
            max_length=teacher_tokenizer1.model_max_length,
            return_tensors="pt",
        )
        batch["teacher1_input_ids"] = teacher1_padded.input_ids
        if "attention_mask" in teacher1_padded:
            batch["teacher1_attention_mask"] = teacher1_padded.attention_mask
        
        # 3. 处理 Teacher 2 Prompt IDs
        teacher2_ids_list = [example["teacher2_prompt_ids"] for example in examples]
        teacher2_padded = teacher_tokenizer2.pad(
            {"input_ids": teacher2_ids_list},
            padding="max_length",
            max_length=teacher_tokenizer2.model_max_length,
            return_tensors="pt",
        )
        batch["teacher2_input_ids"] = teacher2_padded.input_ids
        if "attention_mask" in teacher2_padded:
            batch["teacher2_attention_mask"] = teacher2_padded.attention_mask

        # 4. 生成高斯噪声作为 pixel_values (latents)
        # 确保 latent_shape 是正确的，对于SD1.x通常是 (4, H/8, W/8)
        # 例如，如果原始图像是 512x512，那么 latent_shape 是 (4, 64, 64)
        # 0.18215 是 Stable Diffusion VAE 的缩放因子
        pixel_values = [torch.randn(latent_shape) * 0.18215 for _ in examples]
        batch["pixel_values"] = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

        # 5. (可选) 处理 mask (如果存在)
        # 检查第一个样本是否有 "mask" 键，并且其值不是 None
        if "mask" in examples[0] and examples[0]["mask"] is not None:
            # 确保所有样本都有 mask，或者需要更复杂的处理逻辑
            # 这里假设如果第一个有，那么所有都有，或者可以安全地堆叠
            try:
                batch["mask"] = torch.stack([example["mask"] for example in examples])
            except Exception as e:
                print(f"Warning: Could not stack masks. Error: {e}")
                # 你可能需要决定如何处理这种情况，例如跳过 mask 或填充
        
        # 6. (可选) 保留原始文本
        if "raw_text" in examples[0]:
            batch["raw_text"] = [example["raw_text"] for example in examples]

        return batch
    
    # 不使用 cached_latents 流程
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print("Using Gaussian noise latents as instance_images.")
    return train_dataloader

import torch
from torch.utils.data import DataLoader

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
    # teacher_unet, # 不再需要教师UNet
    student_unet,
    student_text_encoder,  # 学生模型的文本编码器
    # teacher_text_encoder,  # 不再需要教师文本编码器
    scheduler,
    vae,                  # VAE 模型用于解码
    global_step,          # 当前全局训练步数
    save_image_every_n_steps=100, # 保持与函数定义一致的默认值
    # 注意: output_dir 将在 train_inversion_tf_style 中动态设置
    output_dir_for_loss_step="/root/lora_train/pic_train", # 临时默认值，会被覆盖
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
):
    weight_dtype = torch.float32
    if mixed_precision:
        # VAE解码通常需要较高精度，即使其他部分是fp16
        vae_decode_input_dtype = torch.float32 # VAE解码前的输入（缩放后）的数据类型
    else:
        vae_decode_input_dtype = torch.float32

    # 真实潜变量 (x0)
    # 假设 batch["pixel_values"] 是 VAE 编码后的潜变量
    # 如果 batch["pixel_values"] 是图像，需要先编码
    if batch["pixel_values"].ndim == 4 and batch["pixel_values"].shape[1] in [1, 3, 4]: # 检查是否像图像
        # print("Warning: pixel_values in batch seem to be images, encoding to latents with VAE.")
        # 此处需要确保 vae 在正确的设备上，并且输入也正确
        # 为了简化，我们假设 batch["pixel_values"] 已经是潜变量
        # 如果不是，需要添加 VAE 编码步骤
        # with torch.no_grad():
        #     latents_gt_x0 = vae.encode(batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)).latent_dist.sample()
        #     latents_gt_x0 = latents_gt_x0 * vae.config.scaling_factor
        # pass # 假设已经是潜变量
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)

    else: # 假设已经是潜变量
        latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)

    bsz = latents_gt_x0.shape[0]

    # 为每张图片采样一个随机的时间步 t
    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier), # 如果使用 t_multiplier，调整最大时间步
        (bsz,),
        device=latents_gt_x0.device,
    )
    timesteps = timesteps.long()

    # 采样我们将添加到潜变量中并尝试预测的噪声 epsilon
    noise = torch.randn_like(latents_gt_x0)

    # 根据时间步向潜变量添加噪声
    # noisy_latents 就是 x_t
    noisy_latents = scheduler.add_noise(latents_gt_x0, noise, timesteps)

    # 准备UNet的输入潜变量
    if mixed_precision:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float16)
    else:
        student_unet_input_latents = noisy_latents.to(dtype=torch.float32)

    # --- 学生模型部分 (文本嵌入) ---
    student_input_ids = batch.get("aux1_input_ids")
    if student_input_ids is None:
        # print("警告: student_text_encoder 尝试使用 aux1_input_ids，但未在批处理中找到。回退到 input_ids。")
        student_input_ids = batch["input_ids"]
    student_input_ids = student_input_ids.to(student_text_encoder.device)

    student_attention_mask = batch.get("aux1_attention_mask")
    if student_attention_mask is not None:
        student_attention_mask = student_attention_mask.to(student_text_encoder.device)

    student_text_encoder_output_dtype_for_autocast = torch.float32
    if hasattr(student_text_encoder, 'dtype') and student_text_encoder.dtype == torch.float16:
         student_text_encoder_output_dtype_for_autocast = torch.float16

    if mixed_precision and student_text_encoder_output_dtype_for_autocast == torch.float16:
        with torch.cuda.amp.autocast(enabled=True): # autocast for text_encoder if it's fp16
            student_encoder_hidden_states = student_text_encoder(
                input_ids=student_input_ids, attention_mask=student_attention_mask
            )[0]
    else:
        _temp_states = student_text_encoder(
            input_ids=student_input_ids, attention_mask=student_attention_mask
        )[0]
        # 确保输出类型与后续使用一致
        # student_encoder_hidden_states = _temp_states.to(dtype=weight_dtype) # or student_unet.dtype
        # 基于 UNet 的期望类型进行转换
        unet_expected_dtype = student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype
        student_encoder_hidden_states = _temp_states.to(dtype=unet_expected_dtype)


    # --- 学生模型UNet预测噪声 ---
    student_unet_internal_dtype = student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype
    student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(dtype=student_unet_internal_dtype)

    if mixed_precision: # UNet is in fp16
        with torch.cuda.amp.autocast(enabled=True):
            student_pred_noise = student_unet(
                student_unet_input_latents.to(student_unet_internal_dtype),
                timesteps,
                student_encoder_hidden_states_for_unet
            ).sample
    else: # UNet is in fp32 or its native precision
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
            raise ValueError(f"Mask 形状异常: {mask.shape}. 期望 [B, 1, H, W] 或 [B, H, W]")

        mask = (mask + 0.01).pow(mask_temperature)
        mask = mask / mask.max()
        pred_dtype = student_pred_noise.dtype
        mask = mask.to(dtype=pred_dtype)
        student_pred_noise = student_pred_noise * mask
        target_noise = target_noise * mask

    loss = F.mse_loss(student_pred_noise.float(), target_noise.float(), reduction="none").mean([1, 2, 3]).mean()

    if global_step % save_image_every_n_steps == 0 and save_image_every_n_steps > 0:
        with torch.no_grad():
            latents_gt_x0_viz = latents_gt_x0[0:1].detach()
            student_pred_noise_viz = student_pred_noise[0:1].detach() # Use the (possibly masked) prediction for viz
            noisy_latents_viz = noisy_latents[0:1].detach()
            timestep_viz = timesteps[0:1].detach()

            # Get pred_x0 from student
            # scheduler.step expects model_output (predicted noise), timestep, and sample (noisy_latents xt)
            # Ensure DDPMScheduler gets a single integer for timestep if that's what it expects
            # For other schedulers, timestep_viz might be fine.
            # Common schedulers like DDPMScheduler, DDIMScheduler, PNDMScheduler expect a single timestep value here.
            current_timestep_int = timestep_viz.item() if timestep_viz.numel() == 1 else timestep_viz[0].item()

            scheduler_output_student = scheduler.step(
                model_output=student_pred_noise_viz.to(dtype=noisy_latents_viz.dtype),
                timestep=torch.tensor([current_timestep_int], device=noisy_latents_viz.device) if not isinstance(current_timestep_int, int) else current_timestep_int, # Ensure timestep is scalar or tensor(scalar)
                sample=noisy_latents_viz
            )
            pred_x0_latent_student = scheduler_output_student.pred_original_sample

            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            pred_x0_latent_student_for_vae = pred_x0_latent_student.to(dtype=vae_decode_input_dtype) / scaling_factor
            latent_x0_gt_viz_for_vae = latents_gt_x0_viz.to(dtype=vae_decode_input_dtype) / scaling_factor

            vae_internal_dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float32
            with torch.cuda.amp.autocast(enabled=False):
                pred_image_student = vae.decode(pred_x0_latent_student_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
                gt_image = vae.decode(latent_x0_gt_viz_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample

            pred_image_student = (pred_image_student / 2 + 0.5).clamp(0, 1)
            gt_image = (gt_image / 2 + 0.5).clamp(0, 1)

            if not os.path.exists(output_dir_for_loss_step): # Use the passed output_dir
                os.makedirs(output_dir_for_loss_step, exist_ok=True)

            t_val = current_timestep_int
            save_image(pred_image_student, os.path.join(output_dir_for_loss_step, f"step_{global_step}_student_pred_x0_t{t_val}.png"))
            save_image(gt_image, os.path.join(output_dir_for_loss_step, f"step_{global_step}_gt_x0_t{t_val}.png"))

            noisy_input_latent_for_vae = noisy_latents_viz.to(dtype=vae_decode_input_dtype) / scaling_factor
            with torch.cuda.amp.autocast(enabled=False):
                noisy_input_image = vae.decode(noisy_input_latent_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
            noisy_input_image = (noisy_input_image / 2 + 0.5).clamp(0, 1)
            save_image(noisy_input_image, os.path.join(output_dir_for_loss_step, f"step_{global_step}_noisy_input_xt_t{t_val}.png"))
    return loss

def train_inversion_with_dual_feature_alignment(
    # --- "Teacher" 模型参数 ---
    teacher1_unet,
    teacher2_unet,
    teacher1_text_encoder,
    teacher2_text_encoder,
    # --- "Student" 模型参数 ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader 参数 ---
    dataloader1,
    dataloader2,
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
    text_encoder_alignment_layers=[ # 用于Text Encoder特征对齐的层
        'text_model.encoder.layers.0',
        'text_model.encoder.layers.6', 
        'text_model.encoder.layers.11'
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
):
    """
    train_inversion_with_dual_feature_alignment 函数：
    增强版的文本逆向训练，同时使用UNet和Text Encoder的特征对齐。
    
    新增功能：
    1. UNet层间特征对齐 (原有功能)
    2. Text Encoder层间特征对齐 (新增功能)
    3. 可以独立控制两种特征对齐的权重
    """
    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_inversion_dual_feat_align")
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard 日志 (双重特征对齐的文本逆向) 将保存到: {tb_log_path}")

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

    # UNet特征提取器
    teacher1_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )
    teacher2_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )

    # --- Text Encoder特征对齐组件设置 ---
    text_encoder_layer_weights = {
        layer: 1.0 for layer in text_encoder_alignment_layers
    }
    # 可以为特定层设置不同权重
    if 'text_model.encoder.layers.11' in text_encoder_alignment_layers:
        text_encoder_layer_weights['text_model.encoder.layers.11'] = 1.5
    if 'text_model.encoder.layers.6' in text_encoder_alignment_layers:
        text_encoder_layer_weights['text_model.encoder.layers.6'] = 1.2

    text_encoder_feature_alignment_loss_fn = TextEncoderFeatureAlignmentLoss(
        alignment_layers=text_encoder_alignment_layers,
        loss_weights=text_encoder_layer_weights,
        loss_type=text_encoder_loss_type,
        pooling_strategy=text_encoder_pooling_strategy,
        temperature=1.0
    )

    # Text Encoder特征提取器
    text_encoder_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("训练步数 (文本逆向 + 双重特征对齐)")
    global_step = 0

    # 备份原始的、非占位符的词元嵌入
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        if wandb.run is None:
            wandb.init(
                project=f"textual_inversion_project_{out_name}", 
                name=f"{out_name}_inversion_dual_feat_align_run", 
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

    iter_dataloader1 = iter(dataloader1)
    iter_dataloader2 = iter(dataloader2)

    # 从模型确定设备
    unet_device = next(student_unet.parameters()).device
    text_encoder_device = next(student_text_encoder.parameters()).device
    vae_device = vae.device

    # 主训练循环
    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # --- 选择当前教师模型和数据加载器 ---
        if global_step % 2 == 0:
            current_dataloader_iter = iter_dataloader1
            current_dataloader_obj = dataloader1
            current_teacher_unet = teacher1_unet
            current_teacher_text_encoder = teacher1_text_encoder
            current_teacher_unet_extractor = teacher1_unet_extractor
            teacher_name_log = "T1"
        else:
            current_dataloader_iter = iter_dataloader2
            current_dataloader_obj = dataloader2
            current_teacher_unet = teacher2_unet
            current_teacher_text_encoder = teacher2_text_encoder
            current_teacher_unet_extractor = teacher2_unet_extractor
            teacher_name_log = "T2"
        
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            if global_step % 2 == 0:
                iter_dataloader1 = iter(current_dataloader_obj)
                batch = next(iter_dataloader1)
            else:
                iter_dataloader2 = iter(current_dataloader_obj)
                batch = next(iter_dataloader2)

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
                t_mutliplier=0.8,
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

            # 6. 总损失
            current_step_total_loss = (
                noise_pred_weight * ti_loss_current_step +
                unet_feature_align_weight * unet_feature_align_loss_val +
                text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
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
                        writer.add_scalar('梯度/文本嵌入范数_双重FA', grad_norm.item(), global_step)
            else:
                print(f"步骤 {global_step}: 警告：在双重FA更新期间未找到文本嵌入的梯度。")

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
                    writer.add_scalar('嵌入/当前范数_双重FA', current_norm_val, global_step)

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
            writer.add_scalar('损失_双重FA/总损失_单步', current_step_total_loss.item(), global_step)
            writer.add_scalar('损失_双重FA/TI损失_单步', ti_loss_current_step.item(), global_step)
            writer.add_scalar('损失_双重FA/UNet特征对齐损失_单步', unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val), global_step)
            writer.add_scalar('损失_双重FA/TextEncoder特征对齐损失_单步', text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val), global_step)
            writer.add_scalar('学习率_双重FA/文本嵌入', current_lr_val, global_step)
            writer.add_scalar('教师模型_双重FA', 1 if (global_step-1) % 2 == 0 else 2, global_step)
            writer.add_scalar('Weights/TI_Loss_Weight', noise_pred_weight, global_step)
            writer.add_scalar('Weights/UNet_Feature_Weight', unet_feature_align_weight, global_step)
            writer.add_scalar('Weights/TextEncoder_Feature_Weight', text_encoder_feature_align_weight, global_step)
            
            # 记录各层的UNet特征损失
            if isinstance(unet_feature_loss_dict_current, dict):
                for layer_name, layer_loss in unet_feature_loss_dict_current.items():
                    writer.add_scalar(f'UNet特征损失_双重FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
            
            # 记录各层的Text Encoder特征损失
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                for layer_name, layer_loss in text_encoder_feature_loss_dict_current.items():
                    writer.add_scalar(f'TextEncoder特征损失_双重FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
        
        # wandb 日志记录
        wandb_step_logs = {}
        if log_wandb and global_step % 10 == 0:
            wandb_step_logs.update({
                'step': global_step,
                'total_loss_step_dual_fa': current_step_total_loss.item(),
                'ti_loss_step_dual_fa': ti_loss_current_step.item(),
                'unet_feature_align_loss_step_dual_fa': unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val),
                'text_encoder_feature_align_loss_step_dual_fa': text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val),
                'learning_rate_dual_fa': current_lr_val,
                'teacher_model_used_dual_fa': 1 if (global_step-1) % 2 == 0 else 2,
            })
            
            # 添加各层特征损失
            if isinstance(unet_feature_loss_dict_current, dict):
                wandb_step_logs.update({
                    f"unet_feat_align_layer_{k.replace('.', '_')}_dual_fa": v_loss.item() if torch.is_tensor(v_loss) else float(v_loss)
                    for k, v_loss in unet_feature_loss_dict_current.items()
                })
            
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                wandb_step_logs.update({
                    f"text_encoder_feat_align_layer_{k.replace('.', '_')}_dual_fa": v_loss.item() if torch.is_tensor(v_loss) else float(v_loss)
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
                save_path=os.path.join(current_save_dir, f"step_inv_dual_fa_{global_step}.safetensors"),
                save_lora=False,
            )
            print(f"\n ✓ 双重FA检查点已保存: step_inv_dual_fa_{global_step}.safetensors")

            # 计算区间平均损失
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_ti_loss_interval = accumulated_ti_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_unet_feature_loss_interval = accumulated_unet_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_text_encoder_feature_loss_interval = accumulated_text_encoder_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0

            if writer:
                writer.add_scalar('损失_双重FA/总损失_平均区间', avg_total_loss_interval, global_step)
                writer.add_scalar('损失_双重FA/TI损失_平均区间', avg_ti_loss_interval, global_step)
                writer.add_scalar('损失_双重FA/UNet特征对齐损失_平均区间', avg_unet_feature_loss_interval, global_step)
                writer.add_scalar('损失_双重FA/TextEncoder特征对齐损失_平均区间', avg_text_encoder_feature_loss_interval, global_step)
            
            wandb_interval_logs = {
                "loss_total_avg_interval_dual_fa": avg_total_loss_interval,
                "loss_ti_avg_interval_dual_fa": avg_ti_loss_interval,
                "loss_unet_feature_align_avg_interval_dual_fa": avg_unet_feature_loss_interval,
                "loss_text_encoder_feature_align_avg_interval_dual_fa": avg_text_encoder_feature_loss_interval,
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
    print(f"TRAINING COMPLETED!")
    print(f"="*80)
    print(f"Total steps completed: {global_step}")
    print(f"Final learning rate: {current_lr_val:.2e}")
    
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_step_inv_{global_step}.safetensors")
    save_all(
        unet=student_unet, text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(final_save_dir, f"final_step_inv_{global_step}.safetensors"),
        save_lora=False,
    )
    
    print(f"✓ Final model saved to: {final_model_path}")
    print(f"✓ TensorBoard logs saved to: {tb_log_path}")
    print(f"Training completed with feature alignment!")
    print(f"="*80)

def scan_lora_folder(lora_folder_path: str) -> Dict[str, str]:
    """
    扫描文件夹中的LoRA文件，返回文件名到完整路径的映射
    文件名格式应为: filename.safetensors，对应的placeholder_token为 <filename>
    """
    lora_files = {}
    folder_path = Path(lora_folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"LoRA folder path does not exist: {lora_folder_path}")
    
    # 查找所有.safetensors文件
    for file_path in folder_path.glob("*.safetensors"):
        file_stem = file_path.stem  # 获取不带扩展名的文件名
        lora_files[file_stem] = str(file_path)
    
    print(f"Found {len(lora_files)} LoRA files:")
    for name, path in lora_files.items():
        print(f"  - {name}: {path}")
    
    return lora_files

def create_teacher_pipelines(lora_files: Dict[str, str], pretrained_model_name_or_path: str, device: str):
    """
    为每个LoRA文件创建teacher pipeline
    """
    teacher_pipes = {}
    
    for lora_name, lora_path in lora_files.items():
        print(f"Creating teacher pipeline for: {lora_name}")
        
        # 创建pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.float16
        ).to(device)
        
        # 设置scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # 应用LoRA
        patch_pipe(
            pipe,
            lora_path,
            patch_text=True,
            patch_ti=True,
            patch_unet=True,
        )
        
        # 调整LoRA缩放
        tune_lora_scale(pipe.unet, 1.0)
        tune_lora_scale(pipe.text_encoder, 1.0)
        
        teacher_pipes[lora_name] = pipe
    
    return teacher_pipes

def create_multi_datasets(teacher_pipes: Dict, token_maps: Dict, use_template, student_tokenizer, device: str):
    """
    为每个teacher创建对应的数据集
    """
    datasets = {}
    dataloaders = {}
    
    for lora_name, teacher_pipe in teacher_pipes.items():
        print(f"Creating dataset for: {lora_name}")
        
        dataset = PivotalTuningDatasetCapationLoraGenerated(
            sd_pipeline=teacher_pipe,
            device=device,
            main_tokenizer=teacher_pipe.tokenizer,
            use_template=use_template,
            token_map=token_maps[lora_name],
            dataset_size=10,  # 可以根据需要调整
            transform_size=512,
            h_flip=True,
            aux_tokenizer1=student_tokenizer,
            save_generated_images_path="/root/lora_train/pic",
            save_image_prefix=f"lora_{lora_name}"
        )
        
        dataloader = text2img_dataloader_combined_with_latent_caching(
            train_dataset=dataset,
            train_batch_size=1,  # 这里使用固定值，可以作为参数传入
            main_tokenizer=teacher_pipe.tokenizer,
            aux_tokenizer1=student_tokenizer,
            vae=teacher_pipe.vae,
            cached_latents=True
        )
        
        datasets[lora_name] = dataset
        dataloaders[lora_name] = dataloader
    
    return datasets, dataloaders

def perform_multi_teacher_tuning(
    # --- "Teacher" 模型参数 ---
    teacher_pipes: Dict[str, Any], # 接收整个 pipelines 字典
    # --- "Student" 模型参数 ---
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL, # VAE 用于学生模型输入编码 (如果 latents 未缓存)
    student_text_encoder: CLIPTextModel,
    # --- Dataloader 参数 ---
    dataloaders: Dict[str, torch.utils.data.DataLoader], # 接收 dataloaders 字典
    # --- 核心训练参数 ---
    num_steps: int,
    cached_latents: bool,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer, # LoRA 参数的优化器
    save_steps: int,
    placeholder_tokens: List[str], # 学生模型的占位符 (如果 LoRA 针对特定概念)
    placeholder_token_ids: List[int], # 对应的 ID
    save_path: str, # 输出目录的根路径
    # --- 学习率调度器 ---
    lr_scheduler_lora: torch.optim.lr_scheduler._LRScheduler,
    # --- LoRA 相关参数 (用于 save_all) ---
    lora_unet_target_modules: set,
    lora_clip_target_modules: set,
    # --- 损失和日志相关 ---
    mask_temperature: float,
    tokenizer: CLIPTokenizer, # 学生模型的分词器
    out_name: str, # 输出文件和日志的前缀
    log_wandb: bool,
    wandb_log_prompt_cnt: int, # 用于 wandb 日志中样本生成的数量
    class_token: str, # 用于提示构建或评估
    train_inpainting: bool, # 是否为修复任务训练 (可能影响数据处理或损失)
    # --- 损失权重 ---
    feature_align_weight: float,
    noise_pred_weight: float,
    # --- 其他可配置参数 (可以保持默认或从 extra_args 传递) ---
    feature_alignment_unet_layers: List[str] = [ # UNet 特征对齐层
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    unet_return_dict: bool = True, # UNet 特征提取器是否返回字典
    mixed_precision: str = "no", # 例如 "no", "fp16", "bf16"
    tensorboard_log_dir: str = "runs", # TensorBoard 日志根目录
    device: str = "cuda" # 训练设备
):
    """
    多教师 LoRA 微调函数，包含特征对齐，参数与 train_multi_lora_distillation 中的调用匹配。
    """
    # --- 从 teacher_pipes 和 dataloaders 字典中提取列表 ---
    ordered_teacher_names = sorted(list(teacher_pipes.keys()))
    if not ordered_teacher_names:
        raise ValueError("No teacher pipelines provided in teacher_pipes.")

    teacher_unets_list: List[UNet2DConditionModel] = []
    teacher_text_encoders_list: List[CLIPTextModel] = []
    dataloaders_list: List[torch.utils.data.DataLoader] = []

    for name in ordered_teacher_names:
        pipe = teacher_pipes[name]
        if not hasattr(pipe, 'unet') or not isinstance(pipe.unet, UNet2DConditionModel):
            raise ValueError(f"Teacher pipeline '{name}' does not have a valid UNet model.")
        if not hasattr(pipe, 'text_encoder') or not isinstance(pipe.text_encoder, CLIPTextModel):
            raise ValueError(f"Teacher pipeline '{name}' does not have a valid text encoder.")
        teacher_unets_list.append(pipe.unet)
        teacher_text_encoders_list.append(pipe.text_encoder)

        if name not in dataloaders:
            raise ValueError(f"No dataloader found for teacher '{name}' in the dataloaders dictionary.")
        dataloaders_list.append(dataloaders[name])

    num_teachers = len(teacher_unets_list)

    # --- 确定设备 ---
    if device is None:
        if student_unet.device != torch.device("meta"):
            device = student_unet.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Warning: Device not specified. Defaulting to {device}.")
    current_device = torch.device(device)

    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    current_tensorboard_log_dir = os.path.join(tensorboard_log_dir, out_name + "_tuning")
    os.makedirs(current_tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=current_tensorboard_log_dir)
    print(f"TensorBoard logging (Multi-Teacher LoRA Tuning) to: {current_tensorboard_log_dir}")

    # --- UNet特征对齐组件设置 ---
    alignment_layers_for_loss = feature_alignment_unet_layers
    layer_weights_for_loss = { # 示例权重，可以配置
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }
    layer_weights_for_loss = {k: v for k, v in layer_weights_for_loss.items() if k in alignment_layers_for_loss}

    feature_alignment_loss_fn = FeatureAlignmentLoss( # 假设 FeatureAlignmentLoss 已定义
        alignment_layers=alignment_layers_for_loss,
        loss_weights=layer_weights_for_loss,
        loss_type="mse"
    )
    teacher_extractors = [ # 每个教师一个 UNet 特征提取器
        UNetFeatureExtractor( # 假设 UNetFeatureExtractor 已定义
            target_layers=alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        ) for _ in range(num_teachers)
    ]
    student_extractor = UNetFeatureExtractor( # 学生模型的 UNet 特征提取器
        target_layers=alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps), desc="Multi-Teacher LoRA Tuning")
    global_step = 0

    iter_dataloaders = [iter(dl) for dl in dataloaders_list]

    accumulated_total_loss = 0.0
    accumulated_noise_loss = 0.0
    accumulated_feature_loss = 0.0
    step_count_in_interval = 0

    # 将学生模型移到目标设备
    student_unet.to(current_device)
    student_text_encoder.to(current_device)
    vae.to(current_device) # VAE 也需要到正确设备

    student_unet.train()
    # 根据 train_text_encoder 或 continue_inversion (如果在主函数中决定) 设置 student_text_encoder 模式
    # 这里假设如果 student_text_encoder 被传递，它就应该被训练（LoRA层或嵌入）
    student_text_encoder.train()


    for step_idx in range(num_steps):
        current_teacher_idx = global_step % num_teachers

        current_dataloader_iter = iter_dataloaders[current_teacher_idx]
        current_dataloader_obj = dataloaders_list[current_teacher_idx]
        # 将当前教师模型移到目标设备
        current_teacher_unet = teacher_unets_list[current_teacher_idx].to(current_device)
        current_teacher_text_encoder = teacher_text_encoders_list[current_teacher_idx].to(current_device)
        current_teacher_extractor = teacher_extractors[current_teacher_idx]
        teacher_name_log = ordered_teacher_names[current_teacher_idx]

        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            print(f"Info: Dataloader for Teacher Tuning '{teacher_name_log}' exhausted. Re-initializing.")
            iter_dataloaders[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(iter_dataloaders[current_teacher_idx])

        # 将批次数据移到目标设备
        batch = {k: v.to(current_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad() # 清除 LoRA 参数的梯度

        main_student_module = student_unet.module if hasattr(student_unet, 'module') else student_unet
        expected_latents_dtype = main_student_module.dtype

        if cached_latents:
            latents = batch["pixel_values"].to(dtype=expected_latents_dtype)
        else:
            pixel_values_for_vae = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype) # VAE 可能有自己的dtype
            with torch.no_grad():
                latents = vae.encode(pixel_values_for_vae).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(device=current_device, dtype=expected_latents_dtype)

        # 学生模型的输入 ID (通常与教师模型相同，除非有特殊处理)
        input_ids = batch["input_ids"]
        # student_input_ids = batch.get('aux1_input_ids', input_ids) # 如果有学生特定输入

        # 教师文本编码（无梯度）
        teacher_unet_internal_dtype = next(current_teacher_unet.parameters()).dtype
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
                teacher_raw_hidden_states = current_teacher_text_encoder(
                    input_ids, output_hidden_states=True
                ).hidden_states[-1]
                teacher_encoder_hidden_states = teacher_raw_hidden_states.to(dtype=teacher_unet_internal_dtype)

        # 学生文本编码（有梯度，如果文本编码器的LoRA层或嵌入被训练）
        student_unet_internal_dtype = next(student_unet.parameters()).dtype
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
            # 假设学生使用相同的input_ids，除非有特殊逻辑
            student_raw_hidden_states = student_text_encoder(
                input_ids, output_hidden_states=True
            ).hidden_states[-1]
            student_encoder_hidden_states = student_raw_hidden_states.to(dtype=student_unet_internal_dtype)


        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

        # 教师 UNet 特征提取（无梯度）
        with torch.no_grad():
            _, teacher_features = current_teacher_extractor.extract_features(
                unet_model=current_teacher_unet,
                sample=noisy_latents.to(dtype=teacher_unet_internal_dtype),
                timestep=timesteps,
                encoder_hidden_states=teacher_encoder_hidden_states,
                return_dict=unet_return_dict, use_grad=False
            )
        teacher_features = {k: v.to(current_device) for k, v in teacher_features.items()}


        # 学生 UNet 前向传播及特征提取（有梯度，流向LoRA层）
        # 噪声预测损失 (Knowledge Distillation part)
        # loss_step_gaussian_noise 内部会进行学生 UNet 的前向传播
        noise_pred_loss_val = loss_step_gaussian_noise( # 假设 loss_step_gaussian_noise 已定义
            batch=batch, # 包含学生模型需要的信息
            student_unet=student_unet,
            student_text_encoder=student_text_encoder,
            vae=vae, # 用于解码或损失计算（如果需要像素空间）
            global_step=global_step,
            scheduler=scheduler,
            t_mutliplier=0.8, # 示例，可配置
            mixed_precision=use_mixed_precision,
            mask_temperature=mask_temperature,
            # 如果 loss_step_gaussian_noise 需要 device 参数，请传递 current_device
        )

        # 学生 UNet 特征提取（用于对齐损失）
        _, student_features = student_extractor.extract_features(
            unet_model=student_unet,
            sample=noisy_latents.to(dtype=student_unet_internal_dtype),
            timestep=timesteps,
            encoder_hidden_states=student_encoder_hidden_states, # 带梯度 (如果文本LoRA在训练)
            return_dict=unet_return_dict, use_grad=True # 梯度流向 LoRA 层
        )
        student_features = {k: v.to(current_device) for k, v in student_features.items()}


        # 特征对齐损失
        feature_align_loss_val, feature_loss_dict = feature_alignment_loss_fn(
            teacher_features, student_features
        )

        # 确保所有损失都在同一设备上
        noise_pred_loss_val = noise_pred_loss_val.to(current_device)
        feature_align_loss_val = feature_align_loss_val.to(current_device)

        # 动态权重调整 (可选)
        _tuning_weight_adjuster = create_tuning_weight_adjuster( # 假设已定义
            noise_pred_weight, feature_align_weight
        )
        current_losses_for_adj = {
            'noise_pred': noise_pred_loss_val.item(),
            'feature_align': feature_align_loss_val.item(),
        }
        updated_weights = _tuning_weight_adjuster.update_weights(current_losses_for_adj)
        current_noise_pred_weight = updated_weights['noise_pred']
        current_feature_align_weight = updated_weights['feature_align']

        # 总损失
        total_loss = (
            current_noise_pred_weight * noise_pred_loss_val +
            current_feature_align_weight * feature_align_loss_val
        )

        # 反向传播。梯度将通过 optimizer 更新在 params_to_optimize 中定义的参数 (LoRA层等)
        total_loss.backward()
        # 可选：梯度裁剪
        # params_to_clip = []
        # for group in optimizer.param_groups:
        #    params_to_clip.extend(group['params'])
        # torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        optimizer.step()
        lr_scheduler_lora.step()


        current_total_loss_item = total_loss.item()
        current_noise_loss_item = noise_pred_loss_val.item()
        current_feature_loss_item = feature_align_loss_val.item()

        accumulated_total_loss += current_total_loss_item
        accumulated_noise_loss += current_noise_loss_item
        accumulated_feature_loss += current_feature_loss_item
        step_count_in_interval += 1

        current_lr = lr_scheduler_lora.get_last_lr()[0]
        progress_bar.set_postfix({
            'Total': f'{current_total_loss_item:.4f}',
            'Noise': f'{current_noise_loss_item:.4f}',
            'Feat': f'{current_feature_loss_item:.4f}',
            'LR': f'{current_lr:.2e}',
            'Teacher': teacher_name_log[:10]
        })
        progress_bar.update(1)

        # TensorBoard 日志
        if writer:
            writer.add_scalar('Loss_Tune/Total_Step', current_total_loss_item, global_step)
            writer.add_scalar('Loss_Tune/Noise_Pred_Step', current_noise_loss_item, global_step)
            writer.add_scalar('Loss_Tune/Feature_Align_Step', current_feature_loss_item, global_step)
            writer.add_scalar('LearningRate_Tune', current_lr, global_step)
            writer.add_scalar('Teacher_Model_Idx_Tune', current_teacher_idx + 1, global_step)
            # ... 其他权重日志 ...
            if isinstance(feature_loss_dict, dict):
                for layer_name, layer_loss in feature_loss_dict.items():
                    writer.add_scalar(f'FeatureLoss_Tune_Layer/{layer_name.replace(".", "_")}', float(layer_loss), global_step)


        # wandb 日志 (如果启用)
        if log_wandb and global_step % 10 == 0: # 每10步记录一次示例
            # wandb.log({ ... })
            # 可以在这里加入使用当前学生模型生成图像并记录到wandb的逻辑
            # 例如，使用 placeholder_tokens, class_token, tokenizer, wandb_log_prompt_cnt
            pass


        global_step += 1

        if global_step > 0 and global_step % save_steps == 0:
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            # ... 计算其他平均损失 ...
            print(f"\nCHECKPOINT - Tuning Step {global_step}/{num_steps}")
            print(f"  Avg Total Loss: {avg_total_loss_interval:.6f}")
            # ... 打印其他平均损失 ...

            current_save_dir = os.path.join(save_path, out_name)
            os.makedirs(current_save_dir, exist_ok=True)
            checkpoint_filename = f"tuning_step_{global_step}.safetensors"
            save_all( # 假设 save_all 已定义
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=os.path.join(current_save_dir, checkpoint_filename),
                save_lora=True, # LoRA tuning 保存的是LoRA层
                target_replace_module_unet=lora_unet_target_modules,
                target_replace_module_text=lora_clip_target_modules
            )
            print(f"✓ Tuning Checkpoint saved to: {os.path.join(current_save_dir, checkpoint_filename)}\n")

            accumulated_total_loss = 0.0
            accumulated_noise_loss = 0.0
            accumulated_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    if writer:
        writer.close()

    # --- 保存最终模型 ---
    final_save_dir = os.path.join(save_path, out_name)
    os.makedirs(final_save_dir, exist_ok=True)
    final_model_filename = f"tuning_final_step_{global_step}.safetensors"
    save_all(
        unet=student_unet,
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(final_save_dir, final_model_filename),
        save_lora=True,
        target_replace_module_unet=lora_unet_target_modules,
        target_replace_module_text=lora_clip_target_modules
    )
    print(f"\n✓ Final Tuning model saved to: {os.path.join(final_save_dir, final_model_filename)}")
    print(f"✓ TensorBoard logs for Tuning saved to: {current_tensorboard_log_dir}")

def train_multi_teacher_inversion(
    # --- "Teacher" 模型参数 ---
    teacher_pipes: Dict[str, Any], # 接收整个 pipelines 字典
    # --- "Student" 模型参数 ---
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    student_text_encoder: CLIPTextModel,
    # --- Dataloader 参数 ---
    dataloaders: Dict[str, torch.utils.data.DataLoader], # 接收 dataloaders 字典
    # --- 核心训练参数 ---
    num_steps: int,
    scheduler: DDPMScheduler,
    index_no_updates: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    save_steps: int,
    placeholder_token_ids: List[int],
    placeholder_tokens: List[str],
    save_path: str,
    # --- 学习率调度器 ---
    lr_scheduler_main: torch.optim.lr_scheduler._LRScheduler,
    # --- LoRA 相关参数 (TI 中通常不直接使用，但 save_all 可能需要) ---
    lora_unet_target_modules: set,
    lora_clip_target_modules: set,
    # --- 输出和日志相关 ---
    out_name: str,
    tokenizer: CLIPTokenizer,
    test_image_path: Optional[str], # 对应 instance_data_dir
    cached_latents: bool,
    # --- 梯度累积 ---
    accum_iter: int,
    # --- 日志 ---
    log_wandb: bool,
    wandb_log_prompt_cnt: int, # 注意：此参数在核心训练循环中未直接使用，可能用于评估或日志回调
    # --- 其他TI相关 ---
    class_token: str, # 注意：此参数在核心训练循环中未直接使用，可能用于提示构建或评估
    train_inpainting: bool, # 注意：此参数在核心训练循环中未直接使用
    clip_ti_decay: bool,
    # --- 新增的或从extra_args提取的参数，如果需要的话 ---
    mask_temperature: float = 1.0, # 保持原样，或从 extra_args 获取
    unet_feature_align_weight: float = 0.01, # 保持原样，或从 extra_args 获取
    text_encoder_feature_align_weight: float = 0.02, # 保持原样，或从 extra_args 获取
    noise_pred_weight: float = 1.0, # 保持原样，或从 extra_args 获取
    unet_feature_alignment_layers: List[str] = [ # 保持默认值
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block', 'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    text_encoder_alignment_layers: List[str] = [ # 保持默认值
        'text_model.encoder.layers.0', 'text_model.encoder.layers.6', 'text_model.encoder.layers.11'
    ],
    text_encoder_pooling_strategy: str = "mean", # 保持默认值
    text_encoder_loss_type: str = "mse", # 保持默认值
    unet_return_dict: bool = True, # 保持默认值
    mixed_precision: str = "no", # 假设从 extra_args 或全局配置获取
    tensorboard_log_dir: str = "runs", # 可以设为 os.path.join(save_path, "tensorboard_logs/inversion")
    device: str = "cuda" # 需要从调用者处获取或在此处确定
):
    """
    多教师文本逆向 (TI) 训练函数，参数与 train_multi_lora_distillation 中的调用匹配。
    """
    # --- 从 teacher_pipes 和 dataloaders 字典中提取列表 ---
    # 确保提取顺序一致，例如按字典的键排序
    ordered_teacher_names = sorted(list(teacher_pipes.keys()))
    if not ordered_teacher_names:
        raise ValueError("No teacher pipelines provided in teacher_pipes.")

    teacher_unets_list: List[UNet2DConditionModel] = []
    teacher_text_encoders_list: List[CLIPTextModel] = []
    dataloaders_list: List[torch.utils.data.DataLoader] = []

    for name in ordered_teacher_names:
        pipe = teacher_pipes[name]
        if not hasattr(pipe, 'unet') or not isinstance(pipe.unet, UNet2DConditionModel):
            raise ValueError(f"Teacher pipeline '{name}' does not have a valid UNet model.")
        if not hasattr(pipe, 'text_encoder') or not isinstance(pipe.text_encoder, CLIPTextModel):
            raise ValueError(f"Teacher pipeline '{name}' does not have a valid text encoder.")
        teacher_unets_list.append(pipe.unet)
        teacher_text_encoders_list.append(pipe.text_encoder)

        if name not in dataloaders:
            raise ValueError(f"No dataloader found for teacher '{name}' in the dataloaders dictionary.")
        dataloaders_list.append(dataloaders[name])

    if not (len(teacher_unets_list) == len(teacher_text_encoders_list) == len(dataloaders_list)):
        # 这个检查理论上在上面的循环中已经部分处理了
        raise ValueError("Mismatch in the number of extracted teacher UNets, text encoders, and dataloaders.")
    num_teachers = len(teacher_unets_list)

    # --- 确定设备 ---
    # 优先使用传入的 device 参数，否则尝试从学生模型推断
    if device is None:
        if student_unet.device != torch.device("meta"): # 检查模型是否已在具体设备上
            device = student_unet.device
        else: # 如果模型是meta device，则需要显式指定device，或默认为cuda
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Warning: Device not specified and student_unet is on meta device. Defaulting to {device}.")
    current_device = torch.device(device)


    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    # 使用 save_path 构建 tensorboard 日志路径，确保唯一性
    current_tensorboard_log_dir = os.path.join(tensorboard_log_dir, out_name + "_inversion")
    os.makedirs(current_tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=current_tensorboard_log_dir)
    print(f"TensorBoard logging (Multi-Teacher Inversion) to: {current_tensorboard_log_dir}")


    # --- UNet特征对齐组件设置 ---
    unet_alignment_layers_for_loss = unet_feature_alignment_layers
    unet_layer_weights_for_fa_loss = {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    } # 可以根据需要调整
    unet_layer_weights_for_fa_loss = {
        k: v for k, v in unet_layer_weights_for_fa_loss.items()
        if k in unet_alignment_layers_for_loss
    }
    unet_feature_alignment_loss_fn = FeatureAlignmentLoss( # 假设 FeatureAlignmentLoss 已定义
        alignment_layers=unet_alignment_layers_for_loss,
        loss_weights=unet_layer_weights_for_fa_loss,
        loss_type="mse"
    )
    teacher_unet_extractors = [
        UNetFeatureExtractor( # 假设 UNetFeatureExtractor 已定义
            target_layers=unet_alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        ) for _ in range(num_teachers)
    ]
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )

    # --- Text Encoder特征对齐组件设置 ---
    text_encoder_layer_weights = {layer: 1.0 for layer in text_encoder_alignment_layers}
    if 'text_model.encoder.layers.11' in text_encoder_alignment_layers: # 示例权重调整
        text_encoder_layer_weights['text_model.encoder.layers.11'] = 1.5
    text_encoder_feature_alignment_loss_fn = TextEncoderFeatureAlignmentLoss( # 假设已定义
        alignment_layers=text_encoder_alignment_layers,
        loss_weights=text_encoder_layer_weights,
        loss_type=text_encoder_loss_type,
        pooling_strategy=text_encoder_pooling_strategy,
        temperature=1.0 # 示例温度
    )
    text_encoder_feature_extractor = TextEncoderFeatureExtractor( # 假设已定义
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps), desc="Multi-Teacher TI Steps")
    global_step = 0

    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone().to(current_device)
    index_no_updates_on_device = index_no_updates.to(current_device)
    index_updates = ~index_no_updates_on_device


    if log_wandb:
        # wandb.init(...) # 根据您的wandb设置逻辑
        pass

    accumulated_total_loss = 0.0
    accumulated_ti_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    iter_dataloaders = [iter(dl) for dl in dataloaders_list]

    # 将学生模型移到目标设备 (如果尚未移动)
    student_unet.to(current_device)
    student_text_encoder.to(current_device)
    vae.to(current_device)


    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        current_teacher_idx = global_step % num_teachers
        current_dataloader_iter = iter_dataloaders[current_teacher_idx]
        current_dataloader_obj = dataloaders_list[current_teacher_idx]
        # 将当前教师模型移到目标设备
        current_teacher_unet = teacher_unets_list[current_teacher_idx].to(current_device)
        current_teacher_text_encoder = teacher_text_encoders_list[current_teacher_idx].to(current_device)
        current_teacher_unet_extractor = teacher_unet_extractors[current_teacher_idx]
        teacher_name_log = ordered_teacher_names[current_teacher_idx]


        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            print(f"Info: Dataloader for Teacher Inversion '{teacher_name_log}' exhausted. Re-initializing.")
            iter_dataloaders[current_teacher_idx] = iter(current_dataloader_obj)
            batch = next(iter_dataloaders[current_teacher_idx])

        # 将批次数据移到目标设备
        # 假设 batch 是一个字典，其值是张量
        batch = {k: v.to(current_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


        main_student_unet = student_unet.module if hasattr(student_unet, 'module') else student_unet
        expected_latents_dtype = main_student_unet.dtype

        if cached_latents:
            # 假设 "pixel_values" 键包含的是潜变量
            latents = batch["pixel_values"].to(dtype=expected_latents_dtype)
        else:
            pixel_values_for_vae = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values_for_vae).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(device=current_device, dtype=expected_latents_dtype)

        input_ids = batch["input_ids"] # 已经在 current_device 上
        attention_mask = batch.get("attention_mask", None) # 已经在 current_device 上 (如果存在)

        main_teacher_unet = current_teacher_unet.module if hasattr(current_teacher_unet, 'module') else current_teacher_unet
        teacher_unet_internal_dtype = next(main_teacher_unet.parameters()).dtype
        student_unet_internal_dtype = next(main_student_unet.parameters()).dtype

        # 教师文本编码器（无梯度）
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
                teacher_text_output, teacher_text_features = text_encoder_feature_extractor.extract_features(
                    text_encoder=current_teacher_text_encoder,
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_grad=False
                )
                teacher_encoder_hidden_states = teacher_text_output.to(dtype=teacher_unet_internal_dtype)
        teacher_text_features = {k: v.to(current_device) for k, v in teacher_text_features.items()}


        # 学生文本编码器（有梯度）
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
            student_text_output, student_text_features = text_encoder_feature_extractor.extract_features(
                text_encoder=student_text_encoder,
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_grad=True
            )
            student_encoder_hidden_states = student_text_output.to(dtype=student_unet_internal_dtype)
        student_text_features = {k: v.to(current_device) for k, v in student_text_features.items()}


        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

        # 教师UNet前向传播（无梯度）
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
                _, teacher_unet_features = current_teacher_unet_extractor.extract_features(
                    unet_model=current_teacher_unet,
                    sample=noisy_latents.to(dtype=teacher_unet_internal_dtype), # 已在 current_device
                    timestep=timesteps, # 已在 current_device
                    encoder_hidden_states=teacher_encoder_hidden_states, # 已在 current_device
                    return_dict=unet_return_dict, use_grad=False
                )
        teacher_unet_features = {k: v.to(current_device) for k, v in teacher_unet_features.items()}

        # 学生UNet特征提取（用于对齐，梯度流向文本嵌入）
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and current_device.type == 'cuda')):
            _, student_unet_features = student_unet_extractor.extract_features(
                unet_model=student_unet,
                sample=noisy_latents.to(dtype=student_unet_internal_dtype), # 已在 current_device
                timestep=timesteps, # 已在 current_device
                encoder_hidden_states=student_encoder_hidden_states, # 已在 current_device, 带梯度
                return_dict=unet_return_dict, use_grad=True
            )
        student_unet_features = {k: v.to(current_device) for k, v in student_unet_features.items()}


        # TI 核心损失 (例如，噪声预测损失)
        # loss_step_gaussian_noise 函数内部会进行学生 UNet 的前向传播
        ti_loss_current_step = loss_step_gaussian_noise( # 假设 loss_step_gaussian_noise 已定义
            batch=batch, student_unet=student_unet, student_text_encoder=student_text_encoder,
            vae=vae, global_step=global_step, scheduler=scheduler, t_mutliplier=0.8, # 示例 t_multiplier
            mixed_precision=use_mixed_precision, mask_temperature=mask_temperature,
            # 如果 loss_step_gaussian_noise 需要 device 参数，请传递 current_device
        )

        # 特征对齐损失
        unet_feature_align_loss_val, unet_feature_loss_dict_current = unet_feature_alignment_loss_fn(
            teacher_unet_features, student_unet_features
        )
        text_encoder_feature_align_loss_val, text_encoder_feature_loss_dict_current = text_encoder_feature_alignment_loss_fn(
            teacher_features=teacher_text_features, student_features=student_text_features,
            teacher_attention_mask=attention_mask, student_attention_mask=attention_mask
        )

        # 确保所有损失都在同一设备上
        ti_loss_current_step = ti_loss_current_step.to(current_device)
        unet_feature_align_loss_val = unet_feature_align_loss_val.to(current_device)
        text_encoder_feature_align_loss_val = text_encoder_feature_align_loss_val.to(current_device)


        # 动态权重调整 (可选)
        _inversion_weight_adjuster = create_inversion_weight_adjuster( # 假设已定义
            noise_pred_weight, unet_feature_align_weight, text_encoder_feature_align_weight
        )
        current_losses_for_adj = {
            'ti_loss': ti_loss_current_step.item(),
            'unet_feature_align': unet_feature_align_loss_val.item(),
            'text_encoder_feature_align': text_encoder_feature_align_loss_val.item()
        }
        updated_weights = _inversion_weight_adjuster.update_weights(current_losses_for_adj)
        current_noise_pred_weight = updated_weights['ti_loss']
        current_unet_feature_align_weight = updated_weights['unet_feature_align']
        current_text_encoder_feature_align_weight = updated_weights['text_encoder_feature_align']

        # 总损失
        current_step_total_loss = (
            current_noise_pred_weight * ti_loss_current_step +
            current_unet_feature_align_weight * unet_feature_align_loss_val +
            current_text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
        )

        loss_for_backward = current_step_total_loss / accum_iter
        loss_for_backward.backward() # 梯度会累积到 student_text_encoder 的词嵌入参数上

        accumulated_total_loss += current_step_total_loss.detach().item()
        accumulated_ti_loss += ti_loss_current_step.detach().item()
        accumulated_unet_feature_loss += unet_feature_align_loss_val.detach().item()
        accumulated_text_encoder_feature_loss += text_encoder_feature_align_loss_val.detach().item()
        step_count_in_interval += 1

        if (global_step + 1) % accum_iter == 0:
            # 可选: 梯度裁剪 (如果需要)
            # torch.nn.utils.clip_grad_norm_(student_text_encoder.get_input_embeddings().parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                embed_weights = student_text_encoder.get_input_embeddings().weight
                if clip_ti_decay:
                    updated_embeds = embed_weights[index_updates, :]
                    pre_norm = updated_embeds.norm(dim=-1, keepdim=True)
                    lambda_ = min(1.0, 100 * lr_scheduler_main.get_last_lr()[0])
                    normalized_embeds = F.normalize(updated_embeds, dim=-1)
                    target_norm = pre_norm + lambda_ * (0.4 - pre_norm) # 0.4 是示例目标范数
                    embed_weights.data[index_updates] = normalized_embeds * target_norm
                # 恢复未被更新的原始嵌入
                embed_weights.data[index_no_updates_on_device] = orig_embeds_params[index_no_updates_on_device]

        lr_scheduler_main.step()
        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0]

        log_dict_postfix = {
            "Total": f"{current_step_total_loss.item():.4f}",
            "TI": f"{ti_loss_current_step.item():.4f}",
            "UFeat": f"{unet_feature_align_loss_val.item():.4f}",
            "TFeat": f"{text_encoder_feature_align_loss_val.item():.4f}",
            "LR": f"{current_lr_val:.2e}",
            "Teacher": teacher_name_log[:10] # 截断教师名称以防过长
        }
        progress_bar.set_postfix(**log_dict_postfix)

        # TensorBoard 日志记录
        if writer:
            writer.add_scalar('Loss_Inv/Total_Step', current_step_total_loss.item(), global_step)
            writer.add_scalar('Loss_Inv/TI_Step', ti_loss_current_step.item(), global_step)
            writer.add_scalar('Loss_Inv/UNet_Feat_Step', unet_feature_align_loss_val.item(), global_step)
            writer.add_scalar('Loss_Inv/TextEnc_Feat_Step', text_encoder_feature_align_loss_val.item(), global_step)
            writer.add_scalar('LearningRate_Inv', current_lr_val, global_step)
            writer.add_scalar('Teacher_Model_Idx_Inv', current_teacher_idx + 1, global_step)
            # ... 其他权重日志 ...

        # wandb 日志记录 (如果启用)
        if log_wandb and global_step % 10 == 0: # 每10步记录一次示例
            # wandb.log({ ... })
            pass

        if global_step > 0 and global_step % save_steps == 0:
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            # ... 计算其他平均损失 ...
            print(f"\nCHECKPOINT - TI Step {global_step}/{num_steps}")
            print(f"  Avg Total Loss: {avg_total_loss_interval:.6f}")
            # ... 打印其他平均损失 ...

            current_save_dir = os.path.join(save_path, out_name) # 保存到 output_dir/out_name/
            os.makedirs(current_save_dir, exist_ok=True)
            checkpoint_filename = f"inversion_step_{global_step}.safetensors"
            save_all( # 假设 save_all 已定义
                unet=student_unet, text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=os.path.join(current_save_dir, checkpoint_filename),
                save_lora=False, # TI 保存的是嵌入，不是LoRA层
                target_replace_module_unet=lora_unet_target_modules, # 传递以防 save_all 需要
                target_replace_module_text=lora_clip_target_modules  # 传递以防 save_all 需要
            )
            print(f"✓ TI Checkpoint saved to: {os.path.join(current_save_dir, checkpoint_filename)}\n")

            accumulated_total_loss = 0.0
            accumulated_ti_loss = 0.0
            accumulated_unet_feature_loss = 0.0
            accumulated_text_encoder_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    if writer:
        writer.close()

    # --- 保存最终模型 ---
    final_save_dir = os.path.join(save_path, out_name)
    os.makedirs(final_save_dir, exist_ok=True)
    final_model_filename = f"inversion_final_step_{global_step}.safetensors"
    save_all(
        unet=student_unet, text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(final_save_dir, final_model_filename),
        save_lora=False,
        target_replace_module_unet=lora_unet_target_modules,
        target_replace_module_text=lora_clip_target_modules
    )
    print(f"\n✓ Final TI model saved to: {os.path.join(final_save_dir, final_model_filename)}")
    print(f"✓ TensorBoard logs for TI saved to: {current_tensorboard_log_dir}")

def train_inversion_with_multi_teachers_feature_alignment(
    # --- "Teachers" 模型参数 (支持多个教师) ---
    teacher_unets: list,  # 教师模型UNet列表
    teacher_text_encoders: list,  # 教师模型Text Encoder列表
    # --- "Student" 模型参数 ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader 参数 (支持多个数据加载器) ---
    dataloaders: list,  # 数据加载器列表，与教师模型一一对应
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
    text_encoder_feature_align_weight: float = 0.5, # Text Encoder特征对齐损失的权重
    text_encoder_alignment_layers=[ # 用于Text Encoder特征对齐的层
        'text_model.encoder.layers.0',
        'text_model.encoder.layers.6', 
        'text_model.encoder.layers.11'
    ],
    text_encoder_pooling_strategy: str = "mean", # "mean", "cls", "max", "none"
    text_encoder_loss_type: str = "mse", # "mse", "l1", "cosine"
    # --- 教师模型选择策略 ---
    teacher_selection_strategy: str = "round_robin", # "round_robin", "random", "weighted"
    teacher_weights: list = None, # 用于weighted策略的权重列表，如果为None则均等权重
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
):
    """
    train_inversion_with_multi_teachers_feature_alignment 函数：
    支持多个教师模型的增强版文本逆向训练，同时使用UNet和Text Encoder的特征对齐。
    
    新增功能：
    1. 支持任意数量的教师模型
    2. 灵活的教师模型选择策略（轮转、随机、加权）
    3. UNet层间特征对齐
    4. Text Encoder层间特征对齐
    5. 可以独立控制各种特征对齐的权重
    
    参数说明：
    - teacher_unets: 教师UNet模型列表
    - teacher_text_encoders: 教师Text Encoder模型列表
    - dataloaders: 对应的数据加载器列表
    - teacher_selection_strategy: 教师选择策略
        - "round_robin": 轮流使用教师模型
        - "random": 随机选择教师模型
        - "weighted": 根据权重选择教师模型
    - teacher_weights: 教师模型权重列表（仅在weighted策略下使用）
    """
    
    # 验证输入参数
    num_teachers = len(teacher_unets)
    if len(teacher_text_encoders) != num_teachers:
        raise ValueError(f"教师UNet数量({num_teachers})与教师Text Encoder数量({len(teacher_text_encoders)})不匹配")
    if len(dataloaders) != num_teachers:
        raise ValueError(f"教师模型数量({num_teachers})与数据加载器数量({len(dataloaders)})不匹配")
    
    if teacher_selection_strategy == "weighted":
        if teacher_weights is None:
            teacher_weights = [1.0] * num_teachers  # 默认均等权重
        elif len(teacher_weights) != num_teachers:
            raise ValueError(f"教师权重数量({len(teacher_weights)})与教师模型数量({num_teachers})不匹配")
        # 归一化权重
        total_weight = sum(teacher_weights)
        teacher_weights = [w / total_weight for w in teacher_weights]
    
    print(f"初始化多教师训练，共有 {num_teachers} 个教师模型")
    print(f"教师选择策略: {teacher_selection_strategy}")
    if teacher_selection_strategy == "weighted":
        print(f"教师权重: {teacher_weights}")

    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + f"_inversion_multi_teachers_{num_teachers}_feat_align")
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard 日志 (多教师特征对齐的文本逆向) 将保存到: {tb_log_path}")

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

    # 为每个教师UNet创建特征提取器
    teacher_unet_extractors = []
    for i in range(num_teachers):
        extractor = UNetFeatureExtractor(
            target_layers=unet_alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        )
        teacher_unet_extractors.append(extractor)
    
    # 学生UNet特征提取器
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )

    # --- Text Encoder特征对齐组件设置 ---
    text_encoder_layer_weights = {
        layer: 1.0 for layer in text_encoder_alignment_layers
    }
    # 可以为特定层设置不同权重
    if 'text_model.encoder.layers.11' in text_encoder_alignment_layers:
        text_encoder_layer_weights['text_model.encoder.layers.11'] = 1.5
    if 'text_model.encoder.layers.6' in text_encoder_alignment_layers:
        text_encoder_layer_weights['text_model.encoder.layers.6'] = 1.2

    text_encoder_feature_alignment_loss_fn = TextEncoderFeatureAlignmentLoss(
        alignment_layers=text_encoder_alignment_layers,
        loss_weights=text_encoder_layer_weights,
        loss_type=text_encoder_loss_type,
        pooling_strategy=text_encoder_pooling_strategy,
        temperature=1.0
    )

    # Text Encoder特征提取器
    text_encoder_feature_extractor = TextEncoderFeatureExtractor(
        target_layers=text_encoder_alignment_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description(f"训练步数 (文本逆向 + {num_teachers}教师特征对齐)")
    global_step = 0

    # 备份原始的、非占位符的词元嵌入
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        if wandb.run is None:
            wandb.init(
                project=f"textual_inversion_project_{out_name}", 
                name=f"{out_name}_inversion_multi_teachers_{num_teachers}_feat_align_run", 
                reinit=True
            )
        wandb.config.update({
            k:v for k,v in locals().items() 
            if isinstance(v, (int, float, str, bool, list, dict)) and k not in ['teacher_unets', 'teacher_text_encoders', 'dataloaders']
        })
        wandb.config.update({
            'num_teachers': num_teachers,
            'teacher_selection_strategy': teacher_selection_strategy
        })
        preped_clip = prepare_clip_model_sets() if 'prepare_clip_model_sets' in globals() else None

    index_updates = ~index_no_updates

    # 用于区间日志记录的累积损失
    accumulated_total_loss = 0.0
    accumulated_ti_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    # 初始化数据加载器迭代器
    dataloader_iterators = [iter(dataloader) for dataloader in dataloaders]
    
    # 教师选择相关的状态
    current_teacher_index = 0  # 用于round_robin策略
    teacher_usage_count = [0] * num_teachers  # 记录每个教师的使用次数

    # 从模型确定设备
    unet_device = next(student_unet.parameters()).device
    text_encoder_device = next(student_text_encoder.parameters()).device
    vae_device = vae.device

    def select_teacher_index(step):
        """根据策略选择教师模型索引"""
        nonlocal current_teacher_index
        
        if teacher_selection_strategy == "round_robin":
            idx = step % num_teachers
            return idx
        elif teacher_selection_strategy == "random":
            return random.randint(0, num_teachers - 1)
        elif teacher_selection_strategy == "weighted":
            return random.choices(range(num_teachers), weights=teacher_weights)[0]
        else:
            raise ValueError(f"不支持的教师选择策略: {teacher_selection_strategy}")

    # 主训练循环
    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # --- 选择当前教师模型和数据加载器 ---
        teacher_idx = select_teacher_index(global_step)
        teacher_usage_count[teacher_idx] += 1
        
        current_teacher_unet = teacher_unets[teacher_idx]
        current_teacher_text_encoder = teacher_text_encoders[teacher_idx]
        current_teacher_unet_extractor = teacher_unet_extractors[teacher_idx]
        current_dataloader_iter = dataloader_iterators[teacher_idx]
        current_dataloader_obj = dataloaders[teacher_idx]
        teacher_name_log = f"T{teacher_idx+1}"
        
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            dataloader_iterators[teacher_idx] = iter(current_dataloader_obj)
            batch = next(dataloader_iterators[teacher_idx])

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
                t_mutliplier=0.8,
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

            # 权重调整器
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

            # 6. 总损失
            current_step_total_loss = (
                noise_pred_weight * ti_loss_current_step +
                unet_feature_align_weight * unet_feature_align_loss_val +
                text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
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
                        writer.add_scalar('梯度/文本嵌入范数_多教师FA', grad_norm.item(), global_step)
            else:
                print(f"步骤 {global_step}: 警告：在多教师FA更新期间未找到文本嵌入的梯度。")

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
                    writer.add_scalar('嵌入/当前范数_多教师FA', current_norm_val, global_step)

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
            writer.add_scalar('损失_多教师FA/总损失_单步', current_step_total_loss.item(), global_step)
            writer.add_scalar('损失_多教师FA/TI损失_单步', ti_loss_current_step.item(), global_step)
            writer.add_scalar('损失_多教师FA/UNet特征对齐损失_单步', unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val), global_step)
            writer.add_scalar('损失_多教师FA/TextEncoder特征对齐损失_单步', text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val), global_step)
            writer.add_scalar('学习率_多教师FA/文本嵌入', current_lr_val, global_step)
            writer.add_scalar('教师模型_多教师FA/当前使用教师', teacher_idx + 1, global_step)
            writer.add_scalar('Weights/TI_Loss_Weight', noise_pred_weight, global_step)
            writer.add_scalar('Weights/UNet_Feature_Weight', unet_feature_align_weight, global_step)
            writer.add_scalar('Weights/TextEncoder_Feature_Weight', text_encoder_feature_align_weight, global_step)
            
            # 记录各教师的使用次数
            for i, count in enumerate(teacher_usage_count):
                writer.add_scalar(f'教师模型_多教师FA/教师{i+1}_使用次数', count, global_step)
            
            # 记录各层的UNet特征损失
            if isinstance(unet_feature_loss_dict_current, dict):
                for layer_name, layer_loss in unet_feature_loss_dict_current.items():
                    writer.add_scalar(f'UNet特征损失_多教师FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
            
            # 记录各层的Text Encoder特征损失
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                for layer_name, layer_loss in text_encoder_feature_loss_dict_current.items():
                    writer.add_scalar(f'TextEncoder特征损失_多教师FA_层/{layer_name.replace(".", "_")}', layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss), global_step)
        
        # wandb 日志记录 (继续上面被截断的部分)
        wandb_step_logs = {}
        if log_wandb and global_step % 10 == 0:
            wandb_step_logs.update({
                'step': global_step,
                'total_loss_step_multi_teachers_fa': current_step_total_loss.item(),
                'ti_loss_step_multi_teachers_fa': ti_loss_current_step.item(),
                'unet_feature_align_loss_step_multi_teachers_fa': unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val),
                'text_encoder_feature_align_loss_step_multi_teachers_fa': text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val),
                'learning_rate_multi_teachers_fa': current_lr_val,
                'current_teacher_index': teacher_idx + 1,
                'ti_loss_weight': noise_pred_weight,
                'unet_feature_weight': unet_feature_align_weight,
                'text_encoder_feature_weight': text_encoder_feature_align_weight,
            })
            
            # 记录教师使用统计
            for i, count in enumerate(teacher_usage_count):
                wandb_step_logs[f'teacher_{i+1}_usage_count'] = count
            
            # 记录层级损失详情
            if isinstance(unet_feature_loss_dict_current, dict):
                for layer_name, layer_loss in unet_feature_loss_dict_current.items():
                    clean_layer_name = layer_name.replace(".", "_")
                    wandb_step_logs[f'unet_layer_loss_{clean_layer_name}'] = layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss)
            
            if isinstance(text_encoder_feature_loss_dict_current, dict):
                for layer_name, layer_loss in text_encoder_feature_loss_dict_current.items():
                    clean_layer_name = layer_name.replace(".", "_")
                    wandb_step_logs[f'text_encoder_layer_loss_{clean_layer_name}'] = layer_loss.item() if torch.is_tensor(layer_loss) else float(layer_loss)
            
            wandb.log(wandb_step_logs)

        # 定期保存检查点
        if global_step % save_steps == 0 and global_step > 0:
            print(f"\n保存检查点 - 步骤 {global_step}")
            
            # 创建保存目录
            checkpoint_save_dir = os.path.join(save_path, f"checkpoint_step_{global_step}")
            if not os.path.exists(checkpoint_save_dir):
                os.makedirs(checkpoint_save_dir, exist_ok=True)
            
            # 保存模型权重
            save_all(
                unet=student_unet, 
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=os.path.join(checkpoint_save_dir, f"checkpoint_step_{global_step}.safetensors"),
                save_lora=False,
            )
            
            # 保存训练状态信息
            training_state = {
                'global_step': global_step,
                'num_teachers': num_teachers,
                'teacher_selection_strategy': teacher_selection_strategy,
                'teacher_usage_count': teacher_usage_count,
                'teacher_weights': teacher_weights if teacher_selection_strategy == "weighted" else None,
                'current_losses': {
                    'total_loss': current_step_total_loss.item(),
                    'ti_loss': ti_loss_current_step.item(),
                    'unet_feature_align_loss': unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val),
                    'text_encoder_feature_align_loss': text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val),
                },
                'current_weights': {
                    'noise_pred_weight': noise_pred_weight,
                    'unet_feature_align_weight': unet_feature_align_weight,
                    'text_encoder_feature_align_weight': text_encoder_feature_align_weight,
                },
                'learning_rate': current_lr_val,
            }
            
            # 保存训练状态到JSON文件
            with open(os.path.join(checkpoint_save_dir, f"training_state_step_{global_step}.json"), 'w') as f:
                json.dump(training_state, f, indent=2, ensure_ascii=False)
            
            print(f"检查点已保存到: {checkpoint_save_dir}")
            
            # 如果设置了wandb，记录检查点保存事件
            if log_wandb:
                wandb.log({
                    'checkpoint_saved': global_step,
                    'checkpoint_path': checkpoint_save_dir
                }, step=global_step)

        # 区间损失平均记录 (每50步记录一次平均值)
        if global_step % 50 == 0 and step_count_in_interval > 0:
            avg_total_loss = accumulated_total_loss / step_count_in_interval
            avg_ti_loss = accumulated_ti_loss / step_count_in_interval
            avg_unet_feature_loss = accumulated_unet_feature_loss / step_count_in_interval
            avg_text_encoder_feature_loss = accumulated_text_encoder_feature_loss / step_count_in_interval
            
            if writer:
                writer.add_scalar('损失_多教师FA/总损失_50步平均', avg_total_loss, global_step)
                writer.add_scalar('损失_多教师FA/TI损失_50步平均', avg_ti_loss, global_step)
                writer.add_scalar('损失_多教师FA/UNet特征对齐损失_50步平均', avg_unet_feature_loss, global_step)
                writer.add_scalar('损失_多教师FA/TextEncoder特征对齐损失_50步平均', avg_text_encoder_feature_loss, global_step)
            
            if log_wandb:
                wandb.log({
                    'avg_total_loss_50_steps': avg_total_loss,
                    'avg_ti_loss_50_steps': avg_ti_loss,
                    'avg_unet_feature_loss_50_steps': avg_unet_feature_loss,
                    'avg_text_encoder_feature_loss_50_steps': avg_text_encoder_feature_loss,
                }, step=global_step)
            
            # 重置累积损失
            accumulated_total_loss = 0.0
            accumulated_ti_loss = 0.0
            accumulated_unet_feature_loss = 0.0
            accumulated_text_encoder_feature_loss = 0.0
            step_count_in_interval = 0

    # 训练完成后的最终保存
    print(f"\n训练完成！总步数: {global_step}")
    print(f"教师模型使用统计: {dict(zip([f'Teacher_{i+1}' for i in range(num_teachers)], teacher_usage_count))}")
    
    # 创建最终保存目录
    final_save_dir = os.path.join(save_path, "final_model")
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)
    
    # 保存最终模型
    print("正在保存最终模型...")
    save_all(
        unet=student_unet, 
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(final_save_dir, f"final_step_inv_{global_step}.safetensors"),
        save_lora=False,
    )
    
    # 保存最终训练统计信息
    final_training_stats = {
        'total_steps': global_step,
        'num_teachers': num_teachers,
        'teacher_selection_strategy': teacher_selection_strategy,
        'teacher_usage_statistics': {
            f'teacher_{i+1}': {
                'usage_count': teacher_usage_count[i],
                'usage_percentage': (teacher_usage_count[i] / global_step) * 100
            } for i in range(num_teachers)
        },
        'final_weights': {
            'noise_pred_weight': noise_pred_weight,
            'unet_feature_align_weight': unet_feature_align_weight,
            'text_encoder_feature_align_weight': text_encoder_feature_align_weight,
        },
        'training_config': {
            'unet_feature_alignment_layers': unet_feature_alignment_layers,
            'text_encoder_alignment_layers': text_encoder_alignment_layers,
            'text_encoder_pooling_strategy': text_encoder_pooling_strategy,
            'text_encoder_loss_type': text_encoder_loss_type,
        }
    }
    
    # 保存最终统计信息
    with open(os.path.join(final_save_dir, "final_training_stats.json"), 'w') as f:
        json.dump(final_training_stats, f, indent=2, ensure_ascii=False)
    
    print(f"最终模型已保存到: {final_save_dir}")
    print("训练统计信息:")
    for i in range(num_teachers):
        usage_pct = (teacher_usage_count[i] / global_step) * 100
        print(f"  教师模型 {i+1}: 使用 {teacher_usage_count[i]} 次 ({usage_pct:.1f}%)")
    
    # 关闭TensorBoard writer
    if writer:
        writer.close()
        print(f"TensorBoard 日志已保存到: {tb_log_path}")
    
    # 最终wandb日志
    if log_wandb:
        wandb.log({
            'training_completed': True,
            'final_step': global_step,
            'final_model_path': final_save_dir,
        })
        
        # 记录教师使用统计的表格
        teacher_usage_table = wandb.Table(
            columns=['Teacher', 'Usage Count', 'Usage Percentage'],
            data=[[f'Teacher_{i+1}', teacher_usage_count[i], f'{(teacher_usage_count[i] / global_step) * 100:.1f}%'] 
                  for i in range(num_teachers)]
        )
        wandb.log({'teacher_usage_statistics': teacher_usage_table})
        
        print("训练日志已同步到 Weights & Biases")
    
    # return {
    #     'final_model_path': final_save_dir,
    #     'total_steps': global_step,
    #     'teacher_usage_count': teacher_usage_count,
    #     'final_losses': {
    #         'total_loss': current_step_total_loss.item(),
    #         'ti_loss': ti_loss_current_step.item(),
    #         'unet_feature_align_loss': unet_feature_align_loss_val.item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val),
    #         'text_encoder_feature_align_loss': text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val),
    #     }
    # }

def train_multi_lora_distillation(
    lora_folder_path: str,
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = False,
    use_template: Literal[None, "object", "style"] = None,
    train_inpainting: bool = False,
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
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
    device="cuda:3",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "multi_lora_distill_project",
    wandb_entity: str = "multi_lora_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_multi_lora",
    feature_align_weight: float = 0.01,
    noise_pred_weight: float = 1.0,
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"multi_lora_steps_{max_train_steps_ti}_lr_{learning_rate_ti}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # 1. 扫描LoRA文件夹
    lora_files = scan_lora_folder(lora_folder_path)
    if len(lora_files) == 0:
        raise ValueError(f"No LoRA files found in {lora_folder_path}")

    # 2. 创建placeholder tokens和initializer tokens
    placeholder_tokens_list = []
    initializer_tokens_list = []
    token_maps = {}
    
    for lora_name in lora_files.keys():
        placeholder_token = f"<{lora_name}>"
        placeholder_tokens_list.append(placeholder_token)
        initializer_tokens_list.append("<rand-0.017>")  # 默认随机初始化
        token_maps[lora_name] = {"DUMMY": placeholder_token}
        
    print("Multi-LoRA Placeholder Tokens:", placeholder_tokens_list)
    print("Multi-LoRA Initializer Tokens:", initializer_tokens_list)

    # 3. 获取student模型
    student_text_encoder, student_vae, student_unet, student_tokenizer, placeholder_token_ids, _ = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens_list,
        [],  # 第二组placeholder tokens为空
        initializer_tokens_list,
        [],  # 第二组initializer tokens为空
        device=device,
    )

    # 4. 创建teacher pipelines
    teacher_pipes = create_teacher_pipelines(lora_files, pretrained_model_name_or_path, device)

    # 5. 创建数据集和数据加载器
    datasets, dataloaders = create_multi_datasets(
        teacher_pipes, token_maps, use_template, student_tokenizer, device
    )

    # 6. 设置噪声调度器
    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            student_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 7. 设置学习率
    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = learning_rate_text * gradient_accumulation_steps * train_batch_size
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    # 8. 设置模型参数的可训练性
    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.encoder.parameters(),
        student_text_encoder.text_model.final_layer_norm.parameters(),
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # 9. 设置index_no_updates
    index_no_updates = torch.ones(len(student_tokenizer), dtype=torch.bool, device=student_text_encoder.device)
    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    # 10. 执行Textual Inversion (如果需要)
    if perform_inversion:
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

        # 这里需要修改train_inversion函数以支持多个teacher
        train_multi_teacher_inversion(
            teacher_pipes=teacher_pipes,
            student_unet=student_unet,
            vae=student_vae,
            student_text_encoder=student_text_encoder,
            dataloaders=dataloaders,
            num_steps=max_train_steps_ti,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            save_steps=save_steps,
            placeholder_token_ids=placeholder_token_ids,
            placeholder_tokens=placeholder_tokens_list,
            save_path=output_dir,
            lr_scheduler_main=lr_scheduler,
            lora_unet_target_modules=lora_unet_target_modules,
            lora_clip_target_modules=lora_clip_target_modules,
            out_name=out_name,
            tokenizer=student_tokenizer,
            test_image_path=instance_data_dir,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=proxy_token,
            train_inpainting=train_inpainting,
            clip_ti_decay=clip_ti_decay,
            device=device,
        )

        del ti_optimizer

    # 11. LoRA微调
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("Using Extended UNet LoRA")
        lora_unet_target_modules = lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        unet_lora_params, _ = inject_trainable_lora_extended(
            student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )

    print(f"UNet has {len(unet_lora_params)} LoRA parameters")
    inspect_lora(student_unet)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

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
    else:
        student_text_encoder.requires_grad_(False)

    if train_text_encoder:
        student_text_encoder_lora_params, _ = inject_trainable_lora(
            student_text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank,
        )
        params_to_optimize += [
            {
                "params": itertools.chain(*student_text_encoder_lora_params),
                "lr": text_encoder_lr,
            }
        ]
        inspect_lora(student_text_encoder)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    student_unet.train()
    if train_text_encoder:
        student_text_encoder.train()

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )

    # 12. 执行多teacher LoRA微调
    perform_multi_teacher_tuning(
        teacher_pipes=teacher_pipes,
        student_unet=student_unet,
        vae=list(teacher_pipes.values())[0].vae,  # 使用第一个teacher的VAE
        student_text_encoder=student_text_encoder,
        dataloaders=dataloaders,
        num_steps=max_train_steps_tuning,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens_list,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=student_tokenizer,
        out_name=out_name,
        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=proxy_token,
        train_inpainting=train_inpainting,
        feature_align_weight=feature_align_weight,
        noise_pred_weight=noise_pred_weight,
        device=device,
    )


if __name__ == '__main__':
    import argparse
    import os # 确保导入 os
    import torch # 确保导入 torch
    import itertools # 确保导入 itertools
    from typing import Optional, Literal, Dict, List, Any # 确保导入类型提示
    # 假设 get_models, create_teacher_pipelines, create_multi_datasets,
    # inject_trainable_lora, inject_trainable_lora_extended, inspect_lora,
    # UNET_EXTENDED_TARGET_REPLACE, get_scheduler, scan_lora_folder 等函数已定义或导入
    # 例如:
    # from .utils import (
    #     scan_lora_folder, get_models, create_teacher_pipelines, create_multi_datasets,
    #     inject_trainable_lora, inject_trainable_lora_extended, inspect_lora, get_scheduler
    # )
    # from .unet_target_replace import UNET_EXTENDED_TARGET_REPLACE
    # from diffusers import DDPMScheduler
    # import torch.optim as optim
    # import wandb # 如果使用wandb

    # --- (在此处粘贴或导入 train_multi_lora_distillation, 
    #        perform_multi_teacher_tuning, train_multi_teacher_inversion 
    #        以及它们依赖的辅助函数/类) ---
    # 或者确保它们在当前作用域内可见

    parser = argparse.ArgumentParser(description='Training script for Multi-LoRA Distillation')

    # --- 修改核心输入参数 ---
    parser.add_argument('--lora_folder_path', type=str, required=True,
                        help='Path to the folder containing multiple LoRA model files (.safetensors or .bin)')
    # 移除了 lora_path1 和 lora_path2

    parser.add_argument('--instance_data_dir', type=str, required=False, default="", # 用于TI的测试图像路径或数据集的根目录
                        help='Directory for instance data (e.g., test images for TI, or root for dataset creation)')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Path to pretrained base model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving models and logs')

    # --- 其他参数保持不变或根据 train_multi_lora_distillation 函数定义调整 ---
    parser.add_argument('--train_text_encoder', action='store_true', default=True, help='Whether to train text encoder LoRA') # 默认值与函数定义一致
    parser.add_argument('--pretrained_vae_name_or_path', type=str, default=None, help='Path to pretrained VAE (optional)')
    parser.add_argument('--revision', type=str, default=None, help='Revision of pretrained model')
    parser.add_argument('--perform_inversion', action='store_true', default=True, help='Perform Textual Inversion for placeholder tokens') # 默认值与函数定义一致
    parser.add_argument('--use_template', type=str, choices=[None, "object", "style"], default=None, help='Prompt template to use for dataset creation (e.g., "a photo of a {}")') # 与函数定义一致
    parser.add_argument('--train_inpainting', action='store_true', default=False, help='Train for inpainting task (not fully integrated in provided snippet)')

    # placeholder_tokens 和 initializer_tokens 将在 train_multi_lora_distillation 内部根据 LoRA 文件名自动生成
    # 因此从命令行移除，除非有特殊需求覆盖自动生成逻辑
    # parser.add_argument('--placeholder_token_at_data', type=str, default=None, help='Placeholder token string present in data (if any)') # 这个可以保留，如果数据集描述已包含占位符
    # parser.add_argument('--initializer_tokens', type=str, default=None, help='Initializer tokens (comma-separated, if specific init needed)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution for training images and VAE encoding')
    parser.add_argument('--color_jitter', action='store_true', default=True, help='Apply color jitter augmentation during dataset creation')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size per device for TI and LoRA tuning')
    parser.add_argument('--sample_batch_size', type=int, default=1, help='Sampling batch size (used by some helper logging/eval, not core)')
    parser.add_argument('--max_train_steps_tuning', type=int, default=1000, help='Max training steps for LoRA tuning phase')
    parser.add_argument('--max_train_steps_ti', type=int, default=1000, help='Max training steps for Textual Inversion phase')
    parser.add_argument('--save_steps', type=int, default=100, help='Save a checkpoint every X steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients before an optimizer step')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Enable gradient checkpointing for UNet to save memory')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA matrices')

    # lora_unet_target_modules 和 lora_clip_target_modules 的默认值应该已经是集合了
    # 如果命令行传入的是字符串，train_multi_lora_distillation 应该在内部处理或期望集合类型
    # 保持默认值为函数定义中的类型 (set)
    # 如果从命令行解析，通常解析为字符串，然后在代码中转换为集合
    parser.add_argument('--lora_unet_target_modules', type=str, default="CrossAttention,Attention,GEGLU",
                       help='Target modules for UNet LoRA (comma-separated string, e.g., "CrossAttention,Attention,GEGLU")')
    parser.add_argument('--lora_clip_target_modules', type=str, default="CLIPSdpaAttention",
                       help='Target modules for CLIP LoRA (comma-separated string, e.g., "CLIPAttention,CLIPMLP")')

    parser.add_argument('--lora_dropout_p', type=float, default=0.0, help='Dropout probability for LoRA layers')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='LoRA scaling factor alpha/rank')
    parser.add_argument('--use_extended_lora', action='store_true', default=False, help='Use extended LoRA target modules for UNet')
    parser.add_argument('--clip_ti_decay', action='store_true', default=True, help='Enable decay/regularization for TI embeddings') # 默认值与函数定义一致
    parser.add_argument('--learning_rate_unet', type=float, default=1e-4, help='Learning rate for UNet LoRA parameters')
    parser.add_argument('--learning_rate_text', type=float, default=1e-5, help='Learning rate for text encoder LoRA parameters')
    parser.add_argument('--learning_rate_ti', type=float, default=5e-4, help='Learning rate for Textual Inversion embeddings')
    parser.add_argument('--continue_inversion', action='store_true', default=False, # 原来是True，根据函数默认值设为False，按需开启
                        help='Continue training TI embeddings alongside LoRA tuning')
    parser.add_argument('--continue_inversion_lr', type=float, default=None, help='Specific LR for continued TI training (if different from initial TI LR)')
    parser.add_argument('--use_face_segmentation_condition', action='store_true', default=False, help='Use face segmentation condition (not fully integrated)')
    parser.add_argument('--cached_latents', action='store_true', default=True, help='Use pre-cached image latents if available in dataset') # 默认值与函数定义一致
    parser.add_argument('--use_mask_captioned_data', action='store_true', default=False, help='Use mask captioned data (not fully integrated)')
    parser.add_argument('--mask_temperature', type=float, default=1.0, help='Temperature for mask in masked loss (if applicable)')
    parser.add_argument('--scale_lr', action='store_true', default=False, help='Scale learning rate by batch size and grad accumulation steps')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler type for TI (e.g., linear, cosine)')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of warmup steps for TI LR scheduler')
    parser.add_argument('--lr_scheduler_lora', type=str, default='linear', help='Learning rate scheduler type for LoRA tuning')
    parser.add_argument('--lr_warmup_steps_lora', type=int, default=0, help='Number of warmup steps for LoRA LR scheduler')
    parser.add_argument('--weight_decay_ti', type=float, default=0.00, help='Weight decay for TI optimizer')
    parser.add_argument('--weight_decay_lora', type=float, default=0.001, help='Weight decay for LoRA optimizer')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False, help='Use 8-bit Adam optimizer (requires bitsandbytes)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0", "cpu")')
    parser.add_argument('--log_wandb', action='store_true', default=False, help='Log training metrics to Weights & Biases')
    parser.add_argument('--wandb_log_prompt_cnt', type=int, default=10, help='Number of prompts to log to W&B for image generation samples')
    parser.add_argument('--wandb_project_name', type=str, default='multi_lora_distill_project', help='W&B project name') # 与函数定义一致
    parser.add_argument('--wandb_entity', type=str, default='multi_lora_entity', help='W&B entity (username or team name)') # 与函数定义一致
    parser.add_argument('--proxy_token', type=str, default='person', help='Proxy token for class-specific prompts or regularization (if applicable)')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', default=False,
                       help='Enable xformers memory efficient attention in UNet (requires xformers installed)')
    parser.add_argument('--out_name', type=str, default='final_multi_lora', help='Base name for saved model files and logs') # 与函数定义一致

    # 新增的蒸馏特定参数
    parser.add_argument('--feature_align_weight', type=float, default=0.01,
                        help='Weight for feature alignment loss during LoRA tuning')
    parser.add_argument('--noise_pred_weight', type=float, default=1.0,
                        help='Weight for noise prediction (distillation) loss during LoRA tuning')

    # 可以添加一个 extra_args 的解析方式，比如接受一个JSON字符串
    parser.add_argument('--extra_args', type=str, default=None,
                        help='JSON string for extra arguments to be passed as a dictionary (e.g., \'{"mixed_precision": "fp16"}\')')


    args = parser.parse_args()

    # Convert argparse Namespace to dictionary
    args_dict = vars(args)

    # --- 处理需要特殊转换的参数 ---
    # 例如，将逗号分隔的字符串转换为集合
    if args_dict.get('lora_unet_target_modules'):
        args_dict['lora_unet_target_modules'] = set(args_dict['lora_unet_target_modules'].split(','))
    else: # 如果命令行没有提供，则使用函数定义的默认值
        args_dict['lora_unet_target_modules'] = {"CrossAttention", "Attention", "GEGLU"}


    if args_dict.get('lora_clip_target_modules'):
        args_dict['lora_clip_target_modules'] = set(args_dict['lora_clip_target_modules'].split(','))
    else: # 如果命令行没有提供，则使用函数定义的默认值
        args_dict['lora_clip_target_modules'] = {"CLIPSdpaAttention"}

    # 解析 extra_args (如果提供了)
    if args_dict.get('extra_args'):
        import json
        try:
            extra_args_dict = json.loads(args_dict['extra_args'])
            args_dict['extra_args'] = extra_args_dict # 替换为解析后的字典
        except json.JSONDecodeError:
            print(f"Warning: Could not parse --extra_args '{args_dict['extra_args']}' as JSON. Passing as None.")
            args_dict['extra_args'] = None
    else:
        # 确保 extra_args 至少是一个空字典，如果 train_multi_lora_distillation 期望它存在
        args_dict['extra_args'] = {} # 或者 None，取决于函数内部如何处理


    # 调用 train_multi_lora_distillation 函数
    # 确保所有在 train_multi_lora_distillation 函数签名中定义的参数都在 args_dict 中，
    # 或者函数本身有默认值。
    # 移除 lora_path1 和 lora_path2，因为现在使用 lora_folder_path
    # args_dict.pop('lora_path1', None)
    # args_dict.pop('lora_path2', None)

    # 确保 placeholder_tokens 相关参数如果train_multi_lora_distillation不再直接使用，则不传递
    # args_dict.pop('placeholder_tokens', None) # 由函数内部生成
    # args_dict.pop('placeholder_tokens1', None) # 不再使用
    # args_dict.pop('placeholder_tokens2', None) # 不再使用

    train_multi_lora_distillation(**args_dict)