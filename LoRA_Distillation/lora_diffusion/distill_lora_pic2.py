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
from torch import nn
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

# ... (函数定义和前面的参数、变量初始化保持不变) ...
# def loss_step_gaussian_noise(
#     batch,
#     # teacher_unet, # 不再需要教师UNet
#     student_unet,
#     student_text_encoder,  # 学生模型的文本编码器
#     # teacher_text_encoder,  # 不再需要教师文本编码器
#     scheduler,
#     vae,                  # VAE 模型用于解码
#     global_step,          # 当前全局训练步数
#     save_image_every_n_steps=100,
#     output_dir="/root/lora_train/pic_train", # 确保路径有效或使用临时目录
#     t_mutliplier=1.0,
#     mixed_precision=False,
#     mask_temperature=1.0,
# ):
#     weight_dtype = torch.float32
#     if mixed_precision:
#         # VAE解码通常需要较高精度，即使其他部分是fp16
#         vae_decode_input_dtype = torch.float32 # VAE解码前的输入（缩放后）的数据类型
#     else:
#         vae_decode_input_dtype = torch.float32

#     # 真实潜变量 (x0)
#     latents_gt_x0 = batch["pixel_values"].to(device=student_unet.device, dtype=weight_dtype)
#     bsz = latents_gt_x0.shape[0]

#     # 为每张图片采样一个随机的时间步 t
#     timesteps = torch.randint(
#         0,
#         int(scheduler.config.num_train_timesteps * t_mutliplier), # 如果使用 t_multiplier，调整最大时间步
#         (bsz,),
#         device=latents_gt_x0.device,
#     )
#     timesteps = timesteps.long()

#     # 采样我们将添加到潜变量中并尝试预测的噪声 epsilon
#     noise = torch.randn_like(latents_gt_x0)

#     # 根据时间步向潜变量添加噪声
#     # noisy_latents 就是 x_t
#     noisy_latents = scheduler.add_noise(latents_gt_x0, noise, timesteps)

#     # 准备UNet的输入潜变量 (如果使用混合精度，可能转换为fp16)
#     if mixed_precision:
#         student_unet_input_latents = noisy_latents.to(dtype=torch.float16)
#     else:
#         student_unet_input_latents = noisy_latents.to(dtype=torch.float32) # 如果可能，应与 student_unet.dtype 匹配

#     # --- 学生模型部分 (文本嵌入) ---
#     # 学生模型使用 "aux1_input_ids" 或回退到 "input_ids"
#     student_input_ids = batch.get("aux1_input_ids")
#     if student_input_ids is None:
#         print("警告: student_text_encoder 尝试使用 aux1_input_ids，但未在批处理中找到。回退到 input_ids。")
#         student_input_ids = batch["input_ids"]
#     student_input_ids = student_input_ids.to(student_text_encoder.device)

#     student_attention_mask = batch.get("aux1_attention_mask")
#     if student_attention_mask is not None:
#         student_attention_mask = student_attention_mask.to(student_text_encoder.device)

#     # 确定学生文本编码器的输出数据类型以用于autocast
#     # 如果未指定或混合精度关闭，则默认为float32
#     student_text_encoder_output_dtype_for_autocast = torch.float32
#     if hasattr(student_text_encoder, 'dtype') and student_text_encoder.dtype == torch.float16:
#          student_text_encoder_output_dtype_for_autocast = torch.float16

#     if mixed_precision and student_text_encoder_output_dtype_for_autocast == torch.float16:
#         with torch.cuda.amp.autocast(enabled=True):
#             student_encoder_hidden_states = student_text_encoder(
#                 input_ids=student_input_ids, attention_mask=student_attention_mask
#             )[0]
#     else:
#         # 以全精度或编码器的本机精度执行
#         # 如果UNet启用了混合精度，则确保输出随后转换为UNet的适当类型
#         _temp_states = student_text_encoder(
#             input_ids=student_input_ids, attention_mask=student_attention_mask
#         )[0]
#         if hasattr(student_text_encoder, 'dtype'): # 像HuggingFace模型那样有 .dtype 属性
#             student_encoder_hidden_states = _temp_states.to(student_text_encoder.dtype)
#         else: # 如果模型没有 .dtype 属性，则回退
#             student_encoder_hidden_states = _temp_states.to(torch.float32)


#     # --- 学生模型UNet预测噪声 ---
#     # 确定学生UNet期望的输入数据类型
#     student_unet_internal_dtype = student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype

#     # 如果UNet启用了混合精度，确保文本嵌入与UNet期望的数据类型匹配
#     student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(dtype=student_unet_internal_dtype)
#     # student_unet_input_latents 已在上面准备好

#     if mixed_precision: # 这通常意味着UNet在fp16下运行
#         with torch.cuda.amp.autocast(enabled=True):
#             student_pred_noise = student_unet(
#                 student_unet_input_latents.to(student_unet_internal_dtype), # 确保潜变量与UNet数据类型匹配
#                 timesteps,
#                 student_encoder_hidden_states_for_unet
#             ).sample
#     else: # UNet在fp32或其本机精度下运行
#         student_pred_noise = student_unet(
#             student_unet_input_latents.to(student_unet_internal_dtype),
#             timesteps,
#             student_encoder_hidden_states_for_unet
#         ).sample

#     # --- Mask 和 Loss 计算 ---
#     # 损失的目标是我们添加的原始 'noise'
#     target_noise = noise

#     if batch.get("mask", None) is not None:
#         mask = batch["mask"].to(student_pred_noise.device)
#         if mask.ndim == 3:
#             mask = mask.unsqueeze(1) # 应该是 [B, 1, H, W]
#         elif mask.ndim != 4 or mask.shape[1] != 1:
#             raise ValueError(f"Mask 形状异常: {mask.shape}. 期望 [B, 1, H, W] 或 [B, H, W]")

#         # 应用掩码温度并归一化
#         mask = (mask + 0.01).pow(mask_temperature) # 加0.01以避免当pow<1时掩码值为0的问题
#         mask = mask / mask.max() # 将掩码归一化到 [0, 1] 范围

#         pred_dtype = student_pred_noise.dtype
#         mask = mask.to(dtype=pred_dtype)

#         # 在计算损失之前，将掩码应用于预测和目标
#         student_pred_noise = student_pred_noise * mask
#         target_noise = target_noise * mask


#     # loss = F.mse_loss(student_pred_noise.float(), target_noise.float(), reduction="mean")
#     # 或者，如果您想要像原来那样按样本计算损失然后求平均：
#     loss = F.mse_loss(student_pred_noise.float(), target_noise.float(), reduction="none").mean([1, 2, 3]).mean()


#     # --- 图像保存 ---
#     if global_step % save_image_every_n_steps == 0:
#         with torch.no_grad():
#             # 可视化批次中的第一项
#             latents_gt_x0_viz = latents_gt_x0[0:1].detach()
#             student_pred_noise_viz = student_pred_noise[0:1].detach() # 这是（可能被掩码的）预测噪声

#             # 如果应用了掩码，为了可视化，我们可能想看到未掩码的预测或使用原始噪声进行x0预测。
#             # 为简单起见，我们将使用（可能被掩码的）student_pred_noise_viz 来获取 pred_x0。
#             # scheduler.step 期望输入的是模型预测的噪声。

#             noisy_latents_viz = noisy_latents[0:1].detach() # 用于可视化的 x_t
#             timestep_viz = timesteps[0:1].detach()         # 用于可视化的 t

#             # 使用 scheduler 从 noisy_latents (xt) 和 predicted_noise (模型输出) 预测 x0
#             scheduler_output_student = scheduler.step(
#                 model_output=student_pred_noise_viz.to(dtype=noisy_latents_viz.dtype), # model_output 是预测的噪声
#                 timestep=timestep_viz.item(), # DDPMScheduler 期望单个整数
#                 sample=noisy_latents_viz      # 这是 x_t
#             )
#             pred_x0_latent_student = scheduler_output_student.pred_original_sample

#             # 准备 VAE 解码的潜变量
#             scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

#             pred_x0_latent_student_for_vae = pred_x0_latent_student.to(dtype=vae_decode_input_dtype) / scaling_factor
#             latent_x0_gt_viz_for_vae = latents_gt_x0_viz.to(dtype=vae_decode_input_dtype) / scaling_factor

#             # 使用 VAE 解码图像
#             # VAE 可能有其自身的精度要求（通常是fp32以保证稳定性）
#             vae_internal_dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float32

#             # 如果VAE对低精度敏感，则为其禁用autocast，或确保其以本机精度运行
#             with torch.cuda.amp.autocast(enabled=False): # 对VAE通常是好的做法
#                 pred_image_student = vae.decode(pred_x0_latent_student_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
#                 gt_image = vae.decode(latent_x0_gt_viz_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample

#             # 后处理图像以便保存
#             pred_image_student = (pred_image_student / 2 + 0.5).clamp(0, 1)
#             gt_image = (gt_image / 2 + 0.5).clamp(0, 1)

#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir, exist_ok=True)

#             t_val = timestep_viz.item()
#             save_image(pred_image_student, os.path.join(output_dir, f"step_{global_step}_student_pred_x0_t{t_val}.png"))
#             save_image(gt_image, os.path.join(output_dir, f"step_{global_step}_gt_x0_t{t_val}.png"))

#             # （可选）也可以保存带噪声的输入 (xt)
#             noisy_input_latent_for_vae = noisy_latents_viz.to(dtype=vae_decode_input_dtype) / scaling_factor # 注意这里是 noisy_latents_viz
#             with torch.cuda.amp.autocast(enabled=False):
#                 noisy_input_image = vae.decode(noisy_input_latent_for_vae.to(device=vae.device, dtype=vae_internal_dtype)).sample
#             noisy_input_image = (noisy_input_image / 2 + 0.5).clamp(0, 1)
#             save_image(noisy_input_image, os.path.join(output_dir, f"step_{global_step}_noisy_input_xt_t{t_val}.png"))

#     return loss

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

class CrossAttentionLossWeighting(nn.Module):
    """
    交叉注意力模块，用于动态调整两个损失之间的权重 (优化版)
    """
    def __init__(self, hidden_dim=768, num_heads=8, temp_init=1.0, learn_temp=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
            )
        self.head_dim = hidden_dim // num_heads
        
        self.loss_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        final_weight_linear = nn.Linear(hidden_dim, 2)
        torch.nn.init.zeros_(final_weight_linear.bias) # Initialize bias of the pre-softmax linear layer to zeros

        self.weight_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            final_weight_linear,
            nn.Softmax(dim=-1)
        )
        
        if learn_temp:
            self.temperature = nn.Parameter(torch.tensor(float(temp_init))) # Ensure float
        else:
            self.register_buffer('temperature', torch.tensor(float(temp_init)))
        
    def forward(self, loss1_detached, loss2_detached, step_ratio=0.0):
        """
        Args:
            loss1_detached: 第一个数据加载器的损失 (detached)
            loss2_detached: 第二个数据加载器的损失 (detached)
            step_ratio: 训练进度比例 (0-1)
        Returns:
            weights: [weight1, weight2] 权重张量
        """
        if loss1_detached.requires_grad or loss2_detached.requires_grad:
            # This is a safeguard. The caller should .detach() the losses.
            # If not detached, gradients from this module's loss_projection
            # could flow back to the original loss calculation, which is usually not intended.
            # print("Warning: CrossAttentionLossWeighting received losses that require grad. Consider detaching them.")
            pass

        if loss1_detached.dim() == 0:
            loss1_detached = loss1_detached.unsqueeze(0)
        if loss2_detached.dim() == 0:
            loss2_detached = loss2_detached.unsqueeze(0)
            
        loss1_feat = self.loss_projection(loss1_detached.unsqueeze(-1).float()) # Ensure float for linear layer
        loss2_feat = self.loss_projection(loss2_detached.unsqueeze(-1).float())
        
        current_device = loss1_feat.device
        step_encoding_val = torch.sin(torch.tensor(step_ratio * torch.pi, device=current_device, dtype=loss1_feat.dtype))
        
        loss1_feat = loss1_feat + step_encoding_val * 0.1
        loss2_feat = loss2_feat + step_encoding_val * 0.1
        
        inputs = torch.stack([loss1_feat.squeeze(0), loss2_feat.squeeze(0)], dim=0).unsqueeze(0)
        
        B, L, D = inputs.shape
        if not (B == 1 and L == 2 and D == self.hidden_dim):
             raise ValueError(f"Expected inputs shape [1, 2, {self.hidden_dim}], got {inputs.shape}")
        
        queries = self.query_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        current_temp = self.temperature
        if isinstance(self.temperature, nn.Parameter) and self.temperature.requires_grad: # Clamp if learnable
            current_temp = self.temperature.clamp(0.01, 10.0)

        attention_scores = attention_scores * current_temp
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, L, D)
        
        combined_features = torch.cat([attended_values[:, 0], attended_values[:, 1]], dim=-1)
        weights = self.weight_output(combined_features).squeeze(0) 
        
        return weights

class AdaptiveLossBalancer(nn.Module):
    """
    自适应损失平衡器，基于损失历史和梯度信息动态调整权重
    """
    def __init__(self, window_size=100):
        super().__init__()
        self.window_size = window_size
        self.loss1_history = []
        self.loss2_history = []
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta_param = nn.Parameter(torch.tensor(1.0)) # Renamed to avoid conflict
        
    def forward(self, loss1_detached, loss2_detached, grad_norm1=None, grad_norm2=None):
        """
        Args:
            loss1_detached: 第一个损失 (detached scalar)
            loss2_detached: 第二个损失 (detached scalar)
            grad_norm1: 第一个损失相关的梯度范数 (scalar or None)
            grad_norm2: 第二个损失相关的梯度范数 (scalar or None)
        """
        self.loss1_history.append(loss1_detached.item())
        self.loss2_history.append(loss2_detached.item())
        
        if len(self.loss1_history) > self.window_size:
            self.loss1_history.pop(0)
            self.loss2_history.pop(0)
        
        current_device = loss1_detached.device
        current_dtype = loss1_detached.dtype

        if len(self.loss1_history) >= 2:
            loss1_trend_val = self.loss1_history[-1] - self.loss1_history[-2]
            loss2_trend_val = self.loss2_history[-1] - self.loss2_history[-2]
            trend_factor = torch.tanh(torch.tensor(loss2_trend_val - loss1_trend_val, device=current_device, dtype=current_dtype))
        else:
            trend_factor = torch.tensor(0.0, device=current_device, dtype=current_dtype)
        
        loss_ratio = loss1_detached / (loss1_detached + loss2_detached + 1e-8)
        
        if grad_norm1 is not None and grad_norm2 is not None:
            grad_norm1_tensor = torch.as_tensor(grad_norm1, device=current_device, dtype=current_dtype)
            grad_norm2_tensor = torch.as_tensor(grad_norm2, device=current_device, dtype=current_dtype)
            # Handle cases where a grad norm might be zero to avoid NaN in grad_ratio
            if (grad_norm1_tensor + grad_norm2_tensor).item() < 1e-8 : # if both are effectively zero
                grad_ratio = torch.tensor(0.5, device=current_device, dtype=current_dtype) # Assign neutral ratio
            else:
                grad_ratio = grad_norm1_tensor / (grad_norm1_tensor + grad_norm2_tensor + 1e-8)
            balance_factor = 0.7 * loss_ratio + 0.3 * grad_ratio
        else:
            balance_factor = loss_ratio
        
        weight1 = torch.sigmoid(self.alpha + self.beta_param * trend_factor + balance_factor)
        weight2 = 1.0 - weight1
        
        return torch.stack([weight1, weight2])

def train_inversion(
    # --- "Teacher" 模型参数 (保留API，但当前loss_step不直接使用) ---
    teacher1_unet, teacher2_unet, teacher1_text_encoder, teacher2_text_encoder,
    # --- "Student" 模型参数 ---
    student_unet, vae, student_text_encoder,
    # --- Dataloader 参数 ---
    dataloader1, dataloader2,
    # --- 核心训练参数 ---
    num_steps: int, scheduler, index_no_updates, optimizer, save_steps: int,
    placeholder_token_ids, placeholder_tokens, save_path: str, out_name: str, 
    # --- 学习率调度器 ---
    lr_scheduler_main,
    # --- LoRA 相关参数 (TI 中未使用) ---
    lora_unet_target_modules=None, lora_clip_target_modules=None,
    # --- 输出和日志相关 ---
    tokenizer=None, test_image_path: str = None, cached_latents: bool = False,
    # --- 损失函数特定参数 ---
    mask_temperature: float = 1.0, t_multiplier_loss: float = 1.0, save_image_every_n_steps_loss: int = 200,
    # --- 其他有默认值的参数 ---
    accum_iter: int = 1, log_wandb: bool = False, wandb_log_prompt_cnt: int = 10,
    class_token: str = "person", mixed_precision: bool = False, clip_ti_decay: bool = True,
    tensorboard_log_dir: str = "runs",
    # --- 动态权重相关参数 ---
    use_cross_attention_weighting: bool = True, ca_temp_init: float = 1.0, ca_learn_temp: bool = True,
    use_adaptive_balancing: bool = True,
    ca_lr: float = 1e-5, # Adjusted LR for weighting modules
    ab_lr: float = 1e-5, # Adjusted LR
):
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_inversion")
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging for inversion to: {tb_log_path}")

    loss_step_image_output_dir = os.path.join(save_path, out_name, "loss_step_debug_images_inversion")
    os.makedirs(loss_step_image_output_dir, exist_ok=True)
    print(f"Debug images from loss_step_gaussian_noise will be saved to: {loss_step_image_output_dir}")

    device = next(student_unet.parameters()).device

    cross_attention_weighter, ca_optimizer = None, None
    if use_cross_attention_weighting:
        cross_attention_weighter = CrossAttentionLossWeighting(
            temp_init=ca_temp_init, learn_temp=ca_learn_temp
        ).to(device)
        cross_attention_weighter.train()
        ca_optimizer = torch.optim.Adam(cross_attention_weighter.parameters(), lr=ca_lr)
    
    adaptive_balancer, ab_optimizer = None, None
    if use_adaptive_balancing:
        adaptive_balancer = AdaptiveLossBalancer().to(device)
        adaptive_balancer.train()
        ab_optimizer = torch.optim.Adam(adaptive_balancer.parameters(), lr=ab_lr)

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps (Inversion w/ Dynamic Loss Weighting)")
    global_step = 0

    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()
    index_updates = ~torch.tensor(index_no_updates, dtype=torch.bool, device=device) if not isinstance(index_no_updates, torch.Tensor) else ~index_no_updates.to(device)
    
    grad_norm_for_ab = None # For AdaptiveBalancer using previous step's gradient on embeddings

    iter_dataloader1 = iter(dataloader1)
    iter_dataloader2 = iter(dataloader2)

    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # Zero gradients at the beginning of accumulation cycle or each step if no accumulation
        if (global_step % accum_iter) == 0:
            optimizer.zero_grad(set_to_none=True) # More memory efficient
            if ca_optimizer: ca_optimizer.zero_grad(set_to_none=True)
            if ab_optimizer: ab_optimizer.zero_grad(set_to_none=True)
        
        try: batch1 = next(iter_dataloader1)
        except StopIteration: iter_dataloader1 = iter(dataloader1); batch1 = next(iter_dataloader1)
        try: batch2 = next(iter_dataloader2)
        except StopIteration: iter_dataloader2 = iter(dataloader2); batch2 = next(iter_dataloader2)
        
        # Batch to device logic (assuming batch is a dict of tensors)
        # batch1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch1.items()}
        # batch2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch2.items()}


        lr_scheduler_main.step()
            
        loss1_raw = loss_step_gaussian_noise(
            batch=batch1, student_unet=student_unet, student_text_encoder=student_text_encoder,
            scheduler=scheduler, vae=vae, global_step=global_step,
            save_image_every_n_steps=save_image_every_n_steps_loss,
            output_dir_for_loss_step=loss_step_image_output_dir,
            t_mutliplier=t_multiplier_loss, mixed_precision=mixed_precision,
            mask_temperature=mask_temperature
        )
        loss2_raw = loss_step_gaussian_noise(
            batch=batch2, student_unet=student_unet, student_text_encoder=student_text_encoder,
            scheduler=scheduler, vae=vae, global_step=global_step,
            save_image_every_n_steps=save_image_every_n_steps_loss,
            output_dir_for_loss_step=loss_step_image_output_dir,
            t_mutliplier=t_multiplier_loss, mixed_precision=mixed_precision,
            mask_temperature=mask_temperature
        )

        step_ratio = global_step / num_steps
        
        ca_weights_val = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        if use_cross_attention_weighting and cross_attention_weighter:
            ca_weights_val = cross_attention_weighter(loss1_raw.detach(), loss2_raw.detach(), step_ratio)
            writer.add_scalar('Weights_Inv/CA_L1', ca_weights_val[0].item(), global_step)
            writer.add_scalar('Weights_Inv/CA_L2', ca_weights_val[1].item(), global_step)
            if hasattr(cross_attention_weighter, 'temperature'):
                 writer.add_scalar('Weights_Inv/CA_Temp', cross_attention_weighter.temperature.item(), global_step)

        ab_weights_val = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        if use_adaptive_balancing and adaptive_balancer:
            ab_weights_val = adaptive_balancer(loss1_raw.detach(), loss2_raw.detach(), grad_norm_for_ab, grad_norm_for_ab)
            writer.add_scalar('Weights_Inv/AB_L1', ab_weights_val[0].item(), global_step)
            writer.add_scalar('Weights_Inv/AB_L2', ab_weights_val[1].item(), global_step)
        
        if use_cross_attention_weighting and use_adaptive_balancing:
            final_weights = 0.6 * ca_weights_val + 0.4 * ab_weights_val
            final_weights = final_weights / final_weights.sum()
        elif use_cross_attention_weighting: final_weights = ca_weights_val
        elif use_adaptive_balancing: final_weights = ab_weights_val
        else: final_weights = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        
        current_step_loss = final_weights[0] * loss1_raw + final_weights[1] * loss2_raw
        loss_for_backward = current_step_loss / accum_iter
        loss_for_backward.backward()

        # Store grad norm for next step's AdaptiveLossBalancer (after backward, before step)
        if use_adaptive_balancing:
            text_embed_grad = student_text_encoder.get_input_embeddings().weight.grad
            if text_embed_grad is not None:
                grad_on_updated_embeds = text_embed_grad[index_updates, :]
                if grad_on_updated_embeds.numel() > 0:
                    grad_norm_for_ab = grad_on_updated_embeds.norm().item()
                else: grad_norm_for_ab = None
            else: grad_norm_for_ab = None

        writer.add_scalar('Weights_Inv/Final_L1', final_weights[0].item(), global_step)
        writer.add_scalar('Weights_Inv/Final_L2', final_weights[1].item(), global_step)

        if (global_step + 1) % accum_iter == 0:
            text_embed_grad_main = student_text_encoder.get_input_embeddings().weight.grad
            if text_embed_grad_main is not None:
                grad_norm_main = text_embed_grad_main[index_updates, :].norm(dim=-1).mean()
                writer.add_scalar('Gradients_Inv/Emb_Norm', grad_norm_main.item(), global_step)
            
            optimizer.step()
            if ca_optimizer: ca_optimizer.step()
            if ab_optimizer: ab_optimizer.step()
            
            # Zero gradients again after stepping if not handled at the beginning of the next accumulation cycle
            # optimizer.zero_grad(set_to_none=True) 
            # if ca_optimizer: ca_optimizer.zero_grad(set_to_none=True)
            # if ab_optimizer: ab_optimizer.zero_grad(set_to_none=True)
            # No, zeroing at the start of the loop / accumulation cycle is better.

            with torch.no_grad(): # TI specific embedding updates
                if clip_ti_decay:
                    idx_update_tensor = index_updates.squeeze() # Ensure it's 1D for indexing
                    pre_norm = student_text_encoder.get_input_embeddings().weight[idx_update_tensor, :].norm(dim=-1, keepdim=True)
                    lambda_ = min(1.0, 100 * lr_scheduler_main.get_last_lr()[0])
                    student_text_encoder.get_input_embeddings().weight[idx_update_tensor, :] = F.normalize(
                        student_text_encoder.get_input_embeddings().weight[idx_update_tensor, :], dim=-1
                    ) * (pre_norm + lambda_ * (0.4 - pre_norm))
                current_norm_val = student_text_encoder.get_input_embeddings().weight[index_updates.squeeze(), :].norm(dim=-1).mean().item()
                writer.add_scalar('Embeddings_Inv/Norm', current_norm_val, global_step)
                student_text_encoder.get_input_embeddings().weight[~index_updates.squeeze()] = orig_embeds_params[~index_updates.squeeze()]
        
        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0] # 获取当前学习率
        logs = {
            "loss": f"{current_step_loss.item():.4f}", 
            "l1": f"{loss1_raw.item():.4f}", 
            "l2": f"{loss2_raw.item():.4f}",
            "w1": f"{final_weights[0].item():.2f}", 
            "w2": f"{final_weights[1].item():.2f}", # <--- 添加 w2
            "lr": f"{current_lr_val:.2e}", # 使用获取到的学习率
        }
        progress_bar.set_postfix(**logs)

        writer.add_scalar('Loss_Inv/Combined', current_step_loss.item(), global_step)
        writer.add_scalar('Loss_Inv/L1_raw', loss1_raw.item(), global_step)
        writer.add_scalar('Loss_Inv/L2_raw', loss2_raw.item(), global_step)
        writer.add_scalar('LearningRate_Inv/Main', lr_scheduler_main.get_last_lr()[0], global_step)

        if global_step % save_steps == 0 and global_step > 0:
            current_save_dir = os.path.join(save_path, out_name)
            os.makedirs(current_save_dir, exist_ok=True)
            save_all(
                unet=student_unet, text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
                save_path=os.path.join(current_save_dir, f"step_inv_{global_step}.safetensors"),
                save_lora=False,
            )
            if cross_attention_weighter: torch.save(cross_attention_weighter.state_dict(), os.path.join(current_save_dir, f"caw_inv_{global_step}.pt"))
            if adaptive_balancer: torch.save(adaptive_balancer.state_dict(), os.path.join(current_save_dir, f"ab_inv_{global_step}.pt"))
            print(f"Saved inversion checkpoint at step {global_step} to {current_save_dir}")

        if global_step >= num_steps: break

    writer.close()
    progress_bar.close()
    print("Inversion training finished.")
    final_save_dir = os.path.join(save_path, out_name)
    os.makedirs(final_save_dir, exist_ok=True)
    save_all(
        unet=student_unet, text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(final_save_dir, f"final_inv_{global_step}.safetensors"),
        save_lora=False,
    )
    if cross_attention_weighter: torch.save(cross_attention_weighter.state_dict(), os.path.join(final_save_dir, "final_caw_inv.pt"))
    if adaptive_balancer: torch.save(adaptive_balancer.state_dict(), os.path.join(final_save_dir, "final_ab_inv.pt"))
    print(f"Final inversion model and balancers saved to {final_save_dir}")


def perform_tuning(
    # --- Teacher models (Placeholders for API consistency if loss_step doesn't use them) ---
    teacher1_unet, teacher2_unet, teacher1_text_encoder, teacher2_text_encoder,
    # --- Student models ---
    student_unet, vae, student_text_encoder,
    # --- DataLoaders ---
    dataloader1, dataloader2,
    # --- Core training parameters ---
    num_steps: int, 
    scheduler, # Noise scheduler (e.g., DDIMScheduler)
    optimizer, # Main optimizer for LoRA params
    save_steps: int,
    placeholder_token_ids, # May not be strictly needed for general LoRA, but pass to save_all
    placeholder_tokens,    # May not be strictly needed for general LoRA, but pass to save_all
    save_path: str, # Base path for saving checkpoints and final model
    # --- Learning rate scheduler for LoRA ---
    lr_scheduler_lora,
    # --- LoRA specific parameters (passed to save_all) ---
    lora_unet_target_modules,
    lora_clip_target_modules,
    out_name: str,
    # --- Loss parameters ---
    mask_temperature: float = 1.0,
    # --- Output and logging ---
    # Name for this run, used in save paths and TensorBoard
    tokenizer=None, # Optional tokenizer
    cached_latents: bool = False, # If dataloaders provide latents directly
    log_wandb: bool = False, 
    tensorboard_log_dir: str = "runs",
    # --- Dynamic loss weighting parameters ---
    use_cross_attention_weighting: bool = True, 
    ca_temp_init: float = 1.0, 
    ca_learn_temp: bool = True,
    use_adaptive_balancing: bool = True,
    ca_lr: float = 1e-5, # Learning rate for CrossAttentionLossWeighting optimizer
    ab_lr: float = 1e-5, # Learning rate for AdaptiveLossBalancer optimizer
    gradient_clip_val: float = 1.0, # Max norm for gradient clipping
    # --- Parameters for loss_step_gaussian_noise from original perform_tuning ---
    save_image_every_n_steps_loss: int = 200, 
    t_multiplier_loss: float = 0.8,          
    mixed_precision: bool = True, # For mixed precision training if loss_step supports it
):
    device = next(student_unet.parameters()).device
    trainable_params_unet = sum(p.numel() for p in student_unet.parameters() if p.requires_grad)
    trainable_params_text_encoder = sum(p.numel() for p in student_text_encoder.parameters() if p.requires_grad)
    print(f"Trainable UNet LoRA parameters: {trainable_params_unet}")
    print(f"Trainable Text Encoder LoRA parameters: {trainable_params_text_encoder}")

    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_tuning")
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging for tuning to: {tb_log_path}")
    
    loss_step_image_output_dir = os.path.join(save_path, out_name, "loss_step_debug_images_tuning")
    os.makedirs(loss_step_image_output_dir, exist_ok=True)

    cross_attention_weighter, ca_optimizer = None, None
    if use_cross_attention_weighting:
        cross_attention_weighter = CrossAttentionLossWeighting(
            temp_init=ca_temp_init, learn_temp=ca_learn_temp
        ).to(device)
        cross_attention_weighter.train()
        ca_optimizer = torch.optim.Adam(cross_attention_weighter.parameters(), lr=ca_lr)
    
    adaptive_balancer, ab_optimizer = None, None
    if use_adaptive_balancing:
        adaptive_balancer = AdaptiveLossBalancer().to(device)
        adaptive_balancer.train()
        ab_optimizer = torch.optim.Adam(adaptive_balancer.parameters(), lr=ab_lr)

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps (LoRA Tuning w/ Dynamic Weighting & save_all)")
    global_step = 0

    student_unet.train()
    student_text_encoder.train()

    iter_dataloader1 = iter(dataloader1)
    iter_dataloader2 = iter(dataloader2)

    grad_norm_unet_ab, grad_norm_text_enc_ab = None, None 

    if log_wandb:
        try:
            import wandb
            wandb.init(project="lora-tuning-dynamic-saveall", name=out_name, reinit=True)
        except ImportError: print("Wandb not installed, disabling wandb logging."); log_wandb = False

    for step_idx in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        if ca_optimizer: ca_optimizer.zero_grad(set_to_none=True)
        if ab_optimizer: ab_optimizer.zero_grad(set_to_none=True)
        
        try:
            # Simplified dataloader iteration
            if global_step % len(dataloader1) == 0 and global_step > 0 : iter_dataloader1 = iter(dataloader1)
            batch1 = next(iter_dataloader1)
            if global_step % len(dataloader2) == 0 and global_step > 0 : iter_dataloader2 = iter(dataloader2)
            batch2 = next(iter_dataloader2)
        except StopIteration: # Should ideally not happen if num_steps is managed well with dataloader sizes
            print("Warning: Dataloader exhausted unexpectedly. Re-initializing.")
            iter_dataloader1 = iter(dataloader1); batch1 = next(iter_dataloader1)
            iter_dataloader2 = iter(dataloader2); batch2 = next(iter_dataloader2)
        
        # Batch to device logic would go here if not handled by dataloader
        # batch1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch1.items()}
        # batch2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch2.items()}

        lr_scheduler_lora.step()
            
        loss1 = loss_step_gaussian_noise(
            batch=batch1, student_unet=student_unet, student_text_encoder=student_text_encoder,
            vae=vae, global_step=global_step, scheduler=scheduler,
            output_dir_for_loss_step=loss_step_image_output_dir,
            save_image_every_n_steps=save_image_every_n_steps_loss,
            t_mutliplier=t_multiplier_loss, mixed_precision=mixed_precision,
            mask_temperature=mask_temperature
        )
        loss2 = loss_step_gaussian_noise(
            batch=batch2, student_unet=student_unet, student_text_encoder=student_text_encoder,
            vae=vae, global_step=global_step, scheduler=scheduler,
            output_dir_for_loss_step=loss_step_image_output_dir,
            save_image_every_n_steps=save_image_every_n_steps_loss,
            t_mutliplier=t_multiplier_loss, mixed_precision=mixed_precision,
            mask_temperature=mask_temperature
        )

        step_ratio = global_step / num_steps
        
        ca_weights_val = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        if use_cross_attention_weighting and cross_attention_weighter:
            ca_weights_val = cross_attention_weighter(loss1.detach(), loss2.detach(), step_ratio)
            writer.add_scalar('Weights_Tune/CA_L1', ca_weights_val[0].item(), global_step)
            writer.add_scalar('Weights_Tune/CA_L2', ca_weights_val[1].item(), global_step)
            if hasattr(cross_attention_weighter, 'temperature'):
                writer.add_scalar('Weights_Tune/CA_Temp', cross_attention_weighter.temperature.item(), global_step)

        ab_weights_val = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        if use_adaptive_balancing and adaptive_balancer:
            ab_weights_val = adaptive_balancer(loss1.detach(), loss2.detach(), grad_norm_unet_ab, grad_norm_text_enc_ab)
            writer.add_scalar('Weights_Tune/AB_L1', ab_weights_val[0].item(), global_step)
            writer.add_scalar('Weights_Tune/AB_L2', ab_weights_val[1].item(), global_step)
        
        if use_cross_attention_weighting and use_adaptive_balancing:
            final_weights = 0.6 * ca_weights_val + 0.4 * ab_weights_val
            final_weights = final_weights / final_weights.sum()
        elif use_cross_attention_weighting: final_weights = ca_weights_val
        elif use_adaptive_balancing: final_weights = ab_weights_val
        else: final_weights = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)
        
        current_loss = final_weights[0] * loss1 + final_weights[1] * loss2
        current_loss.backward()

        if use_adaptive_balancing:
            try:
                unet_grads = [p.grad.norm().item() for p in student_unet.parameters() if p.grad is not None and p.requires_grad]
                grad_norm_unet_ab = sum(unet_grads) / len(unet_grads) if unet_grads else None
                text_enc_grads = [p.grad.norm().item() for p in student_text_encoder.parameters() if p.grad is not None and p.requires_grad]
                grad_norm_text_enc_ab = sum(text_enc_grads) / len(text_enc_grads) if text_enc_grads else None
            except Exception as e: print(f"Warn: Grad norm calculation for AB failed: {e}"); grad_norm_unet_ab=None; grad_norm_text_enc_ab=None

        params_to_clip = itertools.chain(filter(lambda p: p.requires_grad, student_unet.parameters()),
                                         filter(lambda p: p.requires_grad, student_text_encoder.parameters()))
        torch.nn.utils.clip_grad_norm_(params_to_clip, gradient_clip_val)

        optimizer.step()
        if ca_optimizer: ca_optimizer.step()
        if ab_optimizer: ab_optimizer.step()

        progress_bar.update(1)
        current_lr_val_lora = optimizer.param_groups[0]['lr']
        logs_pb = {
            "loss": f"{current_loss.item():.4f}", "l1": f"{loss1.item():.4f}", "l2": f"{loss2.item():.4f}",
            "w1": f"{final_weights[0].item():.2f}", "w2": f"{final_weights[1].item():.2f}",
            "lr": f"{current_lr_val_lora:.2e}"}
        progress_bar.set_postfix(logs_pb)

        writer.add_scalar('Loss_Tune/Combined', current_loss.item(), global_step)
        writer.add_scalar('Loss_Tune/L1_raw', loss1.item(), global_step)
        writer.add_scalar('Loss_Tune/L2_raw', loss2.item(), global_step)
        writer.add_scalar('Weights_Tune/Final_L1', final_weights[0].item(), global_step)
        writer.add_scalar('Weights_Tune/Final_L2', final_weights[1].item(), global_step)
        writer.add_scalar('LearningRate_Tune/LoRA', current_lr_val_lora, global_step)
        if grad_norm_unet_ab is not None: writer.add_scalar('Gradients_Tune/UNet_AvgNorm_AB', grad_norm_unet_ab, global_step)
        if grad_norm_text_enc_ab is not None: writer.add_scalar('Gradients_Tune/TextEnc_AvgNorm_AB', grad_norm_text_enc_ab, global_step)

        if log_wandb and global_step % 20 == 0: 
            wandb_logs = {k.replace("_Tune",""):v for k,v in logs_pb.items()} 
            wandb_logs.update({
                'step': global_step,
                'weights/final1': final_weights[0].item(), 'weights/final2': final_weights[1].item(),
            })
            if use_cross_attention_weighting: wandb_logs.update({'weights/ca1': ca_weights_val[0].item(), 'weights/ca2': ca_weights_val[1].item()})
            if use_adaptive_balancing: wandb_logs.update({'weights/ab1': ab_weights_val[0].item(), 'weights/ab2': ab_weights_val[1].item()})
            wandb.log(wandb_logs)

        if global_step > 0 and global_step % save_steps == 0:
            # Path for save_all, mimicking your original structure (files directly under out_name)
            save_all_path = os.path.join(save_path, out_name, f"step_tune_{global_step}.safetensors")
            save_all_dir = os.path.dirname(save_all_path) # Directory for auxiliary files
            os.makedirs(save_all_dir, exist_ok=True)

            print(f"\nSaving tuning checkpoint at step {global_step} using save_all to {save_all_path}...")
            save_all(
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=save_all_path,
                save_lora=True, 
                target_replace_module_text=lora_clip_target_modules,
                target_replace_module_unet=lora_unet_target_modules,
            )
            
            # Save auxiliary modules in the same directory as the save_all output
            if cross_attention_weighter: 
                torch.save(cross_attention_weighter.state_dict(), 
                           os.path.join(save_all_dir, f"caw_tune_step_{global_step}.pt"))
            if adaptive_balancer: 
                torch.save(adaptive_balancer.state_dict(), 
                           os.path.join(save_all_dir, f"ab_tune_step_{global_step}.pt"))
            
            print(f"Tuning checkpoint (via save_all) and auxiliary modules saved for step {global_step}")

        global_step += 1
        if global_step >= num_steps: break

    writer.close()
    progress_bar.close()
    
    # --- 最终模型保存 ---
    print(f"\nTraining completed! Saving final tuning model using save_all...")
    final_model_save_dir_base = os.path.join(save_path, out_name) 
    os.makedirs(final_model_save_dir_base, exist_ok=True)

    final_model_path_save_all = os.path.join(final_model_save_dir_base, f"final_tune_{global_step}.safetensors")
    save_all(
        unet=student_unet,
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path_save_all,
        save_lora=True,
        target_replace_module_text=lora_clip_target_modules,
        target_replace_module_unet=lora_unet_target_modules,
    )
    
    if cross_attention_weighter: 
        torch.save(cross_attention_weighter.state_dict(), 
                   os.path.join(final_model_save_dir_base, "final_caw_tune.pt"))
    if adaptive_balancer: 
        torch.save(adaptive_balancer.state_dict(), 
                   os.path.join(final_model_save_dir_base, "final_ab_tune.pt"))
    
    print(f"Final tuning model (via save_all) and auxiliary balancers saved in {final_model_save_dir_base}")

    if log_wandb: wandb.finish()
    print("LoRA tuning finished successfully!")


def train(
    lora_path1: str,
    lora_path2: str,
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
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
    out_name: str = "final_lora",
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    # print(placeholder_tokens, initializer_tokens)
    if len(placeholder_tokens1) == 0:
        placeholder_tokens1 = []
        print("PTI : Placeholder Tokens not given, using null token")
    elif len(placeholder_tokens2) == 0:
        placeholder_tokens2 = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens1 = placeholder_tokens1.split("|")
        placeholder_tokens2 = placeholder_tokens2.split("|")

        assert (
            sorted(placeholder_tokens1) == placeholder_tokens1
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"
        
        assert (
            sorted(placeholder_tokens2) == placeholder_tokens2
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens1 = ["<rand-0.017>"] * len(placeholder_tokens1)
        initializer_tokens2 = ["<rand-0.017>"] * len(placeholder_tokens2)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens1) == len(
        placeholder_tokens1
    ), "Unequal Initializer token for Placeholder tokens."
    
    assert len(initializer_tokens2) == len(
        placeholder_tokens2
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens1)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map1 = {"DUMMY": "".join(placeholder_tokens1)}
        token_map2 = {"DUMMY": "".join(placeholder_tokens2)}

    print("PTI : Placeholder Tokens", placeholder_tokens1)
    print("PTI : Placeholder Tokens", placeholder_tokens2)
    print("PTI : Initializer Tokens", initializer_tokens1)
    print("PTI : Initializer Tokens", initializer_tokens2)

    # get the models
    student_text_encoder, student_vae, student_unet, student_tokenizer, placeholder_token_ids1, placeholder_token_ids2 = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens1,
        placeholder_tokens2,
        initializer_tokens1,
        initializer_tokens2,
        device=device,
    )
    
    # get the student's models
    # student_text_encoder = text_encoder
    # student_vae = vae
    # student_unet = unet
    # student_tokenizer = tokenizer
    # student_placeholder_token_ids1 = placeholder_token_ids1
    # student_placeholder_token_ids2 = placeholder_token_ids2
    
    
    # get the teacher's models
    model_id = pretrained_model_name_or_path
    teacher1_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    teacher2_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    teacher1_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(teacher1_pipe.scheduler.config)
    teacher2_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(teacher2_pipe.scheduler.config)
    
    patch_pipe(
        teacher1_pipe,
        lora_path1,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    
    tune_lora_scale(teacher1_pipe.unet, 1.0)
    tune_lora_scale(teacher1_pipe.text_encoder, 1.0)
    
    patch_pipe(
        teacher2_pipe,
        lora_path2,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    
    tune_lora_scale(teacher2_pipe.unet, 1.0)
    tune_lora_scale(teacher2_pipe.text_encoder, 1.0)
    
    teacher1_text_encoder = teacher1_pipe.text_encoder
    teacher1_vae = teacher1_pipe.vae
    teacher1_unet = teacher1_pipe.unet
    teacher1_tokenizer = teacher1_pipe.tokenizer
    # teacher1_placeholder_token_ids = placeholder_token_ids
    
    teacher2_text_encoder = teacher2_pipe.text_encoder
    teacher2_vae = teacher2_pipe.vae
    teacher2_unet = teacher2_pipe.unet
    teacher2_tokenizer = teacher2_pipe.tokenizer
    # teacher2_placeholder_token_ids = placeholder_token_ids

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
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti
        
    lora_dataset1 = PivotalTuningDatasetCapationLoraGenerated(
        # ... (所有其他参数与之前相同) ...
        sd_pipeline=teacher1_pipe,
        device=device,
        # ...
        main_tokenizer=teacher1_tokenizer,
        use_template=use_template,
        token_map=token_map1,
        dataset_size=10, # 例如生成20张
        transform_size=512,
        h_flip=True,
        aux_tokenizer1=student_tokenizer,

        # 新增参数以保存图片
        save_generated_images_path="/root/lora_train/pic",
        save_image_prefix="lora_cat_1" # 图片文件名的前缀
    )
    
    lora_dataset2 = PivotalTuningDatasetCapationLoraGenerated(
        # ... (所有其他参数与之前相同) ...
        sd_pipeline=teacher2_pipe,
        device=device,
        # ...
        main_tokenizer=teacher2_tokenizer,
        use_template=use_template,
        token_map=token_map2,
        dataset_size=10, # 例如生成20张
        transform_size=512,
        h_flip=True,
        aux_tokenizer1=student_tokenizer,

        # 新增参数以保存图片
        save_generated_images_path="/root/lora_train/pic",
        save_image_prefix="lora_cat_2" # 图片文件名的前缀
    )
    
    train_dataloader1 = text2img_dataloader_combined_with_latent_caching(
        train_dataset=lora_dataset1,
        train_batch_size=train_batch_size,
        main_tokenizer=teacher1_tokenizer,    # 必须与 lora_dataset 初始化时用的 main_tokenizer 相同
        aux_tokenizer1=student_tokenizer,
        vae=teacher1_vae,
        cached_latents=cached_latents
    )
    
    train_dataloader2 = text2img_dataloader_combined_with_latent_caching(
        train_dataset=lora_dataset2,
        train_batch_size=train_batch_size,
        main_tokenizer=teacher2_tokenizer,    # 必须与 lora_dataset 初始化时用的 main_tokenizer 相同
        aux_tokenizer1=student_tokenizer,
        vae=teacher2_vae,
        cached_latents=cached_latents
    )

    # index_no_updates1 = torch.arange(len(student_tokenizer)) != -1
    # index_no_updates2 = torch.arange(len(student_tokenizer)) != -1

    # for tok_id in placeholder_token_ids1:
    #     index_no_updates1[tok_id] = False
        
    # for tok_id in placeholder_token_ids2:
    #     index_no_updates2[tok_id] = False

    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.encoder.parameters(),
        student_text_encoder.text_model.final_layer_norm.parameters(),
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # if cached_latents:
    #     student_vae = None
        
    placeholder_token_ids = placeholder_token_ids1 + placeholder_token_ids2
    placeholder_tokens = placeholder_tokens1 + placeholder_tokens2

    index_no_updates = torch.ones(len(student_tokenizer), dtype=torch.bool, device=student_text_encoder.device)

    # 遍历合并后的所有占位符 token ID，将它们标记为可更新 (False)
    for tok_id in placeholder_token_ids: # 使用合并后的 ID 列表
        index_no_updates[tok_id] = False

    # STEP 1 : Perform Inversion
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

        train_inversion(
            teacher1_unet=teacher1_unet,
            teacher2_unet=teacher2_unet,
            teacher1_text_encoder=teacher1_text_encoder,
            teacher2_text_encoder=teacher2_text_encoder,
            num_steps=max_train_steps_tuning,
            lora_unet_target_modules=lora_unet_target_modules,
            lora_clip_target_modules=lora_clip_target_modules,
            out_name=out_name,
            student_unet=student_unet,
            vae=student_vae,
            student_text_encoder=student_text_encoder,
            dataloader1=train_dataloader1,
            dataloader2=train_dataloader2,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler_main=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            # train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=student_tokenizer,
            clip_ti_decay=clip_ti_decay,
        )

        del ti_optimizer

    # Next perform Tuning with LoRA:
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("PTI : USING EXTENDED UNET!!!")
        lora_unet_target_modules = (
            lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        )
        print("PTI : Will replace modules: ", lora_unet_target_modules)

        unet_lora_params, _ = inject_trainable_lora_extended(
            student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )
    print(f"PTI : has {len(unet_lora_params)} lora")

    print("PTI : Before training:")
    inspect_lora(student_unet)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    student_text_encoder.requires_grad_(False)

    if continue_inversion:
        params_to_optimize += [
            {
                "params": student_text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr
                if continue_inversion_lr is not None
                else ti_lr,
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
        
    print(params_to_optimize)

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

    perform_tuning(
        teacher1_unet=teacher1_unet,
        teacher2_unet=teacher2_unet,
        teacher1_text_encoder=teacher1_text_encoder,
        teacher2_text_encoder=teacher2_text_encoder,
        student_unet=student_unet,
        vae=teacher1_vae,
        student_text_encoder=student_text_encoder,
        dataloader1=train_dataloader1,
        dataloader2=train_dataloader2,
        num_steps=max_train_steps_tuning,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=student_tokenizer,
        out_name=out_name,
        # test_image_path=instance_data_dir,
        log_wandb=log_wandb,
        # wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=class_token,
        train_inpainting=train_inpainting,
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Training script for LoRA model')
    
    # Required arguments
    parser.add_argument('--lora_path1', type=str, required=True, help='Path to first LoRA model')
    parser.add_argument('--lora_path2', type=str, required=True, help='Path to second LoRA model')
    parser.add_argument('--instance_data_dir', type=str, required=False, default="", help='Directory containing instance data')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving models')
    
    # Optional arguments (with defaults matching the train function)
    parser.add_argument('--train_text_encoder', action='store_true', default=True, help='Whether to train text encoder')
    parser.add_argument('--pretrained_vae_name_or_path', type=str, default=None, help='Path to pretrained VAE')
    parser.add_argument('--revision', type=str, default=None, help='Revision of pretrained model')
    parser.add_argument('--perform_inversion', action='store_true', default=True, help='Perform inversion')
    parser.add_argument('--use_template', type=str, choices=[None, 'object', 'style'], default=None, help='Template to use')
    parser.add_argument('--train_inpainting', action='store_true', default=False, help='Train for inpainting')
    parser.add_argument('--placeholder_tokens', type=str, default='', help='Placeholder tokens')
    parser.add_argument('--placeholder_tokens1', type=str, default='', help='First set of placeholder tokens')
    parser.add_argument('--placeholder_tokens2', type=str, default='', help='Second set of placeholder tokens')
    parser.add_argument('--placeholder_token_at_data', type=str, default=None, help='Placeholder token at data')
    parser.add_argument('--initializer_tokens', type=str, default=None, help='Initializer tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution for training')
    parser.add_argument('--color_jitter', action='store_true', default=True, help='Apply color jitter')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--sample_batch_size', type=int, default=1, help='Sampling batch size')
    parser.add_argument('--max_train_steps_tuning', type=int, default=1000, help='Maximum number of tuning steps')
    parser.add_argument('--max_train_steps_ti', type=int, default=1000, help='Maximum number of text inversion steps')
    parser.add_argument('--save_steps', type=int, default=100, help='Steps between saving checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Enable gradient checkpointing')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    # Adding all missing arguments from the error message
    parser.add_argument('--lora_unet_target_modules', type=str, default={"CrossAttention","Attention","GEGLU"}, 
                       help='Target modules for UNet LoRA (comma-separated)')
    parser.add_argument('--lora_clip_target_modules', type=str, default={"CLIPSdpaAttention"}, 
                       help='Target modules for CLIP LoRA (comma-separated)')
    parser.add_argument('--lora_dropout_p', type=float, default=0.0, help='LoRA dropout probability')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='LoRA scale factor')
    parser.add_argument('--use_extended_lora', action='store_true', default=False, help='Use extended LoRA')
    parser.add_argument('--clip_ti_decay', action='store_true', default=True, help='Enable CLIP TI decay')
    parser.add_argument('--learning_rate_unet', type=float, default=1e-4, help='Learning rate for UNet')
    parser.add_argument('--learning_rate_text', type=float, default=1e-5, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_ti', type=float, default=5e-4, help='Learning rate for text inversion')
    parser.add_argument('--continue_inversion', action='store_true', default=True, help='Continue inversion')
    parser.add_argument('--continue_inversion_lr', type=float, default=None, help='Learning rate for continued inversion')
    parser.add_argument('--use_face_segmentation_condition', action='store_true', default=False, help='Use face segmentation condition')
    parser.add_argument('--cached_latents', action='store_true', default=True, help='Use cached latents')
    parser.add_argument('--use_mask_captioned_data', action='store_true', default=False, help='Use mask captioned data')
    parser.add_argument('--mask_temperature', type=float, default=1.0, help='Mask temperature')
    parser.add_argument('--scale_lr', action='store_true', default=False, help='Scale learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Learning rate warmup steps')
    parser.add_argument('--lr_scheduler_lora', type=str, default='linear', help='LoRA learning rate scheduler')
    parser.add_argument('--lr_warmup_steps_lora', type=int, default=0, help='LoRA learning rate warmup steps')
    parser.add_argument('--weight_decay_ti', type=float, default=0.00, help='Weight decay for text inversion')
    parser.add_argument('--weight_decay_lora', type=float, default=0.001, help='Weight decay for LoRA')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False, help='Use 8-bit Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--log_wandb', action='store_true', default=False, help='Log to Weights & Biases')
    parser.add_argument('--wandb_log_prompt_cnt', type=int, default=10, help='Number of prompts to log to W&B')
    parser.add_argument('--wandb_project_name', type=str, default='new_pti_project', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='new_pti_entity', help='W&B entity')
    parser.add_argument('--proxy_token', type=str, default='person', help='Proxy token')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', default=False, 
                       help='Enable xformers memory efficient attention')
    parser.add_argument('--out_name', type=str, default='final_lora', help='Name for the output model')
    
    args = parser.parse_args()
    
    # Convert argparse Namespace to dictionary
    args_dict = vars(args)
    
    # Convert string arguments to appropriate data types
    # Convert comma-separated strings to sets for target modules
    # if 'lora_unet_target_modules' in args_dict:
    #     args_dict['lora_unet_target_modules'] = set(args_dict['lora_unet_target_modules'].split(','))
    
    # if 'lora_clip_target_modules' in args_dict:
    #     args_dict['lora_clip_target_modules'] = set(args_dict['lora_clip_target_modules'].split(','))
    
    # # Make instance_data_dir optional
    # if args_dict.get('instance_data_dir') is None:
    #     args_dict['instance_data_dir'] = ""
    
    # Call the train function with the parsed arguments
    train(**args_dict)