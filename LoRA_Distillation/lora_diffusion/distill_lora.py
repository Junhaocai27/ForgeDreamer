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

sys.path.append('/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora')
from lora_diffusion import (
    PivotalTuningDatasetCapation,
    PivotalTuningDatasetCapationPromptOnly,
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


def inpainting_dataloader(
    train_dataset, train_batch_size, tokenizer, vae, text_encoder
):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask_values = [example["instance_masks"] for example in examples]
        masked_image_values = [
            example["instance_masked_images"] for example in examples
        ]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask_values += [example["class_masks"] for example in examples]
            masked_image_values += [
                example["class_masked_images"] for example in examples
            ]

        pixel_values = (
            torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        )
        mask_values = (
            torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        )
        masked_image_values = (
            torch.stack(masked_image_values)
            .to(memory_format=torch.contiguous_format)
            .float()
        )

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "masked_image_values": masked_image_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader

def loss_step_gaussian_noise(
    batch,
    teacher_unet,
    student_unet,
    student_text_encoder, # Student's text encoder
    teacher_text_encoder, # Teacher's text encoder (paired with teacher_unet)
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=False,
    # teacher_guidance_scale=1.0, # This seems unused in the provided snippet
    mask_temperature=1.0,
):
    weight_dtype = torch.float32 # 通常 latents 和 unet 的内部计算用 float32 或 float16 (if mixed_precision)
                                # text encoder embeds 通常是 float32

    latents = batch["pixel_values"].to(student_unet.device) # 应该已经是 float32 或根据混合精度设置
    if mixed_precision:
        latents = latents.to(dtype=torch.float16) # For UNet input if mixed_precision for UNet
    else:
        latents = latents.to(dtype=torch.float32)

    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    # ----------------------------------------------------------------------
    # 1. 获取 Student 模型的文本条件嵌入 (需要梯度)
    # ----------------------------------------------------------------------
    student_input_ids = batch["student_input_ids"].to(student_text_encoder.device)
    # 注意: 如果 student_text_encoder 也参与训练 (例如，通过LoRA)，那么它不应该在 no_grad() 内。
    # 如果 student_text_encoder 是固定的，并且只训练 student_unet，理论上可以 no_grad，
    # 但通常如果 text_encoder 是“student”的一部分，我们希望梯度能流过它。
    # 你的原始代码已经将它移出了 no_grad，这是正确的。

    # 获取 attention_mask (如果 DataLoader 提供了)
    student_attention_mask = batch.get("student_attention_mask")
    if student_attention_mask is not None:
        student_attention_mask = student_attention_mask.to(student_text_encoder.device)

    if mixed_precision: # 如果 student_text_encoder 本身也配置为混合精度
        with torch.cuda.amp.autocast(enabled=True): # Ensure autocast is enabled
            student_encoder_hidden_states = student_text_encoder(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask, # 传递 attention_mask
            )[0] # 通常是 [0] last_hidden_state, 有些模型是 .last_hidden_state
    else:
        student_encoder_hidden_states = student_text_encoder(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask, # 传递 attention_mask
        )[0]
    # student_encoder_hidden_states 通常是 float32，U-Net会根据情况转换


    # ----------------------------------------------------------------------
    # 2. 获取 Teacher 模型的文本条件嵌入 (不需要梯度)
    # ----------------------------------------------------------------------
    teacher_input_ids = batch["teacher1_input_ids"].to(teacher_text_encoder.device) # 或者 teacher2_input_ids
    # 获取 attention_mask (如果 DataLoader 提供了)
    teacher_attention_mask = batch.get("teacher1_attention_mask") # 或者 teacher2_attention_mask
    if teacher_attention_mask is not None:
        teacher_attention_mask = teacher_attention_mask.to(teacher_text_encoder.device)

    with torch.no_grad(): # Teacher 模型不计算梯度
        if mixed_precision: # 如果 teacher 模型也用混合精度 (通常 teacher 是评估模式，可能不强制 autocast)
                            # 但如果 teacher_unet 在 autocast 内，teacher_text_encoder 也应该保持一致性
            with torch.cuda.amp.autocast(enabled=True):
                teacher_encoder_hidden_states_no_grad = teacher_text_encoder(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask, # 传递 attention_mask
                )[0]
                # teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.detach() # .detach() in no_grad is redundant but harmless
        else:
            teacher_encoder_hidden_states_no_grad = teacher_text_encoder(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask, # 传递 attention_mask
            )[0]
            # teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.detach()

        # 确保 teacher embeds 的 dtype 与 teacher_unet 期望的一致
        # 通常 teacher_unet 会在内部处理，但如果 teacher_unet 使用 autocast, embeds 也应在 autocast 内生成
        # 如果 teacher_unet 输入 latents 是 float16, embeds 也最好是 float16
        if mixed_precision and hasattr(teacher_unet, 'dtype') and teacher_unet.dtype == torch.float16:
             teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.to(torch.float16)
        else:
             teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.to(torch.float32)


        # ----------------------------------------------------------------------
        # 3. 使用 Teacher 模型预测 (不需要梯度)
        # ----------------------------------------------------------------------
        # latents for teacher_unet, ensure correct dtype
        teacher_latents = latents.to(teacher_unet.device, dtype=teacher_unet.dtype if hasattr(teacher_unet, 'dtype') else weight_dtype)

        if mixed_precision:
            with torch.cuda.amp.autocast(enabled=True):
                teacher_pred = teacher_unet(
                    teacher_latents, timesteps, teacher_encoder_hidden_states_no_grad
                ).sample
        else:
            teacher_pred = teacher_unet(
                teacher_latents, timesteps, teacher_encoder_hidden_states_no_grad
            ).sample


    # ----------------------------------------------------------------------
    # 4. 使用 Student 模型预测 (需要梯度)
    # ----------------------------------------------------------------------
    # student_encoder_hidden_states 应该已经是正确的类型 (通常 float32)
    # student_unet 会根据其 mixed_precision 设置来处理输入类型
    # 如果 student_unet 使用 autocast, embeds 也最好是 float16
    if mixed_precision and hasattr(student_unet, 'dtype') and student_unet.dtype == torch.float16:
        student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(torch.float16)
    else:
        student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(torch.float32) # Or keep as is if already f32

    # latents for student_unet, ensure correct dtype (already handled at the beginning for student_unet)
    student_latents = latents.to(student_unet.device, dtype=student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype)


    if mixed_precision:
        with torch.cuda.amp.autocast(enabled=True):
            student_pred = student_unet(
                student_latents, timesteps, student_encoder_hidden_states_for_unet
            ).sample
    else:
        student_pred = student_unet(
            student_latents, timesteps, student_encoder_hidden_states_for_unet
        ).sample

    # ----------------------------------------------------------------------
    # 5. 应用 Mask (如果存在)
    # ----------------------------------------------------------------------
    if batch.get("mask", None) is not None:
        mask = batch["mask"].to(student_pred.device) # Mask should be on the same device
        # Ensure mask dtype is compatible with pred dtypes for multiplication
        # Reshape mask to be broadcastable: (bsz, 1, H_latent, W_latent)
        if mask.ndim == 3: # (bsz, H, W) -> (bsz, 1, H, W)
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4 or mask.shape[1] != 1:
             # Or handle error appropriately
            raise ValueError(f"Mask has unexpected shape: {mask.shape}. Expected (bsz, 1, H, W) or (bsz, H, W).")

        mask = (mask + 0.01).pow(mask_temperature) # Apply temperature
        mask = mask / mask.max() # Normalize mask

        # Ensure mask dtype matches predictions before multiplication
        # If predictions are float16, mask should also be float16 or broadcastable float32
        if mixed_precision:
            mask = mask.to(dtype=torch.float16)
        else:
            mask = mask.to(dtype=torch.float32)

        student_pred = student_pred * mask
        teacher_pred = teacher_pred * mask

    # ----------------------------------------------------------------------
    # 6. 计算损失 (MSE Loss)
    # ----------------------------------------------------------------------
    # Loss is usually computed in float32 for stability
    loss = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="none")
    loss = loss.mean([1, 2, 3]) # Mean over spatial dimensions and channels
    loss = loss.mean() # Mean over batch dimension

    return loss

def perform_tuning(
    teacher1_unet,
    teacher2_unet,
    teacher1_text_encoder,
    teacher2_text_encoder,
    student_unet,
    vae,
    student_text_encoder,
    dataloader1,
    dataloader2,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    lr_scheduler_lora,
    lora_unet_target_modules,
    lora_clip_target_modules,
    mask_temperature,
    out_name: str,
    tokenizer,
    # test_image_path: str,
    cached_latents: bool,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
):

    # Debug: Verify parameters are set for training
    trainable_params = {
        "unet": sum(p.requires_grad for p in student_unet.parameters()),
        "text_encoder": sum(p.requires_grad for p in student_text_encoder.parameters())
    }
    print(f"Trainable parameters: {trainable_params}")

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    weight_dtype = torch.float16

    student_unet.train()
    student_text_encoder.train()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    loss_sum = 0.0

    for epoch in range(math.ceil(num_steps / len(dataloader1))):
        for batch1, batch2 in zip(dataloader1, dataloader2):
            lr_scheduler_lora.step()

            optimizer.zero_grad()

            loss1 = loss_step_gaussian_noise(
                batch=batch1,
                teacher_unet=teacher1_unet,
                student_unet=student_unet,
                teacher_text_encoder=teacher1_text_encoder,
                student_text_encoder=student_text_encoder,
                scheduler=scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
            )
            
            loss2 = loss_step_gaussian_noise(
                batch=batch2,
                teacher_unet=teacher2_unet,
                student_unet=student_unet,
                teacher_text_encoder=teacher2_text_encoder,
                student_text_encoder=student_text_encoder,
                scheduler=scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
            )
            
            loss_sum = loss_sum + 0.5 * loss1.detach().item() + 0.5 * loss2.detach().item()
            # loss_sum = loss_sum.detach().item()
            loss = loss1 + loss2

            loss.backward()
            
            # Debug: Check if gradients are flowing to both models
            if global_step % 10 == 0:
                unet_grads = [param.grad.norm().item() for name, param in student_unet.named_parameters() 
                             if param.requires_grad and param.grad is not None]
                text_grads = [param.grad.norm().item() for name, param in student_text_encoder.named_parameters() 
                             if param.requires_grad and param.grad is not None]
                
                if len(unet_grads) > 0:
                    print(f"UNet gradient norm (avg): {sum(unet_grads)/len(unet_grads):.6f}")
                else:
                    print("WARNING: No gradients detected in UNet!")
                    
                if len(text_grads) > 0:
                    print(f"Text encoder gradient norm (avg): {sum(text_grads)/len(text_grads):.6f}")
                else:
                    print("WARNING: No gradients detected in text_encoder!")
            
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(student_unet.parameters(), student_text_encoder.parameters()), 1.0
            )
            optimizer.step()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            global_step += 1

            if global_step % save_steps == 0:
                save_all(
                    unet=student_unet,
                    text_encoder=student_text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_{global_step}.safetensors"
                    ),
                    target_replace_module_text=lora_clip_target_modules,
                    target_replace_module_unet=lora_unet_target_modules,
                )
                moved = (
                    torch.tensor(list(itertools.chain(*inspect_lora(student_unet).values())))
                    .mean()
                    .item()
                )

                print("LORA Unet Moved", moved)
                moved = (
                    torch.tensor(
                        list(itertools.chain(*inspect_lora(student_text_encoder).values()))
                    )
                    .mean()
                    .item()
                )

                print("LORA CLIP Moved", moved)

                # if log_wandb:
                #     with torch.no_grad():
                #         pipe = StableDiffusionPipeline(
                #             vae=vae,
                #             text_encoder=text_encoder,
                #             tokenizer=tokenizer,
                #             unet=student_unet,
                #             scheduler=scheduler,
                #             safety_checker=None,
                #             feature_extractor=None,
                #         )

                #         # open all images in test_image_path
                #         images = []
                #         for file in os.listdir(test_image_path):
                #             if file.endswith(".png") or file.endswith(".jpg"):
                #                 images.append(
                #                     Image.open(os.path.join(test_image_path, file))
                #                 )

                #         wandb.log({"loss": loss_sum / save_steps})
                #         loss_sum = 0.0
                #         wandb.log(
                #             evaluate_pipe(
                #                 pipe,
                #                 target_images=images,
                #                 class_token=class_token,
                #                 learnt_token="".join(placeholder_tokens),
                #                 n_test=wandb_log_prompt_cnt,
                #                 n_step=50,
                #                 clip_model_sets=preped_clip,
                #             )
                #         )

            if global_step >= num_steps:
                break

    save_all(
        student_unet,
        student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(save_path, f"{out_name}.safetensors"),
        target_replace_module_text=lora_clip_target_modules,
        target_replace_module_unet=lora_unet_target_modules,
    )


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
    model_id = "/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base"
    teacher1_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    teacher2_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
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

    train_dataset1 = PivotalTuningDatasetCapationPromptOnly(
        token_map=token_map1,
        use_template=use_template,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer1=teacher1_tokenizer,
        teacher_tokenizer2=teacher2_tokenizer,
    )
    
    # print("预览所有可能的文本:")
    # for i in range(len(train_dataset1)):
    #     item = train_dataset1[i]
    #     print(f"Text {i}: {item['raw_text']}")
    
    train_dataset2 = PivotalTuningDatasetCapationPromptOnly(
        token_map=token_map2,
        use_template=use_template,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer1=teacher1_tokenizer,
        teacher_tokenizer2=teacher2_tokenizer,
    )

    train_dataset1.blur_amount = 200
    train_dataset2.blur_amount = 200
    
    train_dataloader1 = text2img_dataloader_noise(
            train_dataset=train_dataset1,
            train_batch_size=train_batch_size,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer1=teacher1_tokenizer,
            teacher_tokenizer2=teacher2_tokenizer,
            cached_latents=cached_latents,
        )
    
    train_dataloader2 = text2img_dataloader_noise(
            train_dataset=train_dataset2,
            train_batch_size=train_batch_size,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer1=teacher1_tokenizer,
            teacher_tokenizer2=teacher2_tokenizer,
            cached_latents=cached_latents,
        )
    

    index_no_updates1 = torch.arange(len(student_tokenizer)) != -1
    index_no_updates2 = torch.arange(len(student_tokenizer)) != -1

    for tok_id in placeholder_token_ids1:
        index_no_updates1[tok_id] = False
        
    for tok_id in placeholder_token_ids2:
        index_no_updates2[tok_id] = False

    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.encoder.parameters(),
        student_text_encoder.text_model.final_layer_norm.parameters(),
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if cached_latents:
        student_vae = None

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

    train_dataset1.blur_amount = 70
    train_dataset2.blur_amount = 70

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )

    placeholder_token_ids = placeholder_token_ids1 + placeholder_token_ids2
    placeholder_tokens = placeholder_tokens1 + placeholder_tokens2

    perform_tuning(
        teacher1_unet=teacher1_unet,
        teacher2_unet=teacher2_unet,
        teacher1_text_encoder=teacher1_text_encoder,
        teacher2_text_encoder=teacher2_text_encoder,
        student_unet=student_unet,
        vae=student_vae,
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
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
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
