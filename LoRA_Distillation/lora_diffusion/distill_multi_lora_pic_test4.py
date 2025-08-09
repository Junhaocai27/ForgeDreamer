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
from feature_hook_unet import FeatureAlignmentLoss, UNetFeatureExtractor, create_enhanced_feature_alignment_loss
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

class DynamicTeacherModelManager:
    """动态教师模型管理器"""
    
    def __init__(self, teacher_configs: List[dict]):
        self.teacher_configs = teacher_configs
        self.current_teacher_idx = -1
        self.current_teacher_model = None
        self.device = teacher_configs[0]["device"] if teacher_configs else "cuda:0"
        
    def load_teacher(self, teacher_idx: int):
        """加载指定的teacher模型"""
        if teacher_idx == self.current_teacher_idx and self.current_teacher_model is not None:
            return self.current_teacher_model
        
        # 卸载当前模型
        self.unload_current_teacher()
        
        # 加载新模型
        config = self.teacher_configs[teacher_idx]
        print(f"🔄 动态加载教师模型 {teacher_idx+1}: {config['lora_path']}")
        
        # 创建pipeline
        teacher_pipe = StableDiffusionPipeline.from_pretrained(
            config["pretrained_model_path"], 
            torch_dtype=torch.float16
        ).to(self.device)
        
        teacher_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            teacher_pipe.scheduler.config
        )
        
        # 加载LoRA权重
        patch_pipe(
            teacher_pipe,
            config["lora_path"],
            patch_text=True,
            patch_ti=True,
            patch_unet=True,
        )
        
        tune_lora_scale(teacher_pipe.unet, 1.0)
        tune_lora_scale(teacher_pipe.text_encoder, 1.0)
        
        self.current_teacher_model = {
            "unet": teacher_pipe.unet,
            "text_encoder": teacher_pipe.text_encoder,
            "vae": teacher_pipe.vae,
            "tokenizer": teacher_pipe.tokenizer,
            "pipeline": teacher_pipe
        }
        self.current_teacher_idx = teacher_idx
        
        print(f"✓ 教师模型 {teacher_idx+1} 动态加载完成")
        return self.current_teacher_model
    
    def unload_current_teacher(self):
        """卸载当前teacher模型"""
        if self.current_teacher_model is not None:
            # 清理GPU内存
            del self.current_teacher_model
            self.current_teacher_model = None
            self.current_teacher_idx = -1
            
            # 强制垃圾回收
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
    def get_teacher_count(self):
        """获取teacher数量"""
        return len(self.teacher_configs)

    def cleanup(self):
        """清理所有资源"""
        self.unload_current_teacher()
    

def create_multiple_teacher_models_dynamic(teacher_lora_paths: List[str], pretrained_model_name_or_path: str, device: str):
    """
    创建多个教师模型 - 动态加载版本，节省显存
    """
    teacher_configs = []
    
    for i, lora_path in enumerate(teacher_lora_paths):
        print(f"准备教师模型配置 {i+1}/{len(teacher_lora_paths)}: {lora_path}")
        
        # 只保存配置信息，不实际加载模型
        teacher_configs.append({
            "lora_path": lora_path,
            "pretrained_model_path": pretrained_model_name_or_path,
            "device": device,
            "index": i
        })
        
        print(f"✓ 教师模型 {i+1} 配置已准备")
    
    return teacher_configs

def create_multiple_dataloaders_optimized(
    teacher_manager: DynamicTeacherModelManager,
    teacher_info: List[dict], 
    student_tokenizer,
    train_batch_size: int,
    cached_latents: bool,
    use_template,
    device: str
):
    """
    优化的数据加载器创建，减少显存占用
    """
    dataloaders = []
    token_maps = []
    
    for i, info in enumerate(teacher_info):
        model_name = info["name"]
        
        # 为每个模型创建token_map
        token_map = {"DUMMY": f"<{clean_filename_for_token(model_name)}>"}
        token_maps.append(token_map)
        
        # 临时加载teacher模型获取tokenizer
        teacher_model = teacher_manager.load_teacher(i)
        
        # 创建数据集
        dataset = PivotalTuningDatasetCapationLoraGenerated(
            sd_pipeline=teacher_model["pipeline"],
            device=device,
            main_tokenizer=teacher_model["tokenizer"],
            use_template=use_template,
            token_map=token_map,
            dataset_size=10,
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
        
        # 立即卸载以节省显存
        teacher_manager.unload_current_teacher()
    
    return dataloaders, token_maps

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
    save_image_every_n_steps=101,
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

def train_inversion_with_multi_feature_alignment_dynamic(
    # --- "Teacher" 模型参数 (修改为动态管理器) ---
    teacher_manager: DynamicTeacherModelManager,  # 替换原来的teacher列表
    # --- "Student" 模型参数 ---
    student_unet,
    vae,
    student_text_encoder,
    # --- Dataloader 参数 (修改为列表形式) ---
    dataloaders: List,
    # --- 核心训练参数 ---
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    lr_scheduler_main,
    lora_unet_target_modules, 
    lora_clip_target_modules, 
    out_name: str,
    tokenizer,
    test_image_path: str,
    cached_latents: bool,
    mask_temperature: float = 1.0,
    t_multiplier_loss: float = 1.0,
    save_image_every_n_steps_loss: int = 200,
    unet_feature_align_weight: float = 0.01,
    unet_feature_alignment_layers=[
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ],
    text_encoder_feature_align_weight: float = 0.1,
    text_encoder_alignment_layers=[
        'text_model.encoder.layers.0',
        'text_model.encoder.layers.2',
        'text_model.encoder.layers.4',
        'text_model.encoder.layers.6',
        'text_model.encoder.layers.8',
        'text_model.encoder.layers.10',
        'text_model.encoder.layers.11'
    ],
    text_encoder_pooling_strategy: str = "mean",
    text_encoder_loss_type: str = "mse",
    noise_pred_weight: float = 1.0,
    unet_return_dict: bool = True,
    accum_iter: int = 1,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
    mixed_precision: str = "no",
    clip_ti_decay: bool = True,
    tensorboard_log_dir: str = "runs_new",
    teacher_selection_strategy: str = "round_robin",
    teacher_weights: Optional[List[float]] = None,
):
    """
    支持动态teacher加载的文本逆向训练
    """
    
    num_teachers = teacher_manager.get_teacher_count()
    print(f"训练使用 {num_teachers} 个教师模型（动态加载模式）")
    
    use_mixed_precision = (mixed_precision != "no")

    # --- TensorBoard 设置 ---
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_inversion_multi_teacher_dual_feat_align_dynamic")
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard 日志 (动态多teacher双重特征对齐) 将保存到: {tb_log_path}")

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

    # 创建学生UNet特征提取器
    student_unet_extractor = UNetFeatureExtractor(
        target_layers=unet_alignment_layers_for_loss,
        mixed_precision_config=mixed_precision
    )
    
    # 初始化dataloader迭代器
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0

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
    progress_bar.set_description("训练步数 (动态多teacher文本逆向 + 双重特征对齐)")
    global_step = 0

    # 备份原始的、非占位符的词元嵌入
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        if wandb.run is None:
            wandb.init(
                project=f"textual_inversion_project_{out_name}", 
                name=f"{out_name}_inversion_dynamic_multi_teacher_dual_feat_align_run", 
                reinit=True
            )

    index_updates = ~index_no_updates

    # 用于区间日志记录的累积损失
    accumulated_total_loss = 0.0
    accumulated_ti_loss = 0.0
    accumulated_unet_feature_loss = 0.0
    accumulated_text_encoder_feature_loss = 0.0
    step_count_in_interval = 0

    # === 为每个teacher模型的损失统计 ===
    teacher_loss_trackers = []
    for i in range(num_teachers):
        teacher_loss_trackers.append({
            'ti_loss': 0.0,
            'unet_feature_loss': 0.0,
            'text_encoder_feature_loss': 0.0,
            'total_loss': 0.0,
            'step_count': 0,
        })

    # 从模型确定设备
    unet_device = next(student_unet.parameters()).device
    text_encoder_device = next(student_text_encoder.parameters()).device
    vae_device = vae.device

    # 显存优化函数
    def optimize_memory_usage():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            print(f"💾 显存使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已缓存")

    # 主训练循环
    for step_idx in range(num_steps):
        student_unet.eval()
        student_text_encoder.train()

        # --- 选择当前教师模型 ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # 动态加载当前需要的teacher模型
        current_teacher_model = teacher_manager.load_teacher(current_teacher_idx)
        current_teacher_unet = current_teacher_model["unet"]
        current_teacher_text_encoder = current_teacher_model["text_encoder"]
        
        # 为当前teacher创建UNet特征提取器
        current_teacher_unet_extractor = UNetFeatureExtractor(
            target_layers=unet_alignment_layers_for_loss,
            mixed_precision_config=mixed_precision
        )
        
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
                use_grad=True
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
                current_teacher_name=f"teacher_{current_teacher_idx+1}",
                current_teacher_idx=current_teacher_idx,
                save_comparison_grid=True,
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

            # 6. 总损失
            current_step_total_loss = (
                noise_pred_weight * ti_loss_current_step +
                text_encoder_feature_align_weight * text_encoder_feature_align_loss_val
            )
            
            loss_for_backward = current_step_total_loss / accum_iter
            loss_for_backward.backward()

            # === 记录当前teacher的损失到追踪器 ===
            current_ti_loss_item = ti_loss_current_step.detach().item()
            current_unet_feature_loss_item = unet_feature_align_loss_val.detach().item() if torch.is_tensor(unet_feature_align_loss_val) else float(unet_feature_align_loss_val)
            current_text_encoder_feature_loss_item = text_encoder_feature_align_loss_val.detach().item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val)
            current_total_loss_item = current_step_total_loss.detach().item()

            # 更新当前teacher的损失追踪器
            teacher_loss_trackers[current_teacher_idx]['ti_loss'] += current_ti_loss_item
            teacher_loss_trackers[current_teacher_idx]['unet_feature_loss'] += current_unet_feature_loss_item
            teacher_loss_trackers[current_teacher_idx]['text_encoder_feature_loss'] += current_text_encoder_feature_loss_item
            teacher_loss_trackers[current_teacher_idx]['total_loss'] += current_total_loss_item
            teacher_loss_trackers[current_teacher_idx]['step_count'] += 1

            # 记录全局累积损失
            accumulated_total_loss += current_total_loss_item
            accumulated_ti_loss += current_ti_loss_item
            accumulated_unet_feature_loss += current_unet_feature_loss_item
            accumulated_text_encoder_feature_loss += current_text_encoder_feature_loss_item
            step_count_in_interval += 1

        # --- 优化器步骤和嵌入正则化 ---
        if (global_step + 1) % accum_iter == 0:
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

                # 恢复未被更新的原始嵌入
                embed_weights.data[index_no_updates] = orig_embeds_params[index_no_updates]
        
        lr_scheduler_main.step()
        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0]
        
        log_dict_postfix = {
            "总损失": f"{current_step_total_loss.item():.4f}",
            "TI损失": f"{ti_loss_current_step.item():.4f}",
            "Text特征": f"{text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val):.4f}",
            "LR": f"{current_lr_val:.2e}",
            "教师": teacher_name_log
        }
        progress_bar.set_postfix(**log_dict_postfix)

        # TensorBoard记录（简化版本）
        if writer and global_step % 10 == 0:
            writer.add_scalar('损失_动态多teacher/总损失_单步', current_step_total_loss.item(), global_step)
            writer.add_scalar('损失_动态多teacher/TI损失_单步', ti_loss_current_step.item(), global_step)
            writer.add_scalar('损失_动态多teacher/TextEncoder特征对齐损失_单步', text_encoder_feature_align_loss_val.item() if torch.is_tensor(text_encoder_feature_align_loss_val) else float(text_encoder_feature_align_loss_val), global_step)
            writer.add_scalar('教师模型_动态多teacher/当前选择', current_teacher_idx + 1, global_step)

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
                save_path=os.path.join(current_save_dir, f"step_inv_dynamic_multi_teacher_{global_step}.safetensors"),
                save_lora=False,
            )
            print(f"\n ✓ 动态多teacher检查点已保存: step_inv_dynamic_multi_teacher_{global_step}.safetensors")

            # 显存优化：定期清理
            if global_step % (save_steps * 2) == 0:
                teacher_manager.unload_current_teacher()
                optimize_memory_usage()

            # 重置累积损失
            accumulated_total_loss = 0.0
            accumulated_ti_loss = 0.0
            accumulated_unet_feature_loss = 0.0
            accumulated_text_encoder_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    
    # 最终清理
    teacher_manager.cleanup()
    
    print(f"\n动态多teacher文本逆向训练完成!")
    writer.close()

# def perform_tuning_multi_teacher(
#     # --- "Teacher" 模型参数 (修改为列表形式) ---
#     teacher_unets: List,  # 从 teacher1_unet, teacher2_unet 改为列表
#     teacher_text_encoders: List,  # 从 teacher1_text_encoder, teacher2_text_encoder 改为列表
#     # --- "Student" 模型参数 ---
#     student_unet,
#     vae,
#     student_text_encoder,
#     # --- Dataloader 参数 (修改为列表形式) ---
#     dataloaders: List,  # 从 dataloader1, dataloader2 改为列表
#     # --- 核心训练参数 ---
#     num_steps,
#     cached_latents,
#     scheduler,
#     optimizer,
#     save_steps,
#     placeholder_tokens,
#     placeholder_token_ids,
#     save_path,
#     lr_scheduler_lora,
#     lora_unet_target_modules,
#     lora_clip_target_modules,
#     mask_temperature,
#     tokenizer,
#     out_name,
#     # --- 可选参数 ---
#     mixed_precision="no",
#     log_wandb=False,
#     wandb_log_prompt_cnt=10,
#     class_token="person",
#     train_inpainting=False,
#     feature_align_weight=0.005,
#     noise_pred_weight=1.0,
#     feature_alignment_unet_layers=[
#         'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
#         'mid_block',
#         'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
#     ],
#     unet_return_dict=True,
#     tensorboard_log_dir: str = "runs_new",
#     # --- 新增多teacher参数 ---
#     teacher_selection_strategy: str = "round_robin",  # 新增：teacher选择策略
#     teacher_weights: Optional[List[float]] = None,  # 新增：teacher权重
# ):
#     """
#     执行支持多teacher的LoRA微调，包含特征对齐
#     新增详细的TensorBoard日志记录功能，包括每个teacher的损失分析
    
#     Args:
#         teacher_unets: List of teacher UNet models
#         teacher_text_encoders: List of teacher text encoders
#         dataloaders: List of dataloaders corresponding to each teacher
#         teacher_selection_strategy: "round_robin", "weighted_random", "adaptive"
#         teacher_weights: Weights for each teacher (used with weighted_random)
#     """
    
#     # 验证输入参数
#     assert len(teacher_unets) == len(teacher_text_encoders) == len(dataloaders), \
#         "teacher_unets, teacher_text_encoders, and dataloaders must have the same length"
    
#     num_teachers = len(teacher_unets)
#     print(f"LoRA调优使用 {num_teachers} 个教师模型")
    
#     import os
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm
#     from torch.utils.tensorboard import SummaryWriter
    
#     print(f"=" * 80)
#     print(f"Starting Multi-Teacher LoRA Tuning with Feature Alignment")
#     print(f"Number of teachers: {num_teachers}")
#     print(f"Total steps: {num_steps}")
#     print(f"Save steps: {save_steps}")
#     print(f"Teacher selection strategy: {teacher_selection_strategy}")
#     print(f"Feature alignment weight: {feature_align_weight}")
#     print(f"Noise prediction weight: {noise_pred_weight}")
#     print(f"Mixed precision: {mixed_precision}")
#     print(f"Output name: {out_name}")
#     print(f"=" * 80)

#     # --- 增强的TensorBoard设置 ---
#     tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_multi_teacher_lora_tuning")
#     if not os.path.exists(tb_log_path):
#         os.makedirs(tb_log_path, exist_ok=True)
#     writer = SummaryWriter(log_dir=tb_log_path)
#     print(f"TensorBoard 日志 (多teacher LoRA调优) 将保存到: {tb_log_path}")

#     # 特征对齐损失函数组件
#     alignment_layers_for_loss = feature_alignment_unet_layers
#     layer_weights_for_loss = {
#         'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
#         'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
#     }
#     layer_weights_for_loss = {k: v for k, v in layer_weights_for_loss.items() if k in alignment_layers_for_loss}

#     feature_alignment_loss_fn = FeatureAlignmentLoss(
#         alignment_layers=alignment_layers_for_loss,
#         loss_weights=layer_weights_for_loss,
#         loss_type="mse"
#     )

#     # 为每个teacher创建UNet特征提取器
#     teacher_extractors = []
#     for i, teacher_unet in enumerate(teacher_unets):
#         extractor = UNetFeatureExtractor(
#             target_layers=feature_alignment_unet_layers,
#             mixed_precision_config=mixed_precision
#         )
#         teacher_extractors.append(extractor)
    
#     # 创建学生UNet特征提取器
#     student_extractor = UNetFeatureExtractor(
#         target_layers=feature_alignment_unet_layers,
#         mixed_precision_config=mixed_precision
#     )

#     progress_bar = tqdm(range(num_steps), desc="Multi-Teacher LoRA Tuning with Feature Alignment")
#     global_step = 0

#     # 初始化dataloader迭代器
#     dataloader_iters = [iter(dl) for dl in dataloaders]
#     teacher_selection_index = 0  # 用于round_robin策略

#     # === 新增：为每个teacher模型的损失统计 ===
#     teacher_loss_trackers = []
#     for i in range(num_teachers):
#         teacher_loss_trackers.append({
#             'noise_pred_loss': 0.0,
#             'feature_align_loss': 0.0,
#             'total_loss': 0.0,
#             'step_count': 0,
#         })

#     # 用于区间日志记录的累积损失
#     accumulated_total_loss = 0.0
#     accumulated_noise_loss = 0.0
#     accumulated_feature_loss = 0.0
#     step_count_in_interval = 0

#     student_unet.train()
#     student_text_encoder.train()

#     for step_idx in range(num_steps):
#         # --- 选择当前教师模型 ---
#         current_teacher_idx = select_current_teacher(
#             global_step=global_step,
#             num_teachers=num_teachers,
#             strategy=teacher_selection_strategy,
#             weights=teacher_weights,
#             selection_index=teacher_selection_index
#         )
        
#         # 获取当前选择的模型和数据
#         current_teacher_unet = teacher_unets[current_teacher_idx]
#         current_teacher_text_encoder = teacher_text_encoders[current_teacher_idx]
#         current_teacher_extractor = teacher_extractors[current_teacher_idx]
#         current_dataloader_iter = dataloader_iters[current_teacher_idx]
#         current_dataloader_obj = dataloaders[current_teacher_idx]
#         teacher_name_log = f"T{current_teacher_idx + 1}"
        
#         # 更新round_robin索引
#         if teacher_selection_strategy == "round_robin":
#             teacher_selection_index = (teacher_selection_index + 1) % num_teachers

#         # 获取批次数据
#         try:
#             batch = next(current_dataloader_iter)
#         except StopIteration:
#             print(f"Info: dataloader for teacher {current_teacher_idx + 1} exhausted at step {global_step}. Re-initializing.")
#             dataloader_iters[current_teacher_idx] = iter(current_dataloader_obj)
#             batch = next(dataloader_iters[current_teacher_idx])

#         optimizer.zero_grad()

#         # --- Latent 处理 ---
#         main_module = student_unet.module if hasattr(student_unet, 'module') else student_unet
#         expected_latents_dtype = main_module.dtype if hasattr(main_module, 'dtype') else torch.float32

#         if cached_latents:
#             latents = batch["pixel_values"].to(device=main_module.device, dtype=expected_latents_dtype)
#         else:
#             vae_device = vae.device
#             input_pixels_for_vae = batch["pixel_values"].to(device=vae_device, dtype=torch.float32)
#             with torch.no_grad():
#                 latents = vae.encode(input_pixels_for_vae).latent_dist.sample() * vae.config.scaling_factor
#             latents = latents.to(device=main_module.device, dtype=expected_latents_dtype)

#         # --- 文本编码 ---
#         text_encoder_device = student_text_encoder.device if hasattr(student_text_encoder, 'device') else main_module.device
#         input_ids = batch["input_ids"].to(device=text_encoder_device)
#         student_input_ids = batch.get('aux1_input_ids', input_ids).to(device=text_encoder_device)

#         teacher_unet_internal_dtype = next(current_teacher_unet.parameters()).dtype
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
#                 teacher_raw_hidden_states = current_teacher_text_encoder(
#                     input_ids, output_hidden_states=True
#                 ).hidden_states[-1]
#                 teacher_encoder_hidden_states = teacher_raw_hidden_states.to(dtype=teacher_unet_internal_dtype)

#         student_unet_internal_dtype = next(student_unet.parameters()).dtype
#         with torch.cuda.amp.autocast(enabled=(mixed_precision != "no" and text_encoder_device.type == 'cuda')):
#             student_raw_hidden_states = student_text_encoder(
#                 student_input_ids, output_hidden_states=True
#             ).hidden_states[-1]
#             student_encoder_hidden_states = student_raw_hidden_states.to(dtype=student_unet_internal_dtype)

#         # --- 噪声和时间步 ---
#         noise = torch.randn_like(latents)
#         bsz = latents.shape[0]
#         timesteps = torch.randint(
#             0, scheduler.config.num_train_timesteps,
#             (bsz,), device=latents.device
#         ).long()
#         noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

#         # --- 教师模型前向传播 ---
#         with torch.no_grad():
#             teacher_noise_pred, teacher_features = current_teacher_extractor.extract_features(
#                 unet_model=current_teacher_unet,
#                 sample=noisy_latents,
#                 timestep=timesteps,
#                 encoder_hidden_states=teacher_encoder_hidden_states,
#                 return_dict=unet_return_dict,
#                 use_grad=False
#             )

#         # --- 学生模型前向传播（仅提取特征用于对齐损失）---
#         _, student_features = student_extractor.extract_features(
#             unet_model=student_unet,
#             sample=noisy_latents,
#             timestep=timesteps,
#             encoder_hidden_states=student_encoder_hidden_states,
#             return_dict=unet_return_dict,
#             use_grad=True
#         )

#         # --- 计算损失 ---
#         # 噪声预测损失：使用原有的 loss_step_gaussian_noise 函数
#         noise_pred_loss = loss_step_gaussian_noise(
#             batch=batch,
#             student_unet=student_unet,
#             student_text_encoder=student_text_encoder,
#             vae=vae,
#             global_step=global_step,
#             scheduler=scheduler,
#             t_mutliplier=0.8,
#             mixed_precision=(mixed_precision != "no"),
#             mask_temperature=mask_temperature,
#             # 新增参数：传递teacher信息用于保存
#             current_teacher_name=f"teacher_{current_teacher_idx+1}",
#             current_teacher_idx=current_teacher_idx,
#             save_comparison_grid=True,
#         )

#         # 特征对齐损失：学生模型特征 vs 教师模型特征
#         feature_align_loss, feature_loss_dict = feature_alignment_loss_fn(
#             teacher_features, student_features
#         )

#         tuning_weight_adjuster = create_tuning_weight_adjuster(
#             noise_pred_weight=noise_pred_weight,
#             feature_align_weight=feature_align_weight
#         )

#         current_losses = {
#             'noise_pred': noise_pred_loss.item(),
#             'feature_align': feature_align_loss.item() if isinstance(feature_align_loss, torch.Tensor) else feature_align_loss,
#         }

#         updated_weights = tuning_weight_adjuster.update_weights(current_losses)
#         noise_pred_weight = updated_weights['noise_pred']
#         feature_align_weight = updated_weights['feature_align']

#         # 总损失
#         total_loss = (
#             noise_pred_weight * noise_pred_loss +
#             feature_align_weight * feature_align_loss
#         )

#         # --- 反向传播和优化器步骤 ---
#         total_loss.backward()
        
#         # 可选：梯度裁剪
#         # torch.nn.utils.clip_grad_norm_(
#         #     itertools.chain(student_unet.parameters(), student_text_encoder.parameters()), 1.0
#         # )
        
#         optimizer.step()
#         lr_scheduler_lora.step()

#         # === 新增：记录当前teacher的损失到追踪器 ===
#         current_total_loss_item = total_loss.item()
#         current_noise_loss_item = noise_pred_loss.item()
#         current_feature_loss_item = feature_align_loss.item() if isinstance(feature_align_loss, torch.Tensor) else feature_align_loss

#         # 更新当前teacher的损失追踪器
#         teacher_loss_trackers[current_teacher_idx]['noise_pred_loss'] += current_noise_loss_item
#         teacher_loss_trackers[current_teacher_idx]['feature_align_loss'] += current_feature_loss_item
#         teacher_loss_trackers[current_teacher_idx]['total_loss'] += current_total_loss_item
#         teacher_loss_trackers[current_teacher_idx]['step_count'] += 1

#         # 记录全局累积损失
#         accumulated_total_loss += current_total_loss_item
#         accumulated_noise_loss += current_noise_loss_item
#         accumulated_feature_loss += current_feature_loss_item
#         step_count_in_interval += 1

#         current_lr = lr_scheduler_lora.get_last_lr()[0]
#         progress_bar.set_postfix({
#             'total_loss': f'{current_total_loss_item:.4f}',
#             'noise_loss': f'{current_noise_loss_item:.4f}',
#             'feature_loss': f'{current_feature_loss_item:.4f}',
#             'lr': f'{current_lr:.2e}',
#             'teacher': teacher_name_log  # 显示使用的是哪个教师模型
#         })
#         progress_bar.update(1)

#         # === 新增：详细的TensorBoard记录 ===
#         if writer:
#             # 全局损失记录
#             writer.add_scalar('损失_多teacher_LoRA/总损失_单步', current_total_loss_item, global_step)
#             writer.add_scalar('损失_多teacher_LoRA/噪声预测损失_单步', current_noise_loss_item, global_step)
#             writer.add_scalar('损失_多teacher_LoRA/特征对齐损失_单步', current_feature_loss_item, global_step)
#             writer.add_scalar('学习率_多teacher_LoRA/LoRA学习率', current_lr, global_step)
#             writer.add_scalar('教师模型_多teacher_LoRA/当前选择', current_teacher_idx + 1, global_step)
#             writer.add_scalar('权重_多teacher_LoRA/噪声预测权重', noise_pred_weight, global_step)
#             writer.add_scalar('权重_多teacher_LoRA/特征对齐权重', feature_align_weight, global_step)

#             # === 新增：按teacher分类的损失记录 ===
#             teacher_name_for_tb = f"Teacher_{current_teacher_idx+1:02d}"
            
#             # 当前teacher的单步损失
#             writer.add_scalar(f'Teacher损失_LoRA/{teacher_name_for_tb}/噪声预测损失_单步', current_noise_loss_item, global_step)
#             writer.add_scalar(f'Teacher损失_LoRA/{teacher_name_for_tb}/特征对齐损失_单步', current_feature_loss_item, global_step)
#             writer.add_scalar(f'Teacher损失_LoRA/{teacher_name_for_tb}/总损失_单步', current_total_loss_item, global_step)
            
#             # Teacher选择频率统计
#             teacher_selection_counts = [tracker['step_count'] for tracker in teacher_loss_trackers]
#             for i, count in enumerate(teacher_selection_counts):
#                 writer.add_scalar(f'Teacher统计_LoRA/Teacher_{i+1:02d}_选择次数', count, global_step)
            
#             # 记录各层的特征损失
#             if isinstance(feature_loss_dict, dict):
#                 for layer_name, layer_loss in feature_loss_dict.items():
#                     clean_layer_name = layer_name.replace(".", "_")
#                     layer_loss_value = layer_loss.item() if isinstance(layer_loss, torch.Tensor) else layer_loss
#                     writer.add_scalar(f'特征损失_多teacher_LoRA_层/{clean_layer_name}', layer_loss_value, global_step)
#                     # 按teacher分类记录
#                     writer.add_scalar(f'Teacher_特征层损失_LoRA/{teacher_name_for_tb}/{clean_layer_name}', layer_loss_value, global_step)

#         # wandb 记录
#         if log_wandb and global_step % 10 == 0:
#             log_data_wandb = {
#                 'step': global_step,
#                 'total_loss_step_multi_teacher_lora': current_total_loss_item,
#                 'noise_pred_loss_step_multi_teacher_lora': current_noise_loss_item,
#                 'feature_align_loss_step_multi_teacher_lora': current_feature_loss_item,
#                 'learning_rate_multi_teacher_lora': current_lr,
#                 'teacher_model_used_multi_teacher_lora': current_teacher_idx + 1,
#                 'noise_pred_weight_lora': noise_pred_weight,
#                 'feature_align_weight_lora': feature_align_weight,
#             }
#             if isinstance(feature_loss_dict, dict):
#                 log_data_wandb.update({
#                     f"feature_loss_layer_{k.replace('.', '_')}_multi_teacher_lora": 
#                     v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss 
#                     for k, v_loss in feature_loss_dict.items()
#                 })
#             # wandb.log(log_data_wandb)

#         global_step += 1

#         # --- 保存检查点 ---
#         if global_step > 0 and global_step % save_steps == 0:
#             # 计算全局区间平均损失
#             avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
#             avg_noise_loss_interval = accumulated_noise_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
#             avg_feature_loss_interval = accumulated_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0

#             print(f"\n" + "="*60)
#             print(f"MULTI-TEACHER LoRA CHECKPOINT - Step {global_step}/{num_steps}")
#             print(f"="*60)
#             print(f"Average Losses (last {step_count_in_interval} steps):")
#             print(f"  • Total Loss:           {avg_total_loss_interval:.6f}")
#             print(f"  • Noise Prediction:     {avg_noise_loss_interval:.6f} (weight: {noise_pred_weight})")
#             print(f"  • Feature Alignment:    {avg_feature_loss_interval:.6f} (weight: {feature_align_weight})")
#             print(f"  • Learning Rate:        {current_lr:.2e}")
#             print(f"  • Progress:             {(global_step/num_steps)*100:.1f}%")
#             print(f"  • Current Teacher:      Teacher {current_teacher_idx + 1}/{num_teachers}")
#             print(f"  • Teacher Strategy:     {teacher_selection_strategy}")
            
#             # 显示各层特征损失详情
#             if isinstance(feature_loss_dict, dict) and feature_loss_dict:
#                 print(f"Feature Loss Details:")
#                 for layer_name, layer_loss in feature_loss_dict.items():
#                     layer_loss_value = layer_loss.item() if isinstance(layer_loss, torch.Tensor) else layer_loss
#                     print(f"  • {layer_name:<20}: {layer_loss_value:.6f}")

#             # === 新增：计算和记录每个teacher的平均损失 ===
#             if writer:
#                 # 全局平均损失
#                 writer.add_scalar('损失_多teacher_LoRA/总损失_平均区间', avg_total_loss_interval, global_step)
#                 writer.add_scalar('损失_多teacher_LoRA/噪声预测损失_平均区间', avg_noise_loss_interval, global_step)
#                 writer.add_scalar('损失_多teacher_LoRA/特征对齐损失_平均区间', avg_feature_loss_interval, global_step)
                
#                 # 每个teacher的平均损失和统计
#                 for i, tracker in enumerate(teacher_loss_trackers):
#                     teacher_name = f"Teacher_{i+1:02d}"
#                     step_count = tracker['step_count']
                    
#                     if step_count > 0:
#                         avg_noise_loss_teacher = tracker['noise_pred_loss'] / step_count
#                         avg_feature_loss_teacher = tracker['feature_align_loss'] / step_count
#                         avg_total_loss_teacher = tracker['total_loss'] / step_count
                        
#                         # 记录每个teacher的平均损失
#                         writer.add_scalar(f'Teacher平均损失_LoRA/{teacher_name}/噪声预测损失_平均', avg_noise_loss_teacher, global_step)
#                         writer.add_scalar(f'Teacher平均损失_LoRA/{teacher_name}/特征对齐损失_平均', avg_feature_loss_teacher, global_step)
#                         writer.add_scalar(f'Teacher平均损失_LoRA/{teacher_name}/总损失_平均', avg_total_loss_teacher, global_step)
                        
#                         # 记录teacher使用频率
#                         teacher_usage_frequency = step_count / global_step
#                         writer.add_scalar(f'Teacher统计_LoRA/{teacher_name}/使用频率', teacher_usage_frequency, global_step)
                        
#                         print(f"  📊 {teacher_name} 统计:")
#                         print(f"    • 使用次数: {step_count}/{global_step} ({teacher_usage_frequency:.2%})")
#                         print(f"    • 平均噪声预测损失: {avg_noise_loss_teacher:.6f}")
#                         print(f"    • 平均特征对齐损失: {avg_feature_loss_teacher:.6f}")
#                         print(f"    • 平均总损失: {avg_total_loss_teacher:.6f}")
#                     else:
#                         print(f"  📊 {teacher_name}: 未使用")
                
#                 # 计算teacher损失对比指标
#                 active_teachers = [i for i, tracker in enumerate(teacher_loss_trackers) if tracker['step_count'] > 0]
#                 if len(active_teachers) > 1:
#                     # 计算teacher损失方差（衡量teacher间的损失差异）
#                     avg_losses_per_teacher = []
#                     for i in active_teachers:
#                         tracker = teacher_loss_trackers[i]
#                         avg_total_loss = tracker['total_loss'] / tracker['step_count']
#                         avg_losses_per_teacher.append(avg_total_loss)
                    
#                     if avg_losses_per_teacher:
#                         import statistics
#                         teacher_loss_mean = statistics.mean(avg_losses_per_teacher)
#                         teacher_loss_variance = statistics.variance(avg_losses_per_teacher) if len(avg_losses_per_teacher) > 1 else 0.0
#                         teacher_loss_std = statistics.stdev(avg_losses_per_teacher) if len(avg_losses_per_teacher) > 1 else 0.0
                        
#                         writer.add_scalar('Teacher对比_LoRA/损失均值', teacher_loss_mean, global_step)
#                         writer.add_scalar('Teacher对比_LoRA/损失方差', teacher_loss_variance, global_step)
#                         writer.add_scalar('Teacher对比_LoRA/损失标准差', teacher_loss_std, global_step)
                        
#                         # 损失一致性指标 (标准差/均值，越小表示teacher间损失越一致)
#                         if teacher_loss_mean > 0:
#                             loss_consistency = teacher_loss_std / teacher_loss_mean
#                             writer.add_scalar('Teacher对比_LoRA/损失一致性系数', loss_consistency, global_step)

#             print(f"="*60)

#             # 重置累积损失
#             accumulated_total_loss = 0.0
#             accumulated_noise_loss = 0.0
#             accumulated_feature_loss = 0.0
#             step_count_in_interval = 0

#             # 保存模型
#             current_save_dir = os.path.join(save_path, out_name)
#             if not os.path.exists(current_save_dir):
#                 os.makedirs(current_save_dir, exist_ok=True)

#             checkpoint_path = os.path.join(current_save_dir, f"multi_teacher_lora_step_{global_step}.safetensors")
#             save_all(
#                 unet=student_unet,
#                 text_encoder=student_text_encoder,
#                 placeholder_token_ids=placeholder_token_ids,
#                 placeholder_tokens=placeholder_tokens,
#                 save_path=checkpoint_path,
#                 target_replace_module_unet=lora_unet_target_modules,
#                 target_replace_module_text=lora_clip_target_modules
#             )
#             print(f"✓ Multi-Teacher LoRA Checkpoint saved to: {checkpoint_path}\n")

#         if global_step >= num_steps:
#             break

#     progress_bar.close()

#     # === 新增：最终统计报告 ===
#     print(f"\n" + "="*80)
#     print(f"多TEACHER LoRA训练完成 - 最终统计报告")
#     print(f"="*80)
#     print(f"总完成步数: {global_step}")
#     print(f"使用的teacher模型数量: {num_teachers}")
#     print(f"Teacher选择策略: {teacher_selection_strategy}")
#     print(f"最终学习率: {current_lr:.2e}")
    
#     # 记录最终的teacher使用统计到TensorBoard
#     if writer:
#         for i, tracker in enumerate(teacher_loss_trackers):
#             teacher_name = f"Teacher_{i+1:02d}"
#             step_count = tracker['step_count']
            
#             if step_count > 0:
#                 final_avg_noise_loss = tracker['noise_pred_loss'] / step_count
#                 final_avg_feature_loss = tracker['feature_align_loss'] / step_count
#                 final_avg_total_loss = tracker['total_loss'] / step_count
#                 final_usage_freq = step_count / global_step
                
#                 # 记录最终统计
#                 writer.add_scalar(f'最终统计_LoRA/{teacher_name}/最终平均噪声预测损失', final_avg_noise_loss, global_step)
#                 writer.add_scalar(f'最终统计_LoRA/{teacher_name}/最终平均特征对齐损失', final_avg_feature_loss, global_step)
#                 writer.add_scalar(f'最终统计_LoRA/{teacher_name}/最终平均总损失', final_avg_total_loss, global_step)
#                 writer.add_scalar(f'最终统计_LoRA/{teacher_name}/最终使用频率', final_usage_freq, global_step)
                
#                 print(f"\n📈 {teacher_name} 最终统计:")
#                 print(f"   总使用次数: {step_count}/{global_step} ({final_usage_freq:.2%})")
#                 print(f"   最终平均噪声预测损失: {final_avg_noise_loss:.6f}")
#                 print(f"   最终平均特征对齐损失: {final_avg_feature_loss:.6f}")
#                 print(f"   最终平均总损失: {final_avg_total_loss:.6f}")

#     writer.close()

#     # --- 保存最终模型 ---
#     final_save_dir = os.path.join(save_path, out_name)
#     if not os.path.exists(final_save_dir):
#         os.makedirs(final_save_dir, exist_ok=True)

#     final_model_path = os.path.join(final_save_dir, f"final_multi_teacher_lora_step_{global_step}.safetensors")
#     save_all(
#         unet=student_unet,
#         text_encoder=student_text_encoder,
#         placeholder_token_ids=placeholder_token_ids,
#         placeholder_tokens=placeholder_tokens,
#         save_path=final_model_path,
#         target_replace_module_unet=lora_unet_target_modules,
#         target_replace_module_text=lora_clip_target_modules
#     )
    
#     print(f"✓ Final multi-teacher LoRA model saved to: {final_model_path}")
#     print(f"✓ TensorBoard logs saved to: {tb_log_path}")
#     print(f"Multi-teacher LoRA training completed with feature alignment!")
#     print(f"="*80)

def perform_tuning_multi_teacher_dynamic(
    teacher_manager: DynamicTeacherModelManager,  # 使用动态管理器
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
    use_alternating_optimization: bool = True,
    alternating_interval: int = 5,
    alternating_schedule: str = "fixed",
    noise_only_steps: int = 100,
    feature_only_steps: int = 0,
    **kwargs
):
    """
    支持动态teacher加载的LoRA微调
    """
    
    num_teachers = teacher_manager.get_teacher_count()
    print(f"LoRA调优使用 {num_teachers} 个教师模型（动态加载模式）")
    
    import os
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    
    # === 交替优化控制器 ===
    class AlternatingOptimizationController:
        def __init__(self):
            self.current_mode = "noise"  # "noise" 或 "feature"
            self.mode_switch_count = 0
            self.last_switch_step = 0
            self.noise_loss_history = []
            self.feature_loss_history = []
            
        def should_switch_mode(self, global_step):
            """判断是否应该切换优化模式"""
            if alternating_schedule == "fixed":
                # 固定间隔切换
                if global_step > 0 and (global_step - self.last_switch_step) >= alternating_interval:
                    return True
            elif alternating_schedule == "adaptive":
                # 简单的自适应策略：基于损失趋势
                if len(self.noise_loss_history) >= 5 and len(self.feature_loss_history) >= 5:
                    # 如果当前优化的损失在最近几步没有明显下降，则切换
                    if self.current_mode == "noise":
                        recent_trend = (self.noise_loss_history[-1] - self.noise_loss_history[-3]) / (self.noise_loss_history[-3] + 1e-8)
                        if recent_trend > -0.01:  # 没有明显下降
                            return True
                    else:
                        recent_trend = (self.feature_loss_history[-1] - self.feature_loss_history[-3]) / (self.feature_loss_history[-3] + 1e-8)
                        if recent_trend > -0.01:  # 没有明显下降
                            return True
            return False
            
        def switch_mode(self, global_step):
            """切换优化模式"""
            old_mode = self.current_mode
            self.current_mode = "feature" if self.current_mode == "noise" else "noise"
            self.mode_switch_count += 1
            self.last_switch_step = global_step
            
            print(f"🔄 [Step {global_step}] 切换优化模式: {old_mode} -> {self.current_mode} (第{self.mode_switch_count}次切换)")
            
        def get_current_mode(self, global_step):
            """获取当前步的优化模式"""
            # 处理特殊阶段
            if global_step < noise_only_steps:
                return "noise"
            elif feature_only_steps > 0 and global_step >= (num_steps - feature_only_steps):
                return "feature"
            
            # 正常交替阶段
            return self.current_mode
            
        def update_loss_history(self, noise_loss, feature_loss):
            """更新损失历史"""
            self.noise_loss_history.append(noise_loss)
            self.feature_loss_history.append(feature_loss)
            
            # 保持历史长度
            if len(self.noise_loss_history) > 20:
                self.noise_loss_history.pop(0)
            if len(self.feature_loss_history) > 20:
                self.feature_loss_history.pop(0)

    # 创建交替优化控制器
    if use_alternating_optimization:
        alt_controller = AlternatingOptimizationController()
        print(f"✅ 启用交替优化模式:")
        print(f"   • 交替间隔: {alternating_interval} 步")
        print(f"   • 调度策略: {alternating_schedule}")
        if noise_only_steps > 0:
            print(f"   • 仅Noise优化: 前 {noise_only_steps} 步")
        if feature_only_steps > 0:
            print(f"   • 仅Feature优化: 后 {feature_only_steps} 步")
    else:
        alt_controller = None
        print(f"📊 使用传统联合损失优化模式")

    print(f"=" * 80)
    print(f"Starting Multi-Teacher LoRA Tuning with {'Alternating' if use_alternating_optimization else 'Joint'} Optimization (Dynamic)")
    print(f"Number of teachers: {num_teachers}")
    print(f"Total steps: {num_steps}")
    print(f"Save steps: {save_steps}")
    print(f"Teacher selection strategy: {teacher_selection_strategy}")
    print(f"Feature alignment weight: {feature_align_weight}")
    print(f"Noise prediction weight: {noise_pred_weight}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Output name: {out_name}")
    print(f"=" * 80)

    # --- 创建带时间戳的TensorBoard设置 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_mode = "alternating" if use_alternating_optimization else "joint"
    tb_log_path = os.path.join(tensorboard_log_dir, f"{out_name}_{opt_mode}_multi_teacher_lora_dynamic_{timestamp}")
    
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"📊 TensorBoard 日志将保存到: {tb_log_path}")

    # 特征对齐损失函数组件
    alignment_layers_for_loss = feature_alignment_unet_layers
    layer_weights_for_loss = {
        'mid_block': 2.0, 'down_blocks.2': 1.5, 'down_blocks.3': 1.5,
        'up_blocks.0': 1.5, 'up_blocks.1': 1.5,
    }
    layer_weights_for_loss = {k: v for k, v in layer_weights_for_loss.items() if k in alignment_layers_for_loss}

    # 使用增强的特征对齐损失函数
    feature_alignment_loss_fn = create_enhanced_feature_alignment_loss(
        alignment_layers=alignment_layers_for_loss,
        loss_weights=layer_weights_for_loss,
        loss_type="mse",
        feature_selection_strategy="adaptive",  # 智能选择特征
        normalize_features=True,  # 启用特征归一化
        channel_alignment="projection"  # 使用投影对齐通道
    )
    
    # 创建学生UNet特征提取器
    student_extractor = UNetFeatureExtractor(
        target_layers=feature_alignment_unet_layers,
        mixed_precision_config=mixed_precision
    )

    progress_bar = tqdm(range(num_steps), desc=f"Multi-Teacher LoRA ({'Alternating' if use_alternating_optimization else 'Joint'}) Training (Dynamic)")
    global_step = 0

    # 初始化dataloader迭代器
    dataloader_iters = [iter(dl) for dl in dataloaders]
    teacher_selection_index = 0  # 用于round_robin策略

    # === 为每个teacher模型的损失统计 ===
    teacher_loss_trackers = []
    for i in range(num_teachers):
        teacher_loss_trackers.append({
            'noise_pred_loss': 0.0,
            'feature_align_loss': 0.0,
            'total_loss': 0.0,
            'step_count': 0,
        })

    # 用于区间日志记录的累积损失
    accumulated_total_loss = 0.0
    accumulated_noise_loss = 0.0
    accumulated_feature_loss = 0.0
    step_count_in_interval = 0

    student_unet.train()
    student_text_encoder.train()

    # 显存优化函数
    def optimize_memory_usage():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            print(f"💾 显存使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已缓存")

    # ===== 主训练循环 =====
    for step_idx in range(num_steps):
        # --- 选择当前教师模型 ---
        current_teacher_idx = select_current_teacher(
            global_step=global_step,
            num_teachers=num_teachers,
            strategy=teacher_selection_strategy,
            weights=teacher_weights,
            selection_index=teacher_selection_index
        )
        
        # 动态加载当前teacher
        current_teacher_model = teacher_manager.load_teacher(current_teacher_idx)
        current_teacher_unet = current_teacher_model["unet"]
        current_teacher_text_encoder = current_teacher_model["text_encoder"]
        
        # 为当前teacher创建特征提取器
        current_teacher_extractor = UNetFeatureExtractor(
            target_layers=feature_alignment_unet_layers,
            mixed_precision_config=mixed_precision
        )
        
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

        # --- 前向传播计算两种损失（用于监控和决策） ---
        # 计算noise prediction loss
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

        # 计算feature alignment loss
        # Latent 处理
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

        # 文本编码
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

        # 噪声和时间步
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=expected_latents_dtype)

        # 教师模型前向传播（提取特征）
        with torch.no_grad():
            teacher_noise_pred, teacher_features = current_teacher_extractor.extract_features(
                unet_model=current_teacher_unet,
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=teacher_encoder_hidden_states,
                return_dict=unet_return_dict,
                use_grad=False
            )

        # 学生模型前向传播（提取特征）
        _, student_features = student_extractor.extract_features(
            unet_model=student_unet,
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=student_encoder_hidden_states,
            return_dict=unet_return_dict,
            use_grad=True
        )

        # 计算特征对齐损失
        feature_align_loss, feature_loss_dict = feature_alignment_loss_fn(
            teacher_features, student_features
        )

        # 记录损失值用于监控
        current_noise_loss_item = noise_pred_loss.item()
        current_feature_loss_item = feature_align_loss.item() if isinstance(feature_align_loss, torch.Tensor) else feature_align_loss

        # === 交替优化逻辑 ===
        if use_alternating_optimization:
            # 更新损失历史
            alt_controller.update_loss_history(current_noise_loss_item, current_feature_loss_item)
            
            # 检查是否需要切换模式
            if alt_controller.should_switch_mode(global_step):
                alt_controller.switch_mode(global_step)
            
            # 获取当前优化模式
            current_mode = alt_controller.get_current_mode(global_step)
            
            # 根据当前模式选择要优化的损失
            optimizer.zero_grad()
            
            if current_mode == "noise":
                # 只优化noise prediction loss
                actual_loss = noise_pred_loss
                loss_for_display = {
                    'optimizing': 'noise',
                    'active_loss': current_noise_loss_item,
                    'monitoring_loss': current_feature_loss_item
                }
            else:  # feature mode
                # 只优化feature alignment loss
                actual_loss = feature_align_loss
                loss_for_display = {
                    'optimizing': 'feature',
                    'active_loss': current_feature_loss_item,
                    'monitoring_loss': current_noise_loss_item
                }
            
            # 反向传播和优化
            actual_loss.backward()
            optimizer.step()
            lr_scheduler_lora.step()
            
            # 记录实际优化的损失作为total_loss
            current_total_loss_item = actual_loss.item()
            
        else:
            # 传统的联合优化
            tuning_weight_adjuster = create_tuning_weight_adjuster(
                noise_pred_weight=noise_pred_weight,
                feature_align_weight=feature_align_weight
            )

            current_losses = {
                'noise_pred': current_noise_loss_item,
                'feature_align': current_feature_loss_item,
            }

            updated_weights = tuning_weight_adjuster.update_weights(current_losses)
            noise_pred_weight = updated_weights['noise_pred']
            feature_align_weight = updated_weights['feature_align']

            # 总损失
            total_loss = (
                noise_pred_weight * noise_pred_loss +
                feature_align_weight * feature_align_loss
            )

            # 反向传播和优化器步骤
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

        # === 记录统计信息 ===
        # 更新当前teacher的损失追踪器
        teacher_loss_trackers[current_teacher_idx]['noise_pred_loss'] += current_noise_loss_item
        teacher_loss_trackers[current_teacher_idx]['feature_align_loss'] += current_feature_loss_item
        teacher_loss_trackers[current_teacher_idx]['total_loss'] += current_total_loss_item
        teacher_loss_trackers[current_teacher_idx]['step_count'] += 1

        # 记录全局累积损失
        accumulated_total_loss += current_total_loss_item
        accumulated_noise_loss += current_noise_loss_item
        accumulated_feature_loss += current_feature_loss_item
        step_count_in_interval += 1

        current_lr = lr_scheduler_lora.get_last_lr()[0]
        
        # 显示进度信息
        mode_display = current_mode.upper()
        progress_bar.set_postfix({
            'mode': mode_display,
            'active_loss': f'{loss_for_display["active_loss"]:.4f}',
            'noise_loss': f'{current_noise_loss_item:.4f}',
            'feature_loss': f'{current_feature_loss_item:.4f}',
            'lr': f'{current_lr:.2e}',
            'teacher': teacher_name_log
        })
        progress_bar.update(1)

        # === TensorBoard记录 ===
        if writer:
            # 基础损失记录
            writer.add_scalar('损失_动态/噪声预测损失_单步', current_noise_loss_item, global_step)
            writer.add_scalar('损失_动态/特征对齐损失_单步', current_feature_loss_item, global_step)
            writer.add_scalar('损失_动态/实际训练损失_单步', current_total_loss_item, global_step)
            writer.add_scalar('学习率_动态/LoRA学习率', current_lr, global_step)
            writer.add_scalar('教师模型_动态/当前选择', current_teacher_idx + 1, global_step)
            
            # 交替优化特有记录
            if use_alternating_optimization:
                writer.add_scalar('交替优化_动态/当前模式', 1 if current_mode == "noise" else 2, global_step)  # 1=noise, 2=feature
                writer.add_scalar('交替优化_动态/模式切换次数', alt_controller.mode_switch_count, global_step)
                writer.add_scalar('交替优化_动态/距离上次切换步数', global_step - alt_controller.last_switch_step, global_step)
                
                # 记录优化效果趋势
                if len(alt_controller.noise_loss_history) >= 5:
                    recent_noise_trend = (alt_controller.noise_loss_history[-1] - alt_controller.noise_loss_history[-5]) / (alt_controller.noise_loss_history[-5] + 1e-8)
                    writer.add_scalar('交替优化_动态/噪声损失趋势', recent_noise_trend, global_step)
                
                if len(alt_controller.feature_loss_history) >= 5:
                    recent_feature_trend = (alt_controller.feature_loss_history[-1] - alt_controller.feature_loss_history[-5]) / (alt_controller.feature_loss_history[-5] + 1e-8)
                    writer.add_scalar('交替优化_动态/特征损失趋势', recent_feature_trend, global_step)
            else:
                # 传统模式的权重记录
                writer.add_scalar('权重_动态/噪声预测权重', noise_pred_weight, global_step)
                writer.add_scalar('权重_动态/特征对齐权重', feature_align_weight, global_step)

            # 按teacher分类的损失记录
            teacher_name_for_tb = f"Teacher_{current_teacher_idx+1:02d}"
            writer.add_scalar(f'Teacher损失_动态/{teacher_name_for_tb}/噪声预测损失_单步', current_noise_loss_item, global_step)
            writer.add_scalar(f'Teacher损失_动态/{teacher_name_for_tb}/特征对齐损失_单步', current_feature_loss_item, global_step)
            writer.add_scalar(f'Teacher损失_动态/{teacher_name_for_tb}/总损失_单步', current_total_loss_item, global_step)

        global_step += 1

        # 定期清理显存
        if global_step % 50 == 0:
            teacher_manager.unload_current_teacher()
            optimize_memory_usage()

        # --- 保存检查点 ---
        if global_step > 0 and global_step % save_steps == 0:
            # 计算区间平均损失
            avg_total_loss_interval = accumulated_total_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_noise_loss_interval = accumulated_noise_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0
            avg_feature_loss_interval = accumulated_feature_loss / step_count_in_interval if step_count_in_interval > 0 else 0.0

            print(f"\n" + "="*60)
            print(f"MULTI-TEACHER LoRA CHECKPOINT (DYNAMIC) - Step {global_step}/{num_steps}")
            if use_alternating_optimization:
                print(f"交替优化模式 - 当前优化: {current_mode.upper()}")
                print(f"模式切换次数: {alt_controller.mode_switch_count}")
            else:
                print(f"联合损失优化模式")
            print(f"="*60)
            
            print(f"Average Losses (last {step_count_in_interval} steps):")
            print(f"  • Actual Training Loss:  {avg_total_loss_interval:.6f}")
            print(f"  • Noise Prediction:      {avg_noise_loss_interval:.6f}")
            print(f"  • Feature Alignment:     {avg_feature_loss_interval:.6f}")
            print(f"  • Learning Rate:         {current_lr:.2e}")
            print(f"  • Current Teacher:       Teacher {current_teacher_idx + 1}/{num_teachers}")

            # 交替优化统计
            if use_alternating_optimization:
                noise_steps = sum(1 for i in range(max(0, global_step - step_count_in_interval), global_step) 
                                if alt_controller.get_current_mode(i) == "noise")
                feature_steps = step_count_in_interval - noise_steps
                print(f"  • Optimization Distribution:")
                print(f"    - Noise Steps:         {noise_steps} ({noise_steps/step_count_in_interval*100:.1f}%)")
                print(f"    - Feature Steps:       {feature_steps} ({feature_steps/step_count_in_interval*100:.1f}%)")

            # Teacher统计
            print(f"\n📊 Teacher Performance Summary:")
            for i, tracker in enumerate(teacher_loss_trackers):
                teacher_name = f"Teacher {i+1}"
                step_count = tracker['step_count']
                
                if step_count > 0:
                    avg_noise_loss_teacher = tracker['noise_pred_loss'] / step_count
                    avg_feature_loss_teacher = tracker['feature_align_loss'] / step_count
                    avg_total_loss_teacher = tracker['total_loss'] / step_count
                    teacher_usage_frequency = step_count / global_step
                    
                    print(f"  📊 {teacher_name} 统计:")
                    print(f"    • 使用次数: {step_count}/{global_step} ({teacher_usage_frequency:.2%})")
                    print(f"    • 平均噪声预测损失: {avg_noise_loss_teacher:.6f}")
                    print(f"    • 平均特征对齐损失: {avg_feature_loss_teacher:.6f}")
                    print(f"    • 平均总损失: {avg_total_loss_teacher:.6f}")
                else:
                    print(f"  📊 {teacher_name}: 未使用")

            # 保存模型
            current_save_dir = os.path.join(save_path, out_name)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir, exist_ok=True)

            checkpoint_path = os.path.join(current_save_dir, f"dynamic_multi_teacher_lora_step_{global_step}.safetensors")
            save_all(
                unet=student_unet,
                text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                placeholder_tokens=placeholder_tokens,
                save_path=checkpoint_path,
                target_replace_module_unet=lora_unet_target_modules,
                target_replace_module_text=lora_clip_target_modules
            )
            print(f"✓ Dynamic Multi-Teacher LoRA Checkpoint saved to: {checkpoint_path}\n")

            # 重置累积损失
            accumulated_total_loss = 0.0
            accumulated_noise_loss = 0.0
            accumulated_feature_loss = 0.0
            step_count_in_interval = 0

        if global_step >= num_steps:
            break

    progress_bar.close()
    
    # === 最终统计报告 ===
    print(f"\n" + "="*80)
    print(f"多TEACHER {'交替优化' if use_alternating_optimization else '联合优化'} LoRA训练完成（动态模式）")
    print(f"="*80)
    
    if use_alternating_optimization:
        total_noise_steps = sum(1 for i in range(global_step) if alt_controller.get_current_mode(i) == "noise")
        total_feature_steps = global_step - total_noise_steps
        print(f"优化统计:")
        print(f"  • 总步数: {global_step}")
        print(f"  • 噪声优化步数: {total_noise_steps} ({total_noise_steps/global_step*100:.1f}%)")
        print(f"  • 特征优化步数: {total_feature_steps} ({total_feature_steps/global_step*100:.1f}%)")
        print(f"  • 模式切换次数: {alt_controller.mode_switch_count}")
        print(f"  • 最终优化模式: {alt_controller.current_mode.upper()}")

    # 最终清理
    teacher_manager.cleanup()
    writer.close()

    # --- 保存最终模型 ---
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    final_model_path = os.path.join(final_save_dir, f"final_dynamic_multi_teacher_lora_step_{global_step}.safetensors")
    save_all(
        unet=student_unet,
        text_encoder=student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=final_model_path,
        target_replace_module_unet=lora_unet_target_modules,
        target_replace_module_text=lora_clip_target_modules
    )
    
    print(f"✓ Final dynamic multi-teacher LoRA model saved to: {final_model_path}")
    print(f"✓ TensorBoard logs saved to: {tb_log_path}")
    print(f"Dynamic multi-teacher LoRA training completed with {'alternating' if use_alternating_optimization else 'joint'} optimization!")
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


# def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
#                                gradient_checkpointing, enable_xformers_memory_efficient_attention,
#                                placeholder_token_ids,
#                                # 新增简单参数
#                                unfreeze_text_encoder_layers: bool = True):
#     """
#     设置模型训练配置 - 简化版本
    
#     Args:
#         unfreeze_text_encoder_layers: 是否解冻一些text encoder层来提高训练效果
#     """
#     if gradient_checkpointing:
#         student_unet.enable_gradient_checkpointing()

#     if enable_xformers_memory_efficient_attention:
#         from diffusers.utils.import_utils import is_xformers_available
#         if is_xformers_available():
#             student_unet.enable_xformers_memory_efficient_attention()
#         else:
#             raise ValueError("xformers is not available. Make sure it is installed correctly")

#     student_unet.requires_grad_(False)
#     student_vae.requires_grad_(False)

#     # === 简化的Text Encoder解冻策略 ===
#     if unfreeze_text_encoder_layers:
#         print("🔓 启用Text Encoder部分解冻，提高训练效果...")
        
#         # 先冻结所有参数
#         student_text_encoder.requires_grad_(False)
        
#         # 只解冻最后2层的注意力机制和LayerNorm
#         total_layers = len(student_text_encoder.text_model.encoder.layers)
        
#         # 解冻最后2层
#         for i in range(max(0, total_layers - 2), total_layers):
#             # 解冻注意力层
#             for param in student_text_encoder.text_model.encoder.layers[i].self_attn.parameters():
#                 param.requires_grad = True
#             # 解冻LayerNorm
#             for param in student_text_encoder.text_model.encoder.layers[i].layer_norm1.parameters():
#                 param.requires_grad = True
#             for param in student_text_encoder.text_model.encoder.layers[i].layer_norm2.parameters():
#                 param.requires_grad = True
#             print(f"  ✓ 解冻 encoder.layers.{i}")
        
#         # 解冻final layer norm
#         for param in student_text_encoder.text_model.final_layer_norm.parameters():
#             param.requires_grad = True
#         print(f"  ✓ 解冻 final_layer_norm")
        
#         # 统计解冻的参数
#         total_params = sum(p.numel() for p in student_text_encoder.parameters())
#         trainable_params = sum(p.numel() for p in student_text_encoder.parameters() if p.requires_grad)
#         unfrozen_ratio = trainable_params / total_params * 100
        
#         print(f"📊 Text Encoder解冻统计:")
#         print(f"  • 总参数: {total_params:,}")
#         print(f"  • 可训练参数: {trainable_params:,}")
#         print(f"  • 解冻比例: {unfrozen_ratio:.2f}%")
        
#     else:
#         print("🔒 使用原始Text Encoder冻结策略（仅embedding可训练）")
        
#         # 原始的冻结策略
#         params_to_freeze = itertools.chain(
#             student_text_encoder.text_model.encoder.parameters(),
#             student_text_encoder.text_model.final_layer_norm.parameters(),
#             student_text_encoder.text_model.embeddings.position_embedding.parameters(),
#         )
#         for param in params_to_freeze:
#             param.requires_grad = False

def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
                               gradient_checkpointing, enable_xformers_memory_efficient_attention,
                               placeholder_token_ids, enable_adaptive_gradient_clipping=True,
                               enable_mixed_precision=True, train_vae_decoder=True,
                               enable_deeper_text_training=True, use_layerwise_lr=True):
    """
    设置模型训练配置 - 简化优化版
    """
    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
        if hasattr(student_text_encoder, 'gradient_checkpointing_enable'):
            student_text_encoder.gradient_checkpointing_enable()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            student_unet.enable_xformers_memory_efficient_attention()
            if hasattr(student_text_encoder, 'enable_xformers_memory_efficient_attention'):
                try:
                    student_text_encoder.enable_xformers_memory_efficient_attention()
                except:
                    print("Text encoder xformers attention failed, continuing...")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 基础设置
    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    # 改进的VAE训练策略
    if train_vae_decoder:
        # 训练更多VAE层以提升重建质量
        for name, param in student_vae.named_parameters():
            if 'decoder' in name and any(layer in name for layer in [
                'conv_out', 'norm_out', 'up_blocks.3', 'up_blocks.2', 'mid_block'
            ]):
                param.requires_grad = True
                print(f"启用VAE参数训练: {name}")

    # 改进的文本编码器训练策略
    if enable_deeper_text_training:
        # 训练更多文本编码器层，提升语义理解
        if hasattr(student_text_encoder.text_model.encoder, 'layers'):
            total_layers = len(student_text_encoder.text_model.encoder.layers)
            # 训练后1/3的层，而不是只训练最后2层
            trainable_layers = max(3, total_layers // 3)
            
            for i, layer in enumerate(student_text_encoder.text_model.encoder.layers):
                if i >= total_layers - trainable_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"解冻文本编码器第{i}层")
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # 保持关键组件可训练
        student_text_encoder.text_model.final_layer_norm.requires_grad_(True)
        
        # 让embedding层也可微调，提升token理解
        for param in student_text_encoder.text_model.embeddings.token_embedding.parameters():
            param.requires_grad = True
            
    else:
        # 原始策略
        params_to_freeze = itertools.chain(
            student_text_encoder.text_model.embeddings.position_embedding.parameters(),
            student_text_encoder.text_model.encoder.layers[:-2].parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False
        student_text_encoder.text_model.final_layer_norm.requires_grad_(True)

    return {
        'deeper_text_training': enable_deeper_text_training,
        'layerwise_lr': use_layerwise_lr
    }

# def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
#                                gradient_checkpointing, enable_xformers_memory_efficient_attention,
#                                placeholder_token_ids):
#     """
#     设置模型训练配置
#     """
#     if gradient_checkpointing:
#         student_unet.enable_gradient_checkpointing()

#     if enable_xformers_memory_efficient_attention:
#         from diffusers.utils.import_utils import is_xformers_available
#         if is_xformers_available():
#             student_unet.enable_xformers_memory_efficient_attention()
#         else:
#             raise ValueError("xformers is not available. Make sure it is installed correctly")

#     student_unet.requires_grad_(False)
#     student_vae.requires_grad_(False)

#     # 冻结文本编码器的大部分参数
#     params_to_freeze = itertools.chain(
#         student_text_encoder.text_model.encoder.parameters(),
#         student_text_encoder.text_model.final_layer_norm.parameters(),
#         student_text_encoder.text_model.embeddings.position_embedding.parameters(),
#     )
#     for param in params_to_freeze:
#         param.requires_grad = False

# def setup_model_training_config(student_unet, student_text_encoder, student_vae, 
#                                gradient_checkpointing, enable_xformers_memory_efficient_attention,
#                                placeholder_token_ids, enable_adaptive_gradient_clipping=True,
#                                enable_mixed_precision=True, train_vae_decoder=False):
#     """
#     设置模型训练配置 - 增强版
#     """
#     if gradient_checkpointing:
#         student_unet.enable_gradient_checkpointing()
#         # 为text encoder也启用gradient checkpointing以节省显存
#         if hasattr(student_text_encoder, 'gradient_checkpointing_enable'):
#             student_text_encoder.gradient_checkpointing_enable()

#     if enable_xformers_memory_efficient_attention:
#         from diffusers.utils.import_utils import is_xformers_available
#         if is_xformers_available():
#             student_unet.enable_xformers_memory_efficient_attention()
#             # 如果text encoder支持xformers，也启用
#             if hasattr(student_text_encoder, 'enable_xformers_memory_efficient_attention'):
#                 try:
#                     student_text_encoder.enable_xformers_memory_efficient_attention()
#                 except:
#                     print("Text encoder xformers attention failed, continuing...")
#         else:
#             raise ValueError("xformers is not available. Make sure it is installed correctly")

#     # 基础设置
#     student_unet.requires_grad_(False)
#     student_vae.requires_grad_(False)

#     # 可选：训练VAE decoder的部分层以提升重建质量
#     if train_vae_decoder:
#         # 只训练decoder的最后几层
#         for name, param in student_vae.named_parameters():
#             if 'decoder' in name and ('conv_out' in name or 'norm_out' in name):
#                 param.requires_grad = True
#                 print(f"启用VAE参数训练: {name}")

#     # 更精细的文本编码器冻结策略
#     # 完全冻结位置编码和大部分transformer层，但保留部分层的灵活性
#     params_to_freeze = itertools.chain(
#         student_text_encoder.text_model.embeddings.position_embedding.parameters(),
#         # 冻结前面大部分encoder层，保留后面几层的可训练性
#         student_text_encoder.text_model.encoder.layers[:-2].parameters() if hasattr(student_text_encoder.text_model.encoder, 'layers') else [],
#     )
    
#     for param in params_to_freeze:
#         param.requires_grad = False
    
#     # 保持final_layer_norm可训练以适应新的表征
#     student_text_encoder.text_model.final_layer_norm.requires_grad_(True)

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

# def setup_lora_parameters_enhanced(student_unet, student_text_encoder, train_text_encoder,
#                                 continue_inversion, use_extended_lora, lora_rank,
#                                 lora_unet_target_modules, lora_clip_target_modules,
#                                 lora_dropout_p, lora_scale, unet_lr, text_encoder_lr,
#                                 ti_lr, continue_inversion_lr,
#                                 # 增强的参数
#                                 text_encoder_unfrozen_lr: float = None,
#                                 unfreeze_more_layers: bool = True):
#     """
#     设置LoRA参数 - 增强版本，支持更多解冻的text encoder参数
#     """
#     # UNet LoRA设置
#     if not use_extended_lora:
#         unet_lora_params, _ = inject_trainable_lora(
#             student_unet,
#             r=lora_rank,
#             target_replace_module=lora_unet_target_modules,
#             dropout_p=lora_dropout_p,
#             scale=lora_scale,
#         )
#     else:
#         print("使用扩展UNet LoRA")
#         lora_unet_target_modules = lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
#         unet_lora_params, _ = inject_trainable_lora_extended(
#             student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
#         )

#     params_to_optimize = [
#         {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
#     ]

#     # === 增强的Text Encoder参数处理 ===
    
#     # 1. TI embedding参数
#     if continue_inversion:
#         ti_lr_to_use = continue_inversion_lr if continue_inversion_lr is not None else ti_lr
#         params_to_optimize.append({
#             "params": student_text_encoder.get_input_embeddings().parameters(),
#             "lr": ti_lr_to_use,
#         })
#         print(f"✓ 添加TI embedding参数 (lr: {ti_lr_to_use})")

#     # 2. 解冻的Text Encoder参数（非embedding）
#     unfrozen_params = []
#     for name, param in student_text_encoder.named_parameters():
#         # 跳过embedding参数和LoRA参数
#         if ('embeddings.token_embedding' in name or 
#             'embeddings.position_embedding' in name or 
#             'lora_' in name):
#             continue
#         # 收集其他可训练参数
#         if param.requires_grad:
#             unfrozen_params.append(param)
    
#     if unfrozen_params:
#         # 根据解冻程度调整学习率
#         if unfreeze_more_layers:
#             # 更多解冻需要更小的学习率
#             unfrozen_lr = text_encoder_unfrozen_lr if text_encoder_unfrozen_lr is not None else text_encoder_lr * 0.05
#             print(f"🚀 增强解冻模式 - 使用更小的学习率")
#         else:
#             # 标准解冻使用适中的学习率
#             unfrozen_lr = text_encoder_unfrozen_lr if text_encoder_unfrozen_lr is not None else text_encoder_lr * 0.1
        
#         params_to_optimize.append({
#             "params": unfrozen_params,
#             "lr": unfrozen_lr,
#         })
#         unfrozen_count = sum(p.numel() for p in unfrozen_params)
#         print(f"✓ 添加解冻的Text Encoder参数: {unfrozen_count:,} 参数 (lr: {unfrozen_lr})")

#     # 3. Text Encoder LoRA参数
#     if train_text_encoder:
#         text_encoder_lora_params, _ = inject_trainable_lora(
#             student_text_encoder,
#             target_replace_module=lora_clip_target_modules,
#             r=lora_rank,
#         )
#         params_to_optimize.append({
#             "params": itertools.chain(*text_encoder_lora_params),
#             "lr": text_encoder_lr,
#         })
#         lora_count = sum(p.numel() for p in itertools.chain(*text_encoder_lora_params))
#         print(f"✓ 添加Text Encoder LoRA参数: {lora_count:,} 参数 (lr: {text_encoder_lr})")

#     # 4. 统计和建议
#     total_trainable = sum(
#         sum(p.numel() for p in param_group["params"]) 
#         for param_group in params_to_optimize
#     )
#     print(f"\n📊 优化器参数组统计:")
#     for i, param_group in enumerate(params_to_optimize):
#         group_params = sum(p.numel() for p in param_group["params"])
#         lr = param_group["lr"]
#         print(f"  组 {i+1}: {group_params:,} 参数, lr={lr}")
#     print(f"  总计: {total_trainable:,} 可训练参数")
    
#     # 学习率建议
#     print(f"\n💡 学习率建议:")
#     if unfreeze_more_layers:
#         print(f"  • 使用增强解冻模式，建议:")
#         print(f"    - UNet LoRA: {unet_lr} (标准)")
#         print(f"    - Text Encoder解冻参数: {unfrozen_lr} (很小)")
#         print(f"    - Text Encoder LoRA: {text_encoder_lr} (适中)")
#         print(f"    - 监控梯度范数，避免梯度爆炸")
#     else:
#         print(f"  • 使用标准解冻模式，当前设置合理")

#     return params_to_optimize, None

# def setup_lora_parameters(student_unet, student_text_encoder, train_text_encoder,
#                                 continue_inversion, use_extended_lora, lora_rank,
#                                 lora_unet_target_modules, lora_clip_target_modules,
#                                 lora_dropout_p, lora_scale, unet_lr, text_encoder_lr,
#                                 ti_lr, continue_inversion_lr,
#                                 # 新增简单参数
#                                 text_encoder_unfrozen_lr: float = None):
#     """
#     设置LoRA参数 - 简化版本，支持解冻的text encoder参数
#     """
#     # UNet LoRA设置
#     if not use_extended_lora:
#         unet_lora_params, _ = inject_trainable_lora(
#             student_unet,
#             r=lora_rank,
#             target_replace_module=lora_unet_target_modules,
#             dropout_p=lora_dropout_p,
#             scale=lora_scale,
#         )
#     else:
#         print("使用扩展UNet LoRA")
#         lora_unet_target_modules = lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
#         unet_lora_params, _ = inject_trainable_lora_extended(
#             student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
#         )

#     params_to_optimize = [
#         {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
#     ]

#     # === 简化的Text Encoder参数处理 ===
    
#     # 1. TI embedding参数
#     if continue_inversion:
#         ti_lr_to_use = continue_inversion_lr if continue_inversion_lr is not None else ti_lr
#         params_to_optimize.append({
#             "params": student_text_encoder.get_input_embeddings().parameters(),
#             "lr": ti_lr_to_use,
#         })
#         print(f"✓ 添加TI embedding参数 (lr: {ti_lr_to_use})")

#     # 2. 解冻的Text Encoder参数（非embedding）
#     unfrozen_params = []
#     for name, param in student_text_encoder.named_parameters():
#         # 跳过embedding参数和LoRA参数
#         if ('embeddings.token_embedding' in name or 
#             'embeddings.position_embedding' in name or 
#             'lora_' in name):
#             continue
#         # 收集其他可训练参数
#         if param.requires_grad:
#             unfrozen_params.append(param)
    
#     if unfrozen_params:
#         # 使用较低的学习率给解冻的参数
#         unfrozen_lr = text_encoder_unfrozen_lr if text_encoder_unfrozen_lr is not None else text_encoder_lr * 0.1
#         params_to_optimize.append({
#             "params": unfrozen_params,
#             "lr": unfrozen_lr,
#         })
#         unfrozen_count = sum(p.numel() for p in unfrozen_params)
#         print(f"✓ 添加解冻的Text Encoder参数: {unfrozen_count:,} 参数 (lr: {unfrozen_lr})")

#     # 3. Text Encoder LoRA参数
#     if train_text_encoder:
#         text_encoder_lora_params, _ = inject_trainable_lora(
#             student_text_encoder,
#             target_replace_module=lora_clip_target_modules,
#             r=lora_rank,
#         )
#         params_to_optimize.append({
#             "params": itertools.chain(*text_encoder_lora_params),
#             "lr": text_encoder_lr,
#         })
#         lora_count = sum(p.numel() for p in itertools.chain(*text_encoder_lora_params))
#         print(f"✓ 添加Text Encoder LoRA参数: {lora_count:,} 参数 (lr: {text_encoder_lr})")

#     return params_to_optimize, None

def setup_lora_parameters(student_unet, student_text_encoder, train_text_encoder,
                         continue_inversion, use_extended_lora, lora_rank,
                         lora_unet_target_modules, lora_clip_target_modules,
                         lora_dropout_p, lora_scale, unet_lr, text_encoder_lr,
                         ti_lr, continue_inversion_lr, 
                         use_higher_lora_lr=True, use_layerwise_text_lr=True):
    """
    设置LoRA参数 - 保持原结构，简单优化学习率
    """
    # UNet LoRA设置 (保持不变)
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

    # 简单提升UNet LoRA学习率
    effective_unet_lr = unet_lr * 1.2 if use_higher_lora_lr else unet_lr
    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": effective_unet_lr},
    ]

    # 文本编码器设置
    student_text_encoder.requires_grad_(False)
    
    if continue_inversion:
        # 提升token embedding学习率，加强概念学习
        effective_ti_lr = continue_inversion_lr if continue_inversion_lr is not None else ti_lr
        if use_higher_lora_lr:
            effective_ti_lr *= 1.5  # 提升token学习率
            
        params_to_optimize += [
            {
                "params": student_text_encoder.get_input_embeddings().parameters(),
                "lr": effective_ti_lr,
                "weight_decay": 0.01,  # 添加少量权重衰减
            }
        ]
        student_text_encoder.requires_grad_(True)
        
        # 智能冻结：保留更多可训练层
        if hasattr(student_text_encoder.text_model.encoder, 'layers'):
            total_layers = len(student_text_encoder.text_model.encoder.layers)
            freeze_layers = max(0, total_layers - 4)  # 保留后4层可训练
            
            for i, layer in enumerate(student_text_encoder.text_model.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            # 原始冻结策略
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
        
        # 为文本编码器使用分层学习率
        if use_layerwise_text_lr and len(text_encoder_lora_params) > 1:
            # 后面的层用更高的学习率
            for i, param_group in enumerate(text_encoder_lora_params):
                layer_lr = text_encoder_lr * (1.0 + i * 0.1)  # 逐层递增学习率
                params_to_optimize += [
                    {
                        "params": param_group,
                        "lr": layer_lr,
                    }
                ]
                print(f"文本编码器LoRA第{i}组学习率: {layer_lr:.6f}")
        else:
            # 统一学习率，但稍微提升
            effective_text_lr = text_encoder_lr * 1.1 if use_higher_lora_lr else text_encoder_lr
            params_to_optimize += [
                {
                    "params": itertools.chain(*text_encoder_lora_params),
                    "lr": effective_text_lr,
                }
            ]

    return params_to_optimize, None

# def setup_lora_parameters(student_unet, student_text_encoder, train_text_encoder,
#                          continue_inversion, use_extended_lora, lora_rank,
#                          lora_unet_target_modules, lora_clip_target_modules,
#                          lora_dropout_p, lora_scale, unet_lr, text_encoder_lr,
#                          ti_lr, continue_inversion_lr, 
#                          # 新增参数
#                          enable_attention_lora=True, enable_mlp_lora=True,
#                          enable_conv_lora=True, enable_time_embed_lora=False,
#                          adaptive_lora_rank=False, min_lora_rank=4, max_lora_rank=64,
#                          train_cross_attention_only=False, enable_bias_tuning=False):
#     """
#     设置LoRA参数 - 增强版，支持更细粒度的可训练参数控制
#     """
    
#     # 扩展的UNet LoRA目标模块
#     ENHANCED_UNET_TARGET_REPLACE = set([
#         "Transformer2DModel",  # 基础attention模块
#         "Attention",
#         "CrossAttention", 
#         "SelfAttention",
#         "GEGLU",  # activation函数
#         "FeedForward"  # MLP层
#     ])
    
#     # 如果启用卷积LoRA
#     if enable_conv_lora:
#         ENHANCED_UNET_TARGET_REPLACE.update([
#             "Conv2d",
#             "GroupNorm",
#             "ResnetBlock2D",
#             "Downsample2D",
#             "Upsample2D"
#         ])
    
#     # 如果启用时间嵌入LoRA
#     if enable_time_embed_lora:
#         ENHANCED_UNET_TARGET_REPLACE.update([
#             "TimestepEmbedding",
#             "Timesteps"
#         ])

#     # UNet LoRA设置
#     if not use_extended_lora:
#         # 合并原有和新增的目标模块
#         enhanced_unet_targets = lora_unet_target_modules | ENHANCED_UNET_TARGET_REPLACE
        
#         unet_lora_params, _ = inject_trainable_lora(
#             student_unet,
#             r=lora_rank,
#             target_replace_module=enhanced_unet_targets,
#             dropout_p=lora_dropout_p,
#             scale=lora_scale,
#         )
#     else:
#         print("使用扩展UNet LoRA")
#         enhanced_unet_targets = lora_unet_target_modules | ENHANCED_UNET_TARGET_REPLACE
#         unet_lora_params, _ = inject_trainable_lora_extended(
#             student_unet, r=lora_rank, target_replace_module=enhanced_unet_targets
#         )

#     # 自适应学习率：为不同类型的层设置不同学习率
#     params_to_optimize = []
    
#     # 将UNet LoRA参数按类型分组
#     attention_params = []
#     conv_params = []
#     other_params = []
    
#     for param_group in unet_lora_params:
#         # 这里需要根据实际的参数名称来分类
#         # 简化处理：直接使用统一学习率，实际使用时可以根据参数名称细分
#         other_params.extend(param_group)
    
#     params_to_optimize.append({
#         "params": itertools.chain(*unet_lora_params), 
#         "lr": unet_lr,
#         "weight_decay": 1e-4  # 添加权重衰减
#     })

#     # 增强的文本编码器设置
#     student_text_encoder.requires_grad_(False)
    
#     # Textual Inversion增强
#     if continue_inversion:
#         # 不仅训练token embedding，还训练相关的projection层
#         ti_params = [student_text_encoder.get_input_embeddings().parameters()]
        
#         # 如果存在token type embeddings也加入训练
#         if hasattr(student_text_encoder.text_model.embeddings, 'token_type_embeddings'):
#             ti_params.append(student_text_encoder.text_model.embeddings.token_type_embeddings.parameters())
            
#         params_to_optimize.append({
#             "params": itertools.chain(*ti_params),
#             "lr": continue_inversion_lr if continue_inversion_lr is not None else ti_lr,
#             "weight_decay": 1e-5  # 较小的权重衰减
#         })
        
#         student_text_encoder.requires_grad_(True)
        
#         # 更精细的冻结策略：保留最后几层transformer的可训练性
#         params_to_freeze = itertools.chain(
#             student_text_encoder.text_model.encoder.layers[:-2].parameters() 
#             if hasattr(student_text_encoder.text_model.encoder, 'layers') else 
#             student_text_encoder.text_model.encoder.parameters(),
#             student_text_encoder.text_model.embeddings.position_embedding.parameters(),
#         )
#         for param in params_to_freeze:
#             param.requires_grad = False

#     # 增强的文本编码器LoRA训练
#     if train_text_encoder:
#         # 扩展的CLIP LoRA目标模块
#         enhanced_clip_targets = lora_clip_target_modules | {
#             "CLIPAttention",
#             "CLIPMLP", 
#             "CLIPEncoderLayer",
#             "CLIPTextTransformer"
#         }
        
#         # 如果只训练cross-attention相关的部分
#         if train_cross_attention_only:
#             enhanced_clip_targets = {"CLIPAttention"}
        
#         text_encoder_lora_params, _ = inject_trainable_lora(
#             student_text_encoder,
#             target_replace_module=enhanced_clip_targets,
#             r=lora_rank,
#             dropout_p=lora_dropout_p,
#         )
        
#         params_to_optimize.append({
#             "params": itertools.chain(*text_encoder_lora_params),
#             "lr": text_encoder_lr,
#             "weight_decay": 1e-4
#         })

#     # 如果启用bias tuning，添加bias参数到训练中
#     if enable_bias_tuning:
#         bias_params = []
#         for name, param in itertools.chain(student_unet.named_parameters(), 
#                                          student_text_encoder.named_parameters()):
#             if 'bias' in name and param.requires_grad:
#                 bias_params.append(param)
        
#         if bias_params:
#             params_to_optimize.append({
#                 "params": bias_params,
#                 "lr": min(unet_lr, text_encoder_lr) * 0.1,  # 较小的学习率
#                 "weight_decay": 0  # bias通常不使用权重衰减
#             })

#     return params_to_optimize, {
#         'total_trainable_params': sum(p.numel() for group in params_to_optimize for p in group['params'] if p.requires_grad),
#         'unet_lora_params': len(unet_lora_params) if 'unet_lora_params' in locals() else 0,
#         'text_encoder_lora_params': len(text_encoder_lora_params) if 'text_encoder_lora_params' in locals() else 0
#     }

def get_trainable_parameters_info(student_unet, student_text_encoder, student_vae):
    """
    获取可训练参数的详细信息
    """
    info = {
        'unet': {'total': 0, 'trainable': 0, 'trainable_names': []},
        'text_encoder': {'total': 0, 'trainable': 0, 'trainable_names': []},
        'vae': {'total': 0, 'trainable': 0, 'trainable_names': []}
    }
    
    # 统计UNet参数
    for name, param in student_unet.named_parameters():
        info['unet']['total'] += param.numel()
        if param.requires_grad:
            info['unet']['trainable'] += param.numel()
            info['unet']['trainable_names'].append(name)
    
    # 统计Text Encoder参数
    for name, param in student_text_encoder.named_parameters():
        info['text_encoder']['total'] += param.numel()
        if param.requires_grad:
            info['text_encoder']['trainable'] += param.numel()
            info['text_encoder']['trainable_names'].append(name)
    
    # 统计VAE参数
    for name, param in student_vae.named_parameters():
        info['vae']['total'] += param.numel()
        if param.requires_grad:
            info['vae']['trainable'] += param.numel()
            info['vae']['trainable_names'].append(name)
    
    return info

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
    use_alternating_optimization: bool = True,
    alternating_interval: int = 5,
    alternating_schedule: str = "fixed",
    noise_only_steps: int = 100,
    feature_only_steps: int = 0,
):
    """
    支持多teacher蒸馏的训练函数（动态加载模式）
    
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

    # --- 4. 创建多个教师模型（动态加载模式） ---
    print("📦 使用动态加载模式创建教师模型配置...")
    teacher_configs = create_multiple_teacher_models_dynamic(
        teacher_lora_paths=teacher_lora_paths,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device
    )

    # 创建动态teacher管理器
    teacher_manager = DynamicTeacherModelManager(teacher_configs)

    # --- 5. 创建数据加载器（需要临时加载一个teacher来获取tokenizer） ---
    print("📊 创建数据加载器...")
    
    dataloaders, token_maps = create_multiple_dataloaders_optimized(
        teacher_manager=teacher_manager,
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
        placeholder_token_ids=placeholder_token_ids,
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
        print(f"开始多Teacher文本逆向训练（动态加载模式，共 {num_teachers} 个教师模型）")
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

        # 调用动态多teacher文本逆向训练函数
        train_inversion_with_multi_feature_alignment_dynamic(
            teacher_manager=teacher_manager,  # 使用动态管理器
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
    print(f"开始多Teacher LoRA微调训练（动态加载模式，共 {num_teachers} 个教师模型）")
    print(f"{'='*80}")

    # 设置LoRA参数
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

    # 特征对齐层配置
    feature_alignment_unet_layers = [
        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
        'mid_block',
        'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
    ]

    # 调用动态多teacher LoRA微调函数
    perform_tuning_multi_teacher_dynamic(
        teacher_unets=teacher_manager,  # 使用动态管理器
        teacher_text_encoders=[],  # 空列表，因为使用动态管理器
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

    # --- 11. 最终清理 ---
    teacher_manager.cleanup()

    print(f"\n{'='*80}")
    print(f"多Teacher蒸馏训练完成！共使用了 {num_teachers} 个教师模型（动态加载模式）")
    print(f"最终模型已保存到: {output_dir}")
    print(f"显存占用已优化：始终只加载1个teacher模型")
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
