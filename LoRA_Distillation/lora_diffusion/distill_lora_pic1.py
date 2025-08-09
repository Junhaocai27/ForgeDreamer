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
import numpy as np

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

# --- START: Inlined utils.py content (Simplified/Placeholder) ---
class Controller:
    def __init__(self, self_layers_range: tuple = (0, 16)):
        self.num_self_layers = -1
        self.cur_self_layer = 0
        if isinstance(self_layers_range, tuple) and len(self_layers_range) == 2:
            self.self_layers = list(range(self_layers_range[0], self_layers_range[1]))
        elif isinstance(self_layers_range, list):
            self.self_layers = self_layers_range
        else:
            raise ValueError("self_layers_range should be a tuple (start, end) or a list of indices")

    def step(self):
        self.cur_self_layer = 0

class DataCache:
    def __init__(self):
        self.q_list = []
        self.k_list = []
        self.v_list = []
        self.out_list = [] # Output of F.scaled_dot_product_attention

    def clear(self):
        self.q_list.clear(); self.k_list.clear(); self.v_list.clear(); self.out_list.clear()

    def add(self, q, k, v, out):
        self.q_list.append(q); self.k_list.append(k); self.v_list.append(v); self.out_list.append(out)

    def get(self):
        return self.q_list[:], self.k_list[:], self.v_list[:], self.out_list[:]

def register_attn_control(unet, controller, cache):
    # --- _attn_forward_hook_factory and registration logic from previous answer ---
    # This is a complex function. For brevity, I'm assuming its correct implementation
    # as provided in the "整合后的代码" section of my previous long answer.
    # It should iterate unet.named_modules(), find "Attention" layers,
    # and replace their forward method with a hook that uses `controller` and `cache`.
    # The hook should:
    #   1. Perform the original attention calculation (or re-implement it if necessary).
    #   2. If it's a self-attention layer and controller.cur_self_layer is in controller.self_layers:
    #      cache.add(q, k, v, sdp_out)
    #   3. Increment controller.cur_self_layer for self-attention.
    #   4. Return the attention output.
    # controller.num_self_layers should be set to the number of attention modules found/hooked.
    # --- Placeholder for the actual register_attn_control implementation ---
    # print(f"Placeholder: register_attn_control called for a UNet. Controller: {controller}, Cache: {cache}")
    # For a minimal working example, this function needs to be fully implemented as discussed before.
    # A very simplified version (DOES NOT ACTUALLY HOOK, JUST SETS COUNT for demo):
    # count = 0
    # for name, module in unet.named_modules():
    #     if "Attention" in module.__class__.__name__ and hasattr(module, 'to_q'):
    #         count +=1
    # controller.num_self_layers = count
    # return
    # --- More complete (but still needs care) version of hook factory and registration ---
    def _attn_forward_hook_factory_internal(captured_controller, captured_cache):
        def _hooked_forward(self_module, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
            is_self_attn = encoder_hidden_states is None
            residual = hidden_states
            temb = kwargs.get('temb', None)

            if hasattr(self_module, 'spatial_norm') and self_module.spatial_norm is not None:
                hidden_states = self_module.spatial_norm(hidden_states, temb)
            
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                bs_in, ch_in, h_in, w_in = hidden_states.shape
                hidden_states = hidden_states.view(bs_in, ch_in, h_in * w_in).transpose(1, 2)
            else:
                bs_in, _, ch_in = hidden_states.shape # Approximate
                h_in, w_in = -1,-1

            q = self_module.to_q(hidden_states)
            _ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            if not is_self_attn and hasattr(self_module, 'norm_cross') and self_module.norm_cross:
                 _ehs = self_module.norm_encoder_hidden_states(_ehs)
            k = self_module.to_k(_ehs)
            v = self_module.to_v(_ehs)

            head_dim = q.shape[-1] // self_module.heads
            q_r = q.view(bs_in, -1, self_module.heads, head_dim).transpose(1, 2)
            k_r = k.view(bs_in, -1, self_module.heads, head_dim).transpose(1, 2)
            v_r = v.view(bs_in, -1, self_module.heads, head_dim).transpose(1, 2)
            
            # Simplified mask handling for F.sdp
            sdp_mask = None
            if attention_mask is not None and not is_self_attn:
                encoder_attention_mask = kwargs.get('encoder_attention_mask', attention_mask)
                if encoder_attention_mask is not None:
                    if encoder_attention_mask.ndim == 2: sdp_mask = encoder_attention_mask[:, None, None, :].to(q_r.device, dtype=q_r.dtype)
                    elif encoder_attention_mask.ndim == 4: sdp_mask = encoder_attention_mask.to(q_r.device, dtype=q_r.dtype)
            
            sdp_out = F.scaled_dot_product_attention(q_r, k_r, v_r, attn_mask=sdp_mask, dropout_p=0.0, is_causal=False)

            if is_self_attn and captured_controller.cur_self_layer in captured_controller.self_layers:
                captured_cache.add(q_r.detach().clone(), k_r.detach().clone(), v_r.detach().clone(), sdp_out.detach().clone())

            out_hs = sdp_out.transpose(1, 2).reshape(bs_in, -1, self_module.heads * head_dim)
            out_hs = self_module.to_out[0](out_hs)
            out_hs = self_module.to_out[1](out_hs)

            if input_ndim == 4:
                 if h_in != -1: out_hs = out_hs.transpose(-1, -2).reshape(bs_in, ch_in, h_in, w_in)
                 else: out_hs = out_hs.transpose(-1, -2).reshape(bs_in, ch_in, int(np.sqrt(out_hs.shape[1]//ch_in)), -1) # Fallback
            
            out_hs = out_hs + residual
            if hasattr(self_module, 'rescale_output_factor'): out_hs = out_hs / self_module.rescale_output_factor
            if is_self_attn: captured_controller.cur_self_layer += 1
            return out_hs
        return _hooked_forward

    hooked_count = 0
    for name, module in unet.named_modules():
        is_attn_module = ("Attention" in module.__class__.__name__ or "attn" in name.lower()) and \
                         hasattr(module, 'to_q') and hasattr(module, 'to_k') and \
                         hasattr(module, 'to_v') and hasattr(module, 'to_out') and \
                         hasattr(module, 'heads')
        if is_attn_module:
            if not hasattr(module, '_original_forward_distill'): # Avoid multiple hooks from same system
                module._original_forward_distill = module.forward
            new_forward_fn = _attn_forward_hook_factory_internal(controller, cache)
            module.forward = new_forward_fn.__get__(module, module.__class__)
            hooked_count += 1
    controller.num_self_layers = hooked_count
    # print(f"Registered {hooked_count} attention modules for {unet.__class__.__name__}.")
# --- END: Inlined utils.py ---

# --- START: Inlined losses.py (Simplified/Placeholder) ---
def ad_loss(q_list, ks_list_teacher, vs_list_teacher, self_out_list_student, scale=1.0, source_mask=None, target_mask=None):
    loss = torch.tensor(0.0, device=q_list[0].device if q_list else "cpu")
    if not q_list or not ks_list_teacher or not vs_list_teacher or not self_out_list_student: return loss
    if not (len(q_list) == len(ks_list_teacher) == len(vs_list_teacher) == len(self_out_list_student)):
        # print("Warning: ad_loss length mismatch, skipping.")
        return loss
    for q_s, k_t, v_t, out_s in zip(q_list, ks_list_teacher, vs_list_teacher, self_out_list_student):
        # mock_target_out = F.scaled_dot_product_attention(q_s * scale, k_t.detach(), v_t.detach()) # k_t, v_t from teacher
        # loss += F.mse_loss(out_s.float(), mock_target_out.float())
        # For a real ad_loss, ensure dimensions match and teacher K/V are appropriately batched/repeated if q_s has larger batch
        # Simplified for placeholder:
        if q_s.shape[0] == k_t.shape[0]: # Simple case, batch sizes match
             mock_target_out = F.scaled_dot_product_attention(q_s * scale, k_t.detach(), v_t.detach())
        elif k_t.shape[0] == 1 and q_s.shape[0] > 1 : # Teacher has batch 1, student has larger batch
             k_t_rpt = k_t.repeat(q_s.shape[0], 1, 1, 1)
             v_t_rpt = v_t.repeat(q_s.shape[0], 1, 1, 1)
             mock_target_out = F.scaled_dot_product_attention(q_s * scale, k_t_rpt.detach(), v_t_rpt.detach())
        else: # Mismatch that's harder to handle simply
            # print(f"Skipping ad_loss for one layer due to incompatible batch sizes: q_s {q_s.shape}, k_t {k_t.shape}")
            continue
        loss += F.mse_loss(out_s.float(), mock_target_out.float())

    return loss / len(q_list) if q_list else torch.tensor(0.0, device=loss.device)

def q_loss(pred_list, target_list): # Generic MSE list loss
    loss = torch.tensor(0.0, device=pred_list[0].device if pred_list else "cpu")
    if not pred_list or not target_list or len(pred_list) != len(target_list): return loss
    for pred, target in zip(pred_list, target_list):
        loss += F.mse_loss(pred.float(), target.float().detach())
    return loss / len(pred_list) if pred_list else torch.tensor(0.0, device=loss.device)
# --- END: Inlined losses.py ---

def loss_step_distill_lora(
    batch,
    teacher_unet,
    teacher_text_encoder,
    student_unet,
    student_text_encoder,
    scheduler, # DDPMScheduler instance
    vae,
    # For feature extraction:
    teacher_attention_controller,
    teacher_cache,
    student_attention_controller,
    student_cache,
    global_step: int,
    tuning: bool, # 这个参数在当前模仿的逻辑中可能影响不大，但保留
    # Loss weights:
    w_text_embed: float = 1.0,
    w_noise_pred: float = 1.0,
    w_feat_q: float = 0.1,
    w_feat_k: float = 0.1,
    w_feat_v: float = 0.1,
    w_feat_self_out: float = 0.1,
    w_feat_ad: float = 0.1,
    ad_loss_attn_scale: float = 1.0,
    # Other params
    save_image_every_n_steps: int = 100,
    output_dir_for_loss_step: str = "debug_distill_images",
    t_mutliplier: float = 1.0,
    mixed_precision_unet: bool = True, # 控制此函数内部是否对学生模型使用 autocast
    mask_temperature: float = 1.0,
    **kwargs # Absorbs any extra parameters passed from calling function
):
    device = student_unet.device

    # 学生模型参数期望为 FP32 (尤其是 LoRA 层)
    student_model_native_dtype = student_unet.dtype # 应该是 torch.float32
    if student_model_native_dtype != torch.float32:
        if global_step % 200 == 0: #减少打印频率
            print(f"[loss_step_distill_lora Warning G{global_step}] Student UNet native dtype is {student_model_native_dtype}, expected torch.float32 for stable training without external GradScaler.")

    # autocast 内部运算的目标 dtype
    autocast_compute_dtype = torch.float16 if mixed_precision_unet else student_model_native_dtype

    # 教师模型 dtype
    teacher_unet_dtype = next(teacher_unet.parameters()).dtype
    teacher_text_encoder_dtype = next(teacher_text_encoder.parameters()).dtype

    # 潜变量和噪声，初始为学生模型的原生 (FP32) dtype
    latents_gt_x0 = batch["pixel_values"].to(device=device, dtype=student_model_native_dtype)
    bsz = latents_gt_x0.shape[0]

    timesteps = torch.randint(
        0, int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,), device=latents_gt_x0.device,
    ).long()

    noise = torch.randn_like(latents_gt_x0) # FP32 噪声
    noisy_latents_xt = scheduler.add_noise(latents_gt_x0, noise, timesteps) # noisy_latents_xt 也是 FP32

    # --- 教师模型前向传播 (在 torch.no_grad() 中，不使用此函数内的 autocast) ---
    with torch.no_grad():
        teacher_unet.eval()
        teacher_text_encoder.eval()

        teacher_input_ids = batch.get("input_ids_teacher", batch["input_ids"]).to(device)
        teacher_attn_mask = batch.get("attention_mask_teacher", batch.get("attention_mask", None))
        if teacher_attn_mask is not None: teacher_attn_mask = teacher_attn_mask.to(device)

        # 教师文本编码器按其原生 dtype 运行
        _teacher_text_encoder_output = teacher_text_encoder(teacher_input_ids, attention_mask=teacher_attn_mask)
        target_text_embeds_raw = _teacher_text_encoder_output[0] # dtype is teacher_text_encoder_dtype

        # 准备教师 UNet 输入
        teacher_unet_input_latents = noisy_latents_xt.to(dtype=teacher_unet_dtype)
        target_text_embeds_for_teacher_unet = target_text_embeds_raw.to(dtype=teacher_unet_dtype)

        target_noise_pred_raw = teacher_unet(
            sample=teacher_unet_input_latents,
            timestep=timesteps,
            encoder_hidden_states=target_text_embeds_for_teacher_unet
        ).sample # dtype is teacher_unet_dtype

        # 特征提取 (教师)
        teacher_cache.clear()
        teacher_attention_controller.step()
        _ = teacher_unet(
            sample=teacher_unet_input_latents, # 确保输入一致
            timestep=timesteps,
            encoder_hidden_states=target_text_embeds_for_teacher_unet
        ).sample
        q_teacher_list, k_teacher_list, v_teacher_list, self_out_teacher_list = teacher_cache.get()

    # --- 学生模型前向传播 (受 mixed_precision_unet 控制的 autocast) ---
    student_input_ids = batch.get("input_ids_student", batch.get("aux1_input_ids", batch["input_ids"])).to(device)
    student_attn_mask = batch.get("attention_mask_student", batch.get("aux1_attention_mask", batch.get("attention_mask", None)))
    if student_attn_mask is not None: student_attn_mask = student_attn_mask.to(device)

    # 学生文本编码器
    # 输入的 dtype 应该和 student_text_encoder.dtype 匹配，autocast 会处理内部
    with torch.cuda.amp.autocast(enabled=mixed_precision_unet, dtype=autocast_compute_dtype if mixed_precision_unet else None):
        _student_text_encoder_output = student_text_encoder(student_input_ids, attention_mask=student_attn_mask)
        student_text_embeds_raw = _student_text_encoder_output[0] # dtype 受 autocast 影响

    # 学生 UNet
    # 准备学生 UNet 输入，使其与 autocast 内部期望的 dtype 一致
    student_unet_input_latents = noisy_latents_xt.to(dtype=autocast_compute_dtype if mixed_precision_unet else student_model_native_dtype)
    student_text_embeds_for_student_unet = student_text_embeds_raw.to(dtype=autocast_compute_dtype if mixed_precision_unet else student_model_native_dtype)

    with torch.cuda.amp.autocast(enabled=mixed_precision_unet, dtype=autocast_compute_dtype if mixed_precision_unet else None):
        student_noise_pred_raw = student_unet(
            sample=student_unet_input_latents,
            timestep=timesteps,
            encoder_hidden_states=student_text_embeds_for_student_unet
        ).sample # dtype 受 autocast 影响

        # 特征提取 (学生)
        student_cache.clear()
        student_attention_controller.step()
        _ = student_unet( # 使用与上面相同的输入
            sample=student_unet_input_latents,
            timestep=timesteps,
            encoder_hidden_states=student_text_embeds_for_student_unet
        ).sample
        q_student_list, k_student_list, v_student_list, self_out_student_list = student_cache.get()

    # --- 4. Calculate Losses (确保所有输入到 F.mse_loss 等的张量是 FP32) ---
    loss_text = w_text_embed * F.mse_loss(
        student_text_embeds_raw.float(),       # 转为 FP32
        target_text_embeds_raw.float().detach()# 转为 FP32 并 detach
    )

    actual_target_noise_for_loss = noise # 原始 FP32 噪声
    student_pred_for_loss = student_noise_pred_raw # 可能 FP16 或 FP32

    if batch.get("mask", None) is not None:
        img_mask = batch["mask"].to(device=device) # Mask 先到 device
        # 调整 mask dtype 和 shape
        img_mask = img_mask.to(dtype=student_pred_for_loss.dtype) # 与 pred 同 dtype 以便乘法
        if img_mask.ndim == 3: img_mask = img_mask.unsqueeze(1)
        if img_mask.shape[-2:] != student_pred_for_loss.shape[-2:]:
            img_mask = F.interpolate(img_mask, size=student_pred_for_loss.shape[-2:], mode='nearest')

        img_mask_powered = (img_mask + 0.01).pow(mask_temperature)
        img_mask_normalized = img_mask_powered / (img_mask_powered.max() + 1e-8)
        
        loss_pred = w_noise_pred * F.mse_loss(
            (student_pred_for_loss * img_mask_normalized).float(), # 应用 mask 后转 FP32
            (actual_target_noise_for_loss * img_mask_normalized.to(actual_target_noise_for_loss.dtype)).float().detach() # 应用 mask (转为noise同dtype) 后转 FP32
        )
    else:
        loss_pred = w_noise_pred * F.mse_loss(
            student_pred_for_loss.float(), # 转为 FP32
            actual_target_noise_for_loss.float().detach() # 转为 FP32 并 detach
        )

    # 特征损失
    total_feat_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    len_compare = 0
    # ... (您的特征列表长度检查逻辑可以保留)
    if q_student_list and k_student_list and v_student_list and self_out_student_list and \
       q_teacher_list and k_teacher_list and v_teacher_list and self_out_teacher_list:
        len_compare = min(len(q_student_list), len(q_teacher_list),
                          len(k_student_list), len(k_teacher_list),
                          len(v_student_list), len(v_teacher_list),
                          len(self_out_student_list), len(self_out_teacher_list))
        # (可选的长度不匹配警告)

    if len_compare > 0:
        # 学生特征需要保留梯度，教师特征需要 detach
        # 全部转换为 FP32 进行损失计算
        _qs_s = [f.float() for f in q_student_list[:len_compare] if f is not None]
        _ks_s = [f.float() for f in k_student_list[:len_compare] if f is not None]
        _vs_s = [f.float() for f in v_student_list[:len_compare] if f is not None]
        _os_s = [f.float() for f in self_out_student_list[:len_compare] if f is not None]

        _qt_t_detached = [f.float().detach() for f in q_teacher_list[:len_compare] if f is not None]
        _kt_t_detached = [f.float().detach() for f in k_teacher_list[:len_compare] if f is not None]
        _vt_t_detached = [f.float().detach() for f in v_teacher_list[:len_compare] if f is not None]
        _ot_t_detached = [f.float().detach() for f in self_out_teacher_list[:len_compare] if f is not None]

        # 确保列表非空且长度一致才计算相应损失
        if w_feat_q > 0 and _qs_s and _qt_t_detached and len(_qs_s) == len(_qt_t_detached):
            loss_q_val = q_loss(_qs_s, _qt_t_detached) # 假设 q_loss 返回 FP32
            if not torch.isnan(loss_q_val) and not torch.isinf(loss_q_val): total_feat_loss += w_feat_q * loss_q_val
            else: print(f"NaN/Inf in q_loss at step {global_step}")
        
        if w_feat_k > 0 and _ks_s and _kt_t_detached and len(_ks_s) == len(_kt_t_detached):
            loss_k_val = q_loss(_ks_s, _kt_t_detached)
            if not torch.isnan(loss_k_val) and not torch.isinf(loss_k_val): total_feat_loss += w_feat_k * loss_k_val
            else: print(f"NaN/Inf in k_loss at step {global_step}")

        if w_feat_v > 0 and _vs_s and _vt_t_detached and len(_vs_s) == len(_vt_t_detached):
            loss_v_val = q_loss(_vs_s, _vt_t_detached)
            if not torch.isnan(loss_v_val) and not torch.isinf(loss_v_val): total_feat_loss += w_feat_v * loss_v_val
            else: print(f"NaN/Inf in v_loss at step {global_step}")

        if w_feat_self_out > 0 and _os_s and _ot_t_detached and len(_os_s) == len(_ot_t_detached):
            loss_so_val = q_loss(_os_s, _ot_t_detached)
            if not torch.isnan(loss_so_val) and not torch.isinf(loss_so_val): total_feat_loss += w_feat_self_out * loss_so_val
            else: print(f"NaN/Inf in self_out_loss at step {global_step}")
            
        if w_feat_ad > 0 and _qs_s and _kt_t_detached and _vt_t_detached and _os_s and \
           len(_qs_s) == len(_kt_t_detached) == len(_vt_t_detached) == len(_os_s):
            loss_ad_val = ad_loss(_qs_s, _kt_t_detached, _vt_t_detached, _os_s, scale=ad_loss_attn_scale)
            if not torch.isnan(loss_ad_val) and not torch.isinf(loss_ad_val): total_feat_loss += w_feat_ad * loss_ad_val
            else: print(f"NaN/Inf in ad_loss at step {global_step}")
    
    total_loss = loss_text + loss_pred + total_feat_loss # 它们应该都已经是 FP32 了
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"NaN/Inf in total_loss at step {global_step}. Components:")
        print(f"  loss_text: {loss_text.item() if torch.is_tensor(loss_text) and not torch.isnan(loss_text) else 'NaN/NotTensor'}")
        print(f"  loss_pred: {loss_pred.item() if torch.is_tensor(loss_pred) and not torch.isnan(loss_pred) else 'NaN/NotTensor'}")
        print(f"  total_feat_loss: {total_feat_loss.item() if torch.is_tensor(total_feat_loss) and not torch.isnan(total_feat_loss) else 'NaN/NotTensor'}")

    # --- 5. Visualization (Optional) ---
    if save_image_every_n_steps > 0 and global_step % save_image_every_n_steps == 0:
        if not os.path.exists(output_dir_for_loss_step):
            os.makedirs(output_dir_for_loss_step, exist_ok=True)
        with torch.no_grad():
            vae.eval()
            # VAE 解码通常在 FP32 下进行以保证质量
            vae_decode_dtype = torch.float32
            vae_native_dtype = vae.dtype # VAE 模型自身的参数 dtype
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

            # 学生预测图像
            student_pred_noise_for_viz = student_noise_pred_raw[0:1] # dtype 受 autocast 影响
            noisy_latents_for_viz = noisy_latents_xt[0:1].to(dtype=student_pred_noise_for_viz.dtype) # 匹配 pred_noise 的 dtype 给 scheduler
            
            if not torch.isnan(student_pred_noise_for_viz).any():
                pred_x0_student_latent = scheduler.step(
                    student_pred_noise_for_viz,
                    timesteps[0:1],
                    noisy_latents_for_viz
                ).pred_original_sample # dtype 应该和 student_pred_noise_for_viz 一致

                if not torch.isnan(pred_x0_student_latent).any():
                    # 解码前转为 VAE 期望的 FP32 输入，并应用 scaling_factor
                    pred_x0_student_latent_for_vae = pred_x0_student_latent.to(dtype=vae_decode_dtype) / scaling_factor
                    with torch.cuda.amp.autocast(enabled=False): # 确保 VAE 解码在 FP32
                        pred_x0_student_img = vae.decode(pred_x0_student_latent_for_vae.to(vae_native_dtype)).sample
                    pred_x0_student_img = (pred_x0_student_img / 2 + 0.5).clamp(0, 1)
                    save_image(pred_x0_student_img, os.path.join(output_dir_for_loss_step, f"step_{global_step}_student_pred_x0.png"))
                else: print(f"NaN in pred_x0_student_latent for viz at G{global_step}.")
            else: print(f"NaN in student_noise_pred for viz at G{global_step}.")

            # GT 图像
            latents_gt_x0_for_viz = latents_gt_x0[0:1] # 原始 FP32 潜变量
            if not torch.isnan(latents_gt_x0_for_viz).any():
                latents_gt_x0_for_vae = latents_gt_x0_for_viz.to(dtype=vae_decode_dtype) / scaling_factor
                with torch.cuda.amp.autocast(enabled=False):
                    gt_x0_img = vae.decode(latents_gt_x0_for_vae.to(vae_native_dtype)).sample
                gt_x0_img = (gt_x0_img / 2 + 0.5).clamp(0, 1)
                save_image(gt_x0_img, os.path.join(output_dir_for_loss_step, f"step_{global_step}_gt_x0.png"))
            else: print(f"NaN in latents_gt_x0 for viz at G{global_step}.")
            
    return total_loss

def train_inversion(
    # --- "Teacher" 模型参数 ---
    teacher1_unet, teacher2_unet, teacher1_text_encoder, teacher2_text_encoder,
    # --- "Student" 模型参数 ---
    student_unet, vae, student_text_encoder,
    # --- Dataloader 参数 ---
    dataloader1, dataloader2,
    # --- 核心训练参数 ---
    num_steps: int, scheduler, index_no_updates, optimizer, save_steps: int,
    placeholder_token_ids, placeholder_tokens, save_path: str,
    # --- 学习率调度器 ---
    lr_scheduler_main,
    # --- LoRA 相关参数 (TI 中通常不直接使用) ---
    lora_unet_target_modules, lora_clip_target_modules,
    # --- 输出和日志相关 ---
    out_name: str, tokenizer, test_image_path: str, cached_latents: bool,
    teacher_attention_controller1, teacher_cache1,
    teacher_attention_controller2, teacher_cache2,
    student_attention_controller_main, student_cache_main,
    distill_loss_weights: dict, # Expected to contain all w_* and alpha_teacher1
    distillation_strategy: str,
    mixed_precision_unet_for_distill: bool, # Explicitly for loss_step's unet precision
    # --- 损失函数特定参数 ---
    mask_temperature: float = 1.0, # Passed to loss_step
    # t_multiplier_loss: float = 1.0, # Will be taken from distill_loss_weights or default in loss_step
    # save_image_every_n_steps_loss: int = 200, # Will be taken from distill_loss_weights or default in loss_step
    # --- 其他 ---
    accum_iter: int = 1, log_wandb: bool = False, wandb_log_prompt_cnt: int = 10,
    class_token: str = "person", train_inpainting: bool = False,
    mixed_precision: bool = False, # Used as mixed_precision_unet_for_distill
    clip_ti_decay: bool = True, tensorboard_log_dir: str = "runs/new1",
    # --- 新增蒸馏相关参数 ---

    # Optional: pass specific t_multiplier and save_steps for loss_step's internal viz
    # These can also be part of distill_loss_weights if preferred
    t_multiplier_for_loss_step: float = 1.0,
    save_image_every_n_steps_for_loss_step: int = 200,
):
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_inversion")
    if not os.path.exists(tb_log_path): os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging for inversion to: {tb_log_path}")

    loss_step_image_output_dir = os.path.join(save_path, out_name, "distill_debug_images_inversion_ti")
    if not os.path.exists(loss_step_image_output_dir): os.makedirs(loss_step_image_output_dir, exist_ok=True)
    print(f"Debug images from loss_step_distill_lora (TI) will be saved to: {loss_step_image_output_dir}")

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description(f"Steps (Inversion w/ Distill Loss - {distillation_strategy})")
    global_step = 0
    orig_embeds_params = student_text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = None # prepare_clip_model_sets() # Ensure this function is defined if used
        if 'prepare_clip_model_sets' in globals() and callable(globals()['prepare_clip_model_sets']):
            preped_clip = globals()['prepare_clip_model_sets']()


    index_updates = ~index_no_updates
    loss_sum_interval = 0.0

    iter_dataloader1 = iter(dataloader1)
    iter_dataloader2 = iter(dataloader2)

    alpha_t1 = distill_loss_weights.get("alpha_teacher1", 0.5) # Default if not in dict
    alpha_t2 = 1.0 - alpha_t1

    # --- 辅助函数，用于计算单个教师的蒸馏损失 ---
    def _calculate_loss_using_distill_fn(current_batch, current_teacher_unet, current_teacher_text_encoder,
                                         current_teacher_controller, current_teacher_cache):
        return loss_step_distill_lora(
            batch=current_batch,
            teacher_unet=current_teacher_unet,
            teacher_text_encoder=current_teacher_text_encoder,
            student_unet=student_unet,
            student_text_encoder=student_text_encoder,
            scheduler=scheduler, vae=vae,
            teacher_attention_controller=current_teacher_controller, teacher_cache=current_teacher_cache,
            student_attention_controller=student_attention_controller_main, student_cache=student_cache_main,
            # Unpack all weights from the distill_loss_weights dictionary
            w_text_embed=distill_loss_weights.get("w_text_embed", 1.0),
            w_noise_pred=distill_loss_weights.get("w_noise_pred", 0.1),
            w_feat_q=distill_loss_weights.get("w_feat_q", 0.02),
            w_feat_k=distill_loss_weights.get("w_feat_k", 0.02),
            w_feat_v=distill_loss_weights.get("w_feat_v", 0.02),
            w_feat_self_out=distill_loss_weights.get("w_feat_self_out", 0.02),
            w_feat_ad=distill_loss_weights.get("w_feat_ad", 0.02),
            ad_loss_attn_scale=distill_loss_weights.get("ad_loss_attn_scale", 1.0),
            global_step=global_step, # Pass current global_step for logging/saving inside loss_step
            save_image_every_n_steps=save_image_every_n_steps_for_loss_step,
            output_dir_for_loss_step=loss_step_image_output_dir,
            t_mutliplier=t_multiplier_for_loss_step,
            mixed_precision_unet=mixed_precision_unet_for_distill, # Use the specific arg
            mask_temperature=mask_temperature,
            tuning=False,
        )

    for step_idx in range(num_steps):
        student_unet.eval() 
        student_text_encoder.train() # Only embeddings are trainable due to prior setup

        try: batch1 = next(iter_dataloader1)
        except StopIteration: iter_dataloader1 = iter(dataloader1); batch1 = next(iter_dataloader1)
        
        if distillation_strategy == "alternating" or (distillation_strategy == "weighted_average" and dataloader1 != dataloader2):
            try: batch2 = next(iter_dataloader2)
            except StopIteration: iter_dataloader2 = iter(dataloader2); batch2 = next(iter_dataloader2)
        else: batch2 = batch1 # Use same batch for both teachers

        # Note: lr_scheduler_main.step() is called *after* optimizer.step() in your original code if accum_iter is involved
        
        # Zero gradients before loss calculation if not accumulating, or before accumulation loop
        if (global_step % accum_iter == 0) or accum_iter == 1:
            optimizer.zero_grad()

        current_step_total_loss = torch.tensor(0.0, device=student_unet.device)
        loss_t1_val, loss_t2_val = None, None # For logging

        if distillation_strategy == "weighted_average":
            loss_t1 = _calculate_loss_using_distill_fn(batch1, teacher1_unet, teacher1_text_encoder, teacher_attention_controller1, teacher_cache1)
            loss_t2 = _calculate_loss_using_distill_fn(batch2, teacher2_unet, teacher2_text_encoder, teacher_attention_controller2, teacher_cache2)
            current_step_total_loss = alpha_t1 * loss_t1 + alpha_t2 * loss_t2
            loss_t1_val, loss_t2_val = loss_t1.detach().item(), loss_t2.detach().item()
        elif distillation_strategy == "alternating":
            if global_step % 2 == 0:
                current_step_total_loss = _calculate_loss_using_distill_fn(batch1, teacher1_unet, teacher1_text_encoder, teacher_attention_controller1, teacher_cache1)
                loss_t1_val = current_step_total_loss.detach().item()
            else:
                current_step_total_loss = _calculate_loss_using_distill_fn(batch2, teacher2_unet, teacher2_text_encoder, teacher_attention_controller2, teacher_cache2)
                loss_t2_val = current_step_total_loss.detach().item()
        else:
            raise ValueError(f"Unknown distillation_strategy: {distillation_strategy}")

        loss_for_backward = current_step_total_loss / accum_iter
        loss_for_backward.backward()
        loss_sum_interval += current_step_total_loss.detach().item() # Accumulate for interval average

        if (global_step + 1) % accum_iter == 0:
            if student_text_encoder.get_input_embeddings().weight.grad is not None:
                grad_norm_val = student_text_encoder.get_input_embeddings().weight.grad[index_updates, :].norm(dim=-1).mean().item()
                if writer: writer.add_scalar('Gradients/text_embedding_norm_inversion', grad_norm_val, global_step)
            else:
                 if writer: writer.add_scalar('Gradients/text_embedding_norm_inversion', 0.0, global_step) # Log 0 if no grad
                 print(f"Step {global_step}: WARNING: No gradient for text embeddings during TI update.")

            optimizer.step()
            lr_scheduler_main.step() # Step LR scheduler after optimizer step
            optimizer.zero_grad()  # Zero gradients for the next accumulation cycle or step

            with torch.no_grad():
                if clip_ti_decay:
                    pre_norm = (student_text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1, keepdim=True))
                    lambda_ = min(1.0, 100 * lr_scheduler_main.get_last_lr()[0])
                    student_text_encoder.get_input_embeddings().weight[index_updates] = F.normalize(
                        student_text_encoder.get_input_embeddings().weight[index_updates, :], dim=-1
                    ) * (pre_norm + lambda_ * (0.4 - pre_norm)) # Target norm 0.4

                current_norm_val = (student_text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1)).mean().item()
                if writer: writer.add_scalar('Embeddings/current_norm_inversion', current_norm_val, global_step)
                student_text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

        global_step += 1
        progress_bar.update(1)
        current_lr_val = lr_scheduler_main.get_last_lr()[0] # Get LR after step
        
        logs_postfix = {"loss_inv_distill": current_step_total_loss.detach().item(), "lr_inv": current_lr_val}
        if loss_t1_val is not None: logs_postfix["l_t1_inv"] = loss_t1_val
        if loss_t2_val is not None: logs_postfix["l_t2_inv"] = loss_t2_val
        progress_bar.set_postfix(**logs_postfix)

        if writer:
            writer.add_scalar('Loss/inversion_step_total_distill', current_step_total_loss.detach().item(), global_step)
            if loss_t1_val is not None: writer.add_scalar('Loss/inversion_step_distill_t1', loss_t1_val, global_step)
            if loss_t2_val is not None: writer.add_scalar('Loss/inversion_step_distill_t2', loss_t2_val, global_step)
            writer.add_scalar('LearningRate/inversion', current_lr_val, global_step)

        if global_step % save_steps == 0 and global_step > 0:
            current_save_dir = os.path.join(save_path, out_name)
            if not os.path.exists(current_save_dir): os.makedirs(current_save_dir, exist_ok=True)
            
            # Assuming save_all is defined elsewhere and handles saving TI embeddings
            if 'save_all' in globals() and callable(globals()['save_all']):
                globals()['save_all'](
                    unet=student_unet, text_encoder=student_text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(current_save_dir, f"step_inv_{global_step}.safetensors"),
                    save_lora=False, # Important for TI
                )
            else:
                print(f"Warning: save_all function not found. Skipping model save at step {global_step}.")


            avg_loss_interval = loss_sum_interval / save_steps if save_steps > 0 else 0.0
            wandb_logs_step = {"loss_inv_avg_interval": avg_loss_interval}
            if writer: writer.add_scalar('Loss/inversion_avg_interval', avg_loss_interval, global_step)
            loss_sum_interval = 0.0

            if log_wandb:
                # Ensure evaluate_pipe and other wandb logging components are defined/imported
                # For brevity, I'm omitting the full wandb image logging part here
                # It should be similar to your original code.
                # wandb.log(wandb_logs_step, step=global_step) # step kwarg for wandb
                pass 
            torch.cuda.empty_cache()

        if global_step >= num_steps: break
    
    if writer: writer.close()
    progress_bar.close()
    print("Inversion training (with distillation) finished.")
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir): os.makedirs(final_save_dir, exist_ok=True)
    if 'save_all' in globals() and callable(globals()['save_all']):
        globals()['save_all'](
            unet=student_unet, text_encoder=student_text_encoder,
            placeholder_token_ids=placeholder_token_ids,
            placeholder_tokens=placeholder_tokens,
            save_path=os.path.join(final_save_dir, f"final_step_inv_{global_step}.safetensors"),
            save_lora=False,
        )
        print(f"Final inversion model saved to {os.path.join(final_save_dir, f'final_step_inv_{global_step}.safetensors')}")
    else:
        print("Warning: save_all function not found. Skipping final model save.")

import torch
import os
import math # Added for isnan/isinf checks
from tqdm import tqdm
import itertools
from torch.utils.tensorboard import SummaryWriter # Ensure this is imported

# Assume these functions are defined elsewhere:
# loss_step_distill_lora, save_all (or it's in globals())

def perform_tuning(
    # --- "Teacher" 模型参数 ---
    teacher1_unet, teacher2_unet, teacher1_text_encoder, teacher2_text_encoder,
    # --- "Student" 模型参数 ---
    student_unet, vae, student_text_encoder, # LoRA parts are trainable
    # --- Dataloader 参数 ---
    dataloader1, dataloader2,
    # --- 核心训练参数 ---
    num_steps, scheduler, optimizer, save_steps: int, # optimizer for LoRA params
    placeholder_token_ids, placeholder_tokens, save_path, # 主保存路径
    # --- 学习率调度器 ---
    lr_scheduler_lora, # LoRA phase LR scheduler
    # --- LoRA 配置 ---
    lora_unet_target_modules, lora_clip_target_modules,
    # --- Attention Control & Cache (for distillation) ---
    teacher_attention_controller1, teacher_cache1,
    teacher_attention_controller2, teacher_cache2,
    student_attention_controller_main, student_cache_main,
    # --- Distillation Loss Weights ---
    distill_loss_weights: dict, # Expected to contain all w_* and alpha_teacher1
    distillation_strategy: str,
    # 这个参数现在控制 loss_step_distill_lora 内部是否对学生模型使用 autocast
    mixed_precision_unet_for_distill: bool,
    # --- 其他训练参数 ---
    mask_temperature: float, # Passed to loss_step
    out_name: str, # 用于TensorBoard日志目录 和 模型保存子目录
    tokenizer, # Retained for interface
    cached_latents: bool, # Retained for interface
    log_wandb: bool = False, # Retained for interface
    wandb_log_prompt_cnt: int = 10, # Retained for interface
    class_token: str = "person", # Retained for interface
    train_inpainting: bool = False, # Retained for interface
    tensorboard_log_dir: str = "runs", # TensorBoard日志的基础目录
    # --- 新增蒸馏相关参数 ---
    t_multiplier_for_loss_step: float = 1.0,
    save_image_every_n_steps_for_loss_step: int = 200,
):
    """
    执行LoRA蒸馏训练的主函数，模仿第一个perform_tuning的结构进行修改。
    (移除了外部 GradScaler 和外部 autocast)
    """

    # Debug: Verify parameters are set for training
    trainable_params_unet = sum(p.numel() for p in student_unet.parameters() if p.requires_grad)
    trainable_params_text_encoder = sum(p.numel() for p in student_text_encoder.parameters() if p.requires_grad)
    print(f"Trainable UNet parameters: {trainable_params_unet}")
    print(f"Trainable Text Encoder parameters: {trainable_params_text_encoder}")
    
    if student_unet.dtype != torch.float32 or student_text_encoder.dtype != torch.float32:
        print(f"WARNING: Student UNet dtype: {student_unet.dtype}, Student Text Encoder dtype: {student_text_encoder.dtype}. "
              "For stable training without external GradScaler, FP32 parameters are recommended.")

    # ========== 初始化和验证 ==========
    def validate_loss_weights(distill_loss_weights_val):
        required_keys = ["w_noise_pred", "w_text_embed", "w_feat_q", "w_feat_k", "w_feat_v"]
        for key in required_keys:
            lora_key = f"{key}_lora"
            weight = distill_loss_weights_val.get(lora_key, distill_loss_weights_val.get(key, 0.0))
            if not (isinstance(weight, (int, float)) and not math.isnan(weight) and not math.isinf(weight) and 0 <= weight <= 10.0):
                print(f"Warning: Unusual or invalid weight for {key}: {weight}. Using default 0.0 or 1.0.")
    
    validate_loss_weights(distill_loss_weights)
    
    # 不再使用外部 GradScaler
    scaler = None
    if mixed_precision_unet_for_distill:
        print("LoRA Tuning: Intending to use mixed precision via internal autocast in loss_step.")
    else:
        print("LoRA Tuning: Not using mixed precision (no internal autocast in loss_step).")
    
    tb_log_path = os.path.join(tensorboard_log_dir, out_name + "_lora_tuning")
    if not os.path.exists(tb_log_path): 
        os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging for LoRA tuning to: {tb_log_path}")

    loss_step_image_output_dir = os.path.join(save_path, out_name, "distill_debug_images_lora_tuning")
    if not os.path.exists(loss_step_image_output_dir): 
        os.makedirs(loss_step_image_output_dir, exist_ok=True)
    print(f"Debug images from loss_step_distill_lora (LoRA tuning) will be saved to: {loss_step_image_output_dir}")

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description(f"LoRA Tuning (Distill - {distillation_strategy}) Steps")
    global_step = 0

    student_unet.train()
    student_text_encoder.train()
    teacher1_unet.eval()
    teacher2_unet.eval()
    teacher1_text_encoder.eval()
    teacher2_text_encoder.eval()
    vae.eval()

    iter_dataloader1 = iter(dataloader1)
    iter_dataloader2 = iter(dataloader2) if dataloader2 is not None and dataloader2 != dataloader1 else iter_dataloader1
    
    alpha_t1 = distill_loss_weights.get("alpha_teacher1", 0.5)
    alpha_t2 = 1.0 - alpha_t1
    
    nan_count = 0
    max_nan_tolerance = 10

    def _calculate_loss_using_distill_fn_lora(current_batch, current_teacher_unet, current_teacher_text_encoder,
                                              current_teacher_controller, current_teacher_cache):
        try:
            return loss_step_distill_lora(
                batch=current_batch,
                teacher_unet=current_teacher_unet,
                teacher_text_encoder=current_teacher_text_encoder,
                student_unet=student_unet,
                student_text_encoder=student_text_encoder,
                scheduler=scheduler, 
                vae=vae,
                teacher_attention_controller=current_teacher_controller, 
                teacher_cache=current_teacher_cache,
                student_attention_controller=student_attention_controller_main, 
                student_cache=student_cache_main,
                # 传递权重
                w_text_embed=distill_loss_weights.get("w_text_embed_lora", distill_loss_weights.get("w_text_embed", 0.1)),
                w_noise_pred=distill_loss_weights.get("w_noise_pred_lora", distill_loss_weights.get("w_noise_pred", 1.0)),
                w_feat_q=distill_loss_weights.get("w_feat_q_lora", distill_loss_weights.get("w_feat_q", 0.1)),
                w_feat_k=distill_loss_weights.get("w_feat_k_lora", distill_loss_weights.get("w_feat_k", 0.1)),
                w_feat_v=distill_loss_weights.get("w_feat_v_lora", distill_loss_weights.get("w_feat_v", 0.1)),
                w_feat_self_out=distill_loss_weights.get("w_feat_self_out_lora", distill_loss_weights.get("w_feat_self_out", 0.1)),
                w_feat_ad=distill_loss_weights.get("w_feat_ad_lora", distill_loss_weights.get("w_feat_ad", 0.1)),
                ad_loss_attn_scale=distill_loss_weights.get("ad_loss_attn_scale", 1.0),
                global_step=global_step,
                save_image_every_n_steps=save_image_every_n_steps_for_loss_step,
                output_dir_for_loss_step=loss_step_image_output_dir,
                t_mutliplier=t_multiplier_for_loss_step,
                # mixed_precision_unet_for_distill 控制 loss_step_distill_lora 内部的 autocast
                mixed_precision_unet=mixed_precision_unet_for_distill,
                mask_temperature=mask_temperature,
                tuning=True,
            )
        except Exception as e:
            print(f"Error in LoRA loss calculation at step {global_step}: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(float('nan'), device=student_unet.device, requires_grad=True)

    def safe_get_next_batch(data_iter, dataloader, loader_name):
        try:
            batch = next(data_iter)
            return batch, data_iter
        except StopIteration:
            print(f"Dataloader {loader_name} exhausted and re-initialized at step {global_step}.")
            new_iter = iter(dataloader)
            try:
                batch = next(new_iter)
                return batch, new_iter
            except StopIteration:
                print(f"ERROR: Dataloader {loader_name} is empty even after re-initialization. Stopping.")
                raise
    
    for step_idx in range(num_steps):
        try:
            batch1, iter_dataloader1 = safe_get_next_batch(iter_dataloader1, dataloader1, "dataloader1")
            if dataloader2 is not None and dataloader2 != dataloader1:
                batch2, iter_dataloader2 = safe_get_next_batch(iter_dataloader2, dataloader2, "dataloader2")
            else:
                batch2 = batch1

            optimizer.zero_grad(set_to_none=True)
            loss_t1, loss_t2, total_loss = None, None, None

            # 损失计算不再包裹在外部 autocast 中
            if distillation_strategy == "weighted_average":
                loss_t1 = _calculate_loss_using_distill_fn_lora(
                    batch1, teacher1_unet, teacher1_text_encoder,
                    teacher_attention_controller1, teacher_cache1
                )
                if dataloader2 is not None:
                     loss_t2 = _calculate_loss_using_distill_fn_lora(
                        batch2, teacher2_unet, teacher2_text_encoder,
                        teacher_attention_controller2, teacher_cache2
                    )
                else:
                    loss_t2 = torch.tensor(0.0, device=loss_t1.device if torch.is_tensor(loss_t1) else student_unet.device)

                if torch.is_tensor(loss_t1) and (torch.isnan(loss_t1).any() or torch.isinf(loss_t1).any()) or \
                   (torch.is_tensor(loss_t2) and (torch.isnan(loss_t2).any() or torch.isinf(loss_t2).any())):
                    print(f"Warning: NaN/Inf loss from distill_fn at step {global_step}. Skipping.")
                    nan_count += 1
                    if nan_count > max_nan_tolerance: break
                    continue
                
                if torch.is_tensor(loss_t1) and torch.is_tensor(loss_t2):
                    total_loss = alpha_t1 * loss_t1 + alpha_t2 * loss_t2
                elif torch.is_tensor(loss_t1):
                    total_loss = loss_t1
                else: # Should not happen if error handling in _calculate_loss_using_distill_fn_lora is robust
                    print(f"Error: loss_t1 is not a tensor at G{global_step}. Skipping.")
                    nan_count += 1; continue


            elif distillation_strategy == "alternating":
                if global_step % 2 == 0:
                    total_loss = _calculate_loss_using_distill_fn_lora(
                        batch1, teacher1_unet, teacher1_text_encoder,
                        teacher_attention_controller1, teacher_cache1
                    )
                    loss_t1 = total_loss
                else:
                    if dataloader2 is None:
                         total_loss = _calculate_loss_using_distill_fn_lora(
                            batch1, teacher1_unet, teacher1_text_encoder, # Fallback to T1
                            teacher_attention_controller1, teacher_cache1
                        )
                         loss_t1 = total_loss # Log as t1
                    else:
                        total_loss = _calculate_loss_using_distill_fn_lora(
                            batch2, teacher2_unet, teacher2_text_encoder,
                            teacher_attention_controller2, teacher_cache2
                        )
                        loss_t2 = total_loss
                
                if not torch.is_tensor(total_loss) or torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    print(f"Warning: NaN/Inf loss (alternating) at step {global_step}. Skipping.")
                    nan_count += 1
                    if nan_count > max_nan_tolerance: break
                    continue
            else:
                raise ValueError(f"Unknown distillation_strategy: {distillation_strategy}")

            if not torch.is_tensor(total_loss) or torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"Skipping step {global_step} due to invalid total loss before backward: {total_loss}")
                nan_count += 1
                if nan_count > max_nan_tolerance: break
                continue

            # 直接反向传播
            total_loss.backward()

            # 梯度检查（在裁剪前）
            if global_step > 0 and global_step % 50 == 0: # 避免第0步且减少频率
                unet_grads_norms = [p.grad.norm().item() for n, p in student_unet.named_parameters()
                                   if p.requires_grad and p.grad is not None and hasattr(p.grad, 'norm')]
                text_grads_norms = [p.grad.norm().item() for n, p in student_text_encoder.named_parameters()
                                   if p.requires_grad and p.grad is not None and hasattr(p.grad, 'norm')]

                if unet_grads_norms:
                    avg_unet_grad_norm = sum(unet_grads_norms) / len(unet_grads_norms)
                    print(f"Step {global_step}: UNet grad norm (avg, pre-clip): {avg_unet_grad_norm:.6f}")
                    if writer: writer.add_scalar("Gradients/UNet_Avg_Norm_PreClip", avg_unet_grad_norm, global_step)
                # else: print(f"Step {global_step}: WARNING: No gradients in UNet pre-clip!") # 可选

                if text_grads_norms:
                    avg_text_grad_norm = sum(text_grads_norms) / len(text_grads_norms)
                    print(f"Step {global_step}: TextEnc grad norm (avg, pre-clip): {avg_text_grad_norm:.6f}")
                    if writer: writer.add_scalar("Gradients/TextEncoder_Avg_Norm_PreClip", avg_text_grad_norm, global_step)
                # else: print(f"Step {global_step}: WARNING: No gradients in Text Encoder pre-clip!") # 可选
            
            clipped_grad_norm_val = torch.nn.utils.clip_grad_norm_(
                itertools.chain(student_unet.parameters(), student_text_encoder.parameters()), 
                max_norm=1.0
            )

            if torch.isnan(clipped_grad_norm_val).any() or torch.isinf(clipped_grad_norm_val).any():
                print(f"Warning: Invalid gradient norm AFTER clipping at step {global_step}: {clipped_grad_norm_val}. Skipping optimizer step.")
                nan_count += 1
                if nan_count > max_nan_tolerance: break
                continue

            # 直接优化器步骤
            optimizer.step()
            lr_scheduler_lora.step()
            
            progress_bar.update(1)
            current_lr = lr_scheduler_lora.get_last_lr()[0]

            logs = {
                "loss": total_loss.detach().item(),
                "lr": current_lr,
                "grad_norm_clip": clipped_grad_norm_val.item() if torch.is_tensor(clipped_grad_norm_val) else clipped_grad_norm_val # handle non-tensor case
            }
            loss_t1_item = loss_t1.detach().item() if torch.is_tensor(loss_t1) and loss_t1 is not None else float('nan')
            loss_t2_item = loss_t2.detach().item() if torch.is_tensor(loss_t2) and loss_t2 is not None else float('nan')

            if not math.isnan(loss_t1_item): logs["loss_t1"] = loss_t1_item
            if not math.isnan(loss_t2_item) and distillation_strategy == "weighted_average" and dataloader2 is not None:
                 logs["loss_t2"] = loss_t2_item
            # (其他日志逻辑)
            if nan_count > 0: logs["nan_skips"] = nan_count
            progress_bar.set_postfix(**logs)

            if writer:
                writer.add_scalar('Loss_LoRA/total_distill', total_loss.detach().item(), global_step)
                if not math.isnan(loss_t1_item):
                     writer.add_scalar('Loss_LoRA/distill_t1', loss_t1_item, global_step)
                if not math.isnan(loss_t2_item) and distillation_strategy == "weighted_average" and dataloader2 is not None:
                     writer.add_scalar('Loss_LoRA/distill_t2', loss_t2_item, global_step)
                
                writer.add_scalar('LearningRate_LoRA/tuning', current_lr, global_step)
                if torch.is_tensor(clipped_grad_norm_val) and not (torch.isnan(clipped_grad_norm_val).any() or torch.isinf(clipped_grad_norm_val).any()):
                    writer.add_scalar('Gradients_LoRA/total_norm_clipped', clipped_grad_norm_val.item(), global_step)
                if nan_count > 0:
                     writer.add_scalar('Debug_LoRA/nan_skip_count', nan_count, global_step)

            global_step += 1

            if global_step > 0 and global_step % save_steps == 0:
                current_save_dir = os.path.join(save_path, out_name)
                if not os.path.exists(current_save_dir):
                    os.makedirs(current_save_dir, exist_ok=True)
                save_filename = f"step_lora_{global_step}.safetensors"
                save_full_path = os.path.join(current_save_dir, save_filename)
                
                save_fn_to_use = globals().get('save_all') # Assume save_all is globally available
                if save_fn_to_use and callable(save_fn_to_use):
                    try:
                        save_fn_to_use(
                            unet=student_unet, text_encoder=student_text_encoder,
                            placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
                            save_path=save_full_path, save_lora=True,
                            target_replace_module_text=lora_clip_target_modules,
                            target_replace_module_unet=lora_unet_target_modules,
                        )
                        print(f"Saved LoRA weights at tuning step {global_step} to {save_full_path}")
                    except Exception as e:
                        print(f"Error saving LoRA model at step {global_step}: {e}")
                else:
                    print(f"Warning: 'save_all' function not found. Skipping LoRA save at step {global_step}.")

            if global_step >= num_steps:
                print(f"Reached target number of steps ({num_steps}).")
                break
        
        except StopIteration:
            print(f"A dataloader became empty and could not be refilled. Stopping training at step {global_step}.")
            break
        except Exception as e:
            print(f"Major error in LoRA tuning loop at step {global_step}: {e}")
            import traceback
            traceback.print_exc()
            nan_count += 1
            if nan_count > max_nan_tolerance:
                print(f"Too many errors/NaNs ({nan_count}). Stopping LoRA tuning.")
                break
            continue 
    
    if writer: writer.close()
    progress_bar.close()
    
    print(f"LoRA tuning (with distillation) finished. Total steps: {global_step}. Total NaN/Error skips: {nan_count}")
    
    final_save_dir = os.path.join(save_path, out_name)
    if not os.path.exists(final_save_dir): 
        os.makedirs(final_save_dir, exist_ok=True)
    final_save_filename = f"final_step_lora_{global_step}.safetensors"
    final_save_full_path = os.path.join(final_save_dir, final_save_filename)

    save_fn_to_use = globals().get('save_all')
    if save_fn_to_use and callable(save_fn_to_use):
        try:
            save_fn_to_use(
                unet=student_unet, text_encoder=student_text_encoder,
                placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
                save_path=final_save_full_path, save_lora=True,
                target_replace_module_text=lora_clip_target_modules,
                target_replace_module_unet=lora_unet_target_modules,
            )
            print(f"Final LoRA model saved to {final_save_full_path}")
        except Exception as e:
            print(f"Error saving final LoRA model: {e}")
    else:
        print("Warning: 'save_all' function not found. Skipping final LoRA model save.")
    
    return global_step, nan_count


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
    distillation_strategy: str = "weighted_average",
    attention_layers_to_monitor_range: tuple = (0, 16),
    distill_loss_weights_config_ti: Optional[dict] = None, # For Textual Inversion
    distill_loss_weights_config_lora: Optional[dict] = None, # For LoRA tuning
    mixed_precision_unet_for_distill: bool = True,
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
        dataset_size=100, # 例如生成20张
        transform_size=512,
        h_flip=True,
        aux_tokenizer1=student_tokenizer,

        # 新增参数以保存图片
        save_generated_images_path="/root/lora_train/pic_new1",
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
        dataset_size=100, # 例如生成20张
        transform_size=512,
        h_flip=True,
        aux_tokenizer1=student_tokenizer,

        # 新增参数以保存图片
        save_generated_images_path="/root/lora_train/pic_new2",
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

    # --- 1. Init Controllers & Caches ---
    print("Initializing Attention Controllers and DataCaches for distillation...")
    teacher1_attention_controller = Controller(self_layers_range=attention_layers_to_monitor_range)
    teacher1_data_cache = DataCache()
    teacher2_attention_controller = Controller(self_layers_range=attention_layers_to_monitor_range)
    teacher2_data_cache = DataCache()
    student_attention_controller = Controller(self_layers_range=attention_layers_to_monitor_range)
    student_data_cache = DataCache()

    # --- 2. Register Hooks (ONCE before training loops) ---
    # Ensure UNets are on device and teacher UNets are in eval mode
    teacher1_unet.to(device).eval().requires_grad_(False)
    teacher2_unet.to(device).eval().requires_grad_(False)
    student_unet.to(device) # student_unet mode (train/eval) will be set in TI/LoRA functions

    print("Registering attention control for Teacher1 UNet...")
    register_attn_control(teacher1_unet, teacher1_attention_controller, teacher1_data_cache)
    print("Registering attention control for Teacher2 UNet...")
    register_attn_control(teacher2_unet, teacher2_attention_controller, teacher2_data_cache)
    print("Registering attention control for Student UNet...")
    register_attn_control(student_unet, student_attention_controller, student_data_cache)

    # Define default distillation loss weights if not provided
    if distill_loss_weights_config_ti is None:
        distill_loss_weights_config_ti = {
            "w_text_embed": 1.0, "w_noise_pred": 0.1, # Noise pred less critical for TI
            "w_feat_q": 0.02, "w_feat_k": 0.02, "w_feat_v": 0.02,
            "w_feat_self_out": 0.02, "w_feat_ad": 0.02,
            "ad_loss_attn_scale": 1.0, "alpha_teacher1": 0.5,
        }
    if distill_loss_weights_config_lora is None:
        distill_loss_weights_config_lora = {
            "w_text_embed": 0.5, "w_noise_pred": 1.0, # Noise pred more important for LoRA
            "w_feat_q": 0.1, "w_feat_k": 0.1, "w_feat_v": 0.1,
            "w_feat_self_out": 0.1, "w_feat_ad": 0.1,
            "ad_loss_attn_scale": 1.0, "alpha_teacher1": 0.5,
        }

    # STEP 1 : Perform Inversion
    if perform_inversion:
        print("Starting Textual Inversion with Attention Distillation...")

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
            # ... (pass all existing necessary params for train_inversion) ...
            teacher1_unet=teacher1_unet, teacher2_unet=teacher2_unet,
            teacher1_text_encoder=teacher1_text_encoder, teacher2_text_encoder=teacher2_text_encoder,
            student_unet=student_unet, vae=student_vae, student_text_encoder=student_text_encoder,
            dataloader1=train_dataloader1, dataloader2=train_dataloader2,
            num_steps=max_train_steps_ti, scheduler=noise_scheduler, index_no_updates=index_no_updates,
            optimizer=ti_optimizer, save_steps=save_steps,
            placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
            save_path=output_dir, lr_scheduler_main=lr_scheduler,
            lora_unet_target_modules=lora_unet_target_modules, lora_clip_target_modules=lora_clip_target_modules,
            out_name=out_name, tokenizer=student_tokenizer, test_image_path=instance_data_dir,
            cached_latents=cached_latents, mask_temperature=mask_temperature, # from train_inversion defaults
            accum_iter=gradient_accumulation_steps, log_wandb=log_wandb, # from train_inversion defaults
            # ... other train_inversion specific params ...
            # NEW PARAMS FOR DISTILLATION
            teacher_attention_controller1=teacher1_attention_controller, teacher_cache1=teacher1_data_cache,
            teacher_attention_controller2=teacher2_attention_controller, teacher_cache2=teacher2_data_cache,
            student_attention_controller_main=student_attention_controller, student_cache_main=student_data_cache,
            distill_loss_weights=distill_loss_weights_config_ti, # Use TI specific weights
            distillation_strategy=distillation_strategy,
            mixed_precision_unet_for_distill=mixed_precision_unet_for_distill, # Pass this through
            # Pass params for loss_step_distill_lora's image saving if different from train_inversion's defaults
            # save_image_every_n_steps_loss=200, # Example: could be a new arg to train()
            # t_multiplier_loss=1.0,          # Example: could be a new arg to train()
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

    print("Starting LoRA Tuning with Attention Distillation...")

    perform_tuning(
            # ... (pass all existing necessary params for perform_tuning) ...
            teacher1_unet=teacher1_unet, teacher2_unet=teacher2_unet,
            teacher1_text_encoder=teacher1_text_encoder, teacher2_text_encoder=teacher2_text_encoder,
            student_unet=student_unet, vae=student_vae, student_text_encoder=student_text_encoder,
            dataloader1=train_dataloader1, dataloader2=train_dataloader2,
            num_steps=max_train_steps_tuning, scheduler=noise_scheduler, optimizer=lora_optimizers,
            save_steps=save_steps, placeholder_token_ids=placeholder_token_ids, placeholder_tokens=placeholder_tokens,
            save_path=output_dir, lr_scheduler_lora=lr_scheduler_lora,
            lora_unet_target_modules=lora_unet_target_modules, lora_clip_target_modules=lora_clip_target_modules,
            mask_temperature=mask_temperature, out_name=out_name, tokenizer=student_tokenizer,
            cached_latents=cached_latents, log_wandb=log_wandb,
            # ... other perform_tuning specific params ...
            # NEW PARAMS FOR DISTILLATION
            teacher_attention_controller1=teacher1_attention_controller, teacher_cache1=teacher1_data_cache,
            teacher_attention_controller2=teacher2_attention_controller, teacher_cache2=teacher2_data_cache,
            student_attention_controller_main=student_attention_controller, student_cache_main=student_data_cache,
            distill_loss_weights=distill_loss_weights_config_lora, # Use LoRA specific weights
            distillation_strategy=distillation_strategy,
            mixed_precision_unet_for_distill=mixed_precision_unet_for_distill,
            # save_image_every_n_steps_loss=200, # Example
            # t_multiplier_loss=1.0,          # Example
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
