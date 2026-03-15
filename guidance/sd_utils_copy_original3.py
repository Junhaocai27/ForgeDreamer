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

# 强制关闭 PyTorch 原生的 FlashAttention 和 MemEfficientAttention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# 强制开启最原始、最精确的 Math 计算模式 (对应旧环境的隐式回退)
torch.backends.cuda.enable_math_sdp(True)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard导入
from .hypergraph_enhancer import StaticGradientHypergraphEnhancer, ImprovedStaticDHGLatentHypergraph, SimplifiedDHGLatentHypergraph, DirectSimilarityDHGLatentHypergraph
from .mask_utils import SubjectMaskGenerator

from .sd_step import *

# def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
#     """
#     【增强版 HSV 蒙版生成器】适用于白色背景图像中物体的分割。
#     """
#     if verbose:
#         print("--- [generate_advanced_mask_hsv] 使用HSV分割白色背景物体 ---")

#     if original_image.ndim != 3 or original_image.shape[2] != 3:
#         if verbose: print("错误: 输入图像必须是3通道BGR格式。")
#         return None

#     height, width = original_image.shape[:2]

#     # 转换为 HSV 色彩空间
#     hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     # --- 步骤 1: 白色背景剔除 ---
#     # 白色区域的 S 非常低，V 很高，我们用它来判断背景
#     # 白色条件: S < 30 且 V > 200
#     white_bg_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
#     fg_mask = cv2.bitwise_not(white_bg_mask)  # 主体区域 = 非白色区域

#     if verbose:
#         white_ratio = np.mean(white_bg_mask > 0)
#         print(f"[HSV] 背景白色比例: {white_ratio*100:.2f}%")

#     # --- 步骤 2: 边缘 & 连通区域增强 ---
#     edges = cv2.Canny(fg_mask, 50, 150)
#     dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
#     fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

#     # --- 步骤 3: 形态学处理 ---
#     kernel_size = max(3, min(9, int(min(height, width) / 200)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     if verbose:
#         print(f"形态学处理完成，核大小: {kernel_size}")

#     # --- 步骤 4: 连通域分析保留最大主体 ---
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
#     if num_labels <= 1:
#         if verbose: print("未找到主体区域，返回初步掩码。")
#         return fg_mask
    
#     largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     final_mask = np.zeros_like(fg_mask)
#     final_mask[labels == largest_component] = 255
#     if verbose: print(f"保留最大连通域作为主体，面积: {stats[largest_component, cv2.CC_STAT_AREA]}")

#     # --- 步骤 5: 轮廓优化 & 模糊边缘 ---
#     contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         if verbose: print("未找到轮廓，直接返回。")
#         return final_mask

#     refined_mask = np.zeros_like(final_mask)
#     cv2.drawContours(refined_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

#     refined_mask = cv2.dilate(refined_mask, np.ones((3, 3), np.uint8), iterations=1)
#     refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)

#     if verbose: print("边缘优化完成，生成最终掩码。")
#     return refined_mask

# def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
#     """
#     【修改版】适用于白色背景图像中物体的分割，生成一个值为 0 和 1 的二值掩码。
#     """
#     if verbose:
#         print("--- [generate_advanced_mask_hsv] 使用HSV分割白色背景物体 ---")

#     if original_image.ndim != 3 or original_image.shape[2] != 3:
#         if verbose: print("错误: 输入图像必须是3通道BGR格式。")
#         return None

#     height, width = original_image.shape[:2]

#     # 转换为 HSV 色彩空间
#     hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
#     # --- 步骤 1: 白色背景剔除 ---
#     # 白色区域的 S 非常低，V 很高
#     white_bg_mask = cv2.inRange(hsv, (0, 0, 240), (180, 10, 255))
#     fg_mask = cv2.bitwise_not(white_bg_mask)  # 主体区域 = 非白色区域 (0/255)

#     if verbose:
#         white_ratio = np.mean(white_bg_mask > 0)
#         print(f"[HSV] 背景白色比例: {white_ratio*100:.2f}%")

#     # --- 步骤 2: 边缘 & 连通区域增强 ---
#     edges = cv2.Canny(fg_mask, 50, 150)
#     dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
#     fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

#     # --- 步骤 3: 形态学处理 ---
#     kernel_size = max(3, min(9, int(min(height, width) / 200)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     if verbose:
#         print(f"形态学处理完成，核大小: {kernel_size}")

#     # --- 步骤 4: 连通域分析保留最大主体 ---
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
#     if num_labels <= 1:
#         if verbose: print("未找到主体区域，返回初步清理后的掩码。")
#         # 【修改点 1】: 如果提前返回，也要确保是 0/1 格式
#         return (fg_mask / 255).astype(np.uint8)
    
#     largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     final_mask_255 = np.zeros_like(fg_mask)
#     final_mask_255[labels == largest_component] = 255
#     if verbose: print(f"保留最大连通域作为主体，面积: {stats[largest_component, cv2.CC_STAT_AREA]}")

#     # --- 步骤 5: 转换为 0/1 二值掩码 ---
#     # 【修改点 2】: 移除了原有的轮廓平滑和高斯模糊步骤。
#     # 用简单的除法和类型转换来生成 0/1 掩码。
#     final_mask_01 = (final_mask_255 / 255).astype(np.uint8)

#     if verbose: print("生成 0/1 二值掩码完成。")
#     return final_mask_01

def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    """
    修改版：适用于白色背景图像中物体的分割，返回 0/1 二值掩码（无最大连通域筛选）。
    """
    if verbose:
        print("--- [generate_advanced_mask_hsv] 使用HSV分割白色背景物体 ---")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("错误: 输入图像必须是3通道BGR格式。")
        return None

    height, width = original_image.shape[:2]

    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # 步骤 1: 剔除白色背景（低饱和度 + 高亮度）
    white_bg_mask = cv2.inRange(hsv, (0, 0, 240), (180, 10, 255))
    fg_mask = cv2.bitwise_not(white_bg_mask)  # 主体区域 = 非白色区域 (0/255)

    if verbose:
        white_ratio = np.mean(white_bg_mask > 0)
        print(f"[HSV] 背景白色比例: {white_ratio*100:.2f}%")

    # 步骤 2: 边缘增强（可选）
    edges = cv2.Canny(fg_mask, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

    # 步骤 3: 形态学处理（噪声去除 + 缝隙填补）
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if verbose:
        print(f"形态学处理完成，核大小: {kernel_size}")

    # 步骤 4: 直接返回处理后的前景掩码（不再使用最大连通域）
    final_mask_01 = (fg_mask / 255).astype(np.uint8)

    if verbose: print("生成 0/1 二值掩码完成（未使用连通域筛选）。")
    return final_mask_01


def generate_advanced_mask(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    """
    【核心蒙版生成器】从一个numpy数组格式的图像中分割出纯色背景下的物体。
    该函数不执行任何文件I/O，使其易于在其他代码中重用。

    参数:
    original_image (np.ndarray): 输入的BGR图像数组 (形状为 H, W, 3), dtype=uint8。
    verbose (bool): 是否打印详细的处理步骤信息。

    返回:
    np.ndarray | None: 
        - 最终的单通道灰度蒙版 (形状为 H, W), dtype=uint8 (值为 0-255)。
        - 如果处理失败，则返回 None。
    """
    if verbose:
        print("--- [generate_advanced_mask] 开始高级分割 ---")
    
    # 确保图像是3通道的
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("错误: 输入图像必须是3通道BGR格式。")
        return None

    height, width = original_image.shape[:2]

    # --- 步骤 2: 多通道分析选择最佳通道 ---
    b, g, r = cv2.split(original_image)
    channels = [b, g, r]
    channel_names = ['Blue', 'Green', 'Red']
    variances = [np.var(channel) for channel in channels]
    best_channel_idx = np.argmax(variances)
    best_channel = channels[best_channel_idx]
    if verbose: print(f"选择了{channel_names[best_channel_idx]}通道进行分割（方差: {variances[best_channel_idx]:.2f}）")

    # --- 步骤 3: 边缘检测辅助分割 ---
    median_val = np.median(best_channel)
    lower_thresh = max(0, int(0.5 * median_val))
    upper_thresh = min(255, int(1.5 * median_val))
    edges = cv2.Canny(best_channel, lower_thresh, upper_thresh)
    if verbose: print(f"边缘检测阈值: {lower_thresh}-{upper_thresh}")

    # --- 步骤 4: 改进的Otsu二值化 ---
    blurred = cv2.GaussianBlur(best_channel, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- 步骤 5: 结合边缘信息改进蒙版 ---
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(otsu_mask, dilated_edges)
    
    # --- 步骤 6: 智能形态学处理 ---
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if verbose: print(f"使用{kernel_size}x{kernel_size}椭圆核进行形态学处理...")

    # --- 步骤 7: 连通域分析保留最大区域 ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    if num_labels <= 1:
        if verbose: print("未发现有效的连通域，可能分割失败。")
        # 如果没有找到物体，返回一个全黑的蒙版或者当前处理结果
        return cleaned_mask
        
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(cleaned_mask)
    final_mask[labels == largest_component] = 255
    if verbose: print(f"保留最大连通域，面积: {stats[largest_component, cv2.CC_STAT_AREA]} 像素")

    # --- 步骤 8 & 10 (合并): 边界优化和平滑 ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose: print("最终蒙版中未找到轮廓。")
        return final_mask # 返回之前的蒙版

    largest_contour = max(contours, key=cv2.contourArea)
    refined_mask = np.zeros_like(final_mask)
    cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # 轻微膨胀以确保边缘完整性，然后模糊以平滑边缘
    kernel_dilate = np.ones((3, 3), np.uint8) # 使用较小的核进行膨胀
    refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0) # 最终的平滑处理
    
    if verbose: print("已优化并平滑物体边界...")
    
    # 返回最终的、高质量的单通道蒙版 (主体为白色 255, 背景为黑色 0)
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

        # ====================  在这里添加超图模块初始化 ====================
        self.use_hypergraph = getattr(guidance_opt, 'use_hypergraph', False)  # 默认使用超图增强
        if self.use_hypergraph:
            self.hypergraph_enhancer = StaticGradientHypergraphEnhancer(
                patch_size=getattr(guidance_opt, 'hg_patch_size', 4),
                alpha=getattr(guidance_opt, 'hg_alpha', 0.6),
                similarity_threshold=getattr(guidance_opt, 'hg_sim_thresh', 0.8),
                device=self.device # 传递设备信息
            )
        else:
            self.hypergraph_enhancer = None
        # ===================================================================

        # ====================  新增DHG Latent超图模块初始化 ====================
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

        # 添加课程学习参数
        self.curriculum_schedule = getattr(guidance_opt, 'curriculum_schedule', 'linear')
        self.curriculum_phases = getattr(guidance_opt, 'curriculum_phases', ['coarse', 'medium', 'fine'])
        self.phase_transitions = getattr(guidance_opt, 'phase_transitions', [1500, 2500, 5000])
        
        # 时间步范围定义
        self.timestep_ranges = {
            'coarse': [0.7, 0.98],     # 高噪声，学习粗粒度结构
            'medium': [0.3, 0.8],      # 中等噪声，学习中等细节
            'fine': [0.02, 0.5],       # 低噪声，学习精细细节
            'adaptive': [0.02, 0.98]   # 自适应范围
        }

        # 添加TensorBoard writer
        self.writer = SummaryWriter(log_dir=guidance_opt.tensorboard_log_dir if hasattr(guidance_opt, 'tensorboard_log_dir') else './runs/sd_training_new')

        model_key = "/root/LucidDreamer/stable-diffusion-2-1-base"
        assert model_key is not None

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

        # --- 新增以下代码，强制回退到基础 Attention 处理器 ---
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

        # ==================== MASKING SETUP ====================
        # self.use_subject_mask = use_subject_mask
        # self.mask_strategy = mask_strategy
        # self.mask_on_subject = mask_on_subject
        # self.subject_mask_generator = None
        # self.mask_cache = {}  # 缓存: { image_index: latent_mask_tensor }

        # if self.use_subject_mask:
        #     print("[SD Init] Masking is enabled. Initializing SubjectMaskGenerator.")
        #     self.subject_mask_generator = SubjectMaskGenerator(
        #         self.device,
        #         sam_checkpoint_path=sam_checkpoint_path
        #     )
        #     # 检查SAM是否成功加载
        #     if self.subject_mask_generator.mask_generator is None:
        #         print("[SD Init] WARN: SAM model failed to load. Disabling subject masking for this run.")
        #         self.use_subject_mask = False
        #     else:
        #         # 从 guidance_opt 获取图像尺寸，用于计算 latent 尺寸
        #         H = getattr(guidance_opt, 'H', 512)
        #         W = getattr(guidance_opt, 'W', 512)
        #         self.latent_height = H // 8
        #         self.latent_width = W // 8
        #         print(f"[SD Init] Masking configured for latent space of size ({self.latent_height}, {self.latent_width}).")
        #         print(f"[SD Init] Masking strategy: '{self.mask_strategy}'. Mask on subject: {self.mask_on_subject}.")

        # print(f'[INFO] Stable Diffusion class initialized.')

        # self.use_subject_mask = use_subject_mask
        # self.mask_strategy = mask_strategy
        # self.mask_on_subject = mask_on_subject
        # self.subject_mask_generator = None
        # self.mask_cache = {}  # 缓存: { image_index: latent_mask_tensor }

        # if self.use_subject_mask:
        #     print("[SD Init] Masking is enabled. Initializing SubjectMaskGenerator with FastSAM.")
        #     self.subject_mask_generator = SubjectMaskGenerator(
        #         self.device,
        #         fastsam_checkpoint_path="/root/LucidDreamer/FastSAM/FastSAM.pt"
        #     )
        #     # 检查FastSAM是否成功加载
        #     if self.subject_mask_generator.mask_generator is None:
        #         print("[SD Init] WARN: FastSAM model failed to load. Disabling subject masking for this run.")
        #         self.use_subject_mask = False
        #     else:
        #         # 从 guidance_opt 获取图像尺寸，用于计算 latent 尺寸
        #         H = getattr(guidance_opt, 'H', 512)
        #         W = getattr(guidance_opt, 'W', 512)
        #         self.latent_height = H // 8
        #         self.latent_width = W // 8
        #         print(f"[SD Init] FastSAM masking configured for latent space of size ({self.latent_height}, {self.latent_width}).")
        #         print(f"[SD Init] Masking strategy: '{self.mask_strategy}'. Mask on subject: {self.mask_on_subject}.")

        # print(f'[INFO] Stable Diffusion class initialized with FastSAM masking.')

        # ==================== 蒙版设置 (修改后) ====================
        self.use_subject_mask = use_subject_mask
        self.mask_strategy = mask_strategy
        self.mask_on_subject = mask_on_subject
        # self.subject_mask_generator = None  # 这个生成器对象不再需要
        self.mask_cache = {}

        if self.use_subject_mask:
            # 现在我们默认使用 'advanced' 策略，无需在此处加载复杂的模型
            H = getattr(guidance_opt, 'H', 512)
            W = getattr(guidance_opt, 'W', 512)
            self.latent_height = H // 8
            self.latent_width = W // 8
            print(f"[SD Init] 蒙版功能已启用，策略: '{self.mask_strategy}'。")
            print(f"[SD Init] 用于蒙版的Latent空间尺寸: ({self.latent_height}, {self.latent_width}).")
            print(f"[SD Init] 在主体上应用蒙版: {self.mask_on_subject}。")
        else:
            print("[SD Init] 蒙版功能已禁用。")

        print("[INFO] Initializing Three-Phase Annealing Curriculum for timesteps and guidance.")
        
        # --- 新的课程学习参数 ---
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
        
        # (可选) 损失权重
        self.loss_weights = {
            'coarse': getattr(guidance_opt, 'loss_weight_coarse', [1.0, 1.0]),
            'refine': getattr(guidance_opt, 'loss_weight_refine', [1.0, 0.8]),
            'fine':   getattr(guidance_opt, 'loss_weight_fine', [0.8, 0.5]),
        }

    def get_biased_time_step(self, warm_up_rate):
        """
        根据训练进度 warm_up_rate 使用非均匀分布从时间步中采样 ind_t。
        高噪声时间步比例会随着 warm_up_rate 逐步增加。
        """

        # 时间步范围
        min_step = self.min_step
        max_step = self.max_step + int(self.warmup_step * warm_up_rate)
        total_steps = max_step - min_step + 1

        # 随着 warm_up_rate 引入更多高噪声比例（最多40%）
        high_ratio = min(0.1 + 0.3 * warm_up_rate, 0.4)
        low_ratio = 0.3
        mid_ratio = 1.0 - low_ratio - high_ratio

        # 各区间的范围
        low_range = (min_step, min_step + int(total_steps * low_ratio))
        mid_range = (low_range[1], low_range[1] + int(total_steps * mid_ratio))
        high_range = (mid_range[1], max_step + 1)

        # 根据比例进行采样
        rand_val = torch.rand(1).item()
        if rand_val < low_ratio:
            chosen_range = low_range
        elif rand_val < low_ratio + mid_ratio:
            chosen_range = mid_range
        else:
            chosen_range = high_range

        # 最终随机选出一个时间步
        ind_t = torch.randint(
            chosen_range[0], chosen_range[1],
            (1,), dtype=torch.long,
            generator=self.noise_gen,
            device=self.device
        )[0]

        return ind_t
    
    # <<< 新增的辅助函数 >>>
    def get_annealed_params(self, iteration):
        """
        根据当前迭代步数，计算退火后的时间步范围和指导强度。
        """
        # 1. 确定当前阶段
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

        # 2. 计算在当前阶段的进度 (0.0 to 1.0)
        # 防止除以零
        if phase_end_iter == phase_start_iter:
            progress = 1.0
        else:
            progress = (iteration - phase_start_iter) / (phase_end_iter - phase_start_iter)

        # 3. 线性插值 (lerp) 计算当前参数
        def lerp(start, end, progress):
            return start + progress * (end - start)

        # 获取当前阶段的参数范围
        t_range_start, t_range_end = self.t_ranges[phase]
        guidance_start, guidance_end = self.guidance_scales[phase]
        loss_weight_start, loss_weight_end = self.loss_weights[phase]

        # 计算当前值
        # 注意：对于 t_range，我们通常保持其在阶段内固定，而不是插值。
        # 如果需要，也可以对 t_range 进行退火，但从固定范围开始更简单。
        current_t_range = [t_range_start, t_range_end]
        current_guidance_scale = lerp(guidance_start, guidance_end, progress)
        current_loss_weight = lerp(loss_weight_start, loss_weight_end, progress)

        # 4. 根据 t_range 采样一个时间步
        min_step = int(self.num_train_timesteps * current_t_range[0])
        max_step = int(self.num_train_timesteps * current_t_range[1])
        
        # 确保 min_step 和 max_step 有效
        max_step = max(min_step + 1, max_step)
        
        ind_t = torch.randint(min_step, max_step, (1,), dtype=torch.long, device=self.device)[0]

        return ind_t, current_guidance_scale, current_loss_weight, phase


    # def _get_or_generate_latent_mask(self, image_tensor, image_index=None): # image_index 现在是可选的
    #     """
    #     内部辅助方法：根据输入的图像张量，生成并返回一个latent空间的掩码。
    #     (已移除缓存逻辑，以确保每次都为新图像生成新掩码)
    #     """
    #     # 动态计算目标 latent 尺寸
    #     # image_tensor 的形状是 [C, H, W]
    #     target_height, target_width = image_tensor.shape[1], image_tensor.shape[2]
    #     latent_height = target_height // 8
    #     latent_width = target_width // 8

    #     print(f"[Masking] Generating a new mask for image of size ({target_height}, {target_width}) -> latent size ({latent_height}, {latent_width})...")
        
    #     # 1. 转换Tensor为NumPy数组
    #     image_np_uint8 = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
    #     # 2. 生成全分辨率掩码
    #     full_res_mask_np = self.subject_mask_generator.generate_mask_from_np(
    #         image_np_uint8, strategy=self.mask_strategy)
        
    #     # 3. 处理生成失败
    #     if full_res_mask_np is None:
    #         print(f"[Masking] WARN: Mask generation FAILED. Using a passthrough (all ones) mask.")
    #         latent_mask = torch.ones((1, 1, latent_height, latent_width), device=self.device, dtype=self.precision_t)
    #         return latent_mask

    #     # 4. 转换回Tensor并降采样到正确的 latent 尺寸
    #     mask_tensor = torch.from_numpy(full_res_mask_np).to(self.device).float().unsqueeze(0).unsqueeze(0)
    #     latent_mask = F.interpolate(mask_tensor, size=(latent_height, latent_width), mode='nearest-exact')

    #     # 5. 处理掩码反转
    #     if not self.mask_on_subject:
    #         latent_mask = 1.0 - latent_mask

    #     # 6. 直接返回，不存入缓存
    #     return latent_mask.to(self.precision_t)

    # def _get_or_generate_latent_mask(self, image_tensor, image_index=None, **mask_kwargs):
    #     """
    #     内部辅助方法：根据输入的图像张量，生成并返回一个latent空间的掩码。
    #     使用FastSAM进行掩码生成
        
    #     Args:
    #         image_tensor: 输入图像张量 [C, H, W]
    #         image_index: 图像索引（可选）
    #         **mask_kwargs: 传递给FastSAM的额外参数
    #     """
    #     # 动态计算目标 latent 尺寸
    #     # image_tensor 的形状是 [C, H, W]
    #     target_height, target_width = image_tensor.shape[1], image_tensor.shape[2]
    #     latent_height = target_height // 8
    #     latent_width = target_width // 8

    #     print(f"[FastSAM Masking] Generating a new mask for image of size ({target_height}, {target_width}) -> latent size ({latent_height}, {latent_width})...")
        
    #     # 1. 转换Tensor为NumPy数组
    #     image_np_uint8 = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
    #     # 2. 准备FastSAM参数
    #     fastsam_params = {
    #         'strategy': self.mask_strategy,
    #         'imgsz': mask_kwargs.get('imgsz', 1024),
    #         'conf': mask_kwargs.get('conf', 0.4),
    #         'iou': mask_kwargs.get('iou', 0.9),
    #         'retina': mask_kwargs.get('retina', True)
    #     }
        
    #     # 添加特定提示参数
    #     if 'text_prompt' in mask_kwargs:
    #         fastsam_params['text_prompt'] = mask_kwargs['text_prompt']
    #     if 'point_prompt' in mask_kwargs:
    #         fastsam_params['point_prompt'] = mask_kwargs['point_prompt']
    #         fastsam_params['point_label'] = mask_kwargs.get('point_label', [1])
    #     if 'box_prompt' in mask_kwargs:
    #         fastsam_params['box_prompt'] = mask_kwargs['box_prompt']
        
    #     # 3. 生成全分辨率掩码
    #     full_res_mask_np = self.subject_mask_generator.generate_mask_from_np(
    #         image_np_uint8, **fastsam_params)
        
    #     # 4. 处理生成失败
    #     if full_res_mask_np is None:
    #         print(f"[FastSAM Masking] WARN: Mask generation FAILED. Using a passthrough (all ones) mask.")
    #         latent_mask = torch.ones((1, 1, latent_height, latent_width), device=self.device, dtype=self.precision_t)
    #         return latent_mask

    #     # # ==================== 关键修改点 ====================
    #     # # 根据您的观察，生成的掩码是主体为0，背景为1。
    #     # # 为了修正这个问题，我们在这里将其反转，以确保后续步骤处理的掩码是 主体为1，背景为0。
    #     # full_res_mask_np = 1 - full_res_mask_np
    #     # # =====================================================

    #     # 5. 转换回Tensor并降采样到正确的 latent 尺寸
    #     mask_tensor = torch.from_numpy(full_res_mask_np).to(self.device).float().unsqueeze(0).unsqueeze(0)
    #     latent_mask = F.interpolate(mask_tensor, size=(latent_height, latent_width), mode='nearest-exact')

    #     # 6. 处理掩码反转
    #     if not self.mask_on_subject:
    #         latent_mask = 1.0 - latent_mask

    #     # 7. 直接返回，不存入缓存
    #     return latent_mask.to(self.precision_t)

    def _get_or_generate_latent_mask(self, image_tensor, image_index=None, **mask_kwargs):
        """
        内部辅助方法：根据输入的图像张量，生成并返回一个latent空间的掩码。
        *** 此方法已被修改，以使用我们新的高级分割器 ***
        
        参数:
            image_tensor (torch.Tensor): 输入图像张量，形状为 [C, H, W]，数值范围 [0, 1]。
            image_index (int, optional): 图像索引 (用于缓存，此处未使用)。
            **mask_kwargs: 不再被新方法使用，但保留以兼容API。
        """
        # 动态计算目标 latent 尺寸
        target_height, target_width = image_tensor.shape[1], image_tensor.shape[2]
        latent_height = target_height // 8
        latent_width = target_width // 8

        # print(f"[蒙版生成] 使用 '{self.mask_strategy}' 策略生成新蒙版...")
        
        # 1. 转换Tensor为OpenCV兼容的NumPy数组
        # PyTorch Tensor: [C, H, W], RGB, 0-1
        # OpenCV aumPy:   [H, W, C], BGR, 0-255
        image_np_rgb = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        # 将RGB转换为BGR，因为我们的函数基于OpenCV
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # 2. <<< 核心替换点 >>>
        #    调用我们新的高级蒙版生成器，而不是FastSAM。
        full_res_mask_np = generate_advanced_mask_hsv(image_np_bgr, verbose=False) # 在集成时通常关闭详细输出
        
        # 3. 处理蒙版生成失败的情况
        if full_res_mask_np is None:
            # print(f"[蒙版生成] 警告: 高级蒙版生成失败。将使用全通(全为1)的蒙版。")
            latent_mask = torch.ones((1, 1, latent_height, latent_width), device=self.device, dtype=self.precision_t)
            return latent_mask

        # 4. 将蒙版转换回Tensor，并降采样到正确的latent尺寸
        # full_res_mask_np 是单通道 (H, W)，需要添加批次和通道维度 -> (1, 1, H, W)
        mask_tensor = torch.from_numpy(full_res_mask_np).to(self.device).float().unsqueeze(0).unsqueeze(0) / 255.0
        # 使用双线性插值降采样到latent空间尺寸
        latent_mask = F.interpolate(mask_tensor, size=(latent_height, latent_width), mode='bilinear', align_corners=False)

        # 5. 处理蒙版反转逻辑 (例如，如果你想对背景应用guidance)
        if not self.mask_on_subject:
            latent_mask = 1.0 - latent_mask

        # 6. 返回最终的latent mask
        return latent_mask.to(self.precision_t)
        

    def get_curriculum_timestep(self, iteration, warm_up_rate=0):
        """
        根据训练阶段返回合适的时间步
        """
        current_phase = self.get_current_phase(iteration)
        
        if current_phase == 'adaptive':
            # 自适应时间步选择
            return self.get_adaptive_timestep(iteration, warm_up_rate)
        else:
            # 固定阶段时间步
            t_range = self.timestep_ranges[current_phase]
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            
            ind_t = torch.randint(min_step, max_step + 1, (1,), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
            return ind_t
    
    def get_current_phase(self, iteration):
        """确定当前训练阶段"""
        for i, transition in enumerate(self.phase_transitions):
            if iteration < transition:
                return self.curriculum_phases[i]
        return self.curriculum_phases[-1]
    
    def get_adaptive_timestep(self, iteration, warm_up_rate):
        """
        自适应时间步选择 - 基于训练进度和梯度历史
        """
        # 基础进度调度
        progress = min(iteration / 5000, 1.0)  # 假设3000步为完整训练
        
        # 动态调整时间步范围
        if progress < 0.3:
            # 早期：重点学习粗粒度结构
            t_min, t_max = 0.6, 0.98
            weight_high = 0.8  # 更多高噪声时间步
        elif progress < 0.7:
            # 中期：平衡学习
            t_min, t_max = 0.2, 0.9
            weight_high = 0.5
        else:
            # 后期：重点学习细节
            t_min, t_max = 0.02, 0.6
            weight_high = 0.2  # 更多低噪声时间步
        
        # 基于梯度历史调整
        if hasattr(self, 'grad_history') and len(self.grad_history) > 10:
            recent_grad_volatility = np.std(self.grad_history[-10:])
            if recent_grad_volatility > 0.1:  # 梯度不稳定
                # 使用更高的噪声水平来稳定训练
                t_min = max(t_min, 0.4)
                weight_high = min(weight_high + 0.2, 0.9)
        
        # 采样时间步
        min_step = int(self.num_train_timesteps * t_min)
        max_step = int(self.num_train_timesteps * t_max)
        
        # 加权采样：根据weight_high调整分布
        if torch.rand(1) < weight_high:
            # 采样高噪声时间步
            high_min = int((min_step + max_step) * 0.6)
            ind_t = torch.randint(high_min, max_step + 1, (1,), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            # 采样低噪声时间步
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
                           image_indices=None, # <--- 接收批量的图像索引
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

        # ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        # if iteration < opt.warmup_iter:  # 阶段一：warmup阶段，随机采样
        #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        # else:  # 阶段二：线性递减采样
        #     decay_range = self.max_step - self.min_step + 1
        #     weights = torch.linspace(1.0, 0.1, decay_range, device=self.device)  # 高 -> 低 权重
        #     weights = weights / weights.sum()
        #     sampled_index = torch.multinomial(weights, 1, replacement=True)[0]
        #     ind_t = sampled_index + self.min_step

        # if iteration < opt.warmup_iter:
        #     # 阶段一：随机采样
        #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        # else:
        #     # 阶段二：线性递减采样
        #     decay_iter = iteration - opt.warmup_iter
        #     decay_total = opt.iterations - opt.warmup_iter

        #     # 计算一个线性插值比例（从1到0）
        #     decay_ratio = 1.0 - (decay_iter / decay_total)
            
        #     # 根据比例计算 ind_t
        #     ind_t_f = self.min_step + decay_ratio * (self.max_step - self.min_step)
        #     ind_t = int(ind_t_f)
        #     ind_t = torch.tensor(ind_t, dtype=torch.long, device='cuda')

        # if iteration < opt.warmup_iter:
        #     # 阶段一：随机采样
        #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        # else:
        #     # 阶段二：线性递减采样，但保持在25-35范围
        #     decay_iter = iteration - opt.warmup_iter
        #     decay_total = opt.iterations - opt.warmup_iter
            
        #     # 计算线性插值比例（从1到0）
        #     decay_ratio = 1.0 - (decay_iter / decay_total)
            
        #     # 设置最小时间步阈值（例如25）
        #     min_final_step = 25
        #     max_final_step = 35
            
        #     # 根据比例计算 ind_t，但限制在 [min_final_step, self.max_step] 范围内
        #     ind_t_f = max(min_final_step, self.min_step + decay_ratio * (self.max_step - self.min_step))
        #     ind_t = int(ind_t_f)
        #     ind_t = torch.tensor(ind_t, dtype=torch.long, device='cuda')

        if self.use_dhg_latent_hypergraph and self.dhg_latent_hypergraph is not None:
            if iteration < opt.warmup_iter:
                # 阶段一：warmup期间，在[min_step, max_step + warmup部分]之间随机采样
                upper_bound = self.max_step + int(self.warmup_step * warm_up_rate)
                ind_t = torch.randint(self.min_step, upper_bound + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
            else:
                # 阶段二：线性递减最大采样步数范围，在[min_step, 当前最大值]之间随机采样
                decay_iter = iteration - opt.warmup_iter
                decay_total = opt.iterations - opt.warmup_iter

                # 计算当前最大值（线性递减）
                current_max_step = int(self.min_step + (1.0 - decay_iter / decay_total) * (self.max_step - self.min_step))
                current_max_step = max(self.min_step, current_max_step)

                # 在[min_step, 当前最大值]之间随机采样
                ind_t = torch.randint(self.min_step, current_max_step + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        # if iteration < opt.warmup_iter:
        #     # 阶段一：warmup期间，在[min_step, max_step + warmup部分]之间随机采样
        #     upper_bound = self.max_step + int(self.warmup_step * warm_up_rate)
        #     ind_t = torch.randint(self.min_step, upper_bound + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        # else:
        #     # 阶段二：线性递减最大采样步数范围，在[min_step, 当前最大值]之间随机采样
        #     decay_iter = iteration - opt.warmup_iter
        #     decay_total = opt.iterations - opt.warmup_iter

        #     # 计算当前最大值（线性递减）
        #     current_max_step = int(self.min_step + (1.0 - decay_iter / decay_total) * (self.max_step - self.min_step))
        #     current_max_step = max(self.min_step, current_max_step)

        #     # 在[min_step, 当前最大值]之间随机采样
        #     ind_t = torch.randint(self.min_step, current_max_step + 1, (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        # ind_t = self.get_biased_time_step(warm_up_rate)
        # ind_t = self.get_curriculum_timestep(iteration, warm_up_rate)
        # ind_t, current_guidance_scale, current_loss_weight, current_phase = self.get_annealed_params(iteration)

        # min_final_step = 25
        # ind_prev_t_raw = ind_t - current_delta_t
        # ind_prev_t = torch.clamp(ind_prev_t_raw, min=min_final_step)

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

            # =================== 1. 为 pred_rgb (batch_size=4) 生成掩码 ===================
            # 既然总是重新生成，就不再需要 image_indices 和缓存逻辑了
            mask_list_input = []
            # print("[Masking] Generating masks for input images (pred_rgb)...")
            for i in range(pred_rgb.shape[0]):
                # 直接调用生成，不传递 image_index
                mask = self._get_or_generate_latent_mask(pred_rgb[i], image_index=None)
                mask_list_input.append(mask)
            latent_mask_input = torch.cat(mask_list_input, dim=0)


            # =================== 2. 为 pred_x0_pos (batch_size=4) 生成掩码 ===================
            # 这部分逻辑保持不变，因为它本来就是实时生成的
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

        # 在这里打印grad的形状
        # print(f"[DEBUG] grad.shape: {grad.shape}")
        # print(f"[DEBUG] grad.dtype: {grad.dtype}")
        # print(f"[DEBUG] grad.device: {grad.device}")
        # print(f"[DEBUG] grad requires_grad: {grad.requires_grad}")

        # ==================== 在这里应用超图增强 ====================
        if self.use_hypergraph and self.hypergraph_enhancer is not None:
            original_grad = grad.clone() # 保存原始梯度用于监控
            
            # 直接调用，无需担心反向传播
            grad = self.hypergraph_enhancer(grad) 
            
            # 监控
            with torch.no_grad():
                original_grad_norm = torch.norm(original_grad).item()
                enhanced_grad_norm = torch.norm(grad).item()
                self.writer.add_scalar('Hypergraph/Enhancement_Ratio', enhanced_grad_norm / (original_grad_norm + 1e-9), iteration)
                
                # 可视化原始梯度和增强后梯度的差异
                if iteration % (guidance_opt.vis_interval * 5) == 0:
                    grad_diff = torch.abs(grad - original_grad).mean(dim=1, keepdim=True)
                    grad_diff_vis = F.interpolate(grad_diff, size=(512, 512), mode='bilinear')
                    grad_diff_vis = grad_diff_vis / (grad_diff_vis.max() + 1e-8)
                    self.writer.add_images('Hypergraph/Grad_Difference', grad_diff_vis.repeat(1, 3, 1, 1), iteration)
        # ==========================================================

        # ==================== 应用DHG Latent超图增强 ====================
        dhg_latent_loss = 0
        if self.use_dhg_latent_hypergraph and self.dhg_latent_hypergraph is not None:
            # print(f"[DEBUG] Applying DHG latent hypergraph enhancement...")
            
            try:
                # 使用改进的DHG latent超图处理器（传入iteration用于重构判断）
                grad_enhancement, dhg_latent_loss = self.dhg_latent_hypergraph(
                    latents,          # 原始latents [B, 4, 64, 64]
                    (prev_latents_noisy-pred_noise),    # 加噪latents [B, 4, 64, 64]
                    iteration,        # 当前迭代数，用于判断是否重构\
                    mask1=latent_mask_pred,
                    mask2=latent_mask_input,
                )
                
                # print(f"[DEBUG] DHG Latent grad_enhancement.shape: {grad_enhancement.shape}")
                # print(f"[DEBUG] DHG Latent loss: {dhg_latent_loss.item()}")
                
                # ==================== 梯度平衡与融合 ====================
                with torch.no_grad():
                    # 1. 计算两个梯度的L2范数（幅度）
                    original_grad_norm = torch.norm(grad)
                    enhancement_norm = torch.norm(grad_enhancement)
                    
                    # 2. 计算缩放因子，使增强梯度的范数与原始梯度的范数对齐
                    # 添加一个小的epsilon防止除以零
                    scale_factor = original_grad_norm / (enhancement_norm + 1e-8)
                    
                    # 获取超参数权重
                    dhg_enhancement_weight = getattr(guidance_opt, 'dhg_latent_weight', 0.3) # 建议默认值设小一点

                # 3. 应用缩放和加权
                # 现在的 dhg_enhancement_weight 控制了“增强梯度”相对于“原始梯度”的强度比例
                # 例如, weight=0.1 意味着增强梯度的贡献强度是原始梯度的10%
                scaled_enhancement = grad_enhancement * scale_factor * dhg_enhancement_weight

                original_grad = grad.clone() # 用于监控
                grad = grad + scaled_enhancement
                
                # 固定权重融合
                # dhg_enhancement_weight = getattr(guidance_opt, 'dhg_latent_weight', 0.5)
                # original_grad = grad.clone()
                # grad = grad + dhg_enhancement_weight * grad_enhancement
                
                # 记录监控信息
                with torch.no_grad():
                    enhancement_norm = torch.norm(scaled_enhancement).item()
                    original_norm = torch.norm(original_grad).item()
                    enhanced_norm = torch.norm(grad).item()
                    
                    self.writer.add_scalar('DHG_Latent/EnhancementNorm', enhancement_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/OriginalGradNorm', original_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/EnhancedGradNorm', enhanced_norm, iteration)
                    self.writer.add_scalar('DHG_Latent/Loss', dhg_latent_loss.item(), iteration)
                    self.writer.add_scalar('DHG_Latent/EnhancementWeight', dhg_enhancement_weight, iteration)
                    
                    # 记录超图重构信息
                    if hasattr(self.dhg_latent_hypergraph, 'last_reconstruction_iter'):
                        steps_since_reconstruction = iteration - self.dhg_latent_hypergraph.last_reconstruction_iter
                        self.writer.add_scalar('DHG_Latent/StepsSinceReconstruction', steps_since_reconstruction, iteration)
                    
                    # 计算增强效果统计
                    enhancement_ratio = enhanced_norm / (original_norm + 1e-9)
                    self.writer.add_scalar('DHG_Latent/EnhancementRatio', enhancement_ratio, iteration)
                    
                    # 可视化增强效果
                    if iteration % 100 == 0:
                        # 增强可视化
                        enhancement_vis = scaled_enhancement.abs().mean(dim=1, keepdim=True)
                        enhancement_vis = F.interpolate(enhancement_vis, size=(512, 512), mode='bilinear')
                        enhancement_vis = enhancement_vis / (enhancement_vis.max() + 1e-8)
                        self.writer.add_images('DHG_Latent/Enhancement_Visualization', 
                                            enhancement_vis.repeat(1, 3, 1, 1), iteration)
                        
                        # 梯度差异可视化
                        grad_diff = torch.abs(grad - original_grad).mean(dim=1, keepdim=True)
                        grad_diff_vis = F.interpolate(grad_diff, size=(512, 512), mode='bilinear')
                        grad_diff_vis = grad_diff_vis / (grad_diff_vis.max() + 1e-8)
                        self.writer.add_images('DHG_Latent/Grad_Difference', 
                                            grad_diff_vis.repeat(1, 3, 1, 1), iteration)
                        
                        # 原始和增强后的梯度对比
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

        # 更全面的梯度监控策略
        with torch.no_grad():
            # === 基础统计信息 ===
            grad_norm = torch.norm(grad).item()
            grad_mean = torch.mean(grad).item()
            grad_std = torch.std(grad).item()
            grad_max = torch.max(grad).item()
            grad_min = torch.min(grad).item()
            grad_abs_mean = torch.mean(torch.abs(grad)).item()
            
            # === 通道级别分析 ===
            for i in range(grad.shape[1]):  # 4个潜在通道
                channel_grad = grad[:, i, :, :]
                channel_norm = torch.norm(channel_grad).item()
                channel_mean = torch.mean(channel_grad).item()
                channel_std = torch.std(channel_grad).item()
                
                self.writer.add_scalar(f'Gradient/Channel_{i}_Norm', channel_norm, iteration)
                self.writer.add_scalar(f'Gradient/Channel_{i}_Mean', channel_mean, iteration)
                self.writer.add_scalar(f'Gradient/Channel_{i}_Std', channel_std, iteration)
            
            # === 空间分析 ===
            # 计算不同区域的梯度强度
            h, w = grad.shape[2], grad.shape[3]
            center_grad = grad[:, :, h//4:3*h//4, w//4:3*w//4]
            edge_grad = grad.clone()
            edge_grad[:, :, h//4:3*h//4, w//4:3*w//4] = 0
            
            center_norm = torch.norm(center_grad).item()
            edge_norm = torch.norm(edge_grad).item()
            
            self.writer.add_scalar('Gradient/Center_Norm', center_norm, iteration)
            self.writer.add_scalar('Gradient/Edge_Norm', edge_norm, iteration)
            self.writer.add_scalar('Gradient/Center_Edge_Ratio', center_norm / (edge_norm + 1e-8), iteration)
            
            # === 梯度变化趋势 ===
            # 保存历史梯度norm用于趋势分析
            if not hasattr(self, 'grad_history'):
                self.grad_history = []
            self.grad_history.append(grad_norm)
            
            # 保持最近100步的历史
            if len(self.grad_history) > 100:
                self.grad_history.pop(0)
            
            # 计算梯度变化趋势
            if len(self.grad_history) >= 10:
                recent_grads = self.grad_history[-10:]
                grad_trend = (recent_grads[-1] - recent_grads[0]) / 10
                grad_volatility = np.std(recent_grads)
                
                self.writer.add_scalar('Gradient/Trend', grad_trend, iteration)
                self.writer.add_scalar('Gradient/Volatility', grad_volatility, iteration)
            
            # === 相对于不同基准的分析 ===
            # 与初始噪声的比较
            noise_norm = torch.norm(noise).item() if 'noise' in locals() else 0
            pred_noise_norm = torch.norm(pred_noise).item()
            target_norm = torch.norm(target).item()
            
            self.writer.add_scalar('Gradient/Grad_vs_Noise_Ratio', grad_norm / (noise_norm + 1e-8), iteration)
            self.writer.add_scalar('Gradient/Grad_vs_PredNoise_Ratio', grad_norm / (pred_noise_norm + 1e-8), iteration)
            self.writer.add_scalar('Gradient/Grad_vs_Target_Ratio', grad_norm / (target_norm + 1e-8), iteration)
            
            # === 梯度健康度指标 ===
            # 检测梯度异常
            grad_has_nan = torch.isnan(grad).any().item()
            grad_has_inf = torch.isinf(grad).any().item()
            grad_zero_ratio = (grad == 0).float().mean().item()
            grad_positive_ratio = (grad > 0).float().mean().item()
            
            self.writer.add_scalar('Gradient/Has_NaN', float(grad_has_nan), iteration)
            self.writer.add_scalar('Gradient/Has_Inf', float(grad_has_inf), iteration)
            self.writer.add_scalar('Gradient/Zero_Ratio', grad_zero_ratio, iteration)
            self.writer.add_scalar('Gradient/Positive_Ratio', grad_positive_ratio, iteration)
            
            # === 多尺度分析 ===
            # 不同尺度的梯度强度
            grad_pooled_2x2 = F.avg_pool2d(grad.abs(), 2)
            grad_pooled_4x4 = F.avg_pool2d(grad.abs(), 4)
            
            fine_scale_norm = torch.norm(grad).item()
            medium_scale_norm = torch.norm(grad_pooled_2x2).item()
            coarse_scale_norm = torch.norm(grad_pooled_4x4).item()
            
            self.writer.add_scalar('Gradient/Fine_Scale_Norm', fine_scale_norm, iteration)
            self.writer.add_scalar('Gradient/Medium_Scale_Norm', medium_scale_norm, iteration)
            self.writer.add_scalar('Gradient/Coarse_Scale_Norm', coarse_scale_norm, iteration)
            
            # === 基础记录 ===
            self.writer.add_scalar('Gradient/Norm', grad_norm, iteration)
            self.writer.add_scalar('Gradient/Mean', grad_mean, iteration)
            self.writer.add_scalar('Gradient/Std', grad_std, iteration)
            self.writer.add_scalar('Gradient/Max', grad_max, iteration)
            self.writer.add_scalar('Gradient/Min', grad_min, iteration)
            self.writer.add_scalar('Gradient/AbsMean', grad_abs_mean, iteration)
            
            # === 分布记录 ===
            self.writer.add_histogram('Gradient/Distribution', grad.flatten(), iteration)
            
            # 记录每个通道的分布
            for i in range(grad.shape[1]):
                self.writer.add_histogram(f'Gradient/Channel_{i}_Distribution', 
                                        grad[:, i, :, :].flatten(), iteration)
            
            # === 训练相关信息 ===
            self.writer.add_scalar('Training/Timestep', t.item(), iteration)
            self.writer.add_scalar('Training/GuidanceScale', guidance_opt.guidance_scale, iteration)
            self.writer.add_scalar('Training/WarmupRate', warm_up_rate, iteration)
            self.writer.add_scalar('Training/CurrentDeltaT', current_delta_t, iteration)
            
            # === 定期保存梯度可视化 ===
            if iteration % (guidance_opt.vis_interval * 5) == 0:
                # 保存梯度的空间分布图
                grad_vis = grad.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
                grad_vis = F.interpolate(grad_vis, size=(512, 512), mode='bilinear')
                grad_vis = grad_vis / (grad_vis.max() + 1e-8)  # 归一化
                
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