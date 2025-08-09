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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
import matplotlib.pyplot as plt
import os.path as osp
import cv2
from .vision_transformer import vit_small

from .sd_step import *
import dhg
from dhg.nn import HGNNPConv

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
                 ddim_inv=False, use_control_net=True, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file('/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base', use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained('/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base', torch_dtype=self.precision_t, local_files_only=True)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained('/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base' if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained("/home/s414e2/CJH/Text-to-3D/DreamControl/ControlNet",torch_dtype=self.precision_t).to(device)

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
    
    # def update_with_hypergraph(self, pred_rgb):
    #     """
    #     使用DHG库通过超图更新单张图像，使用KNN构建超边
        
    #     Args:
    #         pred_rgb (torch.Tensor): 形状为 [B, 3, H, W] 的RGB图像张量
                
    #     Returns:
    #         torch.Tensor: 更新后的pred_rgb，维度保持[B, 3, H, W]
    #     """
    #     B, C, H, W = pred_rgb.shape
    #     device = pred_rgb.device
        
    #     # 创建列表存储更新后的图像
    #     updated_rgbs = []
        
    #     for b in range(B):
    #         # 获取当前处理的单个图像
    #         current_img = pred_rgb[b:b+1]  # 保持[1, 3, H, W]维度
            
    #         # 1. 下采样图像以减少计算复杂度
    #         downscale_factor = 16  # 降低分辨率因子
    #         h_small, w_small = H // downscale_factor, W // downscale_factor
        
    #         # 进行下采样 - 注意这里使用current_img而不是pred_rgb
    #         pooled_rgb = F.avg_pool2d(current_img, kernel_size=downscale_factor)  # [1, 3, H/16, W/16]
        
    #         # 2. 将图像重塑为像素特征集合
    #         # [1, 3, h_small, w_small] -> [h_small*w_small, 3]
    #         pixels = pooled_rgb.reshape(1, 3, h_small * w_small).permute(0, 2, 1).reshape(h_small * w_small, 3)
        
    #         # 计算余弦相似度矩阵
    #         norm_pixels = F.normalize(pixels, p=2, dim=1)
    #         similarity_matrix = torch.mm(norm_pixels, norm_pixels.t())  # [h_small*w_small, h_small*w_small]
        
    #         # 4. 使用KNN构建超边
    #         k = 8  # 每个像素连接到最相似的k个像素
    #         e_list = []
        
    #         # 采样像素以减少计算量
    #         stride = 2  # 每隔几个像素采样一个
        
    #         # 注意：这里避免使用与外部循环相同的变量名
    #         for pixel_idx in range(0, h_small * w_small, stride):
    #             # 对每个像素，找出最相似的k个像素
    #             _, indices = torch.topk(similarity_matrix[pixel_idx], k=min(k+1, h_small * w_small))  # +1是因为包括自身
    #             e_list.append(indices.tolist())
        
    #         # 5. 创建超图
    #         num_pixels = h_small * w_small
    #         hypergraph = dhg.Hypergraph(num_v=num_pixels, e_list=e_list)
        
    #         # 6. 创建超图卷积层
    #         key = f'_hgnn_conv_{h_small}_{w_small}'
    #         if not hasattr(self, key):
    #             setattr(self, key, HGNNPConv(
    #                 in_channels=3,
    #                 out_channels=3,
    #                 use_bn=False,
    #                 bias=True
    #             ).to(device))
            
    #             # 初始化为略微接近单位矩阵的权重
    #             with torch.no_grad():
    #                 conv = getattr(self, key)
    #                 if hasattr(conv, 'weight'):
    #                     nn.init.eye_(conv.weight)
    #                     conv.weight.data += 0.05 * torch.randn_like(conv.weight.data)
        
    #         # 7. 应用超图卷积
    #         hgnn_conv = getattr(self, key)
    #         with torch.no_grad():
    #             updated_features = hgnn_conv(pixels, hypergraph)
        
    #         # 8. 重构更新后的图像
    #         # 计算更新差异
    #         feature_diff = updated_features - pixels
            
    #         # 将特征差异重塑回空间维度
    #         # [h_small*w_small, 3] -> [1, 3, h_small, w_small]
    #         feature_diff = feature_diff.reshape(h_small, w_small, 3).permute(2, 0, 1).unsqueeze(0)
            
    #         # 上采样回原始分辨率
    #         feature_diff = F.interpolate(feature_diff, size=(H, W), mode='bilinear', align_corners=False)
            
    #         # 应用更新 - 使用固定的混合因子
    #         blend_factor = 0.25  # 控制更新强度
    #         updated_rgb = current_img + blend_factor * feature_diff
            
    #         # 确保像素值在有效范围内
    #         updated_rgb = torch.clamp(updated_rgb, 0.0, 1.0)
        
    #         # 添加到结果列表
    #         updated_rgbs.append(updated_rgb)
        
    #     # 将所有更新后的图像拼接回原始批次大小
    #     return torch.cat(updated_rgbs, dim=0)
    
    # def update_with_multi_view_hypergraph(self, pred_rgb):
    #     """
    #     构建多视角联合超图，使不同角度图像间建立高阶关联
        
    #     Args:
    #         pred_rgb (torch.Tensor): 形状为 [B, 3, H, W] 的RGB图像张量，B为视角数(通常为4)
                
    #     Returns:
    #         torch.Tensor: 更新后的pred_rgb，维度保持[B, 3, H, W]
    #     """
    #     B, C, H, W = pred_rgb.shape
    #     device = pred_rgb.device
        
    #     with torch.no_grad():  # 确保所有超图操作都不需要梯度
    #         # 1. 首先对每个视角单独进行超图更新
    #         single_view_updated = self.update_with_hypergraph(pred_rgb)
            
    #         # 2. 再进行多视角联合超图处理
    #         # 2.1 对每个视角下采样以减少计算复杂度
    #         downscale_factor = 16
    #         h_small, w_small = H // downscale_factor, W // downscale_factor
            
    #         # 下采样所有视角的图像
    #         pooled_rgbs = F.avg_pool2d(pred_rgb, kernel_size=downscale_factor)  # [B, 3, h_small, w_small]
            
    #         # 2.2 从每个视角提取特征向量
    #         all_features = []  # 存储所有视角的所有像素特征
    #         point_positions = []  # 存储每个特征点的(视角, h, w)信息
            
    #         # 为了减少计算量，每个视角只采样部分像素
    #         sample_stride = 2  # 采样步长
            
    #         for b in range(B):
    #             # 将当前视角的下采样图像重塑为像素特征
    #             view_features = pooled_rgbs[b].permute(1, 2, 0)  # [h_small, w_small, 3]
    #             view_features = view_features.reshape(h_small * w_small, 3)
                
    #             # 采样部分像素以减少计算量
    #             sampled_indices = list(range(0, h_small * w_small, sample_stride))
    #             sampled_features = view_features[sampled_indices]
                
    #             # 记录采样的特征及其位置信息
    #             all_features.append(sampled_features)
                
    #             # 为每个特征点记录其(视角, h, w)坐标
    #             for idx in sampled_indices:
    #                 h_pos = idx // w_small
    #                 w_pos = idx % w_small
    #                 point_positions.append((b, h_pos, w_pos))
            
    #         # 合并所有视角的特征为一个大矩阵
    #         all_features = torch.cat(all_features, dim=0)  # [n_points, 3]
            
    #         # 2.3 计算跨视角的特征相似度
    #         # 首先归一化特征
    #         norm_features = F.normalize(all_features, p=2, dim=1)
    #         # 计算相似度矩阵
    #         cross_view_similarity = torch.mm(norm_features, norm_features.t())  # [n_points, n_points]
            
    #         # 2.4 基于相似度构建跨视角超边
    #         k = 12  # 每个特征点连接到的最相似点数量
    #         e_list = []
            
    #         # 为每个特征点构建一个超边
    #         for i in range(len(all_features)):
    #             # 找出最相似的k个特征点
    #             _, indices = torch.topk(cross_view_similarity[i], k=min(k, len(all_features)))
    #             e_list.append(indices.tolist())
            
    #         # 2.5 创建跨视角超图
    #         multi_view_graph = dhg.Hypergraph(num_v=len(all_features), e_list=e_list)
            
    #         # 2.6 创建并应用超图卷积
    #         key = '_multi_view_hgnn_conv'
    #         if not hasattr(self, key):
    #             setattr(self, key, HGNNPConv(
    #                 in_channels=3,
    #                 out_channels=3,
    #                 use_bn=False,
    #                 bias=True
    #             ).to(device))
                
    #             # 初始化权重
    #             conv = getattr(self, key)
    #             if hasattr(conv, 'weight'):
    #                 nn.init.eye_(conv.weight)
    #                 conv.weight.data = conv.weight.data + 0.05 * torch.randn_like(conv.weight.data)  # 避免使用+=原地操作
            
    #         # 应用超图卷积
    #         hgnn_conv = getattr(self, key)
    #         updated_features = hgnn_conv(all_features, multi_view_graph)
            
    #         # 2.7 计算特征差异
    #         feature_diff = updated_features - all_features
            
    #         # 2.8 将更新后的特征映射回各个视角
    #         multi_view_diffs = torch.zeros(B, 3, h_small, w_small, device=device)
    #         count_map = torch.zeros(B, 1, h_small, w_small, device=device)
            
    #         # 将特征差异映射回原位置
    #         for idx, (b, h, w) in enumerate(point_positions):
    #             # 使用非原地操作
    #             new_values = multi_view_diffs[b, :, h, w] + feature_diff[idx]
    #             multi_view_diffs[b, :, h, w] = new_values
                
    #             # 使用非原地操作
    #             new_count = count_map[b, 0, h, w] + 1
    #             count_map[b, 0, h, w] = new_count
            
    #         # 处理未被采样的位置(使用邻近点的平均值填充)
    #         for b in range(B):
    #             # 找出未被采样的位置(count=0的位置)
    #             zero_mask = (count_map[b, 0] == 0)
                
    #             if zero_mask.any():
    #                 # 对于未采样的位置，使用邻域平均值填充
    #                 kernel_size = 3
    #                 padding = kernel_size // 2
                    
    #                 # 使用平均池化操作来获取邻域平均值
    #                 avg_diffs = F.avg_pool2d(
    #                     F.pad(multi_view_diffs[b].unsqueeze(0), (padding, padding, padding, padding), mode='reflect'),
    #                     kernel_size=kernel_size,
    #                     stride=1
    #                 ).squeeze(0)
                    
    #                 # 获取邻域计数的平均值
    #                 avg_counts = F.avg_pool2d(
    #                     F.pad(count_map[b].unsqueeze(0), (padding, padding, padding, padding), mode='reflect'),
    #                     kernel_size=kernel_size,
    #                     stride=1
    #                 ).squeeze(0)
                    
    #                 # 非原地操作填充未采样位置
    #                 multi_view_diffs_clone = multi_view_diffs.clone()
    #                 multi_view_diffs_clone[b, :, zero_mask] = avg_diffs[:, zero_mask]
    #                 multi_view_diffs = multi_view_diffs_clone
                    
    #                 count_map_clone = count_map.clone()
    #                 count_map_clone[b, :, zero_mask] = avg_counts[:, zero_mask]
    #                 count_map = count_map_clone
            
    #         # 避免除零错误
    #         count_map = torch.clamp(count_map, min=1.0)
            
    #         # 计算平均差异
    #         multi_view_diffs = multi_view_diffs / count_map
            
    #         # 2.9 上采样回原始分辨率
    #         multi_view_diffs = F.interpolate(multi_view_diffs, size=(H, W), mode='bilinear', align_corners=False)
        
    #     # 从无梯度区域出来，重新连接计算图
    #     # 3. 融合单视角和多视角的结果
    #     # 控制单视角和多视角更新的权重
    #     single_view_weight = 0.6  # 单视角结果权重
    #     multi_view_weight = 0.4   # 多视角结果权重
        
    #     # 应用多视角特征差异，使用较小的混合因子以避免过度更新
    #     multi_view_blend_factor = 0.2  # 控制多视角更新强度
        
    #     # 融合两种结果 - 注意这里要保留梯度连接
    #     pred_rgb_updated = pred_rgb + multi_view_blend_factor * multi_view_diffs
    #     final_result = single_view_weight * single_view_updated + multi_view_weight * pred_rgb_updated
        
    #     # 确保像素值在有效范围内，使用非原地操作
    #     final_result = torch.clamp(final_result, 0.0, 1.0)
        
    #     return final_result
    
    def extract_edge_attention_mask(self, pred_rgb, save_folder=None, iteration=0, vis_interval=50):
        """
        结合DINO自注意力和边缘检测，提取图像的边缘注意力掩码
        
        Args:
            pred_rgb (torch.Tensor): 形状为 [B, 3, H, W] 的RGB图像张量
            save_folder: 保存可视化结果的文件夹
            iteration: 当前迭代次数
            vis_interval: 可视化间隔
            
        Returns:
            torch.Tensor: 带有注意力权重的边缘掩码，维度为[B, 1, H, W]
        """
        B, C, H, W = pred_rgb.shape
        device = pred_rgb.device
        
        # 如果模型还未初始化，先加载DINO ViT模型
        if not hasattr(self, 'dino_model'):
            try:
                self.dino_model = vit_small(patch_size=8, num_classes=0)
                
                # 加载预训练权重
                pretrained_weights = '/home/s414e2/CJH/Text-to-3D/LucidDreamer/Hyper3DG/dino_deitsmall8_pretrain.pth'
                if osp.isfile(pretrained_weights):
                    state_dict = torch.load(pretrained_weights, map_location="cpu")
                    if "teacher" in state_dict:
                        state_dict = state_dict["teacher"]
                    # 去除不需要的前缀
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                    self.dino_model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded DINO weights from {pretrained_weights}")
                else:
                    print(f"DINO weights not found at {pretrained_weights}, using random initialization")
                
                self.dino_model.to(device)
                self.dino_model.eval()
                
                # 定义图像转换
                self.dino_transform = T.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            except (ImportError, FileNotFoundError) as e:
                print(f"无法加载DINO模型，将使用基于OpenCV的边缘检测替代: {e}")
                self.dino_model = None
        
        # 创建一个列表存储所有视角的边缘掩码
        edge_masks = []
        
        for b in range(B):
            # 获取当前视角的图像
            current_img = pred_rgb[b]  # [3, H, W]
            
            # 转换为OpenCV格式并提取边缘
            np_img = current_img.detach().cpu().permute(1, 2, 0).numpy()
            np_img = (np_img * 255).astype(np.uint8)
            
            # 边缘检测
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)  # 使用较低的阈值来捕获更多边缘
            
            # 膨胀边缘以扩大影响区域
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # 应用高斯模糊使边缘更平滑
            edge_mask = cv2.GaussianBlur(dilated_edges, (9, 9), 0)
            
            # 如果DINO模型可用，生成注意力掩码
            if self.dino_model is not None:
                with torch.no_grad():
                    # 归一化图像并调整为DINO输入格式
                    dino_input = self.dino_transform(current_img).unsqueeze(0)
                    
                    # 确保输入尺寸是补丁大小的倍数
                    patch_size = 8
                    h, w = dino_input.shape[2], dino_input.shape[3]
                    h_pad = (h // patch_size) * patch_size
                    w_pad = (w // patch_size) * patch_size
                    dino_input = dino_input[:, :, :h_pad, :w_pad]
                    
                    # 获取最后一层的自注意力图
                    attentions = self.dino_model.get_last_selfattention(dino_input.to(device))
                    
                    # 处理注意力图
                    nh = attentions.shape[1]  # 头的数量
                    # 只保留输出补丁注意力
                    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
                    w_featmap = h_pad // patch_size
                    h_featmap = w_pad // patch_size
                    attentions = attentions.reshape(nh, w_featmap, h_featmap)
                    attentions = nn.functional.interpolate(
                        attentions.unsqueeze(0), 
                        size=(H, W), 
                        mode="bicubic"
                    )[0].cpu().numpy()
                    
                    # 使用第一个注意力头作为掩码
                    attn_mask = attentions[0]
                    
                    # 将注意力掩码与边缘掩码结合
                    combined_mask = attn_mask * (edge_mask / 255.0)
                    
                    # 归一化结合掩码
                    if combined_mask.sum() > 0:
                        combined_mask = combined_mask / combined_mask.max()
                    
                    # 将结合掩码转换为张量
                    mask_tensor = torch.from_numpy(combined_mask).float().to(device).unsqueeze(0)
                
            else:
                # 如果DINO不可用，只使用边缘掩码
                mask_tensor = torch.from_numpy(edge_mask / 255.0).float().to(device).unsqueeze(0)
            
            edge_masks.append(mask_tensor)
            
            # 可视化掩码
            if save_folder is not None and iteration % vis_interval == 0:
                mask_vis_dir = osp.join(save_folder, 'edge_masks')
                Path(mask_vis_dir).mkdir(exist_ok=True, parents=True)
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(np_img)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(edge_mask, cmap='jet')
                plt.title('Edge Mask')
                plt.axis('off')
                
                if self.dino_model is not None:
                    plt.subplot(1, 3, 3)
                    plt.imshow(combined_mask, cmap='jet')
                    plt.title('Combined Attention-Edge Mask')
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(osp.join(mask_vis_dir, f'edge_mask_view{b}_iter{iteration}.png'))
                plt.close()
        
        # 将所有视角的掩码堆叠为批次张量
        edge_attention_masks = torch.stack(edge_masks, dim=0)  # [B, 1, H, W]
        
        return edge_attention_masks
    
    def extract_lowpass_mask(self, pred_rgb, save_folder=None, iteration=0, vis_interval=50):
        """
        提取图像的低通区域掩码 - 边缘为0，非边缘区域为1
        
        Args:
            pred_rgb (torch.Tensor): 形状为 [B, 3, H, W] 的RGB图像张量
            save_folder: 保存可视化结果的文件夹
            iteration: 当前迭代次数
            vis_interval: 可视化间隔
            
        Returns:
            torch.Tensor: 低通掩码，维度为[B, 1, H, W]，边缘为0，非边缘为1
        """
        import numpy as np
        import cv2
        from pathlib import Path
        import matplotlib.pyplot as plt
        import os.path as osp
        
        B, C, H, W = pred_rgb.shape
        device = pred_rgb.device
        
        # 创建一个列表存储所有视角的低通掩码
        lowpass_masks = []
        
        for b in range(B):
            # 获取当前视角的图像
            current_img = pred_rgb[b]  # [3, H, W]
            
            # 转换为OpenCV格式并提取边缘
            np_img = current_img.detach().cpu().permute(1, 2, 0).numpy()
            np_img = (np_img * 255).astype(np.uint8)
            
            # 边缘检测 - 专注于获取高频边缘
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            # 先用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 使用Canny边缘检测，调整阈值以捕获更多边缘
            edges = cv2.Canny(blurred, 50, 150)  # 降低阈值以捕获更多细节边缘
            
            # 膨胀边缘以扩大影响区域
            # kernel = np.ones((3, 3), np.uint8)
            # dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # 应用高斯模糊平滑边缘
            edge_mask = cv2.GaussianBlur(edges, (5, 5), 0)
            
            # 二值化处理
            # 先归一化到0-1范围
            edge_mask = edge_mask / 255.0
            # 边缘区域（高值）设为0，非边缘区域设为1（注意这里取反）
            lowpass_mask = 1.0 - (edge_mask > 0.2).astype(np.float32)
            
            # 转换为张量
            mask_tensor = torch.from_numpy(lowpass_mask).float().to(device).unsqueeze(0)
            lowpass_masks.append(mask_tensor)
            
            # 可视化掩码
            if save_folder is not None and iteration % vis_interval == 0:
                mask_vis_dir = osp.join(save_folder, 'lowpass_masks')
                Path(mask_vis_dir).mkdir(exist_ok=True, parents=True)
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(np_img)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(edges, cmap='gray')
                plt.title('Canny Edges')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(lowpass_mask, cmap='gray')
                plt.title('Low-Pass Mask (Edges=0, Smooth=1)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(osp.join(mask_vis_dir, f'lowpass_mask_view{b}_iter{iteration}.png'))
                plt.close()
        
        # 将所有视角的掩码堆叠为批次张量
        lowpass_masks = torch.stack(lowpass_masks, dim=0)  # [B, 1, H, W]
        
        return lowpass_masks
    
    def apply_edge_guided_loss(self, pred_rgb, pred_x0, edge_masks, guidance_opt, iteration):
        """
        应用边缘引导损失，使模型更加关注边缘区域的一致性
        
        Args:
            pred_rgb: 原始输入图像 [B, 3, H, W]
            hyper_rgbs: 超图处理后的图像 [B, 3, H, W]
            pred_x0: SD模型预测的去噪图像 [B, 3, H, W]
            edge_masks: 边缘注意力掩码 [B, 1, H, W]
            guidance_opt: 训练选项
            iteration: 当前迭代次数
            
        Returns:
            边缘一致性损失值
        """
        # 获取边缘一致性损失权重
        edge_loss_weight = getattr(guidance_opt, 'edge_loss_weight', 0.8)
        
        # 使用边缘掩码强调边缘区域
        # 计算边缘区域的超图输出与SD预测之间的L1损失
        edge_pixel_losses = torch.abs(pred_rgb - pred_x0) * edge_masks
        
        # 只对边缘区域计算平均损失
        edge_area = edge_masks.sum() + 1e-8
        edge_loss = edge_pixel_losses.sum() / edge_area
        
        # 动态权重调整
        if hasattr(guidance_opt, 'edge_warmup_iters') and iteration < guidance_opt.edge_warmup_iters:
            # 热身阶段，逐渐增加权重
            current_weight = edge_loss_weight * (iteration / guidance_opt.edge_warmup_iters)
        else:
            current_weight = edge_loss_weight
        
        # 应用边缘损失权重
        weighted_edge_loss = current_weight * edge_loss
        
        return weighted_edge_loss
    
    def gaussian_low_pass_filter(self, input_tensor, kernel_size=15, sigma=5.0):
        """
        对输入张量应用高斯低通滤波器
        
        Args:
            input_tensor: 输入张量 [B, C, H, W]
            kernel_size: 高斯核大小
            sigma: 高斯核标准差
            
        Returns:
            低通滤波后的张量
        """
        device = input_tensor.device
        dtype = input_tensor.dtype
        
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建高斯核
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 创建二维高斯核
        gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)  # [kernel_size, kernel_size]
        
        # 为每个通道创建独立的滤波器
        C = input_tensor.shape[1]
        gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
        
        # 使用分组卷积进行滤波
        padding = kernel_size // 2
        filtered = F.conv2d(
            input_tensor, 
            gaussian_2d, 
            padding=padding, 
            groups=C
        )
        
        return filtered

    def apply_hypergraph_modeling_no_downsample(self, pred_rgb, pred_x0, iteration, guidance_opt, save_folder=None):
        """
        DINO特征超图建模与优化的完整实现 - 无下采样版本
        
        Args:
            pred_rgb: 输入RGB图像 [B, 3, H, W]
            pred_x0: SD模型预测的去噪图像 [B, 3, H, W]
            iteration: 当前迭代次数
            guidance_opt: 训练选项
            save_folder: 可视化结果保存路径
            
        Returns:
            torch.Tensor: 超图优化后的特征表示
            float: 超图一致性损失值
        """
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import os
        import cv2
        from pathlib import Path
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import dhg
        from dhg.nn import HGNNPConv
        
        # 获取超图相关参数
        hypergraph_warmup_iters = getattr(guidance_opt, 'hypergraph_warmup_iters', 100)
        hypergraph_update_interval = getattr(guidance_opt, 'hypergraph_update_interval', 50)
        num_patches = getattr(guidance_opt, 'num_patches', 16)  # K-means聚类数量
        knn_neighbors = getattr(guidance_opt, 'knn_neighbors', 8)  # 每个节点的邻居数量
        
        # 如果未达到预热阶段，直接返回输入和零损失
        if iteration < hypergraph_warmup_iters:
            return pred_rgb, 0.0
        
        device = pred_rgb.device
        batch_size = pred_rgb.shape[0]
        B, C, H, W = pred_rgb.shape
        
        # 创建一个列表存储每个批次图像的超图优化结果
        hypergraph_optimized_images = []
        hypergraph_losses = []
        
        # 对批次中的每个图像单独处理
        for b in range(batch_size):
            # 获取当前处理的单个图像
            current_rgb = pred_rgb[b:b+1]  # 保持[1, 3, H, W]维度
            current_sd_pred = pred_x0[b:b+1]  # SD预测的图像
            
            # ---------- 1. 提取图像特征用于聚类 ----------
            # 直接使用原始图像，不进行下采样
            image_np = current_rgb[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]
            h, w, c = image_np.shape
            
            # 创建像素坐标网格
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # 为了提高聚类效率，在保持原始分辨率的同时，可以采样部分像素
            # 使用步长采样可以减少计算量但保留空间分布
            stride = max(1, min(h, w) // 200)  # 自适应步长，确保采样点不超过~40000个
            
            # 使用步长采样像素
            sampled_image = image_np[::stride, ::stride, :]
            sampled_x = x_coords[::stride, ::stride].reshape(-1, 1)
            sampled_y = y_coords[::stride, ::stride].reshape(-1, 1)
            
            # 将像素值和坐标组合成特征
            pixel_features = np.column_stack([
                sampled_image.reshape(-1, c),    # 像素RGB值
                sampled_x / w,                   # 归一化x坐标
                sampled_y / h                    # 归一化y坐标
            ])
            
            # ---------- 2. 使用K-means进行聚类 ----------
            # 确保补丁数量合理
            effective_num_patches = min(num_patches, pixel_features.shape[0] // 10)
            kmeans = KMeans(n_clusters=effective_num_patches, random_state=0, max_iter=100).fit(pixel_features)
            
            # 将聚类标签扩展回原始图像大小
            # 首先为采样点建立索引映射
            sampled_indices = np.zeros((h, w), dtype=bool)
            sampled_indices[::stride, ::stride] = True
            
            # 创建完整的标签图
            full_labels = np.zeros((h, w), dtype=int)
            
            # 为每个像素分配最近的聚类中心
            print(f"[超图处理] 批次 {b}：将聚类标签映射到原始分辨率...")
            
            # 为提高效率，我们可以使用最近邻搜索而不是逐像素计算
            from sklearn.neighbors import NearestNeighbors
            
            # 获取聚类中心
            cluster_centers = kmeans.cluster_centers_[:, :c]  # 只使用RGB部分
            sampled_labels = kmeans.labels_
            
            # 构建每个聚类的像素集合
            cluster_pixel_sets = {}
            for cluster_id in range(effective_num_patches):
                cluster_indices = np.where(sampled_labels == cluster_id)[0]
                y_indices = sampled_y[cluster_indices].flatten()
                x_indices = sampled_x[cluster_indices].flatten()
                
                # 存储这个聚类中的所有像素坐标
                cluster_pixel_sets[cluster_id] = list(zip(y_indices, x_indices))
            
            # ---------- 3. 提取每个聚类的补丁 ----------
            patches = []
            sd_patches = []
            patch_positions = []
            
            for cluster_id in range(effective_num_patches):
                if cluster_id not in cluster_pixel_sets or not cluster_pixel_sets[cluster_id]:
                    continue
                    
                # 获取当前聚类的像素坐标
                pixels = np.array(cluster_pixel_sets[cluster_id])
                y_indices = pixels[:, 0]
                x_indices = pixels[:, 1]
                
                # 计算边界框
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                
                # 确保边界框有最小尺寸
                min_size = 16
                if x2 - x1 < min_size:
                    pad = (min_size - (x2 - x1)) // 2
                    x1 = max(0, x1 - pad)
                    x2 = min(w - 1, x2 + pad)
                if y2 - y1 < min_size:
                    pad = (min_size - (y2 - y1)) // 2
                    y1 = max(0, y1 - pad)
                    y2 = min(h - 1, y2 + pad)
                
                # 由于我们直接在原始分辨率上操作，这里不需要缩放回原始大小
                orig_x1, orig_y1, orig_x2, orig_y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 确保边界不超出原始图像
                orig_x1 = max(0, orig_x1)
                orig_y1 = max(0, orig_y1)
                orig_x2 = min(W - 1, orig_x2)
                orig_y2 = min(H - 1, orig_y2)
                
                # 检查边界框是否有效
                if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
                    continue
                    
                # 提取补丁
                patch = current_rgb[0, :, orig_y1:orig_y2+1, orig_x1:orig_x2+1]
                sd_patch = current_sd_pred[0, :, orig_y1:orig_y2+1, orig_x1:orig_x2+1]
                
                # 调整补丁大小为统一尺寸
                target_size = (64, 64)  # DINO推荐的输入尺寸
                patch_resized = F.interpolate(patch.unsqueeze(0), size=target_size, mode='bilinear')[0]
                sd_patch_resized = F.interpolate(sd_patch.unsqueeze(0), size=target_size, mode='bilinear')[0]
                
                patches.append(patch_resized)
                sd_patches.append(sd_patch_resized)
                patch_positions.append((orig_x1, orig_y1, orig_x2, orig_y2))
            
            # 将补丁列表转换为批次张量
            if patches:
                patches_tensor = torch.stack(patches).to(device)
                sd_patches_tensor = torch.stack(sd_patches).to(device)
                positions_tensor = torch.tensor(patch_positions).to(device)
            else:
                # 如果没有有效补丁，返回整个图像作为单一补丁
                patches_tensor = F.interpolate(current_rgb, size=(64, 64), mode='bilinear')
                sd_patches_tensor = F.interpolate(current_sd_pred, size=(64, 64), mode='bilinear')
                positions_tensor = torch.tensor([[0, 0, W-1, H-1]]).to(device)
            
            # ---------- 4. 使用DINO提取视觉特征 ----------
            # 如果DINO模型尚未加载，则加载它
            if not hasattr(self, 'dino_hypergraph_model') or self.dino_hypergraph_model is None:
                # 加载用于超图处理的DINO ViT模型
                self.dino_hypergraph_model = vit_small(patch_size=8, num_classes=0)
                
                # 加载预训练权重
                dino_weights_path = '/home/s414e2/CJH/Text-to-3D/LucidDreamer/Hyper3DG/dino_deitsmall8_pretrain.pth'
                if os.path.isfile(dino_weights_path):
                    state_dict = torch.load(dino_weights_path, map_location="cpu")
                    if "teacher" in state_dict:
                        state_dict = state_dict["teacher"]
                    # 去除不需要的前缀
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                    self.dino_hypergraph_model.load_state_dict(state_dict, strict=False)
                
                self.dino_hypergraph_model = self.dino_hypergraph_model.to(device).eval()
                
                # 定义标准化变换 - 为超图处理单独命名
                self.dino_hypergraph_normalize = T.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            
            # 提取DINO特征
            batch_size = 32  # 批处理大小
            num_patches = patches_tensor.shape[0]
            feature_dim = 384  # DINO ViT-Small的特征维度
            
            # 初始化特征张量
            patch_features = torch.zeros(num_patches, feature_dim, device=device)
            sd_patch_features = torch.zeros(num_patches, feature_dim, device=device)
            
            with torch.no_grad():
                # 批量处理补丁
                for i in range(0, num_patches, batch_size):
                    batch_end = min(i + batch_size, num_patches)
                    batch_patches = patches_tensor[i:batch_end]
                    batch_sd_patches = sd_patches_tensor[i:batch_end]
                    
                    # 确保数据类型一致性 - 显式转换为float32
                    normalized_patches = torch.stack([self.dino_hypergraph_normalize(patch.float()) for patch in batch_patches])
                    normalized_sd_patches = torch.stack([self.dino_hypergraph_normalize(patch.float()) for patch in batch_sd_patches])

                    # 提取特征
                    batch_features = self.dino_hypergraph_model(normalized_patches)
                    batch_sd_features = self.dino_hypergraph_model(normalized_sd_patches)
                    
                    patch_features[i:batch_end] = batch_features
                    sd_patch_features[i:batch_end] = batch_sd_features
            
            # ---------- 5. 构建/更新超图 ----------
            # 检查是否应更新超图结构
            should_update_hypergraph = (iteration % hypergraph_update_interval == 0)
            
            # 超图键名 - 对每个批次/视角单独处理
            hypergraph_key = f'hypergraph_b{b}'
            
            if not hasattr(self, hypergraph_key) or should_update_hypergraph:
                # 构建新的超图
                # 使用kNN构建超边
                e_list = []
                
                # 计算特征之间的余弦相似度
                norm_features = F.normalize(patch_features, p=2, dim=1)
                similarity_matrix = torch.mm(norm_features, norm_features.t())
                
                # 对每个节点找到最相似的k个节点
                for i in range(num_patches):
                    # 排除自身
                    sim_scores = similarity_matrix[i].clone()
                    sim_scores[i] = -1.0
                    
                    # 找出top-k个相似节点
                    _, indices = torch.topk(sim_scores, k=min(knn_neighbors, num_patches-1))
                    
                    # 创建超边 - 包含节点自身和它的k个邻居
                    hyperedge = [i] + indices.tolist()
                    e_list.append(hyperedge)
                
                # 创建超图
                hypergraph = dhg.Hypergraph(num_v=num_patches, e_list=e_list)
                
                # 获取超图的关联矩阵
                incidence_matrix = torch.zeros(num_patches, len(e_list), device=device)
                for e_idx, e in enumerate(e_list):
                    for v in e:
                        incidence_matrix[v, e_idx] = 1.0
                
                # 缓存超图和关联矩阵
                setattr(self, hypergraph_key, hypergraph)
                setattr(self, f'{hypergraph_key}_incidence', incidence_matrix)
                
                if should_update_hypergraph:
                    print(f"[超图更新] 迭代 {iteration}, 批次 {b}, 构建了新的超图结构，节点数: {num_patches}, 超边数: {len(e_list)}")
            else:
                # 使用缓存的超图
                hypergraph = getattr(self, hypergraph_key)
            
            # ---------- 6. 应用超图卷积 ----------
            # 为每个批次/视角创建一个独立的超图卷积层
            hgnn_key = f'hgnn_conv_b{b}'
            
            if not hasattr(self, hgnn_key):
                # 创建新的超图卷积层
                hgnn_conv = HGNNPConv(
                    in_channels=feature_dim,
                    out_channels=feature_dim,
                    use_bn=False,
                    bias=True
                ).to(device)
                
                # 初始化为接近单位矩阵的权重
                with torch.no_grad():
                    if hasattr(hgnn_conv, 'weight'):
                        nn.init.eye_(hgnn_conv.weight)
                        # 添加少量随机噪声
                        hgnn_conv.weight.data = hgnn_conv.weight.data + 0.05 * torch.randn_like(hgnn_conv.weight.data)
                
                setattr(self, hgnn_key, hgnn_conv)
            
            # 获取超图卷积层
            hgnn_conv = getattr(self, hgnn_key)
            
            # 应用超图卷积
            updated_features = hgnn_conv(patch_features, hypergraph)
            
            # 添加残差连接
            updated_features = updated_features + 0.2 * patch_features
            
            # ---------- 7. 计算超图一致性损失 ----------
            # 比较超图优化后的特征与SD预测的特征
            hypergraph_loss = F.mse_loss(updated_features, sd_patch_features)
            hypergraph_losses.append(hypergraph_loss)
            
            # ---------- 8. 从优化后的特征重建图像 ----------
            # 创建一个全零图像
            reconstructed_image = torch.zeros(current_rgb.shape, device=device)
            weight_map = torch.zeros((1, 1, H, W), device=device)
            
            # 为每个特征找到最相似的原始补丁并放回原位置
            for i, (x1, y1, x2, y2) in enumerate(positions_tensor):
                # 提取位置信息
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 确保边界有效
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 定义权重 - 从边缘到中心的平滑过渡
                patch_h, patch_w = y2-y1+1, x2-x1+1
                weight = torch.ones((1, 1, patch_h, patch_w), device=device)
                
                # 生成与原始RGB补丁相同大小的特征补丁
                patch_size = (patch_h, patch_w)
                
                # 重新缩放特征为RGB通道大小
                feature_rgb = updated_features[i].view(1, feature_dim, 1, 1)
                feature_rgb = feature_rgb.expand(-1, -1, patch_h, patch_w)
                
                # 使用1x1卷积将特征维度转换为3个RGB通道
                if not hasattr(self, f'feat_to_rgb_conv_{feature_dim}'):
                    # 创建特征到RGB的投影层
                    feat_to_rgb = nn.Conv2d(feature_dim, 3, kernel_size=1).to(device)
                    # 初始化权重
                    nn.init.normal_(feat_to_rgb.weight, std=0.02)
                    nn.init.zeros_(feat_to_rgb.bias)
                    setattr(self, f'feat_to_rgb_conv_{feature_dim}', feat_to_rgb)
                
                # 获取特征到RGB的投影层
                feat_to_rgb = getattr(self, f'feat_to_rgb_conv_{feature_dim}')
                
                # 投影到RGB空间
                patch_to_add = feat_to_rgb(feature_rgb)
                
                # 残差方式：原始补丁 + 特征补丁的差异
                orig_patch = current_rgb[0, :, y1:y2+1, x1:x2+1]
                
                # 计算加权和
                alpha = 0.5  # 混合因子
                blended_patch = (1-alpha) * orig_patch + alpha * patch_to_add[0]
                
                # 添加到重建图像
                reconstructed_image[0, :, y1:y2+1, x1:x2+1] += blended_patch
                weight_map[0, 0, y1:y2+1, x1:x2+1] += weight[0, 0]
            
            # 避免除以零
            weight_map = torch.clamp(weight_map, min=1.0)
            
            # 归一化重建图像
            reconstructed_image = reconstructed_image / weight_map.repeat(1, 3, 1, 1)
            
            # 将结果添加到整体输出
            hypergraph_optimized_images.append(reconstructed_image)
            
            # ---------- 9. 可视化（如果需要） ----------
            if save_folder is not None and iteration % guidance_opt.vis_interval == 0:
                vis_dir = Path(save_folder) / 'hypergraph_vis'
                vis_dir.mkdir(exist_ok=True, parents=True)
                
                # 1. 保存图像对比
                plt.figure(figsize=(15, 5))
                
                # 在第1198行附近修改plt.imshow部分:
                plt.subplot(1, 3, 1)
                # 转换为float32确保兼容性
                img_np = current_rgb[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                plt.imshow(img_np)
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                # 转换为float32确保兼容性
                recon_np = reconstructed_image[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                plt.imshow(recon_np)
                plt.title('Hypergraph Optimized (No Downsample)')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                # 转换为float32确保兼容性
                sd_np = current_sd_pred[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                plt.imshow(sd_np)
                plt.title('SD Prediction')
                plt.axis('off')
                
                # 创建补丁可视化图像
                patch_vis = current_rgb[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32).copy()
                
                # 为每个补丁绘制边界框
                for x1, y1, x2, y2 in positions_tensor.cpu().numpy():
                    # 在边界框周围绘制矩形
                    cv2.rectangle(patch_vis, (int(x1), int(y1)), (int(x2), int(y2)), (1, 0, 0), 2)
                
                plt.imshow(patch_vis)
                plt.title(f'Image Patches (No Downsample) - Iter {iteration}')
                plt.axis('off')
                plt.savefig(vis_dir / f'hypergraph_view{b}_iter{iteration}_patches.png')
                plt.close()
        
        # 将所有批次的结果堆叠为一个批次张量
        hypergraph_output = torch.cat(hypergraph_optimized_images, dim=0)
        
        # 计算整个批次的平均超图损失
        avg_hypergraph_loss = sum(hypergraph_losses) / len(hypergraph_losses)
        
        return hypergraph_output, avg_hypergraph_loss

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                        grad_scale=1, use_control_net=True,
                        save_folder:Path=None, iteration=0, warm_up_rate=0, weights=0, 
                        resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None, opt=None):
        # 在数据增强前直接使用DepthAnythingV2从pred_rgb生成深度图
        # try:
        #     from .depth_anything_v2.dpt import DepthAnythingV2
        #     import cv2
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #     import matplotlib
            
        #     print(f'[INFO] Generating depth map with Depth Anything V2')
            
        #     # 配置模型
        #     model_configs = {
        #         'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #         'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        #         'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        #         'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        #     }
            
        #     # 加载模型
        #     depth_anything = DepthAnythingV2(**model_configs['vits'])
        #     # depth_anything.load_state_dict(torch.load('/home/s414e2/CJH/Text-to-3D/Depth-Anything-V2/depth_anything_v2_vitl.pth', map_location='cpu'))
        #     depth_anything.load_state_dict(torch.load('/home/s414e2/CJH/Text-to-3D/LucidDreamer/Depth-Anything-V2/depth_anything_v2_vits.pth', map_location='cpu'))
        #     depth_anything = depth_anything.to(self.device).eval()
            
        #     # 创建一个列表存储所有批次图像的深度图
        #     depth_maps = []

        #     if not hasattr(self, 'depth_cmap'):
        #         self.depth_cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            
        #     with torch.no_grad():
        #         for b in range(pred_rgb.shape[0]):
        #             # 获取当前图像并转换为numpy格式
        #             img = pred_rgb[b].permute(1, 2, 0).cpu().numpy() 
        #             # 将范围从[0,1]转换为[0,255]的uint8
        #             img = (img * 255).astype(np.uint8)
        #             # 转换为BGR格式(OpenCV默认)
        #             img = img[:, :, ::-1].copy()
                    
        #             # 使用depth_anything预测深度图
        #             input_size = 518  # 默认输入大小
        #             depth = depth_anything.infer_image(img, input_size)
                    
        #             # 归一化深度图到[0,1]范围
        #             # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                    
        #             # 转换回PyTorch张量并添加到列表
        #             # depth_tensor = torch.from_numpy(depth).float().to(self.device).unsqueeze(0)  # 添加通道维度
        #             # depth_maps.append(depth_tensor)

        #             # # 按照run.py中的方式处理深度图
        #             # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        #             # depth = depth.astype(np.uint8)

        #             # # 应用彩色映射
        #             # depth = (self.depth_cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        #             # # 转回RGB并归一化
        #             # depth = depth[:, :, ::-1].astype(np.float32) / 255.0
                    
        #             # # 转换为PyTorch张量(通道在前)
        #             # depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
        #             # depth_maps.append(depth_tensor)

        #             # 归一化深度图到[0,1]范围
        #             depth = (depth - depth.min()) / (depth.max() - depth.min())
        #             # 扩展为三通道灰度图
        #             depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                        
        #             # 转换为PyTorch张量(通道在前)
        #             depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
        #             depth_maps.append(depth_tensor)
            
        #     # 将所有深度图堆叠为批次张量
        #     pred_depth_gen = torch.stack(depth_maps, dim=0)  # [B, 1, H, W]
        #     print(f"[INFO] Generated depth map with shape: {pred_depth_gen.shape}")
            
        # except Exception as e:
        #     print(f"[WARNING] Error generating depth map: {e}")
        #     if pred_depth_gen is None:
        #         # 如果生成失败且原始pred_depth为None，创建全零深度图
        #         pred_depth_gen = torch.zeros((pred_rgb.shape[0], 1, pred_rgb.shape[2], pred_rgb.shape[3]), device=pred_rgb.device)


        # 1. 数据增强
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        # 2. 应用超图更新
        # 首先修复update_with_hypergraph方法中的错误
        # hyper_rgbs = self.update_with_multi_view_hypergraph(pred_rgb)

        # 3. 继续原有的编码和处理
        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        # 根据设置选择使用的图像进行latent编码
        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
        # 4. 为超图处理后的图像生成对应的latents
        # hyper_latents, _ = self.encode_imgs(hyper_rgbs.to(self.precision_t))
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8), 
                        dtype=latents.dtype, device=latents.device, 
                        generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), 
                                                                        device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t = int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        # 热身期使用随机时间步，之后使用固定时间步
        # if iteration < opt.warmup_iter:
        #     # 热身阶段：使用随机时间步
        #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), 
        #                         dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        #     if iteration % 100 == 0:
        #         print(f"[ITER {iteration}] 使用随机时间步: {ind_t.item()}")
        # else:
        #     # 精细阶段：使用固定时间步，随迭代逐渐降低噪声水平
        #     remaining_iters = opt.iterations - opt.warmup_iter
        #     current_progress = (iteration - opt.warmup_iter) / remaining_iters
            
        #     # 随着训练进行，噪声从中等逐渐减小到较低水平
        #     # noise_level = max(0.8 - current_progress * 0.4, 0.4)  # 从0.8下降到0.4
        #     noise_level = max(0.5 - current_progress * 0.4, 0.2)  # 从0.5下降到0.2
            
        #     # 计算固定时间步并转换为张量
        #     ind_t_value = int(self.max_step * noise_level)
        #     ind_t = torch.tensor([ind_t_value], dtype=torch.long, device=self.device)[0]
            
        #     if iteration % 100 == 0:
        #         print(f"[ITER {iteration}] 使用固定时间步: {ind_t.item()}, 噪声水平: {noise_level:.2f}")

        # # 计算前一时间步 - 保持原有逻辑
        # ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        # 根据迭代阶段使用不同的固定时间步
        # if iteration < 5000 * 0.3:  # 形状确定阶段
        #     ind_t = int(self.max_step * 0.8)    # 较高噪声，获取基本形状
        # elif iteration < 5000 * 0.7:  # 形状细化阶段
        #     ind_t = int(self.max_step * 0.5)    # 中等噪声，细化形状
        # else:  # 表面细节阶段
        #     ind_t = int(self.max_step * 0.3)    # 较低噪声，优化表面细节
        # # 将整数转换为张量
        # ind_t = torch.tensor([ind_t], dtype=torch.long, device=self.device)[0]
        # ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # 原有的噪声添加和目标计算
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
            latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:

                # 确保ControlNet的所有输入都具有一致的数据类型
                dtype = self.precision_t  # 使用模型设置的精度 

                # 转换输入为统一数据类型
                latent_model_input_typed = latent_model_input.to(dtype)
                tt_typed = tt.to(dtype)
                text_embeddings_typed = text_embeddings.to(dtype)

                pred_depth_input = pred_depth[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input_typed,
                    tt_typed,
                    encoder_hidden_states=text_embeddings_typed,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input_typed, tt_typed, encoder_hidden_states=text_embeddings_typed,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), 
                                    encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8)
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds, weights, B)     

        # 计算预测噪声
        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        # 确定是否应用低通滤波掩码
        lowpass_start_iter = getattr(guidance_opt, 'low_pass_start_iter', 0)  # 默认700次迭代后开始
        
        # 计算梯度
        grad = w(self.alphas[t]) * (pred_noise - target)

        if iteration > lowpass_start_iter:
            with torch.no_grad():
                # 计算SD预测的原始图像x0的latent
                pred_x0_latent_pos = pred_original(self.scheduler, pred_noise, prev_t, prev_latents_noisy)
                # 解码latent为图像
                pred_x0 = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                
                # 提取低通掩码 - 边缘为0，非边缘为1
                lowpass_masks = self.extract_lowpass_mask(pred_x0, save_folder, iteration, guidance_opt.vis_interval)
                
                # 调整掩码到与梯度相同的大小
                lowpass_masks_downsampled = F.interpolate(
                    lowpass_masks, 
                    (grad.shape[2], grad.shape[3]), 
                    mode='nearest'  # 使用最近邻插值保持二值特性
                )
                
                # 应用掩码到梯度 - 边缘区域梯度变为0，非边缘区域保持原始梯度
                grad = grad * lowpass_masks_downsampled
                
                # 打印掩码覆盖率
                if iteration % 100 == 0:
                    mask_coverage = lowpass_masks_downsampled.mean().item() * 100
                    print(f"[低通掩码] 迭代 {iteration}, 非边缘区域覆盖率: {mask_coverage:.2f}%")
                    
                    # 保存掩码可视化（如果需要）
                    if save_folder is not None and iteration % guidance_opt.vis_interval == 0:
                        # 创建彩色掩码，便于可视化
                        mask_vis = F.interpolate(
                            lowpass_masks_downsampled,
                            (resolution[0], resolution[1]),
                            mode='nearest'
                        )
                        
                        # 为了更好的可视化，将掩码转换为热力图色彩
                        colored_mask = torch.zeros(mask_vis.shape[0], 3, mask_vis.shape[2], mask_vis.shape[3], 
                                            device=mask_vis.device)
                        colored_mask[:, 0, :, :] = 1.0 - mask_vis[:, 0]  # 红色通道 - 边缘
                        colored_mask[:, 1, :, :] = mask_vis[:, 0]        # 绿色通道 - 非边缘
                        
                        mask_save_path = os.path.join(save_folder, f"lowpass_mask_{iteration}.jpg")
                        save_image(colored_mask, mask_save_path)

        grad = torch.nan_to_num(grad_scale * grad)
        
        # 计算原始损失
        original_loss = SpecifyGradient.apply(latents, grad)
        
        # 5. 构建SD输出和超图输出之间的一致性损失
        # 首先计算SD预测的原始图像x0
        with torch.no_grad():
            pred_x0_latent = pred_original(self.scheduler, pred_noise, prev_t, prev_latents_noisy)
            pred_x0 = self.decode_latents(pred_x0_latent.type(self.precision_t))

            # 为SD生成的图像预测深度图
            # pred_x0_depth_maps = []
            # if depth_anything is not None:
            #     try:
            #         for b in range(pred_x0.shape[0]):
            #             # 获取当前图像并转换为numpy格式
            #             img = pred_x0[b].permute(1, 2, 0).cpu().numpy() 
            #             # 将范围从[0,1]转换为[0,255]的uint8
            #             img = (img * 255).astype(np.uint8)
            #             # 转换为BGR格式(OpenCV默认)
            #             img = img[:, :, ::-1].copy()
                        
            #             # 使用depth_anything预测深度图
            #             input_size = 518  # 默认输入大小
            #             depth = depth_anything.infer_image(img, input_size)
                        
            #             # 归一化深度图到[0,1]范围
            #             # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                        
            #             # 转换回PyTorch张量并添加到列表
            #             # depth_tensor = torch.from_numpy(depth).float().to(self.device).unsqueeze(0)  # 添加通道维度
            #             # pred_x0_depth_maps.append(depth_tensor)

            #             # # 按照run.py中的方式处理深度图
            #             # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            #             # depth = depth.astype(np.uint8)

            #             # # 应用彩色映射
            #             # depth = (self.depth_cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            #             # # 转回RGB并归一化
            #             # depth = depth[:, :, ::-1].astype(np.float32) / 255.0
                        
            #             # # 转换为PyTorch张量(通道在前)
            #             # depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
            #             # pred_x0_depth_maps.append(depth_tensor)

            #             # 归一化深度图到[0,1]范围
            #             depth = (depth - depth.min()) / (depth.max() - depth.min())
            #             # 扩展为三通道灰度图
            #             depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                            
            #             # 转换为PyTorch张量(通道在前)
            #             depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
            #             pred_x0_depth_maps.append(depth_tensor)
                    
            #         # 将所有深度图堆叠为批次张量
            #         pred_x0_depth = torch.stack(pred_x0_depth_maps, dim=0)  # [B, 1, H, W]
            #         print(f"[INFO] 从SD生成图像生成深度图，形状: {pred_x0_depth.shape}")
            #     except Exception as e:
            #         print(f"[WARNING] 生成SD图像深度图时出错: {e}")
            #         pred_x0_depth = torch.zeros((pred_x0.shape[0], 1, pred_x0.shape[2], pred_x0.shape[3]), device=pred_x0.device)
            # else:
            #     pred_x0_depth = torch.zeros((pred_x0.shape[0], 1, pred_x0.shape[2], pred_x0.shape[3]), device=pred_x0.device)

        # 获取超图一致性损失权重
        # hypergraph_consistency_weight = getattr(guidance_opt, 'hypergraph_consistency_weight', 0.5)

        # 提取边缘注意力掩码
        edge_masks = self.extract_edge_attention_mask(pred_rgb, save_folder, iteration, guidance_opt.vis_interval)

        # 计算标准的超图一致性损失
        # standard_hypergraph_loss = F.l1_loss(hyper_rgbs, pred_x0)

        # 计算边缘引导损失
        edge_guided_loss = self.apply_edge_guided_loss(pred_rgb, pred_x0, edge_masks, guidance_opt, iteration)

        # 计算深度一致性损失 - 新增
        # depth_consistency_weight = getattr(guidance_opt, 'depth_consistency_weight', 1.5)
        # depth_consistency_loss = F.l1_loss(pred_depth_gen, pred_x0_depth)

        # 应用深度一致性损失权重
        # if hasattr(guidance_opt, 'depth_warmup_iters') and iteration < guidance_opt.depth_warmup_iters:
        #     # 热身阶段，逐渐增加权重
        #     current_depth_weight = depth_consistency_weight * (iteration / guidance_opt.depth_warmup_iters)
        # else:
        #     current_depth_weight = depth_consistency_weight

        # 将超图一致性损失添加到总损失中，使用动态权重
        # if hasattr(guidance_opt, 'hypergraph_warmup_iters') and iteration < guidance_opt.hypergraph_warmup_iters:
        #     # 热身阶段，逐渐增加权重
        #     current_weight = hypergraph_consistency_weight * (iteration / guidance_opt.hypergraph_warmup_iters)
        # else:
        #     current_weight = hypergraph_consistency_weight

        # weighted_depth_loss = current_depth_weight * depth_consistency_loss

        hypergraph_output, feature_loss = self.apply_hypergraph_modeling_no_downsample(
            pred_rgb, pred_x0, iteration, guidance_opt, save_folder
        )
        # 计算像素空间损失
        pixel_loss = F.l1_loss(hypergraph_output, pred_x0)
        # 组合这两种损失，使用不同的权重
        feature_weight = 0.8
        pixel_weight = 1.0
        hypergraph_total_loss = feature_weight * feature_loss + pixel_weight * pixel_loss

        # 组合损失：原始损失 + 标准超图损失 + 边缘引导损失
        # 注意：边缘引导损失已经在apply_edge_guided_loss中应用了自己的权重
        # total_loss = original_loss + current_weight * standard_hypergraph_loss + edge_guided_loss
        # total_loss = original_loss
        total_loss = original_loss + edge_guided_loss + hypergraph_total_loss
        
        # 6. 可视化处理
        if iteration % guidance_opt.vis_interval == 0:
        # if iteration % 1 == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder, f"iter_{iteration}_step_{prev_t.item()}.jpg")
            
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True), 
                                        (resolution[0], resolution[1]), 
                                        mode='bilinear', 
                                        align_corners=False).repeat(1, 3, 1, 1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                # 生成边缘强调的图像可视化
                edge_highlighted_rgb = pred_rgb * edge_masks
                # edge_highlighted_hyper = hyper_rgbs * edge_masks
                edge_highlighted_sd = pred_x0 * edge_masks

                # 如果处于低通滤波阶段，添加掩码的可视化
                if iteration > lowpass_start_iter:
                    # 准备低通掩码可视化 - 调整到与其他图像相同的分辨率
                    mask_vis = F.interpolate(
                        lowpass_masks,
                        (resolution[0], resolution[1]),
                        mode='nearest'  # 使用最近邻插值保持二值性质
                    ).repeat(1, 3, 1, 1)  # 复制到三个通道
                    
                    # 将可视化添加到viz_images中
                    viz_images = torch.cat([
                        pred_rgb,                                     # 原始RGB图像
                        edge_masks.repeat(1, 3, 1, 1),               # 边缘掩码
                        edge_highlighted_rgb,                         # 边缘强调的原始图像
                        edge_highlighted_sd,                          # 边缘强调的SD预测
                        pred_depth.repeat(1, 3, 1, 1),                # 深度图
                        pred_alpha.repeat(1, 3, 1, 1),                # Alpha通道
                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),  # 饱和度图
                        latents_rgb,                                  # 原始latent解码图像
                        latents_sp_rgb,                               # SD预测的latent解码图像
                        norm_grad,                                    # 归一化梯度
                        mask_vis,                                     # 低通掩码 (新增)
                        pred_x0_sp,                                   # SD无条件预测
                        pred_x0_pos                                   # SD条件预测
                    ], dim=0)
                
                    save_image(viz_images, save_path_iter)

                # 添加超图处理后的图像到可视化中
                # 添加边缘掩码和边缘强调图像到可视化中
                else:
                    viz_images = torch.cat([
                        pred_rgb,                                     # 原始RGB图像
                        # hyper_rgbs,                                   # 超图处理后的图像
                        edge_masks.repeat(1, 3, 1, 1),               # 边缘掩码 (新增)
                        edge_highlighted_rgb,                         # 边缘强调的原始图像 (新增)
                        # edge_highlighted_hyper,                       # 边缘强调的超图图像 (新增)
                        edge_highlighted_sd,                          # 边缘强调的SD预测 (新增)
                        pred_depth.repeat(1, 3, 1, 1),                # 深度图
                        # pred_depth_gen,            # 原始图像的深度图 (新增)
                        # pred_x0_depth,         # SD生成图像的深度图 (新增)
                        pred_alpha.repeat(1, 3, 1, 1),                # Alpha通道 
                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),  # 饱和度图
                        latents_rgb,                                  # 原始latent解码图像
                        latents_sp_rgb,                               # SD预测的latent解码图像
                        norm_grad,                                    # 归一化梯度
                        pred_x0_sp,                                   # SD无条件预测
                        pred_x0_pos                                   # SD条件预测
                    ], dim=0)
                    
                    save_image(viz_images, save_path_iter)
                
                # 输出超图一致性损失值
                # print(f"Iteration {iteration}, Hypergraph Consistency Loss: {standard_hypergraph_loss.item():.6f}")
        
        return total_loss

    def train_step_perpneg_new(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1, use_control_net=True,
                    save_folder:Path=None, iteration=0, warm_up_rate=0, weights=0, 
                    resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None):
        # 在数据增强前直接使用DepthAnythingV2从pred_rgb生成深度图
        try:
            from .depth_anything_v2.dpt import DepthAnythingV2
            import cv2
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib
            
            print(f'[INFO] Generating depth map with Depth Anything V2')
            
            # 配置模型
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            # 加载模型
            depth_anything = DepthAnythingV2(**model_configs['vits'])
            depth_anything.load_state_dict(torch.load('/home/s414e2/CJH/Text-to-3D/LucidDreamer/Depth-Anything-V2/depth_anything_v2_vits.pth', map_location='cpu'))
            depth_anything = depth_anything.to(self.device).eval()
            
            # 创建一个列表存储所有批次图像的深度图
            depth_maps = []
            
            with torch.no_grad():
                for b in range(pred_rgb.shape[0]):
                    # 获取当前图像并转换为numpy格式
                    img = pred_rgb[b].permute(1, 2, 0).cpu().numpy() 
                    # 将范围从[0,1]转换为[0,255]的uint8
                    img = (img * 255).astype(np.uint8)
                    # 转换为BGR格式(OpenCV默认)
                    img = img[:, :, ::-1].copy()
                    
                    # 使用depth_anything预测深度图
                    input_size = 518  # 默认输入大小
                    depth = depth_anything.infer_image(img, input_size)
                    
                    # 归一化深度图到[0,1]范围
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    # 扩展为三通道灰度图
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                        
                    # 转换为PyTorch张量(通道在前)
                    depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
                    depth_maps.append(depth_tensor)
            
            # 将所有深度图堆叠为批次张量
            pred_depth_gen = torch.stack(depth_maps, dim=0)  # [B, 3, H, W]
            print(f"[INFO] Generated depth map with shape: {pred_depth_gen.shape}")
            
        except Exception as e:
            print(f"[WARNING] Error generating depth map: {e}")
            if pred_depth_gen is None:
                # 如果生成失败且原始pred_depth为None，创建全零深度图
                pred_depth_gen = torch.zeros((pred_rgb.shape[0], 3, pred_rgb.shape[2], pred_rgb.shape[3]), device=pred_rgb.device)

        # 1. 数据增强
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        # 3. 继续原有的编码和处理
        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        # ---------- 两种路径：RGB和深度图 ----------
        
        # RGB路径 - 从RGB图像生成latents
        if as_latent:
            rgb_latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            rgb_latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
        # 深度图路径 - 从深度图生成latents
        depth_latents, _ = self.encode_imgs(pred_depth_gen.to(self.precision_t))
        
        # 生成和共享噪声
        if guidance_opt.fix_noise:
            if not hasattr(self, 'noise_temp') or self.noise_temp is None:
                self.noise_temp = torch.randn((rgb_latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8), 
                                dtype=rgb_latents.dtype, device=rgb_latents.device, 
                                generator=self.noise_gen)
            noise = self.noise_temp
        else:
            noise = torch.randn((rgb_latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8), 
                            dtype=rgb_latents.dtype, device=rgb_latents.device, 
                            generator=self.noise_gen)
        
        # 处理文本嵌入
        weights = weights.reshape(-1) if weights is not None else None
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

        # 确定时间步长
        if guidance_opt.annealing_intervals:
            current_delta_t = int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        # ---------- 生成RGB和深度的噪声图像和目标 ----------
        with torch.no_grad():
            # RGB路径噪声添加
            if not self.ism:
                rgb_prev_latents_noisy = self.scheduler.add_noise(rgb_latents, noise, prev_t)
                rgb_latents_noisy = self.scheduler.add_noise(rgb_latents, noise, t)
                rgb_target = noise
                
                # 深度图路径噪声添加（使用相同的噪声和时间步）
                depth_prev_latents_noisy = self.scheduler.add_noise(depth_latents, noise, prev_t)
                depth_latents_noisy = self.scheduler.add_noise(depth_latents, noise, t)
                depth_target = noise
            else:
                # ISM方法 - RGB路径
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)
                
                # RGB噪声过程
                _, rgb_prev_latents_noisy, rgb_pred_scores_xs = self.add_noise_with_cfg(
                    rgb_latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                    guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta
                )
                _, rgb_latents_noisy, rgb_pred_scores_xt = self.add_noise_with_cfg(
                    rgb_prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                    guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True
                )
                rgb_pred_scores = rgb_pred_scores_xt + rgb_pred_scores_xs
                rgb_target = rgb_pred_scores[0][1]
                
                # 深度图噪声过程
                _, depth_prev_latents_noisy, depth_pred_scores_xs = self.add_noise_with_cfg(
                    depth_latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                    guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta
                )
                _, depth_latents_noisy, depth_pred_scores_xt = self.add_noise_with_cfg(
                    depth_prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                    guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True
                )
                depth_pred_scores = depth_pred_scores_xt + depth_pred_scores_xs
                depth_target = depth_pred_scores[0][1]

        # ---------- RGB路径噪声预测 ----------
        with torch.no_grad():
            # 准备模型输入
            rgb_latent_model_input = rgb_latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            tt = t.reshape(1, 1).repeat(rgb_latent_model_input.shape[0], 1).reshape(-1)

            rgb_latent_model_input = self.scheduler.scale_model_input(rgb_latent_model_input, tt[0])
            
            # 使用ControlNet预测RGB噪声
            if use_control_net:
                dtype = self.precision_t
                rgb_latent_model_input = rgb_latent_model_input.to(dtype)
                tt_typed = tt.to(dtype)
                text_embeddings_typed = text_embeddings.to(dtype)
                
                pred_depth_input = pred_depth[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).to(dtype)
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    rgb_latent_model_input,
                    tt_typed,
                    encoder_hidden_states=text_embeddings_typed,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                rgb_unet_output = self.unet(
                    rgb_latent_model_input, 
                    tt_typed, 
                    encoder_hidden_states=text_embeddings_typed,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
            else:
                rgb_unet_output = self.unet(
                    rgb_latent_model_input.to(self.precision_t), 
                    tt.to(self.precision_t), 
                    encoder_hidden_states=text_embeddings.to(self.precision_t)
                ).sample
            
            # 处理UNet输出
            rgb_unet_output = rgb_unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8)
            rgb_noise_pred_uncond, rgb_noise_pred_text = rgb_unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8), rgb_unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            rgb_delta_noise_preds = rgb_noise_pred_text - rgb_noise_pred_uncond.repeat(K, 1, 1, 1)
            
            # 使用perpendicular_aggregator处理
            if rgb_delta_noise_preds.shape[0] > 1:  # 如果有多个文本条件
                rgb_delta_DSD = weighted_perpendicular_aggregator(rgb_delta_noise_preds, weights, B)
            else:
                rgb_delta_DSD = rgb_delta_noise_preds

        # ---------- 深度图路径噪声预测 ----------
        with torch.no_grad():
            # 准备模型输入
            depth_latent_model_input = depth_latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            
            depth_latent_model_input = self.scheduler.scale_model_input(depth_latent_model_input, tt[0])
            
            # 使用相同的ControlNet和条件进行深度图噪声预测
            if use_control_net:
                # 重用已有的控制网络资源
                depth_unet_output = self.unet(
                    depth_latent_model_input.to(dtype), 
                    tt_typed, 
                    encoder_hidden_states=text_embeddings_typed,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
            else:
                depth_unet_output = self.unet(
                    depth_latent_model_input.to(self.precision_t), 
                    tt.to(self.precision_t), 
                    encoder_hidden_states=text_embeddings.to(self.precision_t)
                ).sample
            
            # 处理UNet输出
            depth_unet_output = depth_unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8)
            depth_noise_pred_uncond, depth_noise_pred_text = depth_unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8), depth_unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8)
            depth_delta_noise_preds = depth_noise_pred_text - depth_noise_pred_uncond.repeat(K, 1, 1, 1)
            
            # 使用perpendicular_aggregator处理
            if depth_delta_noise_preds.shape[0] > 1:  # 如果有多个文本条件
                depth_delta_DSD = weighted_perpendicular_aggregator(depth_delta_noise_preds, weights, B)
            else:
                depth_delta_DSD = depth_delta_noise_preds

        # ---------- 选择使用哪个噪声路径（深度优先，后切换到RGB） ----------
        depth_only_iters = getattr(guidance_opt, 'depth_only_iters', 1500)  # 默认前1000次迭代只用深度图
        
        # 根据迭代次数选择使用哪个噪声预测流程
        if iteration < depth_only_iters:
            # 前期阶段：使用深度图噪声预测
            print(f"[迭代 {iteration}] 使用深度图噪声更新（深度优先阶段）")
            pred_noise = depth_noise_pred_uncond + guidance_opt.guidance_scale * depth_delta_DSD
            target = depth_target
            prev_latents_noisy = depth_prev_latents_noisy
            latents = depth_latents
        else:
            # 后期阶段：使用RGB噪声预测
            print(f"[迭代 {iteration}] 切换到RGB噪声更新（外观优化阶段）")
            pred_noise = rgb_noise_pred_uncond + guidance_opt.guidance_scale * rgb_delta_DSD
            target = rgb_target
            prev_latents_noisy = rgb_prev_latents_noisy
            latents = rgb_latents
        
        # 计算梯度
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (pred_noise - target)
        grad = torch.nan_to_num(grad_scale * grad)
        
        # 应用梯度
        original_loss = SpecifyGradient.apply(latents, grad)
        
        # 提取边缘注意力掩码
        edge_masks = self.extract_edge_attention_mask(pred_rgb, save_folder, iteration, guidance_opt.vis_interval)

        # ---------- 额外处理 - SD预测图像和深度图生成 ----------
        with torch.no_grad():
            if iteration < depth_only_iters:
                # 使用深度路径的噪声预测来生成深度还原图像
                pred_x0_latent = pred_original(self.scheduler, pred_noise, prev_t, prev_latents_noisy)
            else:
                # 使用RGB路径的噪声预测来生成RGB还原图像
                pred_x0_latent = pred_original(self.scheduler, pred_noise, prev_t, prev_latents_noisy)
            
            # 解码latents为图像
            pred_x0 = self.decode_latents(pred_x0_latent.type(self.precision_t))

            # 为SD生成的图像预测深度图
            pred_x0_depth_maps = []
            if depth_anything is not None:
                try:
                    for b in range(pred_x0.shape[0]):
                        # 获取当前图像并转换为numpy格式
                        img = pred_x0[b].permute(1, 2, 0).cpu().numpy() 
                        img = (img * 255).astype(np.uint8)
                        img = img[:, :, ::-1].copy()
                        
                        # 使用depth_anything预测深度图
                        input_size = 518
                        depth = depth_anything.infer_image(img, input_size)
                        
                        # 归一化深度图到[0,1]范围
                        depth = (depth - depth.min()) / (depth.max() - depth.min())
                        # 扩展为三通道灰度图
                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                            
                        # 转换为PyTorch张量(通道在前)
                        depth_tensor = torch.from_numpy(depth).float().to(self.device).permute(2, 0, 1)
                        pred_x0_depth_maps.append(depth_tensor)
                    
                    # 将所有深度图堆叠为批次张量
                    pred_x0_depth = torch.stack(pred_x0_depth_maps, dim=0)
                    print(f"[INFO] 从SD生成图像生成深度图，形状: {pred_x0_depth.shape}")
                except Exception as e:
                    print(f"[WARNING] 生成SD图像深度图时出错: {e}")
                    pred_x0_depth = torch.zeros((pred_x0.shape[0], 3, pred_x0.shape[2], pred_x0.shape[3]), device=pred_x0.device)
            else:
                pred_x0_depth = torch.zeros((pred_x0.shape[0], 3, pred_x0.shape[2], pred_x0.shape[3]), device=pred_x0.device)

        # 计算边缘引导损失
        edge_guided_loss = self.apply_edge_guided_loss(pred_rgb, pred_x0, edge_masks, guidance_opt, iteration)

        # 计算深度一致性损失
        depth_consistency_weight = getattr(guidance_opt, 'depth_consistency_weight', 1.5)
        depth_consistency_loss = F.l1_loss(pred_depth_gen, pred_x0_depth)

        # 应用深度一致性损失权重
        if hasattr(guidance_opt, 'depth_warmup_iters') and iteration < guidance_opt.depth_warmup_iters:
            # 热身阶段，逐渐增加权重
            current_depth_weight = depth_consistency_weight * (iteration / guidance_opt.depth_warmup_iters)
        else:
            current_depth_weight = depth_consistency_weight

        # 组合损失
        total_loss = original_loss + edge_guided_loss + current_depth_weight * depth_consistency_loss
        
        # ---------- 可视化 ----------
        if iteration % guidance_opt.vis_interval == 0:
            # 使用当前阶段的噪声预测
            if iteration < depth_only_iters:
                noise_pred_post = depth_noise_pred_uncond + guidance_opt.guidance_scale * depth_delta_DSD
                latents_viz = depth_latents
                prev_latents_viz = depth_prev_latents_noisy
            else:
                noise_pred_post = rgb_noise_pred_uncond + guidance_opt.guidance_scale * rgb_delta_DSD
                latents_viz = rgb_latents
                prev_latents_viz = rgb_prev_latents_noisy
            
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder, f"iter_{iteration}_step_{prev_t.item()}.jpg")
            
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, rgb_noise_pred_uncond, prev_t, prev_latents_viz)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_viz)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True), 
                                        (resolution[0], resolution[1]), 
                                        mode='bilinear', 
                                        align_corners=False).repeat(1, 3, 1, 1)

                latents_rgb = F.interpolate(lat2rgb(latents_viz), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                # 生成边缘强调的图像可视化
                edge_highlighted_rgb = pred_rgb * edge_masks
                edge_highlighted_sd = pred_x0 * edge_masks
                
                viz_images = torch.cat([
                    pred_rgb,                                     # 原始RGB图像
                    edge_masks.repeat(1, 3, 1, 1),               # 边缘掩码
                    edge_highlighted_rgb,                         # 边缘强调的原始图像
                    edge_highlighted_sd,                          # 边缘强调的SD预测
                    pred_depth.repeat(1, 3, 1, 1),                # 深度图
                    pred_depth_gen,                               # 原始图像的深度图 (新增)
                    pred_x0_depth,                                # SD生成图像的深度图 (新增)
                    pred_alpha.repeat(1, 3, 1, 1),                # Alpha通道 
                    rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),  # 饱和度图
                    latents_rgb,                                  # 原始latent解码图像
                    latents_sp_rgb,                               # SD预测的latent解码图像
                    norm_grad,                                    # 归一化梯度
                    pred_x0_sp,                                   # SD无条件预测
                    pred_x0_pos                                   # SD条件预测
                ], dim=0)
                
                save_image(viz_images, save_path_iter)
                
                # 打印当前状态
                print(f"迭代 {iteration}, 使用路径: {'深度' if iteration < depth_only_iters else 'RGB'}, 深度损失: {depth_consistency_loss.item():.6f}")
        
        return total_loss


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
                pred_depth = pred_depth[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth,
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