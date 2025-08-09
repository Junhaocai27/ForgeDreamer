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
import dhg
from collections import deque
import torchvision.transforms as T
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from sklearn.manifold import TSNE
# import seaborn as sns

from .sd_step import *

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

class HypergraphProcessor(nn.Module):
    """基于DHG的超图处理器 - 使用DINOv2特征矩阵行拼接（显存优化版）"""
    def __init__(self, feature_dim=256, hidden_dim=128, device='cuda:0'):  # 🔥 减少特征维度
        super(HypergraphProcessor, self).__init__()
        
        # 🔥 保存设备信息
        self.device = device
        
        # 🔥 使用DINOv2 ViT-B/14模型提取特征
        print("🔧 从本地加载DINOv2 ViT-B/14模型...")
        
        try:
            # 🔥 方案1: 直接加载完整的DINOv2模型文件
            local_model_path = "/root/LucidDreamer/Hyper3DG/dinov2_vitb14_reg4_pretrain.pth"
            
            if os.path.exists(local_model_path):
                print(f"📁 从本地加载完整DINOv2模型: {local_model_path}")
                checkpoint = torch.load(local_model_path, map_location='cpu')
                
                # 🔥 创建DINOv2 ViT-B/14模型结构
                import timm
                self.dino_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)
                
                # 处理权重加载
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 移除不需要的键（如果存在）
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    if k.startswith('backbone.'):
                        k = k[9:]
                    filtered_state_dict[k] = v
                
                missing_keys, unexpected_keys = self.dino_model.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    print(f"⚠️  缺失的权重键: {len(missing_keys)} 个")
                if unexpected_keys:
                    print(f"⚠️  意外的权重键: {len(unexpected_keys)} 个")
                    
                print("✅ 本地DINOv2权重加载成功")
                
            else:
                print(f"❌ 本地模型文件不存在: {local_model_path}")
                print("🔄 回退到在线加载...")
                import timm
                self.dino_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
                
        except Exception as e:
            print(f"❌ 本地DINOv2加载失败: {e}")
            print("🔄 回退到在线加载...")
            try:
                import timm
                self.dino_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
            except:
                # 最终回退方案 - 使用torch.hub
                print("🔄 使用torch.hub加载DINOv2...")
                self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg4')
        
        # 🔥 将模型移动到指定设备
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()
        
        # 移除分类头
        if hasattr(self.dino_model, 'head'):
            self.dino_model.head = nn.Identity()
        elif hasattr(self.dino_model, 'classifier'):
            self.dino_model.classifier = nn.Identity()
        
        # 冻结DINOv2参数
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        print("✅ DINOv2模型加载完成")
        
        # 🔥 自动检测输入尺寸
        try:
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = self.dino_model.forward_features(test_input)
            input_size = 224
            print(f"✅ DINOv2模型接受224x224输入")
        except Exception as e:
            if "518" in str(e):
                input_size = 518
                print(f"⚠️ DINOv2模型需要518x518输入，自动调整...")
            else:
                input_size = 224
                print(f"⚠️ 无法确定输入尺寸，使用默认224x224")
        
        self.input_size = input_size
        self.downsample = nn.AdaptiveAvgPool2d((input_size, input_size))
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 🔥 计算patch配置
        patch_size = 14
        self.num_patches = (input_size // patch_size) ** 2
        self.dino_patch_dim = 768
        
        print(f"📊 DINOv2配置: 输入尺寸={input_size}x{input_size}, Patch数量={self.num_patches}, 特征维度={self.dino_patch_dim}")
        
        # 🔥 显存优化：不再拼接所有patch特征，而是使用平均池化
        self.feature_pooling = nn.AdaptiveAvgPool1d(feature_dim)  # 直接池化到目标维度
        
        # 🔥 轻量级特征降维网络
        self.feature_dim_reducer = nn.Sequential(
            nn.Linear(self.dino_patch_dim, feature_dim),  # 直接降维
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        
        # 🔥 轻量级HGNN网络
        from dhg.nn.convs.hypergraphs import HGNNConv
        
        self.hgnn_layers = nn.ModuleList([
            HGNNConv(feature_dim, hidden_dim, use_bn=False, drop_rate=0.1),  # 关闭BN节省显存
            HGNNConv(hidden_dim, feature_dim, use_bn=False, drop_rate=0.1)
        ]).to(device)
        
        self.activation = nn.ReLU()
        
        # 🔥 减少缓存队列大小
        self.rendered_queue = deque(maxlen=3)  # 从4改为3
        self.sd_queue = deque(maxlen=3)
        
        # 🔥 循环重建参数
        self.cycle_count = 0
        self.rebuild_cycle = 200  # 增加重建周期，减少计算频率
        self.current_hypergraph_rendered = None
        self.current_hypergraph_sd = None
        
    def preprocess_images(self, images):
        """图像预处理 - 显存优化版"""
        if images.dim() == 3:
            images = images.unsqueeze(0)
            
        B, C, H, W = images.shape
        
        # 处理通道数
        if C == 1:
            images = images.repeat(1, 3, 1, 1)
        elif C == 4:
            images = images[:, :3, :, :]
        elif C != 3:
            images = images[:, :3, :, :]
            
        # 归一化
        images = torch.clamp(images, 0, 1)
        images = self.downsample(images)
        
        # 批量标准化
        normalized_images = torch.stack([self.normalize(img) for img in images])
        
        return normalized_images
    
    def extract_dinov2_patch_features(self, images):
        """
        🔥 显存优化：使用平均池化而非拼接所有patch特征
        """
        try:
            images = self.preprocess_images(images)
            images = images.to(self.device)
            
            with torch.no_grad():
                if hasattr(self.dino_model, 'forward_features'):
                    features = self.dino_model.forward_features(images)  # [B, num_patches+1, 768]
                    patch_features = features[:, 1:, :]  # [B, num_patches, 768]
                else:
                    features = self.dino_model.get_intermediate_layers(images, n=1)[0]
                    patch_features = features
                
                # 🔥 显存优化：使用平均池化代替拼接
                # 原来: [B, num_patches, 768] -> [B, num_patches*768]
                # 现在: [B, num_patches, 768] -> [B, 768] 通过平均池化
                pooled_features = torch.mean(patch_features, dim=1)  # [B, 768]
                
            return pooled_features, patch_features
            
        except Exception as e:
            print(f"DINOv2特征提取错误: {e}")
            B = images.shape[0]
            fake_pooled = torch.randn(B, self.dino_patch_dim, device=self.device)
            fake_patch = torch.randn(B, self.num_patches, self.dino_patch_dim, device=self.device)
            return fake_pooled, fake_patch
    
    def extract_features(self, images):
        """
        🔥 显存优化的特征提取
        """
        try:
            pooled_features, patch_features = self.extract_dinov2_patch_features(images)  # [B, 768], [B, patches, 768]
            pooled_features = pooled_features.to(self.device)
            
            # 简单降维
            reduced_features = self.feature_dim_reducer(pooled_features)  # [B, feature_dim]
            
            return reduced_features, patch_features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            B = images.shape[0] if hasattr(images, 'shape') else 1
            fake_reduced = torch.randn(B, self.feature_dim_reducer[-1].out_features, device=self.device)
            fake_patch = torch.randn(B, self.num_patches, self.dino_patch_dim, device=self.device)
            return fake_reduced, fake_patch
    
    def build_dhg_hypergraph(self, features, k=2, force_rebuild=False):  # 🔥 减少k值
        """显存优化的超图构建"""
        n = features.shape[0]  # n=3 (减少节点数)
        features = features.to(self.device)
        
        should_rebuild = force_rebuild or (self.cycle_count % self.rebuild_cycle == 0)
        
        if n < 2:
            edge_list = [[i] for i in range(n)]
            return dhg.Hypergraph(n, edge_list)
        
        # 🔥 简化的相似度计算
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        
        edge_list = []
        k = min(k, n-1)
        
        _, topk_indices = torch.topk(similarity, k+1, dim=1)
        
        for i in range(n):
            edge = topk_indices[i].cpu().tolist()
            edge_list.append(edge)
        
        # 🔥 简化策略：只保留基本的连接策略
        if n >= 3:
            edge_list.append(list(range(n)))
        
        # 去重
        unique_edges = []
        for edge in edge_list:
            sorted_edge = sorted(edge)
            if sorted_edge not in unique_edges and len(sorted_edge) > 1:
                unique_edges.append(sorted_edge)
        
        if not unique_edges:
            unique_edges = [list(range(n))]
        
        return dhg.Hypergraph(n, unique_edges)
    
    def apply_hgnn(self, node_features, hypergraph):
        """显存优化的HGNN前向传播"""
        x = node_features.to(self.device)
        
        for i, layer in enumerate(self.hgnn_layers):
            x = layer(x, hypergraph)
            if i < len(self.hgnn_layers) - 1:
                x = self.activation(x)
        
        return x
    
    def forward(self, rendered_imgs, sd_imgs, save_folder=None, iteration=0):
        """
        🔥 显存优化的前向传播
        """
        try:
            # 🔥 减少batch size处理
            batch_size = min(rendered_imgs.shape[0], 2)  # 限制batch size
            rendered_imgs = rendered_imgs[:batch_size].to(self.device)
            sd_imgs = sd_imgs[:batch_size].to(self.device)
            
            # 🔥 逐个处理图像以节省显存
            rendered_features_list = []
            sd_features_list = []
            
            with torch.no_grad():  # 🔥 不需要梯度的部分都加上no_grad
                for i in range(batch_size):
                    r_feat, _ = self.extract_features(rendered_imgs[i:i+1])
                    s_feat, _ = self.extract_features(sd_imgs[i:i+1])
                    rendered_features_list.append(r_feat[0])
                    sd_features_list.append(s_feat[0])
                    
                    # 🔥 立即清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 滑窗更新
            for feat in rendered_features_list:
                self.rendered_queue.append(feat)
            for feat in sd_features_list:
                self.sd_queue.append(feat)
            
            # 队列长度检查
            if len(self.rendered_queue) < 3:  # 改为3
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            rendered_batch = torch.stack(list(self.rendered_queue))
            sd_batch = torch.stack(list(self.sd_queue))
            
            rendered_batch = rendered_batch.to(self.device)
            sd_batch = sd_batch.to(self.device)
            
            self.cycle_count += 1
            should_rebuild = (self.cycle_count % self.rebuild_cycle == 0)
            
            # 构建超图
            hg_rendered = self.build_dhg_hypergraph(rendered_batch, force_rebuild=should_rebuild)
            hg_sd = self.build_dhg_hypergraph(sd_batch, force_rebuild=should_rebuild)
            
            # 🔥 减少可视化频率
            if save_folder is not None and iteration % 200 == 0:  # 从100改为200
                try:
                    vis_save_path = os.path.join(save_folder, f"hypergraph_vis_{iteration}.png")
                    # 只可视化超图结构，跳过特征可视化以节省显存
                    self.visualize_hypergraph(hg_rendered, rendered_batch, vis_save_path, iteration, "rendered")
                except Exception as e:
                    print(f"可视化错误: {e}")
            
            # HGNN前向传播
            rendered_updated = self.apply_hgnn(rendered_batch, hg_rendered)
            sd_updated = self.apply_hgnn(sd_batch, hg_sd)
            
            # 🔥 简化损失计算
            rendered_norm = F.normalize(rendered_updated, p=2, dim=1)
            sd_norm = F.normalize(sd_updated, p=2, dim=1)
            
            # 只使用最重要的损失项
            pairwise_loss = 1 - torch.mean(torch.sum(rendered_norm * sd_norm, dim=1))
            
            # 清理显存
            del rendered_updated, sd_updated, rendered_norm, sd_norm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return pairwise_loss
            
        except Exception as e:
            print(f"超图处理错误: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = "/root/LucidDreamer/stable-diffusion-2-1-base"
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t, local_files_only=True)

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
        
        # 🔥 显存优化的超图处理器初始化
        print("🔧 初始化轻量级DHG超图处理器...")
        self.hypergraph_processor = HypergraphProcessor(
            feature_dim=128,  # 进一步减少特征维度
            hidden_dim=64,   # 减少隐藏层维度
            device=device
        ).to(device)
        print("✅ 轻量级DHG超图处理器初始化完成")

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

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                       grad_scale=1,use_control_net=False,
                       save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                       resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):

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
                pred_depth_input = pred_depth[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
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
        latents_noisy = latents_noisy.repeat(4, 1, 1, 1)  # 扩展 4 倍，使 batch 变成 16
        pred_noise = pred_noise.repeat(4, 1, 1, 1)
        pred_noise_2 = self.unet((latents_noisy + pred_noise).to(self.precision_t), t.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
        final_pred = 0.5 * (pred_noise + pred_noise_2)
        final_pred = final_pred[:target.shape[0]]
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (final_pred - target)
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        adaptive_scale = torch.clamp(1 / (grad_norm + 1e-6), min=0.1, max=10.0)
        grad = grad * adaptive_scale
        
        grad = torch.nan_to_num(grad_scale * grad)
        
        # 🔥 DINO拼接特征DHG超图损失计算 (修改部分)
        # 🔥 显存优化的超图损失计算
        hypergraph_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 🔥 减少超图处理频率
        if iteration % 5 == 0:  # 每5次迭代才计算一次超图损失
            try:
                with torch.no_grad():
                    # 🔥 减少处理的图像数量
                    pred_rgb_small = pred_rgb[:1]  # 只处理第一张图
                    pred_x0_latent = (latents_noisy[:1] - torch.sqrt(1 - self.alphas[t]) * final_pred[:1]) / torch.sqrt(self.alphas[t])
                    sd_generated_imgs = self.decode_latents(pred_x0_latent.type(self.precision_t))
                    
                    # 计算超图损失
                    hypergraph_loss = self.hypergraph_processor(pred_rgb_small, sd_generated_imgs, save_folder, iteration)
                    
                    # 立即清理显存
                    del pred_x0_latent, sd_generated_imgs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"超图处理错误: {e}")
                hypergraph_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 主损失计算
        main_loss = SpecifyGradient.apply(latents, grad)
        
        # 🔥 减少超图损失权重
        hypergraph_weight = getattr(guidance_opt, 'hypergraph_weight', 0.05)  # 从0.1减少到0.05
        total_loss = main_loss + hypergraph_weight * hypergraph_loss

        # 可视化部分
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

        # 🔥 可选：损失信息打印
        if iteration % 50 == 0:
            cycle_info = f"Cycle: {self.hypergraph_processor.cycle_count}"
            rebuild_info = f"Next rebuild in: {self.hypergraph_processor.rebuild_cycle - (self.hypergraph_processor.cycle_count % self.hypergraph_processor.rebuild_cycle)}"
            print(f"Iter {iteration}: Main Loss: {main_loss.item():.6f}, DINO-DHG Loss: {hypergraph_loss.item():.6f} | {cycle_info} | {rebuild_info}")

        return total_loss  # 返回总损失


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