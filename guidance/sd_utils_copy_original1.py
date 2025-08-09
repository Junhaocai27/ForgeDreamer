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

# 🔥 添加DHG导入
try:
    import dhg
    from dhg import Hypergraph
    from dhg.nn import HGNNConv
    DHG_AVAILABLE = True
    print("[INFO] DHG library available for hypergraph operations")
except ImportError:
    DHG_AVAILABLE = False
    print("[WARNING] DHG library not available, hypergraph features disabled")


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
        
        # 🔥 添加超图模块（无需训练）
        self.use_hypergraph = DHG_AVAILABLE
        if self.use_hypergraph:
            try:
                # 🔥 强制使用float32以避免类型问题
                hypergraph_dtype = torch.float32
                
                # 超图卷积用于latents增强（固定权重）
                self.latent_hypergraph_conv = HGNNConv(
                    in_channels=4,     # latent维度
                    out_channels=4,    # 输出维度
                    bias=True,
                    drop_rate=0.1
                ).to(device).to(hypergraph_dtype)  # 🔥 强制使用float32
                
                # 超图卷积用于梯度增强（固定权重）
                self.grad_hypergraph_conv = HGNNConv(
                    in_channels=4,     # 梯度维度
                    out_channels=4,    # 输出维度
                    bias=True,
                    drop_rate=0.1
                ).to(device).to(hypergraph_dtype)  # 🔥 强制使用float32
                
                # 🔥 设置为eval模式，不进行训练
                self.latent_hypergraph_conv.eval()
                self.grad_hypergraph_conv.eval()
                
                # 🔥 冻结参数
                for param in self.latent_hypergraph_conv.parameters():
                    param.requires_grad = False
                for param in self.grad_hypergraph_conv.parameters():
                    param.requires_grad = False
                
                # 🔥 确认数据类型
                print(f'[DEBUG] Hypergraph conv dtype: {next(self.latent_hypergraph_conv.parameters()).dtype}')
                print(f'[DEBUG] Precision type: {self.precision_t}')
                
                print(f'[INFO] DHG hypergraph modules initialized (no training)!')
                
            except Exception as e:
                print(f'[WARNING] Failed to initialize hypergraph modules: {e}')
                self.use_hypergraph = False
        
        # 🔥 历史存储（用于超图重建）
        self.timestep_history = []      # 时间步历史
        self.grad_norm_history = []     # 梯度范数历史
        self.loss_history = []          # 损失历史
        self.similarity_history = []    # 相似性历史
        self.latent_history = []        # latent历史
        self.max_history = 8            # 最大历史长度
        
        # 🔥 超图重建参数
        self.rebuild_interval = 50      # 每50次迭代重建一次超图
        self.last_rebuild_iter = 0      # 上次重建的迭代数

        print(f'[INFO] loaded stable diffusion!')
    
        # 🔥 添加超图相关方法
    def build_spatial_hypergraph(self, tensor, sample_ratio=0.3):
        """构建基于空间邻接的超图（每次重建）"""
        if not self.use_hypergraph:
            return None, None
            
        try:
            B, C, H, W = tensor.shape
            
            # 🔥 将tensor转换为float32进行计算，避免精度问题
            if tensor.dtype == torch.float16:
                tensor_calc = tensor.float()
            else:
                tensor_calc = tensor
            
            node_features = tensor_calc.view(B, C, -1).transpose(1, 2).reshape(-1, C)  # [B*H*W, C]
            num_nodes = node_features.shape[0]
            
            # 🔥 构建超边：每个2x2邻域作为一个超边
            hyperedges = []
            step = max(1, int(1 / sample_ratio))  # 采样步长
            
            for b in range(B):
                for h in range(0, H, step):
                    for w in range(0, W, step):
                        edge_nodes = []
                        for dh in [0, min(1, H-1-h)]:
                            for dw in [0, min(1, W-1-w)]:
                                nh, nw = h + dh, w + dw
                                node_idx = b * H * W + nh * W + nw
                                edge_nodes.append(int(node_idx))  # 确保是int类型
                        
                        if len(edge_nodes) > 1:
                            hyperedges.append(edge_nodes)
            
            # 🔥 基于特征相似性构建额外超边（简化版本以避免问题）
            if num_nodes > 16 and num_nodes < 1024:  # 限制大小以避免内存问题
                sample_size = min(num_nodes // 8, 16)  # 减少采样大小
                
                # 在CPU上进行采样以避免设备问题
                sampled_indices = torch.randperm(num_nodes)[:sample_size]
                
                with torch.no_grad():
                    # 转移到CPU进行相似性计算
                    sampled_features = node_features[sampled_indices].cpu()
                    
                    # 简化的相似性计算
                    for i in range(min(sample_size, 8)):  # 进一步限制
                        for j in range(i+1, min(sample_size, 8)):
                            sim = F.cosine_similarity(
                                sampled_features[i:i+1], 
                                sampled_features[j:j+1], 
                                dim=1
                            ).item()
                            
                            if sim > 0.8:  # 提高阈值
                                edge_nodes = [sampled_indices[i].item(), sampled_indices[j].item()]
                                hyperedges.append(edge_nodes)
            
            if not hyperedges:
                # 如果没有超边，创建一些基本的邻接超边
                for i in range(min(num_nodes, 10)):
                    hyperedges.append([i, (i+1) % num_nodes])
            
            # 🔥 确保所有超边索引都是python int类型
            cleaned_hyperedges = []
            for edge in hyperedges:
                cleaned_edge = [int(x) for x in edge if isinstance(x, (int, torch.Tensor))]
                if len(cleaned_edge) > 1:
                    cleaned_hyperedges.append(cleaned_edge)
            
            if not cleaned_hyperedges:
                return None, None
            
            hypergraph = Hypergraph(num_nodes, cleaned_hyperedges)
            return hypergraph, node_features
            
        except Exception as e:
            print(f"Hypergraph construction error: {e}")
            return None, None

    def apply_hypergraph_conv_to_latents(self, latents, iteration):
        """🔥 对latents应用超图卷积（每隔一段时间重建超图）"""
        if not self.use_hypergraph or iteration < 50:
            return latents
        
        try:
            B, C, H, W = latents.shape
            original_device = latents.device
            original_dtype = latents.dtype
            
            # 🔥 每隔rebuild_interval次迭代重建超图
            if iteration - self.last_rebuild_iter >= self.rebuild_interval:
                hypergraph, node_features = self.build_spatial_hypergraph(latents)
                self.last_rebuild_iter = iteration
            else:
                # 复用上次的超图结构但用新的特征
                hypergraph, node_features = self.build_spatial_hypergraph(latents)
            
            if hypergraph is None or node_features is None:
                return latents
            
            # 🔥 确保node_features在正确的设备和数据类型上
            conv_dtype = next(self.latent_hypergraph_conv.parameters()).dtype
            node_features = node_features.to(device=self.device, dtype=conv_dtype)
            
            # 应用超图卷积（不更新权重）
            with torch.no_grad():
                enhanced_features = self.latent_hypergraph_conv(node_features, hypergraph)
                
                # 🔥 重塑时确保数据类型正确
                enhanced_features = enhanced_features.to(dtype=conv_dtype)
                enhanced_latents = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
            
            # 🔥 将输出转换回原始数据类型和设备
            enhanced_latents = enhanced_latents.to(device=original_device, dtype=original_dtype)
            
            # 渐进式混合
            blend_ratio = min(0.2, (iteration - 50) / 2000 * 0.2)
            output_latents = (1 - blend_ratio) * latents + blend_ratio * enhanced_latents
            
            return output_latents
            
        except Exception as e:
            print(f"Latent hypergraph convolution failed: {e}")
            import traceback
            traceback.print_exc()
            return latents

    def apply_hypergraph_conv_to_grad(self, grad, iteration):
        """🔥 对梯度应用超图卷积（每隔一段时间重建超图）"""
        if not self.use_hypergraph or iteration < 100:
            return grad
        
        try:
            B, C, H, W = grad.shape
            original_device = grad.device
            original_dtype = grad.dtype
            
            hypergraph, node_features = self.build_spatial_hypergraph(grad, sample_ratio=0.4)
            
            if hypergraph is None or node_features is None:
                return grad
            
            # 🔥 确保node_features在正确的设备和数据类型上
            conv_dtype = next(self.grad_hypergraph_conv.parameters()).dtype
            node_features = node_features.to(device=self.device, dtype=conv_dtype)
            
            # 应用超图卷积（不更新权重）
            with torch.no_grad():
                enhanced_features = self.grad_hypergraph_conv(node_features, hypergraph)
                
                # 🔥 重塑时确保数据类型正确
                enhanced_features = enhanced_features.to(dtype=conv_dtype)
                enhanced_grad = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
            
            # 🔥 将输出转换回原始数据类型和设备
            enhanced_grad = enhanced_grad.to(device=original_device, dtype=original_dtype)
            
            # 混合原始梯度和增强梯度
            grad_blend_ratio = min(0.15, (iteration - 100) / 2000 * 0.15)
            output_grad = (1 - grad_blend_ratio) * grad + grad_blend_ratio * enhanced_grad
            
            return output_grad
            
        except Exception as e:
            print(f"Gradient hypergraph convolution failed: {e}")
            import traceback
            traceback.print_exc()
            return grad

    def hypergraph_aware_timestep_sampling(self, iteration, warm_up_rate):
        """🔥 基于历史信息的智能时间步采样（无需超图卷积）"""
        # 基于历史时间步分析调整采样范围
        if len(self.timestep_history) > 3:
            recent_timesteps = self.timestep_history[-5:]
            avg_timestep = sum(recent_timesteps) / len(recent_timesteps)
            std_timestep = np.std(recent_timesteps)
            
            # 根据时间步稳定性和平均值调整范围
            if std_timestep > 200:  # 时间步跳跃太大
                t_min, t_max = 0.3, 0.7
            elif std_timestep < 50:  # 时间步很稳定
                if avg_timestep < 300:  # 低噪声区域
                    t_min, t_max = 0.02, 0.5
                elif avg_timestep > 700:  # 高噪声区域
                    t_min, t_max = 0.4, 0.9
                else:  # 中等噪声区域
                    t_min, t_max = 0.1, 0.8
            else:
                t_min, t_max = 0.2, 0.8
                
            # 🔥 基于梯度变化调整
            if len(self.grad_norm_history) > 2:
                recent_grad_changes = []
                for i in range(1, min(4, len(self.grad_norm_history))):
                    change = abs(self.grad_norm_history[-i] - self.grad_norm_history[-i-1])
                    recent_grad_changes.append(change)
                
                avg_grad_change = sum(recent_grad_changes) / len(recent_grad_changes)
                if avg_grad_change > 0.1:  # 梯度变化大，使用更稳定的时间步
                    t_min = max(t_min, 0.25)
                    t_max = min(t_max, 0.75)
        else:
            # 默认渐进策略
            if iteration < 1000:
                t_min, t_max = 0.3, 0.7
            elif iteration < 3000:
                t_min, t_max = 0.1, 0.8
            else:
                t_min, t_max = 0.02, 0.6
        
        # 🔥 修复：计算时间步索引
        min_step = int(self.num_train_timesteps * t_min)
        max_step = int(self.num_train_timesteps * t_max)
        
        # 🔥 修复：warm_up_rate应该减少最大步数，而不是增加
        warmup_reduction = int(self.num_train_timesteps * warm_up_rate * 0.1)  # 降低影响
        max_step = max_step - warmup_reduction
        
        # 🔥 关键修复：确保索引在有效范围内
        min_step = max(0, min_step)
        max_step = min(self.num_train_timesteps - 1, max_step)  # 确保不超出数组边界
        
        # 🔥 确保min_step < max_step
        if min_step >= max_step:
            min_step = 0
            max_step = self.num_train_timesteps - 1
        
        # 生成随机时间步索引
        ind_t = torch.randint(min_step, max_step + 1, (1,),
                            dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        
        # 🔥 最终安全检查
        ind_t = torch.clamp(ind_t, 0, self.num_train_timesteps - 1)
        
        return ind_t

    def update_hypergraph_history(self, timestep, grad_norm, loss, similarity, latents):
        """更新超图历史"""
        self.timestep_history.append(timestep.item())
        self.grad_norm_history.append(grad_norm)
        self.loss_history.append(loss)
        self.similarity_history.append(similarity)
        self.latent_history.append(latents.detach().clone())
        
        # 保持历史长度
        if len(self.timestep_history) > self.max_history:
            self.timestep_history.pop(0)
            self.grad_norm_history.pop(0)
            self.loss_history.pop(0)
            self.similarity_history.pop(0)
            self.latent_history.pop(0)

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
        
        # 🔥 应用超图卷积到latents（每隔一段时间重建超图）
        original_latents = latents.clone()
        latents = self.apply_hypergraph_conv_to_latents(latents, iteration)
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        # 🔥 使用超图感知的时间步采样
        ind_t = self.hypergraph_aware_timestep_sampling(iteration, warm_up_rate)
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
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        # 🔥 计算基础梯度
        base_grad = w(self.alphas[t]) * (pred_noise - target)
        
        # 🔥 应用超图卷积到梯度（每隔一段时间重建超图）
        enhanced_grad = self.apply_hypergraph_conv_to_grad(base_grad, iteration)
        
        # 🔥 计算统计信息
        final_grad_norm = torch.norm(enhanced_grad).item() / 1000.0
        
        grad = torch.nan_to_num(grad_scale * enhanced_grad)
        loss = SpecifyGradient.apply(latents, grad)
        
        # 🔥 计算相似性
        current_similarity = 0.0
        if len(self.grad_norm_history) > 0:
            prev_grad_norm = self.grad_norm_history[-1]
            current_similarity = 1.0 / (1.0 + abs(final_grad_norm - prev_grad_norm))
        
        # 🔥 更新历史
        self.update_hypergraph_history(ind_t, final_grad_norm, loss.item(), current_similarity, latents)

        if iteration % 10 == 0:
            hypergraph_status = "Enabled" if self.use_hypergraph else "Disabled"
            enhancement_factor = torch.norm(enhanced_grad - base_grad).item() if self.use_hypergraph else 0
            rebuild_info = f"Last_Rebuild={self.last_rebuild_iter}" if self.use_hypergraph else ""
            print(f"Iter {iteration}: Loss={loss.item():.6f}, "
                  f"Timestep={ind_t.item()}, "
                  f"Grad_Norm={final_grad_norm:.4f}, "
                  f"Enhancement={enhancement_factor:.4f}, "
                  f"Hypergraph={hypergraph_status}, {rebuild_info}")

        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                # 🔥 使用增强梯度进行可视化
                grad_abs = torch.abs(enhanced_grad.detach())
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

        if iteration % 10 == 0:
            print(f"Loss={loss.item():.6f}")
              
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