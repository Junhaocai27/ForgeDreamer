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

from .sd_step import *

# 在现有导入后添加
# 🔥 添加DHG导入和TensorBoard
try:
    import dhg
    from dhg import Hypergraph
    from dhg.nn import HGNNConv
    DHG_AVAILABLE = True
    print("[INFO] DHG library available for block-based hypergraph noise modeling")
except ImportError:
    DHG_AVAILABLE = False
    print("[WARNING] DHG library not available, using standard noise comparison")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("[INFO] TensorBoard available for gradient logging")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not available, skipping gradient logging")

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
        

        print(f'[INFO] loaded stable diffusion!')

        # 在self.rgb_latent_factors定义之后，print('[INFO] loaded stable diffusion!')之前添加：

        # 🔥 添加基于块的超图噪声建模模块
        self.use_hypergraph_noise = DHG_AVAILABLE
        if self.use_hypergraph_noise:
            try:
                # 块大小配置
                self.block_size = 4  # 4x4块大小
                self.k_hop = 5       # k-hop邻居
                
                # 节点特征维度: 像素值(4) + 空间信息(2) = 6
                self.node_feature_dim = 4 + 2  # noise_channels + spatial_coords
                
                # 固定的超图卷积层（不训练）
                self.noise_feature_extractor = HGNNConv(
                    in_channels=self.node_feature_dim,
                    out_channels=16,    # 特征维度
                    bias=True,
                    drop_rate=0.1
                ).to(device).to(torch.float32)
                
                # 设置为eval模式，不训练
                self.noise_feature_extractor.eval()
                for param in self.noise_feature_extractor.parameters():
                    param.requires_grad = False
                    
                print(f'[INFO] Block-based hypergraph noise extractor initialized!')
                print(f'[INFO] Block size: {self.block_size}x{self.block_size}, K-hop: {self.k_hop}')
                
            except Exception as e:
                print(f'[WARNING] Failed to initialize hypergraph noise modules: {e}')
                self.use_hypergraph_noise = False

        # 🔥 TensorBoard日志
        self.use_tensorboard = TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            log_dir = f"./logs/block_hypergraph_experiment"
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir)
            print(f'[INFO] TensorBoard logging to {log_dir}')

        # 🔥 历史记录
        self.grad_history = []
        self.cosine_sim_history = []
        self.loss_history = []

    def split_noise_into_blocks(self, noise_tensor):
        """将噪声分割成块并添加空间信息"""
        try:
            B, C, H, W = noise_tensor.shape
            
            # 确保能够整除块大小
            pad_h = (self.block_size - H % self.block_size) % self.block_size
            pad_w = (self.block_size - W % self.block_size) % self.block_size
            
            if pad_h > 0 or pad_w > 0:
                noise_tensor = F.pad(noise_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                H, W = noise_tensor.shape[2], noise_tensor.shape[3]
            
            # 分割成块
            num_blocks_h = H // self.block_size
            num_blocks_w = W // self.block_size
            
            # 重塑为块: [B, C, num_blocks_h, block_size, num_blocks_w, block_size]
            blocks = noise_tensor.view(B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size)
            # 重新排列: [B, num_blocks_h, num_blocks_w, C, block_size, block_size]
            blocks = blocks.permute(0, 2, 4, 1, 3, 5)
            
            # 🔥 计算每个块的像素值特征 (均值)
            pixel_features = blocks.mean(dim=[4, 5])  # [B, num_blocks_h, num_blocks_w, C]
            
            # 🔥 添加空间信息 (归一化的块坐标)
            spatial_coords = []
            for b in range(B):
                coords = []
                for h_idx in range(num_blocks_h):
                    for w_idx in range(num_blocks_w):
                        # 归一化坐标
                        norm_h = h_idx / max(1, num_blocks_h - 1)
                        norm_w = w_idx / max(1, num_blocks_w - 1)
                        coords.append([norm_h, norm_w])
                spatial_coords.append(coords)
            
            spatial_coords = torch.tensor(spatial_coords, device=noise_tensor.device, dtype=torch.float32)
            # [B, num_blocks_h * num_blocks_w, 2]
            spatial_coords = spatial_coords.view(B, num_blocks_h * num_blocks_w, 2)
            
            # 重塑像素特征: [B, num_blocks_h * num_blocks_w, C]
            pixel_features = pixel_features.view(B, num_blocks_h * num_blocks_w, C)
            
            # 🔥 拼接像素值和空间信息: [B, num_blocks, C+2]
            node_features = torch.cat([pixel_features, spatial_coords], dim=2)
            
            return node_features, num_blocks_h, num_blocks_w
            
        except Exception as e:
            print(f"Block splitting error: {e}")
            return None, 0, 0

    def compute_k_hop_neighbors(self, num_blocks_h, num_blocks_w, k_hop=2):
        """计算k-hop邻居关系构建超边"""
        try:
            hyperedges = []
            total_blocks = num_blocks_h * num_blocks_w
            
            # 构建网格邻接关系
            def get_block_idx(h, w):
                return h * num_blocks_w + w
            
            def get_block_coords(idx):
                return idx // num_blocks_w, idx % num_blocks_w
            
            # 🔥 为每个块构建k-hop邻居超边
            for center_idx in range(total_blocks):
                center_h, center_w = get_block_coords(center_idx)
                
                # 收集k-hop邻居
                neighbors = set([center_idx])
                
                for hop in range(1, k_hop + 1):
                    new_neighbors = set()
                    for h_offset in range(-hop, hop + 1):
                        for w_offset in range(-hop, hop + 1):
                            if abs(h_offset) == hop or abs(w_offset) == hop:  # 只取边界
                                nh, nw = center_h + h_offset, center_w + w_offset
                                if 0 <= nh < num_blocks_h and 0 <= nw < num_blocks_w:
                                    neighbor_idx = get_block_idx(nh, nw)
                                    new_neighbors.add(neighbor_idx)
                    neighbors.update(new_neighbors)
                
                if len(neighbors) > 1:
                    hyperedges.append(list(neighbors))
            
            # 🔥 添加基于空间距离的超边
            for i in range(0, total_blocks, max(1, total_blocks // 8)):  # 采样一些中心点
                i_h, i_w = get_block_coords(i)
                distance_edge = []
                
                for j in range(total_blocks):
                    j_h, j_w = get_block_coords(j)
                    # 曼哈顿距离
                    distance = abs(i_h - j_h) + abs(i_w - j_w)
                    if distance <= 3:  # 距离阈值
                        distance_edge.append(j)
                
                if len(distance_edge) > 1:
                    hyperedges.append(distance_edge)
            
            return hyperedges
            
        except Exception as e:
            print(f"K-hop computation error: {e}")
            return []

    def build_block_hypergraph(self, noise_tensor):
        """构建基于块的超图结构"""
        if not self.use_hypergraph_noise:
            return None, None
            
        try:
            # 🔥 分割成块并添加空间信息
            node_features, num_blocks_h, num_blocks_w = self.split_noise_into_blocks(noise_tensor)
            
            if node_features is None:
                return None, None
            
            B = node_features.shape[0]
            total_blocks = node_features.shape[1]
            
            # 🔥 计算k-hop邻居超边
            hyperedges = self.compute_k_hop_neighbors(num_blocks_h, num_blocks_w, self.k_hop)
            
            if not hyperedges:
                # fallback: 创建简单的连续超边
                for i in range(0, total_blocks - 1, 2):
                    hyperedges.append([i, min(i + 1, total_blocks - 1)])
            
            # 🔥 处理批次维度 - 为每个batch创建独立的节点索引
            all_hyperedges = []
            for b in range(B):
                batch_offset = b * total_blocks
                for edge in hyperedges:
                    batch_edge = [node_idx + batch_offset for node_idx in edge]
                    all_hyperedges.append(batch_edge)
            
            # 重塑node_features为 [B * total_blocks, feature_dim]
            node_features = node_features.view(-1, node_features.shape[-1])
            total_nodes = node_features.shape[0]
            
            # 确保超边索引有效
            valid_hyperedges = []
            for edge in all_hyperedges:
                valid_edge = [idx for idx in edge if 0 <= idx < total_nodes]
                if len(valid_edge) > 1:
                    valid_hyperedges.append(valid_edge)
            
            if not valid_hyperedges:
                return None, None
            
            # 🔥 构建DHG超图
            hypergraph = Hypergraph(total_nodes, valid_hyperedges)
            
            return hypergraph, node_features
            
        except Exception as e:
            print(f"Block hypergraph construction error: {e}")
            return None, None

    def extract_block_noise_features(self, noise_tensor):
        """通过基于块的超图卷积提取噪声特征"""
        if not self.use_hypergraph_noise:
            return torch.zeros(1, 16, device=self.device, dtype=torch.float32)
        
        try:
            # 🔥 构建基于块的超图
            hypergraph, node_features = self.build_block_hypergraph(noise_tensor)
            
            if hypergraph is None or node_features is None:
                return torch.zeros(1, 16, device=self.device, dtype=torch.float32)
            
            # 确保数据类型和设备匹配
            node_features = node_features.to(device=self.device, dtype=torch.float32)
            
            # 🔥 超图卷积提取特征（无梯度计算）
            with torch.no_grad():
                # 通过HGNN层传播信息
                enhanced_features = self.noise_feature_extractor(node_features, hypergraph)
                
                # 🔥 全局池化：将所有块特征聚合为一个全局特征向量
                global_features = torch.mean(enhanced_features, dim=0, keepdim=True)  # [1, 16]
                
            return global_features
            
        except Exception as e:
            print(f"Block noise feature extraction failed: {e}")
            return torch.zeros(1, 16, device=self.device, dtype=torch.float32)

    def compute_block_cosine_loss(self, true_noise, pred_noise):
        """计算基于块的余弦相似性损失"""
        if not self.use_hypergraph_noise:
            return torch.tensor(0.0, device=self.device), 0.5
        
        try:
            # 🔥 提取两种噪声的块特征
            true_features = self.extract_block_noise_features(true_noise)    # [1, 16]
            pred_features = self.extract_block_noise_features(pred_noise)    # [1, 16]
            
            # 🔥 计算余弦相似性损失
            cosine_sim = F.cosine_similarity(true_features, pred_features, dim=1)  # [1]
            
            # 损失：1 - cosine_similarity (越相似损失越小)
            cosine_loss = 1.0 - cosine_sim.mean()
            
            return cosine_loss, cosine_sim.item()
            
        except Exception as e:
            print(f"Block cosine loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device), 0.5

    def log_block_metrics(self, iteration, grad_norm, cosine_loss, cosine_sim, main_loss):
        """记录块超图指标到TensorBoard"""
        if not self.use_tensorboard:
            return
        
        try:
            # 基础指标
            self.tb_writer.add_scalar('Training/Main_Loss', main_loss, iteration)
            self.tb_writer.add_scalar('Training/Gradient_Norm', grad_norm, iteration)
            
            if self.use_hypergraph_noise:
                self.tb_writer.add_scalar('Block_Hypergraph/Cosine_Loss', cosine_loss.item(), iteration)
                self.tb_writer.add_scalar('Block_Hypergraph/Cosine_Similarity', cosine_sim, iteration)
                
                # 相似性变化趋势
                if len(self.cosine_sim_history) > 1:
                    sim_change = abs(self.cosine_sim_history[-1] - self.cosine_sim_history[-2])
                    self.tb_writer.add_scalar('Block_Hypergraph/Similarity_Change', sim_change, iteration)
                
                # 梯度变化趋势
                if len(self.grad_history) > 1:
                    grad_change = abs(self.grad_history[-1] - self.grad_history[-2])
                    self.tb_writer.add_scalar('Training/Gradient_Change', grad_change, iteration)
            
            # 每100次迭代创建直方图
            if iteration % 100 == 0 and len(self.grad_history) > 10:
                self.tb_writer.add_histogram('History/Gradient_Norms', 
                                            torch.tensor(self.grad_history[-50:]), iteration)
                if len(self.cosine_sim_history) > 10:
                    self.tb_writer.add_histogram('History/Cosine_Similarities', 
                                                torch.tensor(self.cosine_sim_history[-50:]), iteration)
            
        except Exception as e:
            print(f"Block metrics logging failed: {e}")

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
        # w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        # grad = w(self.alphas[t]) * (pred_noise - target)

        # 🔥 计算基于块的超图余弦损失
        cosine_loss, cosine_sim = self.compute_block_cosine_loss(target, pred_noise)

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        # 🔥 基础梯度计算
        base_grad = w(self.alphas[t]) * (pred_noise - target)

        # 🔥 融合余弦损失到梯度中
        # 可以选择不同的融合策略
        fusion_weight = 0.5  # 调节余弦损失的影响权重
        enhanced_grad = base_grad + fusion_weight * cosine_loss * base_grad

        # 🔥 记录统计信息
        grad_norm = torch.norm(enhanced_grad).item()
        self.grad_history.append(grad_norm)
        self.cosine_sim_history.append(cosine_sim)

        # 保持历史长度
        max_history = 1000
        if len(self.grad_history) > max_history:
            self.grad_history = self.grad_history[-max_history:]
            self.cosine_sim_history = self.cosine_sim_history[-max_history:]

        grad = enhanced_grad
        
        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)

        # 更新loss历史
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]

        # TensorBoard记录（每10次迭代）
        if iteration % 10 == 0:
            self.log_block_metrics(iteration, grad_norm, cosine_loss, cosine_sim, loss.item())

        # 修改打印信息
        if iteration % 10 == 0:
            hypergraph_info = ""
            if self.use_hypergraph_noise:
                hypergraph_info = f", CosLoss={cosine_loss.item():.6f}, CosSim={cosine_sim:.4f}"
            print(f"Iter {iteration}: Loss={loss.item():.6f}, GradNorm={grad_norm:.4f}{hypergraph_info}")

        if iteration % 10 == 0:
            print(f"Loss={loss.item():.6f}")

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