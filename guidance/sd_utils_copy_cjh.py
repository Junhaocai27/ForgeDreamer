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
from collections import deque
from .sd_step import *
from dhg.nn import HGNNConv
from dhg.structure import Hypergraph

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
        # 🔥 新增：创建DINOv2特征可视化保存目录
        self.dinov2_output_dir = "/root/LucidDreamer/dinov2_output"
        os.makedirs(self.dinov2_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dinov2_output_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.dinov2_output_dir, "heatmaps"), exist_ok=True)
        os.makedirs(os.path.join(self.dinov2_output_dir, "pca_vis"), exist_ok=True)
        os.makedirs(os.path.join(self.dinov2_output_dir, "similarity_maps"), exist_ok=True)
        print(f"[INFO] DINOv2 visualization directory created at: {self.dinov2_output_dir}")

        # 可视化计数器
        self.vis_counter = 0

        # 添加用于特征提取和相似度计算的初始化
        self.pred_rgb_queue = deque(maxlen=4)  # 用于存储最近4个pred_rgb
        self.call_count = 0  # 计数器
        self.similarity_results = []  # 存储相似度结果

        # 添加用于特征提取和相似度计算的初始化
        self.pred_rgb_queue = deque(maxlen=4)  # 用于存储最近4个pred_rgb
        self.call_count = 0  # 计数器
        self.similarity_results = []  # 存储相似度结果
        
        # 🔥 新增：初始化DHG的HPConv超图卷积网络
        self.feature_dim = 512  # 压缩后的特征维度
        
        # 使用DHG的HGNNConv
        self.hypergraph_conv = HGNNConv(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            bias=True,
            drop_rate=0.1
        ).to(device)
        
        # 添加额外的MLP层用于特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        ).to(device)
        
        # 存储更新后的特征
        self.updated_features_history = []

        # 🔥 新增：特征损失相关配置
        self.use_dino_feature_loss = getattr(guidance_opt, 'use_dino_feature_loss', True)
        self.feature_loss_weight = getattr(guidance_opt, 'feature_loss_weight', 0.1)
        self.feature_loss_start_iter = getattr(guidance_opt, 'feature_loss_start_iter', 100)  # 从第100轮开始计算特征损失
        
        print(f"[INFO] DINO feature loss enabled: {self.use_dino_feature_loss}")
        print(f"[INFO] Feature loss weight: {self.feature_loss_weight}")
        print(f"[INFO] Feature loss start iteration: {self.feature_loss_start_iter}")

    # 初始化DINOv2模型（使用本地模型）
        try:
            # 使用torch.hub加载本地DINOv2模型
            torch.hub.set_dir('/root/LucidDreamer/dinov2')  # 设置hub目录
            self.dino_model = torch.hub.load('/root/LucidDreamer/dinov2', 'dinov2_vitb14', source='local').to(device)
            
            # 🔥 确保DINOv2模型使用float32精度
            self.dino_model = self.dino_model.float().eval()
            
            # 冻结模型参数
            for param in self.dino_model.parameters():
                param.requires_grad = False
            
            # DINOv2预处理参数
            self.patch_h = 40  # patch数量
            self.patch_w = 40
            self.feat_dim = 768  # DINOv2 ViT-B/14 输出特征维度是768
            
            # 创建DINOv2的预处理transform
            self.dino_transform = T.Compose([
                T.GaussianBlur(9, sigma=(0.1, 2.0)),
                T.Resize((self.patch_h * 14, self.patch_w * 14)),  # 14是patch_size
                T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            self.dino_processor = True  # 标记使用DINOv2
            print("[INFO] Successfully loaded DINOv2 ViT-B/14 model from local path")
            print(f"[INFO] DINOv2 model dtype: {next(self.dino_model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Warning: Failed to load DINOv2 model: {e}")
            print("Using simple CNN for feature extraction")
            self.dino_processor = None
            self.dino_model = self._create_simple_feature_extractor().to(device).float()
        
        print(f'[INFO] loaded stable diffusion!')

        # 🔥 新增：TensorBoard支持
        self.use_tensorboard = getattr(guidance_opt, 'use_tensorboard', True)
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                # 创建TensorBoard日志目录
                tb_log_dir = getattr(guidance_opt, 'tensorboard_log_dir', '/root/LucidDreamer/tensorboard_logs')
                os.makedirs(tb_log_dir, exist_ok=True)
                
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
                self.tb_log_interval = getattr(guidance_opt, 'tb_log_interval', 10)  # 每10次迭代记录一次
                
                print(f"[INFO] TensorBoard logging enabled, log dir: {tb_log_dir}")
                print(f"[INFO] TensorBoard log interval: {self.tb_log_interval}")
                
            except ImportError:
                print("[WARNING] TensorBoard not available, skipping tensorboard logging")
                self.use_tensorboard = False
                self.tb_writer = None
        else:
            self.tb_writer = None
        
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
    
    def extract_and_compute_similarity(self, pred_rgb):
        """
        🔥 修复：确保真正的滑动窗口实现
        """
        self.call_count += 1
        
        # 🔥 打印队列状态用于调试
        # print(f"[DEBUG] Call {self.call_count}: Queue before append: {len(self.pred_rgb_queue)}")
        
        # 将当前pred_rgb添加到队列中（deque会自动维护maxlen=4）
        self.pred_rgb_queue.append(pred_rgb.clone().detach())
        
        # print(f"[DEBUG] Call {self.call_count}: Queue after append: {len(self.pred_rgb_queue)}")
        
        # 🔥 关键修复：只要队列满了就处理，而不是等待特定条件
        if len(self.pred_rgb_queue) == 4:
            # 控制详细输出的频率
            verbose = self.call_count % 100 == 0
            
            if verbose:
                print(f"[INFO] 🔄 SLIDING WINDOW: Processing call {self.call_count}")
                print(f"[INFO] Window range: images {self.call_count-3} to {self.call_count}")
            
            # 🔥 确认这是真正的滑动窗口处理
            window_info = {
                'current_call': self.call_count,
                'window_start': self.call_count - 3,
                'window_end': self.call_count,
                'is_sliding': self.call_count > 4  # 第5次开始才是真正滑动
            }
            
            if verbose:
                print(f"[INFO] Window info: {window_info}")
            
            # 提取当前窗口中所有4个pred_rgb的特征
            features_list = []
            for i, rgb in enumerate(self.pred_rgb_queue):
                feature = self._extract_dino_features(rgb)
                features_list.append(feature)
                if verbose:
                    image_index = self.call_count - 3 + i
                    print(f"[INFO] Extracted feature from image {image_index} (queue position {i}), shape: {feature.shape}")
            
            # 计算相似度
            similarity_results = self._compute_pairwise_similarity_verbose(features_list, verbose)
            self.similarity_results.append(similarity_results)
            
            # 🔥 重要：基于当前窗口构建超图，而不是累积历史
            node_to_hyperedges, hyperedges = self.build_hypergraph_for_current_window(
                similarity_results, verbose
            )
            
            updated_features_list = features_list
            
            if hyperedges:
                if verbose:
                    print(f"[INFO] Applying hypergraph convolution for current window...")
                
                try:
                    updated_features_list = self.apply_hypergraph_convolution_verbose(
                        features_list, hyperedges, verbose
                    )
                    
                    # 存储当前窗口的更新信息
                    self.updated_features_history.append({
                        'iteration': self.call_count,
                        'window_range': (self.call_count-3, self.call_count),
                        'is_sliding_window': self.call_count > 4,
                        'original_features': [feat.clone() for feat in features_list],
                        'updated_features': [feat.clone() for feat in updated_features_list],
                        'similarity_results': similarity_results,
                        'hyperedges': hyperedges.copy()
                    })
                    
                    if verbose:
                        self.analyze_feature_updates(features_list, updated_features_list)
                    
                except Exception as e:
                    print(f"[ERROR] Hypergraph convolution failed: {e}")
            
            # 🔥 关键：不要清空队列！让deque自动管理滑动窗口
            # 下一次append时，deque会自动移除最老的元素
            
            if verbose:
                print(f"[INFO] ✅ Sliding window processing completed for call {self.call_count}")
                print(f"[INFO] Next call will process window: images {self.call_count-2} to {self.call_count+1}")
            
            return similarity_results, updated_features_list
        
        elif len(self.pred_rgb_queue) < 4:
            # 还在积累阶段
            remaining = 4 - len(self.pred_rgb_queue)
            if self.call_count % 50 == 0:
                print(f"[INFO] 📥 Accumulating: {len(self.pred_rgb_queue)}/4, need {remaining} more images")
            return None, None
        
        else:
            # 这种情况不应该发生
            print(f"[ERROR] ❌ Unexpected queue length: {len(self.pred_rgb_queue)}")
            return None, None

    def build_hypergraph_for_current_window(self, current_similarity_results, verbose=False):
        """
        🔥 新方法：只基于当前4个图像窗口构建超图
        不依赖历史累积，真正的滑动窗口方式
        """
        if not current_similarity_results:
            if verbose:
                print("[INFO] No similarity results for current window")
            return {}, {}
        
        hyperedges = {}
        node_to_hyperedges = {}
        
        if verbose:
            print(f"[INFO] 🔧 Building hypergraph for current window (call {self.call_count})")
        
        # 基于当前窗口的4个特征（索引0,1,2,3）构建超图
        for feature_key, feature_data in current_similarity_results.items():
            center_idx = int(feature_key.split('_')[1])  # 0, 1, 2, 或 3
            
            # 创建当前窗口的超边ID
            hyperedge_id = f"window_{self.call_count}_center_{center_idx}"
            
            # 获取相关特征索引（都在0-3范围内）
            top2_indices = feature_data['top2_feature_indices']
            
            # 超边包含中心特征和其top2相似特征
            hyperedge_nodes = [center_idx] + top2_indices
            hyperedge_nodes = sorted(list(set(hyperedge_nodes)))
            
            hyperedges[hyperedge_id] = hyperedge_nodes
            
            # 记录节点到超边的映射
            for node in hyperedge_nodes:
                if node not in node_to_hyperedges:
                    node_to_hyperedges[node] = []
                if hyperedge_id not in node_to_hyperedges[node]:
                    node_to_hyperedges[node].append(hyperedge_id)
        
        if verbose:
            print(f"[INFO] ✅ Built {len(hyperedges)} hyperedges for current window")
            for hyperedge_id, nodes in hyperedges.items():
                print(f"  - {hyperedge_id}: {nodes}")
        
        return node_to_hyperedges, hyperedges

    def get_sliding_window_status(self):
        """
        获取滑动窗口的详细状态信息
        """
        return {
            'current_call': self.call_count,
            'queue_length': len(self.pred_rgb_queue),
            'queue_capacity': self.pred_rgb_queue.maxlen,
            'total_windows_processed': len(self.similarity_results),
            'is_accumulating': len(self.pred_rgb_queue) < 4,
            'is_sliding': self.call_count > 4,
            'current_window_range': (
                max(1, self.call_count-3), self.call_count 
            ) if len(self.pred_rgb_queue) == 4 else None,
            'next_window_range': (
                max(2, self.call_count-2), self.call_count+1
            ) if len(self.pred_rgb_queue) == 4 else None
        }

    def print_sliding_window_debug(self):
        """
        打印滑动窗口的调试信息
        """
        status = self.get_sliding_window_status()
        
        print(f"\n{'='*60}")
        print(f"🔄 SLIDING WINDOW DEBUG INFO")
        print(f"{'='*60}")
        print(f"Current call: {status['current_call']}")
        print(f"Queue: {status['queue_length']}/{status['queue_capacity']}")
        print(f"Status: {'Accumulating' if status['is_accumulating'] else 'Sliding'}")
        print(f"Windows processed: {status['total_windows_processed']}")
        
        if status['current_window_range']:
            print(f"Current window: images {status['current_window_range'][0]} to {status['current_window_range'][1]}")
        
        if status['next_window_range']:
            print(f"Next window: images {status['next_window_range'][0]} to {status['next_window_range'][1]}")
        
        print(f"{'='*60}\n")

    def apply_noise_filtering(self, grad, iteration, opt):
        """
        对梯度进行自适应噪声滤波
        """
        # 🔥 计算梯度的噪声程度
        grad_magnitude = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        grad_std = torch.std(grad.view(grad.shape[0], -1), dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        # 噪声指标：标准差与均值的比值
        noise_indicator = grad_std / (grad_magnitude + 1e-8)
        
        # 🔥 自适应滤波强度
        remaining_iters = opt.iterations - opt.warmup_iter
        progress = max(0, (iteration - opt.warmup_iter) / remaining_iters)
        
        # 随着训练进行，增加滤波强度
        base_filter_strength = 0.1 + progress * 0.3  # 0.1 -> 0.4
        
        # 根据噪声程度调整滤波
        adaptive_filter = base_filter_strength * torch.clamp(noise_indicator, 0.1, 2.0)
        
        # 🔥 应用高斯滤波
        if adaptive_filter.mean() > 0.15:
            # 对高噪声梯度应用平滑
            kernel_size = 3
            sigma = adaptive_filter.mean().item()
            
            # 创建高斯滤波核
            gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(grad.device)
            
            # 对每个通道分别滤波
            filtered_grad = F.conv2d(
                grad.view(-1, 1, grad.shape[2], grad.shape[3]), 
                gaussian_kernel, 
                padding=kernel_size//2, 
                groups=1
            ).view(grad.shape)
            
            # 混合原始梯度和滤波梯度
            alpha = torch.clamp(adaptive_filter, 0, 0.8)
            grad = (1 - alpha) * grad + alpha * filtered_grad
        
        return grad

    def apply_detail_enhancement_hypergraph(self, grad, pred_rgb, iteration, opt):
        """
        使用超图卷积增强梯度中的细节信息
        """
        if iteration < opt.warmup_iter + 1000:
            return grad  # 前期不使用细节增强
        
        try:
            # 🔥 构建细节感知超图
            detail_hyperedges = self.build_detail_aware_hypergraph(pred_rgb, iteration, opt)
            
            if not detail_hyperedges:
                return grad
            
            # 🔥 将梯度重组为块特征
            grad_blocks = self._reshape_grad_to_blocks(grad)
            
            # 🔥 应用细节增强超图卷积
            enhanced_grad_blocks = self._apply_detail_hypergraph_conv(grad_blocks, detail_hyperedges)
            
            # 🔥 重组回原始梯度形状
            enhanced_grad = self._reshape_blocks_to_grad(enhanced_grad_blocks, grad.shape)
            
            # 🔥 混合原始梯度和增强梯度
            alpha = self._compute_detail_enhancement_weight(iteration, opt)
            final_grad = (1 - alpha) * grad + alpha * enhanced_grad
            
            if iteration % 100 == 0:
                detail_strength = torch.norm(enhanced_grad - grad).item()
                print(f"[DETAIL] Iter {iteration}: Detail enhancement strength: {detail_strength:.6f}, alpha: {alpha:.3f}")
            
            return final_grad
            
        except Exception as e:
            print(f"[ERROR] Detail enhancement failed: {e}")
            return grad

    def _reshape_grad_to_blocks(self, grad, block_size=32):
        """
        将梯度重组为块特征
        """
        B, C, H, W = grad.shape
        blocks = []
        block_positions = []
        
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                block = grad[:, :, i:i+block_size, j:j+block_size]
                # 填充到统一大小
                if block.shape[2] < block_size or block.shape[3] < block_size:
                    padded_block = torch.zeros(B, C, block_size, block_size, device=grad.device, dtype=grad.dtype)
                    padded_block[:, :, :block.shape[2], :block.shape[3]] = block
                    block = padded_block
                
                # 展平为特征向量
                block_feature = block.view(B, -1)  # [B, C*block_size*block_size]
                blocks.append(block_feature)
                block_positions.append((i, j))
        
        return blocks, block_positions

    def _apply_detail_hypergraph_conv(self, grad_blocks, detail_hyperedges):
        """
        对梯度块应用超图卷积
        """
        blocks, positions = grad_blocks
        
        if len(blocks) < 4:
            return grad_blocks  # 块数量太少，无法进行有效的超图卷积
        
        # 🔥 创建增强的超图卷积网络
        if not hasattr(self, 'detail_hypergraph_conv'):
            feature_dim = blocks[0].shape[1]  # C*block_size*block_size
            self.detail_hypergraph_conv = HGNNConv(
                in_channels=feature_dim,
                out_channels=feature_dim,
                bias=True,
                drop_rate=0.05
            ).to(self.device)
            
            # 细节增强器
            self.detail_enhancer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.Tanh()  # 使用Tanh来限制输出范围
            ).to(self.device)
        
        # 构建DHG超图
        num_blocks = len(blocks)
        edge_list = []
        
        for hyperedge_id, block_indices in detail_hyperedges.items():
            if len(block_indices) >= 2:
                # 将块坐标转换为块索引
                valid_indices = []
                for block_coord in block_indices:
                    try:
                        block_idx = positions.index(block_coord) if block_coord in positions else None
                        if block_idx is not None and 0 <= block_idx < num_blocks:
                            valid_indices.append(block_idx)
                    except:
                        continue
                
                if len(valid_indices) >= 2:
                    edge_list.append(valid_indices)
        
        # 如果没有有效超边，创建局部连接
        if not edge_list:
            # 创建4-连通的局部超边
            for i in range(min(num_blocks - 1, 20)):  # 限制处理的块数量
                neighbors = [i]
                for j in range(max(0, i-2), min(num_blocks, i+3)):
                    if j != i:
                        neighbors.append(j)
                if len(neighbors) >= 2:
                    edge_list.append(neighbors[:4])  # 每个超边最多4个节点
        
        try:
            # 构建DHG超图
            dhg_hypergraph = Hypergraph(num_v=num_blocks, e_list=edge_list).to(self.device)
            
            # 堆叠所有块特征
            stacked_blocks = torch.stack(blocks, dim=1)  # [B, num_blocks, feature_dim]
            enhanced_blocks = []
            
            # 对每个batch分别处理
            for b in range(stacked_blocks.shape[0]):
                batch_blocks = stacked_blocks[b]  # [num_blocks, feature_dim]
                
                # 应用超图卷积
                conv_output = self.detail_hypergraph_conv(batch_blocks, dhg_hypergraph)
                
                # 应用细节增强器
                enhancement = self.detail_enhancer(conv_output)
                
                # 残差连接 + 细节增强
                enhanced_output = batch_blocks + 0.1 * enhancement  # 小的增强系数
                enhanced_blocks.append(enhanced_output)
            
            # 重新组织结果
            enhanced_stacked = torch.stack(enhanced_blocks, dim=0)  # [B, num_blocks, feature_dim]
            enhanced_block_list = [enhanced_stacked[:, i] for i in range(num_blocks)]
            
            return enhanced_block_list, positions
            
        except Exception as e:
            print(f"[WARNING] Detail hypergraph convolution failed: {e}")
            return grad_blocks

    def _reshape_blocks_to_grad(self, enhanced_grad_blocks, original_shape, block_size=32):
        """
        将增强的块特征重组回梯度形状
        """
        enhanced_blocks, positions = enhanced_grad_blocks
        B, C, H, W = original_shape
        
        # 初始化输出梯度
        enhanced_grad = torch.zeros_like(torch.empty(original_shape, device=enhanced_blocks[0].device))
        
        for block_idx, (i, j) in enumerate(positions):
            if block_idx < len(enhanced_blocks):
                # 重塑块特征
                block_feature = enhanced_blocks[block_idx]  # [B, C*block_size*block_size]
                block = block_feature.view(B, C, block_size, block_size)
                
                # 计算实际区域大小
                actual_h = min(block_size, H - i)
                actual_w = min(block_size, W - j)
                
                # 放回对应位置
                enhanced_grad[:, :, i:i+actual_h, j:j+actual_w] = block[:, :, :actual_h, :actual_w]
        
        return enhanced_grad

    def _compute_detail_enhancement_weight(self, iteration, opt):
        """
        计算细节增强的权重
        """
        progress = max(0, (iteration - opt.warmup_iter) / (opt.iterations - opt.warmup_iter))
        
        if progress < 0.3:
            return 0.0  # 前期不使用
        elif progress < 0.7:
            # 中期逐渐增加
            local_progress = (progress - 0.3) / 0.4
            return 0.2 * local_progress  # 最大权重0.2
        else:
            # 后期保持稳定
            return 0.2

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯滤波核"""
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
        return kernel

    # def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
    #                     grad_scale=1,use_control_net=False,
    #                     save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
    #                     resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None, opt=None):

    #     # 🔥 修复：正确解包返回的元组
    #     result = self.extract_and_compute_similarity(pred_rgb)
        
    #     # 存储更新后的特征用于后续损失计算
    #     updated_features_for_loss = None
        
    #     if result is not None:
    #         # 检查返回值的类型
    #         if isinstance(result, tuple) and len(result) == 2:
    #             # 如果是元组，解包为两个变量
    #             similarity_results, updated_features = result
    #             # 🔥 保存更新后的特征用于损失计算
    #             updated_features_for_loss = updated_features
    #         else:
    #             # 如果是单个值（向后兼容）
    #             similarity_results = result
    #             updated_features = None
            
    #         if similarity_results is not None:
    #             # 打印当前这次计算的配对信息
    #             for feature_key, feature_data in similarity_results.items():
    #                 pairs = feature_data['top2_pairs']
                
    #             # 每隔200迭代保存分析
    #             if iteration % 200 == 0:
    #                 try:
    #                     self.save_hypergraph_to_file(f"hypergraph_iter_{iteration}.txt")
    #                 except Exception as e:
    #                     print(f"[WARNING] Failed to save hypergraph analysis: {e}")
                
    #             # 🔥 新增：如果有更新后的特征，打印更新信息
    #             if updated_features is not None:
    #                 if iteration % 100 == 0:  # 每100轮打印一次
    #                     print(f"[INFO] Features have been updated through hypergraph convolution")
        
    #     # flip aug
    #     pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

    #     # ...existing training code...
    #     B = pred_rgb.shape[0]
    #     K = text_embeddings.shape[0] - 1

    #     if as_latent:      
    #         latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
    #     else:
    #         latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
    #     weights = weights.reshape(-1)
    #     noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

    #     inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

    #     text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

    #     if guidance_opt.annealing_intervals:
    #         current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
    #     else:
    #         current_delta_t =  guidance_opt.delta_t

    #     # 热身期使用随机时间步，之后使用固定时间步
    #     # if iteration < opt.warmup_iter:
    #     #     # 热身阶段：使用随机时间步
    #     #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), 
    #     #                         dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
    #     # else:
    #     #     # 🔥 修复：更稳定的时间步调度
    #     #     remaining_iters = opt.iterations - opt.warmup_iter
    #     #     current_progress = (iteration - opt.warmup_iter) / remaining_iters
            
    #     #     # 🔥 改进：使用更保守的噪声水平，避免后期过低
    #     #     if current_progress < 0.5:
    #     #         # 前半段：从0.7下降到0.4
    #     #         noise_level = 0.7 - current_progress * 0.6  # 0.7 -> 0.4
    #     #     else:
    #     #         # 后半段：保持在0.3-0.4之间，加入随机性
    #     #         base_noise = 0.35
    #     #         noise_variation = 0.1 * (0.5 - abs(current_progress - 0.75))  # 在0.75附近变化最大
    #     #         noise_level = base_noise + noise_variation * (2 * torch.rand(1).item() - 1)
    #     #         noise_level = max(0.25, min(0.45, noise_level))  # 限制在合理范围
            
    #     #     ind_t_value = int(self.max_step * noise_level)
    #     #     ind_t = torch.tensor([ind_t_value], dtype=torch.long, device=self.device)[0]
            
    #     #     if iteration % 100 == 0:
    #     #         print(f"[ITER {iteration}] 稳定时间步: {ind_t.item()}, 噪声水平: {noise_level:.3f}")

    #     # ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

    #     # 🔥 智能时间步调度：结合随机性和确定性
    #     # if iteration < opt.warmup_iter:
    #     #     # 热身阶段：纯随机时间步
    #     #     ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), 
    #     #                         dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
    #     # else:
    #     #     # 🔥 主要训练阶段：智能混合策略
    #     #     remaining_iters = opt.iterations - opt.warmup_iter
    #     #     current_progress = (iteration - opt.warmup_iter) / remaining_iters
            
    #     #     # 计算基础噪声水平
    #     #     if current_progress < 0.3:
    #     #         # 前30%：高噪声，学习整体结构
    #     #         base_noise_level = 0.8 - current_progress * 0.3  # 0.8 -> 0.71
    #     #         noise_variance = 0.15
    #     #     elif current_progress < 0.7:
    #     #         # 中40%：中等噪声，平衡细节和稳定性
    #     #         local_progress = (current_progress - 0.3) / 0.4
    #     #         base_noise_level = 0.71 - local_progress * 0.31  # 0.71 -> 0.4
    #     #         noise_variance = 0.1
    #     #     else:
    #     #         # 后30%：低噪声，精细化细节
    #     #         local_progress = (current_progress - 0.7) / 0.3
    #     #         base_noise_level = 0.4 - local_progress * 0.15  # 0.4 -> 0.25
    #     #         noise_variance = 0.05
            
    #     #     # 🔥 关键：动态混合策略
    #     #     if iteration % 5 == 0:
    #     #         # 每5次迭代中有1次使用确定性时间步，提高细节
    #     #         ind_t_value = int(self.max_step * base_noise_level)
    #     #         strategy = "deterministic"
    #     #     else:
    #     #         # 其他4次使用带约束的随机时间步，保持多样性
    #     #         noise_level = base_noise_level + torch.randn(1).item() * noise_variance
    #     #         noise_level = max(0.2, min(0.85, noise_level))  # 限制范围
    #     #         ind_t_value = int(self.max_step * noise_level)
    #     #         strategy = "constrained_random"
            
    #     #     ind_t = torch.tensor([ind_t_value], dtype=torch.long, device=self.device)[0]
            
    #     #     if iteration % 100 == 0:
    #     #         print(f"[ITER {iteration}] Strategy: {strategy}, t: {ind_t.item()}, "
    #     #             f"base_level: {base_noise_level:.3f}, progress: {current_progress:.1%}")

    #     if iteration < opt.warmup_iter:
    #         # 热身阶段：纯随机时间步
    #         ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), 
    #                             dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
    #     else:
    #         # 🔥 关键修复：基于3000轮的最佳状态设计时间步
    #         remaining_iters = opt.iterations - opt.warmup_iter
    #         current_progress = (iteration - opt.warmup_iter) / remaining_iters
            
    #         # 计算相对于3000轮的进度（假设最佳状态在50%左右）
    #         optimal_progress = 0.5  # 3000轮大约在50%进度
            
    #         if current_progress < 0.3:
    #             # 前30%：逐渐接近最佳状态
    #             base_noise_level = 0.8 - current_progress * 0.3  # 0.8 -> 0.71
    #             noise_variance = 0.15
                
    #         elif current_progress < 0.7:
    #             # 中40%：保持3000轮左右的最佳状态
    #             # 🔥 关键：锁定在最佳金属质感的噪声水平
    #             base_noise_level = 0.5  # 固定在中等噪声水平
    #             noise_variance = 0.08   # 减少变化幅度
                
    #         else:
    #             # 后30%：🔥 不再降低噪声！保持金属质感
    #             # 避免噪声过低导致颜色偏移
    #             base_noise_level = 0.45 + 0.05 * torch.sin(torch.tensor(iteration * 0.02)).item()  # 在0.4-0.5之间周期变化
    #             noise_variance = 0.06
            
    #         # 🔥 调整策略：后期更多使用确定性时间步
    #         if current_progress < 0.5:
    #             deterministic_ratio = 0.2  # 20%确定性
    #         elif current_progress < 0.8:
    #             deterministic_ratio = 0.4  # 40%确定性，保持稳定
    #         else:
    #             deterministic_ratio = 0.6  # 60%确定性，避免随机性导致颜色偏移
            
    #         if torch.rand(1).item() < deterministic_ratio:
    #             # 确定性时间步
    #             ind_t_value = int(self.max_step * base_noise_level)
    #             strategy = "deterministic"
    #         else:
    #             # 约束随机时间步
    #             noise_level = base_noise_level + torch.randn(1).item() * noise_variance
    #             noise_level = max(0.35, min(0.65, noise_level))  # 🔥 限制在合理范围，避免过低
    #             ind_t_value = int(self.max_step * noise_level)
    #             strategy = "constrained_random"
            
    #         ind_t = torch.tensor([ind_t_value], dtype=torch.long, device=self.device)[0]
            
    #         if iteration % 100 == 0:
    #             print(f"[ITER {iteration}] Strategy: {strategy}, t: {ind_t.item()}, "
    #                 f"base_level: {base_noise_level:.3f}, progress: {current_progress:.1%}")

    #     ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

    #     t = self.timesteps[ind_t]
    #     prev_t = self.timesteps[ind_prev_t]

    #     with torch.no_grad():
    #         # step unroll via ddim inversion
    #         if not self.ism:
    #             prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
    #             latents_noisy = self.scheduler.add_noise(latents, noise, t)
    #             target = noise
    #         else:
    #             # Step 1: sample x_s with larger steps
    #             xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
    #             xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
    #             starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

    #             _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
    #                                                                             guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
    #             # Step 2: sample x_t
    #             _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
    #                                                                     guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

    #             pred_scores = pred_scores_xt + pred_scores_xs
    #             target = pred_scores[0][1]

    #     with torch.no_grad():
    #         latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
    #         tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
    #         if use_control_net:
    #             pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
    #             down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
    #                 latent_model_input,
    #                 tt,
    #                 encoder_hidden_states=text_embeddings,
    #                 controlnet_cond=pred_depth_input,
    #                 return_dict=False,
    #             )
    #             unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
    #                                 down_block_additional_residuals=down_block_res_samples,
    #                                 mid_block_additional_residual=mid_block_res_sample).sample
    #         else:
    #             unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

    #         unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
    #         noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
    #         delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
    #         delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds,\
    #                                                         weights,\
    #                                                         B)

    #     pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
    #     latents_noisy = latents_noisy.repeat(4, 1, 1, 1)  # 扩展 4 倍，使 batch 变成 16
    #     pred_noise = pred_noise.repeat(4, 1, 1, 1)
    #     pred_noise_2 = self.unet((latents_noisy + pred_noise).to(self.precision_t), t.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
    #     final_pred = 0.5 * (pred_noise + pred_noise_2)
    #     final_pred = final_pred[:target.shape[0]]
    #     w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
    #     grad = w(self.alphas[t]) * (final_pred - target)
    #     grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
    #     adaptive_scale = torch.clamp(1 / (grad_norm + 1e-6), min=0.1, max=10.0)
    #     grad = grad * adaptive_scale
        
    #     grad = torch.nan_to_num(grad_scale * grad)
    #     grad = self.apply_noise_filtering(grad, iteration, opt)  # 🔥 添加噪声滤波
        
    #     # grad = self.apply_detail_enhancement_hypergraph(grad, pred_rgb, iteration, opt)
        
    #     # 🔥 新增：计算梯度强度并动态调整
    #     grad_magnitude = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        
    #     # 🔥 关键：基于训练进度动态增强梯度
    #     progress = max(0, (iteration - opt.warmup_iter) / (opt.iterations - opt.warmup_iter))
        
    #     if progress < 0.5:
    #         # 前期：大幅增强梯度，促进快速收敛
    #         gradient_boost = 2.0 + (0.5 - progress) * 3.0  # 2.0 -> 5.0
    #     elif progress < 0.8:
    #         # 中期：适中增强，保持收敛动力
    #         gradient_boost = 1.5 + (0.8 - progress) * 1.67  # 1.5 -> 2.0
    #     else:
    #         # 后期：轻微增强，保持稳定
    #         gradient_boost = 1.2
        
    #     # 🔥 应用梯度增强
    #     enhanced_grad = grad * gradient_boost
        
    #     # 🔥 添加方向性引导：确保梯度指向SD渲染结果
    #     if iteration > opt.warmup_iter:
    #         direction_guidance = self.compute_direction_guidance(pred_rgb, enhanced_grad, iteration)
    #         enhanced_grad = enhanced_grad + 0.3 * direction_guidance
        
    #     if iteration % 100 == 0:
    #         print(f"[GRADIENT] Iter {iteration}: Original norm: {grad_magnitude.mean():.6f}, "
    #             f"Boost factor: {gradient_boost:.2f}, Enhanced norm: {torch.norm(enhanced_grad).item():.6f}")
        
    #     # 🔥 计算原始损失
    #     original_loss = SpecifyGradient.apply(latents, enhanced_grad)

    #     # 🔥 计算原始损失
    #     # original_loss = SpecifyGradient.apply(latents, grad)

    #     # 🔥 新增：计算DINO特征MSE损失
    #     dino_feature_loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # 初始化为张量
    #     feature_loss_weight = getattr(guidance_opt, 'feature_loss_weight', 0.1)  # 可配置的权重
        
    #     if iteration % guidance_opt.vis_interval == 0:
    #         noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
    #         lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
    #         save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
    #         with torch.no_grad():
    #             pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
    #             pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
    #             pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
    #             pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

    #             # 🔥 新增：计算pred_x0_pos的DINO特征与更新后pred_rgb特征的MSE损失
    #             dino_feature_loss = self.compute_dino_feature_cosine_loss(
    #                 pred_x0_pos, 
    #                 updated_features_for_loss, 
    #                 iteration
    #             )

    #             grad_abs = torch.abs(grad.detach())
    #             norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

    #             latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
    #             latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

    #             viz_images = torch.cat([pred_rgb, 
    #                                     pred_depth.repeat(1, 3, 1, 1), 
    #                                     pred_alpha.repeat(1, 3, 1, 1), 
    #                                     rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
    #                                     latents_rgb, latents_sp_rgb, 
    #                                     norm_grad,
    #                                     pred_x0_sp, pred_x0_pos],dim=0) 
    #             save_image(viz_images, save_path_iter)
    #     else:
    #         # 🔥 即使不在可视化步骤，也计算特征损失（如果有更新后的特征）
    #         if updated_features_for_loss is not None:
    #             # 需要先计算pred_x0_pos
    #             with torch.no_grad():
    #                 noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
    #                 pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
    #                 pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                    
    #                 dino_feature_loss = self.compute_dino_feature_cosine_loss(
    #                     pred_x0_pos, 
    #                     updated_features_for_loss, 
    #                     iteration
    #                 )

    #     # 🔥 组合最终损失
    #     total_loss = original_loss + feature_loss_weight * dino_feature_loss
    #     # total_loss = original_loss
        
    #     # 🔥 修复：正确转换张量为标量再格式化
    #     if iteration % 100 == 0 and dino_feature_loss.item() > 0:
    #         print(f"[INFO] Iter {iteration}: Original loss: {original_loss.item():.6f}, DINO feature loss: {dino_feature_loss.item():.6f}, Total loss: {total_loss.item():.6f}")

    #     return total_loss

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                    resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None, opt=None):

        # 🔥 修复：正确解包返回的元组
        result = self.extract_and_compute_similarity(pred_rgb)
        
        # 存储更新后的特征用于后续损失计算
        updated_features_for_loss = None
        
        if result is not None:
            # 检查返回值的类型
            if isinstance(result, tuple) and len(result) == 2:
                # 如果是元组，解包为两个变量
                similarity_results, updated_features = result
                # 🔥 保存更新后的特征用于损失计算
                updated_features_for_loss = updated_features
            else:
                # 如果是单个值（向后兼容）
                similarity_results = result
                updated_features = None
            
            if similarity_results is not None:
                # 打印当前这次计算的配对信息
                for feature_key, feature_data in similarity_results.items():
                    pairs = feature_data['top2_pairs']
                
                # 每隔200迭代保存分析
                if iteration % 200 == 0:
                    try:
                        self.save_hypergraph_to_file(f"hypergraph_iter_{iteration}.txt")
                    except Exception as e:
                        print(f"[WARNING] Failed to save hypergraph analysis: {e}")
                
                # 🔥 新增：如果有更新后的特征，打印更新信息
                if updated_features is not None:
                    if iteration % 100 == 0:  # 每100轮打印一次
                        print(f"[INFO] Features have been updated through hypergraph convolution")
        
        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        # ...existing training code...
        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        # 🔥 时间步策略
        if iteration < opt.warmup_iter:
            # 热身阶段：纯随机时间步
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), 
                                dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
            strategy = "warmup_random"
            base_noise_level = ind_t.item() / self.max_step
        else:
            # 🔥 关键修复：基于3000轮的最佳状态设计时间步
            remaining_iters = opt.iterations - opt.warmup_iter
            current_progress = (iteration - opt.warmup_iter) / remaining_iters
            
            # 计算相对于3000轮的进度（假设最佳状态在50%左右）
            optimal_progress = 0.5  # 3000轮大约在50%进度
            
            if current_progress < 0.3:
                # 前30%：逐渐接近最佳状态
                base_noise_level = 0.8 - current_progress * 0.3  # 0.8 -> 0.71
                noise_variance = 0.15
                
            elif current_progress < 0.7:
                # 中40%：保持3000轮左右的最佳状态
                # 🔥 关键：锁定在最佳金属质感的噪声水平
                base_noise_level = 0.5  # 固定在中等噪声水平
                noise_variance = 0.08   # 减少变化幅度
                
            else:
                # 后30%：🔥 不再降低噪声！保持金属质感
                # 避免噪声过低导致颜色偏移
                base_noise_level = 0.45 + 0.05 * torch.sin(torch.tensor(iteration * 0.02)).item()  # 在0.4-0.5之间周期变化
                noise_variance = 0.06
            
            # 🔥 调整策略：后期更多使用确定性时间步
            if current_progress < 0.5:
                deterministic_ratio = 0.2  # 20%确定性
            elif current_progress < 0.8:
                deterministic_ratio = 0.4  # 40%确定性，保持稳定
            else:
                deterministic_ratio = 0.6  # 60%确定性，避免随机性导致颜色偏移
            
            if torch.rand(1).item() < deterministic_ratio:
                # 确定性时间步
                ind_t_value = int(self.max_step * base_noise_level)
                strategy = "deterministic"
            else:
                # 约束随机时间步
                noise_level = base_noise_level + torch.randn(1).item() * noise_variance
                noise_level = max(0.35, min(0.65, noise_level))  # 🔥 限制在合理范围，避免过低
                ind_t_value = int(self.max_step * noise_level)
                strategy = "constrained_random"
            
            ind_t = torch.tensor([ind_t_value], dtype=torch.long, device=self.device)[0]

        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        # 🔥 记录时间步信息到TensorBoard
        self._log_timestep_to_tensorboard(ind_t.item(), base_noise_level, strategy, iteration)

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
        latents_noisy = latents_noisy.repeat(4, 1, 1, 1)  # 扩展 4 倍，使 batch 变成 16
        pred_noise = pred_noise.repeat(4, 1, 1, 1)
        pred_noise_2 = self.unet((latents_noisy + pred_noise).to(self.precision_t), t.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
        final_pred = 0.5 * (pred_noise + pred_noise_2)
        final_pred = final_pred[:target.shape[0]]
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (final_pred - target)
        
        # 🔥 记录原始梯度
        grad_norm_original = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        
        # adaptive_scale = torch.clamp(1 / (grad_norm_original + 1e-6), min=0.1, max=10.0)
        # grad = grad * adaptive_scale
        
        grad = torch.nan_to_num(grad_scale * grad)
        # grad = self.apply_noise_filtering(grad, iteration, opt)  # 🔥 添加噪声滤波
        
        # 🔥 计算梯度强度并动态调整
        grad_magnitude = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        
        # 🔥 关键：基于训练进度动态增强梯度
        progress = max(0, (iteration - opt.warmup_iter) / (opt.iterations - opt.warmup_iter))
        
        if progress < 0.5:
            # 前期：大幅增强梯度，促进快速收敛
            gradient_boost = 2.0 + (0.5 - progress) * 3.0  # 2.0 -> 5.0
        elif progress < 0.8:
            # 中期：适中增强，保持收敛动力
            gradient_boost = 1.5 + (0.8 - progress) * 1.67  # 1.5 -> 2.0
        else:
            # 后期：轻微增强，保持稳定
            gradient_boost = 1.2
        
        # 🔥 应用梯度增强
        enhanced_grad = grad * gradient_boost
        
        # 🔥 记录最终梯度
        final_grad_norm = torch.norm(enhanced_grad, p=2, dim=(1, 2, 3), keepdim=True)
        
        # 🔥 记录梯度信息到TensorBoard
        self._log_gradients_to_tensorboard(
            grad_norm_original.mean().item(),
            grad_magnitude.mean().item(), 
            final_grad_norm.mean().item(),
            gradient_boost,
            iteration
        )
        
        # 🔥 计算原始损失
        original_loss = SpecifyGradient.apply(latents, enhanced_grad)

        # 🔥 新增：计算DINO特征MSE损失
        dino_feature_loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # 初始化为张量
        feature_loss_weight = getattr(guidance_opt, 'feature_loss_weight', 0.1)  # 可配置的权重
        
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                # 🔥 新增：计算pred_x0_pos的DINO特征与更新后pred_rgb特征的MSE损失
                dino_feature_loss = self.compute_dino_feature_cosine_loss(
                    pred_x0_pos, 
                    updated_features_for_loss, 
                    iteration
                )

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
        else:
            # 🔥 即使不在可视化步骤，也计算特征损失（如果有更新后的特征）
            if updated_features_for_loss is not None:
                # 需要先计算pred_x0_pos
                with torch.no_grad():
                    noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
                    pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                    pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                    
                    dino_feature_loss = self.compute_dino_feature_cosine_loss(
                        pred_x0_pos, 
                        updated_features_for_loss, 
                        iteration
                    )

        # 🔥 组合最终损失
        # total_loss = original_loss + feature_loss_weight * dino_feature_loss
        total_loss = original_loss
        
        # 🔥 记录损失信息到TensorBoard
        self._log_losses_to_tensorboard(
            original_loss.item(),
            dino_feature_loss.item(),
            total_loss.item(),
            feature_loss_weight,
            iteration
        )
        
        # 🔥 修复：正确转换张量为标量再格式化
        if iteration % 100 == 0 and dino_feature_loss.item() > 0:
            print(f"[INFO] Iter {iteration}: Original loss: {original_loss.item():.6f}, DINO feature loss: {dino_feature_loss.item():.6f}, Total loss: {total_loss.item():.6f}")

        return total_loss

    def _log_timestep_to_tensorboard(self, timestep_value, noise_level, strategy, iteration):
        """记录时间步相关信息到TensorBoard"""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        if iteration % self.tb_log_interval != 0:
            return
        
        try:
            # 时间步相关指标
            self.tb_writer.add_scalar('Timestep/Value', timestep_value, iteration)
            self.tb_writer.add_scalar('Timestep/Noise_Level', noise_level, iteration)
            
            # 策略类型
            strategy_mapping = {'warmup_random': 0, 'deterministic': 1, 'constrained_random': 2}
            strategy_value = strategy_mapping.get(strategy, 3)
            self.tb_writer.add_scalar('Timestep/Strategy', strategy_value, iteration)
            
            self.tb_writer.flush()
            
        except Exception as e:
            print(f"[WARNING] Timestep TensorBoard logging failed: {e}")

    def _log_gradients_to_tensorboard(self, grad_norm_original, grad_norm_filtered, grad_norm_final, 
                                    gradient_boost, iteration):
        """记录梯度相关信息到TensorBoard"""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        if iteration % self.tb_log_interval != 0:
            return
        
        try:
            # 梯度范数
            self.tb_writer.add_scalar('Gradient/Norm_Original', grad_norm_original, iteration)
            self.tb_writer.add_scalar('Gradient/Norm_After_Filtering', grad_norm_filtered, iteration)
            self.tb_writer.add_scalar('Gradient/Norm_Final', grad_norm_final, iteration)
            
            # 梯度增强因子
            self.tb_writer.add_scalar('Gradient/Boost_Factor', gradient_boost, iteration)
            
            # 梯度变化比例
            if grad_norm_original > 0:
                grad_change_ratio = grad_norm_final / grad_norm_original
                self.tb_writer.add_scalar('Gradient/Change_Ratio', grad_change_ratio, iteration)
            
            self.tb_writer.flush()
            
        except Exception as e:
            print(f"[WARNING] Gradient TensorBoard logging failed: {e}")

    def _log_losses_to_tensorboard(self, original_loss, dino_feature_loss, total_loss, feature_loss_weight, iteration):
        """记录损失相关信息到TensorBoard"""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        if iteration % self.tb_log_interval != 0:
            return
        
        try:
            # 各项损失
            self.tb_writer.add_scalar('Loss/Original_Loss', original_loss, iteration)
            self.tb_writer.add_scalar('Loss/DINO_Feature_Loss', dino_feature_loss, iteration)
            self.tb_writer.add_scalar('Loss/Total_Loss', total_loss, iteration)
            
            # 损失权重
            self.tb_writer.add_scalar('Loss/Feature_Loss_Weight', feature_loss_weight, iteration)
            
            # 损失组成比例
            if total_loss > 0:
                original_ratio = original_loss / total_loss
                feature_ratio = (dino_feature_loss * feature_loss_weight) / total_loss
                
                self.tb_writer.add_scalar('Loss/Original_Loss_Ratio', original_ratio, iteration)
                self.tb_writer.add_scalar('Loss/Feature_Loss_Ratio', feature_ratio, iteration)
            
            self.tb_writer.flush()
            
        except Exception as e:
            print(f"[WARNING] Loss TensorBoard logging failed: {e}")

    def compute_dino_feature_cosine_loss(self, pred_x0_pos, updated_features_list, iteration):
        """
        使用F.cosine_embedding_loss计算pred_x0_pos的DINO特征与更新后pred_rgb特征的余弦损失
        
        Args:
            pred_x0_pos: [B, 3, H, W] 预测的x0正样本图像
            updated_features_list: 超图更新后的特征列表，包含4个张量，每个为[B, feature_dim]
            iteration: 当前迭代次数
            
        Returns:
            cosine_loss: 余弦嵌入损失值
        """
        if updated_features_list is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # 🔥 修复：确保pred_x0_pos是float32类型
            pred_x0_pos = pred_x0_pos.float()
            
            with torch.no_grad():
                # 1. 对pred_x0_pos提取DINO特征
                pred_x0_pos_features = self._extract_dino_features(pred_x0_pos)  # [B, feature_dim]
                
                if iteration % 100 == 0:
                    print(f"[INFO] Extracted DINO features from pred_x0_pos, shape: {pred_x0_pos_features.shape}")
            
            # 2. 将4个更新后的特征进行聚合（取平均）
            # updated_features_list: [feat0[B, 512], feat1[B, 512], feat2[B, 512], feat3[B, 512]]
            stacked_updated_features = torch.stack(updated_features_list, dim=0)  # [4, B, feature_dim]
            aggregated_updated_features = stacked_updated_features.mean(dim=0)  # [B, feature_dim] - 对4个特征取平均
            
            # 🔥 确保聚合后的特征也是float32类型
            aggregated_updated_features = aggregated_updated_features.float()
            
            # 3. 确保两个特征具有相同的维度
            if pred_x0_pos_features.shape[1] != aggregated_updated_features.shape[1]:
                # 如果维度不匹配，使用线性层进行维度调整
                if not hasattr(self, 'feature_dimension_adapter'):
                    self.feature_dimension_adapter = nn.Linear(
                        aggregated_updated_features.shape[1], 
                        pred_x0_pos_features.shape[1]
                    ).to(self.device).float()  # 确保adapter也是float32
                
                aggregated_updated_features = self.feature_dimension_adapter(aggregated_updated_features)
            
            # 🔥 4. 使用F.cosine_embedding_loss计算余弦损失
            # target=1表示我们希望两个特征向量尽可能相似
            target = torch.ones(pred_x0_pos_features.shape[0], device=self.device)  # [B]
            
            cosine_loss = F.cosine_embedding_loss(
                pred_x0_pos_features, 
                aggregated_updated_features.detach(),  # 不传播梯度到更新后的特征
                target,
                margin=0.0,  # 余弦相似度的边界，0表示期望完全相似
                reduction='mean'
            )
            
            if iteration % 100 == 0:
                # 计算实际的余弦相似度用于监控
                cosine_similarity = F.cosine_similarity(
                    pred_x0_pos_features, 
                    aggregated_updated_features.detach(), 
                    dim=1
                ).mean()
                
                print(f"[INFO] DINO feature cosine embedding loss: {cosine_loss.item():.6f}")
                print(f"[INFO] Average cosine similarity: {cosine_similarity.item():.6f}")
                print(f"[INFO] pred_x0_pos features norm: {torch.norm(pred_x0_pos_features).item():.6f}")
                print(f"[INFO] Updated features norm: {torch.norm(aggregated_updated_features).item():.6f}")
            
            return cosine_loss
            
        except Exception as e:
            print(f"[ERROR] Failed to compute DINO feature cosine embedding loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_alternative_dino_feature_mse_loss(self, pred_x0_pos, updated_features_list, iteration):
        """
        计算pred_x0_pos的DINO特征与更新后pred_rgb特征的MSE损失的另一种方法
        这种方法分别计算每个更新后特征与pred_x0_pos特征的MSE，然后取平均
        
        Args:
            pred_x0_pos: [B, 3, H, W] 预测的x0正样本图像
            updated_features_list: 超图更新后的特征列表，包含4个张量，每个为[B, feature_dim]
            iteration: 当前迭代次数
            
        Returns:
            average_mse_loss: 平均MSE损失值
        """
        if updated_features_list is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            with torch.no_grad():
                # 1. 对pred_x0_pos提取DINO特征
                pred_x0_pos_features = self._extract_dino_features(pred_x0_pos)  # [B, feature_dim]
            
            # 2. 分别计算每个更新后特征与pred_x0_pos特征的MSE
            mse_losses = []
            
            for i, updated_feature in enumerate(updated_features_list):
                # updated_feature: [B, feature_dim]
                
                # 确保维度匹配
                if pred_x0_pos_features.shape[1] != updated_feature.shape[1]:
                    if not hasattr(self, f'feature_adapter_{i}'):
                        setattr(self, f'feature_adapter_{i}', 
                            nn.Linear(updated_feature.shape[1], pred_x0_pos_features.shape[1]).to(self.device))
                    
                    adapter = getattr(self, f'feature_adapter_{i}')
                    adapted_feature = adapter(updated_feature)
                else:
                    adapted_feature = updated_feature
                
                # 计算MSE损失
                mse_loss_i = F.mse_loss(
                    pred_x0_pos_features, 
                    adapted_feature.detach(),  # 不传播梯度到更新后的特征
                    reduction='mean'
                )
                mse_losses.append(mse_loss_i)
                
                if iteration % 100 == 0:
                    print(f"[INFO] Feature {i} MSE loss: {mse_loss_i.item():.6f}")
            
            # 3. 取平均
            average_mse_loss = torch.stack(mse_losses).mean()
            
            if iteration % 100 == 0:
                print(f"[INFO] Average DINO feature MSE loss: {average_mse_loss.item():.6f}")
            
            return average_mse_loss
            
        except Exception as e:
            print(f"[ERROR] Failed to compute alternative DINO feature MSE loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
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
    
    def _create_simple_feature_extractor(self):
        """创建一个简单的CNN特征提取器作为DINO的替代"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # 固定输出尺寸
        )

    def _extract_dino_features(self, pred_rgb):
        """
        使用DINOv2提取特征
        
        Args:
            pred_rgb: [B, 3, H, W] RGB图像张量，值域[0,1]
            
        Returns:
            feature: [B, feature_dim] 一维特征向量
        """
        with torch.no_grad():
            # 🔥 修复：确保输入在正确的设备上并转换为float32
            pred_rgb = pred_rgb.to(self.device).float()  # 强制转换为float32
            batch_size = pred_rgb.shape[0]
            
            # 只在特定迭代时打印详细信息
            verbose = self.call_count % 100 == 0
            
            if self.dino_processor is not None:
                # 使用DINOv2模型
                features = []
                
                for i in range(batch_size):
                    # 获取单个图像 [3, H, W]
                    img_tensor = pred_rgb[i]  # [3, H, W]
                    
                    # 🔥 确保所有变换后的张量都是float32类型
                    # 应用DINOv2预处理
                    img_tensor = T.Resize((self.patch_h * 14, self.patch_w * 14))(img_tensor)
                    img_tensor = T.CenterCrop((self.patch_h * 14, self.patch_w * 14))(img_tensor)
                    img_tensor = T.GaussianBlur(9, sigma=(0.1, 2.0))(img_tensor)
                    img_tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img_tensor)
                    
                    # 🔥 确保tensor是float32类型并添加batch维度
                    img_tensor = img_tensor.float().unsqueeze(0).to(self.device)  # [1, 3, H, W]
                    
                    # 使用DINOv2提取特征
                    features_dict = self.dino_model.forward_features(img_tensor)
                    patch_features = features_dict['x_norm_patchtokens']  # [1, N, 768]
                    
                    # 对patch特征进行全局平均池化，得到图像级特征
                    img_feature = patch_features.mean(dim=1)  # [1, 768]
                    features.append(img_feature)
                
                # 拼接所有图像的特征
                features = torch.cat(features, dim=0)  # [B, 768]
                
            else:
                # 使用简单CNN特征提取器作为备选
                features = self.dino_model(pred_rgb)  # [B, 256, 8, 8]
                
                # 下采样和压缩
                if len(features.shape) > 2:
                    # 如果是矩阵形式，先进行下采样
                    features = F.adaptive_avg_pool2d(features, (4, 4))  # [B, C, 4, 4]
                    # 展平为一维
                    features = features.flatten(start_dim=1)  # [B, C*4*4]
            
            # 进一步压缩特征维度（可选）
            if features.shape[1] > 512:
                # 使用线性层压缩到512维
                if not hasattr(self, 'feature_compressor'):
                    self.feature_compressor = nn.Linear(features.shape[1], 512).to(self.device).float()
                    self.feature_compressor.eval()
                features = self.feature_compressor(features)
            
            # L2归一化，便于余弦相似度计算
            features = F.normalize(features, p=2, dim=1)
            
            if verbose:
                print(f"[INFO] Extracted features shape: {features.shape}")
            return features

    def _compute_pairwise_similarity(self, features_list):
        return self._compute_pairwise_similarity_verbose(features_list, verbose=False)

    def get_all_feature_pairs(self):
        """
        获取所有的特征配对信息，返回一个扁平的配对列表
        
        Returns:
            all_pairs: 所有特征配对的列表，格式为 [(i, j), (i, k), ...]
        """
        all_pairs = []
        for similarity_result in self.similarity_results:
            for feature_key, feature_data in similarity_result.items():
                pairs = feature_data['top2_pairs']
                all_pairs.extend(pairs)
        
        return all_pairs

    def get_unique_feature_pairs(self):
        """
        获取所有唯一的特征配对（去重），因为 (i,j) 和 (j,i) 是同一对
        
        Returns:
            unique_pairs: 去重后的特征配对列表
        """
        all_pairs = self.get_all_feature_pairs()
        unique_pairs = set()
        
        for pair in all_pairs:
            # 将配对标准化为 (小索引, 大索引) 来去重
            normalized_pair = tuple(sorted(pair))
            unique_pairs.add(normalized_pair)
        
        return list(unique_pairs)

    def print_similarity_summary(self):
        """
        打印相似度计算的总结信息
        """
        if not self.similarity_results:
            print("[INFO] No similarity results available.")
            return
        
        print(f"\n[INFO] === Similarity History Summary ===")
        print(f"Total computations: {len(self.similarity_results)}")
        
        for idx, similarity_result in enumerate(self.similarity_results):
            print(f"\nComputation {idx + 1}:")
            for feature_key, feature_data in similarity_result.items():
                pairs = feature_data['top2_pairs']
                print(f"  {feature_key}: {pairs[0]}, {pairs[1]}")
        
        # 显示所有唯一配对
        unique_pairs = self.get_unique_feature_pairs()
        print(f"\nAll unique feature pairs across all computations:")
        print(f"Unique pairs: {unique_pairs}")
        print("=" * 50)

    def get_similarity_history(self):
        """获取所有的相似度计算历史"""
        return self.similarity_results

    def clear_similarity_history(self):
        """清空相似度计算历史"""
        self.similarity_results.clear()
        self.pred_rgb_queue.clear()
        self.call_count = 0
    def build_hypergraph(self):
        """
        基于相似度配对结果构建超图
        同一个索引的所有相关特征构成一条超边
        
        Returns:
            hypergraph: 字典，key为节点索引，value为包含该节点的所有超边
            hyperedges: 字典，key为超边ID，value为该超边包含的所有节点
        """
        if not self.similarity_results:
            print("[INFO] No similarity results available for building hypergraph.")
            return {}, {}
        
        # 收集所有的配对信息
        all_pairs = self.get_all_feature_pairs()
        
        # 构建超图：每个特征索引作为中心，收集与它相关的所有特征
        hyperedges = {}
        node_to_hyperedges = {}
        
        # 对每个可能的特征索引创建超边
        for computation_idx, similarity_result in enumerate(self.similarity_results):
            for feature_key, feature_data in similarity_result.items():
                # 提取特征索引（例如从 'feature_0' 中提取 0）
                center_idx = int(feature_key.split('_')[1])
                
                # 创建超边ID
                hyperedge_id = f"computation_{computation_idx}_center_{center_idx}"
                
                # 获取与当前特征相关的所有特征索引
                top2_indices = feature_data['top2_feature_indices']
                
                # 超边包含中心特征和它的top2相似特征
                hyperedge_nodes = [center_idx] + top2_indices
                hyperedge_nodes = sorted(list(set(hyperedge_nodes)))  # 去重并排序
                
                hyperedges[hyperedge_id] = hyperedge_nodes
                
                # 为每个节点记录它所属的超边
                for node in hyperedge_nodes:
                    if node not in node_to_hyperedges:
                        node_to_hyperedges[node] = []
                    if hyperedge_id not in node_to_hyperedges[node]:
                        node_to_hyperedges[node].append(hyperedge_id)
        
        return node_to_hyperedges, hyperedges

    def print_hypergraph(self):
        """
        打印超图的详细信息
        """
        node_to_hyperedges, hyperedges = self.build_hypergraph()
        
        if not hyperedges:
            print("[INFO] No hypergraph to display.")
            return
        
        print("\n" + "="*60)
        print("🔥 HYPERGRAPH VISUALIZATION 🔥")
        print("="*60)
        
        # 1. 打印所有超边
        print("\n📊 HYPEREDGES (超边):")
        print("-" * 40)
        for hyperedge_id, nodes in hyperedges.items():
            print(f"{hyperedge_id}: {{{', '.join(map(str, nodes))}}}")
        
        # 2. 打印节点到超边的映射
        print("\n🔗 NODE TO HYPEREDGES (节点到超边的映射):")
        print("-" * 40)
        for node, edges in sorted(node_to_hyperedges.items()):
            print(f"Node {node}: {edges}")
        
        # 3. 统计信息
        print("\n📈 HYPERGRAPH STATISTICS:")
        print("-" * 40)
        print(f"Total nodes: {len(node_to_hyperedges)}")
        print(f"Total hyperedges: {len(hyperedges)}")
        
        # 计算节点度数（每个节点参与的超边数量）
        node_degrees = {node: len(edges) for node, edges in node_to_hyperedges.items()}
        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        print(f"Average node degree: {avg_degree:.2f}")
        
        # 计算超边大小（每条超边包含的节点数量）
        hyperedge_sizes = {edge_id: len(nodes) for edge_id, nodes in hyperedges.items()}
        avg_hyperedge_size = sum(hyperedge_sizes.values()) / len(hyperedge_sizes) if hyperedge_sizes else 0
        print(f"Average hyperedge size: {avg_hyperedge_size:.2f}")
        
        # 4. 超边大小分布
        print(f"\n📊 HYPEREDGE SIZE DISTRIBUTION:")
        print("-" * 40)
        size_counts = {}
        for size in hyperedge_sizes.values():
            size_counts[size] = size_counts.get(size, 0) + 1
        
        for size in sorted(size_counts.keys()):
            print(f"Size {size}: {size_counts[size]} hyperedges")
        
        # 5. 节点度数分布
        print(f"\n📊 NODE DEGREE DISTRIBUTION:")
        print("-" * 40)
        degree_counts = {}
        for degree in node_degrees.values():
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        for degree in sorted(degree_counts.keys()):
            print(f"Degree {degree}: {degree_counts[degree]} nodes")
        
        print("\n" + "="*60)

    def analyze_hypergraph_patterns(self):
        """
        分析超图中的模式和结构
        """
        node_to_hyperedges, hyperedges = self.build_hypergraph()
        
        if not hyperedges:
            print("[INFO] No hypergraph to analyze.")
            return
        
        print("\n🔍 HYPERGRAPH PATTERN ANALYSIS:")
        print("="*50)
        
        # 1. 找出最活跃的节点（参与最多超边的节点）
        node_degrees = {node: len(edges) for node, edges in node_to_hyperedges.items()}
        max_degree = max(node_degrees.values()) if node_degrees else 0
        most_active_nodes = [node for node, degree in node_degrees.items() if degree == max_degree]
        
        print(f"\n🌟 Most active nodes (degree {max_degree}): {most_active_nodes}")
        
        # 2. 找出最大的超边
        hyperedge_sizes = {edge_id: len(nodes) for edge_id, nodes in hyperedges.items()}
        max_size = max(hyperedge_sizes.values()) if hyperedge_sizes else 0
        largest_hyperedges = [edge_id for edge_id, size in hyperedge_sizes.items() if size == max_size]
        
        print(f"\n📏 Largest hyperedges (size {max_size}): {largest_hyperedges}")
        for edge_id in largest_hyperedges:
            print(f"  {edge_id}: {hyperedges[edge_id]}")
        
        # 3. 找出共同出现频率最高的节点对
        node_pair_counts = {}
        for edge_id, nodes in hyperedges.items():
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    pair = tuple(sorted([nodes[i], nodes[j]]))
                    node_pair_counts[pair] = node_pair_counts.get(pair, 0) + 1
        
        if node_pair_counts:
            max_cooccurrence = max(node_pair_counts.values())
            most_frequent_pairs = [pair for pair, count in node_pair_counts.items() if count == max_cooccurrence]
            
            print(f"\n🤝 Most frequent node pairs (co-occurrence {max_cooccurrence}):")
            for pair in most_frequent_pairs:
                print(f"  {pair}")
        
        print("="*50)

    def save_hypergraph_to_file(self, filename="hypergraph_analysis.txt"):
        """
        将超图分析结果保存到文件
        """
        import os
        
        node_to_hyperedges, hyperedges = self.build_hypergraph()
        
        if not hyperedges:
            print("[INFO] No hypergraph to save.")
            return
        
        save_path = os.path.join("/tmp", filename)  # 可以修改保存路径
        
        with open(save_path, 'w') as f:
            f.write("HYPERGRAPH ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            # 写入超边信息
            f.write("HYPEREDGES:\n")
            f.write("-" * 40 + "\n")
            for hyperedge_id, nodes in hyperedges.items():
                f.write(f"{hyperedge_id}: {{{', '.join(map(str, nodes))}}}\n")
            
            # 写入节点信息
            f.write("\nNODE TO HYPEREDGES:\n")
            f.write("-" * 40 + "\n")
            for node, edges in sorted(node_to_hyperedges.items()):
                f.write(f"Node {node}: {edges}\n")
            
            # 写入统计信息
            node_degrees = {node: len(edges) for node, edges in node_to_hyperedges.items()}
            hyperedge_sizes = {edge_id: len(nodes) for edge_id, nodes in hyperedges.items()}
            
            f.write(f"\nSTATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total nodes: {len(node_to_hyperedges)}\n")
            f.write(f"Total hyperedges: {len(hyperedges)}\n")
            f.write(f"Average node degree: {sum(node_degrees.values()) / len(node_degrees):.2f}\n")
            f.write(f"Average hyperedge size: {sum(hyperedge_sizes.values()) / len(hyperedge_sizes):.2f}\n")
        
        print(f"[INFO] Hypergraph analysis saved to: {save_path}")
    
    def visualize_and_save_features(self, pred_rgb, features_list, iteration):
        """
        可视化并保存DINOv2特征
        
        Args:
            pred_rgb: [B, 3, H, W] 原始RGB图像
            features_list: 包含4个特征张量的列表，每个张量形状为 [B, feature_dim]
            iteration: 当前迭代次数
        """
        self.vis_counter += 1
        print(f"[INFO] Visualizing and saving DINOv2 features (iteration {iteration}, count {self.vis_counter})")
        
        try:
            batch_size = pred_rgb.shape[0]
            print(f"[INFO] Processing {batch_size} batches for visualization")
            
            # 为每个batch中的图像创建可视化
            for b in range(batch_size):
                print(f"[INFO] Processing batch {b+1}/{batch_size}")
                
                # 1. 保存原始图像
                self._save_original_images(pred_rgb[b], b, iteration)
                
                # 2. 可视化特征向量
                self._visualize_feature_vectors(features_list, b, iteration)
                
                # 3. 可视化特征相似度热图
                self._visualize_similarity_heatmap(features_list, b, iteration)
                
                # 🔥 4. 强制执行patch特征可视化（生成heatmaps和pca_vis）
                if self.dino_processor:
                    print(f"[INFO] Starting patch feature visualization for batch {b}")
                    self._visualize_patch_features(pred_rgb[b], b, iteration)
                    print(f"[INFO] Completed patch feature visualization for batch {b}")
                else:
                    print("[WARNING] DINOv2 processor not available, skipping patch visualization")
            
            print(f"[INFO] All visualizations saved to {self.dinov2_output_dir}")
            
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _save_original_images(self, rgb_image, batch_idx, iteration):
        """保存原始RGB图像"""
        import os
        from PIL import Image
        import numpy as np
        
        try:
            # rgb_image: [3, H, W]
            img_np = rgb_image.detach().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            save_path = os.path.join(self.dinov2_output_dir, "features", 
                                    f"iter_{iteration}_batch_{batch_idx}_original.png")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图像
            Image.fromarray(img_np).save(save_path)
            print(f"[INFO] Saved original image to: {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save original image: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_feature_vectors(self, features_list, batch_idx, iteration):
        """可视化特征向量"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'DINOv2 Feature Vectors (Iter {iteration}, Batch {batch_idx})', fontsize=16)
            
            for i, feature in enumerate(features_list):
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                # 获取当前batch的特征向量
                feature_vec = feature[batch_idx].detach().cpu().numpy()
                
                # 绘制特征向量
                ax.plot(feature_vec, linewidth=0.8)
                ax.set_title(f'Feature {i} (dim={len(feature_vec)})')
                ax.set_xlabel('Feature Dimension')
                ax.set_ylabel('Feature Value')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = np.mean(feature_vec)
                std_val = np.std(feature_vec)
                ax.text(0.02, 0.98, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(self.dinov2_output_dir, "features", 
                                    f"iter_{iteration}_batch_{batch_idx}_feature_vectors.png")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Saved feature vectors to: {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to visualize feature vectors: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_patch_features(self, rgb_image, batch_idx, iteration):
        """可视化DINOv2的patch级特征"""
        try:
            with torch.no_grad():
                # 🔥 确保输入图像是float32类型
                rgb_image = rgb_image.float()
                
                # 预处理单个图像
                img_tensor = rgb_image.detach().clone()
                img_tensor = T.Resize((self.patch_h * 14, self.patch_w * 14))(img_tensor)
                img_tensor = T.CenterCrop((self.patch_h * 14, self.patch_w * 14))(img_tensor)
                img_tensor = T.GaussianBlur(9, sigma=(0.1, 2.0))(img_tensor)
                img_tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img_tensor)
                
                # 🔥 确保tensor是float32类型
                img_tensor = img_tensor.float().unsqueeze(0).to(self.device)
                
                print(f"[INFO] Extracting patch features for heatmap visualization...")
                
                # 提取patch特征
                features_dict = self.dino_model.forward_features(img_tensor)
                patch_features = features_dict['x_norm_patchtokens']  # [1, num_patches, 768]
                
                # 重塑为空间形式 [patch_h, patch_w, 768]
                patch_features = patch_features.squeeze(0).cpu().numpy()  # [num_patches, 768]
                patch_features_2d = patch_features.reshape(self.patch_h, self.patch_w, self.feat_dim)
                
                print(f"[INFO] Patch features shape: {patch_features_2d.shape}")
                
                # 1. PCA降维可视化
                self._create_pca_visualization(patch_features_2d, batch_idx, iteration)
                
                # 2. 特征热图可视化
                self._create_feature_heatmaps(patch_features_2d, rgb_image, batch_idx, iteration)
                
        except Exception as e:
            print(f"[ERROR] Patch feature visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_pca_visualization(self, patch_features_2d, batch_idx, iteration):
        """创建PCA降维可视化"""
        try:
            print(f"[INFO] Creating PCA visualization...")
            
            # 检查sklearn是否可用
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                print("[WARNING] sklearn not available, skipping PCA visualization")
                return
                
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            # 重塑特征用于PCA
            h, w, d = patch_features_2d.shape
            features_flat = patch_features_2d.reshape(-1, d)  # [h*w, d]
            
            print(f"[INFO] Running PCA on features shape: {features_flat.shape}")
            
            # PCA降维到3维用于RGB可视化
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features_flat)  # [h*w, 3]
            
            # 归一化到[0,1]范围
            features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
            
            # 重塑回空间形式
            pca_image = features_pca.reshape(h, w, 3)  # [h, w, 3]
            
            # 保存PCA可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(pca_image)
            ax1.set_title('PCA Visualization (RGB)')
            ax1.axis('off')
            
            # 显示PCA解释的方差比例
            explained_var = pca.explained_variance_ratio_
            ax2.bar(range(3), explained_var)
            ax2.set_title('PCA Explained Variance Ratio')
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Explained Variance Ratio')
            ax2.set_xticks(range(3))
            
            plt.tight_layout()
            save_path = os.path.join(self.dinov2_output_dir, "pca_vis", 
                                    f"iter_{iteration}_batch_{batch_idx}_pca.png")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Saved PCA visualization to: {save_path}")
            
        except Exception as e:
            print(f"[ERROR] PCA visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_feature_heatmaps(self, patch_features_2d, original_rgb, batch_idx, iteration):
        """创建特征热图"""
        try:
            print(f"[INFO] Creating feature heatmaps...")
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            h, w, d = patch_features_2d.shape
            print(f"[INFO] Creating heatmaps for patch features shape: {patch_features_2d.shape}")
            
            # 选择几个代表性的特征维度进行可视化
            selected_dims = [0, d//4, d//2, 3*d//4, d-1]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'DINOv2 Feature Heatmaps (Iter {iteration}, Batch {batch_idx})', fontsize=16)
            
            # 显示原始图像
            original_np = original_rgb.detach().cpu().permute(1, 2, 0).numpy()
            original_np = np.clip(original_np, 0, 1)
            axes[0, 0].imshow(original_np)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 显示选定维度的特征热图
            for i, dim in enumerate(selected_dims):
                row, col = (i + 1) // 3, (i + 1) % 3
                ax = axes[row, col]
                
                feature_map = patch_features_2d[:, :, dim]
                im = ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Feature Dim {dim}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(self.dinov2_output_dir, "heatmaps", 
                                    f"iter_{iteration}_batch_{batch_idx}_heatmaps.png")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Saved feature heatmaps to: {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Feature heatmap visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_similarity_heatmap(self, features_list, batch_idx, iteration):
        """可视化4个特征之间的相似度矩阵"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            import torch
            import numpy as np
            import os
            
            # 提取当前batch的特征
            batch_features = [feat[batch_idx].detach() for feat in features_list]
            
            # 计算相似度矩阵
            similarity_matrix = torch.zeros(4, 4)
            for i in range(4):
                for j in range(4):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sim = torch.cosine_similarity(batch_features[i], batch_features[j], dim=0)
                        similarity_matrix[i, j] = sim.item()
            
            # 可视化相似度矩阵
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(similarity_matrix.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            
            # 添加数值标注
            for i in range(4):
                for j in range(4):
                    text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                                ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title(f'Feature Similarity Matrix (Iter {iteration}, Batch {batch_idx})')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Index')
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            save_path = os.path.join(self.dinov2_output_dir, "similarity_maps", 
                                    f"iter_{iteration}_batch_{batch_idx}_similarity.png")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Saved similarity heatmap to: {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to visualize similarity heatmap: {e}")
            import traceback
            traceback.print_exc()

    def save_features_summary(self, iteration):
        """保存特征提取的总结信息"""
        if not self.similarity_results:
            return
        
        summary_file = os.path.join(self.dinov2_output_dir, f"features_summary_iter_{iteration}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"DINOv2 Feature Extraction Summary\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Total feature extractions: {len(self.similarity_results)}\n")
            f.write(f"Feature dimension: {getattr(self, 'feature_compressor', 'N/A')}\n")
            f.write(f"Patch size: {self.patch_h} x {self.patch_w}\n")
            f.write(f"DINOv2 feature dimension: {self.feat_dim}\n\n")
            
            # 写入最新的相似度结果
            if self.similarity_results:
                latest_results = self.similarity_results[-1]
                f.write("Latest Similarity Results:\n")
                f.write("-" * 30 + "\n")
                for feature_key, feature_data in latest_results.items():
                    pairs = feature_data['top2_pairs']
                    f.write(f"{feature_key}: {pairs[0]}, {pairs[1]}\n")
        
        print(f"[INFO] Features summary saved to: {summary_file}")
    def test_visualization_functions(self):
        """测试可视化功能是否正常工作"""
        print("[INFO] Testing visualization functions...")
        
        # 创建模拟数据
        test_rgb = torch.randn(2, 3, 256, 256).to(self.device)
        test_features = [torch.randn(2, 512).to(self.device) for _ in range(4)]
        
        try:
            # 测试基本可视化
            self._save_original_images(test_rgb[0], 0, 9999)
            self._visualize_feature_vectors(test_features, 0, 9999)
            self._visualize_similarity_heatmap(test_features, 0, 9999)
            
            # 测试patch特征可视化
            if self.dino_processor:
                self._visualize_patch_features(test_rgb[0], 0, 9999)
            
            print("[INFO] All visualization functions working correctly!")
            
        except Exception as e:
            print(f"[ERROR] Visualization test failed: {e}")
            import traceback
            traceback.print_exc()

    def apply_hypergraph_convolution(self, features_list, hyperedges):
        return self.apply_hypergraph_convolution_verbose(features_list, hyperedges, verbose=False)

    def _build_dhg_hypergraph(self, hyperedges, num_nodes=4):
        return self._build_dhg_hypergraph_verbose(hyperedges, num_nodes, verbose=False)
    
    def get_latest_updated_features(self):
        """
        获取最新的更新后特征
        
        Returns:
            updated_features: 最新一组更新后的特征列表，如果没有则返回None
        """
        if self.updated_features_history:
            return self.updated_features_history[-1]['updated_features']
        return None
    
    def clear_all_history(self):
        """
        清空所有历史记录
        """
        self.similarity_results.clear()
        self.updated_features_history.clear()
        self.pred_rgb_queue.clear()
        self.call_count = 0
        print("[INFO] All history cleared.")

    def analyze_feature_updates(self, original_features, updated_features):
        """
        分析超图卷积前后特征的变化
        
        Args:
            original_features: 原始特征列表
            updated_features: 更新后的特征列表
        """
        print(f"\n🔍 FEATURE UPDATE ANALYSIS:")
        print("="*50)
        
        total_change = 0
        total_cosine_sim = 0
        
        for i in range(len(original_features)):
            original = original_features[i]  # [B, feature_dim]
            updated = updated_features[i]   # [B, feature_dim]
            
            # 计算特征变化量（L2范数）
            feature_change = torch.norm(updated - original, p=2, dim=1).mean().item()
            total_change += feature_change
            
            # 计算余弦相似度（衡量方向变化）
            cosine_sim = F.cosine_similarity(original, updated, dim=1).mean().item()
            total_cosine_sim += cosine_sim
            
            # 计算特征范数
            original_norm = torch.norm(original, p=2, dim=1).mean().item()
            updated_norm = torch.norm(updated, p=2, dim=1).mean().item()
            
            print(f"Feature {i}:")
            print(f"  - L2 change: {feature_change:.6f}")
            print(f"  - Cosine similarity: {cosine_sim:.6f}")
            print(f"  - Original norm: {original_norm:.6f}")
            print(f"  - Updated norm: {updated_norm:.6f}")
            print(f"  - Norm ratio: {updated_norm/original_norm:.6f}")
        
        # 整体统计
        avg_change = total_change / len(original_features)
        avg_cosine_sim = total_cosine_sim / len(original_features)
        
        print(f"\nOverall Statistics:")
        print(f"  - Average L2 change: {avg_change:.6f}")
        print(f"  - Average cosine similarity: {avg_cosine_sim:.6f}")
        print("="*50)

    def save_hypergraph_analysis(self, iteration):
        """
        保存完整的超图分析结果，包括更新后的特征
        """
        import os
        import json
        
        save_path = os.path.join(self.dinov2_output_dir, f"hypergraph_analysis_iter_{iteration}.json")
        
        analysis_data = {
            'iteration': iteration,
            'total_computations': len(self.similarity_results),
            'total_feature_updates': len(self.updated_features_history),
            'hypergraph_config': {
                'feature_dim': self.feature_dim,
                'num_nodes': 4,
                'hypergraph_conv_type': 'HGNNConv'
            }
        }
        
        # 添加最新的分析结果
        if self.updated_features_history:
            latest_update = self.updated_features_history[-1]
            
            feature_changes = []
            for i in range(4):
                original = latest_update['original_features'][i]
                updated = latest_update['updated_features'][i]
                
                change_norm = torch.norm(updated - original, p=2, dim=1).mean().item()
                cosine_sim = F.cosine_similarity(original, updated, dim=1).mean().item()
                
                feature_changes.append({
                    'feature_index': i,
                    'l2_change': change_norm,
                    'cosine_similarity': cosine_sim,
                    'original_norm': torch.norm(original, p=2, dim=1).mean().item(),
                    'updated_norm': torch.norm(updated, p=2, dim=1).mean().item()
                })
            
            analysis_data['latest_update'] = {
                'iteration': latest_update['iteration'],
                'hyperedges': latest_update['hyperedges'],
                'feature_changes': feature_changes
            }
        
        # 保存历史统计
        if len(self.updated_features_history) > 1:
            history_stats = []
            for update in self.updated_features_history[-5:]:  # 保存最近5次的统计
                stats = []
                for i in range(4):
                    original = update['original_features'][i]
                    updated = update['updated_features'][i]
                    change = torch.norm(updated - original, p=2, dim=1).mean().item()
                    stats.append(change)
                history_stats.append({
                    'iteration': update['iteration'],
                    'avg_change': sum(stats) / len(stats)
                })
            analysis_data['recent_history'] = history_stats
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"[INFO] Hypergraph analysis saved to: {save_path}")

    def _compute_pairwise_similarity_verbose(self, features_list, verbose=True):
        """
        计算特征列表中每个特征与其他所有特征的余弦相似度，并为每个特征找到top2相似的特征
        支持控制输出详细程度
        
        Args:
            features_list: 包含4个特征张量的列表，每个张量形状为 [B, feature_dim]
            verbose: 是否打印详细信息
            
        Returns:
            similarity_results: 字典，包含每个特征的top2相似特征索引
        """
        num_features = len(features_list)
        similarity_results = {}
        
        if verbose:
            print(f"[INFO] Computing pairwise similarities for {num_features} features...")
        
        # 对每个特征分别计算它与其他所有特征的相似度
        for i in range(num_features):
            feature_i = features_list[i]  # [B, feature_dim] - 当前特征
            similarities_for_i = []       # 存储当前特征与其他特征的相似度
            indices_for_i = []           # 存储其他特征的索引
            
            if verbose:
                print(f"[INFO] Computing similarities for feature_{i}:")
            
            # 计算feature_i与其他所有特征的相似度
            for j in range(num_features):
                if i != j:  # 不与自己计算相似度
                    feature_j = features_list[j]  # [B, feature_dim] - 另一个特征
                    
                    # 计算余弦相似度：feature_i 与 feature_j
                    similarity = torch.cosine_similarity(feature_i, feature_j, dim=1)  # [B]
                    similarity_value = similarity.mean().item()  # 取批次平均得到标量
                    
                    similarities_for_i.append(similarity_value)
                    indices_for_i.append(j)
                    
                    if verbose:
                        print(f"  - feature_{i} vs feature_{j}: {similarity_value:.4f}")
            
            # 为当前特征找到top2最相似的特征
            similarities_tensor = torch.tensor(similarities_for_i)
            top2_values, top2_indices = torch.topk(similarities_tensor, k=min(2, len(similarities_for_i)))
            
            # 获取实际的特征索引
            top2_feature_indices = [indices_for_i[idx] for idx in top2_indices.tolist()]
            
            # 🔥 只存储特征配对信息，不存储相似度分数
            similarity_results[f'feature_{i}'] = {
                'top2_pairs': [(i, top2_feature_indices[0]), (i, top2_feature_indices[1])],  # 存储特征对
                'top2_feature_indices': top2_feature_indices  # 保留索引列表以便查看
            }
            
            if verbose:
                print(f"[INFO] Feature_{i} top2 pairs: ({i},{top2_feature_indices[0]}), ({i},{top2_feature_indices[1]})")
        
        # 打印完整的配对总结
        if verbose:
            print("\n[INFO] === Feature Pairs Summary ===")
            for i in range(num_features):
                result = similarity_results[f'feature_{i}']
                pairs = result['top2_pairs']
                print(f"Feature_{i} -> Pairs: {pairs[0]}, {pairs[1]}")
            print("=" * 50)
        
        return similarity_results
    def apply_hypergraph_convolution_verbose(self, features_list, hyperedges, verbose=True):
        """
        使用DHG库的HPConv对一组4个特征应用超图卷积
        支持控制输出详细程度
        
        Args:
            features_list: 包含4个特征张量的列表，每个张量形状为 [B, feature_dim]
            hyperedges: 当前组的超边信息字典
            verbose: 是否打印详细信息
            
        Returns:
            updated_features_list: 更新后的特征列表
        """
        if verbose:
            print(f"[INFO] Applying DHG hypergraph convolution...")
        
        # 1. 构建DHG超图对象
        dhg_hypergraph = self._build_dhg_hypergraph_verbose(hyperedges, num_nodes=4, verbose=verbose)
        
        if dhg_hypergraph is None:
            if verbose:
                print("[WARNING] Failed to build hypergraph, returning original features")
            return features_list
        
        # 2. 准备节点特征矩阵
        # 将4个特征合并为节点特征矩阵 [4, B, feature_dim]
        node_features = torch.stack(features_list, dim=0)  # [4, B, feature_dim]
        batch_size = node_features.shape[1]
        
        updated_features_batch = []
        
        # 3. 对每个batch分别处理
        for b in range(batch_size):
            try:
                # 提取当前batch的特征 [4, feature_dim]
                current_batch_features = node_features[:, b, :]  # [4, feature_dim]
                
                # 应用超图卷积
                with torch.no_grad():
                    # 使用DHG的超图卷积
                    conv_output = self.hypergraph_conv(current_batch_features, dhg_hypergraph)  # [4, feature_dim]
                    
                    # 添加残差连接
                    residual_output = conv_output + current_batch_features
                    
                    # 应用特征增强器
                    enhanced_output = self.feature_enhancer(residual_output)
                    
                    updated_features_batch.append(enhanced_output)
                    
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Hypergraph convolution failed for batch {b}: {e}")
                # 如果失败，使用原始特征
                updated_features_batch.append(current_batch_features)
        
        # 4. 重新组织为原始格式
        updated_features_tensor = torch.stack(updated_features_batch, dim=1)  # [4, B, feature_dim]
        updated_features_list = [updated_features_tensor[i] for i in range(4)]
        
        if verbose:
            print(f"[INFO] Hypergraph convolution completed. Updated {len(updated_features_list)} features.")
        
        return updated_features_list
    def _build_dhg_hypergraph_verbose(self, hyperedges, num_nodes=4, verbose=True):
        """
        基于超边信息构建DHG超图对象
        支持控制输出详细程度
        
        Args:
            hyperedges: 超边信息字典 {'hyperedge_id': [node_list], ...}
            num_nodes: 节点数量（固定为4）
            verbose: 是否打印详细信息
            
        Returns:
            dhg_hypergraph: DHG的Hypergraph对象
        """
        try:
            # 从超边字典中提取超边列表
            edge_list = []
            
            for hyperedge_id, nodes in hyperedges.items():
                # 确保节点索引在有效范围内 [0, 1, 2, 3]
                valid_nodes = [node for node in nodes if 0 <= node < num_nodes]
                if len(valid_nodes) >= 2:  # 至少需要2个节点才能构成有效超边
                    edge_list.append(valid_nodes)
            
            # 如果没有有效超边，创建默认超边
            if not edge_list:
                if verbose:
                    print("[WARNING] No valid hyperedges found, creating default hyperedges")
                # 创建默认的超边：每两个节点一条边
                edge_list = [
                    [0, 1], [0, 2], [0, 3],
                    [1, 2], [1, 3], [2, 3]
                ]
            
            if verbose:
                print(f"[INFO] Building DHG hypergraph with {len(edge_list)} hyperedges:")
                # 🔥 限制打印的超边数量，避免过多输出
                max_print = 10  # 最多打印前10条超边
                for i, edge in enumerate(edge_list[:max_print]):
                    print(f"  Hyperedge {i}: {edge}")
                
                if len(edge_list) > max_print:
                    print(f"  ... and {len(edge_list) - max_print} more hyperedges")
            
            # 使用DHG构建超图
            dhg_hypergraph = Hypergraph(num_v=num_nodes, e_list=edge_list)
            
            # 将超图移到正确的设备上
            dhg_hypergraph = dhg_hypergraph.to(self.device)
            
            return dhg_hypergraph
            
        except Exception as e:
            print(f"[ERROR] Failed to build DHG hypergraph: {e}")
            return None