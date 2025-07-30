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
import cv2
from typing import Union

import torchvision.transforms as T

logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


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

def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    if verbose:
        print("--- [generate_advanced_mask_hsv] Segmenting white background object using HSV ---")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("Error: Input image must be a 3-channel BGR format.")
        return None

    height, width = original_image.shape[:2]

    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    white_bg_mask = cv2.inRange(hsv, (0, 0, 240), (180, 10, 255))
    fg_mask = cv2.bitwise_not(white_bg_mask)

    if verbose:
        white_ratio = np.mean(white_bg_mask > 0)
        print(f"[HSV] Background white ratio: {white_ratio*100:.2f}%")

    edges = cv2.Canny(fg_mask, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if verbose:
        print(f"Morphological processing completed, kernel size: {kernel_size}")

    final_mask_01 = (fg_mask / 255).astype(np.uint8)

    if verbose: print("Generated 0/1 binary mask (without connected component filtering).")
    return final_mask_01


def generate_advanced_mask(original_image: np.ndarray, verbose: bool = False) -> Union[np.ndarray, None]:
    if verbose:
        print("--- [generate_advanced_mask] Starting advanced segmentation ---")
    
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("Error: Input image must be a 3-channel BGR format.")
        return None

    height, width = original_image.shape[:2]

    b, g, r = cv2.split(original_image)
    channels = [b, g, r]
    channel_names = ['Blue', 'Green', 'Red']
    variances = [np.var(channel) for channel in channels]
    best_channel_idx = np.argmax(variances)
    best_channel = channels[best_channel_idx]
    if verbose: print(f"Selected {channel_names[best_channel_idx]} channel for segmentation (variance: {variances[best_channel_idx]:.2f})")

    median_val = np.median(best_channel)
    lower_thresh = max(0, int(0.5 * median_val))
    upper_thresh = min(255, int(1.5 * median_val))
    edges = cv2.Canny(best_channel, lower_thresh, upper_thresh)
    if verbose: print(f"Edge detection thresholds: {lower_thresh}-{upper_thresh}")

    blurred = cv2.GaussianBlur(best_channel, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(otsu_mask, dilated_edges)
    
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if verbose: print(f"Using a {kernel_size}x{kernel_size} elliptical kernel for morphological processing...")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    if num_labels <= 1:
        if verbose: print("No valid connected components found, segmentation might have failed.")
        return cleaned_mask
        
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(cleaned_mask)
    final_mask[labels == largest_component] = 255
    if verbose: print(f"Keeping the largest connected component with an area of {stats[largest_component, cv2.CC_STAT_AREA]} pixels")

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose: print("No contours found in the final mask.")
        return final_mask

    largest_contour = max(contours, key=cv2.contourArea)
    refined_mask = np.zeros_like(final_mask)
    cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    
    if verbose: print("Optimized and smoothed object boundaries...")
    
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
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key

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

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.rgb_latent_factors = torch.tensor([
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        
        self.use_hypergraph = DHG_AVAILABLE
        if self.use_hypergraph:
            try:
                hypergraph_dtype = torch.float32
                
                self.latent_hypergraph_conv = HGNNConv(
                    in_channels=4,
                    out_channels=4,
                    bias=True,
                    drop_rate=0.1
                ).to(device).to(hypergraph_dtype)
                
                self.grad_hypergraph_conv = HGNNConv(
                    in_channels=4,
                    out_channels=4,
                    bias=True,
                    drop_rate=0.1
                ).to(device).to(hypergraph_dtype)
                
                self.latent_hypergraph_conv.eval()
                self.grad_hypergraph_conv.eval()
                
                for param in self.latent_hypergraph_conv.parameters():
                    param.requires_grad = False
                for param in self.grad_hypergraph_conv.parameters():
                    param.requires_grad = False
                
                print(f'[DEBUG] Hypergraph conv dtype: {next(self.latent_hypergraph_conv.parameters()).dtype}')
                print(f'[DEBUG] Precision type: {self.precision_t}')
                
                print(f'[INFO] DHG hypergraph modules initialized (no training)!')
                
            except Exception as e:
                print(f'[WARNING] Failed to initialize hypergraph modules: {e}')
                self.use_hypergraph = False
        
        self.timestep_history = []
        self.grad_norm_history = []
        self.loss_history = []
        self.similarity_history = []
        self.latent_history = []
        self.max_history = 8
        
        self.rebuild_interval = 50
        self.last_rebuild_iter = 0

        print(f'[INFO] loaded stable diffusion!')
    
    def build_spatial_hypergraph(self, tensor, sample_ratio=0.3):
        if not self.use_hypergraph:
            return None, None
            
        try:
            B, C, H, W = tensor.shape
            
            if tensor.dtype == torch.float16:
                tensor_calc = tensor.float()
            else:
                tensor_calc = tensor
            
            node_features = tensor_calc.view(B, C, -1).transpose(1, 2).reshape(-1, C)
            num_nodes = node_features.shape[0]
            
            hyperedges = []
            step = max(1, int(1 / sample_ratio))
            
            for b in range(B):
                for h in range(0, H, step):
                    for w in range(0, W, step):
                        edge_nodes = []
                        for dh in [0, min(1, H-1-h)]:
                            for dw in [0, min(1, W-1-w)]:
                                nh, nw = h + dh, w + dw
                                node_idx = b * H * W + nh * W + nw
                                edge_nodes.append(int(node_idx))
                        
                        if len(edge_nodes) > 1:
                            hyperedges.append(edge_nodes)
            
            if num_nodes > 16 and num_nodes < 1024:
                sample_size = min(num_nodes // 8, 16)
                
                sampled_indices = torch.randperm(num_nodes)[:sample_size]
                
                with torch.no_grad():
                    sampled_features = node_features[sampled_indices].cpu()
                    
                    for i in range(min(sample_size, 8)):
                        for j in range(i+1, min(sample_size, 8)):
                            sim = F.cosine_similarity(
                                sampled_features[i:i+1], 
                                sampled_features[j:j+1], 
                                dim=1
                            ).item()
                            
                            if sim > 0.8:
                                edge_nodes = [sampled_indices[i].item(), sampled_indices[j].item()]
                                hyperedges.append(edge_nodes)
            
            if not hyperedges:
                for i in range(min(num_nodes, 10)):
                    hyperedges.append([i, (i+1) % num_nodes])
            
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
        if not self.use_hypergraph or iteration < 50:
            return latents
        
        try:
            B, C, H, W = latents.shape
            original_device = latents.device
            original_dtype = latents.dtype
            
            if iteration - self.last_rebuild_iter >= self.rebuild_interval:
                hypergraph, node_features = self.build_spatial_hypergraph(latents)
                self.last_rebuild_iter = iteration
            else:
                hypergraph, node_features = self.build_spatial_hypergraph(latents)
            
            if hypergraph is None or node_features is None:
                return latents
            
            conv_dtype = next(self.latent_hypergraph_conv.parameters()).dtype
            node_features = node_features.to(device=self.device, dtype=conv_dtype)
            
            with torch.no_grad():
                enhanced_features = self.latent_hypergraph_conv(node_features, hypergraph)
                
                enhanced_features = enhanced_features.to(dtype=conv_dtype)
                enhanced_latents = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
            
            enhanced_latents = enhanced_latents.to(device=original_device, dtype=original_dtype)
            
            blend_ratio = min(0.2, (iteration - 50) / 2000 * 0.2)
            output_latents = (1 - blend_ratio) * latents + blend_ratio * enhanced_latents
            
            return output_latents
            
        except Exception as e:
            print(f"Latent hypergraph convolution failed: {e}")
            import traceback
            traceback.print_exc()
            return latents

    def apply_hypergraph_conv_to_grad(self, grad, iteration):
        if not self.use_hypergraph or iteration < 100:
            return grad
        
        try:
            B, C, H, W = grad.shape
            original_device = grad.device
            original_dtype = grad.dtype
            
            hypergraph, node_features = self.build_spatial_hypergraph(grad, sample_ratio=0.4)
            
            if hypergraph is None or node_features is None:
                return grad
            
            conv_dtype = next(self.grad_hypergraph_conv.parameters()).dtype
            node_features = node_features.to(device=self.device, dtype=conv_dtype)
            
            with torch.no_grad():
                enhanced_features = self.grad_hypergraph_conv(node_features, hypergraph)
                
                enhanced_features = enhanced_features.to(dtype=conv_dtype)
                enhanced_grad = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
            
            enhanced_grad = enhanced_grad.to(device=original_device, dtype=original_dtype)
            
            grad_blend_ratio = min(0.15, (iteration - 100) / 2000 * 0.15)
            output_grad = (1 - grad_blend_ratio) * grad + grad_blend_ratio * enhanced_grad
            
            return output_grad
            
        except Exception as e:
            print(f"Gradient hypergraph convolution failed: {e}")
            import traceback
            traceback.print_exc()
            return grad

    def hypergraph_aware_timestep_sampling(self, iteration, warm_up_rate):
        if len(self.timestep_history) > 3:
            recent_timesteps = self.timestep_history[-5:]
            avg_timestep = sum(recent_timesteps) / len(recent_timesteps)
            std_timestep = np.std(recent_timesteps)
            
            if std_timestep > 200:
                t_min, t_max = 0.3, 0.7
            elif std_timestep < 50:
                if avg_timestep < 300:
                    t_min, t_max = 0.02, 0.5
                elif avg_timestep > 700:
                    t_min, t_max = 0.4, 0.9
                else:
                    t_min, t_max = 0.1, 0.8
            else:
                t_min, t_max = 0.2, 0.8
                
            if len(self.grad_norm_history) > 2:
                recent_grad_changes = []
                for i in range(1, min(4, len(self.grad_norm_history))):
                    change = abs(self.grad_norm_history[-i] - self.grad_norm_history[-i-1])
                    recent_grad_changes.append(change)
                
                avg_grad_change = sum(recent_grad_changes) / len(recent_grad_changes)
                if avg_grad_change > 0.1:
                    t_min = max(t_min, 0.25)
                    t_max = min(t_max, 0.75)
        else:
            if iteration < 1000:
                t_min, t_max = 0.3, 0.7
            elif iteration < 3000:
                t_min, t_max = 0.1, 0.8
            else:
                t_min, t_max = 0.02, 0.6
        
        min_step = int(self.num_train_timesteps * t_min)
        max_step = int(self.num_train_timesteps * t_max)
        
        warmup_reduction = int(self.num_train_timesteps * warm_up_rate * 0.1)
        max_step = max_step - warmup_reduction
        
        min_step = max(0, min_step)
        max_step = min(self.num_train_timesteps - 1, max_step)
        
        if min_step >= max_step:
            min_step = 0
            max_step = self.num_train_timesteps - 1
        
        ind_t = torch.randint(min_step, max_step + 1, (1,),
                            dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        
        ind_t = torch.clamp(ind_t, 0, self.num_train_timesteps - 1)
        
        return ind_t

    def update_hypergraph_history(self, timestep, grad_norm, loss, similarity, latents):
        self.timestep_history.append(timestep.item())
        self.grad_norm_history.append(grad_norm)
        self.loss_history.append(loss)
        self.similarity_history.append(similarity)
        self.latent_history.append(latents.detach().clone())
        
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
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond)
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

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
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

        ind_t = self.hypergraph_aware_timestep_sampling(iteration, warm_up_rate)
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
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

        base_grad = w(self.alphas[t]) * (pred_noise - target)
        
        enhanced_grad = self.apply_hypergraph_conv_to_grad(base_grad, iteration)
        
        final_grad_norm = torch.norm(enhanced_grad).item() / 1000.0
        
        grad = torch.nan_to_num(grad_scale * enhanced_grad)
        loss = SpecifyGradient.apply(latents, grad)
        
        current_similarity = 0.0
        if len(self.grad_norm_history) > 0:
            prev_grad_norm = self.grad_norm_history[-1]
            current_similarity = 1.0 / (1.0 + abs(final_grad_norm - prev_grad_norm))
        
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

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        
        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])

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
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
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
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence