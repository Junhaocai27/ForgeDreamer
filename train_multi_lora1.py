#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import imageio
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 禁用 tokenizers 并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T
from model.GCN import *
import torchvision.utils as vutils
from PIL import Image
from select_lora import select_lora_for_text

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    #for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    device = embeddings['front'].device

    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
            
        r = r.clone().detach().to(device).to(torch.float16)
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        r = r.clone().detach().to(device).to(torch.float16)
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def prepare_embeddings(guidance_opt, guidance):
    
    # 提取基础触发词
    import re
    base_text = guidance_opt.text  # 例如: "A DSLR photo of <nut>"
    trigger_pattern = r'<([^>]+)>'
    base_triggers = re.findall(trigger_pattern, base_text)
    
    if not base_triggers:
        print("警告: 文本中未找到<>包裹的触发词，将使用原始文本")
        base_trigger = None
    else:
        base_trigger = base_triggers[0]  # 使用第一个触发词
    
    # 创建前视图和上视图的文本
    if base_trigger:
        front_text = base_text.replace(f"<{base_trigger}>", f"<{base_trigger}_front>")
        up_text = base_text.replace(f"<{base_trigger}>", f"<{base_trigger}_up>")
    else:
        front_text = base_text + ", front view"
        up_text = base_text + ", up view"
    
    # 打印所有要使用的文本提示
    print("\n==== 视角特定的文本提示 ====")
    print(f"原始文本: {base_text}")
    print(f"前视文本: {front_text}")
    print(f"上视文本: {up_text}")
    print("============================\n")
    
    embeddings_front = {}
    embeddings_up = {}
    # text embeddings (stable-diffusion) and (IF)
    embeddings_front['default'] = guidance['front'].get_text_embeds([front_text])
    embeddings_front['uncond'] = guidance['front'].get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings_front[d] = guidance['front'].get_text_embeds([f"{front_text}, {d} view"])
    embeddings_front['inverse_text'] = guidance['front'].get_text_embeds(guidance_opt.inverse_text)
    
    embeddings_up['default'] = guidance['up'].get_text_embeds([up_text])
    embeddings_up['uncond'] = guidance['up'].get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings_up[d] = guidance['up'].get_text_embeds([f"{up_text}, {d} view"])
    embeddings_up['inverse_text'] = guidance['up'].get_text_embeds(guidance_opt.inverse_text)
    return embeddings_front, embeddings_up

def create_text_graph(text, view=None, model_name="/home/s414e2/CJH/Text-to-3D/LucidDreamer/bert-base-uncased/bert-base-uncased"):
    from transformers import AutoTokenizer, AutoModel
    import torch
    from torch_geometric.data import Data
    import spacy

    # 加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 加载 spaCy 模型
    nlp = spacy.load("en_core_web_sm")
    
    # 获取单词及其嵌入
    doc = nlp(f"{text}, {view} view")
    nodes = [token.text for token in doc]
    edges = [(token.head.i, token.i) for token in doc if token.head.i != token.i]

    # 生成 BERT 嵌入
    inputs = tokenizer(nodes, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
    outputs = model(**inputs)
    
    # 获取每个单词的嵌入
    last_hidden_state = outputs.last_hidden_state.squeeze(0)  # 移除 batch 维度
    word_ids = inputs.word_ids()  # 获取每个 token 对应的原始单词索引

    # 平均池化每个单词的子词嵌入
    node_features = []
    for i in range(len(nodes)):
        token_indices = [j for j, word_id in enumerate(word_ids) if word_id == i]
        word_embedding = last_hidden_state[token_indices].mean(dim=0)
        node_features.append(word_embedding)
    
    node_features = torch.stack(node_features)

    # 转化为 PyTorch Geometric 的图结构
    edge_index = torch.tensor(edges).t().contiguous()

    # # 检查并修复 edge_index
    # num_nodes = node_features.size(0)
    # valid_mask = (edge_index >= 0) & (edge_index < num_nodes)
    # valid_mask = valid_mask.all(dim=0)
    # edge_index = edge_index[:, valid_mask]

    graph = Data(x=node_features, edge_index=edge_index)

    return graph

def new_prepare_embeddings(guidance_opt, guidance):
    """
    guidance: 包含 `get_text_embeds` 方法的对象，用于生成兼容的嵌入
    num_iterations: GCN迭代的步数，用于丰富图嵌入语义信息
    """
    device = guidance['front'].device  # 使用front视角模型的设备
    
    def generate_graph_embed(graph, model):
        # 初始化嵌入
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        
        # 检查 edge_index 的有效性
        # num_nodes = x.size(0)
        # if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
        #     raise ValueError("edge_index contains invalid indices")
        
        # 打印调试信息
        # print(f"x shape: {x.shape}")
        # print(f"edge_index shape: {edge_index.shape}")
        # print(f"edge_index max: {edge_index.max().item()}, min: {edge_index.min().item()}")
        
        # 多步迭代优化
        x = model(x, edge_index)
        return x.mean(dim=0)  # 返回图全局嵌入

    embeddings = {}

    # 为每种文本类型创建独立的 GCN 模型
    gcn_models = {
        'text': GCN(input_dim=create_text_graph(guidance_opt.text).x.shape[1], 
                    hidden_dim=128, 
                    output_dim=1024).to(device),
        'negative': GCN(input_dim=create_text_graph(guidance_opt.negative).x.shape[1], 
                        hidden_dim=128, 
                        output_dim=1024).to(device),
        'inverse_text': GCN(input_dim=create_text_graph(guidance_opt.inverse_text).x.shape[1], 
                            hidden_dim=128, 
                            output_dim=1024).to(device),
    }

    # 1. Default 图嵌入
    graph = create_text_graph(guidance_opt.text)
    graph_embed = generate_graph_embed(graph, gcn_models['text'])
    embeddings['default'] = graph_embed.unsqueeze(0).repeat(77, 1).unsqueeze(0)  # (1, 77, 1024)

    # 2. Negative 图嵌入
    graph = create_text_graph(guidance_opt.negative)
    graph_embed = generate_graph_embed(graph, gcn_models['negative'])
    embeddings['uncond'] = graph_embed.unsqueeze(0).repeat(77, 1).unsqueeze(0)  # (1, 77, 1024)

    # 3. 生成多视角嵌入 (复用 text 的 GCN)
    for view in ['front', 'side', 'back']:
        graph = create_text_graph(guidance_opt.text, view)
        graph_embed = generate_graph_embed(graph, gcn_models['text'])
        embeddings[view] = graph_embed.unsqueeze(0).repeat(77, 1).unsqueeze(0)  # (1, 77, 1024)

    # 4. Inverse text 图嵌入
    graph = create_text_graph(guidance_opt.inverse_text)
    graph_embed = generate_graph_embed(graph, gcn_models['inverse_text'])
    embeddings['inverse_text'] = graph_embed.unsqueeze(0).repeat(77, 1).unsqueeze(0)  # (1, 77, 1024)

    return embeddings

def guidance_setup(guidance_opt):
    
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils import StableDiffusion
        
        # 分别创建两个视角的Guidance模型
        print("\n==== 创建多视角Guidance模型 ====")
        
        # 查找前视角LoRA文件
        front_lora_path = select_lora_for_text(guidance_opt.text, "front")
        if not front_lora_path:
            print(f"警告: 未找到前视角LoRA文件，将使用空LoRA路径")
            front_lora_path = ""
        else:
            print(f"前视角LoRA: {front_lora_path}")
        
        # 查找上视角LoRA文件
        up_lora_path = select_lora_for_text(guidance_opt.text, "up")
        if not up_lora_path:
            print(f"警告: 未找到上视角LoRA文件，将使用空LoRA路径")
            up_lora_path = ""
        else:
            print(f"上视角LoRA: {up_lora_path}")
        
        # 创建前视角Guidance模型
        print("正在初始化前视角Guidance模型...")
        guidance_front = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                       guidance_opt.t_range, guidance_opt.max_t_range, 
                                       num_train_timesteps=guidance_opt.num_train_timesteps, 
                                       ddim_inv=guidance_opt.ddim_inv,
                                       textual_inversion_path=guidance_opt.textual_inversion_path,
                                       LoRA_path=front_lora_path,
                                       guidance_opt=guidance_opt)
        
        # 创建上视角Guidance模型
        print("正在初始化上视角Guidance模型...")
        guidance_up = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                     guidance_opt.t_range, guidance_opt.max_t_range, 
                                     num_train_timesteps=guidance_opt.num_train_timesteps, 
                                     ddim_inv=guidance_opt.ddim_inv,
                                     textual_inversion_path=guidance_opt.textual_inversion_path,
                                     LoRA_path=up_lora_path,
                                     guidance_opt=guidance_opt)
        
        # 将两个模型包装在字典中
        guidance = {
            'front': guidance_front,
            'up': guidance_up
        }
        
        print("多视角Guidance模型初始化完成！")
        print("============================\n")
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
    # 对字典中的每个模型分别禁用梯度计算
        for view_type, model in guidance.items():
            if model is not None:
                for p in model.parameters():
                    p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embeddings1 = new_prepare_embeddings(guidance_opt, guidance)
    embeddings = new_prepare_embeddings(guidance_opt, guidance)
    embeddings = {k: v * 0.1 for k, v in embeddings.items()} # 0.1倍图嵌入
    embeddings_front, embeddings_up = prepare_embeddings(guidance_opt, guidance)

    # gcn_embed, keys1 = dict_to_tensor(embeddings1, device)
    # clip_embed, keys2 = dict_to_tensor(embeddings2, device)

    # # 创建 EmbeddingFusionMLP 实例
    # mlp_fusion = EmbeddingFusionMLP(input_dim=2048, hidden_dim=2048, output_dim=1024).to(device)

    # # 调用 EmbeddingFusionMLP 的 forward 方法
    # fused_embed_tensor = mlp_fusion.forward(gcn_embed, clip_embed)

    # att_fusion = AttentionFusion(embed_dim=1024).to(device)
    # fused_embed_tensor = att_fusion.forward(gcn_embed, clip_embed)

    # att_fusion_new = EmbeddingFusion(embedding_dim=1024).to(device)
    # fused_embed_tensor = att_fusion_new.forward(clip_embed, gcn_embed)

    # # 将结果转换回 dict
    # embeddings = tensor_to_dict(fused_embed_tensor, keys1)


    # 直接叠加图嵌入生成的embeddings和CLIP生成的embeddings
    # embeddings = {}
    # for key in embeddings1:
    #     embeddings[key] = (embeddings1[key] + embeddings2[key])

    # embeddings.update(embeddings2)
    # 替代方案 - 创建新字典合并嵌入
    embeddings_front_combined = {}
    embeddings_up_combined = {}

    # 复制所有键
    for key in set(list(embeddings_front.keys()) + list(embeddings.keys())):
        # 如果键在两个字典中都存在，则相加
        if key in embeddings_front and key in embeddings:
            embeddings_front_combined[key] = embeddings_front[key] + embeddings[key]
        # 如果只在embeddings_front中存在
        elif key in embeddings_front:
            embeddings_front_combined[key] = embeddings_front[key]
        # 如果只在embeddings中存在
        else:
            embeddings_front_combined[key] = embeddings[key]

    # 对up视角做同样的处理
    for key in set(list(embeddings_up.keys()) + list(embeddings.keys())):
        if key in embeddings_up and key in embeddings:
            embeddings_up_combined[key] = embeddings_up[key] + embeddings[key]
        elif key in embeddings_up:
            embeddings_up_combined[key] = embeddings_up[key]
        else:
            embeddings_up_combined[key] = embeddings[key]

    # 使用合并后的字典替换原字典
    embeddings_front = embeddings_front_combined
    embeddings_up = embeddings_up_combined

    # 平均图嵌入embeddings和CLIP生成的embeddings
    # embeddings = {}
    # for key in embeddings1:
    #     embeddings[key] = (embeddings1[key] + embeddings2[key]) / 2

    return guidance, embeddings_front, embeddings_up


def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #
    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('train_process is in :', save_folder)
    #controlnet
    use_control_net = False
    #set up pretrain diffusion models and text_embedings 
    guidance, embeddings_front, embeddings_up = guidance_setup(guidance_opt)   
    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []

    for iteration in range(first_iter, opt.iterations + 1):        
        #TODO: DEBUG NETWORK_GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # progressively relaxing view range    
        if not opt.use_progressive:                
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

                # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
                
                print('scale up theta_range to:', scene.pose_args.theta_range)
                print('scale up radius_range to:', scene.pose_args.radius_range)
                print('scale up phi_range to:', scene.pose_args.phi_range)
                print('scale up fovy_range to:', scene.pose_args.fovy_range)

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getRandTrainCameras().copy()      

        # 在训练循环中使用
        # if not viewpoint_stack and opt.viewpoint_mode == "six_view":
        #     viewpoint_stack = scene.getSixViewCameras().copy()
        # if not viewpoint_stack and iteration < opt.warmup_iter:
        #     # viewpoint_stack = scene.getFullCoverageCameras(iteration).copy()
        #     viewpoint_stack = scene.getCircleVideoCameras_new(batch_size = int(opt.warmup_iter * guidance_opt.C_batch_size * 0.5)).copy()
        # if not viewpoint_stack:
        #     # viewpoint_stack = scene.getCircleVideoCameras(render45=False).copy()
        #     viewpoint_stack = scene.getCircleVideoCameras().copy()

        # 视点栈选择逻辑
        if not viewpoint_stack:
            # 前期训练使用标准视角
            if iteration < opt.warmup_iter:  # 在热身阶段前期使用标准视角
                viewpoint_stack = scene.getCircleVideoCameras(render45=False).copy()
                render45_mode = False
            else:  # 训练后期使用45度倾斜视角
                viewpoint_stack = scene.getCircleVideoCameras(render45=True).copy()
                render45_mode = True
            
        if iteration % 100 == 0:
            print(f"[ITER {iteration}] 当前使用{'45度倾斜' if render45_mode else '标准'}视角渲染")
        
        # else:
        #     viewpoint_stack = scene.getRandTrainCameras().copy() 
        
        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []

        text_z_inverse_front =torch.cat([embeddings_front['uncond'],embeddings_front['inverse_text']], dim=0)
        text_z_inverse_up =torch.cat([embeddings_up['uncond'],embeddings_up['inverse_text']], dim=0)

        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
            except:
                # viewpoint_stack = scene.getCircleVideoCameras(render45=False).copy()
                viewpoint_stack = scene.getCircleVideoCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                
            #pred text_z
            azimuth = viewpoint_cam.delta_azimuth
            polar = viewpoint_cam.delta_polar  # 获取极角(倾斜角度)

            # 根据delta_polar判断视角类型
            if polar != 0:  # 非零表示45度视角，使用up触发词
                current_embeddings = embeddings_up
                view_type = "up"
                text_z_inverse = text_z_inverse_up
                current_guidance = guidance['up']
                
            else:  # 为零表示正面视角，使用front触发词
                current_embeddings = embeddings_front
                view_type = "front"
                text_z_inverse = text_z_inverse_front
                current_guidance = guidance['front']
            
            text_z = [current_embeddings['uncond']]

            if guidance_opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(current_embeddings, azimuth, guidance_opt)
                text_z.append(text_z_comp)
                weights_.append(weights)

            else:                
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = current_embeddings['front']
                    end_z = current_embeddings['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = current_embeddings['side']
                    end_z = current_embeddings['back']
                text_z.append(r * start_z + (1 - r) * end_z)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                                sh_deg_aug_ratio = dataset.sh_deg_aug_ratio, 
                                bg_aug_ratio = dataset.bg_aug_ratio, 
                                shs_aug_ratio = dataset.shs_aug_ratio, 
                                scale_aug_ratio = dataset.scale_aug_ratio)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cams)

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # Loss
        warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.)
        guidance_scale = guidance_opt.guidance_scale
        _aslatent = False
        if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
            _aslatent=True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
                use_control_net = True
        if guidance_opt.perpneg:
            loss = current_guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images, 
                                                pred_depth=depths, pred_alpha=alphas,
                                                grad_scale=guidance_opt.lambda_guidance,
                                                use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                                weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                                guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse, opt=opt)

            # loss = guidance.train_step_perpneg_new(torch.stack(text_z_, dim=1), images, 
            #                                     pred_depth=depths, pred_alpha=alphas,
            #                                     grad_scale=guidance_opt.lambda_guidance,
            #                                     use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
            #                                     weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
            #                                     guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
        else:
            loss = current_guidance.train_step(torch.stack(text_z_, dim=1), images, 
                                    pred_depth=depths, pred_alpha=alphas,
                                    grad_scale=guidance_opt.lambda_guidance,
                                    use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                    resolution=(gcams.image_h, gcams.image_w),
                                    guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
            #raise ValueError(f'original version not supported.')

        # 添加在此处: 定期生成和保存SD图像
        # if iteration % 10 == 0 & opt.save_sd_images:
        #     sd_gen_folder = os.path.join(save_folder, "sd_generations")
        #     os.makedirs(sd_gen_folder, exist_ok=True)
            
        #     # 为每个视角选择一个代表性的text_z用于生成图像
        #     with torch.no_grad():
        #         # 从text_z_中选取一个用于生成(第一个元素)
        #         selected_text_z = text_z_[0]
                
        #         # 使用无条件嵌入和条件嵌入
        #         if selected_text_z.shape[0] >= 2:
        #             text_embeddings = selected_text_z[:2].unsqueeze(0)  # 只使用无条件和第一个条件嵌入
                    
        #             # 创建潜变量噪声起点
        #             latents = torch.randn(
        #                 (1, 4, gcams.image_h // 8, gcams.image_w // 8),
        #                 device=guidance.device,
        #                 dtype=guidance.precision_t,
        #                 generator=guidance.noise_gen
        #             )
                    
        #             # 设置推理步骤的时间步
        #             guidance.scheduler.set_timesteps(30, device=guidance.device)
                    
        #             # 逐步降噪生成图像
        #             for i, t in enumerate(guidance.scheduler.timesteps):
        #                 # 预处理潜变量输入
        #                 latent_model_input = guidance.scheduler.scale_model_input(latents, t)
        #                 latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
                        
        #                 # 预测噪声
        #                 noise_pred = guidance.unet(
        #                     latent_model_input.to(guidance.precision_t),
        #                     t.to(guidance.precision_t),
        #                     encoder_hidden_states=text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]).to(guidance.precision_t)
        #                 ).sample
                        
        #                 # 执行分类器引导
        #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #                 noise_pred = noise_pred_uncond + guidance_opt.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
        #                 # 计算前一个潜在向量
        #                 latents = guidance.scheduler.step(noise_pred, t, latents).prev_sample
                    
        #             # 解码潜在表示为图像
        #             images_gen = guidance.decode_latents(latents)
                    
        #             # 保存生成的图像
        #             gen_path = os.path.join(sd_gen_folder, f"sd_gen_iter_{iteration}.png")
        #             save_image(images_gen, gen_path)
        #             print(f"[ITER {iteration}] 已保存SD生成图像到 {gen_path}")

        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales,dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths) 
        # loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale #opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_for_log = loss.item() # 纯3D更新loss

            # 添加损失记录到TensorBoard
            if tb_writer:
                # 记录总损失
                tb_writer.add_scalar('loss/total', loss.item(), iteration)
                
                # 记录各个损失组件
                if 'guidance' in locals() and hasattr(guidance, 'last_guidance_loss'):
                    tb_writer.add_scalar('loss/guidance', guidance.last_guidance_loss, iteration)
                    
                # TV损失
                tb_writer.add_scalar('loss/tv', loss_tv.item(), iteration)
                tb_writer.add_scalar('loss/tv_weighted', (opt.lambda_tv * loss_tv).item(), iteration)
                
                # 缩放损失
                tb_writer.add_scalar('loss/scale', loss_scale.item(), iteration)
                tb_writer.add_scalar('loss/scale_weighted', (opt.lambda_scale * loss_scale).item(), iteration)
            
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)  

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0: #or (dataset._white_background and iteration == opt.densify_from_iter)
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene._model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)



def prepare_output_and_logger(args):    
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace)
        
    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok = True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
            #save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))   
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import yaml

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default='/home/s414e2/CJH/Text-to-3D/LucidDreamer/configs/new_experiment/screw/screw13.yaml')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=5) # [2500,5000,7500,10000,12000]
    parser.add_argument("--save_ratio", type=int, default=2) # [10000,12000]
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--device", type=str, default='cuda')

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        # override device
        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

    # save iterations
    test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    args.test_iterations = test_iter

    save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    args.save_iterations = save_iter

    print('Test iter:', args.test_iterations)
    print('Save iter:', args.save_iterations)

    print("Optimizing " + lp._model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_video)

    # All done
    print("\nTraining complete.")
