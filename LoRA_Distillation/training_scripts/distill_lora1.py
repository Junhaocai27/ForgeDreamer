import os
import sys
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import random
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.optim as optim
import itertools
from diffusers.optimization import get_scheduler
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import traceback
import numpy as np
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# 添加lora_diffusion包路径
sys.path.append('/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora')
from lora_diffusion import inject_trainable_lora, inspect_lora
from lora_diffusion import save_all

# 自定义函数代替 create_token_embeddings
def init_token_embeddings(token_ids, embedding_weights, init_text=None, tokenizer=None):
    """增强的token embeddings初始化函数
    
    Args:
        token_ids: 需要初始化的token ID列表
        embedding_weights: 整个embeddings权重矩阵
        init_text: 可选，用于初始化的文本
        tokenizer: 可选，tokenizer用于编码init_text
        
    Returns:
        初始化后的embeddings，可以覆盖原权重
    """
    # 克隆对应token的embeddings
    initialized_weights = embedding_weights[token_ids].clone()
    
    # 如果提供了初始化文本和tokenizer，使用它们进行初始化
    if init_text and tokenizer:
        # 将初始化文本编码为token ID
        init_text_ids = tokenizer(init_text, return_tensors="pt").input_ids[0]
        # 如果文本被编码为多个token，取平均值
        if len(init_text_ids) > 1:
            init_embedding = torch.mean(embedding_weights[init_text_ids], dim=0, keepdim=True)
        else:
            init_embedding = embedding_weights[init_text_ids].clone()
            
        # 对每个placeholder token应用相同的初始化
        for i in range(len(token_ids)):
            initialized_weights[i] = init_embedding
    
    # 添加一些随机噪声以帮助训练
    # 对每个权重添加小的随机扰动，有助于避免局部最小值
    noise_scale = 0.1
    noise = torch.randn_like(initialized_weights) * noise_scale
    initialized_weights = initialized_weights + noise
    
    return initialized_weights


def analyze_lora_file(lora_path):
    """分析LoRA文件结构和信息"""
    print(f"分析LoRA文件: {lora_path}")
    try:
        # 使用safetensors加载LoRA文件
        lora_state_dict = load_file(lora_path)
        
        # 分析键值
        print(f"发现 {len(lora_state_dict)} 个权重键")
        
        # 检查是否存在元数据
        lora_metadata = {}
        try:
            with safe_open(lora_path, framework="pt") as f:
                lora_metadata = f.metadata() or {}
                print(f"元数据: {lora_metadata}")
        except Exception as e:
            print(f"读取元数据时出错: {str(e)}")
        
        # 检查元数据中的占位符信息
        placeholder_tokens = lora_metadata.get("placeholder_tokens", "")
        
        if placeholder_tokens:
            print(f"发现占位符信息: {placeholder_tokens}")
            return {"placeholder_tokens": placeholder_tokens.split(",")}
        else:
            # 检查常见的元数据键名
            for key in lora_state_dict.keys():
                if "token" in key.lower() or "embed" in key.lower():
                    print(f"可能与词嵌入相关的键: {key}")
        
        return lora_metadata
    except Exception as e:
        print(f"分析LoRA文件时出错: {str(e)}")
        traceback.print_exc()
        return {}


def get_diverse_prompt(view_name, view_trigger, step, prompt_templates, object_name):
    """生成更多样化的提示词，使用步数来增加随机性但保持一定的一致性"""
    # 扩展模板库
    if view_name in prompt_templates:
        templates = prompt_templates[view_name] + prompt_templates["generic"]
    else:
        templates = prompt_templates["generic"]
    
    # 使用步数影响随机选择以增加多样性但保持一致性
    template_idx = (step + hash(view_name)) % len(templates)
    template = templates[template_idx]
    
    # 多样化的修饰词
    modifiers = ["", "high quality", "detailed", "4k", "sharp focus", 
                "photorealistic", "studio lighting", "professional photo"]
    modifier_idx = (step + hash(view_trigger)) % len(modifiers)
    modifier = modifiers[modifier_idx]
    
    # 随机添加一些对象特定的描述词
    if object_name.lower() == "screw":
        object_descriptors = ["metallic", "shiny", "industrial", "mechanical", ""]
    else:
        object_descriptors = ["detailed", "realistic", "high-quality", ""]
    
    obj_desc = random.choice(object_descriptors)
    
    # 组合生成最终提示词
    prompt_elements = [template.format(view_trigger)]
    if obj_desc:
        prompt_elements.append(obj_desc)
    if modifier:
        prompt_elements.append(modifier)
    
    return ", ".join(prompt_elements)


def distill_lora_improved(
    base_model_path="/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base",
    output_dir="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/distilled_views/screw",
    front_lora="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex/front/final_lora.safetensors",
    front_trigger="<screw_front>",
    up_lora="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex/up/final_lora.safetensors",
    up_trigger="<screw_up>",
    object_name="screw",
    device="cuda:3",
    batch_size=1,  # 保持小批量以减少内存问题
    gradient_accumulation_steps=4,
    num_steps=1000,
    save_steps=100,
    lora_rank=16,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    lr_warmup_steps=100,
    weight_decay=0.001,
    resolution=512,
    enable_mixed_training=True,
    enable_progressive_training=True,
):
    try:
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        
        # 设置环境变量以便更好地捕获CUDA错误
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建TensorBoard日志目录
        log_dir = os.path.join(output_dir, "logs", f"{object_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志将保存到: {log_dir}")
        print(f"运行 'tensorboard --logdir={log_dir}' 来查看训练进度")
        
        # 记录训练超参数
        writer.add_text("配置/基础模型", base_model_path, 0)
        writer.add_text("配置/前视角LoRA", front_lora, 0)
        writer.add_text("配置/上视角LoRA", up_lora, 0)
        writer.add_text("配置/前视角触发词", front_trigger, 0)
        writer.add_text("配置/上视角触发词", up_trigger, 0)
        writer.add_scalar("配置/批量大小", batch_size, 0)
        writer.add_scalar("配置/总步数", num_steps, 0)
        writer.add_scalar("配置/LoRA秩", lora_rank, 0)
        writer.add_scalar("配置/学习率", learning_rate, 0)
        writer.add_text("配置/混合训练", str(enable_mixed_training), 0)
        writer.add_text("配置/渐进式训练", str(enable_progressive_training), 0)
        
        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 加载基础模型组件 - 按照 cli_lora_pti.py 的方式
        print("加载基础模型组件...")
        
        # 1. 首先加载tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
        )
        
        # 2. 加载text_encoder
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
        ).to(device)
        
        # 3. 加载UNet
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(device)
        
        # 4. 加载VAE (可以稍后释放内存)
        vae = AutoencoderKL.from_pretrained(
            base_model_path,
            subfolder="vae",
        ).to(device)
        
        # 5. 加载噪声调度器
        noise_scheduler = DDPMScheduler.from_pretrained(
            base_model_path, 
            subfolder="scheduler"
        )
        
        # 记录基础模型的tokenizer大小
        base_vocab_size = len(tokenizer)
        print(f"基础模型词汇表大小: {base_vocab_size}")
        
        # 存储教师模型的字典
        teacher_models = {}
        triggers = {
            "front": front_trigger,
            "up": up_trigger
        }
        
        # 分析LoRA文件
        print("分析原始LoRA文件")
        front_lora_info = analyze_lora_file(front_lora)
        up_lora_info = analyze_lora_file(up_lora)
        
        # 处理触发词 - 在加载模型前先处理
        print("处理触发词...")
        all_trigger_words = []
        for view_name, trigger in triggers.items():
            # 去除尖括号
            trigger_word = trigger.strip("<>")
            all_trigger_words.append(trigger_word)
            print(f"{view_name}视角触发词: {trigger} (不带尖括号: {trigger_word})")
        
        unique_trigger_words = list(set(all_trigger_words))
        print(f"去重后的触发词列表: {unique_trigger_words}")
        
        # 创建占位符token列表
        placeholder_tokens = [f"<{word}>" for word in unique_trigger_words]
        
        # 扩展tokenizer以包含这些token
        original_token_count = len(tokenizer)
        tokenizer.add_tokens(placeholder_tokens)
        print(f"添加了 {len(tokenizer) - original_token_count} 个新token到tokenizer")
        
        # 获取token IDs 并进行安全检查
        placeholder_token_ids = []
        for token in placeholder_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            placeholder_token_ids.append(token_id)
            print(f"触发词 {token} 的token ID: {token_id}")
            # 验证ID在有效范围内
            if token_id >= original_token_count:
                print(f"  - 这是新添加的token")
            else:
                print(f"  - 警告：这个token已经存在于原始tokenizer中，可能导致问题")
        
        # 扩展text_encoder的embeddings表以包含新添加的tokens
        text_encoder.resize_token_embeddings(len(tokenizer))
        print(f"已调整text_encoder嵌入表大小至: {text_encoder.get_input_embeddings().weight.shape[0]}")
        
        # 创建前视角模型的Pipeline
        front_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        # 应用LoRA权重
        try:
            if os.path.exists(front_lora):
                print(f"应用前视角LoRA: {front_lora}")
                from lora_diffusion.lora import patch_pipe, tune_lora_scale
                
                # 使用patch_pipe应用LoRA权重
                patch_pipe(
                    front_pipe,
                    front_lora,
                    patch_text=True,
                    patch_ti=True,
                    patch_unet=True
                )
                
                # 调整LoRA缩放系数
                tune_lora_scale(front_pipe.unet, 1.0)
                tune_lora_scale(front_pipe.text_encoder, 1.0)
                
                # 创建前视角教师模型
                front_model = {
                    "text_encoder": front_pipe.text_encoder,
                    "unet": front_pipe.unet
                }
                
                # 添加到teacher_models
                teacher_models["front"] = front_model
                print("前视角LoRA加载成功")
            else:
                print(f"错误：LoRA文件不存在: {front_lora}")
        except Exception as e:
            print(f"加载前视角LoRA时出错: {str(e)}")
            writer.add_text("加载错误", f"前视角LoRA加载失败: {str(e)}", 0)
            traceback.print_exc()
        
        # 创建上视角模型的Pipeline
        up_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        # 应用LoRA权重
        try:
            if os.path.exists(up_lora):
                print(f"应用上视角LoRA: {up_lora}")
                from lora_diffusion.lora import patch_pipe, tune_lora_scale
                
                # 使用patch_pipe应用LoRA权重
                patch_pipe(
                    up_pipe,
                    up_lora,
                    patch_text=True,
                    patch_ti=True,
                    patch_unet=True
                )
                
                # 调整LoRA缩放系数
                tune_lora_scale(up_pipe.unet, 1.0)
                tune_lora_scale(up_pipe.text_encoder, 1.0)
                
                # 创建上视角教师模型
                up_model = {
                    "text_encoder": up_pipe.text_encoder,
                    "unet": up_pipe.unet
                }
                
                # 添加到teacher_models
                teacher_models["up"] = up_model
                print("上视角LoRA加载成功")
            else:
                print(f"错误：LoRA文件不存在: {up_lora}")
        except Exception as e:
            print(f"加载上视角LoRA时出错: {str(e)}")
            writer.add_text("加载错误", f"上视角LoRA加载失败: {str(e)}", 0)
            traceback.print_exc()
        
        # 检查是否至少有一个教师模型
        if not teacher_models:
            print("错误：所有LoRA加载失败，无法继续蒸馏")
            writer.add_text("错误", "所有LoRA加载失败，无法继续蒸馏", 0)
            writer.close()
            return
        
        print(f"加载成功的教师模型: {list(teacher_models.keys())}")
        writer.add_text("加载状态", f"成功加载的教师模型: {list(teacher_models.keys())}", 0)
        
        # 创建学生模型
        print("创建学生模型...")
        
        # 创建新的text_encoder和unet用于学生模型
        student_text_encoder = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
        ).to(device)
        
        student_unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(device)
        
        # 调整学生模型的token嵌入大小
        student_text_encoder.resize_token_embeddings(len(tokenizer))
            
        # 初始化新token的嵌入
        token_embeds = student_text_encoder.get_input_embeddings()
        
        # 对每个触发词使用更精细的初始化
        for i, (token, token_id) in enumerate(zip(placeholder_tokens, placeholder_token_ids)):
            # 从触发词中提取视角信息
            view_info = token.strip("<>").split("_")[-1]  # 例如 "screw_front" -> "front"
            
            # 使用与视角相关的词语进行初始化
            if view_info == "front":
                init_text = f"a front view of {object_name}"
            elif view_info == "up" or view_info == "top":
                init_text = f"a top view of {object_name}"
            else:
                init_text = object_name
                
            print(f"为触发词 {token} 选择初始化文本: '{init_text}'")
            
            # 使用object_name和相关描述词的嵌入初始化
            try:
                # 安全地获取token ids
                init_text_ids = tokenizer(init_text, return_tensors="pt").input_ids[0]
                
                # 使用有效ID的平均嵌入
                valid_ids = init_text_ids[init_text_ids < original_token_count]
                if len(valid_ids) == 0:
                    print(f"警告：'{init_text}'中没有有效token ID，使用随机初始化")
                    token_embeds.weight.data[token_id] = torch.randn_like(token_embeds.weight.data[0]) * 0.1
                else:
                    init_embedding = torch.mean(token_embeds.weight.data[valid_ids], dim=0, keepdim=True)
                    token_embeds.weight.data[token_id] = init_embedding
                    
                    # 添加一些随机噪声以帮助训练
                    noise_scale = 0.1
                    noise = torch.randn_like(token_embeds.weight.data[token_id]) * noise_scale
                    token_embeds.weight.data[token_id] += noise
            except Exception as e:
                print(f"初始化token {token} 时出错: {str(e)}")
                # 使用备选方案
                token_embeds.weight.data[token_id] = torch.randn_like(token_embeds.weight.data[0]) * 0.1
        
        # 冻结基础模型权重
        student_text_encoder.requires_grad_(False)
        student_unet.requires_grad_(False)
        
        # 定义LoRA目标模块 - 确保与cli_lora_pti.py中相同
        unet_lora_target_modules = {"CrossAttention", "Attention", "GEGLU"}
        text_encoder_lora_target_modules = {"CLIPAttention"}  # 修正为CLIPAttention而不是CLIPSdpaAttention
        
        # 注入LoRA到学生模型 - 使用与cli_lora_pti.py相同的方法
        print(f"应用LoRA到学生模型 (rank={lora_rank})...")
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=unet_lora_target_modules,
            dropout_p=0.1,
            scale=lora_rank,
        )
        
        text_encoder_lora_params, _ = inject_trainable_lora(
            student_text_encoder,
            r=lora_rank,
            target_replace_module=text_encoder_lora_target_modules,
            dropout_p=0.1,
            scale=lora_rank,
        )
        
        # 设置token embedding为可训练
        token_embeds.weight.requires_grad = False
        token_embeds.weight[placeholder_token_ids].requires_grad = True
        
        # 训练设置
        student_unet.train()
        student_text_encoder.train()
        
        # 创建优化器
        params_to_optimize = [
            {"params": itertools.chain(*unet_lora_params), "lr": learning_rate},
            {"params": itertools.chain(*text_encoder_lora_params), "lr": learning_rate},
            {"params": [token_embeds.weight[placeholder_token_ids]], "lr": learning_rate * 2.0}  # token嵌入使用更高的学习率
        ]
        
        optimizer = optim.AdamW(params_to_optimize, weight_decay=weight_decay)
        
        # 使用调度器
        lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=num_steps,
        )
        
        # 扩展提示词模板库
        prompt_templates = {
            "generic": [
                "A detailed photo of {}",
                "A close-up shot of {}",
                "A high resolution image of {}",
                "A clear picture of {}"
            ],
            "front": [
                "A front view of {}, detailed",
                "Looking at {} from the front",
                "A picture of {} showing the front side",
                "A forward facing view of {}"
            ],
            "up": [
                "An up view of {}, detailed",
                "Looking at {} from above",
                "A picture of {} showing the top side",
                "A top-down view of {}"
            ],
            "mixed": [
                "A detailed view of {}",
                "A professional photo of {}",
                "A comprehensive image of {}"
            ]
        }
        
        # 开始蒸馏训练
        print("开始蒸馏...")
        progress_bar = tqdm(range(num_steps))
        
        # 记录每个视角的损失历史
        view_losses = {view: [] for view in teacher_models.keys()}
        if enable_mixed_training:
            view_losses["mixed"] = []
        
        # 训练循环
        for step in range(num_steps):
            total_loss = 0.0
            
            # 计算当前训练进度百分比
            progress_percent = step / num_steps
            
            # 动态调整损失权重
            if enable_progressive_training:
                single_view_weight = max(0.9 - progress_percent * 0.4, 0.5)  # 从0.9降到0.5
                mixed_view_weight = 1.0 - single_view_weight  # 从0.1增加到0.5
            else:
                single_view_weight = 0.7
                mixed_view_weight = 0.3
            
            # 对每个视角进行一步训练
            for view_name, view_trigger in triggers.items():
                # 跳过没有对应教师模型的视角
                if view_name not in teacher_models:
                    continue
                
                # 获取当前视角的教师模型
                teacher = teacher_models[view_name]
                
                # 创建多样化的提示词
                prompts = []
                for _ in range(batch_size):
                    # 获取提示词模板
                    if view_name in prompt_templates:
                        templates = prompt_templates[view_name] + prompt_templates["generic"]
                    else:
                        templates = prompt_templates["generic"]
                    
                    # 使用步数影响随机选择
                    template_idx = (step + hash(view_name)) % len(templates)
                    template = templates[template_idx]
                    
                    # 生成提示词
                    prompt = template.format(view_trigger)
                    prompts.append(prompt)
                
                # 生成随机噪声和时间步
                latents = torch.randn((batch_size, 4, 64, 64)).to(device)
                
                # 使用多样化的时间步采样
                if step % 3 == 0:  # 早期时间步
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps // 4, (batch_size,),
                        device=device
                    ).long()
                elif step % 3 == 1:  # 中期时间步
                    timesteps = torch.randint(
                        noise_scheduler.config.num_train_timesteps // 4,
                        noise_scheduler.config.num_train_timesteps * 3 // 4, 
                        (batch_size,),
                        device=device
                    ).long()
                else:  # 后期时间步
                    timesteps = torch.randint(
                        noise_scheduler.config.num_train_timesteps * 3 // 4,
                        noise_scheduler.config.num_train_timesteps, 
                        (batch_size,),
                        device=device
                    ).long()
                
                # 编码提示词
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # 获取教师模型的输出
                with torch.no_grad():
                    teacher_text_embeds = teacher["text_encoder"](text_inputs.input_ids)[0]
                    teacher_output = teacher["unet"](
                        latents, timesteps, encoder_hidden_states=teacher_text_embeds
                    ).sample
                
                # 学生模型处理
                student_text_embeds = student_text_encoder(text_inputs.input_ids)[0]
                student_output = student_unet(
                    latents, timesteps, encoder_hidden_states=student_text_embeds
                ).sample
                
                # 计算损失
                view_loss = F.mse_loss(student_output, teacher_output)
                
                # 应用渐进式权重
                if enable_progressive_training:
                    view_loss = view_loss * single_view_weight
                    
                # 缩放损失
                view_loss = view_loss / gradient_accumulation_steps
                
                # 反向传播
                view_loss.backward()
                
                # 记录损失
                scaled_loss = view_loss.item() * gradient_accumulation_steps
                if enable_progressive_training:
                    scaled_loss = scaled_loss / single_view_weight  # 恢复原始损失值用于记录
                    
                view_losses[view_name].append(scaled_loss)
                
                # 记录到TensorBoard
                writer.add_scalar(f"损失/{view_name}视角", scaled_loss, step)
                
                # 累加损失
                total_loss += scaled_loss
            
            # 混合视角训练
            if enable_mixed_training and len(teacher_models) > 1 and (step % 3 == 0):
                available_views = list(teacher_models.keys())
                
                if len(available_views) >= 2:
                    view1, view2 = random.sample(available_views, 2)
                    
                    # 创建混合视角的提示词
                    mix_prompts = []
                    for _ in range(batch_size):
                        template = random.choice(prompt_templates["mixed"])
                        mix_prompts.append(template.format(random.choice([triggers[view1], triggers[view2]])))
                    
                    # 生成新的随机噪声和时间步
                    mix_latents = torch.randn((batch_size, 4, 64, 64)).to(device)
                    mix_timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                        device=device
                    ).long()
                    
                    # 编码提示词
                    mix_text_inputs = tokenizer(
                        mix_prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # 获取两个教师模型的输出并平均
                    with torch.no_grad():
                        teacher1_embeds = teacher_models[view1]["text_encoder"](mix_text_inputs.input_ids)[0]
                        teacher1_output = teacher_models[view1]["unet"](
                            mix_latents, mix_timesteps, encoder_hidden_states=teacher1_embeds
                        ).sample
                        
                        teacher2_embeds = teacher_models[view2]["text_encoder"](mix_text_inputs.input_ids)[0]
                        teacher2_output = teacher_models[view2]["unet"](
                            mix_latents, mix_timesteps, encoder_hidden_states=teacher2_embeds
                        ).sample
                        
                        teacher_mix_output = (teacher1_output + teacher2_output) / 2.0
                    
                    # 学生模型处理
                    student_mix_embeds = student_text_encoder(mix_text_inputs.input_ids)[0]
                    student_mix_output = student_unet(
                        mix_latents, mix_timesteps, encoder_hidden_states=student_mix_embeds
                    ).sample
                    
                    # 计算混合损失
                    mix_loss = F.mse_loss(student_mix_output, teacher_mix_output)
                    
                    # 应用混合视角的权重
                    if enable_progressive_training:
                        mix_loss = mix_loss * mixed_view_weight
                        
                    # 缩放损失
                    mix_loss = mix_loss / gradient_accumulation_steps
                    mix_loss.backward()
                    
                    # 记录混合训练损失
                    scaled_mix_loss = mix_loss.item() * gradient_accumulation_steps
                    if enable_progressive_training:
                        scaled_mix_loss = scaled_mix_loss / mixed_view_weight  # 恢复原始损失值用于记录
                        
                    view_losses["mixed"].append(scaled_mix_loss)
                    writer.add_scalar("损失/混合视角", scaled_mix_loss, step)
                    
                    # 累加到总损失
                    total_loss += scaled_mix_loss
            
            # 实现梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        itertools.chain(*unet_lora_params),
                        itertools.chain(*text_encoder_lora_params),
                        [token_embeds.weight[placeholder_token_ids]]
                    ), 
                    max_norm=1.0
                )
                
                # 更新权重
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            
            # 计算平均损失
            if len(teacher_models) > 0:
                divisor = len(teacher_models)
                if enable_mixed_training and "mixed" in view_losses and view_losses["mixed"]:
                    divisor += 1
                avg_loss = total_loss / divisor
            else:
                avg_loss = 0
            
            # 记录到TensorBoard
            writer.add_scalar("损失/平均损失", avg_loss, step)
            writer.add_scalar("学习率", lr_scheduler.get_last_lr()[0], step)
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix(loss=avg_loss, lr=lr_scheduler.get_last_lr()[0])
            
            # 每隔一定步数清理CUDA缓存
            if step % 50 == 0:
                torch.cuda.empty_cache()
            
            # 保存检查点
            if (step + 1) % save_steps == 0 or step == num_steps - 1:
                # 创建检查点保存路径
                save_path = os.path.join(
                    output_dir, f"{object_name}_distilled_step_{step + 1}.safetensors"
                )
                
                # 使用save_all函数保存LoRA权重 - 确保传递正确的参数
                save_all(
                    student_unet,
                    student_text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=save_path,
                    target_replace_module_text=text_encoder_lora_target_modules,
                    target_replace_module_unet=unet_lora_target_modules,
                )
                
                print(f"保存检查点至: {save_path}")
                writer.add_text("保存点", f"保存检查点至: {save_path}", step)
        
        # 保存最终结果
        final_path = os.path.join(output_dir, f"{object_name}_final_distilled_lora.safetensors")
        
        # 使用save_all函数保存LoRA权重
        save_all(
            student_unet,
            student_text_encoder,
            placeholder_token_ids=placeholder_token_ids,
            placeholder_tokens=placeholder_tokens,
            save_path=final_path,
            target_replace_module_text=text_encoder_lora_target_modules,
            target_replace_module_unet=unet_lora_target_modules,
        )
        
        print(f"蒸馏完成，最终LoRA保存至: {final_path}")
        print(f"包含的触发词: {placeholder_tokens}")
        writer.add_text("完成", f"蒸馏完成，最终LoRA保存至: {final_path}", num_steps)
        writer.add_text("完成", f"包含的触发词: {placeholder_tokens}", num_steps)
        
        # 关闭TensorBoard writer
        writer.close()
        
        return final_path
    
    except Exception as e:
        print(f"蒸馏过程中发生错误: {str(e)}")
        traceback.print_exc()
        try:
            writer.add_text("严重错误", f"蒸馏失败: {str(e)}", 0)
            writer.close()
        except:
            pass
        return None

if __name__ == "__main__":
    # 直接执行蒸馏
    distill_lora_improved()