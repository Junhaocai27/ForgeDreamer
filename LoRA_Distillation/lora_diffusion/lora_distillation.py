import argparse
import inspect
import itertools
import math
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Optional, Tuple, Literal

import sys
sys.path.append('/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora')
from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inspect_lora,
    save_lora_weight,
    patch_pipe,
)


def distill_multiview_lora(
    # 基础参数
    base_model_path: str,
    output_dir: str,
    # 前视角LoRA
    front_lora: str,
    front_trigger: str,
    # 上视角LoRA
    up_lora: str,
    up_trigger: str,
    # 可选：侧视角LoRA
    side_lora: Optional[str] = None,
    side_trigger: Optional[str] = None,
    # 可选：自定义输出触发词
    output_trigger: Optional[str] = None,
    # 训练参数
    object_name: str = "object",
    seed: int = 42,
    resolution: int = 512,
    batch_size: int = 4,
    num_steps: int = 2000,
    save_steps: int = 200,
    lora_rank: int = 16,
    lora_alpha: float = 16,
    lora_dropout: float = 0.1,
    learning_rate: float = 1e-4,
    lr_scheduler: str = "cosine",
    lr_warmup_steps: int = 100,
    weight_decay: float = 0.01,
    mixed_precision: bool = False,
    device: str = "cuda",
    out_name: str = "final_distilled_lora",
):
    """
    将多个视角的LoRA权重蒸馏到一个统一的LoRA中
    
    参数:
        base_model_path: 基础模型路径
        output_dir: 输出目录
        front_lora: 前视角LoRA路径
        front_trigger: 前视角触发词，如 "<screw_front>"
        up_lora: 上视角LoRA路径
        up_trigger: 上视角触发词，如 "<screw_up>"
        side_lora: 侧视角LoRA路径(可选)
        side_trigger: 侧视角触发词(可选)
        output_trigger: 输出LoRA的触发词，默认与所有输入触发词保持一致
        object_name: 物体名称，用于文件名
        seed: 随机种子
        resolution: 分辨率
        batch_size: 批处理大小
        num_steps: 蒸馏步数
        save_steps: 保存检查点的步数间隔
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: LoRA Dropout概率
        learning_rate: 学习率
        lr_scheduler: 学习率调度器类型
        lr_warmup_steps: 学习率预热步数
        weight_decay: 权重衰减
        mixed_precision: 是否使用混合精度训练
        device: 训练设备
        out_name: 输出文件名
    """
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集视角LoRA路径和触发词
    lora_paths = {}
    triggers = {}
    
    if front_lora and front_trigger:
        lora_paths["front"] = front_lora
        triggers["front"] = front_trigger
    
    if up_lora and up_trigger:
        lora_paths["up"] = up_lora
        triggers["up"] = up_trigger
    
    if side_lora and side_trigger:
        lora_paths["side"] = side_lora
        triggers["side"] = side_trigger
    
    # 打印训练配置
    print(f"开始多视角LoRA蒸馏:")
    print(f"基础模型: {base_model_path}")
    print(f"物体: {object_name}")
    print(f"学习率: {learning_rate}")
    print(f"批量大小: {batch_size}")
    print(f"训练步数: {num_steps}")
    print(f"LoRA秩: {lora_rank}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print("视角与触发词:")
    for view, trigger in triggers.items():
        print(f"  - {view}: {trigger}")
    
    # 加载基础模型
    print("\n加载基础模型组件...")
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model_path,
        subfolder="tokenizer",
    )
    
    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    
    # 设置教师模型 - 为每个视角创建一个模型
    teacher_models = {}
    
    print("\n加载教师模型...")
    for view_name, lora_path in lora_paths.items():
        print(f"加载{view_name}视角的教师模型...")
        
        # 创建基础模型的副本
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
        ).to(device)
        
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(device)
        
        # 创建教师模型字典
        teacher_model = {
            "unet": unet,
            "text_encoder": text_encoder
        }
        
        # 应用LoRA权重
        print(f"将{view_name}视角的LoRA权重应用到教师模型: {lora_path}")
        try:
            patch_pipe(
                teacher_model,
                lora_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True
            )
            print(f"{view_name}视角LoRA应用成功")
        except Exception as e:
            print(f"应用{view_name}视角LoRA时出错: {str(e)}")
            print(f"跳过{view_name}视角")
            continue
        
        # 设置为评估模式
        teacher_model["unet"].eval()
        teacher_model["text_encoder"].eval()
        
        teacher_models[view_name] = teacher_model
    
    # 创建学生模型
    print("\n创建学生模型...")
    student_unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
    ).to(device)
    
    student_text_encoder = CLIPTextModel.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
    ).to(device)
    
    # 冻结基础模型权重
    student_unet.requires_grad_(False)
    student_text_encoder.requires_grad_(False)
    
    # 应用LoRA
    print(f"应用LoRA到学生模型 (rank={lora_rank})...")
    unet_lora_target_modules = {"CrossAttention", "Attention", "GEGLU"}
    text_encoder_lora_target_modules = {"CLIPAttention"}
    
    unet_lora_params, _ = inject_trainable_lora(
        student_unet,
        r=lora_rank,
        target_replace_module=unet_lora_target_modules,
        dropout_p=lora_dropout,
        scale=lora_alpha,
    )
    
    text_encoder_lora_params, _ = inject_trainable_lora(
        student_text_encoder,
        r=lora_rank,
        target_replace_module=text_encoder_lora_target_modules,
        dropout_p=lora_dropout,
        scale=lora_alpha,
    )
    
    # 检查LoRA初始化情况
    print("检查LoRA初始化状态:")
    inspect_lora(student_unet)
    inspect_lora(student_text_encoder)
    
    # 设置训练模式
    student_unet.train()
    student_text_encoder.train()
    
    # 创建优化器
    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": learning_rate},
        {"params": itertools.chain(*text_encoder_lora_params), "lr": learning_rate},
    ]
    
    optimizer = optim.AdamW(params_to_optimize, weight_decay=weight_decay)
    
    # 设置学习率调度器
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_steps,
    )
    
    # 准备提示词模板
    # 为不同的视角创建不同的模板
    prompt_templates = {
        "generic": [
            "A detailed photo of {}",
            "A close-up shot of {}",
            "A high resolution image of {}"
        ],
        "front": [
            "A front view of {}, detailed",
            "Looking at {} from the front",
            "A picture of {} showing the front side"
        ],
        "up": [
            "An up view of {}, detailed",
            "Looking at {} from above",
            "A picture of {} showing the top side"
        ],
        "side": [
            "A side view of {}, detailed",
            "Looking at {} from the side",
            "A picture of {} showing the side"
        ]
    }
    
    # 开始蒸馏训练
    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("蒸馏进度")
    
    # 蒸馏训练循环
    for step in range(num_steps):
        total_loss = 0.0
        
        # 对每个视角进行一步训练
        for view_name, view_trigger in triggers.items():
            # 跳过没有对应教师模型的视角
            if view_name not in teacher_models:
                continue
            
            # 获取当前视角的教师模型
            teacher = teacher_models[view_name]
            
            # 为当前视角选择模板
            if view_name in prompt_templates:
                templates = prompt_templates[view_name] + prompt_templates["generic"]
            else:
                templates = prompt_templates["generic"]
            
            # 创建随机提示词
            prompts = []
            for _ in range(batch_size):
                template = random.choice(templates)
                prompts.append(template.format(view_trigger))
            
            # 生成随机噪声和时间步
            latents = torch.randn((batch_size, 4, 64, 64)).to(device)
            timesteps = torch.randint(
                0, int(noise_scheduler.config.num_train_timesteps * 0.8), (batch_size,),
                device=device
            ).long()
            
            # 获取教师模型的输出
            with torch.no_grad():
                # 编码提示词
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # 获取文本嵌入
                teacher_text_embeds = teacher["text_encoder"](text_inputs.input_ids)[0]
                
                # 获取模型预测
                teacher_output = teacher["unet"](
                    latents, timesteps, encoder_hidden_states=teacher_text_embeds
                ).sample
            
            # 学生模型使用相同的提示词 - 保持视角触发词
            student_text_embeds = student_text_encoder(text_inputs.input_ids)[0]
            student_output = student_unet(
                latents, timesteps, encoder_hidden_states=student_text_embeds
            ).sample
            
            # 计算损失
            view_loss = F.mse_loss(student_output, teacher_output)
            
            # 反向传播
            view_loss.backward()
            
            # 累加损失
            total_loss += view_loss.item()
        
        # 更新权重
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        
        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_postfix(loss=total_loss / len(teacher_models), lr=lr_scheduler.get_last_lr()[0])
        
        # 保存检查点
        if (step + 1) % save_steps == 0 or step == num_steps - 1:
            save_path = os.path.join(
                output_dir, f"{object_name}_distilled_step_{step + 1}.safetensors"
            )
            
            # 保存当前LoRA权重
            save_lora_weight(
                student_unet,
                student_text_encoder,
                save_path,
                target_replace_module_text=text_encoder_lora_target_modules,
                target_replace_module_unet=unet_lora_target_modules
            )
            
            print(f"保存检查点至: {save_path}")
            
            # 检查LoRA权重变化
            moved_unet = torch.tensor(list(itertools.chain(*inspect_lora(student_unet).values()))).mean().item()
            moved_text = torch.tensor(list(itertools.chain(*inspect_lora(student_text_encoder).values()))).mean().item()
            print(f"LORA UNet 移动量: {moved_unet:.6f}")
            print(f"LORA Text Encoder 移动量: {moved_text:.6f}")
    
    # 保存最终结果
    final_path = os.path.join(output_dir, f"{out_name}.safetensors")
    save_lora_weight(
        student_unet,
        student_text_encoder,
        final_path,
        target_replace_module_text=text_encoder_lora_target_modules,
        target_replace_module_unet=unet_lora_target_modules
    )
    
    print(f"蒸馏完成，最终LoRA保存至: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多视角LoRA蒸馏工具 - 将多个视角的LoRA合并为一个，保留原始触发词")
    
    # 基础参数
    parser.add_argument("--base_model", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base", help="基础模型路径")
    parser.add_argument("--output_dir", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/distilled_views/screw", help="输出目录")
    parser.add_argument("--object_name", type=str, default="screw", help="物体名称，用于文件名")
    
    # 前视角LoRA
    parser.add_argument("--front_lora", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex/front/final_lora.safetensors", help="前视角LoRA路径")
    parser.add_argument("--front_trigger", type=str, default="<screw_front>", help="前视角触发词，如 <screw_front>")
    
    # 上视角LoRA
    parser.add_argument("--up_lora", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex/up/final_lora.safetensors", help="上视角LoRA路径")
    parser.add_argument("--up_trigger", type=str, default="<screw_up>", help="上视角触发词，如 <screw_up>")
    
    # 可选：侧视角LoRA
    parser.add_argument("--side_lora", type=str, help="侧视角LoRA路径(可选)")
    parser.add_argument("--side_trigger", type=str, help="侧视角触发词(可选)")
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resolution", type=int, default=512, help="分辨率")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--num_steps", type=int, default=2000, help="蒸馏步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存检查点的步数间隔")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA Dropout概率")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--mixed_precision", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--out_name", type=str, default="final_distilled_lora", help="输出文件名")
    
    args = parser.parse_args()
    
    # 直接执行蒸馏
    distill_multiview_lora(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        front_lora=args.front_lora,
        front_trigger=args.front_trigger,
        up_lora=args.up_lora,
        up_trigger=args.up_trigger,
        side_lora=args.side_lora,
        side_trigger=args.side_trigger,
        object_name=args.object_name,
        seed=args.seed,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        save_steps=args.save_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        device=args.device,
        out_name=args.out_name,
    )