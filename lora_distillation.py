import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import re
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler
from lora_diffusion import (
    patch_pipe, 
    tune_lora_scale, 
    inject_trainable_lora,
    extract_lora_ups_down
)
from accelerate import Accelerator

# 种子设置，确保可复现性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 文本提示数据集
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, text_encoder, device):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        
        # 预计算文本嵌入以提高效率
        self.text_embeddings = []
        for prompt in tqdm(prompts, desc="预计算文本嵌入"):
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                text_embedding = self.text_encoder(text_input_ids)[0]
            # 确保文本嵌入在正确的设备上
            self.text_embeddings.append(text_embedding.to(device))
            
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "text_embedding": self.text_embeddings[idx]
        }

# 蒸馏训练函数
def train_distilled_lora(args):
    # 设置加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.mixed_precision else "no",
    )
    
    # 日志设置
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "training_log.txt")
        
    def log(msg):
        if accelerator.is_main_process:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            print(msg)
    
    # 记录训练参数
    log(f"===== 训练参数 =====")
    for k, v in vars(args).items():
        log(f"{k}: {v}")
    log("====================")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 记录使用的设备
    device = accelerator.device
    log(f"使用设备: {device}")
    
    # 加载基础模型
    log(f"加载基础模型: {args.base_model_path}")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    
    # 主动将所有模型移至同一设备
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    # 将模型设置为评估模式
    vae.eval()
    text_encoder.eval()
    
    # 为学生模型(UNet)注入可训练的LoRA层
    log(f"为学生模型注入LoRA层: rank={args.lora_rank}")
    
    # 添加调试信息
    log(f"UNet类型: {type(unet)}")
    
    # 修改调用方式，并修正参数处理
    try:
        # 尝试使用修改后的参数调用
        log("尝试注入LoRA (基本方法)...")
        result = inject_trainable_lora(
            unet, 
            r=args.lora_rank, 
            target_replace_module=args.unet_target_modules.split(",")
        )
        
        # 根据返回值类型处理参数
        if isinstance(result, tuple) and len(result) >= 2:
            # 如果返回的是元组，使用前两个元素
            _, unet_names = result[:2]
            # 从 unet 中获取可训练参数
            unet_lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            log(f"找到 {len(unet_lora_params)} 个可训练的 LoRA 参数")
        else:
            # 如果返回的不是元组，可能直接返回了 unet
            unet_names = []
            # 从 unet 中获取可训练参数
            unet_lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            log(f"找到 {len(unet_lora_params)} 个可训练的 LoRA 参数")
    
    except TypeError as e:
        log(f"注入 LoRA 层失败: {str(e)}")
        log("尝试简化的注入方法...")
        try:
            # 尝试更简单的调用方式
            inject_trainable_lora(unet, r=args.lora_rank)
            # 手动找出可训练参数
            unet_lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            unet_names = []
            log(f"找到 {len(unet_lora_params)} 个可训练的 LoRA 参数")
        except Exception as e2:
            log(f"简化注入也失败: {str(e2)}")
            log("LoRA 注入失败，无法继续训练")
            return
    
    # 检查是否找到了可训练参数
    if len(unet_lora_params) == 0:
        log("错误：未找到任何可训练的 LoRA 参数，请检查 LoRA 注入")
        return
    
    # 检查所有参数是否在同一设备上
    for i, param in enumerate(unet_lora_params):
        if param.device != device:
            log(f"警告: 参数 {i} 在设备 {param.device} 上，而不是在 {device} 上。尝试移动...")
            unet_lora_params[i] = param.to(device)
    
    # 准备优化器
    params_to_optimize = unet_lora_params
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 准备教师LoRA模型路径
    teacher_lora_paths = []
    teacher_lora_prompts = {}
    
    # 解析输入的教师LoRA模型
    for teacher_info in args.teacher_loras:
        path, concept = teacher_info.split(":")
        teacher_lora_paths.append(path)
        
        # 检查文件是否存在
        if not os.path.exists(path):
            log(f"警告: 教师LoRA文件不存在 - {path}")
            continue
            
        # 生成一组与该概念相关的提示
        prompts = []
        for template in args.prompt_templates:
            prompts.append(template.replace("<concept>", concept))
        teacher_lora_prompts[path] = prompts
        
        log(f"添加教师LoRA: {path}")
        log(f"  相关概念: {concept}")
        log(f"  生成提示数量: {len(prompts)}")
    
    # 检查是否至少有一个有效的教师LoRA
    if len(teacher_lora_paths) == 0:
        log("错误: 没有找到有效的教师LoRA文件")
        return
        
    # 创建一个合并后的提示列表，用于训练
    all_prompts = []
    for prompts_list in teacher_lora_prompts.values():
        all_prompts.extend(prompts_list)
    
    log(f"总训练提示数量: {len(all_prompts)}")
    
    # 创建数据集和数据加载器
    dataset = PromptDataset(all_prompts, tokenizer, text_encoder, device)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # 设置学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # 准备加速器
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # 训练循环准备
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    log(f"开始训练!")
    log(f"  总批次大小 = {total_batch_size}")
    log(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    log(f"  总优化步数 = {args.max_train_steps}")
    
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("训练步骤")
    
    # 训练循环
    for epoch in range(args.num_epochs):
        unet.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # 选择当前批次使用的教师LoRA
                teacher_idx = np.random.randint(0, len(teacher_lora_paths))
                teacher_lora_path = teacher_lora_paths[teacher_idx]
                
                # 加载教师LoRA到临时UNet模型
                teacher_unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet")
                teacher_unet = teacher_unet.to(device)  # 确保模型在正确的设备上
                teacher_unet.eval()
                
                # 检查文件是否存在
                if not os.path.exists(teacher_lora_path):
                    log(f"错误: 教师LoRA文件不存在 - {teacher_lora_path}")
                    continue
                
                # 输出调试信息
                log(f"加载教师LoRA: {teacher_lora_path}")
                
                # 应用教师LoRA
                try:
                    # 首先尝试简单地加载权重文件，检查文件内容
                    try:
                        lora_state_dict = load_file(teacher_lora_path)
                        log(f"成功加载权重文件，包含 {len(lora_state_dict)} 个键")
                    except Exception as e:
                        log(f"加载权重文件失败: {str(e)}")
                        continue
                        
                    # 然后尝试应用到模型
                    patch_pipe(
                        {"unet": teacher_unet},
                        teacher_lora_path,
                        patch_text=False,
                        patch_ti=False,
                        patch_unet=True,
                    )
                    log("成功应用教师LoRA到UNet")
                except TypeError as e:
                    log(f"使用标准 patch_pipe 失败: {str(e)}")
                    # 尝试简化版本的 patch_pipe 调用
                    try:
                        patch_pipe(
                            {"unet": teacher_unet},
                            teacher_lora_path,
                        )
                        log("使用简化方式成功应用教师LoRA")
                    except Exception as e2:
                        log(f"简化 patch_pipe 也失败: {str(e2)}")
                        log(f"跳过此批次，使用下一个教师LoRA")
                        continue
                
                # 确保 text_embedding 在正确的设备上
                encoder_hidden_states = batch["text_embedding"]
                if encoder_hidden_states.device != device:
                    encoder_hidden_states = encoder_hidden_states.to(device)
                    
                # 生成噪声和时间步长 (确保在同一设备上)
                latents = torch.randn(
                    (encoder_hidden_states.shape[0], 4, args.resolution // 8, args.resolution // 8),
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                    device=device,  # 明确指定设备
                )
                
                # 设置时间步长
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (encoder_hidden_states.shape[0],),
                    device=device,  # 明确指定设备
                ).long()
                
                # 添加噪声
                noise = torch.randn_like(latents, device=device)  # 明确指定设备
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 添加数据格式检查
                log(f"噪声潜在空间形状: {noisy_latents.shape}")
                log(f"时间步形状: {timesteps.shape}")
                log(f"编码器隐藏状态形状: {encoder_hidden_states.shape}")
                
                # 使用教师UNet预测噪声前进行检查
                if noisy_latents.dim() != 4:
                    log(f"警告: 噪声潜在空间维度错误 ({noisy_latents.dim()} != 4)，尝试修复...")
                    if noisy_latents.dim() == 3:
                        noisy_latents = noisy_latents.unsqueeze(0)
                    else:
                        log(f"无法修复噪声潜在空间维度，跳过此批次")
                        continue
                
                if encoder_hidden_states.dim() != 3:
                    log(f"警告: 编码器隐藏状态维度错误 ({encoder_hidden_states.dim()} != 3)，尝试修复...")
                    if encoder_hidden_states.dim() == 2:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    elif encoder_hidden_states.dim() == 4:
                        encoder_hidden_states = encoder_hidden_states.squeeze(1)
                    else:
                        log(f"无法修复编码器隐藏状态维度，跳过此批次")
                        continue
                
                # 使用教师UNet预测噪声
                with torch.no_grad():
                    try:
                        teacher_noise_pred = teacher_unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states
                        ).sample
                    except ValueError as e:
                        if "too many values to unpack" in str(e):
                            log(f"教师UNet执行错误: {str(e)}，这可能是编码器隐藏状态形状问题")
                            # 尝试特定的修复方法
                            try:
                                # 重新调整编码器隐藏状态的形状
                                if encoder_hidden_states.dim() == 3:
                                    batch_size, seq_len, dim = encoder_hidden_states.shape
                                    encoder_hidden_states = encoder_hidden_states.view(batch_size, seq_len, dim)
                                    teacher_noise_pred = teacher_unet(
                                        noisy_latents, 
                                        timesteps, 
                                        encoder_hidden_states
                                    ).sample
                                else:
                                    log("无法修复，跳过此批次")
                                    continue
                            except Exception as inner_e:
                                log(f"特定修复也失败: {str(inner_e)}")
                                continue
                        else:
                            log(f"教师UNet执行发生其他错误: {str(e)}")
                            continue
                    except Exception as e:
                        log(f"教师UNet执行错误: {str(e)}")
                        continue
                
                # 使用学生UNet预测噪声
                try:
                    student_noise_pred = unet(
                        noisy_latents, 
                        timesteps, 
                        encoder_hidden_states
                    ).sample
                except Exception as e:
                    log(f"学生UNet执行错误: {str(e)}")
                    continue
                
                # 计算MSE损失
                loss = torch.nn.functional.mse_loss(student_noise_pred, teacher_noise_pred, reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_lora_params, args.max_grad_norm)
                
                # 更新参数
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 每N步保存一次模型
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # 创建权重状态字典
                        lora_state_dict = {}
                        
                        # 从学生UNet中提取LoRA权重
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        for name, param in unwrapped_unet.named_parameters():
                            if "lora_" in name and param.requires_grad:
                                lora_state_dict[name] = param.detach().cpu()
                        
                        # 检查是否找到了LoRA权重
                        if len(lora_state_dict) == 0:
                            log(f"警告: 步骤 {global_step} 没有找到LoRA权重，尝试使用 extract_lora_ups_down 函数")
                            try:
                                lora_state_dict = extract_lora_ups_down(unwrapped_unet, to_cpu=True)
                            except Exception as e:
                                log(f"提取LoRA权重失败: {str(e)}")
                        
                        # 保存LoRA权重
                        if len(lora_state_dict) > 0:
                            save_path = os.path.join(args.output_dir, f"step_{global_step}.safetensors")
                            save_file(lora_state_dict, save_path)
                            log(f"在步骤 {global_step} 保存LoRA权重到 {save_path}")
                        else:
                            log(f"警告: 步骤 {global_step} 未找到可保存的LoRA权重")
                
                if global_step >= args.max_train_steps:
                    break
    
    # 保存最终模型
    if accelerator.is_main_process:
        # 创建权重状态字典
        final_lora_state_dict = {}
        
        # 从学生UNet中提取LoRA权重
        unwrapped_unet = accelerator.unwrap_model(unet)
        for name, param in unwrapped_unet.named_parameters():
            if "lora_" in name and param.requires_grad:
                final_lora_state_dict[name] = param.detach().cpu()
        
        # 检查是否找到了LoRA权重
        if len(final_lora_state_dict) == 0:
            log("警告: 没有找到LoRA权重，尝试使用 extract_lora_ups_down 函数")
            try:
                final_lora_state_dict = extract_lora_ups_down(unwrapped_unet, to_cpu=True)
            except Exception as e:
                log(f"提取LoRA权重失败: {str(e)}")
        
        # 保存最终LoRA权重
        if len(final_lora_state_dict) > 0:
            final_save_path = os.path.join(args.output_dir, "distilled_lora.safetensors")
            save_file(final_lora_state_dict, final_save_path)
            log(f"保存最终蒸馏LoRA权重到 {final_save_path}")
        else:
            log("错误: 无法保存最终LoRA权重，未找到权重参数")
            return
    
    # 测试最终模型
    if args.test_prompts and accelerator.is_main_process:
        log("开始测试最终模型...")
        
        # 加载整个管道用于测试
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16 if args.mixed_precision else torch.float32
        ).to(device)
        
        # 应用蒸馏后的LoRA
        try:
            patch_pipe(
                pipeline,
                os.path.join(args.output_dir, "distilled_lora.safetensors"),
                patch_text=False,
                patch_ti=False,
                patch_unet=True,
            )
        except TypeError as e:
            log(f"使用标准 patch_pipe 失败: {str(e)}")
            # 尝试简化版本的 patch_pipe 调用
            try:
                patch_pipe(
                    pipeline,
                    os.path.join(args.output_dir, "distilled_lora.safetensors"),
                )
            except Exception as e2:
                log(f"简化 patch_pipe 也失败: {str(e2)}")
                log("无法加载蒸馏LoRA进行测试")
                return
        
        try:
            tune_lora_scale(pipeline.unet, args.lora_scale)
        except Exception as e:
            log(f"调整LoRA缩放失败: {str(e)}")
        
        # 生成每个测试提示的图像
        os.makedirs(os.path.join(args.output_dir, "test_samples"), exist_ok=True)
        
        for i, test_prompt in enumerate(args.test_prompts):
            log(f"生成测试图像 {i+1}/{len(args.test_prompts)}: '{test_prompt}'")
            
            # 生成图像
            with torch.autocast("cuda", enabled=args.mixed_precision):  # 修改这里，使用正确的设备
                image = pipeline(
                    test_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            
            # 保存图像
            image_path = os.path.join(args.output_dir, "test_samples", f"sample_{i+1}.png")
            image.save(image_path)
            log(f"保存图像到 {image_path}")
        
        log("测试完成!")
    
    log("训练完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA模型蒸馏训练")
    
    # 模型路径参数
    parser.add_argument("--base_model_path", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base", 
                        help="基础Stable Diffusion模型路径")
    parser.add_argument("--output_dir", type=str, default="/home/s414e2/CJH/Text-to-3D/LucidDreamer/distilled_lora", 
                        help="输出目录，用于保存蒸馏的LoRA权重")
    parser.add_argument("--teacher_loras", type=str, nargs="+", required=True,
                        help="教师LoRA权重路径和关联概念，格式为'路径:概念'，例如'/path/to/lora.safetensors:screw'")
    
    # 训练参数
    parser.add_argument("--resolution", type=int, default=512,
                        help="训练分辨率")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--max_train_steps", type=int, default=3000,
                        help="最大训练步数")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="学习率预热步数")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam优化器beta1参数")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam优化器beta2参数")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Adam优化器权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam优化器epsilon参数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪范数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=float, default=16,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA缩放系数")
    parser.add_argument("--unet_target_modules", type=str, default="Transformer,Attention,CrossAttention,Transformer2DModel",
                        help="要应用LoRA的UNet模块，以逗号分隔")
    
    # 提示模板和测试参数
    parser.add_argument("--prompt_templates", type=str, nargs="+", 
                        default=[
                            "a photo of <concept>",
                            "a rendering of <concept>",
                            "a cropped photo of <concept>",
                            "a close-up photo of <concept>",
                            "a photo of a <concept> on a white background",
                            "<concept> on a solid black background",
                            "<concept> on a solid white background",
                            "a photo of <concept> in the style of professional product photography"
                        ],
                        help="提示模板列表，用<concept>代替教师LoRA中的概念")
    parser.add_argument("--test_prompts", type=str, nargs="+",
                        default=[
                            "a photo of screw on a white background",
                            "a photo of nut on a white background",
                            "a photo of nail on a white background",
                            "a photo of gasket on a white background"
                        ],
                        help="训练后用于测试的提示")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每多少步保存一次模型")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="是否使用混合精度训练")
    
    args = parser.parse_args()
    
    train_distilled_lora(args)