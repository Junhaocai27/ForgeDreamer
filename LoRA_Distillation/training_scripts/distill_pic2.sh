#!/bin/bash

# 多Teacher LoRA自动蒸馏训练脚本
# 使用方法: 将所有要蒸馏的LoRA文件放在一个文件夹中
# 文件名格式: name.safetensors，对应的placeholder token为 <name>
# 支持任意数量的LoRA模型进行多teacher蒸馏

echo "="*80
echo "开始多Teacher LoRA蒸馏训练"
echo "="*80
echo ""

# 检查CUDA设备
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="/root/lora:$PYTHONPATH"

# 训练配置
LORA_MODELS_DIR="/root/lora_weight_before_distill/distill_weight8"
BASE_MODEL="/root/stable-diffusion-2-1-base"
OUTPUT_DIR="/root/lora_train/multi_combine_$(date +%Y%m%d_%H%M%S)"
GENERATED_IMAGES_DIR="/root/lora_train/pic"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GENERATED_IMAGES_DIR"

echo "训练配置:"
echo "  LoRA模型文件夹: $LORA_MODELS_DIR"
echo "  基础模型: $BASE_MODEL"
echo "  输出目录: $OUTPUT_DIR"
echo "  生成图片目录: $GENERATED_IMAGES_DIR"
echo ""

# 检查LoRA文件夹是否存在
if [ ! -d "$LORA_MODELS_DIR" ]; then
    echo "错误: LoRA模型文件夹不存在: $LORA_MODELS_DIR"
    exit 1
fi

# 列出发现的LoRA文件
echo "发现的LoRA文件:"
find "$LORA_MODELS_DIR" -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" | sort
echo ""

# 运行多teacher蒸馏训练
python /root/ForgeDreamer_FX/ForgeDreamer/LoRA_Distillation/lora_diffusion/distill_multi_lora_pic_good_result4.py \
  --lora_models_dir="$LORA_MODELS_DIR" \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --output_dir="$OUTPUT_DIR" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --lr_scheduler_lora="linear" \
  --lr_warmup_steps_lora=100 \
  --use_template="object" \
  --save_steps=100 \
  --max_train_steps_ti=5000 \
  --max_train_steps_tuning=5000 \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001 \
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=16 \
  --perform_inversion \
  --feature_align_weight=0.01 \
  --noise_pred_weight=1.0 \
  --teacher_selection_strategy="round_robin" \
  --auto_generate_placeholder_tokens \
  --out_name="multi_teacher_distilled" \
  --tensorboard_log_dir="$OUTPUT_DIR/tensorboard_logs" \
  --unet_feature_align_weight=0.01 \
  --text_encoder_feature_align_weight=0.02 \
  --mixed_precision="no" \
  --seed=42

# 检查训练结果
TRAIN_EXIT_CODE=$?

echo ""
echo "="*80
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 多Teacher LoRA蒸馏训练成功完成!"
    echo "="*80
    echo ""
    echo "训练结果:"
    echo "  最终模型保存到: $OUTPUT_DIR"
    echo "  生成的图片保存到: $GENERATED_IMAGES_DIR"
    echo "  TensorBoard日志: $OUTPUT_DIR/tensorboard_logs"
    echo ""
    
    # 列出生成的模型文件
    echo "生成的模型文件:"
    find "$OUTPUT_DIR" -name "*.safetensors" | sort
    echo ""
    
    # 显示TensorBoard命令
    echo "查看训练日志 (TensorBoard):"
    echo "  tensorboard --logdir=$OUTPUT_DIR/tensorboard_logs --port=6006"
    echo ""
    
    # 显示磁盘使用情况
    echo "输出目录磁盘使用:"
    du -sh "$OUTPUT_DIR"
    echo ""
    
else
    echo "✗ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "="*80
    echo ""
    echo "请检查上面的错误信息"
    echo ""
fi

# 清理临时文件（可选）
# echo "清理临时文件..."
# find /tmp -name "*torch*" -mtime +1 -delete 2>/dev/null || true

echo "脚本执行完成"
echo "="*80