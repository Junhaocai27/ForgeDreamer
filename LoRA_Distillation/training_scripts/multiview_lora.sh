#!/bin/bash

export MODEL_NAME="/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora_train_img/screw_new_fau"
export OUTPUT_DIR="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex"

# 创建输出目录结构
mkdir -p ${OUTPUT_DIR}/front ${OUTPUT_DIR}/up

# 定义训练函数
train_lora() {
    local view=$1
    local placeholder=$2
    local cuda_device=$3
    
    echo "========================================================"
    echo "开始训练 ${view} 视角，使用占位符：${placeholder}，在 GPU ${cuda_device}"
    echo "========================================================"
    
    lora_pti \
      --pretrained_model_name_or_path=${MODEL_NAME} \
      --instance_data_dir=${INSTANCE_DIR}/${view} \
      --output_dir=${OUTPUT_DIR}/${view} \
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
      --placeholder_tokens="${placeholder}" \
      --use_template="object" \
      --save_steps=100 \
      --max_train_steps_ti=1000 \
      --max_train_steps_tuning=1000 \
      --perform_inversion=True \
      --clip_ti_decay \
      --weight_decay_ti=0.000 \
      --weight_decay_lora=0.001 \
      --continue_inversion \
      --continue_inversion_lr=1e-4 \
      --device="cuda:${cuda_device}" \
      --lora_rank=16 \
      --lora_clip_target_modules="{'CLIPSdpaAttention'}" > ${OUTPUT_DIR}/${view}_training.log 2>&1
      
    echo "${view} 视角训练完成"
}

# 并行启动两个训练任务
train_lora "front" "<screw_front>" 1 &
FRONT_PID=$!

train_lora "up" "<screw_up>" 2 &
UP_PID=$!

# 等待两个训练任务完成
echo "正在等待前视角和上视角训练完成..."
wait $FRONT_PID
echo "前视角训练已完成"
wait $UP_PID
echo "上视角训练已完成"

# 合并模型
echo "========================================================"
echo "开始合并前视角和上视角LoRA模型..."
echo "========================================================"

lora_add \
  "${OUTPUT_DIR}/front/step_inv_1000.safetensors" \
  "${OUTPUT_DIR}/up/step_inv_1000.safetensors" \
  "${OUTPUT_DIR}/front_up_combined.safetensors" \
  --alpha_1 1 \
  --alpha_2 1 \

echo "========================================================"
echo "前视角和上视角LoRA训练和合并完成！"
echo "单独模型位于: ${OUTPUT_DIR}/front/step_inv_1000.safetensors 和 ${OUTPUT_DIR}/up/step_inv_1000.safetensors"
echo "合并模型位于: ${OUTPUT_DIR}/front_up_combined.safetensors"
echo "训练日志位于: ${OUTPUT_DIR}/front_training.log 和 ${OUTPUT_DIR}/up_training.log"
echo "========================================================"