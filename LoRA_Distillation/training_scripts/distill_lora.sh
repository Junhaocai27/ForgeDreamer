#!/bin/bash

# Multi-Teacher LoRA automatic distillation training script
# Usage: place all LoRA files to be distilled in a folder
# File name format: name.safetensors, corresponding placeholder token is <name>
# Supports any number of LoRA models for multi-teacher distillation

printf "%0.s=" {1..80} && echo
echo "Starting Multi-Teacher LoRA distillation training"
printf "%0.s=" {1..80} && echo
echo ""

# Check CUDA devices
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""

# Set environment variables
# export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(cd "$(dirname "$0")/../.."; pwd):$PYTHONPATH"

# Training configuration
LORA_MODELS_DIR="lora_weight_before_distill/distill_weight8"
BASE_MODEL="stable-diffusion-2-1-base"
OUTPUT_DIR="lora_train/multi_combine_$(date +%Y%m%d_%H%M%S)"
GENERATED_IMAGES_DIR="lora_train/pic"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GENERATED_IMAGES_DIR"

echo "Training configuration:"
echo "  LoRA model folder: $LORA_MODELS_DIR"
echo "  Base model: $BASE_MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo "  Generated images directory: $GENERATED_IMAGES_DIR"
echo ""

# Check if LoRA folder exists
if [ ! -d "$LORA_MODELS_DIR" ]; then
    echo "Error: LoRA model folder does not exist: $LORA_MODELS_DIR"
    exit 1
fi

# List discovered LoRA files
echo "Discovered LoRA files:"
find "$LORA_MODELS_DIR" -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" | sort
echo ""

# Run multi-teacher distillation training
python "$(dirname "$0")/../lora_diffusion/distill_lora.py" \
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

# Check training result
TRAIN_EXIT_CODE=$?

echo ""
printf "%0.s=" {1..80} && echo
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Multi-Teacher LoRA distillation training completed successfully!"
    printf "%0.s=" {1..80} && echo
    echo ""
    echo "Training results:"
    echo "  Final model saved to: $OUTPUT_DIR"
    echo "  Generated images saved to: $GENERATED_IMAGES_DIR"
    echo "  TensorBoard logs: $OUTPUT_DIR/tensorboard_logs"
    echo ""
    
    # List generated model files
    echo "Generated model files:"
    find "$OUTPUT_DIR" -name "*.safetensors" | sort
    echo ""
    
    # Show TensorBoard command
    echo "View training logs (TensorBoard):"
    echo "  tensorboard --logdir=$OUTPUT_DIR/tensorboard_logs --port=6006"
    echo ""
    
    # Show disk usage
    echo "Output directory disk usage:"
    du -sh "$OUTPUT_DIR"
    echo ""
    
else
    echo "✗ Training failed, exit code: $TRAIN_EXIT_CODE"
    printf "%0.s=" {1..80} && echo
    echo ""
    echo "Please check the error messages above"
    echo ""
fi

# Clean up temporary files (optional)
# echo "Cleaning up temporary files..."
# find /tmp -name "*torch*" -mtime +1 -delete 2>/dev/null || true

echo "Script execution complete"
printf "%0.s=" {1..80} && echo