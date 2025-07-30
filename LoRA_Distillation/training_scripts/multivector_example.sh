export MODEL_NAME="/root/stable-diffusion-2-1-base"
export INSTANCE_DIR="/root/lora_train_imgs/lora_yellow_light_diode_up"
export OUTPUT_DIR="/root/lora_weight_before_distill/yellow_light_diode_up_new1"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
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
  --placeholder_tokens="<yellow_light_diode_up>" \
  --use_template="object"\
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:3" \
  --lora_rank=16 \
  --lora_clip_target_modules="{'CLIPSdpaAttention'}" \
  # --log_wandb
#  --use_face_segmentation_condition\