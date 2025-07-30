#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora_train_img"
export OUTPUT_DIR="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/cable_gland"

python train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --placeholder_token="<a cable gland with a hole in the surface>" \
  --learnable_property="object"\
  --initializer_token="connector" \
  --save_steps=500 \
  --unfreeze_lora_step=1500 \
  --stochastic_attribute="3d render,4k,highres" # these attributes will be randomly appended to the prompts
  