#!/bin/bash

echo MODEL_NAME="$MODEL_NAME"
echo OUTPUT_DIR="$OUTPUT_DIR"
echo INSTANCE_DATA_DIR="$INSTANCE_DATA_DIR"
echo INSTANCE_PROMPT="$INSTANCE_PROMPT"
echo CLASS_DIR="$CLASS_DIR"
echo CLASS_PROMPT="$CLASS_PROMPT"
echo VALIDATION_PROMPT="$VALIDATION_PROMPT"

if [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$INSTANCE_DATA_DIR" ]  || [ -z "$INSTANCE_PROMPT" ] || [ -z "$CLASS_DIR" ] || [ -z "$CLASS_PROMPT" ] || [ -z "$VALIDATION_PROMPT" ]; then
  echo Please define the above environment variables correctly.
  exit 1
fi

accelerate launch train_dreambooth_sd3.py \
  --pretrained_model_name_or_path="$MODEL_NAME"  \
  --instance_data_dir="$INSTANCE_DATA_DIR" \
  --output_dir="$OUTPUT_DIR-$(date +"%Y%m%d_%H%M%S")" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_epochs=25 \
  --seed="0" \
  --with_prior_preservation \
  --class_data_dir="$CLASS_DIR" \
  --class_prompt="$CLASS_PROMPT"\
  --num_class_images=200 \
  --mixed_precision="fp16" \
  --push_to_hub

#  --optimizer="AdamW" \
#  --use_8bit_adam \
