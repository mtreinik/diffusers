#!/bin/bash

if [ -z "$MODEL_NAME" ] || [ ! -d "$OUTPUT_DIR" ] || [ -z "$INSTANCE_DATA_DIR" ]  || [ -z "$INSTANCE_PROMPT" ] || [ -z "$CLASS_DIR" ] || [ -z "$CLASS_PROMPT" ]; then
  echo Please define the following environment variables correctly: \nMODEL_NAME, OUTPUT_DIR, INSTANCE_DATA_DIR, INSTANCE_PROMPT, CLASS_DIR, CLASS_PROMPT 
  exit 1
fi

accelerate launch train_multi_subject_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir="$INSTANCE_DATA_DIR" \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="$INSTANCE_PROMPT" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="$CLASS_DIR" \
  --class_prompt="$CLASS_PROMPT"\
  --num_class_images=200 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800

