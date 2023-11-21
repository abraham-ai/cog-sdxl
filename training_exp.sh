cog predict \
    -i run_name="steel_no_hard_pivot" \
    -i checkpointing_steps="30" \
    -i debug="True" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="60" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i hard_pivot="False" \
    -i seed="0"

cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="30" \
    -i debug="True" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="60" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="30" \
    -i debug="True" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="60" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i train_batch_size=3 \
    -i resolution=960 \
    -i seed="0"