
cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="400" \
    -i debug="True" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="1200" \
    -i mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i seed="0"
