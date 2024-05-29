cog predict \
    -i run_name="clipx" -i caption_prefix="" \
    -i concept_mode="style" \
    -i train_batch_size="4" \
    -i max_train_steps="40" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx.zip" \
    -i seed="0"