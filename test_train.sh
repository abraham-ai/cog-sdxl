cog predict \
    -i run_name="clipx" -i caption_prefix="" \
    -i concept_mode="object" \
    -i train_batch_size="4" \
    -i max_train_steps="200" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip" \
    -i seed="0"