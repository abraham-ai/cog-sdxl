cog predict \
    -i run_name="xander_SD15" -i caption_prefix="" \
    -i concept_mode="face" \
    -i train_batch_size="4" \
    -i sd_model_version="sd15" \
    -i checkpointing_steps="300" \
    -i max_train_steps="900" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i seed="0"
