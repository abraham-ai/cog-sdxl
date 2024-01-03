cog predict \
    -i run_name="bumba_captioned_sd15" -i caption_prefix="" \
    -i concept_mode="object" \
    -i train_batch_size="4" \
    -i sd_model_version="sd15" \
    -i checkpointing_steps="200" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/bumba_captioned.zip" \
    -i seed="0"

