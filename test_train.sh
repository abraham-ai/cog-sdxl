cog predict \
    -i run_name="bumba_captioned" -i caption_prefix="" \
    -i concept_mode="object" \
    -i debug="True" \
    -i checkpointing_steps="800" \
    -i max_train_steps="600" \
    -i lora_param_scaler="0.25" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/bumba_captioned.zip" \
    -i seed="0"

