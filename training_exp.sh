cog predict \
    -i run_name="clipx_200_style" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="10" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="4000" \
    -i concept_mode="style" \
    -i resolution=960 \
    -i seed="0"
