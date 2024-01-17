cog predict \
    -i run_name="plankton_style" -i caption_prefix="" \
    -i concept_mode="style" \
    -i train_batch_size="4" \
    -i sd_model_version="sdxl" \
    -i max_train_steps="40" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plankton_style.zip" \
    -i seed="0"

cog predict \
    -i run_name="plankton_object" -i caption_prefix="" \
    -i concept_mode="object" \
    -i train_batch_size="4" \
    -i sd_model_version="sdxl" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plankton_object.zip" \
    -i seed="0"