cog predict \
    -i run_name="clipx" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="2000" \
    -i concept_mode="style" \
    -i seed="0"

cog predict \
    -i run_name="does" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip" \
    -i max_train_steps="2000" \
    -i concept_mode="style" \
    -i seed="0"