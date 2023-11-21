
cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="1000" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="does" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip" \
    -i max_train_steps="3000" \
    -i concept_mode="style" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="banny" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_all.zip" \
    -i max_train_steps="1000" \
    -i concept_mode="object" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="kojii_all" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/koji_all.zip" \
    -i max_train_steps="1000" \
    -i concept_mode="object" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="xander" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i max_train_steps="1000" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="True" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="clipx_200_object" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="2000" \
    -i concept_mode="object" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="clipx_200_style" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="2000" \
    -i concept_mode="style" \
    -i resolution=960 \
    -i seed="0"


