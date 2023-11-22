
cog predict \
    -i run_name="xander_2000_hard" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="5" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i max_train_steps="1500" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=1024 \
    -i hard_pivot="False"  \
    -i seed="0"

cog predict \
    -i run_name="xander_2000_soft" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="5" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i max_train_steps="1500" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=1024 \
    -i seed="0"

cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="5" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="1500" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=1024 \
    -i seed="0"

cog predict \
    -i run_name="steel_hard" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="5" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="1500" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=1024 \
    -i hard_pivot="True"  \
    -i seed="0"


cog predict \
    -i run_name="clipx_style" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="3000" \
    -i concept_mode="style" \
    -i resolution=1024 \
    -i seed="0"

cog predict \
    -i run_name="clipx_style_hard" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="3000" \
    -i concept_mode="style" \
    -i resolution=1024 \
    -i hard_pivot="True"  \
    -i seed="0"

cog predict \
    -i run_name="clipx_style_rank12" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="3000" \
    -i concept_mode="style" \
    -i resolution=1024 \
    -i seed="0"

cog predict \
    -i run_name="does_bs3_rank12" \
    -i checkpointing_steps="1000" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip" \
    -i max_train_steps="3000" \
    -i concept_mode="style" \
    -i train_batch_size=3 \
    -i resolution=1024 \
    -i seed="0"