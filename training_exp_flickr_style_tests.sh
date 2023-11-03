
cog predict \
    -i run_name="steel" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i max_train_steps="1000" \
    -i mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="kojii_all" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/koji_all.zip" \
    -i max_train_steps="1000" \
    -i mode="concept" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="clipx_200_concept" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="2000" \
    -i mode="concept" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="clipx_200_style" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i max_train_steps="2000" \
    -i mode="style" \
    -i resolution=960 \
    -i seed="0"


cog predict \
    -i run_name="xander" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i max_train_steps="1000" \
    -i mode="face" \
    -i left_right_flip_augmentation="False" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="banny" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="6" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_all.zip" \
    -i max_train_steps="1000" \
    -i mode="concept" \
    -i resolution=960 \
    -i seed="0"

cog predict \
    -i run_name="flickr_style" \
    -i checkpointing_steps="500" \
    -i debug="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://minio.aws.abraham.fun/creations-stg/d6f8446d13a82bc159f4b26aadca90a888493e92cf0bab1e510cb5354fb065a7.zip|https://minio.aws.abraham.fun/creations-stg/991d70ba870022aef6c893b8335fee53ed9a32e8f998e23ec9dcf2adc0ee3f76.zip|https://minio.aws.abraham.fun/creations-stg/6b25015c2f655915c90c41fc35cc5f42f8a877307c2a8affc2d47ed812cf23c3.zip|https://minio.aws.abraham.fun/creations-stg/fbdc59246ee841bb8303787155a6a0c5cae56d7545a9bd0d5d077a9d8193baff.zip" \
    -i max_train_steps="3000" \
    -i mode="concept" \
    -i resolution=960 \
    -i seed="0"



