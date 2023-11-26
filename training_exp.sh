
cog predict \
    -i run_name="plantoid_800_snr" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i checkpointing_steps="200" \
    -i concept_mode="object" \
    -i debug="True" \
    -i left_right_flip_augmentation="True" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plantoid_imgs_9.zip" \
    -i max_train_steps="500" \
    -i seed="0" \

cog predict \
    -i run_name="xander_uncropped_snr" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i checkpointing_steps="100" \
    -i concept_mode="face" \
    -i debug="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_uncropped.zip" \
    -i max_train_steps="500" \
    -i seed="0" \