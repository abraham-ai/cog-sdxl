cog predict \
    -i run_name="kwebbelkop1_0" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/kwebbelkop1.zip" \
    -i max_train_steps="700" \
    -i seed="0"

cog predict \
    -i run_name="kwebbelkop2_0" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/kwebbelkop2.zip" \
    -i max_train_steps="700" \
    -i seed="0"

cog predict \
    -i run_name="kwebbelkop1_1" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/kwebbelkop1.zip" \
    -i max_train_steps="700" \
    -i seed="1"

cog predict \
    -i run_name="kwebbelkop2_1" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/kwebbelkop2.zip" \
    -i max_train_steps="700" \
    -i seed="1"