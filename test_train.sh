cog predict \
    -i run_name="gene_token_sparse" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i caption_prefix="TOK" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/gene5.zip" \
    -i max_train_steps="1000" \
    -i checkpointing_steps="50" \
    -i off_ratio_power="0.1" \
    -i prodigy_d_coef="0.1" \
    -i l1_penalty="0.5" \
    -i resolution=1024 \
    -i seed="0" \
    -i train_batch_size="3"

cog predict \
    -i run_name="mira_token_sparse" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i caption_prefix="TOK" \
    -i concept_mode="face" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/mira.zip" \
    -i max_train_steps="1000" \
    -i off_ratio_power="0.1" \
    -i prodigy_d_coef="0.1" \
    -i l1_penalty="0.5" \
    -i resolution=1024 \
    -i seed="0" \
    -i train_batch_size="4"

cog predict \
    -i run_name="xander_token_sparse" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i caption_prefix="TOK" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i max_train_steps="1000" \
    -i off_ratio_power="0.1" \
    -i prodigy_d_coef="0.1" \
    -i l1_penalty="0.5" \
    -i resolution=1024 \
    -i seed="0" \
    -i train_batch_size="4"

