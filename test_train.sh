cog predict \
    -i run_name="gene_token_sparse" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i augment_imgs_up_to_n="20" \
    -i concept_mode="face" \
    -i caption_prefix="TOK" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="12" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/gene5.zip" \
    -i max_train_steps="600" \
    -i checkpointing_steps="100" \
    -i seed="0"