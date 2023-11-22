cog predict \
    -i run_name="faces_sweep_000_395" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_001_395" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_002_396" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_003_396" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_004_396" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_005_396" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_006_399" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_007_399" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_008_400" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_009_401" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_010_406" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_011_416" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_012_416" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_013_468" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_014_495" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_015_501" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_016_516" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_017_640" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_018_662" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_019_679" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="700" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="2100" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

