cog predict \
    -i run_name="faces_sweep_000_8815" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_001_8815" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_002_8815" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_003_8816" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_004_8816" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_005_8817" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_006_8818" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_007_8819" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_008_8821" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_009_8827" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_010_8831" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_011_8831" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_012_8832" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_013_8848" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_014_8863" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="False" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.3" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="1e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_015_8864" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="4" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_016_8974" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="16" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip" \
    -i lora_weight_decay="0.005" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.9" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-3" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

cog predict \
    -i run_name="faces_sweep_017_9013" -i caption_prefix="" \
    -i mask_target_prompts="" \
    -i checkpointing_steps="20" \
    -i debug="True" \
    -i hard_pivot="True" \
    -i is_lora="True" \
    -i left_right_flip_augmentation="False" \
    -i lora_rank="8" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip" \
    -i lora_weight_decay="0.015" \
    -i max_train_steps="40" \
    -i mode="face" \
    -i prodigy_d_coef="0.1" \
    -i resolution=1024 \
    -i seed="0" \
    -i ti_lr="3e-4" \
    -i ti_weight_decay="1e-4" \
    -i train_batch_size="2"  

