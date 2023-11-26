

cog predict \
    -i run_name="banny_snr"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip" \
    -i concept_mode="object" \
    -i left_right_flip_augmentation="True" \
    -i snr_gamma="5.0" \
    -i max_train_steps="500" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="banny"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip" \
    -i concept_mode="object" \
    -i left_right_flip_augmentation="True" \
    -i max_train_steps="1000" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="mira_snr"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/mira.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i snr_gamma="5.0" \
    -i max_train_steps="500" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="mira"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/mira.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i max_train_steps="1000" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"


cog predict \
    -i run_name="don_snr"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/don.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i snr_gamma="5.0" \
    -i max_train_steps="500" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="don"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/don.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i max_train_steps="1000" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="xander_snr"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i snr_gamma="5.0" \
    -i max_train_steps="500" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="xander"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip" \
    -i concept_mode="face" \
    -i left_right_flip_augmentation="False" \
    -i max_train_steps="1000" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="clipx_snr"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i concept_mode="style" \
    -i left_right_flip_augmentation="True" \
    -i snr_gamma="5.0" \
    -i max_train_steps="800" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"

cog predict \
    -i run_name="clipx"  \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip" \
    -i concept_mode="style" \
    -i left_right_flip_augmentation="True" \
    -i max_train_steps="1600" \
    -i lora_rank="12" \
    -i augment_imgs_up_to_n="20" \
    -i debug="True" \
    -i seed="0"





