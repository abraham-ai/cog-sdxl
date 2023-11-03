
cog predict \
    -i run_name="does_rank6_960_style" \
    -i lora_training_urls="https://genekogan.com/DOES.zip" \
    -i resolution=960 \
    -i mode=style \
    -i seed=0 \
    -i is_lora=True \
    -i checkpointing_steps=500 \
    -i max_train_steps=3000 \
    -i lora_rank=6 \
    -i hard_pivot=True \
    -i debug=True

cog predict \
    -i run_name="does_rank12_960_style" \
    -i lora_training_urls="https://genekogan.com/DOES.zip" \
    -i resolution=960 \
    -i mode=style \
    -i seed=0 \
    -i is_lora=True \
    -i checkpointing_steps=500 \
    -i max_train_steps=3000 \
    -i lora_rank=12 \
    -i hard_pivot=True \
    -i debug=True

cog predict \
    -i run_name="does_rank12_768_style" \
    -i lora_training_urls="https://genekogan.com/DOES.zip" \
    -i resolution=768 \
    -i mode=style \
    -i seed=0 \
    -i is_lora=True \
    -i checkpointing_steps=500 \
    -i max_train_steps=3000 \
    -i lora_rank=12 \
    -i hard_pivot=True \
    -i debug=True

cog predict \
    -i run_name="does_rank12_768_style_bs_3" \
    -i lora_training_urls="https://genekogan.com/DOES.zip" \
    -i resolution=768 \
    -i mode=style \
    -i train_batch_size=3 \
    -i seed=0 \
    -i is_lora=True \
    -i checkpointing_steps=500 \
    -i max_train_steps=3000 \
    -i lora_rank=12 \
    -i hard_pivot=True \
    -i debug=True


cog predict \
    -i run_name="does_rank12_960_concept" \
    -i lora_training_urls="https://genekogan.com/DOES.zip" \
    -i resolution=960 \
    -i mode=concept \
    -i seed=0 \
    -i is_lora=True \
    -i checkpointing_steps=500 \
    -i max_train_steps=3000 \
    -i lora_rank=12 \
    -i hard_pivot=True \
    -i debug=True