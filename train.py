import os
import shutil
import tarfile
import json

from cog import BaseModel, Input, Path

from predict import SDXL_MODEL_CACHE, SDXL_URL, download_weights
from preprocess import preprocess
from trainer_pti import main

"""
Wrapper around actual trainer.
"""


class TrainingOutput(BaseModel):
    weights: Path


from typing import Tuple

import numpy as np
import torch
def pick_best_gpu_id():
    # pick the GPU with the most free memory:
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    if len(gpu_ids) < 2:
        return
    print(f"# of visible GPUs: {len(gpu_ids)}")
    gpu_mem = []
    for gpu_id in gpu_ids:
        free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
        gpu_mem.append(free_memory)
        print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))    
    best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
    # set this to be the active GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    print("Using GPU %d" %best_gpu_id)


def train(
    input_images: Path = Input(
        description="A .zip or .tar file containing the image files that will be used for fine-tuning"
    ),
    is_style: bool = Input(
        description="Whether the images represent a style to be learned (instead of a concept or face)",
        default=False,
    ),
    seed: int = Input(
        description="Random seed for reproducible training. Leave empty to use a random seed",
        default=None,
    ),
    resolution: int = Input(
        description="Square pixel resolution which your images will be resized to for training",
        default=960,
    ),
    train_batch_size: int = Input(
        description="Batch size (per device) for training",
        default=2,
    ),
    num_train_epochs: int = Input(
        description="Number of epochs to loop through your training dataset",
        default=10000,
    ),
    max_train_steps: int = Input(
        description="Number of individual training steps. Takes precedence over num_train_epochs",
        default=1000,
    ),
    checkpointing_steps: int = Input(
        description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
        default=250,
    ),
    # gradient_accumulation_steps: int = Input(
    #     description="Number of training steps to accumulate before a backward pass. Effective batch size = gradient_accumulation_steps * batch_size",
    #     default=1,
    # ), # todo.
    is_lora: bool = Input(
        description="Whether to use LoRA training. If set to False, will use Full fine tuning",
        default=True,
    ),
    unet_learning_rate: float = Input(
        description="Learning rate for the U-Net. We recommend this value to be somewhere between `1e-6` to `1e-5`.",
        default=1e-6,
    ),
    ti_lr: float = Input(
        description="Scaling of learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
        default=3e-4,
    ),
    lora_lr: float = Input(
        description="Scaling of learning rate for training LoRA embeddings. Don't alter unless you know what you're doing.",
        default=1e-4,
    ),

    ti_weight_decay: float = Input(
        description="weight decay for textual inversion embeddings. Don't alter unless you know what you're doing.",
        default=1e-4,
    ),
    lora_weight_decay: float = Input(
        description="weight decay for LoRa. Don't alter unless you know what you're doing.",
        default=1e-5,
    ),
    lora_rank: int = Input(
        description="Rank of LoRA embeddings. For faces 4 is good, for complex objects you might try 6 or 8",
        default=6,
    ),
    lr_scheduler: str = Input(
        description="Learning rate scheduler to use for training",
        default="constant",
        choices=[
            "constant",
            "linear",
        ],
    ),
    lr_warmup_steps: int = Input(
        description="Number of warmup steps for lr schedulers with warmups.",
        default=50,
    ),
    token_string: str = Input(
        description="A unique string that will be trained to refer to the concept in the input images. Can be anything, but TOK works well",
        default="TOK",
    ),
    # token_map: str = Input(
    #     description="String of token and their impact size specificing tokens used in the dataset. This will be in format of `token1:size1,token2:size2,...`.",
    #     default="TOK:2",
    # ),
    caption_prefix: str = Input(
        description="Text which will be used as prefix during automatic captioning. Must contain the `token_string`. For example, if caption text is 'a photo of TOK', automatic captioning will expand to 'a photo of TOK under a bridge', 'a photo of TOK holding a cup', etc.",
        default="a photo of TOK, ",
    ),
    mask_target_prompts: str = Input(
        description="Prompt that describes part of the image that you will find important. For example, if you are fine-tuning your pet, `photo of a dog` will be a good prompt. Prompt-based masking is used to focus the fine-tuning process on the important/salient parts of the image",
        default=None,
    ),
    crop_based_on_salience: bool = Input(
        description="If you want to crop the image to `target_size` based on the important parts of the image, set this to True. If you want to crop the image based on face detection, set this to False",
        default=True,
    ),
    use_face_detection_instead: bool = Input(
        description="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
        default=False,
    ),
    clipseg_temperature: float = Input(
        description="How blurry you want the CLIPSeg mask to be. We recommend this value be something between `0.5` to `1.0`. If you want to have more sharp mask (but thus more errorful), you can decrease this value.",
        default=1.0,
    ),
    left_right_flip_augmentation: bool = Input(
        description="Add left-right flipped version of each img to the training data, recommended for most cases. If you are learning a face, you prob want to disable this",
        default=True,
    ),
    verbose: bool = Input(description="verbose output", default=True),
    input_images_filetype: str = Input(
        description="Filetype of the input images. Can be either `zip` or `tar`. By default its `infer`, and it will be inferred from the ext of input file.",
        default="infer",
        choices=["zip", "tar", "infer"],
    ),
    run_name: str = Input(
        description="Subdirectory where all files will be saved",
        default="unnamed",
    ),
) -> TrainingOutput:

    pick_best_gpu_id()

    # Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
    token_map = token_string + ":2"

    # Process 'token_to_train' and 'input_data_tar_or_zip'
    inserting_list_tokens = token_map.split(",")

    token_dict = {}
    running_tok_cnt = 0
    all_token_lists = []
    for token in inserting_list_tokens:
        n_tok = int(token.split(":")[1])

        token_dict[token.split(":")[0]] = "".join(
            [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
        )
        all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])

        running_tok_cnt += n_tok

    if not os.path.exists(SDXL_MODEL_CACHE):
        download_weights(SDXL_URL, SDXL_MODEL_CACHE)

    output_dir = os.path.join("loras", run_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    input_dir, n_imgs, trigger_text, segmentation_prompt = preprocess(
        output_dir,
        input_images_filetype=input_images_filetype,
        input_zip_path=input_images,
        caption_text=caption_prefix,
        mask_target_prompts=mask_target_prompts,
        target_size=resolution,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=clipseg_temperature,
        substitution_tokens=list(token_dict.keys()),
        left_right_flip_augmentation=left_right_flip_augmentation
    )

    # Make a dict of all the arguments and save it to args.json:
    args_dict = {
        "input_images": str(input_images),
        "num_training_images": n_imgs,
        "seed": seed,
        "resolution": resolution,
        "train_batch_size": train_batch_size,
        "num_train_epochs": num_train_epochs,
        "max_train_steps": max_train_steps,
        "is_lora": is_lora,
        "unet_learning_rate": unet_learning_rate,
        "ti_lr": ti_lr,
        "lora_lr": lora_lr,
        "ti_weight_decay": ti_weight_decay,
        "lora_weight_decay": lora_weight_decay,
        "lora_rank": lora_rank,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "token_string": token_string,
        "trigger_text": trigger_text,
        "segmentation_prompt": segmentation_prompt,
        "crop_based_on_salience": crop_based_on_salience,
        "use_face_detection_instead": use_face_detection_instead,
        "clipseg_temperature": clipseg_temperature,
        "left_right_flip_augmentation": left_right_flip_augmentation,
        "checkpointing_steps": checkpointing_steps,
        "run_name": run_name,
    }

    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    output_save_dir = main(
        pretrained_model_name_or_path=SDXL_MODEL_CACHE,
        instance_data_dir=os.path.join(input_dir, "captions.csv"),
        output_dir=output_dir,
        seed=seed,
        resolution=resolution,
        train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=1,
        unet_learning_rate=unet_learning_rate,
        ti_lr=ti_lr,
        lora_lr=lora_lr,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        token_dict=token_dict,
        inserting_list_tokens=all_token_lists,
        verbose=verbose,
        checkpointing_steps=checkpointing_steps,
        scale_lr=False,
        max_grad_norm=1.0,
        allow_tf32=True,
        mixed_precision="bf16",
        device="cuda:0",
        lora_rank=lora_rank,
        is_lora=is_lora,
        args_dict=args_dict,
    )

    thumbnail_grid_path = os.path.join(output_save_dir, "validation_grid.jpg")
    out_path = "trained_model.tar"
    directory = Path(output_save_dir)

    with tarfile.open(out_path, "w") as tar:
        for file_path in directory.rglob("*"):
            print(file_path)
            arcname = file_path.relative_to(directory)
            tar.add(file_path, arcname=arcname)

    

    return TrainingOutput(weights=Path(out_path))
