import os
import shutil
import tarfile
import json
import time

from cog import BasePredictor, BaseModel, File, Input, Path
from dotenv import load_dotenv
from predict_old import SDXL_MODEL_CACHE, SDXL_URL, download_weights
from preprocess import preprocess
from trainer_pti import main
from typing import Iterator, Optional

DEBUG_MODE = False

load_dotenv()

"""

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"

"""

class CogOutput(BaseModel):
    files: list[Path]
    name: Optional[str] = None
    thumbnails: Optional[list[Path]] = []
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

    def setup(self):
        print("cog:setup")

    def predict(
        self, 
        name: str = Input(
            description="Name of new LORA concept",
            default=None
        ),
        lora_training_urls: str = Input(
            description="Training images for new LORA concept (can be image urls or a .zip file of images)", 
            default=None
        ),
        mode: str = Input(
            description=" 'face' / 'style' / 'concept' (default)",
            default="concept",
        ),
        checkpoint: str = Input(
            description="Which Stable Diffusion checkpoint to use",
            choices=checkpoint_options.keys(),
            default="sdxl-v1.0"
        ),
        seed: int = Input(
            description="Random seed for reproducible training. Leave empty to use a random seed",
            default=None,
        ),
        resolution: int = Input(
            description="Square pixel resolution which your images will be resized to for training recommended [768-1024]",
            default=896,
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
            default=800,
        ),
        checkpointing_steps: int = Input(
            description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
            default=10000,
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
            description="Learning rate for the U-Net (only used for full finetuning, not for LORA's). Recommended between `1e-6` to `1e-5`.",
            default=1e-6,
        ),
        ti_lr: float = Input(
            description="Learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=3e-4,
        ),
        lora_lr: float = Input(
            description="Learning rate for training LoRA matrices. Don't alter unless you know what you're doing.",
            default=1e-4,
        ),
        ti_weight_decay: float = Input(
            description="weight decay for textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=1e-4,
        ),
        lora_weight_decay: float = Input(
            description="weight decay for LoRa. Don't alter unless you know what you're doing.",
            default=1e-4,
        ),
        lora_rank: int = Input(
            description="Rank of LoRA embeddings. For faces 4 is good, for complex concepts you can try 6 or 8",
            default=4,
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
        # token_map: str = Input(
        #     description="String of token and their impact size specificing tokens used in the dataset. This will be in format of `token1:size1,token2:size2,...`.",
        #     default="TOK:2",
        # ),
        caption_prefix: str = Input(
            description="Prefix text prepended to automatic captioning. Must contain the 'TOK'. Example is 'a photo of TOK, '.  If empty, chatgpt will take care of this automatically",
            default="",
        ),
        left_right_flip_augmentation: bool = Input(
            description="Add left-right flipped version of each img to the training data, recommended for most cases. If you are learning a face, you prob want to disable this",
            default=True,
        ),
        mask_target_prompts: str = Input(
            description="Prompt that describes most important part of the image, will be used for CLIP-segmentation. For example, if you are learning a person 'face' would be a good segmentation prompt",
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
        verbose: bool = Input(description="verbose output", default=True),
        run_name: str = Input(
            description="Subdirectory where all files will be saved",
            default=str(int(time.time())),
        ),
        debug: bool = Input(
            description="for debugging locally only (dont activate this on replicate)",
            default=False,
        ),
        hard_pivot: bool = Input(
            description="Use hard freeze for ti_lr. If set to False, will use soft transition of learning rates",
            default=True,
        ),
    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:

        if checkpoint != "sdxl-v1.0":
            raise ValueError("Only sdxl-v1.0 is supported for now")

        if mode == "face":
            mask_target_prompts = "face"

        print("cog:predict:train_lora")

        # Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
        token_string = "TOK"
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

        output_dir = os.path.join("lora_models_fin", run_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print(lora_training_urls)

        input_dir, n_imgs, trigger_text, segmentation_prompt, captions = preprocess(
            output_dir,
            mode,
            input_zip_path=lora_training_urls,
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
            "mode": mode,
            "input_images": str(lora_training_urls),
            "trainig_captions": captions,
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
            "hard_pivot": hard_pivot,
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
            ti_weight_decay=ti_weight_decay,
            lora_weight_decay=lora_weight_decay,
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
            debug=debug,
            hard_pivot=hard_pivot,
        )

        validation_grid_img_path = os.path.join(output_save_dir, "validation_grid.jpg")
        out_path = "trained_model.tar"
        directory = Path(output_save_dir)

        with tarfile.open(out_path, "w") as tar:
            print("Adding files to tar...")
            for file_path in directory.rglob("*"):
                print(file_path)
                arcname = file_path.relative_to(directory)
                tar.add(file_path, arcname=arcname)

        attributes = args_dict

        print("LORA training finished!")
        print(f"Returning {out_path}")

        if DEBUG_MODE or debug:
            yield Path(out_path)
        else:
            yield CogOutput(files=[Path(out_path)], name=name, thumbnails=[Path(validation_grid_img_path)], attributes=args_dict, isFinal=True, progress=1.0)