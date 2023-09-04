# Bootstrapped from Huggingface diffuser's code.
import fnmatch
import json
import math
import os
import random
import shutil
import numpy as np
from typing import List, Optional

import torch
import torch.utils.checkpoint
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm.auto import tqdm

from dataset_and_utils import (
    PreprocessedDataset,
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict
)

from io_utils import make_validation_img_grid, download_and_prep_training_data

from predict_old import SDXL_MODEL_CACHE
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

def patch_pipe_with_lora(pipe, lora_path):

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        lora_rank      = training_args["lora_rank"]
    
    unet = pipe.unet
    tensors = load_file(os.path.join(lora_path, "lora.safetensors"))
    unet_lora_attn_procs = {}

    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        module = LoRAAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=lora_rank,
        )
        unet_lora_attn_procs[name] = module.to("cuda")

    unet.set_attn_processor(unet_lora_attn_procs)
    unet.load_state_dict(tensors, strict=False)
    handler = TokenEmbeddingsHandler([pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2])
    handler.load_embeddings(os.path.join(lora_path, "embeddings.pti"))
    return pipe

def prepare_prompt_for_lora(prompt, lora_path, verbose = False):
    orig_prompt = prompt
    with open(os.path.join(lora_path, "special_params.json"), "r") as f:
        token_map = json.load(f)

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        trigger_text  = training_args["trigger_text"]
        mode          = training_args["mode"]

    if "<concept>" in prompt:
        prompt = prompt.replace("<concept>", trigger_text)
    else:
        prompt = trigger_text + " " + prompt

    for k, v in token_map.items():
        if k in prompt:
            prompt = prompt.replace(k, v)

    # fix some common mistakes:
    prompt = prompt.replace(",,", ",")
    prompt = prompt.replace("  ", " ")
    prompt = prompt.replace(" .", ".")
    prompt = prompt.replace(" ,", ",")
    
    if verbose:
        print('-------------------------')
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print('-- to:')
        print(prompt)
        print('-------------------------')

    return prompt

@torch.no_grad()
def render_images(lora_path, train_step, seed, lora_scale = 0.8, n_imgs = 4, device = "cuda:0"):
    validation_prompts = [
            'a statue of <concept>',
            'a masterful oil painting portraying <concept> with vibrant colors, brushstrokes and textures',
            'an origami paper sculpture of <concept>',
            'a vibrant low-poly artwork of <concept>, rendered in SVG, vector graphics',
            '<concept>, polaroid photograph',
            'a huge <concept> sand sculpture on a sunny beach, made of sand',
            '<concept> immortalized as an exquisite marble statue with masterful chiseling, swirling marble patterns and textures',
            'a colorful and dynamic <concept> mural sprawling across the side of a building in a city pulsing with life',
    ]

    random.seed(seed)
    validation_prompts = random.sample(validation_prompts, n_imgs)
    validation_prompts[0] = '<concept>'

    torch.cuda.empty_cache()

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_CACHE, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)
    pipeline = patch_pipe_with_lora(pipeline, lora_path)

    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path) for prompt in validation_prompts]
    generator = torch.Generator(device=device).manual_seed(0)
    pipeline_args = {
                "negative_prompt": "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft", 
                "num_inference_steps": 35,
                "guidance_scale": 7,
                }

    with torch.cuda.amp.autocast():
        for i in range(n_imgs):
            pipeline_args["prompt"] = validation_prompts[i]
            print(f"Rendering validation img with prompt: {validation_prompts[i]}")
            image = pipeline(**pipeline_args, generator=generator, cross_attention_kwargs = {"scale": lora_scale}).images[0]
            image.save(os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"), format="JPEG", quality=95)

    # create img_grid:
    img_grid_path = make_validation_img_grid(lora_path)

    del pipeline
    torch.cuda.empty_cache()

def save(output_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_param_to_optimize_names, make_images = True):
    """
    Save the LORA model to output_dir, optionally with some example images

    """
    print(f"Saving checkpoint at step.. {global_step}")
    os.makedirs(output_dir, exist_ok=True)

    if not is_lora:
        tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize_names
        }
        save_file(tensors, f"{output_dir}/unet.safetensors",)

    else:
        lora_tensors = unet_attn_processors_state_dict(unet)
        save_file(lora_tensors, f"{output_dir}/lora.safetensors")

    embedding_handler.save_embeddings(f"{output_dir}/embeddings.pti",)

    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)
    with open(f"{output_dir}/training_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    if make_images:
        render_images(f"{output_dir}", global_step, seed)



def main(
    pretrained_model_name_or_path: Optional[
        str
    ] = "./cache",  # "stabilityai/stable-diffusion-xl-base-1.0",
    revision: Optional[str] = None,
    instance_data_dir: Optional[str] = "./dataset/zeke/captions.csv",
    output_dir: str = "ft_masked_coke",
    seed: Optional[int] = random.randint(0, 2**32 - 1),
    resolution: int = 512,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 1,
    do_cache: bool = True,
    num_train_epochs: int = 600,
    max_train_steps: Optional[int] = None,
    checkpointing_steps: int = 500000,  # default to no checkpoints
    gradient_accumulation_steps: int = 1,  # todo
    unet_learning_rate: float = 1e-5,
    ti_lr: float = 3e-4,
    lora_lr: float = 1e-4,
    lora_weight_decay: float = 1e-4,
    ti_weight_decay: float = 0.0,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,  # todo with tests
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "bf16",
    device: str = "cuda:0",
    token_dict: dict = {"TOKEN": "<s0>"},
    inserting_list_tokens: List[str] = ["<s0>"],
    verbose: bool = True,
    is_lora: bool = True,
    lora_rank: int = 32,
    args_dict: dict = {},
    debug: bool = False,
    hard_pivot: bool = True,
) -> None:
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not seed:
        seed = np.random.randint(0, 2**32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if scale_lr:
        unet_learning_rate = (
            unet_learning_rate * gradient_accumulation_steps * train_batch_size
        )

    (
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(pretrained_model_name_or_path, revision, device, weight_dtype)

    print("# PTI : Loaded models")

    # Initialize new tokens for training.

    embedding_handler = TokenEmbeddingsHandler(
        [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
    )
    embedding_handler.initialize_new_tokens(inserting_toks=inserting_list_tokens)

    text_encoders = [text_encoder_one, text_encoder_two]

    unet_param_to_optimize = []
    # fine tune only attn weights

    text_encoder_parameters = []
    for text_encoder in text_encoders:
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                param.requires_grad = True
                text_encoder_parameters.append(param)
            else:
                param.requires_grad = False

    unet_param_to_optimize_names = []
    if not is_lora:
        WHITELIST_PATTERNS = [
            # "*.attn*.weight",
            # "*ff*.weight",
            "*"
        ]  # TODO : make this a parameter
        BLACKLIST_PATTERNS = ["*.norm*.weight", "*time*"]
        for name, param in unet.named_parameters():
            if any(
                fnmatch.fnmatch(name, pattern) for pattern in WHITELIST_PATTERNS
            ) and not any(
                fnmatch.fnmatch(name, pattern) for pattern in BLACKLIST_PATTERNS
            ):
                param.requires_grad_(True)
                unet_param_to_optimize_names.append(name)
                print(f"Training: {name}")
            else:
                param.requires_grad_(False)

        # Optimizer creation
        params_to_optimize = [
            {
                "params": unet_param_to_optimize,
                "lr": unet_learning_rate,
            },
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": weight_decay_ti,
            },
        ]

    else:
        
        # Do lora-training instead.
        unet.requires_grad_(False)
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            )
            unet_lora_attn_procs[name] = module
            module.to(device)
            unet_lora_parameters.extend(module.parameters())

        unet.set_attn_processor(unet_lora_attn_procs)

        print("Creating optimizer with:")
        print(f"lora_lr: {lora_lr}, lora_weight_decay: {lora_weight_decay}")
        print(f"ti_lr: {ti_lr}, ti_weight_decay: {ti_weight_decay}")

        params_to_optimize = [
            {
                "params": unet_lora_parameters,
                "lr": lora_lr,
                "weight_decay": lora_weight_decay,
            },
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": ti_weight_decay,
            },
        ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        weight_decay=0.0, # this wd doesn't matter, I think
    )

    print(f"# PTI : Loading dataset, do_cache {do_cache}")

    train_dataset = PreprocessedDataset(
        instance_data_dir,
        tokenizer_one,
        tokenizer_two,
        vae.float(),
        do_cache=True,
        substitute_caption_map=token_dict,
    )

    print("# PTI : Loaded dataset")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * gradient_accumulation_steps

    if verbose:
        print(f"# PTI :  Running training ")
        print(f"# PTI :  Num examples = {len(train_dataset)}")
        print(f"# PTI :  Num batches each epoch = {len(train_dataloader)}")
        print(f"# PTI :  Num Epochs = {num_train_epochs}")
        print(f"# PTI :  Instantaneous batch size per device = {train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"# PTI :  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"# PTI :  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    last_save_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), position=0, leave=True)
    checkpoint_dir = os.path.join(str(output_dir), "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(f"{checkpoint_dir}")

    # Experimental: warmup the token embeddings using CLIP-similarity:
    #embedding_handler.pre_optimize_token_embeddings(train_dataset)


    lr_ramp_power = 1.0

    import matplotlib.pyplot as plt

    def plot_torch_hist(parameters, epoch, checkpoint_dir, name, bins=100, min_val=-1, max_val=1, ymax_f = 0.75):

        # Initialize histogram
        hist_values = torch.zeros(bins).cuda()

        n_values = 0

        # Update histogram batch-wise to save memory
        for p in parameters:
            try:
                hist_values += torch.histc(p.data, bins=bins, min=min_val, max=max_val)
            except:
                hist_values += torch.histc(p.float(), bins=bins, min=min_val, max=max_val)
            n_values += p.numel()

        # Convert to NumPy and plot
        hist_values_cpu = hist_values.cpu().numpy()
        plt.figure()
        plt.hist([min_val + (max_val - min_val) * (i + 0.5) / bins for i in range(bins)], bins, weights=hist_values_cpu)
        plt.ylim(0, ymax_f * n_values)
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.title(f'Epoch {epoch} {name} Histogram')
        plt.savefig(f"{checkpoint_dir}/{name}_histogram_{epoch:04d}.png")
        plt.close()


    def plot_learning_rates(first_epoch, num_train_epochs, lr_ramp_power, lora_lr, ti_lr, save_path='learning_rates.png'):
        epochs = range(first_epoch, num_train_epochs)
        lora_lrs = []
        ti_lrs = []
        
        for epoch in epochs:
            completion_f = epoch / num_train_epochs
            # param_groups[0] goes from 0.0 to lora_lr over the course of training
            lora_lrs.append(lora_lr * completion_f ** lr_ramp_power)
            # param_groups[1] goes from ti_lr to 0.0 over the course of training
            ti_lrs.append(ti_lr * (1 - completion_f) ** lr_ramp_power)
            
        plt.figure()
        plt.plot(epochs, lora_lrs, label='LoRA LR')
        plt.plot(epochs, ti_lrs, label='TI LR')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Curves')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    # plot the learning rates:
    def plot_lrs(lora_lrs, ti_lrs, save_path='learning_rates.png'):
        plt.figure()
        plt.plot(range(len(lora_lrs)), lora_lrs, label='LoRA LR')
        plt.plot(range(len(lora_lrs)), ti_lrs, label='TI LR')
        plt.yscale('log')  # Set y-axis to log scale
        plt.ylim(1e-5, 3e-3)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Curves')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    ti_lrs, lora_lrs = [], []
    #plot_learning_rates(first_epoch, num_train_epochs, lr_ramp_power, lora_lr, ti_lr, save_path=os.path.join(checkpoint_dir, 'learning_rates.png'))

    # Count the total number of lora parameters
    total_n_lora_params = sum(p.numel() for p in unet_lora_parameters)

    for epoch in range(first_epoch, num_train_epochs):
        
        if is_lora:
            if hard_pivot:
                if epoch == num_train_epochs // 2:
                    print("# PTI :  Pivot halfway")
                    # remove text encoder parameters from the optimizer
                    optimizer.param_groups = params_to_optimize[:1]

                    # remove the optimizer state corresponding to text_encoder_parameters
                    for param in text_encoder_parameters:
                        if param in optimizer.state:
                            del optimizer.state[param]

            else: # Update learning rates for embeddings and lora:
                completion_f = epoch / num_train_epochs
                # param_groups[0] goes from 0.0 to lora_lr over the course of training
                optimizer.param_groups[0]['lr'] = lora_lr * completion_f ** lr_ramp_power
                # param_groups[1] goes from ti_lr to 0.0 over the course of training
                optimizer.param_groups[1]['lr'] = ti_lr * (1 - completion_f) ** lr_ramp_power

        unet.train()

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")
            progress_bar.refresh()

            if global_step % 200 == 0 and debug and False:
                plot_torch_hist(unet_lora_parameters, global_step, checkpoint_dir, "lora_weights", bins=100, min_val=-0.5, max_val=0.5, ymax_f = 0.07)
                plot_torch_hist(embedding_handler.get_trainable_embeddings(), global_step, checkpoint_dir, "embeddings_weights", bins=100, min_val=-0.05, max_val=0.05, ymax_f = 0.05)

            global_step += 1

            (tok1, tok2), vae_latent, mask = batch
            vae_latent = vae_latent.to(weight_dtype)

            # tokens to text embeds
            prompt_embeds_list = []
            for tok, text_encoder in zip((tok1, tok2), text_encoders):
                prompt_embeds_out = text_encoder(
                    tok.to(text_encoder.device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            # Create Spatial-dimensional conditions.

            original_size = (resolution, resolution)
            target_size = (resolution, resolution)
            crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])

            add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype).repeat(
                bs_embed, 1
            )

            added_kw = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(vae_latent)
            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            )
            timesteps = timesteps.long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_kw,
            ).sample

            loss = (model_pred - noise).pow(2) * mask
            loss = loss.mean()

            sparsity_lambda = 0.0 # disable L1 norm for now
            if sparsity_lambda > 0.0:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / total_n_lora_params
                loss = loss + sparsity_lambda * l1_norm

            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()

            # every step, we reset the non-trainable embeddings to the original embeddings.
            embedding_handler.retract_embeddings(print_stds = (global_step % 50 == 0))

            # Track the learning rates for final plotting:
            try:
                ti_lrs.append(optimizer.param_groups[1]['lr'])
            except:
                ti_lrs.append(0.0)

            lora_lrs.append(optimizer.param_groups[0]['lr'])

            # Print some statistics:
            if (global_step % checkpointing_steps == 0) and (global_step > 201):
                plot_lrs(lora_lrs, ti_lrs, save_path=f'{output_dir}/learning_rates.png')
                output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
                save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_param_to_optimize_names)
                last_save_step = global_step


    # final_save
    if (global_step - last_save_step) > 50:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
    else:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{last_save_step}"

    if not os.path.exists(output_save_dir):
        save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_param_to_optimize_names)
    else:
        print(f"Skipping final save, {output_save_dir} already exists")


    return output_save_dir


if __name__ == "__main__":
    main()
