import os
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import pandas as pd
import gc
import PIL
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig

def load_models(pretrained_model, device, weight_dtype):
    if not isinstance(pretrained_model, dict) or 'path' not in pretrained_model or 'version' not in pretrained_model:
        raise ValueError("pretrained_model must be a dict with 'path' and 'version' keys")

    print(f"Loading model weights from {pretrained_model['path']}...")

    model_dir = os.path.abspath(os.path.dirname(pretrained_model['path']))
    print("Downloding HF model cache to ", model_dir)
    os.environ['TRANSFORMERS_CACHE'] = model_dir
    os.environ['HF_HOME'] = model_dir
    os.environ['HF_DATASETS_CACHE'] = model_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = model_dir
    os.environ['HF_HUB_CACHE'] = model_dir
    os.environ['HF_MODULES_CACHE'] = model_dir

    try:
        pretrained_model['path'] = './models/s15'
        pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model['path'], torch_dtype=torch.float16, use_safetensors=True, cache_dir = model_dir)
        
        #pipe.save_pretrained("./models/s15", safe_serialization=True)

        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        vae = pipe.vae
        unet = pipe.unet
        tokenizer_one = pipe.tokenizer
        text_encoder_one = pipe.text_encoder

        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)

        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=torch.float32)
        text_encoder_one.to(device, dtype=weight_dtype)

        tokenizer_two = text_encoder_two = None
        if pretrained_model['version'] == "sdxl":
            tokenizer_two = pipe.tokenizer_2
            text_encoder_two = pipe.text_encoder_2
            text_encoder_two.requires_grad_(False)
            text_encoder_two.to(device, dtype=weight_dtype)
        
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred while loading the models: {e}")
        raise

    print("Models loaded successfully.")

    return (
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    )


if __name__ == "__main__":

    pretrained_model = {
        "path": "/data/xander/Projects/cog/GitHub_repos/cog-sdxl/models/realisticVisionV60B1_v60B1VAE.safetensors",
        "version": "sd15",
    }

    (
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(pretrained_model, "cuda", torch.bfloat16)