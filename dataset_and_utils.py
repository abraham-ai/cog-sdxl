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

def prepare_image(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


def prepare_mask(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("L"))
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer_1,
        tokenizer_2,
        vae_encoder,
        text_encoder_1=None,
        text_encoder_2=None,
        do_cache: bool = False,
        size: int = 512,
        text_dropout: float = 0.0,
        scale_vae_latents: bool = True,
        substitute_caption_map: Dict[str, str] = {},
    ):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.csv_path = csv_path

        self.caption = self.data["caption"]
        # make it lowercase
        self.caption = self.caption.str.lower()
        for key, value in substitute_caption_map.items():
            self.caption = self.caption.str.replace(key.lower(), value)

        self.image_path = self.data["image_path"]

        if "mask_path" not in self.data.columns:
            self.mask_path = None
        else:
            self.mask_path = self.data["mask_path"]

        if text_encoder_1 is None:
            self.return_text_embeddings = False
        else:
            self.text_encoder_1 = text_encoder_1
            self.text_encoder_2 = text_encoder_2
            self.return_text_embeddings = True
            assert (
                NotImplementedError
            ), "Preprocessing Text Encoder is not implemented yet"

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

        self.vae_encoder = vae_encoder
        self.scale_vae_latents = scale_vae_latents
        self.text_dropout = text_dropout
        self.size = size

        if do_cache:
            self.vae_latents = []
            self.tokens_tuple = []
            self.masks = []

            self.do_cache = True

            print("Captions to train on: ")
            for idx in range(len(self.data)):
                token, vae_latent, mask = self._process(idx)
                self.vae_latents.append(vae_latent)
                self.tokens_tuple.append(token)
                self.masks.append(mask)

            print(f"Cached latents and masks for {len(self.vae_latents)} images.")

            del self.vae_encoder

        else:
            self.do_cache = False

    @torch.no_grad()
    def _process(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        image_path = self.image_path[idx]
        image_path = os.path.join(os.path.dirname(self.csv_path), image_path)

        image = PIL.Image.open(image_path).convert("RGB")

        image = prepare_image(image, self.size, self.size).to(
            dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
        )

        caption = self.caption[idx]
        print(caption)

        # tokenizer_1
        ti1 = self.tokenizer_1(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids

        ti2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids

        vae_latent = self.vae_encoder.encode(image).latent_dist.sample()

        if self.scale_vae_latents:
            vae_latent = vae_latent * self.vae_encoder.config.scaling_factor

        if self.mask_path is None:
            mask = torch.ones_like(
                vae_latent, dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

        else:
            mask_path = self.mask_path[idx]
            mask_path = os.path.join(os.path.dirname(self.csv_path), mask_path)

            mask = PIL.Image.open(mask_path)

            mask = prepare_mask(mask, self.size, self.size).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

            mask = torch.nn.functional.interpolate(
                mask, size=(vae_latent.shape[-2], vae_latent.shape[-1]), mode="nearest"
            )
            mask = mask.repeat(1, vae_latent.shape[1], 1, 1)

        assert len(mask.shape) == 4 and len(vae_latent.shape) == 4

        return (ti1.squeeze(), ti2.squeeze()), vae_latent.squeeze(), mask.squeeze()

    def __len__(self) -> int:
        return len(self.data)

    def atidx(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.do_cache:
            return self.tokens_tuple[idx], self.vae_latents[idx], self.masks[idx]
        else:
            return self._process(idx)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        token, vae_latent, mask = self.atidx(idx)
        return token, vae_latent, mask


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        print("Importing CLIPTextModel")
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        print("Importing CLIPTextModelWithProjection")
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_models(pretrained_model_name_or_path, revision, device, weight_dtype):

    if "juggernaut" not in pretrained_model_name_or_path:
        print("Loading base SDXL...")
        tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=revision,
            use_fast=False,
        )

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, revision, subfolder="text_encoder_2"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision
        )

    if "juggernaut" in pretrained_model_name_or_path:
        print(f"Loading model weights from {pretrained_model_name_or_path}...")
        pipe = StableDiffusionPipeline.from_single_file(
                    pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True)

        # Use fixed scheduler for training:
        noise_scheduler = DDPMScheduler(
            beta_end = 0.012,
            beta_schedule = "scaled_linear",
            beta_start = 0.00085,
            clip_sample = False,
            clip_sample_range = 1.0,
            dynamic_thresholding_ratio = 0.995,
            #"interpolation_type": "linear",
            num_train_timesteps = 1000,
            prediction_type = "epsilon",
            sample_max_value = 1.0,
            #set_alpha_to_one = False,
            #"skip_prk_steps": true,
            steps_offset = 1,
            thresholding = False,
            timestep_spacing = "leading",
            #"trained_betas": null,
            #"use_karras_sigmas": false,
            variance_type = "fixed_small"
            )
        
        # Extract the submodules from the model:
        vae = pipe.vae
        tokenizer_one = pipe.tokenizer
        tokenizer_two = pipe.tokenizer_2
        text_encoder_one = pipe.text_encoder
        text_encoder_two = pipe.text_encoder_2
        unet = pipe.unet

        del pipe

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=torch.float32)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return (
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    )


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[
                f"{attn_processor_key}.{parameter_key}"
            ] = parameter

    return attn_processors_state_dict

import torch
import torch.nn.functional as F

class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    
    def get_trainable_embeddings(self):
        
        trainable_embeddings = []
        for idx, text_encoder in enumerate(self.text_encoders):
            trainable_embeddings.append(
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids
                ]
            )

        return trainable_embeddings

    def find_nearest_tokens(self, query_embedding, tokenizer, text_encoder, idx, distance_metric, top_k = 5):
        # given a query embedding, compute the distance to all embeddings in the text encoder
        # and return the top_k closest tokens

        assert distance_metric in ["l2", "cosine"], "distance_metric should be either 'l2' or 'cosine'"
        
        # get all non-optimized embeddings:
        index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
        embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates]

        # compute the distance between the query embedding and all embeddings:
        if distance_metric == "l2":
            diff = (embeddings - query_embedding.unsqueeze(0))**2
            distances = diff.sum(-1)
            distances, indices = torch.topk(distances, top_k, dim=0, largest=False)
        elif distance_metric == "cosine":
            distances = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)
            distances, indices = torch.topk(distances, top_k, dim=0, largest=True)

        nearest_tokens = tokenizer.convert_ids_to_tokens(indices)
        return nearest_tokens, distances
        

    def print_token_info(self, distance_metric = "cosine"):
        print(f"----------- Closest tokens (distance_metric = {distance_metric}) --------------")
        current_token_embeddings = self.get_trainable_embeddings()
        idx = 0

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            query_embeddings = current_token_embeddings[idx]

            for token_id, query_embedding in enumerate(query_embeddings):
                nearest_tokens, distances = self.find_nearest_tokens(query_embedding, tokenizer, text_encoder, idx, distance_metric)

                # print the results:
                print(f"txt-encoder {idx}, token {token_id}: :")
                for i, (token, dist) in enumerate(zip(nearest_tokens, distances)):
                    print(f"---> {distance_metric} of {dist:.4f}: {token}")

            idx += 1

    def get_start_embedding(self, text_encoder, tokenizer, example_tokens, unk_token_id = 49407, verbose = False, desired_std_multiplier = 0.0):
        print('-----------------------------------------------')
        # do some cleanup:
        example_tokens = [tok.lower() for tok in example_tokens]
        example_tokens = list(set(example_tokens))

        starting_ids = tokenizer.convert_tokens_to_ids(example_tokens)

        # filter out any tokens that are mapped to unk_token_id:
        example_tokens = [tok for tok, tok_id in zip(example_tokens, starting_ids) if tok_id != unk_token_id]
        starting_ids = [tok_id for tok_id in starting_ids if tok_id != unk_token_id]

        if verbose:
            print("Token mapping:")
            for i, token in enumerate(example_tokens):
                print(f"{token} -> {starting_ids[i]}")

        embeddings, stds = [], []
        for i, token_index in enumerate(starting_ids):
            embedding = text_encoder.text_model.embeddings.token_embedding.weight.data[token_index].clone()
            embeddings.append(embedding)
            stds.append(embedding.std())
            #print(f"token: {example_tokens[i]}, embedding-std: {embedding.std():.4f}, embedding-mean: {embedding.mean():.4f}")

        embeddings = torch.stack(embeddings)
        #print(f"Embeddings: {embeddings.shape}, std: {embeddings.std():.4f}, mean: {embeddings.mean():.4f}")

        if verbose:
            # Compute the squared difference
            squared_diff = (embeddings.unsqueeze(1) - embeddings.unsqueeze(0)) ** 2
            squared_l2_dist = squared_diff.sum(-1)
            l2_distance_matrix = torch.sqrt(squared_l2_dist)

            print("Pairwise L2 Distance Matrix:")
            print(" \t" + "\t".join(example_tokens))
            for i, row in enumerate(l2_distance_matrix):
                print(f"{example_tokens[i]}\t" + "\t".join(f"{dist:.4f}" for dist in row))

        print(f"Using {len(embeddings)} embeddings to compute initial embedding...")
        init_embedding = embeddings.mean(dim=0)

        if (desired_std_multiplier is not None) and desired_std_multiplier > 0:
            avg_std        = torch.stack(stds).mean()
            current_std    = init_embedding.std()
            scale_factor   = desired_std_multiplier * avg_std / current_std
            init_embedding = init_embedding * scale_factor
            print(f"Scaled Mean Embedding: std: {init_embedding.std():.4f}, mean: {init_embedding.mean():.4f}")
        
        return init_embedding

    def initialize_new_tokens(self, 
        inserting_toks: List[str],
        starting_toks:  Optional[List[str]] = None,
        seed: int = 0,
        ):

        print("Initializing new tokens...")
        print(inserting_toks)
        torch.manual_seed(seed)

        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(
                inserting_toks, list
            ), "inserting_toks should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_toks
            ), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks

            print("Inserting new tokens into tokenizer:")
            print(self.inserting_toks)

            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = (
                text_encoder.text_model.embeddings.token_embedding.weight.data.std() #(axis=1).mean()
            )
            std_token_mean = (  
                text_encoder.text_model.embeddings.token_embedding.weight.data.mean() #(axis=1).mean()
            )

            print(f"Text encoder {idx} token_embedding_std:  {std_token_embedding}")
            print(f"Text encoder {idx} token_embedding_mean: {std_token_mean}")

            if starting_toks is not None:
                assert isinstance(
                    starting_toks, list
                ), "starting_toks should be a list of strings."
                assert all(
                    isinstance(tok, str) for tok in starting_toks
                ), "All elements in starting_toks should be strings."
                assert len(starting_toks) == len(self.inserting_toks), "starting_toks should have the same length as inserting_toks"
                self.starting_ids = tokenizer.convert_tokens_to_ids(starting_toks)

                print(f"Copying embeddings from starting tokens {starting_toks} to new tokens {self.inserting_toks}")
                print(f"Starting ids: {self.starting_ids}")

                # copy the embeddings of the starting tokens to the new tokens
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids] = text_encoder.text_model.embeddings.token_embedding.weight.data[self.starting_ids].clone()

            else:

                if 1: 
                    init_embeddings = (torch.randn(len(self.train_ids), text_encoder.text_model.config.hidden_size).to(device=self.device).to(dtype=self.dtype) * std_token_embedding)
                else:
                    first_tokens = [
                        "face",
                        "person",
                        "Aiden",
                        "Sophia",
                        "Liam",
                        "Isabella",
                        "Ethan",
                        "Emma",
                        "Lucas",
                        "Olivia",
                        "Noah",
                        "Ava",
                    ]

                    second_tokens = [
                        "face",
                        "person",
                        "Smith",
                        "Johnson",
                        "Williams",
                        "Brown",
                        "Jones",
                        "Garcia",
                        "Miller",
                        "Davis",
                        "Rodriguez",
                        "Martinez",
                    ]

                    start_embedding_one = self.get_start_embedding(text_encoder, tokenizer, first_tokens)
                    start_embedding_two = self.get_start_embedding(text_encoder, tokenizer, second_tokens)
                    init_embeddings = torch.stack([start_embedding_one, start_embedding_two])

                    print(f"init_embedding std: {init_embeddings.std():.4f}, avg-std: {std_token_embedding:.4f}")

                text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = init_embeddings.clone()

            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            idx += 1
            

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################


    def pre_optimize_token_embeddings(self, train_dataset, epochs=10):

        ### THIS FUNCTION IS NOT FINISHED YET

        for idx in range(len(train_dataset)):
            (tok1, tok2), vae_latent, mask = train_dataset[idx]
            image_path = train_dataset.image_path[idx]
            image_path = os.path.join(os.path.dirname(train_dataset.csv_path), image_path)
            image = PIL.Image.open(image_path).convert("RGB")

            print(f"---> Loaded sample {idx}:")
            print("Tokens:")
            print(tok1.shape)
            print(tok2.shape)
            print("Image:")
            print(image.size)

            # tokens to text embeds
            prompt_embeds_list = []
            #for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            for tok, text_encoder in zip((tok1, tok2), self.text_encoders):
                prompt_embeds_out = text_encoder(
                    tok.to(text_encoder.device),
                    output_hidden_states=True,
                )

                print("prompt_embeds_out:")
                print(prompt_embeds_out.shape)

                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            print("prompt_embeds:")
            print(prompt_embeds.shape)
            print("pooled_prompt_embeds:")
            print(pooled_prompt_embeds.shape)

    def save_embeddings(self, file_path: str, txt_encoder_keys = ["clip_l", "clip_g"]):
        assert (
            self.train_ids is not None
        ), "Initialize new tokens before saving embeddings."
        tensors = {}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[
                0
            ] == len(self.tokenizers[0]), "Tokenizers should be the same."
            new_token_embeddings = (
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids
                ]
            )
            tensors[txt_encoder_keys[idx]] = new_token_embeddings

        save_file(tensors, file_path)


    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    def _load_embeddings(self, loaded_embeddings, tokenizer, text_encoder):
        # Assuming new tokens are of the format <s_i>
        self.inserting_toks = [f"<s{i}>" for i in range(loaded_embeddings.shape[0])]
        special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
        tokenizer.add_special_tokens(special_tokens_dict)
        text_encoder.resize_token_embeddings(len(tokenizer))

        self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
        assert self.train_ids is not None, "New tokens could not be converted to IDs."
        text_encoder.text_model.embeddings.token_embedding.weight.data[
            self.train_ids
        ] = loaded_embeddings.to(device=self.device).to(dtype=self.dtype)

    def fix_embedding_std(self, off_ratio_power = 0.1):
        std_penalty = 0.0
        idx = 0

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            index_no_updates    = self.embeddings_settings[f"index_no_updates_{idx}"]
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]
            index_updates = ~index_no_updates

            new_embeddings = (text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates])

            off_ratio = std_token_embedding / new_embeddings.std()
            std_penalty += (off_ratio - 1.0)**2

            if (off_ratio < 0.95) or (off_ratio > 1.05):
                print(f"std-off ratio-{idx} (target-std / embedding-std) = {off_ratio:.4f}, prob not ideal...")

            # rescale the embeddings to have a more similar std as before:
            new_embeddings = new_embeddings * (off_ratio**off_ratio_power)
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                    index_updates
                ] = new_embeddings

            idx += 1


    @torch.no_grad()
    def retract_embeddings(self, print_stds = False):
        idx = 0
        means, stds = [], []

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                index_no_updates
            ] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we can normalize them a bit
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = (
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    index_updates
                ]
            )

            idx += 1

            if 0:
                # get the actual embeddings that will get updated:
                inu = torch.ones((len(tokenizer),), dtype=torch.bool)
                inu[self.train_ids] = False
                updateable_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[~inu].detach().clone().to(dtype=torch.float32).cpu().numpy()
                
                mean_0, mean_1 = updateable_embeddings[0].mean(), updateable_embeddings[1].mean()
                std_0, std_1 = updateable_embeddings[0].std(), updateable_embeddings[1].std()

                means.append((mean_0, mean_1))
                stds.append((std_0, std_1))

                if print_stds:
                    print(f"Text Encoder {idx} token embeddings:")
                    print(f" --- Means: ({mean_0:.6f}, {mean_1:.6f})")
                    print(f" --- Stds:  ({std_0:.6f}, {std_1:.6f})")

    def load_embeddings(self, file_path: str, txt_encoder_keys = ["clip_l", "clip_g"]):
        if not os.path.exists(file_path):
            file_path = file_path.replace(".pti", ".safetensors")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")

        with safe_open(file_path, framework="pt", device=self.device.type) as f:
            for idx in range(len(self.text_encoders)):
                text_encoder = self.text_encoders[idx]
                tokenizer = self.tokenizers[idx]
                try:
                    loaded_embeddings = f.get_tensor(txt_encoder_keys[idx])
                except:
                    loaded_embeddings = f.get_tensor(f"text_encoders_{idx}")
                self._load_embeddings(loaded_embeddings, tokenizer, text_encoder)