import os
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import pandas as pd
import PIL
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
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

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=torch.float32)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)

    print(text_encoder_one)

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


class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, 
        inserting_toks: List[str],
        starting_toks:  Optional[List[str]] = None
        ):

        print("Initializing new tokens...")
        print(inserting_toks)

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
                text_encoder.text_model.embeddings.token_embedding.weight.data.std()
            )
            std_token_mean = (  
                text_encoder.text_model.embeddings.token_embedding.weight.data.mean()
            )

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

                print(f"Text encoder {idx} token_embedding_std:  {std_token_embedding}")
                print(f"Text encoder {idx} token_embedding_mean: {std_token_mean}")

                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids
                ] = (
                    torch.randn(
                        len(self.train_ids), text_encoder.text_model.config.hidden_size
                    )
                    .to(device=self.device)
                    .to(dtype=self.dtype)
                    * std_token_embedding
                )

            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            idx += 1

    def get_trainable_embeddings(self):
        
        trainable_embeddings = []
        for idx, text_encoder in enumerate(self.text_encoders):
            trainable_embeddings.append(
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids
                ]
            )

        return trainable_embeddings
            

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









    def save_embeddings(self, file_path: str):
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
            tensors[f"text_encoders_{idx}"] = new_token_embeddings

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

    @torch.no_grad()
    def retract_embeddings(self, off_ratio_power = 0.1, print_stds = False):
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
            off_ratio = std_token_embedding / new_embeddings.std()

            if off_ratio < 0.95:
                print(f"std-off ratio (avg-std / embedding-std) = {off_ratio:.6f}, this is prob not great..")

            new_embeddings = new_embeddings * (off_ratio**off_ratio_power)
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                index_updates
            ] = new_embeddings

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

    def load_embeddings(self, file_path: str):
        with safe_open(file_path, framework="pt", device=self.device.type) as f:
            for idx in range(len(self.text_encoders)):
                text_encoder = self.text_encoders[idx]
                tokenizer = self.tokenizers[idx]

                loaded_embeddings = f.get_tensor(f"text_encoders_{idx}")
                self._load_embeddings(loaded_embeddings, tokenizer, text_encoder)
