import torch
import torch.nn as nn
from functools import partial
import clip
import torch.optim as optim
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
import kornia
import os, random
import glob
import numpy as np
from PIL import Image

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class PersonalizedCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder with the option of personalization with an aesthetic embedding"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        aesthetic_steps=0,
        lr=0.0001,
        aesthetic_target=None,
        ag_L2_normalization_constant = 0.01,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.version = version

        self.tokenizer              = CLIPTokenizer.from_pretrained(version)
        self.full_clip_processor    = CLIPProcessor.from_pretrained(version)
        self.frozen_full_clip_model = CLIPModel.from_pretrained(version).to(self.device)

        self.aesthetic_steps = aesthetic_steps
        self.lr = lr
        self.aesthetic_target = aesthetic_target
        self.ag_L2_normalization_constant = ag_L2_normalization_constant

        self.image_embs = None
        self.freeze()

    def freeze(self):
        self.frozen_full_clip_model = self.frozen_full_clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def embed_images_into_clip(self, fraction_of_imgs_to_use = 1.0):
        """
        self.aesthetic_target is a list of loaded PIL Images
        """

        assert len(self.aesthetic_target) > 0, "No aesthetic target images provided"
        assert isinstance(self.aesthetic_target[0], Image.Image), "Aesthetic target images must be PIL images"
        
        with torch.no_grad():
            embs = []
            for pil_img in self.aesthetic_target:
                image = self.full_clip_processor(images=pil_img, return_tensors="pt")["pixel_values"].to(self.device)
                embs.append(self.frozen_full_clip_model.get_image_features(image))

        return torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

    def get_aesthetic_embeddings(self, verbose = 1):
        if self.aesthetic_target is None:
            return None
            
        if isinstance(self.aesthetic_target, str):
            if ".pt" in self.aesthetic_target: # Load ag embeddings from file:
                if verbose:
                    print(f"Loading aesthetic embedding from {self.aesthetic_target}")
                self.image_embs = torch.load(self.aesthetic_target).to(self.device)
        else: # Compute aesthetic_embedding from given list of target pil images:
            if verbose:
                print(f"Computing aesthetic embedding from {len(self.aesthetic_target)} images")
            self.image_embs = self.embed_images_into_clip().to(self.device)

        self.image_embs /= self.image_embs.norm(dim=-1, keepdim=True)
        return self.image_embs

    def finetune_text_encoder(self, tokens, verbose = 1):
        """
        self.ag_L2_normalization_constant: adds an L2 loss to constrain the L2 change in the raw text token embeddings

        # TODO: New loss:
            - Keep all img embeddings (no mean)
            - compute all separate CLIP sims
            - only use the good ones as loss contributions:
            - square the sim values (small sims are squashed)
                - eg softmax() layer on top of sims
            - use this loss instead of single sim of avg embeddings

        # TODO add additional L2 regularizer for unconditional prompt embedding ""

        # TODO checkout textual inversion from a single image: 
        # --> https://twitter.com/nearcyan/status/1591885331837898753?s=46&t=rdvNY1y3Rs4MjO0cJ19z6Q

        """
        
        # Reload the clip-txt-encoder with original weights:
        dynamic_full_clip_model = CLIPModel.from_pretrained(self.version, ).to(self.device)
        dynamic_full_clip_model.text_model = dynamic_full_clip_model.text_model.train()
        
        # Turn off gradients for vision component to save gpu memory:
        for param in dynamic_full_clip_model.vision_model.parameters():
            param.requires_grad = False
        
        # Get orig text embeddings:
        orig_text_hidden_features = dynamic_full_clip_model.text_model(input_ids=tokens).last_hidden_state.detach().clone()

        # get orig embedding or uc_tokens:
        uc_tokens = self.tokenizer(
            "",
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(self.device)
        orig_uc_text_hidden_features = dynamic_full_clip_model.text_model(input_ids=uc_tokens).last_hidden_state.detach().clone()

        
        # Get aesthetic image embeddings:
        self.image_embs = self.get_aesthetic_embeddings()

        if self.image_embs is not None:
            # We optimize the model to maximize the similarity between images and input text
            optimizer = optim.Adam(dynamic_full_clip_model.text_model.parameters(), lr=self.lr)

            print(f"Finetuning CLIP text encoder with L2_normalization_constant: {self.ag_L2_normalization_constant:.3f}")

            import time

            for i in range(self.aesthetic_steps):
                optimizer.zero_grad() 

                combined_tokens = torch.cat([tokens, uc_tokens], dim=0)
                combined_outputs = dynamic_full_clip_model.text_model(input_ids=combined_tokens)

                text_hidden_features = combined_outputs.last_hidden_state[0]
                uc_hidden_features   = combined_outputs.last_hidden_state[1]

                # CLIP similarities:
                text_embs = dynamic_full_clip_model.text_projection(combined_outputs[1][0])
                text_embs /= text_embs.norm(dim=-1, keepdim=True)
                sim = text_embs @ self.image_embs.T
                # clamp similarity since optimizing beyond 0.5 is probably not useful:
                sim = torch.clamp(sim, 0, 0.5)

                # L2 differences:
                l2_latent_diff_hidden    = (orig_text_hidden_features - text_hidden_features).norm(dim=-1).mean() / 25.
                uc_l2_latent_diff_hidden = (orig_uc_text_hidden_features - uc_hidden_features).norm(dim=-1).mean() / 25.

                # Compute loss:
                loss = -sim + self.ag_L2_normalization_constant*l2_latent_diff_hidden + 2*self.ag_L2_normalization_constant*uc_l2_latent_diff_hidden

                if verbose:
                    print(f"Step {i:02d}\t--> sim: {sim.mean().item():.3f}, l2_latent_diff: {l2_latent_diff_hidden.item():.3f}, uc_l2_latent_diff: {uc_l2_latent_diff_hidden.item():.3f}")

                # Update the CLIP text encoder weights:
                loss.mean().backward()
                optimizer.step()
            
        # Deactivate finetuning in case self.forward() is called again in the same outer loop (eg when doing animations):
        self.aesthetic_steps = 0
        
        # Some gpu memory optimization:
        dynamic_full_clip_model.text_model = dynamic_full_clip_model.text_model.eval()
        # This breaks the model, why????
        #for param in dynamic_full_clip_model.text_model.parameters():
        #    param.requires_grad = False
        self.freeze()

        return dynamic_full_clip_model

    def forward(self, text):
        with torch.enable_grad():
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(self.device)

            if text[0] != "" and self.aesthetic_steps != 0 and self.lr != 0:  # we can only finetune if we have an input prompt
                dynamic_full_clip_model = self.finetune_text_encoder(tokens)
                z = dynamic_full_clip_model.text_model(input_ids=tokens)
            else:
                z = self.frozen_full_clip_model.text_model(input_ids=tokens)

            return z.last_hidden_state.detach()

    def encode(self, text):
        return self(text)



if __name__ == "__main__":
    from ldm.util import count_params
    model = PersonalizedCLIPEmbedder()
    count_params(model, verbose=True)
