from causvid.models.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface
)
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers import AutoTokenizer
from typing import List
import torch


class SDXLTextEncoder(TextEncoderInterface):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", revision=None
        )

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", revision=None
        )

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", revision=None, use_fast=False
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", revision=None, use_fast=False
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _model_forward(self, batch: dict) -> dict:
        """
        Processes two sets of input token IDs using two separate text encoders, and returns both
        concatenated token-level embeddings and pooled sentence-level embeddings.

        Args:
            batch (dict):
                A dictionary containing:
                    - text_input_ids_one (torch.Tensor): The token IDs for the first tokenizer,
                    of shape [batch_size, num_tokens].
                    - text_input_ids_two (torch.Tensor): The token IDs for the second tokenizer,
                    of shape [batch_size, num_tokens].

        Returns:
            dict: A dictionary with two keys:
                - "prompt_embeds" (torch.Tensor): Concatenated embeddings from the second-to-last
                hidden states of both text encoders, of shape [batch_size, num_tokens, hidden_dim * 2].
                - "pooled_prompt_embeds" (torch.Tensor): Pooled embeddings (final layer output)
                from the second text encoder, of shape [batch_size, hidden_dim].
        """
        text_input_ids_one = batch['text_input_ids_one']
        text_input_ids_two = batch['text_input_ids_two']
        prompt_embeds_list = []

        for text_input_ids, text_encoder in zip([text_input_ids_one, text_input_ids_two], [self.text_encoder_one, self.text_encoder_two]):
            prompt_embeds = text_encoder(
                text_input_ids.to(self.device),
                output_hidden_states=True,
            )

            # We are only interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        # use the second text encoder's pooled prompt embeds (overwrite in for loop)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            len(text_input_ids_one), -1)

        output_dict = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        return output_dict

    def _encode_prompt(self, prompt_list):
        """
        Encodes a list of prompts with two tokenizers and returns a dictionary
        of the resulting tensors.
        """
        text_input_ids_one = self.tokenizer_one(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids

        prompt_dict = {
            'text_input_ids_one': text_input_ids_one,
            'text_input_ids_two': text_input_ids_two
        }
        return prompt_dict

    def forward(self, text_prompts: List[str]) -> dict:
        tokenized_prompts = self._encode_prompt(text_prompts)
        return self._model_forward(tokenized_prompts)


class SDXLVAE(VAEInterface):
    def __init__(self):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae"
        )

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.squeeze(1)
        latent = latent / self.vae.config.scaling_factor
        # ensure the output is float
        image = self.vae.decode(latent).sample.float()
        image = image.unsqueeze(1)
        return image


class SDXLWrapper(DiffusionModelInterface):
    def __init__(self):
        super().__init__()

        self.model = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="unet"
        )

        self.add_time_ids = self._build_condition_input(resolution=1024)

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )

        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: List[dict] = None, current_start: int = None,
        current_end: int = None
    ) -> torch.Tensor:
        # TODO: Check how to apply gradient checkpointing
        # [B, 1, C, H, W] -> [B, C, H, W]
        noisy_image_or_video = noisy_image_or_video.squeeze(1)

        # [B, 1] -> [B]
        timestep = timestep.squeeze(1)

        added_conditions = {
            "time_ids": self.add_time_ids.repeat(noisy_image_or_video.shape[0], 1).to(noisy_image_or_video.device),
            "text_embeds": conditional_dict["pooled_prompt_embeds"]
        }

        pred_noise = self.model(
            sample=noisy_image_or_video,
            timestep=timestep,
            encoder_hidden_states=conditional_dict['prompt_embeds'],
            added_cond_kwargs=added_conditions
        ).sample

        pred_x0 = self.scheduler.convert_noise_to_x0(
            noise=pred_noise,
            xt=noisy_image_or_video,
            timestep=timestep
        )

        # [B, C, H, W] -> [B, 1, C, H, W]
        pred_x0 = pred_x0.unsqueeze(1)

        return pred_x0

    @staticmethod
    def _build_condition_input(resolution):
        original_size = (resolution, resolution)
        target_size = (resolution, resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32)
        return add_time_ids
