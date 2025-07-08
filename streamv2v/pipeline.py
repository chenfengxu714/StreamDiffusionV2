 # Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from fastvideo.models.wan_hf.pipeline_wan import WanPipeline
from fastvideo.distill.solver import InferencePCMFMScheduler


class StreamV2V:
    def __init__(
        self,
        pipe: WanPipeline,
    ):
        self.pipe = pipe

    def prepare(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        num_inference_steps: int = 4,
        guidance_scale: float = 5.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 226,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: str = "np",
        t_start: int = 1,
        cfg_scales=None,
        cus_timesteps=None,
    ):
        self.pipe.check_inputs(prompt, negative_prompt, height, width)
        self.batch_size = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        self.max_sequence_length = max_sequence_length
        self.pipe._attention_kwargs = attention_kwargs
        self.pipe._guidance_scale = guidance_scale
        self.pipe._cfg_scales = cfg_scales
        self.transformer_dtype = self.pipe.transformer.dtype
        self.output_type = output_type
        self.cus_timesteps = cus_timesteps
        self.t_start = t_start

        self.schedulers = [
            InferencePCMFMScheduler(1000, 17, 50) for _ in range(self.num_inference_steps)
        ]
        for scheduler in self.schedulers:
            scheduler.set_timesteps(self.num_inference_steps, device=self.pipe._execution_device)
            if self.cus_timesteps is not None:
                scheduler.set_timesteps(
                    num_inference_steps=len(self.cus_timesteps),
                    sigmas=[x/1000 for x in self.cus_timesteps],
                    device=self.pipe._execution_device
                )
        self.timesteps = self.schedulers[0].timesteps

        self._set_prompt_embeds(prompt, negative_prompt)

        self.latents = self.prepare_latents(
            batch_size=self.batch_size,
            num_channels_latents=self.pipe.transformer.config.in_channels,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            dtype=torch.float32,
            device=self.pipe._execution_device,
            generator=self.generator
        )

    def _set_prompt_embeds(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            negative_prompt,
            self.pipe.do_classifier_free_guidance,
            max_sequence_length=self.max_sequence_length,
        )
        self.prompt_embeds = prompt_embeds.repeat(
            self.batch_size, 1, 1
        ).to(self.transformer_dtype)
        if negative_prompt_embeds is not None:
            self.negative_prompt_embeds = negative_prompt_embeds.repeat(
                self.batch_size, 1, 1
            ).to(self.transformer_dtype)
        else:
            self.negative_prompt_embeds = None

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.pipe.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.pipe.vae_scale_factor_spatial,
            int(width) // self.pipe.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        new_latents: torch.Tensor,
        return_dict: bool = True,
    ):
        self.latents = torch.cat([new_latents, self.latents], dim=0)
        self.latents = self.latents[:self.batch_size]

        new_scheduler = InferencePCMFMScheduler(1000, 17, 50)
        new_scheduler.set_timesteps(self.num_inference_steps, device=self.pipe._execution_device)
        if self.cus_timesteps is not None:
            new_scheduler.set_timesteps(
                num_inference_steps=len(self.cus_timesteps),
                sigmas=[x / 1000 for x in self.cus_timesteps],
                device=self.pipe._execution_device
            )
        self.schedulers = [new_scheduler] + self.schedulers[:-1]

        noise_pred = self.pipe.transformer(
            hidden_states=self.latents.to(self.transformer_dtype),
            timestep=self.timesteps[self.t_start + 1:],
            encoder_hidden_states=self.prompt_embeds,
            attention_kwargs=self.pipe.attention_kwargs,
            return_dict=False,
        )[0]

        if self.pipe.do_classifier_free_guidance:
            noise_uncond = self.pipe.transformer(
                hidden_states=self.latents.to(self.transformer_dtype),
                timestep=self.timesteps[self.t_start + 1:],
                encoder_hidden_states=self.negative_prompt_embeds,
                attention_kwargs=self.pipe.attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + self.pipe.guidance_scale * (noise_pred - noise_uncond)

        for i in range(self.num_inference_steps):
            self.latents[[i]] = self.schedulers[i].step(
                noise_pred[[i]],
                self.timesteps[i + self.t_start + 1],
                self.latents[[i]],
                return_dict=False
            )[0].to(self.pipe.vae.dtype)

        latents = self.latents[[-1]]
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
                 .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                 .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(
            self.pipe.vae.config.latents_std
        ).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.pipe.vae.decode(latents, return_dict=False)[0]
        video = self.pipe.video_processor.postprocess_video(video, output_type=self.output_type)

        # Ensure video is a tensor for WanPipelineOutput
        if not isinstance(video, torch.Tensor):
            if isinstance(video, (list, tuple)):
                video = torch.stack([torch.tensor(frame) for frame in video])
            else:
                video = torch.tensor(video)

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
