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

from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from fastvideo.models.wan_hf.pipeline_wan import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler



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
        cus_timesteps=None,
        flow_shift=3.0,
    ):
        self.pipe.check_inputs(prompt, negative_prompt, height, width)
        self.batch_size = 1
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        self.max_sequence_length = max_sequence_length
        self.pipe._attention_kwargs = attention_kwargs
        self.pipe._guidance_scale = guidance_scale
        self.transformer_dtype = self.pipe.transformer.dtype
        self.output_type = output_type
        self.cus_timesteps = cus_timesteps
        self.t_start = t_start
        self.pipe._guidance_scale = guidance_scale
        self.pipe._cfg_scales = None

        # 4. Prepare timesteps
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.pipe._execution_device)

        self.tde = self.pipe._execution_device
        if cus_timesteps is not None:
            self.scheduler.set_timesteps(num_inference_steps=len(cus_timesteps),sigmas=[x/1000 for x in cus_timesteps])
            self.timesteps = self.scheduler.timesteps
        
        self.add_noise_scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        self.add_noise_scheduler.set_timesteps(50, device=self.tde)

        self._set_prompt_embeds(prompt, negative_prompt)

        self.latents = None

    @property
    def do_classifier_free_guidance(self):
        return self.pipe._guidance_scale > 1.0 or self.pipe._cfg_scales is not None

    def _set_prompt_embeds(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            negative_prompt,
            self.do_classifier_free_guidance,
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

    @torch.no_grad()
    def video_decode(self, latents, return_dict=True):
        if not self.output_type == "latent":
            latents = latents.to(self.pipe.vae.dtype)
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean)
                .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.pipe.vae.decode(latents, return_dict=False)[0]
            video = self.pipe.video_processor.postprocess_video(video, output_type=self.output_type)
        else:
            video = latents

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    @torch.no_grad()
    def final_output(self, return_dict=True):
        output_video = []

        for i in range(len(self.timesteps)-self.t_start-2):
            t = self.timesteps[self.t_start+i+2:self.t_start+i+self.latents.shape[0]+2].to(self.tde)
            latent_model_input = self.latents.to(self.transformer_dtype)

            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=self.prompt_embeds.repeat(self.latents.shape[0], 1, 1),
                attention_kwargs=self.pipe._attention_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance or self.pipe._cfg_scales is not None:
                if (self.pipe._cfg_scales is not None and self.pipe._cfg_scales[i]==1) or self.pipe.guidance_scale==1:
                    pass
                else:
                    noise_uncond = self.pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=self.negative_prompt_embeds,
                        attention_kwargs=self.pipe._attention_kwargs,
                        return_dict=False,
                    )[0]
                    if self.pipe._cfg_scales is None:
                        noise_pred = noise_uncond + self.pipe.guidance_scale * (noise_pred - noise_uncond)
                    else:
                        noise_pred = noise_uncond + self.pipe._cfg_scales[i] * (noise_pred - noise_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            self.latents = self.scheduler.step(noise_pred, t, self.latents, return_dict=False)[0]
            if self.latents.shape[0] >= len(self.timesteps)-self.t_start-2-i:
                latents = self.latents[-1]
                self.latents = self.latents[:-1]

                output_video.append(self.video_decode(latents, return_dict=return_dict).frames[0])
            else:
                continue

        return output_video
    
    @torch.no_grad()
    def __call__(
        self,
        input_video: torch.Tensor,
        return_dict: bool = True,
    ):
        latents = self.pipe.vae.encode(input_video.to(self.pipe._execution_device), return_dict=False)[0].mean
        noise = torch.randn_like(latents)
        timestep = self.add_noise_scheduler.timesteps[self.t_start]
        latents = self.add_noise_scheduler.add_noise(latents, noise, torch.tensor([timestep]).to(self.tde))

        if self.latents is None:
            self.latents = latents
        else:
            self.latents = torch.cat([latents, self.latents], dim=0)
            assert self.latents.shape[0] <= self.num_inference_steps-self.t_start-1, \
                f"latents.shape[0]: {self.latents.shape[0]}, num_inference_steps: {self.num_inference_steps}, t_start: {self.t_start}"
        
        # 6. Denoising loop
        t = self.timesteps[self.t_start+1:self.t_start+self.latents.shape[0]+1].to(self.tde)
        latent_model_input = self.latents.to(self.transformer_dtype)

        noise_pred = self.pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t,
            encoder_hidden_states=self.prompt_embeds.repeat(self.latents.shape[0], 1, 1),
            attention_kwargs=self.pipe._attention_kwargs,
            return_dict=False,
        )[0]


        if self.do_classifier_free_guidance or self.pipe._cfg_scales is not None:
            if (self.pipe._cfg_scales is not None and self.pipe._cfg_scales[0]==1) or self.pipe.guidance_scale==1:
                pass
            else:
                noise_uncond = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=self.negative_prompt_embeds,
                    attention_kwargs=self.pipe._attention_kwargs,
                    return_dict=False,
                )[0]
                if self.pipe._cfg_scales is None:
                    noise_pred = noise_uncond + self.pipe.guidance_scale * (noise_pred - noise_uncond)
                else:
                    noise_pred = noise_uncond + self.pipe._cfg_scales[i] * (noise_pred - noise_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        self.latents = self.scheduler.step(noise_pred, t, self.latents, return_dict=False)[0]

        if self.latents.shape[0] >= self.num_inference_steps-self.t_start-1:
            latents = self.latents[-1]
            self.latents = self.latents[:-1]
            return self.video_decode(latents, return_dict=return_dict).frames[0]

        else:
            return None
