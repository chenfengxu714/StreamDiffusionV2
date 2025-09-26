from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from streamv2v.inference import compute_noise_scale_and_step

import sys
import os
import numpy as np
import time
import threading
from omegaconf import OmegaConf
import random

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

import torch
import torch.distributed as dist

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from typing import List


default_prompt = "Anime style, a young beautiful girl speaking directly to the camera. She has big sparkling eyes, soft long hair flowing gently, and a bright, pure smile. The scene is full of light and warmth, with cherry blossoms floating in the background under a clear blue sky. Soft pastel colors, clean outlines, and a refreshing, heartwarming atmosphere, in Japanese anime aesthetic."

page_content = """<h1 class="text-3xl font-bold">StreamV2V</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://jeff-liangf.github.io/projects/streamv2v/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamV2V
</a>
video-to-video pipeline using
    <a
    href="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">4-step LCM LORA</a
    > with a MJPEG stream server.
</p>
<p class="text-sm">
The base model is <a
href="https://huggingface.co/runwayml/stable-diffusion-v1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD 1.5</a
    >. We also build in <a
    href="https://github.com/Jeff-LiangF/streamv2v/tree/main/demo_w_camera#download-lora-weights-for-better-stylization"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">some LORAs
</a> for better stylization.
</p>
"""

class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        width: int = Field(
            400, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            400, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device):
        torch.set_grad_enabled(False)

        params = self.InputParams()
        config = OmegaConf.load(args.config_path)
        for k, v in args._asdict().items():
            config[k] = v
        config["height"] = params.height
        config["width"] = params.width

        self.device = device
        self.pipeline = CausalStreamInferencePipeline(config, device=device)
        self.pipeline.to(device=device, dtype=torch.bfloat16)

        state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
            'generator']

        self.pipeline.generator.load_state_dict(
            state_dict, strict=True
        )

        self.chunk_size = 4
        self.start_chunk_size = 5
        self.first_batch = True
        self.overlap = args.overlap
        self.width = params.width
        self.height = params.height

        self.prompt = params.prompt
        self.noise_scale = args.noise_scale
        self.images = []
        self.prevs = []
        self.current_start = 0
        self.current_end = self.pipeline.frame_seq_length * 2
        self.end_idx = self.start_chunk_size

        self.images_lock = threading.Lock()
        self.model_lock = threading.Lock()

    def accept_new_params(self, params: "Pipeline.InputParams"):
        image_array = self.image_to_array(params.image, self.width, self.height)
        with self.images_lock:
            self.images.append(image_array)

        if params.prompt and self.prompt != params.prompt:
            self.update_prompt(params.prompt)

    def update_prompt(self, prompt: str):
        self.prompt = prompt
        with self.model_lock:
            self.pipeline.vae.model.first_encode = True
            self.pipeline.vae.model.first_decode = True
            self.pipeline.kv_cache1 = None
            self.pipeline.crossattn_cache = None
            self.pipeline.block_x = None
            self.pipeline.hidden_states = None

            self.current_start = 0
            self.current_end = self.pipeline.frame_seq_length * 2
            self.end_idx = self.start_chunk_size
            self.first_batch = True
            self.prevs = []
            self.pipeline.prepare(noise=torch.zeros(1, 1).to(self.device, dtype=torch.bfloat16), text_prompts=[self.prompt], device=self.device, dtype=torch.bfloat16, current_start=self.current_start, current_end=self.current_end)
            self.current_start = self.current_end
            self.current_end += (self.chunk_size // 4) * self.pipeline.frame_seq_length

    def clear_images(self):
        with self.images_lock:
            self.images = []
        with self.model_lock:
            self.prevs = []
            self.first_batch = True
            self.current_start = 0
            self.current_end = self.pipeline.frame_seq_length * 2
            self.end_idx = self.start_chunk_size

    def predict(self) -> List[Image.Image]:
        # time_start = time.time()
        with self.model_lock:
            num_frames_needed = self.start_chunk_size if self.first_batch else self.chunk_size
            with self.images_lock:
                images = self.read_images(num_frames_needed)
            if len(images) == 0:
                return []

            torch.set_grad_enabled(False)
            if len(self.prevs) > 0:
                total_images = np.concatenate([self.prevs, images], axis=0)
            else:
                total_images = images
            if self.overlap > 0:
                self.prevs = total_images[-self.overlap:]

            total_tensor = torch.from_numpy(total_images).to(dtype=torch.bfloat16).to(device=self.device)
            total_tensor = total_tensor.permute(3, 0, 1, 2).unsqueeze(0)
            new_noise_scale, current_step = compute_noise_scale_and_step(
                input_video_original=total_tensor,
                end_idx=self.end_idx,
                chunck_size=self.chunk_size,
                noise_scale=float(self.noise_scale),
            )
            self.noise_scale = new_noise_scale

            images = torch.from_numpy(images).unsqueeze(0)
            images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=self.device)

            latents = self.pipeline.vae.model.stream_encode(images)
            latents = latents.transpose(2,1)
            noise = torch.randn_like(latents)
            noisy_latents = noise * self.noise_scale + latents * (1 - self.noise_scale)

            with torch.inference_mode():
                if self.first_batch:
                    denoised_pred = self.pipeline.prepare(
                        text_prompts=[self.prompt],
                        device=self.device,
                        dtype=torch.bfloat16,
                        block_mode='input',
                        noise=noisy_latents,
                        current_start=self.current_start,
                        current_end=self.current_end,
                    )
                    video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
                    video = (video * 0.5 + 0.5).clamp(0, 1)
                    video = video[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    self.first_batch = False
                else:
                    denoised_pred = self.pipeline.inference_stream(
                        noise=noisy_latents,
                        current_start=self.current_start,
                        current_end=self.current_end,
                        current_step=current_step,
                    )
                    video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                    video = (video * 0.5 + 0.5).clamp(0, 1)
                    video = video[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()

            self.current_start = self.current_end
            self.current_end += (self.chunk_size // 4) * self.pipeline.frame_seq_length
            self.end_idx += self.chunk_size
            # images = (images + 1) * 127.5
            # random_sleep = random.uniform(0.3, 0.5)
            # time.sleep(random_sleep)

        # time_end = time.time()
        # print(f"FPS model: {len(video) / (time_end - time_start)}")
        return [self.array_to_image(image) for image in video[self.overlap:]]

    def read_images(self, num_images: int):
        if len(self.images) >= num_images:
            images = np.stack(self.images[:num_images], axis=0)
            self.images = self.images[num_images:]
            return images
        else:
            return []

    def image_to_array(
        self, image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> np.ndarray:
        image = image.convert("RGB").resize((width, height))
        image_array = np.array(image)
        if normalize:
            image_array = image_array / 127.5 - 1.0
        return image_array

    def array_to_image(self, image_array: np.ndarray, normalize: bool = True) -> Image.Image:
        if normalize:
            image_array = image_array * 255.0
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        return image
