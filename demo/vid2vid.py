from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline

import sys
import os
import numpy as np
import time
from omegaconf import OmegaConf

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


default_prompt = "A woman is talking"

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

def fold_2x2_spatial(video: torch.Tensor, original_batch: int) -> torch.Tensor:
    B4, C, T, H_half, W_half = video.shape
    assert B4 % 4 == 0 and B4 == original_batch * 4

    video = video.view(original_batch, 2, 2, C, T, H_half, W_half)  # (B, 2, 2, C, T, H//2, W//2)
    video = video.permute(0, 3, 4, 5, 1, 6, 2)  # (B, C, T, H//2, 2, W//2, 2)
    video = video.contiguous().view(original_batch, C, T, H_half * 2, W_half * 2)  # (B, C, T, H, W)

    return video


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
        self.has_started = False
        self.overlap = args.overlap
        self.width = params.width
        self.height = params.height
        self.unfold = args.unfold

        self.prompt = params.prompt
        self.noise_scale = args.noise_scale
        self.pipeline.prepare(noise=torch.zeros(1, 1).to(self.device, dtype=torch.bfloat16), text_prompts=[self.prompt])
        self.images = []
        self.prevs = []
        self.current_start = 0
        self.current_end = self.pipeline.frame_seq_length * 2

    def accept_image(self, params: "Pipeline.InputParams"):
        image_array = self.image_to_array(params.image, self.width, self.height)
        self.images.append(image_array)

    def predict(self) -> List[Image.Image]:
        num_frames_needed = self.chunk_size if self.has_started else self.start_chunk_size
        if len(self.images) + len(self.prevs) >= num_frames_needed:
            torch.set_grad_enabled(False)
            time_start = time.time()

            num_images = num_frames_needed - len(self.prevs)
            step = len(self.images) / (num_images - 1)
            indices = [int(i * step) for i in range(num_images - 1)] + [-1]
            images = np.stack([self.images[i] for i in indices], axis=0)
            self.images = []

            if len(self.prevs) > 0:
                total_images = np.concatenate([self.prevs, images], axis=0)
            else:
                total_images = images
            if self.overlap > 0:
                self.prevs = total_images[-self.overlap:]

            images = torch.from_numpy(images).unsqueeze(0)
            images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=self.device)
            latents = self.pipeline.vae.model.stream_encode(images)
            latents = latents.transpose(2,1)
            noise = torch.randn_like(latents)
            latents = noise * self.noise_scale + latents * (1 - self.noise_scale)

            with torch.inference_mode():
                video = self.pipeline.inference(
                    noise=latents,
                    current_start=self.current_start,
                    current_end=self.current_end,
                )
            if self.unfold:
                video = fold_2x2_spatial(video.transpose(1,2), 1).transpose(1,2)
            video = video[0].permute(0, 2, 3, 1).cpu().numpy()
            self.current_start = self.current_end
            self.current_end += (self.chunk_size // 4) * self.pipeline.frame_seq_length
            self.has_started = True
            time_end = time.time()
            print(f"FPS model: {len(video) / (time_end - time_start)}")
            return [self.array_to_image(image) for image in video[self.overlap:]]
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
