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

        self.chunk_size = (self.pipeline.num_frame_per_block - 1) * 4 + 1
        self.chunk_size = 9
        self.width = params.width
        self.height = params.height
        self.fps = args.fps

        self.prompt = params.prompt
        self.noise = torch.randn(1, 3, 16, self.height // 8, self.width // 8).to(
            device=self.device, dtype=torch.bfloat16
        )
        self.noise_scale = args.noise_scale
        self.pipeline.prepare(noise=self.noise, text_prompts=[self.prompt])
        self.images = []
        self.current_start = 0
        self.current_end = self.chunk_size

    def predict(self, params: "Pipeline.InputParams") -> List[Image.Image]:
        if params.prompt != self.prompt:
            self.prompt = params.prompt
            self.pipeline.prepare(noise=self.noise, text_prompts=[self.prompt])

        image_tensor = self.image_to_tensor(params.image, self.width, self.height)
        self.images.append(image_tensor)

        if len(self.images) >= self.chunk_size:
            time_start = time.time()
            images = torch.stack(self.images[:self.chunk_size], dim=0).unsqueeze(0)
            images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=self.device)
            latents = self.pipeline.vae.model.encode(images, [0,1])
            latents = latents.transpose(2,1)
            latents = self.noise * self.noise_scale + latents * (1 - self.noise_scale)

            # Calculate sequence positions, but keep them within KV cache bounds
            max_seq_length = self.pipeline.frame_seq_length * 30  # KV cache size
            current_start = ((self.current_start // 3) * self.pipeline.frame_seq_length) % max_seq_length
            current_end = ((self.current_end // 3) * self.pipeline.frame_seq_length) % max_seq_length

            # # Ensure current_end is never 0 and is always greater than current_start
            if current_end < current_start:
                self.current_start += self.chunk_size
                self.current_end += self.chunk_size
                current_start = ((self.current_start // 3) * self.pipeline.frame_seq_length) % max_seq_length
                current_end = ((self.current_end // 3) * self.pipeline.frame_seq_length) % max_seq_length

            video = self.pipeline.inference(
                noise=latents,
                current_start=current_start,
                current_end=current_end,
            )[0].permute(0, 2, 3, 1).cpu().numpy()
            self.images = []
            self.current_start += self.chunk_size
            self.current_end += self.chunk_size
            time_end = time.time()
            print(f"FPS model: {len(video) / (time_end - time_start)}")
            return [self.array_to_image(image) for image in video]
            # For testing io sync without model
            # outputs = [
            #     Image.fromarray(((image.cpu().numpy() + 1) * 127.5).astype(np.uint8))
            #     for image in self.images
            # ]
            # time.sleep(0.9)
            # self.images = []
            # return outputs
        return []

    def image_to_tensor(
        self, image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> torch.Tensor:
        image = image.convert("RGB").resize((width, height))
        image_tensor = torch.from_numpy(np.array(image))
        if normalize:
            image_tensor = image_tensor / 127.5 - 1.0
        return image_tensor

    def array_to_image(self, image_array: np.ndarray, normalize: bool = True) -> Image.Image:
        if normalize:
            image_array = image_array * 255.0
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        return image
