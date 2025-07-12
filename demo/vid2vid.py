import sys
import os
import json
import numpy as np

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

from streamv2v.pipeline import StreamV2V
from streamv2v.init_utils import (
    initialize_distributed,
    init_wan_pipe,
)

# base_model = "runwayml/stable-diffusion-v1-5"
base_model = "Jiali/stable-diffusion-1.5"

default_prompt = "A man is talking"

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
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            240, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            400, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        initialize_distributed()
        wan_pipe = init_wan_pipe(args)
        params = self.InputParams()
        self.last_prompt = params.prompt

        self.streamv2v = StreamV2V(wan_pipe)
        self.streamv2v.prepare(
            prompt=self.last_prompt,
            negative_prompt=args.neg_prompt,
            height=params.height,
            width=params.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            t_start=args.t_start,
            flow_shift=args.flow_shift,
            cus_timesteps=[
                torch.tensor([1000]),
                torch.tensor([992]),
                torch.tensor([982]),
                torch.tensor([949]),
                torch.tensor([905]),
                torch.tensor([810])
            ],
        )

        self.images = []
        self.num_frames = args.num_frames
        self.overlap = args.overlap

    def predict(self, params: "Pipeline.InputParams") -> List[Image.Image]:
        if params.prompt != self.last_prompt:
            self.last_prompt = params.prompt
            self.streamv2v.update_prompt(params.prompt)

        image_tensor = self.image_to_tensor(params.image, params.width, params.height)
        self.images.append(image_tensor)

        if len(self.images) >= self.num_frames:
            images = torch.stack(self.images[:self.num_frames], dim=0).unsqueeze(0)
            # [B, T, H, W, C] -> [B, C, T, H, W]
            images = images.permute(0, 4, 1, 2, 3)
            output = self.streamv2v(images)
            self.images = self.images[self.num_frames - self.overlap:]
            if output is not None:
                output_images = [self.array_to_image(image) for image in output]
                return output_images

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
            image_array = (image_array + 1.0) * 127.5
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        return image
