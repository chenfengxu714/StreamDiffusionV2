from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--num_rollout", type=int, default=3)
parser.add_argument("--num_overlap_frames", type=int, default=3)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)
assert args.num_overlap_frames % pipeline.num_frame_per_block == 0, "num_overlap_frames must be divisible by num_frame_per_block"

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

num_rollout = args.num_rollout

os.makedirs(args.output_folder, exist_ok=True)


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]
    start_latents = None
    all_video = []

    for rollout_index in range(num_rollout):
        sampled_noise = torch.randn(
            [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            start_latents=start_latents
        )

        current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()

        start_frame = encode(pipeline.vae, (
            video[:, -4 * (args.num_overlap_frames - 1) - 1:-4 * (args.num_overlap_frames - 1), :] * 2.0 - 1.0
        ).transpose(2, 1).to(torch.bfloat16)).transpose(2, 1).to(torch.bfloat16)

        start_latents = torch.cat(
            [start_frame, latents[:, -(args.num_overlap_frames - 1):]], dim=1
        )

        all_video.append(current_video[:-(4 * (args.num_overlap_frames - 1) + 1)])

    video = np.concatenate(all_video, axis=0)

    export_to_video(
        video, os.path.join(args.output_folder, f"long_video_output_{prompt_index:03d}.mp4"), fps=16)
