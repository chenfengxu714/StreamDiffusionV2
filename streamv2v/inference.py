from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
import time
import numpy as np

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads an .mp4 video and returns it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file.
        max_frames (int, optional): Maximum number of frames to load. If None, loads all.
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing.
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1].

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    h, w = video.shape[-2:]
    aspect_ratio = h / w
    assert 8 / 16 <= aspect_ratio <= 17 / 16, (
        f"Unsupported aspect ratio: {aspect_ratio:.2f} for shape {video.shape}"
    )
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # Final shape: [C, T, H, W]

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--video_path", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = CausalStreamInferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

input_video_original = load_mp4_as_tensor(args.video_path).unsqueeze(0) # [1, C, T, H, W]
input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device="cuda")

chunck_size=(pipeline.num_frame_per_block-1)*4+1
overlap = 3
num_chuncks = (input_video_original.shape[2]-overlap) // (chunck_size-overlap)

dataset = TextDataset(args.prompt_file_path)
os.makedirs(args.output_folder, exist_ok=True)
prompts = [dataset[0]]

video_list = []
cost_time = 0

for i in range(num_chuncks):
    torch.cuda.synchronize()
    start_time = time.time()

    if i==0:
        start_idx = 0
        end_idx = chunck_size
    else:
        start_idx = start_idx+chunck_size-overlap
        end_idx = end_idx+chunck_size-overlap

    input_video = input_video_original[:,:,start_idx:end_idx]

    latents = pipeline.vae.model.encode(input_video, [0,1])
    latents = latents.transpose(2,1)
    
    if i==0:
        pipeline.prepare(noise=latents, text_prompts=prompts)
        noise = torch.randn_like(latents)

    noise_scale = 0.8

    latents = noise*noise_scale + latents*(1-noise_scale)

    video = pipeline.inference(
        noise=latents,
        current_start=(start_idx//3)*pipeline.frame_seq_length,
        current_end=(end_idx//3)*pipeline.frame_seq_length,
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    torch.cuda.synchronize()
    cost_time+=time.time()-start_time

    if i==0:
        video_list.append(video)
    else:
        video_list.append(video[overlap:])

video = np.concatenate(video_list, axis=0)

T=video.shape[0]

print(f"Time taken: {cost_time} seconds")
print(f"FPS: {T/cost_time}")

export_to_video(
    video, os.path.join(args.output_folder, f"output_{0:03d}.mp4"), fps=16)
