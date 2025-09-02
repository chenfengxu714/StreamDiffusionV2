from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
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
    # assert 8 / 16 <= aspect_ratio <= 17 / 16, (
    #     f"Unsupported aspect ratio: {aspect_ratio:.2f} for shape {video.shape}"
    # )
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
parser.add_argument("--noise_scale", type=float, default=0.700)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=832)
parser.add_argument("--fps", type=int, default=30)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
# Add all command-line args into the config for downstream use
for k, v in vars(args).items():
    config[k] = v

pipeline = CausalStreamInferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")['generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0) # [1, C, T, H, W]
input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device="cuda")

chunck_size = 4
num_chuncks = (input_video_original.shape[2]-1) // chunck_size

dataset = TextDataset(args.prompt_file_path)
os.makedirs(args.output_folder, exist_ok=True)
prompts = [dataset[0]]

video_list = []
cost_time = 0
noise_scale = args.noise_scale
num_steps = len(config.denoising_step_list)

torch.cuda.synchronize()
start_time = time.time()

denoised_pred_list = []
for i in range(num_chuncks):

    if i==0:
        start_idx = 0
        end_idx = 5
        current_start = 0
        current_end = pipeline.frame_seq_length*2
    else:
        start_idx = end_idx
        end_idx = end_idx+chunck_size
        current_start = current_end
        current_end = current_end+(chunck_size//4)*pipeline.frame_seq_length

    input_video = input_video_original[:,:,start_idx:end_idx]
    # enhance_scale = depth_anything.infer_video(input_video)
    
    latents = pipeline.vae.model.stream_encode(input_video)

    latents = latents.transpose(2,1)
    
    if i==0:
        pipeline.prepare(batch_size=num_steps, text_prompts=prompts, device="cuda", dtype=torch.bfloat16)

    noise = torch.randn_like(latents)
    noisy_latents = noise*noise_scale + latents*(1-noise_scale)

    if i==0:
        # Split the first chunk into two parts
        denoised_pred_list.append(
            pipeline.inference(
                noise=noisy_latents[:, [0], :, :, :],
                current_start=current_start,
                current_end=pipeline.frame_seq_length,
            )
        )
        denoised_pred_list.append(
            pipeline.inference(
                noise=noisy_latents[:, [1], :, :, :],
                current_start=pipeline.frame_seq_length,
                current_end=current_end,
            )
        )
    else:
        denoised_pred_list.append(
            pipeline.inference(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
            )
        )

    # The first few steps do not return meaningful results
    if i < num_steps - 2:
        continue
    # The latest two outputs are the results of the first chunk, concatenate them
    if i == num_steps - 2:
        denoised_pred = torch.cat(denoised_pred_list[-2:], dim=1)
    else:
        denoised_pred = denoised_pred_list[-1]
    denoised_pred_list = []

    video = pipeline.vae.stream_decode_to_pixel(denoised_pred)
    video = (video * 0.5 + 0.5).clamp(0, 1)

    video = video[0].permute(0, 2, 3, 1).cpu().numpy()

    video_list.append(video)

    torch.cuda.synchronize()
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"Time taken: {cost_time:.4f} seconds, {video.shape[0]} frames, FPS: {video.shape[0]/cost_time:.4f}")
    start_time = end_time

for i in range(num_steps - 2):
    denoised_pred = pipeline.inference(
        noise=torch.zeros_like(noisy_latents),
        current_start=current_start,
        current_end=current_end,
    )
    video = pipeline.vae.stream_decode_to_pixel(denoised_pred)
    video = (video * 0.5 + 0.5).clamp(0, 1)

    video = video[0].permute(0, 2, 3, 1).cpu().numpy()

    video_list.append(video)

    torch.cuda.synchronize()
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"Time taken: {cost_time:.4f} seconds, {video.shape[0]} frames, FPS: {video.shape[0]/cost_time:.4f}")
    start_time = end_time

video = np.concatenate(video_list, axis=0)

export_to_video(
    video, os.path.join(args.output_folder, f"output_{0:03d}.mp4"), fps=args.fps)
