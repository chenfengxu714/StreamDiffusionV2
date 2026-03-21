"""Shared helpers for the StreamDiffusionV2 inference entrypoints."""

import os

import torch
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an mp4 video as a tensor with shape [C, T, H, W]."""
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        _, t, _, _ = video.shape
        video = torch.stack(
            [TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)],
            dim=1,
        )
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video


def merge_cli_config(config_path: str, args) -> OmegaConf:
    """Load a YAML config and overlay CLI arguments onto it."""
    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.create(vars(args) if not isinstance(args, dict) else args)
    config = OmegaConf.merge(config, cli_config)

    # CLI --step should always select the first N non-zero denoising steps from
    # the canonical YAML schedule, then append the terminal zero step back.
    full_denoising_list = list(config.denoising_step_list)
    non_terminal_steps = [step for step in full_denoising_list if int(step) != 0]
    step_value = int(config.step)
    config.denoising_step_list = non_terminal_steps[:step_value]
    config.denoising_step_list.append(0)
    return config


def load_generator_state_dict(checkpoint_folder: str):
    """Load the generator weights from a checkpoint folder."""
    ckpt_path = os.path.join(checkpoint_folder, "model.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("generator", "generator_ema", "state_dict"):
            if key in checkpoint:
                return ckpt_path, checkpoint[key]

    return ckpt_path, checkpoint
