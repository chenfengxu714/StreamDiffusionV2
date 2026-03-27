"""Shared helpers for the StreamDiffusionV2 inference entrypoints."""

import os
from typing import Any

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


def resolve_config_path(config_path: str, args) -> str:
    """Select an alternate config file when runtime flags imply one."""
    fast = bool(args.get("fast", False)) if isinstance(args, dict) else bool(getattr(args, "fast", False))
    if not fast:
        return config_path

    base_name = os.path.basename(config_path)
    if base_name != "wan_causal_dmd_v2v.yaml":
        return config_path

    fast_config_path = os.path.join(os.path.dirname(config_path), "wan_causal_dmd_v2v_fast.yaml")
    return fast_config_path if os.path.exists(fast_config_path) else config_path


def merge_cli_config(config_path: str, args) -> OmegaConf:
    """Load a YAML config and overlay CLI arguments onto it."""
    config_path = resolve_config_path(config_path, args)
    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.create(vars(args) if not isinstance(args, dict) else args)
    config = OmegaConf.merge(config, cli_config)
    config = normalize_acceleration_flags(config)

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


def _get_flag(config: Any, key: str, default=False):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_flag(config: Any, key: str, value) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def normalize_acceleration_flags(config):
    """Apply shared CLI/runtime flag semantics for fast and TensorRT modes."""
    use_taehv = bool(_get_flag(config, "use_taehv", False))
    use_tensorrt = bool(_get_flag(config, "use_tensorrt", False))
    fast = bool(_get_flag(config, "fast", False))

    if fast:
        use_taehv = True
        use_tensorrt = True

    # The current TensorRT path is implemented on top of the TAEHV decoder.
    if use_tensorrt:
        use_taehv = True

    _set_flag(config, "use_taehv", use_taehv)
    _set_flag(config, "use_tensorrt", use_tensorrt)
    _set_flag(config, "fast", fast)
    return config
