from causvid.ode_data.create_lmdb_iterative import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
import os

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

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ODERegressionDataset(Dataset):
    def __init__(self, data_path, max_pair=int(1e8)):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        return {
            "prompts": self.data_dict['prompts'][idx],
            "ode_latent": self.data_dict['latents'][idx].squeeze(0),
        }


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), load_video: bool = False):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair
        self.load_video = load_video
        

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        if self.load_video:
            video_tensor = load_mp4_as_tensor(f"mixkit_videos/{prompts[:100]}.mp4")
        else:
            video_tensor = None
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "video_tensor": video_tensor
        }
