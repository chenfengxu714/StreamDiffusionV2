"""TAEHV-based Wan VAE wrapper used by offline inference with --use_taehv."""

from __future__ import annotations

import os
import urllib.request
from collections import namedtuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from causvid.models.model_interface import VAEInterface
from causvid.models.wan.wan_wrapper import WanVAEWrapper

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_TAEHV_CHECKPOINT = os.path.join(REPO_ROOT, "ckpts", "taew2_1.pth")
DEFAULT_TAEHV_URL = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        x = self.conv(x)
        return x.reshape(-1, c, h, w)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    n, t, c, h, w = x.shape
    if parallel:
        x = x.reshape(n * t, c, h, w)
        for block in model:
            if isinstance(block, MemBlock):
                nt, c, h, w = x.shape
                t = nt // n
                _x = x.reshape(n, t, c, h, w)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :t].reshape(x.shape)
                x = block(x, mem)
            else:
                x = block(x)
        nt, c, h, w = x.shape
        t = nt // n
        x = x.view(n, t, c, h, w)
    else:
        out = []
        work_queue = [TWorkItem(xt, 0) for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]
        mem = [None] * len(model)
        while work_queue:
            xt, block_index = work_queue.pop(0)
            if block_index == len(model):
                out.append(xt)
                continue

            block = model[block_index]
            if isinstance(block, MemBlock):
                if mem[block_index] is None:
                    xt_new = block(xt, xt * 0)
                    mem[block_index] = xt
                else:
                    xt_new = block(xt, mem[block_index])
                    mem[block_index].copy_(xt)
                work_queue.insert(0, TWorkItem(xt_new, block_index + 1))
            elif isinstance(block, TPool):
                if mem[block_index] is None:
                    mem[block_index] = []
                mem[block_index].append(xt)
                if len(mem[block_index]) == block.stride:
                    n, c, h, w = xt.shape
                    xt = block(torch.cat(mem[block_index], 1).view(n * block.stride, c, h, w))
                    mem[block_index] = []
                    work_queue.insert(0, TWorkItem(xt, block_index + 1))
            elif isinstance(block, TGrow):
                xt = block(xt)
                n_out, c_out, h_out, w_out = xt.shape
                batch_size = n_out // block.stride
                grown = xt.view(batch_size, block.stride * c_out, h_out, w_out)
                for xt_next in reversed(grown.chunk(block.stride, dim=1)):
                    work_queue.insert(0, TWorkItem(xt_next, block_index + 1))
            else:
                xt = block(xt)
                work_queue.insert(0, TWorkItem(xt, block_index + 1))
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    latent_channels = 16
    image_channels = 3

    def __init__(self, checkpoint_path=DEFAULT_TAEHV_CHECKPOINT, decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True)):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(TAEHV.image_channels, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            conv(64, TAEHV.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(TAEHV.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], TAEHV.image_channels),
        )
        self.load_state_dict(self.patch_tgrow_layers(torch.load(checkpoint_path, map_location="cpu", weights_only=True)))

    def patch_tgrow_layers(self, state_dict):
        new_state_dict = self.state_dict()
        for index, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{index}.conv.weight"
                if state_dict[key].shape[0] > new_state_dict[key].shape[0]:
                    state_dict[key] = state_dict[key][-new_state_dict[key].shape[0]:]
        return state_dict

    def encode_video(self, x, parallel=True, show_progress_bar=False):
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        return apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)


class TAEHVWanVAEWrapper(VAEInterface):
    """Wan stream encoder with a TAEHV decoder for faster pixel reconstruction."""

    def __init__(
        self,
        model_type: str = "T2V-1.3B",
        checkpoint_path: str | None = None,
        auto_download: bool = True,
        parallel_decode: bool = False,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path or DEFAULT_TAEHV_CHECKPOINT
        self.parallel_decode = parallel_decode
        self.model = SimpleNamespace(first_encode=True, first_decode=True)
        self.decode_context_latents = 3
        self._decode_latent_cache: torch.Tensor | None = None
        self.encoder_vae = WanVAEWrapper(model_type=model_type)
        self.taehv = TAEHV(checkpoint_path=self._resolve_checkpoint(auto_download))

    def _resolve_checkpoint(self, auto_download: bool) -> str:
        if os.path.exists(self.checkpoint_path):
            return self.checkpoint_path

        if not auto_download:
            raise FileNotFoundError(f"TAEHV checkpoint not found: {self.checkpoint_path}")

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_TAEHV_URL, self.checkpoint_path)
        return self.checkpoint_path

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        if args:
            if len(args) >= 1 and not isinstance(args[0], torch.dtype):
                device = args[0]
            if len(args) >= 2 and isinstance(args[1], torch.dtype):
                dtype = args[1]
            elif len(args) == 1 and isinstance(args[0], torch.dtype):
                dtype = args[0]

        encoder_kwargs = {}
        if device is not None:
            encoder_kwargs["device"] = device
        if dtype is not None:
            encoder_kwargs["dtype"] = dtype
        self.encoder_vae.to(**encoder_kwargs)

        taehv_kwargs = {"dtype": torch.float16}
        if device is not None:
            taehv_kwargs["device"] = device
        self.taehv.to(**taehv_kwargs)
        return self

    def _pixels_to_unit_range(self, video: torch.Tensor) -> torch.Tensor:
        return (video * 0.5 + 0.5).clamp(0, 1)

    def _unit_range_to_pixels(self, video: torch.Tensor) -> torch.Tensor:
        return video.mul(2).sub(1).clamp(-1, 1)

    def _to_ntchw(self, video: torch.Tensor) -> torch.Tensor:
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def _to_ncthw(self, video: torch.Tensor) -> torch.Tensor:
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        video = self.taehv.decode_video(latent, parallel=self.parallel_decode, show_progress_bar=False)
        return self._unit_range_to_pixels(video)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode_to_pixel(latent)

    def stream_encode(self, video: torch.Tensor, is_scale=False) -> torch.Tensor:
        self.encoder_vae.model.first_encode = self.model.first_encode
        latent = self.encoder_vae.stream_encode(video, is_scale=is_scale)
        self.model.first_encode = self.encoder_vae.model.first_encode
        return latent

    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        model_dtype = next(self.taehv.parameters()).dtype
        latent = latent.to(dtype=model_dtype)

        # Match Self-Forcing's TAEHV usage: keep a short latent prefix so each
        # incremental decode sees recent temporal context, then trim the
        # already-emitted frames from the pixel output.
        if self.model.first_decode:
            self.model.first_decode = False
            self._decode_latent_cache = None
            decode_latent = latent
            emitted_latents = max(latent.shape[1] - 1, 0)
        else:
            context = self._decode_latent_cache
            decode_latent = latent if context is None else torch.cat([context.to(device=latent.device, dtype=model_dtype), latent], dim=1)
            emitted_latents = latent.shape[1]

        self._decode_latent_cache = decode_latent[:, -min(self.decode_context_latents, decode_latent.shape[1]):].detach().clone()
        video = self.taehv.decode_video(
            decode_latent,
            parallel=self.parallel_decode,
            show_progress_bar=False,
        )
        if emitted_latents:
            video = video[:, -emitted_latents * 4:, :, :, :]
        else:
            video = video[:, 0:0, :, :, :]
        return self._unit_range_to_pixels(video).to(dtype=latent.dtype)
