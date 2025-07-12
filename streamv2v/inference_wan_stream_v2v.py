import argparse
import json
import os
import time
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from diffusers import BitsAndBytesConfig
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    export_to_video,
)

from peft import LoraConfig
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info,
)

from fastvideo.models.wan_hf.modeling_wan import WanTransformer3DModel
from fastvideo.models.wan_hf.pipeline_wan import WanPipeline

from fastvideo.distill.solver import InferencePCMFMScheduler

from pipeline import StreamV2V
from init_utils import (
    initialize_distributed,
    init_wan_pipe,
)

import time
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def split_video_with_overlap(input_video, window_size=17, overlap=1):
    """
    input_video: Tensor of shape (C, T, H, W)
    returns: Tensor of shape (B, C, window_size, H, W)
    """
    C, T, H, W = input_video.shape
    stride = window_size - overlap
    num_batches = (T - overlap) // stride

    batches = []
    for i in range(num_batches):
        start = i * stride
        end = start + window_size
        if end > T:
            break 
        batch = input_video[:, start:end]  
        batches.append(batch.unsqueeze(0))  

    return torch.cat(batches, dim=0) 


def inference(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    pipe = init_wan_pipe(args)

    if args.prompt_path is not None:
        prompt = [line.strip() for line in open(args.prompt_path, "r")][0]
    else:
        prompt = args.prompt

    streamv2v = StreamV2V(pipe)
    streamv2v.prepare(
        prompt=prompt,
        negative_prompt=args.neg_prompt,
        height=args.height,
        width=args.width,
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
    input_video = load_mp4_as_tensor(
        args.reference_video_path,
        resize_hw=(args.height, args.width)
    )  # shape: (C, T, H, W)

    input_video = split_video_with_overlap(input_video, window_size=args.num_frames, overlap=1).unsqueeze(1)
    # input_video = input_video[0].unsqueeze(0)
    T = args.num_frames * input_video.shape[0]

    # output_video = input_video[0][0].permute(1,2,3,0).cpu().numpy()

    start_time = time.time()
    output_video = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(input_video.shape[0]):
            output=streamv2v(input_video[i])
            if output is not None:
                output_video.append(output)
        output_final = streamv2v.final_output()
        output_video.extend(output_final)
    output_video = np.concatenate(output_video, axis=0)

    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time for prompt '{prompt}': {generation_time:.2f} seconds")
    print(f"Generation fps: {T / generation_time:.2f}")

    if len(output_video) > 0:
        os.makedirs(args.output_path, exist_ok=True)
        suffix = prompt.split(".")[0][:150]
        export_to_video(output_video, os.path.join(args.output_path, f"{suffix}.mp4"), fps=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--t_start", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--reference_video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed for evaluation.")
    parser.add_argument("--neg_prompt",
                        type=str,
                        default=None,
                        help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift",
                        type=int,
                        default=7,
                        help="Flow shift parameter.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help=
        "Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default=
        "data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help=
        "Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help=
        "Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver",
                        type=str,
                        default="euler",
                        help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision",
                        type=str,
                        default="bf16",
                        choices=["fp32", "fp16", "bf16", "fp8"])
    parser.add_argument("--rope-theta",
                        type=int,
                        default=256,
                        help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision",
                        type=str,
                        default="fp16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template",
                        type=str,
                        default="dit-llm-encode")
    parser.add_argument("--prompt-template-video",
                        type=str,
                        default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    args = parser.parse_args()
    seed_everything(args.seed)
    
    inference(args)
