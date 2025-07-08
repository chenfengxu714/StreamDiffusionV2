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

import time

def wan_forward_origin(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,

        student=True,
        output_features=False,
        output_features_stride=2,
        final_layer=False,
        unpachify_layer=False,
        midfeat_layer=False,

    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if student:
            self.disable_adapters()


        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if output_features:
            features_list = []
        
        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for _, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

                if output_features and _ % output_features_stride == 0:
                    features_list.append(hidden_states)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if output_features:
            if final_layer:
                ori_features_list = torch.stack(features_list, dim=0)
                new_feat_list = []
                for xfeat in features_list:
                    tmp = (self.norm_out(xfeat.float()) * (1 + scale) + shift).type_as(xfeat)
                    tmp = self.proj_out(tmp)
                    tmp = tmp.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )
                    tmp = tmp.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    tmp = tmp.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                    new_feat_list.append(tmp)
                features_list = torch.stack(new_feat_list, dim=0)
            else:
                ori_features_list = torch.stack(features_list, dim=0)
                features_list = torch.stack(features_list, dim=0) 
        else:
            features_list = None
            ori_features_list = None


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,features_list, ori_features_list)

        return Transformer2DModelOutput(sample=output)

def wan_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,

        student=True,
        output_features=False,
        output_features_stride=2,
        final_layer=False,
        unpachify_layer=False,
        midfeat_layer=False,

    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if student and int(timestep[0])<=978:
            self.set_adapter('lora1')
            self.enable_adapters()
        else:
            return self.wan_forward_origin(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,

                student=student,
                output_features=output_features,
                output_features_stride=output_features_stride,
                final_layer=final_layer,
                unpachify_layer=unpachify_layer,
                midfeat_layer=midfeat_layer,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder_lora(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        if output_features:
            features_list = []

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for _, block in enumerate(self.blocks):  # 30
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

                if output_features and _ % output_features_stride == 0:
                    features_list.append(hidden_states)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out_lora(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out_lora(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if output_features:
            if final_layer:
                ori_features_list = torch.stack(features_list, dim=0)
                new_feat_list = []
                for xfeat in features_list:
                    tmp =  (self.norm_out_lora(xfeat.float()) * (1 + scale) + shift).type_as(xfeat)
                    tmp = self.proj_out_lora(tmp)

                    tmp = tmp.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )
                    tmp = tmp.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    tmp = tmp.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                    new_feat_list.append(tmp)
                features_list = torch.stack(new_feat_list, dim=0)
            else:
                ori_features_list = torch.stack(features_list, dim=0)
                features_list = torch.stack(features_list, dim=0) 
        else:
            features_list = None
            ori_features_list = None
                
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,features_list, ori_features_list)

        return Transformer2DModelOutput(sample=output)

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


def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=local_rank)
    initialize_sequence_parallel_state(world_size)



def inference(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    weight_dtype = torch.bfloat16

    with open(args.model_path+"transformer/config.json", "r") as f:
        wan_config_dict = json.load(f)
    transformer = WanTransformer3DModel(**wan_config_dict)
    flow_shift = args.flow_shift # 5.0 for 720P, 3.0 for 480P

    pipe = WanPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=weight_dtype
    )
    pipe.vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    transformer_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    pipe.transformer.add_adapter(transformer_lora_config, adapter_name="lora1")
    transformer.add_layer()

    from safetensors.torch import load_file as safetensors_load_file
    print('loading from....',args.model_path+'/transformer/diffusion_pytorch_model.safetensors')
    original_state_dict = safetensors_load_file(args.model_path+'/transformer/diffusion_pytorch_model.safetensors')
    pipe.transformer.load_state_dict(original_state_dict, strict=True)
    pipe.transformer.__class__.forward  = wan_forward
    pipe.transformer.__class__.wan_forward_origin = wan_forward_origin

    if args.lora_checkpoint_dir is not None:
        print(f"Loading LoRA weights from {args.lora_checkpoint_dir}")
        config_path = os.path.join(args.lora_checkpoint_dir,
                                   "lora_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        rank = lora_config_dict["lora_params"]["lora_rank"]
        lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        lora_scaling = lora_alpha / rank
        pipe.load_lora_weights(args.lora_checkpoint_dir,
                               adapter_name="default")
        pipe.set_adapters(["default"], [lora_scaling])
        print(
            f"Successfully Loaded LoRA weights from {args.lora_checkpoint_dir}"
        )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    pipe.scheduler = InferencePCMFMScheduler(
        1000,
        17,
        50,
    )

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

    # Pad so that T is a multiple of args.num_frames
    if input_video.shape[1] % args.num_frames != 0:
        pad_len = args.num_frames - (input_video.shape[1] % (args.num_frames))
        last_frame = input_video[:, [-1], :, :]  # shape: (C, 1, H, W)
        trailing_frames = last_frame.repeat(1, pad_len, 1, 1)
        input_video = torch.cat([input_video, trailing_frames], dim=1)

    C, T, H, W = input_video.shape
    input_video = input_video.reshape(T // args.num_frames, C, args.num_frames, H, W)

    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )
    scheduler.set_timesteps(50, device="cuda")

    start_time = time.time()
    output_video = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(input_video)):
            with torch.no_grad():
                latents = pipe.vae.encode(input_video[[i]].to("cuda"), return_dict=False)[0].mean
            latents = scheduler.add_noise(
                latents,
                torch.randn_like(latents),
                torch.tensor([scheduler.timesteps[args.t_start]])
            )
            output = streamv2v(latents).frames[0]
            output_video.append(output)
        # Finish the incomplete denoising for the last few snippets
        for i in range(args.num_inference_steps - 1):
            output = streamv2v(torch.zeros_like(latents)).frames[0]
            output_video.append(output)
        output_video = output_video[args.num_inference_steps - 1:]
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
    
    inference(args)
