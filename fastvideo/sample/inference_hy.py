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

from diffusers import BitsAndBytesConfig
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
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

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    nccl_info,
)

from fastvideo.utils.load import load_transformer
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.distill.solver import InferencePCMFMScheduler
from peft import LoraConfig

def hy_forward_origin(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        student=True,

        output_features=False,
        output_features_stride=8,
        final_layer=False,
        unpachify_layer=False,
        midfeat_layer=False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    
    if student:
        self.disable_adapters()
    if True:
        if True:
            if guidance is None:
                guidance = torch.tensor([6016.0],
                                        device=hidden_states.device,
                                        dtype=torch.bfloat16)

            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get(
                        "scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                    )

            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p, p_t = self.config.patch_size, self.config.patch_size_t
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p
            post_patch_width = width // p

            pooled_projections = encoder_hidden_states[:, 0, :self.config.
                                                    pooled_projection_dim]
            encoder_hidden_states = encoder_hidden_states[:, 1:]

            # 1. RoPE
            image_rotary_emb = self.rope(hidden_states)

            # 2. Conditional embeddings
            
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
            

            hidden_states = self.x_embedder(hidden_states)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states,
                                                        timestep,
                                                        encoder_attention_mask)

            # 3. Attention mask preparation
            latent_sequence_length = hidden_states.shape[1]
            condition_sequence_length = encoder_hidden_states.shape[1]
            sequence_length = latent_sequence_length + condition_sequence_length
            attention_mask = torch.zeros(batch_size,
                                        sequence_length,
                                        sequence_length,
                                        device=hidden_states.device,
                                        dtype=torch.bool)  # [B, N, N]

            effective_condition_sequence_length = encoder_attention_mask.sum(
                dim=1, dtype=torch.int)
            effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

            for i in range(batch_size):
                attention_mask[i, :effective_sequence_length[i], :
                            effective_sequence_length[i]] = True

            # 4. Transformer blocks
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False
                } if is_torch_version(">=", "1.11.0") else {}

                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                if output_features:
                    features_list = []
                # for block in self.single_transformer_blocks:
                for _, block in enumerate(self.single_transformer_blocks):
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                    if output_features and _ % output_features_stride == 0:
                        features_list.append(hidden_states)

            else:
                if output_features:
                    features_list = []
                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, attention_mask,
                        image_rotary_emb)

                # for block in self.single_transformer_blocks:
                for _, block in enumerate(self.single_transformer_blocks):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, attention_mask,
                        image_rotary_emb)
                    if output_features and _ % output_features_stride == 0:
                        features_list.append(hidden_states)

            # 5. Output projection
            hidden_states = self.norm_out(hidden_states, temb) 
            hidden_states = self.proj_out(hidden_states) 
            hidden_states = hidden_states.reshape(batch_size,
                                                post_patch_num_frames,
                                                post_patch_height,
                                                post_patch_width, -1, p_t, p, p)
            hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
            hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            if output_features:
                if final_layer:
                    ori_features_list = torch.stack(features_list, dim=0)
                    new_feat_list = []
                    for xfeat in features_list:
                        tmp = self.norm_out(xfeat, temb)
                        tmp = self.proj_out(tmp)

                        tmp = tmp.reshape(batch_size,
                                            post_patch_num_frames,
                                            post_patch_height,
                                            post_patch_width, -1, p_t, p, p)
                        
                        tmp = tmp.permute(0, 4, 1, 5, 2, 6, 3, 7)
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
                return (hidden_states, features_list, ori_features_list)

            return Transformer2DModelOutput(sample=hidden_states)



def hy_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        student=True,

        output_features=False,
        output_features_stride=8,
        final_layer=False,
        unpachify_layer=False,
        midfeat_layer=False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if student and int(timestep[0])<=981:
            self.set_adapter('lora1')
            self.enable_adapters()
        else:
            return self.hy_forward_origin(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
                student=student,
                output_features=output_features,
                output_features_stride=output_features_stride,
                final_layer=final_layer,
                unpachify_layer=unpachify_layer,
                midfeat_layer=midfeat_layer,
            )

        if guidance is None:
            guidance = torch.tensor([6016.0],
                                    device=hidden_states.device,
                                    dtype=torch.bfloat16)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get(
                    "scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        pooled_projections = encoder_hidden_states[:, 0, :self.config.
                                                   pooled_projection_dim]
        encoder_hidden_states = encoder_hidden_states[:, 1:]

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        if student and int(timestep[0])<=981:
            temb = self.time_text_embed_lora(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        

        hidden_states = self.x_embedder(hidden_states)

        if True:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states,
                                                      timestep,
                                                      encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.zeros(batch_size,
                                     sequence_length,
                                     sequence_length,
                                     device=hidden_states.device,
                                     dtype=torch.bool)  # [B, N, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(
            dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

        for i in range(batch_size):
            attention_mask[i, :effective_sequence_length[i], :
                           effective_sequence_length[i]] = True

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {
                "use_reentrant": False
            } if is_torch_version(">=", "1.11.0") else {}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            if output_features:
                features_list = []
            # for block in self.single_transformer_blocks:
            for _, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
                if output_features and _ % output_features_stride == 0:
                    features_list.append(hidden_states)

        else:
            if output_features:
                features_list = []
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask,
                    image_rotary_emb)

            # for block in self.single_transformer_blocks:
            for _, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask,
                    image_rotary_emb)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(hidden_states)

        # 5. Output projection
        hidden_states = self.norm_out_lora(hidden_states, temb) if student and int(timestep[0])<=981 else self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out_lora(hidden_states) if student and int(timestep[0])<=981 else self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size,
                                              post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, -1, p_t, p, p)
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if output_features:
            if final_layer:
                ori_features_list = torch.stack(features_list, dim=0)
                new_feat_list = []
                for xfeat in features_list:
                    tmp = self.norm_out_lora(xfeat, temb) if student and int(timestep[0])<=981 else self.norm_out(xfeat, temb)
                    tmp = self.proj_out_lora(tmp) if student and int(timestep[0])<=981 else self.proj_out(tmp)

                    tmp = tmp.reshape(batch_size,
                                        post_patch_num_frames,
                                        post_patch_height,
                                        post_patch_width, -1, p_t, p, p)
                    
                    tmp = tmp.permute(0, 4, 1, 5, 2, 6, 3, 7)
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
            return (hidden_states, features_list, ori_features_list)

        return Transformer2DModelOutput(sample=hidden_states)



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
        hy_config_dict = json.load(f)
    transformer = HunyuanVideoTransformer3DModel(**hy_config_dict)

    pipe = HunyuanVideoPipeline.from_pretrained(args.model_path,
                                                transformer=transformer,
                                                torch_dtype=weight_dtype)
    transformer_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    pipe.transformer.add_adapter(transformer_lora_config, adapter_name="lora1")
    transformer.add_layer()

    original_state_dict = safetensors_load_file(args.model_path+'/transformer/diffusion_pytorch_model.safetensors')
    pipe.transformer.load_state_dict(original_state_dict, strict=True)
    
    pipe.transformer.__class__.forward  = hy_forward
    pipe.transformer.__class__.hy_forward_origin = hy_forward_origin

    pipe.enable_vae_tiling()

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

    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = (torch.load(args.prompt_embed_path,
                                    map_location="cpu",
                                    weights_only=True).to(device).unsqueeze(0))
        encoder_attention_mask = (torch.load(
            args.encoder_attention_mask_path,
            map_location="cpu",
            weights_only=True).to(device).unsqueeze(0))
        prompts = None
    elif args.prompt_path is not None:
        prompts = [line.strip() for line in open(args.prompt_path, "r")]
        prompt_embeds = None
        encoder_attention_mask = None
    else:
        prompts = args.prompt
        prompt_embeds = None
        encoder_attention_mask = None

    pipe.scheduler = InferencePCMFMScheduler(
            1000,
            17,
            50,
        )

    if prompts is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                generator = torch.Generator("cpu").manual_seed(args.seed)
                video = pipe.dcm(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    cus_sigmas=[1.000,0.9920,0.9818,0.9124]
                ).frames
                if nccl_info.global_rank <= 0:
                    os.makedirs(args.output_path, exist_ok=True)
                    suffix = prompt.split(".")[0][:150]
                    export_to_video(
                        video[0],
                        os.path.join(args.output_path, f"{suffix}.mp4"),
                        fps=24,
                    )
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generator = torch.Generator("cpu").manual_seed(args.seed)
            videos = pipe.dcm(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=encoder_attention_mask,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                cus_sigmas=[1.000,0.9920,0.9818,0.9124]
            ).frames

        if nccl_info.global_rank <= 0:
            export_to_video(videos[0], args.output_path + ".mp4", fps=24)


def inference_quantization(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_path

    if args.quantization == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer/",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config)
    if args.quantization == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer/",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config)
    elif not args.quantization:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer/",
            torch_dtype=torch.bfloat16).to(device)

    print("Max vram for read transformer:",
          round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3),
          "GiB")
    torch.cuda.reset_max_memory_allocated(device)

    if not args.cpu_offload:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device)
        pipe.transformer = transformer
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id,
                                                    transformer=transformer,
                                                    torch_dtype=torch.bfloat16)
    torch.cuda.reset_max_memory_allocated(device)
    pipe.scheduler._shift = args.flow_shift
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    print("Max vram for init pipeline:",
          round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3),
          "GiB")
    with open(args.prompt) as f:
        prompts = f.readlines()

    generator = torch.Generator("cpu").manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.cuda.reset_max_memory_allocated(device)
    for prompt in prompts:
        start_time = time.perf_counter()
        output = pipe.dcm(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            cus_sigmas=[1.000,0.9920,0.9818,0.9124]
        ).frames[0]
        export_to_video(output,
                        os.path.join(args.output_path, f"{prompt[:100]}.mp4"),
                        fps=args.fps)
        print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
        print(
            "Max vram for denoise:",
            round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3),
            "GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
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
    if args.quantization:
        inference_quantization(args)
    else:
        inference(args)