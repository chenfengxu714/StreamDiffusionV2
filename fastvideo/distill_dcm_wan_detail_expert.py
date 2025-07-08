# !/bin/python3
# isort: skip_file
import os
import types
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from dataclasses import dataclass
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import BaseOutput, logging, check_min_version
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed,
    CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    FP32LayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle)
from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function
from fastvideo.distill.discriminator_wan import Discriminator
from fastvideo.distill.solver import extract_into_tensor
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule
from fastvideo.models.wan_hf.modeling_wan import WanTransformer3DModel
from fastvideo.models.wan_hf.pipeline_wan import WanPipeline
from fastvideo.utils.checkpoint import resume_lora_optimizer, save_checkpoint, save_lora_checkpoint
from fastvideo.utils.communications import broadcast, sp_parallel_dataloader_wrapper, all_gather, all_to_all_4D
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_discriminator_fsdp_kwargs, get_dit_fsdp_kwargs
from fastvideo.utils.load import load_transformer
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group, get_sequence_parallel_state, initialize_sequence_parallel_state)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


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

        if student and int(timestep[0])<=981:
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





def log_validation(
                args,
                transformer,
                device,
                datatype,
                curstep):
    model_id = "pretrained/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, transformer=transformer._fsdp_wrapped_module, torch_dtype=torch.bfloat16)
    from fastvideo.distill.solver import InferencePCMFMScheduler
    scheduler = InferencePCMFMScheduler(
                1000,
                17,
                50,
            )
    pipe.scheduler = scheduler

    pipe.to("cuda")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = ""

    seed=40
    cfg_scale=1.0
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=cfg_scale,
        generator = torch.Generator("cpu").manual_seed(seed),
        cus_timesteps=[torch.tensor([1000]),torch.tensor([992]),torch.tensor([982]),torch.tensor([949]),torch.tensor([905]),torch.tensor([810])],
        ).frames[0]

    filename = os.path.join(
                        args.output_dir,
                        f"validation_step_{curstep}_sample_{6}_guidance_{cfg_scale}_video_{0}.mp4",
                    )
    export_to_video(output, filename, fps=16)




class EulerSolver:

    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (np.arange(1, euler_timesteps + 1) *
                                self.step_ratio).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray(
            [0] + self.euler_timesteps[:-1].tolist())
        
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()
        )  # either use sigma0 or 0

        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.euler_timesteps_prev = torch.from_numpy(
            self.euler_timesteps_prev).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.euler_timesteps_prev = self.euler_timesteps_prev.to(device)

        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index,
                                    model_pred.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index,
                                         model_pred.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev

    def euler_style_multiphase_pred(
        self,
        sample,
        model_pred,
        timestep_index,
        multiphase,
        is_target=False,
    ):
        inference_indices = np.linspace(0,
                                        len(self.euler_timesteps),
                                        num=multiphase,
                                        endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (torch.from_numpy(inference_indices).long().to(
            self.euler_timesteps.device))
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0))
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(
            dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index_end = inference_indices[last_valid_index]

        if is_target:
            sigma = extract_into_tensor(self.sigmas_prev, timestep_index,
                                        sample.shape)
        else:
            sigma = extract_into_tensor(self.sigmas, timestep_index,
                                        sample.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index_end,
                                         sample.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred

        return x_prev, timestep_index_end



def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def get_norm(model_pred, norms, gradient_accumulation_steps):
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) /
                              gradient_accumulation_steps)
    absolute_mean = torch.mean(
        torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(
        torch.abs(model_pred)) / gradient_accumulation_steps
    dist.all_reduce(fro_norm, op=dist.ReduceOp.AVG)
    dist.all_reduce(largest_singular_value, op=dist.ReduceOp.AVG)
    dist.all_reduce(absolute_mean, op=dist.ReduceOp.AVG)
    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(
        largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()



def gan_d_loss(
    discriminator,
    teacher_transformer,
    sample_fake,
    sample_real,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    weight,
    discriminator_head_stride,
):
    loss = 0.0
    # collate sample_fake and sample_real
    with torch.no_grad():
        (_, fake_features, fake_features_ori) = teacher_transformer(
            sample_fake,
            timestep,
            encoder_hidden_states,
            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )
        (_, real_features, real_features_ori) = teacher_transformer(
            sample_real,
            timestep,
            encoder_hidden_states,
            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )

    fake_outputs = discriminator(fake_features_ori)
    real_outputs = discriminator(real_features_ori)
    for fake_output, real_output in zip(fake_outputs, real_outputs):
        loss += (
            torch.mean(weight * torch.relu(fake_output.float() + 1))
            + torch.mean(weight * torch.relu(1 - real_output.float()))
        ) / (discriminator.head_num * discriminator.num_h_per_head)
    return loss


def gan_g_loss(
    discriminator,
    teacher_transformer,
    sample_fake,
    sample_real,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    weight,
    discriminator_head_stride,
):
    loss = 0.0
    (_, features, features_ori) = teacher_transformer(
        sample_fake,
        timestep,
        encoder_hidden_states,

        output_features=True,
        output_features_stride=2,
        return_dict=False,
        final_layer=True,
        unpachify_layer=True,
        student=False,
    )

    with torch.no_grad():
        (_, features_real,features_real_ori) = teacher_transformer(
            sample_real,
            timestep,
            encoder_hidden_states,

            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )
    
    loss_feat = torch.nn.functional.mse_loss(features,features_real) * 10.0

    fake_outputs = discriminator(features_ori,)
    for fake_output in fake_outputs:
        loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
            discriminator.head_num * discriminator.num_h_per_head
        )
    loss = loss * 5.0
    return loss + loss_feat


def distill_one_step_adv(
    transformer,
    model_type,
    teacher_transformer,
    optimizer,
    discriminator,
    discriminator_optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    solver,
    noise_random_generator,
    sp_size,
    max_grad_norm,
    uncond_prompt_embed,
    uncond_prompt_mask,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    adv_weight,
    discriminator_head_stride,
    cur_step,
):
    total_loss = 0.0
    optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0,  # codespell:ignore
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }
    for _ in range(1):
        (
            latents,
            encoder_hidden_states,
            latents_attention_mask,
            encoder_attention_mask,
        ) = next(loader)
        model_input = normalize_dit_input(model_type, latents)

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        index = torch.randint(0,
                              35, (bsz, ),
                              device=model_input.device).long()
        if sp_size > 1:
            broadcast(index)
        # Add noise according to flow matching.
        sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
        sigmas_prev = extract_into_tensor(solver.sigmas_prev, index,
                                          model_input.shape)

        timesteps = (sigmas *
                     noise_scheduler.config.num_train_timesteps).view(-1)
                
        # if squeeze to [], unsqueeze to [1]
        timesteps_prev = (sigmas_prev *
                          noise_scheduler.config.num_train_timesteps).view(-1)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

        # Predict the noise residual
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_kwargs = {
                "hidden_states": noisy_model_input,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timesteps,
                "return_dict": False,
            }

            model_pred = transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                return_dict=False,
                student=True,
            )[0]


        model_pred, end_index = solver.euler_style_multiphase_pred(
                noisy_model_input, model_pred, index, multiphase
            )  # x37
        
        adv_index = torch.empty_like(index)
        for i in range(end_index.size(0)):
            adv_index[i] = torch.randint(
                end_index[i].item(),
                end_index[i].item() + 35,
                (1, ),
                dtype=end_index.dtype,
                device=end_index.device,
            )

        sigmas_end = extract_into_tensor(solver.sigmas_prev, end_index, model_input.shape)
        sigmas_adv = extract_into_tensor(solver.sigmas_prev, adv_index, model_input.shape)
        timesteps_adv = (sigmas_adv * noise_scheduler.config.num_train_timesteps).view(-1)
        # ###########################################

        with torch.no_grad():
            w = distill_cfg
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cond_teacher_output = teacher_transformer(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                    student=False,
                )[0].float()
                
            if not_apply_cfg_solver:
                uncond_teacher_output = cond_teacher_output
            else:
                # Get teacher model prediction on noisy_latents and unconditional embedding
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    uncond_teacher_output = teacher_transformer(
                        noisy_model_input,
                        timesteps,
                        uncond_prompt_embed.unsqueeze(0).expand(bsz, -1, -1),
                        return_dict=False,
                        student=False,
                    )[0].float()

            teacher_output = cond_teacher_output + w * (cond_teacher_output -
                                                        uncond_teacher_output)
            x_prev = solver.euler_step(noisy_model_input, teacher_output,
                                       index)

        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target_pred = transformer(
                        x_prev.float(),
                        timesteps_prev,
                        encoder_hidden_states,
                        return_dict=False,
                        student=True,
                    )[0]

            target, end_index = solver.euler_style_multiphase_pred(
                    x_prev, target_pred, index, multiphase, True
                )

        adv_noise = torch.randn_like(target)
        real_adv = (
            (1 - sigmas_adv) * target + (sigmas_adv - sigmas_end) * adv_noise
        ) / (1 - sigmas_end)
        fake_adv = (
            (1 - sigmas_adv) * model_pred 
            + (sigmas_adv - sigmas_end) * adv_noise
        ) / (1 - sigmas_end)

        huber_c = 0.001
        g_loss = torch.mean(
            torch.sqrt((model_pred.float() - target.float())**2 + huber_c**2) -
            huber_c)
        
        if True:
            discriminator.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g_gan_loss = adv_weight * gan_g_loss(
                    discriminator,
                    teacher_transformer,
                    fake_adv.float(),
                    real_adv.float(),
                    timesteps_adv,
                    encoder_hidden_states.float(),
                    encoder_attention_mask,
                    1.0,
                    discriminator_head_stride,
                )
            g_loss += g_gan_loss
        
        g_loss.backward()

        g_loss = g_loss.detach().clone()
        dist.all_reduce(g_loss, op=dist.ReduceOp.AVG)

        g_grad_norm = transformer.clip_grad_norm_(max_grad_norm).item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if True:
            discriminator_optimizer.zero_grad()
            discriminator.requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d_loss = gan_d_loss(
                    discriminator,
                    teacher_transformer,
                    fake_adv.detach(),
                    real_adv.detach(),
                    timesteps_adv,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    1.0,
                    discriminator_head_stride,
                )

        d_loss.backward()
        d_grad_norm = discriminator.clip_grad_norm_(max_grad_norm).item()
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()


    return g_loss, g_grad_norm, d_loss, d_grad_norm


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")

    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    transformer.__class__.forward  = wan_forward
    transformer.__class__.wan_forward_origin = wan_forward_origin
    # teacher_transformer = deepcopy(transformer)
    teacher_transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        "pretrained/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    discriminator = Discriminator(
        args.discriminator_head_stride,
        total_layers=30,
    )

    # transformer.requires_grad_(False)
    transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    transformer.add_adapter(transformer_lora_config, adapter_name="lora1")
    transformer.add_layer()
    transformer.init_layer()

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        True, #args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    print(fsdp_kwargs, no_split_modules)
    discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(
        args.master_weight_type)

    # if args.use_lora:
    transformer.config.lora_rank = args.lora_rank
    transformer.config.lora_alpha = args.lora_alpha
    transformer.config.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0"
        ]
    transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ] #no_split_modules
    fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )
    teacher_transformer = FSDP(
        teacher_transformer,
        **fsdp_kwargs,
    )
    discriminator = FSDP(
        discriminator,
        **discriminator_fsdp_kwargs,
    )
    main_print("--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules,
                                 args.selective_checkpointing)
        apply_fsdp_checkpointing(teacher_transformer, no_split_modules,
                                 args.selective_checkpointing)
    # Set model as trainable.
    transformer.train()
    teacher_transformer.requires_grad_(False)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)

    if args.scheduler_type == "pcm_linear_quadratic":
        linear_steps = int(noise_scheduler.config.num_train_timesteps *
                           args.linear_range)
        sigmas = linear_quadratic_schedule(
            noise_scheduler.config.num_train_timesteps,
            args.linear_quadratic_threshold,
            linear_steps,
        )
        sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
    else:
        sigmas = noise_scheduler.sigmas
    solver = EulerSolver(
        sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
    )
    solver.to(device)
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(0, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer)
    elif args.resume_from_checkpoint:
        (
            transformer,
            optimizer,
            discriminator,
            discriminator_optimizer,
            init_steps,
        ) = resume_training_generator_discriminator(
            transformer,
            optimizer,
            discriminator,
            discriminator_optimizer,
            args.resume_from_checkpoint,
            rank,
        )
    main_print(f"optimizer: {optimizer}")

    # todo add lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    from fastvideo.dataset.t2v_datasets import WANVideoDataset
    train_dataset = WANVideoDataset(height=480,width=832,fps=16,max_num_frames=81)

    uncond_prompt_embed = train_dataset.uncond_prompt_embed
    uncond_prompt_mask = train_dataset.uncond_prompt_mask
    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps *
        args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps /
                        args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(
        f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError(
            "resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    def get_num_phases(multi_phased_distill_schedule, step):
        # step-phase,step-phase
        multi_phases = multi_phased_distill_schedule.split(",")
        phase = multi_phases[-1].split("-")[-1]
        for step_phases in multi_phases:
            phase_step, phase = step_phases.split("-")
            if step <= int(phase_step):
                return int(phase)
        return phase

    for i in range(init_steps):
        _ = next(loader)
    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        assert args.multi_phased_distill_schedule is not None
        num_phases = get_num_phases(args.multi_phased_distill_schedule, step)

        (
            generator_loss,
            generator_grad_norm,
            discriminator_loss,
            discriminator_grad_norm,
        ) = distill_one_step_adv(
            transformer,
            args.model_type,
            teacher_transformer,
            optimizer,
            discriminator,
            discriminator_optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            solver,
            noise_random_generator,
            args.sp_size,
            args.max_grad_norm,
            uncond_prompt_embed,
            uncond_prompt_mask,
            args.num_euler_timesteps,
            num_phases,
            args.not_apply_cfg_solver,
            args.distill_cfg,
            args.adv_weight,
            args.discriminator_head_stride,
            step,
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix({
            "g_loss": f"{generator_loss:.4f}",
            "d_loss": f"{discriminator_loss:.4f}",
            "g_grad_norm": generator_grad_norm,
            "d_grad_norm": discriminator_grad_norm,
            "step_time": f"{step_time:.2f}s",
        })
        progress_bar.update(1)
        if rank <= 0:
            wandb.log(
                {
                    "generator_loss": generator_loss,
                    "discriminator_loss": discriminator_loss,
                    "generator_grad_norm": generator_grad_norm,
                    "discriminator_grad_norm": discriminator_grad_norm,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                },
                step=step,
            )
        if False: #step==1: #step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank,
                                     args.output_dir, step)
            else:
                # Your existing checkpoint saving code
                if args.use_ema:
                    save_checkpoint(ema_transformer, rank, args.output_dir, step)
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()
        if args.log_validation and (step % 10 == 0 or step==1):
            log_validation(
                args,
                transformer,
                device,
                torch.bfloat16,
                step,
            )

    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                             args.max_train_steps)
    else:
        save_checkpoint(transformer, rank, args.output_dir,
                        args.max_train_steps)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type",
                        type=str,
                        default="mochi",
                        help="The type of model to train.")

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--validation_sampling_steps", type=str, default="64")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")

    parser.add_argument("--validation_steps", type=float, default=64)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
         " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--discriminator_learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help=
        "Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size",
                        type=int,
                        default=1,
                        help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=256,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=128,
                        help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument("--distill_cfg",
                        type=float,
                        default=5.0,
                        help="Distillation coefficient.")
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type",
                        type=str,
                        default="pcm",
                        help="The scheduler type to use.")
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="Threshold for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--adv_weight",
        type=float,
        default=0.1,
        help="The weight of the adversarial loss.",
    )
    parser.add_argument(
        "--discriminator_head_stride",
        type=int,
        default=2,
        help="The stride of the discriminator head.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.001,
                        help="Weight decay to apply.")
    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA.")
    parser.add_argument("--multi_phased_distill_schedule",
                        type=str,
                        default=None)
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument("--hunyuan_teacher_disable_cfg", action="store_true")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)
