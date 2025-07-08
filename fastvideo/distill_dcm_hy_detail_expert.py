# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import time
import types
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import wandb

from diffusers import FlowMatchEulerDiscreteScheduler
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
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import (
    BaseOutput,
    USE_PEFT_BACKEND,
    check_min_version,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

check_min_version("0.31.0")

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule

from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function

from fastvideo.distill.discriminator import Discriminator
from fastvideo.distill.solver import extract_into_tensor

from fastvideo.utils.checkpoint import (
    resume_lora_optimizer,
    resume_training_generator_discriminator,
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.communications import (
    all_gather,
    all_to_all_4D,
    broadcast,
    sp_parallel_dataloader_wrapper,
)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (
    apply_fsdp_checkpointing,
    get_discriminator_fsdp_kwargs,
    get_dit_fsdp_kwargs,
)
from fastvideo.utils.load import load_transformer
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.validation import log_validation


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

        # if student and int(timestep[0])<=981:
        #     encoder_hidden_states = self.context_embedder_lora(encoder_hidden_states,
        #                                               timestep,
        #                                               encoder_attention_mask)
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
    # loss = 0.0
    # # collate sample_fake and sample_real
    # with torch.no_grad():
    #     fake_features = teacher_transformer(
    #         sample_fake,
    #         encoder_hidden_states,
    #         timestep,
    #         encoder_attention_mask,
    #         output_features=True,
    #         output_features_stride=discriminator_head_stride,
    #         return_dict=False,
    #         student=False,
    #     )[1]
    #     real_features = teacher_transformer(
    #         sample_real,
    #         encoder_hidden_states,
    #         timestep,
    #         encoder_attention_mask,
    #         output_features=True,
    #         output_features_stride=discriminator_head_stride,
    #         return_dict=False,
    #         student=False,
    #     )[1]

    # fake_outputs = discriminator(fake_features)
    # real_outputs = discriminator(real_features)
    # for fake_output, real_output in zip(fake_outputs, real_outputs):
    #     loss += (torch.mean(weight * torch.relu(fake_output.float() + 1)) +
    #              torch.mean(weight * torch.relu(1 - real_output.float()))) / (
    #                  discriminator.head_num * discriminator.num_h_per_head)
    # return loss
    loss = 0.0
    # collate sample_fake and sample_real
    with torch.no_grad():
        (_, fake_features, fake_features_ori) = teacher_transformer(
            sample_fake,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            output_features=True,
            output_features_stride=discriminator_head_stride,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )
        (_, real_features, real_features_ori) = teacher_transformer(
            sample_real,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            output_features=True,
            output_features_stride=discriminator_head_stride,
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
    # loss = 0.0
    # features = teacher_transformer(
    #     sample_fake,
    #     encoder_hidden_states,
    #     timestep,
    #     encoder_attention_mask,
    #     output_features=True,
    #     output_features_stride=discriminator_head_stride,
    #     return_dict=False,
    #     student=False,
    # )[1]
    # fake_outputs = discriminator(features, )
    # for fake_output in fake_outputs:
    #     loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
    #         discriminator.head_num * discriminator.num_h_per_head)
    # return loss
    loss = 0.0
    (_, features, features_ori) = teacher_transformer(
        sample_fake,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        output_features=True,
        output_features_stride=discriminator_head_stride,
        return_dict=False,
        final_layer=True,
        unpachify_layer=True,
        student=False,
    )

    with torch.no_grad():
        (_, features_real,features_real_ori) = teacher_transformer(
            sample_real,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            output_features=True,
            output_features_stride=discriminator_head_stride,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )
    
    loss_feat = torch.nn.functional.mse_loss(features,features_real)

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
    optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    (
        latents,
        encoder_hidden_states,
        latents_attention_mask,
        encoder_attention_mask,
    ) = next(loader)
    model_input = normalize_dit_input(model_type, latents)
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    index = torch.randint(
        0, 38, (bsz, ), device=model_input.device
    ).long()
    if sp_size > 1:
        broadcast(index)
    # Add noise according to flow matching.
    # sigmas = get_sigmas(start_timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
    sigmas_prev = extract_into_tensor(solver.sigmas_prev, index,
                                      model_input.shape)

    timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).view(-1)
    # if squeeze to [], unsqueeze to [1]

    timesteps_prev = (sigmas_prev *
                      noise_scheduler.config.num_train_timesteps).view(-1)
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

    # Predict the noise residual
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            noisy_model_input,
            encoder_hidden_states,
            timesteps,
            encoder_attention_mask,  # B, L
            return_dict=False,
            student=True,
        )[0]

    # if accelerator.is_main_process:
    model_pred, end_index = solver.euler_style_multiphase_pred(
        noisy_model_input, model_pred, index, multiphase)

    # # simplified flow matching aka 0-rectified flow matching loss
    # # target = model_input - noise
    # target = model_input
    adv_index = torch.empty_like(end_index)
    for i in range(end_index.size(0)):
        adv_index[i] = torch.randint(
            end_index[i].item(),
            end_index[i].item() + 38,
            (1, ),
            dtype=end_index.dtype,
            device=end_index.device,
        )

    sigmas_end = extract_into_tensor(solver.sigmas_prev, end_index, model_input.shape)
    sigmas_adv = extract_into_tensor(solver.sigmas_prev, adv_index, model_input.shape)
    timesteps_adv = (sigmas_adv * noise_scheduler.config.num_train_timesteps).view(-1)

    with torch.no_grad():
        w = distill_cfg
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond_teacher_output = teacher_transformer(
                noisy_model_input,
                encoder_hidden_states,
                timesteps,
                encoder_attention_mask,  # B, L
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
                    uncond_prompt_embed.unsqueeze(0).expand(bsz, -1, -1),
                    timesteps,
                    uncond_prompt_mask.unsqueeze(0).expand(bsz, -1),
                    return_dict=False,
                    student=False,
                )[0].float()
        teacher_output = cond_teacher_output + w * (
            cond_teacher_output - uncond_teacher_output
        )
        x_prev = solver.euler_step(noisy_model_input, teacher_output, index)

    # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target_pred = transformer(
                x_prev.float(),
                encoder_hidden_states,
                timesteps_prev,
                encoder_attention_mask,  # B, L
                return_dict=False,
                student=True,
            )[0]

        target, end_index = solver.euler_style_multiphase_pred(
            x_prev, target_pred, index, multiphase, True)

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
    # keep the master weight to float32
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    transformer.__class__.forward  = hy_forward
    transformer.__class__.hy_forward_origin = hy_forward_origin
    teacher_transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        "pretrained/hunyuanvideo-community/HunyuanVideo",
        # args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    discriminator = Discriminator(
        args.discriminator_head_stride,
        total_layers=48 if args.model_type == "mochi" else 40,
    )

    # transformer.requires_grad_(False)
    transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    transformer.add_adapter(transformer_lora_config,adapter_name="lora1")
    transformer.add_layer()
    transformer.init_layer()

    main_print(
        f"  Total transformer parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    # discriminator
    main_print(
        f"  Total discriminator parameters = {sum(p.numel() for p in discriminator.parameters() if p.requires_grad) / 1e6} M"
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
    discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(
        args.master_weight_type)
    # if args.use_lora:
    # assert args.model_type == "mochi", "LoRA is only supported for Mochi model."
    transformer.config.lora_rank = args.lora_rank
    transformer.config.lora_alpha = args.lora_alpha
    transformer.config.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0"
        ]
    
    transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
    fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)
    # transformer._no_split_modules = no_split_modules
    # fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
    #         transformer)

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
        sigmas = linear_quadratic_schedule(
            noise_scheduler.config.num_train_timesteps,
            args.linear_quadratic_threshold)
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

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t,
                                  args.cfg)
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
    assert args.gradient_accumulation_steps == 1
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

    # log_validation(args, transformer, device,
    #             torch.bfloat16, 0, scheduler_type=args.scheduler_type, shift=args.shift, num_euler_timesteps=args.num_euler_timesteps, linear_quadratic_threshold=args.linear_quadratic_threshold,ema=False)
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

        # log_validation(
        #         args,
        #         transformer,
        #         device,
        #         torch.bfloat16,
        #         step,
        #         scheduler_type=args.scheduler_type,
        #         shift=args.shift,
        #         num_euler_timesteps=args.num_euler_timesteps,
        #         linear_quadratic_threshold=args.linear_quadratic_threshold,
        #         linear_range=args.linear_range,
        #         ema=False,
        #     )

        assert args.multi_phased_distill_schedule is not None
        num_phases = get_num_phases(args.multi_phased_distill_schedule, step)
        start_time = time.time()
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
        if False: #step>192 and step % args.checkpointing_steps == 0:
            main_print(f"--> saving checkpoint at step {step}")
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank,
                                     args.output_dir, step)
            else:
                # Your existing checkpoint saving code
                # TODO
                # save_checkpoint_generator_discriminator(
                #     transformer,
                #     optimizer,
                #     discriminator,
                #     discriminator_optimizer,
                #     rank,
                #     args.output_dir,
                #     step,
                # )
                import shutil
                if args.checkpoints_total_limit is not None and local_rank<=0:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                
                    removing_checkpoints = []
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        if os.path.exists(removing_checkpoint):
                            shutil.rmtree(removing_checkpoint) 
                save_checkpoint(transformer, rank, args.output_dir, step)
            main_print(f"--> checkpoint saved at step {step}")
            dist.barrier()
        if args.log_validation and (step % args.validation_steps == 0 or step==1):
            log_validation(
                args,
                transformer,
                device,
                torch.bfloat16,
                step,
                scheduler_type=args.scheduler_type,
                shift=args.shift,
                num_euler_timesteps=args.num_euler_timesteps,
                linear_quadratic_threshold=args.linear_quadratic_threshold,
                linear_range=args.linear_range,
                ema=False,
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
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    # validation & logs
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
    parser.add_argument("--validation_prompt_dir", type=str)
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
    parser.add_argument("--multi_phased_distill_schedule",
                        type=str,
                        default=None)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )

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
                        default=3.0,
                        help="Distillation coefficient.")
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type",
                        type=str,
                        default="pcm",
                        help="The scheduler type to use.")
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
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="The threshold of the linear quadratic scheduler.",
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)
