from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch
import torch.distributed as dist

class CausalStreamInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        scale_size = 16
        self.frame_seq_length = (args.height//scale_size) * (args.width//scale_size)
        self.fold=False
        self.kv_cache_length = self.frame_seq_length*args.num_kv_cache
        self.conditional_dict = None

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.generator.model.to(self.device)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        
        for _ in range(self.num_transformer_blocks):
            cache_length = self.kv_cache_length

            kv_cache1.append({
                "k": torch.zeros([batch_size, cache_length, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, cache_length, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache
    
    def prepare(
        self,
        text_prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        block_mode: str='input',
        noise: torch.Tensor = None,
        current_start: int = 0,
        current_end: int = None,
    ):
        self.device = device
        batch_size = noise.shape[0]

        self.conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device, dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep *
                    torch.ones([batch_size], device="cuda",
                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )

        # Pre-allocate hidden_states tensor to avoid memory allocation during inference
        self.batch_size = len(self.denoising_step_list)

        for i in range(self.num_transformer_blocks):
            self.crossattn_cache[i]['k'] = self.crossattn_cache[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.crossattn_cache[i]['v'] = self.crossattn_cache[i]['v'].repeat(self.batch_size, 1, 1, 1)

        self.hidden_states = torch.zeros(
            (self.batch_size, self.num_frame_per_block, *noise.shape[2:]), dtype=noise.dtype, device=device
        )
        if block_mode == 'output':
            self.block_x = torch.zeros(
                (self.batch_size, self.frame_seq_length, 1536), dtype=noise.dtype, device=device
            )

        self.kv_cache_starts = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        self.kv_cache_ends = torch.zeros(self.batch_size, dtype=torch.long, device=device)

        self.timestep = self.denoising_step_list

        self.conditional_dict['prompt_embeds'] = self.conditional_dict['prompt_embeds'].repeat(self.batch_size, 1, 1)
    
        return denoised_pred

    def inference(self, noise: torch.Tensor, current_start: int, current_end: int, \
        current_step: int, block_mode: str='input', block_num=None,\
            patched_x_shape: torch.Tensor=None, block_x: torch.Tensor=None) -> torch.Tensor:

        # hidden_states should be pre-allocated in prepare, no need to create new tensors
        assert self.hidden_states is not None, "hidden_states should be initialized in prepare"
        
        self.hidden_states[1:] = self.hidden_states[:-1].clone()
        self.hidden_states[0] = noise[0]

        if block_x is not None:
            self.block_x[1:] = self.block_x[:-1].clone()
            self.block_x[0] = block_x[0]

        self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
        self.kv_cache_starts[0] = current_start
        
        self.kv_cache_ends[1:] = self.kv_cache_ends[:-1].clone()
        self.kv_cache_ends[0] = current_end

        if block_mode == 'output':
            denoised_pred = self.generator.forward_output(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=self.block_x
            )

            for i in range(len(self.denoising_step_list) - 1):
                denoised_pred[[i]] = self.scheduler.add_noise(
                    denoised_pred[[i]],
                    torch.randn_like(denoised_pred[[i]]),
                    self.denoising_step_list[i + 1] *
                    torch.ones([1], device="cuda",
                                dtype=torch.long)
                )
            patched_x_shape = None

        else:
            denoised_pred, patched_x_shape = self.generator.forward_input(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape
            ) 

        return denoised_pred, patched_x_shape
