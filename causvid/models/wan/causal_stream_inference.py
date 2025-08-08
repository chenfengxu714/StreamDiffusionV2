from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch

class CausalStreamInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
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
        self.kv_cache_length = self.frame_seq_length*args.num_kv_cache
        if args.unfold:
            self.frame_seq_length = self.frame_seq_length // 4
        self.conditional_dict = None

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_length, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_length, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.zeros([batch_size], dtype=torch.long, device=device),
                "local_end_index": torch.zeros([batch_size], dtype=torch.long, device=device)
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
        batch_size: int,
        text_prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.hidden_states = None
        self.kv_cache_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.kv_cache_ends = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.device = device

        self.timestep = torch.cat((
            self.denoising_step_list,
            torch.tensor([0], dtype=torch.int64, device=device)
        ))

        self.conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if batch_size > 1:
            self.conditional_dict['prompt_embeds'] = self.conditional_dict['prompt_embeds'].repeat(batch_size, 1, 1)

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


    def inference(self, noise: torch.Tensor, current_start: int, current_end: int) -> torch.Tensor:
        if self.hidden_states is None or self.hidden_states.shape[1] != noise.shape[1]:
            self.hidden_states = torch.zeros(
                (self.batch_size, *noise.shape[1:]), dtype=noise.dtype, device=noise.device
            )
        self.hidden_states = self.roll_states(noise, self.hidden_states)
        self.kv_cache_starts = self.roll_states(
            torch.tensor([current_start], dtype=torch.long, device=self.device),
            self.kv_cache_starts
        )
        self.kv_cache_ends = self.roll_states(
            torch.tensor([current_end], dtype=torch.long, device=self.device),
            self.kv_cache_ends
        )

        for block_index in range(self.num_transformer_blocks):
            self.kv_cache1[block_index]["k"] = self.roll_states(
                self.kv_cache1[block_index]["k"][[0]].clone(),
                self.kv_cache1[block_index]["k"]
            )
            self.kv_cache1[block_index]["v"] = self.roll_states(
                self.kv_cache1[block_index]["v"][[0]].clone(),
                self.kv_cache1[block_index]["v"]
            )
            self.kv_cache1[block_index]["global_end_index"] = self.roll_states(
                self.kv_cache1[block_index]["global_end_index"][[0]].clone(),
                self.kv_cache1[block_index]["global_end_index"]
            )
            self.kv_cache1[block_index]["local_end_index"] = self.roll_states(
                self.kv_cache1[block_index]["local_end_index"][[0]].clone(),
                self.kv_cache1[block_index]["local_end_index"]
            )

        self.hidden_states = self.generator(
            noisy_image_or_video=self.hidden_states,
            conditional_dict=self.conditional_dict,
            timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=self.kv_cache_starts,
            current_end=self.kv_cache_ends
        )

        for i in range(len(self.denoising_step_list) - 1):
            self.hidden_states[[i]] = self.scheduler.add_noise(
                self.hidden_states[[i]],
                torch.randn_like(self.hidden_states[[i]]),
                self.denoising_step_list[i + 1] * torch.ones([1], device="cuda", dtype=torch.long)
            )

        # The last row is only for updating kv cache, so we return the second last row
        return self.hidden_states[[-2]]

    def roll_states(self, new_state: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        states = states[:self.batch_size - 1]
        return torch.cat((new_state, states), dim=0)
