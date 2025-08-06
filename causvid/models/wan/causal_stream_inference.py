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
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
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
    
    def prepare(self, noise: torch.Tensor, text_prompts: List[str]):
        batch_size = noise.shape[0]
        self.conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        if batch_size > 1:
            self.conditional_dict['prompt_embeds'] = self.conditional_dict['prompt_embeds'].repeat(batch_size, 1, 1)

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False


    def inference(self, noise: torch.Tensor, current_start: int, current_end: int) -> torch.Tensor:
        batch_size = noise.shape[0]

        # Step 2.1: Spatial denoising loop
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

        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=self.conditional_dict,
            timestep=timestep * 0,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start,
            current_end=current_end
        )

        # Step 3: Decode the output
        # video = self.vae.stream_decode_to_pixel(denoised_pred)    
        # video = (video * 0.5 + 0.5).clamp(0, 1)

        return denoised_pred
