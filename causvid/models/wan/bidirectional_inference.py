from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch


class BidirectionalInferencePipeline(torch.nn.Module):
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

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(self, noise: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # initial point
        noisy_image_or_video = noise

        for index, current_timestep in enumerate(self.denoising_step_list):
            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device)

                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video
