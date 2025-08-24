from causvid.models.wan.causal_stream_inference_onestep import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import torch.distributed as dist

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

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
    # assert 8 / 16 <= aspect_ratio <= 17 / 16, (
    #     f"Unsupported aspect ratio: {aspect_ratio:.2f} for shape {video.shape}"
    # )
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

def generate():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--prompt_file_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--noise_scale", type=float, default=0.700)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--memory_snapshot", action="store_true", default=False)
    parser.add_argument("--fold", action="store_true", default=False)

    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    config = OmegaConf.load(args.config_path)
    # Add all command-line args into the config for downstream use
    for k, v in vars(args).items():
        config[k] = v

    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history()

    pipeline = CausalStreamInferencePipeline(config, device="cuda")
    pipeline.to(device="cuda", dtype=torch.bfloat16)

    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")['generator']

    pipeline.generator.load_state_dict(
        state_dict, strict=True
    )

    # Only rank 0 loads and processes the input video
    input_video_original = None
    if rank == 0:
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0) # [1, C, T, H, W]
        input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device="cuda")
        print(input_video_original.shape)

    # Broadcast video shape info to all ranks
    if world_size > 1:
        if rank == 0:
            video_shape = torch.tensor(input_video_original.shape, dtype=torch.long, device="cuda")
        else:
            video_shape = torch.zeros(5, dtype=torch.long, device="cuda")
        dist.broadcast(video_shape, src=0)
        
        if rank != 0:
            # Create dummy tensor with same shape for non-rank-0 processes
            input_video_original = torch.zeros(tuple(video_shape.cpu().numpy()), dtype=torch.bfloat16, device="cuda")

    chunck_size = 4
    num_chuncks = (input_video_original.shape[2]-1) // chunck_size

    dataset = TextDataset(args.prompt_file_path)
    os.makedirs(args.output_folder, exist_ok=True)
    prompts = [dataset[0]]
    
    video_list = []
    cost_time = 0
    noise_scale = args.noise_scale

    for i in range(num_chuncks):
        torch.cuda.synchronize()
        start_time = time.time()

        if i==0:
            start_idx = 0
            end_idx = 5
            current_start = 0
            current_end = pipeline.frame_seq_length*2//world_size
        else:
            start_idx = end_idx
            end_idx = end_idx+chunck_size
            current_start = current_end
            current_end = current_end+(chunck_size//4)*pipeline.frame_seq_length//world_size

        input_video = input_video_original[:,:,start_idx:end_idx]

        if rank == 0:
            l2_dist=(input_video_original[:,:,end_idx-chunck_size:end_idx]-input_video_original[:,:,end_idx-chunck_size-1:end_idx-1])**2
            l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
            noise_scale = (0.8-0.1*l2_dist.item())*0.9+noise_scale*0.1
            current_step = int(1000*noise_scale)-100
        else:
            current_step = 0
            noise_scale = args.noise_scale

        if world_size > 1:
            noise_scale_tensor = torch.tensor([noise_scale, current_step], dtype=torch.float32, device="cuda")
            dist.broadcast(noise_scale_tensor, src=0)
            if rank != 0:
                noise_scale = noise_scale_tensor[0].item()
                current_step = int(noise_scale_tensor[1].item())

        if rank == 0:
            latents = pipeline.vae.model.stream_encode(input_video) 
            latents = latents.transpose(2,1)
            latents = latents.contiguous()
            latents_shape = torch.tensor(latents.shape, dtype=torch.long, device=device)
        else:
            latents_shape = torch.zeros(5, dtype=torch.long, device=device)
        
        dist.broadcast(latents_shape, src=0)
        if rank != 0:
            latents = torch.zeros(tuple(latents_shape), dtype=torch.bfloat16, device=device)
        dist.broadcast(latents, src=0)
            
        if i==0:
            pipeline.prepare(latents, text_prompts=prompts)
        noise = torch.randn_like(latents)
        noisy_latents = noise*noise_scale + latents*(1-noise_scale)
        
        denoised_pred = pipeline.inference(
            noise=noisy_latents, # [1, 4, 16, 16, 60]
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
        )

        video_output = None
        if rank == 0:
            video = pipeline.vae.stream_decode_to_pixel(denoised_pred)  
            
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video.permute(0,2,1,3,4)

            video_output = video[0].permute(0, 2, 3, 1).cpu().numpy()
            
            torch.cuda.synchronize()
            cost_time=time.time()-start_time
            T=video_output.shape[0]
            print(f"Time taken: {cost_time:.4f} seconds, {T} frames, FPS: {T/cost_time:.4f}")

            video_list.append(video_output)
        else:
            torch.cuda.synchronize()
            cost_time=time.time()-start_time

        if args.memory_snapshot and i==3:
            torch.cuda.memory._dump_snapshot("vae_snapshot.pickle")   
            torch.cuda.memory._record_memory_history(enabled=None)
            exit(0)

    if rank == 0:
        video = np.concatenate(video_list, axis=0)

        export_to_video(
            video, os.path.join(args.output_folder, f"output_{0:03d}.mp4"), fps=args.fps)

if __name__ == "__main__":
    generate()
