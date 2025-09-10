from causvid.models.wan.causal_stream_inference_pipe import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import numpy as np

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

import queue
import threading

TAG_LATENT_HDR = 11001
TAG_LATENT_PAY = 11002
TAG_START_END_STEP = 11003
TAG_PATCHED_X_SHAPE = 11004
TAG_LATENT_ORIGIN_HDR = 11005
TAG_LATENT_ORIGIN_PAY = 11006

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

    return video  # [C, T, H, W]

def send_latents_fn(i, latents, original_latents, patched_x_shape, current_start, current_end, current_step, device, rank, outstanding):
    # build header on GPU (NCCL requires GPU tensors for send/recv)
    bsz, slen, cch = latents.shape
    header = torch.tensor([i, bsz, slen, cch], dtype=torch.int64, device=device)

    bsz, cch, tlen, hh, ww = original_latents.shape
    header_origin = torch.tensor([i, bsz, cch, tlen, hh, ww], dtype=torch.int64, device=device)

    # non-blocking sends
    work_h = dist.isend(header, dst=rank+1, tag=TAG_LATENT_HDR)
    work_p = dist.isend(latents, dst=rank+1, tag=TAG_LATENT_PAY)
    work_h_0 = dist.isend(header_origin, dst=rank+1, tag=TAG_LATENT_ORIGIN_HDR)
    work_p_0 = dist.isend(original_latents, dst=rank+1, tag=TAG_LATENT_ORIGIN_PAY)
    work_shape = dist.isend(patched_x_shape, dst=rank+1, tag=TAG_PATCHED_X_SHAPE)
    work_s = dist.isend(torch.tensor([current_start, current_end, current_step], dtype=torch.int64, device=device), dst=rank+1, tag=TAG_START_END_STEP)

    # keep references until send completes
    outstanding.append((work_h, work_p, work_h_0, work_p_0, work_shape, work_s))

def receiver_thread_fn(rank, num_chuncks, device, free_buffers, free_buffers_origin, decode_queue, receiver_stop):
    for _ in range(num_chuncks):
        # receive header first (on GPU)
        lhdr = torch.empty(4, dtype=torch.int64, device=device)
        dist.recv(lhdr, src=rank-1, tag=TAG_LATENT_HDR)

        ci, bsz, slen, cch = [int(x) for x in lhdr.tolist()]
        shape = (bsz, slen, cch)

        # get or allocate buffer (reuse if possible)
        buf = None
        if shape in free_buffers and len(free_buffers[shape]) > 0:
            buf = free_buffers[shape].pop()
        else:
            # allocate new on GPU with same dtype as sent
            buf = torch.empty(shape, dtype=torch.bfloat16, device=device)

        # blocking recv into buf
        dist.recv(buf, src=rank-1, tag=TAG_LATENT_PAY)

        # receive header first (on GPU)
        lhdr = torch.empty(6, dtype=torch.int64, device=device)
        dist.recv(lhdr, src=rank-1, tag=TAG_LATENT_ORIGIN_HDR)

        ci, bsz, cch, tlen, hh, ww = [int(x) for x in lhdr.tolist()]
        shape = (bsz, cch, tlen, hh, ww)

        # get or allocate buffer (reuse if possible)
        buf_0 = None
        if shape in free_buffers_origin and len(free_buffers_origin[shape]) > 0:
            buf_0 = free_buffers_origin[shape].pop()
        else:
            # allocate new on GPU with same dtype as sent
            buf_0 = torch.empty(shape, dtype=torch.bfloat16, device=device)

        # blocking recv into buf
        dist.recv(buf_0, src=rank-1, tag=TAG_LATENT_ORIGIN_PAY)

        patched_x_shape = torch.empty(5, dtype=torch.int64, device=device)
        start_end_step = torch.empty(3, dtype=torch.int64, device=device)
        dist.recv(patched_x_shape, src=rank-1, tag=TAG_PATCHED_X_SHAPE)

        dist.recv(start_end_step, src=rank-1, tag=TAG_START_END_STEP)
        current_start, current_end, current_step = [int(x) for x in start_end_step.tolist()]

        # put into decode queue (this will block if decode queue is full)
        decode_queue.put((ci, buf, buf_0, current_start, current_end, current_step, patched_x_shape))

    # received all expected chunks -> signal done
    receiver_stop.set()

def init_distributed():
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)

def main():
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
    parser.add_argument("--fold", action="store_true", default=False)
    parser.add_argument("--max_outstanding", type=int, default=2, help="max number of outstanding sends/recv to keep")
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    assert world_size >= 2, "world_size must be at least 2"

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v

    if rank == 0:
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
        if input_video_original.dtype != torch.bfloat16:
            input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
        print(f"Input video tensor shape: {input_video_original.shape}")
        b, c, t, h, w = input_video_original.shape
    else:
        input_video_original = None
        b = c = t = h = w = 0

    chunck_size = 4
    if rank == 0:
        num_chuncks = (t - 1) // chunck_size
    else:
        num_chuncks = 0
    num_chuncks_tensor = torch.tensor([num_chuncks], dtype=torch.int64, device=device)
    dist.broadcast(num_chuncks_tensor, src=0)
    num_chuncks = int(num_chuncks_tensor.item())

    # Build pipeline (keep existing init style)
    pipeline = CausalStreamInferencePipeline(config, device=str(device))
    pipeline.to(device=str(device), dtype=torch.bfloat16)
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipeline.generator.load_state_dict(state_dict, strict=True)

    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]

    # Precompute indices (rank0 only)
    chunk_meta = []
    if rank == 0:
        for i in range(num_chuncks):
            if i == 0:
                start_idx = 0
                end_idx = 5
                current_start = 0
                current_end = pipeline.frame_seq_length*2
            else:
                start_idx = chunk_meta[-1][1]
                end_idx = start_idx + chunck_size
                current_start = current_end
                current_end = current_end+(chunck_size//4)*pipeline.frame_seq_length
            chunk_meta.append((start_idx, end_idx, current_start, current_end))
    
    total_block_num = []
    total_blocks = 30
    if world_size == 2:
        # New interval format: [start, end), so [0,15] and [15,30]
        total_block_num = [[0, 15], [15, total_blocks]]
    else:
        # Evenly split [0, total_blocks) into world_size contiguous intervals
        base = total_blocks // world_size
        rem = total_blocks % world_size
        start = 0
        for r in range(world_size):
            size = base + (1 if r < rem else 0)
            end = start + size if r < world_size - 1 else total_blocks
            total_block_num.append([start, end])
            start = end

    block_num = total_block_num[rank]

    noise_scale = args.noise_scale
    MAX_OUTSTANDING = args.max_outstanding

    # ----- RANK 0: encoder + async send (isend) -----
    if rank == 0:
        prepared = False
        outstanding = []  # list of (work_obj, work_obj2, latents_ref, header_ref)
        torch.cuda.synchronize()
        start_time = time.time()
        for i, (start_idx, end_idx, current_start, current_end) in enumerate(chunk_meta):
            inp = input_video_original[:, :, start_idx:end_idx]

            l2_dist=(input_video_original[:,:,end_idx-chunck_size:end_idx]-input_video_original[:,:,end_idx-chunck_size-1:end_idx-1])**2
            l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
            noise_scale = (0.8-0.1*l2_dist.item())*0.9+noise_scale*0.1
            current_step = int(1000*noise_scale)-100
            latents = pipeline.vae.model.stream_encode(inp)  # [B, 4, T, H//16, W//16] or so
            latents = latents.transpose(2,1).contiguous().to(dtype=torch.bfloat16)

            if not prepared:
                pipeline.prepare(latents, text_prompts=prompts)
                prepared = True

            noise = torch.randn_like(latents)
            noisy_latents = noise*noise_scale + latents*(1-noise_scale)
            denoised_pred, patched_x_shape = pipeline.inference(
                noise=noisy_latents, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='input',
                block_num=block_num,
            )

            while len(outstanding) >= MAX_OUTSTANDING:
                oldest = outstanding.pop(0)
                # wait for both header & payload to finish
                try:
                    oldest[0].wait()
                    oldest[1].wait()
                    oldest[2].wait()
                    oldest[3].wait()
                    oldest[4].wait()
                    oldest[5].wait()
                except Exception:
                    raise Exception(f"Error waiting for outstanding chunks {i}")

            send_latents_fn(i, denoised_pred, noisy_latents, patched_x_shape, current_start, current_end, current_step, device, rank, outstanding)

            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            print(f"[rank 0] Encode time: {t:.4f} s, fps: {inp.shape[2]/t:.4f}")
            start_time = end_time

        for (w_h, w_p, w_h_0, w_p_0, w_shape, w_s) in outstanding:
            w_h.wait()
            w_p.wait()
            w_h_0.wait()
            w_p_0.wait()
            w_s.wait()
            w_shape.wait()

    # ----- RANK {world_size - 1}: receiver thread(s) + decode loop -----
    elif rank == world_size - 1:
        os.makedirs(args.output_folder, exist_ok=True)
        prepared = False
        results = {}

        # queue for handing received latents to decoder
        decode_queue = queue.Queue(maxsize=MAX_OUTSTANDING)

        # buffer pool: map from shape tuple -> list of tensors
        free_buffers = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}
        free_buffers_origin = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}

        receiver_stop = threading.Event()

        # start receiver thread
        rcv_thread = threading.Thread(target=receiver_thread_fn, args=(rank,num_chuncks, device, free_buffers, free_buffers_origin, decode_queue, receiver_stop), daemon=True)
        rcv_thread.start()

        processed = 0
        torch.cuda.synchronize()
        start_time = time.time()
        while processed < num_chuncks:
            ci, latents, latents_origin, current_start, current_end, current_step, patched_x_shape = decode_queue.get()  # wait until one chunk available

            shape = tuple(latents.shape)

            if not prepared:
                # pipeline.prepare will likely initialize caches on rank1
                pipeline.prepare(latents, text_prompts=prompts)
                prepared = True

            denoised_pred = pipeline.inference(
                noise=latents_origin, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='output',
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=latents,
            )

            video = pipeline.vae.stream_decode_to_pixel(denoised_pred)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()

            results[ci] = video.cpu().float().numpy()
            processed += 1

            # return buffer to free pool for reuse
            if shape not in free_buffers:
                free_buffers[shape] = []
            free_buffers[shape].append(latents)

            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            print(f"[rank {rank}] Decode time: {t:.4f} s, fps: {video.shape[0]/t:.4f}")
            start_time = end_time

        # wait receiver thread to exit cleanly (should have finished num_chuncks)
        rcv_thread.join(timeout=1.0)

        video_list = [results[i] for i in range(num_chuncks)]
        video = np.concatenate(video_list, axis=0)
        export_to_video(
            video, os.path.join(args.output_folder, f"output_{0:03d}.mp4"), fps=args.fps)
        print(f"Video saved to: {os.path.join(args.output_folder, f'output_{0:03d}.mp4')}")

    # ----- RANK {middle rank}: receiver thread(s) + dit blocks + sender -----
    else:
        os.makedirs(args.output_folder, exist_ok=True)
        prepared = False
        outstanding = []  # list of (work_obj, work_obj2, latents_ref, header_ref)
        results = {}

        # queue for handing received latents to decoder
        decode_queue = queue.Queue(maxsize=MAX_OUTSTANDING)

        # buffer pool: map from shape tuple -> list of tensors
        free_buffers = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}
        free_buffers_origin = {} 

        receiver_stop = threading.Event()

        # start receiver thread
        rcv_thread = threading.Thread(target=receiver_thread_fn, args=(rank,num_chuncks, device, free_buffers, free_buffers_origin, decode_queue, receiver_stop), daemon=True)
        rcv_thread.start()

        processed = 0
        torch.cuda.synchronize()
        start_time = time.time()
        while processed < num_chuncks:
            i, latents, latents_origin, current_start, current_end, current_step, patched_x_shape = decode_queue.get()  # wait until one chunk available

            shape = tuple(latents.shape)

            if not prepared:
                # pipeline.prepare will likely initialize caches on rank1
                pipeline.prepare(latents, text_prompts=prompts)
                prepared = True

            denoised_pred, _ = pipeline.inference(
                noise=latents, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='middle',
                block_num=block_num,
                patched_x_shape=patched_x_shape,
            )

            processed += 1

            while len(outstanding) >= MAX_OUTSTANDING:
                oldest = outstanding.pop(0)
                # wait for both header & payload to finish
                try:
                    oldest[0].wait()
                    oldest[1].wait()
                    oldest[2].wait()
                    oldest[3].wait()
                    oldest[4].wait()
                    oldest[5].wait()
                except Exception:
                    raise Exception(f"Error waiting for outstanding chunks {i}")

            send_latents_fn(i, denoised_pred, latents_origin, patched_x_shape, current_start, current_end, current_step, device, rank, outstanding)

            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            print(f"[rank {rank}] Encode time: {t:.4f} s, fps: {chunck_size/t:.4f}")
            start_time = end_time

    dist.barrier()



if __name__ == "__main__":
    main()