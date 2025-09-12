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

def compute_noise_scale_and_step(input_video_original: torch.Tensor, end_idx: int, chunck_size: int, noise_scale: float):
    l2_dist=(input_video_original[:,:,end_idx-chunck_size:end_idx]-input_video_original[:,:,end_idx-chunck_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    new_noise_scale = (0.8-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step


def compute_balanced_split(total_blocks: int, rank_times: list[float], dit_times: list[float], current_block_nums: list[list[int]]) -> list[list[int]]:
    """
    Compute new block splits for all ranks to balance total rank times.
    
    Args:
        total_blocks: Total number of DiT blocks
        rank_times: List of total iteration times for each rank [t_rank0, t_rank1, ..., t_rankN] (DiT + VAE time)
        dit_times: List of pure DiT inference times for each rank [dit_rank0, dit_rank1, ..., dit_rankN] (DiT time only)
        current_block_nums: List of current block_num format for each rank [[rank0_blocks], [rank1_blocks], ...]
        
    Returns:
        List of new block_num format for each rank, matching the original format:
        - For world_size == 2: [[end_idx_rank0], [start_idx_rank1]]
        - For world_size > 2: [[end_idx_rank0], [start1, end1], [start2, end2], ..., [start_idx_last]]
        Note: Numbers are shared across ranks (rank0_end = rank1_start, rank1_end = rank2_start, etc.)
    """
    num_ranks = len(rank_times)
    if num_ranks == 0 or num_ranks != len(current_block_nums) or num_ranks != len(dit_times):
        return current_block_nums

    # Step 1: Calculate total DiT time and per-block DiT time
    total_dit_time = sum(dit_times)
    dit_time_per_block = total_dit_time / total_blocks
    
    # Step 2: Calculate average rank time
    avg_rank_time = sum(rank_times) / num_ranks
    
    # Step 3: Extract current block counts from current_block_nums (all ranks use [start, end) now)
    current_block_counts = []
    for block_num in current_block_nums:
        # block_num: [start, end) exclusive end
        start_idx, end_idx = int(block_num[0]), int(block_num[1])
        current_block_counts.append(max(0, end_idx - start_idx))
    
    # Step 4: Calculate target block counts based on time differences
    target_blocks = []
    for i in range(num_ranks):
        time_diff = avg_rank_time - rank_times[i]  # positive = needs more time, negative = needs less time
        block_adjustment = time_diff / dit_time_per_block  # convert time difference to block count
        target_count = current_block_counts[i] + block_adjustment
        # Allow zero-length intervals by clamping at 0 (upper bound will be enforced by sum adjustment)
        target_count = max(0, int(round(target_count)))
        target_blocks.append(target_count)
    
    # Step 5: Adjust to ensure total blocks sum to total_blocks
    current_total = sum(target_blocks)
    if current_total != total_blocks:
        diff = total_blocks - current_total
        # When adding, give to ranks with smallest counts first; when removing, take from largest counts first
        if diff > 0:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i))
        else:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i), reverse=True)
        i = 0
        while diff != 0 and num_ranks > 0:
            idx = order[i % num_ranks]
            if diff > 0:
                target_blocks[idx] += 1
                diff -= 1
            else:
                if target_blocks[idx] > 0:
                    target_blocks[idx] -= 1
                    diff += 1
            i += 1
    
    # Step 6: Convert target block counts to contiguous [start, end) intervals from 0 to total_blocks
    # Step 6: Build contiguous [start, end) intervals whose union length equals total_blocks
    new_block_nums = []
    running_start = 0
    for i in range(num_ranks):
        block_count = int(target_blocks[i])
        start_idx = running_start
        end_idx = start_idx + block_count
        # Guard (should not trigger if sums are correct)
        if end_idx > total_blocks:
            end_idx = total_blocks
        new_block_nums.append([start_idx, end_idx])
        running_start = end_idx
    
    return new_block_nums

def broadcast_kv_blocks(pipeline, block_indices: list[int], donor_rank: int):
    """
    Broadcast kv_cache1 entries for the specified block indices from donor_rank to all ranks.
    This ensures the receiver rank has the up-to-date KV cache when ownership moves.
    """
    if len(block_indices) == 0:
        return
    frame_seq_length = pipeline.frame_seq_length
    rank = dist.get_rank()
    for bi in block_indices:
        dist.broadcast(pipeline.kv_cache1[bi]['k'], src=donor_rank)
        dist.broadcast(pipeline.kv_cache1[bi]['v'], src=donor_rank)
        dist.broadcast(pipeline.kv_cache1[bi]['global_end_index'], src=donor_rank)
        dist.broadcast(pipeline.kv_cache1[bi]['local_end_index'], src=donor_rank)
        pipeline.kv_cache1[bi]['global_end_index'] += frame_seq_length * (donor_rank - rank)


def compute_block_owners(block_intervals: torch.Tensor, total_blocks: int) -> torch.Tensor:
    """
    Given block intervals in [start, end) format for all ranks (shape: [world_size, 2]),
    return a tensor of length total_blocks where each entry is the owner rank of that block index.
    Empty intervals [s,s] are handled (no ownership change for that range).
    """
    world_size = block_intervals.shape[0]
    owners = torch.full((total_blocks,), -1, dtype=torch.int64, device=block_intervals.device)
    for r in range(world_size):
        s = int(block_intervals[r, 0].item())
        e = int(block_intervals[r, 1].item())
        if e > s:
            owners[s:e] = r
    return owners

def rebalance_kv_cache_by_diff(pipeline, old_block_intervals: torch.Tensor, new_block_intervals: torch.Tensor, total_blocks: int):
    """
    Compare ownership from old to new intervals and broadcast KV cache for blocks whose owner changes.
    For each moved block i, use the previous owner's rank as src to broadcast
    pipeline.kv_cache1[i]['k'/'v'/...] to all ranks so the new owner has the correct state.
    """
    old_owners = compute_block_owners(old_block_intervals, total_blocks)
    new_owners = compute_block_owners(new_block_intervals, total_blocks)

    moved_by_src = {}
    for i in range(total_blocks):
        o = int(old_owners[i].item())
        n = int(new_owners[i].item())
        if o != n and o >= 0:
            if o not in moved_by_src:
                moved_by_src[o] = []
            moved_by_src[o].append(i)

    dist.barrier()
    # Broadcast per donor rank (can batch multiple blocks per src)
    for src, blocks in moved_by_src.items():
        broadcast_kv_blocks(pipeline, blocks, donor_rank=src)

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

    parser.add_argument("--schedule_block", action="store_true", default=False)
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
        total_block_num = [[0, 25], [25, total_blocks]]
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

    block_num = torch.tensor(total_block_num, dtype=torch.int64, device=device)
    schedule_block = args.schedule_block

    t_dit = 100.
    t_total = 100.

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

            noise_scale, current_step = compute_noise_scale_and_step(input_video_original, end_idx, chunck_size, noise_scale)

            latents = pipeline.vae.model.stream_encode(inp)  # [B, 4, T, H//16, W//16] or so
            latents = latents.transpose(2,1).contiguous().to(dtype=torch.bfloat16)

            if not prepared:
                pipeline.prepare(latents, text_prompts=prompts)
                prepared = True

            noise = torch.randn_like(latents)
            noisy_latents = noise*noise_scale + latents*(1-noise_scale)
            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()

            denoised_pred, patched_x_shape = pipeline.inference(
                noise=noisy_latents, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='input',
                block_num=block_num[rank],
            )

            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < t_dit:
                    t_dit = temp

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

            if t < t_total:
                t_total = t

            if schedule_block and i==3+rank:
                t_total = torch.tensor(t_total, dtype=torch.float32, device=device)
                t_dit = torch.tensor(t_dit, dtype=torch.float32, device=device)

                gather_blocks = [torch.zeros_like(t_dit, dtype=torch.float32, device=device) for _ in range(world_size)]
                dist.all_gather(gather_blocks, t_dit)

                dist.all_gather(gather_blocks, t_total)

                new_block_num = block_num.clone()

                dist.broadcast(new_block_num, src=world_size - 1)

                rebalance_kv_cache_by_diff(pipeline, block_num, new_block_num, total_blocks)

                block_num = new_block_num

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

            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()

            denoised_pred = pipeline.inference(
                noise=latents_origin, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='output',
                block_num=block_num[rank],
                patched_x_shape=patched_x_shape,
                block_x=latents,
            )

            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < t_dit:
                    t_dit = temp

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

            if t < t_total:
                t_total = t

            if schedule_block and ci==3+rank:
                t_total = torch.tensor(t_total, dtype=torch.float32, device=device)
                t_dit = torch.tensor(t_dit, dtype=torch.float32, device=device)

                gather_blocks = [torch.zeros_like(t_dit, dtype=torch.float32, device=device) for _ in range(world_size)]
                dist.all_gather(gather_blocks, t_dit)
                t_dit_list = [t_dit_i.item() for t_dit_i in gather_blocks]

                dist.all_gather(gather_blocks, t_total)
                t_list = [t_i.item() for t_i in gather_blocks]

                new_block_num = torch.tensor(compute_balanced_split(total_blocks, t_list, t_dit_list, block_num), dtype=torch.int64, device=device)
                dist.broadcast(new_block_num, src=world_size - 1)

                rebalance_kv_cache_by_diff(pipeline, block_num, new_block_num, total_blocks)

                block_num = new_block_num

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

            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()

            denoised_pred, _ = pipeline.inference(
                noise=latents, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='middle',
                block_num=block_num[rank],
                patched_x_shape=patched_x_shape,
            )

            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < t_dit:
                    t_dit = temp

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

            if t < t_total:
                t_total = t

            if schedule_block and i==3+rank:
                t_total = torch.tensor(t_total, dtype=torch.float32, device=device)
                t_dit = torch.tensor(t_dit, dtype=torch.float32, device=device)

                gather_blocks = [torch.zeros_like(t_dit, dtype=torch.float32, device=device) for _ in range(world_size)]
                dist.all_gather(gather_blocks, t_dit)

                dist.all_gather(gather_blocks, t_total)

                new_block_num = block_num.clone()

                dist.broadcast(new_block_num, src=world_size - 1)

                rebalance_kv_cache_by_diff(pipeline, block_num, new_block_num, total_blocks)

                block_num = new_block_num

            start_time = end_time

    dist.barrier()



if __name__ == "__main__":
    main()