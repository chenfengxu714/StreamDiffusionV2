from causvid.models.wan.causal_stream_inference_onestep import CausalStreamInferencePipeline
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


def process_chunk_with_pipeline(pipeline: CausalStreamInferencePipeline,
                                input_video: torch.Tensor,
                                prompts,
                                noise_scale: float,
                                current_start: int,
                                current_end: int,
                                current_step: int,
                                is_first_chunk: bool,
                                end_idx: int,
                                chunck_size: int = 4) -> np.ndarray:
    device = next(p for p in pipeline.generator.parameters()).device
    input_video = input_video.to(device=device, dtype=torch.bfloat16)

    # l2_dist = (input_video[:,:,end_idx-chunck_size:end_idx] -
    #             input_video[:,:,end_idx-chunck_size-1:end_idx-1])**2
    # l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    # noise_scale = (0.8-0.1*l2_dist.item())*0.9 + noise_scale*0.1
    current_step = int(1000*noise_scale)-100

    latents = pipeline.vae.model.stream_encode(input_video)  # [B, 4, T, H//16, W//16]
    latents = latents.transpose(2,1)

    if is_first_chunk or getattr(pipeline, "kv_cache1", None) is None:
        pipeline.prepare(latents, text_prompts=prompts)

    # noise = torch.randn_like(latents)
    # noisy_latents = noise*noise_scale + latents*(1-noise_scale)

    # denoised_pred = pipeline.inference(
    #     noise=noisy_latents,
    #     current_start=current_start,
    #     current_end=current_end,
    #     current_step=current_step,
    # )
    denoised_pred = latents

    video = pipeline.vae.stream_decode_to_pixel(denoised_pred)  # [B, T, H, W, 3]
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video.permute(0,2,1,3,4)  # [B, C, T, H, W]
    video_output = video[0].permute(0, 2, 3, 1).contiguous().float().cpu().numpy()  # [T,H,W,3]
    return video_output


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
    parser.add_argument("--max_outstanding", type=int, default=4, help="max number of outstanding sends/recv to keep")
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

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v

    # Rank 0 读取原始视频
    if rank == 0:
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
        if input_video_original.dtype != torch.bfloat16:
            input_video_original = input_video_original.to(dtype=torch.bfloat16)
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

    assert world_size >= 2, "需要至少2个GPU进行编码-解码流水线"

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

    TAG_LATENT_HDR = 11001
    TAG_LATENT_PAY = 11002

    noise_scale = args.noise_scale
    MAX_OUTSTANDING = args.max_outstanding

    # ----- RANK 0: encoder + async send (isend) -----
    if rank == 0:
        prepared = False
        outstanding = []  # list of (work_obj, work_obj2, latents_ref, header_ref)
        torch.cuda.synchronize()
        start_time = time.time()
        for i, (start_idx, end_idx, current_start, current_end) in enumerate(chunk_meta):
            inp = input_video_original[:, :, start_idx:end_idx].to(device)

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

            denoised_pred = pipeline.inference(
                noise=noisy_latents, # [1, 4, 16, 16, 60]
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )

            latents = denoised_pred

            # 控制并发数量，超过则等待最早的完成
            while len(outstanding) >= MAX_OUTSTANDING:
                oldest = outstanding.pop(0)
                # wait for both header & payload to finish
                try:
                    oldest[0].wait()
                    oldest[1].wait()
                except Exception:
                    # fallback: try waiting on payload then header
                    oldest[1].wait()
                    oldest[0].wait()
                # release references by popping

            # build header on GPU (NCCL requires GPU tensors for send/recv)
            bsz, tlen, cch, hh, ww = latents.shape
            header = torch.tensor([i, bsz, tlen, cch, hh, ww], dtype=torch.int64, device=device)

            # non-blocking sends
            work_h = dist.isend(header, dst=1, tag=TAG_LATENT_HDR)
            work_p = dist.isend(latents, dst=1, tag=TAG_LATENT_PAY)

            # keep references until send completes
            outstanding.append((work_h, work_p, latents, header))
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            print(f"[rank0] Encode time: {t:.4f} s, fps: {inp.shape[2]/t:.4f}")
            start_time = end_time

        # 等待所有未完成发送结束
        for (w_h, w_p, _, _) in outstanding:
            w_h.wait()
            w_p.wait()

        # rank0 不再写文件，由 rank1 负责聚合与写出

    # ----- RANK 1: receiver thread(s) + decode loop -----
    elif rank == 1:
        os.makedirs(args.output_folder, exist_ok=True)
        prepared = False
        results = {}

        # queue for handing received latents to decoder
        decode_queue = queue.Queue(maxsize=MAX_OUTSTANDING + 2)

        # buffer pool: map from shape tuple -> list of tensors
        free_buffers = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}

        receiver_stop = threading.Event()

        def receiver_thread_fn():
            # each thread must set device
            torch.cuda.set_device(local_rank)
            device_t = torch.device(f"cuda:{local_rank}")

            for _ in range(num_chuncks):
                # receive header first (on GPU)
                lhdr = torch.empty(6, dtype=torch.int64, device=device_t)
                dist.recv(lhdr, src=0, tag=TAG_LATENT_HDR)
                ci, bsz, tlen, cch, hh, ww = [int(x) for x in lhdr.tolist()]
                shape = (bsz, tlen, cch, hh, ww)

                # get or allocate buffer (reuse if possible)
                buf = None
                if shape in free_buffers and len(free_buffers[shape]) > 0:
                    buf = free_buffers[shape].pop()
                else:
                    # allocate new on GPU with same dtype as sent
                    buf = torch.empty(shape, dtype=torch.bfloat16, device=device_t)

                # blocking recv into buf
                dist.recv(buf, src=0, tag=TAG_LATENT_PAY)

                # put into decode queue (this will block if decode queue is full)
                decode_queue.put((ci, buf))

            # received all expected chunks -> signal done
            receiver_stop.set()

        # start receiver thread
        rcv_thread = threading.Thread(target=receiver_thread_fn, daemon=True)
        rcv_thread.start()

        processed = 0
        torch.cuda.synchronize()
        start_time = time.time()
        while processed < num_chuncks:
            ci, latents = decode_queue.get()  # wait until one chunk available

            shape = tuple(latents.shape)

            if not prepared:
                # pipeline.prepare will likely initialize caches on rank1
                pipeline.prepare(latents, text_prompts=prompts)
                prepared = True

            denoised_pred = latents  # (we keep it simple; in real code do inference)
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
            print(f"[rank1] Decode time: {t:.4f} s, fps: {video.shape[0]/t:.4f}")
            start_time = end_time

        # wait receiver thread to exit cleanly (should have finished num_chuncks)
        rcv_thread.join(timeout=5.0)

        # 拼接与写文件（在 rank1 完成）
        video_list = [results[i] for i in range(num_chuncks)]
        video = np.concatenate(video_list, axis=0)
        export_to_video(
            video, os.path.join(args.output_folder, f"output_{0:03d}.mp4"), fps=args.fps)
        print(f"Video saved to: {os.path.join(args.output_folder, f'output_{0:03d}.mp4')}")

    dist.barrier()



if __name__ == "__main__":
    main()