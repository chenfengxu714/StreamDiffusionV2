from causvid.models.wan.causal_stream_inference_pipe import CausalStreamInferencePipeline

import sys
import os
import numpy as np
import time
import threading
from omegaconf import OmegaConf
from multiprocessing import Queue, Event, Process
import multiprocessing as mp
import random
import queue

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

import torch
import torch.distributed as dist

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from typing import List


default_prompt = "A panda is performing kung fu"

page_content = """<h1 class="text-3xl font-bold">StreamV2V</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://jeff-liangf.github.io/projects/streamv2v/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamV2V
</a>
video-to-video pipeline using
    <a
    href="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">4-step LCM LORA</a
    > with a MJPEG stream server.
</p>
<p class="text-sm">
The base model is <a
href="https://huggingface.co/runwayml/stable-diffusion-v1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD 1.5</a
    >. We also build in <a
    href="https://github.com/Jeff-LiangF/streamv2v/tree/main/demo_w_camera#download-lora-weights-for-better-stylization"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">some LORAs
</a> for better stylization.
</p>
"""

TAG_LATENT_HDR = 11001
TAG_LATENT_PAY = 11002
TAG_START_END_STEP = 11003
TAG_PATCHED_X_SHAPE = 11004
TAG_LATENT_ORIGIN_HDR = 11005
TAG_LATENT_ORIGIN_PAY = 11006


def fold_2x2_spatial(video: torch.Tensor, original_batch: int) -> torch.Tensor:
    B4, C, T, H_half, W_half = video.shape
    assert B4 % 4 == 0 and B4 == original_batch * 4

    video = video.view(original_batch, 2, 2, C, T, H_half, W_half)  # (B, 2, 2, C, T, H//2, W//2)
    video = video.permute(0, 3, 4, 5, 1, 6, 2)  # (B, C, T, H//2, 2, W//2, 2)
    video = video.contiguous().view(original_batch, C, T, H_half * 2, W_half * 2)  # (B, C, T, H, W)

    return video


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        width: int = Field(
            400, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            400, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args):
        torch.set_grad_enabled(False)

        params = self.InputParams()
        config = OmegaConf.load(args.config_path)
        for k, v in args._asdict().items():
            config[k] = v
        config["height"] = params.height
        config["width"] = params.width

        self.width = params.width
        self.height = params.height
        self.prompt = params.prompt

        self.input_queue = Queue()
        self.output_queue = Queue()

        total_blocks = 29
        if args.world_size == 2:
            total_block_num = [[19], [19]]
        else:
            total_block_num.append([0])
            num_middle = args.world_size - 2
            for i in range(num_middle):
                start = int(i * total_blocks // num_middle)
                end = int((i + 1) * total_blocks // num_middle)
                total_block_num.append([start, end])
            total_block_num.append([total_blocks])

        self.prepare_events = [Event() for _ in range(args.world_size)]
        self.p_input = Process(
                target=input_process,
                args=(0, total_block_num[0], config, self.prompt, self.prepare_events[0], self.input_queue),
                daemon=True
            )
        self.p_middles = [
            Process(
                target=middle_process,
                args=(i, total_block_num[i], config, self.prompt, self.prepare_events[i]),
                daemon=True
            )
            for i in range(1, args.world_size - 1)
        ]
        self.p_output = Process(
            target=output_process,
            args=(args.world_size - 1, total_block_num[-1], config, self.prompt, self.prepare_events[-1], self.output_queue),
            daemon=True
        )

        self.p_input.start()
        for p in self.p_middles:
            p.start()
        self.p_output.start()

        for event in self.prepare_events:
            event.wait()

    def accept_new_params(self, params: "Pipeline.InputParams"):
        image_array = self.image_to_array(params.image, self.width, self.height)
        self.input_queue.put(image_array)

        if params.prompt and self.prompt != params.prompt:
            self.update_prompt(params.prompt)

    def update_prompt(self, prompt: str):
        self.prompt = prompt

    def predict(self) -> List[Image.Image]:
        qsize = self.output_queue.qsize()
        results = []
        for _ in range(qsize):
            results.append(self.array_to_image(self.output_queue.get()))
        return results

    def image_to_array(
        self, image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> np.ndarray:
        image = image.convert("RGB").resize((width, height))
        image_array = np.array(image)
        if normalize:
            image_array = image_array / 127.5 - 1.0
        return image_array

    def array_to_image(self, image_array: np.ndarray, normalize: bool = True) -> Image.Image:
        if normalize:
            image_array = image_array * 255.0
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        return image


def select_images(images, num_images: int):
    if len(images) < num_images:
        return []
    step = len(images) / (num_images - 1)
    indices = [int(i * step) for i in range(num_images - 1)] + [-1]
    selected_images = np.stack([images[i] for i in indices], axis=0)
    return selected_images


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

def receiver_thread_fn(rank, device, free_buffers, free_buffers_origin, decode_queue):
    while True:
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


def input_process(rank, block_num, args, prompt, prepare_event, input_queue):
    init_dist_tcp(rank, args.world_size)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    pipeline = prepare_pipeline(device, args, prompt)
    current_start = 0
    current_end = pipeline.frame_seq_length * 2
    outstanding = []
    last_image = None
    num_frames_needed = 5
    num_batch = 0
    chunk_size = 4
    prepare_event.set()

    while True:
        qsize = input_queue.qsize()
        if qsize < num_frames_needed:
            time.sleep(0.01)
            continue

        images = []
        for _ in range(qsize):
            images.append(input_queue.get())

        images = select_images(images, num_frames_needed)
        images = torch.from_numpy(images).unsqueeze(0)
        images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=device)

        if last_image is None:
            l2_dist=(images[:,:,-chunk_size:] - images[:,:,-chunk_size-1:-1])**2
        else:
            l2_dist=(images[:,:,-chunk_size:] - 
                torch.cat([last_image, images[:,:,-chunk_size:-1]], dim=2)
            )**2
        l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
        noise_scale = (0.8-0.1*l2_dist.item())*0.9+args.noise_scale*0.1
        current_step = int(1000*noise_scale)-100
        latents = pipeline.vae.model.stream_encode(images)  # [B, 4, T, H//16, W//16] or so
        latents = latents.transpose(2,1).contiguous().to(dtype=torch.bfloat16)

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

        while len(outstanding) >= args.max_outstanding:
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
                raise Exception(f"Error waiting for outstanding chunks {num_batch}")

        send_latents_fn(num_batch, denoised_pred, noisy_latents, patched_x_shape, current_start, current_end, current_step, device, rank, outstanding)

        current_start = current_end
        current_end += (chunk_size // 4) * pipeline.frame_seq_length
        num_frames_needed = chunk_size
        last_image = images[:,:,[-1]]
        num_batch += 1

def output_process(rank, block_num, args, prompt, prepare_event, output_queue):
    init_dist_tcp(rank, args.world_size)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    decode_queue = queue.Queue(maxsize=args.max_outstanding)
    free_buffers = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}
    free_buffers_origin = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}

    # start receiver thread
    rcv_thread = threading.Thread(target=receiver_thread_fn, args=(rank, device, free_buffers, free_buffers_origin, decode_queue), daemon=True)
    rcv_thread.start()

    torch.cuda.synchronize()

    results = {}
    next_batch = 0
    pipeline = prepare_pipeline(device, args, prompt)
    prepare_event.set()

    while True:
        ci, latents, latents_origin, current_start, current_end, current_step, patched_x_shape = decode_queue.get()  # wait until one chunk available

        shape = tuple(latents.shape)

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

        # return buffer to free pool for reuse
        if shape not in free_buffers:
            free_buffers[shape] = []
        free_buffers[shape].append(latents)

        torch.cuda.synchronize()

        if next_batch in results:
            for video in results[next_batch]:
                output_queue.put(video)
            del results[next_batch]
            next_batch += 1


def middle_process(rank, block_num, args, prompt, prepare_event):
    init_dist_tcp(rank, args.world_size)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # queue for handing received latents to decoder
    decode_queue = queue.Queue(maxsize=args.max_outstanding)
    
    # buffer pool: map from shape tuple -> list of tensors
    free_buffers = {}  # {(bsz,tlen,cch,hh,ww): [tensor1, ...]}
    free_buffers_origin = {}

    # start receiver thread
    rcv_thread = threading.Thread(target=receiver_thread_fn, args=(rank, device, free_buffers, free_buffers_origin, decode_queue), daemon=True)
    rcv_thread.start()

    outstanding = []
    pipeline = prepare_pipeline(device, args, prompt)
    torch.cuda.synchronize()

    prepare_event.set()

    while True:
        i, latents, latents_origin, current_start, current_end, current_step, patched_x_shape = decode_queue.get()

        denoised_pred, _ = pipeline.inference(
            noise=latents, # [1, 4, 16, 16, 60]
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            block_mode='middle',
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        )

        while len(outstanding) >= args.max_outstanding:
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


def prepare_pipeline(device, args, prompt):
    pipeline = CausalStreamInferencePipeline(args, device=str(device))
    pipeline.to(device=str(device), dtype=torch.bfloat16)
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
    pipeline.generator.load_state_dict(state_dict, strict=True)

    noise = torch.randn(1, 1).to(device=device, dtype=torch.bfloat16)
    pipeline.prepare(noise=noise, text_prompts=[prompt])
    return pipeline

def init_dist_tcp(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: int = 29500):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

