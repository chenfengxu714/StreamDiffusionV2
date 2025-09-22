import sys
import os
import numpy as np
import time
from omegaconf import OmegaConf
from multiprocessing import Queue, Event, Process, Manager
from streamv2v.inference_pipe import InferencePipelineManager


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
        self.args = config

        total_blocks = 30
        if args.world_size == 2:
            self.total_block_num = [[0, 15], [15, total_blocks]]
        else:
            base = total_blocks // args.world_size
            rem = total_blocks % args.world_size
            start = 0
            self.total_block_num = []
            for r in range(args.world_size):
                size = base + (1 if r < rem else 0)
                end = start + size if r < args.world_size - 1 else total_blocks
                self.total_block_num.append([start, end])
                start = end

        self.prepare_processes()

    def prepare_processes(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_events = [Event() for _ in range(self.args.world_size)]
        self.stop_event = Event()
        self.prompt_dict = Manager().dict()
        self.prompt_dict["prompt"] = self.prompt
        self.p_input = Process(
                target=input_process,
                args=(0, self.total_block_num[0], self.args, self.prompt_dict, self.prepare_events[0], self.stop_event, self.input_queue),
                daemon=True
            )
        self.p_middles = [
            Process(
                target=middle_process,
                args=(i, self.total_block_num[i], self.args, self.prompt_dict, self.prepare_events[i], self.stop_event),
                daemon=True
            )
            for i in range(1, self.args.world_size - 1)
        ]
        self.p_output = Process(
            target=output_process,
            args=(self.args.world_size - 1, self.total_block_num[-1], self.args, self.prompt_dict, self.prepare_events[-1], self.stop_event, self.output_queue),
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
            self.prompt = params.prompt
            self.prompt_dict["prompt"] = self.prompt

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

    def close(self):
        print("Setting stop event...")
        self.stop_event.set()

        # Give processes a short time to exit gracefully
        timeout = 1.0
        
        print("Waiting for input process to terminate...")
        self.p_input.join(timeout=timeout)
        if self.p_input.is_alive():
            print("Input process didn't terminate gracefully, forcing termination")
            self.p_input.terminate()
            self.p_input.join(timeout=0.5)
            if self.p_input.is_alive():
                print("Force killing input process")
                self.p_input.kill()
        
        print("Waiting for middle processes to terminate...")
        for i, p in enumerate(self.p_middles):
            p.join(timeout=timeout)
            if p.is_alive():
                print(f"Middle process {i} didn't terminate gracefully, forcing termination")
                p.terminate()
                p.join(timeout=0.5)
                if p.is_alive():
                    print(f"Force killing middle process {i}")
                    p.kill()
        
        print("Waiting for output process to terminate...")
        self.p_output.join(timeout=timeout)
        if self.p_output.is_alive():
            print("Output process didn't terminate gracefully, forcing termination")
            self.p_output.terminate()
            self.p_output.join(timeout=0.5)
            if self.p_output.is_alive():
                print("Force killing output process")
                self.p_output.kill()
        
        print("Destroying process group...")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Error destroying process group: {e}")
        
        print("Pipeline closed successfully")


def input_process(rank, block_num, args, prompt_dict, prepare_event, stop_event, input_queue):
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_dist_tcp(rank, args.world_size, device=device)
    block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

    pipeline_manager = prepare_pipeline(args, device, rank, args.world_size)
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    first_batch_num_frames = 5
    chunk_size = 4
    is_running = False
    prompt = prompt_dict["prompt"]

    prepare_event.set()

    while not stop_event.is_set():
        # Check if prompt has changed
        if is_running and prompt_dict["prompt"] != prompt:
            prompt = prompt_dict["prompt"]
            # Send stop signal by chunk_idx=-1 to the other ranks
            with torch.cuda.stream(pipeline_manager.com_stream):
                pipeline_manager.data_transfer.send_latent_data_async(
                    chunk_idx=-1,
                    latents=denoised_pred.new_zeros([1] * denoised_pred.ndim),
                    original_latents=pipeline_manager.pipeline.hidden_states.new_zeros([1] * pipeline_manager.pipeline.hidden_states.ndim),
                    patched_x_shape=patched_x_shape,
                    current_start=pipeline_manager.pipeline.kv_cache_starts,
                    current_end=pipeline_manager.pipeline.kv_cache_ends,
                    current_step=int(current_step),
                )
                pipeline_manager.data_transfer.send_prompt_async(prompt, device)
                # Receive all the pending data from previous batches to ensure all send/recv are completed
                for _ in range(min(chunk_idx, args.world_size - 1)):
                    pipeline_manager.data_transfer.receive_latent_data_async(num_steps)
            is_running = False
            outstanding = []

        if not is_running:
            images = read_images_from_queue(input_queue, first_batch_num_frames, device, stop_event, prefer_latest=True)
            if images is None:
                return
            pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
            init_first_batch_for_input_process(args, device, pipeline_manager, images, prompt, block_num)    

            chunk_idx = 0
            current_start = pipeline_manager.pipeline.frame_seq_length * 2
            current_end = current_start + (chunk_size // 4) * pipeline_manager.pipeline.frame_seq_length
            last_image = images[:,:,[-1]]
            outstanding = []
            pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

        images = read_images_from_queue(input_queue, chunk_size, device, stop_event)
        if images is None:
            break

        if args.schedule_block:
            torch.cuda.synchronize()
            start_vae = time.time()

        l2_dist=(
            images - torch.cat([last_image, images[:,:,-chunk_size:-1]], dim=2)
        ) ** 2
        l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
        noise_scale = (0.8-0.1*l2_dist.item())*0.9+args.noise_scale*0.1
        current_step = int(1000*noise_scale)-100

        latents = pipeline_manager.pipeline.vae.model.stream_encode(images)  # [B, 4, T, H//16, W//16] or so
        latents = latents.transpose(2,1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        noisy_latents = noise * noise_scale + latents * (1-noise_scale)

        # Measure DiT time if scheduling is enabled
        if args.schedule_block:
            torch.cuda.synchronize()
            start_dit = time.time()
            t_vae = start_dit - start_vae

        denoised_pred, patched_x_shape = pipeline_manager.pipeline.inference(
            noise=noisy_latents, # [1, 4, 16, 16, 60]
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            block_mode='input',
            block_num=block_num,
        )
        # pipeline_manager.logger.info(f"[Rank {rank}] Inference done for chunk {chunk_idx}")

        # Update DiT timing
        if args.schedule_block:
            torch.cuda.synchronize()
            temp = time.time() - start_dit
            if temp < pipeline_manager.t_dit:
                pipeline_manager.t_dit = temp

        pipeline_manager.processed += 1

        # Handle communication
        with torch.cuda.stream(pipeline_manager.com_stream):
            if pipeline_manager.processed >= pipeline_manager.world_size:
                # Receive data from previous rank
                # pipeline_manager.logger.info(f"Rank {rank} receiving data")
                latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)
                # pipeline_manager.logger.info(f"Rank {rank} received chunk {latent_data.chunk_idx}")

        torch.cuda.current_stream().wait_stream(pipeline_manager.com_stream)

        # Wait for outstanding operations
        while len(outstanding) >= args.max_outstanding:
            oldest = outstanding.pop(0)
            for work in oldest:
                work.wait()

        # Send data to next rank
        with torch.cuda.stream(pipeline_manager.com_stream):
            work_objects = pipeline_manager.data_transfer.send_latent_data_async(
                chunk_idx=chunk_idx,
                latents=denoised_pred,
                original_latents=pipeline_manager.pipeline.hidden_states,
                patched_x_shape=patched_x_shape,
                current_start=pipeline_manager.pipeline.kv_cache_starts,
                current_end=pipeline_manager.pipeline.kv_cache_ends,
                current_step=current_step
            )
            outstanding.append(work_objects)
            # pipeline_manager.logger.info(f"[Rank {rank}] Scheduled send chunk {chunk_idx} to next rank")

        if args.schedule_block:
            t_total = pipeline_manager.t_dit + t_vae
            if t_total < pipeline_manager.t_total:
                pipeline_manager.t_total = t_total

        # Handle block scheduling
        if args.schedule_block and pipeline_manager.processed >= pipeline_manager.schedule_step:
            with torch.cuda.stream(pipeline_manager.control_stream):
                schedule_check = torch.tensor(1, dtype=torch.int64, device=pipeline_manager.device)
                reply = dist.broadcast(schedule_check, src=pipeline_manager.world_size - 1, async_op=True)
            reply.wait()
            pipeline_manager._handle_block_scheduling(block_num, total_blocks=30)
            args.schedule_block = False

        if pipeline_manager.processed >= pipeline_manager.world_size:
            pipeline_manager.pipeline.hidden_states = latent_data.original_latents
            pipeline_manager.pipeline.kv_cache_starts.copy_(latent_data.current_start)
            pipeline_manager.pipeline.kv_cache_ends.copy_(latent_data.current_end)

        last_image = images[:,:,[-1]]
        chunk_idx += 1
        current_start = current_end
        current_end += (chunk_size // 4) * pipeline_manager.pipeline.frame_seq_length
        is_running = True

def output_process(rank, block_num, args, prompt_dict, prepare_event, stop_event, output_queue):
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_dist_tcp(rank, args.world_size, device=device)
    block_num = torch.tensor(block_num, dtype=torch.int64, device=device)
    
    pipeline_manager = prepare_pipeline(args, device, rank, args.world_size)
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    prompt = prompt_dict["prompt"]
    is_running = False
    need_update_prompt = False
    prepare_event.set()

    while not stop_event.is_set():
        # Check if prompt has changed
        if need_update_prompt:
            prompt = pipeline_manager.data_transfer.recv_prompt_async()
            is_running = False
            need_update_prompt = False
            outstanding = []

        if not is_running:
            pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
            images = init_first_batch_for_output_process(args, device, pipeline_manager, prompt, block_num)
            for image in images:
                output_queue.put(image)
            outstanding = []
            pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

        # Receive data from previous rank
        with torch.cuda.stream(pipeline_manager.com_stream):
            # pipeline_manager.logger.info(f"Rank {rank} receiving data")
            latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)
            # pipeline_manager.logger.info(f"Rank {rank} received chunk {latent_data.chunk_idx}")
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
        torch.cuda.current_stream().wait_stream(pipeline_manager.com_stream)
        # pipeline_manager.logger.info(f"[Rank {rank}] Received chunk {latent_data.chunk_idx} from previous rank")

        # Measure DiT time if scheduling is enabled
        if args.schedule_block:
            torch.cuda.synchronize()
            start_dit = time.time()
        
        # Run inference
        denoised_pred, _ = pipeline_manager.pipeline.inference(
            noise=latent_data.original_latents,
            current_start=latent_data.current_start,
            current_end=latent_data.current_end,
            current_step=latent_data.current_step,
            block_mode='output',
            block_num=block_num,
            patched_x_shape=latent_data.patched_x_shape,
            block_x=latent_data.latents,
        )
        # pipeline_manager.logger.info(f"[Rank {rank}] Inference done for chunk {latent_data.chunk_idx}")
        
        # Update DiT timing
        if args.schedule_block:
            torch.cuda.synchronize()
            temp = time.time() - start_dit
            if temp < pipeline_manager.t_dit:
                pipeline_manager.t_dit = temp
        
        pipeline_manager.processed += 1
        
        # Wait for outstanding operations
        while len(outstanding) >= args.max_outstanding:
            oldest = outstanding.pop(0)
            for work in oldest:
                work.wait()

        with torch.cuda.stream(pipeline_manager.com_stream):
            work_objects = pipeline_manager.data_transfer.send_latent_data_async(
                chunk_idx=latent_data.chunk_idx,
                latents=latent_data.latents,
                original_latents=denoised_pred,
                patched_x_shape=latent_data.patched_x_shape,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step
            )
            outstanding.append(work_objects)
            # pipeline_manager.logger.info(f"[Rank {rank}] Scheduled send chunk {latent_data.chunk_idx} to next rank")
        
        # Decode and save video
        if pipeline_manager.processed >= num_steps * pipeline_manager.world_size - 1:
            if args.schedule_block:
                torch.cuda.synchronize()
                start_vae = time.time()

            video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()

            for image in video.cpu().float().numpy():
                output_queue.put(image)
            # pipeline_manager.logger.info(f"[Rank {rank}] Completed chunk {latent_data.chunk_idx}")
            
            torch.cuda.synchronize()

            if args.schedule_block:
                t_vae = time.time() - start_vae
                t_total = t_vae + pipeline_manager.t_dit
                if t_total < pipeline_manager.t_total:
                    pipeline_manager.t_total = t_total

            # Handle block scheduling
            if args.schedule_block and pipeline_manager.processed >= pipeline_manager.schedule_step - rank:
                with torch.cuda.stream(pipeline_manager.control_stream):
                    schedule_check = torch.tensor(1, dtype=torch.int64, device=pipeline_manager.device)
                    reply = dist.broadcast(schedule_check, src=pipeline_manager.world_size - 1, async_op=True)
                reply.wait()
                pipeline_manager._handle_block_scheduling(block_num, total_blocks=30)
                args.schedule_block = False

        is_running = True

def middle_process(rank, block_num, args, prompt_dict, prepare_event, stop_event):
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_dist_tcp(rank, args.world_size, device=device)
    block_num = torch.tensor(block_num, dtype=torch.int64, device=device)
    
    pipeline_manager = prepare_pipeline(args, device, rank, args.world_size)
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    prompt = prompt_dict["prompt"]
    is_running = False
    need_update_prompt = False

    prepare_event.set()

    while not stop_event.is_set():
        if need_update_prompt:
            prompt = pipeline_manager.data_transfer.recv_prompt_async()
            pipeline_manager.logger.info(f"Rank {rank} sending dummy data")
            with torch.cuda.stream(pipeline_manager.com_stream):
                outstanding.append(pipeline_manager.data_transfer.send_latent_data_async(
                    chunk_idx=-1,
                    latents=denoised_pred.new_zeros([1] * denoised_pred.ndim),
                    original_latents=latent_data.original_latents,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=int(latent_data.current_step),
                ))
                outstanding.append(pipeline_manager.data_transfer.send_prompt_async(prompt, device))
            is_running = False
            need_update_prompt = False
            outstanding = []

        if not is_running:
            pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
            init_first_batch_for_middle_process(args, device, pipeline_manager, prompt, block_num)
            outstanding = []
            pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

        # Receive data from previous rank
        with torch.cuda.stream(pipeline_manager.com_stream):
            latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
        torch.cuda.current_stream().wait_stream(pipeline_manager.com_stream)
        # pipeline_manager.logger.info(f"[Rank {rank}] Received chunk {latent_data.chunk_idx} from previous rank")

        if args.schedule_block:
            torch.cuda.synchronize()
            start_dit = time.time()
        
        # Run inference
        denoised_pred, _ = pipeline_manager.pipeline.inference(
            noise=latent_data.original_latents,
            current_start=latent_data.current_start,
            current_end=latent_data.current_end,
            current_step=latent_data.current_step,
            block_mode='middle',
            block_num=block_num,
            patched_x_shape=latent_data.patched_x_shape,
            block_x=latent_data.latents,
        )
        # pipeline_manager.logger.info(f"[Rank {rank}] Inference done for chunk {latent_data.chunk_idx}")
        
        if args.schedule_block:
            torch.cuda.synchronize()
            temp = time.time() - start_dit
            if temp < pipeline_manager.t_dit:
                pipeline_manager.t_dit = temp

        pipeline_manager.processed += 1

        # Wait for outstanding operations
        while len(outstanding) >= args.max_outstanding:
            oldest = outstanding.pop(0)
            for work in oldest:
                work.wait()
        
        # Send data to next rank
        with torch.cuda.stream(pipeline_manager.com_stream):
            work_objects = pipeline_manager.data_transfer.send_latent_data_async(
                chunk_idx=latent_data.chunk_idx,
                latents=denoised_pred,
                original_latents=latent_data.original_latents,
                patched_x_shape=latent_data.patched_x_shape,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step
            )
            outstanding.append(work_objects)
            # pipeline_manager.logger.info(f"[Rank {rank}] Scheduled send chunk {latent_data.chunk_idx} to next rank")

        torch.cuda.synchronize()

        if args.schedule_block:
            t_total = pipeline_manager.t_dit
            if t_total < pipeline_manager.t_total:
                pipeline_manager.t_total = t_total

        # Handle block scheduling
        if args.schedule_block and pipeline_manager.processed >= pipeline_manager.schedule_step - rank:
            with torch.cuda.stream(pipeline_manager.control_stream):
                schedule_check = torch.tensor(1, dtype=torch.int64, device=pipeline_manager.device)
                reply = dist.broadcast(schedule_check, src=pipeline_manager.world_size - 1, async_op=True)
            reply.wait()
            pipeline_manager._handle_block_scheduling(block_num, total_blocks=30)
            args.schedule_block = False

        is_running = True


def init_dist_tcp(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: int = 29500, device: torch.device = None):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )


def prepare_pipeline(args, device, rank, world_size):
    pipeline_manager = InferencePipelineManager(args, device, rank, world_size)
    pipeline_manager.load_model(args.checkpoint_folder)
    return pipeline_manager


def read_images_from_queue(queue, num_frames_needed, device, stop_event=None, prefer_latest=False):
    print(f"Queue size: {queue.qsize()}")
    while queue.qsize() < num_frames_needed:
        if stop_event and stop_event.is_set():
            return None
        time.sleep(0.1)

    if prefer_latest:
        read_size = queue.qsize()
    else:
        read_size = min(queue.qsize(), num_frames_needed * 2)
    images = []
    for _ in range(read_size):
        images.append(queue.get())

    if prefer_latest:
        images = np.stack(images[-num_frames_needed:], axis=0)
    else:
        images = select_images(images, num_frames_needed)
    images = torch.from_numpy(images).unsqueeze(0)
    images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=device)
    return images


def select_images(images, num_images: int):
    if len(images) < num_images:
        return []
    step = len(images) / (num_images - 1)
    indices = [int(i * step) for i in range(num_images - 1)] + [-1]
    selected_images = np.stack([images[i] for i in indices], axis=0)
    return selected_images


def init_first_batch_for_input_process(args, device, pipeline_manager, images, prompt, block_num):
    pipeline_manager.pipeline.vae.model.first_encode = True
    pipeline_manager.pipeline._init_denoising_step_list(args, device)
    pipeline_manager.pipeline.kv_cache1 = None
    pipeline_manager.processed = 0
    latents = pipeline_manager.pipeline.vae.model.stream_encode(images)
    latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
    noise = torch.randn_like(latents)
    noisy_latents = noise * args.noise_scale + latents * (1 - args.noise_scale)

    # First broadcast the shape information
    latents_shape = torch.tensor(latents.shape, dtype=torch.int64, device=device)
    pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
    # Then broadcast noisy_latents
    pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)

    pipeline_manager.prepare_pipeline(
        text_prompts=[prompt],
        noise=noisy_latents,
        block_mode='input',
        current_start=0,
        current_end=pipeline_manager.pipeline.frame_seq_length * 2,
        block_num=block_num,
    )
    
    torch.cuda.empty_cache()
    dist.barrier()


def init_first_batch_for_output_process(args, device, pipeline_manager, prompt, block_num):
    pipeline_manager.pipeline.vae.model.first_decode = True
    pipeline_manager.pipeline._init_denoising_step_list(args, device)
    pipeline_manager.pipeline.kv_cache1 = None
    pipeline_manager.processed = 0
    # Other ranks receive shape info first
    latents_shape = torch.zeros(5, dtype=torch.int64, device=device)
    pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
    # Create tensor with same shape for receiving broadcast data
    noisy_latents = torch.zeros(tuple(latents_shape.tolist()), dtype=torch.bfloat16, device=device)

    # Receive the broadcasted noisy_latents
    pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)

    denoised_pred = pipeline_manager.prepare_pipeline(
        text_prompts=[prompt],
        noise=noisy_latents,
        block_mode='output',
        current_start=0,
        current_end=pipeline_manager.pipeline.frame_seq_length * 2,
        block_num=block_num,
    )

    # Clear unused GPU memory
    torch.cuda.empty_cache()
    dist.barrier()

    video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video[0].permute(0, 2, 3, 1).contiguous()

    return video.cpu().float().numpy()


def init_first_batch_for_middle_process(args, device, pipeline_manager, prompt, block_num):
    pipeline_manager.pipeline._init_denoising_step_list(args, device)
    pipeline_manager.pipeline.kv_cache1 = None
    pipeline_manager.processed = 0
    # Other ranks receive shape info first
    latents_shape = torch.zeros(5, dtype=torch.int64, device=device)
    pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
    # Create tensor with same shape for receiving broadcast data
    noisy_latents = torch.zeros(tuple(latents_shape.tolist()), dtype=torch.bfloat16, device=device)
    # Receive the broadcasted noisy_latents
    pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)

    pipeline_manager.prepare_pipeline(
        text_prompts=[prompt],
        noise=noisy_latents,
        block_mode='output',
        current_start=0,
        current_end=pipeline_manager.pipeline.frame_seq_length * 2,
        block_num=block_num,
    )

    # Clear unused GPU memory
    torch.cuda.empty_cache()
    dist.barrier()
