import sys
import os
import time
from multiprocessing import Queue, Event, Process, Manager
from streamv2v.inference_pipe import InferencePipelineManager
from util import read_images_from_queue

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

import torch
import torch.distributed as dist

from vid2vid import Pipeline


class MultiGPUPipeline(Pipeline):
    def prepare(self):
        total_blocks = 30
        if self.args.world_size == 2:
            self.total_block_num = [[0, 15], [15, total_blocks]]
        else:
            base = total_blocks // self.args.world_size
            rem = total_blocks % self.args.world_size
            start = 0
            self.total_block_num = []
            for r in range(self.args.world_size):
                size = base + (1 if r < rem else 0)
                end = start + size if r < self.args.world_size - 1 else total_blocks
                self.total_block_num.append([start, end])
                start = end

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
        self.processes = [self.p_input] + self.p_middles + [self.p_output]

        for p in self.processes:
            p.start()

        for event in self.prepare_events:
            event.wait()


def input_process(rank, block_num, args, prompt_dict, prepare_event, stop_event, input_queue):
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.gpu_ids[rank])
    device = torch.device(f"cuda:{args.gpu_ids[rank]}")
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
            noise_scale = args.noise_scale
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
        noise_scale = (0.8-0.1*l2_dist.item())*0.9+noise_scale*0.1
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
    torch.cuda.set_device(args.gpu_ids[rank])
    device = torch.device(f"cuda:{args.gpu_ids[rank]}")
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
    torch.cuda.set_device(args.gpu_ids[rank])
    device = torch.device(f"cuda:{args.gpu_ids[rank]}")
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
