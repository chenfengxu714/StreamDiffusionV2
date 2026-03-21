import sys
import os
import time
from multiprocessing import Queue, Event, Process, Manager

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamv2v.inference_pipe import InferencePipelineManager, compute_default_block_distribution
from util import clear_queue, get_num_transformer_blocks, read_images_from_queue, resolve_worker_device

import torch
import torch.distributed as dist

from vid2vid import Pipeline, report_worker_error, wait_for_processes_ready


class MultiGPUPipeline(Pipeline):
    def prepare(self):
        self.total_blocks = get_num_transformer_blocks(self.args)
        self.total_block_num = compute_default_block_distribution(
            total_blocks=self.total_blocks,
            world_size=self.args.num_gpus,
        )

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_events = [Event() for _ in range(self.args.num_gpus)]
        self.stop_event = Event()
        self.restart_event = Event()
        self.error_queue = Queue()
        self.prompt_dict = Manager().dict()
        self.prompt_dict["prompt"] = self.prompt
        self.p_input = Process(
            target=input_process,
            args=(0, self.total_block_num, self.total_blocks, self.args, self.prompt_dict, self.prepare_events[0], self.restart_event, self.stop_event, self.input_queue, self.error_queue),
            daemon=True,
        )
        self.p_middles = [
            Process(
                target=middle_process,
                args=(i, self.total_block_num, self.total_blocks, self.args, self.prompt_dict, self.prepare_events[i], self.stop_event, self.error_queue),
                daemon=True,
            )
            for i in range(1, self.args.num_gpus - 1)
        ]
        self.p_output = Process(
            target=output_process,
            args=(self.args.num_gpus - 1, self.total_block_num, self.total_blocks, self.args, self.prompt_dict, self.prepare_events[-1], self.stop_event, self.output_queue, self.error_queue),
            daemon=True,
        )
        self.processes = [self.p_input] + self.p_middles + [self.p_output]

        for process in self.processes:
            process.start()

        wait_for_processes_ready(
            processes=self.processes,
            ready_events=self.prepare_events,
            error_queue=self.error_queue,
        )


def input_process(rank, block_num, total_blocks, args, prompt_dict, prepare_event, restart_event, stop_event, input_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        chunk_size = pipeline_manager.get_demo_chunk_size()
        first_batch_num_frames = pipeline_manager.get_demo_first_batch_num_frames()
        is_running = False
        prompt = prompt_dict["prompt"]
        schedule_block = args.schedule_block

        torch.cuda.memory._record_memory_history(max_entries=100000)

        prepare_event.set()

        while not stop_event.is_set():
            if is_running and (prompt_dict["prompt"] != prompt or restart_event.is_set()):
                if restart_event.is_set():
                    clear_queue(input_queue)
                    restart_event.clear()
                prompt = prompt_dict["prompt"]
                pipeline_manager.send_demo_input_prompt_update(
                    prompt=prompt,
                    device=device,
                    num_steps=num_steps,
                    chunk_idx=session.chunk_idx,
                    denoised_pred=denoised_pred,
                    patched_x_shape=patched_x_shape,
                    current_step=session.current_step,
                )
                is_running = False
                outstanding = []

            if not is_running:
                images = read_images_from_queue(input_queue, first_batch_num_frames, device, stop_event)
                if images is None:
                    return
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                session = pipeline_manager.start_demo_input_stream_session(
                    prompt=prompt,
                    images=images,
                    block_num=block_num[rank],
                    noise_scale=args.noise_scale,
                )
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            pipeline_manager.maybe_refresh_demo_input_window(session)

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_vae = time.time()

            if session.input_batch == 0:
                images = read_images_from_queue(input_queue, chunk_size, device, stop_event)
                if images is None:
                    break
                pipeline_manager.prepare_demo_input_batch(session, images)

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()
                t_vae = start_dit - start_vae

            denoised_pred, patched_x_shape = pipeline_manager.run_demo_input_step(
                session=session,
                block_num=block_num[rank],
                previous_latent_data=latent_data if "latent_data" in locals() else None,
            )

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1

            with torch.cuda.stream(pipeline_manager.com_stream):
                if pipeline_manager.processed >= pipeline_manager.world_size:
                    if "latent_data" in locals():
                        pipeline_manager.data_transfer.release_latent_data(latent_data)
                    latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)

            torch.cuda.current_stream().wait_stream(pipeline_manager.com_stream)
            pipeline_manager._wait_for_outstanding(outstanding)

            with torch.cuda.stream(pipeline_manager.com_stream):
                work_objects = pipeline_manager.data_transfer.send_latent_data_async(
                    chunk_idx=session.chunk_idx,
                    latents=denoised_pred,
                    original_latents=pipeline_manager.pipeline.hidden_states,
                    patched_x_shape=patched_x_shape,
                    current_start=pipeline_manager.pipeline.kv_cache_starts,
                    current_end=pipeline_manager.pipeline.kv_cache_ends,
                    current_step=session.current_step,
                )
                outstanding.append(work_objects)
                if schedule_block and pipeline_manager.processed >= pipeline_manager.schedule_step:
                    pipeline_manager._handle_block_scheduling(block_num, total_blocks=total_blocks)
                    schedule_block = False

            if schedule_block:
                t_total = pipeline_manager.t_dit + t_vae
                if t_total < pipeline_manager.t_total:
                    pipeline_manager.t_total = t_total

            pipeline_manager.advance_demo_input_stream_session(session, images)
            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_input_rank_{rank}")
        raise


def output_process(rank, block_num, total_blocks, args, prompt_dict, prepare_event, stop_event, output_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        prompt = prompt_dict["prompt"]
        is_running = False
        need_update_prompt = False
        schedule_block = args.schedule_block
        prepare_event.set()

        while not stop_event.is_set():
            if need_update_prompt:
                prompt = pipeline_manager.data_transfer.recv_prompt_async()
                is_running = False
                need_update_prompt = False
                outstanding = []

            if not is_running:
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                images = pipeline_manager.prepare_demo_worker_session(
                    prompt=prompt,
                    block_mode="output",
                    block_num=block_num[rank],
                    decode_initial=True,
                )
                for image in images:
                    output_queue.put(image)
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            latent_data = pipeline_manager._receive_latent_data(latent_data if "latent_data" in locals() else None, num_steps)
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
            schedule_block = pipeline_manager._maybe_schedule_blocks(
                schedule_block,
                pipeline_manager.schedule_step - rank,
                block_num,
                total_blocks=total_blocks,
            )

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()

            denoised_pred, _ = pipeline_manager._run_worker_stage("output", latent_data, block_num[rank])

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1
            pipeline_manager._wait_for_outstanding(outstanding)
            pipeline_manager._send_worker_result("output", outstanding, latent_data, denoised_pred)

            if pipeline_manager.processed >= num_steps * pipeline_manager.world_size - 1:
                if schedule_block:
                    pipeline_manager._sync_for_timing(schedule_block)
                    start_vae = time.time()

                for image in pipeline_manager._decode_prediction(denoised_pred):
                    output_queue.put(image)

                torch.cuda.synchronize()

                if schedule_block:
                    t_vae = time.time() - start_vae
                    t_total = t_vae + pipeline_manager.t_dit
                    if t_total < pipeline_manager.t_total:
                        pipeline_manager.t_total = t_total

            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_output_rank_{rank}")
        raise


def middle_process(rank, block_num, total_blocks, args, prompt_dict, prepare_event, stop_event, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        prompt = prompt_dict["prompt"]
        is_running = False
        need_update_prompt = False
        schedule_block = args.schedule_block

        prepare_event.set()

        while not stop_event.is_set():
            if need_update_prompt:
                prompt = pipeline_manager.data_transfer.recv_prompt_async()
                pipeline_manager.logger.info(f"Rank {rank} sending dummy data")
                pipeline_manager.send_demo_middle_prompt_update(
                    prompt=prompt,
                    device=device,
                    denoised_pred=denoised_pred,
                    latent_data=latent_data,
                )
                is_running = False
                need_update_prompt = False
                outstanding = []

            if not is_running:
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                pipeline_manager.prepare_demo_worker_session(
                    prompt=prompt,
                    block_mode="middle",
                    block_num=block_num[rank],
                )
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            latent_data = pipeline_manager._receive_latent_data(latent_data if "latent_data" in locals() else None, num_steps)
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
            schedule_block = pipeline_manager._maybe_schedule_blocks(
                schedule_block,
                pipeline_manager.schedule_step - rank,
                block_num,
                total_blocks=total_blocks,
            )

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()

            denoised_pred, _ = pipeline_manager._run_worker_stage("middle", latent_data, block_num[rank])

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1
            pipeline_manager._wait_for_outstanding(outstanding)
            pipeline_manager._send_worker_result("middle", outstanding, latent_data, denoised_pred)

            torch.cuda.synchronize()

            if schedule_block:
                t_total = pipeline_manager.t_dit
                if t_total < pipeline_manager.t_total:
                    pipeline_manager.t_total = t_total

            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_middle_rank_{rank}")
        raise


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
