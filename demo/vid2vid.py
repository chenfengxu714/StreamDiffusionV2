import sys
import os
from omegaconf import OmegaConf
from multiprocessing import Queue, Manager, Event, Process
from util import read_images_from_queue, image_to_array, array_to_image, clear_queue

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

import torch

from pydantic import BaseModel, Field
from PIL import Image
from typing import List
from streamv2v.inference import SingleGPUInferencePipeline
from streamv2v.inference import compute_noise_scale_and_step


default_prompt = "Cyberpunk-inspired figure, neon-lit hair highlights, augmented cybernetic facial features, glowing interface holograms floating around, futuristic cityscape reflected in eyes, vibrant neon color palette, cinematic sci-fi style"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusionV2</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://streamdiffusionv2.github.io/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusionV2
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""

class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        prompt: str = Field(
            default_prompt,
            title="Update your prompt here",
            field="textarea",
            id="prompt",
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        restart: bool = Field(
            default=False,
            title="Restart",
            description="Restart the streaming",
        )

    def __init__(self, args):
        torch.set_grad_enabled(False)

        params = self.InputParams()
        config = OmegaConf.load(args.config_path)
        for k, v in args._asdict().items():
            config[k] = v
        config["height"] = params.height
        config["width"] = params.width

        full_denoising_list = [700, 600, 500, 400, 0]
        step_value = config.step
        if step_value <= 1:
            config.denoising_step_list = [700, 0]
        elif step_value == 2:
            config.denoising_step_list = [700, 500, 0]
        elif step_value == 3:
            config.denoising_step_list = [700, 600, 400, 0]
        else:
            config.denoising_step_list = full_denoising_list

        self.prompt = params.prompt
        self.args = config
        self.prepare()

    def prepare(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_event = Event()
        self.stop_event = Event()
        self.restart_event = Event()
        self.prompt_dict = Manager().dict()
        self.prompt_dict["prompt"] = self.prompt
        self.process = Process(
            target=generate_process,
            args=(self.args, self.prompt_dict, self.prepare_event, self.restart_event, self.stop_event, self.input_queue, self.output_queue),
            daemon=True
        )
        self.process.start()
        self.processes = [self.process]
        self.prepare_event.wait()

    def accept_new_params(self, params: "Pipeline.InputParams"):
        if hasattr(params, "image"):
            image_array = image_to_array(params.image, self.args.width, self.args.height)
            self.input_queue.put(image_array)

        if hasattr(params, "prompt") and params.prompt and self.prompt != params.prompt:
            self.prompt = params.prompt
            self.prompt_dict["prompt"] = self.prompt

        if hasattr(params, "restart") and params.restart:
            self.restart_event.set()
            clear_queue(self.output_queue)

    def produce_outputs(self) -> List[Image.Image]:
        qsize = self.output_queue.qsize()
        results = []
        for _ in range(qsize):
            results.append(array_to_image(self.output_queue.get()))
        return results

    def close(self):
        print("Setting stop event...")
        self.stop_event.set()

        print("Waiting for processes to terminate...")
        for i, process in enumerate(self.processes):
            process.join(timeout=1.0)
            if process.is_alive():
                print(f"Process {i} didn't terminate gracefully, forcing termination")
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    print(f"Force killing process {i}")
                    process.kill()
        print("Pipeline closed successfully")


def generate_process(args, prompt_dict, prepare_event, restart_event, stop_event, input_queue, output_queue):
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{args.gpu_ids.split(',')[0]}")

    pipeline_manager = SingleGPUInferencePipeline(args, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    base_chunk_size = pipeline_manager.base_chunk_size
    chunk_size = base_chunk_size * args.num_frame_per_block
    first_batch_num_frames = 1 + chunk_size
    is_running = False
    input_batch = 0
    prompt = prompt_dict["prompt"]

    prepare_event.set()

    while not stop_event.is_set():
        # Prepare first batch
        if not is_running or prompt_dict["prompt"] != prompt or restart_event.is_set():
            prompt = prompt_dict["prompt"]
            if restart_event.is_set():
                clear_queue(input_queue)
                restart_event.clear()
            images = read_images_from_queue(input_queue, first_batch_num_frames, device, stop_event, dynamic_batch=False)

            noise_scale = args.noise_scale
            init_noise_scale = args.noise_scale

            pipeline_manager.pipeline.vae.model.first_encode = True
            pipeline_manager.pipeline.vae.model.first_decode = True
            pipeline_manager.pipeline.kv_cache1 = None
            pipeline_manager.pipeline.crossattn_cache = None
            pipeline_manager.pipeline.block_x = None
            pipeline_manager.pipeline.hidden_states = None
            latents = pipeline_manager.pipeline.vae.stream_encode(images)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

            # Prepare pipeline
            current_start = 0
            current_end = pipeline_manager.pipeline.frame_seq_length * (1 + chunk_size//4)
            if pipeline_manager.pipeline.kv_cache1 is not None:
                pipeline_manager.pipeline.reset_kv_cache()
                pipeline_manager.pipeline.reset_crossattn_cache()
            denoised_pred = pipeline_manager.prepare_pipeline(
                text_prompts=[prompt],
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end
            )

            video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            for image in video.cpu().float().numpy():
                output_queue.put(image)

            current_start = current_end
            current_end += (chunk_size // base_chunk_size) * pipeline_manager.pipeline.frame_seq_length
            last_image = images[:,:,[-1]]
            processed = 0
            is_running = True

        if current_start//pipeline_manager.pipeline.frame_seq_length >= pipeline_manager.t_refresh:
            current_start = pipeline_manager.pipeline.kv_cache_length - pipeline_manager.pipeline.frame_seq_length
            current_end = current_start + (chunk_size // base_chunk_size) * pipeline_manager.pipeline.frame_seq_length

        if input_batch == 0:
            images = read_images_from_queue(input_queue, chunk_size, device, stop_event, dynamic_batch=True)
            num_frames = images.shape[2]
            input_batch = num_frames // chunk_size
        
            noise_scale, current_step = compute_noise_scale_and_step(
                input_video_original=torch.cat([last_image, images], dim=2),
                end_idx=num_frames +1,
                chunk_size=num_frames ,
                noise_scale=float(noise_scale),
                init_noise_scale=float(init_noise_scale),
            )

            latents = pipeline_manager.pipeline.vae.stream_encode(images)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        
        denoised_pred = pipeline_manager.pipeline.inference_stream(
            noise=noisy_latents[:, -input_batch].unsqueeze(1),
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
        )
        input_batch-=1

        processed += 1
        
        # VAE decoding - only start decoding after num_steps
        if processed >= num_steps:
            video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            # Update timing
            for image in video.cpu().float().numpy():
                output_queue.put(image)

        current_start = current_end
        current_end += (chunk_size // base_chunk_size) * pipeline_manager.pipeline.frame_seq_length
        last_image = images[:,:,[-1]]
