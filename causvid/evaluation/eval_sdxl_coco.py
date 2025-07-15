# pip install git+https://github.com/openai/CLIP.git
# pip install open_clip_torch
from causvid.evaluation.coco_eval.coco_evaluator import evaluate_model, compute_clip_score
from diffusers import DiffusionPipeline, LCMScheduler, DDIMScheduler
from causvid.util import launch_distributed_job
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
import os


def load_generator(checkpoint_path, generator):
    # sometime the state_dict is not fully saved yet
    counter = 0
    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")[
                'generator']
            break
        except:
            print(f"fail to load checkpoint {checkpoint_path}")
            time.sleep(1)

            counter += 1

            if counter > 100:
                return None

    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    print(generator.load_state_dict(state_dict, strict=True))
    return generator


def sample(pipeline, prompt_list, denoising_step_list, batch_size):
    num_prompts = len(prompt_list)
    num_steps = len(denoising_step_list)

    images = []
    all_prompts = []
    for i in tqdm(range(0, num_prompts, batch_size)):
        batch_prompt = prompt_list[i:i + batch_size]
        timesteps = None if isinstance(
            pipeline.scheduler, DDIMScheduler) else denoising_step_list
        batch_images = pipeline(prompt=batch_prompt, num_inference_steps=num_steps, timesteps=timesteps,
                                guidance_scale=0, output_type='np').images
        batch_images = (batch_images * 255.0).astype("uint8")
        images.extend(batch_images)
        all_prompts.extend(batch_prompt)

        torch.cuda.empty_cache()

    all_images = np.stack(images, axis=0)

    data_dict = {"all_images": all_images, "all_captions": all_prompts}

    return data_dict


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoising_step_list", type=int,
                        nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--eval_res", type=int, default=256)
    parser.add_argument("--scheduler", type=str,
                        choices=['ddim', 'lcm'], default='lcm')

    args = parser.parse_args()

    # Step 1: Setup the environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Step 2: Create the generator
    launch_distributed_job()
    device = torch.cuda.current_device()

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32).to(device)
    if args.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing")
    elif args.scheduler == "lcm":
        pipeline.scheduler = LCMScheduler.from_config(
            pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None

    # Step 3: Generate images
    prompt_list = []
    with open(args.prompt_path, "r") as f:
        for line in f:
            prompt_list.append(line.strip())

    generator = load_generator(os.path.join(
        args.checkpoint_path, "model.pt"), pipeline.unet)

    if generator is None:
        return

    pipeline.unet = generator
    data_dict = sample(pipeline, prompt_list,
                       args.denoising_step_list, args.batch_size)

    # Step 4: Evaluate the generated images

    # evaluate and write stats to file
    if dist.get_rank() == 0:
        fid = evaluate_model(
            args, device, data_dict["all_images"], patch_fid=False)

        clip_score = compute_clip_score(
            images=data_dict["all_images"],
            captions=data_dict["all_captions"],
            clip_model="ViT-G/14",
            device=device,
            how_many=len(data_dict["all_images"])
        )
        print(f"fid {fid} clip score {clip_score}")

        with open(os.path.join(args.checkpoint_path, "eval_stats.txt"), "w") as f:
            f.write(f"fid {fid} clip score {clip_score}")


if __name__ == "__main__":
    main()
