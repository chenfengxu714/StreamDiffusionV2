# pip install git+https://github.com/openai/CLIP.git
# pip install open_clip_torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler, DDIMScheduler
from causvid.util import launch_distributed_job
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
import os
import re


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
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--scheduler", type=str, choices=['ddim', 'lcm'], default='lcm')

    args = parser.parse_args()

    # Step 1: Setup the environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Step 2: Create the generator
    launch_distributed_job()
    device = torch.cuda.current_device()

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32).to(device)
    if args.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing")
    elif args.scheduler == "lcm":
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

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

    os.makedirs(args.output_dir, exist_ok=True)

    def sanitize_filename(name):
        """
        Remove any characters that are not alphanumeric, spaces, underscores, or hyphens.
        Then replace spaces with underscores.
        """
        # Remove unwanted characters (anything not a word character, space, or hyphen)
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores and strip leading/trailing whitespace
        return name.strip().replace(' ', '_')

    for idx, (img_array, prompt) in enumerate(zip(data_dict['all_images'], data_dict['all_captions'])):
        # Split the prompt into words and take the first four words.
        words = prompt.split()
        if len(words) >= 10:
            base_name = ' '.join(words[:10])
        else:
            base_name = ' '.join(words)

        # Sanitize the base file name to remove problematic characters.
        base_name = sanitize_filename(base_name)

        # Append the index to ensure uniqueness (in case two prompts share the same first four words).
        file_name = f"{base_name}_{idx}.jpg"

        # Create a PIL Image from the numpy array.
        image = Image.fromarray(img_array)

        # Save the image in the specified folder.
        image.save(os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    main()
