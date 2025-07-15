from causvid.models.wan.bidirectional_inference import BidirectionalInferencePipeline
from huggingface_hub import hf_hub_download
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipe = BidirectionalInferencePipeline(config, device="cuda")

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipe.generator.load_state_dict(state_dict)

pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

dataset = TextDataset(args.prompt_file_path)

os.makedirs(args.output_folder, exist_ok=True)

for index in tqdm(range(len(dataset))):
    prompt = dataset[index]
    video = pipe.inference(
        noise=torch.randn(
            1, 21, 16, 60, 104, generator=torch.Generator(device="cuda").manual_seed(42),
            dtype=torch.bfloat16, device="cuda"
        ),
        text_prompts=[prompt]
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{index:03d}.mp4"), fps=16)
