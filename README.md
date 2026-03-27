# StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation (MLSys 2026)

[Tianrui Feng](https://jerryfeng2003.github.io/)<sup>1</sup>, [Zhi Li](https://scholar.google.com/citations?user=C6kPjgwAAAAJ&hl)<sup>2</sup>, [Shuo Yang](https://andy-yang-1.github.io/)<sup>2</sup>, [Haocheng Xi](https://haochengxi.github.io/)<sup>2</sup>, [Muyang Li](https://lmxyy.me/)<sup>3</sup>, [Xiuyu Li](https://xiuyuli.com/)<sup>1</sup>, [Lvmin Zhang](https://lllyasviel.github.io/lvmin_zhang/)<sup>4</sup>, [Keting Yang](https://www.linkedin.com/in/kellyzpeng/)<sup>5</sup>, [Kelly Peng](https://www.linkedin.com/in/kellyzpeng/)<sup>6</sup>, [Song Han](https://hanlab.mit.edu/songhan)<sup>7</sup>, [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/)<sup>4</sup>, [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/)<sup>2</sup>, [Akio Kodaira](https://scholar.google.com/citations?hl=ja&user=15X3cioAAAAJ)<sup>8</sup>, [Chenfeng Xu](https://www.chenfengx.com/)<sup>†,1</sup>

<sup>1</sup>UT Austin, <sup>2</sup>UC Berkeley, <sup>3</sup>Nunchaku AI, <sup>4</sup>Stanford University, <sup>5</sup>Independent Researcher, <sup>6</sup>First Intelligence, <sup>7</sup>MIT, <sup>8</sup>Shizhuku AI

<sup>†</sup> Project lead, corresponding to [xuchenfeng@utexas.edu](mailto:xuchenfeng@utexas.edu)

[![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://streamdiffusionv2.github.io/) [![arXiv](https://img.shields.io/badge/Arxiv-2511.07399-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.07399) [![Hugging Face](https://img.shields.io/badge/HuggingFace-Space-blue.svg?logo=huggingface)](https://huggingface.co/jerryfeng/StreamDiffusionV2)

<p align="center">
  <image src="./assets/demo-1.gif" controls width="800">
  <image src="./assets/demo-2.gif" controls width="800">
  <image src="./assets/demo-3.gif" controls width="800">
</p>

## Overview

StreamDiffusionV2 is an open-source interactive diffusion pipeline for real-time streaming applications. It scales across diverse GPU setups, supports flexible denoising steps, and delivers high FPS for creators and platforms. Further details are available on our project [homepage](https://streamdiffusionv2.github.io/).

## News
- **[2026-03-27]** StreamDiffusionV2 is now available on [PyPI](https://pypi.org/project/streamdiffusionv2/). Install the environment via `pip install streamdiffusionv2`.
- **[2026-03-27]** Added optional TAEHV-VAE support for inference via `--use_taehv` and `USE_TAEHV=1`.
- **[2026-03-06]** Update Ring-buffer KV Cache for efficient sliding window attention.
- **[2026-01-26]** 🎉 [StreamDiffusionV2](https://arxiv.org/abs/2511.07399) is accepted by MLSys 2026!
- **[2025-11-10]** 🚀 We have released our [paper](https://arxiv.org/abs/2511.07399) at arXiv. Check it for more details!
- **[2025-10-18]** Release our model checkpoint on [huggingface](https://huggingface.co/jerryfeng/StreamDiffusionV2/).
- **[2025-10-06]** 🔥 Our [StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2) is publicly released! Check our project [homepage](https://streamdiffusionv2.github.io/) for more details.

## Prerequisites

- OS: Linux with NVIDIA GPU
- CUDA-compatible GPU and drivers

## Installation

```shell
conda create -n streamdiffusionv2 python=3.10 -y
conda activate streamdiffusionv2

# PyPI
pip install streamdiffusionv2

# Optional but recommended for better throughput
pip install "streamdiffusionv2[flash-attn]"
```

If you are installing from a local checkout of this repository instead of PyPI:

```shell
conda create -n streamdiffusionv2 python=3.10
conda activate streamdiffusionv2
pip install .

# Optional but recommended for better throughput
pip install ".[flash-attn]"
```

The package install includes the Python dependencies required for both offline inference and the demo backend. The demo frontend still requires Node.js 18 as described in [demo/README.md](demo/README.md).

## Download Checkpoints

```shell
# 1.3B Model
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v/*"

# 14B Model
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v_14b/*"
```
We use the 14B model from [CausVid-Plus](https://github.com/GoatWu/CausVid-Plus) for offline inference demo.

### Optional: TAEHV-VAE Checkpoint

If you want to enable the lightweight TAEHV decoder, download its checkpoint once:

```shell
curl -L https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o ckpts/taew2_1.pth
```

The offline inference code can also download this file automatically on first use, but keeping it in `ckpts/taew2_1.pth` avoids that extra startup step.

## Usage Example

We provide a simple example of how to use StreamDiffusionV2. For more detailed examples, please refer to [streamv2v](./streamv2v/) directory.

### Single GPU

```python
import numpy as np

from streamdiffusionv2 import StreamDiffusionV2Pipeline, export_video, load_video

stream = StreamDiffusionV2Pipeline(
    checkpoint_folder="ckpts/wan_causal_dmd_v2v",
    mode="single",
)
stream.prepare("A dog walks on the grass, realistic")

video = load_video("examples/original.mp4", height=480, width=832)
decoded_chunks = []
noise_scale = stream.noise_scale

for video_chunk in stream.chunk_video(video):
    encoded_chunk = stream.encode_chunk(
        video,
        video_chunk,
        previous_noise_scale=noise_scale,
        initial_noise_scale=stream.noise_scale,
    )
    noise_scale = encoded_chunk.noise_scale
    denoised_chunk = stream.denoise_chunk(encoded_chunk)
    if denoised_chunk is None:
        continue
    decoded_chunks.append(stream.decode_chunk(denoised_chunk))

output = np.concatenate(decoded_chunks, axis=0)
export_video(output, "outputs/python_single.mp4", fps=16)
```

### Single GPU Without Stream-Batch

```python
import numpy as np

from streamdiffusionv2 import StreamDiffusionV2Pipeline, export_video, load_video

stream = StreamDiffusionV2Pipeline(
    checkpoint_folder="ckpts/wan_causal_dmd_v2v",
    mode="single-wo",
)
stream.prepare("A dog walks on the grass, realistic")

video = load_video("examples/original.mp4", height=480, width=832)
decoded_chunks = []
noise_scale = stream.noise_scale

for video_chunk in stream.chunk_video(video):
    encoded_chunk = stream.encode_chunk(
        video,
        video_chunk,
        previous_noise_scale=noise_scale,
        initial_noise_scale=stream.noise_scale,
    )
    noise_scale = encoded_chunk.noise_scale
    denoised_chunk = stream.denoise_chunk(encoded_chunk)
    if denoised_chunk is None:
        continue
    decoded_chunks.append(stream.decode_chunk(denoised_chunk))

output = np.concatenate(decoded_chunks, axis=0)
export_video(output, "outputs/python_single_wo.mp4", fps=16)
```

### Multi-GPU Pipeline

Pipeline-parallel inference still launches multiple worker processes, so the Python API for that mode stays as one imported function:

```python
from streamdiffusionv2 import run_video_to_video

run_video_to_video(
    mode="pipe",
    checkpoint_folder="ckpts/wan_causal_dmd_v2v",
    video_path="examples/original.mp4",
    prompt="A dog walks on the grass, realistic",
    output_path="outputs/python_pipe.mp4",
    gpu_ids=[0, 1],
    num_gpus=2,
)
```

### Optional Acceleration

The staged API can be reconfigured before `prepare(...)`:

```python
from streamdiffusionv2 import StreamDiffusionV2Pipeline

stream = StreamDiffusionV2Pipeline(checkpoint_folder="ckpts/wan_causal_dmd_v2v")
stream.enable_acceleration(fast=True)
stream.prepare("A dog walks on the grass, realistic")
```

`fast=True` enables `use_taehv` and `use_tensorrt`, and it automatically switches the default config from `wan_causal_dmd_v2v.yaml` to `wan_causal_dmd_v2v_fast.yaml`.

## Offline Inference

All offline inference entrypoints are unified under `run_v2v.sh`.

Choose one mode first:

- `single`: single-GPU streaming inference
- `single-wo`: single-GPU inference without Stream-batch
- `pipe`: multi-GPU pipeline inference

Quick start:

```shell
./run_v2v.sh single
./run_v2v.sh single-wo
./run_v2v.sh pipe
./run_v2v.sh pipe --profile
```

Use `--profile` only when you want synchronized throughput measurements.

The legacy wrappers `v2v.sh`, `v2v_wo.sh`, and `pipe_v2v.sh` still work, but they now forward to the same shared entrypoint.

### Common Arguments

The most important options are:

- `--config_path`: model config YAML
- `--checkpoint_folder`: checkpoint directory
- `--video_path`: input video
- `--prompt_file_path`: prompt text file
- `--output_folder`: output directory
- `--height` and `--width`: output resolution
- `--fps`: target output FPS
- `--step`: number of denoising steps used during inference
- `--use_taehv`: use Wan stream encode with the TAEHV decoder for faster VAE decoding

You can pass overrides either as CLI flags or as environment variables. For example:

```shell
OUTPUT_FOLDER=outputs/run_single ./run_v2v.sh single
VIDEO_PATH=examples/original.mp4 PROMPT_FILE_PATH=examples/prompt.txt ./run_v2v.sh single-wo
NPROC_PER_NODE=2 MASTER_PORT=29511 ./run_v2v.sh pipe
./run_v2v.sh single --use_taehv
```

### Single GPU

This is the standard offline path when you run on one GPU.

```shell
./run_v2v.sh single \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/prompt.txt \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
```

To enable the TAEHV decoder in this mode:

```shell
./run_v2v.sh single --use_taehv
```

### Multi-GPU

Use this mode when you want to split inference across multiple GPUs.

```shell
./run_v2v.sh pipe \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/prompt.txt \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
# --schedule_block  # optional: enable block scheduling
```

To enable the TAEHV decoder in pipeline mode:

```shell
./run_v2v.sh pipe --use_taehv
```

Notes:

- `--schedule_block` is optional and can improve throughput on some multi-GPU setups.
- Adjust `NPROC_PER_NODE`, `--height`, `--width`, and `--fps` to match your hardware and target workload.
- `./run_v2v.sh pipe --profile` is intended for profiling runs, not normal benchmarking or deployment.

## Online Inference (Web UI)
A minimal web demo is available under `demo/`. For setup and startup, please refer to [demo](demo/README.md).
- Access in a browser after startup: `http://0.0.0.0:7860` or `http://localhost:7860`
- To enable the TAEHV decoder in the web demo, start it with `USE_TAEHV=1`.


## To-do List

- [x] Demo and inference pipeline.
- [ ] Dynamic scheduler for various workload.
- [ ] Training code.
- [ ] FP8 support.
- [ ] TensorRT support.

## Acknowledgements
StreamDiffusionV2 is inspired by the prior works [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) and [StreamV2V](https://github.com/Jeff-LiangF/streamv2v). Our Causal DiT builds upon [CausVid](https://github.com/tianweiy/CausVid), and the rolling KV cache design is inspired by [Self-Forcing](https://github.com/guandeh17/Self-Forcing).

We are grateful to the team members of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for their support. We also thank [First Intelligence](https://first-intelligence.com) and [Daydream](https://docs.daydream.live/) team for their great feedback.

We also especially thank DayDream team for the great collaboration and incorporating our StreamDiffusionV2 pipeline into their cool [Demo UI](https://github.com/daydreamlive/scope). 

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ or a citation.
```BibTeX
@article{feng2025streamdiffusionv2,
  title={StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation},
  author={Feng, Tianrui and Li, Zhi and Yang, Shuo and Xi, Haocheng and Li, Muyang and Li, Xiuyu and Zhang, Lvmin and Yang, Keting and Peng, Kelly and Han, Song and others},
  journal={arXiv preprint arXiv:2511.07399},
  year={2025}
}
```
