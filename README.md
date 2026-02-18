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
- **[2026-01-26]** 🎉 [StreamDiffusionV2](https://arxiv.org/abs/2511.07399) is accepted by MLSys 2026!
- **[2025-11-10]** 🚀 We have released our [paper](https://arxiv.org/abs/2511.07399) at arXiv. Check it for more details!
- **[2025-10-18]** Release our model checkpoint on [huggingface](https://huggingface.co/jerryfeng/StreamDiffusionV2/).
- **[2025-10-06]** 🔥 Our [StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2) is publicly released! Check our project [homepage](https://streamdiffusionv2.github.io/) for more details.

## Prerequisites

- OS: Linux with NVIDIA GPU
- CUDA-compatible GPU and drivers

## Installation

```shell
conda create -n stream python=3.10.0
conda activate stream
# Require CUDA 12.4 or above, please check via `nvcc -V`
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt 
python setup.py develop
```

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

## Offline Inference

### Single GPU

```shell
python streamv2v/inference.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/original.mp4 \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
```
Note: `--step` sets how many denoising steps are used during inference.

### Multi-GPU

```shell
torchrun --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/original.mp4 \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
# --schedule_block  # optional: enable block scheduling
```
Note: `--step` sets how many denoising steps are used during inference. Enabling `--schedule_block` can provide optimal throughput.

Adjust `--nproc_per_node` to your GPU count. For different resolutions or FPS, change `--height`, `--width`, and `--fps` accordingly.

## Online Inference (Web UI)
A minimal web demo is available under `demo/`. For setup and startup, please refer to [demo](demo/README.md).
- Access in a browser after startup: `http://0.0.0.0:7860` or `http://localhost:7860`


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
