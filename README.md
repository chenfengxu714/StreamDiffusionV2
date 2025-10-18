# StreamDiffusionV2: An Open-Source Streaming System for Real-Time Interactive Video Generation

[Tianrui Feng](https://jerryfeng2003.github.io/)<sup>*,1</sup>, [Zhi Li](https://scholar.google.com/citations?user=C6kPjgwAAAAJ&hl)<sup>1</sup>, [Haocheng Xi](https://haochengxi.github.io/)<sup>1</sup>, [Muyang Li](https://lmxyy.me/)<sup>2</sup>, [Shuo Yang](https://andy-yang-1.github.io/)<sup>1</sup>, [Xiuyu Li](https://xiuyuli.com/)<sup>1</sup>, [Lvmin Zhang](https://lllyasviel.github.io/lvmin_zhang/)<sup>3</sup>, [Kelly Peng](https://www.linkedin.com/in/kellyzpeng/)<sup>4</sup>, [Song Han](https://hanlab.mit.edu/songhan)<sup>2</sup>, [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/)<sup>3</sup>, [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/)<sup>1</sup>, [Akio Kodaira](https://scholar.google.com/citations?hl=ja&user=15X3cioAAAAJ)<sup>1</sup>, [Chenfeng Xu](https://www.chenfengx.com/)<sup>†,1,5</sup>

<sup>1</sup>UC Berkeley   <sup>2</sup>MIT   <sup>3</sup>Stanford University   <sup>4</sup>First Intelligence   <sup>5</sup>UT Austin 

<sup>†</sup> Project lead, corresponding to [xuchenfeng@berkeley.edu](mailto:xuchenfeng@berkeley.edu)

<sup>*</sup> Work done when Tianrui Feng was a visiting student at UC Berkeley advised by Chenfeng Xu.

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://streamdiffusionv2.github.io/)

<p align="center">
  <image src="./assets/demo-1.gif" controls width="800">
  <image src="./assets/demo-2.gif" controls width="800">
  <image src="./assets/demo-3.gif" controls width="800">
</p>

## Overview

StreamDiffusionV2 is an open-source interactive diffusion pipeline for real-time streaming applications. It scales across diverse GPU setups, supports flexible denoising steps, and delivers high FPS for creators and platforms. Further details are available on our project [homepage](https://streamdiffusionv2.github.io/).

## News
- **[2025-10-18]** Release our model checkpoint on [huggingface](https://huggingface.co/jerryfeng/StreamDiffusionV2/)
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
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts/wan_causal_dmd_v2v
```

## Offline Inference

### Single GPU

```shell
python streamv2v/inference.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path prompt.txt \
--video_path original.mp4 \
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
--prompt_file_path prompt.txt \
--video_path original.mp4 \
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
@article{streamdiffusionv2,
  title={StreamDiffusionV2: An Open-Sourced Interactive Diffusion Pipeline for Streaming Applications},
  author={Tianrui Feng and Zhi Li and Haocheng Xi and Muyang Li and Shuo Yang and Xiuyu Li and Lvmin Zhang and Kelly Peng and Song Han and Maneesh Agrawala and Kurt Keutzer and Akio Kodaira and Chenfeng Xu},
  journal={Project Page},
  year={2025},
  url={https://streamdiffusionv2.github.io/}
}
```
