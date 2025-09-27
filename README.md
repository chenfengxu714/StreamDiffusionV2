# StreamDiffusionV2

## Overview
StreamDiffusionV2 provides streaming video-to-video (V2V) and related inference utilities. This README covers installation, checkpoint setup, and how to run single- and multi-GPU inference. A simple web demo is also available.

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
```
Then download the checkpoint of generator from [Google Drive](https://drive.google.com/drive/folders/1YpOObikpsiNBsfTVv1w4EIbegE_UglY2?usp=sharing) and put into ckpts/wan_causal_dmd_v2v. 

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

### Multi-GPU (single node)
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
Note: `--step` sets how many denoising steps are used during inference. On NVIDIA H100 GPUs, enabling `--schedule_block` can provide optimal throughput.

Adjust `--nproc_per_node` to your GPU count. For different resolutions or FPS, change `--height`, `--width`, and `--fps` accordingly.

## Online Inference (Web UI)
A minimal web demo is available under `demo/`. For setup and startup, please refer to [demo/README.md](demo/README.md).
- Access in a browser after startup: `http://0.0.0.0:7860` or `http://localhost:7860`
