# StreamDiffusionV2

Installation
```shell
conda create -n dcm python=3.10.0
conda activate dcm
git clone https://github.com/Vchitect/DCM
cd DCM
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install flash-attn --no-build-isolation
```

Download Checkpoints
```shell
huggingface-cli download --resume-download cszy98/DCM --local-dir ./ckpt
```

Inference
```shell
bash scripts/inference/inference_wan.sh
```