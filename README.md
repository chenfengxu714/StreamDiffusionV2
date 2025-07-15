# StreamDiffusionV2

Installation
```shell
conda create -n stream python=3.10.0
conda activate stream
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt 
python setup.py develop
```

Download Checkpoints
```shell
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download --resume-download tianweiy/CausVid --local-dir ./ckpts --include autoregressive_checkpoint/*
```

Stream V2V Inference
```shell
bash v2v_inference
```