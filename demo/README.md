# StreamDiffusionV2 Demo (Web UI)

This demo provides a simple web interface for live video-to-video inference using the backend in this repository. It supports webcam or screen capture input in the browser.

## Prerequisites
- Python 3.10 with the root package installed via `pip install streamdiffusionv2` or `pip install .`
- Node.js 18
- NVIDIA GPU recommended (single or multi-GPU)

## Setup
1) Complete the Python package install and checkpoint setup from the root `README.md`.
2) Build the frontend and start the backend via the script:
```shell
# Install
cd demo
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 18

cd frontend
npm install
npm run build
cd ../

# Start
chmod +x start.sh
./start.sh
```
The script will:
- Install and build the frontend (`npm install && npm run build` in `demo/frontend`)
- Launch the backend on port `7860` and host `0.0.0.0`

You can override the runtime configuration through environment variables, for example:
```shell
HOST=0.0.0.0 PORT=7860 GPU_IDS=4 STEP=2 ./start.sh
```

## Enable TAEHV-VAE

The demo supports the TAEHV decoder for online inference.

1) Download the checkpoint once:
```shell
curl -L https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o ../ckpts/taew2_1.pth
```

2) Start the demo with `USE_TAEHV=1`:
```shell
USE_TAEHV=1 ./start.sh
```

You can combine it with your normal launch overrides, for example:
```shell
HOST=0.0.0.0 PORT=7860 GPU_IDS=4 STEP=2 USE_TAEHV=1 ./start.sh
```
## Access
- Local: `http://0.0.0.0:7860` or `http://localhost:7860`
- Remote server: `http://<server-ip>:7860` (ensure the port is open)

## Multi-GPU and Denoising Timesteps
- `start.sh` derives `num_gpus` from `GPU_IDS`. To enable multi-GPU inference on a single node or adjust denoising steps, override the environment variables. For example:
```shell
GPU_IDS=4,5 STEP=2 ./start.sh
```
With TAEHV enabled:
```shell
GPU_IDS=4,5 STEP=2 USE_TAEHV=1 ./start.sh
```
Our model supports denoising steps from 1 to 4 — feel free to set this value as needed.  
For real-time live-streaming applications, we found that using **2 steps** provides a good balance between speed and quality.


## Troubleshooting
- Camera not available:
  - Allow camera/microphone access for the site in your browser.
  - Error example: `Cannot read properties of undefined (reading 'enumerateDevices')`.
- Frontend not reachable:
  - Ensure the build succeeded (look for `frontend build success`).
  - Check that port 7860 is free, or adjust the port in the script and visit the new port.
  - For remote servers, open the port in firewall/security group.
- Model errors:
  - Verify that all checkpoints were downloaded and placed in the expected directories.
  - If `USE_TAEHV=1` is enabled, verify that `ckpts/taew2_1.pth` exists.

For advanced usage and CLI-based inference, see the root `README.md` (single-GPU and multi-GPU inference scripts).
