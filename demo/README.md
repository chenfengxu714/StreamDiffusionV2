# StreamDiffusionV2 Demo (Web UI)

This demo provides a simple web interface for live video-to-video inference using the backend in this repository. It supports webcam or screen capture input in the browser.

## Prerequisites
- Python 3.10 (follow the root README for environment setup)
- Node.js 18
- NVIDIA GPU recommended (single or multi-GPU)

## Setup
1) Complete the Python environment and model checkpoint setup as described in the root `README.md` (Installation and Download Checkpoints).
2) Build the frontend and start the backend via the script:
```
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
- Launch the backend with `torchrun` on port `7860` and host `0.0.0.0`

## Access
- Local: `http://0.0.0.0:7860` or `http://localhost:7860`
- Remote server: `http://<server-ip>:7860` (ensure the port is open)

## Multi-GPU and Denoising Timesteps
- Edit `start.sh`: By default, `start.sh` uses `num_gpus=1`. To enable multi-GPU inference on a single node or adjust the denoising steps, edit the corresponding variables in `start.sh`.  For example:
```
python main.py --port 7860 --host 0.0.0.0 --num_gpus 2 --gpu_ids 0,1 --step 2
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

For advanced usage and CLI-based inference, see the root `README.md` (single-GPU and multi-GPU inference scripts).
