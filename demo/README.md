# Vid2Vid demo with a camera

This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs video-to-video with a live webcam feed or screen capture on a web browser.

Tested with VSCode with remote GPU server. The Live Server extension would help to open the local Chorme page.

## Usage

### install Node.js 18+

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
# Restart terminal
nvm install 18
```

### Download lora weights for better stylization

```bash
pip install gdown
pip install --upgrade gdown
cd demo_w_camera
gdown https://drive.google.com/drive/folders/1D7g-dnCQnjjogTPX-B3fttgdrp9nKeKw -O lora_weights --folder
```

| Trigger words                                            | LORA weights     | Source      |
|----------------------------------------------------------|------------------|-------------|
| 'pixelart' ,  'pixel art' ,  'Pixel art' ,  'PixArFK'    | [Google drive](https://drive.google.com/file/d/1_-kEVFw_LnV1J2Nho6nZt4PUbymamypK/view?usp=drive_link) | [Civitai](https://civitai.com/models/185743/8bitdiffuser-64x-or-a-perfect-pixel-art-model) |
| 'lowpoly', 'low poly', 'Low poly'                        | [Google drive](https://drive.google.com/file/d/1ZClfRljzKmxsU1Jj5OMwIuXQcnA1DwO9/view?usp=drive_link) | [Civitai](https://civitai.com/models/110435/y5-low-poly-style) |
| 'Claymation', 'claymation'                               | [Google drive](https://drive.google.com/file/d/1GvPCbrPqJYj0_nRppSc2UD_1eRME-1tG/view?usp=drive_link) | [Civitai](https://civitai.com/models/25258/claymation-miniature) |
| 'crayons', 'Crayons', 'crayons doodle', 'Crayons doodle' | [Google drive](https://drive.google.com/file/d/12ZMOy8CMzwB32RHSmff0h2TJC3lFDBmW/view?usp=drive_link) | [Civitai](https://civitai.com/models/90558/child-type-doodles) |
| 'sketch', 'Sketch', 'pencil drawing', 'Pencil drawing'   | [Google drive](https://drive.google.com/file/d/1NIBujegFMvFdjCW0vdrmD6fbNFKNROE4/view?usp=drive_link) | [Civitai](https://civitai.com/models/155490/pencil-sketch-or) |
| 'oil painting', 'Oil painting'                           | [Google drive](https://drive.google.com/file/d/1fmS3fGeja0RM8YbZtbKw20fjXNzHrnxz/view?usp=drive_link) | [Civitai](https://civitai.com/models/84542/oil-paintingoil-brush-stroke) |


### Install

You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
python main.py 
```

or 

```
chmod +x start.sh
./start.sh
```

then open `http://0.0.0.0:7860` in your browser. (*If `http://0.0.0.0:7860` does not work well, try `http://localhost:7860`)

### Common Bugs

#### Camera Not Enabled Issue
- **Error Message**: `Cannot read properties of undefined (reading 'enumerateDevices')`.
- **Related GitHub Issue**: [No webcam detected](https://github.com/radames/Real-Time-Latent-Consistency-Model/issues/17)

**Potential Workaround**:  
This issue occurs when the camera is not allowed for the web browser. Add ```http://localhost:7860,http://0.0.0.0:7860``` to [chorme setting](https://github.com/radames/Real-Time-Latent-Consistency-Model/issues/17#issuecomment-1811957196).
