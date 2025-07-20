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

### Install

You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

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
