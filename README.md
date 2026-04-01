# Real-Time Streaming Adaptation of SAM3

![sam3-real-time](https://github.com/user-attachments/assets/f2a01527-c1c8-486f-9a10-cbed1ca6a2b4)

This repository transforms [SAM3](https://github.com/facebookresearch/sam3)’s offline video inference into a live, real-time streaming pipeline. Instead of preloading and processing an entire video sequence offline, it ingests frames incrementally and performs per-frame inference on the fly. This allows SAM3 to work with any live video source (e.g. webcams, RTSP streams, [YARP](https://www.yarp.it/latest/index.html) ports), enabling online operation and expanding use cases to robotics, teleoperation, live surveillance, AR/VR and real-time content creation.

## Latest updates

**03/27/2026 -- SAM 3.1 Object Multiplex is released. It introduces a shared-memory approach for joint multi-object tracking that is significantly faster without sacrificing accuracy.**

- A new suite of improved model checkpoints (denoted as **SAM 3.1**) are released on [Hugging Face](https://huggingface.co/facebook/sam3.1). See [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md) for full details.
  * To use the new SAM 3.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.

## Latest updates

**03/27/2026 -- SAM 3.1 Object Multiplex is released. It introduces a shared-memory approach for joint multi-object tracking that is significantly faster without sacrificing accuracy.**

- A new suite of improved model checkpoints (denoted as **SAM 3.1**) are released on [Hugging Face](https://huggingface.co/facebook/sam3.1). See [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md) for full details.
  * To use the new SAM 3.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.

## Installation
This fork uses **Pixi** for environment management to provide fully reproducible setups with fast and stable dependency resolution.

### 1. Install Pixi (if needed)
As of December 2025, you can install Pixi on Linux and MacOS via:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
Or if you prefer using `wget`:
```bash
wget -qO- https://pixi.sh/install.sh | sh
```
For Windows or more recent instructions, visit the [Pixi installation guide](https://pixi.sh/dev/installation/).

### 2. Create the environment
From the repository root:

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Enter the environment
```bash
pixi shell
```

### 4. Verify SAM3 loads from this fork
```bash
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

5. **Optional dependencies for faster inference**
```bash
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

## Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)


### Demo Notebook
Mirroring the original demo notebooks, `examples/sam3_stream_predictor_example.ipynb` demonstrates how to run SAM3 in real-time on a video stream. The notebook loads a video file, starts a streaming session, adds a text prompt on frame 0 and pushes frames incrementally, running per-frame inference with optional visualization and FPS reporting.

### CLI Script
For a command-line run, use `scripts/inference/video_stream.py`. It mirrors the notebook flow: opens a live source (webcam/video/YARP), starts a streaming session, adds a text prompt on the first frame and performs per-frame inference (with optional visualization and saving). Note that you can also potentially add more textual prompts in later frames.

- Basic webcam example:
  - `python scripts/inference/video_stream.py --stream_type webcam --webcam_index 0 --viz_results --save_video`
- Flags of interest:
  - `--stream_type {webcam|video|yarp}`: choose the input source
  - `--video_path PATH`: path to a video file when `--stream_type video`
  - `--viz_results`: display live overlays
  - `--save_images` / `--save_video`: store outputs under `outputs/<run_id>/`
  - `--run_output_name NAME`: set a custom run id (else datetime is used)

## Current Limitations and Known Issues
- **Single-GPU streaming:** The provided streaming predictor targets one GPU. Multi-GPU support exists in the base model but isn’t integrated into the streaming predictor yet.
- **Memory leaking**: There seems to be a memory leak during long streaming sessions. The cause is under investigation. I got an OOM after ~5 minutes of streaming at 480p on a NVIDIA GeForce RTX 3090.
