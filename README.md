# ComfyUI-MASt3R

ComfyUI custom nodes for [MASt3R (Matching And Stereo 3D Reconstruction)](https://github.com/naver/mast3r) - a state-of-the-art 3D reconstruction model from Naver Corporation.

MASt3R extends DUSt3R with dense local feature matching capabilities, providing significantly improved 3D reconstruction quality, especially for challenging scenes with repetitive patterns or complex geometry.

![MASt3R](https://github.com/naver/mast3r/raw/main/assets/mast3r.jpg)

## Features

- **MASt3R Model Loader**: Loads the heavy ViT-Large model with efficient memory handling.
- **MASt3R 3D Reconstruction**: The core node. Takes images, runs global alignment + local refinement, and outputs a `.glb` file.
- **MASt3R Scene to Depth Maps**: Extracts high-quality depth maps from the reconstructed scene for use in other workflows (ControlNet, etc.).
- **MASt3R Scene to Camera Poses**: Extracts estimated camera trajectories for animation or video production.

---

## Installation

### 1. Install the Nodes

**Option A: Git Clone (Recommended)**
Navigate to your `ComfyUI/custom_nodes` folder and run:
```bash
git clone [https://github.com/jfirma1/ComfyUI-mast3r.git](https://github.com/jfirma1/ComfyUI-mast3r.git)
cd ComfyUI-mast3r
pip install -r requirements.txt
