
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

```

**Option B: Download ZIP**

1. Download this repository as a ZIP file.
2. Extract it into `ComfyUI/custom_nodes/ComfyUI-mast3r`.
3. Open a terminal in that folder and run `pip install -r requirements.txt`.

### 2. Download Model Weights

You need the pre-trained MASt3R model weights. Download the file below and place it in the `ComfyUI-mast3r/checkpoints` folder.

**Command Line:**

```bash
cd ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints
wget [https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth)

```

**Manual Download:**
[Click here to download MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth)

---

## Usage

### Basic Workflow

1. **Load Model**: Add the `MASt3R Model Loader` node.
2. **Reconstruct**: Add the `MASt3R 3D Reconstruction` node.
* Connect the model.
* **Input Images**: You can either connect an image batch (from "Load Image") OR provide a folder path in `image_folder_path`.


3. **Output**: The node saves a `scene.glb` to your output folder and returns the scene object for further processing (depth maps/poses).

### Input Methods

* **`images` (Input)**: Best for small batches or generating images inside ComfyUI.
* **`image_folder_path` (String)**: Best for large datasets. Enter the absolute path to a folder containing your source images (PNG, JPG, BMP, WEBP).

---

## Recommended Settings

MASt3R's memory usage explodes based on two factors: **Image Resolution** and **Scene Graph Type**.

> **⚠️ CRITICAL WARNING:** Do not use `image_size: 1024` with `scenegraph_type: complete` if you have more than 10-12 images. You **will** run out of memory.

### 1. Safe Settings Guide (By GPU VRAM)

| GPU VRAM | Safe Image Size | Safe Image Count (`complete`) | Safe Image Count (`swin`) |
| --- | --- | --- | --- |
| **12GB - 16GB** | 224 - 512 | < 5 images | < 15 images |
| **24GB** (3090/4090) | 512 | 8 - 12 images | 30+ images |
| **48GB** (A40/A6000) | **512 - 768** | **20 - 30 images** | **100+ images** |
| **48GB** (A40/A6000) | **1024** | **< 10 images** | **40+ images** |
| **80GB+** (A100/H100) | 1024 | 20+ images | 100+ images |

### 2. Choosing Your Resolution

* **512 (Recommended)**: The sweet spot. Good geometry, stable memory usage. Use this for datasets with 20+ images.
* **768 (High Quality)**: Use this if you have a powerful GPU (24GB+) and fewer than 15 images.
* **1024 (Maximum)**: Only use this for **very small object scans** (<10 images) or **video sequences** using the `swin` graph.

### 3. Scene Graph Types

The scene graph determines how many pairs are calculated.

* **`complete` (Heavy)**: Matches *every* image to *every* other image ( pairs).
* **Use for:** Unordered collections, object scans.
* **Limit:** Keep under 20 images on 48GB VRAM (at 512px).


* **`swin` (Efficient)**: Matches images in a sliding window (e.g., Frame 1 matches 2, 3, 4).
* **Use for:** Video sequences, turntables.
* **Benefit:** Memory usage is linear. You can process 100+ images easily.


* **`oneref`**: Matches all images to a single reference image (defined by `refid`).
* **Use for:** Star-pattern camera arrays or specific "hero shot" setups.



### 4. Optimization Presets

| Preset Goal | Optim Level | Niter1 | Niter2 | Notes |
| --- | --- | --- | --- | --- |
| **Fast Preview** | `coarse` | 100 | 0 | Checks camera positions only. |
| **Standard** | `refine` | 300 | 200 | Good balance for most scenes. |
| **High Quality** | `refine+depth` | 500 | 300 | **Best for meshes.** Optimizes geometry and depth maps. |

---

## Parameter Guide

| Parameter | Description |
| --- | --- |
| **image_size** | **Crucial.** Controls memory usage and detail. Start at 512. |
| **optim_level** | Controls the pipeline depth. `refine+depth` produces the highest quality meshes. |
| **niter1** | **Global Alignment Iterations.** Increase if cameras are initialized in the wrong locations. |
| **niter2** | **Refinement Iterations.** Increase to sharpen the mesh and reduce noise. |
| **lr1** | **Stage 1 Learning Rate.** Controls how fast cameras move during alignment. Default `0.07`. |
| **lr2** | **Stage 2 Learning Rate.** Controls refinement precision. Default `0.01`. |
| **matching_conf_thr** | **Match Strictness (Default: 5.0).** Increase (>8.0) if you see "flying pixels" connecting unrelated objects. |
| **min_conf_thr** | **Output Cleanup (Default: 1.5).** Points with confidence lower than this are deleted. Increase to 2.5+ for cleaner, sparser clouds. |
| **TSDF_thresh** | **Mesh Smoothing (Default: 0.0).** If set to `0.01` - `0.05`, runs volume fusion to create a smooth, watertight mesh instead of a point cloud. |
| **mask_sky** | Set to `True` for outdoor scenes to automatically remove the sky dome (infinite depth points). |
| **shared_intrinsics** | Set to `True` if all images come from the exact same camera/lens at the same focal length. Improves stability. |
| **cam_size** | Controls the visual size of the camera frustums in the output GLB file. |

## Tips for Success

1. **Memory Management**: If you crash with an OOM (Out Of Memory) error, the first thing you should do is **lower `image_size**`. The second thing is to switch from `complete` to `swin` graph.
2. **Outdoor Scenes**: Always enable `mask_sky` for outdoor photos, or you will get a giant sphere of points surrounding your scene.
3. **Clean Meshes**: For a clean, usable 3D mesh (not just points), set `as_pointcloud` to `False` and `TSDF_thresh` to roughly `0.02`.
4. **Video Inputs**: If processing a video frame sequence, always use `scenegraph_type: swin` with a `winsize` of 3-5.

## Requirements

* PyTorch
* scipy
* trimesh
* roma
* PIL/Pillow
* numpy

## Credits

* [MASt3R](https://github.com/naver/mast3r) by Naver Corporation
* [DUSt3R](https://github.com/naver/dust3r) by Naver Corporation
* [CroCo](https://github.com/naver/croco) by Naver Corporation

## License

This project includes code from MASt3R, DUSt3R, and CroCo which are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use only).

## Citation

If you use this in your research, please cite the original papers:

```bibtex
@inproceedings{mast3r_arxiv24,
    author = {Vincent Leroy and Yohann Cabon and Jerome Revaud},
    title = {Grounding Image Matching in 3D with MASt3R},
    booktitle = {arXiv preprint},
    year = {2024}
}

@inproceedings{dust3r_cvpr24,
    author = {Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
    title = {DUSt3R: Geometric 3D Vision Made Easy},
    booktitle = {CVPR},
    year = {2024}
}

```

```

```
