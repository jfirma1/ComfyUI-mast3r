# ComfyUI-MASt3R

ComfyUI custom nodes for [MASt3R (Matching And Stereo 3D Reconstruction)](https://github.com/naver/mast3r) — a 3D reconstruction model from Naver.

This fork adds practical workflow improvements for ComfyUI use, including:

- **MASt3R Model Loader** for the ViT-Large checkpoint
- **MASt3R 3D Reconstruction** that outputs a `.glb`
- **Scene to Depth Maps** export
- **Scene to Camera Poses** export
- **Retrieval scenegraph support** for larger image sets
- **Fresh per-run cache directories** to reduce stale-cache write failures

![MASt3R](https://github.com/naver/mast3r/raw/main/assets/mast3r.jpg)

---

## Features

- **MASt3R Model Loader**: Loads the checkpoint with practical memory handling for ComfyUI workflows.
- **MASt3R 3D Reconstruction**: Runs sparse global alignment and exports a `.glb` scene.
- **MASt3R Scene to Depth Maps**: Extracts depth maps from the reconstructed scene for downstream workflows.
- **MASt3R Scene to Camera Poses**: Extracts camera trajectories for animation, previs, and layout work.
- **Retrieval Mode**: Adds `scenegraph_type = retrieval`, with support for retrieval weights and codebook files.
- **Safer Cache Behavior**: Uses a fresh sparse-GA cache directory per run instead of reusing one persistent cache folder.

---

## Installation

### 1. Install the Nodes

**Option A: Git Clone**

Navigate to your `ComfyUI/custom_nodes` folder and run:

```bash
git clone https://github.com/jfirma1/ComfyUI-mast3r.git
cd ComfyUI-mast3r
pip install -r requirements.txt
```

**Option B: Download ZIP**

1. Download this repository as a ZIP file.
2. Extract it into `ComfyUI/custom_nodes/ComfyUI-mast3r`.
3. Open a terminal in that folder and run:

```bash
pip install -r requirements.txt
```

### 2. Download the Base MASt3R Checkpoint

Place the checkpoint below in `ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints/`.

```bash
mkdir -p ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints
cd ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints
wget -O MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

### 3. Optional: Enable Retrieval Mode

Retrieval mode requires two additional files in the same `checkpoints/` folder:

- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`

Download them with:

```bash
mkdir -p /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints && \
wget -O /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
```

```bash
mkdir -p /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints && \
wget -O /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

### 4. Optional: Retrieval Dependencies (`faiss-cpu` + ASMK)

Retrieval mode also needs `faiss-cpu` and `asmk` installed.

```bash
python -m pip install -U pip setuptools wheel cython faiss-cpu
cd /tmp
rm -rf /tmp/asmk
git clone --depth 1 https://github.com/jenicek/asmk /tmp/asmk
cd /tmp/asmk/cython
cythonize *.pyx
cd /tmp/asmk
python -m pip install --no-build-isolation .
cd /
python - <<'PY'
import faiss
import asmk
from asmk import asmk_method
print("faiss ok:", faiss.__file__)
print("asmk ok:", asmk.__file__)
print("asmk_method ok:", asmk_method.__file__)
PY
```

> Important: run the final verification from outside `/tmp/asmk`, or Python may import the source tree instead of the installed package.

---

## Usage

### Basic Workflow

1. Add **MASt3R Model Loader**.
2. Add **MASt3R 3D Reconstruction**.
3. Connect the loaded model.
4. Provide source images either by:
   - connecting an image batch to the `images` input, or
   - entering a folder in `image_folder_path`
5. The node writes a `.glb` and returns the scene object for depth-map / camera-pose downstream nodes.

### Input Methods

- **`images` input**: Best for smaller batches or images already inside ComfyUI.
- **`image_folder_path`**: Best for larger datasets or extracted video frames.

---

## Scenegraph Modes

The scenegraph controls how image pairs are built.

### `complete`
Matches every image with every other image.

- Best for: small unordered image sets, object scans
- Strongest connectivity
- Most expensive in memory and compute

### `swin`
Sliding window matching.

- Best for: sequential video frames, walkthroughs, turntables
- More stable for long sequences
- Much cheaper than `complete`

### `logwin`
Windowed matching with some longer-range links.

- Best for: long indoor sequences where `swin` is too local but `complete` is too heavy
- Often a good default for room scans

### `oneref`
Matches all images to one reference image.

- Best for: star-pattern captures or explicit hero/reference-frame setups

### `retrieval`
Builds pairs from retrieval-selected key images and neighbors.

- Best for: larger image sets where `complete` is too expensive
- Requires the retrieval model, codebook, `faiss-cpu`, and `asmk`
- In this node:
  - `winsize` = number of key images
  - `refid` = number of neighbors

---

## Recommended Settings

## Quick Practical Rules

- Start with **`image_size = 512` or `768`**.
- Use **`complete`** only for smaller curated image sets.
- Use **`logwin`**, **`swin`**, or **`retrieval`** for longer sequences.
- For one camera / one lens / one phone clip, set **`shared_intrinsics = True`**.
- While tuning, keep **`as_pointcloud = True`**. Mesh export is less forgiving.

### Preset A — Indoor Room Walkthrough (40–80 frames)

```text
image_size = 768
scenegraph_type = retrieval
winsize = 12 to 20
refid = 6 to 10
optim_level = refine+depth
lr1 = 0.05
niter1 = 500
lr2 = 0.005
niter2 = 500
matching_conf_thr = 1.0 to 2.0
min_conf_thr = 1.0 to 1.5
shared_intrinsics = True
as_pointcloud = True
TSDF_thresh = 0.0
clean_depth = True
```

### Preset B — Long Video Sequence Without Retrieval

```text
image_size = 768
scenegraph_type = logwin
winsize = 4 to 6
optim_level = refine+depth
lr1 = 0.05
niter1 = 500
lr2 = 0.005
niter2 = 500
matching_conf_thr = 1.0 to 2.0
min_conf_thr = 1.2 to 1.8
shared_intrinsics = True
as_pointcloud = True
TSDF_thresh = 0.0
clean_depth = True
```

### Preset C — Small High-Quality Curated Set (12–24 images)

```text
image_size = 768 or 1024
scenegraph_type = complete
optim_level = refine+depth
lr1 = 0.04 to 0.05
niter1 = 500 to 600
lr2 = 0.004 to 0.005
niter2 = 500 to 600
matching_conf_thr = 1.0 to 2.0
min_conf_thr = 1.2 to 1.8
shared_intrinsics = True if all images come from the same camera/lens
as_pointcloud = True during tuning
```

### VRAM / Scaling Guidance

| GPU VRAM | Safer Starting Point |
| --- | --- |
| 12–16 GB | 224–512 px, avoid `complete` on bigger sets |
| 24 GB | 512 px stable, 768 px for smaller sets |
| 48 GB | 512–768 px practical for many workflows |
| 80 GB+ | More room for 1024 px experiments and denser graphs |

> Warning: `complete` scales quadratically with image count. Do not use `1024 + complete` on large image sets unless you know your memory budget can handle it.

---

## Parameter Guide

| Parameter | What it really does |
| --- | --- |
| **image_size** | Primary quality / memory tradeoff. Start at `512` or `768`. |
| **scenegraph_type** | Controls pair construction strategy: `complete`, `swin`, `logwin`, `oneref`, or `retrieval`. |
| **winsize** | For `swin` / `logwin`: window size. For `retrieval`: number of key images. |
| **refid** | For `oneref`: reference image index. For `retrieval`: number of neighbors. |
| **optim_level** | `coarse`, `refine`, or `refine+depth`. `refine+depth` is the best quality preset. |
| **niter1** | Stage 1 / alignment iterations. Increase if camera layout is unstable. |
| **niter2** | Stage 2 / refinement iterations. Increase for slower, tighter convergence. |
| **lr1** | Stage 1 learning rate. Lower it slightly if camera solves feel unstable. |
| **lr2** | Stage 2 learning rate. Lower values can help refine more gently. |
| **matching_conf_thr** | Match strictness during alignment. Raise it if you get spurious floaters or bad long-range connections. |
| **min_conf_thr** | Export cleanup threshold. Higher values make the GLB cleaner but sparser. Lower values recover more detail but also more noise. |
| **TSDF_thresh** | Optional TSDF cleanup during export. Keep at `0.0` while solving; try small values later only if the solve is already good. |
| **as_pointcloud** | Recommended `True` while tuning. Point clouds reveal the real solve more honestly than mesh export. |
| **clean_depth** | Helpful cleanup for exported dense depth. Usually leave enabled. |
| **mask_sky** | Use for outdoor captures to avoid huge sky shells. |
| **shared_intrinsics** | Set `True` for one camera / one lens / one zoom setting. |
| **cam_size** | Visual size of the camera frustums in the GLB preview. |
| **retrieval_model_name** | Retrieval checkpoint selector. Leave on Auto unless you need to force a specific file. |
| **retrieval_model_path** | Optional manual override path to the `*_retrieval_trainingfree.pth` file. |

---

## Practical Notes About the GLB Output

The `.glb` preview is useful, but it is not the same thing as a final production mesh pipeline.

- A valid `.glb` can still contain poor geometry if the solve is weak.
- A "melted" room usually means the reconstruction or export thresholds are off, not that the file container is corrupted.
- A very sparse result usually means export thresholds are too aggressive.

Good rule of thumb:

1. First get a believable **point cloud**.
2. Then try mesh export.
3. Only then experiment with mild `TSDF_thresh` values.

---

## Troubleshooting

### `PytorchStreamWriter failed writing file data/0`

This is usually a cache / disk write failure, not a MASt3R math failure.

Try:

```bash
rm -rf /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/cache/*
rm -f /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/input/extracted_frame_*.jpg
```

Then check free space:

```bash
df -h /workspace
du -sh /workspace/ComfyUI/custom_nodes/ComfyUI-mast3r/cache
```

### Retrieval mode import errors

If retrieval fails with `faiss` / `asmk` import errors, reinstall the retrieval dependencies and verify imports from outside `/tmp/asmk`.

### GLB looks melted or folded

- keep `as_pointcloud = True`
- raise `matching_conf_thr` a bit
- keep `TSDF_thresh = 0.0`
- avoid overly aggressive graph settings on long sequences

### GLB is too sparse

- lower `min_conf_thr`
- keep `TSDF_thresh = 0.0`
- try `image_size = 768`
- use stronger graph connectivity (`logwin` or `retrieval`) instead of making export pruning harsher

### Too many images, not enough detail

More frames do not automatically mean more detail. Prefer fewer, better-spaced, more diverse frames over many nearly identical adjacent frames.

---

## Optional RunPod Startup Script Addition (Retrieval Dependencies)

If you use a startup script to restore dependencies on Pod launch, add a retrieval section:

```bash
# MAST3R RETRIEVAL DEPENDENCIES (faiss + ASMK)
python -m pip install --break-system-packages --no-cache-dir -q \
    pip setuptools wheel cython faiss-cpu

cd /tmp
rm -rf /tmp/asmk
git clone --depth 1 https://github.com/jenicek/asmk /tmp/asmk
cd /tmp/asmk/cython
cythonize *.pyx
cd /tmp/asmk
python -m pip install --break-system-packages --no-build-isolation --no-cache-dir .

cd /
python - <<'PY'
import faiss
import asmk
from asmk import asmk_method
print("faiss ok:", faiss.__file__)
print("asmk ok:", asmk.__file__)
print("asmk_method ok:", asmk_method.__file__)
PY
```

---

## Requirements

- PyTorch
- scipy
- trimesh
- roma
- Pillow
- numpy
- `faiss-cpu` and `asmk` for retrieval mode

---

## Credits

- [MASt3R](https://github.com/naver/mast3r) by Naver Corporation
- [DUSt3R](https://github.com/naver/dust3r) by Naver Corporation
- [CroCo](https://github.com/naver/croco) by Naver Corporation

---

## License

This project includes code from MASt3R, DUSt3R, and CroCo which are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use only).

---

## Citation

If you use this in research, please cite the original papers:

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
