# ComfyUI-MASt3R

ComfyUI custom nodes for [MASt3R (Matching And Stereo 3D Reconstruction)](https://github.com/naver/mast3r) - a state-of-the-art 3D reconstruction model from Naver Corporation.

MASt3R extends DUSt3R with dense local feature matching capabilities, providing improved 3D reconstruction quality especially for challenging scenes.

![MASt3R](https://github.com/naver/mast3r/raw/main/assets/mast3r.jpg)

## Features

- **Mast3rLoader** - Load MASt3R model checkpoints
- **Mast3rRun** - Run 3D reconstruction on input images, outputs GLB files
- **Mast3rSceneToDepthMaps** - Extract depth maps from reconstructed scenes  
- **Mast3rSceneToPoses** - Extract camera poses for video/animation tools

## Installation

### Option 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-mast3r.git
cd ComfyUI-mast3r
pip install -r requirements.txt
```

### Option 2: Download ZIP

1. Download the repository as ZIP
2. Extract to `ComfyUI/custom_nodes/ComfyUI-mast3r`
3. Install dependencies: `pip install -r requirements.txt`

### Download Model Weights

Download the MASt3R pretrained weights and place them in the `checkpoints` folder:

```bash
cd ComfyUI/custom_nodes/ComfyUI-mast3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

## Usage

### Basic Workflow

1. **Load Model**: Add `MASt3R Model Loader` node
   - Select the checkpoint file
   - Choose device (cuda/cpu)

2. **Run Reconstruction**: Add `MASt3R 3D Reconstruction` node
   - Connect the model output
   - Either connect images OR provide a folder path containing images
   - Configure reconstruction parameters
   - Output: GLB file path and scene object

3. **Optional**: Use `MASt3R Scene to Depth Maps` or `MASt3R Scene to Camera Poses` to extract additional data

### Input Options

The `Mast3rRun` node accepts images in two ways:
- **images**: Connect images from other ComfyUI nodes (Load Image, etc.)
- **image_folder_path**: Provide a path to a folder containing images (supports PNG, JPG, BMP, WEBP)

## Recommended Settings

### By GPU VRAM

| VRAM | image_size | niter1 | niter2 | Notes |
|------|------------|--------|--------|-------|
| 8-12GB | 512 | 200 | 200 | Keep image count low (<8) |
| 16-24GB | 512-768 | 300 | 300 | Good for most uses |
| 32GB+ | 768-1024 | 500 | 500 | High quality |
| 48GB+ (A40, A6000) | 1024 | 500-800 | 500-800 | Maximum quality |

### Quick Presets

**Fast Preview:**
| Parameter | Value |
|-----------|-------|
| image_size | 512 |
| niter1 | 200 |
| niter2 | 200 |
| optim_level | coarse |

**Balanced Quality:**
| Parameter | Value |
|-----------|-------|
| image_size | 512 |
| niter1 | 300 |
| niter2 | 300 |
| optim_level | refine+depth |
| matching_conf_thr | 5.0 |

**High Quality (24GB+ VRAM):**
| Parameter | Value |
|-----------|-------|
| image_size | 768 |
| niter1 | 500 |
| niter2 | 500 |
| optim_level | refine+depth |
| matching_conf_thr | 5.0 |
| min_conf_thr | 2.0 |

**Maximum Quality (48GB+ VRAM):**
| Parameter | Value |
|-----------|-------|
| image_size | 1024 |
| niter1 | 800 |
| niter2 | 800 |
| optim_level | refine+depth |
| matching_conf_thr | 5.0 |
| min_conf_thr | 2.0 |

### Parameter Guide

| Parameter | Description | Recommendations |
|-----------|-------------|-----------------|
| image_size | Input image resolution | Higher = better quality but more VRAM |
| scenegraph_type | How images are paired | `complete` for <15 images, `swin` for more |
| winsize | Window size for swin/logwin | 5-7 for longer sequences |
| optim_level | Optimization depth | `refine+depth` for best quality |
| lr1 / niter1 | Coarse alignment | Higher niter1 = better initial alignment |
| lr2 / niter2 | Fine refinement | Higher niter2 = more detailed reconstruction |
| min_conf_thr | Point confidence filter | Lower (1.0) = more points, Higher (3.0) = cleaner |
| matching_conf_thr | Match filtering | 5.0 helps filter bad matches |
| as_pointcloud | Output format | true = pointcloud, false = mesh |
| clean_depth | Depth cleanup | Keep enabled |
| mask_sky | Remove sky points | Enable for outdoor scenes |
| TSDF_thresh | Mesh smoothing | 0.01-0.02 for smoother mesh output |

### Scene Graph Types

- **complete**: All possible image pairs - best quality but O(n²) complexity. Use for <15 images.
- **swin**: Sliding window - good for video sequences, scales linearly
- **logwin**: Logarithmic window - good for long sequences with some long-range connections
- **oneref**: Match all images to one reference - fastest but lower quality

## Output

The main output is a `.glb` file containing:
- 3D point cloud or mesh
- Camera frustums showing estimated poses
- Texture/color information

View in any GLB-compatible viewer (Blender, online 3D viewers, etc.)

## Tips

1. **Image Count**: With `complete` graph, processing time grows quadratically. For >15 images, switch to `swin`
2. **Outdoor Scenes**: Enable `mask_sky` to remove sky points that can distort the reconstruction
3. **Cleaner Output**: Increase `min_conf_thr` to 2.0-3.0 to filter low-confidence points
4. **Smoother Mesh**: Set `TSDF_thresh` to 0.01-0.02 and disable `as_pointcloud`
5. **Memory Issues**: Reduce `image_size` or switch to `swin`/`logwin` scene graph

## Requirements

- PyTorch
- scipy
- trimesh
- roma
- PIL/Pillow
- numpy

## Credits

- [MASt3R](https://github.com/naver/mast3r) by Naver Corporation
- [DUSt3R](https://github.com/naver/dust3r) by Naver Corporation
- [CroCo](https://github.com/naver/croco) by Naver Corporation

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
