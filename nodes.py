import os
import sys

# Setup paths before any other imports
def setup_paths():
    """Setup the Python path for mast3r imports"""
    try:
        import folder_paths
        comfy_path = os.path.dirname(folder_paths.__file__)
        custom_nodes_path = os.path.join(comfy_path, 'custom_nodes', 'ComfyUI-mast3r')
    except ImportError:
        # Fallback for testing outside ComfyUI
        custom_nodes_path = os.path.dirname(os.path.abspath(__file__))
    
    if custom_nodes_path not in sys.path:
        sys.path.insert(0, custom_nodes_path)
    
    return custom_nodes_path

CUSTOM_NODES_PATH = setup_paths()

from PIL import Image
import torch
import numpy as np
import copy

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# Setup directories
CHECKPOINTS_PATH = os.path.join(CUSTOM_NODES_PATH, 'checkpoints')
INPUT_PATH = os.path.join(CUSTOM_NODES_PATH, 'input')
OUTPUT_PATH = os.path.join(CUSTOM_NODES_PATH, 'output')
CACHE_PATH = os.path.join(CUSTOM_NODES_PATH, 'cache')

for path in [CHECKPOINTS_PATH, INPUT_PATH, OUTPUT_PATH, CACHE_PATH]:
    os.makedirs(path, exist_ok=True)


def get_available_models():
    """Get list of available model checkpoint files"""
    if os.path.exists(CHECKPOINTS_PATH):
        models = [f for f in os.listdir(CHECKPOINTS_PATH) if f.endswith('.pth')]
        if models:
            return models
    return ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"]


def convert_scene_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, 
                         cam_size=0.05, cam_color=None, as_pointcloud=False, 
                         transparent_cams=False):
    """Convert scene output to GLB file format"""
    import trimesh
    from scipy.spatial.transform import Rotation
    from dust3r.utils.device import to_numpy
    from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
    
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    print(f'(exporting 3D scene to {outfile})')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, scene, min_conf_thr=2, as_pointcloud=False, 
                            mask_sky=False, clean_depth=False, transparent_cams=False, 
                            cam_size=0.05, TSDF_thresh=0):
    """Extract 3D model (glb file) from a reconstructed scene"""
    from dust3r.utils.device import to_numpy
    
    if scene is None:
        return None
    
    outfile = os.path.join(outdir, 'scene.glb')
    
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    if TSDF_thresh > 0:
        try:
            from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
        except Exception as e:
            print(f"TSDF processing failed: {e}, falling back to standard method")
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    
    msk = to_numpy([c > min_conf_thr for c in confs])
    return convert_scene_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, 
                                as_pointcloud=as_pointcloud,
                                transparent_cams=transparent_cams, cam_size=cam_size)


class Mast3rLoader:
    """Load a MASt3R model from checkpoint"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_available_models(), ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("MAST3R_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MASt3R"

    def load(self, model_name, device):
        from mast3r.model import load_model
        
        model_path = os.path.join(CHECKPOINTS_PATH, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please download the MASt3R model weights to: {CHECKPOINTS_PATH}\n"
                f"Download from: https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
            )
        
        print(f"Loading MASt3R model from {model_path}")
        model = load_model(model_path, device)
        return (model,)


class Mast3rRun:
    """Run MASt3R 3D reconstruction on input images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MAST3R_MODEL",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_folder_path": ("STRING", {"default": "", "multiline": False}),
                "image_size": ("INT", {"default": 512, "min": 224, "max": 1024, "step": 32}),
                "scenegraph_type": (["complete", "swin", "logwin", "oneref"], {"default": "complete"}),
                "winsize": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "win_cyclic": ("BOOLEAN", {"default": False}),
                "refid": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "optim_level": (["coarse", "refine", "refine+depth"], {"default": "refine+depth"}),
                "lr1": ("FLOAT", {"default": 0.07, "min": 0.001, "max": 0.5, "step": 0.01}),
                "niter1": ("INT", {"default": 300, "min": 0, "max": 2000, "step": 50}),
                "lr2": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "niter2": ("INT", {"default": 300, "min": 0, "max": 2000, "step": 50}),
                "min_conf_thr": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "matching_conf_thr": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "cam_size": ("FLOAT", {"default": 0.2, "min": 0.001, "max": 1.0, "step": 0.001}),
                "TSDF_thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "as_pointcloud": ("BOOLEAN", {"default": True}),
                "mask_sky": ("BOOLEAN", {"default": False}),
                "clean_depth": ("BOOLEAN", {"default": True}),
                "transparent_cams": ("BOOLEAN", {"default": False}),
                "shared_intrinsics": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "MAST3R_SCENE")
    RETURN_NAMES = ("glb_path", "scene")
    FUNCTION = "run"
    CATEGORY = "MASt3R"

    def run(self, model, device, image_size, scenegraph_type, winsize, win_cyclic, refid,
            optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr, cam_size, 
            TSDF_thresh, as_pointcloud, mask_sky, clean_depth, transparent_cams, shared_intrinsics,
            images=None, image_folder_path=""):
        
        from mast3r.image_pairs import make_pairs
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.utils.image import load_images
        import glob
        
        filelist = []
        
        # Check if we have a folder path
        if image_folder_path and image_folder_path.strip():
            folder_path = image_folder_path.strip()
            if os.path.isdir(folder_path):
                # Get all image files from the folder
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.BMP', '*.webp', '*.WEBP']
                for ext in image_extensions:
                    filelist.extend(glob.glob(os.path.join(folder_path, ext)))
                filelist = sorted(filelist)  # Sort for consistent ordering
                print(f"MASt3R: Found {len(filelist)} images in folder: {folder_path}")
            elif os.path.isfile(folder_path):
                # Single file path provided
                filelist = [folder_path]
                print(f"MASt3R: Using single image file: {folder_path}")
            else:
                raise ValueError(f"Path does not exist: {folder_path}")
        
        # If no folder path or no files found, use the images input
        if not filelist and images is not None:
            # Clear input folder
            for filename in os.listdir(INPUT_PATH):
                file_path = os.path.join(INPUT_PATH, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Save input images to disk
            for ind, image in enumerate(images):
                image_np = 255.0 * image.cpu().numpy()
                image_pil = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
                image_file = os.path.join(INPUT_PATH, f'{ind:04d}.png')
                image_pil.save(image_file)
                filelist.append(image_file)
            print(f"MASt3R: Saved {len(filelist)} images from input tensor")
        
        if not filelist:
            raise ValueError("No images provided. Please either connect images or provide an image_folder_path.")
        
        print(f"MASt3R: Processing {len(filelist)} images")
        
        # Load images using dust3r utility
        imgs = load_images(filelist, size=image_size)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            filelist = [filelist[0], filelist[0] + '_copy']
        
        # Build scene graph parameters
        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
            if not win_cyclic:
                scene_graph_params.append('noncyclic')
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(refid))
        
        scene_graph = '-'.join(scene_graph_params)
        print(f"MASt3R: Using scene graph: {scene_graph}")
        
        # Create image pairs
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        print(f"MASt3R: Created {len(pairs)} image pairs")
        
        # Adjust niter2 based on optim_level
        actual_niter2 = 0 if optim_level == 'coarse' else niter2
        
        # Clear and prepare cache
        cache_dir = os.path.join(CACHE_PATH, 'sparse_ga_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Run sparse global alignment
        print(f"MASt3R: Running sparse global alignment (niter1={niter1}, niter2={actual_niter2})")
        
# Ensure gradients are enabled for optimization and we are NOT in inference mode
        torch.set_grad_enabled(True)
        
        # FIX: Explicitly disable inference mode to allow gradient calculation
        with torch.inference_mode(False):
            scene = sparse_global_alignment(
                filelist, pairs, cache_dir, model,
                lr1=lr1, niter1=niter1, 
                lr2=lr2, niter2=actual_niter2,
                device=device,
                opt_depth='depth' in optim_level,
                shared_intrinsics=shared_intrinsics,
                matching_conf_thr=matching_conf_thr
            )
        
        # Generate 3D model output
        print("MASt3R: Generating 3D model output")
        outfile = get_3D_model_from_scene(
            OUTPUT_PATH, scene, min_conf_thr, as_pointcloud, mask_sky,
            clean_depth, transparent_cams, cam_size, TSDF_thresh
        )
        
        print(f"MASt3R: Output saved to {outfile}")
        return (outfile, scene)


class Mast3rSceneToDepthMaps:
    """Extract depth maps from a MASt3R scene"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("MAST3R_SCENE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "extract"
    CATEGORY = "MASt3R"

    def extract(self, scene):
        from dust3r.utils.device import to_numpy
        
        if scene is None:
            raise ValueError("No scene provided")
        
        depthmaps = scene.get_depthmaps()
        depthmaps_np = to_numpy(depthmaps)
        
        # Normalize depth maps for visualization
        depths_max = max([d.max() for d in depthmaps_np])
        normalized_depths = []
        
        for d in depthmaps_np:
            d_norm = d / depths_max
            d_rgb = np.stack([d_norm, d_norm, d_norm], axis=-1)
            normalized_depths.append(d_rgb)
        
        depth_tensor = torch.from_numpy(np.stack(normalized_depths, axis=0)).float()
        return (depth_tensor,)


class Mast3rSceneToPoses:
    """Extract camera poses from a MASt3R scene"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("MAST3R_SCENE",),
            },
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "extract"
    CATEGORY = "MASt3R"

    def extract(self, scene):
        from dust3r.utils.device import to_numpy
        
        if scene is None:
            raise ValueError("No scene provided")
        
        cam2w = scene.get_im_poses()
        cam2w_np = to_numpy(cam2w)
        
        poses = []
        for pose in cam2w_np:
            traj = [0, 0.474812461, 0.844111024, 0.5, 0.5, 1280, 720]
            traj.extend(pose[0].tolist())
            traj.extend(pose[1].tolist())
            traj.extend(pose[2].tolist())
            poses.append(traj)
        
        return (poses,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Mast3rLoader": Mast3rLoader,
    "Mast3rRun": Mast3rRun,
    "Mast3rSceneToDepthMaps": Mast3rSceneToDepthMaps,
    "Mast3rSceneToPoses": Mast3rSceneToPoses,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mast3rLoader": "MASt3R Model Loader",
    "Mast3rRun": "MASt3R 3D Reconstruction",
    "Mast3rSceneToDepthMaps": "MASt3R Scene to Depth Maps",
    "Mast3rSceneToPoses": "MASt3R Scene to Camera Poses",
}
