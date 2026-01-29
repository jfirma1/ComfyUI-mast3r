# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r submodule import - Modified for ComfyUI bundled structure
# --------------------------------------------------------

import sys
import os.path as path

# For ComfyUI-mast3r, dust3r is bundled in the same package
HERE_PATH = path.normpath(path.dirname(__file__))
COMFYUI_MAST3R_PATH = path.normpath(path.join(HERE_PATH, '../..'))

# Add the ComfyUI-mast3r path so dust3r can be imported directly
if COMFYUI_MAST3R_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_MAST3R_PATH)
