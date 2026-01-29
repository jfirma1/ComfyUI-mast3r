# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import - Modified for ComfyUI bundled structure
# --------------------------------------------------------

import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
COMFYUI_MAST3R_PATH = path.normpath(path.join(HERE_PATH, '../..'))
CROCO_REPO_PATH = path.join(COMFYUI_MAST3R_PATH, 'croco')
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, 'models')

# check the presence of models directory in repo to be sure its cloned
if path.isdir(CROCO_MODELS_PATH):
    # workaround for sibling import
    if CROCO_REPO_PATH not in sys.path:
        sys.path.insert(0, CROCO_REPO_PATH)
else:
    raise ImportError(f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
                      "Please ensure croco is bundled with ComfyUI-mast3r.")
