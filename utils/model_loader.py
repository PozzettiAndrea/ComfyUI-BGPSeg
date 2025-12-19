# SPDX-License-Identifier: GPL-3.0-or-later
# Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
Model loader utilities for BGPSeg
Downloads and caches pretrained models from Google Drive
"""

from pathlib import Path
from typing import Optional, Dict

# BGPSeg model URLs and metadata
BGPSEG_MODELS = {
    "bgpseg_abc": {
        "folder_id": "1qev6yadvatGxGm-9HQBNiNIIpDNe9D1A",
        "files": {
            "BGPSeg_model.pth": "BGFE main model (10 primitive types)",
            "Boundary_model.pth": "BoundaryNet boundary predictor",
        },
        "description": "BGPSeg pretrained on ABC Primitive dataset (boundary-guided segmentation)",
    },
}


def get_bgpseg_models_dir() -> Path:
    """
    Get the models directory for BGPSeg models.
    Creates ComfyUI/models/cadrecon/bgpseg/ if it doesn't exist.
    """
    current_dir = Path(__file__).parent.parent  # ComfyUI-BGPSeg/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "bgpseg"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_bgpseg_model_path(filename: str) -> Optional[Path]:
    """Get the path to a specific BGPSeg model file."""
    models_dir = get_bgpseg_models_dir()
    model_path = models_dir / filename
    if model_path.exists():
        return model_path
    return None


def download_bgpseg_models(force_download: bool = False) -> Optional[Path]:
    """
    Download BGPSeg models from Google Drive if not already cached.

    Downloads both:
    - BGPSeg_model.pth (BGFE main model)
    - Boundary_model.pth (BoundaryNet predictor)

    Note: Google Drive downloads require gdown for large files.
    Install with: pip install gdown
    """
    model_info = BGPSEG_MODELS["bgpseg_abc"]
    models_dir = get_bgpseg_models_dir()

    bgpseg_path = models_dir / "BGPSeg_model.pth"
    boundary_path = models_dir / "Boundary_model.pth"

    if bgpseg_path.exists() and boundary_path.exists() and not force_download:
        print(f"[OK] BGPSeg models already downloaded: {models_dir}")
        return models_dir

    print(f"[BGPSeg] Models not found locally, downloading...")
    print(f"   Description: {model_info['description']}")

    try:
        import gdown
        folder_id = model_info["folder_id"]
        print(f"[BGPSeg] Downloading from Google Drive folder...")

        gdown.download_folder(id=folder_id, output=str(models_dir), quiet=False)

        if bgpseg_path.exists() and boundary_path.exists():
            print(f"[OK] BGPSeg models downloaded: {models_dir}")
            return models_dir
        else:
            print(f"[WARN] Some model files missing after download")

    except ImportError:
        print("[WARN] gdown not installed. Install with: pip install gdown")

    print(f"\n[WARN] Automatic download failed. Please download manually:")
    print(f"   1. Visit: https://drive.google.com/drive/folders/{model_info['folder_id']}")
    print(f"   2. Download both .pth files:")
    for filename, desc in model_info["files"].items():
        print(f"      - {filename} ({desc})")
    print(f"   3. Save to: {models_dir}")
    return None


def list_bgpseg_models() -> Dict[str, bool]:
    """
    List BGPSeg model files and their download status.

    Returns:
        Dictionary mapping filenames to whether they are downloaded
    """
    models_status = {}
    for filename in BGPSEG_MODELS["bgpseg_abc"]["files"].keys():
        model_path = get_bgpseg_model_path(filename)
        models_status[filename] = model_path is not None
    return models_status
