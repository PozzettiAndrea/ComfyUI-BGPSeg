# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI-BGPSeg - Boundary-Guided Primitive Segmentation

Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra

Paper: "BGPSeg: Boundary-Guided Primitive Segmentation for Point Clouds" (IEEE TIP 2025)
"""

import sys

if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
