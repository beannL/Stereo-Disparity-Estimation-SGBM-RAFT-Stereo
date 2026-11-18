"""Shared file conventions for the full-size Middlebury 2005 scenes."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("Art", "Dolls", "Reindeer")


def read_pair(scene_dir):
    images = []
    for name in ("view1.png", "view5.png"):
        with Image.open(scene_dir / name) as image:
            images.append(np.array(image.convert("RGB")))
    left, right = images
    if left.shape != right.shape:
        raise ValueError(f"Image sizes differ in {scene_dir}: {left.shape} and {right.shape}")
    return left, right


def read_map(path):
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected an 8-bit grayscale disparity map at {path}")
        return np.array(image)


def map_path(directory, scene):
    return directory / f"{scene}_disparity.png"


def save_map(path, disparity):
    if disparity.ndim != 2 or not np.isfinite(disparity).all():
        raise ValueError(f"Expected a finite 2-D disparity estimate for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(disparity, 0, 255)).astype(np.uint8)
    Image.fromarray(pixels).save(path)
