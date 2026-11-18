"""Run the raw and post-processed SGBM baselines on the three rectified pairs."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from stereo_io import ROOT, SCENES, map_path, read_pair, save_map


def build_matchers():
    block_size = 9
    left = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=256, blockSize=block_size,
        P1=8 * block_size**2, P2=32 * block_size**2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return left, cv2.ximgproc.createRightMatcher(left)


def consistent_pixels(left, right, min_right, tolerance=1.0):
    """Check signed pixel disparities at corresponding image coordinates."""
    rows, cols = left.shape
    x_right = np.rint(np.arange(cols)[None, :] - left).astype(np.int32)
    inside = (x_right >= 0) & (x_right < cols)
    sampled = right[np.arange(rows)[:, None], np.clip(x_right, 0, cols - 1)]
    # OpenCV stores invalid disparities at minDisparity - 1 in each view.
    valid = inside & (left >= 0) & (sampled >= min_right) & (sampled <= 0)
    return valid & (np.abs(left + sampled) <= tolerance)


def estimate_pair(left_rgb, right_rgb, left_matcher, right_matcher):
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    left = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    right = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
    keep = consistent_pixels(left, right, right_matcher.getMinDisparity())
    if not np.any(keep):
        raise ValueError("No consistent matches are available for inpainting.")
    filtered = cv2.medianBlur(left, 5)
    # A median can be invalid beside a large hole. The original accepted value
    # is a usable seed there. Only accepted pixels serve as inpainting seeds.
    filtered = np.where(filtered >= 0, filtered, left).astype(np.float32)
    filled = cv2.inpaint(filtered, (~keep).astype(np.uint8), 3, cv2.INPAINT_NS)
    return left, filled, keep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    left_matcher, right_matcher = build_matchers()
    for scene in SCENES:
        left, right = read_pair(args.data_dir / scene)
        raw, processed, keep = estimate_pair(left, right, left_matcher, right_matcher)
        # Unmatched raw estimates (-1) become 0 in the exported 8-bit file.
        save_map(map_path(args.results_dir / "sgbm_raw", scene), raw)
        save_map(map_path(args.results_dir / "sgbm_processed", scene), processed)
        print(f"{scene}: {keep.mean():.2%} passed the left-right check.", flush=True)


if __name__ == "__main__":
    main()
