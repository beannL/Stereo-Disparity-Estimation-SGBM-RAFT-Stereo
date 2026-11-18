"""Recreate the report figures and print the Art measurements used in the text."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from stereo_io import ROOT, SCENES, map_path, read_map


# Zero-based image coordinates. Right and bottom limits are exclusive.
CROP = (0, 440, 420, 1000)
SAMPLES = (("o", 50, 740), ("^", 300, 710))


def show_map(axis, disparity):
    axis.imshow(disparity, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.axis("off")


def overview(data_dir, maps_dir, output):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.68))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.934, wspace=0.035, hspace=0.035)
    for col, scene in enumerate(SCENES):
        top, bottom = axes[:, col]
        with Image.open(data_dir / scene / "view1.png") as image:
            top.imshow(image)
        top.axis("off")
        top.set_title(scene, fontsize=10, pad=4)
        show_map(bottom, read_map(map_path(maps_dir, scene)))
    x0, y0, x1, y1 = CROP
    axes[0, 0].add_patch(Rectangle((x0 + 1, y0), x1 - x0 - 1, y1 - y0,
                                   fill=False, edgecolor="#ffe372", linewidth=1.4))
    fig.savefig(output / "scene_overview.png", dpi=230)
    plt.close(fig)


def baseline_comparison(results_dir, reference, output):
    raw = read_map(map_path(results_dir / "sgbm_raw", "Art"))
    processed = read_map(map_path(results_dir / "sgbm_processed", "Art"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.04))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.004, top=0.89, wspace=0.035)
    for axis, array, title in zip(axes, (raw, processed, reference),
                                  ("Raw SGBM", "Processed SGBM", "Ground truth")):
        show_map(axis, array)
        axis.set_title(title, fontsize=10, pad=4)
    fig.savefig(output / "art_sgbm.png", dpi=230)
    plt.close(fig)

    labeled_band = reference[:, :256] != 0
    for name, prediction in (("Raw", raw), ("Processed", processed)):
        err = np.abs(prediction[:, :256].astype(float) - reference[:, :256])
        print(f"{name} SGBM, Art leftmost 256 columns: {err[labeled_band].mean():.2f} px MAE")


def art_detail(data_dir, maps_dir, reference, output):
    x0, y0, x1, y1 = CROP
    with Image.open(data_dir / "Art" / "view1.png") as image:
        rgb = np.array(image.crop(CROP))
    gt = reference[y0:y1, x0:x1]
    pred = read_map(map_path(maps_dir, "Art"))[y0:y1, x0:x1]
    error = np.abs(pred.astype(float) - gt)
    valid = gt != 0

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.91))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.908, bottom=0.145, wspace=0.055)
    axes[0].imshow(rgb)
    axes[0].axis("off")
    # Mark the two measured locations without covering their centers.
    for marker, x, y in SAMPLES:
        for color, width in (("black", 2.3), ("white", 1.15)):
            axes[0].plot(x - x0, y - y0, marker, markerfacecolor="none",
                         markeredgecolor=color, markeredgewidth=width, markersize=7.5)
    show_map(axes[1], gt)
    show_map(axes[2], pred)
    cmap = plt.get_cmap("inferno").with_extremes(bad="#9dc6d1")
    colored = axes[3].imshow(np.ma.masked_where(~valid, error), cmap=cmap,
                             vmin=0, vmax=20, interpolation="nearest")
    axes[3].axis("off")
    for axis, title in zip(axes, ("Left image", "Ground truth", "RAFT-Stereo", "Absolute error (px)")):
        axis.set_title(title, fontsize=9.5, pad=4)
    pos = axes[3].get_position()
    bar_axis = fig.add_axes((pos.x0, 0.095, pos.width, 0.025))
    bar = fig.colorbar(colored, cax=bar_axis, orientation="horizontal", ticks=(0, 10, 20))
    bar.ax.set_xticklabels(("0", "10", "20+"))
    bar.ax.get_xticklabels()[-1].set_ha("right")
    bar.ax.tick_params(labelsize=8, length=2.5, pad=1.5)
    fig.savefig(output / "art_detail.png", dpi=250)
    plt.close(fig)

    outside = np.arange(x0, x1)[None, :] < gt
    for label, mask in (("Outside", valid & outside), ("Inside", valid & ~outside)):
        print(f"{label} right-image bounds, Art crop: {mask.sum()} pixels, {error[mask].mean():.2f} px MAE")
    for _, x, y in SAMPLES:
        print(f"Sample ({x}, {y}): ground truth {gt[y-y0, x-x0]}, prediction {pred[y-y0, x-x0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--maps-dir", type=Path, default=ROOT / "disparity_maps")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    reference = read_map(args.data_dir / "Art" / "disp1.png")
    overview(args.data_dir, args.maps_dir, output)
    baseline_comparison(args.results_dir, reference, output)
    art_detail(args.data_dir, args.maps_dir, reference, output)


if __name__ == "__main__":
    main()
