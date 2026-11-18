"""Calculate PSNR from saved 8-bit disparity PNGs and the course ground truth."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from stereo_io import ROOT, SCENES, map_path, read_map


def psnr(mse):
    return math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def score(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction {prediction.shape} differs from reference {reference.shape}.")
    labeled = reference != 0
    if not labeled.any():
        raise ValueError("No labeled reference pixels.")
    squared_error = (prediction.astype(np.float64) - reference) ** 2
    full_sse = squared_error.sum()
    unknown_sse = squared_error[~labeled].sum()
    return {
        "pixels": reference.size,
        "labeled_pixels": int(labeled.sum()),
        "full_psnr_db": psnr(squared_error.mean()),
        "labeled_psnr_db": psnr(squared_error[labeled].mean()),
        "unknown_sse_pct": 100.0 * unknown_sse / full_sse if full_sse else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--maps-dir", type=Path, default=ROOT / "disparity_maps")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--final-only", action="store_true", help="Score only the submitted RAFT-Stereo maps.")
    args = parser.parse_args()
    methods = {"RAFT-Stereo": args.maps_dir}
    if not args.final_only:
        methods["SGBM raw"] = args.results_dir / "sgbm_raw"
        methods["SGBM processed"] = args.results_dir / "sgbm_processed"

    rows = []
    for scene in SCENES:
        reference = read_map(args.data_dir / scene / "disp1.png")
        for name, directory in methods.items():
            prediction = read_map(map_path(directory, scene))
            metrics = score(prediction, reference)
            rows.append({"scene": scene, "method": name, **metrics})
            print(f"{scene:8}  {name:15}  full {metrics['full_psnr_db']:6.2f} dB  "
                  f"labeled {metrics['labeled_psnr_db']:6.2f} dB")

    for method in methods:
        samples = [r for r in rows if r["method"] == method and r["scene"] in SCENES]
        rows.append({
            "scene": "Mean", "method": method,
            "full_psnr_db": np.mean([r["full_psnr_db"] for r in samples]),
            "labeled_psnr_db": np.mean([r["labeled_psnr_db"] for r in samples]),
        })
    args.results_dir.mkdir(parents=True, exist_ok=True)
    filename = "psnr_final_only.csv" if args.final_only else "psnr_results.csv"
    with (args.results_dir / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, (float, np.floating)) else v
                             for k, v in row.items()})


if __name__ == "__main__":
    main()
