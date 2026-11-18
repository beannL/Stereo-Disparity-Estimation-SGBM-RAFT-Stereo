"""Estimate left-view disparity from rectified image pairs using RAFT-Stereo."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from stereo_io import ROOT, SCENES, map_path, read_pair, save_map

# The upstream modules import one another as core.*. Use their bundled root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "raft_stereo"))
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


def load_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}. See the checkpoint link in README.md.")
    config = SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=False,
    )
    model = RAFTStereo(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval()


@torch.inference_mode()
def estimate(model, left, right, device):
    def tensor(image):
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)

    left_tensor = tensor(left)
    padder = InputPadder(left_tensor.shape, divis_by=32)
    left_pad, right_pad = padder.pad(left_tensor, tensor(right))
    _, flow = model(left_pad, right_pad, iters=32, test_mode=True)
    # Flow is x_right - x_left. The submitted disparity uses x_left - x_right.
    disparity = -padder.unpad(flow)[0, 0]
    if tuple(disparity.shape) != left.shape[:2]:
        raise ValueError("Unpadded prediction and left image have different sizes.")
    return disparity.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "disparity_maps")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "raftstereo-middlebury.pth")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    for scene in SCENES:
        print(f"Estimating {scene} on {args.device} ...", flush=True)
        left, right = read_pair(args.data_dir / scene)
        disparity = estimate(model, left, right, args.device)
        path = map_path(args.output_dir, scene)
        save_map(path, disparity)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
