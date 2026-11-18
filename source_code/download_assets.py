"""Fetch the exact public course images and the official RAFT-Stereo checkpoint."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
# Google Drive file ID and SHA-256 of the original download.
ASSETS = {
    "data/Art/disp1.png": (
        "1DnnLlBzT-gM-EVhnJwsm-_CTM89g1mhu",
        "4dfa1d682518486c9a3affe567951fc8ccd03c65825e79dfa8f91ecb9c5d2e48",
    ),
    "data/Art/view1.png": (
        "1GEaAo52-bRwVYP1wc4e1_jHhEFTx4AMa",
        "4cb22b9db1ef50192fbea9944378f54a159a6fd051e7de2a03e9ca5eaece3172",
    ),
    "data/Art/view5.png": (
        "1aN4hfolHD4rH1NADDDgUuTH79OEzEGQB",
        "d8bd23c772dc9650f1bbc76af571b18433fe3e2f69d151ce21b8aa84561d526c",
    ),
    "data/Dolls/disp1.png": (
        "1VnW6YVRyJINE4G_RCebnaKj-gRmNjEH6",
        "dd3a142c3250243b6a1e500fb718cc369c7d2078f1fe2d948525866836b8b375",
    ),
    "data/Dolls/view1.png": (
        "1WCNehqaIEoLXXO_k5d_n22xTuAYR06Ai",
        "323429b700cb18ece80aae44fd58b4e6c7407fa7601fa4bb27d1473e6961d6f1",
    ),
    "data/Dolls/view5.png": (
        "1kDb8jJZR6p7_ccF6sNoXohGBKoUyBXtG",
        "afc50c23a58d0771a2dc4c1bb8fd857db921226af0afa1fc2eb88b47713a829f",
    ),
    "data/Reindeer/disp1.png": (
        "1nAjshUVJ7_ZRQ-W4DUfXf1QtSC0IZu_Q",
        "f182c9edefd3c3d7e294edb05648537350e67a6a50939599ee53b97276164f0a",
    ),
    "data/Reindeer/view1.png": (
        "1VlEcHds4RnkzNAUTaCE4o2lLCAOmZszJ",
        "c4ffc52f47e10b46e18052899c75a930ff4aba3ef4abcb6865d96338e9ae16a2",
    ),
    "data/Reindeer/view5.png": (
        "1kcBjxaibK0dd5loTxDbFEL3ebEDT-Ssk",
        "06430b26c2366b92cfa1b23e82fa697a3c128594459299479cb43b350e3cdcf6",
    ),
    "models/raftstereo-middlebury.pth": (
        "1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd",
        "d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819",
    ),
}


def fetch(destination, file_id, expected):
    if destination.is_file():
        with destination.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Existing file differs from the course asset: {destination}")
        print(f"Verified {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Download did not match the expected file: {destination.name}")
        temporary.replace(destination)
        print(f"Downloaded {destination}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="Skip the neural network checkpoint.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Parent of the data and models directories.")
    args = parser.parse_args()
    for relative, (file_id, checksum) in ASSETS.items():
        if args.data_only and relative.startswith("models/"):
            continue
        fetch(args.root / relative, file_id, checksum)


if __name__ == "__main__":
    main()
