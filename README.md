# Stereo disparity estimation

The three PNGs in `disparity_maps/` are the final left-view estimates. The three-page report is `final_report.pdf`, with an editable copy in `final_report.docx`. `results/psnr_results.csv` contains scores for those PNGs and both SGBM baselines. Run the commands below from the directory containing this README with Python 3.11 or 3.12. The reported inference run used Python 3.12 and a CPU.

## Verify the submitted scores

The following commands download the course images, verify their checksums, and score the nine saved maps. This check requires NumPy and Pillow.

```bash
python -m pip install numpy Pillow
python source_code/download_assets.py --data-only
python source_code/evaluate.py
```

To score just the three final maps, run `python source_code/evaluate.py --final-only`. This writes a separate `results/psnr_final_only.csv`. If the course data are already on disk, use `--data-dir PATH` with the evaluation and image-processing scripts. The path must contain `Art/`, `Dolls/`, and `Reindeer/`, each with `view1.png`, `view5.png`, and `disp1.png`.

## Reproduce the maps and figures

For a Windows or Linux CPU environment, install the CPU build of PyTorch first. Then install the remaining packages and run the experiment.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r source_code/requirements.txt
python source_code/download_assets.py
python source_code/run_raft_inference.py
python source_code/stereo_disparity.py
python source_code/evaluate.py
python source_code/make_figures.py
```

The download script fetches the course PNGs into `data/` and the official Middlebury checkpoint into `models/raftstereo-middlebury.pth`. It verifies each file against its SHA-256 digest. For manual download, use the [course folder](https://drive.google.com/drive/folders/1pdDkFpa59m4A02pLkUqI1H2Zb_GerLkK) and the [authors’ checkpoint](https://drive.google.com/file/d/1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd/view).

## Outputs and evaluation

| Location | Contents |
| --- | --- |
| `disparity_maps/` | Final RAFT-Stereo maps for Art, Dolls, and Reindeer. One gray level represents one pixel of left-view disparity. |
| `results/sgbm_raw/` | Left SGBM estimates before the additional consistency check, median filter and hole filling. Negative unmatched values are encoded as zero. |
| `results/sgbm_processed/` | Corresponding SGBM maps after those three operations. |
| `results/psnr_results.csv` | Per-scene and mean full-image and labeled-pixel PSNR, in dB. |
| `results/figures/` | Three report figures, including the Art crop and its absolute error map. The figure script also prints the local measurements reported in the text. |

Both inference scripts read only the stereo photographs. The evaluation script reopens the exported grayscale PNGs, checks that their shapes match the references, and computes both scores with a fixed peak of 255. The full-image score uses every pixel. The labeled-pixel score uses locations where the ground-truth value is greater than zero. The `Mean` rows average the three scene scores before rounding. `unknown_sse_pct` is the percentage of full-image squared error at locations with unknown ground truth. It is reported per scene.

## Implementation credits

The inference wrapper calls the authors’ RAFT-Stereo model using 32 updates and the official Middlebury weights. The upstream inference modules and their MIT license are included in `source_code/vendor/raft_stereo/`. The resource revision is listed in `UPSTREAM.txt`. The SGBM experiment uses the OpenCV contrib build for its right-view matcher.

- [Middlebury 2006 stereo scenes and disparity encoding](https://vision.middlebury.edu/stereo/data/scenes2006/)
- [RAFT-Stereo implementation](https://github.com/princeton-vl/RAFT-Stereo)
- [OpenCV StereoSGBM documentation](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
