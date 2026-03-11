# 🫀 ACuRE: Accurate Continuity-Regularized SpO₂ Estimation Using Liquid Time-Constant Networks

> ## 🏆 **Accepted as Oral at WACV 2026**
> ## 📄 **Paper:** [ACuRE: Accurate Continuity-Regularized SpO₂ Estimation Using Liquid Time-Constant Networks](https://openaccess.thecvf.com/content/WACV2026/papers/Ahmad_ACuRE_Accurate_Continuity-Regularized_SpO2_Estimation_Using_Liquid_Time-Constant_Networks_WACV_2026_paper.pdf)

This repository contains the official implementation of our **WACV 2026 Oral** paper.


<!-- > **ACuRE: Accurate Continuity-Regularized SpO₂ Estimation Using Liquid Time-Constant Networks** -->

<!-- <p align="center">
  <img src="docs/figure1_framework.png" width="90%">
</p> -->

---

## 📄 Paper Overview

The **ACuRE** framework introduces a **physics-informed and physiology-inspired** pipeline for accurate and robust **SpO₂ estimation from facial videos**.
It combines **AC/DC component separation**, **3D spatiotemporal feature extraction**, and **Liquid Time-Constant (LTC)**-based temporal modeling, guided by **continuity regularization** via a **physics-informed PDE loss**.

### 🧩 Key Highlights

* **AC/DC Decomposition:** Physiologically interpretable signal separation from facial video.
* **Physics-Informed Regularization:** Enforces **continuity constraints** (∂ρ/∂t + ∇·(ρv) = 0) for temporal consistency.
* **LTC Dynamics:** Recurrent ODE-based layer for adaptive temporal modeling.
* **Robust Estimation:** Outperforms state-of-the-art methods on **PURE**, **BH-rPPG**, and **VIPL-HR** datasets.

---

## 🧪 Framework Overview

<p align="center">
  <img src="docs/figure1_framework.png" width="90%">
</p>

<!-- > **Figure:** Overview of ACuRE framework for SpO₂ estimation.
> The pipeline includes AC/DC feature extraction via 3D convolutions,
> physics-informed regularization, and temporal dynamics modeled with LTC. -->

---

## 📊 Experimental Results

<p align="center">
  <img src="docs/figure_results_comparison.png" width="90%">
</p>

<!-- > **Figure:** Comparison of ground truth (orange) and predicted (blue) SpO₂ values across datasets. -->

---

## 🧠 Visualization of Learned Heat Flow

<p align="center">
  <img src="docs/figure_heatmaps.png" width="90%">
</p>

> **Figure:** Visualization of spatiotemporal heat propagation learned across frames.

---

## 🧹 Data Pre‑processing

Our pipeline crops faces with Mediapipe landmarks, stacks RGB+HSV+HLS+Lab channels, rescales frames to **32×32**, resamples SpO₂/PPG to video length, and drops clips with noisy GT (<90 or >100). Each script saves compressed `.npz` files with `video`, `wave`, and `fps` fields used by `dataset.py`.

### PURE → `Dataset/Pure_data_video`
- Prepare paired video `.mp4` and JSON (`/FullPackage` oximeter) files.
- Run (edit paths as needed):
```bash
python - <<'PY'
from data_preprocesing.PURE import preprocess_and_save_spatio_temporal_maps
preprocess_and_save_spatio_temporal_maps(
    video_dir="/path/to/PURE_videos",        # *.mp4
    json_dir="/path/to/PURE_json",           # matching *.json
    output_dir="Dataset/Pure_data_video",    # destination inside repo
    image_dir="Dataset/PURE_preview_png"     # optional previews
)
PY
```

### BH-rPPG → `Dataset/Bh_rPPG_dataset`
- Point `--input` to the raw BH-rPPG release root (each session has `sensor.csv` + `*.avi`).
- Run:
```bash
python data_preprocesing/BH_rPPG.py \
  --input /path/to/Pub_BH-rPPG_FULL_compack \
  --output Dataset/Bh_rPPG_dataset
```

### VIPLR → `Dataset/VIPLR_data_video`
- Provide the VIPL-Raw tree (skip infrared `source4` is handled automatically).
- Run:
```bash
python data_preprocesing/VIPLR.py \
  --input /path/to/VIPLR_zip_data \
  --output Dataset/VIPLR_data_video
```

> Tip: If your data lives elsewhere, adjust the `map_dir` variables at the bottom of each training/testing script to point to your `.npz` folders.

---

## 🚀 Train & Test
All scripts perform **5-fold subject-wise CV** with LTC temporal blocks and save weights/plots under `results/<DATASET>_res/`.

### PURE
- **Train:**
```bash
python PURE/PURE_training.py
```
  - expects `Dataset/Pure_data_video` (edit `map_dir` if different)
  - outputs to `results/PURE_res/{PURE_weight, PURE_Plots_eval, PURE_checkpoints}`
- **Test (uses saved fold weights):**
```bash
python PURE/PURE_test.py
```
  - set `map_dir` and `base_model_save_path` in the script to your locations before running.

### BH-rPPG
- **Train:**
```bash
python BHRPPG/BHRPPG_training.py
```
  - uses `Dataset/Bh_rPPG_dataset`
  - saves to `results/BHRPPG_res/{BHRPPG_weight, BHRPPG_Plots_eval, BHRPPG_checkpoints}`
- **Test:**
```bash
python BHRPPG/BHRPPG_test.py
```
  - expects the same dataset path and the weights above.

### VIPLR (VIPL-HR)
- **Train:**
```bash
python VIPLR/VIPLR_training.py
```
  - uses `Dataset/VIPLR_data_video`
  - saves to `results/VIPLR_res/{VIPLR_weight, VIPLR_Plots_eval, VIPLR_checkpoints}`
- **Test:**
```bash
python VIPLR/VIPLR_testing.py
```
  - loads weights from `results/VIPLR_res/VIPLR_weight` by default.

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Ahmad_2026_WACV,
    author    = {Ahmad, Shahzad and Mishra, Divya and Bano, Sania and Chanda, Sukalpa and Rawat, Yogesh Singh},
    title     = {ACuRE: Accurate Continuity-Regularized SpO2 Estimation Using Liquid Time-Constant Networks},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7250-7259}
}

```

---

## 🏛️ Acknowledgements

This work was conducted at **Østfold University** in collaboration with **UCF**.
We thank the authors of PURE, BH-rPPG, and VIPL-HR datasets for making their data publicly available.
