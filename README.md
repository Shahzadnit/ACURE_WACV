# 🫀 ACuRE: Accurate Continuity-Regularized SpO₂ Estimation Using Liquid Time-Constant Networks

### 🧠 Accepted at **WACV 2026**

This repository contains the official implementation of our **WACV 2026** paper:

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

## Training and Evaluation for PURE
```
python PURE/main_fold_PURE_bb_abl_physnet.py

python PURE/evaluation_folds_bb.py

```


## Training and Evaluation for Bh-rPPG
```
python BHRPPG/main_fold_BHRPPG_bb_abl_physent.py

python BHRPPG/evaluation_folds_bb.py

```

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

