# 🔥 Wildfire Prediction & Simulation — PyTorchFire Implementation

> Built during a Machine Learning Internship at **Interlinked** (Summer 2025) · Berkeley, CA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethngo7/pytorchfire/blob/main/calibration.ipynb)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA%20T4-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project implements and extends a **GPU-accelerated wildfire simulator** based on Differentiable Cellular Automata (DCA), grounded in the research paper:

> Xia, Z. & Cheng, S. (2025). *PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata.* arXiv:2502.18738

The core idea: rather than training a supervised ML model that overfits to a specific region, we use a **physics-informed simulation** that calibrates its own parameters via gradient descent against real observed fire data — making it generalizable across ecoregions.

**Key outcomes:**
- Achieved **60% improvement in prediction accuracy** after parameter calibration vs. uncalibrated baseline
- Processed **100,000+ geospatial data points** across wind, slope, vegetation, and ignition tensors
- Delivered the calibrated model as the core of Interlinked's wildfire prediction software

---

## What's in this repo

pytorchfire/
├── notebooks/
│   ├── calibration.ipynb          # Parameter calibration pipeline (main demo)
│   └── wildfire_ROS_models.ipynb  # Rate of Spread (ROS) physics model exploration
│
├── wildfire_ROS_models/           # ROS model package
├── PyFireStation/                 # Geospatial data ingestion tools
├── scripts/
│   ├── train_nn.py                # Neural network training pipeline (PyTorch)
│   └── sobol_sensitivity_analysis.py  # Sobol sensitivity analysis on model params
│
├── tests/
├── docs/
├── requirements.txt
└── setup.py

---

## How the pipeline works

### 1. Data ingestion
Real-world geospatial data is loaded as NumPy tensors across 6 input channels:

| Tensor | Description |
|---|---|
| `wind_velocity` | Average wind speed per cell (m/s) |
| `wind_towards_direction` | Wind direction per cell (degrees from East) |
| `slope` | Elevation slope to neighboring cells (3×3 per cell) |
| `p_veg` | Vegetation type probability scaling factor |
| `p_den` | Vegetation density probability scaling factor |
| `initial_ignition` | Boolean tensor of initial fire starting state |

### 2. Simulation (forward pass)
The **Wildfire Cellular Automata** model propagates fire stochastically across cells. Each cell's ignition probability is a function of wind, slope, and fuel — computed in parallel on GPU.

### 3. Parameter calibration (backward pass)
Using gradient descent on the differentiable CA model, we calibrate 5 physics parameters (`a`, `p_h`, `p_continue`, `c_1`, `c_2`) against observed fire behavior. This is where the accuracy improvement comes from.

### 4. Analysis & visualization
Post-calibration, we generate:
- Side-by-side animations: calibrated simulation vs. ground truth
- Classification metrics: Accuracy, Precision, Recall, F1, IoU, Temporal Error
- **Directional spread analysis**: how fire distributes N/S/E/W over time
- Time-series of burn area progression by quadrant

### 5. Rate of Spread (ROS) models
Separate physics-based ROS models (in `wildfire_ROS_models/`) implement named fire spread formulations including **Rothermel (1972)**, **RothermelAndrews (2018)**, **Balbi (2020)**, and **Cruz** — complementing the CA simulation with interpretable speed estimates.

### 6. Sensitivity analysis
`sobol_sensitivity_analysis.py` runs Sobol indices to identify which input parameters most strongly drive prediction variance — useful for prioritizing data collection and model tuning.

---

## Results

| Metric | Before Calibration | After Calibration |
|---|---|---|
| Accuracy | baseline | **+60% improvement** |
| F1 Score | low | significantly higher |
| IoU | low | significantly higher |
| Temporal Error | high | reduced |

*Full metric outputs are in `notebooks/calibration.ipynb`.*

---

## Quickstart

**Run in Google Colab (recommended — requires GPU)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethngo7/pytorchfire/blob/main/calibration.ipynb)

**Run locally**

```bash
git clone https://github.com/ethngo7/pytorchfire.git
cd pytorchfire
pip install -r requirements.txt
jupyter notebook notebooks/calibration.ipynb
```

> **Note:** A CUDA-capable GPU is strongly recommended. The notebook will fall back to CPU but significantly slower.

---

## Tech stack

- **PyTorch 2.6** — model training, GPU tensor ops, gradient descent
- **pytorchfire** (Xia & Cheng, 2025) — differentiable CA simulator
- **NumPy / Matplotlib** — data processing and visualization
- **scikit-learn** — evaluation metrics
- **SALib** — Sobol sensitivity analysis
- **Google Colab (T4 GPU)** — training environment

---

## Data

This project uses the [PyTorchFire dataset](https://doi.org/10.17632/nx2wsksp9k.1) (Mendeley Data), which includes real environmental data for California fire events including the **Bear Fire (2020)** and **Pier Fire (2017)**.

Raw data files are not included in this repo. To run the notebook, either:
1. Use the automatic dataset download in `calibration.ipynb` (Cell 3), or
2. Upload your own `.npy`/`.npz` tensors following the format described in the notebook

---

## About

Built at **Interlinked** (Berkeley, CA) as part of a summer ML internship focused on applying deep learning to wildfire prediction and firefighting strategy. The calibrated model became the core of the company's wildfire prediction software.

Based on research by Zeyu Xia and Sibo Cheng — full paper: [arXiv:2502.18738](https://arxiv.org/abs/2502.18738)
