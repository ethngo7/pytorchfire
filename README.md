# 🔥 Wildfire Prediction & Simulation — PyTorchFire

> Machine Learning Internship · **Interlinked** · Summer 2025 · Berkeley, CA

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA%20T4-green)

---

## Start here

**`notebooks/calibration.ipynb`** — the full pipeline: load real California fire data → simulate wildfire spread on GPU → calibrate physics parameters via gradient descent → compare before/after results with directional spread analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethngo7/wildfire-prediction-pytorch/blob/main/notebooks/calibration.ipynb)

---

## What this project does

Implements and extends a GPU-accelerated wildfire simulator based on **Differentiable Cellular Automata (DCA)**, grounded in:

> Xia & Cheng (2025). *PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata.* [arXiv:2502.18738](https://arxiv.org/abs/2502.18738)

Instead of a supervised ML model that overfits to one region, this uses a **physics-informed simulation** that self-corrects its parameters against real observed fire data — making it generalizable across ecoregions.

**Results:**
- **60% improvement in prediction accuracy** after parameter calibration vs. uncalibrated baseline
- **100,000+ geospatial data points** processed across wind, slope, vegetation, and ignition tensors
- Delivered as the core of Interlinked's wildfire prediction software

---

## Pipeline

| Step | What happens |
|---|---|
| Data ingestion | Load wind, slope, vegetation, and ignition tensors from real CA fire events |
| Forward pass | Stochastic cellular automata propagates fire across cells on GPU |
| Backward pass | Gradient descent calibrates 5 physics parameters against observed fire data |
| Evaluation | Accuracy, Precision, Recall, F1, IoU, Temporal Error — before and after calibration |
| Visualization | Side-by-side animations, directional spread analysis (N/S/E/W over time) |
| ROS models | Rothermel (1972), RothermelAndrews (2018), Balbi (2020), Cruz — physics-based rate of spread |
| Sensitivity | Sobol indices identify which parameters most drive prediction variance |

---

## Repo structure

```
wildfire-prediction-pytorch/
├── notebooks/
│   ├── calibration.ipynb          # Main demo — full calibration pipeline
│   └── wildfire_ROS_models.ipynb  # Rate of Spread model exploration
├── wildfire_ROS_models/           # ROS model implementations
├── scripts/                       # Training and analysis scripts
├── sobol_sensitivity_analysis.py  # Sobol sensitivity analysis
├── train_nn.py                    # Neural network training pipeline
├── requirements.txt
└── setup.py
```

---

## Tech stack

PyTorch 2.6 · pytorchfire · NumPy · Matplotlib · scikit-learn · SALib · Google Colab (T4 GPU)

---

## Data

Uses the [PyTorchFire dataset](https://doi.org/10.17632/nx2wsksp9k.1) (Mendeley Data) — real environmental data for the **Bear Fire (2020)** and **Pier Fire (2017)** in California. Raw data is not included; the notebook downloads it automatically (Cell 3).
