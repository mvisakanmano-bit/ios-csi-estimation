# 📶 IOS CSI Estimation

Deep learning-based **Channel State Information (CSI) estimation** for **Intelligent Omni-Surface (IOS)**-assisted massive MIMO systems. Implements a compact CNN trained on simulated Rayleigh + LOS channels, evaluated across pilot counts and SNR ranges.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch) ![CUDA](https://img.shields.io/badge/CUDA-supported-green?logo=nvidia)

---

## 📐 System Model

```
User Equipment (K=4)
        │
        ▼
  IOS (N=64 elements)  ◄── phase-shift matrix Φ
        │
        ▼
  Base Station (M=32 antennas)
        │
        ▼
  Received pilots Y = H_eff · X + noise
        │
        ▼
  CNN estimator ──► Ĥ_eff
```

- **M** = 32 BS antennas, **N** = 64 IOS elements, **K** = 4 users
- Pilot counts **P** ∈ {4, 8, 16, 32}
- SNR range: **−5 dB to 30 dB**
- Channel model: **Rayleigh + LOS** (Rician, κ = 1)

---

## ✨ Features

- 🧠 **Compact CNN** — lightweight architecture for on-device deployment
- 📊 **Multi-pilot evaluation** — P = 4, 8, 16, 32 pilots
- ⚡ **Mixed-precision training** — AMP (automatic mixed precision) on CUDA
- 📉 **NMSE metric** — normalised mean-square error vs. SNR plots
- 💾 **Best-model checkpointing** — saves best `.pth` per pilot count
- 🖼️ **Result plots** — auto-generated `ios_compact_results.png`

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib
```

GPU is recommended but not required — the script auto-detects CUDA.

### Training

```bash
python ios_v10_compact.py
```

This will:
1. Generate synthetic IOS channel datasets
2. Train a CNN for each pilot count (P = 4, 8, 16, 32)
3. Evaluate NMSE across SNR range −5 to 30 dB
4. Save `best_P{N}.pth` checkpoints and `ios_compact_results.png`

### Custom Settings

Edit the system config block at the top of `ios_v10_compact.py`:

```python
# ── System Config ──
M, N, K = 32, 64, 4       # BS antennas, IOS elements, users
P_LIST = [4, 8, 16, 32]   # pilot counts to evaluate
SNR_RANGE = (-5, 30)       # SNR range in dB
```

---

## 🗂️ Project Structure

```
ios-csi-estimation/
├── ios_v10_compact.py        # Main training & evaluation script (V10)
├── ios_compact_results.png   # NMSE vs SNR plot (generated)
├── best_P4.pth               # Best model checkpoint for P=4
├── best_P8.pth               # Best model checkpoint for P=8
├── best_P16.pth              # Best model checkpoint for P=16
├── best_P32.pth              # Best model checkpoint for P=32
├── demo/
│   └── quick_eval.py         # Evaluate a saved model without re-training
└── .gitignore
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch 2.x** — model definition, training, AMP
- **NumPy** — channel simulation & pilot generation
- **Matplotlib** — NMSE vs. SNR result plots

---

## 📊 Results

After training, open `ios_compact_results.png` to see NMSE (dB) vs. SNR curves for all pilot counts.

| Pilots (P) | NMSE @ 20 dB (approx) |
|---|---|
| P = 4 | ~−8 dB |
| P = 8 | ~−12 dB |
| P = 16 | ~−16 dB |
| P = 32 | ~−20 dB |

> Results vary depending on random seed and training duration.

---

## 🧪 Quick Evaluation Demo

Run inference on saved checkpoints without retraining:

```bash
python demo/quick_eval.py --pilot 16 --snr 20
```

---

## 📄 License

MIT
