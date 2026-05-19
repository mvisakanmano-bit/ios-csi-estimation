"""
demo/quick_eval.py

Quickly evaluate a saved IOS CSI checkpoint without re-training.
Loads best_P{pilot}.pth and reports NMSE at a given SNR.

Usage:
    python demo/quick_eval.py                        # defaults: P=16, SNR=20
    python demo/quick_eval.py --pilot 8 --snr 10
    python demo/quick_eval.py --pilot 32 --snr 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── Add parent directory so we can import the model ───────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Re-declare model architecture (must match ios_v10_compact.py) ─

M, N, K = 32, 64, 4  # BS antennas, IOS elements, users


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class CompactEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32), ResBlock(32),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.head = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        return self.head(self.enc(x))


# ── Signal helpers (minimal, matching main script) ────────────────

def build_pilots(P):
    Peff = min(P, M)
    rows = np.round(np.linspace(0, M - 1, Peff)).astype(int)
    base = (
        np.exp(1j * 2 * np.pi * rows[:, None] * np.arange(M)[None, :] / M) / np.sqrt(M)
    ).astype(np.complex64)
    if P <= M:
        return base, base.conj().T
    pil = np.tile(base, (int(np.ceil(P / Peff)), 1))[:P].astype(np.complex64)
    return (
        pil @ np.linalg.inv((pil.conj().T @ pil).astype(np.complex128)).astype(np.complex64),
        pil.conj().T,
    )


def add_noise(Y, snr_db):
    s = np.sqrt(np.mean(np.abs(Y) ** 2) / (2 * 10 ** (snr_db / 10)))
    return (Y + s * (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)).astype(np.complex64)).astype(np.complex64)


def gen_channels(bs, L=2, kappa=1.0):
    def steer(n, a):
        return (np.exp(1j * np.pi * np.arange(n, dtype=np.float32)[None, :] * np.sin(a).astype(np.float32)[:, None]) / np.sqrt(n)).astype(np.complex64)
    los_s = np.sqrt(kappa / (kappa + 1))
    nlos_s = np.sqrt(1 / (kappa + 1))
    phi = np.exp(1j * np.random.uniform(0, 2 * np.pi, N)).astype(np.complex64)
    z = np.zeros(bs, dtype=np.float32)
    H_BI = los_s * steer(N, z)[:, :, None] * steer(M, z)[:, None, :].conj()
    for _ in range(L):
        a = nlos_s * (np.random.randn(bs) + 1j * np.random.randn(bs)).astype(np.complex64) / np.sqrt(2 * L)
        H_BI += a[:, None, None] * steer(N, np.random.uniform(-1, 1, bs))[:, :, None] * steer(M, np.random.uniform(-1, 1, bs))[:, None, :].conj()
    H_IU = (np.random.randn(bs, K, N) + 1j * np.random.randn(bs, K, N)).astype(np.complex64) / np.sqrt(2)
    return (H_IU @ (H_BI * phi[:, None])).astype(np.complex64)  # (bs, K, M)


def to_3ch(H_ls):
    H_ls = H_ls[None] if H_ls.ndim == 2 else H_ls
    scale = np.sqrt(np.mean(np.abs(H_ls) ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    Hn = H_ls / scale
    ri = np.stack([Hn.real, Hn.imag], axis=1).astype(np.float32)
    log_s = np.broadcast_to((np.log(scale) / 3).reshape(-1, 1, 1, 1), (H_ls.shape[0], 1, K, M)).copy().astype(np.float32)
    return np.concatenate([ri, log_s], axis=1), scale


# ── Main evaluation ───────────────────────────────────────────────

def evaluate(pilot: int, snr_db: float, num_samples: int = 500) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(ROOT, f"best_P{pilot}.pth")

    if not os.path.exists(ckpt_path):
        print(f"\n  ✗  Checkpoint not found: {ckpt_path}")
        print("     Run ios_v10_compact.py first to train and save the model.\n")
        sys.exit(1)

    model = CompactEstimator().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ls_mat, _ = build_pilots(pilot)
    nmse_list = []

    with torch.no_grad():
        H_true = gen_channels(num_samples)  # (N, K, M) complex
        X = np.ones((M, pilot), dtype=np.complex64) / np.sqrt(M)
        Y = np.einsum("bkm,mp->bkp", H_true, X)
        Y_noisy = add_noise(Y, snr_db)
        H_ls = (Y_noisy @ ls_mat).astype(np.complex64)

        inp, scale = to_3ch(H_ls)
        inp_t = torch.from_numpy(inp).to(device)
        out = model(inp_t).cpu().numpy()  # (N, 2, K, M)

        scale = scale[:, 0, 0, 0]
        H_hat = (out[:, 0] + 1j * out[:, 1]).astype(np.complex64) * scale[:, None, None]

        err = np.mean(np.abs(H_hat - H_true) ** 2, axis=(1, 2))
        ref = np.mean(np.abs(H_true) ** 2, axis=(1, 2))
        nmse_list = (err / (ref + 1e-10)).tolist()

    return float(np.mean(nmse_list))


def main():
    parser = argparse.ArgumentParser(description="Quick eval of a saved IOS CSI checkpoint")
    parser.add_argument("--pilot",   type=int,   default=16, choices=[4, 8, 16, 32], help="Pilot count")
    parser.add_argument("--snr",     type=float, default=20, help="SNR in dB (default: 20)")
    parser.add_argument("--samples", type=int,   default=500, help="Test samples (default: 500)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📶  IOS CSI Estimation — Quick Eval")
    print(f"    Checkpoint : best_P{args.pilot}.pth")
    print(f"    SNR        : {args.snr} dB")
    print(f"    Samples    : {args.samples}")
    print(f"    Device     : {device}")
    print()

    nmse = evaluate(args.pilot, args.snr, args.samples)
    nmse_db = 10 * np.log10(nmse + 1e-10)

    print(f"  ✅  NMSE = {nmse:.6f}  ({nmse_db:.2f} dB)  @ P={args.pilot}, SNR={args.snr} dB\n")


if __name__ == "__main__":
    main()
