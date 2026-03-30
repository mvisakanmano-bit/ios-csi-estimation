import os
import sys
import tempfile
import warnings
import time
import argparse
from collections import namedtuple
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --- SETTINGS ----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

USE_COMPILE = (
    hasattr(torch, "compile")
    and torch.__version__ >= "2.0"
    and sys.platform != "win32"
)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

NUM_WORKERS = 2 if sys.platform != "win32" else 0
TMPDIR = tempfile.gettempdir()


# --- SYSTEM PARAMETERS ------------------------------------------------------
class Sys:
    M = 64    # BS antennas
    N = 128   # IOS elements
    K = 4     # UE users
    P_max = 128
    P_list = [4, 8, 16, 32, 64, 128]


# --- IOS PHASE PROFILE HELPERS ----------------------------------------------
def make_phi_vec(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return np.exp(1j * rng.uniform(0.0, 2 * np.pi, n)).astype(np.complex64)


# --- PILOT CACHE ------------------------------------------------------------
PilotSet = namedtuple("PilotSet", ["ls", "ct", "ct_batched"])
_PILOTS: Dict[int, PilotSet] = {}


def _build_pilot_cache():
    for P in Sys.P_list:
        P_eff = min(P, Sys.M)
        row_idx = np.round(np.linspace(0, Sys.M - 1, P_eff)).astype(int)
        idx = row_idx[:, None] * np.arange(Sys.M)[None, :]
        pil_base = (np.exp(1j * 2 * np.pi * idx / Sys.M) / np.sqrt(Sys.M)).astype(np.complex64)

        if P <= Sys.M:
            pil, pil_ls = pil_base, pil_base
        else:
            repeats = int(np.ceil(P / P_eff))
            pil = np.tile(pil_base, (repeats, 1))[:P].astype(np.complex64)
            gram = (pil.conj().T @ pil).astype(np.complex128)
            pil_ls = (pil @ np.linalg.inv(gram).astype(np.complex64)).astype(np.complex64)

        _PILOTS[P] = PilotSet(ls=pil_ls, ct=pil.conj().T, ct_batched=pil.conj().T[None, ...])


_build_pilot_cache()


def get_pilots(P: int) -> PilotSet:
    return _PILOTS[P]


# --- SIGNAL HELPERS ---------------------------------------------------------
def add_complex_noise(Y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN at a given SNR (dB)."""
    sigma = np.sqrt(np.mean(np.abs(Y) ** 2) / (2.0 * 10 ** (snr_db / 10)))
    noise = sigma * (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)).astype(np.complex64)
    return (Y + noise).astype(np.complex64)


def ls_estimate(Y: np.ndarray, P: int) -> np.ndarray:
    """Compute LS channel estimate from received signal Y."""
    return (Y @ _PILOTS[P].ls).astype(np.complex64)


def normalize_and_format(H_ls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex H_ls to 3-channel model input + scale factor.

    Works for single sample (K, M) or batch (B, K, M).
    Returns: (input_3ch, scale)
        - Single: (3, K, M), scalar scale
        - Batch:  (B, 3, K, M), (B, 1, 1) scale
    """
    batched = H_ls.ndim == 3
    if not batched:
        H_ls = H_ls[None, ...]  # add batch dim

    B = H_ls.shape[0]
    scale = np.sqrt(np.mean(np.abs(H_ls) ** 2, axis=(1, 2), keepdims=True)) + 1e-8  # (B,1,1)
    H_n = H_ls / scale

    ri = np.stack([H_n.real, H_n.imag], axis=1).astype(np.float32)  # (B, 2, K, M)
    log_s = (np.log(scale) / 3.0).reshape(B, 1, 1, 1)
    log_ch = np.broadcast_to(log_s, (B, 1, Sys.K, Sys.M)).copy().astype(np.float32)
    out = np.concatenate([ri, log_ch], axis=1)  # (B, 3, K, M)

    if not batched:
        return out[0], float(scale[0, 0, 0])
    return out, scale


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL MODEL — V10: 25° angular spread
# ═══════════════════════════════════════════════════════════════════════════════
V10_ANGULAR_SPREAD_DEG = 25.0


def _ula_steering(n_ant: int, angles_rad: np.ndarray) -> np.ndarray:
    m = np.arange(n_ant, dtype=np.float32)[None, :]
    phi = np.sin(angles_rad).astype(np.float32)[:, None]
    return (np.exp(1j * np.pi * m * phi) / np.sqrt(n_ant)).astype(np.complex64)


def _add_nlos_clusters(H, n_rx, n_tx, bs, L, sigma, nlos_s):
    # Pre-generate all random angles and gains for L clusters at once
    theta_rx = np.random.uniform(-np.pi / 3, np.pi / 3, (L, bs)) + np.random.randn(L, bs) * sigma
    theta_tx = np.random.uniform(-np.pi / 3, np.pi / 3, (L, bs)) + np.random.randn(L, bs) * sigma
    alphas = nlos_s * (np.random.randn(L, bs) + 1j * np.random.randn(L, bs)).astype(np.complex64) / np.sqrt(2.0 * L)

    for l in range(L):
        a_rx = _ula_steering(n_rx, theta_rx[l])
        a_tx = _ula_steering(n_tx, theta_tx[l])
        H += alphas[l, :, None, None] * a_rx[:, :, None] * a_tx[:, None, :].conj()


def _generate_channel_batch(bs, L, kappa, angular_spread_deg, phi_vec):
    sigma = np.deg2rad(angular_spread_deg)
    los_s = np.float32(np.sqrt(kappa / (kappa + 1.0)))
    nlos_s = np.float32(np.sqrt(1.0 / (kappa + 1.0)))
    zeros_bs = np.zeros(bs, dtype=np.float32)

    # H_BI: (bs, N, M) — BS-to-IOS
    a_rx = _ula_steering(Sys.N, zeros_bs)
    a_tx = _ula_steering(Sys.M, zeros_bs)
    H_BI = los_s * a_rx[:, :, None] * a_tx[:, None, :].conj()
    _add_nlos_clusters(H_BI, Sys.N, Sys.M, bs, L, sigma, nlos_s)

    # H_IU: (bs, K, N) — IOS-to-UE
    los_ang = np.full(bs, np.deg2rad(5.0), dtype=np.float32)
    a_rx_iu = _ula_steering(Sys.K, los_ang)
    a_tx_iu = _ula_steering(Sys.N, los_ang)
    H_IU = los_s * a_rx_iu[:, :, None] * a_tx_iu[:, None, :].conj()
    _add_nlos_clusters(H_IU, Sys.K, Sys.N, bs, L, sigma, nlos_s)

    # Cascade: H_eff = H_IU · diag(phi) · H_BI  →  (bs, K, M)
    H_eff = np.matmul(H_IU, phi_vec[None, :, None] * H_BI)

    norms = np.linalg.norm(H_eff.reshape(bs, -1), axis=1, keepdims=True)
    H_eff /= (norms[:, :, None] + 1e-8)
    return H_eff.astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL BANK
# ═══════════════════════════════════════════════════════════════════════════════
class ChannelBank:
    def __init__(
        self,
        n_samples: int,
        L: int = 3,
        kappa: float = 1.0,
        angular_spread_deg: float = V10_ANGULAR_SPREAD_DEG,
        chunk_size: int = 500,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n = n_samples
        self.L, self.kappa = L, kappa
        self.angular_spread_deg = angular_spread_deg
        self.rng = rng or np.random.default_rng()
        self.phi_vec = make_phi_vec(Sys.N, self.rng)

        print(f"    Precomputing {n_samples} V10 channels (kappa={kappa}, L={L}, spread={angular_spread_deg}deg)...")
        t0 = time.time()
        self.H_eff = self._generate_chunked(n_samples, chunk_size)
        print(f"    Done in {time.time()-t0:.1f}s | Memory: {self.H_eff.nbytes/1e6:.0f} MB")

    def _generate_chunked(self, n, chunk):
        H = np.empty((n, Sys.K, Sys.M), dtype=np.complex64)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            H[s:e] = _generate_channel_batch(e - s, self.L, self.kappa, self.angular_spread_deg, self.phi_vec)
        return H

    def generate_eval(self, n: int, chunk: int = 500) -> np.ndarray:
        return self._generate_chunked(n, chunk)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — 3-CHANNEL INPUT: [H_ls_real | H_ls_imag | log_scale]
# ═══════════════════════════════════════════════════════════════════════════════
class PrecomputedDataset(Dataset):
    def __init__(self, bank, P, snr_range=(-5, 30), augment=True, progressive_snr=None):
        self.bank = bank
        self.P = P
        self.snr_lo, self.snr_hi = snr_range
        self.augment = augment
        self.prog_snr = progressive_snr
        self._pilots = _PILOTS[P]

    def __len__(self):
        return self.bank.n

    def __getitem__(self, idx):
        H_eff = self.bank.H_eff[idx].copy()

        # SNR selection (progressive or uniform)
        if self.prog_snr is not None and np.random.rand() < 0.5:
            snr_db = np.random.uniform(self.prog_snr, self.snr_hi)
        else:
            snr_db = np.random.uniform(self.snr_lo, self.snr_hi)

        # Received signal with noise
        Y_noisy = add_complex_noise(H_eff @ self._pilots.ct, snr_db)

        # Data augmentation
        if self.augment:
            r = np.random.rand(2)
            if r[0] < 0.3:
                phase = np.exp(1j * np.random.uniform(0, 2 * np.pi)).astype(np.complex64)
                Y_noisy *= phase
                H_eff *= phase
            if r[1] < 0.2:
                scale = np.float32(np.random.uniform(0.85, 1.15))
                Y_noisy *= scale
                H_eff *= scale

        # LS estimate → 3-channel input
        H_ls = ls_estimate(Y_noisy, self.P)
        H_ls_3ch, scale = normalize_and_format(H_ls)  # single sample path
        H_eff_n = (H_eff / scale).astype(np.complex64)

        H_tgt = np.stack([H_eff_n.real, H_eff_n.imag], axis=-1)
        return torch.from_numpy(H_ls_3ch), torch.from_numpy(H_tgt)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
        self.cbam = CBAM(ch)

    def forward(self, x):
        return x + self.cbam(self.block(x))


class MultiScaleStem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, 32, 3, padding=1, bias=False)
        self.branch2 = nn.Conv2d(in_ch, 16, 5, padding=2, bias=False)
        self.branch3 = nn.Conv2d(in_ch, 16, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.fuse = nn.Conv2d(64, out_ch, 1, bias=False)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.fuse(F.relu(self.bn(out), inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 2), stride=(1, 2), bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# ANGULAR DEALIASING PATH — depthwise-separable, K preserved
# ═══════════════════════════════════════════════════════════════════════════════
class AngularDealiasingPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, (1, 17), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 17), padding=(0, 8), groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.pw1 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 17), padding=(0, 8), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, 17), padding=(0, 8), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 17), padding=(0, 8), groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 2, 1)

    def forward(self, x_ri):
        # FFT: antenna → angular domain
        x_c = torch.view_as_complex(x_ri.permute(0, 2, 3, 1).contiguous())
        X_fft = torch.fft.fft(x_c, dim=-1)
        X_ri = torch.stack([X_fft.real, X_fft.imag], dim=1)

        # Angular CNN — K dimension preserved throughout
        f = self.pw1(self.dw1(self.layer1(X_ri)))
        f = self.pw2(self.dw2(self.layer3(f)))
        corr_fft_ri = self.head(f)

        # IFFT: angular → antenna domain
        corr_c = torch.fft.ifft(
            torch.view_as_complex(corr_fft_ri.permute(0, 2, 3, 1).contiguous()), dim=-1
        )
        return torch.stack([corr_c.real, corr_c.imag], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# V10 MAIN MODEL: DualPathIOSNet
# ═══════════════════════════════════════════════════════════════════════════════
class DualPathIOSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = MultiScaleStem(3, 64)

        # Path A: Spatial denoising (U-Net, SNR-gated)
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), ResBlock(128),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256), ResBlock(256),
        )
        self.snr_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.denoise_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
        )

        # Path B: Angular dealiasing (persistent gate)
        self.dealias_path = AngularDealiasingPath()
        self.log_dealias_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        skip = x[:, :2].permute(0, 2, 3, 1).contiguous()

        # Path A: Spatial denoising
        f0 = self.stem(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        snr_g = self.snr_gate(f2)
        d0 = self.dec1(self.dec2(f2, f1), f0)
        denoise_corr = self.denoise_head(d0).permute(0, 2, 3, 1).contiguous()

        # Path B: Angular dealiasing
        dealias_corr = self.dealias_path(x[:, :2])
        dealias_g = torch.sigmoid(self.log_dealias_gate)

        return skip + snr_g.view(-1, 1, 1, 1) * denoise_corr + dealias_g * dealias_corr


# --- EMA ---------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.data.clone() for p in self.params]
        self.backup = []

    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self):
        self.backup = [p.data.clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    def restore(self):
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.backup = []


# --- LOSS ---------------------------------------------------------------------
def nmse_loss(pred, true):
    err = torch.sum((pred - true) ** 2, dim=(1, 2, 3))
    return torch.mean(err / (torch.sum(true ** 2, dim=(1, 2, 3)) + 1e-8))


def advanced_loss(pred, true):
    loss = nmse_loss(pred, true)
    p = torch.view_as_complex(pred.contiguous())
    t = torch.view_as_complex(true.contiguous())
    num = torch.abs(torch.sum(p * t.conj(), dim=(1, 2)))
    den = torch.norm(p.reshape(p.size(0), -1), dim=1) * torch.norm(t.reshape(t.size(0), -1), dim=1) + 1e-8
    return loss + 0.1 * (1.0 - torch.mean(num / den))


# --- INFERENCE HELPER --------------------------------------------------------
def batched_infer_numpy(model, x_np, batch_size=256):
    preds = []
    for i in range(0, x_np.shape[0], batch_size):
        x = torch.from_numpy(x_np[i:i + batch_size]).to(DEVICE)
        with torch.no_grad(), autocast("cuda", enabled=USE_AMP):
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train_fast(P, train_bank, val_bank, epochs=40, patience=12):
    print("\n" + "=" * 60)
    print(f"  Training  P={P}  |  {epochs} epochs  |  device={DEVICE}  |  AMP={USE_AMP}  |  compile={USE_COMPILE}")
    print("=" * 60)

    model = DualPathIOSNet().to(DEVICE)
    print(f"  DualPathIOSNet: initial dealias_gate = {torch.sigmoid(model.log_dealias_gate).item():.3f}")

    if USE_COMPILE:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    ema = EMA(model, decay=0.9995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=USE_AMP)

    train_ds_early = PrecomputedDataset(train_bank, P, augment=True, progressive_snr=10)
    train_ds_full = PrecomputedDataset(train_bank, P, augment=True)
    val_ds = PrecomputedDataset(val_bank, P, augment=False)

    switch_epoch = max(2, epochs // 5)
    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                     persistent_workers=(NUM_WORKERS > 0))

    def make_train_ld(ds):
        return DataLoader(ds, batch_size=128, shuffle=True, **dl_kwargs)

    train_ld = make_train_ld(train_ds_early)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False, **dl_kwargs)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=len(train_ld) * epochs,
        pct_start=0.1, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4,
    )

    best_val, no_imp = float("inf"), 0
    tr_hist, vl_hist = [], []
    ckpt = os.path.join(TMPDIR, f"iosnet_v10_P{P}.pth")

    for ep in range(1, epochs + 1):
        ep_start = time.time()

        if ep == switch_epoch:
            print(f"  >> Switching to full SNR range at epoch {ep}")
            train_ld = make_train_ld(train_ds_full)

        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, y in tqdm(train_ld, desc=f"P={P}|Ep{ep:02d}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=USE_AMP):
                loss = advanced_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            sched.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches
        tr_hist.append(epoch_loss)

        # Validate with EMA weights
        ema.apply_shadow()
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad(), autocast("cuda", enabled=USE_AMP):
            for x, y in val_ld:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                val_loss += nmse_loss(model(x), y).item()
                val_n += 1
        val_loss /= val_n
        vl_hist.append(val_loss)
        ema.restore()

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        dg = torch.sigmoid(raw_model.log_dealias_gate).item()
        print(f"  P={P} | Ep {ep:02d}/{epochs} | Train {epoch_loss:.4f} | Val {val_loss:.4f} | "
              f"dealias_gate {dg:.3f} | LR {sched.get_last_lr()[0]:.2e} | {time.time()-ep_start:.1f}s")

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            ema.apply_shadow()
            torch.save((model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(), ckpt)
            ema.restore()
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  * Early stop at epoch {ep}")
                break

    (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(
        torch.load(ckpt, map_location=DEVICE, weights_only=True)
    )
    return model, tr_hist, vl_hist


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def _nmse_db(est, true):
    """Per-sample NMSE in dB. est, true: (n, K, M) complex."""
    n = est.shape[0]
    diff = (est - true).reshape(n, -1)
    ref = true.reshape(n, -1)
    return np.linalg.norm(diff, axis=1) ** 2 / (np.linalg.norm(ref, axis=1) ** 2 + 1e-8)


def evaluate_fast(models, train_bank, n_mc=500):
    snrs = np.arange(-5, 36, 5)
    results = {P: {"cnn": [], "ls": [], "cnn_samples_20": None, "ls_samples_20": None,
                    "cnn_per_user_20": None, "ls_per_user_20": None} for P in Sys.P_list}

    print(f"\n[Evaluation] Generating {n_mc} fresh V10 eval channels...")
    H_true = train_bank.generate_eval(n_mc)
    print(f"[Evaluation] Running NMSE vs SNR (n={n_mc})...")

    for P in Sys.P_list:
        pilots = get_pilots(P)
        model = models[P]
        model.eval()

        Y_clean = np.matmul(H_true, np.broadcast_to(pilots.ct_batched, (n_mc,) + pilots.ct.shape))

        for snr_db in tqdm(snrs, desc=f"P={P}"):
            Y_noisy = add_complex_noise(Y_clean, snr_db)
            H_ls = ls_estimate(Y_noisy, P)

            # Prepare model input using shared helper
            H_ls_3ch, scale = normalize_and_format(H_ls)  # (n_mc, 3, K, M), (n_mc, 1, 1)

            # CNN prediction → complex channel estimate
            cnn_preds = batched_infer_numpy(model, H_ls_3ch, batch_size=250)
            H_cnn = (cnn_preds[..., 0] + 1j * cnn_preds[..., 1]) * scale

            # NMSE
            cnn_nmse = _nmse_db(H_cnn, H_true)
            ls_nmse = _nmse_db(H_ls, H_true)
            results[P]["cnn"].append(10 * np.log10(np.mean(cnn_nmse)))
            results[P]["ls"].append(10 * np.log10(np.mean(ls_nmse)))

            if snr_db == 20:
                results[P]["cnn_samples_20"] = 10 * np.log10(cnn_nmse + 1e-12)
                results[P]["ls_samples_20"] = 10 * np.log10(ls_nmse + 1e-12)

                true_per_user = np.linalg.norm(H_true.reshape(n_mc, Sys.K, -1), axis=2) ** 2 + 1e-8
                cnn_per_user = np.linalg.norm((H_cnn - H_true).reshape(n_mc, Sys.K, -1), axis=2) ** 2
                ls_per_user = np.linalg.norm((H_ls - H_true).reshape(n_mc, Sys.K, -1), axis=2) ** 2
                results[P]["cnn_per_user_20"] = 10 * np.log10(np.mean(cnn_per_user / true_per_user, axis=0))
                results[P]["ls_per_user_20"] = 10 * np.log10(np.mean(ls_per_user / true_per_user, axis=0))

    return results, snrs


# --- PLOTTING ----------------------------------------------------------------
_COLORS = ["#C0392B", "#148F77", "#1A5276", "#B7950B", "#7D3C98", "#1A6B4A"]
_BG = "white"
_AX_BG = "#F8F9FA"
_TEXT = "#1a1a2e"
_GRID_C = "#CCCCCC"
_SPINE_C = "#AAAAAA"


def _style(ax, title, xlabel, ylabel, legend=True, legend_kw=None):
    ax.set_facecolor(_AX_BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=_TEXT)
    ax.set_xlabel(xlabel, fontsize=9, color=_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=_TEXT)
    ax.grid(True, color=_GRID_C, alpha=0.8, linewidth=0.6)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_C)
    if legend:
        kw = dict(fontsize=7, framealpha=0.9, facecolor="white", edgecolor=_SPINE_C)
        kw.update(legend_kw or {})
        ax.legend(**kw)


def plot_fast(results, snrs, histories, out="ios_results_v10.png"):
    fig = plt.figure(figsize=(20, 11), facecolor=_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35, left=0.06, right=0.97, top=0.92, bottom=0.09)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    fig.suptitle("IOS-Assisted CSI Estimation — CNN vs LS Benchmark  [V10]",
                 fontsize=14, fontweight="bold", color=_TEXT, y=0.97)

    def _overhead(P):
        return int(100 * (1 - P / Sys.P_max))

    # Panel 0: NMSE vs SNR
    ax = axes[0]
    for i, P in enumerate(Sys.P_list):
        c = _COLORS[i]
        ax.plot(snrs, results[P]["cnn"], "o-", c=c, lw=2.2, ms=6,
                label=f"CNN  P={P:2d} ({_overhead(P)}% less overhead)")
        ax.plot(snrs, results[P]["ls"], "s--", c=c, lw=1.4, ms=4, alpha=0.5, label=f"LS   P={P:2d}")
    _style(ax, "NMSE vs SNR — CNN vs LS Benchmark  [V10]", "SNR (dB)", "NMSE (dB)")

    # Panel 1: CNN gain over LS
    ax = axes[1]
    for i, P in enumerate(Sys.P_list):
        gain = [results[P]["ls"][j] - results[P]["cnn"][j] for j in range(len(snrs))]
        ax.plot(snrs, gain, "o-", c=_COLORS[i], lw=2.2, ms=6, label=f"P={P} ({_overhead(P)}% less)")
    ax.axhline(0, color=_TEXT, lw=0.8, alpha=0.4, linestyle="--")
    _style(ax, "CNN Gain over LS  [V10]", "SNR (dB)", "NMSE Gain (dB)")

    # Panel 2: Training / Validation Loss
    ax = axes[2]
    for i, P in enumerate(Sys.P_list):
        tr, vl = histories[P]
        ax.plot(tr, "-", c=_COLORS[i], lw=1.5, alpha=0.8, label=f"Train P={P}")
        ax.plot(vl, "--", c=_COLORS[i], lw=1.5, alpha=0.8, label=f"Val   P={P}")
    ax.set_yscale("log")
    _style(ax, "Training / Validation Loss  [V10]", "Epoch", "NMSE Loss")

    # Panel 3: NMSE bar chart at SNR=20 dB
    ax = axes[3]
    idx20 = int(np.argmin(np.abs(snrs - 20)))
    x_pos = np.arange(len(Sys.P_list))
    width = 0.35
    ls_vals = [results[P]["ls"][idx20] for P in Sys.P_list]
    cnn_vals = [results[P]["cnn"][idx20] for P in Sys.P_list]
    bars_ls = ax.bar(x_pos - width / 2, ls_vals, width, color="#95A5A6", label="LS")
    bars_cnn = ax.bar(x_pos + width / 2, cnn_vals, width, color="#148F77", label="CNN")
    for bars, vals in [(bars_ls, ls_vals), (bars_cnn, cnn_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.15, f"{val:.1f}",
                    ha="center", va="top", fontsize=7.5, color="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"P={P}\n({_overhead(P)}% less)" for P in Sys.P_list], fontsize=8)
    _style(ax, "NMSE at SNR = 20 dB  [V10]", "Pilot count", "NMSE (dB)")

    # Panel 4: ECDF at SNR=20 dB
    ax = axes[4]
    for i, P in enumerate(Sys.P_list):
        cs, ls_ = results[P]["cnn_samples_20"], results[P]["ls_samples_20"]
        if cs is None:
            continue
        cdf = np.linspace(0, 1, len(cs))
        ax.plot(np.sort(cs), cdf, "-", c=_COLORS[i], lw=2)
        ax.plot(np.sort(ls_), cdf, "--", c=_COLORS[i], lw=1.2, alpha=0.55)
    _style(ax, "NMSE ECDF @ SNR=20 dB  (solid=CNN, dashed=LS)  [V10]", "NMSE (dB)", "CDF", legend=False)

    # Panel 5: Per-user heatmap at SNR=20 dB
    ax = axes[5]
    matrix, row_labels = [], []
    for P in Sys.P_list:
        cnn_u, ls_u = results[P]["cnn_per_user_20"], results[P]["ls_per_user_20"]
        if cnn_u is None:
            continue
        matrix += [cnn_u, ls_u]
        row_labels += [f"P={P} CNN", f"P={P} LS"]
    if matrix:
        mat = np.array(matrix)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=np.percentile(mat, 5), vmax=np.percentile(mat, 95))
        ax.set_xticks(range(Sys.K))
        ax.set_xticklabels([f"UE {k+1}" for k in range(Sys.K)], fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7.5)
        for r in range(len(row_labels)):
            for c_ in range(Sys.K):
                ax.text(c_, r, f"{mat[r, c_]:.1f}", ha="center", va="center",
                        fontsize=7.5, color="black" if -15 < mat[r, c_] < -2 else "white")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.ax.tick_params(colors=_TEXT, labelsize=7)
        cb.set_label("NMSE (dB)", color=_TEXT, fontsize=8)
        for sep in range(2, len(row_labels), 2):
            ax.axhline(sep - 0.5, color=_SPINE_C, lw=0.8, alpha=0.7)
    _style(ax, "Per-user NMSE heatmap @ SNR=20 dB  [V10]", "User equipment", "Estimator", legend=False)

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=_BG)
    print(f"\nPlot saved -> {os.path.abspath(out)}")


# --- SUMMARY -----------------------------------------------------------------
def print_summary(results, snrs):
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY  [V10]")
    print("=" * 70)
    idx20 = int(np.argmin(np.abs(snrs - 20)))
    print(f"\n  At SNR = 20 dB:")
    print(f"  {'P':>4}  {'Overhead':>9}  {'CNN (dB)':>11}  {'LS (dB)':>9}  {'Gain (dB)':>10}")
    print("-" * 70)
    for P in Sys.P_list:
        ov = 100.0 * (1 - P / Sys.P_max)
        cnn, ls = results[P]["cnn"][idx20], results[P]["ls"][idx20]
        gain = ls - cnn
        print(f"  {P:>4}  {ov:>8.0f}%  {cnn:>10.2f}  {ls:>8.2f}  {gain:>9.2f}{'  +' if gain > 0 else '  x'}")
    print("=" * 70)
    print("\n  Full SNR summary (CNN gain = LS - CNN, dB):")
    print(f"  {'SNR':>6}  " + "  ".join(f"P={P:>2}" for P in Sys.P_list))
    print("-" * 50)
    for i, snr in enumerate(snrs):
        gains = [results[P]["ls"][i] - results[P]["cnn"][i] for P in Sys.P_list]
        print(f"  {snr:>5} dB  " + "  ".join(f"{g:>+6.2f}" for g in gains))
    print("=" * 70)


# --- CHANNEL MODEL SANITY CHECK ----------------------------------------------
def channel_sanity_check():
    print("\n" + "-" * 60)
    print("  V10 CHANNEL SANITY CHECK  (angular_spread = 25deg)")
    print("-" * 60)

    H_batch = _generate_channel_batch(200, L=3, kappa=1.0,
                                       angular_spread_deg=V10_ANGULAR_SPREAD_DEG,
                                       phi_vec=make_phi_vec(Sys.N))

    H_fft = np.fft.fft(H_batch, axis=-1)
    energy = np.mean(np.abs(H_fft) ** 2, axis=(0, 1))
    energy /= energy.sum()

    sorted_e = np.sort(energy)[::-1]
    n_bins_90 = int(np.argmax(np.cumsum(sorted_e) >= 0.90)) + 1

    _, sv, _ = np.linalg.svd(H_batch[0])
    sv_norm = sv / sv.sum()
    eff_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12)))

    alias_period = Sys.M // 4
    cluster_width = int(np.sin(np.deg2rad(V10_ANGULAR_SPREAD_DEG)) * Sys.M)

    print(f"  Angular spread              : {V10_ANGULAR_SPREAD_DEG}°")
    print(f"  Cluster width (DFT bins)    : ~{cluster_width}")
    print(f"  Alias period (P=4)          : {alias_period} bins")
    print(f"  Cluster > alias period?     : {'YES' if cluster_width > alias_period else 'NO  (aliases ambiguous)'}")
    print(f"  DFT bins containing 90% E   : {n_bins_90} / {Sys.M}")
    print(f"  DFT bin 0 energy fraction   : {energy[0]*100:.1f}%")
    print(f"  Effective channel rank      : {eff_rank:.1f} / {min(Sys.K, Sys.M)}")
    print("-" * 60 + "\n")


# --- CLI ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="IOS CSI Estimation — V10 (DualPathIOSNet)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-sanity", action="store_true")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    total_start = time.time()

    print("=" * 70)
    print("  IOS CSI ESTIMATION -- V10  (DualPathIOSNet)")
    print("=" * 70)
    print(f"  System  : M={Sys.M} | N={Sys.N} | K={Sys.K} | P_max={Sys.P_max}")
    print(f"  P_list  : {Sys.P_list}")
    print(f"  Device  : {DEVICE}")
    print(f"  AMP     : {USE_AMP}")
    print(f"  Compile : {USE_COMPILE}  (PyTorch {torch.__version__})")
    print(f"  Workers : {NUM_WORKERS}")
    print()
    print("  V10 changes vs V9:")
    print()
    print("  FIX 1 -- DEPTHWISE-SEPARABLE ANGULAR PATH  [Architecture]")
    print("    V9: (K,1) conv collapsed all users -> single shared repr -> expand.")
    print("    V10: depthwise (1x17, groups=C) preserves (B,C,K,M) throughout.")
    print()
    print("  FIX 2 -- EXTENDED PILOT RANGE: P_max 32 -> 128  [Coverage]")
    print("    P_list now covers [4, 8, 16, 32, 64, 128].")
    print("    P>M: pilot matrix tiled; LS via exact pseudoinverse Phi(Phi^H Phi)^-1.")
    print()

    if not args.no_sanity:
        channel_sanity_check()

    print("=" * 60)
    print("  STEP 1: Precomputing V10 Channel Banks")
    print("=" * 60)
    train_bank = ChannelBank(args.train_samples, L=3, kappa=1.0,
                             angular_spread_deg=V10_ANGULAR_SPREAD_DEG)
    val_bank = ChannelBank(args.val_samples, L=3, kappa=1.0,
                           angular_spread_deg=V10_ANGULAR_SPREAD_DEG)

    models, histories = {}, {}
    for P in Sys.P_list:
        m, tr, vl = train_fast(P, train_bank, val_bank, epochs=args.epochs)
        models[P] = m
        histories[P] = (tr, vl)

    if not args.no_eval:
        results, snrs = evaluate_fast(models, train_bank, n_mc=args.mc_samples)
        if not args.no_plot:
            plot_fast(results, snrs, histories, out="ios_results_v10.png")
        print_summary(results, snrs)

    print(f"\nTotal time: {(time.time() - total_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
