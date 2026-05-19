"""IOS CSI Estimation - Compact Simulation Version (V10)"""
import os, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.cuda.is_available()
torch.manual_seed(42); np.random.seed(42)

# ── System Config ──
M, N, K = 32, 64, 4          # BS antennas, IOS elements, users (halved M/N for speed)
P_LIST = [4, 8, 16, 32]      # pilot counts
SNR_RANGE = (-5, 30)

# ── Pilots ──
def build_pilots(P):
    Peff = min(P, M)
    rows = np.round(np.linspace(0, M-1, Peff)).astype(int)
    base = (np.exp(1j*2*np.pi*rows[:,None]*np.arange(M)[None,:]/M)/np.sqrt(M)).astype(np.complex64)
    if P <= M:
        return base, base.conj().T
    pil = np.tile(base, (int(np.ceil(P/Peff)), 1))[:P].astype(np.complex64)
    return pil @ np.linalg.inv((pil.conj().T @ pil).astype(np.complex128)).astype(np.complex64), pil.conj().T

PILOTS = {P: build_pilots(P) for P in P_LIST}  # (ls_mat, pil_ct)

# ── Signal Helpers ──
def add_noise(Y, snr_db):
    s = np.sqrt(np.mean(np.abs(Y)**2) / (2*10**(snr_db/10)))
    return (Y + s*(np.random.randn(*Y.shape)+1j*np.random.randn(*Y.shape)).astype(np.complex64)).astype(np.complex64)

def ls_est(Y, P):
    return (Y @ PILOTS[P][0]).astype(np.complex64)

def to_3ch(H_ls):
    """Complex H_ls -> (3,K,M) float: [real, imag, log_scale]."""
    batched = H_ls.ndim == 3
    if not batched: H_ls = H_ls[None]
    scale = np.sqrt(np.mean(np.abs(H_ls)**2, axis=(1,2), keepdims=True)) + 1e-8
    Hn = H_ls / scale
    ri = np.stack([Hn.real, Hn.imag], axis=1).astype(np.float32)
    log_s = np.broadcast_to((np.log(scale)/3).reshape(-1,1,1,1), (H_ls.shape[0],1,K,M)).copy().astype(np.float32)
    out = np.concatenate([ri, log_s], axis=1)
    return (out[0], float(scale[0,0,0])) if not batched else (out, scale)

# ── Channel Model (simplified Rayleigh + LOS) ──
def steering(n, angles):
    return (np.exp(1j*np.pi*np.arange(n,dtype=np.float32)[None,:]*np.sin(angles).astype(np.float32)[:,None])/np.sqrt(n)).astype(np.complex64)

def gen_channels(bs, L=2, kappa=1.0):
    """Generate (bs, K, M) effective IOS channels."""
    los_s, nlos_s = np.sqrt(kappa/(kappa+1)), np.sqrt(1/(kappa+1))
    phi = np.exp(1j*np.random.uniform(0,2*np.pi,N)).astype(np.complex64)
    z = np.zeros(bs, dtype=np.float32)
    # BS-IOS
    H_BI = los_s * steering(N,z)[:,:,None] * steering(M,z)[:,None,:].conj()
    for _ in range(L):
        a = nlos_s*(np.random.randn(bs)+1j*np.random.randn(bs)).astype(np.complex64)/np.sqrt(2*L)
        H_BI += a[:,None,None]*steering(N,np.random.uniform(-1,1,bs))[:,:,None]*steering(M,np.random.uniform(-1,1,bs))[:,None,:].conj()
    # IOS-UE
    ang = np.full(bs, 0.09, dtype=np.float32)
    H_IU = los_s * steering(K,ang)[:,:,None] * steering(N,ang)[:,None,:].conj()
    for _ in range(L):
        a = nlos_s*(np.random.randn(bs)+1j*np.random.randn(bs)).astype(np.complex64)/np.sqrt(2*L)
        H_IU += a[:,None,None]*steering(K,np.random.uniform(-1,1,bs))[:,:,None]*steering(N,np.random.uniform(-1,1,bs))[:,None,:].conj()
    # Cascade
    H = np.matmul(H_IU, phi[None,:,None]*H_BI)
    norms = np.linalg.norm(H.reshape(bs,-1), axis=1, keepdims=True)
    return (H / (norms[:,:,None]+1e-8)).astype(np.complex64)

# ── Dataset ──
class ChanDataset(Dataset):
    def __init__(self, H, P, augment=True):
        self.H, self.P, self.aug = H, P, augment
        self.pil_ct = PILOTS[P][1]
    def __len__(self): return len(self.H)
    def __getitem__(self, i):
        h = self.H[i].copy()
        snr = np.random.uniform(*SNR_RANGE)
        Y = add_noise(h @ self.pil_ct, snr)
        if self.aug and np.random.rand()<0.3:
            ph = np.exp(1j*np.random.uniform(0,2*np.pi)).astype(np.complex64)
            Y *= ph; h *= ph
        hls = ls_est(Y, self.P)
        x, sc = to_3ch(hls)
        tgt = np.stack([(h/sc).real, (h/sc).imag], axis=-1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(tgt)

# ── Model ──
class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(ch//8, 4)
        self.fc = nn.Sequential(nn.Conv2d(ch,mid,1,bias=False), nn.ReLU(True), nn.Conv2d(mid,ch,1,bias=False))
        self.sp = nn.Conv2d(2,1,5,padding=2,bias=False)
    def forward(self, x):
        ca = torch.sigmoid(self.fc(x.mean((-2,-1),keepdim=True)) + self.fc(x.amax((-2,-1),keepdim=True)))
        x = x * ca
        sa = torch.sigmoid(self.sp(torch.cat([x.mean(1,keepdim=True), x.amax(1,keepdim=True)], 1)))
        return x * sa

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(True), nn.Conv2d(ch,ch,3,padding=1,bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True), nn.Conv2d(ch,ch,3,padding=1,bias=False))
        self.att = CBAM(ch)
    def forward(self, x): return x + self.att(self.net(x))

class IOSNet(nn.Module):
    def __init__(self, base_ch=48):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(nn.Conv2d(3,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
        self.enc1 = nn.Sequential(nn.Conv2d(c,c*2,3,stride=(1,2),padding=1,bias=False),nn.BatchNorm2d(c*2),nn.ReLU(True),ResBlock(c*2))
        self.enc2 = nn.Sequential(nn.Conv2d(c*2,c*4,3,stride=(1,2),padding=1,bias=False),nn.BatchNorm2d(c*4),nn.ReLU(True),ResBlock(c*4))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(c*4,c*2,kernel_size=(1,2),stride=(1,2),bias=False),nn.BatchNorm2d(c*2),nn.ReLU(True))
        self.dec1_conv = nn.Sequential(nn.Conv2d(c*4,c*2,3,padding=1,bias=False),nn.BatchNorm2d(c*2),nn.ReLU(True),ResBlock(c*2))
        self.dec0 = nn.Sequential(nn.ConvTranspose2d(c*2,c,kernel_size=(1,2),stride=(1,2),bias=False),nn.BatchNorm2d(c),nn.ReLU(True))
        self.head = nn.Sequential(nn.Conv2d(c*2,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(True),nn.Conv2d(c,2,1))

    def forward(self, x):
        skip_in = x[:,:2].permute(0,2,3,1).contiguous()
        f0 = self.stem(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        d1 = self.dec1_conv(torch.cat([self.dec2(f2), f1], 1))
        d0 = self.head(torch.cat([self.dec0(d1), f0], 1))
        return skip_in + d0.permute(0,2,3,1).contiguous()

# ── Loss ──
def nmse_loss(p, t):
    e = (p-t).pow(2).sum((1,2,3))
    return (e / (t.pow(2).sum((1,2,3))+1e-8)).mean()

# ── Training ──
def train(P, H_train, H_val, epochs=20):
    print(f"\n--- Training P={P} | {epochs} ep | {DEVICE} ---")
    model = IOSNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=AMP)
    train_dl = DataLoader(ChanDataset(H_train, P), batch_size=128, shuffle=True, num_workers=0)
    val_dl = DataLoader(ChanDataset(H_val, P, augment=False), batch_size=256, num_workers=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, patience, ckpt = 1e9, 0, f"best_P{P}.pth"
    tr_h, vl_h = [], []

    for ep in range(1, epochs+1):
        model.train(); tloss, nb = 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=AMP):
                loss = nmse_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tloss += loss.item(); nb += 1
        sched.step(); tr_h.append(tloss/nb)

        model.eval(); vloss, vn = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                vloss += nmse_loss(model(x.to(DEVICE)), y.to(DEVICE)).item(); vn += 1
        vl_h.append(vloss/vn)
        print(f"  Ep {ep:02d}/{epochs} | Train {tr_h[-1]:.4f} | Val {vl_h[-1]:.4f}")

        if vl_h[-1] < best:
            best, patience = vl_h[-1], 0; torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= 8: print(f"  Early stop ep {ep}"); break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, tr_h, vl_h

# ── Evaluate ──
def evaluate(models, n_mc=300):
    snrs = np.arange(-5, 36, 5)
    H_true = gen_channels(n_mc)
    results = {}
    for P in P_LIST:
        model = models[P]; model.eval()
        Y_clean = np.matmul(H_true, np.broadcast_to(PILOTS[P][1][None], (n_mc,)+PILOTS[P][1].shape))
        cnn_nmse, ls_nmse = [], []
        for snr in snrs:
            Yn = add_noise(Y_clean, snr)
            Hls = ls_est(Yn, P)
            x3, sc = to_3ch(Hls)
            with torch.no_grad():
                pred = model(torch.from_numpy(x3).to(DEVICE)).cpu().numpy()
            Hcnn = (pred[...,0]+1j*pred[...,1])*sc
            cnn_e = np.linalg.norm((Hcnn-H_true).reshape(n_mc,-1),axis=1)**2 / (np.linalg.norm(H_true.reshape(n_mc,-1),axis=1)**2+1e-8)
            ls_e = np.linalg.norm((Hls-H_true).reshape(n_mc,-1),axis=1)**2 / (np.linalg.norm(H_true.reshape(n_mc,-1),axis=1)**2+1e-8)
            cnn_nmse.append(10*np.log10(cnn_e.mean())); ls_nmse.append(10*np.log10(ls_e.mean()))
        results[P] = {"cnn": cnn_nmse, "ls": ls_nmse}
    return results, snrs

# ── Plot ──
def plot(results, snrs, histories, out="ios_compact_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), tight_layout=True)
    colors = ["#C0392B","#148F77","#1A5276","#B7950B"]

    for i, P in enumerate(P_LIST):
        axes[0].plot(snrs, results[P]["cnn"], "o-", c=colors[i], lw=2, ms=5, label=f"CNN P={P}")
        axes[0].plot(snrs, results[P]["ls"], "s--", c=colors[i], lw=1.2, ms=3, alpha=.5, label=f"LS  P={P}")
    axes[0].set(xlabel="SNR (dB)", ylabel="NMSE (dB)", title="NMSE vs SNR"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=.5)

    for i, P in enumerate(P_LIST):
        gain = [results[P]["ls"][j]-results[P]["cnn"][j] for j in range(len(snrs))]
        axes[1].plot(snrs, gain, "o-", c=colors[i], lw=2, ms=5, label=f"P={P}")
    axes[1].axhline(0, c="k", lw=.5, ls="--")
    axes[1].set(xlabel="SNR (dB)", ylabel="Gain (dB)", title="CNN Gain over LS"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=.5)

    for i, P in enumerate(P_LIST):
        axes[2].plot(histories[P][0], "-", c=colors[i], lw=1.5, label=f"Train P={P}")
        axes[2].plot(histories[P][1], "--", c=colors[i], lw=1.5, label=f"Val P={P}")
    axes[2].set(xlabel="Epoch", ylabel="Loss", title="Training Loss"); axes[2].set_yscale("log"); axes[2].legend(fontsize=6); axes[2].grid(True, alpha=.5)

    plt.savefig(out, dpi=150); print(f"Plot saved -> {os.path.abspath(out)}")

# ── Summary ──
def summary(results, snrs):
    idx20 = int(np.argmin(np.abs(snrs-20)))
    print(f"\n{'='*50}\n  RESULTS @ SNR=20dB\n{'='*50}")
    print(f"  {'P':>4}  {'CNN(dB)':>8}  {'LS(dB)':>8}  {'Gain':>6}")
    for P in P_LIST:
        c, l = results[P]["cnn"][idx20], results[P]["ls"][idx20]
        print(f"  {P:>4}  {c:>8.2f}  {l:>8.2f}  {l-c:>+6.2f}")
    print("="*50)

# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--mc-n", type=int, default=300)
    args = ap.parse_args()

    t0 = time.time()
    print(f"IOS CSI Compact | M={M} N={N} K={K} | P={P_LIST} | {DEVICE}")
    print("Generating channels...")
    H_tr, H_vl = gen_channels(args.train_n), gen_channels(args.val_n)

    models, hist = {}, {}
    for P in P_LIST:
        m, tr, vl = train(P, H_tr, H_vl, args.epochs)
        models[P], hist[P] = m, (tr, vl)

    print("\nEvaluating...")
    res, snrs = evaluate(models, args.mc_n)
    plot(res, snrs, hist)
    summary(res, snrs)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
