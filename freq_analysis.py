"""
Frequency analysis of the human signal in YOLOv3 feature maps.

1. DELTA FFT: FFT on (with-without) per channel, log2 amplified if weak
2. POLYNOMIAL FFT: treat channel activations as polynomial coeffs, 1D FFT
3. 2D FFT on delta maps: full spatial frequency of human signal
"""
import os, sys, json, csv
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\freq_deep"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128,128,128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE), arr

def forward_capture(model, x):
    caps = {}
    los = []
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional","upsample","maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups",1))
            gi = int(md.get("group_id",0))
            x = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, x.size(2) if hasattr(x,'size') else 416)
        if md["type"] == "convolutional":
            caps[i] = x.detach().clone()
        los.append(x)
    return caps

def radial_profile(spec, H, W):
    cy, cx = H//2, W//2
    y, x = np.indices((H, W))
    r = np.sqrt((y-cy)**2 + (x-cx)**2).astype(int)
    rmax = min(cy, cx)
    prof = np.zeros(rmax+1)
    for ri in range(rmax+1):
        m = r == ri
        if m.any():
            prof[ri] = spec[m].mean()
    return prof, rmax

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("Loading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("Loading images...")
    iw, aw = load_image(IMG_WITH)
    iwo, awo = load_image(IMG_WITHOUT)

    print("Forward passes...")
    cw = forward_capture(model, iw)
    cwo = forward_capture(model, iwo)

    csv_rows = []
    all_results = {}

    for li in LAYERS:
        if li not in cw or li not in cwo: continue
        fw = cw[li].squeeze(0).cpu().numpy()  # (C,H,W)
        fo = cwo[li].squeeze(0).cpu().numpy()
        C, H, W = fw.shape
        delta = fw - fo  # (C,H,W) the human signal

        print(f"\nLayer {li}: ({C},{H},{W})")

        # --- 1. DELTA FFT per channel (1D on flattened spatial) ---
        # Flatten each channel to 1D, FFT, average power across channels
        delta_flat = delta.reshape(C, -1)  # (C, H*W)
        ffts_1d = np.abs(np.fft.fft(delta_flat, axis=1))**2  # (C, H*W)
        mean_1d_power = ffts_1d.mean(axis=0)  # (H*W,)

        # Check if signal is weak -> log2 amplify
        max_power = mean_1d_power.max()
        is_weak = max_power < 1e-6
        if is_weak:
            # log2 transform before FFT: amplifies weak spectral features
            delta_log = np.log2(np.abs(delta) + 1.0) * np.sign(delta)
            delta_log_flat = delta_log.reshape(C, -1)
            ffts_log = np.abs(np.fft.fft(delta_log_flat, axis=1))**2
            mean_1d_power = ffts_log.mean(axis=0)
            tag = "log2"
        else:
            tag = "raw"

        # Top frequencies (excluding DC)
        half = len(mean_1d_power) // 2
        pos_power = mean_1d_power[:half]
        pos_power[0] = 0  # zero out DC
        top_freq_idx = np.argsort(pos_power)[::-1][:10]
        top_freq_vals = pos_power[top_freq_idx]

        # --- 2. POLYNOMIAL FFT on embeddings ---
        # Treat each channel's activation as polynomial coefficients
        # a0 + a1*x + a2*x^2 + ... evaluated at roots of unity = FFT
        # Use mean-pooled rows as coeffs (H coefficients per channel)
        poly_coeffs = delta.mean(axis=2)  # (C, H) -- avg over W -> poly degree H-1
        poly_fft = np.abs(np.fft.fft(poly_coeffs, axis=1))**2  # (C, H)
        mean_poly_power = poly_fft.mean(axis=0)
        poly_top = np.argsort(mean_poly_power[1:])[::-1][:5] + 1  # skip DC

        # Also do polynomial FFT on column-pooled (W coefficients)
        poly_coeffs_w = delta.mean(axis=1)  # (C, W)
        poly_fft_w = np.abs(np.fft.fft(poly_coeffs_w, axis=1))**2
        mean_poly_power_w = poly_fft_w.mean(axis=0)

        # --- 3. 2D FFT on delta maps ---
        # Per-channel 2D FFT of delta, then average
        specs_2d = np.zeros((H, W), dtype=np.float32)
        for c in range(C):
            f2 = np.fft.fft2(delta[c])
            specs_2d += np.abs(np.fft.fftshift(f2))**2
        specs_2d /= C
        specs_2d_log = np.log1p(specs_2d)

        # Radial profile of 2D delta spectrum
        rad_prof, rmax = radial_profile(specs_2d, H, W)
        rad_prof_norm = rad_prof / (rad_prof.sum() + 1e-12)

        # Frequency bands
        r4 = rmax // 4
        r2 = rmax // 2
        lf = rad_prof[:r4].sum() / (rad_prof.sum()+1e-12)
        mf = rad_prof[r4:r2].sum() / (rad_prof.sum()+1e-12)
        hf = rad_prof[r2:].sum() / (rad_prof.sum()+1e-12)

        # Peak radial frequency
        peak_r = np.argmax(rad_prof[1:]) + 1  # skip DC

        # Top channels by 2D spectral power in delta
        ch_2d_power = np.zeros(C)
        for c in range(C):
            ch_2d_power[c] = (np.abs(np.fft.fft2(delta[c]))**2).sum()
        top_ch = np.argsort(ch_2d_power)[::-1][:10].tolist()

        all_results[li] = {
            "shape": [C,H,W], "weak_signal": is_weak, "transform": tag,
            "top_freq_1d": top_freq_idx.tolist(), "top_freq_power": top_freq_vals.tolist(),
            "poly_top_freq_h": poly_top.tolist(), "poly_top_freq_w": np.argsort(mean_poly_power_w[1:])[::-1][:5].tolist(),
            "lf": float(lf), "mf": float(mf), "hf": float(hf),
            "peak_radial_freq": int(peak_r), "top_2d_channels": [int(c) for c in top_ch],
        }

        csv_rows.append({
            "layer": li, "C": C, "H": H, "W": W, "weak": is_weak, "transform": tag,
            "top1_freq": int(top_freq_idx[0]), "top1_power": float(top_freq_vals[0]),
            "lf": float(lf), "mf": float(mf), "hf": float(hf),
            "peak_r": int(peak_r), "top_ch_2d": str(top_ch[:5]),
        })

        # --- PLOTS ---
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Layer {li} ({C},{H},{W}) — Human Signal Frequency Analysis [{tag}]", fontsize=14)

        # 1D delta FFT spectrum
        axes[0,0].plot(pos_power, linewidth=1.5, color='darkred')
        axes[0,0].set_title(f"1D FFT of delta (per-channel avg) [{tag}]")
        axes[0,0].set_xlabel("Frequency bin")
        axes[0,0].set_ylabel("Power")
        axes[0,0].set_yscale("log")
        axes[0,0].grid(True, alpha=0.3)
        for fi in top_freq_idx[:3]:
            axes[0,0].axvline(fi, color='orange', alpha=0.5, linestyle='--')

        # Polynomial FFT (row-pooled, H coeffs)
        axes[0,1].plot(mean_poly_power, linewidth=1.5, color='darkblue')
        axes[0,1].set_title("Polynomial FFT (row-pooled, H coeffs)")
        axes[0,1].set_xlabel("Frequency (polynomial degree)")
        axes[0,1].set_ylabel("Power")
        axes[0,1].set_yscale("log")
        axes[0,1].grid(True, alpha=0.3)

        # Polynomial FFT (col-pooled, W coeffs)
        axes[0,2].plot(mean_poly_power_w, linewidth=1.5, color='darkgreen')
        axes[0,2].set_title("Polynomial FFT (col-pooled, W coeffs)")
        axes[0,2].set_xlabel("Frequency (polynomial degree)")
        axes[0,2].set_ylabel("Power")
        axes[0,2].set_yscale("log")
        axes[0,2].grid(True, alpha=0.3)

        # 2D FFT delta spectrum (log)
        im = axes[1,0].imshow(specs_2d_log, cmap='inferno', aspect='auto')
        axes[1,0].set_title("2D FFT of delta (log, avg across channels)")
        plt.colorbar(im, ax=axes[1,0], fraction=0.046)

        # Radial profile
        axes[1,1].plot(rad_prof_norm, linewidth=2, color='purple')
        axes[1,1].set_title(f"Radial profile — LF={lf:.3f} MF={mf:.3f} HF={hf:.3f}")
        axes[1,1].set_xlabel("Radial frequency")
        axes[1,1].set_ylabel("Fraction of power")
        axes[1,1].set_yscale("log")
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axvline(peak_r, color='red', alpha=0.7, linestyle='--', label=f'peak r={peak_r}')
        axes[1,1].legend()

        # Top channels 2D power
        axes[1,2].bar(range(len(top_ch)), ch_2d_power[top_ch], color='darkorange')
        axes[1,2].set_title("Top 10 channels by 2D delta spectral power")
        axes[1,2].set_xlabel("Rank")
        axes[1,2].set_ylabel("Total 2D power")
        axes[1,2].set_xticks(range(len(top_ch)))
        axes[1,2].set_xticklabels([str(c) for c in top_ch], fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"freq_L{li:03d}.png"), dpi=150)
        plt.close()

        print(f"  LF={lf:.4f} MF={mf:.4f} HF={hf:.4f}  peak_r={peak_r}  "
              f"top1d_freq={top_freq_idx[0]}  weak={is_weak}  top_ch={top_ch[:3]}")

    # Cross-layer summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Cross-Layer: Human Signal Frequency Analysis", fontsize=16)
    ls = sorted(all_results.keys())
    x = range(len(ls))

    lfs = [all_results[l]["lf"] for l in ls]
    hfs = [all_results[l]["hf"] for l in ls]
    peaks = [all_results[l]["peak_radial_freq"] for l in ls]
    top1s = [all_results[l]["top_freq_1d"][0] for l in ls]

    axes[0,0].bar(x, lfs, alpha=0.7, label="LF power", color="green")
    axes[0,0].bar(x, hfs, alpha=0.7, label="HF power", color="red")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,0].set_title("LF vs HF power of human delta per layer")
    axes[0,0].legend()

    axes[0,1].bar(x, peaks, color="purple")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,1].set_title("Peak radial frequency of human signal per layer")

    axes[1,0].bar(x, top1s, color="darkred")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[1,0].set_title("Top 1D frequency bin of human delta per layer")

    # Band ratio stacked
    mfs = [all_results[l]["mf"] for l in ls]
    axes[1,1].bar(x, lfs, label="LF", color="green")
    axes[1,1].bar(x, mfs, bottom=lfs, label="MF", color="orange")
    axes[1,1].bar(x, hfs, bottom=[l+m for l,m in zip(lfs,mfs)], label="HF", color="red")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[1,1].set_title("Frequency band distribution (stacked)")
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "freq_cross_layer.png"), dpi=200)
    plt.close()

    # Save CSV + JSON
    with open(os.path.join(OUTPUT_DIR, "freq_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader(); w.writerows(csv_rows)

    jdata = {}
    for li, r in all_results.items():
        jdata[str(li)] = {k: (v if not isinstance(v, list) else v) for k, v in r.items()}
    with open(os.path.join(OUTPUT_DIR, "freq_summary.json"), "w") as f:
        json.dump(jdata, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))

    print("\n" + "="*70)
    print("SUMMARY: Dominant Human-Introduced Frequencies")
    print("="*70)
    print(f"\n{'Layer':>6}  {'Shape':>14}  {'LF':>7}  {'MF':>7}  {'HF':>7}  {'PeakR':>6}  {'Top1D':>6}  {'Transform'}")
    print("-"*80)
    for li in ls:
        r = all_results[li]
        s = r["shape"]
        print(f"{li:6d}  ({s[0]:3d},{s[1]:3d},{s[2]:3d})  "
              f"{r['lf']:7.4f}  {r['mf']:7.4f}  {r['hf']:7.4f}  "
              f"{r['peak_radial_freq']:6d}  {r['top_freq_1d'][0]:6d}  {r['transform']}")

    print(f"\nOutputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
