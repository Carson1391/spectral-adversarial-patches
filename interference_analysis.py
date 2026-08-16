"""
4-Way FFT Analysis: Raw Image + Model Embeddings, With and Without Human.

1. FFT of raw image WITHOUT human (pixel frequency ground truth)
2. FFT of model embeddings WITHOUT human (how network processes empty scene)
3. FFT of raw image WITH human (pixel frequency with person)
4. FFT of model embeddings WITH human (how network processes person scene)

Then compare all four to see interference patterns:
  - Image interference: FFT(with) vs FFT(without) at pixel level
  - Embedding interference: FFT(with) vs FFT(without) at each network layer
  - Cross-spectrum: FFT(with) * conj(FFT(without)) for both image and embeddings
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\interference"
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
    prof = np.zeros(rmax+1, dtype=np.float64)
    for ri in range(rmax+1):
        m = r == ri
        if m.any():
            prof[ri] = spec[m].mean()
    return prof, rmax

def band_ratios(prof, rmax):
    r4 = max(1, rmax // 4)
    r2 = max(1, rmax // 2)
    total = prof.sum() + 1e-12
    lf = prof[:r4].sum() / total
    mf = prof[r4:r2].sum() / total
    hf = prof[r2:].sum() / total
    return lf, mf, hf

def fft_2d_image(arr_2d):
    """2D FFT on a single 2D array. Returns shifted power spectrum, radial profile, band ratios."""
    fft = np.fft.fft2(arr_2d)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    H, W = arr_2d.shape
    rad, rmax = radial_profile(power, H, W)
    lf, mf, hf = band_ratios(rad, rmax)
    return {
        "power": power,
        "radial": rad,
        "rmax": rmax,
        "lf": lf, "mf": mf, "hf": hf,
        "magnitude": np.abs(fft_shift),
        "phase": np.angle(fft_shift),
    }

def fft_feature_maps(feat):
    """2D FFT on feature maps (C, H, W). Averages power spectrum across channels.
    Also returns per-channel total power for ranking."""
    C, H, W = feat.shape
    specs = np.zeros((H, W), dtype=np.float64)
    ch_power = np.zeros(C)
    for c in range(C):
        fft = np.fft.fft2(feat[c])
        fft_shift = np.fft.fftshift(fft)
        p = np.abs(fft_shift) ** 2
        specs += p
        ch_power[c] = p.sum()
    specs /= C
    rad, rmax = radial_profile(specs, H, W)
    lf, mf, hf = band_ratios(rad, rmax)
    return {
        "mean_power": specs,
        "radial": rad,
        "rmax": rmax,
        "lf": lf, "mf": mf, "hf": hf,
        "ch_power": ch_power,
        "top_channels": np.argsort(ch_power)[::-1][:10].tolist(),
    }

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("4-Way FFT: Raw Image + Embeddings, With and Without Human")
    print("="*70)

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("Loading images...")
    iw, arr_w = load_image(IMG_WITH)
    iwo, arr_wo = load_image(IMG_WITHOUT)
    H, W, _ = arr_w.shape

    # ============================================================
    # 1. FFT OF RAW IMAGES (with and without human, separately)
    # ============================================================
    print("\n--- FFT OF RAW IMAGES ---")
    img_fft = {}
    for tag, arr in [("with", arr_w), ("without", arr_wo)]:
        ch_results = {}
        for cn, ci in [("R", 0), ("G", 1), ("B", 2)]:
            ch_results[cn] = fft_2d_image(arr[:, :, ci])
        # Grayscale (average of RGB)
        gray = arr.mean(axis=2)
        ch_results["gray"] = fft_2d_image(gray)
        img_fft[tag] = ch_results
        g = ch_results["gray"]
        print(f"  Image {tag} (gray): LF={g['lf']:.4f} MF={g['mf']:.4f} HF={g['hf']:.4f}")

    # ============================================================
    # 2. FFT OF MODEL EMBEDDINGS (with and without human, separately)
    # ============================================================
    print("\n--- FFT OF MODEL EMBEDDINGS ---")
    print("Forward passes...")
    caps_w = forward_capture(model, iw)
    caps_wo = forward_capture(model, iwo)

    emb_fft = {"with": {}, "without": {}}
    for li in LAYERS:
        if li not in caps_w or li not in caps_wo: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()
        fo = caps_wo[li].squeeze(0).cpu().numpy()
        emb_fft["with"][li] = fft_feature_maps(fw)
        emb_fft["without"][li] = fft_feature_maps(fo)
        ew = emb_fft["with"][li]
        eo = emb_fft["without"][li]
        print(f"  Layer {li:3d} ({fw.shape[0]:4d},{fw.shape[1]:3d},{fw.shape[2]:3d}): "
              f"with LF={ew['lf']:.4f} MF={ew['mf']:.4f} HF={ew['hf']:.4f}  "
              f"without LF={eo['lf']:.4f} MF={eo['mf']:.4f} HF={eo['hf']:.4f}")

    # ============================================================
    # 3. INTERFERENCE COMPARISONS
    # ============================================================
    print("\n--- INTERFERENCE PATTERNS ---")

    # 3a. Image-level: FFT of delta (with - without) at pixel level
    img_delta_fft = {}
    for cn in ["R", "G", "B", "gray"]:
        if cn == "gray":
            d = arr_w.mean(axis=2) - arr_wo.mean(axis=2)
        else:
            ci = {"R": 0, "G": 1, "B": 2}[cn]
            d = arr_w[:, :, ci] - arr_wo[:, :, ci]
        img_delta_fft[cn] = fft_2d_image(d)
        r = img_delta_fft[cn]
        print(f"  Image delta {cn}: LF={r['lf']:.4f} MF={r['mf']:.4f} HF={r['hf']:.4f}")

    # 3b. Image cross-spectrum: FFT(with) * conj(FFT(without))
    img_cross = {}
    for cn in ["R", "G", "B", "gray"]:
        if cn == "gray":
            a = arr_w.mean(axis=2)
            b = arr_wo.mean(axis=2)
        else:
            ci = {"R": 0, "G": 1, "B": 2}[cn]
            a = arr_w[:, :, ci]
            b = arr_wo[:, :, ci]
        fa = np.fft.fft2(a)
        fb = np.fft.fft2(b)
        cross = fa * np.conj(fb)
        cross_power = np.abs(np.fft.fftshift(cross)) ** 2
        rad, rmax = radial_profile(cross_power, H, W)
        lf, mf, hf = band_ratios(rad, rmax)
        # Coherence: |cross|^2 / (power_a * power_b)
        coh = np.abs(cross) ** 2 / (np.abs(fa)**2 * np.abs(fb)**2 + 1e-12)
        coh_shifted = np.fft.fftshift(coh)
        img_cross[cn] = {
            "cross_power": cross_power, "coherence": coh_shifted,
            "radial": rad, "lf": lf, "mf": mf, "hf": hf,
            "mean_coherence": float(coh.mean()),
        }
        print(f"  Cross-spectrum {cn}: LF={lf:.4f} MF={mf:.4f} HF={hf:.4f}  coh={coh.mean():.4f}")

    # 3c. Embedding delta: FFT of (with - without) feature maps per layer
    emb_delta = {}
    for li in LAYERS:
        if li not in caps_w or li not in caps_wo: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()
        fo = caps_wo[li].squeeze(0).cpu().numpy()
        delta = fw - fo
        emb_delta[li] = fft_feature_maps(delta)
        r = emb_delta[li]
        print(f"  Embedding delta L{li:3d}: LF={r['lf']:.4f} MF={r['mf']:.4f} HF={r['hf']:.4f}")

    # 3d. Embedding cross-spectrum: FFT(with) * conj(FFT(without)) per layer
    emb_cross = {}
    for li in LAYERS:
        if li not in caps_w or li not in caps_wo: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()
        fo = caps_wo[li].squeeze(0).cpu().numpy()
        C, Hf, Wf = fw.shape
        cross_power = np.zeros((Hf, Wf), dtype=np.float64)
        for c in range(C):
            fa = np.fft.fft2(fw[c])
            fb = np.fft.fft2(fo[c])
            cross_power += np.abs(np.fft.fftshift(fa * np.conj(fb))) ** 2
        cross_power /= C
        rad, rmax = radial_profile(cross_power, Hf, Wf)
        lf, mf, hf = band_ratios(rad, rmax)
        emb_cross[li] = {"radial": rad, "lf": lf, "mf": mf, "hf": hf, "cross_power": cross_power}

    # ============================================================
    # PLOTS
    # ============================================================
    print("\nGenerating plots...")

    # --- Plot 1: Raw Image FFT (with, without, delta, cross) ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Raw Image FFT: With Human vs Without Human (Grayscale)", fontsize=16)

    gw = img_fft["with"]["gray"]
    go = img_fft["without"]["gray"]
    gd = img_delta_fft["gray"]
    gc = img_cross["gray"]

    r = np.arange(len(gw["radial"]))
    axes[0,0].plot(r, gw["radial"] / (gw["radial"].sum()+1e-12), label="With human", color="red", linewidth=2)
    axes[0,0].plot(r, go["radial"] / (go["radial"].sum()+1e-12), label="Without human", color="blue", linewidth=2)
    axes[0,0].set_yscale("log")
    axes[0,0].set_title("Radial power: with vs without")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(r, gd["radial"] / (gd["radial"].sum()+1e-12), label="Delta FFT", color="green", linewidth=2)
    axes[0,1].set_yscale("log")
    axes[0,1].set_title("Delta (with - without) radial power")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(r, gc["radial"] / (gc["radial"].sum()+1e-12), label="Cross-power", color="purple", linewidth=2)
    axes[0,2].set_yscale("log")
    axes[0,2].set_title("Cross-spectrum (interference) radial power")
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)

    bands = ["LF", "MF", "HF"]
    x_b = np.arange(3)
    axes[0,3].bar(x_b - 0.3, [gw["lf"], gw["mf"], gw["hf"]], 0.2, label="With", color="red", alpha=0.7)
    axes[0,3].bar(x_b - 0.1, [go["lf"], go["mf"], go["hf"]], 0.2, label="Without", color="blue", alpha=0.7)
    axes[0,3].bar(x_b + 0.1, [gd["lf"], gd["mf"], gd["hf"]], 0.2, label="Delta", color="green", alpha=0.7)
    axes[0,3].bar(x_b + 0.3, [gc["lf"], gc["mf"], gc["hf"]], 0.2, label="Cross", color="purple", alpha=0.7)
    axes[0,3].set_xticks(x_b); axes[0,3].set_xticklabels(bands)
    axes[0,3].set_title("Band ratios: with / without / delta / cross")
    axes[0,3].legend()

    im = axes[1,0].imshow(np.log1p(gw["power"]), cmap="inferno", aspect="auto")
    axes[1,0].set_title("2D spectrum: WITH human (log)")
    plt.colorbar(im, ax=axes[1,0], fraction=0.046)

    im = axes[1,1].imshow(np.log1p(go["power"]), cmap="inferno", aspect="auto")
    axes[1,1].set_title("2D spectrum: WITHOUT human (log)")
    plt.colorbar(im, ax=axes[1,1], fraction=0.046)

    im = axes[1,2].imshow(np.log1p(gd["power"]), cmap="inferno", aspect="auto")
    axes[1,2].set_title("2D spectrum: DELTA (log)")
    plt.colorbar(im, ax=axes[1,2], fraction=0.046)

    im = axes[1,3].imshow(np.log1p(gc["cross_power"]), cmap="hot", aspect="auto")
    axes[1,3].set_title("2D cross-spectrum (interference, log)")
    plt.colorbar(im, ax=axes[1,3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_image_fft_4way.png"), dpi=150)
    plt.close()

    # --- Plot 2: Per-layer embedding FFT comparison ---
    n_layers = len(emb_fft["with"])
    fig, axes = plt.subplots(n_layers, 4, figsize=(24, 5 * n_layers))
    fig.suptitle("Per-Layer Embedding FFT: With vs Without vs Delta vs Cross", fontsize=16)

    for row, li in enumerate(sorted(emb_fft["with"].keys())):
        ew = emb_fft["with"][li]
        eo = emb_fft["without"][li]
        ed = emb_delta[li]
        ec = emb_cross[li]
        s = caps_w[li].squeeze(0).cpu().numpy().shape

        r = np.arange(len(ew["radial"]))
        axes[row, 0].plot(r, ew["radial"] / (ew["radial"].sum()+1e-12), label="With", color="red", linewidth=1.5)
        axes[row, 0].plot(r, eo["radial"] / (eo["radial"].sum()+1e-12), label="Without", color="blue", linewidth=1.5)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(f"L{li} ({s[0]},{s[1]},{s[2]}) - With vs Without")
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(r, ed["radial"] / (ed["radial"].sum()+1e-12), label="Delta", color="green", linewidth=1.5)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_title(f"L{li} - Delta radial power")
        axes[row, 1].grid(True, alpha=0.3)

        axes[row, 2].plot(r, ec["radial"] / (ec["radial"].sum()+1e-12), label="Cross", color="purple", linewidth=1.5)
        axes[row, 2].set_yscale("log")
        axes[row, 2].set_title(f"L{li} - Cross-spectrum (interference)")
        axes[row, 2].grid(True, alpha=0.3)

        bands = ["LF", "MF", "HF"]
        x_b = np.arange(3)
        axes[row, 3].bar(x_b - 0.3, [ew["lf"], ew["mf"], ew["hf"]], 0.2, label="With", color="red", alpha=0.7)
        axes[row, 3].bar(x_b - 0.1, [eo["lf"], eo["mf"], eo["hf"]], 0.2, label="Without", color="blue", alpha=0.7)
        axes[row, 3].bar(x_b + 0.1, [ed["lf"], ed["mf"], ed["hf"]], 0.2, label="Delta", color="green", alpha=0.7)
        axes[row, 3].bar(x_b + 0.3, [ec["lf"], ec["mf"], ec["hf"]], 0.2, label="Cross", color="purple", alpha=0.7)
        axes[row, 3].set_xticks(x_b); axes[row, 3].set_xticklabels(bands, fontsize=8)
        axes[row, 3].set_title(f"L{li} - Band ratios")
        axes[row, 3].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "embedding_fft_per_layer.png"), dpi=150)
    plt.close()

    # --- Plot 3: Cross-layer summary ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Cross-Layer: Image vs Embedding Frequency Analysis", fontsize=16)

    ls = sorted(emb_fft["with"].keys())
    x = range(len(ls))

    # LF: with vs without per layer
    lf_w = [emb_fft["with"][l]["lf"] for l in ls]
    lf_wo = [emb_fft["without"][l]["lf"] for l in ls]
    axes[0,0].plot(x, lf_w, "o-", label="Embedding LF (with human)", color="red", linewidth=2)
    axes[0,0].plot(x, lf_wo, "s--", label="Embedding LF (without human)", color="blue", linewidth=2)
    axes[0,0].axhline(img_fft["with"]["gray"]["lf"], color="red", alpha=0.3, linestyle=":", label="Pixel LF (with)")
    axes[0,0].axhline(img_fft["without"]["gray"]["lf"], color="blue", alpha=0.3, linestyle=":", label="Pixel LF (without)")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,0].set_title("LF power: image (dotted) vs embeddings (solid) per layer")
    axes[0,0].legend()

    # HF: with vs without per layer
    hf_w = [emb_fft["with"][l]["hf"] for l in ls]
    hf_wo = [emb_fft["without"][l]["hf"] for l in ls]
    axes[0,1].plot(x, hf_w, "o-", label="Embedding HF (with human)", color="darkred", linewidth=2)
    axes[0,1].plot(x, hf_wo, "s--", label="Embedding HF (without human)", color="darkblue", linewidth=2)
    axes[0,1].axhline(img_fft["with"]["gray"]["hf"], color="darkred", alpha=0.3, linestyle=":", label="Pixel HF (with)")
    axes[0,1].axhline(img_fft["without"]["gray"]["hf"], color="darkblue", alpha=0.3, linestyle=":", label="Pixel HF (without)")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,1].set_title("HF power: image (dotted) vs embeddings (solid) per layer")
    axes[0,1].legend()

    # Delta LF and HF per layer
    d_lf = [emb_delta[l]["lf"] for l in ls]
    d_hf = [emb_delta[l]["hf"] for l in ls]
    axes[1,0].bar(x, d_lf, alpha=0.7, label="Delta LF", color="green")
    axes[1,0].bar(x, d_hf, alpha=0.7, label="Delta HF", color="red")
    axes[1,0].axhline(img_delta_fft["gray"]["lf"], color="green", linestyle=":", alpha=0.5, label="Pixel delta LF")
    axes[1,0].axhline(img_delta_fft["gray"]["hf"], color="red", linestyle=":", alpha=0.5, label="Pixel delta HF")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[1,0].set_title("Delta band ratios: pixel (dotted) vs embedding (bars) per layer")
    axes[1,0].legend()

    # Cross-spectrum LF and HF per layer
    c_lf = [emb_cross[l]["lf"] for l in ls]
    c_hf = [emb_cross[l]["hf"] for l in ls]
    axes[1,1].bar(x, c_lf, alpha=0.7, label="Cross LF", color="purple")
    axes[1,1].bar(x, c_hf, alpha=0.7, label="Cross HF", color="orange")
    axes[1,1].axhline(img_cross["gray"]["lf"], color="purple", linestyle=":", alpha=0.5, label="Pixel cross LF")
    axes[1,1].axhline(img_cross["gray"]["hf"], color="orange", linestyle=":", alpha=0.5, label="Pixel cross HF")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[1,1].set_title("Cross-spectrum band ratios: pixel (dotted) vs embedding (bars)")
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_layer_interference.png"), dpi=150)
    plt.close()

    # --- Plot 4: 2D spectra side by side for key layers ---
    for li in [0, 54, 62, 105]:
        if li not in emb_fft["with"]: continue
        ew = emb_fft["with"][li]
        eo = emb_fft["without"][li]
        ed = emb_delta[li]
        ec = emb_cross[li]
        s = caps_w[li].squeeze(0).cpu().numpy().shape

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(f"Layer {li} ({s[0]},{s[1]},{s[2]}) - 2D Spectra", fontsize=14)

        im = axes[0].imshow(np.log1p(ew["mean_power"]), cmap="inferno", aspect="auto")
        axes[0].set_title("WITH human (log)")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        im = axes[1].imshow(np.log1p(eo["mean_power"]), cmap="inferno", aspect="auto")
        axes[1].set_title("WITHOUT human (log)")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        im = axes[2].imshow(np.log1p(ed["mean_power"]), cmap="inferno", aspect="auto")
        axes[2].set_title("DELTA (log)")
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        im = axes[3].imshow(np.log1p(ec["cross_power"]), cmap="hot", aspect="auto")
        axes[3].set_title("CROSS-spectrum (log)")
        plt.colorbar(im, ax=axes[3], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"2d_spectra_L{li:03d}.png"), dpi=150)
        plt.close()

    # ============================================================
    # SAVE DATA
    # ============================================================
    print("\nSaving data...")

    json_data = {
        "raw_image_fft": {
            "with": {cn: {"lf": float(img_fft["with"][cn]["lf"]), "mf": float(img_fft["with"][cn]["mf"]), "hf": float(img_fft["with"][cn]["hf"])} for cn in ["R","G","B","gray"]},
            "without": {cn: {"lf": float(img_fft["without"][cn]["lf"]), "mf": float(img_fft["without"][cn]["mf"]), "hf": float(img_fft["without"][cn]["hf"])} for cn in ["R","G","B","gray"]},
            "delta": {cn: {"lf": float(img_delta_fft[cn]["lf"]), "mf": float(img_delta_fft[cn]["mf"]), "hf": float(img_delta_fft[cn]["hf"])} for cn in ["R","G","B","gray"]},
            "cross": {cn: {"lf": float(img_cross[cn]["lf"]), "mf": float(img_cross[cn]["mf"]), "hf": float(img_cross[cn]["hf"]), "mean_coherence": img_cross[cn]["mean_coherence"]} for cn in ["R","G","B","gray"]},
        },
        "embedding_fft": {
            "with": {str(li): {"shape": list(caps_w[li].squeeze(0).shape), "lf": float(emb_fft["with"][li]["lf"]), "mf": float(emb_fft["with"][li]["mf"]), "hf": float(emb_fft["with"][li]["hf"]), "top_channels": emb_fft["with"][li]["top_channels"]} for li in emb_fft["with"]},
            "without": {str(li): {"shape": list(caps_wo[li].squeeze(0).shape), "lf": float(emb_fft["without"][li]["lf"]), "mf": float(emb_fft["without"][li]["mf"]), "hf": float(emb_fft["without"][li]["hf"]), "top_channels": emb_fft["without"][li]["top_channels"]} for li in emb_fft["without"]},
            "delta": {str(li): {"lf": float(emb_delta[li]["lf"]), "mf": float(emb_delta[li]["mf"]), "hf": float(emb_delta[li]["hf"]), "top_channels": emb_delta[li]["top_channels"]} for li in emb_delta},
            "cross": {str(li): {"lf": float(emb_cross[li]["lf"]), "mf": float(emb_cross[li]["mf"]), "hf": float(emb_cross[li]["hf"])} for li in emb_cross},
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "interference_4way.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  {json_path}")

    # CSV
    csv_rows = []
    for li in sorted(emb_fft["with"].keys()):
        ew = emb_fft["with"][li]
        eo = emb_fft["without"][li]
        ed = emb_delta[li]
        ec = emb_cross[li]
        s = caps_w[li].squeeze(0).cpu().numpy().shape
        csv_rows.append({
            "layer": li, "C": s[0], "Hf": s[1], "Wf": s[2],
            "emb_lf_with": ew["lf"], "emb_mf_with": ew["mf"], "emb_hf_with": ew["hf"],
            "emb_lf_without": eo["lf"], "emb_mf_without": eo["mf"], "emb_hf_without": eo["hf"],
            "emb_lf_delta": ed["lf"], "emb_mf_delta": ed["mf"], "emb_hf_delta": ed["hf"],
            "emb_lf_cross": ec["lf"], "emb_mf_cross": ec["mf"], "emb_hf_cross": ec["hf"],
            "pix_lf_with": img_fft["with"]["gray"]["lf"], "pix_hf_with": img_fft["with"]["gray"]["hf"],
            "pix_lf_without": img_fft["without"]["gray"]["lf"], "pix_hf_without": img_fft["without"]["gray"]["hf"],
            "pix_lf_delta": img_delta_fft["gray"]["lf"], "pix_hf_delta": img_delta_fft["gray"]["hf"],
        })
    csv_path = os.path.join(OUTPUT_DIR, "interference_4way.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader(); w.writerows(csv_rows)
    print(f"  {csv_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY: 4-Way FFT Analysis")
    print("="*70)

    print("\n1. RAW PIXEL FFT (grayscale):")
    gw = img_fft["with"]["gray"]
    go = img_fft["without"]["gray"]
    gd = img_delta_fft["gray"]
    gc = img_cross["gray"]
    print(f"   With human:    LF={gw['lf']:.4f} MF={gw['mf']:.4f} HF={gw['hf']:.4f}")
    print(f"   Without human: LF={go['lf']:.4f} MF={go['mf']:.4f} HF={go['hf']:.4f}")
    print(f"   Delta:         LF={gd['lf']:.4f} MF={gd['mf']:.4f} HF={gd['hf']:.4f}")
    print(f"   Cross-spectrum:LF={gc['lf']:.4f} MF={gc['mf']:.4f} HF={gc['hf']:.4f}  coherence={gc['mean_coherence']:.4f}")

    print("\n2. EMBEDDING FFT per layer:")
    print(f"   {'Layer':>6}  {'Shape':>14}  "
          f"{'With LF/HF':>14}  {'Without LF/HF':>16}  {'Delta LF/HF':>14}  {'Cross LF/HF':>14}")
    print("   " + "-"*95)
    for li in sorted(emb_fft["with"].keys()):
        ew = emb_fft["with"][li]
        eo = emb_fft["without"][li]
        ed = emb_delta[li]
        ec = emb_cross[li]
        s = caps_w[li].squeeze(0).cpu().numpy().shape
        print(f"   {li:6d}  ({s[0]:4d},{s[1]:3d},{s[2]:3d})  "
              f"{ew['lf']:.4f}/{ew['hf']:.4f}    "
              f"{eo['lf']:.4f}/{eo['hf']:.4f}     "
              f"{ed['lf']:.4f}/{ed['hf']:.4f}    "
              f"{ec['lf']:.4f}/{ec['hf']:.4f}")

    print(f"\nOutputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
