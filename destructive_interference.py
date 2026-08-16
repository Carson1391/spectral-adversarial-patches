"""
Destructive Interference Analysis.

For each network layer, identify the exact (kx, ky) frequency bins where the
human signal (delta = with - without) has the most power. Then compute the
phase of each bin and the 180-degree phase-shifted signal that would cancel it.

This tells us: "What frequency content, at what phase, needs to be injected
to destructively interfere with the human signal at each layer?"

Pipeline:
  1. Forward pass both images, capture feature maps
  2. Compute delta = features(with) - features(without) per layer
  3. 2D FFT on delta feature maps (per channel, then aggregate)
  4. Identify top-K frequency bins by power
  5. For each top bin: report (kx, ky), magnitude, phase, and the canceling phase
  6. Compute the destructive interference signal: -delta_fft (180 deg phase shift)
  7. Inverse FFT the canceled signal to see what spatial pattern would cancel the human
  8. Report per-layer: dominant frequencies, required cancel pattern, and bandwidth
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\destructive"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]
TOP_K = 20  # top frequency bins to report per layer

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

def classify_band(kx, ky, H, W):
    """Classify a frequency bin as LF, MF, or HF based on radial distance."""
    cy, cx = H//2, W//2
    r = np.sqrt((kx - cx)**2 + (ky - cy)**2)
    rmax = min(cy, cx)
    r4 = max(1, rmax // 4)
    r2 = max(1, rmax // 2)
    if r <= r4: return "LF"
    elif r <= r2: return "MF"
    else: return "HF"

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("Destructive Interference Analysis")
    print("What frequencies cancel the human signal at each layer?")
    print("="*70)

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("Loading images...")
    iw, arr_w = load_image(IMG_WITH)
    iwo, arr_wo = load_image(IMG_WITHOUT)

    print("Forward passes...")
    caps_w = forward_capture(model, iw)
    caps_wo = forward_capture(model, iwo)

    # Also compute pixel-level delta FFT for reference
    pixel_delta_gray = (arr_w - arr_wo).mean(axis=2)
    fft_pixel = np.fft.fft2(pixel_delta_gray)
    fft_pixel_shift = np.fft.fftshift(fft_pixel)
    power_pixel = np.abs(fft_pixel_shift) ** 2
    H_pix, W_pix = pixel_delta_gray.shape
    rad_pix, rmax_pix = radial_profile(power_pixel, H_pix, W_pix)
    lf_pix, mf_pix, hf_pix = band_ratios(rad_pix, rmax_pix)

    # Pixel-level top frequencies
    print("\n--- PIXEL-LEVEL DELTA FFT ---")
    print(f"  Shape: ({H_pix}, {W_pix})  LF={lf_pix:.4f} MF={mf_pix:.4f} HF={hf_pix:.4f}")
    # Top-K pixel frequency bins
    flat_idx = np.argsort(power_pixel.flatten())[::-1][:TOP_K]
    pixel_top = []
    for fi in flat_idx:
        ky, kx = np.unravel_index(fi, power_pixel.shape)
        mag = np.abs(fft_pixel_shift[ky, kx])
        phase = np.angle(fft_pixel_shift[ky, kx])
        band = classify_band(kx, ky, H_pix, W_pix)
        pixel_top.append({
            "kx": int(kx), "ky": int(ky), "band": band,
            "power": float(power_pixel[ky, kx]),
            "magnitude": float(mag),
            "phase_rad": float(phase),
            "phase_deg": float(np.degrees(phase)),
            "cancel_phase_deg": float(np.degrees(phase) + 180.0),
        })
    for i, pt in enumerate(pixel_top[:10]):
        print(f"  #{i+1} ({pt['kx']:3d},{pt['ky']:3d}) {pt['band']} "
              f"power={pt['power']:.2e} phase={pt['phase_deg']:.1f}deg "
              f"cancel={pt['cancel_phase_deg']:.1f}deg")

    # ============================================================
    # PER-LAYER EMBEDDING ANALYSIS
    # ============================================================
    all_results = {}
    csv_rows = []

    for li in LAYERS:
        if li not in caps_w or li not in caps_wo: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()  # (C, Hf, Wf)
        fo = caps_wo[li].squeeze(0).cpu().numpy()
        C, Hf, Wf = fw.shape
        delta = fw - fo  # (C, Hf, Wf)

        # Per-channel 2D FFT of delta
        # Accumulate power across channels to find dominant bins
        total_power = np.zeros((Hf, Wf), dtype=np.float64)
        # Also store per-channel FFT for top-bin analysis
        channel_ffts = np.zeros((C, Hf, Wf), dtype=np.complex128)

        for c in range(C):
            fft_c = np.fft.fft2(delta[c])
            fft_c_shift = np.fft.fftshift(fft_c)
            channel_ffts[c] = fft_c_shift
            total_power += np.abs(fft_c_shift) ** 2

        # Average power across channels
        avg_power = total_power / C

        # Radial profile and band ratios
        rad, rmax = radial_profile(avg_power, Hf, Wf)
        lf, mf, hf = band_ratios(rad, rmax)

        # Top-K frequency bins by total power
        flat_idx = np.argsort(avg_power.flatten())[::-1][:TOP_K]
        top_bins = []
        for fi in flat_idx:
            ky, kx = np.unravel_index(fi, avg_power.shape)
            power_val = avg_power[ky, kx]
            band = classify_band(kx, ky, Hf, Wf)

            # Find which channels contribute most to this bin
            ch_powers = np.abs(channel_ffts[:, ky, kx]) ** 2
            top_ch = np.argsort(ch_powers)[::-1][:5]

            # Average phase across top channels (weighted by power)
            weights = ch_powers[top_ch]
            phases = np.angle(channel_ffts[top_ch, ky, kx])
            if weights.sum() > 0:
                # Circular mean of phase
                mean_phase = np.angle(np.sum(weights * np.exp(1j * phases)) / weights.sum())
            else:
                mean_phase = 0.0

            top_bins.append({
                "kx": int(kx), "ky": int(ky), "band": band,
                "power": float(power_val),
                "magnitude": float(np.sqrt(power_val)),
                "phase_rad": float(mean_phase),
                "phase_deg": float(np.degrees(mean_phase)),
                "cancel_phase_deg": float(np.degrees(mean_phase) + 180.0),
                "top_channels": top_ch.tolist(),
                "channel_powers": [float(p) for p in ch_powers[top_ch]],
            })

        # Compute the destructive interference signal
        # To cancel the human signal: inject -delta (180 deg phase shift)
        # In frequency domain: multiply delta FFT by -1
        # In spatial domain: -delta
        cancel_signal = -delta  # (C, Hf, Wf)

        # Also compute a frequency-selective cancel: only cancel HF bins
        cancel_hf = np.zeros_like(delta)
        cy_f, cx_f = Hf//2, Wf//2
        r4 = max(1, rmax // 4)
        r2 = max(1, rmax // 2)
        for c in range(C):
            fft_c = np.fft.fft2(delta[c])
            fft_c_shift = np.fft.fftshift(fft_c)
            # Create HF mask
            y, x = np.indices((Hf, Wf))
            r = np.sqrt((y - cy_f)**2 + (x - cx_f)**2)
            hf_mask = r > r2
            # Zero out non-HF, negate HF
            fft_c_shift[~hf_mask] = 0
            fft_c_shift[hf_mask] = -fft_c_shift[hf_mask]
            cancel_hf[c] = np.fft.ifft2(np.fft.ifftshift(fft_c_shift)).real

        # Energy metrics
        delta_energy = float((delta ** 2).sum())
        cancel_energy = float((cancel_signal ** 2).sum())
        hf_delta_energy = float((delta ** 2).sum() * hf)
        hf_cancel_energy = float((cancel_hf ** 2).sum())

        # Per-band power
        lf_power = float(avg_power[:r4].sum() / (avg_power.sum() + 1e-12))
        mf_power = float(avg_power[r4:r2].sum() / (avg_power.sum() + 1e-12))
        hf_power = float(avg_power[r2:].sum() / (avg_power.sum() + 1e-12))

        all_results[li] = {
            "shape": [C, Hf, Wf],
            "bands": {"lf": lf, "mf": mf, "hf": hf},
            "band_power": {"lf": lf_power, "mf": mf_power, "hf": hf_power},
            "delta_energy": delta_energy,
            "hf_delta_energy": hf_delta_energy,
            "top_bins": top_bins,
        }

        csv_rows.append({
            "layer": li, "C": C, "Hf": Hf, "Wf": Wf,
            "lf": lf, "mf": mf, "hf": hf,
            "delta_energy": delta_energy,
            "hf_delta_energy": hf_delta_energy,
            "top1_kx": top_bins[0]["kx"], "top1_ky": top_bins[0]["ky"],
            "top1_band": top_bins[0]["band"], "top1_power": top_bins[0]["power"],
            "top1_phase_deg": top_bins[0]["phase_deg"],
            "top1_cancel_deg": top_bins[0]["cancel_phase_deg"],
        })

        print(f"\n  Layer {li:3d} ({C:4d},{Hf:3d},{Wf:3d}): "
              f"LF={lf:.4f} MF={mf:.4f} HF={hf:.4f}  "
              f"delta_energy={delta_energy:.2e}  hf_energy={hf_delta_energy:.2e}")
        for i, tb in enumerate(top_bins[:5]):
            print(f"    #{i+1} ({tb['kx']:3d},{tb['ky']:3d}) {tb['band']} "
                  f"power={tb['power']:.2e} phase={tb['phase_deg']:.1f}deg "
                  f"cancel={tb['cancel_phase_deg']:.1f}deg "
                  f"ch={tb['top_channels'][:3]}")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\nGenerating plots...")

    # --- Plot 1: Pixel-level destructive interference ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pixel-Level Destructive Interference Analysis", fontsize=16)

    # Delta image
    axes[0,0].imshow(pixel_delta_gray, cmap="gray")
    axes[0,0].set_title("Pixel delta (with - without)")

    # Delta FFT power
    im = axes[0,1].imshow(np.log1p(power_pixel), cmap="inferno", aspect="auto")
    axes[0,1].set_title("Delta FFT power (log)")
    plt.colorbar(im, ax=axes[0,1], fraction=0.046)

    # Top-K frequency bins marked
    axes[0,2].imshow(np.log1p(power_pixel), cmap="inferno", aspect="auto")
    for i, pt in enumerate(pixel_top[:10]):
        axes[0,2].plot(pt["kx"], pt["ky"], "r*", markersize=10)
        axes[0,2].annotate(f"#{i+1}", (pt["kx"], pt["ky"]), color="white", fontsize=7)
    axes[0,2].set_title("Top-10 frequency bins (red stars)")

    # Cancel signal = -delta
    axes[1,0].imshow(-pixel_delta_gray, cmap="gray")
    axes[1,0].set_title("Cancel signal (-delta)")

    # Phase map
    phase_pix = np.angle(fft_pixel_shift)
    im = axes[1,1].imshow(phase_pix, cmap="twilight", aspect="auto")
    axes[1,1].set_title("Phase of delta FFT (radians)")
    plt.colorbar(im, ax=axes[1,1], fraction=0.046)

    # HF-only cancel signal
    fft_pix_hf = np.fft.fftshift(np.fft.fft2(pixel_delta_gray))
    y, x = np.indices(power_pixel.shape)
    r = np.sqrt((y - H_pix//2)**2 + (x - W_pix//2)**2)
    r2_pix = max(1, rmax_pix // 2)
    hf_mask_pix = r > r2_pix
    fft_pix_hf[~hf_mask_pix] = 0
    fft_pix_hf[hf_mask_pix] = -fft_pix_hf[hf_mask_pix]
    cancel_hf_pix = np.fft.ifft2(np.fft.ifftshift(fft_pix_hf)).real
    axes[1,2].imshow(cancel_hf_pix, cmap="RdBu_r")
    axes[1,2].set_title("HF-only cancel signal (what cancels human HF)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pixel_destructive.png"), dpi=150)
    plt.close()

    # --- Plot 2: Per-layer key layers ---
    for li in [0, 12, 54, 62, 75, 105]:
        if li not in all_results: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()
        fo = caps_wo[li].squeeze(0).cpu().numpy()
        delta = fw - fo
        C, Hf, Wf = delta.shape
        res = all_results[li]

        # Average delta across channels for visualization
        delta_avg = delta.mean(axis=0)
        cancel_avg = -delta_avg

        # Average FFT power
        specs = np.zeros((Hf, Wf), dtype=np.float64)
        for c in range(C):
            f2 = np.fft.fft2(delta[c])
            specs += np.abs(np.fft.fftshift(f2)) ** 2
        specs /= C

        # HF-only cancel (average across channels)
        cy_f, cx_f = Hf//2, Wf//2
        y, x = np.indices((Hf, Wf))
        r = np.sqrt((y - cy_f)**2 + (x - cx_f)**2)
        rmax_f = min(cy_f, cx_f)
        r2_f = max(1, rmax_f // 2)
        hf_mask = r > r2_f

        cancel_hf_avg = np.zeros((Hf, Wf))
        for c in range(C):
            fft_c = np.fft.fftshift(np.fft.fft2(delta[c]))
            fft_c[~hf_mask] = 0
            fft_c[hf_mask] = -fft_c[hf_mask]
            cancel_hf_avg += np.fft.ifft2(np.fft.ifftshift(fft_c)).real
        cancel_hf_avg /= C

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Layer {li} ({C},{Hf},{Wf}) - Destructive Interference  "
                     f"LF={res['bands']['lf']:.3f} MF={res['bands']['mf']:.3f} HF={res['bands']['hf']:.3f}",
                     fontsize=14)

        # Delta spatial (avg)
        axes[0,0].imshow(delta_avg, cmap="RdBu_r")
        axes[0,0].set_title("Delta spatial (avg channels)")

        # Delta FFT power
        im = axes[0,1].imshow(np.log1p(specs), cmap="inferno", aspect="auto")
        axes[0,1].set_title("Delta FFT power (log, avg channels)")
        plt.colorbar(im, ax=axes[0,1], fraction=0.046)

        # Top bins marked
        axes[0,2].imshow(np.log1p(specs), cmap="inferno", aspect="auto")
        for i, tb in enumerate(res["top_bins"][:10]):
            axes[0,2].plot(tb["kx"], tb["ky"], "r*", markersize=8)
            axes[0,2].annotate(f"#{i+1}", (tb["kx"], tb["ky"]), color="white", fontsize=6)
        axes[0,2].set_title("Top-10 frequency bins")

        # Full cancel signal
        axes[1,0].imshow(cancel_avg, cmap="RdBu_r")
        axes[1,0].set_title("Full cancel signal (-delta, avg)")

        # HF-only cancel
        axes[1,1].imshow(cancel_hf_avg, cmap="RdBu_r")
        axes[1,1].set_title("HF-only cancel signal (avg)")

        # Phase of top bins
        # Compute average phase map (weighted by power)
        phase_map = np.zeros((Hf, Wf))
        for c in range(C):
            fft_c = np.fft.fftshift(np.fft.fft2(delta[c]))
            phase_map += np.angle(fft_c) * (np.abs(fft_c) ** 2)
        phase_map /= (specs * C + 1e-12)
        im = axes[1,2].imshow(phase_map, cmap="twilight", aspect="auto")
        axes[1,2].set_title("Phase map (power-weighted avg)")
        plt.colorbar(im, ax=axes[1,2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"destructive_L{li:03d}.png"), dpi=150)
        plt.close()

    # --- Plot 3: Cross-layer summary ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Cross-Layer Destructive Interference Summary", fontsize=16)

    ls = sorted(all_results.keys())
    x = range(len(ls))

    # Band ratios per layer
    lf_vals = [all_results[l]["bands"]["lf"] for l in ls]
    mf_vals = [all_results[l]["bands"]["mf"] for l in ls]
    hf_vals = [all_results[l]["bands"]["hf"] for l in ls]
    axes[0,0].bar(x, lf_vals, alpha=0.7, label="LF", color="blue")
    axes[0,0].bar(x, mf_vals, alpha=0.7, label="MF", color="orange", bottom=lf_vals)
    axes[0,0].bar(x, hf_vals, alpha=0.7, label="HF", color="red",
                  bottom=[a+b for a,b in zip(lf_vals, mf_vals)])
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,0].set_title("Delta band ratios per layer (stacked)")
    axes[0,0].legend()

    # HF energy per layer
    hf_energy = [all_results[l]["hf_delta_energy"] for l in ls]
    axes[0,1].bar(x, hf_energy, color="red", alpha=0.7)
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[0,1].set_title("HF delta energy per layer")
    axes[0,1].set_yscale("log")

    # Top bin phase distribution per layer
    for l in ls:
        phases = [tb["phase_deg"] for tb in all_results[l]["top_bins"][:10]]
        axes[1,0].scatter([l]*len(phases), phases, alpha=0.6, s=30)
    axes[1,0].set_xlabel("Layer")
    axes[1,0].set_ylabel("Phase (degrees)")
    axes[1,0].set_title("Top-10 bin phase distribution per layer")
    axes[1,0].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[1,0].axhline(180, color="red", linestyle="--", alpha=0.3, label="cancel line")
    axes[1,0].axhline(-180, color="red", linestyle="--", alpha=0.3)
    axes[1,0].legend()

    # Top bin band classification per layer
    band_counts = {"LF": [], "MF": [], "HF": []}
    for l in ls:
        bins = all_results[l]["top_bins"][:10]
        counts = {"LF": 0, "MF": 0, "HF": 0}
        for b in bins:
            counts[b["band"]] += 1
        for k in band_counts:
            band_counts[k].append(counts[k])
    x_b = np.arange(len(ls))
    axes[1,1].bar(x_b - 0.25, band_counts["LF"], 0.25, label="LF", color="blue", alpha=0.7)
    axes[1,1].bar(x_b, band_counts["MF"], 0.25, label="MF", color="orange", alpha=0.7)
    axes[1,1].bar(x_b + 0.25, band_counts["HF"], 0.25, label="HF", color="red", alpha=0.7)
    axes[1,1].set_xticks(x_b); axes[1,1].set_xticklabels([str(l) for l in ls], fontsize=8)
    axes[1,1].set_title("Top-10 bin band classification per layer")
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "destructive_cross_layer.png"), dpi=150)
    plt.close()

    # ============================================================
    # SAVE DATA
    # ============================================================
    print("\nSaving data...")

    # JSON
    json_data = {
        "pixel_level": {
            "shape": [H_pix, W_pix],
            "bands": {"lf": lf_pix, "mf": mf_pix, "hf": hf_pix},
            "top_bins": pixel_top,
        },
        "embedding_levels": {},
    }
    for li, res in all_results.items():
        json_data["embedding_levels"][str(li)] = {
            "shape": res["shape"],
            "bands": res["bands"],
            "band_power": res["band_power"],
            "delta_energy": res["delta_energy"],
            "hf_delta_energy": res["hf_delta_energy"],
            "top_bins": res["top_bins"],
        }

    json_path = os.path.join(OUTPUT_DIR, "destructive_interference.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  {json_path}")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "destructive_interference.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader(); w.writerows(csv_rows)
    print(f"  {csv_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY: Destructive Interference Analysis")
    print("="*70)

    print("\n1. PIXEL LEVEL:")
    print(f"   Delta bands: LF={lf_pix:.4f} MF={mf_pix:.4f} HF={hf_pix:.4f}")
    print(f"   Top-5 frequency bins:")
    for i, pt in enumerate(pixel_top[:5]):
        print(f"     #{i+1} ({pt['kx']:3d},{pt['ky']:3d}) {pt['band']} "
              f"power={pt['power']:.2e} phase={pt['phase_deg']:.1f}deg "
              f"-> inject at {pt['cancel_phase_deg']:.1f}deg to cancel")

    print("\n2. EMBEDDING LEVEL (top-5 bins per key layer):")
    for li in [0, 54, 62, 75, 105]:
        if li not in all_results: continue
        res = all_results[li]
        print(f"\n   Layer {li} ({res['shape'][0]},{res['shape'][1]},{res['shape'][2]}): "
              f"LF={res['bands']['lf']:.4f} MF={res['bands']['mf']:.4f} HF={res['bands']['hf']:.4f}")
        for i, tb in enumerate(res["top_bins"][:5]):
            print(f"     #{i+1} ({tb['kx']:3d},{tb['ky']:3d}) {tb['band']} "
                  f"power={tb['power']:.2e} phase={tb['phase_deg']:.1f}deg "
                  f"-> cancel at {tb['cancel_phase_deg']:.1f}deg "
                  f"ch={tb['top_channels'][:3]}")

    print("\n3. DESTRUCTIVE INTERFERENCE STRATEGY:")
    print("   To cancel the human signal at a given layer, inject a signal")
    print("   with the same frequency content but 180-degree phase shift.")
    print("   The cancel signal = -delta (full band) or -delta_HF (HF only).")
    print("   Key layers for HF cancellation:")
    for li in [62, 63, 75]:
        if li not in all_results: continue
        res = all_results[li]
        hf_bins = [b for b in res["top_bins"] if b["band"] == "HF"]
        print(f"   Layer {li}: {len(hf_bins)}/{len(res['top_bins'][:10])} top bins are HF, "
              f"HF energy={res['hf_delta_energy']:.2e}")

    print(f"\nOutputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
