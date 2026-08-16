"""
Constructive Interference Downwards + HF Propagation Test.

Two experiments:

EXPERIMENT 1: Constructive interference downward
  - Inject sinusoidal pixel oscillations at various (kx, ky) frequencies and phases
    into the "with human" image
  - Forward pass each modified image
  - Measure how far the feature maps move FROM "with human" TOWARD "without human"
  - Score = ||features(modified) - features(with)|| / ||features(without) - features(with)||
  - Score > 0 means moving toward no-person; score = 1.0 means fully reached no-person state
  - Test LF, MF, HF oscillations at multiple amplitudes and phases

EXPERIMENT 2: HF propagation through the network
  - Inject pure HF sinusoidal patterns at the pixel level
  - Track how much HF survives to each layer's feature maps
  - Compare to LF injection: does HF actually attenuate more than LF?
  - This answers: "does injected HF matter more or does it get killed?"

  - Also test: does adding HF to the human image suppress detection more than LF?
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\constructive"
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

def make_sinusoid(H, W, kx, ky, phase_deg, amplitude):
    """Create a 2D sinusoidal pattern: amplitude * cos(2*pi*(kx*x/W + ky*y/H) + phase)"""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Normalize frequencies to [0, 1] range
    fx = kx / W
    fy = ky / H
    phase_rad = np.radians(phase_deg)
    pattern = amplitude * np.cos(2 * np.pi * (fx * x + fy * y) + phase_rad)
    return pattern.astype(np.float32)

def add_pattern_to_image(arr, pattern):
    """Add a 2D pattern to all 3 channels of an HWC image, clipped to [0, 1]."""
    out = arr.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] + pattern, 0, 1)
    out[:, :, 1] = np.clip(out[:, :, 1] + pattern, 0, 1)
    out[:, :, 2] = np.clip(out[:, :, 2] + pattern, 0, 1)
    return out

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

def fft_feature_maps(feat):
    """Average 2D FFT power across channels, return band ratios."""
    C, H, W = feat.shape
    specs = np.zeros((H, W), dtype=np.float64)
    for c in range(C):
        fft = np.fft.fft2(feat[c])
        specs += np.abs(np.fft.fftshift(fft)) ** 2
    specs /= C
    rad, rmax = radial_profile(specs, H, W)
    lf, mf, hf = band_ratios(rad, rmax)
    return lf, mf, hf, specs


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("Constructive Interference Downwards + HF Propagation Test")
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

    print("Baseline forward passes...")
    caps_w = forward_capture(model, iw)
    caps_wo = forward_capture(model, iwo)

    # Baseline: distance from with->without at each layer
    baseline_dist = {}
    for li in LAYERS:
        if li not in caps_w or li not in caps_wo: continue
        fw = caps_w[li].squeeze(0)
        fo = caps_wo[li].squeeze(0)
        baseline_dist[li] = (fo - fw).norm().item()

    # ============================================================
    # EXPERIMENT 1: Constructive interference downward
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Sinusoidal pixel oscillation -> push toward no-person")
    print("="*70)

    # Test frequencies spanning LF, MF, HF
    # For 416x416: LF ~ k=1-52, MF ~ k=53-208, HF ~ k=209-416
    # But we use 2D (kx, ky) so radial frequency matters
    test_freqs = [
        # LF
        ("LF_k1",    1,  0, "LF"),
        ("LF_k2",    2,  0, "LF"),
        ("LF_k5",    5,  0, "LF"),
        ("LF_k10",  10,  0, "LF"),
        ("LF_k1d1",  1,  1, "LF"),
        ("LF_k5d5",  5,  5, "LF"),
        # MF
        ("MF_k30",  30,  0, "MF"),
        ("MF_k50",  50,  0, "MF"),
        ("MF_k30d30", 30, 30, "MF"),
        ("MF_k50d50", 50, 50, "MF"),
        ("MF_k80",  80,  0, "MF"),
        # HF
        ("HF_k100", 100,  0, "HF"),
        ("HF_k150", 150,  0, "HF"),
        ("HF_k100d100", 100, 100, "HF"),
        ("HF_k150d150", 150, 150, "HF"),
        ("HF_k200", 200,  0, "HF"),
        ("HF_k200d200", 200, 200, "HF"),
    ]

    # Test phases: 0, 90, 180, 270 (which phase pushes hardest toward no-person?)
    test_phases = [0, 45, 90, 135, 180, 225, 270, 315]

    # Test amplitudes
    test_amplitudes = [0.05, 0.10, 0.20, 0.30]

    # Use amplitude 0.20 for the phase sweep (moderate)
    amp_sweep = 0.20

    # Results storage
    exp1_results = {}  # {freq_name: {phase: {layer: suppression_score}}}
    exp1_csv = []

    print(f"\nTesting {len(test_freqs)} frequencies x {len(test_phases)} phases x amp={amp_sweep}")
    print(f"{'Freq':>16} {'Phase':>6} {'Band':>4}  ", end="")
    print(f"{'L0':>6} {'L12':>6} {'L54':>6} {'L62':>6} {'L75':>6} {'L105':>6}  {'Avg':>6}")

    for fname, kx, ky, band in test_freqs:
        exp1_results[fname] = {"kx": kx, "ky": ky, "band": band, "phases": {}}

        for phase in test_phases:
            pattern = make_sinusoid(H, W, kx, ky, phase, amp_sweep)
            arr_mod = add_pattern_to_image(arr_w, pattern)
            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
            caps_mod = forward_capture(model, tensor_mod)

            phase_scores = {}
            for li in LAYERS:
                if li not in caps_w or li not in caps_wo or li not in caps_mod: continue
                fm = caps_mod[li].squeeze(0)
                fw = caps_w[li].squeeze(0)
                fo = caps_wo[li].squeeze(0)

                # How far did modified move from "with"?
                dist_mod_from_with = (fm - fw).norm().item()
                # How far is "without" from "with"?
                dist_wo_from_with = baseline_dist[li]

                # Suppression score: how much of the with->without distance did we cover?
                # Also check direction: is modified moving TOWARD without?
                # Direction = dot product of (mod - with) and (without - with)
                delta_mod = (fm - fw).flatten()
                delta_wo = (fo - fw).flatten()
                if dist_wo_from_with > 0:
                    direction = torch.dot(delta_mod, delta_wo).item() / (dist_wo_from_with * dist_mod_from_with + 1e-12)
                else:
                    direction = 0.0

                # Suppression score: positive = moving toward no-person
                # = fraction of with->without distance covered in the right direction
                score = direction * dist_mod_from_with / (dist_wo_from_with + 1e-12)
                phase_scores[li] = {
                    "score": float(score),
                    "direction": float(direction),
                    "dist_from_with": float(dist_mod_from_with),
                    "dist_with_to_without": float(dist_wo_from_with),
                }

            exp1_results[fname]["phases"][phase] = phase_scores

            # Print key layers
            key_layers = [0, 12, 54, 62, 75, 105]
            scores = [phase_scores.get(l, {}).get("score", 0) for l in key_layers]
            avg_score = np.mean([s for s in scores if s != 0]) if any(s != 0 for s in scores) else 0
            print(f"{fname:>16} {phase:6d} {band:>4}  ", end="")
            for s in scores:
                print(f"{s:6.3f}", end=" ")
            print(f" {avg_score:6.3f}")

            exp1_csv.append({
                "freq": fname, "kx": kx, "ky": ky, "band": band,
                "phase": phase, "amplitude": amp_sweep,
                "avg_score": avg_score,
                **{f"L{l}_score": phase_scores.get(l, {}).get("score", 0) for l in key_layers},
                **{f"L{l}_dir": phase_scores.get(l, {}).get("direction", 0) for l in key_layers},
            })

    # Find best (freq, phase) for suppression
    print("\n--- BEST SUPPRESSION COMBOS ---")
    all_combos = []
    for fname, fr in exp1_results.items():
        for phase, scores in fr["phases"].items():
            avg = np.mean([s["score"] for s in scores.values()])
            all_combos.append((avg, fname, phase, fr["band"], fr["kx"], fr["ky"]))
    all_combos.sort(reverse=True)
    for i, (avg, fname, phase, band, kx, ky) in enumerate(all_combos[:15]):
        print(f"  #{i+1} {fname:>16} phase={phase:3d}deg {band} avg_score={avg:.4f}")

    # ============================================================
    # EXPERIMENT 2: HF propagation through the network
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Does injected HF survive to deep layers?")
    print("="*70)

    # Inject pure LF, MF, HF at same amplitude and track through layers
    propagation_freqs = [
        ("LF_k2",    2,   0, "LF"),
        ("LF_k5",    5,   0, "LF"),
        ("MF_k50",  50,   0, "MF"),
        ("MF_k100", 100,  0, "MF"),
        ("HF_k200", 200,   0, "HF"),
        ("HF_k200d200", 200, 200, "HF"),
    ]
    amp_prop = 0.20

    # Baseline: feature map band ratios without any injection
    print("\nBaseline band ratios (no injection):")
    baseline_bands = {}
    for li in LAYERS:
        if li not in caps_w: continue
        fw = caps_w[li].squeeze(0).cpu().numpy()
        lf, mf, hf, _ = fft_feature_maps(fw)
        baseline_bands[li] = {"lf": lf, "mf": mf, "hf": hf}

    prop_results = {}
    prop_csv = []

    for fname, kx, ky, band in propagation_freqs:
        print(f"\n  Injecting {fname} (kx={kx}, ky={ky}, amp={amp_prop}):")
        pattern = make_sinusoid(H, W, kx, ky, 0, amp_prop)  # phase=0
        arr_mod = add_pattern_to_image(arr_w, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        caps_mod = forward_capture(model, tensor_mod)

        prop_results[fname] = {"band": band, "kx": kx, "ky": ky, "layers": {}}

        for li in LAYERS:
            if li not in caps_mod or li not in caps_w: continue
            fm = caps_mod[li].squeeze(0).cpu().numpy()
            fw = caps_w[li].squeeze(0).cpu().numpy()

            # Band ratios of modified features
            lf_m, mf_m, hf_m, specs_m = fft_feature_maps(fm)
            # Band ratios of original features
            lf_o, mf_o, hf_o, specs_o = fft_feature_maps(fw)

            # Delta = modified - original (what the injection introduced)
            delta = fm - fw
            lf_d, mf_d, hf_d, specs_d = fft_feature_maps(delta)

            # How much of the injected frequency survived?
            # Measure: HF power in delta / total power in delta
            # If HF survives, hf_d should be high; if it gets killed, lf_d dominates

            # Also measure: where did the injected energy end up?
            # If we injected HF (k=200), did it stay HF or get converted to LF?

            prop_results[fname]["layers"][li] = {
                "orig_bands": {"lf": lf_o, "mf": mf_o, "hf": hf_o},
                "mod_bands": {"lf": lf_m, "mf": mf_m, "hf": hf_m},
                "delta_bands": {"lf": lf_d, "mf": mf_d, "hf": hf_d},
            }

            prop_csv.append({
                "freq": fname, "band": band, "kx": kx, "ky": ky,
                "layer": li,
                "orig_lf": lf_o, "orig_mf": mf_o, "orig_hf": hf_o,
                "mod_lf": lf_m, "mod_mf": mf_m, "mod_hf": hf_m,
                "delta_lf": lf_d, "delta_mf": mf_d, "delta_hf": hf_d,
            })

            print(f"    L{li:3d}: delta bands LF={lf_d:.4f} MF={mf_d:.4f} HF={hf_d:.4f}  "
                  f"(orig: LF={lf_o:.4f} HF={hf_o:.4f}  mod: LF={lf_m:.4f} HF={hf_m:.4f})")

    # ============================================================
    # AMPLITUDE SWEEP: does more amplitude = more suppression?
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Amplitude sweep for best frequency")
    print("="*70)

    # Use the best frequency from exp1 (we'll use a few candidates)
    amp_test_freqs = [
        ("LF_k5",    5,   0, "LF"),
        ("MF_k50",  50,   0, "MF"),
        ("HF_k200", 200,   0, "HF"),
    ]
    amp_test_phases = [0, 180]
    amp_test_amps = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    amp_results = {}
    amp_csv = []

    for fname, kx, ky, band in amp_test_freqs:
        amp_results[fname] = {}
        for phase in amp_test_phases:
            amp_results[fname][phase] = {}
            for amp in amp_test_amps:
                pattern = make_sinusoid(H, W, kx, ky, phase, amp)
                arr_mod = add_pattern_to_image(arr_w, pattern)
                tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
                caps_mod = forward_capture(model, tensor_mod)

                scores = {}
                for li in LAYERS:
                    if li not in caps_w or li not in caps_wo or li not in caps_mod: continue
                    fm = caps_mod[li].squeeze(0)
                    fw = caps_w[li].squeeze(0)
                    fo = caps_wo[li].squeeze(0)
                    dist_mod = (fm - fw).norm().item()
                    dist_wo = baseline_dist[li]
                    delta_mod = (fm - fw).flatten()
                    delta_wo = (fo - fw).flatten()
                    if dist_wo > 0 and dist_mod > 0:
                        direction = torch.dot(delta_mod, delta_wo).item() / (dist_wo * dist_mod + 1e-12)
                    else:
                        direction = 0
                    score = direction * dist_mod / (dist_wo + 1e-12)
                    scores[li] = float(score)

                avg_score = np.mean(list(scores.values()))
                amp_results[fname][phase][amp] = {"avg_score": avg_score, "scores": scores}
                amp_csv.append({
                    "freq": fname, "phase": phase, "amplitude": amp,
                    "avg_score": avg_score,
                    **{f"L{l}_score": scores.get(l, 0) for l in [0, 54, 62, 75, 105]},
                })
                print(f"  {fname:>12} phase={phase:3d} amp={amp:.2f}: avg_suppression={avg_score:.4f}")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\nGenerating plots...")

    # --- Plot 1: Suppression heatmap (freq x phase) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Constructive Interference: Sinusoidal Oscillation Suppression Score\n"
                 "(higher = more suppression of person signal, amp=0.20)", fontsize=14)

    key_plot_layers = [0, 54, 62, 75, 105]
    for idx, li in enumerate(key_plot_layers):
        ax = axes[idx // 3][idx % 3]
        freq_labels = [f[0] for f in test_freqs]
        phase_labels = test_phases
        matrix = np.zeros((len(freq_labels), len(phase_labels)))
        for fi, (fname, _, _, _) in enumerate(test_freqs):
            for pi, phase in enumerate(phase_labels):
                if phase in exp1_results[fname]["phases"]:
                    matrix[fi, pi] = exp1_results[fname]["phases"][phase].get(li, {}).get("score", 0)
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(phase_labels)))
        ax.set_xticklabels([str(p) for p in phase_labels], fontsize=7)
        ax.set_yticks(range(len(freq_labels)))
        ax.set_yticklabels(freq_labels, fontsize=6)
        ax.set_xlabel("Phase (degrees)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Layer {li}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 6th subplot: average across layers
    ax = axes[1][2]
    matrix_avg = np.zeros((len(freq_labels), len(phase_labels)))
    for fi, (fname, _, _, _) in enumerate(test_freqs):
        for pi, phase in enumerate(phase_labels):
            if phase in exp1_results[fname]["phases"]:
                scores = [exp1_results[fname]["phases"][phase].get(l, {}).get("score", 0)
                          for l in key_plot_layers]
                matrix_avg[fi, pi] = np.mean(scores)
    im = ax.imshow(matrix_avg, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(phase_labels)))
    ax.set_xticklabels([str(p) for p in phase_labels], fontsize=7)
    ax.set_yticks(range(len(freq_labels)))
    ax.set_yticklabels(freq_labels, fontsize=6)
    ax.set_xlabel("Phase (degrees)")
    ax.set_title("Average across key layers")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "suppression_heatmap.png"), dpi=150)
    plt.close()

    # --- Plot 2: HF propagation through layers ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Frequency Propagation: Where Does Injected Energy End Up?\n"
                 "(Delta band ratios = what the injection produced at each layer)", fontsize=14)

    for idx, fname in enumerate(prop_results.keys()):
        ax = axes[idx // 3][idx % 3]
        ls = sorted(prop_results[fname]["layers"].keys())
        lf_d = [prop_results[fname]["layers"][l]["delta_bands"]["lf"] for l in ls]
        mf_d = [prop_results[fname]["layers"][l]["delta_bands"]["mf"] for l in ls]
        hf_d = [prop_results[fname]["layers"][l]["delta_bands"]["hf"] for l in ls]
        x = range(len(ls))
        ax.bar(x, lf_d, alpha=0.7, label="Delta LF", color="blue")
        ax.bar(x, mf_d, alpha=0.7, label="Delta MF", color="orange", bottom=lf_d)
        ax.bar(x, hf_d, alpha=0.7, label="Delta HF", color="red",
               bottom=[a+b for a,b in zip(lf_d, mf_d)])
        ax.set_xticks(x); ax.set_xticklabels([str(l) for l in ls], fontsize=7, rotation=45)
        ax.set_title(f"Injected {fname} ({prop_results[fname]['band']})")
        ax.set_ylabel("Band ratio of delta")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hf_propagation.png"), dpi=150)
    plt.close()

    # --- Plot 3: Amplitude sweep ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Amplitude Sweep: More Oscillation = More Suppression?", fontsize=14)
    for idx, fname in enumerate(amp_test_freqs):
        ax = axes[idx]
        fname_str = fname[0]
        for phase in amp_test_phases:
            amps = sorted(amp_results[fname_str][phase].keys())
            avgs = [amp_results[fname_str][phase][a]["avg_score"] for a in amps]
            ax.plot(amps, avgs, "o-", label=f"phase={phase}", linewidth=2)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Avg suppression score")
        ax.set_title(f"{fname_str} ({fname[3]})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "amplitude_sweep.png"), dpi=150)
    plt.close()

    # --- Plot 4: Best suppression per layer ---
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Best Suppression Score per Layer (across all freq/phase)", fontsize=14)
    ls = sorted(exp1_results[list(exp1_results.keys())[0]]["phases"][0].keys())
    best_per_layer = []
    best_freq_per_layer = []
    for li in ls:
        best_score = -999
        best_name = ""
        for fname, fr in exp1_results.items():
            for phase, scores in fr["phases"].items():
                s = scores.get(li, {}).get("score", 0)
                if s > best_score:
                    best_score = s
                    best_name = f"{fname}_p{phase}"
        best_per_layer.append(best_score)
        best_freq_per_layer.append(best_name)
    x = range(len(ls))
    ax.bar(x, best_per_layer, color=["green" if s > 0 else "red" for s in best_per_layer], alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([str(l) for l in ls], fontsize=8)
    ax.set_ylabel("Best suppression score")
    ax.set_xlabel("Layer")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    for i, (s, n) in enumerate(zip(best_per_layer, best_freq_per_layer)):
        ax.annotate(f"{s:.3f}", (i, s), fontsize=6, ha="center",
                    va="bottom" if s > 0 else "top")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_suppression_per_layer.png"), dpi=150)
    plt.close()

    # ============================================================
    # SAVE DATA
    # ============================================================
    print("\nSaving data...")

    # Exp1 JSON
    exp1_json = {}
    for fname, fr in exp1_results.items():
        exp1_json[fname] = {
            "kx": fr["kx"], "ky": fr["ky"], "band": fr["band"],
            "phases": {
                str(p): {str(l): s for l, s in scores.items()}
                for p, scores in fr["phases"].items()
            }
        }

    # Exp2 JSON
    exp2_json = {}
    for fname, fr in prop_results.items():
        exp2_json[fname] = {
            "band": fr["band"], "kx": fr["kx"], "ky": fr["ky"],
            "layers": {str(l): v for l, v in fr["layers"].items()}
        }

    # Exp3 JSON
    exp3_json = {}
    for fname_str, phases in amp_results.items():
        exp3_json[fname_str] = {
            str(p): {str(a): {"avg_score": v["avg_score"], "scores": {str(k): val for k, val in v["scores"].items()}}
                      for a, v in amps_d.items()}
            for p, amps_d in phases.items()
        }

    all_json = {
        "experiment_1_suppression": exp1_json,
        "experiment_2_propagation": exp2_json,
        "experiment_3_amplitude": exp3_json,
    }
    json_path = os.path.join(OUTPUT_DIR, "constructive_interference.json")
    with open(json_path, "w") as f:
        json.dump(all_json, f, indent=2)
    print(f"  {json_path}")

    # CSVs
    with open(os.path.join(OUTPUT_DIR, "exp1_suppression.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=exp1_csv[0].keys())
        w.writeheader(); w.writerows(exp1_csv)
    with open(os.path.join(OUTPUT_DIR, "exp2_propagation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prop_csv[0].keys())
        w.writeheader(); w.writerows(prop_csv)
    with open(os.path.join(OUTPUT_DIR, "exp3_amplitude.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=amp_csv[0].keys())
        w.writeheader(); w.writerows(amp_csv)
    print(f"  CSVs saved")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n1. BEST SUPPRESSION (constructive interference downward):")
    print("   Top-5 (freq, phase) combos that push features toward no-person:")
    for i, (avg, fname, phase, band, kx, ky) in enumerate(all_combos[:5]):
        print(f"   #{i+1} {fname:>16} phase={phase:3d}deg {band}  avg_score={avg:.4f}")

    print("\n2. HF PROPAGATION (does injected HF survive?):")
    for fname in prop_results:
        fr = prop_results[fname]
        # Compare delta HF at layer 0 vs layer 62
        l0 = fr["layers"].get(0, {}).get("delta_bands", {}).get("hf", 0)
        l62 = fr["layers"].get(62, {}).get("delta_bands", {}).get("hf", 0)
        l75 = fr["layers"].get(75, {}).get("delta_bands", {}).get("hf", 0)
        print(f"   {fname:>16}: delta HF at L0={l0:.4f} -> L62={l62:.4f} -> L75={l75:.4f}")

    print("\n3. AMPLITUDE SWEEP (does more amplitude help?):")
    for fname_str in amp_results:
        for phase in [0, 180]:
            scores = [(a, amp_results[fname_str][phase][a]["avg_score"]) for a in sorted(amp_results[fname_str][phase].keys())]
            best_amp = max(scores, key=lambda x: x[1])
            print(f"   {fname_str:>12} phase={phase:3d}: best amp={best_amp[0]:.2f} score={best_amp[1]:.4f}")

    print(f"\nOutputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
