"""
Orthogonal Axis + Scalar Resonance Accumulation Test.

EXPERIMENT 1: Orthogonal Axis Attack
  - Previous tests only used horizontal (kx, 0) and diagonal (kx, ky)
  - Never tested the orthogonal axis: vertical (0, ky), anti-diagonal (kx, -ky)
  - The person signal may have a dominant spatial orientation
  - Test all 8 orientations at the same frequency and amplitude to find
    which axis the person signal is most vulnerable to

EXPERIMENT 2: Scalar Resonance Accumulation
  - Apply the same sinusoidal perturbation N times cumulatively
  - Each iteration adds the pattern to the already-perturbed image
  - Measure if suppression grows linearly, super-linearly (resonance), or saturates
  - This tests whether constructive interference self-reinforces like a scalar
  - Test at multiple frequencies and orientations to find which exhibits
    the strongest resonance effect
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\orthogonal_resonance"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size - nw) // 2, (size - nh) // 2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE), arr


def forward_capture(model, x):
    caps = {}
    los = []
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional", "upsample", "maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups", 1))
            gi = int(md.get("group_id", 0))
            x = comb[:, gs * gi:gs * (gi + 1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, x.size(2) if hasattr(x, 'size') else 416)
        if md["type"] == "convolutional":
            caps[i] = x.detach().clone()
        los.append(x)
    return caps


def make_sinusoid(H, W, kx, ky, phase_deg, amplitude):
    # 2D sinusoidal pattern: amplitude * cos(2*pi*(kx*x/W + ky*y/H) + phase)
    # Supports negative ky for anti-diagonal orientations
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fx = kx / W
    fy = ky / H
    phase_rad = np.radians(phase_deg)
    pattern = amplitude * np.cos(2 * np.pi * (fx * x + fy * y) + phase_rad)
    return pattern.astype(np.float32)


def add_pattern_to_image(arr, pattern):
    out = arr.copy()
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + pattern, 0, 1)
    return out


def compute_suppression(caps_mod, caps_w, caps_wo, baseline_dist, layers):
    # Returns per-layer suppression scores and direction cosines
    # score = direction * dist(mod, with) / dist(without, with)
    # positive score = moving toward no-person state
    scores = {}
    for li in layers:
        if li not in caps_w or li not in caps_wo or li not in caps_mod:
            continue
        fm = caps_mod[li].squeeze(0)
        fw = caps_w[li].squeeze(0)
        fo = caps_wo[li].squeeze(0)

        dist_mod_from_with = (fm - fw).norm().item()
        dist_wo_from_with = baseline_dist[li]

        delta_mod = (fm - fw).flatten()
        delta_wo = (fo - fw).flatten()
        if dist_wo_from_with > 0:
            direction = torch.dot(delta_mod, delta_wo).item() / (
                dist_wo_from_with * dist_mod_from_with + 1e-12
            )
        else:
            direction = 0.0

        score = direction * dist_mod_from_with / (dist_wo_from_with + 1e-12)
        scores[li] = {
            "score": float(score),
            "direction": float(direction),
            "dist_from_with": float(dist_mod_from_with),
            "dist_with_to_without": float(dist_wo_from_with),
        }
    return scores


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Orthogonal Axis + Scalar Resonance Accumulation Test")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Loading images...")
    iw, arr_w = load_image(IMG_WITH)
    iwo, arr_wo = load_image(IMG_WITHOUT)
    H, W, _ = arr_w.shape

    print("Baseline forward passes...")
    caps_w = forward_capture(model, iw)
    caps_wo = forward_capture(model, iwo)

    baseline_dist = {}
    for li in LAYERS:
        if li not in caps_w or li not in caps_wo:
            continue
        fw = caps_w[li].squeeze(0)
        fo = caps_wo[li].squeeze(0)
        baseline_dist[li] = (fo - fw).norm().item()

    key_layers = [0, 12, 54, 62, 75, 105]

    # ============================================================
    # EXPERIMENT 1: Orthogonal Axis Attack
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Orthogonal Axis — which orientation suppresses most?")
    print("=" * 70)

    # Test all 8 orientations at the same radial frequency
    # Radial frequency k = sqrt(kx^2 + ky^2) should be equal for fair comparison
    # Use k=200 (HF) since that was the best suppressor
    # Also test k=50 (MF) and k=5 (LF) for comparison
    k_hf = 200
    k_mf = 50
    k_lf = 5

    # 8 orientations: 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 degrees
    orientations_deg = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

    test_configs = []
    for k_val, band_name in [(k_lf, "LF"), (k_mf, "MF"), (k_hf, "HF")]:
        for theta in orientations_deg:
            theta_rad = np.radians(theta)
            kx = int(round(k_val * np.cos(theta_rad)))
            ky = int(round(k_val * np.sin(theta_rad)))
            # Avoid (0,0) — if both round to 0, force at least 1
            if kx == 0 and ky == 0:
                kx = 1
            name = f"{band_name}_k{k_val}_theta{int(theta)}"
            test_configs.append((name, kx, ky, band_name, theta))

    amp = 0.20
    phase = 0  # Phase doesn't matter much for HF per our previous findings

    print(f"\nTesting {len(test_configs)} orientation configs at amp={amp}, phase={phase}")
    print(f"{'Name':>24} {'kx':>5} {'ky':>5} {'theta':>6} {'Band':>4}  ", end="")
    print(f"{'L0':>6} {'L12':>6} {'L54':>6} {'L62':>6} {'L75':>6} {'L105':>6}  {'Avg':>6}")

    exp1_results = {}
    exp1_csv = []

    for name, kx, ky, band, theta in test_configs:
        pattern = make_sinusoid(H, W, kx, ky, phase, amp)
        arr_mod = add_pattern_to_image(arr_w, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        caps_mod = forward_capture(model, tensor_mod)

        scores = compute_suppression(caps_mod, caps_w, caps_wo, baseline_dist, LAYERS)
        exp1_results[name] = {
            "kx": kx, "ky": ky, "band": band, "theta": theta,
            "amp": amp, "phase": phase, "scores": scores,
        }

        layer_scores = [scores.get(l, {}).get("score", 0) for l in key_layers]
        avg_score = float(np.mean([s for s in layer_scores if s != 0])) if any(s != 0 for s in layer_scores) else 0.0

        print(f"{name:>24} {kx:5d} {ky:5d} {theta:6.1f} {band:>4}  ", end="")
        for s in layer_scores:
            print(f"{s:6.3f}", end=" ")
        print(f" {avg_score:6.3f}")

        exp1_csv.append({
            "name": name, "kx": kx, "ky": ky, "theta": theta, "band": band,
            "amp": amp, "phase": phase, "avg_score": avg_score,
            **{f"L{l}_score": scores.get(l, {}).get("score", 0) for l in key_layers},
            **{f"L{l}_dir": scores.get(l, {}).get("direction", 0) for l in key_layers},
        })

    # Rank by avg suppression score
    print("\n--- ORIENTATION RANKING (best suppressors) ---")
    ranked = sorted(exp1_csv, key=lambda x: x["avg_score"], reverse=True)
    for i, row in enumerate(ranked[:15]):
        print(f"  #{i+1} {row['name']:>24} kx={row['kx']:4d} ky={row['ky']:4d} "
              f"theta={row['theta']:5.1f} {row['band']:>3} avg={row['avg_score']:.4f}")

    # Find best orientation per band
    print("\n--- BEST ORIENTATION PER BAND ---")
    for band_name in ["LF", "MF", "HF"]:
        band_rows = [r for r in ranked if r["band"] == band_name]
        if band_rows:
            best = band_rows[0]
            worst = band_rows[-1]
            print(f"  {band_name}: best={best['name']} (theta={best['theta']:.1f}, avg={best['avg_score']:.4f}), "
                  f"worst={worst['name']} (theta={worst['theta']:.1f}, avg={worst['avg_score']:.4f}), "
                  f"ratio={best['avg_score']/(worst['avg_score']+1e-12):.2f}x")

    # Plot: polar chart of suppression vs orientation for each band
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection="polar"))
    for ax_idx, band_name in enumerate(["LF", "MF", "HF"]):
        ax = axes[ax_idx]
        band_rows = [r for r in exp1_csv if r["band"] == band_name]
        thetas = [np.radians(r["theta"]) for r in band_rows]
        scores = [r["avg_score"] for r in band_rows]
        # Close the loop
        thetas.append(thetas[0])
        scores.append(scores[0])
        ax.plot(thetas, scores, "o-", linewidth=2, markersize=6)
        ax.fill(thetas, scores, alpha=0.25)
        ax.set_title(f"{band_name} Suppression vs Orientation\n(k={k_lf if band_name=='LF' else k_mf if band_name=='MF' else k_hf}, amp={amp})", fontsize=11)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(135)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "orientation_polar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved orientation polar plot: {OUTPUT_DIR}/orientation_polar.png")

    # ============================================================
    # EXPERIMENT 2: Scalar Resonance Accumulation
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Scalar Resonance — does suppression compound with iteration?")
    print("=" * 70)

    # Apply the same perturbation N times cumulatively
    # Each iteration: arr_mod = clip(arr_mod + pattern, 0, 1)
    # Measure suppression at each iteration
    # If suppression grows linearly: simple accumulation
    # If super-linearly: resonance / constructive interference
    # If sub-linearly: saturation / clipping effects

    max_iters = 20
    per_iter_amp = 0.05  # Small per-iteration amplitude to avoid immediate clipping

    # Test the best orientations from Experiment 1 plus the original diagonal
    resonance_configs = [
        # Original best from previous tests
        ("HF_k200d200", 200, 200, "HF", 45.0),
        # Horizontal (original)
        ("HF_k200h", 200, 0, "HF", 0.0),
        # Vertical (orthogonal)
        ("HF_k200v", 0, 200, "HF", 90.0),
        # Anti-diagonal
        ("HF_k200ad", 200, -200, "HF", 135.0),
        # Best from exp1 if different
    ]

    # Add the best orientation from exp1
    best_exp1 = ranked[0]
    resonance_configs.append((
        f"BEST_{best_exp1['name']}",
        best_exp1["kx"], best_exp1["ky"], best_exp1["band"], best_exp1["theta"]
    ))

    # Also test MF and LF for comparison
    resonance_configs.append(("MF_k50d50", 50, 50, "MF", 45.0))
    resonance_configs.append(("LF_k5d5", 5, 5, "LF", 45.0))

    print(f"\nMax iterations: {max_iters}, per-iteration amplitude: {per_iter_amp}")
    print(f"Max cumulative amplitude (no clipping): {max_iters * per_iter_amp}")
    print(f"\n{'Name':>24} {'kx':>5} {'ky':>5} {'Band':>4}  ", end="")
    print(f"{'Iter1':>6} {'Iter5':>6} {'Iter10':>6} {'Iter15':>6} {'Iter20':>6}  {'Growth':>8}")

    exp2_results = {}
    exp2_csv = []

    for name, kx, ky, band, theta in resonance_configs:
        pattern = make_sinusoid(H, W, kx, ky, 0, per_iter_amp)
        arr_accum = arr_w.copy()

        iter_scores = []
        iter_details = {}

        for it in range(1, max_iters + 1):
            # Add pattern cumulatively
            arr_accum = add_pattern_to_image(arr_accum, pattern)
            tensor_mod = torch.from_numpy(arr_accum).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            caps_mod = forward_capture(model, tensor_mod)

            scores = compute_suppression(caps_mod, caps_w, caps_wo, baseline_dist, LAYERS)
            layer_scores = [scores.get(l, {}).get("score", 0) for l in key_layers]
            avg_score = float(np.mean([s for s in layer_scores if s != 0])) if any(s != 0 for s in layer_scores) else 0.0

            iter_scores.append(avg_score)
            iter_details[it] = {
                "avg_score": avg_score,
                "cumulative_amp": it * per_iter_amp,
                "per_layer": {l: scores.get(l, {}).get("score", 0) for l in key_layers},
            }

            # Check how many pixels are clipped (saturated at 0 or 1)
            clipped = np.mean((arr_accum <= 0.001) | (arr_accum >= 0.999)) * 100
            iter_details[it]["pct_clipped"] = float(clipped)

        exp2_results[name] = {
            "kx": kx, "ky": ky, "band": band, "theta": theta,
            "per_iter_amp": per_iter_amp, "max_iters": max_iters,
            "iterations": iter_details,
        }

        # Growth rate: ratio of final score to first iteration score
        growth = iter_scores[-1] / (iter_scores[0] + 1e-12) if iter_scores[0] > 1e-8 else 0.0

        # Print key iterations
        i1 = iter_scores[0] if len(iter_scores) > 0 else 0
        i5 = iter_scores[4] if len(iter_scores) > 4 else 0
        i10 = iter_scores[9] if len(iter_scores) > 9 else 0
        i15 = iter_scores[14] if len(iter_scores) > 14 else 0
        i20 = iter_scores[19] if len(iter_scores) > 19 else 0

        print(f"{name:>24} {kx:5d} {ky:5d} {band:>4}  ", end="")
        print(f"{i1:6.3f} {i5:6.3f} {i10:6.3f} {i15:6.3f} {i20:6.3f}  {growth:8.2f}x")

        for it in range(1, max_iters + 1):
            exp2_csv.append({
                "name": name, "kx": kx, "ky": ky, "band": band, "theta": theta,
                "iteration": it, "cumulative_amp": it * per_iter_amp,
                "avg_score": iter_details[it]["avg_score"],
                "pct_clipped": iter_details[it]["pct_clipped"],
                **{f"L{l}_score": iter_details[it]["per_layer"].get(l, 0) for l in key_layers},
            })

    # Analyze growth patterns
    print("\n--- GROWTH ANALYSIS ---")
    for name in exp2_results:
        scores = [exp2_results[name]["iterations"][it]["avg_score"] for it in range(1, max_iters + 1)]
        # Linear fit: score = a * iteration + b
        iters = np.arange(1, max_iters + 1)
        if len(scores) > 1:
            linear_a = np.polyfit(iters, scores, 1)[0]
            # Quadratic fit: score = a * iteration^2 + b * iteration + c
            quad_a = np.polyfit(iters, scores, 2)[0]
            # If quad_a > 0: super-linear (resonance)
            # If quad_a ~ 0: linear accumulation
            # If quad_a < 0: sub-linear (saturation)
            growth_type = "RESONANCE (super-linear)" if quad_a > 0.001 else \
                          "LINEAR" if abs(quad_a) <= 0.001 else \
                          "SATURATING (sub-linear)"
            print(f"  {name:>24}: linear_slope={linear_a:.4f}, quad_coeff={quad_a:.6f} -> {growth_type}")

    # Plot: suppression vs iteration for all configs
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in exp2_results:
        scores = [exp2_results[name]["iterations"][it]["avg_score"] for it in range(1, max_iters + 1)]
        iters = np.arange(1, max_iters + 1)
        ax.plot(iters, scores, "o-", label=name, markersize=4, linewidth=2)
    ax.set_xlabel("Iteration (cumulative application count)", fontsize=12)
    ax.set_ylabel("Avg Suppression Score", fontsize=12)
    ax.set_title(f"Scalar Resonance Accumulation\n(per-iter amp={per_iter_amp}, max {max_iters} iterations)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Full suppression (score=1.0)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "resonance_accumulation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved resonance plot: {OUTPUT_DIR}/resonance_accumulation.png")

    # Plot: per-layer resonance for the best config
    best_res_name = max(exp2_results.keys(), key=lambda n: exp2_results[n]["iterations"][max_iters]["avg_score"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, li in enumerate(key_layers):
        ax = axes[idx]
        scores = [exp2_results[best_res_name]["iterations"][it]["per_layer"].get(li, 0) for it in range(1, max_iters + 1)]
        iters = np.arange(1, max_iters + 1)
        ax.plot(iters, scores, "o-", color="red" if li in [54, 62, 75] else "blue", markersize=4, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Suppression Score")
        ax.set_title(f"Layer {li}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    plt.suptitle(f"Per-Layer Resonance Accumulation — {best_res_name}\n(per-iter amp={per_iter_amp})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "resonance_per_layer.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-layer resonance plot: {OUTPUT_DIR}/resonance_per_layer.png")

    # Plot: clipping percentage vs iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in exp2_results:
        clips = [exp2_results[name]["iterations"][it]["pct_clipped"] for it in range(1, max_iters + 1)]
        iters = np.arange(1, max_iters + 1)
        ax.plot(iters, clips, "s-", label=name, markersize=3, linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("% Pixels Clipped (at 0 or 1)", fontsize=12)
    ax.set_title("Pixel Saturation During Accumulation", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clipping_vs_iteration.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved clipping plot: {OUTPUT_DIR}/clipping_vs_iteration.png")

    # ============================================================
    # EXPERIMENT 3: Resonance with feedback — re-derive pattern from current state
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Adaptive Resonance — pattern adjusts to current feature state")
    print("=" * 70)

    # Instead of adding the same fixed pattern each time, compute the delta between
    # current features and no-person features, then derive the optimal cancellation
    # pattern from that delta. This is true constructive interference — each iteration
    # pushes harder in the direction that works.

    # We can't backprop through the model (it's in eval mode with no_grad), but we can
    # use the feature delta direction to scale the pattern amplitude adaptively

    # Simple version: start with a fixed pattern, but increase amplitude each iteration
    # proportional to how much more suppression is needed
    adaptive_configs = [
        ("HF_k200d200_adaptive", 200, 200, "HF"),
        ("HF_k200v_adaptive", 0, 200, "HF"),
        ("BEST_adaptive", None, None, None),  # Use best from exp1
    ]

    # Use best orientation from exp1
    best_kx = best_exp1["kx"]
    best_ky = best_exp1["ky"]
    adaptive_configs[2] = (f"BEST_{best_exp1['name']}_adaptive", best_kx, best_ky, best_exp1["band"])

    max_iters_adaptive = 15
    initial_amp = 0.02  # Start small
    amp_growth_factor = 1.5  # Multiply amplitude by this each iteration if suppression is increasing

    print(f"\nMax iterations: {max_iters_adaptive}, initial amp: {initial_amp}, growth factor: {amp_growth_factor}")
    print(f"{'Name':>30}  {'Iter1':>6} {'Iter5':>6} {'Iter10':>6} {'Iter15':>6}  {'FinalAmp':>8}")

    exp3_results = {}

    for name, kx, ky, band in adaptive_configs:
        arr_accum = arr_w.copy()
        current_amp = initial_amp
        iter_scores_adapt = []
        iter_details_adapt = {}

        for it in range(1, max_iters_adaptive + 1):
            pattern = make_sinusoid(H, W, kx, ky, 0, current_amp)
            arr_accum = add_pattern_to_image(arr_accum, pattern)
            tensor_mod = torch.from_numpy(arr_accum).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            caps_mod = forward_capture(model, tensor_mod)

            scores = compute_suppression(caps_mod, caps_w, caps_wo, baseline_dist, LAYERS)
            layer_scores = [scores.get(l, {}).get("score", 0) for l in key_layers]
            avg_score = float(np.mean([s for s in layer_scores if s != 0])) if any(s != 0 for s in layer_scores) else 0.0

            iter_scores_adapt.append(avg_score)
            iter_details_adapt[it] = {
                "avg_score": avg_score,
                "amplitude_used": current_amp,
                "cumulative_amp": current_amp * it,
                "per_layer": {l: scores.get(l, {}).get("score", 0) for l in key_layers},
            }

            # Adaptive: if suppression is still increasing, grow amplitude
            if it > 1 and avg_score > iter_scores_adapt[-2]:
                current_amp *= amp_growth_factor
            # If suppression stalled or decreased, hold amplitude
            # Cap at 0.50 to avoid total saturation
            current_amp = min(current_amp, 0.50)

        exp3_results[name] = {
            "kx": kx, "ky": ky, "band": band,
            "initial_amp": initial_amp, "amp_growth": amp_growth_factor,
            "max_iters": max_iters_adaptive,
            "iterations": iter_details_adapt,
        }

        i1 = iter_scores_adapt[0] if len(iter_scores_adapt) > 0 else 0
        i5 = iter_scores_adapt[4] if len(iter_scores_adapt) > 4 else 0
        i10 = iter_scores_adapt[9] if len(iter_scores_adapt) > 9 else 0
        i15 = iter_scores_adapt[14] if len(iter_scores_adapt) > 14 else 0
        final_amp = iter_details_adapt[max_iters_adaptive]["amplitude_used"]

        print(f"{name:>30}  {i1:6.3f} {i5:6.3f} {i10:6.3f} {i15:6.3f}  {final_amp:8.4f}")

    # Plot adaptive vs fixed resonance comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    # Fixed amp from exp2
    for name in ["HF_k200d200", "HF_k200v"]:
        if name in exp2_results:
            scores = [exp2_results[name]["iterations"][it]["avg_score"] for it in range(1, min(max_iters_adaptive, max_iters) + 1)]
            iters = np.arange(1, len(scores) + 1)
            ax.plot(iters, scores, "o--", label=f"{name} (fixed amp)", markersize=4, linewidth=2, alpha=0.7)
    # Adaptive from exp3
    for name in exp3_results:
        scores = [exp3_results[name]["iterations"][it]["avg_score"] for it in range(1, max_iters_adaptive + 1)]
        iters = np.arange(1, max_iters_adaptive + 1)
        ax.plot(iters, scores, "s-", label=f"{name} (adaptive amp)", markersize=5, linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Avg Suppression Score", fontsize=12)
    ax.set_title("Fixed vs Adaptive Amplitude Resonance", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_vs_fixed_resonance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved adaptive comparison plot: {OUTPUT_DIR}/adaptive_vs_fixed_resonance.png")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # JSON
    json_out = {
        "experiment_1_orthogonal_axis": {
            "description": "Test all 8 orientations at LF/MF/HF to find which axis suppresses the person signal most",
            "amplitude": amp,
            "phase": phase,
            "results": exp1_results,
            "ranking": [{"rank": i+1, "name": r["name"], "kx": r["kx"], "ky": r["ky"],
                         "theta": r["theta"], "band": r["band"], "avg_score": r["avg_score"]}
                        for i, r in enumerate(ranked[:15])],
        },
        "experiment_2_scalar_resonance": {
            "description": "Apply same perturbation cumulatively N times — does suppression compound?",
            "per_iteration_amplitude": per_iter_amp,
            "max_iterations": max_iters,
            "results": exp2_results,
        },
        "experiment_3_adaptive_resonance": {
            "description": "Adaptive amplitude — grow perturbation when suppression is still increasing",
            "initial_amplitude": initial_amp,
            "growth_factor": amp_growth_factor,
            "max_iterations": max_iters_adaptive,
            "results": exp3_results,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "orthogonal_resonance.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # CSV
    csv1_path = os.path.join(OUTPUT_DIR, "orthogonal_axis.csv")
    with open(csv1_path, "w", newline="") as f:
        if exp1_csv:
            w = csv.DictWriter(f, fieldnames=exp1_csv[0].keys())
            w.writeheader()
            w.writerows(exp1_csv)
    print(f"Saved CSV: {csv1_path}")

    csv2_path = os.path.join(OUTPUT_DIR, "resonance_accumulation.csv")
    with open(csv2_path, "w", newline="") as f:
        if exp2_csv:
            w = csv.DictWriter(f, fieldnames=exp2_csv[0].keys())
            w.writeheader()
            w.writerows(exp2_csv)
    print(f"Saved CSV: {csv2_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
