"""
Angle dependence + hallucination onset + k12_stripes frequency analysis.

Three experiments:
1. Hallucination onset: find exact amplitude where total_persons > baseline for each pattern/size
2. Viewing angle dependence: rotate patch 0/15/30/45 deg, measure embedding corruption direction shift
3. k12_stripes frequency analysis: FFT comparison vs other patterns
"""

import os
import csv
import json
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "CUDA required"

from patch_scale_pipeline import (
    Darknet, CONFIG_PATH, WEIGHTS_PATH, IMG_SIZE, IMG_WITH,
    PATCH_CX, PATCH_CY, WEARER_THRESHOLD, PRINT_W_PX, PRINT_H_PX,
    DETECTION_LAYERS, COCO_NAMES,
    load_image, make_deformable_mask, make_patch_pattern,
    simulate_camera, forward_capture_v3, get_dets_v3,
    extract_embedding, cosine_similarity, l2_distance,
    OUTPUT_DIR
)

# ============================================================
# Experiment 1: Hallucination Onset from existing data
# ============================================================
def analyze_hallucination_onset():
    print("=" * 70)
    print("EXPERIMENT 1: HALLUCINATION ONSET")
    print("=" * 70)

    csv_path = os.path.join(OUTPUT_DIR, "pipeline_results.csv")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["amplitude"] = float(r["amplitude"])
            r["total_persons"] = int(r["total_persons"])
            r["wearer_count"] = int(r["wearer_count"])
            r["bystander_count"] = int(r["bystander_count"])
            r["baseline_wearer"] = int(r["baseline_wearer"])
            r["baseline_bystander"] = int(r["baseline_bystander"])
            rows.append(r)

    # Baseline is 15 persons (4 wearer + 11 bystander)
    baseline = 15

    # Group by patch_size + pattern
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["patch_size"], r["pattern"])
        groups[key].append(r)

    hallucination_results = []
    for (ps, pat), group_rows in sorted(groups.items()):
        group_rows.sort(key=lambda x: x["amplitude"])
        hallucination_onset = None
        max_persons = 0
        max_amp = 0
        for r in group_rows:
            if r["total_persons"] > baseline and hallucination_onset is None:
                hallucination_onset = r["amplitude"]
            if r["total_persons"] > max_persons:
                max_persons = r["total_persons"]
                max_amp = r["amplitude"]

        result = {
            "patch_size": ps,
            "pattern": pat,
            "baseline_persons": baseline,
            "hallucination_onset": hallucination_onset,
            "max_persons": max_persons,
            "max_persons_amp": max_amp,
            "phantom_count": max_persons - baseline,
        }
        hallucination_results.append(result)

        if hallucination_onset is not None:
            print(f"  {ps:12s} {pat:15s}: onset={hallucination_onset:.3f}, max={max_persons} ({max_persons-baseline} phantoms) at amp={max_amp:.3f}")
        else:
            print(f"  {ps:12s} {pat:15s}: NO hallucination (max={max_persons})")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "hallucination_onset.json")
    with open(out_path, "w") as f:
        json.dump(hallucination_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Key finding: hard ceiling for k12_stripes
    k12_stripes = [r for r in hallucination_results if r["pattern"] == "k12_stripes"]
    print(f"\n  k12_stripes hallucination analysis:")
    for r in k12_stripes:
        if r["hallucination_onset"] is not None:
            print(f"    {r['patch_size']}: phantoms start at amp={r['hallucination_onset']:.3f}, max {r['phantom_count']} phantoms")
        else:
            print(f"    {r['patch_size']}: NO hallucination at any tested amplitude")

    return hallucination_results


# ============================================================
# Experiment 2: Viewing Angle Dependence
# ============================================================
def rotate_patch(pattern, mask, angle_deg):
    """Rotate a 2D pattern and mask around center, returning same-size output."""
    h, w = pattern.shape
    cx, cy = w / 2, h / 2
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Build coordinate meshgrid for output
    y_out, x_out = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Map output coords to input coords (inverse rotation)
    dx = x_out - cx
    dy = y_out - cy
    x_in = cos_t * dx - sin_t * dy + cx
    y_in = sin_t * dx + cos_t * dy + cy

    # Bilinear interpolation
    x0 = np.floor(x_in).astype(int)
    y0 = np.floor(y_in).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    wx = x_in - np.floor(x_in)
    wy = y_in - np.floor(y_in)
    wx = np.clip(wx, 0, 1)
    wy = np.clip(wy, 0, 1)

    pat_rot = (pattern[y0, x0] * (1 - wx) * (1 - wy) +
               pattern[y1, x0] * (1 - wx) * wy +
               pattern[y0, x1] * wx * (1 - wy) +
               pattern[y1, x1] * wx * wy)
    mask_rot = (mask[y0, x0] * (1 - wx) * (1 - wy) +
                mask[y1, x0] * (1 - wx) * wy +
                mask[y0, x1] * wx * (1 - wy) +
                mask[y1, x1] * wx * wy)

    return pat_rot.astype(np.float32), mask_rot.astype(np.float32)


def run_angle_experiment(v3_model, arr_base, wearers, bystanders, clean_embs):
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: VIEWING ANGLE DEPENDENCE")
    print(f"{'=' * 70}")

    angles = [0, 5, 10, 15, 20, 30, 45]
    amplitudes = [0.005, 0.02, 0.04, 0.08]
    # Test k12_stripes (best dual-purpose) and k12_square_d (max corruption)
    patterns = [
        ("k12_stripes", 12, "stripes_v"),
        ("k12_square_d", 12, "square_d"),
        ("digits_196", 0, "digits_196"),
    ]

    # Use xlarge_16pct rays
    shirt_rays = [110, 90, 120, 85, 115, 95, 125, 90, 110, 95, 120, 85]
    patch_w_416 = int(max(shirt_rays) * 2)  # 250px

    RENDER_SCALE = 4
    render_w = patch_w_416 * RENDER_SCALE
    render_h = patch_w_416 * RENDER_SCALE

    # Camera config
    cam_dict = {
        "render_scale": RENDER_SCALE,
        "blur_sigma": 2.5,
        "jpeg_quality": 75,
        "perspective_warp": 0.05,
        "final_w": patch_w_416,
        "final_h": patch_w_416,
    }

    # Base mask at render resolution
    base_mask, _ = make_deformable_mask(
        render_h, render_w, render_w // 2, render_h // 2,
        [r * RENDER_SCALE for r in shirt_rays], 12
    )
    mask_416_direct, _ = make_deformable_mask(
        IMG_SIZE, IMG_SIZE, PATCH_CX, PATCH_CY, shirt_rays, 12
    )

    results = []

    for pat_label, k_patch, tex_type in patterns:
        print(f"\n  Pattern: {pat_label}")
        base_pat = make_patch_pattern(render_w, render_h, k_patch, tex_type, 1.0)

        for angle in angles:
            # Rotate pattern and mask at render resolution
            if angle == 0:
                pat_rot = base_pat.copy()
                mask_rot = base_mask.copy()
            else:
                pat_rot, mask_rot = rotate_patch(base_pat, base_mask, angle)

            for amp in amplitudes:
                pat_scaled = pat_rot * amp

                # Simulate camera capture
                pat_416, mask_416_cam = simulate_camera(pat_scaled, mask_rot, cam_dict)

                # Composite onto image
                arr_patched = arr_base.copy()
                ph, pw = pat_416.shape
                x0 = PATCH_CX - pw // 2
                y0 = PATCH_CY - ph // 2

                full_pat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

                px0 = max(0, x0)
                py0 = max(0, y0)
                px1 = min(IMG_SIZE, x0 + pw)
                py1 = min(IMG_SIZE, y0 + ph)
                sx0 = px0 - x0
                sy0 = py0 - y0
                sx1 = sx0 + (px1 - px0)
                sy1 = sy0 + (py1 - py0)

                full_pat[py0:py1, px0:px1] = pat_416[sy0:sy1, sx0:sx1]
                full_mask[py0:py1, px0:px1] = mask_416_cam[sy0:sy1, sx0:sx1]

                for c in range(3):
                    arr_patched[:, :, c] = np.clip(
                        arr_base[:, :, c] * (1 - full_mask) +
                        (arr_base[:, :, c] + full_pat) * full_mask,
                        0, 1
                    )

                # Run YOLOv3
                tensor_patch = torch.from_numpy(arr_patched).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    caps_patch, _ = forward_capture_v3(v3_model, tensor_patch)
                patch_dets = get_dets_v3(v3_model, tensor_patch, conf=0.1)
                patch_persons = [d for d in patch_dets if d["class_name"] == "person"]

                # Extract wearer embeddings at all 3 detection heads
                wearer_l2 = {}
                wearer_cos = {}
                wearer_embs = {}  # Store actual embedding vectors for cross-angle comparison

                for layer_name, layer_idx in DETECTION_LAYERS.items():
                    if wearers:
                        w_vecs = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in wearers]
                        w_emb = np.mean(w_vecs, axis=0)
                        w_clean = clean_embs["wearer"][layer_name]
                        wearer_l2[layer_name] = l2_distance(w_emb, w_clean)
                        wearer_cos[layer_name] = cosine_similarity(w_emb, w_clean)
                        wearer_embs[layer_name] = w_emb

                avg_w_l2 = float(np.mean(list(wearer_l2.values())))
                avg_w_cos = float(np.mean(list(wearer_cos.values())))

                # Also extract bystander embeddings
                bystander_l2 = {}
                bystander_cos = {}
                for layer_name, layer_idx in DETECTION_LAYERS.items():
                    if bystanders:
                        b_vecs = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in bystanders]
                        b_emb = np.mean(b_vecs, axis=0)
                        b_clean = np.zeros(255, dtype=np.float32)
                        for be in clean_embs["bystanders"]:
                            if be["layer"] == layer_name:
                                b_clean = be["vec"]
                                break
                        bystander_l2[layer_name] = l2_distance(b_emb, b_clean)
                        bystander_cos[layer_name] = cosine_similarity(b_emb, b_clean)

                avg_b_l2 = float(np.mean(list(bystander_l2.values())))
                avg_b_cos = float(np.mean(list(bystander_cos.values())))

                result = {
                    "pattern": pat_label,
                    "angle": angle,
                    "amplitude": amp,
                    "total_persons": len(patch_persons),
                    "wearer_l2": avg_w_l2,
                    "bystander_l2": avg_b_l2,
                    "wearer_cos": avg_w_cos,
                    "bystander_cos": avg_b_cos,
                    "L81_l2": wearer_l2["L81_52x52"],
                    "L93_l2": wearer_l2["L93_26x26"],
                    "L105_l2": wearer_l2["L105_13x13"],
                    "L81_cos": wearer_cos["L81_52x52"],
                    "L93_cos": wearer_cos["L93_26x26"],
                    "L105_cos": wearer_cos["L105_13x13"],
                }
                results.append(result)

                # Store embeddings for cross-angle cosine
                result["_embs"] = {k: v.copy() for k, v in wearer_embs.items()}

                print(f"    angle={angle:2d} amp={amp:.3f}: P={len(patch_persons):2d} W_L2={avg_w_l2:.2f} W_cos={avg_w_cos:.4f} B_L2={avg_b_l2:.2f}")

    # Cross-angle cosine similarity: compare embedding at angle X vs angle 0
    print(f"\n  Cross-angle cosine similarity (wearer embedding at angle vs angle 0):")
    cross_angle_results = []
    for pat_label, _, _ in patterns:
        for amp in amplitudes:
            # Find angle=0 reference
            ref = None
            for r in results:
                if r["pattern"] == pat_label and r["angle"] == 0 and r["amplitude"] == amp:
                    ref = r
                    break
            if ref is None:
                continue

            for angle in angles:
                cur = None
                for r in results:
                    if r["pattern"] == pat_label and r["angle"] == angle and r["amplitude"] == amp:
                        cur = r
                        break
                if cur is None:
                    continue

                # Cosine between embedding at this angle vs angle 0
                cos_vals = []
                for layer_name in DETECTION_LAYERS:
                    ref_vec = ref["_embs"][layer_name]
                    cur_vec = cur["_embs"][layer_name]
                    cos_vals.append(cosine_similarity(cur_vec, ref_vec))

                cross_cos = float(np.mean(cos_vals))
                cross_angle_results.append({
                    "pattern": pat_label,
                    "amplitude": amp,
                    "angle": angle,
                    "cross_angle_cosine": cross_cos,
                    "L81_cross_cos": cos_vals[0],
                    "L93_cross_cos": cos_vals[1],
                    "L105_cross_cos": cos_vals[2],
                })
                print(f"    {pat_label:15s} amp={amp:.3f} angle={angle:2d} vs 0: cross_cos={cross_cos:.6f}")

    # Clean up stored embeddings from results before saving
    for r in results:
        r.pop("_embs", None)

    # Save
    angle_path = os.path.join(OUTPUT_DIR, "angle_dependence.json")
    with open(angle_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Angle results saved: {angle_path}")

    cross_path = os.path.join(OUTPUT_DIR, "cross_angle_cosine.json")
    with open(cross_path, "w") as f:
        json.dump(cross_angle_results, f, indent=2)
    print(f"  Cross-angle cosine saved: {cross_path}")

    # Plot: L2 vs angle for each pattern at amp=0.04
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (pat_label, _, _) in enumerate(patterns):
        ax = axes[i]
        for amp in amplitudes:
            angle_data = [(r["angle"], r["wearer_l2"]) for r in results
                          if r["pattern"] == pat_label and r["amplitude"] == amp]
            angle_data.sort()
            angles_plot = [d[0] for d in angle_data]
            l2_plot = [d[1] for d in angle_data]
            ax.plot(angles_plot, l2_plot, "o-", label=f"amp={amp:.3f}")
        ax.set_xlabel("Viewing Angle (degrees)")
        ax.set_ylabel("Wearer L2 Distance")
        ax.set_title(f"{pat_label}: L2 vs Angle")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "angle_l2.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Plot: cross-angle cosine vs angle
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (pat_label, _, _) in enumerate(patterns):
        ax = axes[i]
        for amp in amplitudes:
            angle_data = [(r["angle"], r["cross_angle_cosine"]) for r in cross_angle_results
                          if r["pattern"] == pat_label and r["amplitude"] == amp]
            angle_data.sort()
            angles_plot = [d[0] for d in angle_data]
            cos_plot = [d[1] for d in angle_data]
            ax.plot(angles_plot, cos_plot, "o-", label=f"amp={amp:.3f}")
        ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="DeepSORT threshold (0.95)")
        ax.axhline(y=0.70, color="orange", linestyle="--", alpha=0.5, label="Track break (0.70)")
        ax.set_xlabel("Viewing Angle (degrees)")
        ax.set_ylabel("Cosine Similarity vs Angle 0")
        ax.set_title(f"{pat_label}: Cross-Angle Cosine")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.95, 1.0001)
    plt.tight_layout()
    cross_plot_path = os.path.join(OUTPUT_DIR, "cross_angle_cosine.png")
    plt.savefig(cross_plot_path, dpi=150)
    plt.close()
    print(f"  Cross-angle plot saved: {cross_plot_path}")

    return results, cross_angle_results


# ============================================================
# Experiment 3: k12_stripes Frequency Analysis
# ============================================================
def analyze_pattern_frequencies():
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: PATTERN FREQUENCY ANALYSIS")
    print(f"{'=' * 70}")

    # Generate patterns at 256x256 for FFT analysis
    SIZE = 256
    patterns = [
        ("k3_stripes", 3, "stripes_v"),
        ("k6_stripes", 6, "stripes_v"),
        ("k12_stripes", 12, "stripes_v"),
        ("k25_stripes", 25, "stripes_v"),
        ("k3_sine_d", 3, "sinusoid_d"),
        ("k6_sine_d", 6, "sinusoid_d"),
        ("k12_sine_d", 12, "sinusoid_d"),
        ("k6_square_d", 6, "square_d"),
        ("k12_square_d", 12, "square_d"),
        ("k25_square_d", 25, "square_d"),
        ("digits_196", 0, "digits_196"),
    ]

    freq_results = []
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for idx, (label, k_patch, tex_type) in enumerate(patterns):
        pat = make_patch_pattern(SIZE, SIZE, k_patch, tex_type, 1.0)

        # 2D FFT
        fft2d = np.fft.fft2(pat)
        fft_shifted = np.fft.fftshift(fft2d)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)

        # 1D power spectrum (radial average)
        cy, cx = SIZE // 2, SIZE // 2
        y, x = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_int = r.astype(int)
        radial_sum = np.bincount(r_int.ravel(), magnitude.ravel())
        radial_freq = np.arange(len(radial_sum))

        # Dominant frequencies (top 5 peaks)
        flat_mag = magnitude.flatten()
        top_indices = np.argsort(flat_mag)[-10:][::-1]
        top_freqs = []
        for ti in top_indices:
            fy, fx = ti // SIZE, ti % SIZE
            fy_shifted = fy - cx
            fx_shifted = fx - cx
            freq_mag = np.sqrt(fx_shifted ** 2 + fy_shifted ** 2)
            top_freqs.append({
                "fx": int(fx_shifted),
                "fy": int(fy_shifted),
                "radial_freq": float(freq_mag),
                "magnitude": float(magnitude[fy, fx]),
            })

        # Spectral entropy (measure of frequency diversity)
        mag_normalized = magnitude / (magnitude.sum() + 1e-12)
        spectral_entropy = -np.sum(mag_normalized * np.log(mag_normalized + 1e-12))

        # Energy concentration: what % of spectral energy is in top 5 frequencies
        total_energy = np.sum(magnitude ** 2)
        top5_energy = np.sum(sorted(flat_mag ** 2, reverse=True)[:5])
        energy_concentration = float(top5_energy / (total_energy + 1e-12))

        result = {
            "pattern": label,
            "k_patch": k_patch,
            "texture": tex_type,
            "spectral_entropy": float(spectral_entropy),
            "energy_concentration_top5": float(energy_concentration),
            "dominant_freqs": top_freqs[:5],
            "peak_radial_freq": float(top_freqs[0]["radial_freq"]) if top_freqs else 0,
        }
        freq_results.append(result)

        print(f"  {label:15s}: entropy={spectral_entropy:.2f}, concentration={energy_concentration:.3f}, peak_freq={result['peak_radial_freq']:.1f}")

        # Plot pattern + FFT
        row = idx // 4
        col = (idx % 4) * 2
        if col < 4:
            axes[row, col].imshow(pat, cmap="gray", vmin=-1, vmax=1)
            axes[row, col].set_title(f"{label}")
            axes[row, col].axis("off")
        col_fft = (idx % 4) * 2 + 1
        if col_fft < 4:
            axes[row, col_fft].imshow(magnitude_log, cmap="hot")
            axes[row, col_fft].set_title(f"{label} FFT")
            axes[row, col_fft].axis("off")

    plt.tight_layout()
    freq_plot_path = os.path.join(OUTPUT_DIR, "pattern_fft_analysis.png")
    plt.savefig(freq_plot_path, dpi=150)
    plt.close()
    print(f"\n  FFT plot saved: {freq_plot_path}")

    # Save JSON
    freq_json_path = os.path.join(OUTPUT_DIR, "pattern_frequency_analysis.json")
    with open(freq_json_path, "w") as f:
        json.dump(freq_results, f, indent=2)
    print(f"  Frequency analysis saved: {freq_json_path}")

    # Key comparison: k12_stripes vs others
    print(f"\n  Key comparison:")
    k12s = [r for r in freq_results if r["pattern"] == "k12_stripes"][0]
    k12sq = [r for r in freq_results if r["pattern"] == "k12_square_d"][0]
    k6s = [r for r in freq_results if r["pattern"] == "k6_stripes"][0]
    k25s = [r for r in freq_results if r["pattern"] == "k25_stripes"][0]
    digits = [r for r in freq_results if r["pattern"] == "digits_196"][0]

    print(f"    k12_stripes:  entropy={k12s['spectral_entropy']:.2f}, concentration={k12s['energy_concentration_top5']:.3f}")
    print(f"    k6_stripes:   entropy={k6s['spectral_entropy']:.2f}, concentration={k6s['energy_concentration_top5']:.3f}")
    print(f"    k25_stripes:  entropy={k25s['spectral_entropy']:.2f}, concentration={k25s['energy_concentration_top5']:.3f}")
    print(f"    k12_square_d: entropy={k12sq['spectral_entropy']:.2f}, concentration={k12sq['energy_concentration_top5']:.3f}")
    print(f"    digits_196:   entropy={digits['spectral_entropy']:.2f}, concentration={digits['energy_concentration_top5']:.3f}")

    return freq_results


# ============================================================
# Experiment 4: Federated Learning Volume Estimation
# ============================================================
def estimate_poisoning_volume(v3_model, arr_base, wearers, bystanders, clean_embs):
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 4: FEDERATED LEARNING POISONING VOLUME")
    print(f"{'=' * 70}")

    # Extract clean wearer embedding (average across all 3 layers)
    clean_vecs = []
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if wearers:
            w_vecs = [extract_embedding(
                forward_capture_v3(v3_model, torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE))[0],
                layer_idx, d["cx"], d["cy"]
            ) for d in wearers]
            clean_vecs.append(np.mean(w_vecs, axis=0))

    clean_emb = np.mean(clean_vecs, axis=0)  # (255,) average across layers
    clean_norm = np.linalg.norm(clean_emb)

    # Simulate poisoned embedding at Profile A (amp=0.005, L2~2.6)
    # and Profile B (amp=0.04, L2~11.16)
    # We need the actual poisoned embedding
    shirt_rays = [110, 90, 120, 85, 115, 95, 125, 90, 110, 95, 120, 85]
    patch_w_416 = int(max(shirt_rays) * 2)
    RENDER_SCALE = 4
    render_w = patch_w_416 * RENDER_SCALE
    render_h = patch_w_416 * RENDER_SCALE

    cam_dict = {
        "render_scale": RENDER_SCALE,
        "blur_sigma": 2.5,
        "jpeg_quality": 75,
        "perspective_warp": 0.05,
        "final_w": patch_w_416,
        "final_h": patch_w_416,
    }

    base_mask, _ = make_deformable_mask(
        render_h, render_w, render_w // 2, render_h // 2,
        [r * RENDER_SCALE for r in shirt_rays], 12
    )

    base_pat = make_patch_pattern(render_w, render_h, 12, "stripes_v", 1.0)

    # Get poisoned embeddings at Profile A and B
    poisoned_embs = {}
    for profile_name, amp in [("profile_a", 0.005), ("profile_b", 0.04)]:
        pat_scaled = base_pat * amp
        pat_416, mask_416_cam = simulate_camera(pat_scaled, base_mask, cam_dict)

        arr_patched = arr_base.copy()
        ph, pw = pat_416.shape
        x0 = PATCH_CX - pw // 2
        y0 = PATCH_CY - ph // 2
        full_pat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        px0, py0 = max(0, x0), max(0, y0)
        px1 = min(IMG_SIZE, x0 + pw)
        py1 = min(IMG_SIZE, y0 + ph)
        sx0, sy0 = px0 - x0, py0 - y0
        sx1, sy0_end = sx0 + (px1 - px0), sy0 + (py1 - py0)
        full_pat[py0:py1, px0:px1] = pat_416[sy0:sy0_end, sx0:sx1]
        full_mask[py0:py1, px0:px1] = mask_416_cam[sy0:sy0_end, sx0:sx1]

        for c in range(3):
            arr_patched[:, :, c] = np.clip(
                arr_base[:, :, c] * (1 - full_mask) +
                (arr_base[:, :, c] + full_pat) * full_mask,
                0, 1
            )

        tensor_patch = torch.from_numpy(arr_patched).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            caps_patch, _ = forward_capture_v3(v3_model, tensor_patch)

        pois_vecs = []
        for layer_name, layer_idx in DETECTION_LAYERS.items():
            if wearers:
                w_vecs = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in wearers]
                pois_vecs.append(np.mean(w_vecs, axis=0))

        poisoned_embs[profile_name] = np.mean(pois_vecs, axis=0)

    # Simulate a training batch of N embeddings
    # Clean embeddings have natural variation: simulate with Gaussian noise
    # Assume clean embedding variation: sigma = 0.5 * clean_norm / 255 per dimension
    # (this is a rough estimate of natural embedding variation from lighting/angle)
    batch_sizes = [100, 500, 1000, 5000, 10000]
    poison_fractions = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Natural embedding noise: estimate from baseline L2 at amp=0
    # From pipeline data, amp=0 L2 is ~0.20-0.30, so sigma ~ 0.25
    natural_sigma = 0.25

    results = []
    print(f"\n  Clean embedding norm: {clean_norm:.2f}")
    print(f"  Natural noise sigma: {natural_sigma:.2f}")
    print(f"  Profile A poisoned L2: {l2_distance(poisoned_embs['profile_a'], clean_emb):.2f}")
    print(f"  Profile B poisoned L2: {l2_distance(poisoned_embs['profile_b'], clean_emb):.2f}")

    for batch_size in batch_sizes:
        for frac in poison_fractions:
            n_poison = int(batch_size * frac)
            n_clean = batch_size - n_poison

            for profile_name in ["profile_a", "profile_b"]:
                pois_emb = poisoned_embs[profile_name]
                pois_l2 = l2_distance(pois_emb, clean_emb)

                # Simulate batch: clean embeddings with natural noise + poisoned embeddings
                clean_batch = clean_emb[np.newaxis, :] + np.random.randn(n_clean, 255).astype(np.float32) * natural_sigma
                poison_batch = pois_emb[np.newaxis, :] + np.random.randn(n_poison, 255).astype(np.float32) * natural_sigma
                full_batch = np.vstack([clean_batch, poison_batch])

                # Compute batch mean
                batch_mean = np.mean(full_batch, axis=0)
                clean_mean = np.mean(clean_batch, axis=0)

                # Shift: L2 distance between batch mean and clean mean
                shift_l2 = l2_distance(batch_mean, clean_mean)
                shift_cos = cosine_similarity(batch_mean, clean_mean)

                # Also compute: does the poisoned mean fall outside 1 std of clean distribution?
                clean_stds = np.std(clean_batch, axis=0)
                outside_1std = np.sum(np.abs(batch_mean - clean_mean) > clean_stds)
                outside_2std = np.sum(np.abs(batch_mean - clean_mean) > 2 * clean_stds)

                result = {
                    "batch_size": batch_size,
                    "poison_fraction": frac,
                    "n_poison": n_poison,
                    "profile": profile_name,
                    "poison_l2": float(pois_l2),
                    "batch_mean_shift_l2": float(shift_l2),
                    "batch_mean_shift_cos": float(shift_cos),
                    "dims_outside_1std": int(outside_1std),
                    "dims_outside_2std": int(outside_2std),
                    "total_dims": 255,
                }
                results.append(result)

    # Print key results
    print(f"\n  Key results (shift > 1 std in >10% of dimensions):")
    for r in results:
        pct_outside = r["dims_outside_1std"] / r["total_dims"] * 100
        if pct_outside > 10:
            print(f"    batch={r['batch_size']:5d} frac={r['poison_fraction']:.2f} {r['profile']:10s}: shift_L2={r['batch_mean_shift_l2']:.3f} cos={r['batch_mean_shift_cos']:.6f} dims>1std={r['dims_outside_1std']}/255 ({pct_outside:.0f}%)")

    # Find threshold: minimum poison fraction for >1std shift in >10% of dims
    print(f"\n  Minimum poison fraction for meaningful distribution shift (>10% dims outside 1 std):")
    for profile_name in ["profile_a", "profile_b"]:
        for batch_size in batch_sizes:
            profile_results = [r for r in results if r["profile"] == profile_name and r["batch_size"] == batch_size]
            profile_results.sort(key=lambda x: x["poison_fraction"])
            threshold = None
            for r in profile_results:
                pct = r["dims_outside_1std"] / r["total_dims"] * 100
                if pct > 10:
                    threshold = r["poison_fraction"]
                    break
            if threshold:
                print(f"    {profile_name} batch={batch_size:5d}: threshold = {threshold:.2f} ({int(batch_size * threshold)} poisoned samples)")
            else:
                print(f"    {profile_name} batch={batch_size:5d}: no threshold found in tested range")

    # Save
    vol_path = os.path.join(OUTPUT_DIR, "poisoning_volume.json")
    with open(vol_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Volume analysis saved: {vol_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for profile_name in ["profile_a", "profile_b"]:
        ax = axes[0] if profile_name == "profile_a" else axes[1]
        for batch_size in batch_sizes:
            profile_results = [r for r in results if r["profile"] == profile_name and r["batch_size"] == batch_size]
            profile_results.sort(key=lambda x: x["poison_fraction"])
            fracs = [r["poison_fraction"] for r in profile_results]
            shifts = [r["batch_mean_shift_l2"] for r in profile_results]
            ax.plot(fracs, shifts, "o-", label=f"batch={batch_size}")
        ax.set_xlabel("Poison Fraction")
        ax.set_ylabel("Batch Mean Shift (L2)")
        ax.set_title(f"{profile_name}: Distribution Shift vs Poison Volume")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    vol_plot_path = os.path.join(OUTPUT_DIR, "poisoning_volume.png")
    plt.savefig(vol_plot_path, dpi=150)
    plt.close()
    print(f"  Volume plot saved: {vol_plot_path}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Experiment 1: hallucination onset (from existing CSV, no model needed)
    hallucination_results = analyze_hallucination_onset()

    # Load model for experiments 2-4
    print(f"\nLoading YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters():
        p.requires_grad_(False)
    print(f"  Model loaded on {DEVICE}")

    arr_base = load_image(IMG_WITH, IMG_SIZE)

    # Get baseline detections
    tensor_base = torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_persons = [d for d in base_dets if d["class_name"] == "person"]
    for d in base_persons:
        dist = math.sqrt((d["cx"] - PATCH_CX) ** 2 + (d["cy"] - PATCH_CY) ** 2)
        d["dist_to_patch"] = dist
        d["is_wearer"] = dist < WEARER_THRESHOLD
    wearers = [d for d in base_persons if d["is_wearer"]]
    bystanders = [d for d in base_persons if not d["is_wearer"]]
    print(f"  Baseline: {len(base_persons)} persons ({len(wearers)} wearer, {len(bystanders)} bystander)")

    # Get clean embeddings
    with torch.no_grad():
        caps_clean, _ = forward_capture_v3(v3_model, tensor_base)
    clean_embs = {"wearer": {}, "bystanders": []}
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if wearers:
            w_vecs = [extract_embedding(caps_clean, layer_idx, d["cx"], d["cy"]) for d in wearers]
            clean_embs["wearer"][layer_name] = np.mean(w_vecs, axis=0)
        if bystanders:
            for d in bystanders:
                b_vec = extract_embedding(caps_clean, layer_idx, d["cx"], d["cy"])
                clean_embs["bystanders"].append({"layer": layer_name, "det": d, "vec": b_vec})

    # Experiment 2: viewing angle dependence
    angle_results, cross_angle_results = run_angle_experiment(v3_model, arr_base, wearers, bystanders, clean_embs)

    # Experiment 3: frequency analysis (no model needed)
    freq_results = analyze_pattern_frequencies()

    # Experiment 4: federated learning volume
    volume_results = estimate_poisoning_volume(v3_model, arr_base, wearers, bystanders, clean_embs)

    print(f"\n{'=' * 70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: hallucination_onset.json, angle_dependence.json, cross_angle_cosine.json,")
    print(f"         pattern_frequency_analysis.json, poisoning_volume.json")
    print(f"  Plots: angle_l2.png, cross_angle_cosine.png, pattern_fft_analysis.png, poisoning_volume.png")


if __name__ == "__main__":
    main()
