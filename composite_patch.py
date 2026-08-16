"""
Composite Super-Patch — Stack ALL effective patterns into one multi-frequency,
multi-scale, multi-strategy adversarial patch.

Layers (blended with alpha compositing):
  1. Sierpinski k=196 image fractal (d4) — self-similar multi-scale, best L93 hit
  2. Sinusoidal fractal (d4) — broadband k=3,9,27,81,243 coverage
  3. Trained poison patch — optimized embedding corruption
  4. Trained suppress patch — optimized objectness suppression
  5. k=167 diagonal carrier — suppression frequency
  6. k=196 diagonal carrier — disruption frequency
  7. k=208 diagonal carrier — hallucination frequency

Blend strategy: weighted alpha composite where each layer contributes its
unique frequency content. The Sierpinski tiling provides the structural
backbone (triangles at all scales), the sinusoidal fractal fills in
broadband, the trained patches add optimized gradients, and the individual
k=167/196/208 diagonals add sharp frequency peaks.

Also tests multiple blend ratios to find the optimal stacking.
"""

import os, sys, math, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

# Import generators from previous scripts
from fractal_patch import (
    generate_fractal_patch_rgb, compute_2d_fft_mag, radial_average,
    fwd_all, load_img, extract_emb_at, gap_embedding,
    apply_patch_to_image, load_patch_image,
    DETECTION_LAYERS
)
from fractal_image_patch import (
    make_k196_diagonal_source, generate_sierpinski_image_patch_rgb,
    subdivide_triangle
)

CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
POISON_PATCH = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\poison\poison_patch.png"
SUPPRESS_PATCH = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\suppress\suppress_patch.png"
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\composite_patch"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416

os.makedirs(OUT, exist_ok=True)


def make_diagonal_carrier(H, W, k, amp=0.5, phase=0.0):
    """Single-frequency diagonal sinusoid (kx=ky=k)."""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pat = amp * np.cos(2 * np.pi * k * (x / W + y / H) + phase)
    return pat.astype(np.float32)


def make_diagonal_carrier_rgb(H, W, k, amp=0.5):
    """RGB diagonal carrier with per-channel phase offsets."""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pat = np.zeros((H, W, 3), dtype=np.float32)
    phases = [0.0, 2.094, 4.189]
    for c, ph in enumerate(phases):
        pat[:, :, c] = amp * np.cos(2 * np.pi * k * (x / W + y / H) + ph)
    # Normalize to [0, 1]
    pat = (pat - pat.min()) / (pat.max() - pat.min())
    return pat


def normalize_01(arr):
    """Normalize array to [0, 1]."""
    if arr.max() - arr.min() > 0:
        return (arr - arr.min()) / (arr.max() - arr.min())
    return np.full_like(arr, 0.5)


def blend_patches(layers, weights, H, W):
    """Alpha composite multiple patch layers with given weights.

    layers: list of (patch_rgb, mask) tuples
    weights: list of float weights (will be normalized)

    Returns: (blended_rgb, combined_mask)
    """
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    blended = np.zeros((H, W, 3), dtype=np.float32)
    combined_mask = np.zeros((H, W), dtype=np.float32)

    for (patch_rgb, mask), w in zip(layers, weights):
        blended += patch_rgb * mask[:, :, None] * w
        combined_mask = np.maximum(combined_mask, mask)

    # Normalize the blended result within the mask region
    masked = blended * combined_mask[:, :, None]
    if masked.max() > 0:
        blended = normalize_01(blended)

    return blended.astype(np.float32), combined_mask


def build_triangle_mask(H, W, cx, cy, size):
    """Build an equilateral triangle mask."""
    h = size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - size/2, cy + h/3)
    v2 = (cx + size/2, cy + h/3)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32([v0, v1, v2]), 1.0)
    return mask


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Composite Super-Patch — Stacking ALL Effective Patterns")
    print("=" * 70)
    print(f"Device: {DEV}")

    H, W = IS, IS
    patch_cx, patch_cy = IS // 2, int(IS * 0.58)
    outer_size = 200

    # ============================================================
    # 1. Generate all component layers
    # ============================================================
    print("\n--- Generating component layers ---")

    components = {}

    # Layer 1: Sierpinski k=196 image fractal (d4)
    print("  [1/7] Sierpinski k=196 image fractal (d4)...")
    sierp_patch, sierp_mask = generate_sierpinski_image_patch_rgb(
        H, W, patch_cx, patch_cy, outer_size, max_depth=4, src_size=256
    )
    components["sierp_k196_d4"] = (sierp_patch, sierp_mask)

    # Layer 2: Sinusoidal fractal (d4)
    print("  [2/7] Sinusoidal fractal (d4)...")
    fract_patch, fract_mask = generate_fractal_patch_rgb(
        H, W, patch_cx, patch_cy, outer_size, max_depth=4, base_amp=0.5
    )
    components["fractal_sin_d4"] = (fract_patch, fract_mask)

    # Layer 3: Trained poison patch
    print("  [3/7] Trained poison patch...")
    if os.path.exists(POISON_PATCH):
        poison_arr, poison_mask = load_patch_image(
            POISON_PATCH, H, W, patch_cx, patch_cy, outer_size
        )
        components["poison_trained"] = (poison_arr, poison_mask)
    else:
        print("    WARNING: poison patch not found, skipping")

    # Layer 4: Trained suppress patch
    print("  [4/7] Trained suppress patch...")
    if os.path.exists(SUPPRESS_PATCH):
        supp_arr, supp_mask = load_patch_image(
            SUPPRESS_PATCH, H, W, patch_cx, patch_cy, outer_size
        )
        components["suppress_trained"] = (supp_arr, supp_mask)
    else:
        print("    WARNING: suppress patch not found, skipping")

    # Layer 5: k=167 diagonal carrier
    print("  [5/7] k=167 diagonal carrier...")
    k167 = make_diagonal_carrier_rgb(H, W, 167, amp=0.5)
    tri_mask = build_triangle_mask(H, W, patch_cx, patch_cy, outer_size)
    components["k167_diag"] = (k167, tri_mask)

    # Layer 6: k=196 diagonal carrier
    print("  [6/7] k=196 diagonal carrier...")
    k196 = make_diagonal_carrier_rgb(H, W, 196, amp=0.5)
    components["k196_diag"] = (k196, tri_mask.copy())

    # Layer 7: k=208 diagonal carrier
    print("  [7/7] k=208 diagonal carrier...")
    k208 = make_diagonal_carrier_rgb(H, W, 208, amp=0.5)
    components["k208_diag"] = (k208, tri_mask.copy())

    print(f"  Generated {len(components)} component layers")

    # Save component visualizations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (name, (patch, mask)) in enumerate(components.items()):
        ax = axes[i // 4][i % 4]
        ax.imshow(patch)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    plt.suptitle("Component Layers for Composite Patch", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/components_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 2. Define blend configurations
    # ============================================================
    print("\n--- Defining blend configurations ---")

    comp_names = list(components.keys())
    n_comp = len(comp_names)

    blends = {}

    # Config A: Equal weight all layers
    blends["equal_all"] = {name: 1.0 for name in comp_names}

    # Config B: Sierpinski-dominant (sierp gets 3x weight)
    blends["sierp_dominant"] = {name: 3.0 if "sierp" in name else 1.0 for name in comp_names}

    # Config C: Fractal-dominant (sierp + sinusoidal fractal get 2x)
    blends["fractal_dominant"] = {
        name: 2.0 if ("sierp" in name or "fractal_sin" in name) else 1.0
        for name in comp_names
    }

    # Config D: Trained-dominant (poison + suppress get 3x)
    blends["trained_dominant"] = {
        name: 3.0 if "trained" in name else 1.0 for name in comp_names
    }

    # Config E: Frequency-weighted (k=196 gets 2x, sierp gets 2x, trained get 1.5x)
    blends["freq_weighted"] = {
        "sierp_k196_d4": 2.0,
        "fractal_sin_d4": 1.0,
        "poison_trained": 1.5,
        "suppress_trained": 1.5,
        "k167_diag": 1.0,
        "k196_diag": 2.0,
        "k208_diag": 1.0,
    }

    # Config F: Maximum aggression (all weighted by their individual L2 shift performance)
    # Based on forward pass results: sierp L93=8.49, fractal L81=8.36, poison L81=8.48
    blends["max_aggression"] = {
        "sierp_k196_d4": 2.5,   # best L93
        "fractal_sin_d4": 2.0,  # broadband
        "poison_trained": 2.0,  # best L81
        "suppress_trained": 1.5,
        "k167_diag": 1.0,
        "k196_diag": 1.5,
        "k208_diag": 1.0,
    }

    # Config G: Sierpinski + k-frequency stack only (no trained patches)
    blends["untrained_super"] = {
        "sierp_k196_d4": 2.0,
        "fractal_sin_d4": 1.5,
        "k167_diag": 1.0,
        "k196_diag": 1.5,
        "k208_diag": 1.0,
    }

    # ============================================================
    # 3. Generate all composite patches
    # ============================================================
    print("\n--- Generating composite patches ---")

    composite_patches = {}
    for blend_name, weights in blends.items():
        # Filter to only available components
        avail_names = [n for n in comp_names if n in weights]
        layers = [components[n] for n in avail_names]
        w = [weights[n] for n in avail_names]

        patch_rgb, mask = blend_patches(layers, w, H, W)
        composite_patches[blend_name] = (patch_rgb, mask)
        area_pct = float(np.mean(mask)) * 100
        print(f"  {blend_name}: {len(avail_names)} layers, area={area_pct:.1f}%")

        # Save composite patch
        patch_uint8 = (patch_rgb * 255).astype(np.uint8)
        Image.fromarray(patch_uint8).save(f"{OUT}/composite_{blend_name}_416.png")

        # FFT analysis
        gray = np.mean(patch_rgb, axis=2)
        mag = compute_2d_fft_mag(gray)
        radial = radial_average(mag, H, W)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Mask")
        axes[0].axis("off")
        axes[1].imshow(patch_rgb)
        axes[1].set_title(f"Composite ({blend_name})")
        axes[1].axis("off")
        axes[2].imshow(mag, cmap="inferno", extent=[-W//2, W//2, H//2, -H//2])
        axes[2].set_title("2D FFT")
        axes[2].axis("off")
        axes[3].plot(range(len(radial)), radial, "b-", linewidth=0.8)
        for k in [3, 9, 27, 81, 167, 196, 208, 243]:
            if k < len(radial):
                axes[3].axvline(x=k, color="r", linestyle="--", alpha=0.4)
                axes[3].text(k, radial.max() * 0.9, str(k), fontsize=6, color="r", rotation=90)
        axes[3].set_title("Radial FFT")
        axes[3].set_xlabel("k")
        axes[3].set_xlim(0, min(250, len(radial)))
        plt.suptitle(f"Composite Patch: {blend_name}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/composite_{blend_name}_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ============================================================
    # 4. Load YOLOv3 and test all composites
    # ============================================================
    print("\n--- Loading YOLOv3 ---")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    arr_w = load_img(IMG_WITH, IS)
    arr_wo = load_img(IMG_WITHOUT, IS)

    tensor_clean = torch.from_numpy(arr_w).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_clean, _ = fwd_all(model, tensor_clean)
    tensor_empty = torch.from_numpy(arr_wo).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_empty, _ = fwd_all(model, tensor_empty)

    person_signal = {}
    for lname, lidx in DETECTION_LAYERS.items():
        person_signal[lname] = (caps_clean[lidx] - caps_empty[lidx]).squeeze(0)

    person_sx, person_sy = IS // 2, int(IS * 0.58)
    baseline_emb = {}
    for lname, lidx in DETECTION_LAYERS.items():
        gap = gap_embedding(caps_clean, lidx)
        point = extract_emb_at(caps_clean, lidx, person_sx, person_sy)
        baseline_emb[lname] = {"gap": gap, "point": point}

    print(f"  Baseline extracted at {list(DETECTION_LAYERS.keys())}")

    print("\n--- Testing composite patches ---")
    results = {}

    # Test all composites
    for pname, (patch_rgb, mask) in composite_patches.items():
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            point = extract_emb_at(caps_mod, lidx, person_sx, person_sy)
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_emb[lname]["gap"].unsqueeze(0))[0])
            cos_point = float(F.cosine_similarity(
                point.unsqueeze(0), baseline_emb[lname]["point"].unsqueeze(0))[0])
            l2_gap = float(torch.norm(gap - baseline_emb[lname]["gap"]).item())
            l2_point = float(torch.norm(point - baseline_emb[lname]["point"]).item())
            raw_l2 = float(torch.norm(gap).item())
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)
            ps = person_signal[lname]
            overlap = float(F.cosine_similarity(
                delta.flatten().unsqueeze(0),
                ps.flatten().unsqueeze(0))[0])
            metrics[lname] = {
                "cos_gap": cos_gap, "cos_point": cos_point,
                "l2_shift_gap": l2_gap, "l2_shift_point": l2_point,
                "raw_l2_gap": raw_l2, "person_overlap": overlap,
            }
        results[pname] = metrics
        print(f"  {pname}:")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} cos_pt={m['cos_point']:.4f} "
                  f"l2_shift={m['l2_shift_gap']:.3f} overlap={m['person_overlap']:.4f}")

    # Test individual best patches for comparison
    print("\n--- Testing individual comparison patches ---")
    comparisons = [
        ("sierp_img_d4", components.get("sierp_k196_d4")),
        ("fractal_d4", components.get("fractal_sin_d4")),
        ("poison", components.get("poison_trained")),
        ("suppress", components.get("suppress_trained")),
    ]
    for pname, comp in comparisons:
        if comp is None:
            continue
        patch_rgb, mask = comp
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            point = extract_emb_at(caps_mod, lidx, person_sx, person_sy)
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_emb[lname]["gap"].unsqueeze(0))[0])
            cos_point = float(F.cosine_similarity(
                point.unsqueeze(0), baseline_emb[lname]["point"].unsqueeze(0))[0])
            l2_gap = float(torch.norm(gap - baseline_emb[lname]["gap"]).item())
            l2_point = float(torch.norm(point - baseline_emb[lname]["point"]).item())
            raw_l2 = float(torch.norm(gap).item())
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)
            ps = person_signal[lname]
            overlap = float(F.cosine_similarity(
                delta.flatten().unsqueeze(0),
                ps.flatten().unsqueeze(0))[0])
            metrics[lname] = {
                "cos_gap": cos_gap, "cos_point": cos_point,
                "l2_shift_gap": l2_gap, "l2_shift_point": l2_point,
                "raw_l2_gap": raw_l2, "person_overlap": overlap,
            }
        results[pname] = metrics
        print(f"  {pname} (individual):")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} cos_pt={m['cos_point']:.4f} "
                  f"l2_shift={m['l2_shift_gap']:.3f} overlap={m['person_overlap']:.4f}")

    # ============================================================
    # 5. Find best composite and generate print-ready version
    # ============================================================
    print("\n--- Finding best composite ---")

    # Score: sum of L2 shifts across all 3 layers (higher = more corruption)
    best_name = None
    best_score = 0
    for pname in composite_patches:
        score = sum(results[pname][l]["l2_shift_gap"] for l in DETECTION_LAYERS)
        if score > best_score:
            best_score = score
            best_name = pname
    print(f"  Best composite: {best_name} (total L2 shift = {best_score:.3f})")

    # Generate print-ready version of best composite
    print("\n--- Generating print-ready version ---")
    PRINT_W, PRINT_H = 3600, 4800
    print_cx, print_cy = PRINT_W // 2, int(PRINT_H * 0.45)
    print_outer = min(int(200 * PRINT_W / IS * 3.5), int(PRINT_W * 0.5))

    # Regenerate best composite at print resolution
    best_weights = blends[best_name]
    avail_names = [n for n in comp_names if n in best_weights]

    print_components = {}
    # Sierpinski at print res
    if "sierp_k196_d4" in avail_names:
        p, m = generate_sierpinski_image_patch_rgb(
            PRINT_H, PRINT_W, print_cx, print_cy, print_outer,
            max_depth=5, src_size=1024  # deeper for print res
        )
        print_components["sierp_k196_d4"] = (p, m)
    # Sinusoidal fractal at print res
    if "fractal_sin_d4" in avail_names:
        p, m = generate_fractal_patch_rgb(
            PRINT_H, PRINT_W, print_cx, print_cy, print_outer,
            max_depth=5, base_amp=0.5
        )
        print_components["fractal_sin_d4"] = (p, m)
    # Trained patches at print res
    for tname, tpath in [("poison_trained", POISON_PATCH), ("suppress_trained", SUPPRESS_PATCH)]:
        if tname in avail_names and os.path.exists(tpath):
            img = Image.open(tpath).convert("RGB").resize((PRINT_W, PRINT_H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            mask = build_triangle_mask(PRINT_H, PRINT_W, print_cx, print_cy, print_outer)
            print_components[tname] = (arr, mask)
    # Diagonal carriers at print res
    for kname, k in [("k167_diag", 167), ("k196_diag", 196), ("k208_diag", 208)]:
        if kname in avail_names:
            pat = make_diagonal_carrier_rgb(PRINT_H, PRINT_W, k, amp=0.5)
            mask = build_triangle_mask(PRINT_H, PRINT_W, print_cx, print_cy, print_outer)
            print_components[kname] = (pat, mask)

    print_layers = [print_components[n] for n in avail_names if n in print_components]
    print_w = [best_weights[n] for n in avail_names if n in print_components]
    print_patch, print_mask = blend_patches(print_layers, print_w, PRINT_H, PRINT_W)

    print_uint8 = (print_patch * 255).astype(np.uint8)
    print_path = f"{OUT}/composite_{best_name}_print_3600x4800_300dpi.png"
    Image.fromarray(print_uint8).save(print_path)
    print(f"  Saved: {print_path}")

    # Also save all composites at print res
    for blend_name in blends:
        bw = blends[blend_name]
        avail = [n for n in comp_names if n in bw and n in print_components]
        if not avail:
            continue
        pl = [print_components[n] for n in avail]
        pw = [bw[n] for n in avail]
        pp, pm = blend_patches(pl, pw, PRINT_H, PRINT_W)
        pu = (pp * 255).astype(np.uint8)
        Image.fromarray(pu).save(f"{OUT}/composite_{blend_name}_print_3600x4800_300dpi.png")
        print(f"  Saved: composite_{blend_name}_print_3600x4800_300dpi.png")

    # ============================================================
    # 6. Comparison plot
    # ============================================================
    print("\n--- Generating comparison plot ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    layers = list(DETECTION_LAYERS.keys())
    patch_names = list(results.keys())
    x = np.arange(len(patch_names))

    for row, metric in enumerate(["l2_shift_gap", "cos_gap"]):
        for col, lname in enumerate(layers):
            ax = axes[row][col]
            vals = [results[p][lname][metric] for p in patch_names]
            colors = []
            for p in patch_names:
                if "composite" in p or "equal" in p or "dominant" in p or "aggression" in p or "untrained" in p or "weighted" in p:
                    colors.append("#FF6F00")  # orange for composites
                elif "sierp" in p:
                    colors.append("#2196F3")
                elif "fractal" in p:
                    colors.append("#9C27B0")
                elif "poison" in p:
                    colors.append("#F44336")
                else:
                    colors.append("#4CAF50")
            ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(patch_names, rotation=45, ha="right", fontsize=6)
            ax.set_title(f"{lname} — {metric}")
            ax.grid(True, alpha=0.3, axis="y")
            if metric == "cos_gap":
                ax.set_ylim(0, 1.0)
                ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Composite Super-Patch vs Individual Patches — YOLOv3", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 7. Save results
    # ============================================================
    json_path = f"{OUT}/composite_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    csv_path = f"{OUT}/composite_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "layer", "cos_gap", "cos_point", "l2_shift_gap",
                     "l2_shift_point", "raw_l2_gap", "person_overlap"])
        for pname in results:
            for lname in results[pname]:
                m = results[pname][lname]
                w.writerow([pname, lname, f"{m['cos_gap']:.6f}", f"{m['cos_point']:.6f}",
                            f"{m['l2_shift_gap']:.6f}", f"{m['l2_shift_point']:.6f}",
                            f"{m['raw_l2_gap']:.6f}", f"{m['person_overlap']:.6f}"])
    print(f"Saved CSV: {csv_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nComponent layers: {len(components)}")
    print(f"Blend configurations tested: {len(blends)}")
    print(f"\nBest composite: {best_name}")
    print(f"  Total L2 shift (L81+L93+L105): {best_score:.3f}")

    print(f"\nL2 shift comparison (higher = more corruption):")
    print(f"{'Patch':<30} {'L81':<10} {'L93':<10} {'L105':<10} {'TOTAL':<10}")
    for pname in results:
        r = results[pname]
        total = r["L81_52x52"]["l2_shift_gap"] + r["L93_26x26"]["l2_shift_gap"] + r["L105_13x13"]["l2_shift_gap"]
        print(f"{pname:<30} "
              f"{r['L81_52x52']['l2_shift_gap']:<10.3f} "
              f"{r['L93_26x26']['l2_shift_gap']:<10.3f} "
              f"{r['L105_13x13']['l2_shift_gap']:<10.3f} "
              f"{total:<10.3f}")

    print(f"\nGAP cosine comparison (lower = more corruption):")
    print(f"{'Patch':<30} {'L81':<10} {'L93':<10} {'L105':<10}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<30} "
              f"{r['L81_52x52']['cos_gap']:<10.4f} "
              f"{r['L93_26x26']['cos_gap']:<10.4f} "
              f"{r['L105_13x13']['cos_gap']:<10.4f}")

    print(f"\nPerson-point cosine (lower = more local corruption):")
    print(f"{'Patch':<30} {'L81':<10} {'L93':<10} {'L105':<10}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<30} "
              f"{r['L81_52x52']['cos_point']:<10.4f} "
              f"{r['L93_26x26']['cos_point']:<10.4f} "
              f"{r['L105_13x13']['cos_point']:<10.4f}")

    print(f"\nPrint-ready (all composites): {OUT}/composite_*_print_3600x4800_300dpi.png")
    print(f"Best for printing: {print_path}")
    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
