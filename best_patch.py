"""
Two wearable scale-invariant patches:
  1. SUPPRESS: Sierpinski tiling of trained suppress patch — kills person detection
  2. POISON: Sierpinski tiling of trained poison patch — corrupts embeddings if detected

Each patch tiles the gradient-optimized image into a Sierpinski triangle so
the optimized content appears at every scale. Both are triangular, wearable,
and print-ready at 3600x4800 300dpi.
"""
import os, sys, math, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

from fractal_patch import (
    fwd_all, load_img, extract_emb_at, gap_embedding,
    apply_patch_to_image, compute_2d_fft_mag, radial_average,
    DETECTION_LAYERS
)
from fractal_image_patch import subdivide_triangle, warp_image_to_triangle

CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
POISON = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\poison\poison_patch.png"
SUPPRESS = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\suppress\suppress_patch.png"
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\best_patch"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416
os.makedirs(OUT, exist_ok=True)


def build_tri_mask(H, W, cx, cy, size):
    h = size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - size/2, cy + h/3)
    v2 = (cx + size/2, cy + h/3)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32([v0, v1, v2]), 1.0)
    return mask, (v0, v1, v2)


def render_sierpinski_image(H, W, cx, cy, outer_size, src_img, max_depth=4):
    """Tile src_img into Sierpinski sub-triangles. Each sub-triangle gets
    a warped copy of the source image, including inverted center triangles."""
    h = outer_size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - outer_size/2, cy + h/3)
    v2 = (cx + outer_size/2, cy + h/3)

    triangles = []
    subdivide_triangle(v0, v1, v2, max_depth, triangles)

    src_H, src_W = src_img.shape[:2]
    s_h = src_W * math.sqrt(3) / 2
    s_up = [(src_W/2, src_H/2 - s_h*2/3), (0, src_H/2 + s_h/3), (src_W, src_H/2 + s_h/3)]
    s_dn = [(src_W/2, src_H/2 + s_h*2/3), (src_W, src_H/2 - s_h/3), (0, src_H/2 - s_h/3)]

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)

    triangles.sort(key=lambda t: t[3])

    for tv0, tv1, tv2, level, orient in triangles:
        st = s_up if orient == "up" else s_dn
        warped, tri_mask = warp_image_to_triangle(src_img, st, [tv0, tv1, tv2], (H, W))
        uncovered = np.maximum(0, tri_mask - coverage)
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c] * uncovered
        coverage = np.maximum(coverage, tri_mask)

    if canvas.max() > 0:
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())

    outer_mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(outer_mask, np.int32([v0, v1, v2]), 1.0)
    return np.clip(canvas, 0, 1).astype(np.float32), outer_mask


def load_img_array(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Two Wearable Scale-Invariant Patches — Suppress + Poison")
    print("=" * 70)

    H, W = IS, IS
    cx, cy = IS // 2, int(IS * 0.58)
    outer = 200

    # Load trained patch images
    print("\nLoading trained patches...")
    poison_src = load_img_array(POISON)
    suppress_src = load_img_array(SUPPRESS)
    print(f"  poison: {poison_src.shape}")
    print(f"  suppress: {suppress_src.shape}")

    # Generate Sierpinski tilings at multiple depths
    print("\nGenerating Sierpinski tilings...")
    patches = {}

    for name, src in [("suppress", suppress_src), ("poison", poison_src)]:
        src_r = cv2.resize(src, (256, 256), interpolation=cv2.INTER_LINEAR)
        for depth in [3, 4, 5]:
            patch, mask = render_sierpinski_image(H, W, cx, cy, outer, src_r, max_depth=depth)
            pname = f"{name}_d{depth}"
            patches[pname] = (patch, mask)
            tris = []
            subdivide_triangle((cx, cy - outer*math.sqrt(3)/2*2/3),
                              (cx-outer/2, cy+outer*math.sqrt(3)/2/3),
                              (cx+outer/2, cy+outer*math.sqrt(3)/2/3), depth, tris)
            print(f"  {pname}: {len(tris)} sub-triangles, area={np.mean(mask)*100:.1f}%")
            Image.fromarray((patch * 255).astype(np.uint8)).save(f"{OUT}/{pname}_416.png")

    # Load YOLOv3
    print("\nLoading YOLOv3...")
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

    # Test all patches
    print("\nTesting patches...")
    results = {}

    # Sierpinski tilings
    for pname, (patch_rgb, mask) in patches.items():
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, cx, cy)
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

    # Comparison: original trained patches (no Sierpinski)
    print("\nTesting original trained patches (no tiling)...")
    for pname, ppath in [("poison_orig", POISON), ("suppress_orig", SUPPRESS)]:
        img = Image.open(ppath).convert("RGB").resize((W, H), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        mask, _ = build_tri_mask(H, W, cx, cy, outer)
        arr_mod = apply_patch_to_image(arr_w, arr, mask, cx, cy)
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

    # Print-ready versions
    print("\nGenerating print-ready versions...")
    PRINT_W, PRINT_H = 3600, 4800
    pcx, pcy = PRINT_W // 2, int(PRINT_H * 0.45)
    pouter = min(int(200 * PRINT_W / IS * 3.5), int(PRINT_W * 0.5))

    for name, src in [("suppress", suppress_src), ("poison", poison_src)]:
        src_hr = cv2.resize(src, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        for depth in [4, 5]:
            patch, mask = render_sierpinski_image(
                PRINT_H, PRINT_W, pcx, pcy, pouter, src_hr, max_depth=depth)
            patch_u8 = (patch * 255).astype(np.uint8)
            path = f"{OUT}/{name}_d{depth}_print_3600x4800_300dpi.png"
            Image.fromarray(patch_u8).save(path)
            print(f"  Saved: {path}")

    # FFT analysis for each
    print("\nGenerating FFT analysis...")
    for pname, (patch_rgb, mask) in patches.items():
        gray = np.mean(patch_rgb, axis=2)
        mag = compute_2d_fft_mag(gray)
        radial = radial_average(mag, H, W)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Mask")
        axes[0].axis("off")
        axes[1].imshow(patch_rgb)
        axes[1].set_title(pname)
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
        axes[3].set_xlim(0, min(250, len(radial)))
        plt.suptitle(f"{pname} — Sierpinski tiling of trained patch", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{pname}_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Comparison plot
    print("\nGenerating comparison plot...")
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
                if "suppress" in p:
                    colors.append("#F44336")
                elif "poison" in p:
                    colors.append("#2196F3")
                else:
                    colors.append("#9C27B0")
            ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(patch_names, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{lname} — {metric}")
            ax.grid(True, alpha=0.3, axis="y")
            if metric == "cos_gap":
                ax.set_ylim(0, 1.0)
                ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Sierpinski-Tiled Trained Patches vs Original — YOLOv3", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    json_path = f"{OUT}/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    csv_path = f"{OUT}/results.csv"
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

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nL2 shift (higher = more corruption):")
    print(f"{'Patch':<25} {'L81':<10} {'L93':<10} {'L105':<10} {'TOTAL':<10}")
    for pname in results:
        r = results[pname]
        total = r["L81_52x52"]["l2_shift_gap"] + r["L93_26x26"]["l2_shift_gap"] + r["L105_13x13"]["l2_shift_gap"]
        print(f"{pname:<25} "
              f"{r['L81_52x52']['l2_shift_gap']:<10.3f} "
              f"{r['L93_26x26']['l2_shift_gap']:<10.3f} "
              f"{r['L105_13x13']['l2_shift_gap']:<10.3f} "
              f"{total:<10.3f}")

    print(f"\nGAP cosine (lower = more corruption):")
    print(f"{'Patch':<25} {'L81':<10} {'L93':<10} {'L105':<10}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<25} "
              f"{r['L81_52x52']['cos_gap']:<10.4f} "
              f"{r['L93_26x26']['cos_gap']:<10.4f} "
              f"{r['L105_13x13']['cos_gap']:<10.4f}")

    print(f"\nPrint-ready:")
    print(f"  SUPPRESS: {OUT}/suppress_d4_print_3600x4800_300dpi.png")
    print(f"  SUPPRESS: {OUT}/suppress_d5_print_3600x4800_300dpi.png")
    print(f"  POISON:   {OUT}/poison_d4_print_3600x4800_300dpi.png")
    print(f"  POISON:   {OUT}/poison_d5_print_3600x4800_300dpi.png")
    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
