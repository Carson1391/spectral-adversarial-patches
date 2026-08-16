"""
Hexagonal patch: 6 equilateral triangles arranged around a center point.
Each triangle contains a Sierpinski tiling of a trained patch image.

Configurations:
  - hex_poison: all 6 triangles = Sierpinski poison tiling
  - hex_suppress: all 6 triangles = Sierpinski suppress tiling
  - hex_alternating: 3 poison + 3 suppress alternating
  - hex_3p3s_split: top 3 poison, bottom 3 suppress

More area coverage (~30%) than single triangle (~10%), still wearable.
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
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\hex_patch"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416
os.makedirs(OUT, exist_ok=True)


def hexagon_vertices(cx, cy, radius):
    """6 triangle vertices for a hexagon centered at (cx, cy).
    Returns list of 6 (v0, v1, v2) triangles, each pointing outward from center."""
    tris = []
    for i in range(6):
        # Angle for each triangle sector
        a0 = i * math.pi / 3 - math.pi / 2  # Start from top
        a1 = (i + 1) * math.pi / 3 - math.pi / 2
        # Center point is shared by all triangles
        center = (cx, cy)
        # Outer two vertices
        p1 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0))
        p2 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1))
        tris.append((center, p1, p2))
    return tris


def render_sierpinski_in_triangle(H, W, tv0, tv1, tv2, src_img, max_depth=3):
    """Tile src_img into Sierpinski sub-triangles within the given triangle."""
    triangles = []
    subdivide_triangle(tv0, tv1, tv2, max_depth, triangles)

    src_H, src_W = src_img.shape[:2]
    s_h = src_W * math.sqrt(3) / 2
    s_up = [(src_W/2, src_H/2 - s_h*2/3), (0, src_H/2 + s_h/3), (src_W, src_H/2 + s_h/3)]
    s_dn = [(src_W/2, src_H/2 + s_h*2/3), (src_W, src_H/2 - s_h/3), (0, src_H/2 - s_h/3)]

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)

    triangles.sort(key=lambda t: t[3])

    for stv0, stv1, stv2, level, orient in triangles:
        st = s_up if orient == "up" else s_dn
        warped, tri_mask = warp_image_to_triangle(src_img, st, [stv0, stv1, stv2], (H, W))
        uncovered = np.maximum(0, tri_mask - coverage)
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c] * uncovered
        coverage = np.maximum(coverage, tri_mask)

    if canvas.max() > 0:
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())

    return np.clip(canvas, 0, 1).astype(np.float32), coverage


def render_hex_patch(H, W, cx, cy, radius, src_img, max_depth=3):
    """Full hexagon: 6 triangles, each with Sierpinski tiling of src_img."""
    hex_tris = hexagon_vertices(cx, cy, radius)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)

    for tv0, tv1, tv2 in hex_tris:
        tri_patch, tri_mask = render_sierpinski_in_triangle(H, W, tv0, tv1, tv2, src_img, max_depth)
        uncovered = np.maximum(0, tri_mask - mask)
        for c in range(3):
            canvas[:, :, c] += tri_patch[:, :, c] * uncovered
        mask = np.maximum(mask, tri_mask)

    if canvas.max() > 0:
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())

    return np.clip(canvas, 0, 1).astype(np.float32), mask


def render_hex_split(H, W, cx, cy, radius, src_a, src_b, max_depth=3, split_mode="alternating"):
    """Hexagon with two source images. split_mode: 'alternating' or 'top_bottom'."""
    hex_tris = hexagon_vertices(cx, cy, radius)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)

    for i, (tv0, tv1, tv2) in enumerate(hex_tris):
        if split_mode == "alternating":
            src = src_a if i % 2 == 0 else src_b
        elif split_mode == "top_bottom":
            # Triangles 0,1,5 = top half, 2,3,4 = bottom half
            src = src_a if i in [0, 1, 5] else src_b
        else:
            src = src_a

        tri_patch, tri_mask = render_sierpinski_in_triangle(H, W, tv0, tv1, tv2, src, max_depth)
        uncovered = np.maximum(0, tri_mask - mask)
        for c in range(3):
            canvas[:, :, c] += tri_patch[:, :, c] * uncovered
        mask = np.maximum(mask, tri_mask)

    if canvas.max() > 0:
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())

    return np.clip(canvas, 0, 1).astype(np.float32), mask


def load_img_array(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Hexagonal Sierpinski Patches — 6 triangles, trained patch tiling")
    print("=" * 70)

    H, W = IS, IS
    cx, cy = IS // 2, int(IS * 0.58)
    radius = 120  # Hexagon radius — distance from center to outer vertex

    # Load trained patch images
    print("\nLoading trained patches...")
    poison_src = load_img_array(POISON)
    suppress_src = load_img_array(SUPPRESS)
    poison_src = cv2.resize(poison_src, (256, 256), interpolation=cv2.INTER_LINEAR)
    suppress_src = cv2.resize(suppress_src, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(f"  poison: {poison_src.shape}")
    print(f"  suppress: {suppress_src.shape}")

    # Generate hex patches
    print("\nGenerating hexagonal patches...")
    patches = {}

    # All poison
    for depth in [2, 3, 4]:
        p, m = render_hex_patch(H, W, cx, cy, radius, poison_src, max_depth=depth)
        pname = f"hex_poison_d{depth}"
        patches[pname] = (p, m)
        print(f"  {pname}: area={np.mean(m)*100:.1f}%")
        Image.fromarray((p * 255).astype(np.uint8)).save(f"{OUT}/{pname}_416.png")

    # All suppress
    for depth in [2, 3, 4]:
        p, m = render_hex_patch(H, W, cx, cy, radius, suppress_src, max_depth=depth)
        pname = f"hex_suppress_d{depth}"
        patches[pname] = (p, m)
        print(f"  {pname}: area={np.mean(m)*100:.1f}%")
        Image.fromarray((p * 255).astype(np.uint8)).save(f"{OUT}/{pname}_416.png")

    # Alternating poison/suppress
    for depth in [2, 3, 4]:
        p, m = render_hex_split(H, W, cx, cy, radius, poison_src, suppress_src,
                                max_depth=depth, split_mode="alternating")
        pname = f"hex_alt_d{depth}"
        patches[pname] = (p, m)
        print(f"  {pname}: area={np.mean(m)*100:.1f}%")
        Image.fromarray((p * 255).astype(np.uint8)).save(f"{OUT}/{pname}_416.png")

    # Top/bottom split: top 3 poison, bottom 3 suppress
    for depth in [2, 3, 4]:
        p, m = render_hex_split(H, W, cx, cy, radius, poison_src, suppress_src,
                                max_depth=depth, split_mode="top_bottom")
        pname = f"hex_split_d{depth}"
        patches[pname] = (p, m)
        print(f"  {pname}: area={np.mean(m)*100:.1f}%")
        Image.fromarray((p * 255).astype(np.uint8)).save(f"{OUT}/{pname}_416.png")

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

    # Test all hex patches
    print("\nTesting hex patches...")
    results = {}

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

    # Also test the single-triangle d3 for comparison
    print("\nTesting single-triangle baseline (from best_patch)...")
    from best_patch import render_sierpinski_image
    for name, src in [("poison", poison_src), ("suppress", suppress_src)]:
        p, m = render_sierpinski_image(H, W, cx, cy, 200, src, max_depth=3)
        pname = f"single_{name}_d3"
        arr_mod = apply_patch_to_image(arr_w, p, m, cx, cy)
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
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} l2_shift={m['l2_shift_gap']:.3f}")

    # Print-ready versions of best configs
    print("\nGenerating print-ready versions...")
    PRINT_W, PRINT_H = 3600, 4800
    pcx, pcy = PRINT_W // 2, int(PRINT_H * 0.45)
    pradius = int(radius * PRINT_W / IS * 3.5)

    poison_hr = cv2.resize(poison_src, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    suppress_hr = cv2.resize(suppress_src, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    print_configs = [
        ("hex_poison_d3", poison_hr, "poison", "all"),
        ("hex_suppress_d3", suppress_hr, "suppress", "all"),
        ("hex_alt_d3", (poison_hr, suppress_hr), "alt", "alternating"),
        ("hex_split_d3", (poison_hr, suppress_hr), "split", "top_bottom"),
    ]

    for pname, src, label, mode in print_configs:
        if mode == "all":
            p, m = render_hex_patch(PRINT_H, PRINT_W, pcx, pcy, pradius, src, max_depth=3)
        elif mode == "alternating":
            p, m = render_hex_split(PRINT_H, PRINT_W, pcx, pcy, pradius, src[0], src[1],
                                    max_depth=3, split_mode="alternating")
        elif mode == "top_bottom":
            p, m = render_hex_split(PRINT_H, PRINT_W, pcx, pcy, pradius, src[0], src[1],
                                    max_depth=3, split_mode="top_bottom")
        path = f"{OUT}/{pname}_print_3600x4800_300dpi.png"
        Image.fromarray((p * 255).astype(np.uint8)).save(path)
        print(f"  Saved: {path}")

    # FFT analysis
    print("\nGenerating FFT analysis...")
    for pname, (patch_rgb, mask) in patches.items():
        gray = np.mean(patch_rgb, axis=2)
        mag = compute_2d_fft_mag(gray)
        radial = radial_average(mag, H, W)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title(f"Mask ({np.mean(mask)*100:.0f}%)")
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
        plt.suptitle(f"{pname}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{pname}_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Comparison plot
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
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
                elif "alt" in p:
                    colors.append("#4CAF50")
                elif "split" in p:
                    colors.append("#FF9800")
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

    plt.suptitle("Hexagonal Sierpinski Patches vs Single Triangle — YOLOv3", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(f"{OUT}/results.csv", "w", newline="") as f:
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
    print("SUMMARY — L2 Shift (higher = more corruption)")
    print(f"{'='*70}")
    print(f"{'Patch':<25} {'L81':<10} {'L93':<10} {'L105':<10} {'TOTAL':<10} {'Area':<8}")
    for pname in results:
        r = results[pname]
        total = r["L81_52x52"]["l2_shift_gap"] + r["L93_26x26"]["l2_shift_gap"] + r["L105_13x13"]["l2_shift_gap"]
        area = np.mean(patches[pname][1]) * 100 if pname in patches else 10.2
        print(f"{pname:<25} "
              f"{r['L81_52x52']['l2_shift_gap']:<10.3f} "
              f"{r['L93_26x26']['l2_shift_gap']:<10.3f} "
              f"{r['L105_13x13']['l2_shift_gap']:<10.3f} "
              f"{total:<10.3f} "
              f"{area:<8.1f}%")

    print(f"\nPrint-ready:")
    print(f"  POISON hex:    {OUT}/hex_poison_d3_print_3600x4800_300dpi.png")
    print(f"  SUPPRESS hex:  {OUT}/hex_suppress_d3_print_3600x4800_300dpi.png")
    print(f"  ALT hex:       {OUT}/hex_alt_d3_print_3600x4800_300dpi.png")
    print(f"  SPLIT hex:     {OUT}/hex_split_d3_print_3600x4800_300dpi.png")
    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
