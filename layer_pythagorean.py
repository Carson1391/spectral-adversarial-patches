"""
3-scale Pythagorean detection using classic PyTorch-YOLOv3 Darknet class.

YOLOv3 detects at 3 scales via YOLOLayer modules:
  Scale 1: stride 8  (mask 6,7,8 -> large anchors, fine grid)
  Scale 2: stride 16 (mask 3,4,5 -> medium anchors, medium grid)
  Scale 3: stride 32 (mask 0,1,2 -> small anchors, coarse grid)

Each YOLOLayer output: (B, 3, H, W, 85) = 3 anchors x (4 bbox + 1 obj + 80 cls)

For each scale, find the grid cell where class 0 (person) has the highest
objectness * class_score, extract (x, y) grid position, compute:
  c^2 = x^2 + y^2
  z   = sqrt(c^2)

Outputs:
  - triangles_3scales.png
  - z_3scales.png
  - heatmaps_3scales.png
  - detections_3scales.csv
"""

import os
import sys
import csv
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add PyTorch-YOLOv3 to path
sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
# Bypass imgaug (incompatible with numpy 2.0) with a stub module
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
# Now safe to import
from pytorchyolo.models import Darknet

# ----------------------------- CONFIG -----------------------------
CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMAGE_PATH   = r"C:\Users\carso\Desktop\YODO\data\coco_person\images\000000000036.jpg"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\layer_pythagorean"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 608  # must be multiple of 32
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_image(img_path, img_size):
    """Load and preprocess image to model input tensor."""
    import cv2
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize to img_size with padding
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to img_size
    canvas = Image.new("RGB", (img_size, img_size), (128, 128, 128))
    canvas.paste(img_resized, ((img_size - new_w) // 2, (img_size - new_h) // 2))

    # To tensor
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, (orig_w, orig_h), (new_w, new_h)


def hook_yolo_layers(model):
    """Hook the 3 YOLOLayer modules to capture raw pre-decoded output."""
    captured = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # inp[0] is the raw conv output before YOLOLayer processing: (B, 255, H, W)
            # out is the decoded output: (B, num_anchors*H*W, 85)
            captured[f"yolo_{idx}_input"] = inp[0].detach().clone()
            captured[f"yolo_{idx}_output"] = out.detach().clone()
        return hook_fn

    hooks = []
    yolo_layers = [m for m in model.modules() if m.__class__.__name__ == "YOLOLayer"]
    for i, yolo in enumerate(yolo_layers):
        h = yolo.register_forward_hook(make_hook(i))
        hooks.append(h)
        print(f"  Hooked YOLOLayer {i}: stride={yolo.anchors}")

    return captured, hooks


def parse_yolo_output(raw_input, yolo_layer, scale_idx, img_size):
    """
    Parse raw YOLO layer input (B, 255, H, W) into per-cell person scores.

    255 = 3 anchors * (4 + 1 + 80)
    """
    B, C, H, W = raw_input.shape
    na = yolo_layer.num_anchors  # 3
    no = yolo_layer.no           # 85 = 4+1+80

    # Reshape: (B, 3, 85, H, W) -> (B, 3, H, W, 85)
    out = raw_input.view(B, na, no, H, W).permute(0, 1, 3, 4, 2).contiguous()

    # Components (raw, pre-sigmoid)
    tx = out[..., 0]   # (B, 3, H, W)
    ty = out[..., 1]
    tw = out[..., 2]
    th = out[..., 3]
    obj_raw = out[..., 4]    # objectness logit
    cls_raw = out[..., 5:]   # (B, 3, H, W, 80) class logits

    # Apply sigmoid for scores
    obj = obj_raw.sigmoid()
    cls = cls_raw.sigmoid()

    # Person score = objectness * class_0_score
    person_score = obj * cls[..., 0]  # (B, 3, H, W)

    # Find peak across all anchors and spatial locations
    flat_idx = torch.argmax(person_score[0]).item()
    a_idx = flat_idx // (H * W)
    spatial_idx = flat_idx % (H * W)
    y_grid = spatial_idx // W
    x_grid = spatial_idx % W

    peak_score = person_score[0, a_idx, y_grid, x_grid].item()

    # Decode box center to image space
    stride = img_size // H
    # bx = (sigmoid(tx) + x_grid) * stride
    bx = (torch.sigmoid(tx[0, a_idx, y_grid, x_grid]).item() + x_grid) * stride
    by = (torch.sigmoid(ty[0, a_idx, y_grid, x_grid]).item() + y_grid) * stride

    # Raw values
    raw_tx = tx[0, a_idx, y_grid, x_grid].item()
    raw_ty = ty[0, a_idx, y_grid, x_grid].item()
    sig_tx = torch.sigmoid(tx[0, a_idx, y_grid, x_grid]).item()
    sig_ty = torch.sigmoid(ty[0, a_idx, y_grid, x_grid]).item()

    # Anchor
    anchor = yolo_layer.anchors[a_idx].tolist()

    # Person score map (max across anchors) for heatmap
    score_map = person_score[0].max(dim=0)[0].cpu().numpy()  # (H, W)

    # Pythagorean
    c_sq_grid = float(x_grid) ** 2 + float(y_grid) ** 2
    z_grid = math.sqrt(c_sq_grid)
    c_sq_image = bx ** 2 + by ** 2
    z_image = math.sqrt(c_sq_image)

    return {
        "scale_idx": scale_idx,
        "stride": stride,
        "H": H,
        "W": W,
        "anchor_idx": a_idx,
        "anchor_wh": anchor,
        "x_grid": float(x_grid),
        "y_grid": float(y_grid),
        "raw_tx": raw_tx,
        "raw_ty": raw_ty,
        "sig_tx": sig_tx,
        "sig_ty": sig_ty,
        "bx_image": bx,
        "by_image": by,
        "person_score": peak_score,
        "c_sq_grid": c_sq_grid,
        "z_grid": z_grid,
        "c_sq_image": c_sq_image,
        "z_image": z_image,
        "score_map": score_map,
    }


def plot_triangles(results, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for i, r in enumerate(results):
        # Grid space (top row)
        ax = axes[0, i]
        x, y, z = r["x_grid"], r["y_grid"], r["z_grid"]
        gs = r["H"]
        pad = gs * 0.1
        ax.set_xlim(-pad, gs + pad)
        ax.set_ylim(gs + pad, -pad)
        ax.set_aspect("equal")
        c = colors[i]
        ax.plot([0, x], [0, 0], color=c, linewidth=2)
        ax.plot([x, x], [0, y], color=c, linewidth=2)
        ax.plot([x, 0], [y, 0], color=c, linewidth=2.5, linestyle="--")
        ax.scatter(x, y, color=c, s=80, zorder=5, edgecolors="black")
        ax.scatter(0, 0, color="red", s=50, marker="*", zorder=5)
        ax.annotate(f"z={z:.1f}", xy=(x / 2, y / 2), fontsize=11, color=c,
                    fontweight="bold", xytext=(5, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=c))
        ax.text(x / 2, -pad * 0.3, f"x={x:.0f}", color=c, fontsize=9, ha="center")
        ax.text(x + pad * 0.1, y / 2, f"y={y:.0f}", color=c, fontsize=9, va="center")
        ax.set_title(f"Scale {i+1} (stride {r['stride']}, {r['W']}x{r['H']})\nGrid space", fontsize=10)

        # Image space (bottom row)
        ax2 = axes[1, i]
        bx, by, bz = r["bx_image"], r["by_image"], r["z_image"]
        img_size = IMG_SIZE
        pad2 = img_size * 0.05
        ax2.set_xlim(-pad2, img_size + pad2)
        ax2.set_ylim(img_size + pad2, -pad2)
        ax2.set_aspect("equal")
        ax2.plot([0, bx], [0, 0], color=c, linewidth=2)
        ax2.plot([bx, bx], [0, by], color=c, linewidth=2)
        ax2.plot([bx, 0], [by, 0], color=c, linewidth=2.5, linestyle="--")
        ax2.scatter(bx, by, color=c, s=80, zorder=5, edgecolors="black")
        ax2.scatter(0, 0, color="red", s=50, marker="*", zorder=5)
        ax2.annotate(f"z={bz:.1f}", xy=(bx / 2, by / 2), fontsize=11, color=c,
                     fontweight="bold", xytext=(5, 5), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=c))
        ax2.text(bx / 2, -pad2 * 0.3, f"x={bx:.1f}", color=c, fontsize=9, ha="center")
        ax2.text(bx + pad2 * 0.1, by / 2, f"y={by:.1f}", color=c, fontsize=9, va="center")
        ax2.set_title(f"Scale {i+1} (stride {r['stride']})\nImage space ({img_size}x{img_size})", fontsize=10)

    fig.suptitle("YOLOv3 Person Detection: Pythagorean Triangles at 3 Scales\n"
                 "x^2 + y^2 = z^2  (top: grid space, bottom: image space)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved triangles: {out_path}")


def plot_z_bar(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"Scale {i+1}\n{r['W']}x{r['H']}\nstride {r['stride']}" for i, r in enumerate(results)]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    ax = axes[0]
    z_vals = [r["z_grid"] for r in results]
    bars = ax.bar(range(3), z_vals, color=colors, edgecolor="black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("z = sqrt(x^2 + y^2)  (grid cells)")
    ax.set_title("z in grid space")
    for bar, z in zip(bars, z_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"z={z:.1f}", ha="center", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    z_vals_img = [r["z_image"] for r in results]
    bars2 = ax2.bar(range(3), z_vals_img, color=colors, edgecolor="black")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("z = sqrt(bx^2 + by^2)  (pixels)")
    ax2.set_title("z in image space")
    for bar, z in zip(bars2, z_vals_img):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"z={z:.1f}", ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("z values at 3 detection scales (person class 0)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved z bar chart: {out_path}")


def plot_heatmaps(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, r in enumerate(results):
        ax = axes[i]
        smap = r["score_map"]
        ax.imshow(smap, cmap="hot", aspect="auto")
        ax.scatter(r["x_grid"], r["y_grid"], color="cyan", s=100, marker="x",
                   linewidths=2, zorder=5)
        ax.set_title(f"Scale {i+1} (stride {r['stride']}, {r['W']}x{r['H']})\n"
                     f"peak=({r['x_grid']:.0f}, {r['y_grid']:.0f})  "
                     f"score={r['person_score']:.4f}\n"
                     f"z_grid={r['z_grid']:.1f}  z_image={r['z_image']:.1f}",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle("Person (class 0) score heatmaps: objectness x class_score at each scale",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmaps: {out_path}")


def save_csv(results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scale", "stride", "grid_W", "grid_H", "anchor_idx", "anchor_wh",
            "x_grid", "y_grid", "raw_tx", "raw_ty", "sig_tx", "sig_ty",
            "bx_image", "by_image", "person_score",
            "c_sq_grid", "z_grid", "c_sq_image", "z_image",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scale": r["scale_idx"] + 1,
                "stride": r["stride"],
                "grid_W": r["W"],
                "grid_H": r["H"],
                "anchor_idx": r["anchor_idx"],
                "anchor_wh": r["anchor_wh"],
                "x_grid": r["x_grid"],
                "y_grid": r["y_grid"],
                "raw_tx": r["raw_tx"],
                "raw_ty": r["raw_ty"],
                "sig_tx": r["sig_tx"],
                "sig_ty": r["sig_ty"],
                "bx_image": r["bx_image"],
                "by_image": r["by_image"],
                "person_score": r["person_score"],
                "c_sq_grid": r["c_sq_grid"],
                "z_grid": r["z_grid"],
                "c_sq_image": r["c_sq_image"],
                "z_image": r["z_image"],
            })
    print(f"Saved CSV: {out_path}")


def main():
    print("=" * 60)
    print("Loading classic YOLOv3 (Darknet)...")
    print("=" * 60)
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    # Get YOLO layers
    yolo_layers = [m for m in model.modules() if m.__class__.__name__ == "YOLOLayer"]
    print(f"Found {len(yolo_layers)} YOLO layers")

    print(f"\n{'=' * 60}")
    print("Preprocessing image...")
    print("=" * 60)
    img_tensor, orig_size, resized_size = preprocess_image(IMAGE_PATH, IMG_SIZE)
    print(f"Image: {IMAGE_PATH}")
    print(f"Original: {orig_size}  Resized: {resized_size}  Input: {img_tensor.shape}")

    print(f"\n{'=' * 60}")
    print("Hooking YOLO layers...")
    print("=" * 60)
    captured, hooks = hook_yolo_layers(model)

    print(f"\n{'=' * 60}")
    print("Running forward pass...")
    print("=" * 60)
    with torch.no_grad():
        output = model(img_tensor)
    print(f"Final output shape: {output.shape}")

    print(f"\nCaptured keys: {list(captured.keys())}")
    for k, v in captured.items():
        print(f"  {k}: {v.shape}")

    print(f"\n{'=' * 60}")
    print("Parsing each scale for person (class 0) peak...")
    print("=" * 60)

    results = []
    for i in range(len(yolo_layers)):
        in_key = f"yolo_{i}_input"
        if in_key not in captured:
            print(f"  {in_key} not captured, skipping")
            continue
        r = parse_yolo_output(captured[in_key], yolo_layers[i], i, IMG_SIZE)
        results.append(r)

        print(f"\n  Scale {i+1} (stride {r['stride']}, {r['W']}x{r['H']}):")
        print(f"    Peak grid cell: ({r['x_grid']:.0f}, {r['y_grid']:.0f})")
        print(f"    Anchor: #{r['anchor_idx']}  wh={r['anchor_wh']}")
        print(f"    Raw tx={r['raw_tx']:.4f}  ty={r['raw_ty']:.4f}")
        print(f"    Sig tx={r['sig_tx']:.4f}  ty={r['sig_ty']:.4f}")
        print(f"    Decoded center (image): ({r['bx_image']:.1f}, {r['by_image']:.1f})")
        print(f"    Person score: {r['person_score']:.6f}")
        print(f"    --- Pythagorean ---")
        print(f"    Grid:   x^2 + y^2 = {r['c_sq_grid']:.0f}  z = {r['z_grid']:.2f}")
        print(f"    Image:  bx^2 + by^2 = {r['c_sq_image']:.0f}  z = {r['z_image']:.2f}")

    for h in hooks:
        h.remove()

    if not results:
        print("No scale outputs parsed!")
        return

    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print("=" * 60)
    plot_triangles(results, os.path.join(OUTPUT_DIR, "triangles_3scales.png"))
    plot_z_bar(results, os.path.join(OUTPUT_DIR, "z_3scales.png"))
    plot_heatmaps(results, os.path.join(OUTPUT_DIR, "heatmaps_3scales.png"))
    save_csv(results, os.path.join(OUTPUT_DIR, "detections_3scales.csv"))

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
