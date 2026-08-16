"""
Run YOLOv3 on a COCO person image, extract class-0 (person) detections,
and visualize the Pythagorean triangles: x^2 + y^2 = c^2, z = sqrt(c^2).

For each person detection's bounding box center (x, y):
  - Draw the right triangle: origin (0,0) -> (x, 0) -> (x, y) -> origin
  - Compute c^2 = x^2 + y^2
  - Compute z = sqrt(c^2) = distance from origin to (x, y)
  - Annotate z on the hypotenuse

Saves figure to outputs_clothing/pythagorean_detections.png
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ultralytics import YOLO

# ----------------------------- CONFIG -----------------------------
MODEL_PATH = r"C:\Users\carso\Desktop\YODO\YOLOv3\yolov3u.pt"
IMAGE_PATH = r"C:\Users\carso\Desktop\YODO\data\coco_person\images\000000000036.jpg"
OUTPUT_DIR = r"C:\Users\carso\Desktop\YODO\outputs_clothing"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "pythagorean_detections.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pythagorean_detections.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------


def run_detection():
    """Run YOLOv3 inference, return list of person detections with center coords."""
    model = YOLO(MODEL_PATH)
    results = model.predict(IMAGE_PATH, verbose=False, device=DEVICE)
    r = results[0]

    print(f"Model classes: {model.names}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Image size: {r.orig_shape}")
    print(f"Total detections: {len(r.boxes)}\n")

    persons = []
    for b in r.boxes:
        cid = int(b.cls)
        if cid != 0:  # class 0 = person
            continue
        conf = float(b.conf)
        xywh = b.xywh[0].tolist()  # [cx, cy, w, h]
        x, y, w, h = xywh
        persons.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
        })
        print(f"  person  conf={conf:.3f}  center=({x:.1f}, {y:.1f})  wh=({w:.1f}, {h:.1f})")

    return persons, r.orig_shape


def compute_pythagorean(persons):
    """For each detection: c^2 = x^2 + y^2, z = sqrt(c^2)."""
    rows = []
    for p in persons:
        x, y = p["x"], p["y"]
        c_sq = x ** 2 + y ** 2
        z = np.sqrt(c_sq)
        rows.append({
            "x": x,
            "y": y,
            "w": p["w"],
            "h": p["h"],
            "conf": p["conf"],
            "c_squared": c_sq,
            "z": z,
        })
        print(f"  x={x:.1f}  y={y:.1f}  c^2={c_sq:.1f}  z=sqrt(c^2)={z:.1f}")
    return rows


def plot_triangles(rows, img_shape, out_path):
    """Plot the image with Pythagorean triangles drawn from origin to each detection center."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # --- Left panel: triangles overlaid on image ---
    ax = axes[0]
    img_h, img_w = img_shape
    ax.set_xlim(-img_w * 0.05, img_w * 1.05)
    ax.set_ylim(img_h * 1.05, -img_h * 0.05)  # flip y to match image coords
    ax.set_aspect("equal")
    ax.set_title("Pythagorean Triangles: origin -> (x, 0) -> (x, y) -> origin", fontsize=13)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))

    for i, r in enumerate(rows):
        x, y, z = r["x"], r["y"], r["z"]
        c = colors[i]

        # Draw the three sides of the right triangle
        # Side 1: origin (0,0) -> (x, 0) — the "x" leg
        ax.plot([0, x], [0, 0], color=c, linewidth=2, linestyle="-")
        # Side 2: (x, 0) -> (x, y) — the "y" leg
        ax.plot([x, x], [0, y], color=c, linewidth=2, linestyle="-")
        # Side 3: (x, y) -> origin (0,0) — the hypotenuse (z)
        ax.plot([x, 0], [y, 0], color=c, linewidth=2.5, linestyle="--")

        # Mark the detection center
        ax.scatter(x, y, color=c, s=60, zorder=5, edgecolors="black", linewidths=0.5)

        # Annotate z on the hypotenuse midpoint
        mid_x, mid_y = x / 2, y / 2
        ax.annotate(
            f"z={z:.0f}",
            xy=(mid_x, mid_y),
            fontsize=9,
            color=c,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=c),
        )

        # Annotate x and y on the legs
        ax.text(x / 2, -img_h * 0.02, f"x={x:.0f}", color=c, fontsize=8, ha="center")
        ax.text(x + img_w * 0.01, y / 2, f"y={y:.0f}", color=c, fontsize=8, va="center")

    # Mark origin
    ax.scatter(0, 0, color="red", s=80, marker="*", zorder=5, label="origin (0,0)")
    ax.legend(loc="upper right")

    # --- Right panel: z values bar chart ---
    ax2 = axes[1]
    z_vals = [r["z"] for r in rows]
    labels = [f"det {i+1}\n({r['x']:.0f},{r['y']:.0f})" for i, r in enumerate(rows)]
    bars = ax2.bar(range(len(z_vals)), z_vals, color=colors[:len(z_vals)], edgecolor="black")
    ax2.set_xticks(range(len(z_vals)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("z = sqrt(x^2 + y^2)  (pixels from origin)")
    ax2.set_title("z values: distance from origin to each person detection center", fontsize=13)
    for bar, z in zip(bars, z_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"z={z:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def save_csv(rows, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "w", "h", "conf", "c_squared", "z"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved CSV: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Running YOLOv3 detection for class 0 (person)...")
    print("=" * 60)
    persons, img_shape = run_detection()

    if not persons:
        print("No person detections found. Try another image.")
        return

    print(f"\n{'=' * 60}")
    print(f"Pythagorean computation: c^2 = x^2 + y^2, z = sqrt(c^2)")
    print(f"{'=' * 60}")
    rows = compute_pythagorean(persons)

    print(f"\n{'=' * 60}")
    print(f"Plotting triangles and z values...")
    print(f"{'=' * 60}")
    plot_triangles(rows, img_shape, OUTPUT_FIG)
    save_csv(rows, OUTPUT_CSV)

    print(f"\nDone.")
    print(f"  Figure: {OUTPUT_FIG}")
    print(f"  CSV:    {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
