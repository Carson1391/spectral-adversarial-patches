"""
Darknet-53 / YOLO backbone outlier-weight scanner.
Adapts the "super weight" question (LLM paper: arXiv 2411.07191) to conv nets.

Darknet-53 has NO softmax and NO down_proj/gate_proj/up_proj structure, so the
literal LLM mechanism (softmax attention sink -> SwiGLU Hadamard -> down_proj
amplification) does not exist here. There is nothing to "not find" in that
sense by definition, not by absence of effort.

What this script actually does instead, and what genuinely transfers:
  1. Loads every Conv2d / Linear / BatchNorm weight tensor in the checkpoint.
  2. For each tensor, computes a z-score for every element relative to that
     tensor's own mean/std (per-tensor, since conv weight scales differ layer
     to layer -- this mirrors the paper's "outlier relative to local
     distribution" logic rather than a global cutoff).
  3. Flags the single most extreme-z weight per layer AND the global top-K
     outliers across the whole network.
  4. Computes each tensor's excess kurtosis -- a model whose weights are
     "held up" by a few extreme values (the CNN analog of what the LLM paper
     found) will show heavy-tailed kurtosis, a normal well-trained tensor
     will not.
  5. Outputs: console summary, full CSV of ranked outliers, and two plots
     (max |weight| per layer across the network depth, and a kurtosis-per-
     layer bar chart) so you can see at a glance whether any layer looks like
     an LLM-style super-weight layer or whether the distribution is just
     ordinary CNN weight decay.

No architecture class is instantiated -- this reads the state_dict directly,
so it works whether your .pt is a raw state_dict, an Ultralytics YOLO
checkpoint ({'model': ...}), or a plain Darknet .pt. Point CHECKPOINT_PATH
at your real file and run.
"""

import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------
CHECKPOINT_PATH = r"C:\Users\carso\Desktop\YODO\YOLOv3\yolov3u.pt"  # EDIT ME
OUTPUT_DIR = r"C:\Users\carso\Desktop\modelspace\superweight_scan_out"
TOP_K_GLOBAL = 50          # how many global outliers to report/save
Z_FLAG_THRESHOLD = 8.0     # |z| above this is flagged as an outlier candidate
MIN_TENSOR_ELEMENTS = 64   # skip tiny tensors (e.g. single bias scalars)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------


def load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "model" in obj:
        model_obj = obj["model"]
        if hasattr(model_obj, "state_dict"):
            sd = model_obj.state_dict()
        elif isinstance(model_obj, dict):
            sd = model_obj
        else:
            raise ValueError(
                f"Unrecognized 'model' entry type in checkpoint: {type(model_obj)}"
            )
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")

    clean = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v
    return clean


def excess_kurtosis(x: torch.Tensor) -> float:
    x = x.double()
    mean = x.mean()
    std = x.std(unbiased=False)
    if std.item() == 0.0:
        return 0.0
    z = (x - mean) / std
    m4 = (z ** 4).mean().item()
    return m4 - 3.0  # excess kurtosis; 0 = Gaussian, large = heavy-tailed


def scan_checkpoint(path: str):
    print(f"Loading checkpoint: {path}")
    sd = load_state_dict(path)
    if not sd:
        raise RuntimeError("No tensors found in checkpoint -- check the path/format.")

    print(f"Found {len(sd)} tensors. Device for compute: {DEVICE}\n")

    layer_summaries = []   # one row per tensor: max|w|, z of max, kurtosis, shape
    global_outliers = []   # flattened candidate list across all tensors

    for name, tensor in sd.items():
        if tensor.numel() < MIN_TENSOR_ELEMENTS:
            continue
        if not torch.is_floating_point(tensor):
            continue

        t = tensor.to(DEVICE).float()
        flat = t.flatten()

        mean = flat.mean()
        std = flat.std(unbiased=False)
        if std.item() == 0.0:
            continue

        z = (flat - mean) / std
        abs_z = z.abs()

        max_idx = torch.argmax(abs_z).item()
        max_val = flat[max_idx].item()
        max_z = z[max_idx].item()

        kurt = excess_kurtosis(flat.cpu())

        coord = np.unravel_index(max_idx, tensor.shape)

        layer_summaries.append({
            "layer": name,
            "shape": tuple(tensor.shape),
            "numel": tensor.numel(),
            "max_abs_weight": max_val,
            "max_abs_weight_z": max_z,
            "excess_kurtosis": kurt,
            "coord_of_max": coord,
        })

        # collect candidates for the global top-K (only strongly flagged ones,
        # capped per-layer at 5 to avoid one huge tensor dominating the list)
        top_local = torch.topk(abs_z, k=min(5, abs_z.numel()))
        for rank, idx in enumerate(top_local.indices.tolist()):
            zval = z[idx].item()
            if abs(zval) >= Z_FLAG_THRESHOLD:
                c = np.unravel_index(idx, tensor.shape)
                global_outliers.append({
                    "layer": name,
                    "coord": c,
                    "weight_value": flat[idx].item(),
                    "z_score": zval,
                    "tensor_shape": tuple(tensor.shape),
                })

        del t, flat, z, abs_z
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    global_outliers.sort(key=lambda r: abs(r["z_score"]), reverse=True)
    return layer_summaries, global_outliers[:TOP_K_GLOBAL]


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote CSV: {path}")


def make_plots(layer_summaries, out_dir):
    names = [r["layer"] for r in layer_summaries]
    max_abs = [r["max_abs_weight"] for r in layer_summaries]
    max_z = [r["max_abs_weight_z"] for r in layer_summaries]
    kurt = [r["excess_kurtosis"] for r in layer_summaries]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.15), 5))
    ax.plot(x, max_abs, linewidth=1)
    ax.scatter(x, max_abs, s=10)
    ax.set_xlabel("Layer index (network depth order)")
    ax.set_ylabel("Max |weight| in tensor")
    ax.set_title("Darknet-53 checkpoint: max weight magnitude per layer")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "max_weight_per_layer.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {p1}")

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.15), 5))
    ax.bar(x, kurt, width=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer index (network depth order)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Darknet-53 checkpoint: weight-distribution kurtosis per layer\n"
                 "(near 0 = Gaussian-ish; large = a few weights dominate the tensor)")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "kurtosis_per_layer.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {p2}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    layer_summaries, top_outliers = scan_checkpoint(CHECKPOINT_PATH)

    print("=" * 70)
    print(f"Scanned {len(layer_summaries)} weight tensors.")
    print("=" * 70)

    print("\nTop 10 global outlier weights (by |z-score| within their own tensor):")
    for r in top_outliers[:10]:
        print(f"  z={r['z_score']:+.2f}  val={r['weight_value']:+.5f}  "
              f"layer={r['layer']}  coord={r['coord']}  shape={r['tensor_shape']}")

    layers_sorted_by_kurt = sorted(layer_summaries, key=lambda r: r["excess_kurtosis"], reverse=True)
    print("\nTop 5 layers by excess kurtosis (most 'peaky'/outlier-driven tensors):")
    for r in layers_sorted_by_kurt[:5]:
        print(f"  kurtosis={r['excess_kurtosis']:+.2f}  layer={r['layer']}  "
              f"shape={r['shape']}  max|w|={r['max_abs_weight']:+.5f} (z={r['max_abs_weight_z']:+.2f})")

    if not top_outliers:
        print(f"\nNo weights crossed |z| >= {Z_FLAG_THRESHOLD} anywhere in the network.")
        print("That is itself the result: this checkpoint does not show an LLM-style")
        print("super-weight signature -- weight magnitude is smoothly distributed,")
        print("consistent with Darknet-53 having no softmax/down_proj mechanism to")
        print("manufacture one. Lower Z_FLAG_THRESHOLD if you want to inspect milder")
        print("outliers instead of only extreme ones.")

    layer_csv = os.path.join(OUTPUT_DIR, "layer_summary.csv")
    write_csv(
        layer_summaries,
        layer_csv,
        fieldnames=["layer", "shape", "numel", "max_abs_weight",
                    "max_abs_weight_z", "excess_kurtosis", "coord_of_max"],
    )

    outlier_csv = os.path.join(OUTPUT_DIR, "top_global_outliers.csv")
    write_csv(
        top_outliers,
        outlier_csv,
        fieldnames=["layer", "coord", "weight_value", "z_score", "tensor_shape"],
    )

    make_plots(layer_summaries, OUTPUT_DIR)

    print("\nDone. All outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()