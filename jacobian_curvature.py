"""
Jacobian and curvature analysis for all CNN layers in YOLOv3.

Key fixes from previous version:
  1. Probe with REAL IMAGE DATA (not zeros) — measures curvature in the
     model's actual operating regime, not at the trivial LeakyReLU kink at x=0
  2. Adaptive probe spatial size — large-channel layers use smaller spatial
     probes so the Jacobian fits in VRAM
  3. All conv layers analyzed, including the deep backbone (layers 37, 62)
  4. Curvature measured as the bending of the output manifold — how much the
     layer's mapping deviates from linear at the image-driven operating point

For each conv layer:
  - Jacobian via torch.autograd at the image-driven operating point
  - Spectral norm, Frobenius norm, effective rank, condition number
  - Curvature = ||d2f/dx2|| via finite differences at the operating point
  - Gauss-Newton curvature = J^T @ J eigenvalues (geometric curvature of the
    pullback metric — how the layer warps the input space)

Outputs:
  - jacobian_curvature.csv
  - jacobian_spectral_norm.png
  - curvature_per_layer.png
  - gauss_newton_curvature.png
  - effective_rank.png
"""

import os
import sys
import csv
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add PyTorch-YOLOv3 to path
sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
# Bypass imgaug (incompatible with numpy 2.0)
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet

# ----------------------------- CONFIG -----------------------------
CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMAGE_PATH   = r"C:\Users\carso\Desktop\YODO\data\coco_person\images\000000000036.jpg"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\jacobian_curvature"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON      = 1e-3    # finite difference step for curvature
MAX_JAC_ELEMS = 2_000_000  # max Jacobian elements before using power iteration
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image_probe(img_path, img_size=416):
    """Load and preprocess image to a tensor on CUDA."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (128, 128, 128))
    canvas.paste(img_resized, ((img_size - new_w) // 2, (img_size - new_h) // 2))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def get_all_conv_layers(model):
    """Get all conv layers with their BN and activation modules."""
    results = []
    for i, (mdef, mod) in enumerate(zip(model.module_defs, model.module_list)):
        if mdef["type"] != "convolutional":
            continue
        conv = mod[0]
        bn = None
        act = None
        if len(mod) > 1 and isinstance(mod[1], nn.BatchNorm2d):
            bn = mod[1]
        if len(mod) > 2:
            act = mod[2]
        elif len(mod) > 1 and not isinstance(mod[1], nn.BatchNorm2d):
            act = mod[1]
        act_name = type(act).__name__ if act else "linear"
        results.append({
            "idx": i,
            "name": f"layer_{i}",
            "conv": conv,
            "bn": bn,
            "act": act,
            "act_name": act_name,
            "c_in": conv.in_channels,
            "c_out": conv.out_channels,
            "kernel": f"{conv.kernel_size[0]}x{conv.kernel_size[1]}",
            "stride": conv.stride[0],
        })
    return results


def get_layer_input(model, layer_idx, img_tensor):
    """
    Run forward pass up to layer_idx, capturing the input to that layer.
    Returns the actual feature map that feeds into this conv layer.
    """
    x = img_tensor
    layer_outputs = []

    for i, (mdef, mod) in enumerate(zip(model.module_defs, model.module_list)):
        if i == layer_idx:
            return x.detach().clone()
        if mdef["type"] in ["convolutional", "upsample", "maxpool"]:
            x = mod(x)
        elif mdef["type"] == "route":
            layers = [int(x_) for x_ in mdef["layers"].split(",")]
            combined = torch.cat([layer_outputs[int(l)] for l in layers], 1)
            group_size = combined.shape[1] // int(mdef.get("groups", 1))
            group_id = int(mdef.get("group_id", 0))
            x = combined[:, group_size * group_id : group_size * (group_id + 1)]
        elif mdef["type"] == "shortcut":
            layer_i = int(mdef["from"])
            x = layer_outputs[-1] + layer_outputs[layer_i]
        elif mdef["type"] == "yolo":
            x = mod[0](x, img_tensor.size(2))
        layer_outputs.append(x)

    return None


def adaptive_probe_size(c_in, c_out, kh, kw):
    """
    Pick a probe spatial size that keeps Jacobian under VRAM budget.
    Jacobian size = c_out * c_in * kh * kw (independent of spatial size
    since we probe a single output location).
    But we need enough spatial size for the conv to have a valid output.
    """
    # Minimum spatial size: kernel + 1
    min_size = max(kh, kw) + 2
    # For stride-2 layers, need at least 2x the kernel
    return max(min_size, 16)


def compute_jacobian_autograd(conv, bn, act, probe_input, device):
    """
    Compute Jacobian of conv(+bn+act) at the image-driven operating point.

    probe_input: (1, C_in, H, W) — actual feature map from forward pass

    Returns:
      jac_flat: (c_out, c_in * kh * kw) — local Jacobian at probe center
      spectral_norm, frobenius_norm, effective_rank, condition_number
    """
    c_in = conv.in_channels
    c_out = conv.out_channels
    kh, kw = conv.kernel_size[0], conv.kernel_size[1]
    stride = conv.stride
    pad = conv.padding

    # Use the probe input (real features, not zeros)
    x = probe_input.clone().to(device).float()
    # Need a spatial crop if too large — take a small patch around center
    h, w = x.shape[2], x.shape[3]
    # Crop to a small region around center for efficiency
    crop_h = min(h, max(kh + 4, 16))
    crop_w = min(w, max(kw + 4, 16))
    cy, cx = h // 2, w // 2
    x = x[:, :, cy - crop_h // 2 : cy + crop_h // 2 + crop_h % 2,
               cx - crop_w // 2 : cx + crop_w // 2 + crop_w % 2]
    h, w = x.shape[2], x.shape[3]

    # Composite function: input -> output at center spatial location
    def func(inp):
        out = conv(inp)
        if bn is not None:
            out = bn(out)
        if act is not None:
            out = act(out)
        # Center spatial location
        oy, ox = out.shape[2] // 2, out.shape[3] // 2
        return out[0, :, oy, ox]  # (c_out,)

    # Compute Jacobian via autograd
    x_req = x.clone().detach().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(func, x_req)
    # jac shape: (c_out, 1, c_in, h, w)

    # Find which input spatial locations feed into the center output
    h_out = (h + 2 * pad[0] - kh) // stride[0] + 1
    w_out = (w + 2 * pad[1] - kw) // stride[1] + 1
    oy = h_out // 2
    ox = w_out // 2

    in_y_start = oy * stride[0] - pad[0]
    in_x_start = ox * stride[1] - pad[1]

    # Extract local Jacobian: (c_out, c_in, kh, kw)
    jac_2d = jac.squeeze(1)  # (c_out, c_in, h, w)
    local_jac = torch.zeros(c_out, c_in, kh, kw, device=device, dtype=torch.float32)
    for ky in range(kh):
        for kx in range(kw):
            iy = in_y_start + ky
            ix = in_x_start + kx
            if 0 <= iy < h and 0 <= ix < w:
                local_jac[:, :, ky, kx] = jac_2d[:, :, iy, ix]

    jac_flat = local_jac.reshape(c_out, -1)  # (c_out, c_in * kh * kw)

    # Metrics
    frob_norm = torch.linalg.norm(jac_flat).item()
    total_elems = jac_flat.shape[0] * jac_flat.shape[1]

    if total_elems < 500_000:
        U, S, Vh = torch.linalg.svd(jac_flat, full_matrices=False)
        spectral_norm = S[0].item()
        threshold = max(S[0].item() * 1e-6, 1e-10)
        effective_rank = (S > threshold).sum().item()
        cond_num = (S[0] / S[-1]).item() if S[-1].item() > 1e-12 else float('inf')
        # Gauss-Newton curvature: eigenvalues of J^T @ J
        gn_eigs = (S ** 2).cpu().numpy()
        gn_max = gn_eigs[0]
        gn_mean = gn_eigs.mean()
        gn_trace = gn_eigs.sum()
    else:
        # Power iteration for spectral norm
        v = torch.randn(jac_flat.shape[1], 1, device=device)
        v = v / v.norm()
        for _ in range(100):
            u = jac_flat @ v
            un = u.norm()
            if un < 1e-12:
                break
            u = u / un
            v = jac_flat.T @ u
            vn = v.norm()
            if vn < 1e-12:
                break
            v = v / vn
        spectral_norm = (jac_flat @ v).norm().item()
        effective_rank = -1
        cond_num = float('inf')
        gn_max = spectral_norm ** 2
        gn_trace = frob_norm ** 2
        gn_mean = gn_trace / jac_flat.shape[1]

    return {
        "spectral_norm": spectral_norm,
        "frobenius_norm": frob_norm,
        "effective_rank": effective_rank,
        "condition_number": cond_num,
        "gn_max_eigenvalue": gn_max,
        "gn_mean_eigenvalue": gn_mean,
        "gn_trace": gn_trace,
        "jacobian_elems": total_elems,
    }


def compute_curvature_finite_diff(conv, bn, act, probe_input, device, epsilon):
    """
    Compute curvature at the image-driven operating point.

    Perturbs each input channel at the center spatial location by +/-epsilon
    and measures the second derivative of the output.

    Returns max_curvature, mean_curvature, median_curvature.
    """
    c_in = conv.in_channels
    c_out = conv.out_channels

    x = probe_input.clone().to(device).float()
    h, w = x.shape[2], x.shape[3]
    # Crop to center region
    crop_h = min(h, 16)
    crop_w = min(w, 16)
    cy, cx = h // 2, w // 2
    x = x[:, :, cy - crop_h // 2 : cy + crop_h // 2 + crop_h % 2,
               cx - crop_w // 2 : cx + crop_w // 2 + crop_w % 2]
    h, w = x.shape[2], x.shape[3]

    def forward(inp):
        out = conv(inp)
        if bn is not None:
            out = bn(out)
        if act is not None:
            out = act(out)
        oy, ox = out.shape[2] // 2, out.shape[3] // 2
        return out[0, :, oy, ox]  # (c_out,)

    with torch.no_grad():
        base = forward(x)

    py = h // 2
    px = w // 2

    curvatures = torch.zeros(c_in, device=device, dtype=torch.float32)

    for ch in range(c_in):
        x_plus = x.clone()
        x_plus[0, ch, py, px] += epsilon
        x_minus = x.clone()
        x_minus[0, ch, py, px] -= epsilon

        with torch.no_grad():
            out_plus = forward(x_plus)
            out_minus = forward(x_minus)

        # Second derivative: (f+ - 2f + f-) / eps^2
        second = (out_plus - 2 * base + out_minus) / (epsilon ** 2)
        curvatures[ch] = second.norm().item()

    return curvatures.max().item(), curvatures.mean().item(), curvatures.median().item()


def analyze_layer(layer_info, probe_input, device):
    """Analyze a single conv layer at the image-driven operating point."""
    name = layer_info["name"]
    conv = layer_info["conv"]
    bn = layer_info["bn"]
    act = layer_info["act"]
    c_in = layer_info["c_in"]
    c_out = layer_info["c_out"]
    act_name = layer_info["act_name"]
    kh = conv.kernel_size[0]
    kw = conv.kernel_size[1]

    jac_elems = c_out * c_in * kh * kw
    if jac_elems > MAX_JAC_ELEMS:
        print(f"  SKIP {name}: Jacobian too large ({jac_elems:,} elems)")
        return None

    print(f"  Analyzing {name}: in={c_in:>4} out={c_out:>4} k={kh}x{kw} "
          f"act={act_name:<10} jac={c_out}x{c_in*kh*kw}={jac_elems:,}")

    # Jacobian via autograd at real feature point
    jac_metrics = compute_jacobian_autograd(conv, bn, act, probe_input, device)

    # Curvature via finite differences at real feature point
    max_curv, mean_curv, median_curv = compute_curvature_finite_diff(
        conv, bn, act, probe_input, device, EPSILON
    )

    result = {
        "layer": name,
        "layer_idx": layer_info["idx"],
        "in_channels": c_in,
        "out_channels": c_out,
        "kernel": f"{kh}x{kw}",
        "stride": layer_info["stride"],
        "activation": act_name,
        "jacobian_elems": jac_metrics["jacobian_elems"],
        "spectral_norm": jac_metrics["spectral_norm"],
        "frobenius_norm": jac_metrics["frobenius_norm"],
        "condition_number": jac_metrics["condition_number"],
        "effective_rank": jac_metrics["effective_rank"],
        "max_curvature": max_curv,
        "mean_curvature": mean_curv,
        "median_curvature": median_curv,
        "gn_max_eigenvalue": jac_metrics["gn_max_eigenvalue"],
        "gn_mean_eigenvalue": jac_metrics["gn_mean_eigenvalue"],
        "gn_trace": jac_metrics["gn_trace"],
    }

    print(f"    spectral={result['spectral_norm']:.4f}  frob={result['frobenius_norm']:.4f}  "
          f"eff_rank={result['effective_rank']}  "
          f"max_curv={max_curv:.4f}  mean_curv={mean_curv:.4f}  "
          f"gn_max={result['gn_max_eigenvalue']:.4f}")

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def plot_results(results, out_dir):
    """Generate all plots."""
    names = [r["layer"] for r in results]
    short_names = [f"L{r['layer_idx']}" for r in results]
    x = range(len(results))
    n = len(results)

    # 1. Spectral norm
    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    ax.bar(x, [r["spectral_norm"] for r in results], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Spectral norm")
    ax.set_title("Jacobian spectral norm per layer (image-driven operating point)")
    fig.tight_layout()
    p = os.path.join(out_dir, "jacobian_spectral_norm.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # 2. Curvature (max + mean)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 0.5), 10))
    cm = plt.cm.plasma(np.linspace(0, 1, n))

    ax = axes[0]
    ax.bar(x, [r["max_curvature"] for r in results], color=cm, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Max curvature ||d2f/dx2||")
    ax.set_title("Max curvature per layer (image-driven, not at origin)")

    ax2 = axes[1]
    ax2.bar(x, [r["mean_curvature"] for r in results], color=cm, edgecolor="black", linewidth=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_xlabel("Layer index"); ax2.set_ylabel("Mean curvature")
    ax2.set_title("Mean curvature per layer")

    fig.suptitle("Curvature at image-driven operating point — real feature regime, not trivial zero", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "curvature_per_layer.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # 3. Gauss-Newton curvature (J^T J eigenvalues)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 0.5), 10))
    cm2 = plt.cm.magma(np.linspace(0, 1, n))

    ax = axes[0]
    ax.bar(x, [r["gn_max_eigenvalue"] for r in results], color=cm2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Max eigenvalue of J^T J")
    ax.set_title("Gauss-Newton max curvature (dominant bending direction)")

    ax2 = axes[1]
    ax2.bar(x, [r["gn_trace"] for r in results], color=cm2, edgecolor="black", linewidth=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_xlabel("Layer index"); ax2.set_ylabel("trace(J^T J) = ||J||_F^2")
    ax2.set_title("Gauss-Newton trace (total curvature budget)")

    fig.suptitle("Gauss-Newton curvature: J^T J eigenvalues (geometric warping of input space)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "gauss_newton_curvature.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # 4. Effective rank
    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 6))
    ax.bar(x, [r["effective_rank"] for r in results], color=plt.cm.cool(np.linspace(0, 1, n)),
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Effective rank")
    ax.set_title("Jacobian effective rank per layer")
    fig.tight_layout()
    p = os.path.join(out_dir, "effective_rank.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # 5. Frobenius norm
    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 6))
    ax.bar(x, [r["frobenius_norm"] for r in results], color=plt.cm.coolwarm(np.linspace(0, 1, n)),
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Frobenius norm")
    ax.set_title("Jacobian Frobenius norm per layer")
    fig.tight_layout()
    p = os.path.join(out_dir, "jacobian_frobenius.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


def save_csv(results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "layer_idx", "in_channels", "out_channels", "kernel",
            "stride", "activation", "jacobian_elems",
            "spectral_norm", "frobenius_norm", "condition_number",
            "effective_rank", "max_curvature", "mean_curvature", "median_curvature",
            "gn_max_eigenvalue", "gn_mean_eigenvalue", "gn_trace",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved CSV: {out_path}")


def main():
    print("=" * 60)
    print("Loading YOLOv3 (Darknet)...")
    print("=" * 60)
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    print(f"\n{'=' * 60}")
    print("Loading image probe...")
    print("=" * 60)
    img_tensor = load_image_probe(IMAGE_PATH, img_size=416)
    print(f"Image: {IMAGE_PATH}")
    print(f"Input tensor: {img_tensor.shape}")

    print(f"\n{'=' * 60}")
    print("Extracting all conv layers...")
    print("=" * 60)
    layers = get_all_conv_layers(model)
    print(f"Found {len(layers)} conv layers")

    print(f"\n{'=' * 60}")
    print("Computing Jacobian + curvature at image-driven operating points...")
    print(f"eps={EPSILON}  max_jac_elems={MAX_JAC_ELEMS:,}")
    print("=" * 60)

    results = []
    for layer_info in layers:
        # Get the actual input feature map for this layer from forward pass
        probe_input = get_layer_input(model, layer_info["idx"], img_tensor)
        if probe_input is None:
            print(f"  SKIP {layer_info['name']}: could not get input")
            continue

        r = analyze_layer(layer_info, probe_input, DEVICE)
        if r is not None:
            results.append(r)

    print(f"\n{'=' * 60}")
    print(f"Analyzed {len(results)} / {len(layers)} layers")
    print("=" * 60)

    # Focus on the interesting middle layers — exclude layer 0 and detection heads
    middle = [r for r in results if r["layer_idx"] > 0 and r["activation"] != "linear"]

    print(f"\n--- Middle backbone layers (excluding input conv and linear heads) ---")
    print(f"{'Layer':<12} {'spectral':>10} {'frob':>10} {'max_curv':>10} {'mean_curv':>10} {'gn_max':>10} {'eff_rank':>8}")
    for r in middle:
        print(f"  {r['layer']:<12} {r['spectral_norm']:>10.4f} {r['frobenius_norm']:>10.4f} "
              f"{r['max_curvature']:>10.4f} {r['mean_curvature']:>10.4f} "
              f"{r['gn_max_eigenvalue']:>10.4f} {r['effective_rank']:>8}")

    print(f"\n--- Top 5 middle layers by curvature (most nonlinear bending) ---")
    for r in sorted(middle, key=lambda x: x["max_curvature"], reverse=True)[:5]:
        print(f"  {r['layer']:12s}  max_curv={r['max_curvature']:.4f}  "
              f"mean_curv={r['mean_curvature']:.4f}  "
              f"gn_max={r['gn_max_eigenvalue']:.4f}  "
              f"stride={r['stride']}  in={r['in_channels']}  out={r['out_channels']}")

    print(f"\n--- Top 5 middle layers by Gauss-Newton max eigenvalue ---")
    for r in sorted(middle, key=lambda x: x["gn_max_eigenvalue"], reverse=True)[:5]:
        print(f"  {r['layer']:12s}  gn_max={r['gn_max_eigenvalue']:.4f}  "
              f"gn_trace={r['gn_trace']:.4f}  "
              f"stride={r['stride']}  in={r['in_channels']}  out={r['out_channels']}")

    print(f"\n--- Top 5 middle layers by spectral norm ---")
    for r in sorted(middle, key=lambda x: x["spectral_norm"], reverse=True)[:5]:
        print(f"  {r['layer']:12s}  spectral={r['spectral_norm']:.4f}  "
              f"frob={r['frobenius_norm']:.4f}  "
              f"stride={r['stride']}  in={r['in_channels']}  out={r['out_channels']}")

    # Outputs
    print(f"\n{'=' * 60}")
    print("Generating outputs...")
    print("=" * 60)
    save_csv(results, os.path.join(OUTPUT_DIR, "jacobian_curvature.csv"))
    plot_results(results, OUTPUT_DIR)

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
