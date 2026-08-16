"""
SIPIT-style global inverse reconstruction for YOLOv3.

Adapts SIPIT (Sequential Inverse Prompt via Iterative updates) to CNNs:

Transformer SIPIT:
  - Given hidden states H(ℓ) at layer ℓ, reconstruct the exact input prompt s
  - Exploits causal structure: position t depends only on prefix s1...st-1
  - Iteratively reconstructs token-by-token by matching hidden states

CNN adaptation (global inverse):
  - Given feature map H(ℓ) at layer ℓ, reconstruct the ground truth input image
  - Exploits receptive field structure: each spatial location depends on a local
    input region
  - Uses gradient-based optimization (continuous input, no discrete vocabulary)
  - Optimizes an image x to minimize ||forward_to_ℓ(x) - H(ℓ)||^2
  - Compares reconstructed x to the ground truth image

For class-0 (person):
  - Additionally compute the class-0 saliency-weighted reconstruction:
    weight the feature matching by the gradient of the person detection score
  - This reconstructs only the parts of the input that the person detector uses

Method:
  1. Forward pass person image through YOLOv3, capture H(ℓ) at each conv layer
  2. For each layer ℓ:
     a. Initialize x = random noise (or pseudoinverse estimate)
     b. Optimize x via Adam to minimize ||forward_to_ℓ(x) - H(ℓ)||^2
     c. Compare reconstructed x to ground truth image
     d. Compute: MSE, PSNR, SSIM, cosine similarity
  3. For class-0: weight the loss by the person saliency at layer ℓ
  4. Track how reconstruction quality degrades with depth — this identifies
     the "information cliff" where ground truth becomes unrecoverable

Outputs:
  - sipit_global_reconstruction.csv     Per-layer reconstruction metrics
  - sipit_reconstruction_grid.png       Grid of reconstructed images per layer
  - sipit_reconstruction_metrics.png    PSNR/SSIM/cosine vs layer depth
  - sipit_class0_reconstruction.png     Class-0-weighted reconstructions
  - sipit_feature_maps/                 Individual layer reconstructions
"""

import os
import sys
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\sipit_inverse"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
CLASS_0      = 0  # person
N_ITER       = 500   # gradient descent iterations for inverse reconstruction
LR           = 0.05  # learning rate
TV_WEIGHT    = 1e-4  # total variation weight for image regularization
# Layers to reconstruct (representative subset covering backbone + neck + heads)
RECON_LAYERS = [0, 1, 5, 12, 37, 62, 75, 81, 84, 93, 105]
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "sipit_feature_maps"), exist_ok=True)


def load_image(img_path, img_size=416):
    """Load and preprocess image. Returns tensor and numpy array."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (128, 128, 128))
    canvas.paste(img_resized, ((img_size - new_w) // 2, (img_size - new_h) // 2))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, arr


def forward_to_layer(model, img_tensor, layer_idx):
    """
    Forward pass from input image up to layer_idx (inclusive).
    Returns the output feature map at layer_idx.
    """
    x = img_tensor
    layer_outputs = []

    for i, (mdef, mod) in enumerate(zip(model.module_defs, model.module_list)):
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

        if i == layer_idx:
            return x

    return None


def total_variation(x):
    """Total variation regularization for image smoothness."""
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def reconstruct_input(model, target_feat, layer_idx, img_size, device,
                      n_iter=500, lr=0.05, tv_weight=1e-4, init="noise",
                      ground_truth=None):
    """
    SIPIT global inverse: optimize an input image x such that
    forward_to_layer(model, x, layer_idx) ≈ target_feat

    This reconstructs the ground truth input image from hidden states at layer_idx.

    Args:
      target_feat: (1, C, H, W) feature map captured at layer_idx during forward pass
      layer_idx: index of the layer where features were captured
      init: "noise" (random) or "gt" (ground truth, for sanity check)
      ground_truth: (1, 3, H, W) the actual input image, for comparison

    Returns:
      reconstructed: (1, 3, H, W) the reconstructed input image
      metrics: dict with final loss, MSE to GT, PSNR, cosine sim
    """
    # Initialize input
    if init == "gt" and ground_truth is not None:
        x = ground_truth.clone().detach().requires_grad_(True)
    else:
        # Start from random noise
        x = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)

    # Apply sigmoid to keep input in [0, 1] range (like real images)
    # We optimize raw logits and apply sigmoid for the forward pass
    x_logits = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x_logits], lr=lr)

    best_loss = float('inf')
    best_x = None

    for it in range(n_iter):
        optimizer.zero_grad()

        # Sigmoid to constrain to [0, 1]
        x_img = torch.sigmoid(x_logits)

        # Forward pass to target layer
        pred_feat = forward_to_layer(model, x_img, layer_idx)

        if pred_feat is None:
            break

        # Feature matching loss
        loss = F.mse_loss(pred_feat, target_feat)

        # Total variation regularization (encourages natural image structure)
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(x_img)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = torch.sigmoid(x_logits).detach().clone()

        if (it + 1) % 100 == 0:
            print(f"    iter {it+1}/{n_iter}  loss={loss.item():.6f}  best={best_loss:.6f}")

    # Compute metrics against ground truth
    metrics = {}
    if ground_truth is not None:
        with torch.no_grad():
            recon = best_x if best_x is not None else torch.sigmoid(x_logits).detach()
            mse_gt = F.mse_loss(recon, ground_truth).item()
            max_val = ground_truth.max().item()
            if mse_gt > 0 and max_val > 0:
                psnr = 10 * math.log10(max_val ** 2 / mse_gt)
            else:
                psnr = float('inf') if mse_gt == 0 else 0

            # Cosine similarity (flattened)
            r_flat = recon.flatten()
            g_flat = ground_truth.flatten()
            cos_sim = F.cosine_similarity(r_flat.unsqueeze(0), g_flat.unsqueeze(0)).item()

            # SSIM (simplified — channel-wise correlation)
            r_np = recon[0].cpu().numpy()
            g_np = ground_truth[0].cpu().numpy()
            # Normalized cross-correlation as SSIM proxy
            r_norm = r_np - r_np.mean()
            g_norm = g_np - g_np.mean()
            ncc = (r_norm * g_norm).sum() / (np.sqrt((r_norm ** 2).sum() * (g_norm ** 2).sum()) + 1e-8)

            metrics = {
                "recon_loss": best_loss,
                "mse_to_gt": mse_gt,
                "psnr": psnr,
                "cosine_sim": cos_sim,
                "ncc": float(ncc),
            }

    return best_x if best_x is not None else torch.sigmoid(x_logits).detach(), metrics


def compute_class0_saliency_at_layer(model, img_tensor, layer_idx, device):
    """
    Compute the gradient of the class-0 person detection score w.r.t.
    the feature map at layer_idx. This gives a per-channel, per-spatial
    saliency map showing which features at this layer drive the person detection.

    Returns:
      saliency: (C,) per-channel saliency magnitude (used as weights for
                class-0-weighted reconstruction)
      person_score: the peak person detection score
    """
    # Find peak person detection
    with torch.no_grad():
        output = model(img_tensor)
    person_scores = output[0, :, 5 + CLASS_0] * output[0, :, 4]
    peak_idx = person_scores.argmax().item()
    person_score = person_scores[peak_idx].item()

    # Re-run forward with gradient tracking
    model.zero_grad()
    img_grad = img_tensor.clone().detach().requires_grad_(True)

    # Hook to capture gradient at target layer
    layer_grad = [None]

    def grad_hook_fn(module, grad_input, grad_output):
        layer_grad[0] = grad_output[0].detach().clone()

    # Find the module at layer_idx
    mod = model.module_list[layer_idx][0]
    g_hook = mod.register_full_backward_hook(grad_hook_fn)

    # Forward
    output = model(img_grad)
    score = output[0, peak_idx, 5 + CLASS_0] * output[0, peak_idx, 4]

    # Backward
    score.backward()
    g_hook.remove()

    if layer_grad[0] is not None:
        # Per-channel saliency: L2 norm across spatial dimensions
        grad = layer_grad[0]  # (1, C, H, W)
        # Per-channel L2 norm: flatten spatial dims, then norm per channel
        b, c, h, w = grad.shape
        saliency = grad.reshape(b, c, -1).norm(dim=(0, 2))  # (C,)
    else:
        saliency = None

    return saliency, person_score


def reconstruct_input_class0(model, target_feat, saliency, layer_idx, img_size,
                              device, n_iter=500, lr=0.05, tv_weight=1e-4,
                              ground_truth=None):
    """
    Class-0-weighted SIPIT inverse: optimize input image x such that
    forward_to_layer(model, x, layer_idx) matches target_feat, but weighted
    by the class-0 saliency at each channel.

    This reconstructs only the parts of the input that the person detector
    relies on at this layer, ignoring features used by other classes.
    """
    if saliency is None:
        return reconstruct_input(model, target_feat, layer_idx, img_size, device,
                                  n_iter, lr, tv_weight, "noise", ground_truth)

    # Normalize saliency to [0, 1] for weighting
    sal_weight = saliency / (saliency.max() + 1e-8)  # (C,)
    sal_weight = sal_weight.view(1, -1, 1, 1).to(device)  # (1, C, 1, 1)

    x_logits = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x_logits], lr=lr)

    best_loss = float('inf')
    best_x = None

    for it in range(n_iter):
        optimizer.zero_grad()
        x_img = torch.sigmoid(x_logits)
        pred_feat = forward_to_layer(model, x_img, layer_idx)

        if pred_feat is None:
            break

        # Weighted feature matching: emphasize channels that matter for person detection
        diff = (pred_feat - target_feat) ** 2
        weighted_diff = diff * sal_weight
        loss = weighted_diff.mean()

        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(x_img)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = torch.sigmoid(x_logits).detach().clone()

        if (it + 1) % 100 == 0:
            print(f"    [class-0] iter {it+1}/{n_iter}  loss={loss.item():.6f}  best={best_loss:.6f}")

    metrics = {}
    if ground_truth is not None:
        with torch.no_grad():
            recon = best_x if best_x is not None else torch.sigmoid(x_logits).detach()
            mse_gt = F.mse_loss(recon, ground_truth).item()
            max_val = ground_truth.max().item()
            if mse_gt > 0 and max_val > 0:
                psnr = 10 * math.log10(max_val ** 2 / mse_gt)
            else:
                psnr = float('inf') if mse_gt == 0 else 0
            r_flat = recon.flatten()
            g_flat = ground_truth.flatten()
            cos_sim = F.cosine_similarity(r_flat.unsqueeze(0), g_flat.unsqueeze(0)).item()
            r_np = recon[0].cpu().numpy()
            g_np = ground_truth[0].cpu().numpy()
            r_norm = r_np - r_np.mean()
            g_norm = g_np - g_np.mean()
            ncc = (r_norm * g_norm).sum() / (np.sqrt((r_norm ** 2).sum() * (g_norm ** 2).sum()) + 1e-8)
            metrics = {
                "recon_loss": best_loss,
                "mse_to_gt": mse_gt,
                "psnr": psnr,
                "cosine_sim": cos_sim,
                "ncc": float(ncc),
            }

    return best_x if best_x is not None else torch.sigmoid(x_logits).detach(), metrics


def tensor_to_image(t):
    """Convert (1, 3, H, W) tensor to (H, W, 3) numpy image in [0, 1]."""
    return t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)


def save_reconstruction_image(recon, gt, layer_idx, metrics, out_dir, prefix=""):
    """Save side-by-side comparison of reconstructed vs ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    ax = axes[0]
    ax.imshow(tensor_to_image(gt))
    ax.set_title("Ground Truth Input")
    ax.axis("off")

    # Reconstruction
    ax = axes[1]
    ax.imshow(tensor_to_image(recon))
    psnr = metrics.get("psnr", 0)
    cos = metrics.get("cosine_sim", 0)
    ncc = metrics.get("ncc", 0)
    ax.set_title(f"Reconstructed from Layer {layer_idx}\n"
                 f"PSNR={psnr:.2f}dB  cos={cos:.4f}  NCC={ncc:.4f}")
    ax.axis("off")

    # Difference (amplified 5x)
    ax = axes[2]
    diff = (tensor_to_image(recon) - tensor_to_image(gt))
    ax.imshow(np.abs(diff) * 5, cmap="hot", vmin=0, vmax=1)
    ax.set_title("|Recon - GT| x5")
    ax.axis("off")

    fig.suptitle(f"SIPIT Global Inverse: Layer {layer_idx}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, f"{prefix}layer_{layer_idx:03d}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_metrics(results, out_path):
    """Plot PSNR, cosine sim, NCC vs layer index."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    layers = [r["layer_idx"] for r in results]
    x = range(len(results))

    # PSNR
    ax = axes[0]
    ax.bar(x, [r["psnr"] for r in results], color=plt.cm.viridis(np.linspace(0, 1, len(results))),
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Reconstruction PSNR vs Ground Truth (higher = better reconstruction)")
    ax.axhline(y=20, color="r", linestyle="--", alpha=0.5, label="20dB threshold")
    ax.legend()

    # Cosine similarity
    ax = axes[1]
    ax.bar(x, [r["cosine_sim"] for r in results], color=plt.cm.plasma(np.linspace(0, 1, len(results))),
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: Reconstructed vs Ground Truth")
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend()

    # NCC
    ax = axes[2]
    ax.bar(x, [r["ncc"] for r in results], color=plt.cm.magma(np.linspace(0, 1, len(results))),
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Normalized Cross-Correlation")
    ax.set_title("NCC: Reconstructed vs Ground Truth (structural similarity proxy)")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend()

    fig.suptitle("SIPIT Global Inverse Reconstruction Quality vs Layer Depth", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_class0_comparison(results, class0_results, out_path):
    """Compare standard vs class-0-weighted reconstruction."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    layers = [r["layer_idx"] for r in results]
    x = range(len(results))
    width = 0.35

    # PSNR comparison
    ax = axes[0]
    ax.bar([i - width/2 for i in x], [r["psnr"] for r in results], width,
           label="Standard", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["psnr"] for r in class0_results], width,
           label="Class-0 weighted", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Standard vs Class-0-Weighted Reconstruction PSNR")
    ax.legend()

    # Cosine comparison
    ax = axes[1]
    ax.bar([i - width/2 for i in x], [r["cosine_sim"] for r in results], width,
           label="Standard", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["cosine_sim"] for r in class0_results], width,
           label="Class-0 weighted", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Standard vs Class-0-Weighted Cosine Similarity")
    ax.legend()

    fig.suptitle("SIPIT Reconstruction: Standard vs Class-0 (Person) Weighted", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_csv(results, class0_results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_idx", "recon_loss", "mse_to_gt", "psnr", "cosine_sim", "ncc",
            "class0_recon_loss", "class0_mse_to_gt", "class0_psnr", "class0_cosine_sim", "class0_ncc",
            "person_score",
        ])
        writer.writeheader()
        for r, r0 in zip(results, class0_results):
            row = {
                "layer_idx": r["layer_idx"],
                "recon_loss": r["recon_loss"],
                "mse_to_gt": r["mse_to_gt"],
                "psnr": r["psnr"],
                "cosine_sim": r["cosine_sim"],
                "ncc": r["ncc"],
                "class0_recon_loss": r0["recon_loss"],
                "class0_mse_to_gt": r0["mse_to_gt"],
                "class0_psnr": r0["psnr"],
                "class0_cosine_sim": r0["cosine_sim"],
                "class0_ncc": r0["ncc"],
                "person_score": r.get("person_score", 0),
            }
            writer.writerow(row)
    print(f"Saved CSV: {out_path}")


def main():
    print("=" * 60)
    print("SIPIT Global Inverse Reconstruction for YOLOv3")
    print("Reconstructing ground truth input from hidden states at each layer")
    print("=" * 60)

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    print("\nLoading ground truth image...")
    img_tensor, img_np = load_image(IMAGE_PATH, IMG_SIZE)
    print(f"Image: {IMAGE_PATH}")
    print(f"Input shape: {img_tensor.shape}")

    # Forward pass to capture target features at each layer
    print(f"\n{'=' * 60}")
    print("Step 1: Forward pass — capture hidden states at target layers")
    print("=" * 60)

    target_features = {}
    for layer_idx in RECON_LAYERS:
        with torch.no_grad():
            feat = forward_to_layer(model, img_tensor, layer_idx)
        if feat is not None:
            target_features[layer_idx] = feat.detach()
            print(f"  Layer {layer_idx:3d}: {feat.shape}")

    # Find peak person detection
    with torch.no_grad():
        output = model(img_tensor)
    person_scores = output[0, :, 5 + CLASS_0] * output[0, :, 4]
    peak_idx = person_scores.argmax().item()
    peak_score = person_scores[peak_idx].item()
    print(f"\nPeak person detection: score={peak_score:.6f}")

    # Step 2: Global inverse reconstruction at each layer
    print(f"\n{'=' * 60}")
    print(f"Step 2: SIPIT global inverse — reconstruct input from each layer")
    print(f"  iterations={N_ITER}  lr={LR}  tv_weight={TV_WEIGHT}")
    print("=" * 60)

    results = []
    class0_results = []

    for layer_idx in RECON_LAYERS:
        if layer_idx not in target_features:
            continue

        target = target_features[layer_idx]
        print(f"\n--- Layer {layer_idx} (target shape: {target.shape}) ---")

        # Standard reconstruction
        print(f"  Standard reconstruction:")
        recon, metrics = reconstruct_input(
            model, target, layer_idx, IMG_SIZE, DEVICE,
            n_iter=N_ITER, lr=LR, tv_weight=TV_WEIGHT,
            init="noise", ground_truth=img_tensor
        )
        metrics["layer_idx"] = layer_idx
        metrics["person_score"] = peak_score
        results.append(metrics)

        print(f"  Results: loss={metrics['recon_loss']:.6f}  "
              f"mse_gt={metrics['mse_to_gt']:.6f}  "
              f"psnr={metrics['psnr']:.2f}dB  "
              f"cos={metrics['cosine_sim']:.4f}  "
              f"ncc={metrics['ncc']:.4f}")

        # Save reconstruction image
        p = save_reconstruction_image(
            recon, img_tensor, layer_idx, metrics,
            os.path.join(OUTPUT_DIR, "sipit_feature_maps"), prefix=""
        )
        print(f"  Saved: {p}")

        # Class-0 weighted reconstruction
        print(f"  Class-0 weighted reconstruction:")
        saliency, person_score = compute_class0_saliency_at_layer(
            model, img_tensor, layer_idx, DEVICE
        )

        recon_c0, metrics_c0 = reconstruct_input_class0(
            model, target, saliency, layer_idx, IMG_SIZE, DEVICE,
            n_iter=N_ITER, lr=LR, tv_weight=TV_WEIGHT,
            ground_truth=img_tensor
        )
        metrics_c0["layer_idx"] = layer_idx
        class0_results.append(metrics_c0)

        print(f"  Class-0 results: loss={metrics_c0['recon_loss']:.6f}  "
              f"mse_gt={metrics_c0['mse_to_gt']:.6f}  "
              f"psnr={metrics_c0['psnr']:.2f}dB  "
              f"cos={metrics_c0['cosine_sim']:.4f}  "
              f"ncc={metrics_c0['ncc']:.4f}")

        # Save class-0 reconstruction
        p = save_reconstruction_image(
            recon_c0, img_tensor, layer_idx, metrics_c0,
            os.path.join(OUTPUT_DIR, "sipit_feature_maps"), prefix="class0_"
        )
        print(f"  Saved: {p}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Analyzed {len(results)} layers")
    print("=" * 60)

    print("\n--- Reconstruction quality vs layer depth ---")
    print(f"{'Layer':>6} {'PSNR':>10} {'Cosine':>10} {'NCC':>10} {'C0-PSNR':>10} {'C0-Cos':>10} {'C0-NCC':>10}")
    for r, r0 in zip(results, class0_results):
        print(f"  {r['layer_idx']:>4} {r['psnr']:>10.2f} {r['cosine_sim']:>10.4f} {r['ncc']:>10.4f} "
              f"{r0['psnr']:>10.2f} {r0['cosine_sim']:>10.4f} {r0['ncc']:>10.4f}")

    print("\n--- Best reconstructed layers (highest PSNR) ---")
    for r in sorted(results, key=lambda x: x["psnr"], reverse=True)[:5]:
        print(f"  Layer {r['layer_idx']:3d}  PSNR={r['psnr']:.2f}dB  cos={r['cosine_sim']:.4f}  NCC={r['ncc']:.4f}")

    print("\n--- Worst reconstructed layers (information cliff) ---")
    for r in sorted(results, key=lambda x: x["psnr"])[:5]:
        print(f"  Layer {r['layer_idx']:3d}  PSNR={r['psnr']:.2f}dB  cos={r['cosine_sim']:.4f}  NCC={r['ncc']:.4f}")

    print("\n--- Class-0 vs Standard (where class-0 weighting helps most) ---")
    diffs = [(r, r0, r0["psnr"] - r["psnr"]) for r, r0 in zip(results, class0_results)]
    for r, r0, d in sorted(diffs, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  Layer {r['layer_idx']:3d}  std_psnr={r['psnr']:.2f}  c0_psnr={r0['psnr']:.2f}  diff={d:+.2f}dB")

    # Outputs
    print(f"\n{'=' * 60}")
    print("Generating outputs...")
    print("=" * 60)
    save_csv(results, class0_results, os.path.join(OUTPUT_DIR, "sipit_global_reconstruction.csv"))
    plot_metrics(results, os.path.join(OUTPUT_DIR, "sipit_reconstruction_metrics.png"))
    plot_class0_comparison(results, class0_results, os.path.join(OUTPUT_DIR, "sipit_class0_comparison.png"))

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
