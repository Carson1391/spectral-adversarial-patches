"""
Combined L2 / FFT / Graph Laplacian / KFAC analysis for YOLOv3.

Four complementary analyses on the YOLOv3 Darknet model:

1. L2 SPATIAL ERROR MAP
   - Forward pass person image, capture features at each conv layer
   - Reconstruct input from each layer (SIPIT global inverse)
   - Compute per-pixel L2 error map: |reconstruction - ground_truth|
   - Shows WHERE in the image information is lost at each layer depth

2. FFT FREQUENCY ANALYSIS
   - 2D FFT of the reconstruction error at each layer
   - Measures which spatial frequencies survive vs get destroyed
   - Power spectrum ratio: |FFT(recon)| / |FFT(GT)| per frequency band
   - Identifies the frequency cutoff at each layer depth

3. GRAPH LAPLACIAN (layer-layer information flow)
   - Build adjacency graph between layers based on feature similarity
     (cosine similarity of flattened feature maps)
   - Compute graph Laplacian L = D - A
   - Eigendecomposition reveals community structure in information flow
   - Fiedler vector shows the main split in how information propagates
   - Also: per-channel graph Laplacian within each layer (channel communities)

4. KFAC (Kronecker-Factored Approximate Curvature)
   - For each conv layer, compute the Fisher Information Matrix approximated
     as A ⊗ G where:
       A = E[xx^T]  (input covariance, Kronecker factor 1)
       G = E[gg^T]  (gradient covariance, Kronecker factor 2)
     x = input activations (flattened per spatial location)
     g = gradients of class-0 loss w.r.t. output (flattened per spatial location)
   - This gives the REAL curvature of the loss landscape at each layer
   - Metrics: trace(A⊗G) = trace(A)*trace(G), max eigenvalue of A and G,
     condition number of A and G, effective rank

All computations on CUDA. Outputs to outputs_clothing/l2_fft_laplacian_kfac/
"""

import os
import sys
import csv
import math
import json
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\l2_fft_laplacian_kfac"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
CLASS_0      = 0  # person
N_ITER       = 300   # iterations for SIPIT inverse reconstruction
LR           = 0.05
TV_WEIGHT    = 1e-4
# Layers to analyze — key feature scales + high-curvature outliers from Jacobian analysis
# Outliers: 54 (deep backbone 3x3), 60 (pre-downsample 3x3, highest Frobenius),
#           63 (backbone-to-neck handoff, 3rd highest curvature), 92 (pre-26x26-head 3x3, 2nd highest Frobenius)
RECON_LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

# Adversarial patch relevance annotations for plots
ADV_ANNOTATIONS = {
    0:  "Input layer",
    1:  "1st downsample",
    5:  "2nd downsample",
    12: "3rd downsample",
    37: "4th downsample",
    54: "Deep backbone 3x3 (high curvature outlier)",
    60: "Pre-downsample 3x3 (highest Frobenius in model)",
    62: "5th downsample (deepest 3x3, highest KFAC trace)",
    63: "Backbone-to-neck handoff (3rd highest curvature)",
    75: "Neck 1x1 conv",
    81: "13x13 detection head (sharpest single direction)",
    84: "26x26 neck conv",
    92: "Pre-26x26-head 3x3 (2nd highest Frobenius)",
    93: "26x26 detection head",
    105: "52x52 detection head",
}
# KFAC: sample this many spatial locations per layer for covariance estimation
KFAC_SAMPLES = 256
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "l2_error_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fft_spectra"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "kfac_eigenvalues"), exist_ok=True)


def load_image(img_path, img_size=416):
    """Load and preprocess image."""
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
    """Forward pass from input up to layer_idx (inclusive)."""
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


def forward_capture_all(model, img_tensor):
    """Forward pass capturing input and output for every conv layer."""
    captured = {}
    layer_outputs = []
    x = img_tensor
    for i, (mdef, mod) in enumerate(zip(model.module_defs, model.module_list)):
        input_to_layer = x.detach().clone()
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
        if mdef["type"] == "convolutional":
            conv = mod[0]
            bn = None
            act = None
            if len(mod) > 1 and isinstance(mod[1], nn.BatchNorm2d):
                bn = mod[1]
            if len(mod) > 2:
                act = mod[2]
            elif len(mod) > 1 and not isinstance(mod[1], nn.BatchNorm2d):
                act = mod[1]
            captured[i] = {
                "input": input_to_layer,
                "output": x.detach().clone(),
                "conv": conv,
                "bn": bn,
                "act": act,
                "act_name": type(act).__name__ if act else "linear",
                "c_in": conv.in_channels,
                "c_out": conv.out_channels,
                "kernel": (conv.kernel_size[0], conv.kernel_size[1]),
                "stride": conv.stride[0],
            }
        layer_outputs.append(x)
    return captured


def total_variation(x):
    """TV regularization for image smoothness."""
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def reconstruct_input(model, target_feat, layer_idx, img_size, device,
                      n_iter=300, lr=0.05, tv_weight=1e-4):
    """SIPIT global inverse: optimize input image to match target features."""
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
        loss = F.mse_loss(pred_feat, target_feat)
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(x_img)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = torch.sigmoid(x_logits).detach().clone()
    return best_x if best_x is not None else torch.sigmoid(x_logits).detach(), best_loss


# ============================================================
# 1. L2 SPATIAL ERROR MAP
# ============================================================

def analyze_l2_error(recon, gt, layer_idx, out_dir):
    """
    Per-pixel L2 error map between reconstruction and ground truth.
    Also compute per-region statistics (center vs border, person vs background).
    """
    with torch.no_grad():
        # Per-pixel L2 error: (1, 3, H, W) -> (H, W)
        l2_map = (recon - gt).pow(2).sum(dim=1).sqrt()[0].cpu().numpy()

        # Per-channel L2
        l2_per_channel = (recon - gt).pow(2).mean(dim=(2, 3)).sqrt()[0].cpu().numpy()

        # Overall stats
        mean_l2 = l2_map.mean()
        max_l2 = l2_map.max()
        # Center region (where person typically is)
        h, w = l2_map.shape
        center_region = l2_map[h//4:3*h//4, w//4:3*w//4]
        border_region = np.concatenate([
            l2_map[:h//4].ravel(), l2_map[3*h//4:].ravel(),
            l2_map[h//4:3*h//4, :w//4].ravel(),
            l2_map[h//4:3*h//4, 3*w//4:].ravel()
        ])
        center_mean = center_region.mean()
        border_mean = border_region.mean()

    # Save error map visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(tensor_to_image(gt))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(tensor_to_image(recon))
    axes[1].set_title(f"Reconstructed (Layer {layer_idx})")
    axes[1].axis("off")
    im = axes[2].imshow(l2_map, cmap="hot", vmin=0, vmax=l2_map.max())
    axes[2].set_title(f"L2 Error Map (mean={mean_l2:.4f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    # Per-channel bar chart
    ch_names = ["R", "G", "B"]
    axes[3].bar(ch_names, l2_per_channel, color=["red", "green", "blue"], edgecolor="black")
    axes[3].set_title("Per-Channel L2 Error")
    axes[3].set_ylabel("L2 Error")
    fig.suptitle(f"L2 Spatial Error Analysis — Layer {layer_idx}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, f"l2_error_layer_{layer_idx:03d}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)

    return {
        "layer_idx": layer_idx,
        "mean_l2": float(mean_l2),
        "max_l2": float(max_l2),
        "center_mean_l2": float(center_mean),
        "border_mean_l2": float(border_mean),
        "l2_r": float(l2_per_channel[0]),
        "l2_g": float(l2_per_channel[1]),
        "l2_b": float(l2_per_channel[2]),
    }


# ============================================================
# 2. FFT FREQUENCY ANALYSIS
# ============================================================

def analyze_fft(recon, gt, layer_idx, out_dir):
    """
    2D FFT analysis of reconstruction vs ground truth.
    Measures which spatial frequencies survive at each layer depth.
    """
    with torch.no_grad():
        # Convert to grayscale (mean across channels)
        recon_gray = recon[0].mean(dim=0).cpu().numpy()  # (H, W)
        gt_gray = gt[0].mean(dim=0).cpu().numpy()

        # 2D FFT (shifted so DC is center)
        fft_gt = np.fft.fftshift(np.fft.fft2(gt_gray))
        fft_recon = np.fft.fftshift(np.fft.fft2(recon_gray))

        mag_gt = np.abs(fft_gt)
        mag_recon = np.abs(fft_recon)

        # Power spectrum (magnitude squared)
        power_gt = mag_gt ** 2
        power_recon = mag_recon ** 2

        # Ratio: how much of each frequency is preserved
        ratio = power_recon / (power_gt + 1e-8)

        # Frequency bands (radial bins from center)
        h, w = gt_gray.shape
        cy, cx = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        r = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        # Define frequency bands
        max_r = min(cy, cx)
        n_bands = 10
        band_edges = np.linspace(0, max_r, n_bands + 1)
        band_powers_gt = []
        band_powers_recon = []
        band_ratios = []
        for i in range(n_bands):
            mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
            if mask.sum() > 0:
                pg = power_gt[mask].mean()
                pr = power_recon[mask].mean()
                band_powers_gt.append(pg)
                band_powers_recon.append(pr)
                band_ratios.append(pr / (pg + 1e-8))
            else:
                band_powers_gt.append(0)
                band_powers_recon.append(0)
                band_ratios.append(0)

        # Overall spectral correlation
        spec_corr = np.corrcoef(mag_gt.ravel(), mag_recon.ravel())[0, 1]

        # High-frequency survival: ratio at highest band
        hf_ratio = band_ratios[-1] if band_ratios else 0
        lf_ratio = band_ratios[0] if band_ratios else 0

    # Save FFT visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Row 1: spatial + FFT magnitude
    axes[0, 0].imshow(gt_gray, cmap="gray")
    axes[0, 0].set_title("Ground Truth (grayscale)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.log1p(mag_gt), cmap="hot")
    axes[0, 1].set_title("|FFT(GT)| (log)")
    axes[0, 2].imshow(np.log1p(mag_recon), cmap="hot")
    axes[0, 2].set_title(f"|FFT(Recon)| (log) — Layer {layer_idx}")

    # Row 2: power spectra + band ratios
    axes[1, 0].imshow(np.log1p(power_gt), cmap="inferno")
    axes[1, 0].set_title("Power Spectrum GT (log)")
    axes[1, 1].imshow(np.log1p(power_recon), cmap="inferno")
    axes[1, 1].set_title("Power Spectrum Recon (log)")
    # Band ratio plot
    band_labels = [f"Band {i}" for i in range(n_bands)]
    axes[1, 2].bar(band_labels, band_ratios, color=plt.cm.coolwarm(np.linspace(0, 1, n_bands)),
                   edgecolor="black", linewidth=0.3)
    axes[1, 2].set_title("Frequency Band Survival Ratio")
    axes[1, 2].set_ylabel("Power Recon / Power GT")
    axes[1, 2].axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    axes[1, 2].set_xticklabels(band_labels, fontsize=7, rotation=45, ha="right")

    fig.suptitle(f"FFT Frequency Analysis — Layer {layer_idx}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, f"fft_layer_{layer_idx:03d}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)

    return {
        "layer_idx": layer_idx,
        "spec_correlation": float(spec_corr),
        "lf_survival": float(lf_ratio),
        "hf_survival": float(hf_ratio),
        "band_ratios": [float(r) for r in band_ratios],
        "band_powers_gt": [float(p) for p in band_powers_gt],
        "band_powers_recon": [float(p) for p in band_powers_recon],
    }


# ============================================================
# 3. GRAPH LAPLACIAN
# ============================================================

def analyze_graph_laplacian(captured, model, img_tensor, out_dir):
    """
    Two graph Laplacian analyses:

    A) Layer-layer graph: adjacency = cosine similarity between flattened
       feature maps of consecutive/all pairs of layers. Shows information
       flow community structure.

    B) Channel-channel graph per key layer: adjacency = cosine similarity
       between channels within a layer. Shows feature community structure.
    """

    # --- A) Layer-layer graph ---
    layer_indices = sorted(captured.keys())
    n_layers = len(layer_indices)

    # Compute pairwise cosine similarity between layer outputs
    # Use per-channel mean as descriptor (fixed-size per layer after padding)
    # Each layer: (C,) vector of channel-wise mean activations
    raw_vectors = []
    for idx in layer_indices:
        feat = captured[idx]["output"][0]  # (C, H, W)
        ch_mean = feat.mean(dim=(1, 2)).cpu().numpy()  # (C,)
        raw_vectors.append(ch_mean)
    max_c = max(v.shape[0] for v in raw_vectors)
    # Pad each vector to max_c with zeros
    feat_vectors = np.zeros((n_layers, max_c), dtype=np.float32)
    for i, v in enumerate(raw_vectors):
        feat_vectors[i, :len(v)] = v

    # Cosine similarity adjacency
    norms = np.linalg.norm(feat_vectors, axis=1, keepdims=True)
    feat_norm = feat_vectors / (norms + 1e-8)
    A_layers = feat_norm @ feat_norm.T  # (n_layers, n_layers)

    # Threshold: keep only strong connections (high threshold for sparse graph)
    threshold = 0.8
    A_thresh = np.where(A_layers > threshold, A_layers, 0)
    np.fill_diagonal(A_thresh, 0)

    # Degree matrix and Laplacian
    deg = A_thresh.sum(axis=1)
    D_mat = np.diag(deg)
    L_layers = D_mat - A_thresh

    # Eigendecomposition
    from numpy.linalg import eigvalsh, eigh
    eigvals_layers, eigvecs_layers = eigh(L_layers)

    # Fiedler vector (2nd smallest eigenvalue)
    fiedler_layers = eigvecs_layers[:, 1] if n_layers > 1 else np.zeros(n_layers)

    # --- B) Channel-channel graph for selected layers ---
    channel_laplacians = {}
    for idx in RECON_LAYERS:
        if idx not in captured:
            continue
        feat = captured[idx]["output"][0]  # (C, H, W)
        C, H, W = feat.shape
        # Subsample spatial locations for efficiency
        n_samples = min(H * W, 400)
        flat = feat.reshape(C, -1)[:, :n_samples].cpu().numpy()  # (C, n_samples)
        # Channel cosine similarity
        ch_norms = np.linalg.norm(flat, axis=1, keepdims=True)
        ch_norm = flat / (ch_norms + 1e-8)
        A_ch = ch_norm @ ch_norm.T  # (C, C)
        A_ch_thresh = np.where(A_ch > 0.8, A_ch, 0)
        np.fill_diagonal(A_ch_thresh, 0)
        ch_deg = A_ch_thresh.sum(axis=1)
        ch_D = np.diag(ch_deg)
        L_ch = ch_D - A_ch_thresh
        ch_eigvals = eigvalsh(L_ch)
        channel_laplacians[idx] = {
            "n_channels": C,
            "n_edges": int((A_ch_thresh > 0).sum() // 2),
            "n_isolated": int((ch_deg == 0).sum()),
            "smallest_eigvals": ch_eigvals[:10].tolist(),
            "fiedler_value": float(ch_eigvals[1]) if len(ch_eigvals) > 1 else 0,
            "spectral_gap": float(ch_eigvals[1] - ch_eigvals[0]) if len(ch_eigvals) > 1 else 0,
        }

    # --- Visualize ---
    # --- Visualize (4 separate large figures for readability) ---

    # Fig 1: Layer-layer adjacency heatmap (large, readable)
    fig1, ax = plt.subplots(1, 1, figsize=(24, 20))
    im = ax.imshow(A_layers, cmap="hot", aspect="auto")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{layer_indices[i]}" for i in range(n_layers)], fontsize=6, rotation=90)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{layer_indices[i]}" for i in range(n_layers)], fontsize=6)
    ax.set_title("Layer-Layer Cosine Similarity (Adjacency Matrix)", fontsize=16)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Cosine similarity")
    # Mark the analyzed layers with red border ticks
    for analyzed in RECON_LAYERS:
        if analyzed in layer_indices:
            pos = layer_indices.index(analyzed)
            ax.axvline(x=pos, color="cyan", linewidth=0.5, alpha=0.5)
            ax.axhline(y=pos, color="cyan", linewidth=0.5, alpha=0.5)
    fig1.tight_layout()
    p1 = os.path.join(out_dir, "graph_laplacian_adjacency.png")
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Saved: {p1}")

    # Fig 2: Layer-layer Laplacian eigenvalues + Fiedler vector (side by side, large)
    fig2, axes2 = plt.subplots(1, 2, figsize=(24, 10))

    # Eigenvalues
    ax = axes2[0]
    n_show = min(30, len(eigvals_layers))
    colors_eig = plt.cm.viridis(np.linspace(0, 1, n_show))
    ax.bar(range(n_show), eigvals_layers[:n_show], color=colors_eig,
           edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Eigenvalue index", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title("Layer Graph Laplacian Eigenvalues", fontsize=14)
    # Annotate the Fiedler (2nd smallest)
    if n_show >= 2:
        ax.annotate(f"Fiedler\n{eigvals_layers[1]:.2e}",
                    xy=(1, eigvals_layers[1]), xytext=(5, max(eigvals_layers[:n_show]) * 0.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    fontsize=10, color="red", fontweight="bold")

    # Fiedler vector — only show non-zero entries for readability
    ax = axes2[1]
    # Find non-zero entries (connected layers only)
    nonzero_mask = np.abs(fiedler_layers) > 1e-10
    nonzero_idx = np.where(nonzero_mask)[0]
    nonzero_vals = fiedler_layers[nonzero_mask]
    nonzero_labels = [f"L{layer_indices[i]}" for i in nonzero_idx]
    colors_fiedler = plt.cm.coolwarm(np.linspace(0, 1, len(nonzero_idx)))
    bars = ax.bar(range(len(nonzero_idx)), nonzero_vals, color=colors_fiedler,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(nonzero_idx)))
    ax.set_xticklabels(nonzero_labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Fiedler vector value", fontsize=12)
    ax.set_title("Fiedler Vector — Connected Layers Only (main information flow split)", fontsize=14)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    # Annotate values on bars
    for bar, val, idx in zip(bars, nonzero_vals, nonzero_idx):
        y_pos = val + 0.02 if val >= 0 else val - 0.04
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:.3f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=8, fontweight="bold")
    # Color the background by community
    pos_layers = [i for i, v in enumerate(nonzero_vals) if v > 0]
    neg_layers = [i for i, v in enumerate(nonzero_vals) if v < 0]
    if pos_layers:
        ax.axvspan(min(pos_layers) - 0.5, max(pos_layers) + 0.5, alpha=0.1, color="blue", label="Community A")
    if neg_layers:
        ax.axvspan(min(neg_layers) - 0.5, max(neg_layers) + 0.5, alpha=0.1, color="red", label="Community B")
    ax.legend(fontsize=10)

    fig2.suptitle("Layer Graph Laplacian — Eigenvalues & Fiedler Vector", fontsize=16)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(out_dir, "graph_laplacian_eigenvalues.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {p2}")

    # Fig 3: Channel Fiedler values per layer (large, readable)
    fig3, ax = plt.subplots(1, 1, figsize=(18, 8))
    viz_idx = [idx for idx in RECON_LAYERS if idx in channel_laplacians]
    fiedler_vals = [channel_laplacians[idx]["fiedler_value"] for idx in viz_idx]
    spectral_gaps = [channel_laplacians[idx]["spectral_gap"] for idx in viz_idx]
    n_edges_ch = [channel_laplacians[idx]["n_edges"] for idx in viz_idx]
    n_iso_ch = [channel_laplacians[idx]["n_isolated"] for idx in viz_idx]
    x = range(len(viz_idx))
    width = 0.35
    ax.bar([i - width/2 for i in x], fiedler_vals, width, label="Fiedler value",
           color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], spectral_gaps, width, label="Spectral gap",
           color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    labels = [f"L{idx}\n{ADV_ANNOTATIONS.get(idx, '')[:20]}" for idx in viz_idx]
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Channel Graph: Fiedler Value & Spectral Gap per Layer", fontsize=14)
    ax.legend(fontsize=11)
    # Annotate edge/isolated counts
    for i, idx in enumerate(viz_idx):
        ax.text(i, max(max(fiedler_vals), max(spectral_gaps)) * 1.1,
                f"{n_edges_ch[i]} edges\n{n_iso_ch[i]} iso",
                ha="center", fontsize=7, color="gray")
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "graph_laplacian_channels.png")
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"Saved: {p3}")

    # Combined overview (smaller, for quick reference)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    ax = axes[0, 0]
    im = ax.imshow(A_layers, cmap="hot", aspect="auto")
    ax.set_title("Layer Adjacency (full)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 1]
    ax.bar(range(min(20, len(eigvals_layers))), eigvals_layers[:20],
           color=plt.cm.viridis(np.linspace(0, 1, min(20, len(eigvals_layers)))),
           edgecolor="black", linewidth=0.3)
    ax.set_title("Laplacian Eigenvalues", fontsize=12)
    ax = axes[1, 0]
    ax.bar(range(len(nonzero_idx)), nonzero_vals, color=colors_fiedler,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(nonzero_idx)))
    ax.set_xticklabels(nonzero_labels, fontsize=8, rotation=45, ha="right")
    ax.set_title("Fiedler Vector (connected layers)", fontsize=12)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax = axes[1, 1]
    ax.bar([i - width/2 for i in x], fiedler_vals, width, label="Fiedler",
           color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], spectral_gaps, width, label="Gap",
           color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{idx}" for idx in viz_idx], fontsize=8, rotation=45, ha="right")
    ax.set_title("Channel Fiedler per Layer", fontsize=12)
    ax.legend(fontsize=10)
    fig.suptitle("Graph Laplacian Overview", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "graph_laplacian.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    return {
        "layer_graph": {
            "n_layers": n_layers,
            "n_edges": int((A_thresh > 0).sum() // 2),
            "n_isolated": int((deg == 0).sum()),
            "smallest_eigvals": eigvals_layers[:10].tolist(),
            "fiedler_value": float(eigvals_layers[1]) if n_layers > 1 else 0,
            "fiedler_vector": fiedler_layers.tolist(),
            "layer_indices": layer_indices,
        },
        "channel_graphs": channel_laplacians,
    }


# ============================================================
# 4. KFAC (Kronecker-Factored Approximate Curvature)
# ============================================================

def compute_kfac(model, captured, img_tensor, layer_idx, device, n_samples=256):
    """
    Compute KFAC approximation of the Fisher Information Matrix for a conv layer.

    Fisher ≈ A ⊗ G where:
      A = E[xx^T]  (input covariance)
        x = input activation at a single spatial location, shape (C_in * kh * kw,)
        A has shape (C_in * kh * kw, C_in * kh * kw)
      G = E[gg^T]  (gradient covariance)
        g = gradient of loss w.r.t. output at a single spatial location, shape (C_out,)
        G has shape (C_out, C_out)

    For class-0: loss = -person_score (we want the curvature of the person detection)

    Returns dict with eigenvalues of A and G, trace, condition number, etc.
    """
    info = captured[layer_idx]
    conv = info["conv"]
    bn = info["bn"]
    act = info["act"]
    c_in = info["c_in"]
    c_out = info["c_out"]
    kh, kw = info["kernel"]

    input_feat = info["input"]  # (1, C_in, H, W)
    output_feat = info["output"]  # (1, C_out, H_out, W_out)

    # --- Compute A: input covariance ---
    # Extract conv input patches: for each output spatial location,
    # the input patch is (C_in, kh, kw) -> flatten to (C_in * kh * kw,)
    # Use unfold to extract all patches
    with torch.no_grad():
        # Unfold input: (1, C_in * kh * kw, n_locations)
        patches = F.unfold(
            input_feat,
            kernel_size=(kh, kw),
            padding=conv.padding,
            stride=conv.stride
        )  # (1, C_in * kh * kw, n_locs)
        patches = patches.squeeze(0).T  # (n_locs, C_in * kh * kw)

        # Subsample locations
        n_locs = patches.shape[0]
        if n_locs > n_samples:
            idx = torch.randperm(n_locs, device=device)[:n_samples]
            patches = patches[idx]

        # A = (1/N) * X^T @ X  — (C_in*kh*kw, C_in*kh*kw)
        A = (patches.T @ patches) / patches.shape[0]

    # --- Compute G: gradient covariance ---
    # Need gradients of the class-0 person score w.r.t. the conv output
    # at each spatial location

    # Find peak person detection
    with torch.no_grad():
        output = model(img_tensor)
    person_scores = output[0, :, 5 + CLASS_0] * output[0, :, 4]
    peak_idx = person_scores.argmax().item()

    # Re-run forward with gradient tracking
    model.zero_grad()
    img_grad = img_tensor.clone().detach().requires_grad_(True)

    # Hook to capture output gradients at this layer
    output_grad = [None]

    def fwd_hook(module, inp, out):
        # Store the output for gradient computation
        output_grad[0] = out

    def bwd_hook(module, grad_input, grad_output):
        # grad_output[0]: (1, C_out, H_out, W_out) — gradient w.r.t. output
        output_grad[0] = grad_output[0].detach().clone()

    conv_mod = model.module_list[layer_idx][0]
    f_hook = conv_mod.register_forward_hook(fwd_hook)
    g_hook = conv_mod.register_full_backward_hook(bwd_hook)

    # Forward
    output = model(img_grad)
    score = output[0, peak_idx, 5 + CLASS_0] * output[0, peak_idx, 4]
    # Use loss = -log(score) so gradients are meaningful (raw score ~0.996 has tiny gradient)
    loss = -torch.log(score + 1e-8)

    # Backward
    loss.backward()

    f_hook.remove()
    g_hook.remove()

    if output_grad[0] is None:
        return None

    with torch.no_grad():
        grad_out = output_grad[0]  # (1, C_out, H_out, W_out)
        # Flatten spatial: (C_out, n_locs) -> transpose -> (n_locs, C_out)
        g_flat = grad_out[0].reshape(c_out, -1).T  # (n_locs, C_out)

        n_g_locs = g_flat.shape[0]
        if n_g_locs > n_samples:
            idx = torch.randperm(n_g_locs, device=device)[:n_samples]
            g_flat = g_flat[idx]

        # G = (1/N) * G^T @ G  — (C_out, C_out)
        G = (g_flat.T @ g_flat) / g_flat.shape[0]

    # --- Compute metrics ---
    with torch.no_grad():
        # Eigenvalues of A and G
        eig_A = torch.linalg.eigvalsh(A)
        eig_G = torch.linalg.eigvalsh(G)

        # Filter out near-zero eigenvalues
        eig_A_pos = eig_A[eig_A > 1e-10]
        eig_G_pos = eig_G[eig_G > 1e-10]

        trace_A = eig_A.sum().item()
        trace_G = eig_G.sum().item()
        trace_fisher = trace_A * trace_G  # trace(A ⊗ G) = trace(A) * trace(G)

        max_eig_A = eig_A[-1].item() if len(eig_A) > 0 else 0
        max_eig_G = eig_G[-1].item() if len(eig_G) > 0 else 0
        min_eig_A = eig_A_pos[0].item() if len(eig_A_pos) > 0 else 0
        min_eig_G = eig_G_pos[0].item() if len(eig_G_pos) > 0 else 0

        cond_A = max_eig_A / min_eig_A if min_eig_A > 0 else float('inf')
        cond_G = max_eig_G / min_eig_G if min_eig_G > 0 else float('inf')

        # Effective rank (participation ratio): (sum(eig))^2 / sum(eig^2)
        er_A = (eig_A.sum() ** 2 / (eig_A ** 2).sum()).item() if (eig_A ** 2).sum() > 0 else 0
        er_G = (eig_G.sum() ** 2 / (eig_G ** 2).sum()).item() if (eig_G ** 2).sum() > 0 else 0

        # Max eigenvalue of A ⊗ G ≈ max_eig_A * max_eig_G (for Kronecker product)
        max_fisher_eig = max_eig_A * max_eig_G

    # Save eigenvalue distributions for visualization
    np.save(os.path.join(OUTPUT_DIR, "kfac_eigenvalues", f"eig_A_layer_{layer_idx:03d}.npy"),
            eig_A.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "kfac_eigenvalues", f"eig_G_layer_{layer_idx:03d}.npy"),
            eig_G.cpu().numpy())

    return {
        "layer_idx": layer_idx,
        "c_in": c_in,
        "c_out": c_out,
        "kernel": f"{kh}x{kw}",
        "stride": info["stride"],
        "trace_A": trace_A,
        "trace_G": trace_G,
        "trace_fisher": trace_fisher,
        "max_eig_A": max_eig_A,
        "max_eig_G": max_eig_G,
        "max_fisher_eig": max_fisher_eig,
        "cond_A": cond_A,
        "cond_G": cond_G,
        "eff_rank_A": er_A,
        "eff_rank_G": er_G,
        "n_samples_A": patches.shape[0],
        "n_samples_G": g_flat.shape[0],
    }


def plot_kfac(kfac_results, out_dir):
    """Plot KFAC metrics across layers — large, annotated, adversarial-patch-focused."""
    n = len(kfac_results)
    names = [f"L{r['layer_idx']}" for r in kfac_results]
    sub_names = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'], '')[:18]}" for r in kfac_results]
    x = range(n)

    # Color: highlight high-curvature layers in red, zero-gradient in gray
    def bar_colors(values, threshold_ratio=0.3):
        max_val = max(values) if max(values) > 0 else 1
        colors = []
        for v in values:
            if v == 0:
                colors.append("#cccccc")
            elif v > max_val * threshold_ratio:
                colors.append("#e74c3c")
            else:
                colors.append("#3498db")
        return colors

    fig, axes = plt.subplots(3, 2, figsize=(22, 24))

    # Trace of Fisher (total curvature) — KEY METRIC for patch attacks
    ax = axes[0, 0]
    vals = [r["trace_fisher"] for r in kfac_results]
    ax.bar(x, vals, color=bar_colors(vals), edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("trace(A \u2297 G)", fontsize=12)
    ax.set_title("KFAC Total Curvature (Fisher Trace)\nRED = high curvature (patch leverage) | GRAY = zero gradient", fontsize=12)
    ax.set_yscale("log")
    max_i = vals.index(max(vals))
    ax.annotate(f"MAX: {vals[max_i]:.2e}\n{ADV_ANNOTATIONS.get(kfac_results[max_i]['layer_idx'], '')}",
                xy=(max_i, vals[max_i]), xytext=(max_i + 1, vals[max_i] * 3),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")
    for i, v in enumerate(vals):
        if v == 0:
            ax.text(i, 1e-12, "NO GRAD\n(patch irrelevant)", ha="center", fontsize=7, color="gray")

    # Max Fisher eigenvalue — sharpest direction
    ax = axes[0, 1]
    vals = [r["max_fisher_eig"] for r in kfac_results]
    ax.bar(x, vals, color=bar_colors(vals), edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("max(eig(A \u2297 G))", fontsize=12)
    ax.set_title("KFAC Max Curvature Direction\nsharpest attack axis per layer", fontsize=12)
    ax.set_yscale("log")
    max_i = vals.index(max(vals))
    ax.annotate(f"MAX: {vals[max_i]:.2e}\n{ADV_ANNOTATIONS.get(kfac_results[max_i]['layer_idx'], '')}",
                xy=(max_i, vals[max_i]), xytext=(max_i + 1, vals[max_i] * 3),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")

    # Condition numbers — ill-conditioned = exploitable
    ax = axes[1, 0]
    width = 0.35
    cond_a_vals = [min(r["cond_A"], 1e15) for r in kfac_results]
    cond_g_vals = [min(r["cond_G"], 1e15) if r["cond_G"] != float('inf') else 1e15 for r in kfac_results]
    ax.bar([i - width/2 for i in x], cond_a_vals, width,
           label="cond(A) \u2014 input space", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], cond_g_vals, width,
           label="cond(G) \u2014 gradient space", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Condition number (capped at 1e15)", fontsize=12)
    ax.set_title("KFAC Condition Numbers\nHIGHER = more exploitable input subspace", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Effective ranks — diversity of attack directions
    ax = axes[1, 1]
    ax.bar([i - width/2 for i in x], [r["eff_rank_A"] for r in kfac_results], width,
           label="eff_rank(A) \u2014 input dirs", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["eff_rank_G"] for r in kfac_results], width,
           label="eff_rank(G) \u2014 gradient dirs", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Effective rank (participation ratio)", fontsize=12)
    ax.set_title("KFAC Effective Ranks\nMORE dirs = harder to attack | FEWER = easier", fontsize=12)
    ax.legend(fontsize=11)
    er_g_vals = [r["eff_rank_G"] for r in kfac_results]
    min_i = er_g_vals.index(min(er_g_vals))
    ax.annotate(f"MIN: {er_g_vals[min_i]:.1f}\n(easiest to attack)",
                xy=(min_i + width/2, er_g_vals[min_i]),
                xytext=(min_i + 2, max(er_g_vals) * 0.7),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")

    # Trace A and Trace G separately
    ax = axes[2, 0]
    ax.bar([i - width/2 for i in x], [r["trace_A"] for r in kfac_results], width,
           label="trace(A) \u2014 input energy", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["trace_G"] for r in kfac_results], width,
           label="trace(G) \u2014 gradient energy", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Trace", fontsize=12)
    ax.set_title("KFAC Factor Traces (input vs gradient energy)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Max eigenvalues A and G
    ax = axes[2, 1]
    ax.bar([i - width/2 for i in x], [r["max_eig_A"] for r in kfac_results], width,
           label="max(eig(A))", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["max_eig_G"] for r in kfac_results], width,
           label="max(eig(G))", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Max eigenvalue", fontsize=12)
    ax.set_title("KFAC Factor Max Eigenvalues", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    fig.suptitle("KFAC: Kronecker-Factored Fisher Information per Layer (class-0 person)\n"
                 "RED = high curvature (patch leverage) | GRAY = zero gradient (patch irrelevant)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, "kfac_analysis.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


# ============================================================
# UTILITIES
# ============================================================

def tensor_to_image(t):
    """Convert (1, 3, H, W) tensor to (H, W, 3) numpy in [0, 1]."""
    return t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)


def save_combined_csv(l2_results, fft_results, kfac_results, out_path):
    """Save all metrics to a single CSV."""
    rows = []
    for l2, fft in zip(l2_results, fft_results):
        row = {
            "layer_idx": l2["layer_idx"],
            "mean_l2": l2["mean_l2"],
            "max_l2": l2["max_l2"],
            "center_l2": l2["center_mean_l2"],
            "border_l2": l2["border_mean_l2"],
            "l2_r": l2["l2_r"], "l2_g": l2["l2_g"], "l2_b": l2["l2_b"],
            "fft_spec_corr": fft["spec_correlation"],
            "fft_lf_survival": fft["lf_survival"],
            "fft_hf_survival": fft["hf_survival"],
        }
        # Add KFAC if available
        kfac = next((k for k in kfac_results if k["layer_idx"] == l2["layer_idx"]), None)
        if kfac:
            row.update({
                "kfac_trace_fisher": kfac["trace_fisher"],
                "kfac_max_eig": kfac["max_fisher_eig"],
                "kfac_cond_A": kfac["cond_A"],
                "kfac_cond_G": kfac["cond_G"],
                "kfac_eff_rank_A": kfac["eff_rank_A"],
                "kfac_eff_rank_G": kfac["eff_rank_G"],
            })
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Combined L2 / FFT / Graph Laplacian / KFAC Analysis for YOLOv3")
    print("=" * 60)

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    print("\nLoading image...")
    img_tensor, img_np = load_image(IMAGE_PATH, IMG_SIZE)
    print(f"Image: {IMAGE_PATH}")
    print(f"Input: {img_tensor.shape}")

    # Forward pass capturing all conv layer features
    print(f"\n{'=' * 60}")
    print("Forward pass — capturing all conv layer features")
    print("=" * 60)
    captured = forward_capture_all(model, img_tensor)
    print(f"Captured {len(captured)} conv layers")

    # --- 1. L2 + 2. FFT (require SIPIT reconstruction) ---
    print(f"\n{'=' * 60}")
    print(f"Phase 1-2: L2 Error + FFT Frequency Analysis")
    print(f"SIPIT reconstruction: {N_ITER} iters, lr={LR}")
    print("=" * 60)

    l2_results = []
    fft_results = []

    for layer_idx in RECON_LAYERS:
        if layer_idx not in captured:
            continue

        target = captured[layer_idx]["output"].detach()
        print(f"\n--- Layer {layer_idx} (target: {target.shape}) ---")

        # SIPIT reconstruction
        recon, loss = reconstruct_input(
            model, target, layer_idx, IMG_SIZE, DEVICE,
            n_iter=N_ITER, lr=LR, tv_weight=TV_WEIGHT
        )
        print(f"  Reconstruction loss: {loss:.6f}")

        # L2 analysis
        l2_res = analyze_l2_error(recon, img_tensor, layer_idx,
                                   os.path.join(OUTPUT_DIR, "l2_error_maps"))
        l2_results.append(l2_res)
        print(f"  L2: mean={l2_res['mean_l2']:.4f}  center={l2_res['center_mean_l2']:.4f}  "
              f"border={l2_res['border_mean_l2']:.4f}")

        # FFT analysis
        fft_res = analyze_fft(recon, img_tensor, layer_idx,
                               os.path.join(OUTPUT_DIR, "fft_spectra"))
        fft_results.append(fft_res)
        print(f"  FFT: spec_corr={fft_res['spec_correlation']:.4f}  "
              f"LF_survival={fft_res['lf_survival']:.4f}  "
              f"HF_survival={fft_res['hf_survival']:.4f}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 3. Graph Laplacian ---
    print(f"\n{'=' * 60}")
    print("Phase 3: Graph Laplacian Analysis")
    print("=" * 60)
    laplacian_results = analyze_graph_laplacian(captured, model, img_tensor, OUTPUT_DIR)

    print(f"\nLayer graph: {laplacian_results['layer_graph']['n_layers']} layers, "
          f"{laplacian_results['layer_graph']['n_edges']} edges, "
          f"{laplacian_results['layer_graph']['n_isolated']} isolated")
    print(f"  Fiedler value: {laplacian_results['layer_graph']['fiedler_value']:.6f}")
    print(f"  Smallest eigenvalues: {[f'{v:.4f}' for v in laplacian_results['layer_graph']['smallest_eigvals'][:5]]}")

    print(f"\nChannel graphs:")
    for idx, cg in laplacian_results['channel_graphs'].items():
        print(f"  Layer {idx:3d}: {cg['n_channels']} ch, {cg['n_edges']} edges, "
              f"{cg['n_isolated']} isolated, Fiedler={cg['fiedler_value']:.6f}, "
              f"gap={cg['spectral_gap']:.6f}")

    # --- 4. KFAC ---
    print(f"\n{'=' * 60}")
    print(f"Phase 4: KFAC (Kronecker-Factored Approximate Curvature)")
    print(f"Samples per layer: {KFAC_SAMPLES}")
    print("=" * 60)

    kfac_results = []
    for layer_idx in RECON_LAYERS:
        if layer_idx not in captured:
            continue
        print(f"\n  Layer {layer_idx}: in={captured[layer_idx]['c_in']} "
              f"out={captured[layer_idx]['c_out']} "
              f"k={captured[layer_idx]['kernel']}")

        kfac = compute_kfac(model, captured, img_tensor, layer_idx, DEVICE,
                            n_samples=KFAC_SAMPLES)
        if kfac is not None:
            kfac_results.append(kfac)
            print(f"    trace_fisher={kfac['trace_fisher']:.6e}  "
                  f"max_eig={kfac['max_fisher_eig']:.6e}  "
                  f"cond_A={kfac['cond_A']:.2f}  cond_G={kfac['cond_G']:.2f}  "
                  f"eff_rank_A={kfac['eff_rank_A']:.1f}  eff_rank_G={kfac['eff_rank_G']:.1f}")
        else:
            print(f"    SKIP: no gradient captured")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print("\n--- L2 Error: where is information lost? ---")
    print(f"{'Layer':>6} {'Mean L2':>10} {'Center':>10} {'Border':>10} {'Ratio C/B':>10}")
    for r in l2_results:
        ratio = r["center_mean_l2"] / (r["border_mean_l2"] + 1e-8)
        print(f"  {r['layer_idx']:>4} {r['mean_l2']:>10.4f} {r['center_mean_l2']:>10.4f} "
              f"{r['border_mean_l2']:>10.4f} {ratio:>10.4f}")

    print("\n--- FFT: frequency survival ---")
    print(f"{'Layer':>6} {'Spec Corr':>10} {'LF Surv':>10} {'HF Surv':>10} {'HF/LF':>10}")
    for r in fft_results:
        hf_lf = r["hf_survival"] / (r["lf_survival"] + 1e-8)
        print(f"  {r['layer_idx']:>4} {r['spec_correlation']:>10.4f} {r['lf_survival']:>10.4f} "
              f"{r['hf_survival']:>10.4f} {hf_lf:>10.4f}")

    print("\n--- KFAC: real curvature of class-0 loss landscape ---")
    print(f"{'Layer':>6} {'trace(F)':>14} {'max_eig':>14} {'cond_A':>10} {'cond_G':>10} {'er_A':>8} {'er_G':>8}")
    for r in kfac_results:
        print(f"  {r['layer_idx']:>4} {r['trace_fisher']:>14.6e} {r['max_fisher_eig']:>14.6e} "
              f"{r['cond_A']:>10.2f} {r['cond_G']:>10.2f} {r['eff_rank_A']:>8.1f} {r['eff_rank_G']:>8.1f}")

    print("\n--- Top 5 layers by KFAC trace (highest curvature) ---")
    for r in sorted(kfac_results, key=lambda x: x["trace_fisher"], reverse=True)[:5]:
        print(f"  Layer {r['layer_idx']:3d}  trace_F={r['trace_fisher']:.6e}  "
              f"max_eig={r['max_fisher_eig']:.6e}  "
              f"in={r['c_in']}  out={r['c_out']}  k={r['kernel']}")

    print("\n--- Top 5 layers by KFAC max eigenvalue (sharpest curvature direction) ---")
    for r in sorted(kfac_results, key=lambda x: x["max_fisher_eig"], reverse=True)[:5]:
        print(f"  Layer {r['layer_idx']:3d}  max_eig={r['max_fisher_eig']:.6e}  "
              f"trace_F={r['trace_fisher']:.6e}  "
              f"in={r['c_in']}  out={r['c_out']}  k={r['kernel']}")

    # --- Outputs ---
    print(f"\n{'=' * 60}")
    print("Generating outputs...")
    print("=" * 60)

    save_combined_csv(l2_results, fft_results, kfac_results,
                      os.path.join(OUTPUT_DIR, "combined_analysis.csv"))

    # Save Laplacian results as JSON
    with open(os.path.join(OUTPUT_DIR, "graph_laplacian.json"), "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(laplacian_results, f, indent=2, default=convert)
    print(f"Saved: graph_laplacian.json")

    # KFAC plot
    if kfac_results:
        plot_kfac(kfac_results, OUTPUT_DIR)

    # L2 summary plot — large, annotated, with center/border ratio
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    names = [f"L{r['layer_idx']}" for r in l2_results]
    sub_names = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'], '')[:18]}" for r in l2_results]
    x = range(len(l2_results))

    # Mean L2 error
    ax = axes[0]
    mean_vals = [r["mean_l2"] for r in l2_results]
    # Color: gradient from green (low error) to red (high error)
    colors_l2 = plt.cm.RdYlGn_r(np.linspace(0, 1, len(l2_results)))
    ax.bar(x, mean_vals, color=colors_l2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean L2 Error", fontsize=12)
    ax.set_title("L2 Reconstruction Error per Layer — LOWER = more info preserved", fontsize=13)
    # Annotate min and max
    min_i = mean_vals.index(min(mean_vals))
    max_i = mean_vals.index(max(mean_vals))
    ax.annotate(f"MIN: {mean_vals[min_i]:.4f}\n{ADV_ANNOTATIONS.get(l2_results[min_i]['layer_idx'], '')}",
                xy=(min_i, mean_vals[min_i]), xytext=(min_i + 1, mean_vals[min_i] * 0.5),
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
                fontsize=9, color="green", fontweight="bold")
    ax.annotate(f"MAX: {mean_vals[max_i]:.4f}\n{ADV_ANNOTATIONS.get(l2_results[max_i]['layer_idx'], '')}",
                xy=(max_i, mean_vals[max_i]), xytext=(max_i + 1, mean_vals[max_i] * 1.1),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")

    # Center vs Border
    ax = axes[1]
    width = 0.35
    ax.bar([i - width/2 for i in x], [r["center_mean_l2"] for r in l2_results], width,
           label="Center (person region)", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in x], [r["border_mean_l2"] for r in l2_results], width,
           label="Border (background)", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("L2 Error", fontsize=12)
    ax.set_title("Center vs Border L2 Error — where is information lost?", fontsize=13)
    ax.legend(fontsize=11)

    # Center/Border ratio — KEY for patch placement
    ax = axes[2]
    ratios = [r["center_mean_l2"] / (r["border_mean_l2"] + 1e-8) for r in l2_results]
    colors_ratio = plt.cm.coolwarm(np.linspace(0, 1, len(ratios)))
    ax.bar(x, ratios, color=colors_ratio, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Center / Border Ratio", fontsize=12)
    ax.set_title("Center/Border L2 Ratio — HIGHER = person region more vulnerable to reconstruction loss\n"
                 "~1.0 = uniform loss (no spatial preference for patch placement)", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Uniform (1.0)")
    ax.legend(fontsize=11)
    # Annotate max and min ratio
    max_r_i = ratios.index(max(ratios))
    min_r_i = ratios.index(min(ratios))
    ax.annotate(f"MAX ratio: {ratios[max_r_i]:.3f}\n(person most vulnerable here)",
                xy=(max_r_i, ratios[max_r_i]), xytext=(max_r_i + 2, ratios[max_r_i] + 0.05),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")
    ax.annotate(f"MIN ratio: {ratios[min_r_i]:.3f}\n(most uniform)",
                xy=(min_r_i, ratios[min_r_i]), xytext=(min_r_i + 2, ratios[min_r_i] - 0.08),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                fontsize=9, color="blue", fontweight="bold")

    fig.suptitle("L2 Spatial Error Analysis — Adversarial Patch Relevance", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTPUT_DIR, "l2_summary.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # FFT summary plot — large, annotated, with HF/LF ratio
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fft_sub_names = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'], '')[:18]}" for r in fft_results]
    fft_x = range(len(fft_results))

    ax = axes[0]
    spec_vals = [r["spec_correlation"] for r in fft_results]
    ax.bar(fft_x, spec_vals,
           color=plt.cm.plasma(np.linspace(0, 1, len(fft_results))), edgecolor="black", linewidth=0.3)
    ax.set_xticks(fft_x); ax.set_xticklabels(fft_sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spectral Correlation", fontsize=12)
    ax.set_title("FFT Spectral Correlation (Recon vs GT) — HIGHER = frequency structure preserved", fontsize=13)
    # Annotate min (biggest frequency shift)
    min_i = spec_vals.index(min(spec_vals))
    ax.annotate(f"MIN: {spec_vals[min_i]:.4f}\n{ADV_ANNOTATIONS.get(fft_results[min_i]['layer_idx'], '')}",
                xy=(min_i, spec_vals[min_i]), xytext=(min_i + 1, spec_vals[min_i] - 0.02),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")

    ax = axes[1]
    width = 0.35
    lf_vals = [r["lf_survival"] for r in fft_results]
    hf_vals = [r["hf_survival"] for r in fft_results]
    ax.bar([i - width/2 for i in fft_x], lf_vals, width,
           label="Low-freq survival", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i + width/2 for i in fft_x], hf_vals, width,
           label="High-freq survival", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(fft_x); ax.set_xticklabels(fft_sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Power Ratio (Recon/GT)", fontsize=12)
    ax.set_title("Low vs High Frequency Survival per Layer", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect preservation (1.0)")
    ax.legend(fontsize=11)

    # HF/LF ratio — KEY for patch frequency design
    ax = axes[2]
    hf_lf_ratios = [h / (l + 1e-8) for h, l in zip(hf_vals, lf_vals)]
    colors_hf = plt.cm.coolwarm(np.linspace(0, 1, len(hf_lf_ratios)))
    ax.bar(fft_x, hf_lf_ratios, color=colors_hf, edgecolor="black", linewidth=0.3)
    ax.set_xticks(fft_x); ax.set_xticklabels(fft_sub_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("HF/LF Ratio", fontsize=12)
    ax.set_title("HF/LF Survival Ratio — HIGHER = more HF noise injected (patch should use LF to survive)\n"
                 "LOW = LF and HF equally preserved (patch frequency less critical)", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Equal (1.0)")
    ax.legend(fontsize=11)
    # Annotate the cliff (biggest jump)
    diffs = [hf_lf_ratios[i+1] - hf_lf_ratios[i] for i in range(len(hf_lf_ratios)-1)]
    cliff_i = diffs.index(max(diffs)) + 1
    ax.annotate(f"FREQ CLIFF\n{hf_lf_ratios[cliff_i-1]:.1f} -> {hf_lf_ratios[cliff_i]:.1f}",
                xy=(cliff_i, hf_lf_ratios[cliff_i]),
                xytext=(cliff_i + 1, hf_lf_ratios[cliff_i] + 5),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=9, color="red", fontweight="bold")

    fig.suptitle("FFT Frequency Analysis — Adversarial Patch Frequency Design", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTPUT_DIR, "fft_summary.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
