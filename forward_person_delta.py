"""
Forward-Direction Person Signal Analysis for YOLOv3.

Takes a paired image set (with human / without human) and measures:
  1. Per-layer, per-channel activation delta (with - without)
  2. Raw 2D FFT on feature maps (not reconstructions)
  3. Which channels are most person-sensitive
  4. Spatial activation maps showing where the person signal concentrates
  5. Gradient saliency from person detection back to input pixels

No SIPIT. No reconstruction. Pure forward analysis.
All CUDA. Outputs to outputs_clothing/forward_analysis/.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# Add PyTorch-YOLOv3 to path
sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
# Bypass imgaug (incompatible with numpy 2.0)
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet

# ----------------------------- CONFIG -----------------------------
CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416

# Layers to analyze — all major conv layers across the backbone + detection heads
ANALYSIS_LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "activation_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fft_feature_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "saliency_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "person_delta_maps"), exist_ok=True)

# YOLOv3 COCO class names (person = class 0)
COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# ----------------------------- IMAGE LOADING -----------------------------

def load_image(img_path, img_size=416):
    """Load and preprocess image to YOLOv3 input tensor (1, 3, H, W) on CUDA."""
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


# ----------------------------- FORWARD CAPTURE -----------------------------

def forward_capture_all(model, img_tensor):
    """Forward pass capturing output tensor for every conv layer.
    Returns dict {layer_idx: (output_tensor, module_def)}."""
    captured = {}
    layer_outputs = []
    x = img_tensor
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
        if mdef["type"] == "convolutional":
            captured[i] = {
                "output": x.detach().clone(),
                "shape": tuple(x.shape),
                "in_channels": mod[0].in_channels,
                "out_channels": mod[0].out_channels,
            }
        layer_outputs.append(x)
    return captured


def forward_full_with_grad(model, img_tensor):
    """Full forward pass with gradient tracking. Returns all layer outputs and yolo outputs.
    img_tensor must have requires_grad=True for saliency."""
    layer_outputs = []
    yolo_outputs = []
    x = img_tensor
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
            yolo_outputs.append(x)
        layer_outputs.append(x)
    return layer_outputs, yolo_outputs


# ============================================================
# 1. ACTIVATION STRENGTH + PERSON DELTA
# ============================================================

def compute_activation_stats(feat):
    """Per-channel activation statistics.
    feat: (1, C, H, W) tensor
    Returns dict with per-channel mean, L2-norm, sparsity, max activation."""
    # feat is (1, C, H, W) — squeeze batch
    f = feat.squeeze(0)  # (C, H, W)
    # Per-channel mean over spatial dims
    ch_mean = f.mean(dim=[1, 2])  # (C,)
    # Per-channel L2 norm
    ch_l2 = f.norm(p=2, dim=[1, 2])  # (C,)
    # Sparsity: fraction of near-zero activations (|act| < 0.01)
    ch_sparsity = (f.abs() < 0.01).float().mean(dim=[1, 2])  # (C,)
    # Per-channel max activation
    ch_max = f.max(dim=1)[0].max(dim=1)[0]  # (C,)
    return {
        "mean": ch_mean.cpu().numpy(),
        "l2": ch_l2.cpu().numpy(),
        "sparsity": ch_sparsity.cpu().numpy(),
        "max": ch_max.cpu().numpy(),
    }


def compute_person_delta(feat_with, feat_without):
    """Per-channel delta between with-human and without-human feature maps.
    Returns per-channel L2 delta, mean delta, and relative delta (delta / baseline)."""
    f_w = feat_with.squeeze(0)    # (C, H, W)
    f_wo = feat_without.squeeze(0)  # (C, H, W)
    # Absolute difference
    diff = f_w - f_wo  # (C, H, W)
    # Per-channel L2 norm of the difference
    ch_delta_l2 = diff.norm(p=2, dim=[1, 2]).cpu().numpy()  # (C,)
    # Per-channel mean of absolute difference
    ch_delta_mean = diff.abs().mean(dim=[1, 2]).cpu().numpy()  # (C,)
    # Relative delta: how much did this channel change relative to its baseline L2
    baseline_l2 = f_wo.norm(p=2, dim=[1, 2]).cpu().numpy()
    ch_delta_rel = np.where(baseline_l2 > 1e-8, ch_delta_l2 / baseline_l2, 0.0)
    # Spatial delta map: where in the feature map did the most change happen
    # Average across channels to get a (H, W) spatial delta
    spatial_delta = diff.abs().mean(dim=0).cpu().numpy()  # (H, W)
    return {
        "delta_l2": ch_delta_l2,
        "delta_mean": ch_delta_mean,
        "delta_rel": ch_delta_rel,
        "spatial_delta": spatial_delta,
    }


# ============================================================
# 2. RAW FFT ON FEATURE MAPS
# ============================================================

def compute_feature_fft(feat, layer_idx, top_k=20):
    """2D FFT on raw feature maps (not reconstructions).
    For each channel, compute the 2D power spectrum, then aggregate.
    feat: (1, C, H, W) tensor
    Returns radial power profile, top-K frequency-responsive channels, and mean spectrum."""
    f = feat.squeeze(0).cpu().numpy()  # (C, H, W)
    C, H, W = f.shape

    # Per-channel 2D FFT — shift zero freq to center
    # Compute power spectrum for each channel
    spectra = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        fft2d = np.fft.fft2(f[c])
        fft2d_shifted = np.fft.fftshift(fft2d)
        spectra[c] = np.abs(fft2d_shifted) ** 2

    # Radial power profile: average power at each radial frequency
    cy, cx = H // 2, W // 2
    y_coords, x_coords = np.indices((H, W))
    radial = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r + 1, dtype=np.float32)
    radial_count = np.zeros(max_r + 1, dtype=np.float32)
    for r in range(max_r + 1):
        mask = radial == r
        if mask.any():
            radial_profile[r] = spectra[:, mask].mean()
            radial_count[r] = mask.sum()

    # Normalize radial profile
    total_power = radial_profile.sum()
    if total_power > 0:
        radial_profile_norm = radial_profile / total_power
    else:
        radial_profile_norm = radial_profile

    # Low/medium/high frequency bands
    # Low: 0-25% of max radius, Medium: 25-50%, High: 50-100%
    r_low = max_r // 4
    r_mid = max_r // 2
    low_power = radial_profile[:r_low].sum()
    mid_power = radial_profile[r_low:r_mid].sum()
    high_power = radial_profile[r_mid:].sum()
    total = low_power + mid_power + high_power
    if total > 0:
        lf_ratio = low_power / total
        mf_ratio = mid_power / total
        hf_ratio = high_power / total
    else:
        lf_ratio = mf_ratio = hf_ratio = 0.0

    # Top-K channels by total spectral power (most frequency-active channels)
    ch_total_power = spectra.sum(axis=(1, 2))
    top_channels = np.argsort(ch_total_power)[::-1][:top_k]

    # Per-channel high/low frequency ratio
    ch_lf = spectra[:, :r_low].sum(axis=(1, 2)) if r_low > 0 else np.zeros(C)
    ch_hf = spectra[:, r_mid:].sum(axis=(1, 2)) if r_mid < max_r else np.zeros(C)
    ch_hl_ratio = np.where(ch_lf > 1e-12, ch_hf / (ch_lf + 1e-12), 0.0)

    return {
        "radial_profile": radial_profile_norm,
        "radial_raw": radial_profile,
        "lf_ratio": lf_ratio,
        "mf_ratio": mf_ratio,
        "hf_ratio": hf_ratio,
        "top_channels": top_channels.tolist(),
        "top_ch_power": ch_total_power[top_channels].tolist(),
        "ch_hl_ratio": ch_hl_ratio,
        "mean_spectrum": spectra.mean(axis=0),  # (H, W) average power spectrum
    }


def compute_fft_delta(feat_with, feat_without, layer_idx):
    """FFT delta: how does the frequency content change when person is present?
    Compares radial power profiles of with vs without."""
    fft_w = compute_feature_fft(feat_with, layer_idx, top_k=10)
    fft_wo = compute_feature_fft(feat_without, layer_idx, top_k=10)

    # Delta in frequency band ratios
    delta_lf = fft_w["lf_ratio"] - fft_wo["lf_ratio"]
    delta_mf = fft_w["mf_ratio"] - fft_wo["mf_ratio"]
    delta_hf = fft_w["hf_ratio"] - fft_wo["hf_ratio"]

    # Delta in radial profile
    radial_delta = fft_w["radial_profile"] - fft_wo["radial_profile"]

    return {
        "delta_lf": delta_lf,
        "delta_mf": delta_mf,
        "delta_hf": delta_hf,
        "radial_delta": radial_delta,
        "with": fft_w,
        "without": fft_wo,
    }


# ============================================================
# 3. GRADIENT SALIENCY
# ============================================================

def compute_saliency(model, img_tensor):
    """Compute gradient saliency: backprop from person detection score to input pixels.
    Shows which input pixels drive the person detection."""
    # Enable gradients for this pass
    model.eval()
    x = img_tensor.clone().detach().requires_grad_(True)

    # Forward pass with gradient tracking
    layer_outputs = []
    yolo_outputs = []
    cur = x
    for i, (mdef, mod) in enumerate(zip(model.module_defs, model.module_list)):
        if mdef["type"] in ["convolutional", "upsample", "maxpool"]:
            cur = mod(cur)
        elif mdef["type"] == "route":
            layers = [int(x_) for x_ in mdef["layers"].split(",")]
            combined = torch.cat([layer_outputs[int(l)] for l in layers], 1)
            group_size = combined.shape[1] // int(mdef.get("groups", 1))
            group_id = int(mdef.get("group_id", 0))
            cur = combined[:, group_size * group_id : group_size * (group_id + 1)]
        elif mdef["type"] == "shortcut":
            layer_i = int(mdef["from"])
            cur = layer_outputs[-1] + layer_outputs[layer_i]
        elif mdef["type"] == "yolo":
            cur = mod[0](cur, x.size(2))
            yolo_outputs.append(cur)
        layer_outputs.append(cur)

    # YOLO outputs are (1, num_anchors, grid, grid, 5+nc) for each scale
    # We want the person class score (class 0)
    # Concatenate all yolo outputs and extract person confidence
    person_scores = []
    for yo in yolo_outputs:
        # yo shape: (1, num_anchors, grid_h, grid_w, 5 + num_classes)
        # obj_conf = yo[..., 4], class_conf = yo[..., 5] for person (class 0)
        # person_score = obj_conf * class_conf
        obj_conf = yo[..., 4]
        class_conf = yo[..., 5]  # class 0 = person
        score = obj_conf * class_conf  # (1, num_anchors, grid_h, grid_w)
        person_scores.append(score.max())

    if not person_scores:
        return None

    # Sum of max person scores across all scales
    person_loss = torch.stack(person_scores).sum()

    # Backprop to input
    model.zero_grad()
    person_loss.backward()

    # Saliency: absolute gradient magnitude per pixel
    saliency = x.grad.abs().squeeze(0).cpu().numpy()  # (3, H, W)
    # Average across channels for visualization
    saliency_gray = saliency.mean(axis=0)  # (H, W)

    return {
        "saliency_rgb": saliency,
        "saliency_gray": saliency_gray,
        "person_score": person_loss.item(),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_activation_comparison(stats_with, stats_without, layer_idx, save_dir):
    """Plot per-channel activation strength: with vs without human."""
    C = len(stats_with["l2"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Layer {layer_idx} — Activation Strength: With vs Without Human", fontsize=14)

    ch_idx = np.arange(C)
    axes[0, 0].bar(ch_idx, stats_with["l2"], alpha=0.6, label="With human", color="red")
    axes[0, 0].bar(ch_idx, stats_without["l2"], alpha=0.6, label="Without human", color="blue")
    axes[0, 0].set_title("Per-channel L2 norm")
    axes[0, 0].set_xlabel("Channel")
    axes[0, 0].set_ylabel("L2 norm")
    axes[0, 0].legend()

    axes[0, 1].bar(ch_idx, stats_with["mean"], alpha=0.6, label="With human", color="red")
    axes[0, 1].bar(ch_idx, stats_without["mean"], alpha=0.6, label="Without human", color="blue")
    axes[0, 1].set_title("Per-channel mean activation")
    axes[0, 1].set_xlabel("Channel")
    axes[0, 1].legend()

    axes[1, 0].bar(ch_idx, stats_with["sparsity"], alpha=0.6, label="With human", color="red")
    axes[1, 0].bar(ch_idx, stats_without["sparsity"], alpha=0.6, label="Without human", color="blue")
    axes[1, 0].set_title("Per-channel sparsity (fraction near-zero)")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].legend()

    axes[1, 1].bar(ch_idx, stats_with["max"], alpha=0.6, label="With human", color="red")
    axes[1, 1].bar(ch_idx, stats_without["max"], alpha=0.6, label="Without human", color="blue")
    axes[1, 1].set_title("Per-channel max activation")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"activation_L{layer_idx:03d}.png"), dpi=150)
    plt.close()


def plot_person_delta(delta, layer_idx, save_dir):
    """Plot per-channel person delta — which channels respond to human presence."""
    C = len(delta["delta_l2"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Layer {layer_idx} — Person Signal Delta (With - Without)", fontsize=14)

    ch_idx = np.arange(C)
    axes[0].bar(ch_idx, delta["delta_l2"], color="darkred")
    axes[0].set_title("Per-channel L2 delta (absolute)")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("|feat_with - feat_without|_2")

    axes[1].bar(ch_idx, delta["delta_rel"], color="darkorange")
    axes[1].set_title("Per-channel relative delta (delta / baseline)")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Relative change")

    # Spatial delta heatmap
    im = axes[2].imshow(delta["spatial_delta"], cmap="hot", aspect="auto")
    axes[2].set_title("Spatial delta map (avg across channels)")
    axes[2].set_xlabel("Feature W")
    axes[2].set_ylabel("Feature H")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"person_delta_L{layer_idx:03d}.png"), dpi=150)
    plt.close()


def plot_feature_fft(fft_result, layer_idx, save_dir, tag=""):
    """Plot raw feature map FFT — radial power profile + spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Layer {layer_idx} — Raw Feature Map FFT {tag}", fontsize=14)

    # Radial power profile
    r = np.arange(len(fft_result["radial_profile"]))
    axes[0].plot(r, fft_result["radial_profile"], linewidth=2, color="blue")
    axes[0].set_title("Radial power profile (normalized)")
    axes[0].set_xlabel("Radial frequency")
    axes[0].set_ylabel("Fraction of total power")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Frequency band ratios as bar
    bands = ["Low", "Medium", "High"]
    ratios = [fft_result["lf_ratio"], fft_result["mf_ratio"], fft_result["hf_ratio"]]
    axes[1].bar(bands, ratios, color=["green", "orange", "red"])
    axes[1].set_title("Frequency band power distribution")
    axes[1].set_ylabel("Fraction of total power")
    for i, v in enumerate(ratios):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)

    # Mean 2D power spectrum (log scale)
    spec = np.log1p(fft_result["mean_spectrum"])
    im = axes[2].imshow(spec, cmap="inferno", aspect="auto")
    axes[2].set_title("Mean 2D power spectrum (log)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fft_feat_L{layer_idx:03d}{tag}.png"), dpi=150)
    plt.close()


def plot_fft_delta(fft_delta, layer_idx, save_dir):
    """Plot FFT delta between with and without human."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Layer {layer_idx} — FFT Delta: With Human vs Without", fontsize=14)

    # Radial profile comparison
    r = np.arange(len(fft_delta["with"]["radial_profile"]))
    axes[0].plot(r, fft_delta["with"]["radial_profile"], linewidth=2, label="With human", color="red")
    axes[0].plot(r, fft_delta["without"]["radial_profile"], linewidth=2, label="Without human", color="blue")
    axes[0].set_title("Radial power profile comparison")
    axes[0].set_xlabel("Radial frequency")
    axes[0].set_ylabel("Fraction of total power")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Band ratio delta
    bands = ["Low", "Medium", "High"]
    deltas = [fft_delta["delta_lf"], fft_delta["delta_mf"], fft_delta["delta_hf"]]
    colors = ["green" if d >= 0 else "darkred" for d in deltas]
    axes[1].bar(bands, deltas, color=colors)
    axes[1].set_title("Frequency band delta (with - without)")
    axes[1].set_ylabel("Change in power fraction")
    for i, v in enumerate(deltas):
        axes[1].text(i, v + 0.001 * (1 if v >= 0 else -1), f"{v:+.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fft_delta_L{layer_idx:03d}.png"), dpi=150)
    plt.close()


def plot_saliency(saliency, img_arr, save_dir, tag=""):
    """Plot gradient saliency overlaid on input image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Gradient Saliency — Person Detection {tag}", fontsize=14)

    axes[0].imshow(img_arr)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # Saliency heatmap
    sal = saliency["saliency_gray"]
    sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    im = axes[1].imshow(sal_norm, cmap="hot", aspect="auto")
    axes[1].set_title("Saliency (gradient magnitude)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(img_arr)
    axes[2].imshow(sal_norm, cmap="hot", alpha=0.5, aspect="auto")
    axes[2].set_title("Saliency overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"saliency{tag}.png"), dpi=150)
    plt.close()


def plot_top_channels_grid(feat_with, feat_without, top_chs, layer_idx, save_dir):
    """Visualize the top person-sensitive channels as spatial activation maps."""
    n = min(len(top_chs), 8)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(f"Layer {layer_idx} — Top {n} Person-Sensitive Channels", fontsize=14)

    f_w = feat_with.squeeze(0).cpu().numpy()
    f_wo = feat_without.squeeze(0).cpu().numpy()

    for i, ch in enumerate(top_chs[:n]):
        # With human
        im_w = axes[0, i].imshow(f_w[ch], cmap="viridis", aspect="auto")
        axes[0, i].set_title(f"Ch {ch} (with)")
        axes[0, i].axis("off")
        plt.colorbar(im_w, ax=axes[0, i], fraction=0.046)

        # Without human
        im_wo = axes[1, i].imshow(f_wo[ch], cmap="viridis", aspect="auto")
        axes[1, i].set_title(f"Ch {ch} (without)")
        axes[1, i].axis("off")
        plt.colorbar(im_wo, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"top_channels_L{layer_idx:03d}.png"), dpi=150)
    plt.close()


def plot_summary(all_results, save_dir):
    """Cross-layer summary plot: person signal strength per layer."""
    layers = sorted(all_results.keys())
    n = len(layers)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Cross-Layer Summary: Person Signal Analysis", fontsize=16)

    # 1. Mean L2 delta per layer
    mean_deltas = [all_results[l]["delta"]["delta_l2"].mean() for l in layers]
    max_deltas = [all_results[l]["delta"]["delta_l2"].max() for l in layers]
    x = range(n)
    axes[0, 0].bar(x, mean_deltas, alpha=0.7, label="Mean channel L2 delta", color="darkred")
    axes[0, 0].bar(x, max_deltas, alpha=0.5, label="Max channel L2 delta", color="orange")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(l) for l in layers], fontsize=8)
    axes[0, 0].set_title("Person signal strength per layer (L2 delta)")
    axes[0, 0].set_xlabel("Layer index")
    axes[0, 0].legend()

    # 2. Mean relative delta per layer
    mean_rel = [all_results[l]["delta"]["delta_rel"].mean() for l in layers]
    max_rel = [all_results[l]["delta"]["delta_rel"].max() for l in layers]
    axes[0, 1].bar(x, mean_rel, alpha=0.7, label="Mean relative delta", color="darkblue")
    axes[0, 1].bar(x, max_rel, alpha=0.5, label="Max relative delta", color="cyan")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(l) for l in layers], fontsize=8)
    axes[0, 1].set_title("Relative person signal per layer (delta / baseline)")
    axes[0, 1].set_xlabel("Layer index")
    axes[0, 1].legend()

    # 3. FFT band ratios: with vs without
    lf_w = [all_results[l]["fft_with"]["lf_ratio"] for l in layers]
    lf_wo = [all_results[l]["fft_without"]["lf_ratio"] for l in layers]
    hf_w = [all_results[l]["fft_with"]["hf_ratio"] for l in layers]
    hf_wo = [all_results[l]["fft_without"]["hf_ratio"] for l in layers]
    axes[1, 0].plot(x, lf_w, "o-", label="LF (with human)", color="green")
    axes[1, 0].plot(x, lf_wo, "s--", label="LF (without human)", color="darkgreen")
    axes[1, 0].plot(x, hf_w, "o-", label="HF (with human)", color="red")
    axes[1, 0].plot(x, hf_wo, "s--", label="HF (without human)", color="darkred")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(l) for l in layers], fontsize=8)
    axes[1, 0].set_title("Frequency band ratios: with vs without human")
    axes[1, 0].set_xlabel("Layer index")
    axes[1, 0].set_ylabel("Fraction of total power")
    axes[1, 0].legend()

    # 4. FFT delta: high-frequency change due to person
    hf_delta = [all_results[l]["fft_delta"]["delta_hf"] for l in layers]
    lf_delta = [all_results[l]["fft_delta"]["delta_lf"] for l in layers]
    axes[1, 1].bar(x, hf_delta, alpha=0.7, label="HF delta", color="red")
    axes[1, 1].bar(x, lf_delta, alpha=0.5, label="LF delta", color="green")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(l) for l in layers], fontsize=8)
    axes[1, 1].set_title("Frequency band delta (with - without)")
    axes[1, 1].set_xlabel("Layer index")
    axes[1, 1].set_ylabel("Change in power fraction")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cross_layer_summary.png"), dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 60)
    print("Forward-Direction Person Signal Analysis")
    print(f"Device: {DEVICE}")
    print(f"With human:    {IMG_WITH}")
    print(f"Without human: {IMG_WITHOUT}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    # Load images
    print("\nLoading images...")
    img_with, arr_with = load_image(IMG_WITH, IMG_SIZE)
    img_without, arr_without = load_image(IMG_WITHOUT, IMG_SIZE)
    print(f"  With human:    {img_with.shape}")
    print(f"  Without human: {img_without.shape}")

    # Forward pass on both
    print("\nForward pass (with human)...")
    cap_with = forward_capture_all(model, img_with)
    print(f"  Captured {len(cap_with)} conv layers")

    print("Forward pass (without human)...")
    cap_without = forward_capture_all(model, img_without)
    print(f"  Captured {len(cap_without)} conv layers")

    # Saliency (with human only — which pixels drive person detection)
    print("\nComputing gradient saliency...")
    # Re-enable gradients temporarily for saliency
    for p in model.parameters():
        p.requires_grad_(True)
    sal_with = compute_saliency(model, img_with)
    sal_without = compute_saliency(model, img_without)
    for p in model.parameters():
        p.requires_grad_(False)

    if sal_with:
        print(f"  Person score (with human): {sal_with['person_score']:.4f}")
        plot_saliency(sal_with, arr_with, os.path.join(OUTPUT_DIR, "saliency_maps"), tag="_with")
    if sal_without:
        print(f"  Person score (without human): {sal_without['person_score']:.4f}")
        plot_saliency(sal_without, arr_without, os.path.join(OUTPUT_DIR, "saliency_maps"), tag="_without")

    # Per-layer analysis
    print(f"\nAnalyzing {len(ANALYSIS_LAYERS)} layers...")
    all_results = {}
    csv_rows = []

    for li in ANALYSIS_LAYERS:
        if li not in cap_with or li not in cap_without:
            print(f"  Layer {li}: not captured, skipping")
            continue

        feat_w = cap_with[li]["output"]
        feat_wo = cap_without[li]["output"]
        C = feat_w.shape[1]
        H, W = feat_w.shape[2], feat_w.shape[3]

        print(f"  Layer {li:3d}: shape=({C}, {H}, {W})  ", end="")

        # 1. Activation stats
        stats_w = compute_activation_stats(feat_w)
        stats_wo = compute_activation_stats(feat_wo)

        # 2. Person delta
        delta = compute_person_delta(feat_w, feat_wo)

        # 3. Raw FFT on feature maps
        fft_w = compute_feature_fft(feat_w, li, top_k=20)
        fft_wo = compute_feature_fft(feat_wo, li, top_k=20)
        fft_delta = compute_fft_delta(feat_w, feat_wo, li)

        # Top person-sensitive channels (by relative delta)
        top_chs = np.argsort(delta["delta_rel"])[::-1][:10].tolist()

        # Store results
        all_results[li] = {
            "shape": (C, H, W),
            "stats_with": stats_w,
            "stats_without": stats_wo,
            "delta": delta,
            "fft_with": fft_w,
            "fft_without": fft_wo,
            "fft_delta": fft_delta,
            "top_channels": top_chs,
        }

        # CSV row
        csv_rows.append({
            "layer": li,
            "channels": C,
            "feat_h": H,
            "feat_w": W,
            "mean_delta_l2": float(delta["delta_l2"].mean()),
            "max_delta_l2": float(delta["delta_l2"].max()),
            "mean_delta_rel": float(delta["delta_rel"].mean()),
            "max_delta_rel": float(delta["delta_rel"].max()),
            "lf_with": fft_w["lf_ratio"],
            "mf_with": fft_w["mf_ratio"],
            "hf_with": fft_w["hf_ratio"],
            "lf_without": fft_wo["lf_ratio"],
            "mf_without": fft_wo["mf_ratio"],
            "hf_without": fft_wo["hf_ratio"],
            "delta_lf": fft_delta["delta_lf"],
            "delta_mf": fft_delta["delta_mf"],
            "delta_hf": fft_delta["delta_hf"],
            "top_ch_by_delta": str(top_chs[:5]),
        })

        print(f"delta_l2={delta['delta_l2'].mean():.4f}  "
              f"delta_rel={delta['delta_rel'].mean():.4f}  "
              f"hf_delta={fft_delta['delta_hf']:+.4f}  "
              f"top_ch={top_chs[:3]}")

        # Plots
        plot_activation_comparison(stats_w, stats_wo, li,
                                   os.path.join(OUTPUT_DIR, "activation_maps"))
        plot_person_delta(delta, li,
                          os.path.join(OUTPUT_DIR, "person_delta_maps"))
        plot_feature_fft(fft_w, li,
                         os.path.join(OUTPUT_DIR, "fft_feature_maps"), tag="_with")
        plot_feature_fft(fft_wo, li,
                         os.path.join(OUTPUT_DIR, "fft_feature_maps"), tag="_without")
        plot_fft_delta(fft_delta, li,
                       os.path.join(OUTPUT_DIR, "fft_feature_maps"))
        plot_top_channels_grid(feat_w, feat_wo, top_chs, li,
                               os.path.join(OUTPUT_DIR, "person_delta_maps"))

    # Cross-layer summary
    print("\nGenerating cross-layer summary...")
    plot_summary(all_results, OUTPUT_DIR)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "person_signal_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_path}")

    # Save JSON summary (top channels per layer)
    json_path = os.path.join(OUTPUT_DIR, "person_signal_summary.json")
    json_data = {}
    for li, res in all_results.items():
        json_data[str(li)] = {
            "shape": list(res["shape"]),
            "mean_delta_l2": float(res["delta"]["delta_l2"].mean()),
            "max_delta_l2": float(res["delta"]["delta_l2"].max()),
            "mean_delta_rel": float(res["delta"]["delta_rel"].mean()),
            "max_delta_rel": float(res["delta"]["delta_rel"].max()),
            "lf_with": float(res["fft_with"]["lf_ratio"]),
            "hf_with": float(res["fft_with"]["hf_ratio"]),
            "lf_without": float(res["fft_without"]["lf_ratio"]),
            "hf_without": float(res["fft_without"]["hf_ratio"]),
            "delta_lf": float(res["fft_delta"]["delta_lf"]),
            "delta_hf": float(res["fft_delta"]["delta_hf"]),
            "top_10_person_channels": [int(c) for c in res["top_channels"]],
        }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Person Signal Per Layer")
    print("=" * 60)
    print(f"\n{'Layer':>6}  {'Shape':>14}  {'Mean Delta':>10}  {'Max Delta':>10}  "
          f"{'Mean Rel':>10}  {'HF Delta':>10}  {'Top Channels'}")
    print("-" * 90)
    for li in sorted(all_results.keys()):
        r = all_results[li]
        s = r["shape"]
        print(f"{li:6d}  ({s[0]:3d},{s[1]:3d},{s[2]:3d})  "
              f"{r['delta']['delta_l2'].mean():10.4f}  "
              f"{r['delta']['delta_l2'].max():10.4f}  "
              f"{r['delta']['delta_rel'].mean():10.4f}  "
              f"{r['fft_delta']['delta_hf']:+10.4f}  "
              f"{r['top_channels'][:5]}")

    # Identify the most person-sensitive layers
    print("\n" + "=" * 60)
    print("RANKING: Most Person-Sensitive Layers (by mean relative delta)")
    print("=" * 60)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["delta"]["delta_rel"].mean(), reverse=True)
    for rank, (li, r) in enumerate(ranked[:5], 1):
        print(f"  #{rank} Layer {li:3d}: mean_rel_delta={r['delta']['delta_rel'].mean():.4f}, "
              f"max_rel_delta={r['delta']['delta_rel'].max():.4f}, "
              f"top_channels={r['top_channels'][:5]}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
