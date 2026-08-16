"""
Channel Anchor Probe & Cloud Poisoning Pipeline Simulator

Three-phase experiment:

PHASE 1: Anchor Channel Extraction
  - Hook channels 170, 171 (and neighbors) at L93 (52x52) and L105 (26x26)
  - Inject 1/196 digit patterns with known encoding
  - Measure per-channel SNR of the carry structure
  - Determine which channels carry the most readable signal
  - Build a decoder that reads the 1/196 payload from specific channel activations

PHASE 2: Patch → YOLOv3 → Embedding Pipeline
  - Simulate a patch/sticker on the torso region (20% area)
  - Run through YOLOv3 with full feature capture
  - Extract the embedding (feature vector at detection head input)
  - Measure how much of the 1/196 payload survives in the embedding
  - Test different patch shapes (irregular polygon, triangle) with digit textures

PHASE 3: Cloud Poisoning Simulation
  - Simulate the attack scenario: physical patch → camera → YOLOv3 → embeddings → cloud model
  - Generate poisoned embeddings with encoded 1/196 payload
  - Show that the payload is detectable in the embedding output
  - Calculate how many poisoned samples needed to influence a downstream model
  - Test if the payload survives common embedding transformations (L2 norm, PCA, quantization)
"""

import os, sys, json, csv, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from scipy.stats import pearsonr

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")

from pytorchyolo.models import Darknet

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\anchor_channel"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416

COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLOv3 detection head layers
# L93 = 52x52 head (small objects), L105 = 26x26 head (large objects)
# Actually: L93 is the yolo layer for 26x26, L105 is for 13x13
# Let me check: YOLOv3 has 3 detection heads at 52x52, 26x26, 13x13
# The conv layers before each yolo layer are where we hook
DETECTION_LAYERS = {
    "L81_52x52": 81,   # Before first yolo (52x52, small objects)
    "L93_26x26": 93,   # Before second yolo (26x26, medium objects)
    "L105_13x13": 105, # Before third yolo (13x13, large objects)
}

# Anchor channels to probe (from research context: channels 170, 171)
ANCHOR_CHANNELS = [168, 169, 170, 171, 172, 173, 174, 175]
# Also probe a broader set
BROAD_CHANNELS = list(range(160, 180)) + list(range(0, 20)) + list(range(240, 256))


# ============================================================
# Utilities
# ============================================================

def get_decimal_expansion(numerator, denominator, num_digits=500):
    getcontext().prec = num_digits + 50
    val = Decimal(numerator) / Decimal(denominator)
    digits_str = str(val)[2:]
    return [int(d) for d in digits_str[:num_digits]]

def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return arr

def make_sinusoid(H, W, kx, ky, phase_deg, amp):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.cos(2*np.pi*(kx/W*x + ky/H*y) + np.radians(phase_deg))).astype(np.float32)

def make_square_wave(H, W, kx, ky, amp):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.sign(np.cos(2*np.pi*(kx/W*x + ky/H*y)))).astype(np.float32)

def add_pattern(arr, pat):
    out = arr.copy()
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] + pat, 0, 1)
    return out

def add_pattern_patch(arr, pat, mask):
    out = arr.copy()
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] * (1 - mask) + (out[:,:,c] + pat) * mask, 0, 1)
    return out

def make_deformable_mask(H, W, cx, cy, ray_lengths, num_rays=None):
    R = num_rays if num_rays else len(ray_lengths)
    dtheta = 2 * math.pi / R
    endpoints = []
    for i in range(R):
        angle = i * dtheta
        r = ray_lengths[i % len(ray_lengths)]
        ex = cx + r * math.cos(angle)
        ey = cy + r * math.sin(angle)
        endpoints.append((ex, ey))
    mask = np.zeros((H, W), dtype=np.float32)
    polygon = endpoints + [endpoints[0]]
    img_mask = Image.new("F", (W, H), 0.0)
    draw = ImageDraw.Draw(img_mask)
    draw.polygon([(p[0], p[1]) for p in polygon], fill=1.0)
    mask = np.array(img_mask, dtype=np.float32)
    return mask, endpoints

def digits_to_spatial_1d(digits, width, height, amp=0.15, mode="row"):
    pattern = np.zeros((height, width), dtype=np.float32)
    if mode == "row":
        for x in range(width):
            d = digits[x % len(digits)]
            pattern[:, x] = (d / 9.0) * amp
    elif mode == "tile":
        idx = 0
        for y in range(height):
            for x in range(width):
                d = digits[idx % len(digits)]
                pattern[y, x] = (d / 9.0) * amp
                idx += 1
    return pattern

def digits_to_spatial_2d_slots(digits, width, height, slot_width=2, amp=0.15):
    slot_values = []
    for i in range(0, len(digits) - slot_width + 1, slot_width):
        val = 0
        for j in range(slot_width):
            val = val * 10 + digits[i + j]
        slot_values.append(val)
    pattern = np.zeros((height, width), dtype=np.float32)
    for x in range(width):
        sv = slot_values[x % len(slot_values)]
        pattern[:, x] = (sv / 99.0) * amp
    return pattern, slot_values


# ============================================================
# YOLOv3 forward pass with full feature capture
# ============================================================

def forward_capture_v3(model, x):
    caps = {}
    los = []
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional", "upsample", "maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups", 1))
            gi = int(md.get("group_id", 0))
            x = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, x.size(2) if hasattr(x, 'size') else 416)
        if md["type"] == "convolutional":
            caps[i] = x.detach().clone()
        los.append(x)
    return caps, x

def get_dets_v3(model, x, conf=0.1):
    with torch.no_grad():
        output = model(x)
    dets = []
    if output is None: return dets
    out = output.cpu().numpy()
    if out.ndim == 3: out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
            })
    return dets


# ============================================================
# PHASE 1: Anchor Channel Extraction
# ============================================================

def phase1_anchor_channels(v3_model, arr_base):
    """
    Hook specific channels at detection head layers.
    Inject known 1/196 digit patterns and measure which channels carry
    the most readable signal. Build a decoder.
    """
    print(f"\n{'='*70}")
    print("PHASE 1: Anchor Channel Extraction")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    digits_196 = get_decimal_expansion(1, 196, 500)

    # Get baseline feature maps
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)

    # Inject digit pattern
    pattern = digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="row")
    arr_mod = add_pattern(arr_base, pattern)
    tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mod_caps, _ = forward_capture_v3(v3_model, tensor_mod)

    # Analyze each detection head layer
    results = {}

    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if layer_idx not in mod_caps or layer_idx not in base_caps:
            print(f"  {layer_name}: layer not found, skipping")
            continue

        fm_mod = mod_caps[layer_idx][0].cpu().numpy()  # (C, H_l, W_l)
        fm_base = base_caps[layer_idx][0].cpu().numpy()
        delta = fm_mod - fm_base  # (C, H_l, W_l)

        C, fh, fw = delta.shape
        print(f"\n  {layer_name} (L{layer_idx}): {C} channels, {fh}x{fw} spatial")

        # Per-channel analysis
        channel_data = []
        for ch in range(C):
            ch_delta = delta[ch].flatten()
            ch_mod = fm_mod[ch].flatten()
            ch_base = fm_base[ch].flatten()

            # Signal strength
            mean_abs = float(np.mean(np.abs(ch_delta)))
            max_abs = float(np.max(np.abs(ch_delta)))
            std_delta = float(np.std(ch_delta))

            # SNR: signal power / noise power
            # Signal = delta (what changed due to injection)
            # Noise = baseline activation variance
            signal_power = float(np.mean(ch_delta ** 2))
            noise_power = float(np.var(ch_base))
            snr_db = 10 * math.log10(signal_power / (noise_power + 1e-10)) if signal_power > 0 else -100

            # Check if the 42-digit period structure is visible in this channel
            # Compute spatial autocorrelation at period lag
            if fh > 1 and fw > 10:
                # 1D autocorrelation along x axis (where digit pattern was applied)
                ch_2d = delta[ch]  # (fh, fw)
                row_avg = ch_2d.mean(axis=0)  # (fw,)
                if row_avg.std() > 1e-8:
                    # Autocorrelation at lag=42 (the period)
                    ac = np.correlate(row_avg, row_avg, mode='full')
                    ac_center = len(ac) // 2
                    if ac_center + 42 < len(ac):
                        ac_at_42 = float(ac[ac_center + 42] / (ac[ac_center] + 1e-10))
                    else:
                        ac_at_42 = 0
                else:
                    ac_at_42 = 0
            else:
                ac_at_42 = 0

            channel_data.append({
                "channel": ch,
                "mean_abs_delta": mean_abs,
                "max_abs_delta": max_abs,
                "std_delta": std_delta,
                "snr_db": snr_db,
                "period_autocorr_42": ac_at_42,
            })

        # Sort by SNR
        sorted_channels = sorted(channel_data, key=lambda x: -x["snr_db"])

        # Top 20 channels by SNR
        top20 = sorted_channels[:20]
        print(f"    Top 5 channels by SNR:")
        for cd in top20[:5]:
            print(f"      Ch{cd['channel']:3d}: SNR={cd['snr_db']:+.2f}dB, "
                  f"delta={cd['mean_abs_delta']:.6f}, AC@42={cd['period_autocorr_42']:.4f}")

        # Specifically check anchor channels 170, 171
        anchor_results = {}
        for ach in ANCHOR_CHANNELS:
            if ach < C:
                cd = channel_data[ach]
                anchor_results[ach] = cd
                rank = next(i for i, c in enumerate(sorted_channels) if c["channel"] == ach)
                print(f"    Anchor Ch{ach}: SNR={cd['snr_db']:+.2f}dB (rank {rank}/{C}), "
                      f"AC@42={cd['period_autocorr_42']:.4f}")

        # Find channels with strongest period-42 autocorrelation
        sorted_by_ac = sorted(channel_data, key=lambda x: -abs(x["period_autocorr_42"]))
        top_ac = sorted_by_ac[:10]
        print(f"    Top 5 channels by period-42 autocorrelation:")
        for cd in top_ac[:5]:
            print(f"      Ch{cd['channel']:3d}: AC@42={cd['period_autocorr_42']:.4f}, "
                  f"SNR={cd['snr_db']:+.2f}dB")

        results[layer_name] = {
            "shape": [C, fh, fw],
            "top20_channels": top20,
            "anchor_channels": anchor_results,
            "top_period42_channels": top_ac,
            "all_channels": channel_data,
        }

        # Save heatmap of anchor channel activations
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ach in enumerate(ANCHOR_CHANNELS[:8]):
            ax = axes[i//4, i%4]
            if ach < C:
                ax.imshow(delta[ach], cmap='RdBu_r', aspect='auto')
                ax.set_title(f"Ch{ach} delta (SNR={channel_data[ach]['snr_db']:+.1f}dB)")
                ax.axis('off')
        plt.suptitle(f"{layer_name} — Anchor Channel Deltas (1/196 injection)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase1_{layer_name}_anchor_heatmaps.png", dpi=150)
        plt.close()

        # Save top-SNR channel heatmaps
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i, cd in enumerate(top20[:10]):
            ax = axes[i//5, i%5]
            ch = cd["channel"]
            ax.imshow(delta[ch], cmap='RdBu_r', aspect='auto')
            ax.set_title(f"Ch{ch} (SNR={cd['snr_db']:+.1f}dB, AC={cd['period_autocorr_42']:.3f})")
            ax.axis('off')
        plt.suptitle(f"{layer_name} — Top 10 Channels by SNR", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase1_{layer_name}_top_snr.png", dpi=150)
        plt.close()

    return results


# ============================================================
# PHASE 1b: Decoder — extract payload from channel activations
# ============================================================

def phase1b_decoder(v3_model, arr_base):
    """
    Build and test a decoder that reads the 1/196 payload from
    specific channel activations at the detection head.
    """
    print(f"\n{'='*70}")
    print("PHASE 1b: Decoder — Extract Payload from Channel Activations")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape

    # Encode 8 different payloads (3-bit encoding) using different amplitudes
    # of the 1/196 digit pattern
    payloads = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # 8 levels = 3 bits
    digits_196 = get_decimal_expansion(1, 196, 500)

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)

    # Collect embeddings for each payload
    embeddings = {}  # layer -> payload -> channel activations

    for layer_name, layer_idx in DETECTION_LAYERS.items():
        embeddings[layer_name] = {}

    for payload_idx, scale in enumerate(payloads):
        # Inject digit pattern at this amplitude
        amp = 0.05 * scale
        pattern = digits_to_spatial_1d(digits_196, W, H, amp=amp, mode="row")
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mod_caps, _ = forward_capture_v3(v3_model, tensor_mod)

        for layer_name, layer_idx in DETECTION_LAYERS.items():
            if layer_idx not in mod_caps:
                continue
            fm_mod = mod_caps[layer_idx][0].cpu().numpy()
            fm_base = base_caps[layer_idx][0].cpu().numpy()
            delta = fm_mod - fm_base  # (C, H_l, W_l)

            # Extract embedding: mean activation per channel (the "embedding" a cloud model would receive)
            channel_means = delta.mean(axis=(1, 2))  # (C,)
            channel_stds = delta.std(axis=(1, 2))  # (C,)
            channel_maxs = np.abs(delta).max(axis=(1, 2))  # (C,)

            embeddings[layer_name][payload_idx] = {
                "scale": scale,
                "channel_means": channel_means.tolist(),
                "channel_stds": channel_stds.tolist(),
                "channel_maxs": channel_maxs.tolist(),
            }

    # Test decoder: can we recover the payload from channel activations?
    print("\n  Decoder test — recovering payload from channel activations:")
    decoder_results = {}

    for layer_name in DETECTION_LAYERS:
        if layer_name not in embeddings or not embeddings[layer_name]:
            continue

        # Build feature matrix: each row = a payload, columns = channel features
        payload_scales = []
        features = []
        for pidx in sorted(embeddings[layer_name].keys()):
            ed = embeddings[layer_name][pidx]
            payload_scales.append(ed["scale"])
            # Feature vector: [channel_means, channel_stds, channel_maxs] for top channels
            feat = np.concatenate([ed["channel_means"], ed["channel_stds"], ed["channel_maxs"]])
            features.append(feat)

        features = np.array(features)
        payload_scales = np.array(payload_scales)

        # Test linear decoder: can a linear combination of channels recover the payload?
        # Use correlation between each channel feature and the payload scale
        n_features = features.shape[1]
        correlations = []
        for fi in range(n_features):
            if features[:, fi].std() > 1e-8:
                r, p = pearsonr(features[:, fi], payload_scales)
                correlations.append({"feature_idx": fi, "r": float(r), "p": float(p)})
            else:
                correlations.append({"feature_idx": fi, "r": 0, "p": 1})

        # Sort by absolute correlation
        sorted_corrs = sorted(correlations, key=lambda x: -abs(x["r"]))
        top_features = sorted_corrs[:10]

        # Best single-channel decoder
        best = top_features[0]
        print(f"\n  {layer_name}:")
        print(f"    Best decoder feature: idx={best['feature_idx']}, r={best['r']:+.4f}, p={best['p']:.6f}")
        print(f"    Top 5 features:")
        for tf in top_features[:5]:
            sig = "***" if tf["p"] < 0.05 else ""
            print(f"      idx={tf['feature_idx']:4d}: r={tf['r']:+.4f}, p={tf['p']:.6f} {sig}")

        # Multi-channel decoder: linear regression
        from numpy.linalg import lstsq
        # Solve: payload_scales = features @ weights
        # Add bias term
        X = np.column_stack([features, np.ones(len(payload_scales))])
        weights, residuals, rank, sv = lstsq(X, payload_scales, rcond=None)
        predicted = X @ weights
        decode_corr, decode_p = pearsonr(predicted, payload_scales)
        rmse = float(np.sqrt(np.mean((predicted - payload_scales) ** 2)))

        print(f"    Multi-channel decoder: r={decode_corr:+.4f}, RMSE={rmse:.4f}")

        decoder_results[layer_name] = {
            "top_features": top_features,
            "best_single_r": best["r"],
            "best_single_p": best["p"],
            "multichannel_r": float(decode_corr),
            "multichannel_rmse": rmse,
            "n_features": n_features,
        }

        # Plot: predicted vs actual payload
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(payload_scales, predicted, c='blue', s=100, zorder=5)
        ax.plot([min(payload_scales), max(payload_scales)], [min(payload_scales), max(payload_scales)],
                'r--', alpha=0.5, label="perfect decode")
        ax.set_xlabel("Actual Payload Scale")
        ax.set_ylabel("Decoded Payload Scale")
        ax.set_title(f"{layer_name} — Multi-Channel Decoder (r={decode_corr:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase1b_{layer_name}_decoder.png", dpi=150)
        plt.close()

    return {"embeddings": embeddings, "decoder_results": decoder_results,
            "payloads": payloads}


# ============================================================
# PHASE 2: Patch → YOLOv3 → Embedding Pipeline
# ============================================================

def phase2_patch_pipeline(v3_model, arr_base):
    """
    Simulate a physical patch (sticker/shirt) with 1/196 digit texture
    placed on the torso. Extract the embedding that would be sent to a cloud model.
    """
    print(f"\n{'='*70}")
    print("PHASE 2: Patch → YOLOv3 → Embedding Pipeline")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    digits_196 = get_decimal_expansion(1, 196, 500)

    # Patch placement: torso center
    patch_cx, patch_cy = 208, 240

    # Create patch shapes (irregular polygon = best from triangular_patch_test)
    shapes = {
        "deformed_r12_large": make_deformable_mask(
            H, W, patch_cx, patch_cy,
            [110, 90, 105, 85, 100, 95, 115, 80, 100, 90, 110, 85], 12),
        "triangle_r100": make_deformable_mask(H, W, patch_cx, patch_cy, [100]*3, 3),
        "circle_r80": make_deformable_mask(H, W, patch_cx, patch_cy, [80]*32, 32),
    }

    # Patch textures
    textures = {
        "digits_196_row": digits_to_spatial_1d(digits_196, W, H, amp=0.20, mode="row"),
        "digits_196_tile": digits_to_spatial_1d(digits_196, W, H, amp=0.20, mode="tile"),
        "slots_2digit": digits_to_spatial_2d_slots(digits_196, W, H, amp=0.20)[0],
        "k167_d": make_sinusoid(H, W, 167, 167, 0, 0.20),
        "stripes_v_13px": (0.20 * np.sign(np.sin(2 * np.pi * np.arange(W) / 13.0))).astype(np.float32) * np.ones((H, W), dtype=np.float32),
        "composite_digits_k167": (digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="row") +
                                   make_sinusoid(H, W, 167, 167, 0, 0.10)).astype(np.float32),
    }

    # Get baseline (no patch)
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])
    print(f"  Baseline: {base_person} person detections")

    # Baseline embedding (what cloud model receives without patch)
    base_embeddings = {}
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if layer_idx in base_caps:
            fm = base_caps[layer_idx][0].cpu().numpy()
            # Embedding = flattened mean activation per channel
            base_embeddings[layer_name] = fm.mean(axis=(1, 2))

    results = {}

    for shape_name, (mask, endpoints) in shapes.items():
        patch_area = float(np.mean(mask)) * 100
        shape_results = {}

        for tex_name, texture in textures.items():
            combo = f"{shape_name}__{tex_name}"

            # Apply patch
            arr_mod = add_pattern_patch(arr_base, texture, mask)
            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mod_caps, _ = forward_capture_v3(v3_model, tensor_mod)
            dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
            person_count = len([d for d in dets if d["class_name"] == "person"])

            # Extract embedding
            patch_embeddings = {}
            embedding_corruption = {}  # How much the embedding changed

            for layer_name, layer_idx in DETECTION_LAYERS.items():
                if layer_idx not in mod_caps or layer_name not in base_embeddings:
                    continue

                fm_mod = mod_caps[layer_idx][0].cpu().numpy()
                patch_emb = fm_mod.mean(axis=(1, 2))  # (C,)
                base_emb = base_embeddings[layer_name]

                # Embedding corruption metrics
                l2_dist = float(np.linalg.norm(patch_emb - base_emb))
                cos_sim = float(np.dot(patch_emb, base_emb) /
                               (np.linalg.norm(patch_emb) * np.linalg.norm(base_emb) + 1e-10))
                kl_div = float(np.sum(patch_emb * np.log((patch_emb + 1e-10) / (base_emb + 1e-10))))

                # Payload survival: how much of the 1/196 signal is in the embedding?
                # Check if embedding correlates with the digit pattern structure
                delta_emb = patch_emb - base_emb

                # Check for 1/196 value in embedding
                target_val = 1.0 / 196.0
                near_target = float(np.mean(np.abs(delta_emb - target_val) < 0.1 * target_val))
                near_neg_target = float(np.mean(np.abs(delta_emb + target_val) < 0.1 * target_val))

                # SNR of the embedding change
                signal_power = float(np.mean(delta_emb ** 2))
                noise_power = float(np.var(base_emb))
                snr_db = 10 * math.log10(signal_power / (noise_power + 1e-10)) if signal_power > 0 else -100

                patch_embeddings[layer_name] = {
                    "embedding": patch_emb.tolist(),
                    "delta_embedding": delta_emb.tolist(),
                    "l2_distance": l2_dist,
                    "cosine_similarity": cos_sim,
                    "kl_divergence": kl_div,
                    "near_inv196": near_target,
                    "near_neg_inv196": near_neg_target,
                    "snr_db": snr_db,
                }
                embedding_corruption[layer_name] = {
                    "l2": l2_dist, "cos_sim": cos_sim, "snr_db": snr_db,
                    "near_inv196": near_target,
                }

            shape_results[tex_name] = {
                "person_count": person_count,
                "patch_area_pct": patch_area,
                "suppressed": person_count == 0 and base_person > 0,
                "person_reduction": base_person - person_count,
                "embedding_corruption": embedding_corruption,
            }

            tag = ""
            if person_count == 0: tag = " [SUPPRESSED]"
            elif person_count < base_person: tag = f" [person {base_person}->{person_count}]"

            # Print embedding corruption summary
            l105_corr = embedding_corruption.get("L105_13x13", {})
            print(f"  {combo:45s}: person={person_count}/{base_person}, "
                  f"L105 L2={l105_corr.get('l2',0):.4f}, cos={l105_corr.get('cos_sim',0):.4f}, "
                  f"SNR={l105_corr.get('snr_db',0):+.1f}dB{tag}")

        results[shape_name] = shape_results

    return {"results": results, "base_person": base_person,
            "base_embedding_norm": {k: float(np.linalg.norm(v)) for k, v in base_embeddings.items()}}


# ============================================================
# PHASE 3: Cloud Poisoning Simulation
# ============================================================

def phase3_cloud_poisoning(v3_model, arr_base):
    """
    Simulate the full attack pipeline:
    physical patch → camera → YOLOv3 → embeddings → cloud model training data

    Test if poisoned embeddings:
    1. Carry detectable 1/196 payload
    2. Survive common embedding transformations (L2 norm, PCA, quantization)
    3. Can be detected by a poisoned-sample detector
    4. Estimate how many poisoned samples needed to influence downstream model
    """
    print(f"\n{'='*70}")
    print("PHASE 3: Cloud Poisoning Simulation")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    digits_196 = get_decimal_expansion(1, 196, 500)

    # Create patch (best combo from phase 2: deformed_r12 + digits)
    patch_cx, patch_cy = 208, 240
    mask, endpoints = make_deformable_mask(
        H, W, patch_cx, patch_cy,
        [110, 90, 105, 85, 100, 95, 115, 80, 100, 90, 110, 85], 12)

    # Get baseline embedding
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)

    # Generate N poisoned embeddings with different payload amplitudes
    N_POISONED = 50
    amplitudes = np.linspace(0.01, 0.30, N_POISONED)

    print(f"  Generating {N_POISONED} poisoned embeddings...")

    poisoned_embeddings = {layer_name: [] for layer_name in DETECTION_LAYERS}
    clean_embeddings = {layer_name: [] for layer_name in DETECTION_LAYERS}

    # Also generate clean embeddings with slight noise (simulating natural variation)
    rng = np.random.RandomState(42)

    for i, amp in enumerate(amplitudes):
        # Poisoned: patch with digit pattern at this amplitude
        pattern = digits_to_spatial_1d(digits_196, W, H, amp=amp, mode="row")
        arr_mod = add_pattern_patch(arr_base, pattern, mask)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mod_caps, _ = forward_capture_v3(v3_model, tensor_mod)

        for layer_name, layer_idx in DETECTION_LAYERS.items():
            if layer_idx in mod_caps:
                fm = mod_caps[layer_idx][0].cpu().numpy()
                emb = fm.mean(axis=(1, 2))
                poisoned_embeddings[layer_name].append(emb)

        # Clean: slight random noise (natural camera variation)
        noise = rng.randn(*arr_base.shape) * 0.01
        arr_clean = np.clip(arr_base + noise, 0, 1).astype(np.float32)
        tensor_clean = torch.from_numpy(arr_clean).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            clean_caps, _ = forward_capture_v3(v3_model, tensor_clean)

        for layer_name, layer_idx in DETECTION_LAYERS.items():
            if layer_idx in clean_caps:
                fm = clean_caps[layer_idx][0].cpu().numpy()
                emb = fm.mean(axis=(1, 2))
                clean_embeddings[layer_name].append(emb)

    results = {}

    for layer_name in DETECTION_LAYERS:
        if not poisoned_embeddings[layer_name] or not clean_embeddings[layer_name]:
            continue

        poison_mat = np.array(poisoned_embeddings[layer_name])  # (N, C)
        clean_mat = np.array(clean_embeddings[layer_name])  # (N, C)
        base_emb = base_caps[DETECTION_LAYERS[layer_name]][0].cpu().numpy().mean(axis=(1, 2))

        print(f"\n  {layer_name}: {poison_mat.shape[1]} channels, {N_POISONED} samples each")

        # 1. Payload detectability: can we distinguish poisoned from clean?
        # Compute distance from baseline for each
        poison_dist = np.linalg.norm(poison_mat - base_emb, axis=1)
        clean_dist = np.linalg.norm(clean_mat - base_emb, axis=1)

        # Separability: how well can we distinguish?
        mean_poison_dist = float(poison_dist.mean())
        mean_clean_dist = float(clean_dist.mean())
        std_poison_dist = float(poison_dist.std())
        std_clean_dist = float(clean_dist.std())

        # Overlap coefficient (lower = more separable)
        from scipy.stats import norm
        if std_poison_dist > 0 and std_clean_dist > 0:
            # Simple separability metric
            sep = abs(mean_poison_dist - mean_clean_dist) / (std_poison_dist + std_clean_dist + 1e-10)
        else:
            sep = 0

        print(f"    Poison dist: {mean_poison_dist:.4f} +/- {std_poison_dist:.4f}")
        print(f"    Clean dist:  {mean_clean_dist:.4f} +/- {std_clean_dist:.4f}")
        print(f"    Separability: {sep:.4f}")

        # 2. Payload survival through transformations
        # L2 normalization (common in embedding pipelines)
        poison_l2 = poison_mat / (np.linalg.norm(poison_mat, axis=1, keepdims=True) + 1e-10)
        clean_l2 = clean_mat / (np.linalg.norm(clean_mat, axis=1, keepdims=True) + 1e-10)
        base_l2 = base_emb / (np.linalg.norm(base_emb) + 1e-10)

        poison_l2_dist = np.linalg.norm(poison_l2 - base_l2, axis=1)
        clean_l2_dist = np.linalg.norm(clean_l2 - base_l2, axis=1)
        sep_l2 = abs(poison_l2_dist.mean() - clean_l2_dist.mean()) / (poison_l2_dist.std() + clean_l2_dist.std() + 1e-10)

        # Quantization (8-bit, common in edge deployment)
        poison_quant = np.round(poison_mat * 127).astype(np.int8).astype(np.float32) / 127.0
        clean_quant = np.round(clean_mat * 127).astype(np.int8).astype(np.float32) / 127.0
        base_quant = np.round(base_emb * 127).astype(np.int8).astype(np.float32) / 127.0
        poison_q_dist = np.linalg.norm(poison_quant - base_quant, axis=1)
        clean_q_dist = np.linalg.norm(clean_quant - base_quant, axis=1)
        sep_quant = abs(poison_q_dist.mean() - clean_q_dist.mean()) / (poison_q_dist.std() + clean_q_dist.std() + 1e-10)

        # PCA (reduce to 32 dims, common in embedding compression)
        from sklearn.decomposition import PCA
        all_emb = np.vstack([poison_mat, clean_mat])
        pca = PCA(n_components=min(32, min(all_emb.shape)))
        all_pca = pca.fit_transform(all_emb)
        poison_pca = all_pca[:N_POISONED]
        clean_pca = all_pca[N_POISONED:]
        base_pca = pca.transform(base_emb.reshape(1, -1))[0]
        poison_pca_dist = np.linalg.norm(poison_pca - base_pca, axis=1)
        clean_pca_dist = np.linalg.norm(clean_pca - base_pca, axis=1)
        sep_pca = abs(poison_pca_dist.mean() - clean_pca_dist.mean()) / (poison_pca_dist.std() + clean_pca_dist.std() + 1e-10)

        print(f"    L2 norm separability: {sep_l2:.4f}")
        print(f"    8-bit quant separability: {sep_quant:.4f}")
        print(f"    PCA-32 separability: {sep_pca:.4f}")

        # 3. Correlation between payload amplitude and embedding distance
        r_amp, p_amp = pearsonr(amplitudes, poison_dist)
        print(f"    Corr(amp, embedding_dist): r={r_amp:+.4f}, p={p_amp:.6f}")

        # 4. Estimate poisoning fraction needed
        # If separability is d, then fraction needed ~ 1/(1+d^2) (rough estimate)
        # More precisely: if we can shift the mean embedding by delta, how many samples
        # to shift the cloud model's learned representation by 1 std?
        delta_mean = poison_mat.mean(axis=0) - clean_mat.mean(axis=0)
        clean_std = clean_mat.std(axis=0).mean()
        shift_per_sample = np.linalg.norm(delta_mean) / (clean_std * np.sqrt(len(poison_mat)))
        # To shift by 1 std: need 1/shift_per_sample^2 samples
        n_needed = float(1.0 / (shift_per_sample ** 2 + 1e-10))
        poison_fraction = n_needed / (n_needed + 1000)  # fraction of 1000-sample dataset

        print(f"    Embedding shift per sample: {shift_per_sample:.6f}")
        print(f"    Samples needed for 1-std shift: {n_needed:.0f}")
        print(f"    Poison fraction (of 1000): {poison_fraction:.4f} ({poison_fraction*100:.1f}%)")

        results[layer_name] = {
            "separability_raw": float(sep),
            "separability_l2norm": float(sep_l2),
            "separability_quant8bit": float(sep_quant),
            "separability_pca32": float(sep_pca),
            "corr_amp_dist": float(r_amp),
            "corr_amp_p": float(p_amp),
            "mean_poison_dist": mean_poison_dist,
            "mean_clean_dist": mean_clean_dist,
            "shift_per_sample": float(shift_per_sample),
            "samples_needed_1std": n_needed,
            "poison_fraction_1000": float(poison_fraction),
        }

        # Plot: poisoned vs clean embedding distances
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist([clean_dist, poison_dist], bins=20, label=["Clean", "Poisoned"], alpha=0.7)
        axes[0].set_title(f"{layer_name} — Embedding Distance from Baseline")
        axes[0].set_xlabel("L2 Distance"); axes[0].legend()
        axes[1].scatter(amplitudes, poison_dist, c='red', label="Poisoned", s=20)
        axes[1].axhline(y=mean_clean_dist, color='blue', linestyle='--', label="Clean mean")
        axes[1].set_title(f"Payload Amplitude vs Embedding Distance (r={r_amp:.3f})")
        axes[1].set_xlabel("Patch Amplitude"); axes[1].set_ylabel("L2 Distance"); axes[1].legend()
        axes[2].bar(["Raw", "L2-norm", "8-bit", "PCA-32"],
                    [sep, sep_l2, sep_quant, sep_pca], color='steelblue')
        axes[2].set_title("Separability After Transformations")
        axes[2].set_ylabel("Separability Score")
        plt.suptitle(f"Phase 3: {layer_name} — Cloud Poisoning Analysis", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase3_{layer_name}_poisoning.png", dpi=150)
        plt.close()

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("ANCHOR CHANNEL PROBE & CLOUD POISONING PIPELINE")
    print("="*70)
    print(f"Device: {DEVICE}")

    arr_base = load_image(IMG_WITH, IMG_SIZE)

    # Load YOLOv3
    print("\nLoading YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters():
        p.requires_grad_(False)

    all_results = {}

    # Phase 1: Anchor channel extraction
    all_results["phase1_anchor_channels"] = phase1_anchor_channels(v3_model, arr_base)

    # Phase 1b: Decoder
    all_results["phase1b_decoder"] = phase1b_decoder(v3_model, arr_base)

    # Phase 2: Patch pipeline
    all_results["phase2_patch_pipeline"] = phase2_patch_pipeline(v3_model, arr_base)

    # Phase 3: Cloud poisoning
    all_results["phase3_cloud_poisoning"] = phase3_cloud_poisoning(v3_model, arr_base)

    del v3_model
    torch.cuda.empty_cache()

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\n{'='*70}")
    print("Saving results...")

    json_path = f"{OUTPUT_DIR}/anchor_channel.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # CSV summary
    csv_path = f"{OUTPUT_DIR}/anchor_channel.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "layer", "metric", "value"])

        # Phase 1: top channels
        for layer_name, lr in all_results["phase1_anchor_channels"].items():
            for cd in lr.get("top20_channels", [])[:5]:
                writer.writerow(["phase1", layer_name, f"ch{cd['channel']}_snr", cd["snr_db"]])
                writer.writerow(["phase1", layer_name, f"ch{cd['channel']}_ac42", cd["period_autocorr_42"]])

        # Phase 1b: decoder
        for layer_name, dr in all_results["phase1b_decoder"]["decoder_results"].items():
            writer.writerow(["phase1b", layer_name, "best_single_r", dr["best_single_r"]])
            writer.writerow(["phase1b", layer_name, "multichannel_r", dr["multichannel_r"]])
            writer.writerow(["phase1b", layer_name, "multichannel_rmse", dr["multichannel_rmse"]])

        # Phase 2: embedding corruption
        for shape_name, sr in all_results["phase2_patch_pipeline"]["results"].items():
            for tex_name, tr in sr.items():
                for layer_name, ec in tr.get("embedding_corruption", {}).items():
                    writer.writerow(["phase2", f"{shape_name}__{tex_name}__{layer_name}", "l2", ec["l2"]])
                    writer.writerow(["phase2", f"{shape_name}__{tex_name}__{layer_name}", "cos_sim", ec["cos_sim"]])
                    writer.writerow(["phase2", f"{shape_name}__{tex_name}__{layer_name}", "snr_db", ec["snr_db"]])

        # Phase 3: poisoning
        for layer_name, pr in all_results["phase3_cloud_poisoning"].items():
            writer.writerow(["phase3", layer_name, "separability_raw", pr["separability_raw"]])
            writer.writerow(["phase3", layer_name, "separability_l2norm", pr["separability_l2norm"]])
            writer.writerow(["phase3", layer_name, "separability_quant8bit", pr["separability_quant8bit"]])
            writer.writerow(["phase3", layer_name, "separability_pca32", pr["separability_pca32"]])
            writer.writerow(["phase3", layer_name, "poison_fraction_1000", pr["poison_fraction_1000"]])
    print(f"Saved CSV: {csv_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\nPHASE 1: Anchor Channels (top 3 by SNR at each detection head):")
    for layer_name, lr in all_results["phase1_anchor_channels"].items():
        for cd in lr.get("top20_channels", [])[:3]:
            print(f"  {layer_name}: Ch{cd['channel']:3d} SNR={cd['snr_db']:+.2f}dB AC@42={cd['period_autocorr_42']:.4f}")

    print("\nPHASE 1b: Decoder (multi-channel linear regression):")
    for layer_name, dr in all_results["phase1b_decoder"]["decoder_results"].items():
        print(f"  {layer_name}: r={dr['multichannel_r']:+.4f}, RMSE={dr['multichannel_rmse']:.4f}")

    print("\nPHASE 2: Patch Pipeline (embedding corruption at L105):")
    for shape_name, sr in all_results["phase2_patch_pipeline"]["results"].items():
        for tex_name, tr in sr.items():
            ec = tr.get("embedding_corruption", {}).get("L105_13x13", {})
            tag = " [SUPP]" if tr["suppressed"] else f" [person {tr.get('person_count','?')}]"
            print(f"  {shape_name}__{tex_name}: L2={ec.get('l2',0):.4f}, "
                  f"cos={ec.get('cos_sim',0):.4f}, SNR={ec.get('snr_db',0):+.1f}dB{tag}")

    print("\nPHASE 3: Cloud Poisoning:")
    for layer_name, pr in all_results["phase3_cloud_poisoning"].items():
        print(f"  {layer_name}: sep_raw={pr['separability_raw']:.3f}, "
              f"sep_l2={pr['separability_l2norm']:.3f}, "
              f"sep_quant={pr['separability_quant8bit']:.3f}, "
              f"sep_pca={pr['separability_pca32']:.3f}")
        print(f"    Poison fraction needed: {pr['poison_fraction_1000']:.4f} ({pr['poison_fraction_1000']*100:.1f}% of 1000 samples)")
        print(f"    Samples for 1-std shift: {pr['samples_needed_1std']:.0f}")

    print(f"\n{'='*70}")
    print("DONE — All results in outputs_clothing/forward_analysis/anchor_channel/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
