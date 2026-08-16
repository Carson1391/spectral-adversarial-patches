"""
Covert Channel Probe — Can 1/196 carry information through YOLO's feature pipeline?

Tests whether a modulated 1/196 spatial carrier can encode information that leaks
into detection outputs (confidence, class probabilities, bbox coordinates) without
destroying primary object detection.

Channel model:
  [Encoder: 1/196 * (1 + m * signal)] -> [YOLO network] -> [Output: dets + features] -> [Decoder]

Experiments:
  1. Feature map probing — per-channel SNR of 1/196 at every conv layer
  2. Amplitude modulation — inject AM-encoded signals, measure survival
  3. Output correlation — correlate injected value with confidence/class/bbox
  4. Mutual information — I(injected; output) for channel capacity in bits
  5. Multi-bit encoding — test 1, 2, 4, 8-bit patterns
  6. Backdoor test — can we shift confidence/bbox without killing detection?
  7. Cross-model — does the channel work on v8/11/26 too?
"""

import os, sys, json, csv, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")

from pytorchyolo.models import Darknet
from ultralytics import YOLO

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\covert_channel"
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


# ============================================================
# Utilities
# ============================================================

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
    """Broadband square wave carrier — frequency content spread across many harmonics."""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.sign(np.cos(2*np.pi*(kx/W*x + ky/H*y)))).astype(np.float32)

def add_pattern(arr, pat):
    out = arr.copy()
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] + pat, 0, 1)
    return out

def add_pattern_patch(arr, pat, mask):
    """Apply pattern only within the patch mask region."""
    out = arr.copy()
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] * (1 - mask) + (out[:,:,c] + pat) * mask, 0, 1)
    return out


# ============================================================
# YOLOv3 forward pass with full feature capture
# ============================================================

def forward_capture_v3(model, x):
    """Run forward pass and capture feature maps at every conv layer."""
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
    if output is None:
        return dets
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "bbox_w": float(row[2] - row[0]),
                "bbox_h": float(row[3] - row[1]),
                "bbox_cx": float((row[0] + row[2]) / 2),
                "bbox_cy": float((row[1] + row[3]) / 2),
            })
    return dets

def get_dets_ultralytics(model, pil_img, conf=0.1):
    results = model(pil_img, verbose=False)
    dets = []
    for r in results:
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            dets.append({
                "class_id": cls_id,
                "class_name": model.names.get(cls_id, f"c{cls_id}"),
                "confidence": float(r.boxes.conf[i].item()),
                "bbox": [float(r.boxes.xyxy[i][0]), float(r.boxes.xyxy[i][1]),
                         float(r.boxes.xyxy[i][2]), float(r.boxes.xyxy[i][3])],
                "bbox_w": float(r.boxes.xyxy[i][2] - r.boxes.xyxy[i][0]),
                "bbox_h": float(r.boxes.xyxy[i][3] - r.boxes.xyxy[i][1]),
                "bbox_cx": float((r.boxes.xyxy[i][0] + r.boxes.xyxy[i][2]) / 2),
                "bbox_cy": float((r.boxes.xyxy[i][1] + r.boxes.xyxy[i][3]) / 2),
            })
    return [d for d in dets if d["confidence"] >= conf]


# ============================================================
# EXPERIMENT 1: Per-channel feature map probing
# ============================================================

def experiment_1_feature_probing(v3_model, arr_base):
    """
    Inject 1/196 in different carriers, capture feature maps at every conv layer,
    measure per-channel statistics: how many channels carry the 1/196 signal?
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Per-Channel Feature Map Probing")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

    carriers = {
        "flat_inv196": np.full((H, W), 1.0/196.0, dtype=np.float32),
        "k196d_amp_inv196": make_sinusoid(H, W, 196, 196, 0, 1.0/196.0),
        "k200d_amp_inv196": make_sinusoid(H, W, 200, 200, 0, 1.0/196.0),
        "k167d_amp_inv196": make_sinusoid(H, W, 167, 167, 0, 1.0/196.0),
        "square196_amp_inv196": make_square_wave(H, W, 196, 196, 1.0/196.0),
        "square200_amp_inv196": make_square_wave(H, W, 200, 200, 1.0/196.0),
        "anticlose_inv196_k200d": np.full((H, W), 1.0/196.0, dtype=np.float32) + make_sinusoid(H, W, 200, 200, 0, 0.15),
        "anticlose_inv196_sq200": np.full((H, W), 1.0/196.0, dtype=np.float32) + make_square_wave(H, W, 200, 200, 0.15),
        "control_none": np.zeros((H, W), dtype=np.float32),
    }

    results = {}

    # First get baseline feature maps (no injection)
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)

    for name, carrier in carriers.items():
        print(f"\n  Carrier: {name}")
        arr_mod = add_pattern(arr_base, carrier)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            caps, _ = forward_capture_v3(v3_model, tensor_mod)

        layer_results = {}
        for layer_idx in LAYERS:
            if layer_idx not in caps or layer_idx not in base_caps:
                continue

            fm = caps[layer_idx][0]  # (C, H, W)
            base_fm = base_caps[layer_idx][0]
            delta = (fm - base_fm).cpu().numpy()  # (C, H, W) — what changed due to injection

            C = delta.shape[0]
            # Per-channel analysis
            channel_stats = []
            for ch in range(C):
                ch_data = delta[ch].flatten()
                if ch_data.std() < 1e-8:
                    channel_stats.append({"ch": ch, "mean_delta": 0, "std_delta": 0, "near_inv196": 0, "near_neg_inv196": 0, "max_delta": 0})
                    continue
                # Fraction of values near 1/196 (within 10% of 1/196)
                target = 1.0 / 196.0
                near = np.mean(np.abs(ch_data - target) < 0.1 * target) if target > 0 else 0
                # Also check fraction near -1/196 (negative phase)
                near_neg = np.mean(np.abs(ch_data + target) < 0.1 * target) if target > 0 else 0
                channel_stats.append({
                    "ch": ch,
                    "mean_delta": float(ch_data.mean()),
                    "std_delta": float(ch_data.std()),
                    "near_inv196": float(near),
                    "near_neg_inv196": float(near_neg),
                    "max_delta": float(np.abs(ch_data).max()),
                })

            # Aggregate stats
            means = [cs["mean_delta"] for cs in channel_stats]
            stds = [cs["std_delta"] for cs in channel_stats]
            nears = [cs["near_inv196"] for cs in channel_stats]
            max_deltas = [cs["max_delta"] for cs in channel_stats]

            # Count channels with significant signal
            sig_threshold = 0.001
            sig_channels = sum(1 for m in max_deltas if m > sig_threshold)

            layer_results[layer_idx] = {
                "num_channels": C,
                "mean_delta": float(np.mean(means)),
                "std_delta": float(np.mean(stds)),
                "max_delta": float(np.max(max_deltas)),
                "channels_with_signal": sig_channels,
                "signal_fraction": float(sig_channels / C) if C > 0 else 0,
                "avg_near_inv196": float(np.mean(nears)),
                "top5_channels": sorted(channel_stats, key=lambda x: -x["max_delta"])[:5],
            }

            if layer_idx in [0, 1, 62, 105]:
                print(f"    L{layer_idx:3d}: {C:4d} ch, sig={sig_channels:4d}/{C} ({sig_channels/C*100:.1f}%), "
                      f"max_delta={np.max(max_deltas):.6f}, mean_delta={np.mean(means):.6f}")

        results[name] = layer_results

    return results


# ============================================================
# EXPERIMENT 2: Amplitude modulation survival
# ============================================================

def experiment_2_am_modulation(v3_model, arr_base):
    """
    Inject AM signal: carrier * (1 + m * modulating_signal)
    Test if modulating signal survives to output feature maps.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Amplitude Modulation Survival")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    LAYERS = [0, 1, 12, 62, 105]

    # Carrier: k=200 square wave (broadband, survives batch norm)
    # Modulating signal: slow sinusoid at k=5 (easily distinguishable from carrier)
    carrier_freq = 200
    mod_freq = 5
    base_amp = 1.0 / 196.0  # 1/196 as the base amplitude

    # Test different modulation depths
    mod_depths = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = {}

    # Get baseline (no injection)
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)

    for m in mod_depths:
        print(f"\n  Modulation depth m={m:.2f}")
        # AM signal: base_amp * (1 + m * cos(2*pi*mod_freq*x/W))
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        mod_signal = np.cos(2 * np.pi * mod_freq * x / W).astype(np.float32)
        # Square wave carrier * AM envelope
        carrier = make_square_wave(H, W, carrier_freq, carrier_freq, 1.0)
        am_signal = (base_amp * (1 + m * mod_signal) * carrier).astype(np.float32)

        arr_mod = add_pattern(arr_base, am_signal)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            caps, _ = forward_capture_v3(v3_model, tensor_mod)

        layer_results = {}
        for layer_idx in LAYERS:
            if layer_idx not in caps:
                continue

            fm = caps[layer_idx][0].cpu().numpy()  # (C, H, W)
            base_fm = base_caps[layer_idx][0].cpu().numpy()
            delta = fm - base_fm  # (C, H, W)

            C, fh, fw = delta.shape

            # Check if the modulating frequency (k=5) appears in the feature map delta
            # Compute 2D FFT of the delta and look for energy at k=5
            # Average across channels first
            avg_delta = delta.mean(axis=0)  # (H, W)

            # 1D FFT along x axis (where modulation was applied), averaged over y
            fft_x = np.abs(np.fft.fft(avg_delta.mean(axis=0)))[:fw//2]
            # Energy at mod_freq (k=5)
            mod_energy = float(fft_x[mod_freq] if mod_freq < len(fft_x) else 0)
            total_energy = float(fft_x.sum() + 1e-10)
            mod_energy_ratio = mod_energy / total_energy

            # Also check carrier frequency (k=200) energy
            carrier_energy = float(fft_x[min(carrier_freq, len(fft_x)-1)])
            carrier_energy_ratio = carrier_energy / total_energy

            # Per-channel: find channels with strongest mod_freq signal
            channel_mod_energies = []
            for ch in range(min(C, 50)):  # Sample first 50 channels
                ch_fft = np.abs(np.fft.fft(delta[ch].mean(axis=0)))[:fw//2]
                ch_mod_e = float(ch_fft[mod_freq] if mod_freq < len(ch_fft) else 0)
                channel_mod_energies.append(ch_mod_e)

            layer_results[layer_idx] = {
                "mod_freq": mod_freq,
                "mod_depth": m,
                "mod_energy": mod_energy,
                "mod_energy_ratio": float(mod_energy_ratio),
                "carrier_energy": carrier_energy,
                "carrier_energy_ratio": float(carrier_energy_ratio),
                "top_channel_mod_energy": max(channel_mod_energies) if channel_mod_energies else 0,
                "mean_channel_mod_energy": float(np.mean(channel_mod_energies)) if channel_mod_energies else 0,
                "feature_shape": [C, fh, fw],
            }

            print(f"    L{layer_idx:3d}: mod_energy={mod_energy:.4f} ({mod_energy_ratio:.4f} of total), "
                  f"carrier_energy={carrier_energy:.4f}, top_ch={max(channel_mod_energies):.4f}")

        results[f"m={m:.2f}"] = layer_results

    return results


# ============================================================
# EXPERIMENT 3: Output correlation — does injected value shift detections?
# ============================================================

def experiment_3_output_correlation(v3_model, arr_base):
    """
    Inject N different encoded values and measure correlation between
    injected value and detection outputs (confidence, bbox, class).
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Output Correlation — Does Injected Value Shift Detections?")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape

    # Encode different values using 1/196 * scale factor
    # scale from 0.5 to 5.0 (10 steps) — this changes the amplitude of the injected signal
    scales = np.linspace(0.5, 5.0, 20)
    carrier_freq = 200

    results = []
    baseline_dets = None

    for scale in scales:
        # Inject 1/196 * scale, modulated by square wave carrier
        injected_val = (1.0 / 196.0) * scale
        carrier = make_square_wave(H, W, carrier_freq, carrier_freq, injected_val)

        arr_mod = add_pattern(arr_base, carrier)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        dets = get_dets_v3(v3_model, tensor_mod, conf=0.01)  # Low threshold to catch weak shifts

        if baseline_dets is None:
            # Get baseline with zero injection
            tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
            baseline_dets = get_dets_v3(v3_model, tensor_base, conf=0.01)

        # Summarize detections
        person_dets = [d for d in dets if d["class_name"] == "person"]
        non_person = [d for d in dets if d["class_name"] != "person"]

        result = {
            "scale": float(scale),
            "injected_val": float(injected_val),
            "total_dets": len(dets),
            "person_count": len(person_dets),
            "non_person_count": len(non_person),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
        }

        if person_dets:
            confs = [d["confidence"] for d in person_dets]
            result["person_mean_conf"] = float(np.mean(confs))
            result["person_max_conf"] = float(np.max(confs))
            result["person_min_conf"] = float(np.min(confs))
            result["person_std_conf"] = float(np.std(confs))
            # Bbox stats
            bws = [d["bbox_w"] for d in person_dets]
            bhs = [d["bbox_h"] for d in person_dets]
            cxs = [d["bbox_cx"] for d in person_dets]
            cys = [d["bbox_cy"] for d in person_dets]
            result["person_mean_bbox_w"] = float(np.mean(bws))
            result["person_mean_bbox_h"] = float(np.mean(bhs))
            result["person_mean_cx"] = float(np.mean(cxs))
            result["person_mean_cy"] = float(np.mean(cys))
            result["person_std_cx"] = float(np.std(cxs))
            result["person_std_cy"] = float(np.std(cys))
        else:
            result["person_mean_conf"] = 0
            result["person_max_conf"] = 0
            result["person_min_conf"] = 0
            result["person_std_conf"] = 0
            result["person_mean_bbox_w"] = 0
            result["person_mean_bbox_h"] = 0
            result["person_mean_cx"] = 0
            result["person_mean_cy"] = 0
            result["person_std_cx"] = 0
            result["person_std_cy"] = 0

        results.append(result)

        tag = ""
        if result["person_count"] == 0: tag = " [PERSON_SUPPRESSED]"
        elif non_person: tag = f" [HALLUC:{result['non_person_classes']}]"
        print(f"  scale={scale:.2f} (val={injected_val:.6f}): {result['total_dets']:2d} dets, "
              f"person={result['person_count']}, conf={result['person_mean_conf']:.3f}{tag}")

    # Compute correlations
    scales_arr = np.array([r["scale"] for r in results])
    vals_arr = np.array([r["injected_val"] for r in results])
    confs_arr = np.array([r["person_mean_conf"] for r in results])
    counts_arr = np.array([r["person_count"] for r in results])
    bbox_w_arr = np.array([r["person_mean_bbox_w"] for r in results])
    bbox_h_arr = np.array([r["person_mean_bbox_h"] for r in results])
    cx_arr = np.array([r["person_mean_cx"] for r in results])
    cy_arr = np.array([r["person_mean_cy"] for r in results])

    correlations = {}
    for name, y in [("person_conf", confs_arr), ("person_count", counts_arr),
                     ("bbox_w", bbox_w_arr), ("bbox_h", bbox_h_arr),
                     ("bbox_cx", cx_arr), ("bbox_cy", cy_arr)]:
        if y.std() > 1e-8 and vals_arr.std() > 1e-8:
            r_pearson, p_pearson = pearsonr(vals_arr, y)
            correlations[name] = {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "significant": p_pearson < 0.05,
            }
        else:
            correlations[name] = {"pearson_r": 0, "pearson_p": 1, "significant": False}

    print(f"\n  CORRELATIONS (injected_val vs output):")
    for name, corr in correlations.items():
        sig = "***" if corr["significant"] else ""
        print(f"    {name:15s}: r={corr['pearson_r']:+.4f}, p={corr['pearson_p']:.4f} {sig}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0,0].plot(vals_arr, confs_arr, 'b.-'); axes[0,0].set_title("Person Mean Confidence vs Injected Val")
    axes[0,0].set_xlabel("Injected Value (1/196 * scale)"); axes[0,0].set_ylabel("Mean Confidence")
    axes[0,1].plot(vals_arr, counts_arr, 'r.-'); axes[0,1].set_title("Person Count vs Injected Val")
    axes[0,1].set_xlabel("Injected Value"); axes[0,1].set_ylabel("Person Count")
    axes[0,2].plot(vals_arr, bbox_w_arr, 'g.-'); axes[0,2].set_title("BBox Width vs Injected Val")
    axes[0,2].set_xlabel("Injected Value"); axes[0,2].set_ylabel("Mean BBox Width")
    axes[1,0].plot(vals_arr, bbox_h_arr, 'm.-'); axes[1,0].set_title("BBox Height vs Injected Val")
    axes[1,0].set_xlabel("Injected Value"); axes[1,0].set_ylabel("Mean BBox Height")
    axes[1,1].plot(vals_arr, cx_arr, 'c.-'); axes[1,1].set_title("BBox Center X vs Injected Val")
    axes[1,1].set_xlabel("Injected Value"); axes[1,1].set_ylabel("Mean Center X")
    axes[1,2].plot(vals_arr, cy_arr, 'y.-'); axes[1,2].set_title("BBox Center Y vs Injected Val")
    axes[1,2].set_xlabel("Injected Value"); axes[1,2].set_ylabel("Mean Center Y")
    plt.suptitle("Experiment 3: Output Correlation with 1/196 Injection", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp3_output_correlation.png", dpi=150)
    plt.close()

    return {"trials": results, "correlations": correlations, "baseline_person_count": len([d for d in baseline_dets if d["class_name"] == "person"])}


# ============================================================
# EXPERIMENT 4: Mutual information — channel capacity
# ============================================================

def experiment_4_mutual_information(v3_model, arr_base):
    """
    Calculate mutual information I(injected_signal; output_features) to
    quantify the covert channel capacity in bits.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 4: Mutual Information — Channel Capacity")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    carrier_freq = 200

    # Encode N distinct symbols using different amplitudes
    # Use 16 levels (4-bit encoding) to test channel capacity
    N_LEVELS = 16
    levels = np.linspace(0.5, 5.0, N_LEVELS)
    N_TRIALS_PER_LEVEL = 5  # Multiple trials per level for MI estimation

    all_inputs = []
    all_outputs = []

    for level_idx, scale in enumerate(levels):
        injected_val = (1.0 / 196.0) * scale
        for trial in range(N_TRIALS_PER_LEVEL):
            # Add small random noise to make each trial unique
            noise = np.random.RandomState(trial * 100 + level_idx).randn(H, W) * 0.001
            carrier = make_square_wave(H, W, carrier_freq, carrier_freq, injected_val) + noise.astype(np.float32)

            arr_mod = add_pattern(arr_base, carrier)
            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

            dets = get_dets_v3(v3_model, tensor_mod, conf=0.01)
            person_dets = [d for d in dets if d["class_name"] == "person"]

            # Output features: person count, mean conf, mean bbox
            if person_dets:
                output = {
                    "person_count": len(person_dets),
                    "mean_conf": float(np.mean([d["confidence"] for d in person_dets])),
                    "mean_bbox_w": float(np.mean([d["bbox_w"] for d in person_dets])),
                    "mean_bbox_h": float(np.mean([d["bbox_h"] for d in person_dets])),
                    "mean_cx": float(np.mean([d["bbox_cx"] for d in person_dets])),
                    "mean_cy": float(np.mean([d["bbox_cy"] for d in person_dets])),
                }
            else:
                output = {
                    "person_count": 0, "mean_conf": 0, "mean_bbox_w": 0,
                    "mean_bbox_h": 0, "mean_cx": 0, "mean_cy": 0,
                }

            all_inputs.append(level_idx)
            all_outputs.append(output)

    inputs = np.array(all_inputs)

    # Calculate MI for each output dimension
    mi_results = {}
    for output_name in ["person_count", "mean_conf", "mean_bbox_w", "mean_bbox_h", "mean_cx", "mean_cy"]:
        outputs = np.array([o[output_name] for o in all_outputs])

        # Discretize outputs into bins for MI calculation
        n_bins = min(10, len(set(outputs)))
        if n_bins < 2:
            mi_results[output_name] = {"mi_bits": 0, "n_bins": 0}
            continue

        # Use quantile-based binning
        bins = np.quantile(outputs, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 1e-8
        bins[-1] += 1e-8
        discretized = np.digitize(outputs, bins) - 1
        discretized = np.clip(discretized, 0, n_bins - 1)

        mi = mutual_info_score(inputs, discretized) / math.log(2)  # Convert to bits

        mi_results[output_name] = {
            "mi_bits": float(mi),
            "n_bins": n_bins,
            "max_possible_bits": float(math.log2(N_LEVELS)),
            "efficiency": float(mi / math.log2(N_LEVELS)) if N_LEVELS > 1 else 0,
        }

        print(f"  I(input; {output_name:15s}) = {mi:.4f} bits "
              f"(max={math.log2(N_LEVELS):.4f}, efficiency={mi/math.log2(N_LEVELS)*100:.1f}%)")

    # Total channel capacity (sum of MI across independent outputs)
    total_mi = sum(r["mi_bits"] for r in mi_results.values())
    print(f"\n  Total channel capacity (sum of MI): {total_mi:.4f} bits per transmission")
    print(f"  Theoretical max: {math.log2(N_LEVELS):.4f} bits (log2({N_LEVELS}) levels)")

    return {"mi_per_output": mi_results, "total_mi_bits": total_mi, "n_levels": N_LEVELS,
            "n_trials": len(all_inputs)}


# ============================================================
# EXPERIMENT 5: Multi-bit encoding
# ============================================================

def experiment_5_multibit_encoding(v3_model, arr_base):
    """
    Test if we can encode and decode multiple bits through the channel.
    Encode known bit patterns and check if they're distinguishable in output.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 5: Multi-Bit Encoding")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    carrier_freq = 200

    # Test 1-bit, 2-bit, 4-bit, 8-bit encoding
    bit_widths = [1, 2, 4, 8]
    results = {}

    for n_bits in bit_widths:
        n_symbols = 2 ** n_bits
        print(f"\n  Testing {n_bits}-bit encoding ({n_symbols} symbols)...")

        # Each symbol maps to a different amplitude
        # Use the full range: 0.5 to 5.0 scale of 1/196
        scales = np.linspace(0.5, 5.0, n_symbols)

        symbol_outputs = {}  # symbol -> list of output features

        for sym in range(n_symbols):
            scale = scales[sym]
            injected_val = (1.0 / 196.0) * scale
            carrier = make_square_wave(H, W, carrier_freq, carrier_freq, injected_val)

            arr_mod = add_pattern(arr_base, carrier)
            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

            dets = get_dets_v3(v3_model, tensor_mod, conf=0.01)
            person_dets = [d for d in dets if d["class_name"] == "person"]

            if person_dets:
                output_vec = [
                    len(person_dets),
                    float(np.mean([d["confidence"] for d in person_dets])),
                    float(np.std([d["confidence"] for d in person_dets])),
                    float(np.mean([d["bbox_w"] for d in person_dets])),
                    float(np.mean([d["bbox_h"] for d in person_dets])),
                    float(np.mean([d["bbox_cx"] for d in person_dets])),
                    float(np.mean([d["bbox_cy"] for d in person_dets])),
                ]
            else:
                output_vec = [0, 0, 0, 0, 0, 0, 0]

            symbol_outputs[sym] = output_vec

        # Check separability: can we distinguish symbols from output?
        output_matrix = np.array([symbol_outputs[s] for s in range(n_symbols)])  # (n_symbols, 7)

        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        if n_symbols > 1:
            dists = pdist(output_matrix, metric='euclidean')
            min_dist = float(dists.min())
            max_dist = float(dists.max())
            mean_dist = float(dists.mean())
        else:
            min_dist = max_dist = mean_dist = 0

        # Separability score: min_dist / mean_dist (higher = more separable)
        separability = min_dist / (mean_dist + 1e-10)

        # Check if nearest neighbor is correct (trivial for ordered encoding)
        # For amplitude encoding, symbols should be ordered by output value
        # Check monotonicity of primary output (person_count or mean_conf)
        primary_output = output_matrix[:, 1]  # mean_conf
        if primary_output.std() > 1e-8:
            # Check if output is monotonic with symbol
            from scipy.stats import spearmanr
            rho, p = spearmanr(range(n_symbols), primary_output)
            monotonic = abs(rho) > 0.9 and p < 0.05
        else:
            rho, p, monotonic = 0, 1, False

        results[f"{n_bits}bit"] = {
            "n_symbols": n_symbols,
            "min_pairwise_dist": min_dist,
            "max_pairwise_dist": max_dist,
            "mean_pairwise_dist": mean_dist,
            "separability": float(separability),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "monotonic": monotonic,
            "output_matrix": output_matrix.tolist(),
        }

        print(f"    Separability: {separability:.4f} (min_dist={min_dist:.4f}, mean_dist={mean_dist:.4f})")
        print(f"    Monotonic: {monotonic} (rho={rho:.3f}, p={p:.4f})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, n_bits in enumerate(bit_widths):
        ax = axes[i//2, i%2]
        key = f"{n_bits}bit"
        mat = np.array(results[key]["output_matrix"])
        ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_title(f"{n_bits}-bit ({results[key]['n_symbols']} symbols)\n"
                     f"sep={results[key]['separability']:.3f}, mono={results[key]['monotonic']}")
        ax.set_xlabel("Output Feature Index")
        ax.set_ylabel("Symbol Index")
    plt.suptitle("Experiment 5: Multi-Bit Encoding Separability", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp5_multibit.png", dpi=150)
    plt.close()

    return results


# ============================================================
# EXPERIMENT 6: Backdoor test — subtle confidence/bbox shifts
# ============================================================

def experiment_6_backdoor(v3_model, arr_base):
    """
    Test if a subtle 1/196 injection can shift confidence scores or bbox
    coordinates WITHOUT destroying the primary detection. This is the
    "backdoor" scenario: detection still works but outputs are biased.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 6: Backdoor Test — Subtle Output Manipulation")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    carrier_freq = 200

    # Get baseline detections
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    baseline_dets = get_dets_v3(v3_model, tensor_base, conf=0.01)
    baseline_person = [d for d in baseline_dets if d["class_name"] == "person"]
    print(f"  Baseline: {len(baseline_person)} person detections")

    # Test subtle injections — small amplitudes that don't kill detection
    amplitudes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    results = []

    for amp in amplitudes:
        # Use 1/196 as the amplitude, modulated by square wave carrier
        # This is a very subtle injection
        carrier = make_square_wave(H, W, carrier_freq, carrier_freq, amp)

        arr_mod = add_pattern(arr_base, carrier)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        dets = get_dets_v3(v3_model, tensor_mod, conf=0.01)
        person_dets = [d for d in dets if d["class_name"] == "person"]

        # Match to baseline by IoU to track per-detection shifts
        shifts = []
        for bp in baseline_person:
            best_match = None
            best_iou = 0
            for pd in person_dets:
                # Simple IoU
                ix1 = max(bp["bbox"][0], pd["bbox"][0])
                iy1 = max(bp["bbox"][1], pd["bbox"][1])
                ix2 = min(bp["bbox"][2], pd["bbox"][2])
                iy2 = min(bp["bbox"][3], pd["bbox"][3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                area_b = bp["bbox_w"] * bp["bbox_h"]
                area_p = pd["bbox_w"] * pd["bbox_h"]
                iou = inter / (area_b + area_p - inter + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_match = pd

            if best_match and best_iou > 0.3:
                shift = {
                    "conf_shift": best_match["confidence"] - bp["confidence"],
                    "cx_shift": best_match["bbox_cx"] - bp["bbox_cx"],
                    "cy_shift": best_match["bbox_cy"] - bp["bbox_cy"],
                    "w_shift": best_match["bbox_w"] - bp["bbox_w"],
                    "h_shift": best_match["bbox_h"] - bp["bbox_h"],
                    "iou": best_iou,
                }
                shifts.append(shift)

        # Aggregate shifts
        if shifts:
            mean_conf_shift = float(np.mean([s["conf_shift"] for s in shifts]))
            mean_cx_shift = float(np.mean([s["cx_shift"] for s in shifts]))
            mean_cy_shift = float(np.mean([s["cy_shift"] for s in shifts]))
            mean_w_shift = float(np.mean([s["w_shift"] for s in shifts]))
            mean_h_shift = float(np.mean([s["h_shift"] for s in shifts]))
            max_conf_shift = float(np.max([abs(s["conf_shift"]) for s in shifts]))
            max_cx_shift = float(np.max([abs(s["cx_shift"]) for s in shifts]))
            max_cy_shift = float(np.max([abs(s["cy_shift"]) for s in shifts]))
        else:
            mean_conf_shift = mean_cx_shift = mean_cy_shift = 0
            mean_w_shift = mean_h_shift = 0
            max_conf_shift = max_cx_shift = max_cy_shift = 0

        result = {
            "amplitude": float(amp),
            "person_count": len(person_dets),
            "baseline_count": len(baseline_person),
            "matched_count": len(shifts),
            "detection_preserved": len(person_dets) > 0,
            "mean_conf_shift": mean_conf_shift,
            "mean_cx_shift": mean_cx_shift,
            "mean_cy_shift": mean_cy_shift,
            "mean_w_shift": mean_w_shift,
            "mean_h_shift": mean_h_shift,
            "max_conf_shift": max_conf_shift,
            "max_cx_shift": max_cx_shift,
            "max_cy_shift": max_cy_shift,
        }
        results.append(result)

        print(f"  amp={amp:.3f}: person={len(person_dets):2d}/{len(baseline_person)}, "
              f"conf_shift={mean_conf_shift:+.4f}, cx_shift={mean_cx_shift:+.2f}, "
              f"cy_shift={mean_cy_shift:+.2f}, preserved={result['detection_preserved']}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    amps = [r["amplitude"] for r in results]
    axes[0,0].plot(amps, [r["mean_conf_shift"] for r in results], 'b.-')
    axes[0,0].set_title("Confidence Shift vs Injection Amplitude")
    axes[0,0].set_xlabel("Amplitude"); axes[0,0].set_ylabel("Mean Confidence Shift")
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0,1].plot(amps, [r["person_count"] for r in results], 'r.-')
    axes[0,1].set_title("Person Count vs Injection Amplitude")
    axes[0,1].set_xlabel("Amplitude"); axes[0,1].set_ylabel("Person Count")
    axes[0,1].axhline(y=len(baseline_person), color='k', linestyle='--', alpha=0.3, label="baseline")
    axes[0,1].legend()
    axes[1,0].plot(amps, [r["mean_cx_shift"] for r in results], 'g.-', label="cx")
    axes[1,0].plot(amps, [r["mean_cy_shift"] for r in results], 'm.-', label="cy")
    axes[1,0].set_title("BBox Center Shift vs Amplitude")
    axes[1,0].set_xlabel("Amplitude"); axes[1,0].set_ylabel("Mean Center Shift (px)")
    axes[1,0].legend()
    axes[1,1].plot(amps, [r["mean_w_shift"] for r in results], 'c.-', label="width")
    axes[1,1].plot(amps, [r["mean_h_shift"] for r in results], 'y.-', label="height")
    axes[1,1].set_title("BBox Size Shift vs Amplitude")
    axes[1,1].set_xlabel("Amplitude"); axes[1,1].set_ylabel("Mean Size Shift (px)")
    axes[1,1].legend()
    plt.suptitle("Experiment 6: Backdoor — Subtle Output Manipulation", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp6_backdoor.png", dpi=150)
    plt.close()

    return {"trials": results, "baseline_person_count": len(baseline_person)}


# ============================================================
# EXPERIMENT 7: Cross-model channel test
# ============================================================

def experiment_7_cross_model(arr_base):
    """
    Test if the covert channel works on YOLOv8, YOLO11, YOLO26.
    Quick test: inject 5 different amplitudes, measure output correlation.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 7: Cross-Model Channel Test")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    carrier_freq = 200
    amplitudes = np.linspace(0.005, 0.2, 10)

    model_configs = [
        ("YOLOv8", r"C:\Users\carso\Desktop\YODO\YOLOv8\yolov8l.pt"),
        ("YOLO11", r"C:\Users\carso\Desktop\YODO\YOLO11\yolo11l.pt"),
        ("YOLO26", r"C:\Users\carso\Desktop\YODO\YOLO26\yolo26l.pt"),
    ]

    pil_base = Image.fromarray((arr_base * 255).astype(np.uint8))
    all_results = {}

    for model_name, model_path in model_configs:
        print(f"\n  {model_name}...")
        model = YOLO(model_path)
        model.to(DEVICE)

        # Baseline
        base_dets = get_dets_ultralytics(model, pil_base, conf=0.01)
        base_person = [d for d in base_dets if d["class_name"] == "person"]
        print(f"    Baseline: {len(base_person)} person dets")

        trials = []
        for amp in amplitudes:
            carrier = make_square_wave(H, W, carrier_freq, carrier_freq, amp)
            arr_mod = add_pattern(arr_base, carrier)
            pil_mod = Image.fromarray((arr_mod * 255).astype(np.uint8))

            dets = get_dets_ultralytics(model, pil_mod, conf=0.01)
            person_dets = [d for d in dets if d["class_name"] == "person"]
            non_person = [d for d in dets if d["class_name"] != "person"]

            trial = {
                "amplitude": float(amp),
                "person_count": len(person_dets),
                "non_person_count": len(non_person),
                "non_person_classes": list(set(d["class_name"] for d in non_person)),
            }
            if person_dets:
                trial["mean_conf"] = float(np.mean([d["confidence"] for d in person_dets]))
                trial["mean_cx"] = float(np.mean([d["bbox_cx"] for d in person_dets]))
                trial["mean_cy"] = float(np.mean([d["bbox_cy"] for d in person_dets]))
            else:
                trial["mean_conf"] = 0
                trial["mean_cx"] = 0
                trial["mean_cy"] = 0

            trials.append(trial)
            tag = ""
            if trial["person_count"] == 0: tag = " [SUPPRESSED]"
            elif non_person: tag = f" [HALLUC:{trial['non_person_classes']}]"
            print(f"    amp={amp:.3f}: person={trial['person_count']}, conf={trial['mean_conf']:.3f}{tag}")

        # Correlation
        amps_arr = np.array([t["amplitude"] for t in trials])
        confs_arr = np.array([t["mean_conf"] for t in trials])
        counts_arr = np.array([t["person_count"] for t in trials])

        corrs = {}
        for name, y in [("conf", confs_arr), ("count", counts_arr)]:
            if y.std() > 1e-8:
                r, p = pearsonr(amps_arr, y)
                corrs[name] = {"r": float(r), "p": float(p), "sig": p < 0.05}
            else:
                corrs[name] = {"r": 0, "p": 1, "sig": False}

        all_results[model_name] = {"trials": trials, "correlations": corrs,
                                    "baseline_person": len(base_person)}

        for name, c in corrs.items():
            print(f"    corr({name}, amp): r={c['r']:+.4f}, p={c['p']:.4f} {'***' if c['sig'] else ''}")

        del model
        torch.cuda.empty_cache()

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("COVERT CHANNEL PROBE — 1/196 as Information Carrier Through YOLO")
    print("="*70)
    print(f"Device: {DEVICE}")

    arr_base = load_image(IMG_WITH, IMG_SIZE)
    H, W, _ = arr_base.shape

    # Load YOLOv3
    print("\nLoading YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters():
        p.requires_grad_(False)

    all_results = {}

    # Run all experiments
    all_results["exp1_feature_probing"] = experiment_1_feature_probing(v3_model, arr_base)
    all_results["exp2_am_modulation"] = experiment_2_am_modulation(v3_model, arr_base)
    all_results["exp3_output_correlation"] = experiment_3_output_correlation(v3_model, arr_base)
    all_results["exp4_mutual_information"] = experiment_4_mutual_information(v3_model, arr_base)
    all_results["exp5_multibit_encoding"] = experiment_5_multibit_encoding(v3_model, arr_base)
    all_results["exp6_backdoor"] = experiment_6_backdoor(v3_model, arr_base)

    del v3_model
    torch.cuda.empty_cache()

    # Cross-model
    all_results["exp7_cross_model"] = experiment_7_cross_model(arr_base)

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\n{'='*70}")
    print("Saving results...")

    json_path = f"{OUTPUT_DIR}/covert_channel.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # CSV summary
    csv_path = f"{OUTPUT_DIR}/covert_channel.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "key", "metric", "value"])
        # Exp 3 correlations
        for name, corr in all_results["exp3_output_correlation"]["correlations"].items():
            writer.writerow(["exp3_correlation", name, "pearson_r", corr["pearson_r"]])
            writer.writerow(["exp3_correlation", name, "pearson_p", corr["pearson_p"]])
        # Exp 4 MI
        for name, mi in all_results["exp4_mutual_information"]["mi_per_output"].items():
            writer.writerow(["exp4_mi", name, "mi_bits", mi["mi_bits"]])
        writer.writerow(["exp4_mi", "TOTAL", "mi_bits", all_results["exp4_mutual_information"]["total_mi_bits"]])
        # Exp 5 separability
        for key, res in all_results["exp5_multibit_encoding"].items():
            writer.writerow(["exp5_multibit", key, "separability", res["separability"]])
            writer.writerow(["exp5_multibit", key, "monotonic", res["monotonic"]])
        # Exp 6 backdoor
        for trial in all_results["exp6_backdoor"]["trials"]:
            writer.writerow(["exp6_backdoor", f"amp={trial['amplitude']}", "conf_shift", trial["mean_conf_shift"]])
            writer.writerow(["exp6_backdoor", f"amp={trial['amplitude']}", "cx_shift", trial["mean_cx_shift"]])
            writer.writerow(["exp6_backdoor", f"amp={trial['amplitude']}", "cy_shift", trial["mean_cy_shift"]])
        # Exp 7 cross-model
        for model_name, res in all_results["exp7_cross_model"].items():
            for cname, corr in res["correlations"].items():
                writer.writerow(["exp7_cross_model", f"{model_name}_{cname}", "pearson_r", corr["r"]])
    print(f"Saved CSV: {csv_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\n1. FEATURE PROBING (1/196 survival per layer):")
    for carrier, layers in all_results["exp1_feature_probing"].items():
        if carrier == "control_none":
            continue
        l105 = layers.get(105, {})
        l1 = layers.get(1, {})
        print(f"   {carrier:30s}: L1 sig={l1.get('channels_with_signal',0)}/{l1.get('num_channels',0)}, "
              f"L105 sig={l105.get('channels_with_signal',0)}/{l105.get('num_channels',0)}")

    print("\n2. AM MODULATION (k=5 mod signal survival):")
    for key, layers in all_results["exp2_am_modulation"].items():
        l105 = layers.get(105, {})
        print(f"   {key}: L105 mod_energy_ratio={l105.get('mod_energy_ratio',0):.6f}")

    print("\n3. OUTPUT CORRELATION (injected_val vs detection output):")
    for name, corr in all_results["exp3_output_correlation"]["correlations"].items():
        sig = "SIGNIFICANT" if corr["significant"] else "ns"
        print(f"   {name:15s}: r={corr['pearson_r']:+.4f} ({sig})")

    print("\n4. MUTUAL INFORMATION (channel capacity):")
    for name, mi in all_results["exp4_mutual_information"]["mi_per_output"].items():
        print(f"   I(input; {name:15s}) = {mi['mi_bits']:.4f} bits")
    print(f"   TOTAL: {all_results['exp4_mutual_information']['total_mi_bits']:.4f} bits/transmission")

    print("\n5. MULTI-BIT ENCODING (separability):")
    for key, res in all_results["exp5_multibit_encoding"].items():
        print(f"   {key}: separability={res['separability']:.4f}, monotonic={res['monotonic']}")

    print("\n6. BACKDOOR (subtle shifts without killing detection):")
    for trial in all_results["exp6_backdoor"]["trials"]:
        if trial["detection_preserved"] and abs(trial["mean_conf_shift"]) > 0.01:
            print(f"   amp={trial['amplitude']:.3f}: conf_shift={trial['mean_conf_shift']:+.4f}, "
                  f"cx_shift={trial['mean_cx_shift']:+.2f}px, cy_shift={trial['mean_cy_shift']:+.2f}px, "
                  f"DETECTION PRESERVED")

    print("\n7. CROSS-MODEL:")
    for model_name, res in all_results["exp7_cross_model"].items():
        for cname, corr in res["correlations"].items():
            sig = "SIGNIFICANT" if corr["sig"] else "ns"
            print(f"   {model_name} corr({cname}, amp): r={corr['r']:+.4f} ({sig})")

    print(f"\n{'='*70}")
    print("DONE — All results in outputs_clothing/forward_analysis/covert_channel/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
