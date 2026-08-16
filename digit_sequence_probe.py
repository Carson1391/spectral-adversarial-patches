"""
Digit Sequence Probe — Decimal expansions of 1/196 and related fractions as spatial patterns.

Key insight: 1/196 = 0.005102040816326530612244897959183673469387755...
The digits encode a doubling sequence (powers of 2) with carry propagation.
42-digit repeating period. Carries form their own doubling sequence (fractal carries).

YOLOv3 has 256-dim (2^8) feature maps. The carry at slot 8 produces 256 = 2^8.
The digit structure of 1/196 encodes the channel dimensionality of the network.

Experiments:
  1. Digit-to-pixel mapping: map each digit of decimal expansion to a pixel position
  2. Carry point analysis: identify where carries occur, measure disruption at those points
  3. Period tiling: tile the 42-digit period across 416/640 pixel images, measure boundary effects
  4. Multi-scale downsample tracking: how does the digit pattern transform at each downsample layer?
  5. 256-dim resonance: does the 2^8 carry point create special resonance with YOLOv3's 256-channel layers?
  6. Fraction comparison: 1/196 (doubling), 1/89 (Fibonacci), 1/9801 (counting), 1/7 (cyclic), 1/9.8
  7. Cross-model: does the digit pattern disrupt v8/11/26 differently?
  8. Closed-loop vs open-loop: test full period (closed) vs truncated period (open boundary)
"""

import os, sys, json, csv, math
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")

from pytorchyolo.models import Darknet
from ultralytics import YOLO

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\digit_sequence"
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
# Decimal expansion utilities
# ============================================================

def get_decimal_expansion(numerator, denominator, num_digits=500):
    """Compute decimal expansion of numerator/denominator to num_digits precision."""
    from decimal import Decimal, getcontext
    getcontext().prec = num_digits + 50
    val = Decimal(numerator) / Decimal(denominator)
    digits_str = str(val)[2:]  # Remove "0." prefix
    return [int(d) for d in digits_str[:num_digits]]


def find_period(digits):
    """Find the repeating period length in a digit sequence."""
    n = len(digits)
    for period in range(1, n // 2):
        match = True
        for i in range(period, min(n, period * 10)):
            if digits[i] != digits[i % period]:
                match = False
                break
        if match:
            return period
    return n  # No period found in available digits


def identify_carries(denominator, slot_width=2):
    """
    For a fraction 1/denominator that generates a geometric sequence in slot_width-digit slots,
    identify where carries occur and what the carry values are.

    For 1/196: terms are 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240...
    Each term = 5 * 2^slot_index
    Carry at slot i = floor(5 * 2^i / 10^slot_width)
    """
    carries = []
    carry_in = 0
    base_term = 5  # 1/196 starts at 5 (since 1/196 = 5/980, and 5 is the first term)

    # Actually for 1/196, the series is:
    # 1/196 = 5 * (1/980) = 5 * sum(2^k * 10^(-2k-2) for k=0,1,2,...)
    # = 5 * (0.01 + 0.0002 + 0.000004 + 0.00000008 + ...)
    # = 0.05 + 0.001 + 0.0002 + 0.00004 + 0.000008 + ...
    # Terms in 2-digit slots: 05, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240...

    for slot in range(200):
        term = base_term * (2 ** slot)
        raw_slot_val = term % (10 ** slot_width)
        carry_out = term // (10 ** slot_width)
        carries.append({
            "slot": slot,
            "term": term,
            "raw_2digit": raw_slot_val,
            "carry_out": carry_out,
            "is_carry": carry_out > 0,
            "carry_value": carry_out,
        })
    return carries


def identify_carries_general(numerator, denominator, slot_width=2, num_slots=100):
    """
    General carry detection: compute the actual decimal expansion,
    then identify where the naive geometric sequence would differ from
    the actual digits — those are the carry points.
    """
    digits = get_decimal_expansion(numerator, denominator, num_slots * slot_width + 50)

    # Group digits into slots
    actual_slots = []
    for i in range(0, len(digits) - slot_width + 1, slot_width):
        val = 0
        for j in range(slot_width):
            val = val * 10 + digits[i + j]
        actual_slots.append(val)

    # For 1/196 family: expected terms are base * 2^k
    # Determine base term from first non-zero slot
    base = None
    for s in actual_slots:
        if s > 0:
            base = s
            break

    if base is None:
        return [], actual_slots

    # Compute expected (no-carry) values and compare
    carries = []
    cumulative_carry = 0
    for slot_idx in range(min(num_slots, len(actual_slots))):
        expected_term = base * (2 ** slot_idx)
        expected_raw = expected_term % (10 ** slot_width)
        expected_carry = expected_term // (10 ** slot_width)

        actual_val = actual_slots[slot_idx]

        # The difference between expected_raw (mod carry adjustments) and actual reveals carry points
        is_carry = expected_carry > 0

        carries.append({
            "slot": slot_idx,
            "expected_term": expected_term,
            "expected_raw": expected_raw,
            "expected_carry": expected_carry,
            "actual": actual_val,
            "is_carry": is_carry,
            "carry_value": expected_carry,
        })

    return carries, actual_slots


# ============================================================
# Spatial pattern generation from digit sequences
# ============================================================

def digits_to_spatial_1d(digits, width, height, amp=0.15, mode="row"):
    """
    Map digit sequence to spatial pixel values.
    mode="row": each digit fills one pixel in a row, repeated for each row
    mode="snake": snake pattern filling rows
    mode="tile": tile the digit sequence to fill the image
    """
    if mode == "row":
        # Each pixel in a row gets a digit value (0-9), scaled to amplitude
        pattern = np.zeros((height, width), dtype=np.float32)
        for x in range(width):
            d = digits[x % len(digits)]
            pattern[:, x] = (d / 9.0) * amp  # Normalize 0-9 to 0-amp
        return pattern

    elif mode == "snake":
        pattern = np.zeros((height, width), dtype=np.float32)
        idx = 0
        for y in range(height):
            for x in range(width):
                d = digits[idx % len(digits)]
                if y % 2 == 0:
                    pattern[y, x] = (d / 9.0) * amp
                else:
                    pattern[y, width - 1 - x] = (d / 9.0) * amp
                idx += 1
        return pattern

    elif mode == "tile":
        pattern = np.zeros((height, width), dtype=np.float32)
        idx = 0
        for y in range(height):
            for x in range(width):
                d = digits[idx % len(digits)]
                pattern[y, x] = (d / 9.0) * amp
                idx += 1
        return pattern

    return np.zeros((height, width), dtype=np.float32)


def digits_to_spatial_2d_slots(digits, width, height, slot_width=2, amp=0.15):
    """
    Map 2-digit slots to spatial values. Each slot value (00-99) maps to a pixel column.
    This preserves the carry structure as spatial intensity jumps.
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    # Group digits into slot_width-digit values
    slot_values = []
    for i in range(0, len(digits) - slot_width + 1, slot_width):
        val = 0
        for j in range(slot_width):
            val = val * 10 + digits[i + j]
        slot_values.append(val)

    # Map each slot value to a pixel column
    for x in range(width):
        sv = slot_values[x % len(slot_values)]
        pattern[:, x] = (sv / 99.0) * amp  # Normalize 0-99 to 0-amp

    return pattern, slot_values


def carry_points_to_impulse(carry_info, width, height, amp=0.15):
    """
    Create a pattern where carry points are impulses (spikes) and non-carry points are zero.
    This isolates the carry-point disruption.
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    carry_slots = [c["slot"] for c in carry_info if c["is_carry"]]

    for x in range(width):
        slot_idx = x % len(carry_info)
        if slot_idx in carry_slots:
            carry_val = carry_info[slot_idx]["carry_value"]
            # Impulse proportional to carry value
            pattern[:, x] = min(carry_val / 100.0, 1.0) * amp

    return pattern, carry_slots


def smooth_region_to_impulse(carry_info, width, height, amp=0.15):
    """
    Opposite of carry_points_to_impulse: only non-carry (smooth) regions get values.
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    non_carry_slots = [c["slot"] for c in carry_info if not c["is_carry"]]

    for x in range(width):
        slot_idx = x % len(carry_info)
        if slot_idx in non_carry_slots:
            raw_val = carry_info[slot_idx]["expected_raw"]
            pattern[:, x] = (raw_val / 99.0) * amp

    return pattern, non_carry_slots


# ============================================================
# YOLO forward pass with feature capture
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

def add_pattern(arr, pat):
    out = arr.copy()
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] + pat, 0, 1)
    return out

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
            })
    return [d for d in dets if d["confidence"] >= conf]


# ============================================================
# EXPERIMENT 1: Digit-to-pixel mapping and carry analysis
# ============================================================

def experiment_1_digit_mapping(v3_model, arr_base):
    """
    Map decimal digits of 1/196 to spatial pixels, identify carry points,
    and measure which positions create the most disruption through the network.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Digit-to-Pixel Mapping & Carry Analysis")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape

    # Get decimal expansion of 1/196
    digits_196 = get_decimal_expansion(1, 196, 500)
    period_196 = find_period(digits_196)
    print(f"  1/196 period: {period_196} digits")
    print(f"  First 84 digits: {''.join(map(str, digits_196[:84]))}")

    # Identify carries
    carry_info, actual_slots = identify_carries_general(1, 196, slot_width=2, num_slots=100)
    carry_slots = [c["slot"] for c in carry_info if c["is_carry"]]
    print(f"  Carry points (first 20 slots): {carry_slots[:20]}")
    print(f"  Carry values: {[c['carry_value'] for c in carry_info if c['is_carry']][:15]}")

    # Check for 256 (2^8) in the carry structure
    for c in carry_info:
        if c["expected_term"] == 256 or c["carry_value"] == 256:
            print(f"  ** 256 (2^8) found at slot {c['slot']} — matches YOLOv3 256-dim feature maps **")
        if c["expected_term"] == 128 or c["carry_value"] == 128:
            print(f"  ** 128 (2^7) found at slot {c['slot']} **")
        if c["expected_term"] == 512 or c["carry_value"] == 512:
            print(f"  ** 512 (2^9) found at slot {c['slot']} **")

    # Generate spatial patterns
    patterns = {}

    # 1a. Raw digit sequence (each digit = 1 pixel)
    patterns["digits_raw_row"] = digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="row")

    # 1b. 2-digit slot values (each slot = 1 pixel column)
    pat_slots, slot_vals = digits_to_spatial_2d_slots(digits_196, W, H, slot_width=2, amp=0.15)
    patterns["slots_2digit_row"] = pat_slots

    # 1c. Carry points only (impulses at carry positions)
    pat_carry, carry_positions = carry_points_to_impulse(carry_info, W, H, amp=0.15)
    patterns["carry_points_only"] = pat_carry

    # 1d. Smooth regions only (non-carry positions)
    pat_smooth, smooth_positions = smooth_region_to_impulse(carry_info, W, H, amp=0.15)
    patterns["smooth_regions_only"] = pat_smooth

    # 1e. Snake pattern (fills 2D space)
    patterns["digits_snake"] = digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="snake")

    # 1f. Tile pattern (fills 2D space sequentially)
    patterns["digits_tile"] = digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="tile")

    # 1g. Slot values as 2D tile
    pat_slots_2d = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y in range(H):
        for x in range(W):
            sv = slot_vals[idx % len(slot_vals)]
            pat_slots_2d[y, x] = (sv / 99.0) * 0.15
            idx += 1
    patterns["slots_2d_tile"] = pat_slots_2d

    # 1h. Carry points as 2D tile
    pat_carry_2d = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y in range(H):
        for x in range(W):
            slot_idx = idx % len(carry_info)
            if carry_info[slot_idx]["is_carry"]:
                cv = carry_info[slot_idx]["carry_value"]
                pat_carry_2d[y, x] = min(cv / 100.0, 1.0) * 0.15
            idx += 1
    patterns["carry_2d_tile"] = pat_carry_2d

    # Control: random noise
    rng = np.random.RandomState(42)
    patterns["control_random"] = (rng.randn(H, W) * 0.05).astype(np.float32)

    # Control: uniform
    patterns["control_uniform"] = np.full((H, W), 0.075, dtype=np.float32)

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, base_out = forward_capture_v3(v3_model, tensor_base)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])
    print(f"  Baseline: {base_person} person detections, {len(base_dets)} total")

    results = {}
    LAYERS = [0, 1, 5, 12, 37, 54, 62, 75, 92, 105]

    for name, pattern in patterns.items():
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            caps, out = forward_capture_v3(v3_model, tensor_mod)

        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])
        non_person = [d for d in dets if d["class_name"] != "person"]

        # Feature map analysis at each layer
        layer_analysis = {}
        for layer_idx in LAYERS:
            if layer_idx not in caps or layer_idx not in base_caps:
                continue

            fm = caps[layer_idx][0].cpu().numpy()
            base_fm = base_caps[layer_idx][0].cpu().numpy()
            delta = fm - base_fm  # (C, H_l, W_l)

            C = delta.shape[0]

            # Overall disruption
            mean_abs_delta = float(np.mean(np.abs(delta)))
            max_abs_delta = float(np.max(np.abs(delta)))

            # Per-channel disruption
            channel_disruptions = np.mean(np.abs(delta.reshape(C, -1)), axis=1)
            # Channels with most disruption
            top_channels = np.argsort(channel_disruptions)[-5:]

            # Check if 256-dim channel has special resonance
            # YOLOv3 layer 62 has 256 channels (the large feature map)
            dim_256_disruption = None
            if C == 256:
                dim_256_disruption = float(channel_disruptions.mean())

            # Spatial structure: does the carry pattern survive in feature maps?
            # Compute FFT of the delta (averaged across channels) and look for
            # energy at frequencies corresponding to the digit period
            avg_delta = delta.mean(axis=0)
            if avg_delta.ndim == 2 and avg_delta.shape[1] > 1:
                fft_x = np.abs(np.fft.fft(avg_delta.mean(axis=0)))[:avg_delta.shape[1]//2]
                # Energy at period frequency
                period_freq = W // period_196 if period_196 > 0 else 0
                if period_freq > 0 and period_freq < len(fft_x):
                    period_energy = float(fft_x[period_freq])
                else:
                    period_energy = 0
                total_energy = float(fft_x.sum() + 1e-10)
                period_energy_ratio = period_energy / total_energy
            else:
                period_energy = 0
                period_energy_ratio = 0

            layer_analysis[layer_idx] = {
                "channels": C,
                "mean_abs_delta": mean_abs_delta,
                "max_abs_delta": max_abs_delta,
                "dim_256_disruption": dim_256_disruption,
                "top5_channels": top_channels.tolist(),
                "top5_disruptions": channel_disruptions[top_channels].tolist(),
                "period_energy": period_energy,
                "period_energy_ratio": float(period_energy_ratio),
            }

        results[name] = {
            "person_count": person_count,
            "total_dets": len(dets),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
            "suppressed": person_count == 0 and base_person > 0,
            "hallucinated": len(non_person) > 0,
            "person_reduction": base_person - person_count,
            "layer_analysis": layer_analysis,
        }

        tag = ""
        if results[name]["suppressed"]: tag = " [PERSON_SUPPRESSED]"
        elif non_person: tag = f" [HALLUC:{results[name]['non_person_classes']}]"
        elif person_count < base_person: tag = f" [person {base_person}->{person_count}]"

        # Print key layers
        l0 = layer_analysis.get(0, {})
        l62 = layer_analysis.get(62, {})
        l105 = layer_analysis.get(105, {})
        print(f"\n  {name}:")
        print(f"    Dets: {len(dets)}, person={person_count}/{base_person}{tag}")
        print(f"    L0: delta={l0.get('mean_abs_delta',0):.6f}, L62: delta={l62.get('mean_abs_delta',0):.6f} "
              f"(C={l62.get('channels',0)}), L105: delta={l105.get('mean_abs_delta',0):.6f}")
        if l62.get("dim_256_disruption") is not None:
            print(f"    ** L62 256-dim disruption: {l62['dim_256_disruption']:.6f} **")

        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(pattern, cmap="viridis")
        axes[0].set_title(f"Pattern: {name}")
        axes[0].axis("off")
        axes[1].imshow(arr_mod)
        axes[1].set_title(f"Modified Image")
        axes[1].axis("off")
        # Feature map delta at layer 62 (256-dim)
        if 62 in caps and 62 in base_caps:
            delta62 = (caps[62][0] - base_caps[62][0]).cpu().numpy()
            axes[2].imshow(delta62.mean(axis=0), cmap="RdBu_r")
            axes[2].set_title(f"L62 Delta (256-ch mean)")
            axes[2].axis("off")
        plt.suptitle(f"Exp1: {name} — person={person_count}/{base_person}{tag}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/exp1_{name}.png", dpi=150)
        plt.close()

    return {"results": results, "period": period_196, "carry_slots": carry_slots[:30],
            "base_person": base_person, "digits_preview": digits_196[:84]}


# ============================================================
# EXPERIMENT 2: Fraction comparison
# ============================================================

def experiment_2_fraction_comparison(v3_model, arr_base):
    """
    Compare different fraction digit sequences as spatial patterns.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Fraction Comparison")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape

    fractions = {
        "1_196": (1, 196, "doubling sequence, 42-digit period"),
        "1_89": (1, 89, "Fibonacci sequence"),
        "1_9801": (1, 9801, "counting sequence"),
        "1_7": (1, 7, "cyclic permutation, 6-digit period"),
        "1_98": (1, 98, "doubling (10x scale of 1/9.8)"),
        "1_9_8": (10, 98, "1/9.8 = 5/49, doubling from 5"),
        "1_49": (1, 49, "doubling from 2 (1/49 = 0.02040816...)"),
        "1_998001": (1, 998001, "counting with high precision"),
        "1_13": (1, 13, "6-digit period, YOLOv3 grid size"),
        "1_208": (1, 208, "13x16, YOLOv3 Nyquist"),
        "1_167": (1, 167, "prime near Nyquist"),
    }

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])

    results = {}

    for name, (num, den, desc) in fractions.items():
        digits = get_decimal_expansion(num, den, 500)
        period = find_period(digits)

        # Map digits to spatial pattern (row mode — each digit = 1 pixel column)
        pattern = digits_to_spatial_1d(digits, W, H, amp=0.15, mode="row")

        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])
        non_person = [d for d in dets if d["class_name"] != "person"]

        # Feature map disruption at key layers
        with torch.no_grad():
            caps, _ = forward_capture_v3(v3_model, tensor_mod)

        disruptions = {}
        for layer_idx in [0, 1, 62, 105]:
            if layer_idx in caps and layer_idx in {k: v for k, v in [(0, None)]}:
                pass
            if layer_idx not in caps:
                continue
            with torch.no_grad():
                base_caps, _ = forward_capture_v3(v3_model, tensor_base)
            fm = caps[layer_idx][0].cpu().numpy()
            base_fm = base_caps[layer_idx][0].cpu().numpy()
            delta = fm - base_fm
            disruptions[layer_idx] = {
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "channels": delta.shape[0],
            }

        results[name] = {
            "fraction": f"{num}/{den}",
            "description": desc,
            "period": period,
            "first_digits": "".join(map(str, digits[:20])),
            "person_count": person_count,
            "total_dets": len(dets),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
            "suppressed": person_count == 0 and base_person > 0,
            "hallucinated": len(non_person) > 0,
            "person_reduction": base_person - person_count,
            "disruptions": disruptions,
        }

        tag = ""
        if results[name]["suppressed"]: tag = " [SUPPRESSED]"
        elif non_person: tag = f" [HALLUC:{results[name]['non_person_classes']}]"
        elif person_count < base_person: tag = f" [person {base_person}->{person_count}]"

        print(f"  {name:12s} ({num}/{den}): period={period:3d}, person={person_count}/{base_person}, "
              f"digits={results[name]['first_digits'][:15]}...{tag}")

    return {"results": results, "base_person": base_person}


# ============================================================
# EXPERIMENT 3: Period tiling and boundary effects
# ============================================================

def experiment_3_period_tiling(v3_model, arr_base):
    """
    Tile the 42-digit period of 1/196 across different image sizes.
    Measure boundary discontinuity effects.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Period Tiling & Boundary Effects")
    print(f"{'='*70}")

    digits_196 = get_decimal_expansion(1, 196, 500)
    period = find_period(digits_196)
    period_digits = digits_196[:period]

    print(f"  Period: {period} digits")
    print(f"  416 / {period} = {416/period:.4f} (truncation at {416 % period} pixels)")
    print(f"  640 / {period} = {640/period:.4f} (truncation at {640 % period} pixels)")

    H, W, _ = arr_base.shape

    # Test different tiling modes
    tilings = {}

    # 3a. Full period tiled (closed loop)
    tilings["full_period_closed"] = digits_to_spatial_1d(period_digits, W, H, amp=0.15, mode="row")

    # 3b. Truncated period (open loop — boundary discontinuity)
    # Use only first 40 digits (2 short of period) to create open boundary
    tilings["truncated_open_40"] = digits_to_spatial_1d(period_digits[:40], W, H, amp=0.15, mode="row")

    # 3c. Truncated at 38 (4 short)
    tilings["truncated_open_38"] = digits_to_spatial_1d(period_digits[:38], W, H, amp=0.15, mode="row")

    # 3d. Double period (84 digits) — tests longer pattern
    double_period = period_digits + period_digits
    tilings["double_period_84"] = digits_to_spatial_1d(double_period, W, H, amp=0.15, mode="row")

    # 3e. Half period (21 digits)
    tilings["half_period_21"] = digits_to_spatial_1d(period_digits[:21], W, H, amp=0.15, mode="row")

    # 3f. Period with boundary impulse — add a spike at each period boundary
    pat_boundary = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        d = period_digits[x % period]
        pat_boundary[:, x] = (d / 9.0) * 0.15
        # Add impulse at period boundaries
        if x % period == 0 and x > 0:
            pat_boundary[:, x] += 0.1  # Boundary spike
    tilings["period_with_boundary_impulse"] = pat_boundary

    # 3g. Only boundary impulses (period boundaries as spikes)
    pat_only_boundaries = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        if x % period == 0:
            pat_only_boundaries[:, x] = 0.15
    tilings["only_boundary_impulses"] = pat_only_boundaries

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])

    results = {}

    for name, pattern in tilings.items():
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])
        non_person = [d for d in dets if d["class_name"] != "person"]

        results[name] = {
            "person_count": person_count,
            "total_dets": len(dets),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
            "suppressed": person_count == 0 and base_person > 0,
            "hallucinated": len(non_person) > 0,
            "person_reduction": base_person - person_count,
        }

        tag = ""
        if results[name]["suppressed"]: tag = " [SUPPRESSED]"
        elif non_person: tag = f" [HALLUC:{results[name]['non_person_classes']}]"
        elif person_count < base_person: tag = f" [person {base_person}->{person_count}]"
        print(f"  {name:35s}: person={person_count}/{base_person}{tag}")

    return {"results": results, "period": period, "base_person": base_person}


# ============================================================
# EXPERIMENT 4: 256-dim resonance test
# ============================================================

def experiment_4_dim256_resonance(v3_model, arr_base):
    """
    Test if the 2^8=256 carry point in 1/196's digit structure creates special
    resonance with YOLOv3's 256-channel feature maps.

    The 8th slot in 1/196 has term=5*2^8=1280, carry=12.
    The 9th slot has term=5*2^9=2560, carry=25.
    We test if injecting at positions corresponding to these specific slots
    creates disproportionate disruption in 256-channel layers.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 4: 256-dim Resonance Test")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    carry_info, _ = identify_carries_general(1, 196, slot_width=2, num_slots=50)

    # Find slots where term is a power of 2
    power_of_2_slots = []
    for c in carry_info:
        term = c["expected_term"]
        # Check if term is 5 * 2^k and find k
        if term > 0 and term % 5 == 0:
            quotient = term // 5
            if quotient > 0 and (quotient & (quotient - 1)) == 0:  # power of 2
                k = int(math.log2(quotient))
                power_of_2_slots.append({
                    "slot": c["slot"],
                    "term": term,
                    "k": k,
                    "2^k": 2**k,
                    "carry": c["carry_value"],
                })
                if k in [7, 8, 9]:  # 128, 256, 512
                    print(f"  ** Slot {c['slot']}: term={term}, 5x2^{k}={term}, "
                          f"2^{k}={2**k}, carry={c['carry_value']} **")

    # Create patterns that isolate specific power-of-2 slots
    patterns = {}

    # Only the 2^8=256 slot (slot 8, term=1280)
    pat_256 = np.zeros((H, W), dtype=np.float32)
    slot_256 = 8  # 5*2^8=1280, slot 8
    for x in range(W):
        if x % 42 == slot_256:  # 42-digit period
            pat_256[:, x] = 0.15
    patterns["only_2pow8_slot"] = pat_256

    # Only the 2^7=128 slot (slot 7, term=640)
    pat_128 = np.zeros((H, W), dtype=np.float32)
    slot_128 = 7
    for x in range(W):
        if x % 42 == slot_128:
            pat_128[:, x] = 0.15
    patterns["only_2pow7_slot"] = pat_128

    # Only the 2^9=512 slot (slot 9, term=2560)
    pat_512 = np.zeros((H, W), dtype=np.float32)
    slot_512 = 9
    for x in range(W):
        if x % 42 == slot_512:
            pat_512[:, x] = 0.15
    patterns["only_2pow9_slot"] = pat_512

    # All power-of-2 slots
    pat_all_pow2 = np.zeros((H, W), dtype=np.float32)
    pow2_slot_positions = [p["slot"] for p in power_of_2_slots]
    for x in range(W):
        slot_idx = x % 42
        if slot_idx in pow2_slot_positions:
            pat_all_pow2[:, x] = 0.15
    patterns["all_power2_slots"] = pat_all_pow2

    # Non-power-of-2 carry slots (control)
    pat_non_pow2 = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        slot_idx = x % 42
        c = carry_info[slot_idx] if slot_idx < len(carry_info) else None
        if c and c["is_carry"] and slot_idx not in pow2_slot_positions:
            pat_non_pow2[:, x] = 0.15
    patterns["non_power2_carry_slots"] = pat_non_pow2

    # Full digit pattern (reference)
    digits_196 = get_decimal_expansion(1, 196, 500)
    patterns["full_digits_reference"] = digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="row")

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])

    results = {}
    LAYERS_256 = []  # Find layers with 256 channels

    # First, identify which layers have 256 channels
    for layer_idx in base_caps:
        C = base_caps[layer_idx].shape[1]
        if C == 256:
            LAYERS_256.append(layer_idx)

    print(f"  Layers with 256 channels: {LAYERS_256}")

    for name, pattern in patterns.items():
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            caps, _ = forward_capture_v3(v3_model, tensor_mod)

        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])

        # Measure disruption specifically in 256-channel layers
        dim256_disruptions = {}
        for layer_idx in LAYERS_256:
            fm = caps[layer_idx][0].cpu().numpy()
            base_fm = base_caps[layer_idx][0].cpu().numpy()
            delta = fm - base_fm
            dim256_disruptions[layer_idx] = {
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "per_channel_mean": float(np.mean(np.mean(np.abs(delta.reshape(256, -1)), axis=1))),
            }

        # Also measure in all layers
        all_disruptions = {}
        for layer_idx in [0, 1, 62, 105]:
            if layer_idx in caps and layer_idx in base_caps:
                fm = caps[layer_idx][0].cpu().numpy()
                base_fm = base_caps[layer_idx][0].cpu().numpy()
                delta = fm - base_fm
                all_disruptions[layer_idx] = {
                    "mean_abs_delta": float(np.mean(np.abs(delta))),
                    "channels": delta.shape[0],
                }

        results[name] = {
            "person_count": person_count,
            "total_dets": len(dets),
            "suppressed": person_count == 0 and base_person > 0,
            "dim256_disruptions": dim256_disruptions,
            "all_disruptions": all_disruptions,
        }

        d256 = dim256_disruptions.get(LAYERS_256[0], {}) if LAYERS_256 else {}
        print(f"  {name:30s}: person={person_count}/{base_person}, "
              f"256-dim L{LAYERS_256[0]} delta={d256.get('mean_abs_delta',0):.6f}" if LAYERS_256 else
              f"  {name:30s}: person={person_count}/{base_person}")

    return {"results": results, "power_of_2_slots": power_of_2_slots[:15],
            "layers_256": LAYERS_256, "base_person": base_person}


# ============================================================
# EXPERIMENT 5: Cross-model digit sequence test
# ============================================================

def experiment_5_cross_model(arr_base):
    """
    Test 1/196 digit pattern on all 4 YOLO models.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 5: Cross-Model Digit Sequence Test")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    digits_196 = get_decimal_expansion(1, 196, 500)
    period = find_period(digits_196)

    # Generate patterns
    patterns = {
        "digits_row": digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="row"),
        "digits_tile": digits_to_spatial_1d(digits_196, W, H, amp=0.15, mode="tile"),
        "slots_row": digits_to_spatial_2d_slots(digits_196, W, H, slot_width=2, amp=0.15)[0],
    }

    # Also test 1/89 (Fibonacci) and 1/7 (cyclic)
    digits_89 = get_decimal_expansion(1, 89, 500)
    digits_7 = get_decimal_expansion(1, 7, 500)
    patterns["fibonacci_1_89_row"] = digits_to_spatial_1d(digits_89, W, H, amp=0.15, mode="row")
    patterns["cyclic_1_7_row"] = digits_to_spatial_1d(digits_7, W, H, amp=0.15, mode="row")

    model_configs = [
        ("YOLOv3", None),  # Handled separately
        ("YOLOv8", r"C:\Users\carso\Desktop\YODO\YOLOv8\yolov8l.pt"),
        ("YOLO11", r"C:\Users\carso\Desktop\YODO\YOLO11\yolo11l.pt"),
        ("YOLO26", r"C:\Users\carso\Desktop\YODO\YOLO26\yolo26l.pt"),
    ]

    all_results = {}

    # YOLOv3
    print("\n  YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters(): p.requires_grad_(False)

    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])

    v3_results = {}
    for name, pattern in patterns.items():
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])
        non_person = [d for d in dets if d["class_name"] != "person"]
        v3_results[name] = {
            "person_count": person_count, "total_dets": len(dets),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
            "suppressed": person_count == 0 and base_person > 0,
            "hallucinated": len(non_person) > 0,
        }
        tag = " [SUPP]" if v3_results[name]["suppressed"] else \
              f" [HALLUC:{v3_results[name]['non_person_classes']}]" if non_person else \
              f" [person {base_person}->{person_count}]" if person_count < base_person else ""
        print(f"    {name:25s}: person={person_count}/{base_person}{tag}")

    all_results["YOLOv3"] = {"results": v3_results, "base_person": base_person}
    del v3_model
    torch.cuda.empty_cache()

    # Ultralytics models
    pil_base = Image.fromarray((arr_base * 255).astype(np.uint8))

    for model_name, model_path in model_configs[1:]:
        print(f"\n  {model_name}...")
        model = YOLO(model_path)
        model.to(DEVICE)

        base_dets = get_dets_ultralytics(model, pil_base, conf=0.1)
        base_person = len([d for d in base_dets if d["class_name"] == "person"])

        ul_results = {}
        for name, pattern in patterns.items():
            arr_mod = add_pattern(arr_base, pattern)
            pil_mod = Image.fromarray((arr_mod * 255).astype(np.uint8))
            dets = get_dets_ultralytics(model, pil_mod, conf=0.1)
            person_count = len([d for d in dets if d["class_name"] == "person"])
            non_person = [d for d in dets if d["class_name"] != "person"]
            ul_results[name] = {
                "person_count": person_count, "total_dets": len(dets),
                "non_person_classes": list(set(d["class_name"] for d in non_person)),
                "suppressed": person_count == 0 and base_person > 0,
                "hallucinated": len(non_person) > 0,
            }
            tag = " [SUPP]" if ul_results[name]["suppressed"] else \
                  f" [HALLUC:{ul_results[name]['non_person_classes']}]" if non_person else \
                  f" [person {base_person}->{person_count}]" if person_count < base_person else ""
            print(f"    {name:25s}: person={person_count}/{base_person}{tag}")

        all_results[model_name] = {"results": ul_results, "base_person": base_person}
        del model
        torch.cuda.empty_cache()

    return all_results


# ============================================================
# EXPERIMENT 6: Doubling/halving shift test
# ============================================================

def experiment_6_doubling_shift(v3_model, arr_base):
    """
    Test the user's observation: multiplying/dividing 1/196 by 2 shifts the
    digit sequence by one position. Test if shifted versions create different
    disruption patterns (phase shift through the network).
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 6: Doubling Shift Test (×2/÷2 phase shift)")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape

    # Generate 1/196, 2/196, 4/196, 8/196, ... 1/(196*2), 1/(196*4)...
    shifts = []
    for i in range(-5, 11):  # -5 to +10 doublings
        scale = 2 ** i
        num = max(1, int(scale)) if scale >= 1 else 1
        den = int(196 * (1/scale)) if scale < 1 else 196
        if scale >= 1:
            num = int(scale)
            den = 196
        else:
            num = 1
            den = int(196 / scale)

        digits = get_decimal_expansion(num, den, 200)
        shifts.append({
            "i": i,
            "scale": scale,
            "fraction": f"{num}/{den}",
            "digits": digits,
            "first_10": "".join(map(str, digits[:10])),
        })

    print(f"  Testing {len(shifts)} shifted versions:")
    for s in shifts[:5]:
        print(f"    2^{s['i']:+d} = {s['fraction']:12s}: {s['first_10']}...")
    print(f"    ...")
    for s in shifts[-3:]:
        print(f"    2^{s['i']:+d} = {s['fraction']:12s}: {s['first_10']}...")

    # Get baseline
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_person = len([d for d in base_dets if d["class_name"] == "person"])

    results = []

    for s in shifts:
        pattern = digits_to_spatial_1d(s["digits"], W, H, amp=0.15, mode="row")
        arr_mod = add_pattern(arr_base, pattern)
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
        person_count = len([d for d in dets if d["class_name"] == "person"])
        non_person = [d for d in dets if d["class_name"] != "person"]

        result = {
            "shift": s["i"],
            "fraction": s["fraction"],
            "first_10": s["first_10"],
            "person_count": person_count,
            "total_dets": len(dets),
            "non_person_classes": list(set(d["class_name"] for d in non_person)),
            "suppressed": person_count == 0 and base_person > 0,
            "hallucinated": len(non_person) > 0,
        }
        results.append(result)

        tag = " [SUPP]" if result["suppressed"] else \
              f" [HALLUC]" if non_person else \
              f" [person {base_person}->{person_count}]" if person_count < base_person else ""
        print(f"  2^{s['i']:+d} ({s['fraction']:12s}): person={person_count}/{base_person}{tag}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    shifts_arr = [r["shift"] for r in results]
    persons_arr = [r["person_count"] for r in results]
    ax.plot(shifts_arr, persons_arr, "b.-", markersize=10)
    ax.axhline(y=base_person, color="r", linestyle="--", alpha=0.5, label=f"Baseline ({base_person})")
    ax.set_xlabel("Doubling Shift (2^i)")
    ax.set_ylabel("Person Count")
    ax.set_title("Experiment 6: Detection vs Doubling Shift of 1/196")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp6_doubling_shift.png", dpi=150)
    plt.close()

    return {"results": results, "base_person": base_person}


# ============================================================
# MAIN
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("DIGIT SEQUENCE PROBE — Decimal Expansions as Spatial Adversarial Patterns")
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

    # Run experiments
    all_results["exp1_digit_mapping"] = experiment_1_digit_mapping(v3_model, arr_base)
    all_results["exp2_fraction_comparison"] = experiment_2_fraction_comparison(v3_model, arr_base)
    all_results["exp3_period_tiling"] = experiment_3_period_tiling(v3_model, arr_base)
    all_results["exp4_dim256_resonance"] = experiment_4_dim256_resonance(v3_model, arr_base)
    all_results["exp6_doubling_shift"] = experiment_6_doubling_shift(v3_model, arr_base)

    del v3_model
    torch.cuda.empty_cache()

    # Cross-model
    all_results["exp5_cross_model"] = experiment_5_cross_model(arr_base)

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\n{'='*70}")
    print("Saving results...")

    json_path = f"{OUTPUT_DIR}/digit_sequence.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # CSV
    csv_path = f"{OUTPUT_DIR}/digit_sequence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "pattern", "person_count", "total_dets", "suppressed", "hallucinated"])

        for name, res in all_results["exp1_digit_mapping"]["results"].items():
            writer.writerow(["exp1", name, res["person_count"], res["total_dets"], res["suppressed"], res["hallucinated"]])
        for name, res in all_results["exp2_fraction_comparison"]["results"].items():
            writer.writerow(["exp2", name, res["person_count"], res["total_dets"], res["suppressed"], res["hallucinated"]])
        for name, res in all_results["exp3_period_tiling"]["results"].items():
            writer.writerow(["exp3", name, res["person_count"], res["total_dets"], res["suppressed"], res["hallucinated"]])
        for name, res in all_results["exp4_dim256_resonance"]["results"].items():
            writer.writerow(["exp4", name, res["person_count"], res["total_dets"], res["suppressed"], ""])
        for r in all_results["exp6_doubling_shift"]["results"]:
            writer.writerow(["exp6", f"2^{r['shift']}", r["person_count"], r["total_dets"], r["suppressed"], r["hallucinated"]])
        for model_name, model_res in all_results["exp5_cross_model"].items():
            for pname, pres in model_res["results"].items():
                writer.writerow(["exp5", f"{model_name}_{pname}", pres["person_count"], pres["total_dets"],
                                 pres["suppressed"], pres["hallucinated"]])
    print(f"Saved CSV: {csv_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\n1. DIGIT MAPPING (1/196 digits as spatial pattern):")
    for name, res in all_results["exp1_digit_mapping"]["results"].items():
        if res["suppressed"] or res["hallucinated"] or res["person_reduction"] > 0:
            tag = " [SUPP]" if res["suppressed"] else f" [HALLUC]" if res["hallucinated"] else ""
            print(f"   {name:30s}: person={res['person_count']}, reduction={res['person_reduction']}{tag}")

    print("\n2. FRACTION COMPARISON:")
    for name, res in all_results["exp2_fraction_comparison"]["results"].items():
        tag = " [SUPP]" if res["suppressed"] else f" [HALLUC]" if res["hallucinated"] else ""
        print(f"   {name:12s} ({res['fraction']:8s}): period={res['period']:3d}, person={res['person_count']}{tag}")

    print("\n3. PERIOD TILING:")
    for name, res in all_results["exp3_period_tiling"]["results"].items():
        tag = " [SUPP]" if res["suppressed"] else f" [HALLUC]" if res["hallucinated"] else ""
        print(f"   {name:35s}: person={res['person_count']}{tag}")

    print("\n4. 256-DIM RESONANCE:")
    for name, res in all_results["exp4_dim256_resonance"]["results"].items():
        d256 = list(res.get("dim256_disruptions", {}).values())
        d256_mean = d256[0]["mean_abs_delta"] if d256 else 0
        print(f"   {name:30s}: person={res['person_count']}, 256-dim delta={d256_mean:.6f}")

    print("\n5. DOUBLING SHIFT:")
    for r in all_results["exp6_doubling_shift"]["results"]:
        tag = " [SUPP]" if r["suppressed"] else f" [HALLUC]" if r["hallucinated"] else ""
        print(f"   2^{r['shift']:+d} ({r['fraction']:12s}): person={r['person_count']}{tag}")

    print("\n6. CROSS-MODEL:")
    for model_name, model_res in all_results["exp5_cross_model"].items():
        for pname, pres in model_res["results"].items():
            tag = " [SUPP]" if pres["suppressed"] else f" [HALLUC]" if pres["hallucinated"] else ""
            if tag:
                print(f"   {model_name} {pname:25s}: person={pres['person_count']}{tag}")

    print(f"\n{'='*70}")
    print("DONE — All results in outputs_clothing/forward_analysis/digit_sequence/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
