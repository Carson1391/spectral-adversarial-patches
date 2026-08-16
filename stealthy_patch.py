"""
Stealthy Regime Probe & Print-Ready Patch Generator

1. Tests varying patch sizes (5-25% area) and amplitudes (0.05-0.30)
2. For each combo, records which bboxes are suppressed (near patch = wearer vs far = bystander)
3. Finds the sweet spot: wearer suppressed, bystanders preserved, embedding still poisoned
4. Generates print-ready 3600x4800 300dpi PNG of the optimal patch for physical testing
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\stealthy_patch"
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

# Print export constants
PRINT_DPI = 300
PRINT_W_IN = 12
PRINT_H_IN = 16
PRINT_W_PX = int(PRINT_W_IN * PRINT_DPI)  # 3600
PRINT_H_PX = int(PRINT_H_IN * PRINT_DPI)  # 4800


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
    img_mask = Image.new("F", (W, H), 0.0)
    draw = ImageDraw.Draw(img_mask)
    polygon = endpoints + [endpoints[0]]
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


# ============================================================
# YOLOv3 forward pass with feature capture
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
            # pytorchyolo output format: (cx, cy, w, h, conf, cls)
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [cx, cy, w, h],
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            })
    return dets


# ============================================================
# Experiment 1: Collateral Suppression Analysis
# ============================================================

def experiment_collateral(v3_model, arr_base):
    """
    Test varying patch sizes and amplitudes.
    For each combo, classify suppressed detections as:
    - WEARER: bbox center within patch radius
    - BYSTANDER: bbox center outside patch radius
    Find the sweet spot: wearer suppressed, bystanders preserved.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Collateral Suppression Analysis")
    print(f"{'='*70}")

    H, W, _ = arr_base.shape
    # Main person detected at (183, 292) with conf=0.999 — place patch on torso
    patch_cx, patch_cy = 183, 292  # actual person center from find_persons.py

    # Get baseline detections
    tensor_base = torch.from_numpy(arr_base).permute(2,0,1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_persons = [d for d in base_dets if d["class_name"] == "person"]
    print(f"  Baseline: {len(base_persons)} person detections")

    # Classify baseline persons as wearer vs bystander based on distance to patch center
    # Wearer threshold based on the main person's bbox size
    # Main person is 58x170 — wearer zone = 60px (torso width)
    wearer_threshold = 60
    for d in base_persons:
        dist = math.sqrt((d["cx"] - patch_cx)**2 + (d["cy"] - patch_cy)**2)
        d["dist_to_patch"] = dist
        d["is_wearer"] = dist < wearer_threshold

    wearers = [d for d in base_persons if d["is_wearer"]]
    bystanders = [d for d in base_persons if not d["is_wearer"]]
    print(f"  Wearer detections (dist<{wearer_threshold}): {len(wearers)}")
    print(f"  Bystander detections (dist>={wearer_threshold}): {len(bystanders)}")
    for d in base_persons:
        tag = "WEARER" if d["is_wearer"] else "bystander"
        print(f"    {tag}: cx={d['cx']:.0f}, cy={d['cy']:.0f}, conf={d['confidence']:.3f}, dist={d['dist_to_patch']:.0f}")

    # Test grid: patch sizes × amplitudes × textures
    patch_configs = [
        # (label, ray_lengths, num_rays)
        ("tiny_r40", [35, 30, 40, 25, 38, 28, 42, 30, 35, 28, 40, 25], 12),
        ("small_r60", [55, 45, 60, 40, 58, 48, 62, 45, 55, 48, 60, 40], 12),
        ("medium_r80", [75, 60, 80, 55, 78, 65, 82, 60, 75, 65, 80, 55], 12),
        ("large_r100", [95, 75, 100, 70, 98, 80, 102, 75, 95, 80, 100, 70], 12),
        ("xlarge_r120", [115, 90, 120, 85, 118, 95, 122, 90, 115, 95, 120, 85], 12),
    ]

    amplitudes = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    textures = {
        "k167_d": lambda amp: make_sinusoid(H, W, 167, 167, 0, amp),
        "k167_square": lambda amp: make_square_wave(H, W, 167, 167, amp),
        "k200_d": lambda amp: make_sinusoid(H, W, 200, 200, 0, amp),
        "stripes_13px": lambda amp: (amp * np.sign(np.sin(2*np.pi*np.arange(W)/13.0))).astype(np.float32) * np.ones((H,W), dtype=np.float32),
    }

    # Also get baseline embeddings for poisoning measurement
    with torch.no_grad():
        base_caps, _ = forward_capture_v3(v3_model, tensor_base)
    base_emb_l81 = base_caps[81][0].cpu().numpy().mean(axis=(1,2)) if 81 in base_caps else None
    base_emb_l93 = base_caps[93][0].cpu().numpy().mean(axis=(1,2)) if 93 in base_caps else None
    base_emb_l105 = base_caps[105][0].cpu().numpy().mean(axis=(1,2)) if 105 in base_caps else None

    results = []
    sweet_spots = []

    for patch_label, ray_lengths, num_rays in patch_configs:
        mask, endpoints = make_deformable_mask(H, W, patch_cx, patch_cy, ray_lengths, num_rays)
        patch_area_pct = float(np.mean(mask)) * 100
        patch_radius = float(np.mean(ray_lengths))

        for tex_name, tex_fn in textures.items():
            for amp in amplitudes:
                texture = tex_fn(amp)
                arr_mod = add_pattern_patch(arr_base, texture, mask)
                tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

                dets = get_dets_v3(v3_model, tensor_mod, conf=0.1)
                persons = [d for d in dets if d["class_name"] == "person"]

                # Count wearer vs bystander survivors
                wearer_survived = 0
                bystander_survived = 0
                for d in persons:
                    dist = math.sqrt((d["cx"] - patch_cx)**2 + (d["cy"] - patch_cy)**2)
                    if dist < patch_radius * 0.8:  # wearer zone scales with patch
                        wearer_survived += 1
                    else:
                        bystander_survived += 1

                wearer_suppressed = len(wearers) - wearer_survived
                bystander_suppressed = len(bystanders) - bystander_survived

                # Embedding corruption (poisoning potential)
                with torch.no_grad():
                    mod_caps, _ = forward_capture_v3(v3_model, tensor_mod)
                emb_l105 = mod_caps[105][0].cpu().numpy().mean(axis=(1,2)) if 105 in mod_caps else None
                emb_l81 = mod_caps[81][0].cpu().numpy().mean(axis=(1,2)) if 81 in mod_caps else None

                l2_l105 = float(np.linalg.norm(emb_l105 - base_emb_l105)) if emb_l105 is not None else 0
                l2_l81 = float(np.linalg.norm(emb_l81 - base_emb_l81)) if emb_l81 is not None else 0
                cos_l105 = float(np.dot(emb_l105, base_emb_l105) / (np.linalg.norm(emb_l105) * np.linalg.norm(base_emb_l105) + 1e-10)) if emb_l105 is not None else 1.0

                result = {
                    "patch": patch_label,
                    "texture": tex_name,
                    "amplitude": amp,
                    "patch_area_pct": patch_area_pct,
                    "patch_radius": patch_radius,
                    "total_persons": len(persons),
                    "wearer_survived": wearer_survived,
                    "wearer_suppressed": wearer_suppressed,
                    "bystander_survived": bystander_survived,
                    "bystander_suppressed": bystander_suppressed,
                    "wearer_suppression_rate": wearer_suppressed / max(len(wearers), 1),
                    "bystander_suppression_rate": bystander_suppressed / max(len(bystanders), 1),
                    "emb_l2_l105": l2_l105,
                    "emb_l2_l81": l2_l81,
                    "emb_cos_l105": cos_l105,
                }
                results.append(result)

                # Check for sweet spot: wearer mostly suppressed, bystanders mostly preserved
                if (wearer_suppressed >= 1 and
                    bystander_suppressed == 0 and
                    l2_l105 > 0.5):  # embedding still corrupted
                    sweet_spots.append(result)

                # Print notable results
                if bystander_suppressed > 0:
                    tag = f" COLLATERAL:{bystander_suppressed}"
                elif wearer_suppressed > 0:
                    tag = f" wearer_suppressed:{wearer_suppressed}"
                else:
                    tag = ""

                if wearer_suppressed > 0 or bystander_suppressed > 0:
                    print(f"  {patch_label:14s} {tex_name:14s} amp={amp:.2f}: "
                          f"persons={len(persons):2d} wearer={wearer_survived}/{len(wearers)} "
                          f"bystander={bystander_survived}/{len(bystanders)} "
                          f"emb_L2={l2_l105:.3f}{tag}")

    print(f"\n  SWEET SPOTS (wearer suppressed, 0 collateral, embedding corrupted):")
    if sweet_spots:
        # Sort by embedding corruption (higher = better poisoning) then by wearer suppression
        sweet_spots.sort(key=lambda x: (-x["emb_l2_l105"], -x["wearer_suppression_rate"]))
        for sp in sweet_spots[:10]:
            print(f"    {sp['patch']:14s} {sp['texture']:14s} amp={sp['amplitude']:.2f}: "
                  f"wearer={sp['wearer_suppressed']}/{len(wearers)} suppressed, "
                  f"bystander={sp['bystander_suppressed']}/{len(bystanders)} collateral, "
                  f"emb_L2={sp['emb_l2_l105']:.3f}, cos={sp['emb_cos_l105']:.6f}")
    else:
        print(f"    None found — relaxing to bystander_suppressed <= 1")
        relaxed = [r for r in results if r["wearer_suppressed"] >= 1 and r["bystander_suppressed"] <= 1 and r["emb_l2_l105"] > 0.5]
        if relaxed:
            relaxed.sort(key=lambda x: (x["bystander_suppressed"], -x["emb_l2_l105"]))
            for sp in relaxed[:10]:
                print(f"    {sp['patch']:14s} {sp['texture']:14s} amp={sp['amplitude']:.2f}: "
                      f"wearer={sp['wearer_suppressed']}/{len(wearers)} suppressed, "
                      f"bystander={sp['bystander_suppressed']}/{len(bystanders)} collateral, "
                      f"emb_L2={sp['emb_l2_l105']:.3f}")
        else:
            print(f"    Still none — showing best suppression with minimal collateral:")
            best = sorted(results, key=lambda x: (x["bystander_suppressed"], -x["wearer_suppression_rate"]))[:5]
            for sp in best:
                print(f"    {sp['patch']:14s} {sp['texture']:14s} amp={sp['amplitude']:.2f}: "
                      f"wearer={sp['wearer_suppressed']}/{len(wearers)}, "
                      f"bystander={sp['bystander_suppressed']}/{len(bystanders)}, "
                      f"emb_L2={sp['emb_l2_l105']:.3f}")

    # Plot: suppression map
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (patch_label, ray_lengths, num_rays) in enumerate(patch_configs[:6]):
        ax = axes[i//3, i%3]
        patch_results = [r for r in results if r["patch"] == patch_label and r["texture"] == "k167_d"]
        if not patch_results:
            ax.set_title(f"{patch_label} (no data)")
            continue
        amps = [r["amplitude"] for r in patch_results]
        wearer_rates = [r["wearer_suppression_rate"] for r in patch_results]
        bystander_rates = [r["bystander_suppression_rate"] for r in patch_results]
        emb_l2s = [r["emb_l2_l105"] for r in patch_results]

        ax.plot(amps, wearer_rates, 'r-o', label="Wearer suppression", markersize=4)
        ax.plot(amps, bystander_rates, 'b-s', label="Bystander collateral", markersize=4)
        ax2 = ax.twinx()
        ax2.plot(amps, emb_l2s, 'g--^', label="Embedding L2", markersize=4, alpha=0.7)
        ax2.set_ylabel("Embedding L2 (poisoning)", color='green')
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Suppression Rate")
        ax.set_title(f"{patch_label} ({patch_results[0]['patch_area_pct']:.1f}% area) + k167")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.suptitle("Collateral Suppression: Wearer vs Bystander (k=167 diagonal)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/collateral_suppression_map.png", dpi=150)
    plt.close()

    return {"results": results, "sweet_spots": sweet_spots,
            "baseline_wearers": len(wearers), "baseline_bystanders": len(bystanders)}


# ============================================================
# Experiment 2: Generate Print-Ready Patch
# ============================================================

def generate_print_patch(collateral_results):
    """
    Generate a print-ready 3600x4800 300dpi PNG of the optimal patch.
    Uses the best sweet spot parameters, or the best available combo.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Generate Print-Ready Patch")
    print(f"{'='*70}")

    # Select best parameters
    sweet = collateral_results.get("sweet_spots", [])
    all_results = collateral_results.get("results", [])

    if sweet:
        best = sweet[0]  # Already sorted by embedding corruption
        print(f"  Using sweet spot: {best['patch']} + {best['texture']} amp={best['amplitude']:.2f}")
    elif all_results:
        # Use the one with most wearer suppression and least collateral
        best = sorted(all_results, key=lambda x: (x["bystander_suppressed"], -x["wearer_suppression_rate"], -x["emb_l2_l105"]))[0]
        print(f"  Using best available: {best['patch']} + {best['texture']} amp={best['amplitude']:.2f}")
    else:
        # Fallback: medium patch + k167 at moderate amplitude
        best = {"patch": "medium_r80", "texture": "k167_d", "amplitude": 0.15,
                "patch_area_pct": 10.0, "patch_radius": 70}
        print(f"  Using fallback: medium_r80 + k167_d amp=0.15")

    # Parse patch size from label
    patch_label = best["patch"]
    if "tiny" in patch_label: base_rays = [35, 30, 40, 25, 38, 28, 42, 30, 35, 28, 40, 25]
    elif "small" in patch_label: base_rays = [55, 45, 60, 40, 58, 48, 62, 45, 55, 48, 60, 40]
    elif "medium" in patch_label: base_rays = [75, 60, 80, 55, 78, 65, 82, 60, 75, 65, 80, 55]
    elif "large" in patch_label: base_rays = [95, 75, 100, 70, 98, 80, 102, 75, 95, 80, 100, 70]
    elif "xlarge" in patch_label: base_rays = [115, 90, 120, 85, 118, 95, 122, 90, 115, 95, 120, 85]
    else: base_rays = [75, 60, 80, 55, 78, 65, 82, 60, 75, 65, 80, 55]

    tex_name = best["texture"]
    amp = best["amplitude"]

    # Generate at print resolution
    # The patch should be centered on the print image
    # Scale rays from 416-space to print-space
    # Print image is 3600x4800 (12x16 inches at 300dpi)
    # The patch occupies the center, scaled up from the 416px test space
    # Scale factor: print is ~8.6x larger than 416
    scale = PRINT_W_PX / IMG_SIZE  # ~8.65
    print_rays = [r * scale for r in base_rays]

    print_cx = PRINT_W_PX // 2
    print_cy = PRINT_H_PX // 2

    # Create mask at print resolution
    mask, endpoints = make_deformable_mask(PRINT_H_PX, PRINT_W_PX, print_cx, print_cy, print_rays, 12)
    print(f"  Print patch: {PRINT_W_PX}x{PRINT_H_PX}, rays={print_rays[:4]}..., area={np.mean(mask)*100:.1f}%")

    # Generate texture at print resolution
    # k=167 in 416 space → in print space, scale k proportionally
    # k_print = k_416 * (PRINT_W_PX / IMG_SIZE) to maintain same spatial frequency
    k_scaled = int(167 * scale)  # ~1445

    if tex_name == "k167_d":
        y, x = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")
        texture = (amp * np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y))).astype(np.float32)
    elif tex_name == "k167_square":
        y, x = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")
        texture = (amp * np.sign(np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y)))).astype(np.float32)
    elif tex_name == "k200_d":
        k_scaled = int(200 * scale)
        y, x = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")
        texture = (amp * np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y))).astype(np.float32)
    elif tex_name == "stripes_13px":
        stripe_period = int(13 * scale)
        texture = (amp * np.sign(np.sin(2*np.pi*np.arange(PRINT_W_PX)/stripe_period))).astype(np.float32)
        texture = texture * np.ones((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
    else:
        y, x = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")
        texture = (amp * np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y))).astype(np.float32)

    # Create the patch image: white background with texture inside mask
    # For printing on white paper/fabric, use 0.5 + texture (centered around gray)
    # The texture modulates around 0.5 (mid-gray) with amplitude
    patch_img = np.full((PRINT_H_PX, PRINT_W_PX, 3), 0.5, dtype=np.float32)  # gray background

    # Apply texture only inside mask
    for c in range(3):
        patch_img[:,:,c] = np.clip(0.5 + texture * mask, 0, 1)

    # Outside mask: white (for printing on white paper, then cut out)
    for c in range(3):
        patch_img[:,:,c] = patch_img[:,:,c] * mask + 1.0 * (1 - mask)

    # Convert to PIL and save
    patch_pil = Image.fromarray((patch_img * 255).astype(np.uint8))
    patch_pil.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))

    # Save print-ready version (with alpha for transparency)
    fname_alpha = f"stealthy_patch_{tex_name}_amp{amp:.2f}_{patch_label}_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath_alpha = os.path.join(os.path.dirname(OUTPUT_DIR), "..", fname_alpha)
    fpath_alpha = os.path.abspath(fpath_alpha)
    patch_pil.save(fpath_alpha, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"  Saved (alpha): {fpath_alpha}")

    # Also save without alpha (white background, for direct printing)
    patch_noalpha = Image.fromarray((patch_img * 255).astype(np.uint8))
    fname_noalpha = f"stealthy_patch_{tex_name}_amp{amp:.2f}_{patch_label}_{PRINT_W_PX}x{PRINT_H_PX}_300dpi_noalpha.png"
    fpath_noalpha = os.path.join(os.path.dirname(OUTPUT_DIR), "..", fname_noalpha)
    fpath_noalpha = os.path.abspath(fpath_noalpha)
    patch_noalpha.save(fpath_noalpha, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"  Saved (no alpha): {fpath_noalpha}")

    # Save mask outline only (for cutting guide)
    mask_outline = Image.new("RGB", (PRINT_W_PX, PRINT_H_PX), (255, 255, 255))
    draw = ImageDraw.Draw(mask_outline)
    # Draw polygon outline
    outline_points = [(int(p[0]), int(p[1])) for p in endpoints]
    draw.polygon(outline_points, outline=(255, 0, 0), fill=None)
    fname_outline = f"stealthy_patch_outline_{patch_label}_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath_outline = os.path.join(os.path.dirname(OUTPUT_DIR), "..", fname_outline)
    fpath_outline = os.path.abspath(fpath_outline)
    mask_outline.save(fpath_outline, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"  Saved (outline): {fpath_outline}")

    # Also generate a second variant: the digit pattern patch (1/196)
    # This is for the cloud poisoning scenario
    digits_196 = get_decimal_expansion(1, 196, 500)
    # Scale digit pattern to print resolution
    # Each digit maps to a column in 416 space → scale to print space
    digit_texture = np.zeros((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
    for x in range(PRINT_W_PX):
        # Map x back to 416 space to get the digit
        x_416 = int(x / scale) % len(digits_196)
        d = digits_196[x_416]
        digit_texture[:, x] = (d / 9.0) * amp

    digit_img = np.full((PRINT_H_PX, PRINT_W_PX, 3), 0.5, dtype=np.float32)
    for c in range(3):
        digit_img[:,:,c] = np.clip(0.5 + digit_texture * mask, 0, 1)
    for c in range(3):
        digit_img[:,:,c] = digit_img[:,:,c] * mask + 1.0 * (1 - mask)

    digit_pil = Image.fromarray((digit_img * 255).astype(np.uint8))
    digit_pil.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    fname_digit = f"stealthy_patch_digits196_amp{amp:.2f}_{patch_label}_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath_digit = os.path.join(os.path.dirname(OUTPUT_DIR), "..", fname_digit)
    fpath_digit = os.path.abspath(fpath_digit)
    digit_pil.save(fpath_digit, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"  Saved (digit 1/196): {fpath_digit}")

    # Generate a composite: k167 + digit pattern (dual-purpose)
    composite_texture = (texture * 0.6 + digit_texture * 0.4).astype(np.float32)
    comp_img = np.full((PRINT_H_PX, PRINT_W_PX, 3), 0.5, dtype=np.float32)
    for c in range(3):
        comp_img[:,:,c] = np.clip(0.5 + composite_texture * mask, 0, 1)
    for c in range(3):
        comp_img[:,:,c] = comp_img[:,:,c] * mask + 1.0 * (1 - mask)

    comp_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
    comp_pil.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    fname_comp = f"stealthy_patch_composite_k167d_digits196_amp{amp:.2f}_{patch_label}_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath_comp = os.path.join(os.path.dirname(OUTPUT_DIR), "..", fname_comp)
    fpath_comp = os.path.abspath(fpath_comp)
    comp_pil.save(fpath_comp, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"  Saved (composite): {fpath_comp}")

    return {
        "best_params": best,
        "print_paths": {
            "alpha": fpath_alpha,
            "noalpha": fpath_noalpha,
            "outline": fpath_outline,
            "digit196": fpath_digit,
            "composite": fpath_comp,
        },
        "print_size": f"{PRINT_W_IN}x{PRINT_H_IN} inches at {PRINT_DPI} DPI ({PRINT_W_PX}x{PRINT_H_PX}px)",
        "scale_factor": scale,
        "k_scaled": k_scaled,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("STEALTHY REGIME PROBE & PRINT-READY PATCH GENERATOR")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Print: {PRINT_W_IN}x{PRINT_H_IN} inches at {PRINT_DPI} DPI ({PRINT_W_PX}x{PRINT_H_PX}px)")

    arr_base = load_image(IMG_WITH, IMG_SIZE)

    # Load YOLOv3
    print("\nLoading YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters():
        p.requires_grad_(False)

    all_results = {}

    # Experiment 1: Collateral suppression analysis
    collateral = experiment_collateral(v3_model, arr_base)
    all_results["collateral"] = collateral

    # Experiment 2: Generate print-ready patch
    print_info = generate_print_patch(collateral)
    all_results["print_info"] = print_info

    del v3_model
    torch.cuda.empty_cache()

    # Save results
    print(f"\n{'='*70}")
    print("Saving results...")

    json_path = f"{OUTPUT_DIR}/stealthy_patch.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    csv_path = f"{OUTPUT_DIR}/stealthy_patch.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch", "texture", "amplitude", "patch_area_pct", "patch_radius",
                         "total_persons", "wearer_survived", "wearer_suppressed",
                         "bystander_survived", "bystander_suppressed",
                         "wearer_suppression_rate", "bystander_suppression_rate",
                         "emb_l2_l105", "emb_l2_l81", "emb_cos_l105"])
        for r in collateral["results"]:
            writer.writerow([r["patch"], r["texture"], r["amplitude"], r["patch_area_pct"],
                             r["patch_radius"], r["total_persons"], r["wearer_survived"],
                             r["wearer_suppressed"], r["bystander_survived"], r["bystander_suppressed"],
                             r["wearer_suppression_rate"], r["bystander_suppression_rate"],
                             r["emb_l2_l105"], r["emb_l2_l81"], r["emb_cos_l105"]])
    print(f"Saved CSV: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\nBaseline: {collateral['baseline_wearers']} wearers, {collateral['baseline_bystanders']} bystanders")

    print(f"\nSweet spots (wearer suppressed, 0 collateral, embedding corrupted):")
    sweet = collateral.get("sweet_spots", [])
    if sweet:
        for sp in sweet[:5]:
            print(f"  {sp['patch']:14s} {sp['texture']:14s} amp={sp['amplitude']:.2f}: "
                  f"wearer_supp={sp['wearer_suppressed']}, collateral={sp['bystander_suppressed']}, "
                  f"emb_L2={sp['emb_l2_l105']:.3f}")
    else:
        print("  None found with 0 collateral — showing best minimal-collateral:")
        all_r = collateral["results"]
        best = sorted(all_r, key=lambda x: (x["bystander_suppressed"], -x["wearer_suppression_rate"]))[:5]
        for sp in best:
            print(f"  {sp['patch']:14s} {sp['texture']:14s} amp={sp['amplitude']:.2f}: "
                  f"wearer_supp={sp['wearer_suppressed']}, collateral={sp['bystander_suppressed']}, "
                  f"emb_L2={sp['emb_l2_l105']:.3f}")

    print(f"\nPrint-ready patches generated:")
    for label, path in print_info["print_paths"].items():
        print(f"  {label:12s}: {path}")

    print(f"\nPrint instructions:")
    print(f"  Size: {PRINT_W_IN}x{PRINT_H_IN} inches at {PRINT_DPI} DPI")
    print(f"  Print on white paper or transfer paper for fabric")
    print(f"  Cut along the deformable polygon outline")
    print(f"  Place on torso center of clothing")
    print(f"  For alpha version: transparent outside patch shape")
    print(f"  For noalpha version: white outside patch (cut to shape)")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
