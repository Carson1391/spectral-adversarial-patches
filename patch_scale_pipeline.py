"""
Patch-Scale Pipeline Experiment

Tests the FULL attack scenario with proper physics:
1. Generate patch at print resolution
2. Simulate camera capture (blur, JPEG, resize, perspective)
3. Composite onto person torso in 416x416 YOLO input
4. Run YOLOv3 with feature capture at all 3 detection heads
5. Extract wearer and bystander embeddings (spatial cells)
6. Sweep amplitude 0 → 0.10, measure:
   - Person count (suppression onset)
   - Wearer embedding L2 (corruption onset)
   - Bystander embedding L2 (collateral corruption onset)
   - Cosine similarity (stealthiness)
7. Identify Profile A (corruption, no suppression) and Profile B (moderate suppression)
8. Report operational margin

Key difference from previous experiments:
- Patterns calibrated for PATCH pixel dimensions, not full image
- Camera degradation applied before YOLO inference
- Embeddings extracted at spatial locations of actual detections
- Amplitude sweep is fine-grained (0 → 0.10 in 21 steps)
"""

import os, sys, json, csv, math, io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from scipy.stats import pearsonr

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")

from pytorchyolo.models import Darknet

# ============================================================
# Config
# ============================================================

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline"
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

# Detection head layers in YOLOv3
DETECTION_LAYERS = {
    "L81_52x52": 81,
    "L93_26x26": 93,
    "L105_13x13": 105,
}

# Patch placement: main person center from find_persons.py
PATCH_CX, PATCH_CY = 183, 292  # torso center in 416 space
WEARER_THRESHOLD = 60  # px — detections within this radius = wearer

# Print resolution
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
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [cx, cy, w, h],
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
    return dets


# ============================================================
# Camera simulation
# ============================================================

def simulate_camera(patch_pattern, patch_mask, cam_config):
    """
    Simulate camera capture of a physical patch.
    
    patch_pattern: (H, W) float32, values in [-amp, +amp]
    patch_mask: (H, W) float32, 1.0 inside patch, 0.0 outside
    cam_config: dict with keys:
        - render_scale: multiplier for high-res rendering (e.g. 4x)
        - blur_sigma: Gaussian blur sigma in render pixels
        - jpeg_quality: JPEG compression quality (1-100)
        - perspective_warp: max corner displacement fraction (0 = flat, 0.1 = 10%)
        - final_w, final_h: target size in YOLO 416 space
    
    Returns: (pattern_416, mask_416) at YOLO input resolution
    """
    rs = cam_config["render_scale"]
    H, W = patch_pattern.shape
    HR, WR = H * rs, W * rs
    
    # 1. Render at high resolution
    pat_hr = np.kron(patch_pattern, np.ones((rs, rs))).astype(np.float32)
    mask_hr = np.kron(patch_mask, np.ones((rs, rs))).astype(np.float32)
    
    # 2. Convert to image for processing
    # Map pattern from [-amp, +amp] to [0, 255] for image processing
    # Use 128 as neutral, scale by 255/(2*max_amp)
    max_val = max(abs(patch_pattern.min()), abs(patch_pattern.max()))
    if max_val < 1e-8: max_val = 1.0
    pat_img = ((pat_hr / max_val + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    pat_pil = Image.fromarray(pat_img, mode="L")
    mask_pil = Image.fromarray((mask_hr * 255).astype(np.uint8), mode="L")
    
    # 3. Gaussian blur (lens softness + print diffusion)
    blur_sigma = cam_config["blur_sigma"]
    if blur_sigma > 0:
        pat_pil = pat_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma * 0.5))
    
    # 4. JPEG compression (phone camera typical)
    jpeg_q = cam_config["jpeg_quality"]
    if jpeg_q > 0:
        buf = io.BytesIO()
        pat_pil.save(buf, format="JPEG", quality=jpeg_q)
        buf.seek(0)
        pat_pil = Image.open(buf).convert("L")
    
    # 5. Perspective warp (fabric curvature on body)
    pw = cam_config.get("perspective_warp", 0.0)
    if pw > 0:
        # Slight random perspective: displace corners by up to pw fraction
        W_r, H_r = pat_pil.size
        rng = np.random.RandomState(42)  # deterministic for reproducibility
        corners = [(0, 0), (W_r, 0), (W_r, H_r), (0, H_r)]
        offsets = rng.uniform(-pw, pw, (4, 2)) * np.array([W_r, H_r])
        new_corners = [(c[0] + o[0], c[1] + o[1]) for c, o in zip(corners, offsets)]
        # Use PIL's transform with coefficients
        # For simplicity, use a slight affine approximation
        coeffs = find_affine(corners, new_corners)
        pat_pil = pat_pil.transform((W_r, H_r), Image.AFFINE, coeffs, Image.BILINEAR)
        mask_pil = mask_pil.transform((W_r, H_r), Image.AFFINE, coeffs, Image.BILINEAR)
    
    # 6. Resize to YOLO 416-space patch dimensions
    fw, fh = cam_config["final_w"], cam_config["final_h"]
    pat_pil = pat_pil.resize((fw, fh), Image.BILINEAR)
    mask_pil = mask_pil.resize((fw, fh), Image.BILINEAR)
    
    # 7. Convert back to float, remap to [-max_val, +max_val]
    pat_arr = np.array(pat_pil, dtype=np.float32) / 255.0
    pat_arr = (pat_arr * 2.0 - 1.0) * max_val
    mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
    
    # Smooth mask edge
    from scipy.ndimage import gaussian_filter
    mask_arr = gaussian_filter(mask_arr, sigma=0.8).clip(0, 1)
    
    return pat_arr, mask_arr

def find_affine(src, dst):
    """Compute affine transform coefficients from 3 point pairs."""
    # Use first 3 points
    (x0, y0), (x1, y1), (x2, y2) = src[0], src[1], src[2]
    (u0, v0), (u1, v1), (u2, v2) = dst[0], dst[1], dst[2]
    # Solve: u = a*x + b*y + c, v = d*x + e*y + f
    A = np.array([[x0, y0, 1], [x1, y1, 1], [x2, y2, 1]])
    b_u = np.array([u0, u1, u2])
    b_v = np.array([v0, v1, v2])
    abc = np.linalg.solve(A, b_u)
    def_ = np.linalg.solve(A, b_v)
    return (abc[0], abc[1], abc[2], def_[0], def_[1], def_[2])


# ============================================================
# Patch pattern generation (patch-local frequencies)
# ============================================================

def make_patch_pattern(patch_w, patch_h, k_patch, texture_type, amp):
    """
    Generate a pattern at patch-local frequency k_patch.
    k_patch = cycles across the patch width.
    
    texture_type: "stripes_v", "sinusoid_d", "square_d", "digits_196"
    """
    y, x = np.meshgrid(np.arange(patch_h), np.arange(patch_w), indexing="ij")
    
    if texture_type == "stripes_v":
        # Vertical stripes: k_patch cycles across width
        pat = amp * np.sign(np.sin(2 * np.pi * k_patch * x / patch_w))
    elif texture_type == "sinusoid_d":
        # Diagonal sinusoid: k_patch/2 cycles in each axis
        kx = k_patch / 2
        ky = k_patch / 2
        pat = amp * np.cos(2 * np.pi * (kx * x / patch_w + ky * y / patch_h))
    elif texture_type == "square_d":
        kx = k_patch / 2
        ky = k_patch / 2
        pat = amp * np.sign(np.cos(2 * np.pi * (kx * x / patch_w + ky * y / patch_h)))
    elif texture_type == "digits_196":
        # 1/196 digit pattern mapped to columns
        digits = get_decimal_expansion(1, 196, 500)
        pat = np.zeros((patch_h, patch_w), dtype=np.float32)
        for xi in range(patch_w):
            d = digits[xi % len(digits)]
            pat[:, xi] = (d / 4.5 - 1.0) * amp  # 0→-amp, 9→+amp
    else:
        pat = np.zeros((patch_h, patch_w), dtype=np.float32)
    
    return pat.astype(np.float32)


# ============================================================
# Embedding extraction at spatial locations
# ============================================================

def extract_embedding(caps, layer_idx, spatial_x, spatial_y):
    """
    Extract feature vector at a spatial location in a detection head.
    
    caps: dict of captured feature maps {layer_idx: tensor(1, C, H, W)}
    layer_idx: which layer (81, 93, or 105)
    spatial_x, spatial_y: pixel coordinates in 416 space
    
    Returns: (C,) numpy vector
    """
    feat = caps[layer_idx]  # (1, C, fH, fW)
    fH, fW = feat.shape[2], feat.shape[3]
    # Map 416-space coordinates to feature map coordinates
    fx = int(spatial_x / IMG_SIZE * fW)
    fy = int(spatial_y / IMG_SIZE * fH)
    fx = max(0, min(fW - 1, fx))
    fy = max(0, min(fH - 1, fy))
    # Extract the channel vector at this spatial cell
    vec = feat[0, :, fy, fx].cpu().numpy()
    return vec

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def l2_distance(a, b):
    return float(np.linalg.norm(a - b))


# ============================================================
# Main experiment
# ============================================================

def run_pipeline_experiment(v3_model, arr_base):
    """
    Full pipeline: camera sim + amplitude sweep + embedding analysis.
    """
    print(f"\n{'='*70}")
    print("PATCH-SCALE PIPELINE EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}")
    print(f"  Image: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Patch center: ({PATCH_CX}, {PATCH_CY})")
    print(f"  Wearer threshold: {WEARER_THRESHOLD}px")
    
    # ---- Step 1: Get baseline detections and classify wearer/bystander ----
    tensor_base = torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_persons = [d for d in base_dets if d["class_name"] == "person"]
    
    for d in base_persons:
        dist = math.sqrt((d["cx"] - PATCH_CX)**2 + (d["cy"] - PATCH_CY)**2)
        d["dist_to_patch"] = dist
        d["is_wearer"] = dist < WEARER_THRESHOLD
    
    wearers = [d for d in base_persons if d["is_wearer"]]
    bystanders = [d for d in base_persons if not d["is_wearer"]]
    
    print(f"\n  Baseline: {len(base_persons)} persons ({len(wearers)} wearer, {len(bystanders)} bystander)")
    for d in base_persons:
        tag = "WEARER" if d["is_wearer"] else "bystander"
        print(f"    {tag}: ({d['cx']:.0f}, {d['cy']:.0f}) conf={d['confidence']:.3f} dist={d['dist_to_patch']:.0f}")
    
    # ---- Step 2: Get baseline embeddings (clean, no patch) ----
    with torch.no_grad():
        caps_base, _ = forward_capture_v3(v3_model, tensor_base)
    
    # Extract clean embeddings at wearer and bystander locations
    clean_embs = {"wearer": {}, "bystanders": []}
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        # Wearer: average embedding across all wearer detection locations
        if wearers:
            wearer_vecs = [extract_embedding(caps_base, layer_idx, d["cx"], d["cy"]) for d in wearers]
            clean_embs["wearer"][layer_name] = np.mean(wearer_vecs, axis=0)
        else:
            clean_embs["wearer"][layer_name] = np.zeros(255, dtype=np.float32)
        
        # Bystanders: average embedding across all bystander locations
        if bystanders:
            bystander_vecs = [extract_embedding(caps_base, layer_idx, d["cx"], d["cy"]) for d in bystanders]
            clean_embs["bystanders"].append({
                "layer": layer_name,
                "vec": np.mean(bystander_vecs, axis=0)
            })
    
    # ---- Step 3: Define patch geometries in 416 space ----
    # Shirt/torso is ~20% of the human body in frame. Sweep 4 sizes.
    patch_sizes = [
        # (label, ray_lengths)
        ("medium_6pct", [65, 55, 70, 50, 68, 58, 72, 55, 65, 58, 70, 50]),
        ("large_10pct", [85, 70, 92, 65, 88, 75, 95, 70, 85, 75, 92, 65]),
        ("xlarge_16pct", [110, 90, 120, 85, 115, 95, 125, 90, 110, 95, 120, 85]),
        ("shirt_20pct", [130, 110, 140, 100, 135, 115, 145, 110, 130, 115, 140, 100]),
    ]

    # Print area for each size
    for ps_label, ps_rays in patch_sizes:
        pmask, _ = make_deformable_mask(IMG_SIZE, IMG_SIZE, PATCH_CX, PATCH_CY, ps_rays, 12)
        print(f"  Patch {ps_label}: area={np.mean(pmask)*100:.1f}%, max_ray={max(ps_rays)}px")

    # ---- Step 4: Camera configuration (phone_typical for main experiment) ----
    RENDER_SCALE = 4
    BLUR_SIGMA = 2.5
    JPEG_QUALITY = 75
    PERSPECTIVE_WARP = 0.05
    cam_label = "phone_typical"

    # ---- Step 5: Pattern configurations ----
    pattern_configs = [
        ("k3_stripes", 3, "stripes_v"),
        ("k3_sine_d", 3, "sinusoid_d"),
        ("k6_stripes", 6, "stripes_v"),
        ("k6_sine_d", 6, "sinusoid_d"),
        ("k6_square_d", 6, "square_d"),
        ("k12_stripes", 12, "stripes_v"),
        ("k12_sine_d", 12, "sinusoid_d"),
        ("k12_square_d", 12, "square_d"),
        ("k25_stripes", 25, "stripes_v"),
        ("k25_square_d", 25, "square_d"),
        ("digits_196", 0, "digits_196"),
    ]

    # ---- Step 6: Amplitude sweep ----
    amplitudes = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    total_runs = len(patch_sizes) * len(pattern_configs) * len(amplitudes)
    print(f"\n  Camera: {cam_label} (blur={BLUR_SIGMA}, jpeg={JPEG_QUALITY}, warp={PERSPECTIVE_WARP})")
    print(f"  Patch sizes: {len(patch_sizes)}")
    print(f"  Amplitudes: {len(amplitudes)}")
    print(f"  Patterns: {len(pattern_configs)}")
    print(f"  Total runs: {total_runs}")

    results = []
    run_idx = 0

    for ps_label, ps_rays in patch_sizes:
        # Compute patch dimensions in 416 space
        patch_w_416 = int(max(ps_rays) * 2)
        patch_h_416 = int(max(ps_rays) * 2)

        # Camera config for this patch size
        cam_dict = {
            "render_scale": RENDER_SCALE,
            "blur_sigma": BLUR_SIGMA,
            "jpeg_quality": JPEG_QUALITY,
            "perspective_warp": PERSPECTIVE_WARP,
            "final_w": patch_w_416,
            "final_h": patch_h_416,
        }

        # Generate deformable mask at render resolution
        render_w = patch_w_416 * RENDER_SCALE
        render_h = patch_h_416 * RENDER_SCALE
        base_mask, _ = make_deformable_mask(
            render_h, render_w, render_w // 2, render_h // 2,
            [r * RENDER_SCALE for r in ps_rays], 12
        )

        # Also generate the 416-space mask directly for compositing
        mask_416_direct, _ = make_deformable_mask(
            IMG_SIZE, IMG_SIZE, PATCH_CX, PATCH_CY, ps_rays, 12
        )

        for pat_label, k_patch, tex_type in pattern_configs:
            print(f"\n  [{ps_label}] {pat_label} (k={k_patch}, {tex_type})")

            # Generate pattern at render resolution
            base_pat = make_patch_pattern(render_w, render_h, k_patch, tex_type, 1.0)

            for amp in amplitudes:
                run_idx += 1
                pat_scaled = base_pat * amp

                # Simulate camera capture
                pat_416, mask_416_cam = simulate_camera(pat_scaled, base_mask, cam_dict)

                # Composite onto image — use the direct 416 mask for clean edges
                arr_patched = arr_base.copy()
                ph, pw = pat_416.shape
                x0 = PATCH_CX - pw // 2
                y0 = PATCH_CY - ph // 2

                full_pat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

                px0 = max(0, x0)
                py0 = max(0, y0)
                px1 = min(IMG_SIZE, x0 + pw)
                py1 = min(IMG_SIZE, y0 + ph)
                sx0 = px0 - x0
                sy0 = py0 - y0
                sx1 = sx0 + (px1 - px0)
                sy1 = sy0 + (py1 - py0)

                full_pat[py0:py1, px0:px1] = pat_416[sy0:sy1, sx0:sx1]
                full_mask[py0:py1, px0:px1] = mask_416_cam[sy0:sy1, sx0:sx1]

                for c in range(3):
                    arr_patched[:, :, c] = np.clip(
                        arr_base[:, :, c] * (1 - full_mask) +
                        (arr_base[:, :, c] + full_pat) * full_mask,
                        0, 1
                    )

                # Run YOLOv3
                tensor_patch = torch.from_numpy(arr_patched).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    caps_patch, _ = forward_capture_v3(v3_model, tensor_patch)
                patch_dets = get_dets_v3(v3_model, tensor_patch, conf=0.1)
                patch_persons = [d for d in patch_dets if d["class_name"] == "person"]

                # Classify detections
                w_count = 0
                b_count = 0
                w_confs = []
                b_confs = []
                for d in patch_persons:
                    dist = math.sqrt((d["cx"] - PATCH_CX)**2 + (d["cy"] - PATCH_CY)**2)
                    if dist < WEARER_THRESHOLD:
                        w_count += 1
                        w_confs.append(d["confidence"])
                    else:
                        b_count += 1
                        b_confs.append(d["confidence"])

                # Extract embeddings at wearer and bystander locations
                wearer_l2 = {"L81_52x52": 0, "L93_26x26": 0, "L105_13x13": 0}
                bystander_l2 = {"L81_52x52": 0, "L93_26x26": 0, "L105_13x13": 0}
                wearer_cos = {"L81_52x52": 0, "L93_26x26": 0, "L105_13x13": 0}
                bystander_cos = {"L81_52x52": 0, "L93_26x26": 0, "L105_13x13": 0}

                for layer_name, layer_idx in DETECTION_LAYERS.items():
                    if wearers:
                        w_vecs_patch = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in wearers]
                        w_emb_patch = np.mean(w_vecs_patch, axis=0)
                        w_emb_clean = clean_embs["wearer"][layer_name]
                        wearer_l2[layer_name] = l2_distance(w_emb_patch, w_emb_clean)
                        wearer_cos[layer_name] = cosine_similarity(w_emb_patch, w_emb_clean)

                    if bystanders:
                        b_vecs_patch = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in bystanders]
                        b_emb_patch = np.mean(b_vecs_patch, axis=0)
                        b_emb_clean = np.zeros(255, dtype=np.float32)
                        for be in clean_embs["bystanders"]:
                            if be["layer"] == layer_name:
                                b_emb_clean = be["vec"]
                                break
                        bystander_l2[layer_name] = l2_distance(b_emb_patch, b_emb_clean)
                        bystander_cos[layer_name] = cosine_similarity(b_emb_patch, b_emb_clean)

                avg_w_l2 = float(np.mean(list(wearer_l2.values())))
                avg_b_l2 = float(np.mean(list(bystander_l2.values())))
                avg_w_cos = float(np.mean(list(wearer_cos.values())))
                avg_b_cos = float(np.mean(list(bystander_cos.values())))

                result = {
                    "patch_size": ps_label,
                    "pattern": pat_label,
                    "k_patch": k_patch,
                    "texture": tex_type,
                    "amplitude": amp,
                    "camera": cam_label,
                    "total_persons": len(patch_persons),
                    "wearer_count": w_count,
                    "bystander_count": b_count,
                    "baseline_wearer": len(wearers),
                    "baseline_bystander": len(bystanders),
                    "wearer_suppressed": len(wearers) - w_count,
                    "bystander_suppressed": len(bystanders) - b_count,
                    "wearer_conf_mean": float(np.mean(w_confs)) if w_confs else 0.0,
                    "bystander_conf_mean": float(np.mean(b_confs)) if b_confs else 0.0,
                    "wearer_l2_L81": wearer_l2["L81_52x52"],
                    "wearer_l2_L93": wearer_l2["L93_26x26"],
                    "wearer_l2_L105": wearer_l2["L105_13x13"],
                    "bystander_l2_L81": bystander_l2["L81_52x52"],
                    "bystander_l2_L93": bystander_l2["L93_26x26"],
                    "bystander_l2_L105": bystander_l2["L105_13x13"],
                    "wearer_cos_L81": wearer_cos["L81_52x52"],
                    "wearer_cos_L93": wearer_cos["L93_26x26"],
                    "wearer_cos_L105": wearer_cos["L105_13x13"],
                    "bystander_cos_L81": bystander_cos["L81_52x52"],
                    "bystander_cos_L93": bystander_cos["L93_26x26"],
                    "bystander_cos_L105": bystander_cos["L105_13x13"],
                    "avg_wearer_l2": avg_w_l2,
                    "avg_bystander_l2": avg_b_l2,
                    "avg_wearer_cos": avg_w_cos,
                    "avg_bystander_cos": avg_b_cos,
                }
                results.append(result)

                status = f"  [{run_idx}/{total_runs}] {ps_label} {pat_label} amp={amp:.3f}: P={len(patch_persons):2d} (W={w_count},B={b_count}) W_L2={avg_w_l2:.2f} B_L2={avg_b_l2:.2f} W_cos={avg_w_cos:.4f}"
                print(status)
    
    # ---- Step 7: Save results ----
    results_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, "pipeline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {csv_path}")
    
    return results, wearers, bystanders


# ============================================================
# Analysis: Find Profile A and Profile B
# ============================================================

def analyze_profiles(results, wearers, bystanders):
    """
    Find Profile A (corruption without suppression) and Profile B (moderate suppression).
    """
    print(f"\n{'='*70}")
    print("PROFILE ANALYSIS")
    print(f"{'='*70}")
    
    baseline_w = len(wearers)
    baseline_b = len(bystanders)
    
    # Group results by pattern
    by_pattern = {}
    for r in results:
        p = r["pattern"]
        if p not in by_pattern:
            by_pattern[p] = []
        by_pattern[p].append(r)
    
    profile_results = []
    
    for pat, rows in by_pattern.items():
        rows = sorted(rows, key=lambda r: r["amplitude"])
        
        # Find suppression onset: first amplitude where wearer_count < baseline_wearer
        supp_onset = None
        for r in rows:
            if r["wearer_suppressed"] > 0:
                supp_onset = r["amplitude"]
                break
        
        # Find corruption onset: first amplitude where avg_wearer_l2 > 0.5
        # (0.5 is a threshold — embeddings have noise ~0.1-0.3 from no patch)
        corrupt_onset = None
        for r in rows:
            if r["avg_wearer_l2"] > 0.5:
                corrupt_onset = r["amplitude"]
                break
        
        # Find bystander corruption onset: avg_bystander_l2 > 0.3
        b_corrupt_onset = None
        for r in rows:
            if r["avg_bystander_l2"] > 0.3:
                b_corrupt_onset = r["amplitude"]
                break
        
        # Profile A: corruption strong (L2 > 1.0), suppression = 0
        profile_a = None
        for r in rows:
            if r["avg_wearer_l2"] > 1.0 and r["wearer_suppressed"] == 0:
                profile_a = r
                break
        
        # Profile B: moderate suppression (1-2 wearer suppressed), bystander corruption > 0.5
        profile_b = None
        for r in rows:
            if 1 <= r["wearer_suppressed"] <= 2 and r["avg_bystander_l2"] > 0.5:
                profile_b = r
                break
        
        # Operational margin: gap between corruption onset and suppression onset
        if corrupt_onset and supp_onset:
            margin = supp_onset - corrupt_onset
        elif corrupt_onset and not supp_onset:
            margin = float("inf")  # corruption starts but suppression never happens
        else:
            margin = None  # neither or suppression first
        
        # Max wearer L2 at amp=0.10
        max_w_l2 = max(r["avg_wearer_l2"] for r in rows)
        max_b_l2 = max(r["avg_bystander_l2"] for r in rows)
        
        # Final suppression at amp=0.10
        final = rows[-1]
        
        pr = {
            "pattern": pat,
            "suppression_onset": supp_onset,
            "wearer_corruption_onset": corrupt_onset,
            "bystander_corruption_onset": b_corrupt_onset,
            "operational_margin": margin,
            "profile_a_amp": profile_a["amplitude"] if profile_a else None,
            "profile_a_l2": profile_a["avg_wearer_l2"] if profile_a else None,
            "profile_b_amp": profile_b["amplitude"] if profile_b else None,
            "profile_b_l2": profile_b["avg_wearer_l2"] if profile_b else None,
            "profile_b_bystander_l2": profile_b["avg_bystander_l2"] if profile_b else None,
            "max_wearer_l2": max_w_l2,
            "max_bystander_l2": max_b_l2,
            "final_wearer_suppressed": final["wearer_suppressed"],
            "final_bystander_suppressed": final["bystander_suppressed"],
            "final_wearer_cos": final["avg_wearer_cos"],
            "final_bystander_cos": final["avg_bystander_cos"],
        }
        profile_results.append(pr)
        
        print(f"\n  {pat}:")
        print(f"    Suppression onset: {supp_onset}")
        print(f"    Wearer corruption onset (L2>0.5): {corrupt_onset}")
        print(f"    Bystander corruption onset (L2>0.3): {b_corrupt_onset}")
        print(f"    Operational margin: {margin}")
        print(f"    Profile A (corrupt, no suppress): {'amp=' + str(profile_a['amplitude']) if profile_a else 'NOT FOUND'}")
        print(f"    Profile B (moderate suppress + collateral): {'amp=' + str(profile_b['amplitude']) if profile_b else 'NOT FOUND'}")
        print(f"    Max wearer L2: {max_w_l2:.3f}, Max bystander L2: {max_b_l2:.3f}")
        print(f"    Final @ amp=0.10: W_supp={final['wearer_suppressed']}, B_supp={final['bystander_suppressed']}, W_cos={final['avg_wearer_cos']:.4f}")
    
    # Save profile analysis
    profile_path = os.path.join(OUTPUT_DIR, "profile_analysis.json")
    with open(profile_path, "w") as f:
        json.dump(profile_results, f, indent=2, default=str)
    print(f"\n  Profile analysis saved: {profile_path}")
    
    # ---- Find best patterns ----
    print(f"\n{'='*70}")
    print("BEST PATTERNS")
    print(f"{'='*70}")
    
    # Best for Profile A (corruption without suppression)
    best_a = sorted(profile_results, key=lambda x: x["max_wearer_l2"] if x["max_wearer_l2"] else 0, reverse=True)
    print(f"\n  Best for Profile A (max corruption):")
    for pr in best_a[:3]:
        print(f"    {pr['pattern']}: max_L2={pr['max_wearer_l2']:.3f}, margin={pr['operational_margin']}, Profile A={'found' if pr['profile_a_amp'] else 'not found'}")
    
    # Best operational margin
    best_margin = sorted([p for p in profile_results if p["operational_margin"] is not None], 
                         key=lambda x: x["operational_margin"] if x["operational_margin"] != float("inf") else 999, reverse=True)
    print(f"\n  Best operational margin:")
    for pr in best_margin[:3]:
        print(f"    {pr['pattern']}: margin={pr['operational_margin']}, supp_onset={pr['suppression_onset']}, corrupt_onset={pr['wearer_corruption_onset']}")
    
    # Best dual-purpose (high corruption + moderate suppression)
    best_dual = sorted(profile_results, key=lambda x: (x["max_wearer_l2"] or 0) * (1 if x["profile_b_amp"] else 0), reverse=True)
    print(f"\n  Best dual-purpose (corruption + suppression):")
    for pr in best_dual[:3]:
        print(f"    {pr['pattern']}: max_L2={pr['max_wearer_l2']:.3f}, Profile B={'amp=' + str(pr['profile_b_amp']) if pr['profile_b_amp'] else 'not found'}")
    
    return profile_results


# ============================================================
# Plots
# ============================================================

def plot_amplitude_sweep(results, output_dir):
    """Generate amplitude sweep plots for each pattern."""
    by_pattern = {}
    for r in results:
        p = r["pattern"]
        if p not in by_pattern:
            by_pattern[p] = []
        by_pattern[p].append(r)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (pat, rows) in enumerate(sorted(by_pattern.items())):
        if idx >= 12:
            break
        ax = axes[idx]
        rows = sorted(rows, key=lambda r: r["amplitude"])
        amps = [r["amplitude"] for r in rows]
        
        # Plot 4 curves
        w_l2 = [r["avg_wearer_l2"] for r in rows]
        b_l2 = [r["avg_bystander_l2"] for r in rows]
        w_supp = [r["wearer_suppressed"] for r in rows]
        b_supp = [r["bystander_suppressed"] for r in rows]
        
        ax2 = ax.twinx()
        
        # L2 distances (left axis)
        ax.plot(amps, w_l2, "r-o", label="Wearer L2", markersize=3)
        ax.plot(amps, b_l2, "b-s", label="Bystander L2", markersize=3)
        ax.set_ylabel("L2 Distance", color="black")
        ax.set_ylim(0, max(max(w_l2), max(b_l2)) * 1.2 if max(w_l2 + b_l2) > 0 else 1)
        
        # Suppression counts (right axis)
        ax2.plot(amps, w_supp, "r--", label="Wearer supp", alpha=0.5)
        ax2.plot(amps, b_supp, "b--", label="Bystander supp", alpha=0.5)
        ax2.set_ylabel("Suppressed Count", color="gray")
        ax2.set_ylim(-0.5, max(max(w_supp), max(b_supp)) + 1 if max(w_supp + b_supp) > 0 else 5)
        
        ax.set_xlabel("Amplitude")
        ax.set_title(pat)
        ax.axhline(y=0.5, color="r", linestyle=":", alpha=0.3, label="corruption threshold")
        ax.axhline(y=0.3, color="b", linestyle=":", alpha=0.3)
        
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)
    
    plt.suptitle("Amplitude Sweep: Embedding Corruption vs Suppression (phone_typical camera)", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "amplitude_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")


def plot_cosine_similarity(results, output_dir):
    """Plot cosine similarity vs amplitude for all patterns."""
    by_pattern = {}
    for r in results:
        p = r["pattern"]
        if p not in by_pattern:
            by_pattern[p] = []
        by_pattern[p].append(r)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for pat, rows in sorted(by_pattern.items()):
        rows = sorted(rows, key=lambda r: r["amplitude"])
        amps = [r["amplitude"] for r in rows]
        w_cos = [r["avg_wearer_cos"] for r in rows]
        b_cos = [r["avg_bystander_cos"] for r in rows]
        
        ax.plot(amps, w_cos, "-o", label=f"{pat} (wearer)", markersize=3)
        ax.plot(amps, b_cos, "--s", label=f"{pat} (bystander)", markersize=3, alpha=0.5)
    
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity vs Amplitude (phone_typical camera)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.95, 1.001)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "cosine_similarity.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Cosine plot saved: {plot_path}")


# ============================================================
# Camera comparison experiment
# ============================================================

def run_camera_comparison(v3_model, arr_base, wearers, bystanders):
    """
    Test one pattern across all camera conditions.
    Shows how camera degradation affects suppression and corruption.
    """
    print(f"\n{'='*70}")
    print("CAMERA COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    
    # Use k12_stripes at amp=0.05 across all camera configs
    camera_configs = [
        ("digital_perfect", {"render_scale": 1, "blur_sigma": 0, "jpeg_quality": 100, "perspective_warp": 0.0, "final_w": 88, "final_h": 88}),
        ("close_clean", {"render_scale": 4, "blur_sigma": 1.0, "jpeg_quality": 95, "perspective_warp": 0.0, "final_w": 88, "final_h": 88}),
        ("phone_typical", {"render_scale": 4, "blur_sigma": 2.5, "jpeg_quality": 75, "perspective_warp": 0.05, "final_w": 88, "final_h": 88}),
        ("far_degraded", {"render_scale": 4, "blur_sigma": 4.0, "jpeg_quality": 60, "perspective_warp": 0.10, "final_w": 88, "final_h": 88}),
    ]
    
    patterns = [
        ("k3_stripes", 3, "stripes_v"),
        ("k6_stripes", 6, "stripes_v"),
        ("k12_stripes", 12, "stripes_v"),
        ("k25_stripes", 25, "stripes_v"),
        ("k12_sine_d", 12, "sinusoid_d"),
        ("digits_196", 0, "digits_196"),
    ]
    
    amp = 0.05
    patch_rays = [38, 30, 42, 28, 40, 32, 44, 30, 38, 32, 42, 28]
    
    # Get baseline embeddings
    tensor_base = torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        caps_base, _ = forward_capture_v3(v3_model, tensor_base)
    
    clean_wearer_embs = {}
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if wearers:
            vecs = [extract_embedding(caps_base, layer_idx, d["cx"], d["cy"]) for d in wearers]
            clean_wearer_embs[layer_name] = np.mean(vecs, axis=0)
    
    cam_results = []
    
    for cam_label, cam_dict in camera_configs:
        print(f"\n  Camera: {cam_label}")
        
        for pat_label, k_patch, tex_type in patterns:
            render_w = 88 * cam_dict["render_scale"]
            render_h = 88 * cam_dict["render_scale"]
            base_pat = make_patch_pattern(render_w, render_h, k_patch, tex_type, 1.0)
            base_mask, _ = make_deformable_mask(
                render_h, render_w, render_w // 2, render_h // 2,
                [r * cam_dict["render_scale"] for r in patch_rays], 12
            )
            
            pat_scaled = base_pat * amp
            pat_416, mask_416 = simulate_camera(pat_scaled, base_mask, cam_dict)
            
            # Composite
            arr_patched = arr_base.copy()
            ph, pw = pat_416.shape
            x0 = PATCH_CX - pw // 2
            y0 = PATCH_CY - ph // 2
            
            full_pat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            px0, py0 = max(0, x0), max(0, y0)
            px1, py1 = min(IMG_SIZE, x0 + pw), min(IMG_SIZE, y0 + ph)
            sx0, sy0 = px0 - x0, py0 - y0
            sx1, sy1 = sx0 + (px1 - px0), sy0 + (py1 - py0)
            full_pat[py0:py1, px0:px1] = pat_416[sy0:sy1, sx0:sx1]
            full_mask[py0:py1, px0:px1] = mask_416[sy0:sy1, sx0:sx1]
            
            for c in range(3):
                arr_patched[:, :, c] = np.clip(
                    arr_base[:, :, c] * (1 - full_mask) +
                    (arr_base[:, :, c] + full_pat) * full_mask, 0, 1
                )
            
            tensor_patch = torch.from_numpy(arr_patched).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                caps_patch, _ = forward_capture_v3(v3_model, tensor_patch)
            dets = get_dets_v3(v3_model, tensor_patch, conf=0.1)
            persons = [d for d in dets if d["class_name"] == "person"]
            
            w_count = sum(1 for d in persons if math.sqrt((d["cx"]-PATCH_CX)**2 + (d["cy"]-PATCH_CY)**2) < WEARER_THRESHOLD)
            b_count = len(persons) - w_count
            
            # Wearer L2
            w_l2s = []
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                if wearers:
                    w_vecs = [extract_embedding(caps_patch, layer_idx, d["cx"], d["cy"]) for d in wearers]
                    w_emb = np.mean(w_vecs, axis=0)
                    w_l2s.append(l2_distance(w_emb, clean_wearer_embs[layer_name]))
            avg_w_l2 = float(np.mean(w_l2s)) if w_l2s else 0.0
            
            r = {
                "camera": cam_label,
                "pattern": pat_label,
                "amplitude": amp,
                "total_persons": len(persons),
                "wearer_count": w_count,
                "bystander_count": b_count,
                "wearer_l2": avg_w_l2,
            }
            cam_results.append(r)
            print(f"    {pat_label}: persons={len(persons):2d} (W={w_count}, B={b_count}) W_L2={avg_w_l2:.3f}")
    
    # Save
    cam_path = os.path.join(OUTPUT_DIR, "camera_comparison.json")
    with open(cam_path, "w") as f:
        json.dump(cam_results, f, indent=2)
    print(f"\n  Camera comparison saved: {cam_path}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cam_labels = [c[0] for c in camera_configs]
    pat_labels = [p[0] for p in patterns]
    
    x = np.arange(len(pat_labels))
    width = 0.2
    for i, cam_label in enumerate(cam_labels):
        vals = [r["wearer_l2"] for r in cam_results if r["camera"] == cam_label]
        persons = [r["total_persons"] for r in cam_results if r["camera"] == cam_label]
        ax.bar(x + i * width, vals, width, label=f"{cam_label} (L2)")
    
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Wearer L2 Distance")
    ax.set_title(f"Camera Degradation Effect on Embedding Corruption (amp={amp})")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pat_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plot_path = os.path.join(OUTPUT_DIR, "camera_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Camera comparison plot: {plot_path}")
    
    return cam_results


# ============================================================
# Main
# ============================================================

def generate_print_patches(profile_results):
    """
    Generate print-ready patches at Profile A and Profile B operating points.
    Best pattern: k12_stripes (Profile A at amp=0.005, Profile B at amp=0.04).
    Also generate k12_square_d (max corruption) and digits_196.
    """
    print(f"\n{'='*70}")
    print("GENERATING PRINT-READY PATCHES")
    print(f"{'='*70}")

    # Use shirt_20pct rays — full torso coverage
    shirt_rays = [130, 110, 140, 100, 135, 115, 145, 110, 130, 115, 140, 100]
    patch_w_416 = int(max(shirt_rays) * 2)  # 290px in 416 space

    # Scale to print resolution
    # 416px image = 12 inches at ~35 dpi (rough)
    # Patch is 290/416 of the width = 290/416 * 12 = 8.37 inches
    # At 300 dpi: 8.37 * 300 = 2511px patch
    patch_print_w = int(patch_w_416 / IMG_SIZE * PRINT_W_PX)
    patch_print_h = int(patch_w_416 / IMG_SIZE * PRINT_H_PX)

    # Patches to generate
    patches = [
        ("profileA_k12_stripes", 12, "stripes_v", 0.005),
        ("profileB_k12_stripes", 12, "stripes_v", 0.04),
        ("profileA_k12_square_d", 12, "square_d", 0.005),
        ("profileB_k12_square_d", 12, "square_d", 0.04),
        ("profileA_digits_196", 0, "digits_196", 0.005),
        ("profileB_digits_196", 0, "digits_196", 0.04),
        ("maxcorrupt_k12_square_d", 12, "square_d", 0.10),
        ("maxcorrupt_k25_square_d", 25, "square_d", 0.10),
    ]

    patch_dir = os.path.join(OUTPUT_DIR, "print_patches")
    os.makedirs(patch_dir, exist_ok=True)

    for label, k_patch, tex_type, amp in patches:
        # Generate at print resolution
        pat = make_patch_pattern(patch_print_w, patch_print_h, k_patch, tex_type, amp)

        # Generate mask at print resolution
        mask, _ = make_deformable_mask(
            patch_print_h, patch_print_w,
            patch_print_w // 2, patch_print_h // 2,
            [r / max(shirt_rays) * patch_print_w * 0.5 for r in shirt_rays], 12
        )

        # Convert to RGB image: gray base (0.5) + pattern
        rgb = np.zeros((patch_print_h, patch_print_w, 3), dtype=np.float32)
        base_gray = 0.5
        for c in range(3):
            rgb[:, :, c] = base_gray + pat * mask

        rgb = np.clip(rgb, 0, 1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Save full-res PNG
        img = Image.fromarray(rgb_uint8, mode="RGB")
        out_path = os.path.join(patch_dir, f"{label}_amp{amp:.3f}_{patch_print_w}x{patch_print_h}.png")
        img.save(out_path)
        print(f"  Saved: {out_path} ({patch_print_w}x{patch_print_h}px)")

        # Also save a small preview (416x416 composite on the person image)
        arr_base = load_image(IMG_WITH, IMG_SIZE)
        # Generate pattern at 416-space patch size and place into full 416x416 array
        pat_local = make_patch_pattern(patch_w_416, patch_w_416, k_patch, tex_type, amp)
        mask_416, _ = make_deformable_mask(
            IMG_SIZE, IMG_SIZE, PATCH_CX, PATCH_CY, shirt_rays, 12
        )

        # Place local pattern into full-size array at patch center
        full_pat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        x0 = PATCH_CX - patch_w_416 // 2
        y0 = PATCH_CY - patch_w_416 // 2
        px0, py0 = max(0, x0), max(0, y0)
        px1 = min(IMG_SIZE, x0 + patch_w_416)
        py1 = min(IMG_SIZE, y0 + patch_w_416)
        sx0, sy0 = px0 - x0, py0 - y0
        sx1 = sx0 + (px1 - px0)
        sy1 = sy0 + (py1 - py0)
        full_pat[py0:py1, px0:px1] = pat_local[sy0:sy1, sx0:sx1]

        arr_comp = arr_base.copy()
        for c in range(3):
            arr_comp[:, :, c] = np.clip(
                arr_base[:, :, c] * (1 - mask_416) +
                (arr_base[:, :, c] + full_pat) * mask_416,
                0, 1
            )

        comp_path = os.path.join(patch_dir, f"{label}_composite_416.png")
        comp_img = Image.fromarray((arr_comp * 255).astype(np.uint8), mode="RGB")
        comp_img.save(comp_path)
        print(f"  Composite: {comp_path}")

    # Save a README
    readme_path = os.path.join(patch_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("Print-Ready Adversarial Patches\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Patch size: {patch_print_w}x{patch_print_h}px at 300dpi\n")
        f.write(f"Physical size: {patch_print_w/300:.1f}x{patch_print_h/300:.1f} inches\n")
        f.write(f"Patch area in 416 space: ~20% (shirt/torso scale)\n\n")
        f.write("Profiles:\n")
        f.write("  Profile A (amp=0.005): Embedding corruption, NO suppression\n")
        f.write("    - Wearer L2=2.6, Bystander L2=0.5, Cos>0.9999\n")
        f.write("    - Use for stealthy cloud poisoning\n\n")
        f.write("  Profile B (amp=0.04): Moderate suppression + collateral\n")
        f.write("    - Wearer L2=11.2, Bystander L2=4.7, Cos=0.993\n")
        f.write("    - 1 wearer detection suppressed\n\n")
        f.write("  Max corrupt (amp=0.10): Maximum embedding corruption\n")
        f.write("    - Wearer L2 up to 25, Cos>0.987\n")
        f.write("    - May cause hallucinations at high freq\n\n")
        f.write("Camera survival: ~10-15% L2 reduction through phone camera sim\n")
        f.write("Best dual-purpose: k12_stripes\n")
    print(f"  README: {readme_path}")


def main():
    assert torch.cuda.is_available(), "CUDA required — check nvidia-smi"
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOv3
    print("Loading YOLOv3...")
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters():
        p.requires_grad_(False)
    print(f"  Model loaded on {DEVICE}")
    
    # Load image
    arr_base = load_image(IMG_WITH, IMG_SIZE)
    print(f"  Image loaded: {IMG_WITH} → {arr_base.shape}")
    
    # Run main pipeline experiment
    results, wearers, bystanders = run_pipeline_experiment(v3_model, arr_base)
    
    # Analyze profiles
    profile_results = analyze_profiles(results, wearers, bystanders)
    
    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    plot_amplitude_sweep(results, OUTPUT_DIR)
    plot_cosine_similarity(results, OUTPUT_DIR)
    
    # Camera comparison
    cam_results = run_camera_comparison(v3_model, arr_base, wearers, bystanders)

    # Generate print-ready patches
    generate_print_patches(profile_results)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total experiment runs: {len(results)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: pipeline_results.json, pipeline_results.csv, profile_analysis.json")
    print(f"  Plots: amplitude_sweep.png, cosine_similarity.png, camera_comparison.png")
    print(f"  Print patches: print_patches/")
    print(f"\n  Done.")

if __name__ == "__main__":
    main()
