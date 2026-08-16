"""
Physical Printability Test Pipeline

Simulates the full physical chain that digital testing misses:
1. Printer gamut clipping (consumer inkjet/dye-sub saturation limits)
2. Fabric weave destruction (sub-millimeter texture noise)
3. Wrinkle shadows (local contrast variation from fabric folds)
4. Viewing distance scaling (patch occupies fewer pixels at distance)
5. Lighting variation (indoor/outdoor/overcast/night color shifts)

Generates 3 print variants at amp=0.02:
- Raw k12_stripes (digital optimized)
- Palette-projected (32 printable colors)
- TV-smoothed (bilateral total variation regularizer)

Runs all 3 through physical sim, measures L2 survival.
"""

import os
import io
import json
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "CUDA required"

from patch_scale_pipeline import (
    Darknet, CONFIG_PATH, WEIGHTS_PATH, IMG_SIZE, IMG_WITH,
    PATCH_CX, PATCH_CY, WEARER_THRESHOLD,
    DETECTION_LAYERS, COCO_NAMES,
    load_image, make_deformable_mask, make_patch_pattern,
    forward_capture_v3, get_dets_v3,
    extract_embedding, cosine_similarity, l2_distance,
    OUTPUT_DIR
)

# ============================================================
# Physical degradation models
# ============================================================

def apply_printer_gamut(pattern_rgb, gamut="srgb_consumer_inkjet"):
    """
    Simulate printer gamut clipping.
    Consumer inkjet printers clip saturated extremes and have limited dynamic range.
    Typical gamut: ~85% of sRGB, black point ~15-20 (not 0), white point ~235-240.
    """
    p = pattern_rgb.copy().astype(np.float32)

    if gamut == "srgb_consumer_inkjet":
        # Consumer inkjet: clips blacks to ~18, whites to ~238
        # Reduces saturation by ~12% in extreme regions
        black_point = 18.0
        white_point = 238.0
        sat_reduction = 0.88  # 12% saturation loss

        # Clip to printer dynamic range
        p = np.clip(p, black_point, white_point)

        # Reduce saturation in RGB
        gray = np.mean(p, axis=2, keepdims=True)
        p = gray + (p - gray) * sat_reduction

    elif gamut == "dye_sublimation":
        # Dye-sub: wider gamut but softer, clips at ~22 and ~232
        black_point = 22.0
        white_point = 232.0
        sat_reduction = 0.92

        p = np.clip(p, black_point, white_point)
        gray = np.mean(p, axis=2, keepdims=True)
        p = gray + (p - gray) * sat_reduction

    elif gamut == "laser_printer":
        # Laser: narrow gamut, high contrast but posterized
        black_point = 25.0
        white_point = 230.0
        sat_reduction = 0.75

        p = np.clip(p, black_point, white_point)
        gray = np.mean(p, axis=2, keepdims=True)
        p = gray + (p - gray) * sat_reduction

        # Laser printers posterize midtones
        p = np.round(p / 16) * 16

    return np.clip(p, 0, 255).astype(np.uint8)


def apply_fabric_weave(pattern_rgb, fabric="cotton_standard", weave_density=0.15):
    """
    Simulate fabric weave destruction.
    Cotton weave creates sub-millimeter texture that acts as high-frequency noise.
    This is worse than JPEG because it's spatially structured, not smooth.
    """
    p = pattern_rgb.copy().astype(np.float32)
    h, w, c = p.shape

    if fabric == "cotton_standard":
        # Cotton thread count: ~120 threads/inch = ~47 threads/cm
        # At 300dpi print, weave period = 300/120 = 2.5px
        # Weave creates a 2D grid pattern with ~2.5px period
        weave_period = 2.5
        weave_amp = weave_density * 255  # amplitude of weave noise

        # Create weave pattern: grid of sinusoids
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        weave = (np.sin(2 * np.pi * x / weave_period) *
                 np.sin(2 * np.pi * y / weave_period))
        weave = weave * weave_amp

        # Add random thread irregularity
        rng = np.random.RandomState(42)
        thread_noise = rng.randn(h, w) * weave_amp * 0.3
        weave_total = weave + thread_noise

        for ch in range(c):
            p[:, :, ch] += weave_total

    elif fabric == "polyester_smooth":
        # Polyester: smoother, less weave, period ~1.5px, lower amplitude
        weave_period = 1.5
        weave_amp = weave_density * 0.5 * 255

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        weave = (np.sin(2 * np.pi * x / weave_period) *
                 np.sin(2 * np.pi * y / weave_period))
        weave = weave * weave_amp

        for ch in range(c):
            p[:, :, ch] += weave

    elif fabric == "cotton_heavy":
        # Heavy cotton (t-shirt material): coarse weave, period ~3.5px, high amplitude
        weave_period = 3.5
        weave_amp = weave_density * 1.5 * 255

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        weave = (np.sin(2 * np.pi * x / weave_period) *
                 np.sin(2 * np.pi * y / weave_period))
        weave = weave * weave_amp

        # Heavy cotton has more random thread variation
        rng = np.random.RandomState(42)
        thread_noise = rng.randn(h, w) * weave_amp * 0.5
        weave_total = weave + thread_noise

        for ch in range(c):
            p[:, :, ch] += weave_total

    return np.clip(p, 0, 255).astype(np.uint8)


def apply_wrinkle_shadows(pattern_rgb, n_wrinkles=5, shadow_strength=0.12):
    """
    Simulate wrinkle shadows on fabric.
    Real wrinkles cast soft shadows that shift pixel values unpredictably.
    Modeled as random soft dark bands at various angles.
    """
    p = pattern_rgb.copy().astype(np.float32)
    h, w, c = p.shape

    rng = np.random.RandomState(123)
    wrinkle_field = np.zeros((h, w), dtype=np.float32)

    for i in range(n_wrinkles):
        # Random wrinkle: a soft sinusoidal band at random position and angle
        angle = rng.uniform(0, np.pi)
        cx_w = rng.uniform(0, w)
        cy_w = rng.uniform(0, h)
        freq = rng.uniform(0.005, 0.02)  # spatial frequency of wrinkle
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0) * shadow_strength

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Project onto wrinkle direction
        proj = (x - cx_w) * np.cos(angle) + (y - cy_w) * np.sin(angle)
        wrinkle = amp * np.sin(2 * np.pi * freq * proj + phase)
        # Smooth the wrinkle to make it soft
        wrinkle = gaussian_filter(wrinkle, sigma=3.0)
        wrinkle_field += wrinkle

    # Wrinkle shadows darken (negative shift)
    for ch in range(c):
        p[:, :, ch] *= (1.0 + wrinkle_field)

    return np.clip(p, 0, 255).astype(np.uint8)


def apply_viewing_distance(pattern_rgb, patch_w_416, distance_ft=15):
    """
    Simulate viewing distance scaling.
    At 15 feet, a 416x416 input frame means the 20% patch occupies fewer pixels.
    The pattern is downsampled and re-upsampled, losing high-frequency detail.
    """
    # At 15 feet with a typical phone camera (70 degree FOV, 1080px):
    # A person ~6ft tall fills ~60% of frame height
    # In 416x416 YOLO input, person is ~250px tall
    # 20% patch = ~80x80 pixels in 416 space
    # At 20 feet: ~60x60 pixels
    # At 10 feet: ~120x120 pixels

    if distance_ft <= 10:
        effective_patch_px = 120
    elif distance_ft <= 15:
        effective_patch_px = 80
    elif distance_ft <= 20:
        effective_patch_px = 60
    else:
        effective_patch_px = 40

    p = Image.fromarray(pattern_rgb)
    # Downsample to effective resolution then back up
    p_small = p.resize((effective_patch_px, effective_patch_px), Image.BILINEAR)
    p_back = p_small.resize(p.size, Image.BILINEAR)

    return np.array(p_back)


def apply_lighting_variation(pattern_rgb, condition="outdoor_daylight"):
    """
    Simulate lighting variation.
    Different conditions shift color temperature and brightness.
    """
    p = pattern_rgb.copy().astype(np.float32)

    if condition == "outdoor_daylight":
        # Daylight: slightly blue, high brightness
        p[:, :, 0] *= 0.98  # slight red reduction
        p[:, :, 2] *= 1.03  # slight blue boost
        p *= 1.05  # brightness up

    elif condition == "outdoor_overcast":
        # Overcast: flat, cool, lower contrast
        p[:, :, 0] *= 0.92
        p[:, :, 1] *= 0.96
        p[:, :, 2] *= 1.02
        p *= 0.88  # brightness down
        # Reduce contrast
        gray = np.mean(p, axis=2, keepdims=True)
        p = gray + (p - gray) * 0.85

    elif condition == "indoor_fluorescent":
        # Fluorescent: greenish, harsh
        p[:, :, 0] *= 0.95
        p[:, :, 1] *= 1.04
        p[:, :, 2] *= 0.97
        p *= 0.92

    elif condition == "indoor_warm":
        # Warm indoor (incandescent): yellow/orange shift
        p[:, :, 0] *= 1.06
        p[:, :, 1] *= 1.03
        p[:, :, 2] *= 0.88
        p *= 0.95

    elif condition == "streetlight_night":
        # Sodium vapor streetlight: heavy yellow/orange, low brightness
        p[:, :, 0] *= 1.15
        p[:, :, 1] *= 1.05
        p[:, :, 2] *= 0.65
        p *= 0.55  # much darker
        # Very low contrast
        gray = np.mean(p, axis=2, keepdims=True)
        p = gray + (p - gray) * 0.6

    return np.clip(p, 0, 255).astype(np.uint8)


def apply_camera_phone(pattern_rgb, blur_sigma=2.5, jpeg_quality=75):
    """Phone camera processing: blur + JPEG."""
    p = Image.fromarray(pattern_rgb)
    if blur_sigma > 0:
        p = p.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    if jpeg_quality > 0:
        buf = io.BytesIO()
        p.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        p = Image.open(buf).convert("RGB")
    return np.array(p)


# ============================================================
# Print variant generation
# ============================================================

def palette_project(pattern_rgb, n_colors=32):
    """
    Quantize to n printable colors using median cut.
    Simulates the limited color palette of fabric printing.
    """
    p = Image.fromarray(pattern_rgb)
    quantized = p.quantize(colors=n_colors, method=Image.MEDIANCUT)
    return np.array(quantized.convert("RGB"))


def bilateral_tv_smoothing(pattern_rgb, weight=0.1, n_iters=3):
    """
    Bilateral total variation smoothing.
    Structured regularizer that smooths while preserving edges.
    """
    p = pattern_rgb.copy().astype(np.float32) / 255.0
    h, w, c = p.shape

    for _ in range(n_iters):
        for ch in range(c):
            u = p[:, :, ch]
            # Gradients
            ux = np.roll(u, -1, axis=1) - u
            uy = np.roll(u, -1, axis=0) - u

            # Bilateral weight: reduce smoothing near edges
            gray = np.mean(p, axis=2)
            gx = np.abs(np.roll(gray, -1, axis=1) - gray)
            gy = np.abs(np.roll(gray, -1, axis=0) - gray)
            wx = np.exp(-(gx ** 2) / 0.02)
            wy = np.exp(-(gy ** 2) / 0.02)

            # TV update
            ux *= weight * wx
            uy *= weight * wy

            u_new = u - (ux - np.roll(ux, 1, axis=1)) - (uy - np.roll(uy, 1, axis=0))
            p[:, :, ch] = np.clip(u_new, 0, 1)

    return (p * 255).astype(np.uint8)


# ============================================================
# Full physical pipeline
# ============================================================

def simulate_physical_chain(pattern_rgb, config):
    """
    Run the full physical degradation chain.
    pattern_rgb: (H, W, 3) uint8 RGB image of the patch
    config: dict of physical parameters
    Returns: degraded patch at 416-space resolution
    """
    p = pattern_rgb.copy()

    # Step 1: Printer gamut clipping
    p = apply_printer_gamut(p, gamut=config["printer"])

    # Step 2: Fabric weave destruction
    p = apply_fabric_weave(p, fabric=config["fabric"], weave_density=config["weave_density"])

    # Step 3: Wrinkle shadows
    p = apply_wrinkle_shadows(p, n_wrinkles=config["n_wrinkles"], shadow_strength=config["wrinkle_strength"])

    # Step 4: Lighting variation
    p = apply_lighting_variation(p, condition=config["lighting"])

    # Step 5: Viewing distance scaling
    p = apply_viewing_distance(p, config["patch_w_416"], distance_ft=config["distance_ft"])

    # Step 6: Phone camera capture
    p = apply_camera_phone(p, blur_sigma=config["blur_sigma"], jpeg_quality=config["jpeg_quality"])

    return p


def composite_patch_on_image(arr_base, patch_rgb_416, mask_416):
    """Composite a 416-space RGB patch onto the base image using mask."""
    # patch_rgb_416: (ph, pw, 3) uint8
    # mask_416: (IMG_SIZE, IMG_SIZE) float32
    ph, pw, _ = patch_rgb_416.shape
    x0 = PATCH_CX - pw // 2
    y0 = PATCH_CY - ph // 2

    full_pat = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    px0, py0 = max(0, x0), max(0, y0)
    px1 = min(IMG_SIZE, x0 + pw)
    py1 = min(IMG_SIZE, y0 + ph)
    sx0, sy0 = px0 - x0, py0 - y0
    sx1 = sx0 + (px1 - px0)
    sy1 = sy0 + (py1 - py0)

    full_pat[py0:py1, px0:px1] = patch_rgb_416[sy0:sy1, sx0:sx1] / 255.0

    arr_comp = arr_base.copy()
    for c in range(3):
        arr_comp[:, :, c] = np.clip(
            arr_base[:, :, c] * (1 - mask_416) +
            full_pat[:, :, c] * mask_416,
            0, 1
        )
    return arr_comp


# ============================================================
# Main experiment
# ============================================================

def run_physical_experiment():
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

    # Baseline detections
    tensor_base = torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    base_dets = get_dets_v3(v3_model, tensor_base, conf=0.1)
    base_persons = [d for d in base_dets if d["class_name"] == "person"]
    for d in base_persons:
        dist = math.sqrt((d["cx"] - PATCH_CX) ** 2 + (d["cy"] - PATCH_CY) ** 2)
        d["dist_to_patch"] = dist
        d["is_wearer"] = dist < WEARER_THRESHOLD
    wearers = [d for d in base_persons if d["is_wearer"]]
    bystanders = [d for d in base_persons if not d["is_wearer"]]
    print(f"  Baseline: {len(base_persons)} persons ({len(wearers)} wearer, {len(bystanders)} bystander)")

    # Clean embeddings
    with torch.no_grad():
        caps_clean, _ = forward_capture_v3(v3_model, tensor_base)
    clean_embs = {"wearer": {}, "bystanders": []}
    for layer_name, layer_idx in DETECTION_LAYERS.items():
        if wearers:
            w_vecs = [extract_embedding(caps_clean, layer_idx, d["cx"], d["cy"]) for d in wearers]
            clean_embs["wearer"][layer_name] = np.mean(w_vecs, axis=0)
        if bystanders:
            for d in bystanders:
                b_vec = extract_embedding(caps_clean, layer_idx, d["cx"], d["cy"])
                clean_embs["bystanders"].append({"layer": layer_name, "det": d, "vec": b_vec})

    # Patch geometry: xlarge_16pct
    shirt_rays = [110, 90, 120, 85, 115, 95, 125, 90, 110, 95, 120, 85]
    patch_w_416 = int(max(shirt_rays) * 2)  # 250px in 416 space
    mask_416, _ = make_deformable_mask(IMG_SIZE, IMG_SIZE, PATCH_CX, PATCH_CY, shirt_rays, 12)

    # Test both patterns at amp=0.02 (middle of Profile A window)
    AMP = 0.02
    PRINT_RES = 1000

    patterns_to_test = [
        ("k12_stripes", 12, "stripes_v"),
        ("digits_196", 0, "digits_196"),
    ]

    print_dir = os.path.join(OUTPUT_DIR, "physical_print_test")
    os.makedirs(print_dir, exist_ok=True)

    # Generate all pattern x variant combinations
    all_variants = {}  # key: "pattern_variant", value: (pattern_label, var_label, img)

    for pat_label, k_patch, tex_type in patterns_to_test:
        pat_raw = make_patch_pattern(PRINT_RES, PRINT_RES, k_patch, tex_type, AMP)

        # Convert to RGB: gray base (128) + pattern
        rgb_raw = np.zeros((PRINT_RES, PRINT_RES, 3), dtype=np.float32)
        base_gray = 128.0
        for c in range(3):
            rgb_raw[:, :, c] = base_gray + pat_raw * 255.0
        rgb_raw = np.clip(rgb_raw, 0, 255).astype(np.uint8)

        # Generate 3 variants per pattern
        pat_variants = {
            "raw": rgb_raw,
            "palette_32": palette_project(rgb_raw, n_colors=32),
            "tv_smoothed": bilateral_tv_smoothing(rgb_raw, weight=0.1, n_iters=3),
        }

        for var_label, img in pat_variants.items():
            key = f"{pat_label}_{var_label}"
            all_variants[key] = (pat_label, var_label, img)
            path = os.path.join(print_dir, f"{key}_amp{AMP:.3f}_{PRINT_RES}px.png")
            Image.fromarray(img).save(path)
            print(f"  Saved: {path}")

    print(f"\n  Generated {len(all_variants)} print-ready PNGs ({len(patterns_to_test)} patterns x 3 variants)")

    # Physical conditions to test
    conditions = [
        # (label, printer, fabric, weave_density, n_wrinkles, wrinkle_strength, lighting, distance_ft, blur_sigma, jpeg_quality)
        ("digital_perfect", "srgb_consumer_inkjet", "polyester_smooth", 0.0, 0, 0.0, "outdoor_daylight", 5, 0, 100),
        ("indoor_close_clean", "srgb_consumer_inkjet", "cotton_standard", 0.08, 2, 0.06, "indoor_fluorescent", 10, 1.5, 85),
        ("outdoor_15ft_daylight", "srgb_consumer_inkjet", "cotton_standard", 0.15, 5, 0.12, "outdoor_daylight", 15, 2.5, 75),
        ("outdoor_20ft_daylight", "srgb_consumer_inkjet", "cotton_standard", 0.15, 5, 0.12, "outdoor_daylight", 20, 2.5, 75),
        ("outdoor_15ft_overcast", "srgb_consumer_inkjet", "cotton_standard", 0.15, 5, 0.12, "outdoor_overcast", 15, 2.5, 75),
        ("indoor_15ft_warm", "srgb_consumer_inkjet", "cotton_standard", 0.15, 5, 0.12, "indoor_warm", 15, 2.5, 75),
        ("night_15ft_streetlight", "srgb_consumer_inkjet", "cotton_standard", 0.15, 5, 0.12, "streetlight_night", 15, 2.5, 75),
        ("heavy_cotton_15ft", "srgb_consumer_inkjet", "cotton_heavy", 0.22, 7, 0.18, "outdoor_daylight", 15, 2.5, 75),
        ("dye_sub_15ft", "dye_sublimation", "cotton_standard", 0.15, 5, 0.12, "outdoor_daylight", 15, 2.5, 75),
        ("laser_15ft", "laser_printer", "cotton_standard", 0.15, 5, 0.12, "outdoor_daylight", 15, 2.5, 75),
    ]

    results = []

    for cond_label, printer, fabric, weave_d, n_wr, wr_str, lighting, dist_ft, blur, jpeg in conditions:
        config = {
            "printer": printer,
            "fabric": fabric,
            "weave_density": weave_d,
            "n_wrinkles": n_wr,
            "wrinkle_strength": wr_str,
            "lighting": lighting,
            "distance_ft": dist_ft,
            "blur_sigma": blur,
            "jpeg_quality": jpeg,
            "patch_w_416": patch_w_416,
        }

        print(f"\n  Condition: {cond_label}")

        for key, (pat_label, var_label, var_img) in all_variants.items():
            # Run physical chain
            degraded = simulate_physical_chain(var_img, config)

            # Resize to 416-space patch size
            degraded_pil = Image.fromarray(degraded)
            degraded_416 = np.array(degraded_pil.resize((patch_w_416, patch_w_416), Image.BILINEAR))

            # Composite onto base image
            arr_comp = composite_patch_on_image(arr_base, degraded_416, mask_416)

            # Run YOLOv3
            tensor_comp = torch.from_numpy(arr_comp).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                caps_comp, _ = forward_capture_v3(v3_model, tensor_comp)
            comp_dets = get_dets_v3(v3_model, tensor_comp, conf=0.1)
            comp_persons = [d for d in comp_dets if d["class_name"] == "person"]

            # Extract wearer embeddings
            wearer_l2 = {}
            wearer_cos = {}
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                if wearers:
                    w_vecs = [extract_embedding(caps_comp, layer_idx, d["cx"], d["cy"]) for d in wearers]
                    w_emb = np.mean(w_vecs, axis=0)
                    w_clean = clean_embs["wearer"][layer_name]
                    wearer_l2[layer_name] = l2_distance(w_emb, w_clean)
                    wearer_cos[layer_name] = cosine_similarity(w_emb, w_clean)

            avg_w_l2 = float(np.mean(list(wearer_l2.values())))
            avg_w_cos = float(np.mean(list(wearer_cos.values())))

            # Bystander embeddings
            bystander_l2 = {}
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                if bystanders:
                    b_vecs = [extract_embedding(caps_comp, layer_idx, d["cx"], d["cy"]) for d in bystanders]
                    b_emb = np.mean(b_vecs, axis=0)
                    b_clean = np.zeros(255, dtype=np.float32)
                    for be in clean_embs["bystanders"]:
                        if be["layer"] == layer_name:
                            b_clean = be["vec"]
                            break
                    bystander_l2[layer_name] = l2_distance(b_emb, b_clean)

            avg_b_l2 = float(np.mean(list(bystander_l2.values())))

            # Count wearer/bystander detections
            w_count = sum(1 for d in comp_persons if math.sqrt((d["cx"] - PATCH_CX)**2 + (d["cy"] - PATCH_CY)**2) < WEARER_THRESHOLD)
            b_count = len(comp_persons) - w_count

            result = {
                "condition": cond_label,
                "pattern": pat_label,
                "variant": var_label,
                "printer": printer,
                "fabric": fabric,
                "weave_density": weave_d,
                "lighting": lighting,
                "distance_ft": dist_ft,
                "blur_sigma": blur,
                "jpeg_quality": jpeg,
                "total_persons": len(comp_persons),
                "wearer_count": w_count,
                "bystander_count": b_count,
                "wearer_l2": avg_w_l2,
                "bystander_l2": avg_b_l2,
                "wearer_cos": avg_w_cos,
                "L81_l2": wearer_l2["L81_52x52"],
                "L93_l2": wearer_l2["L93_26x26"],
                "L105_l2": wearer_l2["L105_13x13"],
            }
            results.append(result)

            status = f"    {pat_label:15s} {var_label:12s}: P={len(comp_persons):2d} (W={w_count},B={b_count}) W_L2={avg_w_l2:.2f} B_L2={avg_b_l2:.2f} W_cos={avg_w_cos:.4f}"
            print(status)

            # Save composite preview for key conditions
            if cond_label in ("digital_perfect", "outdoor_15ft_daylight", "outdoor_20ft_daylight", "night_15ft_streetlight"):
                comp_path = os.path.join(print_dir, f"composite_{cond_label}_{pat_label}_{var_label}.png")
                Image.fromarray((arr_comp * 255).astype(np.uint8)).save(comp_path)

    # Save results
    results_path = os.path.join(print_dir, "physical_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # CSV
    csv_path = os.path.join(print_dir, "physical_results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {csv_path}")

    # ---- Analysis ----
    print(f"\n{'=' * 70}")
    print("PHYSICAL DEGRADATION ANALYSIS")
    print(f"{'=' * 70}")

    # For each pattern+variant, compute L2 survival ratio vs digital_perfect
    digital_ref = {}
    for key, (pat_label, var_label, _) in all_variants.items():
        for r in results:
            if r["condition"] == "digital_perfect" and r["pattern"] == pat_label and r["variant"] == var_label:
                digital_ref[key] = r["wearer_l2"]
                break

    var_labels = ["raw", "palette_32", "tv_smoothed"]
    pat_labels = [p[0] for p in patterns_to_test]

    print(f"\n  L2 survival ratio (wearer L2 / digital_perfect L2):")
    header = f"  {'Condition':30s}"
    for pl in pat_labels:
        for vl in var_labels:
            header += f" {pl[:6]}_{vl[:4]:>10s}"
    print(header)
    for cond_label, *_ in conditions:
        line = f"  {cond_label:30s}"
        for pl in pat_labels:
            for vl in var_labels:
                for r in results:
                    if r["condition"] == cond_label and r["pattern"] == pl and r["variant"] == vl:
                        ref_key = f"{pl}_{vl}"
                        ratio = r["wearer_l2"] / digital_ref[ref_key] if digital_ref.get(ref_key, 0) > 0 else 0
                        line += f" {ratio:>10.2f}"
                        break
        print(line)

    print(f"\n  Absolute wearer L2 by condition:")
    header = f"  {'Condition':30s}"
    for pl in pat_labels:
        for vl in var_labels:
            header += f" {pl[:6]}_{vl[:4]:>10s}"
    print(header)
    for cond_label, *_ in conditions:
        line = f"  {cond_label:30s}"
        for pl in pat_labels:
            for vl in var_labels:
                for r in results:
                    if r["condition"] == cond_label and r["pattern"] == pl and r["variant"] == vl:
                        line += f" {r['wearer_l2']:>10.2f}"
                        break
        print(line)

    # Key question: does L2 > 1.0 survive physical degradation?
    print(f"\n  L2 > 1.0 threshold (attack survives):")
    for cond_label, *_ in conditions:
        line = f"    {cond_label:30s}"
        for pl in pat_labels:
            for vl in var_labels:
                for r in results:
                    if r["condition"] == cond_label and r["pattern"] == pl and r["variant"] == vl:
                        line += f"  {'YES' if r['wearer_l2'] > 1.0 else 'NO':>6s}"
                        break
        print(line)

    # Plot: L2 by condition for each pattern x variant
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    x = np.arange(len(conditions))
    width = 0.12
    colors = ["steelblue", "orange", "green", "red", "purple", "brown"]
    for ax_idx, pl in enumerate(pat_labels):
        ax = axes[ax_idx]
        for i, vl in enumerate(var_labels):
            l2_vals = []
            for cond_label, *_ in conditions:
                for r in results:
                    if r["condition"] == cond_label and r["pattern"] == pl and r["variant"] == vl:
                        l2_vals.append(r["wearer_l2"])
                        break
            ax.bar(x + i * width, l2_vals, width, label=vl, color=colors[i])
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="L2=1.0 threshold")
        ax.set_xlabel("Physical Condition")
        ax.set_ylabel("Wearer L2 Distance")
        ax.set_title(f"{pl}: Embedding Corruption Survival (amp=0.02)")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([c[0] for c in conditions], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = os.path.join(print_dir, "physical_degradation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Plot: degradation chain breakdown for both patterns (raw variant, outdoor 15ft daylight)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    chain_steps = ["Digital", "+Printer", "+Fabric", "+Wrinkles", "+Lighting", "+Distance", "+Camera"]

    def measure_l2(degraded_rgb):
        degraded_pil = Image.fromarray(degraded_rgb)
        degraded_416 = np.array(degraded_pil.resize((patch_w_416, patch_w_416), Image.BILINEAR))
        arr_comp = composite_patch_on_image(arr_base, degraded_416, mask_416)
        tensor_comp = torch.from_numpy(arr_comp).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            caps_comp, _ = forward_capture_v3(v3_model, tensor_comp)
        w_l2s = []
        for layer_name, layer_idx in DETECTION_LAYERS.items():
            if wearers:
                w_vecs = [extract_embedding(caps_comp, layer_idx, d["cx"], d["cy"]) for d in wearers]
                w_emb = np.mean(w_vecs, axis=0)
                w_l2s.append(l2_distance(w_emb, clean_embs["wearer"][layer_name]))
        return float(np.mean(w_l2s))

    for ax_idx, (pl, k_patch, tex_type) in enumerate(patterns_to_test):
        ax = axes[ax_idx]
        base_rgb = all_variants[f"{pl}_raw"][2].copy()
        p = base_rgb.copy()

        chain_l2 = [digital_ref[f"{pl}_raw"]]

        p = apply_printer_gamut(p, "srgb_consumer_inkjet")
        chain_l2.append(measure_l2(p))
        p = apply_fabric_weave(p, "cotton_standard", 0.15)
        chain_l2.append(measure_l2(p))
        p = apply_wrinkle_shadows(p, 5, 0.12)
        chain_l2.append(measure_l2(p))
        p = apply_lighting_variation(p, "outdoor_daylight")
        chain_l2.append(measure_l2(p))
        p = apply_viewing_distance(p, patch_w_416, 15)
        chain_l2.append(measure_l2(p))
        p = apply_camera_phone(p, 2.5, 75)
        chain_l2.append(measure_l2(p))

        ax.bar(chain_steps, chain_l2, color="steelblue")
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="L2=1.0 threshold")
        ax.set_ylabel("Wearer L2 Distance")
        ax.set_title(f"{pl}: Degradation Chain (raw, amp=0.02, outdoor 15ft)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(chain_l2):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    chain_plot_path = os.path.join(print_dir, "degradation_chain.png")
    plt.savefig(chain_plot_path, dpi=150)
    plt.close()
    print(f"  Chain plot saved: {chain_plot_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"  Output: {print_dir}")
    print(f"  Print-ready PNGs: {len(all_variants)} variants ({len(patterns_to_test)} patterns x 3 variants) at {PRINT_RES}x{PRINT_RES}px")
    print(f"  Results: physical_results.json, physical_results.csv")
    print(f"  Plots: physical_degradation.png, degradation_chain.png")


if __name__ == "__main__":
    run_physical_experiment()
