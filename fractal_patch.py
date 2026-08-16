"""
Scale-Invariant Fractal Patch — triangles within triangles at all scales.

Design principle: instead of one frequency (k=27) that only works at one scale,
generate a self-similar fractal with frequency content at k=3,9,27,81,243,729
simultaneously. At any distance, at any patch size, some scale of the pattern
will hit the network's vulnerable frequency.

Structure:
  Level 0: Large triangle (k~3, hits early conv layers 0-12)
  Level 1: 3 sub-triangles (k~9, hits layers 12-37)
  Level 2: 9 sub-triangles (k~27, hits layers 37-62)
  Level 3: 27 sub-triangles (k~81, hits layers 62-75)
  Level 4: 81 sub-triangles (k~243, hits layers 75-81)
  Level 5: 243 sub-triangles (k~729, broadband for detection heads L81/L93/L105)

Each level's sub-triangles are filled with a sinusoidal texture at the
appropriate frequency for that scale, with per-channel RGB phase offsets
so different channels hit different frequency bands.

The amplitude is normalized per level so each frequency band carries equal
energy — this is the 1/f property that makes the pattern scale-invariant.

Output:
  - Print-ready PNG at 3600x4800 300dpi
  - Test-resolution PNG at 416x416 for YOLOv3 forward pass
  - FFT spectrum showing multi-scale frequency content
  - Forward pass embedding shift at L81/L93/L105 compared to baseline + poison
"""

import os, sys, math, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "PyTorch-YOLOv3"))
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CFG = os.path.join(BASE, "PyTorch-YOLOv3", "config", "yolov3.cfg")
WTS = os.path.join(BASE, "yolov3.weights")
IMG_WITH = os.path.join(BASE, "withhuman.png")
IMG_WITHOUT = os.path.join(BASE, "withouthuman.png")
POISON_PATCH = os.path.join(BASE, "outputs_clothing", "forward_analysis", "patch_pipeline", "dual_optim", "poison", "poison_patch.png")
SUPPRESS_PATCH = os.path.join(BASE, "outputs_clothing", "forward_analysis", "patch_pipeline", "dual_optim", "suppress", "suppress_patch.png")
OUT = os.path.join(BASE, "outputs_clothing", "forward_analysis", "fractal_patch")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416  # inference size

DETECTION_LAYERS = {"L81_52x52": 81, "L93_26x26": 93, "L105_13x13": 105}

os.makedirs(OUT, exist_ok=True)


# ============================================================
# Fractal patch generation
# ============================================================

def triangle_vertices(cx, cy, size, orientation="up"):
    """Return the 3 vertices of an equilateral triangle centered at (cx, cy)."""
    h = size * math.sqrt(3) / 2
    if orientation == "up":
        return [(cx, cy - h * 2/3), (cx - size/2, cy + h/3), (cx + size/2, cy + h/3)]
    else:
        return [(cx, cy + h * 2/3), (cx - size/2, cy - h/3), (cx + size/2, cy - h/3)]


def subdivide_triangle(v0, v1, v2, depth, out_triangles):
    """Recursively subdivide a triangle into Sierpinski pattern.
    Returns list of (cx, cy, size, level) for each filled sub-triangle."""
    if depth == 0:
        cx = (v0[0] + v1[0] + v2[0]) / 3
        cy = (v0[1] + v1[1] + v2[1]) / 3
        # Size = distance from centroid to vertex * 2
        size = 2 * math.sqrt((v0[0]-cx)**2 + (v0[1]-cy)**2)
        out_triangles.append((cx, cy, size, 0))
        return

    # Midpoints of edges
    m01 = ((v0[0]+v1[0])/2, (v0[1]+v1[1])/2)
    m12 = ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2)
    m20 = ((v2[0]+v0[0])/2, (v2[1]+v0[1])/2)

    # Three sub-triangles (Sierpinski: skip the inverted center one)
    level = depth
    subdivide_triangle(v0, m01, m20, depth-1, out_triangles)
    subdivide_triangle(m01, v1, m12, depth-1, out_triangles)
    subdivide_triangle(m20, m12, v2, depth-1, out_triangles)

    # Also add the inverted center triangle at this level for frequency content
    # (standard Sierpinski skips it, but we WANT energy at every scale)
    cx = (m01[0] + m12[0] + m20[0]) / 3
    cy = (m01[1] + m12[1] + m20[1]) / 3
    size = 2 * math.sqrt((m01[0]-cx)**2 + (m01[1]-cy)**2)
    out_triangles.append((cx, cy, size, level))


def generate_fractal_texture(H, W, cx, cy, outer_size, max_depth=5,
                              base_amp=0.5, channel_phase_offset=0.0):
    """Generate a multi-scale fractal texture.

    Each level of the Sierpinski subdivision gets filled with a sinusoidal
    texture whose spatial frequency matches that scale. Amplitude is normalized
    per level so each frequency band carries equal energy (1/f property).

    channel_phase_offset: phase shift in radians for this color channel,
    so R/G/B channels hit different frequency bands.
    """
    # Collect all triangles at all levels
    v0, v1, v2 = triangle_vertices(cx, cy, outer_size, "up")
    triangles = []
    subdivide_triangle(v0, v1, v2, max_depth, triangles)

    # Build coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    texture = np.zeros((H, W), dtype=np.float32)

    # Group triangles by level
    by_level = {}
    for tcx, tcy, tsize, level in triangles:
        if level not in by_level:
            by_level[level] = []
        by_level[level].append((tcx, tcy, tsize))

    # For each level, generate the appropriate frequency texture
    # and mask it to the triangles at that level
    for level, tris in sorted(by_level.items()):
        # Frequency at this level: k = 3^level * base_k
        # base_k chosen so level 0 has k~3 (very low frequency)
        k = max(1, 3 ** level)

        # Amplitude normalization: 1/sqrt(N_triangles_at_this_level) for equal energy
        # But also scale by 1/sqrt(k) for 1/f spectral shape
        amp = base_amp / math.sqrt(k)

        # Phase offset per channel
        phase = channel_phase_offset + level * 0.37  # golden angle-ish increment

        # Generate sinusoidal texture at this frequency
        # Use diagonal direction for better coverage
        sinusoid = amp * np.cos(2 * np.pi * k * (x_grid / W + y_grid / H) + phase)

        # Build mask for all triangles at this level
        level_mask = np.zeros((H, W), dtype=np.float32)
        img_mask = Image.new("F", (W, H), 0.0)
        draw = ImageDraw.Draw(img_mask)
        for tcx, tcy, tsize in tris:
            verts = triangle_vertices(tcx, tcy, tsize, "up")
            draw.polygon(verts, fill=1.0)
        level_mask = np.array(img_mask, dtype=np.float32)

        # Add this level's contribution
        texture += sinusoid * level_mask

    # Normalize to [-1, 1] range, then shift to [0, 1] for image
    if texture.max() - texture.min() > 0:
        texture = (texture - texture.min()) / (texture.max() - texture.min())
    else:
        texture = np.full((H, W), 0.5, dtype=np.float32)

    return texture.astype(np.float32)


def generate_fractal_patch_rgb(H, W, cx, cy, outer_size, max_depth=5,
                                base_amp=0.5, add_broadband=True):
    """Generate RGB fractal patch with per-channel phase offsets.

    R channel: phase 0.0 (hits k=3,9,27,81,243 at base phase)
    G channel: phase 2.094 (120deg offset, shifts frequency response)
    B channel: phase 4.189 (240deg offset, different frequency response)

    This ensures different channels stimulate different feature channels in
    the first conv layer, maximizing the number of corrupted feature maps.
    """
    tex_r = generate_fractal_texture(H, W, cx, cy, outer_size, max_depth,
                                      base_amp, channel_phase_offset=0.0)
    tex_g = generate_fractal_texture(H, W, cx, cy, outer_size, max_depth,
                                      base_amp, channel_phase_offset=2.094)
    tex_b = generate_fractal_texture(H, W, cx, cy, outer_size, max_depth,
                                      base_amp, channel_phase_offset=4.189)

    patch = np.stack([tex_r, tex_g, tex_b], axis=2)

    # Add broadband noise within the patch region for detection head corruption
    if add_broadband:
        # Build overall triangular mask
        v0, v1, v2 = triangle_vertices(cx, cy, outer_size, "up")
        img_mask = Image.new("F", (W, H), 0.0)
        draw = ImageDraw.Draw(img_mask)
        draw.polygon([v0, v1, v2], fill=1.0)
        overall_mask = np.array(img_mask, dtype=np.float32)

        # Broadband noise: sum of high-frequency sinusoids
        rng = np.random.RandomState(42)
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        broadband = np.zeros((H, W), dtype=np.float32)
        for k in [100, 150, 200, 167, 208, 196]:
            phase = rng.uniform(0, 2 * np.pi)
            angle = rng.uniform(0, np.pi)
            broadband += 0.05 * np.cos(2 * np.pi * k *
                (x_grid * math.cos(angle) / W + y_grid * math.sin(angle) / H) + phase)
        broadband = broadband / np.abs(broadband).max() * 0.15

        for c in range(3):
            patch[:, :, c] = np.clip(patch[:, :, c] + broadband * overall_mask, 0, 1)

    # Build final triangular mask (outer triangle only)
    v0, v1, v2 = triangle_vertices(cx, cy, outer_size, "up")
    img_mask = Image.new("F", (W, H), 0.0)
    draw = ImageDraw.Draw(img_mask)
    draw.polygon([v0, v1, v2], fill=1.0)
    mask = np.array(img_mask, dtype=np.float32)

    return patch, mask


# ============================================================
# Forward pass analysis (reuse carrier_layer_analysis approach)
# ============================================================

def fwd_all(model, x):
    """Forward capture all conv layer outputs."""
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
            x = mo[0](x, IS)
        if md["type"] == "convolutional":
            caps[i] = x.detach()
        los.append(x)
    return caps, x


def load_img(path, sz=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(sz/w, sz/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (sz, sz), (128, 128, 128))
    c.paste(r, ((sz-nw)//2, (sz-nh)//2))
    return np.array(c, dtype=np.float32) / 255.0


def extract_emb_at(caps, layer_idx, spatial_x, spatial_y):
    """Extract embedding vector at a spatial location from a feature map."""
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW-1, int(spatial_x / IS * fW)))
    fy = max(0, min(fH-1, int(spatial_y / IS * fH)))
    return feat[0, :, fy, fx]


def gap_embedding(caps, layer_idx):
    """Global average pool embedding from a feature map."""
    feat = caps[layer_idx]
    return F.adaptive_avg_pool2d(feat, 1).squeeze()


def apply_patch_to_image(arr, patch_rgb, mask, cx, cy):
    """Apply RGB patch to image at (cx, cy) using mask."""
    H, W, _ = arr.shape
    out = arr.copy()
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - mask) + patch_rgb[:, :, c] * mask
    return np.clip(out, 0, 1)


def load_patch_image(path, H, W, cx, cy, target_size):
    """Load a patch image and place it at (cx, cy) with triangular mask."""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Build triangular mask
    v0, v1, v2 = triangle_vertices(cx, cy, target_size, "up")
    img_mask = Image.new("F", (W, H), 0.0)
    draw = ImageDraw.Draw(img_mask)
    draw.polygon([v0, v1, v2], fill=1.0)
    mask = np.array(img_mask, dtype=np.float32)
    return arr, mask


# ============================================================
# FFT analysis
# ============================================================

def compute_2d_fft_mag(texture):
    """Compute 2D FFT magnitude spectrum of a grayscale texture."""
    f = np.fft.fft2(texture)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log1p(mag)
    return mag


def radial_average(mag, H, W):
    """Compute radial average of FFT magnitude — power at each frequency k."""
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(H, W) // 2
    radial = np.zeros(max_r, dtype=np.float32)
    count = np.zeros(max_r, dtype=np.float32)
    for i in range(max_r):
        mask = (r == i)
        if mask.any():
            radial[i] = mag[mask].mean()
            count[i] = mask.sum()
    return radial[:max_r]


# ============================================================
# Main
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Scale-Invariant Fractal Patch — Multi-Scale Frequency Attack")
    print("=" * 70)
    print(f"Device: {DEV}")

    H, W = IS, IS

    # ============================================================
    # 1. Generate fractal patches at multiple depths
    # ============================================================
    print("\n--- Generating fractal patches ---")

    # Patch placement: center of person torso
    patch_cx, patch_cy = IS // 2, int(IS * 0.58)
    # Large triangle: 200px on 416px image = ~18% area
    outer_size = 200

    patches = {}
    for depth in [3, 4, 5]:
        patch_rgb, mask = generate_fractal_patch_rgb(
            H, W, patch_cx, patch_cy, outer_size,
            max_depth=depth, base_amp=0.5, add_broadband=True
        )
        patches[f"fractal_d{depth}"] = (patch_rgb, mask)
        area_pct = float(np.mean(mask)) * 100
        print(f"  fractal_d{depth}: area={area_pct:.1f}%, "
              f"{3**depth} sub-triangles, "
              f"freq bands k={','.join(str(3**l) for l in range(depth+1))}")

        # Save visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title(f"Mask (d={depth})")
        axes[0].axis("off")
        axes[1].imshow(patch_rgb)
        axes[1].set_title(f"RGB Patch (d={depth})")
        axes[1].axis("off")
        # FFT of grayscale version
        gray = np.mean(patch_rgb, axis=2)
        mag = compute_2d_fft_mag(gray)
        axes[2].imshow(mag, cmap="inferno", extent=[-W//2, W//2, H//2, -H//2])
        axes[2].set_title("2D FFT Magnitude")
        axes[2].axis("off")
        # Radial average
        radial = radial_average(mag, H, W)
        axes[3].plot(range(len(radial)), radial, "b-", linewidth=0.8)
        # Mark k=3^n peaks
        for n in range(depth + 1):
            k = 3 ** n
            if k < len(radial):
                axes[3].axvline(x=k, color="r", linestyle="--", alpha=0.5)
                axes[3].text(k, radial.max() * 0.9, f"k={k}", fontsize=7, color="r")
        axes[3].set_title("Radial FFT (red = k=3^n)")
        axes[3].set_xlabel("Frequency k")
        axes[3].set_ylabel("log(1+|F|)")
        axes[3].set_xlim(0, min(200, len(radial)))
        plt.suptitle(f"Fractal Patch depth={depth} — Scale-Invariant Multi-Frequency", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/fractal_d{depth}_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save patch PNG
        patch_uint8 = (patch_rgb * 255).astype(np.uint8)
        Image.fromarray(patch_uint8).save(f"{OUT}/fractal_d{depth}_416.png")

    # ============================================================
    # 2. Load YOLOv3 and run forward pass analysis
    # ============================================================
    print("\n--- Loading YOLOv3 ---")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load test image
    arr_w = load_img(IMG_WITH, IS)
    arr_wo = load_img(IMG_WITHOUT, IS)

    # Baseline: clean image (no patch)
    tensor_clean = torch.from_numpy(arr_w).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_clean, _ = fwd_all(model, tensor_clean)

    # Baseline: empty frame (no person)
    tensor_empty = torch.from_numpy(arr_wo).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_empty, _ = fwd_all(model, tensor_empty)

    # Person signal: difference between with-human and without-human
    person_signal = {}
    for lname, lidx in DETECTION_LAYERS.items():
        person_signal[lname] = (caps_clean[lidx] - caps_empty[lidx]).squeeze(0)

    # Extract baseline embeddings at detection layers
    # Use GAP for whole-frame, and center-point for person (torso location)
    person_sx, person_sy = IS // 2, int(IS * 0.58)
    baseline_emb = {}
    for lname, lidx in DETECTION_LAYERS.items():
        gap = gap_embedding(caps_clean, lidx)
        point = extract_emb_at(caps_clean, lidx, person_sx, person_sy)
        baseline_emb[lname] = {"gap": gap, "point": point}

    print(f"  Baseline embeddings extracted at {list(DETECTION_LAYERS.keys())}")

    # ============================================================
    # 3. Test all patches + comparison patches
    # ============================================================
    print("\n--- Testing patches through YOLOv3 forward pass ---")

    results = {}

    # Test fractal patches
    for pname, (patch_rgb, mask) in patches.items():
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            point = extract_emb_at(caps_mod, lidx, person_sx, person_sy)

            # Cosine similarity to baseline
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_emb[lname]["gap"].unsqueeze(0))[0])
            cos_point = float(F.cosine_similarity(
                point.unsqueeze(0), baseline_emb[lname]["point"].unsqueeze(0))[0])

            # L2 shift
            l2_gap = float(torch.norm(gap - baseline_emb[lname]["gap"]).item())
            l2_point = float(torch.norm(point - baseline_emb[lname]["point"]).item())

            # Raw L2 norm
            raw_l2 = float(torch.norm(gap).item())

            # Person signal overlap: how much does the patch overlap with person-encoding channels
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)
            ps = person_signal[lname]
            overlap = float(F.cosine_similarity(
                delta.flatten().unsqueeze(0),
                ps.flatten().unsqueeze(0))[0])

            metrics[lname] = {
                "cos_gap": cos_gap, "cos_point": cos_point,
                "l2_shift_gap": l2_gap, "l2_shift_point": l2_point,
                "raw_l2_gap": raw_l2, "person_overlap": overlap,
            }

        results[pname] = metrics
        print(f"  {pname}:")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} cos_pt={m['cos_point']:.4f} "
                  f"l2_shift={m['l2_shift_gap']:.3f} overlap={m['person_overlap']:.4f}")

    # Test comparison patches (poison, suppress) if they exist
    comparison_patches = []
    for pname, ppath in [("poison", POISON_PATCH), ("suppress", SUPPRESS_PATCH)]:
        if os.path.exists(ppath):
            comparison_patches.append((pname, ppath))

    for pname, ppath in comparison_patches:
        patch_arr, patch_mask = load_patch_image(ppath, H, W, patch_cx, patch_cy, outer_size)
        arr_mod = apply_patch_to_image(arr_w, patch_arr, patch_mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            point = extract_emb_at(caps_mod, lidx, person_sx, person_sy)
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_emb[lname]["gap"].unsqueeze(0))[0])
            cos_point = float(F.cosine_similarity(
                point.unsqueeze(0), baseline_emb[lname]["point"].unsqueeze(0))[0])
            l2_gap = float(torch.norm(gap - baseline_emb[lname]["gap"]).item())
            l2_point = float(torch.norm(point - baseline_emb[lname]["point"]).item())
            raw_l2 = float(torch.norm(gap).item())
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)
            ps = person_signal[lname]
            overlap = float(F.cosine_similarity(
                delta.flatten().unsqueeze(0),
                ps.flatten().unsqueeze(0))[0])
            metrics[lname] = {
                "cos_gap": cos_gap, "cos_point": cos_point,
                "l2_shift_gap": l2_gap, "l2_shift_point": l2_point,
                "raw_l2_gap": raw_l2, "person_overlap": overlap,
            }
        results[pname] = metrics
        print(f"  {pname} (comparison):")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} cos_pt={m['cos_point']:.4f} "
                  f"l2_shift={m['l2_shift_gap']:.3f} overlap={m['person_overlap']:.4f}")

    # ============================================================
    # 4. Per-channel delta analysis (top channels corrupted)
    # ============================================================
    print("\n--- Per-channel delta analysis ---")

    for pname in list(results.keys()):
        if pname.startswith("fractal"):
            patch_rgb, mask = patches[pname]
        else:
            continue  # only detailed analysis for fractal patches

        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        for lname, lidx in DETECTION_LAYERS.items():
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)  # (C, fH, fW)
            # Average over spatial dims to get per-channel impact
            channel_delta = delta.abs().mean(dim=[1, 2]).cpu().numpy()
            top_idx = np.argsort(channel_delta)[-20:][::-1]

            # Save channel deltas
            csv_path = f"{OUT}/channel_deltas_{pname}_{lname}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["channel", "abs_delta", "person_signal_norm"])
                ps = person_signal[lname]
                ps_per_channel = ps.abs().mean(dim=[1, 2]).cpu().numpy()
                for idx in range(len(channel_delta)):
                    w.writerow([idx, f"{channel_delta[idx]:.6f}",
                                f"{ps_per_channel[idx]:.6f}"])

            print(f"  {pname} {lname} top 5 channels: "
                  f"{[(int(i), round(float(channel_delta[i]), 4)) for i in top_idx[:5]]}")

    # ============================================================
    # 5. Generate print-ready version (3600x4800 300dpi)
    # ============================================================
    print("\n--- Generating print-ready version ---")

    PRINT_W, PRINT_H = 3600, 4800
    print_cx, print_cy = PRINT_W // 2, int(PRINT_H * 0.45)
    # Scale outer size proportionally: 200px on 416 -> ~1730px on 3600
    print_outer = int(200 * PRINT_W / IS * 3.5)  # larger for print
    print_outer = min(print_outer, int(PRINT_W * 0.5))

    for depth in [4, 5]:
        patch_rgb, mask = generate_fractal_patch_rgb(
            PRINT_H, PRINT_W, print_cx, print_cy, print_outer,
            max_depth=depth, base_amp=0.5, add_broadband=True
        )
        patch_uint8 = (patch_rgb * 255).astype(np.uint8)
        out_path = f"{OUT}/fractal_d{depth}_print_3600x4800_300dpi.png"
        Image.fromarray(patch_uint8).save(out_path)
        print(f"  Saved: {out_path} ({print_outer}px triangle)")

        # Also save just the patch crop (no background)
        v0, v1, v2 = triangle_vertices(print_cx, print_cy, print_outer, "up")
        x1 = int(min(v0[0], v1[0], v2[0]))
        x2 = int(max(v0[0], v1[0], v2[0]))
        y1 = int(min(v0[1], v1[1], v2[1]))
        y2 = int(max(v0[1], v1[1], v2[1]))
        crop = patch_uint8[y1:y2, x1:x2]
        Image.fromarray(crop).save(f"{OUT}/fractal_d{depth}_crop.png")

    # ============================================================
    # 6. Comparison plot
    # ============================================================
    print("\n--- Generating comparison plot ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    layers = list(DETECTION_LAYERS.keys())
    patch_names = list(results.keys())
    x = np.arange(len(patch_names))
    width = 0.25

    for row, metric in enumerate(["cos_gap", "l2_shift_gap"]):
        for col, lname in enumerate(layers):
            ax = axes[row][col]
            vals = [results[p][lname][metric] for p in patch_names]
            colors = ["#2196F3" if "fractal" in p else "#FF5722" if "poison" in p else "#4CAF50"
                      for p in patch_names]
            ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(patch_names, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{lname} — {metric}")
            ax.grid(True, alpha=0.3, axis="y")
            if metric == "cos_gap":
                ax.set_ylim(0, 1.0)
                ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
                ax.legend(fontsize=7)

    plt.suptitle("Fractal Patch vs Poison vs Suppress — YOLOv3 Embedding Impact", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 7. Save results JSON + CSV
    # ============================================================
    json_path = f"{OUT}/fractal_patch_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    csv_path = f"{OUT}/fractal_patch_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "layer", "cos_gap", "cos_point", "l2_shift_gap",
                     "l2_shift_point", "raw_l2_gap", "person_overlap"])
        for pname in results:
            for lname in results[pname]:
                m = results[pname][lname]
                w.writerow([pname, lname, f"{m['cos_gap']:.6f}", f"{m['cos_point']:.6f}",
                            f"{m['l2_shift_gap']:.6f}", f"{m['l2_shift_point']:.6f}",
                            f"{m['raw_l2_gap']:.6f}", f"{m['person_overlap']:.6f}"])
    print(f"Saved CSV: {csv_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nGenerated fractal patches at depths 3, 4, 5")
    print(f"Frequency bands: k=3,9,27,81,243,729 (powers of 3)")
    print(f"Print-ready: {OUT}/fractal_d4_print_3600x4800_300dpi.png")
    print(f"             {OUT}/fractal_d5_print_3600x4800_300dpi.png")
    print(f"\nEmbedding shift comparison (GAP cosine, lower = more corruption):")
    print(f"{'Patch':<20} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<20} "
              f"{r['L81_52x52']['cos_gap']:<15.4f} "
              f"{r['L93_26x26']['cos_gap']:<15.4f} "
              f"{r['L105_13x13']['cos_gap']:<15.4f}")
    print(f"\nL2 shift comparison (higher = more corruption):")
    print(f"{'Patch':<20} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<20} "
              f"{r['L81_52x52']['l2_shift_gap']:<15.3f} "
              f"{r['L93_26x26']['l2_shift_gap']:<15.3f} "
              f"{r['L105_13x13']['l2_shift_gap']:<15.3f}")
    print(f"\nPerson signal overlap (higher = more aligned with person channels):")
    print(f"{'Patch':<20} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<20} "
              f"{r['L81_52x52']['person_overlap']:<15.4f} "
              f"{r['L93_26x26']['person_overlap']:<15.4f} "
              f"{r['L105_13x13']['person_overlap']:<15.4f}")

    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
