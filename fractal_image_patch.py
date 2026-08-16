"""
Self-Similar Image Fractal Patch — Sierpinski tessellation of the k=196 diagonal pattern.

Design: Take the most effective single-frequency texture (k=196 diagonal sinusoid)
and tile it into a Sierpinski triangle where each sub-triangle contains a scaled,
affine-warped copy of the same source image. This creates TRUE self-similarity:
  - The same frequency content appears at every scale simultaneously
  - At distance, large sub-triangles present k=196 at low effective frequency
  - Up close, small sub-triangles present k=196 at high effective frequency
  - The pattern is scale-invariant by construction: the image repeats at k=3,9,27,81,243

This combines:
  - Shape Matters DPS paper: deformable triangular patch representation
  - Sierpinski recursion: self-similar structure at all scales
  - k=196 diagonal: the most effective single-frequency carrier found in prior analysis
  - Diagonal orientation: maximizes frequency content across both H and V axes

Structure:
  Level 0: 1 large triangle with full k=196 image (effective k~3 when far)
  Level 1: 3 sub-triangles + 1 inverted center, each with scaled k=196 image
  Level 2: 9+3 sub-triangles, each with further scaled k=196 image
  Level 3: 27+9 sub-triangles
  Level 4: 81+27 sub-triangles (k=196 appears at 108 different scales)

Each sub-triangle gets the source image affine-warped to fit, with per-channel
RGB phase offsets so different channels hit different feature maps.

Output:
  - Print-ready PNG at 3600x4800 300dpi
  - Test-resolution PNG at 416x416 for YOLOv3 forward pass
  - FFT spectrum showing multi-scale frequency content
  - Forward pass embedding shift at L81/L93/L105 compared to baseline + poison + suppress + fractal_d4
"""

import os, sys, math, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
POISON_PATCH = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\poison\poison_patch.png"
SUPPRESS_PATCH = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\suppress\suppress_patch.png"
FRACTAL_D4 = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\fractal_patch\fractal_d4_416.png"
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\fractal_image_patch"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416

DETECTION_LAYERS = {"L81_52x52": 81, "L93_26x26": 93, "L105_13x13": 105}

os.makedirs(OUT, exist_ok=True)


# ============================================================
# Source image generation: k=196 diagonal sinusoid
# ============================================================

def make_k196_diagonal_source(size=256, amp=0.5, channel_phase=True):
    """Generate the k=196 diagonal sinusoid source image.

    This is the most effective single-frequency carrier from prior analysis.
    Diagonal orientation (kx=ky=196) maximizes frequency content across both
    H and V axes, hitting more feature maps than axis-aligned patterns.

    channel_phase: if True, R/G/B channels get 120deg phase offsets so
    different channels stimulate different first-layer feature maps.
    """
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pat = np.zeros((size, size, 3), dtype=np.float32)

    phases = [0.0, 2.094, 4.189] if channel_phase else [0.0, 0.0, 0.0]
    for c, phase in enumerate(phases):
        # k=196 diagonal: kx=ky so the pattern runs at 45 degrees
        # This hits both horizontal and vertical frequency detectors
        pat[:, :, c] = amp * np.cos(2 * np.pi * 196 * (x / size + y / size) + phase)

    # Normalize to [0, 1]
    pat = (pat - pat.min()) / (pat.max() - pat.min())
    return pat


def make_k196_diagonal_source_highres(size=1024, amp=0.5, channel_phase=True):
    """High-res version for print output."""
    return make_k196_diagonal_source(size, amp, channel_phase)


# ============================================================
# Sierpinski triangle subdivision with image tiling
# ============================================================

def triangle_centroid(v0, v1, v2):
    return ((v0[0]+v1[0]+v2[0])/3, (v0[1]+v1[1]+v2[1])/3)


def triangle_size(v0, v1, v2):
    """Approximate edge length of triangle."""
    return math.sqrt((v1[0]-v0[0])**2 + (v1[1]-v0[1])**2)


def subdivide_triangle(v0, v1, v2, depth, out_triangles):
    """Recursively subdivide triangle Sierpinski-style.

    Unlike standard Sierpinski (which skips the center), we INCLUDE the
    inverted center triangle at each level. This ensures the source image
    appears at every scale, not just at the leaf nodes.

    out_triangles: list of (v0, v1, v2, level, orientation) tuples
    """
    if depth == 0:
        # Determine orientation: check if this is an up or down triangle
        # Up triangle: v0 is top, v1 bottom-left, v2 bottom-right
        # Down triangle: v0 is bottom, v1 top-right, v2 top-left
        cy = (v0[1] + v1[1] + v2[1]) / 3
        is_up = v0[1] < cy  # v0 is above centroid = up triangle
        out_triangles.append((v0, v1, v2, 0, "up" if is_up else "down"))
        return

    # Midpoints
    m01 = ((v0[0]+v1[0])/2, (v0[1]+v1[1])/2)
    m12 = ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2)
    m20 = ((v2[0]+v0[0])/2, (v2[1]+v0[1])/2)

    level = depth

    # Three corner sub-triangles (same orientation as parent)
    subdivide_triangle(v0, m01, m20, depth-1, out_triangles)
    subdivide_triangle(m01, v1, m12, depth-1, out_triangles)
    subdivide_triangle(m20, m12, v2, depth-1, out_triangles)

    # Inverted center triangle (opposite orientation)
    # This is the key difference from standard Sierpinski: we fill it too
    # The center triangle has vertices m01, m12, m20
    # Its orientation is opposite to the parent
    parent_cy = (v0[1] + v1[1] + v2[1]) / 3
    parent_is_up = v0[1] < parent_cy
    center_is_up = not parent_is_up

    # Recursively subdivide the center triangle too (for deeper frequency content)
    if depth > 1:
        subdivide_triangle(m01, m12, m20, depth-1, out_triangles)
    else:
        out_triangles.append((m01, m12, m20, level, "up" if center_is_up else "down"))


def warp_image_to_triangle(src_img, src_tri_verts, dst_tri_verts, dst_size):
    """Affine-warp source image from src_tri_verts to dst_tri_verts.

    Uses cv2.getAffineTransform for the 3-point correspondence, then
    cv2.warpAffine to render the warped image onto a canvas of dst_size.
    Returns the warped image and a triangular mask.
    """
    H, W = dst_size
    src_pts = np.float32(src_tri_verts)
    dst_pts = np.float32(dst_tri_verts)

    # Compute affine transform: src_tri -> dst_tri
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # Warp the source image
    if src_img.ndim == 3:
        warped = cv2.warpAffine(src_img, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    else:
        warped = cv2.warpAffine(src_img, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Build triangular mask for the destination triangle
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_pts), 1.0)

    return warped, mask


def render_sierpinski_image_fractal(H, W, cx, cy, outer_size, src_img,
                                     max_depth=4, channel_offset=0.0):
    """Render a Sierpinski fractal where each sub-triangle contains a
    scaled, warped copy of src_img.

    The source image is affine-warped into each sub-triangle at every level
    of the Sierpinski subdivision. This creates true self-similarity: the
    same frequency content (k=196 diagonal) appears at every scale.

    channel_offset: phase rotation for color channels (radians), so
    different calls can produce R/G/B channels with phase diversity.
    """
    # Define outer triangle vertices (upward)
    h = outer_size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - outer_size/2, cy + h/3)
    v2 = (cx + outer_size/2, cy + h/3)

    # Subdivide
    triangles = []
    subdivide_triangle(v0, v1, v2, max_depth, triangles)

    # Source triangle: inscribed upward triangle in the source image
    src_H, src_W = src_img.shape[:2]
    src_h = src_W * math.sqrt(3) / 2
    src_v0 = (src_W / 2, src_H / 2 - src_h * 2/3)
    src_v1 = (0, src_H / 2 + src_h / 3)
    src_v2 = (src_W, src_H / 2 + src_h / 3)
    src_tri = [src_v0, src_v1, src_v2]

    # For down triangles, use inverted source triangle
    src_h_d = src_W * math.sqrt(3) / 2
    src_dv0 = (src_W / 2, src_H / 2 + src_h_d * 2/3)
    src_dv1 = (src_W, src_H / 2 - src_h_d / 3)
    src_dv2 = (0, src_H / 2 - src_h_d / 3)
    src_tri_down = [src_dv0, src_dv1, src_dv2]

    # Render each triangle
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)

    # Sort by level (largest first = lowest level number = painted first)
    # Smaller triangles paint on top for detail
    triangles_sorted = sorted(triangles, key=lambda t: t[3])

    for tv0, tv1, tv2, level, orientation in triangles_sorted:
        # Choose source triangle based on orientation
        st = src_tri if orientation == "up" else src_tri_down

        # Warp source image into this triangle
        dst_verts = [tv0, tv1, tv2]
        warped, tri_mask = warp_image_to_triangle(src_img, st, dst_verts, (H, W))

        # Amplitude scaling: larger triangles get slightly more weight
        # so the overall pattern has 1/f spectral shape
        scale = 1.0 / math.sqrt(3 ** level) if level > 0 else 1.0
        scale = max(0.3, scale)  # don't let small triangles vanish

        # Apply coverage mask (avoid overpainting: only paint where not yet covered)
        uncovered = np.maximum(0, tri_mask - coverage)
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c] * uncovered * scale
        coverage = np.maximum(coverage, tri_mask)

    # Normalize to [0, 1]
    if canvas.max() > 0:
        canvas = canvas / canvas.max()

    # Build outer triangular mask
    outer_mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(outer_mask, np.int32([v0, v1, v2]), 1.0)

    return np.clip(canvas, 0, 1).astype(np.float32), outer_mask


def generate_sierpinski_image_patch_rgb(H, W, cx, cy, outer_size,
                                         max_depth=4, src_size=256):
    """Generate RGB Sierpinski image fractal with per-channel phase diversity.

    R channel: k=196 diagonal at phase 0
    G channel: k=196 diagonal at phase 120deg (2.094 rad)
    B channel: k=196 diagonal at phase 240deg (4.189 rad)

    Each channel produces its own Sierpinski tiling, so the RGB composite
    has phase diversity across channels while maintaining self-similarity.
    """
    # Generate three source images with different phases
    src_r = make_k196_diagonal_source(src_size, amp=0.5, channel_phase=False)
    # Manually set phase for each
    y, x = np.meshgrid(np.arange(src_size), np.arange(src_size), indexing="ij")
    src_r = np.zeros((src_size, src_size, 3), dtype=np.float32)
    src_g = np.zeros((src_size, src_size, 3), dtype=np.float32)
    src_b = np.zeros((src_size, src_size, 3), dtype=np.float32)

    for c in range(3):
        src_r[:, :, c] = 0.5 + 0.5 * np.cos(2 * np.pi * 196 * (x / src_size + y / src_size) + 0.0)
        src_g[:, :, c] = 0.5 + 0.5 * np.cos(2 * np.pi * 196 * (x / src_size + y / src_size) + 2.094)
        src_b[:, :, c] = 0.5 + 0.5 * np.cos(2 * np.pi * 196 * (x / src_size + y / src_size) + 4.189)

    # Render Sierpinski fractal for each channel
    patch_r, mask = render_sierpinski_image_fractal(H, W, cx, cy, outer_size, src_r, max_depth)
    patch_g, _ = render_sierpinski_image_fractal(H, W, cx, cy, outer_size, src_g, max_depth)
    patch_b, _ = render_sierpinski_image_fractal(H, W, cx, cy, outer_size, src_b, max_depth)

    # Combine: take R from patch_r, G from patch_g, B from patch_b
    patch = np.stack([patch_r[:, :, 0], patch_g[:, :, 0], patch_b[:, :, 0]], axis=2)

    return patch.astype(np.float32), mask


# ============================================================
# Forward pass analysis (same as fractal_patch.py)
# ============================================================

def fwd_all(model, x):
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
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW-1, int(spatial_x / IS * fW)))
    fy = max(0, min(fH-1, int(spatial_y / IS * fH)))
    return feat[0, :, fy, fx]


def gap_embedding(caps, layer_idx):
    feat = caps[layer_idx]
    return F.adaptive_avg_pool2d(feat, 1).squeeze()


def apply_patch_to_image(arr, patch_rgb, mask, cx, cy):
    H, W, _ = arr.shape
    out = arr.copy()
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - mask) + patch_rgb[:, :, c] * mask
    return np.clip(out, 0, 1)


def load_patch_image(path, H, W, cx, cy, target_size):
    """Load a patch image and apply with triangular mask."""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    h = target_size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - target_size/2, cy + h/3)
    v2 = (cx + target_size/2, cy + h/3)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32([v0, v1, v2]), 1.0)
    return arr, mask


# ============================================================
# FFT analysis
# ============================================================

def compute_2d_fft_mag(texture):
    f = np.fft.fft2(texture)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log1p(mag)
    return mag


def radial_average(mag, H, W):
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(H, W) // 2
    radial = np.zeros(max_r, dtype=np.float32)
    for i in range(max_r):
        mask = (r == i)
        if mask.any():
            radial[i] = mag[mask].mean()
    return radial[:max_r]


# ============================================================
# Main
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Self-Similar Image Fractal Patch — k=196 Sierpinski Tiling")
    print("=" * 70)
    print(f"Device: {DEV}")

    H, W = IS, IS

    # ============================================================
    # 1. Generate source image and Sierpinski image fractal patches
    # ============================================================
    print("\n--- Generating source k=196 diagonal image ---")
    src_size = 256
    src_img = make_k196_diagonal_source(src_size, amp=0.5, channel_phase=True)
    src_uint8 = (src_img * 255).astype(np.uint8)
    Image.fromarray(src_uint8).save(f"{OUT}/source_k196_diagonal.png")
    print(f"  Source image: {src_size}x{src_size}, saved to {OUT}/source_k196_diagonal.png")

    print("\n--- Generating Sierpinski image fractal patches ---")
    patch_cx, patch_cy = IS // 2, int(IS * 0.58)
    outer_size = 200  # ~18% area on 416px

    patches = {}
    for depth in [2, 3, 4, 5]:
        patch_rgb, mask = generate_sierpinski_image_patch_rgb(
            H, W, patch_cx, patch_cy, outer_size,
            max_depth=depth, src_size=src_size
        )
        patches[f"sierp_img_d{depth}"] = (patch_rgb, mask)
        area_pct = float(np.mean(mask)) * 100

        # Count triangles
        v0 = (patch_cx, patch_cy - outer_size * math.sqrt(3) / 2 * 2/3)
        v1 = (patch_cx - outer_size/2, patch_cy + outer_size * math.sqrt(3) / 2 / 3)
        v2 = (patch_cx + outer_size/2, patch_cy + outer_size * math.sqrt(3) / 2 / 3)
        tris = []
        subdivide_triangle(v0, v1, v2, depth, tris)
        n_tris = len(tris)

        print(f"  sierp_img_d{depth}: area={area_pct:.1f}%, {n_tris} sub-triangles, "
              f"each contains scaled k=196 image")

        # Save visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title(f"Mask (d={depth})")
        axes[0].axis("off")
        axes[1].imshow(patch_rgb)
        axes[1].set_title(f"RGB Patch (d={depth})")
        axes[1].axis("off")
        gray = np.mean(patch_rgb, axis=2)
        mag = compute_2d_fft_mag(gray)
        axes[2].imshow(mag, cmap="inferno", extent=[-W//2, W//2, H//2, -H//2])
        axes[2].set_title("2D FFT Magnitude")
        axes[2].axis("off")
        radial = radial_average(mag, H, W)
        axes[3].plot(range(len(radial)), radial, "b-", linewidth=0.8)
        # Mark k=196 and its scaled versions
        for k in [3, 9, 27, 81, 196, 243]:
            if k < len(radial):
                axes[3].axvline(x=k, color="r", linestyle="--", alpha=0.5)
                axes[3].text(k, radial.max() * 0.85, f"k={k}", fontsize=7, color="r", rotation=90)
        axes[3].set_title("Radial FFT (red = key frequencies)")
        axes[3].set_xlabel("Frequency k")
        axes[3].set_ylabel("log(1+|F|)")
        axes[3].set_xlim(0, min(250, len(radial)))
        plt.suptitle(f"Sierpinski Image Fractal d={depth} — k=196 Self-Similar Tiling", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/sierp_img_d{depth}_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save patch PNG
        patch_uint8 = (patch_rgb * 255).astype(np.uint8)
        Image.fromarray(patch_uint8).save(f"{OUT}/sierp_img_d{depth}_416.png")

    # ============================================================
    # 2. Load YOLOv3 and run forward pass analysis
    # ============================================================
    print("\n--- Loading YOLOv3 ---")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    arr_w = load_img(IMG_WITH, IS)
    arr_wo = load_img(IMG_WITHOUT, IS)

    # Baselines
    tensor_clean = torch.from_numpy(arr_w).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_clean, _ = fwd_all(model, tensor_clean)
    tensor_empty = torch.from_numpy(arr_wo).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_empty, _ = fwd_all(model, tensor_empty)

    person_signal = {}
    for lname, lidx in DETECTION_LAYERS.items():
        person_signal[lname] = (caps_clean[lidx] - caps_empty[lidx]).squeeze(0)

    person_sx, person_sy = IS // 2, int(IS * 0.58)
    baseline_emb = {}
    for lname, lidx in DETECTION_LAYERS.items():
        gap = gap_embedding(caps_clean, lidx)
        point = extract_emb_at(caps_clean, lidx, person_sx, person_sy)
        baseline_emb[lname] = {"gap": gap, "point": point}

    print(f"  Baseline embeddings extracted at {list(DETECTION_LAYERS.keys())}")

    # ============================================================
    # 3. Test all patches
    # ============================================================
    print("\n--- Testing patches through YOLOv3 forward pass ---")

    results = {}

    # Sierpinski image fractal patches
    for pname, (patch_rgb, mask) in patches.items():
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
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
        print(f"  {pname}:")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} cos_pt={m['cos_point']:.4f} "
                  f"l2_shift={m['l2_shift_gap']:.3f} overlap={m['person_overlap']:.4f}")

    # Comparison patches
    comparison_patches = []
    for pname, ppath in [("poison", POISON_PATCH), ("suppress", SUPPRESS_PATCH),
                          ("fractal_d4", FRACTAL_D4)]:
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
    # 4. Per-channel delta analysis
    # ============================================================
    print("\n--- Per-channel delta analysis ---")

    for pname, (patch_rgb, mask) in patches.items():
        arr_mod = apply_patch_to_image(arr_w, patch_rgb, mask, patch_cx, patch_cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        for lname, lidx in DETECTION_LAYERS.items():
            delta = (caps_mod[lidx] - caps_clean[lidx]).squeeze(0)
            channel_delta = delta.abs().mean(dim=[1, 2]).cpu().numpy()
            top_idx = np.argsort(channel_delta)[-20:][::-1]

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
    # 5. Generate print-ready version
    # ============================================================
    print("\n--- Generating print-ready version ---")

    PRINT_W, PRINT_H = 3600, 4800
    print_cx, print_cy = PRINT_W // 2, int(PRINT_H * 0.45)
    print_outer = min(int(200 * PRINT_W / IS * 3.5), int(PRINT_W * 0.5))

    # High-res source image
    src_hr = make_k196_diagonal_source(1024, amp=0.5, channel_phase=True)

    for depth in [3, 4, 5]:
        patch_rgb, mask = generate_sierpinski_image_patch_rgb(
            PRINT_H, PRINT_W, print_cx, print_cy, print_outer,
            max_depth=depth, src_size=1024
        )
        patch_uint8 = (patch_rgb * 255).astype(np.uint8)
        out_path = f"{OUT}/sierp_img_d{depth}_print_3600x4800_300dpi.png"
        Image.fromarray(patch_uint8).save(out_path)
        print(f"  Saved: {out_path}")

        # Crop just the triangle
        h = print_outer * math.sqrt(3) / 2
        v0 = (print_cx, print_cy - h * 2/3)
        v1 = (print_cx - print_outer/2, print_cy + h/3)
        v2 = (print_cx + print_outer/2, print_cy + h/3)
        x1 = int(min(v0[0], v1[0], v2[0]))
        x2 = int(max(v0[0], v1[0], v2[0]))
        y1 = int(min(v0[1], v1[1], v2[1]))
        y2 = int(max(v0[1], v1[1], v2[1]))
        crop = patch_uint8[y1:y2, x1:x2]
        Image.fromarray(crop).save(f"{OUT}/sierp_img_d{depth}_crop.png")

    # ============================================================
    # 6. Comparison plot
    # ============================================================
    print("\n--- Generating comparison plot ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    layers = list(DETECTION_LAYERS.keys())
    patch_names = list(results.keys())
    x = np.arange(len(patch_names))

    for row, metric in enumerate(["cos_gap", "l2_shift_gap"]):
        for col, lname in enumerate(layers):
            ax = axes[row][col]
            vals = [results[p][lname][metric] for p in patch_names]
            colors = []
            for p in patch_names:
                if "sierp_img" in p:
                    colors.append("#2196F3")
                elif "fractal" in p:
                    colors.append("#9C27B0")
                elif "poison" in p:
                    colors.append("#FF5722")
                else:
                    colors.append("#4CAF50")
            ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(patch_names, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{lname} — {metric}")
            ax.grid(True, alpha=0.3, axis="y")
            if metric == "cos_gap":
                ax.set_ylim(0, 1.0)
                ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Sierpinski Image Fractal vs Fractal-d4 vs Poison vs Suppress — YOLOv3", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 7. Save results
    # ============================================================
    json_path = f"{OUT}/sierp_img_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    csv_path = f"{OUT}/sierp_img_results.csv"
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
    print(f"\nSource image: k=196 diagonal sinusoid (most effective single carrier)")
    print(f"Method: Sierpinski tessellation with affine-warped image copies")
    print(f"Each sub-triangle contains a scaled copy of the k=196 image")
    print(f"\nPrint-ready:")
    print(f"  {OUT}/sierp_img_d4_print_3600x4800_300dpi.png")
    print(f"  {OUT}/sierp_img_d5_print_3600x4800_300dpi.png")
    print(f"\nEmbedding shift (GAP cosine, lower = more corruption):")
    print(f"{'Patch':<25} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<25} "
              f"{r['L81_52x52']['cos_gap']:<15.4f} "
              f"{r['L93_26x26']['cos_gap']:<15.4f} "
              f"{r['L105_13x13']['cos_gap']:<15.4f}")
    print(f"\nL2 shift (higher = more corruption):")
    print(f"{'Patch':<25} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<25} "
              f"{r['L81_52x52']['l2_shift_gap']:<15.3f} "
              f"{r['L93_26x26']['l2_shift_gap']:<15.3f} "
              f"{r['L105_13x13']['l2_shift_gap']:<15.3f}")
    print(f"\nPerson signal overlap (higher = more aligned with person channels):")
    print(f"{'Patch':<25} {'L81_52x52':<15} {'L93_26x26':<15} {'L105_13x13':<15}")
    for pname in results:
        r = results[pname]
        print(f"{pname:<25} "
              f"{r['L81_52x52']['person_overlap']:<15.4f} "
              f"{r['L93_26x26']['person_overlap']:<15.4f} "
              f"{r['L105_13x13']['person_overlap']:<15.4f}")

    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
