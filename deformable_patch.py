"""
Shape Matters deformable patch attack — full implementation.

R rays from center, lengths r = {r1,...,rR} are learnable parameters.
Differentiable polygon mask via sigmoid-signed-distance product.
Joint shape + texture optimization with area constraint ps.

Loss:
  L = L_adv                    if area <= ps
  L = L_adv + beta * mean(M)   if area > ps

L_adv = embedding L2 shift (maximize corruption) for poison
L_adv = detection confidence suppression for suppress

Mask sharpening: when area reaches target ratio s, binarize mask,
then fine-tune texture to sharpened shape.

Texture: initialized from Sierpinski-tiled trained patch (our best).
Shape: R=24 rays, optimized via gradient descent.
"""
import os, sys, math, json, csv, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "PyTorch-YOLOv3"))
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

from fractal_patch import (
    fwd_all, load_img, extract_emb_at, gap_embedding,
    apply_patch_to_image, compute_2d_fft_mag, radial_average,
    DETECTION_LAYERS
)

# Forward pass that retains gradients (for optimization)
# fwd_all in fractal_patch.py detaches conv outputs, which kills backprop.
# This version keeps the graph alive so we can backprop through the model
# to the input image and patch parameters.
def fwd_all_grad(model, x):
    """Forward capture all conv layer outputs — gradient-preserving."""
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
            caps[i] = x  # NO detach — keep gradient graph
        los.append(x)
    return caps, x

from fractal_image_patch import subdivide_triangle, warp_image_to_triangle

CFG = os.path.join(BASE, "PyTorch-YOLOv3", "config", "yolov3.cfg")
WTS = os.path.join(BASE, "yolov3.weights")
IMG_WITH = os.path.join(BASE, "withhuman.png")
IMG_WITHOUT = os.path.join(BASE, "withouthuman.png")
POISON = os.path.join(BASE, "outputs_clothing", "forward_analysis", "patch_pipeline", "dual_optim", "poison", "poison_patch.png")
SUPPRESS = os.path.join(BASE, "outputs_clothing", "forward_analysis", "patch_pipeline", "dual_optim", "suppress", "suppress_patch.png")
OUT = os.path.join(BASE, "outputs_clothing", "forward_analysis", "deformable_patch_rings")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Differentiable Deformable Polygon Mask (Shape Matters Eq. 4)
# ============================================================

def differentiable_polygon_mask(ray_lengths, cx, cy, H, W, lam=-100.0, n_repeats=1):
    """
    Build a differentiable polygon mask from R base ray lengths, optionally
    replicated n_repeats times around the center for rotational symmetry.

    With n_repeats=1: standard R-vertex polygon (Shape Matters formulation).
    With n_repeats>1: base R rays are repeated n_repeats times, creating
      R*n_repeats total vertices with n_repeats-fold rotational symmetry.
      E.g., R=3, n_repeats=24 -> 72-vertex star/flower with 24-fold symmetry.

    Only the R base ray lengths are learnable — the replication is deterministic,
    so gradients flow to the base parameters only.

    Args:
        ray_lengths: (R,) tensor of base ray lengths, requires_grad=True
        cx, cy: center coordinates (floats)
        H, W: image size
        lam: sharpness parameter (negative for inside-positive)
        n_repeats: number of times to replicate base polygon around center

    Returns:
        mask: (H, W) tensor in [0, 1], differentiable w.r.t. ray_lengths
    """
    R_base = ray_lengths.shape[0]
    R_total = R_base * n_repeats
    dtheta = 2.0 * math.pi / R_total

    # Replicate base ray lengths: [r0, r1, r2] -> [r0, r1, r2, r0, r1, r2, ...]
    if n_repeats > 1:
        # Expand ray_lengths to R_total by tiling
        idx = torch.arange(R_total, device=ray_lengths.device) % R_base
        ray_expanded = ray_lengths[idx]
    else:
        ray_expanded = ray_lengths

    R = R_total

    # Vertex positions: (R_total, 2) — use expanded ray lengths
    angles = torch.arange(R, device=ray_lengths.device, dtype=ray_lengths.dtype) * dtheta
    vx = cx + ray_expanded * torch.cos(angles)
    vy = cy + ray_expanded * torch.sin(angles)

    # Pixel grid: (H, W, 2)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=ray_lengths.device, dtype=ray_lengths.dtype),
        torch.arange(W, device=ray_lengths.device, dtype=ray_lengths.dtype),
        indexing="ij"
    )
    px = xx  # (H, W)
    py = yy  # (H, W)

    # For each edge i: from vertex i to vertex (i+1) % R
    # Edge vector: (vx_next - vx_i, vy_next - vy_i)
    # Inward normal (for CCW polygon): (-dy, dx) normalized
    # Signed distance = dot(pixel - vertex_i, normal)
    # Inside if signed_dist > 0

    mask = torch.ones((H, W), device=ray_lengths.device, dtype=ray_lengths.dtype)

    for i in range(R):
        j = (i + 1) % R
        # Edge from vertex i to vertex j
        ex = vx[j] - vx[i]
        ey = vy[j] - vy[i]
        # Edge length
        elen = torch.sqrt(ex * ex + ey * ey + 1e-8)
        # Inward normal: rotate edge 90 degrees (for CCW ordering, inward = left normal)
        # Normal = (-ey, ex) / elen  (points inward for CCW)
        nx = -ey / elen
        ny = ex / elen
        # Vector from vertex i to pixel
        dx = px - vx[i]
        dy = py - vy[i]
        # Signed distance to edge line
        sd = dx * nx + dy * ny
        # Sigmoid activation: Phi(sd) = sigmoid(lam * sd)
        # When sd > 0 (inside), sigmoid(lam * sd) -> 1 (since lam < 0, lam*sd < 0, sigmoid -> 0)
        # Wait — lam is negative. sigmoid(negative * positive) = sigmoid(negative) -> 0
        # We want inside -> 1, so use sigmoid(-lam * sd) = sigmoid(|lam| * sd)
        # Or equivalently: Phi = (tanh(lam * (sd - threshold)) + 1) / 2
        # Paper uses Phi(x) = (tanh(lambda*(x-1)) + 1) / 2 with lambda = -100
        # This gives Phi(0) = (tanh(-100*(-1)) + 1)/2 = (tanh(100)+1)/2 ~ 1
        # Phi(1) = (tanh(0)+1)/2 = 0.5
        # Phi(2) = (tanh(-100)+1)/2 ~ 0
        # So Phi maps: x<1 -> ~1, x=1 -> 0.5, x>1 -> ~0
        # This is a "inside if x < 1" convention where x is normalized distance
        
        # Use the paper's formulation: Phi(sd_normalized) where sd_normalized = 1 - sd/scale
        # Simpler: just use sigmoid(|lam| * sd) for inside=1
        phi = torch.sigmoid(-lam * sd)  # -lam is positive, so inside (sd>0) -> sigmoid(positive) -> 1
        mask = mask * phi

    return mask


def compute_area(mask):
    """Compute patch area as fraction of image."""
    return mask.mean()


# ============================================================
# Sierpinski texture generation (fixed, not optimized)
# ============================================================

def load_img_array(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def render_sierpinski_texture(H, W, cx, cy, outer_size, src_img, max_depth=3):
    """Pre-render Sierpinski-tiled texture at full image size. Returns (H,W,3) array."""
    h = outer_size * math.sqrt(3) / 2
    v0 = (cx, cy - h * 2/3)
    v1 = (cx - outer_size/2, cy + h/3)
    v2 = (cx + outer_size/2, cy + h/3)

    triangles = []
    subdivide_triangle(v0, v1, v2, max_depth, triangles)

    src_H, src_W = src_img.shape[:2]
    s_h = src_W * math.sqrt(3) / 2
    s_up = [(src_W/2, src_H/2 - s_h*2/3), (0, src_H/2 + s_h/3), (src_W, src_H/2 + s_h/3)]
    s_dn = [(src_W/2, src_H/2 + s_h*2/3), (src_W, src_H/2 - s_h/3), (0, src_H/2 - s_h/3)]

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)
    triangles.sort(key=lambda t: t[3])

    for stv0, stv1, stv2, level, orient in triangles:
        st = s_up if orient == "up" else s_dn
        warped, tri_mask = warp_image_to_triangle(src_img, st, [stv0, stv1, stv2], (H, W))
        uncovered = np.maximum(0, tri_mask - coverage)
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c] * uncovered
        coverage = np.maximum(coverage, tri_mask)

    if canvas.max() > 0:
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())
    return np.clip(canvas, 0, 1).astype(np.float32)


# ============================================================
# Sinusoid texture initialization (Path B — k196 proven baseline)
# ============================================================

def make_sinusoid_texture(H, W, kx, ky, amp=0.30, rgb_variations=True):
    """
    Generate a 3-channel sinusoid texture for patch initialization.
    Uses per-channel frequency variations to maximize spectral coverage.

    k196 proven: 77% detection reduction on YOLOv3, near-suppression on v8/v11.
    Per-channel RGB variant (k196, k167, k208) combines all three key frequencies.

    Args:
        H, W: texture size
        kx, ky: base spatial frequency (cycles across image width/height)
        amp: amplitude (0.30 matches triangular_patch_test.py)
        rgb_variations: if True, use different k per channel for broader spectral hit

    Returns:
        texture: (H, W, 3) float32 array in [0, 1]
    """
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pat = np.zeros((H, W, 3), dtype=np.float32)

    if rgb_variations:
        # Per-channel frequencies: k196 (disrupt), k167 (suppress), k208 (hallucinate)
        # Each channel carries a different key frequency — broader spectral attack
        freqs = [(kx, ky), (167, 167), (208, 208)]
    else:
        # Single frequency across all channels
        freqs = [(kx, ky)] * 3

    for c, (fx, fy) in enumerate(freqs):
        # Sinusoid: amp * cos(2*pi*(fx/W*x + fy/H*y))
        # Center at 0.5 so range is [0.5-amp, 0.5+amp]
        pat[:, :, c] = 0.5 + amp * np.cos(2 * np.pi * (fx / W * x + fy / H * y))

    return np.clip(pat, 0, 1).astype(np.float32)


def make_fractal_sinusoid_texture(H, W, cx=None, cy=None,
                                   freqs=None, amps=None, rgb_variations=True):
    """
    Multi-scale fractal composite sinusoid texture for physical scale robustness.

    Contains k=49, k=98, k=196, k=392 simultaneously. At any capture scale,
    one frequency band aligns with the network's vulnerable layers:

    | Capture Width | k=49 eff | k=98 eff | k=196 eff | k=392 eff |
    |---------------|----------|----------|-----------|-----------|
    | 416px (digital) | 49    | 98       | 196       | 392       |
    | 300px          | 35    | 71       | 141       | 283       |
    | 200px          | 24    | 47       | 94        | 188       |
    | 500px          | 59    | 118      | 235       | 471       |

    At 200px capture, k=392 maps to k=188 — within the vulnerable band [167-208].
    At 300px, k=392 maps to k=283, and k=196 maps to k=141 — k=392 is closer.
    The fractal ensures coverage across distances.

    Combines two components:
    1. Cartesian sinusoids at each frequency — provides spatial frequency content
    2. Angular rosette: sin(k * theta) — creates self-similar triangle pattern
       (triangles within triangles, matching the deformable triangle shape)

    Args:
        H, W: texture size
        cx, cy: center for angular component (defaults to image center)
        freqs: list of spatial frequencies [49, 98, 196, 392]
        amps: amplitudes per frequency (decreasing for higher k to maintain contrast)
        rgb_variations: if True, use per-channel phase shifts for broader spectral hit

    Returns:
        texture: (H, W, 3) float32 array in [0, 1]
    """
    if freqs is None:
        freqs = [49, 98, 196, 392]
    if amps is None:
        # Higher frequencies get slightly lower amplitude — keeps visual contrast
        # while ensuring all bands have sufficient energy to corrupt the network
        amps = [0.12, 0.10, 0.08, 0.06]
    if cx is None:
        cx = W / 2
    if cy is None:
        cy = H / 2

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Polar coordinates for angular rosette component
    dx = x - cx
    dy = y - cy
    theta = np.arctan2(dy, dx)  # angle from center
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_max = min(H, W) / 2
    # Radial envelope: fade at edges to avoid discontinuity at patch boundary
    r_env = np.clip(1.0 - (r / r_max) ** 2, 0, 1)

    # Per-channel phase shifts for broader spectral coverage
    if rgb_variations:
        # R: no shift, G: progressive, B: maximum — creates per-channel interference
        phase_sets = [
            [0, 0, 0, 0],
            [np.pi / 4, np.pi / 3, np.pi / 2, np.pi],
            [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ]
    else:
        phase_sets = [[0] * len(freqs)] * 3

    pat = np.zeros((H, W, 3), dtype=np.float32)

    for c in range(3):
        val = 0.5  # center at gray

        for i, (k, amp) in enumerate(zip(freqs, amps)):
            phase = phase_sets[c][i]

            # Component 1: Cartesian sinusoid (spatial frequency content)
            # Diagonal direction for better 2D coverage
            cart = np.cos(2 * np.pi * k / W * x + phase) + np.cos(2 * np.pi * k / H * y + phase * 0.7)
            cart = cart / 2  # normalize to [-1, 1]

            # Component 2: Angular rosette (self-similar triangle pattern)
            # sin(k * theta) creates k-fold rotational symmetry
            # k=3 -> triangle, k=6 -> hexagram, etc.
            # Use k//3 for angular to create triangle-compatible symmetry
            k_angular = max(3, k // 3)  # 49->16, 98->32, 196->65, 392->130
            angular = np.sin(k_angular * theta + phase)

            # Combine: 70% Cartesian (frequency targeting) + 30% angular (shape alignment)
            val += amp * (0.7 * cart + 0.3 * angular * r_env)

        pat[:, :, c] = val

    return np.clip(pat, 0, 1).astype(np.float32)


# ============================================================
# Optimization
# ============================================================

def optimize_deformable_patch(model, arr_clean, baseline_caps, baseline_embs,
                              person_signal, texture_np, cx, cy,
                              R=24, ps=0.15, beta=10.0, lr_shape=0.5, lr_tex=0.01,
                              n_epochs=200, sharpen_at=0.12,
                              attack_mode="poison", lam=-100.0, n_repeats=1,
                              area_min=0.0, lambda_area=50.0):
    """
    Joint shape + texture optimization following Shape Matters paper.

    Args:
        model: YOLOv3 model
        arr_clean: (H,W,3) clean image numpy
        baseline_caps: dict of layer_name -> captured feature tensor
        baseline_embs: dict of layer_name -> {"gap": tensor, "point": tensor}
        person_signal: dict of layer_name -> person signal tensor
        texture_np: (H,W,3) initial texture (Sierpinski-tiled trained patch)
        cx, cy: patch center
        R: number of rays
        ps: area upper limit (fraction)
        beta: area penalty weight
        lr_shape: learning rate for ray lengths
        lr_tex: learning rate for texture
        n_epochs: optimization epochs
        sharpen_at: area ratio at which to sharpen mask
        attack_mode: "poison" (maximize embedding shift) or "suppress" (minimize detection confidence)
        lam: mask sharpness parameter
        area_min: minimum patch area fraction — prevents suppress mode from shrinking patch to zero
        lambda_area: weight for area_min penalty term

    Returns:
        ray_lengths (R,), texture (H,W,3), mask (H,W), history dict
    """
    H, W, _ = arr_clean.shape
    device = next(model.parameters()).device

    # Initialize ray lengths — start as circle
    r_init = 80.0
    ray_lengths = torch.full((R,), r_init, device=device, requires_grad=True)
    # n_repeats is passed through for mask construction

    # Initialize texture as learnable tensor
    texture = torch.from_numpy(texture_np).to(device).clone()
    texture.requires_grad_(True)

    # Clean image tensor
    clean_tensor = torch.from_numpy(arr_clean).permute(2, 0, 1).unsqueeze(0).to(device)

    # Optimizers
    opt_shape = torch.optim.Adam([ray_lengths], lr=lr_shape)
    opt_tex = torch.optim.Adam([texture], lr=lr_tex)

    history = {"epoch": [], "loss": [], "area": [], "l2_shift": [],
               "cos_gap": [], "ray_min": [], "ray_max": [], "ray_std": []}

    sharpened = False

    for epoch in range(n_epochs):
        opt_shape.zero_grad()
        opt_tex.zero_grad()

        # Build differentiable mask (with n_repeats for rotational symmetry)
        mask = differentiable_polygon_mask(ray_lengths, cx, cy, H, W, lam=lam,
                                            n_repeats=n_repeats)
        area = compute_area(mask)

        # Apply patch: x_adv = texture * mask + clean * (1 - mask)
        mask_3d = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) for broadcasting
        tex_chw = texture.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        clean_chw = clean_tensor  # (1,3,H,W)
        adv_img = tex_chw * mask_3d + clean_chw * (1.0 - mask_3d)
        adv_img = torch.clamp(adv_img, 0, 1)

        # Forward pass — gradient-preserving for backprop to patch params
        caps_mod, _ = fwd_all_grad(model, adv_img)

        # Compute loss
        if attack_mode == "poison":
            # Maximize embedding L2 shift across all detection layers
            total_l2 = 0.0
            total_cos = 0.0
            for lname, lidx in DETECTION_LAYERS.items():
                gap = gap_embedding(caps_mod, lidx)
                l2 = torch.norm(gap - baseline_embs[lname]["gap"])
                total_l2 = total_l2 + l2
                cos = F.cosine_similarity(
                    gap.unsqueeze(0), baseline_embs[lname]["gap"].unsqueeze(0))[0]
                total_cos = total_cos + cos
            # L_adv = -L2_shift (minimize negative = maximize shift)
            L_adv = -total_l2
        else:  # suppress
            # Minimize person detection confidence using raw logits (pre-sigmoid)
            # Sigmoid saturates and kills gradients, so use raw objectness scores
            # Also use class confidence for person (class 0) from all 3 scales
            total_obj = 0.0
            for lname, lidx in DETECTION_LAYERS.items():
                feat = caps_mod[lidx]  # (1, C, H, W)
                n_anchors = 3
                cls_per_anchor = feat.shape[1] // n_anchors  # 85 for COCO
                for anchor in range(n_anchors):
                    ch_obj = anchor * cls_per_anchor + 4
                    # Raw objectness logit (pre-sigmoid) — gradients flow
                    obj_logit = feat[0, ch_obj, :, :]
                    # Also class 0 (person) confidence logit
                    ch_cls0 = anchor * cls_per_anchor + 5
                    cls0_logit = feat[0, ch_cls0, :, :]
                    # Combined detection score = obj * cls — use sum of logits
                    total_obj = total_obj + obj_logit.mean() + cls0_logit.mean()
            L_adv = total_obj  # minimize raw logits

        # Area constraint (Eq. 8) — upper limit prevents patch from growing too large
        if area <= ps:
            loss = L_adv
        else:
            loss = L_adv + beta * area

        # Note: area_min is enforced as a hard projection after optimizer step,
        # not as a soft penalty. Soft penalties (quadratic, linear) were overwhelmed
        # by the adversarial loss gradient. Projection is non-negotiable.

        loss.backward()

        # Gradient step
        opt_shape.step()
        opt_tex.step()

        # Clamp ray lengths to valid range
        with torch.no_grad():
            ray_lengths.clamp_(10, min(H, W) // 2)
            texture.clamp_(0, 1)

            # Hard projection: enforce both area_min and ps (upper bound)
            # Soft penalties (beta*area) are overwhelmed when L_adv is large
            if area_min > 0 or True:
                mask_check = differentiable_polygon_mask(ray_lengths, cx, cy, H, W,
                                                          lam=lam, n_repeats=n_repeats)
                area_check = mask_check.mean().item()
                # Lower bound: scale up if below area_min
                if area_min > 0 and area_check < area_min:
                    scale_factor = (area_min / max(area_check, 1e-6)) ** 0.5
                    ray_lengths.mul_(scale_factor)
                # Upper bound: scale down if above ps
                if area_check > ps:
                    scale_factor = (ps / max(area_check, 1e-6)) ** 0.5
                    ray_lengths.mul_(scale_factor)
                ray_lengths.clamp_(10, min(H, W) // 2)

        # Mask sharpening when area reaches target
        if not sharpened and area.item() >= sharpen_at:
            print(f"    [Epoch {epoch}] Area reached {area.item():.4f} >= {sharpen_at}, sharpening mask")
            lam = lam * 3  # Increase sharpness
            sharpened = True

        # Record
        with torch.no_grad():
            l2_val = 0.0
            cos_val = 0.0
            for lname, lidx in DETECTION_LAYERS.items():
                gap = gap_embedding(caps_mod, lidx)
                l2_val += torch.norm(gap - baseline_embs[lname]["gap"]).item()
                cos_val += F.cosine_similarity(
                    gap.unsqueeze(0), baseline_embs[lname]["gap"].unsqueeze(0))[0].item()

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["area"].append(area.item())
            history["l2_shift"].append(l2_val)
            history["cos_gap"].append(cos_val / len(DETECTION_LAYERS))
            history["ray_min"].append(ray_lengths.min().item())
            history["ray_max"].append(ray_lengths.max().item())
            history["ray_std"].append(ray_lengths.std().item())

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:3d}: loss={loss.item():.4f} area={area.item():.4f} "
                  f"l2_shift={l2_val:.3f} cos={cos_val/len(DETECTION_LAYERS):.4f} "
                  f"r=[{ray_lengths.min().item():.1f}, {ray_lengths.max().item():.1f}] "
                  f"r_std={ray_lengths.std().item():.1f}")

    # Final mask (sharpened)
    with torch.no_grad():
        final_mask = differentiable_polygon_mask(ray_lengths, cx, cy, H, W, lam=lam,
                                                    n_repeats=n_repeats)
        final_mask = (final_mask > 0.5).float()  # Binarize
        final_texture = texture.detach().cpu().numpy()
        final_rays = ray_lengths.detach().cpu().numpy()

    return final_rays, final_texture, final_mask.cpu().numpy(), history


# ============================================================
# Visualization
# ============================================================

def visualize_deformable(ray_lengths, texture, mask, cx, cy, H, W, save_path, title="", n_repeats=1):
    """Visualize the deformable patch: shape, texture, masked texture, overlay."""
    R_base = len(ray_lengths)
    R_total = R_base * n_repeats
    dtheta = 2 * math.pi / R_total
    # Expand ray lengths for display
    endpoints = []
    for i in range(R_total):
        r = ray_lengths[i % R_base]
        angle = i * dtheta
        ex = cx + r * math.cos(angle)
        ey = cy + r * math.sin(angle)
        endpoints.append((ex, ey))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Shape (mask)
    axes[0].imshow(mask, cmap="gray")
    poly_pts = endpoints + [endpoints[0]]
    poly_x = [p[0] for p in poly_pts]
    poly_y = [p[1] for p in poly_pts]
    axes[0].plot(poly_x, poly_y, "r-", linewidth=1.5)
    axes[0].set_title(f"Shape (R={R_base}x{n_repeats}={R_total}, area={np.mean(mask)*100:.1f}%)")
    axes[0].axis("off")

    # Texture
    axes[1].imshow(np.clip(texture, 0, 1))
    axes[1].set_title("Texture")
    axes[1].axis("off")

    # Masked texture
    masked = texture * mask[:, :, None]
    axes[2].imshow(np.clip(masked, 0, 1))
    axes[2].set_title("Masked Texture")
    axes[2].axis("off")

    # Ray diagram — show base rays in blue, all vertices in red
    axes[3].set_xlim(0, W)
    axes[3].set_ylim(H, 0)
    axes[3].set_aspect("equal")
    for i in range(R_total):
        r = ray_lengths[i % R_base]
        angle = i * dtheta
        ex = cx + r * math.cos(angle)
        ey = cy + r * math.sin(angle)
        # Highlight base rays (first R_base) in blue, replicated in light blue
        if i < R_base:
            axes[3].plot([cx, ex], [cy, ey], "b-", linewidth=1.5)
        else:
            axes[3].plot([cx, ex], [cy, ey], "c-", linewidth=0.3, alpha=0.5)
        axes[3].plot(ex, ey, "r.", markersize=2)
    axes[3].plot(cx, cy, "k+", markersize=10)
    poly_x = [p[0] for p in endpoints + [endpoints[0]]]
    poly_y = [p[1] for p in endpoints + [endpoints[0]]]
    axes[3].plot(poly_x, poly_y, "r-", linewidth=1.5)
    axes[3].set_title(f"Rays (R={R_base} base x {n_repeats} repeats)")
    axes[3].set_facecolor("lightgray")

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_history(history, save_path, title=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(history["epoch"], history["loss"])
    axes[0, 0].set_title("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["epoch"], [a * 100 for a in history["area"]])
    axes[0, 1].set_title("Area (%)")
    axes[0, 1].axhline(y=15, color="r", linestyle="--", alpha=0.5, label="ps=15%")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history["epoch"], history["l2_shift"])
    axes[0, 2].set_title("L2 Shift")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history["epoch"], history["cos_gap"])
    axes[1, 0].set_title("Cosine GAP (avg)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["epoch"], history["ray_min"], label="min")
    axes[1, 1].plot(history["epoch"], history["ray_max"], label="max")
    axes[1, 1].plot(history["epoch"], history["ray_std"], label="std")
    axes[1, 1].set_title("Ray lengths")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Shape Matters — Deformable Patch Attack")
    print("Joint shape + texture optimization with area constraint")
    print("=" * 70)

    H, W = IS, IS
    cx, cy = IS // 2, int(IS * 0.58)

    # Load trained patches for texture initialization
    print("\nLoading trained patches...")
    poison_src = load_img_array(POISON)
    suppress_src = load_img_array(SUPPRESS)
    poison_src = cv2.resize(poison_src, (256, 256), interpolation=cv2.INTER_LINEAR)
    suppress_src = cv2.resize(suppress_src, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Path B: Initialize with k196 sinusoid (proven 77% suppression on YOLOv3)
    # Per-channel RGB variants: k196 (disrupt), k167 (suppress), k208 (hallucinate)
    # This is the original init that produced the concentric ring FFT pattern
    print("Preparing k196 sinusoid texture (RGB variants: 196/167/208)...")
    poison_tex = make_sinusoid_texture(H, W, kx=196, ky=196, amp=0.30, rgb_variations=True)
    suppress_tex = make_sinusoid_texture(H, W, kx=196, ky=196, amp=0.30, rgb_variations=True)
    Image.fromarray((poison_tex * 255).astype(np.uint8)).save(f"{OUT}/init_k196_sinusoid.png")
    print(f"  Saved init texture: {OUT}/init_k196_sinusoid.png")

    # Load YOLOv3
    print("\nLoading YOLOv3...")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Baseline forward pass
    print("Computing baseline embeddings...")
    arr_w = load_img(IMG_WITH, IS)
    arr_wo = load_img(IMG_WITHOUT, IS)

    tensor_clean = torch.from_numpy(arr_w).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_clean, _ = fwd_all(model, tensor_clean)
    tensor_empty = torch.from_numpy(arr_wo).permute(2, 0, 1).unsqueeze(0).to(DEV)
    caps_empty, _ = fwd_all(model, tensor_empty)

    person_signal = {}
    for lname, lidx in DETECTION_LAYERS.items():
        person_signal[lname] = (caps_clean[lidx] - caps_empty[lidx]).squeeze(0)

    person_sx, person_sy = IS // 2, int(IS * 0.58)
    baseline_embs = {}
    for lname, lidx in DETECTION_LAYERS.items():
        gap = gap_embedding(caps_clean, lidx)
        point = extract_emb_at(caps_clean, lidx, person_sx, person_sy)
        baseline_embs[lname] = {"gap": gap, "point": point}

    print(f"  Baseline ready at {list(DETECTION_LAYERS.keys())}")

    # Run optimization for both attack modes
    all_results = {}

    # R=3, n_repeats=24: 72 vertices, rays collapse to circle → concentric ring FFT
    R = 3
    N_REPEATS = 24
    for mode, tex_init in [("poison", poison_tex), ("suppress", suppress_tex)]:
        print(f"\n{'='*50}")
        print(f"Optimizing {mode.upper()} deformable patch (Path B)")
        print(f"  R=3 rays, n_repeats=24 = {R*N_REPEATS} vertices (circle)")
        print(f"  Texture init: k196 sinusoid (RGB variants 196/167/208)")
        print(f"  ps=15%, beta=10, lr_shape=1.0, lr_tex=0.01, area_min=0.08")
        print(f"{'='*50}")

        t0 = time.time()
        # Both modes need area_min — poison also collapsed mid-run without it
        # Linear penalty with lambda=5000 counters suppress loss (~-200) and poison loss
        rays, tex, mask, hist = optimize_deformable_patch(
            model, arr_w, caps_clean, baseline_embs, person_signal,
            tex_init, cx, cy,
            R=R, ps=0.15, beta=10.0, lr_shape=1.0, lr_tex=0.01,
            n_epochs=300, sharpen_at=0.12,
            attack_mode=mode, lam=-100.0, n_repeats=N_REPEATS,
            area_min=0.08, lambda_area=5000.0
        )
        elapsed = time.time() - t0
        print(f"  Optimization complete in {elapsed:.1f}s")

        # Evaluate final patch
        arr_mod = apply_patch_to_image(arr_w, tex, mask, cx, cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            point = extract_emb_at(caps_mod, lidx, person_sx, person_sy)
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_embs[lname]["gap"].unsqueeze(0))[0])
            cos_point = float(F.cosine_similarity(
                point.unsqueeze(0), baseline_embs[lname]["point"].unsqueeze(0))[0])
            l2_gap = float(torch.norm(gap - baseline_embs[lname]["gap"]).item())
            l2_point = float(torch.norm(point - baseline_embs[lname]["point"]).item())
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

        area_pct = np.mean(mask) * 100
        total_l2 = sum(m["l2_shift_gap"] for m in metrics.values())
        print(f"\n  Final {mode} patch: area={area_pct:.1f}%, total L2={total_l2:.3f}")
        for lname, m in metrics.items():
            print(f"    {lname}: cos_gap={m['cos_gap']:.4f} l2_shift={m['l2_shift_gap']:.3f} "
                  f"overlap={m['person_overlap']:.4f}")

        # Save outputs
        tag = f"{mode}_R3_k196_nrep24"
        Image.fromarray((tex * 255).astype(np.uint8)).save(f"{OUT}/{tag}_texture_416.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"{OUT}/{tag}_mask_416.png")
        masked_tex = np.clip(tex * mask[:, :, None], 0, 1)
        Image.fromarray((masked_tex * 255).astype(np.uint8)).save(f"{OUT}/{tag}_patch_416.png")
        Image.fromarray((arr_mod * 255).astype(np.uint8)).save(f"{OUT}/{tag}_applied_416.png")

        visualize_deformable(rays, tex, mask, cx, cy, H, W,
                            f"{OUT}/{tag}_visualization.png",
                            f"Deformable {mode} — R=3 n_rep=24 circle, k196 sinusoid init, ps=15%",
                            n_repeats=N_REPEATS)
        plot_history(hist, f"{OUT}/{tag}_history.png",
                     f"{mode} optimization history")

        # Save ray lengths
        np.save(f"{OUT}/{tag}_rays.npy", rays)

        # FFT analysis
        gray = np.mean(masked_tex, axis=2)
        mag = compute_2d_fft_mag(gray)
        radial = radial_average(mag, H, W)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.clip(masked_tex, 0, 1))
        axes[0].set_title(f"{mode} patch")
        axes[0].axis("off")
        axes[1].imshow(mag, cmap="inferno", extent=[-W//2, W//2, H//2, -H//2])
        axes[1].set_title("2D FFT")
        axes[1].axis("off")
        axes[2].plot(range(len(radial)), radial, "b-", linewidth=0.8)
        for k in [3, 9, 27, 81, 167, 196, 208, 243]:
            if k < len(radial):
                axes[2].axvline(x=k, color="r", linestyle="--", alpha=0.4)
                axes[2].text(k, radial.max() * 0.9, str(k), fontsize=6, color="r", rotation=90)
        axes[2].set_title("Radial FFT")
        axes[2].set_xlim(0, min(250, len(radial)))
        plt.suptitle(f"{mode} deformable patch — FFT analysis")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{tag}_fft.png", dpi=150, bbox_inches="tight")
        plt.close()

        all_results[tag] = {
            "mode": mode, "R": R, "n_repeats": N_REPEATS, "ps": 0.15,
            "area_pct": area_pct, "total_l2": total_l2,
            "ray_lengths": rays.tolist(),
            "metrics": metrics,
            "history": hist,
        }

    # Comparison: fixed-shape Sierpinski (no shape optimization)
    print(f"\n{'='*50}")
    print("Comparison: fixed triangle Sierpinski (no shape optimization)")
    print(f"{'='*50}")

    from best_patch import render_sierpinski_image
    for mode, src in [("poison", poison_src), ("suppress", suppress_src)]:
        tri_mask = np.zeros((H, W), dtype=np.float32)
        h_val = 200 * math.sqrt(3) / 2
        v0 = (cx, cy - h_val * 2/3)
        v1 = (cx - 100, cy + h_val/3)
        v2 = (cx + 100, cy + h_val/3)
        cv2.fillConvexPoly(tri_mask, np.int32([v0, v1, v2]), 1.0)
        tri_tex, _ = render_sierpinski_image(H, W, cx, cy, 200, src, max_depth=3)

        arr_mod = apply_patch_to_image(arr_w, tri_tex, tri_mask, cx, cy)
        tensor_mod = torch.from_numpy(arr_mod).permute(2, 0, 1).unsqueeze(0).to(DEV)
        caps_mod, _ = fwd_all(model, tensor_mod)

        metrics = {}
        for lname, lidx in DETECTION_LAYERS.items():
            gap = gap_embedding(caps_mod, lidx)
            l2_gap = float(torch.norm(gap - baseline_embs[lname]["gap"]).item())
            cos_gap = float(F.cosine_similarity(
                gap.unsqueeze(0), baseline_embs[lname]["gap"].unsqueeze(0))[0])
            metrics[lname] = {"cos_gap": cos_gap, "l2_shift_gap": l2_gap}

        total_l2 = sum(m["l2_shift_gap"] for m in metrics.values())
        area_pct = np.mean(tri_mask) * 100
        tag = f"fixed_tri_{mode}"
        all_results[tag] = {
            "mode": mode, "area_pct": area_pct, "total_l2": total_l2,
            "metrics": metrics,
        }
        print(f"  {tag}: area={area_pct:.1f}%, total L2={total_l2:.3f}")

    # Print-ready versions
    print("\nGenerating print-ready versions...")
    PRINT_W, PRINT_H = 3600, 4800
    pcx, pcy = PRINT_W // 2, int(PRINT_H * 0.45)
    scale = PRINT_W / IS * 3.5

    for mode in ["poison", "suppress"]:
        tag = f"{mode}_R3_k196_nrep24"
        rays = np.load(f"{OUT}/{tag}_rays.npy")
        tex = np.array(Image.open(f"{OUT}/{tag}_texture_416.png").convert("RGB"), dtype=np.float32) / 255.0

        # Scale rays to print resolution
        rays_scaled = rays * scale
        mask_hr = differentiable_polygon_mask(
            torch.from_numpy(rays_scaled).to(DEV),
            pcx, pcy, PRINT_H, PRINT_W, lam=-200.0,
            n_repeats=N_REPEATS
        ).cpu().numpy()
        mask_hr = (mask_hr > 0.5).astype(np.float32)

        # Scale texture
        tex_hr = cv2.resize(tex, (PRINT_W, PRINT_H), interpolation=cv2.INTER_LANCZOS4)
        # White background outside mask — for printing on white fabric
        patch_hr = np.clip(tex_hr * mask_hr[:, :, None] + (1.0 - mask_hr[:, :, None]) * 1.0, 0, 1)

        path = f"{OUT}/{tag}_print_3600x4800_300dpi.png"
        Image.fromarray((patch_hr * 255).astype(np.uint8)).save(path)
        print(f"  Saved: {path}")

    # Save results
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "mode", "area_pct", "total_l2", "layer", "cos_gap", "l2_shift_gap"])
        for tag, res in all_results.items():
            if "metrics" in res:
                for lname, m in res["metrics"].items():
                    w.writerow([tag, res.get("mode", ""), f"{res.get('area_pct', 0):.2f}",
                                f"{res.get('total_l2', 0):.3f}", lname,
                                f"{m.get('cos_gap', 0):.6f}", f"{m.get('l2_shift_gap', 0):.6f}"])

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Deformable vs Fixed Shape")
    print(f"{'='*70}")
    print(f"{'Patch':<30} {'Area':<8} {'L81':<10} {'L93':<10} {'L105':<10} {'TOTAL':<10}")
    for tag, res in all_results.items():
        m = res.get("metrics", {})
        l81 = m.get("L81_52x52", {}).get("l2_shift_gap", 0)
        l93 = m.get("L93_26x26", {}).get("l2_shift_gap", 0)
        l105 = m.get("L105_13x13", {}).get("l2_shift_gap", 0)
        total = res.get("total_l2", 0)
        area = res.get("area_pct", 0)
        print(f"{tag:<30} {area:<8.1f} {l81:<10.3f} {l93:<10.3f} {l105:<10.3f} {total:<10.3f}")

    print(f"\nPrint-ready:")
    print(f"  POISON:   {OUT}/poison_R3_k196_nrep24_print_3600x4800_300dpi.png")
    print(f"  SUPPRESS: {OUT}/suppress_R3_k196_nrep24_print_3600x4800_300dpi.png")
    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
