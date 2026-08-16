"""
Direct concentric ring patch generator — no optimizer, no fractal imports.

Produces a circular patch with radial sinusoid texture:
  p(r) = sin(196 * r/R) + 0.5*sin(81*r/R) + 0.3*sin(27*r/R) + 0.15*sin(9*r/R)

The radial coordinate creates perfect concentric rings.
Sub-harmonics k=9,27,81,196 match what the optimizer discovered in the
original R=3 n_repeats=24 run.

Output:
  - 416x416 patch + mask for YOLOv3
  - 2D FFT showing concentric ring structure
  - Radial FFT profile
  - 3600x4800 300dpi print-ready
  - YOLOv3 forward pass embedding shift at L81/L93/L105
"""
import os, sys, math, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ─── YOLOv3 model loading (standalone, no fractal imports) ─────────────── #
sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\concentric_rings"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416

DETECTION_LAYERS = {"L81_52x52": 81, "L93_26x26": 93, "L105_13x13": 105}

os.makedirs(OUT, exist_ok=True)


# ─── Forward pass (standalone) ─────────────────────────────────────────── #
def fwd_all(model, x):
    """Forward capture all conv layer outputs (detached)."""
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
    """Load and pad image to sz x sz."""
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
    ph, pw = patch_rgb.shape[:2]
    x0 = int(cx - pw // 2)
    y0 = int(cy - ph // 2)
    x1, y1 = x0 + pw, y0 + ph
    cx_src, cy_src = 0, 0
    if x0 < 0:
        cx_src = -x0
        x0 = 0
    if y0 < 0:
        cy_src = -y0
        y0 = 0
    if x1 > W:
        x1 = W
    if y1 > H:
        y1 = H
    dx = x1 - x0
    dy = y1 - y0
    if dx > 0 and dy > 0:
        for c in range(3):
            out[y0:y1, x0:x1, c] = (
                patch_rgb[cy_src:cy_src+dy, cx_src:cx_src+dx, c] * mask[cy_src:cy_src+dy, cx_src:cx_src+dx]
                + out[y0:y1, x0:x1, c] * (1.0 - mask[cy_src:cy_src+dy, cx_src:cx_src+dx])
            )
    return out


def compute_2d_fft_mag(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return mag


def radial_average(mag, H, W):
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.bincount(r.ravel())
    return radial[:min(H, W) // 2]


# ─── Radial ring texture generation ────────────────────────────────────── #
def generate_concentric_ring_patch(H, W, cx=None, cy=None, radius=None,
                                    freqs=None, amps=None, rgb_phase=True):
    """
    Generate concentric ring patch using radial sinusoids.

    p(r) = sum_i amp_i * sin(2*pi * freq_i * r / R_max + phase)

    Radial coordinate creates perfect concentric rings.
    Sub-harmonics k=9,27,81,196 provide multi-scale frequency content.

    Args:
        H, W: image size
        cx, cy: circle center
        radius: circle radius in pixels
        freqs: radial frequencies [9, 27, 81, 196]
        amps: amplitudes per frequency [0.15, 0.30, 0.50, 1.0]
        rgb_phase: per-channel phase offsets

    Returns:
        texture: (H, W, 3) float32 in [0, 1]
        mask: (H, W) float32 circle mask
    """
    if cx is None:
        cx = W / 2
    if cy is None:
        cy = H / 2
    if radius is None:
        radius = min(H, W) * 0.35
    if freqs is None:
        freqs = [9, 27, 81, 196]
    if amps is None:
        amps = [0.15, 0.30, 0.50, 1.0]

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r / radius  # [0, 1] inside circle

    # Circle mask — smooth edge
    mask = np.clip(1.0 - (r / radius) ** 8, 0, 1).astype(np.float32)

    # Per-channel phase offsets: R=0, G=2pi/3, B=4pi/3
    if rgb_phase:
        phase_offsets = [0.0, 2.094, 4.189]
    else:
        phase_offsets = [0.0, 0.0, 0.0]

    pat = np.zeros((H, W, 3), dtype=np.float32)

    for c in range(3):
        phase_c = phase_offsets[c]
        val = 0.5  # center at gray

        for k, amp in zip(freqs, amps):
            # Radial sinusoid: sin(2*pi * k * r_norm + phase)
            # k=196 means 196 cycles from center to edge
            radial = np.sin(2 * np.pi * k * r_norm + phase_c + k * 0.01)
            val += amp * 0.25 * radial

        pat[:, :, c] = val

    pat = np.clip(pat, 0, 1).astype(np.float32)

    # Apply circle mask — white outside
    pat = pat * mask[:, :, None] + (1.0 - mask[:, :, None]) * 1.0

    return pat, mask


# ─── Main ──────────────────────────────────────────────────────────────── #
def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("=" * 70)
    print("Concentric Ring Patch — Direct Generation")
    print("Radial sinusoid: k=9+27+81+196, no optimizer")
    print("=" * 70)

    H, W = IS, IS
    cx, cy = IS // 2, int(IS * 0.58)
    radius = IS * 0.35  # ~15% area

    freqs = [9, 27, 81, 196]
    amps = [0.15, 0.30, 0.50, 1.0]

    # Generate 416x416 patch
    print(f"\nGenerating {IS}x{IS} concentric ring patch...")
    print(f"  Center: ({cx}, {cy}), Radius: {radius}px")
    print(f"  Frequencies: {freqs}")
    print(f"  Amplitudes: {amps}")
    texture, mask = generate_concentric_ring_patch(
        H, W, cx=cx, cy=cy, radius=radius,
        freqs=freqs, amps=amps, rgb_phase=True
    )

    area_pct = np.mean(mask > 0.5) * 100
    print(f"  Circle area: {area_pct:.1f}%")

    masked_tex = np.clip(texture * mask[:, :, None], 0, 1)

    # Save 416x416
    Image.fromarray((texture * 255).astype(np.uint8)).save(f"{OUT}/rings_416.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"{OUT}/rings_mask_416.png")
    Image.fromarray((masked_tex * 255).astype(np.uint8)).save(f"{OUT}/rings_patch_416.png")
    print(f"  Saved: {OUT}/rings_416.png")
    print(f"  Saved: {OUT}/rings_patch_416.png")

    # FFT analysis
    print("\nGenerating FFT analysis...")
    gray = np.mean(masked_tex, axis=2)
    mag = compute_2d_fft_mag(gray)
    radial = radial_average(mag, H, W)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.clip(masked_tex, 0, 1))
    axes[0].set_title("Concentric ring patch (416x416)")
    axes[0].axis("off")

    axes[1].imshow(mag, cmap="inferno", extent=[-W//2, W//2, H//2, -H//2])
    axes[1].set_title("2D FFT — concentric rings")
    axes[1].axis("off")

    axes[2].plot(range(len(radial)), radial, "b-", linewidth=0.8)
    for k in [3, 9, 27, 81, 167, 196, 208, 243]:
        if k < len(radial):
            axes[2].axvline(x=k, color="r", linestyle="--", alpha=0.4)
            axes[2].text(k, radial.max() * 0.9, str(k), fontsize=7, color="r", rotation=90)
    axes[2].set_title("Radial FFT profile")
    axes[2].set_xlim(0, min(250, len(radial)))
    axes[2].set_xlabel("Frequency (k)")
    axes[2].set_ylabel("Magnitude")

    plt.suptitle("Concentric Ring Patch — k=9+27+81+196 radial sinusoid", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/rings_fft.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT}/rings_fft.png")

    # Print-ready version
    print("\nGenerating print-ready version...")
    PRINT_W, PRINT_H = 3600, 4800
    pcx, pcy = PRINT_W // 2, int(PRINT_H * 0.45)
    pradius = min(PRINT_H, PRINT_W) * 0.12

    tex_hr, mask_hr = generate_concentric_ring_patch(
        PRINT_H, PRINT_W, cx=pcx, cy=pcy, radius=pradius,
        freqs=freqs, amps=amps, rgb_phase=True
    )
    Image.fromarray((tex_hr * 255).astype(np.uint8)).save(
        f"{OUT}/rings_print_3600x4800_300dpi.png")
    print(f"  Saved: {OUT}/rings_print_3600x4800_300dpi.png")

    # YOLOv3 forward pass
    print("\nLoading YOLOv3 for forward pass...")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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

    # Apply patch and measure
    arr_mod = apply_patch_to_image(arr_w, texture, mask, cx, cy)
    Image.fromarray((arr_mod * 255).astype(np.uint8)).save(f"{OUT}/rings_applied_416.png")

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

    total_l2 = sum(m["l2_shift_gap"] for m in metrics.values())
    print(f"\n  Forward pass results: area={area_pct:.1f}%, total L2={total_l2:.3f}")
    for lname, m in metrics.items():
        print(f"    {lname}: cos_gap={m['cos_gap']:.4f} l2_shift={m['l2_shift_gap']:.3f} "
              f"overlap={m['person_overlap']:.4f}")

    # Save results
    results = {
        "concentric_rings": {
            "freqs": freqs, "amps": amps,
            "radius": radius, "area_pct": area_pct,
            "total_l2": total_l2,
            "metrics": metrics,
        }
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll outputs in: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
