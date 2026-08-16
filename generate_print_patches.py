"""
Generate additional print-ready patches at multiple amplitudes for physical testing.
The stealthy_patch.py found sweet spots at amp=0.05 (nearly invisible).
This generates visible versions at amp=0.10, 0.15, 0.20 for practical testing.
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw
from decimal import Decimal, getcontext

OUTPUT_DIR = r"C:\Users\carso\Desktop\YODO\outputs_clothing"
PRINT_DPI = 300
PRINT_W_IN = 12
PRINT_H_IN = 16
PRINT_W_PX = int(PRINT_W_IN * PRINT_DPI)  # 3600
PRINT_H_PX = int(PRINT_H_IN * PRINT_DPI)  # 4800
IMG_SIZE = 416
SCALE = PRINT_W_PX / IMG_SIZE  # ~8.65

def get_decimal_expansion(numerator, denominator, num_digits=500):
    getcontext().prec = num_digits + 50
    val = Decimal(numerator) / Decimal(denominator)
    digits_str = str(val)[2:]
    return [int(d) for d in digits_str[:num_digits]]

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

# Medium r80 patch (sweet spot size) — scaled to print resolution
base_rays = [75, 60, 80, 55, 78, 65, 82, 60, 75, 65, 80, 55]
print_rays = [r * SCALE for r in base_rays]
print_cx = PRINT_W_PX // 2
print_cy = PRINT_H_PX // 2

mask, endpoints = make_deformable_mask(PRINT_H_PX, PRINT_W_PX, print_cx, print_cy, print_rays, 12)
print(f"Patch area: {np.mean(mask)*100:.1f}% of print")

# Generate at multiple amplitudes and textures
configs = [
    # (label, texture_fn, amp)
    # Stealthy regime (0 collateral, 1 wearer suppressed)
    ("stripes13_amp0.05", "stripes_13px", 0.05),
    ("stripes13_amp0.08", "stripes_13px", 0.08),
    # Moderate regime (some suppression, visible when printed)
    ("k167d_amp0.10", "k167_d", 0.10),
    ("k167d_amp0.15", "k167_d", 0.15),
    ("k167d_amp0.20", "k167_d", 0.20),
    # Aggressive regime (maximum suppression)
    ("k167sq_amp0.15", "k167_square", 0.15),
    ("k167sq_amp0.20", "k167_square", 0.20),
    # Digit pattern (cloud poisoning)
    ("digits196_amp0.15", "digits_196", 0.15),
    ("digits196_amp0.20", "digits_196", 0.20),
    # Composite (dual-purpose: suppress + poison)
    ("composite_amp0.10", "composite", 0.10),
    ("composite_amp0.15", "composite", 0.15),
    ("composite_amp0.20", "composite", 0.20),
]

digits_196 = get_decimal_expansion(1, 196, 500)

for label, tex_type, amp in configs:
    # Generate texture at print resolution
    y, x = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")

    if tex_type == "k167_d":
        k_scaled = int(167 * SCALE)
        texture = (amp * np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y))).astype(np.float32)
    elif tex_type == "k167_square":
        k_scaled = int(167 * SCALE)
        texture = (amp * np.sign(np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y)))).astype(np.float32)
    elif tex_type == "stripes_13px":
        stripe_period = int(13 * SCALE)
        texture = (amp * np.sign(np.sin(2*np.pi*np.arange(PRINT_W_PX)/stripe_period))).astype(np.float32)
        texture = texture * np.ones((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
    elif tex_type == "digits_196":
        texture = np.zeros((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
        for xi in range(PRINT_W_PX):
            x_416 = int(xi / SCALE) % len(digits_196)
            d = digits_196[x_416]
            texture[:, xi] = (d / 9.0) * amp
    elif tex_type == "composite":
        # k167 diagonal (60%) + digit pattern (40%)
        k_scaled = int(167 * SCALE)
        k167_tex = (amp * 0.6 * np.cos(2*np.pi*(k_scaled/PRINT_W_PX*x + k_scaled/PRINT_H_PX*y)))
        digit_tex = np.zeros((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
        for xi in range(PRINT_W_PX):
            x_416 = int(xi / SCALE) % len(digits_196)
            d = digits_196[x_416]
            digit_tex[:, xi] = (d / 9.0) * amp * 0.4
        texture = (k167_tex + digit_tex).astype(np.float32)
    else:
        continue

    # Create image: gray background with texture inside mask, white outside
    img = np.full((PRINT_H_PX, PRINT_W_PX, 3), 0.5, dtype=np.float32)
    for c in range(3):
        img[:,:,c] = np.clip(0.5 + texture * mask, 0, 1)
    # White outside mask
    for c in range(3):
        img[:,:,c] = img[:,:,c] * mask + 1.0 * (1 - mask)

    # Save with alpha
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    fname = f"stealthy_patch_{label}_medium_r80_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    pil.save(fpath, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"Saved: {fname}")

print(f"\nDone — {len(configs)} patches generated in {OUTPUT_DIR}")
print(f"Print size: {PRINT_W_IN}x{PRINT_H_IN} inches at {PRINT_DPI} DPI")
print(f"\nRecommended for physical testing:")
print(f"  1. composite_amp0.15 — dual-purpose (suppress + poison), visible")
print(f"  2. k167d_amp0.15 — best suppression, visible diagonal pattern")
print(f"  3. k167sq_amp0.20 — aggressive, strong square wave")
print(f"  4. digits196_amp0.15 — cloud poisoning, digit pattern")
print(f"  5. stripes13_amp0.05 — stealthy (barely visible, 0 collateral)")
