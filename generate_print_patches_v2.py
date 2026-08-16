"""
Generate print-ready patches with VISIBLE patterns for physical testing.
- Low frequencies (5-30 cycles across image, not 1445)
- High contrast (black/white, not faint gray)
- Clean white background outside polygon (no alpha checkerboard)
- Actual diagonal lines for k167
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

# Medium r80 patch — scaled to print
base_rays = [75, 60, 80, 55, 78, 65, 82, 60, 75, 65, 80, 55]
print_rays = [r * SCALE for r in base_rays]
print_cx = PRINT_W_PX // 2
print_cy = PRINT_H_PX // 2

mask, endpoints = make_deformable_mask(PRINT_H_PX, PRINT_W_PX, print_cx, print_cy, print_rays, 12)
print(f"Patch area: {np.mean(mask)*100:.1f}% of print")

digits_196 = get_decimal_expansion(1, 196, 500)

# Anti-alias the mask edge (smooth boundary)
from scipy.ndimage import gaussian_filter
mask_smooth = gaussian_filter(mask, sigma=2.0).clip(0, 1)

y_grid, x_grid = np.meshgrid(np.arange(PRINT_H_PX), np.arange(PRINT_W_PX), indexing="ij")

# ============================================================
# Generate visible patterns
# ============================================================

# For physical testing, we need patterns that are:
# 1. Visible to the human eye (low frequency, high contrast)
# 2. Still adversarial (the frequency matters, not the visual appearance)
# 3. The YOLO model will downsample to 416px, so what matters is the
#    frequency in 416-space. A pattern with k=10 in print space (3600px)
#    becomes k=10/8.65 = ~1.16 in 416 space — too low.
#    We need k=167 in 416 space, which means k=167*8.65=1445 in print space.
#    But that's invisible. Solution: use a VISIBLE base pattern at low-k
#    AND modulate it with the high-k adversarial frequency.
#    OR: just use the high-k pattern but with binary contrast (black/white)
#    so the fine pattern is at least visible as texture.

# Actually, the key insight: for PHYSICAL testing, the camera will downsample.
# The print at 300dpi has fine detail that the camera won't capture fully.
# What matters is the AVERAGE value in each pixel after camera capture.
# So we should print the pattern at a frequency that:
# - Is visible on paper (can see the texture)
# - After camera capture + YOLO resize to 416, produces the right frequency

# For a phone camera at ~1m distance, the effective resolution is maybe 200-400 DPI.
# So our 300dpi print will be roughly 1:1 with camera pixels.
# After YOLO resizes to 416, the scale factor is ~416/3600 = 0.115.
# So k=10 in print → k=10*0.115 = 1.16 in YOLO space. Too low.
# k=100 in print → k=11.5 in YOLO space. Reasonable.
# k=200 in print → k=23 in YOLO space. Good.
# k=1445 in print → k=167 in YOLO space. Correct but invisible.

# Compromise: use k=200-400 in print space (visible as fine stripes)
# and accept that the effective k in YOLO space will be 23-46.
# OR: use binary patterns (square waves) at high k — they're visible as
# fine stripe texture even if individual lines aren't resolvable.

# Best approach: generate at the CORRECT k for YOLO (k=1445) but with
# BINARY contrast (pure black/white). This creates a visible fine texture
# that the camera will capture and YOLO will see at k=167.

configs = [
    # (label, description, texture_fn)
]

# 1. k=167 diagonal — BINARY (black/white) at correct YOLO frequency
# This will look like fine diagonal stripes
def make_k167_diagonal_binary():
    k_scaled = int(167 * SCALE)  # 1445
    phase = 2*np.pi*(k_scaled/PRINT_W_PX*x_grid + k_scaled/PRINT_H_PX*y_grid)
    return np.sign(np.cos(phase)).astype(np.float32)  # +1 or -1

# 2. k=167 diagonal — SINE at correct frequency, high amplitude
def make_k167_diagonal_sine():
    k_scaled = int(167 * SCALE)
    phase = 2*np.pi*(k_scaled/PRINT_W_PX*x_grid + k_scaled/PRINT_H_PX*y_grid)
    return np.cos(phase).astype(np.float32)  # -1 to +1

# 3. k=167 square wave — VERTICAL stripes (simpler, more visible)
def make_k167_vertical_binary():
    k_scaled = int(167 * SCALE)
    phase = 2*np.pi*k_scaled/PRINT_W_PX*x_grid
    return np.sign(np.cos(phase)).astype(np.float32)

# 4. 13px stripes — VERTICAL, binary (this is the stealthy pattern)
# 13px in YOLO space = 13*8.65 = 112px in print space
def make_stripes_13px_binary():
    stripe_period = int(13 * SCALE)  # ~112px
    return np.sign(np.sin(2*np.pi*np.arange(PRINT_W_PX)/stripe_period)).astype(np.float32) * np.ones((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)

# 5. 1/196 digit pattern — mapped to vertical bands
# Each digit gets a band of pixels. Digit value → brightness.
def make_digits196_visible():
    # Make each digit a band of ~SCALE pixels wide
    band_width = int(SCALE)  # ~8px per digit
    texture = np.zeros((PRINT_H_PX, PRINT_W_PX), dtype=np.float32)
    for x in range(PRINT_W_PX):
        x_416 = int(x / SCALE) % len(digits_196)
        d = digits_196[x_416]
        # Map digit 0-9 to -1 to +1
        texture[:, x] = (d / 4.5) - 1.0  # 0→-1, 9→+1
    return texture

# 6. Composite: k167 diagonal binary + digit pattern
def make_composite_binary():
    k_tex = make_k167_diagonal_binary()
    d_tex = make_digits196_visible()
    # Blend: 60% k167 + 40% digits
    return (0.6 * k_tex + 0.4 * d_tex).clip(-1, 1).astype(np.float32)

# 7. Low-frequency diagonal for VISIBILITY (k=20 in print = k=2.3 in YOLO)
# This is NOT adversarial but gives a clear visible reference
def make_lowfreq_diagonal_binary():
    k = 20  # 20 cycles across image — clearly visible
    phase = 2*np.pi*(k/PRINT_W_PX*x_grid + k/PRINT_H_PX*y_grid)
    return np.sign(np.cos(phase)).astype(np.float32)

# 8. Medium frequency diagonal (k=100 in print = k=11.5 in YOLO)
def make_medfreq_diagonal_binary():
    k = 100
    phase = 2*np.pi*(k/PRINT_W_PX*x_grid + k/PRINT_H_PX*y_grid)
    return np.sign(np.cos(phase)).astype(np.float32)

# Generate all patches
all_configs = [
    ("k167_diagonal_binary", "k=167 diagonal B&W (correct YOLO freq)", make_k167_diagonal_binary),
    ("k167_diagonal_sine", "k=167 diagonal sine (correct YOLO freq)", make_k167_diagonal_sine),
    ("k167_vertical_binary", "k=167 vertical B&W stripes", make_k167_vertical_binary),
    ("stripes13px_binary", "13px vertical B&W stripes (stealthy)", make_stripes_13px_binary),
    ("digits196_visible", "1/196 digit pattern (brightness bands)", make_digits196_visible),
    ("composite_binary", "k167+digits composite B&W", make_composite_binary),
    ("lowfreq_diagonal_binary", "k=20 diagonal (visible reference)", make_lowfreq_diagonal_binary),
    ("medfreq_diagonal_binary", "k=100 diagonal (medium freq)", make_medfreq_diagonal_binary),
]

for label, desc, tex_fn in all_configs:
    texture = tex_fn()

    # Create image: white background, texture inside mask
    # Use full contrast: texture -1 = black (0), +1 = white (255)
    # Inside mask: map texture from [-1,+1] to [0,1]
    # Outside mask: pure white (1.0)
    img = np.full((PRINT_H_PX, PRINT_W_PX, 3), 1.0, dtype=np.float32)  # white bg
    for c in range(3):
        # Inside mask: texture mapped to 0-1
        inner = (texture + 1.0) / 2.0  # -1→0 (black), +1→1 (white)
        img[:,:,c] = inner * mask_smooth + 1.0 * (1 - mask_smooth)

    # Save as PNG (no alpha — pure white background, clean for printing)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    fname = f"print_patch_{label}_medium_r80_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    pil.save(fpath, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"Saved: {fname}  ({desc})")

    # Also save a version with black background outside (for dark fabric)
    img_dark = np.full((PRINT_H_PX, PRINT_W_PX, 3), 0.0, dtype=np.float32)  # black bg
    for c in range(3):
        inner = (texture + 1.0) / 2.0
        img_dark[:,:,c] = inner * mask_smooth + 0.0 * (1 - mask_smooth)
    pil_dark = Image.fromarray((img_dark * 255).astype(np.uint8))
    fname_dark = f"print_patch_{label}_medium_r80_{PRINT_W_PX}x{PRINT_H_PX}_300dpi_darkbg.png"
    fpath_dark = os.path.join(OUTPUT_DIR, fname_dark)
    pil_dark.save(fpath_dark, dpi=(PRINT_DPI, PRINT_DPI))
    print(f"Saved: {fname_dark}  (black bg version)")

# Also save the mask outline for cutting
outline_img = Image.new("RGB", (PRINT_W_PX, PRINT_H_PX), (255, 255, 255))
draw = ImageDraw.Draw(outline_img)
outline_points = [(int(p[0]), int(p[1])) for p in endpoints]
# Draw thick red outline
for thickness in range(3):
    draw.polygon([(p[0]-thickness, p[1]-thickness) for p in outline_points],
                 outline=(255, 0, 0), fill=None)
    draw.polygon([(p[0]+thickness, p[1]+thickness) for p in outline_points],
                 outline=(255, 0, 0), fill=None)
fname_outline = f"print_patch_outline_medium_r80_{PRINT_W_PX}x{PRINT_H_PX}_300dpi.png"
outline_img.save(os.path.join(OUTPUT_DIR, fname_outline), dpi=(PRINT_DPI, PRINT_DPI))
print(f"Saved: {fname_outline}  (cutting guide)")

print(f"\nDone — {len(all_configs)*2 + 1} files in {OUTPUT_DIR}")
print(f"Print size: {PRINT_W_IN}x{PRINT_H_IN} inches at {PRINT_DPI} DPI")
print(f"\nRecommended for physical testing:")
print(f"  1. print_patch_k167_diagonal_binary — correct k=167, B&W diagonal, adversarial")
print(f"  2. print_patch_composite_binary — k167+digits dual-purpose, B&W")
print(f"  3. print_patch_stripes13px_binary — stealthy 13px vertical stripes")
print(f"  4. print_patch_digits196_visible — 1/196 digit brightness bands")
print(f"  5. print_patch_lowfreq_diagonal_binary — visible reference (NOT adversarial)")
print(f"\n  White bg = for light fabric / paper sticker")
print(f"  Dark bg = for dark fabric (black background outside patch)")
