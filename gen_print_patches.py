"""
Generate print-ready patch PNGs for physical testing.
Both patterns, multiple amplitudes, high resolution for quality printing.
"""
import numpy as np
from PIL import Image
import os

def get_decimal_expansion(num, den, n_digits):
    """Get n_digits of decimal expansion of num/den."""
    digits = []
    r = num
    for _ in range(n_digits):
        r *= 10
        digits.append(r // den)
        r = r % den
    return digits

def make_stripes_v(w, h, k, amp):
    """Vertical stripes: k cycles across width."""
    x = np.arange(w)
    pat = amp * np.sign(np.sin(2 * np.pi * k * x / w))
    return np.tile(pat, (h, 1)).astype(np.float32)

def make_digits_196(w, h, amp):
    """1/196 digit pattern mapped to columns."""
    digits = get_decimal_expansion(1, 196, max(w, 500))
    pat = np.zeros((h, w), dtype=np.float32)
    for xi in range(w):
        d = digits[xi % len(digits)]
        pat[:, xi] = (d / 4.5 - 1.0) * amp
    return pat

def pattern_to_printable_rgb(pat, amp, res):
    """
    Map pattern to printable RGB.
    Pattern values are in [-amp, +amp].
    Map to full printable range for maximum contrast while preserving structure.
    Center at 128, scale so amp maps to +-128 (full range).
    This preserves the relative pattern structure at maximum print contrast.
    """
    # Scale: map [-amp, +amp] to [0, 255]
    # So 0 -> 128, +amp -> 255, -amp -> 0
    scale = 127.0 / amp if amp > 0 else 1.0
    rgb = np.zeros((res, res, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = 128.0 + pat * scale
    return np.clip(rgb, 0, 255).astype(np.uint8)

def pattern_to_actual_rgb(pat, amp, res):
    """
    Map pattern to actual amplitude RGB (what the model sees).
    Pattern values in [-amp, +amp], mapped to 128 +- amp*255.
    This is the true amplitude - very subtle gray-on-gray.
    """
    rgb = np.zeros((res, res, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = 128.0 + pat * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

# Output directory
out_dir = "outputs_clothing/forward_analysis/patch_pipeline/print_ready"
os.makedirs(out_dir, exist_ok=True)

# Generate at 3000x3000 (10 inches at 300dpi)
RES = 3000

patterns = [
    ("k12_stripes", 12, make_stripes_v),
    ("digits_196", None, make_digits_196),
]

amplitudes = [0.02, 0.04, 0.08]

print(f"Generating print-ready patches at {RES}x{RES}px (10in @ 300dpi)")
print(f"Output: {out_dir}/")
print()

for pat_name, k, gen_func in patterns:
    for amp in amplitudes:
        # Generate pattern
        if k is not None:
            pat = gen_func(RES, RES, k, amp)
        else:
            pat = gen_func(RES, RES, amp)

        # Version 1: Full contrast (max printable range, preserves structure)
        rgb_full = pattern_to_printable_rgb(pat, amp, RES)
        fname = f"{pat_name}_amp{amp:.3f}_fullcontrast_{RES}px.png"
        path = os.path.join(out_dir, fname)
        Image.fromarray(rgb_full, mode="RGB").save(path, optimize=True)
        print(f"  {fname} ({os.path.getsize(path)//1024}KB)")

        # Version 2: Actual amplitude (true to what model sees, very subtle)
        rgb_actual = pattern_to_actual_rgb(pat, amp, RES)
        fname = f"{pat_name}_amp{amp:.3f}_actual_{RES}px.png"
        path = os.path.join(out_dir, fname)
        Image.fromarray(rgb_actual, mode="RGB").save(path, optimize=True)
        print(f"  {fname} ({os.path.getsize(path)//1024}KB)")

# Also generate a neutral gray reference (amp=0)
gray = np.full((RES, RES, 3), 128, dtype=np.uint8)
fname = f"neutral_gray_{RES}px.png"
path = os.path.join(out_dir, fname)
Image.fromarray(gray, mode="RGB").save(path)
print(f"\n  {fname} (reference - no pattern)")

print(f"\nDone. {len(patterns) * len(amplitudes) * 2 + 1} images generated.")
print(f"\nPrint at 10x10 inches (300dpi) or scale to desired patch size.")
print(f"Full contrast = pattern structure at maximum print visibility.")
print(f"Actual = true amplitude (very subtle, may not print well on consumer printers).")
