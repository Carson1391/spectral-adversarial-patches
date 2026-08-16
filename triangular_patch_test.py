"""
Deformable Triangular Patch Attack — Shape Matters approach applied to YOLO detection.

Implements the deformable patch representation from "Shape Matters: Deformable Patch Attack"
combined with our frequency-based findings (k=167 suppression, k=208 hallucination, k=196 disruption).

Key elements from the paper:
- R rays from center → R triangles forming the patch contour
- Ray lengths r = {r1, r2, ..., rR} define the shape (deformable)
- Joint shape + texture optimization
- Area constraint: smaller patch = stronger result

Our additions:
- Triangular frequency textures (k=167/208/196 sinusoids clipped to triangle mask)
- Sierpinski/nested triangle patterns (triangles within triangles)
- Lychrel-number-based patterns (196, 295, 394, 493, 592, 689, 788, 887)
- Per-channel color variation (different frequency per RGB channel)
- Line/stripe patterns within triangular mask
- Tests on YOLOv3, YOLOv8, YOLO11, YOLO26

The patch is placed on the torso/chest region of the person (where clothing would be).
"""

import os, sys, json, csv, math
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")

from ultralytics import YOLO

IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\triangular_patch"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"

COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

LYCHREL_NUMBERS = [196, 295, 394, 493, 592, 689, 788, 887, 1997, 2998]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Deformable Patch Representation (from Shape Matters paper)
# ============================================================

def make_deformable_mask(H, W, cx, cy, ray_lengths, num_rays=None):
    """
    Create a deformable triangular mask following the Shape Matters paper.
    
    R rays from center (cx, cy), each pair of adjacent rays forms a triangle.
    ray_lengths = [r1, r2, ..., rR] — length of each ray.
    The mask is 1 inside the polygon, 0 outside.
    
    Uses the differentiable activation Phi(x) = (tanh(lambda*(x-1)) + 1) / 2
    with lambda = -100 for near-binary mask.
    """
    R = num_rays if num_rays else len(ray_lengths)
    dtheta = 2 * math.pi / R
    
    # Calculate ray endpoints
    endpoints = []
    for i in range(R):
        angle = i * dtheta
        r = ray_lengths[i % len(ray_lengths)]
        ex = cx + r * math.cos(angle)
        ey = cy + r * math.sin(angle)
        endpoints.append((ex, ey))
    
    # Build mask using PIL polygon fill (fast, exact)
    mask = np.zeros((H, W), dtype=np.float32)
    # Create polygon and fill it
    polygon = endpoints + [endpoints[0]]  # close the polygon
    img_mask = Image.new("F", (W, H), 0.0)
    draw = ImageDraw.Draw(img_mask)
    # PIL expects (x, y) tuples
    draw.polygon([(p[0], p[1]) for p in polygon], fill=1.0)
    mask = np.array(img_mask, dtype=np.float32)
    
    return mask, endpoints


def make_sierpinski_mask(H, W, cx, cy, size, depth=3):
    """
    Sierpinski triangle — triangles within triangles.
    depth=0: single triangle
    depth=1: 3 triangles
    depth=2: 9 triangles
    depth=3: 27 triangles
    """
    mask = np.zeros((H, W), dtype=np.float32)
    
    def draw_sierpinski(cx, cy, size, depth):
        if depth == 0:
            # Draw a single upward triangle
            h = size * math.sqrt(3) / 2
            pts = [(cx, cy - h*2/3), (cx - size/2, cy + h/3), (cx + size/2, cy + h/3)]
            img_tmp = Image.new("F", (W, H), 0.0)
            draw = ImageDraw.Draw(img_tmp)
            draw.polygon(pts, fill=1.0)
            nonlocal mask
            mask += np.array(img_tmp, dtype=np.float32)
            return
        
        half = size / 2
        h = size * math.sqrt(3) / 2
        # Top triangle
        draw_sierpinski(cx, cy - h/3, half, depth - 1)
        # Bottom left
        draw_sierpinski(cx - half/2, cy + h/6, half, depth - 1)
        # Bottom right
        draw_sierpinski(cx + half/2, cy + h/6, half, depth - 1)
    
    draw_sierpinski(cx, cy, size, depth)
    return np.clip(mask, 0, 1).astype(np.float32)


def make_nested_triangles_mask(H, W, cx, cy, outer_size, num_layers=5):
    """
    Nested concentric triangles — triangles inside triangles, alternating fill.
    Creates a bullseye-like triangular pattern.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    for layer in range(num_layers):
        size = outer_size * (1 - layer / num_layers)
        h = size * math.sqrt(3) / 2
        pts = [(cx, cy - h*2/3), (cx - size/2, cy + h/3), (cx + size/2, cy + h/3)]
        img_tmp = Image.new("F", (W, H), 0.0)
        draw = ImageDraw.Draw(img_tmp)
        fill_val = 1.0 if layer % 2 == 0 else 0.0
        draw.polygon(pts, fill=fill_val)
        if layer % 2 == 0:
            mask = np.maximum(mask, np.array(img_tmp, dtype=np.float32))
        else:
            mask = np.minimum(mask, 1.0 - np.array(img_tmp, dtype=np.float32))
    return np.clip(mask, 0, 1).astype(np.float32)


def make_sinusoid(H, W, kx, ky, phase_deg, amp):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.cos(2*np.pi*(kx/W*x + ky/H*y) + np.radians(phase_deg))).astype(np.float32)


def make_sinusoid_rgb(H, W, kx_r, ky_r, kx_g, ky_g, kx_b, ky_b, phase_deg, amp):
    """Per-channel sinusoid — different frequency for each RGB channel."""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pat = np.zeros((H, W, 3), dtype=np.float32)
    for c, (kx, ky) in enumerate([(kx_r, ky_r), (kx_g, ky_g), (kx_b, ky_b)]):
        pat[:,:,c] = amp * np.cos(2*np.pi*(kx/W*x + ky/H*y) + np.radians(phase_deg))
    return pat


def make_stripes(H, W, spacing, angle_deg, amp):
    """Line/stripe pattern at given angle and spacing."""
    angle = math.radians(angle_deg)
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Project onto direction perpendicular to stripe orientation
    proj = x * math.cos(angle) + y * math.sin(angle)
    return (amp * np.sign(np.sin(2 * np.pi * proj / spacing))).astype(np.float32)


def make_lychrel_pattern(H, W, lychrel_num, amp):
    """
    Lychrel-inspired spatial pattern based on the number's digits.
    Uses the number as a spatial frequency and creates a reverse-and-add
    interference pattern.
    """
    # k = lychrel_num mod (H//2) to keep within Nyquist
    k = lychrel_num % (H // 2)
    if k == 0: k = 1
    # Reverse of the number's digits as second frequency
    rev = int(str(lychrel_num)[::-1])
    k2 = rev % (H // 2)
    if k2 == 0: k2 = 1
    # Interference between the number and its reverse
    pat1 = make_sinusoid(H, W, k, k, 0, amp)
    pat2 = make_sinusoid(H, W, k2, k2, 0, amp)
    return ((pat1 + pat2) * 0.5).astype(np.float32)


def apply_patch_to_image(arr, patch_texture, mask, cx, cy):
    """
    Apply a textured patch to the image at (cx, cy) using the given mask.
    patch_texture: (H, W) or (H, W, 3) — the texture to fill the patch with
    mask: (H, W) — 1 inside patch, 0 outside
    """
    H, W, _ = arr.shape
    out = arr.copy()
    
    if patch_texture.ndim == 2:
        # Same texture for all channels
        for c in range(3):
            out[:,:,c] = out[:,:,c] * (1 - mask) + patch_texture * mask
    else:
        # Per-channel texture
        for c in range(3):
            out[:,:,c] = out[:,:,c] * (1 - mask) + patch_texture[:,:,c] * mask
    
    return np.clip(out, 0, 1)


def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return arr, c


def get_dets_ultralytics(model, pil_img, conf=0.1):
    results = model(pil_img, verbose=False)
    dets = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            dets.append({
                "class_id": cls_id,
                "class_name": model.names.get(cls_id, f"c{cls_id}"),
                "confidence": float(boxes.conf[i].item()),
                "bbox": [float(boxes.xyxy[i][0]), float(boxes.xyxy[i][1]),
                         float(boxes.xyxy[i][2]), float(boxes.xyxy[i][3])],
            })
    return [d for d in dets if d["confidence"] >= conf]


def get_dets_yolov3(model, img_tensor, conf=0.1):
    with torch.no_grad():
        output = model(img_tensor)
    dets = []
    if output is None: return dets
    out = output.cpu().numpy()
    if out.ndim == 3: out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
            })
    return dets


def draw_dets(pil_img, dets, title, save_path, patch_outline=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pil_img)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{d['class_name']} {d['confidence']:.2f}", color="lime", fontsize=8,
                fontweight="bold", bbox=dict(facecolor="black", alpha=0.5))
    if patch_outline:
        poly = patches.Polygon(patch_outline, closed=True, linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(poly)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_patch_visualization(mask, texture, save_path, title="Patch"):
    """Save a visualization of the patch mask and texture."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Mask")
    axes[0].axis("off")
    if texture.ndim == 2:
        axes[1].imshow(texture, cmap="viridis")
    else:
        axes[1].imshow(np.clip(texture, 0, 1))
    axes[1].set_title("Texture")
    axes[1].axis("off")
    # Masked texture
    if texture.ndim == 2:
        masked = texture * mask
        axes[2].imshow(masked, cmap="viridis")
    else:
        masked = texture * mask[:,:,None]
        axes[2].imshow(np.clip(masked, 0, 1))
    axes[2].set_title("Masked Texture")
    axes[2].axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("Deformable Triangular Patch Attack — Shape Matters + Frequency Injection")
    print("="*70)
    print(f"Device: {DEVICE}")

    SIZE = 416  # Use 416 for all models for direct comparison
    H, W = SIZE, SIZE
    
    # Load image and find person bounding box to place patch on torso
    arr_w, img_w_pil = load_image(IMG_WITH, SIZE)
    arr_wo, img_wo_pil = load_image(IMG_WITHOUT, SIZE)
    
    # Person is roughly centered in the image — place patch on torso
    # Based on prior analyses, person occupies roughly center of image
    # Torso center: approximately (208, 240) — slightly below center
    patch_cx, patch_cy = 208, 240
    
    # ============================================================
    # Define patch shapes (deformable masks)
    # ============================================================
    print("\n--- Generating patch shapes ---")
    
    shapes = {}
    
    # 1. Circle (R=32 rays, equal length) — baseline
    r_circle = 80
    shapes["circle_r80"] = make_deformable_mask(H, W, patch_cx, patch_cy, [r_circle]*32, 32)
    
    # 2. Triangle (R=3, equal length)
    r_tri = 100
    shapes["triangle_r100"] = make_deformable_mask(H, W, patch_cx, patch_cy, [r_tri]*3, 3)
    
    # 3. Deformed triangle (R=3, unequal rays — asymmetric)
    shapes["triangle_deformed"] = make_deformable_mask(H, W, patch_cx, patch_cy, [120, 80, 100], 3)
    
    # 4. Hexagon (R=6)
    shapes["hexagon_r80"] = make_deformable_mask(H, W, patch_cx, patch_cy, [80]*6, 6)
    
    # 5. Deformed octagon (R=8, varying rays — clothing-like irregular shape)
    shapes["octagon_deformed"] = make_deformable_mask(H, W, patch_cx, patch_cy, [90, 70, 85, 75, 95, 65, 80, 88], 8)
    
    # 6. Sierpinski triangle (depth=3 — 27 sub-triangles)
    shapes["sierpinski_d3"] = (make_sierpinski_mask(H, W, patch_cx, patch_cy, 180, depth=3), None)
    
    # 7. Sierpinski triangle (depth=4 — 81 sub-triangles, more detail)
    shapes["sierpinski_d4"] = (make_sierpinski_mask(H, W, patch_cx, patch_cy, 180, depth=4), None)
    
    # 8. Nested triangles (5 layers, alternating)
    shapes["nested_tri_5"] = (make_nested_triangles_mask(H, W, patch_cx, patch_cy, 180, 5), None)
    
    # 9. Small triangle (R=3, r=60 — smaller patch, harder but more realistic)
    shapes["triangle_small_r60"] = make_deformable_mask(H, W, patch_cx, patch_cy, [60]*3, 3)
    
    # 10. Large deformed shape (R=12, irregular — maximizes coverage)
    shapes["deformed_r12_large"] = make_deformable_mask(H, W, patch_cx, patch_cy, 
                                                         [110, 90, 105, 85, 100, 95, 115, 80, 100, 90, 110, 85], 12)
    
    # Normalize shapes to (mask, endpoints) format
    normalized_shapes = {}
    for name, val in shapes.items():
        if isinstance(val, tuple):
            normalized_shapes[name] = val
        else:
            mask, endpoints = val
            normalized_shapes[name] = (mask, endpoints)
    shapes = normalized_shapes
    
    # Save shape visualizations
    for name, (mask, endpoints) in shapes.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mask, cmap="gray")
        if endpoints:
            poly = patches.Polygon(endpoints, closed=True, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(poly)
        ax.set_title(f"Shape: {name}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shape_{name}.png", dpi=100)
        plt.close()
    
    # ============================================================
    # Define patch textures (the frequency patterns)
    # ============================================================
    print("--- Generating patch textures ---")
    
    textures = {}
    amp = 0.30  # Higher amplitude since patch is small
    
    # Frequency-based textures (our key findings)
    textures["k167_d"] = make_sinusoid(H, W, 167, 167, 0, amp)
    textures["k208_d"] = make_sinusoid(H, W, 208, 208, 0, amp)
    textures["k196_d"] = make_sinusoid(H, W, 196, 196, 0, amp)
    
    # Per-channel color textures — different frequency per RGB channel
    textures["rgb_167_208_196"] = make_sinusoid_rgb(H, W, 167, 167, 208, 208, 196, 196, 0, amp)
    textures["rgb_167_196_208"] = make_sinusoid_rgb(H, W, 167, 167, 196, 196, 208, 208, 0, amp)
    textures["rgb_208_167_196"] = make_sinusoid_rgb(H, W, 208, 208, 167, 167, 196, 196, 0, amp)
    
    # Single-channel injection (only one color channel modified)
    for ch_name, ch_idx in [("red", 0), ("green", 1), ("blue", 2)]:
        pat = np.zeros((H, W, 3), dtype=np.float32)
        kx, ky = 167, 167
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pat[:,:,ch_idx] = amp * np.cos(2*np.pi*(kx/W*x + ky/H*y))
        textures[f"single_{ch_name}_k167"] = pat
    
    # Lychrel number patterns
    for ln in LYCHREL_NUMBERS:
        textures[f"lychrel_{ln}"] = make_lychrel_pattern(H, W, ln, amp)
    
    # Stripe/line patterns
    textures["stripes_h_13px"] = make_stripes(H, W, 13, 0, amp)  # horizontal stripes, 13px spacing
    textures["stripes_d_13px"] = make_stripes(H, W, 13, 45, amp)  # diagonal stripes
    textures["stripes_h_32px"] = make_stripes(H, W, 32, 0, amp)  # 32px = power of 2
    textures["stripes_v_13px"] = make_stripes(H, W, 13, 90, amp)  # vertical
    
    # Composite: k167 suppressor + k208 hallucinator
    comp = make_sinusoid(H, W, 167, 167, 0, amp) + make_sinusoid(H, W, 208, 208, 0, amp)
    comp = (comp / np.abs(comp).max() * amp).astype(np.float32)
    textures["composite_167_208"] = comp
    
    # Composite: k167 + 1/196 offset (anti-closure)
    comp2 = make_sinusoid(H, W, 167, 167, 0, amp) + np.full((H, W), 1.0/196.0, dtype=np.float32)
    comp2 = (comp2 / np.abs(comp2).max() * amp).astype(np.float32)
    textures["composite_167_inv196"] = comp2
    
    # Control: random noise
    rng = np.random.RandomState(42)
    textures["random_noise"] = (rng.randn(H, W) * amp * 0.5).astype(np.float32)
    
    # Control: uniform gray (no spatial variation)
    textures["uniform_gray"] = np.full((H, W), 0.5, dtype=np.float32)
    
    # Save texture visualizations
    for name, tex in textures.items():
        save_patch_visualization(np.ones((H, W), dtype=np.float32), tex,
                                 f"{OUTPUT_DIR}/texture_{name}.png", f"Texture: {name}")
    
    # ============================================================
    # Test all shape x texture combinations
    # ============================================================
    print(f"\n--- Testing {len(shapes)} shapes x {len(textures)} textures = {len(shapes)*len(textures)} combinations ---")
    
    all_results = {}
    csv_rows = []
    
    # Load YOLOv3
    print("\nLoading YOLOv3...")
    from pytorchyolo.models import Darknet
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters(): p.requires_grad_(False)
    
    # Baseline
    tensor_w = torch.from_numpy(arr_w).permute(2,0,1).unsqueeze(0).to(DEVICE)
    det_w_baseline = get_dets_yolov3(v3_model, tensor_w, conf=0.1)
    print(f"  YOLOv3 baseline: {len(det_w_baseline)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w_baseline]}")
    
    v3_results = {}
    interesting_count = 0
    
    for shape_name, (mask, endpoints) in shapes.items():
        for tex_name, texture in textures.items():
            combo_name = f"{shape_name}__{tex_name}"
            
            # Apply patch
            arr_mod = apply_patch_to_image(arr_w, texture, mask, patch_cx, patch_cy)
            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
            dets = get_dets_yolov3(v3_model, tensor_mod, conf=0.1)
            
            # Analyze
            classes = {}
            for d in dets:
                cn = d["class_name"]
                if cn not in classes: classes[cn] = []
                classes[cn].append(round(d["confidence"], 3))
            
            person_count = len(classes.get("person", []))
            total = len(dets)
            suppressed = total == 0
            person_suppressed = person_count == 0 and len(det_w_baseline) > 0
            hallucinated = any(c != "person" for c in classes)
            person_reduced = person_count < len([d for d in det_w_baseline if d["class_name"] == "person"])
            
            # Patch area (percentage of image)
            patch_area = float(np.mean(mask)) * 100
            
            v3_results[combo_name] = {
                "shape": shape_name, "texture": tex_name,
                "total_dets": total, "person_dets": person_count,
                "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
                "suppressed": suppressed, "person_suppressed": person_suppressed,
                "hallucinated": hallucinated, "person_reduced": person_reduced,
                "patch_area_pct": patch_area,
            }
            
            # Print interesting results
            tag = ""
            if suppressed: tag = " [TOTAL_SUPPRESS]"
            elif person_suppressed: tag = " [PERSON_SUPPRESS]"
            elif hallucinated: tag = f" [HALLUC:{','.join(c for c in classes if c != 'person')}]"
            elif person_reduced: tag = f" [person {len([d for d in det_w_baseline if d['class_name']=='person'])}->{person_count}]"
            
            if tag:
                print(f"  {combo_name:>45}: {total:2d} dets, person={person_count}, area={patch_area:.1f}%{tag}")
                interesting_count += 1
                # Save visualization for interesting cases
                pil_mod = Image.fromarray((arr_mod * 255).astype(np.uint8))
                draw_dets(pil_mod, dets, f"YOLOv3 {combo_name}{tag}",
                          f"{OUTPUT_DIR}/v3_{combo_name}.png", patch_outline=endpoints)
            
            for cls_name, confs in classes.items():
                csv_rows.append({"model": "YOLOv3", "shape": shape_name, "texture": tex_name,
                                 "class": cls_name, "count": len(confs),
                                 "max_conf": max(confs), "total": total,
                                 "patch_area": patch_area, "person_dets": person_count})
    
    all_results["YOLOv3"] = v3_results
    print(f"\n  YOLOv3: {interesting_count} interesting results out of {len(shapes)*len(textures)} combinations")
    
    del v3_model
    torch.cuda.empty_cache()
    
    # ============================================================
    # Test on YOLOv8, YOLO11, YOLO26
    # ============================================================
    model_configs = [
        ("YOLOv8", r"C:\Users\carso\Desktop\YODO\YOLOv8\yolov8l.pt"),
        ("YOLO11", r"C:\Users\carso\Desktop\YODO\YOLO11\yolo11l.pt"),
        ("YOLO26", r"C:\Users\carso\Desktop\YODO\YOLO26\yolo26l.pt"),
    ]
    
    for model_name, model_path in model_configs:
        print(f"\n{'='*70}")
        print(f"{model_name} ({SIZE}x{SIZE})")
        print(f"{'='*70}")
        
        ul_model = YOLO(model_path)
        ul_model.to(DEVICE)
        
        # Baseline
        det_w_ul = get_dets_ultralytics(ul_model, img_w_pil, conf=0.1)
        print(f"  {model_name} baseline: {len(det_w_ul)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w_ul[:5]]}")
        
        ul_results = {}
        interesting_count = 0
        
        for shape_name, (mask, endpoints) in shapes.items():
            for tex_name, texture in textures.items():
                combo_name = f"{shape_name}__{tex_name}"
                
                arr_mod = apply_patch_to_image(arr_w, texture, mask, patch_cx, patch_cy)
                pil_mod = Image.fromarray((arr_mod * 255).astype(np.uint8))
                dets = get_dets_ultralytics(ul_model, pil_mod, conf=0.1)
                
                classes = {}
                for d in dets:
                    cn = d["class_name"]
                    if cn not in classes: classes[cn] = []
                    classes[cn].append(round(d["confidence"], 3))
                
                person_count = len(classes.get("person", []))
                total = len(dets)
                suppressed = total == 0
                person_suppressed = person_count == 0 and len([d for d in det_w_ul if d["class_name"] == "person"]) > 0
                hallucinated = any(c != "person" for c in classes)
                person_reduced = person_count < len([d for d in det_w_ul if d["class_name"] == "person"])
                patch_area = float(np.mean(mask)) * 100
                
                ul_results[combo_name] = {
                    "shape": shape_name, "texture": tex_name,
                    "total_dets": total, "person_dets": person_count,
                    "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
                    "suppressed": suppressed, "person_suppressed": person_suppressed,
                    "hallucinated": hallucinated, "person_reduced": person_reduced,
                    "patch_area_pct": patch_area,
                }
                
                tag = ""
                if suppressed: tag = " [TOTAL_SUPPRESS]"
                elif person_suppressed: tag = " [PERSON_SUPPRESS]"
                elif hallucinated: tag = f" [HALLUC:{','.join(c for c in classes if c != 'person')}]"
                elif person_reduced: tag = f" [person {len([d for d in det_w_ul if d['class_name']=='person'])}->{person_count}]"
                
                if tag:
                    print(f"  {combo_name:>45}: {total:2d} dets, person={person_count}, area={patch_area:.1f}%{tag}")
                    interesting_count += 1
                    draw_dets(pil_mod, dets, f"{model_name} {combo_name}{tag}",
                              f"{OUTPUT_DIR}/{model_name.lower().replace(' ','')}_{combo_name}.png",
                              patch_outline=endpoints)
                
                for cls_name, cls_data in classes.items():
                    csv_rows.append({"model": model_name, "shape": shape_name, "texture": tex_name,
                                     "class": cls_name, "count": cls_data["count"],
                                     "max_conf": cls_data["max_conf"], "total": total,
                                     "patch_area": patch_area, "person_dets": person_count})
        
        all_results[model_name] = ul_results
        print(f"\n  {model_name}: {interesting_count} interesting results out of {len(shapes)*len(textures)} combinations")
        
        del ul_model
        torch.cuda.empty_cache()
    
    # ============================================================
    # CROSS-MODEL ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL ANALYSIS")
    print(f"{'='*70}")
    
    # Find combinations that work across ALL models
    print("\n--- COMBINATIONS THAT SUPPRESS PERSON ON ALL MODELS ---")
    all_model_keys = list(all_results.keys())
    for combo_name in v3_results:
        suppress_count = sum(1 for mk in all_model_keys 
                            if all_results[mk].get(combo_name, {}).get("person_suppressed", False))
        if suppress_count >= 2:  # at least 2 models
            areas = [all_results[mk].get(combo_name, {}).get("patch_area_pct", 0) for mk in all_model_keys]
            print(f"  {combo_name:>45}: person suppressed on {suppress_count}/{len(all_model_keys)} models, area={areas[0]:.1f}%")
    
    print("\n--- COMBINATIONS THAT CAUSE HALLUCINATION ---")
    for combo_name in v3_results:
        halluc_count = sum(1 for mk in all_model_keys 
                          if all_results[mk].get(combo_name, {}).get("hallucinated", False))
        if halluc_count > 0:
            print(f"  {combo_name:>45}: hallucinated on {halluc_count}/{len(all_model_keys)} models")
    
    # Best shapes (which shapes suppress most often across models/textures)
    print("\n--- BEST SHAPES (most frequent person suppression) ---")
    shape_stats = {}
    for shape_name, _ in shapes.items():
        suppress_total = 0
        combo_total = 0
        for mk in all_model_keys:
            for combo_name in all_results[mk]:
                if combo_name.startswith(shape_name + "__"):
                    combo_total += 1
                    if all_results[mk][combo_name].get("person_suppressed", False):
                        suppress_total += 1
        if combo_total > 0:
            shape_stats[shape_name] = (suppress_total, combo_total, suppress_total/combo_total)
    
    for shape_name, (s, t, ratio) in sorted(shape_stats.items(), key=lambda x: -x[1][2]):
        print(f"  {shape_name:>25}: {s}/{t} = {ratio:.1%}")
    
    # Best textures
    print("\n--- BEST TEXTURES (most frequent person suppression) ---")
    tex_stats = {}
    for tex_name in textures:
        suppress_total = 0
        combo_total = 0
        for mk in all_model_keys:
            for combo_name in all_results[mk]:
                if combo_name.endswith("__" + tex_name):
                    combo_total += 1
                    if all_results[mk][combo_name].get("person_suppressed", False):
                        suppress_total += 1
        if combo_total > 0:
            tex_stats[tex_name] = (suppress_total, combo_total, suppress_total/combo_total)
    
    for tex_name, (s, t, ratio) in sorted(tex_stats.items(), key=lambda x: -x[1][2])[:15]:
        print(f"  {tex_name:>25}: {s}/{t} = {ratio:.1%}")
    
    # ============================================================
    # SAVE
    # ============================================================
    print(f"\n{'='*70}")
    print("Saving results...")
    
    json_path = f"{OUTPUT_DIR}/triangular_patch.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")
    
    csv_path = f"{OUTPUT_DIR}/triangular_patch.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            all_fields = set()
            for row in csv_rows: all_fields.update(row.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_fields))
            w.writeheader()
            for row in csv_rows: w.writerow({k: row.get(k, "") for k in sorted(all_fields)})
    print(f"Saved CSV: {csv_path}")
    
    # Plot: shape effectiveness comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    shape_names_sorted = sorted(shape_stats.keys())
    shape_ratios = [shape_stats[sn][2] for sn in shape_names_sorted]
    ax.bar(range(len(shape_names_sorted)), shape_ratios, color="steelblue")
    ax.set_xticks(range(len(shape_names_sorted)))
    ax.set_xticklabels(shape_names_sorted, rotation=45, ha="right")
    ax.set_ylabel("Person Suppression Rate")
    ax.set_title("Shape Effectiveness Across All Models & Textures")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shape_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot: texture effectiveness comparison (top 15)
    fig, ax = plt.subplots(figsize=(14, 6))
    top_tex = sorted(tex_stats.items(), key=lambda x: -x[1][2])[:15]
    tex_names = [t[0] for t in top_tex]
    tex_ratios = [t[1][2] for t in top_tex]
    ax.barh(range(len(tex_names)), tex_ratios, color="coral")
    ax.set_yticks(range(len(tex_names)))
    ax.set_yticklabels(tex_names)
    ax.set_xlabel("Person Suppression Rate")
    ax.set_title("Top 15 Textures Across All Models & Shapes")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/texture_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plots: {OUTPUT_DIR}/shape_comparison.png, texture_comparison.png")
    print("\nDONE")

if __name__ == "__main__":
    main()
