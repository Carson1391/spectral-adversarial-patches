"""
Gradient-optimized adversarial patch for YOLOv3.

Objective: Find patch pixels that maximize embedding corruption at L81/L93/L105
while keeping the person detected (box stays) and maintaining printability (TV loss).

Loss = -w_emb * sum(emb_l2_per_layer)        # maximize embedding shift
       + w_supp * suppression_penalty          # penalize losing person detections
       - w_cos * cosine_similarity             # push embedding direction away
       + w_tv  * total_variation               # printability/smoothness

EOT: Each iteration applies random rotation, scale, blur to the patch before
compositing, so the optimized patch is robust to physical transformations.

Uses our existing PyTorch-YOLOv3 model and withhuman.png base image.
No pytorch3d/open3d needed - pure gradient descent through frozen YOLOv3.
"""

import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\grad_optim"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416

# Patch placement (torso center from find_persons.py)
PATCH_CX, PATCH_CY = 183, 292
WEARER_THRESHOLD = 60

# Detection head layers in YOLOv3
DETECTION_LAYERS = {
    "L81_52x52": 81,
    "L93_26x26": 93,
    "L105_13x13": 105,
}

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

# Optimization hyperparameters
PATCH_SIZE_416 = 80  # patch size in 416 space (~19% of image width)
PATCH_RES = 300  # optimization resolution (pixels)
NUM_EPOCHS = 500
LR = 0.05
BATCH_EOT = 4  # number of EOT transforms per step

# Loss weights
W_EMB = 1.0      # embedding corruption weight
W_SUPP = 0.5     # suppression penalty weight (keep person detected)
W_COS = 0.3      # cosine direction corruption weight
W_TV = 0.01      # total variation (printability)


# ============================================================
# Differentiable YOLOv3 forward pass with feature capture
# ============================================================

def forward_capture_diff(model, x):
    """
    Differentiable forward pass through YOLOv3.
    Captures feature maps at detection head layers WITHOUT detaching.
    Returns: (caps_dict, output)
    caps_dict[layer_idx] = feature tensor with gradients
    """
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
            x = mo[0](x, IMG_SIZE)  # Pass 416 so bbox coords are in 416 space
        if md["type"] == "convolutional":
            caps[i] = x  # NO detach - keep gradients
        los.append(x)
    return caps, x


# ============================================================
# Image loading
# ============================================================

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


# ============================================================
# Differentiable patch compositing
# ============================================================

def composite_patch_diff(base_img, patch_rgb, cx, cy, ps):
    """
    Differentiable compositing of patch onto base image.
    Uses additive perturbation: base + (patch - 0.5) * mask
    
    base_img: (1, 3, H, W) tensor on device
    patch_rgb: (1, 3, ps, ps) tensor, values in [0, 1]
    cx, cy: patch center in image space
    ps: patch size in image space
    
    Returns: (1, 3, H, W) composited image
    """
    H, W = base_img.shape[2], base_img.shape[3]
    
    # Create smooth circular mask via gaussian falloff
    yy, xx = torch.meshgrid(
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        indexing="ij"
    )
    # Circular mask with soft edge
    r = ps / 2.0
    dist = torch.sqrt((xx - r + 0.5)**2 + (yy - r + 0.5)**2)
    mask = torch.clamp(1.0 - (dist - r * 0.85) / (r * 0.15), 0.0, 1.0)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, ps, ps)
    
    # Patch placement
    x0 = int(cx - ps // 2)
    y0 = int(cy - ps // 2)
    
    # Clamp to image bounds
    px0, py0 = max(0, x0), max(0, y0)
    px1 = min(W, x0 + ps)
    py1 = min(H, y0 + ps)
    sx0, sy0 = px0 - x0, py0 - y0
    sx1 = sx0 + (px1 - px0)
    sy1 = sy0 + (py1 - py0)
    
    # Build full-size patch and mask tensors
    full_patch = torch.zeros_like(base_img)
    full_mask = torch.zeros(1, 1, H, W, device=base_img.device, dtype=base_img.dtype)
    
    full_patch[:, :, py0:py1, px0:px1] = patch_rgb[:, :, sy0:sy1, sx0:sx1]
    full_mask[:, :, py0:py1, px0:px1] = mask[:, :, sy0:sy1, sx0:sx1]
    
    # Additive compositing: base + (patch - 0.5) * mask
    # Patch [0,1] -> perturbation [-0.5, +0.5] * mask * amplitude
    # amplitude=0.3 gives +-0.15 perturbation range, strong enough for gradients
    composited = base_img + (full_patch - 0.5) * full_mask * 0.3
    composited = torch.clamp(composited, 0.0, 1.0)
    
    return composited


# ============================================================
# EOT transforms (differentiable)
# ============================================================

def eot_transform(patch, ps):
    """
    Apply random EOT transformation to patch for physical robustness.
    Random rotation, scale, translation, and Gaussian blur.
    
    patch: (1, 3, ps, ps) tensor
    Returns: transformed (1, 3, ps, ps) tensor
    """
    # Random rotation (-15 to +15 degrees)
    angle = (torch.rand(1, device=patch.device) - 0.5) * 30.0
    # Random scale (0.7 to 1.3)
    scale = 0.7 + torch.rand(1, device=patch.device) * 0.6
    # Random translation (-10% to +10%)
    tx = (torch.rand(1, device=patch.device) - 0.5) * 0.2 * ps
    ty = (torch.rand(1, device=patch.device) - 0.5) * 0.2 * ps
    
    # Build affine matrix
    theta = torch.zeros(1, 2, 3, device=patch.device, dtype=patch.dtype)
    cos_a = torch.cos(angle * math.pi / 180.0)
    sin_a = torch.sin(angle * math.pi / 180.0)
    theta[:, 0, 0] = cos_a / scale
    theta[:, 0, 1] = -sin_a / scale
    theta[:, 0, 2] = tx / ps * 2  # normalize to [-1, 1] grid space
    theta[:, 1, 0] = sin_a / scale
    theta[:, 1, 1] = cos_a / scale
    theta[:, 1, 2] = ty / ps * 2
    
    grid = F.affine_grid(theta, patch.shape, align_corners=False)
    transformed = F.grid_sample(patch, grid, align_corners=False, padding_mode="reflection")
    
    # Random Gaussian blur via conv2d with random kernel
    blur_radius = torch.randint(0, 3, (1,)).item()
    if blur_radius > 0:
        ksize = blur_radius * 2 + 1
        sigma = 0.5 + torch.rand(1).item() * 1.0
        # Create 1D Gaussian kernel
        k1d = torch.exp(-torch.arange(-ksize//2, ksize//2+1, device=patch.device, dtype=patch.dtype)**2 / (2 * sigma**2))
        k1d = k1d / k1d.sum()
        # Separable 2D blur
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        k2d = k2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        transformed = F.conv2d(transformed, k2d, padding=ksize//2, groups=3)
    
    return transformed


# ============================================================
# Loss functions
# ============================================================

def tv_loss(patch):
    """Total variation loss for printability."""
    tv_h = torch.mean(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
    return tv_h + tv_w


def extract_embedding_diff(caps, layer_idx, spatial_x, spatial_y, clean_emb):
    """
    Extract embedding at spatial location and compute L2 distance to clean.
    Differentiable - gradients flow through caps.
    
    caps: dict of feature tensors {layer_idx: tensor(1, C, H, W)}
    layer_idx: which layer (81, 93, 105)
    spatial_x, spatial_y: pixel coords in 416 space
    clean_emb: (C,) numpy array - clean embedding to compare against
    
    Returns: (l2_dist, cos_sim) tensors
    """
    feat = caps[layer_idx]  # (1, C, fH, fW)
    fH, fW = feat.shape[2], feat.shape[3]
    # Map 416-space coordinates to feature map coordinates
    fx = int(spatial_x / IMG_SIZE * fW)
    fy = int(spatial_y / IMG_SIZE * fH)
    fx = max(0, min(fW - 1, fx))
    fy = max(0, min(fH - 1, fy))
    
    # Extract channel vector at this spatial cell
    vec = feat[0, :, fy, fx]  # (C,)
    
    clean = torch.from_numpy(clean_emb).to(vec.device, dtype=vec.dtype)
    
    l2 = torch.norm(vec - clean)
    cos = F.cosine_similarity(vec.unsqueeze(0), clean.unsqueeze(0)).squeeze()
    
    return l2, cos


def get_dets_from_output(output, conf=0.1):
    """Parse YOLOv3 output to get person detections."""
    dets = []
    if output is None:
        return dets
    out = output.detach().cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            if cls == 0:  # person
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                dets.append({"cx": cx, "cy": cy, "w": w, "h": h, "conf": float(row[4])})
    return dets


# ============================================================
# Main optimization
# ============================================================

def run_optimization():
    print(f"Device: {DEVICE}")
    print(f"Patch resolution: {PATCH_RES}x{PATCH_RES}")
    print(f"Patch size in 416 space: {PATCH_SIZE_416}px")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}, EOT batch: {BATCH_EOT}")
    print(f"Loss weights: emb={W_EMB}, supp={W_SUPP}, cos={W_COS}, tv={W_TV}")
    print()
    
    # Load model
    print("Loading YOLOv3...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Model loaded on {DEVICE}")
    
    # Load base image
    arr_base = load_image(IMG_WITH, IMG_SIZE)
    base_tensor = torch.from_numpy(arr_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Get clean baseline: forward pass without patch, capture embeddings
    print("Computing clean baseline embeddings...")
    with torch.no_grad():
        clean_caps, clean_output = forward_capture_diff(model, base_tensor)
        clean_dets = get_dets_from_output(clean_output)
        
        # Find wearer detection (closest to patch center)
        wearer_dets = []
        for d in clean_dets:
            dist = math.sqrt((d["cx"] - PATCH_CX)**2 + (d["cy"] - PATCH_CY)**2)
            if dist < WEARER_THRESHOLD:
                wearer_dets.append(d)
        
        if not wearer_dets:
            # Use closest detection
            if clean_dets:
                wearer_dets = [min(clean_dets, key=lambda d: 
                    math.sqrt((d["cx"]-PATCH_CX)**2 + (d["cy"]-PATCH_CY)**2))]
            else:
                print("ERROR: No person detections in base image!")
                return
        
        wearer = wearer_dets[0]
        print(f"  Wearer detection: cx={wearer['cx']:.1f}, cy={wearer['cy']:.1f}, conf={wearer['conf']:.3f}")
        print(f"  Total persons in base: {len(clean_dets)}")
        
        # Extract clean embeddings at wearer location for all 3 layers
        clean_embs = {}
        for layer_name, layer_idx in DETECTION_LAYERS.items():
            feat = clean_caps[layer_idx]
            fH, fW = feat.shape[2], feat.shape[3]
            fx = int(wearer["cx"] / IMG_SIZE * fW)
            fy = int(wearer["cy"] / IMG_SIZE * fH)
            fx = max(0, min(fW - 1, fx))
            fy = max(0, min(fH - 1, fy))
            clean_embs[layer_name] = feat[0, :, fy, fx].cpu().numpy()
            print(f"  {layer_name}: clean embedding shape {clean_embs[layer_name].shape}")
    
    # Initialize patch: random noise centered at 0.5 (gray)
    # patch is (1, 3, PATCH_RES, PATCH_RES) in [0, 1]
    patch = torch.rand(1, 3, PATCH_RES, PATCH_RES, device=DEVICE) * 0.4 + 0.3
    patch.requires_grad_(True)
    
    # Optimizer
    optimizer = torch.optim.Adam([patch], lr=LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)
    
    # Logging
    history = []
    best_emb_l2 = 0.0
    best_patch = None
    
    print(f"\n{'='*70}")
    print("STARTING OPTIMIZATION")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        
        total_emb_l2 = torch.zeros(1, device=DEVICE)
        total_cos = torch.zeros(1, device=DEVICE)
        total_supp = torch.zeros(1, device=DEVICE)
        total_tv = torch.zeros(1, device=DEVICE)
        n_valid = 0
        
        for eot_idx in range(BATCH_EOT):
            # Apply EOT transform
            patch_eot = eot_transform(patch, PATCH_RES)
            
            # Resize to 416-space patch size
            patch_416 = F.interpolate(patch_eot, size=(PATCH_SIZE_416, PATCH_SIZE_416), 
                                       mode='bilinear', align_corners=False)
            
            # Composite onto base image (differentiable)
            composited = composite_patch_diff(base_tensor, patch_416, PATCH_CX, PATCH_CY, PATCH_SIZE_416)
            
            # Forward pass through YOLOv3 (differentiable)
            caps, output = forward_capture_diff(model, composited)
            
            # Extract embeddings at wearer location for all 3 layers
            emb_l2_sum = torch.zeros(1, device=DEVICE)
            cos_sum = torch.zeros(1, device=DEVICE)
            
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                l2, cos = extract_embedding_diff(
                    caps, layer_idx, wearer["cx"], wearer["cy"], clean_embs[layer_name]
                )
                emb_l2_sum = emb_l2_sum + l2
                cos_sum = cos_sum + cos
            
            # Average over layers
            emb_l2_avg = emb_l2_sum / len(DETECTION_LAYERS)
            cos_avg = cos_sum / len(DETECTION_LAYERS)
            
            # Suppression penalty: penalize if person detection confidence drops
            # We want the box to stay, so penalize low confidence
            # Use the raw output to find person detections near wearer
            with torch.no_grad():
                dets = get_dets_from_output(output, conf=0.05)
                wearer_dets_comp = [d for d in dets if 
                    math.sqrt((d["cx"]-wearer["cx"])**2 + (d["cy"]-wearer["cy"])**2) < WEARER_THRESHOLD]
            
            if wearer_dets_comp:
                # Person still detected - no suppression penalty
                supp_pen = torch.zeros(1, device=DEVICE)
            else:
                # Person suppressed - penalize
                # Use max objectness in the wearer's spatial region as differentiable proxy
                # Get the YOLO output and find the max confidence near wearer location
                # output shape: (1, N, 5+num_classes) or (1, N, 6)
                if output is not None and output.shape[1] > 0:
                    # Find detections near wearer by computing distances
                    out_cpu = output.detach().cpu().numpy()
                    if out_cpu.ndim == 3:
                        out_cpu = out_cpu[0]
                    # Find max objectness in wearer region
                    wearer_region_mask = []
                    for row in out_cpu:
                        if len(row) >= 6:
                            cx, cy = float(row[0]), float(row[1])
                            dist = math.sqrt((cx - wearer["cx"])**2 + (cy - wearer["cy"])**2)
                            if dist < WEARER_THRESHOLD and int(row[5]) == 0:
                                wearer_region_mask.append(float(row[4]))
                    
                    if wearer_region_mask:
                        max_conf = max(wearer_region_mask)
                        # Penalize proportional to confidence drop
                        supp_pen = torch.tensor(max(0.0, wearer["conf"] - max_conf) * 10, 
                                               device=DEVICE, dtype=patch.dtype)
                    else:
                        # Full suppression - heavy penalty
                        supp_pen = torch.tensor(wearer["conf"] * 10, device=DEVICE, dtype=patch.dtype)
                else:
                    supp_pen = torch.tensor(5.0, device=DEVICE, dtype=patch.dtype)
            
            total_emb_l2 = total_emb_l2 + emb_l2_avg
            total_cos = total_cos + cos_avg
            total_supp = total_supp + supp_pen
            n_valid += 1
        
        # Average over EOT batch
        if n_valid > 0:
            avg_emb_l2 = total_emb_l2 / n_valid
            avg_cos = total_cos / n_valid
            avg_supp = total_supp / n_valid
        else:
            continue
        
        # TV loss on the patch itself
        total_tv = tv_loss(patch)
        
        # Combined loss:
        # Maximize embedding L2 -> minimize -emb_l2
        # Keep person detected -> minimize supp penalty
        # Corrupt direction -> minimize -cos (push cosine down)
        # Printability -> minimize tv
        loss = -W_EMB * avg_emb_l2 + W_SUPP * avg_supp - W_COS * avg_cos + W_TV * total_tv
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Clamp patch to [0, 1]
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
        
        # Logging
        emb_l2_val = avg_emb_l2.item()
        cos_val = avg_cos.item()
        supp_val = avg_supp.item()
        tv_val = total_tv.item()
        loss_val = loss.item()
        
        history.append({
            "epoch": epoch,
            "loss": loss_val,
            "emb_l2": emb_l2_val,
            "cos": cos_val,
            "supp_penalty": supp_val,
            "tv": tv_val,
            "lr": scheduler.get_last_lr()[0],
        })
        
        # Track best patch by embedding L2 (with low suppression)
        if emb_l2_val > best_emb_l2 and supp_val < 0.5:
            best_emb_l2 = emb_l2_val
            best_patch = patch.detach().cpu().clone()
        
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            elapsed = time.time() - t_start
            print(f"  E{epoch:4d}  loss={loss_val:8.3f}  emb_l2={emb_l2_val:7.2f}  "
                  f"cos={cos_val:.4f}  supp={supp_val:5.2f}  tv={tv_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.0f}s)")
    
    # Save best patch
    if best_patch is None:
        best_patch = patch.detach().cpu().clone()
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Best embedding L2: {best_emb_l2:.2f}")
    print(f"  Time: {time.time()-t_start:.1f}s")
    
    # Save print-ready patch (high resolution)
    patch_np = best_patch[0].permute(1, 2, 0).numpy()  # (H, W, 3) in [0,1]
    patch_uint8 = (patch_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save at optimization resolution
    patch_path = os.path.join(OUTPUT_DIR, "optimized_patch.png")
    Image.fromarray(patch_uint8, mode="RGB").save(patch_path)
    print(f"  Patch saved: {patch_path}")
    
    # Save high-res print version (3000x3000 for 10in @ 300dpi)
    patch_pil = Image.fromarray(patch_uint8, mode="RGB")
    patch_hr = patch_pil.resize((3000, 3000), Image.LANCZOS)
    patch_hr_path = os.path.join(OUTPUT_DIR, "optimized_patch_3000px.png")
    patch_hr.save(patch_hr_path)
    print(f"  Print-ready: {patch_hr_path}")
    
    # Save history
    hist_path = os.path.join(OUTPUT_DIR, "optimization_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {hist_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = [h["epoch"] for h in history]
    
    axes[0, 0].plot(epochs, [h["emb_l2"] for h in history], color="steelblue")
    axes[0, 0].set_title("Embedding L2 (higher = better)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="L2=1.0")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, [h["cos"] for h in history], color="orange")
    axes[0, 1].set_title("Cosine Similarity (lower = more corrupted)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].axhline(y=0.99, color="r", linestyle="--", alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, [h["supp_penalty"] for h in history], color="red")
    axes[1, 0].set_title("Suppression Penalty (lower = person still detected)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, [h["loss"] for h in history], color="green")
    axes[1, 1].set_title("Total Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Curves: {plot_path}")
    
    # Final evaluation: compare clean vs patched
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    with torch.no_grad():
        # Clean baseline
        clean_dets = get_dets_from_output(clean_output, conf=0.1)
        clean_persons = [d for d in clean_dets if d["cx"] > 0]  # all persons
        
        # Patched
        patch_416_final = F.interpolate(best_patch.to(DEVICE), 
                                        size=(PATCH_SIZE_416, PATCH_SIZE_416),
                                        mode='bilinear', align_corners=False)
        composited_final = composite_patch_diff(base_tensor, patch_416_final, 
                                                 PATCH_CX, PATCH_CY, PATCH_SIZE_416)
        caps_final, output_final = forward_capture_diff(model, composited_final)
        patched_dets = get_dets_from_output(output_final, conf=0.1)
        
        # Per-layer L2
        print(f"\n  Per-layer embedding corruption:")
        for layer_name, layer_idx in DETECTION_LAYERS.items():
            feat_final = caps_final[layer_idx]
            fH, fW = feat_final.shape[2], feat_final.shape[3]
            fx = int(wearer["cx"] / IMG_SIZE * fW)
            fy = int(wearer["cy"] / IMG_SIZE * fH)
            fx = max(0, min(fW - 1, fx))
            fy = max(0, min(fH - 1, fy))
            vec_final = feat_final[0, :, fy, fx].cpu().numpy()
            l2 = float(np.linalg.norm(vec_final - clean_embs[layer_name]))
            cos = float(np.dot(vec_final, clean_embs[layer_name]) / 
                       (np.linalg.norm(vec_final) * np.linalg.norm(clean_embs[layer_name]) + 1e-8))
            print(f"    {layer_name}: L2={l2:.2f}  cos={cos:.4f}")
        
        # Detection comparison
        clean_wearer = [d for d in clean_dets if 
            math.sqrt((d["cx"]-wearer["cx"])**2 + (d["cy"]-wearer["cy"])**2) < WEARER_THRESHOLD]
        patched_wearer = [d for d in patched_dets if 
            math.sqrt((d["cx"]-wearer["cx"])**2 + (d["cy"]-wearer["cy"])**2) < WEARER_THRESHOLD]
        
        print(f"\n  Detections:")
        print(f"    Clean:   {len(clean_dets)} persons, wearer conf={clean_wearer[0]['conf']:.3f}" if clean_wearer else f"    Clean:   {len(clean_dets)} persons, no wearer")
        print(f"    Patched: {len(patched_dets)} persons, wearer conf={patched_wearer[0]['conf']:.3f}" if patched_wearer else f"    Patched: {len(patched_dets)} persons, WEARER SUPPRESSED")
        
        # Save composite preview
        comp_np = composited_final[0].permute(1, 2, 0).cpu().numpy()
        comp_path = os.path.join(OUTPUT_DIR, "composite_preview.png")
        Image.fromarray((comp_np * 255).clip(0, 255).astype(np.uint8)).save(comp_path)
        print(f"\n  Composite preview: {comp_path}")
    
    # Save evaluation results
    eval_results = {
        "best_emb_l2": best_emb_l2,
        "patch_resolution": PATCH_RES,
        "patch_size_416": PATCH_SIZE_416,
        "num_epochs": NUM_EPOCHS,
        "eot_batch": BATCH_EOT,
        "loss_weights": {"emb": W_EMB, "supp": W_SUPP, "cos": W_COS, "tv": W_TV},
        "final_clean_persons": len(clean_dets),
        "final_patched_persons": len(patched_dets),
        "wearer_suppressed": len(patched_wearer) == 0,
    }
    eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Eval results: {eval_path}")
    
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_optimization()
