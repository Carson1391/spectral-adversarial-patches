"""
TPS-aware multi-wearer aligned adversarial patch optimizer.

Adds two major features over aligned_optimizer.py:
1. Differentiable 2D Thin Plate Spline (TPS) deformation to simulate
   clothing wrinkles, folds, and draping. The patch is warped by smooth
   random displacement fields during EOT, so the optimized patch is
   robust to fabric deformation when worn.

2. Train/test split with held-out validation persons. Every VAL_INTERVAL
   epochs, the patch is evaluated on unseen persons to track overfitting.
   Both train and val metrics are logged and plotted.

TPS deformation model:
- Grid of KxK control points placed on the patch
- Each control point gets a random (dx, dy) displacement drawn from a
  Gaussian process with spatial correlation (simulating smooth fabric folds)
- TPS interpolation computes dense displacement field from control points
- grid_sample applies the warp differentiably

Loss = -w_emb * mean_l2                          # maximize shift magnitude
       -w_align * mean_pairwise_cosine(deltas)    # align corruption directions
       -w_centroid * mean_cos_to_centroid         # pull toward centroid
       +w_supp * suppression_penalty              # keep person detected
       +w_tv * total_variation                    # printability
"""
import os, sys, json, math, time, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
COCO_DIR     = r"C:\Users\carso\Desktop\YODO\data\coco_person\images"
CENTROID_DIR = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\alignment_test"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim_tps"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416

DETECTION_LAYERS = {
    "L81_52x52": 81,
    "L93_26x26": 93,
    "L105_13x13": 105,
}

PATCH_SIZE_416 = 80
PATCH_RES = 300
NUM_EPOCHS = 500
LR = 0.03
BATCH_PERSONS = 4
EOT_PER_PERSON = 2
MAX_PERSONS = 200          # total persons to collect
TRAIN_SPLIT = 0.8          # 80% train, 20% held-out validation
VAL_INTERVAL = 25          # evaluate on val set every N epochs
VAL_PERSONS = 10           # persons to sample for val evaluation

# TPS deformation params
TPS_GRID = 5               # 5x5 control point grid
TPS_STRENGTH = 0.08        # max displacement as fraction of patch size
TPS_PROB = 0.5             # probability of applying TPS each EOT step

# Loss weights
W_EMB = 1.0
W_ALIGN = 2.0
W_CENTROID = 1.5
W_SUPP = 0.5
W_TV = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Model forward (differentiable, with feature capture)
# ============================================================

def forward_capture_diff(model, x):
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
            x = mo[0](x, IMG_SIZE)
        if md["type"] == "convolutional":
            caps[i] = x
        los.append(x)
    return caps, x

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

def get_person_dets(output, conf=0.3):
    dets = []
    if output is None:
        return dets
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf and int(row[5]) == 0:
            dets.append({"cx": float(row[0]), "cy": float(row[1]),
                        "w": float(row[2]), "h": float(row[3]),
                        "conf": float(row[4])})
    return dets

# ============================================================
# Differentiable patch compositing
# ============================================================

def composite_patch_diff(base_img, patch_rgb, cx, cy, ps):
    H, W = base_img.shape[2], base_img.shape[3]
    yy, xx = torch.meshgrid(
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        indexing="ij"
    )
    r = ps / 2.0
    dist = torch.sqrt((xx - r + 0.5)**2 + (yy - r + 0.5)**2)
    mask = torch.clamp(1.0 - (dist - r * 0.85) / (r * 0.15), 0.0, 1.0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    x0 = int(cx - ps // 2)
    y0 = int(cy - ps // 2)
    px0, py0 = max(0, x0), max(0, y0)
    px1 = min(W, x0 + ps)
    py1 = min(H, y0 + ps)
    sx0, sy0 = px0 - x0, py0 - y0
    sx1 = sx0 + (px1 - px0)
    sy1 = sy0 + (py1 - py0)
    full_patch = torch.zeros_like(base_img)
    full_mask = torch.zeros(1, 1, H, W, device=base_img.device, dtype=base_img.dtype)
    full_patch[:, :, py0:py1, px0:px1] = patch_rgb[:, :, sy0:sy1, sx0:sx1]
    full_mask[:, :, py0:py1, px0:px1] = mask[:, :, sy0:sy1, sx0:sx1]
    composited = base_img + (full_patch - 0.5) * full_mask * 0.3
    composited = torch.clamp(composited, 0.0, 1.0)
    return composited

# ============================================================
# EOT: Rigid transforms (rotation, scale, blur)
# ============================================================

def eot_rigid(patch, ps):
    angle = (torch.rand(1, device=patch.device) - 0.5) * 30.0
    scale = 0.7 + torch.rand(1, device=patch.device) * 0.6
    tx = (torch.rand(1, device=patch.device) - 0.5) * 0.2 * ps
    ty = (torch.rand(1, device=patch.device) - 0.5) * 0.2 * ps
    theta = torch.zeros(1, 2, 3, device=patch.device, dtype=patch.dtype)
    cos_a = torch.cos(angle * math.pi / 180.0)
    sin_a = torch.sin(angle * math.pi / 180.0)
    theta[:, 0, 0] = cos_a / scale
    theta[:, 0, 1] = -sin_a / scale
    theta[:, 0, 2] = tx / ps * 2
    theta[:, 1, 0] = sin_a / scale
    theta[:, 1, 1] = cos_a / scale
    theta[:, 1, 2] = ty / ps * 2
    grid = F.affine_grid(theta, patch.shape, align_corners=False)
    transformed = F.grid_sample(patch, grid, align_corners=False, padding_mode="reflection")
    blur_radius = torch.randint(0, 3, (1,)).item()
    if blur_radius > 0:
        ksize = blur_radius * 2 + 1
        sigma = 0.5 + torch.rand(1).item() * 1.0
        k1d = torch.exp(-torch.arange(-ksize//2, ksize//2+1, device=patch.device, dtype=patch.dtype)**2 / (2 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        k2d = k2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        transformed = F.conv2d(transformed, k2d, padding=ksize//2, groups=3)
    return transformed

# ============================================================
# TPS: Differentiable 2D Thin Plate Spline deformation
# Simulates fabric wrinkling, folding, and draping
# ============================================================

def tps_kernel(r):
    # TPS radial basis: r^2 * log(r), with safe handling at r=0
    r_safe = r.clamp(min=1e-6)
    return r_safe ** 2 * torch.log(r_safe)

def generate_tps_warp(patch, grid_k=TPS_GRID, strength=TPS_STRENGTH):
    """
    Generate a differentiable TPS warp field for the patch.
    
    Places a grid_k x grid_k grid of control points, displaces them
    with spatially-correlated random noise (simulating fabric folds),
    and computes the TPS interpolation to get a dense displacement field.
    
    Returns a warped patch via grid_sample.
    
    patch: (1, 3, H, W) tensor
    """
    _, _, H, W = patch.shape
    device = patch.device
    dtype = patch.dtype
    
    # Control point grid in normalized [-1, 1] coordinates
    # grid_k points evenly spaced
    cp_coords = torch.linspace(-1.0, 1.0, grid_k, device=device, dtype=dtype)
    cp_y, cp_x = torch.meshgrid(cp_coords, cp_coords, indexing="ij")
    cp_x = cp_x.reshape(-1)  # (K^2,)
    cp_y = cp_y.reshape(-1)
    K2 = grid_k * grid_k
    
    # Generate spatially-correlated random displacements
    # Use a smooth Gaussian process: displace control points with
    # correlation that decays with distance (simulating fabric folds)
    # Random displacements in [-strength, strength] * 2 (normalized space)
    raw_dx = torch.randn(K2, device=device, dtype=dtype)
    raw_dy = torch.randn(K2, device=device, dtype=dtype)
    
    # Spatially smooth the displacements using the control point distances
    # This makes nearby control points move similarly (fabric-like folds)
    cp_dist = torch.sqrt((cp_x.unsqueeze(0) - cp_x.unsqueeze(1))**2 +
                         (cp_y.unsqueeze(0) - cp_y.unsqueeze(1))**2 + 1e-8)
    # Gaussian smoothing kernel with correlation length ~ 1/grid_k
    sigma_smooth = 2.0 / grid_k
    smooth_kernel = torch.exp(-cp_dist**2 / (2 * sigma_smooth**2))
    smooth_kernel = smooth_kernel / smooth_kernel.sum(dim=1, keepdim=True)
    
    dx = (smooth_kernel @ raw_dx) * strength * 2.0  # scale to normalized space
    dy = (smooth_kernel @ raw_dy) * strength * 2.0
    
    # Source control points: (K^2, 2) in [-1, 1]
    src = torch.stack([cp_x, cp_y], dim=1)  # (K^2, 2)
    # Target control points: source + displacement
    tgt = src + torch.stack([dx, dy], dim=1)  # (K^2, 2)
    
    # Build TPS interpolation
    # We need to compute the TPS weights that map src -> tgt
    # Then apply to a dense grid to get the warp field
    
    # Pairwise distances between source control points
    src_dist = torch.sqrt(
        (src.unsqueeze(0) - src.unsqueeze(1))**2
    ).sum(dim=2)  # (K^2, K^2)
    
    # TPS kernel matrix
    K_mat = tps_kernel(src_dist)  # (K^2, K^2)
    
    # Affine constraint matrix P: (K^2, 3)
    P = torch.cat([torch.ones(K2, 1, device=device, dtype=dtype), src], dim=1)  # (K^2, 3)
    
    # Full TPS system matrix L: (K^2+3, K^2+3)
    L_top = torch.cat([K_mat, P], dim=1)  # (K^2, K^2+3)
    L_bot = torch.cat([P.t(), torch.zeros(3, 3, device=device, dtype=dtype)], dim=1)  # (3, K^2+3)
    L_full = torch.cat([L_top, L_bot], dim=0)  # (K^2+3, K^2+3)
    
    # Right-hand side: target displacements (what we want to interpolate)
    # TPS solves for weights w and affine params a such that:
    #   f(x) = sum_i w_i * phi(|x - cp_i|) + a0 + a1*x + a2*y
    # We want f(cp_i) = tgt_i, so RHS = tgt with 3 zeros for affine
    rhs = torch.cat([tgt, torch.zeros(3, 2, device=device, dtype=dtype)], dim=0)  # (K^2+3, 2)
    
    # Solve the linear system
    # L_full * [w; a] = rhs
    # Use least squares for numerical stability
    weights = torch.linalg.lstsq(L_full, rhs).solution  # (K^2+3, 2)
    w_tps = weights[:K2]  # (K^2, 2) TPS weights
    a_aff = weights[K2:]  # (3, 2) affine params
    
    # Generate dense sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    grid_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (H*W, 2)
    
    # Compute TPS warping at each grid point
    # dist from each grid point to each control point
    dist_gc = torch.sqrt(
        (grid_flat.unsqueeze(1) - src.unsqueeze(0))**2
    ).sum(dim=2)  # (H*W, K^2)
    
    phi_gc = tps_kernel(dist_gc)  # (H*W, K^2)
    
    # Warp: phi @ w_tps + [1, x, y] @ a_aff
    affine_part = torch.cat([
        torch.ones(H*W, 1, device=device, dtype=dtype),
        grid_flat
    ], dim=1)  # (H*W, 3)
    
    warp = phi_gc @ w_tps + affine_part @ a_aff  # (H*W, 2) - target positions
    
    # warp gives target positions for each source grid point
    # For grid_sample, we need the sampling grid in (x, y) order, normalized [-1, 1]
    # warp[:, 0] = target_x, warp[:, 1] = target_y
    sample_grid = warp.reshape(H, W, 2).unsqueeze(0)  # (1, H, W, 2)
    
    # Clamp to valid range
    sample_grid = torch.clamp(sample_grid, -1.5, 1.5)
    
    # Apply warp via grid_sample
    warped = F.grid_sample(patch, sample_grid, align_corners=False, padding_mode="reflection")
    
    return warped

def eot_transform_with_tps(patch, ps):
    """
    Full EOT: rigid transform + optional TPS fabric deformation + blur.
    """
    # First apply rigid EOT (rotation, scale, translation)
    transformed = eot_rigid(patch, ps)
    
    # Then optionally apply TPS fabric deformation
    if torch.rand(1).item() < TPS_PROB:
        transformed = generate_tps_warp(transformed, grid_k=TPS_GRID, strength=TPS_STRENGTH)
    
    return transformed

# ============================================================
# TV loss
# ============================================================

def tv_loss(patch):
    tv_h = torch.mean(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
    return tv_h + tv_w

# ============================================================
# Embedding extraction
# ============================================================

def extract_emb_at(caps, layer_idx, spatial_x, spatial_y):
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = int(spatial_x / IMG_SIZE * fW)
    fy = int(spatial_y / IMG_SIZE * fH)
    fx = max(0, min(fW - 1, fx))
    fy = max(0, min(fH - 1, fy))
    return feat[0, :, fy, fx]

# ============================================================
# Pre-compute clean embeddings with train/test split
# ============================================================

def precompute_clean_embeddings(model):
    print("Pre-computing clean embeddings for COCO persons...")
    centroids = {}
    for layer_name in DETECTION_LAYERS:
        cpath = os.path.join(CENTROID_DIR, f"centroid_{layer_name}.npy")
        if os.path.exists(cpath):
            centroids[layer_name] = torch.from_numpy(np.load(cpath)).to(DEVICE)
            print(f"  Loaded centroid for {layer_name}: norm={centroids[layer_name].norm():.2f}")
        else:
            centroids[layer_name] = None

    image_files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    random.shuffle(image_files)

    persons = []
    for idx, fname in enumerate(image_files):
        if len(persons) >= MAX_PERSONS:
            break
        path = os.path.join(COCO_DIR, fname)
        arr = load_image(path, IMG_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            caps, output = forward_capture_diff(model, tensor)
        dets = get_person_dets(output, conf=0.3)
        for d in dets:
            if len(persons) >= MAX_PERSONS:
                break
            clean_embs = {}
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                emb = extract_emb_at(caps, layer_idx, d["cx"], d["cy"])
                clean_embs[layer_name] = emb.detach().clone()
            persons.append({
                "image_tensor": tensor.detach(),
                "cx": d["cx"],
                "cy": d["cy"],
                "conf": d["conf"],
                "clean_embs": clean_embs,
            })
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1} images scanned, {len(persons)} persons collected")

    for layer_name in DETECTION_LAYERS:
        if centroids[layer_name] is None:
            embs = torch.stack([p["clean_embs"][layer_name] for p in persons])
            centroids[layer_name] = embs.mean(dim=0)
            print(f"  Computed centroid for {layer_name}: norm={centroids[layer_name].norm():.2f}")

    # Train/test split
    n_train = int(len(persons) * TRAIN_SPLIT)
    train_persons = persons[:n_train]
    val_persons = persons[n_train:]
    print(f"  Total persons: {len(persons)}  (train={len(train_persons)}, val={len(val_persons)})")
    return train_persons, val_persons, centroids

# ============================================================
# Evaluate on a set of persons (no grad, no TPS — clean eval)
# ============================================================

@torch.no_grad()
def evaluate_patch(model, patch_tensor, persons, centroids):
    """
    Evaluate patch on a set of persons without EOT/TPS.
    Returns mean L2, mean alignment, mean centroid cos, mean supp.
    """
    all_deltas = {ln: [] for ln in DETECTION_LAYERS}
    all_l2 = {ln: [] for ln in DETECTION_LAYERS}
    all_cos_centroid = {ln: [] for ln in DETECTION_LAYERS}
    total_supp = 0.0
    n_valid = 0

    patch_416 = F.interpolate(patch_tensor, size=(PATCH_SIZE_416, PATCH_SIZE_416),
                              mode='bilinear', align_corners=False)

    for person in persons:
        base = person["image_tensor"]
        cx, cy = person["cx"], person["cy"]
        composited = composite_patch_diff(base, patch_416, cx, cy, PATCH_SIZE_416)
        caps, output = forward_capture_diff(model, composited)

        for layer_name, layer_idx in DETECTION_LAYERS.items():
            patched_vec = extract_emb_at(caps, layer_idx, cx, cy)
            clean_vec = person["clean_embs"][layer_name]
            delta = patched_vec - clean_vec
            all_deltas[layer_name].append(delta)
            all_l2[layer_name].append(torch.norm(delta))
            to_centroid = centroids[layer_name] - clean_vec
            cos_c = F.cosine_similarity(delta.unsqueeze(0), to_centroid.unsqueeze(0)).squeeze()
            all_cos_centroid[layer_name].append(cos_c)

        dets = get_person_dets(output, conf=0.05)
        wearer_found = any(math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60 for d in dets)
        if not wearer_found:
            total_supp += person["conf"] * 5
        n_valid += 1

    # Mean L2
    mean_l2 = sum(torch.stack(all_l2[ln]).mean().item() for ln in DETECTION_LAYERS) / len(DETECTION_LAYERS)

    # Mean alignment
    mean_align = 0.0
    for ln in DETECTION_LAYERS:
        deltas = all_deltas[ln]
        if len(deltas) < 2:
            continue
        ds = torch.stack(deltas)
        norms = torch.norm(ds, dim=1, keepdim=True).clamp(min=1e-8)
        dn = ds / norms
        cm = torch.mm(dn, dn.t())
        M = cm.shape[0]
        triu = torch.triu_indices(M, M, offset=1)
        pc = cm[triu[0], triu[1]]
        if len(pc) > 0:
            mean_align += pc.mean().item()
    mean_align /= len(DETECTION_LAYERS)

    # Mean centroid cos
    mean_cc = sum(torch.stack(all_cos_centroid[ln]).mean().item() for ln in DETECTION_LAYERS) / len(DETECTION_LAYERS)

    avg_supp = total_supp / max(n_valid, 1)

    return mean_l2, mean_align, mean_cc, avg_supp

# ============================================================
# Main optimization
# ============================================================

def run_tps_optimization():
    print(f"Device: {DEVICE}")
    print(f"Patch resolution: {PATCH_RES}x{PATCH_RES}")
    print(f"Patch size in 416 space: {PATCH_SIZE_416}px (~10in @ 300DPI)")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}")
    print(f"Batch: {BATCH_PERSONS} persons x {EOT_PER_PERSON} EOT = {BATCH_PERSONS*EOT_PER_PERSON} forward passes/step")
    print(f"TPS: grid={TPS_GRID}x{TPS_GRID}, strength={TPS_STRENGTH}, prob={TPS_PROB}")
    print(f"Train/test split: {TRAIN_SPLIT} (val every {VAL_INTERVAL} epochs on {VAL_PERSONS} persons)")
    print(f"Loss weights: emb={W_EMB}, align={W_ALIGN}, centroid={W_CENTROID}, supp={W_SUPP}, tv={W_TV}")
    print()

    print("Loading YOLOv3...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Model loaded on {DEVICE}")

    train_persons, val_persons, centroids = precompute_clean_embeddings(model)
    N_train = len(train_persons)

    # Warm start from aligned patch (previous best)
    warm_start = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim\aligned_patch.png"
    if os.path.exists(warm_start):
        print(f"Warm starting from aligned patch: {warm_start}")
        patch_pil = Image.open(warm_start).convert("RGB")
        patch_arr = np.array(patch_pil, dtype=np.float32) / 255.0
        patch = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        patch = F.interpolate(patch, size=(PATCH_RES, PATCH_RES), mode='bilinear', align_corners=False)
    else:
        print("Initializing random patch...")
        patch = torch.rand(1, 3, PATCH_RES, PATCH_RES, device=DEVICE) * 0.4 + 0.3

    patch.requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    train_history = []
    val_history = []
    best_val_score = -1e9
    best_patch = None
    overfit_epochs = 0  # counter for consecutive epochs where val < train by margin

    print(f"\n{'='*70}")
    print("STARTING TPS-AWARE ALIGNED OPTIMIZATION")
    print(f"{'='*70}\n")

    t_start = time.time()

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        batch_indices = random.sample(range(N_train), min(BATCH_PERSONS, N_train))

        all_deltas = {ln: [] for ln in DETECTION_LAYERS}
        all_l2 = {ln: [] for ln in DETECTION_LAYERS}
        all_cos_centroid = {ln: [] for ln in DETECTION_LAYERS}
        total_supp = torch.zeros(1, device=DEVICE)
        n_forward = 0

        for pidx in batch_indices:
            person = train_persons[pidx]
            base = person["image_tensor"]
            cx, cy = person["cx"], person["cy"]

            for eot_idx in range(EOT_PER_PERSON):
                # EOT with TPS fabric deformation
                patch_eot = eot_transform_with_tps(patch, PATCH_RES)
                patch_416 = F.interpolate(patch_eot, size=(PATCH_SIZE_416, PATCH_SIZE_416),
                                          mode='bilinear', align_corners=False)
                composited = composite_patch_diff(base, patch_416, cx, cy, PATCH_SIZE_416)
                caps, output = forward_capture_diff(model, composited)

                for layer_name, layer_idx in DETECTION_LAYERS.items():
                    patched_vec = extract_emb_at(caps, layer_idx, cx, cy)
                    clean_vec = person["clean_embs"][layer_name]
                    delta = patched_vec - clean_vec
                    all_deltas[layer_name].append(delta)
                    l2 = torch.norm(delta)
                    all_l2[layer_name].append(l2)
                    to_centroid = centroids[layer_name] - clean_vec
                    cos_centroid = F.cosine_similarity(
                        delta.unsqueeze(0), to_centroid.unsqueeze(0)
                    ).squeeze()
                    all_cos_centroid[layer_name].append(cos_centroid)

                with torch.no_grad():
                    dets = get_person_dets(output, conf=0.05)
                    wearer_found = any(
                        math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60
                        for d in dets
                    )
                if not wearer_found:
                    supp_pen = torch.tensor(person["conf"] * 5, device=DEVICE, dtype=patch.dtype)
                else:
                    supp_pen = torch.zeros(1, device=DEVICE)
                total_supp = total_supp + supp_pen
                n_forward += 1

        # Mean L2 shift
        mean_l2 = torch.zeros(1, device=DEVICE)
        for ln in DETECTION_LAYERS:
            if all_l2[ln]:
                mean_l2 = mean_l2 + torch.stack(all_l2[ln]).mean()
        mean_l2 = mean_l2 / len(DETECTION_LAYERS)

        # Alignment loss
        mean_align = torch.zeros(1, device=DEVICE)
        for ln in DETECTION_LAYERS:
            deltas = all_deltas[ln]
            if len(deltas) < 2:
                continue
            delta_stack = torch.stack(deltas)
            norms = torch.norm(delta_stack, dim=1, keepdim=True).clamp(min=1e-8)
            delta_norm = delta_stack / norms
            cos_mat = torch.mm(delta_norm, delta_norm.t())
            M = cos_mat.shape[0]
            triu_idx = torch.triu_indices(M, M, offset=1)
            pairwise_cos = cos_mat[triu_idx[0], triu_idx[1]]
            if len(pairwise_cos) > 0:
                mean_align = mean_align + pairwise_cos.mean()
        mean_align = mean_align / len(DETECTION_LAYERS)

        # Centroid attraction
        mean_cos_centroid = torch.zeros(1, device=DEVICE)
        for ln in DETECTION_LAYERS:
            if all_cos_centroid[ln]:
                mean_cos_centroid = mean_cos_centroid + torch.stack(all_cos_centroid[ln]).mean()
        mean_cos_centroid = mean_cos_centroid / len(DETECTION_LAYERS)

        avg_supp = total_supp / max(n_forward, 1)
        tv = tv_loss(patch)

        loss = -W_EMB * mean_l2 - W_ALIGN * mean_align - W_CENTROID * mean_cos_centroid \
               + W_SUPP * avg_supp + W_TV * tv

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        l2_val = mean_l2.item()
        align_val = mean_align.item()
        cent_val = mean_cos_centroid.item()
        supp_val = avg_supp.item()
        tv_val = tv.item()
        loss_val = loss.item()

        train_history.append({
            "epoch": epoch, "loss": loss_val, "emb_l2": l2_val,
            "alignment": align_val, "centroid_cos": cent_val,
            "supp": supp_val, "tv": tv_val,
            "lr": scheduler.get_last_lr()[0],
        })

        # Track best patch by train score
        score = l2_val * (1 + align_val) * (1 + max(0, cent_val)) if supp_val < 0.5 else -1e9

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            elapsed = time.time() - t_start
            print(f"  E{epoch:4d}  loss={loss_val:8.3f}  l2={l2_val:6.2f}  "
                  f"align={align_val:6.4f}  centroid={cent_val:6.4f}  "
                  f"supp={supp_val:5.2f}  tv={tv_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.0f}s)")

        # Validation evaluation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            val_sample = random.sample(val_persons, min(VAL_PERSONS, len(val_persons)))
            v_l2, v_align, v_cent, v_supp = evaluate_patch(
                model, patch.detach(), val_sample, centroids
            )
            v_score = v_l2 * (1 + v_align) * (1 + max(0, v_cent)) if v_supp < 0.5 else -1e9

            val_history.append({
                "epoch": epoch, "val_l2": v_l2, "val_alignment": v_align,
                "val_centroid_cos": v_cent, "val_supp": v_supp, "val_score": v_score,
            })

            # Overfitting check: if train alignment >> val alignment
            overfit_gap = align_val - v_align
            overfit_status = ""
            if overfit_gap > 0.15 and v_l2 < l2_val * 0.7:
                overfit_epochs += 1
                overfit_status = "  *** OVERFIT WARNING ***"
            else:
                overfit_epochs = 0

            print(f"  VAL E{epoch:4d}  l2={v_l2:6.2f}  align={v_align:6.4f}  "
                  f"centroid={v_cent:6.4f}  supp={v_supp:5.2f}  "
                  f"score={v_score:7.2f}  (gap={overfit_gap:.4f}){overfit_status}")

            # Track best by validation score
            if v_score > best_val_score:
                best_val_score = v_score
                best_patch = patch.detach().cpu().clone()

    # ============================================================
    # Save results
    # ============================================================

    if best_patch is None:
        best_patch = patch.detach().cpu().clone()

    print(f"\n{'='*70}")
    print("TPS-AWARE ALIGNED OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Best val score: {best_val_score:.3f}")
    print(f"  Time: {time.time()-t_start:.1f}s")

    patch_np = best_patch[0].permute(1, 2, 0).numpy()
    patch_uint8 = (patch_np * 255).clip(0, 255).astype(np.uint8)

    patch_path = os.path.join(OUTPUT_DIR, "tps_aligned_patch.png")
    Image.fromarray(patch_uint8, mode="RGB").save(patch_path)
    print(f"  Patch saved: {patch_path}")

    patch_pil = Image.fromarray(patch_uint8, mode="RGB")
    patch_hr = patch_pil.resize((3000, 3000), Image.LANCZOS)
    patch_hr_path = os.path.join(OUTPUT_DIR, "tps_aligned_patch_3000px.png")
    patch_hr.save(patch_hr_path)
    print(f"  Print-ready: {patch_hr_path}")

    # Save history
    with open(os.path.join(OUTPUT_DIR, "train_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val_history.json"), "w") as f:
        json.dump(val_history, f, indent=2)
    print(f"  History saved (train + val)")

    # Training curves with train + val overlay
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = [h["epoch"] for h in train_history]
    val_ep = [h["epoch"] for h in val_history]

    axes[0, 0].plot(ep, [h["emb_l2"] for h in train_history], color="steelblue", label="train")
    axes[0, 0].plot(val_ep, [h["val_l2"] for h in val_history], "o-", color="red", label="val", markersize=4)
    axes[0, 0].set_title("Embedding L2 (higher = better)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, [h["alignment"] for h in train_history], color="steelblue", label="train")
    axes[0, 1].plot(val_ep, [h["val_alignment"] for h in val_history], "o-", color="red", label="val", markersize=4)
    axes[0, 1].set_title("Alignment Cosine (higher = more aligned)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="cos=0.5")
    axes[0, 1].axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="cos=0.8")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(ep, [h["centroid_cos"] for h in train_history], color="steelblue", label="train")
    axes[0, 2].plot(val_ep, [h["val_centroid_cos"] for h in val_history], "o-", color="red", label="val", markersize=4)
    axes[0, 2].set_title("Centroid Cosine (higher = toward centroid)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(ep, [h["supp"] for h in train_history], color="steelblue", label="train")
    axes[1, 0].plot(val_ep, [h["val_supp"] for h in val_history], "o-", color="red", label="val", markersize=4)
    axes[1, 0].set_title("Suppression Penalty")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, [h["tv"] for h in train_history], color="teal")
    axes[1, 1].set_title("TV Loss (printability)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(ep, [h["loss"] for h in train_history], color="green", label="train loss")
    axes[1, 2].set_title("Total Loss")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Curves: {os.path.join(OUTPUT_DIR, 'training_curves.png')}")

    # Overfitting summary
    if val_history:
        final_val = val_history[-1]
        final_train = train_history[-1]
        print(f"\n  OVERFITTING CHECK:")
        print(f"    Train align: {final_train['alignment']:.4f}  Val align: {final_val['val_alignment']:.4f}  "
              f"gap: {final_train['alignment'] - final_val['val_alignment']:.4f}")
        print(f"    Train l2:    {final_train['emb_l2']:.2f}  Val l2:    {final_val['val_l2']:.2f}  "
              f"ratio: {final_val['val_l2']/max(final_train['emb_l2'],1e-8):.3f}")
        if overfit_epochs > 0:
            print(f"    WARNING: {overfit_epochs} consecutive overfit detections at end of training")
        else:
            print(f"    No overfitting detected")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_tps_optimization()
