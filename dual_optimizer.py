"""
Dual-mode TPS-aware adversarial patch optimizer.

Produces TWO patches with different objectives:

1. SUPPRESS patch: Maximize human disappearance. Loss rewards the wearer
   not being detected. No embedding corruption needed — if there's no box,
   there's nothing to poison. Uses objectness loss directly from YOLO output.

2. POISON patch: Keep person detected but corrupt embeddings. Maximizes
   L2 shift + cross-wearer alignment + centroid attraction. This is the
   cloud poisoning payload — person boxes enter the pipeline with shifted
   feature vectors.

Both patches use TPS fabric deformation during EOT for clothing robustness.
Both have train/test split with overfitting tracking.
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim"
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
MAX_PERSONS = 200
TRAIN_SPLIT = 0.8
VAL_INTERVAL = 25
VAL_PERSONS = 10

# TPS params
TPS_GRID = 5
TPS_STRENGTH = 0.08
TPS_PROB = 0.5

# Suppress mode loss weights
SUPP_W_OBJ = 3.0       # objectness suppression (main driver)
SUPP_W_TV = 0.01        # printability
SUPP_W_CONF = 2.0       # confidence reduction for survivors

# Poison mode loss weights
POISON_W_EMB = 1.0
POISON_W_ALIGN = 2.0
POISON_W_CENTROID = 1.5
POISON_W_SUPP = 0.5     # penalize suppression (want them detected)
POISON_W_TV = 0.01

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

def get_person_dets(output, conf=0.25):
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

def get_all_dets(output, conf=0.25):
    dets = []
    if output is None:
        return dets
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            dets.append({"cls": int(row[5]), "conf": float(row[4]),
                        "cx": float(row[0]), "cy": float(row[1]),
                        "w": float(row[2]), "h": float(row[3])})
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
# EOT: Rigid + TPS
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

def tps_kernel(r):
    r_safe = r.clamp(min=1e-6)
    return r_safe ** 2 * torch.log(r_safe)

def generate_tps_warp(patch, grid_k=TPS_GRID, strength=TPS_STRENGTH):
    _, _, H, W = patch.shape
    device = patch.device
    dtype = patch.dtype
    cp_coords = torch.linspace(-1.0, 1.0, grid_k, device=device, dtype=dtype)
    cp_y, cp_x = torch.meshgrid(cp_coords, cp_coords, indexing="ij")
    cp_x = cp_x.reshape(-1)
    cp_y = cp_y.reshape(-1)
    K2 = grid_k * grid_k
    raw_dx = torch.randn(K2, device=device, dtype=dtype)
    raw_dy = torch.randn(K2, device=device, dtype=dtype)
    cp_dist = torch.sqrt((cp_x.unsqueeze(0) - cp_x.unsqueeze(1))**2 +
                         (cp_y.unsqueeze(0) - cp_y.unsqueeze(1))**2 + 1e-8)
    sigma_smooth = 2.0 / grid_k
    smooth_kernel = torch.exp(-cp_dist**2 / (2 * sigma_smooth**2))
    smooth_kernel = smooth_kernel / smooth_kernel.sum(dim=1, keepdim=True)
    dx = (smooth_kernel @ raw_dx) * strength * 2.0
    dy = (smooth_kernel @ raw_dy) * strength * 2.0
    src = torch.stack([cp_x, cp_y], dim=1)
    tgt = src + torch.stack([dx, dy], dim=1)
    src_dist = torch.sqrt((src.unsqueeze(0) - src.unsqueeze(1))**2).sum(dim=2)
    K_mat = tps_kernel(src_dist)
    P = torch.cat([torch.ones(K2, 1, device=device, dtype=dtype), src], dim=1)
    L_top = torch.cat([K_mat, P], dim=1)
    L_bot = torch.cat([P.t(), torch.zeros(3, 3, device=device, dtype=dtype)], dim=1)
    L_full = torch.cat([L_top, L_bot], dim=0)
    rhs = torch.cat([tgt, torch.zeros(3, 2, device=device, dtype=dtype)], dim=0)
    weights = torch.linalg.lstsq(L_full, rhs).solution
    w_tps = weights[:K2]
    a_aff = weights[K2:]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    grid_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    dist_gc = torch.sqrt((grid_flat.unsqueeze(1) - src.unsqueeze(0))**2).sum(dim=2)
    phi_gc = tps_kernel(dist_gc)
    affine_part = torch.cat([torch.ones(H*W, 1, device=device, dtype=dtype), grid_flat], dim=1)
    warp = phi_gc @ w_tps + affine_part @ a_aff
    sample_grid = warp.reshape(H, W, 2).unsqueeze(0)
    sample_grid = torch.clamp(sample_grid, -1.5, 1.5)
    warped = F.grid_sample(patch, sample_grid, align_corners=False, padding_mode="reflection")
    return warped

def eot_transform_with_tps(patch, ps):
    transformed = eot_rigid(patch, ps)
    if torch.rand(1).item() < TPS_PROB:
        transformed = generate_tps_warp(transformed)
    return transformed

def tv_loss(patch):
    tv_h = torch.mean(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
    return tv_h + tv_w

def extract_emb_at(caps, layer_idx, spatial_x, spatial_y):
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW-1, int(spatial_x / IMG_SIZE * fW)))
    fy = max(0, min(fH-1, int(spatial_y / IMG_SIZE * fH)))
    return feat[0, :, fy, fx]

# ============================================================
# Pre-compute clean embeddings
# ============================================================

def precompute_clean_embeddings(model):
    print("Pre-computing clean embeddings for COCO persons...")
    centroids = {}
    for layer_name in DETECTION_LAYERS:
        cp = None
        for suffix in ["52x52", "26x26", "13x13"]:
            cp2 = os.path.join(CENTROID_DIR, f"centroid_{layer_name}_{suffix}.npy")
            if os.path.exists(cp2):
                cp = cp2
                break
        if cp:
            centroids[layer_name] = torch.from_numpy(np.load(cp)).to(DEVICE)
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
                "cx": d["cx"], "cy": d["cy"],
                "conf": d["conf"],
                "clean_embs": clean_embs,
            })
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1} images scanned, {len(persons)} persons collected")

    for layer_name in DETECTION_LAYERS:
        if centroids[layer_name] is None:
            embs = torch.stack([p["clean_embs"][layer_name] for p in persons])
            centroids[layer_name] = embs.mean(dim=0)

    n_train = int(len(persons) * TRAIN_SPLIT)
    train_persons = persons[:n_train]
    val_persons = persons[n_train:]
    print(f"  Total: {len(persons)}  (train={len(train_persons)}, val={len(val_persons)})")
    return train_persons, val_persons, centroids

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_patch(model, patch_tensor, persons, centroids, mode="poison"):
    """
    Evaluate patch on persons. Returns comprehensive stats.
    mode='suppress': focus on suppression rate + confidence drop
    mode='poison': focus on L2 shift + alignment + centroid cos
    """
    patch_416 = F.interpolate(patch_tensor, size=(PATCH_SIZE_416, PATCH_SIZE_416),
                              mode='bilinear', align_corners=False)

    n_suppressed = 0
    n_detected = 0
    confs_clean = []
    confs_patched = []
    l2_shifts = {ln: [] for ln in DETECTION_LAYERS}
    cos_centroids = {ln: [] for ln in DETECTION_LAYERS}
    all_deltas = {ln: [] for ln in DETECTION_LAYERS}

    for person in persons:
        base = person["image_tensor"]
        cx, cy = person["cx"], person["cy"]
        confs_clean.append(person["conf"])

        composited = composite_patch_diff(base, patch_416, cx, cy, PATCH_SIZE_416)
        caps, output = forward_capture_diff(model, composited)

        dets = get_person_dets(output, conf=0.25)
        wearer = [d for d in dets if math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60]

        if wearer:
            n_detected += 1
            confs_patched.append(wearer[0]["conf"])
        else:
            n_suppressed += 1
            confs_patched.append(0.0)

        for ln, li in DETECTION_LAYERS.items():
            pvec = extract_emb_at(caps, li, cx, cy)
            cvec = person["clean_embs"][ln]
            delta = pvec - cvec
            l2_shifts[ln].append(torch.norm(delta).item())
            all_deltas[ln].append(delta)
            if centroids[ln] is not None:
                to_c = centroids[ln] - cvec
                cos_centroids[ln].append(
                    F.cosine_similarity(delta.unsqueeze(0), to_c.unsqueeze(0)).squeeze().item()
                )

    n_total = len(persons)
    supp_rate = n_suppressed / n_total
    mean_conf_clean = np.mean(confs_clean)
    mean_conf_patched = np.mean(confs_patched)
    conf_drop = mean_conf_clean - mean_conf_patched

    mean_l2 = np.mean([np.mean(l2_shifts[ln]) for ln in DETECTION_LAYERS])
    mean_cos_cent = np.mean([np.mean(cos_centroids[ln]) for ln in DETECTION_LAYERS if cos_centroids[ln]])

    # Alignment
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

    return {
        "n_total": n_total, "n_detected": n_detected, "n_suppressed": n_suppressed,
        "suppression_rate": supp_rate,
        "mean_conf_clean": mean_conf_clean, "mean_conf_patched": mean_conf_patched,
        "conf_drop": conf_drop,
        "mean_l2": mean_l2, "mean_align": mean_align, "mean_cos_centroid": mean_cos_cent,
        "l2_per_layer": {ln: np.mean(l2_shifts[ln]) for ln in DETECTION_LAYERS},
    }

# ============================================================
# SUPPRESS MODE OPTIMIZATION
# ============================================================

def extract_prenms_objectness(caps, layer_idx, cx, cy):
    """
    Extract pre-NMS objectness scores at the wearer's spatial location
    from a YOLOv3 detection head feature map.

    YOLOv3 detection layer output: 255 channels = 3 anchors x 85 values
    Per anchor: [x, y, w, h, objectness, 80 class scores]
    Objectness channels: 4, 9, 14 (one per anchor)

    Returns sigmoid(objectness) averaged across 3 anchors — smooth gradient
    signal that doesn't vanish when post-NMS detection disappears.
    """
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW - 1, int(cx / IMG_SIZE * fW)))
    fy = max(0, min(fH - 1, int(cy / IMG_SIZE * fH)))

    # Objectness channels: 4, 9, 14 (anchor 0, 1, 2)
    obj_anchors = []
    for anchor in range(3):
        ch = anchor * 85 + 4  # objectness channel for this anchor
        raw = feat[0, ch, fy, fx]
        obj_anchors.append(torch.sigmoid(raw))

    # Also grab person class score (channel 5+0=5, 10+0=10, 15+0=15 for person class 0)
    # Minimizing person-specific objectness * class_score is stronger signal
    cls_anchors = []
    for anchor in range(3):
        ch_obj = anchor * 85 + 4
        ch_cls = anchor * 85 + 5  # class 0 = person
        obj = torch.sigmoid(feat[0, ch_obj, fy, fx])
        cls = torch.sigmoid(feat[0, ch_cls, fy, fx])
        cls_anchors.append(obj * cls)

    # Mean across anchors
    mean_obj = torch.stack(obj_anchors).mean()
    mean_cls_obj = torch.stack(cls_anchors).mean()

    return mean_obj, mean_cls_obj

def run_suppress_optimization(model, train_persons, val_persons, centroids):
    print(f"\n{'#'*70}")
    print("# SUPPRESS MODE OPTIMIZATION")
    print(f"{'#'*70}")
    print(f"Objective: Kill the human detection box")
    print(f"Loss: minimize pre-NMS objectness * person_class at wearer location")
    print(f"  + person_class_score (push class confidence down)")
    print(f"  + TV loss for printability")
    print(f"Weights: obj={SUPP_W_OBJ}, conf={SUPP_W_CONF}, tv={SUPP_W_TV}")
    print()

    # Warm start from TPS patch
    warm = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim_tps\tps_aligned_patch.png"
    if os.path.exists(warm):
        patch_pil = Image.open(warm).convert("RGB")
        patch_arr = np.array(patch_pil, dtype=np.float32) / 255.0
        patch = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        patch = F.interpolate(patch, size=(PATCH_RES, PATCH_RES), mode='bilinear', align_corners=False)
        print(f"Warm starting from TPS patch")
    else:
        patch = torch.rand(1, 3, PATCH_RES, PATCH_RES, device=DEVICE) * 0.4 + 0.3
        print("Random init")

    patch.requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    train_history = []
    val_history = []
    best_val_supp = -1.0
    best_patch = None
    t_start = time.time()
    N_train = len(train_persons)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        batch_indices = random.sample(range(N_train), min(BATCH_PERSONS, N_train))

        total_obj_loss = torch.zeros(1, device=DEVICE)
        total_cls_loss = torch.zeros(1, device=DEVICE)
        n_forward = 0

        for pidx in batch_indices:
            person = train_persons[pidx]
            base = person["image_tensor"]
            cx, cy = person["cx"], person["cy"]

            for eot_idx in range(EOT_PER_PERSON):
                patch_eot = eot_transform_with_tps(patch, PATCH_RES)
                patch_416 = F.interpolate(patch_eot, size=(PATCH_SIZE_416, PATCH_SIZE_416),
                                          mode='bilinear', align_corners=False)
                composited = composite_patch_diff(base, patch_416, cx, cy, PATCH_SIZE_416)
                caps, output = forward_capture_diff(model, composited)

                # Pre-NMS objectness loss at all 3 detection layers
                # This gives smooth gradient even when post-NMS detection vanishes
                for layer_name, layer_idx in DETECTION_LAYERS.items():
                    mean_obj, mean_cls_obj = extract_prenms_objectness(caps, layer_idx, cx, cy)
                    # Weight deeper layers more — they drive final detection
                    layer_w = {"L81_52x52": 0.5, "L93_26x26": 1.0, "L105_13x13": 1.5}
                    w = layer_w.get(layer_name, 1.0)
                    total_obj_loss = total_obj_loss + w * mean_cls_obj
                    n_forward += 1

        # Mean objectness * class score across all forward passes and layers
        if n_forward > 0:
            mean_obj = total_obj_loss / n_forward
        else:
            mean_obj = total_obj_loss

        tv = tv_loss(patch)

        # Loss: minimize pre-NMS person objectness * class score + TV
        loss = SUPP_W_OBJ * mean_obj + SUPP_W_TV * tv

        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        obj_val = mean_obj.item()
        tv_val = tv.item()
        loss_val = loss.item()

        train_history.append({
            "epoch": epoch, "loss": loss_val, "obj": obj_val,
            "tv": tv_val,
            "lr": scheduler.get_last_lr()[0],
        })

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            elapsed = time.time() - t_start
            print(f"  E{epoch:4d}  loss={loss_val:8.4f}  obj={obj_val:.6f}  "
                  f"tv={tv_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.0f}s)")

        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            val_sample = random.sample(val_persons, min(VAL_PERSONS, len(val_persons)))
            vres = evaluate_patch(model, patch.detach(), val_sample, centroids, mode="suppress")
            val_history.append({
                "epoch": epoch,
                "val_suppression_rate": vres["suppression_rate"],
                "val_conf_drop": vres["conf_drop"],
                "val_mean_conf_patched": vres["mean_conf_patched"],
            })
            print(f"  VAL E{epoch:4d}  supp={vres['suppression_rate']:.1%}  "
                  f"conf_drop={vres['conf_drop']:.4f}  "
                  f"patched_conf={vres['mean_conf_patched']:.4f}")
            if vres["suppression_rate"] > best_val_supp:
                best_val_supp = vres["suppression_rate"]
                best_patch = patch.detach().cpu().clone()

    if best_patch is None:
        best_patch = patch.detach().cpu().clone()

    # Save
    supp_dir = os.path.join(OUTPUT_DIR, "suppress")
    os.makedirs(supp_dir, exist_ok=True)

    patch_np = best_patch[0].permute(1, 2, 0).numpy()
    patch_uint8 = (patch_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(patch_uint8, "RGB").save(os.path.join(supp_dir, "suppress_patch.png"))
    Image.fromarray(patch_uint8, "RGB").resize((3000, 3000), Image.LANCZOS).save(
        os.path.join(supp_dir, "suppress_patch_3000px.png"))

    with open(os.path.join(supp_dir, "train_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)
    with open(os.path.join(supp_dir, "val_history.json"), "w") as f:
        json.dump(val_history, f, indent=2)

    # Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ep = [h["epoch"] for h in train_history]
    val_ep = [h["epoch"] for h in val_history]
    axes[0].plot(ep, [h["obj"] for h in train_history], color="red")
    axes[0].plot(val_ep, [h["val_suppression_rate"] for h in val_history], "o-", color="green", markersize=4)
    axes[0].set_title("Objectness Loss / Suppression Rate")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, [h["loss"] for h in train_history], color="steelblue")
    axes[1].set_title("Total Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(val_ep, [h["val_suppression_rate"] for h in val_history], "o-", color="green", markersize=4)
    axes[2].set_title("Val Suppression Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(supp_dir, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\n  SUPPRESS MODE COMPLETE")
    print(f"  Best val suppression: {best_val_supp:.1%}")
    print(f"  Time: {time.time()-t_start:.1f}s")
    print(f"  Patch: {os.path.join(supp_dir, 'suppress_patch.png')}")
    print(f"  Print: {os.path.join(supp_dir, 'suppress_patch_3000px.png')}")

    return best_patch

# ============================================================
# POISON MODE OPTIMIZATION
# ============================================================

def run_poison_optimization(model, train_persons, val_persons, centroids):
    print(f"\n{'#'*70}")
    print("# POISON MODE OPTIMIZATION")
    print(f"{'#'*70}")
    print(f"Objective: Keep person detected, corrupt embeddings maximally")
    print(f"Loss: -w_emb*l2 -w_align*cos -w_centroid*cos_cent +w_supp*supp +w_tv*tv")
    print(f"Weights: emb={POISON_W_EMB}, align={POISON_W_ALIGN}, centroid={POISON_W_CENTROID}, "
          f"supp={POISON_W_SUPP}, tv={POISON_W_TV}")
    print()

    # Warm start from TPS patch
    warm = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim_tps\tps_aligned_patch.png"
    if os.path.exists(warm):
        patch_pil = Image.open(warm).convert("RGB")
        patch_arr = np.array(patch_pil, dtype=np.float32) / 255.0
        patch = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        patch = F.interpolate(patch, size=(PATCH_RES, PATCH_RES), mode='bilinear', align_corners=False)
        print(f"Warm starting from TPS patch")
    else:
        patch = torch.rand(1, 3, PATCH_RES, PATCH_RES, device=DEVICE) * 0.4 + 0.3

    patch.requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    train_history = []
    val_history = []
    best_val_score = -1e9
    best_patch = None
    t_start = time.time()
    N_train = len(train_persons)

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
                    all_l2[layer_name].append(torch.norm(delta))
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

        mean_l2 = torch.zeros(1, device=DEVICE)
        for ln in DETECTION_LAYERS:
            if all_l2[ln]:
                mean_l2 = mean_l2 + torch.stack(all_l2[ln]).mean()
        mean_l2 = mean_l2 / len(DETECTION_LAYERS)

        mean_align = torch.zeros(1, device=DEVICE)
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
                mean_align = mean_align + pc.mean()
        mean_align = mean_align / len(DETECTION_LAYERS)

        mean_cos_centroid = torch.zeros(1, device=DEVICE)
        for ln in DETECTION_LAYERS:
            if all_cos_centroid[ln]:
                mean_cos_centroid = mean_cos_centroid + torch.stack(all_cos_centroid[ln]).mean()
        mean_cos_centroid = mean_cos_centroid / len(DETECTION_LAYERS)

        avg_supp = total_supp / max(n_forward, 1)
        tv = tv_loss(patch)

        loss = -POISON_W_EMB * mean_l2 - POISON_W_ALIGN * mean_align \
               - POISON_W_CENTROID * mean_cos_centroid \
               + POISON_W_SUPP * avg_supp + POISON_W_TV * tv

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

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            elapsed = time.time() - t_start
            print(f"  E{epoch:4d}  loss={loss_val:8.3f}  l2={l2_val:6.2f}  "
                  f"align={align_val:6.4f}  centroid={cent_val:6.4f}  "
                  f"supp={supp_val:5.2f}  tv={tv_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.0f}s)")

        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            val_sample = random.sample(val_persons, min(VAL_PERSONS, len(val_persons)))
            vres = evaluate_patch(model, patch.detach(), val_sample, centroids, mode="poison")
            v_score = vres["mean_l2"] * (1 + vres["mean_align"]) * (1 + max(0, vres["mean_cos_centroid"])) \
                      if vres["suppression_rate"] < 0.5 else -1e9
            val_history.append({
                "epoch": epoch,
                "val_l2": vres["mean_l2"], "val_align": vres["mean_align"],
                "val_centroid": vres["mean_cos_centroid"],
                "val_suppression": vres["suppression_rate"],
                "val_score": v_score,
            })
            print(f"  VAL E{epoch:4d}  l2={vres['mean_l2']:.2f}  align={vres['mean_align']:.4f}  "
                  f"centroid={vres['mean_cos_centroid']:.4f}  supp={vres['suppression_rate']:.1%}  "
                  f"score={v_score:.2f}")
            if v_score > best_val_score:
                best_val_score = v_score
                best_patch = patch.detach().cpu().clone()

    if best_patch is None:
        best_patch = patch.detach().cpu().clone()

    # Save
    poison_dir = os.path.join(OUTPUT_DIR, "poison")
    os.makedirs(poison_dir, exist_ok=True)

    patch_np = best_patch[0].permute(1, 2, 0).numpy()
    patch_uint8 = (patch_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(patch_uint8, "RGB").save(os.path.join(poison_dir, "poison_patch.png"))
    Image.fromarray(patch_uint8, "RGB").resize((3000, 3000), Image.LANCZOS).save(
        os.path.join(poison_dir, "poison_patch_3000px.png"))

    with open(os.path.join(poison_dir, "train_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)
    with open(os.path.join(poison_dir, "val_history.json"), "w") as f:
        json.dump(val_history, f, indent=2)

    # Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = [h["epoch"] for h in train_history]
    val_ep = [h["epoch"] for h in val_history]
    axes[0,0].plot(ep, [h["emb_l2"] for h in train_history], color="steelblue", label="train")
    axes[0,0].plot(val_ep, [h["val_l2"] for h in val_history], "o-", color="red", markersize=4, label="val")
    axes[0,0].set_title("Embedding L2"); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(ep, [h["alignment"] for h in train_history], color="steelblue", label="train")
    axes[0,1].plot(val_ep, [h["val_align"] for h in val_history], "o-", color="red", markersize=4, label="val")
    axes[0,1].set_title("Alignment"); axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)
    axes[0,2].plot(ep, [h["centroid_cos"] for h in train_history], color="steelblue", label="train")
    axes[0,2].plot(val_ep, [h["val_centroid"] for h in val_history], "o-", color="red", markersize=4, label="val")
    axes[0,2].set_title("Centroid Cos"); axes[0,2].legend(fontsize=8); axes[0,2].grid(True, alpha=0.3)
    axes[1,0].plot(ep, [h["supp"] for h in train_history], color="orange")
    axes[1,0].set_title("Suppression Penalty"); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(ep, [h["tv"] for h in train_history], color="teal")
    axes[1,1].set_title("TV Loss"); axes[1,1].grid(True, alpha=0.3)
    axes[1,2].plot(val_ep, [h["val_suppression"] for h in val_history], "o-", color="red", markersize=4)
    axes[1,2].set_title("Val Suppression Rate"); axes[1,2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(poison_dir, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\n  POISON MODE COMPLETE")
    print(f"  Best val score: {best_val_score:.3f}")
    print(f"  Time: {time.time()-t_start:.1f}s")
    print(f"  Patch: {os.path.join(poison_dir, 'poison_patch.png')}")
    print(f"  Print: {os.path.join(poison_dir, 'poison_patch_3000px.png')}")

    return best_patch

# ============================================================
# SIDE-BY-SIDE EVALUATION
# ============================================================

@torch.no_grad()
def final_evaluation(model, suppress_patch, poison_patch, val_persons, centroids):
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE FINAL EVALUATION")
    print(f"{'='*70}")

    results = {}
    for name, ppatch in [("SUPPRESS", suppress_patch), ("POISON", poison_patch)]:
        vres = evaluate_patch(model, ppatch, val_persons, centroids, mode="poison")
        results[name] = vres
        print(f"\n  {name} PATCH:")
        print(f"    Suppression rate:   {vres['suppression_rate']:.1%}  ({vres['n_suppressed']}/{vres['n_total']})")
        print(f"    Mean conf (clean):  {vres['mean_conf_clean']:.4f}")
        print(f"    Mean conf (patched):{vres['mean_conf_patched']:.4f}")
        print(f"    Confidence drop:    {vres['conf_drop']:.4f} ({100*vres['conf_drop']/vres['mean_conf_clean']:.1f}%)")
        print(f"    L2 shift:           {vres['mean_l2']:.2f}")
        print(f"    Alignment cos:      {vres['mean_align']:.4f}")
        print(f"    Centroid cos:       {vres['mean_cos_centroid']:.4f}")
        for ln in DETECTION_LAYERS:
            print(f"      {ln}: L2={vres['l2_per_layer'][ln]:.2f}")

    # Comparison table
    print(f"\n  {'Metric':<25s} {'SUPPRESS':>12s} {'POISON':>12s}")
    print(f"  {'-'*50}")
    print(f"  {'Suppression rate':<25s} {results['SUPPRESS']['suppression_rate']:>11.1%} {results['POISON']['suppression_rate']:>11.1%}")
    print(f"  {'Conf drop':<25s} {results['SUPPRESS']['conf_drop']:>12.4f} {results['POISON']['conf_drop']:>12.4f}")
    print(f"  {'L2 shift':<25s} {results['SUPPRESS']['mean_l2']:>12.2f} {results['POISON']['mean_l2']:>12.2f}")
    print(f"  {'Alignment':<25s} {results['SUPPRESS']['mean_align']:>12.4f} {results['POISON']['mean_align']:>12.4f}")
    print(f"  {'Centroid cos':<25s} {results['SUPPRESS']['mean_cos_centroid']:>12.4f} {results['POISON']['mean_cos_centroid']:>12.4f}")

    # Save
    comparison = {
        "suppress": {k: v for k, v in results["SUPPRESS"].items() if not hasattr(v, 'item')},
        "poison": {k: v for k, v in results["POISON"].items() if not hasattr(v, 'item')},
    }
    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Comparison saved: {os.path.join(OUTPUT_DIR, 'comparison.json')}")

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Patch: {PATCH_RES}x{PATCH_RES} -> {PATCH_SIZE_416}px in 416 space (~10in @ 300DPI)")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}, Batch: {BATCH_PERSONS}x{EOT_PER_PERSON}")
    print(f"TPS: grid={TPS_GRID}x{TPS_GRID}, strength={TPS_STRENGTH}, prob={TPS_PROB}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    print("Loading YOLOv3...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_persons, val_persons, centroids = precompute_clean_embeddings(model)

    # Run both optimizations
    suppress_patch = run_suppress_optimization(model, train_persons, val_persons, centroids)
    poison_patch = run_poison_optimization(model, train_persons, val_persons, centroids)

    # Final side-by-side evaluation on full val set
    final_evaluation(model, suppress_patch, poison_patch, val_persons, centroids)

    print(f"\n{'='*70}")
    print("DUAL-MODE OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Suppress patch: {OUTPUT_DIR}/suppress/suppress_patch_3000px.png")
    print(f"  Poison patch:   {OUTPUT_DIR}/poison/poison_patch_3000px.png")
    print(f"  Comparison:     {OUTPUT_DIR}/comparison.json")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
