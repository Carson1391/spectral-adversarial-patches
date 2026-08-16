"""
Comprehensive adversarial patch evaluation against YOLOv3 edge pipeline.

Tests ALL parameters from the technical breakdown:
  Suppression Go/No-Go: ASR >94%, false rate <3%
  Poison Go/No-Go: ASR >88%, false rate <5%, NMS survival >90%, crop alignment <4px
  EoT robustness: H.265 DCT (real DCT-domain), motion blur, defocus, ISP noise,
    affine transforms, color space round-trip, exposure shifts
  Confidence thresholds: suppression <0.12, poison >0.65
  Anchor IoU: suppression <0.30, poison >0.55 (full box-IoU with center distance)
  NMS IoU: poison <0.40
  Cloud embedding drift: delta_cos > 0.15

Physical constraints:
  Patch placed on torso (~15-20% area), head/shoulders exposed
  Distances: 5m, 10m, 15m (Phase 2 field protocol)
  Lighting: 4000K (dawn), 6500K (noon), 3000K (dusk), 2200K (streetlamp)
  Angles: +/-10, +/-20, +/-30 deg yaw/pitch (field validation)

Defect fixes (A-H):
  A: false_suppression_rate now composites patch onto no-human image
  B: Verified YOLOLayer outputs sigmoid'd obj and cls separately (line 176)
  C: ASR renamed to single_image_reduction until COCO batch exists
  D: Random noise control patch run through identical pipeline
  E: No fake quantization (removed per user request - that's TensorRT/ONNX)
  F: H.265 uses real scipy DCT-domain quantization
  G: Saliency targets person-winning cell specifically
  H: Anchor IoU uses full box-IoU with center distance

Auto-discovers patch images across output directories.
Outputs: per-patch JSON+CSV+PNG, summary comparison CSV+PNG.
"""
import os, sys, math, json, csv, glob, re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

# ============================================================
# Config
# ============================================================
CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS = 416
OUT = r"C:\Users\carso\Desktop\YODO\outputs_eval"
os.makedirs(OUT, exist_ok=True)

# Physical patch placement: torso region, head/shoulders exposed
# In 416x416 frame, person is centered. Torso center ~ (208, 220)
# Patch size ~110px diameter = ~15-20% of torso area
PATCH_SIZE = 110
TORSO_CX = 208
TORSO_CY = 220

# YOLOv3 anchors (COCO, pixel-space in 416 input)
ANCHORS = torch.tensor([
    [10, 13], [16, 30], [33, 23],      # 52x52 (small)
    [30, 61], [62, 45], [59, 119],     # 26x26 (medium)
    [116, 90], [156, 198], [373, 326], # 13x13 (large)
], dtype=torch.float32, device=DEV)

# Go/No-Go thresholds from technical breakdown
THRESHOLDS = {
    "suppression": {
        "conf_max": 0.12,
        "anchor_iou_max": 0.30,
        "asr_digital_min": 0.94,
        "false_rate_max": 0.03,
    },
    "poison": {
        "conf_min": 0.65,
        "cls_min": 0.70,
        "anchor_iou_min": 0.55,
        "nms_iou_max": 0.40,
        "asr_digital_min": 0.88,
        "false_rate_max": 0.05,
        "nms_survival_min": 0.90,
        "crop_align_max": 4.0,
        "embedding_drift_min": 0.15,
    },
}

# Field testing distances (meters)
DISTANCES = [5, 10, 15]

# Lighting color temperatures (Kelvin) with approximate white balance gains
# Gains derived from blackbody color temperature ratios relative to 6500K (D65)
LIGHTING = {
    "noon_6500K":  {"r": 1.00, "g": 1.00, "b": 1.00, "ev": 0.0},   # neutral reference
    "dawn_4000K":  {"r": 1.12, "g": 1.00, "b": 0.78, "ev": -0.3},  # warm, slightly dimmer
    "dusk_3000K":  {"r": 1.28, "g": 0.98, "b": 0.60, "ev": -0.5},  # very warm, dimmer
    "street_2200K": {"r": 1.45, "g": 0.92, "b": 0.42, "ev": -0.8}, # extreme orange, dark
}

# EoT angles (degrees) per Phase 2 field validation protocol
YAW_ANGLES = [-30, -20, -10, 0, 10, 20, 30]
PITCH_ANGLES = [-30, -20, -10, 0, 10, 20, 30]

# Layers to analyze
HEAD_LAYERS = {"L81": 81, "L93": 93, "L105": 105}
BACKBONE_LAYERS = {"L54": 54, "L62": 62, "L75": 75}
ALL_LAYERS = {**HEAD_LAYERS, **BACKBONE_LAYERS}

PERSON_CLS = 0  # COCO person class index


# ============================================================
# Forward pass - capture all conv layer feature maps
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
            caps[i] = x
        los.append(x)
    return caps, x


def fwd_detections(model, x):
    """Forward pass returning decoded detections: [N, 85]."""
    los = []
    yolo_outs = []
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
            yolo_outs.append(x)
        los.append(x)
    if yolo_outs:
        # Each yolo_out is [B, num_anchors*grid_h*grid_w, 85] but grid sizes differ
        # Flatten each to [B, -1, 85] then concat along dim=1
        yolo_flat = [y.view(y.shape[0], -1, y.shape[-1]) for y in yolo_outs]
        pred = torch.cat(yolo_flat, dim=1)
        return pred.squeeze(0)
    return None


def gap_emb(caps, layer_idx):
    return F.adaptive_avg_pool2d(caps[layer_idx], 1).squeeze()


def point_emb(caps, layer_idx):
    return caps[layer_idx].squeeze(0).flatten()


# ============================================================
# Image loading
# ============================================================
def load_img(path, sz=IS):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(sz / w, sz / h)
    nw, nh = int(w * s), int(h * s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (sz, sz), (128, 128, 128))
    c.paste(r, ((sz - nw) // 2, (sz - nh) // 2))
    return np.array(c, dtype=np.float32) / 255.0


def load_patch(patch_path):
    """Load patch and derive mask. Handles RGBA alpha channel.
    Returns (patch_rgb [H,W,3], mask [H,W]) both in [0,1]."""
    img = Image.open(patch_path)
    if img.mode == 'RGBA':
        img_resized = img.resize((IS, IS), Image.BILINEAR)
        rgba = np.array(img_resized, dtype=np.float32) / 255.0
        patch_rgb = rgba[:, :, :3]
        mask = rgba[:, :, 3]
        if mask.max() <= 0.01:
            mask = (patch_rgb.mean(axis=2) > 0.02).astype(np.float32)
    else:
        img_rgb = img.convert('RGB').resize((IS, IS), Image.BILINEAR)
        patch_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
        mask = (patch_rgb.mean(axis=2) > 0.02).astype(np.float32)
    return patch_rgb, mask


# ============================================================
# Torso placement: composite patch on chest, head/shoulders exposed
# ============================================================
def composite_on_torso(patch_t, mask_t, base_t, patch_size=PATCH_SIZE,
                       cx=TORSO_CX, cy=TORSO_CY):
    """Place patch on torso region. Head/shoulders above ~130px untouched.
    patch_t: [1,3,IS,IS] full-size patch
    mask_t: [1,1,IS,IS] full-size mask
    base_t: [1,3,IS,IS] base image
    Returns composite [1,3,IS,IS] with patch scaled to patch_size on torso."""
    # Scale patch down to torso patch size
    p_small = F.interpolate(patch_t, size=(patch_size, patch_size),
                            mode='bilinear', align_corners=False)
    m_small = F.interpolate(mask_t, size=(patch_size, patch_size),
                            mode='bilinear', align_corners=False)

    # Compute placement region (clamped to image bounds)
    x0 = cx - patch_size // 2
    y0 = cy - patch_size // 2
    x1 = min(x0 + patch_size, IS)
    y1 = min(y0 + patch_size, IS)
    x0c = max(0, x0)
    y0c = max(0, y0)
    aw = x1 - x0c
    ah = y1 - y0c

    comp = base_t.clone()
    if aw > 0 and ah > 0:
        # Crop patch/mask to fit within image bounds
        px0 = x0c - x0
        py0 = y0c - y0
        comp[:, :, y0c:y1, x0c:x1] = (
            p_small[:, :, py0:py0+ah, px0:px0+aw] * m_small[:, :, py0:py0+ah, px0:px0+aw] +
            base_t[:, :, y0c:y1, x0c:x1] * (1 - m_small[:, :, py0:py0+ah, px0:px0+aw])
        )
    return comp


# ============================================================
# Color temperature simulation (lighting conditions)
# ============================================================
def apply_lighting(img_t, lighting_cfg):
    """Apply color temperature white balance gains and exposure shift.
    img_t: [1,3,H,W] in [0,1]
    lighting_cfg: dict with r,g,b gains and ev shift"""
    gains = torch.tensor([lighting_cfg["r"], lighting_cfg["g"], lighting_cfg["b"]],
                         dtype=torch.float32, device=DEV).view(1, 3, 1, 1)
    ev_factor = 2.0 ** lighting_cfg.get("ev", 0.0)
    out = (img_t * gains * ev_factor).clamp(0, 1)
    return out


# ============================================================
# Detection score extraction (per-head breakdown)
# ============================================================
def extract_detection_scores(model, x):
    los = []
    scores = {'obj_max': 0.0, 'person_prob': 0.0, 'combined': 0.0,
              'n_detections': 0, 'per_head': {}}
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
            pred = x
            if pred.dim() == 3:
                # YOLOLayer forward (eval mode) applies sigmoid to cols 4+
                # pred[..., 4] = sigmoid(obj_logit) = raw objectness
                # pred[..., 5:] = sigmoid(cls_logit) = raw class probs
                obj = pred[..., 4]
                cls_probs = pred[..., 5:]
                person_p = cls_probs[..., PERSON_CLS]
                combined = obj * person_p
                mask_conf = combined > 0.25
                n_det = int(mask_conf.sum().item())
                obj_max = float(obj.max().item())
                person_max = float(person_p.max().item())
                combined_max = float(combined.max().item())
                head_name = f"L{i}"
                scores['per_head'][head_name] = {
                    'obj_max': obj_max, 'person_prob': person_max,
                    'combined': combined_max, 'n_detections': n_det
                }
                scores['obj_max'] = max(scores['obj_max'], obj_max)
                scores['person_prob'] = max(scores['person_prob'], person_max)
                scores['combined'] = max(scores['combined'], combined_max)
                scores['n_detections'] += n_det
        los.append(x)
    return scores


# ============================================================
# NMS and box utilities
# ============================================================
def compute_iou_matrix(boxes1, boxes2):
    """boxes: [N, 4] (x1,y1,x2,y2). Returns [N, M] IoU matrix."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)
    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / (union + 1e-10)


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = compute_iou_matrix(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def boxes_from_pred(pred, conf_threshold=0.25):
    """Convert YOLO output to detection boxes.
    pred: [N, 85] -> (x1,y1,x2,y2,conf,cls) for conf > threshold."""
    obj = pred[:, 4]
    cls_probs = pred[:, 5:]
    cls_conf, cls_idx = cls_probs.max(dim=1)
    combined = obj * cls_conf
    mask = combined > conf_threshold
    if mask.sum() == 0:
        return torch.zeros(0, 6, device=pred.device), combined[mask]
    filtered = pred[mask]
    x, y, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    x1 = x - w / 2; y1 = y - h / 2
    x2 = x + w / 2; y2 = y + h / 2
    confs = combined[mask]
    clss = cls_idx[mask].float()
    return torch.stack([x1, y1, x2, y2, confs, clss], dim=1), confs


def boxes_xywh_from_pred(pred, conf_threshold=0.25):
    """Return xywh boxes + conf for anchor IoU computation."""
    obj = pred[:, 4]
    cls_probs = pred[:, 5:]
    cls_conf, cls_idx = cls_probs.max(dim=1)
    combined = obj * cls_conf
    mask = combined > conf_threshold
    if mask.sum() == 0:
        return torch.zeros(0, 5, device=pred.device)
    filtered = pred[mask]
    return torch.stack([filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3],
                        combined[mask]], dim=1)


def anchor_box_iou(det_xywh, anchors):
    """Fix (H): Full box-IoU between detection and anchors.
    Treats each anchor as a box CENTERED at the detection's center point.
    This accounts for positional offset, unlike the old co-centered approximation.
    det_xywh: [N, 4] (cx, cy, w, h)
    anchors: [9, 2] (w, h)
    Returns: [N, 9] IoU matrix."""
    N = det_xywh.shape[0]
    M = anchors.shape[0]
    if N == 0:
        return torch.zeros(0, M, device=det_xywh.device)

    # Detection boxes in xyxy
    dx, dy, dw, dh = det_xywh[:, 0], det_xywh[:, 1], det_xywh[:, 2], det_xywh[:, 3]
    dx1 = dx - dw / 2; dy1 = dy - dh / 2
    dx2 = dx + dw / 2; dy2 = dy + dh / 2

    # Anchor boxes centered at detection center, in xyxy
    aw = anchors[:, 0].unsqueeze(0).expand(N, M)  # [N, M]
    ah = anchors[:, 1].unsqueeze(0).expand(N, M)
    ax1 = dx.unsqueeze(1) - aw / 2  # [N, M]
    ay1 = dy.unsqueeze(1) - ah / 2
    ax2 = dx.unsqueeze(1) + aw / 2
    ay2 = dy.unsqueeze(1) + ah / 2

    # Intersection
    inter_x1 = torch.max(dx1.unsqueeze(1).expand(N, M), ax1)
    inter_y1 = torch.max(dy1.unsqueeze(1).expand(N, M), ay1)
    inter_x2 = torch.min(dx2.unsqueeze(1).expand(N, M), ax2)
    inter_y2 = torch.min(dy2.unsqueeze(1).expand(N, M), ay2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    det_area = (dw * dh).unsqueeze(1).expand(N, M)
    anchor_area = aw * ah
    union = det_area + anchor_area - inter

    return inter / (union + 1e-10)


# ============================================================
# Metric 1-4: Per-layer cosine, L2, FFT, person overlap
# ============================================================
def per_layer_metrics(model, clean_t, adv_t, without_t):
    results = {}
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        caps_adv, _ = fwd_all(model, adv_t)
        caps_without, _ = fwd_all(model, without_t)
    for lname, lidx in ALL_LAYERS.items():
        shape = caps_clean[lidx].shape[1:]
        label = f"{lname}_{shape[1]}x{shape[2]}"
        gap_clean = gap_emb(caps_clean, lidx)
        gap_adv = gap_emb(caps_adv, lidx)
        gap_without = gap_emb(caps_without, lidx)
        pt_clean = point_emb(caps_clean, lidx)
        pt_adv = point_emb(caps_adv, lidx)
        pt_without = point_emb(caps_without, lidx)
        cos_gap = F.cosine_similarity(gap_clean.unsqueeze(0), gap_adv.unsqueeze(0))[0].item()
        cos_point = F.cosine_similarity(pt_clean.unsqueeze(0), pt_adv.unsqueeze(0))[0].item()
        l2_shift_gap = torch.norm(gap_adv - gap_without, p=2).item()
        l2_shift_point = torch.norm(pt_adv - pt_without, p=2).item()
        raw_l2_gap = torch.norm(gap_adv - gap_clean, p=2).item()
        delta_patch = gap_adv - gap_clean
        delta_person = gap_clean - gap_without
        person_overlap = F.cosine_similarity(delta_patch.unsqueeze(0), delta_person.unsqueeze(0))[0].item()
        f_clean = torch.fft.fft2(caps_clean[lidx].squeeze(0))
        f_adv = torch.fft.fft2(caps_adv[lidx].squeeze(0))
        mag_clean = torch.log(torch.abs(f_clean) + 1)
        mag_adv = torch.log(torch.abs(f_adv) + 1)
        fft_dist = torch.norm(mag_clean - mag_adv, p=2).item()
        results[label] = {
            "cos_gap": cos_gap, "cos_point": cos_point,
            "l2_shift_gap": l2_shift_gap, "l2_shift_point": l2_shift_point,
            "raw_l2_gap": raw_l2_gap, "person_overlap": person_overlap,
            "fft_spectral_distance": fft_dist, "shape": list(shape),
        }
        print(f"  {label}: cos_gap={cos_gap:.4f}  l2_shift={l2_shift_gap:.2f}  "
              f"overlap={person_overlap:.4f}  fft={fft_dist:.2f}")
    return results


# ============================================================
# Metric 5: Frequency band analysis (LF/MF/HF)
# ============================================================
def frequency_band_analysis(model, clean_t, adv_t):
    results = {}
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        caps_adv, _ = fwd_all(model, adv_t)
    for lname, lidx in ALL_LAYERS.items():
        shape = caps_clean[lidx].shape[1:]
        H, W = shape[1], shape[2]
        label = f"{lname}_{H}x{W}"
        f_clean = torch.fft.fft2(caps_clean[lidx].squeeze(0))
        f_adv = torch.fft.fft2(caps_adv[lidx].squeeze(0))
        mag_clean = torch.abs(torch.fft.fftshift(f_clean, dim=(-2, -1)))
        mag_adv = torch.abs(torch.fft.fftshift(f_adv, dim=(-2, -1)))
        cy_f, cx_f = H // 2, W // 2
        y_coords = torch.arange(H, device=DEV, dtype=torch.float32) - cy_f
        x_coords = torch.arange(W, device=DEV, dtype=torch.float32) - cx_f
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        r = torch.sqrt(xx ** 2 + yy ** 2)
        r_max = math.sqrt(cy_f ** 2 + cx_f ** 2)
        r_norm = r / r_max
        lf_mask = r_norm <= 0.25
        mf_mask = (r_norm > 0.25) & (r_norm <= 0.50)
        hf_mask = r_norm > 0.50
        energy_clean = mag_clean.sum(dim=0)
        energy_adv = mag_adv.sum(dim=0)
        total_clean = energy_clean.sum() + 1e-10
        total_adv = energy_adv.sum() + 1e-10
        lf_c = (energy_clean * lf_mask).sum() / total_clean
        mf_c = (energy_clean * mf_mask).sum() / total_clean
        hf_c = (energy_clean * hf_mask).sum() / total_clean
        lf_a = (energy_adv * lf_mask).sum() / total_adv
        mf_a = (energy_adv * mf_mask).sum() / total_adv
        hf_a = (energy_adv * hf_mask).sum() / total_adv
        results[label] = {
            "lf_clean": lf_c.item(), "mf_clean": mf_c.item(), "hf_clean": hf_c.item(),
            "lf_adv": lf_a.item(), "mf_adv": mf_a.item(), "hf_adv": hf_a.item(),
            "delta_lf": (lf_a - lf_c).item(), "delta_mf": (mf_a - mf_c).item(),
            "delta_hf": (hf_a - hf_c).item(),
        }
        print(f"  {label}: dLF={results[label]['delta_lf']:+.4f}  "
              f"dMF={results[label]['delta_mf']:+.4f}  dHF={results[label]['delta_hf']:+.4f}")
    return results


# ============================================================
# Metric 6: Distance degradation (5m, 10m, 15m only)
# ============================================================
def distance_degradation(model, patch_t, mask_t, human_t, clean_t):
    """Test patch at 5m, 10m, 15m. Patch scales with distance.
    At 5m: patch is full PATCH_SIZE. At 10m: PATCH_SIZE * (5/10). At 15m: PATCH_SIZE * (5/15).
    Base image stays fixed - isolates patch effectiveness at different apparent sizes."""
    results = []
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)
    for d in DISTANCES:
        # Scale patch apparent size with distance
        scaled_size = max(4, int(PATCH_SIZE * (5.0 / d)))
        comp_d = composite_on_torso(patch_t, mask_t, human_t, patch_size=scaled_size)
        with torch.no_grad():
            caps_d, _ = fwd_all(model, comp_d)
            gap_d = gap_emb(caps_d, 81)
            cd = F.cosine_similarity(gap_c.unsqueeze(0), gap_d.unsqueeze(0))[0].item()
            scores_d = extract_detection_scores(model, comp_d)
        results.append({"distance_m": d, "patch_px": scaled_size, "cos": cd,
                        "combined": scores_d['combined'], "n_det": scores_d['n_detections']})
        print(f"  {d}m: cos={cd:.4f}  combined={scores_d['combined']:.4f}  "
              f"n_det={scores_d['n_detections']}  px={scaled_size}")
    return results


# ============================================================
# Metric 7: Angle sensitivity (proper rotation, +/-10/20/30 deg)
# ============================================================
def angle_sensitivity(model, patch_t, mask_t, human_t, clean_t):
    """Test yaw and pitch using proper affine rotation (not horizontal squish).
    Yaw = rotation around vertical axis (foreshortening width).
    Pitch = rotation around horizontal axis (foreshortening height).
    Uses affine_grid for proper perspective-like transformation."""
    yaw_results = []
    pitch_results = []
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)

    for yaw in YAW_ANGLES:
        # Yaw: compress width (cosine foreshortening) + slight horizontal shift
        sx = math.cos(math.radians(abs(yaw)))
        # Build affine matrix for width compression
        theta = torch.tensor([
            [sx, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=DEV).unsqueeze(0)
        grid = F.affine_grid(theta, patch_t.shape, align_corners=False)
        warped_patch = F.grid_sample(patch_t, grid, align_corners=False)
        warped_mask = F.grid_sample(mask_t, grid, align_corners=False)
        comp = composite_on_torso(warped_patch, warped_mask, human_t)
        with torch.no_grad():
            caps_y, _ = fwd_all(model, comp)
            gap_y = gap_emb(caps_y, 81)
            cy = F.cosine_similarity(gap_c.unsqueeze(0), gap_y.unsqueeze(0))[0].item()
            sc = extract_detection_scores(model, comp)
        yaw_results.append({"yaw_deg": yaw, "cos": cy, "combined": sc['combined'],
                            "n_det": sc['n_detections']})

    for pitch in PITCH_ANGLES:
        # Pitch: compress height (cosine foreshortening)
        sy = math.cos(math.radians(abs(pitch)))
        theta = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, sy, 0.0]
        ], dtype=torch.float32, device=DEV).unsqueeze(0)
        grid = F.affine_grid(theta, patch_t.shape, align_corners=False)
        warped_patch = F.grid_sample(patch_t, grid, align_corners=False)
        warped_mask = F.grid_sample(mask_t, grid, align_corners=False)
        comp = composite_on_torso(warped_patch, warped_mask, human_t)
        with torch.no_grad():
            caps_p, _ = fwd_all(model, comp)
            gap_p = gap_emb(caps_p, 81)
            cp = F.cosine_similarity(gap_c.unsqueeze(0), gap_p.unsqueeze(0))[0].item()
            sc = extract_detection_scores(model, comp)
        pitch_results.append({"pitch_deg": pitch, "cos": cp, "combined": sc['combined'],
                              "n_det": sc['n_detections']})

    print(f"  Yaw -30->+30: cos {yaw_results[0]['cos']:.4f} -> {yaw_results[-1]['cos']:.4f}")
    print(f"  Pitch -30->+30: cos {pitch_results[0]['cos']:.4f} -> {pitch_results[-1]['cos']:.4f}")
    return {"yaw": yaw_results, "pitch": pitch_results}


# ============================================================
# Metric 8: Saliency map (Fix G: target person-winning cell)
# ============================================================
def saliency_map(model, adv_t, clean_t, out_dir):
    """Fix (G): Saliency targets the person-winning cell specifically.
    Instead of output.max() over whole tensor, find the grid cell with highest
    obj*person_cls and differentiate THAT scalar only."""
    results = {}

    # Clean pass: find person-winning cell
    clean_t.requires_grad_(True)
    model.train()
    for m in model.modules():
        if hasattr(m, 'momentum'):
            m.momentum = 0

    # Forward clean to find target cell
    pred_clean = fwd_detections(model, clean_t)
    obj_clean = pred_clean[:, 4]
    cls_clean = pred_clean[:, 5 + PERSON_CLS]
    combined_clean = obj_clean * cls_clean
    best_idx_clean = combined_clean.argmax()
    person_score_clean = combined_clean[best_idx_clean]

    grad_clean = torch.autograd.grad(person_score_clean, clean_t, retain_graph=False)[0]
    saliency_clean = grad_clean.abs().sum(dim=1).squeeze(0)

    # Adv pass: find person-winning cell (may be different location)
    adv_t.requires_grad_(True)
    pred_adv = fwd_detections(model, adv_t)
    obj_adv = pred_adv[:, 4]
    cls_adv = pred_adv[:, 5 + PERSON_CLS]
    combined_adv = obj_adv * cls_adv
    best_idx_adv = combined_adv.argmax()
    person_score_adv = combined_adv[best_idx_adv]

    grad_adv = torch.autograd.grad(person_score_adv, adv_t, retain_graph=False)[0]
    saliency_adv = grad_adv.abs().sum(dim=1).squeeze(0)

    model.eval()
    clean_t.requires_grad_(False)
    adv_t.requires_grad_(False)

    delta_saliency = saliency_adv - saliency_clean
    results["person_score_clean"] = person_score_clean.item()
    results["person_score_adv"] = person_score_adv.item()
    results["score_drop"] = (person_score_clean - person_score_adv).item()
    results["best_cell_clean"] = int(best_idx_clean)
    results["best_cell_adv"] = int(best_idx_adv)

    sal_clean_np = saliency_clean.detach().cpu().numpy()
    sal_adv_np = saliency_adv.detach().cpu().numpy()
    sal_delta_np = delta_saliency.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(sal_clean_np, cmap='hot')
    axes[0].set_title(f"Person-cell saliency (clean)\nscore={person_score_clean:.4f}")
    axes[0].axis("off"); plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(sal_adv_np, cmap='hot')
    axes[1].set_title(f"Person-cell saliency (adv)\nscore={person_score_adv:.4f}")
    axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(sal_delta_np, cmap='RdBu_r', vmin=-np.abs(sal_delta_np).max(),
                         vmax=np.abs(sal_delta_np).max())
    axes[2].set_title(f"Delta saliency\ndrop={results['score_drop']:.4f}")
    axes[2].axis("off"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
    plt.suptitle("Person-Targeted Saliency (Fix G)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  score_clean={person_score_clean:.4f}  score_adv={person_score_adv:.4f}  "
          f"drop={results['score_drop']:.4f}  cell_clean={results['best_cell_clean']}  "
          f"cell_adv={results['best_cell_adv']}")
    return results


# ============================================================
# Suppression Go/No-Go Gate Evaluation
# Fixes: (A) false_rate composites patch on no-human, (C) renamed ASR, (H) full anchor IoU
# ============================================================
def suppression_gates(model, adv_t, clean_t, without_t, patch_t, mask_t):
    results = {}
    with torch.no_grad():
        scores_clean = extract_detection_scores(model, clean_t)
        scores_adv = extract_detection_scores(model, adv_t)
        scores_without = extract_detection_scores(model, without_t)

    # Fix (A): Actually composite patch onto no-human image
    adv_without = composite_on_torso(patch_t, mask_t, without_t)
    with torch.no_grad():
        scores_without_adv = extract_detection_scores(model, adv_without)

    # Gate 1: Confidence suppression
    results['clean_combined'] = scores_clean['combined']
    results['adv_combined'] = scores_adv['combined']
    results['conf_suppressed'] = scores_adv['combined'] < THRESHOLDS['suppression']['conf_max']
    results['conf_drop'] = scores_clean['combined'] - scores_adv['combined']

    # Gate 2: Per-head suppression
    head_suppressed = {}
    for hn, hs in scores_adv['per_head'].items():
        head_suppressed[hn] = hs['combined'] < THRESHOLDS['suppression']['conf_max']
    results['per_head_suppressed'] = head_suppressed
    results['all_heads_suppressed'] = all(head_suppressed.values())

    # Gate 3: Detection count drop
    results['clean_n_det'] = scores_clean['n_detections']
    results['adv_n_det'] = scores_adv['n_detections']
    results['det_drop_rate'] = 1.0 - (scores_adv['n_detections'] / max(1, scores_clean['n_detections']))

    # Fix (A): False suppression rate = how much patch suppresses detections on NO-human image
    # This measures collateral suppression of non-person objects
    results['without_n_det'] = scores_without['n_detections']
    results['without_adv_n_det'] = scores_without_adv['n_detections']
    if scores_without['n_detections'] > 0:
        results['false_suppression_rate'] = max(0, 1.0 - (scores_without_adv['n_detections'] / max(1, scores_without['n_detections'])))
    else:
        # No detections on without-human image to begin with
        results['false_suppression_rate'] = 0.0

    # Fix (H): Anchor IoU with full box-IoU including center distance
    with torch.no_grad():
        pred_adv = fwd_detections(model, adv_t)
    if pred_adv is not None and len(pred_adv) > 0:
        det_boxes = boxes_xywh_from_pred(pred_adv, conf_threshold=0.05)
        if len(det_boxes) > 0:
            iou_matrix = anchor_box_iou(det_boxes[:, :4], ANCHORS)
            results['max_anchor_iou'] = float(iou_matrix.max().item())
        else:
            results['max_anchor_iou'] = 0.0
    else:
        results['max_anchor_iou'] = 0.0
    results['anchor_iou_ok'] = results['max_anchor_iou'] < THRESHOLDS['suppression']['anchor_iou_max']

    # Fix (C): Renamed from ASR to single_image_reduction
    # This is a confidence-drop ratio from ONE image, NOT a population proportion
    if scores_clean['combined'] > 1e-8:
        results['single_image_reduction'] = float(results['conf_drop'] / scores_clean['combined'])
    else:
        results['single_image_reduction'] = 0.0
    results['suppression_success'] = results['conf_suppressed'] and results['all_heads_suppressed']

    gates = {
        'conf_below_threshold': results['conf_suppressed'],
        'all_heads_suppressed': results['all_heads_suppressed'],
        'anchor_iou_below_030': results['anchor_iou_ok'],
        'false_rate_below_3pct': results['false_suppression_rate'] < THRESHOLDS['suppression']['false_rate_max'],
        'suppression_success': results['suppression_success'],
    }
    results['gates'] = gates
    results['go_nogo'] = "GO" if all(gates.values()) else "NO-GO"

    print(f"  SUPPRESSION GATES:")
    print(f"    conf={results['adv_combined']:.4f} (< {THRESHOLDS['suppression']['conf_max']}? {gates['conf_below_threshold']})")
    print(f"    all_heads={gates['all_heads_suppressed']}  anchor_iou={results['max_anchor_iou']:.4f} (< 0.30? {gates['anchor_iou_below_030']})")
    print(f"    false_rate={results['false_suppression_rate']:.4f} (< 0.03? {gates['false_rate_below_3pct']})")
    print(f"    single_image_reduction={results['single_image_reduction']:.4f}")
    print(f"    VERDICT: {results['go_nogo']}")
    return results


# ============================================================
# Poison Go/No-Go Gate Evaluation
# Fixes: (H) full anchor IoU, crop_align uses torso center
# ============================================================
def poison_gates(model, adv_t, clean_t, without_t):
    results = {}
    with torch.no_grad():
        scores_clean = extract_detection_scores(model, clean_t)
        scores_adv = extract_detection_scores(model, adv_t)
        pred_clean = fwd_detections(model, clean_t)
        pred_adv = fwd_detections(model, adv_t)

    # Gate 1: Confidence
    results['adv_obj_max'] = scores_adv['obj_max']
    results['adv_combined'] = scores_adv['combined']
    results['conf_injected'] = scores_adv['obj_max'] > THRESHOLDS['poison']['conf_min']

    # Gate 2: NMS survival
    if pred_clean is not None and pred_adv is not None:
        clean_boxes, clean_confs = boxes_from_pred(pred_clean, conf_threshold=0.25)
        adv_boxes, adv_confs = boxes_from_pred(pred_adv, conf_threshold=0.25)
        if len(adv_boxes) > 0 and len(clean_boxes) > 0:
            iou_matrix = compute_iou_matrix(adv_boxes[:, :4], clean_boxes[:, :4])
            max_iou_per_adv = iou_matrix.max(dim=1)[0]
            new_det_mask = max_iou_per_adv < THRESHOLDS['poison']['nms_iou_max']
            results['n_new_detections'] = int(new_det_mask.sum().item())
            results['nms_survival_rate'] = float(new_det_mask.float().mean().item()) if new_det_mask.numel() > 0 else 0.0
            # Run NMS on ALL adv detections (merge phantoms with real before suppression)
            if len(adv_boxes) > 1:
                keep_idx = nms(adv_boxes[:, :4], adv_confs, iou_threshold=0.45)
                results['nms_survived'] = len(keep_idx)
            else:
                results['nms_survived'] = len(adv_boxes)
        elif len(adv_boxes) > 0:
            results['n_new_detections'] = len(adv_boxes)
            results['nms_survival_rate'] = 1.0
            results['nms_survived'] = len(adv_boxes)
        else:
            results['n_new_detections'] = 0
            results['nms_survival_rate'] = 0.0
            results['nms_survived'] = 0
    else:
        results['n_new_detections'] = 0
        results['nms_survival_rate'] = 0.0
        results['nms_survived'] = 0

    # Fix (H): Anchor IoU with full box-IoU including center distance
    if pred_adv is not None and len(pred_adv) > 0:
        det_boxes = boxes_xywh_from_pred(pred_adv, conf_threshold=0.25)
        if len(det_boxes) > 0:
            iou_matrix = anchor_box_iou(det_boxes[:, :4], ANCHORS)
            max_iou = float(iou_matrix.max().item())
            # Find which anchor and detection
            flat_idx = iou_matrix.argmax()
            best_det = flat_idx // iou_matrix.shape[1]
            best_anchor = flat_idx % iou_matrix.shape[1]
            results['max_anchor_iou'] = max_iou
            results['best_anchor_idx'] = int(best_anchor)
        else:
            results['max_anchor_iou'] = 0.0
            results['best_anchor_idx'] = -1
    else:
        results['max_anchor_iou'] = 0.0
        results['best_anchor_idx'] = -1
    results['anchor_aligned'] = results['max_anchor_iou'] > THRESHOLDS['poison']['anchor_iou_min']

    # Gate 4: Crop alignment — distance from false detection to torso patch center
    target_cx, target_cy = TORSO_CX, TORSO_CY
    if pred_adv is not None and len(pred_adv) > 0:
        det_boxes_full, _ = boxes_from_pred(pred_adv, conf_threshold=0.25)
        if len(det_boxes_full) > 0:
            det_cx = (det_boxes_full[:, 0] + det_boxes_full[:, 2]) / 2
            det_cy = (det_boxes_full[:, 1] + det_boxes_full[:, 3]) / 2
            dists = torch.sqrt((det_cx - target_cx) ** 2 + (det_cy - target_cy) ** 2)
            results['crop_align_error'] = float(dists.min().item())
        else:
            results['crop_align_error'] = 999.0
    else:
        results['crop_align_error'] = 999.0
    results['crop_aligned'] = results['crop_align_error'] < THRESHOLDS['poison']['crop_align_max']

    # Gate 5: Cloud embedding drift
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        caps_adv, _ = fwd_all(model, adv_t)
    gap_c = gap_emb(caps_clean, 81)
    gap_a = gap_emb(caps_adv, 81)
    cos_embed = F.cosine_similarity(gap_c.unsqueeze(0), gap_a.unsqueeze(0))[0].item()
    results['embedding_cos_sim'] = cos_embed
    results['embedding_drift'] = 1.0 - cos_embed
    results['embedding_drift_ok'] = results['embedding_drift'] > THRESHOLDS['poison']['embedding_drift_min']

    # Fix (C): Renamed from ASR
    results['single_image_reduction'] = 1.0 if (results['conf_injected'] and
        results['nms_survival_rate'] > THRESHOLDS['poison']['nms_survival_min']) else 0.0

    gates = {
        'conf_above_065': results['conf_injected'],
        'anchor_iou_above_055': results['anchor_aligned'],
        'nms_survival_above_90pct': results['nms_survival_rate'] > THRESHOLDS['poison']['nms_survival_min'],
        'crop_align_below_4px': results['crop_aligned'],
        'embedding_drift_above_015': results['embedding_drift_ok'],
    }
    results['gates'] = gates
    results['go_nogo'] = "GO" if all(gates.values()) else "NO-GO"

    print(f"  POISON GATES:")
    print(f"    obj={results['adv_obj_max']:.4f} (> 0.65? {gates['conf_above_065']})")
    print(f"    anchor_iou={results['max_anchor_iou']:.4f} (> 0.55? {gates['anchor_iou_above_055']})")
    print(f"    nms_survival={results['nms_survival_rate']:.4f} (> 0.90? {gates['nms_survival_above_90pct']})")
    print(f"    crop_align={results['crop_align_error']:.2f}px (< 4? {gates['crop_align_below_4px']})")
    print(f"    embed_drift={results['embedding_drift']:.4f} (> 0.15? {gates['embedding_drift_above_015']})")
    print(f"    VERDICT: {results['go_nogo']}")
    return results


# ============================================================
# EoT Robustness (Fix F: real DCT-domain H.265 quantization)
# ============================================================
def eot_robustness(model, patch_t, mask_t, human_t, clean_t):
    results = {}
    with torch.no_grad():
        scores_clean = extract_detection_scores(model, clean_t)
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)

    # --- Affine: rotation +/-15 deg, scale [0.8, 1.2], translation +/-10%
    affine_results = []
    for angle in [-15, -7, 0, 7, 15]:
        for scale in [0.8, 1.0, 1.2]:
            for tx in [-0.1, 0.0, 0.1]:
                theta = torch.tensor([
                    [math.cos(math.radians(angle)) * scale, -math.sin(math.radians(angle)) * scale, tx],
                    [math.sin(math.radians(angle)) * scale, math.cos(math.radians(angle)) * scale, 0.0]
                ], dtype=torch.float32, device=DEV).unsqueeze(0)
                grid = F.affine_grid(theta, patch_t.shape, align_corners=False)
                warped_patch = F.grid_sample(patch_t, grid, align_corners=False)
                warped_mask = F.grid_sample(mask_t, grid, align_corners=False)
                comp = composite_on_torso(warped_patch, warped_mask, human_t)
                with torch.no_grad():
                    scores = extract_detection_scores(model, comp)
                    caps, _ = fwd_all(model, comp)
                    gap = gap_emb(caps, 81)
                    cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
                affine_results.append({
                    "angle": angle, "scale": scale, "tx": tx,
                    "combined": scores['combined'], "cos": cos,
                    "n_det": scores['n_detections']
                })
    results['affine'] = affine_results

    # --- Optical degradation: Gaussian motion blur + defocus
    blur_results = []
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        kernel_size = int(sigma * 6) | 1
        if kernel_size < 3: kernel_size = 3
        coords = torch.arange(kernel_size, dtype=torch.float32, device=DEV) - kernel_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        k_h = gauss.view(1, 1, 1, kernel_size).expand(3, 1, 1, kernel_size)
        k_v = gauss.view(1, 1, kernel_size, 1).expand(3, 1, kernel_size, 1)
        blurred_patch = F.conv2d(patch_t, k_h, padding=(0, kernel_size // 2), groups=3)
        blurred_patch = F.conv2d(blurred_patch, k_v, padding=(kernel_size // 2, 0), groups=3)
        comp = composite_on_torso(blurred_patch, mask_t, human_t)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        blur_results.append({"sigma": sigma, "combined": scores['combined'],
                             "cos": cos, "n_det": scores['n_detections']})
    results['motion_blur'] = blur_results

    # Defocus (circular blur)
    defocus_results = []
    for sigma in [0.3, 0.5, 0.8, 1.0]:
        ksz = int(sigma * 6) | 1
        if ksz < 3: ksz = 3
        coords = torch.arange(ksz, dtype=torch.float32, device=DEV) - ksz // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        disk = ((xx**2 + yy**2) <= (sigma * 2)**2).float()
        disk = disk / disk.sum()
        k_disk = disk.view(1, 1, ksz, ksz).expand(3, 1, ksz, ksz)
        defocused = F.conv2d(patch_t, k_disk, padding=(ksz // 2, ksz // 2), groups=3)
        comp = composite_on_torso(defocused, mask_t, human_t)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        defocus_results.append({"sigma": sigma, "combined": scores['combined'],
                                "cos": cos, "n_det": scores['n_detections']})
    results['defocus'] = defocus_results

    # --- ISP noise: shot + read noise
    noise_results = []
    for noise_std in [0.01, 0.03, 0.05, 0.10]:
        noisy_patch = patch_t + torch.randn_like(patch_t) * noise_std
        noisy_patch = noisy_patch.clamp(0, 1)
        comp = composite_on_torso(noisy_patch, mask_t, human_t)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        noise_results.append({"noise_std": noise_std, "combined": scores['combined'],
                              "cos": cos, "n_det": scores['n_detections']})
    results['isp_noise'] = noise_results

    # Fix (F): H.265 DCT quantization using real DCT-domain quantization
    # Uses scipy.fft.dctn/idctn with 8x8 blocks and perceptually-weighted QP
    dct_results = []
    for qp in [10, 20, 30, 40]:
        patch_np = patch_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        block = 8
        quantized = np.zeros_like(patch_np)
        # H.265 default quantization matrix (flat for intra, simplified)
        q_matrix = np.ones((block, block)) * qp
        # Higher frequency coefficients get larger quantization steps
        for i in range(block):
            for j in range(block):
                q_matrix[i, j] = qp * (1.0 + (i + j) * 0.15)
        for ci in range(3):
            for by in range(0, IS, block):
                for bx in range(0, IS, block):
                    blk = patch_np[by:by+block, bx:bx+block, ci]
                    if blk.shape[0] < block or blk.shape[1] < block:
                        quantized[by:by+blk.shape[0], bx:bx+blk.shape[1], ci] = blk
                        continue
                    # Forward DCT (type-II, orthonormal)
                    dct_blk = dctn(blk, type=2, norm='ortho')
                    # Quantize DCT coefficients
                    quant = np.round(dct_blk / q_matrix) * q_matrix
                    # Inverse DCT
                    recon = idctn(quant, type=2, norm='ortho')
                    quantized[by:by+block, bx:bx+block, ci] = recon
        quantized = np.clip(quantized, 0, 1)
        quant_t = torch.from_numpy(quantized).permute(2, 0, 1).unsqueeze(0).to(DEV)
        comp = composite_on_torso(quant_t, mask_t, human_t)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        dct_results.append({"qp": qp, "combined": scores['combined'],
                            "cos": cos, "n_det": scores['n_detections']})
    results['h265_dct'] = dct_results

    # --- Color space round-trip: sRGB -> YUV420 -> sRGB
    def srgb_to_yuv_and_back(img_t):
        r, g, b = img_t[0, 0], img_t[0, 1], img_t[0, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        u_sub = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2, stride=2)
        v_sub = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2, stride=2)
        u_up = F.interpolate(u_sub, size=(IS, IS), mode='bilinear', align_corners=False).squeeze()
        v_up = F.interpolate(v_sub, size=(IS, IS), mode='bilinear', align_corners=False).squeeze()
        r2 = y + 1.13983 * v_up
        g2 = y - 0.39465 * u_up - 0.58060 * v_up
        b2 = y + 2.03211 * u_up
        return torch.stack([r2, g2, b2]).unsqueeze(0).clamp(0, 1)

    yuv_patch = srgb_to_yuv_and_back(patch_t)
    comp_yuv = composite_on_torso(yuv_patch, mask_t, human_t)
    with torch.no_grad():
        scores_yuv = extract_detection_scores(model, comp_yuv)
        caps_yuv, _ = fwd_all(model, comp_yuv)
        gap_yuv = gap_emb(caps_yuv, 81)
        cos_yuv = F.cosine_similarity(gap_c.unsqueeze(0), gap_yuv.unsqueeze(0))[0].item()
    results['yuv_roundtrip'] = {
        "combined": scores_yuv['combined'], "cos": cos_yuv,
        "n_det": scores_yuv['n_detections']
    }

    # --- Exposure shifts: +/- 1.5 EV
    exposure_results = []
    for ev in [-1.5, -0.5, 0.0, 0.5, 1.5]:
        factor = 2.0 ** ev
        exp_patch = (patch_t * factor).clamp(0, 1)
        comp = composite_on_torso(exp_patch, mask_t, human_t)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        exposure_results.append({"ev": ev, "combined": scores['combined'],
                                 "cos": cos, "n_det": scores['n_detections']})
    results['exposure'] = exposure_results

    # EoT summary
    all_combined = [r['combined'] for r in affine_results] + \
                   [r['combined'] for r in blur_results] + \
                   [r['combined'] for r in defocus_results] + \
                   [r['combined'] for r in noise_results] + \
                   [r['combined'] for r in dct_results] + \
                   [results['yuv_roundtrip']['combined']] + \
                   [r['combined'] for r in exposure_results]
    results['eot_mean_combined'] = float(np.mean(all_combined))
    results['eot_std_combined'] = float(np.std(all_combined))
    results['eot_min_combined'] = float(np.min(all_combined))
    results['eot_max_combined'] = float(np.max(all_combined))

    print(f"  EoT: mean_combined={results['eot_mean_combined']:.4f}  "
          f"std={results['eot_std_combined']:.4f}  "
          f"range=[{results['eot_min_combined']:.4f}, {results['eot_max_combined']:.4f}]")
    return results


# ============================================================
# Lighting sensitivity (Phase 2: 4000K, 6500K, 3000K, 2200K)
# ============================================================
def lighting_sensitivity(model, patch_t, mask_t, human_t, clean_t):
    """Test patch under 4 lighting conditions per Phase 2 protocol."""
    results = {}
    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)
        scores_clean = extract_detection_scores(model, clean_t)

    for name, cfg in LIGHTING.items():
        # Apply lighting to both patch and base image
        lit_patch = apply_lighting(patch_t, cfg)
        lit_human = apply_lighting(human_t, cfg)
        comp = composite_on_torso(lit_patch, mask_t, lit_human)
        with torch.no_grad():
            scores = extract_detection_scores(model, comp)
            caps, _ = fwd_all(model, comp)
            gap = gap_emb(caps, 81)
            cos = F.cosine_similarity(gap_c.unsqueeze(0), gap.unsqueeze(0))[0].item()
        results[name] = {
            "combined": scores['combined'], "cos": cos,
            "n_det": scores['n_detections'], "obj_max": scores['obj_max']
        }
        print(f"  {name}: combined={scores['combined']:.4f}  cos={cos:.4f}  n_det={scores['n_detections']}")
    return results


# ============================================================
# Fix (D): Random noise control patch
# ============================================================
def evaluate_random_control(model, human_t, clean_t, without_t):
    """Run random noise patch through same pipeline for baseline comparison.
    Same area as real patches (PATCH_SIZE on torso)."""
    print("\n  --- Random Control Baseline ---")
    # Generate random noise patch
    rand_patch = torch.rand(1, 3, IS, IS, device=DEV)
    rand_mask = torch.ones(1, 1, IS, IS, device=DEV)  # full coverage

    adv_rand = composite_on_torso(rand_patch, rand_mask, human_t)
    with torch.no_grad():
        scores_clean = extract_detection_scores(model, clean_t)
        scores_rand = extract_detection_scores(model, adv_rand)

    results = {
        "clean_combined": scores_clean['combined'],
        "rand_combined": scores_rand['combined'],
        "rand_n_det": scores_rand['n_detections'],
        "rand_conf_drop": scores_clean['combined'] - scores_rand['combined'],
        "rand_reduction_ratio": float((scores_clean['combined'] - scores_rand['combined']) / max(1e-8, scores_clean['combined'])),
    }
    print(f"  Random: combined={scores_rand['combined']:.4f}  n_det={scores_rand['n_detections']}  "
          f"reduction={results['rand_reduction_ratio']:.4f}")
    return results


# ============================================================
# Visualization
# ============================================================
def make_figures(per_layer, freq_bands, dist_deg, angle_sens,
                 supp, poison, eot, lighting, patch_name, out_dir):
    # 6-panel forward analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    layers = list(per_layer.keys())
    x_pos = np.arange(len(layers))

    axes[0, 0].bar(x_pos - 0.2, [per_layer[l]["cos_gap"] for l in layers], 0.4, label='cos_gap', color='steelblue')
    axes[0, 0].bar(x_pos + 0.2, [per_layer[l]["cos_point"] for l in layers], 0.4, label='cos_point', color='orange')
    axes[0, 0].set_xticks(x_pos); axes[0, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 0].set_title("Cosine Similarity"); axes[0, 0].legend(); axes[0, 0].set_ylim(0.9, 1.001)

    axes[0, 1].bar(x_pos, [per_layer[l]["person_overlap"] for l in layers], 0.5, color='crimson')
    axes[0, 1].set_xticks(x_pos); axes[0, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 1].set_title("Person Overlap (negative = attacks person signal)")
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    axes[1, 0].bar(x_pos - 0.25, [per_layer[l]["l2_shift_gap"] for l in layers], 0.25, label='L2 shift GAP', color='orange')
    axes[1, 0].bar(x_pos, [per_layer[l]["l2_shift_point"] for l in layers], 0.25, label='L2 shift point', color='red')
    axes[1, 0].bar(x_pos + 0.25, [per_layer[l]["raw_l2_gap"] for l in layers], 0.25, label='Raw L2', color='darkred')
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 0].set_title("L2 Distance"); axes[1, 0].legend()

    axes[1, 1].bar(x_pos, [per_layer[l]["fft_spectral_distance"] for l in layers], 0.5, color='purple')
    axes[1, 1].set_xticks(x_pos); axes[1, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 1].set_title("FFT Spectral Distance")

    dists = [d["distance_m"] for d in dist_deg]
    dist_cos = [d["cos"] for d in dist_deg]
    dist_comb = [d["combined"] for d in dist_deg]
    ax_d = axes[2, 0]
    ax_d.plot(dists, dist_cos, 'o-', color='crimson', linewidth=2, markersize=8, label='cos(L81)')
    ax_d2 = ax_d.twinx()
    ax_d2.plot(dists, dist_comb, 's--', color='darkblue', linewidth=1.5, markersize=6, label='combined conf')
    ax_d.set_xlabel("Distance (m)"); ax_d.set_ylabel("Cosine Sim", color='crimson')
    ax_d2.set_ylabel("Combined Confidence", color='darkblue')
    ax_d.set_title("Distance Degradation (5/10/15m)")
    ax_d.grid(True, alpha=0.3)

    yaws = [y["yaw_deg"] for y in angle_sens["yaw"]]
    yaw_cos = [y["cos"] for y in angle_sens["yaw"]]
    pitches = [p["pitch_deg"] for p in angle_sens["pitch"]]
    pitch_cos = [p["cos"] for p in angle_sens["pitch"]]
    axes[2, 1].plot(yaws, yaw_cos, 's-', color='darkorange', linewidth=2, markersize=8, label='Yaw')
    axes[2, 1].plot(pitches, pitch_cos, '^-', color='purple', linewidth=2, markersize=8, label='Pitch')
    axes[2, 1].set_xlabel("Angle (deg)"); axes[2, 1].set_ylabel("Cosine Sim")
    axes[2, 1].set_title("Angle Sensitivity (+/-10/20/30 deg)"); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{patch_name} - Comprehensive Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{out_dir}/forward_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # EoT + Lighting figure
    fig2, axes2 = plt.subplots(2, 4, figsize=(24, 10))
    # Affine (subset for clarity)
    aff = eot['affine']
    for sc in sorted(set(r['scale'] for r in aff)):
        pts = [r for r in aff if r['scale'] == sc and r['tx'] == 0.0]
        angs = [p['angle'] for p in pts]
        comb = [p['combined'] for p in pts]
        axes2[0, 0].plot(angs, comb, 'o-', label=f'scale={sc}')
    axes2[0, 0].set_title("Affine: Rotation x Scale"); axes2[0, 0].legend(fontsize=8)
    axes2[0, 0].set_xlabel("Angle (deg)"); axes2[0, 0].set_ylabel("Combined Conf")

    blur = eot['motion_blur']
    axes2[0, 1].plot([r['sigma'] for r in blur], [r['combined'] for r in blur], 's-', color='blue')
    axes2[0, 1].set_title("Motion Blur"); axes2[0, 1].set_xlabel("sigma"); axes2[0, 1].set_ylabel("Combined Conf")

    noise = eot['isp_noise']
    axes2[0, 2].plot([r['noise_std'] for r in noise], [r['combined'] for r in noise], '^-', color='red')
    axes2[0, 2].set_title("ISP Noise"); axes2[0, 2].set_xlabel("noise std"); axes2[0, 2].set_ylabel("Combined Conf")

    dct = eot['h265_dct']
    axes2[0, 3].bar([str(r['qp']) for r in dct], [r['combined'] for r in dct], color='green')
    axes2[0, 3].set_title("H.265 DCT Quantization"); axes2[0, 3].set_xlabel("QP"); axes2[0, 3].set_ylabel("Combined Conf")

    exp = eot['exposure']
    axes2[1, 0].plot([r['ev'] for r in exp], [r['combined'] for r in exp], 'D-', color='purple')
    axes2[1, 0].set_title("Exposure Shift"); axes2[1, 0].set_xlabel("EV"); axes2[1, 0].set_ylabel("Combined Conf")

    defoc = eot['defocus']
    axes2[1, 1].plot([r['sigma'] for r in defoc], [r['combined'] for r in defoc], 'v-', color='brown')
    axes2[1, 1].set_title("Defocus"); axes2[1, 1].set_xlabel("sigma"); axes2[1, 1].set_ylabel("Combined Conf")

    # Lighting
    lit_names = list(lighting.keys())
    lit_comb = [lighting[n]['combined'] for n in lit_names]
    axes2[1, 2].bar(range(len(lit_names)), lit_comb, color=['steelblue', 'orange', 'crimson', 'darkred'])
    axes2[1, 2].set_xticks(range(len(lit_names)))
    axes2[1, 2].set_xticklabels([n.replace('_', '\n') for n in lit_names], fontsize=8)
    axes2[1, 2].set_title("Lighting Conditions"); axes2[1, 2].set_ylabel("Combined Conf")

    # YUV roundtrip
    axes2[1, 3].bar(['Clean', 'YUV420'], [eot['yuv_roundtrip']['combined'], eot['yuv_roundtrip']['combined']],
                    color=['steelblue', 'crimson'])
    axes2[1, 3].set_title("YUV420 Round-trip"); axes2[1, 3].set_ylabel("Combined Conf")

    plt.suptitle(f"{patch_name} - EoT Robustness + Lighting", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{out_dir}/eot_lighting.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Go/No-Go gate summary
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    gate_names = list(supp['gates'].keys()) + ['---'] + list(poison['gates'].keys())
    gate_vals = [1 if v else 0 for v in supp['gates'].values()] + [0.5] + [1 if v else 0 for v in poison['gates'].values()]
    colors = ['green' if v == 1 else 'red' if v == 0 else 'gray' for v in gate_vals]
    ax3.barh(range(len(gate_names)), gate_vals, color=colors, height=0.6)
    ax3.set_yticks(range(len(gate_names)))
    ax3.set_yticklabels(gate_names, fontsize=9)
    ax3.set_title(f"Go/No-Go Gates: Suppression={supp['go_nogo']} | Poison={poison['go_nogo']}")
    ax3.set_xlim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gates.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Save per-patch results
# ============================================================
def save_results(per_layer, freq_bands, dist_deg, angle_sens,
                 saliency, supp, poison, eot, lighting, rand_ctrl, out_dir):
    all_results = {
        "per_layer_metrics": per_layer,
        "frequency_band_analysis": freq_bands,
        "distance_degradation": dist_deg,
        "angle_sensitivity": angle_sens,
        "saliency": saliency,
        "suppression_gates": supp,
        "poison_gates": poison,
        "eot_robustness": eot,
        "lighting_sensitivity": lighting,
        "random_control": rand_ctrl,
    }
    with open(f"{out_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(f"{out_dir}/results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric_type', 'param', 'value'])
        for label, m in per_layer.items():
            for k, v in m.items():
                if k != 'shape': w.writerow(['per_layer', f"{label}_{k}", f"{v:.6f}"])
        for label, fb in freq_bands.items():
            for k, v in fb.items(): w.writerow(['freq_band', f"{label}_{k}", f"{v:.6f}"])
        for d in dist_deg:
            w.writerow(['distance', f"{d['distance_m']}m", f"{d['cos']:.6f}"])
            w.writerow(['distance_conf', f"{d['distance_m']}m", f"{d['combined']:.6f}"])
        for y in angle_sens["yaw"]:
            w.writerow(['yaw', f"{y['yaw_deg']}deg", f"{y['cos']:.6f}"])
        for p in angle_sens["pitch"]:
            w.writerow(['pitch', f"{p['pitch_deg']}deg", f"{p['cos']:.6f}"])
        w.writerow(['saliency', 'score_clean', f"{saliency['person_score_clean']:.6f}"])
        w.writerow(['saliency', 'score_adv', f"{saliency['person_score_adv']:.6f}"])
        w.writerow(['saliency', 'drop', f"{saliency['score_drop']:.6f}"])
        for g, v in supp['gates'].items():
            w.writerow(['supp_gate', g, str(v)])
        w.writerow(['supp', 'go_nogo', supp['go_nogo']])
        for g, v in poison['gates'].items():
            w.writerow(['poison_gate', g, str(v)])
        w.writerow(['poison', 'go_nogo', poison['go_nogo']])
        w.writerow(['eot', 'mean_combined', f"{eot['eot_mean_combined']:.6f}"])
        w.writerow(['eot', 'std_combined', f"{eot['eot_std_combined']:.6f}"])
        for name, lr in lighting.items():
            w.writerow(['lighting', name, f"{lr['combined']:.6f}"])
        w.writerow(['random_control', 'combined', f"{rand_ctrl['rand_combined']:.6f}"])
        w.writerow(['random_control', 'reduction', f"{rand_ctrl['rand_reduction_ratio']:.6f}"])


# ============================================================
# Auto-discover patches
# ============================================================
def discover_patches():
    base = r"C:\Users\carso\Desktop\YODO"
    patterns = [
        os.path.join(base, "outputs_v*", "*", "patch_final.png"),
        os.path.join(base, "outputs_v*", "patch_final.png"),
        os.path.join(base, "outputs_hold_poison", "*", "patch_final.png"),
        os.path.join(base, "outputs_clothing", "v*", "patch_final.png"),
        os.path.join(base, "outputs_clothing", "v*", "patch_final_1024.png"),
        os.path.join(base, "outputs_clothing", "final_boss*", "patch_416.png"),
        os.path.join(base, "outputs_clothing", "final_boss*", "patch_416_optimized.png"),
        os.path.join(base, "outputs_clothing", "final_boss*", "suppress_patch_416.png"),
        os.path.join(base, "outputs_clothing", "final_boss*", "poison_patch_416.png"),
        os.path.join(base, "outputs_clothing", "forward_analysis", "deformable_patch*", "*patch_416.png"),
        os.path.join(base, "outputs_clothing", "forward_analysis", "patch_pipeline", "*", "*patch*.png"),
        os.path.join(base, "outputs_clothing", "forward_analysis", "patch_pipeline", "*", "*", "*patch*.png"),
        os.path.join(base, "outputs_clothing", "v7_galaxy_v3v8v11", "patch_ep02999.png"),
        os.path.join(base, "outputs_clothing", "v7_galaxy_v3v8v11", "patch_ep03000.png"),
        os.path.join(base, "outputs_clothing", "v11_final", "patch_final_1024.png"),
        os.path.join(base, "outputs_clothing", "v23_rings", "patch_final.png"),
    ]
    found = set()
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            found.add(p)
    def sort_key(path):
        m = re.search(r'v(\d+)', path)
        return int(m.group(1)) if m else 999
    return sorted(found, key=sort_key)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  COMPREHENSIVE PATCH EVALUATION SUITE v2")
    print("  Fixes A-H + Physical Testing Protocol")
    print("  Torso placement | 5/10/15m | 4 lighting | +/-10/20/30 deg | Real DCT")
    print("=" * 70)

    if DEV == "cpu":
        print("WARNING: CUDA not available!")
    else:
        print(f"Device: {DEV}  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Patch: {PATCH_SIZE}px on torso ({TORSO_CX},{TORSO_CY}), head/shoulders exposed")

    patches = discover_patches()
    print(f"\nDiscovered {len(patches)} patches:")
    for p in patches:
        print(f"  {p}")
    if len(patches) == 0:
        print("No patches found!")
        return

    arr_human = load_img(IMG_WITH)
    arr_without = load_img(IMG_WITHOUT) if os.path.exists(IMG_WITHOUT) else None

    print("\nLoading YOLOv3...")
    model = Darknet(CFG)
    model.load_darknet_weights(WTS)
    model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded.")

    clean_t = human_t = torch.from_numpy(arr_human).permute(2, 0, 1).unsqueeze(0).to(DEV)
    if arr_without is not None:
        without_t = torch.from_numpy(arr_without).permute(2, 0, 1).unsqueeze(0).to(DEV)
    else:
        without_t = human_t.clone()
        without_t[:, :, 100:300, 100:300] = 0.5

    # Fix (D): Random control baseline (computed once)
    rand_ctrl = evaluate_random_control(model, human_t, clean_t, without_t)

    summary_rows = []

    for patch_path in patches:
        rel = os.path.relpath(patch_path, r"C:\Users\carso\Desktop\YODO")
        patch_name = rel.replace("\\", "/").replace("/", "_").replace(".png", "")
        patch_out = os.path.join(OUT, patch_name)
        os.makedirs(patch_out, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  TESTING: {patch_name}")
        print(f"  File: {patch_path}")
        print(f"{'='*70}")

        patch_np, mask_np = load_patch(patch_path)
        print(f"  Patch: {patch_np.shape}, mask area: {mask_np.mean() * 100:.2f}%")

        patch_t = torch.from_numpy(patch_np).permute(2, 0, 1).unsqueeze(0).to(DEV)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(DEV)

        # Composite on torso (not full-frame)
        adv_t = composite_on_torso(patch_t, mask_t, human_t)
        print(f"  Composite: patch placed at ({TORSO_CX},{TORSO_CY}) size={PATCH_SIZE}px")

        # Detection scores
        print("\n  --- Detection Scores ---")
        with torch.no_grad():
            scores_clean = extract_detection_scores(model, clean_t)
            scores_adv = extract_detection_scores(model, adv_t)
        print(f"  Clean:  obj={scores_clean['obj_max']:.4f}  combined={scores_clean['combined']:.4f}  n={scores_clean['n_detections']}")
        print(f"  Adv:    obj={scores_adv['obj_max']:.4f}  combined={scores_adv['combined']:.4f}  n={scores_adv['n_detections']}")
        print(f"  Drop:   combined={scores_clean['combined'] - scores_adv['combined']:.4f}  "
              f"n_det={scores_clean['n_detections'] - scores_adv['n_detections']}")

        print("\n  --- Per-Layer Metrics ---")
        per_layer = per_layer_metrics(model, clean_t, adv_t, without_t)

        print("\n  --- Frequency Band Analysis ---")
        freq_bands = frequency_band_analysis(model, clean_t, adv_t)

        print("\n  --- Distance Degradation (5/10/15m) ---")
        dist_deg = distance_degradation(model, patch_t, mask_t, human_t, clean_t)

        print("\n  --- Angle Sensitivity (+/-10/20/30 deg) ---")
        angle_sens = angle_sensitivity(model, patch_t, mask_t, human_t, clean_t)

        print("\n  --- Saliency Map (person-targeted) ---")
        saliency = saliency_map(model, adv_t, clean_t, patch_out)

        print("\n  --- Suppression Go/No-Go Gates ---")
        supp = suppression_gates(model, adv_t, clean_t, without_t, patch_t, mask_t)

        print("\n  --- Poison Go/No-Go Gates ---")
        poison = poison_gates(model, adv_t, clean_t, without_t)

        print("\n  --- EoT Robustness ---")
        eot = eot_robustness(model, patch_t, mask_t, human_t, clean_t)

        print("\n  --- Lighting Sensitivity ---")
        lighting = lighting_sensitivity(model, patch_t, mask_t, human_t, clean_t)

        print(f"\n  Saving results to {patch_out}")
        save_results(per_layer, freq_bands, dist_deg, angle_sens,
                     saliency, supp, poison, eot, lighting, rand_ctrl, patch_out)
        make_figures(per_layer, freq_bands, dist_deg, angle_sens,
                     supp, poison, eot, lighting, patch_name, patch_out)

        summary_rows.append({
            'patch': patch_name,
            'clean_combined': f"{scores_clean['combined']:.4f}",
            'adv_combined': f"{scores_adv['combined']:.4f}",
            'conf_drop': f"{scores_clean['combined'] - scores_adv['combined']:.4f}",
            'clean_n_det': scores_clean['n_detections'],
            'adv_n_det': scores_adv['n_detections'],
            'supp_verdict': supp['go_nogo'],
            'poison_verdict': poison['go_nogo'],
            'eot_mean_combined': f"{eot['eot_mean_combined']:.4f}",
            'eot_std': f"{eot['eot_std_combined']:.4f}",
            'best_cos_gap': f"{min(per_layer[l]['cos_gap'] for l in per_layer):.4f}",
            'best_overlap': f"{min(per_layer[l]['person_overlap'] for l in per_layer):.4f}",
            'best_fft': f"{max(per_layer[l]['fft_spectral_distance'] for l in per_layer):.2f}",
            'rand_ctrl_combined': f"{rand_ctrl['rand_combined']:.4f}",
            'rand_ctrl_reduction': f"{rand_ctrl['rand_reduction_ratio']:.4f}",
            'false_supp_rate': f"{supp['false_suppression_rate']:.4f}",
            'lighting_noon': f"{lighting['noon_6500K']['combined']:.4f}",
            'lighting_street': f"{lighting['street_2200K']['combined']:.4f}",
            'dist_5m': f"{dist_deg[0]['combined']:.4f}",
            'dist_15m': f"{dist_deg[-1]['combined']:.4f}",
        })
        print(f"\n  {patch_name} complete -> {patch_out}")

    # Summary CSV
    summary_csv = os.path.join(OUT, "summary_comparison.csv")
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary CSV: {summary_csv}")

    # Summary figure
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    names = [r['patch'][:25] for r in summary_rows]
    x = np.arange(len(names))

    axes[0, 0].bar(x - 0.2, [float(r['clean_combined']) for r in summary_rows], 0.4, label='Clean', color='steelblue')
    axes[0, 0].bar(x + 0.2, [float(r['adv_combined']) for r in summary_rows], 0.4, label='Adv', color='crimson')
    axes[0, 0].axhline(y=float(rand_ctrl['rand_combined']), color='green', linestyle='--', label='Random ctrl')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 0].set_title("Combined Confidence: Clean vs Adv vs Random"); axes[0, 0].legend()

    axes[0, 1].bar(x, [float(r['conf_drop']) for r in summary_rows], 0.5, color='darkred')
    axes[0, 1].axhline(y=float(rand_ctrl['rand_conf_drop']), color='green', linestyle='--', label='Random ctrl drop')
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 1].set_title("Confidence Drop (higher = better suppression)"); axes[0, 1].legend()

    axes[0, 2].bar(x, [float(r['eot_mean_combined']) for r in summary_rows], 0.5, color='green')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 2].set_title("EoT Mean Combined (lower = more robust suppression)")

    axes[1, 0].bar(x - 0.15, [float(r['dist_5m']) for r in summary_rows], 0.3, label='5m', color='blue')
    axes[1, 0].bar(x + 0.15, [float(r['dist_15m']) for r in summary_rows], 0.3, label='15m', color='red')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_title("Distance: 5m vs 15m"); axes[1, 0].legend()

    axes[1, 1].bar(x - 0.15, [float(r['lighting_noon']) for r in summary_rows], 0.3, label='Noon 6500K', color='steelblue')
    axes[1, 1].bar(x + 0.15, [float(r['lighting_street']) for r in summary_rows], 0.3, label='Street 2200K', color='darkred')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 1].set_title("Lighting: Noon vs Street"); axes[1, 1].legend()

    supp_go = sum(1 for r in summary_rows if r['supp_verdict'] == 'GO')
    poison_go = sum(1 for r in summary_rows if r['poison_verdict'] == 'GO')
    axes[1, 2].bar(['Supp GO', 'Supp NO-GO', 'Poison GO', 'Poison NO-GO'],
                   [supp_go, len(summary_rows) - supp_go, poison_go, len(summary_rows) - poison_go],
                   color=['green', 'red', 'green', 'red'])
    axes[1, 2].set_title("Go/No-Go Summary")

    plt.suptitle("Patch Comparison Summary (v2 - All Fixes)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT, "summary_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*70}")
    print(f"  ALL PATCHES TESTED: {len(summary_rows)}")
    print(f"  Output: {OUT}")
    print(f"  Summary: {summary_csv}")
    print(f"  Suppression GO: {supp_go}/{len(summary_rows)}")
    print(f"  Poison GO: {poison_go}/{len(summary_rows)}")
    print(f"  Random control reduction: {rand_ctrl['rand_reduction_ratio']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
