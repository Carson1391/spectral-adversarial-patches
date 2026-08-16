"""
Standalone patch metrics test script.
Tests a generated adversarial patch against YOLOv3 with full metric suite:

1. Per-layer cosine similarity (GAP + point-level)
2. Per-layer L2 distance (GAP-shifted + point-shifted + raw)
3. FFT spectral distance per layer
4. Person overlap correlation
5. Frequency band analysis (LF/MF/HF) per layer
6. Distance degradation (patch footprint scaling)
7. Angle sensitivity (yaw/pitch foreshortening)
8. Quantization effect (INT8/INT4 PTQ simulation)
9. Saliency map (gradient from person detection score)

Saves: JSON results, CSV metrics, PNG figures.
"""
import os, sys, math, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Layers to analyze — detection heads + key backbone layers
HEAD_LAYERS = {"L81": 81, "L93": 93, "L105": 105}
BACKBONE_LAYERS = {"L54": 54, "L62": 62, "L75": 75}
ALL_LAYERS = {**HEAD_LAYERS, **BACKBONE_LAYERS}

# Output directory
OUT = r"C:\Users\carso\Desktop\YODO\outputs_clothing\final_boss_v2\metrics"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Forward pass — capture all conv layer feature maps
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


def gap_emb(caps, layer_idx):
    # GAP embedding: average pool to 1x1 then flatten
    return F.adaptive_avg_pool2d(caps[layer_idx], 1).squeeze()


def point_emb(caps, layer_idx):
    # Point embedding: flatten spatial dimensions
    return caps[layer_idx].squeeze(0).flatten()


def extract_detection_scores(model, x):
    """Run forward pass and extract YOLOv3 detection scores.
    Returns dict with max objectness, person class prob, combined confidence.
    COCO person class index = 0."""
    person_cls = 0
    los = []
    cur = x
    scores = {'obj_max': 0.0, 'person_prob': 0.0, 'combined': 0.0,
              'n_detections': 0, 'per_head': {}}
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional", "upsample", "maxpool"]:
            cur = mo(cur)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups", 1))
            gi = int(md.get("group_id", 0))
            cur = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            cur = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            cur = mo[0](cur, IS)
            pred = cur
            if pred.dim() == 3:
                obj = pred[..., 4]
                cls_probs = pred[..., 5:]
                person_p = cls_probs[..., person_cls]
                combined = obj * person_p
                mask_conf = combined > 0.25
                n_det = int(mask_conf.sum().item())
                obj_max = float(obj.max().item())
                person_max = float(person_p.max().item())
                combined_max = float(combined.max().item())
                head_name = f"L{i}"
                scores['per_head'][head_name] = {
                    'obj_max': obj_max,
                    'person_prob': person_max,
                    'combined': combined_max,
                    'n_detections': n_det
                }
                scores['obj_max'] = max(scores['obj_max'], obj_max)
                scores['person_prob'] = max(scores['person_prob'], person_max)
                scores['combined'] = max(scores['combined'], combined_max)
                scores['n_detections'] += n_det
        los.append(cur)
    return scores


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
    img = Image.open(patch_path).convert("RGB").resize((IS, IS), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_mask(mask_path):
    img = Image.open(mask_path).convert("L").resize((IS, IS), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ============================================================
# Metric 1-4: Per-layer cosine, L2, FFT, person overlap
# ============================================================
def per_layer_metrics(model, clean_t, adv_t, without_t):
    """Compute all per-layer metrics matching fractal_patch_results.json schema."""
    results = {}

    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        caps_adv, _ = fwd_all(model, adv_t)
        caps_without, _ = fwd_all(model, without_t)

    for lname, lidx in ALL_LAYERS.items():
        shape = caps_clean[lidx].shape[1:]
        label = f"{lname}_{shape[1]}x{shape[2]}"

        # GAP embeddings
        gap_clean = gap_emb(caps_clean, lidx)
        gap_adv = gap_emb(caps_adv, lidx)
        gap_without = gap_emb(caps_without, lidx)

        # Point embeddings (flattened spatial)
        pt_clean = point_emb(caps_clean, lidx)
        pt_adv = point_emb(caps_adv, lidx)
        pt_without = point_emb(caps_without, lidx)

        # Cosine similarity: adv vs clean (how much patch perturbs person signal)
        cos_gap = F.cosine_similarity(gap_clean.unsqueeze(0), gap_adv.unsqueeze(0))[0].item()
        cos_point = F.cosine_similarity(pt_clean.unsqueeze(0), pt_adv.unsqueeze(0))[0].item()

        # L2 of shifted delta: (adv - clean) - (without - clean) = adv - without
        # This isolates patch effect from person absence effect
        l2_shift_gap = torch.norm(gap_adv - gap_without, p=2).item()
        l2_shift_point = torch.norm(pt_adv - pt_without, p=2).item()

        # Raw L2: just |adv - clean|
        raw_l2_gap = torch.norm(gap_adv - gap_clean, p=2).item()

        # Person overlap: cosine between patch delta (adv-clean) and person signal (clean-without)
        # Negative = patch attacks person signal (good for adversarial)
        delta_patch = gap_adv - gap_clean
        delta_person = gap_clean - gap_without
        person_overlap = F.cosine_similarity(delta_patch.unsqueeze(0), delta_person.unsqueeze(0))[0].item()

        # FFT spectral distance: L2 of log-magnitude difference
        f_clean = torch.fft.fft2(caps_clean[lidx].squeeze(0))
        f_adv = torch.fft.fft2(caps_adv[lidx].squeeze(0))
        mag_clean = torch.log(torch.abs(f_clean) + 1)
        mag_adv = torch.log(torch.abs(f_adv) + 1)
        fft_dist = torch.norm(mag_clean - mag_adv, p=2).item()

        results[label] = {
            "cos_gap": cos_gap,
            "cos_point": cos_point,
            "l2_shift_gap": l2_shift_gap,
            "l2_shift_point": l2_shift_point,
            "raw_l2_gap": raw_l2_gap,
            "person_overlap": person_overlap,
            "fft_spectral_distance": fft_dist,
            "shape": list(shape),
        }
        print(f"  {label}: cos_gap={cos_gap:.4f}  cos_point={cos_point:.4f}  "
              f"l2_shift_gap={l2_shift_gap:.2f}  l2_shift_point={l2_shift_point:.2f}  "
              f"raw_l2={raw_l2_gap:.2f}  overlap={person_overlap:.4f}  fft={fft_dist:.2f}")

    return results


# ============================================================
# Metric 5: Frequency band analysis (LF/MF/HF)
# ============================================================
def frequency_band_analysis(model, clean_t, adv_t):
    """Compute LF/MF/HF energy split per layer for clean vs adversarial."""
    results = {}

    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        caps_adv, _ = fwd_all(model, adv_t)

    for lname, lidx in ALL_LAYERS.items():
        shape = caps_clean[lidx].shape[1:]
        H, W = shape[1], shape[2]
        label = f"{lname}_{H}x{W}"

        # FFT magnitude per channel, averaged
        f_clean = torch.fft.fft2(caps_clean[lidx].squeeze(0))  # [C, H, W]
        f_adv = torch.fft.fft2(caps_adv[lidx].squeeze(0))
        mag_clean = torch.abs(torch.fft.fftshift(f_clean, dim=(-2, -1)))
        mag_adv = torch.abs(torch.fft.fftshift(f_adv, dim=(-2, -1)))

        # Radial frequency bins
        cy_f, cx_f = H // 2, W // 2
        y_coords = torch.arange(H, device=DEV, dtype=torch.float32) - cy_f
        x_coords = torch.arange(W, device=DEV, dtype=torch.float32) - cx_f
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        r = torch.sqrt(xx ** 2 + yy ** 2)
        r_max = math.sqrt(cy_f ** 2 + cx_f ** 2)
        r_norm = r / r_max

        # Bands: LF 0-25%, MF 25-50%, HF 50-100%
        lf_mask = r_norm <= 0.25
        mf_mask = (r_norm > 0.25) & (r_norm <= 0.50)
        hf_mask = r_norm > 0.50

        # Energy per band (sum across channels)
        energy_clean = mag_clean.sum(dim=0)  # [H, W]
        energy_adv = mag_adv.sum(dim=0)
        total_clean = energy_clean.sum() + 1e-10
        total_adv = energy_adv.sum() + 1e-10

        lf_clean = (energy_clean * lf_mask).sum() / total_clean
        mf_clean = (energy_clean * mf_mask).sum() / total_clean
        hf_clean = (energy_clean * hf_mask).sum() / total_clean
        lf_adv = (energy_adv * lf_mask).sum() / total_adv
        mf_adv = (energy_adv * mf_mask).sum() / total_adv
        hf_adv = (energy_adv * hf_mask).sum() / total_adv

        delta_lf = (lf_adv - lf_clean).item()
        delta_mf = (mf_adv - mf_clean).item()
        delta_hf = (hf_adv - hf_clean).item()

        results[label] = {
            "lf_clean": lf_clean.item(),
            "mf_clean": mf_clean.item(),
            "hf_clean": hf_clean.item(),
            "lf_adv": lf_adv.item(),
            "mf_adv": mf_adv.item(),
            "hf_adv": hf_adv.item(),
            "delta_lf": delta_lf,
            "delta_mf": delta_mf,
            "delta_hf": delta_hf,
        }
        print(f"  {label}: LF {lf_clean:.3f}→{lf_adv:.3f} (d={delta_lf:+.4f})  "
              f"MF {mf_clean:.3f}→{mf_adv:.3f} (d={delta_mf:+.4f})  "
              f"HF {hf_clean:.3f}→{hf_adv:.3f} (d={delta_hf:+.4f})")

    return results


# ============================================================
# Metric 6: Distance degradation
# ============================================================
def distance_degradation(model, patch_t, mask_t, human_t, clean_t, outer_size, cx, cy):
    """Simulate patch at different distances by scaling patch footprint."""
    distances = [5, 10, 20, 30, 40, 50, 60]
    results = []

    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)

    for d in distances:
        scale = 5.0 / d  # relative to 5m baseline
        new_size = max(1, int(outer_size * scale))
        if new_size < 4:
            results.append({"distance_m": d, "patch_px": new_size, "cos": 0.5})
            print(f"  {d}m: cos=0.5000  patch_px={new_size} (too small)")
            continue

        p_small = F.interpolate(patch_t, size=(new_size, new_size), mode='bilinear', align_corners=False)
        m_small = F.interpolate(mask_t, size=(new_size, new_size), mode='bilinear', align_corners=False)

        comp_d = human_t.clone()
        x0 = cx - new_size // 2
        y0 = cy - new_size // 2
        x1 = min(x0 + new_size, IS)
        y1 = min(y0 + new_size, IS)
        x0 = max(0, x0); y0 = max(0, y0)
        aw = x1 - x0; ah = y1 - y0
        if aw > 0 and ah > 0:
            comp_d[:, :, y0:y1, x0:x1] = (
                p_small[:, :, :ah, :aw] * m_small[:, :, :ah, :aw] +
                human_t[:, :, y0:y1, x0:x1] * (1 - m_small[:, :, :ah, :aw])
            )

        with torch.no_grad():
            caps_d, _ = fwd_all(model, comp_d)
            gap_d = gap_emb(caps_d, 81)
            cd = F.cosine_similarity(gap_c.unsqueeze(0), gap_d.unsqueeze(0))[0].item()
        results.append({"distance_m": d, "patch_px": new_size, "cos": cd})
        print(f"  {d}m: cos={cd:.4f}  patch_px={new_size}")

    return results


# ============================================================
# Metric 7: Angle sensitivity (yaw + pitch)
# ============================================================
def angle_sensitivity(model, patch_t, mask_t, human_t, clean_t):
    """Simulate yaw and pitch by affine foreshortening."""
    yaw_angles = [0, 15, 30, 45, 60, 75, 90]
    pitch_angles = [0, 5, 10, 15, 20, 25, 30]
    yaw_results = []
    pitch_results = []

    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)

    for yaw in yaw_angles:
        if yaw == 0:
            # Baseline: full composite
            comp = patch_t * mask_t + human_t * (1 - mask_t)
            with torch.no_grad():
                caps_y, _ = fwd_all(model, comp)
                gap_y = gap_emb(caps_y, 81)
                cy = F.cosine_similarity(gap_c.unsqueeze(0), gap_y.unsqueeze(0))[0].item()
            yaw_results.append({"yaw_deg": yaw, "cos": cy})
            print(f"  yaw={yaw}deg: cos={cy:.4f}")
            continue
        sx = math.cos(math.radians(yaw))
        warped = F.interpolate(patch_t, size=(IS, max(1, int(IS * sx))), mode='bilinear', align_corners=False)
        if warped.shape[-1] < IS:
            pad = (IS - warped.shape[-1]) // 2
            warped = F.pad(warped, (pad, IS - warped.shape[-1] - pad, 0, 0))
        comp_y = warped * mask_t + human_t * (1 - mask_t)
        with torch.no_grad():
            caps_y, _ = fwd_all(model, comp_y)
            gap_y = gap_emb(caps_y, 81)
            cy = F.cosine_similarity(gap_c.unsqueeze(0), gap_y.unsqueeze(0))[0].item()
        yaw_results.append({"yaw_deg": yaw, "cos": cy})
        print(f"  yaw={yaw}deg: cos={cy:.4f}")

    for pitch in pitch_angles:
        if pitch == 0:
            pitch_results.append({"pitch_deg": pitch, "cos": yaw_results[0]["cos"]})
            print(f"  pitch={pitch}deg: cos={yaw_results[0]['cos']:.4f}")
            continue
        sy = math.cos(math.radians(pitch))
        warped = F.interpolate(patch_t, size=(max(1, int(IS * sy)), IS), mode='bilinear', align_corners=False)
        if warped.shape[-2] < IS:
            pad = (IS - warped.shape[-2]) // 2
            warped = F.pad(warped, (0, 0, pad, IS - warped.shape[-2] - pad))
        comp_p = warped * mask_t + human_t * (1 - mask_t)
        with torch.no_grad():
            caps_p, _ = fwd_all(model, comp_p)
            gap_p = gap_emb(caps_p, 81)
            cp = F.cosine_similarity(gap_c.unsqueeze(0), gap_p.unsqueeze(0))[0].item()
        pitch_results.append({"pitch_deg": pitch, "cos": cp})
        print(f"  pitch={pitch}deg: cos={cp:.4f}")

    return {"yaw": yaw_results, "pitch": pitch_results}


# ============================================================
# Metric 8: Quantization effect (INT8/INT4 PTQ simulation)
# ============================================================
def quantization_effect(model, adv_t, clean_t):
    """Simulate INT8 and INT4 post-training quantization by weight clipping."""
    quant_levels = ['FP32', 'INT8', 'INT4']
    results = []

    # Save original weights
    orig_weights = {}
    for name, param in model.named_parameters():
        orig_weights[name] = param.data.clone()

    with torch.no_grad():
        caps_clean, _ = fwd_all(model, clean_t)
        gap_c = gap_emb(caps_clean, 81)

    for qlevel in quant_levels:
        if qlevel == 'FP32':
            for name, param in model.named_parameters():
                param.data = orig_weights[name].clone()
        elif qlevel == 'INT8':
            # INT8: scale to [-128, 127], round, scale back
            for name, param in model.named_parameters():
                w = orig_weights[name]
                w_max = w.abs().max().clamp(min=1e-8)
                scale = 127.0 / w_max
                w_q = torch.round(w * scale) / scale
                param.data = w_q
        elif qlevel == 'INT4':
            # INT4: scale to [-7, 7], round, scale back
            for name, param in model.named_parameters():
                w = orig_weights[name]
                w_max = w.abs().max().clamp(min=1e-8)
                scale = 7.0 / w_max
                w_q = torch.round(w * scale) / scale
                param.data = w_q

        with torch.no_grad():
            caps_q, _ = fwd_all(model, adv_t)
            gap_q = gap_emb(caps_q, 81)
            cq = F.cosine_similarity(gap_c.unsqueeze(0), gap_q.unsqueeze(0))[0].item()
        results.append({"precision": qlevel, "cos": cq})
        print(f"  {qlevel}: cos={cq:.4f}")

    # Restore original weights
    for name, param in model.named_parameters():
        param.data = orig_weights[name].clone()

    return results


# ============================================================
# Metric 9: Saliency map (gradient from person detection score)
# ============================================================
def saliency_map(model, adv_t, clean_t):
    """Compute gradient saliency: which input pixels drive person detection."""
    results = {}

    # Person class index in COCO = 0
    # YOLOv3 output: [B, num_anchors * (5+80), H, W] per scale
    # We use the detection output and find person score

    clean_t.requires_grad_(True)
    model.train()  # Need gradients but no BN updates
    for m in model.modules():
        if hasattr(m, 'momentum'):
            m.momentum = 0

    # Forward clean
    _, out_clean = fwd_all(model, clean_t)
    # out_clean is the final YOLO output — find person score
    # Use max activation across all outputs as proxy for person detection score
    person_score_clean = out_clean.max()

    # Gradient w.r.t. input
    grad_clean = torch.autograd.grad(person_score_clean, clean_t, retain_graph=False)[0]
    saliency_clean = grad_clean.abs().sum(dim=1).squeeze(0)

    # Forward adversarial
    adv_t.requires_grad_(True)
    _, out_adv = fwd_all(model, adv_t)
    person_score_adv = out_adv.max()
    grad_adv = torch.autograd.grad(person_score_adv, adv_t, retain_graph=False)[0]
    saliency_adv = grad_adv.abs().sum(dim=1).squeeze(0)

    # Delta saliency: where does patch change the gradient?
    delta_saliency = saliency_adv - saliency_clean

    model.eval()
    clean_t.requires_grad_(False)
    adv_t.requires_grad_(False)

    results["person_score_clean"] = person_score_clean.item()
    results["person_score_adv"] = person_score_adv.item()
    results["score_drop"] = (person_score_clean - person_score_adv).item()

    # Save saliency maps
    sal_clean_np = saliency_clean.detach().cpu().numpy()
    sal_adv_np = saliency_adv.detach().cpu().numpy()
    sal_delta_np = delta_saliency.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(sal_clean_np, cmap='hot')
    axes[0].set_title(f"Saliency (clean)\nscore={person_score_clean:.4f}")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sal_adv_np, cmap='hot')
    axes[1].set_title(f"Saliency (adversarial)\nscore={person_score_adv:.4f}")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(sal_delta_np, cmap='RdBu_r', vmin=-np.abs(sal_delta_np).max(),
                         vmax=np.abs(sal_delta_np).max())
    axes[2].set_title(f"Delta saliency\nscore_drop={results['score_drop']:.4f}")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle("Gradient Saliency: Where the patch redirects person detection", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  person_score_clean={person_score_clean:.4f}  person_score_adv={person_score_adv:.4f}  "
          f"drop={results['score_drop']:.4f}")

    return results


# ============================================================
# Visualization
# ============================================================
def make_figures(per_layer, freq_bands, dist_deg, angle_sens, quant_eff, saliency, out_dir=OUT):
    """Generate comprehensive analysis figures."""

    # Figure 1: 6-panel forward analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

    layers = list(per_layer.keys())
    x_pos = np.arange(len(layers))

    # Row 0: cos_gap + cos_point
    axes[0, 0].bar(x_pos - 0.2, [per_layer[l]["cos_gap"] for l in layers], 0.4, label='cos_gap', color='steelblue')
    axes[0, 0].bar(x_pos + 0.2, [per_layer[l]["cos_point"] for l in layers], 0.4, label='cos_point', color='orange')
    axes[0, 0].set_xticks(x_pos); axes[0, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 0].set_title("Cosine Similarity: GAP vs Point Embedding")
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("cos(adv, clean)")
    axes[0, 0].set_ylim(0.9, 1.001)

    # Row 0 right: person overlap
    axes[0, 1].bar(x_pos, [per_layer[l]["person_overlap"] for l in layers], 0.5, color='crimson')
    axes[0, 1].set_xticks(x_pos); axes[0, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 1].set_title("Person Overlap: Negative = patch attacks person signal")
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[0, 1].set_ylabel("cos(delta_patch, delta_person)")

    # Row 1: L2 metrics
    axes[1, 0].bar(x_pos - 0.25, [per_layer[l]["l2_shift_gap"] for l in layers], 0.25, label='L2 shift GAP', color='orange')
    axes[1, 0].bar(x_pos, [per_layer[l]["l2_shift_point"] for l in layers], 0.25, label='L2 shift point', color='red')
    axes[1, 0].bar(x_pos + 0.25, [per_layer[l]["raw_l2_gap"] for l in layers], 0.25, label='Raw L2 GAP', color='darkred')
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 0].set_title("L2 Distance Metrics per Layer")
    axes[1, 0].legend()
    axes[1, 0].set_ylabel("L2 norm")

    # Row 1 right: FFT spectral distance
    axes[1, 1].bar(x_pos, [per_layer[l]["fft_spectral_distance"] for l in layers], 0.5, color='purple')
    axes[1, 1].set_xticks(x_pos); axes[1, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 1].set_title("FFT Spectral Distance per Layer")
    axes[1, 1].set_ylabel("L2(log|FFT(adv)| - log|FFT(clean)|)")

    # Row 2: Distance degradation + Angle sensitivity
    dists = [d["distance_m"] for d in dist_deg]
    dist_cos = [d["cos"] for d in dist_deg]
    axes[2, 0].plot(dists, dist_cos, 'o-', color='crimson', linewidth=2, markersize=8)
    axes[2, 0].set_xlabel("Distance (m)")
    axes[2, 0].set_ylabel("Cosine Similarity (L81)")
    axes[2, 0].set_title("Distance Degradation: patch footprint shrinks below effective threshold")
    axes[2, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='ASR ~50% threshold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    yaws = [y["yaw_deg"] for y in angle_sens["yaw"]]
    yaw_cos = [y["cos"] for y in angle_sens["yaw"]]
    pitches = [p["pitch_deg"] for p in angle_sens["pitch"]]
    pitch_cos = [p["cos"] for p in angle_sens["pitch"]]
    axes[2, 1].plot(yaws, yaw_cos, 's-', color='darkorange', linewidth=2, markersize=8, label='Yaw')
    axes[2, 1].plot(pitches, pitch_cos, '^-', color='purple', linewidth=2, markersize=8, label='Pitch')
    axes[2, 1].set_xlabel("Angle (degrees)")
    axes[2, 1].set_ylabel("Cosine Similarity (L81)")
    axes[2, 1].set_title("Angle Sensitivity: foreshortening breaks alignment")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle("Final Boss v2 — Comprehensive Forward Analysis\n"
                 "(Computational benchmarks, not real-world metrics)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{out_dir}/forward_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Frequency band analysis
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for i, (label, fb) in enumerate(freq_bands.items()):
        ax = axes2[i // 3, i % 3]
        bands = ['LF', 'MF', 'HF']
        clean_vals = [fb["lf_clean"], fb["mf_clean"], fb["hf_clean"]]
        adv_vals = [fb["lf_adv"], fb["mf_adv"], fb["hf_adv"]]
        x = np.arange(3)
        ax.bar(x - 0.2, clean_vals, 0.4, label='Clean', color='steelblue')
        ax.bar(x + 0.2, adv_vals, 0.4, label='Adversarial', color='crimson')
        ax.set_xticks(x); ax.set_xticklabels(bands)
        ax.set_title(f"{label}\ndLF={fb['delta_lf']:+.4f} dMF={fb['delta_mf']:+.4f} dHF={fb['delta_hf']:+.4f}")
        ax.set_ylabel("Fraction of spectral energy")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
    plt.suptitle("Frequency Band Analysis: LF/MF/HF energy split per layer\n"
                 "Patch shifts spectral energy distribution", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{out_dir}/frequency_bands.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: Quantization effect
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    qlevels = [q["precision"] for q in quant_eff]
    qcos = [q["cos"] for q in quant_eff]
    bars = ax3.bar(qlevels, qcos, color=['steelblue', 'forestgreen', 'crimson'], width=0.5)
    ax3.set_ylabel("Cosine Similarity (L81)")
    ax3.set_title("Quantization Effect: INT8/INT4 PTQ simulation\n(lower cos = stronger attack effect)")
    ax3.set_ylim(0.95, 1.001)
    for bar, val in zip(bars, qcos):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/quantization_effect.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: Person overlap + score comparison
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
    scores = ['Clean', 'Adversarial']
    vals = [saliency["person_score_clean"], saliency["person_score_adv"]]
    bars4 = ax4.bar(scores, vals, color=['steelblue', 'crimson'], width=0.5)
    ax4.set_ylabel("Max detection score")
    ax4.set_title(f"Person Detection Score: Clean vs Adversarial\nDrop = {saliency['score_drop']:.4f}")
    for bar, val in zip(bars4, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/detection_score.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Save results
# ============================================================
def save_results(per_layer, freq_bands, dist_deg, angle_sens, quant_eff, saliency, out_dir=OUT):
    # JSON
    json_path = f"{out_dir}/patch_metrics.json"
    all_results = {
        "per_layer_metrics": per_layer,
        "frequency_band_analysis": freq_bands,
        "distance_degradation": dist_deg,
        "angle_sensitivity": angle_sens,
        "quantization_effect": quant_eff,
        "saliency": saliency,
    }
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # CSV
    csv_path = f"{out_dir}/patch_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric_type', 'param', 'value'])
        for label, m in per_layer.items():
            for k, v in m.items():
                if k != 'shape':
                    w.writerow(['per_layer', f"{label}_{k}", f"{v:.6f}"])
        for label, fb in freq_bands.items():
            for k, v in fb.items():
                w.writerow(['freq_band', f"{label}_{k}", f"{v:.6f}"])
        for d in dist_deg:
            w.writerow(['distance_degradation', f"{d['distance_m']}m", f"{d['cos']:.6f}"])
        for y in angle_sens["yaw"]:
            w.writerow(['angle_yaw', f"{y['yaw_deg']}deg", f"{y['cos']:.6f}"])
        for p in angle_sens["pitch"]:
            w.writerow(['angle_pitch', f"{p['pitch_deg']}deg", f"{p['cos']:.6f}"])
        for q in quant_eff:
            w.writerow(['quantization', q['precision'], f"{q['cos']:.6f}"])
        w.writerow(['saliency', 'person_score_clean', f"{saliency['person_score_clean']:.6f}"])
        w.writerow(['saliency', 'person_score_adv', f"{saliency['person_score_adv']:.6f}"])
        w.writerow(['saliency', 'score_drop', f"{saliency['score_drop']:.6f}"])
    print(f"CSV saved: {csv_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  PATCH METRICS TEST SUITE")
    print("  Testing adversarial patch against YOLOv3 with full metric suite")
    print("=" * 70)

    # Verify CUDA
    if DEV == "cpu":
        print("WARNING: CUDA not available. Running on CPU.")
    else:
        print(f"Device: {DEV}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the patch image the user provided (screenshot from fractal_patch dir)
    patch_files = [
        ("screenshot_171959", r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\fractal_patch\Screenshot 2026-07-07 171959.png"),
    ]

    # Load images
    arr_human = load_img(IMG_WITH)
    arr_without = load_img(IMG_WITHOUT) if os.path.exists(IMG_WITHOUT) else None

    # Load model
    print("\nLoading YOLOv3...")
    model = Darknet(CFG)
    model.load_darknet_weights(WTS)
    model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded.")

    clean_t = human_t = torch.from_numpy(arr_human).permute(2, 0, 1).unsqueeze(0).to(DEV)

    # Without-human tensor
    if arr_without is not None:
        without_t = torch.from_numpy(arr_without).permute(2, 0, 1).unsqueeze(0).to(DEV)
    else:
        without_t = human_t.clone()
        without_t[:, :, 100:300, 100:300] = 0.5

    # Test each fractal patch
    for patch_name, patch_path in patch_files:
        if not os.path.exists(patch_path):
            print(f"SKIP: {patch_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  TESTING: {patch_name}")
        print(f"  File: {patch_path}")
        print(f"{'='*70}")

        patch_np = load_patch(patch_path)
        # Derive mask from patch: any non-black pixel = mask
        mask_np = (patch_np.mean(axis=2) > 0.02).astype(np.float32)
        print(f"  Patch: {patch_np.shape}, mask area: {mask_np.mean() * 100:.2f}%")

        patch_t = torch.from_numpy(patch_np).permute(2, 0, 1).unsqueeze(0).to(DEV)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(DEV)

        # Composite: patch applied to human image
        adv_t = patch_t * mask_t + human_t * (1.0 - mask_t)

        # Per-patch output dir
        patch_out = os.path.join(OUT, patch_name)
        os.makedirs(patch_out, exist_ok=True)

        # ================================================================
        # Detection Scores (objectness + person class probability)
        # ================================================================
        print("\n" + "=" * 50)
        print("  Detection Scores (YOLOv3 output)")
        print("=" * 50)
        with torch.no_grad():
            scores_clean = extract_detection_scores(model, clean_t)
            scores_adv = extract_detection_scores(model, adv_t)
        print(f"  Clean:  obj={scores_clean['obj_max']:.4f}  person_prob={scores_clean['person_prob']:.4f}  "
              f"combined={scores_clean['combined']:.4f}  n_det={scores_clean['n_detections']}")
        print(f"  Adv:    obj={scores_adv['obj_max']:.4f}  person_prob={scores_adv['person_prob']:.4f}  "
              f"combined={scores_adv['combined']:.4f}  n_det={scores_adv['n_detections']}")
        print(f"  Drop:   obj={scores_clean['obj_max'] - scores_adv['obj_max']:.4f}  "
              f"person={scores_clean['person_prob'] - scores_adv['person_prob']:.4f}  "
              f"combined={scores_clean['combined'] - scores_adv['combined']:.4f}  "
              f"n_det_drop={scores_clean['n_detections'] - scores_adv['n_detections']}")
        for hn in scores_clean['per_head']:
            sc = scores_clean['per_head'][hn]
            sa = scores_adv['per_head'][hn]
            print(f"    {hn}: clean(obj={sc['obj_max']:.3f}, person={sc['person_prob']:.3f}, "
                  f"combined={sc['combined']:.3f}, n={sc['n_detections']}) -> "
                  f"adv(obj={sa['obj_max']:.3f}, person={sa['person_prob']:.3f}, "
                  f"combined={sa['combined']:.3f}, n={sa['n_detections']})")

        # ================================================================
        # Metric 1-4: Per-layer cosine, L2, FFT, person overlap
        # ================================================================
        print("\n" + "=" * 50)
        print("  Per-Layer Metrics (cosine, L2, FFT, person overlap)")
        print("=" * 50)
        per_layer = per_layer_metrics(model, clean_t, adv_t, without_t)

        # ================================================================
        # Metric 5: Frequency band analysis
        # ================================================================
        print("\n" + "=" * 50)
        print("  Frequency Band Analysis (LF/MF/HF)")
        print("=" * 50)
        freq_bands = frequency_band_analysis(model, clean_t, adv_t)

        # ================================================================
        # Metric 6: Distance degradation
        # ================================================================
        print("\n" + "=" * 50)
        print("  Distance Degradation")
        print("=" * 50)
        outer_size = 380
        cx, cy = IS // 2, IS // 2
        dist_deg = distance_degradation(model, patch_t, mask_t, human_t, clean_t, outer_size, cx, cy)

        # ================================================================
        # Metric 7: Angle sensitivity
        # ================================================================
        print("\n" + "=" * 50)
        print("  Angle Sensitivity (yaw + pitch)")
        print("=" * 50)
        angle_sens = angle_sensitivity(model, patch_t, mask_t, human_t, clean_t)

        # ================================================================
        # Metric 8: Quantization effect
        # ================================================================
        print("\n" + "=" * 50)
        print("  Quantization Effect (INT8/INT4 PTQ)")
        print("=" * 50)
        quant_eff = quantization_effect(model, adv_t, clean_t)

        # ================================================================
        # Metric 9: Saliency map
        # ================================================================
        print("\n" + "=" * 50)
        print("  Saliency Map (gradient from person detection score)")
        print("=" * 50)
        saliency = saliency_map(model, adv_t, clean_t)

        # ================================================================
        # Save results (per-patch)
        # ================================================================
        print("\n" + "=" * 50)
        print(f"  Saving results to {patch_out}")
        print("=" * 50)
        save_results(per_layer, freq_bands, dist_deg, angle_sens, quant_eff, saliency, out_dir=patch_out)

        # ================================================================
        # Generate figures (per-patch)
        # ================================================================
        print("Generating figures...")
        make_figures(per_layer, freq_bands, dist_deg, angle_sens, quant_eff, saliency, out_dir=patch_out)

        print(f"\n  {patch_name} complete -> {patch_out}")

    print(f"\n{'=' * 70}")
    print("  ALL PATCHES TESTED")
    print(f"  Output: {OUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
