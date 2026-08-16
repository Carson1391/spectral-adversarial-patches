#!/usr/bin/env python3
"""
Per-Layer Human/Meter Dimension Extraction for YOLOv3u.

For each backbone + FPN layer, extract feature embeddings from real person
images and real parking-meter images. Classify each channel dimension as:
  - HUMAN-ONLY:  activated by persons, not meters  → suppress (drive negative)
  - METER-ONLY:  activated by meters, not persons  → amplify (boost positive)
  - SHARED:      activated by both                 → boost toward meter (hold box)
  - OTHER:       neither or noise                  → leave alone

Also computes the classification head weights (cv3[2].weight at each scale)
to identify which feature dims matter for person vs meter class scores.

Energy conservation: sum of |amplify| = sum of |suppress| so total L2 norm
of the feature embedding stays constant when the patch is applied.
"""
import os, sys, json, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

DEVICE = torch.device('cuda')
PERSON = 0
METER = 12  # COCO class 12 = parking meter

# Layers to probe — backbone blocks + FPN outputs feeding Detect
# These are the feature maps with semantic meaning
PROBE_LAYERS = [4, 6, 8, 10, 11, 12, 14, 16, 19, 20, 22, 23, 26, 27]
# L4  = backbone res2  (128ch, 160px) — low-level
# L6  = backbone res3  (256ch, 80px)  — early semantic
# L8  = backbone res4  (512ch, 40px)  — mid semantic
# L10 = backbone res5  (1024ch, 20px) — high semantic
# L11 = SPP bottleneck (1024ch, 20px)
# L12 = FPN reduce     (512ch, 20px)  → feeds Detect scale-2 (stride 32)
# L14 = FPN reduce2    (512ch, 20px)
# L16 = FPN lateral    (256ch, 20px)  → upsampled
# L19 = FPN merge1     (512ch, 40px)  → feeds Detect scale-1 (stride 16)
# L20 = FPN bottleneck (512ch, 40px)
# L22 = FPN reduce3    (512ch, 40px)
# L23 = FPN lateral2   (128ch, 40px)  → upsampled
# L26 = FPN merge2     (256ch, 80px)  → feeds Detect scale-0 (stride 8)
# L27 = FPN bottleneck2(256ch, 80px)


def load_images(path, max_n=50, size=640):
    """Load images, resize to 640x640, return tensor (N, 3, 640, 640)."""
    if not os.path.exists(path):
        return None
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:max_n]
    if not files:
        return None
    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert('RGB').resize((size, size))
        imgs.append(T.ToTensor()(img))
    return torch.stack(imgs).to(DEVICE)


def extract_layer_features(model, imgs, layer_indices):
    """Forward pass with hooks, return {layer_idx: (N, C, H, W)}."""
    feats = {}
    hooks = []
    
    def make_hook(idx):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                feats[idx] = out.detach()
        return hook

    for i in layer_indices:
        hooks.append(model.model[i].register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(imgs)
    
    for h in hooks:
        h.remove()
    
    return feats


def classify_dims(human_feats, meter_feats, threshold=0.3):
    """
    Given (N_h, C, H, W) person features and (N_m, C, H, W) meter features,
    classify each of the C channels.
    
    For each channel c:
      human_activation = mean over all person images and spatial locations of |feat|
      meter_activation = same for meter images
      
      human_score = human_activation / (human_activation + meter_activation + eps)
      meter_score = meter_activation / (human_activation + meter_activation + eps)
    
    Classification:
      If human_score > 1-threshold and meter_score < threshold → HUMAN-ONLY
      If meter_score > 1-threshold and human_score < threshold → METER-ONLY
      If both > threshold → SHARED
      Else → OTHER
    """
    # Global average pool + absolute value
    human_gap = human_feats.abs().mean(dim=(0, 2, 3))   # (C,)
    meter_gap = meter_feats.abs().mean(dim=(0, 2, 3))   # (C,)
    
    total = human_gap + meter_gap + 1e-8
    human_score = human_gap / total   # (C,) in [0,1]
    meter_score = meter_gap / total   # (C,) in [0,1]
    
    human_only = (human_score > 1 - threshold) & (meter_score < threshold)
    meter_only = (meter_score > 1 - threshold) & (human_score < threshold)
    shared = (human_score > threshold) & (meter_score > threshold)
    other = ~human_only & ~meter_only & ~shared
    
    return {
        'human_only': human_only.cpu().numpy(),
        'meter_only': meter_only.cpu().numpy(),
        'shared': shared.cpu().numpy(),
        'other': other.cpu().numpy(),
        'human_score': human_score.cpu().numpy(),
        'meter_score': meter_score.cpu().numpy(),
        'human_gap': human_gap.cpu().numpy(),
        'meter_gap': meter_gap.cpu().numpy(),
        'dim': human_feats.shape[1],
    }


def classify_head_weights(model):
    """
    Extract classification head weights at each scale.
    cv3[s][2] is the final 1x1 conv: weight shape (80, 256, 1, 1).
    
    For person (class 0) and meter (class 12), find which of the 256
    feature dims contribute most to each class score, and classify:
      - If w_person[c] is large and w_meter[c] is small → HUMAN dim
      - If w_meter[c] is large and w_person[c] is small → METER dim
      - If both are large → SHARED dim
    """
    det = model.model[28]
    results = {}
    
    for s in range(3):
        w = det.cv3[s][2].weight  # (80, 256, 1, 1)
        w = w.squeeze(-1).squeeze(-1)  # (80, 256)
        
        w_person = w[PERSON]   # (256,)
        w_meter = w[METER]     # (256,)
        
        # Normalize
        p_abs = w_person.abs()
        m_abs = w_meter.abs()
        total = p_abs + m_abs + 1e-8
        
        p_score = p_abs / total
        m_score = m_abs / total
        
        threshold = 0.3
        human_only = (p_score > 1 - threshold) & (m_score < threshold)
        meter_only = (m_score > 1 - threshold) & (p_score < threshold)
        shared = (p_score > threshold) & (m_score > threshold)
        other = ~human_only & ~meter_only & ~shared
        
        results[f'scale_{s}'] = {
            'dim': 256,
            'human_only': human_only.cpu().numpy(),
            'meter_only': meter_only.cpu().numpy(),
            'shared': shared.cpu().numpy(),
            'other': other.cpu().numpy(),
            'w_person': w_person.cpu().numpy(),
            'w_meter': w_meter.cpu().numpy(),
            'cos_sim': float(F.cosine_similarity(w_person.unsqueeze(0), w_meter.unsqueeze(0)).item()),
        }
    
    return results


def compute_energy_balanced_targets(dim_info, eps_human=1.0, eps_meter=1.0, eps_shared=0.5):
    """
    Compute per-dim adjustment that conserves total L2 energy.
    
    Δ[c] = -eps_human  if HUMAN-ONLY  (suppress)
         = +eps_meter  if METER-ONLY  (amplify)
         = +eps_shared if SHARED       (boost toward meter, hold box)
         = 0           if OTHER
    
    Energy conservation: scale all positive terms so that
        sum(positive_deltas) = sum(negative_deltas)
    This keeps ||x + Δ|| ≈ ||x||.
    """
    dim = dim_info['dim']
    delta = np.zeros(dim, dtype=np.float32)
    
    h_mask = dim_info['human_only']
    m_mask = dim_info['meter_only']
    s_mask = dim_info['shared']
    
    delta[h_mask] = -eps_human
    delta[m_mask] = +eps_meter
    delta[s_mask] = +eps_shared
    
    # Energy conservation: scale positives to match negatives
    neg_energy = np.abs(delta[delta < 0]).sum()
    pos_energy = delta[delta > 0].sum()
    
    if pos_energy > 0 and neg_energy > 0:
        scale = neg_energy / pos_energy
        delta[delta > 0] *= scale
    
    return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_imgs', default='data/coco_person_strong/images',
                        help='Path to person images')
    parser.add_argument('--meter_imgs', default='data/coco_meter/images',
                        help='Path to parking meter images (will try to download if missing)')
    parser.add_argument('--model', default='YOLOv3/yolov3u.pt')
    parser.add_argument('--out', default='layer_dims.json')
    parser.add_argument('--max_imgs', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Dim classification threshold (lower = stricter)')
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.abspath(__file__))
    
    print(f'Loading YOLOv3u from {args.model}...')
    yolo = YOLO(os.path.join(base, args.model))
    model = yolo.model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Load images
    person_path = os.path.join(base, args.person_imgs)
    meter_path = os.path.join(base, args.meter_imgs)
    
    print(f'Loading person images from {person_path}...')
    person_imgs = load_images(person_path, args.max_imgs)
    
    if person_imgs is None:
        print('WARNING: No person images found. Using synthetic person-like patches.')
        # Create synthetic: use the model's own person archetype from gradient ascent
        person_imgs = synthesize_person_images(model, n=20)
    
    print(f'Loading meter images from {meter_path}...')
    meter_imgs = load_images(meter_path, args.max_imgs)
    
    if meter_imgs is None:
        print('WARNING: No meter images found. Using synthetic meter archetype.')
        meter_imgs = synthesize_meter_images(model, n=20)
    
    print(f'Person images: {person_imgs.shape[0]}')
    print(f'Meter images: {meter_imgs.shape[0]}')
    
    # Extract features at each layer
    print(f'\n=== Extracting features at layers {PROBE_LAYERS} ===')
    person_feats = extract_layer_features(model, person_imgs, PROBE_LAYERS)
    meter_feats = extract_layer_features(model, meter_imgs, PROBE_LAYERS)
    
    # Classify dims at each layer
    results = {}
    print(f'\n=== Per-layer dimension classification ===')
    print(f'{"Layer":>5} {"Dim":>5} {"Human":>6} {"Meter":>6} {"Shared":>6} {"Other":>6}  Type')
    print('-' * 60)
    
    for idx in PROBE_LAYERS:
        if idx not in person_feats or idx not in meter_feats:
            print(f'  L{idx:2d}: MISSING')
            continue
        
        pf = person_feats[idx]
        mf = meter_feats[idx]
        
        info = classify_dims(pf, mf, threshold=args.threshold)
        results[f'L{idx}'] = info
        
        n_h = info['human_only'].sum()
        n_m = info['meter_only'].sum()
        n_s = info['shared'].sum()
        n_o = info['other'].sum()
        
        # Layer type
        layer_type = type(model.model[idx]).__name__
        
        print(f'L{idx:2d}   {info["dim"]:5d} {n_h:6d} {n_m:6d} {n_s:6d} {n_o:6d}  {layer_type}')
        
        # Show top human and meter dims
        h_dims = np.where(info['human_only'])[0]
        m_dims = np.where(info['meter_only'])[0]
        s_dims = np.where(info['shared'])[0]
        
        if len(h_dims) > 0:
            h_scores = info['human_score'][h_dims]
            top_h = h_dims[np.argsort(-h_scores)][:5]
            print(f'         Human-only top: {top_h.tolist()}')
        if len(m_dims) > 0:
            m_scores = info['meter_score'][m_dims]
            top_m = m_dims[np.argsort(-m_scores)][:5]
            print(f'         Meter-only top: {top_m.tolist()}')
        if len(s_dims) > 0:
            # Shared with strongest activation
            s_act = info['human_gap'][s_dims] + info['meter_gap'][s_dims]
            top_s = s_dims[np.argsort(-s_act)][:5]
            print(f'         Shared top:     {top_s.tolist()}')
        
        # Compute energy-balanced target
        delta = compute_energy_balanced_targets(info)
        results[f'L{idx}']['energy_delta'] = delta.tolist()
        results[f'L{idx}']['layer_type'] = layer_type
    
    # Classification head weights
    print(f'\n=== Classification head (cv3) weight analysis ===')
    head_info = classify_head_weights(model)
    for s in range(3):
        info = head_info[f'scale_{s}']
        n_h = info['human_only'].sum()
        n_m = info['meter_only'].sum()
        n_s = info['shared'].sum()
        n_o = info['other'].sum()
        print(f'Scale {s} (D=256): human={n_h} meter={n_m} shared={n_s} other={n_o}  cos={info["cos_sim"]:.3f}')
        
        h_dims = np.where(info['human_only'])[0]
        m_dims = np.where(info['meter_only'])[0]
        s_dims = np.where(info['shared'])[0]
        if len(h_dims) > 0:
            print(f'  Human dims: {h_dims.tolist()}')
        if len(m_dims) > 0:
            print(f'  Meter dims: {m_dims.tolist()}')
        if len(s_dims) > 0:
            print(f'  Shared dims: {s_dims.tolist()}')
        
        delta = compute_energy_balanced_targets(info)
        head_info[f'scale_{s}']['energy_delta'] = delta.tolist()
    
    results['head_weights'] = head_info
    
    # Convert numpy arrays to lists for JSON
    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        return obj

    # Save
    out_path = os.path.join(base, args.out)
    with open(out_path, 'w') as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f'\nSaved to {out_path}')
    
    # Print summary
    print(f'\n=== ENERGY-CONSERVING ADJUSTMENT SUMMARY ===')
    total_h = 0
    total_m = 0
    total_s = 0
    for key in results:
        if key.startswith('L') and isinstance(results[key], dict) and 'dim' in results[key]:
            total_h += results[key]['human_only'].count(True) if isinstance(results[key]['human_only'], list) else sum(results[key]['human_only'])
            total_m += results[key]['meter_only'].count(True) if isinstance(results[key]['meter_only'], list) else sum(results[key]['meter_only'])
            total_s += results[key]['shared'].count(True) if isinstance(results[key]['shared'], list) else sum(results[key]['shared'])
    print(f'Total human-only dims (suppress):     {total_h}')
    print(f'Total meter-only dims (amplify):      {total_m}')
    print(f'Total shared dims (boost→meter):      {total_s}')
    print(f'Energy is conserved per-layer: sum(+Δ) = sum(|−Δ|)')


def synthesize_person_images(model, n=20):
    """Generate synthetic person-activating images via gradient ascent on person class."""
    print('  Synthesizing person archetype images via gradient ascent...')
    imgs = []
    for i in range(n):
        x = torch.rand(1, 3, 640, 640, device=DEVICE) * 0.3 + 0.35
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(50):
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            # Maximize person class confidence
            person_conf = out[0, 4 + PERSON, :]
            loss = -person_conf.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        imgs.append(x.detach())
    return torch.cat(imgs)


def synthesize_meter_images(model, n=20):
    """Generate synthetic meter-activating images via gradient ascent on meter class."""
    print('  Synthesizing meter archetype images via gradient ascent...')
    imgs = []
    for i in range(n):
        x = torch.rand(1, 3, 640, 640, device=DEVICE) * 0.3 + 0.35
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(50):
            out = model(model._scale_input if hasattr(model, '_scale_input') else x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            meter_conf = out[0, 4 + METER, :]
            loss = -meter_conf.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        imgs.append(x.detach())
    return torch.cat(imgs)


if __name__ == '__main__':
    main()
