#!/usr/bin/env python3
"""
Faithful Darknet YOLOv3 per-layer dimension extraction.

Uses eriklindernoren/PyTorch-YOLOv3 port (from AdvReal repo) which preserves
the dual objectness/class heads that original Darknet uses — and that the
Flock pico3 runs.

Extracts:
  1. Backbone + FPN layer features (hooked, GAP, classified human/meter/shared)
  2. Detection head conv weights — SPLIT into:
     - Objectness weights (channels 4, 89, 174 in the 255-output conv)
       These drive BOX GENERATION ("is there an object here?")
     - Class weights (person=5,90,175 and meter=17,102,187)
       These drive CLASSIFICATION ("what is it?")
  3. Saves everything to CSV

This is the architecturally honest map for the pico3 target.
"""
import sys, os, json, csv, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add the PyTorch-YOLOv3 path
Y3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       'AdvReal', 'detlib', 'HHDet', 'yolov3', 'PyTorch_YOLOv3')
sys.path.insert(0, Y3_PATH)

from pytorchyolo.models import load_model
from pytorchyolo.utils.parse_config import parse_model_config

DEVICE = torch.device('cuda')
PERSON = 0
METER = 12

# Darknet-53 backbone bands by resolution (1-indexed from cfg)
# Band 0: 416px, 32ch  (layer 1)
# Band 1: 208px, 64ch  (layers 2-5, residual x1)
# Band 2: 104px, 128ch (layers 6-12, residual x2)
# Band 3: 52px,  256ch (layers 13-37, residual x8)
# Band 4: 26px,  512ch (layers 38-62, residual x8)
# Band 5: 13px,  1024ch(layers 63-81, residual x4 + SPP-like convs)

# Layers to hook — one per band (the output of each residual block group)
# Using 0-indexed module_list indices (module_defs[0] is [net] hyperparams, so module_list[0] = cfg layer 1)
HOOK_LAYERS = {
    0:  ('Band 0', '416px', 'Conv 32'),
    1:  ('Band 1', '208px', 'Conv 64 + res'),
    4:  ('Band 1', '208px', 'Res1 out'),
    11: ('Band 2', '104px', 'Res2 out'),
    36: ('Band 3', '52px',  'Res3 out (x8)'),
    61: ('Band 4', '26px',  'Res4 out (x8)'),
    74: ('Band 5', '13px',  'Res5 out (x4)'),
    81: ('Band 5', '13px',  'SPP convs out'),
    # FPN layers
    91: ('FPN-16', '26px',  'FPN merge1 out'),
    98: ('FPN-8',  '52px',  'FPN merge2 out'),
    # Detection head convs (the 255-channel output convs)
    81: ('Band 5', '13px',  'Pre-detect conv'),
}

# The 3 detection head conv layers (0-indexed in module_list)
# Layer 82 in cfg = module_list index 81 (since module_defs[0] is [net])
# Actually: module_defs has [net] at index 0, then layers start at index 1
# module_list index = cfg layer number - 1
# cfg layer 82 (conv 255) → module_list[81]
# cfg layer 94 (conv 255) → module_list[93]  
# cfg layer 106 (conv 255) → module_list[105]
DETECT_CONVS = [81, 93, 105]
DETECT_STRIDES = [32, 16, 8]
DETECT_ANCHORS = ['large (116x90, 156x198, 373x326)', 
                   'medium (30x61, 62x45, 59x119)', 
                   'small (10x13, 16x30, 33x23)']


def load_images(path, max_n=50, size=416):
    """Load images, resize, return (N, 3, size, size) tensor."""
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


def extract_features(model, imgs, layer_indices):
    """Forward pass with hooks, return {layer_idx: (N, C, H, W)}."""
    feats = {}
    hooks = []
    
    def make_hook(idx):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                feats[idx] = out.detach()
        return hook

    for i in layer_indices:
        hooks.append(model.module_list[i].register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(imgs)
    
    for h in hooks:
        h.remove()
    
    return feats


def classify_dims(human_feats, meter_feats, threshold=0.3):
    """Classify each channel dim as human-only, meter-only, shared, or other."""
    human_gap = human_feats.abs().mean(dim=(0, 2, 3))
    meter_gap = meter_feats.abs().mean(dim=(0, 2, 3))
    
    total = human_gap + meter_gap + 1e-8
    human_score = human_gap / total
    meter_score = meter_gap / total
    
    human_only = (human_score > 1 - threshold) & (meter_score < threshold)
    meter_only = (meter_score > 1 - threshold) & (human_score < threshold)
    shared = (human_score > threshold) & (meter_score > threshold)
    other = ~human_only & ~meter_only & ~shared
    
    return {
        'human_only': human_only.cpu().numpy(),
        'meter_only': meter_only.cpu().numpy(),
        'shared': shared.cpu().numpy(),
        'other': other.cpu().numpy(),
        'dim': human_feats.shape[1],
    }


def extract_head_weights(model):
    """
    Extract the detection head conv weights, split into:
      - Objectness: which backbone features drive box generation
      - Person class: which features drive person classification  
      - Meter class: which features drive meter classification
    
    Each detect conv outputs 255 = 3 anchors × 85 channels.
    Per anchor: [0:4]=box, [4]=obj, [5:85]=cls
    So in the 255-channel output:
      Anchor 0: obj=4,  person=5,  meter=17
      Anchor 1: obj=89, person=90, meter=102
      Anchor 2: obj=174, person=175, meter=187
    """
    results = {}
    
    for i, (conv_idx, stride, anchor_desc) in enumerate(zip(DETECT_CONVS, DETECT_STRIDES, DETECT_ANCHORS)):
        conv = model.module_list[conv_idx][0]  # Conv2d
        w = conv.weight  # (255, in_channels, 1, 1)
        in_ch = w.shape[1]
        w_flat = w.squeeze(-1).squeeze(-1)  # (255, in_channels)
        
        # Objectness weights (average across 3 anchors)
        obj_channels = [4, 89, 174]
        obj_w = w_flat[obj_channels].mean(dim=0)  # (in_channels,)
        
        # Person class weights (average across 3 anchors)
        person_channels = [5, 90, 175]
        person_w = w_flat[person_channels].mean(dim=0)
        
        # Meter class weights (average across 3 anchors)
        meter_channels = [17, 102, 187]
        meter_w = w_flat[meter_channels].mean(dim=0)
        
        # Classify each input feature dim
        # For obj: which features drive objectness high?
        # For person/meter: which features drive each class?
        
        # Objectness vs nothing: just magnitude (obj doesn't have a "counterpart")
        obj_abs = obj_w.abs()
        
        # Person vs meter classification
        p_abs = person_w.abs()
        m_abs = meter_w.abs()
        total_cls = p_abs + m_abs + 1e-8
        p_score = p_abs / total_cls
        m_score = m_abs / total_cls
        
        threshold = 0.3
        human_only = (p_score > 1 - threshold) & (m_score < threshold)
        meter_only = (m_score > 1 - threshold) & (p_score < threshold)
        shared = (p_score > threshold) & (m_score > threshold)
        other = ~human_only & ~meter_only & ~shared
        
        # Objectness: which backbone features drive box generation
        obj_abs = obj_w.abs().cpu().numpy()
        
        # Cosine similarity between person and meter weight vectors
        cos_cls = float(F.cosine_similarity(person_w.unsqueeze(0), meter_w.unsqueeze(0)).item())
        
        # Classify objectness: which dims are most important for box firing?
        obj_top_idx = np.argsort(-obj_abs)[:20]
        
        results[f'head_stride{stride}'] = {
            'dim': in_ch,
            'conv_idx': conv_idx,
            'stride': stride,
            'anchors': anchor_desc,
            # Classification dims
            'human_only': human_only.cpu().numpy(),
            'meter_only': meter_only.cpu().numpy(),
            'shared': shared.cpu().numpy(),
            'other': other.cpu().numpy(),
            'cos_sim_cls': cos_cls,
            # Objectness weights — which features drive box generation
            'obj_weights': obj_w.cpu().numpy(),
            'person_weights': person_w.cpu().numpy(),
            'meter_weights': meter_w.cpu().numpy(),
            # Top-K objectness dims
            'obj_top_dims': obj_top_idx.tolist(),
            'obj_top_weights': obj_abs[obj_top_idx].tolist(),
        }
    
    return results


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    cfg_path = os.path.join(Y3_PATH, 'config', 'yolov3.cfg')
    weights_path = os.path.join(base, 'yolov3.weights')
    
    print(f'Loading faithful Darknet YOLOv3...')
    print(f'  cfg: {cfg_path}')
    print(f'  weights: {weights_path}')
    
    model = load_model(cfg_path, weights_path).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    print(f'Model loaded. {sum(p.numel() for p in model.parameters())} parameters')
    print(f'YOLO layers: {len(model.yolo_layers)}')
    
    # Load images
    person_path = os.path.join(base, 'data', 'coco_person_strong', 'images')
    meter_path = os.path.join(base, 'data', 'coco_meter', 'images')
    
    print(f'\nLoading person images from {person_path}...')
    person_imgs = load_images(person_path, 50, 416)
    
    if person_imgs is None:
        print('ERROR: No person images found!')
        return
    print(f'  {person_imgs.shape[0]} images')
    
    # No meter images — synthesize via gradient ascent on meter class
    print(f'No meter images — synthesizing via gradient ascent on meter class logit...')
    meter_imgs = synthesize_class_images(model, METER, n=20, size=416)
    print(f'  {meter_imgs.shape[0]} synthetic meter images')
    
    # Hook the last CONV before each shortcut (shortcuts are empty Sequential, hooks don't fire)
    # Also hook FPN merge convs and detection head convs
    hook_indices = [0, 1, 3, 10, 35, 60, 73, 81, 91, 103]
    # 0  = Band 0, 416px, 32ch (first conv)
    # 1  = Band 1, 208px, 64ch (stride-2 conv)
    # 3  = Band 1, 208px, 64ch (res1 last conv, before shortcut[4])
    # 10 = Band 2, 104px, 128ch (res2 last conv, before shortcut[11])
    # 35 = Band 3, 52px, 256ch (res3 last conv, before shortcut[36])
    # 60 = Band 4, 26px, 512ch (res4 last conv, before shortcut[61])
    # 73 = Band 5, 13px, 1024ch (res5 last conv, before shortcut[74])
    # 81 = Band 5, 13px, 255ch (detect conv stride 32)
    # 91 = FPN-16, 26px, 256ch (FPN merge conv)
    # 103= FPN-8, 52px, 256ch (FPN merge conv, before detect[105])
    print(f'\nExtracting features at layers {hook_indices}...')
    person_feats = extract_features(model, person_imgs, hook_indices)
    meter_feats = extract_features(model, meter_imgs, hook_indices)
    
    # Classify backbone dims
    print(f'\n=== Backbone/FPN dimension classification ===')
    band_info = {
        0:  ('Band 0', '416px', 'Conv 32'),
        1:  ('Band 1', '208px', 'Conv 64 (stride2)'),
        3:  ('Band 1', '208px', 'Res1 out'),
        10: ('Band 2', '104px', 'Res2 out'),
        35: ('Band 3', '52px',  'Res3 out (x8)'),
        60: ('Band 4', '26px',  'Res4 out (x8)'),
        73: ('Band 5', '13px',  'Res5 out (x4)'),
        81: ('Detect', '13px',  'Detect conv (s32)'),
        91: ('FPN-16', '26px',  'FPN merge1'),
        103:('FPN-8',  '52px',  'FPN merge2'),
    }
    
    rows = []
    for idx in hook_indices:
        if idx not in person_feats or idx not in meter_feats:
            print(f'  L{idx}: MISSING')
            continue
        
        info = classify_dims(person_feats[idx], meter_feats[idx])
        band, res, desc = band_info[idx]
        
        h_dims = [i for i, x in enumerate(info['human_only']) if x]
        m_dims = [i for i, x in enumerate(info['meter_only']) if x]
        s_dims = [i for i, x in enumerate(info['shared']) if x]
        
        print(f'  L{idx:3d} {band:>8} {res:>6} {desc:<15} D={info["dim"]:>5}  H={len(h_dims):>3} M={len(m_dims):>3} S={len(s_dims):>3}')
        
        rows.append([
            idx, band, res, desc, info['dim'],
            ';'.join(str(x) for x in h_dims),
            ';'.join(str(x) for x in m_dims),
            ';'.join(str(x) for x in s_dims),
        ])
    
    # Extract head weights
    print(f'\n=== Detection head weight analysis (objectness + class) ===')
    head_info = extract_head_weights(model)
    
    for key in sorted(head_info.keys()):
        v = head_info[key]
        nh = sum(v['human_only'])
        nm = sum(v['meter_only'])
        ns = sum(v['shared'])
        print(f'  {key} (D={v["dim"]}, {v["anchors"]}):')
        print(f'    Class: H={nh} M={nm} S={ns} cos={v["cos_sim_cls"]:.3f}')
        print(f'    Objectness top dims: {v["obj_top_dims"][:10]}')
        print(f'    Objectness top weights: {[f"{w:.3f}" for w in v["obj_top_weights"][:10]]}')
        
        h_dims = [i for i, x in enumerate(v['human_only']) if x]
        m_dims = [i for i, x in enumerate(v['meter_only']) if x]
        s_dims = [i for i, x in enumerate(v['shared']) if x]
        
        rows.append([
            f'H{v["stride"]}', f'Head-{v["stride"]}', f'stride {v["stride"]}', 
            f'obj+cls ({v["anchors"][:20]})', v['dim'],
            ';'.join(str(x) for x in h_dims),
            ';'.join(str(x) for x in m_dims),
            ';'.join(str(x) for x in s_dims),
        ])
    
    # Write CSV
    csv_path = os.path.join(base, 'darknet_layer_dims.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'band', 'resolution', 'description', 'total_dims', 
                    'human_only_dims', 'meter_only_dims', 'shared_dims'])
        w.writerows(rows)
    print(f'\nCSV saved to {csv_path}')
    
    # Also save objectness top dims to a separate file
    obj_path = os.path.join(base, 'objectness_dims.json')
    obj_data = {}
    for key in head_info:
        v = head_info[key]
        obj_data[key] = {
            'stride': v['stride'],
            'dim': v['dim'],
            'obj_top_dims': v['obj_top_dims'],
            'obj_top_weights': v['obj_top_weights'],
            'obj_all_weights': v['obj_weights'].tolist(),
        }
    with open(obj_path, 'w') as f:
        json.dump(obj_data, f, indent=2)
    print(f'Objectness weights saved to {obj_path}')


def synthesize_class_images(model, target_class, n=20, size=416):
    """Generate images that maximize a specific class via gradient ascent."""
    imgs = []
    for i in range(n):
        x = torch.rand(1, 3, size, size, device=DEVICE) * 0.3 + 0.35
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(50):
            out = model(x)
            # out: (1, 10647, 85) — [0:4]=box, [4]=obj, [5:85]=cls
            cls_conf = out[0, :, 5 + target_class]
            obj_conf = out[0, :, 4]
            # Maximize class confidence × objectness
            loss = -(cls_conf * obj_conf).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        imgs.append(x.detach())
    return torch.cat(imgs)


if __name__ == '__main__':
    main()
