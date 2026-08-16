#!/usr/bin/env python3
"""
v22 — Hold-and-Poison via per-layer feature dim manipulation.

Strategy:
1. Faithful Darknet YOLOv3 surrogate (NOT yolov3u)
2. DAP triangle mask (~20% torso)
3. Per-layer dim maps from darknet_layer_dims.csv:
   - human-only dims → DOWN (suppress person semantic)
   - meter-only dims → UP (inject meter features)
   - shared dims → UP (hold person box + raise meter)
4. Objectness loss → hold box
5. No-switch penalty → only person or meter, penalize 78 others
6. EoT/TPS cloth simulation
7. Heavy TV + NPS for printability

Train on grey canvas; patch must itself look human-enough to fire a box.
"""
import os, sys, math, argparse, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Faithful Darknet YOLOv3 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")
LAYER_DIMS_PATH = os.path.join(BASE, "darknet_layer_dims.csv")

DEVICE = torch.device('cuda')
PERSON = 0
METER = 12
IMG_SIZE = 416
PATCH_SIZE = 300
N_RAYS = 64


# ─── 1. MODEL ─────────────────────────────────────────────────────────── #

def load_frozen_model():
    print("Loading faithful Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params frozen")
    return model


class FeatureHook:
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.activation = output
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            self.activation = output[0]

    def remove(self):
        self.handle.remove()


def register_hooks(model, layer_names):
    features = {}
    hooks = []
    modules = dict(model.named_modules())

    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
            elif isinstance(o, (tuple, list)) and isinstance(o[0], torch.Tensor):
                features[name] = o[0]
        return fn

    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))
    return features, hooks


# ─── 2. DAP TRIANGLE MASK ─────────────────────────────────────────────── #

class DAPMask(nn.Module):
    """Differentiable triangle mask from Shape Matters (DAP)."""
    def __init__(self, n_rays=64, size=300):
        super().__init__()
        self.n_rays = n_rays
        self.size = size
        self.lam = -100.0
        # Initialize rays to a circle
        init_len = size * 0.45
        self.rays = nn.Parameter(torch.full((n_rays,), init_len, device=DEVICE))
        self.center = nn.Parameter(torch.tensor([size/2.0, size/2.0], device=DEVICE))

    def forward(self):
        angles = torch.linspace(0, 2*math.pi, self.n_rays+1, device=DEVICE)[:-1]
        # Vertices: center + ray * (cos, sin)
        vx = self.center[0] + self.rays * torch.cos(angles)
        vy = self.center[1] + self.rays * torch.sin(angles)
        vertices = torch.stack([vx, vy], dim=1)  # (n_rays, 2)

        # Build pixel grid
        ys = torch.arange(self.size, device=DEVICE, dtype=torch.float32)
        xs = torch.arange(self.size, device=DEVICE, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

        # For each pixel, compute point-in-triangle for each pair of rays
        mask = torch.zeros(self.size*self.size, device=DEVICE)
        for i in range(self.n_rays):
            v1 = self.center
            v2 = vertices[i]
            v3 = vertices[(i+1) % self.n_rays]
            cross1 = (v2[0]-v1[0])*(pts[:,1]-v1[1]) - (v2[1]-v1[1])*(pts[:,0]-v1[0])
            cross2 = (v3[0]-v2[0])*(pts[:,1]-v2[1]) - (v3[1]-v2[1])*(pts[:,0]-v2[0])
            cross3 = (v1[0]-v3[0])*(pts[:,1]-v3[1]) - (v1[1]-v3[1])*(pts[:,0]-v3[0])
            inside = (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0)
            if i == 0:
                mask = inside.float()
            else:
                mask = torch.maximum(mask, inside.float())

        mask = mask.reshape(self.size, self.size)
        mask = (torch.tanh(self.lam * (mask - 0.5)) + 1) / 2
        return mask

    def area(self):
        angles = torch.linspace(0, 2*math.pi, self.n_rays+1, device=DEVICE)[:-1]
        vx = self.center[0] + self.rays * torch.cos(angles)
        vy = self.center[1] + self.rays * torch.sin(angles)
        x = torch.cat([vx, vx[:1]])
        y = torch.cat([vy, vy[:1]])
        return 0.5 * torch.abs(torch.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))


# ─── 3. EoT/TPS ───────────────────────────────────────────────────────── #

def tps_warp(patch, intensity=0.04):
    c, h, w = patch.shape
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = intensity * torch.sin(2 * math.pi * ys * 2 + 0.7)
    oy = intensity * torch.cos(2 * math.pi * xs * 2 + 1.3)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    return F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                         padding_mode='border', align_corners=True).squeeze(0)


def eot_jitter(patch):
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.2
    patch = (patch + bright).clamp(0, 1)
    contrast = 0.85 + torch.rand(1, device=DEVICE) * 0.3
    mean = patch.mean()
    patch = ((patch - mean) * contrast + mean).clamp(0, 1)
    return patch


# ─── 4. LOSS COMPONENTS ───────────────────────────────────────────────── #

def tv_loss(patch):
    tv_h = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    tv_w = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    return tv_h + tv_w


def nps_loss(patch):
    palette = torch.tensor([
        [0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[0,255,255],[255,0,255],
        [128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],[192,192,192],[64,64,64]
    ], device=DEVICE, dtype=torch.float32) / 255.0
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def load_layer_dims():
    """Load per-layer dim maps from CSV (semicolon-separated indices)."""
    dims = {}
    with open(LAYER_DIMS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            def parse(s):
                if not s or s.strip() == '':
                    return []
                return [int(x) for x in s.split(';')]
            dims[layer] = {
                'human': parse(row['human_only_dims']),
                'meter': parse(row['meter_only_dims']),
                'shared': parse(row['shared_dims']),
            }
    return dims


def per_layer_energy_loss(features, layer_dims, w_human=3.0, w_meter=3.0, w_shared=3.0):
    """
    Energy-conserving dim manipulation.
    - human-only dims → DOWN (drive toward zero)
    - meter-only dims → UP
    - shared dims → UP
    """
    total = torch.tensor(0.0, device=DEVICE)
    stats = {}

    for name, feat in features.items():
        if name not in layer_dims:
            continue
        ld = layer_dims[name]
        h_dims = ld['human']
        m_dims = ld['meter']
        s_dims = ld['shared']

        # Global mean across spatial dims
        f = feat.mean(dim=(2, 3))  # (B, C)

        loss_h = -f[:, h_dims].abs().mean() if h_dims else torch.tensor(0.0, device=DEVICE)
        loss_m = -f[:, m_dims].mean() if m_dims else torch.tensor(0.0, device=DEVICE)
        loss_s = -f[:, s_dims].mean() if s_dims else torch.tensor(0.0, device=DEVICE)

        layer_loss = w_human * loss_h + w_meter * loss_m + w_shared * loss_s
        total = total + layer_loss

        stats[name] = {
            'h': float(loss_h.item()) if h_dims else 0,
            'm': float(loss_m.item()) if m_dims else 0,
            's': float(loss_s.item()) if s_dims else 0,
        }

    return total, stats


def output_loss(preds, target_class=METER, topk=500):
    """
    preds: (1, 10647, 85)
    - Hold objectness on anchors overlapping patch region
    - Push target class up, person class down
    - Penalize all other classes (no switching)
    """
    obj_logit = preds[:, :, 4]
    cls_logits = preds[:, :, 5:85]

    obj_sig = obj_logit.sigmoid()
    cls_sig = cls_logits.sigmoid()

    person_conf = obj_sig * cls_sig[:, :, PERSON]
    target_conf = obj_sig * cls_sig[:, :, target_class]

    other_mask = torch.ones(80, device=DEVICE, dtype=torch.bool)
    other_mask[PERSON] = False
    other_mask[target_class] = False
    other_confs = obj_sig.unsqueeze(-1) * cls_sig[:, :, other_mask]

    # Focus on anchors where target or person is strong
    focus = torch.maximum(target_conf, person_conf)
    if focus.numel() > topk:
        _, top_idx = focus.squeeze(0).topk(topk)
    else:
        top_idx = torch.arange(focus.shape[1], device=DEVICE)

    top_obj = obj_sig[0, top_idx]
    top_person = person_conf[0, top_idx]
    top_target = target_conf[0, top_idx]
    top_other = other_confs[0, top_idx, :]

    loss_obj = -top_obj.mean()
    loss_target = -top_target.mean()
    loss_person = top_person.mean()
    loss_other = top_other.mean()

    loss_cls = loss_target + loss_person + 2.0 * loss_other

    return loss_obj, loss_cls, {
        'obj': loss_obj.item(),
        'target': loss_target.item(),
        'person': loss_person.item(),
        'other': loss_other.item(),
    }


# ─── 5. PREVIEW / LOGGING ─────────────────────────────────────────────── #

def save_preview(patch, mask, epoch, out_dir, losses):
    img_np = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask_np = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    msk = Image.fromarray(mask_np).convert('RGB')

    txt = f"e{epoch:04d} obj:{losses['obj']:.3f} t:{losses['target']:.3f} p:{losses['person']:.3f} o:{losses['other']:.3f}"
    new = Image.new('RGB', (img.width * 2, img.height + 24), (32, 32, 32))
    new.paste(img, (0, 0))
    new.paste(msk, (img.width, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(new)
    draw.text((4, img.height + 4), txt, fill=(0, 255, 0))
    new.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','obj','target','person','other','layer','tv','nps','shape','total'])
    return path


# ─── 6. TRAINING LOOP ─────────────────────────────────────────────────── #

def train(epochs=800, lr=0.01, w_obj=2.0, w_cls=3.0, w_layer=1.0,
          w_human=3.0, w_meter=3.0, w_shared=3.0,
          w_tv=2.5, w_nps=0.1, w_shape=2000.0, area_limit=0.25,
          out_dir='outputs_hold_poison/v4'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)

    model = load_frozen_model()

    # Hook all layers that appear in layer_dims.csv
    layer_dims = load_layer_dims()
    layer_names = list(layer_dims.keys())
    features, hooks = register_hooks(model, layer_names)
    print(f"Hooked {len(hooks)} layers from dim map")

    # Init patch from nebula or noise
    base_path = os.path.join(BASE, '010_crab_nebula.jpg')
    if os.path.exists(base_path):
        img = Image.open(base_path).convert('RGB').resize((PATCH_SIZE, PATCH_SIZE))
        arr = np.array(img).transpose(2, 0, 1) / 255.0
        patch = torch.tensor(arr, device=DEVICE, dtype=torch.float32)
        print("Patch init: crab nebula")
    else:
        patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE, dtype=torch.float32)
        print("Patch init: noise")
    patch.requires_grad_(True)

    dap = DAPMask(n_rays=N_RAYS, size=PATCH_SIZE).to(DEVICE)

    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr)

    best_total = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"V22 HOLD-AND-POISON (per-layer dims)")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_obj={w_obj} w_cls={w_cls} w_layer={w_layer}")
    print(f"  w_human={w_human} w_meter={w_meter} w_shared={w_shared}")
    print(f"  w_tv={w_tv} w_nps={w_nps} w_shape={w_shape} area_limit={area_limit}")
    print(f"{'='*60}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        # DAP mask
        mask = dap.forward()  # (300,300)
        m3 = mask.unsqueeze(0).unsqueeze(0)  # (1,1,300,300)

        # Apply patch with mask onto neutral canvas
        warped = tps_warp(patch)
        augmented = eot_jitter(warped)
        patch_masked = augmented * m3  # black outside triangle

        canvas = torch.full((1, 3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
        y0 = (IMG_SIZE - PATCH_SIZE) // 2
        x0 = (IMG_SIZE - PATCH_SIZE) // 2
        canvas[:, :, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = patch_masked

        # Forward
        features.clear()
        out = model(canvas)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.unsqueeze(0)  # (1, 10647, 85)

        # Losses
        loss_obj, loss_cls, out_stats = output_loss(out, target_class=METER)
        loss_layer, _ = per_layer_energy_loss(features, layer_dims, w_human, w_meter, w_shared)
        loss_tv = tv_loss(patch)
        loss_nps = nps_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = (w_obj * loss_obj +
                 w_cls * loss_cls +
                 w_layer * loss_layer +
                 w_tv * loss_tv +
                 w_nps * loss_nps +
                 loss_shape)

        total.backward()
        opt.step()

        with torch.no_grad():
            patch.clamp_(0, 1)
            dap.rays.clamp_(PATCH_SIZE*0.1, PATCH_SIZE*0.5)

        stats = {
            'obj': out_stats['obj'],
            'target': out_stats['target'],
            'person': out_stats['person'],
            'other': out_stats['other'],
            'layer': loss_layer.item(),
            'tv': loss_tv.item(),
            'nps': loss_nps.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['obj'], stats['target'], stats['person'], stats['other'],
                stats['layer'], stats['tv'], stats['nps'], stats['shape'], stats['total']])

        if total.item() < best_total:
            best_total = total.item()
            best_epoch = ep
            save_preview(patch, mask, ep, out_dir, stats)
            img_np = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(out_dir, 'checkpoint_best.png'))

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"Ep {ep:4d}/{epochs} | tot={total.item():8.3f} | "
                  f"obj={out_stats['obj']:6.3f} | t={out_stats['target']:6.3f} | "
                  f"p={out_stats['person']:6.3f} | o={out_stats['other']:6.3f} | "
                  f"lay={loss_layer.item():7.1f} | tv={loss_tv.item():.4f} | "
                  f"area={area.item():.3f} | best={best_epoch}")

    # Final
    final = (patch.detach() * m3.detach()).clamp(0, 1)
    final_np = (final[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(final_np).save(os.path.join(out_dir, 'patch_final.png'))
    # Copy best to final
    best_path = os.path.join(out_dir, 'checkpoint_best.png')
    if os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, os.path.join(out_dir, 'patch_final.png'))
        print(f"\nBest patch (epoch {best_epoch}) copied to final")

    for h in hooks:
        h.remove()

    print(f"Done. Output: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_obj', type=float, default=2.0)
    parser.add_argument('--w_cls', type=float, default=3.0)
    parser.add_argument('--w_layer', type=float, default=1.0)
    parser.add_argument('--w_human', type=float, default=3.0)
    parser.add_argument('--w_meter', type=float, default=3.0)
    parser.add_argument('--w_shared', type=float, default=3.0)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=2000.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--out', type=str, default='outputs_hold_poison/v4')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_obj, args.w_cls, args.w_layer,
          args.w_human, args.w_meter, args.w_shared,
          args.w_tv, args.w_nps, args.w_shape, args.area_limit, args.out)
