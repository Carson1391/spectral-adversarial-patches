#!/usr/bin/env python3
"""
v24 — Feature-space multi-stable poisoning.

Attacks mid-backbone feature vectors (where attribute classifiers branch)
instead of final detection class logits.

Loss:
  L = alpha * ||feat||_2            (confident feature vector)
    - lambda * Var(feat_1, ..., feat_V)  (inconsistent across views)
    - beta * obj                      (hold box)
    + TV + NPS + shape

The patch learns to produce a strong but view-dependent attribute signature,
poisoning re-id / attribute consistency without needing a specific target class.
"""
import os, sys, math, argparse, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")

DEVICE = torch.device('cuda')
IMG_SIZE = 416
PATCH_SIZE = 300
N_RAYS = 32


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DAPMask(nn.Module):
    def __init__(self, size=300, n_rays=32):
        super().__init__()
        self.size = size
        self.n_rays = n_rays
        self.lam = -100.0
        init_r = size * 0.35
        self.rays = nn.Parameter(torch.full((n_rays,), init_r, device=DEVICE))
        self.center = nn.Parameter(torch.tensor([size/2.0, size/2.0], device=DEVICE))

    def forward(self):
        angles = torch.linspace(0, 2*math.pi, self.n_rays+1, device=DEVICE)[:-1]
        vx = self.center[0] + self.rays * torch.cos(angles)
        vy = self.center[1] + self.rays * torch.sin(angles)
        vertices = torch.stack([vx, vy], dim=1)

        ys = torch.arange(self.size, device=DEVICE, dtype=torch.float32)
        xs = torch.arange(self.size, device=DEVICE, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

        mask = torch.zeros(self.size*self.size, device=DEVICE)
        for i in range(self.n_rays):
            v1, v2, v3 = self.center, vertices[i], vertices[(i+1) % self.n_rays]
            c1 = (v2[0]-v1[0])*(pts[:,1]-v1[1]) - (v2[1]-v1[1])*(pts[:,0]-v1[0])
            c2 = (v3[0]-v2[0])*(pts[:,1]-v2[1]) - (v3[1]-v2[1])*(pts[:,0]-v2[0])
            c3 = (v1[0]-v3[0])*(pts[:,1]-v3[1]) - (v1[1]-v3[1])*(pts[:,0]-v3[0])
            inside = (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
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


def init_rings(size=300, n_rings=8):
    img = Image.new('RGB', (size, size), (255,255,255))
    draw = ImageDraw.Draw(img)
    cx, cy = size//2, size//2
    colors = [(0,0,0),(255,0,0),(0,0,255),(0,128,0),(255,255,0),(128,0,128),(255,128,0),(0,128,128)]
    max_r = int(math.hypot(cx, cy))
    for i in range(n_rings, -1, -1):
        r = int(max_r * (i / n_rings))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=colors[i % len(colors)])
    arr = np.array(img).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def init_chevrons(size=300, n=12):
    img = Image.new('RGB', (size, size), (255,255,255))
    draw = ImageDraw.Draw(img)
    colors = [(0,0,0),(255,0,0),(0,0,255),(0,128,0),(255,255,0)]
    for y in range(-size, size*2, size//n):
        color = colors[(y // (size//n)) % len(colors)]
        pts = [(-10, y), (size//2, y+size//n), (size+10, y)]
        draw.polygon(pts, fill=color)
    arr = np.array(img).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


PALETTE = torch.tensor([
    [0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],
    [255,255,0],[0,255,255],[255,0,255],
    [128,0,0],[0,128,0],[0,0,128],[128,128,0],
    [128,0,128],[0,128,128],[192,192,192],[64,64,64]
], device=DEVICE, dtype=torch.float32) / 255.0


def quantize_ste(patch, palette=PALETTE):
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    idx = torch.argmin(torch.cdist(p, palette), dim=1)
    q = palette[idx].reshape(patch.shape[1], patch.shape[2], 3).permute(2, 0, 1)
    return patch + (q - patch).detach()


def nps_loss(patch, palette=PALETTE):
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def tps_warp(patch, intensity=0.04):
    c, h, w = patch.shape
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = intensity * torch.sin(2 * math.pi * ys * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    oy = intensity * torch.cos(2 * math.pi * xs * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    return F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                         padding_mode='border', align_corners=True).squeeze(0)


def affine_transform(patch, angle=None, scale=None):
    c, h, w = patch.shape
    if angle is None:
        angle = (torch.rand(1, device=DEVICE) - 0.5) * 2.0
    if scale is None:
        scale = 0.7 + torch.rand(1, device=DEVICE) * 0.6
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=patch.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    return F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                         padding_mode='border', align_corners=True).squeeze(0)


def color_jitter(patch):
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.3
    patch = (patch + bright).clamp(0, 1)
    contrast = 0.75 + torch.rand(1, device=DEVICE) * 0.5
    mean = patch.mean()
    patch = ((patch - mean) * contrast + mean).clamp(0, 1)
    gamma = 0.8 + torch.rand(1, device=DEVICE) * 0.4
    patch = patch ** gamma
    return patch


def bilateral_tv(patch):
    tv_h = (patch[:, :, 4:] - patch[:, :, :-4]).abs().mean()
    tv_w = (patch[:, 4:, :] - patch[:, :-4, :]).abs().mean()
    return tv_h + tv_w


def feature_loss(feats_list, alpha=1.0, lam=2.0):
    """
    feats_list: list of (B, C, H, W) or (B, C) feature tensors from multiple views
    Loss = alpha * mean(||feat||_2) - lambda * Var(feats across views)
    """
    # Global pool each view -> (B, C)
    pooled = []
    for f in feats_list:
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        pooled.append(f)

    stack = torch.stack(pooled, dim=0)  # (V, B, C)

    # Confidence: high L2 norm
    conf = stack.norm(dim=-1).mean()

    # Variance across views: maximize inconsistency
    var = stack.var(dim=0).mean()

    loss = alpha * conf - lam * var
    return loss, {'conf': conf.item(), 'var': var.item()}


def obj_loss(preds_list):
    objs = torch.stack([p[:, :, 4].sigmoid() for p in preds_list], dim=0)
    return -objs.mean()


def register_hooks(model, layer_names):
    features = {}
    hooks = []
    modules = dict(model.named_modules())
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))
    return features, hooks


def save_preview(patch, mask, epoch, out_dir, stats):
    img = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    msk = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (img.shape[1]*2, img.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(img), (0,0))
    combined.paste(Image.fromarray(msk).convert('RGB'), (img.shape[1],0))
    txt = f"e{epoch:04d} conf:{stats['conf']:.2f} var:{stats['var']:.2f} obj:{stats['obj']:.3f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, img.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','conf','var','obj','tv','nps','shape','total'])
    return path


def train(epochs=800, lr=0.01, n_views=8, alpha=1.0, lam=2.0,
          w_obj=0.5, w_tv=0.1, w_nps=0.1, w_shape=5000.0, area_limit=0.25,
          init='rings', out_dir='outputs_v24'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)

    model = load_frozen_model()

    # Mid-backbone hooks
    layer_names = ['module_list.12.leaky_12', 'module_list.37.leaky_37']
    features, hooks = register_hooks(model, layer_names)
    print(f"Hooked {len(hooks)} mid-backbone layers")

    if init == 'rings':
        patch = init_rings(PATCH_SIZE)
    elif init == 'chevrons':
        patch = init_chevrons(PATCH_SIZE)
    else:
        patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    patch.requires_grad_(True)
    print(f"Init: {init}")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr)

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV24 Feature-Space Multi-Stable Poison")
    print(f"  epochs={epochs} lr={lr} n_views={n_views}")
    print(f"  alpha={alpha} lambda={lam} w_obj={w_obj}")
    print(f"  w_tv={w_tv} w_nps={w_nps} w_shape={w_shape}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        preds_list = []
        feats_list = []
        for _ in range(n_views):
            aug = masked_patch
            aug = tps_warp(aug)
            aug = affine_transform(aug)
            aug = color_jitter(aug)
            aug = quantize_ste(aug)

            canvas = torch.full((1, 3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
            y0 = (IMG_SIZE - PATCH_SIZE) // 2
            x0 = (IMG_SIZE - PATCH_SIZE) // 2
            canvas[:, :, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = aug.unsqueeze(0)

            features.clear()
            out = model(canvas)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds_list.append(out)
            # Collect hooked features from this view
            view_feats = []
            for n in layer_names:
                if n in features:
                    view_feats.append(features[n])
            feats_list.append(torch.cat([f.mean(dim=(2,3)) for f in view_feats], dim=1))

        loss_feat, feat_stats = feature_loss(feats_list, alpha, lam)
        loss_o = obj_loss(preds_list)
        loss_tv = bilateral_tv(patch)
        loss_n = nps_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = w_obj * loss_o + loss_feat + w_tv * loss_tv + w_nps * loss_n + loss_shape

        total.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([patch] + list(dap.parameters()), 10.0)

        opt.step()

        with torch.no_grad():
            patch.clamp_(0, 1)
            dap.rays.clamp_(PATCH_SIZE*0.1, PATCH_SIZE*0.5)
            if ep % 10 == 0:
                patch.copy_(quantize_ste(patch))

        stats = {
            'conf': feat_stats['conf'],
            'var': feat_stats['var'],
            'obj': loss_o.item(),
            'tv': loss_tv.item(),
            'nps': loss_n.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['conf'], stats['var'], stats['obj'],
                stats['tv'], stats['nps'], stats['shape'], stats['total']])

        if total.item() < best_total:
            best_total = total.item()
            best_epoch = ep
            save_preview(patch, mask, ep, out_dir, stats)
            Image.fromarray((patch.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)).save(
                os.path.join(out_dir, 'checkpoint_best.png'))

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"Ep {ep:4d}/{epochs} | tot={total.item():8.3f} | "
                  f"conf={stats['conf']:.2f} | var={stats['var']:.2f} | "
                  f"obj={stats['obj']:6.3f} | tv={stats['tv']:.4f} | "
                  f"area={area.item():.3f} | best={best_epoch}")

    final = (patch.detach() * mask.detach()).clamp(0, 1)
    final_np = (final.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(final_np).save(os.path.join(out_dir, 'patch_final.png'))
    if os.path.exists(os.path.join(out_dir, 'checkpoint_best.png')):
        import shutil
        shutil.copy(os.path.join(out_dir, 'checkpoint_best.png'),
                    os.path.join(out_dir, 'patch_final.png'))

    for h in hooks:
        h.remove()

    print(f"Done: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_views', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--w_obj', type=float, default=0.5)
    parser.add_argument('--w_tv', type=float, default=0.1)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=5000.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--init', type=str, default='rings', choices=['rings','chevrons','noise'])
    parser.add_argument('--out', type=str, default='outputs_v24')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.n_views, args.alpha, args.lam,
          args.w_obj, args.w_tv, args.w_nps, args.w_shape, args.area_limit,
          args.init, args.out)
