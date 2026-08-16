#!/usr/bin/env python3
"""
v31 — Activation Maximization on person/clothing feature channels.

1. Hook FPN layers at all 3 scales (stride 8/16/32)
2. Use probed top person-activating channels
3. Maximize mean activation across targeted channels simultaneously
4. DAP triangle + TPS cloth deformation + EoT
5. Structured init (rings or noise)
"""
import os, sys, math, argparse, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")
DEVICE = torch.device('cuda')
IMG_SIZE = 416
PATCH_SIZE = 224
N_RAYS = 32

# FPN layers at all 3 scales
FPN_LAYERS = {
    's32': 'module_list.80.leaky_80',
    's16': 'module_list.92.leaky_92',
    's8':  'module_list.104.leaky_104',
}

# Probed top person-activating channels per layer (from probe_activations.py)
TOP_CHANNELS = {
    's32': [722, 25, 453, 571, 469, 538, 104, 249, 640, 702, 161, 1014, 589, 274, 24, 198, 653, 284, 737, 217, 417, 277, 581, 670, 300, 173, 395, 709, 31, 330, 482, 266, 186, 770, 433, 814, 383, 478, 892, 315, 923, 560, 946, 764, 632, 462, 605, 52, 935, 597],
    's16': [432, 90, 369, 209, 259, 507, 158, 334, 125, 9, 147, 185, 384, 59, 339, 43, 120, 19, 64, 164, 494, 329, 461, 141, 81, 175, 250, 399, 316, 318, 198, 353, 444, 295, 226, 9, 488, 133, 78, 393, 109, 360, 171, 243, 222, 449, 66, 307, 237, 74],
    's8':  [165, 191, 30, 154, 137, 166, 108, 181, 70, 132, 146, 37, 23, 16, 71, 148, 246, 192, 57, 112, 85, 95, 209, 227, 6, 60, 214, 79, 232, 129, 183, 249, 14, 187, 73, 244, 236, 42, 11, 180, 139, 217, 47, 211, 240, 86, 113, 213, 96, 56],
}


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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


class DAPMask(nn.Module):
    def __init__(self, size=224, n_rays=32):
        super().__init__()
        self.size = size
        self.n_rays = n_rays
        self.lam = -100.0
        init_r = size * 0.32
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


def init_rings(size=224, n_rings=8):
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


PALETTE = torch.tensor([
    [0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],
    [255,255,0],[0,255,255],[255,0,255],
    [128,0,0],[0,128,0],[0,0,128],[128,128,0],
    [128,0,128],[0,128,128],[192,192,192],[64,64,64]
], device=DEVICE, dtype=torch.float32) / 255.0


def nps_loss(patch, palette=PALETTE):
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def augment_patch(patch, mask):
    """Apply TPS + affine + color jitter to simulate cloth/camera variation."""
    c, h, w = patch.shape

    # TPS
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = 0.04 * torch.sin(2 * math.pi * ys * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    oy = 0.04 * torch.cos(2 * math.pi * xs * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    p = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

    # Affine
    angle = (torch.rand(1, device=DEVICE) - 0.5) * 0.6
    scale = 0.9 + torch.rand(1, device=DEVICE) * 0.2
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    p = F.grid_sample(p.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(m.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

    # Color jitter
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.2
    p = (p + bright).clamp(0, 1)
    contrast = 0.85 + torch.rand(1, device=DEVICE) * 0.3
    mean = p.mean()
    p = ((p - mean) * contrast + mean).clamp(0, 1)
    gamma = 0.9 + torch.rand(1, device=DEVICE) * 0.2
    p = p ** gamma

    return p, m


def paste_on_canvas(patch, mask, y0=96, x0=96):
    canvas = torch.full((3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
    h, w = patch.shape[1], patch.shape[2]
    m3 = mask.unsqueeze(0)
    region = canvas[:, y0:y0+h, x0:x0+w]
    region = region * (1 - m3) + patch * m3
    canvas[:, y0:y0+h, x0:x0+w] = region
    return canvas


def bilateral_tv(patch):
    tv_h = (patch[:, :, 4:] - patch[:, :, :-4]).abs().mean()
    tv_w = (patch[:, 4:, :] - patch[:, :-4, :]).abs().mean()
    return tv_h + tv_w


def activation_loss(features, top_channels, layer_names, alpha=1.0):
    """Maximize mean activation of targeted channels across all scales."""
    total_loss = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, name in layer_names.items():
        if name not in features:
            continue
        f = features[name]
        chs = top_channels[scale]
        # spatial mean per channel, then mean over targeted channels
        act = f[0, chs, :, :].mean()
        loss = -act  # minimize negative = maximize activation
        total_loss = total_loss + alpha * loss
        stats[f'act_{scale}'] = act.item()
    return total_loss, stats


def save_preview(patch, mask, epoch, out_dir, stats):
    p = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    m = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (p.shape[1]*2, p.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(p), (0,0))
    combined.paste(Image.fromarray(m).convert('RGB'), (p.shape[1],0))
    txt = f"e{epoch:04d} s8:{stats.get('act_s8',0):.3f} s16:{stats.get('act_s16',0):.3f} s32:{stats.get('act_s32',0):.3f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','act_s8','act_s16','act_s32','tv','nps','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_act=1.0, w_tv=2.5, w_nps=0.1,
          w_shape=500.0, area_limit=0.25, init='rings', out_dir='outputs_v31'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_hooks(model, list(FPN_LAYERS.values()))
    print(f"Hooked {len(hooks)} FPN layers: {list(FPN_LAYERS.keys())}")
    print(f"Targeting top {sum(len(v) for v in TOP_CHANNELS.values())} channels across scales")

    if init == 'rings':
        patch = init_rings(PATCH_SIZE)
    else:
        patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    patch.requires_grad_(True)
    print(f"Init: {init}")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr)

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV31 Activation Maximization")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_act={w_act} w_tv={w_tv} w_nps={w_nps} w_shape={w_shape}\n")
    print(f"  TensorBoard: tensorboard --logdir={tb_dir}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        # Apply random deformation
        aug_patch, aug_mask = augment_patch(masked_patch, mask)

        # Paste onto neutral canvas
        canvas = paste_on_canvas(aug_patch, aug_mask)

        # Forward
        features.clear()
        out = model(canvas.unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]

        loss_act, act_stats = activation_loss(features, TOP_CHANNELS, FPN_LAYERS, alpha=w_act)
        loss_tv = bilateral_tv(patch)
        loss_n = nps_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_act + w_tv * loss_tv + w_nps * loss_n + loss_shape

        if torch.isnan(total) or torch.isinf(total):
            print(f"NaN/Inf at epoch {ep}, stopping. Best epoch {best_epoch}")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_([patch] + list(dap.parameters()), 5.0)
        opt.step()

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)
            dap.rays.clamp_(PATCH_SIZE*0.1, PATCH_SIZE*0.5)

        stats = {
            'act_s8': act_stats.get('act_s8', 0),
            'act_s16': act_stats.get('act_s16', 0),
            'act_s32': act_stats.get('act_s32', 0),
            'tv': loss_tv.item(),
            'nps': loss_n.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['act_s8'], stats['act_s16'], stats['act_s32'],
                stats['tv'], stats['nps'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Activations/s8', stats['act_s8'], ep)
        writer.add_scalar('Activations/s16', stats['act_s16'], ep)
        writer.add_scalar('Activations/s32', stats['act_s32'], ep)
        writer.add_scalar('Loss/tv', stats['tv'], ep)
        writer.add_scalar('Loss/nps', stats['nps'], ep)
        writer.add_scalar('Loss/shape', stats['shape'], ep)
        writer.add_scalar('Meta/area', area.item(), ep)

        if total.item() < best_total:
            best_total = total.item()
            best_epoch = ep
            save_preview(patch, mask, ep, out_dir, stats)
            img_np = (patch.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(out_dir, 'checkpoint_best.png'))
            writer.add_image('patch_best', patch.detach().clamp(0,1), ep)
            writer.add_image('mask_best', mask.detach().unsqueeze(0).clamp(0,1), ep)

        print(f"Ep {ep:4d}/{epochs} | tot={total.item():8.3f} | "
              f"s8={stats['act_s8']:.3f} | s16={stats['act_s16']:.3f} | "
              f"s32={stats['act_s32']:.3f} | tv={stats['tv']:.4f} | "
              f"area={area.item():.3f} | best={best_epoch}")

    final = (patch.detach() * mask.detach()).clamp(0, 1)
    final_np = (final.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(final_np).save(os.path.join(out_dir, 'patch_final.png'))
    if os.path.exists(os.path.join(out_dir, 'checkpoint_best.png')):
        import shutil
        shutil.copy(os.path.join(out_dir, 'checkpoint_best.png'),
                    os.path.join(out_dir, 'patch_final.png'))

    writer.close()
    for h in hooks:
        h.remove()
    print(f"Done: {out_dir}")
    print(f"TensorBoard: tensorboard --logdir={tb_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_act', type=float, default=1.0)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--init', type=str, default='rings', choices=['rings','noise'])
    parser.add_argument('--out', type=str, default='outputs_v31')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_act, args.w_tv, args.w_nps, args.w_shape,
          args.area_limit, args.init, args.out)
