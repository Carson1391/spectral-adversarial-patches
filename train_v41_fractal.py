#!/usr/bin/env python3
"""
v41 — Fractal Human via Activation Maximization on YOLOv3 FPN.

User specification:
1. White noise init (no real human image prior)
2. Hook YOLOv3 Neck/FPN layers at all 3 scales
3. Find Class 0 (person) correlated channels via activation maximization
4. Maximize mean activation of those channels simultaneously
5. DAP triangle mask + proper TPS fabric deformation
6. Lower TV so texture stays rich/noisy instead of collapsing to stripes
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
N_RAYS = 64

FPN_LAYERS = {
    's32': 'module_list.80.leaky_80',
    's16': 'module_list.92.leaky_92',
    's8':  'module_list.104.leaky_104',
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


def load_imgs(dir_path, max_imgs=100):
    if not os.path.exists(dir_path):
        return []
    files = []
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.jpg', '.png')):
            files.append(os.path.join(dir_path, f))
    files = files[:max_imgs]
    if len(files) == 0:
        return []
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(np.array(img).transpose(2, 0, 1) / 255.0)
    return [torch.tensor(x, dtype=torch.float32, device=DEVICE) for x in imgs]


def probe_person_channels(model, features, person_dir, bg_dir=None, topk=60):
    person_imgs = load_imgs(person_dir, max_imgs=100)
    bg_imgs = load_imgs(bg_dir, max_imgs=100) if bg_dir else []

    print(f"Probing with {len(person_imgs)} person images, {len(bg_imgs)} bg images")

    def collect_features(imgs):
        acts = {n: [] for n in FPN_LAYERS.values()}
        for i in range(0, len(imgs), 8):
            batch = torch.stack(imgs[i:i+8])
            features.clear()
            with torch.no_grad():
                _ = model(batch)
            for n in FPN_LAYERS.values():
                if n in features:
                    f = features[n]
                    acts[n].append(f.mean(dim=(2, 3)).cpu().numpy())
        return {n: np.concatenate(a, axis=0) for n, a in acts.items() if len(a) > 0}

    p_acts = collect_features(person_imgs) if len(person_imgs) > 0 else {}
    b_acts = collect_features(bg_imgs) if len(bg_imgs) > 0 else {}

    channels = {}
    for scale, name in FPN_LAYERS.items():
        if name not in p_acts:
            continue
        p_mean = p_acts[name].mean(axis=0)
        p_std = p_acts[name].std(axis=0)
        if name in b_acts and len(b_acts[name]) > 0:
            b_mean = b_acts[name].mean(axis=0)
            score = (p_mean - b_mean) * p_mean + p_std
        else:
            score = p_mean + p_std

        top = np.argsort(score)[::-1][:topk]
        channels[scale] = [int(x) for x in top]
        print(f"{scale}: top channels = {channels[scale][:12]}...")
    return channels


class DAPMask(nn.Module):
    def __init__(self, size=224, n_rays=64):
        super().__init__()
        self.size = size
        self.n_rays = n_rays
        self.lam = -100.0
        init_r = size * 0.30
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


def tps_warp(patch, grid_size=4, intensity=0.08):
    """
    Thin Plate Spline warp using grid_sample with a regular control grid.
    intensity: max displacement relative to patch size.
    """
    c, h, w = patch.shape

    # Regular control grid
    y_steps = torch.linspace(-1, 1, grid_size, device=DEVICE)
    x_steps = torch.linspace(-1, 1, grid_size, device=DEVICE)
    yy, xx = torch.meshgrid(y_steps, x_steps, indexing='ij')
    src = torch.stack([xx, yy], dim=-1)  # (G, G, 2)

    # Random destination displacements
    dst = src + (torch.rand_like(src) - 0.5) * 2 * intensity

    # Bilinear interpolation to full grid
    grid = F.interpolate(src.permute(2,0,1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    grid = grid.permute(0,2,3,1)

    offset = F.interpolate((dst - src).permute(2,0,1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    offset = offset.permute(0,2,3,1)

    warped = F.grid_sample(patch.unsqueeze(0), grid + offset,
                           mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
    return warped


def eot_transform(patch, mask):
    c, h, w = patch.shape

    # TPS fabric wrinkle
    p = tps_warp(patch, grid_size=4, intensity=0.12)
    m = tps_warp(mask.unsqueeze(0), grid_size=4, intensity=0.12).squeeze(0)

    # Affine
    angle = (torch.rand(1, device=DEVICE) - 0.5) * 0.6
    scale = 0.88 + torch.rand(1, device=DEVICE) * 0.24
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    p = F.grid_sample(p.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(m.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()

    # Blur
    if torch.rand(1).item() < 0.6:
        k = 5
        sigma = 0.6 + torch.rand(1).item() * 0.8
        half = k // 2
        x = torch.arange(-half, half+1, device=DEVICE, dtype=torch.float32)
        g = torch.exp(-x*x/(2*sigma*sigma))
        g = g / g.sum()
        kv = g.view(1, 1, k, 1).expand(c, 1, k, 1)
        p = p.view(1, c, h, w)
        p = F.conv2d(p, kv, padding=(half, 0), groups=c)
        p = p.view(c, h, w)
        kh = g.view(1, 1, 1, k).expand(c, 1, 1, k)
        p = p.view(1, c, h, w)
        p = F.conv2d(p, kh, padding=(0, half), groups=c)
        p = p.view(c, h, w)

    # Color jitter
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.20
    p = (p + bright).clamp(0, 1)
    contrast = 0.85 + torch.rand(1, device=DEVICE) * 0.30
    mean = p.mean()
    p = ((p - mean) * contrast + mean).clamp(0, 1)
    gamma = 0.88 + torch.rand(1, device=DEVICE) * 0.24
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


def tv_loss(patch):
    tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def activation_loss(features, channels, alpha=1.0):
    total = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, name in FPN_LAYERS.items():
        if name not in features:
            continue
        f = features[name]
        chs = channels[scale]
        act = f[0, chs, :, :].mean()
        total = total + alpha * (-act)
        stats[f'act_{scale}'] = act.item()
    return total, stats


def save_preview(patch, mask, epoch, out_dir, stats):
    p = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    m = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (p.shape[1]*2, p.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(p), (0,0))
    combined.paste(Image.fromarray(m).convert('RGB'), (p.shape[1],0))
    txt = f"e{epoch:04d} s8:{stats.get('act_s8',0):.2f} s16:{stats.get('act_s16',0):.2f} s32:{stats.get('act_s32',0):.2f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','act_s8','act_s16','act_s32','tv','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_act=1.0, w_tv=1.0,
          w_shape=500.0, area_limit=0.25, topk=60,
          out_dir='outputs_v41'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_hooks(model, list(FPN_LAYERS.values()))

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    channels = probe_person_channels(model, features, person_dir, topk=topk)
    if len(channels) == 0:
        raise RuntimeError("No channels probed")

    # Pure white noise init
    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    patch.requires_grad_(True)
    print("Init: pure white noise")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV41 Fractal Human / Activation Maximization")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_act={w_act} w_tv={w_tv} w_shape={w_shape}\n")
    print(f"  TensorBoard: tensorboard --logdir={tb_dir}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        aug_patch, aug_mask = eot_transform(masked_patch, mask)
        canvas = paste_on_canvas(aug_patch, aug_mask)

        features.clear()
        out = model(canvas.unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]

        loss_act, act_stats = activation_loss(features, channels, alpha=w_act)
        loss_tv = w_tv * tv_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_act + loss_tv + loss_shape

        if torch.isnan(total) or torch.isinf(total):
            print(f"NaN/Inf at epoch {ep}, stopping. Best epoch {best_epoch}")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_([patch] + list(dap.parameters()), 3.0)
        opt.step()

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)
            dap.rays.clamp_(PATCH_SIZE*0.08, PATCH_SIZE*0.50)

        stats = {
            'act_s8': act_stats.get('act_s8', 0),
            'act_s16': act_stats.get('act_s16', 0),
            'act_s32': act_stats.get('act_s32', 0),
            'tv': loss_tv.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['act_s8'], stats['act_s16'], stats['act_s32'],
                stats['tv'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Activations/s8', stats['act_s8'], ep)
        writer.add_scalar('Activations/s16', stats['act_s16'], ep)
        writer.add_scalar('Activations/s32', stats['act_s32'], ep)
        writer.add_scalar('Loss/tv', stats['tv'], ep)
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
              f"s8={stats['act_s8']:5.2f} s16={stats['act_s16']:5.2f} s32={stats['act_s32']:5.2f} | "
              f"tv={stats['tv']:.3f} | area={area.item():.3f} | best={best_epoch}")

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
    parser.add_argument('--w_tv', type=float, default=1.0)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=60)
    parser.add_argument('--out', type=str, default='outputs_v41')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_act, args.w_tv,
          args.w_shape, args.area_limit, args.topk, args.out)
