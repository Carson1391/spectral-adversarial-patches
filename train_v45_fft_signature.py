#!/usr/bin/env python3
"""
v45 — Unique Human FFT Signature attack.

Ryan's idea:
1. Forward a person image through all CNN layers.
2. Compute FFT of activations at every layer.
3. Forward the same image without the human (background/neutral) through same layers.
4. Compute FFT of background activations.
5. Subtract: person_fft - bg_fft = unique human FFT signature per layer.
6. Optimize patch so its forward pass reproduces these unique human FFT signatures.

This directly targets the frequency components the model associates with humans,
not just any strong activation.
"""
import os, sys, math, argparse, csv, random
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
PERSON = 0

# FPN layers where we also do channel activation maximization
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


def register_all_hooks(model):
    """Register forward hooks on all convolutional / leaky layers."""
    features = {}
    hooks = []
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.LeakyReLU)):
            hooks.append(m.register_forward_hook(make_hook(n)))
    return features, hooks


def load_imgs(dir_path, max_imgs=100):
    if not os.path.exists(dir_path):
        return []
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg','.png'))])
    files = files[:max_imgs]
    imgs = []
    for f in files:
        img = Image.open(os.path.join(dir_path, f)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(np.array(img).transpose(2, 0, 1) / 255.0)
    return [torch.tensor(x, dtype=torch.float32, device=DEVICE) for x in imgs]


def collect_fft_signatures(model, features, imgs, layers_of_interest):
    """For each layer name, collect FFT magnitudes averaged over images."""
    sigs = {n: [] for n in layers_of_interest}
    for i in range(0, len(imgs), 8):
        batch = torch.stack(imgs[i:i+8])
        features.clear()
        with torch.no_grad():
            _ = model(batch)
        for n in layers_of_interest:
            if n not in features:
                continue
            f = features[n]  # (B, C, H, W)
            # Average over channels, then FFT per image
            f_avg = f.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            fft = torch.fft.fft2(f_avg, dim=(-2, -1))
            mag = torch.fft.fftshift(fft.abs(), dim=(-2, -1))
            sigs[n].append(mag.cpu().numpy())
    out = {}
    for n in layers_of_interest:
        if len(sigs[n]) == 0:
            continue
        arr = np.concatenate(sigs[n], axis=0)  # (N, 1, H, W)
        out[n] = arr.mean(axis=0).squeeze(0)  # (H, W)
    return out


def compute_human_fft_signatures(model, features, person_dir, bg_dir=None):
    print("Computing unique human FFT signatures...")
    person_imgs = load_imgs(person_dir, max_imgs=100)
    if bg_dir is not None:
        bg_imgs = load_imgs(bg_dir, max_imgs=100)
    else:
        # Use grey/neutral images as background
        bg_imgs = [torch.full((3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE) for _ in range(50)]

    # Get all layer names that have spatial outputs
    dummy = torch.full((1, 3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
    features.clear()
    with torch.no_grad():
        _ = model(dummy)
    layers_of_interest = [n for n, t in features.items() if len(t.shape) == 4 and t.shape[2] >= 4 and t.shape[3] >= 4]
    print(f"Found {len(layers_of_interest)} spatial layers")

    person_sigs = collect_fft_signatures(model, features, person_imgs, layers_of_interest)
    bg_sigs = collect_fft_signatures(model, features, bg_imgs, layers_of_interest)

    human_sigs = {}
    for n in layers_of_interest:
        if n not in person_sigs or n not in bg_sigs:
            continue
        diff = person_sigs[n] - bg_sigs[n]
        diff = np.maximum(diff, 0)  # keep only person-excess frequencies
        diff = diff / (diff.max() + 1e-8)
        human_sigs[n] = torch.tensor(diff, dtype=torch.float32, device=DEVICE)
    return layers_of_interest, human_sigs


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


def init_from_human_fft(human_sigs, layers_of_interest, patch_size=224):
    """
    Combine human FFT signatures from multiple layers into a patch init.
    For each signature, resize to patch size, take inverse FFT with random phase,
    then sum/merge.
    """
    acc = torch.zeros(3, patch_size, patch_size, device=DEVICE)
    count = 0
    # Pick a subset of layers to avoid noise from too many small layers
    selected = [n for n in layers_of_interest if n in human_sigs and human_sigs[n].numel() >= 16*16]
    selected = selected[::max(1, len(selected)//8)]  # sample ~8 layers
    print(f"Init from {len(selected)} layers")
    for n in selected:
        sig = human_sigs[n].cpu().numpy()
        sig_big = np.array(Image.fromarray((sig * 255).astype(np.uint8)).resize((patch_size, patch_size)))
        mag = sig_big / 255.0
        phase = np.random.rand(patch_size, patch_size) * 2 * np.pi
        complex_spec = mag * np.exp(1j * phase)
        img = np.fft.ifft2(np.fft.ifftshift(complex_spec)).real
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        for c in range(3):
            acc[c] += torch.tensor(img, dtype=torch.float32, device=DEVICE)
        count += 1
    if count > 0:
        acc = acc / count
    # Add noise for richness
    acc = acc * 0.8 + torch.rand_like(acc) * 0.2
    return acc


def tps_warp(patch, grid_size=4, intensity=0.10):
    c, h, w = patch.shape
    y_steps = torch.linspace(-1, 1, grid_size, device=DEVICE)
    x_steps = torch.linspace(-1, 1, grid_size, device=DEVICE)
    yy, xx = torch.meshgrid(y_steps, x_steps, indexing='ij')
    src = torch.stack([xx, yy], dim=-1)
    dst = src + (torch.rand_like(src) - 0.5) * 2 * intensity
    grid = F.interpolate(src.permute(2,0,1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True).permute(0,2,3,1)
    offset = F.interpolate((dst - src).permute(2,0,1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True).permute(0,2,3,1)
    return F.grid_sample(patch.unsqueeze(0), grid + offset, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)


def eot_transform(patch, mask):
    c, h, w = patch.shape
    p = tps_warp(patch, grid_size=4, intensity=0.12)
    m = tps_warp(mask.unsqueeze(0), grid_size=4, intensity=0.12).squeeze(0)

    angle = (torch.rand(1, device=DEVICE) - 0.5) * 0.6
    scale = 0.88 + torch.rand(1, device=DEVICE) * 0.24
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    p = F.grid_sample(p.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(m.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()

    if torch.rand(1).item() < 0.6:
        k = 5
        sigma = 0.6 + torch.rand(1).item() * 0.8
        half = k // 2
        x = torch.arange(-half, half+1, device=DEVICE, dtype=torch.float32)
        g = torch.exp(-x*x/(2*sigma*sigma))
        g = g / g.sum()
        for kernel in [g.view(1,1,k,1).expand(c,1,k,1), g.view(1,1,1,k).expand(c,1,1,k)]:
            p = p.view(1, c, h, w)
            if kernel.shape[2] == k:
                p = F.conv2d(p, kernel, padding=(half, 0), groups=c)
            else:
                p = F.conv2d(p, kernel, padding=(0, half), groups=c)
            p = p.view(c, h, w)

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


def fft_signature_loss(features, human_sigs, layers_of_interest, alpha=1.0):
    """
    For each layer, compute FFT of mean activation map and match the human signature.
    """
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {}
    count = 0
    for n in layers_of_interest:
        if n not in human_sigs or n not in features:
            continue
        f = features[n]  # (1, C, H, W)
        target = human_sigs[n]
        h, w = target.shape

        # Current activation FFT
        f_avg = f.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        fft = torch.fft.fft2(f_avg, dim=(-2, -1))
        mag = torch.fft.fftshift(fft.abs(), dim=(-2, -1)).squeeze()  # (H, W)
        mag = mag / (mag.max() + 1e-8)

        # Resize target if needed
        if target.shape != mag.shape:
            target_resized = F.interpolate(target.unsqueeze(0).unsqueeze(0), size=mag.shape, mode='bilinear', align_corners=True).squeeze()
        else:
            target_resized = target

        diff = (mag - target_resized).abs().mean()
        loss = loss + alpha * diff
        stats[n] = diff.item()
        count += 1
    if count > 0:
        loss = loss / count
    return loss, stats


def channel_activation_loss(features, alpha=1.0):
    """Maximize mean activation of all channels at FPN layers."""
    total = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, name in FPN_LAYERS.items():
        if name not in features:
            continue
        f = features[name]
        act = f.mean()
        total = total + alpha * (-act)
        stats[f'act_{scale}'] = act.item()
    return total, stats


def purity_loss(pred):
    cls = pred[:, :, 5:85].sigmoid()
    other = torch.cat([cls[:, :, :PERSON], cls[:, :, PERSON+1:]], dim=2)
    topk_vals, _ = other.topk(min(20, other.shape[2]), dim=2)
    loss = topk_vals.mean()
    max_other = other.max().item()
    return loss, {'max_other': max_other}


def save_preview(patch, mask, epoch, out_dir, stats):
    p = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    m = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (p.shape[1]*2, p.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(p), (0,0))
    combined.paste(Image.fromarray(m).convert('RGB'), (p.shape[1],0))
    txt = f"e{epoch:04d} fft={stats.get('fft_mean',0):.3f} act={stats.get('act_mean',0):.2f} pure={stats.get('max_other',0):.2f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','fft_mean','act_mean','max_other','tv','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_fft=2.0, w_act=0.5, w_purity=0.3, w_tv=1.0,
          w_shape=500.0, area_limit=0.25, out_dir='outputs_v45'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_all_hooks(model)

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    layers_of_interest, human_sigs = compute_human_fft_signatures(model, features, person_dir)

    patch = init_from_human_fft(human_sigs, layers_of_interest)
    patch.requires_grad_(True)
    print("Init: merged inverse-FFT from unique human frequency signatures")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV45 Unique Human FFT Signature")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_fft={w_fft} w_act={w_act} w_purity={w_purity} w_tv={w_tv} w_shape={w_shape}\n")

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

        loss_fft, fft_stats = fft_signature_loss(features, human_sigs, layers_of_interest, alpha=w_fft)
        loss_act, act_stats = channel_activation_loss(features, alpha=w_act)
        loss_pur, pur_stats = purity_loss(out)
        loss_pur = w_purity * loss_pur
        loss_tv = w_tv * tv_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_fft + loss_act + loss_pur + loss_tv + loss_shape

        if torch.isnan(total) or torch.isinf(total):
            print(f"NaN/Inf at epoch {ep}, stopping. Best epoch {best_epoch}")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_([patch] + list(dap.parameters()), 3.0)
        opt.step()

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)
            dap.rays.clamp_(PATCH_SIZE*0.08, PATCH_SIZE*0.50)

        fft_vals = list(fft_stats.values())
        act_vals = list(act_stats.values())
        stats = {
            'fft_mean': np.mean(fft_vals) if fft_vals else 0,
            'act_mean': np.mean(act_vals) if act_vals else 0,
            'max_other': pur_stats['max_other'],
            'tv': loss_tv.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['fft_mean'], stats['act_mean'], stats['max_other'],
                stats['tv'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('FFT/mean', stats['fft_mean'], ep)
        writer.add_scalar('Activations/mean', stats['act_mean'], ep)
        writer.add_scalar('Purity/max_other', stats['max_other'], ep)
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
              f"fft={stats['fft_mean']:.3f} | act={stats['act_mean']:.3f} | "
              f"pure={stats['max_other']:.3f} | tv={stats['tv']:.3f} | area={area.item():.3f} | best={best_epoch}")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_fft', type=float, default=2.0)
    parser.add_argument('--w_act', type=float, default=0.5)
    parser.add_argument('--w_purity', type=float, default=0.3)
    parser.add_argument('--w_tv', type=float, default=1.0)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--out', type=str, default='outputs_v45')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_fft, args.w_act, args.w_purity,
          args.w_tv, args.w_shape, args.area_limit, args.out)
