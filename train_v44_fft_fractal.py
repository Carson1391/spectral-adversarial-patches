#!/usr/bin/env python3
"""
v44 — FFT-guided Fractal Human patch.

Idea:
1. Forward person images through YOLOv3 FPN layers.
2. For each person channel at each scale, compute average FFT magnitude of its
   activation map.
3. This gives us a "frequency fingerprint" of what excites the model's person
   detectors.
4. Synthesize the patch as a sum of sinusoids / inverse-FFT from those dominant
   frequencies, then fine-tune against the actual channel activation loss.

The patch starts in frequency space, so it naturally has the spatial
frequencies the model associates with humans.
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
    modules = dict(model.named_modules())
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    hooks = []
    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))
    return features, hooks


def load_person_crops(dir_path, max_imgs=100):
    if not os.path.exists(dir_path):
        return []
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg','.png'))])
    files = files[:max_imgs]
    imgs = []
    for f in files:
        img = Image.open(os.path.join(dir_path, f)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(np.array(img).transpose(2, 0, 1) / 255.0)
    return [torch.tensor(x, dtype=torch.float32, device=DEVICE) for x in imgs]


def probe_person_channels_and_frequencies(model, features, crops, topk=60):
    """
    Returns:
      channels: dict scale -> list of channel indices
      freqs: dict scale -> (B, H, W) average FFT magnitude of top channels
    """
    print(f"Probing person channels + frequencies from {len(crops)} crops")
    acts = {n: [] for n in FPN_LAYERS.values()}
    fft_mags = {n: [] for n in FPN_LAYERS.values()}

    for i in range(0, len(crops), 8):
        batch = torch.stack(crops[i:i+8])
        features.clear()
        with torch.no_grad():
            _ = model(batch)
        for n in FPN_LAYERS.values():
            if n not in features:
                continue
            f = features[n]  # (B, C, H, W)
            acts[n].append(f.mean(dim=(2, 3)).cpu().numpy())
            # FFT magnitude per channel per image
            fft = torch.fft.fft2(f, dim=(-2, -1))
            mag = torch.fft.fftshift(fft.abs(), dim=(-2, -1))
            fft_mags[n].append(mag.cpu().numpy())

    acts = {n: np.concatenate(a, axis=0) for n, a in acts.items() if len(a) > 0}
    fft_mags = {n: np.concatenate(a, axis=0) for n, a in fft_mags.items() if len(a) > 0}

    channels = {}
    freq_targets = {}
    for scale, name in FPN_LAYERS.items():
        mean = acts[name].mean(axis=0)
        std = acts[name].std(axis=0)
        score = mean + std
        top = np.argsort(score)[::-1][:topk]
        channels[scale] = [int(x) for x in top]
        print(f"{scale}: top channels = {channels[scale][:10]}...")

        # Average FFT magnitude over top person channels
        top_mags = fft_mags[name][:, channels[scale], :, :].mean(axis=0).mean(axis=0)
        top_mags = top_mags / (top_mags.max() + 1e-8)
        freq_targets[scale] = torch.tensor(top_mags, dtype=torch.float32, device=DEVICE)
        print(f"{scale}: freq target shape {freq_targets[scale].shape}")
    return channels, freq_targets


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


def fft_texture_init(freq_targets, scale='s16'):
    """
    Initialize patch texture from the frequency fingerprint of a person channel.
    freq_targets[scale] has shape (H, W).
    We resize to PATCH_SIZE and use as magnitude spectrum, with random phase.
    """
    target = freq_targets[scale].cpu().numpy()
    h, w = target.shape
    # Make it patch-sized
    target_big = np.array(Image.fromarray((target * 255).astype(np.uint8)).resize((PATCH_SIZE, PATCH_SIZE)))
    mag = target_big / 255.0

    # Random phase
    phase = np.random.rand(PATCH_SIZE, PATCH_SIZE) * 2 * np.pi
    complex_spec = mag * np.exp(1j * phase)

    # Inverse FFT
    img = np.fft.ifft2(np.fft.ifftshift(complex_spec)).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Stack to RGB
    img_rgb = np.stack([img, img, img], axis=0)
    return torch.tensor(img_rgb, dtype=torch.float32, device=DEVICE)


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


def fft_loss(patch, freq_targets, alpha=0.5):
    """
    Encourage the patch's FFT magnitude to match the person-channel frequency
    fingerprint.
    """
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, target_mag in freq_targets.items():
        # Resize target to patch size
        target_resized = F.interpolate(target_mag.unsqueeze(0).unsqueeze(0),
                                       size=(PATCH_SIZE, PATCH_SIZE),
                                       mode='bilinear', align_corners=True).squeeze()
        patch_gray = patch.mean(dim=0)  # (H, W)
        fft = torch.fft.fft2(patch_gray)
        mag = torch.fft.fftshift(fft.abs())
        mag = mag / (mag.max() + 1e-8)
        diff = (mag - target_resized).abs().mean()
        loss = loss + alpha * diff
        stats[f'fft_{scale}'] = diff.item()
    return loss, stats


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
    txt = f"e{epoch:04d} s8:{stats.get('act_s8',0):.2f} s16:{stats.get('act_s16',0):.2f} s32:{stats.get('act_s32',0):.2f} pure:{stats.get('max_other',0):.2f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','act_s8','act_s16','act_s32','fft_s8','fft_s16','fft_s32','purity','tv','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_act=1.0, w_fft=0.5, w_purity=0.3, w_tv=1.0,
          w_shape=500.0, area_limit=0.25, topk=60, out_dir='outputs_v44'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_hooks(model, list(FPN_LAYERS.values()))

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    crops = load_person_crops(person_dir, max_imgs=100)
    channels, freq_targets = probe_person_channels_and_frequencies(model, features, crops, topk=topk)

    # FFT-based initialization from s16 frequency fingerprint
    patch = fft_texture_init(freq_targets, scale='s16')
    patch.requires_grad_(True)
    print("Init: FFT texture from person-channel frequency fingerprint")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV44 FFT-guided Fractal Human")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_act={w_act} w_fft={w_fft} w_purity={w_purity} w_tv={w_tv} w_shape={w_shape}\n")

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
        loss_fft, fft_stats = fft_loss(patch, freq_targets, alpha=w_fft)
        loss_pur, pur_stats = purity_loss(out)
        loss_pur = w_purity * loss_pur
        loss_tv = w_tv * tv_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_act + loss_fft + loss_pur + loss_tv + loss_shape

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
            'fft_s8': fft_stats.get('fft_s8', 0),
            'fft_s16': fft_stats.get('fft_s16', 0),
            'fft_s32': fft_stats.get('fft_s32', 0),
            'max_other': pur_stats['max_other'],
            'tv': loss_tv.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['act_s8'], stats['act_s16'], stats['act_s32'],
                stats['fft_s8'], stats['fft_s16'], stats['fft_s32'],
                stats['max_other'], stats['tv'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Activations/s8', stats['act_s8'], ep)
        writer.add_scalar('Activations/s16', stats['act_s16'], ep)
        writer.add_scalar('Activations/s32', stats['act_s32'], ep)
        writer.add_scalar('FFT/s8', stats['fft_s8'], ep)
        writer.add_scalar('FFT/s16', stats['fft_s16'], ep)
        writer.add_scalar('FFT/s32', stats['fft_s32'], ep)
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
              f"s8={stats['act_s8']:5.2f} s16={stats['act_s16']:5.2f} s32={stats['act_s32']:5.2f} | "
              f"fft={stats['fft_s16']:.3f} | pure={stats['max_other']:.3f} | tv={stats['tv']:.3f} | area={area.item():.3f} | best={best_epoch}")

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
    parser.add_argument('--w_act', type=float, default=1.0)
    parser.add_argument('--w_fft', type=float, default=0.5)
    parser.add_argument('--w_purity', type=float, default=0.3)
    parser.add_argument('--w_tv', type=float, default=1.0)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=60)
    parser.add_argument('--out', type=str, default='outputs_v44')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_act, args.w_fft, args.w_purity,
          args.w_tv, args.w_shape, args.area_limit, args.topk, args.out)
