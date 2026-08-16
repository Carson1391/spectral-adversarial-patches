#!/usr/bin/env python3
"""
v46 — Per-channel FFT subtraction for unique human frequency ID.

Ryan's corrected idea:
1. Forward a person image through the network.
2. For EVERY CNN layer and EVERY CHANNEL, compute real FFT (rfft2) of the
   activation map.
3. Forward the same image with the human removed (background) through the network.
4. Compute rfft2 of background activations for every layer/channel.
5. Subtract: human_fft_id = person_fft - background_fft per layer, per channel.
6. This gives the unique human frequency signature for each channel.
7. Optimize the patch so the patch-induced activations match these human
   frequency signatures.
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


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def register_all_hooks(model):
    """Register forward hooks on all convolutional layers to capture every activation map."""
    features = {}
    hooks = []
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
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


def collect_per_channel_rfft(model, features, imgs, layers_of_interest):
    """
    Collect per-channel rfft2 magnitude for each layer, averaged over images.
    Returns dict: layer_name -> (C, H_f, W_f) tensor.
    """
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
            # Per-channel real FFT
            fft = torch.fft.rfft2(f, dim=(-2, -1))
            mag = fft.abs()  # (B, C, H_f, W_f)
            sigs[n].append(mag.cpu().numpy())

    out = {}
    for n in layers_of_interest:
        if len(sigs[n]) == 0:
            continue
        arr = np.concatenate(sigs[n], axis=0)  # (N, C, H_f, W_f)
        out[n] = torch.tensor(arr.mean(axis=0), dtype=torch.float32, device=DEVICE)
    return out


def compute_human_fft_ids(model, features, person_dir, bg_dir=None):
    print("Computing per-channel human FFT IDs...")
    person_imgs = load_imgs(person_dir, max_imgs=100)
    if bg_dir is not None:
        bg_imgs = load_imgs(bg_dir, max_imgs=100)
    else:
        bg_imgs = [torch.full((3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE) for _ in range(50)]

    # Determine spatial layers
    dummy = torch.full((1, 3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
    features.clear()
    with torch.no_grad():
        _ = model(dummy)
    layers_of_interest = [n for n, t in features.items()
                          if len(t.shape) == 4 and t.shape[2] >= 4 and t.shape[3] >= 4]
    print(f"Found {len(layers_of_interest)} spatial conv layers")

    person_sigs = collect_per_channel_rfft(model, features, person_imgs, layers_of_interest)
    bg_sigs = collect_per_channel_rfft(model, features, bg_imgs, layers_of_interest)

    human_ids = {}
    active_layers = []
    for n in layers_of_interest:
        if n not in person_sigs or n not in bg_sigs:
            continue
        diff = person_sigs[n] - bg_sigs[n]  # (C, H_f, W_f)
        diff = torch.maximum(diff, torch.zeros_like(diff))
        # Normalize per channel
        max_per_ch = diff.amax(dim=(1,2), keepdim=True).clamp(min=1e-8)
        diff = diff / max_per_ch
        # Keep only channels that have strong human-specific frequency
        channel_energy = diff.sum(dim=(1,2))
        if channel_energy.max() < 0.1:
            continue
        human_ids[n] = diff
        active_layers.append(n)

    print(f"Human FFT IDs computed for {len(active_layers)} layers")
    return active_layers, human_ids


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


def init_from_human_fft_ids(human_ids, patch_size=224):
    """
    Initialize patch by summing inverse-rfft2 of human FFT IDs across channels/layers.
    """
    acc = torch.zeros(3, patch_size, patch_size, device=DEVICE)
    count = 0
    # Use a sample of channels from a sample of layers
    layers = list(human_ids.keys())
    layers = layers[::max(1, len(layers)//8)]
    for n in layers:
        sig = human_ids[n]  # (C, H_f, W_f)
        n_ch = min(8, sig.shape[0])
        for ch in range(0, sig.shape[0], max(1, sig.shape[0]//n_ch)):
            mag = sig[ch].cpu().numpy()
            # Resize magnitude to patch spatial freq shape
            ph, pw = mag.shape
            target_h = patch_size
            target_w = patch_size // 2 + 1  # rfft2 width
            mag_big = np.array(Image.fromarray((mag * 255).astype(np.uint8)).resize((target_w, target_h)))
            mag_big = mag_big / 255.0
            phase = np.random.rand(target_h, target_w) * 2 * np.pi
            complex_spec = mag_big * np.exp(1j * phase)
            img = np.fft.irfft2(complex_spec, s=(target_h, target_h))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            for c in range(3):
                acc[c] += torch.tensor(img, dtype=torch.float32, device=DEVICE)
            count += 1
    if count > 0:
        acc = acc / count
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


def fft_id_loss(features, human_ids, active_layers, alpha=1.0):
    """
    For each layer/channel in human_ids, compute current activation's rfft2
    magnitude and match the human FFT ID.
    """
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {}
    count = 0
    for n in active_layers:
        if n not in features or n not in human_ids:
            continue
        f = features[n]  # (1, C, H, W)
        target = human_ids[n]  # (C, H_f, W_f)
        C, H_f, W_f = target.shape

        # Current per-channel rfft2
        fft = torch.fft.rfft2(f[0], dim=(-2, -1))
        mag = fft.abs()  # (C, H_f, W_f)

        # Normalize per channel
        max_per_ch = mag.amax(dim=(1,2), keepdim=True).clamp(min=1e-8)
        mag = mag / max_per_ch

        # Resize target if spatial sizes differ
        if mag.shape != target.shape:
            target = F.interpolate(target.unsqueeze(0), size=mag.shape[1:], mode='bilinear', align_corners=True).squeeze(0)

        diff = (mag - target).abs().mean()
        loss = loss + alpha * diff
        stats[n] = diff.item()
        count += 1
    if count > 0:
        loss = loss / count
    return loss, stats


def channel_activation_loss(features, alpha=1.0):
    """Maximize mean activation of all channels at key FPN layers."""
    fpn_layers = {
        's32': 'module_list.80.leaky_80',
        's16': 'module_list.92.leaky_92',
        's8':  'module_list.104.leaky_104',
    }
    total = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, name in fpn_layers.items():
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
          w_shape=500.0, area_limit=0.25, out_dir='outputs_v46'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_all_hooks(model)

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    active_layers, human_ids = compute_human_fft_ids(model, features, person_dir)

    patch = init_from_human_fft_ids(human_ids)
    patch.requires_grad_(True)
    print("Init: per-channel inverse-rfft2 from unique human FFT IDs")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV46 Per-Channel Human FFT ID")
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

        loss_fft, fft_stats = fft_id_loss(features, human_ids, active_layers, alpha=w_fft)
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
    parser.add_argument('--out', type=str, default='outputs_v46')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_fft, args.w_act, args.w_purity,
          args.w_tv, args.w_shape, args.area_limit, args.out)
