#!/usr/bin/env python3
"""
v47 — Train patch using only human-specific channels identified by 2D FFT analysis.

Pipeline:
1. Load precomputed 2D FFT analysis from fft_2d_analysis/summary.json
2. Classify channels as human-specific, shared, or inactive per layer
3. Hook those specific channels during training
4. Optimize patch to:
   - Maximize activation of human-specific channels
   - Match their 2D frequency signatures (FFT fingerprint)
   - Suppress activation of shared background channels (so patch doesn't just look like background)
   - Ignore inactive channels
"""
import os, sys, math, argparse, csv, json, random
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


def register_selective_hooks(model, layer_channel_map):
    """Hook only specific channels in specific conv layers."""
    features = {}
    hooks = []
    def make_hook(name, channels):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                # Store only selected channels to save memory
                if o.shape[1] >= max(channels) + 1:
                    features[name] = o[:, channels, :, :]
        return fn
    modules = dict(model.named_modules())
    for layer_name, channels in layer_channel_map.items():
        if layer_name in modules and len(channels) > 0:
            hooks.append(modules[layer_name].register_forward_hook(make_hook(layer_name, channels)))
    return features, hooks


def classify_channels(summary_path, human_threshold=2.0, min_score=5.0):
    """
    Read 2D FFT summary and classify channels.
    Returns: dict layer_name -> {'human': [...], 'shared': [...], 'inactive': [...]}
    """
    with open(summary_path, 'r') as f:
        data = json.load(f)

    classification = {}
    for entry in data:
        layer = entry['layer']
        scores = np.array(entry['top_scores'])
        channels = entry['top_channels']
        mean_diff = entry['mean_diff']
        std_diff = np.std(scores) if len(scores) > 1 else 0

        human = []
        shared = []
        inactive = []
        for ch, score in zip(channels, scores):
            if score > max(min_score, mean_diff + human_threshold * std_diff):
                human.append(ch)
            elif score < mean_diff * 0.3:
                inactive.append(ch)
            else:
                shared.append(ch)

        # Also include non-top channels as shared/inactive
        classification[layer] = {
            'human': human,
            'shared': shared,
            'inactive': inactive,
            'mean_diff': mean_diff
        }
        print(f"{layer}: human={len(human)} shared={len(shared)} inactive={len(inactive)} (mean_diff={mean_diff:.2f})")
    return classification


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


def human_channel_loss(features, classification, w_human=1.0, w_shared=-0.2):
    """
    Maximize activation of human-specific channels, suppress shared channels.
    """
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {'human_act': 0.0, 'shared_act': 0.0, 'n_human': 0, 'n_shared': 0}

    for layer_name, cls in classification.items():
        if layer_name not in features:
            continue
        f = features[layer_name]  # (1, K_selected, H, W)

        # We need to map back to channel indices. The hook stored selected channels
        # in order. We need the original full channel activations to select by class.
        # Instead, re-hook to store full activations for these layers.
        pass

    return loss, stats


# Simpler: just maximize all stored channels since we only hooked human+shared from top-K
# Let's rewrite with full-channel hooks for selected layers


def register_full_hooks(model, layer_names):
    features = {}
    hooks = []
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    modules = dict(model.named_modules())
    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))
    return features, hooks


def human_channel_loss_full(features, classification, w_human=1.0, w_shared=-0.2):
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {'human_act': 0.0, 'shared_act': 0.0, 'n_human': 0, 'n_shared': 0}

    for layer_name, cls in classification.items():
        if layer_name not in features:
            continue
        f = features[layer_name][0]  # (C, H, W)

        if len(cls['human']) > 0:
            human_f = f[cls['human']]
            human_act = human_f.mean()
            loss = loss + w_human * (-human_act)
            stats['human_act'] += human_act.item()
            stats['n_human'] += 1

        if len(cls['shared']) > 0:
            shared_f = f[cls['shared']]
            shared_act = shared_f.mean()
            loss = loss + w_shared * (-shared_act)  # suppress shared
            stats['shared_act'] += shared_act.item()
            stats['n_shared'] += 1

    if stats['n_human'] > 0:
        stats['human_act'] /= stats['n_human']
    if stats['n_shared'] > 0:
        stats['shared_act'] /= stats['n_shared']
    return loss, stats


def fft_signature_loss(features, classification, target_masks, w_fft=0.5):
    """
    Match 2D FFT of human-specific channels to the precomputed human frequency signatures.
    target_masks: dict layer_name -> (K, H, W) numpy arrays
    """
    loss = torch.tensor(0.0, device=DEVICE)
    stats = {}
    count = 0

    for layer_name, cls in classification.items():
        if layer_name not in features or layer_name not in target_masks:
            continue
        if len(cls['human']) == 0:
            continue
        f = features[layer_name][0]  # (C, H, W)
        human_f = f[cls['human']].unsqueeze(0)  # (1, K, H, W)

        fft = torch.fft.rfft2(human_f, dim=(-2, -1))
        mag = fft.abs().squeeze(0)  # (K, H_f, W_f)
        mag = mag / (mag.amax(dim=(1,2), keepdim=True) + 1e-8)

        target = torch.tensor(target_masks[layer_name], dtype=torch.float32, device=DEVICE)
        # target: (K, H_t, W_t). Resize to match mag: (K, H_f, W_f)
        if target.shape != mag.shape:
            target = target.unsqueeze(0).unsqueeze(0)  # (1, 1, K, H_t, W_t)
            target = F.interpolate(target, size=mag.shape, mode='trilinear', align_corners=True)
            target = target.squeeze(0).squeeze(0)  # (K, H_f, W_f)

        diff = (mag - target).abs().mean()
        loss = loss + w_fft * diff
        stats[layer_name] = diff.item()
        count += 1
    if count > 0:
        loss = loss / count
    return loss, stats


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
    txt = f"e{epoch:04d} h={stats.get('human_act',0):.2f} s={stats.get('shared_act',0):.2f} p={stats.get('max_other',0):.2f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','human_act','shared_act','fft','purity','tv','shape','total'])
    return path


def load_target_masks():
    path = os.path.join(BASE, 'fft_2d_analysis', 'target_frequency_masks.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


def train(epochs=800, lr=0.01, w_human=1.0, w_shared=-0.2, w_fft=0.5,
          w_purity=0.3, w_tv=1.0, w_shape=500.0, area_limit=0.25,
          summary_path=None, out_dir='outputs_v47'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    if summary_path is None:
        summary_path = os.path.join(BASE, 'fft_2d_analysis', 'summary.json')
    classification = classify_channels(summary_path)

    layer_names = list(classification.keys())
    model = load_frozen_model()
    features, hooks = register_full_hooks(model, layer_names)

    target_masks = load_target_masks()

    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    patch.requires_grad_(True)
    print("Init: pure white noise")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV47 Human-Specific Channel Attack")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_human={w_human} w_shared={w_shared} w_fft={w_fft} w_purity={w_purity} w_tv={w_tv}\n")

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

        loss_human, human_stats = human_channel_loss_full(features, classification, w_human=w_human, w_shared=w_shared)
        loss_fft, fft_stats = fft_signature_loss(features, classification, target_masks, w_fft=w_fft)
        loss_pur, pur_stats = purity_loss(out)
        loss_pur = w_purity * loss_pur
        loss_tv = w_tv * tv_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_human + loss_fft + loss_pur + loss_tv + loss_shape

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
        stats = {
            'human_act': human_stats['human_act'],
            'shared_act': human_stats['shared_act'],
            'fft': np.mean(fft_vals) if fft_vals else 0,
            'max_other': pur_stats['max_other'],
            'tv': loss_tv.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['human_act'], stats['shared_act'], stats['fft'],
                stats['max_other'], stats['tv'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Activations/human', stats['human_act'], ep)
        writer.add_scalar('Activations/shared', stats['shared_act'], ep)
        writer.add_scalar('FFT/mean', stats['fft'], ep)
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
              f"human={stats['human_act']:.3f} | shared={stats['shared_act']:.3f} | "
              f"fft={stats['fft']:.3f} | pure={stats['max_other']:.3f} | tv={stats['tv']:.3f} | best={best_epoch}")

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
    parser.add_argument('--w_human', type=float, default=1.0)
    parser.add_argument('--w_shared', type=float, default=-0.2)
    parser.add_argument('--w_fft', type=float, default=0.5)
    parser.add_argument('--w_purity', type=float, default=0.3)
    parser.add_argument('--w_tv', type=float, default=1.0)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--summary', type=str, default=None)
    parser.add_argument('--out', type=str, default='outputs_v47')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_human, args.w_shared, args.w_fft,
          args.w_purity, args.w_tv, args.w_shape, args.area_limit,
          args.summary, args.out)
