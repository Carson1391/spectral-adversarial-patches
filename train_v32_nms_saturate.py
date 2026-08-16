#!/usr/bin/env python3
"""
v32 — NMS Saturation / "Fractal Human" Patch.

Objective: Make the camera see an overwhelming number of overlapping
high-confidence person detections, choking the edge-node NMS/tracker.

Strategy:
- Pure Class 0 (person) signal only — no meter, no class flipping.
- Maximize activations of probed person-specific channels at all 3 YOLO scales.
- Maximize objectness × person-class confidence across many anchors.
- Heavy TV loss (lambda ~3) for smooth, camera-robust color blocks.
- DAP triangle + strong TPS/EoT fabric deformation.
- Structured geometric init (rings/chevrons), not random noise.
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
PERSON = 0
IMG_SIZE = 416
PATCH_SIZE = 224
N_RAYS = 32

FPN_LAYERS = {
    's32': 'module_list.80.leaky_80',
    's16': 'module_list.92.leaky_92',
    's8':  'module_list.104.leaky_104',
}

# Person-specific channels (probed from real person images)
PERSON_CHANNELS = {
    's32': [589, 469, 226, 344, 722, 538, 423, 25, 1014, 104, 640, 157, 24, 274, 869, 571, 892, 269, 653, 453, 933, 702, 200, 737, 284, 823, 249, 37, 198, 217],
    's16': [432, 339, 19, 327, 209, 409, 158, 90, 47, 507, 259, 369, 174, 384, 505, 122, 185, 334, 125, 147, 59, 156, 441, 473, 9, 421, 148, 74, 227, 43],
    's8':  [165, 70, 112, 71, 181, 74, 166, 246, 157, 146, 191, 37, 30, 82, 137, 240, 92, 57, 108, 49, 194, 132, 59, 67, 16, 84, 23, 226, 50, 218],
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


def eot_transform(patch, mask):
    """TPS + affine + blur + brightness/contrast/gamma."""
    c, h, w = patch.shape

    # TPS wrinkle simulation
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = 0.06 * torch.sin(2 * math.pi * ys * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    oy = 0.06 * torch.cos(2 * math.pi * xs * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    p = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

    # Affine
    angle = (torch.rand(1, device=DEVICE) - 0.5) * 0.8
    scale = 0.85 + torch.rand(1, device=DEVICE) * 0.3
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    p = F.grid_sample(p.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(m.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

    # Blur (camera/printing)
    if torch.rand(1).item() < 0.5:
        k = 5
        sigma = 0.8 + torch.rand(1).item()
        half = k // 2
        x = torch.arange(-half, half+1, device=DEVICE, dtype=torch.float32)
        g = torch.exp(-x*x/(2*sigma*sigma))
        g = g / g.sum()

        # Vertical blur: kernel (C,1,k,1)
        kv = g.view(1, 1, k, 1).expand(c, 1, k, 1)
        p = p.view(1, c, h, w)
        p = F.conv2d(p, kv, padding=(half, 0), groups=c)
        p = p.view(c, h, w)

        # Horizontal blur: kernel (C,1,1,k)
        kh = g.view(1, 1, 1, k).expand(c, 1, 1, k)
        p = p.view(1, c, h, w)
        p = F.conv2d(p, kh, padding=(0, half), groups=c)
        p = p.view(c, h, w)

    # Color jitter
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.25
    p = (p + bright).clamp(0, 1)
    contrast = 0.8 + torch.rand(1, device=DEVICE) * 0.4
    mean = p.mean()
    p = ((p - mean) * contrast + mean).clamp(0, 1)
    gamma = 0.85 + torch.rand(1, device=DEVICE) * 0.3
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
    """Total variation for smooth, camera-robust patterns."""
    tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def person_activation_loss(features, channels, alpha=1.0):
    total = torch.tensor(0.0, device=DEVICE)
    stats = {}
    for scale, name in FPN_LAYERS.items():
        if name not in features:
            continue
        f = features[name]
        chs = channels[scale]
        # maximize mean activation of targeted channels
        act = f[0, chs, :, :].mean()
        total = total + alpha * (-act)
        stats[f'act_{scale}'] = act.item()
    return total, stats


def nms_saturate_loss(preds, topk=200):
    """
    Maximize objectness * person_conf across top-K anchors.
    This pushes many anchors to fire as 'person', choking NMS.
    """
    obj_sig = preds[:, :, 4].sigmoid()
    cls_sig = preds[:, :, 5:85].sigmoid()
    person_conf = obj_sig * cls_sig[:, :, PERSON]

    # Top-K highest person confidences
    if person_conf.numel() > topk:
        vals, _ = person_conf.view(-1).topk(topk)
    else:
        vals = person_conf.view(-1)
    return -vals.mean()


def save_preview(patch, mask, epoch, out_dir, stats):
    p = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    m = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (p.shape[1]*2, p.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(p), (0,0))
    combined.paste(Image.fromarray(m).convert('RGB'), (p.shape[1],0))
    txt = f"e{epoch:04d} s8:{stats.get('act_s8',0):.2f} s16:{stats.get('act_s16',0):.2f} s32:{stats.get('act_s32',0):.2f} sat:{stats.get('sat',0):.3f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','act_s8','act_s16','act_s32','sat','tv','nps','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_act=1.0, w_sat=2.0, w_tv=3.0, w_nps=0.1,
          w_shape=500.0, area_limit=0.25, init='rings', out_dir='outputs_v32'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()
    features, hooks = register_hooks(model, list(FPN_LAYERS.values()))
    print(f"Hooked {len(hooks)} FPN layers")
    print(f"Targeting {sum(len(v) for v in PERSON_CHANNELS.values())} person-specific channels")

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

    print(f"\nV32 NMS Saturation / Fractal Human")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_act={w_act} w_sat={w_sat} w_tv={w_tv} w_nps={w_nps} w_shape={w_shape}\n")
    print(f"  TensorBoard: tensorboard --logdir={tb_dir}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        aug_patch, aug_mask = eot_transform(masked_patch, mask)
        canvas = paste_on_canvas(aug_patch, aug_mask)

        features.clear()
        out = model(canvas.unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]

        loss_act, act_stats = person_activation_loss(features, PERSON_CHANNELS, alpha=w_act)
        loss_sat = w_sat * nms_saturate_loss(out)
        loss_tv = w_tv * tv_loss(patch)
        loss_n = w_nps * nps_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_act + loss_sat + loss_tv + loss_n + loss_shape

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
            'sat': loss_sat.item(),
            'tv': loss_tv.item(),
            'nps': loss_n.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['act_s8'], stats['act_s16'], stats['act_s32'], stats['sat'],
                stats['tv'], stats['nps'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Activations/s8', stats['act_s8'], ep)
        writer.add_scalar('Activations/s16', stats['act_s16'], ep)
        writer.add_scalar('Activations/s32', stats['act_s32'], ep)
        writer.add_scalar('Loss/sat', stats['sat'], ep)
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
              f"s8={stats['act_s8']:5.2f} s16={stats['act_s16']:5.2f} s32={stats['act_s32']:5.2f} | "
              f"sat={stats['sat']:6.3f} | tv={stats['tv']:.3f} | "
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
    parser.add_argument('--w_sat', type=float, default=2.0)
    parser.add_argument('--w_tv', type=float, default=3.0)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--init', type=str, default='rings', choices=['rings','noise'])
    parser.add_argument('--out', type=str, default='outputs_v32')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_act, args.w_sat, args.w_tv, args.w_nps,
          args.w_shape, args.area_limit, args.init, args.out)
