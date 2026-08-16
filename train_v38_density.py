#!/usr/bin/env python3
"""
v38 — Feature Density Saturation with balanced losses.

Key insight from v37: objectness logits collapsed to 0 because losses were
unbalanced. This version explicitly pushes objectness AND person class
probability up on many anchors, then rewards spatial scatter of centers.

Loss components:
  1. obj_loss: -mean(sigmoid(obj_logit)) — push many anchors to fire
  2. cls_loss: -mean(sigmoid(person_cls)) — make them person class
  3. conf_loss: -mean(obj * person_cls) — joint confidence volume
  4. scatter_loss: variance of predicted box centers, weighted by confidence
  5. TV + NPS + shape regularizers
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

STRIDES = [32, 16, 8]
GRID_SIZES = [13, 26, 52]


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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


def init_chevrons(size=224, n_chevrons=10):
    img = Image.new('RGB', (size, size), (255,255,255))
    draw = ImageDraw.Draw(img)
    colors = [(0,0,0),(255,0,0),(0,0,255),(255,255,0),(0,255,0),(128,0,128)]
    step = size // n_chevrons
    for i, y in enumerate(range(0, size, step)):
        color = colors[i % len(colors)]
        draw.polygon([(0,y+step//2), (size//2,y), (size,y+step//2), (size,y+step), (size//2,y+step//2), (0,y+step)], fill=color)
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
    c, h, w = patch.shape

    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = 0.08 * torch.sin(2 * math.pi * ys * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    oy = 0.08 * torch.cos(2 * math.pi * xs * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    p = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

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

    if torch.rand(1).item() < 0.5:
        k = 5
        sigma = 0.8 + torch.rand(1).item()
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
    tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def decode_to_grids(pred):
    """Return list of (scale_tensor, stride, grid) where scale_tensor is (B,A,H,W,85)."""
    B = pred.shape[0]
    outputs = []
    offset = 0
    for stride, grid in zip(STRIDES, GRID_SIZES):
        n = grid * grid * 3
        scale = pred[:, offset:offset+n, :].view(B, grid, grid, 3, 85)
        scale = scale.permute(0, 3, 1, 2, 4)
        outputs.append((scale, stride, grid))
        offset += n
    return outputs


def density_saturation_loss(pred):
    """
    Push objectness and person class up on many anchors.
    Reward spatial scatter of predicted box centers.
    """
    all_obj = []
    all_cls = []
    all_conf = []
    all_cx = []
    all_cy = []

    for scale, stride, grid in decode_to_grids(pred):
        B, A, H, W, D = scale.shape
        tx = scale[..., 0]
        ty = scale[..., 1]
        tw = scale[..., 2]
        th = scale[..., 3]
        obj_logit = scale[..., 4]
        cls_logits = scale[..., 5:85]

        obj = obj_logit.sigmoid()
        person_cls = cls_logits[..., PERSON].sigmoid()
        conf = obj * person_cls

        # Box centers in image coords
        cx_grid = torch.arange(W, device=DEVICE, dtype=torch.float32).view(1, 1, 1, W)
        cy_grid = torch.arange(H, device=DEVICE, dtype=torch.float32).view(1, 1, H, 1)
        bx = (tx.sigmoid() + cx_grid) * stride
        by = (ty.sigmoid() + cy_grid) * stride

        all_obj.append(obj)
        all_cls.append(person_cls)
        all_conf.append(conf)
        all_cx.append(bx)
        all_cy.append(by)

    obj = torch.cat([x.reshape(-1) for x in all_obj])
    cls = torch.cat([x.reshape(-1) for x in all_cls])
    conf = torch.cat([x.reshape(-1) for x in all_conf])
    cx = torch.cat([x.reshape(-1) for x in all_cx])
    cy = torch.cat([y.reshape(-1) for y in all_cy])

    # Primary: push objectness up on all anchors (this is the "attention" signal)
    loss_obj = -obj.mean()

    # Secondary: push person class up
    loss_cls = -cls.mean()

    # Tertiary: push joint confidence up (volume)
    loss_conf = -conf.mean()

    # Scatter: weighted variance of box centers (spread the boxes)
    weights = conf.clamp(min=1e-6)
    weights = weights / weights.sum()
    mean_cx = (weights * cx).sum()
    mean_cy = (weights * cy).sum()
    var_cx = (weights * (cx - mean_cx) ** 2).sum()
    var_cy = (weights * (cy - mean_cy) ** 2).sum()
    loss_scatter = -(var_cx + var_cy)

    total = loss_obj + loss_cls + loss_conf + 0.001 * loss_scatter

    stats = {
        'obj': obj.mean().item(),
        'cls': cls.mean().item(),
        'conf_mean': conf.mean().item(),
        'max_conf': conf.max().item(),
        'scatter': (var_cx + var_cy).item(),
        'conf_50': (conf > 0.5).float().sum().item(),
        'conf_70': (conf > 0.7).float().sum().item(),
        'conf_90': (conf > 0.9).float().sum().item(),
    }
    return total, stats


def save_preview(patch, mask, epoch, out_dir, stats):
    p = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    m = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (p.shape[1]*2, p.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(p), (0,0))
    combined.paste(Image.fromarray(m).convert('RGB'), (p.shape[1],0))
    txt = f"e{epoch:04d} obj={stats.get('obj',0):.3f} cls={stats.get('cls',0):.3f} max={stats.get('max_conf',0):.3f} >0.5:{stats.get('conf_50',0):.0f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, p.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','obj','cls','conf_mean','max_conf','scatter','conf_50','conf_70','conf_90','tv','nps','shape','total'])
    return path


def train(epochs=800, lr=0.01, w_density=2.0, w_tv=3.0, w_nps=0.1,
          w_shape=500.0, area_limit=0.25, init='chevrons', out_dir='outputs_v38'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)
    tb_dir = os.path.join(out_dir, 'tb_logs')
    writer = SummaryWriter(tb_dir)

    model = load_frozen_model()

    if init == 'chevrons':
        patch = init_chevrons(PATCH_SIZE)
    else:
        patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    patch.requires_grad_(True)
    print(f"Init: {init}")

    dap = DAPMask(size=PATCH_SIZE, n_rays=N_RAYS).to(DEVICE)
    opt = torch.optim.Adam([patch] + list(dap.parameters()), lr=lr, betas=(0.5, 0.999))

    best_total = float('inf')
    best_epoch = 0

    print(f"\nV38 Density Saturation (balanced)")
    print(f"  epochs={epochs} lr={lr}")
    print(f"  w_density={w_density} w_tv={w_tv} w_nps={w_nps} w_shape={w_shape}\n")
    print(f"  TensorBoard: tensorboard --logdir={tb_dir}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        aug_patch, aug_mask = eot_transform(masked_patch, mask)
        canvas = paste_on_canvas(aug_patch, aug_mask)

        out = model(canvas.unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]

        loss_density, dens_stats = density_saturation_loss(out)
        loss_density = w_density * loss_density
        loss_tv = w_tv * tv_loss(patch)
        loss_n = w_nps * nps_loss(patch)
        area = dap.area() / (PATCH_SIZE * PATCH_SIZE)
        loss_shape = w_shape * F.relu(area - area_limit)

        total = loss_density + loss_tv + loss_n + loss_shape

        if torch.isnan(total) or torch.isinf(total):
            print(f"NaN/Inf at epoch {ep}, stopping. Best epoch {best_epoch}")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_([patch] + list(dap.parameters()), 2.0)
        opt.step()

        with torch.no_grad():
            patch.clamp_(min=1e-3, max=1.0-1e-3)
            dap.rays.clamp_(PATCH_SIZE*0.1, PATCH_SIZE*0.5)

        stats = {
            'obj': dens_stats['obj'],
            'cls': dens_stats['cls'],
            'conf_mean': dens_stats['conf_mean'],
            'max_conf': dens_stats['max_conf'],
            'scatter': dens_stats['scatter'],
            'conf_50': dens_stats['conf_50'],
            'conf_70': dens_stats['conf_70'],
            'conf_90': dens_stats['conf_90'],
            'tv': loss_tv.item(),
            'nps': loss_n.item(),
            'shape': loss_shape.item(),
            'total': total.item(),
        }

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, datetime.now().isoformat(),
                stats['obj'], stats['cls'], stats['conf_mean'], stats['max_conf'], stats['scatter'],
                stats['conf_50'], stats['conf_70'], stats['conf_90'],
                stats['tv'], stats['nps'], stats['shape'], stats['total']])

        writer.add_scalar('Loss/total', stats['total'], ep)
        writer.add_scalar('Density/obj', stats['obj'], ep)
        writer.add_scalar('Density/cls', stats['cls'], ep)
        writer.add_scalar('Density/conf_mean', stats['conf_mean'], ep)
        writer.add_scalar('Density/max_conf', stats['max_conf'], ep)
        writer.add_scalar('Density/scatter', stats['scatter'], ep)
        writer.add_scalar('Density/conf_50', stats['conf_50'], ep)
        writer.add_scalar('Density/conf_70', stats['conf_70'], ep)
        writer.add_scalar('Density/conf_90', stats['conf_90'], ep)
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
              f"obj={stats['obj']:.3f} | cls={stats['cls']:.3f} | "
              f"conf={stats['conf_mean']:.3f} | max={stats['max_conf']:.3f} | "
              f">0.5:{stats['conf_50']:4.0f} >0.7:{stats['conf_70']:4.0f} >0.9:{stats['conf_90']:4.0f} | "
              f"tv={stats['tv']:.3f} | area={area.item():.3f} | best={best_epoch}")

    final = (patch.detach() * mask.detach()).clamp(0, 1)
    final_np = (final.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(final_np).save(os.path.join(out_dir, 'patch_final.png'))
    if os.path.exists(os.path.join(out_dir, 'checkpoint_best.png')):
        import shutil
        shutil.copy(os.path.join(out_dir, 'checkpoint_best.png'),
                    os.path.join(out_dir, 'patch_final.png'))

    writer.close()
    print(f"Done: {out_dir}")
    print(f"TensorBoard: tensorboard --logdir={tb_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_density', type=float, default=2.0)
    parser.add_argument('--w_tv', type=float, default=3.0)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=500.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--init', type=str, default='chevrons', choices=['chevrons','noise'])
    parser.add_argument('--out', type=str, default='outputs_v38')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.w_density, args.w_tv, args.w_nps,
          args.w_shape, args.area_limit, args.init, args.out)
