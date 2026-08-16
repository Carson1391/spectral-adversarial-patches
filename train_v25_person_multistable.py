#!/usr/bin/env python3
"""
v25 — Person-image multi-stable attribute poison.

Composite patch onto real person crops. The person box already exists.
The patch hijacks mid-backbone features to make attributes swing wildly
across augmented views while keeping the person box.

Loss:
  L = alpha * ||feat||_2         (strong feature response)
    - lambda * Var(feat_views)   (inconsistent across views)
    - beta * obj                 (hold person box)
    + TV + NPS + shape
"""
import os, sys, math, argparse, csv, random
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
PATCH_SIZE = 224
N_RAYS = 32


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


def quantize_ste(patch, palette=PALETTE):
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    idx = torch.argmin(torch.cdist(p, palette), dim=1)
    q = palette[idx].reshape(patch.shape[1], patch.shape[2], 3).permute(2, 0, 1)
    return patch + (q - patch).detach()


def nps_loss(patch, palette=PALETTE):
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def load_person_imgs(dir_path, max_imgs=100):
    files = []
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.jpg', '.png')):
            files.append(os.path.join(dir_path, f))
    files = files[:max_imgs]
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(np.array(img).transpose(2,0,1) / 255.0)
    return [torch.tensor(x, dtype=torch.float32, device=DEVICE) for x in imgs]


def paste_patch(background, patch, mask, y0, x0):
    """Composite patch onto background using mask."""
    canvas = background.clone()
    h, w = patch.shape[1], patch.shape[2]
    m3 = mask.unsqueeze(0)  # (1, h, w)
    region = canvas[:, y0:y0+h, x0:x0+w]
    region = region * (1 - m3) + patch * m3
    canvas[:, y0:y0+h, x0:x0+w] = region
    return canvas


def augment_view(patch, mask, background):
    # Random TPS
    c, h, w = patch.shape
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = 0.04 * torch.sin(2 * math.pi * ys * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    oy = 0.04 * torch.cos(2 * math.pi * xs * 2 + torch.rand(1, device=DEVICE)*2*math.pi)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    p = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze(0)
    m = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                      padding_mode='border', align_corners=True).squeeze()

    # Random rotation + scale
    angle = (torch.rand(1, device=DEVICE) - 0.5) * 1.0
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

    # Color jitter
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.2
    p = (p + bright).clamp(0, 1)
    contrast = 0.85 + torch.rand(1, device=DEVICE) * 0.3
    mean = p.mean()
    p = ((p - mean) * contrast + mean).clamp(0, 1)
    gamma = 0.9 + torch.rand(1, device=DEVICE) * 0.2
    p = p ** gamma

    p = quantize_ste(p)

    # Random paste position on torso
    bg_h, bg_w = background.shape[1], background.shape[2]
    y0 = int(bg_h * (0.25 + torch.rand(1, device=DEVICE).item() * 0.25))
    x0 = int((bg_w - w) / 2 + (torch.rand(1, device=DEVICE).item() - 0.5) * 40)
    x0 = max(0, min(bg_w - w, x0))
    y0 = max(0, min(bg_h - h, y0))

    canvas = paste_patch(background, p, m, y0, x0)
    return canvas


def bilateral_tv(patch):
    tv_h = (patch[:, :, 4:] - patch[:, :, :-4]).abs().mean()
    tv_w = (patch[:, 4:, :] - patch[:, :-4, :]).abs().mean()
    return tv_h + tv_w


def feature_loss(feats_list, alpha=1.0, lam=2.0):
    pooled = []
    for f in feats_list:
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        pooled.append(f)
    stack = torch.stack(pooled, dim=0).squeeze(1)  # (V, C)
    conf = stack.norm(dim=-1).mean()
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


def save_preview(patch, mask, epoch, out_dir, stats, bg):
    img = (patch.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    msk = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    bg_img = (bg.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    combined = Image.new('RGB', (img.shape[1]*3, img.shape[0]+24), (32,32,32))
    combined.paste(Image.fromarray(img), (0,0))
    combined.paste(Image.fromarray(msk).convert('RGB'), (img.shape[1],0))
    combined.paste(Image.fromarray(bg_img), (img.shape[1]*2,0))
    txt = f"e{epoch:04d} conf:{stats['conf']:.2f} var:{stats['var']:.2f} obj:{stats['obj']:.3f}"
    draw = ImageDraw.Draw(combined)
    draw.text((4, img.shape[0]+4), txt, fill=(0,255,0))
    combined.save(os.path.join(out_dir, 'preview_best.png'))


def setup_csv(out_dir):
    path = os.path.join(out_dir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','time','conf','var','obj','tv','nps','shape','total'])
    return path


def train(epochs=800, lr=0.01, n_views=4, alpha=1.0, lam=2.0,
          w_obj=0.5, w_tv=0.1, w_nps=0.1, w_shape=5000.0, area_limit=0.25,
          init='rings', out_dir='outputs_v25', max_imgs=50):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)

    model = load_frozen_model()

    layer_names = ['module_list.12.leaky_12', 'module_list.37.leaky_37']
    features, hooks = register_hooks(model, layer_names)
    print(f"Hooked {len(hooks)} mid-backbone layers")

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    person_imgs = load_person_imgs(person_dir, max_imgs)
    print(f"Loaded {len(person_imgs)} person images")

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

    print(f"\nV25 Person-Image Multi-Stable Attribute Poison")
    print(f"  epochs={epochs} lr={lr} n_views={n_views}")
    print(f"  alpha={alpha} lambda={lam} w_obj={w_obj}")
    print(f"  w_tv={w_tv} w_nps={w_nps} w_shape={w_shape}\n")

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        mask = dap.forward()
        masked_patch = patch * mask.unsqueeze(0)

        bg = random.choice(person_imgs)

        preds_list = []
        feats_list = []
        for _ in range(n_views):
            canvas = augment_view(masked_patch, mask, bg)
            canvas = canvas.unsqueeze(0)  # (1, 3, 416, 416)

            features.clear()
            out = model(canvas)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds_list.append(out)

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
            save_preview(patch, mask, ep, out_dir, stats, bg)
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
    parser.add_argument('--n_views', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--w_obj', type=float, default=0.5)
    parser.add_argument('--w_tv', type=float, default=0.1)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_shape', type=float, default=5000.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--init', type=str, default='rings', choices=['rings','noise'])
    parser.add_argument('--out', type=str, default='outputs_v25')
    parser.add_argument('--max_imgs', type=int, default=50)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.n_views, args.alpha, args.lam,
          args.w_obj, args.w_tv, args.w_nps, args.w_shape, args.area_limit,
          args.init, args.out, args.max_imgs)
