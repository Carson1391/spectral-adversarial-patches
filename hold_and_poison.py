#!/usr/bin/env python3
"""
hold_and_poison.py — Faithful Darknet YOLOv3 hold-and-poison trainer.

Hold the person box (maximize objectness) while pushing classification
toward a target class (parking meter). Uses FPN/Neck hooks, EoT/TPS,
and live CSV/plot logging.
"""
import os, sys, math, argparse, time, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Faithful Darknet YOLOv3 (NOT yolov3u) ─────────────────────────────── #
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")

DEVICE = torch.device('cuda')
PERSON = 0
DEFAULT_TARGET = 12  # parking meter
IMG_SIZE = 416
PATCH_SIZE = 224


# ─── 1. MODEL LOAD + FPN DISCOVERY ────────────────────────────────────── #

def load_frozen_model():
    """Load faithful Darknet YOLOv3, freeze weights."""
    print("Loading faithful Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params frozen")
    return model


def discover_fpn_layers(model):
    """
    Probe to find FPN/Neck output layers by spatial shape.
    At 416px input: stride 8→52x52, stride 16→26x26, stride 32→13x13.
    """
    target_shapes = {(52, 52), (26, 26), (13, 13)}
    temp_acts = {}
    temp_hooks = []

    def probe_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor) and o.dim() == 4:
                temp_acts[name] = tuple(o.shape[2:])
        return fn

    for n, m in model.named_modules():
        temp_hooks.append(m.register_forward_hook(probe_hook(n)))

    with torch.no_grad():
        _ = model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))

    for h in temp_hooks:
        h.remove()

    fpn_names = [n for n, s in temp_acts.items() if s in target_shapes]
    return fpn_names


def register_fpn_hooks(model, layer_names):
    """Register persistent forward hooks on FPN layers."""
    features = {}
    hooks = []

    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
            elif isinstance(o, (tuple, list)) and isinstance(o[0], torch.Tensor):
                features[name] = o[0]
        return fn

    modules = dict(model.named_modules())
    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))

    return features, hooks


# ─── 2. EoT TRANSFORMATIONS ───────────────────────────────────────────── #

def tps_warp(patch, intensity=0.04):
    """Thin Plate Spline approximation via sinusoidal displacement."""
    c, h, w = patch.shape
    ys = torch.linspace(-1, 1, h, device=DEVICE).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=DEVICE).view(1, -1)
    ox = intensity * torch.sin(2 * math.pi * ys * 2 + 0.7)
    oy = intensity * torch.cos(2 * math.pi * xs * 2 + 1.3)
    grid = torch.stack([xs.expand(h, w) + ox, ys.expand(h, w) + oy], dim=-1).unsqueeze(0)
    return F.grid_sample(
        patch.unsqueeze(0), grid, mode='bilinear',
        padding_mode='border', align_corners=True
    ).squeeze(0)


def eot_jitter(patch):
    """Random brightness/contrast jitter."""
    bright = (torch.rand(1, device=DEVICE) - 0.5) * 0.2
    patch = (patch + bright).clamp(0, 1)
    contrast = 0.85 + torch.rand(1, device=DEVICE) * 0.3
    mean = patch.mean()
    patch = ((patch - mean) * contrast + mean).clamp(0, 1)
    return patch


def random_affine(patch):
    """Small random rotation/scale for robustness."""
    c, h, w = patch.shape
    angle = (torch.rand(1, device=DEVICE) - 0.5) * 0.2  # radians
    scale = 0.9 + torch.rand(1, device=DEVICE) * 0.2
    theta = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle),  scale * math.cos(angle), 0]
    ], device=DEVICE, dtype=patch.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=True)
    return F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                         padding_mode='border', align_corners=True).squeeze(0)


# ─── 3. COMPOSITE LOSS ────────────────────────────────────────────────── #

def tv_loss(patch):
    tv_h = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    tv_w = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    return tv_h + tv_w


def nps_loss(patch):
    """Nearest-printable-color — pull toward 16-color palette."""
    palette = torch.tensor([
        [0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[0,255,255],[255,0,255],
        [128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],[192,192,192],[64,64,64]
    ], device=DEVICE, dtype=patch.dtype) / 255.0
    p = patch.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def composite_loss(preds, fpn_features, patch, target_class, alpha, beta, gamma, delta, fpn_w, topk=500):
    """
    Hold-and-Poison objective:
    - alpha * Loss_obj:  maximize objectness (HOLD the person box)
    - beta  * Loss_cls:  maximize target_class, minimize person_class (POISON semantic)
    - beta2 * Loss_other: penalize ALL other classes (NO SWITCHING)
    - gamma * Loss_TV:    smoothness / printability
    - delta * Loss_NPS:   printable colors
    - fpn_w * FPN_norm:   keep feature energy near baseline human

    preds: (1, 10647, 85) where [0:4]=box, [4]=obj, [5:85]=cls
    """
    obj_logit = preds[:, :, 4]      # (1, N)
    cls_logits = preds[:, :, 5:85]  # (1, N, 80)

    obj_sig = obj_logit.sigmoid()   # (1, N)
    cls_sig = cls_logits.sigmoid()  # (1, N, 80)

    # Combined confidence per anchor
    person_conf = obj_sig * cls_sig[:, :, PERSON]
    target_conf = obj_sig * cls_sig[:, :, target_class]
    other_mask = torch.ones(80, device=DEVICE, dtype=torch.bool)
    other_mask[PERSON] = False
    other_mask[target_class] = False
    other_confs = obj_sig.unsqueeze(-1) * cls_sig[:, :, other_mask]  # (1, N, 78)

    # Focus on anchors where target and person compete (highest obj * max(target,person))
    focus_score = torch.maximum(target_conf, person_conf)
    if focus_score.numel() > topk:
        _, top_idx = focus_score.squeeze(0).topk(topk)
    else:
        top_idx = torch.arange(focus_score.shape[1], device=DEVICE)

    top_obj = obj_sig[0, top_idx]
    top_person = person_conf[0, top_idx]
    top_target = target_conf[0, top_idx]
    top_other = other_confs[0, top_idx, :]  # (K, 78)

    # HOLD: maximize objectness
    loss_obj = -top_obj.mean()

    # POISON: target up, person down
    loss_target = -top_target.mean()
    loss_person = top_person.mean()

    # NO SWITCHING: penalize all other classes
    loss_other = top_other.mean()

    loss_cls = loss_target + loss_person + 2.0 * loss_other

    # FPN: keep total feature activation energy
    fpn_boost = torch.tensor(0.0, device=DEVICE)
    if fpn_features:
        for feat in fpn_features.values():
            fpn_boost = fpn_boost - feat.abs().mean()

    total = (alpha * loss_obj +
             beta * loss_cls +
             gamma * tv_loss(patch) +
             delta * nps_loss(patch) +
             fpn_w * fpn_boost)

    return total, {
        'obj': loss_obj.item(),
        'target': loss_target.item(),
        'person': loss_person.item(),
        'other': loss_other.item(),
        'cls': loss_cls.item(),
        'tv': tv_loss(patch).item(),
        'nps': nps_loss(patch).item(),
        'fpn': fpn_boost.item(),
        'total': total.item(),
    }


# ─── 4. LIVE VISUALIZATION ───────────────────────────────────────────── #

def save_preview(patch, epoch, out_dir, losses=None):
    """Save a preview image with loss overlay."""
    img_np = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    if losses is not None:
        txt = f"e{epoch:04d} obj:{losses['obj']:.3f} t:{losses['target']:.3f} p:{losses['person']:.3f} o:{losses['other']:.3f}"
        new = Image.new('RGB', (img.width, img.height + 24), (32, 32, 32))
        new.paste(img, (0, 0))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(new)
        draw.text((4, img.height + 4), txt, fill=(0, 255, 0))
        img = new

    path = os.path.join(out_dir, 'preview_best.png')  # overwritten each best
    img.save(path)
    return path


def setup_csv(out_dir):
    csv_path = os.path.join(out_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'time', 'obj', 'target', 'person', 'other', 'cls', 'tv', 'nps', 'fpn', 'total'])
    return csv_path


def log_csv(csv_path, epoch, stats):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, datetime.now().isoformat(),
                         stats['obj'], stats['target'], stats['person'], stats['other'], stats['cls'],
                         stats['tv'], stats['nps'], stats['fpn'], stats['total']])


# ─── 5. TRAINING LOOP ─────────────────────────────────────────────────── #

def train(epochs=500, lr=0.05, alpha=10.0, beta=5.0, gamma=0.01, delta=0.1, fpn_w=0.1,
          target_class=DEFAULT_TARGET, out_dir='outputs_hold_poison'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = setup_csv(out_dir)

    model = load_frozen_model()

    # Discover and hook FPN layers
    fpn_names = discover_fpn_layers(model)
    print(f"\nFPN layers found: {len(fpn_names)}")
    for n in fpn_names[:5]:
        print(f"  {n}")
    if len(fpn_names) > 5:
        print(f"  ... and {len(fpn_names)-5} more")

    fpn_features, hooks = register_fpn_hooks(model, fpn_names)

    # Init patch: white noise or nebula base
    if os.path.exists(os.path.join(BASE, '010_crab_nebula.jpg')):
        img = Image.open(os.path.join(BASE, '010_crab_nebula.jpg')).convert('RGB').resize((PATCH_SIZE, PATCH_SIZE))
        arr = np.array(img).transpose(2, 0, 1) / 255.0
        patch = torch.tensor(arr, device=DEVICE, dtype=torch.float32, requires_grad=True)
        print("Patch init: crab nebula")
    else:
        patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE, requires_grad=True)
        print("Patch init: white noise")

    opt = torch.optim.Adam([patch], lr=lr)

    best_total = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"HOLD-AND-POISON TRAINING")
    print(f"  Target class: {target_class}")
    print(f"  Alpha (obj):  {alpha}")
    print(f"  Beta (cls):   {beta}  (target up / person down)")
    print(f"  Gamma (tv):   {gamma}")
    print(f"  Delta (nps):  {delta}")
    print(f"  FPN weight:   {fpn_w}")
    print(f"  Epochs:       {epochs}  LR: {lr}")
    print(f"  Patch:        {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Live CSV:     {csv_path}")
    print(f"  Best preview: overwritten each epoch when total improves")
    print(f"{'='*60}\n")

    start = time.time()
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)

        # EoT
        warped = tps_warp(patch)
        warped = random_affine(warped)
        augmented = eot_jitter(warped)

        # Paste onto neutral canvas
        canvas = torch.full((3, IMG_SIZE, IMG_SIZE), 0.5, device=DEVICE)
        y0 = (IMG_SIZE - PATCH_SIZE) // 2
        x0 = (IMG_SIZE - PATCH_SIZE) // 2
        canvas[:, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = augmented

        # Forward pass
        fpn_features.clear()
        out = model(canvas.unsqueeze(0))
        if isinstance(out, (list, tuple)):
            out = out[0]

        # Composite loss
        loss, stats = composite_loss(
            out, fpn_features, patch,
            target_class, alpha, beta, gamma, delta, fpn_w
        )

        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)

        log_csv(csv_path, ep, stats)

        # Save preview only for current best (overwritten)
        is_best = stats['total'] < best_total
        if is_best:
            best_total = stats['total']
            best_epoch = ep
            save_preview(patch, ep, out_dir, stats)
            save_patch(patch, os.path.join(out_dir, 'checkpoint_best.png'))

        # Console print every 20 epochs
        if ep % 20 == 0 or ep == epochs - 1:
            elapsed = time.time() - start
            print(
                f"Ep {ep:4d}/{epochs} | "
                f"tot={stats['total']:8.3f} | obj={stats['obj']:6.3f} | "
                f"t={stats['target']:6.3f} | p={stats['person']:6.3f} | o={stats['other']:6.3f} | "
                f"tv={stats['tv']:.5f} | nps={stats['nps']:.4f} | "
                f"best={best_epoch} | {elapsed/60:.1f}m"
            )

    # Final save
    final_path = os.path.join(out_dir, 'patch_final.png')
    save_patch(patch, final_path)

    # Copy best checkpoint to final if best exists
    best_path = os.path.join(out_dir, 'checkpoint_best.png')
    if os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, final_path)
        print(f"Best patch (epoch {best_epoch}) copied to final")

    for h in hooks:
        h.remove()

    print(f"\n{'='*60}")
    print(f"DONE. Final patch: {final_path}")
    print(f"CSV log: {csv_path}")
    print(f"Previews in: {out_dir}")
    print(f"{'='*60}")

    return patch


def save_patch(patch, path):
    arr = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ─── 6. ENTRY POINT ───────────────────────────────────────────────────── #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hold-and-Poison Trainer')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=10.0, help='Objectness weight')
    parser.add_argument('--beta', type=float, default=5.0, help='Semantic weight')
    parser.add_argument('--gamma', type=float, default=0.01, help='TV weight')
    parser.add_argument('--delta', type=float, default=0.1, help='NPS weight')
    parser.add_argument('--fpn_w', type=float, default=0.1, help='FPN boost weight')
    parser.add_argument('--target', type=int, default=DEFAULT_TARGET, help='Target class ID')
    parser.add_argument('--out', type=str, default='outputs_hold_poison')
    args = parser.parse_args()

    train(
        epochs=args.epochs, lr=args.lr,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, fpn_w=args.fpn_w,
        target_class=args.target, out_dir=args.out,
    )
