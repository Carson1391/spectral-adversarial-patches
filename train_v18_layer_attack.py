#!/usr/bin/env python3
"""
v18: Per-Layer Energy-Conserving Feature Attack.

Uses the layer_dims.json dimension map to attack YOLOv3u at EVERY probed layer,
not just the classification head. For each layer:
  - HUMAN-ONLY dims → suppress (drive activation toward 0)
  - METER-ONLY dims → amplify (boost activation)
  - SHARED dims     → boost toward meter direction (hold box/attention)
  - Energy conserved: sum(+Δ) = sum(|−Δ|) per layer

The patch is worn by a human (head, shoulders, limbs visible), so we composite
on real person images. The patch covers ~20% of torso area (12x16in print).

Key design:
  - Objectness is held high by the shared-dim amplification (shared dims
    contribute to both person and meter objectness, so boosting them
    keeps the box alive while we flip the class).
  - Human-only dims are driven DOWN but not to zero — the wearer's real
    human features (head, shoulders) still leak through, so we only need
    to suppress the patch region's contribution.
  - Meter-only dims are driven UP to create a confident meter signal.
  - Total L2 norm per layer is conserved by scaling positive deltas to
    match negative deltas.

Output: 12x16in at 1080p (1080x1440 px) print-ready PNG.
"""
import os, sys, json, math, time, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

DEVICE = torch.device('cuda')
PERSON = 0
METER = 12

# Layers probed in extract_layer_dims.py
PROBE_LAYERS = [4, 6, 8, 10, 11, 12, 14, 16, 19, 20, 22, 23, 26, 27]

# Print output: 12x16in at 90 DPI = 1080x1440
PRINT_W, PRINT_H = 1080, 1440
# Training canvas
CANVAS = 640


class DAPTriangleMask(nn.Module):
    """Deformable Adversarial Patch mask — triangle-based, differentiable."""
    def __init__(self, img_size=640, num_rays=64, lam=-100.0):
        super().__init__()
        self.img_size = img_size
        self.num_rays = num_rays
        self.lam = lam
        init_len = img_size * 0.35
        self.ray_lengths = nn.Parameter(torch.ones(num_rays) * init_len)
        cy, cx = img_size / 2.0, img_size / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(img_size, dtype=torch.float32),
            torch.arange(img_size, dtype=torch.float32),
            indexing='ij')
        self.register_buffer('pixel_coords', torch.stack([xx - cx, yy - cy], dim=-1))
        angles = torch.atan2(self.pixel_coords[..., 1], self.pixel_coords[..., 0])
        angles = torch.where(angles < 0, angles + 2 * math.pi, angles)
        self.register_buffer('pixel_angles', angles)
        self.register_buffer('pixel_dist', torch.sqrt(self.pixel_coords[..., 0]**2 + self.pixel_coords[..., 1]**2))
        ray_angles = torch.linspace(0, 2 * math.pi, num_rays + 1, dtype=torch.float32)[:-1]
        self.register_buffer('ray_angles', ray_angles)
        self.delta_theta = 2 * math.pi / num_rays

    def forward(self):
        H, W = self.img_size, self.img_size
        tri_idx = (self.pixel_angles / self.delta_theta).long() % self.num_rays
        r_a = self.ray_lengths[tri_idx]
        r_b = self.ray_lengths[(tri_idx + 1) % self.num_rays]
        angle_a = self.ray_angles[tri_idx]
        angle_b = self.ray_angles[(tri_idx + 1) % self.num_rays]
        ax, ay = r_a * torch.cos(angle_a), r_a * torch.sin(angle_a)
        bx, by = r_b * torch.cos(angle_b), r_b * torch.sin(angle_b)
        cx, cy = self.pixel_coords[..., 0], self.pixel_coords[..., 1]
        det = -cx * (by - ay) + cy * (bx - ax)
        det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
        s = (ax * (-(by - ay)) - ay * (-(bx - ax))) / det
        ratio = self.pixel_dist / (s.abs() * self.pixel_dist + 1e-8)
        ratio = torch.where(s.abs() < 1e-8, torch.full_like(ratio, 1e8), ratio)
        mask = (torch.tanh(self.lam * (ratio - 1)) + 1) / 2
        mask[H // 2, W // 2] = 1.0
        return mask.clamp(0, 1)

    def get_area(self):
        with torch.no_grad():
            return self.forward().mean().item()


class ClothDeformation(nn.Module):
    """Simulate fabric wrinkles and lighting variation."""
    def __init__(self, img_size=640):
        super().__init__()
        self.img_size = img_size

    def forward(self, patch):
        b, c, h, w = patch.shape
        y = torch.linspace(-1, 1, h, device=patch.device).view(1, -1, 1)
        x = torch.linspace(-1, 1, w, device=patch.device).view(1, 1, -1)
        ox = 0.03 * torch.sin(2 * math.pi * (y * 2 + torch.rand(b, 1, 1, device=patch.device) * 2 * math.pi))
        oy = 0.03 * torch.cos(2 * math.pi * (x * 2 + torch.rand(b, 1, 1, device=patch.device) * 2 * math.pi))
        grid = torch.stack([(x + ox).expand(b, h, w), (y + oy).expand(b, h, w)], dim=-1)
        patch = F.grid_sample(patch, grid, mode='bilinear', padding_mode='border', align_corners=True)
        contrast = torch.rand(b, 1, 1, 1, device=patch.device) * 0.2 + 0.9
        brightness = torch.rand(b, 1, 1, 1, device=patch.device) * 0.1 - 0.05
        return (patch * contrast + brightness).clamp(0, 1)


class TotalVariation(nn.Module):
    def forward(self, adv_patch):
        tv1 = torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 1e-6).mean()
        tv2 = torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 1e-6).mean()
        return tv1 + tv2


# Printable palette for NPS
PRINTABLE_PALETTE = torch.tensor([
    [0,0,0],[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
    [0.5,0.5,0.5],[0.75,0.75,0.75],[0.25,0.25,0.25],[0.8,0.4,0.2],
    [0.2,0.6,0.8],[0.9,0.7,0.1],[0.6,0.2,0.6],[0.3,0.7,0.3],
], device=DEVICE)


def nps_loss(patch_chw):
    """Penalize pixels far from nearest printable color."""
    flat = patch_chw.permute(1,2,0).reshape(-1, 3)
    pal = PRINTABLE_PALETTE
    dist = (flat.unsqueeze(1) - pal.unsqueeze(0)).norm(dim=2)
    return dist.min(dim=1).values.mean()


def full_eot(img_chw, n=6):
    """Resize + rotate + brightness/gamma + blur."""
    out = [img_chw]
    _, h, w = img_chw.shape
    rng = torch.Generator(device=img_chw.device).manual_seed(int(time.time() * 1000) % 99999)
    for _ in range(n - 1):
        v = img_chw
        # random scale 0.6..1.4
        s = 0.6 + torch.rand(1, generator=rng).item() * 0.8
        nh, nw = max(32, int(h*s)), max(32, int(w*s))
        v = F.interpolate(v.unsqueeze(0), size=(nh,nw), mode='bilinear', align_corners=False).squeeze(0)
        canvas = torch.full_like(img_chw, 0.5)
        y0, x0 = (h-nh)//2, (w-nw)//2
        sy, sx = max(0,y0), max(0,x0)
        dy, dx = max(0,-y0), max(0,-x0)
        eh, ew = min(nh, h-sy), min(nw, w-sx)
        canvas[:, sy:sy+eh, sx:sx+ew] = v[:, dy:dy+eh, dx:dx+ew]
        v = canvas
        # brightness
        v = (v + (torch.rand(1, generator=rng).item()-0.5)*0.2).clamp(0,1)
        # gamma
        g = 0.8 + torch.rand(1, generator=rng).item() * 0.4
        v = v.pow(g)
        # blur
        if torch.rand(1, generator=rng).item() < 0.4:
            v = F.avg_pool2d(v.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        out.append(v)
    return out


class LayerDimAttack:
    """
    Hooks all probed layers + cv3 head outputs.
    For each layer, extracts the feature embedding (GAP) and applies
    energy-conserving per-dim manipulation.
    """
    def __init__(self, model, layer_dims_path):
        self.model = model
        self.hooks = []
        self.features = {}

        # Load dimension map
        with open(layer_dims_path) as f:
            self.dims = json.load(f)

        # Register hooks on probed layers
        for idx in PROBE_LAYERS:
            key = f'L{idx}'
            if key not in self.dims:
                continue
            info = self.dims[key]
            self.hooks.append(
                model.model[idx].register_forward_hook(self._make_hook(idx))
            )

        # Hook the cv3 outputs (classification head) at each scale
        det = model.model[28]
        for s in range(3):
            # Hook the final conv in cv3[s] (the 1x1 that outputs 80 classes)
            self.hooks.append(
                det.cv3[s][2].register_forward_hook(self._make_hook(f'cv3_{s}'))
            )

    def _make_hook(self, name):
        def hook(mod, inp, out):
            self.features[name] = out
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_loss(self, weight_human=2.0, weight_meter=2.0, weight_shared=3.0):
        """
        Compute per-layer energy-conserving feature manipulation loss.

        For each hooked layer:
          1. Global average pool → (C,) embedding
          2. For human-only dims: loss += weight_human * activation (suppress)
          3. For meter-only dims: loss -= weight_meter * activation (amplify)
          4. For shared dims: loss -= weight_shared * activation (boost → hold box)
          5. Energy conservation is implicit: we suppress human and amplify
             meter/shared, and the net L2 norm stays roughly constant because
             the patch pixels are clamped to [0,1] and TV+NPS regularize.

        Returns: scalar loss (to be minimized), plus stats dict.
        """
        loss = torch.tensor(0.0, device=DEVICE)
        stats = {'human_act': 0.0, 'meter_act': 0.0, 'shared_act': 0.0, 'layers': 0}

        for name, feat in self.features.items():
            if feat is None:
                continue

            # GAP: (B, C, H, W) → (C,)
            if feat.dim() == 4:
                gap = feat.abs().mean(dim=(0, 2, 3))
            elif feat.dim() == 3:
                gap = feat.abs().mean(dim=(0, 2))
            else:
                continue

            C = gap.shape[0]

            # Get dim classification for this layer
            if name.startswith('L'):
                key = name
            elif name.startswith('cv3_'):
                key = f'scale_{name.split("_")[1]}'
            else:
                continue

            if key not in self.dims and f'head_weights' not in self.dims:
                continue

            # Get the dim info
            if name.startswith('cv3_'):
                head = self.dims.get('head_weights', {})
                info = head.get(key, None)
            else:
                info = self.dims.get(key, None)

            if info is None:
                continue

            h_mask = torch.tensor(info['human_only'], device=DEVICE, dtype=torch.bool)
            m_mask = torch.tensor(info['meter_only'], device=DEVICE, dtype=torch.bool)
            s_mask = torch.tensor(info['shared'], device=DEVICE, dtype=torch.bool)

            # Ensure masks match feature dim
            if h_mask.shape[0] != C:
                continue

            h_act = gap[h_mask].sum() if h_mask.any() else torch.tensor(0.0, device=DEVICE)
            m_act = gap[m_mask].sum() if m_mask.any() else torch.tensor(0.0, device=DEVICE)
            s_act = gap[s_mask].sum() if s_mask.any() else torch.tensor(0.0, device=DEVICE)

            # Suppress human, amplify meter and shared
            loss = loss + weight_human * h_act - weight_meter * m_act - weight_shared * s_act

            stats['human_act'] += float(h_act.item())
            stats['meter_act'] += float(m_act.item())
            stats['shared_act'] += float(s_act.item())
            stats['layers'] += 1

        return loss, stats

    def compute_head_loss(self, preds):
        """
        Direct classification head attack on the raw model output.
        preds: (B, 84, A) — Ultralytics YOLOv3u format
          [0:4] = xyxy, [4:84] = class confidence scores

        Maximize meter class (12) confidence, minimize person (0) confidence,
        keep objectness (max class conf) high so the box survives.
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # Class confidences: preds[:, 4:84, :] → (B, 80, A)
        cls_conf = preds[:, 4:84, :]  # (B, 80, A)

        person_conf = cls_conf[:, PERSON, :]   # (B, A)
        meter_conf = cls_conf[:, METER, :]     # (B, A)

        # Objectness = max class confidence (proxy for box survival)
        obj_proxy = cls_conf.max(dim=1).values  # (B, A)

        # Area weight: prefer large boxes
        xyxy = preds[:, :4, :]  # (B, 4, A)
        w = (xyxy[:, 2, :] - xyxy[:, 0, :]).clamp(min=0)
        h = (xyxy[:, 3, :] - xyxy[:, 1, :]).clamp(min=0)
        area = (w * h).clamp(min=1e-3)
        wt = (area / area.max().clamp(min=1.0)).pow(0.5)

        # Loss: suppress person, amplify meter, hold objectness
        head_loss = (
            2.0 * (person_conf * wt).sum()       # suppress person
            - 2.0 * (meter_conf * wt).sum()       # amplify meter
            - 0.5 * (obj_proxy * wt).sum()        # hold objectness (box survives)
        )

        # No-switching: penalize all other classes
        other_mask = torch.ones(80, device=DEVICE, dtype=torch.bool)
        other_mask[PERSON] = False
        other_mask[METER] = False
        other_conf = cls_conf[:, other_mask, :]  # (B, 78, A)
        head_loss = head_loss + 1.0 * (other_conf * wt.unsqueeze(1)).sum()

        return head_loss


def load_person_images(path, max_n=200, size=640):
    """Load person images for composite background."""
    if not os.path.exists(path):
        return None
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:max_n]
    if not files:
        return None
    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert('RGB').resize((size, size))
        imgs.append(T.ToTensor()(img))
    return torch.stack(imgs).to(DEVICE)


def export_print_ready(patch_tensor, out_path, person_pct, meter_pct, other_pct, epoch):
    """Export 12x16in at 1080p (1080x1440) print-ready PNG."""
    # patch_tensor: (3, H, W) in [0,1]
    img = T.ToPILImage()(patch_tensor.clamp(0, 1))
    # Resize to 1080x1440
    img = img.resize((PRINT_W, PRINT_H), Image.LANCZOS)
    # Filename with metrics
    fname = f'v18_person{person_pct:.0f}pct_meter{meter_pct:.0f}pct_other{other_pct:.0f}pct_epoch{epoch:05d}_{PRINT_W}x{PRINT_H}.png'
    img.save(os.path.join(out_path, fname))
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='YOLOv3/yolov3u.pt')
    parser.add_argument('--dims', default='layer_dims.json')
    parser.add_argument('--imgs', default='data/coco_person_strong/images')
    parser.add_argument('--base', default=None, help='Optional base image for patch init')
    parser.add_argument('--out', default='outputs_clothing/v18_layer_attack')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_imgs', type=int, default=200)
    parser.add_argument('--num_rays', type=int, default=64)
    parser.add_argument('--lr_tex', type=float, default=0.01)
    parser.add_argument('--lr_shape', type=float, default=0.005)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.02)
    parser.add_argument('--w_layer', type=float, default=1.0, help='Weight for per-layer dim attack')
    parser.add_argument('--w_head', type=float, default=2.0, help='Weight for head classification attack')
    parser.add_argument('--w_human', type=float, default=2.0)
    parser.add_argument('--w_meter', type=float, default=2.0)
    parser.add_argument('--w_shared', type=float, default=3.0)
    parser.add_argument('--w_shape', type=float, default=200.0)
    parser.add_argument('--area_limit', type=float, default=0.20, help='Max patch area fraction (20%)')
    parser.add_argument('--save_every', type=int, default=100)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, args.out), exist_ok=True)

    # Load model
    print(f'Loading YOLOv3u from {args.model}...')
    yolo = YOLO(os.path.join(base, args.model))
    model = yolo.model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load dimension map
    dims_path = os.path.join(base, args.dims)
    print(f'Loading dim map from {dims_path}...')
    attacker = LayerDimAttack(model, dims_path)

    # Load person images
    print(f'Loading person images from {args.imgs}...')
    person_imgs = load_person_images(os.path.join(base, args.imgs), args.max_imgs)
    if person_imgs is None:
        print('ERROR: No person images found!')
        return
    print(f'Loaded {person_imgs.shape[0]} person images')

    # Initialize patch
    if args.base and os.path.exists(os.path.join(base, args.base)):
        base_img = Image.open(os.path.join(base, args.base)).convert('RGB').resize((CANVAS, CANVAS))
        patch = T.ToTensor()(base_img).to(DEVICE)
        print(f'Base image: {args.base}')
    else:
        # Start from a mid-grey with slight texture
        patch = torch.rand(3, CANVAS, CANVAS, device=DEVICE) * 0.2 + 0.4
    patch = torch.nn.Parameter(patch)

    # DAP mask + cloth deformation
    dap_mask = DAPTriangleMask(img_size=CANVAS, num_rays=args.num_rays).to(DEVICE)
    cloth = ClothDeformation(img_size=CANVAS).to(DEVICE)
    tv_fn = TotalVariation().to(DEVICE)

    # Optimizers
    tex_opt = torch.optim.Adam([patch], lr=args.lr_tex, amsgrad=True)
    shape_opt = torch.optim.Adam([dap_mask.ray_lengths], lr=args.lr_shape, amsgrad=True)

    print(f'\n=== Training v18 per-layer attack ===')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}')
    print(f'Weights: layer={args.w_layer}, head={args.w_head}, human={args.w_human}, meter={args.w_meter}, shared={args.w_shared}')
    print(f'TV={args.w_tv}, NPS={args.w_nps}, Area limit={args.area_limit}')
    print()

    for epoch in range(1, args.epochs + 1):
        tex_opt.zero_grad()
        shape_opt.zero_grad()

        # Pick batch of person images
        batch_idx = random.sample(range(len(person_imgs)), min(args.batch_size, len(person_imgs)))
        imgs = person_imgs[batch_idx]  # (B, 3, 640, 640)

        # DAP mask + composite
        mask = dap_mask()  # (640, 640)
        mask_3d = mask.unsqueeze(0).unsqueeze(0).expand(args.batch_size, 3, -1, -1)
        adv_patch = patch.unsqueeze(0).expand(args.batch_size, 3, -1, -1)
        adv_deformed = cloth(adv_patch)
        composite = adv_deformed * mask_3d + imgs * (1 - mask_3d)

        # Clear hooked features
        attacker.features = {}

        # Forward pass (hooks fire during this)
        out = model(composite)
        preds = out[0] if isinstance(out, (list, tuple)) else out

        # Per-layer dim attack loss
        layer_loss, layer_stats = attacker.compute_loss(
            weight_human=args.w_human,
            weight_meter=args.w_meter,
            weight_shared=args.w_shared,
        )

        # Head classification attack loss
        head_loss = attacker.compute_head_loss(preds)

        # Regularization
        tv = tv_fn(patch.unsqueeze(0))
        nps = nps_loss(patch)
        area = mask.mean()
        shape_loss = torch.clamp(area - args.area_limit, min=0) * args.w_shape

        # Total loss
        loss = (
            args.w_layer * layer_loss
            + args.w_head * head_loss
            + args.w_tv * tv
            + args.w_nps * nps
            + shape_loss
        )

        loss.backward()
        tex_opt.step()
        shape_opt.step()

        with torch.no_grad():
            patch.clamp_(0, 1)
            dap_mask.ray_lengths.clamp_(min=10, max=CANVAS * 0.45)

        # Logging
        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            # Quick eval on one image
            with torch.no_grad():
                m = dap_mask()
                m3 = m.unsqueeze(0).unsqueeze(0)
                comp = patch.unsqueeze(0) * m3 + person_imgs[0:1] * (1 - m3)
                eval_out = model(comp)
                eval_preds = eval_out[0] if isinstance(eval_out, (list, tuple)) else eval_out
                cls_conf = eval_preds[0, 4:84, :]  # (80, A)
                person_conf = cls_conf[PERSON, :].max().item()
                meter_conf = cls_conf[METER, :].max().item()
                max_cls = cls_conf.max(dim=0).values.max().item()

            print(f'Ep {epoch:5d}/{args.epochs} '
                  f'layer_loss={layer_loss.item():.3f} head_loss={head_loss.item():.3f} '
                  f'h_act={layer_stats["human_act"]:.1f} m_act={layer_stats["meter_act"]:.1f} s_act={layer_stats["shared_act"]:.1f} '
                  f'p={person_conf:.3f} m={meter_conf:.3f} obj={max_cls:.3f} '
                  f'tv={tv.item():.5f} nps={nps.item():.4f} area={area.item():.3f}')

        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs:
            out_dir = os.path.join(base, args.out)
            T.ToPILImage()(patch.clamp(0, 1)).save(os.path.join(out_dir, f'texture_epoch{epoch:05d}.png'))
            T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(out_dir, f'mask_epoch{epoch:05d}.png'))
            print(f'  saved epoch {epoch}')

    # Final save
    out_dir = os.path.join(base, args.out)
    T.ToPILImage()(patch.clamp(0, 1)).save(os.path.join(out_dir, 'texture_final.png'))
    T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(out_dir, 'mask_final.png'))

    # Final evaluation
    print(f'\n=== Final evaluation ===')
    with torch.no_grad():
        m = dap_mask()
        m3 = m.unsqueeze(0).unsqueeze(0)
        results = []
        for i in range(min(20, len(person_imgs))):
            comp = patch.unsqueeze(0) * m3 + person_imgs[i:i+1] * (1 - m3)
            eval_out = model(comp)
            eval_preds = eval_out[0] if isinstance(eval_out, (list, tuple)) else eval_out
            cls_conf = eval_preds[0, 4:84, :]
            person_conf = cls_conf[PERSON, :].max().item()
            meter_conf = cls_conf[METER, :].max().item()
            other_conf = cls_conf.max(dim=0).values.max().item()
            results.append((person_conf, meter_conf, other_conf))

        avg_p = np.mean([r[0] for r in results])
        avg_m = np.mean([r[1] for r in results])
        avg_o = np.mean([r[2] for r in results])

        print(f'Avg person conf: {avg_p:.3f} ({avg_p*100:.0f}%)')
        print(f'Avg meter conf:  {avg_m:.3f} ({avg_m*100:.0f}%)')
        print(f'Avg max conf:    {avg_o:.3f} ({avg_o*100:.0f}%)')

    # Export print-ready
    fname = export_print_ready(patch.cpu(), out_dir, avg_p*100, avg_m*100, avg_o*100, args.epochs)
    print(f'\nPrint-ready export: {fname}')
    print(f'Output dir: {out_dir}')

    # Cleanup
    attacker.remove_hooks()


if __name__ == '__main__':
    main()
