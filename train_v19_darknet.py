#!/usr/bin/env python3
"""
v19: Faithful Darknet YOLOv3 Energy-Conserving Patch Trainer.

ARCHITECTURE: Uses faithful PyTorch-YOLOv3 port (NOT Ultralytics yolov3u).
  - Separate objectness (1 sigmoid) + class (80 independent logistics) channels.
  - Output: 10647 anchors × 85 = [0:4]=box [4]=obj [5:85]=cls.
  - Matches Flock pico3 architecture head-for-head.

ENERGY STRATEGY (Ryan's design):
  - Human-only dims: SUPPRESS but NOT to zero. Keep enough for objectness to fire.
    Wearer's real head/shoulders/limbs leak human through 80% of frame anyway.
  - Shared dims: AMPLIFY HARDEST (highest weight). Double duty:
    holds box via objectness AND pushes class toward meter.
  - Meter-only dims: AMPLIFY. Pure meter signal, no downside.
  - Energy conservation: sum(+Δ) = sum(|−Δ|) per layer. Total L2 norm constant.
  - Net effect: confident box (shared holds it), class flips person→meter.

LOSS = w_obj * obj_loss + w_cls * cls_loss + w_layer * layer_loss + TV + NPS + shape_loss
  - obj_loss: maximize objectness logits (fire the box) — HEAVIEST weight
  - cls_loss: suppress person class, amplify meter class, penalize all 78 others
  - layer_loss: per-layer energy-conserving dim manipulation from darknet_layer_dims.csv
  - TV=2.5 (AdvReal), NPS printable palette, DAP triangle shape, cloth deformation

Output: 12x16in 1080p (1080x1440) print-ready PNG.
"""
import os, sys, json, math, time, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add PyTorch-YOLOv3 path
Y3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'AdvReal', 'detlib', 'HHDet', 'yolov3', 'PyTorch_YOLOv3')
sys.path.insert(0, Y3_PATH)

from pytorchyolo.models import load_model

DEVICE = torch.device('cuda')
PERSON = 0
METER = 12
CANVAS = 416  # Darknet native input size
PRINT_W, PRINT_H = 1080, 1440  # 12x16in at 90dpi

# Backbone layers to hook (from extract_darknet_dims.py)
HOOK_LAYERS = [0, 1, 3, 10, 35, 60, 73, 81, 91, 103]
# Detection head conv indices (255-channel output convs)
DETECT_CONVS = [81, 93, 105]

# Objectness channels in 255-output: per anchor [0:4]=box, [4]=obj, [5:85]=cls
# Anchor 0: obj=4,  Anchor 1: obj=89,  Anchor 2: obj=174
OBJ_CHANNELS = [4, 89, 174]
# Person class channels: anchor0=5, anchor1=90, anchor2=175
PERSON_CHANNELS = [5, 90, 175]
# Meter class channels: anchor0=17, anchor1=102, anchor2=187
METER_CHANNELS = [17, 102, 187]


class DAPTriangleMask(nn.Module):
    def __init__(self, img_size=416, num_rays=64, lam=-100.0):
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


class ClothDeformation(nn.Module):
    def __init__(self, img_size=416):
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
    def forward(self, p):
        return torch.abs(p[:, :, 1:] - p[:, :, :-1] + 1e-6).mean() + \
               torch.abs(p[:, 1:, :] - p[:, :-1, :] + 1e-6).mean()


PRINTABLE_PALETTE = torch.tensor([
    [0,0,0],[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
    [0.5,0.5,0.5],[0.75,0.75,0.75],[0.25,0.25,0.25],[0.8,0.4,0.2],
    [0.2,0.6,0.8],[0.9,0.7,0.1],[0.6,0.2,0.6],[0.3,0.7,0.3],
], device=DEVICE)


def nps_loss(patch_chw):
    flat = patch_chw.permute(1,2,0).reshape(-1, 3)
    dist = (flat.unsqueeze(1) - PRINTABLE_PALETTE.unsqueeze(0)).norm(dim=2)
    return dist.min(dim=1).values.mean()


def full_eot(img_chw, n=6):
    out = [img_chw]
    _, h, w = img_chw.shape
    rng = torch.Generator(device=img_chw.device).manual_seed(int(time.time() * 1000) % 99999)
    for _ in range(n - 1):
        v = img_chw
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
        v = (v + (torch.rand(1, generator=rng).item()-0.5)*0.2).clamp(0,1)
        g = 0.8 + torch.rand(1, generator=rng).item() * 0.4
        v = v.pow(g)
        if torch.rand(1, generator=rng).item() < 0.4:
            v = F.avg_pool2d(v.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        out.append(v)
    return out


class LayerDimAttack:
    """
    Hooks backbone layers + detection head convs.
    Per-layer energy-conserving dim manipulation.
    """
    def __init__(self, model, dims_path):
        self.model = model
        self.hooks = []
        self.features = {}

        with open(dims_path) as f:
            import csv
            reader = csv.DictReader(f)
            self.dims = {}
            for row in reader:
                layer = row['layer']
                info = {
                    'dim': int(row['total_dims']),
                    'human_only': [int(x) for x in row['human_only_dims'].split(';') if x],
                    'meter_only': [int(x) for x in row['meter_only_dims'].split(';') if x],
                    'shared': [int(x) for x in row['shared_dims'].split(';') if x],
                    'band': row['band'],
                    'resolution': row['resolution'],
                    'description': row['description'],
                }
                self.dims[layer] = info

        # Hook backbone layers
        for idx in HOOK_LAYERS:
            self.hooks.append(
                model.module_list[idx].register_forward_hook(self._make_hook(str(idx)))
            )

    def _make_hook(self, name):
        def hook(mod, inp, out):
            self.features[name] = out
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def compute_layer_loss(self, w_human=3.0, w_meter=3.0, w_shared=3.0):
        """
        Energy strategy (Ryan's final design):
        1. Human-only dims: DOWN HARD (w_human high — drive to zero, max suppress)
        2. Meter-only dims: UP HARD (w_meter high — the meter signal we want)
        3. Shared dims: ALL UP (w_shared high — holds human box via objectness
           AND raises meter simultaneously. Shared compensates for human suppression.)
        Energy balance: human down hard, meter+shared up hard. Shared keeps box alive.
        """
        loss = torch.tensor(0.0, device=DEVICE)
        stats = {'h_act': 0.0, 'm_act': 0.0, 's_act': 0.0, 'layers': 0}

        for name, feat in self.features.items():
            if feat is None or name not in self.dims:
                continue

            info = self.dims[name]
            C = feat.shape[1]
            gap = feat.abs().mean(dim=(0, 2, 3))

            h_dims = [d for d in info['human_only'] if d < C]
            m_dims = [d for d in info['meter_only'] if d < C]
            s_dims = [d for d in info['shared'] if d < C]

            if not h_dims and not m_dims and not s_dims:
                continue

            h_act = gap[h_dims].sum() if h_dims else torch.tensor(0.0, device=DEVICE)
            m_act = gap[m_dims].sum() if m_dims else torch.tensor(0.0, device=DEVICE)
            s_act = gap[s_dims].sum() if s_dims else torch.tensor(0.0, device=DEVICE)

            # Human-only: suppress HARD
            loss = loss + w_human * h_act
            # Meter-only: amplify HARD
            loss = loss - w_meter * m_act
            # Shared: ALL UP — holds box AND raises meter
            loss = loss - w_shared * s_act

            stats['h_act'] += float(h_act.item())
            stats['m_act'] += float(m_act.item())
            stats['s_act'] += float(s_act.item())
            stats['layers'] += 1

        return loss, stats

    def compute_head_loss(self, yolo_outputs, mask_2d=None):
        """
        Direct attack on detection head output — FOCUSED on patch region.
        Only penalize/suppress on anchors that overlap the DAP mask area.
        Anchors looking at the person's head/legs (outside mask) are ignored.

        Training mode returns list of 3 tensors:
          [0] (B, 3, 13, 13, 85)  — stride 32
          [1] (B, 3, 26, 26, 85)  — stride 16
          [2] (B, 3, 52, 52, 85)  — stride 8
        """
        if not isinstance(yolo_outputs, (list, tuple)):
            yolo_outputs = [yolo_outputs]

        obj_loss = torch.tensor(0.0, device=DEVICE)
        cls_loss = torch.tensor(0.0, device=DEVICE)

        scale_weights = [3.0, 1.5, 0.5]  # s32, s16, s8
        grid_sizes = [13, 26, 52]  # spatial sizes per scale

        for scale_idx, preds in enumerate(yolo_outputs):
            B, A, H, W, _ = preds.shape
            sw = scale_weights[scale_idx]

            # Build spatial mask: which grid cells overlap the DAP patch?
            if mask_2d is not None:
                # Downsample mask to this grid size
                cell_mask = F.adaptive_avg_pool2d(
                    mask_2d.unsqueeze(0).unsqueeze(0), (H, W)
                ).squeeze()  # (H, W)
                # Expand to (A, H, W) — all anchors in active cells count
                cell_mask = cell_mask.unsqueeze(0).expand(A, H, W)
                # Flatten to (A*H*W,)
                anchor_mask = cell_mask.reshape(-1)
                # Only use anchors where mask > 0.1 (patch is visible)
                anchor_mask = (anchor_mask > 0.1).float()
            else:
                anchor_mask = torch.ones(A * H * W, device=DEVICE)

            n_active = anchor_mask.sum().clamp(min=1.0)

            flat = preds.reshape(B, A * H * W, 85)

            obj_logits = flat[:, :, 4]
            person_logits = flat[:, :, 5 + PERSON]
            meter_logits = flat[:, :, 5 + METER]

            obj_sig = obj_logits.sigmoid()
            person_sig = person_logits.sigmoid()
            meter_sig = meter_logits.sigmoid()

            # Apply mask — only penalize/amplify in patch region
            am = anchor_mask.unsqueeze(0)  # (1, A*H*W)

            # OBJ: maximize objectness in patch region
            obj_loss = obj_loss - sw * (obj_sig * am).sum() / n_active

            # CLS: suppress person in patch region, amplify meter hard
            cls_loss = cls_loss + sw * 3.0 * (person_sig * am).sum() / n_active
            cls_loss = cls_loss - sw * 5.0 * (meter_sig * am).sum() / n_active

            # No-switching: light penalty on other classes in patch region
            for c in range(80):
                if c != PERSON and c != METER:
                    cls_sig = flat[:, :, 5 + c].sigmoid()
                    cls_loss = cls_loss + sw * 0.1 * (cls_sig * am).sum() / n_active

        return obj_loss, cls_loss


def load_person_images(path, max_n=200, size=416):
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


def export_print_ready(patch_tensor, out_path, person_pct, meter_pct, obj_pct, epoch):
    img = T.ToPILImage()(patch_tensor.clamp(0, 1))
    img = img.resize((PRINT_W, PRINT_H), Image.LANCZOS)
    fname = f'v19_person{person_pct:.0f}pct_meter{meter_pct:.0f}pct_obj{obj_pct:.0f}pct_epoch{epoch:05d}_{PRINT_W}x{PRINT_H}.png'
    img.save(os.path.join(out_path, fname))
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=os.path.join(Y3_PATH, 'config', 'yolov3.cfg'))
    parser.add_argument('--weights', default='yolov3.weights')
    parser.add_argument('--dims', default='darknet_layer_dims.csv')
    parser.add_argument('--imgs', default='data/coco_person_strong/images')
    parser.add_argument('--base', default=None)
    parser.add_argument('--out', default='outputs_clothing/v19_darknet')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_imgs', type=int, default=200)
    parser.add_argument('--num_rays', type=int, default=64)
    parser.add_argument('--lr_tex', type=float, default=0.01)
    parser.add_argument('--lr_shape', type=float, default=0.005)
    parser.add_argument('--w_obj', type=float, default=5.0, help='Objectness loss weight (HEAVIEST — fire box)')
    parser.add_argument('--w_cls', type=float, default=3.0, help='Class loss weight')
    parser.add_argument('--w_layer', type=float, default=0.1, help='Per-layer dim loss weight (LOW — dont crush obj)')
    parser.add_argument('--w_human', type=float, default=3.0, help='Human dim suppress (HARD — drive to zero)')
    parser.add_argument('--w_meter', type=float, default=3.0, help='Meter dim amplify (HARD — meter signal)')
    parser.add_argument('--w_shared', type=float, default=3.0, help='Shared ALL UP (holds box + raises meter)')
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.02)
    parser.add_argument('--w_shape', type=float, default=200.0)
    parser.add_argument('--area_limit', type=float, default=0.20)
    parser.add_argument('--save_every', type=int, default=100)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, args.out), exist_ok=True)

    print(f'Loading faithful Darknet YOLOv3...')
    print(f'  cfg: {args.cfg}')
    print(f'  weights: {args.weights}')
    model = load_model(args.cfg, os.path.join(base, args.weights)).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # We need training mode to get RAW pre-sigmoid outputs
    # But we also want hooks to fire — training mode returns list of yolo outputs
    model.train()  # Keep in train mode for raw logits
    for p in model.parameters():
        p.requires_grad_(False)  # But still frozen

    print(f'Model loaded. {sum(p.numel() for p in model.parameters())} parameters frozen')

    # Load dim map
    dims_path = os.path.join(base, args.dims)
    print(f'Loading dim map from {dims_path}...')
    attacker = LayerDimAttack(model, dims_path)

    # No person images needed — train patch standalone on neutral canvas
    # The patch IS the signal. Test on person images separately after training.
    print(f'Training on neutral canvas (no person images)')

    # Init patch
    if args.base and os.path.exists(os.path.join(base, args.base)):
        base_img = Image.open(os.path.join(base, args.base)).convert('RGB').resize((CANVAS, CANVAS))
        patch = T.ToTensor()(base_img).to(DEVICE)
    else:
        patch = torch.rand(3, CANVAS, CANVAS, device=DEVICE) * 0.2 + 0.4
    patch = torch.nn.Parameter(patch)

    dap_mask = DAPTriangleMask(img_size=CANVAS, num_rays=args.num_rays).to(DEVICE)
    cloth = ClothDeformation(img_size=CANVAS).to(DEVICE)
    tv_fn = TotalVariation().to(DEVICE)

    tex_opt = torch.optim.Adam([patch], lr=args.lr_tex, amsgrad=True)
    shape_opt = torch.optim.Adam([dap_mask.ray_lengths], lr=args.lr_shape, amsgrad=True)

    print(f'\n=== v19 Training ===')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}')
    print(f'Weights: obj={args.w_obj} cls={args.w_cls} layer={args.w_layer}')
    print(f'Dim weights: human={args.w_human}(DOWN) meter={args.w_meter}(UP) shared={args.w_shared}(UP)')
    print(f'TV={args.w_tv} NPS={args.w_nps} Area={args.area_limit}')
    print()

    for epoch in range(1, args.epochs + 1):
        tex_opt.zero_grad()
        shape_opt.zero_grad()

        # Neutral grey background — patch is the only signal
        imgs = torch.full((args.batch_size, 3, CANVAS, CANVAS), 0.5, device=DEVICE)

        mask = dap_mask()
        mask_3d = mask.unsqueeze(0).unsqueeze(0).expand(args.batch_size, 3, -1, -1)
        adv_patch = patch.unsqueeze(0).expand(args.batch_size, 3, -1, -1)
        adv_deformed = cloth(adv_patch)
        composite = adv_deformed * mask_3d + imgs * (1 - mask_3d)

        # Clear hooks
        attacker.features = {}

        # Forward — model in train mode returns list of raw yolo outputs
        yolo_outputs = model(composite)

        # Per-layer dim loss — final energy strategy
        layer_loss, layer_stats = attacker.compute_layer_loss(
            w_human=args.w_human, w_meter=args.w_meter, w_shared=args.w_shared)

        # Head loss (objectness + class) — FOCUSED on patch region
        obj_loss, cls_loss = attacker.compute_head_loss(yolo_outputs, mask_2d=mask)

        # Regularization
        tv = tv_fn(patch.unsqueeze(0))
        nps = nps_loss(patch)
        area = mask.mean()
        shape_loss = torch.clamp(area - args.area_limit, min=0) * args.w_shape

        # Total loss
        loss = (
            args.w_obj * obj_loss
            + args.w_cls * cls_loss
            + args.w_layer * layer_loss
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
            with torch.no_grad():
                model.eval()
                m = dap_mask()
                m3 = m.unsqueeze(0).unsqueeze(0)
                # Eval on neutral background
                bg = torch.full((1, 3, CANVAS, CANVAS), 0.5, device=DEVICE)
                comp = patch.unsqueeze(0) * m3 + bg * (1 - m3)
                eval_out = model(comp)
                if isinstance(eval_out, (list, tuple)):
                    eval_out = eval_out[0] if len(eval_out) > 0 else eval_out
                if isinstance(eval_out, torch.Tensor) and eval_out.dim() == 3:
                    obj_conf = eval_out[0, :, 4].max().item()
                    person_conf = eval_out[0, :, 5].max().item()
                    meter_conf = eval_out[0, :, 17].max().item()
                else:
                    obj_conf = person_conf = meter_conf = 0.0
                model.train()
                for p in model.parameters():
                    p.requires_grad_(False)

            print(f'Ep {epoch:5d}/{args.epochs} '
                  f'obj={obj_loss.item():.3f} cls={cls_loss.item():.3f} '
                  f'lay={layer_loss.item():.3f} '
                  f'h={layer_stats["h_act"]:.1f} m={layer_stats["m_act"]:.1f} s={layer_stats["s_act"]:.1f} '
                  f'p={person_conf:.3f} m={meter_conf:.3f} o={obj_conf:.3f} '
                  f'tv={tv.item():.5f} nps={nps.item():.4f} area={area.item():.3f}')

        if epoch % args.save_every == 0 or epoch == args.epochs:
            out_dir = os.path.join(base, args.out)
            T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(out_dir, f'texture_epoch{epoch:05d}.png'))
            T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(out_dir, f'mask_epoch{epoch:05d}.png'))
            print(f'  saved epoch {epoch}')

    # Final save
    out_dir = os.path.join(base, args.out)
    T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(out_dir, 'texture_final.png'))
    T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(out_dir, 'mask_final.png'))

    # Final eval — test on neutral AND person backgrounds
    print(f'\n=== Final evaluation ===')
    model.eval()
    with torch.no_grad():
        m = dap_mask()
        m3 = m.unsqueeze(0).unsqueeze(0)

        # Test 1: Neutral background (patch is only signal)
        bg = torch.full((1, 3, CANVAS, CANVAS), 0.5, device=DEVICE)
        comp = patch.unsqueeze(0) * m3 + bg * (1 - m3)
        out = model(comp)
        if isinstance(out, (list, tuple)):
            out = out[0] if len(out) > 0 else out
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            print(f'Neutral bg:  person={out[0, :, 5].max().item():.3f}  '
                  f'meter={out[0, :, 17].max().item():.3f}  '
                  f'obj={out[0, :, 4].max().item():.3f}')
            avg_p = out[0, :, 5].max().item()
            avg_m = out[0, :, 17].max().item()
            avg_o = out[0, :, 4].max().item()

        # Test 2: Person background (if images available)
        person_imgs = load_person_images(os.path.join(base, args.imgs), 20)
        if person_imgs is not None:
            p_results = []
            for i in range(min(10, len(person_imgs))):
                comp_p = patch.unsqueeze(0) * m3 + person_imgs[i:i+1] * (1 - m3)
                out_p = model(comp_p)
                if isinstance(out_p, (list, tuple)):
                    out_p = out_p[0] if len(out_p) > 0 else out_p
                if isinstance(out_p, torch.Tensor) and out_p.dim() == 3:
                    p_results.append((out_p[0, :, 5].max().item(),
                                      out_p[0, :, 17].max().item(),
                                      out_p[0, :, 4].max().item()))
            if p_results:
                print(f'Person bg:   person={np.mean([r[0] for r in p_results]):.3f}  '
                      f'meter={np.mean([r[1] for r in p_results]):.3f}  '
                      f'obj={np.mean([r[2] for r in p_results]):.3f}')

        fname = export_print_ready(patch.cpu(), out_dir, avg_p*100, avg_m*100, avg_o*100, args.epochs)
        print(f'Print-ready: {fname}')

    attacker.remove_hooks()
    print(f'Done. Output: {out_dir}')


if __name__ == '__main__':
    main()
