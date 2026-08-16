#!/usr/bin/env python3
"""
v16: Feature-level adversarial clothing trainer.

Attack at the feature embedding level, not the sigmoid output.
- Hold attention: keep objectness high so the box survives (model sees "something here")
- Redirect: push feature embedding x toward w_parking_meter, away from w_person
- Harmonic loss: d_i = ||w_i - x||_2, p_i = (1/d_i^n) / sum(1/d_j^n)
- DAP triangle mask for cloth deformation
- Ensemble: YOLOv3u + YOLOv8n + YOLO11n
- Custom base image for visual look
"""
import os, sys, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

DEVICE = torch.device('cuda')
PERSON = 0
PARKING_METER = 12


class DAPTriangleMask(nn.Module):
    def __init__(self, img_size=640, num_rays=64, lam=-100.0):
        super().__init__()
        self.img_size = img_size
        self.num_rays = num_rays
        self.lam = lam
        init_len = img_size * 0.4
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
    def __init__(self, img_size=640):
        super().__init__()
        self.img_size = img_size

    def forward(self, patch):
        b, c, h, w = patch.shape
        y = torch.linspace(-1, 1, h, device=patch.device).view(1, -1, 1)
        x = torch.linspace(-1, 1, w, device=patch.device).view(1, 1, -1)
        ox = 0.04 * torch.sin(2 * math.pi * (y * 2 + torch.rand(b, 1, 1, device=patch.device) * 2 * math.pi))
        oy = 0.04 * torch.cos(2 * math.pi * (x * 2 + torch.rand(b, 1, 1, device=patch.device) * 2 * math.pi))
        grid = torch.stack([(x + ox).expand(b, h, w), (y + oy).expand(b, h, w)], dim=-1)
        patch = F.grid_sample(patch, grid, mode='bilinear', padding_mode='border', align_corners=True)
        contrast = torch.rand(b, 1, 1, 1, device=patch.device) * 0.3 + 0.85
        brightness = torch.rand(b, 1, 1, 1, device=patch.device) * 0.15 - 0.075
        return (patch * contrast + brightness).clamp(0, 1)


class TotalVariation(nn.Module):
    def forward(self, adv_patch):
        tv1 = torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 1e-6).sum()
        tv2 = torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 1e-6).sum()
        return (tv1 + tv2) / adv_patch.numel()


class FeatureExtractor:
    """Extract feature embedding x and class weights w from YOLO classification head."""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.cls_layer_name = None
        self.D = None
        self.n = None

        # Find the final classification conv layer (output 80 channels)
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight
                if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
                    self.cls_layer_name = name
                    self.cls_module = module
                    self.D = w.shape[1]  # input channels = feature dim
                    self.n = math.sqrt(self.D)
                    self.w_flat = w.reshape(80, -1).detach()  # (80, D*kh*kw)
                    # For simplicity, use mean over kernel: (80, D)
                    self.w_flat = w.mean(dim=(2,3)).detach()  # (80, D)
                    print(f'  Found cls layer: {name}, D={self.D}, n={self.n:.1f}')
                    break

        def hook(module, input, output):
            # input[0] is (B, D, H, W) - the feature embedding before classification
            self.features['x'] = input[0]

        self.cls_module.register_forward_hook(hook)

    def get_features_and_scores(self, img_input):
        """Run model, return feature embedding x and harmonic distances."""
        out = self.model(img_input)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # (B, 4+80, anchors)

        # Get feature embedding from hook
        x = self.features['x']  # (B, D, H_feat, W_feat)
        B, D, Hf, Wf = x.shape
        n_anchors = Hf * Wf

        # Flatten x to (B, D, anchors)
        x_flat = x.reshape(B, D, n_anchors)

        # Class weights: w_flat is (80, D)
        # Compute L2 distance per anchor: d_i = ||w_i - x||_2
        # w_flat: (80, D), x_flat: (B, D, anchors)
        # d[b, c, a] = ||w[c, :] - x[b, :, a]||_2
        # Use broadcasting: w (80, D) -> (80, D, 1), x (B, D, anchors) -> (B, 80, D, anchors) via expand
        w = self.w_flat.to(x.device)  # (80, D)
        # diff shape: (B, 80, D, anchors) - too much memory for large grids
        # Instead compute per-batch: for each image, d = cdist(x.T, w) -> (anchors, 80)
        d_list = []
        for b in range(B):
            xb = x_flat[b]  # (D, anchors)
            # d[c, a] = ||w[c] - x[:, a]||_2 = sqrt(||w[c]||^2 - 2*w[c].dot(x[:,a]) + ||x[:,a]||^2)
            w_norm = (w ** 2).sum(dim=1)  # (80,)
            x_norm = (xb ** 2).sum(dim=0)  # (anchors,)
            dot = w @ xb  # (80, anchors)
            d_b = torch.sqrt(torch.clamp(w_norm.unsqueeze(1) - 2 * dot + x_norm.unsqueeze(0), min=1e-8))  # (80, anchors)
            d_list.append(d_b)
        d = torch.stack(d_list)  # (B, 80, anchors)

        # Harmonic probabilities: p_i = (1/d_i^n) / sum_j (1/d_j^n)
        # Use log-space for numerical stability: log p_i = -n*log(d_i) - logsumexp(-n*log(d_j))
        log_d = torch.log(d + 1e-8)  # (B, 80, anchors)
        log_inv_d = -self.n * log_d  # (B, 80, anchors)
        log_norm = torch.logsumexp(log_inv_d, dim=1, keepdim=True)  # (B, 1, anchors)
        log_p = log_inv_d - log_norm  # (B, 80, anchors)
        p = torch.exp(log_p)  # (B, 80, anchors)

        # Objectness from raw output (channel 4 = objectness-ish in v8 format)
        # Actually v8 output is (B, 4+80, anchors) with no explicit objectness
        # Use max class prob as confidence proxy
        confidence = p.max(dim=1)[0]  # (B, anchors)

        return {
            'x': x_flat,  # (B, D, anchors) feature embedding
            'd': d,  # (B, 80, anchors) L2 distances
            'p': p,  # (B, 80, anchors) harmonic probabilities
            'confidence': confidence,  # (B, anchors)
            'pred': pred,
            'n_anchors': n_anchors,
        }


def feature_level_loss(feat_ext, img_batch, gt_boxes, iou_thresh=0.45, phase=1,
                       perfect_meter_x=None, human_x=None):
    """
    Attack using REAL feature embeddings from the model:
    1. Shared dims (44): BOOST toward meter — part of meter, holds attention
    2. Human-only dims (58): drive to 0 or negative — kill person, poison future training
    3. Meter-only dims (34): DON'T TOUCH — already correct
    4. Other dims: IGNORE
    5. HarMax: -log(p_meter) targeting the real perfect_meter_x
    """
    img_input = F.interpolate(img_batch, (640, 640), mode='bilinear', align_corners=False)
    img_input = img_input[:, [2, 1, 0], :, :] * 255.0

    info = feat_ext.get_features_and_scores(img_input)
    pred = info['pred']
    p = info['p']
    confidence = info['confidence']
    x_flat = info['x']
    d = info['d']

    B = pred.shape[0]
    n_anchors = info['n_anchors']
    D = feat_ext.D

    # Use real embeddings to classify dims
    human_abs = human_x.abs()
    meter_abs = perfect_meter_x.abs()
    h_thresh = human_abs.max() * 0.1
    m_thresh = meter_abs.max() * 0.1

    person_idx = ((human_abs > h_thresh) & (meter_abs < m_thresh)).nonzero(as_tuple=True)[0]
    meter_idx = ((meter_abs > m_thresh) & (human_abs < h_thresh)).nonzero(as_tuple=True)[0]
    shared_idx = ((human_abs > h_thresh) & (meter_abs > m_thresh)).nonzero(as_tuple=True)[0]

    grid_size = int(math.sqrt(n_anchors))
    cell_size = 640.0 / grid_size

    losses = []
    for i in range(B):
        conf = confidence[i]
        gt = gt_boxes[i].to(pred.device)
        gt_cx = (gt[0] + gt[2]) / 2 / cell_size
        gt_cy = (gt[1] + gt[3]) / 2 / cell_size
        gt_w = (gt[2] - gt[0]) / cell_size
        gt_h = (gt[3] - gt[1]) / cell_size
        anchor_ys = torch.arange(grid_size, device=pred.device).float().unsqueeze(1).expand(grid_size, grid_size).reshape(-1)
        anchor_xs = torch.arange(grid_size, device=pred.device).float().unsqueeze(0).expand(grid_size, grid_size).reshape(-1)
        x1 = gt_cx - gt_w / 2 - 1; y1 = gt_cy - gt_h / 2 - 1
        x2 = gt_cx + gt_w / 2 + 1; y2 = gt_cy + gt_h / 2 + 1
        inside = (anchor_xs >= x1) & (anchor_xs <= x2) & (anchor_ys >= y1) & (anchor_ys <= y2)
        top_idx = inside.nonzero(as_tuple=True)[0] if inside.sum() > 0 else conf.topk(min(10, n_anchors)).indices

        x_loc = x_flat[i][:, top_idx]  # (D, n_loc)

        # 1. SHARED (44): BOOST toward meter — highest weight
        if len(shared_idx) > 0:
            x_s = x_loc[shared_idx, :]
            meter_target = perfect_meter_x[shared_idx].unsqueeze(1)  # (n_shared, 1)
            # Pull shared dims toward the perfect meter's shared values
            shared_loss = F.mse_loss(x_s, meter_target.expand_as(x_s))
        else:
            shared_loss = torch.tensor(0.0, device=pred.device)

        # 2. HUMAN-ONLY (58): drive to 0 or negative — kill person
        if len(person_idx) > 0:
            x_p = x_loc[person_idx, :]
            # Drive toward 0 (or negative). Use ReLU to only penalize positive values.
            # This pushes human-only features down to zero, then stops.
            # For poisoning: allow going negative by using mean (not ReLU)
            person_loss = x_p.mean()  # minimize → pushes negative
        else:
            person_loss = torch.tensor(0.0, device=pred.device)

        # 3. METER-ONLY (34): DON'T TOUCH — no loss term

        # 4. OTHER: IGNORE — no loss term

        # 5. HarMax: -log(p_meter) targeting real perfect_meter_x
        # Compute L2 distance to the real perfect meter embedding
        # x_loc is (D, n_loc), perfect_meter_x is (D,)
        # d_meter[a] = ||perfect_meter_x - x_loc[:, a]||_2
        d_meter = torch.norm(perfect_meter_x.unsqueeze(1) - x_loc, dim=0)  # (n_loc,)
        # d_person[a] = ||human_x - x_loc[:, a]||_2
        d_person = torch.norm(human_x.unsqueeze(1) - x_loc, dim=0)  # (n_loc,)

        # HarMax probability for meter using real distances
        # Need distances to ALL class anchors for proper normalization
        # Use the precomputed d from feat_ext for all 80 classes, but replace
        # d_meter with our real distance
        d_all = d[i, :, top_idx].clone()  # (80, n_loc)
        d_all[PARKING_METER, :] = d_meter  # replace with real distance

        log_d = torch.log(d_all + 1e-8)
        log_inv_d = -feat_ext.n * log_d
        log_norm = torch.logsumexp(log_inv_d, dim=0, keepdim=True)
        log_p = log_inv_d - log_norm
        harm_loss = -log_p[PARKING_METER, :].mean()

        # Hold attention
        conf_loss = -conf[top_idx].mean()

        loss = (
            3.0 * shared_loss     # shared: BOOST toward real meter (highest)
            + 2.0 * person_loss    # human-only: drive to 0/negative
            + 0.5 * harm_loss       # HarMax: -log(p_meter) with real distances
            + 0.3 * conf_loss        # hold attention
        )
        losses.append(loss)

    return torch.stack(losses).mean()


def load_person_images(imgs_dir, img_size=640, n=300):
    paths = sorted([os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:n]
    detector = YOLO('YOLOv3/yolov3u.pt')
    results = []
    for p in paths:
        img = Image.open(p).convert('RGB').resize((img_size, img_size))
        img_t = T.ToTensor()(img)
        det = detector.predict(img, verbose=False, classes=[0])[0]
        if len(det.boxes) > 0:
            best = det.boxes.conf.argmax()
            box = det.boxes.xyxy[best].cpu()
            orig_w, orig_h = Image.open(p).size
            box = box * torch.tensor([img_size/orig_w, img_size/orig_h, img_size/orig_w, img_size/orig_h])
            results.append((img_t, box))
    print(f'Loaded {len(results)} person images with ground truth boxes')
    return results


def main(args):
    os.makedirs(args.out, exist_ok=True)

    # Load base image
    base_tensor = None
    if args.base and os.path.exists(args.base):
        base_pil = Image.open(args.base).convert('RGB').resize((args.img_size, args.img_size))
        base_tensor = T.ToTensor()(base_pil).to(DEVICE)
        patch = base_tensor.clone()
        print(f'Base image: {args.base}')
    else:
        patch = torch.rand(3, args.img_size, args.img_size, device=DEVICE) * 0.3 + 0.35
    patch = torch.nn.Parameter(patch)

    # DAP mask
    dap_mask = DAPTriangleMask(img_size=args.img_size, num_rays=args.num_rays).to(DEVICE)
    cloth_deform = ClothDeformation(img_size=args.img_size).to(DEVICE)
    tv_loss_fn = TotalVariation().to(DEVICE)

    # Load models — v3 only for training (Flock target), saves ~60% VRAM
    models = {}
    feat_extractors = {}
    if os.path.exists('YOLOv3/yolov3u.pt'):
        m = YOLO('YOLOv3/yolov3u.pt').to(DEVICE)
        m.model.eval()
        for p in m.model.parameters(): p.requires_grad = False
        models['v3'] = m
        feat_extractors['v3'] = FeatureExtractor(m.model)
    print(f'Models: {list(models.keys())}')

    person_data = load_person_images(args.imgs, args.img_size, args.max_imgs)

    # Load real feature embeddings
    perfect_meter_x = torch.load('outputs_clothing/v16_harmonic/perfect_meter_v3.pt').to(DEVICE)
    human_x = torch.load('outputs_clothing/v16_harmonic/perfect_human_v3.pt').to(DEVICE)
    print(f'Loaded perfect_meter_x (norm={perfect_meter_x.norm():.2f}) and human_x (norm={human_x.norm():.2f})')

    tex_opt = torch.optim.Adam([patch], lr=args.lr_tex, amsgrad=True)
    shape_opt = torch.optim.Adam([dap_mask.ray_lengths], lr=args.lr_shape, amsgrad=True)

    W_TV = args.w_tv
    W_SHAPE = args.w_shape
    AREA_LIMIT = args.area_limit

    for epoch in range(1, args.epochs + 1):
        # Phase 1: HarMax only (epochs 1 to phase1_epochs)
        # Phase 2: HarMax + per-dim poisoning (epochs phase1_epochs+1 to end)
        phase = 1 if epoch <= args.phase1_epochs else 2
        if epoch % 100 == 90:
            tex_opt.param_groups[0]['lr'] /= args.lr_decay
            shape_opt.param_groups[0]['lr'] /= args.lr_decay

        total_det = 0; total_tv = 0; total_loss = 0; n_batches = 0

        indices = torch.randperm(len(person_data))
        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) < 1: continue
            actual_bs = len(batch_idx)

            imgs = torch.stack([person_data[i][0] for i in batch_idx]).to(DEVICE)
            gt_boxes = [person_data[i][1] for i in batch_idx]

            tex_opt.zero_grad(); shape_opt.zero_grad()

            mask = dap_mask()
            mask_3d = mask.unsqueeze(0).unsqueeze(0).expand(actual_bs, 3, -1, -1)
            adv_patch = patch.unsqueeze(0).expand(actual_bs, -1, -1, -1)
            adv_deformed = cloth_deform(adv_patch)
            composite = adv_deformed * mask_3d + imgs * (1 - mask_3d)

            # Feature-level loss — v3 only
            det_loss = 0
            for name in models:
                det_loss += feature_level_loss(feat_extractors[name], composite, gt_boxes, args.iou_thresh, phase=phase,
                                               perfect_meter_x=perfect_meter_x, human_x=human_x)

            tv = tv_loss_fn(patch.unsqueeze(0))
            area = mask.mean()
            shape_loss = torch.clamp(area - AREA_LIMIT, min=0) * W_SHAPE

            loss = det_loss + W_TV * tv + shape_loss

            loss.backward()
            tex_opt.step(); shape_opt.step()

            with torch.no_grad():
                patch.clamp_(0, 1)
                dap_mask.ray_lengths.clamp_(min=10, max=args.img_size * 0.5)

            total_det += det_loss.item(); total_tv += tv.item()
            total_loss += loss.item(); n_batches += 1

        if n_batches > 0:
            area = dap_mask.get_area()
            print(f'Ep {epoch:4d}/{args.epochs} phase={phase} det={total_det/n_batches:.4f} tv={total_tv/n_batches:.6f} area={area:.3f} loss={total_loss/n_batches:.4f} lr={tex_opt.param_groups[0]["lr"]:.5f}')

        if epoch % args.save_every == 0:
            T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(args.out, f'texture_epoch{epoch:05d}.png'))
            T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(args.out, f'mask_epoch{epoch:05d}.png'))
            with torch.no_grad():
                mask_v = dap_mask()
                sample = person_data[0][0].to(DEVICE)
                comp = patch * mask_v + sample * (1 - mask_v)
                T.ToPILImage()(comp.clamp(0,1).cpu()).save(os.path.join(args.out, f'composite_epoch{epoch:05d}.png'))
            print(f'  saved epoch {epoch}')

    T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(args.out, 'texture_final.png'))
    T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(args.out, 'mask_final.png'))
    print('Final saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='data/galaxy_style_ref.png')
    parser.add_argument('--imgs', default='data/coco_person/images')
    parser.add_argument('--out', default='outputs_clothing/v16_harmonic')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_imgs', type=int, default=300)
    parser.add_argument('--num_rays', type=int, default=64)
    parser.add_argument('--lr_tex', type=float, default=0.01)
    parser.add_argument('--lr_shape', type=float, default=0.005)
    parser.add_argument('--lr_decay', type=float, default=1.1)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_shape', type=float, default=200.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--iou_thresh', type=float, default=0.45)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--phase1_epochs', type=int, default=400, help='Epochs for HarMax-only phase before adding per-dim poisoning')
    args = parser.parse_args()
    main(args)
