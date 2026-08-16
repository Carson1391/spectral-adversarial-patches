#!/usr/bin/env python3
"""
v17: NMS Saturation Trainer — Fractal Human Attack.

Attack at the NMS layer, NOT the classifier.
- Encode a perfect human feature map via DAP triangles
- Model outputs 500+ overlapping 95%+ confidence boxes
- NMS paralysis → detection dies
- Loss = -(sum_confidence + spatial_variance_of_box_centers)
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


class NMSGaussian(nn.Module):
    """
    NMS Saturation Loss.
    
    Objective: maximize (alpha * sum_confidence + beta * spatial_variance)
    We minimize the negative: -(alpha * sum_conf + beta * var_centers)
    
    This forces the model to output many scattered high-confidence human boxes,
    paralyzing NMS.
    """
    def __init__(self, person_class=0, conf_thresh=0.5, nms_iou=0.5):
        super().__init__()
        self.person_class = person_class
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou

    def forward(self, pred_raw, img_size, mask=None):
        """
        pred_raw: (B, 4+80, n_anchors) — YOLOv3 raw output
                  [x_center, y_center, w, h, class0_prob, class1_prob, ...]
        img_size: grid size in pixels (e.g., 640)
        mask: (B, 3, H, W) — DAP triangle mask
        Returns: dict with loss components and stats
        """
        B, C, n_anchors = pred_raw.shape
        device = pred_raw.device
        
        # Scale box coordinates to pixel space
        scale = img_size / 640.0  # normalize to grid, then to pixels
        x_center = pred_raw[..., 0] * scale
        y_center = pred_raw[..., 1] * scale
        w_box = pred_raw[..., 2] * scale
        h_box = pred_raw[..., 3] * scale
        
        # Objectness * class probability = final confidence
        class_prob = pred_raw[..., self.person_class + 4]  # class 0 is person
        obj_score = pred_raw[..., 4]  # objectness
        confidence = obj_score * class_prob
        
        # Count high-confidence boxes
        conf_mask = confidence > self.conf_thresh
        
        n_high_conf = conf_mask.sum().float()
        
        # Extract centers of high-confidence boxes
        xc = x_center[conf_mask]
        yc = y_center[conf_mask]
        
        # Spatial variance of box centers
        if n_high_conf > 1:
            var_x = torch.var(xc)
            var_y = torch.var(yc)
            spatial_var = var_x + var_y
        else:
            spatial_var = torch.tensor(0.0, device=device)
        
        # Sum of all confidence scores (total activation volume)
        total_conf = confidence.sum()
        
        # NMS IoU estimate: how many high-conf boxes overlap each other
        # Use a simple IoU matrix for the high-conf boxes (sparse)
        if n_high_conf > 100:
            # Subsample for speed
            indices = torch.randperm(n_high_conf, device=device)[:100]
            xc_s = xc[indices]
            yc_s = yc[indices]
            w_s = w_box[conf_mask][indices]
            h_s = h_box[conf_mask][indices]
            
            # IoU pairs
            iou = []
            for i in range(len(xc_s)):
                x1 = xc_s[i] - w_s[i] / 2
                y1 = yc_s[i] - h_s[i] / 2
                x2 = xc_s[i] + w_s[i] / 2
                y2 = yc_s[i] + h_s[i] / 2
                
                iou_sum = 0.0
                cnt = 0
                for j in range(i+1, len(xc_s)):
                    x1j = xc_s[j] - w_s[j] / 2
                    y1j = yc_s[j] - h_s[j] / 2
                    x2j = xc_s[j] + w_s[j] / 2
                    y2j = yc_s[j] + h_s[j] / 2
                    
                    inter_x1 = max(x1, x1j)
                    inter_y1 = max(y1, y1j)
                    inter_x2 = min(x2, x2j)
                    inter_y2 = min(y2, y2j)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (x2j - x1j) * (y2j - y1j)
                        union = area1 + area2 - inter
                        iou_val = inter / union if union > 0 else 0
                        iou_sum += iou_val
                        cnt += 1
                
                iou.append(iou_sum / cnt if cnt > 0 else 0)
            
            mean_iou = torch.tensor(iou, device=device).mean()
        else:
            mean_iou = torch.tensor(0.0, device=device)
        
        # The NMS saturation loss: maximize confidence + spatial variance
        # We minimize negative of these
        loss = -(alpha * total_conf + beta * spatial_var)
        
        return {
            'loss': loss,
            'n_high_conf': n_high_conf,
            'total_conf': total_conf,
            'spatial_var': spatial_var,
            'mean_iou': mean_iou,
            'conf_mask': conf_mask,
            'confidence': confidence,
            'x_center': x_center,
            'y_center': y_center,
            'w_box': w_box,
            'h_box': h_box,
            'class_prob': class_prob,
            'obj_score': obj_score,
        }


def nms_saturation_loss(pred_raw, img_size, mask, alpha=10.0, beta=5.0):
    """Convenience wrapper."""
    nms_fn = NMSGaussian(person_class=PERSON, conf_thresh=0.5)
    return nms_fn(pred_raw, img_size, mask)


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
    print(f'Patch shape: {patch.shape}, device: {patch.device}')
    
    # DAP mask + cloth deformation
    dap_mask = DAPTriangleMask(img_size=args.img_size, num_rays=args.num_rays).to(DEVICE)
    cloth_deform = ClothDeformation(img_size=args.img_size).to(DEVICE)
    tv_loss_fn = TotalVariation().to(DEVICE)
    
    # Load model — v3 only (Flock target)
    print('Loading YOLOv3u...')
    model = YOLO(f'YOLOv3/yolov3u.pt').to(DEVICE)
    model.model.eval()
    for p in model.model.parameters(): p.requires_grad = False
    
    # Find the classification output layer to hook into
    # We need the raw detection head output (before NMS)
    def hook_fn(module, input, output):
        # output is raw model prediction tensor
        pass
    
    # We'll use Ultralytics predict with no_grad to get raw preds
    # But we need differentiable access — so we use the raw model forward
    # YOLOv3 raw output: (B, 4+80, n_anchors) for each scale level
    # Total: (B, sum_of_anchors, 84)
    
    print(f'Model loaded: {list(model.model.modules())[-10:]}')
    
    # Load person images
    person_paths = sorted([os.path.join(args.imgs, f) for f in os.listdir(args.imgs) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:args.max_imgs]
    
    # Load base images for composite
    base_paths = sorted([os.path.join(args.bases, f) for f in os.listdir(args.bases)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:args.max_bases]
    
    # Combine base and person for training pairs
    # Alternate between bases and persons, or use person with DAP patch overlay
    print(f'Loaded {len(person_paths)} person images, {len(base_paths)} base images')
    
    # Optimizers
    tex_opt = torch.optim.Adam([patch], lr=args.lr_tex, amsgrad=True)
    shape_opt = torch.optim.Adam([dap_mask.ray_lengths], lr=args.lr_shape, amsgrad=True)
    
    W_TV = args.w_tv
    W_SHAPE = args.w_shape
    AREA_LIMIT = args.area_limit
    
    nms_fn = NMSGaussian(person_class=PERSON, conf_thresh=args.conf_thresh)
    
    for epoch in range(1, args.epochs + 1):
        tex_opt.zero_grad()
        shape_opt.zero_grad()
        
        batch_idx = list(range(min(args.batch_size, len(person_paths))))
        
        # Get person images
        imgs = []
        for i in batch_idx:
            img = Image.open(person_paths[i]).convert('RGB').resize((args.img_size, args.img_size))
            imgs.append(T.ToTensor()(img).to(DEVICE))
        
        gt_boxes = []
        for i in batch_idx:
            img_pil = Image.open(person_paths[i]).convert('RGB')
            orig_w, orig_h = img_pil.size
            gt = torch.tensor([orig_w * 0.3, orig_h * 0.3, orig_w * 0.7, orig_h * 0.7], device=DEVICE)
            gt_boxes.append(gt)
        
        mask = dap_mask()
        mask_3d = mask.unsqueeze(0).unsqueeze(0).expand(len(batch_idx), 3, -1, -1)
        adv_patch = patch.unsqueeze(0).expand(len(batch_idx), -1, -1, -1)
        adv_deformed = cloth_deform(adv_patch)
        composite = adv_deformed * mask_3d + torch.stack(imgs).unsqueeze(1).expand(-1, 3, -1, -1) * (1 - mask_3d)
        
        # NMS Saturation Loss
        nms_info = nms_fn(composite, args.img_size, mask_3d)
        det_loss = nms_info['loss']
        
        tv = tv_loss_fn(patch.unsqueeze(0))
        area = mask.mean()
        shape_loss = torch.clamp(area - AREA_LIMIT, min=0) * W_SHAPE
        
        loss = det_loss + W_TV * tv + shape_loss
        
        loss.backward()
        tex_opt.step()
        shape_opt.step()
        
        with torch.no_grad():
            patch.clamp_(0, 1)
            dap_mask.ray_lengths.clamp_(min=10, max=args.img_size * 0.5)
        
        n_high = nms_info['n_high_conf'].item()
        tv_val = tv.item()
        total_conf = nms_info['total_conf'].item()
        spatial_var = nms_info['spatial_var'].item()
        
        print(f'Ep {epoch:5d}/{args.epochs} n_high_conf={n_high:6.0f} total_conf={total_conf:.2f} '
              f'spatial_var={spatial_var:.2f} tv={tv_val:.6f} area={area:.3f} loss={-det_loss.item():.4f} '
              f'lr_tex={tex_opt.param_groups[0]["lr"]:.5f}')
        
        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs:
            T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(args.out, f'texture_epoch{epoch:05d}.png'))
            T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(args.out, f'mask_epoch{epoch:05d}.png'))
            print(f'  saved epoch {epoch}')
    
    # Final save
    T.ToPILImage()(patch.clamp(0,1)).save(os.path.join(args.out, 'texture_final.png'))
    T.ToPILImage()(dap_mask().detach().cpu()).save(os.path.join(args.out, 'mask_final.png'))
    print('Final saved.')
    
    # Evaluate final patch
    print('\n=== Evaluating final patch ===')
    evaluate_final(patch, dap_mask, args, base_paths, person_paths, args.img_size)


def evaluate_final(patch, dap_mask, args, base_paths, person_paths, img_size):
    """Run the final patch through the model and report NMS stats."""
    from ultralytics import YOLO
    
    model = YOLO('YOLOv3/yolov3u.pt').to(DEVICE)
    model.model.eval()
    
    tex = T.ToTensor()(Image.open(os.path.join(args.out, 'texture_final.png')).convert('RGB')).unsqueeze(0).to(DEVICE)
    mask_l = Image.open(os.path.join(args.out, 'mask_final.png')).convert('L').resize(tex.shape[2:])
    mask_t = T.ToTensor()(mask_l).unsqueeze(0).unsqueeze(1).to(DEVICE)
    
    tex_r = F.interpolate(tex, (img_size, img_size), mode='bilinear', align_corners=False)
    mask_r = F.interpolate(mask_t, (img_size, img_size), mode='bilinear', align_corners=False)
    
    print('\n--- Raw model output (before NMS) ---')
    raw = model.model(tex_r * mask_r + (1 - mask_r) * tex.unsqueeze(0))
    print(f'Raw output shape: {raw.shape}')
    
    # Sum confidence
    conf = raw[..., 4] * raw[..., 5]  # obj * person_class
    high_conf = (conf > 0.5).sum().item()
    print(f'High-confidence boxes (>0.5): {high_conf}')
    print(f'Total confidence sum: {conf.sum().item():.2f}')
    
    # After NMS
    print('\n--- After NMS (Ultralytics) ---')
    comp = tex_r * mask_r + (1 - mask_r) * tex.unsqueeze(0)
    comp_pil = T.ToPILImage()(comp.squeeze(0).clamp(0, 1))
    res = model.predict(comp_pil, verbose=False)[0]
    print(f'Detections after NMS: {len(res.boxes)}')
    if len(res.boxes) > 0:
        for cls, conf_val in zip(res.boxes.cls, res.boxes.conf):
            print(f'  Class {cls}: {conf_val:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='data/galaxy_style_ref.png', help='Base image path')
    parser.add_argument('--bases', default='data/galaxy_style_ref.png')
    parser.add_argument('--imgs', default='data/coco_person_strong/images')
    parser.add_argument('--out', default='outputs_clothing/v17_nms_saturation')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_imgs', type=int, default=200)
    parser.add_argument('--max_bases', type=int, default=200)
    parser.add_argument('--num_rays', type=int, default=64)
    parser.add_argument('--lr_tex', type=float, default=0.01)
    parser.add_argument('--lr_shape', type=float, default=0.005)
    parser.add_argument('--lr_decay', type=float, default=1.1)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_shape', type=float, default=200.0)
    parser.add_argument('--area_limit', type=float, default=0.25)
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='NMS confidence threshold for box counting')
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()
    main(args)
