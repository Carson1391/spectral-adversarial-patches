#!/usr/bin/env python3
"""
v20 — Graph-isolated attack loss.
Only patch_tensor carries grad. No NMS, no rescale, no graph bloat.
Top-K anchor slicing, cloned canvas, detached area weights.
"""
import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Faithful Darknet YOLOv3 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")


def load_model_frozen():
    model = load_model(model_path=CFG, weights_path=WEIGHTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_attack_loss(model, patch_tensor, canvas_size=416, person_id=0):
    """Graph-isolated loss. Only patch_tensor carries grad."""
    canvas = torch.full((3, canvas_size, canvas_size), 0.5, device=patch_tensor.device)
    pH, pW = patch_tensor.shape[1], patch_tensor.shape[2]
    y0, x0 = (canvas_size - pH) // 2, (canvas_size - pW) // 2
    inp = canvas.clone()
    inp[:, y0:y0+pH, x0:x0+pW] = patch_tensor
    inp = inp.unsqueeze(0)

    with torch.enable_grad():
        preds = model(inp)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        # Raw logits — pre-sigmoid
        obj_logit = preds[:, 4:5, :]           # (1, 1, N)
        cls_logit = preds[:, 5 + person_id:6 + person_id, :]  # (1, 1, N)

        conf = (obj_logit.sigmoid() * cls_logit.sigmoid()).squeeze(0).squeeze(0)  # (N,)

        with torch.no_grad():
            xyxy = preds[:, :4, :].squeeze(0)
            w = (xyxy[2] - xyxy[0]).clamp(min=0)
            h = (xyxy[3] - xyxy[1]).clamp(min=0)
            area = (w * h).clamp(min=1e-3)
            weight = (area / area.max().clamp(min=1.0)).pow(0.5)

        k = 500
        if conf.shape[0] > k:
            top_vals, top_idx = conf.topk(k)
            weight = weight[top_idx]
        else:
            top_vals = conf

        loss = -(top_vals * weight).sum()
    return loss


def tv_loss(patch):
    """Total variation — smoothness."""
    return (torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]).mean() +
            torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]).mean())


def nps_loss(patch):
    """Nearest-printable-color — pull toward 16-color CMYK palette."""
    palette = torch.tensor([
        [0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[0,255,255],[255,0,255],
        [128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],[192,192,192],[64,64,64]
    ], device=patch.device, dtype=patch.dtype) / 255.0
    p = patch.permute(0, 2, 3, 1).reshape(-1, 3)
    dists = torch.cdist(p, palette)
    return dists.min(dim=1)[0].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_obj', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading faithful Darknet YOLOv3...")
    model = load_model_frozen().to(device)
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()):,} parameters frozen")

    # Init patch from base image or grey
    if args.base:
        base_path = os.path.join(BASE, args.base)
        img = Image.open(base_path).convert('RGB').resize((416, 416))
        patch_init = torch.tensor(np.array(img).transpose(2,0,1), dtype=torch.float32, device=device) / 255.0
        patch_init = patch_init.unsqueeze(0)
        print(f"Patch init from {args.base}")
    else:
        patch_init = torch.full((1, 3, 416, 416), 0.5, device=device)
        print("Patch init: grey")

    patch = patch_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=args.lr, amsgrad=True)

    out_dir = os.path.join(BASE, "outputs_clothing/v20_isolated")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== v20 Training ===")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Weights: obj={args.w_obj} tv={args.w_tv} nps={args.w_nps}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        loss = compute_attack_loss(model, patch, canvas_size=416, person_id=0)
        tv = tv_loss(patch)
        nps = nps_loss(patch)

        total = args.w_obj * loss + args.w_tv * tv + args.w_nps * nps
        total.backward()
        optimizer.step()

        with torch.no_grad():
            patch.clamp_(0, 1)

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                preds = model(patch)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                obj_s = preds[:, 4:5, :].sigmoid().mean().item()
                cls_s = preds[:, 5:85, :].sigmoid().mean(dim=1).max().item()
                person_s = preds[:, 5, :].sigmoid().mean().item()
            print(f"Ep {epoch:5d}/{args.epochs} "
                  f"obj_loss={loss.item():.3f} tv={tv.item():.4f} nps={nps.item():.4f} "
                  f"obj_s={obj_s:.3f} cls_max={cls_s:.3f} person={person_s:.3f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                img_np = (patch[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(out_dir, f"texture_epoch{epoch:04d}.png"))
                Image.fromarray(img_np).save(os.path.join(out_dir, "texture_final.png"))
            print(f"  saved epoch {epoch}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
