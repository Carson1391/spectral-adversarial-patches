"""
Archetype Patch Generator — FIXED.

Black-box surrogate attack against edge-YOLOv3 (fine-tuned pico3).
Surrogate model: yolov3u.pt (Darknet-53 backbone, shared ancestry).

Mechanism:
  Pure activation maximization on person-class logit (class 0).
  Start from nested-equilateral-triangle seed (4 scales: 1x,2x,4x,8x).
  Backpropagate gradient INTO THE INPUT IMAGE (not model weights).
  Iterate. Result = model's internal archetype of "person".

Distance robustness:
  Triangle hierarchy covers multiple receptive-field scales natively.
  Additional EOT-style resize augmentations during optimization simulate
  viewer-distance variation.

Output:
  Printable PNG + tensor NPZ + config JSON.

FIXES from original archetype.py.txt:
  1. Ultralytics YOLOv3u output is (B, 4+nc, A) where [4:4+nc] are class
     confidence scores DIRECTLY (no separate objectness channel). Original
     code treated index 4 as objectness and index 5+cls as class prob,
     computing obj*cls_p = class0*class1 — nonsensical.
  2. Area weighting check (<=CANVAS_HOST+1) failed because box coords can
     reach ~648 on a 640 canvas. Fixed threshold to allow margin.
  3. Output indexing: preds[batch, channel, anchor] not preds[batch, anchor, channel].
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("pip install ultralytics") from e

log = logging.getLogger("archetype")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERSON_CLASS_ID = 0  # COCO class 0 = person
CANVAS_HOST = 640
PATCH_SIZE = 224
NUM_SCALES = 4
SCALE_FACTORS = (1.0, 0.5, 0.25, 0.125)
INIT_RADII_FRAC = (0.45, 0.30, 0.20, 0.12)


# ---------------------------------------------------------------------------
# Triangle seed renderer
# ---------------------------------------------------------------------------

def _meshgrid(size: int, dev: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=dev),
        torch.arange(size, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _soft_triangle_alpha(verts: torch.Tensor, pts: torch.Tensor, sigma: float) -> torch.Tensor:
    sds = []
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        vi, vj, vk = verts[i], verts[j], verts[k]
        edge = vj - vi
        normal = torch.stack([-edge[1], edge[0]])
        normal = normal / (normal.norm() + 1e-8)
        if torch.dot(normal, vk - vi) > 0:
            normal = -normal
        sds.append((pts - vi.unsqueeze(0)) @ normal)
    sdf = torch.stack(sds, dim=0).min(dim=0).values
    return torch.sigmoid(sdf / sigma)


def render_triangle_seed(size: int = PATCH_SIZE, sigma: float = 1.5, dev: torch.device = DEVICE) -> torch.Tensor:
    """Return (3,size,size) float32 in [0,1]: nested colored triangles."""
    pts = _meshgrid(size, dev)
    canvas = torch.zeros((3, size, size), device=dev)
    cx = cy = size / 2.0
    rng = torch.Generator(device=dev).manual_seed(0)

    for idx, sf in enumerate(SCALE_FACTORS):
        r = size * INIT_RADII_FRAC[idx] * sf
        rot = torch.rand(1, generator=rng, device=dev).item() * 2 * math.pi
        ang = rot + torch.linspace(0, 4 * math.pi / 3, 3, device=dev)  # 3 vertices: 0, 2π/3, 4π/3
        verts = torch.stack([cx + r * torch.cos(ang), cy + r * torch.sin(ang)], dim=-1)
        alpha = _soft_triangle_alpha(verts, pts, sigma).view(size, size).unsqueeze(0)
        col = torch.rand((3, 1, 1), generator=rng, device=dev) * 0.6 + 0.2
        canvas = canvas * (1 - alpha) + col * alpha

    return canvas.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Frozen YOLOv3-u wrapper exposing person-logit
# ---------------------------------------------------------------------------

class ArchetypeTarget:
    def __init__(self, weights: str):
        log.info("Loading surrogate %s on %s", weights, DEVICE)
        yolo = YOLO(weights).to(DEVICE)
        self.inner = yolo.model.to(DEVICE).eval()
        for p in self.inner.parameters():
            p.requires_grad_(False)
        log.info("Frozen. Parameters: %d", sum(p.numel() for p in self.inner.parameters()))

    def person_logit_sum(self, patch_chw: torch.Tensor) -> torch.Tensor:
        """
        Paste patch centrally on 640x640 grey canvas, forward, return SCALAR
        = sum of person-class confidence across all candidate boxes,
        weighted by sqrt(box area) to privilege large stride-32 firings.

        Ultralytics YOLOv3u output: (B, 4+nc, A) = (1, 84, 8400)
          [0:4]   = xyxy box coords (pixel space)
          [4:84]  = class confidence scores (sigmoid, unified — NO separate obj)
        Person class 0 confidence = preds[:, 4, :]
        """
        _, pH, pW = patch_chw.shape
        canvas = torch.full((3, CANVAS_HOST, CANVAS_HOST), 0.5, device=patch_chw.device)
        y0 = (CANVAS_HOST - pH) // 2
        x0 = (CANVAS_HOST - pW) // 2
        canvas[:, y0:y0 + pH, x0:x0 + pW] = patch_chw
        x = canvas.unsqueeze(0)

        out = self.inner(x)
        # Ultralytics returns tuple (preds, feature_maps)
        if isinstance(out, (list, tuple)):
            preds = out[0]
        else:
            preds = out

        # preds shape: (B, 84, 8400) = (B, 4+80, A)
        # FIX: person confidence is directly at index 4+PERSON_CLASS_ID = 4
        # No separate objectness channel in Ultralytics v8/v3u format
        person_conf = preds[0, 4 + PERSON_CLASS_ID, :]  # (8400,)

        # Box coords for area weighting: preds[0, 0:4, :] = (4, 8400) xyxy
        xyxy = preds[0, :4, :]  # (4, A)
        w = (xyxy[2, :] - xyxy[0, :]).clamp(min=0)
        h = (xyxy[3, :] - xyxy[1, :]).clamp(min=0)
        area = (w * h).clamp(min=1e-3)
        weight = (area / area.max().clamp(min=1.0)).pow(0.5)

        return (person_conf * weight).sum()


# ---------------------------------------------------------------------------
# EOT transform bank (distance simulation)
# ---------------------------------------------------------------------------

def eot_transforms(img_chw: torch.Tensor, n: int = 4) -> List[torch.Tensor]:
    """Generate resized copies simulating viewer distance variation."""
    out = [img_chw]
    _, h, w = img_chw.shape
    for s in [0.7, 1.3, 0.5, 1.6][:n - 1]:
        nh, nw = max(8, int(h * s)), max(8, int(w * s))
        scaled = F.interpolate(img_chw.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
        # Pad or crop back to original size, centered
        canvas = torch.full_like(img_chw, 0.5)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        sy, sx = max(0, y0), max(0, x0)
        dy, dx = max(0, -y0), max(0, -x0)
        eh = min(nh, h - sy)
        ew = min(nw, w - sx)
        canvas[:, sy:sy + eh, sx:sx + ew] = scaled[:, dy:dy + eh, dx:dx + ew]
        out.append(canvas)
    return out


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

@dataclass
class Config:
    iters: int = 600
    lr_pixel: float = 0.08
    tv_weight: float = 1e-3
    nps_weight: float = 5e-2          # loose printability
    eot_samples: int = 4
    sigma_anneal: Tuple[float, float] = (2.0, 0.6)
    snapshot_every: int = 50
    out_dir: str = "./runs/archetype_v1"
    weights: str = "yolov3u.pt"


def optimize(target: ArchetypeTarget, cfg: Config) -> Tuple[torch.Tensor, List[float]]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Init patch FROM TRIANGLE SEED, mark as leaf with grad
    patch = render_triangle_seed(PATCH_SIZE, sigma=cfg.sigma_anneal[0]).clone().detach_()
    patch.requires_grad_(True)

    opt = torch.optim.Adam([patch], lr=cfg.lr_pixel)
    history: List[float] = []

    for it in range(cfg.iters):
        frac = it / max(1, cfg.iters - 1)
        sigma = cfg.sigma_anneal[0] + (cfg.sigma_anneal[1] - cfg.sigma_anneal[0]) * frac

        views = eot_transforms(patch, n=cfg.eot_samples)
        total_loss = torch.tensor(0.0, device=DEVICE)
        total_score = 0.0

        for v in views:
            s = target.person_logit_sum(v)
            total_loss = total_loss - s
            total_score += float(s.item())

        # TV reg
        dx = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
        dy = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
        total_loss = total_loss + cfg.tv_weight * (dx + dy)

        # Loose NPS: penalize extreme outliers (keep printable-ish)
        total_loss = total_loss + cfg.nps_weight * ((patch - 0.5).pow(2).mean())

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        avg_score = total_score / len(views)
        history.append(avg_score)
        if it % 20 == 0 or it == cfg.iters - 1:
            log.info("iter %4d  avg_score=%.6f  sigma=%.2f", it, avg_score, sigma)
        if it % cfg.snapshot_every == 0 or it == cfg.iters - 1:
            snap = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(snap).save(out / f"snap_{it:04d}.png")

    final = patch.detach().cpu()
    png_arr = (final.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(png_arr).save(out / "archetype_final.png")
    np.savez_compressed(out / "tensor.npz", patch=final.numpy())
    with open(out / "config.json", "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    log.info("Done. Artefacts in %s", out.resolve())
    return final, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "train":
        cfg = Config(
            weights=r"C:\Users\carso\Desktop\YODO\YOLOv3\yolov3u.pt",
            out_dir=r"C:\Users\carso\Desktop\YODO\runs\archetype_v1",
        )
        tgt = ArchetypeTarget(cfg.weights)
        patch, hist = optimize(tgt, cfg)
        log.info("Final score: %.6f", hist[-1])
    elif cmd == "smoke":
        t = render_triangle_seed(224, sigma=1.5)
        log.info("Seed ok shape=%s range=[%.3f,%.3f]", tuple(t.shape), t.min().item(), t.max().item())
    else:
        print("usage: python archetype.py [train|smoke]")
