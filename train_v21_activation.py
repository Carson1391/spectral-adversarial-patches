#!/usr/bin/env python3
"""
v21 — Activation Maximization + DAPatch
Pipeline:
1. White noise init
2. Hook Class 0 (Person) feature channels in middle + late layers
3. Maximize mean activation across all hooked layers
4. Gradient ascent on pixels (frozen model)
5. DAPatch triangular deformation (TPS-style) during forward pass
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


class FeatureHook:
    """Hook to capture activations from a specific layer."""
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if isinstance(output, (tuple, list)):
            self.activation = output[0]
        else:
            self.activation = output

    def remove(self):
        self.handle.remove()


def load_model_frozen():
    model = load_model(model_path=CFG, weights_path=WEIGHTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def hook_person_layers(model):
    """
    Hook middle and late layers that correlate with Class 0 (Person).
    Middle layers: assemble components (shoulders, limbs, torso)
    Late layers: complete semantic concept of person
    Returns list of FeatureHook objects.
    """
    hooks = []
    # Darknet-53 backbone has ~100+ layers in module_list
    # Middle layers (~30-60): component assembly
    # Late layers (~70-100): semantic concepts
    # Detection heads are at the end
    layers_to_hook = [20, 30, 40, 50, 60, 70, 80, 85, 90, 95]

    for idx in layers_to_hook:
        if idx < len(model.module_list):
            layer = model.module_list[idx]
            hook = FeatureHook(layer)
            hooks.append(hook)

    print(f"Hooked {len(hooks)} layers: {layers_to_hook}")
    return hooks


def dapatch_deform(patch_tensor, intensity=0.05):
    """
    Differentiable triangular deformation (TPS-style).
    Simulates cloth physics: random perturbation of triangular mesh vertices.
    """
    B, C, H, W = patch_tensor.shape
    device = patch_tensor.device

    # Create grid
    grid = F.affine_grid(torch.eye(2, 3, device=device).unsqueeze(0), (B, C, H, W))

    # Add random triangular deformation
    # Simulate cloth wrinkles via smooth random displacement
    noise = torch.randn(B, 2, H // 4, W // 4, device=device) * intensity
    noise_up = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)

    grid_deformed = grid + noise_up.permute(0, 2, 3, 1)

    # Clamp to valid range
    grid_deformed = grid_deformed.clamp(-1, 1)

    deformed = F.grid_sample(patch_tensor, grid_deformed, mode='bilinear',
                             padding_mode='reflection', align_corners=False)
    return deformed


def compute_activation_loss(model, patch_tensor, hooks, person_class=0):
    """
    Maximize mean activation of Class 0 (Person) feature channels
    across all hooked layers simultaneously.
    """
    # Apply DAPatch deformation
    deformed = dapatch_deform(patch_tensor, intensity=0.03)

    # Forward pass
    inp = deformed
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)

    with torch.enable_grad():
        _ = model(inp)

        # Collect activations from all hooked layers
        total_activation = 0.0
        num_layers = 0

        for hook in hooks:
            if hook.activation is not None:
                act = hook.activation
                # Mean activation across spatial dimensions and batch
                # We want ALL feature channels to fire for person
                layer_mean = act.mean()
                total_activation = total_activation + layer_mean
                num_layers += 1

        if num_layers == 0:
            return torch.tensor(0.0, device=patch_tensor.device)

        # Negative because we want to MAXIMIZE (gradient ascent = minimize negative)
        loss = -total_activation / num_layers

    return loss


def tv_loss(patch):
    """Total variation — smoothness for physical printing."""
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--w_tv', type=float, default=2.5)
    parser.add_argument('--w_nps', type=float, default=0.1)
    parser.add_argument('--w_act', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading faithful Darknet YOLOv3...")
    model = load_model_frozen().to(device)
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()):,} parameters frozen")

    # 1. Initialize with pure random static (white noise)
    patch = torch.rand(1, 3, 416, 416, device=device, requires_grad=True)
    print("Patch init: pure random static (white noise)")

    # 2. Hook target feature channels
    hooks = hook_person_layers(model)

    optimizer = torch.optim.Adam([patch], lr=args.lr, amsgrad=True)

    out_dir = os.path.join(BASE, "outputs_clothing/v21_activation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== v21 Activation Maximization Training ===")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Weights: act={args.w_act} tv={args.w_tv} nps={args.w_nps}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # 3. Multi-layer activation objective
        act_loss = compute_activation_loss(model, patch, hooks, person_class=0)
        tv = tv_loss(patch)
        nps = nps_loss(patch)

        # 4. Gradient ascent on pixels
        total = args.w_act * act_loss + args.w_tv * tv + args.w_nps * nps
        total.backward()
        optimizer.step()

        with torch.no_grad():
            patch.clamp_(0, 1)

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                # Clear hooks for clean eval
                for h in hooks:
                    h.activation = None
                # Run clean forward pass to check activation
                _ = model(patch)
                act_sum = sum(h.activation.mean().item() for h in hooks if h.activation is not None)
                act_avg = act_sum / max(len(hooks), 1)
            print(f"Ep {epoch:5d}/{args.epochs} "
                  f"act_loss={act_loss.item():.4f} tv={tv.item():.4f} nps={nps.item():.4f} "
                  f"avg_activation={act_avg:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                img_np = (patch[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(out_dir, f"texture_epoch{epoch:04d}.png"))
                Image.fromarray(img_np).save(os.path.join(out_dir, "texture_final.png"))
            print(f"  saved epoch {epoch}")

    # Cleanup
    for h in hooks:
        h.remove()

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
