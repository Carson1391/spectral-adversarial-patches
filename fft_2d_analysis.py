#!/usr/bin/env python3
"""
2D spatial FFT analysis: per-channel frequency differences between human and no-human.

No sorting by magnitude. We keep spatial structure (C,H,W), compute 2D rfft2 per
channel, and measure spectral difference between the two images per channel.
Channels with largest frequency differences are the unique human signature channels.
"""
import os, sys, math

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")
DEVICE = torch.device('cuda')
IMG_SIZE = 416


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def register_all_hooks(model):
    features = {}
    hooks = []
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o.detach()
        return fn
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(n)))
    return features, hooks


def load_img(path):
    from PIL import Image
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def get_features(model, features, img_path):
    img = load_img(img_path)
    features.clear()
    with torch.no_grad():
        _ = model(img)
    return {n: t[0].clone() for n, t in features.items() if len(t.shape) == 4}


def channel_2d_fft_spectrum(f):
    """
    f: (C, H, W) tensor
    Returns (C, H_f, W_f) normalized 2D FFT magnitude spectra per channel.
    Original version: linear magnitude, no log2 transform.
    """
    fft = torch.fft.rfft2(f, dim=(-2, -1))
    mag = fft.abs()  # (C, H_f, W_f)
    mag = mag + 1e-8
    max_per_ch = mag.amax(dim=(1,2), keepdim=True)
    norm = mag / max_per_ch.clamp(min=1e-8)
    return norm


def analyze_layer_2d(layer_name, f_human, f_nohuman, out_dir, top_k=12):
    """
    For each channel, compute 2D FFT difference between human and no-human.
    Rank channels by total spectral difference.
    Visualize top differentiating channels.
    """
    spec_h = channel_2d_fft_spectrum(f_human)  # (C, H_f, W_f)
    spec_n = channel_2d_fft_spectrum(f_nohuman)

    diff = (spec_h - spec_n).abs()  # (C, H_f, W_f)
    diff_score = diff.sum(dim=(1, 2))  # (C,)

    top_idx = torch.argsort(diff_score, descending=True)[:top_k].cpu().numpy()
    bot_idx = torch.argsort(diff_score, descending=False)[:top_k//2].cpu().numpy()

    C, H, W = f_human.shape

    # Plot top differentiating channels
    n_cols = 4
    n_rows = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    fig.suptitle(f"Layer: {layer_name} | Top {top_k} channels by 2D FFT difference", fontsize=14)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, ch in enumerate(top_idx):
        # Human activation map
        axes[i, 0].imshow(f_human[ch].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f"Ch {ch} Human Act")
        axes[i, 0].axis('off')

        # No-human activation map
        axes[i, 1].imshow(f_nohuman[ch].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f"Ch {ch} No-Human Act")
        axes[i, 1].axis('off')

        # Human 2D FFT
        axes[i, 2].imshow(spec_h[ch].cpu().numpy(), cmap='hot')
        axes[i, 2].set_title(f"Ch {ch} Human FFT")
        axes[i, 2].axis('off')

        # Difference FFT
        axes[i, 3].imshow(diff[ch].cpu().numpy(), cmap='seismic')
        axes[i, 3].set_title(f"Ch {ch} |FFT_H - FFT_NH| (score={diff_score[ch]:.2f})")
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = layer_name.replace('/', '_').replace('.', '_')
    path = os.path.join(out_dir, f"{safe_name}_top{top_k}_2dfft.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save per-channel diff scores
    return {
        'top_channels': top_idx.tolist(),
        'top_scores': diff_score[top_idx].cpu().numpy().tolist(),
        'bot_channels': bot_idx.tolist(),
        'bot_scores': diff_score[bot_idx].cpu().numpy().tolist(),
        'mean_diff': diff_score.mean().item(),
        'max_diff': diff_score.max().item(),
        'human_mean_act': f_human.mean(dim=(1,2)).cpu().numpy().tolist(),
        'nohuman_mean_act': f_nohuman.mean(dim=(1,2)).cpu().numpy().tolist(),
        'freq_diff_maps': diff[top_idx].cpu().numpy(),
        'path': path
    }


def build_target_frequency_masks(results, patch_h=224, patch_w=224):
    """
    For top differentiating channels, resize their frequency-difference maps to
    patch size and save as synthesis targets.
    """
    import torch.nn.functional as F
    targets = {}
    for layer_name, res in results.items():
        if 'freq_diff_maps' not in res:
            continue
        diff_maps = torch.tensor(res['freq_diff_maps'], dtype=torch.float32)  # (K, H_f, W_f)
        resized = F.interpolate(diff_maps.unsqueeze(1), size=(patch_h, patch_w//2 + 1), mode='bilinear', align_corners=True)
        targets[layer_name] = resized.squeeze(1).numpy()
    return targets


def main():
    model = load_frozen_model()
    features, hooks = register_all_hooks(model)

    human_path = os.path.join(BASE, "withhuman.png")
    nohuman_path = os.path.join(BASE, "withouthuman.png")

    print("Extracting features...")
    feat_human = get_features(model, features, human_path)
    feat_nohuman = get_features(model, features, nohuman_path)

    out_dir = os.path.join(BASE, "fft_2d_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Analyze only key layers: early, mid, FPN/Neck, and detection-head layers
    all_layers = list(feat_human.keys())
    key_layers = []
    for n in all_layers:
        try:
            num = int(n.split('.')[1].split('_')[0])
            if num >= 75 or num <= 10 or num in [41, 47, 53, 59, 87, 93, 99]:
                key_layers.append(n)
        except:
            pass
    layer_names = sorted(set(key_layers))
    print(f"Analyzing {len(layer_names)} key layers")

    results = {}
    for layer_name in layer_names:
        res = analyze_layer_2d(layer_name, feat_human[layer_name], feat_nohuman[layer_name], out_dir, top_k=12)
        results[layer_name] = res
        print(f"{layer_name}: top ch={res['top_channels'][:5]}, scores={res['top_scores'][:5]}, mean_diff={res['mean_diff']:.3f}")

    # Build synthesis targets
    targets = build_target_frequency_masks(results)
    np.save(os.path.join(out_dir, "target_frequency_masks.npy"), targets)

    # Save summary
    import json
    summary = []
    for layer_name, res in results.items():
        summary.append({
            'layer': layer_name,
            'top_channels': res['top_channels'],
            'top_scores': res['top_scores'],
            'mean_diff': res['mean_diff'],
            'max_diff': res['max_diff'],
            'plot': os.path.basename(res['path'])
        })
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Saved to {out_dir}")
    print(f"Plots: {len(layer_names)}")
    print(f"Target frequency masks: {len(targets)} layers")

    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
