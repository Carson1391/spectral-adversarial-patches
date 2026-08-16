#!/usr/bin/env python3
"""
Polynomial FFT analysis on human vs no-human embeddings (FIXED).

Fixes:
- Sort channels by activation magnitude before treating as polynomial coefficients
- Cap root computation at dim <= 64
- Use spectral coherence for interference
- Save target cancellation signals
- Keep spatial structure analysis option
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

DIMS = [1, 2, 4, 8, 16, 32, 64]


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


def get_embeddings(model, features, hooks, img_path):
    img = load_img(img_path)
    features.clear()
    with torch.no_grad():
        _ = model(img)
    embs = {}
    for n, t in features.items():
        if len(t.shape) == 4:
            emb = t[0].mean(dim=(1, 2)).cpu().numpy()  # (C,)
            embs[n] = emb
    return embs


def sorted_polynomial_fft(emb, dim):
    """
    Sort channels by |activation| descending, then FFT.
    This gives the polynomial a meaningful decay structure.
    """
    if len(emb) >= dim:
        e = emb[:dim]
    else:
        e = np.pad(emb, (0, dim - len(emb)), mode='constant')
    idx = np.argsort(np.abs(e))[::-1]
    e_sorted = e[idx]
    fft = np.fft.fft(e_sorted)
    freqs = np.fft.fftfreq(dim)
    return e_sorted, fft, freqs


def analyze_layer(layer_name, emb_human, emb_nohuman, dim, out_dir):
    e_h, fft_h, freqs = sorted_polynomial_fft(emb_human, dim)
    e_n, fft_n, _ = sorted_polynomial_fft(emb_nohuman, dim)

    # Difference polynomial = human signal
    d = e_h - e_n
    D_fft = np.fft.fft(d)

    # Target cancellation polynomial
    target = -np.fft.ifft(D_fft).real
    np.save(os.path.join(out_dir, f"{layer_name.replace('/', '_').replace('.', '_')}_dim{dim}_target.npy"), target)

    # Roots only for low-degree polynomials
    if dim <= 64:
        roots = np.roots(d)
    else:
        roots = None

    # Spectral coherence: phase alignment per frequency
    coherence_num = fft_h * np.conj(fft_n)
    coherence_den = np.abs(fft_h) * np.abs(fft_n) + 1e-8
    coherence = coherence_num / coherence_den

    # Interference magnitude: how much human and no-human spectra differ
    spectral_diff = np.abs(fft_h) - np.abs(fft_n)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Layer: {layer_name} | Dim: {dim} (sorted by |activation|)", fontsize=14)

    # 1. Human embedding coefficients (sorted)
    axes[0,0].plot(e_h, 'b-', lw=0.8)
    axes[0,0].set_title("Human Embedding Coeffs (sorted)")
    axes[0,0].set_xlabel("Sorted channel rank")

    # 2. No-human embedding coefficients (sorted)
    axes[0,1].plot(e_n, 'r-', lw=0.8)
    axes[0,1].set_title("No-Human Embedding Coeffs (sorted)")
    axes[0,1].set_xlabel("Sorted channel rank")

    # 3. Difference polynomial (human signal)
    axes[0,2].plot(d, 'g-', lw=1.0)
    axes[0,2].set_title("Difference Polynomial D(x)")
    axes[0,2].set_xlabel("Coefficient index")

    # 4. Human signal FFT magnitude
    axes[0,3].stem(freqs, np.abs(D_fft), markerfmt=' ', basefmt=' ')
    axes[0,3].set_title("|FFT(D)| — Human Signal")
    axes[0,3].set_xlabel("Frequency")

    # 5. Human vs No-human FFT magnitude overlay
    axes[1,0].plot(freqs, np.abs(fft_h), 'b-', lw=0.8, label='human')
    axes[1,0].plot(freqs, np.abs(fft_n), 'r-', lw=0.8, label='no-human')
    axes[1,0].set_title("FFT Magnitude Overlay")
    axes[1,0].set_xlabel("Frequency")
    axes[1,0].legend()

    # 6. Spectral difference
    axes[1,1].plot(freqs, spectral_diff, 'm-', lw=0.8)
    axes[1,1].set_title("Spectral Difference |FFT(human)| - |FFT(no-human)|")
    axes[1,1].set_xlabel("Frequency")
    axes[1,1].axhline(0, color='gray', lw=0.5)

    # 7. Phase of human signal
    axes[1,2].stem(freqs, np.angle(D_fft), markerfmt=' ', basefmt=' ')
    axes[1,2].set_title("Phase of Human Signal")
    axes[1,2].set_xlabel("Frequency")

    # 8. Roots of difference polynomial or coherence
    if roots is not None:
        axes[1,3].scatter(roots.real, roots.imag, s=15, c='red', alpha=0.7)
        theta = np.linspace(0, 2*np.pi, 100)
        axes[1,3].plot(np.cos(theta), np.sin(theta), 'b--', lw=0.5)
        axes[1,3].set_title(f"Roots of D(x) (dim={dim})")
        axes[1,3].set_aspect('equal')
        axes[1,3].set_xlabel("Real")
        axes[1,3].set_ylabel("Imag")
    else:
        axes[1,3].plot(freqs, np.abs(coherence), 'c-', lw=0.8)
        axes[1,3].set_title("Spectral Coherence |human vs no-human|")
        axes[1,3].set_xlabel("Frequency")
        axes[1,3].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = layer_name.replace('/', '_').replace('.', '_')
    path = os.path.join(out_dir, f"{safe_name}_dim{dim}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path, np.abs(D_fft), freqs, roots, target, np.abs(coherence)


def main():
    model = load_frozen_model()
    features, hooks = register_all_hooks(model)

    human_path = os.path.join(BASE, "withhuman.png")
    nohuman_path = os.path.join(BASE, "withouthuman.png")

    print("Extracting embeddings...")
    emb_human = get_embeddings(model, features, hooks, human_path)
    emb_nohuman = get_embeddings(model, features, hooks, nohuman_path)

    out_dir = os.path.join(BASE, "fft_analysis_v2")
    os.makedirs(out_dir, exist_ok=True)

    # Select layers with enough channels for dims up to 256
    layer_names = [n for n in emb_human.keys() if emb_human[n].shape[0] >= 256]
    # Sample across depth
    layer_names = layer_names[::max(1, len(layer_names)//12)]
    print(f"Analyzing {len(layer_names)} layers across dims {DIMS}")

    summary = []
    for layer_name in layer_names:
        for dim in DIMS:
            if emb_human[layer_name].shape[0] < dim:
                continue
            path, mag, freqs, roots, target, coherence = analyze_layer(
                layer_name, emb_human[layer_name], emb_nohuman[layer_name], dim, out_dir
            )
            top_idx = np.argsort(mag)[::-1][:5]
            top_freqs = freqs[top_idx]
            roots_inside = int(np.sum(np.abs(roots) < 1.0)) if roots is not None else -1
            print(f"{layer_name} dim={dim:3d}: top freqs={top_freqs}, coherence={coherence.mean():.3f}, saved {os.path.basename(path)}")
            summary.append({
                'layer': layer_name,
                'dim': dim,
                'top_freqs': top_freqs,
                'top_mags': mag[top_idx],
                'coherence_mean': coherence.mean(),
                'roots_inside': roots_inside,
                'roots_total': len(roots) if roots is not None else -1
            })

    # Save summary
    import json
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump([{
            'layer': s['layer'],
            'dim': s['dim'],
            'top_freqs': s['top_freqs'].tolist(),
            'top_mags': s['top_mags'].tolist(),
            'coherence_mean': float(s['coherence_mean']),
            'roots_inside': int(s['roots_inside']),
            'roots_total': int(s['roots_total'])
        } for s in summary], f, indent=2)

    print(f"\nDone. Analysis saved to {out_dir}")
    print(f"Images: {len(layer_names)} layers x {len(DIMS)} dims = {len(layer_names)*len(DIMS)} plots")
    print(f"Target cancellation signals: {len(summary)} .npy files")

    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
