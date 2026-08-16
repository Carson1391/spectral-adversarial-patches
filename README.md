# YODO — Adversarial Patch Attacks on YOLOv3

Research project exploring adversarial patch generation against YOLOv3 object detection, focused on person-class evasion via structured frequency-domain attacks.

## Core Concept

Instead of gradient-based pixel optimization alone, patches are constructed from mathematically structured patterns (Sierpinski fractals, Cartesian sinusoids, FFT phase shifts) designed to disrupt specific detection head activations while maintaining physical printability.

## Key Files

| File | Purpose |
|---|---|
| `final_boss.py` | Main patch generation: nested Sierpinski triangles, k-spread frequencies, FFT depth encoding, Pascal mod-n void geometry, gradient optimization loop |
| `test_patch_metrics.py` | Comprehensive test suite: cosine similarity, L2/FFT distance, distance degradation, angle sensitivity, quantization effects, saliency maps, detection scores |
| `deformable_patch.py` | Deformable patch variant with TPS warping for physical robustness |
| `train_v45_fft_signature.py` | FFT signature attack: matches patch FFT to human-specific frequency signatures extracted from model activations |

## Quick Start

```bash
# Generate and optimize a patch
python final_boss.py

# Run full metric suite against a patch
python test_patch_metrics.py
```

## Requirements

- CUDA-capable GPU (tested on RTX 5060 Ti)
- PyTorch with CUDA
- YOLOv3 weights (`yolov3.weights`) — download separately
- See `PyTorch-YOLOv3/` for the model implementation

## Output Structure

```
outputs_clothing/final_boss_v2/
  patch_416.png              # Math-only patch
  patch_416_optimized.png    # Gradient-optimized patch
  mask_416.png               # Patch mask
  composite.png              # Patch overlaid on human
  metrics/                   # Test results (JSON, CSV, figures)
```

## Technical Details

- **k-spread**: Compound wave numbers per Sierpinski branch level for broadband frequency targeting
- **Fourier section**: Per-harmonic FFT phase shifts with 1/n magnitude falloff for multi-plane depth encoding
- **Pascal mod-n**: Variable void geometry (mod 2 = standard Sierpinski, mod 3/5/7 = alternative patterns)
- **1/196 modulation**: 42-cycle coprime persistence for downsampling armor
- **Optimization**: Gradient-based evasion maximizing L2 activation disruption at detection heads (layers 81, 93, 105)

## Metrics Tracked

- Cosine similarity (gap + point embeddings) per layer
- L2 norm shift per layer
- FFT spectral distance
- Person overlap (directional attack on person signal)
- Frequency band analysis (LF/MF/HF energy split)
- Distance degradation (5m–60m simulated)
- Angle sensitivity (yaw 0–90°, pitch 0–30°)
- Quantization robustness (FP32/INT8/INT4)
- Detection scores (objectness, person class probability, n_detections)
