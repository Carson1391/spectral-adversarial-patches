# YODO — Adversarial Patch Research for YOLOv3 Evasion

**AI security research portfolio — frequency-domain adversarial attacks on object detection**

Systematic investigation of how structured frequency patterns disrupt YOLOv3 person detection. 11 analyses spanning graph theory, spectral decomposition, interference physics, and cross-model testing. The research identified specific vulnerable frequencies (k=196), person-anchor channels (170), and a frequency cascade through the network that informed a multi-scale fractal patch design achieving complete evasion after gradient optimization.

---

## Research Question

Can mathematically constructed frequency patterns evade YOLOv3 person detection more reliably than gradient-optimized noise, and if so, what are the mechanisms?

## Methodology

**Paired image comparison**: Forward-pass `withhuman.png` and `withouthuman.png` through all 75 conv layers. Subtract feature maps to isolate the *person signal* — the activation delta that means "human present." Analyze this delta in frequency domain, graph structure, and embedding space.

**Model**: YOLOv3 Darknet-53, 416x416 input, COCO weights, CUDA (RTX 5060 Ti).

---

## Key Findings

### 1. Graph Laplacian — Channel Isolation (`l2_fft_laplacian_kfac.py`)

Built channel-level correlation graphs for every conv layer. Mid-backbone layers (37–75) have near-total channel isolation — 87–96% of channels are disconnected from each other. Person-sensitive channels operate independently. Detection heads (81, 93, 105) recombine everything densely.

| Layer | Channels | Isolated % | Interpretation |
| ------- | ---------- | ----------- | ---------------- |
| 37 | 512 | 96.7% | Maximum specialization |
| 54 | 512 | 95.1% | Person channels are isolated specialists |
| 62 | 1024 | 87.0% | Sparse, 13% connected |
| 75 | 512 | 96.1% | Near-total specialization |
| 81 | 255 | 5.1% | Detection head recombines |

**Implication**: Disrupting one person channel doesn't cascade. You need broadband frequency content to hit multiple isolated channels simultaneously.

### 2. Person-Sensitive Layers (`forward_person_delta.py`)

Ranked every layer by activation change when a human is present:

| Rank | Layer | Shape | Mean Rel Delta | Key Channels |
| ------ | ------- | ------- | --------------- | -------------- |
| 1 | **54** | 512×26×26 | **55.9%** | 479, 31, 51, 184, 422 |
| 2 | **75** | 512×13×13 | **50.4%** | **170**, 17, 322, 84, 292 |
| 3 | 84 | 256×13×13 | 45.3% | 167, 234, 32, 211, 157 |
| 4 | 63 | 512×13×13 | 42.3% | 422, 147, 47, 8, 406 |

### 3. Cross-Layer Person Anchor Channels

| Channel | Appears At | Role |
| --------- | ----------- | ------ |
| **170** | 75, 93, 105 | Person anchor — all three detection scales |
| **171** | 93, 105 | Person anchor at medium and fine scales |
| **422** | 54, 63 | Deep person feature across backbone |
| **47** | 54, 63, 75 | Consistent person feature across deep layers |

Channel 170 is the single most important neuron for person detection — it appears at every detection head. Targeting it directly is the most efficient attack vector.

### 4. Frequency Cascade Through the Network (`freq_analysis.py`)

The person signal undergoes a frequency transformation as it propagates:

| Depth | Layers | Dominant Band | Interpretation |
| ------- | -------- | -------------- | ---------------- |
| Early | 0–5 | LF (76–96%) | Person is a large low-frequency blob |
| Mid | 12–37 | LF→MF transition | Edges, limbs, boundaries encoded |
| Deep | 54–60 | LF + MF peak (26–28%) | Medium-scale patterns at 26×26 |
| Deepest | **62–75** | **MF dominant (41–43%)** | At 13×13, person is high-frequency |
| Heads | 81–105 | LF returns (53–94%) | Decision collapses to low frequency |

**Critical insight**: Layers 62–75 are the bottleneck where the person signal shifts from low to mid-frequency. This is the most vulnerable point — a patch that injects strong MF energy here disrupts the signal before it reaches detection heads.

### 5. Interference Physics (`interference_analysis.py`, `constructive_interference.py`, `destructive_interference.py`)

Tested whether patch frequencies constructively or destructively interfere with the person signal:

- **Destructive interference** (phase cancellation): Requires precise phase alignment with person FFT — fragile, breaks under any geometric transform.
- **Constructive interference** (amplification + saturation): More robust. Overwhelm person channels with stronger same-frequency content, causing activation saturation and downstream suppression.
- **Cross-spectrum analysis**: Person-background cross-spectrum is near-zero above LF — the person signal is uncorrelated with background at MF/HF, confirming frequency-domain attack surface.

### 6. Cross-Model Testing (`triangular_patch_test.py`)

Tested 10 shapes × 27 textures across YOLOv3, YOLOv8, YOLO11, and YOLO26:

**Best single frequency**: **k=196** — suppresses person on all 4 models, minimal hallucinations.

| Control | v3 Dets | v3 Conf | v8 Dets | v26 Dets |
| --------- | --------- | --------- | --------- | ---------- |
| uniform_gray | 13 | 0.955 | 1 | 1 |
| random_noise | 8 | 0.522 | 1 | 1 |
| **k196 sinusoid** | **3** | **0.425** | 1 | 1 |

On YOLOv3, k196 is **2.7× more effective than noise** and **4.3× more than gray**. This confirms frequency-targeted corruption, not generic visual disruption.

**Shape efficiency**: Triangle (7.6% area) outperforms circle (11.6% area) per-pixel. Sharp corners create stronger frequency content at mask boundaries.

**Model robustness ranking**: YOLOv8 (most fragile) < YOLO11 < YOLOv3 < YOLO26 (most robust).

### 7. Hallucination Patterns

Sinusoid textures trigger class-specific hallucinations on Ultralytics models:

| Frequency | v8 | v11 | v26 |
| ----------- | ----- | ----- | ----- |
| k208 | sports ball, umbrella | umbrella, traffic light | umbrella |
| k196 | bench | traffic light | None |
| composite | bench, train | traffic light, bench | None |

k208 consistently triggers "umbrella" across all Ultralytics models — it activates channels that share features with curved edges and canopy textures.

---

## Patch Design: From Findings to Implementation

The analyses converged on a design:

1. **Multi-scale fractal** — k=49, 98, 196, 392 simultaneously. At any capture distance, one harmonic lands in the vulnerable MF band (layers 62–75).
2. **Sierpinski triangle mask** — 7.6% area, sharp corners for boundary frequency content, nested scales for broadband coverage.
3. **FFT phase-shift depth encoding** — per-harmonic phase shifts with 1/n falloff create multi-plane depth hallucination in feature maps.
4. **1/196 digit modulation** — 42-cycle coprime to 2^n for downsampling armor.
5. **Gradient optimization** — maximize L2 disruption at detection heads (L81/93/105) constrained to mask region.

### Results

| Metric | Math-Only | After Optimization |
| --- | --- | --- |
| Person detections | 5 → 5 | 4 → **0** |
| Combined confidence | 0.9995 → 0.9783 | 0.9995 → **0.0033** |
| Objectness | 0.9995 → 0.9783 | 0.9995 → **0.9379** |
| L81 cosine disruption | 0.36% | **5.25%** |
| L93 cosine disruption | 0.13% | **3.73%** |

The math-only construction provides a strong prior. The optimization loop converts it to complete evasion.

### Physical Scale Robustness

| Capture Width | k=49 eff. | k=98 eff. | k=196 eff. | k=392 eff. |
| --------------- | ----------- | ----------- | ------------- | ------------- |
| 416px (digital) | 49 | 98 | 196 | 392 |
| 300px | 35 | 71 | 141 | 283 |
| 200px | 24 | 47 | 94 | **188** |
| 500px | 59 | 118 | **235** | 471 |

At any capture distance, at least one harmonic lands in the vulnerable k=167–208 band.

---

## Repository Structure

### Analysis Scripts (research phase)

| File | Analysis | Key Output |
| --- | --- | --- |
| `l2_fft_laplacian_kfac.py` | Graph Laplacian + L2/FFT/KFAC | Channel isolation maps, Fiedler vectors |
| `forward_person_delta.py` | Person signal extraction + saliency | Person-sensitive layer rankings, anchor channels |
| `freq_analysis.py` | 1D/2D/Polynomial FFT | Frequency cascade, per-layer band ratios |
| `interference_analysis.py` | 4-way interference (raw + embedding) | Cross-spectrum, LF/MF/HF energy splits |
| `constructive_interference.py` | Constructive interference + HF propagation | Activation saturation patterns |
| `destructive_interference.py` | Destructive interference (phase cancel) | Phase alignment requirements |
| `triangular_patch_test.py` | Cross-model: 10 shapes × 27 textures × 4 models | k196 identified as optimal frequency |
| `capture_embeddings_webcam.py` | Live webcam embedding capture | Real-time feature visualization |

### Patch Generation & Evaluation (implementation phase)

| File | Purpose |
| --- | --- |
| `final_boss.py` | Sierpinski fractal patch + k-spread + FFT depth + gradient optimization |
| `test_patch_metrics.py` | 9-metric evaluation suite (cosine, L2, FFT, frequency bands, distance, angle, quantization, saliency, detection scores) |
| `deformable_patch.py` | Deformable polygon mask variant (Shape Matters paper) with gradient-preserving forward pass |
| `train_v45_fft_signature.py` | Alternative approach: match patch FFT to human-specific frequency signatures |
| `fractal_patch.py` | Shared utilities: forward pass, embeddings, FFT, patch compositing |
| `fractal_image_patch.py` | Self-similar image tiling into Sierpinski triangle structure |

### Documentation

| File | Content |
|---|---|
| `MASTER_FINDINGS.md` | Full research document — all 11 analyses with methodology, results, and interpretation |

### Results

| Directory | Content |
| --- | --- |
| `results/COMPREHENSIVE_SUMMARY.json` | Master results across all analyses |
| `results/person_signal_summary.json` | Person-sensitive layer rankings + anchor channels |
| `results/figures/` | Key figures: cross-layer summary, fractal comparison, shape/texture comparison, interference, frequency cascade |
| `results/fractal/` | Fractal patch images (d3/d4/d5 at 416px) + per-layer channel delta CSVs |
| `results/triangular/` | Cross-model test data: 10 shapes × 27 textures × 4 models (CSV) |
| `results/interference/` | 4-way interference analysis: raw + embedding FFT |
| `results/freq_deep/` | Deep frequency analysis: per-layer band ratios + frequency cascade |

---

## Setup

```bash
git clone https://github.com/Carson1391/yodo.git
cd yodo

# Requirements
pip install torch numpy pillow matplotlib opencv-python

# YOLOv3 model
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git

# Download weights (248MB — not in repo)
# Place yolov3.weights in project root
# https://pjreddie.com/media/files/yolov3.weights

# Test images (not in repo)
# Place withhuman.png and withouthuman.png in project root

# Run analyses
python forward_person_delta.py      # Person signal extraction
python freq_analysis.py             # Deep frequency analysis
python triangular_patch_test.py     # Cross-model texture testing

# Generate and evaluate patches
python final_boss.py                # Build + optimize fractal patch
python test_patch_metrics.py        # Full 9-metric evaluation
```

---

## Key Design Decisions

- **Frequency-domain attack surface**: CNN activations have well-defined frequency responses. Targeting specific bands (MF at layers 62–75) is more reliable than pixel-space perturbation.
- **Construction before optimization**: Mathematical structure provides a strong prior. Pure optimization from random noise converges to weaker local minima.
- **Channel 170 targeting**: The cross-scale person anchor channel is the single highest-value attack target. k=196 specifically excites it.
- **Triangle over circle**: Sharp corners produce stronger frequency content at mask boundaries — 7.6% area triangle outperforms 11.6% circle.
- **Physical constraints**: 300 DPI printable, 7.6% area, designed for fabric — not just a digital attack.

---

## License

Research code — academic exploration of adversarial vulnerabilities in object detection systems.
