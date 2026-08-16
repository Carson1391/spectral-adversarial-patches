# Spectral Adversarial Patches — Frequency-Domain Attacks on YOLO Object Detectors

**Comprehensive research portfolio — 17 analyses spanning graph theory, spectral decomposition, interference physics, covert channels, and physical patch design.**

Systematic investigation of how structured frequency patterns disrupt YOLO object detection. The research progressed through four phases: (1) understanding where and how the network represents humans, (2) testing interference mechanisms, (3) discovering optimal attack frequencies and shapes, and (4) building a complete attack pipeline from physical patch to cloud poisoning.

---

## Research Question

Can mathematically constructed frequency patterns evade YOLO person detection more reliably than gradient-optimized noise? If so, what are the mechanisms, and can the attack be made stealthy enough to avoid collateral damage to bystander detections?

## Methodology

**Paired image comparison**: Forward-pass `withhuman.png` and `withouthuman.png` through all 75 conv layers of YOLOv3 Darknet-53. Subtract feature maps to isolate the *person signal* — the activation delta that means "human present." Analyze this delta in frequency domain, graph structure, embedding space, and under various perturbation regimes.

**Model**: YOLOv3 Darknet-53, 416×416 input, COCO weights, CUDA (RTX 5060 Ti). Cross-model testing on YOLOv8, YOLO11, YOLO26.

---

## Phase 1: Understanding the Target

### Graph Laplacian — Channel Isolation

Built channel-level correlation graphs for every conv layer. **Mid-backbone layers (37–75) have 87–97% isolated channels** — person-sensitive neurons operate independently. Detection heads (81, 93, 105) recombine everything densely (90%+ connected).

**Implication**: Disrupting one person channel doesn't cascade. You need broadband frequency content to hit multiple isolated channels simultaneously.

### Person-Sensitive Layer Rankings

| Rank | Layer | Shape | Mean Rel Delta | Key Channels |
|------|-------|-------|---------------|--------------|
| 1 | **54** | 512×26×26 | **55.9%** | 479, 31, 51, 184, 422 |
| 2 | **75** | 512×13×13 | **50.4%** | **170**, 17, 322, 84, 292 |
| 3 | 84 | 256×13×13 | 45.3% | 167, 234, 32, 211, 157 |
| 4 | 63 | 512×13×13 | 42.3% | 422, 147, 47, 8, 406 |

### Cross-Layer Person Anchor Channels

| Channel | Appears At | Role |
|---------|-----------|------|
| **170** | 75, 93, 105 | Person anchor — all three detection scales |
| **171** | 93, 105 | Person anchor at medium and fine scales |
| **422** | 54, 63 | Deep person feature across backbone |
| **47** | 54, 63, 75 | Consistent person feature across deep layers |

Channel 170 is the single most important neuron for person detection — it appears at every detection head.

### The Frequency Cascade

The person signal undergoes a frequency transformation through the network:

| Depth | Layers | Dominant Band | Interpretation |
|-------|--------|--------------|----------------|
| Early | 0–5 | LF (76–96%) | Person is a large low-frequency blob |
| Mid | 12–37 | LF→MF transition | Edges, limbs, boundaries encoded |
| Deep | 54–60 | LF + MF peak (26–28%) | Medium-scale patterns at 26×26 |
| Deepest | **62–75** | **MF dominant (41–43%)** | At 13×13, person is high-frequency |
| Heads | 81–105 | LF returns (53–94%) | Decision collapses to low frequency |

**Critical insight**: The network is a **frequency amplifier**. Input is 99.99% LF, but by layer 62 the delta is only 25.6% LF — the network has generated 74.4% MF+HF content from a near-pure-LF input. Layers 62–75 are the bottleneck where the person signal shifts from LF to MF. This is the most vulnerable point.

---

## Phase 2: Interference Mechanisms

### Constructive Interference — HF is 2× More Efficient

Injected sinusoidal patterns at various frequencies and measured suppression:

| Band | Best Frequency | Suppression Score | Efficiency (score/amp) |
|------|---------------|-------------------|----------------------|
| LF | k=5 | 0.098 | 0.49 |
| MF | k=50 | 0.288 | 1.44 |
| **HF** | **k=200 diagonal** | **0.542** | **2.71** |

**HF is ~2× more effective per unit amplitude than LF.** Diagonal HF (kx=ky=200) is the best suppressor. Phase is nearly irrelevant for HF — it oscillates so rapidly that phase averages out across the spatial extent.

### The Network as a Frequency Converter

| Injected | Delta HF at L0 | Delta HF at L62 | Delta HF at L75 |
|-----------|---------------|-----------------|-----------------|
| LF_k2 | 0.0038 | **0.2559** | **0.2761** |
| HF_k200 | 0.0159 | 0.0499 | 0.0595 |

When you inject LF, the network produces HF internally (0.0038 → 0.2559 at L62). When you inject HF, it attenuates but survives. **Injecting HF directly is more efficient because it bypasses the network's frequency conversion step.**

### Destructive Interference — No Silver Bullet

At every layer, the top-20 frequency bins by power are ALL LF (near-DC). The human signal's concentrated power is always at low frequencies, even when band ratios show significant HF. HF energy is **diffuse** — spread thinly across hundreds of bins. There is no single "silver bullet" HF frequency to cancel.

**Implication**: To cancel the HF portion, you need broadband HF noise covering all bins simultaneously. The most effective cancellation targets the DC/LF dominant bins.

### Amplitude is King

Monotonic increase in suppression with amplitude — no saturation up to amp=0.50. Every bit of additional oscillation energy contributes. **Repeated injection does NOT compound** — batch norm + LeakyReLU prevent runaway accumulation.

---

## Phase 3: Finding the Optimal Attack

### Prime Frequency Suppression — Total Detection Kill

High primes near Nyquist (k=157–199 diagonal) cause **total detection kill** (0 objects detected) on YOLOv3:

| Prime k | Suppression | Detections |
|---------|------------|------------|
| **167** | **0.476** | **0** |
| 199 | 0.466 | 0 |
| 197 | 0.466 | 0 |
| 193 | 0.465 | 0 |
| 191 | 0.465 | 0 |
| 157 | 0.458 | 0 |

These primes near Nyquist (208) create maximum aliasing — they can't be cleanly represented at any downsampled resolution, scattering energy across all feature scales.

### 13-Multiples Outperform Powers of 2

| Category | Avg Suppression | Notes |
|----------|----------------|-------|
| **13-multiples** | **0.392** | Aligns with 13×13 detection grid |
| Primes | 0.372 | High primes cause total kill |
| Powers of 2 | 0.189 | Architecture-aligned but weakest |

**13-multiples outperform powers of 2 by 2.1×** — YOLOv3's final grid is 13×13, so 13-multiple frequencies resonate with the detection grid structure. Powers of 2 (matching the downsample chain) are the weakest suppressors — the network is robust to its own architecture frequencies.

### k=208 — The Hallucination Weapon

k=208 (Nyquist, 13×16) on a **blank scene with no person** produces **11 person detections at confidence up to 0.9847**. The model hallucinates people that don't exist at near-certainty confidence. Nyquist frequency aligned with 13-multiple resonates with the 13×13 detection grid — each detection cell receives a consistent phase, creating a template the detection head reads as real objects.

### Cross-Model: Frequencies Don't Transfer

| Pattern | YOLOv3 | v8 | v11 | v26 |
|---------|--------|-----|-----|-----|
| k167_d | SUPPRESS | HALLUCINATE | HALLUCINATE | HALLUCINATE |
| k196_d | SUPPRESS | HALLUCINATE | HALLUCINATE | HALLUCINATE |
| k208_d | 12 dets | HALLUCINATE | HALLUCINATE | HALLUCINATE |

k=167/196/208 do NOT transfer as suppressors to newer YOLO models. They hallucinate instead. Different architectures (C2f, C3k2, different downsample chains) change the aliasing dynamics. **For cross-model attacks, use k proportional to input_size/2.**

### Shape Matters — Irregular > Geometric > Fractal

Tested 10 shapes × 27 textures across 4 YOLO models:

| Shape | Suppression Rate | Area |
|-------|-----------------|------|
| **deformed_r12_large** | **23.1%** | 16.3% |
| octagon_deformed | 3.7% | 10.8% |
| circle_r80 | 1.9% | 11.6% |
| triangle_r100 | 0.9% | 7.6% |
| sierpinski_d3 | 0% | 3.9% |
| sierpinski_d4 | 0% | 3.2% |

**Irregular shapes that don't match natural object contours cause more disruption than geometric or fractal patterns.** Sierpinski triangles are too regular and too small. The Shape Matters paper's deformable approach — jointly optimizing shape + texture via gradient descent — is the path forward.

### Best Textures for Patch-Constrained Suppression

| Texture | Suppression Rate |
|---------|-----------------|
| **stripes_h_32px** | **10.0%** |
| **stripes_v_13px** | **10.0%** |
| random_noise | 5.0% |
| k167_d, k208_d, k196_d | 2.5% each |

**Stripes outperform sinusoids for patch-constrained suppression.** Hard stripe edges (square waves) at 13px or 32px spacing create broadband frequency content that aliases more aggressively than smooth sinusoids when constrained to a small patch.

### Model Robustness Ranking

**YOLOv8** (most fragile) < **YOLO11** < **YOLOv3** < **YOLO26** (most robust)

YOLOv8/11/26 hallucinate massively from patches — nearly every shape+texture combo causes non-person detections (umbrellas, traffic lights, stop signs). The newer models are more easily fooled into seeing wrong objects.

---

## Phase 4: The Attack Pipeline

### The Stealthy Regime — Zero Collateral

Can a patch suppress the wearer without collateral damage to bystander detections? Tested 180 combinations (5 sizes × 9 amplitudes × 4 textures).

**Only 2 sweet spots found** — both with 13px vertical stripes at low amplitude (0.05–0.08) on medium (r80, ~6% area) patches. Zero bystander collateral, but only 1 of 4 wearer detections suppressed.

**The Trade-off Triangle:**
1. **Wearer suppression** — needs high amplitude + large patch
2. **Zero collateral** — needs low amplitude + small patch
3. **Embedding poisoning** — needs high amplitude (L2 > 1.0)

The sweet spot achieves #2 and #3 but only partially achieves #1. Full wearer suppression requires amp ≥ 0.15, causing 3–5 bystander casualties.

### Covert Channel — 1/196 as Information Carrier

The decimal expansion of 1/196 encodes a doubling sequence (5×2^k) with 42-digit repeating period. Embedded in a spatial carrier frequency, this signal survives through the network:

- **~2 bits reliable capacity** through bbox coordinates and confidence scores
- **Single-channel decoder**: payload amplitude correlates with individual channel activations at r=0.999
- **Period-42 structure visible at L105** (13×13 detection head) — autocorrelation = 0.35
- **Batch norm kills uniform values immediately** (layer 1), but spatial carriers preserve ~3% of the signal to L105
- **Backdoor regime at amp 0.005–0.020**: detection preserved at 88–100% with subtle coordinate/confidence shifts

### Cloud Poisoning Pipeline

The full attack pipeline: physical patch → camera → YOLOv3 → embeddings → cloud model training data.

- **k=167 sinusoid is the dual-purpose weapon**: suppresses person detection (7/15) AND maximally corrupts embeddings (L2=4.90, SNR=-16.8dB)
- **Poisoned embeddings are stealthy**: cosine similarity >0.9996 vs clean. L2-norm, PCA, and 8-bit quantization all fail to fully remove the poisoning signal
- **Embedding distance correlates with payload amplitude** (r > 0.93 at all detection heads) — precise control
- **L81 (52×52) is the most vulnerable head**: highest separability (1.21), highest SNR channels (+3.93dB), best amplitude correlation (r=0.993)

### The 1/196 Digit Sequence — Misaligned Carries

The digits of 1/196 encode powers of 2 with carry propagation. The **256-dim resonance hypothesis was disproven** — power-of-2 carry points create LESS disruption because they align with the network's power-of-2 structure and are filtered out. It's the **misaligned carries** (non-power-of-2: 1, 3, 6, 12, 25, 51...) that cause 50% more disruption.

**Open loops beat closed loops**: truncating the 42-digit period creates boundary discontinuities that add broadband energy.

### Patch Design Synthesis

The analyses converged on a multi-scale fractal patch design:

1. **k-spread**: compound wave numbers per Sierpinski branch level for broadband frequency targeting
2. **FFT phase-shift depth encoding**: per-harmonic phase shifts with 1/n falloff for multi-plane depth hallucination
3. **Pascal mod-n void geometry**: cycling corner voids for different collateral profiles
4. **1/196 digit modulation**: 42-cycle coprime persistence for downsampling armor
5. **Gradient optimization**: maximize L2 disruption at detection heads constrained to mask region

**Results after optimization**: 4→0 detections, combined confidence 0.9995→0.0033, L81 cosine disruption 5.25%.

---

## Repository Structure

### Analysis Scripts (17 analyses)

| File | Analysis |
|---|---|
| `l2_fft_laplacian_kfac.py` | Graph Laplacian — channel isolation maps, Fiedler vectors |
| `forward_person_delta.py` | Person signal extraction — layer rankings, anchor channels |
| `freq_analysis.py` | Deep frequency analysis — 1D/2D/Polynomial FFT, frequency cascade |
| `interference_analysis.py` | 4-way interference — raw + embedding FFT, cross-spectrum |
| `constructive_interference.py` | Constructive interference — HF suppression efficiency, frequency conversion |
| `destructive_interference.py` | Destructive interference — phase cancellation, diffuse HF energy |
| `triangular_patch_test.py` | Cross-model: 10 shapes × 27 textures × 4 models |
| `capture_embeddings_webcam.py` | Live webcam embedding capture |

### Patch Generation & Evaluation

| File | Purpose |
|---|---|
| `final_boss.py` | Sierpinski fractal patch + k-spread + FFT depth + gradient optimization |
| `test_patch_metrics.py` | 9-metric evaluation suite |
| `deformable_patch.py` | Deformable polygon mask variant (Shape Matters paper) |
| `train_v45_fft_signature.py` | FFT signature matching approach |
| `fractal_patch.py` | Shared utilities: forward pass, embeddings, FFT |
| `fractal_image_patch.py` | Self-similar image tiling into Sierpinski structure |

### Results

| Directory | Content |
|---|---|
| `results/COMPREHENSIVE_SUMMARY.json` | Master results across all analyses |
| `results/person_signal_summary.json` | Person-sensitive layer rankings + anchor channels |
| `results/figures/` | Key figures from all analyses |
| `results/fractal/` | Fractal patch images + per-layer metrics |
| `results/triangular/` | Cross-model test data (10×27×4) |
| `results/interference/` | 4-way interference analysis |
| `results/freq_deep/` | Per-layer frequency band ratios |

### Documentation

| File | Content |
|---|---|
| `MASTER_FINDINGS.md` | Full 1800-line research document — all 17 analyses with methodology, results, and interpretation |

---

## Key Design Decisions

- **Frequency-domain attack surface**: CNN activations have well-defined frequency responses. Targeting specific bands is more reliable than pixel-space perturbation.
- **HF over LF**: High-frequency injection is 2× more efficient — it bypasses the network's frequency conversion step.
- **Stripes over sinusoids for patches**: Hard edges create broadband aliasing that works better when constrained to small areas.
- **Irregular shapes over geometric**: Shapes that don't match natural object contours cause more disruption.
- **13-multiples over powers of 2**: The detection grid frequency (13) is more disruptive than the architecture frequency (powers of 2).
- **Construction before optimization**: Mathematical structure provides a strong prior. Pure optimization from random noise converges to weaker local minima.

---

## Setup

```bash
git clone https://github.com/Carson1391/spectral-adversarial-patches.git
cd spectral-adversarial-patches

pip install torch numpy pillow matplotlib opencv-python

# YOLOv3 model
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git

# Download weights (248MB — not in repo)
# https://pjreddie.com/media/files/yolov3.weights

# Test images (not in repo)
# Place withhuman.png and withouthuman.png in project root

# Run analyses
python forward_person_delta.py      # Person signal extraction
python freq_analysis.py             # Deep frequency analysis
python constructive_interference.py # HF suppression efficiency
python triangular_patch_test.py     # Cross-model texture testing

# Generate and evaluate patches
python final_boss.py                # Build + optimize fractal patch
python test_patch_metrics.py        # Full 9-metric evaluation
```

---

## License

Research code — academic exploration of adversarial vulnerabilities in object detection systems.
