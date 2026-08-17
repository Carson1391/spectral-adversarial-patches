# Spectral Adversarial Patches — Frequency-Domain Attacks on YOLO Object Detectors

**Research portfolio — 17 analyses spanning graph theory, spectral decomposition, interference physics, covert channels, and physical patch design.**

---

## Four Novel Findings

These findings challenge standard assumptions about how neural networks process information and provide hard, actionable data for new attack vectors. Each is supported by empirical measurements across all 75 conv layers of YOLOv3.

### 1. The Frequency Cascade — Neural Frequency Amplification

Most researchers view object detectors as spatial pattern matchers. The data proves the network acts as a **frequency amplifier and converter**.

A human in a raw image is overwhelmingly low-frequency (99.99% LF). Yet the network actively converts this tiny perturbation into a massive broadband signal deep inside the model:

| Depth | Layers | LF | MF | HF | Interpretation |
|-------|--------|-----|-----|-----|----------------|
| Input | raw pixels | 99.99% | 0.004% | 0.001% | Person is an LF blob |
| Early | 0–5 | 76–96% | 3–10% | 2–15% | Edges begin encoding |
| Mid | 12–37 | 54–65% | 16–21% | 19–25% | Limbs, boundaries |
| Deep | 54–60 | 51–58% | **26–28%** | 16–21% | MF peaks at 26×26 |
| Deepest | **62–75** | **26–34%** | **41–43%** | **25–34%** | **MF dominant — broadband** |
| Heads | 81–105 | 53–94% | 5–29% | 1–5% | Decision collapses to LF |

**The paper hook**: Tracking this exact frequency cascade (LF input → MF peaks in mid-layers → broadband in deep layers → sudden collapse back to LF at detection heads) offers a mathematically rigorous way to explain how deep learning models encode spatial concepts. The network doesn't just detect patterns — it *manufactures* frequency content.

### 2. The Graph Laplacian "Stealth Corridor" — Topological Vulnerability

Graph Laplacian analysis of channel connectivity reveals a massive structural vulnerability in YOLOv3.

Mid-backbone layers operate as extreme, isolated specialists:

| Layer | Channels | Edges | Isolated | Spectral Gap |
|-------|----------|-------|----------|-------------|
| 0 | 32 | 127 | 5 (16%) | 3.4e-16 |
| 37 | 512 | 9 | **495 (97%)** | 2.0e-8 |
| 54 | 512 | 193 | **487 (95%)** | 1.4e-15 |
| 62 | 1024 | 282 | **891 (87%)** | 1.5e-8 |
| 75 | 512 | 20 | **492 (96%)** | 4.7e-16 |
| 81 | 255 | 27,885 | 13 (5%) | 9.9e-7 |
| 93 | 255 | 28,625 | 12 (5%) | 1.1e-6 |
| 105 | 255 | 28,769 | 10 (4%) | 3.2e-7 |

**The paper hook**: This maps a perfect "stealth corridor." An adversarial signal can sneak through the isolated mid-layers — where disrupting one channel doesn't cascade and trigger generalized network failure — and directly bomb the dense recombination layers at the end to force misclassification. The Fiedler vector concentrates on layers 61–65 and 74–75, identifying the exact structural bottlenecks.

### 3. Direct Diagonal HF Injection is ~2× More Efficient

Standard adversarial patches rely on low-frequency (LF), smooth gradients. The constructive interference data proves this is the wrong approach.

Because the network acts as a frequency converter (Finding #1), injecting LF forces the network to do the conversion work internally. Injecting HF directly bypasses this step:

| Band | Best Frequency | Suppression Score | Efficiency (score/amp) |
|------|---------------|-------------------|----------------------|
| LF | k=5 | 0.098 | 0.49 |
| MF | k=50 | 0.288 | 1.44 |
| **HF** | **k=200 diagonal** | **0.542** | **2.71** |

**HF is 2× more effective per unit amplitude than LF.** Diagonal HF (kx=ky=200) is the best suppressor. Three additional properties make this actionable:

- **Phase is irrelevant for HF** — scores vary by <0.7% across all phase angles (0°–180°). HF oscillates so rapidly that phase averages out across the spatial extent.
- **Monotonic scaling without saturation** — suppression increases linearly with amplitude up to at least 0.50. Every bit of additional energy contributes.
- **Repeated injection does NOT compound** — batch norm + LeakyReLU prevent runaway accumulation. Single-shot high amplitude beats repeated low amplitude.

**The paper hook**: This radically simplifies adversarial patch design. No phase tuning needed. No iterative application. Just maximum diagonal HF amplitude.

### 4. The "Whisper in a Loud Song" — Batch Norm Bypass via Spatial Carriers

Batch normalization is known to normalize away uniform offsets. The data provides empirical proof of how to bypass it using spatial carrier frequencies — a highly actionable exploit.

A uniform payload offset (like 1/196) is perfectly erased by Layer 1 because it has zero variance:

| Pattern | Closure | Survival at L105 |
|---------|---------|-----------------|
| inv_196_offset (uniform) | **Closed at layer 1** | ~0% |
| k196d_amp_inv196 (spatial) | Never fully closed | decays from 29% at L1 |
| anticlose_inv196_k200d | Never fully closed | **~2.9% persists to L105** |
| anticlose_inv196_allprimes | Never fully closed | ~1.2% persists to L105 |

By modulating the payload into a spatial carrier (k=200 diagonal or stacked primes), you create enough structural variance that batch norm cannot filter it. The signal survives all the way to the densely connected Layer 105 detection head.

**The paper hook**: This is concrete proof of a mechanism for deep-layer data poisoning. A 1/196 payload modulated onto a k=200 carrier survives 75 conv layers of normalization and reaches the final detection head at measurable amplitude. The covert channel carries ~2 bits reliably through bbox coordinates and confidence scores (single-channel decoder r=0.999).

---

## From Findings to Attack: What This Changes

Existing adversarial patches are created via black-box gradient optimization — algorithms change pixels repeatedly until the AI fails. The resulting patches happen to look like high-frequency noise because the algorithm stumbled upon it through trial and error.

This research moves adversarial design from **blind trial-and-error to deterministic, mathematical targeting**:

| Old Approach | This Research |
|---|---|
| Optimize pixels, observe failure | Map the internal frequency cascade first |
| Generic noise patterns | Specific frequencies targeted to specific layers |
| LF smooth gradients (standard) | Diagonal HF — 2× more efficient, proven |
| Ignore network topology | Exploit the stealth corridor (isolated mid-layers → dense heads) |
| Hope the signal survives | Modulate onto spatial carriers to bypass batch norm |

**Specific targeting data:**

| Priority | Layer | Resolution | Required Content | Key Channels |
|----------|-------|-----------|-----------------|--------------|
| 1st | 54 | 26×26 | MF dominant (26.3%) | 479, 31, 51, 184, 422 |
| 2nd | 63 | 13×13 | Broadband (MF=43%, HF=32%) | 422, 147, 47, 8, 406 |
| 3rd | 75 | 13×13 | MF+HF (41%+25%) | **170**, 17, 322, 84, 292 |
| 4th | 62 | 13×13 | Broadband (MF=41%, HF=34%) | 782, 380, 807, 346, 305 |
| Detection | 93, 105 | 26×26, 52×52 | LF (smooth) | 170, 171 |

Channel 170 is the cross-scale person anchor — it appears at all three detection heads (75, 93, 105). Disrupting it directly attacks the detection decision.

---

## Key Empirical Results

### Prime Frequency Suppression

High primes near Nyquist (k=157–199 diagonal) cause **total detection kill** on YOLOv3:

| Prime k | Suppression | Detections |
|---------|------------|------------|
| **167** | **0.476** | **0** |
| 199 | 0.466 | 0 |
| 197 | 0.466 | 0 |
| 193 | 0.465 | 0 |

These primes near Nyquist (208) alias at every downsample step, scattering energy across all feature scales.

### 13-Multiples Beat Powers of 2

| Category | Avg Suppression | Why |
|----------|----------------|-----|
| **13-multiples** | **0.392** | Aligns with 13×13 detection grid |
| Primes | 0.372 | High primes cause total kill |
| Powers of 2 | 0.189 | Architecture-aligned — filtered out |

The network is robust to its own architecture frequencies (powers of 2 match the downsample chain) but vulnerable to frequencies that align with its detection grid (13) or create aliasing (high primes).

### k=208 — The Hallucination Weapon

k=208 (Nyquist, 13×16) on a blank scene with no person produces **11 person detections at confidence up to 0.9847**. The model hallucinates people that don't exist at near-certainty confidence. Nyquist aligned with 13-multiple resonates with the 13×13 detection grid.

### Shape Matters — Irregular > Geometric > Fractal

10 shapes × 27 textures × 4 YOLO models:

| Shape | Suppression Rate | Area |
|-------|-----------------|------|
| **deformed_r12_large** | **23.1%** | 16.3% |
| octagon_deformed | 3.7% | 10.8% |
| circle_r80 | 1.9% | 11.6% |
| sierpinski_d4 | 0% | 3.2% |

Irregular shapes that don't match natural object contours cause more disruption than geometric or fractal patterns. Stripes (13px/32px) outperform sinusoids for patch-constrained suppression — hard edges create broadband aliasing.

### Cross-Model: Frequencies Don't Transfer

k=167/196/208 suppress on YOLOv3 but **hallucinate** on YOLOv8/11/26. Different architectures change aliasing dynamics. For cross-model attacks, use k proportional to `input_size/2`.

### The Stealthy Regime

Only 2 of 180 combinations achieve zero bystander collateral with any wearer suppression — both with 13px vertical stripes at amp 0.05–0.08 on medium (r80, ~6% area) patches. Full wearer suppression requires amp ≥ 0.15, causing 3–5 bystander casualties.

### Cloud Poisoning Pipeline

k=167 sinusoid is the dual-purpose weapon: suppresses person detection (7/15) AND maximally corrupts embeddings (L2=4.90, SNR=-16.8dB). Poisoned embeddings are stealthy — cosine similarity >0.9996 vs clean. L2-norm, PCA, and 8-bit quantization all fail to remove the poisoning signal.

---

## Repository Structure

### Analysis Scripts

| File | Analysis |
|---|---|
| `l2_fft_laplacian_kfac.py` | Graph Laplacian — channel isolation maps, Fiedler vectors (Finding #2) |
| `forward_person_delta.py` | Person signal extraction — layer rankings, anchor channels |
| `freq_analysis.py` | Deep frequency analysis — 1D/2D/Polynomial FFT, frequency cascade (Finding #1) |
| `interference_analysis.py` | 4-way interference — raw + embedding FFT, cross-spectrum |
| `constructive_interference.py` | Constructive interference — HF efficiency, frequency conversion (Finding #3) |
| `destructive_interference.py` | Destructive interference — phase cancellation, diffuse HF energy |
| `triangular_patch_test.py` | Cross-model: 10 shapes × 27 textures × 4 models |
| `capture_embeddings_webcam.py` | Live webcam embedding capture |

### Patch Generation

| File | Purpose |
|---|---|
| `final_boss.py` | Sierpinski fractal patch + k-spread + FFT depth + gradient optimization |
| `test_patch_metrics.py` | 9-metric evaluation suite |
| `deformable_patch.py` | Deformable polygon mask variant (Shape Matters paper) |
| `train_v45_fft_signature.py` | FFT signature matching approach |
| `fractal_patch.py` | Shared utilities: forward pass, embeddings, FFT |
| `fractal_image_patch.py` | Self-similar image tiling into Sierpinski structure |

### Results & Documentation

| Path | Content |
|---|---|
| `MASTER_FINDINGS.md` | Full 1800-line research document — all 17 analyses |
| `results/` | JSONs, CSVs, and figures from all analyses |

---

## Limitations & Next Steps

**Cross-model generalization**: The entire analysis is on YOLOv3 (Darknet-53, 416×416, COCO). To prove these vulnerabilities are universal flaws in object detection rather than YOLOv3-specific quirks, the same analyses must be run on YOLOv8+ and different architectures (Vision Transformers, ResNet backbones). Preliminary cross-model testing showed frequencies don't transfer — k=167 suppresses on v3 but hallucinates on v8/11/26.

**The log10(2) hypothesis**: The number theory regarding non-integer periods and log10(2) as a universal network constant is not yet in the data. It needs mathematical formalization and empirical testing.

**Joint shape+texture optimization**: The Shape Matters paper's deformable approach — gradient-based joint optimization of ray lengths + pixel values — was identified as the path forward but not yet implemented.

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
```

---

## License

Research code — academic exploration of adversarial vulnerabilities in object detection systems.
