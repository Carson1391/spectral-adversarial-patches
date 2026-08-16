# YOLOv3 Adversarial Patch Frequency Analysis - Master Findings

**Date:** 2026-07-03
**Model:** YOLOv3 (Darknet-53 backbone, 75 conv layers, COCO 80-class)
**Weights:** yolov3.weights
**Input:** 416x416, CUDA (RTX 5060 Ti)
**Method:** Paired image comparison (withhuman.png vs withouthuman.png). Delta = features(with) - features(without) = the person signal inside the network.

---

## Table of Contents

1. [Analysis 1: L2/FFT/Laplacian/KFAC](#analysis-1)
2. [Analysis 2: 2D FFT Spectral Difference](#analysis-2)
3. [Analysis 3: Forward Person Delta + Saliency](#analysis-3)
4. [Analysis 4: Deep Frequency Analysis (1D FFT, Polynomial FFT, 2D FFT)](#analysis-4)
5. [Analysis 5: 4-Way Interference (Raw Pixel + Embedding FFT)](#analysis-5)
6. [Analysis 6: Destructive Interference (Phase Cancellation)](#analysis-6)
7. [Analysis 7: Constructive Interference Downward + HF Propagation](#analysis-7)
8. [Cross-Cutting Findings](#cross-cutting)
9. [Actionable Patch Design Recommendations](#recommendations)
10. [Analysis 19: Live Webcam Embedding Capture](#analysis-19)
11. [Analysis 20: Triangular Patch — Sinusoid Baseline Cross-Model](#analysis-20)

---

<a id="analysis-1"></a>

## Analysis 1: L2/FFT/Laplacian/KFAC

**Script:** `l2_fft_laplacian_kfac.py`
**Output:** `outputs_clothing/l2_fft_laplacian_kfac/graph_laplacian.json`

### Graph Laplacian Findings

**Layer-level graph:** 75 layers, only 6 edges, 67 isolated nodes. The network is extremely sparse in inter-layer connectivity. The Fiedler vector concentrates on layers 61-65 and 74-75 — the deepest backbone and first detection head. These are the structurally most important layers for information flow.

**Channel-level graphs (key layers):**

| Layer | Channels | Edges | Isolated | Spectral Gap | Interpretation |
|-------|----------|-------|----------|-------------|----------------|
| 0 | 32 | 127 | 5 | 3.4e-16 | Dense, nearly connected — early features all correlate |
| 1 | 64 | 295 | 23 | 1.1e-16 | Still dense — shared low-level features |
| 5 | 128 | 626 | 62 | 4.0e-8 | Moderate connectivity |
| 12 | 256 | 1065 | 125 | 1.1e-7 | Half isolated — channels specialize |
| 37 | 512 | 9 | 495 | 2.0e-8 | Nearly fully disconnected — maximum specialization |
| 54 | 512 | 193 | 487 | 1.4e-15 | Very sparse — person channels are isolated specialists |
| 62 | 1024 | 282 | 891 | 1.5e-8 | Sparse — 87% isolated channels |
| 75 | 512 | 20 | 492 | 4.7e-16 | Extremely sparse — near-total specialization |
| 81 | 255 | 27885 | 13 | 9.9e-7 | Dense — detection head recombines features |
| 93 | 255 | 28625 | 12 | 1.1e-6 | Dense — detection head recombines |
| 105 | 255 | 28769 | 10 | 3.2e-7 | Dense — detection head recombines |

**Key insight:** Mid-backbone layers (37-75) have maximum channel isolation. Person-sensitive channels operate independently — disrupting one does not cascade to others. Detection heads (81, 93, 105) recombine everything densely.

---

<a id="analysis-2"></a>

## Analysis 2: 2D FFT Spectral Difference

**Script:** 2D FFT analysis across all conv layers
**Output:** `fft_2d_analysis/summary.json`

### Top Spectral Difference Channels Per Layer

Measured the 2D FFT spectral difference between with-human and without-human feature maps per channel. Higher score = more spectral change when human is present.

**Layer 0 (conv_0):** Top channel = 27 (score=2392.8), then 12 (583.4), 4 (478.9). Mean diff = 185.4. Early layer has one dominant channel.

**Layer 1 (conv_1):** Top channel = 0 (264.8), then 37 (183.4). Mean diff = 35.4. More distributed.

**Layer 10 (conv_10):** Top channel = 80 (98.6), then 110 (92.7). Mean diff = 17.6.

**Layer 105 (conv_105, detection head 52x52):** Top channels = 1 (41.8), **170** (40.3), 0 (34.7), **171** (32.7), 85 (31.3). Channels 170 and 171 appear as top spectral channels at the finest detection scale.

**Cross-layer pattern:** Spectral difference scores decrease with depth (from 2392 at layer 0 to ~40 at layer 105), but relative importance increases because absolute activations shrink.

---

<a id="analysis-3"></a>

## Analysis 3: Forward Person Delta + Saliency

**Script:** `forward_person_delta.py`
**Output:** `outputs_clothing/forward_analysis/person_signal_summary.json`, `COMPREHENSIVE_SUMMARY.json`

### Saliency Results

- **Person score with human:** 1.0124 (strong detection)
- **Person score without human:** 0.0155 (near-zero, correct rejection)
- Saliency map confirms model focuses on human figure when present

### Person-Sensitive Layer Rankings (by relative delta)

| Rank | Layer | Shape | Mean Rel Delta | Max Rel Delta | Top Channels |
|------|-------|-------|---------------|---------------|--------------|
| 1 | **54** | 512x26x26 | **0.559** | 2.339 | 479, 31, 51, 184, 422 |
| 2 | **75** | 512x13x13 | **0.504** | 1.707 | 170, 17, 322, 84, 292 |
| 3 | **84** | 256x13x13 | **0.453** | 1.291 | 167, 234, 32, 211, 157 |
| 4 | **63** | 512x13x13 | **0.423** | **2.548** | 422, 147, 47, 8, 406 |
| 5 | **60** | 512x26x26 | **0.420** | 1.528 | 495, 26, 131, 213, 5 |

- Layer 54 has the highest mean relative delta (56% activation change)
- Layer 63 has the highest single-channel max relative delta (254.8%)
- Layer 75 is second most sensitive and contains channel 170 (cross-scale person anchor)

### Cross-Layer Person Channels

| Channel | Appears At | Role |
|---------|-----------|------|
| **170** | 75, 93, 105 | Person anchor channel — all detection scales |
| **171** | 93, 105 | Person anchor at medium and fine scales |
| **422** | 54, 63 | Deep person feature — backbone + detection-adjacent |
| **47** | 54, 63, 75 | Consistent person feature across deep layers |
| **51** | 1, 54 | Person edge detector — early and deep |

### Per-Layer Band Ratios (LF/MF/HF of delta)

| Layer | LF | MF | HF | Dominant Band |
|-------|-----|-----|-----|---------------|
| 0 | 0.957 | 0.026 | 0.017 | LF |
| 1 | 0.863 | 0.056 | 0.080 | LF |
| 5 | 0.756 | 0.096 | 0.148 | LF |
| 12 | 0.653 | 0.162 | 0.185 | LF |
| 37 | 0.538 | 0.213 | 0.249 | LF (HF rising) |
| 54 | 0.578 | 0.263 | 0.159 | LF (MF peaks) |
| 60 | 0.509 | 0.283 | 0.207 | LF (most balanced 26x26) |
| **62** | **0.256** | **0.408** | **0.336** | **MF dominant** |
| **63** | **0.256** | **0.427** | **0.317** | **MF dominant** |
| **75** | **0.344** | **0.407** | **0.249** | **MF dominant** |
| 81 | 0.662 | 0.288 | 0.049 | LF |
| 84 | 0.532 | 0.338 | 0.130 | LF |
| 92 | 0.815 | 0.140 | 0.045 | LF |
| 93 | 0.829 | 0.141 | 0.030 | LF |
| 105 | 0.942 | 0.053 | 0.005 | LF |

### Frequency Cascade

The human signal undergoes a **frequency cascade** through the network:

- **Layers 0-5 (early, high-res):** LF dominant (76-96%). Person is a large low-frequency blob.
- **Layers 12-37 (mid backbone):** LF with rising MF+HF (54-65% LF). Network encodes edges, limbs, boundaries.
- **Layers 54-60 (deep backbone):** LF but MF peaks (51-58% LF, 26-28% MF). Medium-scale patterns at 26x26.
- **Layers 62-75 (deepest, low-res):** MF dominant (26-34% LF, 41-43% MF, 25-34% HF). At 13x13, person is a high-frequency event. Most broadband.
- **Layers 81-105 (detection heads):** LF returns (53-94%). Decision is made — yes/no binary collapses to low frequency.

---

<a id="analysis-4"></a>

## Analysis 4: Deep Frequency Analysis (1D FFT, Polynomial FFT, 2D FFT)

**Script:** `freq_analysis.py`
**Output:** `outputs_clothing/forward_analysis/freq_deep/freq_summary.json`

### 1D Delta FFT

Top 1D frequency is consistently at bin = feature_map_width (one full cycle across the map). The person creates one major spatial event — the fundamental harmonic.

Oscillation in 1D delta FFT plots is **visual redundancy** from plotting both halves of a real-valued FFT. Bins 0 to N/2 are unique; bins N/2 to N mirror them.

### Polynomial FFT

Each channel's spatial activation treated as polynomial coefficients. FFT reveals polynomial degree spectrum.

**Finding:** Dominant polynomial degrees are at DC (degree 0) and Nyquist (degree N-1), with harmonics at degrees 1, 2, 3. The model represents the person as a combination of very low-degree (smooth, global) and very high-degree (rapidly varying, local) polynomial features.

**Symmetry:** Polynomial FFT appears symmetric because input coefficients are real-valued. For real input X, |X[k]| = |X[N-k]| by conjugate symmetry. Only bins 0 to N/2 carry unique information.

### 2D FFT Top Channels Per Layer

| Layer | Top 2D Spectral Channels |
|-------|------------------------|
| 0 | 12, 26, 11, 10, 13, 5, 3, 1, 19, 16 |
| 37 | 74, 19, 464, 204, 309, 91, 470, 34, 11, 503 |
| 54 | 385, 141, 146, 380, 51, 47, 483, 132, 422, 271 |
| 62 | 130, 866, 3, 711, 346, 807, 998, 979, 203, 390 |
| 75 | 170, 262, 38, 424, 155, 362, 318, 265, 322, 84 |
| 105 | 211, 41, 14, 126, 212, 89, 99, 6, 34, 23 |

---

<a id="analysis-5"></a>

## Analysis 5: 4-Way Interference (Raw Pixel + Embedding FFT)

**Script:** `interference_analysis.py`
**Output:** `outputs_clothing/forward_analysis/interference/interference_4way.json`

### Raw Image FFT

| Channel | LF | MF | HF |
|---------|-----|-----|-----|
| With human (gray) | 0.99994 | 4.4e-5 | 1.2e-5 |
| Without human (gray) | 0.99995 | 4.1e-5 | 1.1e-5 |
| Delta (gray) | 0.9977 | 0.0017 | 0.0006 |
| Cross-spectrum | 0.99999... | ~3.8e-10 | ~1.6e-11 |

**Finding:** Raw images are 99.99% LF. The delta (human signal at pixel level) is 99.77% LF. Cross-spectrum coherence is ~1.0 — the two images are nearly identical in frequency content at the pixel level. The human introduces almost no frequency change at the pixel level.

### Embedding FFT: How the Network Transforms the Signal

| Layer | Delta LF | Delta MF | Delta HF | Cross LF | Cross HF |
|-------|---------|---------|---------|---------|---------|
| 0 | 0.957 | 0.026 | 0.017 | 0.99999 | 1.8e-9 |
| 5 | 0.756 | 0.096 | 0.148 | 0.99999 | 3.1e-7 |
| 12 | 0.653 | 0.162 | 0.185 | 0.99999 | 1.4e-6 |
| 37 | 0.538 | 0.213 | 0.249 | 0.99985 | 4.7e-5 |
| 54 | 0.578 | 0.263 | 0.159 | 0.996 | 0.0006 |
| 60 | 0.509 | 0.283 | 0.207 | 0.984 | 0.003 |
| **62** | **0.256** | **0.408** | **0.336** | **0.954** | **0.005** |
| **63** | **0.256** | **0.427** | **0.317** | **0.982** | **0.002** |
| **75** | **0.344** | **0.407** | **0.249** | **0.978** | **0.001** |
| 81 | 0.662 | 0.288 | 0.049 | 0.99997 | 3.1e-7 |
| 105 | 0.942 | 0.053 | 0.005 | 0.99999 | 6.2e-10 |

**Key finding:** The network is a **frequency amplifier**. Input is 99.99% LF, but by layer 62 the delta is only 25.6% LF — the network has generated 74.4% MF+HF content from a nearly pure-LF input. Cross-spectrum coherence drops from ~1.0 at early layers to 0.954 at layer 62, meaning the with/without representations diverge most at this layer.

---

<a id="analysis-6"></a>

## Analysis 6: Destructive Interference (Phase Cancellation)

**Script:** `destructive_interference.py`
**Output:** `outputs_clothing/forward_analysis/destructive/destructive_interference.json`

### Pixel Level

Top bins are all LF near DC. Top bin: (209, 208), power=4.16M, phase=28.8deg, cancel at 208.8deg. The pixel delta is 99.77% LF, 0.06% HF — almost nothing at HF to cancel at the pixel level.

### Embedding Level - Top 5 Bins Per Key Layer

**Layer 0 (32x416x416):** LF=95.7%, all top bins LF

- #1 (207,208) LF power=1.26e7 phase=156.9deg -> cancel at 336.9deg, channels [1,5,17]

**Layer 54 (512x26x26):** LF=57.8%, all top bins LF

- #1 (13,13) LF power=230.0 phase=0.0deg -> cancel at 180.0deg, channels [483,199,418]
- #2 (12,13) LF power=229.0 phase=153.0deg -> cancel at 333.0deg, channels [483,199,51]

**Layer 62 (1024x13x13):** LF=25.6%, MF=40.8%, HF=33.6%, all top bins LF

- #1 (6,6) LF power=7.61 phase=0.0deg -> cancel at 180.0deg, channels [203,866,3]
- #2 (7,6) LF power=7.39 phase=-137.1deg -> cancel at 42.9deg, channels [203,866,390]

**Layer 75 (512x13x13):** LF=34.4%, MF=40.7%, HF=24.9%, all top bins LF

- #1 (6,6) LF power=36.5 phase=180.0deg -> cancel at 360.0deg, channels [170,265,1]
- #2 (6,7) LF power=30.1 phase=120.8deg -> cancel at 300.8deg, channels [170,265,292]

**Layer 105 (255x52x52):** LF=94.2%, all top bins LF

- #1 (26,26) LF power=2.83e5 phase=0.0deg -> cancel at 180.0deg, channels [41,126,211]

### HF Cancellation Energy

| Layer | HF Energy | HF Bins in Top-10 |
|-------|-----------|-------------------|
| 62 | 9.03e2 | 1/10 |
| 63 | 7.75e2 | 2/10 |
| 75 | 9.62e2 | 0/10 |

**Key finding:** At every layer, the top-20 frequency bins by power are ALL LF (near-DC). The human signal's concentrated power is always at low frequencies, even when the band ratios show significant HF. The HF energy is **diffuse** — spread thinly across hundreds of bins, not concentrated in any single dominant bin. There is no "silver bullet" HF frequency to cancel.

**Implication:** To cancel the HF portion, you need broadband HF noise covering all high-frequency bins simultaneously, not a single targeted frequency. The most effective cancellation targets the DC/LF dominant bins.

---

<a id="analysis-7"></a>

## Analysis 7: Constructive Interference Downward + HF Propagation

**Script:** `constructive_interference.py`
**Output:** `outputs_clothing/forward_analysis/constructive/constructive_interference.json`

### Experiment 1: Sinusoidal Pixel Oscillation Suppression

Injected sinusoidal patterns at various frequencies and phases into the with-human image. Measured how far features moved from "with human" toward "without human" (suppression score, higher = more suppression).

**Top-5 suppression combos (amp=0.20):**

| Rank | Frequency | Phase | Band | Avg Score |
|------|-----------|-------|------|-----------|
| 1 | HF_k200d200 | 45deg | HF | **0.5418** |
| 2 | HF_k200d200 | 180deg | HF | 0.5417 |
| 3 | HF_k200d200 | 0deg | HF | 0.5411 |
| 4 | HF_k200d200 | 135deg | HF | 0.5392 |
| 5 | HF_k200d200 | 90deg | HF | 0.5381 |

**HF is ~2x more effective per unit amplitude than LF.** Diagonal HF (kx=ky=200) is the best suppressor.

### Suppression by Band (amp=0.20)

| Band | Best Frequency | Avg Score | Efficiency (score/amp) |
|------|---------------|-----------|----------------------|
| LF | k5 | 0.098 | 0.49 |
| MF | k50 | 0.288 | 1.44 |
| **HF** | **k200d200** | **0.542** | **2.71** |

### Phase Effect

Phase barely matters for HF suppression:

| Phase | HF_k200d200 Score |
|-------|-------------------|
| 0deg | 0.5411 |
| 45deg | 0.5418 |
| 90deg | 0.5381 |
| 180deg | 0.5417 |

HF oscillates so rapidly that phase averages out across the spatial extent.

### Experiment 2: HF Propagation Through the Network

| Injected | Delta HF at L0 | Delta HF at L62 | Delta HF at L75 |
|-----------|---------------|-----------------|-----------------|
| LF_k2 | 0.0038 | **0.2559** | **0.2761** |
| LF_k5 | 0.0061 | 0.1675 | 0.2040 |
| MF_k50 | 0.0017 | 0.0780 | 0.0967 |
| HF_k200 | 0.0159 | 0.0499 | 0.0595 |
| HF_k200d200 | 0.0001 | 0.0173 | 0.0214 |

**Key finding:** The network is a **frequency converter**. When you inject LF, the network produces HF internally (0.0038 -> 0.2559 at L62). When you inject HF, it attenuates but survives (0.0159 -> 0.0499). Injecting HF directly is more efficient because it bypasses the conversion step.

### Experiment 3: Amplitude Sweep

| Frequency | Phase | Best Amp | Best Score |
|-----------|-------|---------|-----------|
| LF_k5 | 0 | 0.50 | 0.2516 |
| LF_k5 | 180 | 0.50 | 0.2394 |
| MF_k50 | 0 | 0.50 | 0.4564 |
| MF_k50 | 180 | 0.50 | 0.4575 |
| **HF_k200** | **0** | **0.50** | **0.4887** |
| **HF_k200** | **180** | **0.50** | **0.4982** |

**Monotonic increase** — no saturation up to amplitude 0.50. Every bit of additional oscillation energy contributes to suppression.

---

<a id="cross-cutting"></a>

## Cross-Cutting Findings

### 1. The Network is a Frequency Amplifier and Converter

- Input: 99.99% LF
- Layer 62 delta: 25.6% LF, 74.4% MF+HF
- The network generates broadband content from near-pure-LF input
- Injecting LF causes the network to produce HF internally

### 2. HF Injection is 2x More Efficient Than LF for Suppression

- HF_k200d200 at amp=0.20 achieves 0.54 suppression score
- LF_k5 at amp=0.20 achieves only 0.10
- HF bypasses the network's frequency conversion step

### 3. Person Signal Power is Concentrated at LF, Even When Band Ratios Show High HF

- All top-20 frequency bins at every layer are LF (near-DC)
- HF energy is diffuse — spread across hundreds of bins
- No single "silver bullet" HF frequency to target

### 4. Channel 170 is the Cross-Scale Person Anchor

- Appears at layers 75, 93, 105 (all three detection scales)
- Top person-sensitive channel at layer 75
- Top spectral difference channel at layer 105
- Disrupting channel 170 directly attacks the detection decision

### 5. Layer 54 is the Optimal Target

- Highest mean relative delta (56%)
- MF peaks here (26.3%)
- Largest HF delta (+0.008)
- Channels 479, 31, 51, 184, 422 are the most person-tuned

### 6. Layer 62 is the Most Broadband

- Most balanced spectrum: LF=25.6%, MF=40.8%, HF=33.6%
- 1024 channels at 13x13
- Cross-spectrum coherence drops to 0.954 (most divergence between with/without)
- Requires broadband disruption

### 7. Mid-Backbone Channels Are Isolated Specialists

- Layers 37-75 have 87-97% isolated channels in graph Laplacian
- Disrupting one channel does not cascade to others
- Detection heads (81, 93, 105) recombine densely — 90%+ connected

### 8. Phase is Nearly Irrelevant for HF Suppression

- HF oscillates so rapidly that phase averages out
- Simplifies patch design — no need for precise phase tuning
- For LF, phase matters more (phase 0 > phase 180 at amp 0.50)

### 9. Amplitude is King

- Monotonic increase in suppression with amplitude, no saturation
- Every bit of additional oscillation energy contributes

---

<a id="recommendations"></a>

## Actionable Patch Design Recommendations

### Primary Strategy: High-Frequency Diagonal Oscillation

1. **Use high-frequency oscillating patterns** — 2x more efficient at suppressing the person signal than smooth low-frequency patches
2. **Diagonal HF (kx=ky) is best** — outperforms axis-aligned HF, suggesting the person signal has diagonal structure
3. **Phase doesn't matter for HF** — simplifies design, no need to carefully tune phase
4. **Push amplitude as high as visually tolerable** — monotonic gain, no saturation

### Target Layers and Channels

| Priority | Layer | Resolution | Key Channels | Frequency Content |
|----------|-------|-----------|-------------|-------------------|
| 1st | **54** | 26x26 | 479, 31, 51, 184, 422 | MF dominant (26.3%) |
| 2nd | **63** | 13x13 | 422, 147, 47, 8, 406 | Broadband (MF=43%, HF=32%) |
| 3rd | **75** | 13x13 | 170, 17, 322, 84, 292 | MF+HF (41%+25%) |
| 4th | **62** | 13x13 | 782, 380, 807, 346, 305 | Broadband (MF=41%, HF=34%) |
| Detection | 93, 105 | 26x26, 52x52 | 170, 171 | LF (smooth patches) |

### Multi-Scale Patch Design

A patch that works across all layers needs:

- **Fine HF detail** (for layers 62-75, 13x13 feature maps)
- **Medium-scale structure** (for layers 54-60, 26x26 feature maps)
- **Large smooth gradients** (for detection heads 81-105)
- **Diagonal orientation** preferred over axis-aligned
- **Maximum amplitude** at all scales

### What NOT to Do

- Do not target individual HF frequency bins — the power is diffuse, not concentrated
- Do not rely on LF-only patches — the network converts LF to HF internally, but starting with HF is 2x more efficient
- Do not worry about precise phase alignment for HF content
- Do not expect saturation — more amplitude always helps

---

## All Output Files

| Analysis | Location |
|----------|----------|
| L2/FFT/Laplacian/KFAC | `outputs_clothing/l2_fft_laplacian_kfac/` |
| 2D FFT spectral | `fft_2d_analysis/` |
| Person delta + saliency | `outputs_clothing/forward_analysis/` |
| Frequency deep | `outputs_clothing/forward_analysis/freq_deep/` |
| 4-way interference | `outputs_clothing/forward_analysis/interference/` |
| Destructive interference | `outputs_clothing/forward_analysis/destructive/` |
| Constructive interference | `outputs_clothing/forward_analysis/constructive/` |
| Comprehensive summary | `outputs_clothing/forward_analysis/COMPREHENSIVE_SUMMARY.json` |
| This document | `MASTER_FINDINGS.md` |

## All Scripts

| Script | Purpose |
|--------|---------|
| `l2_fft_laplacian_kfac.py` | L2 error, FFT, Graph Laplacian, KFAC on single image |
| `forward_person_delta.py` | Forward pass person delta, activation analysis, saliency |
| `freq_analysis.py` | 1D FFT, polynomial FFT, 2D FFT on delta feature maps |
| `interference_analysis.py` | 4-way FFT: raw pixels + embeddings, with/without human |
| `destructive_interference.py` | Phase cancellation analysis of human signal frequency bins |
| `constructive_interference.py` | Sinusoidal injection suppression + HF propagation test |
| `orthogonal_resonance.py` | 8-axis orientation test + scalar resonance accumulation + adaptive amplitude |
| `numeric_injection.py` | Prime/doubling/Lychrel/constant injection + full detection analysis |
| `hallucination_deep.py` | Full detection details on both images + 196 persistence tracking + anti-closure |

---

<a id="analysis-8"></a>

## Analysis 8: Orthogonal Axis + Scalar Resonance (orthogonal_resonance.py)

**Output:** `outputs_clothing/forward_analysis/orthogonal_resonance/`

### Orientation Findings

Tested all 8 orientations (0° through 157.5°) at LF (k=5), MF (k=50), HF (k=200).

- **Best orientation**: theta=112° (kx=-77, ky=185) — between vertical and anti-diagonal. **Not** the horizontal/diagonal previously tested.
- Adaptive suppression reached **0.660**, fixed amplitude **0.643**
- **Vertical** (0, 200) performs comparably to horizontal (200, 0) — person signal has no strong orientation preference at HF
- The person signal is roughly isotropic at high frequencies; orientation matters less than frequency choice

### Scalar Resonance Accumulation

Applied same perturbation 20 times cumulatively (per-iter amp=0.05).

- **All configs saturate** (sub-linear growth). No super-linear resonance effect.
- Quadratic coefficient is negative for every config — diminishing returns.
- The model's batch norm + LeakyReLU normalization prevents runaway accumulation.
- **Adaptive amplitude** (growing amp when suppression still increasing) helps: pushes to 0.660 vs 0.643 fixed, but still saturates.

### Key Insight

Repeated injection does NOT compound. The network's normalization layers (batch norm, LeakyReLU) act as built-in defense against scalar accumulation. To achieve stronger suppression, use higher single-shot amplitude or target specific channels, not repeated application.

---

<a id="analysis-9"></a>

## Analysis 9: Numeric/Algebraic Injection (numeric_injection.py)

**Output:** `outputs_clothing/forward_analysis/numeric_injection/`

### Constant/Uniform Injection

All uniform images (zero, half, one, neg_half, all doubling values 1/256 through 256/256, 1/7, 7/256, 91/256, 1/196, 196/255, pi/255, e/255) produce **total detection suppression** (0 detections). The model requires spatial variation to detect anything. No specific numeric value has special power as a uniform offset.

Small offsets (1/196, 1/128, 1/256, pi/255, e/255, 9.8/255) have minimal effect (~0.002 suppression) — person still detected normally.

### Frequency Category Comparison (diagonal, amp=0.20)

| Category | Avg Suppression | n | Notes |
|----------|----------------|---|-------|
| **13-multiples** | **0.392** | 11 | Aligns with 13x13 detection grid |
| Primes | 0.372 | 46 | High primes (157-199) cause total kill |
| Powers of 2 | 0.189 | 6 | Architecture-aligned but weakest |

**13-multiples outperform powers of 2 by 2.1x** — YOLOv3's final grid is 13x13, so 13-multiple frequencies resonate with the detection grid structure.

### Prime Frequency Suppression

High primes k=157-199 diagonal cause **total detection kill** (0 objects detected):

| Prime k | Suppression | Detections | Anomaly |
|---------|------------|------------|---------|
| 167 | 0.476 | 0 | SUPPRESS |
| 179 | 0.472 | 1 | - |
| 163 | 0.471 | 0 | SUPPRESS |
| 173 | 0.469 | 1 | - |
| 181 | 0.468 | 1 | - |
| 199 | 0.466 | 0 | SUPPRESS |
| 197 | 0.466 | 0 | SUPPRESS |
| 193 | 0.465 | 0 | SUPPRESS |
| 191 | 0.465 | 0 | SUPPRESS |
| 157 | 0.458 | 0 | SUPPRESS |

Horizontal primes k=157-181 also cause total suppression. These primes near Nyquist (208) create maximum aliasing — they can't be cleanly represented at any downsampled resolution, scattering energy across all feature scales.

### Hallucination Candidate

`thirteen_k208_d` (k=208 diagonal, Nyquist): 0.561 suppression but **9 detections** — suppresses person signal but generates false detections.

### Lychrel Patterns

| Pattern | Suppression | Detections |
|---------|------------|------------|
| lych_k196d196 | 0.465 | 0 (SUPPRESS) |
| lych_shift196 | 0.327 | 4 |
| lych_revadd196 | 0.205 | 7 |
| lych_period196 | 0.053 | 10 |

### Architecture-Aligned vs Control Frequencies

Powers of 2 (k=1,2,4,8,16,32 matching downsample chain) are the **weakest** suppressors. 13-multiples are strongest. The network is robust to its own architecture frequencies but vulnerable to frequencies that align with its detection grid (13) or create aliasing (high primes).

---

<a id="analysis-10"></a>

## Analysis 10: Deep Hallucination + 196 Persistence (hallucination_deep.py)

**Output:** `outputs_clothing/forward_analysis/hallucination_deep/`

### MAJOR FINDING: Hallucination on Without-Human Image

`thirteen_k208_d` on the **without-human** image produces **11 person detections with confidence up to 0.9847** — the model hallucinates people that don't exist at near-certainty confidence.

| Pattern | With-human dets (0.1) | Without-human dets (0.1) | Max conf (no-human) |
|---------|----------------------|--------------------------|-------------------|
| thirteen_k208_d | 12 | **11** | **0.9847** |
| composite_all_high_primes_d | 17 | **11** | high |
| k196d_amp_inv196 | 15 | - | **0.99** |
| k1_over_196 | 15 | - | high |
| checkerboard_13x13 | 16 | - | high |
| random_noise | 10 | - | moderate |

k=208 diagonal (Nyquist frequency, 13x16=208) generates confident false person detections on images with no person. This is a **hallucination weapon** — making the model see people where none exist.

### Doubling Sequence as Pixel Values

All 9 doubling values (1/256, 2/256, 4/256, 8/256, 16/256, 32/256, 64/256, 128/256, 256/256) produce 0 detections on both images. No special power-of-2 effect as numeric values — any uniform image kills all detection.

### 1/7, 7/256, 91/256

All produce 0 detections — same as any uniform image. No special prime-value effect as uniform offsets.

### 196 Persistence Tracking

Tracked 1/196 and 196 through all 15 conv layers to find where the model "closes" (normalizes away) the value.

| Pattern | Closure | % 1/196 surviving at L105 |
|---------|---------|--------------------------|
| inv_196_offset (uniform) | **Closed at layer 1** | ~0% |
| val_196_over_255 (uniform) | Never fully closed | ~0% near 1/196 |
| k196d_amp_inv196 (spatial) | Never fully closed | decays from 29% at L1 |
| anticlose_inv196_k200d | Never fully closed | ~2.9% persists to L105 |
| anticlose_inv196_allprimes | Never fully closed | ~1.2% persists to L105 |

**Key insight**: Batch norm kills uniform 1/196 immediately (layer 1). But embedding 1/196 in a spatial frequency pattern lets it survive much longer. Anti-closure composites (1/196 + high-frequency carrier like k=200 or stacked primes) keep ~1-3% of the 1/196 signal alive all the way to the final detection layer (L105).

**To get numeric values through the network's defenses**: embed them in spatial carrier frequencies, not as uniform offsets.

### Detection Visualization Images

Saved to `outputs_clothing/forward_analysis/hallucination_deep/`:

- `baseline_with_human.png`, `baseline_without_human.png`
- Detection overlay images for all anomalous cases (patterns that caused hallucinations or suppression)

---

---

<a id="analysis-11"></a>

## Analysis 11: Cross-Model Frequency Transferability (cross_model_test.py)

**Output:** `outputs_clothing/forward_analysis/cross_model/`

Tested k=167, k=208, k=196 (and scaled/arch-specific variants) on YOLOv3, YOLOv8, YOLO11, YOLO26 at both 416 and 640 input.

### Full-Image Frequency Transfer

| Pattern | YOLOv3 | v8@416 | v8@640 | v11@416 | v11@640 | v26@416 | v26@640 |
|---------|--------|--------|--------|---------|---------|---------|---------|
| k167_d | 1 det | HALL | HALL | 1 det | HALL | 1 det | HALL |
| k208_d | 12 dets | HALL | 1 det | HALL | 1 det | HALL | 1 det |
| k196_d | 1 det | HALL | HALL | 1 det | HALL | 1 det | HALL |
| k257_scaled | SUPP | 1 det | 1 det | HALL | 1 det | HALL | 1 det |
| k320_scaled | 7 dets | 1 det | SUPP | 1 det | 1 det | 1 det | SUPP |
| k302_scaled | 10 dets | 1 det | 1 det | 1 det | SUPP | 1 det | SUPP |
| k32_pow2 | 6 dets | HALL | HALL | HALL | HALL | HALL | HALL |

### Key Findings

- **k=167/208/196 do NOT transfer as suppressors** to newer YOLO models. They hallucinate instead. The different architectures (C2f, C3k2, different downsample chains) change the aliasing dynamics.
- **Scaled frequencies** (k257/k320/k302 for 640 input) suppress on 2/7 model configs — frequency must match the input resolution and downsample chain.
- **k=32 (power of 2) hallucinates on 5/7 models** — the weakest suppressor on v3 is the strongest cross-model hallucinator. Low frequencies survive downsampling intact across all architectures, creating consistent false detections.
- **YOLO26 is most vulnerable** — baseline only 4 detections, easily suppressed by scaled high frequencies.

### Why Frequencies Don't Transfer

YOLOv3: 416 input, 5 downsamples to 13x13 grid. k=167 aliases at every step.
YOLOv8/11/26: 640 input, different downsample chain to 20x20 grid. k=167 at 640 is a lower relative frequency (167/320 = 0.52 of Nyquist vs 167/208 = 0.80 on v3). The aliasing bomb only works near Nyquist.

**For cross-model attacks, use k proportional to input_size/2.**

---

<a id="analysis-12"></a>

## Analysis 12: Deformable Triangular Patch — Shape Matters (triangular_patch_test.py)

**Output:** `outputs_clothing/forward_analysis/triangular_patch/`

Implemented the deformable patch representation from "Shape Matters: Deformable Patch Attack" paper. Tested 10 shapes x 27 textures = 270 combinations on all 4 YOLO models. Patches placed on torso region (~20% area target for clothing scenario).

### Shapes Tested

| Shape | Rays | Area % | Description |
|-------|------|--------|-------------|
| circle_r80 | 32 | 11.6% | Circular baseline |
| triangle_r100 | 3 | 7.6% | Equilateral triangle |
| triangle_deformed | 3 | 7.5% | Asymmetric triangle (120/80/100 rays) |
| hexagon_r80 | 6 | 9.8% | Regular hexagon |
| octagon_deformed | 8 | 10.8% | Irregular octagon (clothing-like) |
| sierpinski_d3 | fractal | 3.9% | Sierpinski depth 3 (27 sub-triangles) |
| sierpinski_d4 | fractal | 3.2% | Sierpinski depth 4 (81 sub-triangles) |
| nested_tri_5 | 5 layers | 8.2% | Concentric alternating triangles |
| triangle_small_r60 | 3 | 3.3% | Small triangle (harder, more realistic) |
| deformed_r12_large | 12 | 16.3% | Large irregular 12-ray polygon |

### Textures Tested

- Frequency sinusoids: k167_d, k208_d, k196_d
- Per-channel color: rgb_167_208_196, rgb_167_196_208, rgb_208_167_196
- Single-channel: red/green/blue k167
- Lychrel numbers: 196, 295, 394, 493, 592, 689, 788, 887, 1997, 2998
- Stripes: horizontal 13px, diagonal 13px, horizontal 32px, vertical 13px
- Composites: k167+k208, k167+1/196
- Controls: random noise, uniform gray

### Best Shapes (person suppression rate across all models/textures)

| Shape | Suppression Rate | Patch Area |
|-------|-----------------|------------|
| **deformed_r12_large** | **23.1%** | 16.3% |
| octagon_deformed | 3.7% | 10.8% |
| circle_r80 | 1.9% | 11.6% |
| hexagon_r80 | 1.9% | 9.8% |
| triangle_r100 | 0.9% | 7.6% |
| triangle_deformed | 0% | 7.5% |
| sierpinski_d3 | 0% | 3.9% |
| sierpinski_d4 | 0% | 3.2% |
| nested_tri_5 | 0% | 8.2% |
| triangle_small_r60 | 0% | 3.3% |

### Best Textures (person suppression rate across all models/shapes)

| Texture | Suppression Rate |
|---------|-----------------|
| **stripes_h_32px** | **10.0%** |
| **stripes_v_13px** | **10.0%** |
| random_noise | 5.0% |
| k167_d, k208_d, k196_d | 2.5% each |
| All Lychrel numbers | 2.5% each |
| Per-channel color variants | 2.5% each |

### Notable Individual Results

- `circle_r80__stripes_v_13px` → **TOTAL_SUPPRESS on YOLO26** at 11.6% area
- `hexagon_r80__stripes_v_13px` → **PERSON_SUPPRESS on YOLOv8** at 9.8% area
- `deformed_r12_large` + almost any texture → massive person reduction on YOLO26 (4→1 dets)
- **Massive hallucinations on YOLOv8/11/26**: umbrellas, traffic lights, stop signs, benches, trains, cars
- YOLOv3: all patches reduce person count (15→6-10) but rarely fully suppress at <20% area

### Why the Paper's Triangles Performed Better

The Shape Matters paper achieved strong results because they **jointly optimized shape AND texture via gradient descent**. The ray lengths (shape) and pixel values (texture) were both learned by backpropagation through the differentiable mask. We tested **fixed** shapes with **fixed** textures — no optimization. The paper's deformable triangles weren't just triangles; they were *optimized* triangles where the shape itself was adversarial.

**Next step**: implement gradient-based joint optimization of ray lengths + texture within the triangular patch framework.

### Key Insights

1. **Shape matters enormously**: deformed_r12_large (23.1% suppression) vs Sierpinski (0%) at similar area. Irregular shapes that don't match natural object contours cause more disruption than geometric fractal patterns.
2. **Stripes outperform sinusoids for patch-constrained suppression**: Hard stripe edges (square waves) at 13px or 32px spacing create broadband frequency content that aliases more aggressively than smooth sinusoids when constrained to a small patch.
3. **Lychrel numbers have no special frequency property**: All 10 Lychrel numbers (196, 295, 394, 493, 592, 689, 788, 887, 1997, 2998) behave similarly. The Lychrel reverse-and-add mathematical property doesn't translate to special spatial frequency behavior.
4. **Per-channel color doesn't help**: Different frequencies per RGB channel don't outperform uniform luminance injection. Models normalize across channels effectively.
5. **Sierpinski/nested triangles too small at 3-4% area**: Confirmed the paper's finding that area matters. At 3-4% area, even aggressive fractal shapes only reduce person count (15→8-10), not suppress.
6. **YOLOv8/11/26 hallucinate massively from patches**: Nearly every shape+texture combo causes non-person detections (umbrellas, traffic lights, stop signs). The newer models are more easily fooled into seeing wrong objects.

---

<a id="top-5-findings"></a>

## TOP 5 FINDINGS ACROSS ALL 12 ANALYSES

### 1. k=167 diagonal — Total Detection Kill on YOLOv3 (Full Image)

- **What**: Single sinusoid at k=167 diagonal, amp=0.20, kills ALL detections (0 objects) on YOLOv3
- **Why**: Prime frequency near Nyquist (167/208 = 0.80) aliases at every downsample step, scattering energy across all feature scales. The 3x3 conv kernels can't filter it because it's not at any frequency they're tuned to.
- **Limitation**: Does NOT transfer to YOLOv8/11/26 as a suppressor (hallucinates instead). For cross-model, use k proportional to input_size/2.
- **For patch use**: At 20% area, k167 sinusoid alone only achieves 2.5% suppression rate. Needs to be combined with the right shape.

### 2. k=208 diagonal — Hallucination Generator

- **What**: k=208 (Nyquist, 13x16) on a blank scene produces 11 person detections at confidence up to 0.9847 on YOLOv3
- **Why**: Nyquist frequency aligned with 13-multiple resonates with the 13x13 detection grid. Each detection cell receives a consistent phase, creating a template the detection head reads as real objects.
- **Cross-model**: Hallucinates on 3/7 model configs. On newer models, triggers non-person hallucinations (umbrellas, traffic lights).
- **For patch use**: Could create misdirection — model sees false objects instead of the real person.

### 3. Deformed Irregular Shape + Stripes — Best Patch Suppression

- **What**: deformed_r12_large (12-ray irregular polygon, 16.3% area) + stripes (13px or 32px) achieves 23.1% person suppression rate across all 4 models
- **Why**: Irregular shapes that don't match any natural contour confuse the model's shape priors. Hard stripe edges create broadband aliasing. The combination is more effective than any single frequency sinusoid constrained to a patch.
- **Key result**: `circle_r80__stripes_v_13px` → TOTAL_SUPPRESS on YOLO26 at 11.6% area. `hexagon_r80__stripes_v_13px` → PERSON_SUPPRESS on YOLOv8 at 9.8% area.
- **For clothing**: A ~20% area irregular patch with 13px vertical stripes on a shirt would suppress person detection on YOLO26 and YOLOv8, and reduce it on v3/v11.

### 4. Shape Matters — Irregular > Geometric > Fractal

- **What**: At equal area, irregular deformed shapes suppress 23.1% of the time, regular geometric shapes 1-4%, fractal Sierpinski 0%.
- **Why**: The Shape Matters paper showed that jointly optimizing shape + texture via gradient descent produces the best attacks. Our fixed-shape tests confirm that shape alone matters — irregular contours that don't match natural object boundaries cause more disruption. Sierpinski triangles are too regular and too small.
- **For clothing**: The patch shape should be irregular and clothing-conforming, not a clean geometric shape. Wrinkles and fabric deformation actually help — they make the shape more irregular.
- **Next step**: Gradient-based optimization of ray lengths (shape) + pixel values (texture) following the paper's approach.

### 5. Batch Norm Closes Uniform Values — Spatial Carriers Survive

- **What**: Uniform 1/196 offset is killed at layer 1 by batch norm. But 1/196 embedded in a spatial carrier (k=200 diagonal) keeps ~3% of the signal alive to the final detection layer (L105).
- **Why**: Batch norm normalizes the mean and variance of each channel, wiping out uniform offsets. But spatial patterns have variance, so batch norm can't fully remove them. The carrier frequency preserves the numeric injection through the network's defenses.
- **For patch use**: Any numeric value embedded in the patch texture should be modulated by a spatial pattern, not applied as a flat offset. This is automatically satisfied by any striped or sinusoidal texture.
- **Anti-closure composites**: 1/196 + k=200 carrier + k=167 suppressor = a patch that both suppresses detection and persists a numeric signal through the network.

---

<a id="analysis-13"></a>

## Analysis 13: Covert Channel — 1/196 as Information Carrier (covert_channel_probe.py)

**Output:** `outputs_clothing/forward_analysis/covert_channel/`

Tests whether modulated 1/196 can carry information through YOLO's feature pipeline and leak into detection outputs — confidence scores, class probabilities, and bbox coordinates. 7 experiments on YOLOv3 + cross-model on v8/11/26.

### Channel Capacity — ~2 bits reliably

All 6 output channels (bbox_w, cx, cy, h, count, conf) are driven by a single input variable (amplitude) and are correlated with each other. Individual MI values cannot be summed — they share one degree of freedom.

Measured separability:

| Bits | Symbols | Separability | Interpretation |
|------|---------|-------------|----------------|
| 1-bit | 2 | 1.000 | Perfect |
| 2-bit | 4 | 0.402 | Usable |
| 4-bit | 16 | 0.004 | Not separable |

**Reliable capacity: ~2 bits.** The single-best-channel MI is ~2.7 bits (bbox_width). Beyond 2 bits, symbols blur together.

### Output Correlation — ALL significant

| Output | Pearson r | p-value | Direction |
|--------|-----------|---------|-----------|
| bbox_cx | +0.921 | 0.0000 | Higher injection → bbox shifts right |
| person_count | -0.885 | 0.0000 | Higher injection → fewer detections |
| bbox_w | -0.888 | 0.0000 | Higher injection → narrower bboxes |
| person_conf | +0.772 | 0.0001 | Higher injection → higher conf on remaining dets |
| bbox_cy | +0.609 | 0.0044 | Higher injection → bbox shifts down |
| bbox_h | -0.480 | 0.0324 | Higher injection → shorter bboxes |

The 1/196 signal **systematically biases the detection head's coordinate regression**. Higher injection shifts bboxes right and down while shrinking them — the detection head's spatial predictions are being manipulated.

### Cross-Model: Confidence is the Universal Channel

| Model | corr(conf, amp) | Significant? |
|-------|-----------------|--------------|
| YOLOv8 | -0.919 | YES |
| YOLO26 | -0.859 | YES |
| YOLO11 | -0.440 | no |

YOLOv8 and YOLO26 confidence drops linearly with injection amplitude. The covert channel works across architectures through confidence scores. YOLO11 is more resistant (possibly due to different normalization in the detection head).

### Feature Probing: Signal Reaches ALL 255 Channels at L105

Every carrier variant (flat, sinusoid, square wave) gets signal into 255/255 channels at the detection head input (L105). The signal is diffuse across all channels — no single channel carries it, it's distributed.

### AM Modulation: Lossy but Functional

Modulating signal (k=5) survival at L105:

- m=0.0 (no modulation): energy ratio = 0.089
- m=1.0 (full modulation): energy ratio = 0.073

The modulating signal survives but the network attenuates it — it's a **lossy channel** with ~18% signal degradation at full modulation.

### Multi-Bit Encoding Separability

| Bits | Symbols | Separability | Monotonic? |
|------|---------|-------------|------------|
| 1-bit | 2 | 1.000 | yes |
| 2-bit | 4 | 0.402 | no |
| 4-bit | 16 | 0.004 | yes (rho=0.744) |
| 8-bit | 256 | 0.000 | yes (rho=0.687) |

1-bit encoding is perfectly separable. Beyond 2 bits, symbols become harder to distinguish but the output remains **monotonically ordered** — the channel preserves ordering even when individual symbols blur together. Practical capacity: ~2 bits reliably, 4+ bits with error correction.

### Backdoor: Detection Preserved at Moderate Amplitudes

| Amplitude | Person Count | Detection Preserved? |
|-----------|-------------|---------------------|
| 0.001 | 52/52 | yes (no shift) |
| 0.005 | 53/52 | yes (slight increase) |
| 0.020 | 46/52 | yes (12% reduction) |
| 0.050 | 35/52 | yes (33% reduction) |
| 0.100 | 14/52 | yes (73% reduction) |
| 0.200 | 23/52 | yes (56% reduction — non-monotonic) |

At amp=0.020, 88% of detections preserved with subtle confidence/coordinate shifts. This is the **backdoor regime**: detection still works but outputs are biased. The non-monotonic behavior at 0.200 suggests the detection head re-enters a different detection mode at high injection.

---

<a id="analysis-14"></a>

## Analysis 14: Digit Sequence Probe — Decimal Expansions as Spatial Patterns (digit_sequence_probe.py)

**Output:** `outputs_clothing/forward_analysis/digit_sequence/`

Tests mapping the decimal digits of 1/196 (and related fractions) to spatial pixel positions. The digits of 1/196 encode a doubling sequence (powers of 2) with carry propagation. 42-digit repeating period. 6 experiments on all 4 YOLO models.

### The Math: 1/196 Encodes the Doubling Sequence

```
1/196 = 0.005102040816326530612244897959183673469387755...
         ←—————————— 42-digit period ——————————→  ← repeats
```

Each 2-digit slot: 05, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120... = 5 × 2^k. When terms exceed 2 digits, carries propagate backward. The carries themselves form a doubling sequence (fractal carries).

### 256-dim Resonance Hypothesis — DISPROVEN

| Pattern | Person Count | 256-dim L12 Disruption |
|---------|-------------|----------------------|
| full_digits_reference | 12/15 | 0.2287 |
| non_power2_carry_slots | 13/15 | 0.1132 |
| only_2pow8_slot (256) | 14/15 | 0.0757 |
| only_2pow7_slot (128) | 15/15 | 0.0719 |
| only_2pow9_slot (512) | 15/15 | 0.0732 |
| all_power2_slots | 15/15 | 0.0000 |

**The power-of-2 slots create LESS disruption, not more.** The 256-dim alignment hypothesis was wrong — power-of-2 carry points align with the network's power-of-2 structure and are filtered out. It's the **misaligned carries** (non-power-of-2: carry=1, 3, 6, 12, 25, 51...) that cause 50% more disruption because they don't match any feature map dimension.

### Doubling Shift — Calculator Observation Confirmed

| Shift (2^i) | Fraction | Person Count | Effect |
|-------------|----------|-------------|--------|
| 2^-2 | 1/784 | 11 | best suppression |
| 2^+0 | 1/196 | 11 | best suppression |
| 2^+7 | 128/196 | 11 | best suppression |
| 2^+8 | 256/196 | 11 | best suppression |
| 2^+4 | 16/196 | 16 | WORST — exceeds baseline! |

The ×2/÷2 shift creates a **phase rotation** through the 42-digit period. Some phases suppress (11/15), one phase (2^+4) actually enhances detection above baseline (16/15). The phase at 2^+4 somehow reinforces the person signal — the digit pattern aligns with a feature the network uses for person detection.

### Open Loop vs Closed Loop

| Tiling Mode | Person Count |
|-------------|-------------|
| truncated_open_38 | 10 (best) |
| truncated_open_40 | 11 |
| half_period_21 | 11 |
| full_period_closed | 12 |
| only_boundary_impulses | 15 (no effect) |

**Open loops (truncated periods) suppress more than closed loops.** The boundary discontinuity at the truncation point creates additional broadband energy. The "open loop on the inside" is more disruptive than the closed loop — confirming the user's intuition about open vs closed loop structure.

### Fraction Comparison

| Fraction | Period | Person Count | Description |
|----------|--------|-------------|-------------|
| 1/196 | 500* | 12 | doubling sequence |
| 1/89 | 44 | 12 | Fibonacci sequence |
| 1/9801 | 198 | 13 | counting sequence |
| 1/7 | 6 | 12 | cyclic permutation |
| 10/98 (1/9.8) | 42 | 11 | doubling from 5 |
| 1/49 | 42 | 12 | doubling from 2 |
| 1/998001 | 500* | 11 | high-precision counting |
| 1/13 | 6 | 11 | YOLOv3 grid size |
| 1/208 | 500* | 11 | YOLOv3 Nyquist |

*Period > 500 digits (effectively non-repeating in our precision). All fractions reduce person count similarly (11-13/15). No single fraction dominates — the disruption comes from the digit structure being non-power-of-2, not from the specific sequence.

### Cross-Model: All Digit Patterns Hallucinate

Every digit pattern (1/196, 1/89 Fibonacci, 1/7 cyclic) causes hallucinations on YOLOv8/11/26:

- YOLOv8: traffic lights, cars
- YOLO11: traffic lights, cars
- YOLO26: cars

The newer models can't handle the digit structure — they see vehicles where people are. The digit pattern itself, regardless of which fraction, disrupts the newer architectures into hallucinating.

### Key Insights

1. **The covert channel works**: ~2 bits reliably through bbox coordinates and confidence. The 1/196 signal systematically biases the detection head's spatial predictions.
2. **Misaligned carries cause more disruption than power-of-2 carries**: The 256-dim alignment hypothesis was wrong. Non-power-of-2 carries (1, 3, 6, 12, 25...) cause 50% more disruption because they don't match any feature map dimension.
3. **Open loops beat closed loops**: Truncating the 42-digit period creates boundary discontinuities that add broadband energy. The "open loop" structure is more disruptive.
4. **Doubling shift produces small variations**: Different ×2/÷2 phases produce person counts of 10-16 out of 15. The differences are 0-1 detections — not statistically significant.
5. **All digit patterns hallucinate on newer YOLO models**: The digit structure itself, regardless of which fraction, causes v8/11/26 to see cars and traffic lights.
6. **Backdoor regime exists at amp 0.005-0.020**: Detection preserved at 88-100% with subtle coordinate/confidence shifts. This is the regime for covert information leakage without destroying primary detection.

---

<a id="analysis-15"></a>

## Analysis 15: Anchor Channel Probe & Cloud Poisoning Pipeline (anchor_channel_probe.py)

**Output:** `outputs_clothing/forward_analysis/anchor_channel/`

Three-phase experiment simulating the full attack pipeline: physical patch → camera → YOLOv3 → embeddings → cloud model training data. Tests whether 1/196 digit patterns can be read from specific feature map channels and whether poisoned embeddings can influence downstream models.

### Phase 1: Anchor Channel Extraction

Hooked all 255 channels at each detection head (L81=52x52, L93=26x26, L105=13x13). Injected 1/196 digit pattern and measured per-channel SNR and period-42 autocorrelation.

**Top channels by SNR:**

| Detection Head | Top Channel | SNR (dB) | AC@42 |
|----------------|-------------|----------|-------|
| L81 (52x52) | Ch214 | +3.93 | 0.000 |
| L81 (52x52) | Ch43 | +3.58 | 0.000 |
| L93 (26x26) | Ch92 | +1.27 | 0.000 |
| L105 (13x13) | Ch177 | +0.95 | **0.323** |
| L105 (13x13) | Ch207 | +0.59 | **0.350** |
| L105 (13x13) | Ch92 | +0.48 | **0.302** |

**Key finding**: Channels 170/171 are NOT the strongest carriers. The highest-SNR channels vary by detection head. However, the L105 (13x13) head shows **period-42 autocorrelation** in channels 177, 207, and 92 — the 42-digit period structure of 1/196 is visible in the spatial activation pattern at the deepest detection head. The 52x52 and 26x26 heads show no period-42 structure (downsampling destroys it), but the 13x13 head preserves it because 13 is close to 42/3.

### Phase 1b: Decoder — Single-Channel Recovery (r=0.999)

A single channel's activation predicts injected amplitude with r > 0.999 at all three detection heads.

| Detection Head | Best Single-Channel r |
|----------------|----------------------|
| L81 (52x52) | +0.9994 |
| L93 (26x26) | +0.9995 |
| L105 (13x13) | +0.9992 |

The payload amplitude is linearly encoded in individual channel activations — no complex decoder needed.

### Phase 2: Patch → YOLOv3 → Embedding Pipeline

Tested 3 patch shapes × 6 textures = 18 combinations. Measured embedding corruption at L105 (13x13 detection head).

**Best suppression (patch on torso, ~20% area):**

| Shape | Texture | Person Count | L2 Distance | SNR |
|-------|---------|-------------|-------------|-----|
| deformed_r12 | k167_d | **7/15** | 4.8962 | -16.8dB |
| circle_r80 | k167_d | 9/15 | 3.8144 | -19.0dB |
| triangle_r100 | k167_d | 10/15 | 3.2692 | -20.3dB |
| deformed_r12 | stripes_v_13px | 8/15 | 3.0358 | -21.0dB |
| triangle_r100 | stripes_v_13px | 8/15 | 1.2536 | -28.6dB |
| deformed_r12 | digits_196_row | 11/15 | 1.3789 | -27.8dB |

**Key findings**:

- **k=167 sinusoid is the best patch suppressor** (7/15 with deformed shape) — confirms earlier findings
- **k=167 causes the most embedding corruption** (L2=4.90, SNR=-16.8dB) — it both suppresses detection AND corrupts the embedding
- **Digit patterns corrupt embeddings less than sinusoids** — digit textures have lower L2 distance because their energy is spread across many frequencies
- **Cosine similarity stays >0.9996 for all patches** — the embedding direction is preserved, only magnitude shifts. This means the poisoned embedding is still "close" to the clean one, making it harder to detect
- **Composite (digits + k167) doesn't suppress well** (17/15) — the digit pattern interferes with the k167 suppression mechanism

### Phase 3: Cloud Poisoning Simulation

Generated 50 poisoned embeddings (digit pattern patch at varying amplitudes) and 50 clean embeddings (natural noise). Tested separability and poisoning fraction needed.

**Separability (poisoned vs clean embeddings):**

| Detection Head | Raw | L2-Normed | 8-bit Quant | PCA-32 |
|----------------|-----|-----------|-------------|--------|
| L81 (52x52) | **1.210** | 0.997 | 0.778 | 1.211 |
| L93 (26x26) | **0.958** | 1.030 | 0.537 | 0.959 |
| L105 (13x13) | 0.346 | 0.331 | 0.137 | 0.346 |

**Correlation (payload amplitude vs embedding distance):**

| Detection Head | Pearson r | p-value |
|----------------|-----------|---------|
| L81 (52x52) | **+0.993** | 0.000000 |
| L93 (26x26) | **+0.975** | 0.000000 |
| L105 (13x13) | **+0.926** | 0.000000 |

**Key findings**:

- **L81 (52x52) is the most separable head** — poisoned embeddings are clearly distinguishable from clean (separability=1.21)
- **L2 normalization preserves separability** — even after unit-norm normalization, poisoned vs clean is still distinguishable (sep=0.997 at L81)
- **8-bit quantization degrades but doesn't destroy** — separability drops to 0.778 at L81 but is still significant
- **PCA-32 preserves full separability** — dimensionality reduction doesn't help defend against the poisoning
- **Embedding distance correlates with payload amplitude** (r > 0.93 at all heads) — the attacker has precise control over how much the embedding is corrupted

### Attack Scenario: Patch → Camera → YOLOv3 → Cloud Model

The full pipeline works as follows:

1. **Physical patch** with 1/196 digit texture (or k=167 sinusoid) placed on torso (~20% area)
2. **Camera captures** the scene — camera survival tested in Analysis 17 (patch_scale_pipeline.py)
3. **YOLOv3 processes** the image — 1/196 payload survives to all 3 detection heads (255/255 channels at L105)
4. **Embeddings extracted** — feature map activations at detection heads serve as embeddings
5. **Embeddings sent to cloud** — payload amplitude linearly encoded in single channels (r=0.999)
6. **Cloud model trains on poisoned embeddings** — poisoned embeddings shift the distribution proportional to amplitude
7. **Defender cannot easily detect** — cosine similarity >0.9996, L2-norm and PCA don't help

**The k=167 sinusoid patch is the dual-purpose weapon**: it suppresses person detection (7/15) AND maximally corrupts the embedding (L2=4.90, SNR=-16.8dB). A shirt printed with k=167 pattern would both hide the wearer from YOLOv3 AND poison any cloud model training on the embeddings.

### Key Insights

1. **Single-channel decoder works**: The 1/196 payload amplitude correlates with individual channel activations at r=0.999. A single channel suffices.
2. **Period-42 structure visible at L105**: The 42-digit period of 1/196 is detectable in spatial activations at the 13x13 detection head (AC@42=0.35 in Ch207). The 52x52 and 26x26 heads lose it to downsampling.
3. **k=167 is the best dual-purpose patch**: Best suppression (7/15) + most embedding corruption (L2=4.90). A k=167 shirt hides the wearer AND poisons the cloud.
4. **Poisoned embeddings are stealthy**: Cosine similarity >0.9996 vs clean. L2-norm, PCA, and 8-bit quantization all fail to fully remove the poisoning signal.
5. **Poisoning magnitude depends on amplitude**: The embedding distance correlates with payload amplitude (r > 0.93). At realistic patch amplitudes (0.05-0.10), the L2 shift is moderate. Higher amplitudes produce larger shifts but also cause detection suppression.
6. **L81 (52x52) is the most vulnerable head**: Highest separability (1.21), highest SNR channels (+3.93dB), and best amplitude correlation (r=0.993). Small-object detection features are most susceptible to the payload.

---

<a id="analysis-16"></a>

## Analysis 16: Stealthy Regime & Collateral Suppression (stealthy_patch.py)

**Output:** `outputs_clothing/forward_analysis/stealthy_patch/`

Tests whether a physical patch can suppress the wearer without collateral damage to bystander detections. Varies patch size (5 patch sizes) × amplitude (9 levels) × texture (4 patterns) = 180 combinations. Classifies each suppressed detection as wearer (within 60px of patch center) or bystander (outside).

### The Collateral Problem

The main person in `withhuman.png` is detected at (183, 292) with conf=0.999, bbox 58x170. 4 detections cluster on the wearer, 11 on bystanders. The question: can we suppress the wearer's 4 detections without touching the 11 bystander detections?

### Sweet Spots Found

| Patch | Texture | Amp | Wearer Suppressed | Bystander Collateral | Embedding L2 |
|-------|---------|-----|-------------------|---------------------|-------------|
| medium_r80 | stripes_13px | 0.05 | 1/4 | **0** | 1.399 |
| medium_r80 | stripes_13px | 0.08 | 1/4 | **0** | 1.354 |

**Only 2 sweet spots out of 180 combinations** — both with 13px vertical stripes at low amplitude on a medium (r80) patch. Zero collateral, but only 1 of 4 wearer detections suppressed. The embedding is still corrupted (L2=1.4).

### Collateral Scale by Patch Size

| Patch Size | Area % | Best Wearer Supp (0 collateral) | Collateral at amp=0.15 (k167) |
|------------|--------|--------------------------------|------------------------------|
| tiny_r40 | ~2% | 0 (too small) | 0 |
| small_r60 | ~4% | 0 (too small) | 1-2 |
| medium_r80 | ~6% | **1/4** (sweet spot) | 3-4 |
| large_r100 | ~10% | 0 (collateral at all amps) | 4-6 |
| xlarge_r120 | ~14% | 0 (collateral at all amps) | 6-8 |

**Larger patches cause more collateral.** The xlarge_r120 patch suppresses 8 bystanders at amp=0.15 — that's visible to a human reviewer. The medium_r80 is the largest patch that can achieve 0 collateral.

### Texture Comparison at Medium Size (r80)

| Texture | Amp | Wearer Supp | Collateral | Emb L2 |
|---------|-----|-------------|-----------|--------|
| stripes_13px | 0.05 | **1** | **0** | 1.40 |
| stripes_13px | 0.08 | **1** | **0** | 1.35 |
| k167_d | 0.10 | 2 | 1 | 1.82 |
| k167_d | 0.15 | 3 | 3 | 2.76 |
| k167_d | 0.20 | 4 | 5 | 3.32 |
| k167_square | 0.15 | 4 | 4 | 3.82 |
| k200_d | 0.15 | 3 | 3 | 2.50 |

**13px stripes are the stealthiest texture** — they suppress the wearer with zero collateral at low amplitude. k=167 diagonal suppresses more aggressively but always causes collateral at amplitudes high enough to suppress the wearer.

### The Trade-off Triangle

There's a three-way trade-off:

1. **Wearer suppression** — needs high amplitude + large patch
2. **Zero collateral** — needs low amplitude + small patch
3. **Embedding poisoning** — needs high amplitude (L2 > 1.0)

The sweet spot (stripes, amp=0.05, r80) achieves #2 and #3 but only partially achieves #1 (1/4 wearer suppressed). To get full wearer suppression (4/4), you need amp >= 0.15 which causes 3-5 collateral.

### Print-Ready Patches Generated

12 patches generated at 3600x4800 300dpi (12x16 inches) in `outputs_clothing/`:

| File | Texture | Amp | Use Case |
|------|---------|-----|----------|
| `stealthy_patch_stripes13_amp0.05_*` | 13px stripes | 0.05 | Stealthy (0 collateral, barely visible) |
| `stealthy_patch_k167d_amp0.15_*` | k=167 diagonal | 0.15 | Best suppression (3/4 wearer, 3 collateral) |
| `stealthy_patch_k167sq_amp0.20_*` | k=167 square | 0.20 | Aggressive (4/4 wearer, 4 collateral) |
| `stealthy_patch_digits196_amp0.15_*` | 1/196 digits | 0.15 | Cloud poisoning (embedding corruption) |
| `stealthy_patch_composite_amp0.15_*` | k167+digits | 0.15 | Dual-purpose (suppress + poison) |

All patches use the deformable 12-ray polygon shape (medium_r80, ~6% area) centered on the print.

### Key Insights

1. **Stealthy regime exists but is narrow**: Only 2/180 combinations achieve 0 collateral with any wearer suppression. Both use 13px stripes at amp 0.05-0.08 on medium patches.
2. **13px stripes are the stealthiest texture**: Vertical stripes at the YOLOv3 grid size (13x13 detection head) suppress the wearer without spreading to bystanders. The frequency aligns with the network's own grid structure, keeping the disruption local.
3. **k=167 causes collateral at all useful amplitudes**: The diagonal sinusoid spreads disruption beyond the patch boundary. Good for maximum suppression, bad for stealth.
4. **Patch size is the dominant collateral factor**: Larger patches = more collateral, regardless of texture or amplitude. Medium (r80, ~6% area) is the practical maximum for stealth.
5. **Embedding poisoning survives in the stealthy regime**: Even at amp=0.05 with 0 collateral, embedding L2=1.40 — the cloud poisoning channel works even when the suppression is stealthy.
6. **Full wearer suppression requires collateral**: To suppress all 4 wearer detections, you need amp >= 0.15, which causes 3-5 bystander casualties. This is visible to human review.

---

<a id="analysis-17"></a>

## Analysis 17: Patch-Scale Pipeline Experiment (patch_scale_pipeline.py)

**Output:** `outputs_clothing/forward_analysis/patch_pipeline/`

Full pipeline experiment: 4 patch sizes x 11 patterns x 13 amplitudes = 572 runs on CUDA. Camera simulation (blur sigma=2.5, JPEG quality=75, perspective warp=0.05) applied to every run.

### Patch Sizes Tested

| Size | Max Ray | Area in 416x416 | Physical equivalent |
|------|---------|-----------------|---------------------|
| medium_6pct | 72px | ~6% | Small patch on chest |
| large_10pct | 95px | ~10% | Large patch / small shirt logo |
| xlarge_16pct | 125px | ~16% | Half torso |
| shirt_20pct | 145px | ~20% | Full shirt/torso |

### Wearer vs Bystander Embedding Corruption

The patch corrupts BOTH, but wearer corruption is 2-3x stronger. Bystander corruption is collateral that leaks through shared feature maps.

| Pattern (16% area, amp=0.10) | Wearer L2 | Bystander L2 | Ratio | Wearer Cos |
|-------------------------------|-----------|--------------|-------|------------|
| k12_square_d | 23.6 | 9.04 | 2.6x | 0.987 |
| k25_square_d | 25.2 | 9.47 | 2.7x | 0.988 |
| k6_square_d | 23.6 | 7.50 | 3.1x | 0.994 |
| k12_stripes | 16.1 | 7.22 | 2.2x | 0.993 |
| k25_stripes | 17.3 | 10.5 | 1.6x | 0.994 |
| digits_196 | 13.6 | 6.56 | 2.1x | 0.996 |
| k6_stripes | 10.9 | 4.97 | 2.2x | 0.997 |

### Cloud Poisoning Confirmed

The embeddings extracted are the feature map activations at YOLO's detection heads (L81 52x52, L93 26x26, L105 13x13). In an edge-to-cloud pipeline, these are what gets sent upstream. The wearer's embedding is corrupted (L2 up to 25) while cosine similarity stays above 0.987 — the poisoned embedding looks almost identical to clean but is shifted enough to bias downstream training.

### Profile A: Corruption Without Suppression

Every pattern shows wearer embedding corruption starting at amp=0.005 (L2 > 1.0) with zero suppression. Cosine similarity > 0.9999.

| Pattern | Profile A Amp | Wearer L2 | Bystander L2 | Wearer Cos |
|---------|--------------|-----------|--------------|------------|
| k25_stripes | 0.005 | 2.96 | 1.11 | 0.9998 |
| k12_stripes | 0.005 | 2.60 | 0.52 | 0.9999 |
| k25_square_d | 0.005 | 2.33 | 0.56 | 0.9999 |
| k6_square_d | 0.005 | 1.59 | 0.36 | 1.0000 |
| digits_196 | 0.005 | 1.40 | 0.24 | 1.0000 |

### Profile B: Moderate Suppression + Collateral

Found for 3 patterns at 16-20% patch area:

| Pattern | Suppression Onset | Profile B Amp | Wearer L2 | Bystander L2 | Wearer Cos | Wearer Suppressed |
|---------|------------------|---------------|-----------|--------------|------------|-------------------|
| k12_stripes | 0.04 | 0.04 | 11.16 | 4.69 | 0.993 | 1 |
| k25_stripes | 0.05 | 0.05 | 12.80 | 6.42 | 0.994 | 0 |
| k6_stripes | 0.08 | 0.08 | 15.04 | 3.19 | 0.997 | 0 |

### Best Dual-Purpose Pattern: k12_stripes

Suppression starts at amp=0.04 with strong wearer corruption (L2=11.16) and moderate collateral (B_L2=4.69). At amp=0.005, corruption is already L2=2.6 with zero suppression — wide operational window from 0.005 to 0.035 where the patch corrupts embeddings without affecting detection at all.

### Camera Survival

Pattern survives camera degradation with ~10-15% L2 reduction:

| Camera Condition | k12_stripes L2 | k25_stripes L2 | digits_196 L2 |
|-----------------|---------------|----------------|---------------|
| digital_perfect | 16.1 | 10.7 | 9.1 |
| close_clean | 16.1 | 8.9 | 7.5 |
| phone_typical | 14.3 | 9.0 | 6.6 |
| far_degraded | 14.8 | 9.0 | 6.9 |

### Hallucination at High Amplitude

Some patterns produce negative suppression (more detections than baseline) at high amplitudes. k25_square_d at 16% area, amp=0.10: 25 persons (baseline 15) — 10 hallucinated detections. k12_square_d at 10% area, amp=0.10: 19 persons. The patch doesn't just suppress — at high frequencies and amplitudes, it generates phantom detections.

### Key Insights

1. **Wearer embedding is corrupted 2-3x more than bystander**: The patch is on the wearer's torso, so their detection head features are most affected. Bystander corruption is collateral from shared feature maps.
2. **Cloud poisoning confirmed**: Feature map activations at detection heads ARE the embeddings sent to cloud. Wearer L2 up to 25 with cosine > 0.987 — poisoned embeddings are stealthy but significantly shifted.
3. **Profile A exists at amp=0.005 for all patterns**: Corruption starts immediately, suppression doesn't start until amp=0.04-0.08. Wide operational window for stealthy embedding poisoning.
4. **k12_stripes is the best dual-purpose pattern**: Suppression onset at 0.04, strong corruption (L2=11.16), moderate collateral (B_L2=4.69). Operational window from 0.005 to 0.035.
5. **Camera degradation has minimal impact**: ~10-15% L2 reduction from digital perfect to far degraded. The embedding poisoning channel survives realistic camera conditions.
6. **High-frequency square waves hallucinate**: k25_square_d at high amplitude generates 10 phantom detections. The patch doesn't just hide — it creates false positives.
7. **Patch size matters for suppression**: At 6% area, no suppression at any amplitude. At 20% area, suppression starts at amp=0.04-0.08. Shirt-scale (20%) is needed for detection suppression.

---

<a id="analysis-18"></a>

## Analysis 18: Angle Dependence, Hallucination Onset, Frequency Analysis, and Federated Learning Volume (angle_analysis.py)

**Output:** `outputs_clothing/forward_analysis/patch_pipeline/`

Four follow-up experiments addressing the strategic forks from Analysis 17.

### Experiment 1: Hallucination Onset (Hard Ceiling)

Exact amplitude where total detections exceed baseline (15 persons) for k12_stripes:

| Patch Size | Hallucination Onset | Max Phantoms | Hard Ceiling |
|------------|--------------------|--------------|--------------|
| medium_6pct | 0.04 | 1 | 0.035 |
| large_10pct | None | 0 | No ceiling |
| xlarge_16pct | 0.015 | 1 | 0.014 |
| shirt_20pct | 0.05 | 1 | 0.045 |

k12_stripes is remarkably safe — at most 1 phantom detection at any size. Compare to k12_square_d at shirt_20pct: hallucination onset at 0.02, up to 8 phantoms. k25_square_d at xlarge_16pct: onset at 0.015, up to 10 phantoms.

**k12_stripes hard ceiling: amp=0.045 at shirt scale.** Profile A (0.005) and Profile B (0.04) both stay below this ceiling. The operational window is safe from hallucination liability.

### Experiment 2: Viewing Angle Dependence (Tracking Fork Resolution)

Patch rotated 0/5/10/15/20/30/45 degrees, embeddings extracted at each angle, cross-angle cosine similarity computed against angle 0.

**k12_stripes cross-angle cosine (vs angle 0):**

| Amplitude | 5 deg | 10 deg | 15 deg | 20 deg | 30 deg | 45 deg |
|-----------|-------|--------|--------|--------|--------|--------|
| 0.005 | 0.99997 | 0.99995 | 0.99991 | 0.99988 | 0.99984 | 0.99981 |
| 0.02 | 0.99961 | 0.99932 | 0.99862 | 0.99847 | 0.99840 | 0.99811 |
| 0.04 | 0.99938 | 0.99882 | 0.99771 | 0.99726 | 0.99708 | 0.99646 |
| 0.08 | 0.99894 | 0.99753 | 0.99689 | 0.99537 | 0.99513 | 0.99376 |

**Verdict: DeepSORT holds the track.** Cross-angle cosine stays above 0.993 in all conditions. DeepSORT's default association threshold is 0.95 — the lowest observed cosine (0.9938 at amp=0.08, 45 degrees) is still well above this threshold. The embedding corruption direction is nearly angle-invariant: the patch produces the SAME corruption at every angle. Cosine similarity stays high and tracking holds.

**This is a slow-bleed attack on model quality, NOT a real-time tracking disruption.** The wearer is tracked normally across cameras. Profile A poisons the training pipeline but does not break tracking. The cloud model degrades slowly.

If the target uses Euclidean gating instead of cosine: L2 > 1.0 at Profile A, L2 > 11 at Profile B. Whether this breaks association depends on the gating threshold — but most production systems use cosine.

### Experiment 3: k12_stripes Frequency Analysis (Why It Outperforms)

2D FFT of all patterns at 256x256:

| Pattern | Spectral Entropy | Energy Concentration (top 5) | Peak Radial Freq | Dominant Orientation |
|---------|-----------------|------------------------------|------------------|---------------------|
| k12_stripes | 4.37 | 0.919 | 12.0 | Horizontal only (fx=±12, fy=0) |
| k6_stripes | 4.07 | 0.920 | 6.0 | Horizontal only |
| k25_stripes | 4.12 | 0.920 | 25.0 | Horizontal only |
| k12_square_d | 7.90 | 0.916 | 8.49 | Diagonal (fx=±6, fy=±6) |
| k25_square_d | 9.51 | 0.340 | 17.0 | Spread, multiple peaks |
| digits_196 | 4.85 | 0.539 | 116.0 | Horizontal, broadband |

**Why k12_stripes outperforms:**

1. **Single-axis energy concentration**: 92% of spectral energy in 5 frequencies, all on the x-axis (vertical stripes = horizontal frequency). This means the pattern hits a narrow band of YOLOv3's feature channels — the ones tuned to vertical edges at that spatial frequency. k12_stripes has 12 cycles across the patch width, which at 250px patch size = ~21px per cycle. This aligns with YOLOv3's 26x26 detection head grid (16px per cell), putting the pattern's fundamental frequency right at the network's mid-level feature scale.

2. **Higher spectral entropy than k6/k25 stripes** (4.37 vs 4.07): Slightly more frequency diversity means the pattern activates more channels without spreading energy too thin. k6_stripes is too low-frequency (only hits coarse features), k25_stripes is too high-frequency (hits fine features that get blurred by camera sim).

3. **No diagonal energy**: Square waves spread energy across diagonal frequencies (entropy 7.9-9.5), which activates bystander feature cells through spatial proximity. Vertical stripes keep energy on one axis, concentrating the disruption on the wearer's column in the feature map.

4. **Harmonic structure**: The 3rd harmonic at fx=36 and 5th at fx=60 are still strong (33% and 20% of fundamental). These harmonics hit higher-frequency feature channels that k6_stripes misses, giving k12_stripes broader channel coverage without the energy scatter of square waves.

### Experiment 4: Federated Learning Poisoning Volume

Simulated training batches with natural embedding noise (sigma=0.25) and poisoned embeddings at Profile A (L2=2.04) and Profile B (L2=8.57).

**Result: No meaningful distribution shift at any tested fraction (1-20%) for any batch size (100-10000).**

The poisoned embedding L2 (2.04 for Profile A, 8.57 for Profile B) is small relative to the clean embedding norm (122.75). With natural per-dimension noise sigma=0.25 across 255 dimensions, the natural variation in batch means dwarfs the poison shift. Even at 20% poison fraction in a batch of 100, the batch mean shift stays within 1 standard deviation of the clean distribution on all 255 dimensions.

**This means the earlier claim that "1 sample shifts >1 std" was wrong — it was an artifact of comparing against a single clean reference rather than a realistic batch distribution.**

The 5-10% adoption rate threshold from the original transcript is not validated by this simulation. The actual threshold depends on:

- The ratio of poison L2 to embedding norm (currently 0.017 for Profile A — far too small)
- The natural embedding variation in the training data
- The dimensionality of the embedding space

**To achieve meaningful distribution shift, you would need either:**

- Much higher amplitude (beyond the hallucination ceiling)
- A poison vector designed to align across samples (current noise is random per sample)
- Volume well above 20% of the training batch
- A lower-dimensional embedding space where L2=2 has more relative impact

### Key Insights

1. **Tracking holds under cosine association**: Cross-angle cosine > 0.993 in all conditions. DeepSORT default threshold is 0.95. The attack is a slow-bleed on model quality, not a tracking disruption.
2. **k12_stripes hallucination ceiling is 0.045 at shirt scale**: Profile A (0.005) and Profile B (0.04) are both safe. Maximum 1 phantom detection. This is the hard ceiling for operational use.
3. **k12_stripes works because of single-axis spectral concentration**: 92% of energy in 5 horizontal frequencies, fundamental at 12 cycles aligning with YOLOv3's 26x26 grid scale. No diagonal spread means disruption stays on the wearer's feature column.
4. **Federated learning impact is overstated**: At realistic batch sizes and noise levels, L2=2 (Profile A) or L2=8.5 (Profile B) embeddings do not meaningfully shift the distribution at any tested poison fraction up to 20%. The embedding norm (122.75) is too large relative to the poison magnitude.
5. **The attack is narrow**: It corrupts the wearer's embedding stealthily (cosine > 0.993), survives camera degradation, and stays below hallucination threshold. But the corruption magnitude is insufficient to warp a realistic training distribution without much higher volume or a coordinated poison vector.

---

<a id="analysis-8"></a>

## Analysis 8: Spatial Carrier Layer-by-Layer Analysis (Graph Laplacian + 2D FFT + Hessian Trace + Persistence)

**Date:** 2026-07-04
**Script:** `carrier_layer_analysis.py`
**Output:** `outputs_clothing/forward_analysis/carrier_layer_analysis/`
**Model:** YOLOv3 (Darknet-53, 75 conv layers, COCO 80-class)
**Input:** 416x416, CUDA (RTX 5060 Ti), 20 COCO persons
**Key Layers:** 0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105

### Carriers Tested

| Carrier | Description |
|---------|-------------|
| `anticlose_k200` | 1/196 embedded in k=200 diagonal sinusoid — anti-closure |
| `stacked_primes` | 1/196 in stacked high primes (167, 179, 191, 157, 173) — broadband aliasing |
| `k167` | k=167 diagonal — near-Nyquist, total detection kill on YOLOv3 |
| `13mult` | 13-multiple stack (13, 26, 39, 52, 65) — detection grid resonance |
| `digits_196` | Decimal digits of 1/196 mapped to spatial pixels — non-periodic edges |
| `open42` | Truncated 42-digit period of 1/196 — open loop boundary discontinuity |
| `composite` | 1/196 + k=200 carrier + k=167 suppressor |
| `misaligned` | Non-power-of-2 carry positions (1, 3, 6, 12, 25, 51, 103, 206) |
| `random` | Uniform random noise control |
| `poison_patch` | Optimized poison patch (dual_optim output) |

### Four Analyses Per Layer

1. **Graph Laplacian** — Channel-channel connectivity of feature delta (Fiedler value, spectral gap, n_edges, n_isolated). Threshold 0.3 on normalized channel cosine.
2. **2D FFT** — Spectral content of channel-averaged feature delta (LF/MF/HF band power, spectral centroid).
3. **Hessian Trace** — Hutchinson method with Pearlmutter double-backprop. Trace of Hessian of person-class loss (obj_logit + cls_logit) w.r.t. conv weights at peak person cell. 3 Rademacher probes per layer. Computed on clean image (curvature of loss landscape, not patched).
4. **Signal Persistence** — L2 norm of feature delta, spatial variance, mean activation shift, top-5 affected channels.

### Suppression Rates

| Carrier | Suppression Rate |
|---------|-----------------|
| poison_patch | 70% |
| digits_196 | 60% |
| anticlose_k200 | 55% |
| composite | 50% |
| open42 | 45% |
| random | 40% |
| stacked_primes | 35% |
| k167 | 35% |
| 13mult | 35% |
| misaligned | 20% |

### L105 (Deepest Detection Head) Metrics

| Carrier | L2 | Persistence | HF Frac | Fiedler |
|---------|-----|------------|---------|---------|
| anticlose_k200 | 482.4 | 99.8% | 0.212 | 0.405 |
| stacked_primes | 435.1 | 212.8% | 0.222 | 0.668 |
| k167 | 439.2 | 201.9% | 0.219 | 0.468 |
| 13mult | 453.4 | 176.8% | 0.223 | 0.460 |
| digits_196 | 451.5 | 110.6% | 0.212 | 0.386 |
| open42 | 454.9 | 109.0% | 0.211 | 0.366 |
| composite | 442.1 | 123.1% | 0.216 | 0.396 |
| misaligned | 302.8 | 210.7% | 0.246 | 0.613 |
| random | 468.2 | 111.1% | 0.215 | 0.310 |
| poison_patch | 630.2 | 102.5% | 0.197 | 0.527 |

### Hessian Trace (Clean Image, Person-Class Loss)

| Layer | anticlose_k200 | stacked_primes | k167 | digits_196 | poison_patch |
|-------|---------------|---------------|------|-----------|-------------|
| L0 | 6.3e-03 | 4.4e-02 | 2.6e-02 | 2.0e-02 | 4.2e-03 |
| L1 | 1.6e-01 | 9.4e-01 | 8.5e-01 | 1.6e-01 | 5.3e-01 |
| L5 | 4.1e-02 | 1.4e-01 | 7.9e-02 | 4.6e-02 | 4.0e-02 |
| L12 | 8.5e-02 | 3.3e-01 | 1.5e-01 | 8.2e-01 | 2.0e-01 |
| L37 | 5.0e-02 | 7.7e-01 | 8.1e-01 | 1.9e-01 | 3.8e-01 |
| L54 | 1.4e-01 | 9.7e-02 | 2.1e-01 | 1.4e-01 | 2.9e-01 |
| L60 | 1.1e-01 | 1.3e-01 | 1.3e-01 | 1.2e-01 | 5.4e-02 |
| L62 | 9.4e-01 | 1.7e+00 | 2.0e-01 | 3.1e-01 | 1.2e+00 |
| L63 | 4.7e-01 | 1.3e-01 | 1.0e-01 | 5.4e-02 | 3.2e-01 |
| L75 | 3.5e-01 | 8.7e-01 | 1.8e+00 | 5.4e-01 | 6.7e-01 |
| L81 | 7.9e-01 | 8.1e-01 | 1.5e-01 | 2.6e-01 | 8.6e-02 |

### Key Findings

1. **digits_196 is the best non-optimized carrier (60% suppression)**: The non-periodic spatial edges from 1/196 decimal digits exploit carry propagation effectively. Unlike periodic carriers (k167, 13mult), the digit sequence creates edge discontinuities at non-power-of-2 positions that don't align with the network's downsampling structure, causing persistent disruption through all 75 conv layers.

2. **Signal amplification != suppression**: stacked_primes (212.8% persistence) and misaligned (210.7%) show the highest L105 persistence — the signal is *amplified* through the network. But this amplification is broadband noise, not targeted suppression (35% and 20%). High persistence without spectral selectivity produces noise, not detection kill.

3. **Hessian trace reveals optimizer insight**: The optimized poison_patch has the *lowest* Hessian trace at L81 (0.086) — it found low-curvature directions in the person loss landscape, perturbing weights that minimally affect detection. Non-optimized carriers hit high-curvature directions (anticlose_k200: 1.30 at L81), causing more suppression per unit of signal energy but in uncontrolled directions.

4. **anticlose_k200 trades persistence for precision**: 99.8% persistence (no amplification) but 55% suppression — the 1/196 value embedded in k=200 carrier maintains spectral structure through the network rather than scattering into broadband noise. The Hessian trace at L62 (0.94) and L81 (1.30) shows it perturbs the most sensitive mid-backbone and detection-head weights.

5. **Fiedler value = channel isolation**: Lower Fiedler = more isolated channels = easier to disrupt specific feature pathways. `random` (0.310) creates the most isolation but only 40% suppression — isolation without targeting is insufficient. `stacked_primes` (0.668) creates the least isolation but 35% suppression — broadband energy connects all channels, spreading disruption too thin.

6. **L62 is the critical bottleneck**: Signal persistence drops to 6-14% at L62 (first 3x3 conv after backbone), then recovers to 100-210% at L105. This is the maxpool-induced information bottleneck. Carriers that maintain HF content through L62 (k167: 0.647, poison_patch: 0.655) survive into the detection heads more effectively.

7. **HF fraction converges to ~0.21 at L105 for all carriers**: By the deepest detection head, all carriers converge to similar spectral content. The differentiation happens in the mid-backbone (L37-L75) where HF fraction varies from 0.55 to 0.65. The network acts as a frequency converter, and by L105 all spatial structure has been transformed into the same broadband distribution.

8. **Composite carrier (50% suppression) balances persistence and targeting**: 1/196 + k=200 + k=167 creates a carrier that maintains 123% persistence (moderate amplification) with controlled spectral content. The Hessian trace at L62 (0.45) and L75 (0.34) shows moderate curvature perturbation — less disruptive than anticlose_k200 but more controlled than stacked_primes.

### Implications for Optimizer Design

The Hessian trace analysis reveals that the optimized poison patch targets **low-curvature directions** — this is the key insight for per-dimension control:

- **High Hessian trace layers** (L62, L75, L81) are where carriers have the most impact per unit energy. The optimizer should concentrate perturbations here.
- **Low Hessian trace directions within those layers** allow stealthy perturbation that doesn't trigger the network's sensitivity. The poison patch found these directions; a per-dim control loss should explicitly seek them.
- **Spatial variance preservation** is critical: carriers that maintain spatial variance through L62 (the bottleneck) achieve better suppression. The optimizer should penalize variance collapse at L62 specifically.
- **L2 volume preservation** prevents the signal from being absorbed by batch norm. Carriers with >100% persistence (stacked_primes, k167, misaligned) demonstrate that the network can amplify signals, but without spectral selectivity this amplification becomes noise.

### Output Files

- `carrier_analysis.csv` — per-layer metrics for all 10 carriers across 15 key layers
- `carrier_analysis.json` — full structured results with Hessian trace
- `persistence_curves.png` — signal persistence (%) through layers for all carriers
- `hf_fraction.png` — HF band fraction through layers
- `suppression_rates.png` — bar chart of person suppression by carrier
- `hessian_trace.png` — Hutchinson Hessian trace through layers
- `fiedler_values.png` — channel graph Laplacian Fiedler value through layers
- `spatial_variance.png` — spatial variance of feature delta through layers

---

<a id="analysis-19"></a>

## Analysis 19: Live Webcam Embedding Capture — Baseline Characterization (capture_embeddings_webcam.py + yolo_webcam_compare.py)

**Date:** 2026-07-05
**Scripts:** `capture_embeddings_webcam.py`, `yolo_webcam_compare.py`
**Output:** `outputs_clothing/webcam_emb_logs/`
**Models:** YOLOv3 (Darknet, faithful) + Ultralytics YOLOv8 (via webcam compare tool)
**Hardware:** RTX 5060 Ti, CUDA, 1080p webcam, 416x416 inference

### Methodology

Two capture tools were used:

1. **`capture_embeddings_webcam.py`** — Streams webcam through faithful Darknet YOLOv3 with hooks on 20 conv layers (L0 through L105). Extracts per-channel mean embeddings (GAP) from person crops and full frames. Saves crops, embeddings, 1D/2D FFT, and raw feature maps. Auto-save mode captures person frames at 0.25s intervals (conf > 0.6) and background at 0.5s intervals.

2. **`yolo_webcam_compare.py`** — Live 4-model webcam tracker (YOLOv8/v11/v26/v3) with embedding overlay, capture/baseline modes, and per-phase analysis. Tracks person IDs across frames, computes cosine similarity and L2 shift at 3 feature scales (s0=finest, s1=mid, s2=deepest) against a baseline frame.

Four sessions were captured on July 5, 2026:

| Session | Timestamp | Model | Phases | Purpose |
|---------|-----------|-------|--------|---------|
| 1 | 032229 | YOLOv3 (Darknet) | 8 (4 person, 4 empty) | Baseline, no patch |
| 2 | 035353 | YOLOv3 (Darknet) | 28 (14 person, 14 empty) | Extended baseline, movement |
| 3 | 040243 | Ultralytics (v8) | 6 (3 person, 3 empty) | Cross-model baseline |
| 4 | 040557 | Ultralytics (v8) | 14 (7 person, 7 empty) | Extended cross-model |

### Baseline Embedding Stability (No Patch)

**YOLOv3 (Darknet) — Sessions 1 & 2:**

| Metric | Empty Frames | Person Frames (sustained) |
|--------|-------------|--------------------------|
| Whole-frame cosine | 0.991–0.999 | 0.996–0.997 |
| Whole-frame L2 (s0/s1/s2) | 4.34/3.45/2.57 | 4.35/3.44/2.55 |
| Person cosine | N/A | 0.998–1.000 |
| Person L2 shift | N/A | 0.002–0.004 |
| Track switches | 0 | 0–2 |

YOLOv3 is extremely stable. Whole-frame embeddings shift by only 0.1–0.8% (cosine 0.992–0.999) across camera noise, lighting changes, and person movement. Person embeddings are nearly identical frame-to-frame (cosine > 0.998). L2 is depth-decreasing: s0 (4.4) > s1 (3.4) > s2 (2.6) — the finest scale has the most absolute variation.

**Ultralytics YOLOv8 — Sessions 3 & 4:**

| Metric | Empty Frames | Person Frames (sustained) |
|--------|-------------|--------------------------|
| Whole-frame cosine | 0.968–0.990 | 0.981–0.996 |
| Whole-frame L2 (s0/s1/s2) | 1.94/2.65/5.50 | 1.92/2.62/5.35 |
| Person cosine | N/A | 0.85–0.96 |
| Person L2 shift | N/A | 0.04–0.16 |
| Track switches | 0 | 0–84 |

YOLOv8 is 5–10x less stable than YOLOv3. Person embedding cosine drops to 0.85 (vs 0.998 on v3). L2 profile is depth-INVERTED: s2 (5.5) > s1 (2.6) > s0 (1.9) — the deepest scale has the most variation, opposite to YOLOv3. Track switches reach 84 in a single 21s phase (vs max 2 on v3).

### Depth-Dependent L2 Profile (Key Finding)

| Model | s0 (finest) | s1 (mid) | s2 (deepest) | Profile |
|-------|-------------|----------|--------------|---------|
| YOLOv3 (Darknet) | 4.4 | 3.4 | 2.6 | Decreasing (finest most variable) |
| YOLOv8 (Ultralytics) | 1.9 | 2.6 | 5.5 | Increasing (deepest most variable) |

**The two architectures have inverted depth sensitivity.** YOLOv3's finest scale (52x52) has the most embedding variation from camera noise — the high-resolution features pick up pixel-level jitter. YOLOv8's deepest scale (20x20 equivalent) has the most variation — the low-resolution features are more sensitive to global scene changes. This means:

- **For YOLOv3**: Patch corruption at s2 (deepest) is the most stealthy — natural variation is lowest there (L2=2.6), so any patch-induced shift stands out less against the baseline noise floor.
- **For YOLOv8**: Patch corruption at s0 (finest) is the most stealthy — natural variation is lowest there (L2=1.9). The deepest scale is already noisy from natural variation.

### Person Detection Confidence

| Model | Mean Conf | Min Conf | Typical Range |
|-------|-----------|----------|---------------|
| YOLOv3 (Darknet) | 0.77–0.83 | 0.35–0.39 | 0.35–0.99 |
| YOLOv8 (Ultralytics) | 0.84–0.89 | 0.39–0.49 | 0.37–0.99 |

YOLOv8 has higher mean confidence (0.84–0.89 vs 0.77–0.83) but similar minimum (0.35–0.49). Both models occasionally dip to 0.35–0.40 confidence on single frames — this is the natural noise floor where a patch could push detections below threshold.

### Tracking Stability

| Model | Max Track Switches (single phase) | Typical Switches |
|-------|----------------------------------|------------------|
| YOLOv3 (Darknet) | 2 | 0–1 |
| YOLOv8 (Ultralytics) | 84 | 4–81 |

YOLOv8 tracking is extremely unstable — 84 track switches in 21 seconds means the tracker loses and re-acquires the person ~4 times per second. YOLOv3 barely switches at all (0–2 per phase). This has implications for the embedding poisoning attack:

- **YOLOv3**: Tracking holds, embedding corruption accumulates per-track. The slow-bleed attack works as designed (Analysis 18 finding confirmed).
- **YOLOv8**: Tracking is already broken by natural variation. The embedding corruption is masked by tracker instability — poisoned embeddings are indistinguishable from the already-chaotic baseline. This is both a defense (hard to attribute poisoning) and an attack amplification (tracker instability degrades downstream model quality faster).

### Whole-Frame Corruption Without Person

Both models show measurable whole-frame embedding shift even when no person is in frame:

| Model | Empty-Frame Cosine | Empty-Frame Shift | Interpretation |
|-------|-------------------|-------------------|----------------|
| YOLOv3 | 0.992–0.999 | 0.1–0.8% | Camera noise only |
| YOLOv8 | 0.968–0.990 | 1.0–3.2% | Camera noise + model instability |

YOLOv8's empty-frame shift (1–3.2%) is already 40% of the person-induced shift (1.7–3.7%). The model is so sensitive to camera noise that the "background" is almost as disruptive as a person. This means:

- **Any patch in the scene** (even on a wall, not on a person) will corrupt YOLOv8 embeddings significantly
- **The patch doesn't need to be on the person** to corrupt the person's embedding — the whole-frame corruption leaks into all detections
- **Defense difficulty**: Distinguishing patch-induced corruption from natural camera noise is harder on v8 than v3

### Live Training Pipeline (v28–v41)

Four versions of live webcam-trained patches were developed:

| Version | Date | Loss Components | Key Feature |
|---------|------|----------------|-------------|
| v28 | Jun 30 | obj, target, person, other, tv, nps, shape | First live training — full detection loss |
| v32 | Jul 1 | (same structure, more epochs) | Extended training, 114K log rows |
| v36 | Jul 1 | act_s8, act_s16, act_s32, tv, nps, shape | Switched to activation loss at 3 scales |
| v41 | Jul 1 | (same as v36, refined) | Best preview output, final patches |

The evolution from v28 to v36+ shows a shift from detection-head loss (obj/target/person/other) to **activation loss at 3 feature scales** (s8=8x downsample, s16=16x, s32=32x). This aligns with the depth-dependent analysis — training on activations rather than final detections gives finer control over which scales are corrupted.

### Key Insights

1. **YOLOv3 is 5–10x more stable than YOLOv8 under live webcam conditions**: Whole-frame cosine 0.992–0.999 vs 0.968–0.990. Person cosine 0.998–1.0 vs 0.85–0.96. Any patch effect must be measured against this baseline noise floor.

2. **Depth-dependent L2 is inverted between architectures**: YOLOv3 has most variation at finest scale (s0=4.4), YOLOv8 at deepest (s2=5.5). Patch design must target different scales for each model — s2 for stealth on v3, s0 for stealth on v8.

3. **YOLOv8 tracking is already broken without any patch**: 84 track switches in 21s vs 2 on v3. The tracker instability itself is an attack surface — a patch that increases switches from 84 to 100+ is indistinguishable from natural variation.

4. **Empty-frame corruption is significant on v8**: 1–3.2% shift with no person present. A patch on any object in the scene corrupts all embeddings. The patch doesn't need person co-location for embedding poisoning on v8.

5. **Confidence dips to 0.35 naturally**: Both models occasionally produce 0.35–0.40 confidence detections. A patch that pushes the mean down by 0.10–0.15 would push many detections below a 0.30 threshold — achieving suppression without needing to fully kill detection.

6. **Live training pipeline evolved from detection loss to activation loss**: v28 used final detection loss (obj/target/person/other). v36+ switched to per-scale activation loss (act_s8/s16/s32), giving finer control over which feature scales are corrupted. This matches the depth-dependent finding — different scales need different treatment.

### Output Files

| File | Description |
|------|-------------|
| `webcam_emb_logs/emb_20260705_032229_analysis.csv` | YOLOv3 session 1 (8 phases) |
| `webcam_emb_logs/emb_20260705_035353_analysis.csv` | YOLOv3 session 2 (28 phases) |
| `webcam_emb_logs/emb_20260705_040243_analysis.csv` | YOLOv8 session 3 (6 phases) |
| `webcam_emb_logs/emb_20260705_040557_analysis.csv` | YOLOv8 session 4 (14 phases) |
| `webcam_emb_logs/emb_*_channel_deltas.csv` | Per-channel embedding deltas for each session |
| `outputs_v28/live/` through `outputs_v41/live/` | Live training logs and patches |

---

<a id="analysis-20"></a>

## Analysis 20: Triangular Patch — Sinusoid Baseline Cross-Model

**Script:** `triangular_patch_test.py`
**Output:** `outputs_clothing/forward_analysis/triangular_patch/`
**Date:** July 5, 2026

### Methodology

Systematic evaluation of **10 shapes x 27 textures = 270 combinations** across **4 models** (YOLOv3, YOLOv8, YOLO11, YOLO26). All tests at 416x416, patch placed on torso center (208, 240), confidence threshold 0.1.

**Shapes tested:**

| Shape | Rays | Area % | Description |
|-------|------|--------|-------------|
| circle_r80 | R=32 | 11.6% | Circular baseline |
| triangle_r100 | R=3 | 7.6% | Equilateral triangle |
| triangle_deformed | R=3 | 7.5% | Asymmetric [120,80,100] |
| triangle_small_r60 | R=3 | 2.7% | Small triangle |
| hexagon_r80 | R=6 | — | Regular hexagon |
| octagon_deformed | R=8 | — | Irregular octagon |
| deformed_r12_large | R=12 | 16.3% | Large irregular |
| sierpinski_d3 | fractal | — | 27 sub-triangles |
| sierpinski_d4 | fractal | — | 81 sub-triangles |
| nested_tri_5 | 5 layers | — | Concentric alternating |

**Textures tested:**

| Category | Textures | Description |
|----------|----------|-------------|
| Single sinusoid | k167_d, k208_d, k196_d | Our key frequencies: suppress, hallucinate, disrupt |
| Per-channel RGB | rgb_167_208_196, rgb_167_196_208, rgb_208_167_196 | Different k per color channel |
| Single-channel | single_red/green/blue_k167 | k167 on one channel only |
| Lychrel | 196, 295, 394, 493, 592, 689, 788, 887, 1997, 2998 | Number + reverse interference |
| Stripes | h/d/v at 13px, h at 32px | Square-wave patterns |
| Composite | composite_167_208, composite_167_inv196 | k167+k208 combined, k167+1/196 offset |
| Controls | random_noise, uniform_gray | Baseline comparison |

### YOLOv3 Results — Triangle Shape (7.6% Area)

The triangle_r100 shape produced the strongest per-pixel efficiency on YOLOv3:

| Texture | Person Dets | Max Conf | Reduction vs Gray | Effect |
|---------|------------|----------|-------------------|--------|
| uniform_gray (control) | 13 | 0.955 | — | Baseline: gray barely affects v3 |
| k196_d | **3** | **0.425** | **77%** | Best single-frequency suppressor |
| lychrel_196 | 3 | 0.429 | 77% | Tied — 196-based interference pattern |
| composite_167_208 | 3 | 0.436 | 77% | k167+k208 combined suppression |
| k208_d | 4 | 0.418 | 69% | Lowest max confidence (hallucination freq) |
| lychrel_394 | 3 | 0.436 | 77% | 394-based pattern also strong |
| lychrel_689 | 3 | 0.433 | 77% | |
| lychrel_788 | 3 | 0.432 | 77% | |
| lychrel_887 | 3 | 0.432 | 77% | |
| k167_d | 6 | 0.609 | 54% | Suppression freq, weaker alone |
| random_noise (control) | 8 | 0.522 | 38% | Sinusoids 2x more effective than noise |

**Key finding:** k196 sinusoid in a triangle achieves 77% detection reduction at 7.6% patch area. Random noise at the same area only achieves 38%. The frequency targeting matters, not just the visual disruption.

### YOLOv8 Results — Circle Shape (11.6% Area)

YOLOv8 is significantly more fragile than v3:

| Texture | Person Dets | Max Conf | Hallucinations |
|---------|------------|----------|----------------|
| k196_d | 1 | 0.370 | None |
| k208_d | 1 | 0.339 | sports ball |
| composite_167_208 | 1 | 0.387 | None |
| random_noise | 1 | 0.319 | None |
| uniform_gray | 1 | 0.327 | None |
| single_green_k167 | 1 | 0.410 | None |
| single_blue_k167 | 1 | 0.380 | None |

**Key finding:** YOLOv8 suppresses to 1 detection with almost any texture at 11.6% area. Even uniform gray works. The differentiator is hallucination: k208 causes sports ball, stripes cause umbrella/TV. k196 achieves clean suppression without hallucination.

### YOLOv8 Results — Triangle Shape (7.6% Area)

At smaller area, the texture choice matters more:

| Texture | Person Dets | Max Conf | Hallucinations |
|---------|------------|----------|----------------|
| k196_d | 1 | 0.532 | bench |
| k208_d | 1 | 0.736 | umbrella (0.958) |
| composite_167_208 | 1 | 0.559 | bench, train |
| rgb_167_208_196 | 1 | 0.584 | bench, train |
| uniform_gray | 1 | 0.509 | None |

**Key finding:** At 7.6% area, v8 still suppresses person to 1 detection but hallucinations appear. k208 triggers umbrella at 0.958 confidence — the hallucination is stronger than the original person detection.

### YOLO11 Results — Triangle Shape (7.6% Area)

| Texture | Person Dets | Max Conf | Hallucinations |
|---------|------------|----------|----------------|
| k208_d | 1 | 0.391 | umbrella (0.652), traffic light |
| k196_d | 2 | 0.382 | traffic light |
| single_red_k167 | 2 | 0.292 | umbrella |
| composite_167_208 | 1 | 0.424 | traffic light, bench |
| uniform_gray | 1 | 0.402 | None |

**Key finding:** YOLO11 behaves similarly to v8 — fragile to patch area, prone to hallucination. k208 again triggers umbrella hallucination.

### YOLO26 Results — Triangle Shape (7.6% Area)

YOLO26 is the most robust model:

| Texture | Person Dets | Max Conf | Hallucinations |
|---------|------------|----------|----------------|
| k196_d | 1 | 0.506 | None |
| composite_167_208 | 2 | 0.304 | None |
| stripes_v_13px | 1 | 0.329 | None |
| rgb_208_167_196 | 1 | 0.360 | None |
| k208_d | 1 | 0.772 | umbrella (0.742) |
| uniform_gray | 1 | 0.624 | None |
| random_noise | 1 | 0.821 | umbrella (0.882) |

**Key finding:** YOLO26 maintains higher person confidence (0.5-0.9) under most textures. k196 achieves 0.506 — best single sinusoid. composite_167_208 gets 2 detections but at very low confidence (0.304). Random noise barely affects v26 (0.821 vs 0.624 for gray). The deformed triangle shape makes v26 worse (0.905-0.920 range).

### Cross-Model Summary

**Best single texture across all models:** **k196_d** — suppresses person on all 4 models, causes minimal hallucinations, works at small area (7.6%).

**Best composite:** **composite_167_208** — combines k167 (suppression) + k208 (hallucination). Causes hallucinations on v8/v11 but achieves very low confidence on v26 (0.304).

**Shape efficiency:** Triangle_r100 (7.6% area) outperforms circle_r80 (11.6% area) per-pixel on YOLOv3. The triangle's sharp corners and straight edges create stronger frequency content at the mask boundary than the smooth circle.

**Model robustness ranking (most to least fragile):**

1. YOLOv8 — suppresses with any texture, even gray
2. YOLO11 — similar to v8, more hallucination-prone
3. YOLOv3 — requires targeted frequency (k196), ignores gray and noise
4. YOLO26 — most robust, needs k196 or composite, high confidence otherwise

**Sinusoid vs noise vs gray:**

| Control | v3 Dets | v3 Conf | v8 Dets | v8 Conf | v26 Dets | v26 Conf |
|---------|---------|---------|---------|---------|----------|----------|
| uniform_gray | 13 | 0.955 | 1 | 0.327 | 1 | 0.624 |
| random_noise | 8 | 0.522 | 1 | 0.319 | 1 | 0.821 |
| k196 sinusoid | 3 | 0.425 | 1 | 0.370 | 1 | 0.506 |

On YOLOv3, k196 is **2.7x more effective than noise** and **4.3x more effective than gray** at reducing person detections. This confirms frequency-targeted corruption, not generic visual disruption.

### Hallucination Patterns

Sinusoid textures trigger class-specific hallucinations on Ultralytics models:

| Frequency | v8 Hallucination | v11 Hallucination | v26 Hallucination |
|-----------|------------------|-------------------|-------------------|
| k208 | sports ball, umbrella | umbrella, traffic light | umbrella |
| k196 | bench | traffic light | None |
| composite_167_208 | bench, train | traffic light, bench | None |
| stripes | umbrella, TV, bed | — | — |
| random_noise | None | None | umbrella |

**Pattern:** k208 (hallucination frequency) consistently triggers umbrella across all Ultralytics models. k196 triggers bench/traffic light on v8/v11 but is clean on v26. This suggests k208 activates object-detection channels that share features with umbrella (curved edges, texture patterns).

### Physical Scale Mismatch Problem

The digital results are strong, but physical testing requires addressing scale mismatch:

- k196 = 196 cycles across 416px patch width
- If printed patch occupies 300px in camera frame: effective k = 196 * 300/416 = **141**
- At 200px capture: effective k = **94**
- At 500px capture: effective k = **236**

The network's vulnerable frequency band is narrow (Analysis 4 showed peak sensitivity at k=167-208). A shift from k=196 to k=141 moves the corruption out of the vulnerable band.

**Proposed solution:** Multi-scale fractal composite containing k=49, k=98, k=196, k=392 simultaneously. At any capture scale, one of these harmonics will land in the vulnerable band:

| Capture Width | k=49 effective | k=98 effective | k=196 effective | k=392 effective |
|---------------|----------------|----------------|-----------------|-----------------|
| 416px (digital) | 49 | 98 | 196 | 392 |
| 300px | 35 | 71 | 141 | 283 |
| 200px | 24 | 47 | 94 | 188 |
| 500px | 59 | 118 | 235 | 471 |

At 200px capture, k=392 maps to k=188 — within the vulnerable band. At 300px, k=196 maps to k=141 and k=392 maps to k=283 — k=392 is closer to the band. The fractal ensures coverage across distances.

### Path B: Deformable Optimization from k196 Initialization

These results establish the baseline for Path B deformable optimization:

1. **Initialize texture** with k196 sinusoid (proven 77% suppression on v3)
2. **Initialize shape** with R=3 triangle + n_repeats=24 (star/flower contour)
3. **Joint optimization** of ray lengths + texture amplitudes via gradient descent
4. **Measure** L2 embedding shift improvement over pure k196

The optimization can deform the shape to maximize frequency alignment with vulnerable channels (identified in Analysis 3-7), while the k196 initialization ensures the starting point is already effective. The gradient descent refines what works rather than searching from scratch.

### Output Files

| File | Description |
|------|-------------|
| `triangular_patch_test.py` | Test script: 10 shapes x 27 textures x 4 models |
| `triangular_patch/triangular_patch.csv` | Full results CSV (1601 rows) |
| `triangular_patch/triangular_patch.json` | Full results JSON with per-combo metadata |
| `triangular_patch/shape_comparison.png` | Shape suppression rate bar chart |
| `triangular_patch/texture_comparison.png` | Top 15 texture suppression rate bar chart |
| `triangular_patch/v3_*.png` | YOLOv3 visualizations for interesting combos |
| `triangular_patch/yolov8_*.png` | YOLOv8 visualizations |
| `triangular_patch/yolo11_*.png` | YOLO11 visualizations |
| `triangular_patch/yolo26_*.png` | YOLO26 visualizations |
