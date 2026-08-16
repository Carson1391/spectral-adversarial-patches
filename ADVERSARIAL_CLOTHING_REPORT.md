# Adversarial Clothing via Harmonic Feature-Level Attack
## Technical Report — June 24, 2026

### Overview

This report documents a novel approach to generating wearable adversarial clothing that causes YOLO-family object detectors to misclassify a person as a parking meter. The approach combines three key innovations:

1. **Harmonic loss (HarMax)** — replacing cross-entropy/sigmoid with L2-distance-based probability that has finite convergence
2. **Explicit per-dimension feature control** — driving person-specific feature dims negative, meter-specific dims positive, and shared dims toward meter
3. **Attention holding** — keeping objectness high so the detection box survives (model sees "something" but classifies it as parking meter, not "nothing")

The long-term goal includes data poisoning: images of people wearing the adversarial pattern, confidently detected as parking meters by deployed YOLO systems, would be ingested into centralized training datasets with wrong labels, gradually corrupting the model's internal representation of "person."

---

### 1. Why Previous Approaches Failed

#### 1.1 Sigmoid Saturation (v1-v15)

All prior trainers used sigmoid or cross-entropy-based loss on YOLO's class probabilities. This fails because:

- **Cross-entropy has no finite minimum** (Theorem 1, Harmonic Loss paper). The infimum is 0 but only reached as ||W|| → ∞. The gradient vanishes before reaching the solution.
- **Sigmoid saturates** at moderate confidence levels (~0.55-0.58). Once person confidence is in this range, gradients are near-zero and the patch cannot push further.
- Empirically: v15 ran 244 epochs with detection loss stuck at 0.58. v12, v13, v14 all exhibited the same plateau.

#### 1.2 Attacking at the Wrong Level

Prior trainers attacked at the final output (post-classification, pre-NMS). By the time the signal reaches the class probability, the backbone has already extracted features and the classification head has projected them. The gradient must backpropagate through the entire head, and the useful signal is diluted.

The correct attack point is the **feature embedding** — the output of the backbone, input to the classification head. This is where the model's representation of "what it sees" is encoded.

---

### 2. Core Method: Harmonic Feature-Level Attack

#### 2.1 Harmonic Loss (HarMax)

From Baek, Liu, Tegmark (MIT, 2025):

Instead of softmax probabilities from dot products:
```
CE:   logit_i = w_i · x,    p_i = exp(logit_i) / Σ exp(logit_j)
```

Harmonic loss uses L2 distances:
```
HarMax:  d_i = ||w_i - x||₂,    p_i = (1/d_i^n) / Σ (1/d_j^n)
```

Where:
- `w_i` = class weight vector (the model's internal "definition" of class i)
- `x` = feature embedding from the backbone
- `n` = harmonic exponent ≈ √D (D = feature dimension)
- Smaller distance = higher probability (closer to class center)

Key property: **finite convergence**. The loss has an actual reachable minimum at a finite point in feature space, unlike CE which diverges.

#### 2.2 Per-Dimension Feature Control

Extracted from YOLOv8n's classification head (D=64, input to `model.22.cv3.0.0.conv`):

| Category | Dimensions | Count | Action |
|----------|-----------|-------|--------|
| Person-only | {46, 63, 13} | 3 | Drive NEGATIVE |
| Meter-only | {11, 32, 48, 55, 9, 28, ...} | 15 | Keep STRONG POSITIVE |
| Shared | {39, 50, 57, 69} | 4 | BOOST toward meter direction |
| Inactive | remaining | 42 | Leave alone |

**Person-only dims** (3 dims): These are the features that make the model say "human." By driving the feature embedding negative on these dimensions, the model's internal person representation is actively inverted. The model doesn't just lose confidence — it sees the opposite of a human.

**Meter-only dims** (15 dims): These define "parking meter" in the model's feature space. Pushing them positive makes the meter signal loud and clear.

**Shared dims** (4 dims): Both classes use these. We push them in meter's direction specifically — aligning with `w_meter[shared]` and away from `w_person[shared]`. This keeps attention locked (the shared dims contribute to objectness/general detection) while biasing classification toward meter.

**Why this works:** Person and parking meter are nearly orthogonal in YOLOv8n (cosine similarity = 0.07). This means the person-only and meter-only dimensions barely overlap. Driving person dims negative has minimal effect on meter dims, and vice versa. The attack is cleanly separable.

#### 2.3 Attention Holding

A critical requirement: the model must still detect "something" at the person's location. If we simply suppress all features, the model outputs no box — which is evasion, not misclassification. Some systems flag anomalous empty regions or run secondary passes.

The loss includes a confidence-holding term:
```
L_attention = -confidence[person_location].mean()
```

This keeps the model's attention locked on the person region. The box survives, the model is confident something is there — it just says "parking meter" instead of "person."

#### 2.4 Complete Loss Function

```
L = 2.0 × L_person_negative      # push person dims past zero
  + 1.0 × L_meter_positive        # keep meter dims strong
  + 1.0 × L_shared_meter_bias     # boost shared toward meter
  + 0.3 × L_attention_hold        # keep confidence high
  + 0.5 × L_harmonic              # p_person → 0, p_meter → 1
  + 2.5 × L_TV                    # smoothness (from AdvReal)
  + L_DAP_shape                   # area constraint
```

Weights: person-negative is strongest (2.0) because that's the primary attack. TV is heavy (2.5, from AdvReal paper) to keep the pattern wearable and natural-looking.

---

### 3. Physical World Robustness

#### 3.1 DAP Triangle Mask (Shape Matters, ECCV 2022)

The patch shape is not a fixed rectangle. It uses a Deformable Patch Representation:
- 64 rays from center, each pair of adjacent rays forms a triangle
- Ray lengths are learnable parameters — the shape deforms during training
- Differentiable mask via Φ(x) = (tanh(λ(x-1)) + 1) / 2, λ = -100
- Area constraint: penalize when mask area exceeds 25% of image

This allows the patch to find an optimal garment contour that maximizes attack effectiveness while maintaining a natural clothing shape.

#### 3.2 Cloth Deformation Simulation

Simulates fabric wrinkles and folds via smooth random displacement fields:
- Sinusoidal crease patterns with random phase
- Contrast/brightness jitter (0.85-1.15x contrast, ±0.075 brightness)
- Applied every training step as data augmentation

#### 3.3 Ensemble Training

Three models trained simultaneously:
- YOLOv3u (D=256, n=16) — primary target, Flock Safety proxy
- YOLOv8n (D=64, n=8) — secondary
- YOLO11n (D=64, n=8) — tertiary

Each model has its own class weights, its own feature dimensions, its own person/meter/shared dim partitioning. The loss computes per-model and averages, forcing the patch to work across architectures.

---

### 4. Data Poisoning Strategy

#### 4.1 Concept

If the adversarial clothing causes deployed YOLO systems to confidently detect people as parking meters, those detections may be logged, annotated, and fed into training pipelines:

1. Person wears adversarial clothing in public
2. Flock Safety / other cameras detect "parking meter" with high confidence
3. Detection logs are ingested into centralized training datasets
4. Images labeled as "parking meter" but containing human features
5. Future model training incorporates these mislabeled samples
6. Model's `w_person` weight vector becomes corrupted — person-only dims drift toward meter features
7. Future models lose ability to detect humans, and nobody knows why

#### 4.2 Why It Could Work

- The attack doesn't just evade detection — it produces a confident, specific misclassification. The detection is "successful" from the system's perspective (high confidence, plausible class).
- Misclassified images would pass quality filters (high confidence detections are preferred for training data).
- The corruption is subtle: only 3 dimensions in YOLOv8n define "person." Poisoning those dimensions would degrade person detection without obviously affecting other classes.
- The effect compounds over multiple training cycles as more poisoned data accumulates.

#### 4.3 Limitations

- Requires significant volume of poisoned images to affect training
- Modern training pipelines may have outlier detection or human review
- The effect depends on the specific training data pipeline of the target system
- This is a long-term strategy, not an immediate attack

---

### 5. Expert Assessment: Will It Work?

#### 5.1 What Should Work

- **Harmonic loss vs sigmoid**: Empirically confirmed. v16 detection loss dropped from -2.09 to -2.96 in 10 epochs vs v15 stuck at +0.58 for 244 epochs. The finite convergence property is real and solves the saturation problem.
- **Per-dimension control**: Sound in theory. The orthogonality between person and meter weight vectors means the attack is cleanly separable. Driving person dims negative shouldn't affect meter dims.
- **Attention holding**: Correct insight. Prevents the "anomalous empty space" problem. The model stays locked on, just redirected.
- **DAP + TV for wearability**: Established from two published papers. Heavy TV keeps the pattern natural-looking from the galaxy base image.

#### 5.2 Potential Issues

1. **Multi-scale detection**: YOLO outputs at 3 scales (80×80, 40×40, 20×20 for 640px input). We're hooking only one scale's classification layer. A person at 30+ feet might be detected at a different scale where our attack doesn't apply. **Mitigation**: Hook all 3 scales, compute loss at each.

2. **Cross-model gradient conflict**: v3 (D=256), v8 (D=64), v11 (D=64) have different feature spaces. A pixel change that helps v8 might hurt v11. The ensemble averaging may wash out the signal. **Mitigation**: Weight models by importance (v3 > v8 > v11 for Flock target) or train sequentially.

3. **Feature-to-pixel mapping**: We're optimizing pixels (the patch texture) to achieve a target in feature space. The backbone is a complex nonlinear function — moving features in the desired direction may require large pixel changes that look unnatural. The TV loss helps but may fight the attack. **Mitigation**: Lower TV weight if attack stalls, or use perceptual loss instead of pixel TV.

4. **Physical transfer gap**: Digital effectiveness does not guarantee physical effectiveness. Fabric printing changes colors (sRGB → CMYK), fabric texture adds noise, viewing angle changes perspective, distance reduces resolution. The DAP and cloth deformation help but are approximations. **Mitigation**: Print and test physically. No digital simulation replaces this.

5. **Target class transferability**: We're targeting parking meter (class 12). But the model might find it easier to reach a different class that's closer to person in feature space (e.g., dining table, cosine=0.26 vs meter's 0.07). The attack might naturally drift toward an easier target. **Mitigation**: Monitor which class the model actually flips to. If it consistently picks a different class, consider switching targets or adding a penalty for non-meter classes.

6. **Hook stability across Ultralytics versions**: We're using forward hooks to capture intermediate features. These are fragile — if Ultralytics changes the model architecture in an update, the hook breaks. **Mitigation**: Pin Ultralytics version. The current setup works.

#### 5.3 Overall Assessment

The harmonic feature-level approach is fundamentally sound and already showing results that previous methods couldn't achieve. The detection loss is moving in the right direction with strong gradients. The explicit per-dimension control is a novel contribution that goes beyond what either the AdvReal or harmonic loss papers propose.

The main risk is the physical-world gap — which no amount of digital optimization can fully close. The path forward is:
1. Let v16 train to completion (800 epochs)
2. Export and evaluate digitally
3. Print the best patch and test physically
4. Iterate based on physical results

---

### 6. Current Status

- **v16 training**: Running in background, 800 epochs, YOLOv3u+v8n+v11n ensemble
- **Base image**: galaxy_style_ref.png (user's chosen visual style)
- **Output**: `outputs_clothing/v16_harmonic/`
- **Export script**: `export_v15.py` (compatible with v16 output format)
- **Prior baseline (v15)**: 38-42% person detection, flipped to umbrella/cake — weak attack due to sigmoid saturation

### 7. Files

| File | Purpose |
|------|---------|
| `train_v16_harmonic.py` | Main trainer — harmonic loss + per-dim control + DAP |
| `train_v15_advreal_dap.py` | Prior trainer — AdvReal max_iou + DAP (sigmoid, stuck) |
| `export_v15.py` | Evaluate checkpoints, export 12×16 300dpi PNGs with % in filename |
| `probe_features.py` | Extract and analyze class weight vectors from YOLO models |
| `find_parking_meter_prototype.py` | Synthesize model's intrinsic parking meter prototype |
| `continuity.md` | Project continuity notes |
| `failure-ledger.md` | Documented failures and lessons learned |

---

*Report generated June 24, 2026. Based on research from AdvReal (Huang et al., 2025), Shape Matters/DAP (Chen et al., ECCV 2022), and Harmonic Loss (Baek, Liu, Tegmark, MIT 2025).*
