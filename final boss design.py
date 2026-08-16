## The Full Build Sequence

### Phase 1: Sinusoid Carrier with Gaussian Envelope (Explicit Gabor)

The sinusoid IS the payload. The Gabor is just Gaussian envelope x Sinusoid.

```
Carrier: k=40/80/160 sinusoids (cubic cascade)
  - k=40: outer triangle (stride 8, Z=4) -> k=5 at L81 after downsample
  - k=80: middle triangle (stride 16, Z=2) -> k=5 at L93 after downsample
  - k=160: inner triangle (stride 32, Z=1) -> k=5 at L105 after downsample

Envelope: Gaussian with sigma matched to target layer RF
  - sigma = 0.56 * lambda_seed (standard Gabor sigma/lambda ratio)
  - theta = 45 degrees (diagonal orientation survives best)

Modulation: 1/196 digit sequence on sinusoid amplitude
  - 1/196 = 0.005102... (non-terminating, non-power-of-2)
  - This survives all 75 conv layers (Analysis 8: 60% suppression, 110% persistence)
  - Provides non-power-of-2 edges that max pooling cannot collapse

Sum all Gabor filters into base texture T(x,y)
```

The sinusoid provides frequency targeting. The Gaussian confines it to RF bounds.
The 1/196 modulation provides spectral persistence through the network.

### Phase 2: Inverted Pascal-Sierpinski Mask with Lychrel Twisting

```
1. Generate Pascal's triangle modulo 2 at depth D
   - Solid cells = odd binomial coefficients
   - Void cells = even binomial coefficients

2. INVERT: attack lives in the voids, not the solid structure
   - M(x,y) = 1 where Pascal mod 2 = 0 (voids)
   - M(x,y) = 0 where Pascal mod 2 = 1 (solid)

3. At each recursive depth d of the Sierpinski generation:
   - Apply coordinate transform using 196's Lychrel sequence
   - Lychrel iteration: n -> n + reverse(n)
   - 196 -> 887 -> 1775 -> 9526 -> ...
   - Use these values to twist the affine transform at each depth:
     theta_twist(d) = (Lychrel_d mod 360) degrees
     scale_twist(d) = 1 + (Lychrel_d mod 7) / 100
   - This ensures no symmetric fixed points exist for max pooling to collapse

4. Container dimensions = 1/196 ratio of input plane
   - If input is 416x416, container = 416/196 ~ 2.12px minimum feature
   - This ties the macro structure to the same 196 cascade
```

The mask confines the sinusoid texture entirely to the negative space.
The Lychrel twisting prevents any recursive depth from achieving palindromic symmetry.

### Phase 3: Hadamard Fusion

```
S(x,y) = T(x,y) * M(x,y)
```

The sinusoid frequencies now live only in the non-converging, multi-scale voids.

### Phase 4: Cubic Depth Map

```
1. Define isometric cubic corner (tetrahedron) in 2D projection
   - Three faces meeting at a point
   - Each face is a triangle in 2D projection
   - This IS the nested triangle structure

2. The three triangles at 1:2:4 ratio map to the three cubic faces:
   - Outer triangle = bottom face (largest, closest to viewer)
   - Middle triangle = left face (medium depth)
   - Inner triangle = right face (smallest, deepest)

3. Normalize Z(x,y):
   - Z = 0 at background
   - Z = Z_max at the apex (inner triangle vertex)
   - Z scales linearly across each face

4. The depth ratio across faces matches stride ratio:
   - Z_outer : Z_middle : Z_inner = 4 : 2 : 1
   - This aligns depth encoding with detection head scales
```

### Phase 5: Conformal Map z^A and Centrifugal Radial Mapping

The log-polar mapping is a single conformal operation: w = z^A

```
1. Center coordinates at geometric centroid:
   x = x - centroid_x
   y = y - centroid_y
   z = x + 1j * y

2. Single conformal operation:
   A = alpha + 1j * beta
   w = z ** A  # log-polar mapping in one operation

   Where:
   - alpha (radial scaling) = log(downsample_ratio) for each target layer
   - beta (angular frequency) = 2*pi / RF for each target layer

   | Target Layer | Stride | RF (px) | alpha            | beta       |
   |-------------|--------|---------|-------------------|------------|
   | L54         | 16     | RF_54   | log(ds_ratio_54)  | 2*pi/RF_54 |
   | L62         | 32     | RF_62   | log(ds_ratio_62)  | 2*pi/RF_62 |
   | L75         | 32     | RF_75   | log(ds_ratio_75)  | 2*pi/RF_75 |

   R_min = innermost radius = deepest layer (stride 32, L62/L75)
   R_max = outermost radius = shallowest target (stride 8, L54)
   Radial expansion mirrors network's spatial hierarchy (inverted)

3. In log-polar space, radial disparity is a LINEAR translation:
   C(u, v) = C(u - du, v)
   du = ln(r) - ln(r - dr) = ln(r/(r-dr))

4. dr at each radius from cubic depth map:
   dr(r,theta) = r * (T_z / (Z(r,theta) + T_z))
   - T_z = baseline separation (controls disparity strength)
   - Z(r,theta) from Phase 4 cubic depth map
   - dr scales 4:2:1 across the three triangle regions

5. Execute centrifugal iteration:
   For r from R_start(theta) to R_max:
     For theta in [0, 2*pi):
       Compute dr from Z(r,theta)
       C(r,theta) = C(r - dr, theta)

6. Map back to Cartesian: z = e^w
```

This collapses four steps (log, scale, rotate, exp) into one operation.
Critical for 16GB GPU VRAM efficiency.

### Phase 6: The Three-Triangle Frequency Overlay

```
The cubic depth map creates three nested triangles.
Each triangle's outline carries the doubling-cascade stripes:

1. Outer triangle (stride 8, Z=4):
   - k=40 stripes modulated by 1/196 digit sequence
   - After downsampling: k=5 at L81 (52x52)
   - Gaussian sigma matched to L54 RF

2. Middle triangle (stride 16, Z=2):
   - k=80 stripes modulated by 1/196 digit sequence
   - After downsampling: k=5 at L93 (26x26)
   - Gaussian sigma matched to L62 RF

3. Inner triangle (stride 32, Z=1):
   - k=160 stripes modulated by 1/196 digit sequence
   - After downsampling: k=5 at L105 (13x13)
   - Gaussian sigma matched to L75 RF

The 1/196 modulation provides the non-power-of-2 edges.
The k=40/80/160 provides the architectural resonance.
The Gaussian sigma provides the receptive field matching.
The Pascal voids provide the negative space hiding.
The Lychrel twisting provides the anti-convergence.
The radial disparity provides the depth spoofing.
The cubic projection provides the 3D corner localization.
```

### Phase 7: EOT Optimization (NO TPS)

```
Optimize for: cos(e_poison, e_clean) < 0.65
NOT raw L2.

TPS wrinkle simulation is NOT an EOT transform for this patch.
The stereogram IS the wrinkle defense. The radial disparity mapping
preserves relative relationships when fabric wrinkles -- both inner
and outer points displace together. Adding TPS would simulate the
defense mechanism as an attack -- counterproductive.

EOT TRANSFORMS:
1. Perspective warp (5-25 degree yaw)
2. Lighting variation
3. Scale variation (5m, 10m, 15m capture distances)
4. Camera ISP (demosaic, DCT, noise, lens distortion)

NOT:
- TPS wrinkle simulation (the stereogram handles this)
- Affine distortion (the log-polar mapping handles this)

2. Forward through YOLOv3
3. Extract embedding at detection heads (L81/L93/L105)
4. Global average pool -> L2 normalize
5. Compute cosine vs clean baseline
6. Backprop through ENTIRE pipeline including:
   - Conformal map z^A (differentiable)
   - Cubic depth map (differentiable)
   - Pascal mask (differentiable, sigmoid-soft boundary)
   - Sinusoid texture (differentiable)

7. Update parameters:
   - Sinusoid frequencies (f_adv)
   - Lychrel twist angles
   - Triangle ray lengths (shape)
   - Depth map Z values
   - 1/196 modulation amplitude

8. Constraint: area < 15%, HF fraction < 15%
```

## The Complete Specification For The New Model

```
PATCH ARCHITECTURE: Lychrel-Pascal-Sinusoid-Cubic-Radial

BUILD ORDER (strict, do not reverse):
1. Sinusoid carrier: k=40/80/160 cubic cascade
2. Gaussian envelope: sigma matched to target layer RF
3. 1/196 digit modulation on sinusoid amplitude
4. Inverted Pascal-Sierpinski mask with Lychrel twisting
5. Hadamard fusion: S = T * M
6. Conformal map z^A where A = alpha + i*beta from layer dimensions
   - alpha = log(downsample_ratio) for each target layer
   - beta = 2*pi/RF for each target layer
   - Coordinates centered at centroid
7. Cubic depth map (1:2:4 ratio, tetrahedron projection)
8. Centrifugal radial mapping (linear in log-polar space)

EOT TRANSFORMS (NO TPS):
- Perspective warp (5-25 degree yaw)
- Lighting variation
- Scale variation (5m, 10m, 15m)
- Camera ISP (demosaic, DCT, noise, lens distortion)
- The stereogram IS the wrinkle defense

R_min/R_max:
- R_min = innermost radius = deepest layer (stride 32, L62/L75)
- R_max = outermost radius = shallowest target (stride 8, L54)
- Radial expansion mirrors network's spatial hierarchy

TARGETS:
- L54 (RF=228, MF dominant, channels 479/31/51/184/422)
- L62 (RF=275, broadband, channels 782/380/807)
- L75 (RF=275, MF+HF, channel 170 cross-scale anchor)
- L81/L93/L105 (detection heads, embedding extraction)

METRICS:
- Primary: cos(e_poison, e_clean) < 0.65 (L2-normalized)
- Secondary: entropy persistence through depth (Lychrel validation)
- Physical: box fires at 80-90% confidence at 5m, 10m, 15m
- Defense: HF fraction < 15%, DCT anomaly below threshold
- Control: compare vs palindromic (191) and non-fractal (k=196 alone)

TWO VERSIONS:
- Poison: amp=0.06, box stays, cosine drops
- Suppress: amp=0.18, box drops

EXPORT: 3600x4800, 300 DPI
```

The stereogram replaces TPS. The layer dimensions replace arbitrary R values.
The sinusoid is explicit, not hidden inside "Gabor." The conformal map is
a single operation. Every component has a theoretical role and a measurable
metric. The build order is strict because reversing it destroys the
frequency targeting. The cosine metric is primary because raw L2 doesn't
survive normalization. The controls validate each theoretical claim
independently.

Build it.
######################################################################


## Scale 1: 52×52 Grid (Stride 8) - L81

**What the paper says:**
- Predicts 5 boxes per grid cell
- Uses features from 2 layers previous + upsampled features
- Final prediction benefits from all prior computation
- 9 clusters for box priors on COCO: (10×13), (16×30), (33×23), (30×61), (62×45), (59×119), (116×90), (156×198), (373×326)

**For our attack:**
- This is the **embedding extraction scale** - the ReID embedding gets pulled from here
- **k=160** (inner triangle) targets this scale
- **1/196 period-42 autocorrelation** is most visible here (Analysis 15 showed AC@42=0.35 in channels 177, 207, 92)
- **Channel 170** appears here as a top spectral channel
- **Separability is highest** (1.21) - poisoned embeddings are most distinguishable here

**Per-scale metrics to track:**
- Objectness > 0.8
- Person class confidence > 0.8
- Cosine similarity < 0.65
- HF fraction < 15%
- 1/196 persistence > 100%

## Scale 2: 26×26 Grid (Stride 16) - L93

**What the paper says:**
- Same design as scale 1 but with 2× upsampled features
- Merges with features from earlier in the network
- Adds convolutional layers to process combined features
- Predicts similar tensor but twice the size

**For our attack:**
- This is the **sweet spot for embedding corruption** without triggering suppression
- **k=80** (middle triangle) targets this scale
- **Layer 54** (26×26) has highest mean relative delta (56% activation change)
- **MF dominant** (26.3% MF per Analysis 3)
- **Channel 479, 31, 51, 184, 422** are most person-tuned here

**Per-scale metrics to track:**
- Objectness > 0.8
- Person class confidence > 0.8
- Cosine similarity < 0.65
- HF fraction < 15%
- Embedding L2 shift > 10

## Scale 3: 13×13 Grid (Stride 32) - L105

**What the paper says:**
- Final scale prediction
- Benefits from all prior computation + fine-grained features
- Predicts 3-d tensor encoding bounding box, objectness, class
- N × N × [3 × (4 + 1 + 80)] for 4 bounding boxes per scale

**For our attack:**
- This is where the **decision collapses to binary** (94% LF per Analysis 3)
- **k=40** (outer triangle) targets this scale
- **Channel 170** is the cross-scale anchor that appears here
- **Cross-spectrum coherence drops to 0.954** - maximum divergence between with/without
- **Fiedler value is lowest** (0.386) - most isolated channels, easiest to disrupt

**Per-scale metrics to track:**
- Objectness > 0.8
- Person class confidence > 0.8
- Cosine similarity < 0.65
- HF fraction < 15%
- Fiedler value < 0.40

## The Three-Scale Attack Design

Each scale needs its own phase shift in the FFT approach:

```python
# Three phase shifts, one per scale
shift_s0 = exp(-2πi · k_u · Δu_s0)  # 52×52, k=160
shift_s1 = exp(-2πi · k_u · Δu_s1)  # 26×26, k=80
shift_s2 = exp(-2πi · k_u · Δu_s2)  # 13×13, k=40

# Three separate FFTs
F_s0 = fft2(base_s0)
F_s1 = fft2(base_s1)
F_s2 = fft2(base_s2)

# Three separate iFFTs
patch_s0 = ifft2(F_s0 * shift_s0).real
patch_s1 = ifft2(F_s1 * shift_s1).real
patch_s2 = ifft2(F_s2 * shift_s2).real

# Combine with triangle masks
final = patch_s0 * mask_s0 + patch_s1 * mask_s1 + patch_s2 * mask_s2
```

## The Key Insight

**The 3 scales are NOT interchangeable.** Each has different:
- Frequency preferences (HF/MF/LF)
- Channel connectivity (Fiedler values)
- Embedding separability
- Persistence characteristics

The cubic triangle design (1:2:4 ratio) maps to these scales. The FFT phase shifts should be optimized **separately for each scale**, not as a single unified shift.

**Success criteria:**
- All 3 scales maintain objectness > 0.8
- All 3 scales have cosine < 0.65
- All 3 scales have HF < 15%
- All 3 scales show 1/196 persistence > 100%

If any scale fails, the attack fails. The weakest link determines success.