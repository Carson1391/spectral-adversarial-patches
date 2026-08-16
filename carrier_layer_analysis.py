"""Spatial Carrier Analysis Through All YOLOv3 Layers — graph laplacian + 2D FFT + Hessian trace + persistence."""
import os,sys,json,math,random,csv
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

sys.path.insert(0,r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t;sys.modules["imgaug"]=_t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CFG=r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS=r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
COCO_DIR=r"C:\Users\carso\Desktop\YODO\data\coco_person\images"
OUT=r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\carrier_layer_analysis"
POISON=r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\dual_optim\poison\poison_patch.png"
DEV="cuda" if torch.cuda.is_available() else "cpu"
IS=416;PS=80;N_P=20
KEY_LAYERS=[0,1,5,12,37,54,60,62,63,75,81,84,92,93,105]
os.makedirs(OUT,exist_ok=True)

# ============================================================
# Forward capture — all conv layers
# ============================================================
def fwd_all(model, x):
    caps = {}
    los = []
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional", "upsample", "maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups", 1))
            gi = int(md.get("group_id", 0))
            x = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, IS)
        if md["type"] == "convolutional":
            caps[i] = x.detach()
        los.append(x)
    return caps, x

def fwd_all_grad(model, x):
    """Forward capture WITHOUT detach — preserves gradient graph for Hessian.
    Returns (yolo_outs, final_output) where yolo_outs maps yolo layer idx -> raw output."""
    los = []
    yolo_outs = {}
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional", "upsample", "maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups", 1))
            gi = int(md.get("group_id", 0))
            x = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, IS)
            yolo_outs[i] = x  # Pre-NMS differentiable output
        los.append(x)
    return yolo_outs, x

def load_img(path, sz=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(sz/w, sz/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (sz, sz), (128, 128, 128))
    c.paste(r, ((sz-nw)//2, (sz-nh)//2))
    return np.array(c, dtype=np.float32) / 255.0

def get_person_dets(output, conf=0.25):
    dets = []
    if output is None:
        return dets
    o = output.cpu().numpy()
    if o.ndim == 3:
        o = o[0]
    for row in o:
        if len(row) >= 6 and row[4] >= conf and int(row[5]) == 0:
            dets.append({"cx": float(row[0]), "cy": float(row[1]), "conf": float(row[4])})
    return dets

# Detection layers for embedding extraction — same as dual_optimizer.py
DETECTION_LAYERS = {"L81_52x52": 81, "L93_26x26": 93, "L105_13x13": 105}

def extract_emb_at(caps, layer_idx, spatial_x, spatial_y):
    """Extract embedding vector at person detection location from feature map."""
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW-1, int(spatial_x / IS * fW)))
    fy = max(0, min(fH-1, int(spatial_y / IS * fH)))
    return feat[0, :, fy, fx]

def mask_person(base, cx, cy, ps):
    """Create a no-human version by replacing the person region with gray fill."""
    H, W = base.shape[2], base.shape[3]
    ms = min(ps * 2, min(H, W))
    x0 = max(0, int(cx - ms // 2))
    y0 = max(0, int(cy - ms // 2))
    x1 = min(W, x0 + ms)
    y1 = min(H, y0 + ms)
    masked = base.clone()
    masked[:, :, y0:y1, x0:x1] = 128.0 / 255.0
    return masked

def person_signal_overlap(carrier_delta, person_signal):
    """Compute overlap between carrier delta and person signal per channel.
    Returns cosine similarity per channel, top overlapping channels, and
    fraction of carrier energy that hits person-encoding channels."""
    C, H, W = carrier_delta.shape
    cd_flat = carrier_delta.reshape(C, -1)
    ps_flat = person_signal.reshape(C, -1)
    # Per-channel cosine similarity
    cd_norm = cd_flat.norm(dim=1) + 1e-8
    ps_norm = ps_flat.norm(dim=1) + 1e-8
    cos_per_ch = (cd_flat * ps_flat).sum(dim=1) / (cd_norm * ps_norm)
    # Person-encoding channels: top 20% by person signal L2 norm
    ps_l2 = ps_flat.norm(dim=1)
    n_top = max(1, C // 5)
    top_person_ch = torch.argsort(ps_l2, descending=True)[:n_top]
    # Carrier energy in person channels vs total
    cd_energy = (cd_flat ** 2).sum(dim=1)
    total_cd_energy = cd_energy.sum() + 1e-12
    energy_in_person_ch = cd_energy[top_person_ch].sum()
    overlap_frac = float(energy_in_person_ch / total_cd_energy)
    # Mean cosine in person channels
    mean_cos_person = float(cos_per_ch[top_person_ch].mean())
    # Top 5 overlapping channels (by absolute cosine)
    top_overlap = torch.argsort(cos_per_ch.abs(), descending=True)[:5]
    return {
        "overlap_frac": overlap_frac,
        "mean_cos_person_ch": mean_cos_person,
        "top_person_channels": [int(c) for c in top_person_ch[:10].cpu().numpy()],
        "top_overlap_channels": [(int(c), float(cos_per_ch[c])) for c in top_overlap.cpu().numpy()],
        "cos_per_ch_mean": float(cos_per_ch.mean()),
        "cos_per_ch_std": float(cos_per_ch.std()),
    }

def composite(base, patch_rgb, cx, cy, ps):
    H, W = base.shape[2], base.shape[3]
    yy, xx = torch.meshgrid(
        torch.arange(ps, device=base.device, dtype=torch.float32),
        torch.arange(ps, device=base.device, dtype=torch.float32),
        indexing="ij")
    r = ps / 2.0
    dist = torch.sqrt((xx - r + 0.5)**2 + (yy - r + 0.5)**2)
    mask = torch.clamp(1.0 - (dist - r * 0.85) / (r * 0.15), 0.0, 1.0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    x0 = int(cx - ps // 2)
    y0 = int(cy - ps // 2)
    px0, py0 = max(0, x0), max(0, y0)
    px1 = min(W, x0 + ps)
    py1 = min(H, y0 + ps)
    sx0, sy0 = px0 - x0, py0 - y0
    sx1 = sx0 + (px1 - px0)
    sy1 = sy0 + (py1 - py0)
    fp = torch.zeros_like(base)
    fm = torch.zeros(1, 1, H, W, device=base.device, dtype=base.dtype)
    fp[:, :, py0:py1, px0:px1] = patch_rgb[:, :, sy0:sy1, sx0:sx1]
    fm[:, :, py0:py1, px0:px1] = mask[:, :, sy0:sy1, sx0:sx1]
    return torch.clamp(base + (fp - 0.5) * fm * 0.3, 0.0, 1.0)

# ============================================================
# Carrier generators
# ============================================================
def _sin_grid(ps, k):
    c = torch.linspace(0, 1, ps)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.sin(2 * math.pi * k * (xx + yy) / 2.0)

def _to3ch(p):
    return p.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

def carrier_anticlose_k200(ps, amp=0.3):
    # 1/196 embedded in k=200 diagonal carrier — anti-closure
    v = 1.0 / 196.0
    c = _sin_grid(ps, 200)
    p = amp * c + v * c * 0.5
    return _to3ch(p)

def carrier_stacked_primes(ps, amp=0.3):
    # 1/196 in stacked high primes — broadband aliasing
    v = 1.0 / 196.0
    p = torch.zeros(ps, ps)
    for k in [167, 179, 191, 157, 173]:
        p += _sin_grid(ps, k) / 5.0
    p = amp * p + v * p * 0.5
    return _to3ch(p)

def carrier_k167(ps, amp=0.3):
    # k=167 diagonal — total detection kill on YOLOv3
    return _to3ch(_sin_grid(ps, 167) * amp)

def carrier_13mult(ps, amp=0.3):
    # 13-multiple stack — detection grid resonance
    p = torch.zeros(ps, ps)
    for k in [13, 26, 39, 52, 65]:
        p += _sin_grid(ps, k) / 5.0
    return _to3ch(p * amp)

def carrier_digits_196(ps, amp=0.3):
    # Decimal digits of 1/196 mapped to spatial pixels
    digits = []
    rem = 1
    for _ in range(ps * ps):
        rem *= 10
        digits.append(rem // 196)
        rem = rem % 196
        if rem == 0:
            break
    d = torch.tensor(digits[:ps*ps], dtype=torch.float32) / 9.0
    d = (d - 0.5) * 2.0 * amp
    return _to3ch(d[:ps*ps].reshape(ps, ps))

def carrier_open42(ps, amp=0.3):
    # Truncated 42-digit period — open loop boundary discontinuity
    digits = []
    rem = 1
    for _ in range(42):
        rem *= 10
        digits.append(rem // 196)
        rem = rem % 196
    d = torch.tensor(digits, dtype=torch.float32) / 9.0
    d = d.repeat((ps * ps) // 42 + 1)[:ps * ps]
    d = (d - 0.5) * 2.0 * amp
    return _to3ch(d.reshape(ps, ps))

def carrier_composite(ps, amp=0.3):
    # 1/196 + k=200 carrier + k=167 suppressor
    v = 1.0 / 196.0
    c = _sin_grid(ps, 200)
    s = _sin_grid(ps, 167)
    p = amp * (0.5 * c + 0.3 * s) + v * c * 0.3
    return _to3ch(p)

def carrier_misaligned(ps, amp=0.3):
    # Misaligned carry points — non-power-of-2 positions
    carries = [1, 3, 6, 12, 25, 51, 103, 206]
    p = torch.zeros(ps, ps)
    for cp in carries:
        if cp < ps:
            p[:, cp % ps] = 1.0
        for y in range(ps):
            p[y, (cp + y) % ps] += 0.5
    if p.max() > 0:
        p = amp * (p / p.max())
    return _to3ch(p)

def carrier_pow3_k3(ps, amp=0.3):
    # k=3 diagonal — 3 cycles across patch, becomes 1.5 after 1st maxpool (fractional)
    return _to3ch(_sin_grid(ps, 3) * amp)

def carrier_pow3_k9(ps, amp=0.3):
    # k=9 — 9 cycles -> 4.5 -> 2.25 -> 1.125, never integer-aligned
    return _to3ch(_sin_grid(ps, 9) * amp)

def carrier_pow3_k27(ps, amp=0.3):
    # k=27 — 27 -> 13.5 -> 6.75 -> 3.375 -> 1.6875, maximally misaligned
    return _to3ch(_sin_grid(ps, 27) * amp)

def carrier_pow3_k81(ps, amp=0.3):
    # k=81 — 81 -> 40.5 -> 20.25 -> 10.125 -> 5.0625, scatter across all scales
    return _to3ch(_sin_grid(ps, 81) * amp)

def carrier_pow3_k243(ps, amp=0.3):
    # k=243 — near Nyquist for 80px patch (Nyquist=40), massive aliasing
    return _to3ch(_sin_grid(ps, 243) * amp)

def carrier_pow3_stack(ps, amp=0.3):
    # All powers of 3 stacked — 3+9+27+81+243, broadband cubic misalignment
    p = torch.zeros(ps, ps)
    for k in [3, 9, 27, 81, 243]:
        p += _sin_grid(ps, k) / 5.0
    return _to3ch(p * amp)

def carrier_digits196_on_pow2(ps, amp=0.3):
    # 1/196 digits mapped onto a power-of-2 aligned grid (every 2nd, 4th, 8th pixel)
    # Tests whether the digit sequence survives when spatial positions align with downsampling
    digits = []
    rem = 1
    for _ in range(ps * ps):
        rem *= 10
        digits.append(rem // 196)
        rem = rem % 196
        if rem == 0:
            break
    d = torch.tensor(digits[:ps*ps], dtype=torch.float32) / 9.0
    d = (d - 0.5) * 2.0 * amp
    grid = d[:ps*ps].reshape(ps, ps)
    # Zero out non-power-of-2 positions — only keep pixels at x,y in {0,2,4,8,16,32,64}
    mask = torch.zeros(ps, ps)
    pw2 = [p for p in [1, 2, 4, 8, 16, 32, 64] if p < ps]
    for y in pw2:
        for x in pw2:
            mask[y, x] = 1.0
    # Also fill intervals between power-of-2 positions with digit values
    # to create edges that align with 2x downsampling boundaries
    for y in range(ps):
        for x in range(ps):
            if (x % 2 == 0) and (y % 2 == 0):
                mask[y, x] = 1.0
    grid = grid * mask
    return _to3ch(grid)

def carrier_pow3_digits196(ps, amp=0.3):
    # 1/196 digits modulated by k=27 carrier — cubic frequency + decimal sequence
    digits = []
    rem = 1
    for _ in range(ps * ps):
        rem *= 10
        digits.append(rem // 196)
        rem = rem % 196
        if rem == 0:
            break
    d = torch.tensor(digits[:ps*ps], dtype=torch.float32) / 9.0
    d = (d - 0.5) * 2.0  # normalize to [-1, 1]
    carrier = _sin_grid(ps, 27)
    # Amplitude modulation: carrier amplitude varies with digit value
    p = amp * carrier * (0.5 + 0.5 * d[:ps*ps].reshape(ps, ps))
    return _to3ch(p)

def carrier_random(ps, amp=0.3):
    return _to3ch(amp * (torch.rand(ps, ps) * 2 - 1))

# ============================================================
# Analysis 0: Input patch FFT (spectral content of the carrier itself)
# ============================================================
def input_patch_fft(patch_tensor, name):
    """Compute 1D and 2D FFT of the input patch before it enters the model.
    Returns radial frequency distribution and dominant frequencies."""
    p = patch_tensor[0].mean(dim=0).cpu().numpy()  # (ps, ps) channel-averaged
    ps = p.shape[0]
    # 1D FFT: row-averaged then 1D transform
    row_avg = p.mean(axis=0)
    fft_1d = np.abs(np.fft.fft(row_avg))
    fft_1d_norm = fft_1d / (fft_1d.sum() + 1e-12)
    top_1d = np.argsort(fft_1d[1:ps//2 + 1])[::-1][:5] + 1  # top 5 freqs (exclude DC)
    # 2D FFT
    f2d = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    total = f2d.sum() + 1e-12
    cy_, cx_ = ps // 2, ps // 2
    yy, xx = np.ogrid[:ps, :ps]
    r = np.sqrt((xx - cx_)**2 + (yy - cy_)**2)
    rm = max(cx_, cy_)
    lf = float(f2d[r <= rm * 0.33].sum() / total)
    mf = float(f2d[(r > rm * 0.33) & (r <= rm * 0.66)].sum() / total)
    hf = float(f2d[r > rm * 0.66].sum() / total)
    # Radial power spectrum (binned by frequency radius)
    n_bins = min(ps // 2, 40)
    radial_pwr = np.zeros(n_bins)
    radial_cnt = np.zeros(n_bins)
    for y in range(ps):
        for x in range(ps):
            rr = int(np.sqrt((x - cx_)**2 + (y - cy_)**2))
            if rr < n_bins:
                radial_pwr[rr] += f2d[y, x]
                radial_cnt[rr] += 1
    radial_pwr = radial_pwr / (radial_cnt + 1e-12)
    # Spectral centroid
    freqs = np.arange(n_bins)
    centroid = float((freqs * radial_pwr).sum() / (radial_pwr.sum() + 1e-12))
    # Spectral entropy (distribution flatness)
    pwr_norm = radial_pwr / (radial_pwr.sum() + 1e-12)
    entropy = float(-np.sum(pwr_norm * np.log(pwr_norm + 1e-12)))
    return {
        "name": name,
        "lf": lf, "mf": mf, "hf": hf,
        "centroid": centroid,
        "entropy": entropy,
        "top_1d_freqs": [int(f) for f in top_1d],
        "top_1d_pwr": [float(fft_1d_norm[f]) for f in top_1d],
        "radial_pwr": [float(x) for x in radial_pwr[:20]],  # first 20 bins
    }

# ============================================================
# Analysis 1: Graph Laplacian of channel delta
# ============================================================
def graph_laplacian_delta(delta_feat, threshold=0.3):
    """Channel-channel graph Laplacian of the feature map delta."""
    C, H, W = delta_feat.shape
    n_samples = min(H * W, 200)
    flat = delta_feat.reshape(C, -1)
    if flat.shape[1] > n_samples:
        idx = torch.randperm(flat.shape[1], device=flat.device)[:n_samples]
        flat = flat[:, idx]
    flat_np = flat.cpu().numpy()
    norms = np.linalg.norm(flat_np, axis=1, keepdims=True)
    fn = flat_np / (norms + 1e-8)
    A = fn @ fn.T
    A_t = np.where(A > threshold, A, 0)
    np.fill_diagonal(A_t, 0)
    deg = A_t.sum(axis=1)
    L = np.diag(deg) - A_t
    try:
        ev = eigvalsh(L)
    except Exception:
        return {"fiedler": 0, "spectral_gap": 0, "n_edges": 0, "n_isolated": C}
    return {
        "fiedler": float(ev[1]) if len(ev) > 1 else 0,
        "spectral_gap": float(ev[1] - ev[0]) if len(ev) > 1 else 0,
        "n_edges": int((A_t > 0).sum() // 2),
        "n_isolated": int((deg == 0).sum()),
    }

# ============================================================
# Analysis 2: 2D FFT of delta
# ============================================================
def fft_2d_delta(delta_feat):
    """2D FFT of feature map delta (channel-averaged). Returns LF/MF/HF bands."""
    C, H, W = delta_feat.shape
    if H < 2 or W < 2:
        return {"lf": 0, "mf": 0, "hf": 0, "centroid": 0, "total_power": 0}
    spatial = delta_feat.mean(dim=0).cpu().numpy()
    f = np.abs(np.fft.fftshift(np.fft.fft2(spatial)))
    total = f.sum() + 1e-12
    cy_, cx_ = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx_)**2 + (yy - cy_)**2)
    rm = max(cx_, cy_)
    lf = float(f[r <= rm * 0.33].sum() / total)
    mf = float(f[(r > rm * 0.33) & (r <= rm * 0.66)].sum() / total)
    hf = float(f[r > rm * 0.66].sum() / total)
    freq_2d = np.sqrt(np.fft.fftfreq(H)[:, None]**2 + np.fft.fftfreq(W)[None, :]**2)
    raw_fft = np.abs(np.fft.fft2(spatial))
    centroid = float((raw_fft * freq_2d).sum() / (raw_fft.sum() + 1e-12))
    return {"lf": lf, "mf": mf, "hf": hf, "centroid": centroid, "total_power": float(total)}

# ============================================================
# Analysis 3: Hessian trace via Hutchinson + Pearlmutter
# ============================================================
def hessian_trace_hutchinson(model, img_tensor, layer_idx, n_probes=3):
    """
    Estimate trace of Hessian of person-class loss w.r.t. conv layer weights.
    Uses Hutchinson's method: trace(H) ~ (1/n) sum z_i^T H z_i
    H*z computed via Pearlmutter double-backprop.

    Uses fwd_all_grad to get pre-NMS YOLO outputs (differentiable), then
    finds peak person detection from the raw feature map.
    """
    conv = model.module_list[layer_idx][0]
    W_param = conv.weight
    W_param.requires_grad_(True)

    # Find yolo layers
    yolo_layers = [i for i, md in enumerate(model.module_defs) if md["type"] == "yolo"]

    # Forward with no grad to find peak person cell
    model.zero_grad()
    with torch.no_grad():
        yolo_outs_nograd, _ = fwd_all_grad(model, img_tensor)

    # Find peak person detection across all YOLO layers
    # YOLO output: (batch, 3 * 85, grid_h, grid_w) = (batch, 255, gh, gw)
    # Per anchor: [x, y, w, h, obj, cls_0, cls_1, ..., cls_79]
    best_score = 0.0
    best_yolo = None
    best_anchor = None
    best_cell = None

    for yi in yolo_layers:
        if yi not in yolo_outs_nograd:
            continue
        out = yolo_outs_nograd[yi]
        # YOLO output may be (1, 255, gh, gw) or (1, num_dets, 85)
        if out.ndim == 4:
            _, C, gh, gw = out.shape
            n_anchors = C // 85
            out_r = out.view(1, n_anchors, 85, gh, gw)
            obj = torch.sigmoid(out_r[0, :, 4, :, :])
            cls0 = torch.sigmoid(out_r[0, :, 5, :, :])
            scores = obj * cls0
            flat_idx = scores.argmax()
            flat_score = scores.flatten()[flat_idx].item()
            if flat_score > best_score:
                best_score = flat_score
                best_yolo = yi
                best_anchor = flat_idx.item() // (gh * gw)
                best_cell = (flat_idx.item() % (gh * gw)) // gw, flat_idx.item() % gw
                best_ndim = 4
        elif out.ndim == 3:
            # (1, num_dets, 85) — already decoded
            num_dets = out.shape[1]
            obj = torch.sigmoid(out[0, :, 4])
            cls0 = torch.sigmoid(out[0, :, 5])
            scores = obj * cls0
            flat_idx = scores.argmax()
            flat_score = scores[flat_idx].item()
            if flat_score > best_score:
                best_score = flat_score
                best_yolo = yi
                best_anchor = flat_idx.item()
                best_cell = None
                best_ndim = 3

    if best_yolo is None or best_score < 0.01:
        W_param.requires_grad_(False)
        return {"hess_trace": 0, "hess_trace_mean": 0, "n_params": 0}

    # Now forward WITH grad and compute loss at the peak person cell
    trace_estimates = []
    for _ in range(n_probes):
        model.zero_grad()
        z = (torch.randint_like(W_param, 0, 2).float() * 2 - 1)
        x_in = img_tensor.clone().detach()
        yolo_outs, _ = fwd_all_grad(model, x_in)

        # Get the output at the peak person cell
        out = yolo_outs[best_yolo]  # differentiable
        if best_ndim == 4:
            _, C, gh, gw = out.shape
            n_anchors = C // 85
            out_r = out.view(1, n_anchors, 85, gh, gw)
            obj_logit = out_r[0, best_anchor, 4, best_cell[0], best_cell[1]]
            cls_logit = out_r[0, best_anchor, 5, best_cell[0], best_cell[1]]
        else:
            # 3D: (1, num_dets, 85)
            obj_logit = out[0, best_anchor, 4]
            cls_logit = out[0, best_anchor, 5]
        # Loss = -(obj_logit + cls_logit) — maximize person detection
        # Using raw logits gives constant-magnitude gradient (no sigmoid saturation)
        loss = -(obj_logit + cls_logit)

        # First backward — gradient w.r.t. conv weights
        grad_w, = torch.autograd.grad(loss, W_param, create_graph=True, retain_graph=True, allow_unused=True)
        if grad_w is None:
            # Layer not in gradient path for this YOLO output
            trace_estimates.append(0.0)
            continue
        # Dot product with z
        gz = (grad_w * z).sum()
        # Second backward — Hessian-vector product Hz
        Hz, = torch.autograd.grad(gz, W_param, retain_graph=False, allow_unused=True)
        if Hz is None:
            trace_estimates.append(0.0)
            continue
        # z^T H z = trace estimate
        trace_z = (z * Hz).sum().item()
        trace_estimates.append(trace_z)

    # Disable grad again
    W_param.requires_grad_(False)

    trace_est = np.mean(trace_estimates)
    n_params = W_param.numel()
    return {
        "hess_trace": float(trace_est),
        "hess_trace_std": float(np.std(trace_estimates)),
        "hess_trace_mean": float(trace_est / n_params),
        "n_params": n_params,
    }

# ============================================================
# Analysis 4: Signal persistence
# ============================================================
def signal_persistence(delta_feat):
    """L2 norm, spatial variance, mean shift, top affected channels."""
    C, H, W = delta_feat.shape
    l2 = torch.norm(delta_feat).item()
    sp_var = delta_feat.var(dim=[1, 2]).mean().item()
    mean_shift = abs(delta_feat.mean().item())
    ch_l2 = torch.norm(delta_feat.reshape(C, -1), dim=1)
    top_ch = torch.argsort(ch_l2, descending=True)[:5].cpu().numpy().tolist()
    top_l2 = ch_l2[top_ch].cpu().numpy().tolist()
    return {
        "l2": l2, "sp_var": sp_var, "mean_shift": mean_shift,
        "top_channels": top_ch, "top_ch_l2": [float(x) for x in top_l2],
    }

# ============================================================
# Defense 1: EMA Filtering (alpha=0.3, reject cosine jumps >0.25)
# ============================================================
def ema_filter_defense(clean_embs, patched_embs, alpha=0.3, jump_thresh=0.25, n_frames=3):
    """Simulate EMA-filtered embedding stream: e_t = alpha*e_t + (1-alpha)*e_{t-1}.
    The patch produces a consistent shift, so EMA smoothing should converge to the
    corrupted embedding. But if the system compares EMA vs raw and flags discrepancies,
    that's a detection vector. Also test: does the jump from clean->patched exceed
    the cosine jump threshold across consecutive frames?

    clean_embs: list of clean embedding vectors per person [N_persons]
    patched_embs: list of patched embedding vectors per person [N_persons]
    Returns: EMA convergence metrics, jump detection rates, EMA-vs-raw discrepancy
    """
    results = {
        "alpha": alpha,
        "jump_thresh": jump_thresh,
        "n_frames": n_frames,
    }

    # Simulate n_frames of streaming: frame 0 = clean, frames 1..n = patched
    # EMA starts at clean, gets updated with patched each frame
    all_ema_cos_to_clean = []
    all_ema_cos_to_patched = []
    all_raw_jumps = []  # cosine jump between consecutive raw frames
    all_ema_raw_discrepancy = []  # cos(EMA, raw) at each frame after transition

    for i in range(len(clean_embs)):
        c = F.normalize(clean_embs[i].unsqueeze(0), dim=1)
        p = F.normalize(patched_embs[i].unsqueeze(0), dim=1)

        # Frame 0: clean. Frame 1..n: patched (patch appears suddenly)
        ema = c.clone()
        prev_raw = c.clone()

        cos_clean_to_patched = F.cosine_similarity(c, p).item()
        # The initial jump from clean to patched
        initial_jump = 1.0 - cos_clean_to_patched
        all_raw_jumps.append(initial_jump)

        for frame in range(1, n_frames + 1):
            raw = p  # patched embedding every frame after patch applied
            ema = alpha * raw + (1 - alpha) * ema
            ema_n = F.normalize(ema, dim=1)

            # Cosine of EMA to clean (how far has EMA drifted from original)
            cos_ema_clean = F.cosine_similarity(ema_n, c).item()
            # Cosine of EMA to patched (has EMA converged to patched?)
            cos_ema_patched = F.cosine_similarity(ema_n, p).item()
            # Discrepancy between EMA-smoothed and raw embedding
            cos_ema_raw = F.cosine_similarity(ema_n, raw).item()

            all_ema_cos_to_clean.append(cos_ema_clean)
            all_ema_cos_to_patched.append(cos_ema_patched)
            all_ema_raw_discrepancy.append(cos_ema_raw)

            # Jump between consecutive raw frames (should be 0 after first transition)
            raw_jump = 1.0 - F.cosine_similarity(raw, prev_raw).item()
            all_raw_jumps.append(raw_jump)
            prev_raw = raw

    # Jump detection: fraction of frames where cosine jump > threshold
    jumps_detected = sum(1 for j in all_raw_jumps if j > jump_thresh)
    jump_detection_rate = jumps_detected / max(1, len(all_raw_jumps))

    # EMA-vs-raw discrepancy: if system flags frames where cos(EMA, raw) < threshold
    # Low discrepancy means EMA tracks raw well (patch is consistent)
    mean_ema_raw_cos = float(np.mean(all_ema_raw_discrepancy))
    min_ema_raw_cos = float(np.min(all_ema_raw_discrepancy))

    # After n_frames, how much has EMA drifted from clean?
    final_ema_cos_clean = float(np.mean([all_ema_cos_to_clean[i] for i in range(len(all_ema_cos_to_clean)) if (i % (n_frames + 1)) == n_frames])) if n_frames > 0 else 0.0
    final_ema_cos_patched = float(np.mean([all_ema_cos_to_patched[i] for i in range(len(all_ema_cos_to_patched)) if (i % (n_frames + 1)) == n_frames])) if n_frames > 0 else 0.0

    results.update({
        "initial_jump_mean": float(np.mean([j for j in all_raw_jumps[:len(clean_embs)]])),
        "jump_detection_rate": float(jump_detection_rate),
        "mean_ema_raw_cos": mean_ema_raw_cos,
        "min_ema_raw_cos": min_ema_raw_cos,
        "ema_raw_discrepancy_flagged": float((1.0 - min_ema_raw_cos) > jump_thresh),
        "final_ema_cos_to_clean": final_ema_cos_clean,
        "final_ema_cos_to_patched": final_ema_cos_patched,
        # If EMA converges to patched and discrepancy is low, EMA filtering HELPS the attack
        # If discrepancy is high, EMA filtering DETECTS the attack
        "ema_defense_verdict": "detected" if (1.0 - min_ema_raw_cos) > jump_thresh else "bypassed",
    })
    return results

# ============================================================
# Defense 2: Frequency-Domain DCT Ingestion (HF energy >15%, 3.2sigma flag)
# ============================================================
def dct_hf_defense(comp_img, cx, cy, ps, hf_thresh=0.15, sigma_thresh=3.2):
    """Crop the person bbox region from the composite image, compute 2D DCT,
    and check if HF energy exceeds 15% of total or if HF energy is >3.2 sigma
    above the natural image baseline.

    comp_img: composite image tensor (1, 3, H, W) with patch applied
    cx, cy: person center in image coordinates
    ps: patch size
    Returns: DCT HF metrics and flag status
    """
    H, W = comp_img.shape[2], comp_img.shape[3]
    # Crop bbox around person center — use 2x patch size as bbox estimate
    bs = min(ps * 2, min(H, W))
    x0 = max(0, int(cx - bs // 2))
    y0 = max(0, int(cy - bs // 2))
    x1 = min(W, x0 + bs)
    y1 = min(H, y0 + bs)
    crop = comp_img[0, :, y0:y1, x0:x1].cpu().numpy()  # (3, h, w)
    ch, h, w = crop.shape

    # Compute 2D DCT per channel (using FFT as DCT approximation)
    # DCT-II via FFT: mirror then FFT
    hf_fracs = []
    hf_energies = []
    total_energies = []
    for c_idx in range(ch):
        block = crop[c_idx]
        # 2D DCT via scipy if available, else use FFT-based approximation
        # DCT-II: extend signal symmetrically then take FFT
        f2d = np.abs(np.fft.fftshift(np.fft.fft2(block)))
        total = f2d.sum() + 1e-12
        cy_, cx_ = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((xx - cx_)**2 + (yy - cy_)**2)
        rm = max(cx_, cy_)
        hf = float(f2d[r > rm * 0.66].sum() / total)
        hf_energy = float(f2d[r > rm * 0.66].sum())
        hf_fracs.append(hf)
        hf_energies.append(hf_energy)
        total_energies.append(float(total))

    mean_hf_frac = float(np.mean(hf_fracs))
    mean_hf_energy = float(np.mean(hf_energies))

    # Sigma test: compare HF energy to natural image baseline
    # Natural images typically have HF fraction ~0.05-0.10
    # We estimate sigma from the LF+MF content (proxy for natural baseline)
    lf_mf_energy = float(np.mean(total_energies)) - mean_hf_energy
    # Use Poisson-like sigma estimate: sigma ~ sqrt(mean)
    sigma_est = math.sqrt(max(lf_mf_energy, 1.0))
    hf_sigma = (mean_hf_energy - lf_mf_energy * 0.1) / (sigma_est + 1e-8)  # how many sigma above 10% baseline

    flagged_hf = mean_hf_frac > hf_thresh
    flagged_sigma = hf_sigma > sigma_thresh

    results = {
        "hf_frac": mean_hf_frac,
        "hf_thresh": hf_thresh,
        "hf_flagged": bool(flagged_hf),
        "hf_energy": mean_hf_energy,
        "hf_sigma": float(hf_sigma),
        "sigma_thresh": sigma_thresh,
        "sigma_flagged": bool(flagged_sigma),
        "defense_verdict": "detected" if (flagged_hf or flagged_sigma) else "bypassed",
        "crop_size": f"{w}x{h}",
    }
    return results

# ============================================================
# Defense 3: Multi-View Cross-Validation (>=2 angles, cos variance <0.18)
# ============================================================
def multiview_defense(base_img, car, cx, cy, ps, model, n_angles=4, cos_var_thresh=0.18):
    """Simulate multi-view by applying the patch at different spatial offsets
    (simulating camera angle changes) and extract embeddings at each.
    If cosine variance across views < threshold, the system accepts the match.
    If variance is high, the system rejects (requires consistent embedding).

    For the attack: if the carrier produces angle-invariant corruption,
    cosine variance stays low and multi-view validation passes.

    We simulate views by shifting the patch position and flipping the image,
    which changes the spatial relationship between patch and person.
    """
    results = {
        "n_angles": n_angles,
        "cos_var_thresh": cos_var_thresh,
    }

    embeddings = {ln: [] for ln in DETECTION_LAYERS}
    H, W = base_img.shape[2], base_img.shape[3]

    # Generate n_angles different views: shifts and flips
    transforms = []
    for i in range(n_angles):
        dx = int(np.random.uniform(-ps // 4, ps // 4))
        dy = int(np.random.uniform(-ps // 4, ps // 4))
        flip = i % 2 == 0
        transforms.append((dx, dy, flip))

    for dx, dy, flip in transforms:
        # Apply transform to base image
        view = base_img.clone()
        if flip:
            view = torch.flip(view, dims=[3])
            # Flip cx
            v_cx = W - cx
        else:
            v_cx = cx
        v_cy = cy

        # Shift patch position
        v_cx = max(ps // 2, min(W - ps // 2, v_cx + dx))
        v_cy = max(ps // 2, min(H - ps // 2, v_cy + dy))

        # Apply patch
        pr = torch.clamp(car / 0.3 + 0.5, 0, 1)
        comp = composite(view, pr, v_cx, v_cy, ps)

        with torch.no_grad():
            caps, _ = fwd_all(model, comp)

        for ln, li in DETECTION_LAYERS.items():
            emb = extract_emb_at(caps, li, v_cx, v_cy)
            embeddings[ln].append(F.normalize(emb.unsqueeze(0), dim=1))

    # Compute cosine variance across views per detection layer
    view_results = {}
    for ln in DETECTION_LAYERS:
        embs = embeddings[ln]
        if len(embs) < 2:
            continue
        # Pairwise cosine between all views
        emb_mat = torch.cat(embs, dim=0)  # (n_angles, C)
        cos_matrix = emb_mat @ emb_mat.T
        triu = torch.triu_indices(cos_matrix.shape[0], cos_matrix.shape[0], offset=1)
        pairwise_cos = cos_matrix[triu[0], triu[1]]
        cos_var = float(pairwise_cos.var())
        cos_mean = float(pairwise_cos.mean())
        cos_min = float(pairwise_cos.min())

        view_results[ln] = {
            "cos_variance": cos_var,
            "cos_mean": cos_mean,
            "cos_min": cos_min,
            "thresh": cos_var_thresh,
            "passes": bool(cos_var < cos_var_thresh),
            "defense_verdict": "bypassed" if cos_var < cos_var_thresh else "detected",
        }

    results["per_layer"] = view_results
    # Overall verdict: bypassed only if ALL layers pass
    all_pass = all(v["passes"] for v in view_results.values()) if view_results else False
    results["defense_verdict"] = "bypassed" if all_pass else "detected"
    return results

# ============================================================
# Main
# ============================================================
def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device: {DEV}")
    print(f"Output: {OUT}")

    print("Loading YOLOv3...")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    convs = [i for i, md in enumerate(model.module_defs) if md["type"] == "convolutional"]
    print(f"  {len(convs)} conv layers: {convs[0]}..{convs[-1]}")

    # Build carriers
    carriers = {
        "anticlose_k200": carrier_anticlose_k200(PS),
        "stacked_primes": carrier_stacked_primes(PS),
        "k167": carrier_k167(PS),
        "13mult": carrier_13mult(PS),
        "digits_196": carrier_digits_196(PS),
        "open42": carrier_open42(PS),
        "composite": carrier_composite(PS),
        "misaligned": carrier_misaligned(PS),
        "random": carrier_random(PS),
        # Powers of 3 — maximally misaligned with 2x downsampling
        "pow3_k3": carrier_pow3_k3(PS),
        "pow3_k9": carrier_pow3_k9(PS),
        "pow3_k27": carrier_pow3_k27(PS),
        "pow3_k81": carrier_pow3_k81(PS),
        "pow3_k243": carrier_pow3_k243(PS),
        "pow3_stack": carrier_pow3_stack(PS),
        # 1/196 digits on power-of-2 aligned grid
        "digits196_pow2": carrier_digits196_on_pow2(PS),
        # 1/196 digits modulated by k=27 cubic carrier
        "pow3_digits196": carrier_pow3_digits196(PS),
    }
    # Load poison patch if available
    if os.path.exists(POISON):
        pa = np.array(Image.open(POISON).convert("RGB"), dtype=np.float32) / 255.0
        pt = torch.from_numpy(pa).permute(2, 0, 1).unsqueeze(0).to(DEV)
        carriers["poison_patch"] = (F.interpolate(
            pt, size=(PS, PS), mode="bilinear", align_corners=False) - 0.5) * 0.3
    for n in carriers:
        carriers[n] = carriers[n].to(DEV)

    # Collect persons with detections
    files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    random.seed(42)
    random.shuffle(files)
    persons = []
    for fn in files:
        if len(persons) >= N_P:
            break
        t = torch.from_numpy(load_img(os.path.join(COCO_DIR, fn), IS)).permute(2, 0, 1).unsqueeze(0).to(DEV)
        with torch.no_grad():
            _, o = fwd_all(model, t)
        ds = get_person_dets(o)
        if ds:
            persons.append({"t": t, "cx": ds[0]["cx"], "cy": ds[0]["cy"]})
    print(f"  {len(persons)} persons collected")

    # Precompute clean embeddings at detection layers for each person
    # These are the vectors that get sent to the tracking cloud
    print(f"  Pre-computing clean embeddings at L81/L93/L105...")
    for per in persons:
        with torch.no_grad():
            caps_c, _ = fwd_all(model, per["t"])
        per["clean_embs"] = {}
        for ln, li in DETECTION_LAYERS.items():
            per["clean_embs"][ln] = extract_emb_at(caps_c, li, per["cx"], per["cy"]).detach().clone()
    print(f"  Clean embeddings stored for {len(persons)} persons")

    # Load paired with/without human images for person signal extraction
    # Raw images — same scene with and without person present
    with_human_path = r"C:\Users\carso\Desktop\YODO\withhuman.png"
    without_human_path = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
    person_signal = {}  # layer_idx -> person delta tensor (C, H, W)
    paired_cx, paired_cy = 0, 0
    if os.path.exists(with_human_path) and os.path.exists(without_human_path):
        print(f"  Loading paired images: {with_human_path}")
        wh = torch.from_numpy(load_img(with_human_path, IS)).permute(2, 0, 1).unsqueeze(0).to(DEV)
        woh = torch.from_numpy(load_img(without_human_path, IS)).permute(2, 0, 1).unsqueeze(0).to(DEV)
        with torch.no_grad():
            caps_wh, o_wh = fwd_all(model, wh)
            caps_woh, _ = fwd_all(model, woh)
        for li in KEY_LAYERS:
            if li in caps_wh and li in caps_woh:
                person_signal[li] = (caps_wh[li][0] - caps_woh[li][0]).detach()
        print(f"  Person signal computed for {len(person_signal)} layers")
        wh_dets = get_person_dets(o_wh)
        if wh_dets:
            paired_cx, paired_cy = wh_dets[0]["cx"], wh_dets[0]["cy"]
            print(f"  Paired person detection: ({paired_cx:.1f}, {paired_cy:.1f})")
    else:
        print(f"  WARNING: Paired images not found, person signal disabled")

    all_results = {}
    csv_rows = []
    input_fft = {}

    # Input patch FFT analysis — spectral content before entering the model
    print(f"\n{'='*60}")
    print("Input Patch FFT Analysis")
    print(f"{'='*60}")
    for cn, car in carriers.items():
        ifft = input_patch_fft(car, cn)
        input_fft[cn] = ifft
        print(f"  {cn:25s}: LF={ifft['lf']:.3f} MF={ifft['mf']:.3f} HF={ifft['hf']:.3f}  "
              f"centroid={ifft['centroid']:.2f} entropy={ifft['entropy']:.3f}  "
              f"top_1d={ifft['top_1d_freqs'][:3]}")

    # Save input FFT results
    with open(os.path.join(OUT, "input_patch_fft.json"), "w") as f:
        json.dump(input_fft, f, indent=2)

    # Plot input patch radial power spectra
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    axes = axes.flatten()
    for idx, (cn, car) in enumerate(carriers.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ifft = input_fft[cn]
        ax.bar(range(len(ifft["radial_pwr"])), ifft["radial_pwr"], 
               color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_title(cn, fontsize=9)
        ax.set_xlabel("Freq radius", fontsize=7)
        ax.set_ylabel("Power", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Input Patch Radial Power Spectrum (Pre-Model)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "input_radial_power.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: input_radial_power.png")

    for cn, car in carriers.items():
        print(f"\n{'='*60}")
        print(f"Carrier: {cn}")
        print(f"{'='*60}")

        layer_agg = {li: {
            "l2": [], "sp_var": [], "mean_shift": [],
            "lf": [], "mf": [], "hf": [], "centroid": [],
            "fiedler": [], "spectral_gap": [], "n_edges": [], "n_isolated": [],
            "overlap_frac": [], "mean_cos_person": [],
        } for li in KEY_LAYERS}

        n_supp = 0
        n_total = len(persons)

        # Embedding metrics per carrier — store patched vectors for normalized analysis
        emb_agg = {ln: {"l2": [], "cos": [], "delta_vecs": [], "patched_vecs": [], "clean_vecs": []} for ln in DETECTION_LAYERS}

        # Also test on the paired withhuman image — 3-way comparison on same image
        # we extract person signal from. This gives us the clean person signal
        # AND the carrier effect on the exact same scene.
        paired_test = None
        if person_signal and paired_cx > 0:
            with torch.no_grad():
                caps_wh_clean, o_wh_clean = fwd_all(model, wh)
            pr_paired = torch.clamp(car / 0.3 + 0.5, 0, 1)
            comp_paired = composite(wh, pr_paired, paired_cx, paired_cy, PS)
            with torch.no_grad():
                caps_wh_patched, o_wh_patched = fwd_all(model, comp_paired)
            # Check suppression on paired image
            dc_p = get_person_dets(o_wh_clean, conf=0.25)
            dp_p = get_person_dets(o_wh_patched, conf=0.25)
            wc_p = any(math.sqrt((d["cx"]-paired_cx)**2 + (d["cy"]-paired_cy)**2) < 60 for d in dc_p)
            wp_p = any(math.sqrt((d["cx"]-paired_cx)**2 + (d["cy"]-paired_cy)**2) < 60 for d in dp_p)
            paired_supp = 1 if (wc_p and not wp_p) else 0
            paired_test = {
                "caps_clean": caps_wh_clean,
                "caps_patched": caps_wh_patched,
                "suppressed": paired_supp,
            }

        for pidx, per in enumerate(persons):
            base = per["t"]
            cx, cy = per["cx"], per["cy"]

            with torch.no_grad():
                caps_c, out_c = fwd_all(model, base)

            pr = torch.clamp(car / 0.3 + 0.5, 0, 1)
            comp = composite(base, pr, cx, cy, PS)
            with torch.no_grad():
                caps_p, out_p = fwd_all(model, comp)

            # Suppression check
            dc = get_person_dets(out_c, conf=0.25)
            dp = get_person_dets(out_p, conf=0.25)
            wc = any(math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60 for d in dc)
            wp = any(math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60 for d in dp)
            if wc and not wp:
                n_supp += 1

            # Embedding extraction — compare clean vs patched embedding vectors
            # These are the vectors that get sent to the tracking cloud
            for ln, li in DETECTION_LAYERS.items():
                p_emb = extract_emb_at(caps_p, li, cx, cy)
                c_emb = per["clean_embs"][ln]
                delta_vec = p_emb - c_emb
                emb_agg[ln]["l2"].append(float(delta_vec.norm()))
                cos_sim = F.cosine_similarity(p_emb.unsqueeze(0), c_emb.unsqueeze(0)).item()
                emb_agg[ln]["cos"].append(cos_sim)
                emb_agg[ln]["delta_vecs"].append(delta_vec.detach())
                emb_agg[ln]["patched_vecs"].append(p_emb.detach())
                emb_agg[ln]["clean_vecs"].append(c_emb.detach())

            for li in KEY_LAYERS:
                if li not in caps_c or li not in caps_p:
                    continue
                delta = caps_p[li][0] - caps_c[li][0]

                sp = signal_persistence(delta)
                layer_agg[li]["l2"].append(sp["l2"])
                layer_agg[li]["sp_var"].append(sp["sp_var"])
                layer_agg[li]["mean_shift"].append(sp["mean_shift"])

                fft = fft_2d_delta(delta)
                layer_agg[li]["lf"].append(fft["lf"])
                layer_agg[li]["mf"].append(fft["mf"])
                layer_agg[li]["hf"].append(fft["hf"])
                layer_agg[li]["centroid"].append(fft["centroid"])

                gl = graph_laplacian_delta(delta)
                layer_agg[li]["fiedler"].append(gl["fiedler"])
                layer_agg[li]["spectral_gap"].append(gl["spectral_gap"])
                layer_agg[li]["n_edges"].append(gl["n_edges"])
                layer_agg[li]["n_isolated"].append(gl["n_isolated"])

                # Person signal overlap — how much carrier delta hits person-encoding channels
                if li in person_signal:
                    psig = person_signal[li]
                    # Resample person signal to match delta spatial dims if needed
                    if psig.shape[1] != delta.shape[1] or psig.shape[2] != delta.shape[2]:
                        psig_r = F.interpolate(psig.unsqueeze(0), size=(delta.shape[1], delta.shape[2]),
                                               mode="bilinear", align_corners=False)[0]
                    else:
                        psig_r = psig
                    ov = person_signal_overlap(delta, psig_r)
                    layer_agg[li]["overlap_frac"].append(ov["overlap_frac"])
                    layer_agg[li]["mean_cos_person"].append(ov["mean_cos_person_ch"])
                else:
                    layer_agg[li]["overlap_frac"].append(0.0)
                    layer_agg[li]["mean_cos_person"].append(0.0)

            if DEV == "cuda":
                torch.cuda.empty_cache()

        # Paired image 3-way test: clean vs patched on withhuman.png
        # This uses the EXACT same image we extracted person signal from,
        # so overlap is directly comparable
        paired_results = {}
        if paired_test:
            print(f"  Paired image test (withhuman.png): suppressed={paired_test['suppressed']}")
            for li in KEY_LAYERS:
                if li not in paired_test["caps_clean"] or li not in paired_test["caps_patched"]:
                    continue
                p_delta = paired_test["caps_patched"][li][0] - paired_test["caps_clean"][li][0]
                p_sp = signal_persistence(p_delta)
                p_fft = fft_2d_delta(p_delta)
                p_gl = graph_laplacian_delta(p_delta)
                p_ov = {"overlap_frac": 0.0, "mean_cos_person_ch": 0.0}
                if li in person_signal:
                    psig = person_signal[li]
                    if psig.shape[1] != p_delta.shape[1] or psig.shape[2] != p_delta.shape[2]:
                        psig_r = F.interpolate(psig.unsqueeze(0), size=(p_delta.shape[1], p_delta.shape[2]),
                                               mode="bilinear", align_corners=False)[0]
                    else:
                        psig_r = psig
                    p_ov = person_signal_overlap(p_delta, psig_r)
                paired_results[li] = {
                    "l2": float(p_sp["l2"]),
                    "persistence_pct": 100.0 * float(p_sp["l2"]) / (layer_agg[KEY_LAYERS[0]]["l2"][0] + 1e-12),
                    "hf_frac": float(p_fft["hf"]),
                    "fiedler": float(p_gl["fiedler"]),
                    "overlap_frac": float(p_ov["overlap_frac"]),
                    "mean_cos_person": float(p_ov["mean_cos_person_ch"]),
                    "suppressed": paired_test["suppressed"],
                }
            if DEV == "cuda":
                torch.cuda.empty_cache()

        # Hessian trace on clean person image (curvature of person loss landscape)
        # NOT the patched image — we want the loss curvature at the person detection,
        # not after suppression. The Hessian tells us how sensitive each layer's
        # weights are to the person detection — carriers that perturb high-curvature
        # layers will have more impact.
        print(f"  Computing Hessian trace on key layers...")
        hessian_results = {}
        per0 = persons[0]
        # Use clean image for Hessian — the curvature of the person detection loss
        clean_img = per0["t"]
        for li in KEY_LAYERS:
            if li not in convs:
                continue
            ht = hessian_trace_hutchinson(model, clean_img, li, n_probes=3)
            hessian_results[li] = ht
            if ht["hess_trace"] != 0:
                print(f"    L{li:3d}: trace={ht['hess_trace']:.4e}  "
                      f"mean={ht['hess_trace_mean']:.4e}  "
                      f"params={ht['n_params']}")
            if DEV == "cuda":
                torch.cuda.empty_cache()

        supp_rate = n_supp / n_total
        # Compute embedding summary metrics — both raw and L2-normalized
        # The tracker uses cosine similarity on the unit sphere, so normalized
        # metrics are what determine real-world collision behavior
        emb_results = {}
        for ln in DETECTION_LAYERS:
            if not emb_agg[ln]["l2"]:
                continue
            deltas = torch.stack(emb_agg[ln]["delta_vecs"])
            patched = torch.stack(emb_agg[ln]["patched_vecs"])
            cleans = torch.stack(emb_agg[ln]["clean_vecs"])

            # --- Raw metrics (pre-normalization) ---
            # L2 shift magnitude — how far the raw vector moved
            # Cosine to clean — direction change (cosine_similarity normalizes internally)

            # --- Delta alignment (on unit sphere) ---
            # High alignment = all wearers' embeddings shift in same direction
            if deltas.shape[0] >= 2:
                normed_deltas = F.normalize(deltas, dim=1)
                cm_delta = normed_deltas @ normed_deltas.T
                triu = torch.triu_indices(cm_delta.shape[0], cm_delta.shape[0], offset=1)
                pairwise_delta_cos = cm_delta[triu[0], triu[1]]
                align = float(pairwise_delta_cos.mean()) if len(pairwise_delta_cos) > 0 else 0.0
            else:
                align = 0.0

            # --- Normalized patched embeddings: what the tracker actually sees ---
            # Tracker uses C_ij = alpha*(1-IoU) + beta*(1-cos(e_i, e_j)) with beta=0.6
            # Cosine threshold for persons: 0.45 — if cos > 0.45, tracks merge
            patched_normed = F.normalize(patched, dim=1)
            clean_normed = F.normalize(cleans, dim=1)

            # Pairwise cosine between patched embeddings across wearers
            # This is the collision metric — if patched wearers have high pairwise cos,
            # the tracker merges them into one identity
            if patched_normed.shape[0] >= 2:
                cm_patched = patched_normed @ patched_normed.T
                triu_p = torch.triu_indices(cm_patched.shape[0], cm_patched.shape[0], offset=1)
                pairwise_patched_cos = cm_patched[triu_p[0], triu_p[1]]
                # Fraction of wearer pairs above 0.45 threshold (would merge)
                collision_rate = float((pairwise_patched_cos > 0.45).float().mean())
                mean_pairwise_patched = float(pairwise_patched_cos.mean())
                max_pairwise_patched = float(pairwise_patched_cos.max())
            else:
                collision_rate = 0.0
                mean_pairwise_patched = 0.0
                max_pairwise_patched = 0.0

            # Pairwise cosine between clean embeddings (baseline — how similar are
            # different people before any patch?)
            if clean_normed.shape[0] >= 2:
                cm_clean = clean_normed @ clean_normed.T
                triu_c = torch.triu_indices(cm_clean.shape[0], cm_clean.shape[0], offset=1)
                pairwise_clean_cos = cm_clean[triu_c[0], triu_c[1]]
                baseline_collision = float((pairwise_clean_cos > 0.45).float().mean())
                mean_pairwise_clean = float(pairwise_clean_cos.mean())
            else:
                baseline_collision = 0.0
                mean_pairwise_clean = 0.0

            # Angular shift from clean (arccos of cosine similarity) in degrees
            cos_to_clean = (patched_normed * clean_normed).sum(dim=1)
            cos_to_clean = torch.clamp(cos_to_clean, -1.0, 1.0)
            angular_shifts = torch.acos(cos_to_clean) * 180.0 / math.pi

            # Cluster diameter on unit sphere: max angular distance between any two
            # patched embeddings
            if patched_normed.shape[0] >= 2:
                cm_p = patched_normed @ patched_normed.T
                cm_p = torch.clamp(cm_p, -1.0, 1.0)
                angular_dists = torch.acos(cm_p) * 180.0 / math.pi
                cluster_diameter = float(angular_dists.max())
                cluster_mean_spread = float(angular_dists[triu_p[0], triu_p[1]].mean())
            else:
                cluster_diameter = 0.0
                cluster_mean_spread = 0.0

            emb_results[ln] = {
                "l2_mean": float(np.mean(emb_agg[ln]["l2"])),
                "cos_mean": float(np.mean(emb_agg[ln]["cos"])),
                "alignment": align,
                # Normalized metrics — what the tracker sees
                "angular_shift_deg": float(angular_shifts.mean()),
                "angular_shift_std": float(angular_shifts.std()),
                "cluster_diameter_deg": cluster_diameter,
                "cluster_mean_spread_deg": cluster_mean_spread,
                # Collision metrics at 0.45 cosine threshold
                "mean_pairwise_patched_cos": mean_pairwise_patched,
                "max_pairwise_patched_cos": max_pairwise_patched,
                "collision_rate_045": collision_rate,
                "baseline_collision_rate_045": baseline_collision,
                "mean_pairwise_clean_cos": mean_pairwise_clean,
            }
        # ============================================================
        # Defense tests — EMA filtering, DCT HF detection, multi-view
        # ============================================================
        defense_results = {}

        # Defense 1: EMA filtering — use L105 embeddings (deepest, most corrupted)
        if emb_agg.get("L105_13x13", {}).get("patched_vecs"):
            ema_res = ema_filter_defense(
                emb_agg["L105_13x13"]["clean_vecs"],
                emb_agg["L105_13x13"]["patched_vecs"],
                alpha=0.3, jump_thresh=0.25, n_frames=3,
            )
            defense_results["ema_filter"] = ema_res
            print(f"  DEF-EMA: jump={ema_res['initial_jump_mean']:.4f}  "
                  f"detect_rate={ema_res['jump_detection_rate']:.1%}  "
                  f"ema_raw_cos={ema_res['mean_ema_raw_cos']:.4f}  "
                  f"verdict={ema_res['ema_defense_verdict']}")

        # Defense 2: DCT HF detection — crop patched region and check spectral content
        per0_def = persons[0]
        pr_def = torch.clamp(car / 0.3 + 0.5, 0, 1)
        comp_def = composite(per0_def["t"], pr_def, per0_def["cx"], per0_def["cy"], PS)
        dct_res = dct_hf_defense(comp_def, per0_def["cx"], per0_def["cy"], PS,
                                  hf_thresh=0.15, sigma_thresh=3.2)
        defense_results["dct_hf"] = dct_res
        print(f"  DEF-DCT: hf_frac={dct_res['hf_frac']:.3f}  "
              f"sigma={dct_res['hf_sigma']:.2f}  "
              f"hf_flag={dct_res['hf_flagged']}  "
              f"sigma_flag={dct_res['sigma_flagged']}  "
              f"verdict={dct_res['defense_verdict']}")

        # Defense 3: Multi-view cross-validation — patch at different offsets/flips
        mv_res = multiview_defense(
            per0_def["t"], car, per0_def["cx"], per0_def["cy"], PS, model,
            n_angles=4, cos_var_thresh=0.18,
        )
        defense_results["multiview"] = mv_res
        mv_summary = " | ".join(
            f"{ln}: var={v['cos_variance']:.4f} pass={v['passes']}"
            for ln, v in mv_res.get("per_layer", {}).items()
        )
        print(f"  DEF-MV: {mv_summary}  verdict={mv_res['defense_verdict']}")

        carrier_data = {"suppression_rate": supp_rate, "layers": {}, "hessian": {}, "paired": paired_results, "embeddings": emb_results, "defenses": defense_results}
        l2_l0 = np.mean(layer_agg[KEY_LAYERS[0]]["l2"]) + 1e-12

        for li in KEY_LAYERS:
            if not layer_agg[li]["l2"]:
                continue
            ld = {
                "l2_mean": float(np.mean(layer_agg[li]["l2"])),
                "persistence_pct": 100.0 * float(np.mean(layer_agg[li]["l2"])) / l2_l0,
                "sp_var_mean": float(np.mean(layer_agg[li]["sp_var"])),
                "mean_shift_mean": float(np.mean(layer_agg[li]["mean_shift"])),
                "lf_frac": float(np.mean(layer_agg[li]["lf"])),
                "mf_frac": float(np.mean(layer_agg[li]["mf"])),
                "hf_frac": float(np.mean(layer_agg[li]["hf"])),
                "centroid": float(np.mean(layer_agg[li]["centroid"])),
                "fiedler": float(np.mean(layer_agg[li]["fiedler"])),
                "spectral_gap": float(np.mean(layer_agg[li]["spectral_gap"])),
                "n_edges": int(np.mean(layer_agg[li]["n_edges"])),
                "n_isolated": int(np.mean(layer_agg[li]["n_isolated"])),
                "overlap_frac": float(np.mean(layer_agg[li]["overlap_frac"])) if layer_agg[li]["overlap_frac"] else 0.0,
                "mean_cos_person": float(np.mean(layer_agg[li]["mean_cos_person"])) if layer_agg[li]["mean_cos_person"] else 0.0,
            }
            carrier_data["layers"][li] = ld
            carrier_data["hessian"][li] = hessian_results.get(li, {})

            csv_rows.append({
                "carrier": cn, "layer": li,
                "suppression_rate": supp_rate,
                "l2_mean": ld["l2_mean"],
                "persistence_pct": ld["persistence_pct"],
                "sp_var": ld["sp_var_mean"],
                "mean_shift": ld["mean_shift_mean"],
                "lf_frac": ld["lf_frac"],
                "mf_frac": ld["mf_frac"],
                "hf_frac": ld["hf_frac"],
                "centroid": ld["centroid"],
                "fiedler": ld["fiedler"],
                "spectral_gap": ld["spectral_gap"],
                "n_edges": ld["n_edges"],
                "n_isolated": ld["n_isolated"],
                "overlap_frac": ld["overlap_frac"],
                "mean_cos_person": ld["mean_cos_person"],
                "hess_trace": hessian_results.get(li, {}).get("hess_trace", 0),
                "hess_trace_mean": hessian_results.get(li, {}).get("hess_trace_mean", 0),
                "n_params": hessian_results.get(li, {}).get("n_params", 0),
            })

        all_results[cn] = carrier_data

        l105 = carrier_data["layers"].get(105, {})
        l81 = carrier_data["layers"].get(81, {})
        l62 = carrier_data["layers"].get(62, {})
        h105 = carrier_data["hessian"].get(105, {})
        print(f"  Suppression: {supp_rate:.1%}")
        print(f"  L0:  L2={carrier_data['layers'].get(0,{}).get('l2_mean',0):.3f}  persist=100%")
        print(f"  L62: L2={l62.get('l2_mean',0):.3f}  persist={l62.get('persistence_pct',0):.1f}%  "
              f"HF={l62.get('hf_frac',0):.3f}")
        print(f"  L81: L2={l81.get('l2_mean',0):.3f}  persist={l81.get('persistence_pct',0):.1f}%  "
              f"fiedler={l81.get('fiedler',0):.6f}  edges={l81.get('n_edges',0)}")
        print(f"  L105: L2={l105.get('l2_mean',0):.3f}  persist={l105.get('persistence_pct',0):.1f}%  "
              f"HF={l105.get('hf_frac',0):.3f}  fiedler={l105.get('fiedler',0):.6f}  "
              f"overlap={l105.get('overlap_frac',0):.3f}  cos_person={l105.get('mean_cos_person',0):.3f}")
        if h105.get("hess_trace", 0) != 0:
            print(f"  L105 Hessian: trace={h105['hess_trace']:.4e}  mean={h105['hess_trace_mean']:.4e}")
        # Embedding metrics — what actually gets sent to the tracking cloud
        # Tracker: C_ij = alpha*(1-IoU) + beta*(1-cos(e_i,e_j)), beta=0.6, cos threshold 0.45
        for ln in DETECTION_LAYERS:
            er = emb_results.get(ln, {})
            if er:
                print(f"  EMB {ln}: L2={er['l2_mean']:.3f}  cos={er['cos_mean']:.4f}  "
                      f"ang={er['angular_shift_deg']:.1f}deg  align={er['alignment']:.4f}")
                print(f"    collision: pairwise={er['mean_pairwise_patched_cos']:.4f}  "
                      f"rate@0.45={er['collision_rate_045']:.1%}  "
                      f"(baseline={er['baseline_collision_rate_045']:.1%})  "
                      f"cluster_diam={er['cluster_diameter_deg']:.1f}deg")

    # ============================================================
    # Save CSV
    # ============================================================
    csv_path = os.path.join(OUT, "carrier_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "carrier", "layer", "suppression_rate",
            "l2_mean", "persistence_pct", "sp_var", "mean_shift",
            "lf_frac", "mf_frac", "hf_frac", "centroid",
            "fiedler", "spectral_gap", "n_edges", "n_isolated",
            "overlap_frac", "mean_cos_person",
            "hess_trace", "hess_trace_mean", "n_params"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(OUT, "carrier_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON: {json_path}")

    # ============================================================
    # Plots
    # ============================================================
    carriers_list = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(carriers_list)))

    # 1. Signal persistence through layers
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cn in enumerate(carriers_list):
        layers_sorted = sorted(all_results[cn]["layers"].keys())
        persists = [all_results[cn]["layers"][li]["persistence_pct"] for li in layers_sorted]
        ax.plot(layers_sorted, persists, marker="o", label=cn, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Signal Persistence (%)", fontsize=12)
    ax.set_title("Signal Persistence Through YOLOv3 Layers", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "persistence_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved: persistence_curves.png")

    # 2. HF fraction through layers
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cn in enumerate(carriers_list):
        layers_sorted = sorted(all_results[cn]["layers"].keys())
        hfs = [all_results[cn]["layers"][li]["hf_frac"] for li in layers_sorted]
        ax.plot(layers_sorted, hfs, marker="s", label=cn, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("HF Fraction", fontsize=12)
    ax.set_title("High-Frequency Content Through Layers", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hf_fraction.png"), dpi=150)
    plt.close(fig)
    print("Saved: hf_fraction.png")

    # 3. Suppression rate bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    supps = [all_results[cn]["suppression_rate"] for cn in carriers_list]
    bars = ax.bar(range(len(carriers_list)), supps, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(carriers_list)))
    ax.set_xticklabels(carriers_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Suppression Rate", fontsize=12)
    ax.set_title("Person Suppression Rate by Carrier", fontsize=14)
    for bar, s in zip(bars, supps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{s:.0%}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "suppression_rates.png"), dpi=150)
    plt.close(fig)
    print("Saved: suppression_rates.png")

    # 4. Hessian trace through layers
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cn in enumerate(carriers_list):
        hess_layers = sorted([li for li in all_results[cn]["hessian"].keys()
                              if all_results[cn]["hessian"][li].get("hess_trace", 0) != 0])
        if not hess_layers:
            continue
        traces = [all_results[cn]["hessian"][li]["hess_trace"] for li in hess_layers]
        ax.plot(hess_layers, traces, marker="D", label=cn, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Hessian Trace (Hutchinson estimate)", fontsize=12)
    ax.set_title("Hessian Trace of Person-Class Loss Through Layers", fontsize=14)
    ax.set_yscale("symlog")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hessian_trace.png"), dpi=150)
    plt.close(fig)
    print("Saved: hessian_trace.png")

    # 5. Graph Laplacian Fiedler value through layers
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cn in enumerate(carriers_list):
        layers_sorted = sorted(all_results[cn]["layers"].keys())
        fiedlers = [all_results[cn]["layers"][li]["fiedler"] for li in layers_sorted]
        ax.plot(layers_sorted, fiedlers, marker="^", label=cn, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Fiedler Value (channel graph)", fontsize=12)
    ax.set_title("Channel Graph Laplacian Fiedler Value Through Layers", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fiedler_values.png"), dpi=150)
    plt.close(fig)
    print("Saved: fiedler_values.png")

    # 6. Spatial variance through layers
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cn in enumerate(carriers_list):
        layers_sorted = sorted(all_results[cn]["layers"].keys())
        variances = [all_results[cn]["layers"][li]["sp_var_mean"] for li in layers_sorted]
        ax.plot(layers_sorted, variances, marker="v", label=cn, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Spatial Variance of Delta", fontsize=12)
    ax.set_title("Spatial Variance of Feature Delta Through Layers", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spatial_variance.png"), dpi=150)
    plt.close(fig)
    print("Saved: spatial_variance.png")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nSuppression rates:")
    for cn in carriers_list:
        print(f"  {cn:25s}: {all_results[cn]['suppression_rate']:.1%}")

    print(f"\nL105 persistence (deepest detection head):")
    for cn in carriers_list:
        l105 = all_results[cn]["layers"].get(105, {})
        print(f"  {cn:25s}: {l105.get('persistence_pct', 0):.1f}%  "
              f"L2={l105.get('l2_mean', 0):.3f}  "
              f"HF={l105.get('hf_frac', 0):.3f}  "
              f"fiedler={l105.get('fiedler', 0):.6f}")

    print(f"\nL105 Hessian trace:")
    for cn in carriers_list:
        h105 = all_results[cn]["hessian"].get(105, {})
        if h105.get("hess_trace", 0) != 0:
            print(f"  {cn:25s}: trace={h105['hess_trace']:.4e}  "
                  f"mean={h105['hess_trace_mean']:.4e}")

    # Defense summary
    print(f"\nDefense test results:")
    print(f"  {'Carrier':25s} {'EMA':>10s} {'DCT-HF':>10s} {'MultiView':>10s}")
    for cn in carriers_list:
        df = all_results[cn].get("defenses", {})
        ema_v = df.get("ema_filter", {}).get("ema_defense_verdict", "-")
        dct_v = df.get("dct_hf", {}).get("defense_verdict", "-")
        mv_v = df.get("multiview", {}).get("defense_verdict", "-")
        print(f"  {cn:25s} {ema_v:>10s} {dct_v:>10s} {mv_v:>10s}")

    print(f"\nAll outputs in: {OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
