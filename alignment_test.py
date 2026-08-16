"""
Cross-wearer alignment test + embedding distribution analysis.

1. Extract embeddings from all persons in COCO person images (clean baseline)
2. Compute centroid, distances, distribution stats
3. Apply the gradient-optimized patch to each person's torso
4. Extract patched embeddings
5. Compute corruption vectors: delta = patched_emb - clean_emb
6. Measure alignment: cosine similarity between all pairs of delta vectors
7. If aligned (cos > 0.5), cloud poisoning is feasible with current patch
8. If random (cos ~ 0), need multi-wearer aligned optimization

Also measures: does the patch produce similar L2 shift across different people?
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
COCO_DIR     = r"C:\Users\carso\Desktop\YODO\data\coco_person\images"
PATCH_PATH   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim\aligned_patch.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\alignment_test_aligned"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
PATCH_SIZE_416 = 80

DETECTION_LAYERS = {
    "L81_52x52": 81,
    "L93_26x26": 93,
    "L105_13x13": 105,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def forward_capture(model, x):
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
            x = mo[0](x, IMG_SIZE)
        if md["type"] == "convolutional":
            caps[i] = x.detach().clone()
        los.append(x)
    return caps, x

def get_person_dets(output, conf=0.3):
    dets = []
    if output is None:
        return dets
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf and int(row[5]) == 0:
            cx, cy = float(row[0]), float(row[1])
            w, h = float(row[2]), float(row[3])
            dets.append({"cx": cx, "cy": cy, "w": w, "h": h, "conf": float(row[4])})
    return dets

def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return arr

def extract_embedding(caps, layer_idx, spatial_x, spatial_y):
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = int(spatial_x / IMG_SIZE * fW)
    fy = int(spatial_y / IMG_SIZE * fH)
    fx = max(0, min(fW - 1, fx))
    fy = max(0, min(fH - 1, fy))
    return feat[0, :, fy, fx].cpu().numpy()

def composite_patch(base_img, patch_rgb, cx, cy, ps):
    """Composite patch onto base image at person center."""
    H, W = base_img.shape[2], base_img.shape[3]
    
    # Circular mask with soft edge
    yy, xx = torch.meshgrid(
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        torch.arange(ps, device=base_img.device, dtype=torch.float32),
        indexing="ij"
    )
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
    
    full_patch = torch.zeros_like(base_img)
    full_mask = torch.zeros(1, 1, H, W, device=base_img.device, dtype=base_img.dtype)
    
    full_patch[:, :, py0:py1, px0:px1] = patch_rgb[:, :, sy0:sy1, sx0:sx1]
    full_mask[:, :, py0:py1, px0:px1] = mask[:, :, sy0:sy1, sx0:sx1]
    
    # Same amplitude as optimizer: (patch - 0.5) * mask * 0.3
    composited = base_img + (full_patch - 0.5) * full_mask * 0.3
    composited = torch.clamp(composited, 0.0, 1.0)
    return composited


def main():
    print(f"Device: {DEVICE}")
    
    # Load model
    print("Loading YOLOv3...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # Load optimized patch
    print(f"Loading patch: {PATCH_PATH}")
    patch_pil = Image.open(PATCH_PATH).convert("RGB")
    patch_arr = np.array(patch_pil, dtype=np.float32) / 255.0
    patch_tensor = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # Resize to 416-space patch size
    patch_416 = F.interpolate(patch_tensor, size=(PATCH_SIZE_416, PATCH_SIZE_416),
                               mode='bilinear', align_corners=False)
    print(f"  Patch shape: {patch_416.shape}")
    
    # Process all COCO images
    image_files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    print(f"\nProcessing {len(image_files)} images...")
    
    # Collect per-person data
    clean_embs_all = {name: [] for name in DETECTION_LAYERS}
    patched_embs_all = {name: [] for name in DETECTION_LAYERS}
    person_info = []
    
    for idx, fname in enumerate(image_files):
        path = os.path.join(COCO_DIR, fname)
        arr = load_image(path, IMG_SIZE)
        base_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Clean pass
        with torch.no_grad():
            clean_caps, clean_output = forward_capture(model, base_tensor)
            clean_dets = get_person_dets(clean_output, conf=0.3)
        
        if not clean_dets:
            continue
        
        for pi, d in enumerate(clean_dets):
            # Extract clean embeddings
            clean_vecs = {}
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                emb = extract_embedding(clean_caps, layer_idx, d["cx"], d["cy"])
                clean_vecs[layer_name] = emb
                clean_embs_all[layer_name].append(emb)
            
            # Apply patch at person center and re-run
            with torch.no_grad():
                composited = composite_patch(base_tensor, patch_416, d["cx"], d["cy"], PATCH_SIZE_416)
                patched_caps, patched_output = forward_capture(model, composited)
                patched_dets = get_person_dets(patched_output, conf=0.05)
            
            # Find the patched detection closest to this person
            patched_person = None
            for pd in patched_dets:
                pdist = math.sqrt((pd["cx"] - d["cx"])**2 + (pd["cy"] - d["cy"])**2)
                if pdist < 60:
                    patched_person = pd
                    break
            
            if patched_person is None:
                # Person suppressed by patch - still extract embedding at original location
                patched_person = d  # use original coords
            
            # Extract patched embeddings at same spatial location
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                emb = extract_embedding(patched_caps, layer_idx, d["cx"], d["cy"])
                patched_embs_all[layer_name].append(emb)
            
            person_info.append({
                "image": fname,
                "person_idx": pi,
                "cx": d["cx"], "cy": d["cy"],
                "conf": d["conf"],
                "patched_conf": patched_person["conf"] if patched_person else 0.0,
                "suppressed": patched_person is None,
            })
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(image_files)} images processed")
    
    N = len(person_info)
    print(f"\nTotal persons: {N}")
    
    # ============================================================
    # Part 1: Embedding Distribution Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 1: EMBEDDING DISTRIBUTION")
    print(f"{'='*70}")
    
    distribution = {}
    for layer_name in DETECTION_LAYERS:
        embs = np.array(clean_embs_all[layer_name])  # (N, C)
        centroid = np.mean(embs, axis=0)
        dists = np.linalg.norm(embs - centroid, axis=1)
        
        # Pairwise distances (sample 100 if large)
        if N > 100:
            idx_s = np.random.choice(N, min(100, N), replace=False)
            pw = []
            for i in idx_s:
                for j in idx_s:
                    if i < j:
                        pw.append(np.linalg.norm(embs[i] - embs[j]))
            pairwise = np.array(pw)
        else:
            pairwise = np.array([np.linalg.norm(embs[i] - embs[j])
                                for i in range(N) for j in range(i+1, N)])
        
        distribution[layer_name] = {
            "N": N,
            "C": embs.shape[1],
            "centroid_norm": float(np.linalg.norm(centroid)),
            "mean_dist_to_centroid": float(np.mean(dists)),
            "std_dist_to_centroid": float(np.std(dists)),
            "median_dist_to_centroid": float(np.median(dists)),
            "min_dist": float(np.min(dists)),
            "max_dist": float(np.max(dists)),
            "p25": float(np.percentile(dists, 25)),
            "p75": float(np.percentile(dists, 75)),
            "p90": float(np.percentile(dists, 90)),
            "mean_pairwise": float(np.mean(pairwise)),
            "median_pairwise": float(np.median(pairwise)),
        }
        np.save(os.path.join(OUTPUT_DIR, f"centroid_{layer_name}.npy"), centroid)
        np.save(os.path.join(OUTPUT_DIR, f"clean_embs_{layer_name}.npy"), embs)
        np.save(os.path.join(OUTPUT_DIR, f"dist_to_centroid_{layer_name}.npy"), dists)
    
    print(f"\n  {'Layer':15s} {'Mean dist':>10s} {'Std':>8s} {'Median':>10s} {'Min':>8s} {'Max':>8s} {'P25':>8s} {'P75':>8s} {'P90':>8s}")
    print(f"  {'-'*95}")
    for ln in DETECTION_LAYERS:
        r = distribution[ln]
        print(f"  {ln:15s} {r['mean_dist_to_centroid']:10.2f} {r['std_dist_to_centroid']:8.2f} "
              f"{r['median_dist_to_centroid']:10.2f} {r['min_dist']:8.2f} {r['max_dist']:8.2f} "
              f"{r['p25']:8.2f} {r['p75']:8.2f} {r['p90']:8.2f}")
    
    print(f"\n  {'Layer':15s} {'Mean pairwise':>14s} {'Median pairwise':>16s}")
    print(f"  {'-'*50}")
    for ln in DETECTION_LAYERS:
        r = distribution[ln]
        print(f"  {ln:15s} {r['mean_pairwise']:14.2f} {r['median_pairwise']:16.2f}")
    
    # ============================================================
    # Part 2: Cross-Wearer Alignment Test
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 2: CROSS-WEARER ALIGNMENT")
    print(f"{'='*70}")
    
    alignment = {}
    for layer_name in DETECTION_LAYERS:
        clean = np.array(clean_embs_all[layer_name])  # (N, C)
        patched = np.array(patched_embs_all[layer_name])  # (N, C)
        
        # Corruption vectors: delta = patched - clean
        deltas = patched - clean  # (N, C)
        
        # L2 norms of corruption
        l2_norms = np.linalg.norm(deltas, axis=1)  # (N,)
        
        # Cosine similarity between all pairs of delta vectors
        # This measures: do all wearers get pushed in the same direction?
        if N > 200:
            idx_s = np.random.choice(N, 200, replace=False)
            deltas_sample = deltas[idx_s]
        else:
            deltas_sample = deltas
        
        M = len(deltas_sample)
        cos_matrix = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                ni, nj = np.linalg.norm(deltas_sample[i]), np.linalg.norm(deltas_sample[j])
                if ni > 1e-8 and nj > 1e-8:
                    cos_matrix[i, j] = np.dot(deltas_sample[i], deltas_sample[j]) / (ni * nj)
        
        # Upper triangle (excluding diagonal)
        upper = cos_matrix[np.triu_indices(M, k=1)]
        
        # Mean corruption vector (the "average push direction")
        mean_delta = np.mean(deltas, axis=0)
        mean_delta_norm = np.linalg.norm(mean_delta)
        
        # Cosine of each delta with the mean delta
        cos_with_mean = []
        for i in range(N):
            ni = np.linalg.norm(deltas[i])
            if ni > 1e-8 and mean_delta_norm > 1e-8:
                cos_with_mean.append(np.dot(deltas[i], mean_delta) / (ni * mean_delta_norm))
        cos_with_mean = np.array(cos_with_mean)
        
        # Does the corruption point toward the centroid?
        centroid = np.load(os.path.join(OUTPUT_DIR, f"centroid_{layer_name}.npy"))
        # Direction from clean emb to centroid
        to_centroid = centroid - clean  # (N, C)
        # Cosine between corruption direction and centroid direction
        cos_to_centroid = []
        for i in range(N):
            nd = np.linalg.norm(deltas[i])
            nc = np.linalg.norm(to_centroid[i])
            if nd > 1e-8 and nc > 1e-8:
                cos_to_centroid.append(np.dot(deltas[i], to_centroid[i]) / (nd * nc))
        cos_to_centroid = np.array(cos_to_centroid)
        
        alignment[layer_name] = {
            "mean_l2": float(np.mean(l2_norms)),
            "std_l2": float(np.std(l2_norms)),
            "median_l2": float(np.median(l2_norms)),
            "min_l2": float(np.min(l2_norms)),
            "max_l2": float(np.max(l2_norms)),
            "mean_pairwise_cos": float(np.mean(upper)),
            "std_pairwise_cos": float(np.std(upper)),
            "median_pairwise_cos": float(np.median(upper)),
            "frac_cos_pos": float(np.mean(upper > 0)),
            "frac_cos_gt_0.5": float(np.mean(upper > 0.5)),
            "mean_cos_with_mean": float(np.mean(cos_with_mean)),
            "median_cos_with_mean": float(np.median(cos_with_mean)),
            "mean_cos_to_centroid": float(np.mean(cos_to_centroid)),
            "median_cos_to_centroid": float(np.median(cos_to_centroid)),
            "frac_toward_centroid": float(np.mean(cos_to_centroid > 0)),
            "mean_delta_norm": float(mean_delta_norm),
        }
        
        np.save(os.path.join(OUTPUT_DIR, f"deltas_{layer_name}.npy"), deltas)
        np.save(os.path.join(OUTPUT_DIR, f"l2_norms_{layer_name}.npy"), l2_norms)
        np.save(os.path.join(OUTPUT_DIR, f"cos_matrix_{layer_name}.npy"), cos_matrix)
    
    print(f"\n  L2 shift per wearer (patch applied to each person):")
    print(f"  {'Layer':15s} {'Mean L2':>10s} {'Std':>8s} {'Median':>10s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*65}")
    for ln in DETECTION_LAYERS:
        r = alignment[ln]
        print(f"  {ln:15s} {r['mean_l2']:10.2f} {r['std_l2']:8.2f} {r['median_l2']:10.2f} {r['min_l2']:8.2f} {r['max_l2']:8.2f}")
    
    print(f"\n  Cross-wearer cosine alignment (do corruption vectors point same direction?):")
    print(f"  {'Layer':15s} {'Mean cos':>10s} {'Std':>8s} {'Median':>10s} {'%pos':>8s} {'%>0.5':>8s}")
    print(f"  {'-'*65}")
    for ln in DETECTION_LAYERS:
        r = alignment[ln]
        print(f"  {ln:15s} {r['mean_pairwise_cos']:10.4f} {r['std_pairwise_cos']:8.4f} "
              f"{r['median_pairwise_cos']:10.4f} {r['frac_cos_pos']*100:7.1f}% {r['frac_cos_gt_0.5']*100:7.1f}%")
    
    print(f"\n  Cosine with mean corruption direction:")
    print(f"  {'Layer':15s} {'Mean cos':>10s} {'Median':>10s}")
    print(f"  {'-'*40}")
    for ln in DETECTION_LAYERS:
        r = alignment[ln]
        print(f"  {ln:15s} {r['mean_cos_with_mean']:10.4f} {r['median_cos_with_mean']:10.4f}")
    
    print(f"\n  Does corruption point toward centroid?")
    print(f"  {'Layer':15s} {'Mean cos':>10s} {'Median':>10s} {'%toward':>10s}")
    print(f"  {'-'*45}")
    for ln in DETECTION_LAYERS:
        r = alignment[ln]
        print(f"  {ln:15s} {r['mean_cos_to_centroid']:10.4f} {r['median_cos_to_centroid']:10.4f} {r['frac_toward_centroid']*100:9.1f}%")
    
    # ============================================================
    # Part 3: Collision Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 3: COLLISION FEASIBILITY")
    print(f"{'='*70}")
    
    for layer_name in DETECTION_LAYERS:
        r_d = distribution[layer_name]
        r_a = alignment[layer_name]
        mean_d = r_d["mean_dist_to_centroid"]
        mean_shift = r_a["mean_l2"]
        
        print(f"\n  {layer_name}:")
        print(f"    Mean dist to centroid: {mean_d:.2f}")
        print(f"    Mean patch L2 shift:   {mean_shift:.2f}")
        print(f"    Ratio (shift/dist):    {mean_shift/mean_d:.2f}")
        
        if mean_shift >= mean_d:
            print(f"    -> COLLISION ACHIEVABLE: patch shift exceeds distance to centroid")
            print(f"       Most wearers can be pulled to centroid if direction aligns")
        elif mean_shift >= mean_d * 0.5:
            print(f"    -> TIGHT CLUSTER: patch covers >50% of distance to centroid")
            print(f"       Significant compression of embedding space")
        else:
            print(f"    -> PARTIAL: patch covers {mean_shift/mean_d*100:.0f}% of distance to centroid")
            print(f"       Some compression but no collision")
    
    # ============================================================
    # Part 4: Suppression stats
    # ============================================================
    suppressed = sum(1 for p in person_info if p["suppressed"])
    conf_drops = [p["conf"] - p["patched_conf"] for p in person_info if not p["suppressed"]]
    
    print(f"\n{'='*70}")
    print("SUPPRESSION STATS")
    print(f"{'='*70}")
    print(f"  Total persons: {N}")
    print(f"  Suppressed by patch: {suppressed} ({suppressed/N*100:.1f}%)")
    print(f"  Still detected: {N-suppressed} ({(N-suppressed)/N*100:.1f}%)")
    if conf_drops:
        print(f"  Mean confidence drop: {np.mean(conf_drops):.3f}")
        print(f"  Median confidence drop: {np.median(conf_drops):.3f}")
    
    # ============================================================
    # Save results
    # ============================================================
    results = {
        "distribution": distribution,
        "alignment": alignment,
        "suppression": {
            "total": N,
            "suppressed": suppressed,
            "suppression_rate": suppressed / N,
            "mean_conf_drop": float(np.mean(conf_drops)) if conf_drops else 0.0,
        },
        "person_info": person_info,
    }
    results_path = os.path.join(OUTPUT_DIR, "alignment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")
    
    # ============================================================
    # Plots
    # ============================================================
    
    # Plot 1: Distance to centroid histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, ln in enumerate(DETECTION_LAYERS):
        dists = np.load(os.path.join(OUTPUT_DIR, f"dist_to_centroid_{ln}.npy"))
        l2s = np.load(os.path.join(OUTPUT_DIR, f"l2_norms_{ln}.npy"))
        axes[ax_idx].hist(dists, bins=25, color="steelblue", edgecolor="black", alpha=0.6, label="Dist to centroid")
        axes[ax_idx].hist(l2s, bins=25, color="orange", edgecolor="black", alpha=0.6, label="Patch L2 shift")
        axes[ax_idx].axvline(x=np.mean(dists), color="blue", linestyle="-", linewidth=2, label=f"Mean dist={np.mean(dists):.1f}")
        axes[ax_idx].axvline(x=np.mean(l2s), color="red", linestyle="-", linewidth=2, label=f"Mean shift={np.mean(l2s):.1f}")
        axes[ax_idx].set_xlabel("L2 Distance")
        axes[ax_idx].set_ylabel("Count")
        axes[ax_idx].set_title(f"{ln} (N={len(dists)})")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dist_vs_shift_hist.png"), dpi=150)
    plt.close()
    
    # Plot 2: Cosine alignment matrix heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, ln in enumerate(DETECTION_LAYERS):
        cos_mat = np.load(os.path.join(OUTPUT_DIR, f"cos_matrix_{ln}.npy"))
        im = axes[ax_idx].imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[ax_idx].set_title(f"{ln} Corruption Direction Cosine")
        axes[ax_idx].set_xlabel("Wearer j")
        axes[ax_idx].set_ylabel("Wearer i")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cosine_alignment_heatmap.png"), dpi=150)
    plt.close()
    
    # Plot 3: PCA of clean vs patched embeddings
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, ln in enumerate(DETECTION_LAYERS):
        clean = np.load(os.path.join(OUTPUT_DIR, f"clean_embs_{ln}.npy"))
        patched = np.array(patched_embs_all[ln])
        centroid = np.load(os.path.join(OUTPUT_DIR, f"centroid_{ln}.npy"))
        
        all_embs = np.vstack([clean, patched, centroid.reshape(1, -1)])
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_embs)
        
        n = len(clean)
        axes[ax_idx].scatter(all_2d[:n, 0], all_2d[:n, 1], c="steelblue", alpha=0.5, s=15, label="Clean")
        axes[ax_idx].scatter(all_2d[n:2*n, 0], all_2d[n:2*n, 1], c="red", alpha=0.5, s=15, label="Patched")
        axes[ax_idx].scatter(all_2d[2*n, 0], all_2d[2*n, 1], c="green", s=100, marker="*", label="Centroid", zorder=5)
        
        # Draw arrows from clean to patched for first 20
        for i in range(min(20, n)):
            axes[ax_idx].annotate("", xy=all_2d[n+i], xytext=all_2d[i],
                                 arrowprops=dict(arrowstyle="->", color="orange", alpha=0.5, lw=0.5))
        
        axes[ax_idx].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        axes[ax_idx].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        axes[ax_idx].set_title(f"{ln} Clean vs Patched")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_clean_vs_patched.png"), dpi=150)
    plt.close()
    
    print(f"  Plots: dist_vs_shift_hist.png, cosine_alignment_heatmap.png, pca_clean_vs_patched.png")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
