"""
Measure person embedding distribution in YOLOv3 feature space.

For each COCO person image:
1. Run YOLOv3, detect all persons
2. Extract embeddings at each person's spatial location in L81/L93/L105
3. Collect all person embeddings across all images

Then compute:
- Centroid of person embedding distribution (per layer)
- Average distance from each person to centroid
- Std of distances
- Histogram of distances

This tells us whether L2_patch (~11-28) is enough to reach the centroid
(collision = identity merge) or just tightens the cluster.
"""
import os, sys, json, math
import numpy as np
import torch
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
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\embedding_distribution"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416

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


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading YOLOv3...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  Model loaded.")

    all_embeddings = {name: [] for name in DETECTION_LAYERS}
    person_metadata = []

    image_files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    print(f"\nProcessing {len(image_files)} COCO person images...")

    total_persons = 0
    for idx, fname in enumerate(image_files):
        path = os.path.join(COCO_DIR, fname)
        arr = load_image(path, IMG_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            caps, output = forward_capture(model, tensor)
            dets = get_person_dets(output, conf=0.3)

        if not dets:
            continue

        for pi, d in enumerate(dets):
            for layer_name, layer_idx in DETECTION_LAYERS.items():
                emb = extract_embedding(caps, layer_idx, d["cx"], d["cy"])
                all_embeddings[layer_name].append(emb)
            person_metadata.append({
                "image": fname,
                "person_idx": pi,
                "cx": d["cx"], "cy": d["cy"],
                "w": d["w"], "h": d["h"],
                "conf": d["conf"],
            })
            total_persons += 1

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(image_files)} images, {total_persons} persons so far")

    print(f"\nTotal persons detected: {total_persons}")

    results = {}
    for layer_name in DETECTION_LAYERS:
        embs = np.array(all_embeddings[layer_name])
        N, C = embs.shape
        centroid = np.mean(embs, axis=0)

        dists = np.linalg.norm(embs - centroid, axis=1)

        # Pairwise distances (sample if too many)
        if N > 100:
            idx_sample = np.random.choice(N, 100, replace=False)
            pairwise = []
            for i in idx_sample:
                for j in idx_sample:
                    if i < j:
                        pairwise.append(np.linalg.norm(embs[i] - embs[j]))
            pairwise = np.array(pairwise)
        else:
            pairwise = []
            for i in range(N):
                for j in range(i+1, N):
                    pairwise.append(np.linalg.norm(embs[i] - embs[j]))
            pairwise = np.array(pairwise)

        results[layer_name] = {
            "N": N,
            "C": C,
            "centroid_norm": float(np.linalg.norm(centroid)),
            "mean_dist_to_centroid": float(np.mean(dists)),
            "std_dist_to_centroid": float(np.std(dists)),
            "median_dist_to_centroid": float(np.median(dists)),
            "min_dist_to_centroid": float(np.min(dists)),
            "max_dist_to_centroid": float(np.max(dists)),
            "p25_dist_to_centroid": float(np.percentile(dists, 25)),
            "p75_dist_to_centroid": float(np.percentile(dists, 75)),
            "p90_dist_to_centroid": float(np.percentile(dists, 90)),
            "mean_pairwise_dist": float(np.mean(pairwise)),
            "std_pairwise_dist": float(np.std(pairwise)),
            "median_pairwise_dist": float(np.median(pairwise)),
        }

        np.save(os.path.join(OUTPUT_DIR, f"centroid_{layer_name}.npy"), centroid)
        np.save(os.path.join(OUTPUT_DIR, f"embeddings_{layer_name}.npy"), embs)
        np.save(os.path.join(OUTPUT_DIR, f"dists_to_centroid_{layer_name}.npy"), dists)

    results_path = os.path.join(OUTPUT_DIR, "distribution_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    meta_path = os.path.join(OUTPUT_DIR, "person_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(person_metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("PERSON EMBEDDING DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Total persons: {total_persons} from {len(image_files)} images")
    print(f"\n  {'Layer':15s} {'Mean dist':>10s} {'Std':>8s} {'Median':>10s} {'Min':>8s} {'Max':>8s} {'P25':>8s} {'P75':>8s} {'P90':>8s}")
    print(f"  {'-'*95}")
    for layer_name in DETECTION_LAYERS:
        r = results[layer_name]
        print(f"  {layer_name:15s} {r['mean_dist_to_centroid']:10.2f} {r['std_dist_to_centroid']:8.2f} "
              f"{r['median_dist_to_centroid']:10.2f} {r['min_dist_to_centroid']:8.2f} {r['max_dist_to_centroid']:8.2f} "
              f"{r['p25_dist_to_centroid']:8.2f} {r['p75_dist_to_centroid']:8.2f} {r['p90_dist_to_centroid']:8.2f}")

    print(f"\n  {'Layer':15s} {'Mean pairwise':>14s} {'Std pairwise':>14s} {'Median pairwise':>16s}")
    print(f"  {'-'*65}")
    for layer_name in DETECTION_LAYERS:
        r = results[layer_name]
        print(f"  {layer_name:15s} {r['mean_pairwise_dist']:14.2f} {r['std_pairwise_dist']:14.2f} {r['median_pairwise_dist']:16.2f}")

    print(f"\n{'='*70}")
    print("COLLISION ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  If patch shifts embedding by L2_shift toward centroid:")
    print(f"  {'Layer':15s} {'Mean dist':>10s} {'L2=11':>10s} {'L2=20':>10s} {'L2=28':>10s} {'L2=50':>10s}")
    print(f"  {'-'*65}")
    patch_shifts = [11, 20, 28, 50]
    for layer_name in DETECTION_LAYERS:
        r = results[layer_name]
        mean_d = r["mean_dist_to_centroid"]
        row = f"  {layer_name:15s} {mean_d:10.2f}"
        dists = np.load(os.path.join(OUTPUT_DIR, f"dists_to_centroid_{layer_name}.npy"))
        for shift in patch_shifts:
            frac = float(np.mean(dists < shift))
            row += f" {frac*100:8.1f}%"
        print(row)

    print(f"\n  Interpretation:")
    for layer_name in DETECTION_LAYERS:
        r = results[layer_name]
        mean_d = r["mean_dist_to_centroid"]
        if mean_d < 11:
            print(f"    {layer_name}: MEAN dist={mean_d:.1f} < L2=11. Most people reachable. COLLISION ACHIEVABLE.")
        elif mean_d < 28:
            print(f"    {layer_name}: MEAN dist={mean_d:.1f}. L2=28 patch reaches many. TIGHT CLUSTER likely.")
        else:
            print(f"    {layer_name}: MEAN dist={mean_d:.1f}. L2=28 patch only tightens. PARTIAL CONVERGENCE.")

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, layer_name in enumerate(DETECTION_LAYERS):
        dists = np.load(os.path.join(OUTPUT_DIR, f"dists_to_centroid_{layer_name}.npy"))
        axes[ax_idx].hist(dists, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[ax_idx].axvline(x=11, color="orange", linestyle="--", linewidth=2, label="L2=11 (Profile B)")
        axes[ax_idx].axvline(x=28, color="red", linestyle="--", linewidth=2, label="L2=28 (gradient optim)")
        axes[ax_idx].axvline(x=np.mean(dists), color="green", linestyle="-", linewidth=2, label=f"Mean={np.mean(dists):.1f}")
        axes[ax_idx].set_xlabel("L2 Distance to Centroid")
        axes[ax_idx].set_ylabel("Count")
        axes[ax_idx].set_title(f"{layer_name} (N={len(dists)})")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dist_to_centroid_hist.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Histogram: {plot_path}")

    # PCA visualization
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, layer_name in enumerate(DETECTION_LAYERS):
        embs = np.load(os.path.join(OUTPUT_DIR, f"embeddings_{layer_name}.npy"))
        centroid = np.load(os.path.join(OUTPUT_DIR, f"centroid_{layer_name}.npy"))
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(embs)
        centroid_2d = pca.transform(centroid.reshape(1, -1))
        axes[ax_idx].scatter(embs_2d[:, 0], embs_2d[:, 1], c="steelblue", alpha=0.6, s=20, label="Persons")
        axes[ax_idx].scatter(centroid_2d[0, 0], centroid_2d[0, 1], c="red", s=100, marker="*", label="Centroid", zorder=5)
        mean_d = results[layer_name]["mean_dist_to_centroid"]
        scale = np.mean(np.linalg.norm(embs_2d - centroid_2d, axis=1)) / mean_d if mean_d > 0 else 1
        circle = plt.Circle(centroid_2d[0], mean_d * scale, fill=False, color="red", linestyle="--", alpha=0.5, label=f"Mean dist={mean_d:.1f}")
        axes[ax_idx].add_patch(circle)
        axes[ax_idx].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        axes[ax_idx].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        axes[ax_idx].set_title(f"{layer_name} Embedding Space (PCA)")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(OUTPUT_DIR, "embedding_pca.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"  PCA plot: {pca_path}")

    print(f"\n  Results: {results_path}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
