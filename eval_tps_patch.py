"""
Full evaluation of the TPS-aligned patch:
- Human suppression rate and confidence drop
- Embedding corruption (L2 + centroid cosine)
- Collateral damage to non-human detections
- New false detections introduced by patching
- Class-level detection count shifts
"""
import os, sys, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import Counter

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _t
sys.modules["imgaug"] = _t.ModuleType("imgaug")
from pytorchyolo.models import Darknet

COCO = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

CFG = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WTS = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
COCO_DIR = r"C:\Users\carso\Desktop\YODO\data\coco_person\images"
PATCH_PATH = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim_tps\tps_aligned_patch.png"
CENT_DIR = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\alignment_test"
OUTPUT_DIR = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\patch_pipeline\aligned_optim_tps"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 416
PS = 80
DET_LAYERS = {"L81": 81, "L93": 93, "L105": 105}
N_EVAL = 200

def fwd(model, x):
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
            caps[i] = x
        los.append(x)
    return caps, x

def load_img(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    return np.array(c, dtype=np.float32) / 255.0

def get_dets(output, conf=0.25):
    dets = []
    if output is None:
        return dets
    o = output.cpu().numpy()
    if o.ndim == 3:
        o = o[0]
    for row in o:
        if len(row) >= 6 and row[4] >= conf:
            dets.append({"cls": int(row[5]), "conf": float(row[4]),
                        "cx": float(row[0]), "cy": float(row[1]),
                        "w": float(row[2]), "h": float(row[3])})
    return dets

def composite(base, patch_416, cx, cy, ps):
    H, W = base.shape[2], base.shape[3]
    yy, xx = torch.meshgrid(
        torch.arange(ps, device=base.device, dtype=torch.float32),
        torch.arange(ps, device=base.device, dtype=torch.float32),
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
    fp = torch.zeros_like(base)
    fm = torch.zeros(1, 1, H, W, device=base.device, dtype=base.dtype)
    fp[:, :, py0:py1, px0:px1] = patch_416[:, :, sy0:sy1, sx0:sx1]
    fm[:, :, py0:py1, px0:px1] = mask[:, :, sy0:sy1, sx0:sx1]
    return torch.clamp(base + (fp - 0.5) * fm * 0.3, 0.0, 1.0)

def extract_emb(caps, layer_idx, sx, sy):
    feat = caps[layer_idx]
    fH, fW = feat.shape[2], feat.shape[3]
    fx = max(0, min(fW-1, int(sx / IMG_SIZE * fW)))
    fy = max(0, min(fH-1, int(sy / IMG_SIZE * fH)))
    return feat[0, :, fy, fx]

def main():
    print(f"Device: {DEV}")
    print("Loading YOLOv3...")
    model = Darknet(CFG).to(DEV)
    model.load_darknet_weights(WTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load patch
    patch_pil = Image.open(PATCH_PATH).convert("RGB")
    patch_arr = np.array(patch_pil, dtype=np.float32) / 255.0
    patch_t = torch.from_numpy(patch_arr).permute(2, 0, 1).unsqueeze(0).to(DEV)
    patch_416 = F.interpolate(patch_t, size=(PS, PS), mode="bilinear", align_corners=False)
    print(f"Patch loaded: {PATCH_PATH}")

    # Load centroids
    centroids = {}
    for ln in DET_LAYERS:
        cp = None
        for suffix in ["52x52", "26x26", "13x13"]:
            cp2 = os.path.join(CENT_DIR, f"centroid_{ln}_{suffix}.npy")
            if os.path.exists(cp2):
                cp = cp2
                break
        if cp:
            centroids[ln] = torch.from_numpy(np.load(cp)).to(DEV)
            print(f"  Centroid {ln}: loaded")
        else:
            centroids[ln] = None

    files = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    random.seed(42)
    random.shuffle(files)

    stats = {
        "n_images": 0,
        "n_persons_clean": 0,
        "n_persons_patched": 0,
        "suppressed_count": 0,
        "person_confs_clean": [],
        "person_confs_patched": [],
        "n_nonhuman_clean": 0,
        "n_nonhuman_patched": 0,
        "nonhuman_confs_clean": [],
        "nonhuman_confs_patched": [],
        "l2_shifts": {ln: [] for ln in DET_LAYERS},
        "cos_centroid": {ln: [] for ln in DET_LAYERS},
        "new_detections": [],
        "class_shifts": {},
    }

    n_persons = 0
    for idx, fname in enumerate(files):
        if n_persons >= N_EVAL:
            break
        path = os.path.join(COCO_DIR, fname)
        arr = load_img(path, IMG_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEV)
        with torch.no_grad():
            caps_clean, out_clean = fwd(model, tensor)
        clean_dets = get_dets(out_clean, conf=0.25)
        persons = [d for d in clean_dets if d["cls"] == 0]
        if not persons:
            continue

        stats["n_images"] += 1
        nh_clean = [d for d in clean_dets if d["cls"] != 0]
        stats["n_nonhuman_clean"] += len(nh_clean)
        for d in nh_clean:
            stats["nonhuman_confs_clean"].append(d["conf"])
            c = d["cls"]
            if c not in stats["class_shifts"]:
                stats["class_shifts"][c] = [0, 0]
            stats["class_shifts"][c][0] += 1

        for p in persons:
            if n_persons >= N_EVAL:
                break
            cx, cy = p["cx"], p["cy"]
            n_persons += 1
            stats["n_persons_clean"] += 1
            stats["person_confs_clean"].append(p["conf"])

            clean_embs = {}
            for ln, li in DET_LAYERS.items():
                clean_embs[ln] = extract_emb(caps_clean, li, cx, cy).clone()

            comp = composite(tensor, patch_416, cx, cy, PS)
            with torch.no_grad():
                caps_patch, out_patch = fwd(model, comp)
            patch_dets = get_dets(out_patch, conf=0.25)

            wearer = [d for d in patch_dets if d["cls"] == 0 and
                      math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2) < 60]
            if wearer:
                stats["n_persons_patched"] += 1
                stats["person_confs_patched"].append(wearer[0]["conf"])
            else:
                stats["suppressed_count"] += 1

            for ln, li in DET_LAYERS.items():
                pvec = extract_emb(caps_patch, li, cx, cy)
                delta = pvec - clean_embs[ln]
                l2 = torch.norm(delta).item()
                stats["l2_shifts"][ln].append(l2)
                if centroids[ln] is not None:
                    to_c = centroids[ln] - clean_embs[ln]
                    cos = F.cosine_similarity(delta.unsqueeze(0), to_c.unsqueeze(0)).squeeze().item()
                    stats["cos_centroid"][ln].append(cos)

            nh_patch = [d for d in patch_dets if d["cls"] != 0]
            stats["n_nonhuman_patched"] += len(nh_patch)
            for d in nh_patch:
                stats["nonhuman_confs_patched"].append(d["conf"])
                c = d["cls"]
                if c not in stats["class_shifts"]:
                    stats["class_shifts"][c] = [0, 0]
                stats["class_shifts"][c][1] += 1

            for d in patch_dets:
                dist = math.sqrt((d["cx"]-cx)**2 + (d["cy"]-cy)**2)
                if dist > 60:
                    was_in_clean = any(
                        math.sqrt((cd["cx"]-d["cx"])**2 + (cd["cy"]-d["cy"])**2) < 30
                        for cd in clean_dets if cd["cls"] == d["cls"]
                    )
                    if not was_in_clean:
                        stats["new_detections"].append({
                            "cls": COCO[d["cls"]] if d["cls"] < 80 else str(d["cls"]),
                            "conf": d["conf"]
                        })

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1} images, {n_persons} persons evaluated")

    # Summary
    print()
    print("=" * 70)
    print("TPS-ALIGNED PATCH FULL EVALUATION")
    print("=" * 70)

    n_p = stats["n_persons_clean"]
    n_supp = stats["suppressed_count"]
    n_det = stats["n_persons_patched"]
    print(f"\nImages: {stats['n_images']}")
    print(f"Persons (clean): {n_p}")
    print(f"Persons still detected: {n_det} ({100*n_det/n_p:.1f}%)")
    print(f"Persons suppressed: {n_supp} ({100*n_supp/n_p:.1f}%)")

    if stats["person_confs_clean"]:
        mc = np.mean(stats["person_confs_clean"])
        med_c = np.median(stats["person_confs_clean"])
        if stats["person_confs_patched"]:
            mc_p = np.mean(stats["person_confs_patched"])
            med_p = np.median(stats["person_confs_patched"])
        else:
            mc_p = 0
            med_p = 0
        print(f"\nPerson confidence:")
        print(f"  Clean:   mean={mc:.4f}  median={med_c:.4f}")
        print(f"  Patched: mean={mc_p:.4f}  median={med_p:.4f}")
        print(f"  Drop:    {mc-mc_p:.4f} ({100*(mc-mc_p)/mc:.1f}%)")

    print(f"\nNon-human detections:")
    print(f"  Clean:   {stats['n_nonhuman_clean']}  (mean/img={stats['n_nonhuman_clean']/max(stats['n_images'],1):.2f})")
    print(f"  Patched: {stats['n_nonhuman_patched']}  (mean/img={stats['n_nonhuman_patched']/max(stats['n_images'],1):.2f})")
    if stats["nonhuman_confs_clean"]:
        print(f"  Clean conf:   mean={np.mean(stats['nonhuman_confs_clean']):.4f}")
    if stats["nonhuman_confs_patched"]:
        print(f"  Patched conf: mean={np.mean(stats['nonhuman_confs_patched']):.4f}")

    print(f"\nEmbedding corruption (detected persons only):")
    for ln in DET_LAYERS:
        if stats["l2_shifts"][ln]:
            l2s = stats["l2_shifts"][ln]
            print(f"  {ln}: L2 mean={np.mean(l2s):.2f}  std={np.std(l2s):.2f}  median={np.median(l2s):.2f}")
            if stats["cos_centroid"][ln]:
                cs = stats["cos_centroid"][ln]
                pos_pct = 100 * np.mean([c > 0 for c in cs])
                print(f"        cos_centroid mean={np.mean(cs):.4f}  %>0={pos_pct:.1f}%")

    print(f"\nNew false detections from patching: {len(stats['new_detections'])}")
    if stats["new_detections"]:
        cls_counts = Counter(d["cls"] for d in stats["new_detections"])
        for cls, cnt in cls_counts.most_common(10):
            confs = [d["conf"] for d in stats["new_detections"] if d["cls"] == cls]
            print(f"  {cls:20s}: {cnt}x  (mean conf={np.mean(confs):.3f})")

    print(f"\nClass-level shifts (clean -> patched):")
    for cls, (c_cnt, p_cnt) in sorted(stats["class_shifts"].items(), key=lambda x: x[1][0], reverse=True)[:15]:
        name = COCO[cls] if cls < 80 else str(cls)
        delta = p_cnt - c_cnt
        sign = "+" if delta > 0 else ""
        print(f"  {name:20s}: {c_cnt:4d} -> {p_cnt:4d}  ({sign}{delta})")

    # Save JSON
    results = {
        "n_images": stats["n_images"],
        "n_persons_clean": n_p,
        "n_persons_detected": n_det,
        "n_persons_suppressed": n_supp,
        "suppression_rate": n_supp / n_p,
        "person_conf_clean_mean": float(mc) if stats["person_confs_clean"] else 0,
        "person_conf_patched_mean": float(mc_p) if stats["person_confs_patched"] else 0,
        "confidence_drop": float(mc - mc_p) if stats["person_confs_clean"] else 0,
        "n_nonhuman_clean": stats["n_nonhuman_clean"],
        "n_nonhuman_patched": stats["n_nonhuman_patched"],
        "l2_shifts": {ln: {"mean": float(np.mean(stats["l2_shifts"][ln])), "std": float(np.std(stats["l2_shifts"][ln]))} for ln in DET_LAYERS if stats["l2_shifts"][ln]},
        "cos_centroid": {ln: {"mean": float(np.mean(stats["cos_centroid"][ln]))} for ln in DET_LAYERS if stats["cos_centroid"][ln]},
        "n_new_detections": len(stats["new_detections"]),
        "class_shifts": {COCO[k] if k < 80 else str(k): v for k, v in stats["class_shifts"].items()},
    }
    out_path = os.path.join(OUTPUT_DIR, "full_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
