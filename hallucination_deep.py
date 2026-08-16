"""
Deep Hallucination Analysis — full detection details, both images, composite patterns.

No shortcuts. Every detection logged with class, confidence, bbox.
Tests on both withhuman AND withouthuman images.
Multiple confidence thresholds to catch weak hallucinations.
Composite patterns combining best suppressors.
"""

import os, sys, json, csv, math
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\hallucination_deep"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128,128,128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE), arr

def forward_capture(model, x):
    caps = {}
    los = []
    for i, (md, mo) in enumerate(zip(model.module_defs, model.module_list)):
        if md["type"] in ["convolutional","upsample","maxpool"]:
            x = mo(x)
        elif md["type"] == "route":
            ls = [int(v) for v in md["layers"].split(",")]
            comb = torch.cat([los[l] for l in ls], 1)
            gs = comb.shape[1] // int(md.get("groups",1))
            gi = int(md.get("group_id",0))
            x = comb[:, gs*gi:gs*(gi+1)]
        elif md["type"] == "shortcut":
            x = los[-1] + los[int(md["from"])]
        elif md["type"] == "yolo":
            x = mo[0](x, x.size(2) if hasattr(x,'size') else 416)
        if md["type"] == "convolutional":
            caps[i] = x.detach().clone()
        los.append(x)
    return caps, x

def make_sinusoid(H, W, kx, ky, phase_deg, amp):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.cos(2*np.pi*(kx/W*x + ky/H*y) + np.radians(phase_deg))).astype(np.float32)

def add_pattern(arr, pat):
    out = arr.copy()
    for c in range(3): out[:,:,c] = np.clip(out[:,:,c] + pat, 0, 1)
    return out

def get_dets_full(model, x, conf_thresh=0.1):
    # Full detection details: class, confidence, bbox
    with torch.no_grad():
        output = model(x)
    dets = []
    if output is None:
        return dets
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf_thresh:
            cls = int(row[5])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"class_{cls}",
                "confidence": float(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "bbox_size": float((row[2]-row[0]) * (row[3]-row[1])),
            })
    return dets

def get_dets_multi_thresh(model, x):
    # Get detections at multiple confidence thresholds
    with torch.no_grad():
        output = model(x)
    if output is None:
        return {}
    out = output.cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    results = {}
    for thresh in [0.01, 0.05, 0.1, 0.25, 0.5]:
        dets = []
        for row in out:
            if len(row) >= 6 and row[4] >= thresh:
                cls = int(row[5])
                dets.append({
                    "class_id": cls,
                    "class_name": COCO_NAMES[cls] if cls < 80 else f"class_{cls}",
                    "confidence": float(row[4]),
                    "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                })
        results[f"thresh_{thresh}"] = dets
    return results

def draw_detections(arr, dets, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{d['class_name']} {d['confidence']:.2f}",
                color="lime", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5))
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("Deep Hallucination Analysis — Full Detection Details")
    print("="*70)

    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    iw, arr_w = load_image(IMG_WITH)
    iwo, arr_wo = load_image(IMG_WITHOUT)
    H, W, _ = arr_w.shape

    # Baseline detections at multiple thresholds
    print("\n--- BASELINE DETECTIONS ---")
    for name, tensor, arr in [("with_human", iw, arr_w), ("without_human", iwo, arr_wo)]:
        multi = get_dets_multi_thresh(model, tensor)
        for thresh_key, dets in multi.items():
            classes = {}
            for d in dets:
                cn = d["class_name"]
                if cn not in classes:
                    classes[cn] = []
                classes[cn].append(d["confidence"])
            class_summary = {k: {"count": len(v), "confs": [round(c,3) for c in sorted(v, reverse=True)]} for k, v in classes.items()}
            print(f"  {name} {thresh_key}: {len(dets)} dets: {class_summary}")
        # Draw baseline
        draw_detections(arr, multi["thresh_0.25"], f"Baseline {name} (conf>=0.25)",
                        f"{OUTPUT_DIR}/baseline_{name}.png")

    # Patterns to test — focus on the interesting ones from numeric_injection results
    # Include the hallucination candidate (thirteen_k208_d), best suppressors, and composites
    amp = 0.20

    test_patterns = []

    # 1. The hallucination candidate: thirteen_k208 diagonal
    test_patterns.append(("thirteen_k208_d", make_sinusoid(H, W, 208, 208, 0, amp)))

    # 2. Best prime suppressors (diagonal)
    for k in [167, 163, 179, 173, 181, 199, 197, 193, 191, 157]:
        test_patterns.append((f"prime_k{k}_d", make_sinusoid(H, W, k, k, 0, amp)))

    # 3. Best prime suppressors (horizontal) — these also suppressed
    for k in [157, 163, 167, 173, 179, 181]:
        test_patterns.append((f"prime_k{k}_h", make_sinusoid(H, W, k, 0, 0, amp)))

    # 4. Zero image — what does YOLO see in pure black?
    test_patterns.append(("zero_image", np.zeros((H, W), dtype=np.float32)))

    # 5. Half image — what does YOLO see in uniform gray?
    test_patterns.append(("half_image", np.full((H, W), 0.5, dtype=np.float32)))

    # 6. White image
    test_patterns.append(("white_image", np.ones((H, W), dtype=np.float32)))

    # 7. Random noise — does YOLO hallucinate from noise?
    rng = np.random.RandomState(42)
    noise = (rng.randn(H, W) * amp).astype(np.float32)
    test_patterns.append(("random_noise", noise))

    # 8. Composite: thirteen_k208_d + prime_k167_d (best suppressor + hallucination candidate)
    comp1 = make_sinusoid(H, W, 208, 208, 0, amp) + make_sinusoid(H, W, 167, 167, 0, amp)
    comp1 = (comp1 / np.abs(comp1).max() * amp).astype(np.float32)
    test_patterns.append(("composite_k208d_k167d", comp1))

    # 9. Composite: prime_k167_h + prime_k167_d (both orientations of best suppressor)
    comp2 = make_sinusoid(H, W, 167, 0, 0, amp) + make_sinusoid(H, W, 0, 167, 0, amp)
    comp2 = (comp2 / np.abs(comp2).max() * amp).astype(np.float32)
    test_patterns.append(("composite_k167_hv", comp2))

    # 10. Composite: multiple high primes stacked
    comp3 = np.zeros((H, W), dtype=np.float32)
    for k in [157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        comp3 += make_sinusoid(H, W, k, k, 0, amp/10)
    test_patterns.append(("composite_all_high_primes_d", comp3))

    # 11. Lychrel k=196 diagonal (was a suppressor)
    test_patterns.append(("lychrel_k196d196", make_sinusoid(H, W, 196, 196, 0, amp)))

    # 12. Architecture-aligned k=32 diagonal (best power-of-2)
    test_patterns.append(("arch_k32_d", make_sinusoid(H, W, 32, 32, 0, amp)))

    # 13. Low frequency k=13 diagonal (13-multiple, smallest)
    test_patterns.append(("thirteen_k13_d", make_sinusoid(H, W, 13, 13, 0, amp)))

    # 14. Very high frequency k=208 horizontal
    test_patterns.append(("thirteen_k208_h", make_sinusoid(H, W, 208, 0, 0, amp)))

    # 15. Checkerboard pattern at 13x13 grid (matches detection grid)
    checker = np.zeros((H, W), dtype=np.float32)
    cell = H // 13  # 32 pixels per cell
    for y in range(H):
        for x in range(W):
            if ((y // cell) + (x // cell)) % 2 == 0:
                checker[y, x] = amp
    test_patterns.append(("checkerboard_13x13", checker))

    # 16. Checkerboard at 26x26 grid (medium detection scale)
    cell26 = H // 26  # 16 pixels per cell
    checker26 = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if ((y // cell26) + (x // cell26)) % 2 == 0:
                checker26[y, x] = amp
    test_patterns.append(("checkerboard_26x26", checker26))

    # 17. Checkerboard at 52x52 grid (fine detection scale)
    cell52 = H // 52  # 8 pixels per cell
    checker52 = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if ((y // cell52) + (x // cell52)) % 2 == 0:
                checker52[y, x] = amp
    test_patterns.append(("checkerboard_52x52", checker52))

    # 18. Doubling sequence as uniform pixel offsets: 1/256, 2/256, 4/256, ... 256/256
    # Powers of 2 as numeric values — does the model treat these specially?
    for pwr in range(9):  # 2^0 through 2^8
        val = (2 ** pwr) / 256.0
        test_patterns.append((f"doubling_val_{2**pwr}_over256", np.full((H, W), val, dtype=np.float32)))

    # 19. 1/7 — prime not in YOLOv3 architecture anywhere
    test_patterns.append(("inv_7", np.full((H, W), 1.0/7.0, dtype=np.float32)))
    # Also 7/256, 7*13=91/256
    test_patterns.append(("seven_over_256", np.full((H, W), 7.0/256.0, dtype=np.float32)))
    test_patterns.append(("ninetyone_over_256", np.full((H, W), 91.0/256.0, dtype=np.float32)))

    # 20. 1/196 and 196 as uniform offsets — track if model closes these
    test_patterns.append(("inv_196_offset", np.full((H, W), 1.0/196.0, dtype=np.float32)))
    test_patterns.append(("val_196_over_255", np.full((H, W), 196.0/255.0, dtype=np.float32)))
    test_patterns.append(("val_196_over_256", np.full((H, W), 196.0/256.0, dtype=np.float32)))

    # 21. k=196 as spatial frequency in all orientations
    for theta_deg, (kx, ky) in [("h", (196, 0)), ("v", (0, 196)), ("d", (196, 196)), ("ad", (196, -196))]:
        test_patterns.append((f"k196_{theta_deg}", make_sinusoid(H, W, kx, ky, 0, amp)))

    # 22. 1/196 as spatial frequency — extremely low frequency, one cycle per 196 pixels
    test_patterns.append(("k1_over_196", make_sinusoid(H, W, 1, 1, 0, 1.0/196.0)))

    # 23. Composite: 1/196 offset + k196 diagonal — does the offset prevent closure of the frequency?
    comp_196 = np.full((H, W), 1.0/196.0, dtype=np.float32) + make_sinusoid(H, W, 196, 196, 0, amp)
    comp_196 = np.clip(comp_196, -amp, amp).astype(np.float32)
    test_patterns.append(("composite_inv196_offset_k196d", comp_196))

    # 24. Anti-closure: k196 pattern with amplitude exactly 1/196
    test_patterns.append(("k196d_amp_inv196", make_sinusoid(H, W, 196, 196, 0, 1.0/196.0)))

    # 25. Anti-closure: k196 pattern with amplitude 196/255
    test_patterns.append(("k196d_amp_196over255", make_sinusoid(H, W, 196, 196, 0, 196.0/255.0)))

    # Run all patterns on BOTH images
    all_results = {}
    csv_rows = []

    for img_name, tensor_base, arr_base in [("with_human", iw, arr_w), ("without_human", iwo, arr_wo)]:
        print(f"\n{'='*70}")
        print(f"TESTING ON: {img_name}")
        print(f"{'='*70}")

        img_results = {}

        for pat_name, pattern in test_patterns:
            # For constant images (zero, half, white), replace entirely
            if pat_name in ("zero_image", "half_image", "white_image") or pat_name.startswith("doubling_val_") or pat_name in ("inv_7", "seven_over_256", "ninetyone_over_256", "inv_196_offset", "val_196_over_255", "val_196_over_256"):
                arr_mod = np.stack([pattern]*3, axis=2)
            else:
                arr_mod = add_pattern(arr_base, pattern)

            tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)

            # Get full detections at multiple thresholds
            multi = get_dets_multi_thresh(model, tensor_mod)

            # Primary analysis at conf >= 0.1 (catch weak hallucinations)
            dets_01 = multi["thresh_0.1"]
            dets_25 = multi["thresh_0.25"]
            dets_001 = multi["thresh_0.01"]

            # Classify what happened
            baseline_classes = {d["class_name"] for d in get_dets_full(model, tensor_base, 0.1)}
            mod_classes = {d["class_name"] for d in dets_01}

            hallucinated = mod_classes - baseline_classes
            disappeared = baseline_classes - mod_classes

            # Count per class at 0.1 threshold
            class_counts = {}
            for d in dets_01:
                cn = d["class_name"]
                if cn not in class_counts:
                    class_counts[cn] = {"count": 0, "confs": [], "bboxes": []}
                class_counts[cn]["count"] += 1
                class_counts[cn]["confs"].append(round(d["confidence"], 4))
                class_counts[cn]["bboxes"].append([round(b, 1) for b in d["bbox"]])

            # Also at 0.01 threshold — catch everything
            class_counts_001 = {}
            for d in dets_001:
                cn = d["class_name"]
                if cn not in class_counts_001:
                    class_counts_001[cn] = 0
                class_counts_001[cn] += 1

            anomalies = []
            if len(dets_01) > 0 and len(get_dets_full(model, tensor_base, 0.1)) == 0:
                anomalies.append("HALLUCINATION_FROM_NOTHING")
            if hallucinated:
                anomalies.append(f"HALLUCINATED:{','.join(hallucinated)}")
            if len(dets_01) == 0 and len(get_dets_full(model, tensor_base, 0.1)) > 0:
                anomalies.append("TOTAL_SUPPRESSION")
            if disappeared and not hallucinated:
                anomalies.append(f"SUPPRESSED:{','.join(disappeared)}")

            img_results[pat_name] = {
                "dets_at_0.01": len(dets_001),
                "dets_at_0.1": len(dets_01),
                "dets_at_0.25": len(dets_25),
                "classes_at_0.1": class_counts,
                "classes_at_0.01": class_counts_001,
                "hallucinated_classes": list(hallucinated),
                "disappeared_classes": list(disappeared),
                "anomalies": anomalies,
                "all_detections_0.1": [{"class": d["class_name"], "conf": round(d["confidence"],4),
                                        "bbox": [round(b,1) for b in d["bbox"]]} for d in dets_01],
            }

            # Print summary
            tag = f" {'|'.join(anomalies)}" if anomalies else ""
            print(f"  {pat_name:>28}: dets(0.01)={len(dets_01):2d} dets(0.1)={len(dets_01):2d} dets(0.25)={len(dets_25):2d} "
                  f"classes={list(class_counts.keys())}{tag}")

            # Draw detections for interesting cases
            if anomalies or len(dets_01) > 5:
                draw_detections(arr_mod, dets_01,
                                f"{img_name} + {pat_name} (conf>=0.1) {tag}",
                                f"{OUTPUT_DIR}/{img_name}_{pat_name}.png")

            for csv_class, csv_data in class_counts.items():
                csv_rows.append({
                    "image": img_name, "pattern": pat_name,
                    "class": csv_class, "count": csv_data["count"],
                    "max_conf": max(csv_data["confs"]),
                    "confs": str(csv_data["confs"]),
                    "anomalies": "|".join(anomalies),
                })

        all_results[img_name] = img_results

    # ============================================================
    # CROSS-IMAGE ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-IMAGE ANALYSIS")
    print(f"{'='*70}")

    # Which patterns hallucinate on without_human but not on with_human?
    print("\n--- HALLUCINATIONS ON WITHOUT-HUMAN IMAGE ---")
    for pat_name in [p[0] for p in test_patterns]:
        wo_data = all_results["without_human"].get(pat_name, {})
        w_data = all_results["with_human"].get(pat_name, {})
        wo_halluc = wo_data.get("hallucinated_classes", [])
        wo_dets = wo_data.get("dets_at_0.1", 0)
        wo_classes = wo_data.get("classes_at_0.1", {})
        if wo_dets > 0 or wo_halluc:
            print(f"  {pat_name:>28}: {wo_dets} dets, classes={list(wo_classes.keys())}, "
                  f"halluc={wo_halluc}, anomalies={wo_data.get('anomalies', [])}")

    # Which patterns suppress person on with_human?
    print("\n--- PERSON SUPPRESSION ON WITH-HUMAN IMAGE ---")
    for pat_name in [p[0] for p in test_patterns]:
        w_data = all_results["with_human"].get(pat_name, {})
        w_classes = w_data.get("classes_at_0.1", {})
        person_count = w_classes.get("person", {}).get("count", 0)
        baseline_person = len(get_dets_full(model, iw, 0.1))
        if person_count < baseline_person:
            print(f"  {pat_name:>28}: person dets {baseline_person} -> {person_count}, "
                  f"total dets={w_data.get('dets_at_0.1', 0)}, "
                  f"anomalies={w_data.get('anomalies', [])}")

    # Which patterns cause the most diverse hallucination classes?
    print("\n--- MOST DIVERSE HALLUCINATION CLASSES ---")
    for pat_name in [p[0] for p in test_patterns]:
        for img_name in ["with_human", "without_human"]:
            data = all_results[img_name].get(pat_name, {})
            classes = data.get("classes_at_0.01", {})
            if len(classes) > 3:
                print(f"  {img_name:>14} + {pat_name:>28}: {len(classes)} classes at 0.01 thresh: {classes}")

    # ============================================================
    # EXPERIMENT: 196 Persistence — where does the model close it?
    # ============================================================
    print(f"\n{'='*70}")
    print("196 PERSISTENCE — tracking 1/196 and 196 through network layers")
    print(f"{'='*70}")

    # For each 196-related pattern, capture feature maps at all conv layers
    # and measure: does the 1/196 value survive? Does 196 survive?
    # Track: mean activation, fraction of activations near 1/196, fraction near 196,
    # fraction near 0 (closed), and the actual distribution of values
    LAYERS_TRACK = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]

    persistence_patterns = [
        ("inv_196_offset", np.full((H, W), 1.0/196.0, dtype=np.float32)),
        ("val_196_over_255", np.full((H, W), 196.0/255.0, dtype=np.float32)),
        ("k196d_amp_inv196", make_sinusoid(H, W, 196, 196, 0, 1.0/196.0)),
        ("k196d_amp_196over255", make_sinusoid(H, W, 196, 196, 0, 196.0/255.0)),
        ("composite_inv196_k196d", np.full((H, W), 1.0/196.0, dtype=np.float32) + make_sinusoid(H, W, 196, 196, 0, amp)),
        # Anti-closure: 1/196 + high-freq carrier to prevent normalization
        ("anticlose_inv196_k200d", np.full((H, W), 1.0/196.0, dtype=np.float32) + make_sinusoid(H, W, 200, 200, 0, amp)),
        # Anti-closure: 196/255 + k167 (best prime suppressor)
        ("anticlose_196over255_k167d", np.full((H, W), 196.0/255.0, dtype=np.float32) + make_sinusoid(H, W, 167, 167, 0, amp)),
        # Anti-closure: 1/196 + all high primes stacked
        ("anticlose_inv196_allprimes", np.full((H, W), 1.0/196.0, dtype=np.float32) + 
         sum(make_sinusoid(H, W, k, k, 0, amp/15) for k in [157,163,167,173,179,181,191,193,197,199])),
        # Control: plain image (no injection)
        ("control_no_injection", np.zeros((H, W), dtype=np.float32)),
    ]

    persistence_results = {}

    for pat_name, pattern in persistence_patterns:
        if pat_name == "control_no_injection":
            arr_mod = arr_w.copy()
        elif pat_name.startswith("inv_196") or pat_name.startswith("val_196") or pat_name.startswith("anticlose_inv196") or pat_name == "composite_inv196_k196d":
            # These are additive offsets + possible sinusoid
            if pattern.ndim == 2:
                arr_mod = add_pattern(arr_w, pattern)
            else:
                arr_mod = arr_w.copy()
                for c in range(3): arr_mod[:,:,c] = np.clip(arr_mod[:,:,c] + pattern, 0, 1)
        else:
            arr_mod = add_pattern(arr_w, pattern)

        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        caps_mod, _ = forward_capture(model, tensor_mod)
        caps_base, _ = forward_capture(model, iw)

        layer_track = {}
        for li in LAYERS_TRACK:
            if li not in caps_mod: continue
            fm = caps_mod[li].squeeze(0).cpu().numpy()
            fb = caps_base[li].squeeze(0).cpu().numpy() if li in caps_base else np.zeros_like(fm)
            delta = fm - fb

            # Check if 1/196 value survives anywhere
            inv196 = 1.0 / 196.0
            near_inv196 = np.mean(np.abs(delta - inv196) < 0.001) * 100
            # Check if 196 value survives anywhere
            near_196 = np.mean(np.abs(delta - 196.0) < 0.5) * 100
            # Check if values are near 0 (model closed the injection)
            near_zero = np.mean(np.abs(delta) < 0.001) * 100
            # Distribution stats
            layer_track[li] = {
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta)),
                "delta_max": float(np.max(delta)),
                "delta_min": float(np.min(delta)),
                "pct_near_inv196": float(near_inv196),
                "pct_near_196": float(near_196),
                "pct_near_zero": float(near_zero),
                "feat_mean": float(np.mean(fm)),
                "feat_std": float(np.std(fm)),
            }

        persistence_results[pat_name] = layer_track

        # Print persistence trace
        print(f"\n  {pat_name}:")
        print(f"    {'Layer':>6} {'dMean':>8} {'dStd':>8} {'dMax':>8} {'dMin':>8} {'%near0':>7} {'%near1/196':>10} {'%near196':>9}")
        for li in LAYERS_TRACK:
            if li not in layer_track: continue
            lt = layer_track[li]
            print(f"    {li:6d} {lt['delta_mean']:8.4f} {lt['delta_std']:8.4f} {lt['delta_max']:8.3f} {lt['delta_min']:8.3f} "
                  f"{lt['pct_near_zero']:7.2f} {lt['pct_near_inv196']:10.4f} {lt['pct_near_196']:9.4f}")

    # Identify where closure happens (delta goes to ~0)
    print(f"\n--- CLOSURE ANALYSIS: where does the model normalize away the injection? ---")
    for pat_name in persistence_results:
        layers = persistence_results[pat_name]
        # Find first layer where pct_near_zero > 50% (majority of delta is closed)
        closure_layer = None
        for li in sorted(layers.keys()):
            if layers[li]["pct_near_zero"] > 50:
                closure_layer = li
                break
        if closure_layer is not None:
            print(f"  {pat_name:>32}: closed at layer {closure_layer} (>50% near-zero delta)")
        else:
            # Find layer with highest near-zero percentage
            max_close = max(layers.items(), key=lambda x: x[1]["pct_near_zero"])
            print(f"  {pat_name:>32}: never fully closed, max {max_close[1]['pct_near_zero']:.1f}% near-zero at L{max_close[0]}")

    # Plot: persistence trace for key patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    key_pats = ["inv_196_offset", "k196d_amp_inv196", "anticlose_inv196_k200d", "anticlose_inv196_allprimes"]
    for idx, pat_name in enumerate(key_pats):
        if pat_name not in persistence_results: continue
        ax = axes[idx]
        layers_sorted = sorted(persistence_results[pat_name].keys())
        near0 = [persistence_results[pat_name][li]["pct_near_zero"] for li in layers_sorted]
        near_inv = [persistence_results[pat_name][li]["pct_near_inv196"] for li in layers_sorted]
        dmeans = [persistence_results[pat_name][li]["delta_mean"] for li in layers_sorted]
        ax.plot(layers_sorted, near0, "o-", label="% near zero (closed)", color="red", linewidth=2)
        ax.plot(layers_sorted, near_inv, "s-", label="% near 1/196", color="blue", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(layers_sorted, dmeans, "^--", label="delta mean", color="green", linewidth=1.5, alpha=0.7)
        ax2.set_ylabel("Delta Mean", color="green", fontsize=9)
        ax.set_xlabel("Conv Layer Index")
        ax.set_ylabel("Percentage")
        ax.set_title(f"196 Persistence: {pat_name}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/persistence_196.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved persistence plot: {OUTPUT_DIR}/persistence_196.png")

    all_results["persistence_196"] = persistence_results

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\n{'='*70}")
    print("Saving results...")
    json_path = f"{OUTPUT_DIR}/hallucination_deep.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    csv_path = f"{OUTPUT_DIR}/hallucination_deep.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            all_fields = set()
            for row in csv_rows: all_fields.update(row.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_fields))
            w.writeheader()
            for row in csv_rows: w.writerow({k: row.get(k, "") for k in sorted(all_fields)})
    print(f"Saved CSV: {csv_path}")

    print(f"\nDetection visualization images saved to: {OUTPUT_DIR}/")
    print("\nDONE")

if __name__ == "__main__":
    main()
