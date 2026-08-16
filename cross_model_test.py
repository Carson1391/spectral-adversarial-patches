"""
Cross-Model Transferability Test — do k=167, k=208, k=196 work on YOLOv8, YOLO11, YOLO26?

Tests the three significant findings from YOLOv3 analysis on newer YOLO architectures:
- k=167 diagonal: total detection suppression (aliasing bomb)
- k=208 diagonal: hallucination generator (grid resonator)
- k=196 diagonal: diffuse disruption (misaligned non-power)

Each model has DIFFERENT architecture:
- YOLOv3: Darknet-53, 75 conv, 13x13 final grid, input 416
- YOLOv8: C2f modules, decoupled head, input 640 (default)
- YOLO11: C3k2 modules, input 640
- YOLO26: latest architecture, input 640

Key question: do the SAME spatial frequencies transfer, or does each architecture
need its own frequency? The downsample chains differ, so k=167 may not alias
the same way at 640 input.
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

from ultralytics import YOLO

IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\cross_model"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv3 config (for Darknet direct loading)
CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"

COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image_pil(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    return c

def make_sinusoid(H, W, kx, ky, phase_deg, amp):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (amp * np.cos(2*np.pi*(kx/W*x + ky/H*y) + np.radians(phase_deg))).astype(np.float32)

def add_pattern_to_pil(pil_img, pattern):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:,:,c] = np.clip(arr[:,:,c] + pattern, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def get_dets_ultralytics(model, img, conf=0.1):
    results = model(img, verbose=False)
    dets = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            c = int(boxes.cls[i].item())
            dets.append({
                "class_id": c,
                "class_name": model.names.get(c, f"c{c}"),
                "confidence": float(boxes.conf[i].item()),
                "bbox": [float(boxes.xyxy[i][0]), float(boxes.xyxy[i][1]),
                         float(boxes.xyxy[i][2]), float(boxes.xyxy[i][3])],
            })
    return [d for d in dets if d["confidence"] >= conf]

def get_dets_yolov3(model, img_tensor, conf=0.1):
    with torch.no_grad():
        output = model(img_tensor)
    dets = []
    if output is None: return dets
    out = output.cpu().numpy()
    if out.ndim == 3: out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            dets.append({
                "class_id": cls,
                "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                "confidence": float(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
            })
    return dets

def draw_dets(pil_img, dets, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pil_img)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{d['class_name']} {d['confidence']:.2f}", color="lime", fontsize=8,
                fontweight="bold", bbox=dict(facecolor="black", alpha=0.5))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70)
    print("Cross-Model Transferability: k=167, k=208, k=196")
    print("="*70)
    print(f"Device: {DEVICE}")

    # Key frequencies to test
    # For YOLOv3 (416 input): k=167, 208, 196 as tested
    # For YOLOv8/11/26 (640 input): need to scale frequencies
    # 167/416 * 640 = 257 (scaled equivalent)
    # 208/416 * 640 = 320 (scaled equivalent)
    # 196/416 * 640 = 302 (scaled equivalent)
    # Also test the ORIGINAL k values (167, 208, 196) at 640 to see if
    # absolute frequency matters vs relative frequency

    test_configs = [
        # (name, kx, ky, amp, description)
        ("k167_d", 167, 167, 0.20, "YOLOv3 aliasing bomb (prime near Nyquist)"),
        ("k208_d", 208, 208, 0.20, "YOLOv3 grid resonator (13x16, Nyquist)"),
        ("k196_d", 196, 196, 0.20, "YOLOv3 misaligned non-power"),
        # Scaled for 640 input
        ("k257_d_scaled", 257, 257, 0.20, "k=167 scaled to 640 (aliasing bomb)"),
        ("k320_d_scaled", 320, 320, 0.20, "k=208 scaled to 640 (grid resonator)"),
        ("k302_d_scaled", 302, 302, 0.20, "k=196 scaled to 640 (misaligned)"),
        # Architecture-specific for 640: downsample chain is 640->320->160->80->40->20
        # Final grid: 20x20 (not 13x13 like YOLOv3)
        # 20-multiples: k=20, 40, 80, 160, 320
        ("k320_d_20mult", 320, 320, 0.20, "320=20x16, 20-grid resonator at 640"),
        ("k160_d_20mult", 160, 160, 0.20, "160=20x8, 20-grid mid resonator"),
        # High primes near 640/2=320 Nyquist
        ("k317_d_prime", 317, 317, 0.20, "prime near 640 Nyquist (317)"),
        ("k313_d_prime", 313, 313, 0.20, "prime near 640 Nyquist (313)"),
        ("k311_d_prime", 311, 311, 0.20, "prime near 640 Nyquist (311)"),
        ("k307_d_prime", 307, 307, 0.20, "prime near 640 Nyquist (307)"),
        # Also test at 416 input for ultralytics models (they accept variable size)
        ("k167_d_416", 167, 167, 0.20, "k=167 at 416 input (original)"),
        ("k208_d_416", 208, 208, 0.20, "k=208 at 416 input (original)"),
        # Controls
        ("k32_d_pow2", 32, 32, 0.20, "k=32 power of 2 (weak on v3)"),
        ("k100_d_ctrl", 100, 100, 0.20, "k=100 control (non-special)"),
    ]

    all_results = {}
    csv_rows = []

    # ============================================================
    # YOLOv3 (416 input, Darknet-53)
    # ============================================================
    print("\n" + "="*70)
    print("YOLOv3 (416x416, Darknet-53)")
    print("="*70)

    from pytorchyolo.models import Darknet
    v3_model = Darknet(CONFIG_PATH).to(DEVICE)
    v3_model.load_darknet_weights(WEIGHTS_PATH)
    v3_model.eval()
    for p in v3_model.parameters(): p.requires_grad_(False)

    SIZE_V3 = 416
    img_w_v3 = load_image_pil(IMG_WITH, SIZE_V3)
    img_wo_v3 = load_image_pil(IMG_WITHOUT, SIZE_V3)

    # Baseline
    import numpy as np
    arr_w_v3 = np.array(img_w_v3, dtype=np.float32) / 255.0
    tensor_w_v3 = torch.from_numpy(arr_w_v3).permute(2,0,1).unsqueeze(0).to(DEVICE)
    det_w_v3 = get_dets_yolov3(v3_model, tensor_w_v3, conf=0.1)
    print(f"  Baseline (with human): {len(det_w_v3)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w_v3]}")

    v3_results = {}
    for name, kx, ky, amp, desc in test_configs:
        H, W = SIZE_V3, SIZE_V3
        pattern = make_sinusoid(H, W, kx, ky, 0, amp)
        img_mod = add_pattern_to_pil(img_w_v3, pattern)
        arr_mod = np.array(img_mod, dtype=np.float32) / 255.0
        tensor_mod = torch.from_numpy(arr_mod).permute(2,0,1).unsqueeze(0).to(DEVICE)
        dets = get_dets_yolov3(v3_model, tensor_mod, conf=0.1)

        classes = {}
        for d in dets:
            cn = d["class_name"]
            if cn not in classes: classes[cn] = []
            classes[cn].append(round(d["confidence"], 3))

        person_count = len(classes.get("person", []))
        total = len(dets)
        suppressed = total == 0 and len(det_w_v3) > 0
        hallucinated = any(c not in {"person"} for c in classes)  # non-person classes

        v3_results[name] = {
            "kx": kx, "ky": ky, "amp": amp, "desc": desc,
            "total_dets": total, "person_dets": person_count,
            "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
            "suppressed": suppressed, "hallucinated": hallucinated,
        }

        tag = ""
        if suppressed: tag = " [SUPPRESSED]"
        elif hallucinated: tag = " [HALLUCINATED]"
        elif person_count < len(det_w_v3): tag = f" [person {len(det_w_v3)}->{person_count}]"
        print(f"  {name:>20}: {total:2d} dets, person={person_count}, classes={list(classes.keys())}{tag}")

        for cls_name, confs in classes.items():
            csv_rows.append({"model": "YOLOv3", "pattern": name, "kx": kx, "ky": ky,
                             "class": cls_name, "count": len(confs),
                             "max_conf": max(confs), "total": total})

        # Draw interesting cases
        if suppressed or hallucinated or person_count != len(det_w_v3):
            draw_dets(img_mod, dets, f"YOLOv3 + {name}{tag}", f"{OUTPUT_DIR}/v3_{name}.png")

    all_results["YOLOv3"] = v3_results

    # Free v3 model
    del v3_model
    torch.cuda.empty_cache()

    # ============================================================
    # YOLOv8 (640 input default, but we test both 416 and 640)
    # ============================================================
    print("\n" + "="*70)
    print("YOLOv8 (640x640 default, also 416)")
    print("="*70)

    v8_model = YOLO(r"C:\Users\carso\Desktop\YODO\YOLOv8\yolov8l.pt")
    v8_model.to(DEVICE)

    # Test at 416 (same as v3) and 640 (native)
    for size_label, img_size in [("416", 416), ("640", 640)]:
        img_w = load_image_pil(IMG_WITH, img_size)
        img_wo = load_image_pil(IMG_WITHOUT, img_size)

        # Baseline
        det_w = get_dets_ultralytics(v8_model, img_w, conf=0.1)
        print(f"\n  YOLOv8 @{img_size} baseline (with human): {len(det_w)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w[:5]]}")

        v8_key = f"YOLOv8_{size_label}"
        v8_results = {}

        for name, kx, ky, amp, desc in test_configs:
            H, W = img_size, img_size
            pattern = make_sinusoid(H, W, kx, ky, 0, amp)
            img_mod = add_pattern_to_pil(img_w, pattern)
            dets = get_dets_ultralytics(v8_model, img_mod, conf=0.1)

            classes = {}
            for d in dets:
                cn = d["class_name"]
                if cn not in classes: classes[cn] = []
                classes[cn].append(round(d["confidence"], 3))

            person_count = len(classes.get("person", []))
            total = len(dets)
            suppressed = total == 0 and len(det_w) > 0
            hallucinated = any(c not in {"person"} for c in classes)

            v8_results[name] = {
                "kx": kx, "ky": ky, "amp": amp, "desc": desc,
                "total_dets": total, "person_dets": person_count,
                "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
                "suppressed": suppressed, "hallucinated": hallucinated,
            }

            tag = ""
            if suppressed: tag = " [SUPPRESSED]"
            elif hallucinated: tag = " [HALLUCINATED]"
            elif person_count < len(det_w): tag = f" [person {len(det_w)}->{person_count}]"
            print(f"  {name:>20}: {total:2d} dets, person={person_count}, classes={list(classes.keys())}{tag}")

            for cls_name, confs in classes.items():
                csv_rows.append({"model": v8_key, "pattern": name, "kx": kx, "ky": ky,
                                 "class": cls_name, "count": len(confs),
                                 "max_conf": max(confs), "total": total})

            if suppressed or hallucinated or (person_count < len(det_w) and person_count >= 0):
                draw_dets(img_mod, dets, f"YOLOv8 @{img_size} + {name}{tag}", f"{OUTPUT_DIR}/v8_{size_label}_{name}.png")

        all_results[v8_key] = v8_results

    del v8_model
    torch.cuda.empty_cache()

    # ============================================================
    # YOLO11 (640 input default)
    # ============================================================
    print("\n" + "="*70)
    print("YOLO11 (640x640 default, also 416)")
    print("="*70)

    v11_model = YOLO(r"C:\Users\carso\Desktop\YODO\YOLO11\yolo11l.pt")
    v11_model.to(DEVICE)

    for size_label, img_size in [("416", 416), ("640", 640)]:
        img_w = load_image_pil(IMG_WITH, img_size)
        img_wo = load_image_pil(IMG_WITHOUT, img_size)

        det_w = get_dets_ultralytics(v11_model, img_w, conf=0.1)
        print(f"\n  YOLO11 @{img_size} baseline (with human): {len(det_w)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w[:5]]}")

        v11_key = f"YOLO11_{size_label}"
        v11_results = {}

        for name, kx, ky, amp, desc in test_configs:
            H, W = img_size, img_size
            pattern = make_sinusoid(H, W, kx, ky, 0, amp)
            img_mod = add_pattern_to_pil(img_w, pattern)
            dets = get_dets_ultralytics(v11_model, img_mod, conf=0.1)

            classes = {}
            for d in dets:
                cn = d["class_name"]
                if cn not in classes: classes[cn] = []
                classes[cn].append(round(d["confidence"], 3))

            person_count = len(classes.get("person", []))
            total = len(dets)
            suppressed = total == 0 and len(det_w) > 0
            hallucinated = any(c not in {"person"} for c in classes)

            v11_results[name] = {
                "kx": kx, "ky": ky, "amp": amp, "desc": desc,
                "total_dets": total, "person_dets": person_count,
                "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
                "suppressed": suppressed, "hallucinated": hallucinated,
            }

            tag = ""
            if suppressed: tag = " [SUPPRESSED]"
            elif hallucinated: tag = " [HALLUCINATED]"
            elif person_count < len(det_w): tag = f" [person {len(det_w)}->{person_count}]"
            print(f"  {name:>20}: {total:2d} dets, person={person_count}, classes={list(classes.keys())}{tag}")

            for cls_name, confs in classes.items():
                csv_rows.append({"model": v11_key, "pattern": name, "kx": kx, "ky": ky,
                                 "class": cls_name, "count": len(confs),
                                 "max_conf": max(confs), "total": total})

            if suppressed or hallucinated or (person_count < len(det_w) and person_count >= 0):
                draw_dets(img_mod, dets, f"YOLO11 @{img_size} + {name}{tag}", f"{OUTPUT_DIR}/v11_{size_label}_{name}.png")

        all_results[v11_key] = v11_results

    del v11_model
    torch.cuda.empty_cache()

    # ============================================================
    # YOLO26 (640 input default)
    # ============================================================
    print("\n" + "="*70)
    print("YOLO26 (640x640 default, also 416)")
    print("="*70)

    v26_model = YOLO(r"C:\Users\carso\Desktop\YODO\YOLO26\yolo26l.pt")
    v26_model.to(DEVICE)

    for size_label, img_size in [("416", 416), ("640", 640)]:
        img_w = load_image_pil(IMG_WITH, img_size)
        img_wo = load_image_pil(IMG_WITHOUT, img_size)

        det_w = get_dets_ultralytics(v26_model, img_w, conf=0.1)
        print(f"\n  YOLO26 @{img_size} baseline (with human): {len(det_w)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w[:5]]}")

        v26_key = f"YOLO26_{size_label}"
        v26_results = {}

        for name, kx, ky, amp, desc in test_configs:
            H, W = img_size, img_size
            pattern = make_sinusoid(H, W, kx, ky, 0, amp)
            img_mod = add_pattern_to_pil(img_w, pattern)
            dets = get_dets_ultralytics(v26_model, img_mod, conf=0.1)

            classes = {}
            for d in dets:
                cn = d["class_name"]
                if cn not in classes: classes[cn] = []
                classes[cn].append(round(d["confidence"], 3))

            person_count = len(classes.get("person", []))
            total = len(dets)
            suppressed = total == 0 and len(det_w) > 0
            hallucinated = any(c not in {"person"} for c in classes)

            v26_results[name] = {
                "kx": kx, "ky": ky, "amp": amp, "desc": desc,
                "total_dets": total, "person_dets": person_count,
                "classes": {k: {"count": len(v), "max_conf": max(v)} for k, v in classes.items()},
                "suppressed": suppressed, "hallucinated": hallucinated,
            }

            tag = ""
            if suppressed: tag = " [SUPPRESSED]"
            elif hallucinated: tag = " [HALLUCINATED]"
            elif person_count < len(det_w): tag = f" [person {len(det_w)}->{person_count}]"
            print(f"  {name:>20}: {total:2d} dets, person={person_count}, classes={list(classes.keys())}{tag}")

            for cls_name, confs in classes.items():
                csv_rows.append({"model": v26_key, "pattern": name, "kx": kx, "ky": ky,
                                 "class": cls_name, "count": len(confs),
                                 "max_conf": max(confs), "total": total})

            if suppressed or hallucinated or (person_count < len(det_w) and person_count >= 0):
                draw_dets(img_mod, dets, f"YOLO26 @{img_size} + {name}{tag}", f"{OUTPUT_DIR}/v26_{size_label}_{name}.png")

        all_results[v26_key] = v26_results

    del v26_model
    torch.cuda.empty_cache()

    # ============================================================
    # CROSS-MODEL COMPARISON
    # ============================================================
    print("\n" + "="*70)
    print("CROSS-MODEL COMPARISON")
    print("="*70)

    # Compare key patterns across all models
    key_patterns = ["k167_d", "k208_d", "k196_d", "k257_d_scaled", "k320_d_scaled", "k302_d_scaled",
                    "k317_d_prime", "k320_d_20mult", "k32_d_pow2", "k100_d_ctrl"]

    print(f"\n{'Pattern':>20} | ", end="")
    for mk in all_results.keys():
        print(f"{mk:>16} |", end="")
    print()
    print("-" * (22 + 19 * len(all_results)))

    for pat in key_patterns:
        print(f"{pat:>20} | ", end="")
        for mk in all_results.keys():
            data = all_results[mk].get(pat, {})
            total = data.get("total_dets", -1)
            supp = data.get("suppressed", False)
            halluc = data.get("hallucinated", False)
            tag = "SUPP" if supp else "HALL" if halluc else str(total)
            print(f"{tag:>16} |", end="")
        print()

    # Which patterns transfer across ALL models?
    print("\n--- TRANSFERABILITY ---")
    for pat in key_patterns:
        suppress_count = sum(1 for mk in all_results if all_results[mk].get(pat, {}).get("suppressed", False))
        halluc_count = sum(1 for mk in all_results if all_results[mk].get(pat, {}).get("hallucinated", False))
        if suppress_count > 0 or halluc_count > 0:
            print(f"  {pat:>20}: suppressed on {suppress_count}/{len(all_results)} models, "
                  f"hallucinated on {halluc_count}/{len(all_results)} models")

    # ============================================================
    # SAVE
    # ============================================================
    print("\nSaving results...")
    json_path = f"{OUTPUT_DIR}/cross_model.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    csv_path = f"{OUTPUT_DIR}/cross_model.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            all_fields = set()
            for row in csv_rows: all_fields.update(row.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_fields))
            w.writeheader()
            for row in csv_rows: w.writerow({k: row.get(k, "") for k in sorted(all_fields)})
    print(f"Saved CSV: {csv_path}")

    # Plot: cross-model suppression comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(key_patterns))
    width = 0.15
    model_keys = list(all_results.keys())
    for i, mk in enumerate(model_keys):
        totals = [all_results[mk].get(pat, {}).get("total_dets", -1) for pat in key_patterns]
        ax.bar(x + i*width, totals, width, label=mk)
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Total Detections (conf >= 0.1)")
    ax.set_title("Cross-Model Detection Count by Pattern")
    ax.set_xticks(x + width * (len(model_keys)-1)/2)
    ax.set_xticklabels(key_patterns, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cross_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot: {OUTPUT_DIR}/cross_model_comparison.png")

    print("\nDONE")

if __name__ == "__main__":
    main()
