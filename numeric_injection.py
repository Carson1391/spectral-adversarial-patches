"""Numeric/Algebraic Injection Attacks on YOLOv3."""
import os, sys, json, csv, math
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
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
IMG_WITHOUT  = r"C:\Users\carso\Desktop\YODO\withouthuman.png"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\forward_analysis\numeric_injection"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416
LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]
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

def add_offset(arr, off):
    out = arr.copy()
    for c in range(3): out[:,:,c] = np.clip(out[:,:,c] + off, 0, 1)
    return out

def get_dets(model, x, conf=0.25):
    with torch.no_grad(): output = model(x)
    dets = []
    if output is None: return dets
    out = output.cpu().numpy()
    if out.ndim == 3: out = out[0]
    for row in out:
        if len(row) >= 6 and row[4] >= conf:
            cls = int(row[5])
            dets.append({"class_id": cls, "class_name": COCO_NAMES[cls] if cls < 80 else f"c{cls}",
                         "confidence": float(row[4])})
    return dets

def supp_scores(caps_m, caps_w, caps_wo, bd, layers):
    s = {}
    for li in layers:
        if li not in caps_w or li not in caps_wo or li not in caps_m: continue
        fm, fw, fo = caps_m[li].squeeze(0), caps_w[li].squeeze(0), caps_wo[li].squeeze(0)
        dm, dw = (fm-fw).norm().item(), bd[li]
        dot = torch.dot((fm-fw).flatten(), (fo-fw).flatten()).item() / (dw*dm + 1e-12)
        s[li] = dot * dm / (dw + 1e-12)
    return s

def feat_stats(caps, layers):
    st = {}
    for li in layers:
        if li not in caps: continue
        f = caps[li].squeeze(0)
        st[li] = {"mean": float(f.mean()), "std": float(f.std()), "max": float(f.max()),
                  "min": float(f.min()), "frac0": float((f==0).float().mean()),
                  "l2": float(f.norm())}
    return st

KL = [0, 12, 54, 62, 75, 105]

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("="*70); print("Numeric/Algebraic Injection Attacks"); print("="*70)
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    iw, arr_w = load_image(IMG_WITH)
    iwo, arr_wo = load_image(IMG_WITHOUT)
    H, W, _ = arr_w.shape
    caps_w, _ = forward_capture(model, iw)
    caps_wo, _ = forward_capture(model, iwo)
    bd = {li: (caps_wo[li].squeeze(0)-caps_w[li].squeeze(0)).norm().item() for li in LAYERS if li in caps_w and li in caps_wo}
    det_w = get_dets(model, iw)
    print(f"Baseline: {len(det_w)} dets: {[(d['class_name'],round(d['confidence'],2)) for d in det_w]}")
    results = {}; csv_rows = []

    # EXP 1: Constants
    print("\n--- EXP 1: Constant/Zero Injection ---")
    consts = [("zero",0.0),("half",0.5),("one",1.0),("neg_half",-0.5),
              ("inv_196",1/196),("inv_128",1/128),("inv_256",1/256),
              ("inv_13",1/13),("inv_26",1/26),("inv_52",1/52),
              ("inv_104",1/104),("inv_208",1/208),("inv_416",1/416),
              ("9.8_255",9.8/255),("pi_255",math.pi/255),("e_255",math.e/255)]
    e1 = {}
    for name, val in consts:
        if name in ("zero","half","one"): arr_m = np.full_like(arr_w, val)
        else: arr_m = add_offset(arr_w, val)
        tm = torch.from_numpy(arr_m).permute(2,0,1).unsqueeze(0).to(DEVICE)
        cm, _ = forward_capture(model, tm)
        dm = get_dets(model, tm)
        sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
        st = feat_stats(cm, LAYERS)
        avg = float(np.mean([sc.get(l,0) for l in KL if l in sc]))
        anom = []
        if len(dm) > len(det_w)+2: anom.append("HALLUCINATION")
        if len(dm)==0 and len(det_w)>0: anom.append("TOTAL_SUPPRESSION")
        for li in KL:
            if li in st and (math.isnan(st[li]["mean"]) or math.isinf(st[li]["mean"])):
                anom.append("NAN"); break
        cls_list = {d["class_name"] for d in dm}
        e1[name] = {"val":val,"supp":avg,"dets":len(dm),"classes":list(cls_list),"anom":anom,
                    "per_layer":{l:sc.get(l,0) for l in KL},"stats":{l:st.get(l,{}) for l in KL}}
        print(f"  {name:>12} v={val:8.5f} supp={avg:6.3f} dets={len(dm):2d} cls={cls_list} {anom}")
        for l in KL: csv_rows.append({"exp":"const","name":name,"val":val,"layer":l,
                                      "supp":sc.get(l,0),"dets":len(dm),"mean":st.get(l,{}).get("mean",0)})
    results["exp1_constants"] = e1

    # EXP 2: Doubling-sequence architecture-aligned frequencies
    print("\n--- EXP 2: Architecture-Aligned Doubling Frequencies ---")
    arch = [(416//r, r) for r in [416,208,104,52,26,13]]  # k=1,2,4,8,16,32
    arch_k = [k for k, r in arch]  # powers of 2 frequency list, used in exp6 and plots
    ctrl = [3,6,12,24,48,96]
    amp = 0.20; e2 = {}
    for label, freqs in [("arch",arch),("ctrl",ctrl)]:
        for item in freqs:
            k_val, res = item if label=="arch" else (item, 0)
            for on, (kx,ky) in [("h",(k_val,0)),("d",(k_val,k_val))]:
                nm = f"{label}_k{k_val}_{on}"
                pat = make_sinusoid(H,W,kx,ky,0,amp)
                arr_m = add_pattern(arr_w, pat)
                tm = torch.from_numpy(arr_m).permute(2,0,1).unsqueeze(0).to(DEVICE)
                cm, _ = forward_capture(model, tm)
                dm = get_dets(model, tm)
                sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
                avg = float(np.mean([sc.get(l,0) for l in KL if l in sc]))
                e2[nm] = {"k":k_val,"res":res,"orient":on,"supp":avg,"dets":len(dm),
                          "per_layer":{l:sc.get(l,0) for l in KL}}
                print(f"  {nm:>20} k={k_val:3d} supp={avg:6.3f} dets={len(dm):2d}")
                for l in KL: csv_rows.append({"exp":f"{label}_freq","name":nm,"k":k_val,"layer":l,"supp":sc.get(l,0),"dets":len(dm)})
    # Compare arch vs ctrl
    print("\n  ARCH vs CTRL comparison (diagonal):")
    for k_arch, res in arch:
        a = e2.get(f"arch_k{k_arch}_d",{}).get("supp",0)
        closest = min(ctrl, key=lambda c: abs(c-k_arch))
        c = e2.get(f"ctrl_k{closest}_d",{}).get("supp",0)
        r = a/(c+1e-12)
        tag = "ARCH WINS" if r>1.1 else "CTRL WINS" if r<0.9 else "~equal"
        print(f"    arch k={k_arch:3d} supp={a:.4f} vs ctrl k={closest:3d} supp={c:.4f} ratio={r:.2f}x {tag}")
    results["exp2_arch_freqs"] = e2

    # EXP 3: Lychrel-inspired patterns
    print("\n--- EXP 3: Lychrel-Inspired Non-Convergent Patterns ---")
    lych = []
    # Period-196 blocks
    p_a = np.zeros((H,W),dtype=np.float32)
    for x in range(W):
        if (x%196)<98: p_a[:,x] = amp
    lych.append(("lych_period196",p_a))
    # Reverse-and-add
    rng = np.random.RandomState(196)
    base = rng.randn(416).astype(np.float32)
    rab = (base + base[::-1]); rab = (rab/np.abs(rab).max()*amp).astype(np.float32)
    lych.append(("lych_revadd196",np.tile(rab[None,:],(H,1))))
    # 196-row shift
    p_c = np.zeros((H,W),dtype=np.float32)
    for y in range(H):
        sh = (196*y)%W
        p_c[y] = np.roll(np.cos(np.linspace(0,2*np.pi*196,W)).astype(np.float32)*amp, sh)
    lych.append(("lych_shift196",p_c))
    # k=196 diagonal
    lych.append(("lych_k196d196",make_sinusoid(H,W,196,196,0,amp)))
    # Control k=128
    lych.append(("ctrl_k128d128",make_sinusoid(H,W,128,128,0,amp)))

    e3 = {}
    for name, pat in lych:
        arr_m = add_pattern(arr_w, pat)
        tm = torch.from_numpy(arr_m).permute(2,0,1).unsqueeze(0).to(DEVICE)
        cm, _ = forward_capture(model, tm)
        dm = get_dets(model, tm)
        sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
        st = feat_stats(cm, LAYERS)
        avg = float(np.mean([sc.get(l,0) for l in KL if l in sc]))
        anom = []
        if len(dm)>len(det_w)+2: anom.append("HALLUC")
        if len(dm)==0 and len(det_w)>0: anom.append("SUPPRESS")
        for li in KL:
            if li in st and (math.isnan(st[li]["mean"]) or math.isinf(st[li]["mean"])):
                anom.append("NAN"); break
        cls_list = {d["class_name"] for d in dm}
        e3[name] = {"supp":avg,"dets":len(dm),"classes":list(cls_list),"anom":anom,
                    "per_layer":{l:sc.get(l,0) for l in KL}}
        print(f"  {name:>20} supp={avg:6.3f} dets={len(dm):2d} cls={cls_list} {anom}")
        for l in KL: csv_rows.append({"exp":"lychrel","name":name,"layer":l,"supp":sc.get(l,0),"dets":len(dm)})
    results["exp3_lychrel"] = e3

    # EXP 4: Iterative Lychrel reverse-and-add accumulation
    print("\n--- EXP 4: Iterative Lychrel Accumulation ---")
    max_it = 15; per_amp = 0.03
    rng2 = np.random.RandomState(196)
    cur_pat = (rng2.randn(H,W).astype(np.float32) / np.abs(rng2.randn(H,W)).max() * per_amp)
    arr_acc = arr_w.copy()
    e4 = {}
    for it in range(1, max_it+1):
        arr_acc = add_pattern(arr_acc, cur_pat)
        tm = torch.from_numpy(arr_acc).permute(2,0,1).unsqueeze(0).to(DEVICE)
        cm, _ = forward_capture(model, tm)
        dm = get_dets(model, tm)
        sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
        avg = float(np.mean([sc.get(l,0) for l in KL if l in sc]))
        cls_list = {d["class_name"] for d in dm}
        e4[it] = {"supp":avg,"dets":len(dm),"classes":list(cls_list),
                  "per_layer":{l:sc.get(l,0) for l in KL}}
        print(f"  iter={it:2d} supp={avg:6.3f} dets={len(dm):2d} cls={cls_list}")
        # Lychrel step: reverse pattern and add to itself
        cur_pat = cur_pat + cur_pat[::-1, ::-1]
        cur_pat = cur_pat / (np.abs(cur_pat).max() + 1e-12) * per_amp
        for l in KL: csv_rows.append({"exp":"lych_iter","name":"lychrel_iter","it":it,"layer":l,"supp":sc.get(l,0),"dets":len(dm)})
    results["exp4_lychrel_iterative"] = e4

    # EXP 5: Iterative doubling-sequence frequency accumulation
    print("\n--- EXP 5: Iterative Doubling-Frequency Accumulation ---")
    # Apply k=1, then k=2, then k=4, etc. cumulatively
    arr_acc2 = arr_w.copy()
    e5 = {}
    for i, (k_val, res) in enumerate(arch):
        pat = make_sinusoid(H, W, k_val, k_val, 0, 0.05)
        arr_acc2 = add_pattern(arr_acc2, pat)
        tm = torch.from_numpy(arr_acc2).permute(2,0,1).unsqueeze(0).to(DEVICE)
        cm, _ = forward_capture(model, tm)
        dm = get_dets(model, tm)
        sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
        avg = float(np.mean([sc.get(l,0) for l in KL if l in sc]))
        cls_list = {d["class_name"] for d in dm}
        e5[f"cum_k{k_val}"] = {"supp":avg,"dets":len(dm),"classes":list(cls_list),
                               "per_layer":{l:sc.get(l,0) for l in KL}}
        print(f"  +k={k_val:3d} (res={res:3d}) supp={avg:6.3f} dets={len(dm):2d} cls={cls_list}")
        for l in KL: csv_rows.append({"exp":"doubling_cum","name":f"cum_k{k_val}","k":k_val,"layer":l,"supp":sc.get(l,0),"dets":len(dm)})
    results["exp5_doubling_cumulative"] = e5

    # EXP 6: Prime frequency injection
    # YOLOv3 internal structure is all powers of 2 and multiples of 13
    # Primes that don't divide into any of these may cause non-aligning interference
    # Test all primes up to 416/2=208 (Nyquist) as spatial frequencies
    print("\n--- EXP 6: Prime Frequency Injection ---")
    def sieve_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    primes = sieve_primes(208)  # Nyquist limit for 416
    # Also test 13-multiples (architecture-aligned) and powers of 2 for comparison
    arch_aligned = [1, 2, 4, 8, 16, 32]  # powers of 2 (downsample chain)
    thirteen_mults = [13, 26, 39, 52, 65, 78, 91, 104, 130, 156, 208]  # multiples of 13

    e6 = {}
    # Test primes
    for k_val in primes:
        for on, (kx, ky) in [("h", (k_val, 0)), ("d", (k_val, k_val))]:
            nm = f"prime_k{k_val}_{on}"
            pat = make_sinusoid(H, W, kx, ky, 0, amp)
            arr_m = add_pattern(arr_w, pat)
            tm = torch.from_numpy(arr_m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            cm, _ = forward_capture(model, tm)
            dm = get_dets(model, tm)
            sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
            avg = float(np.mean([sc.get(l, 0) for l in KL if l in sc]))
            cls_list = {d["class_name"] for d in dm}
            anom = []
            if len(dm) > len(det_w) + 2: anom.append("HALLUC")
            if len(dm) == 0 and len(det_w) > 0: anom.append("SUPPRESS")
            e6[nm] = {"k": k_val, "orient": on, "supp": avg, "dets": len(dm),
                      "classes": list(cls_list), "anom": anom,
                      "per_layer": {l: sc.get(l, 0) for l in KL}}
            if anom or avg > 0.3:  # Only print interesting ones
                print(f"  {nm:>20} supp={avg:6.3f} dets={len(dm):2d} {anom}")
            for l in KL:
                csv_rows.append({"exp": "prime", "name": nm, "k": k_val, "orient": on,
                                 "layer": l, "supp": sc.get(l, 0), "dets": len(dm)})

    # Test 13-multiples for comparison
    for k_val in thirteen_mults:
        for on, (kx, ky) in [("h", (k_val, 0)), ("d", (k_val, k_val))]:
            nm = f"thirteen_k{k_val}_{on}"
            pat = make_sinusoid(H, W, kx, ky, 0, amp)
            arr_m = add_pattern(arr_w, pat)
            tm = torch.from_numpy(arr_m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            cm, _ = forward_capture(model, tm)
            dm = get_dets(model, tm)
            sc = supp_scores(cm, caps_w, caps_wo, bd, LAYERS)
            avg = float(np.mean([sc.get(l, 0) for l in KL if l in sc]))
            cls_list = {d["class_name"] for d in dm}
            anom = []
            if len(dm) > len(det_w) + 2: anom.append("HALLUC")
            if len(dm) == 0 and len(det_w) > 0: anom.append("SUPPRESS")
            e6[nm] = {"k": k_val, "orient": on, "supp": avg, "dets": len(dm),
                      "classes": list(cls_list), "anom": anom,
                      "per_layer": {l: sc.get(l, 0) for l in KL}}
            if anom or avg > 0.3:
                print(f"  {nm:>20} supp={avg:6.3f} dets={len(dm):2d} {anom}")
            for l in KL:
                csv_rows.append({"exp": "thirteen", "name": nm, "k": k_val, "orient": on,
                                 "layer": l, "supp": sc.get(l, 0), "dets": len(dm)})

    # Rank primes by suppression
    print("\n  Top prime frequencies by suppression (diagonal):")
    prime_d = {k: v for k, v in e6.items() if k.startswith("prime_") and k.endswith("_d")}
    ranked_primes = sorted(prime_d.items(), key=lambda x: x[1]["supp"], reverse=True)
    for i, (nm, data) in enumerate(ranked_primes[:10]):
        print(f"    #{i+1} k={data['k']:3d} supp={data['supp']:.4f} dets={data['dets']:2d} {data['anom']}")

    # Compare: primes vs 13-multiples vs powers-of-2 (all diagonal)
    print("\n  Category comparison (diagonal, avg suppression):")
    prime_avg = np.mean([v["supp"] for k, v in e6.items() if k.startswith("prime_") and k.endswith("_d")])
    thirteen_avg = np.mean([v["supp"] for k, v in e6.items() if k.startswith("thirteen_") and k.endswith("_d")])
    arch_avg = np.mean([e2.get(f"arch_k{k}_d", {}).get("supp", 0) for k in arch_k])
    print(f"    Primes:        avg={prime_avg:.4f} (n={len([k for k in e6 if k.startswith('prime_') and k.endswith('_d')])})")
    print(f"    13-multiples:  avg={thirteen_avg:.4f} (n={len([k for k in e6 if k.startswith('thirteen_') and k.endswith('_d')])})")
    print(f"    Powers of 2:   avg={arch_avg:.4f} (n={len(arch_k)})")

    # Find primes that cause hallucinations or total suppression
    print("\n  Anomalous prime frequencies:")
    for nm, data in e6.items():
        if data["anom"]:
            print(f"    {nm:>20} k={data['k']:3d} supp={data['supp']:.4f} dets={data['dets']:2d} "
                  f"cls={data['classes']} {data['anom']}")

    results["exp6_primes"] = e6

    # PLOTS
    print("\nGenerating plots...")
    # Plot 1: Constant injection suppression
    fig, ax = plt.subplots(figsize=(12,5))
    names = list(e1.keys()); supps = [e1[n]["supp"] for n in names]
    ax.barh(names, supps); ax.set_xlabel("Avg Suppression Score")
    ax.set_title("Constant/Offset Injection — Suppression Effect")
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/constants.png",dpi=150); plt.close()

    # Plot 2: Arch vs Ctrl frequency comparison
    fig, ax = plt.subplots(figsize=(10,6))
    arch_k = [k for k,r in arch]; arch_s = [e2.get(f"arch_k{k}_d",{}).get("supp",0) for k in arch_k]
    ctrl_k = ctrl; ctrl_s = [e2.get(f"ctrl_k{k}_d",{}).get("supp",0) for k in ctrl_k]
    ax.plot(arch_k, arch_s, "o-", label="Architecture-aligned (powers of 2)", linewidth=2, markersize=8)
    ax.plot(ctrl_k, ctrl_s, "s--", label="Control (non powers of 2)", linewidth=2, markersize=8)
    ax.set_xlabel("Spatial Frequency k"); ax.set_ylabel("Avg Suppression Score")
    ax.set_title("Architecture-Aligned vs Control Frequencies (Diagonal)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/arch_vs_ctrl.png",dpi=150); plt.close()

    # Plot 3: Lychrel iterative
    fig, ax = plt.subplots(figsize=(10,6))
    iters = list(e4.keys()); lych_s = [e4[i]["supp"] for i in iters]
    ax.plot(iters, lych_s, "o-", color="red", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Avg Suppression Score")
    ax.set_title("Iterative Lychrel Reverse-and-Add Accumulation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/lychrel_iter.png",dpi=150); plt.close()

    # Plot 4: Doubling cumulative
    fig, ax = plt.subplots(figsize=(10,6))
    dk = list(e5.keys()); ds = [e5[k]["supp"] for k in dk]
    ax.plot(range(1,len(dk)+1), ds, "s-", color="blue", linewidth=2, markersize=8)
    ax.set_xlabel("Cumulative Step (k=1,2,4,8,16,32 added)")
    ax.set_ylabel("Avg Suppression Score")
    ax.set_title("Cumulative Doubling-Frequency Injection")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/doubling_cum.png",dpi=150); plt.close()

    # Plot 5: Prime frequencies suppression scatter
    fig, ax = plt.subplots(figsize=(14, 6))
    prime_ks = [v["k"] for k, v in e6.items() if k.startswith("prime_") and k.endswith("_d")]
    prime_ss = [v["supp"] for k, v in e6.items() if k.startswith("prime_") and k.endswith("_d")]
    thirteen_ks = [v["k"] for k, v in e6.items() if k.startswith("thirteen_") and k.endswith("_d")]
    thirteen_ss = [v["supp"] for k, v in e6.items() if k.startswith("thirteen_") and k.endswith("_d")]
    arch_ks_plot = arch_k
    arch_ss_plot = [e2.get(f"arch_k{k}_d", {}).get("supp", 0) for k in arch_ks_plot]
    ax.scatter(prime_ks, prime_ss, c="red", label=f"Primes (avg={prime_avg:.4f})", s=40, alpha=0.7, zorder=3)
    ax.scatter(thirteen_ks, thirteen_ss, c="blue", label=f"13-multiples (avg={thirteen_avg:.4f})", s=60, marker="s", alpha=0.7, zorder=3)
    ax.scatter(arch_ks_plot, arch_ss_plot, c="green", label=f"Powers of 2 (avg={arch_avg:.4f})", s=80, marker="D", alpha=0.8, zorder=3)
    ax.set_xlabel("Spatial Frequency k (diagonal)", fontsize=12)
    ax.set_ylabel("Avg Suppression Score", fontsize=12)
    ax.set_title("Prime vs Architecture-Aligned Frequencies — Suppression Comparison", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/primes_vs_arch.png", dpi=150, bbox_inches="tight"); plt.close()

    # SAVE
    json_path = f"{OUTPUT_DIR}/numeric_injection.json"
    with open(json_path, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")
    csv_path = f"{OUTPUT_DIR}/numeric_injection.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            all_fields = set()
            for row in csv_rows: all_fields.update(row.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_fields)); w.writeheader()
            for row in csv_rows: w.writerow({k: row.get(k, "") for k in sorted(all_fields)})
    print(f"Saved CSV: {csv_path}")
    print("\nDONE")

if __name__ == "__main__":
    main()
