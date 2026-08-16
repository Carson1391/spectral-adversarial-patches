"""
Batch multi-image L2 / FFT / KFAC analysis across COCO person images.
Samples 15 diverse person images, runs L2/FFT/KFAC on each, aggregates.
Outputs to outputs_clothing/batch_multi_image/
"""
import os, sys, csv, json, glob, random, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types; sys.modules["imgaug"] = _types.ModuleType("imgaug")
from pytorchyolo.models import Darknet
from l2_fft_laplacian_kfac import (
    load_image, forward_capture_all, reconstruct_input,
    analyze_l2_error, analyze_fft, compute_kfac, ADV_ANNOTATIONS,
)

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMAGE_DIR    = r"C:\Users\carso\Desktop\YODO\data\coco_person\images"
OUTPUT_DIR   = r"C:\Users\carso\Desktop\YODO\outputs_clothing\batch_multi_image"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 416; CLASS_0 = 0; N_ITER = 200; LR = 0.05; TV_WEIGHT = 1e-4
RECON_LAYERS = [0, 1, 5, 12, 37, 54, 60, 62, 63, 75, 81, 84, 92, 93, 105]
KFAC_SAMPLES = 256; N_IMAGES = 15; RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "l2_error_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fft_spectra"), exist_ok=True)


def select_diverse_images(image_dir, n=15, seed=42):
    all_imgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if len(all_imgs) <= n: return all_imgs
    rng = random.Random(seed)
    sizes = sorted([(os.path.getsize(p), p) for p in all_imgs])
    bins = np.array_split(range(len(sizes)), n)
    return [sizes[rng.choice(b)][1] for b in bins]


def run_single_image(model, img_path, img_idx):
    print(f"\n{'='*60}\nImage {img_idx+1}/{N_IMAGES}: {os.path.basename(img_path)}\n{'='*60}")
    img_tensor, _ = load_image(img_path, IMG_SIZE)
    captured = forward_capture_all(model, img_tensor)
    l2_res, fft_res, kfac_res = [], [], []
    for li in RECON_LAYERS:
        if li not in captured: continue
        target = captured[li]["output"].detach()
        recon, loss = reconstruct_input(model, target, li, IMG_SIZE, DEVICE, n_iter=N_ITER, lr=LR, tv_weight=TV_WEIGHT)
        save_l2 = os.path.join(OUTPUT_DIR, "l2_error_maps") if img_idx == 0 else OUTPUT_DIR
        save_fft = os.path.join(OUTPUT_DIR, "fft_spectra") if img_idx == 0 else OUTPUT_DIR
        l2 = analyze_l2_error(recon, img_tensor, li, save_l2)
        l2["img_idx"] = img_idx; l2["img_name"] = os.path.basename(img_path)
        l2_res.append(l2)
        fft = analyze_fft(recon, img_tensor, li, save_fft)
        fft["img_idx"] = img_idx; fft["img_name"] = os.path.basename(img_path)
        fft_res.append(fft)
        kf = compute_kfac(model, captured, img_tensor, li, DEVICE, n_samples=KFAC_SAMPLES)
        if kf:
            kf["img_idx"] = img_idx; kf["img_name"] = os.path.basename(img_path)
            kfac_res.append(kf)
        hflf = fft["hf_survival"] / (fft["lf_survival"] + 1e-8)
        kt = kf["trace_fisher"] if kf else 0
        print(f"  L{li:3d}: L2={l2['mean_l2']:.4f}  FFT={fft['spec_correlation']:.4f}  HF/LF={hflf:.1f}  KFAC={kt:.2e}")
        if DEVICE == "cuda": torch.cuda.empty_cache()
    return l2_res, fft_res, kfac_res


def aggregate(results, metric_keys):
    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer_idx"], []).append(r)
    agg = []
    for li in sorted(by_layer.keys()):
        rows = by_layer[li]
        e = {"layer_idx": li, "n_images": len(rows)}
        for k in metric_keys:
            vals = [r[k] for r in rows]
            e[f"{k}_mean"] = float(np.mean(vals))
            e[f"{k}_std"] = float(np.std(vals))
            e[f"{k}_min"] = float(np.min(vals))
            e[f"{k}_max"] = float(np.max(vals))
        agg.append(e)
    return agg


def plot_cross_image(agg_l2, agg_fft, agg_kfac, out_dir):
    sn = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'],'')[:18]}" for r in agg_l2]
    x = range(len(agg_l2))

    # L2 cross-image
    fig, axes = plt.subplots(3, 1, figsize=(22, 20))
    ax = axes[0]
    m = [r["mean_l2_mean"] for r in agg_l2]; s = [r["mean_l2_std"] for r in agg_l2]
    ax.bar(x, m, yerr=s, color=plt.cm.RdYlGn_r(np.linspace(0,1,len(agg_l2))), edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(sn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean L2", fontsize=12)
    ax.set_title(f"Cross-Image L2 Error (mean+/-std, n={agg_l2[0]['n_images']})", fontsize=13)
    min_i = m.index(min(m)); max_i = m.index(max(m))
    ax.annotate(f"MIN: {m[min_i]:.4f}", xy=(min_i,m[min_i]), xytext=(min_i+1,m[min_i]*0.5), arrowprops=dict(arrowstyle="->",color="green",lw=2), fontsize=9, color="green", fontweight="bold")
    ax.annotate(f"MAX: {m[max_i]:.4f}", xy=(max_i,m[max_i]), xytext=(max_i+1,m[max_i]*1.05), arrowprops=dict(arrowstyle="->",color="red",lw=2), fontsize=9, color="red", fontweight="bold")

    ax = axes[1]; w = 0.35
    cm = [r["center_mean_l2_mean"] for r in agg_l2]; cs = [r["center_mean_l2_std"] for r in agg_l2]
    bm = [r["border_mean_l2_mean"] for r in agg_l2]; bs = [r["border_mean_l2_std"] for r in agg_l2]
    ax.bar([i-w/2 for i in x], cm, w, yerr=cs, label="Center", color="steelblue", edgecolor="black", linewidth=0.3, capsize=4)
    ax.bar([i+w/2 for i in x], bm, w, yerr=bs, label="Border", color="coral", edgecolor="black", linewidth=0.3, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(sn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("L2 Error", fontsize=12); ax.set_title("Center vs Border (cross-image)", fontsize=13); ax.legend(fontsize=11)

    ax = axes[2]
    ratios = [c/(b+1e-8) for c,b in zip(cm,bm)]
    rstds = [ratio*np.sqrt((cs[i]/(cm[i]+1e-8))**2+(bs[i]/(bm[i]+1e-8))**2) for i,ratio in enumerate(ratios)]
    ax.bar(x, ratios, yerr=rstds, color=plt.cm.coolwarm(np.linspace(0,1,len(ratios))), edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(sn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Center/Border Ratio", fontsize=12)
    ax.set_title("C/B Ratio (cross-image) — HIGH = person more vulnerable", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Uniform")
    ax.legend(fontsize=11)
    fig.suptitle("Cross-Image L2 Analysis — Patch Robustness", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95]); p=os.path.join(out_dir,"cross_image_l2.png"); fig.savefig(p,dpi=150); plt.close(fig); print(f"Saved: {p}")

    # FFT cross-image
    fig, axes = plt.subplots(3, 1, figsize=(22, 20))
    fsn = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'],'')[:18]}" for r in agg_fft]
    fx = range(len(agg_fft))
    ax = axes[0]
    sm = [r["spec_correlation_mean"] for r in agg_fft]; ss = [r["spec_correlation_std"] for r in agg_fft]
    ax.bar(fx, sm, yerr=ss, color=plt.cm.plasma(np.linspace(0,1,len(agg_fft))), edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(fx); ax.set_xticklabels(fsn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spectral Corr", fontsize=12); ax.set_title(f"Cross-Image FFT Corr (n={agg_fft[0]['n_images']})", fontsize=13)

    ax = axes[1]; w=0.35
    lm = [r["lf_survival_mean"] for r in agg_fft]; ls = [r["lf_survival_std"] for r in agg_fft]
    hm = [r["hf_survival_mean"] for r in agg_fft]; hs = [r["hf_survival_std"] for r in agg_fft]
    ax.bar([i-w/2 for i in fx], lm, w, yerr=ls, label="LF survival", color="steelblue", edgecolor="black", linewidth=0.3, capsize=4)
    ax.bar([i+w/2 for i in fx], hm, w, yerr=hs, label="HF survival", color="coral", edgecolor="black", linewidth=0.3, capsize=4)
    ax.set_xticks(fx); ax.set_xticklabels(fsn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Power Ratio", fontsize=12); ax.set_title("LF vs HF Survival (cross-image)", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5); ax.legend(fontsize=11)

    ax = axes[2]
    hflf = [h/(l+1e-8) for h,l in zip(hm,lm)]
    hflf_s = [r*np.sqrt((hs[i]/(hm[i]+1e-8))**2+(ls[i]/(lm[i]+1e-8))**2) for i,r in enumerate(hflf)]
    ax.bar(fx, hflf, yerr=hflf_s, color=plt.cm.coolwarm(np.linspace(0,1,len(hflf))), edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(fx); ax.set_xticklabels(fsn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("HF/LF Ratio", fontsize=12)
    ax.set_title("HF/LF Ratio (cross-image) — consistent = universal patch viable", fontsize=13)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5); ax.legend(fontsize=11)
    fig.suptitle("Cross-Image FFT — Patch Frequency Design Validation", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95]); p=os.path.join(out_dir,"cross_image_fft.png"); fig.savefig(p,dpi=150); plt.close(fig); print(f"Saved: {p}")

    # KFAC cross-image
    if not agg_kfac: return
    ksn = [f"L{r['layer_idx']}\n{ADV_ANNOTATIONS.get(r['layer_idx'],'')[:18]}" for r in agg_kfac]
    kx = range(len(agg_kfac))
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    ax = axes[0,0]
    tm = [r["trace_fisher_mean"] for r in agg_kfac]; ts = [r["trace_fisher_std"] for r in agg_kfac]
    mx = max(tm) if max(tm)>0 else 1
    cols = ["#e74c3c" if v>mx*0.3 else ("#cccccc" if v==0 else "#3498db") for v in tm]
    ax.bar(kx, tm, yerr=ts, color=cols, edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(kx); ax.set_xticklabels(ksn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("trace(F)", fontsize=12); ax.set_title(f"Cross-Image KFAC Trace (n={agg_kfac[0]['n_images']})", fontsize=13)
    ax.set_yscale("log")
    mi = tm.index(max(tm))
    ax.annotate(f"MAX: {tm[mi]:.2e}", xy=(mi,tm[mi]), xytext=(mi+1,tm[mi]*3), arrowprops=dict(arrowstyle="->",color="red",lw=2), fontsize=9, color="red", fontweight="bold")

    ax = axes[0,1]
    em = [r["max_fisher_eig_mean"] for r in agg_kfac]; es = [r["max_fisher_eig_std"] for r in agg_kfac]
    ax.bar(kx, em, yerr=es, color=cols, edgecolor="black", linewidth=0.3, capsize=5)
    ax.set_xticks(kx); ax.set_xticklabels(ksn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("max eig(F)", fontsize=12); ax.set_title("Cross-Image KFAC Max Eig", fontsize=13); ax.set_yscale("log")

    ax = axes[1,0]
    ra = [r["eff_rank_A_mean"] for r in agg_kfac]; rsa = [r["eff_rank_A_std"] for r in agg_kfac]
    rg = [r["eff_rank_G_mean"] for r in agg_kfac]; rsg = [r["eff_rank_G_std"] for r in agg_kfac]
    ax.bar([i-w/2 for i in kx], ra, w, yerr=rsa, label="eff_rank(A)", color="steelblue", edgecolor="black", linewidth=0.3, capsize=4)
    ax.bar([i+w/2 for i in kx], rg, w, yerr=rsg, label="eff_rank(G)", color="coral", edgecolor="black", linewidth=0.3, capsize=4)
    ax.set_xticks(kx); ax.set_xticklabels(ksn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Effective Rank", fontsize=12); ax.set_title("Cross-Image KFAC Eff Ranks", fontsize=13); ax.legend(fontsize=11)

    ax = axes[1,1]
    ca = [min(r["cond_A_mean"],1e15) for r in agg_kfac]
    cg = [min(r["cond_G_mean"],1e15) if r["cond_G_mean"]!=float('inf') else 1e15 for r in agg_kfac]
    ax.bar([i-w/2 for i in kx], ca, w, label="cond(A)", color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar([i+w/2 for i in kx], cg, w, label="cond(G)", color="coral", edgecolor="black", linewidth=0.3)
    ax.set_xticks(kx); ax.set_xticklabels(ksn, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Condition Number", fontsize=12); ax.set_title("Cross-Image KFAC Cond Numbers", fontsize=13)
    ax.set_yscale("log"); ax.legend(fontsize=11)

    fig.suptitle("Cross-Image KFAC — Patch Leverage Consistency", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95]); p=os.path.join(out_dir,"cross_image_kfac.png"); fig.savefig(p,dpi=150); plt.close(fig); print(f"Saved: {p}")


def main():
    print("="*60)
    print(f"Batch Multi-Image Analysis: {N_IMAGES} COCO person images")
    print(f"Device: {DEVICE}")
    print("="*60)

    print("\nLoading model...")
    model = Darknet(CONFIG_PATH).to(DEVICE)
    model.load_darknet_weights(WEIGHTS_PATH)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f"Model loaded on {DEVICE}")

    images = select_diverse_images(IMAGE_DIR, N_IMAGES, RANDOM_SEED)
    print(f"Selected {len(images)} images")

    all_l2, all_fft, all_kfac = [], [], []
    for i, img_path in enumerate(images):
        l2, fft, kf = run_single_image(model, img_path, i)
        all_l2.extend(l2); all_fft.extend(fft); all_kfac.extend(kf)

    # Save per-image CSVs
    for name, data in [("per_image_l2", all_l2), ("per_image_fft", all_fft), ("per_image_kfac", all_kfac)]:
        if not data: continue
        p = os.path.join(OUTPUT_DIR, f"{name}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
        print(f"Saved: {p}")

    # Aggregate
    agg_l2 = aggregate(all_l2, ["mean_l2","max_l2","center_mean_l2","border_mean_l2","l2_r","l2_g","l2_b"])
    agg_fft = aggregate(all_fft, ["spec_correlation","lf_survival","hf_survival"])
    agg_kfac = aggregate(all_kfac, ["trace_fisher","max_fisher_eig","trace_A","trace_G","max_eig_A","max_eig_G","cond_A","cond_G","eff_rank_A","eff_rank_G"])

    for name, data in [("aggregated_l2", agg_l2), ("aggregated_fft", agg_fft), ("aggregated_kfac", agg_kfac)]:
        if not data: continue
        p = os.path.join(OUTPUT_DIR, f"{name}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
        print(f"Saved: {p}")

    # Plots
    plot_cross_image(agg_l2, agg_fft, agg_kfac, OUTPUT_DIR)

    # Summary
    print(f"\n{'='*60}\nSUMMARY (aggregated across {N_IMAGES} images)\n{'='*60}")
    print(f"\nL2 Error (mean+/-std):")
    print(f"{'Layer':>6} {'Mean L2':>12} {'+/-':>8} {'C/B Ratio':>10}")
    for r in agg_l2:
        ratio = r["center_mean_l2_mean"] / (r["border_mean_l2_mean"] + 1e-8)
        print(f"  {r['layer_idx']:>4} {r['mean_l2_mean']:>12.4f} {r['mean_l2_std']:>8.4f} {ratio:>10.4f}")
    print(f"\nFFT (mean+/-std):")
    print(f"{'Layer':>6} {'Spec Corr':>12} {'+/-':>8} {'HF/LF':>10}")
    for r in agg_fft:
        hflf = r["hf_survival_mean"] / (r["lf_survival_mean"] + 1e-8)
        print(f"  {r['layer_idx']:>4} {r['spec_correlation_mean']:>12.4f} {r['spec_correlation_std']:>8.4f} {hflf:>10.1f}")
    print(f"\nKFAC (mean+/-std):")
    print(f"{'Layer':>6} {'trace(F)':>14} {'+/-':>10} {'max_eig':>14} {'er_G':>8}")
    for r in agg_kfac:
        print(f"  {r['layer_idx']:>4} {r['trace_fisher_mean']:>14.2e} {r['trace_fisher_std']:>10.2e} {r['max_fisher_eig_mean']:>14.2e} {r['eff_rank_G_mean']:>8.1f}")

    # Consistency report
    print(f"\n{'='*60}\nCONSISTENCY REPORT\n{'='*60}")
    # Low std/mean ratio = consistent across images
    for name, agg, key in [("L2 mean", agg_l2, "mean_l2"), ("FFT corr", agg_fft, "spec_correlation"), ("KFAC trace", agg_kfac, "trace_fisher")]:
        print(f"\n{name}:")
        for r in agg:
            cv = r[f"{key}_std"] / (r[f"{key}_mean"] + 1e-12) if r[f"{key}_mean"] != 0 else 0
            consistency = "STABLE" if cv < 0.15 else ("MODERATE" if cv < 0.35 else "VARIABLE")
            print(f"  L{r['layer_idx']:3d}: CV={cv:.3f} {consistency}")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
