#!/usr/bin/env python3
"""
Evaluate v15 AdvReal+DAP patches and export print-ready images to YODO root.
Filenames include person detection % and target misclass %.
"""
import os, argparse, math
from collections import Counter
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from ultralytics import YOLO

COCO = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
        10:'fire hydrant',11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',
        20:'elephant',21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',
        30:'skis',31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',
        38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',
        48:'sandwich',49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'couch',
        58:'potted plant',59:'bed',60:'dining table',61:'toilet',62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',
        68:'microwave',69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',
        78:'hair drier',79:'toothbrush'}


def evaluate_patch(texture_path, mask_path, models, img_files, img_size=640):
    """Evaluate a texture+mask pair. Returns per-model stats."""
    tex = T.ToTensor()(Image.open(texture_path).convert('RGB')).unsqueeze(0).to('cuda')
    mask = Image.open(mask_path).convert('RGB')

    results = {}
    for mname, model in models.items():
        person, other, none = 0, 0, 0
        classes = []
        for img_path in img_files:
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            img_t = T.ToTensor()(img).unsqueeze(0)
            mask_t = T.ToTensor()(mask.resize((img_size, img_size))).unsqueeze(0)[:, 0:1, :, :].to('cuda')
            img_t = img_t.to('cuda')
            tex_r = F.interpolate(tex, (img_size, img_size), mode='bilinear', align_corners=False)
            comp_t = tex_r * mask_t + img_t * (1 - mask_t)
            comp_pil = T.ToPILImage()(comp_t.squeeze(0).clamp(0, 1))
            res = model.predict(comp_pil, verbose=False)[0]
            if len(res.boxes) == 0:
                none += 1
                classes.append((-1, 'none', 0.0))
            else:
                best = int(res.boxes.conf.argmax())
                cls = int(res.boxes.cls[best])
                conf = float(res.boxes.conf[best])
                classes.append((cls, COCO.get(cls, '?'), conf))
                if cls == 0:
                    person += 1
                else:
                    other += 1
        n = len(img_files)
        # Top misclass
        mis = [c for c in classes if c[0] != 0 and c[0] != -1]
        top_name = 'unknown'
        top_conf = 0.0
        if mis:
            top_id = Counter([c[0] for c in mis]).most_common(1)[0][0]
            top_name = COCO.get(top_id, '?')
            top_conf = float(np.mean([c[2] for c in mis if c[0] == top_id]))
        results[mname] = {
            'person_pct': person / n * 100,
            'other_pct': other / n * 100,
            'none_pct': none / n * 100,
            'top_target': top_name,
            'top_conf': top_conf,
        }
    return results


def export_print_ready(texture_path, mask_path, out_path, dpi=300, width_in=12, height_in=16):
    """Export texture+mask as print-ready PNG at given DPI."""
    tex_rgb = Image.open(texture_path).convert('RGB')
    mask_l = Image.open(mask_path).convert('L').resize(tex_rgb.size)
    tex_rgba = tex_rgb.copy()
    tex_rgba.putalpha(mask_l)
    width_px = int(width_in * dpi)
    height_px = int(height_in * dpi)
    tex_print = tex_rgba.resize((width_px, height_px), Image.Resampling.LANCZOS)
    tex_print.save(out_path, dpi=(dpi, dpi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='outputs_clothing/v15_advreal_dap')
    parser.add_argument('--out', default='C:/Users/carso/Desktop/YODO')
    parser.add_argument('--imgs', default='data/coco_person/images')
    parser.add_argument('-n', type=int, default=50, help='Number of test images')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.out)

    # Load models
    models = {}
    if os.path.exists('YOLOv8/yolov8n.pt'):
        models['v8'] = YOLO('YOLOv8/yolov8n.pt')
    if os.path.exists('YOLO11/yolo11n.pt'):
        models['v11'] = YOLO('YOLO11/yolo11n.pt')
    if os.path.exists('YOLOv3/yolov3u.pt'):
        models['v3'] = YOLO('YOLOv3/yolov3u.pt')

    # Test images
    img_files = sorted([os.path.join(args.imgs, f) for f in os.listdir(args.imgs)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:args.n]

    # Find all texture/mask pairs
    pairs = []
    for f in sorted(os.listdir(base_dir)):
        if f.startswith('texture_') and f.endswith('.png'):
            epoch = f[len('texture_'):-len('.png')]
            mask_path = os.path.join(base_dir, f'mask_{epoch}.png')
            if os.path.exists(mask_path):
                pairs.append((epoch, os.path.join(base_dir, f), mask_path))

    # Evaluate final and last few checkpoints
    candidates = []
    for p in pairs:
        epoch_str = p[0]
        if epoch_str == 'final':
            candidates.append(p)
        else:
            try:
                ep_num = int(epoch_str.replace('epoch', '').lstrip('0') or '0')
                if ep_num >= 100:  # only evaluate later epochs
                    candidates.append(p)
            except ValueError:
                pass
    if not candidates and pairs:
        candidates = pairs[-3:]

    print(f'Evaluating {len(candidates)} checkpoints on {len(img_files)} images...')
    all_results = []
    for epoch, tex_path, mask_path in candidates:
        print(f'  Evaluating epoch {epoch}...')
        stats = evaluate_patch(tex_path, mask_path, models, img_files)
        # Average across models
        avg_person = np.mean([s['person_pct'] for s in stats.values()])
        avg_other = np.mean([s['other_pct'] for s in stats.values()])
        # Dominant target
        all_targets = [s['top_target'] for s in stats.values()]
        top_target = Counter(all_targets).most_common(1)[0][0] if all_targets else 'unknown'
        avg_target_conf = np.mean([s['top_conf'] for s in stats.values()])

        all_results.append({
            'epoch': epoch,
            'tex': tex_path,
            'mask': mask_path,
            'stats': stats,
            'avg_person': avg_person,
            'avg_other': avg_other,
            'top_target': top_target,
            'avg_target_conf': avg_target_conf,
        })
        for mname, s in stats.items():
            print(f'    {mname}: person={s["person_pct"]:.1f}% other={s["other_pct"]:.1f}% top={s["top_target"]}({s["top_conf"]:.2f})')

    if not all_results:
        print('No checkpoints found!')
        return

    # Sort by evasion (lowest person detection = best)
    all_results.sort(key=lambda x: x['avg_person'])

    # Export top 3 to YODO root with % in filename
    dpi = 300
    w_in, h_in = 12, 16
    w_px, h_px = int(w_in * dpi), int(h_in * dpi)

    saved = []
    for i, r in enumerate(all_results[:3]):
        person_pct = int(r['avg_person'])
        target_pct = int(r['avg_other'])
        target_name = r['top_target'].replace(' ', '_')
        epoch = r['epoch']
        fname = f"v15_person{person_pct}pct_{target_name}{target_pct}pct_epoch{epoch}_{w_px}x{h_px}_300dpi.png"
        fpath = os.path.join(out_dir, fname)
        export_print_ready(r['tex'], r['mask'], fpath, dpi, w_in, h_in)
        saved.append((fpath, r))
        print(f'Saved: {fname}')

    # Write README
    readme = os.path.join(out_dir, 'v15_BEST_PATCHES_README.txt')
    with open(readme, 'w') as f:
        f.write('AdvReal+DAP Adversarial Clothing Patches (v15)\n')
        f.write('=' * 55 + '\n\n')
        f.write(f'Print size: {w_in} x {h_in} inches at {dpi} DPI ({w_px} x {h_px} px)\n')
        f.write(f'Method: AdvReal max_iou loss + DAP triangle deformable mask\n')
        f.write(f'Models: {list(models.keys())}\n\n')
        for fpath, r in saved:
            f.write(f'File: {os.path.basename(fpath)}\n')
            f.write(f'  Epoch: {r["epoch"]}\n')
            f.write(f'  Avg person detected: {r["avg_person"]:.1f}%\n')
            f.write(f'  Avg misclassified: {r["avg_other"]:.1f}%\n')
            f.write(f'  Top target: {r["top_target"]} (conf {r["avg_target_conf"]:.2f})\n')
            for mname, s in r["stats"].items():
                f.write(f'  {mname}: person={s["person_pct"]:.1f}% other={s["other_pct"]:.1f}% top={s["top_target"]}({s["top_conf"]:.2f})\n')
            f.write('\n')
    print(f'Readme: {readme}')
    print('Done!')


if __name__ == '__main__':
    main()
