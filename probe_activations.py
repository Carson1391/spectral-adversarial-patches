#!/usr/bin/env python3
"""
probe_activations.py — Find person/clothing feature channels in YOLOv3.

Load person images, forward through model, collect activations at all 3 FPN
scales and mid-backbone layers. Rank channels by how much they activate on
person vs background. Save top channels per layer.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")
DEVICE = torch.device('cuda')

# Layers to probe
PROBE_LAYERS = [
    'module_list.36.leaky_36',   # mid-backbone 1 (208px)
    'module_list.61.leaky_61',   # mid-backbone 2 (104px)
    'module_list.80.leaky_80',   # FPN stride 32 (13px)
    'module_list.92.leaky_92',   # FPN stride 16 (26px)
    'module_list.104.leaky_104', # FPN stride 8  (52px)
]


def load_frozen_model():
    print("Loading Darknet YOLOv3...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_person_imgs(dir_path, max_imgs=100):
    files = []
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.jpg', '.png')):
            files.append(os.path.join(dir_path, f))
    files = files[:max_imgs]
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((416, 416))
        imgs.append(np.array(img).transpose(2,0,1) / 255.0)
    return torch.tensor(np.stack(imgs), dtype=torch.float32, device=DEVICE)


def register_hooks(model, layer_names):
    features = {}
    hooks = []
    modules = dict(model.named_modules())
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o
        return fn
    for n in layer_names:
        if n in modules:
            hooks.append(modules[n].register_forward_hook(make_hook(n)))
    return features, hooks


def main():
    model = load_frozen_model()
    features, hooks = register_hooks(model, PROBE_LAYERS)

    person_dir = os.path.join(BASE, 'data', 'coco_person_strong', 'images')
    person_imgs = load_person_imgs(person_dir, max_imgs=100)
    print(f"Loaded {len(person_imgs)} person images")

    print("Probing...")
    batch_size = 8
    layer_acts = {n: [] for n in PROBE_LAYERS}

    try:
        for i in range(0, len(person_imgs), batch_size):
            batch = person_imgs[i:i+batch_size]
            features.clear()
            with torch.no_grad():
                _ = model(batch)
            for n in PROBE_LAYERS:
                if n in features:
                    f = features[n]
                    act = f.mean(dim=(2, 3)).cpu().numpy()
                    layer_acts[n].append(act)
    except Exception as e:
        print(f"ERROR during probing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nTop channels per layer (mean activation on person images):")
    print("\nTop channels per layer (mean activation on person images):")
    print("=" * 80)
    for n in PROBE_LAYERS:
        if len(layer_acts[n]) == 0:
            continue
        acts = np.concatenate(layer_acts[n], axis=0)
        mean_act = acts.mean(axis=0)
        std_act = acts.std(axis=0)
        # Rank by mean + std
        score = mean_act + std_act
        top = np.argsort(score)[::-1][:20]

        print(f"\n{n}  (D={len(mean_act)}, shape from one sample)")
        print(f"  mean={mean_act.mean():.3f} std={mean_act.std():.3f}")
        print(f"  Top 20 channel indices:")
        print(f"  {list(top)}")
        print(f"  Top 20 scores: {score[top].round(3).tolist()}")

    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
