#!/usr/bin/env python3
"""
Live webcam embedding capture for adversarial clothing.

Streams your webcam through faithful Darknet YOLOv3, detects persons,
extracts per-layer embeddings from the person crop, and estimates distance
from bbox size. Saves crops + embeddings for later person-vs-background FFT
analysis.

Controls:
  q     quit
  s     save current person frame (if person detected)
  b     save current background frame (no person)
  a     toggle auto-save when person confidence > 0.8
  r     toggle video recording
  space pause/unpause
"""
import os, sys, time, argparse, csv, datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3"))
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import non_max_suppression

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "yolov3.weights")
CFG = os.path.join(BASE, "AdvReal/detlib/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 416
PERSON_CLASS = 0

# Layers to extract embeddings from. Mix of early, mid, FPN, detection-head.
SELECTED_LAYERS = [
    'module_list.0.conv_0',
    'module_list.10.conv_10',
    'module_list.41.conv_41',
    'module_list.54.conv_54',  # deep backbone 3x3, high curvature
    'module_list.59.conv_59',
    'module_list.60.conv_60',  # last 3x3 at 26x26, highest Frobenius
    'module_list.63.conv_63',  # backbone-to-neck handoff, sharp loss landscape
    'module_list.80.conv_80',
    'module_list.87.conv_87',
    'module_list.88.conv_88',
    'module_list.89.conv_89',
    'module_list.90.conv_90',
    'module_list.91.conv_91',
    'module_list.92.conv_92',  # neck last 3x3 before 26x26 head
    'module_list.99.conv_99',
    'module_list.100.conv_100',
    'module_list.101.conv_101',
    'module_list.102.conv_102',
    'module_list.103.conv_103',
    'module_list.104.conv_104',
    'module_list.105.conv_105',
]


def load_model_and_hooks():
    print(f"Loading YOLOv3 on {DEVICE}...")
    model = load_model(model_path=CFG, weights_path=WEIGHTS).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    features = {}
    hooks = []
    modules = dict(model.named_modules())
    def make_hook(name):
        def fn(m, i, o):
            if isinstance(o, torch.Tensor):
                features[name] = o.detach()
        return fn
    for name in SELECTED_LAYERS:
        if name in modules:
            hooks.append(modules[name].register_forward_hook(make_hook(name)))
    return model, features, hooks


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return t


def estimate_distance(bbox_h, frame_h):
    ratio = bbox_h / frame_h
    if ratio > 0.5:
        return 'near'
    elif ratio > 0.2:
        return 'mid'
    else:
        return 'far'


def extract_embeddings(features, classification=None):
    """Return dict layer_name -> per-channel mean vector (numpy)."""
    embs = {}
    for name in SELECTED_LAYERS:
        if name not in features:
            continue
        f = features[name][0]  # (C, H, W)
        embs[name] = f.mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
    return embs


def draw_embedding_bars(frame, embs, layer_name='module_list.104.conv_104', max_bars=64):
    """Draw a small bar chart of channel means for the selected layer."""
    if layer_name not in embs:
        return frame
    vec = embs[layer_name]
    vec = vec[:max_bars]
    h, w = frame.shape[:2]
    bar_w = w // (len(vec) + 2)
    bar_h_max = 80
    v_min, v_max = vec.min(), vec.max()
    rng = max(v_max - v_min, 1e-6)
    norm = (vec - v_min) / rng
    for i, val in enumerate(norm):
        bh = int(val * bar_h_max)
        x = (i + 1) * bar_w
        y1 = h - 20
        y0 = y1 - bh
        color = (0, int(255 * val), int(255 * (1 - val)))
        cv2.rectangle(frame, (x, y0), (x + bar_w - 1, y1), color, -1)
    cv2.putText(frame, f"embed: {layer_name}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def save_capture(save_dir, prefix, frame, embs, bbox, conf, distance, features):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = os.path.join(save_dir, prefix, ts)
    os.makedirs(out, exist_ok=True)

    # Save full frame with overlay
    cv2.imwrite(os.path.join(out, 'frame.jpg'), frame)

    # Save cropped person
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out, 'crop.jpg'), crop)

    # Save embeddings
    np.savez_compressed(os.path.join(out, 'embeddings.npz'), **embs)

    # Save 1D FFT of embedding (polynomial treatment)
    emb_fft = {f"fft_{k}": np.fft.fft(v) for k, v in embs.items()}
    np.savez_compressed(os.path.join(out, 'fft_1d.npz'), **{k: v.astype(np.complex64) for k, v in emb_fft.items()})

    # Save 2D FFT of person crop and key feature maps
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        crop_fft = np.fft.rfft2(crop_gray)
        np.savez_compressed(os.path.join(out, 'fft_2d.npz'),
                            crop_magnitude=np.abs(crop_fft).astype(np.float16),
                            crop_phase=np.angle(crop_fft).astype(np.float16))

    # Save raw feature maps for selected layers
    for name in SELECTED_LAYERS:
        if name in features:
            f = features[name][0].cpu().numpy().astype(np.float16)
            np.save(os.path.join(out, f'{name.replace(".","_")}.npy'), f)
            f_mean = f.mean(axis=0)
            f_fft = np.fft.rfft2(f_mean)
            np.savez_compressed(os.path.join(out, f'{name.replace(".","_")}_fft2d.npz'),
                                magnitude=np.abs(f_fft).astype(np.float16),
                                phase=np.angle(f_fft).astype(np.float16))

    # Metadata
    meta = {
        'timestamp': ts,
        'prefix': prefix,
        'bbox': bbox.tolist() if bbox is not None else None,
        'conf': float(conf),
        'distance': distance,
        'layers': list(embs.keys())
    }
    import json
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=0, help=' webcam index')
    parser.add_argument('--bg_interval', type=float, default=0.5, help='seconds between background captures')
    parser.add_argument('--person_interval', type=float, default=0.25, help='seconds between person captures')
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--save_dir', type=str, default='embeddings_capture')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--auto_save', action='store_true')
    args = parser.parse_args()

    save_dir = os.path.join(BASE, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'person'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'background'), exist_ok=True)

    # CSV log
    csv_path = os.path.join(save_dir, 'log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'type', 'conf', 'distance', 'bbox', 'folder'])

    model, features, hooks = load_model_and_hooks()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open webcam {args.source}")
        return

    print("\nControls:")
    print("  q = quit")
    print("  s = save person frame")
    print("  b = save background frame")
    print("  a = toggle auto-save")
    print("  r = toggle recording")
    print("  space = pause\n")

    auto_save = args.auto_save
    recording = args.record
    paused = False
    writer = None
    last_bg_save = 0
    last_person_save = 0
    last_distance = 'none'
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        else:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' ') or key == 27 or key == ord('q'):
                paused = False
                if key == 27 or key == ord('q'):
                    break
            continue

        features.clear()
        t = preprocess(frame)
        with torch.no_grad():
            det = model(t)
            if isinstance(det, (tuple, list)):
                det = det[0]
            preds = non_max_suppression(det, args.conf, 0.45)

        person_bbox = None
        person_conf = 0.0
        distance = 'none'

        if preds[0] is not None:
            for p in preds[0]:
                cls_id = int(p[5])
                conf = float(p[4])
                if cls_id == PERSON_CLASS and conf > person_conf:
                    person_conf = conf
                    ih, iw = frame.shape[:2]
                    scale_x = iw / IMG_SIZE
                    scale_y = ih / IMG_SIZE
                    person_bbox = np.array([
                        p[0].item() * scale_x,
                        p[1].item() * scale_y,
                        p[2].item() * scale_x,
                        p[3].item() * scale_y
                    ])

        if person_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in person_bbox]
            distance = estimate_distance(y2 - y1, frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {person_conf:.2f} {distance}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Extract embeddings
        embs = extract_embeddings(features)
        if embs:
            frame = draw_embedding_bars(frame, embs)

        # Status text
        status = f"auto={'ON' if auto_save else 'OFF'} rec={'ON' if recording else 'OFF'} frame={frame_count}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('YOLOv3 Embedding Stream', frame)

        # Auto-save logic
        now = time.time()
        if auto_save:
            if person_bbox is not None and person_conf >= args.conf:
                # Save full frame + crop every person_interval seconds
                if now - last_person_save >= args.person_interval or distance != last_distance:
                    folder = save_capture(save_dir, 'person', frame, embs, person_bbox, person_conf, distance, features)
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([datetime.datetime.now().isoformat(), 'person', person_conf, distance, person_bbox.tolist(), folder])
                    print(f"auto-saved person: {distance} conf={person_conf:.2f}")
                    last_person_save = now
                    last_distance = distance
            else:
                # Save background every bg_interval seconds
                if now - last_bg_save >= args.bg_interval:
                    folder = save_capture(save_dir, 'background', frame, embs, None, 0.0, 'none', features)
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([datetime.datetime.now().isoformat(), 'background', 0.0, 'none', 'None', folder])
                    print(f"auto-saved background")
                    last_bg_save = now

        # Recording
        if recording:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(os.path.join(save_dir, 'recording.avi'), fourcc, 20.0, (w, h))
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s') and person_bbox is not None:
            folder = save_capture(save_dir, 'person', frame, embs, person_bbox, person_conf, distance, features)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.datetime.now().isoformat(), 'person', person_conf, distance, person_bbox.tolist(), folder])
            print(f"saved person: {distance} conf={person_conf:.2f} -> {folder}")
        elif key == ord('b'):
            folder = save_capture(save_dir, 'background', frame, embs, None, 0.0, 'none', features)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.datetime.now().isoformat(), 'background', 0.0, 'none', 'None', folder])
            print(f"saved background -> {folder}")
        elif key == ord('a'):
            auto_save = not auto_save
            print(f"auto-save: {'ON' if auto_save else 'OFF'}")
        elif key == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
            print(f"recording: {'ON' if recording else 'OFF'}")
        elif key == ord(' '):
            paused = True
            print("paused")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    for i in range(4):
        cv2.waitKey(1)
    for h in hooks:
        h.remove()
    print(f"Done. Saved captures to {save_dir}")


if __name__ == '__main__':
    main()
