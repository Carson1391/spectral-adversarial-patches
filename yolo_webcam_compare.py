"""
yolo_webcam_compare.py
Live webcam TRACKING comparison of YOLOv8 / YOLO11 / YOLO26 / YOLOv3.

All four models are loaded at startup so switching is instant.

Controls:
  1/2/3/4    : switch active model in SINGLE mode
  m          : toggle SINGLE / ALL-4 grid mode
  c          : cycle class preset (all, road, person, vehicle, two-wheel)
  f          : toggle fast mode (predict vs track)
  e          : toggle embedding overlay display
  t          : toggle CAPTURE / REGULAR mode (or click CAPTURE button)
  b          : start 5s countdown then auto-capture baseline (time to get in/out of frame)
  Shift+B    : instant baseline capture (no countdown)
  + / -      : nudge confidence up/down
  q / Esc    : quit
  Mouse      : click buttons at the bottom / top
"""

import os, csv, time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO

# ----------------------------- CONFIG -----------------------------
ROOT    = r"C:\Users\carso\Desktop\YODO"
SIZE    = "n"            # n / s / m / l / x  (same size across all three)
TRACKER = "bytetrack.yaml"   # or "botsort.yaml"
CAM_INDEX = 0
FRAME_W, FRAME_H = 1920, 1080
IMGSZ = 1280             # inference resolution (independent of display)
USE_HALF = True          # FP16 on GPU (great for RTX 50-series)

COCO_PRESETS = {
    "all":       None,
    "road":      [0, 1, 2, 3, 5, 7, 9, 11, 12],   # person/bicycle/car/motorcycle/bus/truck/traffic light/stop sign/parking meter
    "person":    [0],
    "vehicle":   [2, 3, 5, 7],                      # car/motorcycle/bus/truck
    "two-wheel": [1, 3],                            # bicycle/motorcycle
}

MODELS = [
    ("YOLOv8", rf"{ROOT}\YOLOv8\yolov8{SIZE}.pt"),
    ("YOLO11", rf"{ROOT}\YOLO11\yolo11{SIZE}.pt"),
    ("YOLO26", rf"{ROOT}\YOLO26\yolo26{SIZE}.pt"),
    ("YOLOv3", rf"{ROOT}\YOLOv3\yolov3u.pt"),
]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = USE_HALF and (DEVICE != "cpu") and torch.cuda.is_available()
print(f"[init] Device: {'CUDA:0' if DEVICE == 0 else 'CPU'} | size={SIZE} | tracker={TRACKER} | imgsz={IMGSZ} | half={USE_HALF}")
# Embedding extraction hooks — capture feature maps from detection head input
feat_maps = {}  # model_name -> list of feature tensors

def make_pre_hook(mname):
    def hook(module, inp):
        # Detect head receives a list/tuple of feature maps from neck
        feat_maps[mname] = inp[0] if isinstance(inp, (list, tuple)) and len(inp) > 0 else inp
    return hook

models = []
for name, w in MODELS:
    m = YOLO(w)
    m.to(DEVICE)
    if USE_HALF:
        m.model = m.model.half()
    # Register pre-hook on the Detect module (last layer of the model)
    detect_mod = m.model.model[-1]
    detect_mod.register_forward_pre_hook(make_pre_hook(name))
    models.append((name, m))
    print(f"  {name:<8} -> {w}")
print("[init] Ready.\n")

# ----------------------------- STATE -----------------------------
active = 0
mode = "single"           # "single" or "all"
preset_names = list(COCO_PRESETS.keys())
preset_idx = 1            # start with "road"
CLASSES = COCO_PRESETS[preset_names[preset_idx]]
conf = 0.35
fast_mode = False        # True = predict() without tracker
show_embeddings = True   # toggle with 'e' key

# ----------------------------- EMBEDDING CAPTURE -----------------------------
# YOLOv3 only — comprehensive multi-scale embedding collection
# Modes: "regular" = just detection/tracking display, "capture" = full embedding logging
emb_mode = "regular"   # "regular" or "capture"
show_embeddings = True
# Sampling intervals
WHOLE_FRAME_INTERVAL = 0.5   # seconds
PERSON_INTERVAL = 0.25       # seconds
# Baselines (captured on press 'b' or CAPTURE button)
wf_baseline = None           # list of per-scale L2-normalized GAP vectors
person_baseline = {}         # track_id -> list of per-scale L2-normalized bbox vectors
person_baseline_bbox = {}    # track_id -> (cx, cy) for spatial fallback matching
prev_track_ids = set()       # for detecting track ID switches
track_switches = []          # list of (timestamp, old_ids, new_ids)
# Live state
live_wf_scales = []          # list of (raw_vec, l2_norm, normalized_vec) per scale
live_person_scales = {}      # track_id -> list of (raw_vec, l2_norm, normalized_vec) per scale
live_wf_cos = 1.0
live_wf_shift = 0.0
last_whole_sample = 0.0
last_person_sample = 0.0
# Baseline countdown — press 'b', then N seconds to get in/out of frame
baseline_countdown_end = 0.0   # timestamp when countdown expires (0 = not counting)
BASELINE_DELAY = 5.0           # seconds to wait before auto-capturing baseline
baseline_label = ""            # label shown on screen during countdown
# CSV + image logging
LOG_DIR = os.path.join(ROOT, "outputs_clothing", "webcam_emb_logs")
CROP_DIR = os.path.join(LOG_DIR, "crops")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
log_session_ts = time.strftime("%Y%m%d_%H%M%S")
log_csv_path = os.path.join(LOG_DIR, f"emb_{log_session_ts}.csv")
log_file = None
log_writer = None
header_written = False
scale_dims = []  # detected at first sample, e.g. [256, 512, 1024] for YOLOv3

# ----------------------------- UI THEME -----------------------------
# BGR colors
C_BG        = (22, 24, 26)
C_PANEL     = (38, 42, 46)
C_PANEL_L   = (52, 58, 64)
C_ACCENT    = (0, 170, 255)      # amber/orange
C_ACCENT_2  = (0, 220, 120)      # green
C_TEXT      = (235, 238, 242)
C_TEXT_DIM  = (155, 160, 168)
C_BORDER    = (75, 80, 88)
C_RED       = (70, 70, 240)
C_YELLOW    = (60, 205, 255)
C_WHITE     = (255, 255, 255)

BTN_H = 26
BTN_GAP = 10
TOP_Y = 14
BOT_Y = FRAME_H - 32

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------- UI HELPERS -----------------------------
def text_size(text, scale, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0]

def draw_text(frame, text, x, y, scale=0.6, color=C_TEXT, thickness=1, align="left", shadow=True):
    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def draw_pill(frame, x1, y1, x2, y2, color, active=False):
    r = 6
    # body
    cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
    # caps
    cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
    # border
    if active:
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), C_ACCENT, 1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), C_ACCENT, 1)
        cv2.circle(frame, (x1 + r, y1 + r), r, C_ACCENT, 1)
        cv2.circle(frame, (x2 - r, y1 + r), r, C_ACCENT, 1)
        cv2.circle(frame, (x1 + r, y2 - r), r, C_ACCENT, 1)
        cv2.circle(frame, (x2 - r, y2 - r), r, C_ACCENT, 1)

def overlay_alpha(frame, rect, color, alpha=0.7):
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2].astype(np.float32)
    col = np.array(color, dtype=np.float32)
    blended = (1 - alpha) * sub + alpha * col
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

def preset_label():
    p = preset_names[preset_idx]
    cls = CLASSES if CLASSES is not None else list(range(80))
    return f"{p} ({len(cls)} classes)"

def button_rects():
    n = len(MODELS)
    total_w = n * 140 + (n - 1) * BTN_GAP
    x = (FRAME_W - total_w) // 2
    rects = []
    for _ in MODELS:
        rects.append((x, TOP_Y, x + 140, TOP_Y + BTN_H))
        x += 140 + BTN_GAP
    return rects

def bottom_buttons():
    # (x1, y1, x2, y2, label, action) — aligned to bottom-right corner
    widths = [80, 90, 70, 70, 100]
    labels = ["MODE", "CLASSES", "CONF+", "CONF-", "CAPTURE"]
    actions = ["toggle_mode", "cycle_preset", "conf_up", "conf_down", "toggle_capture"]
    total = sum(widths) + (len(widths) - 1) * BTN_GAP
    x = FRAME_W - total - 14
    btns = []
    for w, lbl, act in zip(widths, labels, actions):
        btns.append((x, BOT_Y, x + w, BOT_Y + BTN_H, lbl, act))
        x += w + BTN_GAP
    return btns

def on_mouse(event, mx, my, flags, _p):
    global active, mode, conf, emb_mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for x1, y1, x2, y2, label, action in bottom_buttons():
        if x1 <= mx <= x2 and y1 <= my <= y2:
            if action == "toggle_mode":
                mode = "all" if mode == "single" else "single"
            elif action == "cycle_preset":
                cycle_preset()
            elif action == "conf_up":
                conf = min(0.95, conf + 0.05)
            elif action == "conf_down":
                conf = max(0.05, conf - 0.05)
            elif action == "toggle_capture":
                toggle_capture_mode()
            return
    for i, (x1, y1, x2, y2) in enumerate(button_rects()):
        if x1 <= mx <= x2 and y1 <= my <= y2:
            active = i
            mode = "single"  # clicking a model button always exits all-4 mode
            print(f"[switch] active model = {MODELS[active][0]}")
            return

def cycle_preset():
    global preset_idx, CLASSES
    preset_idx = (preset_idx + 1) % len(preset_names)
    CLASSES = COCO_PRESETS[preset_names[preset_idx]]

def draw_panel(frame, result, names):
    h, w = frame.shape[:2]
    rows = []
    for b in result.boxes:
        tid = int(b.id) if b.id is not None else -1
        rows.append((names[int(b.cls)], tid, float(b.conf)))
    rows.sort(key=lambda t: -t[2]); rows = rows[:10]

    panel_w, row_h = 260, 24
    x0 = w - panel_w - 16
    y0 = TOP_Y + BTN_H + 20
    panel_h = 34 + row_h * max(len(rows), 1)

    # semi-transparent background card
    overlay_alpha(frame, (x0, y0, x0 + panel_w, y0 + panel_h), C_PANEL, 0.85)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), C_BORDER, 1)

    # header
    overlay_alpha(frame, (x0, y0, x0 + panel_w, y0 + 28), C_PANEL_L, 0.9)
    draw_text(frame, "TRACKS", x0 + 10, y0 + 20, 0.55, C_ACCENT, 1)
    draw_text(frame, f"{len(rows)} obj", x0 + panel_w - 70, y0 + 20, 0.45, C_TEXT_DIM, 1)

    if not rows:
        draw_text(frame, "none", x0 + 10, y0 + 52, 0.55, C_TEXT_DIM, 1)
        return

    for i, (lbl, tid, c) in enumerate(rows):
        y = y0 + 34 + row_h * i
        bar_w = int((panel_w - 20) * c)
        cv2.rectangle(frame, (x0 + 10, y + 10), (x0 + 10 + bar_w, y + 16),
                      C_ACCENT_2 if tid >= 0 else C_ACCENT, -1)
        idtxt = f"#{tid}" if tid >= 0 else "#-"
        draw_text(frame, f"{lbl}", x0 + 12, y + 6, 0.5, C_TEXT, 1)
        draw_text(frame, f"{idtxt} {c:.0%}", x0 + panel_w - 78, y + 6, 0.5, C_TEXT, 1)

def draw_top_buttons(frame):
    for i, (x1, y1, x2, y2) in enumerate(button_rects()):
        on = (i == active) and (mode == "single")
        color = C_ACCENT if on else C_PANEL
        draw_pill(frame, x1, y1, x2, y2, color, active=on)
        lbl = f"{i+1}  {MODELS[i][0]}"
        tw, th = text_size(lbl, 0.6, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y2 - 10
        draw_text(frame, lbl, tx, ty, 0.6, C_WHITE if on else C_TEXT, 1, shadow=True)

def draw_bottom_bar(frame, fps, n, frame_w, frame_h):
    # buttons on the bottom-right, status on the bottom-left
    for x1, y1, x2, y2, label, _ in bottom_buttons():
        on = (label == "MODE" and mode == "all")
        draw_pill(frame, x1, y1, x2, y2, C_ACCENT if on else C_PANEL, active=on)
        tw, th = text_size(label, 0.50, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y2 - 7
        draw_text(frame, label, tx, ty, 0.50, C_WHITE if on else C_TEXT, 1)

    ty = BOT_Y + BTN_H - 8
    # left-aligned mode/model label
    left = f"{mode.upper()}  |  {MODELS[active][0]} {SIZE}"
    draw_text(frame, left, 14, ty, 0.55, C_ACCENT, 1)

    # small info labels next to the mode label
    info = [
        ("FPS", f"{fps:5.1f}"),
        ("CONF", f"{conf:.2f}"),
        ("TRACKS", str(n)),
        ("PRESET", preset_label().upper()),
        ("CAM", f"{frame_w}x{frame_h}"),
        ("INFER", str(IMGSZ)),
        ("HALF", str(USE_HALF)),
        ("FAST", str(fast_mode)),
    ]
    x = 14 + text_size(left, 0.55, 1)[0] + 20
    for label, val in info:
        txt = f"{label}:{val}"
        draw_text(frame, txt, x, ty, 0.48, C_TEXT if label == "FPS" else C_TEXT_DIM, 1)
        x += text_size(txt, 0.48, 1)[0] + 14

def extract_all_scales(mname, bbox=None, img_w=None, img_h=None):
    """Extract per-scale embeddings from all detection head feature maps.
    If bbox is None: whole-frame GAP per scale.
    If bbox given: average-pool the bbox region per scale.
    Returns list of dicts: [{raw, l2, norm, fH, fW, scale_idx}, ...]
    """
    feats = feat_maps.get(mname)
    if feats is None or not isinstance(feats, (list, tuple)) or len(feats) == 0:
        return []
    results = []
    for i, feat in enumerate(feats):
        if feat is None:
            continue
        fH, fW = feat.shape[2], feat.shape[3]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            fx1 = max(0, int(x1 / img_w * fW))
            fy1 = max(0, int(y1 / img_h * fH))
            fx2 = min(fW, max(fx1 + 1, int(x2 / img_w * fW)))
            fy2 = min(fH, max(fy1 + 1, int(y2 / img_h * fH)))
            raw = feat[0, :, fy1:fy2, fx1:fx2].float().mean(dim=(1, 2)).cpu()
        else:
            raw = feat[0].float().mean(dim=(1, 2)).cpu()
        l2 = float(raw.norm())
        norm = F.normalize(raw, dim=0) if l2 > 0 else raw
        results.append({'raw': raw, 'l2': l2, 'norm': norm, 'fH': fH, 'fW': fW, 'scale_idx': i})
    return results

def track_one(model, frame):
    global live_wf_scales, live_person_scales, live_wf_cos, live_wf_shift
    global last_whole_sample, last_person_sample, scale_dims
    global prev_track_ids, track_switches, log_writer, header_written
    mname = MODELS[active][0] if mode == "single" else None
    if fast_mode:
        r = model.predict(frame, conf=conf, imgsz=IMGSZ, classes=CLASSES,
                          device=DEVICE, half=USE_HALF, verbose=False)[0]
    else:
        r = model.track(frame, conf=conf, imgsz=IMGSZ, classes=CLASSES,
                        tracker=TRACKER, persist=True, device=DEVICE,
                        half=USE_HALF, verbose=False)[0]
    annotated = r.plot()
    draw_panel(annotated, r, model.names)
    if not show_embeddings or mname is None or emb_mode != "capture":
        return annotated, r
    img_h, img_w = frame.shape[:2]
    now = time.time()
    persons = [b for b in r.boxes if int(b.cls) == 0]
    # --- Detect track ID switches ---
    current_ids = set(int(b.id) if b.id is not None else -1 for b in persons)
    if prev_track_ids and current_ids and current_ids != prev_track_ids:
        lost = prev_track_ids - current_ids
        gained = current_ids - prev_track_ids
        if lost or gained:
            track_switches.append((now, sorted(lost), sorted(gained)))
            print(f"[track] SWITCH: lost={sorted(lost)} gained={sorted(gained)}")
    prev_track_ids = current_ids
    # --- Auto-sample whole-frame every 0.5s ---
    if now - last_whole_sample >= WHOLE_FRAME_INTERVAL:
        last_whole_sample = now
        wf = extract_all_scales(mname)
        if wf:
            live_wf_scales = wf
            if not scale_dims:
                scale_dims = [s['raw'].shape[0] for s in wf]
                print(f"[emb] YOLOv3 scale dims: {scale_dims} (total={sum(scale_dims)})")
            # Compute per-scale and overall cosine to baseline
            if wf_baseline is not None and len(wf_baseline) == len(wf):
                per_scale_cos = []
                for live_s, base_s in zip(wf, wf_baseline):
                    c = float(F.cosine_similarity(live_s['norm'].unsqueeze(0), base_s.unsqueeze(0))[0])
                    per_scale_cos.append(c)
                # Overall = cosine of concatenated normalized vectors
                live_cat = torch.cat([s['norm'] for s in wf], dim=0)
                base_cat = torch.cat(wf_baseline, dim=0)
                live_wf_cos = float(F.cosine_similarity(live_cat.unsqueeze(0), base_cat.unsqueeze(0))[0])
                live_wf_shift = 1.0 - live_wf_cos
                # Log to CSV
                if log_writer is not None and header_written:
                    row = [now, 'WHOLE', '', '', '', '', '', '',
                           live_wf_cos, live_wf_shift, len(persons), len(track_switches)]
                    row += [f"{c:.6f}" for c in per_scale_cos]
                    row += [f"{s['l2']:.4f}" for s in wf]
                    row += [v.item() for s in wf for v in s['norm']]
                    row += ['']
                    log_writer.writerow(row)
    # --- Auto-sample per-person every 0.25s ---
    if persons and (now - last_person_sample >= PERSON_INTERVAL):
        last_person_sample = now
        live_person_scales = {}
        for b in persons:
            tid = int(b.id) if b.id is not None else -1
            bbox = b.xyxy[0].cpu().numpy()
            conf_val = float(b.conf)
            scales = extract_all_scales(mname, bbox, img_w, img_h)
            if not scales:
                continue
            live_person_scales[tid] = scales
            # Per-scale cosine to baseline
            per_scale_cos = [1.0] * len(scales)
            overall_cos = 1.0
            overall_shift = 0.0
            # Match to baseline: exact track ID first, then nearest bbox center
            matched_baseline = None
            matched_tid = tid
            if tid in person_baseline and len(person_baseline[tid]) == len(scales):
                matched_baseline = person_baseline[tid]
            elif tid == -1 and person_baseline:
                # Fast mode: no track ID — match by nearest bbox center
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                best_dist = float('inf')
                for btid, bscales in person_baseline.items():
                    if len(bscales) != len(scales):
                        continue
                    bcx, bcy = person_baseline_bbox.get(btid, (0, 0))
                    dist = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        matched_baseline = bscales
                        matched_tid = btid
            if matched_baseline is not None and len(matched_baseline) == len(scales):
                per_scale_cos = []
                for live_s, base_s in zip(scales, matched_baseline):
                    c = float(F.cosine_similarity(live_s['norm'].unsqueeze(0), base_s.unsqueeze(0))[0])
                    per_scale_cos.append(c)
                live_cat = torch.cat([s['norm'] for s in scales], dim=0)
                base_cat = torch.cat(matched_baseline, dim=0)
                overall_cos = float(F.cosine_similarity(live_cat.unsqueeze(0), base_cat.unsqueeze(0))[0])
                overall_shift = 1.0 - overall_cos
            # Save person crop
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            crop = frame[max(0,y1):y2, max(0,x1):x2].copy()
            crop_path = os.path.join(CROP_DIR, f"{log_session_ts}_t{now:.3f}_id{tid}.jpg")
            cv2.imwrite(crop_path, crop)
            # Log to CSV
            if log_writer is not None and header_written:
                row = [now, 'PERSON', tid, f"{bbox[0]:.1f}", f"{bbox[1]:.1f}",
                       f"{bbox[2]:.1f}", f"{bbox[3]:.1f}", f"{conf_val:.4f}",
                       overall_cos, overall_shift, len(persons), len(track_switches)]
                row += [f"{c:.6f}" for c in per_scale_cos]
                row += [f"{s['l2']:.4f}" for s in scales]
                row += [v.item() for s in scales for v in s['norm']]
                row += [crop_path]
                log_writer.writerow(row)
    # --- Draw whole-frame info (top-left) ---
    y0 = TOP_Y + BTN_H + 20
    if live_wf_scales:
        if wf_baseline is not None:
            lines = [f"WHOLE cos={live_wf_cos:.4f} shift={live_wf_shift:.4f}"]
            for i, s in enumerate(live_wf_scales):
                lines.append(f"  S{i} L2={s['l2']:.2f} {s['fH']}x{s['fW']}")
            color = (0, 220, 120) if live_wf_shift < 0.01 else (60, 205, 255) if live_wf_shift < 0.03 else (70, 70, 240)
        else:
            lines = ["WHOLE FRAME", "NO BASELINE — press b"]
            color = C_TEXT_DIM
        for i, line in enumerate(lines):
            draw_text(annotated, line, 14, y0 + i * 18, 0.5, color, 1)
    # --- Draw per-person info next to each bbox ---
    for b in persons:
        tid = int(b.id) if b.id is not None else -1
        bbox = b.xyxy[0].cpu().numpy()
        x1, y1 = int(bbox[0]), int(bbox[1])
        conf_val = float(b.conf)
        if tid in live_person_scales:
            scales = live_person_scales[tid]
            if tid in person_baseline:
                live_cat = torch.cat([s['norm'] for s in scales], dim=0)
                base_cat = torch.cat(person_baseline[tid], dim=0)
                cos = float(F.cosine_similarity(live_cat.unsqueeze(0), base_cat.unsqueeze(0))[0])
                shift = 1.0 - cos
                p_lines = [f"#{tid} conf={conf_val:.2f} cos={cos:.4f}", f"shift={shift:.4f}"]
                for i, s in enumerate(scales):
                    p_lines.append(f"  S{i} L2={s['l2']:.2f}")
                color = (0, 220, 120) if shift < 0.01 else (60, 205, 255) if shift < 0.03 else (70, 70, 240)
            else:
                p_lines = [f"#{tid} conf={conf_val:.2f} NO BASE"]
                for i, s in enumerate(scales):
                    p_lines.append(f"  S{i} L2={s['l2']:.2f}")
                color = C_TEXT_DIM
        else:
            p_lines = [f"#{tid} sampling..."]
            color = C_TEXT_DIM
        for i, line in enumerate(p_lines):
            draw_text(annotated, line, x1 + 2, y1 - 20 - i * 16, 0.45, color, 1)
    return annotated, r

# ----------------------------- MODES -----------------------------
def run_single(frame):
    name, model = models[active]
    annotated, r = track_one(model, frame)
    draw_top_buttons(annotated)
    draw_bottom_bar(annotated, fps, len(r.boxes), frame.shape[1], frame.shape[0])
    # Mode indicator
    mode_color = (70, 70, 240) if emb_mode == "capture" else C_TEXT_DIM
    draw_text(annotated, f"[{emb_mode.upper()}]", 14, FRAME_H - 50, 0.55, mode_color, 1)
    if emb_mode == "capture":
        n_pbase = len(person_baseline)
        wf_status = "SET" if wf_baseline is not None else "NOT SET"
        n_switches = len(track_switches)
        draw_text(annotated, f"WF:{wf_status} persons:{n_pbase} switches:{n_switches}",
                  80, FRAME_H - 50, 0.5, C_ACCENT_2, 1)
    return annotated

def run_all(frame):
    # 2x2 grid of 960x540 panels -> 1920x1080 total.
    panels = []
    total_tracks = 0
    for name, model in models:
        ann, r = track_one(model, frame)
        total_tracks += len(r.boxes)
        # model badge at top-left of each panel
        badge = f" {name} {SIZE} "
        bw, _ = text_size(badge, 0.65, 1)
        overlay_alpha(ann, (12, 12, 22 + bw, 44), C_BG, 0.8)
        cv2.rectangle(ann, (12, 12), (22 + bw, 44), C_ACCENT, 1)
        draw_text(ann, badge, 20, 36, 0.65, C_ACCENT, 1)
        panel = cv2.resize(ann, (960, 540))
        panels.append(panel)
    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    out = np.vstack([top, bot])
    # Bottom bar on the lower half.
    draw_bottom_bar(out, fps, total_tracks, frame.shape[1], frame.shape[0])
    return out

# ----------------------------- MAIN -----------------------------
def toggle_capture_mode():
    global emb_mode, log_file, log_writer, log_csv_path
    if emb_mode == "regular":
        emb_mode = "capture"
        log_file = open(log_csv_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        # Header — will be written after first sample detects dims
        print(f"[capture] Mode ON — logging to {log_csv_path}")
        print(f"[capture] Press 'b' to capture baseline, 'q' or CAPTURE button to stop")
    else:
        emb_mode = "regular"
        if log_file:
            log_file.close()
            log_file = None
            log_writer = None
        print(f"[capture] Mode OFF — CSV saved to {log_csv_path}")

def write_csv_header():
    global log_writer, scale_dims, header_written
    if log_writer is None or not scale_dims:
        return
    header = ['timestamp', 'type', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
              'conf', 'cos_overall', 'shift_overall', 'n_persons', 'n_track_switches']
    header += [f'cos_s{i}' for i in range(len(scale_dims))]
    header += [f'l2_s{i}' for i in range(len(scale_dims))]
    header += [f'd{s}_{j}' for s in range(len(scale_dims)) for j in range(scale_dims[s])]
    header += ['crop_path']
    log_writer.writerow(header)
    header_written = True

def capture_baseline_now(frame):
    """Capture whole-frame + per-person baselines from current frame."""
    global wf_baseline, person_baseline, prev_track_ids, track_switches, scale_dims
    global person_baseline_bbox
    name = MODELS[active][0]
    wf = extract_all_scales(name)
    if wf:
        wf_baseline = [s['norm'] for s in wf]
        if not scale_dims:
            scale_dims = [s['raw'].shape[0] for s in wf]
            print(f"[emb] Scale dims: {scale_dims} (total={sum(scale_dims)})")
        write_csv_header()
        print(f"[baseline] Whole-frame: {len(wf)} scales, dims={scale_dims}")
    img_h, img_w = frame.shape[:2]
    r_cap = models[active][1].track(frame, conf=conf, imgsz=IMGSZ, classes=CLASSES,
                       tracker=TRACKER, persist=True, device=DEVICE,
                       half=USE_HALF, verbose=False)[0]
    person_baseline = {}
    person_baseline_bbox = {}  # tid -> (cx, cy) for spatial matching fallback
    for b in r_cap.boxes:
        if int(b.cls) != 0:
            continue
        tid = int(b.id) if b.id is not None else -1
        if tid < 0:
            continue
        bbox = b.xyxy[0].cpu().numpy()
        scales = extract_all_scales(name, bbox, img_w, img_h)
        if scales:
            person_baseline[tid] = [s['norm'] for s in scales]
            person_baseline_bbox[tid] = ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)
    track_switches = []
    prev_track_ids = set()
    print(f"[baseline] {len(person_baseline)} persons: {sorted(person_baseline.keys())}")

def main():
    global active, mode, conf, fps, fast_mode, show_embeddings
    global wf_baseline, person_baseline, prev_track_ids, track_switches
    global last_whole_sample, last_person_sample, scale_dims
    global log_file, log_writer
    global baseline_countdown_end, baseline_label
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {CAM_INDEX}.")
    win = "YOLO track compare"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    prev, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            # Ensure UI coordinates match the actual display size.
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            if mode == "single":
                out = run_single(frame)
            else:
                out = run_all(frame)

            now = time.time(); dt = now - prev; prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            # --- Baseline countdown: auto-capture when timer expires ---
            if baseline_countdown_end > 0 and now >= baseline_countdown_end:
                print(f"[baseline] CAPTURING NOW — {baseline_label}")
                capture_baseline_now(frame)
                baseline_countdown_end = 0.0
                baseline_label = ""
            elif baseline_countdown_end > 0:
                remaining = baseline_countdown_end - now
                txt = f"BASELINE in {remaining:.1f}s — {baseline_label}"
                draw_text(out, txt, FRAME_W // 2 - 150, FRAME_H // 2 - 20, 1.2, (0, 200, 255), 2)
                # Pulsing border
                pulse = int(200 + 55 * abs(((now * 4) % 2) - 1))
                cv2.rectangle(out, (5, 5), (FRAME_W - 5, FRAME_H - 5), (0, pulse, 255), 4)

            cv2.imshow(win, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
            if k == ord('1'):
                active = 0; mode = "single"; print(f"[switch] active model = {MODELS[active][0]}")
            elif k == ord('2'):
                active = 1; mode = "single"; print(f"[switch] active model = {MODELS[active][0]}")
            elif k == ord('3'):
                active = 2; mode = "single"; print(f"[switch] active model = {MODELS[active][0]}")
            elif k == ord('4'):
                active = 3; mode = "single"; print(f"[switch] active model = {MODELS[active][0]}")
            elif k == ord('m'):
                mode = "all" if mode == "single" else "single"
            elif k == ord('c'):
                cycle_preset()
            elif k == ord('f'):
                fast_mode = not fast_mode
                print(f"[fast] mode = {fast_mode}")
            elif k == ord('e'):
                show_embeddings = not show_embeddings
                print(f"[emb] display {'ON' if show_embeddings else 'OFF'}")
            elif k == ord('t'):
                # Toggle capture/regular mode
                toggle_capture_mode()
            elif k == ord('b'):
                # Start 5s countdown then auto-capture baseline
                if emb_mode != "capture":
                    print("[baseline] Press 't' or CAPTURE button first to enter capture mode")
                else:
                    baseline_countdown_end = time.time() + BASELINE_DELAY
                    # Run a quick detect to count current persons for label
                    r_quick = models[active][1].predict(frame, conf=conf, imgsz=IMGSZ, classes=CLASSES,
                                       device=DEVICE, half=USE_HALF, verbose=False)[0]
                    n_persons = len([b for b in r_quick.boxes if int(b.cls) == 0])
                    if n_persons > 0:
                        baseline_label = f"{n_persons} persons in frame"
                    else:
                        baseline_label = "empty frame"
                    print(f"[baseline] Countdown {BASELINE_DELAY}s — {baseline_label}")
            elif k == ord('B'):
                # Shift+B = instant capture (no countdown)
                if emb_mode != "capture":
                    print("[baseline] Press 't' or CAPTURE button first to enter capture mode")
                else:
                    capture_baseline_now(frame)
            elif k == ord('+') or k == ord('='):
                conf = min(0.95, conf + 0.05)
            elif k == ord('-'):
                conf = max(0.05, conf - 0.05)
    finally:
        if log_file:
            log_file.close()
            print(f"[log] CSV saved to {log_csv_path}")
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
