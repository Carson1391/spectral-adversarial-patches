# cloud_payload.py
"""Surrogate cloud-ingest emulation. Strict separation from observation/detection tiers.

Accepts a predicted box from detect_result.py, crops the INPUT IMAGE at those
coordinates, runs the crop through a frozen ReID-grade encoder, L2-normalizes
the output, and returns the payload vector that would travel upstream.

Never touches YOLO internals. Never imposes its own pooling. The encoder owns
its own preprocessing and its own head — we just feed it what the detector
said to feed it."""

import torch
import torch.nn.functional as F
from torchvision import transforms

class SurrogateEncoder:
    """Wraps a frozen encoder with its native preprocess pipeline."""
    def __init__(self, name, model, input_size, mean, std):
        self.name = name
        self.model = model.eval().requires_grad_(False)
        self.input_size = input_size  # (H, W)
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size, antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ])

    @torch.inference_mode()
    def encode(self, crop_chw_tensor):
        """crop_chw_tensor: [3,H,W] in [0,1]. Returns L2-normalized [D]."""
        x = self.preprocess(crop_chw_tensor).unsqueeze(0)
        vec = self.model(x).squeeze(0)
        return F.normalize(vec, p=2, dim=0)


def crop_from_box(image_chw, box_xyxy):
    """image_chw: [3,H,W] in [0,1]. box_xyxy: tensor[4] in pixel coords."""
    x1, y1, x2, y2 = [int(v.clamp(min=0)) for v in box_xyxy.tolist()]
    x2 = min(x2, image_chw.shape[2]); y2 = min(y2, image_chw.shape[1])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate crop: ({x1},{y1},{x2},{y2})")
    return image_chw[:, y1:y2, x1:x2]


@torch.inference_mode()
def build_payload(encoders, image_chw, predicted_boxes_xyxy, confidences, conf_thresh=0.25):
    """Returns list of payload dicts, one per surviving detection.

    Empty list = no detections survived = no cloud upload (suppression succeeded).
    """
    payloads = []
    for box, conf in zip(predicted_boxes_xyxy, confidences):
        if conf < conf_thresh:
            continue
        try:
            crop = crop_from_box(image_chw, box)
        except ValueError:
            continue
        entry = {
            "box_xyxy": box.tolist(),
            "confidence": float(conf),
            "encodings": {},
        }
        for enc in encoders:
            vec = enc.encode(crop)
            entry["encodings"][enc.name] = {
                "vector": vec.cpu(),
                "norm_before_l2": float(vec.norm()),
            }
        payloads.append(entry)
    return payloads


def compare_payloads(payload_a, payload_b, encoder_name):
    """Angular + Euclidean separation between two payload entries on a chosen encoder."""
    va = payload_a["encodings"][encoder_name]["vector"]
    vb = payload_b["encodings"][encoder_name]["vector"]
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0))[0].item()
    ang = torch.arccos(torch.clamp(cos, -1.0, 1.0)).item()
    l2 = (va - vb).norm().item()
    return {
        "encoder": encoder_name,
        "cosine": cos,
        "angular_separation_rad": ang,
        "euclidean_post_l2": l2,
    }


def match_by_iou(payloads_clean, payloads_adv, iou_thresh=0.4):
    """Pair clean and adv payloads by box IoU so we compare the SAME logical detection."""
    pairs = []
    used_adv = set()
    for pc in payloads_clean:
        bc = torch.tensor(pc["box_xyxy"])
        best_iou, best_j = 0.0, -1
        for j, pa in enumerate(payloads_adv):
            if j in used_adv: continue
            ba = torch.tensor(pa["box_xyxy"])
            iou = _iou(bc, ba)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            pairs.append((pc, payloads_adv[best_j]))
            used_adv.add(best_j)
    return pairs


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-8)