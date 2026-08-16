import numpy as np
from PIL import Image
import sys

sys.path.insert(0, r'C:\Users\carso\Desktop\YODO\adversarial-robustness-toolbox\tests\estimators\object_detection')
from run_external_patch_benchmark import load_rgb, load_patch, apply_external_patch, build_detector, load_rgb, torso_center, detection_summary, apply_nms
import argparse

# Load patch
patch = load_patch(r'C:\Users\carso\Desktop\YODO\outputs_clothing\final_boss\poison_416.png', 416)
print(f"Patch shape: {patch.shape}, dtype: {patch.dtype}")
print(f"Patch min: {patch.min()}, max: {patch.max()}, mean per channel: {patch.mean(axis=(1,2))}")

# Save patch directly
Image.fromarray(np.clip(patch.transpose(1,2,0), 0, 255).astype(np.uint8)).save(r'C:\Users\carso\Desktop\YODO\debug_patch.png')

# Now check what apply_external_patch produces
args = argparse.Namespace(
    reference=[r'C:\Users\carso\Downloads\ChatGPT Image Jul 1, 2026, 04_33_21 AM (1).png'],
    patch=[r'C:\Users\carso\Desktop\YODO\outputs_clothing\final_boss\poison_416.png'],
    weights=r'C:\Users\carso\Desktop\YODO\yolov3.weights',
    config=r'C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg',
    yolo_root=r'C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3',
    no_human=[],
    output='debug',
    input_size=416,
    threshold=0.25,
    device='cpu',
)
detector, darknet = build_detector(args)
image = load_rgb(r'C:\Users\carso\Downloads\ChatGPT Image Jul 1, 2026, 04_33_21 AM (1).png', 416)
pred = detector.predict(image[None, ...])[0]
nms_pred = apply_nms(pred, 0.4)
summary = detection_summary(nms_pred, 0.25)
box = summary["best_person_box"]
if box:
    x1, y1, x2, y2 = box
    cx = int((x1+x2)/2)
    cy = int(y1 + 0.50*(y2-y1))
    tw = max(16, int(0.65*(x2-x1)))
else:
    cx, cy, tw = 208, 208, 104

patched = apply_external_patch(detector, image, patch, cx, cy, tw, scale=0.32, rotation_max=0.0)
print(f"Patched shape: {patched.shape}, min: {patched.min()}, max: {patched.max()}")
print(f"Patched mean per channel: {patched.mean(axis=(1,2))}")

# Save patched
Image.fromarray(np.clip(patched.transpose(1,2,0), 0, 255).astype(np.uint8)).save(r'C:\Users\carso\Desktop\YODO\debug_patched.png')

# Check the patch region specifically
print(f"\nPatch region (around center {cx},{cy}):")
region = patched[:, cy-30:cy+30, cx-30:cx+30]
print(f"  Region mean per channel: {region.mean(axis=(1,2))}")
print(f"  Region min: {region.min()}, max: {region.max()}")
