"""
Quick diagnostic: find person bbox centers in withhuman.png
"""
import sys, math
sys.path.insert(0, r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3")
import types as _types
sys.modules["imgaug"] = _types.ModuleType("imgaug")
import numpy as np
import torch
from PIL import Image
from pytorchyolo.models import Darknet

CONFIG_PATH  = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg"
WEIGHTS_PATH = r"C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\weights\yolov3.weights"
IMG_WITH     = r"C:\Users\carso\Desktop\YODO\withhuman.png"
DEVICE = "cuda"

COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

def load_image(path, size=416):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(r, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32) / 255.0
    return arr

arr = load_image(IMG_WITH, 416)
model = Darknet(CONFIG_PATH).to(DEVICE)
model.load_darknet_weights(WEIGHTS_PATH)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(tensor)

out = output.cpu().numpy()
if out.ndim == 3: out = out[0]

persons = []
for row in out:
    if len(row) >= 6 and row[4] >= 0.1:
        cls = int(row[5])
        if COCO_NAMES[cls] == "person":
            cx = (row[0] + row[2]) / 2
            cy = (row[1] + row[3]) / 2
            w = row[2] - row[0]
            h = row[3] - row[1]
            persons.append({
                "cx": float(cx), "cy": float(cy),
                "x1": float(row[0]), "y1": float(row[1]),
                "x2": float(row[2]), "y2": float(row[3]),
                "w": float(w), "h": float(h),
                "conf": float(row[4]),
            })

print(f"Image size: {arr.shape}")
print(f"Total person detections: {len(persons)}")
print(f"\nPerson bbox centers (sorted by cy, top to bottom):")
persons.sort(key=lambda p: p["cy"])
for i, p in enumerate(persons):
    print(f"  P{i:2d}: cx={p['cx']:6.1f}, cy={p['cy']:6.1f}, "
          f"w={p['w']:5.1f}, h={p['h']:5.1f}, conf={p['conf']:.3f}, "
          f"bbox=({p['x1']:.0f},{p['y1']:.0f},{p['x2']:.0f},{p['y2']:.0f})")

# Find the person closest to center (208, 208)
print(f"\nClosest to image center (208, 208):")
for p in sorted(persons, key=lambda p: math.sqrt((p['cx']-208)**2 + (p['cy']-208)**2))[:3]:
    dist = math.sqrt((p['cx']-208)**2 + (p['cy']-208)**2)
    print(f"  cx={p['cx']:.1f}, cy={p['cy']:.1f}, dist={dist:.1f}, conf={p['conf']:.3f}, "
          f"size={p['w']:.0f}x{p['h']:.0f}")

# Find largest person (best target for patch)
print(f"\nLargest persons by area:")
for p in sorted(persons, key=lambda p: -p['w']*p['h'])[:5]:
    print(f"  cx={p['cx']:.1f}, cy={p['cy']:.1f}, w={p['w']:.0f}, h={p['h']:.0f}, "
          f"area={p['w']*p['h']:.0f}, conf={p['conf']:.3f}")
