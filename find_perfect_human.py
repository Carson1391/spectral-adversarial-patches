import torch, torch.nn.functional as F, math, os
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

m = YOLO('YOLOv3/yolov3u.pt').to('cuda')
model = m.model
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Hook the classification layer input
feat = {}
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight
        if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
            cls_module = module
            D = w.shape[1]
            print(f'Cls layer: {name}, D={D}')
            break

def hook(mod, inp, out):
    feat['x'] = inp[0]

cls_module.register_forward_hook(hook)

# 1. Find perfect human — average feature embedding at person detection locations across real images
imgs = sorted([f for f in os.listdir('data/coco_person/images') if f.endswith('.jpg')])[:50]
human_features = []

for f in imgs:
    img = Image.open(f'data/coco_person/images/{f}').convert('RGB').resize((640, 640))
    img_t = T.ToTensor()(img).unsqueeze(0).cuda()
    inp = img_t[:, [2, 1, 0], :, :] * 255.0
    
    with torch.no_grad():
        # First pass: detect person boxes
        det = m.predict(img, verbose=False, classes=[0])[0]
        if len(det.boxes) == 0:
            continue
        
        # Get the best person box
        best = det.boxes.conf.argmax()
        box = det.boxes.xyxy[best].cpu()
        
        # Second pass: capture features
        out = model(inp)
        x = feat['x']  # (1, D, H, W)
        B, D, H, W = x.shape
        cell = 640.0 / H
        # Map box center to grid location
        cx = ((box[0] + box[2]) / 2 / cell).int().item()
        cy = ((box[1] + box[3]) / 2 / cell).int().item()
        cx = max(0, min(W-1, cx))
        cy = max(0, min(H-1, cy))
        
        # Extract feature at person center — average a 3x3 region
        x_region = x[0, :, max(0,cy-1):cy+2, max(0,cx-1):cx+2]
        human_features.append(x_region.mean(dim=(1,2)))  # (D,)

human_x = torch.stack(human_features).mean(dim=0)  # (D,)
print(f'Human feature vectors collected: {len(human_features)}')
print(f'Human x norm: {human_x.norm().item():.4f}')
torch.save(human_x, 'outputs_clothing/v16_harmonic/perfect_human_v3.pt')

# 2. Load perfect meter
meter_x = torch.load('outputs_clothing/v16_harmonic/perfect_meter_v3.pt').cuda()
print(f'Meter x norm: {meter_x.norm().item():.4f}')

# 3. Classify dims based on ACTUAL feature values, not weights
# Human-only: where human_x is strong but meter_x is weak
# Meter-only: where meter_x is strong but human_x is weak
# Shared: where both are strong
human_abs = human_x.abs()
meter_abs = meter_x.abs()
h_thresh = human_abs.max() * 0.1
m_thresh = meter_abs.max() * 0.1

person_only = (human_abs > h_thresh) & (meter_abs < m_thresh)
meter_only = (meter_abs > m_thresh) & (human_abs < h_thresh)
shared = (human_abs > h_thresh) & (meter_abs > m_thresh)
other = (~person_only) & (~meter_only) & (~shared)

p_idx = person_only.nonzero(as_tuple=True)[0].tolist()
m_idx = meter_only.nonzero(as_tuple=True)[0].tolist()
s_idx = shared.nonzero(as_tuple=True)[0].tolist()
o_count = other.sum().item()

print(f'\nDim classification (from real feature embeddings):')
print(f'  Human-only: {len(p_idx)} dims: {p_idx}')
print(f'  Meter-only: {len(m_idx)} dims: {m_idx}')
print(f'  Shared: {len(s_idx)} dims: {s_idx}')
print(f'  Other: {o_count} dims')

# 4. Cosine similarity between real human and real meter embeddings
cos = F.cosine_similarity(human_x.unsqueeze(0), meter_x.unsqueeze(0)).item()
print(f'\nCosine sim (human_x, meter_x): {cos:.4f}')
print(f'L2 distance: {torch.norm(human_x - meter_x).item():.4f}')

del m
torch.cuda.empty_cache()
