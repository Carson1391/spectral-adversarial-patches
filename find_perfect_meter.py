import torch, torch.nn.functional as F, math
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

m = YOLO('YOLOv3/yolov3u.pt').to('cuda')
model = m.model
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Find the classification layer and hook its input
feat = {}
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight
        if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
            cls_name = name
            cls_module = module
            D = w.shape[1]
            print(f'Cls layer: {name}, D={D}')
            break

def hook(mod, inp, out):
    feat['x'] = inp[0]

cls_module.register_forward_hook(hook)

# Synthesize the model's intrinsic perfect parking meter
# Start from random noise, maximize class 12 output at center anchor
img = torch.nn.Parameter(torch.rand(1, 3, 640, 640, device='cuda') * 0.3 + 0.35)
opt = torch.optim.Adam([img], lr=0.05)

for i in range(300):
    opt.zero_grad()
    inp = img[:, [2,1,0], :, :] * 255.0
    out = model(inp)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    
    # Get class 12 (parking meter) scores at all anchors
    cls_scores = pred[0, 4:, :]  # (80, anchors)
    meter_scores = cls_scores[12, :]  # (anchors,)
    
    # Maximize the best meter score
    loss = -meter_scores.max()
    loss.backward()
    opt.step()
    with torch.no_grad():
        img.clamp_(0, 1)
    
    if i % 50 == 0:
        print(f'iter {i}: meter_score={-loss.item():.4f}')

# Now capture the feature embedding of this perfect meter image
with torch.no_grad():
    inp = img[:, [2,1,0], :, :] * 255.0
    out = model(inp)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    cls_scores = pred[0, 4:, :]
    # Find the anchor with highest meter score
    best_anchor = cls_scores[12, :].argmax().item()
    print(f'\nBest meter anchor: {best_anchor}')
    print(f'Meter score: {cls_scores[12, best_anchor].item():.4f}')
    
    # Get feature embedding at that anchor
    x = feat['x']  # (1, D, H, W)
    B, D, H, W = x.shape
    grid_size = H  # spatial size
    # Find which spatial location the best anchor corresponds to
    anchor_y = best_anchor // W
    anchor_x = best_anchor % W
    print(f'Spatial location: ({anchor_x}, {anchor_y})')
    
    # Extract the feature vector at that location - this is the model's perfect meter
    perfect_meter_x = x[0, :, anchor_y, anchor_x].clone()  # (D,)
    print(f'Perfect meter feature vector shape: {perfect_meter_x.shape}')
    print(f'Perfect meter feature norm: {perfect_meter_x.norm().item():.4f}')
    
    # Save it
    torch.save(perfect_meter_x, 'outputs_clothing/v16_harmonic/perfect_meter_v3.pt')
    
    # Also save the synthesized image
    T.ToPILImage()(img.squeeze(0).clamp(0,1)).save('outputs_clothing/v16_harmonic/perfect_meter_image_v3.png')
    print('Saved perfect_meter_v3.pt and perfect_meter_image_v3.png')

del m
torch.cuda.empty_cache()
