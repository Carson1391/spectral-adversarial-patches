import torch
from ultralytics import YOLO

model = YOLO(r"C:\Users\carso\Desktop\YODO\YOLOv3\yolov3u.pt").model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

patch = torch.rand(3, 224, 224, device="cuda", requires_grad=True)
opt = torch.optim.Adam([patch], lr=0.1)

for step in range(100):
    canvas = torch.full((3, 640, 640), 0.5, device="cuda")
    canvas[:, 208:432, 208:432] = patch
    out = model(canvas.unsqueeze(0))[0]
    obj = out[:, 4:5, :].sigmoid()
    cls = out[:, 5:6, :].sigmoid()
    score = (obj * cls).sum()
    loss = -score
    opt.zero_grad()
    loss.backward()
    opt.step()
    patch.data.clamp_(0, 1)
    if step % 10 == 0:
        print(f"step {step:3d}  person_score={score.item():.2f}")

        from PIL import Image
import numpy as np
arr = (patch.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
Image.fromarray(arr).save("proof.png")
print("saved proof.png")