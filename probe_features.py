import torch
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image
import os

m = YOLO('YOLOv8/yolov8n.pt').to('cuda')
model = m.model

features = {}
def make_hook(name):
    def fn(module, input, output):
        features[name] = input[0].detach()
    return fn

for name, module in model.named_modules():
    if name == 'model.22.cv3.0.2':
        module.register_forward_hook(make_hook('cls_input'))

imgs = sorted([f for f in os.listdir('data/coco_person/images') if f.endswith('.jpg')])[:1]
img = Image.open(f'data/coco_person/images/{imgs[0]}').convert('RGB').resize((640,640))
img_t = T.ToTensor()(img).unsqueeze(0).to('cuda')
img_input = img_t[:, [2,1,0], :, :] * 255.0

with torch.no_grad():
    out = model(img_input)

if 'cls_input' in features:
    x = features['cls_input']
    print(f'Feature embedding x shape: {x.shape}')
    D = x.shape[1]

    cls_conv = dict(model.named_modules())['model.22.cv3.0.2']
    w = cls_conv.weight.squeeze(-1).squeeze(-1)
    w_person = w[0]
    w_meter = w[12]

    x_avg = x[0].mean(-1).mean(-1)

    person_contrib = (w_person * x_avg).abs()
    meter_contrib = (w_meter * x_avg).abs()

    top_p = person_contrib.argsort(descending=True)[:10]
    top_m = meter_contrib.argsort(descending=True)[:10]
    print(f'D={D}, sqrt(D)={D**0.5:.1f}')
    print(f'Top 10 person dims: {top_p.tolist()}')
    print(f'Top 10 meter dims: {top_m.tolist()}')
    overlap = set(top_p.tolist()) & set(top_m.tolist())
    print(f'Overlap: {overlap if overlap else "NONE - fully separable"}')

    p_imp = (person_contrib > person_contrib.max() * 0.1).sum().item()
    m_imp = (meter_contrib > meter_contrib.max() * 0.1).sum().item()
    print(f'Important dims (>10pct max) for person: {p_imp}/{D}')
    print(f'Important dims (>10pct max) for meter: {m_imp}/{D}')

    person_score = (w_person * x_avg).sum().item()
    meter_score = (w_meter * x_avg).sum().item()
    print(f'Person score: {person_score:.4f}')
    print(f'Meter score: {meter_score:.4f}')

del m
torch.cuda.empty_cache()
