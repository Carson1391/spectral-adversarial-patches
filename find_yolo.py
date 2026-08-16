import sys
sys.path.insert(0, r'C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3')
from pytorchyolo.utils.parse_config import parse_model_config
defs = parse_model_config(r'C:\Users\carso\Desktop\YODO\PyTorch-YOLOv3\config\yolov3.cfg')
yolo_indices = [i for i, d in enumerate(defs) if d['type'] == 'yolo']
print('YOLO module indices:', yolo_indices)
for i in yolo_indices:
    m = defs[i].get('mask', '')
    print(f'  module {i}: mask={m}')
