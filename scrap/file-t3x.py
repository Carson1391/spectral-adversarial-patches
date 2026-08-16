import os, glob
from ultralytics import YOLO

folder = os.path.dirname(os.path.abspath(__file__))
model = YOLO(r"C:\Users\carso\Desktop\YODO\YOLOv3\yolov3u.pt")
names = model.names

files = glob.glob(os.path.join(folder, "*.png"))
if not files:
    print("No PNG files found in:", folder)
else:
    for f in files:
        fn = os.path.basename(f)
        r = model.predict(f, verbose=False)[0]
        if len(r.boxes)==0:
            print(f"{fn}: NOTHING DETECTED")
            continue
        tags=[]
        for b in r.boxes.data.tolist():
            cid=int(b[5]); conf=round(float(b[4]),2)
            tags.append(f"{names[cid]}:{conf}")
        print(f"{fn}: {' '.join(tags)}")