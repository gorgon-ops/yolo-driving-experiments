"""Find driving frames with traffic, so the expensive grid is not spent on empty road."""
import pickle
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
DRIVE_START = 3460      # one continuous driving shot from here to the end
ROAD_BAND_Y = 400       # box centres below this are cabin, not road
STEP = 6

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(MASTER)
rows, idx = [], 0
while True:
    if not cap.grab():
        break
    if idx < DRIVE_START or idx % STEP:
        idx += 1
        continue
    ok, f = cap.retrieve()
    idx += 1
    if not ok:
        continue
    r = model.predict(f, imgsz=960, conf=0.30, verbose=False)[0]
    b = r.boxes
    n = 0
    if b is not None and len(b):
        cls = b.cls.cpu().numpy().astype(int)
        xy = b.xyxy.cpu().numpy()
        for box, c in zip(xy, cls):
            if c not in DRIVING_CLASSES or SUPER[c] == "person":
                continue
            if (box[1] + box[3]) / 2 < ROAD_BAND_Y:
                n += 1
    rows.append((idx - 1, n))
    if len(rows) % 300 == 0:
        print(f"  scanned {len(rows)} frames", flush=True)
cap.release()

rows = np.array(rows)
print(f"scanned {len(rows)} driving frames")
print(f"frames with >=1 road object: {(rows[:,1]>=1).sum()}")
print(f"frames with >=2: {(rows[:,1]>=2).sum()}   >=3: {(rows[:,1]>=3).sum()}")
print(f"total road objects seen: {rows[:,1].sum()}")
pickle.dump(rows, open("prescreen.pkl", "wb"))
