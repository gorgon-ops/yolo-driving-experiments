"""Control for B: raw detection with no tracker.

Tracked det/frame fell from 0.64 at 60 fps to 0.33 at 2 fps. Detecting every frame
once and averaging per subsample shows whether that loss is the detector or the tracker.
"""
import pickle
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL, CONF, IMGSZ = "yolo11s.pt", 0.25, 1280
WORK_W, WORK_H = 1280, 720
START_SEC, DUR_SEC = 115.7, 18.0
FPS_CONDITIONS = [60, 30, 20, 15, 10, 5, 2]
BAND = 400.0 * WORK_H / 1080.0

cap = cv2.VideoCapture(MASTER)
src_fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_SEC * src_fps))
frames = []
for _ in range(int(DUR_SEC * src_fps)):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA))
cap.release()
print(f"{len(frames)} frames", flush=True)

model = YOLO(MODEL)
counts = []
for i, f in enumerate(frames):
    r = model.predict(f, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    b = r.boxes
    n = 0
    if b is not None and len(b):
        cls = b.cls.cpu().numpy().astype(int)
        for box, c in zip(b.xyxy.cpu().numpy(), cls):
            if c in DRIVING_CLASSES and SUPER[c] != "person" and (box[1] + box[3]) / 2 < BAND:
                n += 1
    counts.append(n)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(frames)}", flush=True)

counts = np.array(counts)
print(f"\nraw detections, no tracker: {counts.sum()} total, {counts.mean():.3f}/frame\n")
rows = []
for fps in FPS_CONDITIONS:
    stride = int(round(src_fps / fps))
    sub = counts[::stride]
    rows.append((fps, len(sub), float(sub.mean())))
    print(f"  {fps:3d} fps: {len(sub):5d} frames, {sub.mean():.3f} det/frame (raw)")
pickle.dump(dict(counts=counts, rows=rows), open("results_fps_control.pkl", "wb"))
print("\nwrote results_fps_control.pkl")
