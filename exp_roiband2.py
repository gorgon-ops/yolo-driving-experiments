"""Road-band crop with rectangular inference at (320, 1920) vs full frame at 1280.

Full frame scales by 0.667 (a 24 px object arrives as 16 px); the crop keeps native
size and processes 0.61 MP instead of 1.64 MP.
"""
import pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER, greedy_match, size_bin, SIZE_BINS

MASTER = "1080p_video_driving_sample.mp4"
BAND = (90, 410)          # master-y range of the road band (the camera drifts)
ROAD_BAND_Y = 400
CONF = 0.25

CONFIGS = [
    ("full frame  imgsz=1280 (baseline)", None, 1280),
    ("band 90-410  (320 tall) native", (90, 410), (320, 1920)),
    ("band 90-570  (480 tall) native", (90, 570), (480, 1920)),
    ("band 90-730  (640 tall) native", (90, 730), (640, 1920)),
]


def extract(res, off_y, keep_person=False):
    b = res.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4), np.float32), [], np.zeros(0, np.float32)
    xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
    m = np.array([c in DRIVING_CLASSES for c in cls]) & (conf >= CONF)
    xyxy, cls, conf = xyxy[m], cls[m], conf[m]
    if off_y:
        xyxy = xyxy + np.array([0, off_y, 0, off_y], np.float32)
    sup = [SUPER[c] for c in cls]
    k = [i for i, s in enumerate(sup)
         if (keep_person or s != "person") and (xyxy[i][1] + xyxy[i][3]) / 2 < ROAD_BAND_Y]
    return xyxy[k], [sup[i] for i in k], conf[k]


d = pickle.load(open("results_resolution2.pkl", "rb"))
ref_by_frame = {}
for r in d["records"]:
    if r["pipeline"] == "native" and r["res"] == "1080p":
        rb, rs = r["ref_boxes"], r["ref_super"]
        k = [i for i in range(len(rb))
             if rs[i] != "person" and (rb[i][1] + rb[i][3]) / 2 < ROAD_BAND_Y]
        ref_by_frame[r["frame"]] = (rb[k], [rs[i] for i in k])
want = sorted(ref_by_frame)
print(f"{len(want)} frames, {sum(len(v[0]) for v in ref_by_frame.values())} reference objects\n")

model = YOLO("yolo11m.pt")
results = {name: dict(tp=0, nref=0, ndet=0, ms=[], bins=np.zeros((len(SIZE_BINS), 2)))
           for name, _, _ in CONFIGS}

cap = cv2.VideoCapture(MASTER)
idx = 0
while True:
    if not cap.grab():
        break
    if idx not in ref_by_frame:
        idx += 1
        continue
    ok, frame = cap.retrieve()
    idx += 1
    if not ok:
        continue
    rb, rs = ref_by_frame[idx - 1]
    for name, band, imgsz in CONFIGS:
        img = frame if band is None else frame[band[0]:band[1]]
        off = 0 if band is None else band[0]
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=CONF, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        db, ds, dc = extract(r, off)
        rm, dm, _ = greedy_match(rb, rs, db, ds, dc, 0.5)
        R = results[name]
        R["tp"] += int(rm.sum()); R["nref"] += len(rb); R["ndet"] += len(db); R["ms"].append(ms)
        for i in range(len(rb)):
            b = size_bin(rb[i][3] - rb[i][1])
            R["bins"][b, 0] += int(rm[i]); R["bins"][b, 1] += 1
cap.release()

print(f"{'config':38s} {'recall':>7s} {'prec':>7s} {'F1':>7s} {'ms':>7s} {'fps':>6s}   tiny  small    med  large")
for name, _, _ in CONFIGS:
    R = results[name]
    rec = R["tp"] / max(R["nref"], 1)
    pre = R["tp"] / max(R["ndet"], 1)
    f1 = 2 * rec * pre / max(rec + pre, 1e-9)
    ms = float(np.mean(R["ms"]))
    bins = " ".join(f"{(R['bins'][b,0]/R['bins'][b,1] if R['bins'][b,1] else 0):6.3f}"
                    for b in range(len(SIZE_BINS)))
    print(f"{name:38s} {rec:7.3f} {pre:7.3f} {f1:7.3f} {ms:7.0f} {1000/ms:6.2f}  {bins}")
