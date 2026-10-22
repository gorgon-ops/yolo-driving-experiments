"""Experiment A: source resolution vs YOLO detection quality.

Every condition comes from the same 1080p frame. Pseudo-ground-truth is yolo11x at
imgsz=1280 on that untouched frame.
"""
import argparse, pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER, round32

MASTER = "1080p_video_driving_sample.mp4"
REF_MODEL, REF_IMGSZ, REF_CONF = "yolo11x.pt", 1280, 0.35
TEST_MODEL, TEST_CONF = "yolo11m.pt", 0.25

# (label, width, height) -- simulated capture resolutions
RESOLUTIONS = [
    ("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480),
    ("360p", 640, 360), ("240p", 426, 240),
]


def extract(result, keep_conf):
    b = result.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4), np.float32), [], np.zeros(0, np.float32)
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    m = np.array([c in DRIVING_CLASSES for c in cls]) & (conf >= keep_conf)
    return xyxy[m], [SUPER[c] for c in cls[m]], conf[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default="results_resolution.pkl")
    args = ap.parse_args()

    cap = cv2.VideoCapture(MASTER)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // args.frames)
    want = set(range(0, total, step))
    print(f"master {total} frames; sampling every {step} -> {len(want)} frames", flush=True)

    ref_model = YOLO(REF_MODEL)
    test_model = YOLO(TEST_MODEL)

    # pipeline -> how to pick the detector input size for a given source width
    pipelines = {
        "fixed640": lambda w: 640,
        "native": lambda w: round32(w, cap=1280),
    }

    records = []   # one dict per (frame, pipeline, resolution)
    timings = {}   # (pipeline, res) -> list of ms
    idx, done = 0, 0
    t_start = time.time()

    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx not in want:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok:
            continue

        # --- reference on the pristine 1080p frame ---
        r = ref_model.predict(frame, imgsz=REF_IMGSZ, conf=REF_CONF, verbose=False)[0]
        ref_boxes, ref_super, ref_conf = extract(r, REF_CONF)

        for pname, pick in pipelines.items():
            for rname, rw, rh in RESOLUTIONS:
                small = frame if rw == 1920 else cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
                imgsz = pick(rw)
                t0 = time.perf_counter()
                d = test_model.predict(small, imgsz=imgsz, conf=TEST_CONF, verbose=False)[0]
                ms = (time.perf_counter() - t0) * 1000
                db, ds, dc = extract(d, TEST_CONF)
                db = db * (1920.0 / rw)          # back to master coordinates
                timings.setdefault((pname, rname), []).append(ms)
                records.append(dict(frame=idx - 1, pipeline=pname, res=rname, imgsz=imgsz,
                                    det_boxes=db, det_super=ds, det_conf=dc,
                                    ref_boxes=ref_boxes, ref_super=ref_super, ref_conf=ref_conf))
        done += 1
        if done % 10 == 0:
            el = time.time() - t_start
            print(f"  {done}/{len(want)} frames  {el:.0f}s elapsed  "
                  f"eta {el / done * (len(want) - done):.0f}s", flush=True)

    cap.release()
    with open(args.out, "wb") as f:
        pickle.dump(dict(records=records, timings=timings,
                         ref_model=REF_MODEL, test_model=TEST_MODEL,
                         resolutions=RESOLUTIONS), f)
    print(f"wrote {args.out}: {len(records)} records in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
