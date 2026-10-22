"""Experiment A, second pass: compute spent on frames the pre-screen says hold traffic.

The first pass scored only 27 objects (94 % of the footage is empty road). Frames
cluster into 22 traffic events, so boxes are not independent samples.
"""
import argparse, pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER, round32

MASTER = "1080p_video_driving_sample.mp4"
REF_MODEL, REF_IMGSZ, REF_CONF = "yolo11x.pt", 1280, 0.30
TEST_MODEL, TEST_CONF = "yolo11m.pt", 0.25
RESOLUTIONS = [("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480),
               ("360p", 640, 360), ("240p", 426, 240)]
EVENT_GAP = 90     # source frames; a larger gap starts a new traffic event


def event_frames(step):
    rows = pickle.load(open("prescreen.pkl", "rb"))
    hit = rows[rows[:, 1] >= 1][:, 0]
    events, cur = [], [hit[0]]
    for a, b in zip(hit, hit[1:]):
        if b - a <= EVENT_GAP:
            cur.append(b)
        else:
            events.append(cur)
            cur = [b]
    events.append(cur)
    out = {}
    for eid, e in enumerate(events):
        for f in range(int(e[0]), int(e[-1]) + 1, step):
            out[f] = eid
    return out, len(events)


def extract(result, keep_conf):
    b = result.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4), np.float32), [], np.zeros(0, np.float32)
    xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
    m = np.array([c in DRIVING_CLASSES for c in cls]) & (conf >= keep_conf)
    return xyxy[m], [SUPER[c] for c in cls[m]], conf[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=6, help="frame stride inside each event")
    ap.add_argument("--out", default="results_resolution2.pkl")
    args = ap.parse_args()

    want, n_events = event_frames(args.step)
    print(f"{len(want)} frames across {n_events} traffic events", flush=True)

    ref_model, test_model = YOLO(REF_MODEL), YOLO(TEST_MODEL)
    pipelines = {"fixed640": lambda w: 640, "native": lambda w: round32(w, cap=1280)}

    cap = cv2.VideoCapture(MASTER)
    records, timings, idx, done = [], {}, 0, 0
    t_start = time.time()
    while True:
        if not cap.grab():
            break
        if idx not in want:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok:
            continue
        eid = want[idx - 1]
        r = ref_model.predict(frame, imgsz=REF_IMGSZ, conf=REF_CONF, verbose=False)[0]
        rb, rs, rc = extract(r, REF_CONF)

        for pname, pick in pipelines.items():
            for rname, rw, rh in RESOLUTIONS:
                small = frame if rw == 1920 else cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
                imgsz = pick(rw)
                t0 = time.perf_counter()
                d = test_model.predict(small, imgsz=imgsz, conf=TEST_CONF, verbose=False)[0]
                ms = (time.perf_counter() - t0) * 1000
                db, ds, dc = extract(d, TEST_CONF)
                db = db * (1920.0 / rw)
                timings.setdefault((pname, rname), []).append(ms)
                records.append(dict(frame=idx - 1, event=eid, pipeline=pname, res=rname,
                                    imgsz=imgsz, det_boxes=db, det_super=ds, det_conf=dc,
                                    ref_boxes=rb, ref_super=rs, ref_conf=rc))
        done += 1
        if done % 20 == 0:
            el = time.time() - t_start
            print(f"  {done}/{len(want)}  {el:.0f}s  eta {el/done*(len(want)-done):.0f}s", flush=True)
    cap.release()
    with open(args.out, "wb") as f:
        pickle.dump(dict(records=records, timings=timings, n_events=n_events,
                         ref_model=REF_MODEL, test_model=TEST_MODEL,
                         resolutions=RESOLUTIONS), f)
    print(f"wrote {args.out}: {len(records)} records, {done} frames, {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
