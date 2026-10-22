"""Experiment D: does a better tracker undo the frame-rate penalty?

ByteTrack's default gate needs IoU > 0.2, and at 300 km/h and 2 fps a vehicle does
not overlap itself. Tested: a looser gate, and BoT-SORT with global motion
compensation. Everything else matches exp_fps2.
"""
import argparse, pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL, CONF, IMGSZ = "yolo11s.pt", 0.25, 1280
WORK_W, WORK_H = 1280, 720
START_SEC, DUR_SEC = 115.7, 18.0
FPS_CONDITIONS = [60, 30, 20, 15, 10, 5, 2]
BAND = 400.0 * WORK_H / 1080.0

ARMS = [
    ("bytetrack",        "bytetrack.yaml"),
    ("bytetrack_tuned",  "trackers/bytetrack_tuned.yaml"),
    ("botsort_gmc",      "trackers/botsort_gmc.yaml"),
    ("botsort_gmc_tuned", "trackers/botsort_gmc_tuned.yaml"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_trackers.pkl")
    args = ap.parse_args()

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
    print(f"held {len(frames)} frames at {WORK_W}x{WORK_H}", flush=True)

    model = YOLO(MODEL)
    out = {}
    for arm, cfg in ARMS:
        out[arm] = {}
        print(f"\n=== {arm} ({cfg}) ===", flush=True)
        for fps in FPS_CONDITIONS:
            stride = int(round(src_fps / fps))
            idxs = list(range(0, len(frames), stride))
            model.predictor = None          # fresh tracker state per condition
            per_frame, ms = [], []
            t0 = time.time()
            for k, i in enumerate(idxs):
                t1 = time.perf_counter()
                r = model.track(frames[i], imgsz=IMGSZ, conf=CONF, persist=(k > 0),
                                tracker=cfg, verbose=False)[0]
                ms.append((time.perf_counter() - t1) * 1000)
                b = r.boxes
                if b is None or len(b) == 0 or b.id is None:
                    per_frame.append(dict(t=i / src_fps, ids=[], sup=[],
                                          boxes=np.zeros((0, 4), np.float32)))
                    continue
                cls = b.cls.cpu().numpy().astype(int)
                m = np.array([c in DRIVING_CLASSES for c in cls])
                per_frame.append(dict(t=i / src_fps,
                                      ids=b.id.cpu().numpy().astype(int)[m].tolist(),
                                      sup=[SUPER[c] for c in cls[m]],
                                      boxes=b.xyxy.cpu().numpy()[m]))
            road = sum(sum(1 for j in range(len(f["ids"]))
                           if f["sup"][j] != "person"
                           and (f["boxes"][j][1] + f["boxes"][j][3]) / 2 < BAND)
                       for f in per_frame)
            out[arm][fps] = dict(eff_fps=src_fps / stride, stride=stride,
                                 frames=per_frame, infer_ms=float(np.mean(ms)),
                                 n_frames=len(idxs), road_dets=road)
            print(f"  {fps:3d} fps: {len(idxs):5d} frames, {road:4d} road dets, "
                  f"{road/len(idxs):.3f}/frame, {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(dict(arms=out, src_fps=src_fps, model=MODEL, imgsz=IMGSZ,
                         work_h=WORK_H, dur_sec=DUR_SEC, start_sec=START_SEC), f)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
