"""Experiment B, second pass: the densest 18 s window, frames kept at 1280x720, imgsz=1280.

The first pass used a near-empty window and 960x540 at imgsz=640, which hid distant
vehicles and reproduced the cabin-as-car artifact.
"""
import argparse, pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL, CONF, IMGSZ = "yolo11s.pt", 0.25, 1280
WORK_W, WORK_H = 1280, 720
FPS_CONDITIONS = [60, 30, 20, 15, 10, 5, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, default=115.7)
    ap.add_argument("--dur-sec", type=float, default=18.0)
    ap.add_argument("--out", default="results_fps2.pkl")
    args = ap.parse_args()

    cap = cv2.VideoCapture(MASTER)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * src_fps))
    n = int(args.dur_sec * src_fps)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA))
    cap.release()
    print(f"held {len(frames)} frames at {WORK_W}x{WORK_H}", flush=True)

    model = YOLO(MODEL)
    out = {}
    for fps in FPS_CONDITIONS:
        stride = int(round(src_fps / fps))
        idxs = list(range(0, len(frames), stride))
        eff = src_fps / stride
        model.predictor = None
        per_frame, ms = [], []
        t0 = time.time()
        for k, i in enumerate(idxs):
            t1 = time.perf_counter()
            r = model.track(frames[i], imgsz=IMGSZ, conf=CONF, persist=(k > 0),
                            tracker="bytetrack.yaml", verbose=False)[0]
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
        out[fps] = dict(eff_fps=eff, stride=stride, frames=per_frame,
                        infer_ms=float(np.mean(ms)), wall=time.time() - t0,
                        n_frames=len(idxs))
        road = sum(sum(1 for i in range(len(f["ids"]))
                       if f["sup"][i] != "person"
                       and (f["boxes"][i][1] + f["boxes"][i][3]) / 2 < 400 * WORK_H / 1080)
                   for f in per_frame)
        print(f"  {fps:3d} fps: {len(idxs):5d} frames, {np.mean(ms):.0f} ms, "
              f"{road} road detections, {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(dict(conditions=out, src_fps=src_fps, model=MODEL, imgsz=IMGSZ,
                         work_h=WORK_H, dur_sec=args.dur_sec, start_sec=args.start_sec), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
