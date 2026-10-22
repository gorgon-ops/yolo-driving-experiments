"""Experiment B: frame rate vs tracking continuity, reaction latency and keeping up.

Every condition subsamples the same 1080p window, holding resolution and content fixed.
"""
import argparse, pickle, time
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL, CONF, IMGSZ = "yolo11s.pt", 0.30, 640
FPS_CONDITIONS = [60, 30, 20, 15, 10, 5, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, default=120.0)
    ap.add_argument("--dur-sec", type=float, default=40.0)
    ap.add_argument("--out", default="results_fps.pkl")
    args = ap.parse_args()

    cap = cv2.VideoCapture(MASTER)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    start = int(args.start_sec * src_fps)
    n = int(args.dur_sec * src_fps)

    print(f"decoding {n} frames from {args.start_sec}s (src {src_fps:.0f} fps)...", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        # Half size keeps 40 s in RAM; identical for every condition.
        frames.append(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA))
    cap.release()
    print(f"held {len(frames)} frames in memory", flush=True)

    model = YOLO(MODEL)
    out = {}

    for fps in FPS_CONDITIONS:
        stride = int(round(src_fps / fps))
        idxs = list(range(0, len(frames), stride))
        eff_fps = src_fps / stride
        model.predictor = None  # fresh tracker state per condition
        per_frame, infer_ms = [], []
        t0 = time.time()
        for k, i in enumerate(idxs):
            t1 = time.perf_counter()
            r = model.track(frames[i], imgsz=IMGSZ, conf=CONF, persist=(k > 0),
                            tracker="bytetrack.yaml", verbose=False)[0]
            infer_ms.append((time.perf_counter() - t1) * 1000)
            b = r.boxes
            if b is None or len(b) == 0 or b.id is None:
                per_frame.append(dict(t=i / src_fps, ids=[], sup=[], boxes=np.zeros((0, 4), np.float32)))
                continue
            cls = b.cls.cpu().numpy().astype(int)
            m = np.array([c in DRIVING_CLASSES for c in cls])
            per_frame.append(dict(
                t=i / src_fps,
                ids=b.id.cpu().numpy().astype(int)[m].tolist(),
                sup=[SUPER[c] for c in cls[m]],
                boxes=b.xyxy.cpu().numpy()[m],
            ))
        out[fps] = dict(eff_fps=eff_fps, stride=stride, frames=per_frame,
                        infer_ms=float(np.mean(infer_ms)), wall=time.time() - t0,
                        n_frames=len(idxs))
        print(f"  {fps:3d} fps (stride {stride}): {len(idxs):5d} frames, "
              f"{np.mean(infer_ms):.0f} ms/frame, {time.time()-t0:.0f}s wall", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(dict(conditions=out, src_fps=src_fps, model=MODEL,
                         dur_sec=args.dur_sec, start_sec=args.start_sec), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
