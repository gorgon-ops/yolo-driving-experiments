"""Demo video: objects under 24 px are never detected at imgsz=640 and appear at 1280.

Both panels see the same 1080p frame. Any single frame can go either way (usually a
false positive on the logo), so the panels carry running totals and a distant-box
counter, where the effect is unambiguous.
"""
import argparse, os
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL = "yolo11m.pt"
CONF = 0.25
TRACKER = "trackers/bytetrack_tuned.yaml"
CROP = (520, 78, 1820, 318)      # x0,y0,x1,y1 in master coords: the carriageway
ROAD_BAND_Y = 400                # same scoring band as the experiments
TINY_PX = 24                     # box height below this is the bin that matters
SEGMENTS = [(104, 122), (173, 187), (213, 222), (265, 272)]
STRIDE = 2                       # 60 fps source -> 30 fps output
PANELS = [("imgsz=1280", 1280, (70, 230, 70)), ("imgsz=640", 640, (70, 130, 255))]

W = 1920
S = W / (CROP[2] - CROP[0])
PANEL_H = int(round((CROP[3] - CROP[1]) * S))
HDR, FOOT = 50, 44
OUT_H = HDR + (HDR + PANEL_H) * len(PANELS) + FOOT

COL = {"vehicle": (80, 240, 80), "twowheeler": (255, 80, 255),
       "traffic light": (80, 240, 255), "stop sign": (80, 160, 255)}


def draw(frame, res):
    x0, y0, x1, y1 = CROP
    canvas = cv2.resize(frame[y0:y1, x0:x1], (W, PANEL_H), interpolation=cv2.INTER_LINEAR)
    n = tiny = 0
    b = res.boxes
    if b is not None and len(b):
        cls = b.cls.cpu().numpy().astype(int)
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(cls), -1)
        for box, c, cf, tid in zip(xyxy, cls, conf, ids):
            if c not in DRIVING_CLASSES:
                continue
            sup = SUPER[c]
            if sup == "person" or (box[1] + box[3]) / 2 >= ROAD_BAND_Y:
                continue                      # driver and cabin artifacts are not the subject
            h = box[3] - box[1]
            is_tiny = h < TINY_PX
            n += 1
            tiny += int(is_tiny)
            p1 = (int((box[0] - x0) * S), int((box[1] - y0) * S))
            p2 = (int((box[2] - x0) * S), int((box[3] - y0) * S))
            col = (60, 240, 255) if is_tiny else COL[sup]
            cv2.rectangle(canvas, p1, p2, col, 3 if is_tiny else 2)
            tag = f"{DRIVING_CLASSES[c]} {cf:.2f}" + (f" #{tid}" if tid > 0 else "")
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 3, p1[1] - 5)
            cv2.rectangle(canvas, (p1[0], ty - th - 4), (p1[0] + tw + 5, ty + 3), col, -1)
            cv2.putText(canvas, tag, (p1[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas, n, tiny


def header(label, color, n, tiny, tot, tot_tiny):
    bar = np.full((HDR, W, 3), 26, np.uint8)
    cv2.rectangle(bar, (0, 0), (12, HDR), color, -1)
    cv2.putText(bar, label, (28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(bar, f"this frame: {n}", (300, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(bar, f"DISTANT <{TINY_PX}px: {tiny}", (520, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (60, 240, 255) if tiny else (110, 110, 110), 2 if tiny else 1, cv2.LINE_AA)
    cv2.putText(bar, f"running total: {tot}", (900, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(bar, f"running distant: {tot_tiny}", (1220, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (60, 240, 255) if tot_tiny else (110, 110, 110), 2, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_detections.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(MASTER)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    models = [YOLO(MODEL) for _ in PANELS]
    for m in models:
        m.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=640, device="cuda", verbose=False)

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"),
                         src_fps / STRIDE, (W, OUT_H))
    assert vw.isOpened(), "VideoWriter failed to open"

    tot = [0] * len(PANELS)
    tot_tiny = [0] * len(PANELS)
    written = 0
    for si, (a, b) in enumerate(SEGMENTS):
        f0, f1 = int(a * src_fps), int(b * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for m in models:
            m.predictor = None                       # restart tracks at each cut
        print(f"segment {si+1}/{len(SEGMENTS)}: {a}-{b}s", flush=True)
        k = 0
        for fi in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break
            if (fi - f0) % STRIDE:
                continue
            rows = []
            for i, ((label, imgsz, color), model) in enumerate(zip(PANELS, models)):
                r = model.track(frame, imgsz=imgsz, conf=CONF, device="cuda",
                                persist=(k > 0), tracker=TRACKER, verbose=False)[0]
                panel, n, tiny = draw(frame, r)
                tot[i] += n
                tot_tiny[i] += tiny
                rows += [header(label, color, n, tiny, tot[i], tot_tiny[i]), panel]
            k += 1

            title = np.full((HDR, W, 3), 15, np.uint8)
            cv2.putText(title, "Same 1080p frame, same model - only the detector input size differs",
                        (28, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(title, f"t = {fi/src_fps:6.1f}s", (W - 240, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (170, 170, 170), 2, cv2.LINE_AA)
            foot = np.full((FOOT, W, 3), 15, np.uint8)
            cv2.putText(foot, "cyan = object under 24px tall: the bin where imgsz=640 scored 0.000 recall",
                        (28, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 240, 255), 1, cv2.LINE_AA)
            vw.write(np.vstack([title] + rows + [foot]))
            written += 1
    cap.release()
    vw.release()
    print(f"\nwrote {args.out}: {written} frames, {written/(src_fps/STRIDE):.1f}s, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")
    for i, (label, _, _) in enumerate(PANELS):
        print(f"  {label:12s} total {tot[i]:5d}   distant(<{TINY_PX}px) {tot_tiny[i]:4d}")


if __name__ == "__main__":
    main()
