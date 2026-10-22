"""Demo video: low frame rate breaks the tracker, and a looser gate gets most of it back.

Panels: 60 fps default, 5 fps default, 5 fps tuned. The 5 fps panels hold boxes
between detector runs, fading as they go stale, which is what such a pipeline knows.
"""
import argparse, os
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER

MASTER = "1080p_video_driving_sample.mp4"
MODEL = "yolo11s.pt"                 # same model as Experiments B/D
CONF = 0.25
IMGSZ = 1280
CROP = (520, 78, 1820, 318)
ROAD_BAND_Y = 400
SEGMENTS = [(104, 122), (173, 187), (213, 222), (265, 272)]
REF_SPEED_KMH = 200.0                # stated assumption for the blind-distance readout

# (label, source-frame stride, tracker cfg, colour)
PANELS = [
    ("60 fps  -  default ByteTrack",  1, "bytetrack.yaml",                   (70, 230, 70)),
    ("5 fps   -  default ByteTrack", 12, "bytetrack.yaml",                   (70, 110, 255)),
    ("5 fps   -  TUNED ByteTrack",   12, "trackers/bytetrack_tuned.yaml",    (255, 190, 60)),
]

W = 1920
S = W / (CROP[2] - CROP[0])
PANEL_H = int(round((CROP[3] - CROP[1]) * S))
HDR, FOOT = 50, 46
OUT_H = HDR + (HDR + PANEL_H) * len(PANELS) + FOOT
COL = {"vehicle": (80, 240, 80), "twowheeler": (255, 80, 255),
       "traffic light": (80, 240, 255), "stop sign": (80, 160, 255)}


def boxes_from(res):
    out = []
    b = res.boxes
    if b is None or len(b) == 0:
        return out
    cls = b.cls.cpu().numpy().astype(int)
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(cls), -1)
    for box, c, cf, tid in zip(xyxy, cls, conf, ids):
        if c not in DRIVING_CLASSES:
            continue
        sup = SUPER[c]
        if sup == "person" or (box[1] + box[3]) / 2 >= ROAD_BAND_Y:
            continue
        out.append((box.copy(), DRIVING_CLASSES[c], sup, float(cf), int(tid)))
    return out


def draw(frame, held, stale_frac):
    """Held boxes fade towards grey as they go stale."""
    x0, y0, x1, y1 = CROP
    canvas = cv2.resize(frame[y0:y1, x0:x1], (W, PANEL_H), interpolation=cv2.INTER_LINEAR)
    fade = 0.35 + 0.65 * (1.0 - min(stale_frac, 1.0))       # 1.0 fresh -> 0.35 stale
    for box, name, sup, cf, tid in held:
        base = COL[sup]
        col = tuple(int(c * fade + 105 * (1 - fade)) for c in base)
        p1 = (int((box[0] - x0) * S), int((box[1] - y0) * S))
        p2 = (int((box[2] - x0) * S), int((box[3] - y0) * S))
        cv2.rectangle(canvas, p1, p2, col, 2)
        tag = f"{name}" + (f" #{tid}" if tid > 0 else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 3, p1[1] - 5)
        cv2.rectangle(canvas, (p1[0], ty - th - 4), (p1[0] + tw + 5, ty + 3), col, -1)
        cv2.putText(canvas, tag, (p1[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def header(label, color, n, ids_seen, stale_ms, shown, updates):
    bar = np.full((HDR, W, 3), 26, np.uint8)
    cv2.rectangle(bar, (0, 0), (12, HDR), color, -1)
    cv2.putText(bar, label, (28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
    cv2.putText(bar, f"boxes now: {n}", (520, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.64, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(bar, f"track IDs seen: {ids_seen}", (740, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.64, (210, 210, 210), 1, cv2.LINE_AA)
    # Per update, not cumulative: 60 fps updates 12x as often.
    per = shown / max(updates, 1)
    cv2.putText(bar, f"boxes per detector update: {per:.2f}", (1010, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 235, 140), 2, cv2.LINE_AA)
    hot = stale_ms > 60
    m = REF_SPEED_KMH / 3.6 * (stale_ms / 1000.0)
    cv2.putText(bar, f"stale {stale_ms:4.0f} ms = {m:4.1f} m blind",
                (1420, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                (60, 200, 255) if hot else (140, 140, 140), 2 if hot else 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_fps_tracking.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(MASTER)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    models = [YOLO(MODEL) for _ in PANELS]
    for m in models:
        m.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=640, device="cuda", verbose=False)

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), src_fps, (W, OUT_H))
    assert vw.isOpened(), "VideoWriter failed to open"

    shown = [0] * len(PANELS)
    updates = [0] * len(PANELS)
    ids_seen = [set() for _ in PANELS]
    written = 0
    for si, (a, b) in enumerate(SEGMENTS):
        f0, f1 = int(a * src_fps), int(b * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        held = [[] for _ in PANELS]
        last_update = [0] * len(PANELS)
        for m in models:
            m.predictor = None
        print(f"segment {si+1}/{len(SEGMENTS)}: {a}-{b}s", flush=True)
        k = 0
        for fi in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break
            rows = []
            for i, ((label, stride, cfg, color), model) in enumerate(zip(PANELS, models)):
                if k % stride == 0:                      # detector runs on this frame
                    r = model.track(frame, imgsz=IMGSZ, conf=CONF, device="cuda",
                                    persist=(k > 0), tracker=cfg, verbose=False)[0]
                    held[i] = boxes_from(r)
                    last_update[i] = k
                    shown[i] += len(held[i])
                    updates[i] += 1
                    ids_seen[i].update(t for *_, t in held[i] if t > 0)
                age = k - last_update[i]
                stale_ms = age / src_fps * 1000.0
                panel = draw(frame, held[i], age / max(stride - 1, 1))
                rows += [header(label, color, len(held[i]), len(ids_seen[i]),
                                stale_ms, shown[i], updates[i]), panel]
            k += 1

            title = np.full((HDR, W, 3), 15, np.uint8)
            cv2.putText(title, "Same clip, real time. Frame rate does not change what the "
                               "detector sees - it changes what the TRACKER keeps.",
                        (28, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(title, f"t = {fi/src_fps:6.1f}s", (W - 230, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2, cv2.LINE_AA)
            foot = np.full((FOOT, W, 3), 15, np.uint8)
            cv2.putText(foot, f"Boxes fade as they go stale. 'Per update' is comparable across "
                              f"panels; raw totals are not. Blind distance assumes "
                              f"{REF_SPEED_KMH:.0f} km/h (dash reads up to 335).",
                        (28, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 1, cv2.LINE_AA)
            vw.write(np.vstack([title] + rows + [foot]))
            written += 1
    cap.release()
    vw.release()
    print(f"\nwrote {args.out}: {written} frames, {written/src_fps:.1f}s, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")
    for i, (label, *_ ) in enumerate(PANELS):
        print(f"  {label:32s} boxes {shown[i]:6d} / {updates[i]:5d} updates "
              f"= {shown[i]/max(updates[i],1):.3f} per update   distinct IDs {len(ids_seen[i]):4d}")


if __name__ == "__main__":
    main()
