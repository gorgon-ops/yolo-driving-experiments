"""Same frame at every resolution, nearest-neighbour upscaled: what the detector saw."""
import pickle
import cv2, numpy as np
from ultralytics import YOLO
from cvlib import DRIVING_CLASSES, SUPER, round32

MASTER = "1080p_video_driving_sample.mp4"
RESOLUTIONS = [("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480),
               ("360p", 640, 360), ("240p", 426, 240)]
BAND_Y = 400          # road band in master coords
CROP = (90, 400)      # generous: the camera drifts
COL = {"vehicle": (0, 220, 0), "twowheeler": (255, 0, 255),
       "traffic light": (0, 220, 255), "stop sign": (0, 120, 255), "person": (0, 0, 255)}


def pick_frame():
    d = pickle.load(open("results_resolution2.pkl", "rb"))
    best, bestn = None, -1
    for r in d["records"]:
        if r["pipeline"] != "native" or r["res"] != "1080p":
            continue
        rb, rs = r["ref_boxes"], r["ref_super"]
        keep = [i for i in range(len(rb))
                if rs[i] != "person" and (rb[i][1] + rb[i][3]) / 2 < BAND_Y]
        n = len(keep)
        if n > bestn:
            bestn, best = n, r["frame"]
    print(f"frame {best}: {bestn} road objects in the reference")
    return best


def main():
    fidx = pick_frame()
    cap = cv2.VideoCapture(MASTER)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    cap.release()
    assert ok

    model = YOLO("yolo11m.pt")
    panels = []
    for name, w, h in RESOLUTIONS:
        small = frame if w == 1920 else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        imgsz = round32(w, cap=1280)
        r = model.predict(small, imgsz=imgsz, conf=0.25, verbose=False)[0]
        # real low-res pixels back to master scale, no smoothing
        canvas = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        n = 0
        b = r.boxes
        if b is not None and len(b):
            cls = b.cls.cpu().numpy().astype(int)
            sc = 1920.0 / w
            for box, c, cf in zip(b.xyxy.cpu().numpy(), cls, b.conf.cpu().numpy()):
                if c not in DRIVING_CLASSES or SUPER[c] == "person":
                    continue
                x1, y1, x2, y2 = box * sc
                if (y1 + y2) / 2 >= BAND_Y:
                    continue
                n += 1
                col = COL[SUPER[c]]
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                cv2.putText(canvas, f"{DRIVING_CLASSES[c]} {cf:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        crop = canvas[CROP[0]:CROP[1]]
        bar = np.zeros((44, crop.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{name}  {w}x{h}   imgsz={imgsz}   {n} road detections",
                    (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([bar, crop]))
        print(f"  {name:6s} imgsz={imgsz:5d}  {n} road detections")

    grid = np.vstack(panels)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite("fig_visual_resolution.png", grid)
    print("wrote fig_visual_resolution.png", grid.shape)


if __name__ == "__main__":
    main()
