"""A non-learned sign detector (HSV colour + contour shape). Not ground truth: useful
because its errors are uncorrelated with a CNN's and it has no learned scale invariance.
"""
import cv2, numpy as np

# Windscreen, 1920x1080 coords. The left edge excludes the phone mount, whose
# blue screen reads as a sign.
ROI_POLY = np.array([(668, 250), (790, 156), (1150, 130), (1520, 142), (1706, 200),
                     (1716, 300), (1500, 332), (1000, 352), (772, 330), (672, 296)], np.int32)
# The rear-view mirror shows the road behind.
MIRROR_POLY = np.array([(1195, 158), (1470, 158), (1470, 240), (1195, 240)], np.int32)

# German/EU signs: red-rimmed prohibition+warning, blue mandatory+information.
COLOR_RANGES = {
    "red": [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    "blue": [((100, 110, 60), (130, 255, 255))],
}


def roi_mask(w, h):
    sx, sy = w / 1920.0, h / 1080.0
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [(ROI_POLY * (sx, sy)).astype(np.int32)], 255)
    cv2.fillPoly(m, [(MIRROR_POLY * (sx, sy)).astype(np.int32)], 0)
    return m


def detect_signs(bgr, mask=None, min_area=12, max_area=40000,
                 min_fill=0.35, max_aspect=1.9):
    h, w = bgr.shape[:2]
    if mask is None:
        mask = roi_mask(w, h)
    blur = cv2.GaussianBlur(bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for cname, ranges in COLOR_RANGES.items():
        cm = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            cm |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        cm = cv2.bitwise_and(cm, mask)
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, k, iterations=2)
        cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, k, iterations=1)

        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 3 or bh < 3:
                continue
            aspect = max(bw / bh, bh / bw)
            if aspect > max_aspect:
                continue
            # A ring (no-entry) has low fill against its bbox, so measure against the
            # filled contour.
            fill = area / float(bw * bh)
            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / max(per * per, 1e-6)
            if fill < min_fill:
                continue
            out.append(dict(bbox=(x, y, x + bw, y + bh), color=cname, area=float(area),
                            aspect=float(aspect), fill=float(fill), circ=float(circ),
                            side=float(max(bw, bh))))
    return _nms(out)


def _nms(dets, thr=0.3):
    """A red rim and blue face of adjacent signs can overlap; keep the larger."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d["area"])
    keep = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        ok = True
        for k in keep:
            a1, b1, a2, b2 = k["bbox"]
            ix = max(0, min(x2, a2) - max(x1, a1))
            iy = max(0, min(y2, b2) - max(y1, b1))
            inter = ix * iy
            u = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
            if inter / max(u, 1e-6) > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep
