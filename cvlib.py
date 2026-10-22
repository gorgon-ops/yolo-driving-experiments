import numpy as np

# COCO ids that matter for a road-driving scene
DRIVING_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign",
}

# Every COCO detector swaps car/truck/bus; scoring them apart measures label
# noise, not resolution.
SUPER = {
    0: "person", 1: "twowheeler", 3: "twowheeler", 2: "vehicle",
    5: "vehicle", 7: "vehicle", 9: "traffic light", 11: "stop sign",
}

# Box height in 1080p pixels, i.e. how far away.
SIZE_BINS = [
    ("tiny  (<24px)", 0, 24),
    ("small (24-48)", 24, 48),
    ("med   (48-96)", 48, 96),
    ("large (>96px)", 96, 10 ** 9),
]


def size_bin(h):
    for i, (_, lo, hi) in enumerate(SIZE_BINS):
        if lo <= h < hi:
            return i
    return len(SIZE_BINS) - 1


def iou_matrix(a, b):
    """(N,4) xyxy, (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def greedy_match(ref_boxes, ref_super, det_boxes, det_super, det_conf, iou_thr=0.5):
    nref, ndet = len(ref_boxes), len(det_boxes)
    ref_matched = np.zeros(nref, dtype=bool)
    det_matched = np.zeros(ndet, dtype=bool)
    matched_iou = np.zeros(nref, dtype=np.float32)
    if nref == 0 or ndet == 0:
        return ref_matched, det_matched, matched_iou

    ious = iou_matrix(det_boxes, ref_boxes)  # (ndet, nref)
    same = np.array(det_super)[:, None] == np.array(ref_super)[None, :]
    ious = np.where(same, ious, 0.0)

    for di in np.argsort(-np.asarray(det_conf)):
        cand = np.where(~ref_matched, ious[di], -1.0)
        ri = int(np.argmax(cand))
        if cand[ri] >= iou_thr:
            ref_matched[ri] = True
            det_matched[di] = True
            matched_iou[ri] = cand[ri]
    return ref_matched, det_matched, matched_iou


def round32(x, cap=1280):
    """Nearest multiple of 32, clamped: YOLO strides require it."""
    return int(min(cap, max(32, round(x / 32) * 32)))
