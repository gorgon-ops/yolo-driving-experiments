"""Score Experiment B. Per-frame detection is fps-invariant, so these metrics measure what is not."""
import pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cvlib import iou_matrix

# The clip spans 0-335 km/h (an Autobahn top-speed run): report the blind gap
# across that range rather than one assumed speed.
SPEEDS_KMH = [50, 100, 200, 300]
REPORT_SPEED = 200.0   # representative cruising speed for the summary column

# The driver is a permanent, perfectly stable track that would dominate every
# continuity metric, so only the road band is kept.
ROAD_BAND_Y_540 = 200   # overridden per file from work_h when present
DROP_SUPER = {"person"}


def road_only(frames, band=ROAD_BAND_Y_540):
    out = []
    for f in frames:
        keep = [i for i, s in enumerate(f["sup"])
                if s not in DROP_SUPER
                and (f["boxes"][i][1] + f["boxes"][i][3]) / 2 < band]
        out.append(dict(t=f["t"],
                        ids=[f["ids"][i] for i in keep],
                        sup=[f["sup"][i] for i in keep],
                        boxes=f["boxes"][keep] if len(keep) else f["boxes"][:0]))
    return out


def track_stats(frames, eff_fps):
    seen = {}
    for k, f in enumerate(frames):
        for tid in f["ids"]:
            seen.setdefault(tid, []).append(k)
    durations, gaps = [], 0
    for tid, ks in seen.items():
        durations.append((ks[-1] - ks[0] + 1) / eff_fps)
        gaps += sum(1 for a, b in zip(ks, ks[1:]) if b - a > 1)
    return dict(n_tracks=len(seen),
                mean_track_sec=float(np.mean(durations)) if durations else 0.0,
                median_track_sec=float(np.median(durations)) if durations else 0.0,
                frag_per_track=gaps / max(len(seen), 1))


def assoc_difficulty(frames):
    """Mean best-IoU between consecutive frames. Below ~0.3 ByteTrack cannot link an object to itself."""
    vals = []
    for a, b in zip(frames, frames[1:]):
        if len(a["boxes"]) == 0 or len(b["boxes"]) == 0:
            continue
        m = iou_matrix(a["boxes"], b["boxes"])
        vals.extend(m.max(axis=1).tolist())
    return float(np.mean(vals)) if vals else np.nan


def summarise(path="results_fps.pkl"):
    d = pickle.load(open(path, "rb"))
    band = 400.0 * d.get("work_h", 540) / 1080.0
    rows = []
    for fps, c in sorted(d["conditions"].items(), reverse=True):
        fr = road_only(c["frames"], band)
        ef = c["eff_fps"]
        dets = [len(f["ids"]) for f in fr]
        st = track_stats(fr, ef)
        # how far the car travels between two processed frames
        gap_m = (REPORT_SPEED / 3.6) / ef
        rows.append(dict(
            fps=fps, eff_fps=ef, n_frames=c["n_frames"],
            det_per_frame=float(np.mean(dets)),
            mean_assoc_iou=assoc_difficulty(fr),
            gap_m=gap_m,
            **{f"gap_m_{s}": (s / 3.6) / ef for s in SPEEDS_KMH},
            worst_case_latency_ms=1000.0 / ef,
            infer_ms=c["infer_ms"],
            realtime=c["infer_ms"] <= 1000.0 / ef,
            cpu_budget_ratio=c["infer_ms"] / (1000.0 / ef),
            **st))
    return pd.DataFrame(rows), d


def plot(s, path="fig_fps.png", dur=18.0, raw=None):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Experiment B — frame rate vs perception quality\n"
                 f"yolo11s@1280 + ByteTrack, one {dur:.0f}s traffic-dense window, temporally subsampled", fontsize=13)
    x = s.eff_fps.values

    a = ax[0, 0]
    a.plot(x, s.det_per_frame, "o-", color="#4c72b0", lw=2, label="tracked output (ByteTrack)")
    if raw is not None:
        a.plot([r[0] for r in raw], [r[2] for r in raw], "s--", color="#937860", lw=2,
               label="raw detector, no tracker")
    a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([f"{v:.0f}" for v in x])
    a.minorticks_off()
    a.set_xlabel("fps"); a.set_ylabel("road detections / frame")
    a.set_title("Raw detection is fps-invariant;\nthe TRACKER is what loses objects")
    a.grid(alpha=.3); a.set_ylim(bottom=0); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(x, s.mean_assoc_iou, "o-", color="#c44e52", lw=2)
    a.axhline(0.3, ls="--", c="k", lw=1)
    a.text(x.min(), 0.32, "ByteTrack association floor", fontsize=8)
    a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([f"{v:.0f}" for v in x])
    a.minorticks_off()
    a.set_xlabel("fps"); a.set_ylabel("mean best IoU between frames")
    a.set_title("Frame-to-frame overlap — this is what breaks")
    a.grid(alpha=.3); a.set_ylim(0, 1)

    a = ax[1, 0]
    a.plot(x, s.mean_track_sec, "o-", color="#55a868", lw=2, label="mean track lifetime (s)")
    a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([f"{v:.0f}" for v in x])
    a.minorticks_off()
    a.set_xlabel("fps"); a.set_ylabel("seconds"); a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot(x, s.frag_per_track, "s--", color="#8172b2", lw=2, label="fragmentation / track")
    a2.set_ylabel("gaps per track"); a2.legend(fontsize=8, loc="center right")
    a.set_title("Track continuity\n(survivorship-biased: only easy tracks confirm at low fps)",
                fontsize=10); a.grid(alpha=.3)

    a = ax[1, 1]
    for sp in SPEEDS_KMH:
        a.plot(x, s[f"gap_m_{sp}"], "o-", lw=2, label=f"{sp} km/h")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(x); a.set_xticklabels([f"{v:.0f}" for v in x])
    a.minorticks_off()
    a.set_xlabel("fps"); a.set_ylabel("metres (log)")
    a.set_title("Blind distance between processed frames\n(dash in this clip reads up to 335 km/h)")
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8, title="speed")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "results_fps.pkl"
    s, d = summarise(p)
    pd.set_option("display.width", 220)
    cols = ["fps", "n_frames", "det_per_frame", "mean_assoc_iou", "n_tracks",
            "mean_track_sec", "frag_per_track"] + [f"gap_m_{v}" for v in SPEEDS_KMH] + [
            "worst_case_latency_ms", "infer_ms", "cpu_budget_ratio", "realtime"]
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tag = "2" if "2" in p else ""
    raw = None
    try:
        raw = pickle.load(open("results_fps_control.pkl", "rb"))["rows"]
        print("\nraw (no tracker) control:")
        for fps, n, m in raw:
            print(f"  {fps:3d} fps: {n:5d} frames, {m:.3f} det/frame")
    except FileNotFoundError:
        print("\n(no raw control file yet)")
    s.to_csv(f"results_fps{tag}.csv", index=False)
    plot(s, f"fig_fps{tag}.png", dur=d.get("dur_sec", 18.0), raw=raw)
