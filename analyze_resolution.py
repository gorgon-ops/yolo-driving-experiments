import pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cvlib import greedy_match, size_bin, SIZE_BINS

IOU_THR = 0.5
RES_ORDER = ["1080p", "720p", "480p", "360p", "240p"]

# The first ~57 s are a parked beauty shot and a second cabin angle.
DRIVE_START = 3460
# The in-cabin driver is a "person" in every frame and would flatten the curve;
# there are no pedestrians on an Autobahn.
DROP_SUPER = {"person"}
# Below this (1080p coords) is the cabin, which yolo11m at imgsz=640 reads as
# a car ~once per frame.
ROAD_BAND_Y = 400


def _drop(rb, rs, db, ds, dc):
    rk = [i for i, s in enumerate(rs)
          if s not in DROP_SUPER and (rb[i][1] + rb[i][3]) / 2 < ROAD_BAND_Y]
    dk = [j for j, s in enumerate(ds)
          if s not in DROP_SUPER and (db[j][1] + db[j][3]) / 2 < ROAD_BAND_Y]
    return (rb[rk] if len(rb) else rb, [rs[i] for i in rk],
            db[dk] if len(db) else db, [ds[j] for j in dk],
            dc[dk] if len(dc) else dc)


def score(path="results_resolution.pkl"):
    d = pickle.load(open(path, "rb"))
    rows = []
    n_all = n_kept = 0
    for r in d["records"]:
        n_all += 1
        if r["frame"] < DRIVE_START:
            continue
        n_kept += 1
        rb, rs, db, ds, dc = _drop(r["ref_boxes"], r["ref_super"],
                                   r["det_boxes"], r["det_super"], r["det_conf"])
        ref_m, det_m, _ = greedy_match(rb, rs, db, ds, dc, IOU_THR)
        heights = (rb[:, 3] - rb[:, 1]) if len(rb) else np.zeros(0)
        for i in range(len(rb)):
            rows.append(dict(pipeline=r["pipeline"], res=r["res"], imgsz=r["imgsz"],
                             frame=r["frame"], event=r.get("event", -1), kind="ref",
                             sup=rs[i], bin=size_bin(heights[i]), hit=bool(ref_m[i])))
        for j in range(len(db)):
            rows.append(dict(pipeline=r["pipeline"], res=r["res"], imgsz=r["imgsz"],
                             frame=r["frame"], event=r.get("event", -1), kind="det",
                             sup=ds[j], bin=-1, hit=bool(det_m[j])))
    df = pd.DataFrame(rows)
    print(f"kept {n_kept}/{n_all} records (driving segment, road band, driver excluded)")
    ref = df[df.kind == "ref"]
    if len(ref):
        one = ref[(ref.pipeline == "native") & (ref.res == "1080p")]
        print(f"scored objects per condition: {len(one)}  "
              f"across {one.frame.nunique()} frames and {one.event.nunique()} traffic events")
        print("NOTE: boxes within one event are the same vehicles seen repeatedly -- "
              "effective N is closer to the event count than the box count.")

    tim = {k: float(np.mean(v)) for k, v in d["timings"].items()}
    return df, tim, d


def summarise(df, tim):
    out = []
    for (p, res), g in df.groupby(["pipeline", "res"]):
        ref, det = g[g.kind == "ref"], g[g.kind == "det"]
        tp = int(ref.hit.sum())
        recall = tp / max(len(ref), 1)
        prec = det.hit.sum() / max(len(det), 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        ms = tim[(p, res)]
        row = dict(pipeline=p, res=res, imgsz=int(g.imgsz.iloc[0]),
                   n_ref=len(ref), n_det=len(det), recall=recall,
                   precision=prec, f1=f1, ms=ms, fps=1000 / ms)
        for b, (name, _, _) in enumerate(SIZE_BINS):
            sub = ref[ref["bin"] == b]
            row[f"recall_{b}"] = sub.hit.mean() if len(sub) else np.nan
            row[f"n_{b}"] = len(sub)
        out.append(row)
    s = pd.DataFrame(out)
    s["res"] = pd.Categorical(s.res, RES_ORDER, ordered=True)
    return s.sort_values(["pipeline", "res"]).reset_index(drop=True)


def plot(s, path="fig_resolution.png"):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Experiment A — source resolution vs detection quality\n"
                 "yolo11m, scored against yolo11x@1280 pseudo-GT (IoU 0.5)", fontsize=13)
    x = np.arange(len(RES_ORDER))
    colors = {"fixed640": "#c44e52", "native": "#4c72b0"}
    labels = {"fixed640": "detector input fixed at 640",
              "native": "detector input scales with source"}

    a = ax[0, 0]
    for p, g in s.groupby("pipeline"):
        a.plot(x, g.recall.values, "o-", color=colors[p], label=labels[p], lw=2)
    a.set_xticks(x); a.set_xticklabels(RES_ORDER); a.set_ylabel("recall")
    a.set_title("Overall recall"); a.grid(alpha=.3); a.legend(fontsize=8); a.set_ylim(0, 1)

    a = ax[0, 1]
    for p, g in s.groupby("pipeline"):
        a.plot(x, g.precision.values, "s--", color=colors[p], label=labels[p], lw=2)
    a.set_xticks(x); a.set_xticklabels(RES_ORDER); a.set_ylabel("precision")
    a.set_title("Precision (vs pseudo-GT)"); a.grid(alpha=.3); a.legend(fontsize=8); a.set_ylim(0, 1)

    a = ax[1, 0]
    g = s[s.pipeline == "native"]
    w = 0.2
    for b, (name, _, _) in enumerate(SIZE_BINS):
        a.bar(x + (b - 1.5) * w, g[f"recall_{b}"].values, w, label=name)
    a.set_xticks(x); a.set_xticklabels(RES_ORDER); a.set_ylabel("recall")
    a.set_title("Recall by object size — resolution hurts small objects first")
    a.grid(alpha=.3, axis="y"); a.legend(fontsize=8); a.set_ylim(0, 1)

    a = ax[1, 1]
    for p, g in s.groupby("pipeline"):
        a.plot(g.fps.values, g.recall.values, "o-", color=colors[p], label=labels[p], lw=2)
        for _, r in g.iterrows():
            a.annotate(r.res, (r.fps, r.recall), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("throughput on this CPU (fps)"); a.set_ylabel("recall")
    a.set_title("Quality vs cost"); a.grid(alpha=.3); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results_resolution.pkl"
    df, tim, raw = score(path)
    s = summarise(df, tim)
    cols = ["pipeline", "res", "imgsz", "n_ref", "recall", "precision", "f1", "ms", "fps"]
    pd.set_option("display.width", 200)
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    bc = ["pipeline", "res"] + [f"recall_{b}" for b in range(4)] + [f"n_{b}" for b in range(4)]
    print(s[bc].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tag = "2" if "2" in path else ""
    s.to_csv(f"results_resolution{tag}.csv", index=False)
    plot(s, f"fig_resolution{tag}.png")
