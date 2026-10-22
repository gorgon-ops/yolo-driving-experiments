"""Assemble FINDINGS.md from the result CSVs, the control run and the bootstrap."""
import pickle
import pandas as pd

R = pd.read_csv("results_resolution2.csv")
F = pd.read_csv("results_fps2.csv")
CTRL = pickle.load(open("results_fps_control.pkl", "rb"))["rows"]
BOOT = open("bootstrap_output.txt").read()
ORDER = ["1080p", "720p", "480p", "360p", "240p"]
R["res"] = pd.Categorical(R.res, ORDER, ordered=True)
R = R.sort_values(["pipeline", "res"])

f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"
f0 = lambda v: f"{v:.0f}"
s_ = str


def md(df, cols, heads, fmt):
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join(["---"] * len(heads)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c](r[c]) for c in cols) + " |")
    return "\n".join(out)


L = []
A = L.append
A("# Resolution and frame rate vs YOLO detection quality\n")
A("Dashcam clip: 281 s, AutoTopNL German Autobahn run, cabin-POV. Every condition "
  "derives from the **same 1080p master**, so resolution and frame rate vary "
  "independently of codec, bitrate and scene content.\n")

A("## Read this first: what the footage forced\n")
A("Four properties of this video shaped every number below.\n")
A("1. **Only frames 3460-16855 (57.6 s onward) are driving.** The opening ~57 s is a "
  "static beauty shot of the parked car on a cobblestone driveway, plus a second cabin "
  "angle with no road view. Found by frame-differencing; after 57.6 s there are no cuts.\n")
A("2. **The camera is inside the cabin and drifts.** A hand-drawn windscreen ROI that "
  "fits at 207 s sits on the dashboard at 58 s. Geometric ROIs were abandoned; scoring "
  "instead keeps only boxes whose centre lies in the upper 400 px (37%) of the master "
  "frame, which survives the drift.\n")
A("3. **The driver is detected as `person` in every frame** - a large, close, trivially "
  "detected object that would flatten the very curve being measured. Class dropped; "
  "Autobahns have no pedestrians.\n")
A("4. **94% of driving frames contain no road object at all.** A pre-screen of 2233 "
  "frames found 137 with at least one object, clustered into 22 traffic events. The "
  "condition grid was aimed at those events; a uniform sample scored only 27 objects.\n")
A("A fifth constraint is structural: **COCO has no class for European road signs.** "
  "Stock YOLO cannot detect the blue motorway or red prohibition signs in this clip at "
  "any resolution, so the sign question is unanswerable with off-the-shelf weights.\n")

A("## How quality was measured\n")
A("No human labels exist for this footage, so quality is **agreement with a strong "
  "detector**, not absolute truth: `yolo11x` at `imgsz=1280`, conf 0.30, on the "
  "untouched 1080p frame. Test configs run `yolo11m` at conf 0.25, matched greedily by "
  "IoU >= 0.5. `car`/`truck`/`bus` collapse into one `vehicle` class because COCO "
  "detectors swap them constantly and that is label noise, not a resolution effect.\n")
A("No config can score above the reference, and errors the reference also makes are "
  "invisible. This is the right tool for ranking configs against each other and the "
  "wrong tool for quoting an absolute mAP.\n")
A("A classical HSV colour + contour sign detector was built (`classic_signs.py`) as an "
  "*independent* cross-check, on the theory that its errors would be uncorrelated with "
  "a CNN's. Audited over 52 detections it scored **~21% precision**, the failures being "
  "red truck rear panels, taillights, and the driver's blue phone screen. Those errors "
  "correlate with road content, which makes it useless as a reference. Recorded here as "
  "a negative result; it is not used in scoring.\n")

A("## Experiment A - resolution\n")
A(f"**{int(R.n_ref.iloc[0])} scored objects across 180 frames and 17 traffic events.**\n")
cols = ["res", "imgsz", "recall", "precision", "f1", "ms", "fps"]
heads = ["source", "imgsz", "recall", "precision", "F1", "ms/frame", "fps"]
fmt = dict(res=s_, imgsz=f0, recall=f3, precision=f3, f1=f3, ms=f0, fps=f2)
A("**Detector input fixed at 640** (what you get by default):\n")
A(md(R[R.pipeline == "fixed640"], cols, heads, fmt) + "\n")
A("**Detector input scales with source:**\n")
A(md(R[R.pipeline == "native"], cols, heads, fmt) + "\n")

A("### Recall by object size - the headline\n")
A("Bins are box height in master-1080p pixels: roughly how far away the object is.\n")
bc = ["res"] + [f"recall_{i}" for i in range(4)]
bh = ["source", "tiny <24px", "small 24-48", "med 48-96", "large >96px"]
bf = dict(res=s_, **{f"recall_{i}": f3 for i in range(4)})
A("*fixed 640:*\n")
A(md(R[R.pipeline == "fixed640"], bc, bh, bf) + "\n")
A("*scaled input:*\n")
A(md(R[R.pipeline == "native"], bc, bh, bf) + "\n")
n = R[R.pipeline == "native"].iloc[0]
A(f"Bin sizes: n = {int(n.n_0)}, {int(n.n_1)}, {int(n.n_2)}, {int(n.n_3)}.\n")
A("**At `imgsz=640`, objects under 24 px are detected 0% of the time at every source "
  "resolution.** Raising `imgsz` to 1280 lifts that to 0.760. Large objects (>96 px) "
  "barely move at all.\n")
A("![resolution](fig_resolution2.png)\n")
A("![visual](fig_visual_resolution.png)\n")

A("### Which differences are statistically real\n")
A("Boxes within one traffic event are the same vehicles seen repeatedly, so naive "
  "binomial errors on 263 boxes would badly overstate confidence. This bootstrap "
  "resamples whole events (17 clusters, B=4000).\n")
A("```\n" + BOOT[BOOT.index("cluster bootstrap"):].strip() + "\n```\n")
A("Surviving claims: **`imgsz` is what matters, source resolution is not**, and "
  "**1080p buys nothing over 720p**. The 720p-vs-480p gap (+0.163) does *not* reach "
  "significance with 17 clusters.\n")

A("## Experiment B - frame rate\n")
A("`yolo11s` at `imgsz=1280`, 1280x720 frames, densest 18 s window (115.7-133.7 s), "
  "ByteTrack, road band only.\n")
fc = ["fps", "n_frames", "det_per_frame", "mean_assoc_iou", "n_tracks",
      "gap_m_50", "gap_m_200", "gap_m_300", "worst_case_latency_ms"]
fh = ["fps", "frames", "tracked det/frame", "frame-to-frame IoU", "tracks",
      "blind gap @50", "@200", "@300 km/h", "latency ms"]
ff = dict(fps=f0, n_frames=f0, det_per_frame=f3, mean_assoc_iou=f3, n_tracks=f0,
          gap_m_50=f2, gap_m_200=f2, gap_m_300=f2, worst_case_latency_ms=f0)
A(md(F, fc, fh, ff) + "\n")

A("### Control: the detector is fps-invariant, the tracker is not\n")
A("Detection is per-frame independent, so running the detector once over all 1080 "
  "frames and averaging over each condition's subsample isolates the tracker's "
  "contribution.\n")
A("| fps | raw det/frame (no tracker) | tracked det/frame | retained |")
A("|---|---|---|---|")
tr = dict(zip(F.fps, F.det_per_frame))
for fps, _, m in CTRL:
    A(f"| {fps} | {m:.3f} | {tr[fps]:.3f} | {100 * tr[fps] / m:.0f}% |")
A("")
A("Raw detection is flat (0.827-0.917, no trend). Tracked output falls 0.640 to 0.333. "
  "**The entire loss is ByteTrack withholding boxes whose tracks it cannot confirm** - "
  "77% of detections survive tracking at 60 fps, only 36% at 2 fps.\n")
A("Mean frame-to-frame IoU falls 0.871 to 0.464 against ByteTrack's ~0.3 association "
  "floor: by 2 fps a moving vehicle barely overlaps itself between frames, so the "
  "tracker assigns it a new identity.\n")
A("**Track-continuity figures are survivorship-biased.** Mean track lifetime *rises* as "
  "fps falls (0.46 s to 1.30 s) and fragmentation *falls* (1.53 to 0.20). That is not "
  "low fps improving tracking; only large, close, easy objects ever confirm a track at "
  "2 fps.\n")
A("The `infer_ms` column in the CSV was measured under CPU contention and is not "
  "trustworthy. A clean benchmark for this config is ~247 ms/frame, i.e. a ~4 fps "
  "real-time ceiling on this machine (CPU only, no CUDA).\n")
A("![fps](fig_fps2.png)\n")

A("## Limits\n")
A("- **Effective sample is 17 traffic events / 32 tracks**, not 263 boxes. Enough for "
  "the large effects reported; not enough for precise fragmentation rates.\n")
A("- **Pseudo-ground-truth, not ground truth.** Blind spots shared with `yolo11x` are "
  "invisible. Hand-labelling a small set, or a TTA/ensemble reference, would measure "
  "that bias.\n")
A("- **This clip is a poor instrument for the question**: cabin POV wastes ~90% of "
  "pixels, the camera drifts, traffic is sparse, and the most interesting objects "
  "(signs) are not COCO classes. The code transfers unchanged to a road-facing clip "
  "with steady traffic, which would answer it far better.\n")

open("FINDINGS.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote FINDINGS.md")
