# Resolution, frame rate and tracking vs YOLO detection on driving video

How much does input resolution, frame rate and tracker choice actually change
what an off-the-shelf detector sees on the road - measured, not asserted, on a
real dashcam clip where every condition is derived from one 1080p master so the
only thing that varies is the thing under test.

The detector is stock YOLO11 (COCO weights). Quality is scored as agreement
with a strong reference (`yolo11x` at `imgsz=1280` on the untouched frame), so
these numbers rank configurations against each other honestly; they are not
absolute mAP. Full methodology, caveats and the statistics are in
[`FINDINGS.md`](FINDINGS.md).

## What the four experiments found

**A - Resolution.** Detector *input* size (`imgsz`) is the lever that matters;
*source* resolution barely does. At `imgsz=640` objects under 24 px are detected
**0%** of the time at every source resolution; raising `imgsz` to 1280 lifts
that to 0.76. A cluster bootstrap over traffic events confirms `imgsz` scaling
beats fixing it (+0.31 recall, significant) while **1080p buys nothing over
720p** (difference not distinguishable).

**B - Frame rate.** Low fps degrades the *tracker*, not the detector. Raw
detections/frame are flat from 60 to 2 fps (0.83-0.92); tracked output falls
from 0.64 to 0.33 because ByteTrack withholds boxes whose identity it cannot
confirm - 77% of detections survive tracking at 60 fps, only 36% at 2 fps.

**C - Road-band crop.** A tempting idea - crop to the ~310 px road band and run
rectangular native-size inference to keep small objects large - is **refuted**.
Small-object recall does rise (0.681 to 0.750), but precision collapses (0.751
to 0.516) and large-object recall is wrecked (0.857 to 0.643); net F1 is worse
at every band. The precision loss tracks the crop's aspect ratio, an artefact of
feeding YOLO a 6:1 strip it was never trained on.

**D - Tracker.** The frame-rate penalty is fixable, and the fix is cheap.
Loosening ByteTrack's association gate alone recovers 0.333 to **0.500** at 2 fps
(+50%). Global motion compensation (BoT-SORT + GMC) does nothing on its own
(+0.006 det/frame) and *backfires* at 2 fps, where optical-flow ego-motion
estimation breaks down over large frame gaps. On the GPU the loosened-gate
ByteTrack also runs fastest (~20 ms/frame, ~50 fps) - it wins on quality at the
hard end, simplicity, and speed at once.

![resolution recall by object size](fig_resolution2.png)
![frame rate](fig_fps2.png)
![trackers](fig_trackers.png)

## Run it

```bash
# resolution sweep  (Experiment A)
python exp_resolution2.py && python analyze_resolution.py

# frame-rate sweep  (Experiment B) + its raw-detector control
python exp_fps2.py && python exp_fps_control.py && python analyze_fps.py results_fps2.pkl

# road-band crop    (Experiment C)
python exp_roiband2.py

# tracker sweep     (Experiment D)
python exp_trackers.py --out results_trackers.pkl && python analyze_trackers.py
```

Each `exp_*.py` writes a `results_*.pkl`; each `analyze_*.py` scores it and
regenerates the figures. `make_report.py` rebuilds `FINDINGS.md`'s numbers and
`make_visual.py` the qualitative panel.

### Setup

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

A CUDA GPU is strongly recommended - the tracker sweep is ~4 minutes on an
RTX 3060 versus ~1 hour on CPU, and the timing numbers in Experiment D assume a
dedicated GPU. Install the matching PyTorch build, e.g. for CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The COCO YOLO11 weights (`yolo11{n,s,m,x}.pt`) download on first use; the driving
clip is not committed (it is large and third-party).

## What's here

| File | Role |
|------|------|
| `cvlib.py` | shared scoring helpers - class maps, greedy IoU matching, size bins |
| `exp_resolution2.py`, `analyze_resolution.py` | Experiment A: source res x `imgsz` sweep |
| `exp_fps2.py`, `exp_fps_control.py`, `analyze_fps.py` | Experiment B: frame rate, and the fps-invariant detector control |
| `exp_roiband2.py` | Experiment C: road-band crop vs full frame |
| `exp_trackers.py`, `analyze_trackers.py`, `trackers/*.yaml` | Experiment D: ByteTrack / BoT-SORT + GMC, tuned gates |
| `bootstrap_ci.py` | cluster bootstrap over traffic events (correlated-sample CIs) |
| `classic_signs.py`, `prescreen_traffic.py` | classical HSV sign detector (a documented negative result) and the event pre-screen |
| `make_report.py`, `make_visual.py` | regenerate the findings numbers and the visual panel |

## Honest limits

- **Pseudo-ground-truth, not ground truth.** Quality is agreement with a strong
  detector, so blind spots it shares are invisible. Right for ranking configs,
  wrong for quoting absolute mAP.
- **The clip is an imperfect instrument.** Cabin POV wastes ~90% of the pixels,
  the camera drifts, traffic is sparse (94% of frames have no road object), and
  the most interesting objects - European road signs - are not COCO classes, so
  stock weights cannot detect them at any resolution. The pipeline transfers
  unchanged to a road-facing clip with steady traffic, which would answer the
  question far better.
- **Effective sample is ~17 traffic events / 32 tracks**, not the raw box count -
  enough for the large effects reported, not for fine-grained rates.

See [`FINDINGS.md`](FINDINGS.md) for the per-experiment tables, the bootstrap
significance tests, and the reasoning behind every choice above.
