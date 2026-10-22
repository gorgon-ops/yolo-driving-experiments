"""Score Experiment D against B: does a better tracker undo the frame-rate penalty?

bytetrack_tuned loosens the gate (match_thresh 0.95, IoU > 0.05); botsort_gmc adds
global motion compensation; botsort_gmc_tuned does both. Timing here is clean (GPU,
no CPU contention), unlike B's infer_ms.
"""
import pickle
import numpy as np
from analyze_fps import road_only, track_stats, assoc_difficulty


def raw_floor(path="results_fps_control.pkl"):
    rows = pickle.load(open(path, "rb"))["rows"]
    return {int(fps): m for fps, _, m in rows}


def main(path="results_trackers.pkl"):
    d = pickle.load(open(path, "rb"))
    arms = d["arms"]
    band = 400.0 * d["work_h"] / 1080.0          # same road band exp_trackers scored on
    raw = raw_floor()
    fps_order = [60, 30, 20, 15, 10, 5, 2]
    arm_order = ["bytetrack", "bytetrack_tuned", "botsort_gmc", "botsort_gmc_tuned"]

    # --- tracked det/frame: how many detections survive tracking ---
    print(f"model {d['model']}  imgsz {d['imgsz']}  {d['work_h']}p  road band y<{band:.0f}\n")
    print("TRACKED DET/FRAME (higher = tracker withholds fewer boxes)")
    print(f"{'fps':>4} {'raw':>7} " + " ".join(f"{a:>18}" for a in arm_order))
    detpf = {a: {} for a in arm_order}
    for fps in fps_order:
        cells = []
        for a in arm_order:
            c = arms[a][fps]
            v = c["road_dets"] / c["n_frames"]
            detpf[a][fps] = v
            cells.append(f"{v:>18.3f}")
        print(f"{fps:>4} {raw[fps]:>7.3f} " + " ".join(cells))

    # --- retained fraction of the raw floor: the tracker's own loss ---
    print("\nRETAINED vs raw floor (tracked / raw) - the tracker's contribution alone")
    print(f"{'fps':>4} " + " ".join(f"{a:>18}" for a in arm_order))
    for fps in fps_order:
        cells = [f"{100*detpf[a][fps]/raw[fps]:>17.1f}%" for a in arm_order]
        print(f"{fps:>4} " + " ".join(cells))

    # --- track continuity + timing (trustworthy on GPU) ---
    print("\nTRACK CONTINUITY & TIMING (GPU, no contention)")
    for a in arm_order:
        print(f"\n  {a}")
        print(f"    {'fps':>4} {'det/f':>7} {'n_trk':>6} {'assocIoU':>9} {'ms/frame':>9} {'rt_fps':>7}")
        for fps in fps_order:
            c = arms[a][fps]
            fr = road_only(c["frames"], band=band)
            ts = track_stats(fr, c["eff_fps"])
            ai = assoc_difficulty(fr)
            ms = c["infer_ms"]
            print(f"    {fps:>4} {c['road_dets']/c['n_frames']:>7.3f} {ts['n_tracks']:>6d} "
                  f"{ai:>9.3f} {ms:>9.1f} {1000/ms:>7.1f}")

    # --- verdict lines ---
    print("\nVERDICT")
    bt2, btt2 = detpf["bytetrack"][2], detpf["bytetrack_tuned"][2]
    gmc2, gmct2 = detpf["botsort_gmc"][2], detpf["botsort_gmc_tuned"][2]
    print(f"  at 2 fps: bytetrack {bt2:.3f} -> tuned {btt2:.3f} (+{100*(btt2-bt2)/bt2:.0f}%); "
          f"gmc {gmc2:.3f} (=bytetrack); gmc_tuned {gmct2:.3f}")
    print(f"  gmc alone vs bytetrack, mean over fps: "
          f"{np.mean([detpf['botsort_gmc'][f]-detpf['bytetrack'][f] for f in fps_order]):+.3f} det/frame")
    print(f"  gate loosening (tuned) vs default, mean over fps: "
          f"{np.mean([detpf['bytetrack_tuned'][f]-detpf['bytetrack'][f] for f in fps_order]):+.3f} det/frame")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results_trackers.pkl")
