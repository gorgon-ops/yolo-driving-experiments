"""Cluster bootstrap over traffic events: boxes in one event are the same vehicles again,
so resampling whole events keeps the dependence naive binomial errors ignore.
"""
import numpy as np, pandas as pd
from analyze_resolution import score, DRIVE_START

df, tim, raw = score("results_resolution2.pkl")
ref = df[df.kind == "ref"].copy()
rng = np.random.default_rng(0)
events = np.sort(ref.event.unique())
B = 4000


def boot(sel):
    g = ref[sel]
    by = {e: v.hit.values for e, v in g.groupby("event")}
    ks = list(by)
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(ks, size=len(ks), replace=True)
        out[b] = np.concatenate([by[k] for k in pick]).mean()
    return g.hit.mean(), np.percentile(out, [2.5, 97.5]), out


def cmp(la, sa, lb, sb):
    ma, ca, da = boot(sa)
    mb, cb, db = boot(sb)
    diff = da - db
    lo, hi = np.percentile(diff, [2.5, 97.5])
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
    print(f"  {la:28s} {ma:.3f} [{ca[0]:.3f},{ca[1]:.3f}]")
    print(f"  {lb:28s} {mb:.3f} [{cb[0]:.3f},{cb[1]:.3f}]")
    print(f"     diff {ma-mb:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  -> {sig}\n")


N = lambda r: (ref.pipeline == "native") & (ref.res == r)
F = lambda r: (ref.pipeline == "fixed640") & (ref.res == r)
print(f"cluster bootstrap over {len(events)} traffic events, B={B}\n")
print("Does extra source resolution help when imgsz is FIXED at 640?")
cmp("fixed640 @ 1080p", F("1080p"), "fixed640 @ 480p", F("480p"))
print("Does 1080p beat 720p when imgsz scales?")
cmp("native @ 1080p", N("1080p"), "native @ 720p", N("720p"))
print("Does 720p beat 480p when imgsz scales?")
cmp("native @ 720p", N("720p"), "native @ 480p", N("480p"))
print("Does scaling imgsz beat fixing it, at the same 1080p source?")
cmp("native @ 1080p", N("1080p"), "fixed640 @ 1080p", F("1080p"))
print("TINY objects only (<24px): native 1080p vs fixed640 1080p")
T = ref["bin"] == 0
cmp("native @1080p, tiny", N("1080p") & T, "fixed640 @1080p, tiny", F("1080p") & T)
