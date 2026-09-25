"""Up- against down-chirp box centres when the excitation decays during the sweep (no QPU).

Response at band centre c: the node's soft-edged box times exp(-(tau - t_cross)/T1), where the crossing time
t_cross is tau (f - (c - B/2)) / B for an up-chirp and tau ((c + B/2) - f) / B for a down-chirp. Fitted with the
node's find_boxes on 5 MHz steps, 40 random line positions; prints the split (up - down), the mean of the two
and the box fitted to their average, all as bias from the true line.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import box_model, find_boxes  # noqa: E402

B, EDGE, NOISE = 20e6, 1.0e6, 0.02
x = np.arange(-120e6, 120e6 + 1, 5e6)


def response(f, ratio, up):
    inside = box_model(x, f, 1.0, B, EDGE, 0.0)
    frac = np.clip(((f - (x - B / 2)) if up else ((x + B / 2) - f)) / B, 0, 1)   # crossing time / tau
    return inside * np.exp(-ratio * (1 - frac))


def centre(y, rng):
    bs = find_boxes(x, y + rng.normal(0, NOISE, y.size), NOISE, B)
    return max(bs, key=lambda b: b.snr).centre if bs else np.nan


print("tau/T1   split up-down [MHz]   up bias   down bias   mean of the two   fit to the average   height of the average")
for ratio in (0.05, 0.1, 0.2, 0.3, 0.5, 0.77, 1.0, 1.5, 3.0):
    rows = []
    for k in range(40):
        rng = np.random.default_rng(k)
        f = rng.uniform(-30e6, 30e6)
        yu, yd = response(f, ratio, True), response(f, ratio, False)
        cu, cd, ca = centre(yu, rng), centre(yd, rng), centre((yu + yd) / 2, rng)
        rows.append(((cu - cd), cu - f, cd - f, (cu + cd) / 2 - f, ca - f, ((yu + yd) / 2).max()))
    r = np.nanmean(np.array(rows), axis=0)
    s = np.nanstd(np.array(rows), axis=0)
    print(f"{ratio:5.2f}   {r[0] / 1e6:+6.2f} ± {s[0] / 1e6:4.2f}   {r[1] / 1e6:+6.2f}   {r[2] / 1e6:+6.2f}   "
          f"{r[3] / 1e6:+6.2f} ± {s[3] / 1e6:4.2f}   {r[4] / 1e6:+6.2f} ± {s[4] / 1e6:4.2f}   {r[5]:.2f}")
