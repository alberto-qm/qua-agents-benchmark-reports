"""The decay tilt from up- and down-chirps on the 24 Sep hyperbolic-secant test data (linear pulses only, no QPU).

Inside a box, ln(p_up / p_down) = (2 rho / B)(f01 - c) with rho = sweep / T1: the transfer and the edge rounding are
the same for both directions and cancel. Fitted per drive level over the points where both populations exceed 0.3.
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "hs_test"
B = 20e6
for be, qs in (("arbel", ["qB4", "qA6"]), ("gilboa", ["qD5", "qC3", "qD2"])):
    meta = json.loads((D / f"{be}_meta.json").read_text())
    for q in qs:
        plan = meta["plan"][q]
        d = np.load(D / f"{be}_{q}_ladder.npz")
        r0, r1 = np.array([d["r0_I"], d["r0_Q"]]), np.array([d["r1_I"], d["r1_Q"]])
        ax = r1 - r0
        pop = lambda k: ((d[f"{k}_I"] - r0[0]) * ax[0] + (d[f"{k}_Q"] - r0[1]) * ax[1]) / (ax @ ax)
        c = np.asarray(plan["centres"], float)
        tau = plan["tau"]
        out = []
        for lv in range(5):
            pu, pd = pop(f"lin_up_{lv}"), pop(f"lin_dn_{lv}")
            near = np.abs(c - 0) < 60e6   # centres are relative to the stored f01; lines lie within a few MHz
            m = near & (pu > 0.3) & (pd > 0.3)
            if m.sum() < 3 or max(pu[near].max(), pd[near].max()) < 0.5:
                out.append(f"L{lv}: –")
                continue
            y = np.log(pu[m] / pd[m])
            A = np.vstack([c[m] - c[m].mean(), np.ones(m.sum())]).T
            (s, a), res, *_ = np.linalg.lstsq(A, y, rcond=None)
            rho = -s * B / 2
            zero = c[m].mean() - a / s if s != 0 else np.nan
            out.append(f"L{lv}: rho {rho:+.2f} (T1 {tau / rho / 1e3 if rho > 0 else float('nan'):.1f} us, n {m.sum()}, zero {zero / 1e6:+.1f} MHz)")
        print(f"{be} {q}: sweep {tau} ns, stored T1 {plan['T1'] * 1e6:.2f} us (expected rho {tau / (plan['T1'] * 1e9):.2f})")
        for o in out:
            print("    ", o)
