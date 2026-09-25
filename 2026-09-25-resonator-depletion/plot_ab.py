"""02b maps under the four depletion conditions, per backend (no QPU).

figures/ab_<backend>_r<round>.png   rows A B C D, one column per qubit: |S21| normalised per power row against readout
                                    frequency and power, the proposed power (line) and frequency (tick), the lowest-10-row
                                    block marked; title carries the result and the job's QPU time
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
LABEL = {"A": "A · today: 12 µs, power inner", "B": "B · 3 µs, power inner (unit fix only)",
         "C": "C · 3 µs, power outer + reset per round", "D": "D · 1 µs, power outer + reset per round"}


def main(rnd):
    for be, qs in QUBITS.items():
        if not any((HERE / "ab" / be).glob(f"*-r{rnd}-*.nc")):
            continue
        fig, axes = plt.subplots(4, len(qs), figsize=(5.0 * len(qs), 15.5), squeeze=False)
        for c, q in enumerate(qs):
            for r, cond in enumerate("ABCD"):
                ax = axes[r, c]
                f = HERE / "ab" / be / f"{cond}-r{rnd}-{q}.json"
                if not f.exists() or not f.with_suffix(".nc").exists():
                    ax.set_axis_off()
                    continue
                rec = json.loads(f.read_text())
                ds = xr.open_dataset(f.with_suffix(".nc"))
                a = ds["IQ_abs"].sel(qubit=q).transpose("power", "detuning").values
                a = a / np.median(a, axis=1, keepdims=True)
                det = ds.detuning.values / 1e6
                pw = ds.power.values
                ax.pcolormesh(det, pw, a, shading="nearest", cmap="magma", vmin=np.percentile(a, 1), vmax=1.05)
                ax.axhspan(pw[0], pw[9], color="#00bcd4", alpha=0.12, lw=0)
                fr = (rec.get("fit_results") or {}).get(q) or {}
                op, fs = fr.get("optimal_power"), fr.get("frequency_shift")
                ok = bool(fr.get("success"))
                if op is not None and np.isfinite(op):
                    ax.axhline(op, color="#00e676" if ok else "#9e9e9e", lw=1.4)
                if fs is not None and np.isfinite(fs) and op is not None and np.isfinite(op):
                    ax.plot([fs / 1e6], [op], marker="v", ms=9, color="#00e676" if ok else "#9e9e9e", mec="k")
                txt = (f"{op:.1f} dBm, {fs / 1e6:+.2f} MHz" if ok and op is not None else
                       f"no proposal ({fr.get('onset_mechanism')})")
                ax.set_title(f"{q} · {LABEL[cond]}\n{txt} · {rec['qpu_s']:.1f} s QPU", fontsize=8.5, loc="left")
                ax.set_xlabel("readout frequency − stored [MHz]", fontsize=8)
                ax.set_ylabel("readout power [dBm]", fontsize=8)
        fig.suptitle(f"{be}: 02b resonator spectroscopy vs power, round {rnd} (shaded: the lowest 10 power rows)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(FIG / f"ab_{be}_r{rnd}.png", dpi=85)
        plt.close(fig)
        print(be, "ok")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
