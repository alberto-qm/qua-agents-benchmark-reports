"""qolab Q1 without T1, 2 us against 4 us sweeps (q1_sweeps/), with the 13:33 T1-known run as the scale (no QPU).

Top: 03a maps. Bottom: the signal against band centre averaged over drive levels 2-4, on the ground->excited scale of the
T1-known run (0 = its baseline, 1 = its full box): the baseline shows how excited Q1 already is before the chirp.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.chirp import mean_over_directions  # noqa: E402

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "font.size": 8.5,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": "#e4e3df"})


def z_of(path):
    ds = mean_over_directions(xr.open_dataset(path)).sel(qubit="Q1").transpose("drive_level", "detuning")
    return ds.detuning.values / 1e6, ds.I.values + 1j * ds.Q.values


x, zr = z_of(HERE / "dirchk_latest/qolab/03a-T1-Q1-Q2-Q5.nc")
box, far = np.abs(x) < 6, np.abs(x) > 40
g, e = np.median(zr[:, far]), np.median(zr[2:5][:, box])
u = (e - g) / abs(e - g)
scale = lambda z: ((z - g) * np.conj(u)).real / abs(e - g)

runs = [("03a-Q1-2000ns-r1", "2 µs sweep, run 1"), ("03a-Q1-4000ns-r1", "4 µs sweep, run 1"),
        ("03a-Q1-2000ns-r2", "2 µs sweep, run 2"), ("03a-Q1-4000ns-r2", "4 µs sweep, run 2")]
fig, axes = plt.subplots(2, 5, figsize=(22, 8.2), gridspec_kw=dict(height_ratios=(1.25, 1)))
colors = {"2 µs": "#1f6fb2", "4 µs": "#d9822b"}
items = [(HERE / "q1_sweeps" / f"{t}.nc", lab, json.loads((HERE / "q1_sweeps" / f"{t}.json").read_text())["fit_results"]["Q1"])
         for t, lab in runs]
items.append((HERE / "dirchk_latest/qolab/03a-T1-Q1-Q2-Q5.nc", "T1 known (13:33): 4 µs sweep, 352 µs wait",
              json.loads((HERE / "dirchk_latest/qolab/03a-T1-Q1-Q2-Q5.json").read_text())["fit_results"]["Q1"]))
for c, (path, label, fit) in enumerate(items):
    xs, z = z_of(path)
    s = scale(z)
    ax = axes[0, c]
    ax.pcolormesh(xs, np.arange(s.shape[0]), s, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels([f"×{2.0 ** (k - 3):g}" for k in range(7)])
    ok = fit.get("line_identity") == "0-1" and fit.get("frequency_shift") == fit.get("frequency_shift")
    shift = fit.get("frequency_shift")
    if shift is not None and shift == shift:
        ax.axvline(shift / 1e6, color="#e6194b" if ok else "#bdbdbd", lw=1.1, ls="-" if ok else "--")
    res = (f"f₀₁ {shift / 1e6:+.2f} ± {fit['f_01_error'] / 1e6:.2f} MHz" if ok else f"not proposed: {fit.get('line_identity')}")
    base, top = np.median(s[:, np.abs(xs) > 40]), np.median(s[2:5][:, np.abs(xs) < 6])
    ax.text(0.02, 0.03, f"{res}\nbaseline {base:+.2f}, box top {top:.2f}", transform=ax.transAxes, color="w", fontsize=8)
    ax.set_title(("Q1 · " if c == 0 else "") + label + ("" if "T1 known" in label else ", 50 µs wait"))
    ax.set_xlabel("band centre − stored f₀₁ [MHz]")
    if c == 0:
        ax.set_ylabel("drive / x180 prediction")
    b = axes[1, c]
    b.axhline(0, color="#999", lw=0.7)
    b.axhline(1, color="#999", lw=0.7, ls=":")
    b.plot(xs, s[2:5].mean(axis=0), marker="o", ms=3, lw=1.2,
           color=colors.get(label[:4], "#333"))
    b.set_ylim(-0.3, 1.3)
    b.set_xlim(-60, 60)
    b.set_xlabel("band centre − stored f₀₁ [MHz]")
    if c == 0:
        b.set_ylabel("drive levels ×0.5–×2, 0 = ground, 1 = full box")
fig.suptitle("qolab Q1, 03a chirp with T1 missing from the state (50 µs wait): 2 µs against 4 µs sweeps, alternated twice "
             "(25 Sep 13:50, state pulled 13:49); right: the same qubit with T1 known", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "figures" / "q1_sweeps_2v4.png"
fig.savefig(out, dpi=100)
print(out)
