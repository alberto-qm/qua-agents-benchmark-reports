"""Figure for the 03b timing fix (no QPU): fix_03b.png.

A/B: qolab Q1, node 03b as it is on feat/qualibrate-ai (drive and flux step together, 80 us) and
with the fix (// 4 restored, 80 us explicit, flux step 5 us before and after the drive).
C/D: gilboa qD5 with the fix at a 20 us and an 80 us drive (the comparison the 20 Sep revert asked for).
Signal: IQ projected on its principal axis, per-column median removed.
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
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import project_iq  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})

PANELS = [("qolab", "map03b-Q1-main", "Q1", "A · qolab Q1, 03b as on the branch today\n(drive and flux step together, 80 µs)"),
          ("qolab", "map03b-Q1-fix", "Q1", "B · qolab Q1, fixed 03b\n(80 µs drive inside a flux step with 5 µs margins)"),
          ("gilboa", "ab20-qD5-fix", "qD5", "C · gilboa qD5, fixed 03b, 20 µs drive"),
          ("gilboa", "ab80-qD5-fix", "qD5", "D · gilboa qD5, fixed 03b, 80 µs drive")]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (be, tag, q, title) in zip(axes, PANELS):
    ds = xr.open_dataset(HERE / "fixes" / be / f"{tag}.nc").sel(qubit=q)
    rec = json.loads((HERE / "fixes" / be / f"{tag}.json").read_text())
    fit = (rec.get("fit_results") or {}).get(q) or {}
    s = project_iq(ds.I.values, ds.Q.values)
    s = s - np.median(s, axis=0, keepdims=True)
    x = ds.flux_bias.values * 1e3
    f = ds.detuning.values / 1e6
    lim = np.nanpercentile(np.abs(s), 99)
    ax.pcolormesh(x, f, s / lim, shading="nearest", cmap="viridis", vmin=-0.2, vmax=1.0)
    x0 = fit.get("idle_offset_shift")
    if x0 is not None and np.isfinite(x0):
        ax.axvline(x0 * 1e3, color="#e6194b", lw=1.2, ls=":")
        ax.text(0.03, 0.04, f"sweet spot {x0 * 1e3:+.2f} mV\nr² {fit.get('r_squared', float('nan')):.3f}, "
                f"{rec['qpu_s']:.0f} s QPU", transform=ax.transAxes, color="w", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("flux offset from idle [mV]")
    ax.set_ylabel("drive frequency - stored f01 [MHz]")
fig.tight_layout()
fig.savefig(HERE / "fix_03b.png", dpi=130)
print("ok")
