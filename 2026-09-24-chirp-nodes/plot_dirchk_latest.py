"""Up/down-chirp check: T1 missing from the state against T1 known, qubits as columns (no QPU).

figures/dirchk_<backend>.png    row 1: 03a chirp without T1, row 2: with T1; row 3: 03b chirp without T1, row 4: with T1.
Both directions averaged, as the nodes fit them; grey marks what a node does not propose.
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
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import project_iq  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.chirp import mean_over_directions  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RED, GREY = "#e6194b", "#bdbdbd"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})
QUBITS = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
LABEL = {"noT1": "T1 unknown", "T1": "T1 known"}


def find(be, node, factor, q):
    for f in sorted((HERE / "dirchk_latest" / be).glob(f"{node}-{factor}-*.json")):
        if f.name.endswith(".reanalysed.json"):
            continue
        rec = json.loads(f.read_text())
        re_ = f.with_name(f.stem + ".reanalysed.json")
        if re_.exists():  # the committed analysis (ee24095) on the same data
            rec["fit_results"] = json.loads(re_.read_text())
        if q in rec["qubits"]:
            return rec, mean_over_directions(xr.open_dataset(f.with_suffix(".nc")))
    return None, None


def norm(s):
    return s / (np.nanpercentile(s, 99.5) or 1.0)


def panel_03a(ax, be, q, factor):
    rec, ds = find(be, "03a", factor, q)
    if rec is None:
        ax.set_axis_off()
        return
    dq = ds.sel(qubit=q).transpose("drive_level", "detuning")
    x = ds.detuning.values / 1e6
    s = norm(project_iq(dq.I.values, dq.Q.values))
    amps = np.asarray(dq.drive_amplitude.values, float)
    rel = amps / amps[len(amps) // 2]
    ax.pcolormesh(x, np.arange(len(rel)), s, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
    ax.set_yticks(np.arange(len(rel)))
    ax.set_yticklabels([f"×{r:.3g}" for r in rel], fontsize=7)
    r = rec["fit_results"][q]
    ok = r.get("line_identity") == "0-1" and r.get("frequency_shift") == r.get("frequency_shift")
    rho = r.get("sweep_over_t1")
    shift = r.get("frequency_shift")
    if shift is not None and shift == shift:
        ax.axvline(shift / 1e6, color=RED if ok else GREY, lw=1.0, ls="-" if ok else "--")
    txt = (f"f₀₁ {shift / 1e6:+.2f} ± {r['f_01_error'] / 1e6:.2f} MHz" if ok else f"not proposed: {r.get('line_identity')}")
    if rho is not None and rho == rho:
        txt += f"\nsweep/T1 {rho:.2f} ± {r.get('sweep_over_t1_error'):.2f}"
    ax.text(0.02, 0.04, txt + f"\n{rec['qpu_s']:.1f} s QPU (job, {len(rec['qubits'])} qubits)", transform=ax.transAxes,
            color="w", fontsize=7.5)
    wait_us = rec["waits_ns"][q] / 1e3
    tau = rec["namespace"]["sweep_lengths"][q]
    ax.set_title(f"{q} · 03a, {LABEL[factor]}: {tau / 1e3:.2g} µs sweep, {wait_us:.0f} µs wait")
    ax.set_ylabel("drive / x180 prediction")
    ax.set_xlabel("band centre − stored f₀₁ [MHz]")


def panel_03b(ax, be, q, factor):
    rec, ds = find(be, "03b", factor, q)
    if rec is None:
        ax.set_axis_off()
        return
    dq = ds.sel(qubit=q).transpose("detuning", "flux_index")
    det, phi = ds.detuning.values, dq.flux_bias.values
    res = analyse_flux_map(det, phi, dq.I.values, dq.Q.values, 20e6, "chirp")
    s = norm(project_iq(dq.I.values, dq.Q.values))
    ax.pcolormesh(phi * 1e3, det / 1e6, s, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
    ax.plot(phi * 1e3, res["centres"] / 1e6, "o", ms=3.5, mfc="none", mec=RED, mew=1.0)
    r = rec["fit_results"][q]
    proposed = r.get("idle_offset_shift") is not None and r.get("idle_offset_shift") == r.get("idle_offset_shift") \
        and not r.get("short_sweep") and not r.get("decay_refused")
    if np.isfinite(res["x0"]):
        xx = np.linspace(phi.min(), phi.max(), 200)
        col = RED if proposed else GREY
        ax.plot(xx * 1e3, (res["f0"] + res["curvature"] * (xx - res["x0"]) ** 2) / 1e6, color=col, lw=1.1,
                ls="-" if proposed else "--")
        ax.plot([res["x0"] * 1e3], [res["f0"] / 1e6], marker="*", ms=11, color=col, mec="k")
        ax.set_ylim(det.min() / 1e6, det.max() / 1e6)
    if proposed:
        txt = f"sweet spot {r['idle_offset_shift'] * 1e3:+.2f} ± {r['idle_offset_shift_error'] * 1e3:.2f} mV"
    elif r.get("short_sweep"):
        txt = "not proposed: sweep too short"
    elif r.get("decay_refused"):
        txt = f"not proposed: decays in the sweep (sweep/T1 {r.get('sweep_over_t1'):.2f})"
    else:
        txt = f"no arc: {r.get('tracked_columns')}/{r.get('flux_columns')} columns"
    ax.text(0.02, 0.04, txt + f"\n{rec['qpu_s']:.1f} s QPU (job, {len(rec['qubits'])} qubits)", transform=ax.transAxes,
            color="w", fontsize=7.5)
    ax.set_title(f"{q} · 03b, {LABEL[factor]}: {rec['waits_ns'][q] / 1e3:.0f} µs wait")
    ax.set_xlabel("flux offset from idle [mV]")
    ax.set_ylabel("band centre − stored f₀₁ [MHz]")


def main():
    for be, qs in QUBITS.items():
        fig, axes = plt.subplots(4, len(qs), figsize=(5.0 * len(qs), 15.0), squeeze=False)
        for c, q in enumerate(qs):
            panel_03a(axes[0, c], be, q, "noT1")
            panel_03a(axes[1, c], be, q, "T1")
            panel_03b(axes[2, c], be, q, "noT1")
            panel_03b(axes[3, c], be, q, "T1")
        fig.suptitle(f"{be}: latest state (25 Sep): up- and down-chirps, T1 missing (rows 1, 3) and known (rows 2, 4)",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(FIG / f"dirchk_latest_{be}.png", dpi=90)
        plt.close(fig)
        print(be, "ok")


if __name__ == "__main__":
    main()
