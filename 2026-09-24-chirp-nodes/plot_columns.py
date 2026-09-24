"""Chirp against saturation per backend, qubits as columns (no QPU).

figures/cols_1d_<backend>.png    row 1: chirp response against frequency and drive (the wide ladder, 11 drives)
                                 row 2: its line cuts, one trace per drive
                                 row 3: saturation response against frequency and drive (the ladder at the node's
                                        defaults, 9 drives)
                                 row 4: its line cuts
figures/cols_flux_<backend>.png  row 1: chirped flux map at the node defaults (03b chirp), row 2: saturation map (03b
                                 saturation mode); column centres and the fitted parabola from the committed analysis
Shared frequency axis: offset from the stored f01, +-130 MHz (the saturation ladder's range).
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
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import project_iq  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.chirp import short_sweep_warning  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RED, ORANGE = "#e6194b", "#ff9f40"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
XLIM = (-130, 130)


def norm(rows):
    rows = np.asarray(rows, float)
    return rows / (np.nanpercentile(rows, 99.5) or 1.0)


def lines(ax, x, rows, labels):
    k = len(rows)
    for j in range(k):
        ax.plot(x, rows[j] * 0.9 + j, lw=0.9, color=plt.cm.viridis(j / max(1, k - 1)))
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.4, k + 0.2)
    ax.set_xlim(*XLIM)


def markers(ax, ref, alpha):
    if ref is not None:
        ax.axvline(ref, color=RED, lw=0.8, ls=":")
        if alpha:
            ax.axvline(ref - alpha / 2, color=ORANGE, lw=0.8, ls=":")


def one_d(be, rep_c, rep_s, state):
    qs = QUBITS[be]
    fig, axes = plt.subplots(4, len(qs), figsize=(5.0 * len(qs), 13.5),
                             gridspec_kw=dict(height_ratios=(1, 1.35, 1, 1.25)), squeeze=False)
    for c, q in enumerate(qs):
        stored = state["qubits"][q]["f_01"]
        ref_abs = (rep_s.get(f"{be}/{q}") or {}).get("reference")
        ref = (ref_abs - stored) / 1e6 if ref_abs else None
        alpha = (rep_c.get(f"{be}/{q}") or {}).get("03a", {}).get("alpha_full")
        alpha = alpha / 1e6 if alpha else None
        # chirp: wide ladder
        ds = None
        for f in sorted((HERE / be).glob("wide3a-*.nc")):
            d = xr.open_dataset(f)
            if q in list(d.qubit.values):
                ds, rec = d, json.loads(f.with_suffix(".json").read_text())
                break
        if ds is not None:
            dq = ds.sel(qubit=q)
            x = ds.detuning.values / 1e6
            sel = (x >= XLIM[0]) & (x <= XLIM[1])
            s = norm(project_iq(dq.I.values, dq.Q.values))[:, sel]
            amps = np.asarray(dq.drive_amplitude.values, float)
            rel = amps / amps[5]
            kappa = rec["namespace"]["kappa_priors"][q]
            tau = rec["namespace"]["sweep_lengths"][q]
            labels = [f"×{r:.3g}" for r in rel]
            a = axes[0, c]
            a.pcolormesh(x[sel], np.arange(len(rel)), s, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
            a.set_yticks(np.arange(len(rel))[::2])
            a.set_yticklabels(labels[::2], fontsize=7)
            a.set_xlim(*XLIM)
            short = " — sweep too short, node refuses" if short_sweep_warning(20e6, tau) else ""
            a.set_title(f"{q} · chirp, {tau} ns over 20 MHz{short}")
            a.set_ylabel("drive / x180 prediction")
            markers(a, ref, alpha)
            b = axes[1, c]
            lines(b, x[sel], s, [f"{r * kappa * amps[5] / 1e6:.2g} MHz" for r in rel])
            b.set_ylabel("chirp line cuts (Rabi)")
            markers(b, ref, alpha)
        # saturation ladder at the node defaults
        files = sorted((HERE / "satladder_default" / be).glob(f"satdef-x*-{q}.nc"),
                       key=lambda p: float(p.stem.split("-x")[1].split("-")[0]))
        if files:
            rows, gs = [], []
            for f in files:
                d = xr.open_dataset(f).sel(qubit=q)
                rows.append(project_iq(d.I.values[None, :], d.Q.values[None, :])[0])
                gs.append(float(f.stem.split("-x")[1].split("-")[0]))
                xs = d.detuning.values / 1e6
            rows = norm(rows)
            ops = state["qubits"][q]["xy"]["operations"]
            x180 = ops[ops["x180"].split("/")[-1]] if isinstance(ops["x180"], str) else ops["x180"]
            sat = ops[ops["saturation"].split("/")[-1]] if isinstance(ops["saturation"], str) else ops["saturation"]
            kappa_s = 1.0 / (x180["amplitude"] * x180["length"] * 1e-9)
            r1 = 0.5 * sat["amplitude"] * kappa_s / 1e6
            a = axes[2, c]
            a.pcolormesh(xs, np.arange(len(gs)), rows, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
            a.set_yticks(np.arange(len(gs)))
            a.set_yticklabels([f"×{g:g}" for g in gs], fontsize=7)
            a.set_xlim(*XLIM)
            a.set_title(f"{q} · saturation, 20 µs, ×1 = {r1:.1f} MHz Rabi")
            a.set_ylabel("drive / node default")
            markers(a, ref, alpha)
            b = axes[3, c]
            lines(b, xs, rows, [f"{g * r1:.2g} MHz" for g in gs])
            b.set_ylabel("saturation line cuts (Rabi)")
            b.set_xlabel("frequency − stored f01 [MHz]")
            markers(b, ref, alpha)
        for r in range(3):
            axes[r, c].set_xlabel("")
    fig.suptitle(f"{be}: qubit spectroscopy with a chirp (rows 1–2) and with saturation (rows 3–4). "
                 "Red dotted: f01 (fine saturation scan); orange dotted: the 0→2 line, f01 − α/2",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIG / f"cols_1d_{be}.png", dpi=90)
    plt.close(fig)


def flux(be, state):
    qs = QUBITS[be]
    fig, axes = plt.subplots(2, len(qs), figsize=(5.0 * len(qs), 8.4), squeeze=False)
    for c, q in enumerate(qs):
        stored = state["qubits"][q]["f_01"]
        for r, (prefix, pulse, band) in enumerate((("r0-03b-", "chirp", 20e6), ("sat-", "saturation", 20e6))):
            a = axes[r, c]
            found = None
            for f in sorted((HERE / be).glob(f"{prefix}*.nc")):
                d = xr.open_dataset(f)
                if q in list(d.qubit.values):
                    found = (d, json.loads(f.with_suffix(".json").read_text()))
                    break
            if found is None:
                a.set_axis_off()
                continue
            d, rec = found
            dq = d.sel(qubit=q)
            det = d.detuning.values
            phi = dq.flux_bias.values
            res = analyse_flux_map(det, phi, dq.I.values, dq.Q.values, band, pulse)
            s = norm(project_iq(dq.I.values, dq.Q.values))
            a.pcolormesh(phi * 1e3, det / 1e6, s, shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
            a.plot(phi * 1e3, res["centres"] / 1e6, "o", ms=3.5, mfc="none", mec=RED, mew=1.0)
            tau = (rec.get("namespace") or {}).get("sweep_lengths", {}).get(q)
            refused = pulse == "chirp" and tau and short_sweep_warning(band, tau)
            if np.isfinite(res["x0"]):
                xx = np.linspace(phi.min(), phi.max(), 200)
                col = "#bdbdbd" if refused else RED
                a.plot(xx * 1e3, (res["f0"] + res["curvature"] * (xx - res["x0"]) ** 2) / 1e6, color=col, lw=1.1,
                       ls="--" if refused else "-")
                a.plot([res["x0"] * 1e3], [res["f0"] / 1e6], marker="*", ms=11, color=col, mec="k")
                a.set_ylim(det.min() / 1e6, det.max() / 1e6)
                txt = ("not proposed: sweep too short" if refused else
                       f"sweet spot {res['x0'] * 1e3:+.2f} ± {res['x0_error'] * 1e3:.2f} mV")
            else:
                txt = f"no arc: {res['tracked']}/{res['columns']} columns"
            a.text(0.02, 0.03, txt + f"\n{rec['qpu_s']:.0f} s QPU" + (" (job)" if len(rec["qubits"]) > 1 else ""),
                   transform=a.transAxes, color="w", fontsize=7.5)
            a.set_title(f"{q} · {'chirp (03b chirp at its defaults)' if pulse == 'chirp' else 'saturation (03b saturation mode)'}")
            a.set_xlabel("flux offset from idle [mV]")
            a.set_ylabel("frequency − stored f01 [MHz]")
    fig.suptitle(f"{be}: flux maps with a chirp (row 1) and with saturation (row 2)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / f"cols_flux_{be}.png", dpi=90)
    plt.close(fig)


def main():
    rep_c = json.loads((HERE / "replay/replays.json").read_text())
    rep_s = json.loads((HERE / "replay/replays_sat.json").read_text())
    for be in QUBITS:
        state = json.loads((HERE / be / "state/state.json").read_text())
        one_d(be, rep_c, rep_s, state)
        flux(be, state)
        print(be, "ok")


if __name__ == "__main__":
    main()
