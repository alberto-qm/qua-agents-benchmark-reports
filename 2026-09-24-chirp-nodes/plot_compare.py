"""Chirp against saturation, per qubit (no QPU).

compare_capture_<backend>.png  replay outcomes: chirp 03a (left) and saturation 03a (right) over the stored-f01 error
                                and the drive error, three qubits per backend
compare_ladder_<backend>.png   the raw responses behind them: the chirp's wide ladder (11 drives, +-200 MHz) and the
                                saturation ladder (9 drives, +-130 MHz), signal against frequency, one row per drive
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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import project_iq  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OUTCOME = {"pass": ("#1baf7a", 0), "imprecise": ("#a8d8a0", 1), "refused": ("#c9c7c1", 2), "wrong": ("#d6453d", 3)}
CMAP = ListedColormap([OUTCOME[k][0] for k in ("pass", "imprecise", "refused", "wrong")])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
SHORT = {"qD2", "qC3"}


def rabi_info(backend):
    st = json.loads((HERE / backend / "state/state.json").read_text())
    out = {}
    for q in QUBITS[backend]:
        ops = st["qubits"][q]["xy"]["operations"]
        x = ops["x180"]
        x = ops[x.split("/")[-1]] if isinstance(x, str) else x
        s = ops["saturation"]
        s = ops[s.split("/")[-1]] if isinstance(s, str) else s
        kappa = 1.0 / (x["amplitude"] * x["length"] * 1e-9)
        out[q] = dict(kappa=kappa, sat_default_rabi=0.5 * s["amplitude"] * kappa)
    return out


def grid_panel(ax, cells, ykey, ylabel, note):
    ms = sorted({c[ykey] for c in cells})
    Fs = sorted({c["F"] for c in cells})
    Z = np.full((len(ms), len(Fs)), np.nan)
    for c in cells:
        Z[ms.index(c[ykey]), Fs.index(c["F"])] = OUTCOME[c["outcome"]][1]
    ax.imshow(Z, aspect="auto", origin="lower", cmap=CMAP, vmin=-0.5, vmax=3.5,
              extent=[Fs[0] / 1e6 - 2.5, Fs[-1] / 1e6 + 2.5, ms[0] - 0.5, ms[-1] + 0.5])
    ax.set_yticks(ms)
    ax.set_yticklabels([f"×{2.0 ** m:g}" if m >= 0 else f"×1/{2 ** -m:g}" for m in ms])
    ax.set_xlabel("stored f01 error [MHz]")
    ax.set_ylabel(ylabel)
    counts = {k: sum(c["outcome"] == k for c in cells) for k in ("pass", "wrong")}
    ax.text(0.02, 0.97, f"{counts['pass']}/{len(cells)} pass, {counts['wrong']} wrong" + (f"\n{note}" if note else ""),
            transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))


def capture(backend, rep_c, rep_s, info):
    qs = QUBITS[backend]
    fig, axes = plt.subplots(len(qs), 2, figsize=(11.5, 3.1 * len(qs)))
    for i, q in enumerate(qs):
        a = rep_c[f"{backend}/{q}"]["03a"]
        grid_panel(axes[i, 0], a["grid"], "m", "real drive / x180 prediction", "short sweep: refused" if q in SHORT else "")
        axes[i, 0].set_title(f"{q} · chirp 03a (window ±120 MHz)")
        s = rep_s.get(f"{backend}/{q}")
        if s and s["grid"].get("100"):
            cells = [dict(c, m=c["m"]) for c in s["grid"]["100"]]
            missing = sorted({1, 2, 4, 8, 16} - {int(round(c["g"])) for c in cells if c["g"] >= 1})
            note = f"×1 = {info[q]['sat_default_rabi'] / 1e6:.1f} MHz Rabi"
            if missing:
                note += f"; ×{', ×'.join(str(m) for m in missing)} beyond full scale"
            grid_panel(axes[i, 1], cells, "m", "real drive / node default", note)
            axes[i, 1].set_title(f"{q} · saturation 03a (window ±50 MHz, its default)")
    handles = [Patch(color=OUTCOME[k][0], label=k) for k in OUTCOME]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 0.968))
    fig.suptitle(f"{backend}: chirp (left) against saturation (right), replayed over a wrong stored f01 and a wrong drive",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / f"compare_capture_{backend}.png", dpi=100)
    plt.close(fig)


def stack(rows):
    rows = np.array(rows)
    scale = np.nanpercentile(rows, 99.5) or 1.0
    return rows / scale


def ladder(backend, rep_c, rep_s, info):
    qs = QUBITS[backend]
    fig, axes = plt.subplots(len(qs), 2, figsize=(12.5, 3.0 * len(qs)))
    for i, q in enumerate(qs):
        # chirp wide ladder
        for f in sorted((HERE / backend).glob("wide3a-*.nc")):
            ds = xr.open_dataset(f)
            if q in list(ds.qubit.values):
                break
        dq = ds.sel(qubit=q)
        x = ds.detuning.values / 1e6
        s = project_iq(dq.I.values, dq.Q.values)
        ax = axes[i, 0]
        ax.pcolormesh(x, np.arange(s.shape[0]), stack(s), shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
        amps = np.asarray(dq.drive_amplitude.values, float)
        rel = amps / amps[5]
        ax.set_yticks(np.arange(len(rel))[::2])
        ax.set_yticklabels([f"×{r:.3g}" for r in rel[::2]])
        ax.set_ylabel("drive / x180 prediction")
        ax.set_title(f"{q} · chirp: 11 drives, 4 µs sweeps over 20 MHz" if q not in SHORT else f"{q} · chirp (sweep too short)")
        # saturation ladder
        files = sorted((HERE / "satladder_default" / backend).glob(f"satdef-x*-{q}.nc"),
                       key=lambda p: float(p.stem.split("-x")[1].split("-")[0]))
        ax2 = axes[i, 1]
        if files:
            rows, gs = [], []
            xs = None
            for f in files:
                d = xr.open_dataset(f).sel(qubit=q)
                sv = project_iq(d.I.values[None, :], d.Q.values[None, :])[0]
                rows.append(sv)
                gs.append(float(f.stem.split("-x")[1].split("-")[0]))
                xs = d.detuning.values / 1e6
            ax2.pcolormesh(xs, np.arange(len(rows)), stack(rows), shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
            ax2.set_yticks(np.arange(len(gs)))
            ax2.set_yticklabels([f"×{g:g}" for g in gs])
            ax2.set_ylabel("drive / node default")
            ax2.set_title(f"{q} · saturation: 20 µs, ×1 = {info[q]['sat_default_rabi'] / 1e6:.1f} MHz Rabi")
        ref = rep_s.get(f"{backend}/{q}", {}).get("reference")
        a = rep_c[f"{backend}/{q}"]["03a"]
        alpha = a.get("alpha_full")
        st = json.loads((HERE / backend / "state/state.json").read_text())["qubits"][q]["f_01"]
        for a_ in (ax, ax2):
            a_.set_xlabel("frequency − stored f01 [MHz]")
            if ref:
                a_.axvline((ref - st) / 1e6, color="#e6194b", lw=0.9, ls=":")
            if alpha:
                a_.axvline((ref - st) / 1e6 - alpha / 2e6, color="#ff9f40", lw=0.9, ls=":")
        ax2.set_xlim(ax.get_xlim())
    fig.suptitle(f"{backend}: the chirp's response (left) and saturation's (right) against frequency and drive "
                 "(red dotted: f01; orange dotted: the 0→2 line, f01 − α/2)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(HERE / f"compare_ladder_{backend}.png", dpi=100)
    plt.close(fig)


def main():
    rep_c = json.loads((HERE / "replay/replays.json").read_text())
    rep_s = json.loads((HERE / "replay/replays_sat.json").read_text())
    for be in (sys.argv[1].split(",") if len(sys.argv) > 1 else QUBITS):
        info = rabi_info(be)
        capture(be, rep_c, rep_s, info)
        ladder(be, rep_c, rep_s, info)
        print(be, "ok")


if __name__ == "__main__":
    main()
