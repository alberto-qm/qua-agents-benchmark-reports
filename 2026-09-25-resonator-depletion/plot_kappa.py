"""Ring-down traces, fine sweeps and pump-probe per qubit (no QPU).

figures/kappa_<backend>_r<round>.png   one column per qubit; rows: ring-down |z| (log) at the three powers with the fitted
                                        decay, the tail in the IQ plane, |S21| of the fine sweep with the circle-fit model,
                                        and the probe's excited fraction after a top-power readout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kappa_analyse import LEVELS, QUBITS, fit_decay, trace  # noqa: E402

sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from calibration_utils.resonator_spectroscopy.circle_fit import fit_notch_resonator  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
COL = {"low": "#1f77b4", "ro": "#2ca02c", "top": "#d62728"}


def main(rnd):
    for be, qs in QUBITS.items():
        mf = HERE / "kappa" / be / f"r{rnd}_meta.json"
        if not mf.exists():
            continue
        meta_all = json.loads(mf.read_text())
        fig, axes = plt.subplots(4, len(qs), figsize=(5.2 * len(qs), 16), squeeze=False)
        for c, q in enumerate(qs):
            meta = meta_all.get(q)
            if meta is None:
                for r in range(4):
                    axes[r, c].set_axis_off()
                continue
            chunk = meta["chunk_ns"]
            rdf = HERE / "kappa" / be / f"r{rnd}_{q}_ringdown.npz"
            if rdf.exists():
                z = trace(dict(np.load(rdf)), meta["four_demods"])
                t = (np.arange(z.shape[1]) + 0.5) * chunk
                ax = axes[0, c]
                for i, lv in enumerate(LEVELS):
                    ax.semilogy(t, np.abs(z[i]), ".-", ms=3, lw=0.6, color=COL[lv],
                                label=f"{lv} {meta['levels_dbm'][lv]:.1f} dBm")
                    down = t > meta["l_on"] + 80
                    fd = fit_decay((t[down] - meta["l_on"]) * 1e-9, z[i, down])
                    if fd:
                        ax.semilogy(t[down], np.abs(fd["model"]), "-", color=COL[lv], lw=1.6, alpha=0.6)
                        ax.text(0.98, 0.95 - 0.07 * i, f"{lv}: κ/2π {fd['kappa'] / 2 / np.pi / 1e6:.2f} MHz, "
                                f"δ {fd['delta'] / 1e6:+.2f}", transform=ax.transAxes, ha="right", fontsize=7, color=COL[lv])
                ax.axvline(meta["l_on"], color="k", lw=0.6, ls=":")
                ax.set_title(f"{be} {q} · ring-down (40 ns slices)")
                ax.set_xlabel("time in window [ns]")
                ax.legend(fontsize=7, loc="lower left")
                ax = axes[1, c]
                for i, lv in enumerate(LEVELS):
                    down = t > meta["l_on"]
                    zz = z[i, down] / np.abs(z[i, down][0])
                    ax.plot(zz.real, zz.imag, ".-", ms=3, lw=0.6, color=COL[lv], label=lv)
                    ax.plot(zz.real[0], zz.imag[0], "o", color=COL[lv])
                ax.set_aspect("equal")
                ax.set_title("tail in the IQ plane (normalised to its start)")
                ax.legend(fontsize=7)
            swf = HERE / "kappa" / be / f"r{rnd}_{q}_sweep.npz"
            if swf.exists():
                sw = dict(np.load(swf))
                f = meta["f_r"] + np.asarray(meta["sweep_hz"], float)
                ax = axes[2, c]
                for i, lv in enumerate(("low", "ro")):
                    s = sw["sw_I"][i] + 1j * sw["sw_Q"][i]
                    norm = np.median(np.abs(s[:40]))
                    ax.plot((f - meta["f_r"]) / 1e6, np.abs(s) / norm, ".", ms=2, color=COL[lv], label=lv)
                    i0 = int(np.argmin(np.abs(s)))
                    cf = fit_notch_resonator(f, s, f_r_guess=f[i0], q_l_guess=f[i0] / 1e6)
                    if cf.converged and cf.model_s is not None:
                        ax.plot((f - meta["f_r"]) / 1e6, np.abs(cf.model_s) / norm, "-", color=COL[lv], lw=1)
                ax.set_title("fine sweep |S21| (dots) and circle-fit model")
                ax.set_xlabel("frequency − stored readout frequency [MHz]")
                ax.legend(fontsize=7)
            ppf = HERE / "kappa" / be / f"r{rnd}_{q}_pump.npz"
            if ppf.exists():
                pp = dict(np.load(ppf))
                g, e = complex(pp["g_I"], pp["g_Q"]), complex(pp["e_I"], pp["e_Q"])
                zz = pp["pp_I"] + 1j * pp["pp_Q"]
                ax = axes[3, c]
                proj = ((zz - g) * np.conj(e - g)).real / abs(e - g) ** 2
                perp = ((zz - g) * np.conj(e - g)).imag / abs(e - g) ** 2
                taus = np.asarray(meta["taus_ns"]) / 1e3
                ax.semilogx(taus, proj, "o-", color="#d62728", label="along g→e")
                ax.semilogx(taus, perp, "s--", color="#7f7f7f", label="across")
                ax.axhline(0, color="k", lw=0.5)
                ax.axhline(1, color="k", lw=0.5, ls=":")
                for w in (3, 12):
                    ax.axvline(w, color="#1f77b4", lw=0.6, ls=":")
                ax.set_title(f"after a {meta['top_dbm']} dBm readout: committed readout, tau later")
                ax.set_xlabel("tau [µs]")
                ax.set_ylabel("shift / (e − g)")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"kappa_{be}_r{rnd}.png", dpi=90)
        plt.close(fig)
        print(be, "ok")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
