"""Long-pulse steady-state sweep against the qubit's own pulse, and against the ring-down kappa (no QPU).

usage: long_analyse.py      prints a table and writes kappa/long_summary.json; figures/long_<backend>.png
Widths: the Lorentzian FWHM of |S21|^2 (kappa/2pi for a notch resonator at steady state) and the circle fit's f_r/Q_l,
for the 10 us pulse integrated over its last 4 us (committed - 10 dB and committed) and for the readout pulse (committed).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kappa_analyse import QUBITS, amp_fwhm, lorentz_power  # noqa: E402
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from calibration_utils.resonator_spectroscopy.circle_fit import fit_notch_resonator  # noqa: E402


def widths(f, s):
    lz = lorentz_power(f, s)
    i0 = int(np.argmin(np.abs(s)))
    cf = fit_notch_resonator(f, s, f_r_guess=f[i0], q_l_guess=f[i0] / 0.5e6)
    return dict(lorentz=lz["fwhm"] if lz else np.nan, lorentz_err=lz["fwhm_err"] if lz else np.nan,
                f0=lz["f0"] if lz else np.nan, circle=(cf.f_r / cf.q_l) if cf.converged else np.nan,
                amp_fwhm=amp_fwhm(f, s))


def ringdown_kappa(be, q):
    s = json.loads((HERE / "kappa" / "summary.json").read_text())
    ks = []
    for rnd in (1, 2):
        rd = (s.get(f"{be}/{q}/r{rnd}") or {}).get("ringdown") or {}
        for lv in ("low", "ro"):
            k = (rd.get(lv) or {}).get("kappa_down")
            if k is not None and np.isfinite(k):
                ks.append(k / 2 / np.pi)
    return (float(np.mean(ks)), float(np.std(ks))) if ks else (np.nan, np.nan)


def main():
    out = {}
    for be, qs in QUBITS.items():
        mf = HERE / "kappa" / be / "long_meta.json"
        if not mf.exists():
            continue
        meta_all = json.loads(mf.read_text())
        fig, axes = plt.subplots(1, len(qs), figsize=(5.2 * len(qs), 4.2), squeeze=False)
        print(f"\n{be}: kappa/2pi [MHz] ring-down | long pulse low / ro (Lorentz |S21|^2, circle) | readout pulse ro (Lorentz, circle, |S21| width)")
        for c, q in enumerate(qs):
            meta = meta_all.get(q)
            npz = HERE / "kappa" / be / f"long_{q}.npz"
            if meta is None or not npz.exists():
                axes[0, c].set_axis_off()
                continue
            d = dict(np.load(npz))
            f = meta["f_r"] + np.asarray(meta["sweep_hz"], float)
            sl = [d["long_I"][i] + 1j * d["long_Q"][i] for i in range(2)]
            ss = d["short_I"] + 1j * d["short_Q"]
            w = {"long_low": widths(f, sl[0]), "long_ro": widths(f, sl[1]), "short_ro": widths(f, ss)}
            kr, kr_sd = ringdown_kappa(be, q)
            w["ringdown"] = kr
            w["ringdown_sd"] = kr_sd
            w["readout_len"] = meta["readout_len"]
            out[f"{be}/{q}"] = w
            m = lambda v: "—" if not np.isfinite(v) else f"{v / 1e6:.3f}"  # noqa: E731
            print(f"  {q:4s} rd {kr / 1e6:.3f}±{kr_sd / 1e6:.3f} | long low {m(w['long_low']['lorentz'])} ({m(w['long_low']['circle'])}) "
                  f"ro {m(w['long_ro']['lorentz'])} ({m(w['long_ro']['circle'])}) | short {m(w['short_ro']['lorentz'])} "
                  f"({m(w['short_ro']['circle'])}, {m(w['short_ro']['amp_fwhm'])})  readout {meta['readout_len']} ns")
            ax = axes[0, c]
            for s, lab, col in ((sl[0], "10 µs pulse, last 4 µs, committed − 10 dB", "#1f77b4"),
                                (sl[1], "10 µs pulse, last 4 µs, committed", "#2ca02c"),
                                (ss, f"readout pulse ({meta['readout_len']} ns), committed", "#d62728")):
                p2 = np.abs(s) ** 2
                p2 = p2 / np.median(np.concatenate([p2[:30], p2[-30:]]))
                ax.plot((f - meta["f_r"]) / 1e6, p2, ".", ms=2.5, color=col, label=lab)
            if np.isfinite(kr) and np.isfinite(w["long_low"]["f0"]):
                x = (f - w["long_low"]["f0"]) / 1e6
                depth = 1 - np.min(np.abs(sl[0]) ** 2) / np.median(np.abs(sl[0][:30]) ** 2)
                ax.plot((f - meta["f_r"]) / 1e6, 1 - depth / (1 + (2 * x / kr) ** 2), "k--", lw=1,
                        label=f"Lorentzian, width = ring-down κ/2π {kr:.2f} MHz")
            ax.set_title(f"{be} {q}: |S21|² near the resonance", fontsize=9, loc="left")
            ax.set_xlabel("frequency − stored readout frequency [MHz]", fontsize=8)
            ax.legend(fontsize=6.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(HERE / "figures" / f"long_{be}.png", dpi=95)
        plt.close(fig)
    (HERE / "kappa" / "long_summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
