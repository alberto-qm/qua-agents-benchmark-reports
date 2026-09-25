"""Fits for t1_chirp_test.py (no QPU): t1_chirp/summary.json, figures/t1_chirp_<backend>.png, figures/t1_chirp_compare.png.

Signal: I/Q projected on the line from the ground reference to the x180 at the shortest delay (0 = ground, 1 = the
x180's excited state at 16 ns), so the chirp's amplitude reads as its inversion relative to the calibrated pulse.
Fit per pulse: y(t) = A exp(-t/T1) + B.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "t1_chirp"
FIG = HERE / "figures"
QB = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CHIRP, C_X180 = "#1f6fb2", "#d9822b"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def decay(t, A, T1, B):
    return A * np.exp(-t / T1) + B


def fit(t_us, y):
    half = y[0] - 0.5 * (y[0] - y[-1])
    i = int(np.argmax(y < half)) if np.any(y < half) else len(y) - 1
    p0 = [max(1e-3, y[0] - y[-1]), max(0.05, float(t_us[i]) / np.log(2)), float(y[-1])]
    popt, pcov = curve_fit(decay, t_us, y, p0=p0, bounds=([0, 0.02, -1], [3, 5000, 1]), maxfev=20000)
    resid = y - decay(t_us, *popt)
    dof = max(1, y.size - 3)
    # no absolute sigma: curve_fit scales the covariance by the residual variance
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    return popt, perr, float(np.sqrt(np.sum(resid**2) / dof))


def lean_t1(be, q):
    """T1 from 03a's up/down-chirp lean (2 us sweep, T1 missing), when significant."""
    for f in sorted((HERE / "dirchk_latest" / be).glob("03a-noT1-*.json")):
        r = (json.loads(f.read_text()).get("fit_results") or {}).get(q)
        if r and r.get("sweep_over_t1") is not None and r["sweep_over_t1"] == r["sweep_over_t1"]:
            rho, err = r["sweep_over_t1"], r["sweep_over_t1_error"]
            return (2.0 / rho if rho > 3 * err else None), rho, err
    return None, None, None


summary = {}
for be, qs in QB.items():
    meta = json.loads((D / f"{be}_meta.json").read_text()) if (D / f"{be}_meta.json").exists() else {}
    fig, axes = plt.subplots(1, len(qs), figsize=(4.6 * len(qs), 3.9), squeeze=False)
    for c, q in enumerate(qs):
        f = D / f"{be}_{q}.npz"
        ax = axes[0, c]
        if not f.exists():
            ax.set_axis_off()
            continue
        d = np.load(f)
        t = d["delays_ns"] / 1e3
        g = complex(float(d["ground_I"]), float(d["ground_Q"]))
        zc = d["chirp_I"] + 1j * d["chirp_Q"]
        zx = d["x180_I"] + 1j * d["x180_Q"]
        axis = zx[0] - g
        proj = lambda z: ((z - g) * np.conj(axis)).real / abs(axis) ** 2
        yc, yx = proj(zc), proj(zx)
        out = {"delays_us": t.tolist(), "chirp": yc.tolist(), "x180": yx.tolist()}
        for key, y, col, lab in (("chirp", yc, C_CHIRP, "chirp (2 µs, 03a's f₀₁)"), ("x180", yx, C_X180, "calibrated x180")):
            ax.semilogx(t, y, "o", ms=4, color=col, label=lab)
            try:
                popt, perr, rms = fit(t, y)
                tt = np.geomspace(t.min(), t.max(), 300)
                ax.semilogx(tt, decay(tt, *popt), "-", color=col, lw=1.2)
                out[f"{key}_fit"] = {"A": popt[0], "T1_us": popt[1], "B": popt[2], "A_err": perr[0], "T1_err_us": perr[1],
                                     "B_err": perr[2], "rms": rms}
            except Exception as exc:  # noqa: BLE001
                out[f"{key}_fit"] = {"error": repr(exc)}
        stored = (meta.get(q) or {}).get("stored_T1")
        lt, rho, rho_err = lean_t1(be, q)
        out.update(stored_T1_us=stored * 1e6 if stored else None, lean_T1_us=lt, lean_rho=rho, lean_rho_err=rho_err,
                   qpu_s=(meta.get(q) or {}).get("qpu_s"), chirp_centre_mhz=((meta.get(q) or {}).get("centre", np.nan) -
                                                                             (meta.get(q) or {}).get("stored_f01", np.nan)) / 1e6)
        fc, fx = out.get("chirp_fit", {}), out.get("x180_fit", {})
        txt = []
        if "T1_us" in fc:
            txt.append(f"chirp: T1 {fc['T1_us']:.1f} ± {fc['T1_err_us']:.1f} µs, amplitude {fc['A']:.2f}")
        if "T1_us" in fx:
            txt.append(f"x180: T1 {fx['T1_us']:.1f} ± {fx['T1_err_us']:.1f} µs, amplitude {fx['A']:.2f}")
        if stored:
            txt.append(f"stored T1 {stored * 1e6:.1f} µs")
        ax.text(0.03, 0.04, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="bottom")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("delay before readout [µs]")
        if c == 0:
            ax.set_ylabel("excited population (0 = ground, 1 = x180 at 16 ns)")
        ax.set_title(f"{be} {q}")
        if c == len(qs) - 1:
            ax.legend(loc="upper right", fontsize=7.5, frameon=False)
        summary[f"{be}/{q}"] = out
    fig.suptitle(f"{be}: T1 with a chirp as the π pulse (blue) and with the calibrated x180 (orange), 1 ms between shots",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / f"t1_chirp_{be}.png", dpi=110)
    plt.close(fig)

# chirp T1 against x180 T1 (and the stored value)
fig, ax = plt.subplots(figsize=(5.6, 5.2))
lo, hi = 0.5, 300
ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8)
for k, v in summary.items():
    fc, fx = v.get("chirp_fit", {}), v.get("x180_fit", {})
    if "T1_us" in fc and "T1_us" in fx:
        ax.errorbar(fx["T1_us"], fc["T1_us"], xerr=fx["T1_err_us"], yerr=fc["T1_err_us"], fmt="o", ms=5, color=C_CHIRP, capsize=2)
        ax.annotate(k.split("/")[1], (fx["T1_us"], fc["T1_us"]), textcoords="offset points", xytext=(5, -10), fontsize=7.5)
        if v.get("stored_T1_us"):
            ax.plot(fx["T1_us"], v["stored_T1_us"], "x", color="#888", ms=6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("T1 with the calibrated x180 [µs]")
ax.set_ylabel("T1 with the chirp [µs]   (grey ×: the stored T1)")
ax.set_title("Chirp against x180, same job, same delays")
fig.tight_layout()
fig.savefig(FIG / "t1_chirp_compare.png", dpi=110)
(D / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for k, v in summary.items():
    fc, fx = v.get("chirp_fit", {}), v.get("x180_fit", {})
    print(f"{k:12s} chirp {fc.get('T1_us', float('nan')):7.2f} ± {fc.get('T1_err_us', float('nan')):5.2f} us (A {fc.get('A', float('nan')):.2f})   "
          f"x180 {fx.get('T1_us', float('nan')):7.2f} ± {fx.get('T1_err_us', float('nan')):5.2f} us (A {fx.get('A', float('nan')):.2f})   "
          f"stored {v.get('stored_T1_us') or float('nan'):6.2f}   lean {v.get('lean_T1_us') or float('nan'):5.2f}   QPU {v.get('qpu_s') or 0:.1f} s")
