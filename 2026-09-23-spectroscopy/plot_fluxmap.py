"""Plot fluxmap_data.npz -> fluxmap_chirp_vs_saturation.png, and print the per-column fits (no QPU)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.special import erf

HERE = Path(__file__).parent
d = np.load(HERE / "fluxmap_data.npz"); m = json.load(open(HERE / "fluxmap_meta.json"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHIRP, SAT = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f7f8", "#aeb9c4", "#5c6f82", "#1f3346"])  # one hue, light -> dark
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.6,
})

dcs = np.array(m["dcs"]) * 1e3                      # mV
rows = np.array(m["sat_rows"]) / 1e6; cen = np.array(m["chirp_centres"]) / 1e6


def pop(kind, n):
    z0 = np.array([d[f"{kind}_r0I"].ravel().mean(), d[f"{kind}_r0Q"].ravel().mean()])
    ax = np.array([d[f"{kind}_r1I"].ravel().mean(), d[f"{kind}_r1Q"].ravel().mean()]) - z0
    P = ((d[f"{kind}_I"] - z0[0]) * ax[0] + (d[f"{kind}_Q"] - z0[1]) * ax[1]) / (ax @ ax)
    s = (((d[f"{kind}_r0I"].ravel() - z0[0]) * ax[0] + (d[f"{kind}_r0Q"].ravel() - z0[1]) * ax[1]) / (ax @ ax)).std()
    return P, s / np.sqrt(n)


Ps, es = pop("sat", m["n_sat"]); Pc, ec = pop("chirp", m["n_chirp"])
lor = lambda f, A, f0, w, c: A / (1 + ((f - f0) / (w / 2)) ** 2) + c  # noqa: E731


def box(f, A, f0, W, s, c):
    return c + A / 2 * (erf((f - (f0 - W / 2)) / (np.sqrt(2) * s)) - erf((f - (f0 + W / 2)) / (np.sqrt(2) * s)))


def fit_cols(P, x, e, model, p0fun, bounds):
    out = []
    for y in P:
        try:
            p, c = curve_fit(model, x, y, p0=p0fun(x, y), sigma=np.full_like(y, e), absolute_sigma=True, bounds=bounds, maxfev=20000)
            out.append((p[1], np.sqrt(c[1, 1]), p))
        except Exception:  # noqa: BLE001
            out.append((np.nan, np.nan, None))
    return out


sat_fit = fit_cols(Ps, rows, es, lor, lambda x, y: (y.max(), x[np.argmax(y)], 3, 0), ([0, -30, 0.2, -0.2], [1.2, 8, 30, 0.3]))
ch_fit = fit_cols(Pc, cen, ec, box, lambda x, y: (0.95, np.average(x, weights=np.clip(y, 0, None) + 1e-9), 18, 1.0, 0),
                  ([0.3, -30, 10, 0.2, -0.2], [1.3, 8, 26, 5, 0.3]))


def parab(phi, f0, k, p0):
    return f0 + k * (phi - p0) ** 2


def arc(fits):
    f = np.array([a for a, _, _ in fits]); e = np.array([b for _, b, _ in fits]); ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    p, c = curve_fit(parab, dcs[ok], f[ok], p0=(0, m["quad"] / 1e12, 0), sigma=e[ok], absolute_sigma=True)
    return f, e, p, np.sqrt(np.diag(c)), ok


fs, e_s, ps, pse, oks = arc(sat_fit); fc, e_c, pc, pce, okc = arc(ch_fit)
print(f"saturation: apex {pc[0]*0 + ps[0]:+.3f} +/- {pse[0]:.3f} MHz at {ps[2]:+.2f} +/- {pse[2]:.2f} mV; median column error {np.nanmedian(e_s):.3f} MHz; "
      f"QPU {m['qpu_s']['sat']:.1f} s ({len(rows)} rows x {m['n_sat']} shots)")
print(f"chirp     : apex {pc[0]:+.3f} +/- {pce[0]:.3f} MHz at {pc[2]:+.2f} +/- {pce[2]:.2f} mV; median column error {np.nanmedian(e_c):.3f} MHz; "
      f"QPU {m['qpu_s']['chirp']:.1f} s ({len(cen)} centres x {m['n_chirp']} shots)")
print("per column (flux mV: sat f / err | chirp f / err):")
for x, a, b, c_, e_ in zip(dcs, fs, e_s, fc, e_c):
    print(f"  {x:+7.1f}: {a:+7.2f} / {b:.3f} | {c_:+7.2f} / {e_:.3f}")
chirp_widths = [p[2] for _, _, p in ch_fit if p is not None]
print(f"chirp box width median {np.median(chirp_widths):.2f} MHz, plateau median {np.median([p[0] for _, _, p in ch_fit if p is not None]):.2f}")

# ---- figure
fig, axs = plt.subplots(2, 2, figsize=(13, 9.6), gridspec_kw=dict(height_ratios=(1.05, 1)))
fig.suptitle("Qubit spectroscopy vs flux: chirp vs saturation — qolab Q1, 23 Sep 2026", x=0.012, ha="left", fontsize=13, fontweight="bold")
fig.text(0.012, 0.925, "Measured on hardware. Flux pulsed from the idle point, readout at idle, 5×T1 reset per shot. Colour: excited "
         "population (|0⟩→x180 axis).\nSaturation as node 03b runs it today (80 µs, 0.5× stored amplitude, 50 shots); chirp: 4 µs over ±10 MHz "
         "at ~3 MHz Rabi, flux step leading by 5 µs, 200 shots.", ha="left", color=INK2, fontsize=9.3)
xx = np.linspace(dcs.min() - 3, dcs.max() + 3, 300)
dx = dcs[1] - dcs[0]


def edges(v):
    step = v[1] - v[0]
    return np.concatenate([v - step / 2, [v[-1] + step / 2]])


for axm, P, y, fit_f, fit_e, p, kind, col, ttl in (
        (axs[0, 0], Ps, rows, fs, e_s, ps, "sat", SAT, f"A · Saturation map ({len(rows)} rows × 15 columns, {m['qpu_s']['sat']:.1f} s QPU)"),
        (axs[0, 1], Pc, cen, fc, e_c, pc, "chirp", CHIRP, f"B · Chirp map ({len(cen)} band centres × 15 columns, {m['qpu_s']['chirp']:.1f} s QPU)")):
    mesh = axm.pcolormesh(edges(dcs), edges(y), np.clip(P.T, -0.05, 1.0), cmap=SEQ, vmin=0, vmax=1, shading="flat")
    axm.errorbar(dcs, fit_f, fit_e, color=col, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                 lw=0, elinewidth=1.2, label="per-column fit centre")
    axm.plot(xx, parab(xx, *p), color=col, lw=1.4, label="parabola through the centres")
    axm.set_xlabel("flux offset from the idle point [mV]")
    axm.set_ylabel("drive frequency − f₀₁ [MHz]" if kind == "sat" else "band centre − f₀₁ [MHz]")
    axm.set_title(ttl, fontsize=10.5)
    axm.grid(False)
    axm.legend(loc="lower center", fontsize=8.8, facecolor=SURFACE, framealpha=0.9, frameon=True, edgecolor=GRID)
cb = fig.colorbar(mesh, ax=axs[0, :], shrink=0.85, pad=0.01)
cb.set_label("excited population"); cb.outline.set_edgecolor(GRID)

# ---- C: extracted arcs
c = axs[1, 0]
c.errorbar(dcs, fs, e_s, color=SAT, marker="o", markersize=6, markeredgecolor=SURFACE, lw=0, elinewidth=1.2, label="saturation (Lorentzian per column)")
c.errorbar(dcs + 0.8, fc, e_c, color=CHIRP, marker="s", markersize=6, markeredgecolor=SURFACE, lw=0, elinewidth=1.2, label="chirp (box per column)")
c.plot(xx, parab(xx, *ps), color=SAT, lw=1.2); c.plot(xx, parab(xx, *pc), color=CHIRP, lw=1.2, ls="--")
c.set_xlabel("flux offset from the idle point [mV]"); c.set_ylabel("fitted f₀₁ − stored f₀₁ [MHz]")
c.set_title("C · The two arcs (chirp points offset by 0.8 mV for visibility)")
c.legend(loc="lower center", fontsize=9)

# ---- D: per-column precision
dd = axs[1, 1]
dd.plot(dcs, e_s * 1e3, color=SAT, marker="o", markersize=6, markeredgecolor=SURFACE, label=f"saturation, {m['qpu_s']['sat']:.1f} s QPU")
dd.plot(dcs, e_c * 1e3, color=CHIRP, marker="s", markersize=6, markeredgecolor=SURFACE, label=f"chirp, {m['qpu_s']['chirp']:.1f} s QPU")
dd.set_yscale("log")
dd.set_xlabel("flux offset from the idle point [mV]"); dd.set_ylabel("fit uncertainty of the column centre [kHz]")
dd.set_title("D · Per-column precision")
dd.legend(loc="upper center", fontsize=9)
summary = (f"sweet spot   saturation {ps[2]:+.2f} ± {pse[2]:.2f} mV,  f {ps[0]:+.3f} ± {pse[0]:.3f} MHz\n"
           f"             chirp      {pc[2]:+.2f} ± {pce[2]:.2f} mV,  f {pc[0]:+.3f} ± {pce[0]:.3f} MHz")
dd.text(0.02, 0.04, summary, transform=dd.transAxes, fontsize=8.8, color=INK2, family="monospace")

fig.subplots_adjust(left=0.06, right=0.95, top=0.86, bottom=0.07, hspace=0.33, wspace=0.18)
out = HERE / "fluxmap_chirp_vs_saturation.png"
fig.savefig(out, dpi=150)
print(out)
