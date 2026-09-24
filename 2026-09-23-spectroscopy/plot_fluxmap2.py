"""Plot fluxmap2_data.npz -> fluxmap2_chirp_vs_saturation.png (no QPU)."""
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
d = np.load(HERE / "fluxmap2_data.npz"); m = json.load(open(HERE / "fluxmap2_meta.json"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHIRP, SAT = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f7f8", "#aeb9c4", "#5c6f82", "#1f3346"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.6})
dcs = np.array(m["dcs"]) * 1e3; rows = np.array(m["sat_rows"]) / 1e6; cen = np.array(m["chirp_centres"]) / 1e6
W, SIG = 17.4, 1.07
box = lambda f, A, f0, c: c + A / 2 * (erf((f - (f0 - W / 2)) / (np.sqrt(2) * SIG)) - erf((f - (f0 + W / 2)) / (np.sqrt(2) * SIG)))  # noqa: E731
lor = lambda f, A, f0, w, c: A / (1 + ((f - f0) / (w / 2)) ** 2) + c  # noqa: E731
parab = lambda phi, f0, k, p0: f0 + k * (phi - p0) ** 2  # noqa: E731


def popmap(kind, n):
    z0 = np.array([d[f"{kind}_r0I"].ravel().mean(), d[f"{kind}_r0Q"].ravel().mean()])
    ax = np.array([d[f"{kind}_r1I"].ravel().mean(), d[f"{kind}_r1Q"].ravel().mean()]) - z0
    P = lambda I, Q: ((I - z0[0]) * ax[0] + (Q - z0[1]) * ax[1]) / (ax @ ax)  # noqa: E731
    return P(d[f"{kind}_I"], d[f"{kind}_Q"]), P(d[f"{kind}_r0I"].ravel(), d[f"{kind}_r0Q"].ravel()).std() / np.sqrt(n)


def centres(M, x, e, model):
    out = []
    for y in M:
        try:
            if model == "box":
                p, c = curve_fit(box, x, y, p0=(0.95, x[np.argmax(y)], 0), sigma=np.full_like(y, e), absolute_sigma=True)
            else:
                p, c = curve_fit(lor, x, y, p0=(y.max(), x[np.argmax(y)], 3, 0), sigma=np.full_like(y, e), absolute_sigma=True,
                                 bounds=([0, -35, 0.2, -0.2], [1.3, 8, 40, 0.4]), maxfev=20000)
            out.append((p[1], np.sqrt(c[1, 1])))
        except Exception:  # noqa: BLE001
            out.append((np.nan, np.nan))
    f, e_ = np.array(out).T
    ok = np.isfinite(f) & (e_ > 0)
    p, c = curve_fit(parab, dcs[ok], f[ok], p0=(0, -2.5e-3, 0), sigma=e_[ok], absolute_sigma=True)
    return f, e_, p, np.sqrt(np.diag(c))


maps = {}
for kind, n, x, model in (("sat", m["n_sat"], rows, "lor"), ("satm", m["n_sat"], rows, "lor"), ("chirp", m["n_chirp"], cen, "box")):
    M, e = popmap(kind, n)
    maps[kind] = (M, x) + centres(M, x, e, model)

fig, axs = plt.subplots(2, 2, figsize=(13.5, 10))
fig.suptitle("Qubit spectroscopy vs flux: chirp vs saturation — qolab Q1, 23 Sep 2026", x=0.012, ha="left", fontsize=13, fontweight="bold")
fig.text(0.012, 0.922, "All measured on hardware; ~17.5 s QPU per map. Flux pulsed from the idle point, readout at idle, 5×T1 reset per shot. "
         "Colour: excited population.\nSaturation: 80 µs at 0.5× the stored amplitude, 71 rows × 50 shots. Chirp: 4 µs over ±10 MHz at ~3 MHz "
         "Rabi, 23 band centres × 200 shots, flux step leading by 5 µs.", ha="left", color=INK2, fontsize=9.3)


def edges(v):
    s = v[1] - v[0]
    return np.concatenate([v - s / 2, [v[-1] + s / 2]])


xx = np.linspace(-100, 100, 300)
for axm, kind, col, ttl, ylab in (
        (axs[0, 0], "sat", SAT, "A · Saturation as node 03b plays it: drive and flux step start together", "drive frequency − f₀₁ [MHz]"),
        (axs[0, 1], "satm", SAT, "B · Same saturation, drive inside the flux step (5 µs margins)", "drive frequency − f₀₁ [MHz]"),
        (axs[1, 0], "chirp", CHIRP, "C · Chirp (flux step leads by 5 µs)", "band centre − f₀₁ [MHz]")):
    M, y, f, e_, p, pe = maps[kind]
    mesh = axm.pcolormesh(edges(dcs), edges(y), np.clip(M.T, -0.05, 1), cmap=SEQ, vmin=0, vmax=1, shading="flat")
    axm.errorbar(dcs, f, e_, color=col, marker="o" if kind != "chirp" else "s", markersize=6, markeredgecolor=SURFACE,
                 markeredgewidth=1.2, lw=0, elinewidth=1.2, label="column centre (fit)")
    axm.plot(xx, parab(xx, *p), color=col, lw=1.4, label=f"parabola: sweet spot {p[2]:+.2f} ± {pe[2]:.2f} mV")
    axm.set_xlim(-102, 102); axm.set_xlabel("flux offset from the idle point [mV]"); axm.set_ylabel(ylab)
    axm.set_title(ttl); axm.grid(False)
    axm.legend(loc="lower center", fontsize=8.6, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92)
    cb = fig.colorbar(mesh, ax=axm, fraction=0.04, pad=0.015); cb.outline.set_edgecolor(GRID); cb.ax.tick_params(labelsize=8)

dd = axs[1, 1]
dd.plot(xx, m["quad"] * (xx / 1e3) ** 2 / 1e6, color=INK2, ls=":", lw=1.3, label="stored DC flux curvature (prediction)")
for kind, col, mk, fc_, lab in (("sat", SAT, "o", SAT, "saturation, 03b timing"), ("satm", SAT, "o", SURFACE, "saturation, inside flux step"),
                                ("chirp", CHIRP, "s", CHIRP, "chirp")):
    _, _, f, e_, p, pe = maps[kind]
    dd.errorbar(dcs + (0.8 if kind == "chirp" else 0), f, e_, color=col, marker=mk, mfc=fc_, mec=col if fc_ == SURFACE else SURFACE,
                markersize=6, markeredgewidth=1.3, lw=0, elinewidth=1.1, label=lab)
    dd.plot(xx, parab(xx, *p), color=col, lw=1.1, ls="--" if kind == "chirp" else "-", alpha=0.8)
dd.set_xlim(-102, 102); dd.set_ylim(-26, 7)
dd.set_xlabel("flux offset from the idle point [mV]"); dd.set_ylabel("fitted f₀₁ − stored f₀₁ [MHz]")
dd.set_title("D · Extracted arcs (chirp offset by 0.8 mV for visibility)")
dd.legend(loc="lower center", fontsize=8.6, ncol=2)
_, _, _, ec, pc, pce = maps["chirp"]; _, _, _, es, ps, pse = maps["satm"]
dd.text(0.03, 0.97, f"per-column precision (median):  chirp ±{np.nanmedian(ec)*1e3:.0f} kHz   saturation ±{np.nanmedian(es)*1e3:.0f} kHz\n"
        f"sweet spot:  chirp {pc[2]:+.2f} ± {pce[2]:.2f} mV   saturation {ps[2]:+.2f} ± {pse[2]:.2f} mV",
        transform=dd.transAxes, va="top", fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.06, right=0.97, top=0.865, bottom=0.06, hspace=0.3, wspace=0.2)
out = HERE / "fluxmap2_chirp_vs_saturation.png"; fig.savefig(out, dpi=150); print(out)
