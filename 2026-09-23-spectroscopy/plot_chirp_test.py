"""Plot chirp_test_data.npz -> chirp_vs_saturation.png (no QPU)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).parent
d = np.load(HERE / "chirp_test_data.npz")
m = json.load(open(HERE / "chirp_test_meta.json"))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHIRP, SAT, PI = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.6,
})
MK = dict(marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)


def key(job, lab, c):
    return f"{job}__{lab.replace(' ', '~')}_{c}"


def calib(job, r0, r1):
    z0 = np.array([d[key(job, r0, c)].mean() for c in "IQ"])
    ax = np.array([d[key(job, r1, c)].mean() for c in "IQ"]) - z0
    proj = lambda I, Q: ((I - z0[0]) * ax[0] + (Q - z0[1]) * ax[1]) / (ax @ ax)  # noqa: E731
    sigma = proj(d[key(job, r0, "I")], d[key(job, r0, "Q")]).std()
    return proj, sigma


# ---- job 1 points
proj1, _ = calib("job1", "ref0", "ref1")
pt = {}
for lab in m["points"]:
    p = proj1(d[key("job1", lab, "I")], d[key("job1", lab, "Q")])
    pt[lab] = (p.mean(), p.std() / np.sqrt(p.size))

rate = m["rate_hz_per_ns"] * 1e9  # Hz/s
fr_ref = m["fr_ref_nominal"]
chirp_nominal = 4e6  # the setting used for the spectra (scale 0.5)

fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
fig.suptitle("Chirp (adiabatic passage) vs saturation spectroscopy — qolab Q1, 23 Sep 2026",
             x=0.012, ha="left", fontsize=13, fontweight="bold")
fig.text(0.012, 0.935, "Flux at the idle point, 5×T1 thermal reset before every shot, population from IQ projected on the "
         "|0⟩→x180 axis measured in the same job. Error bars: shot noise.", ha="left", color=INK2, fontsize=9.5)

# ---- A: robustness to amplitude error
a = axs[0, 0]
x = np.logspace(np.log10(0.04), np.log10(2.4), 300)
a.plot(x, np.sin(np.pi / 2 * x) ** 2 * (x <= 2), color=PI, ls="--", lw=1.2, alpha=0.9, label="theory: sin²(½π·a)")
a.plot(x, 1 - np.exp(-np.pi ** 2 * (chirp_nominal * x) ** 2 / rate), color=CHIRP, ls="--", lw=1.2, alpha=0.9,
       label="theory: Landau–Zener")
scales = np.array(m["scales"])
cx = scales * fr_ref / chirp_nominal
cy = np.array([pt[f"chirp f01 s{s:g}"] for s in scales])
a.errorbar(cx, cy[:, 0], cy[:, 1], color=CHIRP, lw=0, elinewidth=1.2, label="chirp, 4 µs over ±10 MHz (plateau 0.97)", **MK)
px = np.array([0.7, 1.0, 1.3])
py = np.array([pt[f"pi x{s}"] for s in px])
a.errorbar(px, py[:, 0], py[:, 1], color=PI, lw=0, elinewidth=1.2, label="π pulse (x180, 48 ns)", **MK)
a.axhline(0.5, color=SAT, ls=":", lw=1.4)
a.text(0.045, 0.44, "saturation never exceeds ½", color=INK2, fontsize=9)
a.set_xscale("log")
a.set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2]); a.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1", "2"])
a.set_xlim(0.04, 2.4); a.set_ylim(-0.03, 1.32); a.set_yticks(np.arange(0, 1.01, 0.2))
a.set_xlabel("drive amplitude relative to the intended setting")
a.set_ylabel("excited population")
a.set_title("A · Robustness to a wrong drive amplitude")
for s, (v, _) in zip(px, py):
    a.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(0, 9 if s == 1.0 else -15), ha="center",
               fontsize=8.5, color=INK2)
a.legend(loc="upper left", ncol=2, fontsize=8.8)

# ---- B: spectra
b = axs[0, 1]
proj2, s2 = calib("main", "r0", "r1")
e2 = s2 / np.sqrt(m["n_spec"])
so = np.array(m["sat_off"]) / 1e6; co = np.array(m["chirp_off"]) / 1e6
sp = proj2(d["main__a_I"], d["main__a_Q"]); cp = proj2(d["main__b_I"], d["main__b_Q"])
lor = lambda f, A, f0, w, c: A / (1 + ((f - f0) / (w / 2)) ** 2) + c  # noqa: E731
(A, f0, w, c), _ = curve_fit(lor, so, sp, p0=(0.5, 0, 4, 0))
ff = np.linspace(-30, 30, 600)
b.plot(ff, lor(ff, A, f0, w, c), color=SAT, lw=1.2, alpha=0.8)
b.errorbar(so, sp, e2, color=SAT, lw=0, elinewidth=1, label="saturation 20 µs\n(node default drive)", **MK)
b.errorbar(co, cp, e2, color=CHIRP, lw=1.2, elinewidth=1, label="chirp 4 µs, ±10 MHz band\n(x = band centre)", **MK)
b.axvspan(-10, 10, color=CHIRP, alpha=0.06, lw=0)
b.text(0, 1.03, "one chirp covers this band", ha="center", fontsize=8.5, color=INK2)
b.annotate(f"FWHM {abs(w):.1f} MHz\npeak {A + c:.2f}", (f0 + abs(w) / 2, (A + c) / 2), textcoords="offset points",
           xytext=(70, -12), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
b.set_xlim(-31, 31); b.set_ylim(-0.05, 1.1)
b.set_xlabel("drive frequency (or band centre) − f₀₁  [MHz]")
b.set_ylabel("excited population")
b.set_title("B · What each method sees across ±30 MHz")
b.legend(loc="center right", bbox_to_anchor=(1, 0.66), fontsize=8.8)

# ---- C: at f01 vs at the two-photon frequency
cax = axs[1, 0]
groups = ["at f₀₁", "at f₀₁ − |α|/2\n(0→2 two-photon frequency)"]
bars = [("saturation 20 µs", SAT, None, ["sat20 f01", "sat20 2ph"]),
        ("saturation 80 µs (today's node)", SAT, "///", ["sat80 f01", "sat80 2ph"]),
        ("chirp, f_R 4 MHz nominal", CHIRP, None, ["chirp f01 s0.5", "chirp 2ph s0.5"])]
wbar = 0.24
for i, (lab, col, hatch, keys) in enumerate(bars):
    xs = np.arange(2) + (i - 1) * (wbar + 0.03)
    vals = [pt[k] for k in keys]
    cax.bar(xs, [v for v, _ in vals], wbar, yerr=[e for _, e in vals], color=col if not hatch else SURFACE,
            edgecolor=col, hatch=hatch, linewidth=1.4, label=lab, error_kw=dict(ecolor=INK2, lw=1))
    for xx, (v, _) in zip(xs, vals):
        cax.text(xx, max(v, 0) + 0.035, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
cax.set_xticks([0, 1]); cax.set_xticklabels(groups)
cax.set_ylim(0, 1.1); cax.set_ylabel("excited population")
cax.set_title("C · Signal on the qubit line vs on the 0→2 frequency")
cax.legend(loc="center right", bbox_to_anchor=(1, 0.62))
cax.grid(axis="x", visible=False)

# ---- D: fine scan around the two-photon frequency
dd = axs[1, 1]
proj3, s3 = calib("fine", "r0", "r1")
e3 = s3 / np.sqrt(m["n_spec"])
fo = np.array(m["fine_off"]) / 1e6
fp = proj3(d["fine__a_I"], d["fine__a_Q"])
dd.errorbar(fo, fp, e3, color=SAT, lw=1.0, elinewidth=1, label="saturation 20 µs, 0.1 MHz steps", **MK)
for s, mk in ((0.25, "s"), (0.5, "D"), (1.0, "^")):
    v, e = pt[f"chirp 2ph s{s:g}"]
    dd.errorbar([{0.25: 2.2, 0.5: 2.35, 1.0: 2.5}[s]], [v], [e], color=CHIRP, marker=mk, markersize=7, markeredgecolor=SURFACE, lw=0, elinewidth=1,
                label=f"chirp across ±10 MHz, f_R {s * fr_ref / 1e6:g} MHz: {v:.3f}")
dd.axhline(0.5, color=SAT, ls=":", lw=1.4)
dd.text(2.35, 0.07, "chirp", ha="center", fontsize=8.5, color=INK2)
dd.text(-1.95, 0.52, "height of the real f₀₁ line with the same drive", fontsize=8.5, color=INK2)
dd.set_xlim(-2.1, 2.6); dd.set_ylim(-0.08, 0.62)
dd.set_xlabel("drive frequency − (f₀₁ − |α|/2)  [MHz]   (|α| = 215.5 MHz, placeholder)")
dd.set_ylabel("excited population")
dd.set_title("D · Nothing within ±2 MHz of f₀₁ − |α|/2 — but α is a placeholder")
dd.legend(loc="center right", fontsize=8.5)

fig.tight_layout(rect=(0, 0, 1, 0.925))
out = HERE / "chirp_vs_saturation.png"
fig.savefig(out, dpi=150)
print(out, f"| saturation FWHM {abs(w):.2f} MHz, centre {f0:+.2f} MHz, peak {A + c:.3f}")
