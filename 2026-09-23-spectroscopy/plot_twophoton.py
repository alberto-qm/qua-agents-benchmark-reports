"""Plot the two-photon suppression test -> twophoton_suppression.png (no QPU)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHIRP, SAT = "#2a78d6", "#eb6834"
RAMP5 = ["#f5c7ad", "#f09a70", "#eb6834", "#bf4d1c", "#843310"]  # one hue, light -> dark = weak -> strong drive
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.6,
})
MK = dict(markersize=7, markeredgewidth=1.4)


def load(tag):
    d = np.load(HERE / f"twophoton_data_{tag}.npz")
    return (lambda k: np.asarray(d[k], float).ravel()), json.load(open(HERE / f"twophoton_meta_{tag}.json"))


def frame(g, job):
    z0 = np.array([g(f"{job}__r0_I").mean(), g(f"{job}__r0_Q").mean()])
    ax = np.array([g(f"{job}__r1_I").mean(), g(f"{job}__r1_Q").mean()]) - z0
    L = np.linalg.norm(ax)
    s = ((np.stack([g(f"{job}__r0_I"), g(f"{job}__r0_Q")], 1) - z0) @ (ax / L) / L).std()
    return z0, L, s


def dev(g, job, key, z0, L):
    return np.hypot(g(f"{job}__{key}_I") - z0[0], g(f"{job}__{key}_Q") - z0[1]) / L


def devmean(g, job, key, z0, L):
    """Per-shot data: distance of the AVERAGED IQ point from |0> (averaging magnitudes would add the shot noise)."""
    return np.hypot(g(f"{job}__{key}_I").mean() - z0[0], g(f"{job}__{key}_Q").mean() - z0[1]) / L


g1, m1 = load("-140_-70")
g2, m2 = load("-200_-135")
alpha = m2["alpha_measured"]; rate = m2["rate_hz_per_ns"] * 1e9
fr = np.array(m2["scales"]) * m2["fr_top"] / 1e6
node_fr = 0.5 * 0.021577948205837635 * (1e9 / 48) / 0.0937268099122044 / 1e6  # the node's default drive on Q1

fig = plt.figure(figsize=(13, 9.4))
gs = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
fig.suptitle("Does a chirp ignore the 0→2 two-photon line? — qolab Q1, 23 Sep 2026", x=0.012, ha="left",
             fontsize=13, fontweight="bold")
fig.text(0.012, 0.925, "All points measured on hardware (flux at idle, 5×T1 reset per shot). Signal: IQ distance from |0⟩ in units "
         "of the |0⟩–|1⟩ separation\n(|2⟩ may read out off that axis). Dashed curves: Landau–Zener theory with the measured α, no fitted parameters.",
         ha="left", color=INK2, fontsize=9.3)

# ---- A: finding the 0->2 line
a = fig.add_subplot(gs[0, 0])
for g, m in ((g1, m1), (g2, m2)):
    z0, L, s = frame(g, "S")
    off = np.array(m["s_off"]) / 1e6
    a.plot(off, dev(g, "S", "a", z0, L), color=SAT, lw=1.1, marker="o", markersize=3.2, markeredgewidth=0)
a.axvline(-m2["placeholder_alpha"] / 2e6, color=INK2, ls=":", lw=1.2)
a.text(-m2["placeholder_alpha"] / 2e6 + 1.5, 0.78, f"where the stored α\n({m2['placeholder_alpha']/1e6:.1f} MHz, placeholder)\nputs it",
       fontsize=8.5, color=INK2, va="top")
pk = m2["search_peak_off"] / 1e6
a.annotate(f"0→2 line at f₀₁ − {abs(pk):.0f} MHz\n→ α ≈ {alpha/1e6:.0f} MHz", (pk, 0.6), textcoords="offset points",
           xytext=(-150, 4), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
a.set_xlim(-201, -69); a.set_ylim(-0.02, 0.85)
a.set_xlabel("drive frequency − f₀₁  [MHz]")
a.set_ylabel("signal (|0⟩→|1⟩ = 1)")
a.set_title("A · Searching for the 0→2 line (saturation, 16 MHz Rabi)")

# ---- B: saturation lines at the 0->2 frequency vs drive
b = fig.add_subplot(gs[0, 1])
z0, L, s = frame(g2, "L1")
foff = np.array(m2["f_off"]) / 1e6
for j, f in enumerate(fr):
    b.plot(foff, dev(g2, "L1", f"sat{j}", z0, L), color=RAMP5[j], lw=1.3, marker="o", markersize=3.2, markeredgewidth=0,
           label=f"Rabi {f:g} MHz")
b.set_xlim(-3.05, 3.05); b.set_ylim(-0.02, 0.95)
b.set_xlabel("drive frequency − (0→2 line)  [MHz]")
b.set_ylabel("signal (|0⟩→|1⟩ = 1)")
b.set_title("B · Saturation (20 µs): a sharp 0→2 line from 2 MHz Rabi")
b.legend(loc="upper left", fontsize=8.8, title="saturation drive", title_fontsize=8.8)

# ---- C: the comparison
c = fig.add_subplot(gs[1, :])
z0, L, s = frame(g2, "L2")
err = s / np.sqrt(1000)
sat2 = [dev(g2, "L1", f"sat{j}", *frame(g2, "L1")[:2]).max() for j in range(len(fr))]
ch2 = [devmean(g2, "L2", f"chirp2ph{j}", z0, L) for j in range(len(fr))]
ch1 = [devmean(g2, "L2", f"chirp01_{j}", z0, L) for j in range(len(fr))]
sa1 = [devmean(g2, "L2", f"sat01_{j}", z0, L) for j in range(len(fr))]
xx = np.logspace(np.log10(0.8), np.log10(20), 200)
fr2 = np.sqrt(2) * (xx * 1e6) ** 2 / alpha
c.plot(xx, 1 - np.exp(-(np.pi ** 2 / 2) * fr2 ** 2 / rate), color=CHIRP, ls="--", lw=1.1, alpha=0.8)
c.plot(xx, 1 - np.exp(-np.pi ** 2 * (xx * 1e6) ** 2 / rate), color=CHIRP, ls="--", lw=1.1, alpha=0.8)
c.plot(fr, sa1, color=SAT, marker="o", mfc=SURFACE, mec=SAT, ls="-", lw=1.4, label="saturation 20 µs on f₀₁ (real line)", **MK)
c.plot(fr, sat2, color=SAT, marker="o", mfc=SAT, mec=SURFACE, ls="--", lw=1.4, label="saturation 20 µs on the 0→2 line (peak of B)", **MK)
c.errorbar(fr, ch1, err, color=CHIRP, marker="s", mfc=SURFACE, mec=CHIRP, ls="-", lw=1.4, elinewidth=1,
           label="chirp on f₀₁ (real line)", **MK)
c.errorbar(fr, ch2, err, color=CHIRP, marker="s", mfc=CHIRP, mec=SURFACE, ls="--", lw=1.4, elinewidth=1,
           label="chirp on the 0→2 line", **MK)
c.axvspan(node_fr * 0.93, node_fr * 1.07, color=INK2, alpha=0.12, lw=0)
c.text(node_fr, 1.06, f"node default\n({node_fr:.1f} MHz)", ha="center", va="bottom", fontsize=8.5, color=INK2)
for f, v2, v1 in zip(fr, ch2, sat2):
    c.annotate(f"{v2:.2f}", (f, v2), textcoords="offset points", xytext=(7, -13), ha="left", fontsize=8.5, color=INK2)
    c.annotate(f"{v1:.2f}", (f, v1), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK2)
c.set_xscale("log"); c.set_xticks(fr); c.set_xticklabels([f"{f:g}" for f in fr])
c.set_xlim(0.8, 20); c.set_ylim(-0.05, 1.22)
c.set_yticks(np.arange(0, 1.01, 0.2))
c.set_xlabel("drive strength: nominal Rabi frequency [MHz]  (chirp: 4 µs sweep over ±10 MHz)")
c.set_ylabel("signal (|0⟩→|1⟩ = 1)")
c.set_title("C · Same drive, two targets: saturation lights up the 0→2 line from 2 MHz, the chirp not until 8–16 MHz")
c.legend(loc="center left", bbox_to_anchor=(0.0, 0.375), fontsize=9)

fig.tight_layout(rect=(0, 0, 1, 0.915))
out = HERE / "twophoton_suppression.png"
fig.savefig(out, dpi=150)
print(out)
