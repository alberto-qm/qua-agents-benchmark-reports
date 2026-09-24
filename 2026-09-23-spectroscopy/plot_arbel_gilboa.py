"""Plot the arbel qB4 trap test and the gilboa qD2 short-T1 test (no QPU)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHIRP, SAT, PI = "#2a78d6", "#eb6834", "#1baf7a"
RAMP3 = ["#9cc3ee", "#5b9be0", "#1f5fae"]   # one hue, light -> dark = long -> short sweep
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.6})
MK = dict(markersize=7, markeredgewidth=1.3)


def loader(stem):
    d = np.load(HERE / f"{stem}_data.npz")
    return (lambda k: np.asarray(d[k], float).ravel()), json.load(open(HERE / f"{stem}_meta.json"))


def frame(g, job, r0="r0", r1="r1"):
    z0 = np.array([g(f"{job}__{r0}_I").mean(), g(f"{job}__{r0}_Q").mean()])
    ax = np.array([g(f"{job}__{r1}_I").mean(), g(f"{job}__{r1}_Q").mean()]) - z0
    return z0, ax


def dist(g, job, key, z0, ax):          # IQ distance of the averaged point from |0>, |0>-|1> separation = 1
    return np.hypot(g(f"{job}__{key}_I") - z0[0], g(f"{job}__{key}_Q") - z0[1]) / np.linalg.norm(ax)


def proj(g, job, key, z0, ax):
    return ((g(f"{job}__{key}_I") - z0[0]) * ax[0] + (g(f"{job}__{key}_Q") - z0[1]) * ax[1]) / (ax @ ax)


# ============================== arbel qB4
g, m = loader("trap_qB4")
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=(1.25, 1)))
fig.suptitle("arbel qB4 — the qubit where saturation spectroscopy kept finding the 0→2 line (23 Sep 2026)", x=0.012, ha="left",
             fontsize=12.5, fontweight="bold")
fig.text(0.012, 0.9, f"Measured on hardware, lab snapshot of 22 Sep (f₀₁ {m['f01_hz']/1e9:.5f} GHz, T1 {m['t1']*1e6:.0f} µs). "
         f"Stored saturation amplitude 1.0 → the node's default drive is {m['node_fr']/1e6:.0f} MHz Rabi. "
         "Signal: IQ distance from |0⟩, |0⟩–|1⟩ = 1.", ha="left", color=INK2, fontsize=9.2)
z0, ax = frame(g, "J1a"); so = np.array(m["sat_off"]) / 1e6
a.plot(so, dist(g, "J1a", "a", z0, ax), color=SAT, lw=1.1, marker="o", markersize=2.8, markeredgewidth=0,
       label=f"saturation 20 µs at the node's drive ({m['node_fr']/1e6:.0f} MHz Rabi)")
z0, ax = frame(g, "J1b"); co = np.array(m["chirp_off"]) / 1e6
a.plot(co, dist(g, "J1b", "a", z0, ax), color=CHIRP, lw=1.6, marker="s", markersize=5, markeredgecolor=SURFACE,
       label=f"chirp 4 µs, ±10 MHz band, {m['fr_scan']/1e6:.0f} MHz Rabi (x = band centre)")
f2 = m["f2ph_off"] / 1e6
a.annotate(f"0→2 line\nf₀₁ − {abs(f2):.0f} MHz", (f2, 0.66), textcoords="offset points", xytext=(18, 18), fontsize=8.8, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
a.annotate("real f₀₁", (1.5, 0.72), textcoords="offset points", xytext=(-80, 30), fontsize=8.8, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
a.set_xlim(-127, 27); a.set_ylim(-0.03, 1.22)
a.set_xlabel("drive frequency (or band centre) − f₀₁ [MHz]"); a.set_ylabel("signal (|0⟩→|1⟩ = 1)")
a.set_title("A · The view the agents had, and the chirp's view of the same 150 MHz")
a.legend(loc="upper left", fontsize=8.8)

lad = np.array(m["ladder"]) * m["fr_top"] / 1e6; cl = np.array(m["chirp_ladder"]) * m["fr_top"] / 1e6
drives = list(lad) + [m["node_fr"] / 1e6]
z0, ax = frame(g, "J2")
sat2 = [dist(g, "J2", f"s{j}", z0, ax).max() for j in range(len(drives))]
z0, ax = frame(g, "J3")
sat1 = [dist(g, "J3", f"s1_{j}", z0, ax)[0] for j in range(len(drives))]
ch1 = [dist(g, "J3", f"c1_{j}", z0, ax)[0] for j in range(len(cl))]
ch2 = [dist(g, "J3", f"c2_{j}", z0, ax)[0] for j in range(len(cl))]
xx = np.logspace(np.log10(0.4), np.log10(30), 200); rate = m["rate_hz_per_ns"] * 1e9; alpha = m["alpha_measured"]
b.plot(xx, 1 - np.exp(-np.pi ** 2 * (xx * 1e6) ** 2 / rate), color=CHIRP, ls="--", lw=1.0, alpha=0.8)
b.plot(xx, 1 - np.exp(-(np.pi ** 2 / 2) * (np.sqrt(2) * (xx * 1e6) ** 2 / alpha) ** 2 / rate), color=CHIRP, ls="--", lw=1.0, alpha=0.8)
b.plot(drives, sat1, color=SAT, marker="o", mfc=SURFACE, mec=SAT, lw=1.3, label="saturation on f₀₁", **MK)
b.plot(drives, sat2, color=SAT, marker="o", mfc=SAT, mec=SURFACE, lw=1.3, ls="--", label="saturation on the 0→2 line", **MK)
b.plot(cl, ch1, color=CHIRP, marker="s", mfc=SURFACE, mec=CHIRP, lw=1.3, label="chirp on f₀₁", **MK)
b.plot(cl, ch2, color=CHIRP, marker="s", mfc=CHIRP, mec=SURFACE, lw=1.3, ls="--", label="chirp on the 0→2 line", **MK)
b.axvline(m["node_fr"] / 1e6, color=INK2, ls=":", lw=1.1)
b.text(m["node_fr"] / 1e6 * 0.93, 1.13, "node\ndefault", ha="right", va="top", fontsize=8.5, color=INK2)
b.set_xscale("log"); b.set_xticks([0.5, 1, 2, 4, 8, 16]); b.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
b.set_xlim(0.4, 30); b.set_ylim(-0.05, 1.18); b.set_yticks(np.arange(0, 1.01, 0.2))
b.set_xlabel("drive strength: nominal Rabi frequency [MHz]"); b.set_ylabel("signal (|0⟩→|1⟩ = 1)")
b.set_title("B · Drive ladder on both lines (dashed: Landau–Zener)")
b.legend(loc="center left", bbox_to_anchor=(0, 0.42), fontsize=8.6)
fig.subplots_adjust(left=0.06, right=0.985, top=0.8, bottom=0.12, wspace=0.2)
fig.savefig(HERE / "trap_qB4.png", dpi=150)

# ============================== gilboa qD2
g, m = loader("shortT1_qD2")
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle(f"gilboa qD2 — how short can T1 be? (T1 {m['t1']*1e6:.2f} µs, 23 Sep 2026)", x=0.012, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.012, 0.9, "Measured on hardware; local copy of the 22 Sep snapshot with the C/D qubits marked active (the cloud state parks qD2's "
         "flux at 0 V). Signal: projection on the |0⟩→x180 axis, x180 = 1.", ha="left", color=INK2, fontsize=9.2)
z0, ax = frame(g, "J1", "ref0", "ref1")
amps = np.array(m["amps"]) * m["fr_top"] / 1e6
for tau, col in zip(m["taus"], RAMP3):
    ys = [proj(g, "J1", f"ch{tau}_{s:g}", z0, ax).mean() for s in m["amps"]]
    lab = f"chirp {tau/1000:g} µs" if tau >= 1000 else f"chirp {tau} ns"
    a.plot(amps, ys, color=col, marker="s", mfc=col, mec=SURFACE, lw=1.4, label=lab + " on the stored f₀₁", **MK)
sats = [(m["node_fr"] / 1e6, "sat_node"), (2, "sat_2"), (8, "sat_8")]
a.plot([x for x, _ in sats], [proj(g, "J1", k, z0, ax).mean() for _, k in sats], color=SAT, marker="o", mfc=SURFACE, mec=SAT,
       lw=1.3, label="saturation 20 µs on the stored f₀₁", **MK)
a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4, 8, 16]); a.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
a.set_xlim(0.4, 20); a.set_ylim(-0.05, 1.05)
a.set_xlabel("drive strength: nominal Rabi frequency [MHz]"); a.set_ylabel("excited population (x180 = 1)")
a.set_title("A · A 4 µs sweep is too slow for T1 = 1.3 µs; 250 ns works")
a.legend(loc="upper left", fontsize=8.6)
z0, ax = frame(g, "J2")
so = np.array(m["s_off"]) / 1e6; co = np.array(m["c_off"]) / 1e6
b.plot(so, proj(g, "J2", "a", z0, ax), color=SAT, lw=1.1, marker="o", markersize=3, markeredgewidth=0,
       label=f"saturation 20 µs at the node's drive ({m['node_fr']/1e6:.1f} MHz Rabi)")
bt, bs = m["best_chirp"]
b.plot(co, proj(g, "J2", "b", z0, ax), color=CHIRP, lw=1.5, marker="s", markersize=5, markeredgecolor=SURFACE,
       label=f"chirp {bt} ns, ±10 MHz band, {bs*m['fr_top']/1e6:g} MHz Rabi (x = band centre)")
b.axvline(0, color=INK2, ls=":", lw=1.1); b.text(0.3, 0.13, "stored f₀₁", fontsize=8.5, color=INK2, va="top")
b.set_xlim(-15.5, 15.5); b.set_ylim(-0.05, 1.05)
b.set_xlabel("drive frequency (or band centre) − stored f₀₁ [MHz]"); b.set_ylabel("excited population (x180 = 1)")
b.set_title("B · The real line sits 4.75 MHz below the stored f₀₁")
b.legend(loc="upper right", fontsize=8.6)
fig.subplots_adjust(left=0.06, right=0.985, top=0.8, bottom=0.12, wspace=0.2)
fig.savefig(HERE / "shortT1_qD2.png", dpi=150)
print("ok")
