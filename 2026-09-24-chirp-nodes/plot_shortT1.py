"""Short-T1 chirp variants (shortT1_variants.py): signal vs band centre at five drives, per qubit and pulse (no QPU)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import find_boxes, noise_sigma
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 9.5,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
d = np.load(HERE / "shortT1/shortT1_variants.npz"); meta = json.loads((HERE / "shortT1/shortT1_variants.json").read_text())
LEV = meta["levels"]; KINDS = [("lin20", "linear 20 MHz (the node)"), ("lin80", "linear 80 MHz"), ("hs40", "hyperbolic secant 40 MHz")]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharey="row")
summary = {}
for i, q in enumerate(meta["plan"]):
    p = meta["plan"][q]
    z0 = np.array([d[f"{q}_r0_I"].mean(), d[f"{q}_r0_Q"].mean()]); ax1 = np.array([d[f"{q}_r1_I"].mean(), d[f"{q}_r1_Q"].mean()]) - z0
    for j, (kind, label) in enumerate(KINDS):
        ax = axes[i, j]; pp = p["pulses"][kind]; x = np.asarray(pp["centres"]) / 1e6
        rows = []
        for k in range(len(LEV)):
            I, Q = d[f"{q}_{kind}_{k}_I"], d[f"{q}_{kind}_{k}_Q"]
            s = ((I - z0[0]) * ax1[0] + (Q - z0[1]) * ax1[1]) / (ax1 @ ax1)
            rows.append(s)
            fr = pp["played"][k] * p["kappa"] / 1e6
            ax.plot(x, s + 0.0 * k, color=plt.cm.viridis(k / (len(LEV) - 1)), lw=1.3, marker="o", ms=2.5, label=f"{fr:.0f} MHz Rabi")
        rows = np.array(rows); sig = noise_sigma(rows, axis=1)
        found = []
        for k in range(len(LEV)):
            bx = [b for b in find_boxes(np.asarray(pp["centres"]), rows[k], sig, pp["band"]) if b.inside]
            found.append([(round(b.centre / 1e6, 2), round(b.centre_error / 1e6, 2), round(b.height, 2)) for b in bx])
        summary[f"{q}/{kind}"] = found
        ax.axvline(-4.75 if q == "qD2" else (5648.584 - p["f01"] / 1e6), color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{q} (T1 {p['T1'] * 1e6:.1f} µs, {p['tau']} ns) · {label}")
        ax.set_xlabel("band centre − stored f01 [MHz]")
        if j == 0:
            ax.set_ylabel("excited population (x180 = 1)")
        ax.set_ylim(-0.15, 1.05)
        ax.grid(color=GRID, lw=0.5)
        if i == 0 and j == 2:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Short T1: the node's 20 MHz chirp against a wider linear sweep and a hyperbolic-secant pulse (dotted: the line from saturation / 23 Sep)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(HERE / "shortT1_variants.png", dpi=110)
json.dump(summary, open(HERE / "shortT1/boxes.json", "w"), indent=1)
for k, v in summary.items(): print(k, v)
