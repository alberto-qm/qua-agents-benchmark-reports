"""Summary figures for the depletion / kappa tests (no QPU).

figures/summary_02b.png    per qubit: optimal power and frequency shift under A B C D, both rounds; QPU per job
figures/summary_kappa.png  per qubit: each kappa estimate over the ring-down kappa (log scale)
figures/summary_pump.png   per qubit: the committed readout's shift towards |e>, tau after a top-power readout
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1b1b1a", "#5b5a56", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
QB = [("arbel", "qA5"), ("arbel", "qD1"), ("qolab", "Q1"), ("qolab", "Q2"), ("qolab", "Q5")]
QB_ALL = [("arbel", "qB4")] + QB
CCOL = {"A": "#3b6fb6", "B": "#d1495b", "C": "#2a9d8f", "D": "#e9a23b"}
CLAB = {"A": "A today (12 µs, power inner)", "B": "B 3 µs, power inner", "C": "C 3 µs, power outer",
        "D": "D 1 µs, power outer"}


def fig_02b():
    s = json.loads((HERE / "ab" / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    x = np.arange(len(QB))
    for j, cond in enumerate("ABCD"):
        off = (j - 1.5) * 0.17
        for rnd, mk in ((1, "o"), (2, "s")):
            op, fs, qpu = [], [], []
            for be, q in QB:
                r = s.get(f"{be}/{q}/{cond}/r{rnd}") or {}
                ok = r.get("success") and r.get("optimal_power") is not None
                op.append(r["optimal_power"] - s[f"{be}/{q}/A/r1"]["optimal_power"] if ok else np.nan)
                fs.append((r["frequency_shift"] - s[f"{be}/{q}/A/r1"]["frequency_shift"]) / 1e6 if ok else np.nan)
                qpu.append(r.get("qpu_s", np.nan))
            lab = CLAB[cond] if rnd == 1 else None
            axes[0].plot(x + off, op, mk, color=CCOL[cond], ms=7, mec="white", mew=0.8, label=lab, ls="none")
            axes[1].plot(x + off, fs, mk, color=CCOL[cond], ms=7, mec="white", mew=0.8, ls="none")
            if rnd == 1:
                axes[2].bar(j, np.nanmean(qpu), color=CCOL[cond], width=0.62)
                axes[2].text(j, np.nanmean(qpu) + 0.8, f"{np.nanmean(qpu):.0f} s", ha="center", fontsize=9, color=INK)
    for ax, lab in ((axes[0], "optimal power − A round 1 [dB]"), (axes[1], "frequency shift − A round 1 [MHz]")):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{be}\n{q}" for be, q in QB])
        ax.set_ylabel(lab)
    axes[0].set_title("Proposed readout power (circles round 1, squares round 2)")
    axes[1].set_title("Proposed readout frequency")
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(list("ABCD"))
    axes[2].set_ylabel("QPU per qubit [s]")
    axes[2].set_title("QPU per job (mean of 5 qubits)")
    axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "summary_02b.png", dpi=110)
    plt.close(fig)


def fig_kappa():
    ks = json.loads((HERE / "kappa" / "summary.json").read_text())
    ls = json.loads((HERE / "kappa" / "long_summary.json").read_text())
    ab = json.loads((HERE / "ab" / "summary.json").read_text())
    est = [("long pulse, low power, |S21|² Lorentzian", "#2a9d8f", "o"),
           ("long pulse, low power, circle fit", "#2a9d8f", "D"),
           ("readout pulse, |S21|² Lorentzian", "#d1495b", "o"),
           ("readout pulse, circle fit", "#d1495b", "D"),
           ("02b linewidth (node, A)", "#6a4c93", "s"),
           ("02a FWHM (node)", "#e9a23b", "^")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(QB_ALL))
    for j, (lab, col, mk) in enumerate(est):
        off = (j - 2.5) * 0.12
        ys = []
        for be, q in QB_ALL:
            kr = ls.get(f"{be}/{q}", {}).get("ringdown", np.nan)
            L = ls.get(f"{be}/{q}", {})
            if j == 0:
                v = [L.get("long_low", {}).get("lorentz", np.nan)]
            elif j == 1:
                v = [L.get("long_low", {}).get("circle", np.nan)]
            elif j == 2:
                v = [L.get("short_ro", {}).get("lorentz", np.nan)]
            elif j == 3:
                v = [L.get("short_ro", {}).get("circle", np.nan)]
            elif j == 4:
                v = [(ab.get(f"{be}/{q}/A/r{r}") or {}).get("linewidth") for r in (1, 2)]
            else:
                v = []
                for r in (1, 2):
                    f = HERE / "ab" / be / f"02a-r{r}-{q}.json"
                    if f.exists():
                        v.append(((json.loads(f.read_text()).get("fit_results") or {}).get(q) or {}).get("fwhm"))
            v = [float(a) for a in v if a is not None and np.isfinite(a)]
            ys.append(np.mean(v) / kr if v and np.isfinite(kr) else np.nan)
        ax.semilogy(x + off, ys, mk, color=col, ms=7, mec="white", mew=0.8, ls="none", label=lab,
                    alpha=1.0 if j % 2 == 0 or j >= 4 else 0.75)
    ax.axhline(1, color=INK, lw=1.0)
    ax.axhspan(0.85, 1.18, color="#2a9d8f", alpha=0.08, lw=0)
    ax.set_xticks(x)
    labels = []
    for be, q in QB_ALL:
        kr = ls.get(f"{be}/{q}", {}).get("ringdown", np.nan)
        labels.append(f"{be} {q}\nκ/2π {kr / 1e6:.2f} MHz")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("estimate / ring-down κ")
    ax.set_yticks([0.5, 1, 2, 3, 5, 10, 30])
    ax.set_yticklabels(["0.5", "1", "2", "3", "5", "10", "30"])
    ax.set_title("κ from linewidths, relative to κ from the ring-down (band: ±15 %)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "summary_kappa.png", dpi=110)
    plt.close(fig)


def fig_pump():
    ks = json.loads((HERE / "kappa" / "summary.json").read_text())
    cols = ["#9aa0a6", "#3b6fb6", "#6a4c93", "#d1495b", "#e9a23b", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for (be, q), col in zip(QB_ALL, cols):
        ys = []
        for r in (1, 2):
            p = (ks.get(f"{be}/{q}/r{r}") or {}).get("pump")
            if p:
                ys.append(np.asarray(p["excited"], float))
                taus = np.asarray(p["taus_ns"]) / 1e3
        if not ys:
            continue
        y = np.mean(ys, axis=0)
        meta = (ks.get(f"{be}/{q}/r1") or {}).get("meta") or {}
        dp = meta.get("top_dbm", np.nan) - meta.get("committed_power_dbm", np.nan)
        ax.semilogx(taus, y, "o-", color=col, ms=4, lw=1.4, label=f"{be} {q}: top of 02b = committed + {dp:.0f} dB")
    for w, lab in ((3, "3 µs"), (12, "12 µs")):
        ax.axvline(w, color=INK2, lw=0.8, ls=":")
        ax.text(w * 1.05, 1.12, lab, fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("time after the top-power readout [µs]")
    ax.set_ylabel("next readout's shift, 0 = |g⟩, 1 = |e⟩")
    ax.set_title("A top-of-sweep readout leaves the qubit excited for tens of µs")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "summary_pump.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_02b()
    fig_kappa()
    fig_pump()
    print("ok")
