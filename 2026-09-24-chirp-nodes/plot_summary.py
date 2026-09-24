"""Summary figures for the 24 Sep chirp-node report (no QPU).

capture_03a.png   per qubit: outcome of the 03a replays over (drive mismatch, stored-f_01 error)
capture_03b.png   per qubit: outcome of the 03b replays over (idle-flux error, stored-f_01 error)
sweetspots.png    sweet spot and apex frequency: chirp 03b (start, end), saturation 03b, per qubit
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OUTCOME = {"pass": ("#1baf7a", 0), "imprecise": ("#a8d8a0", 1), "refused": ("#c9c7c1", 2), "wrong": ("#d6453d", 3)}
CMAP = ListedColormap([OUTCOME[k][0] for k in ("pass", "imprecise", "refused", "wrong")])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
SHORT = {"gilboa/qD2", "gilboa/qC3"}


def legend(fig):
    handles = [Patch(color=OUTCOME[k][0], label=k) for k in ("pass", "imprecise", "refused", "wrong")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.955), ncol=4, frameon=False, fontsize=9)


def capture_03a(rep):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    for i, (be, qs) in enumerate(QUBITS.items()):
        for j, q in enumerate(qs):
            ax = axes[i, j]
            a = (rep.get(f"{be}/{q}") or {}).get("03a")
            ax.set_title(f"{be} {q}")
            if not a or not a["grid"]:
                ax.text(0.5, 0.5, "no wide map", ha="center", transform=ax.transAxes)
                continue
            ms = sorted({g["m"] for g in a["grid"]})
            Fs = sorted({g["F"] for g in a["grid"]})
            Z = np.full((len(ms), len(Fs)), np.nan)
            for g in a["grid"]:
                Z[ms.index(g["m"]), Fs.index(g["F"])] = OUTCOME[g["outcome"]][1]
            ax.imshow(Z, aspect="auto", origin="lower", cmap=CMAP, vmin=-0.5, vmax=3.5,
                      extent=[Fs[0] / 1e6 - 2.5, Fs[-1] / 1e6 + 2.5, ms[0] - 0.5, ms[-1] + 0.5])
            ax.set_yticks(ms)
            ax.set_yticklabels([f"x{2.0 ** m:g}" if m >= 0 else f"x1/{2 ** -m:g}" for m in ms])
            ax.set_xlabel("stored f01 error [MHz]")
            ax.set_ylabel("real drive / x180 prediction")
            npass = sum(g["outcome"] == "pass" for g in a["grid"])
            nw = sum(g["outcome"] == "wrong" for g in a["grid"])
            note = f"{npass}/{len(a['grid'])} pass, {nw} wrong"
            if f"{be}/{q}" in SHORT:
                note += "\nshort sweep: the node refuses all"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))
    fig.suptitle("03a chirp: replays of the drive ladder over a wrong stored f01 and a wrong drive calibration",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "capture_03a.png", dpi=130)


def capture_03b(rep):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
    for i, (be, qs) in enumerate(QUBITS.items()):
        for j, q in enumerate(qs):
            ax = axes[i, j]
            b = (rep.get(f"{be}/{q}") or {}).get("03b")
            ax.set_title(f"{be} {q}")
            if not b or not b["grid"]:
                ax.text(0.5, 0.5, "no wide map", ha="center", transform=ax.transAxes)
                continue
            ds = sorted({round(g["delta_phi20"], 3) for g in b["grid"]})
            Fs = sorted({g["F"] for g in b["grid"]})
            Z = np.full((len(Fs), len(ds)), np.nan)
            for g in b["grid"]:
                Z[Fs.index(g["F"]), ds.index(round(g["delta_phi20"], 3))] = OUTCOME[g["outcome"]][1]
            step = ds[1] - ds[0] if len(ds) > 1 else 0.1
            ax.imshow(Z, aspect="auto", origin="lower", cmap=CMAP, vmin=-0.5, vmax=3.5,
                      extent=[ds[0] - step / 2, ds[-1] + step / 2, Fs[0] / 1e6 - 2.5, Fs[-1] / 1e6 + 2.5])
            ax.set_xlabel("idle error [units of the offset that drops f01 by 20 MHz]")
            ax.set_ylabel("stored f01 error [MHz]")
            npass = sum(g["outcome"] == "pass" for g in b["grid"])
            nw = sum(g["outcome"] == "wrong" for g in b["grid"])
            truth = b.get("truth_x0")
            note = f"{npass}/{len(b['grid'])} pass, {nw} wrong"
            if truth is None:
                note += "\nno arc in the wide map"
            if f"{be}/{q}" in SHORT:
                note += "\nshort sweep: the node refuses all"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))
    fig.suptitle("03b chirp: replays of the flux map over a wrong idle point and a wrong stored f01",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "capture_03b.png", dpi=130)


def final_rows():
    live = json.loads((HERE / "replay/live.json").read_text())
    rows = []
    for be, qs in QUBITS.items():
        for q in qs:
            def pick(prefix):
                for k, v in live.items():
                    if k.startswith(f"{be}/{q}/{prefix}") and v.get("final"):
                        return v["final"]
                return {}
            rows.append((be, q, pick("r0-03b"), pick("end-03b"), pick("sat-")))
    return rows


def sweetspots():
    rows = final_rows()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["chirp map, start of session", "chirp map, end of session"]
    colors = ["#2a78d6", "#7fb0ea"]
    marks = ["s", "D"]
    names = []
    for i, (be, q, r0, end, sat) in enumerate(rows):
        names.append(f"{be}\n{q}")
        ok_sat = sat.get("proposed") and sat.get("x0") is not None
        for k, r in enumerate((r0, end)):
            if not (r.get("proposed") and ok_sat):
                continue
            dx = (r["x0"] - sat["x0"]) * 1e3
            ex = np.hypot(r["x0_error"] or 0, sat["x0_error"] or 0) * 1e3
            a.errorbar(i + (k - 0.5) * 0.25, dx, yerr=ex, fmt=marks[k], color=colors[k], ms=6, capsize=2,
                       label=labels[k] if not a.get_legend_handles_labels()[1].count(labels[k]) else None)
            df = (r["f0"] - sat["f0"]) / 1e3
            ef = np.hypot(r["f0_error"] or 0, sat["f0_error"] or 0) / 1e3
            b.errorbar(i + (k - 0.5) * 0.25, df, yerr=ef, fmt=marks[k], color=colors[k], ms=6, capsize=2)
        if not ok_sat or not r0.get("proposed"):
            why = ("no arc in any map" if not ok_sat and not r0.get("proposed") else
                   "chirp refused (short sweep)" if not r0.get("proposed") and ok_sat else "no saturation arc")
            for ax in (a, b):
                ax.text(i, 0.02, why, rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK2,
                        transform=ax.get_xaxis_transform())
    for ax in (a, b):
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.6)
    a.set_ylabel("chirp sweet spot - saturation sweet spot [mV]")
    a.set_title("A · Sweet spot: chirped map minus saturation map (same session)")
    b.set_ylabel("chirp apex f01 - saturation apex f01 [kHz]")
    b.set_title("B · Frequency at the sweet spot: chirped minus saturation")
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "sweetspots.png", dpi=130)


def main():
    rep = json.loads((HERE / "replay/replays.json").read_text())
    capture_03a(rep)
    capture_03b(rep)
    sweetspots()
    print("ok")


if __name__ == "__main__":
    main()
