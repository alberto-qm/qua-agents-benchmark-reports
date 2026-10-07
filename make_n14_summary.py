#!/usr/bin/env python3
"""The n14 summary page: make_n12_summary.py's page for n14, the 1-2 Oct 2026 rerun of n13's 37 single-qubit bring-ups
after the 1 Oct fixes (instrument limits, the flux pulse past QUA's amplitude scale, 03b's scattered-arc refusal, DRAG
aliasing, the DRAG/RB recipe lines), with the qubit lengths read-only in tinycal's profile.

    python3 make_n14_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection; this script points that
collection at the n14 work dirs, supplies n14's operator verdicts (NOT_MEANINGFUL / NOT_COMPLETED below, checked against
the transcripts) and the n14 wording. The recovery table shows n13 and n14 per qubit, ordered by n14's count.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402

rep = summ.rep
OUT = Path(__file__).with_name("2026-10-01-n14-summary.html")
STAMP = (Path.home() / "qab-runs/n14-stamp.txt").read_text().strip()

# (backend, qubit) -> why a completed run's results are not a calibration / why a run did not complete
NOT_MEANINGFUL: dict = {
    ("arbel", "qA2"): "our arbel prep kept the lab's 100 ns depletion time with the square readout, so every Ramsey ran on a "
                      "ringing resonator; one garbage fit moved f_01 −2.4 MHz, DRAG refused 12 sweeps, α −2.1 typed by hand, "
                      "RB 2.3 %",
    ("arbel", "qC3"): "our arbel prep kept the lab's 16 ns depletion time with the square readout: every Ramsey failed, f_01 "
                      "was left 0.77 MHz off and DRAG saw no fringes, RB 4.3 % (the lab's 0.2 %)",
    ("gilboa", "qC5"): "the agent wrote the drive LO as 7.8 GHz over a port playing 5.8 GHz: the right qubit was calibrated "
                       "(RB 0.17 %) but f_01 is stored as 7.743 GHz for a qubit at 5.743 GHz",
}
NOT_COMPLETED: dict = {}


def configure() -> None:
    rep.NIGHTS = {"new": ("n14",)}
    rep.NOT_MEANINGFUL = {("new", b, q): f"completed, not meaningful: {why}" for (b, q), why in NOT_MEANINGFUL.items()}
    rep.NOT_COMPLETED = {("new", b, q): why for (b, q), why in NOT_COMPLETED.items()}
    summ.RECOVERY_NIGHTS = (("n13", "20260930-0222"), ("n14", STAMP))


def context(s: dict) -> str:
    a, b = (n13summ._minute(t) for t in s["span"])
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · n13's {s['n']} qubits on qolab, arbel and gilboa, rerun after the "
            "1 Oct fixes · tinycal with qwen3.8-27b on OpenRouter")


N14 = summ.Night(
    label="n14", stamp=STAMP, out=OUT,
    title="n14 Summary",
    description="n14, 1-2 Oct 2026: n13's 37 single-qubit bring-ups on qolab, arbel and gilboa rerun after the 1 Oct fixes, "
                "with tinycal and qwen3.8-27b on OpenRouter, in one table, three plots and a per-qubit check against IQCC.",
    eyebrow="n14 rerun",
    h1="n14 single-qubit bring-ups",
    context=context,
    chip_sub=lambda s: f"1–2 Oct · {s['n']} qubits",
    rb_note=n13summ.rb_note,
    pull="the pull the run started from (all three backends pulled 1 Oct 21:40; arbel on its square readout pulse)",
    pull_short="the pull the run started from",
    recovery_intro="One qubit per pair of rows: <b>n13</b>, the 30 Sep run, and <b>n14</b>, its rerun after the 1 Oct fixes, each "
                   "against the IQCC pull its own night started from.",
    recovery_sort=("n14", "n13"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n14, fewest first, then by n13's count.",
)


if __name__ == "__main__":
    configure()
    summ.build(N14)
