#!/usr/bin/env python3
"""The n15 summary page: make_n12_summary.py's page for n15, the 2 Oct 2026 rerun of n14's 21 arbel bring-ups with the
square readout's 3000 ns depletion time restored. n14 ran arbel on the square pulse but kept the lab's depletion times
sized for its self-depleting Drachma pulse (16-1100 ns), so active reset handed the gates a still-ringing resonator.
qolab and gilboa were not affected and are not rerun.

    python3 make_n15_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection; this script points that
collection at the n15 work dir, supplies n15's operator verdicts (NOT_MEANINGFUL / NOT_COMPLETED below, checked against
the transcripts) and the n15 wording. The recovery table shows n14 and n15 per arbel qubit, ordered by n15's count.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402

rep = summ.rep
OUT = Path(__file__).with_name("2026-10-02-n15-summary.html")
STAMP = (Path.home() / "qab-runs/n15-stamp.txt").read_text().strip()
N14_STAMP = (Path.home() / "qab-runs/n14-stamp.txt").read_text().strip()

# (backend, qubit) -> why a completed run's results are not a calibration / why a run did not complete
NOT_MEANINGFUL: dict = {
    ("arbel", "qA6"): "calibrated its own 1→2 line at 6.218 GHz, 200 MHz below the 0→1 n14 found at its sweet spot "
                      "(6.419 GHz, 284 MHz above the scrambled seed); no readout ever discriminated (IQ_blobs 50-51 %), "
                      "T1, Ramsey and DRAG failed, RB 0.81 % on a 16 % contrast",
}
NOT_COMPLETED: dict = {
    ("arbel", "qD1"): "escalated at turn 110: committed its own 1→2 line at 5.151 GHz; the 0→1 is at 5.366 GHz "
                      "(n13, n14), 418 MHz above the scrambled seed, and the search never went past +220 MHz; readout 51 %",
}


def configure() -> None:
    rep.NIGHTS = {"new": ("n15",)}
    rep.NOT_MEANINGFUL = {("new", b, q): f"completed, not meaningful: {why}" for (b, q), why in NOT_MEANINGFUL.items()}
    rep.NOT_COMPLETED = {("new", b, q): why for (b, q), why in NOT_COMPLETED.items()}
    summ.RECOVERY_NIGHTS = (("n14", N14_STAMP), ("n15", STAMP))


def context(s: dict) -> str:
    a, b = (n13summ._minute(t) for t in s["span"])
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · n14's {s['n']} arbel qubits, rerun with the square readout's "
            "3000 ns depletion time restored · tinycal with qwen3.8-27b on OpenRouter")


N15 = summ.Night(
    label="n15", stamp=STAMP, out=OUT,
    title="n15 Summary",
    description="n15, 2 Oct 2026: n14's 21 arbel single-qubit bring-ups rerun with the square readout's depletion time "
                "restored, with tinycal and qwen3.8-27b on OpenRouter, in one table, three plots and a per-qubit check "
                "against IQCC.",
    eyebrow="n15 arbel rerun",
    h1="n15 arbel bring-ups",
    context=context,
    chip_sub=lambda s: f"2 Oct · {s['n']} qubits",
    rb_note=n13summ.rb_note,
    pull="the pull the run started from (arbel pulled 2 Oct 03:02, on its square readout pulse with 3000 ns depletion)",
    pull_short="the pull the run started from",
    recovery_intro="One qubit per pair of rows: <b>n14</b>, the 1–2 Oct run, whose arbel half kept the Drachma pulse's "
                   "16–1100 ns depletion times with the square readout, and <b>n15</b>, its rerun with 3000 ns, each "
                   "against the IQCC pull its own night started from.",
    recovery_sort=("n15", "n14"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n15, fewest first, then by n14's count.",
)


if __name__ == "__main__":
    configure()
    summ.build(N15)
