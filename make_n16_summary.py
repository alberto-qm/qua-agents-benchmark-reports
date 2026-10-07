#!/usr/bin/env python3
"""The n16 summary page: make_n12_summary.py's page for n16, the 2 Oct 2026 rerun of the usual 37 single-qubit bring-ups
on the latest libraries (qua-libs b5767c1b, tinycal d3388d7) and the benchmark's state-field allowlist (600b096), with
arbel on its square readout and the 3000 ns depletion time.

    python3 make_n16_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection; this script points that
collection at the n16 work dirs, supplies n16's operator verdicts (NOT_MEANINGFUL / NOT_COMPLETED below, checked against
the transcripts) and the n16 wording. The recovery table shows n14 and n16 per qubit, ordered by n16's count.
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
OUT = Path(__file__).with_name("2026-10-02-n16-summary.html")
STAMP = (Path.home() / "qab-runs/n16-stamp.txt").read_text().strip()
N14_STAMP = (Path.home() / "qab-runs/n14-stamp.txt").read_text().strip()

# (backend, qubit) -> why a completed run's results are not a calibration / why a run did not complete
NOT_MEANINGFUL: dict = {}
NOT_COMPLETED: dict = {}


def configure() -> None:
    rep.NIGHTS = {"new": ("n16",)}
    rep.NOT_MEANINGFUL = {("new", b, q): f"completed, not meaningful: {why}" for (b, q), why in NOT_MEANINGFUL.items()}
    rep.NOT_COMPLETED = {("new", b, q): why for (b, q), why in NOT_COMPLETED.items()}
    summ.RECOVERY_NIGHTS = (("n14", N14_STAMP), ("n16", STAMP))


def context(s: dict) -> str:
    a, b = (n13summ._minute(t) for t in s["span"])
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · the usual {s['n']} qubits on qolab, arbel and gilboa, on "
            "the latest libraries and the state-field allowlist · tinycal with qwen3.8-27b on OpenRouter")


N16 = summ.Night(
    label="n16", stamp=STAMP, out=OUT,
    title="n16 Summary",
    description="n16, 2 Oct 2026: the usual 37 single-qubit bring-ups on qolab, arbel and gilboa on the latest libraries and "
                "the scramble's state-field allowlist, with tinycal and qwen3.8-27b on OpenRouter, in one table, three plots "
                "and a per-qubit check against IQCC.",
    eyebrow="n16 rerun",
    h1="n16 single-qubit bring-ups",
    context=context,
    chip_sub=lambda s: f"2 Oct · {s['n']} qubits",
    rb_note=n13summ.rb_note,
    pull="the pull the run started from (all three backends pulled 2 Oct 16:32; arbel on its square readout with 3000 ns depletion)",
    pull_short="the pull the run started from",
    recovery_intro="One qubit per pair of rows: <b>n14</b>, the 1–2 Oct run, and <b>n16</b>, the 2 Oct rerun on the latest "
                   "libraries and the state-field allowlist, each against the IQCC pull its own night started from.",
    recovery_sort=("n16", "n14"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n16, fewest first, then by n14's count.",
    spec="252e28f8ab313894",
    scramble_note="<p class='small muted'>New from this night: every other field of the pulled state is kept, reset or removed by "
                  "the spec's <span class='mono'>state_fields</span> rules, and a field no rule covers stops the scramble. The "
                  "lab's <span class='mono'>extras.target_RF_frequency</span> (qA6, qB3, qD1) and "
                  "<span class='mono'>resonator.extras.f_01_ground/excited</span> (19 arbel qubits), which reached earlier "
                  "arbel runs unscrambled, are removed.</p>",
)


if __name__ == "__main__":
    configure()
    summ.build(N16)
