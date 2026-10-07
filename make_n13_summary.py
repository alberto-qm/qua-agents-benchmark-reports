#!/usr/bin/env python3
"""The n13 summary page: make_n12_summary.py's page (the scramble, the nine-row results table, the three scatters and the recovery
table) for n13, the 30 Sep 2026 rerun of n12's 37 single-qubit bring-ups after the five fixes.

    python3 make_n13_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection. This script points that collection
at the n13 work dirs (night "new" = n13), swaps in n13's operator verdicts from make_n13_report.py and supplies the n13 wording.
"Off the recipe order" keeps make_n12_report.RECIPE["new"]: the fifteen numbered steps in every n13 system_prompt.md are n12's,
only the prose inside steps 3, 5 and 15 changed. The recovery table shows n12 and n13 per qubit, ordered by n13's count.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_summary as summ  # noqa: E402
import make_n13_report as n13rep  # noqa: E402

rep = summ.rep
OUT = Path(__file__).with_name("2026-09-30-n13-summary.html")


def configure() -> None:
    """Collect n13 as the summarised night, with n13's hand verdicts in make_n12_report's (night, backend, qubit) keys."""
    rep.NIGHTS = {"new": ("n13",)}
    rep.NOT_MEANINGFUL = {("new", b, q): f"completed, not meaningful: {why}"
                          for (night, b, q), why in n13rep.NOT_MEANINGFUL.items() if night == "n13"}
    rep.NOT_COMPLETED = {("new", b, q): why for (night, b, q), why in n13rep.NOT_COMPLETED.items() if night == "n13"}


def names(rs) -> str:
    return ", ".join(f"{r['backend']} {r['q']}" for r in sorted(rs, key=rep.qkey))


def rb_note(s: dict) -> str:
    """What the valid-RB count includes and leaves out, from the data."""
    valid = s["valid"]
    parts = ["fidelity per gate = 1 − EPC/1.875"]
    odd = [r for r in valid if rep.outcome(r)[0] != "ok"]
    if odd:
        parts.append(f"includes {names(odd)} (not meaningful)")
    left = [r for r in s["done"] if not rep.valid_rb(r)]
    flat = [r for r in left if r.get("rb_amp") is not None and r["rb_amp"] < rep.RB_MIN_AMPLITUDE]
    short = [r for r in left if r not in flat]
    if short:
        parts.append(f"excludes {names(short)} (no fit over one decay length)")
    if flat:
        parts.append(f"excludes {names(flat)} (the survival started near its floor: the qubit was not prepared in |0>)")
    depths = sorted({r["rb_depth"] for r in valid})
    covers = [r for r in valid if r["rb_cover"] is not None]
    if len(depths) == 1 and covers:
        low = min(covers, key=lambda r: r["rb_cover"])
        parts.append(f"every final RB at depth {depths[0]}, the shortest fit over {low['rb_cover']:.1f} decay lengths "
                     f"({low['backend']} {low['q']})")
    return "; ".join(parts)


def _minute(t):
    return (t + timedelta(seconds=30)).replace(second=0, microsecond=0)


def context(s: dict) -> str:
    a, b = (_minute(t) for t in s["span"])
    assert a.utcoffset() == b.utcoffset() == timedelta(hours=2), "result.json stamps are not CEST"
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · the same {s['n']} qubits on qolab, arbel and gilboa, rerun after the five "
            "fixes for n12's failures · tinycal with qwen3.8-27b on OpenRouter")


N13 = summ.Night(
    label="n13", stamp="20260930-0222", out=OUT,
    title="n13 Summary",
    description="n13, 30 Sep 2026: n12's 37 single-qubit bring-ups on qolab, arbel and gilboa rerun after five fixes, with tinycal "
                "and qwen3.8-27b on OpenRouter, in one table and three plots.",
    eyebrow="n13 validation run",
    h1="n13 single-qubit bring-ups",
    context=context,
    chip_sub=lambda s: f"30 Sep · {s['n']} qubits",
    rb_note=rb_note,
    pull="the pull the run started from (qolab and gilboa pulled 30 Sep 02:22; arbel reused the 29 Sep 00:17 pull n12 also used)",
    pull_short="the pull the run started from",
    recovery_intro="One qubit per pair of rows: <b>n12</b>, the 29–30 Sep run, and <b>n13</b>, its rerun after the five fixes, each "
                   "against the IQCC pull its own night started from (for arbel, the same 29 Sep 00:17 pull on both nights).",
    recovery_sort=("n13", "n12"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n13, fewest first, then by n12's count.",
)


if __name__ == "__main__":
    configure()
    summ.build(N13)
