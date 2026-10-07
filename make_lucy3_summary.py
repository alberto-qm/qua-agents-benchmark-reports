#!/usr/bin/env python3
"""The lucy3 summary page: make_n12_summary.py's page for lucy3, the 5 Oct 2026 bring-up of all seven qubits of IQCC's lucy
(fixed-frequency coaxmons) with the resonator's line shape in the state and the resonator nodes reading it (qua-libs
feat/resonator-line-shape bf0b0c45, tinycal feat/resonator-line-shape 99d1abe).

    python3 make_lucy3_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection, pointed at the lucy3 work dir.
lucy differs from the three flux-tunable chips in three places this script supplies: the recipe order the off-order count is
judged against (bringup_recipes/fixed_frequency_1q.md), the scramble (lucy's variant: no readout-amplitude kick, no flux
resets), and the recovery table (no sweet spot to find; the readout amplitude is not scrambled, so it gets no verdict). The
recovery table shows lucy1 and lucy2 (30 Sep, the same scramble without the line shape) beside lucy3.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402
from make_fwcmp2_report import esc  # noqa: E402

rep = summ.rep
OUT = Path(__file__).with_name("2026-10-05-lucy3-summary.html")
STAMP = "20261005-1437"
LUCY1, LUCY2 = "20260930-1013", "20260930-1057"
SPEC = "e4abf350db9725e9"  # ~/qab-runs/lucy3/decalibrate_lucy.yaml

# bringup_recipes/fixed_frequency_1q.md (tinycal 99d1abe), in the form make_night4_report.node_decisions judges against; the
# conditional T1 after power_rabi (only when T1_chirp stored none) is left out, as in make_n12_report.RECIPE.
RECIPE = [("02a_resonator_spectroscopy",), ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy",), ("T1_chirp",),
          ("qubit_spectroscopy_fine",), ("power_rabi",), ("readout_power_optimization",), ("readout_frequency_optimization",),
          ("IQ_blobs",), ("power_rabi_error_amplification_x180",), ("ramsey",), ("T1", "T2echo"), ("DRAG_calibration",),
          ("Randomized_benchmarking",)]

SCRAMBLED = [
    ("readout resonator frequency (RF_frequency, f_01)", "+25 MHz", "within ±1 MHz"),
    ("qubit frequency f_01 (and the drive's RF_frequency)", "−50 MHz", "within ±2 MHz"),
    ("x180 amplitude (x90 the same)", "×0.5", "0.7–1.4× the lab's"),
    ("anharmonicity", "set to 200 MHz", ""),
    ("readout threshold, RUS exit threshold, integration-weights angle", "→ 0", ""),
    ("DRAG α, x180 and x90 detuning", "→ 0", ""),
    ("T2*, T2 echo, χ, f₁₂, confusion matrix, stored gate fidelity", "cleared", ""),
    ("the lab's RB numbers in the qubit's extras (1QRB_p, its error, epg)", "removed", ""),
]
KEPT = ("the readout amplitude (the shared spec's ×1.8 kick takes qA, qB, qE and qF above the 1.0 ceiling), T1, pulse and "
        "readout lengths, LO frequencies and wiring")


def scramble_table(night: summ.Night) -> str:
    work = summ.RUNS_DIR / f"{night.label}-lucy-{night.stamp}"
    digest = json.load(open(work / "edit_plan.json")).get("spec_digest")
    assert digest == night.spec, f"lucy3 used scramble spec {digest}, not {night.spec}"
    trs = "".join(f"<tr><td>{esc(what)}</td><td class='num'>{esc(how)}</td><td class='num'>{esc(band)}</td></tr>"
                  for what, how, band in SCRAMBLED)
    return (
        "<h3 class='scr'>What the scramble changes</h3>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>parameter, on every qubit</th>"
        "<th>scrambled</th><th>graded: back in ballpark if</th></tr></thead><tbody>" + trs + "</tbody></table></div>"
        f"<p class='small muted'>Not touched: {esc(KEPT)}. Scramble spec <span class='mono'>{night.spec}</span>, lucy's variant "
        "of <span class='mono'>workloads/decalibrate_chip.yaml</span> (no readout-amplitude kick, no flux resets: lucy's qubits "
        "have no flux line), with the shared spec's <span class='mono'>state_fields</span> rules.</p>" + night.scramble_note)


# ----------------------------------------------------------------------------- recovery against IQCC, per qubit
# make_n12_summary.recovery_cell without the sweet spot, and with the readout amplitude as a value without a verdict: lucy's
# scramble leaves it alone, so "back in its window" would be no evidence of a calibration.
REC_COLS = [
    ("ro_f", "readout f", "Δ MHz", "graded"), ("f01", "f₀₁", "Δ MHz", "graded"), ("x180", "x180 amp", "×", "graded"),
    ("anh", "anharm.", "Δ MHz", "ungraded"), ("drag", "DRAG α", "Δ", "ungraded"),
    ("ro_amp", "readout amp", "×", "physics"), ("t1", "T1", "×", "physics"), ("t2e", "T2 echo", "×", "physics"),
    ("rof", "readout fid.", "Δ pp", "physics"),
]
VERDICT = [k for k, _, _, grp in REC_COLS if grp != "physics"]


def recovery_cell(work: Path, cell: Path, q: str) -> dict:
    """One lucy qubit-run's table cells, each (state, text, hover), as make_n12_summary.recovery_cell."""
    _at, _num = summ._at, summ._num
    night = work.name.split("-")[0]
    plan = {e["path"]: e for e in json.load(open(work / "edit_plan.json"))["edits"]}
    lab = json.load(open(work / "source-state/state.json"))["qubits"][q]
    try:
        fin = json.load(open(cell / "quam_state/state.json"))
    except Exception:  # noqa: BLE001
        fin = {}
    res = json.load(open(cell / "result.json"))
    run_id = res.get("run_id") or ""
    written = {e.get("path") for e in rep.events(run_id, q) if e.get("kind") == "state_write"}
    base = f"/qubits/{q}/"
    out = {"run_id": run_id, "night": night}

    def state_of(path, ok):
        v = _at(fin, base + path)
        if v is summ.MISSING or v is None:
            return "none"
        if base + path not in written and (base + path) in plan and v == plan[base + path].get("new"):
            return "scr"
        return "ok" if ok else "bad"

    def show(x, unit_scale=1.0, unit="", fmt="{:.6g}"):
        return "—" if x is summ.MISSING or x is None else (fmt.format(x / unit_scale) + unit if _num(x) else str(x))

    def hover(label, path, ref, unit_scale=1.0, unit="", fmt="{:.6g}"):
        e = plan.get(base + path)
        v = _at(fin, base + path)
        return (f"{night} · {label}: IQCC {show(ref, unit_scale, unit, fmt)}"
                + (f" · scrambled {show(e.get('new'), unit_scale, unit, fmt)}" if e else " · not scrambled")
                + f" · final {show(v, unit_scale, unit, fmt)}" + ("" if base + path in written else " · never written"))

    graded = {"ro_f": ("resonator/RF_frequency", "readout frequency"), "f01": ("f_01", "f_01"),
              "x180": ("xy/operations/x180_DragCosine/amplitude", "x180 amplitude")}
    for key, (path, label) in graded.items():
        e = plan[base + path]
        b, ref, v = e["ballpark"], e["current"], _at(fin, base + path)
        if not _num(v):
            out[key] = ("none", "—", hover(label, path, ref))
            continue
        if "absolute" in b:
            ok = abs(v - ref) <= b["absolute"]
            text, window = f"{(v - ref) / 1e6:+.{1 if abs(v - ref) >= 10e6 else 2}f}", f"±{b['absolute'] / 1e6:g} MHz"
            h = hover(label, path, ref, 1e9, " GHz", "{:.5f}")
        else:
            lo, hi = b["factor_range"]
            ok = lo <= v / ref <= hi
            text, window = f"{v / ref:.2f}×", f"{lo:g}–{hi:g}× the lab's"
            h = hover(label, path, ref)
        out[key] = (state_of(path, ok), text, h + f" · judge window {window}")

    anh_lab, anh = lab.get("anharmonicity"), _at(fin, base + "anharmonicity")
    if _num(anh) and _num(anh_lab):
        out["anh"] = (state_of("anharmonicity", abs(anh - anh_lab) <= summ.ANH_WINDOW), f"{(anh - anh_lab) / 1e6:+.1f}",
                      hover("anharmonicity", "anharmonicity", anh_lab, 1e6, " MHz", "{:.1f}")
                      + f" · window ±{summ.ANH_WINDOW / 1e6:g} MHz")
    else:
        out["anh"] = ("none", "—", hover("anharmonicity", "anharmonicity", anh_lab, 1e6, " MHz", "{:.1f}"))
    a_path = "xy/operations/x180_DragCosine/alpha"
    a_lab, a = _at(lab, a_path), _at(fin, base + a_path)
    if _num(a) and _num(a_lab):
        tol = max(0.25, 0.5 * abs(a_lab))
        out["drag"] = (state_of(a_path, abs(a - a_lab) <= tol), f"{a - a_lab:+.2f}",
                       hover("DRAG α", a_path, a_lab, fmt="{:.3f}") + f" · window ±{tol:.2f} (±50 % of the lab's, at least ±0.25)")
    else:
        out["drag"] = ("none", "—", hover("DRAG α", a_path, None if a_lab is summ.MISSING else a_lab, fmt="{:.3f}"))

    r_path = "resonator/operations/readout/amplitude"
    r_lab, r = _at(lab, r_path), _at(fin, base + r_path)
    if _num(r) and _num(r_lab) and r_lab:
        h = hover("readout amplitude", r_path, r_lab, fmt="{:.4g}") + " · not scrambled on lucy, so no verdict"
        out["ro_amp"] = (("info", f"{r / r_lab:.2f}×", h) if base + r_path in written else ("none", "kept", h))
    else:
        out["ro_amp"] = ("none", "—", f"{night} · readout amplitude: no value")

    for key, path, label in (("t1", "T1", "T1"), ("t2e", "T2echo", "T2 echo")):
        ref, v = lab.get(path), _at(fin, base + path)
        if not (_num(v) and _num(ref) and ref):
            out[key] = ("none", "—", f"{night} · {label}: IQCC {ref!r} · final {None if v is summ.MISSING else v!r}")
            continue
        h = f"{night} · {label}: IQCC {ref * 1e6:.1f} µs · final {v * 1e6:.4g} µs"
        if base + path not in written:
            out[key] = ("none", "kept", h + " · not remeasured, still the lab's value (the scramble keeps T1)")
            continue
        ratio = v / ref
        bad = not summ.IMPLAUSIBLE[0] <= ratio <= summ.IMPLAUSIBLE[1]
        out[key] = ("bad" if bad else "info", f"{ratio:.2f}×",
                    h + (f" · outside {summ.IMPLAUSIBLE[0]:g}–{summ.IMPLAUSIBLE[1]:g}× the lab's: not drift" if bad else ""))
    ro = res["targets"][0].get("quality", {}).get("readout", {}).get("assignment_fidelity") if res.get("targets") else None
    cm = lab.get("resonator", {}).get("confusion_matrix")
    ro_lab = (float(cm[0][0]) + float(cm[1][1])) / 2 if cm else None
    if _num(ro) and _num(ro_lab):
        out["rof"] = ("info", f"{100 * (ro - ro_lab):+.1f}",
                      f"{night} · readout assignment fidelity: IQCC {100 * ro_lab:.1f} % · measured {100 * ro:.1f} %")
    else:
        out["rof"] = ("none", "—", f"{night} · readout assignment fidelity: no value")
    out["right"] = sum(1 for k in VERDICT if out[k][0] == "ok")
    return out


_recovery_table = summ.recovery_table


def recovery_table(rows, night: summ.Night) -> str:
    """make_n12_summary.recovery_table over lucy's columns, its wording made lucy's."""
    html = _recovery_table(rows, night)
    n = len(VERDICT)
    edits = [
        ("<span class='u'>of 7</span>", f"<span class='u'>of {n}</span>", 1),
        ("physics, no verdict", "no verdict", 1),
        ("<span><span class='rc bias'>−118.6</span> f₀₁ outside, but the right line for the bias the run left the qubit at</span>", "", 1),
        ("<span><span class='rc bad'>▲</span> f₀₁ above the lab's: the lab's stored bias is not the top of its arc</span>", "", 1),
        ("the first four. Sweet spot: the lab's own arc (its f₀₁, anharmonicity and φ0 voltage) evaluated at the run's final "
         f"bias must sit within ±{summ.FLUX_WINDOW / 1e6:g} MHz of the lab's f₀₁, so a sweet spot a period away counts and a mV means "
         "the same on every chip.",
         "the first three. lucy has no flux line, so there is no sweet spot to find; the readout amplitude is not scrambled on lucy, "
         "so it is shown without a verdict.", 1),
    ]
    for old, new, count in edits:
        assert html.count(old) == count, (old[:60], html.count(old))
        html = html.replace(old, new)
    # each run's "right" count follows its run link; the footer's per-column counts are out of the qubits, not the columns
    pattern = re.compile(r"(</a></td><td class='g0'><span class='rc cnt'>\d+)/7</span>")
    runs = sum(1 for r in rows for _ in summ.RECOVERY_NIGHTS) - html.count(">no run<")
    html, marks = pattern.subn(rf"\1/{n}</span>", html)
    assert marks == runs, (marks, runs)
    return html


def configure() -> None:
    rep.BACKENDS = (*rep.BACKENDS, "lucy")
    rep.DEV_MARK["lucy"] = "l"
    rep.ORDER["lucy"] = len(rep.ORDER)
    rep.NIGHTS = {"new": ("lucy3",)}
    rep.RECIPE["new"] = RECIPE
    rep.NOT_MEANINGFUL, rep.NOT_COMPLETED = {}, {}
    summ.RECOVERY_NIGHTS = (("lucy1", LUCY1), ("lucy2", LUCY2), ("lucy3", STAMP))
    summ.REC_COLS, summ.VERDICT = REC_COLS, VERDICT
    summ.recovery_cell, summ.recovery_table, summ.scramble_table = recovery_cell, recovery_table, scramble_table


def context(s: dict) -> str:
    a, b = (n13summ._minute(t) for t in s["span"])
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · all {s['n']} qubits of lucy, IQCC's fixed-frequency coaxmon ring, with the "
            "resonator's line shape (a transmission peak) in the state and the resonator nodes reading it · tinycal with "
            "qwen3.8-27b on OpenRouter")


LUCY3 = summ.Night(
    label="lucy3", stamp=STAMP, out=OUT,
    title="lucy3 Summary",
    description="lucy3, 5 Oct 2026: all seven single-qubit bring-ups on IQCC's lucy (fixed-frequency coaxmons) with the "
                "resonator's line shape in the state, tinycal and qwen3.8-27b on OpenRouter, in one table, three plots and a "
                "per-qubit check against IQCC beside the 30 Sep runs.",
    eyebrow="lucy3 rerun",
    h1="lucy3 single-qubit bring-ups",
    context=context,
    chip_sub=lambda s: f"5 Oct · {s['n']} qubits",
    rb_note=n13summ.rb_note,
    pull="the pull the run started from (5 Oct 14:37, unchanged since 30 Sep: cloud state 59619, its qubit values from IQCC's "
         "15 Aug upload)",
    pull_short="the pull the run started from",
    recovery_intro="One qubit per three rows: <b>lucy1</b> and <b>lucy2</b>, the 30 Sep runs (qA–qC, then qC again and qD–qG) "
                   "whose power sweep read lucy's transmission peaks as dips, and <b>lucy3</b>, 5 Oct, with the line shape in "
                   "the state, each against the IQCC pull its own run started from (the same cloud state all three times).",
    recovery_sort=("lucy3", "lucy2", "lucy1"),
    recovery_sort_text="Rows are sorted by how many came back in lucy3, fewest first, then by lucy2's and lucy1's.",
    spec=SPEC,
    scramble_note="<p class='small muted'>New in this run: after the scramble, every resonator's "
                  "<span class='mono'>extras.line_shape</span> is set to <span class='mono'>peak</span> in both the source and "
                  "the scrambled state (lucy reads each resonator in transmission on a line of its own, so |S| peaks at "
                  "resonance). IQCC's state does not carry the field; the resonator nodes read it, and the benchmark profile "
                  "makes it read-only.</p>",
)


if __name__ == "__main__":
    configure()
    summ.build(LUCY3)
