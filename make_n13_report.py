#!/usr/bin/env python3
"""Build the 30 Sep 2026 report: the n13 validation run, a rerun of n12's 37 single-qubit bring-ups (tinycal + qwen3.8-27b via
OpenRouter) with the five overnight fixes for n12's problems combined and applied uncommitted, compared qubit by qubit with n12.

    python3 make_n13_report.py

Numbers come from each cell's result.json (projected from tinycal's events.jsonl, stamped by qab inspect-state / validate / accept),
the cell's final quam_state and its work dir's source-state (the lab's calibration), the tinycal transcripts (events.jsonl: node
runs, proposals, writes, write-guard refusals) and the two schedulers' logs. The per-qubit rows are the same as
~/qab-runs/n13-compare.py's and the aggregates the same as n13-summary.py's. Hand-written text is limited to the operator notes
below (outcome reasons, the per-fix summaries taken from each fix's REPORT.md, follow-ups); every number in it was checked against
the fix directories under ~/qab-runs/n13/.

The output is page content for a claude.ai artifact: a <title>, the stylesheet and the body, with no <html>/<head>/<body> tags.
CSS and small helpers are shared with make_fwcmp2_report.py.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_fwcmp2_report import CSS, esc, median  # noqa: E402

HOME = Path.home()
QAB = HOME / "qab-runs"
N13 = QAB / "n13"
TINY_RUNS = HOME / "code/QM/tinycal/runs"
OUT = Path(__file__).with_name("2026-09-30-n13-fixes-validation-qwen-tinycal.html")
NIGHTS = {"n12": "20260929-1855", "n13": "20260930-0222"}
BACKENDS = ("qolab", "arbel", "gilboa")
DEV = {"qolab": "q", "arbel": "a", "gilboa": "g"}
KEY = "qwen3-8-27b-openrouter"
TARGETS = [("arbel", "qB4"), ("gilboa", "qC3"), ("qolab", "Q6"), ("qolab", "Q4"), ("arbel", "qD1")]
RB_VALID = 1.0  # decay lengths: below this the RB error per Clifford is not a measurement

# The bring-up graph tinycal ran (bringup_recipes/flux_tunable_1q.md at 9a69868): 18 nodes. The recipe runs the power sweep
# twice (step 4 repeats it at the sweet spot); T1 runs once (step 8 only when T1_chirp stored nothing).
GRAPH = ["resonator_identification", "resonator_spectroscopy_vs_power", "resonator_spectroscopy_vs_flux", "qubit_spectroscopy",
         "T1_chirp", "qubit_spectroscopy_vs_flux", "qubit_spectroscopy_fine", "power_rabi", "readout_power_optimization",
         "readout_frequency_optimization", "IQ_blobs", "ramsey_vs_flux_calibration", "power_rabi_error_amplification_x180", "ramsey",
         "T1", "T2echo", "DRAG_calibration", "Randomized_benchmarking"]
EXPECTED_RUNS = {n: 1 for n in GRAPH} | {"resonator_spectroscopy_vs_power": 2}
INFRA_ERRORS = ("Service Unavailable", "ReadTimeout")  # IQCC 503s and HTTP read timeouts: transient, not the agent's
WRITE_TOL = 1e-4  # relative: the model retypes proposals and rounds them (7126196533.9 -> 7126200000)

# ----------------------------------------------------------------------------- operator notes (by hand, checked against the data)
NOT_MEANINGFUL = {  # (night, backend, qubit) -> why a completed run's results are not a calibration
    ("n12", "gilboa", "qC3"): "T1 written as 0.0299774 s (1000×): 150 ms thermal waits, shots cut to 5–20, garbage fits",
    ("n12", "qolab", "Q6"): "parked at the edge of 02c's ±0.5 V window (0.48 V; lab 0.747 V): f_01 118.6 MHz low",
    ("n13", "arbel", "qA5"): "DRAG refused every sweep; the agent hand-wrote α −6.5, then −6.45, and looped",
}
NOT_COMPLETED = {  # (night, backend, qubit) -> short cause
    ("n12", "arbel", "qB4"): "escalated at the 150-turn cap: readout on 02b's ripple (50.8 %), qubit never found",
    ("n12", "arbel", "qD1"): "failed, turn budget exhausted: no 0→1 line found, chased a noise spike",
}
ROW_NOTE = {  # (backend, qubit) -> note on the n13 row
    ("qolab", "Q1"): "the lab's own 03:03 recalibration also dropped (99.911 → 99.837 %)",
    ("qolab", "Q2"): "now at the lab's flux; n12 parked 0.05 V off",
    ("qolab", "Q3"): "f_01 −24.5 MHz from the lab's every night; the lab value is suspect",
    ("qolab", "Q4"): "one RB run at the preset depth, no override",
    ("qolab", "Q6"): "02c failed, its retry window found 0.747 V",
    ("arbel", "qA5"): "hand-written DRAG α −6.45 (lab −0.96); 40 failed DRAG/RB runs",
    ("arbel", "qA6"): "upper sweet spot, 234 MHz above the lab's parking point",
    ("arbel", "qB3"): "upper sweet spot, 33.5 MHz above the lab's point",
    ("arbel", "qB4"): "02b now tracks the resonance, not the ripple",
    ("arbel", "qC3"): "320 ns x180 at 0.285; lab 96 ns at 0.99",
    ("arbel", "qC5"): "120 ns x180 at 0.29; lab 48 ns at 0.80",
    ("arbel", "qD1"): "at its sweet spot, 368.5 MHz above the lab's parking point",
    ("arbel", "qD3"): "48 ns x180 (n12 120 ns)",
    ("gilboa", "qC1"): "DRAG does not converge, every night",
    ("gilboa", "qC3"): "T1 written in seconds this time",
    ("gilboa", "qC4"): "poor every night",
    ("gilboa", "qD2"): "T1 ≈ 1 µs every night",
    ("gilboa", "qD5"): "03a's next window was past the drive's reach (fixed after the run)",
}

FIXES = [  # one row per n12 problem; numbers from ~/qab-runs/n13/fix-<id>/REPORT.md
    {
        "key": ("arbel", "qB4"), "dir": "fix-qB4",
        "problem": "02b proposed no readout power on any qB4 map from n7 to n12. With no proposal it still reported "
                   "<span class='mono'>dressed_frequency</span> = 7.74685 GHz with no power attached, and a warning called the "
                   "right frequency wrong. The agent committed that point: IQ_blobs 52.5 % there, 96.9 % at the lab's.",
        "cause": "A broad, power-independent trough about 3 MHz above the 0.7 MHz-wide resonance (not the IF-0 notch). The "
                 "whole-window Lorentzian on a linear background fitted the trough (5–6.6 MHz wide), so the tracked line never moved "
                 "with power.",
        "fix": "qua-libs 02b analysis only. When the whole-window fit is wider than 3 linewidths, refit locally on ±3 linewidths. "
               "The track opens from the median of the first 5 rows. The proposed frequency comes from a local Lorentzian at the "
               "proposed power. Every frequency carries its power: <span class='mono'>frequency_power_dbm</span>, "
               "<span class='mono'>dressed_power_max_dbm</span>, <span class='mono'>line_frequency_vs_power</span>.",
        "evidence": "<b>Replay</b>, 232 maps from n7–n12, 37 qubits: proposals 194 → 212, 179 of 193 at the same power, "
                    "<span class='mono'>dressed_frequency</span> more than 1.5 MHz off the lab 26 → 4. One proposal lost (gilboa qD5, n9). "
                    "10 of 12 qB4 maps now propose, all at 7.7442 ± 0.05 GHz. "
                    "<b>Live</b>, arbel: qB4 proposes −22.3 dBm at 7.744171 GHz; controls qB5 and qC2 unchanged. "
                    "<b>Bring-up</b>: completed, 4/4, RB 0.133 %, 19 min. qua-libs tests 537 (+7).",
    },
    {
        "key": ("gilboa", "qC3"), "dir": "fix-qC3",
        "problem": "The agent wrote T1_chirp's 2.99774e-05 s as 0.0299774. Thermal waits became 150 ms a shot. The pre-flight "
                   "lumped the terms and advised cutting shots, and power_rabi and 08b timed out bare. Every later fit ran on "
                   "5–20 shots.",
        "cause": "write_state never checked magnitude. In 401 archived transcripts the replay found 37 unit slips (all qwen3.8-27b), "
                 "33 transcripts affected; 3 survived into a graded state.",
        "fix": "As run in n13: a tinycal write guard (a range check, plus a 10<sup>k</sup> (|k| ≥ 3) slip check against the "
               "node's proposal; refused unless re-sent with confirm). A qua-libs pre-flight with a per-term breakdown, an outright "
               "refusal above a 25 ms reset wait, and coverage of 04b/04c/07/08a/08b. A tinycal timeout hint naming the stored T1. "
               "<span class='tag'>superseded since</span>: see <a href='#since'>since then</a>.",
        "evidence": "<b>Write replay</b>, 12 406 writes in 401 transcripts: 39 refused, 37 of them real slips, none on Claude, "
                    "Ising or qwen-flash. <b>Pre-flight replay</b>: 0 of 639 completed runs refused. <b>Live</b>, gilboa qC3: the "
                    "slipped T1 is refused before the job; the lab's T1 runs (08b estimated 13 s, measured 13.2 s). <b>Model "
                    "replay</b>: after the guard's refusal the model fixed T1 in 8/8 samples; after the original bare timeout, 0/4.",
    },
    {
        "key": ("qolab", "Q6"), "dir": "fix-Q6",
        "problem": "02c saw the arc's apex beyond the ±0.5 V window (apex_in_sweep False, side high) but reported "
                   "\"successful\" with no retry parameters. The agent committed the edge column (0.48 V; lab 0.747 V).",
        "cause": "Reporting, not detection. The node named the edge column <span class='mono'>measured_apex_offset</span>, printed an "
                 "unresolved 3.9 V phi0, and left the port's real range (±2.5 V, amplified LF-FEM) unsaid; the model took ±0.5 V "
                 "for the port limit.",
        "fix": "02c fails when it has no idle offset and returns <span class='mono'>retry_parameters</span> for a wider window "
               "(up to 1.5 V per retry), capped by the port class: ±0.5 V direct, ±2.5 V amplified, ±0.5 V when the class is "
               "unknown. phi0 is NaN below 0.75 period coverage. One recipe sentence in step 3.",
        "evidence": "<b>Replay</b>, 115 maps from n9–n12: idle offset identical in 115/115. The 14 qolab edge maps now retry, "
                    "every window holding the lab's sweet spot; the widest retry reached −2.2 V. <b>Live</b>: Q6 failed, then its "
                    "retry found 0.7447 V (lab 0.7473); control Q5 0.0178 V (lab 0.0189), no retry. <b>Bring-up</b>: 0.7471 V, RB "
                    "0.269 %, 4/4, 20 min.",
    },
    {
        "key": ("qolab", "Q4"), "dir": "fix-Q4",
        "problem": "The agent cut RB's depth to 128; the node raised <span class='mono'>KeyError: 'fit_results'</span> and lost "
                   "the data. At depth 64 it reported 0.778 % over \"1.00 decay lengths\", implying P(0|1) = 0.83 beside a 92 % "
                   "readout.",
        "cause": "<span class='mono'>fit_decay_exp</span> starts at a = 0 on log sweeps of ≤ 10 depths. The failure was "
                 "swallowed without fit_results, and update_state raised before save_results. A free fit on a barely bent curve "
                 "measures its own coverage (true coverage about 0.09 decay lengths).",
        "fix": "A robust fit with fallback starts. Unphysical discriminated fits (A ≤ 0, B ∉ [0, 1], B − A ≥ 0.5) are \"not a "
               "measurement\". An anchored fit (floor 0.5) sizes <span class='mono'>recommended_max_circuit_depth</span>. The node "
               "always saves. One recipe sentence: keep the preset depth.",
        "evidence": "<b>Replay</b>, 120 full datasets and 408 truncations: crashes 35 → 0; 95 good runs bit-identical; 7 flip to "
                    "failed, all dubious; truncation error at the 90th percentile 173 % → 30 %. <b>Live</b>, Q4: the depth-128 "
                    "KeyError reproduced and is now a clean failure naming depth ≥ 1024; depth 2048 identical. qua-libs tests 542.",
    },
    {
        "key": ("arbel", "qD1"), "dir": "fix-qD1",
        "problem": "The lab parks qD1 at 0.227 V, 197 mV off its sweet spot. At the arc maximum the line is at 5.366 GHz, 419 MHz "
                   "above the scrambled f_01. 03a never searched there (n12's widest scan stopped at 5.298 GHz). The fine node then "
                   "called a lone 5σ sample \"UNRESOLVED BUT REAL\", and the agent chased it for 120 turns.",
        "cause": "The qubit works (lab state: chirp line 4.9944 GHz, readout 95.2 %). The search had no rule for where to look "
                 "next, and the fine node judged an unresolved fit by its extrapolated SNR.",
        "fix": "03a returns <span class='mono'>retry_parameters</span>/<span class='mono'>next_window</span>, tiling 675 MHz windows "
               "upward to resonator − 300 MHz, then one below, then \"exhausted\" (escalate). The fine node judges an unresolved "
               "fit by its measured excursion. Recipe steps 3 and 5. The drive-reach limit came later "
               "(<span class='mono'>fix-qD1/followup.patch</span>, not in n13).",
        "evidence": "<b>Replay</b>, 225 chirp runs: 0 suggestions on 90 identified lines; the first suggested window holds "
                    "5.366 GHz in every qD1 run. Fine node, 162 runs: only qD1's spike changes (snr 52.5 → 4.8). <b>Live</b>, arbel: "
                    "at the arc maximum the suggested window finds 5.36627 GHz with its two-photon partner. <b>Bring-up</b>: "
                    "completed in 63 turns, line at turn 14, readout 98.0 %, RB 0.175 %, 19 min.",
    },
]

FOLLOWUPS = [  # from ~/qab-runs/n13-LOG.md, "Follow-ups found tonight"
    "<b>DRAG overrides.</b> When DRAG refuses, the agent can hand-pick an α (arbel qA5), and nothing checks it against RB. "
    "Candidates: RB flags a curve that decays within its first depths (\"the rate is a lower bound\"); the recipe says to keep "
    "the default α when DRAG refuses.",
    "<b>Long x180s on arbel.</b> On qC3 and qC5 the agent lengthens the pulse (320 and 120 ns) instead of using the amplitude "
    "range. The lab runs 96 ns at 0.99 and 48 ns at 0.80.",
    "<b>Drive reach in the 03a tiling.</b> Hit by gilboa qD5 at turn 17. Fixed offline in "
    "<span class='mono'>fix-qD1/followup.patch</span>: the tiling respects the drive's ±850 MHz reach, and all 137 archived "
    "suggestions are playable. Applied in <span class='mono'>~/code/QM/qua-libs</span>, not in the n13 run.",
    "<b>Projection gaps.</b> <span class='mono'>qab accept</span> gets no RB confidence interval, B offset or readout "
    "assignment fidelity from tinycal runs, so it prints OUT OF SPEC on every cell.",
]


# ----------------------------------------------------------------------------- collect
def read_events(run_id: str, q: str) -> list[dict]:
    p = TINY_RUNS / run_id / q / "events.jsonl"
    out = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def numeric(e: dict, key: str):
    v = (e.get("numerics") or {}).get(key)
    return v.get("value") if isinstance(v, dict) else v


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def same_value(a, b) -> bool:
    if is_num(a) and is_num(b):
        return a == b or abs(a - b) <= WRITE_TOL * max(abs(a), abs(b))
    return a == b


def straight_through(ev: list[dict], status: str) -> dict:
    """Did the agent add nothing? Completed; every graph step run once (the power sweep twice) with no failed or refused run;
    every write equal to the latest proposal for its path; no write to an unproposed path; no proposal left unwritten
    unless it was a no-op. IQCC 503s and HTTP read timeouts are not runs; neither is a malformed call with no node name."""
    runs, infra, malformed = [], 0, 0
    for e in ev:
        if e.get("kind") != "tool_call" or e.get("tool") != "run_node":
            continue
        err = e.get("error") or ""
        if e.get("status") == "failed" and any(s in err for s in INFRA_ERRORS):
            infra += 1
            continue
        if e.get("status") == "rejected" and not (e.get("node") or g(e, "arguments", "node")):
            malformed += 1
            continue
        runs.append(e)
    count = collections.Counter(e.get("node") for e in runs)
    failed = [e for e in runs if not (e.get("status") == "completed" and e.get("outcome") == "successful")]
    extra = {n: c - EXPECTED_RUNS.get(n, 0) for n, c in count.items() if c > EXPECTED_RUNS.get(n, 0)}
    skipped = [n for n in GRAPH if count.get(n, 0) < EXPECTED_RUNS[n]]
    ledger, pending = {}, {}
    unproposed, differ, unwritten, param_runs, param_names = [], [], [], 0, collections.Counter()
    for e in ev:
        if e.get("kind") != "tool_call":
            continue
        if e.get("tool") == "run_node" and e.get("status") == "completed":
            params = g(e, "arguments", "parameters") or {}
            if params:
                param_runs += 1
                param_names.update(params.keys())
            for pu in e.get("proposed_updates") or []:
                path = pu["path"]
                if path in pending:  # superseded before it was written
                    v, cur = pending.pop(path)
                    if not same_value(v, cur):
                        unwritten.append(path)
                ledger[path] = pu.get("proposed")
                pending[path] = (pu.get("proposed"), pu.get("current"))
        elif e.get("tool") == "write_state":
            for u in g(e, "arguments", "updates") or []:
                path = u.get("path")
                if path not in ledger:
                    unproposed.append(path)
                elif not same_value(u.get("value"), ledger[path]):
                    differ.append(path)
                else:
                    pending.pop(path, None)
    unwritten += [p for p, (v, cur) in pending.items() if not same_value(v, cur)]
    reasons = []
    if status != "completed":
        reasons.append(status)
    if failed:
        reasons.append(f"{len(failed)} failed or refused run{'s' * (len(failed) != 1)}")
    if extra:
        reasons.append("reran " + ", ".join(f"{n} +{c}" for n, c in extra.items()))
    if skipped:
        reasons.append("skipped " + ", ".join(skipped))
    if unproposed:
        reasons.append(f"{len(unproposed)} write{'s' * (len(unproposed) != 1)} no node proposed")
    if differ:
        reasons.append(f"{len(differ)} write{'s' * (len(differ) != 1)} off the proposal")
    if unwritten:
        reasons.append(f"{len(unwritten)} proposal{'s' * (len(unwritten) != 1)} not written")
    tail = lambda p: p.split("/", 3)[-1]  # noqa: E731
    return {"ok": not reasons, "reasons": reasons, "infra": infra, "malformed": malformed,
            "no_reruns": status == "completed" and not failed and not extra and not skipped,
            "accepted": not unproposed and not differ and not unwritten,
            "unproposed": [tail(p) for p in unproposed], "param_runs": param_runs, "param_names": sorted(param_names),
            "runs": len(runs)}


def guard_refusals(ev: list[dict]) -> list[dict]:
    """The write guard's unit-slip refusals, each with whether the next write to that path was the node's value."""
    out = []
    writes = [(i, e) for i, e in enumerate(ev) if e.get("kind") == "tool_call" and e.get("tool") == "write_state"]
    for k, (i, e) in enumerate(writes):
        for r in e.get("results") or []:
            reason = r.get("reason") or ""
            if r.get("status") != "refused" or "unit slip" not in reason:
                continue
            m = re.search(r"this value is ([0-9.e+\-]+) times", reason)
            fixed = None
            for _, e2 in writes[k + 1:]:
                hit = [x for x in e2.get("results") or [] if x.get("path") == r["path"]]
                if hit:
                    fixed = hit[0].get("status") == "written" and not (hit[0].get("reason"))
                    break
            out.append({"turn": e.get("turn"), "path": r["path"].split("/", 3)[-1], "value": r.get("value"),
                        "factor": float(m.group(1)) if m else None, "fixed": fixed})
    return out


def cell_row(night: str, backend: str, q: str) -> dict:
    work = QAB / f"{night}-{backend}-{NIGHTS[night]}"
    cdir = work / f"{backend}-tinycal-{KEY}-{q}"
    r = json.load(open(cdir / "result.json"))
    x = r["targets"][0]
    run_id = r.get("run_id") or ""
    ev = read_events(run_id, q)
    runs = [e for e in ev if e.get("tool") == "run_node" and e.get("kind") == "tool_call"]
    rb_runs = [e for e in runs if e.get("node") == "Randomized_benchmarking"]
    lab = json.load(open(work / "source-state/state.json"))["qubits"][q]
    final = json.load(open(cdir / "quam_state/state.json"))["qubits"][q]
    x180 = g(final, "xy", "operations", "x180_DragCosine") or {}
    lab_x180 = g(lab, "xy", "operations", "x180_DragCosine") or {}
    qual = x.get("quality") or {}
    t = g(x, "agent", "time") or {}
    judge = g(x, "judge", "identity") or {}
    row = {
        "night": night, "backend": backend, "q": q, "run_id": run_id, "status": x.get("status"),
        "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
        "rb": g(qual, "rb", "error_per_clifford"), "rb_cover": numeric(rb_runs[-1], "decay_lengths_covered") if rb_runs else None,
        "rb_depth": (g(rb_runs[-1], "arguments", "parameters") or {}).get("max_circuit_depth") if rb_runs else None,
        "readout": g(qual, "readout", "assignment_fidelity"), "t1": g(qual, "coherence_limit", "t1_s"),
        "t2e": g(qual, "coherence_limit", "t2echo_s"),
        "x180_amp": x180.get("amplitude"), "x180_len": x180.get("length"), "alpha": x180.get("alpha"),
        "lab_x180_amp": lab_x180.get("amplitude"), "lab_x180_len": lab_x180.get("length"), "lab_alpha": lab_x180.get("alpha"),
        "ro_amp": g(final, "resonator", "operations", "readout", "amplitude"),
        "lab_ro_amp": g(lab, "resonator", "operations", "readout", "amplitude"),
        "df01": (final["f_01"] - lab["f_01"]) / 1e6 if final.get("f_01") and lab.get("f_01") else None,
        "joint": g(final, "z", "joint_offset"), "lab_joint": g(lab, "z", "joint_offset"),
        "lab_fid": g(lab, "gate_fidelity", "averaged"),
        "qpu": t.get("qpu_execution_s"), "wall": t.get("total_s") or g(r, "totals", "time", "total_s"), "model": t.get("model_s"),
        "turns": g(x, "agent", "turns", "total"),
        "failed": sum(1 for e in runs if e.get("outcome") != "successful"), "node_runs": len(runs),
        "ballpark": judge.get("in_ballpark"), "graded": judge.get("graded"), "outside": judge.get("outside") or [],
        "cost": g(r, "totals", "tokens", "estimated_cost_usd") or 0.0,
        "reason": x.get("escalation_reason") or "",
        "st": straight_through(ev, x.get("status")), "guard": guard_refusals(ev),
        "flux_windows": [(e.get("turn"), e.get("outcome"), g(e, "arguments", "parameters") or {}) for e in runs
                         if e.get("node") == "resonator_spectroscopy_vs_flux"],
        "grade_out_of_spec": "OUT OF SPEC" in (cdir / "grade.log").read_text(errors="replace") if (cdir / "grade.log").exists() else None,
    }
    row["valid_rb"] = row["rb"] is not None and (row["rb_cover"] is None or row["rb_cover"] >= RB_VALID)
    row["outcome"], row["why"] = outcome(row)
    return row


def outcome(r: dict) -> tuple[str, str]:
    k = (r["night"], r["backend"], r["q"])
    if r["status"] != "completed":
        return "bad", NOT_COMPLETED.get(k, r["reason"][:140] or r["status"])
    if k in NOT_MEANINGFUL:
        return "warn", NOT_MEANINGFUL[k]
    return "ok", ""


def collect() -> dict[str, dict]:
    nights = {}
    for night, stamp in NIGHTS.items():
        rows = {}
        for backend in BACKENDS:
            work = QAB / f"{night}-{backend}-{stamp}"
            for cdir in sorted(work.glob(f"{backend}-tinycal-{KEY}-*")):
                if (cdir / "result.json").exists():
                    q = cdir.name.rsplit("-", 1)[1]
                    rows[(backend, q)] = cell_row(night, backend, q)
        nights[night] = rows
    return nights


def scheduler_span(night: str) -> tuple[str, str, float]:
    lines = (QAB / f"{night}-scheduler.log").read_text().splitlines()
    t0 = next(l[1:9] for l in lines if "launched" in l)
    t1 = next(l[1:9] for l in lines if "cells finished" in l)
    a, b = datetime.strptime(t0, "%H:%M:%S"), datetime.strptime(t1, "%H:%M:%S")
    secs = (b - a).total_seconds() % 86400
    return t0[:5], t1[:5], secs


def summary(rows: dict, night: str) -> dict:
    rs = list(rows.values())
    done = [r for r in rs if r["status"] == "completed"]
    rb = [r["rb"] for r in rs if r["valid_rb"]]
    ro = [r["readout"] for r in rs if r["readout"] is not None]
    qpu = [r["qpu"] or 0 for r in rs]
    wall = [r["wall"] or 0 for r in rs]
    t0, t1, span = scheduler_span(night)
    st = [r for r in rs if r["st"]["ok"]]
    return {
        "cells": len(rs), "completed": len(done), "meaningful": sum(1 for r in rs if r["outcome"] == "ok"),
        "valid_rb": len(rb), "rb_median": median(rb), "ro_median": median(ro), "ro_95": sum(1 for v in ro if v >= 0.95),
        "ballpark": sum(r["ballpark"] or 0 for r in rs), "graded": sum(r["graded"] or 0 for r in rs),
        "qpu_total": sum(qpu) / 60, "qpu_median": median(qpu) / 60, "wall_median": median(wall) / 60,
        "failed": sum(r["failed"] for r in rs), "node_runs": sum(r["node_runs"] for r in rs),
        "cost": sum(r["cost"] for r in rs), "span": span, "t0": t0, "t1": t1,
        "st": st, "st_no_params": [r for r in st if r["st"]["param_runs"] == 0],
        "no_reruns": sum(1 for r in rs if r["st"]["no_reruns"]), "accepted": sum(1 for r in done if r["st"]["accepted"]),
        "infra": sum(r["st"]["infra"] for r in rs), "malformed": sum(r["st"]["malformed"] for r in rs),
        "quad_unproposed": sum(1 for r in done if "freq_vs_flux_01_quad_term" in r["st"]["unproposed"]),
        "out_of_spec": sum(1 for r in rs if r["grade_out_of_spec"]),
    }


# ----------------------------------------------------------------------------- render helpers
def sg(v, spec="{:+.2f}") -> str:
    """A signed number with a real minus sign; a value that rounds to zero loses its sign."""
    if v is None:
        return "—"
    s = spec.format(v)
    if re.fullmatch(r"[+-]0(\.0*)?", s):
        s = s[1:]
    return s.replace("-", "−")


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pow10(f: float | None) -> str:
    return "?" if not f else "10" + str(round(__import__("math").log10(f))).translate(SUP)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def ro_ratio(r: dict) -> float | None:
    return r["ro_amp"] / r["lab_ro_amp"] if r.get("ro_amp") and r.get("lab_ro_amp") else None


def order_key(r):
    return (BACKENDS.index(r["backend"]), r["q"])


def label(outc: str) -> str:
    return {"ok": "completed, meaningful", "warn": "completed, not meaningful", "bad": "not completed"}[outc]


def qtag(r: dict, extra: str = "") -> str:
    """A qubit name coloured by the run's outcome, with a muted device letter; the full backend and the reason on hover."""
    tip = f"{r['backend']} {r['q']} · {r['night']}: {label(r['outcome'])}"
    if r["why"]:
        tip += f": {r['why']}"
    elif r["valid_rb"]:
        tip += f" · RB {100 * r['rb']:.3f} % · readout {100 * (r['readout'] or 0):.1f} %"
    if extra:
        tip += f" · {extra}"
    return (f"<span class='q q-{r['outcome']}' title='{esc(tip)}'><span class='dv' aria-label='{esc(r['backend'])}'>"
            f"{DEV[r['backend']]}</span>{esc(r['q'])}</span>")


def qlist(rows: list[dict], extra=lambda r: "") -> str:
    if not rows:
        return "<span class='muted'>none</span>"
    return "<span class='ql'>" + "".join(qtag(r, extra(r)) for r in sorted(rows, key=order_key)) + "</span>"


def chip(r: dict | None) -> str:
    if r is None:
        return "<span class='muted'>—</span>"
    text = {"ok": "completed", "warn": "not meaningful", "bad": r["status"]}[r["outcome"]]
    tip = r["why"] or label(r["outcome"])
    return f"<span class='oc oc-{r['outcome']}' title='{esc(tip)}'>{esc(text)}</span>"


def pair(old: str, new: str, cls: str = "") -> str:
    return f"<span class='was'>{old}</span><span class='arr'>→</span><b class='{cls}'>{new}</b>"


def rb_text(r: dict) -> str:
    if r["rb"] is None:
        return "—"
    if not r["valid_rb"]:
        return f"<span title='{r['rb_cover']:.2f} decay lengths: not a measurement'>n/a</span>"
    return f"{100 * r['rb']:.3f}"


def mins(s, spec="{:.0f}") -> str:
    return "—" if s is None else spec.format(s / 60)


def st_mark(r: dict) -> str:
    s = r["st"]
    if s["ok"]:
        extra = f"; passed its own parameters on {s['param_runs']} runs ({', '.join(s['param_names'])})" if s["param_runs"] else ""
        return f"<span class='stk yes' title='{esc(r['night'])}: straight through{esc(extra)}'>✓</span>"
    return f"<span class='stk no' title='{esc(r['night'])}: {esc('; '.join(s['reasons']))}'>·</span>"


def judge_text(r: dict) -> str:
    if not r["graded"]:
        return "—"
    tip = "; ".join(o.split(" came back")[0].split(".", 1)[-1] for o in r["outside"]) or "all in ballpark"
    return f"<span title='{esc(tip)}'>{r['ballpark']}/{r['graded']}</span>"


# ----------------------------------------------------------------------------- sections
def summary_table(s12: dict, s13: dict, n12: dict, n13: dict) -> str:
    def pct(v, d=1):
        return "—" if v is None else f"{100 * v:.{d}f} %"

    def hm(secs):
        return f"{int(secs // 3600)} h {int(secs % 3600 // 60):02d} min"

    def st_cell(s, rows):
        n_np = len(s["st_no_params"])
        return (f"<b>{len(s['st'])}/{s['cells']}</b> {qlist(s['st'])}"
                f"<div class='small muted'>{n_np} of {len(s['st'])} also passed no parameters of its own</div>" if s["st"] else
                f"<b>0/{s['cells']}</b> <span class='muted'>none</span>")

    rows = [
        ("Qubits, by outcome", qlist(list(n12.values())), qlist(list(n13.values())), "qubits"),
        ("Completed", f"{s12['completed']}/{s12['cells']}", f"{s13['completed']}/{s13['cells']}", ""),
        ("Completed with meaningful results", f"{s12['meaningful']}", f"{s13['meaningful']}", ""),
        ("Straight through: the agent added nothing", st_cell(s12, n12), st_cell(s13, n13), "qubits"),
        ("Valid RB (≥ 1 decay length), median error per Clifford",
         f"{s12['valid_rb']}, median {100 * s12['rb_median']:.3f} %", f"{s13['valid_rb']}, median {100 * s13['rb_median']:.3f} %", ""),
        ("Readout assignment, median (qubits ≥ 95 %)", f"{pct(s12['ro_median'])} ({s12['ro_95']})", f"{pct(s13['ro_median'])} ({s13['ro_95']})", ""),
        ("Judge: scrambled parameters back in range", f"{s12['ballpark']}/{s12['graded']}", f"{s13['ballpark']}/{s13['graded']}", ""),
        ("QPU total / median per qubit", f"{s12['qpu_total']:.0f} / {s12['qpu_median']:.1f} min", f"{s13['qpu_total']:.0f} / {s13['qpu_median']:.1f} min", ""),
        ("Wall per qubit, median", f"{s12['wall_median']:.0f} min", f"{s13['wall_median']:.0f} min", ""),
        ("Failed node runs, total (of all node runs)", f"{s12['failed']} <span class='small muted'>of {s12['node_runs']}</span>",
         f"{s13['failed']} <span class='small muted'>of {s13['node_runs']}</span>", ""),
        ("Model cost (tinycal estimate)", f"${s12['cost']:.2f}", f"${s13['cost']:.2f}", ""),
        ("Total run, first launch to last cell", f"{hm(s12['span'])} <span class='small muted'>{s12['t0']}–{s12['t1']}</span>",
         f"{hm(s13['span'])} <span class='small muted'>{s13['t0']}–{s13['t1']}</span>", ""),
    ]
    body = "".join(f"<tr><th class='rowh'>{esc(lab)}</th><td class='{'wrap' if kind else 'num'}'>{a}</td>"
                   f"<td class='{'wrap' if kind else 'num'}'>{b}</td></tr>" for lab, a, b, kind in rows)
    head = ("<thead><tr><th></th><th class='grp'>n12 <span class='small muted'>29 Sep, 18:56</span></th>"
            "<th class='grp'>n13 <span class='small muted'>30 Sep, 02:23</span></th></tr></thead>")
    return f"<div class='scroll'><table class='grid pivot sumt'>{head}<tbody>{body}</tbody></table></div>"


def legend() -> str:
    demo = lambda outc, name, dv, tip: (f"<span class='q q-{outc}' title='{esc(tip)}'><span class='dv'>{dv}</span>{name}</span>")  # noqa: E731
    return ("<div class='qlegend'>"
            f"<span>{demo('ok', 'qC1', 'a', 'arbel qC1')} completed, meaningful</span>"
            f"<span>{demo('warn', 'qC3', 'g', 'gilboa qC3')} completed, results not meaningful</span>"
            f"<span>{demo('bad', 'qB4', 'a', 'arbel qB4')} not completed</span>"
            "<span class='muted'>device letter: <b class='mono'>a</b> arbel · <b class='mono'>g</b> gilboa · <b class='mono'>q</b> qolab; "
            "hover a name for the backend and the reason</span></div>")


def st_note(s12: dict, s13: dict, n13: dict) -> str:
    qc1 = next((r for r in s13["st"]), None)
    qc1_txt = ""
    if qc1:
        names = [n for n in qc1["st"]["param_names"]]
        qc1_txt = (f" By the stricter test, no parameters of its own, none qualifies: n13's one straight-through cell, "
                   f"{esc(qc1['backend'])} {esc(qc1['q'])}, passed its own on {qc1['st']['param_runs']} of {qc1['st']['runs']} runs "
                   f"(<span class='mono'>{esc(', '.join(names))}</span>)."
                   if not s13["st_no_params"] else "")
    retry = sorted({r["q"] for r in n13.values() if r["backend"] == "qolab" and any(o == "failed" for _, o, _ in r["flux_windows"])})
    return (
        "<p class='small'><b>Straight through</b> means the agent contributed nothing: the cell completed; every graph node ran "
        "exactly once (the power sweep twice, as the recipe repeats it at the sweet spot) and no node run failed or was refused; and "
        "every value written was the latest proposal for its path. Numbers count within 1×10<sup>−4</sup> relative, because the model "
        "retypes and rounds. No write went to a path no node had proposed, and every proposal was written unless it was a no-op. "
        "All of it is computed from each cell's <span class='mono'>events.jsonl</span>.</p>"
        "<p class='small muted'>How edge cases were treated. A run that failed with IQCC's 503 Service Unavailable or an HTTP read "
        f"timeout is not counted as a run (n12 {s12['infra']}, n13 {s13['infra']}). Neither is a malformed call with no node name, "
        f"which runs nothing (n12 {s12['malformed']}, n13 {s13['malformed']}). A write the harness refused still counts as a write "
        f"off the proposal. A node that fails on purpose with <span class='mono'>retry_parameters</span> still makes its retry a rerun: "
        f"n13's 02c retries on qolab {', '.join(retry)} disqualify those cells. Model parameters on a run_node call are not part of the "
        f"definition; graph presets never count.{qc1_txt} Of the n13 cells, {s13['no_reruns']} ran every node once without a failed run, "
        f"and {s13['accepted']} wrote exactly what the nodes proposed (n12: {s12['no_reruns']} and {s12['accepted']}). The usual miss "
        f"is a write no node proposed. In {s13['quad_unproposed']} n13 bring-ups that is 03b's curvature, which the agent copies into "
        "<span class='mono'>freq_vs_flux_01_quad_term</span> before ramsey_vs_flux proposes it.</p>")


def fixes_table(n12: dict, n13: dict) -> str:
    def n13_outcome(key):
        a, b = n12[key], n13[key]
        bits = [chip(b), f"RB {rb_text(b)} %" if b["rb"] is not None else "", f"readout {100 * b['readout']:.1f} %",
                f"judge {b['ballpark']}/{b['graded']}", f"{mins(b['wall'])} min wall", plural(b['failed'], 'failed run')]
        was = [chip(a), f"RB {rb_text(a)} %" if a["valid_rb"] else "no valid RB", f"readout {100 * a['readout']:.1f} %",
               f"judge {a['ballpark']}/{a['graded']}", f"{mins(a['wall'])} min", f"{a['failed']} failed"]
        extra = ""
        if key == ("qolab", "Q6"):
            extra = (f"f_01 {sg(b['df01'])} MHz, joint {b['joint']:.4f} V (lab {b['lab_joint']:.4f}); n12 f_01 {sg(a['df01'], '{:+.1f}')} MHz.")
        elif key == ("qolab", "Q4"):
            extra = (f"{b['rb_cover']:.1f} decay lengths at depth {b['rb_depth'] or 2048} (preset); n12 {100 * a['rb']:.3f} % over "
                     f"{a['rb_cover']:.2f} at depth {a['rb_depth']}.")
        elif key == ("arbel", "qD1"):
            extra = (f"At its sweet spot. Judge {b['ballpark']}/{b['graded']}: f_01 is {sg(b['df01'], '{:+.1f}')} MHz from the lab's "
                     f"off-sweet-spot value, and the readout amplitude is {ro_ratio(b):.2f}× the lab's.")
        elif key == ("gilboa", "qC3"):
            slips = [r for r in n13.values() if r["guard"]]
            n_slips = sum(len(r["guard"]) for r in slips)
            fixed = sum(1 for r in slips for x in r["guard"] if x["fixed"])
            where = " ".join(qtag(r) for r in sorted(slips, key=order_key))
            extra = (f"The model wrote qC3's T1 in seconds this time. Elsewhere the guard refused {n_slips} unit slips in "
                     f"{len(slips)} cells ({where}); {fixed} of them were rewritten with the node's value on the next write.")
        elif key == ("arbel", "qB4"):
            extra = f"f_01 {sg(b['df01'])} MHz, x180 {b['x180_amp']:.3f}/{b['x180_len']} ns."
        return ("<div class='oc-line'>" + " · ".join(x for x in bits if x) + "</div>"
                + (f"<div class='small'>{extra}</div>" if extra else "")
                + "<div class='small muted'>n12: " + " · ".join(x for x in was if x) + "</div>")

    body = []
    for f in FIXES:
        r = n13[f["key"]]
        body.append(
            f"<tr><td class='tgtcell'>{qtag(r)}<div class='small muted'>{esc(f['key'][0])}</div>"
            f"<div class='mono path' title='~/qab-runs/n13/{esc(f['dir'])}/REPORT.md'>~/qab-runs/n13/<wbr>{esc(f['dir'])}/<wbr>REPORT.md</div></td>"
            f"<td>{f['problem']}<div class='cause'><b>Root cause.</b> {f['cause']}</div></td>"
            f"<td>{f['fix']}</td><td>{f['evidence']}</td><td>{n13_outcome(f['key'])}</td></tr>")
    head = ("<thead><tr><th>qubit, report</th><th>n12 problem and root cause</th><th>fix</th><th>evidence before n13: replay, live, bring-up</th>"
            "<th>n13 outcome</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small fixes'>{head}<tbody>{''.join(body)}</tbody></table></div>"


def per_qubit_table(n12: dict, n13: dict) -> str:
    targets = set(TARGETS)
    body = []
    for backend in BACKENDS:
        keys = sorted(k for k in n13 if k[0] == backend)
        body.append(f"<tr class='gh'><th colspan='12'>{esc(backend)} <span class='muted'>· {len(keys)} qubits</span></th></tr>")
        for key in keys:
            a, b = n12.get(key), n13[key]
            rb_cls = ""
            if a and a["valid_rb"] and b["valid_rb"]:
                ratio = b["rb"] / a["rb"]
                rb_cls = "worse" if ratio > 1.5 else "better" if ratio < 1 / 1.5 else ""
            elif b["valid_rb"] and a and not a["valid_rb"]:
                rb_cls = "better"
            note = ROW_NOTE.get(key, "")
            if b["guard"]:
                slip = "; ".join(f"{x['path']} ×{pow10(x['factor'])}" for x in b["guard"])
                note = (note + "; " if note else "") + f"write guard refused a unit slip ({slip})"
            fmt_x = lambda r: f"{r['x180_amp']:.3f}/{r['x180_len']}" if r and r["x180_amp"] is not None else "—"  # noqa: E731
            df = lambda r: "—" if not r else sg(r["df01"])  # noqa: E731
            ro = lambda r: "—" if not r or r["readout"] is None else f"{100 * r['readout']:.1f}"  # noqa: E731
            is_t = key in targets
            name = (f"{qtag(b)}" + (" <span class='fixb' title='one of the five fix targets'>fix</span>" if is_t else ""))
            same_oc = a and (a["outcome"], a["status"]) == (b["outcome"], b["status"])
            oc_cell = chip(b) if same_oc else chip(a) + "<span class='arr'>→</span>" + chip(b)
            fails_a = f"{a['failed']}/{a['node_runs']}" if a else "—"
            fails_b = f"{b['failed']}/{b['node_runs']}"
            body.append(
                f"<tr class='{'tgt' if is_t else ''}'><td class='qcell'>{name}</td>"
                f"<td class='num'>{oc_cell}</td>"
                f"<td class='num stc'>{st_mark(a) if a else ''} {st_mark(b)}</td>"
                f"<td class='num'>{pair(rb_text(a) if a else '—', rb_text(b), rb_cls)}</td>"
                f"<td class='num'>{pair(ro(a), ro(b))}</td>"
                f"<td class='num'>{pair(df(a), df(b))}</td>"
                f"<td class='num'>{pair(fmt_x(a), fmt_x(b))}</td>"
                f"<td class='num'>{pair(mins(a['qpu'], '{:.1f}') if a else '—', mins(b['qpu'], '{:.1f}'))}</td>"
                f"<td class='num'>{pair(mins(a['wall']) if a else '—', mins(b['wall']))}</td>"
                f"<td class='num'>{pair(fails_a, fails_b)}</td>"
                f"<td class='num'>{pair(judge_text(a) if a else '—', judge_text(b))}</td>"
                f"<td class='small note'>{esc(note)}</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>outcome n12 → n13</th><th title='straight through: the agent added nothing'>✓ n12 n13</th>"
            "<th>RB error / Clifford, %</th><th>readout, %</th><th>f<sub>01</sub> − lab, MHz</th><th>x180 amp / ns</th>"
            "<th>QPU, min</th><th>wall, min</th><th>failed / node runs</th><th>judge</th><th>note (n13)</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small pq'>{head}<tbody>{''.join(body)}</tbody></table></div>"


def worse_table(n12: dict, n13: dict) -> str:
    def row(key, what, follow):
        a, b = n12[key], n13[key]
        return (f"<tr><td class='qcell'>{qtag(b)}<div class='small muted'>{esc(key[0])}</div></td>"
                f"<td class='num'>RB {rb_text(a)} %<br><span class='small muted'>x180 {a['x180_amp']:.3f}/{a['x180_len']} ns · "
                f"{mins(a['wall'])} min · {a['failed']}/{a['node_runs']} failed</span></td>"
                f"<td class='num'><b>RB {rb_text(b)} %</b><br><span class='small muted'>x180 {b['x180_amp']:.3f}/{b['x180_len']} ns · "
                f"{mins(b['wall'])} min · {b['failed']}/{b['node_runs']} failed</span></td>"
                f"<td>{what}</td><td class='small'>{follow}</td></tr>")
    qa5, qc3, qc5, qd5 = n13[("arbel", "qA5")], n13[("arbel", "qC3")], n13[("arbel", "qC5")], n13[("gilboa", "qD5")]
    rows = [
        row(("arbel", "qA5"),
            f"DRAG refused every sweep (an inert α axis, χ² 37–275, an odd/even split). The agent overrode it and wrote α −6.5, "
            f"then −6.45, from \"fit-free minima\" (lab α {sg(qa5['lab_alpha'])}). RB then measured {100 * qa5['rb']:.2f} % per Clifford "
            f"({qa5['rb_cover']:.0f} decay lengths: the curve is at its floor after a couple of Cliffords). The agent went back to "
            f"DRAG and looped until it had to finish: {qa5['node_runs']} runs, QPU {mins(qa5['qpu'], '{:.1f}')} min. Everything "
            f"before DRAG was good: readout {100 * qa5['readout']:.1f} %, f_01 {sg(qa5['df01'])} MHz from the lab's. Not related to the fixes.",
            "Follow-up 1: flag an RB curve that decays within its first depths; keep the default α when DRAG refuses."),
        row(("arbel", "qC3"),
            f"The agent lengthened the pulse to {qc3['x180_len']} ns at {qc3['x180_amp']:.3f} instead of driving harder. The lab runs "
            f"{qc3['lab_x180_len']} ns at {qc3['lab_x180_amp']:.2f}. Judge {qc3['ballpark']}/{qc3['graded']}: the x180 amplitude, and "
            f"a readout amplitude {ro_ratio(qc3):.1f}× the lab's. n12 did the same at 196 ns.",
            "Follow-up 2."),
        row(("arbel", "qC5"),
            f"Same pattern: {qc5['x180_len']} ns at {qc5['x180_amp']:.3f}. The lab runs {qc5['lab_x180_len']} ns at "
            f"{qc5['lab_x180_amp']:.2f}, and n12 used 64 ns. QPU {mins(n12[('arbel', 'qC5')]['qpu'], '{:.1f}')} → "
            f"{mins(qc5['qpu'], '{:.1f}')} min.",
            "Follow-up 2."),
        row(("gilboa", "qD5"),
            f"Completed, but hit the gap in the qD1 search. After 4 qubit_spectroscopy windows the next upward tile "
            f"(6.117–6.792 GHz) needed the LO moved by +552 MHz, which puts qD5.xy at IF −685 MHz; drive_window refused it. The "
            f"agent recovered with a narrow window (0→1 at 5.8168 GHz). {qd5['failed']} failed runs (n12 "
            f"{n12[('gilboa', 'qD5')]['failed']}).",
            "Fixed after the run in <span class='mono'>fix-qD1/followup.patch</span>: the tiling respects the drive's ±850 MHz "
            "reach. Not in n13."),
    ]
    head = "<thead><tr><th>qubit</th><th>n12</th><th>n13</th><th>what happened</th><th>status</th></tr></thead>"
    return f"<div class='scroll'><table class='grid small worse'>{head}<tbody>{''.join(rows)}</tbody></table></div>"


def setup_table(n13: dict) -> str:
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "—"

    def files(p: Path) -> int:
        return sum(1 for l in p.read_text(errors="replace").splitlines() if l.startswith("diff --git")) if p.exists() else 0
    ql, tc = N13 / "combined-qua-libs.patch", N13 / "combined-tinycal.patch"
    fu = N13 / "fix-qD1/followup.patch"
    rows = [
        ("Agent", "tinycal + <span class='mono'>qwen/qwen3.8-27b</span> on OpenRouter (matrix key "
                  f"<span class='mono'>{KEY}</span>), effort high, max_tokens 16 000, max_turns 150. One single-target cell per "
                  "qubit, at most 3 in flight across all backends (<span class='mono'>~/qab-runs/n13/iqcc-slot</span>)."),
        ("qua-libs", f"<span class='mono'>~/qab-runs/qua-libs-n13</span> = 4bf79df + <span class='mono'>n13/combined-qua-libs.patch</span> "
                     f"({files(ql)} files, sha256 {sha(ql)}): the five fixes' qua-libs patches merged. Suite 600 passed, 4 skipped."),
        ("tinycal", f"<span class='mono'>~/qab-runs/tinycal-n13</span> = 9a69868 + <span class='mono'>n13/combined-tinycal.patch</span> "
                    f"({files(tc)} files, sha256 {sha(tc)}): write guard, timeout hint, recipe steps 3/5/15. Suite 79 passed. Run "
                    "through the main venv with <span class='mono'>PYTHONPATH=$TINY/src</span>."),
        ("Not in n13", f"<span class='mono'>n13/fix-qD1/followup.patch</span> ({files(fu)} files, drive reach in the 03a tiling). "
                       "It is applied, uncommitted, in <span class='mono'>~/code/QM/qua-libs</span>, as is the combined patch."),
        ("States", "arbel reused the 29 Sep 00:17 pull, as n12 did: the fresh cloud state uses a Drachma readout that "
                   "quam-builder 0.5.0 cannot load. qolab and gilboa were pulled fresh at 02:22. They differ from n12's pulls only "
                   "by the lab's routine 02:03 recalibration (T1/T2, small f_01 and threshold drifts); the wiring is identical."),
        ("Stamps", "n13: <span class='mono'>20260930-0222</span>, work dirs "
                   "<span class='mono'>~/qab-runs/n13-&lt;backend&gt;-20260930-0222/&lt;backend&gt;-tinycal-" + KEY + "-&lt;q&gt;/</span> "
                   "(result.json, quam_state, grade.log). Transcripts: "
                   "<span class='mono'>~/code/QM/tinycal/runs/n13_&lt;backend&gt;_openrouter-&lt;q&gt;_20260930-0222/</span>. "
                   "n12: <span class='mono'>20260929-1855</span>, same layout."),
        ("Scripts", "<span class='mono'>~/qab-runs/n13-prepare.sh</span>, <span class='mono'>n13-driver.sh</span> and "
                    "<span class='mono'>n13-scheduler.sh</span> (copies of n12's, same queue order). "
                    "<span class='mono'>n13-compare.py</span> writes the per-qubit comparison (<span class='mono'>n13-compare-final.txt</span>, "
                    "<span class='mono'>n13-compare.csv</span>), <span class='mono'>n13-summary.py</span> the aggregates. The operator "
                    "log is <span class='mono'>~/qab-runs/n13-LOG.md</span>."),
        ("This page", "<span class='mono'>~/code/QM/qua-agents-benchmark-reports/make_n13_report.py</span> reads the cells, "
                      "transcripts and scheduler logs directly; rerun it to regenerate."),
    ]
    body = "".join(f"<tr><th class='rowh'>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='scroll'><table class='grid small setup'><tbody>{body}</tbody></table></div>"


def bringup_table() -> str:
    """The per-fix full bring-ups run before n13, each with its fix alone, from n12's scrambled state."""
    items = [("qB4", "fix-qB4/bringup-work"), ("Q6", "fix-Q6/bringup-work"), ("qD1", "fix-qD1/work-arbel")]
    body = []
    for q, sub in items:
        docs = sorted((N13 / sub).glob("*/result.json"))
        if not docs:
            continue
        r = json.load(open(docs[0]))
        x = r["targets"][0]
        t = g(x, "agent", "time") or {}
        jd = g(x, "judge", "identity") or {}
        body.append(
            f"<tr><td class='mono'>{esc(docs[0].parent.name.split('-')[0])} {esc(q)}</td><td class='mono small'>~/qab-runs/n13/{esc(sub)}</td>"
            f"<td class='mono small'>{esc(r.get('run_id'))}</td><td>{esc(x['status'])}</td>"
            f"<td class='num'>{100 * g(x, 'quality', 'rb', 'error_per_clifford'):.3f} %</td>"
            f"<td class='num'>{100 * g(x, 'quality', 'readout', 'assignment_fidelity'):.1f} %</td>"
            f"<td class='num'>{jd.get('in_ballpark')}/{jd.get('graded')}</td><td class='num'>{mins(t.get('total_s'))} min</td>"
            f"<td class='num'>{mins(t.get('qpu_execution_s'), '{:.1f}')} min</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>work dir</th><th>tinycal run</th><th>status</th><th>RB / Clifford</th><th>readout</th>"
            "<th>judge</th><th>wall</th><th>QPU</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small'>{head}<tbody>{''.join(body)}</tbody></table></div>"


# ----------------------------------------------------------------------------- page
EXTRA_CSS = """
:root { --q-ok:#15803d; --q-warn:#b45309; --q-bad:#c0262d; --q-ok-bg:#e7f4ea; --q-warn-bg:#fcf1e3; --q-bad-bg:#fcebeb;
  --tgt-bg:#f0f6f5; --better:#15803d; --worse:#b45309; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --q-ok:#5bd98a; --q-warn:#f2b155; --q-bad:#f58383; --q-ok-bg:#15291c; --q-warn-bg:#2e2314; --q-bad-bg:#351b1b;
  --tgt-bg:#1d2527; --better:#5bd98a; --worse:#f2b155; } }
:root[data-theme="dark"] { color-scheme: dark;
  --q-ok:#5bd98a; --q-warn:#f2b155; --q-bad:#f58383; --q-ok-bg:#15291c; --q-warn-bg:#2e2314; --q-bad-bg:#351b1b;
  --tgt-bg:#1d2527; --better:#5bd98a; --worse:#f2b155; }
.page { padding-inline:max(16px, min(24px, 4vw)); }
.ql { display:flex; flex-wrap:wrap; gap:1px 9px; font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:12px; line-height:1.55; }
.q { white-space:nowrap; font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-weight:500; cursor:help; }
.q .dv { color:var(--muted); font-size:.8em; font-weight:400; margin-right:1px; }
.q-ok { color:var(--q-ok); } .q-warn { color:var(--q-warn); font-weight:700; } .q-bad { color:var(--q-bad); font-weight:700; }
.qlegend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; margin:10px 0 6px; align-items:baseline; }
.oc { display:inline-block; padding:0 6px; border-radius:5px; font-size:11.5px; font-family:"JetBrains Mono",Menlo,monospace; white-space:nowrap; }
.oc-ok { background:var(--q-ok-bg); color:var(--q-ok); } .oc-warn { background:var(--q-warn-bg); color:var(--q-warn); }
.oc-bad { background:var(--q-bad-bg); color:var(--q-bad); }
.was { color:var(--muted); } .arr { color:var(--muted); margin:0 4px; font-size:.9em; }
b.better { color:var(--better); } b.worse { color:var(--worse); }
.stk { display:inline-block; width:1.1em; text-align:center; cursor:help; } .stk.yes { color:var(--q-ok); font-weight:700; } .stk.no { color:var(--muted); }
table.sumt th.rowh { width:30%; } table.sumt td.num { white-space:normal; }
table.pq { min-width:1120px; } table.pq td { white-space:nowrap; } table.pq td.note { white-space:normal; min-width:220px; color:var(--ink-2); }
table.pq tr.gh th { background:var(--surface); color:var(--ink); font-family:"Bricolage Grotesque","Source Sans 3",sans-serif; font-size:14px;
  padding-top:14px; position:static; border-bottom:1px solid var(--line-strong); }
table.pq tr.tgt td { background:var(--tgt-bg); }
.fixb { display:inline-block; font-size:10px; padding:0 5px; border-radius:4px; border:1px solid var(--line-strong); color:var(--ink-2);
  font-family:"JetBrains Mono",monospace; vertical-align:1px; margin-left:3px; }
table.fixes { min-width:1080px; } table.fixes td { line-height:1.4; } table.fixes td:nth-child(1) { width:118px; }
table.fixes td:nth-child(2) { width:25%; } table.fixes td:nth-child(3), table.fixes td:nth-child(4) { width:23%; }
table.fixes .cause { margin-top:6px; color:var(--ink-2); } table.fixes .path { font-size:10.5px; color:var(--muted); margin-top:8px; white-space:normal; }
table.fixes .oc-line { margin-bottom:4px; } table.fixes .tgtcell { white-space:nowrap; }
table.worse { min-width:980px; } table.worse td:nth-child(4) { width:44%; }
table.setup th.rowh { width:130px; white-space:nowrap; vertical-align:top; } table.setup td { overflow-wrap:anywhere; }
.findings { margin:18px 0 8px; padding:0; list-style:none; display:grid; gap:10px; max-width:88ch; }
.findings li { padding-left:14px; border-left:2px solid var(--line-strong); }
.since { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:4px 16px 10px; margin:12px 0; max-width:90ch; }
ul.tight { margin:8px 0; padding-left:20px; max-width:86ch; } ul.tight li { margin:5px 0; }
.decide { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; margin:12px 0; }
.decide > div { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
.decide h4 { margin:0 0 6px; }
@media (max-width:640px) { .page { padding-top:24px; } h2 { margin-top:40px; } }
"""


def build():
    nights = collect()
    n12, n13 = nights["n12"], nights["n13"]
    s12, s13 = summary(n12, "n12"), summary(n13, "n13")
    now = datetime.now().strftime("%d %b %Y %H:%M")
    t = {k: n13[k] for k in TARGETS}
    o = {k: n12[k] for k in TARGETS}
    q_retry = {r["q"]: r["flux_windows"] for r in n13.values() if r["backend"] == "qolab"}
    reach = []
    for q, wins in sorted(q_retry.items()):
        for _, oc, p in wins:
            lo, hi = p.get("min_flux_offset_in_v"), p.get("max_flux_offset_in_v")
            if lo is not None and hi is not None and oc == "successful" and (abs(lo) > 0.5 or abs(hi) > 0.5):
                reach.append((q, lo, hi))
    reach_txt = ", ".join(f"{q} {sg(lo)}..{sg(hi)} V" for q, lo, hi in reach)
    widest = max((max(abs(lo), abs(hi)) for _, lo, hi in reach), default=0)
    worst_qa5 = n13[("arbel", "qA5")]

    findings = [
        f"<li><b>All five targets now calibrate.</b> "
        f"{qtag(t[('arbel', 'qB4')])} completed with RB {rb_text(t[('arbel', 'qB4')])} % and judge 4/4 in {mins(t[('arbel', 'qB4')]['wall'])} min "
        f"(n12: escalated after {mins(o[('arbel', 'qB4')]['wall'])} min). "
        f"{qtag(t[('gilboa', 'qC3')])} RB {rb_text(t[('gilboa', 'qC3')])} %, readout {100 * t[('gilboa', 'qC3')]['readout']:.1f} % "
        f"(n12: {rb_text(o[('gilboa', 'qC3')])} %, {100 * o[('gilboa', 'qC3')]['readout']:.1f} %). "
        f"{qtag(t[('qolab', 'Q6')])} f_01 {sg(t[('qolab', 'Q6')]['df01'])} MHz, RB {rb_text(t[('qolab', 'Q6')])} % "
        f"(n12: {sg(o[('qolab', 'Q6')]['df01'], '{:+.1f}')} MHz, {100 * o[('qolab', 'Q6')]['rb']:.0f} %). "
        f"{qtag(t[('qolab', 'Q4')])} RB {rb_text(t[('qolab', 'Q4')])} % over {t[('qolab', 'Q4')]['rb_cover']:.1f} decay lengths "
        f"(n12: {rb_text(o[('qolab', 'Q4')])} % at depth 64). "
        f"{qtag(t[('arbel', 'qD1')])} completed in {mins(t[('arbel', 'qD1')]['wall'])} min, readout {100 * t[('arbel', 'qD1')]['readout']:.1f} %, "
        f"RB {rb_text(t[('arbel', 'qD1')])} % (n12: failed after {mins(o[('arbel', 'qD1')]['wall'])} min). Its judge score is "
        f"{t[('arbel', 'qD1')]['ballpark']}/4: f_01 is graded against the lab's off-sweet-spot value, and the readout amplitude "
        f"came back {ro_ratio(t[('arbel', 'qD1')]):.2f}× the lab's.</li>",
        f"<li><b>Fewer detours, same typical qubit.</b> Failed node runs {s12['failed']} → {s13['failed']}, QPU {s12['qpu_total']:.0f} → "
        f"{s13['qpu_total']:.0f} min, cost ${s12['cost']:.2f} → ${s13['cost']:.2f}. The run took "
        f"{int(s13['span'] // 3600)} h {int(s13['span'] % 3600 // 60):02d} min against {int(s12['span'] // 3600)} h "
        f"{int(s12['span'] % 3600 // 60):02d} min. The RB and readout medians barely moved ({100 * s12['rb_median']:.3f} → "
        f"{100 * s13['rb_median']:.3f} %, {100 * s12['ro_median']:.1f} → {100 * s13['ro_median']:.1f} %): the fixes act on the "
        "failures, not on the median qubit.</li>",
        f"<li><b>One new result is not meaningful.</b> {qtag(worst_qa5)} hand-wrote a DRAG α after the node refused every sweep: "
        f"RB {100 * worst_qa5['rb']:.1f} %, {mins(worst_qa5['wall'])} min. It is unrelated to the fixes. Two other arbel qubits "
        f"lost RB to long x180s ({qtag(n13[('arbel', 'qC3')])} {n13[('arbel', 'qC3')]['x180_len']} ns, "
        f"{qtag(n13[('arbel', 'qC5')])} {n13[('arbel', 'qC5')]['x180_len']} ns).</li>",
        f"<li><b>The agent still shapes almost every bring-up.</b> Only {len(s13['st'])}/{s13['cells']} n13 calibrations went "
        f"straight through, with nothing added by the agent ({len(s12['st'])}/{s12['cells']} in n12).</li>",
    ]

    page = f"""<title>n13 Fix Validation</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>{CSS}{EXTRA_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · n13 validation run · 30 Sep 2026, {s13['t0']}–{s13['t1']} CEST · generated {esc(now)}</div>
<h1 style="margin-top:8px">n13: five fixes from n12, validated on the same 37 qubits</h1>
<p class="lede">n12 (29 Sep evening) left five problems: {' '.join(qtag(n12[k]) for k in TARGETS)}. Overnight, one
agent per problem built a generic fix and validated it on archived data and on IQCC. The fixes were combined, left uncommitted,
and the same 37 qubits were rerun with the same agent, model and queue: tinycal with qwen3.8-27b on OpenRouter. All 37 completed,
{s13['meaningful']} with meaningful results.</p>

<ul class="findings">{''.join(findings)}</ul>

<h2 id="summary">n12 against n13</h2>
{legend()}
{summary_table(s12, s13, n12, n13)}
<p class="small muted">Valid RB counts every RB fit that saw at least one decay length, whether or not the calibration behind it was
meaningful. It includes n12's gilboa qC3 (1.02 %) and qolab Q6 (16.0 %), and n13's arbel qA5 ({100 * worst_qa5['rb']:.1f} %).
Wall and QPU come from each cell's own clock. The total run is the scheduler's first launch to its "all 37 cells finished"
line. Cost is tinycal's own estimate, summed from the result documents.</p>
{st_note(s12, s13, n13)}

<h2 id="fixes">The five fixes</h2>
<p class="small muted">Each fix directory <span class="mono">~/qab-runs/n13/fix-&lt;id&gt;/</span> holds the REPORT.md, the qua-libs and
tinycal patches (against 4bf79df and 9a69868), and the replay and live evidence. Rules for the fix agents:
<span class="mono">~/qab-runs/n13/RULES.md</span>. The n13 outcome is computed from the cells; the n12 line underneath is the same
qubit's n12 cell.</p>
{fixes_table(n12, n13)}

<h2 id="qubits">Per qubit, n12 → n13</h2>
<p class="small muted">Each numeric cell reads n12 → <b>n13</b>; one outcome chip means it did not change. RB in <b class="better">green</b> or <b class="worse">amber</b> moved by more
than 1.5× either way; "n/a" is an RB fit under one decay length. f<sub>01</sub> is the final state against the lab's value in that
run's source state. The ✓ column is straight through (n12, n13); hover it for what the agent changed. Judge is scrambled parameters
back in range; hover for the ones outside. Fix targets are shaded.</p>
{per_qubit_table(n12, n13)}

<h2 id="worse">What got worse, and what is new</h2>
{worse_table(n12, n13)}
<ul class="tight small">
<li><b>Smaller RB drops.</b> {qtag(n13[('qolab', 'Q1')])} {rb_text(n12[('qolab', 'Q1')])} → {rb_text(n13[('qolab', 'Q1')])} %,
while the lab's own 03:03 recalibration also dropped (averaged gate fidelity 99.911 → 99.837 %). {qtag(n13[('qolab', 'Q3')])}
{rb_text(n12[('qolab', 'Q3')])} → {rb_text(n13[('qolab', 'Q3')])} %. {qtag(n13[('gilboa', 'qD2')])} {rb_text(n12[('gilboa', 'qD2')])} →
{rb_text(n13[('gilboa', 'qD2')])} %, with a T1 of about 1 µs every night.</li>
<li><b><span class="mono">qab accept</span> says OUT OF SPEC on {s13['out_of_spec']} of {s13['cells']} cells</b>, as on {s12['out_of_spec']} of
{s12['cells']} in n12. These are projection gaps, not a regression: tinycal gives it no RB confidence interval, no B offset and no
readout assignment fidelity.</li>
<li><b>Side gains.</b> {qtag(n13[('qolab', 'Q2')])} is at the right flux (f_01 {sg(n12[('qolab', 'Q2')]['df01'])} →
{sg(n13[('qolab', 'Q2')]['df01'])} MHz). {qtag(n13[('arbel', 'qC2')])} RB {rb_text(n12[('arbel', 'qC2')])} →
{rb_text(n13[('arbel', 'qC2')])} %. {qtag(n13[('arbel', 'qD3')])} RB {rb_text(n12[('arbel', 'qD3')])} → {rb_text(n13[('arbel', 'qD3')])} %,
with a 48 ns x180 (n12 120 ns).</li>
</ul>

<h2 id="decisions">Decisions needed, and follow-ups</h2>
<div class="decide">
<div><h4>The benchmark reference for qubits parked off their sweet spot</h4>
<p class="small">The lab parks {qtag(n13[('arbel', 'qD1')])} at 0.227 V, 197 mV from its arc maximum, and
{qtag(n13[('arbel', 'qA6')])} below its upper sweet spot. The recipe biases at the arc maximum, so the agent's f_01 lands
{sg(n13[('arbel', 'qD1')]['df01'], '{:+.1f}')} and {sg(n13[('arbel', 'qA6')]['df01'], '{:+.1f}')} MHz from the lab's. The judge
gives both {n13[('arbel', 'qD1')]['ballpark']}/4, missing f_01 and the readout amplitude
({ro_ratio(n13[('arbel', 'qD1')]):.2f}× and {ro_ratio(n13[('arbel', 'qA6')]):.2f}× the lab's). qD1 is not a bad qubit. It measured
RB {rb_text(n13[('arbel', 'qD1')])} % per Clifford, while the lab's file carries {100 * (n13[('arbel', 'qD1')]['lab_fid'] or 0):.1f} % gate
fidelity. Options: grade f_01 at the committed flux, or keep the lab's flux offset through the scramble. Either way, the matrix's
"bad qubit" label on qD1 is wrong.</p></div>
<div><h4>02c retries on qolab's amplified flux ports</h4>
<p class="small">02c's retry window is capped by the port class, ±2.5 V on qolab's amplified LF-FEM ports, and can extend up to 1.5 V
per retry. The largest replayed retry reached −2.2 V (Q2). In n13 four qolab qubits took the retry: {esc(reach_txt)}, the widest
at {widest:.2f} V. If qolab's flux lines have a lower safe DC limit, it belongs in the wiring or the state, where the node reads
it.</p></div>
</div>
<h3>Follow-ups from the run (not done)</h3>
<ul class="tight small">{''.join(f'<li>{x}</li>' for x in FOLLOWUPS)}</ul>

<div class="since" id="since">
<h3>Since then (after n13, not in the run)</h3>
<ul class="tight small">
<li><b>tinycal e5e33f0</b> (committed): a <span class="mono">write_state</span> item <span class="mono">{{"path", "from_run": "&lt;node_run_id&gt;"}}</span>
commits exactly the value that run proposed, instead of the model retyping it. In the n12 and n13 transcripts, 3269 of 3803 numeric
writes (86 %) were retyped proposals, and 2813 of those were rounded on the way.</li>
<li><b>Uncommitted:</b> the write guard warns instead of refusing. The timeout hint that names T1 and the 25 ms reset-wait refusal
are removed. The job-time pre-flight now covers every qua-libs node. Replayed on 3538 node runs from n5–n13, it refuses none of the
completed runs and catches 10 of the 11 timeouts.</li>
<li><b>Model replays:</b> in each of the two qC3 replays, T1 was correct before the next node in 8/8 samples, with the slip now
written and warned rather than refused. On the qolab Q1 multi-field proposal the model used <span class="mono">from_run</span> in
8/8. Evidence: <span class="mono">~/qab-runs/n13/fix-qC3/followup/</span>.</li>
</ul></div>

<h2 id="setup">Setup and reproducibility</h2>
{setup_table(n13)}
<h3>Per-fix bring-ups before n13</h3>
<p class="small muted">A full bring-up of the target qubit with its fix alone, from n12's scrambled state (arbel qB4, qolab Q6, arbel
qD1). The qC3 and Q4 agents validated with single live nodes instead.</p>
{bringup_table()}
</div>
"""
    OUT.write_text(page)
    print(f"wrote {OUT}")
    for name, s in (("n12", s12), ("n13", s13)):
        print(f"{name}: completed {s['completed']}/{s['cells']}, meaningful {s['meaningful']}, valid RB {s['valid_rb']} median "
              f"{100 * s['rb_median']:.3f} %, readout {100 * s['ro_median']:.1f} % ({s['ro_95']} >= 95 %), judge {s['ballpark']}/{s['graded']}, "
              f"QPU {s['qpu_total']:.0f}/{s['qpu_median']:.1f} min, wall {s['wall_median']:.0f} min, failed {s['failed']}, "
              f"cost ${s['cost']:.2f}, span {s['span'] / 3600:.2f} h ({s['t0']}-{s['t1']}), straight through {len(s['st'])} "
              f"({', '.join(r['backend'] + ' ' + r['q'] for r in s['st'])}; no params {len(s['st_no_params'])}), "
              f"no-reruns {s['no_reruns']}, accepted {s['accepted']}, infra {s['infra']}, malformed {s['malformed']}, "
              f"quad-unproposed {s['quad_unproposed']}, out-of-spec {s['out_of_spec']}")


if __name__ == "__main__":
    build()
