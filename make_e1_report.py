#!/usr/bin/env python3
"""Build the 30 Sep 2026 report: e1, the first hardware run of qua-decision-engine (a rule-based calibration decision engine,
no language model), against n13 (tinycal + qwen3.8-27b via OpenRouter, the night before) on the same 37 qubits.

    python3 make_e1_report.py

Numbers come from each cell's result.json (projected from tinycal's events.jsonl, stamped by qab inspect-state / validate /
accept), the cell's final quam_state and its work dir's source-state (the lab's calibration), the tinycal event logs (node runs,
next_action, writes, the engine's finish summary), the schedulers' logs (which engine build ran which cell), the arbel resume
log and the engine's own policy module. n13's rows are read by make_n13_report.cell_row, unchanged. Hand-written text is limited
to the operator notes below (causes, explanations, follow-ups), taken from ~/qab-runs/e1-LOG.md.

The run was not finished when this page was first built: arbel's readout failed at about 15:06 and its remaining qubits wait for
~/qab-runs/e1c-resume.sh. Per qubit the latest arbel work dir wins, so rerunning this script after the resume refreshes the page.

The output is page content for a claude.ai artifact: a <title>, the stylesheet and the body, with no <html>/<head>/<body> tags.
CSS and small helpers are shared with make_fwcmp2_report.py and make_n13_report.py.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_fwcmp2_report import CSS, esc, median  # noqa: E402
import make_n13_report as n13r  # noqa: E402
from make_n13_report import EXTRA_CSS as N13_CSS, g, judge_text, mins, numeric, pair, plural, read_events, sg  # noqa: E402

HOME = Path.home()
QAB = HOME / "qab-runs"
TINY_RUNS = HOME / "code/QM/tinycal/runs"
OUT = Path(__file__).with_name("2026-09-30-e1-decision-engine-vs-agent.html")
BACKENDS = n13r.BACKENDS
DEV = n13r.DEV
RB_VALID = n13r.RB_VALID
CEST = timezone(timedelta(hours=2))

STAMP = (QAB / "e1-stamp.txt").read_text().strip()  # qolab and gilboa: 20260930-1333
ARBEL_STAMP = (QAB / "e1-arbel-stamp.txt").read_text().strip()  # arbel, rebuilt with square readout: 20260930-1424
RESUME_STAMP_FILE = QAB / "e1-arbel-c-stamp.txt"  # written by e1c-resume.sh when arbel's readout is back
VOID_WORK = {f"e1-arbel-{STAMP}"}  # the first arbel wave: CONFIG ERROR at open_qm on every cell
QDE = {"e1": QAB / "qde-e1-run", "e1b": QAB / "qde-e1b-run", "e1c": QAB / "qde-e1b-run"}  # engine build per scheduler
SCHED_LOG = {"e1": QAB / "e1-scheduler.log", "e1b": QAB / "e1b-scheduler.log", "e1c": QAB / "e1c-resume.log"}
QL_RUN = QAB / "qua-libs-e1-run"
QL_SC = QL_RUN / "qualibration_graphs/superconducting"
TINY_E1 = QAB / "tinycal-e1"
FAULT_FROM = datetime(2026, 9, 30, 15, 5, tzinfo=CEST)  # arbel's readout: last good IQ_blobs 14:58, first bad 15:10
WORSE_RB = 1.3  # "worse": RB error per Clifford more than 30 % higher (relative), both valid
WORSE_RO = 0.02  # or readout assignment more than 2 points lower

# The nodes that publish next_action (qua-libs ad9dc4ee + e7b13820): graph name -> (library id, analysis package, main rules).
# The rules are a digest of the commit message and the analysis code; the reason codes are read from the code below.
NA_NODES = [
    ("resonator_spectroscopy_vs_power", "02b", "resonator_spectroscopy_vs_amplitude",
     "No onset measured: rerun with the node's wider window; otherwise go on, committing the fallback power (the top of the "
     "flat low-power region) only when the stored power is above it. No line tracked, or the proposal at the sweep's "
     "bottom: escalate."),
    ("resonator_spectroscopy_vs_flux", "02c", "resonator_spectroscopy_vs_flux",
     "Arc apex beyond the sweep: rerun on the wider window it proposes (capped by the port class). Flat, untracked or "
     "outlier maps: escalate with the reason."),
    ("qubit_spectroscopy", "03a chirp", "qubit_spectroscopy_chirp",
     "The line search. No 0→1 line in the window: rerun on the next window of a fixed tiling (up to the readout "
     "resonator − 300 MHz, then one below, each checked against the drive's reach); exhausted: escalate. Excited before the "
     "shot: rerun with a longer wait. Imprecise line: rerun on a narrow window. T1 too short for the sweep: a shorter sweep, "
     "or escalate (<span class='mono'>sweep_too_long_for_t1</span>)."),
    ("T1_chirp", "05b", "T1_chirp",
     "Decay not resolved: rerun once with more shots, then go on without T1. No fit: go on without T1."),
    ("qubit_spectroscopy_fine", "03a", "qubit_spectroscopy",
     "Line narrower than the step: rerun finer. No line: rerun at twice the Rabi drive, then escalate. A line too broad "
     "for its drive: rerun at half the drive."),
    ("power_rabi", "04b", "power_rabi",
     "No credible contrast: rerun with more shots (50 → 200 → 400), then keep the stored x180. π above the amplitude limit: "
     "commit the longer x180 and rerun at it (step back to the line search if that length is two-photon scale). Optimum at "
     "the window edge: rerun on a moved window."),
    ("power_rabi_error_amplification_x180", "04c", "power_rabi_error_amplification",
     "power_rabi's rules, with this node's own defaults and bounds (e.g. an untrusted fit reruns on a narrower amplitude "
     "window)."),
    ("readout_power_optimization", "08b", "readout_power_optimization",
     "Optimum at the sweep edge: rerun on a moved amplitude window, without committing. No finite fit: escalate. An upstream "
     "suspect: step back to power_rabi."),
    ("DRAG_calibration", "10b", "drag_calibration_180_minus180",
     "Optimum pinned at or past the window edge: rerun re-centred. Inert α axis: rerun once seeded, then keep α. Fit "
     "untrusted or not converged: keep the stored α and go on to RB."),
]

# ----------------------------------------------------------------------------- operator notes (by hand, from e1-LOG.md)
ROW_NOTE = {  # (backend, qubit) -> note on the e1 row, from the n13 row (a) and the e1 row (b)
    ("arbel", "qB4"): lambda a, b: f"DRAG refused, α kept at {b['alpha']:g}; the agent had {a['alpha']:.3f}",
    ("arbel", "qD1"): lambda a, b: "the ported line search found the line (at its sweet spot, as in n13)",
    ("gilboa", "qC3"): lambda a, b: "run on engine c1e0a5c: the refused write was not caught, so it stopped on the repeat guard",
    ("gilboa", "qD2"): lambda a, b: f"n13's agent 'completed' it at RB {100 * a['rb']:.1f} %",
    ("qolab", "Q3"): lambda a, b: f"f_01 {sg(b['df01'], '{:+.1f}')} MHz from the lab's, as in n13; the lab value is suspect",
}
STOP_GROUPS = {  # cause -> (title, explanation); the qubits and numbers are computed
    "length": (
        "Pulse length read-only, x180 amplitude capped at 0.6",
        "Since tinycal <span class='mono'>931acd3</span> (30 Sep, 12:33) the benchmark profile holds pulse lengths read-only. "
        "power_rabi caps the x180 amplitude at 0.6 (<span class='mono'>quam_config/instrument_limits.py</span>, whose comment "
        "calls it \"a subjective 'safe' value\"). When π needs more than 0.6 at the stored length, the node's next_action is to "
        "commit a longer x180 and rerun; the state store refuses the length, and the engine stops with the store's reason. The "
        "lab itself drives these qubits above 0.6. The n13 agent ran before 931acd3 and lengthened the pulses instead. "
        "<span class='mono'>origin/feat/qualibrate-ai</span> raised the cap to the hardware ceiling (1.0 on an MW-FEM) in "
        "<span class='mono'>41ced395</span>; it is not applied here, because a hardware safety cap is the user's call."),
    "t1": (
        "T1 too short for the chirp line search",
        "gilboa qD2's T1 is about 0.9 µs. The chirp line search refuses a sweep longer than T1/5 "
        "(<span class='mono'>sweep_too_long_for_t1</span>) and says to find the line with saturation spectroscopy. That is "
        "the correct answer for this qubit; the n13 agent pushed on and \"completed\" it with an RB that is not a calibration."),
    "fault": (
        "arbel readout fault (hardware, not the engine)",
        "From about 15:06 arbel's readout degraded on every qubit it touched. Node runs before 15:05 against after: IQ_blobs "
        "assignment {pre_iq} % against {post_iq} %; resonator_identification contrast SNR {pre_snr} against {post_snr}; "
        "resonator_spectroscopy_vs_power plateau contrast {pre_pl} against {post_pl}. The line search found no qubit line on "
        "{no_line} qubits where n13 found one. The queue waits jumped at the same time (another user on arbel). The cells that "
        "finished after 15:05 are void and are left out of the like-for-like comparison. The timeline below is read from the "
        "cells' events."),
    "other": ("Other stops", "Stops the rules above do not cover; the engine's own reason is shown."),
}
WORSE_WHY = {  # (backend, qubit) -> operator explanation for an e1 result worse than n13's, from the rows (a: n13, b: e1)
    ("arbel", "qB4"): lambda a, b: (
        f"DRAG refused (<span class='mono'>continue_without / fit_untrusted</span>), so α stayed at {b['alpha']:g}; the agent had "
        f"{a['alpha']:.3f} (lab {b['lab_alpha']:.2f}). DRAG's own warnings name the remedy when the optimum is merely uncertain "
        f"(more shots, a denser pulse axis{drag_uncertainty(b)}), but its next_action only re-centres or keeps α. It should rerun "
        "once with them before keeping α."),
    ("gilboa", "qD3"): lambda a, b: (
        "Cause not identified. Rerun after a bare <span class='mono'>Exception:</span> at the first node; the rerun's decisions "
        "are listed below."),
    ("qolab", "Q2"): lambda a, b: "Cause not identified. The readout dropped too.",
}
DECISIONS = [  # hand-written: decisions needed
    ("The 0.6 x180 cap against read-only pulse lengths",
     "Under the benchmark profile a qubit whose π needs more than 0.6 at its stored length cannot be calibrated at all: "
     "power_rabi asks for a longer pulse and the profile refuses it. The lab drives such qubits at 0.675–0.997. Either raise "
     "the cap to the hardware ceiling (<span class='mono'>41ced395</span> on origin/feat/qualibrate-ai, with its sweep "
     "changes), or let the profile allow a longer x180 when power_rabi proposes one. It is a hardware safety limit, so it is "
     "the user's decision."),
    ("Commit the chirp line search to feat/qualibrate-ai",
     "n13's qD1 fix (the tiled line search: <span class='mono'>search.py</span>, <span class='mono'>retry_parameters</span> / "
     "<span class='mono'>next_window</span> in the chirp analysis) never reached feat/qualibrate-ai; only its drive-window half "
     "did (<span class='mono'>13ff96ba</span>). e1 ran it as <span class='mono'>e7b13820</span> on feat/node-next-action. "
     "Without it the engine has no rule for arbel qD1, and neither does an agent on that branch."),
]
FOLLOWUPS = [  # hand-written: follow-ups (not done)
    "<b>DRAG next_action.</b> When the optimum is only uncertain, rerun once with the shots and pulse-axis density DRAG's own "
    "warning names, before keeping α (arbel qB4).",
    "<b>Fresh run ids for reruns.</b> tinycal appends to an existing run's events.jsonl, so gilboa qD3's rerun file also holds "
    "the first try's failed run and finish. This page reads the last start-to-finish segment; the projection does not.",
    "<b>Reset the dev worktree.</b> <span class='mono'>~/qab-runs/qua-libs-e1</span> holds a half-applied cherry-pick of "
    "41ced395 (uncommitted partial changes after an aborted attempt). The run library "
    "<span class='mono'>~/qab-runs/qua-libs-e1-run</span> is untouched.",
    "<b>Finish arbel when its readout recovers.</b> <span class='mono'>~/qab-runs/e1c-resume.sh</span> probes qA1's IQ_blobs "
    "every 15 min and, at ≥ 90 %, pulls a fresh arbel state, rebuilds it with square readout and reruns the 14 arbel qubits "
    "with fresh run ids (it gives up at 17:30). Rerun this script afterwards.",
    "<b>Concurrency for this run</b> was 3 per device on qolab and gilboa and 2 on arbel, 3 while arbel's completed-run queue "
    "median stayed under 30 s (<span class='mono'>e1-arbel-cap.py</span>; 10 s at launch, 40 s at 15:16, 30 s at 15:19), by "
    "the user's instruction for this run.",
]


def drag_uncertainty(r: dict) -> str:
    """'; the optimum uncertain to X steps, where <= Y is wanted', from the last DRAG run's warnings."""
    runs = [e for e in run_calls(r.get("ev") or []) if e.get("node") == "DRAG_calibration"]
    for w in (runs[-1].get("warnings") or []) if runs else []:
        m = re.search(r"uncertain to ([\d.]+) sweep steps \(want <= ([\d.]+)\)", w)
        if m:
            return f"; {r['q']}'s optimum uncertain to {m.group(1)} sweep steps, where ≤ {m.group(2)} is wanted"
    return ""


# ----------------------------------------------------------------------------- small readers
def git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"


def load_json(p: Path):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def ts(e: dict | None) -> datetime | None:
    return datetime.fromisoformat(e["ts"]) if e and e.get("ts") else None


def hm(t: datetime | None, secs=False) -> str:
    return "—" if t is None else t.astimezone(CEST).strftime("%H:%M:%S" if secs else "%H:%M")


def stamp_time(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d-%H%M").replace(tzinfo=CEST)


def segments(ev: list[dict]) -> list[list[dict]]:
    """An events file split at each start event: a rerun under the same run id appends to the same file."""
    starts = [i for i, e in enumerate(ev) if e.get("kind") == "start"] or [0]
    return [ev[a:b] for a, b in zip(starts, starts[1:] + [len(ev)])]


def run_calls(ev: list[dict]) -> list[dict]:
    return [e for e in ev if e.get("kind") == "tool_call" and e.get("tool") == "run_node"]


def next_action(e: dict) -> dict | None:
    v = numeric(e, "next_action")
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v if isinstance(v, dict) else None


def load_policy(root: Path, name: str):
    """The engine's graph policy, as the frozen build that ran has it (plain dataclasses, no dependencies)."""
    spec = importlib.util.spec_from_file_location(name, root / "src/qde/policy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.FLUX_TUNABLE_BRINGUP


DECISION_RE = re.compile(r"^(?P<node>\w+): (?P<kind>accept|rerun|continue_without|step_back|escalate) "
                         r"\((?P<reason>[^,]+), from (?P<src>\w+)\)")


def parse_log(lines: list[str]) -> list[dict]:
    """The engine's decision log (the finish summary after its first line)."""
    out = []
    for ln in lines:
        m = DECISION_RE.match(ln)
        if m:
            out.append(m.groupdict() | {"commit": "commit proposals" in ln})
        elif m := re.match(r"^(\w+): infrastructure failure \(([^)]*)\)", ln):
            out.append({"node": m.group(1), "kind": "infra", "reason": m.group(2), "src": "runner"})
        elif m := re.match(r"^(\w+): skipped", ln):
            out.append({"node": m.group(1), "kind": "skipped", "reason": "condition", "src": "policy"})
        elif m := re.match(r"^(\w+): .*keeping the stored values", ln):
            out.append({"node": m.group(1), "kind": "keep", "reason": ln.split(": ", 1)[1], "src": "engine"})
    return out


# ----------------------------------------------------------------------------- schedulers: which engine build ran which cell
def launches() -> dict[tuple[str, str, str], dict]:
    """(backend, qubit, work stamp) -> the last launch: scheduler, time, engine build."""
    out = {}
    for sched, path in SCHED_LOG.items():
        if not path.exists():
            continue
        for ln in path.read_text(errors="replace").splitlines():
            m = re.match(r"^\[(\d\d:\d\d:\d\d)\] e1[bc]?(?: scheduler)?: launched (\w+) (\w+)(?: \(pid \d+(?:, work e1-\w+-([\d-]+))?\))?", ln)
            if not m:
                continue
            t, be, q, work = m.groups()
            if sched == "e1":
                work = STAMP
            elif sched == "e1c":
                work = RESUME_STAMP_FILE.read_text().strip() if RESUME_STAMP_FILE.exists() else None
            out[(be, q, work)] = {"sched": sched, "time": t, "engine": QDE[sched]}
    return out


def driver_engine(be: str, q: str, stamp: str) -> str | None:
    """The engine sha the driver printed, as a cross-check (driver logs are overwritten by a rerun of the same qubit)."""
    for p in (QAB / f"e1c-{be}-{q}.log", QAB / f"e1-{be}-{q}.log"):
        if p.exists():
            head = p.read_text(errors="replace").splitlines()[:1]
            m = re.search(rf"e1_{be}-{q}_{stamp} \(qua-libs \w+, engine (\w+)", head[0]) if head else None
            if m:
                return m.group(1)
    return None


# ----------------------------------------------------------------------------- collect
def work_dirs(backend: str) -> list[Path]:
    return sorted(p for p in QAB.glob(f"e1-{backend}-2026*") if p.is_dir() and p.name not in VOID_WORK)


def resume_stamp() -> str | None:
    return RESUME_STAMP_FILE.read_text().strip() if RESUME_STAMP_FILE.exists() else None


def e1_cell(backend: str, q: str, work: Path, launch: dict | None) -> dict:
    stamp = work.name.split("-", 2)[2]
    cdir = work / f"{backend}-engine-{q}"
    run_id = f"e1_{backend}-{q}_{stamp}"
    segs = segments(read_events(run_id, q))
    ev = segs[-1]
    earlier = [e for s in segs[:-1] for e in run_calls(s)]
    fin = next((e for e in reversed(ev) if e.get("kind") == "finish"), None)
    eng_dir = (launch or {}).get("engine")
    row = {
        "night": "e1", "backend": backend, "q": q, "stamp": stamp, "work": work, "run_id": run_id, "ev": ev,
        "earlier_runs": len(earlier), "earlier_failed": sum(1 for e in earlier if e.get("outcome") != "successful"),
        "finish_ts": ts(fin), "start_ts": ts(ev[0]) if ev else None, "sched": (launch or {}).get("sched"),
        "engine": git(eng_dir, "rev-parse", "--short", "HEAD") if eng_dir else "?",
        "driver_engine": driver_engine(backend, q, stamp),
        "summary_head": fin["summary"].splitlines()[0] if fin else "", "log": parse_log(fin["summary"].splitlines()[1:]) if fin else [],
    }
    head = row["summary_head"]
    m = re.search(r"; (\d+) node runs \(budget (\d+)\), (\d+) infrastructure retries", head)
    row["engine_runs"], row["budget"], row["infra"] = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, 0)
    reason = re.sub(r"; RB error per Clifford [\d.]+ %", "", head)
    reason = re.sub(r"; \d+ node runs \(budget \d+\), \d+ infrastructure retries$", "", reason)
    row["stop_reason"] = reason.split(": ", 1)[1] if ": " in reason else ""
    doc = load_json(cdir / "result.json")
    if doc is None:
        row.update(state="running" if fin is None else "ungraded", status=None, valid_rb=False, rb=None, readout=None,
                   node_runs=len(run_calls(ev)), failed=0, qpu=None, wall=None, cost=0.0, ballpark=None, graded=None, outside=[])
        return finish_row(row)
    x = doc["targets"][0]
    runs = run_calls(ev)
    rb_runs = [e for e in runs if e.get("node") == "Randomized_benchmarking"]
    lab = g(load_json(work / "source-state/state.json"), "qubits", q) or {}
    final = g(load_json(cdir / "quam_state/state.json"), "qubits", q) or {}
    x180 = g(final, "xy", "operations", "x180_DragCosine") or {}
    lab_x180 = g(lab, "xy", "operations", "x180_DragCosine") or {}
    qual = x.get("quality") or {}
    t = g(x, "agent", "time") or {}
    judge = g(x, "judge", "identity") or {}
    row.update({
        "status": x.get("status"), "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
        "rb": g(qual, "rb", "error_per_clifford"), "rb_cover": numeric(rb_runs[-1], "decay_lengths_covered") if rb_runs else None,
        "rb_ok": bool(rb_runs) and rb_runs[-1].get("outcome") == "successful",
        "readout": g(qual, "readout", "assignment_fidelity"), "t1": g(qual, "coherence_limit", "t1_s"),
        "t2e": g(qual, "coherence_limit", "t2echo_s"),
        "x180_amp": x180.get("amplitude"), "x180_len": x180.get("length"), "alpha": x180.get("alpha"),
        "lab_x180_amp": lab_x180.get("amplitude"), "lab_x180_len": lab_x180.get("length"), "lab_alpha": lab_x180.get("alpha"),
        "ro_amp": g(final, "resonator", "operations", "readout", "amplitude"),
        "lab_ro_amp": g(lab, "resonator", "operations", "readout", "amplitude"),
        "df01": (final["f_01"] - lab["f_01"]) / 1e6 if final.get("f_01") and lab.get("f_01") else None,
        "joint": g(final, "z", "joint_offset"), "lab_joint": g(lab, "z", "joint_offset"),
        "lab_drive_offset": ((g(lab, "xy", "RF_frequency") - lab["f_01"]) / 1e6
                             if isinstance(g(lab, "xy", "RF_frequency"), (int, float)) and lab.get("f_01") else None),
        "fine_f01": next((pu.get("proposed") for e in runs if e.get("node") == "qubit_spectroscopy_fine"
                          for pu in e.get("proposed_updates") or [] if pu["path"].endswith(f"/{q}/f_01")), None),
        "qpu": t.get("qpu_execution_s"), "wall": t.get("total_s") or g(doc, "totals", "time", "total_s"),
        "node_runs": len(runs), "failed": sum(1 for e in runs if e.get("outcome") != "successful"),
        "ballpark": judge.get("in_ballpark"), "graded": judge.get("graded"), "outside": judge.get("outside") or [],
        "cost": g(doc, "totals", "tokens", "estimated_cost_usd") or 0.0, "spec_hash": g(doc, "scramble", "spec_hash"),
    })
    row["valid_rb"] = row["rb"] is not None and row["rb_ok"] and (row["rb_cover"] is None or row["rb_cover"] >= RB_VALID)
    row["state"] = "completed" if row["status"] == "completed" else ("stuck" if row["status"] == "escalated" else row["status"])
    return finish_row(row)


def finish_row(row: dict) -> dict:
    """Fault window, stop cause and outcome class."""
    rs = resume_stamp()
    end = stamp_time(rs) if rs else None
    ft = row["finish_ts"]
    row["fault"] = (row["backend"] == "arbel" and ft is not None and ft >= FAULT_FROM and (end is None or ft < end)
                    and row["state"] not in ("running",))
    if row["fault"]:
        row["state"] = "fault"
    row["cause"], row["cause_data"] = stop_cause(row)
    row["outcome"] = {"completed": "ok", "stuck": "bad", "aborted": "bad", "fault": "fault"}.get(row["state"], "none")
    row["why"] = "" if row["state"] == "completed" else (row["stop_reason"] or row["state"])
    return row


def stop_cause(row: dict) -> tuple[str | None, dict]:
    if row["state"] == "completed" or row["state"] in ("running", "ungraded"):
        return None, {}
    if row["fault"]:
        return "fault", {}
    ev = row["ev"]
    refused = [r for e in ev if e.get("kind") == "tool_call" and e.get("tool") == "write_state" for r in e.get("results") or []
               if r.get("status") == "refused" and r.get("path", "").endswith("/length") and "read-only" in (r.get("reason") or "")]
    pr = [e for e in run_calls(ev) if e.get("node") == "power_rabi"]
    na = next_action(pr[-1]) if pr else None
    if refused and na and na.get("reason") == "pi_above_amplitude_limit":
        last = pr[-1]
        return "length", {"needed": refused[-1].get("value"), "current": numeric(last, "pulse_length_ns"),
                          "pi_amp": numeric(last, "opt_amp"), "cap": numeric(last, "amp_limit_v")}
    if "sweep_too_long_for_t1" in row["summary_head"]:
        m = re.search(r"T1 ~ ([\d.]+) us", row["summary_head"])
        return "t1", {"t1_us": float(m.group(1)) if m else None}
    return "other", {}


def not_run_row(backend: str, q: str) -> dict:
    return {"night": "e1", "backend": backend, "q": q, "state": "not run", "status": None, "outcome": "none", "why": "not run",
            "valid_rb": False, "rb": None, "readout": None, "fault": False, "cause": None, "log": [], "engine": None, "ev": []}


def collect_e1(keys: list[tuple[str, str]]) -> dict:
    lx = launches()
    rows = {}
    for be, q in keys:
        cands = [w for w in work_dirs(be) if (w / f"{be}-engine-{q}").is_dir()]
        if not cands:
            rows[(be, q)] = not_run_row(be, q)
            continue
        work = cands[-1]
        stamp = work.name.split("-", 2)[2]
        rows[(be, q)] = e1_cell(be, q, work, lx.get((be, q, stamp)))
    return rows


def void_wave() -> dict:
    """The first arbel wave (config error at open_qm), for the incident text."""
    work = QAB / sorted(VOID_WORK)[0]
    cells = sorted(p for p in work.glob("arbel-engine-*") if p.is_dir())
    runs, config, keys, stops = 0, 0, collections.Counter(), collections.Counter()
    for c in cells:
        q = c.name.rsplit("-", 1)[1]
        ev = read_events(f"e1_arbel-{q}_{STAMP}", q)
        for e in run_calls(ev):
            runs += 1
            err = e.get("error") or ""
            if "CONFIG ERROR" in err:
                config += 1
                m = re.search(r'CONFIG ERROR in key "([^"]+)" \[([^\]]+)\]', err)
                if m:
                    keys[f"{m.group(1)} [{m.group(2)}]"] += 1
        fin = next((e for e in reversed(ev) if e.get("kind") == "finish"), None)
        if fin:
            stops[fin.get("status")] += 1
    return {"cells": len(cells), "qubits": [c.name.rsplit("-", 1)[1] for c in cells], "runs": runs, "config": config,
            "keys": keys, "stops": stops}


def infra_incident() -> dict:
    """IQCC 500s and bare 'Exception:' errors across every e1 events file, void cells included."""
    e500, bare = [], []
    for d in sorted(TINY_RUNS.glob("e1_*_2026*")):
        name = d.name
        m = re.match(r"e1_(\w+)-(\w+)_([\d-]+)$", name)
        if not m:
            continue
        be, q, stamp = m.groups()
        ev = read_events(name, q)
        for seg in segments(ev):
            fin = next((e for e in reversed(seg) if e.get("kind") == "finish"), None)
            log = "\n".join((fin or {}).get("summary", "").splitlines()[1:])
            for e in run_calls(seg):
                err = (e.get("error") or "").strip()
                if "500 Internal Server Error" in err:
                    e500.append((ts(e), be, q, stamp))
                elif re.fullmatch(r"Exception:?", err):
                    node = e.get("node")
                    if f"{node}: infrastructure failure (Exception)" in log:
                        what = "retried as infrastructure"
                    elif f"{node}: continue_without (Exception" in log:
                        what = "counted as a node failure; the step was skipped and the stored values kept"
                    elif f"{node}: escalate (Exception" in log:
                        what = "counted as a node failure on a critical step: the qubit stopped"
                    else:
                        what = "—"
                    bare.append({"t": ts(e), "be": be, "q": q, "stamp": stamp, "node": node, "what": what,
                                 "void": f"e1-{be}-{stamp}" in VOID_WORK})
    return {"e500": sorted(e500, key=lambda x: x[0]), "bare": sorted(bare, key=lambda x: x["t"])}


def fault_timeline(e1: dict, n13: dict) -> list[dict]:
    """Every arbel cell in every counted arbel work dir, in start order: the readout-side numbers that show the fault.
    A cell a later work dir superseded (the resume) stays here as the evidence."""
    out = []
    be = "arbel"
    for work in work_dirs(be):
        for cdir in sorted(p for p in work.glob(f"{be}-engine-*") if p.is_dir() and ".try" not in p.name):
            q = cdir.name.rsplit("-", 1)[1]
            latest = e1.get((be, q))
            r = latest if latest and latest.get("work") == work else e1_cell(be, q, work, None)
            if r.get("ev"):
                out.append(timeline_row(r, q, n13, superseded=r is not latest))
    return sorted(out, key=lambda x: x["start"] or datetime.max.replace(tzinfo=CEST))


def timeline_row(r: dict, q: str, n13: dict, superseded: bool) -> dict:
    be = "arbel"
    runs = run_calls(r["ev"])
    by = collections.defaultdict(list)
    for e in runs:
        by[e.get("node")].append(e)
    n13ev = read_events(n13[(be, q)]["run_id"], q) if (be, q) in n13 else []
    n13_qs = [e for e in run_calls(n13ev) if e.get("node") == "qubit_spectroscopy"]
    first_line = next((i + 1 for i, e in enumerate(n13_qs) if numeric(e, "line_identity") == "0-1"), None)
    n13_rid = [numeric(e, "contrast_snr") for e in run_calls(n13ev) if e.get("node") == "resonator_identification"]
    return {
        "q": q, "r": r, "superseded": superseded, "start": ts(runs[0]) if runs else r["start_ts"], "end": r["finish_ts"],
        "runs": [(ts(e), e.get("node"), e) for e in runs],
        "snr": [numeric(e, "contrast_snr") for e in by["resonator_identification"]],
        "plateau": [numeric(e, "plateau_contrast") for e in by["resonator_spectroscopy_vs_power"]],
        "lines": [numeric(e, "line_identity") for e in by["qubit_spectroscopy"]],
        "iq": [(ts(e), numeric(e, "readout_fidelity")) for e in by["IQ_blobs"]],
        "n13_line": first_line, "n13_windows": len(n13_qs), "n13_snr": n13_rid[0] if n13_rid else None,
        "n13_ro": n13[(be, q)]["readout"] if (be, q) in n13 else None,
    }


def probes() -> list[dict]:
    """arbel readout probes: the manual one in the operator log and e1c-resume.sh's."""
    out = []
    log = QAB / "e1-LOG.md"
    if log.exists():
        for ln in log.read_text(errors="replace").splitlines():
            m = re.match(r"^- (\d\d:\d\d) PROBE \(([^)]*)\): (\w+) IQ_blobs.*?-> ([\d.]+) % \(was ([\d.]+) %\)(?:, queue (\d+) s)?", ln)
            if m:
                out.append({"t": m.group(1), "src": f"manual, {m.group(2)}", "q": m.group(3), "fid": float(m.group(4)),
                            "was": float(m.group(5)), "queue": float(m.group(6)) if m.group(6) else None})
    rl = SCHED_LOG["e1c"]
    if rl.exists():
        for ln in rl.read_text(errors="replace").splitlines():
            m = re.match(r"^\[(\d\d:\d\d):\d\d\] e1c: probe (\w+) IQ_blobs: ([\d.]+) % \((.*)\)$", ln)
            if m:
                extra = load_json_str(m.group(4))
                out.append({"t": m.group(1), "src": "e1c-resume.sh", "q": m.group(2), "fid": float(m.group(3)), "was": None,
                            "queue": (extra or {}).get("queue_wait_s")})
            elif m := re.match(r"^\[(\d\d:\d\d):\d\d\] e1c: (?!probe|launched)(.*)$", ln):
                out.append({"t": m.group(1), "src": "e1c-resume.sh", "q": None, "fid": None, "was": None, "queue": None,
                            "text": m.group(2)})
    return out


def load_json_str(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return None


def na_reason_codes() -> dict[str, list[str]]:
    """The reason codes each converted node's analysis can publish, read from the run library's code."""
    src = (QL_SC / "calibration_utils/next_action.py").read_text()
    block = re.search(r"REASONS = frozenset\(\s*\{(.*?)\}\s*\)", src, re.S).group(1)
    reasons = set(re.findall(r'"(\w+)"', block))
    out = {}
    for node, _, pkg, _ in NA_NODES:
        text = (QL_SC / "calibration_utils" / pkg / "analysis.py").read_text(errors="replace")
        if node == "power_rabi_error_amplification_x180":  # applies power_rabi's rules
            text += (QL_SC / "calibration_utils/power_rabi/analysis.py").read_text(errors="replace")
        out[node] = sorted(reasons & set(re.findall(r'"(\w+)"', text)))
    return out


# ----------------------------------------------------------------------------- render helpers
LABEL = {"ok": "completed", "warn": "completed, not meaningful", "bad": "stopped", "fault": "hardware fault (void)",
         "none": "not run"}


def qtag(r: dict, extra: str = "") -> str:
    tip = f"{r['backend']} {r['q']} · {r['night']}: {r['state'] if r['night'] == 'e1' else LABEL[r['outcome']]}"
    if r.get("why"):
        tip += f": {r['why']}"
    elif r.get("valid_rb"):
        tip += f" · RB {100 * r['rb']:.3f} % · readout {100 * (r['readout'] or 0):.1f} %"
    if extra:
        tip += f" · {extra}"
    return (f"<span class='q q-{r['outcome']}' title='{esc(tip)}'><span class='dv' aria-label='{esc(r['backend'])}'>"
            f"{DEV[r['backend']]}</span>{esc(r['q'])}</span>")


def qlist(rows: list[dict]) -> str:
    if not rows:
        return "<span class='muted'>none</span>"
    return "<span class='ql'>" + "".join(qtag(r) for r in sorted(rows, key=n13r.order_key)) + "</span>"


def chip(r: dict | None) -> str:
    if r is None:
        return "<span class='muted'>—</span>"
    if r["night"] == "n13":
        text = {"ok": "completed", "warn": "not meaningful", "bad": r["status"]}[r["outcome"]]
    else:
        text = {"completed": "completed", "stuck": "stuck", "aborted": "aborted", "fault": "hw fault", "not run": "not run",
                "running": "running", "ungraded": "not graded"}.get(r["state"], r["state"])
    tip = r["why"] or LABEL[r["outcome"]]
    if r["night"] == "e1" and r["state"] == "stuck":
        tip = "result.json status escalated (the engine's 'stuck'): " + tip
    return f"<span class='oc oc-{r['outcome']}' title='{esc(tip)}'>{esc(text)}</span>"


def rb_text(r: dict | None) -> str:
    if not r or r.get("rb") is None:
        return "—"
    if not r["valid_rb"]:
        cov = r.get("rb_cover")
        why = f"{cov:.2f} decay lengths: not a measurement" if cov is not None else "the RB run failed: not a measurement"
        return f"<span title='{esc(why)}'>n/a</span>"
    return f"{100 * r['rb']:.3f}"


def has(r: dict | None, k: str) -> bool:
    return bool(r) and r.get(k) is not None


def fmt_x(r):
    return f"{r['x180_amp']:.3f}/{r['x180_len']}" if has(r, "x180_amp") else "—"


def fmt_ro(r):
    return f"{100 * r['readout']:.1f}" if has(r, "readout") else "—"


def fmt_df(r):
    return sg(r["df01"]) if has(r, "df01") else "—"


def fmt_alpha(r):
    return sg(r["alpha"], "{:+.2f}") if has(r, "alpha") else "—"


def fmt_runs(r):
    return f"{r['failed']}/{r['node_runs']}" if has(r, "node_runs") else "—"


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f} %"


def drive_offset_note(a: dict, b: dict) -> str:
    """The lab's state drives this qubit away from its stored f_01; the fine node measures around the drive frequency."""
    s = f"the lab's state drives it {sg(b['lab_drive_offset'], '{:+.1f}')} MHz from its stored f_01 (xy.RF_frequency)"
    if b.get("fine_f01") and b.get("df01") is not None:
        s += (f"; qubit_spectroscopy_fine measures around the drive frequency and proposed f_01 there, which the engine "
              f"committed ({sg(b['df01'], '{:+.1f}')} MHz); the n13 agent got the same proposal and kept f_01 at "
              f"{sg(a['df01'], '{:+.2f}')} MHz")
    return s


def short_reason(r: dict, n: int = 150) -> str:
    s = r.get("stop_reason") or r.get("why") or ""
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- aggregates
def agg(rows: list[dict]) -> dict:
    done = [r for r in rows if r["status"] == "completed"]
    rb = [r["rb"] for r in rows if r["valid_rb"]]
    ro = [r["readout"] for r in rows if r.get("readout") is not None]
    qpu = [r.get("qpu") or 0 for r in rows]
    wall = [r.get("wall") or 0 for r in rows]
    runs = [r.get("node_runs") or 0 for r in rows]
    return {
        "n": len(rows), "completed": len(done), "rb1": sum(1 for r in done if r["valid_rb"] and r["rb"] < 0.01),
        "valid_rb": len(rb), "rb_median": median(rb), "ro_median": median(ro), "ro_95": sum(1 for v in ro if v >= 0.95),
        "ballpark": sum(r.get("ballpark") or 0 for r in rows), "graded": sum(r.get("graded") or 0 for r in rows),
        "node_runs": sum(runs), "node_runs_median": median(runs), "failed": sum(r.get("failed") or 0 for r in rows),
        "qpu_total": sum(qpu) / 60, "qpu_median": (median(qpu) or 0) / 60, "wall_median": (median(wall) or 0) / 60,
        "cost": sum(r.get("cost") or 0 for r in rows), "infra": sum(r.get("infra") or 0 for r in rows),
    }


def summary_table(groups: list[tuple[str, str, list[tuple[str, str]]]], n13: dict, e1: dict) -> str:
    """groups: (title, subtitle, keys). Two columns per group: n13 on those qubits, e1 on those qubits."""
    stats = [(agg([n13[k] for k in keys]), agg([e1[k] for k in keys]), keys) for _, _, keys in groups]

    def rbm(s):
        return f"{s['valid_rb']}, median {100 * s['rb_median']:.3f} %" if s["rb_median"] is not None else f"{s['valid_rb']}"

    def cost(s, night):
        return f"${s['cost']:.2f}" if night == "n13" else f"${s['cost']:.2f} <span class='small muted'>(no model)</span>"

    lines = [
        ("Qubits, by outcome", lambda s, night, keys: qlist([(n13 if night == 'n13' else e1)[k] for k in keys]), True),
        ("Completed", lambda s, night, keys: f"{s['completed']}/{s['n']}", False),
        ("Completed with a valid RB under 1 %", lambda s, night, keys: f"{s['rb1']}", False),
        ("Valid RB (≥ 1 decay length), median error per Clifford", lambda s, night, keys: rbm(s), False),
        ("Both with a valid RB: median error per Clifford, same qubits", lambda s, night, keys: both_rb(keys, night), False),
        ("Readout assignment, median (qubits ≥ 95 %)", lambda s, night, keys: f"{pct(s['ro_median'])} ({s['ro_95']})", False),
        ("Judge: scrambled parameters back in range", lambda s, night, keys: f"{s['ballpark']}/{s['graded']}", False),
        ("Node runs, total / median per qubit", lambda s, night, keys: f"{s['node_runs']} / {s['node_runs_median']:.0f}", False),
        ("Failed node runs, total", lambda s, night, keys: f"{s['failed']}", False),
        ("QPU total / median per qubit", lambda s, night, keys: f"{s['qpu_total']:.0f} / {s['qpu_median']:.1f} min", False),
        ("Wall per qubit, median", lambda s, night, keys: f"{s['wall_median']:.0f} min", False),
        ("Model cost (tinycal estimate)", lambda s, night, keys: cost(s, night), False),
    ]
    def both_rb(keys, night):
        both = [k for k in keys if n13[k]["valid_rb"] and e1[k]["valid_rb"]]
        m = median([(n13 if night == "n13" else e1)[k]["rb"] for k in both])
        return f"{len(both)}, median {100 * m:.3f} %" if m is not None else "—"

    body = []
    for lab, fn, wide in lines:
        cells = []
        for sn, se, keys in stats:
            for s, night in ((sn, "n13"), (se, "e1")):
                cells.append(f"<td class='{'wrap' if wide else 'num'}{' gs' if night == 'n13' else ''}'>{fn(s, night, keys)}</td>")
        body.append(f"<tr><th class='rowh'>{esc(lab)}</th>{''.join(cells)}</tr>")
    head1 = "".join(f"<th class='grp gs' colspan='2'>{title}<div class='small muted' style='font-weight:400'>{sub}</div></th>"
                    for title, sub, _ in groups)
    head2 = "".join("<th class='gs'>n13 <span class='small muted'>agent</span></th><th>e1 <span class='small muted'>engine</span></th>"
                    for _ in groups)
    return (f"<div class='scroll'><table class='grid pivot sumt e1sum'><thead><tr><th></th>{head1}</tr><tr><th></th>{head2}</tr>"
            f"</thead><tbody>{''.join(body)}</tbody></table></div>")


def legend() -> str:
    demo = lambda outc, name, dv, tip: f"<span class='q q-{outc}' title='{esc(tip)}'><span class='dv'>{dv}</span>{name}</span>"  # noqa: E731
    return ("<div class='qlegend'>"
            f"<span>{demo('ok', 'Q1', 'q', 'qolab Q1')} completed</span>"
            f"<span>{demo('warn', 'qA5', 'a', 'arbel qA5')} completed, not meaningful (n13 only)</span>"
            f"<span>{demo('bad', 'qC3', 'g', 'gilboa qC3')} stopped (the engine's stuck / aborted)</span>"
            f"<span>{demo('fault', 'qA2', 'a', 'arbel qA2')} arbel readout fault: void</span>"
            f"<span>{demo('none', 'qC1', 'a', 'arbel qC1')} not run</span>"
            "<span class='muted'>device letter: <b class='mono'>a</b> arbel · <b class='mono'>g</b> gilboa · <b class='mono'>q</b> qolab; "
            "hover a name for the backend and the reason</span></div>")


# ----------------------------------------------------------------------------- sections
def engine_section(pol_a, pol_b, sha_a: str, sha_b: str, e1: dict, counted: list[tuple[str, str]]) -> str:
    crit = [s.node for s in pol_b.steps if s.critical]
    codes = na_reason_codes()
    converted = {n for n, *_ in NA_NODES}
    others = [n for n in dict.fromkeys(s.node for s in pol_b.steps) if n not in converted]
    seen = collections.defaultdict(collections.Counter)
    sources = collections.Counter()
    for k in counted:
        for d in e1[k]["log"]:
            if d["kind"] in ("infra", "skipped", "keep"):
                sources[d["kind"]] += 1
                continue
            sources[d["src"]] += 1
            seen[d["node"]][(d["kind"], d["reason"])] += 1

    def seen_text(node):
        c = seen.get(node)
        if not c:
            return "<span class='muted'>—</span>"
        parts = sorted(c.items(), key=lambda kv: (kv[0][0] != "accept", -kv[1]))
        return " · ".join(f"<span class='mono'>{esc(kind)}</span>" + ("" if reason in ("measured", "successful") else
                          f" <span class='muted'>{esc(reason)}</span>") + f" ×{n}" for (kind, reason), n in parts)

    rows = "".join(
        f"<tr><td><span class='mono'>{esc(node)}</span><div class='small muted'>{esc(lib)}</div></td><td>{rule}</td>"
        f"<td class='small'>{' '.join(f'<span class=mono>{esc(c)}</span>' for c in codes[node])}</td>"
        f"<td class='small'>{seen_text(node)}</td></tr>"
        for node, lib, _, rule in NA_NODES)
    nat = f"<div class='scroll'><table class='grid small natab'><thead><tr><th>node</th><th>main rules</th>" \
          f"<th>reason codes in its code</th><th>decisions in e1 (like-for-like cells)</th></tr></thead><tbody>{rows}</tbody></table></div>"
    n_dec = sum(v for k, v in sources.items() if k not in ("infra", "skipped", "keep"))
    src_txt = ", ".join(f"{sources[s]} from {lab}" for s, lab in (("node", "the node's next_action"),
                                                                  ("outcome", "a bare outcome (node without next_action)"),
                                                                  ("retry_parameters", "legacy retry_parameters"),
                                                                  ("policy", "the graph policy (a failure the node gave no action for)"),
                                                                  ("runner", "the runner")) if sources[s])
    return f"""
<p>qua-decision-engine (<span class="mono">~/code/QM/qua-decision-engine</span>, local git) walks the bring-up graph with no
language model. After every node run it decides from the node's structured output only, in this order: an infrastructure failure
(the runner's exception type) is retried; otherwise the node's <span class="mono">next_action</span>; otherwise its legacy
<span class="mono">retry_parameters</span>; otherwise its outcome, with the graph deciding what a bare failure means. It never
reads a warning's text.</p>
<p>A node's <span class="mono">next_action</span> (qua-libs <span class="mono">calibration_utils/next_action.py</span>) is one of
<span class="mono">accept</span>, <span class="mono">rerun</span> (with absolute parameters), <span class="mono">continue_without</span>,
<span class="mono">step_back</span> (naming the node) or <span class="mono">escalate</span>, with a stable reason code and a flag
saying whether this run's proposals may be committed. It travels as JSON in each qubit's fit results, as
<span class="mono">retry_parameters</span> already did, so tinycal passes it through unchanged.</p>
<p>The graph is data (<span class="mono">src/qde/policy.py</span>): {len(pol_b.steps)} steps over {pol_b.graph_nodes} distinct nodes
(the power sweep repeats at the sweet spot; the I/Q T1 runs only when T1_chirp stored none). Critical steps, where a failure the
node gives no action for stops the qubit: {', '.join(f'<span class="mono">{esc(c)}</span>' for c in crit)}. Elsewhere such a
failure keeps the stored values and the walk goes on. Budgets: {pol_b.max_runs} node runs per qubit
({pol_b.run_budget_factor:g} × {pol_b.graph_nodes} graph nodes), {pol_b.steps[0].max_runs} runs per step per visit,
{pol_b.long_max_runs} for the line search (<span class="mono">{esc(', '.join(pol_b.long_reasons))}</span>), never the same node with
the same parameters on the same state twice, each step-back edge once. Infrastructure retries count apart: at most
{pol_b.max_infra_in_a_row} in a row and {pol_b.max_infra_total} in total with a {pol_b.infra_backoff_s:.0f} s × n backoff in
<span class="mono">{esc(sha_b)}</span> ({pol_a.max_infra_in_a_row}, {pol_a.max_infra_total} and {pol_a.infra_backoff_s:.0f} s in
<span class="mono">{esc(sha_a)}</span>).</p>
<p>It runs on tinycal's toolbox (node runner, state store with its write checks, recorder), so node execution, the benchmark
profile's read-only paths, the QPU budget and <span class="mono">events.jsonl</span> are exactly what an agent run gets. Each
decision is logged as a model turn with no model and no tokens, and <span class="mono">project_tinycal.py</span> and
<span class="mono">qab</span> grade an engine run unchanged. Commits use tinycal's <span class="mono">from_run</span> writes: the
value a run proposed, never a retyped one.</p>
<p class="small muted">In the {len(counted)} like-for-like e1 cells the engine took {n_dec} decisions: {src_txt}; plus
{sources['infra']} infrastructure retries and {sources['skipped']} conditional steps skipped.</p>
<h3>Which nodes publish next_action</h3>
<p class="small muted">Converted in qua-libs <span class="mono">ad9dc4ee</span> ("Nodes publish a structured next_action beside their
warnings"); the line search's rules came with <span class="mono">e7b13820</span>. The rules follow what the n13 agent did by hand
and were replayed on archived runs. The nodes without one ({', '.join(f'<span class="mono">{esc(n)}</span>' for n in others)}) are
decided by their outcome.</p>
{nat}"""


def per_qubit_table(n13: dict, e1: dict) -> str:
    body = []
    for backend in BACKENDS:
        keys = sorted(k for k in n13 if k[0] == backend)
        ran = sum(1 for k in keys if e1[k]["state"] != "not run")
        body.append(f"<tr class='gh'><th colspan='13'>{esc(backend)} <span class='muted'>· {len(keys)} qubits, e1 ran {ran}</span></th></tr>")
        for key in keys:
            a, b = n13[key], e1[key]
            rb_cls = ""
            if a["valid_rb"] and b["valid_rb"]:
                ratio = b["rb"] / a["rb"]
                rb_cls = "worse" if ratio > WORSE_RB else "better" if ratio < 1 / WORSE_RB else ""
            notes = []
            if b["state"] not in ("completed", "not run"):
                notes.append(f"<span class='why'>{esc(short_reason(b))}</span>")
            if ROW_NOTE.get(key) and b["state"] != "not run":
                notes.append(esc(ROW_NOTE[key](a, b)))
            if b.get("lab_drive_offset") and abs(b["lab_drive_offset"]) > 10:
                notes.append(esc(drive_offset_note(a, b)))
            if b["state"] != "fault":
                went = collections.Counter()
                for d in b["log"]:
                    if d["kind"] == "continue_without" and d["reason"] == "Exception":
                        went[f"{d['node']} skipped: a bare 'Exception:' from the cloud client, which engine {b['engine']} "
                             "did not retry; the stored values were kept"] += 1
                    elif d["kind"] == "continue_without":
                        what = "went on with its fallback committed" if d.get("commit") else "kept the stored values"
                        went[f"{d['node']}: {what} ({d['reason']}, from {d['src']})"] += 1
                    elif d["kind"] == "keep":
                        went[f"{d['node']}: {d['reason']}"] += 1
                notes += [esc(k + (f" ×{v}" if v > 1 else "")) for k, v in went.items()]
            if b.get("earlier_runs"):
                notes.append(f"its events file also holds an earlier try ({plural(b['earlier_runs'], 'run')}, "
                             f"{b['earlier_failed']} failed), left out of the counts")
            if b["state"] == "not run" and resume_stamp():
                notes.append(f"queued for the arbel resume ({esc(resume_stamp())})")
            eng = (f"<span class='etag' title='{esc(b['sched'])} scheduler'>{esc(b['engine'])}</span>" if b.get("engine") else "")
            is_bad = b["outcome"] in ("bad", "fault")
            body.append(
                f"<tr class='{'fz' if b['outcome'] in ('fault', 'none') else ''}'><td class='qcell'>{qtag(b)}</td>"
                f"<td class='num'>{chip(a)}<span class='arr'>→</span>{chip(b)}</td>"
                f"<td class='num'>{pair(rb_text(a), rb_text(b), rb_cls)}</td>"
                f"<td class='num'>{pair(fmt_ro(a), fmt_ro(b))}</td>"
                f"<td class='num'>{pair(fmt_df(a), fmt_df(b))}</td>"
                f"<td class='num'>{pair(fmt_x(a), fmt_x(b))}</td>"
                f"<td class='num'>{pair(fmt_alpha(a), fmt_alpha(b))}</td>"
                f"<td class='num'>{pair(mins(a['qpu'], '{:.1f}'), mins(b.get('qpu'), '{:.1f}'))}</td>"
                f"<td class='num'>{pair(mins(a['wall']), mins(b.get('wall')))}</td>"
                f"<td class='num'>{pair(fmt_runs(a), fmt_runs(b))}</td>"
                f"<td class='num'>{pair(judge_text(a), judge_text(b) if b.get('graded') is not None else '—')}</td>"
                f"<td class='num'>{eng}</td>"
                f"<td class='small note{' stop' if is_bad else ''}'>{'; '.join(notes)}</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>outcome n13 → e1</th><th>RB error / Clifford, %</th><th>readout, %</th>"
            "<th>f<sub>01</sub> − lab, MHz</th><th>x180 amp / ns</th><th>DRAG α</th><th>QPU, min</th><th>wall, min</th>"
            "<th>failed / node runs</th><th>judge</th><th>engine</th><th>e1: the engine's stop reason, notes</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small pq e1pq'>{head}<tbody>{''.join(body)}</tbody></table></div>"


def stops_section(n13: dict, e1: dict, timeline: list[dict], probe_rows: list[dict], fault_nums: dict) -> str:
    n_void = sum(1 for t in timeline if t["r"]["fault"])
    by = collections.defaultdict(list)
    for k, r in e1.items():
        if r["cause"]:
            by[r["cause"]].append(r)
    parts = []
    short = {"length": "on a read-only pulse length", "t1": "on a T1 too short for the line search",
             "fault": "inside the arbel readout fault", "other": "for other reasons"}
    counts = "; ".join(f"{len(by[c])} {short[c]}" for c in ("length", "t1", "fault", "other") if by[c])
    parts.append(f"<p>e1 did not complete {sum(len(v) for v in by.values())} of the qubits it ran: {esc(counts)}. Grouped by "
                 "cause, from the engine's finish summary, the refused writes and the stopping run's next_action.</p>")
    for cause in ("length", "t1", "other"):
        rows = sorted(by.get(cause, []), key=n13r.order_key)
        if not rows:
            continue
        title, text = STOP_GROUPS[cause]
        trs = []
        for r in rows:
            a = n13[(r["backend"], r["q"])]
            d = r["cause_data"]
            if cause == "length":
                what = (f"π at {d['current']:.0f} ns needs {d['pi_amp']:.3f}, above the {d['cap']:g} cap: power_rabi asks for "
                        f"<b>{d['needed']} ns</b>; the store refuses the length")
                if r.get("lab_drive_offset") and abs(r["lab_drive_offset"]) > 10:
                    what += (f". Also: {esc(drive_offset_note(a, r))}. The lab's own x180 at that drive frequency is above the "
                             f"cap as well, so the stop does not depend on it")
                lab = f"{r['lab_x180_amp']:.3f} at {r['lab_x180_len']} ns"
            elif cause == "t1":
                what = (f"T1 ≈ {d['t1_us']:g} µs: the line search refuses (<span class='mono'>sweep_too_long_for_t1</span>)"
                        if d.get("t1_us") else esc(short_reason(r)))
                lab = f"T1 {r['lab_t1']:.1f} µs" if r.get("lab_t1") else "—"
            else:
                what, lab = esc(short_reason(r, 240)), "—"
            n13_txt = (f"{chip(a)} x180 {fmt_x(a)} ns · RB {rb_text(a)} % · readout {fmt_ro(a)} %"
                       + (f" · T1 {1e6 * a['t1']:.1f} µs" if cause == "t1" and a.get("t1") else ""))
            trs.append(f"<tr><td class='qcell'>{qtag(r)}<div class='small muted'>{esc(r['backend'])} · engine {esc(r['engine'])}</div></td>"
                       f"<td>{what}<div class='small muted mono'>{esc(short_reason(r, 200))}</div></td>"
                       f"<td class='num'>{lab}</td><td class='small'>{n13_txt}</td></tr>")
        lab_head = {"length": "lab's x180", "t1": "lab", "other": ""}[cause]
        parts.append(f"<h3>{esc(title)} <span class='muted small'>· {len(rows)}</span></h3><p class='small'>{text}</p>"
                     f"<div class='scroll'><table class='grid small stops'><thead><tr><th>qubit</th><th>what the data says; the engine's reason</th>"
                     f"<th>{lab_head}</th><th>n13 agent, same qubit</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>")
    if by.get("fault") or timeline:
        title, text = STOP_GROUPS["fault"]
        parts.append(f"<h3 id='fault'>{esc(title)} <span class='muted small'>· {n_void} void cells</span></h3>"
                     f"<p class='small'>{text.format(**fault_nums)}</p>{fault_table(timeline)}{probe_table(probe_rows)}")
    return "".join(parts)


def fault_table(timeline: list[dict]) -> str:
    def nums(xs, spec):
        xs = [x for x in xs if x is not None]
        return " · ".join(spec.format(x) for x in xs) if xs else "—"

    def lines(xs):
        if not xs:
            return "—"
        c = collections.Counter(xs)
        return " · ".join(f"{esc(k)}{'' if v == 1 else f' ×{v}'}" for k, v in c.items())

    trs = []
    for t in timeline:
        r = t["r"]
        iq = " · ".join(f"<b>{v:.1f} %</b> <span class='muted'>{hm(tt)}</span>" for tt, v in t["iq"] if v is not None) or "—"
        n13_line = (f"0→1 in window {t['n13_line']}" if t["n13_line"] else "no line") if t["n13_windows"] else "—"
        n13_snr = f"SNR {t['n13_snr']:.1f}" if t["n13_snr"] is not None else "SNR —"
        trs.append(
            f"<tr class='{'fz' if r['fault'] else ''}'><td class='qcell'>{qtag(r)}</td>"
            f"<td class='num'>{hm(t['start'])}–{hm(t['end'])}</td>"
            f"<td class='num'>{nums(t['snr'], '{:.1f}')}</td><td class='num'>{nums(t['plateau'], '{:.2f}')}</td>"
            f"<td class='num'>{lines(t['lines'])}</td><td class='num'>{iq}</td>"
            f"<td>{chip(r)}{f'<div class=\"small muted\">rerun in {esc(resume_stamp())}</div>' if t['superseded'] else ''}</td>"
            f"<td class='num small muted'>{n13_snr} · {n13_line} · IQ {fmt_ro({'readout': t['n13_ro']})} %</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>ran</th><th>resonator_identification contrast SNR</th>"
            "<th>resonator_spectroscopy_vs_power plateau contrast</th><th>qubit_spectroscopy line identity</th>"
            "<th>IQ_blobs assignment</th><th>e1</th><th>n13, same qubit</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small ftab'>{head}<tbody>{''.join(trs)}</tbody></table></div>"


def probe_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    trs = []
    for p in rows:
        if p.get("fid") is None:
            trs.append(f"<tr><td class='num'>{esc(p['t'])}</td><td colspan='3' class='small'>{esc(p.get('text', ''))}</td>"
                       f"<td class='small muted'>{esc(p['src'])}</td></tr>")
            continue
        was = f" (was {p['was']:.1f} %)" if p.get("was") else ""
        q = f"{p['queue']:.0f} s" if p.get("queue") is not None else "—"
        trs.append(f"<tr><td class='num'>{esc(p['t'])}</td><td class='mono'>arbel {esc(p['q'])}</td>"
                   f"<td class='num'><b>{p['fid']:.1f} %</b>{esc(was)}</td><td class='num'>{q}</td>"
                   f"<td class='small muted'>{esc(p['src'])}</td></tr>")
    return ("<h4>Readout probes since the fault</h4><p class='small muted'>One IQ_blobs on qA1 from the state e1 left it in at "
            "14:58, when it read 96.1 %. The resume starts at ≥ 90 %.</p>"
            "<div class='scroll'><table class='grid small probes'><thead><tr><th>time</th><th>qubit</th><th>IQ_blobs</th>"
            f"<th>queue</th><th>source</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>")


def incidents_section(vw: dict, inf: dict, e1: dict) -> str:
    keys = "; ".join(f"<span class='mono'>{esc(k)}</span> ({v})" for k, v in vw["keys"].items())
    stops = " and ".join(f"{v} {esc(k)}" for k, v in vw["stops"].items())
    rs = resume_stamp()
    e5 = inf["e500"]
    be500 = collections.Counter(b for _, b, _, _ in e5)
    win = f"{hm(e5[0][0], True)}–{hm(e5[-1][0], True)}" if e5 else "—"
    bare = "".join(
        f"<li><span class='mono'>{hm(b['t'], True)}</span> {esc(b['be'])} {esc(b['q'])}{' (void wave)' if b['void'] else ''}, "
        f"<span class='mono'>{esc(b['node'])}</span>, engine {esc(engine_of(e1, b))}: {esc(b['what'])}.</li>" for b in inf["bare"])
    return f"""
<ul class="incidents">
<li><b>(a) The first arbel wave was void.</b> {vw['cells']} arbel cells ({', '.join(esc(q) for q in vw['qubits'])}) ran
{vw['runs']} node runs between them, {vw['config']} of which failed at <span class="mono">open_qm</span> with {keys}. The fresh
13:33 arbel state reads out with a DrachmaReadoutPulse; after scrambling, qB5's waveform reached a sample of 1.0014 &gt; 1, and
because the config holds every qubit, every arbel cell failed. Engine c1e0a5c called the <span class="mono">CloudExecutionError</span>
infrastructure and retried it; the cells ended {stops}. The 29 Sep pull n12/n13 used no longer loads under quam-builder 0.6.0, so arbel was rebuilt
at 14:24 from the same day's pull with every readout pointed back at its square pulse (<span class="mono">readout_square</span>, still
in the state), which is what n12 and n13 ran with (<span class="mono">~/qab-runs/e1-prepare-arbel-square.sh</span>, stamp
<span class="mono">{esc(ARBEL_STAMP)}</span>). Only that work dir{f' and the resume&#x27;s (<span class="mono">{esc(rs)}</span>)' if rs else ''} count.</li>
<li><b>(b) IQCC 500s and bare exceptions.</b> IQCC returned 500 Internal Server Error on all three backends from {win}
({', '.join(f'{v} on {esc(k)}' for k, v in sorted(be500.items()))}), and the cloud client raised bare
<span class="mono">Exception:</span> errors with no message:<ul class="tight">{bare}</ul>
Engine 9f2ffcb, which ran every cell launched from 14:25, retries a bare Exception as infrastructure (backoff 30/60/90/120 s,
4 in a row, 12 in total) and no longer retries a <span class="mono">CONFIG ERROR</span>.</li>
<li><b>(c) The chirp line search was ported.</b> n13's qD1 fix (the tiled line search) never reached feat/qualibrate-ai; only its
drive-window half was committed (<span class="mono">13ff96ba</span>). It was ported into e1's library as
<span class="mono">e7b13820</span> (3-way apply, one import conflict; suite 1805 passed). With it the engine found arbel qD1's
line.</li>
</ul>"""


def engine_of(e1: dict, b: dict) -> str:
    r = e1.get((b["be"], b["q"]))
    if b["void"]:
        return git(QDE["e1"], "rev-parse", "--short", "HEAD")
    if r and r.get("stamp") == b["stamp"] and r.get("earlier_runs") and b["t"] < (r.get("start_ts") or b["t"]):
        return git(QDE["e1"], "rev-parse", "--short", "HEAD") + " (first try)"
    return (r or {}).get("engine") or "?"


def worse_section(n13: dict, e1: dict, counted: list[tuple[str, str]]) -> tuple[str, list, list]:
    worse, better = [], []
    for k in counted:
        a, b = n13[k], e1[k]
        why = []
        if a["valid_rb"] and b["valid_rb"] and b["rb"] > WORSE_RB * a["rb"]:
            why.append("rb")
        if a.get("readout") is not None and b.get("readout") is not None and a["readout"] - b["readout"] > WORSE_RO:
            why.append("ro")
        if why and b["status"] == "completed":
            worse.append((k, why))
        if a["valid_rb"] and b["valid_rb"] and b["rb"] < a["rb"] / WORSE_RB:
            better.append(k)

    def side(r, lab_note=True):
        ratio = r["ro_amp"] / r["lab_ro_amp"] if r.get("ro_amp") and r.get("lab_ro_amp") else None
        bits = [f"x180 {fmt_x(r)} ns", f"α {fmt_alpha(r)}",
                f"T1/T2e {1e6 * r['t1']:.0f}/{1e6 * r['t2e']:.0f} µs" if r.get("t1") and r.get("t2e") else "",
                f"readout amp {ratio:.2f}× lab" if ratio else "",
                f"joint {sg(r['joint'], '{:+.4f}')} V" if r.get("joint") is not None else "",
                f"{r['failed']}/{r['node_runs']} failed"]
        return " · ".join(x for x in bits if x)

    rows = []
    for k, why in sorted(worse, key=lambda kw: n13r.order_key(e1[kw[0]])):
        a, b = n13[k], e1[k]
        decisions = [f"{d['node']}: " + ("infrastructure retry" if d["kind"] == "infra" else d["kind"]) + f" ({d['reason']})"
                     for d in b["log"] if d["kind"] not in ("accept", "skipped")]
        what = WORSE_WHY[k](a, b) if k in WORSE_WHY else "Cause not identified."
        lab = (f"lab α {sg(b['lab_alpha'], '{:+.2f}')}, joint {sg(b['lab_joint'], '{:+.4f}')} V"
               if b.get("lab_alpha") is not None and b.get("lab_joint") is not None else "")
        flag = " and ".join({"rb": "RB more than 30 % higher", "ro": "readout more than 2 points lower"}[w] for w in why)
        rows.append(
            f"<tr><td class='qcell'>{qtag(b)}<div class='small muted'>{esc(k[0])}</div></td>"
            f"<td class='num'>RB {rb_text(a)} %<br>readout {fmt_ro(a)} %<div class='small muted wrapn'>{side(a)}</div></td>"
            f"<td class='num'><b>RB {rb_text(b)} %</b><br><b>readout {fmt_ro(b)} %</b><div class='small muted wrapn'>{side(b)}</div></td>"
            f"<td><div class='small muted'>{esc(flag)}</div>{what}"
            f"<div class='small muted'>{esc(lab)}. The engine's decisions other than accept: "
            f"{esc('; '.join(decisions)) or 'none'}.</div></td></tr>")
    head = "<thead><tr><th>qubit</th><th>n13 agent</th><th>e1 engine</th><th>what happened</th></tr></thead>"
    table = f"<div class='scroll'><table class='grid small worse e1worse'>{head}<tbody>{''.join(rows)}</tbody></table></div>"
    return table, worse, better


def setup_table(e1: dict, pol_b, counted_all: list[tuple[str, str]], n13_specs: list[str]) -> str:
    by_sched = collections.defaultdict(list)
    mismatch = []
    for k, r in e1.items():
        if r.get("sched"):
            by_sched[r["sched"]].append(r)
            if r.get("driver_engine") and r["driver_engine"] != r["engine"]:
                mismatch.append(f"{r['backend']} {r['q']}")
    sha = {s: git(p, "rev-parse", "--short", "HEAD") for s, p in QDE.items()}
    subj = {s: git(p, "log", "-1", "--format=%s") for s, p in QDE.items()}
    ql_log = git(QL_RUN, "log", "--format=%h %s", "-3").splitlines()
    tc = git(TINY_E1, "log", "-1", "--format=%h %s")
    specs = sorted({r.get("spec_hash") for r in e1.values() if r.get("spec_hash")})
    first = ", ".join(f"{r['backend']} {r['q']}" for r in sorted(by_sched["e1"], key=n13r.order_key))
    rs = resume_stamp()
    rows = [
        ("Engine", f"qua-decision-engine, frozen worktrees. <span class='mono'>~/qab-runs/qde-e1-run</span> at "
                   f"<span class='mono'>{esc(sha['e1'])}</span> ({esc(subj['e1'])}) for the first scheduler's cells that count: {esc(first)}; "
                   f"<span class='mono'>~/qab-runs/qde-e1b-run</span> at <span class='mono'>{esc(sha['e1b'])}</span> ({esc(subj['e1b'])}) "
                   f"for the other {len(by_sched['e1b']) + len(by_sched['e1c'])}. Attributed from the scheduler logs"
                   + (f"; the drivers' own logs disagree on {esc(', '.join(mismatch))}" if mismatch else "; the drivers' logs agree") + "."),
        ("Policy", f"<span class='mono'>{esc(pol_b.name)}</span> for graph <span class='mono'>{esc(pol_b.graph)}</span> "
                   f"(qua-libs graph 80): {len(pol_b.steps)} steps, {pol_b.graph_nodes} nodes, {pol_b.max_runs} node runs per qubit."),
        ("qua-libs", "<span class='mono'>~/qab-runs/qua-libs-e1-run</span>, branch feat/node-next-action on feat/qualibrate-ai: "
                     + " · ".join(f"<span class='mono'>{esc(x)}</span>" for x in ql_log) + ". A local branch; no remote holds it."),
        ("tinycal", f"<span class='mono'>~/qab-runs/tinycal-e1</span> at <span class='mono'>{esc(tc)}</span>, run through the main "
                    "tinycal venv. Benchmark profile: pulse and readout lengths read-only (since 931acd3); the 27-run budget is the "
                    "engine's own. One single-target cell per qubit."),
        ("Grading", "Unchanged from n13: <span class='mono'>scripts/project_tinycal.py</span>, then <span class='mono'>qab inspect-state</span>, "
                    "<span class='mono'>validate</span> and <span class='mono'>accept</span> (qua-agents-benchmark)."),
        ("States", f"Fresh pulls at 13:33 for all three backends (stamp <span class='mono'>{esc(STAMP)}</span>); arbel rebuilt at 14:24 "
                   f"with square readout (<span class='mono'>{esc(ARBEL_STAMP)}</span>). Scramble spec "
                   f"<span class='mono'>{esc(', '.join(specs))}</span>"
                   + (", the same as n13's" if specs == n13_specs else f" (n13: <span class='mono'>{esc(', '.join(n13_specs))}</span>)")
                   + ". gilboa's active_qubit_names set to the ten "
                   "C/D qubits, arbel's readout alias reversed, as for n13."
                   + (f" arbel resume: <span class='mono'>{esc(rs)}</span>." if rs else "")),
        ("Concurrency", "3 in flight per device on qolab and gilboa; arbel 2, 3 while its completed-run queue median stays under the "
                        "threshold (<span class='mono'>e1-arbel-cap.py</span>). By the user's instruction for this run."),
        ("Work dirs", "<span class='mono'>~/qab-runs/e1-&lt;backend&gt;-&lt;stamp&gt;/&lt;backend&gt;-engine-&lt;q&gt;/</span> (result.json, "
                      "grade.log, quam_state, run.log). Events: <span class='mono'>~/code/QM/tinycal/runs/e1_&lt;backend&gt;-&lt;q&gt;_&lt;stamp&gt;/"
                      "&lt;q&gt;/events.jsonl</span>. Ignored: <span class='mono'>e1-arbel-20260930-1333</span> (void) and "
                      "<span class='mono'>gilboa-engine-qD3.try1-bare-exception</span>. n13: "
                      f"<span class='mono'>~/qab-runs/n13-&lt;backend&gt;-{esc(n13r.NIGHTS['n13'])}/</span>."),
        ("Scripts", "In <span class='mono'>~/qab-runs/</span>: <span class='mono'>e1-prepare.sh</span> and "
                    "<span class='mono'>e1-prepare-arbel-square.sh</span> (states and scramble), <span class='mono'>e1-driver.sh</span> "
                    "(one cell: engine, projection, grading), <span class='mono'>e1-scheduler.sh</span> and "
                    "<span class='mono'>e1b-scheduler.sh</span> (queues), <span class='mono'>e1-arbel-cap.py</span> (arbel's cap), "
                    "<span class='mono'>e1-status.py</span> (one line per cell), <span class='mono'>e1-compare.py</span> (per-qubit "
                    "rows against n13), <span class='mono'>e1-trace.py</span> (one run's decisions), "
                    "<span class='mono'>e1c-resume.sh</span> (arbel resume). Operator log: <span class='mono'>e1-LOG.md</span>."),
        ("This page", "<span class='mono'>~/code/QM/qua-agents-benchmark-reports/make_e1_report.py</span> reads the cells, events, "
                      "scheduler logs and the engine's policy directly; rerun it to regenerate (it picks up the arbel resume)."),
    ]
    body = "".join(f"<tr><th class='rowh'>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='scroll'><table class='grid small setup'><tbody>{body}</tbody></table></div>"


# ----------------------------------------------------------------------------- page
E1_CSS = """
:root { --q-fault:#7f8a90; --q-fault-bg:#eef2f2; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --q-fault:#8a959b; --q-fault-bg:#22282b; } }
:root[data-theme="dark"] { --q-fault:#8a959b; --q-fault-bg:#22282b; }
.q-fault { color:var(--q-fault); text-decoration:line-through; text-decoration-thickness:1px; }
.q-none { color:var(--muted); font-weight:400; opacity:.75; }
.oc-fault { background:var(--q-fault-bg); color:var(--q-fault); border:1px dashed var(--line-strong); }
.oc-none { background:transparent; color:var(--muted); border:1px dashed var(--line); }
table.e1sum th.rowh { width:22%; } table.e1sum th.gs, table.e1sum td.gs { border-left:1px solid var(--line-strong); }
table.e1sum td.num { white-space:normal; }
table.e1pq { min-width:1260px; } table.e1pq td.note { min-width:260px; } table.e1pq td.note.stop .why { color:var(--q-bad); }
table.e1pq tr.fz td:not(.note):not(.qcell) { opacity:.6; }
.etag { font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--ink-2); border:1px solid var(--line); border-radius:4px; padding:0 4px; cursor:help; }
table.natab { min-width:900px; } table.natab td:nth-child(2) { width:42%; line-height:1.4; } table.natab td:nth-child(3) { width:20%; }
table.natab td:nth-child(3) .mono { display:inline-block; margin:0 4px 2px 0; font-size:11px; }
table.stops td:nth-child(2) { width:44%; } table.stops td { line-height:1.4; }
table.ftab { min-width:1040px; } table.ftab tr.fz td { background:var(--q-bad-bg); } table.ftab td.num { white-space:normal; }
table.probes { max-width:720px; }
table.e1worse td:nth-child(4) { width:46%; line-height:1.4; } .wrapn { white-space:normal; margin-top:4px; }
ul.incidents { margin:12px 0; padding-left:20px; max-width:92ch; } ul.incidents > li { margin:10px 0; line-height:1.5; }
.caveat { background:var(--warn-bg); border-left:3px solid var(--warn); padding:10px 14px; border-radius:6px; margin:14px 0; max-width:86ch; }
"""


def build():
    n13 = {}
    for backend in BACKENDS:
        work = QAB / f"n13-{backend}-{n13r.NIGHTS['n13']}"
        for cdir in sorted(work.glob(f"{backend}-tinycal-{n13r.KEY}-*")):
            if (cdir / "result.json").exists():
                q = cdir.name.rsplit("-", 1)[1]
                n13[(backend, q)] = n13r.cell_row("n13", backend, q)
    keys = sorted(n13, key=lambda k: (BACKENDS.index(k[0]), k[1]))
    e1 = collect_e1(keys)
    for k, r in e1.items():  # the lab's T1, for the T1 stop group
        if r.get("work"):
            r["lab_t1"] = (g(load_json(r["work"] / "source-state/state.json"), "qubits", k[1], "T1") or 0) * 1e6 or None

    ran = [k for k in keys if e1[k]["state"] not in ("not run",)]
    fault = [k for k in keys if e1[k]["state"] == "fault"]
    running = [k for k in keys if e1[k]["state"] in ("running", "ungraded")]
    not_run = [k for k in keys if e1[k]["state"] == "not run"]
    counted = [k for k in ran if k not in fault and k not in running]
    counted_all = [k for k in ran if k not in running]
    s_n, s_e = agg([n13[k] for k in counted]), agg([e1[k] for k in counted])

    pol_a = load_policy(QDE["e1"], "qde_policy_e1")
    pol_b = load_policy(QDE["e1b"], "qde_policy_e1b")
    sha_a, sha_b = git(QDE["e1"], "rev-parse", "--short", "HEAD"), git(QDE["e1b"], "rev-parse", "--short", "HEAD")
    timeline = fault_timeline(e1, n13)
    probe_rows = probes()
    vw = void_wave()
    inf = infra_incident()
    worse_html, worse, better = worse_section(n13, e1, counted)

    stops = {c: [e1[k] for k in counted_all if e1[k]["cause"] == c] for c in ("length", "t1", "other")}
    firsts = [e1[k]["start_ts"] for k in ran if e1[k].get("start_ts")]
    lasts = [e1[k]["finish_ts"] for k in ran if e1[k].get("finish_ts")]
    t0, t1 = (min(firsts) if firsts else None), (max(lasts) if lasts else None)
    now = datetime.now().strftime("%d %b %Y %H:%M")
    rs = resume_stamp()
    fault_end = stamp_time(rs) if rs else None

    def side_of(t):  # a node run before the fault, inside it, or after the resume started
        if t is None:
            return None
        return "pre" if t < FAULT_FROM else ("post" if fault_end is None or t < fault_end else None)

    split = {"pre": collections.defaultdict(list), "post": collections.defaultdict(list)}
    for t in timeline:
        for rt, node, e in t["runs"]:
            sd = side_of(rt)
            if sd:
                split[sd][node].append(e)

    def vals(side, node, key):
        return [v for v in (numeric(e, key) for e in split[side][node]) if isinstance(v, (int, float))]

    def rng(xs, spec="{:.0f}"):
        return f"{spec.format(min(xs))}–{spec.format(max(xs))}" if xs else "—"

    pre_iq, post_iq = vals("pre", "IQ_blobs", "readout_fidelity"), vals("post", "IQ_blobs", "readout_fidelity")
    pre_snr = vals("pre", "resonator_identification", "contrast_snr")
    post_snr = vals("post", "resonator_identification", "contrast_snr")
    no_line = [t["r"] for t in timeline if t["r"]["fault"] and t["lines"] and all(x != "0-1" for x in t["lines"]) and t["n13_line"]]
    fault_all = [t for t in timeline if t["r"]["fault"]]
    fault_rerun = [t for t in fault_all if t["superseded"]]
    fault_nums = {"pre_iq": rng(pre_iq, "{:.1f}"), "post_iq": rng(post_iq, "{:.1f}"), "pre_snr": rng(pre_snr, "{:.1f}"),
                  "post_snr": rng(post_snr, "{:.1f}"),
                  "pre_pl": rng(vals("pre", "resonator_spectroscopy_vs_power", "plateau_contrast"), "{:.2f}"),
                  "post_pl": rng(vals("post", "resonator_spectroscopy_vs_power", "plateau_contrast"), "{:.2f}"),
                  "no_line": len(no_line)}
    done_e1 = [k for k in counted if e1[k]["status"] == "completed"]
    len_diff_e1 = [k for k in done_e1 if e1[k].get("x180_len") != e1[k].get("lab_x180_len")]
    len_diff_n13 = [k for k in counted if n13[k].get("x180_len") != n13[k].get("lab_x180_len")]

    status_line = (f"e1 ran {len(ran)} of the {len(keys)} qubits. {len(fault)} of those are arbel cells that finished inside the "
                   f"readout fault and are void; {len(not_run)} arbel qubits never ran"
                   + (f"; {len(running)} still running" if running else "") + ".")
    worse_names = ", ".join(f"{k[0]} {k[1]}" for k, _ in worse)
    both = [k for k in counted if n13[k]["valid_rb"] and e1[k]["valid_rb"]]
    findings = [
        f"<li><b>Like for like, {len(counted)} qubits.</b> The engine completed {s_e['completed']}/{len(counted)} (the agent "
        f"{s_n['completed']}/{len(counted)}), {s_e['rb1']} with a valid RB under 1 % (agent {s_n['rb1']}). On the {len(both)} "
        f"qubits both measured, the median RB error per Clifford is {100 * median([e1[k]['rb'] for k in both]):.3f} % against "
        f"{100 * median([n13[k]['rb'] for k in both]):.3f} %; readout median {pct(s_e['ro_median'])} against "
        f"{pct(s_n['ro_median'])} over all {len(counted)}. QPU {s_e['qpu_total']:.0f} against "
        f"{s_n['qpu_total']:.0f} min, wall median {s_e['wall_median']:.0f} against {s_n['wall_median']:.0f} min, model cost $0 against "
        f"${s_n['cost']:.2f}.</li>",
        f"<li><b>Outside the fault, every stop came from a rule.</b> {len(stops['length'])} "
        f"({' '.join(qtag(r) for r in sorted(stops['length'], key=n13r.order_key))}) need an x180 longer than the benchmark profile "
        f"allows, because power_rabi caps the amplitude at 0.6 where the lab drives them at "
        f"{rng([r['lab_x180_amp'] for r in stops['length'] if r.get('lab_x180_amp')], '{:.3f}')}. "
        f"{len(stops['t1'])} ({' '.join(qtag(r) for r in stops['t1'])}) has a T1 too short for the chirp line search, which the "
        f"agent pushed through to an RB of {rb_text(n13[('gilboa', 'qD2')])} %."
        + (f" {len(stops['other'])} other ({' '.join(qtag(r) for r in stops['other'])})." if stops["other"] else "") + "</li>",
        f"<li><b>arbel's readout failed at about 15:06</b>, a hardware fault. Node runs before and after 15:05: IQ_blobs "
        f"{fault_nums['pre_iq']} % against {fault_nums['post_iq']} %, resonator_identification contrast SNR {fault_nums['pre_snr']} "
        f"against {fault_nums['post_snr']}; no qubit line on {len(no_line)} qubits where n13 found one. {len(fault_all)} arbel cells "
        f"finished inside it and are void{f' ({len(fault_rerun)} rerun since)' if fault_rerun else ''}; {len(not_run)} arbel qubits "
        f"have not run; "
        + (f"the resume started at {esc(rs)}." if rs else "a resume waits for the readout.") + "</li>",
        f"<li><b>Worse than the agent on {len(worse)}</b> ({worse_names}): RB more than 30 % higher or readout more than 2 points "
        f"lower. Better by the same margin on RB for {len(better)}.</li>",
    ]

    if len_diff_e1:
        len_text = "The engine's x180 length differs from the lab's on " + ", ".join(f"{k[0]} {k[1]}" for k in len_diff_e1) + "."
    else:
        len_text = (f"On all {len(done_e1)} qubits it completed, the engine's x180 length is the lab's: the profile holds lengths "
                    "read-only.")
    len_text += (f" On the same {len(counted)} qubits the n13 agent changed the length on {len(len_diff_n13)}"
                 + (" (" + ", ".join(f"{k[0]} {k[1]} {n13[k]['lab_x180_len']}→{n13[k]['x180_len']} ns" for k in len_diff_n13) + ")"
                    if len_diff_n13 else "") + ".")
    n13_specs = sorted({g(load_json(p), "scramble", "spec_hash") for p in QAB.glob(f"n13-*-{n13r.NIGHTS['n13']}/*-tinycal-*/result.json")}
                       - {None})

    page = f"""<title>e1 Decision Engine</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>{CSS}{N13_CSS}{E1_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · e1 decision engine · 30 Sep 2026, {hm(t0)}–{hm(t1)} CEST · generated {esc(now)}</div>
<h1 style="margin-top:8px">e1: a rule-based decision engine against the agent, on n13's 37 qubits</h1>
<p class="lede">e1 is the first hardware run of qua-decision-engine. It has no language model: it walks the bring-up graph and
decides after every node from that node's structured <span class="mono">next_action</span>. It ran on the same 37 qubits, with the same
scramble and grading, as n13: last night's tinycal run with qwen3.8-27b. The libraries are newer: qua-libs with the nodes' next_action
and the ported line search, and a tinycal whose benchmark profile holds pulse lengths read-only. {esc(status_line)}</p>

<ul class="findings">{''.join(findings)}</ul>

<h2 id="summary">n13 against e1</h2>
{legend()}
<p class="small">{esc(status_line)} Not run: {qlist([e1[k] for k in not_run])}.{(' Running: ' + qlist([e1[k] for k in running]) + '.') if running else ''}</p>
{summary_table([(f"Like for like: {len(counted)} qubits", "arbel fault cells excluded", counted),
                (f"All {len(counted_all)} qubits e1 ran", "arbel fault cells included", counted_all)], n13, e1)}
<p class="small muted">Both columns of a group cover the same qubits. Valid RB is an RB fit that saw at least one decay length
(and, for e1, an RB run that succeeded). Node runs are every run_node call, as in the n13 report, so they include infrastructure
failures: the engine retried {s_e['infra']} in the like-for-like cells and counts them apart from its 27-run budget. Wall and QPU
come from each cell's own clock; the engine spends no model time, which is most of the wall-clock difference. Cost is tinycal's
estimate from the result documents.</p>

<h2 id="engine">What the engine is</h2>
{engine_section(pol_a, pol_b, sha_a, sha_b, e1, counted)}

<h2 id="qubits">Per qubit, n13 → e1</h2>
<p class="small muted">Each numeric cell reads n13 → <b>e1</b>. RB in <b class="better">green</b> or <b class="worse">amber</b>
moved by more than 30 % either way; "n/a" is an RB fit under one decay length or a failed RB run. f<sub>01</sub> is the final state
against the lab's value in that run's own source state. Judge is scrambled parameters back in range; hover for the ones outside.
The engine column is the build that ran the e1 cell (hover for the scheduler). Void and not-run rows are dimmed.</p>
{per_qubit_table(n13, e1)}

<h2 id="stops">Where the engine stopped</h2>
{stops_section(n13, e1, timeline, probe_rows, fault_nums)}

<h2 id="incidents">Incidents</h2>
{incidents_section(vw, inf, e1)}

<h2 id="worse">What got worse than the agent</h2>
<p class="small muted">Computed on the {len(counted)} like-for-like qubits: RB error per Clifford more than 30 % higher (both valid),
or readout assignment more than 2 points lower. The stopped qubits are in the previous section. {len_text}</p>
{worse_html}
<ul class="tight small">
<li><b>Better on RB by the same margin:</b> {' '.join(f"{qtag(e1[k])} {rb_text(n13[k])} → {rb_text(e1[k])} %" for k in sorted(better, key=lambda k: n13r.order_key(e1[k])))}.</li>
</ul>

<h2 id="decisions">Decisions needed, and follow-ups</h2>
<div class="decide">{''.join(f'<div><h4>{esc(t)}</h4><p class="small">{x}</p></div>' for t, x in DECISIONS)}</div>
<h3>Follow-ups from the run (not done)</h3>
<ul class="tight small">{''.join(f'<li>{x}</li>' for x in FOLLOWUPS)}</ul>

<h2 id="setup">Setup and reproducibility</h2>
{setup_table(e1, pol_b, counted_all, n13_specs)}
</div>
"""
    OUT.write_text(page)
    print(f"wrote {OUT}")
    for name, s in (("n13 like-for-like", s_n), ("e1 like-for-like", s_e),
                    ("n13 all-ran", agg([n13[k] for k in counted_all])), ("e1 all-ran", agg([e1[k] for k in counted_all]))):
        print(f"{name}: n {s['n']}, completed {s['completed']}, rb<1% {s['rb1']}, valid RB {s['valid_rb']} median "
              f"{100 * (s['rb_median'] or 0):.3f} %, readout {100 * (s['ro_median'] or 0):.1f} % ({s['ro_95']} >= 95 %), judge "
              f"{s['ballpark']}/{s['graded']}, runs {s['node_runs']}/{s['node_runs_median']}, failed {s['failed']}, QPU "
              f"{s['qpu_total']:.0f}/{s['qpu_median']:.1f} min, wall {s['wall_median']:.0f} min, cost ${s['cost']:.2f}, infra {s['infra']}")
    print(f"ran {len(ran)}, fault {len(fault)} ({', '.join(k[1] for k in fault)}), not run {len(not_run)} "
          f"({', '.join(k[1] for k in not_run)}), running {len(running)}")
    print("stops:", {c: [r['backend'] + ' ' + r['q'] for r in v] for c, v in stops.items()})
    print("worse:", worse, "better:", better)
    mism = [f"{r['backend']} {r['q']}: sched {r['engine']} driver {r['driver_engine']}" for r in e1.values()
            if r.get("driver_engine") and r.get("engine") and r["driver_engine"] != r["engine"]]
    print("engine attribution mismatches:", mism or "none")


if __name__ == "__main__":
    build()
