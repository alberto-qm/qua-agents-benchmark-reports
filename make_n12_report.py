#!/usr/bin/env python3
"""Build the n12 report: the 29-30 Sep 2026 rerun of the n10+n11 full single-qubit bring-ups (tinycal with qwen3.8-27b via
OpenRouter) on the same 37 qubits of qolab, arbel and gilboa, with qua-libs feat/qualibrate-ai 4bf79df.

    python3 make_n12_report.py            # N13_REPORT_URL=<link> turns the n13 pointers into links

Numbers come from each cell's result.json (~/qab-runs/n1{0,1,2}-<backend>-*/<cell>/), its final quam_state, the work dir's
source-state (the lab's reference) and tinycal's events.jsonl (~/code/QM/tinycal/runs/n1x_*). The comparison column is n10+n11
(28-29 Sep, same model, same host, same judge). The per-failure evidence chains and the follow-ups are the operator's reading
of ~/qab-runs/n12-LOG.md and the transcripts, checked against the event logs; the n13 outcomes are from ~/qab-runs/n13-LOG.md.

The output is page content for a claude.ai artifact: a <title>, a <style> block and the page, no <html>/<head>/<body>.
CSS tokens and helpers are shared with make_fwcmp2_report.py.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_night4_report as night4  # noqa: E402
from make_fwcmp2_report import (  # noqa: E402
    CSS, RUNS, TINYCAL_RUNS, context_series, esc, g, gate_fidelity, ktok, load_prices, median, minutes, mtok, token_cost, x180,
)
from make_fwcmp2_report import pct as fpct  # noqa: E402  (fidelity in %, two decimals)

OUT = Path(__file__).with_name("2026-09-29-n12-full-bringup-qwen-tinycal.html")
N13_REPORT_URL = os.environ.get("N13_REPORT_URL", "")  # the published n13 report, when there is one
N13_REPORT_FILE = "2026-09-30-n13-fixes-validation-qwen-tinycal.html"  # make_n13_report.py's output, next to this one
BACKENDS = ("qolab", "arbel", "gilboa")
DEV_MARK = {"qolab": "q", "arbel": "a", "gilboa": "g"}
NIGHTS = {"old": ("n10", "n11"), "new": ("n12",)}
# What a cell's directory name holds for it to be collected: tinycal's agent cells, and (make_e2_summary.py) the
# decision engine's "<backend>-engine-<qubit>".
CELL_MARKS = ("-tinycal-",)
NIGHT_LABEL = {"old": "n10+n11", "new": "n12"}
NIGHT_DATES = {"old": "28–29 Sep", "new": "29–30 Sep"}
TRANSIENT_ERRORS = ("ConnectionError", "ReadTimeout")  # IQCC 503/500 answers and read timeouts: no result came back
REL_TOL = 1e-4  # a write "as proposed": the agent retypes and rounds (7662258540.95 -> 7662260000)
RB_VALID_DECAY_LENGTHS = 1.0
# A discriminated RB survival that falls by less than this from P(0|0) to its floor started near the floor: the qubit was
# not prepared in |0> or the readout could not tell, and the fitted error is a drift (qua-libs MIN_DISCRIMINATED_AMPLITUDE).
RB_MIN_AMPLITUDE = 0.1
MODEL_ID = "qwen/qwen3.8-27b"  # prices.yaml key: the judge's price for the model
SERIES = {"old": "var(--s-old)", "new": "var(--s-new)"}  # one colour per night in the plots and the table header

# The recipe order node_decisions() judges "off the recipe order" against (make_night4_report's rule), one per night.
# n10+n11: tinycal 729535c (graph 19 with T1_coarse, as make_n10_report.py). n12: tinycal 5146a6c + the step-2 edit (graph 18,
# T1_coarse gone). The conditional T1 after power_rabi (only when T1_chirp stored none) is left out, or every normal run would
# count as skipping it. The power sweep appears twice (step 4 repeats it at the sweet spot); T1/T2echo share a step.
_HEAD = [("resonator_identification",), ("resonator_spectroscopy_vs_power",), ("resonator_spectroscopy_vs_flux",),
         ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy",), ("T1_chirp",), ("qubit_spectroscopy_vs_flux",),
         ("qubit_spectroscopy_fine",), ("power_rabi",)]
_TAIL = [("readout_power_optimization",), ("readout_frequency_optimization",), ("IQ_blobs",), ("ramsey_vs_flux_calibration",),
         ("power_rabi_error_amplification_x180",), ("ramsey",), ("T1", "T2echo"), ("DRAG_calibration",), ("Randomized_benchmarking",)]
RECIPE = {"old": _HEAD + [("T1_coarse",)] + _TAIL, "new": _HEAD + _TAIL}


def decisions(night: str, run_id: str, q: str):
    """make_night4_report.node_decisions against this night's recipe (it reads the module's RECIPE globals)."""
    night4.RECIPE[:] = RECIPE[night]
    night4.RECIPE_NODES[:] = list(dict.fromkeys(n for step in RECIPE[night] for n in step))
    return night4.node_decisions(run_id, q)

# ----------------------------------------------------------------------------- operator verdicts
# completed, but the results are not meaningful: (night, backend, qubit) -> why
NOT_MEANINGFUL = {
    ("new", "gilboa", "qC3"): "completed, not meaningful: T1 written as 30 ms instead of 30 µs, so every later fit ran on 4–20 shots "
                              "(RB 1.02 %, readout 71 %, T2echo negative)",
    ("new", "qolab", "Q6"): "completed, not meaningful: parked at the first flux map's edge, f_01 118.6 MHz below the lab's, RB 16 %",
    ("old", "arbel", "qD4"): "completed, not meaningful: calibrated on the two-photon line, 99 MHz below f_01; RB error −3·10⁻⁷",
    ("old", "arbel", "qD5"): "completed, not meaningful: calibrated on the two-photon line, 102.5 MHz below f_01; RB error −1.9 %",
}
# did not complete: (night, backend, qubit) -> short cause
NOT_COMPLETED = {
    ("old", "arbel", "qD1"): "escalated: no 0→1 line found",
    ("old", "arbel", "qD2"): "escalated: committed a resonator 50 MHz above its own, qubit never found",
    ("old", "arbel", "qD3"): "escalated: qubit at the drive's +400 MHz IF edge, every window refused",
    ("old", "gilboa", "qD4"): "escalated: readout 52 % at 0.05 V under the 0.1 V ceiling; f_01 from a rejected fit",
    ("new", "arbel", "qB4"): "escalated at the turn cap: 02b's low-power line committed as the readout frequency",
    ("new", "arbel", "qD1"): "failed: 150-turn budget spent without finding the 0→1 line",
}

# ----------------------------------------------------------------------------- collect
_EVENTS: dict[tuple[str, str], list] = {}


def events(run_id: str, q: str) -> list:
    key = (run_id, q)
    if key not in _EVENTS:
        p = TINYCAL_RUNS / run_id / q / "events.jsonl"
        out = []
        if run_id and p.exists():
            for line in p.read_text(errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        _EVENTS[key] = out
    return _EVENTS[key]


def _num(e, key):
    v = (e.get("numerics") or {}).get(key)
    return v.get("value") if isinstance(v, dict) else v


def _close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return abs(a - b) <= REL_TOL * abs(b) or abs(a - b) <= 1e-15


def straight_through(ev: list) -> dict:
    """What the agent contributed beyond running the graph and writing each proposal, from one cell's events.

    Transient IQCC errors are ignored (no run, no failure). A run that is not status 'completed' with outcome 'successful'
    (a failed fit, a pre-flight refusal, rejected parameters, a 60 s job timeout, a config error) is a failed run. Runs are
    counted per node; the recipe's second resonator_spectroscopy_vs_power at the sweet spot (step 4, after the flux map) is
    part of the path, not a rerun. Proposals are tracked per path: a write must match the latest proposal for its path
    (numbers within REL_TOL, anything else exactly), a proposal must be written before the next one for the same path
    supersedes it, and at the end nothing may be left unwritten unless the proposal equalled the current value."""
    failed, transient, overrides = 0, 0, 0
    counts: collections.Counter = collections.Counter()
    order: list[str] = []
    latest: dict[str, object] = {}
    pending: dict[str, tuple] = {}
    changed = unproposed = unwritten = 0
    unproposed_paths: list[str] = []
    for e in ev:
        if e.get("kind") != "tool_call":
            continue
        if e.get("tool") == "run_node":
            if str(e.get("error") or "").startswith(TRANSIENT_ERRORS):
                transient += 1
                continue
            if e.get("status") == "rejected" and not (e.get("node") or (e.get("arguments") or {}).get("node")):
                continue  # a malformed call with no node name runs nothing
            node = e.get("node") or ""
            counts[node] += 1
            order.append(node)
            if e.get("status") != "completed" or e.get("outcome") != "successful":
                failed += 1
            if e.get("parameters"):
                overrides += 1
            for p in e.get("proposed_updates") or []:
                path = p.get("path")
                if path in pending and not _close(*pending[path]):
                    unwritten += 1  # superseded before it was written
                pending[path] = (p.get("proposed"), p.get("current"))
                latest[path] = p.get("proposed")
        elif e.get("tool") == "write_state":
            items = e.get("results") or [{"path": u.get("path"), "value": u.get("value")}
                                         for u in (e.get("arguments") or {}).get("updates") or []]
            for it in items:
                path, val = it.get("path"), it.get("value")
                if path not in latest:
                    unproposed += 1
                    unproposed_paths.append(str(path))
                elif not _close(val, latest[path]):
                    changed += 1
                pending.pop(path, None)
    unwritten += sum(1 for pv, cv in pending.values() if not _close(pv, cv))
    allowed = lambda n: 2 if n == "resonator_spectroscopy_vs_power" else 1  # noqa: E731
    extra = sum(max(0, c - allowed(n)) for n, c in counts.items())
    idx = [i for i, n in enumerate(order) if n == "resonator_spectroscopy_vs_power"]
    if len(idx) == 2 and "resonator_spectroscopy_vs_flux" not in order[idx[0]:idx[1]]:
        extra += 1  # a second power sweep that is not the sweet-spot repeat
    strict_extra = sum(c - 1 for c in counts.values() if c > 1)
    return {"failed": failed, "extra": extra, "strict_extra": strict_extra, "changed": changed, "unproposed": unproposed,
            "unwritten": unwritten, "overrides": overrides, "transient": transient, "unproposed_paths": unproposed_paths}


PRICES = load_prices()


def collect() -> list[dict]:
    rows = []
    for night, prefixes in NIGHTS.items():
        for prefix in prefixes:
            for work in sorted(RUNS.glob(f"{prefix}-*-2026*")):
                backend = work.name.split("-")[1]
                if not work.is_dir() or backend not in BACKENDS:
                    continue
                try:
                    lab_all = json.load(open(work / "source-state/state.json"))["qubits"]
                except Exception:  # noqa: BLE001
                    lab_all = {}
                for cell in sorted(p for p in work.iterdir() if p.is_dir() and any(m in p.name for m in CELL_MARKS)):
                    q = cell.name.rsplit("-", 1)[1]
                    doc = cell / "result.json"
                    if not doc.exists():
                        continue
                    r = json.load(open(doc))
                    x = r["targets"][0]
                    run_id = r.get("run_id") or ""
                    ev = events(run_id, q)
                    runs = [e for e in ev if e.get("tool") == "run_node" and e.get("kind") == "tool_call"]
                    rb_runs = [e for e in runs if e.get("node") == "Randomized_benchmarking"]
                    try:
                        final = json.load(open(cell / "quam_state/state.json"))["qubits"][q]
                    except Exception:  # noqa: BLE001
                        final = {}
                    lab = lab_all.get(q) or {}
                    L, A = x180(cell / "quam_state/state.json", q)
                    qual = x.get("quality") or {}
                    t = g(x, "agent", "time") or {}
                    ro_amp = g(final, "resonator", "operations", "readout", "amplitude")
                    ro_amp_lab = g(lab, "resonator", "operations", "readout", "amplitude")
                    gf = g(lab, "gate_fidelity") or {}
                    ag = x.get("agent") or {}
                    cs = context_series(cell, run_id, q, "tinycal")
                    rows.append({
                        "tokens": ag.get("tokens") or {}, "judge_cost": token_cost(ag.get("tokens") or {}, MODEL_ID, PRICES),
                        "nodes_exec": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                        "ctx": {"median": median(cs), "end": cs[-1] if cs else None},
                        "overrides": night4.overrides(run_id, q), "decisions": decisions(night, run_id, q),
                        # the lab's value in the pull this cell started from; 0 = never calibrated, not a reference
                        "ref": gf.get("averaged") or None, "ref_date": gf.get("averaged_updated_at") or "",
                        "night": night, "run": prefix, "backend": backend, "q": q, "work": work.name, "cell": cell.name,
                        "run_id": run_id, "status": x.get("status"),
                        "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
                        "rb": g(qual, "rb", "error_per_clifford"),
                        "rb_cover": _num(rb_runs[-1], "decay_lengths_covered") if rb_runs else None,
                        "rb_depth": ((rb_runs[-1].get("parameters") or {}).get("max_circuit_depth") or 2048) if rb_runs else None,
                        "rb_amp": _num(rb_runs[-1], "fit_amplitude") if rb_runs and (rb_runs[-1].get("parameters") or {}).get(
                            "use_state_discrimination", True) else None,
                        "readout": g(qual, "readout", "assignment_fidelity"),
                        "t1": g(qual, "coherence_limit", "t1_s"), "t2e": g(qual, "coherence_limit", "t2echo_s"),
                        "x180_amp": A, "x180_len": L,
                        "df01": ((final.get("f_01") or 0) - (lab.get("f_01") or 0)) / 1e6 if final.get("f_01") and lab.get("f_01") else None,
                        "ro_ratio": ro_amp / ro_amp_lab if isinstance(ro_amp, (int, float)) and isinstance(ro_amp_lab, (int, float)) and ro_amp_lab else None,
                        "qpu_s": t.get("qpu_execution_s"), "wall_s": t.get("total_s") or g(r, "totals", "time", "total_s"),
                        "model_s": t.get("model_s"), "queue_s": t.get("queue_wait_s"),
                        "turns": g(x, "agent", "turns", "total") or g(r, "totals", "turns", "total"),
                        "cost": g(r, "totals", "tokens", "estimated_cost_usd"),
                        "node_runs": len(runs), "not_ok": sum(1 for e in runs if e.get("outcome") != "successful"),
                        "o08b": [float(e["qpu_execution_s"]) for e in runs if e.get("node") == "readout_power_optimization"
                                 and e.get("outcome") == "successful" and e.get("qpu_execution_s")],
                        "ballpark": g(x, "judge", "identity", "in_ballpark"), "graded": g(x, "judge", "identity", "graded"),
                        "outside": g(x, "judge", "identity", "outside") or [],
                        "started": r.get("started_at"), "ended": r.get("ended_at"),
                        "st": straight_through(ev),
                        "quad_unproposed": False,
                        "flux_port_refused": any("the flux sweep reaches" in str(e.get("error") or "") for e in runs),
                    })
    for r in rows:
        r["gate_fid"] = gate_fidelity(r["rb"]) if valid_rb(r) else None
        r["quad_unproposed"] = any(p.endswith("freq_vs_flux_01_quad_term") for p in r["st"]["unproposed_paths"])
    return rows


# ----------------------------------------------------------------------------- per-row verdicts and formatting
def outcome(r) -> tuple[str, str]:
    key = (r["night"], r["backend"], r["q"])
    if r["status"] != "completed":
        return "bad", NOT_COMPLETED.get(key, r["status"])
    if key in NOT_MEANINGFUL:
        return "warn", NOT_MEANINGFUL[key]
    return "ok", "completed"


def is_straight(r, strict=False) -> bool:
    s = r["st"]
    return (r["status"] == "completed" and not s["failed"] and not (s["strict_extra"] if strict else s["extra"])
            and not s["changed"] and not s["unproposed"] and not s["unwritten"])


def valid_rb(r) -> bool:
    return (r["status"] == "completed" and (r["rb"] or 0) > 0
            and (r["rb_cover"] is None or r["rb_cover"] >= RB_VALID_DECAY_LENGTHS)
            and (r.get("rb_amp") is None or r["rb_amp"] >= RB_MIN_AMPLITUDE))


def st_reasons(r) -> str:
    s = r["st"]
    parts = []
    if r["status"] != "completed":
        parts.append("did not complete")
    for k, lab in (("failed", "failed or refused runs"), ("extra", "reruns"), ("changed", "values changed from the proposal"),
                   ("unproposed", "writes to paths no node proposed"), ("unwritten", "proposals left unwritten"),
                   ("overrides", "runs with model-set parameters"), ("transient", "IQCC transients ignored")):
        if s[k]:
            parts.append(f"{s[k]} {lab}")
    return " · ".join(parts) or "straight through"


ORDER = {b: i for i, b in enumerate(BACKENDS)}


def qkey(r):
    return (ORDER[r["backend"]], r["q"])


def qtag(r) -> str:
    cls, why = outcome(r)
    title = f"{r['backend']} {r['q']} ({r['run']}): {why}"
    return (f"<span class='q {cls}' title='{esc(title)}'><span class='dv' aria-hidden='true'>{DEV_MARK[r['backend']]}</span>"
            f"{esc(r['q'])}</span>")


def qubit_list(rows) -> str:
    if not rows:
        return "<span class='muted'>none</span>"
    return "<span class='ql'>" + "".join(qtag(r) for r in sorted(rows, key=qkey)) + "</span>"


def status_word(r) -> str:
    cls, why = outcome(r)
    word = {"ok": r["status"], "warn": "completed, not meaningful", "bad": r["status"]}[cls]
    return f"<span class='qs {cls}' title='{esc(why)}'>{esc(word)}</span>"


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}"


def mins(s, d=0):
    return "—" if s is None else f"{s / 60:.{d}f}"


def us(v):
    if v is None:
        return "—"
    v = v * 1e6
    if abs(v) < 10_000:
        return f"{v:.0f}"
    return f"{v / 1000:.0f} ms" if abs(v) < 1_000_000 else f"{v / 1e6:.1f} s"


def rb_cell(r) -> str:
    rb = r["rb"]
    if rb is None:
        return "—"
    txt = f"{100 * rb:.3f}"
    if rb <= 0:
        return f"<span class='muted' title='non-positive error per Clifford: not a probability'>{txt}</span>"
    if r["rb_cover"] is not None and r["rb_cover"] < RB_VALID_DECAY_LENGTHS:
        return (f"<span title='the fit saw {r['rb_cover']:.2f} decay lengths (max depth {r['rb_depth']}): not a measurement'>"
                f"{txt}<sup>×</sup></span>")
    if r.get("rb_amp") is not None and r["rb_amp"] < RB_MIN_AMPLITUDE:
        return (f"<span title='the survival fell by only {r['rb_amp']:.3f} from P(0|0) to its floor: the qubit was not prepared "
                f"in |0> or the readout could not tell, so this is a drift, not a gate error'>{txt}<sup>×</sup></span>")
    return txt


def legend(extra: str = "") -> str:
    return ("<div class='legend small'>"
            "<span><span class='q ok'>qA1</span> completed with meaningful results</span>"
            "<span><span class='q warn'>Q6</span> completed, results not meaningful (reason on hover)</span>"
            "<span><span class='q bad'>qB4</span> not completed: escalated or failed</span>"
            "<span class='muted'>device prefix: <span class='mono'><span class='dvl'>q</span>qolab · <span class='dvl'>a</span>arbel · "
            "<span class='dvl'>g</span>gilboa</span> (hover a name for the full backend)</span>" + extra + "</div>")


# ----------------------------------------------------------------------------- stats
def night_stats(rows) -> dict:
    done = [r for r in rows if r["status"] == "completed"]
    meaningful = [r for r in done if outcome(r)[0] == "ok"]
    valid = [r for r in done if valid_rb(r)]
    ro = [r["readout"] for r in rows if r["readout"] is not None]
    st = [r for r in rows if is_straight(r)]
    starts = [datetime.fromisoformat(r["started"]) for r in rows if r["started"]]
    ends = [datetime.fromisoformat(r["ended"]) for r in rows if r["ended"]]
    o08b = [v for r in rows for v in r["o08b"]]
    return {
        "n": len(rows), "done": done, "meaningful": meaningful, "valid": valid, "st": st,
        "st_strict": [r for r in rows if is_straight(r, strict=True)],
        "st_noov": [r for r in st if r["st"]["overrides"] == 0],
        "no_reruns": [r for r in rows if r["status"] == "completed" and not r["st"]["failed"] and not r["st"]["extra"]],
        "accepted": [r for r in rows if r["status"] == "completed" and not r["st"]["changed"] and not r["st"]["unproposed"]
                     and not r["st"]["unwritten"]],
        "quad_unproposed": sum(1 for r in rows if r["quad_unproposed"]),
        "rb_med": median([r["rb"] for r in valid]),
        "ro_med": median(ro), "ro_n": len(ro), "ro_95": sum(1 for v in ro if v >= 0.95),
        "judge": (sum(r["ballpark"] or 0 for r in rows), sum(r["graded"] or 0 for r in rows)),
        "judge_ro_amp": sum(1 for r in rows for o in r["outside"] if "readout.amplitude" in o),
        "qpu_tot": sum(r["qpu_s"] or 0 for r in rows), "qpu_med": median([r["qpu_s"] for r in rows if r["qpu_s"] is not None]),
        "wall_med": median([r["wall_s"] for r in rows if r["wall_s"] is not None]),
        "model_med": median([r["model_s"] for r in rows if r["model_s"] is not None]),
        "queue_med": median([r["queue_s"] for r in rows if r["queue_s"] is not None]),
        "turns_med": median([r["turns"] for r in rows if r["turns"] is not None]),
        "runs": sum(r["node_runs"] for r in rows), "not_ok": sum(r["not_ok"] for r in rows),
        "runs_med": median([r["node_runs"] for r in rows]),
        "transient": sum(r["st"]["transient"] for r in rows),
        "cost": sum(r["cost"] or 0 for r in rows),
        "span": (min(starts), max(ends)) if starts else None,
        "o08b_med": median(o08b), "o08b_n": len(o08b),
        "ro_ratio_med": median([r["ro_ratio"] for r in done if r["ro_ratio"]]),
        "ro_ratio_hi": sum(1 for r in done if r["ro_ratio"] and r["ro_ratio"] > 1.6),
        "flux_refused": sum(1 for r in rows if r["flux_port_refused"]),
    }


def span_text(s) -> str:
    a, b = s["span"]
    h, m = divmod(round((b - a).total_seconds() / 60), 60)
    return f"{a:%d %b %H:%M} → {b:%H:%M} ({h} h {m:02d} min)"


# ----------------------------------------------------------------------------- summary table (make_night4_report's "By host" rows)
THRESHOLD = 0.999
STRIP_LO, STRIP_HI = 0.02, 10.0  # gate error in %; the old report's strip stops at 5 %, qolab Q6 sits at 8.5 %


def fidelity_strip(rows, night) -> str:
    """make_night4_report.fidelity_strip with each run's own reference (the lab's value in the cell's source-state): one dot per
    completed run with a valid RB on a log gate-error axis, the 99.9 % line; filled = at or above the reference, hollow = below."""
    import math
    pts = sorted((r for r in rows if r["gate_fid"] is not None), key=lambda r: r["gate_fid"])
    if not pts:
        return ""
    W, H, L, R = 220, 40, 6, 8
    lo, hi = math.log10(STRIP_LO), math.log10(STRIP_HI)
    x = lambda e: L + (W - L - R) * (min(max(math.log10(max(e, STRIP_LO)), lo), hi) - lo) / (hi - lo)  # noqa: E731
    xt = x(100 * (1 - THRESHOLD))
    out = [f'<svg class="strip" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="gate error per completed run">']
    for tick in (0.03, 0.1, 0.3, 1, 3, 10):
        out.append(f'<text x="{x(tick):.1f}" y="{H - 2}" class="tick" text-anchor="middle">{tick:g}%</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{H - 14}" y2="{H - 14}" class="axis"/>')
    out.append(f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="2" y2="{H - 12}" class="thr"/>')
    above = beat = with_ref = 0
    for i, r in enumerate(pts):
        f, ref = r["gate_fid"], r["ref"]
        hollow = ref is not None and f < ref
        above += f >= THRESHOLD
        if ref is not None:
            with_ref += 1
            beat += not hollow
        label = f"{r['backend']} {r['q']} ({r['run']}) · {fpct(f)}" + (f" (reference {fpct(ref)})" if ref is not None else " (no reference)")
        out.append(f'<circle cx="{x(100 * (1 - f)):.1f}" cy="{H - 22 - (i % 3) * 5}" r="3.2" class="dot {night}{" hollow" if hollow else ""}">'
                   f'<title>{esc(label)}</title></circle>')
    out.append("</svg>")
    note = (f"≥ {100 * THRESHOLD:g} %: <b>{above} / {len(pts)}</b>"
            + (f"<br><span class='muted'>at or above the reference: {beat} / {with_ref}</span>" if with_ref else ""))
    return f"<div class='stripwrap'>{''.join(out)}<div class='small'>{note}</div></div>"


def night_stats(rows) -> dict:
    done = [r for r in rows if r["status"] == "completed"]
    rest = [r for r in rows if r["status"] != "completed"]
    meaningful = [r for r in done if outcome(r)[0] == "ok"]
    valid = [r for r in done if valid_rb(r)]
    ro = [r["readout"] for r in rows if r["readout"] is not None]
    st = [r for r in rows if is_straight(r)]
    starts = [datetime.fromisoformat(r["started"]) for r in rows if r["started"]]
    ends = [datetime.fromisoformat(r["ended"]) for r in rows if r["ended"]]
    o08b = [v for r in rows for v in r["o08b"]]
    n = len(done) or 1
    tot = lambda k: sum((r[k] or 0) for r in rows)  # noqa: E731
    tokt = lambda k: sum((r["tokens"].get(k) or 0) for r in rows)  # noqa: E731

    def override_share(rs):  # make_night4_report.pivot_table: (overrides, state writes)
        w = sum(r["overrides"][0] for r in rs if r["overrides"])
        return (sum(r["overrides"][1] for r in rs if r["overrides"]), w) if w else None

    def off_order(rs):  # (stepped back + skipped ahead, node runs judged against the recipe)
        c = collections.Counter()
        for r in rs:
            for d in (r["decisions"] or {}).values():
                c.update(d)
        return (c["back"] + c["ahead"], c["runs"]) if c["runs"] else None
    fids = sorted(r["gate_fid"] for r in done if r["gate_fid"] is not None)
    return {
        "n": len(rows), "done": done, "rest": rest, "meaningful": meaningful, "valid": valid, "st": st,
        "st_strict": [r for r in rows if is_straight(r, strict=True)],
        "st_noov": [r for r in st if r["st"]["overrides"] == 0],
        "no_reruns": [r for r in rows if r["status"] == "completed" and not r["st"]["failed"] and not r["st"]["extra"]],
        "accepted": [r for r in rows if r["status"] == "completed" and not r["st"]["changed"] and not r["st"]["unproposed"]
                     and not r["st"]["unwritten"]],
        "quad_unproposed": sum(1 for r in rows if r["quad_unproposed"]),
        "rb_med": median([r["rb"] for r in valid]), "fids": fids,
        "ro_med": median(ro), "ro_n": len(ro), "ro_95": sum(1 for v in ro if v >= 0.95),
        "judge": (sum(r["ballpark"] or 0 for r in rows), sum(r["graded"] or 0 for r in rows)),
        "judge_ro_amp": sum(1 for r in rows for o in r["outside"] if "readout.amplitude" in o),
        "qpu_tot": tot("qpu_s"), "qpu_med": median([r["qpu_s"] for r in rows if r["qpu_s"] is not None]),
        "wall_med": median([r["wall_s"] for r in rows if r["wall_s"] is not None]),
        "model_med": median([r["model_s"] for r in rows if r["model_s"] is not None]),
        "queue_med": median([r["queue_s"] for r in rows if r["queue_s"] is not None]),
        "turns_med": median([r["turns"] for r in rows if r["turns"] is not None]),
        "runs": sum(r["node_runs"] for r in rows), "not_ok": sum(r["not_ok"] for r in rows),
        "runs_med": median([r["node_runs"] for r in rows]),
        "transient": sum(r["st"]["transient"] for r in rows),
        "cost": sum(r["cost"] or 0 for r in rows), "judge_cost": tot("judge_cost"),
        "span": (min(starts), max(ends)) if starts else None,
        "o08b_med": median(o08b), "o08b_n": len(o08b),
        "ro_ratio_med": median([r["ro_ratio"] for r in done if r["ro_ratio"]]),
        "ro_ratio_hi": sum(1 for r in done if r["ro_ratio"] and r["ro_ratio"] > 1.6),
        "flux_refused": sum(1 for r in rows if r["flux_port_refused"]),
        # per calibration = the total over every qubit-run of the night divided by the number completed (make_night4_report)
        "per_cal": {"model_s": tot("model_s") / n, "qpu_s": tot("qpu_s") / n, "queue_s": tot("queue_s") / n,
                    "cost": tot("judge_cost") / n, "cost_meaningful": tot("judge_cost") / (len(meaningful) or 1),
                    "turns": tot("turns") / n, "nodes": tot("nodes_exec") / n, "reruns": tot("reruns") / n,
                    "tin": tokt("input") / n, "tout": tokt("output") / n, "tcr": tokt("cache_read") / n,
                    "tcw": tokt("cache_creation") / n if any(r["tokens"].get("cache_creation") is not None for r in rows) else None},
        "ov_done": override_share(done), "ov_rest": override_share(rest),
        "oo_done": off_order(done), "oo_rest": off_order(rest),
        "ctx_med": (lambda v: sum(v) / len(v) if v else None)([r["ctx"]["median"] for r in rows if r["ctx"]["median"]]),
        "ctx_end": (lambda v: sum(v) / len(v) if v else None)([r["ctx"]["end"] for r in done if r["ctx"]["end"]]),
    }


def span_text(s) -> str:
    a, b = s["span"]
    h, m = divmod(round((b - a).total_seconds() / 60), 60)
    return f"{a:%d %b %H:%M} → {b:%H:%M} ({h} h {m:02d} min)"


def summary_table(S) -> str:
    def share(done, rest, n, what):
        if done is None:
            return "—"
        out = f"<span class='num'>{100 * done[0] / done[1]:.0f}% ({done[0] / n:.1f} / calibration)</span>"
        if rest is not None:
            out += f"<br><span class='small muted'>not completed: {100 * rest[0] / rest[1]:.0f}% of {rest[1]} {what}</span>"
        return out

    def st_cell(s, k):
        return (f"<b class='num'>{len(s['st'])}/{s['n']}</b> {qubit_list(s['st'])}"
                f"<div class='small muted'>with no model-set parameters either: {len(s['st_noov'])}/{s['n']}</div>"
                f"<div class='small muted'>every node once, none failed: {len(s['no_reruns'])}/{s['n']} {qubit_list(s['no_reruns'])}</div>"
                f"<div class='small muted'>every value as proposed: {len(s['accepted'])}/{s['n']} {qubit_list(s['accepted'])}</div>")

    def valid_cell(s, k):
        note = {"old": "excludes qolab Q2, arbel qC1, gilboa qD5 (fit under one decay length) and arbel qD4/qD5 (non-positive)",
                "new": "includes gilboa qC3 and qolab Q6 (not meaningful) and qolab Q4 (1.003 decay lengths at depth 64)"}[k]
        return (f"<span class='num'><b>{len(s['valid'])}</b>, median {100 * s['rb_med']:.3f} %</span>"
                f"<div class='small muted'>{esc(note)}</div>")

    def fid_cell(s, k):
        f = s["fids"]
        if not f:
            return "—"
        return (f"<span class='num'><b>{fpct(median(f))}</b></span><br><span class='small muted num'>{fpct(f[0])} – {fpct(f[-1])}</span>"
                + fidelity_strip(s["done"], k))

    def pc(key, fmt):
        return lambda s, k: f"<span class='num'>{fmt(s['per_cal'][key])}</span>"
    n_of = lambda s: len(s["done"]) or 1  # noqa: E731
    body = [
        ("qubits measured (green: completed with meaningful results; amber: completed, results not meaningful; red: not completed; "
         "the small letter is the device)", lambda s, k: qubit_list(s["rows"])),
        ("calibrations completed / attempted", lambda s, k: f"<b class='num'>{len(s['done'])}/{s['n']} ({100 * len(s['done']) / s['n']:.0f}%)</b>"),
        ("completed with meaningful results", lambda s, k: f"<b class='num'>{len(s['meaningful'])}/{s['n']}</b>"),
        ("straight through: the agent contributed nothing (definition below the table)", st_cell),
        ("agent overrides: share of state writes that are not a node's proposal (no node proposed the path, or the value is more than "
         "0.1 % from the latest proposal), completed calibrations; below, the runs that did not complete",
         lambda s, k: share(s["ov_done"], s["ov_rest"], n_of(s), "writes")),
        ("node runs off the recipe order: stepped back to an earlier step or skipped one (retries and re-runs of the node just run are "
         "not counted), completed calibrations; below, the runs that did not complete",
         lambda s, k: share(s["oo_done"], s["oo_rest"], n_of(s), "node runs")),
        ("single-qubit gate fidelity, median (min – max), completed runs with a valid RB; dots: gate error per run on a log axis, line at "
         "99.9 %, filled = at or above the qubit's reference fidelity, hollow = below it", fid_cell),
        ("valid RB (the fit saw ≥ 1 decay length): runs, median error per Clifford", valid_cell),
        ("readout assignment fidelity, median over the runs that measured one",
         lambda s, k: f"<span class='num'><b>{100 * s['ro_med']:.1f} %</b> · {s['ro_95']} of {s['ro_n']} ≥ 95 %</span>"),
        ("judge: graded parameters back in range", lambda s, k: f"<span class='num'><b>{s['judge'][0]}/{s['judge'][1]}</b></span>"),
        ("agent time / calibration", pc("model_s", minutes)),
        ("QPU time / calibration", pc("qpu_s", lambda v: f"{v / 60:.1f} min")),
        ("QPU, total / median per qubit-run",
         lambda s, k: f"<span class='num'>{s['qpu_tot'] / 60:.0f} / {s['qpu_med'] / 60:.1f} min</span>"),
        ("queue wait / calibration", pc("queue_s", lambda v: f"{v / 60:.1f} min")),
        ("wall time per qubit-run, median", lambda s, k: f"<span class='num'>{s['wall_med'] / 60:.0f} min</span>"),
        ("judge cost / calibration", lambda s, k: f"<span class='num'>${s['per_cal']['cost']:.2f}</span>"
                                                  f"<br><span class='small muted'>${s['per_cal']['cost_meaningful']:.2f} per meaningful calibration</span>"),
        ("judge cost, total spent", lambda s, k: f"<span class='num'><b>${s['judge_cost']:.2f}</b></span>"
                                                 + ("<br><span class='small muted'>the same as tinycal's own estimate</span>"
                                                    if abs(s["judge_cost"] - s["cost"]) < 0.005 else
                                                    f"<br><span class='small muted'>tinycal's estimate ${s['cost']:.2f}</span>")),
        ("model turns / calibration", pc("turns", lambda v: f"{v:.0f}")),
        ("node runs (re-runs) / calibration", lambda s, k: f"<span class='num'>{s['per_cal']['nodes']:.0f} ({s['per_cal']['reruns']:.0f})</span>"),
        ("node runs / not successful, all qubit-runs", lambda s, k: f"<span class='num'>{s['runs']} / {s['not_ok']}</span>"
         f"<br><span class='small muted'>{s['transient']} of them IQCC 500/503 answers or read timeouts</span>"),
        ("context per model call, median, in k tokens (mean over qubit-runs)", lambda s, k: f"<span class='num'>{ktok(s['ctx_med'])}</span>"),
        ("context at the end of a bring-up, k tokens (mean over completed qubit-runs)", lambda s, k: f"<span class='num'>{ktok(s['ctx_end'])}</span>"),
        ("input tokens / calibration", pc("tin", mtok)),
        ("output tokens / calibration", pc("tout", mtok)),
        ("cached tokens read / written / calibration",
         lambda s, k: f"<span class='num'>{mtok(s['per_cal']['tcr'])} / {mtok(s['per_cal']['tcw']) if s['per_cal']['tcw'] is not None else '—'}</span>"
                      + ("" if s["per_cal"]["tcw"] is not None else "<br><span class='small muted'>OpenRouter reports no cache writes</span>")),
    ]
    trs = "".join(f"<tr><th class='rowh'>{esc(lab)}</th>" + "".join(f"<td>{fn(S[k], k)}</td>" for k in ("old", "new")) + "</tr>"
                  for lab, fn in body)
    sub = {"old": "28–29 Sep · 37 qubits · 3 per device in flight", "new": "29–30 Sep · 37 qubits · 3 in flight in all"}
    head = ("<thead><tr><th></th>" + "".join(
        f"<th class='grp'><span class='chip'><i style='background:{SERIES[k]}'></i>{NIGHT_LABEL[k]}</span>"
        f"<br><span class='small muted'>{sub[k]}</span></th>" for k in ("old", "new")) + "</tr></thead>")
    return f"<div class='scroll'><table class='grid pivot sum'>{head}<tbody>{trs}</tbody></table></div>"


# ----------------------------------------------------------------------------- the two scatters (make_night4_report.error_scatters)
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _day(stamp: str) -> str:
    return f"{int(stamp[8:10])} {_MONTHS[int(stamp[5:7]) - 1]}"


def ref_dates_phrase(rows) -> str:
    """'arbel 29 Sep 00:06, gilboa 6–29 Sep': the stamps of the references actually plotted, per backend."""
    out = []
    for b in BACKENDS:
        stamps = sorted({r["ref_date"] for r in rows if r["backend"] == b and r["ref_date"]})
        if not stamps:
            continue
        days = sorted({st[:10] for st in stamps})
        if len(days) == 1:
            out.append(f"{b} {_day(days[0])}" + (f" {stamps[0][11:16]}" if len(stamps) == 1 else ""))
        else:
            first = _day(days[0])[:-4] if days[0][5:7] == days[-1][5:7] else _day(days[0])
            out.append(f"{b} {first}–{_day(days[-1])}")
    return ", ".join(out)


PLOT_HI = 15.0  # gate error in %: gilboa qD2's lab reference is 10.5 %, qolab Q6's n12 error 8.5 %; nothing is clipped


def scatter_svg(points, *, xlabel, ylabel, lo=STRIP_LO, hi=PLOT_HI, guides=(1.0,), guide_labels=("y = x",), W=560, H=400) -> str:
    """make_night4_report._scatter_svg with one series per night: circles for n10+n11, diamonds for n12 (shape as well as colour).
    ``points`` = (x, y, label, night), gate error in %."""
    import math
    L, R, T, B = 50, 14, 12, 42
    llo, lhi = math.log10(lo), math.log10(hi)
    sx = lambda v: L + (W - L - R) * (min(max(math.log10(v), llo), lhi) - llo) / (lhi - llo)  # noqa: E731
    sy = lambda v: H - B - (H - T - B) * (min(max(math.log10(v), llo), lhi) - llo) / (lhi - llo)  # noqa: E731
    out = [f'<svg class="scatter" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(ylabel)} against {esc(xlabel)}">']
    for tick in (0.03, 0.1, 0.3, 1, 3, 10):
        if lo <= tick <= hi:
            out.append(f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                       f'<line y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" x1="{L}" x2="{W - R}" class="grid"/>'
                       f'<text x="{sx(tick):.1f}" y="{H - B + 15}" class="tick" text-anchor="middle">{tick:g}%</text>'
                       f'<text x="{L - 6}" y="{sy(tick) + 3:.1f}" class="tick" text-anchor="end">{tick:g}%</text>')
    for k, lab in zip(guides, guide_labels):
        x0, x1 = lo, hi / k if k > 1 else hi
        xl = x1 / 4  # label a quarter of the way down the line, just below and right of it
        out.append(f'<line x1="{sx(x0):.1f}" y1="{sy(x0 * k):.1f}" x2="{sx(x1):.1f}" y2="{sy(x1 * k):.1f}" class="guide"/>'
                   f'<text x="{sx(xl) + 7:.1f}" y="{sy(xl * k) + 4:.1f}" class="tick">{esc(lab)}</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" class="frame"/>')
    for night in ("old", "new"):
        for x, y, label, nt in points:
            if nt != night:
                continue
            cx, cy = sx(x), sy(y)
            if night == "old":
                mark = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" class="pt {night}">'
            else:
                d = 5.6
                mark = f'<path d="M{cx:.1f} {cy - d:.1f}L{cx + d:.1f} {cy:.1f}L{cx:.1f} {cy + d:.1f}L{cx - d:.1f} {cy:.1f}Z" class="pt {night}">'
            out.append(mark + f"<title>{esc(label)}</title>" + ("</circle>" if night == "old" else "</path>"))
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 5}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(13,{(T + H - B) / 2:.0f}) rotate(-90)" class="lab" text-anchor="middle">{esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def series_legend(points) -> str:
    n = collections.Counter(p[3] for p in points)
    shape = {"old": "<svg width='12' height='12' aria-hidden='true'><circle cx='6' cy='6' r='4.5' class='pt old'/></svg>",
             "new": "<svg width='12' height='12' aria-hidden='true'><path d='M6 0.6L11.4 6L6 11.4L0.6 6Z' class='pt new'/></svg>"}
    return ("<div class='slegend small'>" + "".join(f"<span>{shape[k]} {NIGHT_LABEL[k]} <span class='muted'>· {n.get(k, 0)} runs</span></span>"
                                                    for k in ("old", "new")) + "</div>")


def error_scatters(rows) -> tuple[str, dict]:
    """Measured gate error against (1) the lab's reference calibration and (2) the T1/T2echo coherence floor."""
    ref_pts, floor_pts, no_ref, no_coh, used = [], [], [], [], []
    plotted = [r for r in rows if r["status"] == "completed" and r["gate_fid"] is not None]
    for r in plotted:
        err = 100 * (1 - r["gate_fid"])
        label = f"{r['backend']} {r['q']} · {r['run']} · {fpct(r['gate_fid'])}"
        if r["ref"] is not None and r["ref"] < 1:
            used.append(r)
            ref_pts.append((100 * (1 - r["ref"]), err, label + f" · reference {fpct(r['ref'])} ({r['ref_date'][:16]})", r["night"]))
        else:
            no_ref.append(r)
        t1, t2e, L = r["t1"], r["t2e"], r["x180_len"]
        if t1 and t2e and L and t1 > 0 and t2e > 0:
            floor = 100 * (L * 1e-9 / 3.0) * (1.0 / t1 + 1.0 / t2e)
            floor_pts.append((floor, err, label + f" · T1 {1e6 * t1:.0f} µs, T2e {1e6 * t2e:.0f} µs, x180 {L:.0f} ns · floor {floor:.3f}%",
                              r["night"]))
        else:
            no_coh.append(r)
    no_valid = [r for r in rows if r["status"] == "completed" and r["gate_fid"] is None]
    names = lambda rs: ", ".join(f"{r['backend']} {r['q']} ({r['run']})" for r in sorted(rs, key=qkey)) or "none"  # noqa: E731
    a = scatter_svg(ref_pts, xlabel="reference calibration's gate error, %", ylabel="measured gate error, %")
    b = scatter_svg(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %", ylabel="measured gate error, %",
                    guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x"))
    phrase = "; ".join(f"{NIGHT_LABEL[k]}: {ref_dates_phrase([r for r in used if r['night'] == k])}" for k in ("old", "new"))
    html = (f"<h3>Measured gate error against the reference calibration</h3>"
            f"<p class='small muted'>One dot per completed qubit-run with a valid RB, log axes, error per gate (error per Clifford ÷ 1.875). "
            f"Below the diagonal the run beat the stored calibration; hover a dot for the qubit. The reference is gate_fidelity.averaged in "
            f"the cell's own source-state, the lab's value in the pull the night started from: {esc(phrase)} (GMT+3, the stamp the cloud "
            f"stores with each value) — the last calibration on file for that qubit, not necessarily the same RB protocol, and a qubit may "
            f"have drifted since. Not shown: runs whose qubit has no reference (averaged = 0), {names(no_ref)}; completed runs without a "
            f"valid RB, {names(no_valid)}.</p>"
            f"<div class='plotpanel'>{series_legend(ref_pts)}{a}</div>"
            f"<h3>Measured gate error against the coherence floor</h3>"
            f"<p class='small muted'>The floor is (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) with the run's own T1, T2echo and x180 length — the "
            f"error a perfect gate of that length would still have. A dot on the diagonal is decoherence-limited; far above it, the control "
            f"is the limit. Runs without a positive T1 or T2echo are not shown: {len(no_coh)} ({names(no_coh)}).</p>"
            f"<div class='plotpanel'>{series_legend(floor_pts)}{b}</div>")
    counts = {"ref": collections.Counter(p[3] for p in ref_pts), "floor": collections.Counter(p[3] for p in floor_pts),
              "no_ref": names(no_ref), "no_coh": names(no_coh), "no_valid": names(no_valid)}
    return html, counts


def per_qubit_table(rows) -> str:
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["backend"], r["q"])][r["night"]] = r
    body = []
    for key in sorted(by, key=lambda k: (ORDER[k[0]], k[1])):
        pair = by[key]
        for i, night in enumerate(("old", "new")):
            r = pair.get(night)
            first = i == 0
            qcell = (f"<td rowspan='2' class='qc'>{qtag(pair.get('new') or r)}</td>" if first else "")
            if r is None:
                body.append(f"<tr class='{night}'>{qcell}<td class='mono small'>{NIGHT_LABEL[night]}</td><td colspan='11' class='muted'>—</td></tr>")
                continue
            ok = is_straight(r)
            ck = (f"<span class='ck {'yes' if ok else 'no'}' title='{esc(st_reasons(r))}'>{'✓' if ok else '✗'}</span>")
            outside = "; ".join(o.split(" came back")[0].split(".", 1)[-1] for o in r["outside"])
            judge = (f"<span title='{esc('outside: ' + outside) if outside else 'all in range'}'>{r['ballpark']}/{r['graded']}</span>"
                     if r["graded"] else "—")
            x = "—" if r["x180_amp"] is None else f"{r['x180_amp']:.3f} / {r['x180_len']}"
            df = "—" if r["df01"] is None else f"{r['df01']:+.2f}"
            body.append(
                f"<tr class='{night}'>{qcell}<td class='mono small'>{esc(r['run'])}</td><td>{status_word(r)}</td><td class='c'>{ck}</td>"
                f"<td class='num'>{rb_cell(r)}</td><td class='num'>{pct(r['readout'])}</td>"
                f"<td class='num'>{df}</td><td class='num'>{x}</td>"
                f"<td class='num'>{us(r['t1'])} / {us(r['t2e'])}</td>"
                f"<td class='num'>{mins(r['qpu_s'], 1)}</td><td class='num'>{mins(r['wall_s'])}</td>"
                f"<td class='num'>{r['not_ok']}/{r['node_runs']}</td><td class='num'>{judge}</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>night</th><th>outcome</th><th title='straight through: the agent contributed nothing'>✓</th>"
            "<th>RB error / Clifford, %</th><th>readout, %</th><th>f₀₁ − lab, MHz</th><th>x180 amp / ns</th><th>T1 / T2echo, µs</th>"
            "<th>QPU, min</th><th>wall, min</th><th>not successful / node runs</th><th>judge</th></tr></thead>")
    return f"<div class='scroll'><table class='grid small pq'>{head}<tbody>{''.join(body)}</tbody></table></div>"


def fixed_table(idx) -> str:
    items = [
        ("arbel", "qD2", "Resonator identification judges every lobe that moved, each in its own window (4bf79df), and leaves out "
                         "the receiver's notch at the readout LO (856a8b5): three candidates, a 57 kHz-wide spur at 7.4539 GHz turned "
                         "down, the resonator at 7.41045 GHz on the first try. The chirp then found 0→1 at 6.0492 GHz."),
        ("arbel", "qD3", "A chirp window past the upconverter's ±400 MHz band now moves the LO for that run (f5464c5) instead of "
                         "being refused: 0→1 at 6.0676 GHz, identified as 0→1."),
        ("arbel", "qD4", "The same LO move, and a line not identified as 0→1 is now a failed outcome (98edd26): 0→1 at "
                         "6.63598 GHz, 3 kHz from the lab's."),
        ("arbel", "qD5", "The same: 0→1 at 5.90790 GHz, with its two-photon partner found 100 MHz below (5.80751 GHz)."),
        ("gilboa", "qD4", "Readout ceiling from the wiring instead of 0.1 of full scale (fb200fe): the power sweep reaches the "
                          "amplitudes the lab uses, the qubit shows in the first chirp."),
    ]

    def brief(r):
        if r is None:
            return "—"
        parts = [status_word(r)]
        if r["df01"] is not None:
            parts.append(f"f₀₁ {r['df01']:+.1f} MHz")
        if r["x180_len"]:
            parts.append(f"x180 {r['x180_len']} ns")
        if r["readout"] is not None:
            parts.append(f"readout {100 * r['readout']:.1f} %")
        parts.append(f"RB {rb_cell(r)} %" if r["rb"] is not None else "no RB")
        parts.append(f"{mins(r['wall_s'])} min")
        return parts[0] + "<div class='small num'>" + " · ".join(parts[1:]) + "</div>"
    body = "".join(
        f"<tr><td class='qc'>{qtag(idx[('new', b, q)])}</td><td>{brief(idx.get(('old', b, q)))}</td><td>{brief(idx.get(('new', b, q)))}</td>"
        f"<td class='small'>{esc(why)}</td></tr>" for b, q, why in items)
    return ("<div class='scroll'><table class='grid small fx'><thead><tr><th>qubit</th><th>n10+n11</th><th>n12</th><th>what changed</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def cost_table(S) -> str:
    def row(lab, fn):
        return f"<tr><th class='rowh'>{esc(lab)}</th>" + "".join(f"<td class='num'>{fn(S[k])}</td>" for k in ("old", "new")) + "</tr>"
    body = "".join([
        row("night, first start → last end", span_text),
        row("cells in flight at most", lambda s: "3 per device (up to 9)" if s is S["old"] else "3 across all devices"),
        row("model time per qubit-run, median", lambda s: f"{s['model_med'] / 60:.1f} min"),
        row("cloud queue per qubit-run, median", lambda s: f"{s['queue_med'] / 60:.1f} min"),
        row("model turns per qubit-run, median", lambda s: f"{s['turns_med']:.0f}"),
        row("node runs per qubit-run, median", lambda s: f"{s['runs_med']:.0f}"),
        row("readout power optimization, QPU per successful run, median", lambda s: f"{s['o08b_med']:.1f} s ({s['o08b_n']} runs)"),
    ])
    head = ("<thead><tr><th></th>" + "".join(f"<th class='grp'><span class='chip'><i style='background:{SERIES[k]}'></i>{NIGHT_LABEL[k]}</span></th>"
                                              for k in ("old", "new")) + "</tr></thead>")
    return f"<div class='scroll'><table class='grid pivot cost'>{head}<tbody>{body}</tbody></table></div>"


# ----------------------------------------------------------------------------- page
EXTRA_CSS = """
:root { --q-ok:#15803d; --q-warn:#b45309; --q-bad:#b91c1c; --q-ok-bg:#e7f5ec; --q-warn-bg:#fdf3e6; --q-bad-bg:#fdecec;
  --s-old:#2a78d6; --s-new:#c2410c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --q-ok:#5cd08c; --q-warn:#f2b24c; --q-bad:#f28585; --q-ok-bg:#14291c; --q-warn-bg:#2d2214; --q-bad-bg:#331a1a;
  --s-old:#3987e5; --s-new:#dd6b20; } }
:root[data-theme="dark"] { color-scheme: dark;
  --q-ok:#5cd08c; --q-warn:#f2b24c; --q-bad:#f28585; --q-ok-bg:#14291c; --q-warn-bg:#2d2214; --q-bad-bg:#331a1a;
  --s-old:#3987e5; --s-new:#dd6b20; }
/* make_night4_report's inline strip and scatter styles, with one colour per night */
.stripwrap { margin-top:6px; text-align:left; max-width:220px; } .stripwrap .small { white-space:normal; line-height:1.35; font-family:inherit; }
svg.strip { display:block; max-width:100%; height:auto; } svg.strip .axis { stroke:var(--muted); stroke-width:.8; }
svg.strip .tick { font-size:8px; fill:var(--muted); font-family:inherit; } svg.strip .thr { stroke:var(--crit); stroke-width:1; stroke-dasharray:3 2; }
svg.strip .dot { fill-opacity:.8; stroke:none; } svg.strip .dot.old { fill:var(--s-old); } svg.strip .dot.new { fill:var(--s-new); }
svg.strip .dot.hollow { fill:none; stroke-width:1.4; } svg.strip .dot.old.hollow { stroke:var(--s-old); } svg.strip .dot.new.hollow { stroke:var(--s-new); }
.plotpanel { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:10px 12px 6px; margin:8px 0 16px; max-width:640px; }
svg.scatter { display:block; width:100%; height:auto; } svg.scatter .grid { stroke:var(--surface-2); stroke-width:1; }
svg.scatter .frame { fill:none; stroke:var(--muted); stroke-width:.8; } svg.scatter .guide { stroke:var(--muted); stroke-width:1; stroke-dasharray:4 3; }
svg.scatter .tick { font-size:11px; fill:var(--muted); font-family:inherit; } svg.scatter .lab { font-size:12px; fill:var(--ink-2); font-family:inherit; }
.pt { fill-opacity:.85; stroke:var(--surface); stroke-width:1; } .pt.old { fill:var(--s-old); } .pt.new { fill:var(--s-new); }
svg.scatter .pt:hover { fill-opacity:1; stroke:var(--ink); }
.slegend { display:flex; flex-wrap:wrap; gap:4px 18px; margin:2px 0 6px; color:var(--ink-2); }
.slegend span { display:inline-flex; align-items:center; gap:6px; } .slegend svg { overflow:visible; }
table.pivot .chip i { width:10px; height:10px; }
.page { padding-block:36px 72px; padding-inline:24px; }
@media (max-width:640px) { .page { padding-inline:16px; } h1 { font-size:27px; } }
.ql { display:inline-flex; flex-wrap:wrap; gap:0 9px; font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:12.5px; line-height:1.6; }
.q { white-space:nowrap; color:var(--q-ok); font-family:"JetBrains Mono",Menlo,Consolas,monospace; }
.q .dv, .dvl { font-size:.74em; color:var(--muted); font-weight:400; margin-right:1px; position:relative; top:-.28em; }
.q.warn { color:var(--q-warn); font-weight:600; text-decoration:underline dotted; text-underline-offset:3px; cursor:help; }
.q.bad { color:var(--q-bad); font-weight:600; cursor:help; }
.q[title] { cursor:help; }
.qs { font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:12px; padding:1px 6px; border-radius:5px; white-space:nowrap; cursor:help; }
.qs.ok { color:var(--q-ok); background:var(--q-ok-bg); } .qs.warn { color:var(--q-warn); background:var(--q-warn-bg); }
.qs.bad { color:var(--q-bad); background:var(--q-bad-bg); }
.ck { cursor:help; font-weight:700; } .ck.yes { color:var(--q-ok); } .ck.no { color:var(--muted); font-weight:400; }
td.c { text-align:center; }
table.sum { min-width:600px; } table.sum th.rowh { width:30%; font-weight:500; font-size:13px; } table.sum td { font-size:14px; }
table.cost { min-width:520px; } table.cost th.rowh { width:44%; }
table.pq tr.old td { color:var(--ink-2); border-bottom-style:dashed; }
table.pq tr.new td { border-bottom:1px solid var(--line-strong); }
table.pq td.qc { vertical-align:middle; border-bottom:1px solid var(--line-strong); font-size:13.5px; }
table.fx td { min-width:150px; } table.fx td.qc { min-width:0; }
.legend { gap:8px 18px; margin:10px 0 4px; color:var(--ink-2); }
.case { margin-top:30px; padding-top:14px; border-top:1px solid var(--line); }
.case h3 { margin-top:0; display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 12px; }
.case .facts { margin:6px 0 8px; color:var(--ink-2); }
ol.chain { max-width:80ch; margin:8px 0 10px; padding-left:24px; }
ol.chain li { margin:4px 0; } ol.chain li::marker { font-family:"JetBrains Mono",monospace; font-size:.85em; color:var(--muted); }
.n13 { background:var(--surface-2); border-left:3px solid var(--q-ok); padding:8px 12px; border-radius:6px; margin:10px 0; max-width:80ch; }
.n13 b:first-child { color:var(--q-ok); }
ul.tight { max-width:80ch; padding-left:20px; } ul.tight li { margin:5px 0; }
table.fu td:first-child { white-space:nowrap; }
.wrapany { overflow-wrap:anywhere; }
.toc { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:14px; margin:14px 0 0; }
"""


def n13_ref(text="the n13 report") -> str:
    return f"<a href='{esc(N13_REPORT_URL)}'>{esc(text)}</a>" if N13_REPORT_URL else f"{esc(text)} (30 Sep)"


def case(idx, b, q, title, chain, n13, old_note=None) -> str:
    r, o = idx[("new", b, q)], idx.get(("old", b, q))
    facts = [f"{mins(r['wall_s'])} min wall", f"{mins(r['qpu_s'], 1)} min QPU", f"{r['turns']} turns",
             f"{r['not_ok']} of {r['node_runs']} node runs not successful", f"${r['cost']:.2f}"]
    old = old_note or ""
    if not old and o:
        old = f"{o['run']}: {o['status']}"
        if o["rb"] and o["rb"] > 0:
            old += f", RB {100 * o['rb']:.3f} %"
        if o["readout"]:
            old += f", readout {100 * o['readout']:.1f} %"
    return (f"<div class='case' id='case-{esc(b)}-{esc(q)}'><h3>{qtag(r)} <span>{esc(title)}</span></h3>"
            f"<div class='facts small'>{status_word(r)} · {' · '.join(esc(f) for f in facts)}"
            f"<br><span class='muted'>{esc(old)}</span></div>"
            f"<ol class='chain'>{''.join(f'<li>{c}</li>' for c in chain)}</ol>"
            f"<div class='n13 small'><b>Fixed in n13.</b> {n13}</div></div>")


def build():
    rows = collect()
    S = {}
    for k in ("old", "new"):
        rs = [r for r in rows if r["night"] == k]
        S[k] = night_stats(rs)
        S[k]["rows"] = rs
    o, n = S["old"], S["new"]
    idx = {(r["night"], r["backend"], r["q"]): r for r in rows}
    now = datetime.now().strftime("%d %b %Y %H:%M")

    # nearest-to-straight-through cells tonight (fewest contributions of the kinds that disqualify)
    def dev(r):
        s = r["st"]
        return s["failed"] + s["extra"] + s["changed"] + s["unproposed"] + s["unwritten"]
    done_new = sorted([r for r in n["done"]], key=lambda r: (dev(r), qkey(r)))
    nearest = [r for r in done_new if dev(r) == dev(done_new[0])] if done_new else []
    nearest_txt = ", ".join(f"{r['backend']} {r['q']}" for r in nearest)
    nearest_why = sorted({" · ".join(f"{r['st'][k]} {lab}" for k, lab in (
        ("failed", "failed or refused run"), ("extra", "rerun"), ("changed", "changed value"),
        ("unproposed", "write no node proposed"), ("unwritten", "proposal left unwritten")) if r["st"][k]) for r in nearest})

    gd = lambda k: [idx[(k, "gilboa", f"qD{i}")] for i in range(1, 6) if (k, "gilboa", f"qD{i}") in idx]  # noqa: E731
    ad = lambda k: [idx[(k, "arbel", f"qD{i}")] for i in range(2, 6) if (k, "arbel", f"qD{i}") in idx]  # noqa: E731
    ro_list = lambda rs: ", ".join("—" if r["readout"] is None else f"{100 * r['readout']:.1f}" for r in rs)  # noqa: E731

    rq = lambda b, q, k="new": idx[(k, b, q)]  # noqa: E731
    qb4, qc3, q6, q4, qd1 = rq("arbel", "qB4"), rq("gilboa", "qC3"), rq("qolab", "Q6"), rq("qolab", "Q4"), rq("arbel", "qD1")
    fail_cost = qb4["cost"] + qd1["cost"]

    cases = [
        case(idx, "arbel", "qB4", "02b's low-power line committed as the readout frequency", [
            "T5–T6 · resonator vs power (02b) found no onset (“too few clean line centres below the onset”) and proposed no power; "
            "the rerun it asked for did no better. It still reported a low-power “dressed” line, 7.74635 GHz.",
            "T7 · the agent committed that line as the readout frequency.",
            "T10–T11 · the resonator flux map put the arc's apex at 7.74385 GHz (0.032 V); the agent committed 7.74381 GHz there.",
            "T12 · 02b at the sweet spot: noise-limited, no power again, dressed line 7.74685 GHz, and a warning that its refit had moved "
            "the given frequency by −4.8 MHz, “check it against resonator_identification”.",
            "T13 · the agent committed 7.74685 GHz at 0.224 V as “the correct low-power value”: 3 MHz above the apex; the lab reads at "
            "7.74374 GHz. At that frequency the |g⟩–|e⟩ separation is about 3 % of its peak (IQ blobs 52.5 % against 96.9 %, measured "
            "by the n13 fix).",
            "T15–T148 · no chirp saw a line, including windows that held f₀₁ (6.6059 GHz). The agent moved the drive LO, tried upconverter 2, "
            "biased the flux ±0.5 V, and never went back to the readout frequency. Escalated at the cap with readout 50.8 %, and f₀₁ "
            f"and x180 still at their scrambled values.",
        ], "A broad, power-independent trough sits about 3 MHz above qB4's 0.7 MHz-wide resonance; 02b's whole-window fit latched "
           "onto it, so no qB4 power map had an onset on any night since n7. 02b now refits locally, and every frequency it reports "
           "carries the power it holds at. n13 qB4: RB 0.133 %, readout 96.4 %, judge 4/4, 23 min."),
        case(idx, "gilboa", "qC3", "T1 written in milliseconds", [
            "T16 · T1_chirp measured T1 = 29.98 ± 1.81 µs and proposed 2.998·10⁻⁵ s.",
            "T17 · the agent wrote 0.0299774 (“T1 measured 30 +/- 1.8 us … Committing”): 30 ms, a thousand times the measurement.",
            "Every thermal reset then waited 5 × T1 = 150 ms per shot. The qubit flux map's pre-flight priced 12 129 s and refused; "
            "power Rabi and readout power optimization timed out at 60 s.",
            "The agent cut shots rather than question T1: four flux maps at 4 shots, all failed; power Rabi at 10; Ramsey, error "
            "amplification and DRAG at 20; T1 at 10–15; RB at 40 sequences × 5 shots to depth 256. Most of it moved to active reset, "
            "which does not wait on T1.",
            f"Completed with RB {100 * qc3['rb']:.3f} %, readout {100 * qc3['readout']:.1f} %, T1 {us(qc3['t1'])} µs and T2echo "
            f"{us(qc3['t2e'])}. The judge gave it {qc3['ballpark']}/{qc3['graded']}: it grades frequencies and amplitudes, and those "
            "survived.",
        ], "tinycal's write guard refuses a value outside the parameter's range or 10ᵏ (k ≥ 3) off the node's proposal unless it is "
           "resent with confirm (replayed on 12 406 writes: 39 refusals, 37 of them real slips, no false positives); the job pre-flight "
           "refuses a reset wait over 25 ms. n13 qC3: RB 0.257 %, readout 98.0 %, judge 4/4. The model wrote T1 correctly, so the guard "
           "did not have to fire."),
        case(idx, "qolab", "Q6", "Sweet spot outside the first flux window, never widened", [
            "The lab's sweet spot is at joint 0.747 V (f₀₁ 5.0917 GHz); the scrambled state starts the flux at 0 V.",
            "T8 · the first resonator flux map (±0.5 V) covered a quarter of a period. The apex lay past the high edge; the node proposed "
            "no idle offset and said “widen the sweep toward the high side until the maximum is inside the data”, but reported the run "
            "successful, with no retry parameters, and printed an unresolved φ₀ of 3.9 V.",
            "T10 · the agent wrote the map's edge, 0.48 V, as the idle offset: “within the port limit of +0.5 V. The port cannot reach "
            "a second maximum a full period (3.9 V) away”. qolab's flux ports are amplified, ±2.5 V.",
            "T22–T45 · the qubit flux map (scatter 227× its errors) put an apex at 0.27 V; the agent went there, lost the resonator, "
            "went back to 0 V, then back to 0.48 V.",
            "T57–T64 · Ramsey vs flux moved it to 0.4875 V. The agent noted “the positive quad_term confirms we're not at the qubit sweet "
            f"spot” and went on to RB: {100 * q6['rb']:.1f} %, f₀₁ {q6['df01']:+.1f} MHz from the lab's.",
            "On n10 the agent widened the map to +1.2 V and found 0.7485 V.",
        ], "02c returns retry parameters that extend the window toward the apex, capped by the port class (±0.5 V direct, ±2.5 V "
           "amplified) and the 60 s budget; with no idle offset the outcome is failed, and φ₀ is withheld below 0.75 of a period. n13 Q6: "
           "f₀₁ −0.31 MHz, RB 0.283 %, judge 4/4, 12 min. Retries can now drive amplified qolab ports up to ±2.5 V."),
        case(idx, "qolab", "Q4", "RB depth cut to 64 around a node crash", [
            "The bring-up graph pins RB's max_circuit_depth at 2048 (cc0c349, after n10's agents cut it on 22 of 34 final RB runs).",
            "T51 · the agent overrode it with 128 (“well under T2echo so the decay fit stays clean”). The node crashed with "
            "KeyError 'fit_results' after 1.0 s of QPU.",
            "T52 · it cut to 64: 0.778 % ± 0.757 % per Clifford over 1.003 decay lengths, just over the node's 1.0 floor, with the node's "
            "own warning that the error was extrapolated and the sweep needed max_circuit_depth ≥ 192. Accepted as final.",
            "The qubit had not regressed: n10 measured 0.073 %, and T1 was 99 µs tonight.",
        ], "The fit started from a = 0 on log sweeps of ten depths or fewer, failed, left no fit_results, and update_state raised before "
           "the data were saved; the depth-64 free fit put its floor on an unreached tail (true coverage about 0.09 decay lengths). The fit "
           "now falls back to other starts, an unphysical fit is not a measurement, an anchored fit sizes recommended_max_circuit_depth, and "
           "the node always saves; the recipe keeps the preset depth. n13 Q4: RB 0.114 % over 4.7 decay lengths, one run at 2048, judge 4/4.",
             old_note=f"n10: completed, RB 0.073 % at depth 2048"),
        case(idx, "arbel", "qD1", "Still failing: the line never found", [
            "The lab parks qD1 at joint 0.227 V, f₀₁ 4.9975 GHz, 197 mV off its sweet spot; at the arc's maximum (0.030 V) the 0→1 line "
            "sits at 5.366 GHz, 419 MHz above the scrambled f₀₁ (found by the n13 fix).",
            "T9–T17 · eight chirp windows came back “no line in the window at any drive level”, the widest 700 MHz on the scrambled "
            "value (up to 5.297 GHz); a 780 MHz window was refused (IF −418…+382 MHz). None reached 5.366 GHz.",
            "T25 · the fine node labelled a lone spike “UNRESOLVED BUT REAL” (SNR 52.5, R² 0.67). The agent chased it for the remaining "
            "125 turns: 15 fine scans, 12 power Rabi and 14 T1 runs.",
            f"Failed at the 150-turn budget: readout {100 * qd1['readout']:.1f} %, f₀₁ {qd1['df01']:+.1f} MHz from the lab's. n10 "
            "escalated on the same qubit, then with every window near the line refused at the IF edge.",
        ], "The chirp now returns the next window when it finds nothing, tiling 675 MHz upward to 300 MHz below the resonator, then one "
           "below, then “exhausted”; the fine node judges an unresolved fit by its measured excursion. n13 qD1: line at turn 14, RB "
           "0.152 %, readout 98.3 %, 13 min. The judge gives it 2/4: its reference f₀₁ is the lab's off-sweet-spot value.",
             old_note="n10: escalated, no 0→1 line found"),
    ]

    followups = [
        ("gilboa qC3", "tinycal write_state: flag a ×1000 change against the node's proposal.",
         "n13: write guard (range, 10ᵏ slip, confirm to override); pre-flight refuses resets over 25 ms and now covers 04b/04c/07/08a/08b."),
        ("qolab Q4", "RB node: a failed fit raises KeyError 'fit_results' instead of reporting (also seen in the reset A/B on Q2).",
         "n13: fallback starts, failure is an outcome, the node always saves."),
        ("qolab Q4", "Refuse RB depth overrides that the node's own T1/T2 say cannot cover 2 decay lengths.",
         "n13: recommended_max_circuit_depth from an anchored fit; free fits under 2 decay lengths fail when it covers < 1."),
        ("qolab Q6", "02c: propose the wider rerun when the arc's apex lies outside the window, as 02b's retry_parameters does.",
         "n13: retry_flux_window(), capped by the port class."),
        ("arbel qB4", "02b: when it proposes no power, do not offer its low-power line as a readout frequency, or say at which power it holds.",
         "n13: local refit; every frequency carries its power."),
        ("arbel qD1", "(not a follow-up at the time: the log called it the same hard case as n10)",
         "n13: chirp next-window tiling; fine node judged by excursion."),
        ("qolab Q4, readout", f"08b's 0.5–3.5× preset settles at 1.6–3× the lab's readout amplitude, outside the judge's 0.6–1.6× band, "
                              f"and costs more QPU (median {n['o08b_med']:.1f} s a run against {o['o08b_med']:.1f} s on n10+n11).",
         "open: a judge band or a preset question"),
        ("gilboa qC1", "DRAG fails on every run (n11 needed ten tries; n12 none converged), so the x180 keeps its default α.", "open"),
        ("arbel qA6, qB3", "Upper sweet spot, 234 and 33.5 MHz above where the lab parks them: the judge flags f₀₁ every night.",
         "open: a benchmark-reference decision (qD1 joined it in n13)"),
    ]
    fu_rows = "".join(f"<tr><td class='mono small'>{esc(a)}</td><td>{esc(b)}</td><td class='small'>{esc(c)}</td></tr>" for a, b, c in followups)

    ro_amp_line = (f"{n['ro_ratio_hi']} of the {len(n['done'])} completions ended above 1.6× the lab's readout amplitude (median ratio "
                   f"{n['ro_ratio_med']:.2f}×; n10+n11: {o['ro_ratio_hi']} of {len(o['done'])}, {o['ro_ratio_med']:.2f}×), and the judge "
                   f"flagged readout amplitude on {n['judge_ro_amp']} cells (n10+n11: {o['judge_ro_amp']}).")

    gc = lambda q: rq("gilboa", q)  # noqa: E731
    go = lambda q: rq("gilboa", q, "old")  # noqa: E731

    scatters_html, scatter_counts = error_scatters(rows)
    page = f"""<title>n12 Bring-up Rerun</title>
<meta name="description" content="n12, 29–30 Sep 2026: tinycal with qwen3.8-27b reran the n10+n11 single-qubit bring-ups on the same 37 qubits of qolab, arbel and gilboa.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{EXTRA_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · the n10+n11 bring-ups rerun, overnight run n12 · generated {esc(now)}</div>
<h1 style="margin-top:8px">tinycal with qwen3.8-27b rerunning the n10+n11 single-qubit bring-ups on the same 37 qubits of qolab, arbel and gilboa, 29–30 Sep 2026</h1>
<p class="lede">One framework, one model, one host, the whole graph, the same qubits as the night before. tinycal with qwen3.8-27b on
OpenRouter (effort high, 16 000-token cap, 150 turns) calibrated each qubit from a scrambled state through the 18 nodes of the flux-tunable
bring-up graph, one cell per qubit, at most three in flight across all devices, from 18:56 to 00:24. Targets: qolab Q1–Q6; arbel all 21
qubits; gilboa the ten C/D qubits (the B row's readout line has a broken TWPA). Against n10+n11: {len(n['done'])} completions instead of
{len(o['done'])}, {len(n['meaningful'])} with meaningful results instead of {len(o['meaningful'])}, readout median {100 * n['ro_med']:.1f} %
instead of {100 * o['ro_med']:.1f} %, wall median {n['wall_med'] / 60:.0f} min instead of {o['wall_med'] / 60:.0f}, for the same
${n['judge_cost']:.2f}. The arbel D row and gilboa qD4 now calibrate; four qubits broke in new ways and arbel qD1 still fails, and all five
were fixed and validated the next night ({n13_ref()}). Numbers are read from each qubit-run's result.json, tinycal's event log and the lab's
pulled state; the operator annotations are marked.</p>
<nav class="toc"><a href="#overview">Overview</a><a href="#plots">Gate error plots</a><a href="#apparatus">Apparatus</a>
<a href="#fixed">What got fixed</a><a href="#failures">New failures</a><a href="#per-qubit">Per qubit</a><a href="#cost">Cost and time</a>
<a href="#followups">Follow-ups</a></nav>

<h2 id="overview">Overview</h2>
<p class="small muted">How to read the table: one qubit-run = tinycal calibrating one qubit from the scrambled state with qwen3.8-27b on
OpenRouter. Gate fidelity = 1 − (RB error per Clifford ÷ 1.875). "Per calibration" = the total over every qubit-run of the night divided by
the number completed. Agent time is time inside model calls; QPU time is execution on the chip; queue wait is the cloud queue. Judge cost =
tokens at prices.yaml (OpenRouter's rate for qwen3.8-27b). Context = prompt size per model call in tokens, from tinycal's events.jsonl.
Both nights ran the same judge (“identity ballpark” on four graded parameters per qubit).</p>
<h3>By night</h3>
{summary_table(S)}
{legend()}
<p class="small muted"><b>Straight through</b> = completed; every graph node run once, with no failed or refused run; every value a node
proposed written as proposed (numbers within 10⁻⁴ relative, to allow the agent's retyping), no write to a path no node proposed, and no
proposal left unwritten unless it was a no-op. Read from each cell's events.jsonl, with the same rule as the n13 report. IQCC 503 and
500 answers and read timeouts are not runs, nor is a malformed call with no node name; 60 s job timeouts, pre-flight refusals, rejected
parameters and config errors count as failed runs. The recipe's second power sweep at the sweet spot is not a rerun; counting it as one
gives the same {len(o['st_strict'])} and {len(n['st_strict'])}. Parameters the model passed are not part of the definition (graph presets
never count); the first sub-line is the stricter test. <b>No cell on either night qualifies</b>, and none even ran every node once without a
failure. The nearest n12 cells ({esc(nearest_txt)}) each had {esc(' / '.join(nearest_why))}. The two commonest contributions: in
{n['flux_refused']} of 37 n12 cells the first resonator flux map was refused because the model asked for ±0.6–1.25 V on a ±0.5 V direct
LF-FEM port ({o['flux_refused']} on n10+n11); in {n['quad_unproposed']} ({o['quad_unproposed']}) the agent copied the qubit flux map's
curvature into freq_vs_flux_01_quad_term, which no node had proposed at that point. “Off the recipe order” is judged against each night's
recipe (n10+n11 with T1_coarse, n12 without). The ✓ column of the per-qubit table has each cell's tally on hover.</p>
<div id="plots">{scatters_html}</div>

<h3 id="apparatus">Apparatus: what changed since n10</h3>
<p><b>qua-libs</b> <span class="mono">~/qab-runs/qua-libs-n12</span> = feat/qualibrate-ai <span class="mono">4bf79df</span>; n10 ran on
feat/bringup-fast-presets <span class="mono">08ebca5</span>. In between, in the order they bear on this night:</p>
<ul class="tight">
<li><b>Drive window</b> <span class="mono small">f5464c5</span>: a spectroscopy window past the upconverter's band moves the LO for that run instead of being refused. <span class="mono small">98edd26</span>, <span class="mono small">1cfb1da</span>: a line not identified as 0→1, or no line, is a failed outcome with no proposal.</li>
<li><b>Drive amplitudes from the chirp</b> <span class="mono small">ce69f5b</span>: an x180 proposal from the measured Rabi rate, and the fine scan driven at 1 MHz of Rabi. Power Rabi in 0.02 steps with no separate x90 sweep <span class="mono small">d90fe34</span>.</li>
<li><b>Readout</b>: the amplitude ceiling comes from the wiring instead of a fixed 0.1 of full scale <span class="mono small">fb200fe</span>; readout power optimization sweeps 0.5–3.5× <span class="mono small">21edcb1</span>; 02b proposes the rerun that can measure an onset it could not <span class="mono small">b5e7d0e</span>; 02a/02b leave out the receiver's notch at IF 0 <span class="mono small">856a8b5</span>.</li>
<li><b>Resonator identification</b> judges every lobe that moved, each in a window of its own <span class="mono small">4bf79df</span>.</li>
<li><b>Active reset</b> through quam's loop with its waits in clock cycles, shared by the single-qubit nodes <span class="mono small">697434f</span>.</li>
<li><b>T1 and RB</b>: T1_coarse dropped from the graph (19 → 18 nodes; T1 on I/Q only when T1_chirp stored none) and T1 sized from the stored value <span class="mono small">2a62517, 5faaee2, 18aebe6</span>; RB withholds a gate fidelity under one decay length <span class="mono small">20397e7</span> and the graph pins its depth at 2048 <span class="mono small">cc0c349</span>.</li>
</ul>
<p><b>tinycal</b> at <span class="mono">5146a6c</span> plus an uncommitted step-2 recipe edit (readout power at the knee only; committed since as
<span class="mono">dfd25ca</span>). Recipe changes since n10: T1 on I/Q only without a T1_chirp value, commit the x180 the chirp proposes,
no power Rabi for x90. <b>States</b>: qolab and gilboa pulled fresh (gilboa's active_qubit_names patched: the pull omitted qC1 and qC3); arbel
reused the 29 Sep 00:17 pull, because its fresh state used a DrachmaReadoutPulse that quam-builder 0.5.0 cannot load. Workload
decalibrate_chip.yaml, scramble spec <span class="mono">f47053f145f74c8c</span>.</p>

<h2 id="fixed">What got fixed</h2>
<p>The four arbel D-row qubits sit +399 to +408 MHz from their drive LOs; on n10+n11 none of their 0→1 lines was ever inside a window
the chirp would play. Tonight all four completed on the right line, and gilboa qD4, which the 0.1 V readout ceiling had held at 52 %,
completed with judge 4/4.</p>
{fixed_table(idx)}
<p><b>D-row readout in general.</b> gilboa qD1–qD5: {ro_list(gd('old'))} % on n10+n11, {ro_list(gd('new'))} % tonight. arbel qD2–qD5:
{ro_list(ad('old'))} % then, {ro_list(ad('new'))} % now. Across all 37 qubits {n['ro_95']} reached ≥ 95 % assignment against {o['ro_95']}.</p>

<h2 id="failures">The new failures, and qD1</h2>
<p>Four qubits that completed properly on n10+n11 did not tonight, and qD1 failed again. Each chain is the operator's reading of the
transcript (T = model turn), checked against the event log. All five were fixed that night and validated in the n13 rerun of all 37
qubits: {n13_ref()}{'' if N13_REPORT_URL else f" <span class='mono small wrapany'>{N13_REPORT_FILE}</span>"}. The two cells that never finished cost ${fail_cost:.2f}, {100 * fail_cost / n['cost']:.0f} % of the night's spend,
and {(qb4['wall_s'] + qd1['wall_s']) / 60:.0f} min of wall.</p>
{''.join(cases)}

<h2 id="per-qubit">Per qubit, n10+n11 against n12</h2>
<p class="small muted">Two rows per qubit: n10 or n11 (muted), then n12. RB<sup>×</sup> = the fit saw under one decay length, not a
measurement. f₀₁ − lab is the final f₀₁ minus the reference in the work dir's source-state. T1 and T2echo are the bring-up's own values
(ms where they are absurd). “Not successful” counts every node run whose outcome was not successful, IQCC transients included. Judge =
graded parameters in range (the ones outside on hover). ✓ = straight through, with the reasons on hover.</p>
{legend()}
{per_qubit_table(rows)}
<h3>Read across the table</h3>
<ul class="tight">
<li><b>Readout amplitude above the lab's.</b> {ro_amp_line} Readout power optimization's 0.5–3.5× sweep settles at 1.6–3× the lab's
amplitude, usually with higher fidelity on qolab, not always on arbel (qB2: 95.3 % at 2.8×).</li>
<li><b>Upper sweet spots.</b> arbel qA6 and qB3 were again calibrated at their upper sweet spots, {rq('arbel', 'qA6')['df01']:+.1f} and
{rq('arbel', 'qB3')['df01']:+.1f} MHz from where the lab parks them; the judge flags f₀₁ on both.</li>
<li><b>Poor every night.</b> gilboa qC1 (RB {100 * go('qC1')['rb']:.1f} → {100 * gc('qC1')['rb']:.1f} %, DRAG never converged tonight),
qC4 ({100 * go('qC4')['rb']:.1f} → {100 * gc('qC4')['rb']:.1f} %) and qD2 (T1 about 1 µs, {100 * go('qD2')['rb']:.1f} → {100 * gc('qD2')['rb']:.1f} %).</li>
<li><b>RB validity.</b> Every completion tonight has a fit over at least one decay length, against {len(o['valid'])} of {len(o['done'])}
on n10+n11. gilboa qD3's {100 * gc('qD3')['rb']:.3f} % covers 15 decay lengths; n10's lower {100 * go('qD3')['rb']:.3f} % was extrapolated
from 1.8. qolab Q2 now has a valid {100 * rq('qolab', 'Q2')['rb']:.3f} % (n10's 0.030 % was under one decay length) but sits 0.05 V off the
lab's joint offset, f₀₁ {rq('qolab', 'Q2')['df01']:+.1f} MHz.</li>
<li><b>Agent choices that cost QPU.</b> qolab Q1 took {rq('qolab', 'Q1')['qpu_s'] / 60:.1f} min of QPU: the agent overrode the graph's active
reset with thermal on error amplification, DRAG (4 runs, 91 s) and RB. qolab Q3's f₀₁ is {rq('qolab', 'Q3')['df01']:+.2f} MHz from the lab's
every night; the lab's value is the suspect one.</li>
</ul>

<h2 id="cost">Cost and time</h2>
<p>The same money bought two more completions and eight fewer minutes of wall per qubit. The saving is the model's: the median qubit
spent {n['model_med'] / 60:.1f} min in model calls against {o['model_med'] / 60:.1f}, in {n['turns_med']:.0f} turns against
{o['turns_med']:.0f}, with {n['runs_med']:.0f} node runs against {o['runs_med']:.0f}. The night itself ran longer, because n12 held
three cells in flight across all devices, where n10+n11 held three per device.</p>
{cost_table(S)}

<h2 id="followups">Follow-ups listed at the time</h2>
<p class="small muted">From the n12 operator log, with what n13 did about each.</p>
<div class="scroll"><table class="grid small fu"><thead><tr><th>from</th><th>follow-up</th><th>status</th></tr></thead><tbody>{fu_rows}</tbody></table></div>

<p class="small muted" style="margin-top:28px">Sources: ~/qab-runs/n12-LOG.md; cells ~/qab-runs/n12-&lt;backend&gt;-20260929-1855/ and
the n10/n11 equivalents; transcripts ~/code/QM/tinycal/runs/n12_*_20260929-1855/. Generated by make_n12_report.py.</p>
</div>"""
    OUT.write_text(page)
    for k in ("old", "new"):
        s = S[k]
        print(f"{NIGHT_LABEL[k]}: completed {len(s['done'])}/{s['n']}, meaningful {len(s['meaningful'])}, straight {len(s['st'])} "
              f"(strict {len(s['st_strict'])}, no overrides {len(s['st_noov'])}), valid RB {len(s['valid'])} median {100 * s['rb_med']:.3f} %, "
              f"readout median {100 * s['ro_med']:.1f} % ({s['ro_95']} >= 95 %), judge {s['judge'][0]}/{s['judge'][1]}, "
              f"QPU {s['qpu_tot'] / 60:.0f} / {s['qpu_med'] / 60:.1f} min, wall median {s['wall_med'] / 60:.0f} min, cost ${s['cost']:.2f}, "
              f"node runs {s['runs']} / {s['not_ok']}, flux refused {s['flux_refused']}")
    print(f"nearest straight-through tonight: {nearest_txt}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
