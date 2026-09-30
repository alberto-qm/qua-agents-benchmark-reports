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
from make_fwcmp2_report import CSS, RUNS, TINYCAL_RUNS, esc, g, median, x180  # noqa: E402

OUT = Path(__file__).with_name("2026-09-29-n12-full-bringup-qwen-tinycal.html")
N13_REPORT_URL = os.environ.get("N13_REPORT_URL", "")  # the published n13 report, when there is one
N13_REPORT_FILE = "2026-09-30-n13-fixes-validation-qwen-tinycal.html"  # make_n13_report.py's output, next to this one
BACKENDS = ("qolab", "arbel", "gilboa")
DEV_MARK = {"qolab": "q", "arbel": "a", "gilboa": "g"}
NIGHTS = {"old": ("n10", "n11"), "new": ("n12",)}
NIGHT_LABEL = {"old": "n10+n11", "new": "n12"}
NIGHT_DATES = {"old": "28–29 Sep", "new": "29–30 Sep"}
TRANSIENT_ERRORS = ("ConnectionError", "ReadTimeout")  # IQCC 503/500 answers and read timeouts: no result came back
REL_TOL = 1e-4  # a write "as proposed": the agent retypes and rounds (7662258540.95 -> 7662260000)
RB_VALID_DECAY_LENGTHS = 1.0

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
                for cell in sorted(p for p in work.iterdir() if p.is_dir() and "-tinycal-" in p.name):
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
                    rows.append({
                        "night": night, "run": prefix, "backend": backend, "q": q, "work": work.name, "cell": cell.name,
                        "run_id": run_id, "status": x.get("status"),
                        "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
                        "rb": g(qual, "rb", "error_per_clifford"),
                        "rb_cover": _num(rb_runs[-1], "decay_lengths_covered") if rb_runs else None,
                        "rb_depth": ((rb_runs[-1].get("parameters") or {}).get("max_circuit_depth") or 2048) if rb_runs else None,
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
            and (r["rb_cover"] is None or r["rb_cover"] >= RB_VALID_DECAY_LENGTHS))


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


# ----------------------------------------------------------------------------- tables
def summary_table(S) -> str:
    def both(fn):
        return "".join(f"<td>{fn(S[k], k)}</td>" for k in ("old", "new"))

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

    body = [
        ("qubits, by outcome", lambda s, k: qubit_list(s["rows"])),
        ("completed", lambda s, k: f"<b class='num'>{len(s['done'])}/{s['n']}</b>"),
        ("completed with meaningful results", lambda s, k: f"<b class='num'>{len(s['meaningful'])}/{s['n']}</b>"),
        ("straight through: the agent contributed nothing", st_cell),
        ("valid RB (fit saw ≥ 1 decay length)", valid_cell),
        ("readout assignment, median", lambda s, k: f"<span class='num'><b>{100 * s['ro_med']:.1f} %</b> · {s['ro_95']} of {s['ro_n']} ≥ 95 %</span>"),
        ("judge: graded parameters in range", lambda s, k: f"<span class='num'><b>{s['judge'][0]}/{s['judge'][1]}</b></span>"),
        ("QPU, total / median per qubit", lambda s, k: f"<span class='num'><b>{s['qpu_tot'] / 60:.0f}</b> / {s['qpu_med'] / 60:.1f} min</span>"),
        ("wall per qubit, median", lambda s, k: f"<span class='num'><b>{s['wall_med'] / 60:.0f} min</b></span>"),
        ("node runs / not successful", lambda s, k: f"<span class='num'>{s['runs']} / {s['not_ok']}</span>"
                                                   f"<div class='small muted'>{s['transient']} of them IQCC 500/503 answers or read timeouts</div>"),
        ("tinycal cost estimate", lambda s, k: f"<span class='num'><b>${s['cost']:.2f}</b> · ${s['cost'] / len(s['meaningful']):.2f} per meaningful calibration</span>"),
    ]
    trs = "".join(f"<tr><th class='rowh'>{esc(lab)}</th>{both(fn)}</tr>" for lab, fn in body)
    head = ("<thead><tr><th></th>" + "".join(f"<th class='grp'>{NIGHT_LABEL[k]} <span class='small muted'>· {NIGHT_DATES[k]}</span></th>"
                                              for k in ("old", "new")) + "</tr></thead>")
    return f"<div class='scroll'><table class='grid pivot sum'>{head}<tbody>{trs}</tbody></table></div>"


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
        row("tinycal cost estimate, total", lambda s: f"${s['cost']:.2f}"),
        row("per completed / per meaningful calibration", lambda s: f"${s['cost'] / len(s['done']):.2f} / ${s['cost'] / len(s['meaningful']):.2f}"),
        row("model time per qubit, median", lambda s: f"{s['model_med'] / 60:.1f} min"),
        row("cloud queue per qubit, median", lambda s: f"{s['queue_med'] / 60:.1f} min"),
        row("QPU per qubit, median (total)", lambda s: f"{s['qpu_med'] / 60:.1f} min ({s['qpu_tot'] / 60:.0f} min)"),
        row("wall per qubit, median", lambda s: f"{s['wall_med'] / 60:.0f} min"),
        row("model turns per qubit, median", lambda s: f"{s['turns_med']:.0f}"),
        row("node runs per qubit, median", lambda s: f"{s['runs_med']:.0f}"),
        row("readout power optimization, QPU per successful run, median", lambda s: f"{s['o08b_med']:.1f} s ({s['o08b_n']} runs)"),
    ])
    head = "<thead><tr><th></th>" + "".join(f"<th class='grp'>{NIGHT_LABEL[k]}</th>" for k in ("old", "new")) + "</tr></thead>"
    return f"<div class='scroll'><table class='grid pivot cost'>{head}<tbody>{body}</tbody></table></div>"


# ----------------------------------------------------------------------------- page
EXTRA_CSS = """
:root { --q-ok:#15803d; --q-warn:#b45309; --q-bad:#b91c1c; --q-ok-bg:#e7f5ec; --q-warn-bg:#fdf3e6; --q-bad-bg:#fdecec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --q-ok:#5cd08c; --q-warn:#f2b24c; --q-bad:#f28585; --q-ok-bg:#14291c; --q-warn-bg:#2d2214; --q-bad-bg:#331a1a; } }
:root[data-theme="dark"] { color-scheme: dark;
  --q-ok:#5cd08c; --q-warn:#f2b24c; --q-bad:#f28585; --q-ok-bg:#14291c; --q-warn-bg:#2d2214; --q-bad-bg:#331a1a; }
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
table.sum { min-width:560px; } table.sum th.rowh { width:26%; } table.sum td { font-size:14px; }
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

    page = f"""<title>n12 Bring-up Rerun</title>
<meta name="description" content="n12, 29–30 Sep 2026: tinycal with qwen3.8-27b reran the n10+n11 single-qubit bring-ups on the same 37 qubits of qolab, arbel and gilboa.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{EXTRA_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · n12 overnight run · generated {esc(now)}</div>
<h1 style="margin-top:8px">n12: the n10+n11 bring-ups rerun on the same 37 qubits, with the 29 Sep libraries</h1>
<p class="lede">tinycal with qwen3.8-27b brought up {len(n['done'])} of 37 qubits against {len(o['done'])} the night before,
{len(n['meaningful'])} with meaningful results against {len(o['meaningful'])}: readout median {100 * n['ro_med']:.1f} % instead of
{100 * o['ro_med']:.1f} %, wall median {n['wall_med'] / 60:.0f} min instead of {o['wall_med'] / 60:.0f}, for the same ${n['cost']:.2f}.
The arbel D row and gilboa qD4 now calibrate. Four qubits that worked on n10+n11 broke in new ways and arbel qD1 still fails; all five
were fixed and validated the next night, see {n13_ref()}.</p>
<nav class="toc"><a href="#summary">Summary</a><a href="#libraries">Libraries</a><a href="#fixed">What got fixed</a>
<a href="#failures">New failures</a><a href="#per-qubit">Per qubit</a><a href="#cost">Cost and time</a><a href="#followups">Follow-ups</a></nav>

<h2 id="summary">Summary</h2>
<p class="small muted">Setup: qolab Q1–Q6, all 21 arbel qubits, gilboa qC1–qC5 and qD1–qD5 (the B row's readout line has a broken TWPA).
qwen3.8-27b on OpenRouter (matrix key qwen3-8-27b-openrouter, effort high, max_tokens 16 000), 150 turns, the full single-qubit graph
(80, flux_tunable_1q recipe), one single-target cell per qubit, at most three in flight across all devices. Judge unchanged since n10:
“identity ballpark” on four graded parameters per qubit.</p>
{summary_table(S)}
{legend()}
<p class="small"><b>Straight through</b> = completed; every graph node run once, with no failed or refused run; every value a node
proposed written as proposed (numbers within 10⁻⁴ relative, to allow the agent's retyping), no write to a path no node proposed, and no
proposal left unwritten unless it was a no-op. Read from each cell's events.jsonl, with the same rule as the n13 report. IQCC 503 and
500 answers and read timeouts are not runs, nor is a malformed call with no node name; 60 s job timeouts, pre-flight refusals, rejected
parameters and config errors count as failed runs. The recipe's second power sweep at the sweet spot is not a rerun; counting it as one
gives the same {len(o['st_strict'])} and {len(n['st_strict'])}. Parameters the model passed are not part of the definition (graph presets
never count); the second line is the stricter test. <b>No cell on either night qualifies</b>, and none even ran every node once without a
failure. The nearest n12 cells ({esc(nearest_txt)}) each had {esc(' / '.join(nearest_why))}. The two commonest contributions: in
{n['flux_refused']} of 37 n12 cells the first resonator flux map was refused because the model asked for ±0.6–1.25 V on a ±0.5 V direct
LF-FEM port ({o['flux_refused']} on n10+n11); in {n['quad_unproposed']} ({o['quad_unproposed']}) the agent copied the qubit flux map's
curvature into freq_vs_flux_01_quad_term, which no node had proposed at that point. The ✓ column below has each cell's tally on hover.</p>

<h2 id="libraries">What changed in the libraries since n10</h2>
<p>qua-libs <span class="mono">~/qab-runs/qua-libs-n12</span> = feat/qualibrate-ai <span class="mono">4bf79df</span>; n10 ran on
feat/bringup-fast-presets <span class="mono">08ebca5</span>. In between, in the order they bear on tonight:</p>
<ul class="tight">
<li><b>Drive window</b> <span class="mono small">f5464c5</span>: a spectroscopy window past the upconverter's band moves the LO for that run instead of being refused. <span class="mono small">98edd26</span>, <span class="mono small">1cfb1da</span>: a line not identified as 0→1, or no line, is a failed outcome with no proposal.</li>
<li><b>Drive amplitudes from the chirp</b> <span class="mono small">ce69f5b</span>: an x180 proposal from the measured Rabi rate, and the fine scan driven at 1 MHz of Rabi. Power Rabi in 0.02 steps with no separate x90 sweep <span class="mono small">d90fe34</span>.</li>
<li><b>Readout</b>: the amplitude ceiling comes from the wiring instead of a fixed 0.1 of full scale <span class="mono small">fb200fe</span>; readout power optimization sweeps 0.5–3.5× <span class="mono small">21edcb1</span>; 02b proposes the rerun that can measure an onset it could not <span class="mono small">b5e7d0e</span>; 02a/02b leave out the receiver's notch at IF 0 <span class="mono small">856a8b5</span>.</li>
<li><b>Resonator identification</b> judges every lobe that moved, each in a window of its own <span class="mono small">4bf79df</span>.</li>
<li><b>Active reset</b> through quam's loop with its waits in clock cycles, shared by the single-qubit nodes <span class="mono small">697434f</span>.</li>
<li><b>T1 and RB</b>: T1_coarse dropped from the graph (19 → 18 nodes; T1 on I/Q only when T1_chirp stored none) and T1 sized from the stored value <span class="mono small">2a62517, 5faaee2, 18aebe6</span>; RB withholds a gate fidelity under one decay length <span class="mono small">20397e7</span> and the graph pins its depth at 2048 <span class="mono small">cc0c349</span>.</li>
</ul>
<p>tinycal at <span class="mono">5146a6c</span> plus an uncommitted step-2 recipe edit (readout power at the knee only; committed since as
<span class="mono">dfd25ca</span>). Recipe changes since n10: T1 on I/Q only without a T1_chirp value, commit the x180 the chirp proposes,
no power Rabi for x90. States: qolab and gilboa pulled fresh (gilboa's active_qubit_names patched: the pull omitted qC1 and qC3); arbel
reused the 29 Sep 00:17 pull, because its fresh state used a DrachmaReadoutPulse that quam-builder 0.5.0 cannot load.</p>

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
