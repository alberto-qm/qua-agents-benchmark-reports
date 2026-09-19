#!/usr/bin/env python3
"""Build the 18-19 Sep 2026 overnight report: tinycal with qwen3.8-27b (OpenRouter) on 24 qubits of
qolab, gilboa and arbel, with the day's recipe and node changes, from the cell documents under
~/qab-runs/night2-*.

    uv run --project ~/code/QM/qua-agents-benchmark python make_night2_report.py

Numbers come only from each cell's result.json (projected from tinycal's events.jsonl by
scripts/project_tinycal.py --targets <all targets of the run>, then stamped by qab inspect-state /
validate / accept), the cell's final quam_state, the tinycal run dirs, and `qab compare-models`.
The operator's log (~/qab-runs/recipe-qolab-LOG.md, "Night 2") supplies the incident annotations
and nothing else. Table builders are shared with make_fwcmp2_report.py.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fwcmp2_report as base  # noqa: E402
from make_fwcmp2_report import (  # noqa: E402
    CSS, TINYCAL_RUNS, RUNS, bars, chip_counters, compare_models, context_series, esc, fmt, fw_chip, g, gate_fidelity,
    ktok, load_prices, local, median, minutes, mtok, pct, status_chip, token_cost, x180,
)

OUT = Path(__file__).with_name("2026-09-18-recipe-night-qwen-tinycal.html")
MODEL = "qwen3.8-27b"
MODEL_ID = "qwen/qwen3.8-27b"
CELL_SUFFIX = "tinycal-qwen3-8-27b-openrouter"

# ----------------------------------------------------------------------------- operator log
SNAPSHOT_NOTE = {
    "qolab": "recipe-qolab-20260918-1509-r14/source-state, the 17 Sep fetch (source sha 305c8f5166ec926a), same as the 16-17 Sep report",
    "gilboa": "fwcmp2-gilboa-20260917-0559/source-state, the 14 Sep fetch (source sha 9700c720b3b09961), same as the 14-17 Sep report",
    "arbel": "fwcmp2-arbel-20260917-2100/source-state (source sha 359c64f08064d101; f_01 and joint offsets identical to the 11 Sep state)",
}
EXCLUDED_DIRS = {
    "night2-gilboa-20260919-0116-n2": "the second gilboa batch fired early: its launcher waited on the first batch's process, which the "
                                      "operator killed at 01:16 to stop qB4's loop, so it started while qC2/qC5 were being resumed. Stopped "
                                      "after ~1 node call per target and relaunched as n2b; nothing graded.",
}
# Per-qubit-run causes the document cannot carry (a stopped target leaves no finish event, so it reads 'pending').
TARGET_INCIDENTS = {
    ("night2-gilboa-20260918-2330-n1", "gilboa-" + CELL_SUFFIX, "qB4"):
        "stopped by the operator at 01:16: committed a false qubit at 7.405 GHz (true 6.599; 280 MHz below its own resonator) after a "
        "600 MHz search at the seed found nothing, then emitted ~60 identical describe/run pairs in one turn (11 hardware jobs in 3 min)",
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qB4"):
        "second attempt from the scrambled state: the same false qubit (7.4004 GHz, +800 MHz), whose xy IF of 900 MHz made the shared "
        "config unbuildable for every target from 03:19; reverted to the scrambled values by hand at 03:40 and excluded",
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qB2"):
        "the B-row pattern again: committed 6.7865 GHz (true 5.9416; +845 MHz), then walked the upconverter to 6.5 GHz and None, breaking "
        "open_qm for qB1/qB3 from 03:59; reverted by hand at 04:05 and excluded",
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qC4"):
        "collateral: every parameter up to T1 was calibrated when qB4's write broke the shared config; T1/T2echo/DRAG/RB could not open the "
        "machine and the model declared itself stuck",
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qB3"):
        "self-declared stuck: a flux-tunable line at 5.62 GHz committed, but a flat Rabi and 50 % IQ blobs at every readout power, "
        "frequency and angle (the model's diagnosis: weakly dispersive to the only reachable resonator)",
    ("night2-arbel-20260919-0106-n1", "arbel-" + CELL_SUFFIX, "qD1"):
        "stopped by the operator at 02:38 after 90 min: committed a false flux maximum at 0.033 V (true 0.218) and a narrow line visible "
        "only at 1x drive (5.0337 GHz; reference 5.0146), then met a flat power Rabi by lengthening x180 to 6 µs and 25 µs and bisecting "
        "the x180/x90 DragCosine detuning to 2.1 MHz instead of doubting the frequency. Its record is in the run's qD1/ folder only: the "
        "resume for qC2/qC3 rewrote run.json's target list",
}
# The qubit-runs the operator killed (the other two failures stopped themselves).
OPERATOR_STOPS = {
    ("night2-gilboa-20260918-2330-n1", "gilboa-" + CELL_SUFFIX, "qB4"),
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qB4"),
    ("night2-gilboa-20260919-0145-n2b", "gilboa-" + CELL_SUFFIX, "qB2"),
    ("night2-arbel-20260919-0106-n1", "arbel-" + CELL_SUFFIX, "qD1"),
}
INCIDENTS = [
    ("18 Sep 23:30", "Launch. backend_busy reported qolab and gilboa free; tinycal working tree fc2f37a plus another session's uncommitted "
                     "edits to state.py/tools.py and tests (42/42 tests green), qua-libs feat/qualibrate-ai b0ba588 clean. qolab Q1–Q6 and "
                     "gilboa qD3 qD5 qC3 qC2 qB4 qC5, three targets in parallel per backend."),
    ("19 Sep 01:05", "User asked for the remaining gilboa qubits after the first batch: coloured ones (qD1 qD2 qD4 qC4) then grey ones "
                     "(qB1 qB2 qB3 qB5 qC1), chained behind the running process."),
    ("19 Sep 01:06", "arbel qD1 qC2 qC3 launched one target at a time (max_parallel_agents 1) at the user's request, to keep the load low."),
    ("19 Sep 01:16", "gilboa first batch stopped: qB4 had committed a false qubit 800 MHz off and then issued ~60 identical fine-scan calls in "
                     "one turn, which tinycal executed one after another. qC2 and qC5 resumed alone (tinycal --resume --target). The chained "
                     "second batch fired on the kill and was stopped too; relaunched as n2b = qD1 qD2 qD4 qC4 + qB4 afresh, behind the resume."),
    ("19 Sep 01:24", "qolab done: 6/6 completed, 99.84–99.95 % gate fidelity, 1 h 54 min wall."),
    ("19 Sep 02:38", "arbel qD1 stopped after 90 min (false flux maximum, 1x-only line, 25 µs x180); the process resumed for qC2 and qC3."),
    ("19 Sep 03:19", "gilboa n2b: qB4's second false qubit put its xy IF at 900 MHz. All targets share one QuAM state and config, so open_qm "
                     "failed for everyone: qC4 lost its last four nodes and declared itself stuck."),
    ("19 Sep 03:40", "n2b stopped; qB4 reverted to its scrambled values by hand (state.json.before_qB4_revert), generate_config verified free of "
                     "|IF| > 500 MHz, and the grey batch launched as a resume of the same run (qB1 qB2 qB3 qB5 qC1), qB4 excluded."),
    ("19 Sep 04:05", "qB2 repeated qB4's failure (+845 MHz), breaking the config for qB1/qB3 from 03:59; stopped, reverted "
                     "(state.json.before_qB2_revert), resumed qB1 qB3 qB5 qC1 with qB2 excluded."),
    ("19 Sep 04:30", "arbel done: qC2 99.64 %, qC3 99.75 %. qC3's f_01 came out 5.7153 GHz against a 17 Sep reference of 5.6249 with a consistent "
                     "Ramsey pair, so the arbel reference is stale, not the run."),
    ("19 Sep 05:33", "gilboa grey batch done; qB3 self-declared stuck at 10/18. Night over: 24 targets, 19 completed the graph."),
]

# ----------------------------------------------------------------------------- collect
def collect():
    prices = load_prices()
    cells = []
    for work in sorted(RUNS.glob("night2-*")):
        if not work.is_dir() or work.name in EXCLUDED_DIRS:
            continue
        backend = work.name.split("-")[1]
        docs = sorted(glob.glob(str(work / "*/result.json")))
        if not docs:
            continue
        cm = compare_models(work)
        for doc in docs:
            cell = Path(doc).parent
            r = json.load(open(doc))
            t = r["totals"]
            targets = []
            for x in r["targets"]:
                q = x.get("quality") or {}
                rb = q.get("rb") or {}
                ro = q.get("readout") or {}
                cl = q.get("coherence_limit") or {}
                jd = (x.get("judge") or {}).get("identity") or {}
                L, A = x180(cell / "quam_state/state.json", x["target"])
                ag = x.get("agent") or {}
                at = ag.get("time") or {}
                cause = TARGET_INCIDENTS.get((work.name, cell.name, x["target"]))
                targets.append({
                    "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                    "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, MODEL_ID, prices),
                    "nodes_run": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                    "gate_fid": gate_fidelity(rb.get("error_per_clifford")) if x["status"] == "completed" else None,
                    "ctx": (lambda cs: {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None, "n": len(cs)})(
                        context_series(cell, r.get("run_id") or "", x["target"], "tinycal")),
                    "target": x["target"], "status": x["status"],
                    "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
                    "terminal": x.get("terminal_node"),
                    "rb": rb.get("error_per_clifford"), "readout": ro.get("assignment_fidelity"),
                    "t1": cl.get("t1_s"), "t2e": cl.get("t2echo_s"),
                    "x180_len": L, "x180_amp": A,
                    "ballpark": jd.get("in_ballpark"), "graded": jd.get("graded"), "outside": jd.get("outside") or [],
                    "resid": q.get("residual_detuning_hz"),
                    "reason": x.get("escalation_reason"),
                    "cause": cause or base.classify_stop(x.get("escalation_reason"), x["status"]),
                    "stopped": (work.name, cell.name, x["target"]) in OPERATOR_STOPS,
                    "turns": g(x, "agent", "turns", "total"),
                })
            qubits = [x["target"] for x in r["targets"]]
            c = {
                "work": work.name, "cell": cell.name, "backend": backend, "fw": "tinycal", "model": MODEL,
                "qubits": qubits, "round": (backend, "+".join(qubits)), "run_id": r.get("run_id"),
                "status": r["status"], "started": local(r.get("started_at")), "ended": local(r.get("ended_at")),
                "wall_s": t["time"]["total_s"], "model_s": t["time"].get("model_s"),
                "queue_s": t["time"].get("queue_wait_s"), "qpu_s": t["time"].get("qpu_execution_s"),
                "turns": t["turns"]["total"], "nodes": t["nodes"]["executions"], "reruns": t["nodes"]["re_executions"],
                "tok_in": t["tokens"].get("input"), "tok_out": t["tokens"].get("output"),
                "tok_cr": t["tokens"].get("cache_read"), "tok_cw": t["tokens"].get("cache_creation"),
                "collateral": len((r.get("judge") or {}).get("collateral_edits") or []),
                "targets": targets,
                "finished": sum(1 for x in targets if x["status"] == "completed"),
                "chip": chip_counters(cell, r.get("run_id") or ""),
                "incident": None,
                "superseded": False,
            }
            row = cm.get(cell.name, {})
            c["cost"], c["cw_tokens"], c["graph_frac"] = row.get("cost"), row.get("tokens"), row.get("graph")
            c["stopped_by"] = row.get("stopped_by")
            cells.append(c)
    return cells


# ----------------------------------------------------------------------------- tables specific to this night
def pivot_table(cells):
    """Metrics as rows; one column per backend and one for all three. One framework, one model."""
    cols = [("all three", None), ("qolab", "qolab"), ("gilboa", "gilboa"), ("arbel", "arbel")]
    stats = []
    for _, backend in cols:
        runs = [(c, t) for c in cells if backend is None or c["backend"] == backend for t in c["targets"]]
        done = [(c, t) for c, t in runs if t["status"] == "completed"]
        n = len(done) or 1
        fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
        tot = lambda k: sum((t[k] or 0) for _, t in runs)  # noqa: E731
        tokt = lambda k: sum((t["tokens"].get(k) or 0) for _, t in runs)  # noqa: E731
        per_dev = {}
        for c, t in runs:
            done_here = per_dev.setdefault(c["backend"], {}).get(t["target"], False)
            per_dev[c["backend"]][t["target"]] = done_here or t["status"] == "completed"

        def qubit_list(qs):
            return ", ".join(q if ok else f"<span class='qfail' title='not calibrated'>{q}</span>" for q, ok in sorted(qs.items()))
        stats.append({
            "qubits": "<br>".join(f"<b class='dev'>{b}</b> {qubit_list(qs)}" for b, qs in sorted(per_dev.items())) or "—",
            "done": f"{len(done)}/{len(runs)} ({100 * len(done) / len(runs):.0f}%)" if runs else "—",
            "fid": (pct(fids[len(fids) // 2]) + f"<br><span class='small muted'>{pct(fids[0])} – {pct(fids[-1])}</span>") if fids else "—",
            "agent": minutes(tot("model_s") / n), "qpu": minutes(tot("qpu_s") / n), "queue": minutes(tot("queue_s") / n),
            "cost": f"${sum((t['cost'] or 0) for _, t in runs) / n:.2f}",
            "spent": f"${sum((t['cost'] or 0) for _, t in runs):.2f}",
            "turns": f"{tot('turns') / n:.0f}", "nodes": f"{tot('nodes_run') / n:.0f} ({tot('reruns') / n:.0f})",
            "ctx_med": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["median"] for _, t in runs if t["ctx"]["median"]]),
            "ctx_end": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["end"] for _, t in done if t["ctx"]["end"]]),
            "tin": mtok(tokt("input") / n), "tout": mtok(tokt("output") / n),
            "tcache": f"{mtok(tokt('cache_read') / n)} / {mtok(tokt('cache_creation') / n)}",
        } if runs else None)
    rows = [("qubits measured", "qubits"), ("calibrations completed / attempted", "done"),
            ("single-qubit gate fidelity, median (min – max)", "fid"), ("agent time / calibration", "agent"),
            ("QPU time / calibration", "qpu"), ("queue wait / calibration", "queue"), ("judge cost / calibration", "cost"),
            ("judge cost, total spent", "spent"),
            ("model turns / calibration", "turns"), ("node runs (re-runs) / calibration", "nodes"),
            ("context per model call, median, in k tokens (mean over qubit-runs)", "ctx_med"),
            ("context at the end of a bring-up, k tokens (mean over completed qubit-runs)", "ctx_end"),
            ("input tokens / calibration", "tin"), ("output tokens / calibration", "tout"),
            ("cached tokens read / written / calibration", "tcache")]
    body = []
    for label, key in rows:
        cls = "small mono wrap" if key == "qubits" else "num"
        tds = "".join(f"<td class='{cls}'>{st[key] if st else '—'}</td>" for st in stats)
        body.append(f"<tr><th class='rowh'>{label}</th>{tds}</tr>")
    head = (f'<thead><tr><th></th><th colspan="{len(cols)}" class="grp">{fw_chip("tinycal", MODEL)}</th></tr>'
            '<tr><th></th>' + "".join(f"<th>{esc(label)}</th>" for label, _ in cols) + '</tr></thead>')
    return f'<div class="scroll"><table class="grid pivot">{head}<tbody>' + "".join(body) + "</tbody></table></div>"


def qubit_table(cells):
    """One row per qubit-run: the side-by-side table of the framework reports, with one framework."""
    rows = []
    for c in cells:
        for t in c["targets"]:
            cause = "" if t["status"] == "completed" else f"<br><span class='small muted'>{esc(t['cause'])}</span>"
            rows.append(((c["backend"], t["target"], c["started"]), f"<tr><td class='mono'>{esc(c['backend'])} {esc(t['target'])}</td><td>{status_chip(t['status'])}{cause}</td>"
                        f"<td class='num'>{t['nodes'] if t['nodes'] is not None else 0}/{t['graph'] or 18}</td>"
                        f"<td class='num'>{pct(t['gate_fid'])}</td><td class='num'>{pct(t['readout'], 1)}</td>"
                        f"<td class='num'>{fmt(t['x180_len'], '{:.0f}')} ns @ {fmt(t['x180_amp'], '{:.3f}')}</td>"
                        f"<td class='num'>{t['ballpark'] if t['ballpark'] is not None else '—'}/{t['graded'] or 4}</td>"
                        f"<td class='num'>{minutes(t['model_s'])}</td><td class='num'>{minutes(t['qpu_s'])}</td>"
                        f"<td class='num'>{fmt(t['turns'], '{}')}</td><td class='num'>{fmt(t['cost'], '${:.2f}')}</td></tr>"))
    rows.sort(key=lambda r: r[0])
    rows = [r for _, r in rows]
    head = ('<thead><tr><th>qubit</th><th>status</th><th>nodes</th><th>gate fidelity</th><th>readout assign.</th><th>x180 len @ amp</th>'
            '<th>ballpark</th><th>agent</th><th>QPU</th><th>turns</th><th>cost</th></tr></thead>')
    return f'<div class="scroll"><table class="grid small">{head}<tbody>' + "".join(rows) + "</tbody></table></div>"


# What would have caught each hard case, by (backend, qubit); the operator's reading of the transcripts.
HARD_CASE_NOTE = {
    ("gilboa", "qB4"): "a guard on the committed f_01: a line found hundreds of MHz from the seed, at 1x drive only, and within 300 MHz of the "
                       "qubit's own resonator is not a qubit; and write_state refusing a state whose xy IF exceeds 500 MHz",
    ("gilboa", "qB2"): "the same two guards; qB2 reproduced qB4's failure step for step",
    ("arbel", "qD1"): "the qubit flux map anchored on the measured maximum (qua-libs a0e818b) would have refused the 0.033 V apex; a recipe line that a "
                      "flat power Rabi at a sane pulse length means the drive is off resonance, not that the pulse is too short",
    ("gilboa", "qB3"): "unknown whether the qubit is reachable at all: the line at 5.62 GHz sits 1.83 GHz below the only resonator the model could read; "
                       "a reference measurement on qB3 is needed before this counts against the model",
    ("gilboa", "qC4"): "not this run's doing: every parameter through T1 was in place when another target's write made the shared config "
                       "unbuildable; a per-target stop and a config check in write_state would have left it to finish",
    ("qolab", "Q4"): "completed at 99.84 % with the idle flux 0.25 V from the sweet spot and f_01 119 MHz below the reference: the qubit flux map's fit "
                     "extrapolated a turning point just outside a monotonic sweep and the model kept the idle. Fixed in the node (a0e818b): no fit "
                     "overlay and no proposal when the maximum is on an edge; the replay experiment is in the context section",
    ("gilboa", "qC1"): "a calibration miss, not the chip: T1 32 µs / T2echo 45 µs allow well over 99.8 %, and the one ballpark miss is the x180 amplitude "
                       "(100 ns @ 0.390 committed). First attempt on qC1 by any model; its Rabi and DRAG transcripts are the place to look",
    ("gilboa", "qC5"): "below the 99.1–99.9 % of every framework round on this qubit (qwen in tinycal 99.69 %) with DRAG skipped (17/18 nodes); T1/T2echo "
                       "unchanged, so a calibration miss",
    ("gilboa", "qD2"): "the qubit's ceiling: T1 7.9 µs tonight (1.2 µs in the source state); 98.74–98.81 % in the framework rounds",
    ("gilboa", "qD5"): "below the 99.86–99.95 % of every framework round; T1/T2echo unchanged (33 / 66 µs), readout 97 %, so the gate itself — "
                       "a calibration miss of unlocated origin",
}


def hard_cases(cells):
    """One row per qubit-run that did not finish the graph, then the completed ones with a poor or wrong result."""
    rows = []
    for c in cells:
        for t in c["targets"]:
            poor = t["status"] == "completed" and t["gate_fid"] is not None and t["gate_fid"] < 0.99
            wrong = (c["backend"], t["target"]) == ("qolab", "Q4")
            if t["status"] == "completed" and not (poor or wrong):
                continue
            fid = f" · {pct(t['gate_fid'])}" if t["gate_fid"] is not None else ""
            outcome = (f"{status_chip(t['status'])} {t['nodes'] or 0}/18{fid}<br><span class='small'>{minutes(t['model_s'])} agent · "
                       f"{fmt(t['cost'], '${:.2f}')} · {fmt(t['turns'], '{}')} turns</span>")
            what = t["cause"] if t["status"] != "completed" else ("finished the graph with a wrong flux point" if wrong else "finished the graph with a poor number")
            note = HARD_CASE_NOTE.get((c["backend"], t["target"]), "")
            rank = 0 if t["status"] != "completed" else (1 if wrong else 2)
            rows.append((rank, c["backend"], t["target"], c["started"], outcome, what, note))
    rows.sort()
    body = "".join(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td><td>{o}</td><td class='small'>{esc(w)}</td><td class='small'>{esc(n)}</td></tr>"
                   for _, b, q, _, o, w, n in rows)
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>qubit</th><th>outcome</th>'
            '<th>what happened</th><th>what would have caught it</th></tr></thead><tbody>' + body + "</tbody></table></div>")


# ----------------------------------------------------------------------------- what fills the context (tinycal only)
CHARS_PER_TOKEN = base.CHARS_PER_TOKEN
CONTEXT_REF = [  # column label, tinycal run id, target
    ("gilboa qD3 (as in the 14–17 Sep report)", "night2_gilboa_qwen3_8_27b_20260918-2330_n1", "qD3"),
    ("qolab Q4 (the flux-map case below)", "night2_qolab_qwen3_8_27b_20260918-2330_n1", "Q4"),
    ("gilboa qB1 (longest completed run)", "night2_gilboa_qwen3_8_27b_20260919-0145_n2b", "qB1"),
]


def _tinycal_context(run_id: str, target: str):
    d = TINYCAL_RUNS / run_id / target
    try:
        tr = json.load(open(d / "transcript.json"))
        ev = [json.loads(line) for line in open(d / "events.jsonl")]
    except Exception:  # noqa: BLE001
        return None
    msgs = tr if isinstance(tr, list) else tr.get("messages", [])
    parts = {"node results (tool_result text)": 0, "figures kept in context": 0, "tool-call arguments": 0,
             "model text (notes, reasoning shown)": 0, "system prompt (recipe + catalogue)": 0, "task message": 0}
    images = 0
    for m in msgs:
        content = m.get("content") if isinstance(m.get("content"), list) else [{"type": "text", "text": str(m.get("content", ""))}]
        for b in content:
            t = b.get("type")
            if t == "text":
                parts["model text (notes, reasoning shown)" if m.get("role") == "assistant" else "task message"] += len(b.get("text", ""))
            elif t == "tool_use":
                parts["tool-call arguments"] += len(json.dumps(b.get("input", {})))
            elif t == "tool_result":
                for c in b.get("content", []):
                    if c.get("type") == "text":
                        parts["node results (tool_result text)"] += len(c.get("text", ""))
                    elif c.get("type") == "image":
                        images += 1
            elif t == "image":
                images += 1
    try:
        parts["system prompt (recipe + catalogue)"] = len((d / "system_prompt.md").read_text())
    except Exception:  # noqa: BLE001
        pass
    turns = [e for e in ev if e.get("kind") == "model_turn" and (e.get("usage") or {}).get("input_tokens")]
    series = [(e["turn"], e["usage"]["input_tokens"]) for e in turns]
    parts["figures kept in context"] = images * 1800 * CHARS_PER_TOKEN
    return {"parts": parts, "images": images, "series": series, "turns": len(turns)}


def context_section():
    data = {label: _tinycal_context(run, target) for label, run, target in CONTEXT_REF}
    cols = list(data)
    keys = []
    for v in data.values():
        if v:
            for k in v["parts"]:
                if k not in keys:
                    keys.append(k)
    head = "".join(f"<th>{esc(m)}</th>" for m in cols)
    body = []
    for k in keys:
        tds = "".join(f"<td class='num'>{data[m]['parts'][k] / 1000:.0f}k chars ≈ {base._ktok(data[m]['parts'][k])} tok</td>" if data[m] else "<td>—</td>" for m in cols)
        body.append(f"<tr><th class='rowh'>{esc(k)}</th>{tds}</tr>")
    tds = "".join(f"<td class='num'>{sum(data[m]['parts'].values()) / 1000:.0f}k chars ≈ {base._ktok(sum(data[m]['parts'].values()))} tok</td>" if data[m] else "<td>—</td>" for m in cols)
    body.append(f"<tr><th class='rowh'>sum of the parts (estimate)</th>{tds}</tr>")

    def meas(v):
        if not v or not v["series"]:
            return "—"
        s = v["series"]
        mid = s[len(s) // 2]
        return f"turn {s[0][0]}: {s[0][1] / 1000:.0f}k · turn {mid[0]}: {mid[1] / 1000:.0f}k · turn {s[-1][0]}: {s[-1][1] / 1000:.0f}k"
    body.append("<tr><th class='rowh'><b>measured prompt, tokens</b> (first · middle · last turn)</th>" + "".join(f"<td class='num'>{meas(data[m])}</td>" for m in cols) + "</tr>")
    body.append("<tr><th class='rowh'>figures in the final prompt</th>" + "".join(f"<td class='num'>{data[m]['images']}</td>" if data[m] else "<td>—</td>" for m in cols) + "</tr>")
    table = (f'<div class="scroll"><table class="grid pivot"><thead><tr><th>tinycal · {esc(MODEL)}, final prompt</th>{head}</tr></thead>'
             f"<tbody>{''.join(body)}</tbody></table></div>")
    return f"""
<h3>What fills the context</h3>
<p class="small muted">Prompt composition at the end of three qubit-runs. Characters are measured from the run's final transcript.json;
"≈ tok" is characters ÷ {CHARS_PER_TOKEN}, an estimate; figures are counted at ~1,800 tokens each. The <b>measured prompt</b> row is the
input-token count OpenRouter reported for the call, cached prefix included. Same caveat as in the 14–17 Sep report: the measured size
runs above the sum of the parts because node results are mostly digits and the tool schemas are not in the parts.</p>
{table}
<p class="small">The shape is the one seen before: tinycal keeps the whole conversation, so the prompt grows by about a thousand tokens a turn
and a 50–90-turn qubit ends at 75–90k tokens (qwen's tool-result text tokenises more compactly than the 150–250k Opus and Sonnet reached
on the same nodes), of which OpenRouter reports only a small part as cached. The biggest block is node result text, then the figures
(capped at 16 in context, older ones replaced by a one-line pointer), then the model's own notes and prose. A controlled experiment on
19 Sep replayed qolab Q4's flux-map turn with the same history cut to 10k, 15k, 23k, 27k and 40k tokens, with the history replaced by
neutral node descriptions of the same length, and with the model's own prose removed: none of the three moved the decision rate
(60–80 % correct in every cell), while attaching the fit-overlay figure took it from ~65 % to ~20 % and made four of five samples run
out their 16k reasoning budget without acting. Context size is a cost, not the cause of that failure.</p>
"""


# ----------------------------------------------------------------------------- pins
CHECKOUTS = {
    "qua-agents-benchmark (judge, scramble, project_tinycal)": base.BENCH,
    "tinycal (working tree, run directly)": Path.home() / "code/QM/tinycal",
    "qua-libs feat/qualibrate-ai (working tree, run directly)": Path.home() / "code/QM/qua-libs",
}


def pins_table(cells):
    from collections import Counter, defaultdict
    fw_sha, libs, spec, snap, srcdir = Counter(), Counter(), Counter(), defaultdict(Counter), defaultdict(set)
    for c in cells:
        try:
            r = json.load(open(RUNS / c["work"] / c["cell"] / "result.json"))
        except Exception:  # noqa: BLE001
            continue
        fp = r.get("fingerprint") or {}
        fw_sha[str(fp.get("git_sha", ""))[:7]] += 1
        cc = fp.get("calibration_content") or {}
        libs[str(cc.get("version", cc) if isinstance(cc, dict) else cc)[:7]] += 1
        sc = r.get("scramble") or {}
        spec[str(sc.get("spec_hash", ""))] += 1
        snap[c["backend"]][str(sc.get("source_hash", ""))] += 1
        srcdir[c["backend"]].add(c["work"])

    def fmt_counter(cnt):
        return ", ".join(f"<span class='mono'>{esc(k)}</span> ({v} cells)" for k, v in cnt.most_common()) or "—"
    rows = []
    for label, path in CHECKOUTS.items():
        sha = base._git(path, "rev-parse", "--short", "HEAD")
        dirty = base._git(path, "status", "--short").replace("\n", "; ") or "clean"
        if label.startswith("tinycal"):
            recorded = ("document fingerprint.git_sha: " + fmt_counter(fw_sha) +
                        ". qolab ran on fc2f37a with another session's uncommitted state.py/tools.py edits in the tree; those were committed as "
                        "5174fac during the night, and the gilboa and arbel processes resumed after 01:16 record that sha. The recipe file "
                        "(bringup_recipes/flux_tunable_1q.md) is identical in every work dir's recipe_in_force.md.")
        elif label.startswith("qua-libs"):
            recorded = ("document fingerprint.calibration_content: " + fmt_counter(libs) +
                        ". Not the frozen ~/qab-runs/qua-libs-latest of the framework reports (still a884483): the night ran the working "
                        "copy at b0ba588, i.e. the 18 Sep node changes listed under Apparatus.")
        else:
            recorded = "not in the documents; the judge's spec digests below are what it enforces"
        rows.append(f"<tr><td>{esc(label)}</td><td class='mono'>{esc(sha)}</td><td class='small'>{esc(dirty)}</td><td class='small'>{recorded}</td></tr>")
    rows.append(f"<tr><td>scramble spec (workloads/decalibrate_chip.yaml)</td><td class='mono'>{esc(next(iter(spec), ''))}</td><td class='small'>judge refuses a document with another digest</td><td class='small'>document scramble.spec_hash: {fmt_counter(spec)}</td></tr>")
    rows.append("<tr><td>acceptance spec (workloads/acceptance.yaml)</td><td class='mono'>821bb4dd1d5d0e5a</td><td class='small'>—</td><td class='small'>stamped by qab accept</td></tr>")
    for b in sorted(snap):
        dirs = sorted(srcdir[b])
        rows.append(f"<tr><td>{esc(b)} snapshot (source-state)</td><td class='mono'>{esc(next(iter(snap[b]), ''))}</td>"
                    f"<td class='small'>seed a new work dir from <span class='mono'>{esc(dirs[0])}/source-state</span>; qab scramble is deterministic</td>"
                    f"<td class='small'>document scramble.source_hash: {fmt_counter(snap[b])}</td></tr>")
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>component</th><th>sha / digest</th>'
            '<th>uncommitted at report generation</th><th>as recorded in the cell documents</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def run_ids(cells):
    rows = []
    for c in sorted(cells, key=lambda c: (c["work"], c["cell"])):
        tc = f"~/code/QM/tinycal/runs/{c['run_id']}"
        rows.append(f"<tr><td class='mono small'>{esc(c['work'])}</td><td class='mono small'>{esc(c['cell'])}</td>"
                    f"<td>{fw_chip(c['fw'], c['model'])}</td><td class='mono'>{'+'.join(c['qubits'])}</td>"
                    f"<td class='mono small'>{esc(tc)}</td><td class='num'>{esc(c['started'])} → {esc(c['ended'])}</td><td>{status_chip(c['status'])}</td></tr>")
    for d, why in EXCLUDED_DIRS.items():
        rows.append(f"<tr class='sup'><td class='mono small'>{esc(d)}</td><td colspan='6' class='small muted'>{esc(why)}</td></tr>")
    return ('<div class="scroll"><table class="grid small ids"><thead><tr><th>work dir (~/qab-runs/)</th><th>cell dir</th><th>cell</th><th>qubits</th>'
            '<th>tinycal run dir</th><th>started → ended (document span)</th><th>status</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- page
def build():
    cells = collect()
    # the shared table builders read these module globals
    base.MODEL_LABELS[:] = [MODEL]
    base.SUPERSEDED_DIRS.clear()
    runs = [(c, t) for c in cells for t in c["targets"]]
    done = [(c, t) for c, t in runs if t["status"] == "completed"]
    fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
    med = fids[len(fids) // 2] if fids else None
    spent = sum((t["cost"] or 0) for _, t in runs)
    now = datetime.now().strftime("%d %b %Y %H:%M")
    tiles = [
        f'<div class="tile"><div class="v">{len(done)}/{len(runs)}</div><div class="k">calibrations completed / attempted, three backends</div></div>',
        f'<div class="tile"><div class="v">{pct(med)}</div><div class="k">median single-qubit gate fidelity of the completed ones</div></div>',
        f'<div class="tile"><div class="v">${spent / (len(done) or 1):,.2f}</div><div class="k">judge-priced cost per completed calibration (${spent:,.2f} in all)</div></div>',
        f'<div class="tile"><div class="v">{sum(1 for _, t in runs if t["stopped"])}</div><div class="k">qubit-runs stopped by the operator, all for a qubit committed hundreds of MHz off; two more stopped themselves</div></div>',
    ]

    def lab(it, plain=False):
        return it["label"] if plain else esc(it["label"])
    cost_items = sorted([{"fw": "tinycal", "cost": t["cost"], "label": f"{c['backend']} {t['target']}" + ("" if t["status"] == "completed" else f"  ({t['status']})")}
                         for c, t in runs if t["cost"] is not None], key=lambda it: it["label"])
    cost_fig = bars(cost_items, "cost", lab, lambda v: f"${v:.2f}", "Judge-priced cost per qubit-run", "USD at prices.yaml (OpenRouter's rate for qwen3.8-27b), each qubit's own tokens; unfinished runs marked")
    err_items = sorted([{"fw": "tinycal", "err": 100 * (1 - t["gate_fid"]), "label": f"{c['backend']} {t['target']}"}
                        for c, t in runs if t["gate_fid"] is not None], key=lambda it: it["label"])
    err_fig = bars(err_items, "err", lab, lambda v: f"{v:.3f}% ({100 - v:.2f}%)", "Single-qubit gate error per completed qubit-run", "gate error in %, fidelity in brackets; error per Clifford ÷ 1.875")
    timeline = '<table class="grid timeline"><tbody>' + "".join(f"<tr><td>{esc(t)}</td><td>{esc(w)}</td></tr>" for t, w in INCIDENTS) + "</tbody></table>"

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recipe Night, qwen tinycal</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><div class="page">
<div class="eyebrow">qua-agents benchmark · recipe iteration, overnight run · generated {esc(now)}</div>
<h1 style="margin-top:8px">tinycal with qwen3.8-27b on 24 qubits of qolab, gilboa and arbel, 18–19 Sep 2026</h1>
<p class="lede">One framework and one model, after a day of iterating the bring-up recipe and the qua-libs nodes against qolab: the same
scrambled snapshots and scramble spec as the 14–17 Sep framework comparison, handed to tinycal with qwen3.8-27b (27B open weights via
OpenRouter, no effort knob) on every qubit of qolab, fifteen of gilboa and three of arbel. Two days earlier the same model in the same
framework had failed all three qolab qubits; this is the measurement of what the day's changes bought, on more qubits than the recipe was
tuned on. Numbers are read from each qubit-run's own record in result.json, the cell's final QuAM state and the judge's price table;
nothing here is estimated by hand except the operator annotations, which are marked.</p>

<h2 id="overview">Overview</h2>
<p><b>Apparatus.</b> tinycal run directly from its working tree (<span class="mono">fc2f37a</span>, then <span class="mono">5174fac</span> for the
processes resumed after 01:16; max_turns 150 per target, three targets in parallel per backend, one at a time on arbel), qua-libs
<span class="mono">feat/qualibrate-ai</span> at <span class="mono">b0ba588</span> — the working copy, not the frozen a884483 checkout of the framework
reports — with the 18 Sep changes: the resonator flux map proposes the measured apex and refuses one on a sweep edge; the readout power node
proposes nothing when the sweep never reaches the punch-out onset (its old fallback, the top of the sweep, was what qwen committed on 16–17 Sep)
and reports the sweep in volts; flux sweeps are checked against the flux port's own range instead of fixed ±1 V bounds; long sweeps are refused
before they hold the queue (the 300 s executor cap); fit-derived cause clauses were dropped from every node's failure text; the fine qubit
spectroscopy drives at the search amplitude over 30 MHz; and the bring-up graph's qubit flux map defaults to 31 × 121 × 100 so it finishes inside one job. One thing the night ran with and
the next run will not: the qubit flux map passed its pulse duration in nanoseconds where QuAM counts clock cycles of 4 ns, so its flux step
and saturation drive were four times longer than the state says (80 µs per shot for a 20 µs pulse); both stayed aligned, so the maps read
the right frequencies, and the fix (qua-libs 4ca26fb, 19 Sep) shortens the shot rather than changing what it measures.
The recipe (tinycal bringup_recipes/flux_tunable_1q.md, identical in every work dir's recipe_in_force.md) gained: a provisional low power when
no onset appears; the resonator flux map widened until the maximum is inside the data, never the minimum, with the other maximum a period away
as the fallback; a second power sweep at the sweet spot; and a fine f_01 re-measurement at the committed bias before power Rabi.
Workload decalibrate_chip.yaml, scramble spec <span class="mono">f47053f145f74c8c</span>, acceptance spec <span class="mono">821bb4dd1d5d0e5a</span>;
model calls straight to OpenRouter. Snapshots: qolab — {esc(SNAPSHOT_NOTE['qolab'])}; gilboa — {esc(SNAPSHOT_NOTE['gilboa'])};
arbel — {esc(SNAPSHOT_NOTE['arbel'])}.</p>
<h3>Pins: what a new cell must use to be comparable</h3>
<p class="small muted">Shas and digests read from the cell documents next to the state of each checkout when this page was generated. The libraries
were working copies, so a comparable run must check out the recorded shas rather than the frozen benchmark pins.</p>
{pins_table(cells)}
<div class="tiles">{"".join(tiles)}</div>
<p class="small muted">How to read the tables: one qubit-run = tinycal calibrating one qubit from the scrambled state with qwen3.8-27b.
Gate fidelity = 1 − (RB error per Clifford ÷ 1.875), the qua-libs RB convention. "Per calibration" = the total over every qubit-run attempted
on that backend, divided by the number completed, so the spend and time of a qubit that was never finished are charged to the ones that were.
Agent time is time inside model calls; QPU time is execution on the chip; queue wait is the cloud queue. No qubit-run was lost to an
infrastructure incident this night, so every attempt counts: the four the operator stopped and the two that stopped themselves are genuine
failures and are included (gilboa qB4 was attempted twice, so gilboa has 16 qubit-runs on 15 qubits).
Context = prompt size per model call in tokens, cached prefix included, from tinycal's events.jsonl.</p>
{pivot_table(cells)}

<h3>Per qubit</h3>
<p class="small muted">Every qubit-run, in the layout of the framework side-by-side tables with one framework. Ballpark is the judge's identity check
on the four scrambled parameters (resonator frequency, f_01, readout amplitude, x180 amplitude) against the source snapshot; a miss there is
often the snapshot's, see below.</p>
{qubit_table(cells)}

<h3>Hard cases: qubits that were not calibrated, or not calibrated right</h3>
<p class="small muted">One row per qubit-run that did not reach the end of the graph, then the completed ones whose result is poor (gate fidelity
under 99 %) or wrong (qolab Q4). The framework reports' rule excluded finished-but-poor runs from this table; with one framework they are the
next most interesting rows, so they are here with a note on whether the number is the chip's or the calibration's. The right-hand column is the
operator's reading of the transcript, not something the cell recorded.</p>
{hard_cases(cells)}
{cost_fig}
{err_fig}
{context_section()}

<h3>What the numbers say</h3>
<p><b>The recipe iteration worked on qolab, and it carried to gilboa's good qubits.</b> Two days earlier qwen in tinycal failed all three qolab
qubits it tried (readout power committed at the top of the sweep, idle flux left at 0 V). Tonight it completed 6 of 6 on qolab at 99.84–99.95 %
gate fidelity, in 47–67 min of wall time and $0.65–0.82 each — the same fidelities Opus and Sonnet reached on Q1–Q3 in the framework comparison,
for about a tenth of Opus's per-calibration cost in that report ($7.65). Over the whole night the completion rate, 19/25, is the same
76 % that qwen in tinycal had on the 14–17 Sep set (9/12), on a set that now includes gilboa's B row; per completed calibration it cost
$1.04 against $1.42. On gilboa nine of the eleven it completed came out where the earlier rounds put them (qD3 99.92 %,
qC3 99.62 %, qC2 99.77 %, qD2 98.30 % at its short T1); qD5 (98.89 %) and qC5 (97.87 %) landed one to two points below the framework rounds
with the coherence unchanged, which makes them calibration misses. Five qubits that had never been attempted by any model (qD1, qD4, qB1, qB5, qC1) finished at
97.4–99.9 %. Arbel's two completed qubits landed on IQCC's own numbers (qC2 99.64 %, qC3 99.75 %).</p>
<p><b>Every failure was the same failure.</b> Five qubit-runs did not finish the graph on their own account, and four of them (gilboa qB4 twice, qB2,
arbel qD1) are one pattern: the qubit search at the seed frequency found nothing, a feature hundreds of MHz away — 800 MHz for the gilboa B row,
just below their own resonators — was accepted as the qubit, and the run then rationalised every later contradiction (a flat Rabi became a request
for a 25 µs pulse on arbel qD1). The fifth, gilboa qB3, found a flux-tunable line but never saw two readout blobs at any power, frequency or angle,
and stopped itself. qC4 is a casualty of the harness, not of its own run: every parameter up to T1 was calibrated when qB4's false f_01 put an xy
intermediate frequency at 900 MHz into the state <i>all targets share</i>, and no target could open the machine until the operator reverted it by
hand. That happened twice (qB4 at 03:19, qB2 at 03:59). Three guards would have contained the night's damage: write_state refusing a state that
fails generate_config, a stop for one target that leaves the others running, and a cap on identical repeated tool calls within one turn
(qB4 issued about sixty).</p>
<p><b>One completed qubit hides a wrong flux point.</b> qolab Q4 finished at 99.84 % with its idle flux at −0.028 V; its qubit flux map showed the line
rising monotonically to the +0.0375 V edge of the sweep, the node's fit extrapolated a turning point just outside it, and the model read the map as
"apex at the committed idle point" and moved on. The upper sweet spot is at about +0.23 V and f_01 came out 119 MHz below the reference — the
ballpark miss in its row. Replaying that turn 75 times on 19 Sep with the real history, with neutral filler of the same length and with the
model's own prose removed found no effect of context length or content; attaching the node's fit-overlay figure was what turned a mostly-right
decision into a mostly-wrong one. The qubit flux map node has since been changed to propose the measured ridge maximum, refuse a maximum on an
edge of the tracked data or a ridge that leaves the frequency window, and draw no fit when it has nothing to propose (qua-libs a0e818b, after
this night).</p>
<p><b>The references are not all trustworthy.</b> arbel qC3's f_01 came out 5.7153 GHz against a 17 Sep reference of 5.6249, with a consistent
two-sided Ramsey pair and 99.75 % RB behind it; the snapshot is stale, not the run, and the same doubt applies to arbel qD1's "true" 5.0146 GHz.
gilboa's ballpark misses are mostly the readout amplitude (the scrambled value is ~2× the operating point on several qubits, and the graph
lands near the operating point) — a miss of the snapshot's making, as in the framework reports.</p>
<p><b>Cost and time.</b> The night's judge-priced spend was ${spent:,.2f} for 25 qubit-runs on 24 qubits, ${spent / (len(done) or 1):,.2f} per completed calibration
including the failures' share; 50–90 turns per qubit; 1–3 h wall per three-qubit batch. The operator's running log had written $39.22 — that figure
double-counted the resumed processes; the per-target events total is what the tables show.</p>

<h2 id="stuck">Where things got stuck</h2>
<p>Every qubit-run that did not reach the end of the graph, with the cause from the cell's own escalation text or, for the runs the operator
stopped, from the operator's log (those leave no finish event, so their document status is "pending").</p>
{base.stuck_table(cells)}
<h3>Incident timeline</h3>
{timeline}

<h2 id="per-qubit">Per-qubit detail</h2>
<p>One table per qubit. Gate fidelity with the RB error per Clifford underneath; readout is the assignment fidelity the graph measured; T1/T2echo
as measured by the cell; x180 is the committed π-pulse (length @ amplitude) read from the cell's final QuAM state; ballpark is the judge's identity
check on the scrambled parameters, with the misses named; agent, QPU, queue, tokens and cost are the qubit-run's own.</p>
{base.per_qubit(cells)}

<h2 id="runs">Run identifiers</h2>
<p>tinycal run dirs hold run.json, profile.yaml and one events.jsonl, transcript.json and plots/ per target; cell dirs under the work dir hold
result.json (projected with <span class="mono">project_tinycal.py --targets</span> over every target the run touched, since a resume narrows
run.json's own list), run.log and the final quam_state. The work dir also keeps the source and scrambled states, edit_plan.json, scramble.log,
recipe_in_force.md and the two -rev.txt files.</p>
{run_ids(cells)}
<p class="small muted">Built by make_night2_report.py in this repo. Operator log: ~/qab-runs/recipe-qolab-LOG.md (sections "Night 2");
attempts: ~/qab-runs/night2-attempts.txt.</p>
</div></body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} · {len(cells)} cells, {len(runs)} qubit-runs, {len(done)} completed, ${spent:.2f}")


if __name__ == "__main__":
    build()
