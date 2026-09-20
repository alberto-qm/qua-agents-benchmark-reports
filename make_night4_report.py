#!/usr/bin/env python3
"""Build the 20-21 Sep 2026 report: tinycal with qwen3.8-27b served by Splash (self-hosted) and by OpenRouter,
on the same scrambled snapshots, on gilboa's C/D rows, all of qolab and three arbel qubits, from the cell
documents under ~/qab-runs/night4-*.

    uv run --project ~/code/QM/qua-agents-benchmark python make_night4_report.py

Numbers come only from each cell's result.json (projected from tinycal's events.jsonl by
scripts/project_tinycal.py, then stamped by qab inspect-state / validate / accept; for a cell still running,
~/qab-runs/night4-refresh.sh writes an interim document the same way), the cell's final quam_state, the
tinycal run dirs and `qab compare-models`. The operator's log (~/qab-runs/recipe-qolab-LOG.md, "Night 4")
supplies the incident annotations and nothing else. Table builders are shared with make_fwcmp2_report.py
and make_night2_report.py.
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fwcmp2_report as base  # noqa: E402
import make_night2_report as night2  # noqa: E402
from make_fwcmp2_report import (  # noqa: E402
    CSS, TINYCAL_RUNS, RUNS, bars, chip_counters, compare_models, context_series, esc, fmt, g, gate_fidelity,
    ktok, load_prices, local, median, minutes, mtok, pct, status_chip, token_cost, x180,
)

OUT = Path(__file__).with_name("2026-09-20-splash-vs-openrouter-qwen-tinycal.html")
ARTIFACT_DIR = __import__("os").environ.get("REPORT_ARTIFACT_DIR")  # also write the page as an Artifact body here
PROVIDERS = {  # cell-dir token -> (label, document model_id for pricing, colour key for the bar figures)
    "openrouter": ("qwen3.8-27b · OpenRouter", "qwen/qwen3.8-27b", "tinycal"),
    "splash": ("qwen3.8-27b · Splash", "incoai/Qwen3.8-27B-Splash", "splash"),
}
base.FW_COLOR["splash"] = "var(--qua)"  # blue for Splash; OpenRouter keeps tinycal's green

# ----------------------------------------------------------------------------- operator log
SNAPSHOT_NOTE = {
    "gilboa": "fresh pull at 16:00 on 20 Sep, with two local patches before scrambling: qC5's y90 amplitude made a reference to x90 "
              "(a state bug IQCC confirmed the same day) and active_qubit_names set to the ten C/D qubits (the cloud state listed only "
              "qC2, which parks every other z line at 0 V — see the timeline)",
    "qolab": "fresh pull at 16:00 on 20 Sep, nothing to patch",
    "arbel": "the 19 Sep 01:06 pull (night2-arbel-20260919-0106-n1), because the 20 Sep cloud state stores every resonator's readout "
             "pulse as a reference (#./readout_square) that the scrambler refuses; wiring identical, f_01 within 0.4 MHz, x180 unchanged, "
             "readout amplitudes 17–33 % apart (inside the ballpark window). qC5's y90 patched as on gilboa",
}
EXCLUDED_DIRS = {
    "night4-gilboa-20260920-1600.flux-at-0V": "the first gilboa cells (both providers, 16:00–17:10): the pulled state's active_qubit_names "
                                              "held only qC2, so every other target ran with its z line at 0 V — absolute flux maps, committed "
                                              "offsets never applied, RB off the sweet spot. Stopped once the cause was found; kept as the "
                                              "record of that failure (OpenRouter qD3 99.92 %, qD5 97.72 % measured at 0 V), not counted.",
    "night4-arbel-20260920-1633.refused-scramble": "the fresh arbel pull: the scrambler refused it (readout stored as a QuAM reference); "
                                                   "relaunched a minute later from the 19 Sep snapshot.",
}
TARGET_INCIDENTS = {}
OPERATOR_STOPS = set()
# Completed runs whose RB number is not a measurement (the fit's own warning says so); their fidelity is withheld from every table.
INVALID_RB = {
    ("gilboa", "openrouter", "qD1"): "finished the graph on an RB fit that is not a measurement (0.009 decay lengths, amplitude 16.5)",
}
INCIDENTS = [
    ("20 Sep 15:27", "The night-3 retry cell on gilboa stopped at the user's request (qD1 past DRAG at turn 55, qC5 on a flux map that tracked "
                     "no line). Plan for the night: every C/D qubit of gilboa and all six of qolab, tinycal with qwen3.8-27b on Splash and on "
                     "OpenRouter from the same scramble per backend; Splash never above two streams, no backend above three targets."),
    ("20 Sep 16:00", "Launch. tinycal main 6aa42ea (write_state refuses a state the ports cannot execute; a 30 min hardware budget per target "
                     "with wrap-up turns; wall cap 8 h), qua-libs feat/qualibrate-ai 553b9ea (the DRAG fit-free baseline fix f9bf1de, plus a "
                     "revert of the flux-map duration change 4ca26fb that later proved unnecessary). qC5's y90 literal patched in the run "
                     "copies. Four cells: gilboa Splash-A (8 targets, one at a time) + OpenRouter (10, two at a time); qolab Splash (6, one "
                     "at a time) + OpenRouter (6, two at a time). Splash at max_tokens 10 000, OpenRouter 16 000, both effort high."),
    ("20 Sep 16:34", "arbel qB4 qC2 qC3 added on both providers (user). The fresh pull was refused by the scrambler — the 20 Sep arbel state "
                     "stores readout as a reference, the 14 Sep shape again — so the cell runs from the 19 Sep snapshot. OpenRouter started at "
                     "once (three at a time), the Splash cell queued behind the two-stream limit."),
    ("20 Sep 17:06", "Root cause of the gilboa flux anomaly seen since night 3: the gilboa cloud state pulled on 20 Sep lists only qC2 in "
                     "active_qubit_names (the 19 Sep pull listed all nine C/D qubits). quam_builder's set_all_fluxes(joint) applies the joint "
                     "idle only to active qubits and calls to_min() — 0 V — on every other z line, so every non-qC2 target was measured at "
                     "0 V: each flux node found the apex '+20 mV from the idle' again (qD5: 0 → 0.024 → 0.051 → 0.058 → 0.060 V), the "
                     "committed offset was never applied, and RB ran off the sweet spot (OpenRouter qD5 97.72 %, f_01 6.6 MHz low). qolab "
                     "and arbel list all their qubits, so they were unaffected. Not the node change 4ca26fb blamed the day before."),
    ("20 Sep 17:10", "Both gilboa cells and both chain scripts stopped (qolab and arbel cells untouched); active_qubit_names set to the ten "
                     "C/D qubits in the run copies (patch_active.py); gilboa relaunched at 17:12 from the same 16:00 pull with a fresh "
                     "scramble. The first flux node of the relaunch put qD3 at 0.0267 then 0.0136 V (reference 0.0140): converging."),
    ("20 Sep 17:20", "gilboa OpenRouter qD5, then qC5 at 18:14: the model widened the resonator flux sweep to ±0.5 V (many flux periods on "
                     "gilboa's direct LF-FEM lines), the arc fit failed (R² 0.08 / 0.16), and the node still proposed the 'measured apex' at "
                     "−0.13 V — a random period — which the model committed. qD5 then found its qubit 240 MHz low with χ ≈ 0 and stopped; "
                     "qC5 never recovered either. A node defect to fix: no joint_offset proposal from a failed fit, and a warning when the "
                     "sweep spans more than a period. The default sweep (what qD3, qD4, qD2 used) works."),
    ("20 Sep 18:13", "Splash answered gilboa qD3 with '503 memory did not become available' three times while the qolab Splash target was "
                     "generating a 10 000-token turn against a 77k context; tinycal's retry ladder (5/15/45/60/120/300 s) carried it through. "
                     "The same happened at 19:06. Two streams of ~80k context plus a 10k generation is at the edge of the server's 184k KV cache."),
    ("20 Sep 18:28", "arbel qB4 (OpenRouter) ended on the hardware budget: 31.4 min of QPU at turn 112 — the first time the cap fired; "
                     "wrap-up turns, then stuck, as designed. qC2 had already stopped itself on a wrong line 103 MHz low (χ ≈ 0, RB 85.7 %); "
                     "qC3 99.74 % with the same f_01 (5.7153 GHz) that night 2 measured against a stale reference."),
    ("20 Sep 18:58", "qolab OpenRouter done: 5 of 6 at 99.89–99.96 %; Q2 committed a readout power without a punch-out onset, got no "
                     "contrast and stopped itself with a flat RB trace — the node's 'no decay' warning read correctly."),
    ("20 Sep 19:41", "gilboa OpenRouter: qD2 95.86 % (the T1 ≈ 1.3 µs qubit; 98.7–98.8 % in the framework rounds); qD1 'completed' on an RB "
                     "fit of 0.009 decay lengths (99.998 % with amplitude 16.5 — meaningless), after three re-runs with a hand-written depth "
                     "list crashed the node. qolab Splash Q1: three of its last turns hit the 10 000-token cap (9–25 min each) but each "
                     "eventually issued a tool call, so it was left running."),
]

# What would have caught each hard case, by (backend, provider token, qubit); the operator's reading of the transcripts.
HARD_CASE_NOTE = {
    ("gilboa", "openrouter", "qD5"): "the resonator flux map proposed a joint_offset from a fit it had itself rejected (R² 0.08) over a ±0.5 V "
                                     "sweep that spans several flux periods; the node must propose nothing without a fit, and warn when the sweep "
                                     "is wider than a period",
    ("gilboa", "openrouter", "qC5"): "the same −0.13 V 'measured apex' from a failed fit (R² 0.16) over ±0.5 V; same fix",
    ("gilboa", "openrouter", "qD1"): "the RB node reported 99.998 % from a fit covering 0.009 decay lengths with amplitude 16.5; the fit's own "
                                     "warning was in the result and the model declared completion anyway. A fit that far outside [0, 1] should "
                                     "return no number",
    ("gilboa", "openrouter", "qD2"): "the T1 ≈ 1.3 µs qubit: 95.9 % against 98.7–98.8 % in the framework rounds; a genuine calibration, poorer "
                                     "than the best on this qubit",
    ("qolab", "openrouter", "Q2"): "readout power committed at 0.035 V with no punch-out onset in the sweep, ~50 % IQ contrast, flat RB; the "
                                   "night-2 recipe's 'provisional low power' line was followed, but no second power sweep at the sweet spot",
    ("arbel", "openrouter", "qC2"): "a line 103 MHz below the reference committed as f_01 (χ ≈ 0 at the upper sweet spot); the model saw the "
                                    "flat power Rabi and flat blobs and blamed the readout instead of the frequency",
    ("arbel", "openrouter", "qB4"): "the hardware budget (30 min of QPU) ran out at turn 112 after 48 node runs; the cap worked as designed — "
                                    "the run had spent its time on readout re-optimisation",
}


# ----------------------------------------------------------------------------- collect
def provider_of(cell_name: str) -> str:
    for token in PROVIDERS:
        if f"-{token}" in cell_name:
            return token
    return "openrouter"


def live_status(run_id: str) -> dict:
    try:
        return json.load(open(TINYCAL_RUNS / run_id / "run.json")).get("status", {})
    except Exception:  # noqa: BLE001
        return {}


def collect():
    prices = load_prices()
    cells = []
    for work in sorted(RUNS.glob("night4-*")):
        if not work.is_dir() or work.name in EXCLUDED_DIRS or "." in work.name.split("-")[-1]:
            continue
        backend = work.name.split("-")[1]
        docs = sorted(glob.glob(str(work / "*/result.json")))
        if not docs:
            continue
        cm = compare_models(work)
        for doc in docs:
            cell = Path(doc).parent
            token = provider_of(cell.name)
            label, model_id, colour = PROVIDERS[token]
            r = json.load(open(doc))
            t = r["totals"]
            live = live_status(r.get("run_id") or "")
            final = (work / f"cell-{cell.name.split('qwen3-8-27b-')[-1]}.log").exists() and \
                "done →" in (work / f"cell-{cell.name.split('qwen3-8-27b-')[-1]}.log").read_text()
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
                status = x["status"]
                if status == "pending":
                    status = live.get(x["target"], "pending")  # running / queued while the cell is alive
                cause = TARGET_INCIDENTS.get((work.name, cell.name, x["target"]))
                targets.append({
                    "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                    "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, model_id, prices),
                    "nodes_run": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                    "gate_fid": (gate_fidelity(rb.get("error_per_clifford")) if x["status"] == "completed"
                                 and (backend, token, x["target"]) not in INVALID_RB else None),
                    "ctx": (lambda cs: {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None, "n": len(cs)})(
                        context_series(cell, r.get("run_id") or "", x["target"], "tinycal")),
                    "target": x["target"], "status": status,
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
                    "provider": token,
                })
            qubits = [x["target"] for x in r["targets"]]
            c = {
                "work": work.name, "cell": cell.name, "backend": backend, "fw": colour, "model": label, "provider": token,
                "qubits": qubits, "round": (backend, "+".join(qubits)), "run_id": r.get("run_id"),
                "status": r["status"] if final else "running", "started": local(r.get("started_at")),
                "ended": local(r.get("ended_at")) if final else "—",
                "wall_s": t["time"]["total_s"], "model_s": t["time"].get("model_s"),
                "queue_s": t["time"].get("queue_wait_s"), "qpu_s": t["time"].get("qpu_execution_s"),
                "turns": t["turns"]["total"], "nodes": t["nodes"]["executions"], "reruns": t["nodes"]["re_executions"],
                "tok_in": t["tokens"].get("input"), "tok_out": t["tokens"].get("output"),
                "tok_cr": t["tokens"].get("cache_read"), "tok_cw": t["tokens"].get("cache_creation"),
                "collateral": len((r.get("judge") or {}).get("collateral_edits") or []),
                "targets": targets,
                "finished": sum(1 for x in targets if x["status"] == "completed"),
                "chip": chip_counters(cell, r.get("run_id") or ""),
                "incident": None, "superseded": False, "final": final,
            }
            row = cm.get(cell.name, {})
            c["cost"], c["cw_tokens"], c["graph_frac"] = row.get("cost"), row.get("tokens"), row.get("graph")
            c["stopped_by"] = row.get("stopped_by")
            cells.append(c)
    return cells


# ----------------------------------------------------------------------------- tables specific to this night
def pivot_table(cells, token):
    """night2's pivot for one provider: metrics as rows, one column per backend and one for all."""
    label, _, colour = PROVIDERS[token]
    sub = [c for c in cells if c["provider"] == token]
    if not sub:
        return f"<p class='muted small'>no {esc(label)} cell has a document yet</p>"
    night2.MODEL = label
    html = night2.pivot_table(sub)
    chip = f'<span class="chip"><i style="background:{base.FW_COLOR[colour]}"></i>{esc("tinycal · " + label)}</span>'
    return html.replace(base.fw_chip("tinycal", label), chip)


def qubit_table(cells):
    """One row per qubit-run, both providers side by side per qubit."""
    by = {}
    for c in cells:
        for t in c["targets"]:
            by.setdefault((c["backend"], t["target"]), {})[c["provider"]] = (c, t)
    order = {"gilboa": 0, "qolab": 1, "arbel": 2}

    def cell_html(entry):
        if entry is None:
            return "<td colspan='6' class='muted small'>not attempted</td>"
        c, t = entry
        done = t["status"] == "completed"
        cause = "" if done else f"<br><span class='small muted'>{esc(t['cause'] if t['status'] not in ('running', 'queued') else t['status'])}</span>"
        fid = pct(t["gate_fid"]) if done else "—"
        return (f"<td>{status_chip(t['status'])}{cause}</td><td class='num'>{t['nodes'] if t['nodes'] is not None else 0}/{t['graph'] or 18}</td>"
                f"<td class='num'>{fid}</td><td class='num'>{t['ballpark'] if t['ballpark'] is not None else '—'}/{t['graded'] or 4}</td>"
                f"<td class='num'>{minutes(t['model_s'])} · {minutes(t['qpu_s'])}</td><td class='num'>{fmt(t['turns'], '{}')} · {fmt(t['cost'], '${:.2f}')}</td>")
    rows = []
    for (b, q), prov in sorted(by.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        rows.append(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td>{cell_html(prov.get('splash'))}{cell_html(prov.get('openrouter'))}</tr>")
    sub = "<th>status</th><th>nodes</th><th>gate fid.</th><th>ballpark</th><th>agent · QPU</th><th>turns · cost</th>"
    head = (f'<thead><tr><th></th><th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["splash"]}">Splash</th>'
            f'<th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["tinycal"]}">OpenRouter</th></tr>'
            f'<tr><th>qubit</th>{sub}{sub}</tr></thead>')
    return f'<div class="scroll"><table class="grid small">{head}<tbody>' + "".join(rows) + "</tbody></table></div>"


def hard_cases(cells):
    rows = []
    for c in cells:
        for t in c["targets"]:
            if t["status"] in ("running", "queued", "pending"):
                continue
            poor = t["status"] == "completed" and t["gate_fid"] is not None and t["gate_fid"] < 0.99
            invalid = INVALID_RB.get((c["backend"], c["provider"], t["target"]))
            if t["status"] == "completed" and not (poor or invalid):
                continue
            fid = f" · {pct(t['gate_fid'])}" if t["gate_fid"] is not None else ""
            outcome = (f"{status_chip(t['status'])} {t['nodes'] or 0}/18{fid}<br><span class='small'>{minutes(t['model_s'])} agent · "
                       f"{fmt(t['cost'], '${:.2f}')} · {fmt(t['turns'], '{}')} turns</span>")
            what = t["cause"] if t["status"] != "completed" else (invalid or "finished the graph with a poor number")
            note = HARD_CASE_NOTE.get((c["backend"], c["provider"], t["target"]), "")
            rank = 0 if t["status"] != "completed" else (1 if invalid else 2)
            rows.append((rank, c["backend"], t["target"], c["provider"], outcome, what, note))
    rows.sort()
    body = "".join(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td><td>{esc(PROVIDERS[p][0].split(' · ')[1])}</td><td>{o}</td>"
                   f"<td class='small'>{esc(w)}</td><td class='small'>{esc(n)}</td></tr>" for _, b, q, p, o, w, n in rows)
    if not rows:
        body = "<tr><td colspan='5' class='muted'>none yet</td></tr>"
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>qubit</th><th>served by</th><th>outcome</th>'
            '<th>what happened</th><th>what would have caught it</th></tr></thead><tbody>' + body + "</tbody></table></div>")


def in_flight(cells):
    items = []
    for c in cells:
        for t in c["targets"]:
            if t["status"] in ("running", "queued"):
                items.append(f"{c['backend']} {t['target']} ({PROVIDERS[c['provider']][0].split(' · ')[1]}, {t['status']})")
    return items


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
    for label, path in night2.CHECKOUTS.items():
        sha = base._git(path, "rev-parse", "--short", "HEAD")
        dirty = base._git(path, "status", "--short").replace("\n", "; ") or "clean"
        if label.startswith("tinycal"):
            recorded = ("document fingerprint.git_sha: " + fmt_counter(fw_sha) + ". Every cell ran main at 6aa42ea (tinycal-rev.txt in each "
                        "work dir); the recipe file bringup_recipes/flux_tunable_1q.md is the one of the 18-19 Sep night.")
        elif label.startswith("qua-libs"):
            recorded = ("document fingerprint.calibration_content: " + fmt_counter(libs) + ". The working copy at 553b9ea (qua-libs-rev.txt in "
                        "each work dir): f9bf1de + the revert of 4ca26fb.")
        else:
            recorded = "not in the documents; the judge's spec digests below are what it enforces"
        rows.append(f"<tr><td>{esc(label)}</td><td class='mono'>{esc(sha)}</td><td class='small'>{esc(dirty)}</td><td class='small'>{recorded}</td></tr>")
    rows.append(f"<tr><td>scramble spec (workloads/decalibrate_chip.yaml)</td><td class='mono'>{esc(next(iter(spec), ''))}</td><td class='small'>judge refuses a document with another digest</td><td class='small'>document scramble.spec_hash: {fmt_counter(spec)}</td></tr>")
    rows.append("<tr><td>acceptance spec (workloads/acceptance.yaml)</td><td class='mono'>821bb4dd1d5d0e5a</td><td class='small'>—</td><td class='small'>stamped by qab accept</td></tr>")
    for b in sorted(snap):
        dirs = sorted(srcdir[b])
        rows.append(f"<tr><td>{esc(b)} snapshot (source-state, patched as described above)</td><td class='mono'>{esc(next(iter(snap[b]), ''))}</td>"
                    f"<td class='small'>seed a new work dir from <span class='mono'>{esc(dirs[0])}/source-state</span>; qab scramble is deterministic</td>"
                    f"<td class='small'>document scramble.source_hash: {fmt_counter(snap[b])}</td></tr>")
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>component</th><th>sha / digest</th>'
            '<th>uncommitted at report generation</th><th>as recorded in the cell documents</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def run_ids(cells):
    rows = []
    for c in sorted(cells, key=lambda c: (c["work"], c["cell"])):
        tc = f"~/code/QM/tinycal/runs/{c['run_id']}"
        chip = f'<span class="chip"><i style="background:{base.FW_COLOR[c["fw"]]}"></i>{esc("tinycal · " + c["model"])}</span>'
        rows.append(f"<tr><td class='mono small'>{esc(c['work'])}</td><td class='mono small'>{esc(c['cell'])}</td>"
                    f"<td>{chip}</td><td class='mono'>{'+'.join(c['qubits'])}</td>"
                    f"<td class='mono small'>{esc(tc)}</td><td class='num'>{esc(c['started'])} → {esc(c['ended'])}</td><td>{status_chip(c['status'])}</td></tr>")
    for d, why in EXCLUDED_DIRS.items():
        rows.append(f"<tr class='sup'><td class='mono small'>{esc(d)}</td><td colspan='6' class='small muted'>{esc(why)}</td></tr>")
    return ('<div class="scroll"><table class="grid small ids"><thead><tr><th>work dir (~/qab-runs/)</th><th>cell dir</th><th>cell</th><th>qubits</th>'
            '<th>tinycal run dir</th><th>started → ended (document span)</th><th>status</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- page
def build():
    cells = collect()
    base.MODEL_LABELS[:] = [v[0] for v in PROVIDERS.values()]
    base.SUPERSEDED_DIRS.clear()
    night2.EXCLUDED_DIRS.clear()
    night2.EXCLUDED_DIRS.update(EXCLUDED_DIRS)
    runs = [(c, t) for c in cells for t in c["targets"]]
    settled = [(c, t) for c, t in runs if t["status"] not in ("running", "queued", "pending")]
    done = [(c, t) for c, t in runs if t["status"] == "completed"]
    flight = in_flight(cells)
    live = bool(flight) or any(not c["final"] for c in cells)
    spent = sum((t["cost"] or 0) for _, t in runs)
    now = datetime.now().strftime("%d %b %Y %H:%M")

    def prov_stats(token):
        d = [t for c, t in done if c["provider"] == token]
        s = [t for c, t in settled if c["provider"] == token]
        f = sorted(t["gate_fid"] for t in d if t["gate_fid"] is not None)
        return len(d), len(s), (f[len(f) // 2] if f else None), (f[0] if f else None), (f[-1] if f else None)
    so, ss, mo, lo_o, hi_o = prov_stats("openrouter")
    sp, sps, mp, lo_p, hi_p = prov_stats("splash")
    tiles = [
        f'<div class="tile"><div class="v">{len(done)}/{len(settled)}</div><div class="k">calibrations completed / settled so far'
        f'{" · " + str(len(flight)) + " in flight or queued" if flight else ""}</div></div>',
        f'<div class="tile"><div class="v">{so}/{ss}</div><div class="k">OpenRouter: completed / settled — median {pct(mo)} ({pct(lo_o)} – {pct(hi_o)})</div></div>',
        f'<div class="tile"><div class="v">{sp}/{sps}</div><div class="k">Splash: completed / settled — median {pct(mp)} ({pct(lo_p)} – {pct(hi_p)})</div></div>',
        f'<div class="tile"><div class="v">${spent:,.2f}</div><div class="k">judge-priced spend so far (OpenRouter only; Splash is self-hosted and unpriced)</div></div>',
    ]

    def lab(it, plain=False):
        return it["label"] if plain else esc(it["label"])
    cost_items = sorted([{"fw": c["fw"], "cost": t["cost"], "label": f"{c['backend']} {t['target']}" + ("" if t["status"] == "completed" else f"  ({t['status']})")}
                         for c, t in runs if t["cost"] is not None and c["provider"] == "openrouter"], key=lambda it: it["label"])
    cost_fig = bars(cost_items, "cost", lab, lambda v: f"${v:.2f}", "Judge-priced cost per qubit-run, OpenRouter", "USD at prices.yaml (OpenRouter's rate for qwen3.8-27b); unfinished runs marked")
    err_items = sorted([{"fw": c["fw"], "err": 100 * (1 - t["gate_fid"]), "label": f"{c['backend']} {t['target']} · {PROVIDERS[c['provider']][0].split(' · ')[1]}"}
                        for c, t in runs if t["gate_fid"] is not None], key=lambda it: it["label"])
    err_fig = bars(err_items, "err", lab, lambda v: f"{v:.3f}% ({100 - v:.2f}%)", "Single-qubit gate error per completed qubit-run", "gate error in %, fidelity in brackets; error per Clifford ÷ 1.875; green OpenRouter, blue Splash")
    timeline = '<table class="grid timeline"><tbody>' + "".join(f"<tr><td>{esc(t)}</td><td>{esc(w)}</td></tr>" for t, w in INCIDENTS) + "</tbody></table>"
    banner = (f'<div class="tile" style="border-color:var(--warn)"><div class="k"><b>Live page</b> — regenerated {esc(now)} while cells are still '
              f'running. In flight or queued: {esc(", ".join(flight)) if flight else "none"}. Rows of running targets are interim '
              f'(projected from the live event log); every number can still move until the cell ends.</div></div>') if live else ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Splash vs OpenRouter, qwen tinycal</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><div class="page">
<div class="eyebrow">qua-agents benchmark · same model, two hosts, overnight run · generated {esc(now)}</div>
<h1 style="margin-top:8px">tinycal with qwen3.8-27b served by Splash and by OpenRouter, on gilboa C/D, qolab and three arbel qubits, 20–21 Sep 2026</h1>
<p class="lede">One framework, one model, two hosts. The same scrambled snapshot per backend was handed to tinycal twice: once with qwen3.8-27b on
the self-hosted Splash server (reasoning_effort high, 10 000-token output cap, at most two concurrent streams — its KV cache holds ~184k
tokens) and once with the same weights on OpenRouter (effort high, 16 000-token cap). Targets: gilboa qC1–qC5 and qD1–qD5 (the B row's
readout line has a broken TWPA and is excluded), qolab Q1–Q6, arbel qB4 qC2 qC3. The harness ran with the guards added after the 18–19 Sep
night — write_state refuses a state the ports cannot execute, and each target has a 30 min budget of hardware time — and the DRAG node's
fit-free cross-check was fixed. Numbers are read from each qubit-run's own record in result.json, the cell's QuAM state and the judge's price
table; the operator annotations are marked.</p>
{banner}

<h2 id="overview">Overview</h2>
<p><b>Apparatus.</b> tinycal main <span class="mono">6aa42ea</span> run from its working tree (max_turns 150, max_qpu_min 30 with six wrap-up turns,
wall cap 8 h), qua-libs <span class="mono">feat/qualibrate-ai</span> at <span class="mono">553b9ea</span>: the RB node reports its fit uncertainty
and warns when the sweep stops short of the asymptote (9f629eb), the qubit flux map fits nothing when no column saw a line (81ef4c0), the DRAG
node reads its fit-free cross-check against the ground level of the first pulse row and withholds it when no vertex is inside the window
(f9bf1de), and the flux-map duration change 4ca26fb is reverted — a precaution taken before the real cause of the gilboa maps was found
(see 17:06 in the timeline), harmless but unnecessary. Workload decalibrate_chip.yaml, scramble spec <span class="mono">f47053f145f74c8c</span>,
acceptance spec <span class="mono">821bb4dd1d5d0e5a</span>. Snapshots: gilboa — {esc(SNAPSHOT_NOTE['gilboa'])}; qolab — {esc(SNAPSHOT_NOTE['qolab'])};
arbel — {esc(SNAPSHOT_NOTE['arbel'])}.</p>
<p><b>Scheduling.</b> Splash serves at most two streams, so its cells run one target at a time and the chain starts the next Splash cell only when a
stream frees; OpenRouter cells run two targets at a time (three on arbel), and no backend ever has more than three targets in flight across the
two providers. Both providers on a backend share one scramble but hold separate copies of the state, so a bad write by one never reaches the other.</p>
<h3>Pins: what a new cell must use to be comparable</h3>
{pins_table(cells)}
<div class="tiles">{"".join(tiles)}</div>
<p class="small muted">How to read the tables: one qubit-run = tinycal calibrating one qubit from the scrambled state with qwen3.8-27b on one host.
Gate fidelity = 1 − (RB error per Clifford ÷ 1.875). "Per calibration" = the total over every qubit-run attempted on that backend divided by the
number completed. Agent time is time inside model calls; QPU time is execution on the chip; queue wait is the cloud queue. Splash tokens are
counted but unpriced (self-hosted). Context = prompt size per model call in tokens, from tinycal's events.jsonl.</p>
<h3>By host</h3>
{pivot_table(cells, "openrouter")}
{pivot_table(cells, "splash")}

<h3>Per qubit, host side by side</h3>
{qubit_table(cells)}

<h3>Hard cases: qubits that were not calibrated, or not calibrated right</h3>
<p class="small muted">One row per settled qubit-run that did not reach the end of the graph, then the completed ones under 99 %. Running and queued
targets are not listed. The right-hand column is the operator's reading of the transcript, not something the cell recorded.</p>
{hard_cases(cells)}
{cost_fig}
{err_fig}

<h3>What the numbers say{" (so far)" if live else ""}</h3>
<p><b>The gilboa results of the last two nights were measured at the wrong flux, and it was the state, not the nodes.</b> Since the 20 Sep pulls the
gilboa cloud state lists only qC2 as active; quam_builder parks every non-active z line at 0 V, so night 3 and the first cells of this night calibrated
gilboa's qubits 14–30 mV from their sweet spots without any node saying so — the flux maps were correct in absolute volts and the harness added the
apex to an idle that was never applied. One line in the state (active_qubit_names) explains the qD5 99.27 % / 97.72 %, the "+20 mV per map" walk and
qC5's blank maps; patched in the run copies at 17:10, the relaunched gilboa qD3 landed its offset within 0.6 mV of the reference. This and the qC5
y90 literal are both cloud-state defects for IQCC.</p>
<p><b>OpenRouter, so far:</b> qolab 5/6 at 99.89–99.96 % (Q2 lost to readout power), arbel qC3 99.74 % with qC2 on a wrong line and qB4 out of hardware
budget, gilboa qD3 99.82 % with the flux fix, qD2 95.9 % on the short-T1 qubit, and two targets (qD5, qC5) sent to −0.13 V by a resonator flux map that
proposed an apex from a fit it had rejected — the one new node defect of the night. <b>Splash</b> is slower per turn (its 10 000-token generations take
5–25 min under two streams, and it returned 503 memory errors to the other stream while doing so); its results are in the tables as they land.</p>

<h2 id="stuck">Where things got stuck</h2>
<h3>Incident timeline</h3>
{timeline}

<h2 id="runs">Run identifiers</h2>
{run_ids(cells)}
</div></body></html>"""
    OUT.write_text(page)
    # the same page as an Artifact body: the host supplies doctype/html/head/body, the title and styles stay at the top
    art = Path(ARTIFACT_DIR) / OUT.name if ARTIFACT_DIR else None
    if art:
        body = page.split("<title>", 1)[1]
        body = "<title>" + body.replace("</head><body>", "").replace("</body></html>", "")
        art.write_text(body)
    print(f"wrote {OUT} · {len(cells)} cells, {len(runs)} qubit-runs, {len(done)} completed, {len(flight)} in flight/queued, ${spent:,.2f}")


if __name__ == "__main__":
    build()
