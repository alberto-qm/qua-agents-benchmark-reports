#!/usr/bin/env python3
"""The n17b summary page: make_n17_summary.py's page for n17b, the 6-7 Oct 2026 run of tinycal with qwen3.8-27b on n16's
37 single-qubit bring-ups through the chirp bring-up graph with the FPGA-adaptive pi pulse and DRAG
(FluxTunableTransmon_ChirpBringUp, graph 84), beside n16 (the same agent and model on graph 80, 2 Oct) on the same qubits.

    python3 make_n17b_summary.py

It can run while n17b is still going: only the cells with a result.json are collected, and n16's column covers the same
qubits. ~/qab-runs/n17b-LOG.md has the run's libraries (the readout confusion matrix written whole, IQ blobs again with
the x180 after the adaptive DRAG, 04f/04g refusing without a matrix, the recipe's adaptive-node rules) and the hardware
check before launch. The user asked for a page about n17b against n16 only. The output is page content for a claude.ai
artifact.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_e2_summary as e2s  # noqa: E402
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402
from make_fwcmp2_report import CSS, esc, median, minutes  # noqa: E402
from make_fwcmp2_report import pct as fpct  # noqa: E402

rep = summ.rep
QAB = Path.home() / "qab-runs"
OUT = Path(__file__).with_name("2026-10-07-n17b-summary.html")
STAMP = (QAB / "n17b-stamp.txt").read_text().strip()
N16_STAMP = (QAB / "n16-stamp.txt").read_text().strip()
LABEL = {"old": "n16", "new": "n17b"}
STORE = Path.home() / ".qualibrate/user_storage/tinycal/measurements.sqlite3"
N_QUBITS = 37  # n16's set: qolab Q1-Q6, arbel all 21, gilboa qC1-qC5 qD1-qD5

# Graph 84's order (tinycal recipe flux_tunable_1q_chirp_adaptive.md): the power sweep repeats at the sweet spot (step 4),
# T1_chirp precedes the flux map, IQ blobs runs again with the x180 after the adaptive DRAG, and T1/T2echo share a step.
RECIPE_84 = [("resonator_identification",), ("resonator_spectroscopy_vs_power",), ("resonator_spectroscopy_vs_flux",),
             ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy_chirp",), ("T1_chirp",),
             ("qubit_spectroscopy_vs_flux_chirp",), ("readout_frequency_chirp",), ("readout_power_photons_chirp",),
             ("IQ_blobs_chirp",), ("pi_pulse_adaptive",), ("drag_adaptive",), ("IQ_blobs",), ("T1", "T2echo"),
             ("Randomized_benchmarking",)]


def configure() -> None:
    graph80 = rep.RECIPE["new"]
    rep.NIGHTS = {"old": ("n16",), "new": ("n17b",)}
    rep.NIGHT_LABEL = {"old": "n16 (graph 80)", "new": "n17b (graph 84)"}
    rep.CELL_MARKS = ("-tinycal-",)
    rep.RECIPE["old"], rep.RECIPE["new"] = graph80, RECIPE_84
    rep.NOT_MEANINGFUL = {}
    rep.NOT_COMPLETED = {}
    summ.RECOVERY_NIGHTS = (("n16", N16_STAMP), ("n17b", STAMP))


def not_completed(rows: list) -> None:
    """The escalation reason tinycal recorded, as the hover text of each n17b cell that did not complete."""
    for r in rows:
        if r["night"] == "new" and r["status"] != "completed":
            doc = json.load(open(QAB / r["work"] / r["cell"] / "result.json"))
            why = (doc["targets"][0].get("escalation_reason") or r["status"]).strip().splitlines()[0][:300]
            rep.NOT_COMPLETED[("new", r["backend"], r["q"])] = why


def context(s: dict, done: int) -> str:
    a, _ = (n13summ._minute(t) for t in s["span"])
    state = (f"interim: {done} of {N_QUBITS} qubits finished at {datetime.now():%H:%M}, the rest still running"
             if done < N_QUBITS else f"all {N_QUBITS} qubits")
    return (f"{a:%-d %b %Y}, from {a:%H:%M} CEST · {state} · tinycal with qwen3.8-27b on the chirp bring-up graph "
            "(chirp spectroscopy, T1 and readout, the FPGA-adaptive pi pulse and DRAG, then IQ blobs again with the new x180) "
            "· beside n16, the same agent and model on the usual bring-up graph, 2 Oct")


N17 = summ.Night(
    label="n17b", stamp=STAMP, out=OUT,
    title="n17b Summary",
    description="n17b, 6-7 Oct 2026: tinycal and qwen3.8-27b on n16's single-qubit bring-ups through the chirp bring-up graph "
                "with the FPGA-adaptive pi pulse and DRAG, beside n16 on the same qubits: one table, the adaptive nodes, "
                "four plots and a per-qubit check against IQCC.",
    eyebrow="n17b chirp + adaptive graph",
    h1="n17b: chirp bring-up with the adaptive pi pulse",
    context=lambda s: "",
    chip_sub=lambda s: f"6 Oct · {s['n']} qubits · graph 84, qwen3.8-27b",
    rb_note=n13summ.rb_note,
    pull="the pull each run started from",
    pull_short="the pull each run started from",
    recovery_intro="One qubit per pair of rows: <b>n16</b>, the 2 Oct run on graph 80, and <b>n17b</b>, graph 84 on 6-7 Oct, "
                   "each against the IQCC pull its own run started from. n17b's adaptive pi pulse may lengthen the x180 "
                   "(the profile lets it write the length), and an x180 amplitude at another length is not comparable "
                   "with the lab's: the results table counts those qubits.",
    recovery_sort=("n17b", "n16"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n17b, fewest first, then by n16's count.",
    spec="252e28f8ab313894",
    scramble_note="<p class='small muted'>The same spec as n16, with its state-field allowlist; n17b's states were pulled and "
                  "scrambled afresh on 6 Oct 21:46.</p>",
)


# ----------------------------------------------------------------------------- results table, two runs
def _lab_x180_len(r) -> float | None:
    try:
        lab = json.load(open(QAB / r["work"] / "source-state/state.json"))["qubits"][r["q"]]
        return float(lab["xy"]["operations"]["x180_DragCosine"]["length"])
    except Exception:  # noqa: BLE001
        return None


def _adaptive(r) -> tuple[int, int]:
    """(pi_pulse_adaptive runs, of them failed or refused) in one n17b cell."""
    runs = [e for e in rep.events(r["run_id"], r["q"]) if e.get("kind") == "tool_call" and e.get("tool") == "run_node"
            and e.get("node") == "pi_pulse_adaptive"]
    return len(runs), sum(1 for e in runs if not (e.get("status") == "completed" and e.get("outcome") == "successful"))


def results_table(rows: dict, S: dict) -> str:
    def cells(fn):
        return [fn(k, rows[k], S[k]) for k in ("old", "new")]

    def qubits(k, rs, s):
        return summ.qubit_list(rs)

    def meaningful(k, rs, s):
        return f"<b class='num'>{len(s['meaningful'])}/{s['n']} ({100 * len(s['meaningful']) / s['n']:.0f}%)</b>"

    def valid(k, rs, s):
        fids = [r["gate_fid"] for r in s["valid"] if r["gate_fid"] is not None]
        return (f"<span class='num'><b>{len(s['valid'])}</b>, median {fpct(median(fids))}</span>"
                f"<div class='small muted'>{esc(n13summ.rb_note(s))}</div>")

    def readout(k, rs, s):
        return f"<span class='num'><b>{100 * s['ro_med']:.1f} %</b> · {s['ro_95']} of {s['ro_n']} ≥ 95 %</span>"

    def judge(k, rs, s):
        a, b = s["judge"]
        return f"<span class='num'>{a}/{b} ({100 * a / b:.0f}%)</span>" if b else "<span class='muted'>—</span>"

    def adaptive(k, rs, s):
        if k == "old":
            return ("<span class='muted'>— (power Rabi, error amplification, Ramsey and the DRAG sweep)</span>")
        counts = [_adaptive(r) for r in rs]
        n, bad = sum(c[0] for c in counts), sum(c[1] for c in counts)
        once = sum(1 for c in counts if c[0] == 1 and c[1] == 0)
        return (f"<span class='num'>{n} runs · {bad} failed</span><div class='small muted'>median "
                f"{median([c[0] for c in counts]):.0f} per qubit; right first time on {once} of {len(rs)}</div>")

    def lengths(k, rs, s):
        if k == "old":
            return "<span class='muted'>— (read-only)</span>"
        moved = []
        for r in s["done"]:
            lab = _lab_x180_len(r)
            if lab is not None and r["x180_len"] is not None and abs(r["x180_len"] - lab) > 0.5:
                moved.append(f"{r['backend']} {r['q']} {lab:.0f}→{r['x180_len']:.0f} ns")
        return (f"<span class='num'>{len(moved)} of {len(s['done'])}</span>"
                + (f"<div class='small muted'>{esc(', '.join(moved))}</div>" if moved else ""))

    def model_time(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>{minutes(median([r['model_s'] for r in done if r['model_s'] is not None]))}</span>"
                f"<div class='small muted'>median over the {len(done)} completed calibrations</div>")

    def qpu(k, rs, s):
        done = s["done"]
        fixed = [r["fixed_s"] for r in done if r.get("fixed_s") is not None]
        seq = [r["seq_s"] for r in done if r.get("seq_s") is not None]
        split = ""
        if fixed:
            split = (f"<div class='small'>estimated: compilation and job start-up <b class='num'>{median(fixed) / 60:.1f} min"
                     f"</b>, sequences <b class='num'>{median(seq) / 60:.1f} min</b></div>"
                     f"<div class='small muted'>medians per calibration · in all {sum(r.get('fixed_s') or 0 for r in rs) / 60:.0f}"
                     f" + {sum(r.get('seq_s') or 0 for r in rs) / 60:.0f} min (see QPU time below)</div>")
        return (f"<span class='num'>{median([r['qpu_s'] for r in done if r['qpu_s'] is not None]) / 60:.1f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · total {s['qpu_tot'] / 60:.0f} min</div>"
                + split)

    def wall(k, rs, s):
        done = s["done"]
        w = [r["wall_s"] for r in done if r["wall_s"] is not None]
        par = "3 in flight in all" if k == "old" else "3 in flight per device"
        return (f"<span class='num'>{median(w) / 60:.0f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · longest {max(w) / 60:.0f} min · "
                f"{par}</div>")

    def cost(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>${median([r['judge_cost'] for r in done if r['judge_cost'] is not None]):.2f}</span>"
                f"<div class='small muted'>median over the {len(done)} completed, at prices.yaml · total "
                f"${s['judge_cost']:.2f}</div>")

    def runs(k, rs, s):
        return (f"<span class='num'>{s['runs']} · {s['not_ok']} failed or refused</span>"
                f"<div class='small muted'>median {s['runs_med']:.0f} per qubit</div>")

    def overrides(k, rs, s):
        return e2s._share(s["ov_done"], s["ov_rest"], len(s["done"]) or 1, "writes")

    def off_order(k, rs, s):
        return e2s._share(s["oo_done"], s["oo_rest"], len(s["done"]) or 1, "node runs")

    body = [
        ("qubits measured (green: completed with meaningful results; red: not completed, reason on hover; the small letter "
         "is the device)", cells(qubits)),
        ("completed with meaningful results", cells(meaningful)),
        ("valid RB (the fit saw ≥ 1 decay length): runs, median fidelity per gate", cells(valid)),
        ("readout assignment fidelity, median over the runs that measured one", cells(readout)),
        ("judge: scrambled parameters back in the ballpark (readout frequency and amplitude, f₀₁, x180 amplitude)",
         cells(judge)),
        ("adaptive pi pulse (pi_pulse_adaptive): runs, all qubits", cells(adaptive)),
        ("x180 length changed from the lab's (the adaptive pi pulse chose a longer pulse)", cells(lengths)),
        ("model time / calibration", cells(model_time)),
        ("QPU median time / calibration", cells(qpu)),
        ("wall-clock median time / calibration", cells(wall)),
        ("judge median cost / calibration", cells(cost)),
        ("node runs, all qubits", cells(runs)),
        ("state writes that are not a node's proposal (no node proposed the path, or the value is more than 0.1 % from the "
         "latest proposal), completed calibrations; below, the runs that did not complete", cells(overrides)),
        ("node runs off the recipe order: stepped back to an earlier step or skipped one (retries and re-runs of the node just "
         "run are not counted), completed calibrations; below, the runs that did not complete", cells(off_order)),
    ]
    trs = "".join(f"<tr><th class='rowh'>{esc(lab)}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
                  for lab, vals in body)
    subs = {"old": f"2 Oct · the same {S['old']['n']} qubits · graph 80, qwen3.8-27b", "new": N17.chip_sub(S["new"])}
    head = "<thead><tr><th></th>" + "".join(
        f"<th class='grp'><span class='chip'><i style='background:{rep.SERIES[k]}'></i>{esc(LABEL[k])}</span>"
        f"<br><span class='small muted'>{esc(subs[k])}</span></th>" for k in ("old", "new")) + "</tr></thead>"
    return f"<div class='scroll'><table class='grid pivot sum sum2'>{head}<tbody>{trs}</tbody></table></div>"


# ----------------------------------------------------------------------------- the adaptive pi pulse's readout model
def _first_04f_readout(r) -> tuple[float | None, int, int]:
    """(IQ blobs' readout fidelity before the first pi_pulse_adaptive, its runs, of them failed) in one n17b cell."""
    readout, runs, bad = None, 0, 0
    for e in rep.events(r["run_id"], r["q"]):
        if e.get("kind") != "tool_call" or e.get("tool") != "run_node":
            continue
        if e.get("node") == "IQ_blobs_chirp" and e.get("outcome") == "successful" and readout is None and not runs:
            v = (e.get("numerics") or {}).get("readout_fidelity")
            readout = v.get("value") if isinstance(v, dict) else v
        if e.get("node") == "pi_pulse_adaptive":
            runs += 1
            bad += 0 if (e.get("status") == "completed" and e.get("outcome") == "successful") else 1
    return readout, runs, bad


def _store():
    return sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)


def _fit(db, measurement_id: str, q: str) -> tuple[dict, dict]:
    row = db.execute("select fit_results, parameters from measurements where measurement_id=?", (measurement_id,)).fetchone()
    if not row:
        return {}, {}
    return (json.loads(row[0] or "{}") or {}).get(q) or {}, json.loads(row[1] or "{}") or {}


def _confusion(db, r, node: str):
    """The confusion matrix of the cell's last successful ``node`` run, or None."""
    last = None
    for e in rep.events(r["run_id"], r["q"]):
        if e.get("kind") == "tool_call" and e.get("tool") == "run_node" and e.get("node") == node and e.get("outcome") == "successful":
            last = e
    return _fit(db, last["node_run_id"], r["q"])[0].get("confusion_matrix") if last else None


def _compile_fit(db, rows: list) -> dict:
    """Each adaptive node's QPU time per program execution, split into a fixed cost (compilation and overhead) and a
    cost per shot. IQCC reports one QPU time per job, compile included.

    A least-squares split is unstable here: the shots add ~1 s to a ~27 s execution while executions scatter by a few
    seconds. So the per-shot cost comes from medians: single-execution runs at the default 2,000 shots against runs at
    10,000 shots or more, pooled over every run of the node in tinycal's 6-7 Oct runs (run ids n17*, the same node
    code). The fixed cost is the default runs' median minus their shots, counted over this run's executions."""
    import glob
    import numpy as np

    mine = {(r["run_id"], r["q"]) for r in rows}
    out = {}
    for node in ("pi_pulse_adaptive", "drag_adaptive"):
        default, high, own_execs, own_qpu = [], [], 0, 0.0
        default_by = {}
        for path in glob.glob(str(rep.TINYCAL_RUNS / "n17*_*" / "*" / "events.jsonl")):
            run_id, q = Path(path).parent.parent.name, Path(path).parent.name
            backend = run_id.split("_")[1]
            for e in rep.events(run_id, q):
                if not (e.get("kind") == "tool_call" and e.get("tool") == "run_node" and e.get("node") == node):
                    continue
                if not e.get("qpu_execution_s") or not e.get("node_run_id"):
                    continue
                fr, p = _fit(db, e["node_run_id"], q)
                shots = int(p.get("num_shots") or 0) * int(p.get("num_repetitions") or 1)
                execs = len(fr.get("length_attempts") or []) or 1
                if (run_id, q) in mine:
                    own_execs += execs; own_qpu += float(e["qpu_execution_s"])
                if execs != 1:
                    continue
                if shots == 2000:
                    default.append(float(e["qpu_execution_s"]))
                    default_by.setdefault(backend, []).append(float(e["qpu_execution_s"]))
                elif shots >= 10000:
                    high.append((shots, float(e["qpu_execution_s"])))
        if not default:
            continue
        d_med = float(np.median(default))
        per_shot = 0.0
        if high:
            per_shot = max(0.0, (float(np.median([s for _, s in high])) - d_med) / (float(np.median([n for n, _ in high])) - 2000))
        fixed = d_med - 2000 * per_shot
        out[node] = dict(default=d_med, n_default=len(default), high=float(np.median([s for _, s in high])) if high else None,
                         n_high=len(high), high_shots=int(np.median([n for n, _ in high])) if high else None,
                         per_shot=per_shot, fixed=fixed, execs=own_execs, qpu=own_qpu,
                         by_backend={b: float(np.median(v)) - 2000 * per_shot for b, v in default_by.items()})
    return out


# ----------------------------------------------------------------------------- QPU time: fixed cost against sequences
COMPILE = QAB / "compile-20261007"
GRAPH = {"old": "FluxTunableTransmon_BringUp", "new": "FluxTunableTransmon_ChirpBringUp"}
ADAPTIVE = ("pi_pulse_adaptive", "drag_adaptive")
PROBE_QUBIT = {"arbel": "qA2", "gilboa": "qD5", "qolab": "Q3"}


def _probe_costs() -> dict:
    """{(backend, graph, node): fixed QPU seconds per node run}: the node run's jobs at one shot, every other parameter at
    the graph's preset (compile-20261007/probe_fixed.py, 7 Oct 2026, the n17b libraries, one calibrated qubit per device).
    A node that graphs 80 and 84 share with the same presets ran once, under graph 84."""
    out = {}
    for path in COMPILE.glob("probe/*/runs.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["jobs_qpu_s"]:
                out[(path.parent.name, r["graph"], r["node"])] = sum(r["jobs_qpu_s"])
    return out


def qpu_split(by: dict, db, fit: dict) -> set:
    """Each calibration's QPU time split into the fixed cost of its jobs (compilation and job start-up), r["fixed_s"],
    and the rest, the sequences, r["seq_s"]. A node run's fixed cost is the probe's for its node on its device; for 04f
    and 04g it is the fixed cost per execution on the device (_compile_fit) times the run's executions. It is never
    more than the run's own QPU time. Returns the (backend, node) pairs without a probe value (counted as sequence)."""
    costs = _probe_costs()
    missing = set()
    for k, rows in by.items():
        other = GRAPH["new" if k == "old" else "old"]
        for r in rows:
            fixed, by_node = 0.0, {}
            for e in rep.events(r["run_id"], r["q"]):
                if not (e.get("kind") == "tool_call" and e.get("tool") == "run_node"):
                    continue
                qpu = float(e.get("qpu_execution_s") or 0)
                if qpu <= 0:
                    continue
                node = e.get("node")
                if node in ADAPTIVE:
                    per = (fit.get(node) or {}).get("by_backend", {}).get(r["backend"])
                    execs = 1
                    if e.get("node_run_id"):
                        execs = len(_fit(db, e["node_run_id"], r["q"])[0].get("length_attempts") or []) or 1
                    cost = None if per is None else per * execs
                else:
                    cost = costs.get((r["backend"], GRAPH[k], node), costs.get((r["backend"], other, node)))
                if cost is None:
                    missing.add((r["backend"], node))
                    continue
                fixed += min(cost, qpu)
                by_node[node] = by_node.get(node, 0.0) + min(cost, qpu)
            r["fixed_s"], r["fixed_by_node"] = fixed, by_node
            r["seq_s"] = max(0.0, (r["qpu_s"] or 0.0) - fixed)
    return missing


def adaptive_section(rows: dict, S: dict, fit: dict | None = None) -> str:
    """The adaptive pi pulse and DRAG in this run: the readout model they use, first-time success, compile vs shots."""
    db = _store()
    new = rows["new"]
    cells = [(r, *_first_04f_readout(r)) for r in new]
    cells = [c for c in cells if c[2]]
    lo = [c for c in cells if c[1] is not None and c[1] < 92]
    hi = [c for c in cells if c[1] is not None and c[1] >= 92]

    def line(cs):
        return (f"{sum(1 for c in cs if c[3] == 0)} of {len(cs)} right first time, "
                f"{sum(c[3] for c in cs)} failed runs" if cs else "—")
    pg = {"chirp": [], "x180": []}
    for r in new:
        for key, node in (("chirp", "IQ_blobs_chirp"), ("x180", "IQ_blobs")):
            cm = _confusion(db, r, node)
            if cm:
                pg[key].append((cm[0][0], cm[1][1]))
    def pp(key, i):
        v = [p[i] for p in pg[key]]
        return f"{100 * median(v):.1f} %" if v else "—"
    with_matrix = sum(1 for r in new if r["status"] == "completed"
                      and (json.load(open(QAB / r["work"] / r["cell"] / "quam_state/state.json"))["qubits"][r["q"]]
                           ["resonator"].get("confusion_matrix")) is not None)
    fails = [c for c in cells if c[3]]
    def last_reason(r):
        why = ""
        for e in rep.events(r["run_id"], r["q"]):
            if (e.get("kind") == "tool_call" and e.get("tool") == "run_node" and e.get("node") == "pi_pulse_adaptive"
                    and e.get("outcome") != "successful"):
                v = (e.get("numerics") or {}).get("failure_reason")
                why = (v.get("value") if isinstance(v, dict) else v) or e.get("error") or ""
        return str(why).split(";")[0][:150]
    fail_text = "; ".join(
        f"{c[0]['backend']} {c[0]['q']}: {c[2]} runs, {c[3]} failed, "
        + ("then completed" if c[0]["status"] == "completed" else f"ended by the agent after its second failed run (“{esc(last_reason(c[0]))}”)")
        for c in fails) or "none"
    fit = fit or _compile_fit(db, new)
    tot = sum(r["qpu_s"] or 0 for r in new)
    fit_rows = "".join(
        f"<tr><td>{esc(n)}</td><td class='num'>{f['default']:.1f} s ({f['n_default']} runs)</td>"
        + (f"<td class='num'>{f['high']:.1f} s ({f['n_high']} runs)</td>" if f["high"] is not None else "<td>—</td>")
        + f"<td class='num'>{f['execs']}</td><td class='num'>{f['qpu'] / 60:.1f} min</td></tr>"
        for n, f in fit.items())
    adaptive_min = sum(f["qpu"] for f in fit.values()) / 60
    n16 = {(r["backend"], r["q"]): r for r in rows["old"]}
    clean = [c for c in cells if c[2] == 1 and c[3] == 0]
    def qmed(cs, src=None):
        v = [((src.get((c[0]["backend"], c[0]["q"])) or {}).get("qpu_s") if src else c[0]["qpu_s"]) for c in cs]
        v = [x for x in v if x is not None]
        return f"{median(v) / 60:.1f} min" if v else "—"
    return (
        "<h3 class='scr'>The adaptive pi pulse and DRAG</h3>"
        "<p class='small muted cap2'>04f and 04g weigh every shot by the readout's confusion matrix "
        "(<span class='mono'>resonator.confusion_matrix</span>). In this run it reaches the state in one piece: qualibrate_ai "
        "records a small numeric array as one value and tinycal's write_state takes it, also where the scrambled state "
        "holds null; 04f and 04g refuse to run without one. IQ blobs prepares |e> with a chirp before 04f (no pi pulse yet) "
        "and again with the x180 after the adaptive DRAG, and T1, T2 echo and RB discriminate on the second. "
        f"{with_matrix} of {len(S['new']['done'])} completed calibrations end with a confusion matrix in the state.</p>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>IQ blobs' readout fidelity before 04f</th>"
        "<th>qubits</th><th>pi_pulse_adaptive</th></tr></thead><tbody>"
        f"<tr><td>≥ 92 %</td><td class='num'>{len(hi)}</td><td>{line(hi)}</td></tr>"
        f"<tr><td>&lt; 92 %</td><td class='num'>{len(lo)}</td><td>{line(lo)}</td></tr>"
        "</tbody></table></div>"
        f"<p class='small muted'>Where 04f failed: {fail_text}. The recipe ends a qubit after a second failed run and never "
        "commits a failed run's values.</p>"
        "<p class='small muted'>The readout's two preparations, medians over the qubits: P(g|g) "
        f"{pp('chirp', 0)} with the chirp and {pp('x180', 0)} with the x180; P(e|e) {pp('chirp', 1)} and {pp('x180', 1)}. "
        "The chirp's |e> carries its own losses (decay during the sweep, an incomplete inversion), which is why the "
        "readout is measured again once the x180 exists.</p>"
        "<h3 class='scr'>QPU time: compilation and shots</h3>"
        "<p class='small muted cap2'>IQCC reports one QPU time per job, compilation included. For the adaptive nodes the "
        "QPU time per program execution does not grow measurably with the number of shots (medians over every run of "
        "each node in tinycal's 6-7 Oct runs on these devices, the same node code): a shot is mostly a thermal reset "
        "wait, so 2,000 of them take a few tenths of a second, less than the run-to-run spread of the rest. Nearly all "
        "of an execution is therefore the fixed cost, the compilation (the estimator runs on the FPGA; ~25 s for 04f and "
        "~10 s for 04g on QOP 3.8, from the nodes' README) and job overhead. 04f reruns at a longer pulse inside one node "
        "run, so an execution count can exceed the run count.</p>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>node</th><th>QPU per execution, 2,000 "
        "shots</th><th>QPU per execution, ≥ 10,000 shots</th><th>executions in n17b</th><th>its QPU in n17b"
        "</th></tr></thead><tbody>"
        f"{fit_rows}</tbody></table></div>"
        + fixed_cost_section(rows, fit)
        + f"<p class='small muted'>Median QPU per qubit where 04f was right first time: {qmed(clean)} "
        f"(n16 {qmed(clean, n16)}).</p>")


def _init_test() -> dict:
    """{(backend, node, variant): [QPU per execution, s]} from compile-20261007/compile_test.py (7 Oct 2026)."""
    out = {}
    for path in COMPILE.glob("*/runs.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["error"] is None and len(r["jobs_qpu_s"]) == 1:
                out.setdefault((path.parent.name, r["node"], r["variant"]), []).append(r["jobs_qpu_s"][0])
    return out


def fixed_cost_section(rows: dict, fit: dict) -> str:
    """Every node's fixed cost per run on each device (the probe), what it added up to in each run, and the
    initialize_qpu test."""
    costs = _probe_costs()
    if not costs:
        return ""
    bk = ("arbel", "gilboa", "qolab")

    def cost(b, node):
        if node in ADAPTIVE:
            return (fit.get(node) or {}).get("by_backend", {}).get(b)
        return costs.get((b, GRAPH["new"], node), costs.get((b, GRAPH["old"], node)))

    def spent(k, node=None):
        return sum((r.get("fixed_by_node") or {}).get(node, 0.0) if node else (r.get("fixed_s") or 0.0) for r in rows[k])

    used = {k: {n for r in rows[k] for n in (r.get("fixed_by_node") or {})} for k in ("old", "new")}
    names = used["old"] | used["new"]

    def worst(n):
        return max((cost(b, n) or 0.0) for b in bk)

    big = sorted((n for n in names if worst(n) >= 2), key=lambda n: -worst(n))
    rest = sorted(n for n in names if n not in big)

    def run_tag(n):
        return "both" if n in used["old"] and n in used["new"] else ("n17b" if n in used["new"] else "n16")

    def cells(n):
        return "".join(f"<td class='num'>{c:.1f} s</td>" if (c := cost(b, n)) is not None else "<td>—</td>" for b in bk)

    trs = "".join(
        f"<tr><td>{esc(n)}</td><td>{run_tag(n)}</td>{cells(n)}<td class='num'>{spent('new', n) / 60:.1f} min</td>"
        f"<td class='num'>{spent('old', n) / 60:.1f} min</td></tr>" for n in big)
    def span(b):
        c = [x for n in rest if (x := cost(b, n)) is not None]
        return f"<td class='num'>{min(c):.1f}–{max(c):.1f} s</td>" if c else "<td>—</td>"

    trs += (f"<tr><td>every other node ({len(rest)}), per run</td><td>—</td>{''.join(span(b) for b in bk)}"
            f"<td class='num'>{sum(spent('new', n) for n in rest) / 60:.1f} min</td>"
            f"<td class='num'>{sum(spent('old', n) for n in rest) / 60:.1f} min</td></tr>")
    trs += (f"<tr><th class='rowh'>fixed cost, all nodes</th><td></td><td colspan='3'></td>"
            f"<td class='num'><b>{spent('new') / 60:.0f} min</b></td><td class='num'><b>{spent('old') / 60:.0f} min</b></td></tr>")
    tot = {k: sum(r["qpu_s"] or 0 for r in rows[k]) for k in ("old", "new")}
    seq = {k: sum(r.get("seq_s") or 0 for r in rows[k]) for k in ("old", "new")}

    t = _init_test()

    def mean(b, node, v):
        x = t.get((b, node, v)) or []
        return sum(x) / len(x) if x else None

    def pair(b, node):
        a, c = mean(b, node, "stock"), mean(b, node, "target")
        return f"{a:.1f} → {c:.1f} s" if a is not None and c is not None else "—"

    init_text = ""
    if t:
        init_text = (
            "<p class='small muted'>The program's size in elements does not set it. initialize_qpu sets every qubit's flux "
            "line and starts every TWPA pump inside the program, so 04f's program for one arbel qubit uses 27 elements. A "
            "version that sets only the target's flux line and its own feedline's TWPA pump in the program, and gives every "
            "other flux line the same offset as its port's DC offset in the config, uses 4. Tested 7 Oct, two runs of each "
            "version alternated, QPU per execution stock → target-only: arbel qA2 (27 elements → 4) 04f "
            f"{pair('arbel', 'pi_pulse_adaptive')}, 04g {pair('arbel', 'drag_adaptive')}; qolab Q3 (8 → 3) 04f "
            f"{pair('qolab', 'pi_pulse_adaptive')}, 04g {pair('qolab', 'drag_adaptive')}. Gilboa's configuration is three "
            "times qolab's (48 elements against 18) and 04f costs the same on both; arbel's fixed cost is higher on every "
            "node in the table above, so its extra seconds on 04f are that backend, not the configuration. The fixed cost "
            "is the estimator program itself.</p>")
    return (
        "<p class='small muted cap2'>Every other node, measured the same way: each node of both graphs run once at one "
        "shot, every other parameter at the graph's preset, on one calibrated qubit per device (7 Oct; "
        + ", ".join(f"{b} {PROBE_QUBIT[b]}" for b in bk)
        + "), so the job's QPU time is its fixed cost: compilation and job start-up. A calibration's estimated fixed cost "
        "is the sum over its node runs that reached the QPU (for 04f and 04g, the fixed cost per execution above times "
        "the run's executions), never more than a run's own QPU time; the rest of its QPU time is sequences. n16 ran "
        "graph 80's nodes at qua-libs b5767c1b, probed here with the n17b library's copies.</p>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>node</th><th>run</th><th>arbel</th>"
        "<th>gilboa</th><th>qolab</th><th>fixed cost in n17b</th><th>in n16</th></tr></thead><tbody>"
        f"{trs}</tbody></table></div>"
        f"<p class='small muted'>QPU time, estimated split: n17b {tot['new'] / 60:.0f} min = {spent('new') / 60:.0f} min "
        f"fixed cost + {seq['new'] / 60:.0f} min sequences; n16 on the same qubits {tot['old'] / 60:.0f} min = "
        f"{spent('old') / 60:.0f} + {seq['old'] / 60:.0f} min.</p>"
        + init_text)


# ----------------------------------------------------------------------------- plots: n17b alone, and n17b against n16
def plots(rows: list) -> str:
    """make_e2_summary.plots with n17b in e2's place."""
    saved = e2s.E2, e2s.LABEL
    e2s.LABEL = LABEL
    try:
        html = e2s.plots(rows)
    finally:
        e2s.E2, e2s.LABEL = saved
    # e2's captions name e2 and its pull; the series is n17b here
    return (html.replace("e2's completed runs", "n17b's completed runs")
            .replace("the pull e2 started from (all three backends pulled 6 Oct 03:53, arbel and gilboa on square readout "
                     "with ≥ 3000 ns depletion)",
                     "the pull n17b started from (all three backends pulled 6 Oct 21:46, arbel and gilboa on square readout "
                     "with ≥ 3000 ns depletion)")
            .replace("the pull e2 started from", "the pull n17b started from")
            .replace("e2's error per gate against n16's on the same qubit (2 Oct, tinycal with qwen3.8-27b); below the "
                     "diagonal e2 is better", "n17b's error per gate against n16's on the same qubit (2 Oct, graph 80); "
                     "below the diagonal n17b is better")
            .replace("e2's measured gate error", "n17b's measured gate error")
            .replace("e2&#x27;s measured gate error", "n17b&#x27;s measured gate error")
            .replace("(stuck in e2)", "(not completed in n17b)").replace("(no valid RB in e2)", "(no valid RB in n17b)")
            .replace("· e2 ", "· n17b "))


def build() -> None:
    configure()
    rows = rep.collect()
    new = [r for r in rows if r["night"] == "new"]
    keys = {(r["backend"], r["q"]) for r in new}
    rows = new + [r for r in rows if r["night"] == "old" and (r["backend"], r["q"]) in keys]
    not_completed(rows)
    by = {k: [r for r in rows if r["night"] == k] for k in ("old", "new")}
    assert {r["run"] for r in by["old"]} == {"n16"} and {r["run"] for r in by["new"]} == {"n17b"}, "wrong work dirs collected"
    S = {k: rep.night_stats(v) for k, v in by.items()}
    db = _store()
    fit = _compile_fit(db, by["new"])
    missing = qpu_split(by, db, fit)
    if missing:
        print("no fixed-cost value for", sorted(missing))
    now = datetime.now().strftime("%d %b %Y %H:%M")
    page = f"""<title>{esc(N17.title)}</title>
<meta name="description" content="{esc(N17.description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{rep.EXTRA_CSS}{summ.SUMMARY_CSS}{e2s.SUMMARY2_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · {esc(N17.eyebrow)} · generated {esc(now)}</div>
<h1 style="margin-top:8px">{esc(N17.h1)}</h1>
<p class="context">{esc(context(S['new'], len(new)))}</p>
{summ.scramble_table(N17)}
<h3 class='scr'>Results</h3>
{results_table(by, S)}
{summ.legend(by['new'], N17)}
<p class="small muted uinote">Click a qubit, here or on a plot, to open its run in tinycal's run viewer (<span class="mono">tinycal ui</span> on this machine, port 8765).</p>
{adaptive_section(by, S, fit)}
{plots(rows)}
{summ.recovery_table(by['new'], N17)}
</div>"""
    OUT.write_text(page)
    for k in ("old", "new"):
        s = S[k]
        fids = [r["gate_fid"] for r in s["valid"] if r["gate_fid"] is not None]
        print(f"{LABEL[k]}: completed {len(s['done'])}/{s['n']} · meaningful {len(s['meaningful'])} · valid RB {len(s['valid'])}, "
              f"median {fpct(median(fids))} · readout {100 * s['ro_med']:.1f} % · judge {s['judge'][0]}/{s['judge'][1]} · "
              f"QPU {s['qpu_tot'] / 60:.0f} min · runs {s['runs']} ({s['not_ok']} failed)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
