#!/usr/bin/env python3
"""Build the 14-17 Sep 2026 framework-comparison report (tinycal vs qua-agents vs dag-walk, with
opus-5 and sonnet-5 at high effort and qwen3.8-27b via OpenRouter) from the cell documents under ~/qab-runs/fwcmp2-*.

    uv run --project ~/code/QM/qua-agents-benchmark python make_fwcmp2_report.py
    (any python >= 3.11 works; `qab` is invoked through uv for the judge prices)

Numbers come only from each cell's result.json (written by the runner and stamped by
`qab inspect-state / validate / accept`), the cell's final quam_state, its run.log or
tinycal events.jsonl, and `qab compare-models` (judge-priced cost). Nothing is typed in
by hand except the incident annotations at the top, which are the operator's log.
Re-run it any time: cells still in flight simply show up as "running".
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path.home() / "qab-runs"
BENCH = Path.home() / "code/QM/qua-agents-benchmark"
TINYCAL_RUNS = Path.home() / "code/QM/tinycal/runs"
OUT = Path(__file__).with_name("2026-09-14-frameworks-opus-sonnet-qwen.html")
PRICES = BENCH / "prices.yaml"
GATES_PER_CLIFFORD = 1.875  # average X/Y pulses per single-qubit Clifford, the qua-libs RB convention


def load_prices() -> dict:
    """model id -> {input, output, cache_read, cache_creation} USD per Mtok, first (current) entry."""
    prices, cur = {}, None
    for line in PRICES.read_text().splitlines():
        m = re.match(r"^  ([^#\s]\S*):\s*$", line)
        if m:
            cur = m.group(1); prices[cur] = {}
            continue
        m = re.match(r"^\s+(input|output|cache_read|cache_creation):\s*([\d.]+)", line)
        if m and cur and m.group(1) not in prices[cur]:
            prices[cur][m.group(1)] = float(m.group(2))
    return prices


def token_cost(tok: dict, model_id: str, prices: dict) -> float | None:
    """The judge's formula: uncached input at the input price, cache reads/writes at theirs, output at its."""
    p = prices.get(model_id)
    if not p or tok.get("input") is None:
        return None
    cr, cw = tok.get("cache_read") or 0, tok.get("cache_creation") or 0
    uncached = max(0, (tok.get("input") or 0) - cr - cw)
    return (uncached * p["input"] + (tok.get("output") or 0) * p["output"] + cr * p.get("cache_read", 0) + cw * p.get("cache_creation", 0)) / 1e6


def gate_fidelity(epc):
    return None if epc is None else 1 - epc / GATES_PER_CLIFFORD

# ----------------------------------------------------------------------------- operator log
# Which qubits the matrix comments call good / bad (matrix.yaml, 2026-09-14).
QUBIT_CLASS = {
    "arbel": {"qB4": "good", "qC2": "good", "qD2": "good", "qD1": "bad", "qC3": "bad", "qA4": "bad"},
    "gilboa": {"qC3": "good", "qD5": "good", "qD4": "good", "qD2": "bad", "qD3": "bad", "qC5": "bad"},
    "qolab": {q: "good" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")},
}
SNAPSHOT_NOTE = {
    "arbel": "pinned 2026-09-09 snapshot (night-arbel-20260909-0110): today's arbel cloud state stores readout as a "
             "QuAM reference and the scramble planner refuses it, as on 2026-09-11",
    "gilboa": "fresh fetch 2026-09-14 17:57 (source sha 9700c720b3b09961)",
    "qolab": "fresh fetch 2026-09-14 23:12 (source sha 305c8f5166ec926a)",
}
# Work dirs that hold no gradable cell and why (they are listed, not scored).
EXCLUDED_DIRS = {
    "fwcmp2-arbel-20260914-1757": "first arbel driver: fetch only, scramble refused the cloud state; no cell ran",
    "fwcmp2-gilboa-20260914-2315": "extra-qubit round (qD4 qD2 qC5) started 23:14 and stopped by the operator at 23:35 "
                                   "after the plan changed to Sonnet; 10 min of one tinycal cell, nothing graded",
}
# Per-cell annotations (work dir, cell dir) -> what an outside event did to it.
CELL_INCIDENTS = {
    ("fwcmp2-qolab-20260915-0800", "qolab-sonnet-5-high"):
        "proxy outage 09:24-09:40: the Claude Code binary the proxy spawns was deleted during a disk clean-up; "
        "all three targets escalated on 502s. Superseded by fwcmp2-qolab-20260915-0950.",
    ("fwcmp2-qolab-20260915-0800", "qolab-tinycal-sonnet-5-high"):
        "proxy outage: every target failed at turn 0 after tinycal's retries. Superseded by fwcmp2-qolab-20260915-0950.",
    ("fwcmp2-arbel-20260915-0150", "arbel-tinycal-sonnet-5-high"):
        "proxy outage killed qB4 at node 9/18 (qC2 and qC3 had already finished). Not re-run.",
    ("fwcmp2-gilboa-20260915-0640", "gilboa-sonnet-5-high"):
        "first attempt died after 1 min on a full disk (09:02); this is the 09:51 re-run.",
    ("fwcmp2-gilboa-20260915-0640", "gilboa-dag-walk"):
        "attempt 1 died on the full disk, attempt 2 on git exit 69 (Xcode licence after a 10:53 Xcode update); "
        "this is the 12:09 re-run.",
    ("fwcmp2-arbel-20260915-0955", "arbel-tinycal-opus-5-high"):
        "qD1 failed at node 15/18 (T2echo) on a proxy-side 400 'Upstream stalled with no data for 90 s, 3rd consecutive stall' "
        "at 12:17; tinycal does not retry 4xx. Resumed from its transcript at 12:50 (tinycal --resume) and completed at 13:47; "
        "the 33 min idle gap is not in its agent or QPU time.",
    ("fwcmp2-arbel-20260915-0150", "arbel-tinycal-sonnet-5-high-qB4-rerun"):
        "qB4 re-run from the scrambled state as its own single-target cell (user request), after the proxy outage killed "
        "the original qB4 target. Attempt of 15 Sep 13:47 stopped by the operator; attempt of 23:54 met the arbel gateway outage; "
        "attempt of 16 Sep 09:15 lost its last node to the proxy stall breaker at 11:30 and was resumed from its transcript at 11:45 "
        "(the 90 s window and the idle gaps are not in its agent/QPU time); completed 12:54.",
    ("fwcmp2-gilboa-20260916-0932", "gilboa-qwen3-8-27b-openrouter.stopped-1542"):
        "qwen3.8-27b via OpenRouter. qD5 (12:23) and qD3 (12:25) each hit two back-to-back 'Connection error' replies from "
        "OpenRouter; qua-agents retries only 429s, so the deterministic policy used both auto-retries in a second and skipped "
        "the target. qC3 hand-wrote f_01 = 7.75 GHz at turn 19 (dispersive-shift sign inverted), which puts the xy element 2.2 GHz "
        "from its LO, over the QOP's 500 MHz IF cap: every node since died at open_qm. A hand-set revert was within the per-path budget, but apply_state_updates only writes as a rider on an unreviewed node observation, and no node could produce one any more. "
        "Cancelled by the operator at 15:42 after 12 identical failures; the document records the run as infra_failed, targets pending. "
        "Set aside as .stopped-1542 and the cell redone from the same scramble in the night of 16-17 Sep.",
    ("fwcmp2-qolab-20260916-2350", "qolab-qwen3-8-27b-openrouter"):
        "ABORTED by the operator at 15:16 on 17 Sep after 10 h 24 min because it was going nowhere: Q2 skipped at 09:20 after ten "
        "auto-retries on output-ceiling failures; Q1 silent since 14:31 on its ninth retry-with-guidance; Q3 had run qubit_spectroscopy 100 "
        "times while already holding the right f_01 (5.0609 GHz vs 5.0610 true) and never committed a node beyond it. 185 completed turns, "
        "218 truncated generations, ~$15 spent, nothing past node 5 on any qubit. Cancelled through the API, so the runner wrote its document; "
        "the qubit-runs are scored as genuine calibration failures (no infrastructure fault), and the cell is not superseded.",
}
# Per-qubit-run causes the document itself cannot carry (a cancelled run leaves its targets 'pending' with no reason).
TARGET_INCIDENTS = {
    ("fwcmp2-gilboa-20260916-0932", "gilboa-qwen3-8-27b-openrouter.stopped-1542", "qD3"): "provider connection error (OpenRouter), skipped by policy",
    ("fwcmp2-gilboa-20260916-0932", "gilboa-qwen3-8-27b-openrouter.stopped-1542", "qD5"): "provider connection error (OpenRouter), skipped by policy",
    ("fwcmp2-gilboa-20260916-0932", "gilboa-qwen3-8-27b-openrouter.stopped-1542", "qC3"): "config fault: IF outside ±500 MHz (self-inflicted f_01), cancelled",
    ("fwcmp2-qolab-20260916-2350", "qolab-qwen3-8-27b-openrouter", "Q1"): "output-ceiling loops, 9 retries, aborted by operator after 10 h",
    ("fwcmp2-qolab-20260916-2350", "qolab-qwen3-8-27b-openrouter", "Q2"): "output-ceiling loops, skipped by policy after 10 retries",
    ("fwcmp2-qolab-20260916-2350", "qolab-qwen3-8-27b-openrouter", "Q3"): "100 qubit_spectroscopy runs with the right f_01, never advanced; aborted by operator after 10 h",
}
SUPERSEDED_DIRS = {"fwcmp2-qolab-20260915-0800"}  # kept as the record, not counted in totals

INCIDENTS = [
    ("14 Sep 17:57", "arbel scramble refused today's cloud state (readout stored as a QuAM reference). Arbel re-launched at 18:02 "
                     "from the pinned 2026-09-09 snapshot, the same fix as on 2026-09-11. No hardware touched."),
    ("14 Sep 18:35", "QM-cloud 503 wave on arbel after 145-230 s queue waits, all three tinycal targets at once; each retried by the model."),
    ("14 Sep 23:35", "extra-qubit round on gilboa (qD4 qD2 qC5) stopped 10 min in when the plan changed to Sonnet; nothing graded."),
    ("15 Sep 09:02", "disk full (212 MiB free of 460 GiB). gilboa qD4 qua-agents/sonnet died after 1 min (proxy ENOSPC, evidence DB "
                     "unopenable); its dag-walk could not open a log. Re-run at 09:51 / 12:09."),
    ("15 Sep 09:24", "the disk clean-up emptied node_modules/@anthropic-ai/claude-code/bin in the proxy project: every model call "
                     "answered 502 'Claude Code process exited unexpectedly (code 127)' until a reinstall at 09:40. "
                     "qolab sonnet round lost (both cells), arbel tinycal/sonnet lost qB4."),
    ("15 Sep 12:07", "Xcode update (10:53) put /usr/bin/git behind a licence prompt (exit 69); the runner's qua-libs pin check failed "
                     "at cell start. gilboa qD4 dag-walk attempt 2 died. Licence accepted 12:11; attempt 3 at 12:09 ran with "
                     "DEVELOPER_DIR pointed at the Command Line Tools."),
    ("16 Sep 19:43", "parameter boundary (user requests): cells started after 19:43 (tinycal) / 19:45 (qua-agents) run with a cap of "
                     "~150 model turns per target (tinycal max_turns 360 -> 150; qua-agents worker_budgets.recursion_limit 360 -> 300); "
                     "qua-agents cells started after 19:58 also run with supervisor.max_auto_retries 2 -> 10 (set by a second session "
                     "sharing the machine). Earlier cells ran with the old values; templates are rendered at cell start."),
    ("16 Sep 17:32", "network/DNS outage from this machine (proxy 500 'Can't reach the API server, ENOTFOUND') 17:32-17:51: "
                     "arbel qD1 qua-agents/sonnet escalated at node 4/18 after three failed model turns. Back by 17:54."),
    ("16 Sep 11:30", "proxy stall breaker again: arbel qB4 tinycal/sonnet re-run lost its last node (RB) to the 400 'Upstream stalled, "
                     "3rd consecutive stall' at node 17/18; resumed at 11:45 after the breaker's 90 s window and completed 12:54."),
    ("15 Sep 23:55", "arbel QOP gateway outage: every QM open on arbel aborted with 'Gateway health is not good enough to answer'; "
                     "the qB4 tinycal re-run gave up in 2 min. gilboa and qolab unaffected."),
    ("16 Sep 09:33", "gilboa third round: qwen3.8-27b (OpenRouter) on the same snapshot and scramble as the opus/sonnet rounds, tinycal first. "
                     "tinycal completed all three qubits by 11:44 (9 transient OpenRouter connection errors, each retried after 5 s)."),
    ("16 Sep 12:23", "qua-agents · qwen3.8-27b: OpenRouter 'Connection error' twice in a row on qD5 (12:23) and qD3 (12:25); the framework retries "
                     "only 429s, so both auto-retries were spent in a second and the policy skipped the target. Provider-side, not calibration."),
    ("16 Sep 15:42", "qua-agents · qwen3.8-27b qC3 cancelled by the operator: at turn 19 the model hand-wrote f_01 = 7.75 GHz (dispersive-shift "
                     "sign inverted), the xy IF went to 2.2 GHz > 500 MHz and every node since failed at open_qm; a revert needs an unreviewed "
                     "observation to ride on and none could be produced. 12 identical failures in 1 h 45 min before the stop."),
    ("17 Sep 15:16", "qolab qua-agents · qwen3.8-27b ABORTED by the operator after 10 h 24 min: going nowhere (Q2 skipped, Q1 stalled on its 9th "
                     "retry, Q3 looping on qubit_spectroscopy with the correct frequency already committed, 218 truncated generations, ~$15). "
                     "Cooperative cancel via the API, document written; qolab handed to the Q4 Q5 Q6 round."),
    ("15 Sep 12:17", "proxy answered a 400 'Upstream stalled with no data for 89998 ms, the 3rd consecutive stall on this session' to "
                     "arbel tinycal/opus on qD1 at node 15/18; tinycal treats 4xx as final, so the target failed after 2 h 25 min."),
]

# ----------------------------------------------------------------------------- helpers
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def local(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%d %b %H:%M")
    except ValueError:
        return ts


def fmt(v, spec="{:.2f}", none="—"):
    return none if v is None else spec.format(v)


def pct(v, digits=2):
    return "—" if v is None else f"{100 * v:.{digits}f}%"


def minutes(s):
    return "—" if s is None else f"{s / 60:.0f} min"


def mtok(n):
    return "—" if n is None else f"{n / 1e6:.2f}M"


def ktok(n):
    """Prompt sizes read better in thousands: 105k rather than 0.11M."""
    return "—" if n is None else f"{n / 1e3:.0f}k"


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def classify_stop(reason: str | None, status: str) -> str:
    """Turn an escalation reason into a short, honest cause."""
    if status == "completed":
        return "ran to completion"
    r = (reason or "").lower()
    if "502" in r or "code 127" in r:
        return "proxy outage (502)"
    if "stalled with no data" in r:
        return "proxy upstream stall (400, 3 × 90 s)"
    if "enotfound" in r or "can't reach the api server" in r:
        return "proxy/network outage (500, DNS)"
    if "gateway health" in r or "qmconnectionerror" in r:
        return "QOP gateway down (cloud)"
    if "enospc" in r or "no space" in r:
        return "disk full"
    if status == "stuck" or "blocked at f_01" in r or "not calibrated" in r or "could not be located" in r \
            or "never found" in r or "no qubit line" in r or "could not locate" in r:
        return "tinycal stuck: f_01 never found"
    if ("iq_blobs" in r or "discrimination" in r or "readout" in r) and ("not completed" in r or "never" in r or "could not" in r):
        return "tinycal stuck: no readout discrimination"
    if status == "escalated" and reason and ("completed" in r or "calibrated" in r):
        return "tinycal stuck: gave up before the end of the graph"
    if "recursion limit" in r:
        # qolab Q2 (qua-agents): the machine could not be opened after f_01 was committed out of the IF window
        return "turn budget exhausted"
    if "dag walk stopped" in r:
        return "default sweep found nothing (2 attempts)"
    if "intermediate frequency" in r or "open_qm" in r or "un-openable" in r:
        return "config fault: IF outside ±500 MHz"
    if status == "failed":
        return "model error"
    if not r:
        return status
    return status + ": " + r.strip().split("\n")[0][:80]


# matrix.yaml model key -> (label used everywhere below, the document's model_id for pricing)
MODEL_KEYS = {
    "opus-5-high": ("opus-5 high", "anthropic:claude-opus-5"),
    "sonnet-5-high": ("sonnet-5 high", "anthropic:claude-sonnet-5"),
    "qwen3-8-27b-openrouter": ("qwen3.8-27b", "qwen/qwen3.8-27b"),  # 16-17 Sep, all three backends, OpenRouter, no effort knob
}
MODEL_LABELS = [label for label, _ in MODEL_KEYS.values()]
MODEL_ID = {label: model_id for label, model_id in MODEL_KEYS.values()}


def cell_kind(cell_dir: str, backend: str):
    """cell dir name -> (framework, model)"""
    n = cell_dir[len(backend) + 1:]
    if n == "dag-walk":
        return "dag-walk", "—"
    fw = "tinycal" if n.startswith("tinycal-") else "qua-agents"
    n = n.removeprefix("tinycal-")
    model = MODEL_KEYS[n][0] if n in MODEL_KEYS else n
    return fw, model


def x180(state_path: Path, target: str):
    try:
        q = json.load(open(state_path))["qubits"][target]
    except Exception:
        return None, None
    ops = g(q, "xy", "operations") or {}
    for name in ("x180_DragCosine", "x180", "x180_DragGaussian"):
        op = ops.get(name)
        if isinstance(op, dict):
            return op.get("length"), op.get("amplitude")
    return None, None


def chip_counters(cell: Path, run_id: str) -> dict:
    txt = ""
    lg = cell / "run.log"
    if lg.exists():
        txt += lg.read_text(errors="replace")
    for ev in glob.glob(str(TINYCAL_RUNS / run_id / "*/events.jsonl")):
        txt += Path(ev).read_text(errors="replace")
    return {
        "empties": txt.count("0 data variables"),
        "503s": txt.count("503 Service"),
        "timeouts": txt.count("timed out after"),
        "config": txt.count("intermediate frequency less than") + txt.count("intermediate frequency greater than"),
    }


def context_series(cell: Path, run_id: str, target: str, fw: str) -> list[int]:
    """Prompt size (input tokens, cached prefix included) of every model call for one target, in order."""
    if fw == "tinycal":
        ev = TINYCAL_RUNS / run_id / target / "events.jsonl"
        if not ev.exists():
            return []
        out = []
        for line in ev.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            u = e.get("usage") if e.get("kind") == "model_turn" else None
            if u and u.get("input_tokens"):
                out.append(int(u["input_tokens"]))
        return out
    db = cell / "evidence/audit-slice.sqlite3"
    if not db.exists():
        return []
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select payload from events where run_id=? and payload like '%model.turn.completed%' order by sequence",
                           (run_id,)).fetchall()
        con.close()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for (raw,) in rows:
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if e.get("kind") != "model.turn.completed":
            continue
        pl = e.get("payload") or {}
        if pl.get("target") != target:
            continue
        u = pl.get("usage") or {}
        if u.get("input_tokens"):
            out.append(int(u["input_tokens"]))
    return out


def median(xs):
    xs = sorted(xs)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def compare_models(work: Path) -> dict:
    """model name -> {tokens, cost, stopped_by} from `qab compare-models <work>/*/result.json`."""
    docs = sorted(glob.glob(str(work / "*/result.json")))
    if not docs:
        return {}
    try:
        out = subprocess.run(["uv", "run", "--frozen", "qab", "compare-models", *docs], cwd=BENCH,
                             capture_output=True, text=True, timeout=600).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"compare-models failed for {work.name}: {exc}", file=sys.stderr)
        return {}
    rows = {}
    for line in out.splitlines():
        m = re.match(r"^\s*\d+\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d,]+|-)\s+(\$[\d.,]+|-)\s+(\d+|-)\s+(\d+|-)", line)
        if m:
            name, stopped, tospec, finished, graph, rb, tokens, cost, turns, reexec = m.groups()
            rows[name] = {
                "stopped_by": None if stopped == "-" else stopped,
                "tokens": None if tokens == "-" else int(tokens.replace(",", "")),
                "cost": None if cost == "-" else float(cost.strip("$").replace(",", "")),
                "graph": float(graph),
            }
    return rows


# ----------------------------------------------------------------------------- collect
def collect():
    prices = load_prices()
    cells, rounds, running = [], {}, []
    for work in sorted(RUNS.glob("fwcmp2-*")):
        if not work.is_dir() or work.name in EXCLUDED_DIRS or work.name.startswith("fwcmp2-probe"):
            continue
        backend = work.name.split("-")[1]
        docs = sorted(glob.glob(str(work / "*/result.json")))
        # cells started but without a document yet (in flight or dead)
        for d in sorted(p for p in work.iterdir() if p.is_dir() and (p / "run.log").exists()
                        and not (p / "result.json").exists() and not any(tag in p.name for tag in (".stopped-", ".gateway-", ".failed"))):
            fw, model = cell_kind(d.name, backend)
            stamp = work.name.split("-", 2)[2]
            key = d.name[len(backend) + 1:].removeprefix("tinycal-").replace("-", "_")
            trun = f"fwcmp2_{backend}_{key}_{stamp}"  # the tinycal run id the driver derives for this cell
            alive = subprocess.run(["pgrep", "-f", f"{work.name}/|{trun}|fwcmp2_{backend}_[a-z0-9_]+_{stamp}"],
                                   capture_output=True, text=True).stdout.strip() != ""
            running.append({"work": work.name, "cell": d.name, "backend": backend, "fw": fw, "model": model, "alive": alive,
                            "mtime": datetime.fromtimestamp(d.stat().st_mtime).strftime("%d %b %H:%M")})
        if not docs:
            continue
        cm = compare_models(work)
        for doc in docs:
            cell = Path(doc).parent
            if any(tag in cell.name for tag in (".stopped-", ".gateway-", ".failed")):
                continue  # set-aside casualties: kept on disk as the record, never scored
            r = json.load(open(doc))
            fw, model = cell_kind(cell.name, backend)
            model_id = MODEL_ID.get(model)
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
                targets.append({
                    "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                    "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, model_id, prices) if model_id else None,
                    "nodes_run": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                    "gate_fid": gate_fidelity(rb.get("error_per_clifford")) if x["status"] == "completed" else None,
                    "ctx": (lambda cs: {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None, "n": len(cs)})(
                        context_series(cell, r.get("run_id") or "", x["target"], fw)),
                    "target": x["target"], "status": x["status"],
                    "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
                    "terminal": x.get("terminal_node"),
                    "rb": rb.get("error_per_clifford"), "readout": ro.get("assignment_fidelity"),
                    "t1": cl.get("t1_s"), "t2e": cl.get("t2echo_s"),
                    "x180_len": L, "x180_amp": A,
                    "ballpark": jd.get("in_ballpark"), "graded": jd.get("graded"), "outside": jd.get("outside") or [],
                    "resid": q.get("residual_detuning_hz"),
                    "reason": x.get("escalation_reason"),
                    "cause": TARGET_INCIDENTS.get((work.name, cell.name, x["target"])) or classify_stop(x.get("escalation_reason"), x["status"]),
                    "turns": g(x, "agent", "turns", "total"),
                })
            qubits = [x["target"] for x in r["targets"]]
            round_key = (backend, "+".join(qubits))
            c = {
                "work": work.name, "cell": cell.name, "backend": backend, "fw": fw, "model": model,
                "qubits": qubits, "round": round_key, "run_id": r.get("run_id"),
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
                "incident": CELL_INCIDENTS.get((work.name, cell.name)),
                "superseded": work.name in SUPERSEDED_DIRS,
            }
            row = cm.get(cell.name, {})
            c["cost"], c["cw_tokens"], c["graph_frac"] = row.get("cost"), row.get("tokens"), row.get("graph")
            c["to_spec"] = 0  # every cell so far: the spec needs node outputs qua-libs does not produce
            c["stopped_by"] = row.get("stopped_by")
            cells.append(c)
            rounds.setdefault(round_key, []).append(c)
    return cells, rounds, running


# ----------------------------------------------------------------------------- html bits
FW_COLOR = {"tinycal": "var(--tinycal)", "qua-agents": "var(--qua)", "dag-walk": "var(--dag)"}


def fw_chip(fw, model):
    label = fw if fw == "dag-walk" else f"{fw} · {model}"
    return f'<span class="chip"><i style="background:{FW_COLOR[fw]}"></i>{esc(label)}</span>'


def status_chip(status):
    cls = {"completed": "ok", "escalated": "warn", "partial": "warn", "stuck": "warn", "failed": "crit"}.get(status, "")
    return f'<span class="st {cls}">{esc(status)}</span>'


def bars(items, value_key, label_fn, fmt_fn, title, unit_note, max_value=None):
    """Horizontal bars, one per item, colored by framework, with a native tooltip per bar."""
    vals = [it[value_key] for it in items if it[value_key] is not None]
    if not vals:
        return ""
    vmax = max_value or max(vals)
    rows = []
    for it in items:
        v = it[value_key]
        w = 0 if v is None else max(2, 100 * v / vmax)
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{label_fn(it)}</div>'
            f'<div class="bar-track"><div class="bar" style="width:{w:.1f}%;background:{FW_COLOR[it["fw"]]}" '
            f'title="{esc(label_fn(it, plain=True))}: {esc(fmt_fn(v))}"></div></div>'
            f'<div class="bar-val num">{esc(fmt_fn(v))}</div></div>')
    return (f'<figure><figcaption><b>{esc(title)}</b> <span class="muted">{esc(unit_note)}</span></figcaption>'
            f'<div class="bars">{"".join(rows)}</div></figure>')


def round_title(key):
    backend, qs = key
    parts = []
    return f"{backend} · " + ", ".join(qs.split("+"))


def mean_rb(c):
    v = [t["rb"] for t in c["targets"] if t["rb"] is not None]
    return sum(v) / len(v) if v else None


def included(c):
    """A scored model cell (dag-walk and superseded cells are out of every comparison table)."""
    return not c["superseded"] and c["fw"] != "dag-walk"


def infra_hit(t):
    """A qubit-run ended by an operator-side incident, not by calibration: excluded from averages."""
    return t["cause"].startswith(("proxy", "provider connection error")) or t["cause"] in ("disk full", "QOP gateway down (cloud)")


def framework_table(cells, model):
    """One row per framework for one model: per-calibration averages over the qubit-runs it attempted."""
    rows = []
    for fw in ("tinycal", "qua-agents"):
        runs = [(c, t) for c in cells if included(c) and c["model"] == model and c["fw"] == fw
                for t in c["targets"] if not infra_hit(t)]
        done = [(c, t) for c, t in runs if t["status"] == "completed"]
        if not runs:
            rows.append(f"<tr><td>{fw_chip(fw, model)}</td><td colspan='8' class='muted'>no qubit-runs yet</td></tr>")
            continue
        n = len(done) or 1
        fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
        med = fids[len(fids) // 2] if fids else None
        tot = lambda k: sum((t[k] or 0) for _, t in runs)
        cost = sum((t["cost"] or 0) for _, t in runs)
        rng = f"{pct(fids[0])} – {pct(fids[-1])}" if fids else "—"
        rows.append(
            f"<tr><td>{fw_chip(fw, model)}</td><td class='num'>{len(done)}/{len(runs)}</td>"
            f"<td class='num'>{pct(med)}<br><span class='small muted'>{rng}</span></td>"
            f"<td class='num'>{minutes(tot('model_s') / n)}</td><td class='num'>{minutes(tot('qpu_s') / n)}</td>"
            f"<td class='num'>{minutes(tot('queue_s') / n)}</td><td class='num'>${cost / n:.2f}</td>"
            f"<td class='num'>{tot('turns') / n:.0f}</td><td class='num'>{tot('nodes_run') / n:.0f} ({tot('reruns') / n:.0f})</td></tr>")
    return ('<table class="grid"><thead><tr><th>framework</th><th>calibrations completed / attempted</th>'
            '<th>single-qubit gate fidelity<br><span class="small">median · min – max</span></th><th>agent time / calibration</th>'
            '<th>QPU time / calibration</th><th>queue wait / calibration</th><th>judge cost / calibration</th>'
            '<th>turns / calibration</th><th>node runs (re-runs) / calibration</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def pivot_table(cells):
    """Metrics as rows, framework × model as columns: the at-a-glance comparison."""
    cols = [(model, fw) for model in MODEL_LABELS for fw in ("tinycal", "qua-agents")]
    stats = []
    for model, fw in cols:
        runs = [(c, t) for c in cells if included(c) and c["model"] == model and c["fw"] == fw
                for t in c["targets"] if not infra_hit(t)]
        done = [(c, t) for c, t in runs if t["status"] == "completed"]
        n = len(done) or 1
        fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
        tot = lambda k: sum((t[k] or 0) for _, t in runs)
        tokt = lambda k: sum((t["tokens"].get(k) or 0) for _, t in runs)
        # per backend: qubit -> did this framework×model ever calibrate it (a re-run that finished counts)
        per_dev = {}
        for c, t in runs:
            done_here = per_dev.setdefault(c["backend"], {}).get(t["target"], False)
            per_dev[c["backend"]][t["target"]] = done_here or t["status"] == "completed"
        def qubit_list(qs):
            return ", ".join(q if ok else f"<span class='qfail' title='not calibrated'>{q}</span>" for q, ok in sorted(qs.items()))
        stats.append({
            "qubits": "<br>".join(f"<b class='dev'>{b}</b> {qubit_list(qs)}" for b, qs in sorted(per_dev.items())) or "—",
            "done": f"{len(done)}/{len(runs)} ({100 * len(done) / len(runs):.0f}%)",
            "fid": (pct(fids[len(fids) // 2]) + f"<br><span class='small muted'>{pct(fids[0])} – {pct(fids[-1])}</span>") if fids else "—",
            "agent": minutes(tot("model_s") / n), "qpu": minutes(tot("qpu_s") / n), "queue": minutes(tot("queue_s") / n),
            "cost": f"${sum((t['cost'] or 0) for _, t in runs) / n:.2f}",
            "turns": f"{tot('turns') / n:.0f}", "nodes": f"{tot('nodes_run') / n:.0f} ({tot('reruns') / n:.0f})",
            "ctx_med": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["median"] for _, t in runs if t["ctx"]["median"]]),
            "ctx_end": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["end"] for _, t in done if t["ctx"]["end"]]),
            "tin": mtok(tokt("input") / n), "tout": mtok(tokt("output") / n),
            "tcache": f"{mtok(tokt('cache_read') / n)} / {mtok(tokt('cache_creation') / n)}",
        } if runs else None)
    rows = [("qubits measured", "qubits"), ("calibrations completed / attempted", "done"),
            ("single-qubit gate fidelity, median (min – max)", "fid"), ("agent time / calibration", "agent"),
            ("QPU time / calibration", "qpu"), ("queue wait / calibration", "queue"), ("judge cost / calibration", "cost"),
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
    head = ('<thead><tr><th></th>' + "".join(f'<th colspan="2" class="grp">{esc(m)}</th>' for m in MODEL_LABELS) + '</tr>'
            '<tr><th></th>' + "".join(f"<th>{fw_chip(fw, model).replace(' · ' + model, '')}</th>" for model, fw in cols) + '</tr></thead>')
    return f'<div class="scroll"><table class="grid pivot">{head}<tbody>' + "".join(body) + "</tbody></table></div>"


# ----------------------------------------------------------------------------- what fills the context
CHARS_PER_TOKEN = 3.6  # rough average for this English-plus-numbers prose; every "≈ tokens" below is chars / 3.6, not measured
AUDIT_DB = Path.home() / "qua-agents-db/audit.sqlite3"
# The like-for-like gilboa round, target qD3: one tinycal and one qua-agents cell per model.
CONTEXT_REF = [
    ("opus-5 high", "fwcmp2-gilboa-20260914-1757", "gilboa-tinycal-opus-5-high", "gilboa-opus-5-high"),
    ("sonnet-5 high", "fwcmp2-gilboa-20260914-2340", "gilboa-tinycal-sonnet-5-high", "gilboa-sonnet-5-high"),
    ("qwen3.8-27b", "fwcmp2-gilboa-20260916-0932", "gilboa-tinycal-qwen3-8-27b-openrouter", "gilboa-qwen3-8-27b-openrouter"),
]
CONTEXT_TARGET = "qD3"


def _run_id(work: str, cell: str):
    try:
        return json.load(open(RUNS / work / cell / "result.json")).get("run_id")
    except Exception:
        return None


def _tinycal_context(run_id: str):
    """Character count per block type in the final transcript, image count, and the measured prompt size over the run."""
    d = TINYCAL_RUNS / run_id / CONTEXT_TARGET
    try:
        tr = json.load(open(d / "transcript.json"))
        ev = [json.loads(l) for l in open(d / "events.jsonl")]
    except Exception:
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
    except Exception:
        pass
    turns = [e for e in ev if e.get("kind") == "model_turn" and (e.get("usage") or {}).get("input_tokens")]
    series = [(e["turn"], e["usage"]["input_tokens"]) for e in turns]
    # Anthropic prices a 1500x900 px figure at about w*h/750 tokens; OpenAI-compatible endpoints vary, so this is the same estimate for all
    parts["figures kept in context"] = images * 1800 * CHARS_PER_TOKEN
    return {"parts": parts, "images": images, "series": series, "turns": len(turns)}


def _qua_agents_context(run_id: str):
    """The audit's own payload_sizes for the last worker turn on the reference target, plus the measured prompt-size series."""
    if not AUDIT_DB.exists():
        return None
    con = sqlite3.connect(f"file:{AUDIT_DB}?mode=ro", uri=True)
    rows = con.execute(
        "select json_extract(payload,'$.payload') from events where run_id=? and json_extract(payload,'$.kind')='model.turn.completed' "
        "and json_extract(payload,'$.payload.target')=? order by sequence", (run_id, CONTEXT_TARGET)).fetchall()
    con.close()
    if not rows:
        return None
    turns = [json.loads(r[0]) for r in rows]
    last = turns[-1]
    ps = last.get("payload_sizes") or {}
    static, dynamic = ps.get("static_by_key") or {}, ps.get("dynamic_by_key") or {}
    parts = {
        "node catalogue (static)": static.get("node_catalog", 0),
        "graph guide + writable-state catalogue + connectivity (static)": sum(v for k, v in static.items() if k != "node_catalog"),
        "target state + committed state": dynamic.get("target_state", 0) + dynamic.get("committed_state", 0),
        "search notes (the model's own)": dynamic.get("search_notes", 0),
        "recent node summaries + history": dynamic.get("nodes", 0) + dynamic.get("history", 0),
        "everything else dynamic": sum(v for k, v in dynamic.items() if k not in ("target_state", "committed_state", "search_notes", "nodes", "history")),
        "figures attached": (last.get("image_token_estimate") or 0) * CHARS_PER_TOKEN,
    }
    series = [(t.get("turn_index"), (t.get("usage") or {}).get("input_tokens")) for t in turns if (t.get("usage") or {}).get("input_tokens")]
    u = last.get("usage") or {}
    return {"parts": parts, "series": series, "turns": len(turns), "cache_creation": u.get("cache_creation_tokens"),
            "cached": u.get("cache_read_tokens"), "images_tokens": last.get("image_token_estimate")}


def _ktok(chars):
    return f"{chars / CHARS_PER_TOKEN / 1000:.0f}k"


def context_section():
    cols, tiny, qua = [], {}, {}
    for model, work, tcell, qcell in CONTEXT_REF:
        cols.append(model)
        t_id, q_id = _run_id(work, tcell), _run_id(work, qcell)
        tiny[model] = _tinycal_context(t_id) if t_id else None
        qua[model] = _qua_agents_context(q_id) if q_id else None

    def table(data, framework):
        keys = []
        for v in data.values():
            if v:
                for k in v["parts"]:
                    if k not in keys:
                        keys.append(k)
        head = "".join(f"<th>{esc(m)}</th>" for m in cols)
        body = []
        for k in keys:
            tds = "".join(f"<td class='num'>{data[m]['parts'][k] / 1000:.0f}k chars ≈ {_ktok(data[m]['parts'][k])} tok</td>" if data[m] else "<td>—</td>" for m in cols)
            body.append(f"<tr><th class='rowh'>{esc(k)}</th>{tds}</tr>")
        tds = "".join(f"<td class='num'>{sum(data[m]['parts'].values()) / 1000:.0f}k chars ≈ {_ktok(sum(data[m]['parts'].values()))} tok</td>" if data[m] else "<td>—</td>" for m in cols)
        body.append(f"<tr><th class='rowh'>sum of the parts (estimate)</th>{tds}</tr>")
        def meas(v):
            if not v or not v["series"]:
                return "—"
            s = v["series"]
            mid = s[len(s) // 2]
            return f"turn {s[0][0]}: {s[0][1] / 1000:.0f}k · turn {mid[0]}: {mid[1] / 1000:.0f}k · turn {s[-1][0]}: {s[-1][1] / 1000:.0f}k"
        body.append(f"<tr><th class='rowh'><b>measured prompt, tokens</b> (first · middle · last turn)</th>" + "".join(f"<td class='num'>{meas(data[m])}</td>" for m in cols) + "</tr>")
        if framework == "tinycal":
            body.append("<tr><th class='rowh'>figures in the final prompt</th>" + "".join(f"<td class='num'>{data[m]['images']}</td>" if data[m] else "<td>—</td>" for m in cols) + "</tr>")
        else:
            body.append("<tr><th class='rowh'>last turn: cached / written to cache, tokens</th>" + "".join(
                f"<td class='num'>{(data[m]['cached'] or 0) / 1000:.0f}k / {(data[m]['cache_creation'] or 0) / 1000:.0f}k</td>" if data[m] else "<td>—</td>" for m in cols) + "</tr>")
        return (f'<div class="scroll"><table class="grid pivot"><thead><tr><th>{framework}, gilboa {CONTEXT_TARGET}, final prompt</th>{head}</tr></thead>'
                f"<tbody>{''.join(body)}</tbody></table></div>")

    return f"""
<h3>What fills the context</h3>
<p class="small muted">Prompt composition at the end of the like-for-like gilboa round, target {CONTEXT_TARGET}, one cell per framework and model.
Characters are measured: tinycal from the final transcript.json of the run, qua-agents from the payload_sizes its audit records on every
worker turn. "≈ tok" is characters ÷ {CHARS_PER_TOKEN}, an estimate; figures are counted at ~1,800 tokens each (a 1500×900 px PNG under
Anthropic's w·h/750 rule, used for every model here). The <b>measured prompt</b> row is the real input-token count the provider reported,
cached prefix included. It runs up to 2× the sum of the parts on the tinycal side: node result text is mostly digits, which tokenise at
nearer 2 characters per token than 3.6, and the tool schemas and message framing are not in the parts at all. Read the parts as shares,
and the measured row as the size.</p>
{table(tiny, "tinycal")}
<p class="small">tinycal keeps the whole conversation: nothing is summarised and only figures beyond its 16-image window are dropped, so the
prompt grows by roughly 3–4k tokens per turn and a 60–80-turn qubit ends at 150–250k. Almost all of it is cached prefix, which is why it stays
cheap, but every call still reads it. The biggest block is the node result text (fit tables and numerics for every node run), then the figures.
The recipe and catalogue in the system prompt are a rounding error.</p>
{table(qua, "qua-agents")}
<p class="small">qua-agents rebuilds the prompt from a template every turn, so it plateaus instead of growing: the static part (node catalogue,
graph guide, writable-state catalogue) is identical on every call and sits behind a cache breakpoint; the dynamic part (target state, the model's
search notes, recent node summaries) is rewritten each turn, and the "written to cache" figure is what that costs. The node catalogue alone is
larger than everything tinycal carries except its node results. Reduction levers, both framework-side: tinycal could summarise node results
older than ~10 turns and cap figures at 8; qua-agents could send a short catalogue index with per-node detail on demand and order the dynamic
block so its stable parts sit before the breakpoint.</p>
"""


# ----------------------------------------------------------------------------- pins
CHECKOUTS = {  # what the drivers pointed at; the sha a new run must match to be comparable
    "qua-agents-benchmark (judge, workload, drivers)": BENCH,
    "tinycal": Path.home() / "code/QM/tinycal",
    "qua-agents (integration checkout)": Path.home() / "qab-runs/qua-agents-integration",
    "qua-libs (frozen checkout, both frameworks)": Path.home() / "qab-runs/qua-libs-latest",
}


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def pins_table(cells):
    """Every sha and digest a new cell must reproduce, read from the documents where they are recorded."""
    from collections import Counter, defaultdict
    fw_sha, libs, spec, snap, srcdir = defaultdict(Counter), Counter(), Counter(), defaultdict(Counter), defaultdict(set)
    for c in cells:
        try:
            r = json.load(open(RUNS / c["work"] / c["cell"] / "result.json"))
        except Exception:  # noqa: BLE001
            continue
        fp = r.get("fingerprint") or {}
        fw_sha[c["fw"] if c["fw"] != "dag-walk" else "qua-agents"][str(fp.get("git_sha", ""))[:7]] += 1
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
        sha = _git(path, "rev-parse", "--short", "HEAD")
        dirty = _git(path, "status", "--short").replace("\n", "; ") or "clean"
        recorded = ""
        if label.startswith("tinycal"):
            recorded = "document fingerprint.git_sha: " + fmt_counter(fw_sha["tinycal"])
        elif label.startswith("qua-agents ("):
            recorded = "document fingerprint.git_sha: " + fmt_counter(fw_sha["qua-agents"])
        elif label.startswith("qua-libs"):
            recorded = "document fingerprint.calibration_content: " + fmt_counter(libs) + "; deps.yaml pin " + esc(
                (_git(Path.home() / "qab-runs/qua-agents-integration", "show", "HEAD:benchmarks/deps.yaml") or "").split("qua_libs_ref:")[-1].strip().split("\n")[0][:7])
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


def hard_cases(cells):
    """One row per qubit × model where at least one framework did not calibrate (infrastructure casualties excluded)."""
    by = {}
    for c in cells:
        if not included(c):
            continue
        for t in c["targets"]:
            if infra_hit(t):
                continue
            by.setdefault((c["backend"], t["target"], c["model"]), {}).setdefault(c["fw"], []).append((c, t))
    rows = []
    for key, fws in by.items():
        if "tinycal" not in fws or "qua-agents" not in fws:
            continue
        ct, tt = pick_run(fws["tinycal"])
        cq, tq = pick_run(fws["qua-agents"])
        okt, okq = tt["status"] == "completed", tq["status"] == "completed"
        if okt and okq:
            continue
        if okt and not okq:
            verdict = f"tinycal: finished it at {pct(tt['gate_fid'])}"
        elif okq and not okt:
            verdict = f"qua-agents: finished it at {pct(tq['gate_fid'])}"
        else:
            a, b = tt["cost"] or 0, tq["cost"] or 0
            cheaper = "tinycal" if a < b else "qua-agents"
            ratio = (max(a, b) / min(a, b)) if min(a, b) > 0 else None
            nt, nq = tt["nodes"] or 0, tq["nodes"] or 0
            further = "tinycal" if nt > nq else ("qua-agents" if nq > nt else None)
            verdict = "neither"
            if further:
                verdict += f"; {further} got further ({max(nt, nq)}/18 nodes)"
            if ratio:
                verdict += f"; {cheaper}'s failure cost {ratio:.0f}× less"
        def cell(t):
            fid = f" · {pct(t['gate_fid'])}" if t["gate_fid"] is not None else ""
            return (f"{status_chip(t['status'])} {t['nodes'] or 0}/18{fid}<br><span class='small'>{minutes(t['model_s'])} agent · "
                    f"{fmt(t['cost'], '${:.2f}')}</span><br><span class='small muted'>{esc(t['cause'].replace('tinycal stuck: ', ''))}</span>")
        rank = {(False, False): 0, (False, True): 1, (True, False): 2}[(okt, okq)]
        rows.append((rank, key, cell(tt), cell(tq), verdict))
    rows.sort(key=lambda r: (r[0], r[1]))
    body = "".join(f"<tr><td class='mono'>{esc(k[0])} {esc(k[1])}</td><td>{esc(k[2].replace(' high', ''))}</td>"
                   f"<td>{a}</td><td>{b}</td><td>{esc(v)}</td></tr>" for _, k, a, b, v in rows)
    if not rows:
        body = "<tr><td colspan='5' class='muted'>none</td></tr>"
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>qubit</th><th>model</th>'
            '<th style="border-left:2px solid var(--tinycal)">tinycal</th><th style="border-left:2px solid var(--qua)">qua-agents</th>'
            '<th>which framework did better, and why</th></tr></thead><tbody>' + body + "</tbody></table></div>")


def pick_run(entries):
    """When a qubit was run more than once by the same framework+model (a from-scratch re-run), show the best-founded one."""
    return sorted(entries, key=lambda e: (e[1]["status"] == "completed", not infra_hit(e[1]), len(e[0]["qubits"]) == 1))[-1]


def qubit_side_by_side(cells, model):
    by = {}
    for c in cells:
        if not included(c) or c["model"] != model:
            continue
        for t in c["targets"]:
            by.setdefault((c["backend"], t["target"]), {}).setdefault(c["fw"], []).append((c, t))
    rows = []
    for key in sorted(by):
        cells_html = [f"<td class='mono'>{esc(key[0])} {esc(key[1])}</td>"]
        for fw in ("tinycal", "qua-agents"):
            if fw not in by[key]:
                cells_html.append("<td colspan='5' class='muted'>—</td>")
                continue
            c, t = pick_run(by[key][fw])
            tag = " <span class='small muted'>(re-run)</span>" if "rerun" in c["cell"] else ""
            cause = "" if t["status"] == "completed" else f"<br><span class='small muted'>{esc(t['cause'])}</span>"
            cells_html.append(f"<td>{status_chip(t['status'])}{tag}{cause}</td><td class='num'>{pct(t['gate_fid'])}</td>"
                              f"<td class='num'>{minutes(t['model_s'])}</td><td class='num'>{minutes(t['qpu_s'])}</td>"
                              f"<td class='num'>{fmt(t['cost'], '${:.2f}')}</td>")
        rows.append("<tr>" + "".join(cells_html) + "</tr>")
    head = ('<thead><tr><th rowspan="2">qubit</th><th colspan="5" style="border-left:2px solid var(--tinycal)">tinycal</th>'
            '<th colspan="5" style="border-left:2px solid var(--qua)">qua-agents</th></tr>'
            '<tr><th>status</th><th>gate fidelity</th><th>agent</th><th>QPU</th><th>cost</th>'
            '<th>status</th><th>gate fidelity</th><th>agent</th><th>QPU</th><th>cost</th></tr></thead>')
    return f'<div class="scroll"><table class="grid small">{head}<tbody>' + "".join(rows) + "</tbody></table></div>"


def stuck_table(cells):
    rows = []
    for c in cells:
        if c["fw"] == "dag-walk":
            continue
        for t in c["targets"]:
            if t["status"] == "completed":
                continue
            reason = (t["reason"] or "").strip().replace("\n", " ")
            rows.append(
                f"<tr{' class=sup' if c['superseded'] else ''}><td>{esc(c['backend'])}</td><td class='mono'>{esc(t['target'])}</td>"
                f"<td>{fw_chip(c['fw'], c['model'])}</td><td>{status_chip(t['status'])}</td>"
                f"<td class='num'>{t['nodes'] if t['nodes'] is not None else 0}/{t['graph'] or 18}</td>"
                f"<td class='mono'>{esc(t['terminal'] or '')}</td><td>{esc(t['cause'])}</td>"
                f"<td class='small'>{esc(reason[:220])}{'…' if len(reason) > 220 else ''}</td></tr>")
    return ('<table class="grid"><thead><tr><th>backend</th><th>qubit</th><th>cell</th><th>status</th><th>nodes</th>'
            '<th>last node</th><th>cause</th><th>the cell\'s own words</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def per_qubit(cells):
    out = []
    by_bq = {}
    for c in cells:
        if c["fw"] == "dag-walk":
            continue
        for t in c["targets"]:
            by_bq.setdefault((c["backend"], t["target"]), []).append((c, t))
    order = {"tinycal": 0, "qua-agents": 1}
    for backend in ("arbel", "gilboa", "qolab"):
        out.append(f"<h3 id='q-{backend}'>{backend}</h3>")
        keys = [k for k in by_bq if k[0] == backend]
        for key in sorted(keys, key=lambda k: k[1]):
            entries = sorted(by_bq[key], key=lambda e: (e[0]["model"], order[e[0]["fw"]], len(e[0]["qubits"]) == 1, e[0]["superseded"]))
            out.append(f"<h4>{esc(key[1])}</h4>")
            out.append('<div class="scroll"><table class="grid small"><thead><tr><th>cell (round)</th><th>status</th><th>nodes</th>'
                       '<th>gate fidelity<br><span class="small">(error / Clifford)</span></th><th>readout assign.</th><th>T1 / T2echo</th><th>x180 len @ amp</th>'
                       '<th>ballpark</th><th>resid. detuning</th><th>agent</th><th>QPU</th><th>queue</th><th>turns</th><th>node runs (re)</th>'
                       '<th>context median / end</th><th>tokens in / out</th><th>judge cost</th><th>503 · timeouts · empties</th><th>what stopped it</th></tr></thead><tbody>')
            for c, t in entries:
                rnd = "3-qubit" if len(c["qubits"]) == 3 else ("re-run" if "rerun" in c["cell"] else "single")
                misses = "<br>".join(f"<span class='small muted'>{esc(o.split(' came back')[0].split('.', 1)[1] if '.' in o else o)}</span>" for o in t["outside"])
                cls = ' class="sup"' if c["superseded"] else ""
                tok = t["tokens"]
                out.append(
                    f"<tr{cls}><td>{fw_chip(c['fw'], c['model'])} <span class='muted small'>{rnd}</span></td>"
                    f"<td>{status_chip(t['status'])}</td><td class='num'>{t['nodes'] if t['nodes'] is not None else 0}/{t['graph'] or 18}</td>"
                    f"<td class='num'>{pct(t['gate_fid'])}<br><span class='small muted'>{pct(t['rb'])}</span></td><td class='num'>{pct(t['readout'], 1)}</td>"
                    f"<td class='num'>{fmt(t['t1'] and t['t1'] * 1e6, '{:.1f}')} / {fmt(t['t2e'] and t['t2e'] * 1e6, '{:.1f}')} µs</td>"
                    f"<td class='num'>{fmt(t['x180_len'], '{:.0f}')} ns @ {fmt(t['x180_amp'], '{:.3f}')}</td>"
                    f"<td class='num'>{t['ballpark'] if t['ballpark'] is not None else '—'}/{t['graded'] or 4}{('<br>' + misses) if misses else ''}</td>"
                    f"<td class='num'>{fmt(t['resid'] and t['resid'] / 1e3, '{:+.1f} kHz')}</td>"
                    f"<td class='num'>{minutes(t['model_s'])}</td><td class='num'>{minutes(t['qpu_s'])}</td><td class='num'>{minutes(t['queue_s'])}</td>"
                    f"<td class='num'>{fmt(t['turns'], '{}')}</td><td class='num'>{fmt(t['nodes_run'], '{}')} ({fmt(t['reruns'], '{}')})</td>"
                    f"<td class='num'>{ktok(t['ctx']['median'])} / {ktok(t['ctx']['end'])}</td>"
                    f"<td class='num'>{mtok(tok.get('input'))} / {mtok(tok.get('output'))}</td><td class='num'>{fmt(t['cost'], '${:.2f}')}</td>"
                    f"<td class='num'>{c['chip']['503s']} · {c['chip']['timeouts']} · {c['chip']['empties']}</td>"
                    f"<td>{esc(t['cause'])}{(' <span class=tag>incident</span>') if c['incident'] else ''}</td></tr>")
            out.append("</tbody></table></div>")
    return "".join(out)


def run_ids(cells, running):
    rows = []
    for c in sorted(cells, key=lambda c: (c["work"], c["cell"])):
        gui = (f'<a href="http://127.0.0.1:8080/#/runs">{esc(c["run_id"])}</a>' if c["fw"] != "tinycal" and c["run_id"] else esc(c["run_id"]))
        tc = f"~/code/QM/tinycal/runs/{c['run_id']}" if c["fw"] == "tinycal" else ""
        rows.append(f"<tr{' class=sup' if c['superseded'] else ''}><td class='mono small'>{esc(c['work'])}</td><td class='mono small'>{esc(c['cell'])}</td>"
                    f"<td>{fw_chip(c['fw'], c['model'])}</td><td class='mono'>{'+'.join(c['qubits'])}</td><td class='mono small'>{gui}</td>"
                    f"<td class='mono small'>{esc(tc)}</td><td class='num'>{esc(c['started'])} → {esc(c['ended'])}</td><td>{status_chip(c['status'])}</td></tr>")
    for r in running:
        state = "<span class='st run'>running</span>" if r["alive"] else "<span class='st crit'>stopped, not graded</span>"
        note = "in flight" if r["alive"] else "stopped by the operator before it finished; no result.json"
        rows.append(f"<tr><td class='mono small'>{esc(r['work'])}</td><td class='mono small'>{esc(r['cell'])}</td><td>{fw_chip(r['fw'], r['model'])}</td>"
                    f"<td></td><td class='muted'>{note}</td><td></td><td class='num'>last write {esc(r['mtime'])}</td><td>{state}</td></tr>")
    for d, why in EXCLUDED_DIRS.items():
        rows.append(f"<tr class='sup'><td class='mono small'>{esc(d)}</td><td colspan='7' class='small muted'>{esc(why)}</td></tr>")
    return ('<div class="scroll"><table class="grid small ids"><thead><tr><th>work dir (~/qab-runs/)</th><th>cell dir</th><th>cell</th><th>qubits</th>'
            '<th>run id</th><th>tinycal run dir</th><th>started → ended</th><th>status</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- page
CSS = """
:root { color-scheme: light;
  --ground:#f5f7f7; --surface:#fff; --surface-2:#eef2f2; --ink:#151a1d; --ink-2:#4b555b; --muted:#7f8a90;
  --line:#d8dfe1; --line-strong:#b9c3c7; --accent:#0f766e; --accent-ink:#0b5f59;
  --warn:#b45309; --warn-bg:#fdf3e6; --crit:#b91c1c; --crit-bg:#fdecec; --good:#0f766e; --good-bg:#e6f4f2; --run-bg:#e8eefb;
  --qua:#2a78d6; --tinycal:#1baf7a; --dag:#7f8a90; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --ground:#121516; --surface:#1a1e20; --surface-2:#22282b; --ink:#edf1f2; --ink-2:#b6bfc4; --muted:#8a959b;
  --line:#2c3437; --line-strong:#414c50; --accent:#2dbfb2; --accent-ink:#5fd3c8;
  --warn:#f0a24a; --warn-bg:#2d2214; --crit:#f07575; --crit-bg:#331a1a; --good:#5fd3c8; --good-bg:#16302d; --run-bg:#1d2a3d;
  --qua:#3987e5; --tinycal:#199e70; --dag:#8a959b; } }
:root[data-theme="dark"] { color-scheme: dark;
  --ground:#121516; --surface:#1a1e20; --surface-2:#22282b; --ink:#edf1f2; --ink-2:#b6bfc4; --muted:#8a959b;
  --line:#2c3437; --line-strong:#414c50; --accent:#2dbfb2; --accent-ink:#5fd3c8;
  --warn:#f0a24a; --warn-bg:#2d2214; --crit:#f07575; --crit-bg:#331a1a; --good:#5fd3c8; --good-bg:#16302d; --run-bg:#1d2a3d;
  --qua:#3987e5; --tinycal:#199e70; --dag:#8a959b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--ground); color:var(--ink); font-family:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif; font-size:16px; line-height:1.5; }
.page { max-width:1180px; margin:0 auto; padding:36px 24px 72px; }
h1,h2,h3,h4 { font-family:"Bricolage Grotesque","Source Sans 3",Helvetica,Arial,sans-serif; margin:0; text-wrap:balance; }
h1 { font-size:36px; font-weight:700; line-height:1.1; letter-spacing:-0.01em; }
h2 { font-size:23px; font-weight:600; margin-top:52px; padding-top:16px; border-top:1px solid var(--line-strong); }
h3 { font-size:18px; font-weight:600; margin-top:28px; }
h4 { font-size:16px; font-weight:600; margin-top:18px; }
p { max-width:76ch; margin:10px 0; }
.eyebrow { font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.lede { font-size:18px; color:var(--ink-2); max-width:70ch; }
.mono,.num { font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-variant-numeric:tabular-nums; }
.num { font-size:.92em; white-space:nowrap; }
.small { font-size:.86em; } .muted { color:var(--muted); }
a { color:var(--accent-ink); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.tile .v { font-family:"Bricolage Grotesque",sans-serif; font-size:26px; font-weight:700; line-height:1.1; }
.tile .k { color:var(--muted); font-size:13px; margin-top:4px; }
table.grid { width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); font-size:14px; }
table.grid th { text-align:left; font-weight:600; color:var(--ink-2); background:var(--surface-2); padding:8px 9px; border-bottom:1px solid var(--line-strong); position:sticky; top:0; }
table.grid td { padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
table.grid.small { font-size:13px; }
tr.sup td { opacity:.55; }
table.ids td { white-space:nowrap; }
table.pivot .dev{font-family:inherit;font-weight:700;letter-spacing:.02em} .pivot .qfail{color:var(--crit);font-weight:600}
.pivot th.rowh { text-align:left; font-weight:600; color:var(--ink-2); background:var(--surface-2); position:static; width:24%; }
table.pivot td.wrap { white-space:normal; font-size:12px; line-height:1.35; }
table.pivot { table-layout:fixed; }
table.pivot th.grp { text-align:center; font-size:15px; border-left:1px solid var(--line-strong); }
table.pivot td { text-align:left; vertical-align:top; }
table.pivot thead tr:nth-child(2) th { border-left:1px solid var(--line); }
.scroll { overflow-x:auto; }
.chip { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
.chip i { width:10px; height:10px; border-radius:3px; display:inline-block; }
.st { display:inline-block; padding:1px 7px; border-radius:6px; font-size:12px; font-family:"JetBrains Mono",monospace; background:var(--surface-2); }
.st.ok { background:var(--good-bg); color:var(--good); } .st.warn { background:var(--warn-bg); color:var(--warn); }
.st.crit { background:var(--crit-bg); color:var(--crit); } .st.run { background:var(--run-bg); }
.tag { display:inline-block; padding:0 6px; border-radius:6px; font-size:11px; background:var(--warn-bg); color:var(--warn); font-family:"JetBrains Mono",monospace; }
figure { margin:20px 0; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
figcaption { margin-bottom:10px; }
.bars { display:grid; gap:5px; }
.bar-row { display:grid; grid-template-columns:minmax(200px,34%) 1fr 72px; align-items:center; gap:10px; font-size:13px; }
.bar-track { background:var(--surface-2); border-radius:4px; height:14px; position:relative; }
.bar { height:14px; border-radius:0 4px 4px 0; min-width:2px; }
.bar-val { text-align:right; }
.legend { display:flex; gap:18px; flex-wrap:wrap; margin:8px 0 4px; font-size:13px; }
.timeline td:first-child { white-space:nowrap; font-family:"JetBrains Mono",monospace; font-size:13px; }
.note { background:var(--surface-2); border-left:3px solid var(--accent); padding:10px 14px; border-radius:6px; margin:14px 0; max-width:80ch; }
@media (max-width:640px) { .bar-row { grid-template-columns:1fr; gap:3px; } h1 { font-size:28px; } }
"""


def build():
    cells, rounds, running = collect()
    model_runs = [(c, t) for c in cells if included(c) for t in c["targets"] if not infra_hit(t)]
    now = datetime.now().strftime("%d %b %Y %H:%M")

    tiles = []
    for fw in ("tinycal", "qua-agents"):
        runs = [(c, t) for c, t in model_runs if c["fw"] == fw]
        done = [(c, t) for c, t in runs if t["status"] == "completed"]
        fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
        med = fids[len(fids) // 2] if fids else None
        cost = sum((t["cost"] or 0) for _, t in runs) / (len(done) or 1)
        tiles.append(f'<div class="tile"><div class="v">{len(done)}/{len(runs)}</div><div class="k">{fw}: calibrations completed / attempted, all models</div></div>')
        tiles.append(f'<div class="tile"><div class="v">{pct(med)}</div><div class="k">{fw}: median single-qubit gate fidelity</div></div>')
        tiles.append(f'<div class="tile"><div class="v">${cost:,.2f}</div><div class="k">{fw}: judge-priced cost per completed calibration</div></div>')

    def lab(it, plain=False):
        return it["label"] if plain else esc(it["label"])
    cost_items = sorted([{"fw": c["fw"], "cost": t["cost"], "label": f"{c['backend']} {t['target']} · {c['fw']} · {c['model']}"}
                         for c, t in model_runs if t["cost"] is not None], key=lambda it: it["label"])
    cost_fig = bars(cost_items, "cost", lab, lambda v: f"${v:.2f}", "Judge-priced cost per qubit-run", "USD at prices.yaml, each qubit's own tokens; grouped by qubit so the framework × model bars sit together")
    err_items = sorted([{"fw": c["fw"], "err": 100 * (1 - t["gate_fid"]), "label": f"{c['backend']} {t['target']} · {c['fw']} · {c['model']}"}
                        for c, t in model_runs if t["gate_fid"] is not None], key=lambda it: it["label"])
    err_fig = bars(err_items, "err", lab, lambda v: f"{v:.3f}% ({100 - v:.2f}%)", "Single-qubit gate error per qubit-run", "gate error in %, fidelity in brackets; error per Clifford ÷ 1.875")

    legend = ('<div class="legend">' + "".join(f'<span class="chip"><i style="background:{FW_COLOR[f]}"></i>{f}</span>' for f in ("tinycal", "qua-agents")) + "</div>")
    running_note = ""
    live = [r for r in running if r["alive"]]
    stopped = [r for r in running if not r["alive"]]
    if live:
        running_note += ('<div class="note"><b>Still in flight when this page was built:</b> ' +
                         "; ".join(f"{r['backend']} {r['cell']} ({r['fw']}{'' if r['fw']=='dag-walk' else ' · ' + r['model']})" for r in live) +
                         ". Re-run the generator to refresh.</div>")
    if stopped:
        running_note += ('<div class="note"><b>Stopped before finishing, not graded (operator stop, 15 Sep 13:50):</b> ' +
                         "; ".join(f"{r['backend']} {r['cell']} ({r['fw']} · {r['model']})" for r in stopped) +
                         ". They are not in any table.</div>")
    timeline = '<table class="grid timeline"><tbody>' + "".join(f"<tr><td>{esc(t)}</td><td>{esc(w)}</td></tr>" for t, w in INCIDENTS) + "</tbody></table>"

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frameworks Comparison, Opus Sonnet qwen</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><div class="page">
<div class="eyebrow">qua-agents benchmark · framework comparison · generated {esc(now)}</div>
<h1 style="margin-top:8px">tinycal vs qua-agents with Opus, Sonnet and qwen3.8-27b, 14–17 Sep 2026</h1>
<p class="lede">Same scrambled snapshot per backend, handed to each framework in turn: opus-5 and sonnet-5 at high effort on 14–15 Sep
(three qubits per backend, then one new qubit per round), and qwen3.8-27b (27B open weights via OpenRouter, no effort knob) on 16–17 Sep
on the same snapshots and scrambles, three qubits per round. Numbers are read from each qubit-run's own record in result.json,
the cell's final QuAM state and the judge's price table; nothing here is estimated by hand.</p>
{running_note}

<h2 id="overview">Overview</h2>
<p><b>Apparatus.</b> qua-agents <span class="mono">61209cf</span> (integration checkout, profile_template.yaml modified: recursion_limit 300 from 16 Sep 19:45,
max_auto_retries 10 from 19:58) with qua-libs <span class="mono">a884483</span> for both frameworks; tinycal <span class="mono">298cb6c</span>
(clean until 16 Sep 19:43, then benchmarks/profile_template.yaml max_turns 150; my driver also passes --set max_turns=150); workload decalibrate_chip.yaml, scramble spec
<span class="mono">f47053f145f74c8c</span>; acceptance spec <span class="mono">821bb4dd1d5d0e5a</span>; model calls through the local proxy (:3456).
Every round on a backend starts from that backend's one snapshot, so cells are comparable within a backend:
arbel — {esc(SNAPSHOT_NOTE['arbel'])}; gilboa — {esc(SNAPSHOT_NOTE['gilboa'])}; qolab — {esc(SNAPSHOT_NOTE['qolab'])}.
Phase 1: arbel qB4 qC2 qC3, gilboa qD3 qD5 qC3, qolab Q1 Q2 Q3 as three-qubit cells; phase 2 adds the remaining qubits one per round,
each round = both frameworks × both models. Phase 3 (16–17 Sep): qwen3.8-27b on gilboa qD3 qD5 qC3 and qD2 qD4 qC5, qolab Q1 Q2 Q3 and
Q4 Q5 Q6, arbel qB4 qC2 qC3, both frameworks, no dag-walk; turn budgets capped at 150 per target from 16 Sep 19:45.</p>
<h3>Pins: what a new cell must use to be comparable</h3>
<p class="small muted">Shas and digests read from the cell documents themselves (framework sha, qua-libs sha, scramble spec and snapshot
digests) next to the state of each checkout when this page was generated. A comparable run uses the same four checkouts at these shas
with the same uncommitted template edits, seeds its work dir from the listed source-state (the scramble is deterministic, so the spec and
source digests come out identical and the judge accepts the document), and the same matrix picks. Per-cell profile hashes differ by
design (they carry the model and effort) and are in the run-identifier table.</p>
{pins_table(cells)}
<div class="tiles">{"".join(tiles)}</div>
<p class="small muted">How to read the tables: one qubit-run = one framework calibrating one qubit from the scrambled state with one model.
Gate fidelity = 1 − (RB error per Clifford ÷ 1.875), the qua-libs RB convention. "Per calibration" = the total over every qubit-run the
framework attempted, divided by the number it completed, so the spend and time of a qubit it never finished are charged to it.
Agent time is time inside model calls; QPU time is execution on the chip; queue wait is the cloud queue, neither side's doing.
Qubit-runs ended by the operator-side incidents (proxy outage, disk full) are excluded; genuine calibration failures are included.
Context = prompt size per model call in tokens, cached prefix included (tinycal from its events.jsonl, qua-agents from its audit
events); "end of a bring-up" is the last call of a completed qubit-run.
The dag-walk control (no model, the graph's defaults) stopped at node 2 of 18 on every qubit of every backend and is left out of the
comparison tables; its rows are in the run list.</p>
{legend}
{pivot_table(cells)}

<h3>Hard cases: qubits one framework could not calibrate</h3>
<p class="small muted">One row per qubit and model where at least one framework did not reach the end of the graph. Runs ended by the
operator-side incidents are excluded. A run that finished the graph with a poor number (arbel qB4, arbel qD1 with Opus) is not a hard case
by this rule; those are in the side-by-side tables below. "escalated" on the tinycal side means its model declared itself stuck; on the
qua-agents side it means the turn budget ran out.</p>
{hard_cases(cells)}
{context_section()}

{"".join(f'<h3>Per qubit, framework side by side — {esc(m)}</h3>{qubit_side_by_side(cells, m)}' for m in MODEL_LABELS)}
{cost_fig}
{err_fig}

<h3>What the numbers say</h3>
<p><b>Both frameworks calibrate the same qubits to the same place.</b> On every qubit both frameworks finished, the single-qubit gate
fidelity agrees within about 0.05 % (gilboa qD3 99.84–99.88 %, qD5 99.93–99.95 %, qC3 99.87–99.89 %; gilboa qD2 98.74–98.81 % for all four
qubit-runs; gilboa qD4 99.84–99.88 %). The chip sets the number, not the agent: qD2 has T1 ≈ 1.3 µs, and the gilboa qubits sit near a rough
T1/T2 bound with the IQCC pulse lengths kept (48–56 ns). Where IQCC's own dashboard has a number (arbel qC2 99.90 %, qC3 99.75 %) both
frameworks land on it.</p>
<p><b>Where they differ is cost and time.</b> tinycal does the same job at roughly a quarter to a third of the judge-priced cost of qua-agents
on both models, in about half the agent time, with fewer node re-runs. Sonnet halves the cost again on both frameworks at similar fidelity,
with more turns and re-runs.</p>
<p><b>qwen3.8-27b reaches the same fidelity where it finishes, and the framework decides whether it finishes.</b> On the one like-for-like
round complete so far (gilboa qD3 qD5 qC3, same scramble) both frameworks completed 3/3 at 99.7–99.9 % gate fidelity; tinycal did it in 2 h 10
for $2.01 with 1.6 h inside model calls, qua-agents in 6 h 09 for $5.30 with 17.9 h inside model calls, on its second attempt (the first lost two
targets to a few-second OpenRouter blip, which qua-agents does not retry, and the third to a hand-written 7.75 GHz f_01 that locked the drive
element over the ±500 MHz IF cap). The 11× gap in model time is the same model spending 30 s per turn in tinycal and ~4 min in qua-agents,
where its 12k-token reasoning repeatedly hits the 16k output ceiling and is regenerated. On qolab qwen failed in tinycal on all three qubits:
it committed a readout amplitude 13× the operating point (the node's no-onset fallback proposes the top of the sweep, and the model copied it)
and left the idle flux at 0 V, so the qubit was more than a gigahertz from where it searched; Opus on the identical scramble set both right
and finished Q1 and Q3. The qua-agents qwen cells on qolab, arbel and gilboa's second round are read from their documents when they end.</p>
<p><b>Arbel's poor numbers are gate length, not the qubit the matrix flags.</b> qC3 came out at 99.66–99.76 %; the hard qubit was qB4: both
Opus cells committed f_01 99 MHz below the record and needed 350–380 ns pulses at the recorded amplitude (98.57 % / 97.90 %), Sonnet found
the line and got 99.25 % with a 208 ns pulse at a quarter of the amplitude. On qC3 every cell settled on ~180 ns at half the recorded
amplitude (same pulse area as the recorded 96 ns @ 0.99, which sits at the output ceiling). The per-qubit tables show x180 length and amplitude.</p>
<p><b>The one qubit that defeated both frameworks is qolab Q2.</b> tinycal found the readout, never found f_01, and stopped cleanly after two hours.
qua-agents committed f_01 = 3.4 GHz, which puts the xy intermediate frequency at −1.8 GHz, outside the ±500 MHz window; from then on every
node failed inside open_qm. The model diagnosed this correctly in its own rationale but the harness offered it no action except "run another
node", so it looped 43 times on qubit_spectroscopy_fine until the 360-turn budget ended the target: 6.6 h and $121 for that cell, most of it
on Q2. Three guards would have caught it: validate a proposed f_01 against the LO before committing, an escape when the machine cannot open,
and a breaker on repeated identical failures.</p>
<p><b>Nothing is to spec</b> in any cell, because the acceptance spec asks for an RB confidence interval, a per-state readout fidelity and a
coherence-limited error, and the qua-libs node outputs carry none of them; the ballpark check on the scrambled parameters is the only spec item
that separates cells, and most of its misses are shared by both frameworks (stale snapshot fields, readout amplitude at the 0.1 instrument
ceiling on gilboa).</p>

<h2 id="stuck">Where things got stuck</h2>
<p>Every qubit-run that did not reach the end of the graph, with the cause classified from the cell's own escalation text.
"proxy outage", "disk full" and the git failure are operator-side incidents, not framework or chip behaviour; the timeline below has the details.</p>
{stuck_table(cells)}
<h3>Incident timeline</h3>
{timeline}

<h2 id="per-qubit">Per-qubit detail</h2>
<p>One table per qubit, every model cell that touched it. Gate fidelity with the RB error per Clifford underneath; readout is the assignment
fidelity the graph measured; T1/T2echo as measured by the cell; x180 is the committed π-pulse (length @ amplitude) read from the cell's final
QuAM state; ballpark is the judge's identity check on the scrambled parameters, with the misses named; agent, QPU, queue, tokens and cost are
the qubit-run's own.</p>
{per_qubit(cells)}

<h2 id="runs">Run identifiers</h2>
<p>qua-agents run ids open in the GUI at <a href="http://127.0.0.1:8080/#/runs">127.0.0.1:8080</a>; tinycal run dirs hold run.json and one
events.jsonl per target; cell dirs under the work dir hold result.json, grade.log, run.log, evidence and the final quam_state.</p>
{run_ids(cells, running)}
<p class="small muted">Built by make_fwcmp2_report.py in this repo. Operator log: ~/qab-runs/fwcmp2-STATUS.md; plan: ~/qab-runs/fwcmp2-FOLLOWUP.md;
round-one summary: ~/qab-runs/fwcmp2-20260914-summary.md.</p>
</div></body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} · {len(cells)} cells, {len(model_runs)} scored qubit-runs, {len(live)} in flight, {len(stopped)} stopped")


if __name__ == "__main__":
    build()
