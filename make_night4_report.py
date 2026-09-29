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

import collections
import glob
import re
import json
import subprocess
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
    "flash": ("qwen3.8-flash · OpenRouter", "qwen/qwen3.8-flash", "flash"),  # added 21 Sep 14:29, fixed flux-map node
    # 22 Sep 00:44: qua-agents on the same snapshots, same model and host, three qubits per backend, three workers
    "qa": ("qwen3.8-27b · qua-agents (OpenRouter)", "qwen/qwen3.8-27b", "qa"),
}
# qua-agents cells launched or queued (work dir, cell, backend, qubits, state): listed as in flight until their document exists
QA_QUEUE = [
    ("night4qa-arbel-20260922", "arbel-qwen3-8-27b-openrouter", "arbel", ["qB4", "qC2", "qC3"], "frozen at 11:25 to free arbel (SIGSTOP, resumable)"),
    ("night4qa-qolab-20260922", "qolab-qwen3-8-27b-openrouter", "qolab", ["Q1", "Q2", "Q3"], "done 11:52: Q1 99.93, Q2 and Q3 at the recursion limit"),
    ("night4qa-gilboa-20260922", "gilboa-qwen3-8-27b-openrouter", "gilboa", ["qD3", "qD5", "qC3"], "frozen at 09:56, gilboa left to another user"),
    ("night4qa-qolab-20260922-r2", "qolab-qwen3-8-27b-openrouter", "qolab", ["Q4", "Q5", "Q6"], "running since 11:52"),
    ("night4qa-gilboa-20260922-r2", "gilboa-qwen3-8-27b-openrouter", "gilboa", ["qD4", "qC1", "qC2"], "queued after gilboa round 1"),
    ("night4qa-gilboa-20260922-r3", "gilboa-qwen3-8-27b-openrouter", "gilboa", ["qC4", "qC5", "qD1"], "queued after gilboa round 2"),
    ("night4qa-gilboa-20260922-r4", "gilboa-qwen3-8-27b-openrouter", "gilboa", ["qD2"], "queued after gilboa round 3"),
]
base.FW_COLOR["splash"] = "var(--qua)"  # blue for Splash; OpenRouter keeps tinycal's green
base.FW_COLOR["flash"] = "var(--warn)"  # amber for the flash model
base.FW_COLOR["qa"] = "var(--qa)"  # violet for the qua-agents column (Splash already uses the base's blue)

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
TARGET_INCIDENTS = {
    ("night4-qolab-20260920-1600", "qolab-tinycal-qwen3-8-27b-splash", "Q1"):
        "stopped by the operator at 20:55 after 4.9 h: three consecutive 10 000-token turns with no tool call; RB was flat since "
        "turn 46 after the model's DRAG change broke its own readout",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-A", "qC5"):
        "stopped by the operator at 02:11 after 2.3 h: three consecutive 10 000-token turns with no tool call, on a run that had "
        "already committed the −0.12 V apex from a failed fit",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-C", "qD1"):
        "stopped by the operator at 07:24 after 5.2 h: three consecutive 10 000-token turns with no tool call; the run had been "
        "circling between power Rabi and T1 for three hours",
    ("night4-qolab-20260920-1600", "qolab-tinycal-qwen3-8-27b-splash-B", "Q6"):
        "stopped by the operator at 08:30 after 1.6 h: three consecutive 10 000-token turns with no tool call, at qubit spectroscopy",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash", "qB4"):
        "Splash answered HTTP 408 'HTTP I/O timed out' at 10:26 (turn 27, at power Rabi after 1.8 h) and tinycal did not retry a 408; "
        "not a model failure — re-run as qB4 · rerun under the tinycal that retries it",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash", "qC2"):
        "Splash HTTP 408 at 10:32 (turn 5, right after resonator identification); re-run as qC2 · rerun",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-D", "qD4"):
        "Splash HTTP 408 at 10:35 (turn 6, at readout power); re-run as qD4 · rerun",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash-B", "qB4"):
        "the 408 rerun, stopped by the operator at 14:23 after 1.4 h: turns 18–20 at the 10 000-token cap with no tool call, at qubit "
        "spectroscopy with readout and flux already fine (joint_offset 0.05 V, reference 0.042). Fifth capped-loop stop of the night; "
        "qC2 continues as arbel Splash-C",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash-C", "qC2"):
        "stopped by the operator at 16:15 at turn 29 (power Rabi) to restart on effort medium; it had been in 503 'memory did not "
        "become available' retries since 16:03 while a probe ran as a third Splash stream. Continues as qC2 · rerun in arbel Splash-D",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-D", "qC2"):
        "meant to be stopped at 16:15 (turn 2) for the restart on medium, but the target's process survived the kill and ran on, "
        "uncounted, as a third Splash stream at effort high until the operator found and killed it at 23:14 (84 turns, 43 retries, "
        "at DRAG). Not graded; qC2 was re-run as Splash-G. Its KV footprint is why the two counted streams saw 503 memory errors "
        "and lost their prefix cache from 16:24 on",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-F", "qC3"):
        "stopped by the operator at 21:23 after 3.6 h: turns 44–46 at the 10 000-token cap with no tool call — the first capped loop on "
        "medium effort. Readout, flux and f_01 were right (5.6488 GHz, ref 5.6487), but at turn 24 it committed x180 = 0.019 V from a "
        "low-SNR Rabi (readout amplitude set to 0.06 V, ref 0.18; the OpenRouter run used 0.38–0.58 V for the same pulse) — a π pulse "
        "25× too weak. IQ blobs then overlapped (51 %), the '55 kHz dispersive shift' was an artefact of not exciting the qubit, and the "
        "model spent turns 39–43 searching 7.2–7.5 GHz for a different qubit rather than revisiting the Rabi. Its cell continues as "
        "Splash-H (qC4, qD4 · rerun)",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-flash-openrouter-flash3", "qC2"):
        "never started: OpenRouter 429 'qwen/qwen3.8-flash is temporarily rate-limited' again at 21:24–21:34",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-flash-openrouter-flash4", "qC2"):
        "started at 22:29 through an intermittent OpenRouter 429, then hit a solid one at turn 7 (23:16–23:25, ladder exhausted) and "
        "aborted; 20 retries in 56 min. Four attempts, no arbel qC2 result in the flash column",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-flash-openrouter-flash2", "qC2"):
        "never started: OpenRouter answered 429 'qwen/qwen3.8-flash is temporarily rate-limited' for the whole ten-minute retry ladder at "
        "20:55–21:05; retried as flash3",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC5"):
        "stopped by the operator at 00:44 with the whole cell: OpenRouter had answered 429 for the full ladder on qC1–qC4 (00:14–00:40) and "
        "was doing the same here; the four remaining qubits never started",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qD1"):
        "stopped with the cell at 00:44 (429 ladder in progress); never got a model turn",
    ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash-D", "qC2"):
        "the medium-effort rerun ran the whole graph in 4.6 h (23/25 nodes) and declared itself stuck on a weak dispersive readout; its RB "
        "trace is flat (the '100 %' is not a measurement)",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-F", "qC2"):
        "the medium-effort rerun, aborted at 17:48 at turn 17 (qubit flux map) after ten minutes of connection errors: the tailnet "
        "dropped and Splash lives on the other side of it. Not a model failure; re-run once more as Splash-G after the cell ends",
    ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-D", "qD2"):
        "readout and flux done (joint_offset 0.024 V, reference 0.016), then 5.0–6.3 GHz of qubit spectroscopy found nothing and the "
        "model retuned qD2's dedicated xy upconverter to 8.85 GHz (accepted by write_state — the identity-guard gap again) before "
        "declaring itself stuck at turn 46 after 3 h",
}
# Cells that re-run targets an earlier cell lost (besides the "-R" readout reruns): their rows are labelled "· rerun".
RERUN_CELLS = {"arbel-tinycal-qwen3-8-27b-splash-B", "arbel-tinycal-qwen3-8-27b-splash-C", "arbel-tinycal-qwen3-8-27b-splash-D",
               "gilboa-tinycal-qwen3-8-27b-splash-E",
               # 22 Sep single-stream chain (night4-chain-v7.sh): every Splash target the first pass lost, at effort medium
               "qolab-tinycal-qwen3-8-27b-splash-C", "arbel-tinycal-qwen3-8-27b-splash-E", "gilboa-tinycal-qwen3-8-27b-splash-I",
               # 22 Sep 00:14 flash reruns of the flash column's losses (gilboa's flash cell is a first run, not listed here)
               "qolab-tinycal-qwen3-8-flash-openrouter-flash2", "arbel-tinycal-qwen3-8-flash-openrouter-flash5"}
# Targets on their third or later attempt on the same host: label "· rerun N".
RERUN_ORDINAL = {("arbel-tinycal-qwen3-8-27b-splash-E", "qB4"): 2, ("arbel-tinycal-qwen3-8-27b-splash-E", "qC2"): 3,
                 ("arbel-tinycal-qwen3-8-flash-openrouter-flash5", "qC2"): 3}
# gilboa Splash-F re-runs qC2 (stopped at turn 2) and continues qC3 qC4, plus the qD4 rerun: qC2 and qD4 are reruns, the others first runs.
RERUN_TARGETS = {("gilboa-tinycal-qwen3-8-27b-splash-F", "qC2"), ("gilboa-tinycal-qwen3-8-27b-splash-F", "qD4"),
                 ("gilboa-tinycal-qwen3-8-27b-splash-G", "qC2"), ("gilboa-tinycal-qwen3-8-27b-splash-H", "qD4"),
                 ("arbel-tinycal-qwen3-8-flash-openrouter-flash2", "qC2"), ("arbel-tinycal-qwen3-8-flash-openrouter-flash3", "qC2"),
                 ("arbel-tinycal-qwen3-8-flash-openrouter-flash4", "qC2")}


def is_rerun(cell_name):
    return cell_name.endswith("-R") or cell_name.endswith("-R2") or cell_name in RERUN_CELLS


def rerun_label(cell_name, target=None):
    """Suffix for a target label: '' for a first run, ' · rerun' for the readout reruns, ' · rerun 2' for the node-fix reruns."""
    if cell_name.endswith("-R2"):
        return " · rerun 2"
    if target is not None and (cell_name, target) in RERUN_ORDINAL:
        return f" · rerun {RERUN_ORDINAL[(cell_name, target)]}"
    if target is not None and (cell_name, target) in RERUN_TARGETS:
        return " · rerun"
    return " · rerun" if is_rerun(cell_name) else ""


OPERATOR_STOPS = {("night4-qolab-20260920-1600", "qolab-tinycal-qwen3-8-27b-splash", "Q1"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-A", "qC5"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-C", "qD1"),
                  ("night4-qolab-20260920-1600", "qolab-tinycal-qwen3-8-27b-splash-B", "Q6"),
                  ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash-B", "qB4"),
                  ("night4-arbel-20260920-1634", "arbel-tinycal-qwen3-8-27b-splash-C", "qC2"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-D", "qC2"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-27b-splash-F", "qC3"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC5"),
                  ("night4-gilboa-20260920-1712", "gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qD1")}
# Completed runs whose RB number is not a measurement (the fit's own warning says so); their fidelity is withheld from every table.
INVALID_RB = {
    ("gilboa", "openrouter", "qC2"): "a Rabi that never showed an oscillation at any amplitude or length (124 turns, 51 node runs) until the "
                                     "30 min hardware budget ended it; the RB '100 %' is a flat trace. The x180 the model finally wrote at 92 ns / "
                                     "12 mV says the drive was far too weak or off resonance; the write_state guard refused one of its writes",
    ("gilboa", "openrouter", "qC5 · rerun"): "readout fixed by the new wording (0.13–0.2 V), then the ±0.5 V apex at −0.15 V again, then 50 turns "
                                              "rewiring the drive (LO literal, upconverter 1↔0, LO = 0) before declaring the wiring broken. Two guards: "
                                              "no apex from a failed fit, and write_state refusing identity fields (ports, upconverter, opx_output, LO "
                                              "references)",
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
    ("20 Sep 20:55", "qolab Splash Q1 stopped by the operator after 4.9 h: turns 60–62 all ended at the 10 000-token cap with no tool call "
                     "(13–25 min each). The run had reached RB at turn 46 with a flat trace — its own DRAG change had broken the readout — "
                     "and never recovered. Its interim document stands; the cell was relaunched on Q2–Q6 (splash-B) so the Splash stream "
                     "is not spent on it. The first relaunch carried its own caffeinate, which the chain counted as a third Splash driver; "
                     "restarted without it a minute later."),
    ("20 Sep 21:55", "Recipe line 2 (readout power) reworded by the user in the tinycal working tree: finding the punch-out is now stated as "
                     "important — too low a power gives no SNR, too high no state discrimination — and the provisional-power fallback applies "
                     "only when no punch-out is visible at all. tinycal reads the recipe when a target starts, so every target that started after "
                     "this (gilboa OpenRouter qC4 at 21:58 and the whole Splash queue) runs with the new wording; earlier targets kept the old one. "
                     "Every OpenRouter failure of the night so far (qolab Q2, gilboa qD5 qC5, arbel qB4 qC2) began with 'no onset in the swept "
                     "range' followed by a provisional power 10–20× below the working point."),
    ("20 Sep 22:02", "Reruns of those five with the new wording (user-approved), OpenRouter, same scrambles, cell suffix -R: arbel qB4 qC2 and "
                     "qolab Q2 at once (qolab shares the QPU with the Splash-B cell, separate state copies), gilboa qD5 qC5 chained behind the "
                     "gilboa OpenRouter cell so gilboa never exceeds three targets. They appear as their own rows; the original attempts stay."),
    ("21 Sep 00:10", "Third instance of the ±0.5 V apex, this time on Splash (gilboa qC5): the model asked for ±0.7 V, was refused by the port "
                     "check, swept ±0.5 V, and committed the node's 'measured apex' at −0.12 V from a fit with R² 0.41 — after it had found a "
                     "good readout power (0.084 V, 2 dB under the fade onset). Same node defect, both hosts; qD5 and qC5 attract it because their "
                     "first power sweep finds no onset and the model then reaches for the widest sweep it can get."),
    ("21 Sep 02:02", "gilboa OpenRouter cell done: qC2 ended on the hardware budget after 124 turns on a Rabi that stayed flat (RB "
                     "'100 %' on a flat trace, stuck); the rerun chain launched gilboa qD5 qC5 at once. arbel qC2's rerun completed at "
                     "99.84 % where the original had stopped on a wrong line: the readout step went differently (max_power_dbm raised, "
                     "onset found at −20 dBm, 0.092 V committed)."),
    ("21 Sep 02:11", "gilboa Splash-A stopped by the operator on qC5: turns 28–30 all at the 10 000-token cap with no action, on the "
                     "−0.12 V flux point. Its remaining targets and splash-B's qC3 qC4 continue as one cell (splash-C, one at a time); "
                     "the arbel Splash cell follows. The 'stopped' Q1 and qC5 rows keep their interim documents."),
    ("21 Sep 03:41", "Reruns done: 4 of 5 recovered — qolab Q2 99.81 %, arbel qC2 99.84 %, arbel qB4 99.85 %, gilboa qD5 99.91 % — each "
                     "on a readout power found by raising the sweep or by the optimisation node instead of the old provisional 10–20× low value. "
                     "gilboa qC5 failed again: readout fine this time (0.13–0.2 V) but the ±0.5 V resonator sweep and its 'apex' at −0.15 V "
                     "a fourth time; the node fix, not the recipe, is what qC5 needs. Worse, from 02:32 the model started rewiring the drive: "
                     "it replaced xy/LO_frequency's reference with a literal 5.8 GHz, toggled xy/upconverter between 1 and 0 (a different "
                     "physical channel), pointed LO_frequency at another port's upconverter (con1/1/6, then con1/2/6), and at 03:23 wrote "
                     "LO_frequency = 0, all accepted; write_state refused four other writes (a junk upconverter_frequency key, a dict for "
                     "opx_output). It then declared the wiring 'missing' — it is not (wiring.json: con1/2/6). The guard checks executability; "
                     "it needs an identity rule too: ports, upconverter, opx_output and LO references are not calibration and must be refused."),
    ("21 Sep 07:24", "gilboa Splash-C stopped by the operator on qD1: turns 49–51 at the cap with no action, 5.2 h on one qubit that never "
                     "got past power Rabi (two of three Splash stops are on gilboa's power Rabi). The remaining six gilboa targets continue as "
                     "splash-D; the arbel Splash cell follows. qolab Splash Q5 finished at 99.94 %; Q6 is its last."),
    ("21 Sep 08:30", "qolab Splash-B stopped by the operator on Q6, its last target: turns 24–26 at the cap with no action, at qubit "
                     "spectroscopy after 1.6 h. Fourth Splash stop of the night, all the same shape. qolab Splash ends 4 of 6 (Q2 99.45, Q3 "
                     "99.95, Q4 99.95, Q5 99.94); the freed stream goes to the arbel Splash cell."),
    ("21 Sep 00:23", "arbel's cloud queue slowed: power Rabi and IQ_blobs on the reruns waited 208–953 s for a few seconds of execution "
                     "(gilboa's waits stayed under a minute). Nodes complete; the runs are just slower."),
    ("21 Sep 10:00", "Recipe edited again by the user: a paragraph asking the model to check on the figure that a node's fit and the point it "
                     "proposes make physical sense before committing (motivated by qC5's flux-map apex, see the debugging note). tinycal reads the "
                     "recipe at target start, so gilboa Splash-D from qD4 on and arbel Splash from qC2 on run under it; qD2 and qB4 do not."),
    ("21 Sep 10:26", "Splash degraded: HTTP 408 'HTTP I/O timed out' on three requests between 10:26 and 10:36, then 503 'native frame write "
                     "timed out' / 'engine is recovering' at 11:09. tinycal retried the 503s but not the 408s, so arbel qB4 (turn 27, 1.8 h in), "
                     "arbel qC2 (turn 5) and gilboa qD4 (turn 6) aborted. Fixed in tinycal (392a629: 408 joins 429/5xx in the retry ladder); "
                     "a fifth chain re-runs the three as qB4/qC2 · rerun (arbel Splash-B) and qD4 · rerun (gilboa Splash-E), each once the "
                     "backend's current Splash driver has finished, never more than two Splash streams. gilboa Splash-D qD2 ended 'stuck' at "
                     "10:26 on its own: readout and flux fine, no qubit line in 5.0–6.3 GHz, upconverter retuned to 8.85 GHz."),
    ("21 Sep 13:55", "gilboa qC2 and qC5 re-run on OpenRouter at the user's request (cell openrouter-R2, two in parallel, rows 'qC2 · rerun 2' "
                     "and 'qC5 · rerun 2'), with two changes: the resonator flux-map node fix (qua-libs fix/flux-map-dip-track b8a3e1d — the "
                     "dip tracked on the background-subtracted map, no proposal from a failed fit; this cell imports it from a worktree while "
                     "the Splash cells keep the unfixed working copy) and the recipe's new 'check the figure' paragraph. Same scramble, same "
                     "state as the 17:12 cells. Three gilboa targets in flight (qC1 on Splash)."),
    ("22 Sep 00:14", "Second pass. The single-stream Splash chain (23:51) was restructured into two staggered streams on different backends: "
                     "gilboa Splash-I (qC3 qC5 qD1 qD2) started while qolab Q1 was mid-run at 45k tokens, so the two tails do not meet in the "
                     "KV pool; arbel Splash-E (qB4, qC2) follows when the qolab driver ends. In parallel, qwen3.8-flash re-runs its losses on "
                     "OpenRouter — qolab Q2 Q3, arbel qB4 qC2 — and runs all ten gilboa C/D qubits for the first time (PAR 2), which keeps every "
                     "backend at three targets. All on the fixed nodes (qua-libs 65b4755)."),
    ("22 Sep 12:10", "First qua-agents document: the qolab cell ended at 11:52 when Q2 and Q3 hit the recursion limit (148 turns each) — "
                     "Q1 99.93 % (58 turns, 18 nodes), Q2 stuck in an 88-run T2echo loop after a good start, Q3 never found the qubit "
                     "(46 power sweeps, 27 spectroscopies). Three qubits, 10.4 h, $10 of tokens. Its r2 round (Q4 Q5 Q6) started at 11:52. "
                     "flash9 finished qolab Q3 at 99.95 % once OpenRouter's pool recovered; Q2 ended stuck again. arbel: the qua-agents cell "
                     "was frozen at 11:25 to free the backend and Splash-E qB4 (medium) started alone at 12:07; the gilboa cell stays frozen "
                     "for another user."),
    ("22 Sep 13:50", "Reference calibrations re-pulled (read-only, no QPU): arbel's stored gate fidelities were from 8 Sep 16:05, and the lab "
                     "recalibrated all 21 qubits at 13:10 today; qolab's six were recalibrated at 14:03 and gilboa's qC2 at 14:01, the rest of "
                     "gilboa still 15–16 Sep. The scatter plots now compare against those numbers (snapshot kept in "
                     "~/qab-runs/reference-state-20260922). arbel moved most on qD1 (99.18 → 94.87 %) and qB5 (99.45 → 99.85); qB4, the qubit "
                     "under test, is unchanged at 99.89. Splash-E finished qB4 at 13:23: 94.53 %, the 0→2 two-photon line for the fourth time "
                     "on this qubit — 17 of 18 nodes green on a 6507.17 MHz line, see the hard-case note. arbel is free again."),
    ("22 Sep 08:30", "Overnight, unattended from 02:59 (the operator's watch did not fire again until 08:24). Splash finished its second pass: "
                     "qolab Q6 · rerun 99.94 % (134 min, one capped turn), gilboa qD1 · rerun 99.77 % (57 min), gilboa qD2 · rerun 96.83 % "
                     "(49 min; the 0.7 µs-T1 qubit, reference 98.8 %) — all Splash streams ended at 04:04. OpenRouter's Qwen pool recovered "
                     "around 03:15: the gilboa flash7 cell, left walking its list, then completed qD2 96.33 %, qD3 99.93 %, qD4 99.93 % and "
                     "qD5 99.92 % (37–56 min each, no capped turn) after qC1–qC5 and qD1 had aborted on 429; qolab Q2/Q3 and arbel qB4/qC2 were "
                     "never retried because the probe had been stopped at 02:37. qua-agents · qwen3.8-27b: arbel (since 01:06), qolab (since "
                     "03:34) and gilboa (since 06:22) all still running at 08:24, every backend at three workers; the qolab cell's Q2 has run "
                     "T2echo 27 times and Q3 qubit_spectroscopy 26 times (Q1 is mid-graph), arbel's qC2 qubit_spectroscopy 9 times — a "
                     "qua-agents cell writes its document only at the end, so the cells were left to finish rather than killed."),
    ("22 Sep 02:36", "gilboa Splash-I qC5 · rerun finished at 99.745 % (79 min, 65 turns, no capped turn) — the Splash medium rerun beat the "
                     "OpenRouter result on this qubit (95.9 % with 1 µs gates): it found a 100 ns / 0.26 V gate. Splash's engine restarted "
                     "twice more (01:55, 02:30, 'native frame write timed out'), both ridden out on the retry ladder. The flash probe got a "
                     "200 at 02:02 and again at 02:28 and launched qolab flash6, gilboa flash7 and qolab flash8, but every launched target "
                     "still hit 429 for its whole ladder and aborted with 0–1 turns; the probe was stopped at 02:37 after the third aborting "
                     "launch. Flash stays down until OpenRouter's shared Qwen pool recovers."),
    ("22 Sep 01:24", "gilboa Splash-I qC3 · rerun finished at 99.97 % in 46 minutes (73 turns, no capped turn) — the second target to run on the "
                     "staggered two-stream layout; Splash-I moved on to qC5. qolab Splash-C Q1 · rerun ended stuck after 89 minutes on a wrong "
                     "f_01 (5207/5222 vs 5272 MHz) and a readout put back to 0.02 V; Q6 · rerun started. The arbel qua-agents cell was restarted "
                     "at 01:06 on the rebuilt integration checkout (origin/main fc8de51, 19 commits newer); a flash probe pings OpenRouter every "
                     "5 minutes and relaunches the flash reruns when the shared pool answers again."),
    ("22 Sep 00:44", "qwen3.8-flash is rate-limited again: OpenRouter answered 429 for the whole retry ladder on every flash target launched "
                     "at 00:14 (qolab Q2 Q3, arbel qB4 qC2, gilboa qC1–qC4 in turn); the gilboa cell was stopped at 00:44 rather than let it "
                     "abort ten targets one by one. The flash reruns wait for the limit to lift. qua-agents · qwen3.8-27b (OpenRouter) "
                     "started on arbel (qB4 qC2 qC3, three workers) at 00:44 on the night-4 snapshot — the re-scramble is byte-identical to "
                     "tinycal's — and the working-copy qua-libs 65b4755 (deps.yaml re-pinned for it); the qolab and gilboa cells follow when "
                     "their Splash streams end. First launch failed on the runner's own qua-libs pin check (a884483); relaunched at 00:44."),
    ("21 Sep 23:26", "All cells done. gilboa Splash-H finished qD4 · rerun at 99.93 % in 35 minutes (64 turns, 27 nodes, no capped turn, 99 % "
                     "prefix-cache hits) — the fastest Splash bring-up of the night, and the only one that ever ran as the sole stream. arbel "
                     "flash4 qC2 aborted at turn 7 on a solid OpenRouter 429 (fourth and last attempt)."),
    ("21 Sep 23:14", "A third Splash stream had been running unseen since 16:06: gilboa Splash-D's qC2 target survived the 16:15 kill (its "
                     "python process outlived the driver; the check counted drivers, not run pids) and ran on at effort high for seven hours, "
                     "84 turns, 43 retries. Killed by pid. Every 'two-stream' observation since 16:24 had three streams behind it, which is where "
                     "the 503 memory errors and the zero-cache turns on G/H came from (their hit rate 0.86–0.91, D's 0.67; all three lost the "
                     "prefix at 21:28). Post-mortem of arbel flash qB4 (92.3 %): it calibrated the 0→2 two-photon line (6507 MHz = f_01 − α/2) "
                     "end to end — see the hard-case note; the same trap took OpenRouter's first qB4 and qC2."),
    ("21 Sep 23:08", "gilboa Splash-G finished qC2 · rerun at 99.92 % and Splash-H finished qC4 at 99.09 % — both on medium effort with no capped "
                     "turn; H moved on to qD4 · rerun (the 408-aborted target). arbel flash4 qC2 got through OpenRouter's rate limit at the fourth "
                     "attempt and is running. The power_rabi node fix (first turning point no longer overrides a good fit when it moves the other "
                     "way or sits far below it; qua-libs 65b4755) landed at 23:05 and applies to targets launched from now on."),
    ("21 Sep 21:23", "gilboa Splash-F stopped by the operator on qC3: turns 44–46 at the cap with no action — medium effort is not immune, "
                     "it just pushes the loops to longer contexts (58k here) and rarer situations (a 7.2–7.5 GHz qubit search with no line). "
                     "Relaunched as two streams: Splash-G (qC2 · rerun) and Splash-H (qC4, then qD4 · rerun). arbel Splash-D qC2 ended 'stuck' at "
                     "21:00 after the full graph on a weak dispersive readout (flat RB); the arbel flash qC2 that followed never started — "
                     "OpenRouter rate-limited qwen3.8-flash for ten minutes — and was relaunched as flash3. Splash server restarts by the user "
                     "at 18:54, 19:19, 20:03 were all retried through."),
    ("21 Sep 17:48", "The tailnet dropped for ~12 minutes; Splash (on the user's MacBook) was unreachable. gilboa Splash-F qC2 exhausted its "
                     "retry ladder and aborted at turn 17; the cell moved on to qC3. arbel Splash-D qC2 rode it out on a timeout retry. Chain v6 "
                     "re-runs gilboa qC2 as Splash-G after Splash-F ends. First hour on medium: no capped turn, reasoning at most 2.4k tokens, "
                     "every turn a tool call."),
    ("21 Sep 16:24", "Splash moved to effort medium for everything from here (matrix.yaml). A probe replayed two turns that had ended at the "
                     "10 000-token cap: Splash honours the effort level — medium thought 840–1 280 reasoning tokens where high used 4 000–9 300, "
                     "made the same decisions and took a fifth of the time — but ignores reasoning.max_tokens (6 100–6 400 tokens against a "
                     "4 000 budget). The probe's third stream pushed the arbel qC2 rerun into 503 'memory did not become available' retries at "
                     "16:03, so both Splash cells were stopped at a node boundary (gilboa qC2 at turn 2, arbel qC2 at turn 29) and relaunched on "
                     "medium and the now-merged node fix: gilboa Splash-F (qC2 qC3 qC4 qD4) and arbel Splash-D (qC2). The Splash column is "
                     "therefore split three ways: high effort + old node (everything up to qC1), high + old node with the figure paragraph "
                     "(qD4… none completed), and medium + fixed node from 16:24."),
    ("21 Sep 16:06", "gilboa Splash qC1 completed at 99.02 % after 69 turns and 5.5 h (OpenRouter: 99.68 %), the last target on high effort."),
    ("21 Sep 15:36", "qua-libs feat/qualibrate-ai fast-forwarded to the flux-map fix (b8a3e1d). Running processes keep the code they loaded; "
                     "every target started afterwards runs on the fixed node."),
    ("21 Sep 14:47", "gilboa qC2 · rerun 2 (OpenRouter, fixed flux-map node) completed at 99.89 % after 51 turns — the flux map proposed "
                     "0.011 V (reference 0.006) and every node after it went through first time, on the qubit whose two earlier runs had "
                     "committed −0.16 V and ended at the hardware budget with a flat Rabi. qC5 · rerun 2 is at T1 with 25/26 nodes clean."),
    ("21 Sep 14:29", "qwen3.8-flash (OpenRouter, effort high, 16k max_tokens) added at the user's request as a third column: qolab Q1–Q6 three "
                     "at a time and arbel qB4 + qC3 now, arbel qC2 chained behind the Splash qC2 rerun so the qubit is never driven by two agents. "
                     "Same snapshots and scramble as the other cells; runs on the fixed flux-map node and the recipe with the figure paragraph, "
                     "like the OpenRouter rerun-2 cell."),
    ("21 Sep 14:23", "arbel Splash-B stopped by the operator on the qB4 rerun: turns 18–20 at the cap with no action, at qubit spectroscopy "
                     "(readout and flux done). The remaining target continues as arbel Splash-C (qC2 · rerun). Meanwhile the OpenRouter "
                     "rerun-2 cell with the fixed flux-map node proposed the arc's top on both qubits — qC2 0.011 V (ref 0.006), qC5 "
                     "0.025 V (ref 0.030) — where every earlier run of these two had committed −0.13…−0.16 V."),
]

# What would have caught each hard case, by (backend, provider token, qubit); the operator's reading of the transcripts.
HARD_CASE_NOTE = {
    ("gilboa", "openrouter", "qD5"): "the resonator flux map proposed a joint_offset from a fit it had itself rejected (R² 0.08) over a ±0.5 V "
                                     "sweep that spans several flux periods; the node must propose nothing without a fit, and warn when the sweep "
                                     "is wider than a period",
    ("gilboa", "openrouter", "qC5"): "the same −0.13 V 'measured apex' from a failed fit (R² 0.16) over ±0.5 V; same fix",
    ("gilboa", "splash", "qD1"): "a Rabi with poor contrast at the readout it chose, three hours between power Rabi and T1, then three capped "
                                 "reasoning turns with no action; a breaker on consecutive capped turns, and a lower effort level for Splash",
    ("gilboa", "splash", "qC5"): "the same ±0.5 V sweep and a −0.12 V 'measured apex' from a fit with R² 0.41, committed with a good readout power "
                                 "already in hand; the node must propose nothing without a fit",
    ("gilboa", "openrouter", "qC2"): "a Rabi that never showed an oscillation at any amplitude or length (124 turns, 51 node runs) until the "
                                     "30 min hardware budget ended it; the RB '100 %' is a flat trace. The x180 the model finally wrote at 92 ns / "
                                     "12 mV says the drive was far too weak or off resonance; the write_state guard refused one of its writes",
    ("gilboa", "openrouter", "qC5 · rerun"): "readout fixed by the new wording (0.13–0.2 V), then the ±0.5 V apex at −0.15 V again, then 50 turns "
                                              "rewiring the drive (LO literal, upconverter 1↔0, LO = 0) before declaring the wiring broken. Two guards: "
                                              "no apex from a failed fit, and write_state refusing identity fields (ports, upconverter, opx_output, LO "
                                              "references)",
    ("gilboa", "openrouter", "qD1"): "the RB node reported 99.998 % from a fit covering 0.009 decay lengths with amplitude 16.5; the fit's own "
                                     "warning was in the result and the model declared completion anyway. A fit that far outside [0, 1] should "
                                     "return no number",
    ("gilboa", "openrouter", "qD2"): "the T1 ≈ 1.3 µs qubit: 95.9 % against 98.7–98.8 % in the framework rounds; a genuine calibration, poorer "
                                     "than the best on this qubit",
    ("qolab", "splash", "Q1"): "calibrated to RB at turn 45, then committed a DRAG alpha the fit did not support, which broke the readout it had; "
                              "from turn 49 on, 10 000-token reasoning turns with no action (9 of 14 hit the cap). A breaker on consecutive "
                              "capped turns without a tool call, and a rule that a change followed by a worse IQ_blobs is reverted",
    ("qolab", "splash", "Q6"): "the same capped-reasoning stall, at qubit spectroscopy this time; a breaker on consecutive capped turns without "
                              "a tool call, and a lower effort level for Splash",
    ("qolab", "openrouter", "Q2"): "readout power committed at 0.035 V with no punch-out onset in the sweep, ~50 % IQ contrast, flat RB; the "
                                   "night-2 recipe's 'provisional low power' line was followed, but no second power sweep at the sweet spot",
    ("arbel", "splash", "qB4 · rerun 2"): (
        "the 0→2 two-photon line for the fourth time on this qubit, and the first on Splash: qubit_spectroscopy committed f_01 = 6507.17 MHz "
        "(reference 6605.69, α = 194 MHz, f_01 − α/2 = 6508.5) and the graph then ran to the end on |2⟩ — 17 of 18 nodes, a 600 ns x180, "
        "a 'T1' of 336 µs and a 'T2echo' of 574 µs (the cascade, not the qubit), RB 10.2 % per Clifford. Nothing stopped it: every node fitted "
        "something, and the grader caught it only afterwards, on the f_01 ballpark. The low-drive power-scaling check is the fix"),
    ("arbel", "openrouter", "qC2"): "the 0→2 two-photon line committed as f_01: 5798.05 MHz against a reference f_01 of 5901.15 and α = 205 MHz "
                                    "(f_01 − α/2 = 5798.5); χ ≈ 0, flat power Rabi and flat blobs followed, and the model blamed the readout",
    ("arbel", "openrouter", "qB4"): "the same 0→2 two-photon line: qubit_spectroscopy identified 6504 MHz (reference f_01 6605.69, α = 194 MHz, "
                                    "f_01 − α/2 = 6508.5); the budget then ran out at turn 112 after 48 node runs of readout re-optimisation",
    ("qolab", "qa", "Q2"): "qua-agents: readout, flux and f_01 done by turn ~40, then the worker ran T2echo 88 times in a row (each re-run "
                           "with a different window or shot count) until the recursion limit ended it at turn 148 — no QPU budget stops a "
                           "qua-agents worker; 15 of 18 nodes, 11M input tokens",
    ("qolab", "qa", "Q3"): "qua-agents: 46 readout power sweeps and 27 qubit spectroscopies, no f_01 ever committed; recursion limit at turn 148 "
                           "with 4 of 18 nodes — the qolab readout-power trap again (the qubit line is invisible above ~0.012 V), which the "
                           "worker never got out of",
    ("qolab", "splash", "Q1 · rerun"): "the medium-effort rerun committed f_01 = 5207 then 5222 MHz (reference 5272) and, after trying 0.0025–0.008 V, "
                                       "put the readout back to 0.02 V — where qolab's qubit line is invisible — then read the overlapping blobs as "
                                       "'dispersive coupling far too weak' and declared itself stuck at turn 52 (89 min, 22 nodes, 2 capped turns). "
                                       "The 27b on OpenRouter and flash both found 5272 at 0.0043–0.0051 V readout on the same state",
    ("arbel", "flash", "qB4"): "the 0→2 two-photon line again, calibrated end to end: qubit_spectroscopy's 300 MHz window held a sharp 16σ line at "
                               "6507 MHz that moves with flux (the two-photon line tracks f_01 exactly) and only a broad 4σ hump at the real "
                               "6.58–6.61 GHz, so the node reported identified=True at 6507.2 and the model never doubted it. Everything after is "
                               "the |0⟩→|2⟩ transition driven through two photons: the 48 ns π 'needed 4 V' (9× the reference V·ns), so the gate was "
                               "lengthened to 400 ns; x90 = x180/2 failed because the two-photon angle goes as amplitude²; the 'T1' of 55 µs is "
                               "the |2⟩→|1⟩→|0⟩ cascade (reference T1 31 µs); DRAG had no signal; RB 92.3 %. The rerun that reached 99.85 % "
                               "escaped only because its first 100 MHz window (6505–6605) left the 6507 line on the edge and the flux arc "
                               "then found the apex at 6605",
}


# ----------------------------------------------------------------------------- collect
def cell_log_token(cell_name: str) -> str:
    """The driver's log is cell-<token>.log, token = the cell name after the model key's family prefix."""
    return re.sub(r".*qwen3-8-(27b|flash)-", "", cell_name)


def provider_of(cell_name: str) -> str:
    if "tinycal" not in cell_name:  # a qua-agents cell: <backend>-<model key>
        return "qa"
    if "-flash-" in cell_name:  # qwen3-8-flash-openrouter cells also contain "-openrouter"
        return "flash"
    for token in PROVIDERS:
        if f"-{token}" in cell_name:
            return token
    return "openrouter"


def live_status(run_id: str) -> dict:
    try:
        return json.load(open(TINYCAL_RUNS / run_id / "run.json")).get("status", {})
    except Exception:  # noqa: BLE001
        return {}


def overrides(run_id: str, target: str):
    """(state writes, overrides) from tinycal's events.jsonl. An override is a written value that is not a node's proposal: no node
    ever proposed that path, or the value is more than 0.1 % from the latest one proposed for it. None when there is no event log."""
    path = TINYCAL_RUNS / run_id / target / "events.jsonl"
    if not run_id or not path.exists():
        return None
    proposed, writes, over = {}, 0, 0
    for line in open(path):
        e = json.loads(line)
        if e.get("tool") == "run_node":
            for p in e.get("proposed_updates") or []:
                proposed[p["path"]] = p["proposed"]
        elif e.get("tool") == "write_state":
            for w in e.get("results") or []:
                if w.get("status") != "written":
                    continue
                writes += 1
                v, p = w["value"], proposed.get(w["path"])
                numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (v, p))
                if not (numeric and abs(v - p) <= 1e-3 * max(abs(p), 1e-12)) and v != p:
                    over += 1
    return writes, over


# The bring-up recipe's order (bringup_recipes/flux_tunable_1q.md): one entry per step; the power sweep appears twice (step 4 repeats
# it at the sweet spot) and T1/T2echo share a step, in either order.
RECIPE = [("resonator_identification",), ("resonator_spectroscopy_vs_power",), ("resonator_spectroscopy_vs_flux",),
          ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy",), ("qubit_spectroscopy_vs_flux",), ("qubit_spectroscopy_fine",),
          ("power_rabi",), ("T1_coarse",), ("readout_power_optimization",), ("readout_frequency_optimization",), ("IQ_blobs",),
          ("ramsey_vs_flux_calibration",), ("power_rabi_error_amplification_x180",), ("ramsey",), ("T1", "T2echo"),
          ("DRAG_calibration",), ("Randomized_benchmarking",)]
RECIPE_NODES = list(dict.fromkeys(n for step in RECIPE for n in step))


def _same(v, p):
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (v, p))
    return v == p or (numeric and abs(v - p) <= 1e-3 * max(abs(p), 1e-12))


def node_decisions(run_id: str, target: str):
    """Per node, what the model did with it, from tinycal's events.jsonl; None when there is no event log.
    Sequence (each node run, against the recipe order): in order / retried after a failure / re-run right after a success /
    stepped back to (an earlier step, after later ones ran) / skipped ahead to (a step past the next one); left after a failure is
    charged to the node whose last run failed when the model went elsewhere. Proposals (each run that proposed state updates, judged
    by the writes before the next node run): accepted as proposed / partly written / value changed / declined, and after a decline
    whether the model re-ran the same node."""
    path = TINYCAL_RUNS / run_id / target / "events.jsonl"
    if not run_id or not path.exists():
        return None
    per = {}
    ptr, seen, prev, prev_ok = -1, set(), None, True
    pending = None  # (node, proposals {path: value}, written {path: value}) of the last run that proposed something

    def settle(next_node):
        if pending is None:
            return
        node, props, written = pending
        got = [p for p in props if p in written]
        if len(got) == len(props) and all(_same(written[p], props[p]) for p in props):
            key = "accepted"
        elif any(not _same(written[p], props[p]) for p in got):
            key = "changed"
        elif got:
            key = "partial"
        else:
            key = "declined_rerun" if next_node == node else "declined_other"
        per[node][key] += 1

    for line in open(path):
        e = json.loads(line)
        if e.get("tool") == "run_node":
            n = e.get("node")
            if n not in RECIPE_NODES:
                continue
            settle(n)
            c = per.setdefault(n, collections.Counter())
            ok = e.get("outcome") == "successful"
            c["runs"] += 1
            c["failed"] += not ok
            if prev is not None and prev != n and not prev_ok:
                per[prev]["left_failed"] += 1
            if n == prev:
                c["retry" if not prev_ok else "repeat"] += 1
            else:
                idx = [i for i, step in enumerate(RECIPE) if n in step]
                g = min((i for i in idx if i >= ptr), default=max(idx))
                if g <= ptr:
                    c["in_order" if g == ptr and n not in seen else "back"] += 1
                elif g == ptr + 1:
                    c["in_order"] += 1
                else:
                    c["ahead"] += 1
                ptr = max(ptr, g)
            seen.add(n)
            prev, prev_ok = n, ok
            props = {p["path"]: p["proposed"] for p in e.get("proposed_updates") or []}
            pending = (n, props, {}) if props else None
        elif e.get("tool") == "write_state" and pending is not None:
            for w in e.get("results") or []:
                if w.get("status") == "written" and w["path"] in pending[1]:
                    pending[2][w["path"]] = w["value"]
    settle(None)
    return per


def collect():
    prices = load_prices()
    cells = []
    for work in sorted(list(RUNS.glob("night4-*")) + list(RUNS.glob("night4qa-*"))):
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
            final = (token == "qa") or ((work / f"cell-{cell_log_token(cell.name)}.log").exists() and
                                        "done →" in (work / f"cell-{cell_log_token(cell.name)}.log").read_text())
            alive = final or subprocess.run(["pgrep", "-f", r.get("run_id") or "none"], capture_output=True).returncode == 0
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
                    # the cell is over (its driver wrote "done", or nothing of it is running): a target still pending in
                    # result.json is the one the operator stopped, or one that never started
                    if final or not alive:
                        status = "stopped" if (work.name, cell.name, x["target"]) in OPERATOR_STOPS else "not run"
                cause = TARGET_INCIDENTS.get((work.name, cell.name, x["target"]))
                targets.append({
                    "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                    "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, model_id, prices),
                    "nodes_run": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                    "gate_fid": (gate_fidelity(rb.get("error_per_clifford")) if x["status"] == "completed"
                                 and (backend, token, x["target"] + rerun_label(cell.name, x["target"])) not in INVALID_RB else None),
                    "ctx": (lambda cs: {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None, "n": len(cs)})(
                        context_series(cell, r.get("run_id") or "", x["target"], "qua-agents" if token == "qa" else "tinycal")),
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
                    "overrides": None if token == "qa" else overrides(r.get("run_id") or "", x["target"]),
                    "decisions": None if token == "qa" else node_decisions(r.get("run_id") or "", x["target"]),
                })
            qubits = [x["target"] for x in r["targets"]]
            c = {
                "work": work.name, "cell": cell.name, "backend": backend, "fw": colour, "model": label, "provider": token,
                "qubits": qubits, "round": (backend, "+".join(qubits)), "run_id": r.get("run_id"),
                "status": r["status"] if final else ("stopped" if not alive else "running"), "started": local(r.get("started_at")),
                "ended": local(r.get("ended_at")) if final else ("stopped" if not alive else "—"),
                "wall_s": t["time"]["total_s"], "model_s": t["time"].get("model_s"),
                "queue_s": t["time"].get("queue_wait_s"), "qpu_s": t["time"].get("qpu_execution_s"),
                "turns": t["turns"]["total"], "nodes": t["nodes"]["executions"], "reruns": t["nodes"]["re_executions"],
                "tok_in": t["tokens"].get("input"), "tok_out": t["tokens"].get("output"),
                "tok_cr": t["tokens"].get("cache_read"), "tok_cw": t["tokens"].get("cache_creation"),
                "collateral": len((r.get("judge") or {}).get("collateral_edits") or []),
                "targets": targets,
                "finished": sum(1 for x in targets if x["status"] == "completed"),
                "chip": chip_counters(cell, r.get("run_id") or ""),
                "incident": None, "superseded": False, "final": final or not alive,
            }
            row = cm.get(cell.name, {})
            c["cost"], c["cw_tokens"], c["graph_frac"] = row.get("cost"), row.get("tokens"), row.get("graph")
            c["stopped_by"] = row.get("stopped_by")
            cells.append(c)
    return cells


# ----------------------------------------------------------------------------- tables specific to this night
_REF_FID: dict[str, dict[str, float]] = {}
_REF_DATE: dict[str, dict[str, str]] = {}
# Pulled on 22 Sep 2026, after the labs recalibrated; the night's own snapshots are the fallback for qubits it misses.
REFERENCE_SNAPSHOT = RUNS / "reference-state-20260922"


def _load_reference(backend: str) -> None:
    """Read gate_fidelity.averaged (and its stamp) for every qubit; the freshest snapshot read last wins."""
    _REF_FID[backend], _REF_DATE[backend] = {}, {}
    sources = [work / "source-state" for work in sorted(RUNS.glob(f"night4-{backend}-2026*"))
               if work.name not in EXCLUDED_DIRS]
    sources.append(REFERENCE_SNAPSHOT / backend)
    for src in sources:
        f = src / "state.json"
        if not f.exists():
            continue
        try:
            qubits = json.load(open(f))["qubits"]
        except Exception:  # noqa: BLE001
            continue
        for q, v in qubits.items():
            gf = v.get("gate_fidelity") or {}
            if not gf.get("averaged"):  # 0 or absent: never calibrated, not a reference
                continue
            _REF_FID[backend][q] = gf["averaged"]
            _REF_DATE[backend][q] = gf.get("averaged_updated_at") or ""


def reference_fidelity(backend: str, qubit: str):
    """The qubit's gate_fidelity.averaged in the freshest pulled cloud state, or None."""
    if backend not in _REF_FID:
        _load_reference(backend)
    return _REF_FID[backend].get(qubit)


def reference_date(backend: str, qubit: str) -> str:
    """The averaged_updated_at stamp behind that number, as the cloud stores it (Israel time)."""
    if backend not in _REF_DATE:
        _load_reference(backend)
    return _REF_DATE[backend].get(qubit, "")


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _reference_dates_phrase(used) -> str:
    """'arbel 22 Sep 13:10, gilboa 15–22 Sep' — the stamps of the references actually plotted."""
    def day(s):
        return f"{int(s[8:10])} {_MONTHS[int(s[5:7]) - 1]}"

    out = []
    for b in sorted({b for b, _ in used}):
        stamps = sorted({d for bb, q in used if bb == b and (d := reference_date(b, q))})
        if not stamps:
            continue
        days = sorted({s[:10] for s in stamps})
        if len(days) == 1:
            out.append(f"{b} {day(days[0])} {stamps[0][11:16]}")
        else:  # same month: '15–22 Sep', not '15 Sep–22 Sep'
            first = day(days[0])[:-4] if days[0][5:7] == days[-1][5:7] else day(days[0])
            out.append(f"{b} {first}–{day(days[-1])}")
    return ", ".join(out)


THRESHOLD = 0.999


def fidelity_strip(done):
    """One dot per completed qubit-run on a log gate-error axis, the 99.9 % line, and the count above it.

    Filled dots: at or above the qubit's reference fidelity (the human calibration in the pulled state); hollow: below it.
    """
    import math
    pts = [(c, t) for c, t in done if t["gate_fid"] is not None]
    if not pts:
        return ""
    W, H, L, R = 220, 40, 6, 6
    lo, hi = math.log10(0.02), math.log10(5.0)  # gate error in %, 0.02 % .. 5 %
    x = lambda err_pct: L + (W - L - R) * (min(max(math.log10(max(err_pct, 0.02)), lo), hi) - lo) / (hi - lo)  # noqa: E731
    xt = x(100 * (1 - THRESHOLD))
    parts = [f'<svg class="strip" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="gate error per completed run">']
    for tick in (0.03, 0.1, 0.3, 1, 3):
        parts.append(f'<text x="{x(tick):.1f}" y="{H - 2}" class="tick" text-anchor="middle">{tick:g}%</text>')
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{H - 14}" y2="{H - 14}" class="axis"/>')
    parts.append(f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="2" y2="{H - 12}" class="thr"/>')
    above = 0; beat = 0; with_ref = 0
    for i, (c, t) in enumerate(sorted(pts, key=lambda ct: ct[1]["gate_fid"])):
        f = t["gate_fid"]; err = 100 * (1 - f)
        ref = reference_fidelity(c["backend"], t["target"])
        hollow = ref is not None and f < ref
        if f >= THRESHOLD:
            above += 1
        if ref is not None:
            with_ref += 1
            beat += 0 if hollow else 1
        y = H - 22 - (i % 3) * 5  # three lanes so neighbours do not stack exactly
        style = 'class="dot hollow"' if hollow else 'class="dot"'
        label = f"{c['backend']} {t['target']}{rerun_label(c['cell'], t['target'])} · {pct(f)}" + (f" (reference {pct(ref)})" if ref is not None else "")
        parts.append(f'<circle cx="{x(err):.1f}" cy="{y}" r="3.2" {style}><title>{esc(label)}</title></circle>')
    parts.append("</svg>")
    note = (f"≥ {100 * THRESHOLD:g} %: <b>{above} / {len(pts)}</b>"
            + (f"<br><span class='muted'>at or above the reference: {beat} / {with_ref}</span>" if with_ref else ""))
    return f"<div class='stripwrap'>{''.join(parts)}<div class='small'>{note}</div></div>"


def _scatter_svg(points, *, xlabel, ylabel, lo=0.02, hi=6.0, guides=(1.0,), guide_labels=("y = x",), W=300, H=270):
    """Log-log scatter of gate error in %, one dot per completed run, coloured by host; ``points`` = (x, y, label, host)."""
    import math
    L, R, T, B = 46, 12, 12, 40
    llo, lhi = math.log10(lo), math.log10(hi)
    sx = lambda v: L + (W - L - R) * (min(max(math.log10(v), llo), lhi) - llo) / (lhi - llo)  # noqa: E731
    sy = lambda v: H - B - (H - T - B) * (min(max(math.log10(v), llo), lhi) - llo) / (lhi - llo)  # noqa: E731
    out = [f'<svg class="scatter" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">']
    for tick in (0.03, 0.1, 0.3, 1, 3):
        if lo <= tick <= hi:
            out.append(f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                       f'<line y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" x1="{L}" x2="{W - R}" class="grid"/>'
                       f'<text x="{sx(tick):.1f}" y="{H - B + 14}" class="tick" text-anchor="middle">{tick:g}%</text>'
                       f'<text x="{L - 6}" y="{sy(tick) + 3:.1f}" class="tick" text-anchor="end">{tick:g}%</text>')
    for k, lab in zip(guides, guide_labels):
        x0, x1 = lo, hi / k if k > 1 else hi
        out.append(f'<line x1="{sx(x0):.1f}" y1="{sy(x0 * k):.1f}" x2="{sx(x1):.1f}" y2="{sy(x1 * k):.1f}" class="guide"/>'
                   f'<text x="{sx(x1) - 4:.1f}" y="{sy(x1 * k) - 4:.1f}" class="tick" text-anchor="end">{esc(lab)}</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" class="frame"/>')
    for x, y, label, host in points:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" class="pt" style="fill:{base.FW_COLOR[PROVIDERS[host][2]]}">'
                   f'<title>{esc(label)}</title></circle>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 4}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(12,{(T + H - B) / 2:.0f}) rotate(-90)" class="lab" text-anchor="middle">{esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def _scatter_row(points, **kw):
    """One panel per host, side by side like the table's columns; an empty host gets a placeholder."""
    panels = []
    for h in ("openrouter", "splash", "flash", "qa"):
        pts = [p for p in points if p[3] == h]
        head = (f"<div class='ptitle'><span class='chip'><i style='background:{base.FW_COLOR[PROVIDERS[h][2]]}'></i>{esc(PROVIDERS[h][0])}</span>"
                f" <span class='small muted'>{len(pts)} runs</span></div>")
        body = _scatter_svg(pts, **kw) if pts else "<div class='small muted' style='padding:24px 0'>no completed run yet</div>"
        panels.append(f"<div class='panel'>{head}{body}</div>")
    return "<div class='scattergrid'>" + "".join(panels) + "</div>"


def error_scatters(cells):
    """Measured gate error against (1) the reference calibration's error and (2) the T1/T2echo coherence floor."""
    ref_pts, floor_pts, used_refs = [], [], set()
    for c in cells:
        for t in c["targets"]:
            if t["status"] != "completed" or t["gate_fid"] is None:
                continue
            err = 100 * (1 - t["gate_fid"])
            label = f"{c['backend']} {t['target']}{rerun_label(c['cell'], t['target'])} · {PROVIDERS[c['provider']][0]} · {pct(t['gate_fid'])}"
            ref = reference_fidelity(c["backend"], t["target"])
            if ref is not None and ref < 1:
                used_refs.add((c["backend"], t["target"]))
                ref_pts.append((100 * (1 - ref), err,
                                label + f" · reference {pct(ref)} ({reference_date(c['backend'], t['target'])[:16]})",
                                c["provider"]))
            t1, t2e, L = t.get("t1"), t.get("t2e"), t.get("x180_len")
            if t1 and t2e and L:
                floor = 100 * (L * 1e-9 / 3.0) * (1.0 / t1 + 1.0 / t2e)  # per-gate coherence-limited error
                floor_pts.append((floor, err, label + f" · T1 {1e6 * t1:.0f} µs, T2e {1e6 * t2e:.0f} µs, x180 {L:.0f} ns · floor {floor:.3f}%", c["provider"]))
    a = _scatter_row(ref_pts, xlabel="reference calibration's gate error, %", ylabel="measured gate error, %")
    b = _scatter_row(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %",
                     ylabel="measured gate error, %", guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x"))
    return (f"<h3>Measured gate error against the reference calibration</h3>"
            f"<p class='small muted'>One dot per completed qubit-run, log axes, error per gate (error per Clifford ÷ 1.875). Below the diagonal the "
            f"framework beat the stored calibration; hover a dot for the qubit. The reference is gate_fidelity.averaged in the cloud state, re-pulled on "
            f"22 Sep 13:50 CET: {_reference_dates_phrase(used_refs)} (Israel time, the stamp the cloud stores with each value) — the last calibration on "
            f"file for that qubit, not necessarily the same RB protocol, and a qubit may have drifted since.</p>{a}"
            f"<h3>Measured gate error against the coherence floor</h3>"
            f"<p class='small muted'>The floor is (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) with the run's own T1, T2echo and x180 length — the error a perfect "
            f"gate of that length would still have. A dot on the diagonal is decoherence-limited; far above it, the control is the limit. "
            f"Runs without a T1 or T2echo measurement are not shown.</p>{b}")


def pivot_table(cells):
    """Metrics as rows; one column per host, each over all three backends. Only started targets count as attempted."""
    cols = [(PROVIDERS[t][0], t) for t in ("openrouter", "splash", "flash", "qa")]
    stats = []
    for _, token in cols:
        runs = [(c, t) for c in cells if c["provider"] == token for t in c["targets"]
                if t["status"] not in ("queued", "not run", "pending")]
        done = [(c, t) for c, t in runs if t["status"] == "completed"]
        n = len(done) or 1
        fids = sorted(t["gate_fid"] for _, t in done if t["gate_fid"] is not None)
        tot = lambda k: sum((t[k] or 0) for _, t in runs)  # noqa: E731
        tokt = lambda k: sum((t["tokens"].get(k) or 0) for _, t in runs)  # noqa: E731
        per_dev = {}
        for c, t in runs:  # per qubit: completed beats running beats a model failure beats an infrastructure-only failure
            if t["status"] == "completed":
                kind = "ok"
            elif t["status"] == "running":
                kind = "running"
            elif infra(ATTEMPT_SHORT.get((c["cell"], t["target"]), "")):
                kind = "infra"
            else:
                kind = "fail"
            per_dev.setdefault(c["backend"], {}).setdefault(t["target"], set()).add(kind)
        if token == "qa":  # qua-agents cells write their document at the end: show the running and queued ones meanwhile
            have = {(c["work"], c["cell"]) for c in cells}
            for work, cell, backend, qubits, state in QA_QUEUE:
                if (work, cell) in have:
                    continue
                kind = "running" if (RUNS / work / cell / "run.log").exists() else "queued"
                for q in qubits:
                    per_dev.setdefault(backend, {}).setdefault(q, set()).add(kind)

        def qubit_list(qs):
            out = []
            for q, kinds in sorted(qs.items()):
                if "ok" in kinds:
                    out.append(f"<span class='qok' title='calibrated'>{q}</span>")
                elif "running" in kinds:
                    out.append(f"<span class='muted' title='still running'>{q}&nbsp;(running)</span>")
                elif "queued" in kinds:
                    out.append(f"<span class='muted' title='queued'>{q}&nbsp;(queued)</span>")
                elif "fail" in kinds:
                    out.append(f"<span class='qfail' title='not calibrated'>{q}</span>")
                else:
                    out.append(f"<span class='infra' title='every attempt lost to infrastructure or serving (408/503/429, tailnet, capped loop at high)'>{q}</span>")
            return ", ".join(out)
        priced = token in ("openrouter", "flash")

        def override_share(pairs):
            w = sum(t["overrides"][0] for _, t in pairs if t["overrides"])
            o = sum(t["overrides"][1] for _, t in pairs if t["overrides"])
            return (o, w) if w else None
        ov_done = override_share(done)
        ov_rest = override_share([(c, t) for c, t in runs if t["status"] != "completed"])

        def off_order(pairs):
            tot = collections.Counter()
            for _, t in pairs:
                for c in (t["decisions"] or {}).values():
                    tot.update(c)
            return (tot["back"] + tot["ahead"], tot["runs"]) if tot["runs"] else None
        oo_done = off_order(done)
        oo_rest = off_order([(c, t) for c, t in runs if t["status"] != "completed"])
        stats.append({
            "order": ("—" if oo_done is None else
                      f"{100 * oo_done[0] / oo_done[1]:.0f}% ({oo_done[0] / n:.1f} / calibration)")
                     + ("" if oo_rest is None else
                        f"<br><span class='small muted'>not completed: {100 * oo_rest[0] / oo_rest[1]:.0f}% of {oo_rest[1]} node runs</span>")
                     if token != "qa" else "<span class='small muted'>not measured</span>",
            "override": ("—" if ov_done is None else
                         f"{100 * ov_done[0] / ov_done[1]:.0f}% ({ov_done[0] / n:.1f} / calibration)")
                        + ("" if ov_rest is None else
                           f"<br><span class='small muted'>not completed: {100 * ov_rest[0] / ov_rest[1]:.0f}% of {ov_rest[1]} writes</span>")
                        if token != "qa" else "<span class='small muted'>not measured</span>",
            "qubits": ("<div class='devs'>" + "".join(f"<div class='dev'>{b}</div><div>{qubit_list(qs)}</div>" for b, qs in sorted(per_dev.items()))
                       + "</div>") if per_dev else "—",
            "qa_note": "",
            "done": f"{len(done)}/{len(runs)} ({100 * len(done) / len(runs):.0f}%)" if runs else "—",
            "fid": (pct(fids[len(fids) // 2]) + f"<br><span class='small muted'>{pct(fids[0])} – {pct(fids[-1])}</span>" + fidelity_strip(done)) if fids else "—",
            "agent": minutes(tot("model_s") / n), "qpu": minutes(tot("qpu_s") / n), "queue": minutes(tot("queue_s") / n),
            "cost": f"${sum((t['cost'] or 0) for _, t in runs) / n:.2f}" if priced else "unpriced",
            "spent": f"${sum((t['cost'] or 0) for _, t in runs):.2f}" if priced else "unpriced",
            "turns": f"{tot('turns') / n:.0f}", "nodes": f"{tot('nodes_run') / n:.0f} ({tot('reruns') / n:.0f})",
            "ctx_med": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["median"] for _, t in runs if t["ctx"]["median"]]),
            "ctx_end": (lambda v: ktok(sum(v) / len(v)) if v else "—")([t["ctx"]["end"] for _, t in done if t["ctx"]["end"]]),
            "tin": mtok(tokt("input") / n), "tout": mtok(tokt("output") / n),
            "tcache": f"{mtok(tokt('cache_read') / n)} / {mtok(tokt('cache_creation') / n)}",
        } if runs or per_dev else None)
    rows = [("qubits measured (green: calibrated; red: not calibrated; grey: only infrastructure failures so far, or still running)", "qubits"), ("calibrations completed / attempted", "done"),
            ("agent overrides: share of state writes that are not a node's proposal (no node proposed the path, or the value is "
             "more than 0.1 % from the latest proposal), completed calibrations; below, the runs that did not complete", "override"),
            ("node runs off the recipe order: stepped back to an earlier step or skipped one (retries and re-runs of the node just run "
             "are not counted), completed calibrations; below, the runs that did not complete", "order"),
            ("single-qubit gate fidelity, median (min – max); dots: gate error per completed run on a log axis, line at 99.9 %, filled = at or above the qubit's reference fidelity, hollow = below it", "fid"), ("agent time / calibration", "agent"),
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
    head = ('<thead><tr><th></th>' + "".join(
        f'<th class="grp"><span class="chip"><i style="background:{base.FW_COLOR[PROVIDERS[t][2]]}"></i>{esc(label if t == "qa" else "tinycal · " + label)}</span>'
        f'<br><span class="small muted">all three backends</span></th>' for label, t in cols) + '</tr></thead>')
    return f'<div class="scroll"><table class="grid pivot">{head}<tbody>' + "".join(body) + "</tbody></table></div>"


def decision_table(cells):
    """Per node and host: how the model sequenced the node and what it did with its proposals, over every attempted qubit-run."""
    short = {"resonator_identification": "resonator identification", "resonator_spectroscopy_vs_power": "resonator vs power",
             "resonator_spectroscopy_vs_flux": "resonator vs flux", "qubit_spectroscopy": "qubit spectroscopy",
             "qubit_spectroscopy_vs_flux": "qubit vs flux", "qubit_spectroscopy_fine": "qubit spectroscopy fine",
             "power_rabi": "power Rabi", "T1_coarse": "T1 coarse", "readout_power_optimization": "readout power opt.",
             "readout_frequency_optimization": "readout frequency opt.", "IQ_blobs": "IQ blobs",
             "ramsey_vs_flux_calibration": "Ramsey vs flux", "power_rabi_error_amplification_x180": "x180 error amplification",
             "ramsey": "Ramsey", "T1": "T1", "T2echo": "T2 echo", "DRAG_calibration": "DRAG", "Randomized_benchmarking": "RB"}
    cols = [("runs", "runs"), ("failed", "failed"), ("retry", "retried after a failure"), ("repeat", "re-run right after a success"),
            ("back", "stepped back to"), ("ahead", "skipped ahead to"), ("left_failed", "left after a failure"),
            ("proposed", "runs with a proposal"), ("accepted", "accepted as proposed"), ("partial", "partly written"),
            ("changed", "value changed"), ("declined_rerun", "declined, re-ran the node"), ("declined_other", "declined, ran another node")]
    pct_of = {"failed": "runs", "accepted": "proposed", "partial": "proposed", "changed": "proposed",
              "declined_rerun": "proposed", "declined_other": "proposed"}

    def cell(c, key):
        v = c[key]
        if not v:
            return "<td class='num muted'>·</td>"
        base_n = c[pct_of[key]] if key in pct_of else 0
        share = f" <span class='small muted'>{100 * v / base_n:.0f}%</span>" if base_n else ""
        return f"<td class='num'>{v}{share}</td>"

    out = []
    for i, token in enumerate(("openrouter", "splash", "flash")):
        per, qruns = {}, 0
        for c in cells:
            if c["provider"] != token:
                continue
            for t in c["targets"]:
                d = t.get("decisions")
                if not d:
                    continue
                qruns += 1
                for node, cnt in d.items():
                    per.setdefault(node, collections.Counter()).update(cnt)
        if not per:
            continue
        for cnt in per.values():
            cnt["proposed"] = sum(cnt[k] for k in ("accepted", "partial", "changed", "declined_rerun", "declined_other"))
        total = collections.Counter()
        for cnt in per.values():
            total.update(cnt)
        rows = [f"<tr><td>{esc(short.get(n, n))}</td>{''.join(cell(per[n], k) for k, _ in cols)}</tr>" for n in RECIPE_NODES if n in per]
        rows.append(f"<tr><th class='rowh'>all nodes</th>{''.join(cell(total, k) for k, _ in cols)}</tr>")
        head = ("<thead><tr><th></th><th colspan='7' class='grp'>sequence</th><th colspan='6' class='grp'>proposals</th></tr>"
                "<tr><th>node</th>" + "".join(f"<th>{esc(lab)}</th>" for _, lab in cols) + "</tr></thead>")
        label = PROVIDERS[token][0]
        out.append(f"<details{' open' if i == 0 else ''}><summary><span class='chip'><i style='background:{base.FW_COLOR[PROVIDERS[token][2]]}'></i>"
                   f"tinycal · {esc(label)}</span> <span class='small muted'>{qruns} qubit-runs, {total['runs']} node runs</span></summary>"
                   f"<div class='scroll'><table class='grid small'>{head}<tbody>{''.join(rows)}</tbody></table></div></details>")
    return "".join(out)


def qubit_table(cells):
    """One row per qubit-run, both providers side by side per qubit."""
    by = {}
    for c in cells:
        for t in c["targets"]:
            by.setdefault((c["backend"], t["target"] + rerun_label(c["cell"], t["target"])), {})[c["provider"]] = (c, t)
    order = {"gilboa": 0, "qolab": 1, "arbel": 2}

    def cell_html(entry):
        if entry is None:
            return "<td colspan='6' class='muted small'>not attempted</td>"
        c, t = entry
        done = t["status"] == "completed"
        cause = "" if done else f"<br><span class='small muted'>{esc(t['cause'] if t['status'] not in ('running', 'queued', 'not run') else t['status'])}</span>"
        fid = pct(t["gate_fid"]) if done else "—"
        return (f"<td>{status_chip(t['status'])}{cause}</td><td class='num'>{t['nodes'] if t['nodes'] is not None else 0}/{t['graph'] or 18}</td>"
                f"<td class='num'>{fid}</td><td class='num'>{t['ballpark'] if t['ballpark'] is not None else '—'}/{t['graded'] or 4}</td>"
                f"<td class='num'>{minutes(t['model_s'])} · {minutes(t['qpu_s'])}</td><td class='num'>{fmt(t['turns'], '{}')} · {fmt(t['cost'], '${:.2f}')}</td>")
    rows = []
    for (b, q), prov in sorted(by.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        rows.append(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td>{cell_html(prov.get('splash'))}{cell_html(prov.get('openrouter'))}"
                    f"{cell_html(prov.get('flash'))}{cell_html(prov.get('qa'))}</tr>")
    sub = "<th>status</th><th>nodes</th><th>gate fid.</th><th>ballpark</th><th>agent · QPU</th><th>turns · cost</th>"
    head = (f'<thead><tr><th></th><th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["splash"]}">27b · Splash</th>'
            f'<th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["tinycal"]}">27b · OpenRouter</th>'
            f'<th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["flash"]}">flash · OpenRouter</th>'
            f'<th colspan="6" class="grp" style="border-left:2px solid {base.FW_COLOR["qa"]}">27b · qua-agents</th></tr>'
            f'<tr><th>qubit</th>{sub}{sub}{sub}{sub}</tr></thead>')
    return f'<div class="scroll"><table class="grid small">{head}<tbody>' + "".join(rows) + "</tbody></table></div>"


def hard_cases(cells):
    rows = []
    for c in cells:
        for t in c["targets"]:
            if t["status"] in ("running", "queued", "pending", "not run"):
                continue
            poor = t["status"] == "completed" and t["gate_fid"] is not None and t["gate_fid"] < 0.99
            invalid = INVALID_RB.get((c["backend"], c["provider"], t["target"] + rerun_label(c["cell"], t["target"])))
            if t["status"] == "completed" and not (poor or invalid):
                continue
            fid = f" · {pct(t['gate_fid'])}" if t["gate_fid"] is not None else ""
            outcome = (f"{status_chip(t['status'])} {t['nodes'] or 0}/18{fid}<br><span class='small'>{minutes(t['model_s'])} agent · "
                       f"{fmt(t['cost'], '${:.2f}')} · {fmt(t['turns'], '{}')} turns</span>")
            what = t["cause"] if t["status"] != "completed" else (invalid or "finished the graph with a poor number")
            label = t["target"] + rerun_label(c["cell"], t["target"])
            note = HARD_CASE_NOTE.get((c["backend"], c["provider"], label), HARD_CASE_NOTE.get((c["backend"], c["provider"], t["target"]), ""))
            rank = 0 if t["status"] != "completed" else (1 if invalid else 2)
            is_infra = infra(ATTEMPT_SHORT.get((c["cell"], t["target"]), ""))
            rows.append((rank, c["backend"], t["target"] + rerun_label(c["cell"], t["target"]), c["provider"], outcome, what, note, is_infra))
    rows.sort(key=lambda r: r[:4])
    body = "".join(f"<tr{' class=infra' if inf else ''}><td class='mono'>{esc(b)} {esc(q)}</td><td>{esc(PROVIDERS[p][0].split(' · ')[1])}</td><td>{o}</td>"
                   f"<td class='small'>{esc(w)}</td><td class='small'>{esc(n)}</td></tr>" for _, b, q, p, o, w, n, inf in rows)
    if not rows:
        body = "<tr><td colspan='5' class='muted'>none yet</td></tr>"
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>qubit</th><th>served by</th><th>outcome</th>'
            '<th>what happened</th><th>what would have caught it</th></tr></thead><tbody>' + body + "</tbody></table></div>")


# ----------------------------------------------------------------------------- final per-target table
# Short labels for attempts that did not finish, keyed by (cell, target); anything not listed shows its status.
ATTEMPT_SHORT = {
    ("qolab-tinycal-qwen3-8-27b-splash", "Q1"): "capped loop, high",
    ("qolab-tinycal-qwen3-8-27b-splash-B", "Q6"): "capped loop, high",
    ("qolab-tinycal-qwen3-8-27b-openrouter", "Q2"): "readout power",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash", "Q2"): "QPU budget",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash", "Q3"): "QPU budget",
    ("arbel-tinycal-qwen3-8-27b-openrouter", "qB4"): "two-photon line",
    ("arbel-tinycal-qwen3-8-27b-openrouter", "qC2"): "two-photon line",
    ("arbel-tinycal-qwen3-8-27b-splash", "qB4"): "408",
    ("arbel-tinycal-qwen3-8-27b-splash", "qC2"): "408",
    ("arbel-tinycal-qwen3-8-27b-splash-B", "qB4"): "capped loop, high",
    ("arbel-tinycal-qwen3-8-27b-splash-C", "qC2"): "503",
    ("arbel-tinycal-qwen3-8-27b-splash-D", "qC2"): "wrong line",
    ("arbel-tinycal-qwen3-8-27b-splash-E", "qB4"): "two-photon line",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash", "qB4"): "two-photon line",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash2", "qC2"): "429",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash3", "qC2"): "429",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash4", "qC2"): "429",
    ("gilboa-tinycal-qwen3-8-27b-openrouter", "qC2"): "flat Rabi",
    ("gilboa-tinycal-qwen3-8-27b-openrouter", "qC5"): "flux apex from a rejected fit",
    ("gilboa-tinycal-qwen3-8-27b-openrouter", "qD5"): "flux apex from a rejected fit",
    ("gilboa-tinycal-qwen3-8-27b-openrouter-R", "qC5"): "flux apex from a rejected fit",
    ("gilboa-tinycal-qwen3-8-27b-splash-A", "qC5"): "capped loop, high",
    ("gilboa-tinycal-qwen3-8-27b-splash-C", "qD1"): "capped loop, high",
    ("gilboa-tinycal-qwen3-8-27b-splash-D", "qD4"): "408",
    ("gilboa-tinycal-qwen3-8-27b-splash-F", "qC2"): "tailnet drop",
    ("gilboa-tinycal-qwen3-8-27b-splash-F", "qC3"): "capped loop, medium",
    ("qolab-tinycal-qwen3-8-27b-splash-C", "Q1"): "wrong line, readout 0.02 V",
    ("qolab-qwen3-8-27b-openrouter", "Q2"): "T2echo loop, recursion limit",
    ("qolab-qwen3-8-27b-openrouter", "Q3"): "never found the qubit, recursion limit",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash2", "Q2"): "429",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash2", "Q3"): "429",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash5", "qB4"): "429",
    ("arbel-tinycal-qwen3-8-flash-openrouter-flash5", "qC2"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC1"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC2"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC3"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC4"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qC5"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash", "qD1"): "429",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash6", "Q2"): "429",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash6", "Q3"): "429",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash8", "Q2"): "429",
    ("qolab-tinycal-qwen3-8-flash-openrouter-flash8", "Q3"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qC1"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qC2"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qC3"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qC4"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qC5"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qD1"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qD2"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qD3"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qD4"): "429",
    ("gilboa-tinycal-qwen3-8-flash-openrouter-flash7", "qD5"): "429",
}
# Attempts that do not count as attempts (a stray process, not a graded run).
ATTEMPT_EXCLUDE = {("gilboa-tinycal-qwen3-8-27b-splash-D", "qC2")}
# Notes on completed runs whose number needs a reason, and the earlier failure worth naming next to a rerun's result.
FINAL_NOTE = {("gilboa", "openrouter", "qC5"): "weak χ, 1 µs gates, ref 96.8", ("gilboa", "openrouter", "qD2"): "T1 0.7 µs, ref 98.8",
              ("gilboa", "splash", "qD2"): "T1 0.7 µs, ref 98.8", ("gilboa", "flash", "qD2"): "T1 0.7 µs, ref 98.8",
              ("gilboa", "openrouter", "qD1"): "extrapolated"}
SERVER_CAUSES = {"408", "503", "429", "tailnet drop"}
# Failures that are the infrastructure's or the serving configuration's, not the model's: greyed out in the tables.
# 'capped loop, high' = Splash at effort high spending its whole 10k output cap on reasoning (the level was changed to medium).
INFRA_SHORT = SERVER_CAUSES | {"capped loop, high"}
def infra(short):
    return bool(short) and any(short == k or short.startswith(k + " ") or short.startswith(k + ",") for k in INFRA_SHORT)
FIRST_RUN_NOTE = {("arbel", "openrouter", "qB4"): "two-photon line", ("arbel", "openrouter", "qC2"): "two-photon line"}


def final_table(cells):
    """One row per qubit, one column per host: the best settled run, with the failed attempts before it in brief."""
    attempts = {}  # (backend, target, provider) -> [(cell, t)] in order of the cell's start
    def started(c):
        try:
            return datetime.strptime("2026 " + (c["started"] or ""), "%Y %d %b %H:%M")
        except ValueError:
            return datetime.max
    for c in sorted(cells, key=started):
        for t in c["targets"]:
            if t["status"] == "not run" or (c["cell"], t["target"]) in ATTEMPT_EXCLUDE:
                continue
            attempts.setdefault((c["backend"], t["target"], c["provider"]), []).append((c, t))

    def label(c, t):
        """(text for the table, short reason) of one attempt."""
        st = t["status"]
        note = FINAL_NOTE.get((c["backend"], c["provider"], t["target"]))
        if st == "completed":
            if INVALID_RB.get((c["backend"], c["provider"], t["target"] + rerun_label(c["cell"], t["target"]))):
                return ("completed, RB invalid" + (f" ({note})" if note else "")), None
            fid = f"{100 * t['gate_fid']:.2f}" if t["gate_fid"] is not None else "completed"
            return (f"{fid} ({note})" if note else fid), None
        if st in ("running", "queued"):
            return st, st
        plain = {"escalated": "stuck", "failed": "aborted"}.get(st, st)
        short = ATTEMPT_SHORT.get((c["cell"], t["target"]))
        rb = gate_fidelity(t["rb"]) if t.get("rb") is not None else None
        if st == "escalated" and rb is not None and 0.5 < rb < 0.99:
            return f"{100 * rb:.1f} ({short or plain})", short
        if st == "failed" or short in SERVER_CAUSES:  # a serving failure names its cause alone: "408", "tailnet drop"
            return (short or plain), short
        return (f"{plain} ({short})" if short else plain), short

    def cell_text(key):
        runs = attempts.get(key)
        if not runs:
            return "—", None
        both = [label(c, t) for c, t in runs]
        labels = [b[0] for b in both]
        shorts = [b[1] for b in both]
        grey = lambda txt, sh: f"<span class='infra'>{esc(txt)}</span>" if infra(sh) else esc(txt)  # noqa: E731
        last_c, last_t = runs[-1]
        if last_t["status"] == "completed":
            text = esc(labels[-1] + rerun_label(last_c["cell"], last_t["target"]).replace(" · rerun", " · r").replace("r ", "r"))
            first = FIRST_RUN_NOTE.get((last_c["backend"], last_c["provider"], last_t["target"]))
            if first and len(runs) > 1:
                text += f" (first run: {grey(first, first)})"
            return text, ("valid" if last_t["gate_fid"] is not None else "invalid")
        # every attempt failed (or the last is still running): the chain, repeats collapsed
        if len(runs) > 1 and all(t["status"] == "failed" for _, t in runs) and len(set(labels)) == 1:
            turns = max((t["turns"] or 0) for _, t in runs)
            txt = f"never ran ({labels[0]} ×{len(runs)})" if turns <= 1 else f"{labels[0]} ×{len(runs)}, never past turn {turns}"
            return grey(txt, shorts[0]), None
        parts = []
        for i, (l, sh) in enumerate(zip(labels, shorts)):
            if parts and parts[-1][0] == l:
                parts[-1][1] += 1
            else:
                parts.append([l, 1, i, sh])
        return " → ".join((("r " if i > 0 else "") + grey(l + (f" ×{n}" if n > 1 else ""), sh)) for l, n, i, sh in parts), None

    order = {"qolab": 0, "arbel": 1, "gilboa": 2}
    targets = sorted({(b, q) for b, q, _ in attempts}, key=lambda k: (order.get(k[0], 9), k[1]))
    provs = ["openrouter", "splash", "flash", "qa"]
    rows, finished, via_rerun, attempted = [], {p: 0 for p in provs}, {p: 0 for p in provs}, {p: 0 for p in provs}
    for b, q in targets:
        tds = []
        for pv in provs:
            text, kind = cell_text((b, q, pv))
            if (b, q, pv) in attempts:
                attempted[pv] += 1
            if kind in ("valid", "invalid"):
                finished[pv] += 1
                if " · r" in text:
                    via_rerun[pv] += 1
            klass = "" if kind == "valid" else (" class='muted'" if text == "—" else " class='small'")
            tds.append(f"<td{klass}>{text}</td>")
        rows.append(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td>{''.join(tds)}</tr>")
    foot = "".join(f"<td><b>{finished[pv]}/{attempted[pv]}</b>" + (f" ({via_rerun[pv]} via rerun)" if via_rerun[pv] else "") + "</td>" for pv in provs)
    head = ("<thead><tr><th>qubit</th>" + "".join(f"<th>{esc(' · '.join(reversed(PROVIDERS[pv][0].split(' · ', 1))))}</th>"
            for pv in provs) + "</tr></thead>")
    return (f'<div class="scroll"><table class="grid small">{head}<tbody>' + "".join(rows) +
            f"<tr><td><b>finished</b></td>{foot}</tr></tbody></table></div>")


def in_flight(cells):
    items = []
    for c in cells:
        for t in c["targets"]:
            if t["status"] in ("running", "queued"):
                items.append(f"{c['backend']} {t['target']}{rerun_label(c['cell'], t['target']).replace(' · ', ' ')} ({PROVIDERS[c['provider']][0].split(' · ')[1]}, {t['status']})")
    have = {(c["work"], c["cell"]) for c in cells}
    for work, cell, backend, qubits, state in QA_QUEUE:
        if (work, cell) in have:
            continue
        log = RUNS / work / cell / "run.log"
        if log.exists() and state.startswith("queued"):  # an explicit state (frozen, stopped) is the operator's, not the log's
            state = f"running since {datetime.fromtimestamp((RUNS / work / cell).stat().st_mtime).strftime('%H:%M')}"
        items.append(f"{backend} {' '.join(qubits)} (qua-agents, {state})")
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
    settled = [(c, t) for c, t in runs if t["status"] not in ("running", "queued", "pending", "not run")]
    done = [(c, t) for c, t in runs if t["status"] == "completed"]
    flight = in_flight(cells)
    live = bool(flight) or any(not c["final"] for c in cells)
    spent = sum((t["cost"] or 0) for _, t in runs)
    now = datetime.now().strftime("%d %b %Y %H:%M")
    spent_27b = sum((t["cost"] or 0) for c, t in runs if c["provider"] == "openrouter")
    spent_flash = sum((t["cost"] or 0) for c, t in runs if c["provider"] == "flash")
    n_27b = sum(1 for c, t in runs if c["provider"] == "openrouter" and t["status"] not in ("queued", "pending", "not run"))
    n_flash = sum(1 for c, t in runs if c["provider"] == "flash" and t["status"] not in ("queued", "pending", "not run"))
    cost_line = (f"${spent:,.2f} of OpenRouter credit for the whole night at judge prices: ${spent_27b:,.2f} for {n_27b} qubit-runs of qwen3.8-27b "
                 f"(${spent_27b / max(n_27b, 1):.2f} per run, reruns included) and ${spent_flash:,.2f} for {n_flash} of qwen3.8-flash.")

    def prov_stats(token):
        d = [t for c, t in done if c["provider"] == token]
        s = [t for c, t in settled if c["provider"] == token]
        f = sorted(t["gate_fid"] for t in d if t["gate_fid"] is not None)
        return len(d), len(s), (f[len(f) // 2] if f else None), (f[0] if f else None), (f[-1] if f else None)
    so, ss, mo, lo_o, hi_o = prov_stats("openrouter")
    sp, sps, mp, lo_p, hi_p = prov_stats("splash")
    tiles = [
        f'<div class="tile"><div class="v">{len(done)}/{len(settled)}</div><div class="k">calibrations completed / settled{" so far" if live else ""}'
        f'{" · " + str(len(flight)) + " in flight or queued" if flight else ""}</div></div>',
        f'<div class="tile"><div class="v">{so}/{ss}</div><div class="k">OpenRouter: completed / settled — median {pct(mo)} ({pct(lo_o)} – {pct(hi_o)})</div></div>',
        f'<div class="tile"><div class="v">{sp}/{sps}</div><div class="k">Splash: completed / settled — median {pct(mp)} ({pct(lo_p)} – {pct(hi_p)})</div></div>',
        f'<div class="tile"><div class="v">${spent:,.2f}</div><div class="k">judge-priced spend{" so far" if live else ""} (OpenRouter only; Splash is self-hosted and unpriced)</div></div>',
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
<style>{CSS}
table.pivot .devs{{display:grid;grid-template-columns:max-content 1fr;column-gap:14px;row-gap:3px;align-items:baseline}}
.pivot .qok{{color:var(--good);font-weight:600}}
.stripwrap{{margin-top:6px;text-align:left;max-width:220px}} .stripwrap .small{{white-space:normal;line-height:1.35;font-family:inherit}}
svg.strip{{display:block;max-width:100%}} svg.strip .axis{{stroke:var(--muted,#8a8f98);stroke-width:.8}}
svg.strip .tick{{font-size:8px;fill:var(--muted,#8a8f98);font-family:inherit}} svg.strip .thr{{stroke:var(--crit,#b91c1c);stroke-width:1;stroke-dasharray:3 2}}
svg.strip .dot{{fill:var(--tinycal,#2a7f62);fill-opacity:.75;stroke:none}} svg.strip .dot.hollow{{fill:none;stroke:var(--tinycal,#2a7f62);stroke-width:1.4}}
:root{{--qa:#7c3aed}} @media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--qa:#a78bfa}}}} :root[data-theme="dark"]{{--qa:#a78bfa}}
.scattergrid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:6px 0 14px}} @media (max-width:900px){{.scattergrid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media (max-width:480px){{.scattergrid{{grid-template-columns:1fr}}}} .scattergrid .panel{{min-width:0}} .scattergrid .ptitle{{margin-bottom:4px}}
svg.scatter{{display:block;width:100%;height:auto}} svg.scatter .grid{{stroke:var(--surface-2,#eef2f2);stroke-width:1}}
svg.scatter .frame{{fill:none;stroke:var(--muted,#8a8f98);stroke-width:.8}} svg.scatter .guide{{stroke:var(--muted,#8a8f98);stroke-width:1;stroke-dasharray:4 3}}
svg.scatter .tick{{font-size:10px;fill:var(--muted,#8a8f98);font-family:inherit}} svg.scatter .lab{{font-size:11px;fill:var(--ink-2,#4b555b);font-family:inherit}}
svg.scatter .pt{{fill-opacity:.8;stroke:var(--surface,#fff);stroke-width:.8}}
.infra{{color:var(--muted,#8a8f98);opacity:.75}} tr.infra td{{color:var(--muted,#8a8f98);opacity:.7}} tr.infra .chip{{opacity:.6}}</style></head><body><div class="page">
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
<h3>Final per-target results</h3>
<p class="small muted">Best settled run per qubit and host; "· r" = that result came from a rerun (r2, r3: the third, fourth attempt). Where no attempt
finished, the attempts in order, "→" between them. <span class="infra">Greyed</span>: an infrastructure or serving failure, not the model's — an
unretried 408, a 503 from a third stream, OpenRouter's 429 pool limit, the tailnet drop, or a capped reasoning loop at effort high. Updated as reruns land.</p>
{final_table(cells)}
{error_scatters(cells)}

<h3>Pins: what a new cell must use to be comparable</h3>
{pins_table(cells)}
<div class="tiles">{"".join(tiles)}</div>
<p class="small muted">How to read the tables: one qubit-run = tinycal calibrating one qubit from the scrambled state with qwen3.8-27b on one host.
Gate fidelity = 1 − (RB error per Clifford ÷ 1.875). "Per calibration" = the total over every qubit-run attempted on that backend divided by the
number completed. Agent time is time inside model calls; QPU time is execution on the chip; queue wait is the cloud queue. Splash tokens are
counted but unpriced (self-hosted). Context = prompt size per model call in tokens, from tinycal's events.jsonl.</p>
<h3>By host</h3>
{pivot_table(cells)}

<h3>What the model did with each node</h3>
<p class="small muted">Every attempted qubit-run of the host, first runs and reruns pooled; tinycal's events.jsonl only (qua-agents does not
record node proposals). <b>Sequence</b>, against the recipe order (resonator identification → … → RB, the power sweep twice, T1 and T2 echo in
either order): a failed run retried at once; a successful run re-run at once, e.g. to widen a sweep; a step the model went back to after later
steps had run; a step it jumped to past the next one; and, charged to the node, leaving it for another node while its last run had failed.
<b>Proposals</b>, per run that proposed state updates, from the writes made before the next node run: every proposed value written as
proposed; only some of them written; at least one written with a different value; none written, then the same node re-run or another node
run. Percentages are of the node's runs (failed) or of its runs with a proposal.</p>
{decision_table(cells)}

<h3>Hard cases: qubits that were not calibrated, or not calibrated right</h3>
<p class="small muted">One row per settled qubit-run that did not reach the end of the graph, then the completed ones under 99 %. Running and queued
targets are not listed. Greyed rows are infrastructure or serving failures (408/503/429, tailnet, capped reasoning at effort high), not the
model's. The right-hand column is the operator's reading of the transcript, not something the cell recorded.</p>
{hard_cases(cells)}

<h3>Per qubit, host side by side</h3>
{qubit_table(cells)}
{cost_fig}
{err_fig}

<h3>What the numbers say{" (so far)" if live else ""}</h3>
<p><b>The gilboa results of the last two nights were measured at the wrong flux, and it was the state, not the nodes.</b> Since the 20 Sep pulls the
gilboa cloud state lists only qC2 as active; quam_builder parks every non-active z line at 0 V, so night 3 and the first cells of this night calibrated
gilboa's qubits 14–30 mV from their sweet spots without any node saying so — the flux maps were correct in absolute volts and the harness added the
apex to an idle that was never applied. One line in the state (active_qubit_names) explains the qD5 99.27 % / 97.72 %, the "+20 mV per map" walk and
qC5's blank maps; patched in the run copies at 17:10, the relaunched gilboa qD3 landed its offset within 0.6 mV of the reference. This and the qC5
y90 literal are both cloud-state defects for IQCC.</p>
<p><b>Same model, same recipe, same qubits: the serving host decided the night far more than the model did.</b> Counting a target as done when
any of its runs reached the end of the graph with a valid RB: <b>qwen3.8-27b on OpenRouter finished 19 of 19</b> (qolab 6/6, arbel 3/3, gilboa 10/10; four
needed a rerun, one of them twice), <b>the same weights on Splash finished 11 of 19</b> (qolab 4/6, arbel 1/3, gilboa 6/10), and <b>qwen3.8-flash on
OpenRouter 5 of 9</b> (qolab 4/6, arbel 1/3; it had no gilboa cells). Where both hosts finished, they agree: 99.9 %-class RB on the good qubits (qolab Q3–Q5,
gilboa qD3/qD4/qD5, arbel qC3 at 99.74 vs 99.80), and the same physics-limited numbers on the bad ones (gilboa qC1 99.7/99.0, qC4 99.5/99.1). The model
calibrates; nothing in the Splash column failed because Splash's answers were worse when they arrived.</p>
<p><b>What Splash lost.</b> Every Splash target that did not finish died of serving, not physics: two HTTP 408s tinycal did not retry (fixed at 10:45),
one tailnet drop, one 503 storm from a third stream, and six capped-reasoning loops — three consecutive 10 000-token turns with no tool call — at effort
high (qolab Q1/Q6, gilboa qC5/qD1, arbel qB4) and one at medium (gilboa qC3, at 58k tokens of context). The loops are the model spending its whole output
budget thinking; the same turns replayed at effort medium make the same decisions in 840–1 280 reasoning tokens. Speed was the other cost: at effort
high with two or three streams a Splash bring-up took 2.3–5.5 h against 0.5–1.5 h on OpenRouter, because Splash's KV pool (~180k tokens) cannot hold two
60k-token conversations plus generation, so prefixes were evicted and re-prefilled at ~7–30 tok/s. The one cell that ran as the sole stream at medium —
Splash-H qD4, after the stray third stream was killed at 23:14 — finished in 35 minutes with 99 % cache hits: OpenRouter speed on a laptop. Three
concurrent streams, which the plan assumed Splash could take, it cannot; two is marginal past 60k tokens; one is fine.</p>
<p><b>What flash lost.</b> qwen3.8-flash is fast and right when its first pass is right (qolab Q1/Q4/Q6 at 99.93–99.96 %, arbel qC3 99.78) and burns the
30-minute QPU budget when it is not: it re-runs nodes rather than reasoning about them (qolab Q2/Q3, arbel qB4 out of budget), and OpenRouter
rate-limited the model for the whole retry ladder four times on arbel qC2. Cheap per token, expensive per calibration.</p>
<p><b>The physics traps were shared, and two of them are now node fixes.</b> (1) The resonator flux map proposed an apex from a fit it had rejected
(gilboa qC5, qD5, qD4, three hosts); it now tracks the dip on the background-subtracted map and proposes only what the data supports — 511 stored maps
checked, gilboa qC2/qC5 reruns landed within 10 mV of the reference. (2) The power-Rabi node let a model-free "first turning point" override a good fit
unconditionally; on gilboa qC3 that turned a drive switch-on dip at 20 mV into the π amplitude (24× too small) and started a capped loop. The turning
point now has to be a Rabi turning point — same direction as the fitted lobe, not far below it — before it wins (369 stored sweeps checked). (3) Not yet
fixed: <b>the 0→2 two-photon line</b>. At the default 0.5× saturation drive the two-photon transition, half an anharmonicity below f_01, is sharp and
16σ while the power-broadened f_01 is a 4σ hump, so qubit_spectroscopy identified it — it moves with flux exactly like f_01 — on arbel qB4 (three of four
runs) and qC2. Everything downstream then measures |2⟩: a π pulse "needing 4 V", x90 ≠ x180/2, a T1 that is the |2⟩→|1⟩→|0⟩ cascade, 92 % RB. The
discriminator is the power scaling (two-photon contrast falls as amplitude⁴), which needs a low-drive condition in the identification node; α is not
reliably known at that stage, so it cannot be a lookup. (4) gilboa's cloud state parked non-active z lines at 0 V (above), qC5's weak χ and 1 µs gates
(95.9 %, reference 96.8 %) and qD2's 0.7 µs T1 (95.9 %, reference 98.8 %) are chip facts, not calibration failures — both land near the
reference fidelities of those qubits.</p>
<p><b>22 Sep, after the first pass.</b> The eight Splash targets the first pass lost are being re-run as one Splash stream at effort medium on the
fixed nodes (qolab Q1 Q6, then arbel qB4 qC2, then gilboa qC3 qC5 qD1 qD2 — chain launched 23:51); their rows land in the tables above as they
finish, labelled "· rerun". The tallies in this section are those of the first pass.</p>
<p><b>qua-agents, same snapshots, same model.</b> From 00:44 on 22 Sep the qua-agents framework runs qwen3.8-27b via OpenRouter on the
byte-identical scrambled states and the same qua-libs (65b4755): arbel qB4 qC2 qC3, qolab Q1 Q2 Q3, gilboa qD3 qD5 qC3, three workers per
backend, one cell per backend as its Splash stream ends. Its column fills in the tables above as cells finish (a qua-agents cell writes its
document only at the end; the 16–17 Sep cells of this pairing took ~7 h for three qubits).</p>
<p><b>Cost.</b> {cost_line} Splash's marginal cost was the electricity of one MacBook and the operator's evening.</p>

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
