#!/usr/bin/env python3
"""Build the 28-29 Sep 2026 report: a full single-qubit bring-up with tinycal and qwen3.8-27b via OpenRouter on every usable qubit
of qolab, arbel and gilboa (gilboa's B row excluded), on qua-libs with the bring-up graph's sweeps sized to their fits and active
reset after IQ_blobs, from the cell documents under ~/qab-runs/n10-* and ~/qab-runs/n11-*.

    uv run --project ~/code/QM/qua-agents-benchmark python make_n10_report.py

Numbers come only from each cell's result.json (projected from tinycal's events.jsonl by scripts/project_tinycal.py, then stamped
by qab inspect-state / validate / accept), the cell's final quam_state, the tinycal run dirs (events.jsonl: per-node QPU time) and
the active-reset test records in ~/qab-runs/n10-active-reset/out. The comparison column is the 20-21 Sep night's OpenRouter cells
(make_night4_report.py), same model, same host, class defaults. The operator notes (INCIDENTS, NARRATIVE) are marked as such.
Table builders are shared with make_fwcmp2_report.py and make_night4_report.py.
"""
from __future__ import annotations

import collections
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fwcmp2_report as base  # noqa: E402
import make_night4_report as night4  # noqa: E402
from make_fwcmp2_report import (  # noqa: E402
    CSS, TINYCAL_RUNS, RUNS, bars, context_series, esc, fmt, g, gate_fidelity, ktok, load_prices, local, median, minutes,
    mtok, pct, status_chip, token_cost, x180,
)

OUT = Path(__file__).with_name("2026-09-28-full-bringup-fast-nodes-qwen-tinycal.html")
ARTIFACT_DIR = __import__("os").environ.get("REPORT_ARTIFACT_DIR")  # also write the page as an Artifact body here
MODEL_ID = "qwen/qwen3.8-27b"
BACKENDS = ("qolab", "arbel", "gilboa")
QUA_LIBS = Path.home() / "qab-runs/qua-libs-n10"
ACTIVE_RESET_OUT = RUNS / "n10-active-reset/out"

# The bring-up recipe's order (tinycal bringup_recipes/flux_tunable_1q.md at 729535c): one entry per step; the power sweep appears
# twice (step 4 repeats it at the sweet spot), T1_chirp sits between the line search and the flux map, T1/T2echo share a step.
RECIPE = [("resonator_identification",), ("resonator_spectroscopy_vs_power",), ("resonator_spectroscopy_vs_flux",),
          ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy",), ("T1_chirp",), ("qubit_spectroscopy_vs_flux",),
          ("qubit_spectroscopy_fine",), ("power_rabi",), ("T1_coarse",), ("readout_power_optimization",),
          ("readout_frequency_optimization",), ("IQ_blobs",), ("ramsey_vs_flux_calibration",),
          ("power_rabi_error_amplification_x180",), ("ramsey",), ("T1", "T2echo"), ("DRAG_calibration",), ("Randomized_benchmarking",)]
RECIPE_NODES = list(dict.fromkeys(n for step in RECIPE for n in step))
night4.RECIPE[:] = RECIPE  # node_decisions() reads these module globals
night4.RECIPE_NODES[:] = RECIPE_NODES
SHORT = {"resonator_identification": "resonator identification", "resonator_spectroscopy_vs_power": "resonator vs power",
         "resonator_spectroscopy_vs_flux": "resonator vs flux", "qubit_spectroscopy": "qubit spectroscopy (chirp)",
         "T1_chirp": "T1 (chirp)", "qubit_spectroscopy_vs_flux": "qubit vs flux (chirp)", "qubit_spectroscopy_fine": "qubit spectroscopy fine",
         "power_rabi": "power Rabi", "T1_coarse": "T1 coarse", "readout_power_optimization": "readout power opt.",
         "readout_frequency_optimization": "readout frequency opt.", "IQ_blobs": "IQ blobs",
         "ramsey_vs_flux_calibration": "Ramsey vs flux", "power_rabi_error_amplification_x180": "x180 error amplification",
         "ramsey": "Ramsey", "T1": "T1", "T2echo": "T2 echo", "DRAG_calibration": "DRAG", "Randomized_benchmarking": "RB"}

# QPU seconds of one execution at the node's class defaults, one qubit, T1 = 50 us: the estimate made before the run from the node
# programs (shots x sweep points x per-shot time; 28 Sep). "refused" = over the 60 s job pre-flight at those defaults.
CLASS_DEFAULT_ESTIMATE = {
    "resonator_identification": (1.7, ""), "resonator_spectroscopy_vs_power": (9.0, "63-point grid since c999fff"),
    "resonator_spectroscopy_vs_flux": (7.3, ""), "qubit_spectroscopy": (10.4, "5.4 once T1 is known"), "T1_chirp": (10.9, ""),
    "qubit_spectroscopy_vs_flux": (9.9, ""), "qubit_spectroscopy_fine": (1.5, "graph preset"), "power_rabi": (5.0, ""),
    "T1_coarse": (5.3, "graph preset"), "readout_power_optimization": (40.0, ""), "readout_frequency_optimization": (5.0, ""),
    "IQ_blobs": (2.0, ""), "ramsey_vs_flux_calibration": (12.0, "refused by its pre-flight (priced a thermal reset it never does)"),
    "power_rabi_error_amplification_x180": (26.0, ""), "ramsey": (51.0, "refused (pre-flight 77 s)"),
    "T1": (186.0, "refused"), "T2echo": (19.0, ""), "DRAG_calibration": (488.0, "refused"), "Randomized_benchmarking": (11.0, ""),
}
# What the graph now presets on each node (feat/bringup-fast-presets, 7bbb286 + 08ebca5); nodes not listed run their class defaults.
PRESET_NOTE = {
    "readout_power_optimization": "2000 shots × 12 amplitudes (was 4000 × 20)",
    "ramsey_vs_flux_calibration": "defaults; pre-flight now prices the depletion wait (62e9bdf)",
    "power_rabi_error_amplification_x180": "amplitudes 0.9–1.1 (was 0.8–1.2), active reset",
    "T1": "200 shots × 40 log delays 1–300 µs (was 1000 × 703, 16 ns–60 µs), thermal reset",
    "ramsey": "101 idle times at 40 ns (was 997 at 4 ns), active reset",
    "T2echo": "60 log idle times from 200 ns (was ~703 from 16 ns), thermal reset",
    "DRAG_calibration": "200 shots, α step 0.1, pulse ceiling 100 (was 400, 0.05, T1-derived ~170), active reset",
    "Randomized_benchmarking": "defaults, active reset",
}

# ----------------------------------------------------------------------------- operator notes (filled in by hand)
SNAPSHOT_NOTE = {
    "qolab": "fresh pull at 23:04 on 28 Sep (n10), nothing to patch",
    "arbel": "fresh pulls at 23:04 on 28 Sep (n10) and at the start of n11; readout alias #./readout_square reversed "
             "(patch_readout_alias.py) so the scrambler can follow it, qC5's y90 made a reference to x90",
    "gilboa": "fresh pulls at 23:04 on 28 Sep (n10) and at the start of n11; active_qubit_names set to the ten C/D qubits "
              "(the pull lists them already; the patch is kept as a guard), the B row excluded (broken TWPA on its readout line)",
}
# (time, what happened): the operator's log, ~/qab-runs/n10-LOG.md
INCIDENTS: list[tuple[str, str]] = [
    ("28 Sep 23:02", "qua-libs worktree ~/qab-runs/qua-libs-n10 on feat/bringup-fast-presets off feat/qualibrate-ai 9535a06: 62e9bdf (Ramsey vs "
                     "flux pre-flight priced on the depletion wait) and 7bbb286 (graph presets for T1, Ramsey, T2 echo, error amplification, "
                     "readout power and DRAG; DRAG's new max_derived_pulses). 111 related tests pass; one power-Rabi analysis test fails on "
                     "9535a06 as well."),
    ("23:04", "Fresh pulls of qolab, arbel and gilboa, scrambled: every target has 29 scrambled fields (T1 is not among them)."),
    ("23:05–23:10", "Active reset tested on qolab Q2 (the table above). The class-default DRAG timed out at 60 s with active reset although the "
                    "pre-flight had passed it: the pre-flight priced active reset at three readouts."),
    ("≈23:15", "08ebca5: active reset on error amplification, Ramsey, DRAG and RB; DRAG at 200 shots and α step 0.1; the pre-flight prices "
               "active reset at the measured 60 µs per shot."),
    ("23:17", "n10 launched: 18 single-qubit cells (qolab Q1–Q6, arbel qB4 qC2 qD2 qD1 qC3 qA4, gilboa qC3 qD5 qD4 qD2 qD3 qC5), at most three "
              "per device — the user's cap for this run."),
    ("23:38", "First bring-up finished: qolab Q3, 99.90 % in 19 min wall and 1.9 min of QPU. The judge flags its f_01 (5.0610 GHz against a "
              "5.0855 GHz reference); the value is within 10 kHz of the 20 Sep reference, and the 28 Sep reference comes with a curvature of "
              "−5.7 GHz/V² against −2.2 before. n9 (28 Sep 18:58) committed the same point."),
    ("23:42", "n11 scheduler started: waits for n10's queue to empty, then pulls fresh arbel and gilboa states and fills free slots, counting "
              "n10's cells still running against the three-per-device cap."),
    ("00:12", "gilboa qD2 finished at 97.46 % on a 0.6 µs T1; the lab's own reference for the qubit is 89.50 %."),
    ("00:17", "n11 prepared (pulls at 00:17, 19 targets scrambled) and started on gilboa qC1."),
    ("00:18", "arbel qD2 stuck at turn 65: it committed a resonator 50 MHz above its own and never found the qubit (hard cases)."),
    ("00:38", "gilboa qD4 stuck at turn 99: f_01 from a fine scan the node had rejected, readout at 0.05 V (hard cases)."),
    ("00:57", "arbel qD1 stuck at turn 90: no 0→1 line anywhere in 4.6–5.4 GHz. n10 over: 15 of 18 completed."),
    ("01:04", "gilboa over: 9 of its 10 C/D qubits completed."),
    ("01:32", "arbel qA6 completed at its upper sweet spot, 234 MHz above the point the lab parks it at (the judge flags f_01)."),
    ("02:19", "Last cell launched (arbel qD5)."),
    ("02:25", "arbel qD3 stuck at turn 28: its qubit sits at the drive's +400 MHz edge and every window reaching it was refused (hard cases)."),
    ("03:29", "arbel qD5 'completed' on the two-photon line after the chirp node had flagged it three times; no valid RB."),
    ("03:36", "arbel qD4 'completed' on the two-photon line too; RB error −3·10⁻⁷. n11 over: all 19 cells finished; nothing left running."),
]
NARRATIVE: list[str] = [  # "What the numbers say": the operator's reading of the tables above, at the end of the night
    "<p><b>The graph now runs inside the job limit, and a bring-up costs a sixth of the chip time.</b> One execution of every node at the new "
    "presets is ~106 s of QPU (medians), against ~900 s estimated at the class defaults, three of whose nodes could not run at all under IQCC's "
    "60 s cap. Per completed bring-up the median was 2.7 min of QPU (1.8–10.3). On the 18 qubits that also completed on 20–21 Sep with the same "
    "model and host, QPU fell from 15.0 to 2.4 min, and the cloud queue from 14.6 to 2.8 min — shorter jobs wait less behind each other. The "
    "fidelities did not pay for it: on the 15 of those qubits with a valid RB both nights, the median is 99.89 % tonight against 99.84 %, eight "
    "better, six worse, one level. The nodes that were trimmed now cost 2–9 s each; what is left is resonator vs power, qubit vs flux and power Rabi "
    "(~12 % of the QPU each, the last mostly from the model re-running it at more shots) and Ramsey vs flux (11 %) — which now runs at class "
    "defaults since its pre-flight stopped refusing them, and is the next one to size.</p>",
    "<p><b>The clock is now the model's.</b> A completed bring-up took a median 25 min of wall time: 16 min inside model calls, 4 min in the "
    "cloud queue and 2.7 min on the chip (11 % of the wall). On the same 18 qubits the wall time halved (54 → 25 min) while the model time did "
    "not move (17.6 → 17.3 min). The next speed-up is fewer turns (a median 66 per bring-up, 27 node runs) or faster ones, not the chip: the "
    "hardware part of a single-qubit bring-up is already under three minutes, the agent part is not near it.</p>",
    "<p><b>33 of 37 qubits reached the end of the graph; 28 of them with an RB number that is a measurement.</b> qolab 6/6 (5 valid), arbel "
    "18/21 (15), gilboa 9/10 (8). Median gate fidelity over the 28: 99.905 %, 15 at ≥ 99.9 %, 24 at ≥ 99.5 %, and 19 of the 27 with a reference "
    "at or above the lab's own number (the calibration on file when the qubit ran). Five of the 28 are extrapolated (1.2–2.0 decay lengths) and marked. The five completions without a valid "
    "RB are two different failures: three where the model cut RB's max depth to 512 or less and accepted the node's own 'extrapolated' warning "
    "(qolab Q2 claimed 99.98 % on 0.31 decay lengths), and two calibrated end to end on the wrong transition (next paragraph). A 'completed' "
    "that the judge cannot tell from a good one is the most expensive outcome here; the RB node should withhold its error below one decay "
    "length, or the graph should pin max_circuit_depth.</p>",
    "<p><b>arbel's D row sits at the edge of its drive LOs, and the search could not reach it.</b> The lab drives qD2–qD5 at +399 to "
    "+408 MHz from their LOs, and the chirp nodes refused any window reaching ±400 MHz, so none of those qubits' 0→1 lines was ever inside a "
    "window. qD2 and qD3 found nothing. On qD5 and qD4 the only line in reach was the two-photon 0→2 line, 102 and 99 MHz below f_01. The "
    "node labelled it 'ambiguous' and proposed nothing — three times on qD5 ('grows as drive^3.2–3.7 with no partner line below it … scan "
    "higher') — but reported the run successful, and the model wrote the frequency itself, then went on to 500–800 ns π pulses, a 'sweet "
    "spot at the frequency minimum' and negative RB errors. Fixed after the run (qua-libs f5464c5, 98edd26): a spectroscopy window past the "
    "band edge is played with the LO moved for that run, and an ambiguous line is a failed outcome. At the node defaults on arbel at 11:40, "
    "the search then found qD5 at 5.907913 GHz and qD4 at 6.636046 GHz, 17 and 71 kHz from the lab, each with its 0→2 partner.</p>",
    "<p><b>Readout power is capped below where the labs run it.</b> The resonator power sweep stops at qua-libs' MW-FEM amplitude cap of 0.1 V "
    "(−10 dBm at a +10 dBm full scale); the full-scale powers did not change, so amplitudes compare directly. The lab runs gilboa's D row at "
    "0.24–0.48 V, gilboa qC1/qC2/qC5 at 0.16–0.21 V and arbel qB4 at 0.26 V. There tinycal read out 8–18 dB weaker and gilboa's D row reached "
    "83–85 % assignment; on gilboa qD4 the model went to 0.05 V (0.12× the lab's), got 52 % assignment, never saw its qubit in six chirp "
    "searches, and then wrote a fine-scan frequency the node had marked 'NO LINE DETECTED'. Where the lab runs below the cap (qolab, arbel's "
    "A/C rows) the knee rule landed at 0.4–1.4× the lab's power with 80–98 % assignment. The judge flags readout amplitude on 15 of the 33 "
    "completions — mostly this.</p>",
    "<p><b>The four stuck qubits are four different lessons.</b> arbel qD2 committed a dip 50 MHz above its own resonator; the tell was its "
    "0.44 MHz flux arc under a 2 MHz linewidth. arbel qD3's qubit sits at +399 MHz of a fixed LO; every window reaching it crossed the ±400 MHz "
    "limit and was refused, and the model concluded at turn 28 that there was nothing there (the D-row edge above, now fixed). gilboa qD4 is the readout "
    "cap above. arbel qD1 searched the right range at the right flux and correctly rejected a flux-static feature; the chirp never saw the "
    "qubit — a 'bad' qubit in the matrix notes, one for the operators.</p>",
    "<p><b>Where the judge disagrees with the lab, the lab is sometimes the odd one.</b> qolab Q3's f_01 matches the 20 Sep reference to 10 kHz; "
    "the 28 Sep reference comes with a different curvature. arbel qA6 and qB3 were calibrated at their upper sweet spots, 234 and 33 MHz above "
    "where the lab parks them — the recipe says the apex, and qA6 then beat the lab's own RB (99.76 % vs 99.49 %). arbel qC3's x180 amplitude "
    "is 'wrong' because the model lengthened the pulse from 96 to 160 ns. Only 7 of 33 completions have all four graded parameters in the "
    "ballpark, and most of the misses are one of these, not a bad calibration.</p>",
    "<p><b>Active reset held up.</b> Four nodes ran with it; the model switched it back to thermal on a handful of runs (Ramsey 4, RB 3, error "
    "amplification 3, DRAG 2) and no failure traces to it. Across 1 115 node runs there were 16 pre-flight refusals and 20 timeouts or read "
    "errors (≈3 %). Cost: $27.10 of OpenRouter credit at judge prices for the 37 qubit-runs — $0.82 per completed bring-up, $0.97 per valid RB.</p>",
]
ATTEMPT_SHORT: dict[tuple[str, str], str] = {  # (backend, qubit) -> short cause of a run that did not finish
    ("arbel", "qD2"): "wrong resonator, qubit never found",
    ("arbel", "qD1"): "no 0→1 line found",
    ("gilboa", "qD4"): "f_01 from a rejected fit, readout 0.05 V",
    ("arbel", "qD3"): "qubit at the drive's +400 MHz edge",
}
HARD_CASE_NOTE: dict[tuple[str, str], str] = {  # (backend, qubit) -> what would have caught it (operator's reading of the transcript)
    ("arbel", "qD2"): "It committed a dip at 7.4603 GHz, 50 MHz above qD2's own resonator (7.4101 GHz) and 25 MHz above the scrambled seed; "
                      "identification reported a 2.3-linewidth flux response, but the flux map's arc was 0.44 MHz peak to peak under a 2 MHz "
                      "linewidth. A weak arc means the dip is not this qubit's: back to identification. Its search also stopped at 6.04 GHz; "
                      "the qubit sits at 6.049 GHz.",
    ("arbel", "qD1"): "The reasoning held up: it rejected a flux-static 5.262 GHz feature on three counts (drive⁴ growth, flat Rabi, 0 of 25 flux "
                      "columns). The reference f_01, 4.9975 GHz, lies inside the range it searched at the right flux. A 'bad' qubit in the "
                      "matrix notes (reference 95.45 %); one for the operators before it is counted against a framework.",
    ("gilboa", "qD4"): "Readout at 0.05 V where the lab runs 0.40 V (above qua-libs' 0.1 V MW-FEM cap): 52 % assignment, and the chirp saw no "
                       "line in six searches up to 480 MHz wide. The model then wrote 6.404 GHz from a fine scan the node had marked "
                       "\"NO LINE DETECTED\" (R² 0.27, 25 kHz wide); the qubit is at 6.457 GHz. Two guards: write_state refusing a frequency "
                       "from a rejected fit, and a readout-power ceiling that reaches what the lab runs. 20–21 Sep: 99.92 %.",
    ("qolab", "Q2"): "RB ran once, with max_circuit_depth 512 chosen at the outset (1.0 s of QPU), and the model accepted the node's own "
                     "'extrapolated' warning. At the default 2048 with active reset the job is ~3 s. The RB node should withhold the error "
                     "below one decay length, or the graph should pin the depth.",
    ("gilboa", "qD5"): "RB at depth 512 with a 1/e depth of ~800 Cliffords: the same cut as qolab Q2.",
    ("arbel", "qC1"): "RB depth cut: the fit saw 0.72 decay lengths.",
    ("arbel", "qD3"): "The qubit (6.067 GHz) sits at +399 MHz of a drive LO fixed at 5.668 GHz. Every chirp window that reached it crossed the "
                      "±400 MHz limit and the node refused it (four times; the LO change was refused as an identity field), so the model "
                      "concluded at turn 28 that 5.17–6.17 GHz held no line. n9 hit the same refusals and committed the line 100 MHz below "
                      "instead — the two-photon line. Since qua-libs f5464c5 the node moves the LO for its own run instead of refusing.",
    ("arbel", "qD5"): "Calibrated on the two-photon line: f_01 5.8054 GHz against 5.9079 (−102 MHz, half the anharmonicity). The real "
                      "line sits at +408 MHz of the drive LO, outside every window the node would play. The chirp node called the line it "
                      "could reach ambiguous three times — 'grows as drive^3.2–3.7 with no partner line below it … may be the 0→2 line of a "
                      "qubit above the window; scan higher' — and proposed nothing, but reported success; the model wrote the frequency "
                      "itself and went on: a 500 ns x180, and eight RB runs with negative or unconverged errors.",
    ("arbel", "qD4"): "The same trap as qD5, its qubit at +401 MHz of the LO: f_01 6.537 GHz against 6.636 (−99 MHz), an 800 ns x180 for a 'weakly coupled' qubit, a 'sweet "
                      "spot at the qubit frequency minimum', and a final RB error of −3·10⁻⁷ (a fidelity above 100 %) reported as a "
                      "completed calibration at turn 131. A 0→1 line needs neither a 17× longer pulse nor an inverted parabola.",
}
FINAL_NOTE: dict[tuple[str, str], str] = {  # (backend, qubit) -> note next to a completed run's number
    ("gilboa", "qD2"): "T1 0.6 µs, reference 89.5",
    ("arbel", "qA6"): "at the upper sweet spot, 234 MHz above the lab's parking point",
    ("arbel", "qB3"): "at the upper sweet spot, 33 MHz above the lab's point",
    ("qolab", "Q3"): "f_01 = the 20 Sep reference; the 28 Sep one looks wrong",
    ("gilboa", "qC5"): "48 ns x180; the lab's 96.5 on the same pulse, 95.9 on 20 Sep with 1 µs gates",
    ("arbel", "qD5"): "calibrated on the two-photon line, 102 MHz below f_01",
    ("arbel", "qD4"): "calibrated on the two-photon line, 99 MHz below f_01; RB error −3e-7",
}


# ----------------------------------------------------------------------------- collect
def _events(run_id: str, target: str):
    path = TINYCAL_RUNS / run_id / target / "events.jsonl"
    if not run_id or not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def node_runs(run_id: str, target: str):
    """[(node, qpu_s, outcome, duration_s, queue_s, parameters)] for every run_node call, in order."""
    rows = []
    for e in _events(run_id, target):
        if e.get("tool") != "run_node":
            continue
        f = lambda k: float(e[k]) if e.get(k) not in (None, "None", "") else None  # noqa: E731
        rows.append((e.get("node"), f("qpu_execution_s"), e.get("outcome"), f("duration_s"), f("queue_wait_s"), e.get("parameters")))
    return rows


RB_VALID_DECAY_LENGTHS = 1.0  # below this the RB fit has not seen the decay: its error per Clifford is not a measurement
RB_EXTRAPOLATED_DECAY_LENGTHS = 2.0  # the node warns below ~2: the asymptote, and with it the error, is extrapolated


def rb_coverage(run_id: str, target: str):
    """(decay lengths covered, max_circuit_depth asked for) of the last RB run, from its numerics in events.jsonl."""
    import ast
    last = None
    for e in _events(run_id, target):
        if e.get("tool") == "run_node" and e.get("node") == "Randomized_benchmarking":
            last = e
    if last is None:
        return None, None
    try:
        num = ast.literal_eval(last["numerics"]) if isinstance(last.get("numerics"), str) else (last.get("numerics") or {})
        params = ast.literal_eval(last["parameters"]) if isinstance(last.get("parameters"), str) else (last.get("parameters") or {})
    except (ValueError, SyntaxError):
        return None, None
    cov = num.get("decay_lengths_covered")
    cov = cov.get("value") if isinstance(cov, dict) else cov
    return cov, params.get("max_circuit_depth", 2048)


def interim(run_id: str, target: str) -> dict:
    """A running target's numbers from its live event log."""
    turn, qpu, last, model_s = 0, 0.0, "-", 0.0
    for e in _events(run_id, target):
        try:
            turn = max(turn, int(e.get("turn") or 0))
        except (TypeError, ValueError):
            pass
        if e.get("tool") == "run_node":
            last = e.get("node") or last
            try:
                qpu += float(e.get("qpu_execution_s") or 0)
            except (TypeError, ValueError):
                pass
    return {"turns": turn, "qpu_s": qpu, "last": last}


def reference(backend: str, qubit: str, works):
    """gate_fidelity.averaged and its stamp in the freshest pull of this night that has the qubit."""
    best = (None, "")
    for w in works:
        f = w / "source-state/state.json"
        if not f.exists():
            continue
        try:
            q = json.load(open(f))["qubits"].get(qubit) or {}
        except Exception:  # noqa: BLE001
            continue
        gf = q.get("gate_fidelity") or {}
        if gf.get("averaged"):
            best = (gf["averaged"], gf.get("averaged_updated_at") or "")
    return best


def collect():
    prices = load_prices()
    works = sorted([w for w in list(RUNS.glob("n10-*-2026*")) + list(RUNS.glob("n11-*-2026*"))
                    if w.is_dir() and w.name.split("-")[1] in BACKENDS])
    rows = []
    for work in works:
        backend, night = work.name.split("-")[1], work.name.split("-")[0]
        same_backend = [w for w in works if w.name.split("-")[1] == backend]
        for cell in sorted(p for p in work.iterdir() if p.is_dir() and "-tinycal-" in p.name):
            q = cell.name.rsplit("-", 1)[1]
            doc = cell / "result.json"
            # the lab's calibration in force when the qubit ran: its own work dir's pull (others only as a fallback)
            ref, ref_date = reference(backend, q, [w for w in same_backend if w != work] + [work])
            row = {"night": night, "work": work.name, "cell": cell.name, "backend": backend, "target": q, "fw": "tinycal",
                   "ref": ref, "ref_date": ref_date}
            if not doc.exists():  # the driver writes the document when tinycal ends: still running
                run_id = next((p.name for p in TINYCAL_RUNS.glob(f"{night}_{backend}_openrouter-{q}_*")), "")
                it = interim(run_id, q)
                row.update({"status": "running", "run_id": run_id, "turns": it["turns"], "qpu_s": it["qpu_s"], "last": it["last"],
                            "gate_fid": None, "nodes": None, "cost": None, "model_s": None, "queue_s": None, "wall_s": None,
                            "decisions": None, "overrides": None, "node_runs": node_runs(run_id, q), "ctx": {}, "tokens": {},
                            "started": "", "ended": "—", "ballpark": None, "graded": None, "outside": [], "cause": "running",
                            "t1": None, "t2e": None, "x180_len": None, "readout": None, "reason": None, "alpha": None,
                            "rb_cover": None, "rb_depth": None, "rb_claimed": None})
                rows.append(row)
                continue
            r = json.load(open(doc))
            x = r["targets"][0]
            run_id = r.get("run_id") or ""
            qual = x.get("quality") or {}
            rb, cl, ro = qual.get("rb") or {}, qual.get("coherence_limit") or {}, qual.get("readout") or {}
            ag = x.get("agent") or {}
            at = ag.get("time") or {}
            jd = (x.get("judge") or {}).get("identity") or {}
            L, A = x180(cell / "quam_state/state.json", q)
            cs = context_series(cell, run_id, q, "tinycal")
            row.update({
                "status": x["status"], "run_id": run_id, "started": local(r.get("started_at")), "ended": local(r.get("ended_at")),
                "turns": g(x, "agent", "turns", "total"), "nodes": x.get("nodes_completed"), "graph": x.get("graph_node_count"),
                "node_execs": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                "wall_s": at.get("total_s") or g(r, "totals", "time", "total_s"),
                "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, MODEL_ID, prices),
                "rb": rb.get("error_per_clifford"),
                "rb_cover": rb_coverage(run_id, q)[0], "rb_depth": rb_coverage(run_id, q)[1],
                "gate_fid": gate_fidelity(rb.get("error_per_clifford")) if x["status"] == "completed" and (rb.get("error_per_clifford") or 0) > 0
                and (rb_coverage(run_id, q)[0] is None or rb_coverage(run_id, q)[0] >= RB_VALID_DECAY_LENGTHS) else None,
                # a non-positive error per Clifford is not a probability (fidelity above 100 %): never a claim worth showing
                "rb_claimed": gate_fidelity(rb.get("error_per_clifford")) if (rb.get("error_per_clifford") or 0) > 0 else None,
                "readout": ro.get("assignment_fidelity"), "t1": cl.get("t1_s"), "t2e": cl.get("t2echo_s"), "alpha": qual.get("drag_alpha"),
                "x180_len": L, "x180_amp": A,
                "ballpark": jd.get("in_ballpark"), "graded": jd.get("graded"), "outside": jd.get("outside") or [],
                "reason": x.get("escalation_reason"), "cause": base.classify_stop(x.get("escalation_reason"), x["status"]),
                "terminal": x.get("terminal_node"),
                "ctx": {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None},
                "overrides": night4.overrides(run_id, q), "decisions": night4.node_decisions(run_id, q),
                "node_runs": node_runs(run_id, q), "collateral": len((r.get("judge") or {}).get("collateral_edits") or []),
                "fingerprint": r.get("fingerprint") or {}, "scramble": {k: v for k, v in (r.get("scramble") or {}).items() if k != "entries"},
            })
            rows.append(row)
    return rows


def collect_night4():
    """The 20-21 Sep OpenRouter qwen3.8-27b runs of the same qubits (class defaults then): the settled run per qubit (the last completed
    one, else the last attempt), with its per-node QPU."""
    by = {}
    for work in sorted(RUNS.glob("night4-*-2026*")):
        if not work.is_dir() or work.name in night4.EXCLUDED_DIRS or "." in work.name.split("-")[-1]:
            continue
        backend = work.name.split("-")[1]
        for doc in sorted(glob.glob(str(work / "*-tinycal-qwen3-8-27b-openrouter*/result.json"))):
            r = json.load(open(doc))
            for x in r["targets"]:
                if x["status"] in ("pending", "queued"):
                    continue
                rb = g(x, "quality", "rb", "error_per_clifford")
                at = g(x, "agent", "time") or {}
                label = x["target"] + night4.rerun_label(Path(doc).parent.name, x["target"])  # '· rerun' labels as that report keys them
                invalid = (backend, "openrouter", label) in night4.INVALID_RB  # that report's own verdict: not a measurement
                entry = {"status": x["status"], "gate_fid": gate_fidelity(rb) if x["status"] == "completed" and rb and not invalid else None,
                         "rb_invalid": invalid and x["status"] == "completed",
                         "qpu_s": at.get("qpu_execution_s"), "model_s": at.get("model_s"), "queue_s": at.get("queue_wait_s"),
                         "wall_s": at.get("total_s"), "run_id": r.get("run_id") or "", "target": x["target"],
                         "node_runs": node_runs(r.get("run_id") or "", x["target"])}
                prev = by.get((backend, x["target"]))
                if prev is None or entry["status"] == "completed" or prev["status"] != "completed":
                    by[(backend, x["target"])] = entry
    return by


def active_reset_test():
    """The qolab Q2 pairs run before launch (run_active_reset.py): same node, same parameters, thermal vs active reset."""
    out = {}
    for f in sorted(ACTIVE_RESET_OUT.glob("*.json")):
        try:
            out[f.stem] = json.load(open(f))
        except Exception:  # noqa: BLE001
            pass
    return out


# ----------------------------------------------------------------------------- tables
def _fid_text(r):
    if r["status"] == "running":
        return f"<span class='muted'>running (turn {r['turns']}, {esc(SHORT.get(r['last'], r['last']))})</span>"
    if r["status"] == "completed":
        cov = r.get("rb_cover")
        if r["gate_fid"] is None:
            if r.get("rb_claimed") is not None and cov is not None:
                return (f"completed, <b>RB not a measurement</b> <span class='small muted'>(claimed {100 * r['rb_claimed']:.2f}; the fit saw "
                        f"{cov:.2f} decay lengths, max depth {r.get('rb_depth')})</span>")
            note = FINAL_NOTE.get((r["backend"], r["target"]))
            return "completed, <b>no valid RB</b>" + (f" <span class='small muted'>({esc(note)})</span>" if note else "")
        note = FINAL_NOTE.get((r["backend"], r["target"]))
        if cov is not None and cov < RB_EXTRAPOLATED_DECAY_LENGTHS:
            note = (note + "; " if note else "") + f"extrapolated: {cov:.1f} decay lengths, max depth {r.get('rb_depth')}"
        return f"<b>{100 * r['gate_fid']:.2f}</b>" + (f" <span class='small muted'>({esc(note)})</span>" if note else "")
    short = ATTEMPT_SHORT.get((r["backend"], r["target"]))
    plain = {"escalated": "stuck", "failed": "aborted"}.get(r["status"], r["status"])
    return f"<span class='small'>{esc(plain)}{' (' + esc(short) + ')' if short else ''}</span>"


def final_table(rows, n4):
    order = {b: i for i, b in enumerate(BACKENDS)}
    body = []
    for r in sorted(rows, key=lambda r: (order[r["backend"]], r["target"])):
        old = n4.get((r["backend"], r["target"]))
        if old is None:
            old_txt = "<span class='muted'>—</span>"
        elif old.get("rb_invalid"):
            old_txt = f"<span class='small'>completed, RB invalid</span> <span class='small muted'>· {fmt(old['qpu_s'] / 60 if old['qpu_s'] else None, '{:.1f}')} min QPU</span>"
        elif old["status"] == "completed" and old["gate_fid"] is not None:
            old_txt = f"{100 * old['gate_fid']:.2f} <span class='small muted'>· {fmt(old['qpu_s'] / 60 if old['qpu_s'] else None, '{:.1f}')} min QPU · {minutes(old['wall_s'])}</span>"
        else:
            old_txt = f"<span class='small'>{esc({'escalated': 'stuck', 'failed': 'aborted'}.get(old['status'], old['status']))}</span>"
        ref = f"{100 * r['ref']:.2f} <span class='small muted'>{esc(r['ref_date'][5:16])}</span>" if r["ref"] else "—"
        ballpark = f"{r['ballpark']}/{r['graded']}" if r.get("graded") else "—"
        outside = "; ".join(o.split(" came back")[0].split(".", 1)[-1] for o in r["outside"])
        body.append(
            f"<tr><td class='mono'>{esc(r['backend'])} {esc(r['target'])}</td><td>{_fid_text(r)}</td><td class='num'>{ref}</td>"
            f"<td class='num'>{r['nodes'] if r.get('nodes') is not None else '—'}/{r.get('graph') or 19}</td>"
            f"<td class='num'>{fmt(r['qpu_s'] / 60 if r['qpu_s'] is not None else None, '{:.1f} min')}</td>"
            f"<td class='num'>{minutes(r['model_s'])}</td><td class='num'>{minutes(r['queue_s'])}</td><td class='num'>{minutes(r['wall_s'])}</td>"
            f"<td class='num'>{fmt(r['cost'], '${:.2f}')}</td><td class='num'>{ballpark}"
            f"{'<br><span class=\"small muted\">' + esc(outside) + '</span>' if outside else ''}</td>"
            f"<td class='num'>{old_txt}</td></tr>")
    done = [r for r in rows if r["status"] == "completed"]
    settled = [r for r in rows if r["status"] != "running"]
    head = ("<thead><tr><th>qubit</th><th>gate fidelity, %</th><th>reference, % (stamp)</th><th>nodes</th><th>QPU</th><th>agent</th>"
            "<th>queue</th><th>wall</th><th>cost</th><th>scrambled params back in ballpark</th>"
            "<th>20–21 Sep, OpenRouter, class defaults<br><span class='small muted'>fid · QPU · wall</span></th></tr></thead>")
    calibrated = [r for r in done if r["gate_fid"] is not None]
    foot = f"<tr><td><b>calibrated</b></td><td colspan='10'><b>{len(calibrated)}/{len(settled)}</b> with an RB that is a measurement · {len(done)}/{len(settled)} reached the end of the graph" + (
        f" · {len(rows) - len(settled)} running" if len(rows) > len(settled) else "") + "</td></tr>"
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}{foot}</tbody></table></div>'


def by_backend_table(rows):
    cols = list(BACKENDS) + ["all"]
    stats = []
    for b in cols:
        runs = [r for r in rows if (b == "all" or r["backend"] == b) and r["status"] != "running"]
        done = [r for r in runs if r["status"] == "completed"]
        valid = [r for r in done if r["gate_fid"] is not None]
        if not runs:
            stats.append(None)
            continue
        fids = sorted(r["gate_fid"] for r in done if r["gate_fid"] is not None)
        med = lambda k, rs=done: median([r[k] for r in rs if r.get(k) is not None])  # noqa: E731
        n = len(valid) or 1
        tot = lambda k: sum((r.get(k) or 0) for r in runs)  # noqa: E731
        beat = sum(1 for r in done if r["gate_fid"] is not None and r["ref"] and r["gate_fid"] >= r["ref"])
        with_ref = sum(1 for r in done if r["gate_fid"] is not None and r["ref"])
        stats.append({
            "done": f"{len(valid)}/{len(runs)} ({100 * len(valid) / len(runs):.0f}%)",
            "ended": f"{len(done)}/{len(runs)}",
            "fid": (pct(fids[len(fids) // 2]) + f"<br><span class='small muted'>{pct(fids[0])} – {pct(fids[-1])}</span>") if fids else "—",
            "above": f"{sum(1 for f in fids if f >= 0.999)}/{len(fids)}" if fids else "—",
            "beat": f"{beat}/{with_ref}" if with_ref else "—",
            "wall": minutes(med("wall_s")), "qpu": f"{med('qpu_s') / 60:.1f} min" if med("qpu_s") is not None else "—",
            "agent": minutes(med("model_s")), "queue": minutes(med("queue_s")),
            "qpu_share": f"{100 * sum(r['qpu_s'] or 0 for r in done) / max(sum(r['wall_s'] or 0 for r in done), 1):.0f}%" if done else "—",
            "turns": f"{med('turns'):.0f}" if med("turns") is not None else "—",
            "nodes": f"{med('node_execs'):.0f}" if med("node_execs") is not None else "—",
            "cost": f"${tot('cost') / n:.2f}", "spent": f"${tot('cost'):.2f}",
            "ctx": ktok(median([r["ctx"].get("median") for r in runs if r["ctx"].get("median")])),
            "ctx_end": ktok(median([r["ctx"].get("end") for r in done if r["ctx"].get("end")])),
            "override": (lambda w, o: f"{100 * o / w:.0f}%" if w else "—")(
                sum(r["overrides"][0] for r in done if r["overrides"]), sum(r["overrides"][1] for r in done if r["overrides"])),
        })
    labels = [("qubits calibrated (graph finished, RB a measurement) / settled", "done"),
              ("reached the end of the graph / settled (the agent's own 'completed')", "ended"), ("single-qubit gate fidelity, median (min – max)", "fid"),
              ("completed runs at ≥ 99.9 %", "above"), ("completed runs at or above the qubit's reference fidelity", "beat"),
              ("wall time per bring-up that reached the end, median", "wall"), ("QPU time per bring-up that reached the end, median", "qpu"),
              ("agent (model) time per bring-up that reached the end, median", "agent"), ("cloud queue wait per bring-up that reached the end, median", "queue"),
              ("QPU share of the wall time, completed bring-ups", "qpu_share"),
              ("model turns per completed bring-up, median", "turns"), ("node runs per completed bring-up, median", "nodes"),
              ("state writes that are not a node's proposal, completed", "override"),
              ("judge cost per calibrated qubit (all attempts ÷ calibrated)", "cost"), ("judge cost, total", "spent"),
              ("context per model call, median, k tokens", "ctx"), ("context at the end of a bring-up, median, k tokens", "ctx_end")]
    body = "".join(f"<tr><th class='rowh'>{esc(lab)}</th>" + "".join(f"<td class='num'>{s[k] if s else '—'}</td>" for s in stats) + "</tr>"
                   for lab, k in labels)
    head = "<thead><tr><th></th>" + "".join(f"<th class='grp'>{esc(c)}</th>" for c in cols) + "</tr></thead>"
    return f'<div class="scroll"><table class="grid pivot">{head}<tbody>{body}</tbody></table></div>'


def node_qpu_table(rows, n4):
    """Per node: executions per completed bring-up, median QPU per execution tonight, at the 20-21 Sep defaults, and the class-default
    estimate; the node's share of tonight's QPU."""
    done = [r for r in rows if r["status"] == "completed"]
    now, old = collections.defaultdict(list), collections.defaultdict(list)
    for r in done:
        for node, qpu, *_ in r["node_runs"]:
            if qpu is not None and qpu > 0:
                now[node].append(qpu)
    for e in n4.values():
        if e["status"] != "completed":
            continue
        for node, qpu, *_ in e["node_runs"]:
            if qpu is not None and qpu > 0:
                old[node].append(qpu)
    total_now = sum(sum(v) for v in now.values()) or 1
    n_done = len(done) or 1
    n_old = sum(1 for e in n4.values() if e["status"] == "completed") or 1
    body, sum_med, sum_est = [], 0.0, 0.0
    for node in RECIPE_NODES:
        v, o = now.get(node, []), old.get(node, [])
        est, est_note = CLASS_DEFAULT_ESTIMATE.get(node, (None, ""))
        m = median(v)
        sum_med += m or 0
        sum_est += est or 0
        body.append(
            f"<tr><td>{esc(SHORT.get(node, node))}</td><td class='small'>{esc(PRESET_NOTE.get(node, 'class defaults'))}</td>"
            f"<td class='num'>{len(v) / n_done:.1f}</td><td class='num'><b>{fmt(m, '{:.1f}')}</b>"
            f"<span class='small muted'>{' (' + fmt(min(v), '{:.1f}') + '–' + fmt(max(v), '{:.1f}') + ')' if len(v) > 1 else ''}</span></td>"
            + (f"<td class='num'>{fmt(median(o), '{:.1f}')}<span class='small muted'> ({len(o) / n_old:.1f}×)</span></td>" if o else "<td class='num muted'>not in that graph</td>")
            + f"<td class='num'>{fmt(est, '{:.0f}')}{'<br><span class=\"small muted\">' + esc(est_note) + '</span>' if est_note else ''}</td>"
            f"<td class='num'>{100 * sum(v) / total_now:.0f}%</td></tr>")
    body.append(f"<tr><th class='rowh'>one execution of every node</th><td></td><td></td><td class='num'><b>{sum_med:.0f} s</b></td><td></td>"
                f"<td class='num'>{sum_est:.0f} s</td><td></td></tr>")
    head = ("<thead><tr><th>node</th><th>graph preset tonight</th><th>runs per completed bring-up</th>"
            "<th>QPU per run tonight, median (range), s</th><th>20–21 Sep per run, median, s (runs per bring-up)</th>"
            "<th>class-default estimate, s (T1 = 50 µs)</th><th>share of tonight's QPU</th></tr></thead>")
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}</tbody></table></div>'


def active_reset_table(test):
    if not test:
        return "<p class='muted'>no records</p>"

    def fr(tag, key):
        rec = test.get(tag) or {}
        v = ((rec.get("fit_results") or {}).get("Q2") or {}).get(key)
        return v.get("value") if isinstance(v, dict) else v
    rows = [
        ("IQ blobs, defaults (4000 shots × g/e)", "iq", "readout fidelity", lambda t: fmt(fr(t, "readout_fidelity"), "{:.1f} %")),
        ("DRAG, 50 shots, α step 0.1, pulse ceiling 100", "drag", "α", lambda t: fmt(fr(t, "alpha"), "{:+.3f}")),
        ("RB, defaults (300 sequences × 12 depths × 10 shots)", "rb", "error per Clifford",
         lambda t: fmt(fr(t, "error_per_clifford"), "{:.3f} %") + " ± " + fmt(fr(t, "error_per_clifford_sigma"), "{:.3f}")),
    ]
    body = []
    for label, stem, what, get in rows:
        th, ac = test.get(f"{stem}-thermal") or {}, test.get(f"{stem}-active") or {}
        body.append(f"<tr><td>{esc(label)}</td><td>{esc(what)}</td><td class='num'>{get(f'{stem}-thermal')}</td><td class='num'>{get(f'{stem}-active')}</td>"
                    f"<td class='num'>{fmt(th.get('qpu_s'), '{:.1f}')} → {fmt(ac.get('qpu_s'), '{:.1f}')} s</td></tr>")
    full = test.get("dragfull-active") or {}
    if full:
        body.append(f"<tr><td>DRAG at the class defaults (400 shots, α step 0.05, pulse ceiling 100)</td><td>—</td><td class='num'>—</td>"
                    f"<td class='num'>{'timed out at 60 s' if full.get('error') else 'ran'}</td><td class='num'>—</td></tr>")
    head = "<thead><tr><th>node, parameters</th><th>read-out</th><th>thermal reset</th><th>active reset</th><th>QPU, thermal → active</th></tr></thead>"
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}</tbody></table></div>'


def decision_table(rows):
    cols = [("runs", "runs"), ("failed", "failed"), ("retry", "retried after a failure"), ("repeat", "re-run right after a success"),
            ("back", "stepped back to"), ("ahead", "skipped ahead to"), ("left_failed", "left after a failure"),
            ("proposed", "runs with a proposal"), ("accepted", "accepted as proposed"), ("partial", "partly written"),
            ("changed", "value changed"), ("declined_rerun", "declined, re-ran the node"), ("declined_other", "declined, ran another node")]
    pct_of = {"failed": "runs", "accepted": "proposed", "partial": "proposed", "changed": "proposed",
              "declined_rerun": "proposed", "declined_other": "proposed"}
    per, qruns = {}, 0
    for r in rows:
        if not r.get("decisions"):
            continue
        qruns += 1
        for node, cnt in r["decisions"].items():
            per.setdefault(node, collections.Counter()).update(cnt)
    if not per:
        return "<p class='muted'>no event logs yet</p>"
    for cnt in per.values():
        cnt["proposed"] = sum(cnt[k] for k in ("accepted", "partial", "changed", "declined_rerun", "declined_other"))
    total = collections.Counter()
    for cnt in per.values():
        total.update(cnt)

    def cell(c, key):
        v = c[key]
        if not v:
            return "<td class='num muted'>·</td>"
        base_n = c[pct_of[key]] if key in pct_of else 0
        return f"<td class='num'>{v}" + (f" <span class='small muted'>{100 * v / base_n:.0f}%</span>" if base_n else "") + "</td>"
    body = [f"<tr><td>{esc(SHORT.get(n, n))}</td>{''.join(cell(per[n], k) for k, _ in cols)}</tr>" for n in RECIPE_NODES if n in per]
    body.append(f"<tr><th class='rowh'>all nodes</th>{''.join(cell(total, k) for k, _ in cols)}</tr>")
    head = ("<thead><tr><th></th><th colspan='7' class='grp'>sequence</th><th colspan='6' class='grp'>proposals</th></tr>"
            "<tr><th>node</th>" + "".join(f"<th>{esc(lab)}</th>" for _, lab in cols) + "</tr></thead>")
    return (f"<p class='small muted'>{qruns} qubit-runs, {total['runs']} node runs.</p>"
            f"<div class='scroll'><table class='grid small'>{head}<tbody>{''.join(body)}</tbody></table></div>")


def hard_cases(rows):
    out = []
    for r in rows:
        if r["status"] == "running":
            continue
        poor = r["status"] == "completed" and r["gate_fid"] is not None and r["gate_fid"] < 0.99
        invalid = r["status"] == "completed" and r["gate_fid"] is None
        if r["status"] == "completed" and not (poor or invalid):
            continue
        fid = f" · {pct(r['gate_fid'])}" if r["gate_fid"] is not None else ""
        outcome = (f"{status_chip(r['status'])} {r['nodes'] or 0}/{r.get('graph') or 19}{fid}<br><span class='small'>{minutes(r['model_s'])} agent · "
                   f"{fmt(r['cost'], '${:.2f}')} · {fmt(r['turns'], '{}')} turns</span>")
        what = ATTEMPT_SHORT.get((r["backend"], r["target"])) or (
            r["cause"] if r["status"] != "completed" else
            (f"finished the graph on an RB fit that is not a measurement ({r['rb_cover']:.2f} decay lengths at max depth {r['rb_depth']}; "
             f"claimed {pct(r['rb_claimed'])})" if invalid and r.get("rb_cover") is not None and r.get("rb_claimed") is not None
             else "finished the graph without a valid RB number" if invalid else "finished the graph under 99 %"))
        out.append((0 if r["status"] != "completed" else 1, r["backend"], r["target"], outcome, what,
                    HARD_CASE_NOTE.get((r["backend"], r["target"]), "")))
    out.sort(key=lambda t: t[:3])
    body = "".join(f"<tr><td class='mono'>{esc(b)} {esc(q)}</td><td>{o}</td><td class='small'>{esc(w)}</td><td class='small'>{esc(n)}</td></tr>"
                   for _, b, q, o, w, n in out) or "<tr><td colspan='4' class='muted'>none yet</td></tr>"
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>qubit</th><th>outcome</th><th>what happened</th>'
            '<th>what would have caught it</th></tr></thead><tbody>' + body + "</tbody></table></div>")


def _scatter(points, *, xlabel, ylabel, guides=(1.0,), guide_labels=("y = x",)):
    night4.PROVIDERS.setdefault("n10", ("qwen3.8-27b · OpenRouter", MODEL_ID, "tinycal"))
    return night4._scatter_svg([(a, b, lab, "n10") for a, b, lab in points], xlabel=xlabel, ylabel=ylabel, guides=guides,
                               guide_labels=guide_labels, W=360, H=300)


def error_scatters(rows):
    ref_pts, floor_pts = [], []
    for r in rows:
        if r["status"] != "completed" or r["gate_fid"] is None:
            continue
        err = 100 * (1 - r["gate_fid"])
        label = f"{r['backend']} {r['target']} · {pct(r['gate_fid'])}"
        if r["ref"] and r["ref"] < 1:
            ref_pts.append((100 * (1 - r["ref"]), err, label + f" · reference {pct(r['ref'])} ({r['ref_date'][:16]})"))
        if r["t1"] and r["t2e"] and r["x180_len"]:
            floor = 100 * (r["x180_len"] * 1e-9 / 3.0) * (1.0 / r["t1"] + 1.0 / r["t2e"])
            floor_pts.append((floor, err, label + f" · T1 {1e6 * r['t1']:.0f} µs, T2e {1e6 * r['t2e']:.0f} µs, x180 {r['x180_len']:.0f} ns · floor {floor:.3f}%"))
    a = _scatter(ref_pts, xlabel="reference calibration's gate error, %", ylabel="measured gate error, %") if ref_pts else ""
    b = _scatter(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %", ylabel="measured gate error, %",
                 guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x")) if floor_pts else ""
    return (f"<div class='scattergrid2'><div class='panel'><div class='ptitle small'><b>Against the reference calibration</b> · {len(ref_pts)} runs</div>{a}"
            f"<p class='small muted'>The reference is gate_fidelity.averaged in the night's own fresh pull (the stamp the cloud stores with it, Israel "
            f"time): the last calibration on file, not necessarily the same RB protocol. Below the diagonal the run beat it.</p></div>"
            f"<div class='panel'><div class='ptitle small'><b>Against the coherence floor</b> · {len(floor_pts)} runs</div>{b}"
            f"<p class='small muted'>Floor = (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) with the run's own T1, T2echo and x180 length. On the diagonal the "
            f"gate is decoherence-limited; far above it, the control is the limit.</p></div></div>")


def pins_table(rows):
    shas = collections.Counter(str(r.get("fingerprint", {}).get("git_sha", ""))[:7] for r in rows if r.get("fingerprint"))
    spec = collections.Counter(str(r.get("scramble", {}).get("spec_hash", "")) for r in rows if r.get("scramble"))
    snaps = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("scramble"):
            snaps[r["work"]][str(r["scramble"].get("source_hash", ""))] += 1
    git = lambda p, *a: base._git(p, *a)  # noqa: E731
    ql_log = git(QUA_LIBS, "log", "--oneline", "-4").splitlines()
    fmtc = lambda c: ", ".join(f"<span class='mono'>{esc(k)}</span> ({v})" for k, v in c.most_common()) or "—"  # noqa: E731
    body = [
        f"<tr><td>tinycal (working tree, run directly)</td><td class='mono'>{esc(git(Path.home() / 'code/QM/tinycal', 'rev-parse', '--short', 'HEAD'))}</td>"
        f"<td class='small'>document fingerprint.git_sha: {fmtc(shas)}; recipe bringup_recipes/flux_tunable_1q.md at that commit</td></tr>",
        f"<tr><td>qua-libs <span class='mono'>feat/bringup-fast-presets</span> (worktree ~/qab-runs/qua-libs-n10)</td>"
        f"<td class='mono'>{esc(git(QUA_LIBS, 'rev-parse', '--short', 'HEAD'))}</td><td class='small mono'>{'<br>'.join(esc(l) for l in ql_log)}</td></tr>",
        f"<tr><td>scramble spec (workloads/decalibrate_chip.yaml)</td><td class='mono'>{esc(next(iter(spec), ''))}</td><td class='small'>document scramble.spec_hash: {fmtc(spec)}</td></tr>",
        "<tr><td>acceptance spec (workloads/acceptance.yaml)</td><td class='mono'>821bb4dd1d5d0e5a</td><td class='small'>stamped by qab accept</td></tr>",
    ]
    for w, c in sorted(snaps.items()):
        body.append(f"<tr><td>{esc(w)} snapshot (source-state)</td><td class='mono'>{esc(next(iter(c), ''))}</td>"
                    f"<td class='small'>document scramble.source_hash; seed a new work dir from ~/qab-runs/{esc(w)}/source-state</td></tr>")
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>component</th><th>sha / digest</th><th>as recorded</th></tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>")


def run_ids(rows):
    body = []
    for r in sorted(rows, key=lambda r: (r["work"], r["target"])):
        body.append(f"<tr><td class='mono small'>{esc(r['work'])}</td><td class='mono small'>{esc(r['cell'])}</td><td class='mono'>{esc(r['target'])}</td>"
                    f"<td class='mono small'>~/code/QM/tinycal/runs/{esc(r['run_id'])}</td><td class='num'>{esc(r['started'])} → {esc(r['ended'])}</td>"
                    f"<td>{status_chip(r['status'])}</td></tr>")
    return ('<div class="scroll"><table class="grid small ids"><thead><tr><th>work dir (~/qab-runs/)</th><th>cell dir</th><th>qubit</th>'
            '<th>tinycal run dir</th><th>started → ended</th><th>status</th></tr></thead><tbody>' + "".join(body) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- page
def build():
    rows = collect()
    n4 = collect_night4()
    test = active_reset_test()
    running = [r for r in rows if r["status"] == "running"]
    settled = [r for r in rows if r["status"] != "running"]
    done = [r for r in settled if r["status"] == "completed"]
    valid = [r for r in done if r["gate_fid"] is not None]  # a finished graph whose RB is a measurement: a calibrated qubit
    fids = sorted(r["gate_fid"] for r in valid)
    live = bool(running) or not (RUNS / "n11-scheduler.log").exists() or "all 19 cells finished" not in (RUNS / "n11-scheduler.log").read_text()
    now = datetime.now().strftime("%d %b %Y %H:%M")
    spent = sum(r["cost"] or 0 for r in rows)
    med = lambda k: median([r[k] for r in done if r.get(k) is not None])  # noqa: E731
    same = [(r, n4[(r["backend"], r["target"])]) for r in done if (r["backend"], r["target"]) in n4 and n4[(r["backend"], r["target"])]["status"] == "completed"]
    tiles = [
        f'<div class="tile"><div class="v">{len(valid)}/{len(settled)}</div><div class="k">qubits calibrated, with an RB that is a measurement{" so far" if live else ""}'
        f' · {len(done)} reached the end of the graph{" · " + str(len(running)) + " running" if running else ""}</div></div>',
        f'<div class="tile"><div class="v">{pct(fids[len(fids) // 2]) if fids else "—"}</div><div class="k">median gate fidelity '
        f'({pct(fids[0]) if fids else "—"} – {pct(fids[-1]) if fids else "—"}); {sum(1 for f in fids if f >= 0.999)} at ≥ 99.9 %</div></div>',
        f'<div class="tile"><div class="v">{(med("qpu_s") or 0) / 60:.1f} min</div><div class="k">median QPU per bring-up that reached the end of the graph'
        + (f' (20–21 Sep, same qubits: {median([o["qpu_s"] for _, o in same if o["qpu_s"]]) / 60:.1f} min)' if same else "") + '</div></div>',
        f'<div class="tile"><div class="v">{minutes(med("wall_s"))}</div><div class="k">median wall per bring-up that reached the end of the graph'
        + (f' (20–21 Sep: {minutes(median([o["wall_s"] for _, o in same if o["wall_s"]]))})' if same else "") + '</div></div>',
        f'<div class="tile"><div class="v">${spent:,.2f}</div><div class="k">judge-priced OpenRouter spend{" so far" if live else ""}</div></div>',
    ]
    lab = lambda it, plain=False: it["label"] if plain else esc(it["label"])  # noqa: E731
    order = {b: i for i, b in enumerate(BACKENDS)}
    err_fig = bars(sorted([{"fw": "tinycal", "err": 100 * (1 - r["gate_fid"]), "label": f"{r['backend']} {r['target']}"} for r in done if r["gate_fid"] is not None],
                          key=lambda it: (order[it["label"].split()[0]], it["label"])),
                   "err", lab, lambda v: f"{v:.3f}% ({100 - v:.2f}%)", "Single-qubit gate error per completed bring-up", "gate error in %, fidelity in brackets; error per Clifford ÷ 1.875")
    time_fig = bars(sorted([{"fw": "tinycal", "t": (r["wall_s"] or 0) / 60, "label": f"{r['backend']} {r['target']}" + ("" if r["status"] == "completed" else f"  ({r['status']})")}
                            for r in settled if r["wall_s"]], key=lambda it: (order[it["label"].split()[0]], it["label"])),
                    "t", lab, lambda v: f"{v:.0f} min", "Wall time per qubit-run", "minutes from the first to the last event; unfinished runs marked")
    timeline = '<table class="grid timeline"><tbody>' + "".join(f"<tr><td>{esc(t)}</td><td>{esc(w)}</td></tr>" for t, w in INCIDENTS) + "</tbody></table>"
    banner = (f'<div class="tile" style="border-color:var(--warn)"><div class="k"><b>Live page</b> — regenerated {esc(now)} while cells are still '
              f'running: {esc(", ".join(r["backend"] + " " + r["target"] for r in running)) or "n11 not finished"}. Running rows are interim.</div></div>') if live else ""
    narrative = "\n".join(NARRATIVE) or "<p class='muted'>Written when the night ends.</p>"

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fast-node full bring-up</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}
.scattergrid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:6px 0 14px}} @media (max-width:760px){{.scattergrid2{{grid-template-columns:1fr}}}}
.scattergrid2 .panel{{min-width:0}} .scattergrid2 .ptitle{{margin-bottom:4px}}
svg.scatter{{display:block;width:100%;height:auto}} svg.scatter .grid{{stroke:var(--surface-2,#eef2f2);stroke-width:1}}
svg.scatter .frame{{fill:none;stroke:var(--muted,#8a8f98);stroke-width:.8}} svg.scatter .guide{{stroke:var(--muted,#8a8f98);stroke-width:1;stroke-dasharray:4 3}}
svg.scatter .tick{{font-size:10px;fill:var(--muted,#8a8f98);font-family:inherit}} svg.scatter .lab{{font-size:11px;fill:var(--ink-2,#4b555b);font-family:inherit}}
svg.scatter .pt{{fill-opacity:.8;stroke:var(--surface,#fff);stroke-width:.8}}
table.pivot td.num{{white-space:nowrap}}</style></head><body><div class="page">
<div class="eyebrow">qua-agents benchmark · faster bring-up graph, overnight run · generated {esc(now)}</div>
<h1 style="margin-top:8px">A full single-qubit bring-up with the sweeps sized to their fits: tinycal with qwen3.8-27b on every usable qubit of qolab, arbel and gilboa, 28–29 Sep 2026</h1>
<p class="lede">One framework, one model, one host, the whole graph. tinycal with qwen3.8-27b on OpenRouter (effort high, 16 000-token cap) calibrated
each qubit from a scrambled state through all nineteen nodes of the flux-tunable bring-up graph, from resonator identification to randomized
benchmarking, on qua-libs where the graph now presets the six slow nodes to what their fits need and resets actively after IQ_blobs. Targets:
qolab Q1–Q6; arbel all 21 qubits; gilboa the ten C/D qubits (the B row's readout line has a broken TWPA). n10 ran six qubits per device from
23:17; n11 took the rest of arbel and gilboa as n10's slots freed. At most three targets per device were in flight. Numbers are read from each
qubit-run's own result.json, tinycal's event log and the judge's price table; the operator annotations are marked.</p>
{banner}

<h2 id="overview">Overview</h2>
<div class="tiles">{"".join(tiles)}</div>
<p><b>What changed in the nodes.</b> Run once at their class defaults, the nineteen nodes need ~15 min of QPU for one qubit at T1 = 50 µs, and
three of them (T1 186 s, DRAG 488 s, Ramsey 51 s by its pre-flight) cannot run at all under IQCC's 60 s job limit. The graph now presets T1,
Ramsey, T2 echo, x180 error amplification, readout power and DRAG to sweeps sized to their fits, adds a ceiling to DRAG's T1-derived pulse
count (a new node parameter, None by default), and runs error amplification, Ramsey, DRAG and RB with active reset. Ramsey vs flux's pre-flight
no longer prices a thermal reset the program never does. The table in <a href="#qpu">QPU time per node</a> has every preset.</p>
<p><b>Apparatus.</b> tinycal main (max_turns 150, max_qpu_min 30, 60 s node timeout, wall cap 8 h); qua-libs <span class="mono">feat/bringup-fast-presets</span>
(9535a06 + 62e9bdf, 7bbb286, 08ebca5) frozen in its own worktree; workload decalibrate_chip.yaml, acceptance spec 821bb4dd1d5d0e5a.
Snapshots: qolab — {esc(SNAPSHOT_NOTE['qolab'])}; arbel — {esc(SNAPSHOT_NOTE['arbel'])}; gilboa — {esc(SNAPSHOT_NOTE['gilboa'])}. Each qubit ran as
its own cell on its own copy of the scrambled state (29 fields per qubit: resonator and qubit frequencies, readout amplitude and thresholds,
x180/x90 amplitudes, DRAG α, flux offsets).</p>

<h3>Final per-target results</h3>
<p class="small muted">One row per qubit. QPU = execution on the chip; agent = time inside model calls; queue = the cloud queue; wall = first to
last event. The last column is the same model on the same host on 20–21 Sep, with the class defaults of that day, where that qubit ran.
"Scrambled params back in ballpark" is the judge's identity check on the four graded parameters (named when one came back outside).</p>
{final_table(rows, n4)}
{error_scatters(rows)}

<h3>By backend</h3>
{by_backend_table(rows)}
{time_fig}
{err_fig}

<h2 id="qpu">QPU time per node</h2>
<p class="small muted">Every node run of the completed bring-ups, measured by the IQCC worker (<span class="mono">__qpu_execution_time_seconds</span>).
The 20–21 Sep column is the same model on the class defaults of that day (a saturation line search instead of the chirp, no T1_chirp; T1, Ramsey
and DRAG then ran at smaller defaults than today's classes, before the 60 s limit). The estimate is the pre-run calculation for one execution at
today's class defaults.</p>
{node_qpu_table(rows, n4)}

<h3>Active reset, tested before launch</h3>
<p class="small muted">qolab Q2 on the n10 pull, propose mode (nothing written), each node run twice with identical parameters. Active reset is
quam_builder's measure-and-conditional-π loop (up to 15 attempts) on the thresholds IQ_blobs writes; it cost 65–100 µs per shot here against
365 µs of thermal wait (5 × T1, T1 = 73 µs). The class-default DRAG timed out with it, which is why the graph runs DRAG at 200 shots and α step 0.1,
and why the pre-flight now prices active reset at 60 µs per shot instead of three readouts.</p>
{active_reset_table(test)}

<h3>What the model did with each node</h3>
<p class="small muted">Every settled qubit-run, from tinycal's events.jsonl. <b>Sequence</b>, against the recipe order (resonator identification →
… → RB, the power sweep twice, T1 and T2 echo in either order). <b>Proposals</b>, per run that proposed state updates, from the writes before the
next node run. Percentages are of the node's runs (failed) or of its runs with a proposal.</p>
{decision_table(settled)}

<h3>Hard cases: qubits that were not calibrated, or not calibrated right</h3>
<p class="small muted">Settled runs that did not reach the end of the graph, then completed ones under 99 %. The right-hand column is the operator's
reading of the transcript, not something the cell recorded.</p>
{hard_cases(rows)}

<h3>What the numbers say{" (so far)" if live else ""}</h3>
{narrative}

<h2 id="stuck">Incident timeline</h2>
{timeline}

<h2 id="pins">Pins: what a new cell must use to be comparable</h2>
{pins_table(rows)}

<h2 id="runs">Run identifiers</h2>
{run_ids(rows)}
</div></body></html>"""
    OUT.write_text(page)
    art = Path(ARTIFACT_DIR) / OUT.name if ARTIFACT_DIR else None
    if art:
        body = page.split("<title>", 1)[1]
        art.write_text("<title>" + body.replace("</head><body>", "").replace("</body></html>", ""))
    print(f"wrote {OUT} · {len(rows)} qubit-runs, {len(done)} completed, {len(running)} running, ${spent:,.2f}")


if __name__ == "__main__":
    build()
