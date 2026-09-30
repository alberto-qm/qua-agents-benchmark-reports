#!/usr/bin/env python3
"""Build the 30 Sep 2026 report: the first bring-up of lucy, IQCC's fixed-frequency coaxmon ring, with tinycal and qwen3.8-27b via
OpenRouter on every qubit in its cloud state (qA-qG; qC twice), from the cell documents under ~/qab-runs/lucy1-* and ~/qab-runs/lucy2-*.

    uv run --project ~/code/QM/qua-agents-benchmark python make_lucy_report.py

Numbers come only from each cell's result.json (projected from tinycal's events.jsonl by scripts/project_tinycal.py, then stamped by
qab inspect-state / validate / accept), the cell's final quam_state, the tinycal run dirs (events.jsonl: per-node QPU time), the pulled
lab state (source-state) and, for the qC investigation, tinycal's measurement store (~/.qualibrate/user_storage/tinycal): every RB run on
qC that day with its parameters, fit and figures. The labels on those runs (which state copy, which change) and the operator notes
(INCIDENTS, NARRATIVE, the per-qubit notes) are marked as such. The page's figures and the qC run table are also written to
2026-09-30-lucy/. Table builders are shared with make_n10_report.py, make_fwcmp2_report.py and make_night4_report.py.
"""
from __future__ import annotations

import base64
import collections
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fwcmp2_report as base  # noqa: E402
import make_n10_report as n10  # noqa: E402
import make_night4_report as night4  # noqa: E402
from make_fwcmp2_report import (  # noqa: E402
    CSS, TINYCAL_RUNS, RUNS, bars, context_series, esc, fmt, g, gate_fidelity, load_prices, local, median, minutes, pct,
    status_chip, token_cost, x180,
)

OUT = Path(__file__).with_name("2026-09-30-lucy-fixed-frequency-bringup-qwen-tinycal.html")
DATA = Path(__file__).with_name("2026-09-30-lucy")
MODEL_ID = "qwen/qwen3.8-27b"
BACKEND = "lucy"
NIGHTS = ("lucy1", "lucy2")
STORE = Path.home() / ".qualibrate/user_storage/tinycal"
ARTIFACT_DIR = __import__("os").environ.get("REPORT_ARTIFACT_DIR")  # also write the page as an Artifact body here
GRID_FIGURE = DATA / "figures/iqcc_grid_2026-07-24.png"  # IQCC's own qubit-grid figure of 24 Jul (the device page), kept in the data dir

# The fixed-frequency recipe's order (tinycal bringup_recipes/fixed_frequency_1q.md): 02a replaces resonator identification, the four
# flux nodes of the flux-tunable graph are never run, T1 on I/Q (step 6) only when T1_chirp stored nothing.
RECIPE = [("02a_resonator_spectroscopy",), ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy",), ("T1_chirp",),
          ("qubit_spectroscopy_fine",), ("power_rabi",), ("readout_power_optimization",), ("readout_frequency_optimization",),
          ("IQ_blobs",), ("power_rabi_error_amplification_x180",), ("ramsey",), ("T1", "T2echo"), ("DRAG_calibration",),
          ("Randomized_benchmarking",)]
RECIPE_NODES = list(dict.fromkeys(n for step in RECIPE for n in step))
FLUX_NODES = ("resonator_identification", "resonator_spectroscopy_vs_flux", "qubit_spectroscopy_vs_flux", "ramsey_vs_flux_calibration",
              "02e_resonator_identification", "02c_resonator_spectroscopy_vs_flux", "03b_qubit_spectroscopy_vs_flux_chirp",
              "09a_ramsey_vs_flux_calibration")
# A node called by its library id runs the library's class defaults, not the graph's presets; the tables count it under the
# catalog name it stands for. lucy2's qC and qE called nearly every node this way (the recipe tells the model to call 02a by
# its library id, and it generalised).
LIB_TO_GRAPH = {
    "02b_resonator_spectroscopy_vs_power": "resonator_spectroscopy_vs_power", "03a_qubit_spectroscopy_chirp": "qubit_spectroscopy",
    "05b_T1_chirp": "T1_chirp", "03a_qubit_spectroscopy": "qubit_spectroscopy_fine", "04b_power_rabi": "power_rabi",
    "08b_readout_power_optimization": "readout_power_optimization", "08a_readout_frequency_optimization": "readout_frequency_optimization",
    "07_iq_blobs": "IQ_blobs", "04c_power_rabi_error_amplification_x180": "power_rabi_error_amplification_x180", "06a_ramsey": "ramsey",
    "05_T1": "T1", "06b_echo": "T2echo", "10b_drag_calibration_180_minus_180": "DRAG_calibration",
    "11a_single_qubit_randomized_benchmarking": "Randomized_benchmarking",
}
RB_NAMES = ("Randomized_benchmarking", "11a_single_qubit_randomized_benchmarking")
GRAPH_NODES = 14  # the recipe's nodes (13 steps; T1 and T2 echo are two nodes, 02a is a library node outside the catalog)
night4.RECIPE[:] = RECIPE  # node_decisions() and the n10 table builders read these module globals
night4.RECIPE_NODES[:] = RECIPE_NODES
n10.RECIPE_NODES[:] = RECIPE_NODES
n10.SHORT.update({"02a_resonator_spectroscopy": "resonator spectroscopy (02a)", "02b_resonator_spectroscopy_vs_power": "resonator vs power (02b id)"})
SHORT = n10.SHORT

# ----------------------------------------------------------------------------- operator notes (filled in by hand)
FINAL_NOTE: dict[tuple[str, str], str] = {  # (night, qubit) -> note next to a completed run's number
    ("lucy1", "qC"): "RB taken during qC's bad stretch (10:20–10:35); the same final state gives 99.94 % from 10:47 on",
    ("lucy2", "qE"): "RB run by its library id: read from the event log, the result document has none",
    ("lucy2", "qD"): "found the line only after putting back the readout power 02b had tripled",
    ("lucy2", "qF"): "calibrated on the two-photon line, 85 MHz below f₀₁, with a 4.5 µs x180",
    ("lucy2", "qC"): "on a line 38 MHz below qC's own; its last RB failed",
}
ATTEMPT_SHORT: dict[tuple[str, str], str] = {  # (night, qubit) -> short cause of a run that did not finish
    ("lucy2", "qG"): "readout moved 6.8 MHz off its resonator by the dip-only power sweep; qubit never found",
}
HARD_CASE_NOTE: dict[tuple[str, str], str] = {  # (night, qubit) -> what would have caught it (operator's reading of the transcript)
    ("lucy1", "qC"): "The calibration was right; the RB was not. It ran at 10:28 inside a stretch when every qC measurement with the gates "
                     "looked wrong, and the node called a two-component decay (a drop to ~0.59 within 20 Cliffords, then a slow tail the "
                     "single-exponential fit misses) a valid fit. The model then explained 4.7 % per Clifford with the recipe's long-readout "
                     "note — readout error scales RB's amplitude, not its decay (the note now says so). A check that the RB error lies "
                     "within a few times the T1/T2 floor, and one repeat when it does not, would have caught it.",
    ("lucy2", "qC"): "Ten chirp searches between 11:01 and 11:23, from the same scrambled state and readout that found qC's line at the first "
                     "try in lucy1 (10:18), saw nothing — the default window holds the line, 50 MHz above the seed. The model then searched "
                     "with saturation scans and committed a line at 4.2622 GHz, 38 MHz below qC's 4.3003, whose fitted x180 amplitude came out "
                     "at 3–4 (no pulse can play that); power Rabi never showed contrast, the pulse went to 60 ns at 0.47, and three RB runs "
                     "gave 26 %, 7.2 % and a failed fit. At 13:07 the same chirp on lucy1's final state found qC at 4.30034 GHz, 45 kHz from "
                     "10:25's Ramsey. qC was unreadable twice today (≈10:20–10:35 and ≈11:00–11:25): hold it back until that is understood. "
                     "It also ran 44 of its 45 nodes by library id.",
    ("lucy2", "qF"): "The power sweep tracked the notch beside qF's resonance (02b fits only dips; lucy's resonators are peaks) and proposed "
                     "−26.3 dBm, 0.061 V against the lab's 0.559. Committed at turn 6, it left the readout blind (IQ blobs 52–55 %). The "
                     "chirp saw no 0→1 line (it sits 50 MHz above the seed, inside the default window), but at 500 shots a lone line 34 MHz "
                     "below the seed, which the node called ambiguous — a possible two-photon line — and gave the partner window for. The "
                     "model committed it (4.3214 GHz, 85 MHz under qF's 4.4066: half the 174 MHz anharmonicity), found no Rabi contrast, and "
                     "stretched the x180 from 48 ns through 500, 1000, 20 000 and 50 000 ns to 4496 ns; RB 19 % per Clifford. A readout-power "
                     "guard (no commit 10+ dB below the stored power without an IQ check) or an ambiguous line not being writable would "
                     "each have stopped it; tinycal's read-only pulse lengths (931acd3, after this run) now stop the stretching.",
    ("lucy2", "qG"): "02a found qG's resonator at 9.8364 GHz (the lab's 9.8366) and the model committed it; the dip-only power sweep then "
                     "reported an 'unresolved' line at 9.8432 GHz, 6.8 MHz above, and the model moved the readout there. Several "
                     "linewidths off resonance the readout could not see the qubit: five chirp searches found nothing and it escalated at "
                     "turn 18 — correctly, for what it could see.",
}
# qC's RB runs of 30 Sep in the measurement store: which state copy and which change each one measured (operator's record, from
# ~/qab-runs/lucy1-LOG.md). Runs not listed are labelled from their tinycal run (the agents' own RB).
QC_RB_LABEL = {
    "m-94CD5Z": ("lucy1 agent", "the agent's own RB at the end of its bring-up (preset: active reset)"),
    "m-Q7GNGH": ("final state", "same node, same preset, on the state the agent left"),
    "m-C6FCAB": ("lab state", "the unscrambled cloud state; its readout threshold no longer separates qC's states (amplitude 0.03)"),
    "m-BSA5M3": ("final, lab α", "DRAG α set to the lab's −0.164"),
    "m-DWHVNS": ("final, α = 0", "DRAG α set to 0"),
    "m-J4SEPP": ("final, thermal", "reset_type thermal"),
    "m-02E4SY": ("final, active", "reset_type active (the preset)"),
    "m-NJ2EM0": ("final, thermal", "reset_type thermal, repeat"),
    "m-DT4MM8": ("depletion 3 µs", "active reset, depletion_time 3 µs (as stored)"),
    "m-NY10FB": ("depletion 12 µs", "active reset, depletion_time 12 µs"),
    "m-D4TC4Q": ("depletion 3 µs", "active reset, depletion_time 3 µs, repeat"),
    "m-RS15EE": ("depletion 12 µs", "active reset, depletion_time 12 µs, repeat"),
    "m-48KFHY": ("depletion 6 µs", "active reset, depletion_time 6 µs"),
    "m-9NXG6S": ("depletion 24 µs", "active reset, depletion_time 24 µs"),
}
QC_FIGURES = [  # (measurement id, figure in the store, caption) -- the page embeds these
    ("m-94CD5Z", "figures.amplitude_fit.png", "10:28, the agent's RB: 4.7 % per Clifford. The points drop to ~0.59 within ~20 Cliffords, then "
                                              "fall slowly to 0.49 by 2048; the single-exponential fit (red) puts its floor at 0.536 and misses the tail."),
    ("m-J4SEPP", "figures.amplitude_fit.png", "10:46, the same final state with thermal reset: 0.12 % per Clifford, one clean decay."),
]
PEAK_FIGURES = [  # (tinycal run, qubit, measurement id): lucy's resonators are peaks, and what the dip-only power sweep made of one
    ("lucy1_lucy_openrouter-qC_20260930-1013", "qC", "m-YS6W4Y"),
    ("lucy2_lucy_openrouter-qF_20260930-1057", "qF", "m-PQKPFK"),
]
PEAK_CAPTIONS = {
    "m-YS6W4Y": "qC, 02a (lucy1, 10:17): |IQ| rises to a peak at the resonance, with a notch just above it; 02a fits a peak and finds it.",
    "m-PQKPFK": "qF, 02b (lucy2, 11:25): the resonance is the bright band; the fit, which looks for a dip, tracked the dark notch beside it "
                "(red dots), called a drift onset at −23 dBm and proposed −26.3 dBm (the star), 19 dB below the lab's readout. It was committed.",
}
QC_RAMSEY = ("lucy1_lucy_openrouter-qC_20260930-1013", "m-SD602Q", "10:25, the agent's Ramsey on qC (active reset): the + branch leaves its fit "
             "over the first ~700 ns, and the fit's T2* (9.7 µs) sits far below T2 echo (32 µs).")
# (time, what happened): the operator's logs, ~/qab-runs/lucy1-LOG.md and lucy2-LOG.md
INCIDENTS: list[tuple[str, str]] = [
    ("29 Sep 11:00", "lucy's IQCC token registered; the device added to matrix.yaml (pinned qA qB qC, in no suite). The cloud state is a "
                     "FixedFrequencyQuam with seven qubits (qA–qG): qH, in IQCC's grid figure, was dropped in the 15 Aug upload."),
    ("29 Sep", "Offline: every node of the fixed-frequency recipe builds and serialises its QUA program for a lucy qubit (simulate path, "
               "connection stubbed); the benchmark's scramble spec refuses lucy (readout ×1.8 over 1.0 on qA/qB/qE/qF; five flux fields "
               "with no match), so a lucy variant drops those six fields."),
    ("30 Sep 10:13", "tinycal and qua-libs frozen as ~/qab-runs/tinycal-lucy1 and qua-libs-lucy1 (the main working trees: the n13 fixes and "
                     "the 03a search capped at the drive LO's reach); tests pass. Fresh pull, unchanged since 29 Sep; 161 fields scrambled."),
    ("10:14", "Probe: 02a on qA's lab state found the resonator 100 kHz from the stored value (1.7 s of QPU). Another session moved the "
              "shared tinycal venv to quam-builder 0.6.0 at 10:14:35; lucy1 and lucy2 both ran on it."),
    ("10:15", "lucy1 launched: qA, qB, qC, one cell each, each holding an IQCC slot (three in flight)."),
    ("10:24–10:29", "All three finished (8–13 min wall). qA 99.85 %, qB 99.87 %, qC 97.48 %."),
    ("10:31–10:53", "qC investigated in propose mode (no writes): 14 RB runs on copies of its final state (the table in the qC section). "
                    "The bad values all fall before 10:35; from 10:47 both reset types give 0.11–0.25 % per Clifford."),
    ("10:57", "lucy2 prepared: the same pull (byte-identical) and edit plan; tinycal-lucy2 adds only the corrected RB sentence in the recipe."),
    ("10:58", "lucy2 launched: qC (rerun), qD, qE, qF, qG, each cell waiting for one of the three IQCC slots."),
    ("11:00–11:25", "lucy2's first chirp searches: qE found its line at 11:00; qC (11:01), qD (11:02), qG (11:16) and qF (11:25) found "
                    "none, then or in 45 later searches. qC's readout was as in lucy1; qD's had been tripled and qF's cut 19 dB by the "
                    "power sweep; qG's was moved 6.8 MHz off its resonator by it."),
    ("11:13", "qE completed in 14 min: 99.91 %, 3/3 in ballpark. Its RB ran under the library id, so result.json has none."),
    ("11:24", "qG escalated at turn 18, its readout off resonance."),
    ("11:56", "qD completed: 99.88 % (IQCC's 99.85 %), 3/3 in ballpark, after putting its readout power back."),
    ("12:09", "qC (rerun) 'completed' on a line 38 MHz below its own; its last RB failed."),
    ("during lucy2", "Question from the user: why lucy's resonators are peaks. Each is measured in transmission on a line of its own (the "
              "coaxmon wiring), a Lorentzian peak rather than a side-coupled notch; 02b fits only dips and failed on all its lucy runs."),
    ("12:33–12:34", "Merged after the discussion, neither in these runs: tinycal 931acd3 (readonly_paths; the benchmark profile fixes "
                    "pulse and readout lengths) and benchmark main 34af973 (inspect-state flags a changed pulse length)."),
    ("13:04", "qF 'completed' after 104 turns: the two-photon line, a 4496 ns x180, RB 19 %. lucy2 over."),
    ("13:07", "qC's chirp on lucy1's final state finds its 0→1 line at 4.30034 GHz, 45 kHz from 10:25: qC is readable again."),
]
NARRATIVE: list[str] = [  # "What the numbers say": the operator's reading of the tables above, when lucy2 ended
    "<p><b>Four of the seven qubits were brought up from the scramble to IQCC's own level, and fast.</b> qA 99.854 %, qB 99.871 %, qD 99.880 % "
    "and qE 99.912 %, each with all three graded parameters back and its RB error at its own coherence floor; against IQCC's RB on file "
    "(15 Aug upload) that is level on qA and qD, 0.04 points below on qB, and qE has no IQCC RB to compare (its averaged field reads 99.85 %). "
    "f₀₁ came back 0.5–1.8 MHz from the six-week-old lab values and the resonators within 0.15 MHz. A calibrated bring-up took a median "
    "1.6 min of QPU and 12 min of wall, against 2.7 min and 25 min on the flux-tunable chips on 28–29 Sep: no flux maps to take, and where "
    "the readout was right the chirp found the line at the first search.</p>",
    "<p><b>The biggest lesson is the resonator's line shape.</b> lucy measures each resonator in transmission on a line of its own, so the "
    "resonance is a peak; 02b, the power sweep, fits only dips. It failed or tracked the wrong feature on every lucy qubit, and in lucy2 the "
    "model believed it three times: qF's readout cut 19 dB (blind readout, then the two-photon line, a 4.5 µs x180 and 19 % per Clifford), "
    "qG's readout moved 6.8 MHz off resonance (qubit never found), qD's tripled (21 blind searches before the model undid it). The recipe "
    "sentence written for this chip — 'no onset in the swept range' is expected here — explained a symptom with the wrong cause. The fix is a "
    "stored line shape per resonator that 02b and the other dip-assuming nodes read; until then the recipe should say 02b cannot read lucy.</p>",
    "<p><b>qC is a qubit problem, twice.</b> lucy1 calibrated it correctly — its final state gives 99.94 % at 10:47 — but its own RB ran "
    "in a stretch (≈10:20–10:35) when every qC measurement that played gates came out wrong, and the node called a two-component decay a "
    "valid fit. The lucy2 rerun hit a second stretch (≈11:00–11:25): ten chirp searches saw no line from a state and readout that had "
    "worked, and the model calibrated a feature 38 MHz below. At 10:18, 10:47–10:53 and 13:07 qC was clean. Hold it back from the "
    "benchmark until a Ramsey series says what moves.</p>",
    "<p><b>The model learned the wrong thing from one recipe line.</b> The recipe tells it to call 02a by its library id; in lucy2 two agents "
    "called nearly every node that way (qC 44 of 45, qE 17 of 18). A library id runs the class defaults, not the graph's presets, and "
    "the projection reads only catalog names: qE's RB, T1, T2 echo and readout never reached its result document (its RB is shown here from "
    "the event log, marked). Either tinycal maps a library id onto its catalog entry, or the recipe names 02a without inviting the "
    "generalisation; project_tinycal should read both names either way.</p>",
    "<p><b>Pulse lengths moved where the model was lost.</b> qF's x180 went to 4496 ns and qC's (rerun) to 60 ns; qC, qD and qF doubled "
    "their readouts to 3 µs. The four calibrated runs kept 48 ns. tinycal now refuses writes to pulse and readout lengths in the benchmark "
    "profile (931acd3) and the judge flags a changed length (benchmark 34af973); both landed after these runs.</p>",
    "<p><b>Cost.</b> $5.58 of OpenRouter credit at judge prices and 24 min of QPU for the eight qubit-runs. The four calibrated ones took "
    "$2.14 and 8 min; the four that were not, $3.44 and 16 min — a lost agent is the expensive one.</p>",
]


# ----------------------------------------------------------------------------- collect
def _events(run_id: str, target: str):
    return n10._events(run_id, target)


def graph_runs(run_id: str, target: str):
    """n10.node_runs with library ids folded into the catalog names they stand for."""
    return [(LIB_TO_GRAPH.get(node, node), *rest) for node, *rest in n10.node_runs(run_id, target)]


def library_id_runs(run_id: str, target: str) -> int:
    return sum(1 for node, *_ in n10.node_runs(run_id, target) if node in LIB_TO_GRAPH)


def rb_from_events(run_id: str, target: str):
    """(error per Clifford as a fraction, decay lengths covered, node name) of the last successful RB in the event log, under
    either name. The projection (scripts/project_tinycal.py) reads only 'Randomized_benchmarking', so an RB the model ran by its
    library id never reaches result.json."""
    import ast
    last = None
    for e in _events(run_id, target):
        if e.get("tool") == "run_node" and e.get("node") in RB_NAMES and e.get("outcome") == "successful":
            last = e
    if last is None:
        return None, None, None
    num = ast.literal_eval(last["numerics"]) if isinstance(last.get("numerics"), str) else (last.get("numerics") or {})
    v = lambda k: (num.get(k) or {}).get("value") if isinstance(num.get(k), dict) else num.get(k)  # noqa: E731
    epc = v("error_per_clifford")
    return (epc / 100 if epc is not None else None), v("decay_lengths_covered"), last.get("node")


def length_changes(given: dict, final: dict):
    """[(operation, before, after)] for every xy/resonator operation length that differs between the state the cell was given and
    the one it left -- the same comparison qab inspect-state now stamps (benchmark main, 34af973)."""
    out = []
    for channel in ("xy", "resonator"):
        before, after = g(given, channel, "operations") or {}, g(final, channel, "operations") or {}
        for op in sorted(set(before) | set(after)):
            b, a = before.get(op), after.get(op)
            if isinstance(b, dict) and isinstance(a, dict) and b.get("length") != a.get("length"):
                out.append((f"{channel}.{op}", b.get("length"), a.get("length")))
    return out


def lab_reference(q: dict):
    """IQCC's own RB on file: 1 - EPG from extras.1QRB_p (the decay, same definition as the node's), and the separate
    gate_fidelity.averaged field (its own protocol and stamp)."""
    p = (q.get("extras") or {}).get("1QRB_p")
    rb = 1 - (1 - p) / 2 / base.GATES_PER_CLIFFORD if p else None
    gf = q.get("gate_fidelity") or {}
    return rb, gf.get("averaged"), gf.get("averaged_updated_at") or ""


def collect():
    prices = load_prices()
    works = sorted(w for night in NIGHTS for w in RUNS.glob(f"{night}-{BACKEND}-2026*") if w.is_dir())
    rows = []
    for work in works:
        night = work.name.split("-")[0]
        lab = json.load(open(work / "source-state/state.json"))["qubits"]
        for cell in sorted(p for p in work.iterdir() if p.is_dir() and "-tinycal-" in p.name):
            q = cell.name.rsplit("-", 1)[1]
            ref, ref_avg, ref_date = lab_reference(lab.get(q) or {})
            row = {"night": night, "work": work.name, "cell": cell.name, "backend": night, "target": q, "fw": "tinycal",
                   "ref": ref, "ref_avg": ref_avg, "ref_date": ref_date, "lab": lab.get(q) or {}}
            doc = cell / "result.json"
            if not doc.exists():  # still running
                run_id = next((p.name for p in TINYCAL_RUNS.glob(f"{night}_{BACKEND}_openrouter-{q}_*")), "")
                it = n10.interim(run_id, q)
                row.update({"status": "running", "run_id": run_id, "turns": it["turns"], "qpu_s": it["qpu_s"], "last": it["last"],
                            "gate_fid": None, "nodes": None, "cost": None, "model_s": None, "queue_s": None, "wall_s": None,
                            "decisions": None, "overrides": None, "node_runs": n10.node_runs(run_id, q), "ctx": {}, "tokens": {},
                            "started": "", "ended": "—", "ballpark": None, "graded": None, "outside": [], "cause": "running",
                            "t1": None, "t2e": None, "x180_len": None, "readout": None, "reason": None, "rb_cover": None,
                            "rb_depth": None, "rb_claimed": None, "final": {}})
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
            cover, depth = n10.rb_coverage(run_id, q)
            epc = rb.get("error_per_clifford")
            rb_source = "result.json"
            if epc is None:  # an RB run by its library id never reaches the projection: read it from the event log, labelled
                ev_epc, ev_cover, ev_node = rb_from_events(run_id, q)
                if ev_epc is not None and ev_node != "Randomized_benchmarking":
                    epc, cover, rb_source = ev_epc, ev_cover, f"event log ({ev_node})"
            final = (json.load(open(cell / "quam_state/state.json"))["qubits"].get(q) or {}) if (cell / "quam_state/state.json").exists() else {}
            given = json.load(open(work / "scrambled-state/state.json"))["qubits"].get(q) or {}
            row.update({
                "rb_source": rb_source, "lengths": length_changes(given, final), "lib_runs": library_id_runs(run_id, q),
                "status": x["status"], "run_id": run_id, "started": local(r.get("started_at")), "ended": local(r.get("ended_at")),
                "turns": g(x, "agent", "turns", "total"), "nodes": x.get("nodes_completed"), "graph": GRAPH_NODES,
                "node_execs": g(ag, "nodes", "executions"), "reruns": g(ag, "nodes", "re_executions"),
                "model_s": at.get("model_s"), "qpu_s": at.get("qpu_execution_s"), "queue_s": at.get("queue_wait_s"),
                "wall_s": at.get("total_s") or g(r, "totals", "time", "total_s"),
                "tokens": ag.get("tokens") or {}, "cost": token_cost(ag.get("tokens") or {}, MODEL_ID, prices),
                "rb": epc, "rb_cover": cover, "rb_depth": depth,
                "gate_fid": gate_fidelity(epc) if x["status"] == "completed" and (epc or 0) > 0
                and (cover is None or cover >= n10.RB_VALID_DECAY_LENGTHS) else None,
                "rb_claimed": gate_fidelity(epc) if (epc or 0) > 0 else None,
                "readout": ro.get("assignment_fidelity"), "t1": cl.get("t1_s"), "t2e": cl.get("t2echo_s"), "alpha": qual.get("drag_alpha"),
                "x180_len": L, "x180_amp": A,
                "ballpark": jd.get("in_ballpark"), "graded": jd.get("graded"), "outside": jd.get("outside") or [],
                "reason": x.get("escalation_reason"), "cause": base.classify_stop(x.get("escalation_reason"), x["status"]),
                "ctx": {"median": median(cs), "end": cs[-1] if cs else None, "max": max(cs) if cs else None},
                "overrides": night4.overrides(run_id, q), "decisions": night4.node_decisions(run_id, q),
                "node_runs": graph_runs(run_id, q), "fingerprint": r.get("fingerprint") or {},
                "scramble": {k: v for k, v in (r.get("scramble") or {}).items() if k != "entries"}, "final": final,
                "flux_nodes_run": sorted({n for n, *_ in n10.node_runs(run_id, q) if n in FLUX_NODES}),
            })
            rows.append(row)
    return rows


FLOOR_FACTOR = 3.0  # an RB error per gate within this many times the coherence floor counts as a calibration (the scatter's y = 3x)


def coherence_floor(r):
    """Per-gate coherence floor (t_gate/3)(1/T1 + 1/T2echo) from the run's own numbers, or None."""
    if r.get("t1") and r.get("t2e") and r.get("x180_len"):
        return (r["x180_len"] * 1e-9 / 3.0) * (1.0 / r["t1"] + 1.0 / r["t2e"])
    return None


def is_calibrated(r) -> bool:
    """Completed, all graded parameters back in their windows, and an RB the qubit's own coherence can explain (when T1/T2echo are on
    record). A 'valid' RB on the wrong transition, or during a bad stretch, is a measurement but not a calibration."""
    if r["status"] != "completed" or r["gate_fid"] is None or not r.get("graded") or r.get("ballpark") != r.get("graded"):
        return False
    floor = coherence_floor(r)
    return floor is None or (1 - r["gate_fid"]) <= FLOOR_FACTOR * floor


def qc_store_runs():
    """Every RB on qC in the measurement store on 30 Sep: (id, UTC time, reset, EPC %, sigma %, amplitude, offset, decay lengths)."""
    db = sqlite3.connect(STORE / "measurements.sqlite3")
    out = []
    for mid, created, params, fit in db.execute(
            "select m.measurement_id, m.created_at, m.parameters, m.fit_results from measurements m join measurement_targets t "
            "using(measurement_id) where t.target='qC' and m.node_name='Randomized_benchmarking' and m.created_at >= '2026-09-30' "
            "order by m.created_at"):
        f = (json.loads(fit or "{}").get("qC") or {})
        v = lambda k: (f.get(k) or {}).get("value") if isinstance(f.get(k), dict) else f.get(k)  # noqa: E731
        out.append({"id": mid, "utc": created, "local": datetime.fromisoformat(created).astimezone().strftime("%H:%M"),
                    "reset": json.loads(params or "{}").get("reset_type"), "epc": v("error_per_clifford"),
                    "sigma": v("error_per_clifford_sigma"), "amp": v("fit_amplitude"), "offset": v("fit_offset"),
                    "cover": v("decay_lengths_covered")})
    return out


def _png(path: Path, alt: str, caption: str, width="100%"):
    if not path.exists():
        return f"<p class='small muted'>figure missing: {esc(path.name)}</p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:{width};height:auto;border-radius:6px">'
            f'<figcaption class="small muted">{esc(caption)}</figcaption></figure>')


def stage_data(qc_runs):
    """Copy the page's figures and the qC run table into DATA, the report's data folder."""
    (DATA / "figures").mkdir(parents=True, exist_ok=True)
    for mid, name, _ in QC_FIGURES:
        src = STORE / "data" / mid / name
        if src.exists():
            shutil.copy(src, DATA / "figures" / f"qC_rb_{mid}.png")
    for run, q, mid in PEAK_FIGURES:
        src = TINYCAL_RUNS / run / q / "plots" / f"{mid}_amplitude.png"
        if src.exists():
            shutil.copy(src, DATA / "figures" / f"{q}_resonator_{mid}.png")
    run, mid, _ = QC_RAMSEY
    src = TINYCAL_RUNS / run / "qC/plots" / f"{mid}_amplitude_fit.png"
    if src.exists():
        shutil.copy(src, DATA / "figures" / f"qC_ramsey_{mid}.png")
    (DATA / "qc_rb_runs.json").write_text(json.dumps(
        [dict(r, label=QC_RB_LABEL.get(r["id"], ("", ""))[0], what=QC_RB_LABEL.get(r["id"], ("", ""))[1]) for r in qc_runs], indent=1) + "\n")


# ----------------------------------------------------------------------------- tables
def _key(r):
    return (r["night"], r["target"])


def _fid_text(r):
    if r["status"] == "running":
        return f"<span class='muted'>running (turn {r['turns']}, {esc(SHORT.get(r['last'], r['last']))})</span>"
    note = FINAL_NOTE.get(_key(r))
    if r["status"] == "completed":
        cov = r.get("rb_cover")
        if r["gate_fid"] is None:
            if r.get("rb_claimed") is not None and cov is not None:
                return (f"completed, <b>RB not a measurement</b> <span class='small muted'>(claimed {100 * r['rb_claimed']:.2f}; the fit saw "
                        f"{cov:.2f} decay lengths, max depth {r.get('rb_depth')})</span>")
            return "completed, <b>no valid RB</b>" + (f" <span class='small muted'>({esc(note)})</span>" if note else "")
        if cov is not None and cov < n10.RB_EXTRAPOLATED_DECAY_LENGTHS:
            note = (note + "; " if note else "") + f"extrapolated: {cov:.1f} decay lengths, max depth {r.get('rb_depth')}"
        return f"<b>{100 * r['gate_fid']:.2f}</b>" + (f" <span class='small muted'>({esc(note)})</span>" if note else "")
    short = ATTEMPT_SHORT.get(_key(r))
    plain = {"escalated": "stuck", "failed": "aborted"}.get(r["status"], r["status"])
    return f"<span class='small'>{esc(plain)}{' (' + esc(short) + ')' if short else ''}</span>"


def _dmhz(a, b):
    return f"{(a - b) / 1e6:+.2f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "—"


def final_table(rows):
    body = []
    for r in sorted(rows, key=lambda r: (r["target"], r["night"])):
        ref = (f"{100 * r['ref']:.2f}" if r["ref"] else "—") + (
            f"<br><span class='small muted'>averaged {100 * r['ref_avg']:.2f}</span>" if r.get("ref_avg") else "")
        ballpark = f"{r['ballpark']}/{r['graded']}" if r.get("graded") else "—"
        outside = "; ".join(o.split(" came back")[0].split(".", 1)[-1] for o in r["outside"])
        fin, lab = r.get("final") or {}, r["lab"]
        lengths = "; ".join(f"{esc(op.split('.', 1)[1])} {b}→{a} ns" for op, b, a in r.get("lengths") or []) or "<span class='muted'>none</span>"
        vs_lab = (f"f₀₁ {_dmhz(fin.get('f_01'), lab.get('f_01'))} · res {_dmhz(g(fin, 'resonator', 'f_01'), g(lab, 'resonator', 'f_01'))}"
                  if fin else "—")
        body.append(
            f"<tr><td class='mono'>{esc(r['target'])} <span class='small muted'>{esc(r['night'])}</span></td><td>{_fid_text(r)}</td>"
            f"<td class='num'>{ref}</td><td class='num'>{r['nodes'] if r.get('nodes') is not None else '—'}"
            f"{'<br><span class=\"small muted\">' + str(r['lib_runs']) + ' by library id</span>' if r.get('lib_runs') else ''}</td>"
            f"<td class='num'>{fmt(r['qpu_s'], '{:.0f} s')}</td><td class='num'>{minutes(r['model_s'])}</td><td class='num'>{minutes(r['queue_s'])}</td>"
            f"<td class='num'>{minutes(r['wall_s'])}</td><td class='num'>{fmt(r['cost'], '${:.2f}')}</td><td class='num'>{ballpark}"
            f"{'<br><span class=\"small muted\">' + esc(outside) + '</span>' if outside else ''}</td>"
            f"<td class='num small'>{vs_lab}</td><td class='small'>{lengths}</td></tr>")
    settled = [r for r in rows if r["status"] != "running"]
    done = [r for r in settled if r["status"] == "completed"]
    calibrated = [r for r in done if is_calibrated(r)]
    head = ("<thead><tr><th>qubit <span class='small muted'>run</span></th><th>gate fidelity, %</th><th>IQCC's RB on file, %"
            "<br><span class='small muted'>from 1QRB_p · the averaged field</span></th><th>distinct nodes completed</th><th>QPU</th><th>agent</th><th>queue</th>"
            "<th>wall</th><th>cost</th><th>scrambled params back in ballpark</th><th>final − lab, MHz</th>"
            "<th>pulse lengths changed<br><span class='small muted'>flagged, not graded</span></th></tr></thead>")
    foot = (f"<tr><td><b>calibrated</b></td><td colspan='11'><b>{len(calibrated)}/{len(settled)}</b> qubit-runs with all graded parameters "
            f"back and an RB within {FLOOR_FACTOR:g}× the coherence floor ({', '.join(r['target'] for r in calibrated)}) · "
            f"{len(done)}/{len(settled)} reached the end of the recipe"
            + (f" · {len(rows) - len(settled)} running" if len(rows) > len(settled) else "") + "</td></tr>")
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}{foot}</tbody></table></div>'


def device_table(lab: dict, wiring: dict):
    body = []
    for q in sorted(lab):
        x = lab[q]
        ro = g(x, "resonator", "operations", "readout") or {}
        rb, avg, stamp = lab_reference(x)
        rr = g(wiring, "wiring", "qubits", q, "rr") or g(wiring, "qubits", q, "rr") or {}
        port = lambda s: s.split("/ports/")[-1] if isinstance(s, str) else "—"  # noqa: E731
        us = lambda v: f"{v * 1e6:.1f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
        body.append(
            f"<tr><td class='mono'>{esc(q)}</td><td class='num'>{x['f_01'] / 1e9:.4f}</td><td class='num'>{(x.get('anharmonicity') or 0) / 1e6:.0f}</td>"
            f"<td class='num'>{g(x, 'resonator', 'f_01') / 1e9:.4f}</td><td class='num'>{(g(x, 'resonator', 'f_01') - x['f_01']) / 1e9:.2f}</td>"
            f"<td class='num'>{fmt(ro.get('amplitude'), '{:.3f}')}</td><td class='num'>{ro.get('length')}</td>"
            f"<td class='num'>{us(x.get('T1'))}</td><td class='num'>{us(x.get('T2ramsey'))}</td><td class='num'>{us(x.get('T2echo'))}</td>"
            f"<td class='num'>{fmt(100 * rb if rb else None, '{:.3f}')}</td><td class='num'>{fmt(100 * avg if avg else None, '{:.2f}')}</td>"
            f"<td class='mono small'>{esc(port(rr.get('opx_output')))} → {esc(port(rr.get('opx_input')))}</td></tr>")
    head = ("<thead><tr><th>qubit</th><th>f₀₁, GHz</th><th>α, MHz</th><th>resonator, GHz</th><th>resonator − qubit, GHz</th><th>readout amp</th>"
            "<th>readout, ns</th><th>T1, µs</th><th>T2*, µs</th><th>T2 echo, µs</th><th>RB 1 − EPG, %</th><th>averaged, %</th>"
            "<th>readout line (out → in)</th></tr></thead>")
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}</tbody></table></div>'


def error_scatters(rows):
    ref_pts, floor_pts = [], []
    for r in rows:
        if r["status"] != "completed" or r["gate_fid"] is None:
            continue
        err = 100 * (1 - r["gate_fid"])
        label = f"{r['target']} ({r['night']}) · {pct(r['gate_fid'])}"
        if r["ref"] and r["ref"] < 1:
            ref_pts.append((100 * (1 - r["ref"]), err, label + f" · IQCC {pct(r['ref'])}"))
        if r["t1"] and r["t2e"] and r["x180_len"]:
            floor = 100 * (r["x180_len"] * 1e-9 / 3.0) * (1.0 / r["t1"] + 1.0 / r["t2e"])
            floor_pts.append((floor, err, label + f" · T1 {1e6 * r['t1']:.0f} µs, T2e {1e6 * r['t2e']:.0f} µs, x180 {r['x180_len']:.0f} ns · floor {floor:.3f}%"))
    a = n10._scatter(ref_pts, xlabel="IQCC's gate error on file, %", ylabel="measured gate error, %") if ref_pts else ""
    b = n10._scatter(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %", ylabel="measured gate error, %",
                     guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x")) if floor_pts else ""
    return (f"<div class='scattergrid2'><div class='panel'><div class='ptitle small'><b>Against IQCC's RB on file</b> · {len(ref_pts)} runs</div>{a}"
            f"<p class='small muted'>IQCC's number is 1 − EPG from the RB decay it stores (extras.1QRB_p) in the 15 Aug upload, the same "
            f"definition as the node's; when it was measured is not recorded. Below the diagonal the run beat it.</p></div>"
            f"<div class='panel'><div class='ptitle small'><b>Against the coherence floor</b> · {len(floor_pts)} runs</div>{b}"
            f"<p class='small muted'>Floor = (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) with the run's own T1, T2echo and x180 length. On the "
            f"diagonal the gate is decoherence-limited; far above it, the control is the limit.</p></div></div>")


def node_qpu_table(rows, n10_rows):
    done = [r for r in rows if r["status"] == "completed"]
    now, flux = collections.defaultdict(list), collections.defaultdict(list)
    for r in done:
        for node, qpu, *_ in r["node_runs"]:
            if qpu is not None and qpu > 0:
                now[node].append(qpu)
    for r in n10_rows:
        if r["status"] != "completed":
            continue
        for node, qpu, *_ in r["node_runs"]:
            if qpu is not None and qpu > 0:
                flux[node].append(qpu)
    n_done = len(done) or 1
    total = sum(sum(v) for v in now.values()) or 1
    names = RECIPE_NODES + [n for n in now if n not in RECIPE_NODES]
    body, s_now, s_flux = [], 0.0, 0.0
    for node in names:
        v, o = now.get(node, []), flux.get(node, [])
        if not v and node not in RECIPE_NODES:
            continue
        m, mo = median(v), median(o)
        s_now += m or 0
        s_flux += mo or 0
        body.append(f"<tr><td>{esc(SHORT.get(node, node))}</td><td class='num'>{len(v) / n_done:.1f}</td>"
                    f"<td class='num'><b>{fmt(m, '{:.1f}')}</b><span class='small muted'>{' (' + fmt(min(v), '{:.1f}') + '–' + fmt(max(v), '{:.1f}') + ')' if len(v) > 1 else ''}</span></td>"
                    f"<td class='num'>{fmt(mo, '{:.1f}') if o else '<span class=muted>not in that graph</span>'}</td>"
                    f"<td class='num'>{100 * sum(v) / total:.0f}%</td></tr>")
    body.append(f"<tr><th class='rowh'>one execution of every node</th><td></td><td class='num'><b>{s_now:.0f} s</b></td>"
                f"<td class='num'>{s_flux:.0f} s</td><td></td></tr>")
    head = ("<thead><tr><th>node</th><th>runs per completed bring-up</th><th>QPU per run on lucy, median (range), s</th>"
            "<th>28–29 Sep, flux-tunable chips, median, s</th><th>share of lucy's QPU</th></tr></thead>")
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}</tbody></table></div>'


def qc_table(qc_runs, rows):
    agent_ids = {}
    for r in rows:
        if r["target"] != "qC":
            continue
        for e in _events(r["run_id"], "qC"):
            if e.get("tool") == "run_node" and e.get("node") == "Randomized_benchmarking" and e.get("node_run_id"):
                agent_ids[e["node_run_id"]] = f"{r['night']} agent"
    body = []
    for x in qc_runs:
        label, what = QC_RB_LABEL.get(x["id"], (agent_ids.get(x["id"], "—"), "the agent's own RB" if x["id"] in agent_ids else ""))
        bad = (x["epc"] or 0) > 1.0
        body.append(f"<tr{' class=\"hl\"' if bad else ''}><td class='num'>{esc(x['local'])}</td><td class='mono small'>{esc(x['id'])}</td>"
                    f"<td>{esc(label)}</td><td class='small'>{esc(what)}</td><td>{esc(x['reset'] or '—')}</td>"
                    f"<td class='num'><b>{fmt(x['epc'], '{:.2f}')}</b> <span class='small muted'>± {fmt(x['sigma'], '{:.2f}')}</span></td>"
                    f"<td class='num'>{fmt(x['amp'], '{:.2f}')}</td><td class='num'>{fmt(x['offset'], '{:.2f}')}</td>"
                    f"<td class='num'>{fmt(x['cover'], '{:.1f}')}</td></tr>")
    head = ("<thead><tr><th>time</th><th>measurement</th><th>state</th><th>what changed</th><th>reset</th><th>error per Clifford, %</th>"
            "<th>amplitude</th><th>offset</th><th>decay lengths</th></tr></thead>")
    return f'<div class="scroll"><table class="grid small">{head}<tbody>{"".join(body)}</tbody></table></div>'


def pins_table(rows):
    git = lambda p, *a: base._git(p, *a)  # noqa: E731
    shas = collections.Counter(str(r.get("fingerprint", {}).get("git_sha", ""))[:7] for r in rows if r.get("fingerprint"))
    spec = collections.Counter(str(r.get("scramble", {}).get("spec_hash", "")) for r in rows if r.get("scramble"))
    src = collections.Counter(str(r.get("scramble", {}).get("source_hash", "")) for r in rows if r.get("scramble"))
    fmtc = lambda c: ", ".join(f"<span class='mono'>{esc(k)}</span> ({v})" for k, v in c.most_common()) or "—"  # noqa: E731
    body = []
    for night in NIGHTS:
        wt = RUNS / f"tinycal-{night}"
        changed = git(wt, "status", "--short").splitlines()
        body.append(f"<tr><td>tinycal ~/qab-runs/tinycal-{night} (detached worktree + the main working tree of that hour)</td>"
                    f"<td class='mono'>{esc(git(wt, 'rev-parse', '--short', 'HEAD'))}</td><td class='small'>{len(changed)} changed or new files, "
                    f"recipe bringup_recipes/fixed_frequency_1q.md; document fingerprint.git_sha: {fmtc(shas)}</td></tr>")
    ql = RUNS / "qua-libs-lucy1"
    body += [
        f"<tr><td>qua-libs ~/qab-runs/qua-libs-lucy1 (both runs)</td><td class='mono'>{esc(git(ql, 'rev-parse', '--short', 'HEAD'))}</td>"
        f"<td class='small'>{len(git(ql, 'status', '--short').splitlines())} changed or new files: the n13 fixes and the 03a search capped at the drive LO's reach</td></tr>",
        "<tr><td>shared tinycal venv</td><td class='mono'>quam-builder 0.6.0</td><td class='small'>qualibration-libs 0.3.1, quam 0.6.0, qm-qua 1.4.1</td></tr>",
        f"<tr><td>scramble spec (~/qab-runs/lucy1/decalibrate_lucy.yaml)</td><td class='mono'>{esc(next(iter(spec), ''))}</td>"
        f"<td class='small'>decalibrate_chip.yaml without the five flux fields and the readout-amplitude kick; document scramble.spec_hash: {fmtc(spec)}</td></tr>",
        f"<tr><td>lucy snapshot (source-state, cloud state 59619 of 27 Sep)</td><td class='mono'>{esc(next(iter(src), ''))}</td>"
        f"<td class='small'>document scramble.source_hash: {fmtc(src)}</td></tr>",
        "<tr><td>acceptance spec (workloads/acceptance.yaml)</td><td class='mono'>821bb4dd1d5d0e5a</td><td class='small'>stamped by qab accept</td></tr>",
    ]
    return ('<div class="scroll"><table class="grid small"><thead><tr><th>component</th><th>sha / digest</th><th>as recorded</th></tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- page
def build():
    rows = collect()
    qc_runs = qc_store_runs()
    stage_data(qc_runs)
    n10_rows = n10.collect()
    lab_state = json.load(open(sorted(RUNS.glob("lucy2-lucy-2026*"))[-1] / "source-state/state.json"))
    wiring = json.load(open(sorted(RUNS.glob("lucy2-lucy-2026*"))[-1] / "source-state/wiring.json"))
    running = [r for r in rows if r["status"] == "running"]
    settled = [r for r in rows if r["status"] != "running"]
    done = [r for r in settled if r["status"] == "completed"]
    valid = [r for r in done if r["gate_fid"] is not None]
    calibrated = [r for r in done if is_calibrated(r)]
    fids = sorted(r["gate_fid"] for r in calibrated)
    log2 = RUNS / "lucy2-launch.log"
    live = bool(running) or not log2.exists() or "all five cells finished" not in log2.read_text()
    now = datetime.now().strftime("%d %b %Y %H:%M")
    spent = sum(r["cost"] or 0 for r in rows)
    med = lambda k: median([r[k] for r in calibrated if r.get(k) is not None])  # noqa: E731
    flux_done = [r for r in n10_rows if r["status"] == "completed"]
    tiles = [
        f'<div class="tile"><div class="v">{len(calibrated)}/{len(settled)}</div><div class="k">qubit-runs calibrated: graded parameters back, RB within '
        f'{FLOOR_FACTOR:g}× the coherence floor{" so far" if live else ""} · {len(done)} reached the end of the recipe'
        f'{" · " + str(len(running)) + " running" if running else ""}</div></div>',
        f'<div class="tile"><div class="v">{pct(fids[len(fids) // 2]) if fids else "—"}</div><div class="k">median gate fidelity of the calibrated runs '
        f'({pct(fids[0]) if fids else "—"} – {pct(fids[-1]) if fids else "—"}); {sum(1 for f in fids if f >= 0.999)} at ≥ 99.9 %</div></div>',
        f'<div class="tile"><div class="v">{fmt(med("qpu_s"), "{:.0f} s")}</div><div class="k">median QPU per calibrated bring-up'
        + (f' (flux-tunable chips, 28–29 Sep: {median([r["qpu_s"] for r in flux_done if r["qpu_s"]]) / 60:.1f} min)' if flux_done else "") + '</div></div>',
        f'<div class="tile"><div class="v">{minutes(med("wall_s"))}</div><div class="k">median wall per calibrated bring-up'
        + (f' (28–29 Sep: {minutes(median([r["wall_s"] for r in flux_done if r["wall_s"]]))})' if flux_done else "") + '</div></div>',
        f'<div class="tile"><div class="v">${spent:,.2f}</div><div class="k">judge-priced OpenRouter spend{" so far" if live else ""}</div></div>',
    ]
    lab = lambda it, plain=False: it["label"] if plain else esc(it["label"])  # noqa: E731
    err_fig = bars(sorted([{"fw": "tinycal", "err": 100 * (1 - r["gate_fid"]), "label": f"{r['target']} ({r['night']})"} for r in valid],
                          key=lambda it: it["label"]),
                   "err", lab, lambda v: f"{v:.3f}% ({100 - v:.2f}%)", "Single-qubit gate error per completed bring-up",
                   "gate error in %, fidelity in brackets; error per Clifford ÷ 1.875")
    timeline = '<table class="grid timeline"><tbody>' + "".join(f"<tr><td>{esc(t)}</td><td>{esc(w)}</td></tr>" for t, w in INCIDENTS) + "</tbody></table>"
    banner = (f'<div class="tile" style="border-color:var(--warn)"><div class="k"><b>Live page</b> — regenerated {esc(now)} while cells are still '
              f'running: {esc(", ".join(r["target"] + " (" + r["night"] + ")" for r in running)) or "lucy2 not finished"}. Running rows are interim.</div></div>') if live else ""
    narrative = "\n".join(NARRATIVE) or "<p class='muted'>Written when lucy2 ends.</p>"
    flux_hits = [f"{r['target']} ({r['night']}): {', '.join(r['flux_nodes_run'])}" for r in done if r.get("flux_nodes_run")]
    qc_figs = "".join(_png(DATA / "figures" / f"qC_rb_{mid}.png", f"qC RB {mid}", cap) for mid, _, cap in QC_FIGURES)
    ramsey_fig = _png(DATA / "figures" / f"qC_ramsey_{QC_RAMSEY[1]}.png", "qC Ramsey", QC_RAMSEY[2])
    peak_figs = "".join(_png(DATA / "figures" / f"{q}_resonator_{mid}.png", f"{q} resonator", PEAK_CAPTIONS[mid]) for _, q, mid in PEAK_FIGURES)
    power_runs = [(n, o) for r in settled for n, _, o, *_ in r["node_runs"] if n == "resonator_spectroscopy_vs_power"]
    peak_fail = f"{sum(1 for _, o in power_runs if o != 'successful')} of its {len(power_runs)} runs"
    grid_fig = _png(GRID_FIGURE, "IQCC grid figure", "IQCC's own grid figure for lucy, 24 Jul 2026 02:06: eight qubits on a ring, "
                    "1Q RB per qubit (qA 99.02 %). The cloud state has had seven qubits since the 15 Aug upload.", width="72%")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lucy fixed-frequency bring-up</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}
.scattergrid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:6px 0 14px}} @media (max-width:760px){{.scattergrid2{{grid-template-columns:1fr}}}}
.scattergrid2 .panel{{min-width:0}} .scattergrid2 .ptitle{{margin-bottom:4px}}
.figs2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} @media (max-width:760px){{.figs2{{grid-template-columns:1fr}}}}
figure img{{background:#fff}} tr.hl td{{background:color-mix(in srgb, var(--warn, #b45309) 12%, transparent)}}
svg.scatter{{display:block;width:100%;height:auto}} svg.scatter .grid{{stroke:var(--surface-2,#eef2f2);stroke-width:1}}
svg.scatter .frame{{fill:none;stroke:var(--muted,#8a8f98);stroke-width:.8}} svg.scatter .guide{{stroke:var(--muted,#8a8f98);stroke-width:1;stroke-dasharray:4 3}}
svg.scatter .tick{{font-size:10px;fill:var(--muted,#8a8f98);font-family:inherit}} svg.scatter .lab{{font-size:11px;fill:var(--ink-2,#4b555b);font-family:inherit}}
svg.scatter .pt{{fill-opacity:.8;stroke:var(--surface,#fff);stroke-width:.8}}</style></head><body><div class="page">
<div class="eyebrow">qua-agents benchmark · a new IQCC device · generated {esc(now)}</div>
<h1 style="margin-top:8px">lucy, a fixed-frequency coaxmon ring: the first single-qubit bring-up with tinycal and qwen3.8-27b, 30 Sep 2026</h1>
<p class="lede">One framework, one model, one host, a new kind of chip. lucy is IQCC's fixed-flux, fixed-coupling device: its qubits are
coaxmons (transmons with out-of-plane wiring), with no flux line to tune and no tunable couplers. tinycal with qwen3.8-27b on OpenRouter
(effort high, 16 000-token cap) calibrated each qubit from a scrambled state through a fixed-frequency version of the bring-up recipe — the
flux-tunable graph's nodes without its four flux nodes, resonator spectroscopy (02a) in place of resonator identification — from the resonator
to randomized benchmarking. Targets: every qubit in lucy's cloud state, qA–qG; qC twice. lucy1 ran qA, qB, qC at 10:15; lucy2 ran qD–qG and qC
again from 10:58. At most three targets were in flight. Numbers are read from each qubit-run's own result.json, tinycal's event log and
measurement store, and the judge's price table; the operator annotations are marked.</p>
{banner}

<h2 id="overview">Overview</h2>
<div class="tiles">{"".join(tiles)}</div>

<h3>Final per-target results</h3>
<p class="small muted">One row per qubit-run. QPU = execution on the chip; agent = time inside model calls; queue = the cloud queue; wall = first
to last event. IQCC's RB on file is 1 − EPG from the RB decay stored in the lab state (the node's definition); below it, the separate
gate_fidelity.averaged field, a different protocol. "Scrambled params back in ballpark" is the judge's identity check on the three graded
parameters (f₀₁, resonator, x180 amplitude). The last column is the final state against the unscrambled lab state.</p>
{final_table(rows)}
{error_scatters(rows)}
{err_fig}

<h2 id="device">The device</h2>
<p><b>What the cloud state holds.</b> A <span class="mono">FixedFrequencyQuam</span> with seven FixedFrequencyTransmons (qA–qG) and no
<span class="mono">z</span> line on any of them. IQCC's hourly automatic calibration of lucy stopped on 24 Jul 00:06 UTC (the grid figure below is
from that day, with an eighth qubit, qH). Since then two manual uploads: 15 Aug (qH and the qubit pairs' active list dropped, every qubit
parameter set to values found in no state since 16 Jul) and 27 Sep 13:43 UTC (only the resonator frequencies moved, 0.1–7.9 MHz; the
cross-resonance pairs removed). So the resonators are two days old and f₀₁, the amplitudes and thresholds are at least six weeks old.</p>
<p><b>What makes it different to calibrate.</b> Each qubit has a readout line of its own (seven resonators on seven lines; arbel, qolab and
gilboa put 5–6 on a feedline), so the line on a qubit's readout is its resonator. The resonators sit 4.9–5.6 GHz above the qubits, against
0.6–2.2 GHz on the flux-tunable chips: the dispersive shift and the power-dependent drift scale as g²/Δ, readout runs 1.5–1.9 µs with no
TWPA, and the qubits sit at 4.06–4.57 GHz. Anharmonicities are 170–178 MHz.</p>
{device_table(lab_state["qubits"], wiring)}
{grid_fig}
<h3 id="peak">The resonators are peaks, and the power sweep reads only dips</h3>
<p>Each resonator is measured in transmission, between an input and an output line of its own: off resonance almost nothing reaches the
output, on resonance the resonator passes it, so |IQ| is a Lorentzian <i>peak</i> (with a small notch beside it, likely a weak direct path
between the pins). On arbel, qolab and gilboa the resonators hang off a shared feedline and pull the tone out at resonance: a dip. 02a finds
either shape. 02b, the resonator-vs-power sweep, fits only dips: on lucy it came back "no_resonance" or "unresolved" on {peak_fail}, or reported a
line where there was only the notch beside the peak. Where the model believed it, the readout was damaged: qF's power cut 19 dB, qD's tripled,
qG's frequency moved 6.8 MHz off its resonator — the three worst runs of lucy2 (<a href="#hard">hard cases</a>). The fix proposed after the run is a
per-resonator line shape in the state (<span class="mono">resonator.extras["line_shape"]</span>: dip or peak), written by 02a and read by 02b and the
other nodes that assume a dip (02c, 08a's analysis, the circle fit).</p>
<div class="figs2">{peak_figs}</div>

<h2 id="setup">What it took to run it</h2>
<p><b>The recipe.</b> tinycal takes only the node set and presets from the graph named in the profile, so the flux-tunable graph's catalog was
kept (the current chirp nodes, trimmed sweeps, active reset after IQ_blobs) and a new recipe, <span class="mono">bringup_recipes/fixed_frequency_1q.md</span>,
tells the model the order: 02a resonator spectroscopy (a library node outside the catalog; resonator identification measures by moving the
flux, and reads <span class="mono">qubit.z</span>), the power sweep once, the chirp line search and T1 with the chirp, the fine scan, power Rabi,
readout optimisation, IQ blobs, x180 error amplification, Ramsey, T1 and T2 echo, DRAG, RB — and never the four flux nodes. It carries the flux
recipe's 29–30 Sep edits and three notes for this wiring: the line on a qubit's own readout is its resonator; with the resonator far above the
qubit "no onset in the swept range" means exactly that — wrong, as it turned out: 02b never saw the line at all (next section); and a long readout
against a short T1 limits IQ blobs (lucy2 ran with one more sentence there: readout error scales RB's amplitude, not its decay).
{("Flux nodes run anyway: " + esc("; ".join(flux_hits)) + ". Each failed while building its program (the qubit has no z line) and used no QPU.") if flux_hits else "No agent ran a flux node."}</p>
<p><b>The state loads as it is.</b> QuAM builds the machine from the state's own class, so lucy loads as a FixedFrequencyQuam through qua-libs'
flux-tunable <span class="mono">quam_config</span>, and <span class="mono">initialize_qpu</span> sets no flux. Before any QPU time every recipe node
built and serialised its QUA program for qA and qB with the connection stubbed.</p>
<p><b>The scramble.</b> The benchmark's decalibrate_chip.yaml refuses lucy twice: the readout amplitude's ×1.8 kick takes qA, qB, qE and qF above
the 1.0 ceiling, and five flux fields (<span class="mono">z.joint_offset</span>, <span class="mono">z.independent_offset</span>, φ₀ current and
voltage, the flux curvature) match no qubit — a field that matches nothing is refused even with <span class="mono">on_missing: skip</span>. The lucy
variant drops those six fields: 161 edits across the chip, f₀₁ −50 MHz, resonators +25 MHz, x180/x90 ×0.5, anharmonicity set to 200 MHz,
readout thresholds, angles, DRAG α and the coherence times reset. The readout amplitude is neither scrambled nor graded.</p>

<h2 id="qpu">QPU time per node</h2>
<p class="small muted">Every node run of the completed bring-ups, measured by the IQCC worker. The comparison column is the 28–29 Sep full bring-up on
the flux-tunable chips (same model, host and presets, qua-libs of that night).</p>
{node_qpu_table(rows, n10_rows)}

<h3>What the model did with each node</h3>
<p class="small muted">Every settled qubit-run, from tinycal's events.jsonl. <b>Sequence</b>, against the fixed-frequency recipe's order.
<b>Proposals</b>, per run that proposed state updates, from the writes before the next node run. Percentages are of the node's runs (failed) or of
its runs with a proposal.</p>
{n10.decision_table(settled)}

<h2 id="qc">qC: two bad stretches, not a bad calibration</h2>
<p>lucy1's qC finished the recipe at 97.48 %, 2.5 % per gate on a qubit whose T1 (38 µs) and T2 echo (32 µs) allow ~0.1 %. Every RB run on qC that
day is below, from the measurement store: the agent's, then fourteen in propose mode on copies of the state it left (nothing written). The bad values
all fall between 10:28 and 10:33; from 10:47 the same state gives 0.11–0.12 % per Clifford with thermal reset and 0.14–0.25 % with active reset,
whatever the DRAG α or the depletion wait. The first comparison of the two resets (10:47) came after the bad stretch had ended, so it does not
convict active reset; what active reset does cost qC is contrast (amplitude 0.22–0.32 against 0.37–0.38), rising with the depletion wait — which
also gives a mis-flipped qubit time to relax (T1 38 µs).</p>
{qc_table(qc_runs, rows)}
<div class="figs2">{qc_figs}</div>
<p><b>Why the stretch was bad is not established.</b> The clues point at qC's frequency moving by ~1 MHz for a while: the fine scan at 10:20 put
f₀₁ at 4.3016 GHz and Ramsey at 10:25 at 4.3004 (1.2 MHz apart, against 0.14–0.36 MHz on qA and qB); the Ramsey's first 700 ns do not follow
its fit; x180 error amplification (10:24) kept a residual no amplitude removed (0.056 cycles per pulse against the 0.01 floor on qA/qB); and the
first DRAG (10:27) failed. A two-level defect coupled to the qubit would do this on a fixed-frequency chip. A series of short Ramseys would tell.</p>
{ramsey_fig}
<p><b>It happened again in lucy2.</b> The rerun started from the same scrambled state, and its readout was the one that had worked. Its chirp line
search saw nothing ten times between 11:01 and 11:23, although the default window holds qC's line, and the model went on to calibrate a line 38 MHz
below it (its three RB runs, 26 %, 7.2 % and a failed fit, are in the table above as "lucy2 agent"). At 13:07 the same chirp, from lucy1's final state,
found the 0→1 line at 4.30034 GHz, 45 kHz from 10:25's Ramsey, with its two-photon partner. qC was clean at 10:18, 10:47–10:53 and 13:07 and unreadable
around 10:20–10:35 and 11:00–11:25: a qubit to hold back from the benchmark until that is understood.</p>

<h3 id="hard">Hard cases: qubit-runs that were not calibrated, or not calibrated right</h3>
<p class="small muted">Settled runs that did not reach the end of the recipe, then completed ones under 99 %. The right-hand column is the operator's
reading of the transcript, not something the cell recorded.</p>
{n10.hard_cases(settled)}

<h3>What the numbers say{" (so far)" if live else ""}</h3>
{narrative}

<h2 id="stuck">Incident timeline</h2>
{timeline}

<h2 id="pins">Pins: what a new cell must use to be comparable</h2>
{pins_table(rows)}

<h2 id="runs">Run identifiers</h2>
{n10.run_ids(rows)}
</div></body></html>"""
    OUT.write_text(page)
    if ARTIFACT_DIR:  # the Artifact skeleton supplies doctype, html, head and body: keep the title, stylesheet and content
        body = page.split("<title>", 1)[1]
        (Path(ARTIFACT_DIR) / OUT.name).write_text("<title>" + body.replace("</head><body>", "").replace("</body></html>", ""))
    print(f"wrote {OUT} · {len(rows)} qubit-runs, {len(done)} completed, {len(running)} running, ${spent:,.2f}")


# the n10 builders key their per-qubit notes by (backend, qubit); here "backend" carries the run (lucy1 / lucy2)
n10.FINAL_NOTE.clear(); n10.FINAL_NOTE.update(FINAL_NOTE)  # noqa: E702
n10.ATTEMPT_SHORT.clear(); n10.ATTEMPT_SHORT.update(ATTEMPT_SHORT)  # noqa: E702
n10.HARD_CASE_NOTE.clear(); n10.HARD_CASE_NOTE.update(HARD_CASE_NOTE)  # noqa: E702

if __name__ == "__main__":
    build()
