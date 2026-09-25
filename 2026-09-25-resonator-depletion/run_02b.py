"""02b resonator spectroscopy vs power under four depletion-wait conditions (25 Sep 2026).

usage: run_02b.py <backend> <condition> <round> [--dry]

  A  today's node (power inside, frequency outside), depletion_time as stored and passed to wait()
     unconverted: 3000 ns waits 12 us (gilboa qC3's 5000 ns waits 20 us)
  B  today's node with depletion_time divided by 4 in the local state copy: the 3 us the state
     intends, i.e. the unit fix alone, power still inside
  C  the fix/resonator-depletion-wait node (power outside, frequency inside, depletion_time
     converted to clock cycles, 1 ms reset once per round), depletion_time as stored (3 us)
  D  as C with depletion_time 1000 ns in the local copy: a deliberately short wait (1 us)

One job per qubit (the node's default, not multiplexed). Every condition sweeps the same window per
qubit: 100 powers over 35 dB, the top at min(-10 dBm, committed power + 20 dB) -- -10 dBm is the most
the node can reach with its 0.1 amplitude cap -- and the node's default frequency grid (15 MHz at
0.05 MHz). State: the cloud state pulled 18:03 (~/qab-runs/state-20260925-1803; gilboa's C/D qubits
marked active in the local copy). Propose mode, nothing written to any state. Outputs in ab/<backend>/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

backend, cond, rnd = sys.argv[1], sys.argv[2], int(sys.argv[3])
DRY = "--dry" in sys.argv
TREE = {"A": "qua-libs", "B": "qua-libs", "C": "qua-libs-depletion", "D": "qua-libs-depletion"}[cond]
WT = Path.home() / "code/QM" / TREE / "qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}[backend]
GILBOA_ACTIVE = ["qC1", "qC2", "qC3", "qC4", "qC5", "qD1", "qD2", "qD3", "qD4", "qD5"]
SNAP = Path.home() / "qab-runs/state-20260925-1803" / backend
OUT = HERE / "ab" / backend
OUT.mkdir(parents=True, exist_ok=True)
SPAN_DB = 35
TOP_CEILING_DBM = -10

# The local state copy for this condition.
STATE = OUT / f"state_{cond}"
STATE.mkdir(exist_ok=True)
st = json.loads((SNAP / "state.json").read_text())
if backend == "gilboa":
    st["active_qubit_names"] = GILBOA_ACTIVE
stored_depletion = {q: int(st["qubits"][q]["resonator"]["depletion_time"]) for q in QUBITS}
for q in QUBITS:
    rr = st["qubits"][q]["resonator"]
    if cond == "B":
        rr["depletion_time"] = int(round(stored_depletion[q] / 4))
    elif cond == "D":
        rr["depletion_time"] = 1000
(STATE / "state.json").write_text(json.dumps(st))
shutil.copy(SNAP / "wiring.json", STATE / "wiring.json")
effective_wait_ns = {q: {"A": 4 * stored_depletion[q], "B": 4 * int(round(stored_depletion[q] / 4)),
                         "C": stored_depletion[q], "D": 1000}[cond] for q in QUBITS}

os.environ.setdefault("MPLBACKEND", "Agg")
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import capture_job_timings, configure_iqcc  # noqa: E402

configure_iqcc(STATE, IqccConfig(enabled=not DRY, backend=backend, default_timeout_s=180))
import qualibrate_ai  # noqa: E402

qualibrate_ai.configure(data_root=OUT / "store")
LIB = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam  # noqa: E402

import calibration_utils  # noqa: E402

assert str(WT) in calibration_utils.__file__, calibration_utils.__file__


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [{backend}/02b-{cond}-r{rnd}] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free() -> None:
    while not DRY:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        if not (pids - FROZEN):
            return
        log(f"backend busy (pids {sorted(pids - FROZEN)}); waiting 60 s")
        time.sleep(60)


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return js(x.tolist())
    if isinstance(x, (str, bool)) or x is None:
        return x
    return repr(x)


def window(machine, q):
    p_ro = float(machine.qubits[q].resonator.get_output_power("readout"))
    top = int(min(TOP_CEILING_DBM, round(p_ro + 20)))
    return p_ro, top, top - SPAN_DB


SKIP_DRY = ["execute_qua_program", "simulate_qua_program", "process_data", "fit_data", "plot_data", "update_state",
            "save_results", "load_data"]


def run_one(q: str) -> None:
    machine = Quam.load(str(STATE))
    p_ro, top, bottom = window(machine, q)
    params = dict(min_power_dbm=bottom, max_power_dbm=top)
    tag = f"{cond}-r{rnd}-{q}"
    node = LIB.nodes["02b_resonator_spectroscopy_vs_power"].copy(name="02b_resonator_spectroscopy_vs_power")
    node.bind_machine(machine)
    if DRY:
        from qm import generate_qua_script
        node.run(mode="propose", interactive=True, external=True, skip_actions=SKIP_DRY, qubits=[q], **params)
        txt = generate_qua_script(node.namespace["qua_program"], machine.generate_config())
        log(f"{tag}: program built ({len(txt)} chars), window {bottom}..{top} dBm (committed {p_ro:.1f}), "
            f"wait {effective_wait_ns[q]} ns")
        return
    wait_until_free()
    log(f"{tag}: window {bottom}..{top} dBm (committed {p_ro:.1f} dBm), effective depletion wait "
        f"{effective_wait_ns[q]} ns, tree {TREE}")
    error = None
    t0 = time.perf_counter()
    with capture_job_timings() as timings:
        try:
            node.run(mode="propose", interactive=True, external=True, qubits=[q], **params)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    wall = time.perf_counter() - t0
    qpu = sum(e.get("qpu_execution_s") or 0 for e in timings)
    stderr = [e.get("executor_stderr") for e in timings if e.get("executor_stderr")]
    ds = node.results.get("ds_raw")
    if ds is not None:
        try:
            ds.to_netcdf(OUT / f"{tag}.nc")
        except Exception as exc:  # noqa: BLE001
            log(f"{tag}: netcdf failed ({exc})")
    for label, fig in (node.results.get("figures") or {}).items():
        try:
            fig.savefig(OUT / f"{tag}_{label}.png", dpi=100, bbox_inches="tight")
        except Exception:  # noqa: BLE001
            pass
    record = {"tag": tag, "condition": cond, "round": rnd, "qubit": q, "tree": TREE, "state": str(SNAP),
              "committed_power_dbm": p_ro, "window_dbm": [bottom, top], "stored_depletion_ns": stored_depletion[q],
              "effective_wait_ns": effective_wait_ns[q], "error": error, "wall_s": wall, "qpu_s": qpu,
              "executor_stderr": stderr, "fit_results": node.results.get("fit_results"),
              "proposed": node.proposed_state_updates}
    (OUT / f"{tag}.json").write_text(json.dumps(js(record), indent=1))
    with open(OUT / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(js({k: record[k] for k in ("tag", "condition", "round", "qubit", "effective_wait_ns",
                                                         "error", "wall_s", "qpu_s")})) + "\n")
    r = (node.results.get("fit_results") or {}).get(q) or {}
    keys = ("success", "optimal_power", "onset_power", "onset_mechanism", "resonator_frequency", "frequency_shift",
            "dressed_frequency", "linewidth", "plateau_contrast", "contrast_at_optimal", "bare_frequency")
    log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s, wall {wall:.0f} s"
        + (f", stderr: {' | '.join(s.strip()[:200] for s in stderr)}" if stderr else ""))
    log(f"   {q}: {js({k: r.get(k) for k in keys})} warnings={r.get('warnings')}")


for q in QUBITS:
    if ONLY is None or q in ONLY:
        run_one(q)
log("condition done")
