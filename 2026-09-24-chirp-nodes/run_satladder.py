"""Saturation drive ladder for the saturation-vs-chirp capture comparison (24 Sep 2026).

usage: run_satladder.py <backend>

The saturation node 03a_qubit_spectroscopy (unchanged) over +-200 MHz at 0.25 MHz steps, 150 shots,
20 us pulse, at nine drives g = 1/16 .. 16 times the node's default (0.5 x the stored saturation
amplitude). The node's amplitude factor must stay below 2, so levels above x3.8 scale the stored
saturation amplitude in that level's local state copy instead. A level whose drive would exceed
0.99 of full scale is skipped for that qubit (arbel: its stored amplitude is 1.0, so x2 is the
ceiling and is played at 0.99). Propose mode, local state copies, nothing written to any state.
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

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
backend = sys.argv[1]
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}[backend]
LEVELS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
if len(sys.argv) > 2:  # a subset, e.g. "0.0625,0.125,0.25,0.5,1"
    LEVELS = [float(v) for v in sys.argv[2].split(",")]
OUT = HERE / "satladder" / backend
OUT.mkdir(parents=True, exist_ok=True)
BASE_STATE = HERE / backend / "state"

os.environ.setdefault("MPLBACKEND", "Agg")
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import capture_job_timings, configure_iqcc  # noqa: E402

configure_iqcc(BASE_STATE, IqccConfig(enabled=True, backend=backend, default_timeout_s=60))
import qualibrate_ai  # noqa: E402

qualibrate_ai.configure(data_root=OUT / "store")
LIB = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam  # noqa: E402


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/sat] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free():
    while True:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        if not (pids - FROZEN):
            return
        log(f"backend busy ({sorted(pids - FROZEN)}); waiting 60 s")
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


def sat_path(state: dict, q: str):
    ops = state["qubits"][q]["xy"]["operations"]
    v = ops["saturation"]
    return v.split("/")[-1] if isinstance(v, str) else "saturation"


base = json.loads((BASE_STATE / "state.json").read_text())
stored = {q: float(base["qubits"][q]["xy"]["operations"][sat_path(base, q)]["amplitude"]) for q in QUBITS}
log(f"stored saturation amplitudes {stored}")

for g in LEVELS:
    factor = min(1.9, 0.5 * g)
    level_dir = OUT / f"state_x{g:g}"
    level_dir.mkdir(exist_ok=True)
    state = json.loads(json.dumps(base))
    group, played = [], {}
    for q in QUBITS:
        want = 0.5 * g * stored[q]
        if want > 0.99 * 1.0001 and not (g == 2 and want <= 1.0):
            log(f"x{g:g}: {q} skipped (would need {want:.2f} of full scale)")
            continue
        want = min(want, 0.99)
        new_amp = want / factor
        # the stored amplitude itself may sit at full scale (arbel: 1.0); only a raised one is capped
        if new_amp > max(0.99, stored[q] * 1.0001):
            log(f"x{g:g}: {q} skipped (stored amplitude would be {new_amp:.2f})")
            continue
        state["qubits"][q]["xy"]["operations"][sat_path(state, q)]["amplitude"] = new_amp
        group.append(q)
        played[q] = want
    if not group:
        continue
    (level_dir / "state.json").write_text(json.dumps(state))
    shutil.copy(BASE_STATE / "wiring.json", level_dir / "wiring.json")
    wait_until_free()
    tag = f"satladder-x{g:g}"
    node = LIB.nodes["03a_qubit_spectroscopy"].copy(name="03a_qubit_spectroscopy")
    node.bind_machine(Quam.load(str(level_dir)))
    # One qubit after another, as the agents run it: played together, one qubit's drive sweeping past a
    # neighbour's line (qolab's six sit within 230 MHz) could excite it and draw a line in its trace.
    params = dict(frequency_span_in_mhz=400.0, frequency_step_in_mhz=0.25, num_shots=150, operation_len_in_ns=20000,
                  operation_amplitude_factor=factor, multiplexed=False)
    log(f"{tag}: {group} factor {factor:g}, played amplitudes {played}")
    error = None
    t0 = time.perf_counter()
    with capture_job_timings() as timings:
        try:
            node.run(mode="propose", interactive=True, external=True, qubits=group, **params)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    qpu = sum(e.get("qpu_execution_s") or 0 for e in timings)
    stderr = [e.get("executor_stderr") for e in timings if e.get("executor_stderr")]
    ds = node.results.get("ds_raw")
    if ds is not None:
        try:
            ds.to_netcdf(OUT / f"{tag}.nc")
        except Exception as exc:  # noqa: BLE001
            log(f"netcdf failed: {exc}")
    rec = {"tag": tag, "g": g, "factor": factor, "qubits": group, "played_amplitude": played, "stored_amplitude": stored,
           "params": params, "error": error, "qpu_s": qpu, "wall_s": time.perf_counter() - t0, "executor_stderr": stderr,
           "fit_results": node.results.get("fit_results")}
    (OUT / f"{tag}.json").write_text(json.dumps(js(rec), indent=1))
    log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s" + (f", stderr {stderr}" if stderr else ""))
    for q, r in (node.results.get("fit_results") or {}).items():
        log(f"   {q}: f_01 {r.get('f_01')} detected {r.get('signal_detected')} fwhm {r.get('fitted_fwhm')}")
log("all levels done")
