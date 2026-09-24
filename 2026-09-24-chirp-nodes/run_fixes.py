"""Hardware test of the 03b timing fix and the 09a flux-wait fix (24 Sep 2026).

usage: run_fixes.py <backend> <tree: fix|main> <step>[,<step>...]

tree fix  = the qua-libs-chirp worktree (03b: // 4 restored, 80 us explicit, 5 us flux margins; 09a: // 4)
tree main = the untouched ~/code/QM/qua-libs tree (read only), the "before" runs
Propose mode on local copies of the 22 Sep snapshots; nothing is written to any state.

Steps:
  map03b   qolab Q1: 03b over +-94 mV (15 points), +-30 MHz at 0.5 MHz, 50 shots
  ab20     gilboa qD5: 03b at its defaults with a 20 us drive
  ab80     gilboa qD5: 03b at its defaults with an 80 us drive
  r09a     qolab Q1: 09a with 16..1000 ns idle in 8 ns steps, 11 flux points, 50 shots
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

backend, tree, steps = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ROOT = {"fix": Path.home() / "code/QM/qua-libs-chirp", "main": Path.home() / "code/QM/qua-libs"}[tree]
WT = ROOT / "qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "fixes" / backend
OUT.mkdir(parents=True, exist_ok=True)
SRC = {"qolab": Path.home() / "qab-runs/reference-state-20260922/qolab",
       "gilboa": Path("/private/tmp/claude-734394052/-Users-atosato-code-QM-qua-agents-benchmark/"
                      "50ae3071-f59a-4af0-b332-5646b1ad4122/scratchpad/gilboa-cd-active")}[backend]
STATE = OUT / "state"
if not STATE.exists():
    STATE.mkdir()
    for f in ("state.json", "wiring.json"):
        shutil.copy(SRC / f, STATE / f)
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}

os.environ.setdefault("MPLBACKEND", "Agg")
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import capture_job_timings, configure_iqcc  # noqa: E402

configure_iqcc(STATE, IqccConfig(enabled=True, backend=backend, default_timeout_s=60))
import qualibrate_ai  # noqa: E402

qualibrate_ai.configure(data_root=OUT / "store")
LIB = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam  # noqa: E402

import calibration_utils  # noqa: E402

assert str(WT) in calibration_utils.__file__, calibration_utils.__file__


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/{tree}] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free():
    while True:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        if not (pids - FROZEN[backend]):
            return
        log(f"backend busy ({sorted(pids - FROZEN[backend])}); waiting 60 s")
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


def run(tag, node_name, qubits, **params):
    wait_until_free()
    node = LIB.nodes[node_name].copy(name=node_name)
    node.bind_machine(Quam.load())
    tag = f"{tag}-{tree}"
    log(f"{tag}: {node_name} on {qubits} {params}")
    error = None
    t0 = time.perf_counter()
    with capture_job_timings() as timings:
        try:
            node.run(mode="propose", interactive=True, external=True, qubits=qubits, **params)
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
    for label, fig in (node.results.get("figures") or {}).items():
        try:
            fig.savefig(OUT / f"{tag}_{label}.png", dpi=110, bbox_inches="tight")
        except Exception:  # noqa: BLE001
            pass
    rec = {"tag": tag, "node": node_name, "qubits": qubits, "params": params, "error": error, "qpu_s": qpu,
           "wall_s": time.perf_counter() - t0, "executor_stderr": stderr, "fit_results": node.results.get("fit_results"),
           "proposed": node.proposed_state_updates}
    (OUT / f"{tag}.json").write_text(json.dumps(js(rec), indent=1))
    log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s" + (f", stderr {stderr}" if stderr else ""))
    for q, r in (node.results.get("fit_results") or {}).items():
        keep = {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) and k != "failure_code"}
        log(f"   {q}: {js(keep)}")
        if r.get("warnings"):
            log(f"   {q} warnings: {r['warnings']}")


for step in steps:
    if step == "map03b":
        run("map03b-Q1", "03b_qubit_spectroscopy_vs_flux", ["Q1"], min_flux_offset_in_v=-0.094, max_flux_offset_in_v=0.094,
            num_flux_points=15, frequency_span_in_mhz=60.0, frequency_step_in_mhz=0.5, num_shots=50)
    elif step == "ab20":
        run("ab20-qD5", "03b_qubit_spectroscopy_vs_flux", ["qD5"], operation_len_in_ns=20000)
    elif step == "ab80":
        run("ab80-qD5", "03b_qubit_spectroscopy_vs_flux", ["qD5"], operation_len_in_ns=80000)
    elif step == "r09a":
        run("r09a-Q1", "09a_ramsey_vs_flux_calibration", ["Q1"], min_wait_time_in_ns=16, max_wait_time_in_ns=1000,
            wait_time_step_in_ns=8, flux_num=11, num_shots=50)
    else:
        raise SystemExit(f"unknown step {step}")
log("all steps done: " + ",".join(steps))
