"""qolab Q1, 03a chirp with T1 missing: a 2 us against a 4 us sweep, alternated twice (25 Sep 2026).

Worktree at ee24095 (up- and down-chirps, both band-centre orders), state pulled 13:49 (~/qab-runs/state-20260925-1349,
identical to 13:30) with Q1's T1 removed: 50 us wait between shots. Propose mode, nothing written to any state.
Outputs in q1_sweeps/.
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
OUT = HERE / "q1_sweeps"
STATE = OUT / "state_noT1"
STATE.mkdir(parents=True, exist_ok=True)
SNAP = Path.home() / "qab-runs/state-20260925-1349/qolab"
st = json.loads((SNAP / "state.json").read_text())
st["qubits"]["Q1"].pop("T1", None)
(STATE / "state.json").write_text(json.dumps(st))
shutil.copy(SNAP / "wiring.json", STATE / "wiring.json")

os.environ.setdefault("MPLBACKEND", "Agg")
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import capture_job_timings, configure_iqcc  # noqa: E402

configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
import qualibrate_ai  # noqa: E402

qualibrate_ai.configure(data_root=OUT / "store")
LIB = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam  # noqa: E402


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [qolab/q1_sweeps] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def busy():
    r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), "qolab"],
                       capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
    return [l for l in r.stdout.splitlines() if l.strip().startswith("busy: pid")]


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
    return x if isinstance(x, (str, bool)) or x is None else repr(x)


for rep in (1, 2):
    for tau in (2000, 4000):
        while busy():
            log("backend busy; waiting 60 s")
            time.sleep(60)
        tag = f"03a-Q1-{tau}ns-r{rep}"
        node = LIB.nodes["03a_qubit_spectroscopy_chirp"].copy(name="03a_qubit_spectroscopy_chirp")
        node.bind_machine(Quam.load(str(STATE)))
        log(f"{tag}: T1 {node.machine.qubits['Q1'].T1}, wait {node.machine.qubits['Q1'].thermalization_time} ns, sweep {tau} ns")
        error = None
        with capture_job_timings() as timings:
            try:
                node.run(mode="propose", interactive=True, external=True, qubits=["Q1"], sweep_length_ns=tau)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        qpu = sum(e.get("qpu_execution_s") or 0 for e in timings)
        ds = node.results.get("ds_raw")
        if ds is not None:
            ds.to_netcdf(OUT / f"{tag}.nc")
        r = (node.results.get("fit_results") or {}).get("Q1") or {}
        rec = {"tag": tag, "sweep_ns": tau, "rep": rep, "error": error, "qpu_s": qpu, "fit_results": node.results.get("fit_results"),
               "namespace": {"ladders": {k: {"played": v.played} for k, v in (node.namespace.get("ladders") or {}).items()},
                             "rates": node.namespace.get("rates"), "kappa_priors": node.namespace.get("kappa_priors")}}
        (OUT / f"{tag}.json").write_text(json.dumps(js(rec), indent=1))
        log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s; f01 shift {r.get('frequency_shift')} "
            f"+- {r.get('f_01_error')} [{r.get('line_identity')}] sweep/T1 {r.get('sweep_over_t1')}")
log("all done")
