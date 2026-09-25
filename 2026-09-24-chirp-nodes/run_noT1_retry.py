"""The chirp nodes with T1 missing from the state, against the same state with T1 (25 Sep 2026).

usage: run_noT1_retry.py <backend> <node>:<variant>:<q1,q2,...> [...]

Re-runs jobs of run_noT1.py that failed on a cloud error (25 Sep 09:07-09:08: 503 and read timeouts).

Without T1 the nodes fall back to a 4 us sweep, QuAM to a 50 us wait between shots (5 x 10 us),
03b to a 1 us flux tail. This runs 03a chirp and 03b chirp at their defaults (commit 80c3db3,
up-and-down sweeps) on every qubit measured on 24 Sep, once on a state copy with T1 removed
for those qubits and once on the unchanged copy, interleaved so drift hits both. Propose mode,
nothing written to any state. Outputs in noT1/<backend>/.
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
QUBITS = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}[backend]
# Same qubits per job as the live runs (each job's pre-flight estimate stays under the 60 s cap).
GROUPS_03A = {"arbel": [["qB4", "qA5", "qD1"], ["qA6"]], "qolab": [["Q1", "Q2", "Q5"]],
              "gilboa": [["qD2", "qC3", "qD5"]]}[backend]
GROUPS_03B = {"arbel": [["qB4", "qA5", "qD1"], ["qA6"]], "qolab": [["Q2"], ["Q1", "Q5"]],
              "gilboa": [["qD2", "qC3", "qD5"]]}[backend]
VARIANTS = ("noT1", "T1")
OUT = HERE / "noT1" / backend
OUT.mkdir(parents=True, exist_ok=True)
BASE_STATE = HERE / backend / "state"

base = json.loads((BASE_STATE / "state.json").read_text())
for v in VARIANTS:
    d = OUT / f"state_{v}"
    d.mkdir(exist_ok=True)
    st = json.loads(json.dumps(base))
    if v == "noT1":
        for q in QUBITS:
            st["qubits"][q].pop("T1", None)
    (d / "state.json").write_text(json.dumps(st))
    shutil.copy(BASE_STATE / "wiring.json", d / "wiring.json")

os.environ.setdefault("MPLBACKEND", "Agg")
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import capture_job_timings, configure_iqcc  # noqa: E402

configure_iqcc(BASE_STATE, IqccConfig(enabled=True, backend=backend, default_timeout_s=60))
import qualibrate_ai  # noqa: E402

qualibrate_ai.configure(data_root=OUT / "store")
LIB = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam  # noqa: E402

import calibration_utils  # noqa: E402

assert str(WT) in calibration_utils.__file__, calibration_utils.__file__


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [{backend}/noT1] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free() -> None:
    while True:
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


def run(tag: str, node_name: str, qubits: list[str], variant: str, **params) -> None:
    wait_until_free()
    node = LIB.nodes[node_name].copy(name=node_name)
    node.bind_machine(Quam.load(str(OUT / f"state_{variant}")))
    waits = {q: node.machine.qubits[q].thermalization_time for q in qubits}
    t1s = {q: node.machine.qubits[q].T1 for q in qubits}
    log(f"{tag}: {node_name} on {qubits}, T1 {t1s}, waits {waits} ns {params}")
    error = None
    t0 = time.perf_counter()
    with capture_job_timings() as timings:
        try:
            node.run(mode="propose", interactive=True, external=True, qubits=qubits, **params)
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
            log(f"{tag}: netcdf failed ({exc}); saving I/Q as npz")
            np.savez(OUT / f"{tag}.npz", **{k: np.asarray(ds[k].values) for k in ds.variables})
    for label, fig in (node.results.get("figures") or {}).items():
        try:
            fig.savefig(OUT / f"{tag}_{label}.png", dpi=110, bbox_inches="tight")
        except Exception:  # noqa: BLE001
            pass
    record = {"tag": tag, "node": node_name, "qubits": qubits, "variant": variant, "waits_ns": waits, "T1": t1s, "params": params,
              "error": error, "wall_s": wall, "qpu_s": qpu, "executor_stderr": stderr,
              "fit_results": node.results.get("fit_results"), "proposed": node.proposed_state_updates,
              "namespace": {k: node.namespace.get(k) for k in ("sweep_lengths", "rates", "drive_amplitudes",
                                                                 "flux_offsets", "kappa_priors", "shot_rounds",
                                                                 "flux_step_lengths", "drive_lengths")}}
    ladders = node.namespace.get("ladders")
    if ladders:
        record["namespace"]["ladders"] = {k: {"played": v.played, "prior": v.prior} for k, v in ladders.items()}
    (OUT / f"{tag}.json").write_text(json.dumps(js(record), indent=1))
    with open(OUT / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(js({k: record[k] for k in ("tag", "node", "qubits", "variant", "error", "wall_s", "qpu_s")}))
                 + "\n")
    log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s, wall {wall:.0f} s"
        + (f", stderr: {' | '.join(s.strip()[:200] for s in stderr)}" if stderr else ""))
    for q, r in (node.results.get("fit_results") or {}).items():
        keys = ("f_01", "frequency_shift", "f_01_error", "line_identity", "drive_scale", "box_height", "box_snr",
                "idle_offset_shift", "idle_offset_shift_error", "column_precision", "tracked_columns", "chi2_red")
        log(f"   {q}: {js({k: r.get(k) for k in keys if k in r})} warnings={r.get('warnings')}")


# qA6's drive IF is -342 MHz (see run_noT1.py).
PARAMS_03A = {"qA6": dict(frequency_center_in_mhz=40.0, frequency_span_in_mhz=160.0)}
NODES = {"03a": "03a_qubit_spectroscopy_chirp", "03b": "03b_qubit_spectroscopy_vs_flux_chirp"}
for job in sys.argv[2:]:
    node, variant, qs = job.split(":")
    g = qs.split(",")
    params = PARAMS_03A.get(g[0], {}) if node == "03a" and len(g) == 1 else {}
    run(f"{node}-{variant}-{'-'.join(g)}", NODES[node], g, variant, **params)
log("retries done")
