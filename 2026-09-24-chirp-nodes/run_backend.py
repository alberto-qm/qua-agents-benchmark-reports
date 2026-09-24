"""Hardware test of the chirp spectroscopy nodes on one IQCC backend (24 Sep 2026).

usage: run_backend.py <backend> <step>[,<step>...]

Runs the new nodes from the qua-libs-chirp worktree through qualibrate_ai in propose mode
on a local copy of the 22 Sep reference state (gilboa: the copy with the C/D qubits
active). Nothing is written to any state. Every node run saves its processed dataset
(netcdf), fit results (json), figures (png) and QPU time under <backend>/.

Steps:
  r0      03a chirp + 03b chirp at their defaults (the live runs)
  wide3a  03a chirp with 11 drive levels over +-200 MHz (data for the 03a replays)
  wide3b  03b chirp with 35 flux columns over 2.5x the 20 MHz offset, f01 -50..+40 MHz
  sat     03b in saturation mode (independent sweet-spot reference)
  fine    old 03a: weak saturation, +-5 MHz at 0.1 MHz around the chirp's f_01 (independent f_01 reference)
  end     03a chirp + 03b chirp at their defaults again (drift / repeatability)
  smoke   03a chirp + 03b chirp on the first qubit only
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
REF = Path.home() / "qab-runs/reference-state-20260922"
STATES = {
    "arbel": REF / "arbel",
    "qolab": REF / "qolab",
    "gilboa": Path("/private/tmp/claude-734394052/-Users-atosato-code-QM-qua-agents-benchmark/"
                   "50ae3071-f59a-4af0-b332-5646b1ad4122/scratchpad/gilboa-cd-active"),
}
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
# Our own qua-agents cells, frozen with SIGSTOP since 22 Sep: they submit nothing.
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}
# Qubits per job, so each job's pre-flight estimate stays under the 60 s cap.
GROUPS = {
    "r0-03a": {"arbel": [["qB4", "qA5", "qD1"]], "qolab": [["Q1", "Q2", "Q5"]], "gilboa": [["qD2", "qC3", "qD5"]]},
    "r0-03b": {"arbel": [["qB4", "qA5", "qD1"]], "qolab": [["Q2"], ["Q1", "Q5"]], "gilboa": [["qD2", "qC3", "qD5"]]},
    "wide3a": {"arbel": [["qB4", "qA5", "qD1"]], "qolab": [["Q1"], ["Q2"], ["Q5"]], "gilboa": [["qD2", "qC3", "qD5"]]},
    "wide3b": {"arbel": [["qB4"], ["qA5", "qD1"]], "qolab": [["Q1"], ["Q5"], ["Q2", "lo"], ["Q2", "hi"]],
               "gilboa": [["qD5"], ["qD2", "qC3"]]},
    "sat": {"arbel": [["qB4", "qA5", "qD1"]], "qolab": [["Q2"], ["Q1"], ["Q5"]], "gilboa": [["qD2", "qC3", "qD5"]]},
}

backend = sys.argv[1]
steps = sys.argv[2].split(",")
OUT = HERE / backend
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "state"
if not STATE.exists():
    STATE.mkdir()
    for f in ("state.json", "wiring.json"):
        shutil.copy(STATES[backend] / f, STATE / f)

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


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [{backend}] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free() -> None:
    while True:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        others = pids - FROZEN[backend]
        if not others:
            return
        log(f"backend busy (pids {sorted(others)}); waiting 60 s")
        time.sleep(60)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (str, bool)) or x is None:
        return x
    return repr(x)


def run(tag: str, node_name: str, qubits: list[str], **params) -> dict:
    wait_until_free()
    node = LIB.nodes[node_name].copy(name=node_name)
    node.bind_machine(Quam.load())
    log(f"{tag}: {node_name} on {qubits} {params}")
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
    record = {"tag": tag, "node": node_name, "qubits": qubits, "params": params, "error": error, "wall_s": wall,
              "qpu_s": qpu, "executor_stderr": stderr, "fit_results": node.results.get("fit_results"),
              "proposed": node.proposed_state_updates, "measurement_id": node.measurement_id,
              "namespace": {k: node.namespace.get(k) for k in ("sweep_lengths", "rates", "drive_amplitudes",
                                                                 "flux_offsets", "kappa_priors")}}
    ladders = node.namespace.get("ladders")
    if ladders:
        record["namespace"]["ladders"] = {k: {"played": v.played, "requested": v.requested, "prior": v.prior,
                                              "centre_amplitude": v.centre_amplitude} for k, v in ladders.items()}
    (OUT / f"{tag}.json").write_text(json.dumps(_jsonable(record), indent=1))
    with open(OUT / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(_jsonable({k: record[k] for k in ("tag", "node", "qubits", "error", "wall_s", "qpu_s")})) + "\n")
    log(f"{tag}: {'FAILED ' + error if error else 'done'} -- QPU {qpu:.1f} s, wall {wall:.0f} s"
        + (f", stderr: {' | '.join(s.strip()[:200] for s in stderr)}" if stderr else ""))
    for q, r in (node.results.get("fit_results") or {}).items():
        keys = ("f_01", "frequency_shift", "f_01_error", "line_identity", "drive_scale", "anharmonicity_estimate",
                "selected_drive_amplitude", "idle_offset_shift", "idle_offset_shift_error", "curvature",
                "column_precision", "tracked_columns", "chi2_red")
        brief = {k: r.get(k) for k in keys if k in r}
        log(f"   {q}: {_jsonable(brief)} warnings={r.get('warnings')}")
    return record


def groups(step: str):
    return GROUPS[step][backend]


def wide3b_params(q: str, half: str | None) -> dict:
    m = Quam.load()
    quad = abs(float(m.qubits[q].freq_vs_flux_01_quad_term))
    span = 2.5 * np.sqrt(20e6 / quad)
    xs = np.linspace(-span, span, 35)
    base = dict(frequency_min_in_mhz=-50.0, frequency_max_in_mhz=40.0)
    if half == "lo":
        return dict(base, min_flux_offset_in_v=float(xs[0]), max_flux_offset_in_v=float(xs[17]), num_flux_points=18)
    if half == "hi":
        return dict(base, min_flux_offset_in_v=float(xs[18]), max_flux_offset_in_v=float(xs[-1]), num_flux_points=17)
    return dict(base, flux_span_factor=2.5, num_flux_points=35)


for step in steps:
    if step in ("r0", "end"):
        for g in groups("r0-03a"):
            run(f"{step}-03a-{'-'.join(g)}", "03a_qubit_spectroscopy_chirp", g)
        for g in groups("r0-03b"):
            run(f"{step}-03b-{'-'.join(g)}", "03b_qubit_spectroscopy_vs_flux_chirp", g)
    elif step == "smoke":
        q = QUBITS[backend][0]
        run(f"smoke-03a-{q}", "03a_qubit_spectroscopy_chirp", [q])
        run(f"smoke-03b-{q}", "03b_qubit_spectroscopy_vs_flux_chirp", [q])
    elif step == "wide3a":
        for g in groups("wide3a"):
            run(f"wide3a-{'-'.join(g)}", "03a_qubit_spectroscopy_chirp", g, num_drive_levels=11,
                frequency_span_in_mhz=400.0)
    elif step == "wide3b":
        for g in groups("wide3b"):
            half = g[-1] if g[-1] in ("lo", "hi") else None
            qs = [x for x in g if x not in ("lo", "hi")]
            params = wide3b_params(qs[0], half) if half else dict(frequency_min_in_mhz=-50.0,
                                                                   frequency_max_in_mhz=40.0,
                                                                   flux_span_factor=2.5, num_flux_points=35)
            run(f"wide3b-{'-'.join(g)}", "03b_qubit_spectroscopy_vs_flux_chirp", qs, **params)
    elif step == "sat":
        for g in groups("sat"):
            run(f"sat-{'-'.join(g)}", "03b_qubit_spectroscopy_vs_flux_chirp", g, pulse="saturation",
                num_flux_points=15, frequency_min_in_mhz=-25.0, frequency_max_in_mhz=10.0, num_shots=50)
    elif step == "fine":
        m = Quam.load()
        ref = {}
        for f in sorted(OUT.glob("r0-03a-*.json")):
            ref.update(json.loads(f.read_text())["fit_results"] or {})
        for q in QUBITS[backend]:
            qb = m.qubits[q]
            off = ref.get(q, {}).get("frequency_shift")
            if off is None or not np.isfinite(off):
                log(f"fine: no chirp f_01 for {q}; skipped")
                continue
            kappa = 1.0 / (float(qb.xy.operations["x180"].amplitude) * float(qb.xy.operations["x180"].length) * 1e-9)
            factor = (0.3e6 / kappa) / float(qb.xy.operations["saturation"].amplitude)
            run(f"fine-{q}", "03a_qubit_spectroscopy", [q], frequency_center_in_mhz=round(off / 1e6, 2),
                frequency_span_in_mhz=10.0, frequency_step_in_mhz=0.1, operation_amplitude_factor=float(min(factor, 1.9)),
                operation_len_in_ns=20000)
    else:
        raise SystemExit(f"unknown step {step}")
log("all steps done: " + ",".join(steps))
