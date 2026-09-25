"""T1 with a chirp as the pi pulse, against the calibrated x180, on every qubit of the chirp tests (25 Sep 2026).

usage: t1_chirp_test.py <backend> [--dry]

For each qubit, one job interleaves, at 20 log-spaced delays from 16 ns to 600 us:
  chirp   a 2 us linear up-chirp over 20 MHz (the chirp nodes' sweep without T1) centred on the f01 that 03a found with
          T1 missing (dirchk_latest, 13:32), at the drive 03a selected; then the delay, then the readout
  x180    the state's calibrated x180 at the stored frequency; then the delay, then the readout
and a ground reference (no pulse). Every shot waits 1 ms first (5 T1 up to T1 = 200 us), so nothing depends on
knowing T1. State: the cloud state pulled 15:26 (~/qab-runs/state-20260925-1526; gilboa with the C/D qubits
marked active). Outputs in t1_chirp/. Nothing is written to any state.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
backend = sys.argv[1]
DRY = "--dry" in sys.argv
QUBITS = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}[backend]
SNAP = Path.home() / "qab-runs/state-20260925-1526" / backend
OUT = HERE / "t1_chirp"
OUT.mkdir(exist_ok=True)
STATE = OUT / f"state_{backend}"
STATE.mkdir(exist_ok=True)
st = json.loads((SNAP / "state.json").read_text())
if backend == "gilboa":
    st["active_qubit_names"] = ["qC1", "qC2", "qC3", "qC4", "qC5", "qD1", "qD2", "qD3", "qD4", "qD5"]
(STATE / "state.json").write_text(json.dumps(st))
shutil.copy(SNAP / "wiring.json", STATE / "wiring.json")

BAND = 20_000_000
TAU = 2000                     # the chirp nodes' sweep when T1 is missing
RESET_NS = 1_000_000           # wait before every shot
SHOTS = 200
DELAYS_NS = sorted({int(4 * round(v / 4)) for v in np.geomspace(16, 600_000, 20)})


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/t1_chirp] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free():
    while not DRY:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        if not (pids - FROZEN):
            return
        log(f"backend busy ({sorted(pids - FROZEN)}); waiting 60 s")
        time.sleep(60)


def from_03a(q):
    """f01 and drive from the 13:32 03a run with T1 missing (what 03a would hand on at bring-up)."""
    for f in sorted((HERE / "dirchk_latest" / backend).glob("03a-noT1-*.json")):
        rec = json.loads(f.read_text())
        r = (rec.get("fit_results") or {}).get(q)
        if r and r.get("f_01") == r.get("f_01") and r.get("f_01") is not None:
            return float(r["f_01"]), r.get("selected_drive_amplitude"), r.get("line_identity")
    return None, None, None


def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=not DRY, backend=backend, default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, play, program, save, stream_processing, wait
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    from calibration_utils.qubit_spectroscopy_chirp.chirp import (OPERATION, chirp_rate_hz_per_ns, inject_chirp_operations,
                                                                  rabi_per_amplitude, target_rabi_hz)

    machine = Quam.load(str(STATE))
    config = machine.generate_config()
    rate = chirp_rate_hz_per_ns(BAND, TAU)
    plan = {}
    for name in QUBITS:
        q = machine.qubits[name]
        f01_03a, amp_03a, ident = from_03a(name)
        stored = float(q.f_01)
        centre = f01_03a if f01_03a is not None else stored
        kappa = rabi_per_amplitude(q)
        drive = float(amp_03a) if amp_03a is not None and amp_03a == amp_03a else min(0.99, 2 * target_rabi_hz(rate) / kappa)
        IF0 = int(q.xy.intermediate_frequency)
        IFc = int(round(IF0 + (centre - stored)))
        assert abs(IFc) + BAND / 2 < 400e6, (name, IFc)
        plan[name] = dict(el=q.xy.name, IF_x180=IF0, IF_chirp=IFc, centre=centre, stored_f01=stored, drive=drive,
                          from_03a=ident, stored_T1=float(q.T1) if q.T1 else None)
        log(f"{name}: chirp centre {(centre - stored) / 1e6:+.2f} MHz from the stored f01 (03a: {ident}), drive {drive:.4f}, "
            f"stored T1 {plan[name]['stored_T1']}")
    inject_chirp_operations(config, {plan[n]["el"]: (TAU, plan[n]["drive"]) for n in QUBITS})
    D = len(DELAYS_NS)
    cycles = [d // 4 for d in DELAYS_NS]
    per_shot = RESET_NS * 1e-9 + np.mean(DELAYS_NS) * 1e-9 + 8e-6
    est = SHOTS * (2 * D + 1) * per_shot
    log(f"delays {DELAYS_NS} ns; {SHOTS} rounds x {2 * D + 1} shots x {per_shot * 1e6:.0f} us = {est:.1f} s per qubit")
    assert est < 45

    records = {}
    import contextlib
    session = contextlib.nullcontext(None) if DRY else qm_session(machine.connect(), config, timeout=120)
    with session as qm:
        for name in QUBITS:
            q = machine.qubits[name]
            p = plan[name]
            with program() as prog:
                n = declare(int)
                d = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                streams = {k: (declare_stream(), declare_stream()) for k in ("chirp", "x180", "ground")}
                machine.initialize_qpu(target=q)
                align()
                with for_(n, 0, n < SHOTS, n + 1):
                    with for_each_(d, cycles):
                        for kind in ("chirp", "x180"):
                            wait(RESET_NS // 4, q.xy.name, q.resonator.name)
                            align()
                            if kind == "chirp":
                                q.xy.update_frequency(p["IF_chirp"] - BAND // 2)
                                play(OPERATION, p["el"], chirp=(rate, "Hz/nsec"))
                            else:
                                q.xy.update_frequency(p["IF_x180"])
                                q.xy.play("x180")
                            q.xy.wait(d)
                            align()
                            q.resonator.measure("readout", qua_vars=(I, Q))
                            save(I, streams[kind][0])
                            save(Q, streams[kind][1])
                            align()
                    wait(RESET_NS // 4, q.xy.name, q.resonator.name)
                    align()
                    q.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, streams["ground"][0])
                    save(Q, streams["ground"][1])
                    align()
                with stream_processing():
                    for k, (si, sq) in streams.items():
                        if k == "ground":
                            si.average().save(f"{k}_I")
                            sq.average().save(f"{k}_Q")
                        else:
                            si.buffer(D).average().save(f"{k}_I")
                            sq.buffer(D).average().save(f"{k}_Q")
            keys = [f"{k}_{c}" for k in ("chirp", "x180", "ground") for c in "IQ"]
            if DRY:
                from qm import generate_qua_script
                generate_qua_script(prog, config)
                log(f"{name}: program built")
                continue
            wait_until_free()
            t0 = time.perf_counter()
            job = qm.execute(prog)
            job.result_handles.wait_for_all_values()
            qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
            data = {}
            for k in keys:
                raw = job.result_handles.get(k).fetch_all()
                try:
                    raw = raw["value"]
                except (TypeError, IndexError, KeyError, ValueError):
                    pass
                data[k] = np.asarray(raw, dtype=float)
            np.savez(OUT / f"{backend}_{name}.npz", delays_ns=np.array(DELAYS_NS), **data)
            records[name] = dict(p, qpu_s=qpu, wall_s=time.perf_counter() - t0, shots=SHOTS, reset_ns=RESET_NS, tau_ns=TAU,
                                 band_hz=BAND, delays_ns=DELAYS_NS)
            log(f"{name}: QPU {qpu:.1f} s, wall {time.perf_counter() - t0:.0f} s")
    if not DRY:
        (OUT / f"{backend}_meta.json").write_text(json.dumps(records, indent=1))
    log("all done")


if __name__ == "__main__":
    main()
