"""Where does a calibration shot's time actually go?

The night-4 runs measure ~272 us per shot in qubit_spectroscopy_vs_flux, of which the
5xT1 thermalization wait explains only ~155-306 us and the pulses ~21 us. Two
independent qubits (gilboa qD2, arbel qB4) leave ~65-85 us unaccounted, while
job_preflight assumes 12 us of loop/stream overhead.

This is a ladder: each program adds ONE element of the node's inner loop to the one
before it, so the difference between consecutive rows is that element's per-shot cost.
Timing only -- nothing is fitted and no state is written.

Run:  PYTHONPATH=<superconducting> tinycal-venv-python profile_shot_overhead.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
QUBIT = "Q1"
N_SHOTS = 20_000
TIMEOUT_S = 60  # the new IQCC per-job limit


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=TIMEOUT_S))

    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing, wait
    from qualang_tools.multi_user import qm_session
    from qualang_tools.units import unit
    from quam_config import Quam

    u = unit(coerce_to_integer=True)
    machine = Quam.load()
    qubit = machine.qubits[QUBIT]
    print(f"qolab {QUBIT}: T1 {1e6 * float(qubit.T1):.1f} us, "
          f"thermalization {float(qubit.thermalization_time_factor) * 1e6 * float(qubit.T1):.0f} us, "
          f"saturation {qubit.xy.operations['saturation'].length} ns, "
          f"readout {qubit.resonator.operations['readout'].length} ns", flush=True)

    # Each step is (label, what this step adds, flags) -- flags accumulate down the ladder.
    STEPS = [
        ("1 bare loop",      "for_ + save(n)",            dict()),
        ("2 + readout",      "measure + save(I,Q)",       dict(readout=True)),
        ("3 + aligns",       "3x align() per shot",       dict(readout=True, aligns=True)),
        ("4 + update_freq",  "real-time frequency write", dict(readout=True, aligns=True, freq=True)),
        ("5 + saturation",   "20 us drive pulse",         dict(readout=True, aligns=True, freq=True, drive=True)),
        ("6 + thermal reset", "5xT1 wait",                dict(readout=True, aligns=True, freq=True, drive=True, reset=True)),
    ]

    def build(*, readout=False, aligns=False, freq=False, drive=False, reset=False):
        with program() as prog:
            n = declare(int)
            df = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n_st = declare_stream()
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < N_SHOTS, n + 1):
                save(n, n_st)
                if reset:
                    qubit.reset_qubit_thermal()
                if freq:
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                if aligns:
                    align()
                if drive:
                    qubit.xy.play("saturation", amplitude_scale=0.5, duration=20_000 * u.ns)
                if aligns:
                    align()
                if readout:
                    qubit.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, I_st)
                    save(Q, Q_st)
                else:
                    wait(4)
                if aligns:
                    align()
            with stream_processing():
                n_st.save("n")
                if readout:
                    I_st.buffer(1).average().save("I1")
                    Q_st.buffer(1).average().save("Q1")
        return prog

    qmm = machine.connect()
    config = machine.generate_config()
    rows = []
    with qm_session(qmm, config, timeout=TIMEOUT_S) as qm:
        for label, adds, flags in STEPS:
            try:
                started = time.perf_counter()
                job = qm.execute(build(**flags))
                job.result_handles.wait_for_all_values()
                wall = time.perf_counter() - started
                qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<20} FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                rows.append((label, adds, None, None))
                continue
            per_shot_us = 1e6 * qpu / N_SHOTS
            rows.append((label, adds, qpu, per_shot_us))
            print(f"{label:<20} {qpu:>7.2f}s QPU  {per_shot_us:>8.2f} us/shot  (wall {wall:.1f}s)", flush=True)

    print(f"\n{'step':<20}{'adds':<28}{'us/shot':>10}{'delta':>10}")
    prev = 0.0
    for label, adds, qpu, per in rows:
        if per is None:
            print(f"{label:<20}{adds:<28}{'failed':>10}")
            continue
        print(f"{label:<20}{adds:<28}{per:>10.2f}{per - prev:>10.2f}")
        prev = per
    return 0


if __name__ == "__main__":
    sys.exit(main())
