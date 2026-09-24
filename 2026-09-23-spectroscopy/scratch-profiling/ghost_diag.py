"""Why did the carry-over test show no carry-over? (qolab Q1)

Part 1, timing: is `duration=20_000 * u.ns` a 20 us or an 80 us pulse? (u.ns == 1 and QuAM's
play() takes clock cycles.) Per-shot QPU time of readout-only, +saturation 5000 cycles,
+saturation 20000 cycles, +wait 5000, +wait 20000.

Part 2, populations (all from a 5xT1 reset, pulses in explicit clock cycles, 20 us = 5000):
  ref0 / ref1           readout / x180+readout              -> the |0>-|1> axis
  T1_20                 x180, idle 20 us, readout            -> plain T1 decay over 20 us
  x180|idle|B           x180, readout A, idle 20 us, readout B    vs T1_20: does readout A disturb?
  x180|probe|B          x180, readout A, 20 us detuned drive, B   vs idle: does the probe drive disturb?
  sat_on|idle|B         20 us resonant saturation, A, idle, B
  sat_on|probe|B        20 us resonant saturation, A, probe, B    (the original test, k=0)
  sat_off|probe|B       20 us detuned saturation, A, probe, B     (control)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
N_T = 20_000
N_P = 5_000
SAT = 5000  # clock cycles = 20 us
DETUNE = 10_000_000


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    machine = Quam.load()
    q = machine.qubits["Q1"]
    IF = int(q.xy.intermediate_frequency)
    det = -DETUNE if IF > 0 else DETUNE

    def values(job, name):
        raw = job.result_handles.get(name).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        return np.asarray(raw, dtype=float).ravel()

    # ---- part 1: timing
    def timing(extra):
        with program() as prog:
            n = declare(int); I = declare(fixed); Q = declare(fixed); s = declare_stream()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N_T, n + 1):
                extra()
                align()
                q.resonator.measure("readout", qua_vars=(I, Q))
                save(I, s)
                align()
            with stream_processing():
                s.average().save("I")
        return prog

    far = lambda: q.xy.update_frequency(IF + det)  # noqa: E731  # keep the timing pulses off resonance
    TIMING = [
        ("readout only", lambda: None),
        ("+ saturation duration=5000", lambda: (far(), q.xy.play("saturation", amplitude_scale=0.5, duration=5000))),
        ("+ saturation duration=20000", lambda: (far(), q.xy.play("saturation", amplitude_scale=0.5, duration=20000))),
        ("+ wait 5000 cycles", lambda: q.wait(5000)),
        ("+ wait 20000 cycles", lambda: q.wait(20000)),
    ]

    # ---- part 2: populations
    def seq(name, st, I, Q):
        def ro(tag):
            q.resonator.measure("readout", qua_vars=(I, Q)); save(I, st[f"{name}_I{tag}"]); save(Q, st[f"{name}_Q{tag}"]); align()

        def sat(freq):
            q.xy.update_frequency(freq); q.xy.play("saturation", amplitude_scale=0.5, duration=SAT); align()

        q.reset_qubit_thermal(); align()
        q.xy.update_frequency(IF)
        if name == "ref0":
            ro("A"); return
        if name == "ref1":
            q.xy.play("x180"); align(); ro("A"); return
        if name == "T1_20":
            q.xy.play("x180"); align(); q.wait(SAT); align(); ro("A"); return
        pump, after = name.split("|")[0], name.split("|")[1]
        if pump == "x180":
            q.xy.play("x180"); align()
        elif pump == "sat_on":
            sat(IF)
        else:
            sat(IF + det)
        ro("A")
        if after == "idle":
            q.wait(SAT); align()
        else:
            sat(IF + det)
        ro("B")

    SEQS = ["ref0", "ref1", "T1_20", "x180|idle|B", "x180|probe|B", "sat_on|idle|B", "sat_on|probe|B", "sat_off|probe|B"]

    qmm = machine.connect()
    with qm_session(qmm, machine.generate_config(), timeout=60) as qm:
        print("part 1 -- per-shot QPU time", flush=True)
        base = None
        for label, extra in TIMING:
            job = qm.execute(timing(extra)); job.result_handles.wait_for_all_values()
            per = 1e6 * float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all()) / N_T
            base = per if base is None else base
            print(f"  {label:<30} {per:8.2f} us/shot   (+{per - base:6.2f})", flush=True)

        with program() as prog:
            n = declare(int); I = declare(fixed); Q = declare(fixed)
            st = {}
            for name in SEQS:
                for tag in (("A", "B") if "|" in name else ("A",)):
                    st[f"{name}_I{tag}"] = declare_stream(); st[f"{name}_Q{tag}"] = declare_stream()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N_P, n + 1):
                for name in SEQS:
                    seq(name, st, I, Q)
            with stream_processing():
                for key, s in st.items():
                    s.save_all(key)
        job = qm.execute(prog); job.result_handles.wait_for_all_values()
        qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())

        def iq(name, tag):
            try:
                return values(job, f"{name}_I{tag}"), values(job, f"{name}_Q{tag}")
            except Exception:  # noqa: BLE001  # stream never written (single-readout sequences have no B)
                return None

    z0 = np.array([m.mean() for m in iq("ref0", "A")]); z1 = np.array([m.mean() for m in iq("ref1", "A")]); ax = z1 - z0

    def pop(name, tag):
        r = iq(name, tag)
        if r is None or r[0].size == 0:
            return None
        p = ((r[0] - z0[0]) * ax[0] + (r[1] - z0[1]) * ax[1]) / (ax @ ax)
        return p.mean(), p.std() / np.sqrt(p.size)

    print(f"\npart 2 -- excited population, in units of the ref1 - ref0 separation ({qpu:.1f} s QPU)")
    for name in SEQS:
        a, b = pop(name, "A"), pop(name, "B")
        line = f"  {name:<18} A {a[0]:+.3f} +/- {a[1]:.3f}"
        if b is not None:
            line += f"    B {b[0]:+.3f} +/- {b[1]:.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
