"""Does a short wait leave a copy of an on-resonance pixel in the next pixel? (qolab Q1)

Each pair of shots starts from a full 5xT1 thermal reset:
  pump:  20 us saturation, ON resonance (an arc pixel) or 10 MHz detuned (control); readout A
  wait:  k x T1
  probe: 20 us saturation, 10 MHz detuned (the next flux column, off the arc); readout B
On and off pumps alternate within one job, so drifts cancel.

Carry-over fraction, projected on the |0>-|1> axis given by the pump contrast:
  g(k) = (B_on - B_off) . d / |d|^2,   d = A_on - A_off
Prediction: g = exp(-(k*T1 + gap)/T1), gap ~ 20 us probe drive + readout.
Timing only on the QPU side; nothing written to the state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
QUBIT = "Q1"
N = 20_000
KS = (0, 0.5, 1, 2, 3, 5)
DETUNE_HZ = 10_000_000
OUT = Path(__file__).with_suffix(".json")


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from qualang_tools.units import unit
    from quam_config import Quam

    u = unit(coerce_to_integer=True)
    machine = Quam.load()
    q = machine.qubits[QUBIT]
    t1_us = float(q.T1) * 1e6
    IF = int(q.xy.intermediate_frequency)
    det = -DETUNE_HZ if IF > 0 else DETUNE_HZ  # toward zero IF, so the detuned drive stays in range
    print(f"{QUBIT}: T1 {t1_us:.1f} us, xy IF {IF / 1e6:.2f} MHz, off-resonance drive at IF {det / 1e6:+.0f} MHz", flush=True)

    names = ("IA_on", "QA_on", "IB_on", "QB_on", "IA_off", "QA_off", "IB_off", "QB_off")

    def build(k: float):
        wait_cycles = int(round(k * t1_us * 1000 / 4))
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            st = {name: declare_stream() for name in names}
            machine.initialize_qpu(target=q)
            align()
            with for_(n, 0, n < N, n + 1):
                for tag, pump_if in (("on", IF), ("off", IF + det)):
                    q.reset_qubit_thermal()
                    align()
                    q.xy.update_frequency(pump_if)
                    q.xy.play("saturation", amplitude_scale=0.5, duration=20_000 * u.ns)
                    align()
                    q.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, st[f"IA_{tag}"])
                    save(Q, st[f"QA_{tag}"])
                    align()
                    if wait_cycles >= 4:
                        q.wait(wait_cycles)
                        align()
                    q.xy.update_frequency(IF + det)
                    q.xy.play("saturation", amplitude_scale=0.5, duration=20_000 * u.ns)
                    align()
                    q.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, st[f"IB_{tag}"])
                    save(Q, st[f"QB_{tag}"])
                    align()
            with stream_processing():
                for name in names:
                    st[name].save_all(name)
        return prog

    def values(job, name):
        raw = job.result_handles.get(name).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        return np.asarray(raw, dtype=float).ravel()

    rows = []
    qmm = machine.connect()
    with qm_session(qmm, machine.generate_config(), timeout=60) as qm:
        for k in KS:
            try:
                job = qm.execute(build(k))
                job.result_handles.wait_for_all_values()
                qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
                v = {name: values(job, name) for name in names}
            except Exception as exc:  # noqa: BLE001
                print(f"k={k}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                continue
            d = np.array([v["IA_on"].mean() - v["IA_off"].mean(), v["QA_on"].mean() - v["QA_off"].mean()])
            proj = lambda i, qq: (i * d[0] + qq * d[1]) / (d @ d)  # noqa: E731  # in units of the pump contrast
            b_on, b_off = proj(v["IB_on"], v["QB_on"]), proj(v["IB_off"], v["QB_off"])
            g = b_on.mean() - b_off.mean()
            err = np.sqrt(b_on.var() / b_on.size + b_off.var() / b_off.size)
            pred = float(np.exp(-(k * t1_us + 20.0) / t1_us))
            rows.append(dict(k=k, g=float(g), err=float(err), pred=pred, qpu_s=qpu, n=int(b_on.size),
                             pump_contrast=float(np.hypot(*d))))
            print(f"k={k:<4} wait {k * t1_us:6.1f} us   carry-over {100 * g:6.2f} % +/- {100 * err:4.2f}   "
                  f"predicted {100 * pred:6.2f} %   ({qpu:.1f} s QPU)", flush=True)

    OUT.write_text(json.dumps(dict(qubit=QUBIT, t1_us=t1_us, detune_hz=det, n=N, rows=rows), indent=1))
    ok = [r for r in rows if r["g"] > 3 * r["err"]]
    if len(ok) >= 3:
        k_arr = np.array([r["k"] for r in ok]); lg = np.log([r["g"] for r in ok])
        w = np.array([(r["g"] / r["err"]) ** 2 for r in ok])
        slope, icpt = np.polyfit(k_arr, lg, 1, w=np.sqrt(w))
        t1_eff = t1_us / -slope
        print(f"\nfit over k with >3 sigma: g = exp({slope:.3f} k {icpt:+.3f})  ->  effective T1 {t1_eff:.1f} us "
              f"(state {t1_us:.1f}),  gap {-icpt * t1_eff:.1f} us (model 20)")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
