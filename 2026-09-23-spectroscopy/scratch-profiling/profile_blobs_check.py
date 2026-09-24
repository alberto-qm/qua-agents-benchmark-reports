"""Are qolab Q1's stored readout thresholds still valid? Mean/spread of I for |0> and x180|1>."""
import sys
from pathlib import Path
import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
N = 5_000

def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc
    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam
    machine = Quam.load(); qubit = machine.qubits["Q1"]
    ro = qubit.resonator.operations["readout"]
    with program() as prog:
        n = declare(int); I = declare(fixed); Q = declare(fixed)
        Ig = declare_stream(); Ie = declare_stream()
        machine.initialize_qpu(target=qubit); align()
        with for_(n, 0, n < N, n + 1):
            qubit.reset_qubit_thermal(); align()
            qubit.resonator.measure("readout", qua_vars=(I, Q)); save(I, Ig)
            qubit.reset_qubit_thermal(); align()
            qubit.xy.play("x180"); align()
            qubit.resonator.measure("readout", qua_vars=(I, Q)); save(I, Ie)
        with stream_processing():
            Ig.save_all("Ig"); Ie.save_all("Ie")
    qmm = machine.connect()
    with qm_session(qmm, machine.generate_config(), timeout=60) as qm:
        job = qm.execute(prog); job.result_handles.wait_for_all_values()
        def values(name):
            raw = job.result_handles.get(name).fetch_all()
            try:
                raw = raw["value"]
            except (TypeError, IndexError, KeyError, ValueError):
                pass
            return np.asarray(raw, dtype=float).ravel()
        ig, ie = values("Ig"), values("Ie")
    thr, rus = float(ro.threshold), float(ro.rus_exit_threshold)
    print(f"stored threshold      {thr:+.6f}   rus_exit_threshold {rus:+.6f}")
    print(f"|0>  I mean {ig.mean():+.6f}  std {ig.std():.6f}   P(I > threshold) = {np.mean(ig > thr):.3f}   P(I > rus_exit) = {np.mean(ig > rus):.3f}")
    print(f"|1>  I mean {ie.mean():+.6f}  std {ie.std():.6f}   P(I > threshold) = {np.mean(ie > thr):.3f}")
    fid = 0.5 * (np.mean(ig <= thr) + np.mean(ie > thr))
    best = max(np.linspace(min(ig.min(), ie.min()), max(ig.max(), ie.max()), 400),
               key=lambda t: 0.5 * (np.mean(ig <= t) + np.mean(ie > t)))
    print(f"assignment fidelity at stored threshold {fid:.3f};  best threshold {best:+.6f} -> "
          f"{0.5 * (np.mean(ig <= best) + np.mean(ie > best)):.3f}")

if __name__ == "__main__":
    sys.exit(main())
