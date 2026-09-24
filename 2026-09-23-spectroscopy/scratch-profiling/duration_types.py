"""What does play(duration=<float>) do? Node 03b passes `length * u.ns` = 20000.0 (a float)."""
import sys
from pathlib import Path
STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
N = 20_000

def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc
    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from qualang_tools.units import unit
    from quam_config import Quam
    u = unit(coerce_to_integer=True)
    machine = Quam.load(); q = machine.qubits["Q1"]
    IF = int(q.xy.intermediate_frequency); far = IF - 10_000_000
    L = q.xy.operations["saturation"].length
    node_dur = L * u.ns
    print(f"saturation length {L} ns; node expression length * u.ns = {node_dur!r} ({type(node_dur).__name__})", flush=True)
    cases = [
        ("readout only", None),
        ("no duration arg (stored length)", "default"),
        ("duration=5000 (int, 20 us intended)", 5000),
        ("duration=20000 (int)", 20000),
        ("duration=20000.0 (float, the node's value)", node_dur),
    ]
    def build(d):
        with program() as prog:
            n = declare(int); I = declare(fixed); Q = declare(fixed); s = declare_stream()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N, n + 1):
                if d is not None:
                    q.xy.update_frequency(far)
                    if d == "default":
                        q.xy.play("saturation", amplitude_scale=0.5)
                    else:
                        q.xy.play("saturation", amplitude_scale=0.5, duration=d)
                    align()
                q.resonator.measure("readout", qua_vars=(I, Q)); save(I, s); align()
            with stream_processing():
                s.average().save("I")
        return prog
    with qm_session(machine.connect(), machine.generate_config(), timeout=60) as qm:
        base = None
        for label, d in cases:
            try:
                job = qm.execute(build(d)); job.result_handles.wait_for_all_values()
                per = 1e6 * float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all()) / N
            except Exception as exc:
                print(f"  {label:<45} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True); continue
            base = per if base is None else base
            print(f"  {label:<45} {per:7.2f} us/shot  (pulse {per - base:+6.2f} us)", flush=True)

if __name__ == "__main__":
    sys.exit(main())
