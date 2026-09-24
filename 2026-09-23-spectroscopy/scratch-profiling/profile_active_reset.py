"""Step 6 of the shot-overhead ladder with active reset in place of the 5xT1 wait.

Same loop as profile_shot_overhead.py step 6 (update_frequency, 3 aligns, 20 us
saturation, readout); only the reset changes. max_attempts=1 is one measure plus a
conditional pi; 15 (the default) repeats until the qubit reads ground.
"""
import sys, time
from pathlib import Path

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
N_SHOTS = 100_000


def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc
    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from qualang_tools.units import unit
    from quam_config import Quam

    u = unit(coerce_to_integer=True)
    machine = Quam.load()
    qubit = machine.qubits["Q1"]

    def build(max_attempts):
        with program() as prog:
            n = declare(int); df = declare(int); I = declare(fixed); Q = declare(fixed)
            n_st = declare_stream(); I_st = declare_stream(); Q_st = declare_stream()
            att_st = declare_stream() if max_attempts > 1 else None
            machine.initialize_qpu(target=qubit); align()
            with for_(n, 0, n < N_SHOTS, n + 1):
                save(n, n_st)
                qubit.reset_qubit_active(save_qua_var=att_st, max_attempts=max_attempts)
                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                align()
                qubit.xy.play("saturation", amplitude_scale=0.5, duration=20_000 * u.ns)
                align()
                qubit.resonator.measure("readout", qua_vars=(I, Q))
                save(I, I_st); save(Q, Q_st)
                align()
            with stream_processing():
                n_st.save("n")
                I_st.buffer(1).average().save("I1")
                Q_st.buffer(1).average().save("Q1")
                if att_st is not None:
                    att_st.average().save("attempts")
        return prog

    qmm = machine.connect()
    config = machine.generate_config()
    with qm_session(qmm, config, timeout=60) as qm:
        for attempts in (1, 15):
            try:
                t0 = time.perf_counter()
                job = qm.execute(build(attempts))
                job.result_handles.wait_for_all_values()
                qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
                mean_att = None
                if attempts > 1:
                    try:
                        mean_att = float(job.result_handles.get("attempts").fetch_all())
                    except Exception:
                        pass
                extra = f", mean attempts {mean_att:.2f}" if mean_att is not None else ""
                print(f"[flux initialised] active reset max_attempts={attempts:<3} {qpu:>6.2f}s QPU  "
                      f"{1e6 * qpu / N_SHOTS:>7.2f} us/shot  (wall {time.perf_counter() - t0:.1f}s{extra})", flush=True)
            except Exception as exc:
                print(f"[flux initialised] active reset max_attempts={attempts} FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
