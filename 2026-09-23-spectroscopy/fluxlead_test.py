"""Flux-step lead time before the chirp: does the pulsed flux hold its value? (qolab Q1)

Chirp (4 us, +-10 MHz, ~3 MHz Rabi) at flux offsets -80.9 and +80.9 mV from the idle point, with the z step starting
LEAD before the chirp (0.5 .. 40 us) and held through it. If the pulsed flux droops back toward the idle point, the
fitted f01 moves toward the idle frequency as LEAD grows. 18 band centres (f01-32 .. +10.5 MHz), 150 shots.
"""
import json, sys, time
from pathlib import Path
import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
OUT = Path(__file__).parent
TAU, RAMP, BAND = 4000, 500, 20_000_000
RATE = BAND // TAU
FR_CHIRP, N = 3e6, 150
LEADS_NS = (500, 1000, 2000, 5000, 10000, 20000, 40000)


def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc
    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam
    machine = Quam.load(); q = machine.qubits["Q1"]; el = q.xy.name
    IF0 = int(q.xy.intermediate_frequency)
    pi = q.xy.operations["x180_DragCosine"]; fr_per_amp = (1e9 / pi.length) / pi.amplitude
    const_amp = float(q.z.operations["const"].amplitude)
    phi = float(np.sqrt(20e6 / abs(float(q.freq_vs_flux_01_quad_term)))) * 12 / 14   # the +-80.9 mV columns
    DCS = (-phi, +phi)
    centres = np.arange(-32_000_000, 12_000_000, 2_500_000)
    config = machine.generate_config()
    marker = config["pulses"][config["elements"][el]["operations"]["saturation"]].get("digital_marker", "ON")
    env = np.ones(TAU); r = 0.5 * (1 - np.cos(np.pi * np.arange(RAMP) / RAMP)); env[:RAMP], env[-RAMP:] = r, r[::-1]
    config["waveforms"].update({"fl_zero": {"type": "constant", "sample": 0.0},
                                "fl_chirp_I": {"type": "arbitrary", "samples": (FR_CHIRP / fr_per_amp * env).tolist()}})
    config["pulses"]["fl_chirp_pulse"] = {"operation": "control", "length": TAU,
                                          "waveforms": {"I": "fl_chirp_I", "Q": "fl_zero"}, "digital_marker": marker}
    config["elements"][el]["operations"]["fl_chirp"] = "fl_chirp_pulse"
    combos = [(dc, lead) for dc in DCS for lead in LEADS_NS]
    with program() as prog:
        n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
        st = [(declare_stream(), declare_stream()) for _ in combos]
        r0 = (declare_stream(), declare_stream()); r1 = (declare_stream(), declare_stream())
        machine.initialize_qpu(target=q); align()
        with for_(n, 0, n < N, n + 1):
            q.reset_qubit_thermal(); align(); q.xy.update_frequency(IF0)
            q.resonator.measure("readout", qua_vars=(I, Q)); save(I, r0[0]); save(Q, r0[1]); align()
            q.reset_qubit_thermal(); align(); q.xy.update_frequency(IF0); q.xy.play("x180"); align()
            q.resonator.measure("readout", qua_vars=(I, Q)); save(I, r1[0]); save(Q, r1[1]); align()
            for (dc, lead), s in zip(combos, st):
                with for_(*from_array(f, (IF0 + centres - BAND // 2).astype(int))):
                    q.reset_qubit_thermal(); align()
                    q.z.play("const", amplitude_scale=dc / const_amp, duration=(lead + TAU) // 4)
                    q.xy.update_frequency(f); q.xy.wait(lead // 4)
                    play("fl_chirp" * amp(1.0), el, chirp=(RATE, "Hz/nsec"))
                    align()
                    q.resonator.measure("readout", qua_vars=(I, Q)); save(I, s[0]); save(Q, s[1]); align()
        with stream_processing():
            for k, s in enumerate(st):
                s[0].buffer(len(centres)).average().save(f"c{k}_I"); s[1].buffer(len(centres)).average().save(f"c{k}_Q")
            r0[0].save_all("r0I"); r0[1].save_all("r0Q"); r1[0].save_all("r1I"); r1[1].save_all("r1Q")
    with qm_session(machine.connect(), config, timeout=60) as qm:
        t0 = time.perf_counter(); job = qm.execute(prog); job.result_handles.wait_for_all_values()
        qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
        def get(nm):
            raw = job.result_handles.get(nm).fetch_all()
            try: raw = raw["value"]
            except (TypeError, IndexError, KeyError, ValueError): pass
            return np.asarray(raw, dtype=float)
        data = {k: get(k) for k in ["r0I", "r0Q", "r1I", "r1Q"] + [f"c{i}_{c}" for i in range(len(combos)) for c in "IQ"]}
        print(f"lead test: {qpu:.1f} s QPU (wall {time.perf_counter() - t0:.1f} s)", flush=True)
    np.savez(OUT / "fluxlead_data.npz", **data)
    (OUT / "fluxlead_meta.json").write_text(json.dumps(dict(IF0=IF0, combos=combos, centres=centres.tolist(), n=N, qpu_s=qpu)))
    print("saved", flush=True)

if __name__ == "__main__":
    sys.exit(main())
