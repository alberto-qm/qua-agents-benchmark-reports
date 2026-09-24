"""Chirped vs saturation qubit-spectroscopy-vs-flux map, qolab Q1 (flux pulsed from the idle point).

A  saturation map, as node 03b plays it today: z step and saturation drive played together for
   `length * u.ns` = 20000 clock cycles (80 us), drive 0.5 x stored saturation amplitude; 61 rows of 0.5 MHz
   (f01-25 .. f01+5 MHz), 50 shots (the node default).
B  chirp map: z step starts 5 us before a 4 us chirp over +-10 MHz (5000 Hz/ns) at ~3 MHz nominal Rabi and is held
   through it; 18 band centres f01-32 .. f01+10.5 MHz in 2.5 MHz steps, 200 shots.
15 flux offsets over +-94 mV (arc depth ~20 MHz from the stored quad term). 5xT1 thermal reset every shot, readout
at the idle point, pulses injected in the generated config only, nothing written to the state.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
OUT = Path(__file__).parent
TAU, RAMP, BAND = 4000, 500, 20_000_000
RATE = BAND // TAU
FR_CHIRP = 3e6
LEAD_NS = 5000
N_SAT, N_CHIRP = 50, 200


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    machine = Quam.load()
    q = machine.qubits["Q1"]
    el = q.xy.name
    IF0 = int(q.xy.intermediate_frequency)
    pi = q.xy.operations["x180_DragCosine"]
    fr_per_amp = (1e9 / pi.length) / pi.amplitude
    const_amp = float(q.z.operations["const"].amplitude)
    quad = float(q.freq_vs_flux_01_quad_term)
    phi_max = float(np.sqrt(20e6 / abs(quad)))
    dcs = np.linspace(-phi_max, phi_max, 15)
    sat_rows = np.arange(-25_000_000, 5_000_001, 500_000)
    chirp_centres = np.arange(-32_000_000, 12_000_000, 2_500_000)

    config = machine.generate_config()
    marker = config["pulses"][config["elements"][el]["operations"]["saturation"]].get("digital_marker", "ON")
    env = np.ones(TAU)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(RAMP) / RAMP))
    env[:RAMP], env[-RAMP:] = ramp, ramp[::-1]
    config["waveforms"].update({"fm_zero": {"type": "constant", "sample": 0.0},
                                "fm_chirp_I": {"type": "arbitrary", "samples": (FR_CHIRP / fr_per_amp * env).tolist()}})
    config["pulses"]["fm_chirp_pulse"] = {"operation": "control", "length": TAU,
                                          "waveforms": {"I": "fm_chirp_I", "Q": "fm_zero"}, "digital_marker": marker}
    config["elements"][el]["operations"]["fm_chirp"] = "fm_chirp_pulse"
    print(f"Q1: IF {IF0/1e6:.2f} MHz, flux offsets +-{phi_max*1e3:.1f} mV (15), quad {quad/1e9:.3f} GHz/V^2", flush=True)

    def reset():
        q.reset_qubit_thermal(); align()

    def build(kind):
        rows = sat_rows if kind == "sat" else chirp_centres
        n_sh = N_SAT if kind == "sat" else N_CHIRP
        with program() as prog:
            n = declare(int); f = declare(int); dc = declare(fixed); I = declare(fixed); Q = declare(fixed)
            I_st, Q_st, r0I, r0Q, r1I, r1Q = (declare_stream() for _ in range(6))
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < n_sh, n + 1):
                reset(); q.xy.update_frequency(IF0)
                q.resonator.measure("readout", qua_vars=(I, Q)); save(I, r0I); save(Q, r0Q); align()
                reset(); q.xy.update_frequency(IF0); q.xy.play("x180"); align()
                q.resonator.measure("readout", qua_vars=(I, Q)); save(I, r1I); save(Q, r1Q); align()
                with for_(*from_array(dc, dcs)):
                    with for_(*from_array(f, (IF0 + rows - (BAND // 2 if kind == "chirp" else 0)).astype(int))):
                        reset()
                        if kind == "sat":   # node 03b today: both pulses for 20000 cycles = 80 us, together
                            q.z.play("const", amplitude_scale=dc * (1 / const_amp), duration=20000)
                            q.xy.update_frequency(f)
                            q.xy.play("saturation", amplitude_scale=0.5, duration=20000)
                        else:               # flux step leads the chirp by 5 us and is held through it
                            q.z.play("const", amplitude_scale=dc * (1 / const_amp), duration=(LEAD_NS + TAU) // 4)
                            q.xy.update_frequency(f)
                            q.xy.wait(LEAD_NS // 4)
                            play("fm_chirp" * amp(1.0), el, chirp=(RATE, "Hz/nsec"))
                        align()
                        q.resonator.measure("readout", qua_vars=(I, Q)); save(I, I_st); save(Q, Q_st); align()
            with stream_processing():
                I_st.buffer(len(rows)).buffer(len(dcs)).average().save("I")
                Q_st.buffer(len(rows)).buffer(len(dcs)).average().save("Q")
                r0I.save_all("r0I"); r0Q.save_all("r0Q"); r1I.save_all("r1I"); r1Q.save_all("r1Q")
        return prog

    def get(job, name):
        raw = job.result_handles.get(name).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        return np.asarray(raw, dtype=float)

    data, qpu = {}, {}
    with qm_session(machine.connect(), config, timeout=60) as qm:
        for kind in ("sat", "chirp"):
            t0 = time.perf_counter()
            job = qm.execute(build(kind)); job.result_handles.wait_for_all_values()
            qpu[kind] = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
            for k in ("I", "Q", "r0I", "r0Q", "r1I", "r1Q"):
                data[f"{kind}_{k}"] = get(job, k)
            print(f"{kind}: {qpu[kind]:.1f} s QPU (wall {time.perf_counter() - t0:.1f} s), map shape {data[f'{kind}_I'].shape}", flush=True)

    np.savez(OUT / "fluxmap_data.npz", **data)
    (OUT / "fluxmap_meta.json").write_text(json.dumps(dict(
        IF0=IF0, dcs=dcs.tolist(), sat_rows=sat_rows.tolist(), chirp_centres=chirp_centres.tolist(), quad=quad,
        fr_chirp=FR_CHIRP, lead_ns=LEAD_NS, tau_ns=TAU, band=BAND, n_sat=N_SAT, n_chirp=N_CHIRP, qpu_s=qpu,
        joint_offset=float(q.z.joint_offset)), indent=1))
    print("saved", OUT / "fluxmap_data.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
