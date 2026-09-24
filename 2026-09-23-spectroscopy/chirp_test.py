"""Chirp (adiabatic rapid passage) vs saturation spectroscopy, qolab Q1, flux at the idle point.

Job 1, points (2000 shots each): |0> / x180 references; x180 at 0.7/1.0/1.3 of its amplitude;
       saturation (0.5 x stored amplitude, the node default) for 20 and 80 us at f01 and at the
       two-photon frequency f01 - |alpha|/2; a 4 us chirp sweeping +-10 MHz around f01 at six
       drive strengths; the same chirp around the two-photon frequency at three strengths.
Job 2, spectra (1000 shots per point): saturation 20 us vs drive frequency, f01 +- 30 MHz in 2 MHz
       steps; chirp (+-10 MHz band) vs band centre, f01 +- 30 MHz in 3 MHz steps.
Job 3, fine saturation scan (1000 shots per point) around the two-photon frequency, +-2 MHz.

Every shot starts from the node's 5xT1 thermal reset. Populations come from projecting IQ onto
the ref0 -> ref1 axis measured in the same job. Nothing is written to the state; the chirp
envelope is injected into the generated config only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/qolab")
OUT = Path(__file__).parent
QUBIT = "Q1"
N_PT, N_SPEC = 2000, 1000
TAU, RAMP = 4000, 500            # chirp length and cosine edge ramps, ns
BAND = 20_000_000                # full sweep, Hz
RATE = BAND // TAU               # Hz/ns
FR_REF = 8e6                     # nominal Rabi frequency at chirp scale 1.0, Hz
SAT_AMP = 0.5                    # the node's default operation_amplitude_factor


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="qolab", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    machine = Quam.load()
    q = machine.qubits[QUBIT]
    el = q.xy.name
    IF0 = int(q.xy.intermediate_frequency)
    alpha = float(q.anharmonicity)
    IF2 = int(round(IF0 - abs(alpha) / 2))
    pi = q.xy.operations["x180_DragCosine"]
    fr_per_amp = (1e9 / pi.length) / pi.amplitude          # cosine-envelope pi pulse: f_R,peak = 1/L
    a_ref = FR_REF / fr_per_amp
    sat_fr = SAT_AMP * q.xy.operations["saturation"].amplitude * fr_per_amp

    env = np.ones(TAU)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(RAMP) / RAMP))
    env[:RAMP], env[-RAMP:] = ramp, ramp[::-1]
    config = machine.generate_config()
    sat_pulse = config["pulses"][config["elements"][el]["operations"]["saturation"]]
    config["waveforms"]["chirp_env_I"] = {"type": "arbitrary", "samples": (a_ref * env).tolist()}
    config["waveforms"]["chirp_env_Q"] = {"type": "constant", "sample": 0.0}
    config["pulses"]["chirp_env_pulse"] = {"operation": "control", "length": TAU,
                                           "waveforms": {"I": "chirp_env_I", "Q": "chirp_env_Q"},
                                           "digital_marker": sat_pulse.get("digital_marker", "ON")}
    config["elements"][el]["operations"]["chirp_env"] = "chirp_env_pulse"

    meta = dict(qubit=QUBIT, IF0=IF0, IF2=IF2, alpha=alpha, t1=float(q.T1), rate_hz_per_ns=RATE, tau_ns=TAU,
                band_hz=BAND, fr_ref_nominal=FR_REF, chirp_amp_ref=a_ref, sat_fr_nominal=sat_fr,
                pi_amp=float(pi.amplitude), pi_len=int(pi.length))
    print(f"{QUBIT}: IF {IF0/1e6:.2f} MHz, alpha {alpha/1e6:.1f} MHz -> two-photon IF {IF2/1e6:.2f} MHz; "
          f"chirp {BAND/1e6:.0f} MHz in {TAU/1e3:.0f} us ({RATE} Hz/ns); chirp amp at scale 1 = {a_ref:.4f} "
          f"(nominal f_R {FR_REF/1e6:.0f} MHz); node saturation nominal f_R {sat_fr/1e6:.2f} MHz", flush=True)

    def reset():
        q.reset_qubit_thermal()
        align()

    def readout(I, Q, i_st, q_st):
        q.resonator.measure("readout", qua_vars=(I, Q))
        save(I, i_st)
        save(Q, q_st)
        align()

    def chirp(scale, start_if):
        q.xy.update_frequency(start_if)
        play("chirp_env" * amp(scale), el, chirp=(RATE, "Hz/nsec"))
        align()

    def values(job, name):
        raw = job.result_handles.get(name).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        return np.asarray(raw, dtype=float)

    # ---- job 1: point measurements
    SCALES = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    points = [("ref0", None), ("ref1", None)]
    points += [(f"pi x{s}", ("pi", s)) for s in (0.7, 1.0, 1.3)]
    points += [("sat20 f01", ("sat", IF0, 5000)), ("sat80 f01", ("sat", IF0, 20000)),
               ("sat20 2ph", ("sat", IF2, 5000)), ("sat80 2ph", ("sat", IF2, 20000))]
    points += [(f"chirp f01 s{s:g}", ("chirp", s, IF0)) for s in SCALES]
    points += [(f"chirp 2ph s{s:g}", ("chirp", s, IF2)) for s in (0.25, 0.5, 1.0)]

    def job1():
        with program() as prog:
            n = declare(int); I = declare(fixed); Q = declare(fixed)
            st = {lab: (declare_stream(), declare_stream()) for lab, _ in points}
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N_PT, n + 1):
                for lab, spec in points:
                    reset()
                    q.xy.update_frequency(IF0)
                    if lab == "ref1":
                        q.xy.play("x180"); align()
                    elif spec and spec[0] == "pi":
                        q.xy.play("x180", amplitude_scale=spec[1]); align()
                    elif spec and spec[0] == "sat":
                        q.xy.update_frequency(spec[1])
                        q.xy.play("saturation", amplitude_scale=SAT_AMP, duration=spec[2]); align()
                    elif spec and spec[0] == "chirp":
                        chirp(spec[1], spec[2] - BAND // 2)
                    readout(I, Q, *st[lab])
            with stream_processing():
                for lab, (si, sq) in st.items():
                    si.save_all(f"{lab}|I"); sq.save_all(f"{lab}|Q")
        return prog

    # ---- job 2: spectra; job 3: fine saturation scan at the two-photon frequency
    sat_off = np.arange(-30_000_000, 30_000_001, 2_000_000)
    chirp_off = np.arange(-30_000_000, 30_000_001, 3_000_000)
    fine_off = np.arange(-2_000_000, 2_000_001, 100_000)

    def spectra(kind):
        with program() as prog:
            n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
            r0 = (declare_stream(), declare_stream()); r1 = (declare_stream(), declare_stream())
            a = (declare_stream(), declare_stream()); b = (declare_stream(), declare_stream())
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N_SPEC, n + 1):
                reset(); q.xy.update_frequency(IF0); readout(I, Q, *r0)
                reset(); q.xy.update_frequency(IF0); q.xy.play("x180"); align(); readout(I, Q, *r1)
                if kind == "main":
                    with for_(*from_array(f, (IF0 + sat_off).astype(int))):
                        reset(); q.xy.update_frequency(f)
                        q.xy.play("saturation", amplitude_scale=SAT_AMP, duration=5000); align()
                        readout(I, Q, *a)
                    with for_(*from_array(f, (IF0 + chirp_off - BAND // 2).astype(int))):
                        reset(); chirp(0.5, f); readout(I, Q, *b)
                else:
                    with for_(*from_array(f, (IF2 + fine_off).astype(int))):
                        reset(); q.xy.update_frequency(f)
                        q.xy.play("saturation", amplitude_scale=SAT_AMP, duration=5000); align()
                        readout(I, Q, *a)
            with stream_processing():
                r0[0].save_all("r0|I"); r0[1].save_all("r0|Q"); r1[0].save_all("r1|I"); r1[1].save_all("r1|Q")
                na = len(sat_off) if kind == "main" else len(fine_off)
                a[0].buffer(na).average().save("a|I"); a[1].buffer(na).average().save("a|Q")
                if kind == "main":
                    b[0].buffer(len(chirp_off)).average().save("b|I"); b[1].buffer(len(chirp_off)).average().save("b|Q")
        return prog

    data: dict[str, np.ndarray] = {}
    qpu = {}
    with qm_session(machine.connect(), config, timeout=60) as qm:
        for name, prog, keys in (
            ("job1", job1(), [f"{lab}|{c}" for lab, _ in points for c in "IQ"]),
            ("main", spectra("main"), ["r0|I", "r0|Q", "r1|I", "r1|Q", "a|I", "a|Q", "b|I", "b|Q"]),
            ("fine", spectra("fine"), ["r0|I", "r0|Q", "r1|I", "r1|Q", "a|I", "a|Q"]),
        ):
            t0 = time.perf_counter()
            job = qm.execute(prog)
            job.result_handles.wait_for_all_values()
            qpu[name] = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
            for k in keys:
                data[f"{name}/{k}"] = values(job, k)
            print(f"{name}: {qpu[name]:.1f} s QPU (wall {time.perf_counter() - t0:.1f} s)", flush=True)

    meta.update(qpu_s=qpu, points=[lab for lab, _ in points], sat_off=sat_off.tolist(), chirp_off=chirp_off.tolist(),
                fine_off=fine_off.tolist(), scales=list(SCALES), n_pt=N_PT, n_spec=N_SPEC)
    np.savez(OUT / "chirp_test_data.npz", **{k.replace("/", "__").replace("|", "_").replace(" ", "~"): v
                                            for k, v in data.items()})
    (OUT / "chirp_test_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {OUT / 'chirp_test_data.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
