"""Chirp variants for short T1: gilboa qD2 (T1 1.3 us) and qC3 (1.8 us), 24 Sep 2026.

Three pulses of the node's sweep length (T1/5: 268 and 360 ns), each at five drives around the amplitude
the stored x180 predicts for a clean passage, against band centre:
  lin20   the node's pulse: linear sweep over 20 MHz, raised-cosine edges (time x band 5-7 MHz us)
  lin80   linear sweep over 80 MHz, same edges (time x band 21-29 MHz us)
  hs40    hyperbolic-secant amplitude with a tanh frequency sweep over 40 MHz (beta 5), the sweep written
          into the I/Q samples; the drive rises smoothly while the frequency is still far from the line
Also |0> and x180 references. Local copy of the 22 Sep gilboa snapshot with the C/D qubits active;
pulses injected in the generated config only; nothing written to any state.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "shortT1"
OUT.mkdir(exist_ok=True)
STATE = HERE / "gilboa" / "state"
QUBITS = ["qD2", "qC3"]
LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
SHOTS = 300


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="gilboa", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    from calibration_utils.qubit_spectroscopy_chirp.chirp import envelope, rabi_per_amplitude, sweep_length_ns, target_rabi_hz

    machine = Quam.load(str(STATE))
    config = machine.generate_config()
    config["waveforms"]["sv_zero"] = {"type": "constant", "sample": 0.0}
    plan = {}
    for name in QUBITS:
        q = machine.qubits[name]
        el = q.xy.name
        tau = sweep_length_ns(q.T1)
        kappa = rabi_per_amplitude(q)
        ops = config["elements"][el]["operations"]
        marker = config["pulses"][ops["x180"]].get("digital_marker") if "x180" in ops else None
        pulses = {}
        for kind, band in (("lin20", 20e6), ("lin80", 80e6), ("hs40", 40e6)):
            rate = int(round(band / tau))
            centre_amp = target_rabi_hz(rate) / kappa
            peak = min(0.99, centre_amp * max(LEVELS))
            t = np.arange(tau)
            if kind.startswith("lin"):
                I, Q = peak * envelope(tau), np.zeros(tau)
            else:
                beta = 5.0
                x = beta * (2 * t / (tau - 1) - 1)
                a = 1 / np.cosh(x)
                df = (band / 2) * np.tanh(x)                     # Hz, about the band centre
                phase = 2 * np.pi * np.cumsum(df) * 1e-9
                I, Q = peak * a * np.cos(phase), peak * a * np.sin(phase)
            wi, wq = f"sv_{kind}_{el}_I", f"sv_{kind}_{el}_Q"
            config["waveforms"][wi] = {"type": "arbitrary", "samples": I.tolist()}
            config["waveforms"][wq] = {"type": "arbitrary", "samples": Q.tolist()}
            pulse = {"operation": "control", "length": tau, "waveforms": {"I": wi, "Q": wq}}
            if marker is not None:
                pulse["digital_marker"] = marker
            config["pulses"][f"sv_{kind}_{el}_pulse"] = pulse
            ops[f"sv_{kind}"] = f"sv_{kind}_{el}_pulse"
            scales = [min(0.99, centre_amp * g) / peak for g in LEVELS]
            step = 5e6 if band <= 40e6 else 10e6
            lo, hi = (-150e6, 110e6) if band > 40e6 else (-130e6, 60e6)
            centres = np.arange(lo, hi + 1, step)
            pulses[kind] = dict(band=band, rate=rate, centre_amp=centre_amp, peak=peak, scales=scales,
                                played=[s * peak for s in scales], centres=centres.tolist())
        plan[name] = dict(el=el, tau=tau, kappa=kappa, IF=int(q.xy.intermediate_frequency), f01=float(q.f_01), T1=float(q.T1),
                          pulses=pulses)
        print(name, "tau", tau, "ns; centre amplitudes", {k: round(v["centre_amp"], 3) for k, v in pulses.items()}, flush=True)

    streams = []
    with program() as prog:
        n = declare(int)
        f = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        st = {}
        for name in QUBITS:
            q = machine.qubits[name]
            p = plan[name]
            machine.initialize_qpu(target=q)
            align()
            refs = {k: (declare_stream(), declare_stream()) for k in ("r0", "r1")}
            st[name] = {"refs": refs}
            for kind, pp in p["pulses"].items():
                st[name][kind] = [(declare_stream(), declare_stream()) for _ in LEVELS]
            with for_(n, 0, n < SHOTS, n + 1):
                for k in ("r0", "r1"):
                    q.reset_qubit_thermal()
                    align()
                    q.xy.update_frequency(p["IF"])
                    if k == "r1":
                        q.xy.play("x180")
                    align()
                    q.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, refs[k][0])
                    save(Q, refs[k][1])
                    align()
                for kind, pp in p["pulses"].items():
                    centres = (p["IF"] + np.asarray(pp["centres"]) - (pp["band"] / 2 if kind.startswith("lin") else 0)).astype(int)
                    for j, s in enumerate(pp["scales"]):
                        with for_(*from_array(f, centres)):
                            q.reset_qubit_thermal()
                            align()
                            q.xy.update_frequency(f)
                            if kind.startswith("lin"):
                                play(f"sv_{kind}" * amp(float(s)), p["el"], chirp=(pp["rate"], "Hz/nsec"))
                            else:
                                play(f"sv_{kind}" * amp(float(s)), p["el"])
                            align()
                            q.resonator.measure("readout", qua_vars=(I, Q))
                            save(I, st[name][kind][j][0])
                            save(Q, st[name][kind][j][1])
                            align()
        with stream_processing():
            for name in QUBITS:
                for k, (si, sq) in st[name]["refs"].items():
                    si.average().save(f"{name}_{k}_I")
                    sq.average().save(f"{name}_{k}_Q")
                for kind, pp in plan[name]["pulses"].items():
                    for j in range(len(LEVELS)):
                        si, sq = st[name][kind][j]
                        si.buffer(len(pp["centres"])).average().save(f"{name}_{kind}_{j}_I")
                        sq.buffer(len(pp["centres"])).average().save(f"{name}_{kind}_{j}_Q")
                        streams += [f"{name}_{kind}_{j}_I", f"{name}_{kind}_{j}_Q"]
                streams += [f"{name}_{k}_{c}" for k in ("r0", "r1") for c in "IQ"]

    data = {}
    t0 = time.perf_counter()
    with qm_session(machine.connect(), config, timeout=60) as qm:
        job = qm.execute(prog)
        job.result_handles.wait_for_all_values()
        qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
        for k in streams:
            raw = job.result_handles.get(k).fetch_all()
            try:
                raw = raw["value"]
            except (TypeError, IndexError, KeyError, ValueError):
                pass
            data[k] = np.asarray(raw, dtype=float)
    print(f"QPU {qpu:.1f} s, wall {time.perf_counter() - t0:.0f} s", flush=True)
    np.savez(OUT / "shortT1_variants.npz", **data)
    (OUT / "shortT1_variants.json").write_text(json.dumps(dict(plan=plan, levels=list(LEVELS), shots=SHOTS, qpu_s=qpu), indent=1,
                                                         default=float))
    print("saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
