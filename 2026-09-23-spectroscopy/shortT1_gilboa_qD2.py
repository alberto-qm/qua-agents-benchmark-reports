"""Where does the chirp break down? gilboa qD2, T1 ~1.3 us (lab snapshot of 22 Sep, local copy with the ten C/D
qubits marked active so qD2's flux sits at its idle point; the cloud state is untouched).

J1 points: |0>/x180 references; saturation 20 us on f01 at the node default drive (0.5 x stored amplitude) and at
   2 and 8 MHz nominal Rabi; chirps over +-10 MHz lasting 4 us, 1 us and 250 ns (cosine edges = 1/8 of the length)
   at nominal Rabi 2/4/8/16 MHz.
J2 spectra over f01 +-15 MHz: saturation (node default drive, 0.25 MHz steps) and a chirp band scan with the best
   J1 chirp setting (2.5 MHz steps).
Nominal Rabi from qD2's calibrated x180 (48 ns). 5xT1 thermal reset every shot; nothing written to the state.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

STATE = Path("/private/tmp/claude-734394052/-Users-atosato-code-QM-qua-agents-benchmark/50ae3071-f59a-4af0-b332-5646b1ad4122/scratchpad/gilboa-cd-active")
OUT = Path(__file__).parent
QUBIT = "qD2"
BAND = 20_000_000
TAUS = (4000, 1000, 248)            # ns, multiples of 4 (248 ~ 250)
FR_TOP = 16e6
AMPS = (1 / 8, 1 / 4, 1 / 2, 1.0)   # x FR_TOP -> 2, 4, 8, 16 MHz


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="gilboa", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    machine = Quam.load()
    q = machine.qubits[QUBIT]
    el = q.xy.name
    IF0 = int(q.xy.intermediate_frequency)
    assert abs(IF0) + 40e6 < 400e6, IF0
    pi = q.xy.operations["x180_DragCosine"]
    fr_per_amp = (1e9 / pi.length) / pi.amplitude
    a_top = FR_TOP / fr_per_amp
    node_fr = 0.5 * float(q.xy.operations["saturation"].amplitude) * fr_per_amp
    assert a_top < 0.99, a_top
    print(f"{QUBIT}: f01 {q.f_01/1e6:.2f} MHz, IF {IF0/1e6:.2f} MHz, T1 {q.T1*1e6:.2f} us, active {machine.active_qubit_names}; "
          f"node default drive {node_fr/1e6:.2f} MHz nominal Rabi; amplitude at 16 MHz {a_top:.3f}", flush=True)

    config = machine.generate_config()
    marker = config["pulses"][config["elements"][el]["operations"]["saturation"]].get("digital_marker", "ON")
    config["waveforms"].update({"sT_zero": {"type": "constant", "sample": 0.0}, "sT_sat_I": {"type": "constant", "sample": a_top}})
    config["pulses"]["sT_sat_pulse"] = {"operation": "control", "length": 20000,
                                        "waveforms": {"I": "sT_sat_I", "Q": "sT_zero"}, "digital_marker": marker}
    config["elements"][el]["operations"]["sT_sat"] = "sT_sat_pulse"
    for tau in TAUS:
        rmp = tau // 8
        env = np.ones(tau); r = 0.5 * (1 - np.cos(np.pi * np.arange(rmp) / rmp)); env[:rmp], env[-rmp:] = r, r[::-1]
        config["waveforms"][f"sT_ch{tau}_I"] = {"type": "arbitrary", "samples": (a_top * env).tolist()}
        config["pulses"][f"sT_ch{tau}_pulse"] = {"operation": "control", "length": tau,
                                                  "waveforms": {"I": f"sT_ch{tau}_I", "Q": "sT_zero"}, "digital_marker": marker}
        config["elements"][el]["operations"][f"sT_ch{tau}"] = f"sT_ch{tau}_pulse"

    def reset():
        q.reset_qubit_thermal(); align()

    def ro(I, Q, st):
        q.resonator.measure("readout", qua_vars=(I, Q)); save(I, st[0]); save(Q, st[1]); align()

    def chirp(centre, tau, scale):
        q.xy.update_frequency(centre - BAND // 2)
        play(f"sT_ch{tau}" * amp(scale), el, chirp=(BAND // tau, "Hz/nsec")); align()

    def sat(freq, scale):
        q.xy.update_frequency(freq)
        if scale is None:
            q.xy.play("saturation", amplitude_scale=0.5, duration=5000)
        else:
            play("sT_sat" * amp(scale), el)
        align()

    def pair():
        return declare_stream(), declare_stream()

    data, qpu = {}, {}

    def execute(qm, name, prog, keys):
        t0 = time.perf_counter()
        job = qm.execute(prog); job.result_handles.wait_for_all_values()
        qpu[name] = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
        for k in keys:
            raw = job.result_handles.get(k).fetch_all()
            try:
                raw = raw["value"]
            except (TypeError, IndexError, KeyError, ValueError):
                pass
            data[f"{name}/{k}"] = np.asarray(raw, dtype=float).ravel()
        print(f"{name}: {qpu[name]:.1f} s QPU (wall {time.perf_counter() - t0:.1f} s)", flush=True)

    labels = [("ref0", None), ("ref1", None), ("sat_node", ("s", None)), ("sat_2", ("s", 1 / 8)), ("sat_8", ("s", 1 / 2))]
    labels += [(f"ch{tau}_{a:g}", ("c", tau, a)) for tau in TAUS for a in AMPS]
    with program() as prog:
        n = declare(int); I = declare(fixed); Q = declare(fixed)
        st = {lab: pair() for lab, _ in labels}
        machine.initialize_qpu(target=q); align()
        with for_(n, 0, n < 3000, n + 1):
            for lab, spec in labels:
                reset(); q.xy.update_frequency(IF0)
                if lab == "ref1":
                    q.xy.play("x180"); align()
                elif spec and spec[0] == "s":
                    sat(IF0, spec[1])
                elif spec and spec[0] == "c":
                    chirp(IF0, spec[1], spec[2])
                ro(I, Q, st[lab])
        with stream_processing():
            for lab, s in st.items():
                s[0].save_all(f"{lab}|I"); s[1].save_all(f"{lab}|Q")
    meta = dict(qubit=QUBIT, IF0=IF0, t1=float(q.T1), fr_per_amp=fr_per_amp, a_top=a_top, fr_top=FR_TOP, node_fr=node_fr,
                taus=list(TAUS), amps=list(AMPS), labels=[lab for lab, _ in labels], f01_hz=float(q.f_01))
    with qm_session(machine.connect(), config, timeout=60) as qm:
        execute(qm, "J1", prog, [f"{lab}|{c}" for lab, _ in labels for c in "IQ"])
        z0 = np.array([data["J1/ref0|I"].mean(), data["J1/ref0|Q"].mean()])
        ax = np.array([data["J1/ref1|I"].mean(), data["J1/ref1|Q"].mean()]) - z0
        proj = lambda lab: (((data[f"J1/{lab}|I"] - z0[0]) * ax[0] + (data[f"J1/{lab}|Q"] - z0[1]) * ax[1]) / (ax @ ax)).mean()  # noqa: E731
        best = max(((tau, a) for tau in TAUS for a in AMPS), key=lambda ta: proj(f"ch{ta[0]}_{ta[1]:g}"))
        meta["best_chirp"] = best
        print(f"best chirp setting: {best[0]} ns at {best[1] * FR_TOP / 1e6:g} MHz nominal (signal {proj(f'ch{best[0]}_{best[1]:g}'):.2f})", flush=True)

        s_off = np.arange(-15_000_000, 15_000_001, 250_000)
        c_off = np.arange(-15_000_000, 15_000_001, 2_500_000)
        with program() as prog:
            n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
            r0, r1, a, b = pair(), pair(), pair(), pair()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < 1000, n + 1):
                reset(); q.xy.update_frequency(IF0); ro(I, Q, r0)
                reset(); q.xy.update_frequency(IF0); q.xy.play("x180"); align(); ro(I, Q, r1)
                with for_(*from_array(f, (IF0 + s_off).astype(int))):
                    reset(); sat(f, None); ro(I, Q, a)
                with for_(*from_array(f, (IF0 + c_off - BAND // 2).astype(int))):
                    reset(); q.xy.update_frequency(f)
                    play(f"sT_ch{best[0]}" * amp(best[1]), el, chirp=(BAND // best[0], "Hz/nsec")); align(); ro(I, Q, b)
            with stream_processing():
                for nm, s in (("r0", r0), ("r1", r1)):
                    s[0].save_all(f"{nm}|I"); s[1].save_all(f"{nm}|Q")
                a[0].buffer(len(s_off)).average().save("a|I"); a[1].buffer(len(s_off)).average().save("a|Q")
                b[0].buffer(len(c_off)).average().save("b|I"); b[1].buffer(len(c_off)).average().save("b|Q")
        execute(qm, "J2", prog, ["r0|I", "r0|Q", "r1|I", "r1|Q", "a|I", "a|Q", "b|I", "b|Q"])
        meta.update(s_off=s_off.tolist(), c_off=c_off.tolist())

    meta["qpu_s"] = qpu
    np.savez(OUT / "shortT1_qD2_data.npz", **{k.replace("/", "__").replace("|", "_"): v for k, v in data.items()})
    (OUT / "shortT1_qD2_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
