"""Chirp vs saturation on arbel qB4, the qubit where saturation spectroscopy repeatedly locked onto the 0->2 line.

J1a saturation spectrum at the node's default drive (0.5 x stored saturation amplitude 1.0, 20 us) over
    f01-125 .. f01+25 MHz (0.5 MHz steps): the view the agents had.
J1b chirp band-centre scan (4 us, +-10 MHz, ~3 MHz nominal Rabi) over the same range (2.5 MHz steps).
J2  saturation drive ladder (nominal Rabi 1/2/4/8/16 MHz + the node default), +-3 MHz fine scans around the 0->2 line.
J3  chirp ladder (nominal Rabi 0.5..16 MHz) on f01 and on the 0->2 line; saturation on f01 at the ladder drives.
Nominal Rabi from qB4's calibrated x180 (48 ns cosine, amplitude from the lab state). Flux at idle (initialize_qpu),
5xT1 thermal reset every shot, pulses injected in the generated config only, nothing written to the state.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

STATE = Path("/Users/atosato/qab-runs/reference-state-20260922/arbel")
OUT = Path(__file__).parent
QUBIT = "qB4"
TAU, RAMP, BAND = 4000, 500, 20_000_000
RATE = BAND // TAU
FR_TOP = 16e6
LADDER = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)          # x FR_TOP -> 1, 2, 4, 8, 16 MHz
CHIRP_LADDER = (1 / 32,) + LADDER                     # 0.5 .. 16 MHz
FR_SCAN = 3e6


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=True, backend="arbel", default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    machine = Quam.load()
    q = machine.qubits[QUBIT]
    el = q.xy.name
    IF0 = int(q.xy.intermediate_frequency)
    alpha = float(q.anharmonicity)
    pi = q.xy.operations["x180_DragCosine"]
    fr_per_amp = (1e9 / pi.length) / pi.amplitude
    a_top = FR_TOP / fr_per_amp
    node_fr = 0.5 * float(q.xy.operations["saturation"].amplitude) * fr_per_amp
    lo, hi = IF0 - 125_000_000, IF0 + 25_000_000
    assert abs(lo) < 400e6 and abs(hi) < 400e6, (lo, hi)
    print(f"{QUBIT}: f01 {q.f_01/1e6:.2f} MHz, IF {IF0/1e6:.2f} MHz, alpha {alpha/1e6:.1f} MHz, T1 {q.T1*1e6:.1f} us; "
          f"node default drive = {node_fr/1e6:.1f} MHz nominal Rabi; ladder amplitude at 16 MHz = {a_top:.3f}", flush=True)

    config = machine.generate_config()
    marker = config["pulses"][config["elements"][el]["operations"]["saturation"]].get("digital_marker", "ON")
    env = np.ones(TAU); r = 0.5 * (1 - np.cos(np.pi * np.arange(RAMP) / RAMP)); env[:RAMP], env[-RAMP:] = r, r[::-1]
    config["waveforms"].update({"tr_zero": {"type": "constant", "sample": 0.0},
                                "tr_sat_I": {"type": "constant", "sample": a_top},
                                "tr_chirp_I": {"type": "arbitrary", "samples": (a_top * env).tolist()}})
    config["pulses"].update({
        "tr_sat_pulse": {"operation": "control", "length": 20000, "waveforms": {"I": "tr_sat_I", "Q": "tr_zero"}, "digital_marker": marker},
        "tr_chirp_pulse": {"operation": "control", "length": TAU, "waveforms": {"I": "tr_chirp_I", "Q": "tr_zero"}, "digital_marker": marker}})
    config["elements"][el]["operations"].update({"tr_sat": "tr_sat_pulse", "tr_chirp": "tr_chirp_pulse"})

    def reset():
        q.reset_qubit_thermal(); align()

    def ro(I, Q, st):
        q.resonator.measure("readout", qua_vars=(I, Q)); save(I, st[0]); save(Q, st[1]); align()

    def refs(I, Q, r0, r1):
        reset(); q.xy.update_frequency(IF0); ro(I, Q, r0)
        reset(); q.xy.update_frequency(IF0); q.xy.play("x180"); align(); ro(I, Q, r1)

    def sat(freq, scale):            # scale None = the node's own drive: stored saturation op at 0.5, 20 us
        q.xy.update_frequency(freq)
        if scale is None:
            q.xy.play("saturation", amplitude_scale=0.5, duration=5000)
        else:
            play("tr_sat" * amp(scale), el)
        align()

    def chirp(centre, scale):
        q.xy.update_frequency(centre - BAND // 2); play("tr_chirp" * amp(scale), el, chirp=(RATE, "Hz/nsec")); align()

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

    REF_KEYS = ["r0|I", "r0|Q", "r1|I", "r1|Q"]

    def ref_streams(r0, r1):
        for nm, s in (("r0", r0), ("r1", r1)):
            s[0].save_all(f"{nm}|I"); s[1].save_all(f"{nm}|Q")

    def spectrum_prog(freqs, n_shots, body):
        with program() as prog:
            n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
            r0, r1, a = pair(), pair(), pair()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < n_shots, n + 1):
                refs(I, Q, r0, r1)
                with for_(*from_array(f, freqs.astype(int))):
                    reset(); body(f); ro(I, Q, a)
            with stream_processing():
                ref_streams(r0, r1)
                a[0].buffer(len(freqs)).average().save("a|I"); a[1].buffer(len(freqs)).average().save("a|Q")
        return prog

    sat_off = np.arange(-125_000_000, 25_000_001, 500_000)
    chirp_off = np.arange(-125_000_000, 25_000_001, 2_500_000)
    meta = dict(qubit=QUBIT, IF0=IF0, alpha=alpha, t1=float(q.T1), fr_per_amp=fr_per_amp, a_top=a_top, fr_top=FR_TOP,
                node_fr=node_fr, rate_hz_per_ns=RATE, tau_ns=TAU, sat_off=sat_off.tolist(), chirp_off=chirp_off.tolist(),
                ladder=list(LADDER), chirp_ladder=list(CHIRP_LADDER), fr_scan=FR_SCAN, f01_hz=float(q.f_01))
    with qm_session(machine.connect(), config, timeout=60) as qm:
        execute(qm, "J1a", spectrum_prog(IF0 + sat_off, 60, lambda f: sat(f, None)), REF_KEYS + ["a|I", "a|Q"])
        execute(qm, "J1b", spectrum_prog(IF0 + chirp_off - BAND // 2, 150,
                                         lambda f: (q.xy.update_frequency(f),
                                                    play("tr_chirp" * amp(FR_SCAN / FR_TOP), el, chirp=(RATE, "Hz/nsec")),
                                                    align())), REF_KEYS + ["a|I", "a|Q"])
        # locate the 0->2 line in the saturation spectrum, 70..125 MHz below f01
        z0 = np.array([data["J1a/r0|I"].mean(), data["J1a/r0|Q"].mean()])
        ax = np.array([data["J1a/r1|I"].mean(), data["J1a/r1|Q"].mean()]) - z0
        dev = np.hypot(data["J1a/a|I"] - z0[0], data["J1a/a|Q"] - z0[1]) / np.linalg.norm(ax)
        win = (sat_off <= -70_000_000)
        i = int(np.argmax(np.where(win, dev, -1)))
        f2 = int(IF0 + sat_off[i])
        meta.update(f2ph_off=int(sat_off[i]), f2ph_dev=float(dev[i]), alpha_measured=float(2 * -sat_off[i]))
        print(f"0->2 line (strongest point 70..125 MHz below f01): f01 {sat_off[i]/1e6:+.1f} MHz, signal {dev[i]:.2f} "
              f"-> alpha ~ {2 * -sat_off[i] / 1e6:.1f} MHz", flush=True)

        f_off = np.arange(-3_000_000, 3_000_001, 100_000)
        drives = list(LADDER) + [None]
        with program() as prog:
            n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
            r0, r1 = pair(), pair(); st = [pair() for _ in drives]
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < 150, n + 1):
                refs(I, Q, r0, r1)
                for j, s in enumerate(drives):
                    with for_(*from_array(f, (f2 + f_off).astype(int))):
                        reset(); sat(f, s); ro(I, Q, st[j])
            with stream_processing():
                ref_streams(r0, r1)
                for j, p in enumerate(st):
                    p[0].buffer(len(f_off)).average().save(f"s{j}|I"); p[1].buffer(len(f_off)).average().save(f"s{j}|Q")
        execute(qm, "J2", prog, REF_KEYS + [f"s{j}|{c}" for j in range(len(drives)) for c in "IQ"])
        meta["f_off"] = f_off.tolist()

        labels = ([(f"c2_{j}", "c", f2, s) for j, s in enumerate(CHIRP_LADDER)]
                  + [(f"c1_{j}", "c", IF0, s) for j, s in enumerate(CHIRP_LADDER)]
                  + [(f"s1_{j}", "s", IF0, s) for j, s in enumerate(drives)])
        with program() as prog:
            n = declare(int); I = declare(fixed); Q = declare(fixed)
            r0, r1 = pair(), pair(); st = {lab: pair() for lab, *_ in labels}
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < 600, n + 1):
                refs(I, Q, r0, r1)
                for lab, kind, fr, s in labels:
                    reset()
                    if kind == "c":
                        chirp(fr, s)
                    else:
                        sat(fr, s)
                    ro(I, Q, st[lab])
            with stream_processing():
                ref_streams(r0, r1)
                for lab, p in st.items():
                    p[0].average().save(f"{lab}|I"); p[1].average().save(f"{lab}|Q")
        execute(qm, "J3", prog, REF_KEYS + [f"{lab}|{c}" for lab, *_ in labels for c in "IQ"])

    meta["qpu_s"] = qpu
    np.savez(OUT / "trap_qB4_data.npz", **{k.replace("/", "__").replace("|", "_"): v for k, v in data.items()})
    (OUT / "trap_qB4_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
