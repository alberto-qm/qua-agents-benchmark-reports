"""Does a chirp ignore the 0->2 two-photon line that saturation spectroscopy shows? (qolab Q1)

Q1's stored anharmonicity is a placeholder (215.5 MHz on all six qolab qubits), so:
  S  search: saturation (20 us, ~16 MHz nominal Rabi, arbel-like) over f01-140 .. f01-70 MHz, 0.5 MHz steps.
     The strongest line there is taken as the 0->2 two-photon line; it also measures alpha = 2 (f01 - f_2ph).
     Stops here unless the line stands >= 6 sigma above the local median.
  L1 saturation ladder: nominal Rabi 1/2/4/8/16 MHz, each a +-3 MHz scan (0.1 MHz steps) around the found line.
  L2 chirp ladder (4 us, +-10 MHz, 5000 Hz/ns) at the same strengths, centred on the 0->2 line and on f01;
     saturation at f01 at the same strengths for reference.
Signal: IQ deviation from |0>, in units of the |0>-|1> separation (|2> may read out off the 0-1 axis), plus the
projection on the 0-1 axis. Every shot starts from the node's 5xT1 thermal reset; flux at idle. No state written.
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
FR_TOP = 16e6                       # nominal Rabi at scale 1 for both pulses
SCALES = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
N_S, N_L1, N_L2, N_REF = 200, 200, 1000, 1000


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
    a_top = FR_TOP / fr_per_amp

    config = machine.generate_config()
    marker = config["pulses"][config["elements"][el]["operations"]["saturation"]].get("digital_marker", "ON")
    env = np.ones(TAU)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(RAMP) / RAMP))
    env[:RAMP], env[-RAMP:] = ramp, ramp[::-1]
    config["waveforms"].update({"tp_zero": {"type": "constant", "sample": 0.0},
                                "tp_sat_I": {"type": "constant", "sample": a_top},
                                "tp_chirp_I": {"type": "arbitrary", "samples": (a_top * env).tolist()}})
    config["pulses"].update({
        "tp_sat_pulse": {"operation": "control", "length": 20000, "waveforms": {"I": "tp_sat_I", "Q": "tp_zero"},
                         "digital_marker": marker},
        "tp_chirp_pulse": {"operation": "control", "length": TAU, "waveforms": {"I": "tp_chirp_I", "Q": "tp_zero"},
                           "digital_marker": marker}})
    config["elements"][el]["operations"].update({"tp_sat": "tp_sat_pulse", "tp_chirp": "tp_chirp_pulse"})
    print(f"Q1 IF {IF0/1e6:.2f} MHz; pulse amplitude at scale 1 = {a_top:.4f} (nominal Rabi {FR_TOP/1e6:.0f} MHz)", flush=True)

    def reset():
        q.reset_qubit_thermal(); align()

    def ro(I, Q, st):
        q.resonator.measure("readout", qua_vars=(I, Q)); save(I, st[0]); save(Q, st[1]); align()

    def refs(I, Q, r0, r1):
        reset(); q.xy.update_frequency(IF0); ro(I, Q, r0)
        reset(); q.xy.update_frequency(IF0); q.xy.play("x180"); align(); ro(I, Q, r1)

    def sat(freq, scale):
        q.xy.update_frequency(freq); play("tp_sat" * amp(scale), el); align()

    def chirp(centre, scale):
        q.xy.update_frequency(centre - BAND // 2); play("tp_chirp" * amp(scale), el, chirp=(RATE, "Hz/nsec")); align()

    def get(job, name):
        raw = job.result_handles.get(name).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        return np.asarray(raw, dtype=float).ravel()

    def pair():
        return declare_stream(), declare_stream()

    data, qpu = {}, {}

    def run(qm, name, prog, keys):
        t0 = time.perf_counter()
        job = qm.execute(prog); job.result_handles.wait_for_all_values()
        qpu[name] = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
        for k in keys:
            data[f"{name}/{k}"] = get(job, k)
        print(f"{name}: {qpu[name]:.1f} s QPU (wall {time.perf_counter() - t0:.1f} s)", flush=True)

    def dev(name, a_key):
        """IQ deviation from ref0, in |0>-|1> separation units: (total, along 0-1 axis)."""
        z0 = np.array([data[f"{name}/r0|{c}"].mean() for c in "IQ"])
        ax = np.array([data[f"{name}/r1|{c}"].mean() for c in "IQ"]) - z0
        L = np.linalg.norm(ax)
        D = np.stack([data[f"{name}/{a_key}|I"], data[f"{name}/{a_key}|Q"]], 1) - z0
        noise = (np.stack([data[f"{name}/r0|I"], data[f"{name}/r0|Q"]], 1) - z0) @ (ax / L) / L
        return np.hypot(*D.T) / L, D @ (ax / L) / L, noise.std()

    lo_mhz, hi_mhz = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (-140, -70)
    s_off = np.arange(lo_mhz * 1_000_000, hi_mhz * 1_000_000 + 1, 500_000)
    with qm_session(machine.connect(), config, timeout=60) as qm:
        # ---- S: search
        with program() as prog:
            n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
            r0, r1, a = pair(), pair(), pair()
            machine.initialize_qpu(target=q); align()
            with for_(n, 0, n < N_S, n + 1):
                refs(I, Q, r0, r1)
                with for_(*from_array(f, (IF0 + s_off).astype(int))):
                    reset(); sat(f, 1.0); ro(I, Q, a)
            with stream_processing():
                for nm, st in (("r0", r0), ("r1", r1)):
                    st[0].save_all(f"{nm}|I"); st[1].save_all(f"{nm}|Q")
                a[0].buffer(len(s_off)).average().save("a|I"); a[1].buffer(len(s_off)).average().save("a|Q")
        run(qm, "S", prog, ["r0|I", "r0|Q", "r1|I", "r1|Q", "a|I", "a|Q"])
        tot, par, sig1 = dev("S", "a")
        err = sig1 / np.sqrt(N_S)
        base = np.array([np.median(tot[max(0, i - 10): i + 11]) for i in range(len(tot))])
        i = int(np.argmax(tot - base))
        snr = (tot[i] - base[i]) / err
        f2ph = int(IF0 + s_off[i])
        alpha = 2 * (IF0 - f2ph)
        print(f"search: strongest line at f01 {s_off[i]/1e6:+.1f} MHz, deviation {tot[i]:.3f} over baseline {base[i]:.3f} "
              f"({snr:.1f} sigma) -> alpha ~ {alpha/1e6:.1f} MHz", flush=True)
        meta = dict(IF0=IF0, a_top=a_top, fr_top=FR_TOP, rate_hz_per_ns=RATE, tau_ns=TAU, scales=list(SCALES),
                    s_off=s_off.tolist(), search_peak_off=int(s_off[i]), search_snr=float(snr), alpha_measured=alpha,
                    f2ph_if=f2ph, placeholder_alpha=float(q.anharmonicity))
        if snr < 6:
            print("no line >= 6 sigma in the search window: stopping before the ladders", flush=True)
            meta["stopped"] = "no line in search"
        else:
            # ---- L1: saturation ladder, fine scans around the line
            f_off = np.arange(-3_000_000, 3_000_001, 100_000)
            with program() as prog:
                n = declare(int); f = declare(int); I = declare(fixed); Q = declare(fixed)
                r0, r1 = pair(), pair()
                st = [pair() for _ in SCALES]
                machine.initialize_qpu(target=q); align()
                with for_(n, 0, n < N_L1, n + 1):
                    refs(I, Q, r0, r1)
                    for j, s in enumerate(SCALES):
                        with for_(*from_array(f, (f2ph + f_off).astype(int))):
                            reset(); sat(f, s); ro(I, Q, st[j])
                with stream_processing():
                    for nm, p in (("r0", r0), ("r1", r1)):
                        p[0].save_all(f"{nm}|I"); p[1].save_all(f"{nm}|Q")
                    for j, p in enumerate(st):
                        p[0].buffer(len(f_off)).average().save(f"sat{j}|I"); p[1].buffer(len(f_off)).average().save(f"sat{j}|Q")
            run(qm, "L1", prog, ["r0|I", "r0|Q", "r1|I", "r1|Q"] + [f"sat{j}|{c}" for j in range(len(SCALES)) for c in "IQ"])
            meta["f_off"] = f_off.tolist()

            # ---- L2: chirp ladder at the line and at f01, saturation at f01
            labels = ([(f"chirp2ph{j}", "c2", s) for j, s in enumerate(SCALES)]
                      + [(f"chirp01_{j}", "c1", s) for j, s in enumerate(SCALES)]
                      + [(f"sat01_{j}", "s1", s) for j, s in enumerate(SCALES)])
            with program() as prog:
                n = declare(int); I = declare(fixed); Q = declare(fixed)
                r0, r1 = pair(), pair()
                st = {lab: pair() for lab, _, _ in labels}
                machine.initialize_qpu(target=q); align()
                with for_(n, 0, n < N_L2, n + 1):
                    refs(I, Q, r0, r1)
                    for lab, kind, s in labels:
                        reset()
                        if kind == "c2":
                            chirp(f2ph, s)
                        elif kind == "c1":
                            chirp(IF0, s)
                        else:
                            sat(IF0, s)
                        ro(I, Q, st[lab])
                with stream_processing():
                    for nm, p in (("r0", r0), ("r1", r1)):
                        p[0].save_all(f"{nm}|I"); p[1].save_all(f"{nm}|Q")
                    for lab, p in st.items():
                        p[0].save_all(f"{lab}|I"); p[1].save_all(f"{lab}|Q")
            run(qm, "L2", prog, ["r0|I", "r0|Q", "r1|I", "r1|Q"] + [f"{lab}|{c}" for lab, _, _ in labels for c in "IQ"])
            meta["labels"] = [lab for lab, _, _ in labels]

    meta["qpu_s"] = qpu
    tag = f"_{lo_mhz}_{hi_mhz}"
    np.savez(OUT / f"twophoton_data{tag}.npz", **{k.replace("/", "__").replace("|", "_"): v for k, v in data.items()})
    (OUT / f"twophoton_meta{tag}.json").write_text(json.dumps(meta, indent=1))
    print("saved", OUT / f"twophoton_data{tag}.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
