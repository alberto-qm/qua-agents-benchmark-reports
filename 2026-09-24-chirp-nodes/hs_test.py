"""Hyperbolic-secant chirp against the node's linear chirp, short and long T1 (24 Sep 2026).

usage: hs_test.py <backend> <step>[,<step>...]      steps: ladder, ref, maps_chirp, maps_sat  (per qubit of the backend)

Qubits: gilboa qD2, qC3 (T1 1.3, 1.8 us), qD5 (43 us); arbel qA6 (5.2 us), qB4 (31 us).
Pulses, all of the node's sweep length (T1/5, at most 4 us):
  lin_up / lin_dn   the node's pulse: flat top with raised-cosine edges, linear sweep over 20 MHz, up / down
  hs_up  / hs_dn    hyperbolic-secant amplitude (beta 5), tanh frequency sweep over 40 MHz written into I/Q, up / down
ladder      each pulse at five drives x1/4..x4 of the amplitude the stored x180 predicts for a clean passage, against
            band centre (2.5 MHz steps near f01, 5 MHz further down to hold the 0->2 line), with |0> and x180 references
ref         fine saturation scan, +-10 MHz at 0.1 MHz, 0.3 MHz Rabi, 20 us (3 T1 up to 100 us) after a 5 T1 wait
maps_chirp  flux maps, 15 offsets over +-1.5x the offset that lowers f01 by 20 MHz (stored curvature), flux step leading the
            drive by 5 us: linear at the node's 03b drive (2x target), hyperbolic secant at its target
maps_sat    the same map with a 1 MHz-Rabi saturation drive (3 T1, 20-100 us) inside the flux step
Local state copies (gilboa: C/D qubits active); pulses in the generated config only; nothing written to any state.
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
OUT = HERE / "hs_test"
OUT.mkdir(exist_ok=True)
backend = sys.argv[1]
steps = sys.argv[2].split(",")
DRY = "--dry" in sys.argv
QUBITS = {"gilboa": ["qD2", "qC3", "qD5"], "arbel": ["qA6", "qB4"]}[backend]
STATE = HERE / backend / "state"
LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
BETA = 5.0
BANDS = {"lin": 20e6, "hs": 40e6}
IF_LIMIT = 395e6


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/hs] {msg}"
    print(line, flush=True)
    with open(OUT / f"driver_{backend}.log", "a") as fh:
        fh.write(line + "\n")


def main() -> int:
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled="--dry" not in sys.argv, backend=backend, default_timeout_s=60))
    from qm.qua import align, amp, declare, declare_stream, fixed, for_, for_each_, play, program, save, stream_processing
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    from calibration_utils.flux_range import check_flux_biases
    from calibration_utils.qubit_spectroscopy_chirp.chirp import envelope, rabi_per_amplitude, sweep_length_ns, target_rabi_hz

    machine = Quam.load(str(STATE))
    config = machine.generate_config()
    config["waveforms"]["hs_zero"] = {"type": "constant", "sample": 0.0}
    plan = {}
    for name in QUBITS:
        q = machine.qubits[name]
        el = q.xy.name
        IF = int(q.xy.intermediate_frequency)
        tau = sweep_length_ns(q.T1)
        kappa = rabi_per_amplitude(q)
        T1 = float(q.T1)
        ops = config["elements"][el]["operations"]
        marker = config["pulses"][ops["x180"]].get("digital_marker") if "x180" in ops else None
        pulses = {}
        for fam, band in BANDS.items():
            rate = int(round(band / tau))
            target = target_rabi_hz(rate) / kappa
            peak = min(0.99, target * max(LEVELS))
            t = np.arange(tau)
            for d in ("up", "dn"):
                if fam == "lin":
                    I, Q = peak * envelope(tau), np.zeros(tau)
                else:
                    x = BETA * (2 * t / (tau - 1) - 1)
                    df = (band / 2) * np.tanh(x) * (1 if d == "up" else -1)
                    ph = 2 * np.pi * np.cumsum(df) * 1e-9
                    I, Q = peak / np.cosh(x) * np.cos(ph), peak / np.cosh(x) * np.sin(ph)
                key = f"{fam}_{d}"
                wi, wq = f"hs_{key}_{el}_I", f"hs_{key}_{el}_Q"
                config["waveforms"][wi] = {"type": "arbitrary", "samples": I.tolist()}
                config["waveforms"][wq] = {"type": "arbitrary", "samples": Q.tolist()}
                pulse = {"operation": "control", "length": tau, "waveforms": {"I": wi, "Q": wq}}
                if marker is not None:
                    pulse["digital_marker"] = marker
                config["pulses"][f"hs_{key}_{el}_pulse"] = pulse
                ops[f"hs_{key}"] = f"hs_{key}_{el}_pulse"
                pulses[key] = dict(band=band, rate=rate if d == "up" else -rate, target=target, peak=peak,
                                   scales=[min(0.99, target * g) / peak for g in LEVELS],
                                   played=[min(0.99, target * g) for g in LEVELS])
        # saturation pulses (constant, stretched with duration)
        sat_ref_amp = min(0.99, 0.3e6 / kappa)
        sat_map_amp = min(0.99, 1.0e6 / kappa)
        for k, a in (("satref", sat_ref_amp), ("satmap", sat_map_amp)):
            w = f"hs_{k}_{el}"
            config["waveforms"][w] = {"type": "constant", "sample": float(a)}
            p = {"operation": "control", "length": 1000, "waveforms": {"I": w, "Q": "hs_zero"}}
            if marker is not None:
                p["digital_marker"] = marker
            config["pulses"][f"hs_{k}_{el}_pulse"] = p
            ops[f"hs_{k}"] = f"hs_{k}_{el}_pulse"
        sat_len = int(4 * round(min(100e-6, max(20e-6, 3 * T1)) * 1e9 / 4))
        # frequency windows (offsets from the stored f01), clipped to the upconverter band
        lo = max(-130e6, -IF_LIMIT - IF + 25e6)
        hi = min(60e6, IF_LIMIT - IF - 25e6)
        fine = np.arange(max(lo, -35e6), min(hi, 35e6) + 1, 2.5e6)
        coarse = np.arange(lo, fine.min() - 1, 5e6) if lo < fine.min() - 4e6 else np.array([])
        centres = np.concatenate([coarse, fine])
        quad = abs(float(q.freq_vs_flux_01_quad_term))
        half = 1.5 * np.sqrt(20e6 / quad)
        offsets = np.linspace(-half, half, 15)
        idle = float(q.z.joint_offset)
        check_flux_biases(q, idle + offsets, "hs test flux map")
        const_amp = float(q.z.operations["const"].amplitude)
        map_lin = np.arange(max(-40e6, lo), min(30e6, hi) + 1, 2.5e6)
        map_hs = np.arange(max(-55e6, lo), min(40e6, hi) + 1, 2.5e6)
        map_sat = np.arange(max(-25e6, lo), min(10e6, hi) + 1, 0.5e6)
        ref_df = np.arange(-10e6, 10e6 + 1, 0.1e6)
        short = T1 < 10e-6
        plan[name] = dict(el=el, IF=IF, tau=tau, kappa=kappa, T1=T1, f01=float(q.f_01), pulses=pulses, centres=centres.tolist(),
                          offsets=offsets.tolist(), z_scales=(offsets / const_amp).tolist(), idle=idle,
                          map_lin=map_lin.tolist(), map_hs=map_hs.tolist(), map_sat=map_sat.tolist(), ref_df=ref_df.tolist(),
                          sat_len=sat_len, sat_ref_amp=sat_ref_amp, sat_map_amp=sat_map_amp,
                          shots_ladder=200 if short else 100, shots_map=200 if short else 100,
                          shots_sat=100 if short else 50, shots_ref=300 if short else 200,
                          map_drive={"lin_up": min(0.99, 2 * pulses["lin_up"]["target"]), "hs_up": min(0.99, pulses["hs_up"]["target"])})
        # every pulse spans centre +- band/2 whatever its direction: keep the whole span inside the upconverter band
        for arr in (centres, map_lin, map_hs, map_sat):
            assert IF + arr.min() - 20e6 > -400e6 and IF + arr.max() + 20e6 < 400e6, (name, arr.min(), arr.max())
        log(f"{name}: T1 {T1 * 1e6:.1f} us, sweep {tau} ns, IF {IF / 1e6:.0f} MHz, window {centres.min() / 1e6:.0f}..{centres.max() / 1e6:.0f} MHz "
            f"({len(centres)} centres), targets lin {pulses['lin_up']['target']:.3f} hs {pulses['hs_up']['target']:.3f}")

    def refs(q, I, Q, r0, r1):
        for k, st in (("r0", r0), ("r1", r1)):
            q.reset_qubit_thermal()
            align()
            q.xy.update_frequency(plan[q.name]["IF"])
            if k == "r1":
                q.xy.play("x180")
            align()
            q.resonator.measure("readout", qua_vars=(I, Q))
            save(I, st[0])
            save(Q, st[1])
            align()

    def pair():
        return declare_stream(), declare_stream()

    def play_chirp(q, key, scale, f):
        p = plan[q.name]["pulses"][key]
        if key.startswith("lin"):
            q.xy.update_frequency(f - int(p["band"] // 2) if p["rate"] > 0 else f + int(p["band"] // 2))
            play(f"hs_{key}" * amp(scale), plan[q.name]["el"], chirp=(p["rate"], "Hz/nsec"))
        else:
            q.xy.update_frequency(f)
            play(f"hs_{key}" * amp(scale), plan[q.name]["el"])

    def execute(tag, prog, keys, qm):
        t0 = time.perf_counter()
        if DRY:
            from qm import generate_qua_script
            generate_qua_script(prog, config)
            log(f"{tag}: built ({len(keys)} streams)")
            return 0.0
        job = qm.execute(prog)
        job.result_handles.wait_for_all_values()
        qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
        data = {}
        for k in keys:
            raw = job.result_handles.get(k).fetch_all()
            try:
                raw = raw["value"]
            except (TypeError, IndexError, KeyError, ValueError):
                pass
            data[k] = np.asarray(raw, dtype=float)
        np.savez(OUT / f"{backend}_{tag}.npz", **data)
        log(f"{tag}: QPU {qpu:.1f} s, wall {time.perf_counter() - t0:.0f} s")
        return qpu

    def est(n_shots, per_shot_s):
        return n_shots * per_shot_s * 1.3

    qpu = {}
    import contextlib
    session = contextlib.nullcontext(None) if DRY else qm_session(machine.connect(), config, timeout=120)
    with session as qm:
        for step in steps:
            for name in QUBITS:
                q = machine.qubits[name]
                p = plan[name]
                wait = 5 * p["T1"]
                if step == "ladder":
                    cen = (p["IF"] + np.asarray(p["centres"])).astype(int).tolist()
                    n_sh = p["shots_ladder"]
                    e = est(4 * len(LEVELS) * len(cen) * n_sh, wait + p["tau"] * 1e-9 + 11e-6)
                    assert e < 50, (name, step, e)
                    keys = []
                    with program() as prog:
                        n = declare(int)
                        f = declare(int)
                        I = declare(fixed)
                        Q = declare(fixed)
                        r0, r1 = pair(), pair()
                        st = {(k, j): pair() for k in p["pulses"] for j in range(len(LEVELS))}
                        machine.initialize_qpu(target=q)
                        align()
                        with for_(n, 0, n < n_sh, n + 1):
                            refs(q, I, Q, r0, r1)
                            for k in p["pulses"]:
                                for j, s in enumerate(p["pulses"][k]["scales"]):
                                    with for_each_(f, cen):
                                        q.reset_qubit_thermal()
                                        align()
                                        play_chirp(q, k, float(s), f)
                                        align()
                                        q.resonator.measure("readout", qua_vars=(I, Q))
                                        save(I, st[(k, j)][0])
                                        save(Q, st[(k, j)][1])
                                        align()
                        with stream_processing():
                            for nm, s_ in (("r0", r0), ("r1", r1)):
                                s_[0].average().save(f"{nm}_I")
                                s_[1].average().save(f"{nm}_Q")
                                keys += [f"{nm}_I", f"{nm}_Q"]
                            for (k, j), s_ in st.items():
                                s_[0].buffer(len(cen)).average().save(f"{k}_{j}_I")
                                s_[1].buffer(len(cen)).average().save(f"{k}_{j}_Q")
                                keys += [f"{k}_{j}_I", f"{k}_{j}_Q"]
                    qpu[f"{name}_ladder"] = execute(f"{name}_ladder", prog, keys, qm)
                elif step == "ref":
                    fr = (p["IF"] + np.asarray(p["ref_df"])).astype(int).tolist()
                    n_sh = p["shots_ref"]
                    assert est(len(fr) * n_sh, wait + p["sat_len"] * 1e-9 + 11e-6) < 50
                    with program() as prog:
                        n = declare(int)
                        f = declare(int)
                        I = declare(fixed)
                        Q = declare(fixed)
                        r0, r1, a = pair(), pair(), pair()
                        machine.initialize_qpu(target=q)
                        align()
                        with for_(n, 0, n < n_sh, n + 1):
                            refs(q, I, Q, r0, r1)
                            with for_each_(f, fr):
                                q.reset_qubit_thermal()
                                align()
                                q.xy.update_frequency(f)
                                play("hs_satref", p["el"], duration=p["sat_len"] // 4)
                                align()
                                q.resonator.measure("readout", qua_vars=(I, Q))
                                save(I, a[0])
                                save(Q, a[1])
                                align()
                        with stream_processing():
                            for nm, s_ in (("r0", r0), ("r1", r1)):
                                s_[0].average().save(f"{nm}_I")
                                s_[1].average().save(f"{nm}_Q")
                            a[0].buffer(len(fr)).average().save("a_I")
                            a[1].buffer(len(fr)).average().save("a_Q")
                    qpu[f"{name}_ref"] = execute(f"{name}_ref", prog, ["r0_I", "r0_Q", "r1_I", "r1_Q", "a_I", "a_Q"], qm)
                elif step in ("maps_chirp", "maps_sat"):
                    kinds = ["lin_up", "hs_up"] if step == "maps_chirp" else ["sat"]
                    for kind in kinds:
                        if kind == "sat":
                            fr, n_sh, drive_ns = p["map_sat"], p["shots_sat"], p["sat_len"]
                        else:
                            fr = p["map_lin"] if kind == "lin_up" else p["map_hs"]
                            n_sh, drive_ns = p["shots_map"], p["tau"]
                        fq = (p["IF"] + np.asarray(fr)).astype(int).tolist()
                        tail = 5000 if kind == "sat" else int(4 * round(min(1000, max(100, p["T1"] * 1e9 / 10)) / 4))
                        zlen = 5000 + drive_ns + tail
                        D = len(p["offsets"])
                        e = est(len(fq) * D * n_sh, wait + zlen * 1e-9 + 11e-6)
                        assert e < 50, (name, kind, e)
                        with program() as prog:
                            n = declare(int)
                            f = declare(int)
                            j = declare(int)
                            I = declare(fixed)
                            Q = declare(fixed)
                            zs = declare(fixed, value=[float(v) for v in p["z_scales"]])
                            a = pair()
                            machine.initialize_qpu(target=q)
                            align()
                            with for_(n, 0, n < n_sh, n + 1):
                                with for_each_(f, fq):
                                    with for_(j, 0, j < D, j + 1):
                                        q.reset_qubit_thermal()
                                        align()
                                        q.z.play("const", amplitude_scale=zs[j], duration=zlen // 4)
                                        q.xy.wait(5000 // 4)
                                        if kind == "sat":
                                            q.xy.update_frequency(f)
                                            play("hs_satmap", p["el"], duration=drive_ns // 4)
                                        else:
                                            pp = p["pulses"][kind]
                                            play_chirp(q, kind, float(p["map_drive"][kind] / pp["peak"]), f)
                                        align()
                                        q.resonator.measure("readout", qua_vars=(I, Q))
                                        save(I, a[0])
                                        save(Q, a[1])
                                        align()
                            with stream_processing():
                                a[0].buffer(D).buffer(len(fq)).average().save("a_I")
                                a[1].buffer(D).buffer(len(fq)).average().save("a_Q")
                        qpu[f"{name}_map_{kind}"] = execute(f"{name}_map_{kind}", prog, ["a_I", "a_Q"], qm)
    old = json.loads((OUT / f"{backend}_meta.json").read_text()) if (OUT / f"{backend}_meta.json").exists() else {}
    old_qpu = old.get("qpu_s", {})
    old_qpu.update(qpu)
    (OUT / f"{backend}_meta.json").write_text(json.dumps(dict(plan=plan, levels=list(LEVELS), beta=BETA, qpu_s=old_qpu), indent=1,
                                                         default=float))
    log("done: " + ",".join(steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
