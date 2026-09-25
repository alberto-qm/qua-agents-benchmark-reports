"""Resonator ring-down, a fine complex frequency sweep and a pump-probe, per qubit (25 Sep 2026).

usage: kappa.py <backend> <round> [--dry] [--two-demods]

Three jobs per qubit, each on a fresh copy of the qubit's state with its readout port set up as 02b sets
it for the top of its sweep (set_output_power at the top power, 0.1 amplitude cap), lower powers played
as amplitude scales of that:

  ringdown  a 2 us constant drive at the stored readout frequency followed by 2.48 us of silence, one
            measurement window over both, demodulated on the FPGA in 40 ns slices; three powers (committed
            - 10 dB, committed, top of 02b's sweep). The silent part is the field leaking out of the
            resonator: it decays as exp(-kappa t / 2) and rotates at the resonator-drive detuning, which
            gives kappa in the time domain, without any lineshape model.
  sweep     the committed readout pulse over +-10 MHz in 40 kHz steps, 10 us between points, at committed
            - 10 dB and committed: complex S21 for the circle fit and a Lorentzian fit to |S21|^2.
  pump      the readout pulse at the top of 02b's sweep, then a delay tau (16 ns .. 80 us), then the
            committed readout; plus ground (no pump) and x180 references. How excited a top-power readout
            leaves the qubit, and for how long -- what the next point of a 02b sweep inherits.

400 us before every shot. State: pulled 18:03 (~/qab-runs/state-20260925-1803; gilboa's C/D qubits marked
active in the local copy). Nothing written to any state. Outputs in kappa/<backend>/.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WT = Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
HERE = Path(__file__).resolve().parent
backend, rnd = sys.argv[1], int(sys.argv[2])
DRY = "--dry" in sys.argv
FOUR_DEMODS = "--two-demods" not in sys.argv
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
if ONLY:
    QUBITS = [q for q in QUBITS if q in ONLY]
FROZEN = {"arbel": {"3540", "3546", "3658", "3660"}, "gilboa": {"17418", "17429", "17482", "17483"}, "qolab": set()}[backend]
GILBOA_ACTIVE = ["qC1", "qC2", "qC3", "qC4", "qC5", "qD1", "qD2", "qD3", "qD4", "qD5"]
SNAP = Path.home() / "qab-runs/state-20260925-1803" / backend
OUT = HERE / "kappa" / backend
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "state"
STATE.mkdir(exist_ok=True)
st = json.loads((SNAP / "state.json").read_text())
if backend == "gilboa":
    st["active_qubit_names"] = GILBOA_ACTIVE
(STATE / "state.json").write_text(json.dumps(st))
shutil.copy(SNAP / "wiring.json", STATE / "wiring.json")

RESET_NS = 400_000
CHUNK_CYCLES = 10                      # 40 ns slices
L_ON, L_OFF = 2000, 2480               # 4480 ns window = 112 slices
N_CHUNKS = (L_ON + L_OFF) // (4 * CHUNK_CYCLES)
SHOTS_RD, SHOTS_SW, SHOTS_PP = 5000, 400, 1500
SWEEP_HZ = np.arange(-10e6, 10e6 + 1, 40e3).astype(int)
SWEEP_WAIT_NS = 10_000
TAUS_NS = [16, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 40000, 80000]
TOP_CEILING_DBM, SPAN_DB, CAP = -10, 35, 0.1


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/kappa-r{rnd}] {msg}"
    print(line, flush=True)
    with open(OUT / "driver.log", "a") as fh:
        fh.write(line + "\n")


def wait_until_free():
    while not DRY:
        r = subprocess.run(["uv", "run", "--frozen", "python", str(Path.home() / "qab-runs/backend_busy.py"), backend],
                           capture_output=True, text=True, cwd=Path.home() / "code/QM/qua-agents-benchmark")
        pids = {line.split()[2] for line in r.stdout.splitlines() if line.strip().startswith("busy: pid")}
        if not (pids - FROZEN):
            return
        log(f"backend busy ({sorted(pids - FROZEN)}); waiting 60 s")
        time.sleep(60)


def fetch(job, keys):
    data = {}
    for k in keys:
        raw = job.result_handles.get(k).fetch_all()
        try:
            raw = raw["value"]
        except (TypeError, IndexError, KeyError, ValueError):
            pass
        data[k] = np.asarray(raw, dtype=float)
    return data


def inject_ringdown(config, element, peak):
    """A measurement pulse: L_ON of constant drive, then L_OFF of zeros, one window over both."""
    L = L_ON + L_OFF
    config["waveforms"]["rd_zero"] = {"type": "constant", "sample": 0.0}
    config["waveforms"][f"rd_{element}_I"] = {"type": "arbitrary",
                                             "samples": [float(peak)] * L_ON + [0.0] * L_OFF}
    for name, (c, s) in {"rd_cos": (1.0, 0.0), "rd_sin": (0.0, 1.0), "rd_msin": (0.0, -1.0)}.items():
        config["integration_weights"][name] = {"cosine": [(c, L)], "sine": [(s, L)]}
    config["pulses"][f"rd_{element}_pulse"] = {
        "operation": "measurement", "length": L, "digital_marker": "ON",
        "waveforms": {"I": f"rd_{element}_I", "Q": "rd_zero"},
        "integration_weights": {"rd_cos": "rd_cos", "rd_sin": "rd_sin", "rd_msin": "rd_msin"}}
    config["elements"][element]["operations"]["ringdown"] = f"rd_{element}_pulse"


def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=not DRY, backend=backend, default_timeout_s=120))
    from qm.qua import (align, amp, declare, declare_stream, demod, fixed, for_, for_each_, measure, program, save,
                        stream_processing, update_frequency, wait)
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    mf = OUT / f"r{rnd}_meta.json"
    meta = json.loads(mf.read_text()) if (ONLY and mf.exists()) else {}
    for name in QUBITS:
        machine = Quam.load(str(STATE))
        q = machine.qubits[name]
        rr = q.resonator
        p_ro = float(rr.get_output_power("readout"))
        top = int(min(TOP_CEILING_DBM, round(p_ro + 20)))
        bottom = top - SPAN_DB
        fs_before = getattr(rr.opx_output, "full_scale_power_dbm", None)
        amp_before = float(rr.operations["readout"].amplitude)
        rr.set_output_power(power_in_dbm=top, max_amplitude=CAP)       # as 02b sets it for its sweep
        a_top = float(rr.operations["readout"].amplitude)
        fs = getattr(rr.opx_output, "full_scale_power_dbm", None)

        def scale(p):
            return float(10 ** ((p - top) / 20))

        levels = {"low": p_ro - 10, "ro": p_ro, "top": float(top)}
        rd_scales = [scale(p) for p in levels.values()]
        sw_scales = [scale(p_ro - 10), scale(p_ro)]
        ro_scale = scale(p_ro)
        assert max(rd_scales + sw_scales) < 1.99 and a_top * max(rd_scales + [1.0]) < 0.5, (name, rd_scales, a_top)
        IF = int(rr.intermediate_frequency)
        assert abs(IF) + 10e6 < 400e6
        config = machine.generate_config()
        inject_ringdown(config, rr.name, a_top)
        cycles = [max(4, t // 4) for t in TAUS_NS]
        T1 = float(q.T1) if q.T1 else None
        info = dict(committed_power_dbm=p_ro, top_dbm=top, bottom_dbm=bottom, full_scale_before=fs_before,
                    full_scale=fs, readout_amp_before=amp_before, readout_amp_at_top=a_top, levels_dbm=levels,
                    rd_scales=rd_scales, sw_scales=sw_scales, ro_scale=ro_scale, IF=IF,
                    f_r=float(rr.RF_frequency), readout_len=int(rr.operations["readout"].length),
                    tof=int(rr.time_of_flight), depletion_time=int(rr.depletion_time), T1=T1, taus_ns=TAUS_NS,
                    sweep_hz=SWEEP_HZ.tolist(), chunk_ns=4 * CHUNK_CYCLES, l_on=L_ON, l_off=L_OFF,
                    four_demods=FOUR_DEMODS, shots=dict(rd=SHOTS_RD, sw=SHOTS_SW, pp=SHOTS_PP), reset_ns=RESET_NS,
                    state=str(SNAP))
        log(f"{name}: committed {p_ro:.1f} dBm, 02b window {bottom}..{top} dBm, FS {fs_before}->{fs}, "
            f"readout amp at top {a_top:.4f}, scales rd {np.round(rd_scales, 4).tolist()} ro {ro_scale:.4f}")

        progs = {}
        # --- ring-down
        with program() as prog:
            n = declare(int)
            lv = declare(fixed)
            k = declare(int)
            arrs = [declare(fixed, size=N_CHUNKS) for _ in range(4 if FOUR_DEMODS else 2)]
            sts = [declare_stream() for _ in arrs]
            machine.initialize_qpu(target=q)
            align()
            with for_(n, 0, n < SHOTS_RD, n + 1):
                with for_each_(lv, rd_scales):
                    wait(RESET_NS // 4, rr.name, q.xy.name)
                    align()
                    procs = [demod.sliced("rd_cos", arrs[0], CHUNK_CYCLES, "out1"),
                             demod.sliced("rd_msin", arrs[1], CHUNK_CYCLES, "out1")]
                    if FOUR_DEMODS:
                        procs += [demod.sliced("rd_sin", arrs[2], CHUNK_CYCLES, "out2"),
                                  demod.sliced("rd_cos", arrs[3], CHUNK_CYCLES, "out2")]
                    measure("ringdown" * amp(lv), rr.name, *procs)
                    with for_(k, 0, k < N_CHUNKS, k + 1):
                        for a, s in zip(arrs, sts):
                            save(a[k], s)
            with stream_processing():
                for i, s in enumerate(sts):
                    s.buffer(N_CHUNKS).buffer(len(rd_scales)).average().save(f"rd{i}")
        progs["ringdown"] = (prog, [f"rd{i}" for i in range(len(arrs))])

        # --- fine complex sweep
        with program() as prog:
            n = declare(int)
            lv = declare(fixed)
            df = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            sI, sQ = declare_stream(), declare_stream()
            machine.initialize_qpu(target=q)
            align()
            with for_(n, 0, n < SHOTS_SW, n + 1):
                with for_each_(lv, sw_scales):
                    with for_(*from_array(df, SWEEP_HZ)):
                        update_frequency(rr.name, df + IF)
                        rr.measure("readout", qua_vars=(I, Q), amplitude_scale=lv)
                        rr.wait(SWEEP_WAIT_NS // 4)
                        save(I, sI)
                        save(Q, sQ)
            with stream_processing():
                sI.buffer(len(SWEEP_HZ)).buffer(len(sw_scales)).average().save("sw_I")
                sQ.buffer(len(SWEEP_HZ)).buffer(len(sw_scales)).average().save("sw_Q")
        progs["sweep"] = (prog, ["sw_I", "sw_Q"])

        # --- pump-probe
        with program() as prog:
            n = declare(int)
            d = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            Ip = declare(fixed)
            Qp = declare(fixed)
            st_ = {kk: (declare_stream(), declare_stream()) for kk in ("pp", "g", "e")}
            machine.initialize_qpu(target=q)
            align()
            with for_(n, 0, n < SHOTS_PP, n + 1):
                # ground reference
                wait(RESET_NS // 4, rr.name, q.xy.name)
                align()
                rr.measure("readout", qua_vars=(I, Q), amplitude_scale=ro_scale)
                save(I, st_["g"][0])
                save(Q, st_["g"][1])
                # excited reference
                wait(RESET_NS // 4, rr.name, q.xy.name)
                align()
                q.xy.play("x180")
                align()
                rr.measure("readout", qua_vars=(I, Q), amplitude_scale=ro_scale)
                save(I, st_["e"][0])
                save(Q, st_["e"][1])
                with for_each_(d, cycles):
                    wait(RESET_NS // 4, rr.name, q.xy.name)
                    align()
                    rr.measure("readout", qua_vars=(Ip, Qp))            # the top of 02b's sweep
                    rr.wait(d)
                    rr.measure("readout", qua_vars=(I, Q), amplitude_scale=ro_scale)
                    save(I, st_["pp"][0])
                    save(Q, st_["pp"][1])
            with stream_processing():
                st_["pp"][0].buffer(len(cycles)).average().save("pp_I")
                st_["pp"][1].buffer(len(cycles)).average().save("pp_Q")
                for kk in ("g", "e"):
                    st_[kk][0].average().save(f"{kk}_I")
                    st_[kk][1].average().save(f"{kk}_Q")
        progs["pump"] = (prog, ["pp_I", "pp_Q", "g_I", "g_Q", "e_I", "e_Q"])

        per_shot = {"ringdown": SHOTS_RD * len(rd_scales) * (RESET_NS + L_ON + L_OFF + 1000) * 1e-9,
                    "sweep": SHOTS_SW * len(sw_scales) * len(SWEEP_HZ) * (SWEEP_WAIT_NS + info["readout_len"] + 500) * 1e-9,
                    "pump": SHOTS_PP * (2 + len(cycles)) * (RESET_NS + 3 * info["readout_len"]) * 1e-9
                    + SHOTS_PP * sum(TAUS_NS) * 1e-9}
        log(f"{name}: estimated QPU {', '.join(f'{k} {v:.1f} s' for k, v in per_shot.items())}")
        assert max(per_shot.values()) < 60

        if DRY:
            from qm import generate_qua_script
            for kind, (prog, _) in progs.items():
                generate_qua_script(prog, config)
            log(f"{name}: programs built")
            continue
        info["qpu_s"] = {}
        with qm_session(machine.connect(), config, timeout=120) as qm:
            for kind, (prog, keys) in progs.items():
                wait_until_free()
                t0 = time.perf_counter()
                try:
                    job = qm.execute(prog)
                    job.result_handles.wait_for_all_values()
                    qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
                    data = fetch(job, keys)
                except Exception as exc:  # noqa: BLE001
                    log(f"{name}/{kind}: FAILED {type(exc).__name__}: {exc}")
                    info["qpu_s"][kind] = None
                    continue
                np.savez(OUT / f"r{rnd}_{name}_{kind}.npz", **data)
                info["qpu_s"][kind] = qpu
                log(f"{name}/{kind}: QPU {qpu:.1f} s, wall {time.perf_counter() - t0:.0f} s")
        meta[name] = info
        (OUT / f"r{rnd}_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    log("all done")


if __name__ == "__main__":
    main()
