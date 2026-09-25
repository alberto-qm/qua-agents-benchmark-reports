"""Steady-state resonator sweep with a long pulse, next to the qubit's own readout pulse (25 Sep 2026).

usage: kappa_long.py <backend> [--dry]

Why: the ring-downs give kappa/2pi ~0.2-0.3 MHz on several resonators while the swept dips are ~3x wider. The sweeps
use the qubit's readout pulse (0.8-2 us), no longer than the resonator's fill time 2/kappa, so the resonator never
settles and the dip takes the pulse's own spectral width (~1/T). This sweeps +-3 MHz in 20 kHz steps with
  long   a 10 us pulse integrated over its last 4 us only (the resonator settled), at committed - 10 dB and committed
  short  the qubit's readout pulse at committed, integrated as the nodes do
12 us between points, 300 shots. If the long-pulse dip's width matches the ring-down kappa and the short one does not,
the nodes' linewidths are pulse-limited, not kappa. One job per qubit (~6 s QPU). State: pulled 18:03, gilboa's C/D
qubits marked active in the local copy. Nothing written to any state. Outputs in kappa/<backend>/long_<qubit>.npz.
"""
from __future__ import annotations

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
backend = sys.argv[1]
DRY = "--dry" in sys.argv
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[backend]
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

L_LONG, L_SETTLE = 10_000, 6_000
SWEEP_HZ = np.arange(-3e6, 3e6 + 1, 20e3).astype(int)
WAIT_NS = 12_000
SHOTS = 300
TOP_CEILING_DBM, CAP = -10, 0.1


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{backend}/kappa-long] {msg}"
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


def inject_long(config, element, peak):
    L = L_LONG
    config["waveforms"]["lg_zero"] = {"type": "constant", "sample": 0.0}
    config["waveforms"][f"lg_{element}_I"] = {"type": "constant", "sample": float(peak)}
    # integrate only once the resonator has settled
    for name, (c, s) in {"lg_cos": (1.0, 0.0), "lg_sin": (0.0, 1.0), "lg_msin": (0.0, -1.0)}.items():
        config["integration_weights"][name] = {"cosine": [(0.0, L_SETTLE), (c, L - L_SETTLE)],
                                               "sine": [(0.0, L_SETTLE), (s, L - L_SETTLE)]}
    config["pulses"][f"lg_{element}_pulse"] = {
        "operation": "measurement", "length": L, "digital_marker": "ON",
        "waveforms": {"I": f"lg_{element}_I", "Q": "lg_zero"},
        "integration_weights": {"lg_cos": "lg_cos", "lg_sin": "lg_sin", "lg_msin": "lg_msin"}}
    config["elements"][element]["operations"]["longro"] = f"lg_{element}_pulse"


def main():
    from tinycal.profile import IqccConfig
    from tinycal.qualibrate import configure_iqcc

    configure_iqcc(STATE, IqccConfig(enabled=not DRY, backend=backend, default_timeout_s=120))
    from qm.qua import (align, amp, declare, declare_stream, dual_demod, fixed, for_, for_each_, measure, program, save,
                        stream_processing, update_frequency)
    from qualang_tools.loops import from_array
    from qualang_tools.multi_user import qm_session
    from quam_config import Quam

    meta = {}
    for name in QUBITS:
        machine = Quam.load(str(STATE))
        q = machine.qubits[name]
        rr = q.resonator
        p_ro = float(rr.get_output_power("readout"))
        top = int(min(TOP_CEILING_DBM, round(p_ro + 20)))
        rr.set_output_power(power_in_dbm=top, max_amplitude=CAP)       # the same port setup as kappa.py and 02b
        a_top = float(rr.operations["readout"].amplitude)

        def scale(p):
            return float(10 ** ((p - top) / 20))

        long_scales = [scale(p_ro - 10), scale(p_ro)]
        ro_scale = scale(p_ro)
        IF = int(rr.intermediate_frequency)
        config = machine.generate_config()
        inject_long(config, rr.name, a_top)
        with program() as prog:
            n = declare(int)
            lv = declare(fixed)
            df = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            sl = (declare_stream(), declare_stream())
            ss = (declare_stream(), declare_stream())
            machine.initialize_qpu(target=q)
            align()
            with for_(n, 0, n < SHOTS, n + 1):
                with for_each_(lv, long_scales):
                    with for_(*from_array(df, SWEEP_HZ)):
                        update_frequency(rr.name, df + IF)
                        measure("longro" * amp(lv), rr.name,
                                dual_demod.full(iw1="lg_cos", element_output1="out1", iw2="lg_sin",
                                                element_output2="out2", target=I),
                                dual_demod.full(iw1="lg_msin", element_output1="out1", iw2="lg_cos",
                                                element_output2="out2", target=Q))
                        rr.wait(WAIT_NS // 4)
                        save(I, sl[0])
                        save(Q, sl[1])
                with for_(*from_array(df, SWEEP_HZ)):
                    update_frequency(rr.name, df + IF)
                    rr.measure("readout", qua_vars=(I, Q), amplitude_scale=ro_scale)
                    rr.wait(WAIT_NS // 4)
                    save(I, ss[0])
                    save(Q, ss[1])
            with stream_processing():
                sl[0].buffer(len(SWEEP_HZ)).buffer(len(long_scales)).average().save("long_I")
                sl[1].buffer(len(SWEEP_HZ)).buffer(len(long_scales)).average().save("long_Q")
                ss[0].buffer(len(SWEEP_HZ)).average().save("short_I")
                ss[1].buffer(len(SWEEP_HZ)).average().save("short_Q")
        rl = int(rr.operations["readout"].length)
        est = SHOTS * len(SWEEP_HZ) * (2 * (L_LONG + WAIT_NS) + rl + WAIT_NS) * 1e-9
        info = dict(committed_power_dbm=p_ro, top_dbm=top, readout_amp_at_top=a_top, long_scales=long_scales,
                    ro_scale=ro_scale, long_levels_dbm=[p_ro - 10, p_ro], f_r=float(rr.RF_frequency), IF=IF,
                    readout_len=rl, l_long=L_LONG, l_settle=L_SETTLE, sweep_hz=SWEEP_HZ.tolist(), wait_ns=WAIT_NS,
                    shots=SHOTS, state=str(SNAP), est_qpu_s=est)
        log(f"{name}: committed {p_ro:.1f} dBm; long pulse at {p_ro - 10:.1f} and {p_ro:.1f} dBm, short at {p_ro:.1f}; "
            f"estimated QPU {est:.1f} s")
        assert est < 45
        if DRY:
            from qm import generate_qua_script
            generate_qua_script(prog, config)
            log(f"{name}: program built")
            continue
        wait_until_free()
        t0 = time.perf_counter()
        try:
            with qm_session(machine.connect(), config, timeout=120) as qm:
                job = qm.execute(prog)
                job.result_handles.wait_for_all_values()
                qpu = float(job.result_handles.get("__qpu_execution_time_seconds").fetch_all())
                data = {}
                for k in ("long_I", "long_Q", "short_I", "short_Q"):
                    raw = job.result_handles.get(k).fetch_all()
                    try:
                        raw = raw["value"]
                    except (TypeError, IndexError, KeyError, ValueError):
                        pass
                    data[k] = np.asarray(raw, dtype=float)
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: FAILED {type(exc).__name__}: {exc}")
            continue
        np.savez(OUT / f"long_{name}.npz", **data)
        info["qpu_s"] = qpu
        meta[name] = info
        (OUT / "long_meta.json").write_text(json.dumps(meta, indent=1, default=float))
        log(f"{name}: QPU {qpu:.1f} s, wall {time.perf_counter() - t0:.0f} s")
    log("all done")


if __name__ == "__main__":
    main()
