import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from quam_config import Quam
m = Quam.load("state_check/arbel")
cfg = m.generate_config()
q = m.qubits["qB4"]; el = q.resonator.name
ops = cfg["elements"][el]["operations"]
print("element keys", list(cfg["elements"][el].keys()))
print("ops", ops)
p = cfg["pulses"][ops["readout"]]
print("pulse", {k: v for k, v in p.items()})
for k, v in p["integration_weights"].items():
    iw = cfg["integration_weights"][v]
    print(k, v, {kk: (vv[:2] if isinstance(vv, list) else vv) for kk, vv in iw.items()})
print("wf", {k: cfg["waveforms"][w] if cfg["waveforms"][w]["type"] == "constant" else "arb" for k, w in p["waveforms"].items()})
print("dm", cfg["digital_waveforms"].get(p.get("digital_marker")))
import inspect
print(inspect.getsource(type(q.resonator).measure)[:2500])
