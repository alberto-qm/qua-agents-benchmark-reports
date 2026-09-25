import sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from quam_config import Quam
m = Quam.load("state_check/arbel")
q = m.qubits["qB4"]
print(type(q.resonator).__mro__[:4])
src = inspect.getsource(type(q.resonator).measure)
i = src.find("qua.measure(")
print(src[i:i+900])
print([n for n in dir(q.resonator) if "power" in n])
print(inspect.signature(q.resonator.set_output_power))
