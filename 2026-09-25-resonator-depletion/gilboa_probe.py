"""Is gilboa's QOP answering? Runs one tiny QUA program (a single readout) with a 60 s limit; exit 0 if it ran."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from tinycal.profile import IqccConfig
from tinycal.qualibrate import configure_iqcc
STATE = Path(__file__).resolve().parent / "kappa" / "gilboa" / "state"
configure_iqcc(STATE, IqccConfig(enabled=True, backend="gilboa", default_timeout_s=60))
from qm.qua import program, declare, fixed, align
from qualang_tools.multi_user import qm_session
from quam_config import Quam
m = Quam.load(str(STATE))
q = m.qubits["qD5"]
with program() as prog:
    I = declare(fixed); Q = declare(fixed)
    m.initialize_qpu(target=q)
    align()
    q.resonator.measure("readout", qua_vars=(I, Q))
try:
    with qm_session(m.connect(), m.generate_config(), timeout=60) as qm:
        job = qm.execute(prog)
        job.result_handles.wait_for_all_values()
        print("gilboa ok")
        sys.exit(0)
except Exception as exc:  # noqa: BLE001
    print(f"gilboa down: {type(exc).__name__}: {str(exc)[:200]}")
    sys.exit(1)
