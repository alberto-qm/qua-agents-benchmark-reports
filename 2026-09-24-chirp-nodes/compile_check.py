"""Build the chirp nodes' QUA programs and configs on each backend state, no QPU."""
import os, sys, shutil
from pathlib import Path
WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
os.environ["MPLBACKEND"] = "Agg"
be, state = sys.argv[1], Path(sys.argv[2])
from tinycal.profile import IqccConfig
from tinycal.qualibrate import configure_iqcc
configure_iqcc(state, IqccConfig(enabled=False, backend=be, default_timeout_s=60))
import qualibrate_ai
lib = qualibrate_ai.QualibrationLibrary.from_folder(WT / "calibrations/1Q_calibrations")
from quam_config import Quam
from qm import generate_qua_script
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}[be]
skip = ["execute_qua_program", "simulate_qua_program", "process_data", "fit_data", "plot_data", "update_state", "save_results", "load_data"]
cases = [("03a_qubit_spectroscopy_chirp", {}), ("03a_qubit_spectroscopy_chirp", dict(num_drive_levels=11, frequency_span_in_mhz=400.0)),
         ("03a_qubit_spectroscopy_chirp", dict(drive_prior="none", num_drive_levels=8)),
         ("03b_qubit_spectroscopy_vs_flux_chirp", {}), ("03b_qubit_spectroscopy_vs_flux_chirp", dict(flux_span_factor=2.5, num_flux_points=35, frequency_min_in_mhz=-50.0, frequency_max_in_mhz=40.0)),
         ("03b_qubit_spectroscopy_vs_flux_chirp", dict(pulse="saturation", num_flux_points=15, frequency_min_in_mhz=-25.0, frequency_max_in_mhz=10.0, num_shots=50))]
for name, params in cases:
    node = lib.nodes[name].copy(name=name)
    mod = sys.modules[f"qualibrate_ai_lib_{name}"]
    node.bind_machine(Quam.load())
    node.run(mode="propose", interactive=True, external=True, skip_actions=skip, qubits=QUBITS, **params)
    cfg = mod._config(node) if hasattr(mod, "_config") else None
    # _config reads node.namespace; the module-level function takes the node explicitly
    txt = generate_qua_script(node.namespace["qua_program"], cfg)
    print(be, name, params, "script chars", len(txt), "ok")
