import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from quam_config import Quam
from quam_config.instrument_limits import instrument_limits
Q = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
for be, qs in Q.items():
    m = Quam.load(f"state_check/{be}")
    for n in qs:
        q = m.qubits[n]; rr = q.resonator
        fs = getattr(rr.opx_output, "full_scale_power_dbm", None)
        print(be, n, "P_ro %.1f dBm" % rr.get_output_power("readout"), "FS", fs, "cap", instrument_limits(rr).max_readout_amplitude,
              "IF %.1f MHz" % (rr.intermediate_frequency / 1e6), "f_r %.4f GHz" % (rr.RF_frequency / 1e9),
              "ToF", rr.time_of_flight, "port", getattr(rr.opx_output, "port_id", None), getattr(rr.opx_input, "port_id", None),
              "x180" , "x180" in q.xy.operations, "T1", q.T1, "therm", q.thermalization_time)
