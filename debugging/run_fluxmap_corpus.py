"""Run resonator_spectroscopy_vs_flux's fit_raw_data on every stored flux map. Usage: run_analysis.py <qua-libs root> <out.json>"""
import json, sys, warnings
from pathlib import Path
import numpy as np, xarray as xr
root = Path(sys.argv[1]) / "qualibration_graphs/superconducting"
sys.path.insert(0, str(root))
warnings.filterwarnings("ignore")
from calibration_utils.resonator_spectroscopy_vs_flux.analysis import fit_raw_data

DATA = Path.home() / ".qualibrate/user_storage/tinycal/data"

class Res: pass
class Qb: pass

def node_for(ds):
    det = ds.detuning.values
    class P: pass
    p = P(); p.frequency_span_in_mhz = float(det.max() - det.min()) / 1e6; p.frequency_step_in_mhz = float(np.median(np.diff(det))) / 1e6
    p.input_line_impedance_in_ohm = 50; p.line_attenuation_in_db = 0; p.min_fit_r_squared = 0.8; p.update_flux_min = False
    qubits = []
    for i, q in enumerate(ds.qubit.values):
        r = Res(); r.RF_frequency = float(ds.full_freq.values[i, 0] - det[0]) if "full_freq" in ds.coords else 0.0
        qb = Qb(); qb.resonator = r; qb.name = str(q); qubits.append(qb)
    class N: pass
    n = N(); n.parameters = p; n.namespace = {"qubits": qubits}
    return n

out = {}
for d in sorted(DATA.iterdir()):
    f = d / "ds_raw.h5"
    if not f.exists(): continue
    try:
        ds = xr.open_dataset(f)
    except Exception: continue
    if not ("flux_bias" in ds.dims and "detuning" in ds.dims and "IQ_abs_bg_subtracted" in ds and "I" in ds and "state" not in ds and ds.sizes["detuning"] >= 20):
        ds.close(); continue
    if not (d / "ds_fit.h5").exists() or "peak_freq" not in xr.open_dataset(d / "ds_fit.h5"):
        ds.close(); continue  # not this node
    try:
        fit, res = fit_raw_data(ds, node_for(ds))
        r = res[list(res)[0]]
        out[d.name] = dict(qubit=str(ds.qubit.values[0]), idle=r.idle_offset, r2=r.r_squared, apex=r.measured_apex_offset, in_sweep=r.apex_in_sweep,
                           side=r.apex_side, agrees=r.fit_agrees, shift=r.frequency_shift, period=r.phi0_voltage, reason=r.failure_reason,
                           tracked=int(np.isfinite(fit.peak_freq.values).sum()), cols=int(ds.sizes["flux_bias"]), lo=float(ds.flux_bias.min()), hi=float(ds.flux_bias.max()))
    except Exception as e:
        out[d.name] = dict(error=repr(e)[:200])
    ds.close()
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
print(len(out), "datasets ->", sys.argv[2])
