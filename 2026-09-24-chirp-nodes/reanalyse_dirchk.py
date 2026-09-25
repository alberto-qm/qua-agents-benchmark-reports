"""Re-run the committed chirp analysis on the dirchk datasets (no QPU): <tag>.reanalysed.json beside each run.

The nodes' live fits came from 4b40520; ee24095 changed only the up/down-lean check (levels and point bar).
"""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
from tinycal.profile import IqccConfig  # noqa: E402
from tinycal.qualibrate import configure_iqcc  # noqa: E402

configure_iqcc(HERE / "gilboa/state", IqccConfig(enabled=False, backend="gilboa", default_timeout_s=60))
from quam_config import Quam  # noqa: E402
import calibration_utils.qubit_spectroscopy_chirp.analysis as A  # noqa: E402
import calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis as B  # noqa: E402


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return js(x.tolist())
    return x


for f in sorted((HERE / "dirchk").glob("*/0*.json")):
    if f.name.endswith(".reanalysed.json"):
        continue
    rec = json.loads(f.read_text())
    if rec.get("error") or not f.with_suffix(".nc").exists():
        continue
    be, variant = f.parent.name, rec["variant"]
    m = Quam.load(str(HERE / "dirchk" / be / f"state_{variant}"))
    qs = [m.qubits[q] for q in rec["qubits"]]
    ns = rec["namespace"]
    missing = {q.name: "no T1 in the state" for q in qs if variant == "noT1"}
    node = SimpleNamespace(
        namespace={"qubits": qs, "sweep_lengths": ns["sweep_lengths"], "rates": ns["rates"], "kappa_priors": ns.get("kappa_priors"),
                   "missing_t1": missing, "drive_amplitudes": ns.get("drive_amplitudes"),
                   "flux_offsets": {k: np.asarray(v) for k, v in (ns.get("flux_offsets") or {}).items()}},
        parameters=SimpleNamespace(chirp_band_in_mhz=20.0, pulse="chirp", min_columns=7))
    if ns.get("ladders"):
        node.namespace["ladders"] = {k: SimpleNamespace(played=v["played"]) for k, v in ns["ladders"].items()}
    ds = xr.open_dataset(f.with_suffix(".nc"))
    mod = A if rec["node"].startswith("03a") else B
    _, res = mod.fit_raw_data(ds, node)
    out = {q: asdict(v) for q, v in res.items()}
    f.with_name(f.stem + ".reanalysed.json").write_text(json.dumps(js(out), indent=1))
    for q, r in out.items():
        live = rec["fit_results"][q]
        same = ("frequency_shift" not in r) or (r["frequency_shift"] == live.get("frequency_shift")) or \
               (np.isnan(r["frequency_shift"]) and np.isnan(live.get("frequency_shift", np.nan)))
        print(f"{be:6s} {f.stem:24s} {q:4s} {r.get('line_identity') or ''} refused={r.get('decay_refused')} "
              f"sweep/T1 {r['sweep_over_t1']:+.2f} ± {r['sweep_over_t1_error']:.2f} (live {live.get('sweep_over_t1')}), "
              f"f01/x0 unchanged: {same}")
