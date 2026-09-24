"""Re-fit every live node dataset with the final analysis code (no QPU) -> replay/live.json.

The live runs of 24 Sep used the analysis as it stood at launch; two checks were added after
(the short-sweep guard and the lone-line drive-scale check, both reported in the report).
This applies the committed analysis to the same datasets so the tables show what the nodes
now answer; the live answers are kept beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
from calibration_utils.qubit_spectroscopy_chirp.analysis import analyse_ladder  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.chirp import short_sweep_warning  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

HERE = Path(__file__).resolve().parent
BAND = 20e6
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}


def f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def main():
    out = {}
    for be, qs in QUBITS.items():
        for path in sorted((HERE / be).glob("*.json")):
            tag = path.stem
            if not (tag.startswith(("r0-", "end-", "sat-", "smoke-", "fine-"))):
                continue
            rec = json.loads(path.read_text())
            nc = path.with_suffix(".nc")
            ds = xr.open_dataset(nc) if nc.exists() else None
            for q in rec["qubits"]:
                live = (rec.get("fit_results") or {}).get(q) or {}
                row = {"backend": be, "qubit": q, "tag": tag, "node": rec["node"], "qpu_s": rec["qpu_s"],
                       "wall_s": rec["wall_s"], "live": {k: v for k, v in live.items() if not isinstance(v, (list, dict))}}
                ns = rec.get("namespace") or {}
                if ds is not None and rec["node"] == "03a_qubit_spectroscopy_chirp":
                    dq = ds.sel(qubit=q)
                    amps = np.asarray(dq.drive_amplitude.values, float)
                    r = analyse_ladder(ds.detuning.values, amps, dq.I.values, dq.Q.values, BAND, ns["rates"][q],
                                       ns["kappa_priors"][q])
                    short = short_sweep_warning(BAND, ns["sweep_lengths"][q])
                    ident = "short-sweep" if short else r["line_identity"]
                    row["final"] = {"f01_offset": f(r["f01_offset"]), "f01_error": f(r["f01_offset_error"]), "identity": ident,
                                    "drive_scale": f(r["drive_scale"]), "alpha": f(r["anharmonicity_estimate"]),
                                    "selected_amplitude": f(r["selected_amplitude"]), "sweep_ns": ns["sweep_lengths"][q],
                                    "proposed": ident == "0-1" and f(r["f01_offset"]) is not None,
                                    "warnings": ([short] if short else []) + r["warnings"]}
                elif ds is not None and rec["node"] == "03b_qubit_spectroscopy_vs_flux_chirp":
                    dq = ds.sel(qubit=q)
                    pulse = (rec.get("params") or {}).get("pulse", "chirp")
                    r = analyse_flux_map(ds.detuning.values, dq.flux_bias.values, dq.I.values, dq.Q.values, BAND, pulse)
                    short = short_sweep_warning(BAND, ns["sweep_lengths"][q]) if pulse == "chirp" else None
                    x0 = f(r["x0"])
                    inside = x0 is not None and float(dq.flux_bias.min()) <= x0 <= float(dq.flux_bias.max())
                    errs = r["errors"][np.isfinite(r["errors"])]
                    row["final"] = {"x0": x0, "x0_error": f(r["x0_error"]), "f0": f(r["f0"]), "f0_error": f(r["f0_error"]),
                                    "curvature": f(r["curvature"]), "chi2": f(r["chi2_red"]), "tracked": r["tracked"],
                                    "columns": r["columns"], "column_precision": f(np.median(errs)) if errs.size else None,
                                    "pulse": pulse, "proposed": bool(x0 is not None and inside and not short),
                                    "warnings": ([short] if short else []) + r["warnings"]}
                out[f"{be}/{q}/{tag}"] = row
    (HERE / "replay").mkdir(exist_ok=True)
    (HERE / "replay/live.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        fin = v.get("final") or {}
        print(k, {kk: fin.get(kk) for kk in ("identity", "f01_offset", "x0", "tracked", "proposed")} if fin else v["live"].get("f_01"))


if __name__ == "__main__":
    main()
