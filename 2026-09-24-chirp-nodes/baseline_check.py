"""Excitation before the chirp: the no-T1 03a baseline and box top on the ground->excited axis of the T1-known run (no QPU).

usage: baseline_check.py <folder>...   (folders holding <backend>/03a-noT1-*.nc and 03a-T1-*.nc)
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.chirp import mean_over_directions  # noqa: E402

QB = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD5"]}
out = {}
for folder in sys.argv[1:]:
    for be, qs in QB.items():
        for q in qs:
            fk = next((p for p in sorted(Path(folder, be).glob("03a-T1-*.nc")) if q in p.stem), None)
            fu = next((p for p in sorted(Path(folder, be).glob("03a-noT1-*.nc")) if q in p.stem), None)
            if not fk or not fu:
                continue
            zs = []
            for f in (fk, fu):
                ds = mean_over_directions(xr.open_dataset(f)).sel(qubit=q).transpose("drive_level", "detuning")
                zs.append(ds.I.values + 1j * ds.Q.values)
                x = ds.detuning.values
            lines = np.array([np.median(x[np.argsort(-np.abs(z[3] - np.median(z[3])))[:3]]) for z in zs[:1]])
            box, far = np.abs(x - lines[0]) < 6e6, np.abs(x - lines[0]) > 40e6
            g, e = np.median(zs[0][:, far]), np.median(zs[0][2:5][:, box])
            u = (e - g) / abs(e - g)
            par = lambda z: ((z - g) * np.conj(u)).real / abs(e - g)
            b, t = float(par(np.median(zs[1][:, far]))), float(par(np.median(zs[1][2:5][:, box])))
            out[f"{folder}/{be}/{q}"] = {"baseline": b, "top": t}
            print(f"{folder:14s} {be:6s} {q:4s} T1 unknown: baseline {b:+.2f}, box top {t:+.2f}  (T1 known: 0 and 1)")
Path(__file__).resolve().parent.joinpath("replay/baselines.json").write_text(json.dumps(out, indent=1))
