"""Replays of the saturation node 03a on crops of the saturation drive ladder (no QPU) -> replay/replays_sat.json.

The ladder (run_satladder2.py) measured nine drives g = 1/16..16 x the node's default over +-130 MHz at the
node's own 0.15 MHz step and 300 shots. A node run whose stored saturation amplitude is off by g and whose
stored f_01 is off by F plays drive g and sees F +- 50 MHz (its default span). The node's
own fit_raw_data runs on that crop. Outcomes as for the chirp: pass (within 1 MHz of the reference),
imprecise (1-5 MHz), wrong (further: another line, typically 0->2), refused (the node's "no line detected").

Reference f_01 (at the idle point): the fine saturation scans of the same session where the saturation
node accepted them; qolab Q1 (fine scan rejected) uses the wide chirp map; gilboa qD2 has no reference
today and uses the 23 Sep chirp measurement (stored f_01 - 4.75 MHz).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
import os  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")
from quam_config import Quam  # noqa: E402

from calibration_utils.qubit_spectroscopy.analysis import fit_raw_data  # noqa: E402
from calibration_utils.qubit_spectroscopy.parameters import Parameters  # noqa: E402

HERE = Path(__file__).resolve().parent
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}


def reference(be: str, q: str, stored_f01: float, rep_chirp: dict) -> tuple[float, str]:
    fine = HERE / be / f"fine-{q}.json"
    if fine.exists():
        r = (json.loads(fine.read_text()).get("fit_results") or {}).get(q) or {}
        if r.get("signal_detected") and r.get("f_01"):
            return float(r["f_01"]), "fine saturation scan"
    if be == "gilboa" and q == "qD2":
        return stored_f01 - 4.75e6, "23 Sep chirp (no line today)"
    off = rep_chirp.get(f"{be}/{q}", {}).get("03a", {}).get("truth_offset")
    return stored_f01 + float(off), "wide chirp map"


def main():
    rep_chirp = json.loads((HERE / "replay/replays.json").read_text())
    out = {}
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(QUBITS)
    for be, qs in QUBITS.items():
        if be not in only:
            continue
        machine = Quam.load(str(HERE / be / "state"))
        for q in qs:
            levels = {}
            for f in sorted((HERE / "satladder_default" / be).glob(f"satdef-x*-{q}.nc")):
                rec = json.loads(f.with_suffix(".json").read_text())
                levels[float(rec["g"])] = (xr.open_dataset(f).load(), rec)
            qb = machine.qubits[q]
            truth, source = reference(be, q, float(qb.f_01), rep_chirp)
            res = {"reference": truth, "reference_source": source, "grid": {}, "levels": sorted(g for g, (_, r) in levels.items()
                                                                                                 if q in r["qubits"])}
            for span in (100.0,):
                grid = []
                for g, (ds, rec) in sorted(levels.items()):
                    if q not in rec["qubits"]:
                        continue
                    dq = ds.sel(qubit=[q])
                    for F in np.arange(-80e6, 80e6 + 1, 5e6):
                        lo, hi = F - span / 2 * 1e6, F + span / 2 * 1e6
                        crop = dq.sel(detuning=slice(lo - 1, hi + 1))
                        if crop.sizes["detuning"] < 10 or lo < float(ds.detuning.min()) - 1 or hi > float(ds.detuning.max()) + 1:
                            continue
                        params = Parameters(qubits=[q], frequency_span_in_mhz=span, frequency_step_in_mhz=0.15,
                                            operation_amplitude_factor=float(rec["factor"]), operation_len_in_ns=20000)
                        node = SimpleNamespace(parameters=params, namespace={"qubits": [qb]}, log=lambda *a, **k: None)
                        try:
                            _, fits = fit_raw_data(crop, node)
                            fr = fits[q]
                            f01 = float(getattr(fr, "f_01", np.nan))
                            detected = bool(getattr(fr, "signal_detected", False))
                        except Exception as exc:  # noqa: BLE001
                            f01, detected = float("nan"), False
                        err = abs(f01 - truth) if np.isfinite(f01) else np.inf
                        outcome = ("refused" if not detected else "pass" if err < 1e6 else "imprecise" if err < 5e6 else "wrong")
                        grid.append({"m": int(round(np.log2(g))), "g": g, "F": float(F), "outcome": outcome,
                                     "f01_offset": float(f01 - truth) if np.isfinite(f01) else None})
                res["grid"][f"{span:g}"] = grid
            out[f"{be}/{q}"] = res
            c = {s: {k: sum(x["outcome"] == k for x in gr) for k in ("pass", "imprecise", "refused", "wrong")}
                 for s, gr in res["grid"].items()}
            print(be, q, source, c, flush=True)
    path = HERE / "replay/replays_sat.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(out)
    path.write_text(json.dumps(merged, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
