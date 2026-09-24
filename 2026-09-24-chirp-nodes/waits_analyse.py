"""2 T1 against 5 T1 with up-and-down sweeps: the node's results plus each direction fitted on its own (no QPU).

Writes waits/summary.json and prints one table per node. The up-only and down-only fits show the pull of
leftover excitation directly: a one-way sweep drags the line along its direction, so up - down is twice it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.analysis import analyse_ladder  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.chirp import align_directions  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
BAND = 20e6


def find(be, node, factor, q):
    for f in sorted((HERE / "waits" / be).glob(f"{node}-{factor}T1-*.json")):
        rec = json.loads(f.read_text())
        if q in rec["qubits"]:
            nc = f.with_suffix(".nc")
            return rec, (align_directions(xr.open_dataset(nc)) if nc.exists() else None)
    return None, None


def one_03a(rec, ds, q):
    r = (rec.get("fit_results") or {}).get(q) or {}
    out = {k: r.get(k) for k in ("f_01", "frequency_shift", "f_01_error", "line_identity", "drive_scale", "box_height",
                                 "box_snr", "anharmonicity_estimate")}
    out["warnings"] = r.get("warnings")
    out["qpu_s"], out["job_qubits"] = rec["qpu_s"], rec["qubits"]
    if ds is not None and "direction" in ds.dims:
        dq = ds.sel(qubit=q)
        ns = rec["namespace"]
        played = ns["ladders"][q]["played"]
        for d in ("up", "down"):
            dd = dq.sel(direction=d).transpose("drive_level", "detuning")
            a = analyse_ladder(ds.detuning.values, played, dd.I.values, dd.Q.values, BAND, ns["rates"][q],
                               ns["kappa_priors"][q])
            out[f"shift_{d}"] = a["f01_offset"]
            out[f"identity_{d}"] = a["line_identity"]
            out[f"height_{d}"] = a["box"].height if a.get("box") is not None else None
    return out


def one_03b(rec, ds, q):
    r = (rec.get("fit_results") or {}).get(q) or {}
    out = {k: r.get(k) for k in ("idle_offset_shift", "idle_offset_shift_error", "frequency_shift", "f_01_error",
                                 "column_precision", "tracked_columns", "flux_columns", "chi2_red", "curvature")}
    out["warnings"] = r.get("warnings")
    out["qpu_s"], out["job_qubits"] = rec["qpu_s"], rec["qubits"]
    if ds is not None and "direction" in ds.dims:
        dq = ds.sel(qubit=q)
        for d in ("up", "down"):
            dd = dq.sel(direction=d).transpose("detuning", "flux_index")
            a = analyse_flux_map(ds.detuning.values, dq.flux_bias.values, dd.I.values, dd.Q.values, BAND, "chirp")
            out[f"x0_{d}"], out[f"f0_{d}"], out[f"prec_{d}"] = a["x0"], a["f0"], float(np.nanmedian(a["errors"]))
    return out


def fmt(v, scale=1.0, spec="+.2f"):
    return "—" if v is None or not np.isfinite(v) else format(v * scale, spec)


def main():
    summary = {}
    print("03a chirp: frequency shift from stored f01 [MHz], node result (up/down mean) and each direction; box height ratio")
    for be, qs in QUBITS.items():
        for q in qs:
            row = {}
            for f in (2, 5):
                rec, ds = find(be, "03a", f, q)
                if rec:
                    row[f] = one_03a(rec, ds, q)
            summary[f"{be}/{q}/03a"] = row
            if 2 in row and 5 in row:
                a, b = row[2], row[5]
                hr = (a["box_height"] / b["box_height"]) if a.get("box_height") and b.get("box_height") else None
                print(f"  {be:6s} {q:4s} 2T1 {fmt(a['frequency_shift'], 1e-6)} [{a['line_identity']}] "
                      f"(up {fmt(a.get('shift_up'), 1e-6)}, down {fmt(a.get('shift_down'), 1e-6)})   "
                      f"5T1 {fmt(b['frequency_shift'], 1e-6)} [{b['line_identity']}] "
                      f"(up {fmt(b.get('shift_up'), 1e-6)}, down {fmt(b.get('shift_down'), 1e-6)})   "
                      f"diff {fmt((a['frequency_shift'] or np.nan) - (b['frequency_shift'] or np.nan), 1e-6)}  "
                      f"height 2T1/5T1 {fmt(hr, 1, '.2f')}  drive {fmt(a.get('drive_scale'), 1, '.2f')}/"
                      f"{fmt(b.get('drive_scale'), 1, '.2f')}")
    print("\n03b chirp: sweet spot from idle [mV] (node, up, down), apex frequency [MHz], median column error [MHz]")
    for be, qs in QUBITS.items():
        for q in qs:
            row = {}
            for f in (2, 5):
                rec, ds = find(be, "03b", f, q)
                if rec:
                    row[f] = one_03b(rec, ds, q)
            summary[f"{be}/{q}/03b"] = row
            for f in (2, 5):
                if f in row:
                    a = row[f]
                    print(f"  {be:6s} {q:4s} {f}T1 x0 {fmt(a['idle_offset_shift'], 1e3, '+.3f')} ± "
                          f"{fmt(a['idle_offset_shift_error'], 1e3, '.3f')} (up {fmt(a.get('x0_up'), 1e3, '+.3f')}, "
                          f"down {fmt(a.get('x0_down'), 1e3, '+.3f')})  f0 {fmt(a['frequency_shift'], 1e-6)} "
                          f"(up {fmt(a.get('f0_up'), 1e-6)}, down {fmt(a.get('f0_down'), 1e-6)})  "
                          f"col err {fmt(a['column_precision'], 1e-6, '.2f')}  tracked {a['tracked_columns']}/"
                          f"{a['flux_columns']}  QPU {a['qpu_s']:.1f} s")
    qpu = {}
    for f in sorted((HERE / "waits").glob("*/runs.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            qpu.setdefault(f"{r['node']}/{r['factor']}T1", []).append(r["qpu_s"])
    print("\nQPU per node and wait:", {k: round(sum(v), 1) for k, v in sorted(qpu.items())})
    (HERE / "waits" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
