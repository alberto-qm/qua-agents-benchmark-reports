"""02b under the four depletion conditions: node results, QPU, and what the lowest rows look like (no QPU).

usage: ab_analyse.py        prints one table per backend and writes ab/summary.json

Per qubit, condition and round: success, onset and optimal power, onset mechanism, the proposed frequency (shift from the
stored one), the dressed frequency and linewidth, plateau contrast and QPU time. Low rows: the normalised |S21| traces of
the lowest 10 powers against the next 15 (rows 10-24, still in the linear reference block): the depth of the dip at the
dressed frequency in each block and the position of its minimum -- an excited qubit carried over from the top of the
previous column (A, B) shows as a shallower dip at the dressed frequency in the lowest rows, pulled towards the
dispersively shifted one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
CONDS = ("A", "B", "C", "D")
KEYS = ("success", "optimal_power", "onset_power", "onset_mechanism", "frequency_shift", "dressed_frequency", "linewidth",
        "plateau_contrast", "contrast_at_optimal", "bare_frequency", "bistability_resolvable")


def rows(ds: xr.Dataset, q: str):
    """Normalised |S21| (power x detuning), each row over its own median."""
    a = ds["IQ_abs"].sel(qubit=q).transpose("power", "detuning").values if "IQ_abs" in ds else None
    if a is None:
        i = ds["I"].sel(qubit=q).transpose("power", "detuning").values
        qq = ds["Q"].sel(qubit=q).transpose("power", "detuning").values
        a = np.hypot(i, qq)
    return a / np.median(a, axis=1, keepdims=True)


def low_rows(ds, q):
    a = rows(ds, q)
    det = ds.detuning.values
    ref = a[10:25].mean(axis=0)
    low = a[:10].mean(axis=0)
    k = np.convolve(np.ones(5) / 5, np.ones(1), mode="full")
    sm = lambda x: np.convolve(x, np.ones(5) / 5, mode="same")  # noqa: E731
    i_ref = int(np.argmin(sm(ref)[5:-5])) + 5
    i_low = int(np.argmin(sm(low)[5:-5])) + 5
    return dict(depth_ref=float(1 - sm(ref)[i_ref]), depth_low_at_ref=float(1 - sm(low)[i_ref]),
                pos_ref=float(det[i_ref]), pos_low=float(det[i_low]), rms_low_minus_ref=float(np.std(low - ref)))


def load(be, cond, rnd, q):
    f = HERE / "ab" / be / f"{cond}-r{rnd}-{q}.json"
    if not f.exists():
        return None, None
    rec = json.loads(f.read_text())
    nc = f.with_suffix(".nc")
    ds = xr.open_dataset(nc) if nc.exists() else None
    return rec, ds


def fmt(v, s=1.0, spec="+.2f"):
    if v is None or isinstance(v, str):
        return str(v)
    return "—" if not np.isfinite(v) else format(v * s, spec)


def main():
    summary = {}
    for be, qs in QUBITS.items():
        print(f"\n{be}: cond/round  success  optimal / onset [dBm] (mechanism)  f shift [MHz]  linewidth [MHz]  plateau  "
              f"QPU [s] | low rows: dip depth low/ref at the ref minimum, minimum position low/ref [MHz]")
        for q in qs:
            for cond in CONDS:
                for rnd in (1, 2):
                    rec, ds = load(be, cond, rnd, q)
                    if rec is None:
                        continue
                    r = (rec.get("fit_results") or {}).get(q) or {}
                    row = {k: r.get(k) for k in KEYS}
                    row.update(qpu_s=rec["qpu_s"], wait_ns=rec["effective_wait_ns"], error=rec["error"],
                               window=rec["window_dbm"])
                    if ds is not None:
                        try:
                            row["low_rows"] = low_rows(ds, q)
                        except Exception as exc:  # noqa: BLE001
                            row["low_rows"] = {"error": repr(exc)}
                    summary[f"{be}/{q}/{cond}/r{rnd}"] = row
                    lr = row.get("low_rows") or {}
                    print(f"  {q:4s} {cond} r{rnd}  {str(row['success']):5s}  {fmt(row['optimal_power'], 1, '.1f')} / "
                          f"{fmt(row['onset_power'], 1, '.1f')} ({row['onset_mechanism']})  "
                          f"{fmt(row['frequency_shift'], 1e-6, '+.3f')}  {fmt(row['linewidth'], 1e-6, '.3f')}  "
                          f"{fmt(row['plateau_contrast'], 1, '.3f')}  {row['qpu_s']:.1f} | "
                          f"{fmt(lr.get('depth_low_at_ref'), 1, '.3f')}/{fmt(lr.get('depth_ref'), 1, '.3f')}  "
                          f"{fmt(lr.get('pos_low'), 1e-6, '+.2f')}/{fmt(lr.get('pos_ref'), 1e-6, '+.2f')}"
                          + (f"  ERROR {row['error'][:80]}" if row["error"] else ""))
    (HERE / "ab" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
