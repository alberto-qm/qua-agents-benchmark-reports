"""Offline replays of the chirp nodes' analysis on crops of the wide maps (no QPU).

03a: the wide ladder has 11 drive levels a factor 2 apart centred on the x180 prediction
(x1/32..x32) over +-200 MHz. A node run whose stored f_01 is off by F and whose drive is
g = 2^m times what the x180 predicts sees 7 of those levels (shifted by m) and the band
centres F-120..F+120 MHz. The replay runs analyse_ladder on exactly that crop and grades
the answer against the full-map reference.

03b: the wide map has 35 columns over +-2.5x the 20 MHz offset and band centres f_01
-50..+40 MHz. A node run whose idle is off by delta and whose stored f_01 is off by F sees
the columns within delta +- 1.5x the 20 MHz offset and the centres F-40..F+30 MHz.

usage: replay.py  -> replays.json + capture-range figures in replay/
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
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.chirp import short_sweep_warning  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "replay"
OUT.mkdir(exist_ok=True)
BAND = 20e6
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}


def load(backend: str, prefix: str, qubit: str):
    for f in sorted((HERE / backend).glob(f"{prefix}*.nc")):
        ds = xr.open_dataset(f)
        if qubit in list(ds.qubit.values):
            rec = json.loads(f.with_suffix(".json").read_text())
            return ds, rec
    return None, None


def load_all(backend: str, prefix: str, qubit: str):
    """All datasets for the qubit whose tag starts with prefix (the Q2 wide map comes in two halves)."""
    out = []
    for f in sorted((HERE / backend).glob(f"{prefix}*.nc")):
        ds = xr.open_dataset(f)
        if qubit in list(ds.qubit.values):
            out.append((ds, json.loads(f.with_suffix(".json").read_text())))
    return out


def replay_03a(backend: str, q: str) -> dict | None:
    ds, rec = load(backend, "wide3a-", q)
    if ds is None:
        return None
    dq = ds.sel(qubit=q)
    x = ds.detuning.values.astype(float)
    amps = np.asarray(dq.drive_amplitude.values, float)
    I, Q = dq.I.values, dq.Q.values
    ns = rec["namespace"]
    rate = ns["rates"][q]
    kappa = ns["kappa_priors"][q]
    full = analyse_ladder(x, amps, I, Q, BAND, rate, kappa)
    truth = full["f01_offset"]
    short = short_sweep_warning(BAND, ns["sweep_lengths"][q]) is not None
    res = {"truth_offset": truth, "truth_error": full["f01_offset_error"], "alpha_full": full["anharmonicity_estimate"],
           "drive_scale_full": full["drive_scale"], "identity_full": full["line_identity"],
           "lines_full": full["lines"], "grid": []}
    K = amps.size  # 11, centre level 5
    for m in range(-4, 5):
        levels = [j + 2 + m for j in range(7)]
        levels = [k for k in levels if 0 <= k < K]
        for F in np.arange(-80e6, 80e6 + 1, 5e6):
            sel = (x >= F - 120e6 - 1) & (x <= F + 120e6 + 1)
            r = analyse_ladder(x[sel], amps[levels], I[np.ix_(levels, sel)], Q[np.ix_(levels, sel)], BAND, rate,
                               kappa / 2.0**m if kappa else None)
            err = abs(r["f01_offset"] - truth) if np.isfinite(r["f01_offset"]) else np.inf
            if short or r["line_identity"] != "0-1":
                outcome = "refused"
            elif err < 1e6:
                outcome = "pass"
            elif err < 5e6:
                outcome = "imprecise"  # the right line, located to 1-5 MHz
            else:
                outcome = "wrong"
            res["grid"].append({"m": m, "F": float(F), "levels": len(levels), "outcome": outcome,
                                "f01": r["f01_offset"], "identity": r["line_identity"],
                                "drive_scale": r["drive_scale"], "alpha": r["anharmonicity_estimate"]})
    # The two-photon trap: windows that hold the 0->2 line but not f_01 (partial crops of the wide map).
    res["trap"] = []
    for F in (-130e6, -140e6, -150e6, -160e6):
        sel = (x >= max(F - 120e6, x.min()) - 1) & (x <= F + 120e6 + 1)
        levels = list(range(2, 9))
        r = analyse_ladder(x[sel], amps[levels], I[np.ix_(levels, sel)], Q[np.ix_(levels, sel)], BAND, rate, kappa)
        res["trap"].append({"F": float(F), "window": [float(x[sel].min()), float(x[sel].max())],
                            "identity": r["line_identity"], "f01": r["f01_offset"],
                            "exponent": r["onset_exponent"], "lines": r["lines"]})
    # No drive prior: the absolute ladder 0.005..0.64, from the wide levels nearest in amplitude.
    absolute = 0.005 * 2.0 ** np.arange(8)
    idx = [int(np.argmin(np.abs(np.log(amps / a)))) for a in absolute]
    keep = [i for i, a in zip(idx, absolute) if abs(np.log(amps[i] / a)) < np.log(1.5)]
    keep = sorted(set(keep))
    if len(keep) >= 3:
        sel = (x >= -120e6 - 1) & (x <= 120e6 + 1)
        r = analyse_ladder(x[sel], amps[keep], I[np.ix_(keep, sel)], Q[np.ix_(keep, sel)], BAND, rate, None)
        res["no_prior"] = {"levels": [float(amps[i]) for i in keep], "identity": r["line_identity"],
                           "f01": r["f01_offset"], "ok": bool(np.isfinite(r["f01_offset"]) and abs(r["f01_offset"] - truth) < 1e6)}
    return res


def replay_03b(backend: str, q: str) -> dict | None:
    parts = load_all(backend, "wide3b-", q)
    if not parts:
        return None
    xs, Is, Qs = [], [], []
    for ds, rec in parts:
        dq = ds.sel(qubit=q)
        xs.append(dq.flux_bias.values)
        Is.append(dq.I.values)
        Qs.append(dq.Q.values)
        det = ds.detuning.values.astype(float)
    phi = np.concatenate(xs)
    order = np.argsort(phi)
    phi = phi[order]
    I = np.concatenate(Is, axis=1)[:, order]
    Q = np.concatenate(Qs, axis=1)[:, order]
    full = analyse_flux_map(det, phi, I, Q, BAND)
    short = short_sweep_warning(BAND, parts[0][1]["namespace"]["sweep_lengths"][q]) is not None
    step = float(np.median(np.diff(phi)))
    phi20 = (phi.max() - phi.min()) / 5.0  # the wide span is +-2.5 x phi20
    res = {"truth_x0": full["x0"], "truth_x0_error": full["x0_error"], "truth_f0": full["f0"],
           "truth_curvature": full["curvature"], "phi20": phi20, "chi2_full": full["chi2_red"],
           "tracked_full": full["tracked"], "grid": []}
    tol = max(0.5e-3 if backend != "qolab" else 1e-3, 0.0)
    for kd in range(-9, 10):
        delta = kd * step
        csel = np.abs(phi - delta) <= 1.5 * phi20 + 0.5 * step
        for F in (-10e6, -5e6, 0.0, 5e6, 10e6):
            fsel = (det >= F - 40e6 - 1) & (det <= F + 30e6 + 1)
            r = analyse_flux_map(det[fsel], phi[csel], I[np.ix_(fsel, csel)], Q[np.ix_(fsel, csel)], BAND)
            ok = (np.isfinite(r["x0"]) and abs(r["x0"] - full["x0"]) < max(3 * np.hypot(r["x0_error"], full["x0_error"]), tol)
                  and abs(r["f0"] - full["f0"]) < 0.3e6 and phi[csel].min() < r["x0"] < phi[csel].max())
            outcome = "pass" if ok and not short else ("refused" if short or not np.isfinite(r["x0"]) or r["warnings"] else "wrong")
            if np.isfinite(r["x0"]) and not ok and not (phi[csel].min() < r["x0"] < phi[csel].max()):
                outcome = "refused"  # the node proposes nothing when the turning point is outside the sweep
            res["grid"].append({"delta": float(delta), "delta_phi20": float(delta / phi20), "F": float(F),
                                "outcome": outcome, "x0": r["x0"], "x0_error": r["x0_error"], "f0": r["f0"],
                                "columns": int(csel.sum()), "tracked": r["tracked"], "warnings": r["warnings"]})
    return res


def main() -> None:
    out = {}
    for backend, qs in QUBITS.items():
        for q in qs:
            a = replay_03a(backend, q)
            b = replay_03b(backend, q)
            out[f"{backend}/{q}"] = {"03a": a, "03b": b}
            if a:
                n = len(a["grid"])
                p = sum(g["outcome"] == "pass" for g in a["grid"])
                w = sum(g["outcome"] == "wrong" for g in a["grid"])
                print(f"{backend}/{q} 03a: {p}/{n} pass, {w} wrong; trap {[t['identity'] for t in a['trap']]}; "
                      f"no-prior {a.get('no_prior', {}).get('ok')}")
            if b:
                n = len(b["grid"])
                p = sum(g["outcome"] == "pass" for g in b["grid"])
                w = sum(g["outcome"] == "wrong" for g in b["grid"])
                print(f"{backend}/{q} 03b: {p}/{n} pass, {w} wrong; truth x0 {b['truth_x0'] * 1e3:+.2f} mV")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    (OUT / "replays.json").write_text(json.dumps(clean(out), indent=1))
    print("saved", OUT / "replays.json")


if __name__ == "__main__":
    main()
