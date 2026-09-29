"""Shared loader: every recent 02b map, re-fitted with the frozen n7 node analysis (per-row tracked centres, errors, depths)."""
import sys, os, glob, json, datetime, pickle
from types import SimpleNamespace
import numpy as np, xarray as xr
sys.path.insert(0, os.path.expanduser("~/qab-runs/qua-libs-n7/qualibration_graphs/superconducting"))
from calibration_utils.resonator_spectroscopy_vs_amplitude import analysis as A
from calibration_utils.resonator_spectroscopy_vs_amplitude.parameters import NodeSpecificParameters
BASE = {k: f.default for k, f in NodeSpecificParameters.model_fields.items()}
DATA = os.path.expanduser("~/.qualibrate/user_storage/tinycal/data")
H = os.path.expanduser
# eye picks (the user's table, 28 Sep) keyed by node run id
EYE = {"m-C88Y2R": -36, "m-31N907": -35, "m-W5DGFJ": -32, "m-YY4P5E": -30, "m-WS7TZF": -45,
       "m-Y23DDJ": -22, "m-DVCW85": -22, "m-C6N288": -47, "m-61QP2M": -44}
def n7_labels():
    stamp = open(H("~/qab-runs/n7-stamp.txt")).read().strip(); out = {}
    for run in glob.glob(H(f"~/code/QM/tinycal/runs/n7_*_{stamp}")):
        name = run.split("/")[-1].replace(f"_{stamp}", ""); be = name.split("_")[1]; v = name.rsplit("-", 1)[1]
        for evp in glob.glob(run + "/*/events.jsonl"):
            q = evp.split("/")[-2]
            for l in open(evp):
                e = json.loads(l)
                if e.get("node") == "resonator_spectroscopy_vs_power" and e.get("node_run_id"):
                    out[e["node_run_id"]] = f"n7 {be} {q} {v}"
    return out
def fit(values, x, P, **over):
    return A._fit_single_qubit(values, x, P, SimpleNamespace(parameters=SimpleNamespace(**{**BASE, **over})))
def maps(since=datetime.datetime(2026, 9, 25), min_points=40):
    labels = n7_labels(); out = []
    for p in sorted(glob.glob(f"{DATA}/*/ds_raw.h5")):
        if os.path.getmtime(p) < since.timestamp(): continue
        mid = p.split("/")[-2]
        try: ds = xr.open_dataset(p)
        except Exception: continue
        if "power" not in ds.coords or "detuning" not in ds.coords or len(ds.power) < min_points: continue
        for q in map(str, ds.qubit.values):
            P = ds.power.values.astype(float); x = ds.detuning.values.astype(float)
            V = ds.IQ_abs.sel(qubit=q).transpose("power", "detuning").values.astype(float)
            fs = float(ds.full_scale_power_dbm.sel(qubit=q)) if "full_scale_power_dbm" in ds.coords else float("nan")
            t = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
            out.append(dict(mid=mid, q=q, P=P, x=x, V=V, fs=fs, t=t, label=labels.get(mid, f"{t[:5]} {q}"), eye=EYE.get(mid)))
    return out
if __name__ == "__main__":
    M = maps(); print(len(M))
    cache = []
    for m in M:
        r = fit(m["V"], m["x"], m["P"]); m["r"] = r; cache.append(m)
        print(f"{m['mid']} {m['label']:22s} {len(m['P']):4d} node {r['optimal_power']:+.1f} via {r['onset_mechanism']}")
    pickle.dump(cache, open("maps.pkl", "wb"))
