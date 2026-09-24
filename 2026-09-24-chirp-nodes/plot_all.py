"""Redraw every node figure of the 24 Sep test with the committed analysis and plotting (no QPU).

The node figures saved during the runs came from the analysis as it stood at launch; this re-fits
each saved dataset with the committed fit_raw_data and draws it with the nodes' own plot functions,
into figures/<tag>_ladder.png (03a) and figures/<tag>_flux_map.png (03b, chirp or saturation mode).
The Q2 wide flux map, taken in two halves, is drawn as one map.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

WT = Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"
sys.path.insert(0, str(WT))
from calibration_utils.qubit_spectroscopy_chirp import analysis as a3a  # noqa: E402
from calibration_utils.qubit_spectroscopy_chirp.plotting import plot_ladder  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp import analysis as a3b  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.plotting import plot_flux_map  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
BACKENDS = ("arbel", "qolab", "gilboa")


def qubit(state: dict, name: str):
    q = state["qubits"][name]
    ops = q["xy"]["operations"]

    def op(key):
        v = ops[key]
        return ops[v.split("/")[-1]] if isinstance(v, str) else v

    x180 = op("x180")
    return SimpleNamespace(
        name=name, f_01=float(q["f_01"]), T1=q.get("T1"), freq_vs_flux_01_quad_term=q.get("freq_vs_flux_01_quad_term"),
        xy=SimpleNamespace(RF_frequency=float(q["xy"]["RF_frequency"]),
                           operations={"x180": SimpleNamespace(length=x180["length"], amplitude=x180["amplitude"])}),
        z=SimpleNamespace(flux_point=q["z"].get("flux_point", "joint"), joint_offset=q["z"].get("joint_offset", 0.0),
                          independent_offset=q["z"].get("independent_offset", 0.0)))


def fake_node(rec: dict, qubits: list, pulse: str = "chirp"):
    ns = dict(rec.get("namespace") or {})
    if ns.get("ladders"):
        ns["ladders"] = {k: SimpleNamespace(played=np.asarray(v["played"], float)) for k, v in ns["ladders"].items()}
    ns["qubits"] = qubits
    return SimpleNamespace(namespace=ns, parameters=SimpleNamespace(chirp_band_in_mhz=20.0, pulse=pulse, min_columns=7))


def draw(backend: str, tag: str, ds: xr.Dataset, rec: dict, state: dict) -> None:
    qubits = [qubit(state, q) for q in rec["qubits"]]
    if rec["node"] == "03a_qubit_spectroscopy_chirp":
        _, fits = a3a.fit_raw_data(ds, fake_node(rec, qubits))
        fig = plot_ladder(ds, qubits, {k: asdict(v) for k, v in fits.items()})
        fig.savefig(FIG / f"{backend}_{tag}_ladder.png", dpi=95, bbox_inches="tight")
    else:
        pulse = (rec.get("params") or {}).get("pulse", "chirp")
        ds_fit, fits = a3b.fit_raw_data(ds, fake_node(rec, qubits, pulse))
        fig = plot_flux_map(ds, qubits, ds_fit, {k: asdict(v) for k, v in fits.items()})
        fig.savefig(FIG / f"{backend}_{tag}_flux_map.png", dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for be in BACKENDS:
        state = json.loads((HERE / be / "state/state.json").read_text())
        halves = {}
        for f in sorted((HERE / be).glob("*.json")):
            tag = f.stem
            if not tag.startswith(("r0-", "end-", "sat-", "wide3a-", "wide3b-", "smoke-")):
                continue
            nc = f.with_suffix(".nc")
            if not nc.exists():
                continue
            rec = json.loads(f.read_text())
            ds = xr.open_dataset(nc).load()
            if tag.endswith(("-lo", "-hi")):
                halves[tag[-2:]] = (ds, rec)
                continue
            draw(be, tag, ds, rec, state)
            print(be, tag)
        if halves:
            (lo, rec), (hi, _) = halves["lo"], halves["hi"]
            n = lo.sizes["flux_index"]
            hi = hi.assign_coords(flux_index=hi.flux_index + n)
            ds = xr.concat([lo, hi], dim="flux_index")
            rec = dict(rec, namespace=dict(rec["namespace"], flux_offsets={
                q: list(lo.flux_bias.sel(qubit=q).values) + list(hi.flux_bias.sel(qubit=q).values) for q in rec["qubits"]}))
            draw(be, "wide3b-Q2", ds, rec, state)
            print(be, "wide3b-Q2 (two halves)")


if __name__ == "__main__":
    main()
