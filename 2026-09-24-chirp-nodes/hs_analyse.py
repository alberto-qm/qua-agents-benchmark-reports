"""Analysis of hs_test.py (no QPU) -> hs_test/summary.json, hs_ladder.png, hs_maps.png.

Reference f01: Lorentzian on the fine saturation scan of the same session (signal projected on |0> -> x180, x180 = 1).
Ladder: an erf-box fit for each pulse (lin_up/lin_dn over 20 MHz, hs_up/hs_dn over 40 MHz) at each drive; the 0->1 box is
the one within a band of the reference, the 0->2 box the one alpha/2 below. Reported per pulse family: the centre from
up and down sweeps and their mean against the reference, box height and edge width, the lowest drive where the 0->1 box
is full (>= 0.8 of its largest height) and the lowest where a 0->2 box appears.
Maps: per-column box (chirp) or Lorentzian (saturation) centres and the parabola's sweet spot, via analyse_flux_map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "hs_test"
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs-chirp/qualibration_graphs/superconducting"))
from calibration_utils.qubit_spectroscopy_chirp.boxes import find_boxes, fit_peak, noise_sigma  # noqa: E402
from calibration_utils.qubit_spectroscopy_vs_flux_chirp.analysis import analyse_flux_map  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
ALPHA = {"qD2": 208e6, "qC3": 209e6, "qD5": 208e6, "qA6": 199e6, "qB4": 198e6}


def proj(I, Q, r0, r1):
    ax = r1 - r0
    return ((np.asarray(I) - r0[0]) * ax[0] + (np.asarray(Q) - r0[1]) * ax[1]) / (ax @ ax)


def load(be, tag):
    f = D / f"{be}_{tag}.npz"
    return np.load(f) if f.exists() else None


def main():
    out = {}
    fig_l, axes_l = plt.subplots(5, 4, figsize=(17, 16))
    fig_m, axes_m = plt.subplots(5, 3, figsize=(14, 17))
    row = 0
    for be in ("gilboa", "arbel"):
        meta_f = D / f"{be}_meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        levels = meta["levels"]
        for q, p in meta["plan"].items():
            res = {"T1_us": p["T1"] * 1e6, "sweep_ns": p["tau"]}
            # reference
            ref = load(be, f"{q}_ref")
            f_ref = None
            if ref is not None:
                r0 = np.array([ref["r0_I"].mean(), ref["r0_Q"].mean()])
                r1 = np.array([ref["r1_I"].mean(), ref["r1_Q"].mean()])
                s = proj(ref["a_I"], ref["a_Q"], r0, r1)
                x = np.asarray(p["ref_df"])
                b = fit_peak(x, s, noise_sigma(s), min_snr=4)
                if b is not None:
                    f_ref = b.centre
                    res["reference"] = dict(offset=b.centre, error=b.centre_error, fwhm=b.width, height=b.height)
            # ladder
            lad = load(be, f"{q}_ladder")
            if lad is not None:
                r0 = np.array([lad["r0_I"].mean(), lad["r0_Q"].mean()])
                r1 = np.array([lad["r1_I"].mean(), lad["r1_Q"].mean()])
                x = np.asarray(p["centres"])
                res["pulses"] = {}
                for col, fam in enumerate(("lin", "hs")):
                    band = 20e6 if fam == "lin" else 40e6
                    fam_res = {}
                    for d in ("up", "dn"):
                        key = f"{fam}_{d}"
                        rows = np.array([proj(lad[f"{key}_{j}_I"], lad[f"{key}_{j}_Q"], r0, r1) for j in range(len(levels))])
                        sig = noise_sigma(rows, axis=1)
                        per = []
                        for j in range(len(levels)):
                            boxes = find_boxes(x, rows[j], sig, band, min_snr=4)
                            guess = f_ref if f_ref is not None else 0.0
                            b01 = [b for b in boxes if abs(b.centre - guess) < band]
                            b02 = [b for b in boxes if abs(b.centre - (guess - ALPHA[q] / 2)) < band]
                            b1 = max(b01, key=lambda b: b.height) if b01 else None
                            per.append(dict(level=levels[j], rabi_mhz=p["pulses"][key]["played"][j] * p["kappa"] / 1e6,
                                            c01=None if b1 is None else b1.centre, e01=None if b1 is None else b1.centre_error,
                                            h01=None if b1 is None else b1.height, w01=None if b1 is None else b1.edge_width,
                                            inside=None if b1 is None else b1.inside, two_photon=bool(b02),
                                            h02=max((b.height for b in b02), default=0.0)))
                        fam_res[d] = per
                        a = axes_l[row, 2 * col + (0 if d == "up" else 1)]
                        for j in range(len(levels)):
                            a.plot(x / 1e6, rows[j], lw=1.1, marker="o", ms=2, color=plt.cm.viridis(j / (len(levels) - 1)),
                                   label=f"{per[j]['rabi_mhz']:.1f} MHz")
                        if f_ref is not None:
                            a.axvline(f_ref / 1e6, color="#e6194b", lw=0.9, ls=":")
                            a.axvline((f_ref - ALPHA[q] / 2) / 1e6, color="#ff9f40", lw=0.9, ls=":")
                        a.set_ylim(-0.15, 1.15)
                        a.set_title(f"{q} ({be}, T1 {p['T1'] * 1e6:.1f} µs) · {'linear 20 MHz' if fam == 'lin' else 'hyperbolic secant 40 MHz'} "
                                    f"{'up' if d == 'up' else 'down'}", fontsize=8.5)
                        a.set_xlabel("band centre − stored f01 [MHz]")
                        if col == 0 and d == "up":
                            a.set_ylabel("population (x180 = 1)")
                        a.legend(fontsize=6.5, frameon=False, loc="upper left")
                    # summary per level: mean of up and down, and the full / 0->2 onsets
                    lv = []
                    for j in range(len(levels)):
                        u, dn = fam_res["up"][j], fam_res["dn"][j]
                        cs = [c for c in (u["c01"], dn["c01"]) if c is not None]
                        lv.append(dict(level=levels[j], rabi_mhz=u["rabi_mhz"], up=u["c01"], dn=dn["c01"],
                                       mean=float(np.mean(cs)) if len(cs) == 2 else None,
                                       bias_up=None if (u["c01"] is None or f_ref is None) else u["c01"] - f_ref,
                                       bias_mean=None if (len(cs) < 2 or f_ref is None) else float(np.mean(cs)) - f_ref,
                                       height=np.nanmean([h for h in (u["h01"], dn["h01"]) if h is not None]) if (u["h01"] or dn["h01"]) else None,
                                       edge=np.nanmean([w for w in (u["w01"], dn["w01"]) if w is not None]) if (u["w01"] or dn["w01"]) else None,
                                       two_photon=u["two_photon"] or dn["two_photon"]))
                    hs_ = [x_["height"] or 0 for x_ in lv]
                    hmax = max(hs_) if hs_ else 0
                    full = next((x_["level"] for x_ in lv if hmax > 0 and (x_["height"] or 0) >= 0.8 * hmax), None)
                    tp = next((x_["level"] for x_ in lv if x_["two_photon"]), None)
                    res["pulses"][fam] = dict(levels=lv, full_level=full, two_photon_level=tp, max_height=hmax, per_direction=fam_res)
            # maps
            for k, kind in enumerate(("lin_up", "hs_up", "sat")):
                m = load(be, f"{q}_map_{kind}")
                a = axes_m[row, k]
                if m is None:
                    continue
                fr = np.asarray(p["map_lin"] if kind == "lin_up" else p["map_hs"] if kind == "hs_up" else p["map_sat"])
                phi = np.asarray(p["offsets"])
                I, Q = m["a_I"].reshape(len(fr), len(phi)), m["a_Q"].reshape(len(fr), len(phi))
                band = 20e6 if kind == "lin_up" else 40e6
                r = analyse_flux_map(fr, phi, I, Q, band, "saturation" if kind == "sat" else "chirp", min_columns=6)
                res.setdefault("maps", {})[kind] = dict(x0=r["x0"], x0_error=r["x0_error"], f0=r["f0"], f0_error=r["f0_error"],
                                                        curvature=r["curvature"], tracked=r["tracked"], columns=r["columns"],
                                                        chi2=r["chi2_red"], warnings=r["warnings"],
                                                        precision=float(np.nanmedian(r["errors"])) if np.isfinite(r["errors"]).any() else None)
                from calibration_utils.qubit_spectroscopy_chirp.boxes import project_iq
                s = project_iq(I, Q)
                a.pcolormesh(phi * 1e3, fr / 1e6, s / (np.nanpercentile(s, 99.5) or 1), shading="nearest", cmap="viridis", vmin=-0.1, vmax=1.1)
                a.plot(phi * 1e3, r["centres"] / 1e6, "o", ms=3.5, mfc="none", mec="#e6194b")
                if np.isfinite(r["x0"]):
                    xx = np.linspace(phi.min(), phi.max(), 200)
                    a.plot(xx * 1e3, (r["f0"] + r["curvature"] * (xx - r["x0"]) ** 2) / 1e6, color="#e6194b", lw=1)
                    a.set_ylim(fr.min() / 1e6, fr.max() / 1e6)
                    a.text(0.02, 0.03, f"sweet spot {r['x0'] * 1e3:+.2f} ± {r['x0_error'] * 1e3:.2f} mV\n{r['tracked']}/{r['columns']} columns",
                           transform=a.transAxes, color="w", fontsize=8)
                else:
                    a.text(0.02, 0.03, f"no arc ({r['tracked']}/{r['columns']} columns)", transform=a.transAxes, color="w", fontsize=8)
                a.set_title(f"{q} · {'linear chirp' if kind == 'lin_up' else 'hyperbolic secant' if kind == 'hs_up' else 'saturation'}", fontsize=9)
                a.set_xlabel("flux offset from idle [mV]")
                a.set_ylabel("frequency − stored f01 [MHz]")
            out[f"{be}/{q}"] = res
            row += 1
    fig_l.tight_layout()
    fig_l.savefig(HERE / "hs_ladder.png", dpi=85)
    fig_m.tight_layout()
    fig_m.savefig(HERE / "hs_maps.png", dpi=85)
    (D / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        ref = v.get("reference", {})
        print(k, f"T1 {v['T1_us']:.1f} us; ref {ref.get('offset', float('nan')) / 1e6:+.2f} MHz fwhm {ref.get('fwhm', float('nan')) / 1e6:.2f}")
        for fam, pr in (v.get("pulses") or {}).items():
            cells = [f"{x['rabi_mhz']:.1f}MHz:" + (f"up{x['bias_up'] / 1e6:+.2f}" if x["bias_up"] is not None else "up—")
                     + (f"/mean{x['bias_mean'] / 1e6:+.2f}" if x["bias_mean"] is not None else "") + f"/h{(x['height'] or 0):.2f}"
                     + (f"/e{x['edge'] / 1e6:.1f}" if x["edge"] else "") + ("/2γ" if x["two_photon"] else "") for x in pr["levels"]]
            print(f"   {fam}: full at x{pr['full_level']}, 0->2 from x{pr['two_photon_level']}; " + "  ".join(cells))
        for kind, mm in (v.get("maps") or {}).items():
            x0 = mm["x0"]
            print(f"   map {kind}: " + (f"x0 {x0 * 1e3:+.2f}±{mm['x0_error'] * 1e3:.2f} mV f0 {mm['f0'] / 1e6:+.2f} {mm['tracked']}/{mm['columns']} "
                                        f"prec {(mm['precision'] or 0) / 1e3:.0f}k chi2 {mm['chi2']:.1f}" if x0 is not None and np.isfinite(x0)
                                        else f"no arc {mm['tracked']}/{mm['columns']}"))


if __name__ == "__main__":
    main()
