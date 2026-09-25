"""kappa from the ring-down against kappa from lineshape fits; leftover field and qubit excitation after a wait (no QPU).

usage: kappa_analyse.py            writes kappa/summary.json and prints one table per backend

Per qubit and round:
  ring-down   the silent part of the 40 ns sliced trace, z(t) = A exp((-kappa/2 + 2 pi i delta) t) + C, fitted from
              80 ns after the drive stops; ring-up the same exponential towards the steady state while driven
  circle      the notch circle fit (calibration_utils.resonator_spectroscopy.circle_fit) of the +-10 MHz complex sweep:
              kappa/2pi = f_r / Q_l
  lorentz     a Lorentzian dip fitted to |S21|^2 on a linear background: its FWHM is kappa/2pi for a notch resonator
  amp_fwhm    the full width at half depth of the |S21| dip, the quantity 02a/02b report as a linewidth
  02a, 02b    what the nodes themselves reported in the same session (ab/<backend>/*.json)
Leftover: the fitted top-power ring-down extrapolated to a wait w, over the field the lowest 02b power produces (the
linear 'low' ring-down amplitude scaled down in power) -- the fraction of the next point's signal it adds at the start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "code/QM/qua-libs/qualibration_graphs/superconducting"))
from calibration_utils.resonator_spectroscopy.circle_fit import fit_notch_resonator  # noqa: E402

QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
WAITS_NS = {"1us": 1000, "3us": 3000, "12us": 12000}
LEVELS = ("low", "ro", "top")


def trace(rd: dict, four: bool) -> np.ndarray:
    """Complex envelope per level and slice: I = out1.cos + out2.sin, Q = out1.(-sin) + out2.cos."""
    if four:
        return (rd["rd0"] + rd["rd2"]) + 1j * (rd["rd1"] + rd["rd3"])
    return 2 * (rd["rd0"] + 1j * rd["rd1"])


def fit_decay(t, z, kappa0=2 * np.pi * 1e6, towards=False):
    """z = C + A exp((-k/2 + 2 pi i d) t); returns kappa, its error, A, C, delta and the fit.

    For a ring-down (towards=False) C is held at 0: with the drive off nothing but the emitted field reaches the
    input, and a free offset trades off against a slow decay and pulls kappa low. For a ring-up C is the steady state.
    """
    c0 = z[-5:].mean() if towards else 0j
    a0 = z[0] - c0
    free_c = 1.0 if towards else 0.0

    def model(p):
        a, k, d, c = p[0] + 1j * p[1], p[2], p[3], (p[4] + 1j * p[5]) * free_c
        return c + a * np.exp((-k / 2 + 2j * np.pi * d) * t)

    def res(p):
        r = model(p) - z
        return np.concatenate([r.real, r.imag])

    best = None
    for k0 in (kappa0, kappa0 * 2, kappa0 / 2):
        for d0 in (0.0, 0.3e6, -0.3e6):
            p0 = [a0.real, a0.imag, k0, d0, c0.real, c0.imag]
            try:
                r = least_squares(res, p0, x_scale="jac", max_nfev=4000)
            except Exception:  # noqa: BLE001
                continue
            if r.success and r.x[2] > 0 and (best is None or r.cost < best.cost):
                best = r
    if best is None:
        return None
    npar = 6 if towards else 4
    dof = max(1, 2 * len(z) - npar)
    s2 = 2 * best.cost / dof
    try:
        jac = best.jac[:, :npar]
        cov = np.linalg.inv(jac.T @ jac) * s2
        err = np.concatenate([np.sqrt(np.diag(cov)), np.full(6 - npar, np.nan)])
    except np.linalg.LinAlgError:
        err = np.full(6, np.nan)
    p = best.x
    return dict(kappa=p[2], kappa_err=err[2], A=complex(p[0], p[1]), C=complex(p[4], p[5]) * free_c, delta=p[3],
                delta_err=err[3], model=model(p), rms=np.sqrt(s2))


def lorentz_power(f, s):
    """Lorentzian dip on a linear background fitted to |S21|^2; FWHM = kappa/2pi for a notch resonator."""
    p2 = np.abs(s) ** 2
    i0 = int(np.argmin(p2))
    base = np.median(np.concatenate([p2[:40], p2[-40:]]))
    depth0 = max(1e-9, (base - p2[i0]) / base)

    def model(p):
        b0, b1, dpt, f0, w = p
        return (b0 + b1 * (f - f.mean())) * (1 - dpt / (1 + (2 * (f - f0) / w) ** 2))

    p0 = [base, 0.0, depth0, f[i0], 1e6]
    try:
        r = least_squares(lambda p: model(p) - p2, p0, x_scale="jac", max_nfev=4000,
                          bounds=([0, -np.inf, 0, f.min(), 1e4], [np.inf, np.inf, 1.0, f.max(), 20e6]))
    except Exception:  # noqa: BLE001
        return None
    s2 = 2 * r.cost / max(1, len(f) - 5)
    try:
        err = np.sqrt(np.diag(np.linalg.inv(r.jac.T @ r.jac) * s2))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    return dict(fwhm=abs(r.x[4]), fwhm_err=err[4], f0=r.x[3], depth=r.x[2])


def amp_fwhm(f, s):
    """Full width at half depth of the |S21| dip (smoothed over 3 points), on the median of the outer wings."""
    a = np.convolve(np.abs(s), np.ones(3) / 3, mode="same")
    base = np.median(np.concatenate([a[3:40], a[-40:-3]]))
    i0 = int(np.argmin(a[3:-3])) + 3
    half = base - (base - a[i0]) / 2
    left = i0
    while left > 0 and a[left] < half:
        left -= 1
    right = i0
    while right < len(a) - 1 and a[right] < half:
        right += 1
    if left == 0 or right == len(a) - 1:
        return float("nan")

    def cross(i, j):
        return f[i] + (half - a[i]) * (f[j] - f[i]) / (a[j] - a[i])

    return float(cross(right - 1, right) - cross(left, left + 1))


def node_widths(be, q):
    out = {"02a": [], "02b": {}}
    for f in sorted((HERE / "ab" / be).glob(f"*-{q}.json")):
        rec = json.loads(f.read_text())
        r = (rec.get("fit_results") or {}).get(q) or {}
        if rec["condition"] == "02a":
            if r.get("fwhm") is not None:
                out["02a"].append(float(r["fwhm"]))
        elif r.get("linewidth") is not None:
            out["02b"].setdefault(rec["condition"], []).append(float(r["linewidth"]))
    return out


def analyse(be, q, rnd):
    mf = HERE / "kappa" / be / f"r{rnd}_meta.json"
    if not mf.exists():
        return None
    meta = json.loads(mf.read_text()).get(q)
    if meta is None:
        return None
    out = {"meta": {k: meta[k] for k in ("committed_power_dbm", "top_dbm", "bottom_dbm", "f_r", "readout_len",
                                         "depletion_time", "T1", "levels_dbm", "four_demods")}}
    chunk = meta["chunk_ns"] * 1e-9
    l_on = meta["l_on"] * 1e-9
    rdf = HERE / "kappa" / be / f"r{rnd}_{q}_ringdown.npz"
    if rdf.exists():
        rd = dict(np.load(rdf))
        z = trace(rd, meta["four_demods"])
        n = z.shape[1]
        t = (np.arange(n) + 0.5) * chunk
        out["ringdown"] = {}
        for i, lv in enumerate(LEVELS):
            down = t > l_on + 80e-9
            up = (t > 80e-9) & (t < l_on)
            fd = fit_decay(t[down] - l_on, z[i, down])
            fu = fit_decay(t[up], z[i, up], towards=True)
            out["ringdown"][lv] = dict(
                kappa_down=fd["kappa"] if fd else np.nan, kappa_down_err=fd["kappa_err"] if fd else np.nan,
                delta_down=fd["delta"] if fd else np.nan, A_down=abs(fd["A"]) if fd else np.nan,
                kappa_up=fu["kappa"] if fu else np.nan, kappa_up_err=fu["kappa_err"] if fu else np.nan,
                steady=float(np.abs(z[i, (t > l_on - 400e-9) & (t < l_on - 40e-9)].mean())),
                noise=float(np.std(np.abs(z[i, -10:] - z[i, -10:].mean()))))
        # leftover field from a top-power point over the field of the lowest 02b power, at the start of the next point
        top, low = out["ringdown"]["top"], out["ringdown"]["low"]
        k = top["kappa_down"]
        a_bottom = low["A_down"] * 10 ** ((meta["bottom_dbm"] - meta["levels_dbm"]["low"]) / 20)
        out["leftover_over_bottom"] = {w: float(top["A_down"] * np.exp(-k * ns * 1e-9 / 2) / a_bottom)
                                       for w, ns in WAITS_NS.items()}
        # the same between two neighbouring points at one power (power outer / frequency inner)
        k_ro = out["ringdown"]["ro"]["kappa_down"]
        out["leftover_same_power"] = {w: float(np.exp(-k_ro * ns * 1e-9 / 2)) for w, ns in WAITS_NS.items()}
    swf = HERE / "kappa" / be / f"r{rnd}_{q}_sweep.npz"
    if swf.exists():
        sw = dict(np.load(swf))
        f = meta["f_r"] + np.asarray(meta["sweep_hz"], float)
        out["sweep"] = {}
        for i, lv in enumerate(("low", "ro")):
            s = sw["sw_I"][i] + 1j * sw["sw_Q"][i]
            i0 = int(np.argmin(np.abs(s)))
            cf = fit_notch_resonator(f, s, f_r_guess=f[i0], q_l_guess=f[i0] / 1e6)
            lz = lorentz_power(f, s)
            out["sweep"][lv] = dict(
                circle_kappa_2pi=(cf.f_r / cf.q_l) if cf.converged else np.nan,
                circle_kappa_2pi_err=(cf.f_r / cf.q_l ** 2 * cf.q_l_stderr) if cf.converged else np.nan,
                circle_converged=bool(cf.converged), circle_residual=cf.residual, q_l=cf.q_l, q_c=cf.q_c, q_i=cf.q_i,
                f_r=cf.f_r, lorentz_fwhm=lz["fwhm"] if lz else np.nan, lorentz_fwhm_err=lz["fwhm_err"] if lz else np.nan,
                depth=lz["depth"] if lz else np.nan, amp_fwhm=amp_fwhm(f, s))
    ppf = HERE / "kappa" / be / f"r{rnd}_{q}_pump.npz"
    if ppf.exists():
        pp = dict(np.load(ppf))
        g = complex(pp["g_I"], pp["g_Q"])
        e = complex(pp["e_I"], pp["e_Q"])
        zz = pp["pp_I"] + 1j * pp["pp_Q"]
        axis = e - g
        proj = ((zz - g) * np.conj(axis)).real / abs(axis) ** 2
        perp = ((zz - g) * np.conj(axis)).imag / abs(axis) ** 2
        out["pump"] = dict(taus_ns=meta["taus_ns"], excited=proj.tolist(), perpendicular=perp.tolist(),
                           separation=abs(axis))
    return out


def fmt(v, s=1e-6, spec=".3f"):
    return "—" if v is None or not np.isfinite(v) else format(v * s, spec)


def main():
    summary = {}
    for be, qs in QUBITS.items():
        print(f"\n{be}: kappa/2pi [MHz] -- ring-down low/ro/top, ring-up low, circle low/ro, |S21|^2 Lorentzian low, "
              f"|S21| half-depth width low, 02a fwhm, 02b linewidth (A)")
        for q in qs:
            for rnd in (1, 2):
                a = analyse(be, q, rnd)
                if a is None:
                    continue
                summary[f"{be}/{q}/r{rnd}"] = a
                rd = a.get("ringdown", {})
                sw = a.get("sweep", {})
                nw = node_widths(be, q)
                k2 = lambda lv, key="kappa_down": rd.get(lv, {}).get(key, np.nan) / (2 * np.pi)  # noqa: E731
                print(f"  {q:4s} r{rnd}  rd {fmt(k2('low'))}±{fmt(rd.get('low', {}).get('kappa_down_err', np.nan) / 2 / np.pi)} "
                      f"/{fmt(k2('ro'))}/{fmt(k2('top'))}  up {fmt(k2('low', 'kappa_up'))}  "
                      f"circle {fmt(sw.get('low', {}).get('circle_kappa_2pi', np.nan))}/"
                      f"{fmt(sw.get('ro', {}).get('circle_kappa_2pi', np.nan))}  "
                      f"lor {fmt(sw.get('low', {}).get('lorentz_fwhm', np.nan))}  "
                      f"ampw {fmt(sw.get('low', {}).get('amp_fwhm', np.nan))}  "
                      f"02a {','.join(fmt(v) for v in nw['02a'])}  02b {','.join(fmt(v) for v in nw['02b'].get('A', []))}")
                if "leftover_over_bottom" in a:
                    lo = a["leftover_over_bottom"]
                    sp = a["leftover_same_power"]
                    print(f"        leftover top->bottom at 1/3/12 us: {lo['1us']:.2e} {lo['3us']:.2e} {lo['12us']:.2e};"
                          f" same power {sp['1us']:.2e} {sp['3us']:.2e} {sp['12us']:.2e}")
                if "pump" in a:
                    p = a["pump"]
                    ex = dict(zip(p["taus_ns"], p["excited"]))
                    print(f"        excited after a top-power readout: " +
                          " ".join(f"{t / 1000:g}us {ex[t]:+.3f}" for t in (200, 1000, 3000, 12000, 40000, 80000)))
    (HERE / "kappa" / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda x: None
                                                          if not isinstance(x, (float, int)) else float(x)))


if __name__ == "__main__":
    main()
