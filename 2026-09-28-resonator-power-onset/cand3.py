"""Candidate C: photon-number drift model. Below the break (fade/fork, or the point where the line has moved 50% of
its largest sustained move), fit c(P) = c0 + A*10^(P/10) with inverse-variance weights. Onset = where the fitted drift
reaches tau = max(F*|M|, K*sigma_local) -- a fixed fraction of the line's own move, never below what this map can
resolve at that power. Proposal = top clean row below the onset - buffer."""
import numpy as np
from scipy.optimize import least_squares
from cand import clean_track, smooth
from cand2 import noise_scale
def propose(P, r, f_move=0.10, k_sigma=2.0, cut=0.5, buffer=1.0, use_depth=True, ref_db=8.0, s_exp=1.0, fit_s=False, f_lw=None, f_cap=None, cut_lw=None, guard=False):
    out = dict(prop=None, onset=None, how=None)
    lw = r["linewidth"]
    if not lw: out["how"] = "no line"; return out
    c, e, depth, nb, plateau = r["dressed_position"], r["dressed_position_error"], r["dressed_depth"], r["n_branches"], r["plateau_contrast"]
    idx = clean_track(c, e, depth, nb, lw, plateau)
    node_break = r["onset_power"] if r["onset_found"] and r["onset_mechanism"] in ("fade", "bistability") else None
    if node_break is not None: idx = idx[P[idx] < node_break]        # the line is not one clean dip above the break
    if len(idx) < 10: out["how"] = "too few clean rows"; return out
    g = noise_scale(c[idx], e[idx]); s = g * e[idx]; Pi = P[idx]; ci = c[idx]
    ref = np.flatnonzero(Pi <= Pi[0] + ref_db)
    if len(ref) < 6: ref = np.arange(6)
    c0 = float(np.median(ci[ref])); zt, st = smooth(ci - c0, s)
    kk = np.arange(len(ref), len(idx))
    if not len(kk): out["how"] = "nothing above the reference"; return out
    M = float(zt[kk][np.argmax(np.abs(zt[kk]))]); d = np.sign(M) or 1.0
    out.update(M_lw=M / lw, g=g)
    if abs(M) < max(4 * float(np.median(st[kk])), 0.02 * lw):
        out["how"] = "no significant move"
        if node_break is None: return out
    cut_at = cut * abs(M) if cut_lw is None else min(cut * abs(M), cut_lw * lw) if cut_lw > 0 else cut * abs(M)
    kc = next((k for k in kk if d * zt[k] > cut_at), len(idx) - 1)
    fit_rows = np.arange(0, kc + 1)
    if len(fit_rows) < 8: fit_rows = np.arange(0, min(len(idx), 8))
    x = Pi[fit_rows]; y = ci[fit_rows]; w = 1 / s[fit_rows]
    def res(p):
        ex = p[2] if fit_s else s_exp
        return (p[0] + d * np.exp(p[1]) * 10 ** (ex * (x - x.max()) / 10) - y) * w
    p0 = [c0, np.log(max(abs(M) * cut, 1e-3 * lw)), s_exp]
    try:
        fit = least_squares(res, p0 if fit_s else p0[:2], bounds=([-np.inf, -50, 0.3], [np.inf, 50, 3]) if fit_s else (-np.inf, np.inf))
    except Exception:
        out["how"] = "fit failed"; return out
    A = np.exp(fit.x[1]); ex = fit.x[2] if fit_s else s_exp
    # tau: F of the move, but never below what the rows near the onset resolve (K sigma of a 3-row mean)
    sig_loc = float(np.median(st[max(0, kc - 6):kc + 1]))
    tau_phys = f_move * abs(M) if f_lw is None else (f_lw * lw if f_cap is None else min(f_lw * lw, f_cap * abs(M)))
    tau = max(tau_phys, k_sigma * sig_loc)
    if guard and k_sigma * sig_loc > 2 * tau_phys:
        # the rows near the onset cannot resolve the threshold: an onset found here would be set by the noise
        out.update(how="noise-limited", tau_lw=tau / lw); return out
    drift_onset = float(x.max() + 10 / ex * np.log10(tau / A)) if A > 0 else None
    if drift_onset is not None and drift_onset < Pi[0]: drift_onset = float(Pi[0])      # moving from the first row
    out.update(tau_lw=tau / lw, A=A, ex=ex, fit_top=float(x.max()), fit_bottom=float(x.min()), fit_c0=float(fit.x[0]), d=float(d), lw=lw)
    onset, how = drift_onset, "drift"
    if drift_onset is not None and drift_onset > Pi[-1] + 0.01:
        onset, how = None, "no significant move"
    if node_break is not None and (onset is None or node_break < onset): onset, how = node_break, r["onset_mechanism"]
    if onset is None:
        out["how"] = how; return out
    below = idx[P[idx] < onset]
    if use_depth and plateau and len(below):
        dm = np.array([np.median(depth[max(0, i - 1):i + 2]) for i in range(len(P))])
        keep = below[dm[below] >= 0.9 * plateau]
        if len(keep): below = keep
    top = float(P[below[-1]]) if len(below) else float(P[idx[0]])
    out.update(prop=top - buffer, onset=onset, how=how, top_stable=top)
    return out
