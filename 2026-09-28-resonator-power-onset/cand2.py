"""Candidate B: find where the line has clearly moved, then walk down it while the local slope stays significant.
No level reference is needed near the foot, so a noisy low-power block cannot set the threshold."""
import numpy as np
from cand import clean_track, smooth
def noise_scale(c, e):
    """Global factor between the fit errors and the actual row-to-row scatter (second differences), >= 1."""
    if len(c) < 5: return 1.0
    r = (c[1:-1] - 0.5 * (c[:-2] + c[2:])) / np.sqrt(e[1:-1] ** 2 + 0.25 * (e[:-2] ** 2 + e[2:] ** 2))
    return max(1.0, 1.4826 * float(np.median(np.abs(r - np.median(r)))))
def wslope(P, c, s):
    w = 1 / s ** 2; X = np.column_stack([np.ones_like(P), P - P.mean()])
    A = X.T @ (X * w[:, None]); b = X.T @ (w * c)
    try: cov = np.linalg.inv(A)
    except np.linalg.LinAlgError: return 0.0, np.inf
    beta = cov @ b; return float(beta[1]), float(np.sqrt(cov[1, 1]))
def propose(P, r, window_db=4.0, t_min=2.0, confident=0.3, k_conf=5.0, buffer=1.0, use_depth=True, ref_db=8.0):
    out = dict(prop=None, onset=None, how=None)
    lw = r["linewidth"]
    if not lw: out["how"] = "no line"; return out
    c, e, depth, nb, plateau = r["dressed_position"], r["dressed_position_error"], r["dressed_depth"], r["n_branches"], r["plateau_contrast"]
    idx = clean_track(c, e, depth, nb, lw, plateau)
    if len(idx) < 10: out["how"] = "too few clean rows"; return out
    g = noise_scale(c[idx], e[idx]); s = g * e[idx]; Pi = P[idx]; ci = c[idx]
    ref = np.flatnonzero(Pi <= Pi[0] + ref_db)
    if len(ref) < 6: ref = np.arange(6)
    c0 = float(np.median(ci[ref]))
    zt, st = smooth(ci - c0, s)
    kk = np.arange(len(ref), len(idx))
    if not len(kk): out["how"] = "nothing above the reference"; return out
    M = float(zt[kk][np.argmax(np.abs(zt[kk]))]); d = np.sign(M) or 1.0
    spread = 1.4826 * float(np.median(np.abs(ci[ref] - c0))); s_c0 = 1.25 * spread / np.sqrt(len(ref))
    out.update(M_lw=M / lw, g=g)
    conf = (d * zt > np.maximum(confident * abs(M), k_conf * np.sqrt(st ** 2 + s_c0 ** 2)))
    kc = next((k for k in kk if k + 2 < len(idx) and conf[k:k + 3].all()), None)
    drift_onset = None
    if kc is not None:
        foot = kc
        for k in range(kc, 0, -1):     # walk down: is the line moving right above row k?
            sel = (Pi >= Pi[k]) & (Pi <= Pi[k] + window_db)
            if sel.sum() < 4: continue
            b, sb = wslope(Pi[sel], ci[sel], s[sel])
            if d * b / sb >= t_min: foot = k
            else: break
        drift_onset = float(Pi[foot]); out["confident"] = float(Pi[kc])
    node_onset = r["onset_power"] if r["onset_found"] and r["onset_mechanism"] in ("fade", "bistability") else None
    onset, how = drift_onset, ("drift" if drift_onset is not None else None)
    if node_onset is not None and (onset is None or node_onset < onset): onset, how = node_onset, r["onset_mechanism"]
    if onset is None: out["how"] = "no onset"; return out
    below = idx[P[idx] < onset]
    if use_depth and plateau and len(below):
        dm = np.array([np.median(depth[max(0, i - 1):i + 2]) for i in range(len(P))])
        keep = below[dm[below] >= 0.9 * plateau]
        if len(keep): below = keep
    top = float(P[below[-1]]) if len(below) else float(P[idx[0]])
    out.update(prop=top - buffer, onset=onset, how=how, top_stable=top)
    return out
