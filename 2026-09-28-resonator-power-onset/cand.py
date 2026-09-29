"""Candidate 02b onset rule: departure from the low-power level, measured in the map's own noise, with a floor
at a fraction of the line's total move. Uses the node's per-row Lorentzian centres/errors/depths unchanged."""
import numpy as np
P_REF_DB = 8.0      # fixed reference: the lowest 8 dB of clean rows (min 6 rows)
K_SIGMA = 3.0       # moved = smoothed deviation beyond 3 sigma (row noise + reference uncertainty) ...
F_MOVE = 0.10       # ... and beyond 10% of the line's largest sustained move
PERSIST = 3         # for 3 consecutive clean rows
BUFFER = 1.0
def clean_track(c, e, depth, nb, lw, plateau):
    ok = np.isfinite(c) & np.isfinite(e) & (e <= 0.15 * lw) & (nb <= 1)
    if plateau: ok &= depth >= 0.5 * plateau
    kept = []
    for i in np.flatnonzero(ok):
        if len(kept) >= 3 and abs(c[i] - np.median(c[kept[-3:]])) > max(0.5 * lw, 4 * e[i]):
            continue                      # a jump to another feature, not the line moving
        kept.append(int(i))
    return np.array(kept, int)
def smooth(z, s):
    """3-row inverse-variance mean along the clean track (ends use 2 rows)."""
    zt = np.empty_like(z); st = np.empty_like(s)
    for k in range(len(z)):
        sl = slice(max(0, k - 1), k + 2); w = 1 / s[sl] ** 2
        zt[k] = np.sum(w * z[sl]) / w.sum(); st[k] = 1 / np.sqrt(w.sum())
    return zt, st
def propose(P, r, f_move=F_MOVE, k_sigma=K_SIGMA, persist=PERSIST, ref_db=P_REF_DB, use_depth=True):
    out = dict(prop=None, onset=None, how=None)
    lw = r["linewidth"]
    if not lw: out["how"] = "no line"; return out
    c, e, depth, nb, plateau = r["dressed_position"], r["dressed_position_error"], r["dressed_depth"], r["n_branches"], r["plateau_contrast"]
    idx = clean_track(c, e, depth, nb, lw, plateau)
    if len(idx) < 10: out["how"] = "too few clean rows"; return out
    ref = idx[P[idx] <= P[idx[0]] + ref_db]
    if len(ref) < 6: ref = idx[:6]
    c0 = float(np.median(c[ref]))
    spread = 1.4826 * float(np.median(np.abs(c[ref] - c0)))
    sys_ = np.sqrt(max(0.0, spread ** 2 - float(np.median(e[ref])) ** 2))
    s = np.sqrt(e[idx] ** 2 + sys_ ** 2)
    s_c0 = 1.25 * max(spread, float(np.median(e[ref]))) / np.sqrt(len(ref))
    above = idx[len(ref):]
    if len(above) < persist: out["how"] = "nothing above the reference"; return out
    zt_all, st_all = smooth(c[idx] - c0, s)
    kk = np.arange(len(ref), len(idx))
    M = float(zt_all[kk][np.argmax(np.abs(zt_all[kk]))]); d = np.sign(M) or 1.0
    z = d * zt_all; st = np.sqrt(st_all ** 2 + s_c0 ** 2)
    tau = np.maximum(k_sigma * st, f_move * abs(M))
    moved = z > tau
    out.update(M_lw=M / lw, tau_lw=float(np.median(tau[kk])) / lw, c0=c0, ref_top=float(P[ref[-1]]))
    onset_k = None
    for k in range(len(ref), len(idx) - persist + 1):
        if moved[k:k + persist].all(): onset_k = k; break
    # the node's own fade/fork onsets, which fire where the line stops being one clean dip
    node_onset = r["onset_power"] if r["onset_found"] and r["onset_mechanism"] in ("fade", "bistability") else None
    onset = float(P[idx[onset_k]]) if onset_k is not None else None
    how = "drift" if onset is not None else None
    if node_onset is not None and (onset is None or node_onset < onset): onset, how = node_onset, r["onset_mechanism"]
    if onset is None: out["how"] = "no onset"; return out
    below = idx[(P[idx] < onset)]
    if use_depth and plateau:
        # depth test on a 3-row median, so one noisy row cannot end the stable region
        dm = np.array([np.median(depth[max(0, i - 1):i + 2]) for i in range(len(P))])
        below = below[dm[below] >= 0.9 * plateau] if (dm[below] >= 0.9 * plateau).any() else below
    top = float(P[below[-1]]) if len(below) else float(P[idx[0]])
    out.update(prop=top - BUFFER, onset=onset, how=how, top_stable=top)
    return out
