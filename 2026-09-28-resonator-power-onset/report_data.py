"""Per-map figures and results.json for the 28 Sep onset report: the node's proposal against the photon-model rule (cand3,
final settings), with the fitted drift, the threshold, the eye picks and IQCC's readout power."""
import json, pickle, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cand3, score
FINAL = dict(f_lw=0.05, f_cap=0.25, cut_lw=0.25, guard=True)
H = os.path.expanduser
IQCC = {f"n7 {r['backend']} {r['q']} {r['variant']}": r["ref"] for r in json.load(open(H("~/qab-runs/n7-analysis.json")))}
os.makedirs("figures", exist_ok=True)
M = pickle.load(open("maps.pkl", "rb"))
def order(m):
    return (m["eye"] is None, not m["label"].startswith("n7"), m["label"])
M.sort(key=order)
fin = lambda v: v is not None and np.isfinite(v)
rows = []
for m in M:
    r = m["r"]; P, x = m["P"], m["x"]; o = cand3.propose(P, r, **FINAL)
    node = r["optimal_power"] if fin(r["optimal_power"]) else None
    iq = IQCC.get(m["label"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    V = m["V"] / np.median(m["V"], axis=1, keepdims=True)
    a0.pcolormesh(x / 1e6, P, V, shading="auto", vmin=np.percentile(V, 1), vmax=np.percentile(V, 99), rasterized=True)
    marks = [("node", node, "k", "--"), ("new", o["prop"], "m", "-"), ("eye", m["eye"], "g", ":"), ("IQCC", iq, "tab:blue", "-.")]
    lw = r["linewidth"]
    if lw:
        c, e = r["dressed_position"], r["dressed_position_error"]; ref = r["dressed_frequency"]
        clean = cand3.clean_track(c, e, r["dressed_depth"], r["n_branches"], lw, r["plateau_contrast"])
        a0.plot(c[clean] / 1e6, P[clean], ".", color="r", ms=2.5)
        a0.set_xlim((ref - 6 * lw) / 1e6, (ref + 6 * lw) / 1e6)
        s_ = (c - ref) / lw; ok = np.isfinite(e); other = ok.copy(); other[clean] = False
        a1.errorbar(P[clean], s_[clean], e[clean] / lw, fmt="o", ms=2.5, lw=0.7, color="tab:blue", label="line centre (used)")
        a1.plot(P[other], s_[other], "x", color="0.6", ms=3.5, label="line centre (not used)")
        if "fit_c0" in o:
            Pm = np.linspace(P[clean][0], min(P.max(), o["fit_top"] + 4), 300)
            model = o["fit_c0"] + o["d"] * o["A"] * 10 ** (o["ex"] * (Pm - o["fit_top"]) / 10)
            a1.plot(Pm, (model - ref) / lw, color="m", lw=1.3, label="drift fit  c0 + A·10^(P/10)")
            a1.axvspan(o["fit_bottom"], o["fit_top"], color="m", alpha=0.05)
            a1.axhline((o["fit_c0"] - ref) / lw + o["d"] * o["tau_lw"], color="m", lw=0.8, ls=":", label=f"threshold {o['tau_lw']:.3f} lw")
            if fin(o["onset"]) and o["how"] == "drift":
                a1.plot([o["onset"]], [(o["fit_c0"] - ref) / lw + o["d"] * o["tau_lw"]], "D", color="m", ms=6, label=f"onset {o['onset']:+.1f}")
        vals = s_[clean] if len(clean) else s_[ok]
        if len(vals):
            lo, hi = np.nanmin(vals), np.nanmax(vals); pad = 0.1 * max(hi - lo, 0.1)
            a1.set_ylim(lo - pad, hi + pad)
        b = a1.twinx(); b.plot(P, r["dressed_depth"] / (r["plateau_contrast"] or 1), color="orange", lw=0.8, alpha=0.7)
        b.set_ylim(0, 1.8); b.set_ylabel("dip depth / plateau", color="orange", fontsize=8); b.tick_params(labelsize=7, colors="orange")
    for name, v, col, ls in marks:
        if fin(v):
            a0.axhline(v, color=col, ls=ls, lw=1.4); a1.axvline(v, color=col, ls=ls, lw=1.4, label=f"{name} {v:+.1f}")
    a0.set_xlabel("detuning [MHz]"); a0.set_ylabel("readout power [dBm]")
    a1.set_xlabel("readout power [dBm]"); a1.set_ylabel("line shift [linewidths]"); a1.set_xlim(P.min(), P.max()); a1.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="lower left", framealpha=0.85)
    lws = f"lw {lw / 1e6:.2f} MHz · " if lw else ""
    fig.suptitle(f"{m['label']}   ({m['mid']}, {m['t']})   {lws}node: {r['onset_mechanism']} · new: {o['how']}", fontsize=10)
    fig.tight_layout(); fn = f"figures/{m['mid']}_{m['q']}.png"; fig.savefig(fn, dpi=72); plt.close(fig)
    rows.append(dict(mid=m["mid"], q=m["q"], label=m["label"], t=m["t"], window=[float(P.min()), float(P.max()), len(P)],
                     fs=m["fs"], node=node, node_onset=r["onset_power"], node_mech=r["onset_mechanism"], new=o["prop"],
                     new_onset=o["onset"], new_how=o["how"], M_lw=o.get("M_lw"), tau_lw=o.get("tau_lw"), eye=m["eye"], iqcc=iq,
                     linewidth=lw, figure=fn))
res = {"rows": rows, "final": FINAL}
# scores: node vs the final rule (and the rejected candidates), with the stress set
score._stress_cache = pickle.load(open("stress_cache.pkl", "rb")) if os.path.exists("stress_cache.pkl") else {}
sc = {}
for name, mod, kw in (("node", "node", {}), ("noise threshold (A)", "cand", {}), ("slope walk-down (B)", "cand2", {}),
                      ("photon model, 10 % of move", "cand3", {}), ("photon model, final", "cand3", FINAL)):
    eye, spreads, st = score.score(score.get_rule(mod, kw))
    sc[name] = dict(eye=eye, spreads={g: list(v) for g, v in spreads.items()}, stress={k: list(v) for k, v in st.items()})
res["scores"] = sc; res["eye_order"] = [m["label"] for m in score.M if m["eye"] is not None]
res["groups"] = score.GROUPS
json.dump(res, open("results.json", "w"), indent=1, default=float)
print(len(rows), "figures")
