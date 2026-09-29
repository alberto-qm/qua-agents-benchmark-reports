"""Score onset rules: eye error (9 maps), repeat spread (same resonator, same flux condition), stress (noise x2,
every 2nd row, top 5 dB cut, bottom 10 dB cut) -- each stress re-runs the node's full per-row analysis."""
import pickle, json, sys, os, importlib, numpy as np
from load import fit
M = pickle.load(open("maps.pkl", "rb")); byid = {m["mid"]: m for m in M}
GROUPS = {  # same resonator, same flux condition by the map's shape; full-window maps only (>= 10 dB below the onset)
 "qA5": ["m-4N39V1", "m-E1DFAE", "m-AYREE1", "m-HWQZNS", "m-1B73CA", "m-4D65AB", "m-TVYSCQ"],
 "qD1": ["m-8FC89E", "m-YPX0XZ", "m-C0P0RG", "m-VJS07S"],
 "Q5": ["m-66RF3F", "m-S07069", "m-29FGJ1", "m-C2DZE2", "m-0P9VGW", "m-ZEKVAW"],
 "Q1 ss": ["m-ZYGFDP", "m-2VMJTG"], "Q1 0V": ["m-C88Y2R", "m-EXHHDN"], "Q2 0V": ["m-C6N288", "m-MZPX9R"],
 "Q2 ss": ["m-RM2DYE", "m-SJHPME"], "arbel qC2": ["m-J4A9CS", "m-KBYFQJ"], "arbel qD3": ["m-31N907", "m-W5DGFJ"],
 "gilboa qC2": ["m-DVCW85", "m-0WST5K"], "gilboa qC3": ["m-4NS7C2", "m-XFK0T9"]}
STRESS_SET = [m["mid"] for m in M if m["label"].startswith("n7") or m["mid"] in ("m-EXHHDN", "m-2VMJTG", "m-29FGJ1", "m-C0P0RG", "m-AYREE1", "m-MZPX9R", "m-SJHPME")]
def node_rule(P, r, **kw):
    v = r["optimal_power"]; return dict(prop=float(v) if np.isfinite(v) else None)
def get_rule(name, kw):
    if name == "node": return lambda P, r: node_rule(P, r)
    mod = importlib.import_module(name); return lambda P, r: mod.propose(P, r, **kw)
def row_noise(V):
    return 1.4826 * np.median(np.abs(np.diff(V, axis=1)), axis=1, keepdims=True) / np.sqrt(2)
_stress_cache = {}
def stressed(m, kind, seed=0):
    key = (m["mid"], kind, seed)
    if key in _stress_cache: return _stress_cache[key]
    P, x, V = m["P"], m["x"], m["V"]
    if kind == "noise2":
        V = V + np.random.default_rng(seed).normal(size=V.shape) * np.sqrt(3) * row_noise(V)
    elif kind == "dec2":
        sel = np.arange(seed % 2, len(P), 2); P, V = P[sel], V[sel]
    elif kind == "top5":
        sel = P <= P.max() - 5; P, V = P[sel], V[sel]
    elif kind == "bot10":
        sel = P >= P.min() + 10; P, V = P[sel], V[sel]
    out = (P, fit(V, x, P)); _stress_cache[key] = out; return out
def clip(m, v):   # the gilboa eye picks are the highest power writable at the port's current full scale
    return v if v is None or not np.isfinite(m["fs"]) else min(v, m["fs"] - 20)
def score(rule, stress=True, verbose=False):
    eye = []
    for m in M:
        if m["eye"] is None: continue
        v = rule(m["P"], m["r"])["prop"]; eye.append(None if v is None else clip(m, v) - m["eye"])
    spreads = {}
    for g, ids in GROUPS.items():
        vals = [rule(byid[i]["P"], byid[i]["r"])["prop"] for i in ids]
        got = [v for v in vals if v is not None]
        spreads[g] = (max(got) - min(got) if len(got) > 1 else None, len(got), len(vals))
    st = {}
    if stress:
        for kind, seeds in (("noise2", (0, 1)), ("dec2", (0, 1)), ("top5", (0,)), ("bot10", (0,))):
            d = []; lost = 0; gained = 0
            for mid in STRESS_SET:
                m = byid[mid]; base = rule(m["P"], m["r"])["prop"]
                for sd in seeds:
                    P2, r2 = stressed(m, kind, sd); v = rule(P2, r2)["prop"]
                    if base is None and v is None: continue
                    if base is None: gained += 1; continue
                    if v is None: lost += 1; continue
                    d.append(abs(v - base))
            d = np.array(d); st[kind] = (float(np.median(d)), float(np.percentile(d, 90)), int((d > 3).sum()), lost, gained, len(d))
    return eye, spreads, st
def report(name, eye, spreads, st):
    ok = [abs(x) for x in eye if x is not None]
    print(f"== {name}")
    print(f"   eye: {' '.join('—' if x is None else f'{x:+.1f}' for x in eye)}   | MAE {np.mean(ok):.1f} dB over {len(ok)}/9, max {max(ok):.1f}")
    sp = [v[0] for v in spreads.values() if v[0] is not None]
    print("   repeat spread: " + "  ".join(f"{g} {'—' if v[0] is None else f'{v[0]:.1f}'}({v[1]}/{v[2]})" for g, v in spreads.items()) + f"  | median {np.median(sp):.1f} max {max(sp):.1f}")
    for k, v in st.items():
        print(f"   {k:6s}: |Δ| median {v[0]:.1f} p90 {v[1]:.1f}  >3dB {v[2]}/{v[5]}  lost {v[3]} gained {v[4]}")
if __name__ == "__main__":
    specs = json.loads(sys.argv[1])
    for spec in specs:
        name, kw = spec[0], (spec[1] if len(spec) > 1 else {})
        report(f"{name} {kw}", *score(get_rule(name, kw), stress="--nostress" not in sys.argv))
    pickle.dump(_stress_cache, open("stress_cache.pkl", "wb"))
