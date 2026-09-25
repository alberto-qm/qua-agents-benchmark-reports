"""Build 2026-09-24-chirp-spectroscopy-nodes.html from 2026-09-24-chirp-nodes/ (no QPU, no network).

The data were taken on 24 Sep 2026 by the scripts in that folder (run_backend.py, run_fixes.py), replayed
offline by replay.py and reanalyse.py, and plotted by plot_summary.py and plot_fixes.py. Usage:
    python make_chirp_nodes_report.py [--sync] [--artifact PATH]
--sync copies the run folder ~/qab-runs/chirp-nodes-20260924 into 2026-09-24-chirp-nodes/ first.
"""
from __future__ import annotations

import base64
import collections
import json
import shutil
import sys
from html import escape as esc
from pathlib import Path

import make_fwcmp2_report as base

HERE = Path(__file__).parent
DATA = HERE / "2026-09-24-chirp-nodes"
RUNS = Path.home() / "qab-runs/chirp-nodes-20260924"
OUT = HERE / "2026-09-24-chirp-spectroscopy-nodes.html"
TITLE = "Chirp spectroscopy nodes"
QUBITS = {"arbel": ["qB4", "qA5", "qD1"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
BRANCH = "feat/chirp-spectroscopy"
COMMITS = {"nodes": "be998ef", "fix": "576d230", "fix09a": "e22fa9c", "flat": "7dc6187", "grey": "11f0670", "updown": "80c3db3"}


def sync():
    DATA.mkdir(exist_ok=True)
    for sub in ("arbel", "qolab", "gilboa", "fixes/qolab", "fixes/gilboa", "replay", "figures", "shortT1",
                "satladder_default/arbel", "satladder_default/qolab", "satladder_default/gilboa",
                "satladder/arbel", "satladder/qolab", "satladder/gilboa", "hs_test",
                "waits", "waits/arbel", "waits/qolab", "waits/gilboa", "noT1", "noT1/arbel", "noT1/qolab", "noT1/gilboa"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for f in (RUNS / sub).glob("*"):
            if f.is_file() and f.suffix in (".json", ".nc", ".png", ".log", ".jsonl", ".out", ".txt"):
                if sub.startswith("satladder/") and f.suffix == ".nc":
                    continue  # the superseded 150-shot ladder: fit results and logs only
                shutil.copy(f, DATA / sub / f.name)
    for f in RUNS.glob("*"):
        if f.is_file() and f.suffix in (".py", ".sh", ".png", ".out", ".log"):
            shutil.copy(f, DATA / f.name)


def img(path, caption, alt, max_width=None):
    b64 = base64.b64encode((DATA / path).read_bytes()).decode()
    cap = f"max-width:{max_width}px;" if max_width else ""
    return (f'<figure><figcaption class="small">{caption}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:100%;{cap}height:auto;border-radius:6px"></figure>')


def qpu_of(be, tag):
    f = DATA / be / f"{tag}.json"
    return json.loads(f.read_text())["qpu_s"] if f.exists() else float("nan")


def figrow(items):
    """Small single-qubit figures side by side: [(path, caption, alt), ...]."""
    cells = []
    for path, caption, alt in items:
        b64 = base64.b64encode((DATA / path).read_bytes()).decode()
        cells.append(f'<figure><figcaption class="small">{caption}</figcaption>'
                     f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:100%;height:auto;border-radius:6px"></figure>')
    return f'<div class="figrow">{"".join(cells)}</div>'


def backend_figures(be):
    """Chirp against saturation, qubits as columns: the 1D spectroscopy (maps and line cuts) and the flux maps."""
    qs = QUBITS[be]
    return "\n".join([
        f'<h3>{be} · {", ".join(qs)}</h3>',
        img(f"figures/cols_1d_{be}.png",
            f"{be}, qubit spectroscopy. Rows 1–2: the chirp's wide ladder (11 drives, ×1/32…×32 of the x180 prediction, 4 µs or "
            "T1/5 sweeps over 20 MHz), map and line cuts. Rows 3–4: the saturation node at its defaults (9 drives, ×1/16…×16 of its "
            "default, 20 µs, 0.15 MHz steps), map and line cuts. Line cuts are labelled with the Rabi frequency; red dotted: f₀₁ from "
            "the fine saturation scan, orange dotted: the 0→2 line.", f"chirp and saturation spectroscopy for {be}"),
        img(f"figures/cols_flux_{be}.png",
            f"{be}, flux maps at the start of the session: 03b chirp at its defaults (row 1) and 03b in saturation mode (row 2). "
            "Circles: each column's centre; curve and star: the fitted parabola and sweet spot (grey: not proposed). QPU is the "
            "job's, shared when it held several qubits.", f"chirp and saturation flux maps for {be}"),
    ])


def appendix():
    parts = []

    def group(title, pattern, caption):
        names = sorted(p.name for p in (DATA / "figures").glob(pattern))
        if not names:
            return ""
        body = []
        for name in names:
            be, tag = name.split("_", 1)
            tag = tag.rsplit("_", 1)[0].replace("_flux", "")
            single = tag.count("-") <= 2 and "flux_map" in name
            body.append(img(f"figures/{name}", f"{be} · {tag} · {caption}", f"{be} {tag}", max_width=480 if single else None))
        return f"<details><summary>{title} ({len(names)} figures)</summary>{''.join(body)}</details>"

    parts.append(group("Start of session, as the nodes draw them: 03a chirp ladders", "*_r0-03a-*_ladder.png", "start of session"))
    parts.append(group("Start of session, as the nodes draw them: 03b chirp and saturation-mode maps", "*_r0-03b-*_flux_map.png",
                       "start of session"))
    parts.append(group("Start of session, as the nodes draw them: 03b saturation-mode maps", "*_sat-*_flux_map.png", "saturation mode"))
    parts.append(group("End of session: 03a and 03b chirp at their defaults again", "*_end-*.png", "end of session"))
    parts.append(group("Wide 03a ladders used for the replays (11 levels, ±200 MHz)", "*_wide3a-*_ladder.png", "wide ladder"))
    parts.append(group("Wide 03b maps used for the replays (35 columns, ±2.5× the 20 MHz offset)", "*_wide3b-*_flux_map.png", "wide map"))
    fines = []
    for be, qs in QUBITS.items():
        for q in qs:
            f = DATA / be / f"fine-{q}_amplitude.png"
            if f.exists():
                fines.append(img(f"{be}/fine-{q}_amplitude.png", f"{be} {q} · fine saturation scan (saturation node 03a, ±5 MHz, 0.3 MHz Rabi)", f"fine scan {be} {q}"))
    parts.append(f"<details><summary>Fine saturation scans ({len(fines)} figures)</summary>{''.join(fines)}</details>")
    parts.append(group("Smoke run, qolab Q1", "*_smoke-*.png", "smoke run"))
    waits = []
    for be in QUBITS:
        for f in sorted((DATA / "waits" / be).glob("*.png")):
            tag = f.stem.rsplit("_", 1)[0] if f.stem.endswith("_ladder") else f.stem.replace("_flux_map", "")
            waits.append(img(f"waits/{be}/{f.name}", f"{be} · {tag} · up-and-down sweeps", f"{be} {tag}"))
    if waits:
        parts.append(f"<details><summary>2 T1 against 5 T1, as the nodes draw them ({len(waits)} figures)</summary>{''.join(waits)}</details>")
    return "\n".join(p for p in parts if p)


def table(head, rows, cls="grid small"):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def num(v, fmt="{:.2f}", dash="–"):
    if v is None:
        return dash
    try:
        if v != v:
            return dash
    except TypeError:
        return dash
    return f'<span class="num">{fmt.format(v)}</span>'


def load():
    live = json.loads((DATA / "replay/live.json").read_text())
    rep = json.loads((DATA / "replay/replays.json").read_text())
    states = {be: json.loads((DATA / be / "state.json").read_text()) if (DATA / be / "state.json").exists() else None
              for be in QUBITS}
    return live, rep, states


def pick(live, be, q, prefix):
    for k, v in live.items():
        if k.startswith(f"{be}/{q}/{prefix}"):
            return v
    return None


def fixes():
    out = {}
    for f in sorted((DATA / "fixes").glob("*/*.json")):
        rec = json.loads(f.read_text())
        out[rec["tag"]] = rec
    return out


def build() -> str:
    live, rep, _ = load()
    fx = fixes()
    T1 = {"qB4": 30.9, "qA5": 10.9, "qD1": 18.5, "Q1": 47.7, "Q2": 79.9, "Q5": 42.0, "qD2": 1.34, "qC3": 1.80, "qD5": 43.3}

    # --- live table ---------------------------------------------------------------------------
    rows = []
    diffs = []
    fine_abs = {}
    for be, qs in QUBITS.items():
        for q in qs:
            a0, a1 = pick(live, be, q, "r0-03a"), pick(live, be, q, "end-03a")
            b0, b1 = pick(live, be, q, "r0-03b"), pick(live, be, q, "end-03b")
            sat, fine = pick(live, be, q, "sat-"), pick(live, be, q, "fine-")
            fa0, fa1 = (a0 or {}).get("final") or {}, (a1 or {}).get("final") or {}
            fb0, fb1 = (b0 or {}).get("final") or {}, (b1 or {}).get("final") or {}
            fs = (sat or {}).get("final") or {}
            fine_off = None

            def a_cell(f):
                if not f:
                    return "–"
                ident = f.get("identity")
                if ident != "0-1":
                    return f'<span class="muted">{esc(ident)}</span>'
                return (f'{num(f["f01_offset"] / 1e6, "{:+.2f}")} ± {num(f["f01_error"] / 1e6, "{:.2f}")}')

            def b_cell(f):
                if not f or not f.get("proposed"):
                    why = "refused"
                    if f and f.get("warnings"):
                        w = f["warnings"][0]
                        why = "refused: short sweep" if "too short" in w else f"refused: {f.get('tracked', 0)}/{f.get('columns', 0)} columns"
                    return f'<span class="muted">{why}</span>'
                return f'{num(f["x0"] * 1e3, "{:+.2f}")} ± {num(f["x0_error"] * 1e3, "{:.2f}")}'

            fine_cell = "–"
            if fine and fine["live"].get("f_01"):
                r0rec = json.loads(next((DATA / be).glob(f"r0-03a-*.json")).read_text())
                # stored f_01 = chirp f_01 - chirp offset (both from the live 03a record)
                fr = (r0rec.get("fit_results") or {}).get(q) or {}
                if fr.get("f_01") and fr.get("frequency_shift") is not None and fr["f_01"] == fr["f_01"]:
                    stored_f = fr["f_01"] - fr["frequency_shift"]
                    fine_off = (fine["live"]["f_01"] - stored_f) / 1e6
                    good = (fine["live"].get("r_squared") or 0) > 0.6
                    if good:
                        fine_abs[f"{be}/{q}"] = fine["live"]["f_01"]
                    fine_cell = num(fine_off, "{:+.2f}") if good else '<span class="muted">no line</span>'
                    if good:
                        for fa in (fa0, fa1):
                            if fa.get("identity") == "0-1":
                                diffs.append(fa["f01_offset"] / 1e6 - fine_off)
            alpha = fa0.get("alpha") if fa0.get("identity") == "0-1" else None
            scale = fa0.get("drive_scale") if fa0.get("identity") == "0-1" else None
            rows.append([f"{be}", f"<b>{q}</b>", num(T1[q], "{:.1f}"), num(fa0.get("sweep_ns"), "{:.0f}"),
                         a_cell(fa0), a_cell(fa1), fine_cell, num(scale, "{:.2f}"), num(alpha / 1e6 if alpha else None, "{:.0f}"),
                         b_cell(fb0), b_cell(fb1),
                         (f'{num(fs["x0"] * 1e3, "{:+.2f}")} ± {num(fs["x0_error"] * 1e3, "{:.2f}")}' if fs.get("proposed")
                          else '<span class="muted">no arc</span>')])
    import statistics
    rms = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else float("nan")
    mean = statistics.mean(diffs) if diffs else float("nan")
    worst = max(abs(d) for d in diffs) if diffs else float("nan")

    # --- replay counts ------------------------------------------------------------------------
    rep_rows = []
    tot = collections.Counter()
    tot_b = collections.Counter()
    for be, qs in QUBITS.items():
        for q in qs:
            a, b = rep[f"{be}/{q}"]["03a"], rep[f"{be}/{q}"]["03b"]
            ca = collections.Counter(g["outcome"] for g in a["grid"])
            cb = collections.Counter(g["outcome"] for g in b["grid"])
            tot.update(ca)
            tot_b.update(cb)
            win = [g for g in a["grid"] if -2 <= g["m"] <= 1]
            core = sum(g["outcome"] == "pass" for g in win)
            trap = collections.Counter(t["identity"] for t in a["trap"])
            rep_rows.append([be, f"<b>{q}</b>", f'{num(ca["pass"], "{:d}")} / {sum(ca.values())}',
                             f'{num(core, "{:d}")} / {len(win)}', num(ca["wrong"], "{:d}"),
                             "all 4 refused" if trap.get("0-1", 0) == 0 else f'<b>{trap["0-1"]} of 4 wrong</b>',
                             f'{num(cb["pass"], "{:d}")} / {sum(cb.values())}', num(cb["wrong"], "{:d}"),
                             num(a.get("alpha_full") / 1e6 if a.get("alpha_full") else None, "{:.0f}")])
    n_rep = sum(tot.values()) + sum(tot_b.values())
    n_wrong = tot["wrong"] + tot_b["wrong"]

    # --- saturation replays -------------------------------------------------------------------
    rs = json.loads((DATA / "replay/replays_sat.json").read_text())
    LONG = ["arbel/qB4", "arbel/qA5", "arbel/qD1", "qolab/Q1", "qolab/Q2", "qolab/Q5", "gilboa/qD5"]
    SHORTQ = ["gilboa/qD2", "gilboa/qC3"]

    def tally(keys, chirp, inside=False, drive=None):
        c = collections.Counter()
        for k in keys:
            cells = rep[k]["03a"]["grid"] if chirp else rs[k]["grid"]["100"]
            for x in cells:
                if inside and abs(x["F"]) > 45e6:
                    continue
                if drive and not (drive[0] <= x["m"] <= drive[1]):
                    continue
                c[x["outcome"]] += 1
        n = sum(c.values())
        parts = [f'{num(100 * c[k] / n, "{:.0f}")} % {k}' for k in ("pass", "imprecise", "wrong", "refused") if c[k]]
        return ", ".join(parts), n, c

    sat_rows = []
    for label, keys, kw in (("7 with T1 ≥ 11 µs", LONG, {}),
                            ("same, line inside saturation's window, drive ×1/4…×2", LONG, dict(inside=True, drive=(-2, 1))),
                            ("gilboa qD2, qC3 (T1 1.3, 1.8 µs)", SHORTQ, {}),
                            ("same, line inside saturation's window", SHORTQ, dict(inside=True))):
        ch, nc, _ = tally(keys, True, **kw)
        sa, ns, _ = tally(keys, False, **kw)
        sat_rows.append([label, f"{nc} / {ns}", ch, sa])
    sat_wrong_long = tally(LONG, False)[2]["wrong"]
    sat_recs = [json.loads(p.read_text()) for p in (DATA / "satladder_default").glob("*/satdef-*.json")]
    sat_jobs, sat_qpu = len(sat_recs), sum(r["qpu_s"] for r in sat_recs)
    old_recs = [json.loads(p.read_text()) for p in (DATA / "satladder").glob("*/satladder-*.json")]
    old_jobs, old_qpu = len(old_recs), sum(r["qpu_s"] for r in old_recs)
    st1 = json.loads((DATA / "shortT1/shortT1_variants.json").read_text())

    # --- hyperbolic secant against the linear chirp -------------------------------------------
    hs = json.loads((DATA / "hs_test/summary.json").read_text())
    hs_qpu = sum(sum(json.loads(f.read_text())["qpu_s"].values()) for f in (DATA / "hs_test").glob("*_meta.json"))

    def fam_cell(res, fam):
        pr = (res.get("pulses") or {}).get(fam)
        if not pr or pr["full_level"] is None:
            return "–"
        lv = {x["level"]: x for x in pr["levels"]}
        full = lv[pr["full_level"]]
        tp = lv.get(pr["two_photon_level"]) if pr["two_photon_level"] is not None else None
        bias = full["bias_mean"] if full["bias_mean"] is not None else full["bias_up"]
        up = full["bias_up"]
        s = f'{num(bias / 1e6 if bias is not None else None, "{:+.2f}")} MHz'
        if up is not None and bias is not None and abs(up - bias) > 0.3e6:
            s += f' (up only {num(up / 1e6, "{:+.2f}")})'
        s += f'; full at {num(full["rabi_mhz"], "{:.1f}")} MHz; 0→2 ' + (f'from {num(tp["rabi_mhz"], "{:.0f}")} MHz' if tp else "not seen")
        return s

    def map_cell(res, kind):
        m = (res.get("maps") or {}).get(kind)
        if not m:
            return "–"
        if m["x0"] is None or m["x0"] != m["x0"]:
            return f'<span class="muted">no arc ({m["tracked"]}/{m["columns"]})</span>'
        return f'{num(m["x0"] * 1e3, "{:+.2f}")} ± {num(m["x0_error"] * 1e3, "{:.2f}")}'

    hs_rows = []
    for k in ("gilboa/qD2", "gilboa/qC3", "arbel/qA6", "arbel/qB4", "gilboa/qD5"):
        r = hs.get(k)
        if not r:
            continue
        ref = r.get("reference") or {}
        hs_rows.append([k.replace("/", " "), num(r["T1_us"], "{:.1f}"), f'{num(ref.get("offset", float("nan")) / 1e6, "{:+.2f}")} MHz',
                        fam_cell(r, "lin"), fam_cell(r, "hs"), map_cell(r, "lin_up"), map_cell(r, "hs_up"), map_cell(r, "sat")])

    # --- chirp against saturation, QPU per qubit ------------------------------------------------
    qc = json.loads((DATA / "replay/qpu_compare.json").read_text())
    cmp_rows = []
    for k, v in qc.items():
        be, q = k.split("/")
        cmp_rows.append([be, f"<b>{q}</b>", num(v["T1_us"], "{:.1f}"), num(v["chirp_03a"], "{:.1f} s"), num(v["sat_03a_default"], "{:.1f} s"),
                         num(v["sat_03a_one_drive_240"], "{:.0f} s"), num(v["sat_03a_seven_drives_240"], "{:.0f} s"),
                         num(v["chirp_03b"], "{:.1f} s"), num(v["sat_03b_same_area"], "{:.0f} s")])

    # --- what the maps in section 2 cost -----------------------------------------------------------
    fc = json.loads((DATA / "replay/figure_costs.json").read_text())

    def cost(c, star=True):
        s = num(c["qpu"], "{:.1f} s" if c["qpu"] < 20 else "{:.0f} s")
        return s + ("*" if star and c.get("shared") else "")

    fc_rows = []
    for k, v in fc.items():
        be, q = k.split("/")
        sl = v["sat_ladder"]
        fc_rows.append([be, f"<b>{q}</b>", num(v["T1_us"], "{:.1f}"),
                        f'<b>{cost(v["chirp_ladder_wide"])}</b>', cost(v["chirp_ladder_node"]),
                        f'<b>{num(sl["qpu"], "{:.0f} s")}</b> ({sl["drives"]} drives, {sl["jobs"]} jobs)',
                        f'<b>{cost(v["chirp_flux"])}</b>', f'<b>{cost(v["sat_flux"])}</b>'])

    # --- time to a frequency and a sweet spot ------------------------------------------------------
    tta = json.loads((DATA / "replay/time_to_answer.json").read_text())

    def verdict(ok, text_ok="✓", text_no="no answer"):
        return text_ok if ok else f'<span class="muted">{text_no}</span>'

    tta_rows, tot = [], {"chirp": 0.0, "sat": 0.0, "n": 0}
    for r in tta:
        c_total = r["chirp_03a"] + r["chirp_03b"]
        s_03a = r["sat_03a_tries"] * r["sat_03a_pass"]
        s_total = s_03a + r["sat_03b"]
        sat_res = {"pass": "✓", "imprecise": "1–5 MHz off", "wrong": "wrong line"}.get(r["sat_03a_result"])
        tta_rows.append([
            r["backend"], f'<b>{r["qubit"]}</b>', num(r["T1"], "{:.1f}"),
            f'{num(r["chirp_03a"], "{:.1f} s")} {verdict(r["chirp_03a_ok"], "✓", "refused")}',
            f'{num(r["chirp_03b"], "{:.1f} s")} ' + (f'±{num(r["chirp_x0_err"] * 1e3, "{:.2f}")} mV' if r["chirp_03b_ok"] else verdict(False)),
            f'<b>{num(c_total, "{:.0f} s")}</b>',
            f'{num(s_03a, "{:.1f} s")} ({r["sat_03a_tries"]} {"try" if r["sat_03a_tries"] == 1 else "tries"}) '
            + (sat_res if sat_res else '<span class="muted">refused</span>'),
            f'{num(r["sat_03b"], "{:.1f} s")} ' + (f'±{num(r["sat_x0_err"] * 1e3, "{:.2f}")} mV' if r["sat_03b_ok"] else verdict(False)),
            f'<b>{num(s_total, "{:.0f} s")}</b>'])
        if r["T1"] > 10:
            tot["chirp"] += c_total
            tot["sat"] += s_total
            tot["n"] += 1

    # --- QPU ----------------------------------------------------------------------------------
    qpu_rows = []
    all_qpu = []
    for be in QUBITS:
        runs = [json.loads(l) for l in (DATA / be / "runs.jsonl").read_text().splitlines()]
        all_qpu += [r["qpu_s"] for r in runs]
        qpu_rows.append([be, num(len(runs), "{:d}"), num(sum(r["qpu_s"] for r in runs), "{:.0f} s"),
                         num(max(r["qpu_s"] for r in runs), "{:.1f} s"), num(sum(r["wall_s"] for r in runs) / 60, "{:.0f} min")])
    fix_qpu = sum(r["qpu_s"] for r in fx.values())

    def default_job(be, prefix):
        return [json.loads(p.read_text())["qpu_s"] for p in (DATA / be).glob(f"{prefix}*.json")]

    r03a = {be: default_job(be, "r0-03a-") for be in QUBITS}
    r03b = {be: default_job(be, "r0-03b-") for be in QUBITS}

    # --- fixes --------------------------------------------------------------------------------
    def fit(tag, q):
        return (fx[tag].get("fit_results") or {}).get(q) or {}
    b_main, b_fix = fit("map03b-Q1-main", "Q1"), fit("map03b-Q1-fix", "Q1")
    ab20, ab80 = fit("ab20-qD5-fix", "qD5"), fit("ab80-qD5-fix", "qD5")
    r9m, r9f = fit("r09a-Q1-main", "Q1"), fit("r09a-Q1-fix", "Q1")
    chirp_qD5 = (pick(live, "gilboa", "qD5", "r0-03b") or {}).get("final", {})
    sat_qD5 = (pick(live, "gilboa", "qD5", "sat-") or {}).get("final", {})

    # --- up-and-down sweeps, 2 T1 against 5 T1 ---------------------------------------------------
    ws = json.loads((DATA / "waits/summary.json").read_text())
    w_runs = [json.loads(l) for f in sorted((DATA / "waits").glob("*/runs.jsonl")) for l in f.read_text().splitlines()]
    w_qpu = {f: sum(r["qpu_s"] for r in w_runs if r["factor"] == f) for f in (2, 5)}
    w_node_qpu = {(n, f): sum(r["qpu_s"] for r in w_runs if r["factor"] == f and r["node"].startswith(n)) for n in ("03a", "03b") for f in (2, 5)}

    def fin(v):
        return v is not None and v == v

    def short(r):
        return any("too short" in w for w in (r.get("warnings") or []))

    w3a_rows, w3b_rows = [], []
    wd, wh, ud2, ud5, wbd, wbs = [], [], [], [], [], []
    for be, qs in QUBITS.items():
        for q in qs:
            a = ws.get(f"{be}/{q}/03a") or {}
            a2, a5 = a.get("2") or {}, a.get("5") or {}
            ok2, ok5 = a2.get("line_identity") == "0-1", a5.get("line_identity") == "0-1"

            def acell(r, ok):
                if not r:
                    return "–"
                if not ok:
                    return f'<span class="muted">refused: {"short sweep" if short(r) else esc(str(r.get("line_identity")))}</span>'
                return f'{num(r["frequency_shift"] / 1e6, "{:+.2f}")} ± {num(r["f_01_error"] / 1e6, "{:.2f}")}'

            if ok2 and ok5:
                d = (a2["frequency_shift"] - a5["frequency_shift"]) / 1e6
                h = a2["box_height"] / a5["box_height"]
                u2 = (a2["shift_up"] - a2["shift_down"]) / 1e6
                u5 = (a5["shift_up"] - a5["shift_down"]) / 1e6
                wd.append(d); wh.append(h); ud2.append(u2); ud5.append(u5)
                extra = [num(d, "{:+.2f}"), num(h, "{:.2f}"), num(u2, "{:+.2f}"), num(u5, "{:+.2f}")]
            else:
                extra = ["–"] * 4
            w3a_rows.append([be, f"<b>{q}</b>", num(T1[q], "{:.1f}"), acell(a2, ok2), acell(a5, ok5)] + extra)

            b = ws.get(f"{be}/{q}/03b") or {}
            b2, b5 = b.get("2") or {}, b.get("5") or {}

            def bcell(r):
                if not r:
                    return "–"
                if short(r):
                    return '<span class="muted">refused: short sweep</span>'
                if not fin(r.get("idle_offset_shift")):
                    return f'<span class="muted">no arc ({r.get("tracked_columns")}/{r.get("flux_columns")} columns)</span>'
                return f'{num(r["idle_offset_shift"] * 1e3, "{:+.2f}")} ± {num(r["idle_offset_shift_error"] * 1e3, "{:.2f}")}'

            both = all(r and not short(r) and fin(r.get("idle_offset_shift")) for r in (b2, b5))
            if both:
                d = (b2["idle_offset_shift"] - b5["idle_offset_shift"]) * 1e3
                sig = d / (((b2["idle_offset_shift_error"] ** 2 + b5["idle_offset_shift_error"] ** 2) ** 0.5) * 1e3)
                wbd.append(d); wbs.append(sig)
                dcell = f'{num(d, "{:+.2f}")} ({num(abs(sig), "{:.1f}")}σ)'
            else:
                dcell = "–"
            w3b_rows.append([be, f"<b>{q}</b>", bcell(b2), bcell(b5), dcell,
                             f'{num((b2.get("column_precision") or float("nan")) / 1e6, "{:.2f}")} / {num((b5.get("column_precision") or float("nan")) / 1e6, "{:.2f}")}',
                             f'{num(b2.get("qpu_s"), "{:.1f}")} / {num(b5.get("qpu_s"), "{:.1f}")} s'])
    w_rms = (sum(d * d for d in wd) / len(wd)) ** 0.5
    w_h = sum(wh) / len(wh)
    w_ud2 = sum(ud2) / len(ud2)
    w_ud5 = sum(ud5) / len(ud5)
    w_fine = [((ws[f"{k}/03a"]["5"]["f_01"]) - f) / 1e6 for k, f in fine_abs.items()
              if (ws.get(f"{k}/03a") or {}).get("5", {}).get("line_identity") == "0-1"]
    w_fine_mean = sum(w_fine) / len(w_fine)

    # --- T1 missing from the state ---------------------------------------------------------------
    nt = json.loads((DATA / "noT1/summary.json").read_text())
    nt_runs = [json.loads(l) for f in sorted((DATA / "noT1").glob("*/runs.jsonl")) for l in f.read_text().splitlines()]
    nt_ok = [r for r in nt_runs if not r.get("error")]
    nt_qpu = {v: sum(r["qpu_s"] for r in nt_ok if r["variant"] == v) for v in ("noT1", "T1")}
    nt_failed = len(nt_runs) - len(nt_ok)
    T1n = dict(T1, qA6=5.25)
    NBE = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}
    hs_ref = {k: (v.get("reference") or {}).get("offset") for k, v in hs.items()}
    nt3a_rows, nt3b_rows, nt_d, nt_sig = [], [], [], []
    for be, qs in NBE.items():
        for q in qs:
            a = nt.get(f"{be}/{q}/03a") or {}
            u, k = a.get("noT1") or {}, a.get("T1") or {}

            def ac(r):
                if not r:
                    return "–"
                if r.get("line_identity") != "0-1":
                    why = "short sweep" if short(r) else ("no line" if r.get("line_identity") == "none" else esc(str(r.get("line_identity"))))
                    return f'<span class="muted">refused: {why}</span>'
                return f'{num(r["frequency_shift"] / 1e6, "{:+.2f}")} ± {num(r["f_01_error"] / 1e6, "{:.2f}")}'

            ref = hs_ref.get(f"{be}/{q}")
            if ref is None and f"{be}/{q}" in fine_abs:
                any_r = u if fin(u.get("f_01")) else k
                if fin(any_r.get("f_01")):
                    ref = fine_abs[f"{be}/{q}"] - (any_r["f_01"] - any_r["frequency_shift"])
            hr = (u["box_height"] / k["box_height"]) if fin(u.get("box_height")) and fin(k.get("box_height")) else None
            setting = (f'{num((u.get("sweep_ns") or 0) / 1e3, "{:.1f}")} / {num((u.get("wait_ns") or 0) / 1e3, "{:.0f}")} → '
                       f'{num((k.get("sweep_ns") or 0) / 1e3, "{:.1f}")} / {num((k.get("wait_ns") or 0) / 1e3, "{:.0f}")}')
            nt3a_rows.append([be, f"<b>{q}</b>", num(T1n[q], "{:.1f}"), setting, ac(u), ac(k),
                              num(ref / 1e6 if ref is not None else None, "{:+.2f}"), num(hr, "{:.2f}")])
            b = nt.get(f"{be}/{q}/03b") or {}
            bu, bk = b.get("noT1") or {}, b.get("T1") or {}
            both = all(r and not short(r) and fin(r.get("idle_offset_shift")) for r in (bu, bk))
            if both:
                d = (bu["idle_offset_shift"] - bk["idle_offset_shift"]) * 1e3
                sg = d / (((bu["idle_offset_shift_error"] ** 2 + bk["idle_offset_shift_error"] ** 2) ** 0.5) * 1e3)
                nt_d.append(d); nt_sig.append(sg)
                dcell = f'{num(d, "{:+.2f}")} ({num(abs(sg), "{:.1f}")}σ)'
            else:
                dcell = "–"
            nt3b_rows.append([be, f"<b>{q}</b>", bcell(bu), bcell(bk), dcell,
                              f'{num((bu.get("column_precision") or float("nan")) / 1e6, "{:.2f}")} / {num((bk.get("column_precision") or float("nan")) / 1e6, "{:.2f}")}',
                              f'{num(bu.get("qpu_s"), "{:.1f}")} / {num(bk.get("qpu_s"), "{:.1f}")} s'])

    CSS = base.CSS + """
figure img { display:block; }
ol.recs li, ul.tight li { margin:6px 0; max-width:82ch; }
.corr { border-left:3px solid var(--warn); background:var(--warn-bg); padding:10px 14px; border-radius:6px; margin:14px 0; max-width:88ch; }
.corr li { margin:5px 0; }
h2 .eyebrow { display:block; margin-bottom:4px; }
td .muted, .muted { color:var(--muted); }
.figrow { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
details { margin:10px 0; border:1px solid var(--line); border-radius:8px; padding:8px 12px; background:var(--surface); }
details summary { cursor:pointer; font-weight:600; }
h3 { margin-top:28px; }
"""
    body = f"""
<div class="page">
<p class="eyebrow">qua-agents benchmark · hardware test · 24 Sep 2026 · arbel, qolab, gilboa · nine qubits</p>
<h1>{TITLE}</h1>
<p class="lede">Two new calibration nodes replace the saturation pulse with a frequency chirp: <b>03a chirp</b> searches for the
qubit line over ±120 MHz at seven drive strengths, and <b>03b chirp</b> maps the line against flux with the flux step leading
the drive by 5 µs. On nine qubits of three IQCC chips both nodes ran inside the 60 s job cap, and across {n_rep:,} offline
replays over wrong frequencies, wrong drive calibrations and wrong idle points they gave <b>{n_wrong} wrong answers</b>. The test
also found where the chirp stops working (T1 below ~5 µs) and one way 03a could be fooled by a two-photon line; both are
now refused by the nodes rather than answered. Replayed the same way, the saturation node answered a fifth of the grid on the
long-T1 qubits and picked the 0→2 line in 4 % of it; it is the better tool only below T1 ≈ 5 µs. The fix to node 03b's pulse timing
was tested on the same day. A later change sweeps the frequency up and then down in every shot round; with it both nodes gave the
same answers at a 2 T1 reset wait as at the 5 T1 default, for half the QPU time (§5b).</p>

<div class="tiles">
  <div class="tile"><div class="v">{n_wrong} / {n_rep:,}</div><div class="k">replays that proposed a wrong line or sweet spot (the rest passed or refused)</div></div>
  <div class="tile"><div class="v">{max(all_qpu):.0f} s</div><div class="k">longest job's QPU time of {len(all_qpu)} node jobs; at their defaults 03a took {min(sum(r03a[b]) for b in QUBITS):.0f}–{max(sum(r03a[b]) for b in QUBITS):.0f} s and 03b {min(min(r03b[b]) for b in QUBITS):.0f}–{max(max(r03b[b]) for b in QUBITS):.0f} s per job for three qubits (qolab's 03b in two jobs)</div></div>
  <div class="tile"><div class="v">±{rms:.2f} MHz</div><div class="k">03a's f₀₁ against a fine saturation scan (rms of {len(diffs)} runs, worst {worst:.2f} MHz)</div></div>
  <div class="tile"><div class="v">T1 ≳ 5 µs</div><div class="k">where the chirp works at the default 20 MHz band; below it the nodes refuse and point to saturation</div></div>
</div>

<h2>1 · The two nodes</h2>
<p>Both live in the qua-libs fork on <span class="mono">{BRANCH}</span> (commit <span class="mono">{COMMITS['nodes']}</span>, off
<span class="mono">feat/qualibrate-ai</span> 65b4755), beside the saturation nodes, which are unchanged. The chirp pulse is added to the
generated config at run time only, so nothing new reaches the saved state.</p>
{table(["", "03a_qubit_spectroscopy_chirp", "03b_qubit_spectroscopy_vs_flux_chirp"], [
    ["pulse", "linear chirp over a 20 MHz band, raised-cosine edges; length the shorter of 4 µs and T1/5", "same pulse; drive amplitude from 03a or twice the x180 prediction"],
    ["scan", "band centres f₀₁ ± 120 MHz in 5 MHz steps × 7 drive levels a factor 2 apart, centred on the amplitude the stored x180 predicts (or 0.005–0.64 of full scale with <span class='mono'>drive_prior='none'</span>)",
     "21 flux offsets over ±1.5× the offset that lowers f₀₁ by 20 MHz (from the stored curvature) × band centres f₀₁ −40…+30 MHz in 2.5 MHz steps; the flux step starts 5 µs before the drive and outlasts it"],
    ["order", f"each shot round sweeps the band centre up and then down at every drive level; the fit averages the two (since <span class='mono'>{COMMITS['updown']}</span>, §5b)",
     "frequency inside flux, swept up and then down at every flux offset (same commit)"],
    ["analysis", "an erf-edged box fit per line and level; the 0→1 line is the one that appears at the lowest drive, a later line 40–250 MHz below it is its 0→2 partner (→ anharmonicity); the growth with drive gives the Rabi rate (<span class='mono'>drive_scale</span> against the x180)",
     "a box fit per column, a weighted parabola through the columns → sweet spot (relative to idle), f₀₁ there, curvature"],
    ["writes", "f₀₁ and RF frequency, only for a line identified as 0→1. The drive is reported, not written: <span class='mono'>selected_drive_amplitude</span> (for 03b), <span class='mono'>drive_scale</span> and an x180/x90 amplitude guess, which it writes only with <span class='mono'>update_pulses_amplitude=True</span> (default off)", "idle offset (joint or independent) and f₀₁, only when the turning point lies inside the sweep"],
    ["refuses", "no line; a lone line that grows faster than drive² or needs &gt;5× the predicted drive (a possible 0→2 line of a qubit above the window); sweep × band below 20 MHz·µs", "fewer than 7 columns with a line; turning point outside the sweep; sweep × band below 20 MHz·µs"],
    ["fallback", "the saturation node 03a", "<span class='mono'>pulse='saturation'</span>: a 1 MHz-Rabi drive for 3 T1 (20–100 µs) inside the same flux step, a Lorentzian per column"],
    ["60 s cap", "pre-flight estimate before submission; qubits run one after another", "same; qubits never pulse flux together"]])}

<h2>2 · Live runs at the node defaults</h2>
<p>Each backend ran 03a then 03b at their defaults at the start and again at the end of its session (3–12 min later),
plus two independent references: a fine saturation scan (±5 MHz at 0.1 MHz, 0.3 MHz Rabi)
around the chirp's f₀₁ and 03b in saturation mode. Everything ran on local copies of the 22 Sep snapshots, in propose mode:
nothing was written to any state. The table shows the answers of the committed analysis; §5 lists where it differs from what
the nodes answered live.</p>
{table(["backend", "qubit", "T1 [µs]", "sweep [ns]", "03a f₀₁ − stored, start [MHz]", "end", "fine scan − stored [MHz]",
        "drive scale", "α [MHz]", "03b sweet spot, start [mV]", "end", "saturation map [mV]"], rows)}
<p class="small muted">"no line": the saturation node's own fit rejected the scan (qolab Q1: r² 0.46 on a 25 kHz-wide spike; gilboa qD2:
the scan was centred on 03a's refused answer). α appears where the window held the 0→2 line (α/2 &lt; 120 MHz); on qolab it lies 148–150 MHz below f₀₁ and the
wide map (§3) gives α = 297, 300 and 297 MHz for Q1, Q2, Q5 — the stored 215.5 MHz is a placeholder. Drive scale is the measured
Rabi rate over the one the stored x180 predicts. 03a's f₀₁ differs from the fine scan by {mean:+.2f} MHz on average
(rms {rms:.2f}, worst {worst:.2f}), consistent with a 5 MHz step; it is not the sweep direction (§5b). Fine
spectroscopy after the flux map removes it.</p>

<p>Every qubit follows, chirp against saturation with the qubits as columns: first the 1D spectroscopy (the chirp's drive
ladder, then the saturation node's, each as a map and as line cuts), then the flux maps (chirp, then saturation). All are redrawn
from the saved datasets with the committed analysis; grey marks what a node does not propose. The figures as the nodes draw
them, the end-of-session runs, the wide flux maps and the fine scans are in the appendix.</p>
<p>What each map below cost on the chip:</p>
{table(["backend", "qubit", "T1 [µs]", "frequency vs drive, chirp (the map shown: 11 drives, ±200 MHz, 5 MHz steps, 100 shots)",
        "chirp at the node's defaults (7 drives, ±120 MHz)", "frequency vs drive, saturation (the map shown: ±130 MHz, 0.15 MHz steps, 300 shots)",
        "frequency vs flux, chirp (21 columns × 29 band centres, 100 shots)", "frequency vs flux, saturation (15 columns × 71 rows, 50 shots)"], fc_rows)}
<p class="small muted">QPU time. * a share of a job that measured several qubits, split by each qubit's time per shot; the saturation
drive maps were one job per drive and qubit. Every job adds compile and queue time, 5–90 s today: the saturation maps took 6–9 jobs
per qubit, the chirp maps one. The saturation flux map covers a smaller area than the chirp's (−25…+10 against −40…+30 MHz); over the
same area it would cost 3–5× the chirp map (§7).</p>
{backend_figures("arbel")}
{backend_figures("qolab")}
{backend_figures("gilboa")}

<h2>3 · How far off can the stored values be?</h2>
<p>Instead of re-running the nodes with scrambled states, each qubit was measured once over a wider range (03a: 11 drive levels
×1/32…×32 over ±200 MHz; 03b: 35 columns over ±2.5× the 20 MHz offset, f₀₁ −50…+40 MHz), and the node's own analysis was run
on the crop a scrambled node would have seen. A wrong stored f₀₁ only moves the window, and a wrong drive calibration only
changes which measured levels the node plays, so the crop is exactly what the node would measure. Outcomes: <b>pass</b> (the
right line within 1 MHz, or the sweet spot within 3σ and 0.5 mV — 1 mV on qolab), <b>refused</b> (nothing proposed),
<b>wrong</b> (a proposal outside those bounds).</p>
{img("capture_03a.png", "03a over a stored f₀₁ off by up to ±80 MHz and a drive between 1/16 and 16 times the x180 prediction. It passes everywhere from ×1/4 to ×2, mostly up to ×16 (refusing where the 0→2 partner falls outside a window shifted upward), and refuses from ×1/8 down, where only the no-prior ladder helps. gilboa qD2 and qC3 are refused throughout: their sweeps are too short (§5).", "Grid of 03a replay outcomes per qubit")}
{table(["backend", "qubit", "03a pass", "pass, drive ×1/4…×2", "03a wrong", "lone 0→2 line (4 windows)", "03b pass", "03b wrong", "α from the wide map [MHz]"], rep_rows)}
<p class="small muted">03b replays: the idle point off by up to ±1.3× the 20 MHz offset (±57 mV on arbel and gilboa, ±115–130 mV on qolab)
and the stored f₀₁ off by up to ±10 MHz. On the five qubits with an arc every replay passed; arbel qD1 has no arc in any map (§5),
gilboa qD2/qC3 are refused. "Lone 0→2 line": windows 130–160 MHz below f₀₁ that hold only the two-photon line.</p>
<h3>Why the refusals</h3>
<p>What is left grey in the grid has two causes:</p>
<ul class="tight">
<li><b>Drive ≤ 1/8 of the x180 prediction (bottom rows).</b> The 0→1 line appears only at the top levels and its 0→2 partner never
does, so it looks exactly like the 0→2 line of a qubit above the window seen at a correct drive — the case that fooled 03a on qB4
(§5). The lone-line check refuses both. Three more drive levels recover 40 of 66 on arbel and gilboa and 12–20 of 66 on qolab; a
confirmation scan α/2 above a refused lone line would settle every case with one extra job (§5).</li>
<li><b>T1 below ~5 µs (gilboa qD2, qC3).</b> The short-sweep guard (§5).</li>
</ul>
<p>A third region, at 4–16× the predicted drive with the stored f₀₁ too low, is gone since commit
<span class="mono">{COMMITS['flat']}</span>. There the line is at full height from the weakest level, so there is no rise to fit a
growth exponent to; fitted anyway it came out 7.9 on qolab Q2 (refused as a possible two-photon line) and undefined on Q1 (passed).
The exponent is now undefined without a rise, and the drive check decides: a 0→1 line saturated at the weakest level needs ~8× the
predicted drive, a lone 0→2 line ~64×. That took the replays from 1,485 to 1,617 passes of 2,079 on the seven qubits with long T1,
with no wrong answer and all 28 lone two-photon windows still refused.</p>

<h2>3b · The same replay for saturation</h2>
<p>The saturation node (03a, unchanged) was measured the same way: at nine drives from 1/16 to 16 times its default (half the stored
saturation amplitude) over ±130 MHz, at its own 0.15 MHz step and 300 shots, one qubit per job ({sat_jobs} jobs, {sat_qpu:.0f} s of
QPU). Its own analysis then ran on the crop each scrambled run would see, in its default ±50 MHz window, graded against the fine
saturation scans (qolab Q1, whose fine scan the node rejected, against the chirp; gilboa qD2, with no line found today, against the
23 Sep value). On arbel the stored saturation amplitude is already 1.0, so drives above ×2 are beyond full scale.</p>
{table(["qubits", "replays", "chirp 03a", "saturation 03a"], sat_rows)}
<ul class="tight">
<li><b>Saturation refuses most of the grid</b>, and not only where its narrower window misses the line. Driven hard, the line
broadens past the node's 15 MHz limit — arbel's default is already 21–28 MHz Rabi; on qolab and gilboa this happens from ×4.
Driven weakly, the line is narrower than three steps (the node asks for a finer rescan) or under its SNR threshold. On qolab Q1 it
stayed at SNR 3–6 at every drive, while the chirp, which waits 5×T1 between shots, saw full contrast on the same qubit; the
saturation node does not wait for the qubit to relax.</li>
<li><b>Its wrong answers are the 0→2 line.</b> {sat_wrong_long} of them on the long-T1 qubits, nearly all on arbel with the stored
f₀₁ ≥55 MHz low, where the ±50 MHz window holds the narrow 0→2 line but not the power-broadened 0→1 line: offsets of −99, −103 and
−107 MHz, α/2 — the night-4 failure, reproduced. The chirp gave none.</li>
<li><b>At short T1 saturation answers where the chirp does not</b>: on gilboa qC3 and qD2 it passes 58 % of the replays that have the
line in its window. qD2's 35 wrong answers come at ×8–×16 and are graded against the 23 Sep frequency.</li>
</ul>
{"".join(img(f"compare_capture_{be}.png", f"{be}: replay outcomes, chirp (left) and saturation (right). The drive axes differ: the chirp's is relative to the x180 prediction, saturation's to its own default, whose Rabi frequency is noted per qubit.", f"capture comparison {be}") for be in QUBITS)}
<p class="small muted">The raw responses behind the grids are the 1D figures of §2 (rows 1–2: chirp, rows 3–4: saturation).</p>
<p class="small muted">A first saturation ladder at 150 shots and 0.25 MHz steps ({old_jobs} jobs, {old_qpu:.0f} s of QPU) was replaced by this
one: at a 0.25 MHz step the node rejects lines narrower than 0.75 MHz as undersampled, and at half its shots many lines fell just
under its SNR threshold. Its fit results and logs are kept in <span class="mono">satladder/</span>.</p>

<h2>4 · Chirped against saturation maps</h2>
{img("sweetspots.png", "Sweet spot and apex frequency of the chirped map (start and end of the session) minus the saturation map taken between them, per qubit. arbel qB4/qA5 and gilboa qD5 agree to 0.25 mV or better; the apex frequencies within ±0.3 MHz.", "Chirp minus saturation sweet spots and apex frequencies")}
<p>The two methods agree on the sweet spot to 0.25 mV or better on arbel and gilboa. On qolab Q1 the chirped maps put it at
{num(pick(live,'qolab','Q1','r0-03b')['final']['x0']*1e3, '{:+.2f}')} and {num(pick(live,'qolab','Q1','end-03b')['final']['x0']*1e3, '{:+.2f}')} mV, the saturation map at
{num(pick(live,'qolab','Q1','sat-')['final']['x0']*1e3, '{:+.2f}')} mV, the wide chirped map at +0.34, the fixed 03b at −0.00 and 09a at +0.18…+0.30 mV:
Q1's sweet spot moves by ±0.3 mV between maps within the day, more than any single map's error. With Q1's curvature
(−2.6 kHz/mV²) that is under 1 kHz of frequency.</p>
<p><b>Precision is not where the chirp wins here.</b> With a weak (1 MHz Rabi), long (3 T1) saturation drive placed inside the flux
step, a saturation column is located to 52–114 kHz against the chirp's 93–179 kHz, and the sweet-spot errors are comparable
(0.04–0.25 mV against 0.03–0.23 mV). The 2.5× advantage measured on 23 Sep was against a saturation drive 2.4× stronger (0.5× the stored
amplitude, 2.4 MHz Rabi), whose line is wider. What the chirp adds is not needing that weak, calibrated
drive: its answer is the same from ¼ to 2× the drive, and it does not show the 0→2 line.</p>

<h2>5 · What the test changed, and what it found</h2>
<div class="corr"><ul class="tight">
<li><b>Column errors were 3× too large.</b> The box fit floored each centre's error at step/√24 whenever its fitted edge was narrower than
half a step; edges were ~1.1 MHz at 2.5 MHz steps, and the parabola through the columns scattered with χ²/dof = 0.10 on the 23 Sep
Q1 map. The floor now applies only below a quarter step (χ²/dof 0.3–3 across today's maps). Found on the qolab smoke run, fixed before
the full test.</li>
<li><b>Short sweeps give wrong answers.</b> gilboa qD2 and qC3 (T1 1.3 and 1.8 µs → 268 and 360 ns sweeps) gave peaked and split responses
instead of boxes; 03a found no 0→1 line on qD2, and 03b tracked a spurious line on qD2 and proposed a sweet spot with f₀₁ 25 MHz above the stored value,
~30 MHz above the line found there on 23 Sep. A passage has to be adiabatic (Rabi above ~√rate/π) and start far from the line (Rabi well below half the band), which needs
sweep × band ≳ 20 MHz·µs; with the T1/5 sweep and a 20 MHz band that is T1 ≳ 5 µs. Both nodes now refuse below it and point to
saturation — the ~5 µs fallback threshold proposed before the test, not the ~1 µs read from the 23 Sep qD2 test. The saturation map worked on
qC3 (+0.35 ± 0.20 mV). On qD2 the chirped map still shows the real arc ~5 MHz below the stored f₀₁ (where 23 Sep found the line),
but as a narrow ridge, not a box; the fit took a spurious arc ~30 MHz above it, where the drive switches on 20–30 MHz from the
line at ~16 MHz Rabi and excites it directly. The saturation map found no line on qD2.</li>
<li><b>A lone two-photon line fooled 03a once.</b> In the replay windows that hold only the 0→2 line, 03a refused on the six other qubits with a usable
chirp but took arbel qB4's 0→2 line as 0→1: its growth exponent came out 2.4 from two points in the rise. The growth rate separates them
cleanly: real 0→1 lines read 0.93–1.19 of the x180-predicted Rabi rate, lone 0→2 lines 0.001–0.05. A lone line needing more than 5×
the predicted drive is now refused. The price: with a drive 8× weaker than the x180 says, 03a refuses instead of answering.</li>
<li><b>arbel qD1 is not at a sweet spot.</b> Its line moves ~3 MHz per mV at the stored idle point (+0.227 V), so 7 mV columns land 20–30 MHz
apart and no map — chirped, saturation or wide — holds more than three columns; both nodes refused. Its stored curvature assumes a
sweet spot; a map around the real one needs a wider, finer flux scan.</li>
<li><b>The chirp reads f₀₁ ~0.2 MHz low</b> on average against the fine scan (§2), and the apex 0–0.3 MHz off the saturation map. The
suspect was the box top tilting with T1 decay during an up-sweep, but the up-and-down sweeps added since (§5b) read the same offset, so
the sweep direction is not the cause.</li>
</ul></div>

<h3>Short T1: a wider band and a hyperbolic-secant pulse</h3>
<p>On gilboa qD2 and qC3 one job ({st1['qpu_s']:.0f} s of QPU) compared, at the node's sweep length (T1/5) and five drives, the node's
20 MHz linear sweep with an 80 MHz linear sweep (sweep × band 21–29 MHz·µs) and a hyperbolic-secant pulse whose frequency follows a
tanh over 40 MHz, written into the I/Q samples so the drive rises only while the frequency is still far from the line.</p>
{img("shortT1_variants.png", "Excited population against band centre at five drives. Dotted: the line from saturation (qC3) and from 23 Sep (qD2).", "short-T1 chirp variants on qD2 and qC3")}
<p>Both variants give real boxes — full height from ~11 MHz Rabi on qD2, where the node's pulse gives a narrow peak and, from 11 MHz
Rabi, a spurious response 25–30 MHz above the line — and the hyperbolic secant gives the cleanest: a flat box with no spurious line,
the 0→2 line small until ~22 MHz Rabi. But none is accurate: at useful drives the box centres read 1–3 MHz high (qD2 −3.8 ± 0.3 MHz
against −4.75; qC3 +2.1…+2.6 against −0.1), more at stronger drive, with the boxes lopsided toward higher band centres. That points
at the sweep direction; alternating up- and down-sweeps would test it. Below T1 ≈ 5 µs saturation stays the better tool: its fine
scan put qC3's line at the right place with a 0.6 MHz width.</p>

<h3>Hyperbolic secant against the node's pulse, with controls</h3>
<p>A second test ({hs_qpu:.0f} s of QPU in 26 jobs) put the hyperbolic secant next to the node's linear chirp on five qubits —
gilboa qD2 and qC3 (T1 1.3, 1.8 µs), arbel qA6 (5.2 µs, the only one between 2 and 10 µs), and arbel qB4 and gilboa qD5 as long-T1
controls — each pulse swept up and down at five drives, with a fine saturation scan of the same session as the reference, and
flux maps with the linear chirp (at the node's 03b drive), the hyperbolic secant (at its full-box drive) and saturation.</p>
{table(["qubit", "T1 [µs]", "reference line − stored f₀₁", "linear chirp: box centre − reference, up/down mean", "hyperbolic secant: same",
        "sweet spot, linear map [mV]", "hyperbolic-secant map", "saturation map"], hs_rows)}
{img("hs_ladder.png", "Population against band centre at five drives: linear chirp up and down (left pair), hyperbolic secant up and down (right pair). Red dotted: the reference line; orange dotted: the 0→2 line.", "ladders of both pulses, both directions")}
{img("hs_maps.png", "Flux maps of the same qubits: linear chirp, hyperbolic secant, saturation.", "flux maps with three pulses")}
<ul class="tight">
<li><b>Long T1: no reason to switch.</b> On qB4 and qD5 both pulses place the line within 0.1–0.3 MHz of the reference. The
hyperbolic secant's edges are not sharper (0.8–0.9 MHz, against 0.4–1.6 MHz for the linear chirp), but its box fills only at ~3 MHz
Rabi against ~1 MHz, while the 0→2 line appears from ~11 MHz (or not at all up to 11 MHz) against ~8 MHz: less room between the two. Its flux maps are 3–4× less
precise (±0.23 against ±0.06 mV) with poorer fits, and agree with the linear and saturation maps within that error.</li>
<li><b>T1 = 5.2 µs (qA6): both work</b> within 0.2 MHz at moderate drive; the linear chirp breaks at 16 MHz Rabi, the hyperbolic secant
only drifts to −0.9 MHz at 22 MHz. qA6 showed no arc in any map (1 of 15 columns with every pulse): like arbel qD1, its stored idle
point is probably not a sweet spot.</li>
<li><b>Sweep direction explains most of qD2's high reading.</b> On qD2 the hyperbolic secant swept up reads +9 to +10 MHz at 22–28 MHz
Rabi, but the mean of up and down reads +3; at the drive where the box fills (11 MHz) the mean is +0.8 MHz. The same averaging removes a
smaller effect on long-T1 qubits (qB4's linear chirp at weak drive: up −0.6, mean −0.04 MHz). The nodes now alternate
directions, at the same total number of shots (§5b).</li>
<li><b>qD2: the hyperbolic secant is the only pulse that mapped it.</b> Its map found the arc in 12 of 15 columns, sweet spot
+3.41 ± 0.09 mV with f₀₁ 0.6 MHz above the reference; the linear chirp tracked the spurious arc again (f₀₁ +24.5 MHz) and saturation
found no arc. There is no independent sweet spot to check it against.</li>
<li><b>qC3 is not solved.</b> Both pulses, in both directions, put the box 0.5–3.9 MHz above a clean saturation reference (0.7 MHz wide)
at every drive — with the hyperbolic secant the offset shrinks as the drive grows, the opposite of a Stark shift — and the
hyperbolic-secant map's sweet spot
(+1.91 ± 0.31 mV) disagrees with the saturation map's (−0.54 ± 0.08 mV) by 2.4 mV. The cause is not known.</li>
</ul>
<p>So: the linear chirp stays for T1 above ~5 µs, where it is at least as good and gives better maps. Below that the hyperbolic secant
with up- and down-sweeps averaged is the only chirp that works at all, on one of the two qubits tried; saturation remains the
fallback until it is tried on more.</p>

<h3>Telling the 0→1 line from the 0→2 line</h3>
<p>03a decides in three steps. When both lines are in the window, the one that appears at the lowest drive is 0→1 and a later one
40–250 MHz below it is its 0→2 partner — robust, and it decided every such case. For a lone line it falls back on how the line grows
with drive: the growth exponent (fragile: two or three points in the rise with ×2 steps) and the Rabi rate against the x180 (a wide
margin, 0.93–1.19 for real lines against 0.001–0.05 for lone 0→2 lines, but only as good as the x180). Two better tests:</p>
<ul class="tight">
<li><b>Look for the partner.</b> For a lone line, scan 40–250 MHz above it: a line there that appears at a lower drive means the first
was the 0→2 line. One extra job, only when needed, no calibration involved. A window reaching further below f₀₁ (−200 MHz) also makes
the partner visible more often.</li>
<li><b>Where the box lands in the IQ plane.</b> A chirp through the 0→2 line leaves |2⟩, which the readout places elsewhere than |1⟩.
On all seven qubits where both lines appear, the 0→2 box points 6–24° away from the 0→1 box, always on the same side, while the 0→1
direction holds to 2–6° across drives. Against an x180 reference this labels a lone line in one measurement — clear on arbel and
gilboa (13–24°), marginal on qolab Q2 and Q5 (6°).</li>
</ul>

<h2>5b · Up-and-down sweeps, and a 2 T1 reset wait</h2>
<p>A chirp inside its box swaps |0⟩ and |1⟩, so excitation the reset wait has not removed is carried into the next shot and flipped
back. In a sweep that always runs upward it lands on the next points up the scan: a tail above every box that pulls its centre up.
In 03b, which stepped flux inside frequency, it landed on the next flux column and shifted the arc sideways — and with it the sweet
spot. At the 5 T1 default the effect is negligible, but it grows fast when the wait is short, as it is when the stored T1 is too low
or unknown (QuAM then waits 5 × 10 µs). Since commit <span class="mono">{COMMITS['updown']}</span> both nodes sweep the frequency up
and then down in every shot round and average the two, and 03b sweeps frequency inside flux, so what is left lands in the same
column, on both sides of the line. <span class="mono">num_shots</span> rounds up to an even number, so the QPU time is unchanged; the
raw dataset keeps both directions.</p>
<p>Simulated with the nodes' own fits, carrying each shot's excited population into the next (40 random line positions; with no
carry-over at all the 03a fit itself reads +0.1 ± 0.1 MHz):</p>
{table(["", "5 T1 wait", "2 T1", "1 T1", "0.5 T1"], [
    ["03a line centre [MHz], box height — upward sweeps (before)", "+0.12, 1.00", "+0.32, 0.90", "+0.73, 0.74", "+1.08, 0.59; 8 of 40 fits fail"],
    ["03a — up and down (now)", "+0.15, 1.00", "+0.01, 0.90", "+0.03, 0.74", "+0.06, 0.59"],
    ["03b sweet-spot error [flux steps] (scatter) — flux inside frequency (before)", "0.002 (0.005)", "0.012 (0.027)", "0.05 (0.16)", "0.33 (0.13)"],
    ["03b — frequency inside flux, up and down (now)", "0.002 (0.005)", "0.003 (0.006)", "0.001 (0.012)", "0.003 (0.023)"]])}
<p class="small muted">Frequency inside flux alone already removes the sweet-spot bias (the carry-over then shifts every column by the
same frequency); alternating the direction also removes the apex-frequency bias that leaves (+0.23 MHz at 1 T1) and cuts the
sweet-spot scatter 2–5× below 1 T1. Shuffled orders work too, but one fixed shuffle can build a ghost box the node takes for the line.</p>

<h3>2 T1 against 5 T1 on hardware</h3>
<p>The same nine qubits ran 03a and 03b chirp at their defaults with the new sweeps, once with every qubit's thermalization factor
at 2 (a 2 T1 wait between shots) and once at 5 (the default), back to back: {len(w_runs)} jobs, {w_qpu[2] + w_qpu[5]:.0f} s of QPU,
20:45–20:48, on the same local state copies in propose mode.</p>
{table(["backend", "qubit", "T1 [µs]", "03a f₀₁ − stored, 2 T1 [MHz]", "5 T1", "2 T1 − 5 T1", "box height 2 T1 / 5 T1",
        "up − down, 2 T1 [MHz]", "up − down, 5 T1"], w3a_rows)}
{table(["backend", "qubit", "03b sweet spot, 2 T1 [mV]", "5 T1", "2 T1 − 5 T1", "column error 2 T1 / 5 T1 [MHz]", "QPU of the job, 2 T1 / 5 T1"], w3b_rows)}
<p class="small muted">"up − down": the 0→1 line fitted on the up sweeps alone minus the down sweeps alone. QPU is the job's, shared when
it held several qubits (qolab's 03b ran Q2 alone and Q1 with Q5).</p>
<ul class="tight">
<li><b>2 T1 works on every long-T1 qubit.</b> In 03a all seven identify the 0→1 line at both waits with the same drive scale, and f₀₁
moves by {w_rms:.2f} MHz rms between the waits — the chirp's usual spread against a fine scan (§2). In 03b the six sweet spots agree
within {max(abs(x) for x in wbs):.1f}σ ({min(wbd):+.2f}…{max(wbd):+.2f} mV) with similar errors; qolab Q5's columns are noisier at 2 T1
but its sweet-spot error is unchanged. arbel qD1 found no arc and gilboa qD2 and qC3 were refused at both waits, as at noon.</li>
<li><b>Half the QPU time.</b> 03a took {w_node_qpu[("03a", 2)]:.0f} s against {w_node_qpu[("03a", 5)]:.0f} s, 03b
{w_node_qpu[("03b", 2)]:.0f} s against {w_node_qpu[("03b", 5)]:.0f} s: {100 * w_qpu[2] / w_qpu[5]:.0f} % in all.</li>
<li><b>The signal drops as predicted.</b> The box at 2 T1 is {w_h:.2f} of its 5 T1 height on average, the 1/(1 + e⁻²) = 0.88 that the
carried-over excitation predicts.</li>
<li><b>The pull itself was not resolved.</b> Fitted separately, up and down sweeps should differ by about +0.6 MHz at 2 T1; they
differ by {w_ud2:+.2f} MHz on average ({w_ud5:+.2f} at 5 T1), against ~0.3 MHz of noise per direction. The carry-over is smaller
than simulated, or hidden in that noise; at 2 T1 the alternation guards against a wrong or unknown T1 rather than correcting something
seen.</li>
<li><b>Not the −0.2 MHz offset of §2.</b> Against the noon fine scans the new 03a reads {w_fine_mean:+.2f} MHz on average ({len(w_fine)}
qubits, 5 T1), as the upward-only runs did ({mean:+.2f}): the sweep direction does not cause it. The fine scans are 8.5 hours older.</li>
<li><b>Drift, not the new loop order.</b> gilboa qD5's and arbel qB4's sweet spots sit 0.3 and 0.2 mV below the noon maps at both
waits; the 19:30 maps of §5, taken with the old order, already had them there (qD5 +2.67 chirp / +2.73 saturation, qB4 +0.05 / −0.09 mV).</li>
<li><b>The default stays at 5 T1.</b> The wait is each qubit's <span class="mono">thermalization_time_factor</span> in the state, which
every node uses; a 2 T1 default for the chirp nodes alone needs a node parameter.</li>
</ul>
{img("figures/waits_arbel.png", "arbel: 03a chirp at a 2 T1 wait (row 1) and at 5 T1 (row 2), 03b chirp at 2 T1 (row 3) and 5 T1 (row 4), up and down sweeps averaged. Red: the proposed f₀₁, column centres, parabola and sweet spot; grey: not proposed.", "arbel at 2 T1 and 5 T1")}
{img("figures/waits_qolab.png", "qolab, as above.", "qolab at 2 T1 and 5 T1")}
{img("figures/waits_gilboa.png", "gilboa, as above: qD2 and qC3 (T1 1.3 and 1.8 µs) are refused at both waits for their short sweeps.", "gilboa at 2 T1 and 5 T1")}

<h2>5c · With T1 missing from the state</h2>
<p>Without T1 both nodes still run, on fixed fallbacks: a 4 µs sweep, QuAM's 50 µs wait between shots (5 × 10 µs) and, in 03b, a
1 µs flux tail. They do not say that T1 is missing. On 25 Sep every qubit measured the day before ran 03a and 03b chirp at their
defaults (commit <span class="mono">{COMMITS['updown']}</span>) twice, back to back: on a state copy with T1 removed for those qubits,
and on the unchanged copy. {len(nt_ok)} jobs, {nt_qpu["noT1"] + nt_qpu["T1"]:.0f} s of QPU; {nt_failed} jobs that failed within one
minute on cloud errors (503 Service Unavailable, read timeouts on all three backends) were rerun three minutes later. arbel qA6's 03a
scanned −40…+120 MHz in both: its drive IF (−342 MHz) cannot take ±120 MHz.</p>
{table(["backend", "qubit", "stored T1 [µs]", "sweep / wait [µs], T1 unknown → known", "03a f₀₁ − stored, T1 unknown [MHz]", "T1 known",
        "fine scan − stored", "box height unknown / known"], nt3a_rows)}
{table(["backend", "qubit", "03b sweet spot, T1 unknown [mV]", "T1 known", "difference", "column error unknown / known [MHz]",
        "QPU of the job, unknown / known"], nt3b_rows)}
<p class="small muted">Fine scan: the saturation node at 0.3 MHz Rabi, ±5 MHz — 24 Sep 19:30 for qA6, qB4, qD2, qC3 and qD5 (the
hyperbolic-secant test), 24 Sep noon for the others; qolab Q1's had no clean line.</p>
<ul class="tight">
<li><b>Long T1: the same answers for {100 * nt_qpu["noT1"] / nt_qpu["T1"]:.0f} % of the QPU.</b> On six of the seven qubits with T1 ≥ 11 µs 03a
found the same line (within 0.1–0.3 MHz of the run with T1), with boxes 0.6–1.0 as high; qolab Q1 was refused (no line): its map, and
its 03b map, came out much noisier than Q2's and Q5's in the same jobs, for a reason not found — refused, not wrong. In 03b the six sweet spots agree within
{max(abs(x) for x in nt_sig):.1f}σ; qolab's maps are 2–2.5× less precise, from ~1 T1 waits and 3–6× less QPU.</li>
<li><b>Short T1 is where it goes wrong: the node answers, and reads low.</b> gilboa qD2 (T1 1.3 µs): 03a proposed its line at
−9.94 ± 1.02 MHz against −4.77 from the fine scan, and 03b a sweet spot of +2.41 ± 0.73 mV (the hyperbolic-secant map gave
+3.41 ± 0.09) with f₀₁ 4 MHz low. arbel qA6 (5.2 µs): 03a proposed −3.98 ± 0.59 MHz against −2.47. With T1 in the state both were
refused (qD2: sweep too short; qA6: ambiguous growth). The chirp always sweeps up, and an excitation survives to the readout only
when the crossing comes near the end of the sweep — when the line sits near the top of the band, at band centres below it — so the
box is weighted to its low side and reads low, by up to half the band. The alternation of §5b does not help: it reverses the order of
the band centres, not the chirp.</li>
<li><b>gilboa qC3's stored T1 is probably wrong.</b> Without T1, its 4 µs sweeps gave a flat box at the fine scan's frequency
(−0.09 ± 0.21 against +0.01 MHz) and a clean map (−0.46 ± 0.07 mV, 17 of 21 columns, against −0.54 ± 0.08 from saturation on 24 Sep),
exactly like a long-T1 qubit; with the stored 1.8 µs the nodes played 360 ns sweeps and refused. The unexplained qC3 readings of §5
came from sweeps sized for that T1.</li>
<li><b>What would make a missing T1 safe</b> (not done): alternate the chirp direction as well as the order, so the low reading of
up-chirps and the high reading expected of down-chirps cancel, and their difference flags a sweep too long for the qubit's T1; or read the
box a second time after a few µs of delay, which measures T1 directly; and have the nodes warn when T1 is missing.</li>
</ul>
{img("figures/noT1_arbel.png", "arbel: 03a without T1 (row 1) and with it (row 2), 03b without (row 3) and with (row 4). qD1 and qA6 are not at a sweet spot (no arc either way).", "arbel with and without T1")}
{img("figures/noT1_qolab.png", "qolab, as above.", "qolab with and without T1")}
{img("figures/noT1_gilboa.png", "gilboa, as above. qD2 without T1: a weak box 5 MHz below the line, proposed; with T1: refused. qC3 without T1: a clean box and arc.", "gilboa with and without T1")}

<h2>6 · Node 03b's pulse timing, and 09a</h2>
<p>03b passed nanoseconds to <span class="mono">play(duration=…)</span>, which counts 4 ns clock cycles, so its 20 µs saturation pulse and the
flux step under it ran for 80 µs. The fork fixed that on 19 Sep and reverted it on 20 Sep because two gilboa maps came out in absolute
flux; the real cause, found an hour later, was gilboa's state marking only qC2 active (every other flux line parked at 0 V). Now
(commit <span class="mono">{COMMITS['fix']}</span>): <span class="mono">// 4</span> restored, the drive length set to 80 µs explicitly
(today's timing, and steady state for T1 of tens of µs), and the flux step started 5 µs before the drive and held 5 µs after it.
03c needed no change: the fork's 03c already divides by 4 and leads the drive by 5 µs.</p>
{img("fix_03b.png", "A/B: qolab Q1 at the node's default drive (0.1× the stored saturation amplitude), ±94 mV. Both show the arc; the unfixed node adds a faint line at the idle frequency across the middle columns, which the margins remove. C/D: gilboa qD5 with the fix at 20 and 80 µs, C/D-active state — the comparison the revert asked for.", "03b before and after the timing fix, and gilboa qD5 at 20 and 80 µs")}
{table(["run", "sweet spot − idle", "f₀₁ at the apex − stored", "r²", "QPU"], [
    ["qolab Q1, 03b as on the branch (80 µs, no margins)", f'{num(b_main.get("idle_offset_shift", 0) * 1e3, "{:+.2f} mV")}', f'{num(b_main.get("frequency_shift", 0) / 1e6, "{:+.1f} MHz")}', num(b_main.get("r_squared"), "{:.3f}"), num(fx["map03b-Q1-main"]["qpu_s"], "{:.0f} s")],
    ["qolab Q1, fixed 03b", f'{num(b_fix.get("idle_offset_shift", 0) * 1e3, "{:+.2f} mV")}', f'{num(b_fix.get("frequency_shift", 0) / 1e6, "{:+.1f} MHz")}', num(b_fix.get("r_squared"), "{:.3f}"), num(fx["map03b-Q1-fix"]["qpu_s"], "{:.0f} s")],
    ["gilboa qD5, fixed 03b, 20 µs drive", f'{num(ab20.get("idle_offset_shift", 0) * 1e3, "{:+.2f} mV")}', "–", num(ab20.get("r_squared"), "{:.3f}"), num(fx["ab20-qD5-fix"]["qpu_s"], "{:.0f} s")],
    ["gilboa qD5, fixed 03b, 80 µs drive", f'{num(ab80.get("idle_offset_shift", 0) * 1e3, "{:+.2f} mV")}', "–", num(ab80.get("r_squared"), "{:.3f}"), num(fx["ab80-qD5-fix"]["qpu_s"], "{:.0f} s")],
    ["gilboa qD5, 03b chirp / saturation mode (§2)", f'{num(chirp_qD5.get("x0", 0) * 1e3, "{:+.2f}")} / {num(sat_qD5.get("x0", 0) * 1e3, "{:+.2f}")} mV', "–", "–", "–"]])}
<p>On gilboa qD5 the fixed node gives the same relative sweet spot at 20 and 80 µs (+3.13 and +3.31 mV, against +3.06 and +3.05 mV from
the chirped and saturation maps): the 18–30 mV shifts that prompted the revert do not appear with the correct active-qubit list. On
qolab Q1 the unfixed node also shows the arc at its default drive; the flat map of 23 Sep came from a drive 5× stronger. The fix
moves Q1's sweet spot by 0.44 mV and the apex frequency by 1 MHz, toward the chirped maps.</p>
<p><b>09a</b> (Ramsey vs flux) now waits <span class="mono">x90.length // 4</span> clock cycles on the flux line, upstream's fix of 24 Jul
(bce70ac), now commit <span class="mono">{COMMITS['fix09a']}</span>; before it the flux step started 144 ns late and ran through the second x90. On qolab Q1 the two
versions agree (idle 16–1000 ns, 11 flux points): sweet spot {num(r9m.get("flux_offset", 0) * 1e3, "{:+.2f}")} → {num(r9f.get("flux_offset", 0) * 1e3, "{:+.2f}")} mV, curvature
{num(r9m.get("freq_vs_flux_01_quad_term", 0) / 1e9, "{:.2f}")} → {num(r9f.get("freq_vs_flux_01_quad_term", 0) / 1e9, "{:.2f}")} kHz/mV²
(0.7 s of QPU each: the node reads the state instead of waiting for the qubit to relax). The misalignment matters for large flux offsets and short idle times, which this run did not probe.</p>

<h2>7 · Cost against the 60 s cap</h2>
{table(["backend", "node jobs", "QPU", "longest job", "wall (incl. queue)"], qpu_rows)}
<h3>Chirp against saturation: QPU time for the same scan</h3>
<p>Per qubit, from the jobs above and the saturation ladder at the node's defaults (0.15 MHz steps, 300 shots). A qubit's share of a
multi-qubit job is split by its time per shot.</p>
{table(["backend", "qubit", "T1 [µs]", "03a chirp: 7 drives, ±120 MHz", "saturation: 1 drive, ±50 MHz (its default)",
        "saturation: 1 drive, ±120 MHz", "saturation: 7 drives, ±120 MHz", "03b chirp map", "03b saturation, same area"], cmp_rows)}
<p>The two nodes spend their time differently. A chirp shot waits 5×T1 for the qubit to relax, so 03a costs from under 1 s (gilboa's
short-T1 qubits) to 14 s (qolab Q2, T1 80 µs), and active reset would cut it about tenfold once a readout threshold exists. The
saturation node never waits for T1 — 36 µs a shot whatever the qubit — but a line ~1 MHz wide needs 0.15 MHz steps: 1,600 points
for the chirp's ±120 MHz against the chirp's 49 band centres. So its default ±50 MHz scan at one drive costs about what the chirp's
±120 MHz, seven-drive scan does on long-T1 qubits, and the same coverage costs 8× (qolab Q2) to 45× (arbel qA5) more on
long-T1 qubits and ~220× on gilboa's short-T1 pair. For the flux map both wait 5×T1 per shot
and saturation adds a 20–100 µs drive and five times the frequency points: 3–5× more QPU for the same area (21 columns, −40…+30 MHz,
50 shots against the chirp's 100).</p>
<h3>Time to a frequency and a sweet spot</h3>
<p>The whole path per qubit: 03a for the frequency, then 03b for the sweet spot, at each node's defaults, with the stored f₀₁
right. The chirp side is the measured start-of-session runs. For saturation's 03a the replays of its drive ladder stand in for an
agent that halves the drive each time the node refuses, from its default down, each try charged a full pass (±50 MHz, 0.15 MHz
steps, 300 shots); its 03b is the measured saturation-mode map (15 columns, −25…+10 MHz, 50 shots — a smaller area than the chirp's
21 columns over −40…+30 MHz). The fine saturation scan that follows either path (1–2 s) is left out.</p>
{table(["backend", "qubit", "T1 [µs]", "chirp 03a", "chirp 03b (sweet spot)", "chirp total", "saturation 03a", "saturation 03b (sweet spot)",
        "saturation total"], tta_rows)}
<p>On the {tot["n"]} qubits with T1 ≥ 11 µs the chirp reached a frequency and a sweet spot in {tot["chirp"]:.0f} s of QPU against
{tot["sat"]:.0f} s for saturation, in two jobs per qubit against two to five, and answered wherever saturation did (neither found an
arc on arbel qD1). Saturation's extra cost is its 03a: at its default drive the node refused the line on five of those seven qubits
— too broad on arbel, where the default is 21–28 MHz Rabi, too noisy on qolab Q1, and on gilboa qD5 — and on arbel qB4 and qA5 the answer it
finally accepted was 1–5 MHz off. Where its default drive suits the qubit (qolab Q2 and Q5) saturation is as cheap or cheaper,
because its shots skip the 5×T1 wait. The sweet spots of the two methods are about equally precise. On gilboa's short-T1 pair only
saturation answers: both frequencies, and qC3's sweet spot in 12 s. Each job also costs compile and queue time, 5–90 s per job today.</p>
<p class="small muted">Plus {len(fx)} jobs for the 03b/09a fix tests ({fix_qpu:.0f} s of QPU). No job reported a timeout or partial result on stderr.
The pre-flight estimates ran 1.3–1.6× the measured QPU time, the safe side by design. Wide maps and reference scans are part of the test, not
of the nodes' default cost.</p>

<h2>8 · Open</h2>
<ol class="recs">
<li>Agents have not used the nodes yet; the recipes are written but untested in a campaign.</li>
<li>Short-T1 qubits need the saturation fallback in 03a (the old node) and 03b (<span class="mono">pulse='saturation'</span>). The
hyperbolic secant with up- and down-sweeps averaged worked on qD2 (the only map of it) but not on qC3, whose 1–3 MHz offset is
unexplained; it needs more short-T1 qubits before it goes into the nodes.</li>
<li>For a lone line, replace the growth fit with a partner scan 40–250 MHz above it, and use the box's IQ direction against an x180
reference as a second check; extend the window down to −200 MHz.</li>
<li>The −0.2 MHz frequency offset against the fine scans is not the sweep direction (§5b); a fine scan in the same session as
the chirp would show whether it is real.</li>
<li>A missing T1 is safe only for T1 ≳ 10 µs: shorter-lived qubits get a 4 µs sweep and a line read low, answered (§5c). Alternate
the chirp direction, estimate T1 from the box, and warn. gilboa qC3's stored T1 (1.8 µs) looks wrong.</li>
<li>A 2 T1 reset wait halves the QPU time of both nodes with the same answers (§5b); making it their default needs a node parameter,
since the wait is each qubit's thermalization factor, shared by every node.</li>
<li>arbel qD1's operating point: map it with a finer, wider flux scan.</li>
<li>qolab's stored anharmonicities (215.5 MHz) are placeholders; the chirp measured 297–300 MHz.</li>
<li>The node branch is not merged into <span class="mono">feat/qualibrate-ai</span>: the frozen arbel and gilboa cells read that
working tree and would pick up new nodes when resumed.</li>
</ol>

<h2>Appendix · all other figures</h2>
{appendix()}

<h2>Provenance</h2>
<p class="small">Scripts, data (netcdf datasets, fit results, figures, driver logs) and replay outputs are in
<span class="mono">2026-09-24-chirp-nodes/</span>: <span class="mono">run_backend.py</span> (the node runs),
<span class="mono">run_fixes.py</span> and <span class="mono">chain_fixes.sh</span> (the fix tests), <span class="mono">replay.py</span> and
<span class="mono">reanalyse.py</span> (offline, committed analysis), <span class="mono">plot_summary.py</span>,
<span class="mono">plot_fixes.py</span>, <span class="mono">plot_all.py</span> (every node figure, redrawn),
<span class="mono">plot_columns.py</span> (chirp against saturation, qubits as columns), <span class="mono">order_sim.py</span> and
<span class="mono">order_sim_03b.py</span> (shot-order simulations; outputs in <span class="mono">waits/</span>), <span class="mono">run_waits.py</span>,
<span class="mono">waits_analyse.py</span> and <span class="mono">plot_waits.py</span> (the 2 T1 test), <span class="mono">run_noT1.py</span>,
<span class="mono">run_noT1_retry.py</span>, <span class="mono">noT1_analyse.py</span> and <span class="mono">plot_noT1.py</span> (T1 missing, 25 Sep). States: local copies of <span class="mono">~/qab-runs/reference-state-20260922/</span> (gilboa:
the copy with the ten C/D qubits active). Each launch is logged in <span class="mono">~/qab-runs/recipe-qolab-LOG.md</span>. The
23 Sep investigation behind the nodes: <a href="2026-09-23-spectroscopy-chirp-vs-saturation.html">chirp vs saturation spectroscopy</a>.
Generated by <span class="mono">make_chirp_nodes_report.py</span>.</p>
</div>
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
</head><body>{body}</body></html>
"""


def main() -> int:
    if "--sync" in sys.argv:
        sync()
    for key in ("nodes", "fix"):
        flag = f"--{key}-commit"
        if flag in sys.argv:
            COMMITS[key] = sys.argv[sys.argv.index(flag) + 1]
    page = build()
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1e6:.1f} MB)")
    if "--artifact" in sys.argv:
        dest = Path(sys.argv[sys.argv.index("--artifact") + 1])
        body = page.split("<title>", 1)[1]
        dest.write_text("<title>" + body.replace("</head><body>", "").replace("</body></html>", ""))
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
