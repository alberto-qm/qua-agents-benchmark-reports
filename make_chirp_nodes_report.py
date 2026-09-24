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
COMMITS = {"nodes": "be998ef", "fix": "576d230", "fix09a": "e22fa9c"}


def sync():
    DATA.mkdir(exist_ok=True)
    for sub in ("arbel", "qolab", "gilboa", "fixes/qolab", "fixes/gilboa", "replay", "figures"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for f in (RUNS / sub).glob("*"):
            if f.is_file() and f.suffix in (".json", ".nc", ".png", ".log", ".jsonl", ".out"):
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
    """Start-of-session 03a ladder, 03b chirped map(s) and saturation map(s) for one backend."""
    qs = QUBITS[be]
    tag3a = f"r0-03a-{'-'.join(qs)}"
    out = [f'<h3>{be} · {", ".join(qs)}</h3>']
    out.append(img(f"figures/{be}_{tag3a}_ladder.png",
                   f"{be}, 03a chirp at its defaults, {qpu_of(be, tag3a):.0f} s of QPU for the three qubits. Rows: drive levels, weakest at "
                   "the bottom. Red solid: the 0→1 line proposed; red dashed: its 0→2 partner; grey dashed: a line not proposed.",
                   f"03a chirp ladders for {be}"))
    maps = sorted(p.name for p in (DATA / "figures").glob(f"{be}_r0-03b-*_flux_map.png"))
    for name in maps:
        tag = name[len(be) + 1:-len("_flux_map.png")]
        who = tag[len("r0-03b-"):].replace("-", ", ")
        single = "," not in who
        out.append(img(f"figures/{name}", f"{be} {who}, 03b chirp at its defaults, {qpu_of(be, tag):.0f} s of QPU. Circles: each column's "
                       "box centre; red curve and star: the fitted parabola and sweet spot (grey: not proposed).", f"03b chirped maps {be} {who}",
                       max_width=480 if single else None))
    sats = sorted(p.name for p in (DATA / "figures").glob(f"{be}_sat-*_flux_map.png"))
    items = []
    for name in sats:
        tag = name[len(be) + 1:-len("_flux_map.png")]
        who = tag[len("sat-"):].replace("-", ", ")
        items.append((f"figures/{name}", f"{be} {who}, 03b in saturation mode (reference), {qpu_of(be, tag):.0f} s", f"saturation map {be} {who}"))
    if len(items) == 1:
        out.append(img(*items[0], max_width=None if "," in items[0][1] else 480))
    elif items:
        out.append(figrow(items))
    return "\n".join(out)


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
now refused by the nodes rather than answered. The fix to node 03b's pulse timing was tested on the same day.</p>

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
(rms {rms:.2f}, worst {worst:.2f}), consistent with a 5 MHz step and a slightly tilted box top (T1 decay during the up-sweep); fine
spectroscopy after the flux map removes it.</p>

<p>Every qubit's figures follow, redrawn from the saved datasets with the committed analysis (grey marks what the node does
not propose). End-of-session runs, the wide maps and the fine scans are in the appendix.</p>
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
<h3>Why so many refusals, and would saturation do better?</h3>
<p>Three causes, one per grey region of the grid:</p>
<ul class="tight">
<li><b>Drive ≤ 1/8 of the x180 prediction (bottom rows).</b> The 0→1 line appears only at the top levels and its 0→2 partner never
does, so it looks exactly like the 0→2 line of a qubit above the window seen at a correct drive — the case that fooled 03a on qB4
(§5). The lone-line check refuses both. Before that check these replays passed; three more drive levels recover 40 of 66 on arbel and
gilboa and 12–20 of 66 on qolab; a confirmation scan α/2 above a refused lone line would settle every case with one extra job.</li>
<li><b>Drive ≥ 4× the prediction with the window shifted up (top-right patches).</b> The line is already full-height at the lowest
level, so its growth exponent cannot be measured, and the 0→2 partner has left the bottom of the window. Extending the window
down to −200 MHz (+33 % QPU) removed every such refusal where the wide maps allow the test (qolab Q2 14 → 0, Q5 6 → 0).</li>
<li><b>T1 below ~5 µs (gilboa qD2, qC3).</b> The short-sweep guard (§5).</li>
</ul>
<p><b>Saturation would answer in the first two cases</b> — at a weak drive it sees only the 0→1 line, so its answer would be right. But
it gives the same kind of answer when its window holds only a 0→2 line, and nothing in a single saturation trace says which case
it is in: that is how the night-4 arbel runs committed qB4's 0→2 line. The chirp refuses exactly where the data cannot tell the two
apart. In the third case saturation is simply better: its map found qC3's sweet spot, the chirp's did not.</p>

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
<li><b>The chirp reads f₀₁ ~0.2 MHz low</b> on average against the fine scan (§2), and the apex 0–0.3 MHz off the saturation map; likely the
box top tilting with T1 decay during an up-sweep. Alternating up and down sweeps would cancel it; not done.</li>
</ul></div>

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
<p class="small muted">Plus {len(fx)} jobs for the 03b/09a fix tests ({fix_qpu:.0f} s of QPU). No job reported a timeout or partial result on stderr.
The pre-flight estimates ran 1.3–1.6× the measured QPU time, the safe side by design. Wide maps and reference scans are part of the test, not
of the nodes' default cost.</p>

<h2>8 · Open</h2>
<ol class="recs">
<li>Agents have not used the nodes yet; the recipes are written but untested in a campaign.</li>
<li>Short-T1 qubits need the saturation fallback in 03a (the old node) and 03b (<span class="mono">pulse='saturation'</span>); a wider band
for short sweeps (sweep × band ≥ 20 MHz·µs) might extend the chirp to T1 ~2 µs, untested.</li>
<li>The −0.2 MHz frequency bias: alternate sweep directions.</li>
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
<span class="mono">plot_fixes.py</span>, <span class="mono">plot_all.py</span> (every node figure, redrawn). States: local copies of <span class="mono">~/qab-runs/reference-state-20260922/</span> (gilboa:
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
