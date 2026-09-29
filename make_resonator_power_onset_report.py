"""Build 2026-09-28-resonator-power-onset.html from 2026-09-28-resonator-power-onset/ (no QPU, no network).

Offline analysis, 28 Sep 2026, of every resonator-spectroscopy-vs-power (02b) map taken since 25 Sep: the n7 cells and the
25-26 Sep arbel/qolab maps, re-fitted with the node analysis frozen for n7 (~/qab-runs/qua-libs-n7). report_data.py (in the
data folder) draws one figure per map and writes results.json; cand3.py is the proposed onset rule, score.py the scoring.
Usage:
    python make_resonator_power_onset_report.py [--sync]
--sync copies the scripts, figures and results.json from ~/qab-runs/onset-n7 into 2026-09-28-resonator-power-onset/ first.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from html import escape as esc
from pathlib import Path

import make_fwcmp2_report as base

HERE = Path(__file__).parent
DATA = HERE / "2026-09-28-resonator-power-onset"
RUNS = Path.home() / "qab-runs/onset-n7"
OUT = HERE / "2026-09-28-resonator-power-onset.html"
TITLE = "Resonator power onset"
SCRIPTS = ("load.py", "cand.py", "cand2.py", "cand3.py", "score.py", "report_data.py")


def sync():
    (DATA / "figures").mkdir(parents=True, exist_ok=True)
    for f in SCRIPTS + ("results.json",):
        shutil.copy(RUNS / f, DATA / f)
    for f in (RUNS / "figures").glob("*.png"):
        shutil.copy(f, DATA / "figures" / f.name)


def img(path, alt):
    b64 = base64.b64encode((DATA / path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(alt)}" loading="lazy"></figure>'


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="grid small"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def ok(v):
    return isinstance(v, (int, float)) and v == v


def dbm(v, signed=True):
    return f'<span class="num">{v:+.1f}</span>' if ok(v) else '<span class="muted">none</span>'


def build() -> str:
    res = json.loads((DATA / "results.json").read_text())
    rows = res["rows"]; sc = res["scores"]
    by_label = {r["label"]: r for r in rows}

    # 1. the nine runs the eye picks were made on
    eye_rows = []
    for r in rows:
        if r["eye"] is None:
            continue
        cap = r["new"] if not ok(r["fs"]) or not ok(r["new"]) else min(r["new"], r["fs"] - 20)
        clip = f' → {cap:+.1f} at the ceiling' if ok(r["new"]) and cap < r["new"] - 0.05 else ""
        eye_rows.append([esc(r["label"][3:]), dbm(r["node"]), esc(r["node_mech"]), dbm(r["new"]) + clip,
                         dbm(r["new_onset"]), dbm(r["eye"]), dbm(r["iqcc"])])

    # 2. scores
    score_rows = []
    for name, s in sc.items():
        e = [abs(x) for x in s["eye"] if x is not None]
        above = [x for x in s["eye"] if x is not None]
        sp = [v[0] for v in s["spreads"].values() if v[0] is not None]
        st = s["stress"]
        stress = " · ".join(f"{k} {v[1]:.1f}" for k, v in st.items())
        worst = f"{min(above):+.1f} to {max(above):+.1f}" if above else "–"
        score_rows.append([f"<b>{esc(name)}</b>" if name.endswith("final") else esc(name),
                           f"{sum(e) / len(e):.1f} dB ({len(e)}/9)", worst,
                           f"{sorted(sp)[len(sp) // 2]:.1f} dB", f"{s['spreads']['qA5'][0]:.1f} dB",
                           stress, str(sum(v[2] for v in st.values()))])

    # 3. against IQCC's readout powers (n7 cells)
    iq_rows = []
    for r in rows:
        if r["iqcc"] is None or not r["label"].startswith("n7"):
            continue
        d_new = r["new"] - r["iqcc"] if ok(r["new"]) else None
        d_node = r["node"] - r["iqcc"] if ok(r["node"]) else None
        iq_rows.append((r["label"][3:], [esc(r["label"][3:]) + f' <span class="mono muted">{r["mid"]}</span>', dbm(r["iqcc"]), dbm(r["node"]), dbm(d_node), dbm(r["new"]), dbm(d_new)]))
    iq_rows = [x[1] for x in sorted(iq_rows)]
    def mean_d(be, key):
        v = [r[key] - r["iqcc"] for r in rows if r["label"].startswith(f"n7 {be}") and r["iqcc"] is not None and ok(r[key])]
        return (sum(v) / len(v), min(v), max(v), len(v)) if v else None

    # 4. figure sections
    def figs(sel):
        out = []
        for r in rows:
            if sel(r):
                out.append(img(r["figure"], r["label"]))
        return "\n".join(out)
    eye_figs = figs(lambda r: r["eye"] is not None)
    n7_other = {be: figs(lambda r, be=be: r["eye"] is None and r["label"].startswith(f"n7 {be}")) for be in ("qolab", "arbel", "gilboa")}
    older = figs(lambda r: not r["label"].startswith("n7"))

    all_rows = []
    for r in sorted(rows, key=lambda r: (not r["label"].startswith("n7"), r["label"])):
        all_rows.append([esc(r["label"]), f'<span class="mono">{r["mid"]}</span>', f'{r["window"][0]:.0f}…{r["window"][1]:.0f} / {r["window"][2]}',
                         dbm(r["node"]), esc(r["node_mech"]), dbm(r["new"]), esc(r["new_how"]),
                         f'{r["M_lw"]:+.2f}' if ok(r["M_lw"]) else "–", dbm(r["eye"]) if r["eye"] is not None else "",
                         dbm(r["iqcc"]) if r["iqcc"] is not None else ""])

    final = sc["photon model, final"]; node = sc["node"]
    fe = [abs(x) for x in final["eye"] if x is not None]; ne = [abs(x) for x in node["eye"] if x is not None]
    q = mean_d("qolab", "new"); a = mean_d("arbel", "new"); an = mean_d("arbel", "node")

    CSS = base.CSS + """
figure { margin:14px 0 22px; }
figure img { display:block; width:100%; height:auto; border-radius:6px; }
ul.tight li, ol.recs li { margin:6px 0; max-width:82ch; }
.muted { color:var(--muted); }
h3 { margin-top:28px; }
.legend-note { max-width:82ch; }
"""
    body = f"""
<div class="page">
<p class="eyebrow">qua-agents benchmark · offline analysis · 28 Sep 2026 · arbel, gilboa, qolab · {len(rows)} maps</p>
<h1>{TITLE}</h1>
<p class="lede">Node 02b (resonator spectroscopy vs power) proposes the readout power from where the resonator line starts to move.
It calls the onset when the line has moved 0.5 linewidths, and on the n7 run (28 Sep) that rule failed on nine maps. At 0 V on qolab
the whole move is 0.1–0.7 linewidths, so the node either proposed nothing or a power partway up the move. On arbel and gilboa the line drifts
slowly and steadily, and the onset fell to a late fade or to the sweep's ceiling. A rule that fits the drift as photon number,
c<sub>0</sub> + A·10<sup>P/10</sup>, and calls the onset where that drift reaches 0.05 linewidths, lands within
{sum(fe) / len(fe):.1f} dB of the power picked by eye on all nine (the node: {sum(ne) / len(ne):.1f} dB, five proposed). It
agrees across repeat maps of the same resonator to {sorted(v[0] for v in final['spreads'].values())[len(final['spreads']) // 2]:.1f} dB,
and it holds up under twice the noise, half the power rows and a shortened window. It is prototyped offline; qua-libs is unchanged.</p>

<h2>1 · The nine runs that failed</h2>
<p>Eye: the power picked by eye from the node's figure, the top of the flat part of the line. IQCC: the readout power in IQCC's
calibrated state for that qubit. The gilboa eye picks are the highest power writable at the port's current full scale (−22 dBm):
the node's proposals there need a higher full scale, which tinycal's allowlist does not let it write.</p>
{table(["run", "node", "node onset via", "new", "new onset", "eye", "IQCC"], eye_rows)}

<h2>2 · What the maps show</h2>
<ul class="tight">
<li><b>qolab at 0 V</b>: a clean S-curve from the low-power level to the bare cavity, but the whole move equals the dispersive shift, 0.1–0.7
linewidths with the qubit this far off resonance. A threshold in linewidths is the wrong scale there: Q2 moves 0.12 lw in total.</li>
<li><b>qolab at the sweet spot</b>: a slow drift, then the dip forks or fades and jumps to the bare cavity. The node does reasonably
well here but calls the onset on the steep part.</li>
<li><b>arbel</b>: noisy reference rows, a long slow drift, then a fork. The node's onset came from the late fade, and on qC3 the
tracker jumped to another feature at the top (the −8 lw points).</li>
<li><b>gilboa</b>: the drift starts around −20 dBm and the −10 dBm ceiling cuts it off before 0.5 lw.</li>
<li><b>No usable line</b> (arbel qB4, gilboa qC5 and qD4): the tracker is not on the resonator; any rule has to propose nothing here.</li>
</ul>

<h2>3 · The rule</h2>
<ol class="recs">
<li>Take the node's per-row Lorentzian centres and errors unchanged. Keep rows with a single dip at least half the plateau depth,
and drop jumps of more than half a linewidth from the running track (another feature, not the line moving). Stop at the node's own fade or fork onset.</li>
<li>Fit c(P) = c<sub>0</sub> + A·10<sup>P/10</sup> (a shift proportional to photon number) with inverse-variance weights, on the rows
below the point where the line has moved min(50 % of its largest move, 0.25 lw).</li>
<li>The onset is where the fitted drift A·10<sup>P/10</sup> reaches min(0.05 lw, 25 % of the move). The 25 % cap matters only for
the tiny 0 V moves.</li>
<li>Propose the top clean row below the onset, minus 1 dB (the node's buffer). The node's fade and fork onsets still apply when they come first.</li>
<li>Propose nothing when the rows near the onset cannot resolve the threshold (1σ of a 3-row mean above it). This is what
rejects qB4 on 26 Sep, where the tracker follows a feature 3 MHz from the resonator.</li>
</ol>
<p>Why a model and a physics-scaled threshold: a drift that grows as 10<sup>P/10</sup> has no sharp start, so any threshold set by a
map's own noise moves with that noise. The fit uses every row below the break, so single noisy rows barely move it.</p>

<h2>4 · Scores</h2>
<p>Eye: mean |proposal − eye| over the nine runs (gilboa clipped to −22 dBm), and the range of proposal − eye (negative: below the eye pick).
Repeat spread: max − min proposal over maps of the same resonator in the same flux condition (11 groups, full windows);
qA5 has seven maps over three days. Stress: 90th percentile |Δ| in dB after re-running the node's whole per-row analysis on a
changed map: noise2 adds noise to twice the row noise, dec2 keeps every second power row (1.6 dB steps), top5 cuts the top 5 dB, bot10 cuts the
bottom 10 dB; the last column counts changes over 3 dB across all four.</p>
{table(["rule", "eye error", "proposal − eye, range", "repeat spread (median)", "qA5 spread", "stress p90 [dB]", "stress > 3 dB"], score_rows)}
<ul class="tight">
<li><b>Noise threshold (A)</b>: the onset is where the smoothed shift passes 3σ of the reference rows. It gets close to the eye,
but the noisy low-power reference sets the threshold, so repeats disagree by 4 dB.</li>
<li><b>Slope walk-down (B)</b>: it starts from a clearly moved point and walks down while the local slope is significant. It follows
the slow drift 5–8 dB too low.</li>
<li><b>Photon model at 10 % of the move</b>: the most repeatable, but the move shrinks whenever the transition is cut short (a
coarser grid, a lower ceiling), and the onset slides down with it. Tying the threshold to the linewidth removes that.</li>
</ul>

<h2>5 · Against IQCC's readout powers</h2>
<p>The rule implements "the top of the flat part of the line". That is not how IQCC operates these qubits everywhere:</p>
<ul class="tight">
<li><b>arbel</b>: the new proposals are {a[0]:+.1f} dB from IQCC on average ({a[1]:+.1f} to {a[2]:+.1f}); the node's were {an[0]:+.1f} dB ({an[1]:+.1f} to {an[2]:+.1f}).</li>
<li><b>qolab</b>: {q[0]:+.1f} dB on average ({q[1]:+.1f} to {q[2]:+.1f}). At the sweet spot IQCC runs Q1 at −39 dBm, 12 dB into the drift that starts near −50.</li>
</ul>
<p>A lower readout power costs signal, so which of the two counts as right is a choice to make before the rule goes into the node. The node could
also report the whole band, from the drift start to its current onset, and let the agent choose.</p>
{table(["n7 cell", "IQCC", "node", "node − IQCC", "new", "new − IQCC"], iq_rows)}

<h2>6 · The nine runs, figure by figure</h2>
<p class="legend-note small">Left: |S21| normalised per power row, with the tracked line centres the rule uses (red). Right: the line centre's shift
from the low-power level, in linewidths (blue: used; grey: not used). Magenta curve: the drift fit, over its fit rows (shaded) and extrapolated 4 dB past them.
Dotted magenta line: the threshold. Diamond: the onset. Vertical and horizontal lines: node (black dashed), new (magenta), eye (green
dotted), IQCC (blue dash-dot). Orange: dip depth over its plateau.</p>
{eye_figs}

<h2>7 · The other n7 maps</h2>
<h3>qolab</h3>
{n7_other['qolab']}
<h3>arbel</h3>
{n7_other['arbel']}
<h3>gilboa</h3>
{n7_other['gilboa']}

<h2>8 · The 25–26 Sep maps</h2>
<p>The 25 Sep maps with a −50…−25 dBm window have less than 10 dB of reference below some onsets; they are kept to show how the rule
behaves on them. The −25…−10 dBm qD1 maps have no reference at all, and the rule proposes nothing there.</p>
{older}

<h2>9 · Every map</h2>
{table(["map", "node run", "window [dBm] / rows", "node", "node onset via", "new", "new onset via", "largest move [lw]", "eye", "IQCC"], all_rows)}

<h2>10 · Open</h2>
<ol class="recs">
<li>Which power to propose, given IQCC's operating points (section 5).</li>
<li>Porting the rule into the node's <span class="mono">_fit_single_qubit</span> (it only needs the per-row centres, errors and depths
already computed there), and whether the figure should show the drift fit.</li>
<li>gilboa's proposals (about −21 dBm) need <span class="mono">full_scale_power_dbm</span> raised, which is outside tinycal's allowlist.</li>
<li>Line detection on weak lines (arbel qB4, gilboa qC5 and qD4) is a separate problem: the tracker follows the wrong feature.</li>
</ol>

<h2>Provenance</h2>
<p class="small">Scripts and results are in <span class="mono">2026-09-28-resonator-power-onset/</span>:
<span class="mono">load.py</span> re-fits every 02b map since 25 Sep in <span class="mono">~/.qualibrate/user_storage/tinycal/data</span> with the node
analysis frozen for n7 (<span class="mono">~/qab-runs/qua-libs-n7</span>, 3b2acbf plus the 28 Sep 02b diff);
<span class="mono">cand3.py</span> is the rule (final settings {esc(json.dumps(res['final']))}), <span class="mono">cand.py</span> and
<span class="mono">cand2.py</span> the rejected ones; <span class="mono">score.py</span> the scoring and stress tests;
<span class="mono">report_data.py</span> the figures and <span class="mono">results.json</span>. The eye picks were made on the node's own
figures from the n7 run. IQCC powers come from each n7 cell's source state (<span class="mono">~/qab-runs/n7-analysis.json</span>).
Generated by <span class="mono">make_resonator_power_onset_report.py</span>.</p>
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
    page = build()
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
