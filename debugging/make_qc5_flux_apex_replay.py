"""Debugging note: replaying gilboa qC5's flux-map commit turn with and without a recipe note.

Reads replay_qc5_results.jsonl (written by replay_qc5.py) and writes
2026-09-21-qc5-flux-apex-replay.html next to it. Figures are copies of the run plots
under figures/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_fwcmp2_report import CSS, esc  # noqa: E402

UI = "http://127.0.0.1:8765/#/run"
RUNS = "~/code/QM/tinycal/runs"
CONTEXTS = [
    {
        "key": "original (turn 12)",
        "title": "Context A — first attempt",
        "run": "n4_gilboa_openrouter_20260920-1712",
        "turn": 12, "msg": 23, "node_run": "m-XRRNY2", "at": "20 Sep 18:14",
        "readout": "0.05 V (provisional, ~10× too low — the readout-power failure fixed by the recipe change)",
        "r2": "0.157", "coverage": "0.96 of a fitted 1.04 V period", "star": "−0.13 V",
        "fig_bg": "figures/qC5-original-m-XRRNY2-bg.png", "fig_raw": "figures/qC5-original-m-XRRNY2-raw.png",
        "images": 7,
    },
    {
        "key": "rerun (turn 14)",
        "title": "Context B — readout-fix rerun",
        "run": "n4_gilboa_openrouter-R_20260920-1712",
        "turn": 14, "msg": 27, "node_run": "m-VKHBB9", "at": "21 Sep 02:19",
        "readout": "0.126 V (punch-out found, readout fine)",
        "r2": "0.646", "coverage": "0.38 of a fitted period", "star": "−0.15 V",
        "fig_bg": "figures/qC5-rerun-m-VKHBB9-bg.png", "fig_raw": "figures/qC5-rerun-m-VKHBB9-raw.png",
        "images": 11,
    },
]
NOTE = (
    "Every node that fits the data draws its fit and the point it proposes (a sweet spot, "
    "a resonance, an optimum) on its figure. Before committing a proposal, look at the "
    "figure and check that the fit actually follows the data and that the marked point "
    "sits where physics says it should — the fit can be wrong, and a proposal from a poor "
    "fit is not evidence. If the marked point does not make physical sense, do not "
    "commit it; re-measure so that the feature is unambiguous instead."
)


def bucket(verdict: str) -> str:
    if verdict.startswith("commits apex"):
        v = float(verdict.split()[2])
        return "apex" if abs(v) > 0.05 else "zero"
    if verdict.startswith("re-measures"):
        return "remeasure"
    return "other"


BUCKET_LABEL = {
    "apex": ("commits the node's star (wrong)", "crit"),
    "zero": ("commits ≈0 V read off the figure (right)", "ok"),
    "remeasure": ("re-measures the flux map", "warn"),
    "other": ("other", ""),
}


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "replay_qc5_results.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        ctx, variant, n = [p.strip() for p in r["tag"].split("|")]
        r.update(ctx=ctx, variant=variant, n=n, bucket=bucket(r["verdict"]))
    counts = {(r["ctx"], r["variant"]): Counter(rr["bucket"] for rr in rows if rr["ctx"] == r["ctx"] and rr["variant"] == r["variant"]) for r in rows}
    per = {k: sum(v.values()) for k, v in counts.items()}

    def cell(ctx, variant, b):
        c = counts[(ctx, variant)][b]
        cls = BUCKET_LABEL[b][1] if c else "muted"
        return f'<td class="num"><span class="st {cls}">{c}/{per[(ctx, variant)]}</span></td>'

    summary = ['<div class="scroll"><table class="grid"><thead><tr><th>Context</th><th>Recipe</th>'
               + "".join(f"<th>{esc(BUCKET_LABEL[b][0])}</th>" for b in ("apex", "zero", "remeasure")) + "</tr></thead><tbody>"]
    for c in CONTEXTS:
        for v in ("without note", "with note"):
            summary.append(f"<tr><td>{esc(c['title'])}</td><td>{esc(v)}</td>" + "".join(cell(c["key"], v, b) for b in ("apex", "zero", "remeasure")) + "</tr>")
    summary.append("</tbody></table></div>")

    trials = ['<div class="scroll"><table class="grid small"><thead><tr><th>Context</th><th>Recipe</th><th>#</th><th>Reply</th><th>Model\'s own note (excerpt)</th><th>out tok</th><th>s</th></tr></thead><tbody>']
    for r in rows:
        lab, cls = BUCKET_LABEL[r["bucket"]]
        trials.append(
            f"<tr><td>{esc(r['ctx'])}</td><td>{esc(r['variant'])}</td><td class=\"num\">{esc(r['n'])}</td>"
            f"<td><span class=\"st {cls}\">{esc(r['verdict'][:90])}</span></td>"
            f"<td>{esc((r['note'] or r['text']).strip()[:260])}</td><td class=\"num\">{r['out']}</td><td class=\"num\">{r['s']}</td></tr>")
    trials.append("</tbody></table></div>")

    ctx_rows = []
    for c in CONTEXTS:
        ctx_rows.append(
            f"<tr><td>{esc(c['title'])}</td>"
            f"<td><a href=\"{UI}/{c['run']}/qC5\">{c['run']}</a> · qC5<br><span class=\"small mono\">{RUNS}/{c['run']}/qC5/transcript.json</span></td>"
            f"<td class=\"num\">turn {c['turn']} (messages[:{c['msg']}])</td><td class=\"mono\">{c['node_run']}</td>"
            f"<td>{esc(c['readout'])}</td><td class=\"num\">{c['r2']}</td><td>{esc(c['coverage'])}</td><td class=\"num\">{c['star']}</td><td class=\"num\">{c['images']}</td></tr>")

    figs = []
    for c in CONTEXTS:
        figs.append(
            f"<figure><figcaption><b>{esc(c['title'])}</b> — background-subtracted resonator-vs-flux map the model saw at turn {c['turn']} "
            f"(node run {c['node_run']}, run {c['run']}). The star is the node's proposal; the dotted lines show where it sits relative to the dip.</figcaption>"
            f"<img src=\"{c['fig_bg']}\" alt=\"qC5 flux map, background subtracted\"></figure>"
            f"<figure><figcaption>Same measurement, raw |IQ| (also in the context).</figcaption><img src=\"{c['fig_raw']}\" alt=\"qC5 flux map, raw\"></figure>")
    figs.append(
        "<figure><figcaption><b>Same pattern on qC2</b> (run n4_gilboa_openrouter_20260920-1712, node run m-19X710, committed at turn 10): "
        "arc apex near 0 V, R² 0.37, star at −0.16 V at the top of the frame. Not replayed here.</figcaption>"
        "<img src=\"figures/qC2-original-m-19X710-bg.png\" alt=\"qC2 flux map, background subtracted\"></figure>")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>qC5 flux-apex replay</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}
figure img {{ max-width:100%; height:auto; display:block; }}
blockquote {{ margin:14px 0; padding:10px 16px; border-left:3px solid var(--accent); background:var(--surface); max-width:76ch; }}
</style></head><body><div class="page">
<div class="eyebrow">debugging · gilboa · tinycal · qwen3.8-27b via OpenRouter · 21 Sep 2026</div>
<h1>Does a "check the figure" recipe note stop the model committing a bad flux-map apex?</h1>
<p class="lede">Gilboa qC5 failed twice on the 20–21 Sep night because the model committed the flux offset that
<span class="mono">resonator_spectroscopy_vs_flux</span> proposed from a failed arc fit. The proposed point was visibly off the
resonator dip in the node's own figure. We replayed the exact commit turn — same system prompt, tools, messages and
images — five times each with and without one generic sentence added to the bring-up recipe, on both contexts.</p>

<h2>The failure</h2>
<p>On gilboa's C row the resonator arc is shallow (≈2 MHz peak to peak over a ≈1 V flux period) and its apex is within a
few mV of 0 V (reference <span class="mono">joint_offset</span> 0.006 V). The node's cosine fit fails on the noisy arc
(R² 0.16 and 0.65 against a 0.80 threshold) and falls back to a "measured maximum", which lands on the noisiest column of the
trace: the star is drawn at the <em>top of the frequency axis</em>, −0.13 V / −0.15 V, nowhere near the dip. The proposal
still goes into <span class="mono">proposed_updates</span>, and the model committed it in both runs (turn 12 of the first
attempt, turn 14 of the readout-fix rerun) with notes such as "the measured apex is clearly visible in the plot". At −0.15 V the
qubit sits ≈0.15 Φ₀ down the arc: f₀₁ far from the seed, tiny dispersive shift, and the rest of the run (19 spectroscopies,
22 flat Rabis, then a drive-line rewiring) never recovered.</p>

<h2>Contexts replayed</h2>
<div class="scroll"><table class="grid small"><thead><tr><th>Context</th><th>Run · transcript</th><th>Replayed turn</th><th>Node run</th>
<th>Readout amplitude at the time</th><th>Arc R²</th><th>Period coverage</th><th>Node's star</th><th>Images in context</th></tr></thead>
<tbody>{''.join(ctx_rows)}</tbody></table></div>
<p class="small muted">Contexts are <span class="mono">transcript.json[:N]</span> of the run with the plot pointers restored to the
PNGs that were still inside tinycal's 16-image window at that turn (first image drop in both runs was at turn 20). Model config from
the run's start event: qwen/qwen3.8-27b, effort high, max_tokens 16 000, tool history native. 27–31 k input tokens per call.</p>

<h2>The note</h2>
<p>Inserted into <span class="mono">bringup_recipes/flux_tunable_1q.md</span> as its own paragraph just before "When a node fails or its
result contradicts an earlier one" — i.e. in the system prompt, not as a per-turn message:</p>
<blockquote>{esc(NOTE)}</blockquote>

<h2>Results — 5 trials per cell</h2>
{''.join(summary)}
<p><b>With a clean figure the note flips the outcome.</b> On context B every trial with the note rejected the star: four wrote
<span class="mono">joint_offset = 0</span> reasoning that "the arc clearly peaks near 0 V, not at −0.15 V where the fit placed it"
(true sweet spot 0.006 V), one re-ran the node with more shots. Without the note the same figure was committed at −0.15 V in 4 of 5
trials, one of them "confirmed by visual inspection of the raw data".</p>
<p><b>With a faint figure the note barely helps.</b> On context A the arc is only marginally above the noise (readout 10× too low), and
the model still commits −0.13 V in 3 of 5 trials with the note. The trials that re-measured reasoned correctly ("the proposed apex
frequency is at the edge of the sweep window, not where the dip is visible at ~7.6108 GHz") but were also the long ones (4 000+ output
tokens); the short replies commit. That failure is upstream of the flux map and is what the readout-power recipe change already addresses.</p>
<p>Every re-measure choice was sensible: the same ±0.5 V range (the port limit on gilboa's direct LF-FEM z lines), 200–300 shots,
a 15–20 MHz span or more flux points. None tried an out-of-range sweep.</p>

<h3>Every trial</h3>
{''.join(trials)}

<h2>What to change</h2>
<ul>
<li><b>Recipe</b>: add the note as written; it costs nothing and fixes the clean-figure case outright.</li>
<li><b>Node (qua-libs 02c)</b>: do not put <span class="mono">joint_offset</span> into <span class="mono">proposed_updates</span> when the arc fit fails,
and never draw the star off the measured trace — the "measured maximum" fallback picked a noise column at the frame edge in all three
gilboa cases (qC5 twice, qC2 once). A model that is told "inspect before accepting" in the legend and "proposing the measured maximum instead" in
the warning still accepts it 80 % of the time.</li>
<li><b>Readout first</b>: the faint-figure context shows that no wording rescues a map taken at 10× too little readout power; the punch-out
line in the recipe (already applied for the reruns) is the fix there.</li>
</ul>

<h2>Figures</h2>
{''.join(figs)}

<h2>Reproduce</h2>
<p class="small"><span class="mono">replay_qc5.py</span> (copy of <span class="mono">~/qab-runs/replay_qc5.py</span>) rebuilds the contexts from the run
directories and calls the model through tinycal's own client; <span class="mono">replay_qc5_results.jsonl</span> holds the 20 replies. Run with
<span class="mono">cd ~/code/QM/tinycal && set -a && source .env && set +a && PYTHONPATH=&lt;this dir&gt; .venv/bin/python replay_qc5.py 5</span>.
Runs in the tinycal UI: <a href="{UI}/{CONTEXTS[0]['run']}/qC5">{CONTEXTS[0]['run']}/qC5</a>,
<a href="{UI}/{CONTEXTS[1]['run']}/qC5">{CONTEXTS[1]['run']}/qC5</a>,
<a href="{UI}/n4_gilboa_openrouter_20260920-1712/qC2">n4_gilboa_openrouter_20260920-1712/qC2</a>.
Context: <a href="../2026-09-20-splash-vs-openrouter-qwen-tinycal.html">the 20–21 Sep Splash-vs-OpenRouter report</a>.</p>
</div></body></html>"""
    out = HERE / "2026-09-21-qc5-flux-apex-replay.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, len(rows), "trials")


if __name__ == "__main__":
    main()
