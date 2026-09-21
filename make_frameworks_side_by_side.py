#!/usr/bin/env python3
"""Two charts, side by side: tinycal against qua-agents on the 14-17 Sep 2026 rounds.

    uv run --project ~/code/QM/qua-agents-benchmark python make_frameworks_side_by_side.py

Left: calibrations completed out of those attempted, per model. Right: judge-priced cost per
attempted calibration, per model. The numbers are the ones `make_fwcmp2_report.py` collects for
the 14-17 Sep report (same cell documents, same exclusions: dag-walk, superseded cells and
qubit-runs ended by an operator-side incident are out), so the two pages never disagree.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import make_fwcmp2_report as R

OUT = Path(__file__).with_name("2026-09-21-frameworks-side-by-side.html")
FRAMEWORKS = ("tinycal", "qua-agents")


def esc(s) -> str:
    return html.escape(str(s))


def summarise(cells) -> dict:
    """(model, framework) -> the totals behind the two charts."""
    out = {}
    for model in R.MODEL_LABELS:
        for fw in FRAMEWORKS:
            runs = [t for c in cells if R.included(c) and c["model"] == model and c["fw"] == fw
                    for t in c["targets"] if not R.infra_hit(t)]
            done = [t for t in runs if t["status"] == "completed"]
            fids = sorted(t["gate_fid"] for t in done if t["gate_fid"] is not None)
            cost = sum(t["cost"] or 0 for t in runs)
            out[(model, fw)] = {
                "attempted": len(runs), "completed": len(done),
                "rate": len(done) / len(runs) if runs else None,
                "cost_total": cost, "cost_per_run": cost / len(runs) if runs else None,
                "cost_per_done": cost / len(done) if done else None,
                "fid_median": fids[len(fids) // 2] if fids else None,
                "agent_min": sum((t["model_s"] or 0) for t in runs) / len(runs) / 60 if runs else None,
                "backends": sorted({c["backend"] for c in cells if R.included(c) and c["model"] == model and c["fw"] == fw}),
            }
    return out


# ----------------------------------------------------------------------------- chart
BAR, GAP, GROUP_PAD = 20, 2, 18          # bar thickness, surface gap between the pair, air between models
LABEL_W, VALUE_W, W = 118, 96, 560       # row-label column, room for the value at the tip, total width


def grouped_bars(stats, key, label_fn, max_value, axis_ticks, tick_fmt, title, note, chart_id):
    """Horizontal grouped bars: one row per model, a tinycal bar over a qua-agents bar."""
    plot_w = W - LABEL_W - VALUE_W
    group_h = 2 * BAR + GAP + GROUP_PAD
    top = 8
    H = top + len(R.MODEL_LABELS) * group_h + 26
    x0 = LABEL_W

    def sx(v):
        return x0 + plot_w * v / max_value

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-t">',
             f'<title id="{chart_id}-t">{esc(title)}</title>']
    # gridlines + axis ticks
    for tv in axis_ticks:
        x = sx(tv)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{H - 24}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{H - 8}" text-anchor="middle">{esc(tick_fmt(tv))}</text>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{H - 24}"/>')
    for gi, model in enumerate(R.MODEL_LABELS):
        gy = top + gi * group_h + GROUP_PAD / 2
        parts.append(f'<text class="rowlabel" x="{x0 - 10}" y="{gy + BAR + GAP / 2 + 4:.1f}" text-anchor="end">{esc(model)}</text>')
        for fi, fw in enumerate(FRAMEWORKS):
            st = stats[(model, fw)]
            v = st[key]
            y = gy + fi * (BAR + GAP)
            if v is None:
                parts.append(f'<text class="value muted" x="{x0 + 6}" y="{y + BAR - 5}">no runs</text>')
                continue
            w = max(0.0, sx(v) - x0)
            r = min(4, w / 2)
            # square at the baseline, 4px rounded data-end
            path = (f'M{x0},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{BAR - 2 * r} a{r},{r} 0 0 1 -{r},{r} '
                    f'h-{w - r:.1f} z')
            tip = (f"{fw} · {model}: {label_fn(st)} — {st['completed']}/{st['attempted']} completed, "
                   f"${st['cost_total']:.2f} total at judge prices")
            parts.append(f'<path class="bar {fw.replace("-", "")}" d="{path}" data-tip="{esc(tip)}"/>')
            # hit target taller than the mark
            parts.append(f'<rect class="hit" x="{x0}" y="{y - 1}" width="{plot_w + VALUE_W}" height="{BAR + 2}" data-tip="{esc(tip)}"/>')
            parts.append(f'<text class="value" x="{x0 + w + 6:.1f}" y="{y + BAR - 5}">{esc(label_fn(st))}</text>')
    parts.append("</svg>")
    return (f'<figure><figcaption><b>{esc(title)}</b><span class="small muted">{note}</span></figcaption>'
            + "".join(parts) + "</figure>")


def table(stats):
    head = ("<thead><tr><th>model</th><th>framework</th><th class='num'>completed / attempted</th>"
            "<th class='num'>cost / calibration</th><th class='num'>cost / completed</th><th class='num'>cost total</th>"
            "<th class='num'>gate fidelity, median</th><th class='num'>agent time / calibration</th><th>backends</th></tr></thead>")
    rows = []
    for model in R.MODEL_LABELS:
        for fw in FRAMEWORKS:
            st = stats[(model, fw)]
            if not st["attempted"]:
                continue
            rows.append(
                f"<tr><td>{esc(model)}</td><td><span class='key {fw.replace('-', '')}'></span>{fw}</td>"
                f"<td class='num'>{st['completed']} / {st['attempted']} ({100 * st['rate']:.0f}%)</td>"
                f"<td class='num'>${st['cost_per_run']:.2f}</td>"
                f"<td class='num'>{'$%.2f' % st['cost_per_done'] if st['cost_per_done'] is not None else '—'}</td>"
                f"<td class='num'>${st['cost_total']:.2f}</td>"
                f"<td class='num'>{R.pct(st['fid_median']) if st['fid_median'] is not None else '—'}</td>"
                f"<td class='num'>{st['agent_min']:.0f} min</td><td class='small mono'>{', '.join(st['backends'])}</td></tr>")
    return f"<div class='scroll'><table class='grid'>{head}<tbody>{''.join(rows)}</tbody></table></div>"


# ----------------------------------------------------------------------------- page
CSS = """
:root { --bg:#fcfcfb; --fg:#1d1f21; --muted:#6b7378; --line:#e3e5e7; --grid:#ececea; --card:#ffffff;
  --qua:#2a78d6; --tinycal:#1baf7a; --tip-bg:#1d1f21; --tip-fg:#fcfcfb; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a19; --fg:#e8e8e4; --muted:#9aa2a7; --line:#33363a; --grid:#2a2c2f; --card:#212223;
  --qua:#3987e5; --tinycal:#199e70; --tip-bg:#e8e8e4; --tip-fg:#1a1a19; } }
:root[data-theme="dark"] {
  --bg:#1a1a19; --fg:#e8e8e4; --muted:#9aa2a7; --line:#33363a; --grid:#2a2c2f; --card:#212223;
  --qua:#3987e5; --tinycal:#199e70; --tip-bg:#e8e8e4; --tip-fg:#1a1a19; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  margin:0; padding:24px 16px 48px; }
main { max-width:1180px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; }
.lead { color:var(--muted); margin:0 0 20px; max-width:80ch; }
.legend { display:flex; gap:18px; margin:0 0 12px; font-size:13px; }
.key { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.key.tinycal { background:var(--tinycal); } .key.quaagents { background:var(--qua); }
.charts { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(340px, 100%), 1fr)); gap:20px; }
figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px 8px; min-width:0; }
figcaption { display:flex; flex-direction:column; gap:2px; margin-bottom:8px; }
svg.chart { width:100%; height:auto; display:block; overflow:visible; }
.grid { stroke:var(--grid); stroke-width:1; } .axis { stroke:var(--line); stroke-width:1; }
.tick, .rowlabel, .value { fill:var(--fg); font-size:12px; font-family:inherit; }
.tick { fill:var(--muted); font-size:11px; } .value { font-size:12px; } .value.muted { fill:var(--muted); }
.bar.tinycal { fill:var(--tinycal); } .bar.quaagents { fill:var(--qua); }
.hit { fill:transparent; cursor:default; }
.hit:hover + .value { font-weight:600; }
#tip { position:fixed; pointer-events:none; background:var(--tip-bg); color:var(--tip-fg); font-size:12px; line-height:1.35;
  padding:6px 9px; border-radius:6px; max-width:300px; box-shadow:0 2px 10px rgba(0,0,0,.18); z-index:10; }
h2 { font-size:16px; margin:28px 0 8px; }
.scroll { overflow-x:auto; }
table.grid { border-collapse:collapse; width:100%; font-size:13px; }
table.grid th, table.grid td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
table.grid th { color:var(--muted); font-weight:600; }
.num { text-align:right !important; font-variant-numeric:tabular-nums; }
.small { font-size:12px; } .muted { color:var(--muted); } .mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
ul.notes { padding-left:18px; max-width:90ch; overflow-wrap:anywhere; } ul.notes li { margin-bottom:4px; }
"""

JS = """
const tip = document.getElementById('tip'); tip.hidden = true;
for (const el of document.querySelectorAll('[data-tip]')) {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip; tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + 'px'; tip.style.top = (e.clientY + 16) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.hidden = true; });
}
"""


def build():
    cells, _rounds, _running = R.collect()
    stats = summarise(cells)
    max_cost = max((st["cost_per_run"] or 0) for st in stats.values())
    cost_top = 5 * (int(max_cost / 5) + 1)
    cost_ticks = list(range(0, cost_top + 1, 5 if cost_top <= 30 else 10))
    left = grouped_bars(
        stats, "rate", lambda st: f"{st['completed']}/{st['attempted']} ({100 * st['rate']:.0f}%)", 1.0,
        [0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{100 * v:.0f}%",
        "Calibrations completed", "share of attempted qubit-runs that reached the end of the graph", "c1")
    right = grouped_bars(
        stats, "cost_per_run", lambda st: f"${st['cost_per_run']:.2f}", cost_top,
        cost_ticks, lambda v: f"${v}",
        "Judge cost per calibration", "USD per attempted qubit-run at prices.yaml rates; failures charged to the pool", "c2")
    n_runs = sum(st["attempted"] for st in stats.values())
    legend = ("<div class='legend'><span><span class='key tinycal'></span>tinycal</span>"
              "<span><span class='key quaagents'></span>qua-agents</span></div>")
    notes = "".join(f"<li>{n}</li>" for n in [
        "Same data as <a href='2026-09-14-frameworks-opus-sonnet-qwen.html'>the 14–17 Sep report</a>: cell documents under "
        "<span class='mono'>~/qab-runs/fwcmp2-*</span>, opus-5 and sonnet-5 at high effort (14–15 Sep), qwen3.8-27b via OpenRouter "
        "(16–17 Sep), arbel, gilboa and qolab, each backend scrambled once and handed to both frameworks in sequence.",
        "A qubit-run is one target inside a cell. Excluded, as in that report: the dag-walk control, the superseded "
        "<span class='mono'>fwcmp2-qolab-20260915-0800</span> cells, set-aside cells (<span class='mono'>.stopped-</span>, "
        "<span class='mono'>.gateway-</span>, <span class='mono'>.failed</span>), and qubit-runs ended by an operator-side incident "
        "(proxy outage, provider connection error, full disk, gateway down). The single-target qB4 re-run carries its own cell name and is "
        "not folded into the sonnet-5 pool.",
        "Cost is the judge's: uncached input, output and cache reads/writes at <span class='mono'>prices.yaml</span> rates per model, "
        "summed over the qubit-run's turns. Per calibration = divided by attempted qubit-runs, so a framework that fails more pays for "
        "its failures here; the table also gives cost per completed qubit-run.",
        "Gate fidelity in the table is 1 − EPC / 1.875 from the final RB, median over completed qubit-runs.",
    ])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinycal vs qua-agents</title>
<style>{CSS}</style></head><body>
<main>
<h1>tinycal vs qua-agents, side by side</h1>
<p class="lead">{n_runs} scored qubit-runs from the 14–17 Sep 2026 rounds, grouped by model. Left, how often each framework
finished a single-qubit bring-up; right, what a calibration cost at the judge's prices. Hover a bar for the totals.
Rebuilt {datetime.now().strftime("%d %b %Y %H:%M")}.</p>
{legend}
<div class="charts">{left}{right}</div>
<h2>The numbers behind the charts</h2>
{table(stats)}
<h2>What is counted</h2>
<ul class="notes">{notes}</ul>
</main>
<div id="tip" hidden></div>
<script>{JS}</script>
</body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({n_runs} qubit-runs)")


if __name__ == "__main__":
    build()
