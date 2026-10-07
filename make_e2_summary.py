#!/usr/bin/env python3
"""The e2 summary page: make_n12_summary.py's page for e2, the 6 Oct 2026 run of qua-decision-engine (a rule-based
calibration decision engine, no language model) on n16's 37 single-qubit bring-ups, beside n16 (tinycal with qwen3.8-27b,
2 Oct) on the same qubits.

    python3 make_e2_summary.py

The layout and every number come from make_n12_summary.py and make_n12_report.py's collection, pointed at the e2 and n16
work dirs (the engine's cells are "<backend>-engine-<qubit>", rep.CELL_MARKS). This script adds n16's column to the results
table, n16's series to the plots, and the reruns of the two cells a node defect stopped (~/qab-runs/e2r-* work dirs, each a
copy of e2's prepared state, so the same scramble). The verdicts below are checked against the engine's trace and
~/qab-runs/e2-LOG.md. The output is page content for a claude.ai artifact.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402
from make_fwcmp2_report import CSS, esc, median, minutes  # noqa: E402
from make_fwcmp2_report import pct as fpct  # noqa: E402

rep = summ.rep
QAB = Path.home() / "qab-runs"
TINY_RUNS = Path.home() / "code/QM/tinycal/runs"
OUT = Path(__file__).with_name("2026-10-06-e2-summary.html")
STAMP = (QAB / "e2-stamp.txt").read_text().strip()
N16_STAMP = (QAB / "n16-stamp.txt").read_text().strip()
LABEL = {"old": "n16", "new": "e2"}

# (backend, qubit) -> why an e2 run did not complete (the engine stopped it: "stuck")
NOT_COMPLETED = {
    ("gilboa", "qD2"): "stuck at qubit spectroscopy: T1 ≈ 0.9 µs, too short for a clean chirp over 20 MHz; a correct stop "
                       "(n16's agent completed it at 4.3 % RB)",
    ("gilboa", "qD5"): "stuck at qubit spectroscopy: the chirp analysis missed a line plain on its map, and the window walk "
                       "moved away from it; fixed in qua-libs 3e4a6988, the rerun completed",
    ("arbel", "qC4"): "stuck at resonator vs flux: resonator identification had reported a 59 kHz spur 16 MHz above the "
                      "resonator as successful; fixed in qua-libs 2c35a8e1, the rerun completed",
}

# The reruns, in order: (work dir stamp, backend, qubit, what the library added). All on e2's scrambled state.
RERUNS = [
    ("20261006-0553", "gilboa", "qD5", "3e4a6988", "chirp: excitation gate read with the ground reference's noise; weak lines "
                                                    "found on two pooled drive levels"),
    ("20261006-0557", "gilboa", "qD5", "727a1f29", "+ resonator vs flux: an untracked dip is mapped once more at 4× the shots"),
    ("20261006-0559", "gilboa", "qD5", "f8247d9d", "+ the same for an outlier apex"),
    ("20261006-0614", "arbel", "qC4", "2c35a8e1", "+ resonator identification: a spur is left out and the candidates found "
                                                   "again; it publishes a next action"),
]


def configure() -> None:
    rep.NIGHTS = {"old": ("n16",), "new": ("e2",)}
    rep.NIGHT_LABEL = {"old": "n16 (agent)", "new": "e2 (engine)"}
    rep.CELL_MARKS = ("-tinycal-", "-engine-")
    # both nights ran graph 80 (FluxTunableTransmon_BringUp), the order make_n12_report calls "new"
    rep.RECIPE["old"] = rep.RECIPE["new"]
    rep.NOT_MEANINGFUL = {}
    rep.NOT_COMPLETED = {("new", b, q): why for (b, q), why in NOT_COMPLETED.items()}
    summ.RECOVERY_NIGHTS = (("n16", N16_STAMP), ("e2", STAMP))


def context(s: dict) -> str:
    a, b = (n13summ._minute(t) for t in s["span"])
    return (f"{a:%-d %b %Y}, {a:%H:%M}–{b:%H:%M} CEST · n16's {s['n']} qubits on qolab, arbel and gilboa · qua-decision-engine, "
            "a rule-based walk of the graph that follows each node's structured next action, no language model · beside n16, "
            "tinycal with qwen3.8-27b on 2 Oct")


E2 = summ.Night(
    label="e2", stamp=STAMP, out=OUT,
    title="e2 Summary",
    description="e2, 6 Oct 2026: the decision engine (no language model) on n16's 37 single-qubit bring-ups, beside n16's "
                "tinycal and qwen3.8-27b, in one table, four plots, the reruns of the two stops a node defect caused and a "
                "per-qubit check against IQCC.",
    eyebrow="e2 decision engine",
    h1="e2 decision-engine bring-ups",
    context=context,
    chip_sub=lambda s: f"6 Oct · {s['n']} qubits · engine, no LLM",
    rb_note=n13summ.rb_note,
    pull="the pull each night started from (e2: all three backends pulled 6 Oct 03:53, arbel and gilboa on square readout "
         "with ≥ 3000 ns depletion; n16: 2 Oct 16:32)",
    pull_short="the pull each night started from",
    recovery_intro="One qubit per pair of rows: <b>n16</b>, the 2 Oct run with tinycal and qwen3.8-27b, and <b>e2</b>, the "
                   "decision engine on 6 Oct, each against the IQCC pull its own night started from. e2's stuck cells are "
                   "shown as they ran, before the reruns.",
    recovery_sort=("e2", "n16"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in e2, fewest first, then by n16's count.",
    spec="252e28f8ab313894",
    scramble_note="<p class='small muted'>The same spec as n16, with its state-field allowlist; e2's states were pulled and "
                  "scrambled afresh on 6 Oct.</p>",
)


# ----------------------------------------------------------------------------- results table, two nights
def _share(d, rest, n, what) -> str:
    if d is None:
        return "<span class='muted'>—</span>"
    out = f"<span class='num'>{100 * d[0] / d[1]:.1f}% ({d[0] / n:.1f} / calibration)</span>"
    if rest is not None:
        out += f"<br><span class='small muted'>not completed: {100 * rest[0] / rest[1]:.0f}% of {rest[1]} {what}</span>"
    return out


def results_table(rows: dict, S: dict) -> str:
    def cells(fn):
        return [fn(k, rows[k], S[k]) for k in ("old", "new")]

    def qubits(k, rs, s):
        return summ.qubit_list(rs)

    def meaningful(k, rs, s):
        return f"<b class='num'>{len(s['meaningful'])}/{s['n']} ({100 * len(s['meaningful']) / s['n']:.0f}%)</b>"

    def valid(k, rs, s):
        fids = [r["gate_fid"] for r in s["valid"] if r["gate_fid"] is not None]
        return (f"<span class='num'><b>{len(s['valid'])}</b>, median {fpct(median(fids))}</span>"
                f"<div class='small muted'>{esc(n13summ.rb_note(s))}</div>")

    def readout(k, rs, s):
        return f"<span class='num'><b>{100 * s['ro_med']:.1f} %</b> · {s['ro_95']} of {s['ro_n']} ≥ 95 %</span>"

    def judge(k, rs, s):
        a, b = s["judge"]
        return f"<span class='num'>{a}/{b} ({100 * a / b:.0f}%)</span>"

    def decide_time(k, rs, s):
        done = s["done"]
        if k == "new":
            return ("<span class='num'>0 s</span><div class='small muted'>no language model: each decision is a table "
                    "lookup on the node's next action</div>")
        return (f"<span class='num'>{minutes(median([r['model_s'] for r in done if r['model_s'] is not None]))}</span>"
                f"<div class='small muted'>median over the {len(done)} completed calibrations</div>")

    def qpu(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>{median([r['qpu_s'] for r in done if r['qpu_s'] is not None]) / 60:.1f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · night total "
                f"{s['qpu_tot'] / 60:.0f} min</div>")

    def wall(k, rs, s):
        done = s["done"]
        w = [r["wall_s"] for r in done if r["wall_s"] is not None]
        return (f"<span class='num'>{median(w) / 60:.0f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · longest {max(w) / 60:.0f} min</div>")

    def cost(k, rs, s):
        done = s["done"]
        if k == "new":
            return "<span class='num'>$0</span><div class='small muted'>no language model</div>"
        return (f"<span class='num'>${median([r['judge_cost'] for r in done if r['judge_cost'] is not None]):.2f}</span>"
                f"<div class='small muted'>median over the {len(done)} completed, at prices.yaml · night total "
                f"${s['judge_cost']:.2f}</div>")

    def runs(k, rs, s):
        return (f"<span class='num'>{s['runs']} · {s['not_ok']} failed or refused</span>"
                f"<div class='small muted'>median {s['runs_med']:.0f} per qubit</div>")

    def overrides(k, rs, s):
        out = _share(s["ov_done"], s["ov_rest"], len(s["done"]) or 1, "writes")
        if k == "new":
            paths = {p.split("/", 3)[-1] for r in rs for p in r["st"]["unproposed_paths"]}
            out += (f"<div class='small muted'>all of them the engine's two carry rules ({esc(', '.join(sorted(paths)))}), "
                    "written from the nodes' numerics because no node proposes them</div>")
        return out

    def off_order(k, rs, s):
        return _share(s["oo_done"], s["oo_rest"], len(s["done"]) or 1, "node runs")

    body = [
        ("qubits measured (green: completed with meaningful results; red: not completed, reason on hover; the small letter "
         "is the device)", cells(qubits)),
        ("completed with meaningful results", cells(meaningful)),
        ("valid RB (the fit saw ≥ 1 decay length): runs, median fidelity per gate", cells(valid)),
        ("readout assignment fidelity, median over the runs that measured one", cells(readout)),
        ("judge: scrambled parameters back in the ballpark (readout frequency and amplitude, f₀₁, x180 amplitude)",
         cells(judge)),
        ("decision time / calibration (the agent's model time; the engine's)", cells(decide_time)),
        ("QPU median time / calibration", cells(qpu)),
        ("wall-clock median time / calibration", cells(wall)),
        ("judge median cost / calibration", cells(cost)),
        ("node runs, all qubits", cells(runs)),
        ("state writes that are not a node's proposal (no node proposed the path, or the value is more than 0.1 % from the "
         "latest proposal), completed calibrations; below, the runs that did not complete", cells(overrides)),
        ("node runs off the recipe order: stepped back to an earlier step or skipped one (retries and re-runs of the node just "
         "run are not counted), completed calibrations; below, the runs that did not complete", cells(off_order)),
    ]
    trs = "".join(f"<tr><th class='rowh'>{esc(lab)}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
                  for lab, vals in body)
    subs = {"old": f"2 Oct · {S['old']['n']} qubits · tinycal, qwen3.8-27b", "new": _text_chip(S["new"])}
    head = "<thead><tr><th></th>" + "".join(
        f"<th class='grp'><span class='chip'><i style='background:{rep.SERIES[k]}'></i>{esc(LABEL[k])}</span>"
        f"<br><span class='small muted'>{esc(subs[k])}</span></th>" for k in ("old", "new")) + "</tr></thead>"
    return f"<div class='scroll'><table class='grid pivot sum sum2'>{head}<tbody>{trs}</tbody></table></div>"


def _text_chip(s: dict) -> str:
    return E2.chip_sub(s) if callable(E2.chip_sub) else E2.chip_sub


# ----------------------------------------------------------------------------- plots, both nights
def readout_svg2(points, *, xlabel, ylabel, lo=0.5, hi=40.0, W=460, H=380) -> str:
    """summ.readout_svg with one series per night: circles for n16, diamonds for e2. ``points`` = (x, y, label, night)."""
    L, R, T, B = 50, 14, 12, 42
    llo, lhi = math.log10(lo), math.log10(hi)
    sx = lambda v: L + (W - L - R) * (lhi - min(max(math.log10(v), llo), lhi)) / (lhi - llo)  # noqa: E731
    sy = lambda v: T + (H - T - B) * (min(max(math.log10(v), llo), lhi) - llo) / (lhi - llo)  # noqa: E731
    out = [f'<svg class="scatter" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(ylabel)} against {esc(xlabel)}">']
    for err in (1, 2, 5, 10, 20, 30):
        out.append(f'<line x1="{sx(err):.1f}" x2="{sx(err):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                   f'<line y1="{sy(err):.1f}" y2="{sy(err):.1f}" x1="{L}" x2="{W - R}" class="grid"/>'
                   f'<text x="{sx(err):.1f}" y="{H - B + 15}" class="tick" text-anchor="middle">{100 - err:g}%</text>'
                   f'<text x="{L - 6}" y="{sy(err) + 3:.1f}" class="tick" text-anchor="end">{100 - err:g}%</text>')
    out.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" class="guide"/>'
               f'<text x="{sx(2) - 6:.1f}" y="{sy(2) - 6:.1f}" class="tick" text-anchor="end">y = x</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" class="frame"/>')
    for x, y, label, night in sorted(points, key=lambda p: p[3] != "old"):
        cx, cy = sx(x), sy(y)
        if night == "old":
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" class="pt old"><title>{esc(label)}</title></circle>')
        else:
            d = 5.6
            out.append(f'<path d="M{cx:.1f} {cy - d:.1f}L{cx + d:.1f} {cy:.1f}L{cx:.1f} {cy + d:.1f}L{cx - d:.1f} {cy:.1f}Z" '
                       f'class="pt new"><title>{esc(label)}</title></path>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 5}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(13,{(T + H - B) / 2:.0f}) rotate(-90)" class="lab" text-anchor="middle">{esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def plots(rows: list) -> str:
    """summ.plots for e2's runs alone, and e2's gate error against n16's on the same qubit."""
    ref_pts, floor_pts, ro_pts, no_ref, no_coh, no_ro = [], [], [], [], [], []
    links = {}
    n16_fid = {(r["backend"], r["q"]): r["gate_fid"] for r in rows
               if r["night"] == "old" and r["status"] == "completed" and r["gate_fid"] is not None}
    vs_pts, no_vs = [], []
    for r in rows:
        if r["night"] != "new":
            continue
        tag = f"{r['backend']} {r['q']}"
        if r["status"] == "completed" and r["gate_fid"] is not None and (r["backend"], r["q"]) in n16_fid:
            old = n16_fid[(r["backend"], r["q"])]
            vs_pts.append((100 * (1 - old), 100 * (1 - r["gate_fid"]),
                           f"{tag} · e2 {fpct(r['gate_fid'])} · n16 {fpct(old)}", r["night"]))
            links[esc(vs_pts[-1][2])] = summ.ui_url(r)
        else:
            no_vs.append(tag + (" (stuck in e2)" if r["status"] != "completed"
                                else " (no valid RB in e2)" if r["gate_fid"] is None else " (no valid RB in n16)"))
        if r["status"] == "completed" and r["gate_fid"] is not None:
            err = 100 * (1 - r["gate_fid"])
            label = f"{tag} · {fpct(r['gate_fid'])}"
            if r["ref"] is not None and r["ref"] < 1:
                ref_pts.append((100 * (1 - r["ref"]), err, label + f" · reference {fpct(r['ref'])} ({r['ref_date'][:16]})",
                                r["night"]))
                links[esc(ref_pts[-1][2])] = summ.ui_url(r)
            else:
                no_ref.append(tag)
            t1, t2e, L = r["t1"], r["t2e"], r["x180_len"]
            if t1 and t2e and L and t1 > 0 and t2e > 0:
                floor = 100 * (L * 1e-9 / 3.0) * (1.0 / t1 + 1.0 / t2e)
                floor_pts.append((floor, err, label + f" · T1 {1e6 * t1:.0f} µs, T2e {1e6 * t2e:.0f} µs, x180 {L:.0f} ns · "
                                  f"floor {floor:.3f}%", r["night"]))
                links[esc(floor_pts[-1][2])] = summ.ui_url(r)
            else:
                no_coh.append(tag)
        if r["status"] == "completed":
            ref = summ.lab_readout(r)
            if r["readout"] is None or ref is None:
                no_ro.append(tag)
            else:
                ro_pts.append((100 * (1 - ref), 100 * (1 - r["readout"]),
                               f"{tag} · measured {100 * r['readout']:.1f}% · lab {100 * ref:.1f}%", r["night"]))
                links[esc(ro_pts[-1][2])] = summ.ui_url(r)
    kw = {"W": 460, "H": 380}
    a = rep.scatter_svg(ref_pts, xlabel="reference calibration's gate error, %", ylabel="measured gate error, %", **kw)
    v = rep.scatter_svg(vs_pts, xlabel="n16's measured gate error, same qubit, %", ylabel="e2's measured gate error, %", **kw)
    b = rep.scatter_svg(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %", ylabel="measured gate error, %",
                        guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x"), **kw)
    c = readout_svg2(ro_pts, xlabel="reference readout assignment fidelity", ylabel="measured readout assignment fidelity")
    a, v, b, c = (summ.link_points(x, links) for x in (a, v, b, c))

    not_shown = lambda xs, why: f" Not shown: {esc(', '.join(xs))} ({why})." if xs else ""  # noqa: E731
    lower = sum(1 for x, y, _, _ in vs_pts if y < x)
    return (
        "<div class='plots2'>"
        f"<div class='plotpanel'><h3>Measured gate error against the reference calibration</h3>"
        f"<p class='cap small muted'>e2's completed runs with a valid RB ({len(ref_pts)}), error per gate (EPC ÷ 1.875) "
        f"against the lab's gate_fidelity.averaged in the pull e2 started from (all three backends pulled 6 Oct 03:53, "
        f"arbel and gilboa on square readout with ≥ 3000 ns depletion); below the diagonal beats the lab."
        f"{not_shown(no_ref, 'no reference')}</p>{a}</div>"
        f"<div class='plotpanel'><h3>Measured gate error against n16</h3>"
        f"<p class='cap small muted'>The {len(vs_pts)} qubits with a valid RB in both runs: e2's error per gate against "
        f"n16's on the same qubit (2 Oct, tinycal with qwen3.8-27b); below the diagonal e2 is better, on {lower} of "
        f"{len(vs_pts)}. The two runs started from pulls four days apart, so a qubit whose coherence moved in between "
        f"moves its point too.{(' Not shown: ' + esc(', '.join(no_vs)) + '.') if no_vs else ''}</p>{v}</div>"
        "</div><div class='plots2'>"
        f"<div class='plotpanel'><h3>Measured gate error against the coherence floor</h3>"
        f"<p class='cap small muted'>e2's completed runs ({len(floor_pts)}); floor = (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) "
        f"from the run's own values.{not_shown(no_coh, 'no positive T1/T2echo')}</p>{b}</div>"
        f"<div class='plotpanel'><h3>Measured readout assignment fidelity against the reference calibration</h3>"
        f"<p class='cap small muted'>e2's completed runs ({len(ro_pts)}); IQ_blobs' assignment fidelity, (P(g|g) + P(e|e))/2, "
        f"against the same from the lab's confusion matrix in the pull e2 started from; log axis of the infidelity. Above "
        f"the diagonal beats the lab.{not_shown(no_ro, 'no value')}</p>{c}</div>"
        "</div>")


# ----------------------------------------------------------------------------- the reruns
def rerun_rows() -> dict:
    """work dir name -> the collected row of each rerun cell."""
    saved = rep.NIGHTS, rep.NOT_COMPLETED
    rep.NIGHTS, rep.RECIPE["rerun"] = {"rerun": ("e2r",)}, rep.RECIPE["new"]
    rep.NOT_COMPLETED = {}
    try:
        return {r["work"]: r for r in rep.collect()}
    finally:
        rep.NIGHTS, rep.NOT_COMPLETED = saved


def reruns_section(new_rows: list) -> str:
    got = rerun_rows()
    e2_by_q = {(r["backend"], r["q"]): r for r in new_rows}
    trs = []
    for stamp, b, q, rev, what in RERUNS:
        work = f"e2r-{b}-{stamp}"
        r = got.get(work)
        assert r is not None, f"no rerun cell in {work}"
        rev_file = (QAB / work / "qua-libs-rev.txt").read_text().split()[0]
        assert rev_file == rev, f"{work} ran qua-libs {rev_file}, not {rev}"
        detail = json.load(open(TINY_RUNS / r["run_id"] / "run.json"))["targets_detail"][q]
        when = datetime.fromisoformat(r["started"]).strftime("%H:%M")
        link = summ.UI.format(run_id=r["run_id"], q=q)
        if r["status"] == "completed":
            res = (f"<span class='q ok'>completed</span> RB {100 * r['rb']:.3f} % · readout {100 * r['readout']:.1f} % · "
                   f"judge {r['ballpark']}/{r['graded']}")
        else:
            _, node, reason = (detail["summary"].split(" -- ")[0].split(": ") + ["", ""])[:3]
            res = f"<span class='q bad'>stuck</span> at {esc(node)} ({esc(reason)})"
        trs.append(f"<tr><td>{esc(b)} {esc(q)}</td><td class='num'><a href='{esc(link)}' target='_blank' rel='noopener'>{when}"
                   f"</a></td><td><span class='mono'>{esc(rev)}</span> {esc(what)}</td><td>{res}</td>"
                   f"<td class='num'>{r['node_runs']}</td><td class='num'>{(r['qpu_s'] or 0) / 60:.1f} min</td></tr>")
    # e2 with the final rerun of each stuck qubit in place of its cell
    final = {}
    for stamp, b, q, _, _ in RERUNS:
        final[(b, q)] = got[f"e2r-{b}-{stamp}"]
    merged = [final.get((r["backend"], r["q"]), r) if final.get((r["backend"], r["q"]), {}).get("status") == "completed"
              else r for r in new_rows]
    sm = rep.night_stats(merged)
    fids = [r["gate_fid"] for r in sm["valid"] if r["gate_fid"] is not None]
    extra_qpu = sum((got[f"e2r-{b}-{st}"]["qpu_s"] or 0) for st, b, q, _, _ in RERUNS
                    if got[f"e2r-{b}-{st}"]["status"] != "completed")
    for key in final:
        assert key in e2_by_q
    return (
        "<h3 class='scr'>The two stops a node defect caused, fixed and rerun</h3>"
        "<p class='small muted cap2'>Each fix was replayed offline on the archived runs of that node before it reached the "
        "QPU (296 chirp spectroscopy runs, 747 resonator identifications: no other verdict changed, no empty window gained a "
        "line) and committed to qua-libs <span class='mono'>feat/node-next-action</span>. Each rerun starts from a copy of "
        "e2's own prepared and scrambled state. The two gilboa qD5 attempts that stopped had a readout power 15 and 19 dB under "
        "the power sweep's own onset (its \"fade\" branch), which left the flux map too faint to track; that is not "
        "fixed.</p>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>qubit</th><th>started</th>"
        "<th>qua-libs, and what it added</th><th>result</th><th>node runs</th><th>QPU</th></tr></thead><tbody>"
        + "".join(trs) + "</tbody></table></div>"
        f"<p class='small muted'>With each qubit's completed rerun in place of its stuck cell, e2 reads "
        f"<b>{len(sm['done'])}/{sm['n']} completed</b>, valid RB {len(sm['valid'])} with median fidelity per gate "
        f"{fpct(median(fids))}, readout median {100 * sm['ro_med']:.1f} % ({sm['ro_95']} of {sm['ro_n']} ≥ 95 %), judge "
        f"{sm['judge'][0]}/{sm['judge'][1]}, QPU {sm['qpu_tot'] / 60:.0f} min plus {extra_qpu / 60:.1f} min for the two "
        "attempts that stopped. These fixes were made from the very failures they rerun, so this line is not a blind "
        "measurement.</p>")


SUMMARY2_CSS = """
table.sum2 { min-width:640px; } table.sum2 th.rowh { width:40%; font-weight:500; font-size:13px; } table.sum2 td { font-size:14px; }
table.sum2 td { vertical-align:top; }
.slegend { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 0; } .slegend span { display:inline-flex; align-items:center; gap:6px; }
"""


def build() -> None:
    configure()
    rows = rep.collect()
    by = {k: [r for r in rows if r["night"] == k] for k in ("old", "new")}
    assert {r["run"] for r in by["old"]} == {"n16"} and {r["run"] for r in by["new"]} == {"e2"}, "wrong work dirs collected"
    S = {k: rep.night_stats(v) for k, v in by.items()}
    now = datetime.now().strftime("%d %b %Y %H:%M")
    page = f"""<title>{esc(E2.title)}</title>
<meta name="description" content="{esc(E2.description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{rep.EXTRA_CSS}{summ.SUMMARY_CSS}{SUMMARY2_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · {esc(E2.eyebrow)} · generated {esc(now)}</div>
<h1 style="margin-top:8px">{esc(E2.h1)}</h1>
<p class="context">{esc(context(S['new']))}</p>
{summ.scramble_table(E2)}
<h3 class='scr'>Results</h3>
{results_table(by, S)}
{summ.legend(by['new'], E2)}
<p class="small muted uinote">Click a qubit, here or on a plot, to open its run in tinycal's run viewer (<span class="mono">tinycal ui</span> on this machine, port 8765); the engine records its runs in tinycal's format.</p>
{reruns_section(by['new'])}
{plots(rows)}
{summ.recovery_table(by['new'], E2)}
</div>"""
    OUT.write_text(page)
    for k in ("old", "new"):
        s = S[k]
        fids = [r["gate_fid"] for r in s["valid"] if r["gate_fid"] is not None]
        print(f"{LABEL[k]}: completed {len(s['done'])}/{s['n']} · meaningful {len(s['meaningful'])} · valid RB {len(s['valid'])}, "
              f"median {fpct(median(fids))} · readout {100 * s['ro_med']:.1f} % · judge {s['judge'][0]}/{s['judge'][1]} · "
              f"QPU {s['qpu_tot'] / 60:.0f} min · runs {s['runs']} ({s['not_ok']} failed)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
