#!/usr/bin/env python3
"""A compact n12-only summary page: one table (the full report's "By night" pivot, n12 column only, nine rows), the two gate-error
scatters and the readout-fidelity scatter, n12 series only, and a per-qubit table of the values recovered against IQCC's, n12 beside
its rerun n13.

    python3 make_n12_summary.py

Every number comes from make_n12_report.py's collection (result.json, final quam_state, source-state, tinycal events.jsonl); this
script only selects and lays them out, so the two pages always agree. The output is page content for a claude.ai artifact.

The night-specific wording lives in one ``Night`` (N12 below); make_n13_summary.py builds the same page for the n13 rerun with its own.
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_n12_report as rep  # noqa: E402
from make_fwcmp2_report import CSS, esc, median, minutes  # noqa: E402
from make_fwcmp2_report import pct as fpct  # noqa: E402

OUT = Path(__file__).with_name("2026-09-29-n12-summary.html")

SUMMARY_CSS = """
.page { max-width:1040px; }
h1 { font-size:30px; }
.context { color:var(--ink-2); margin:6px 0 18px; max-width:none; }
table.sum1 { min-width:520px; } table.sum1 th.rowh { width:52%; font-weight:500; font-size:13px; } table.sum1 td { font-size:14px; }
.plots2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin:18px 0 8px; }
@media (max-width:900px) { .plots2 { grid-template-columns:1fr; } }
.plots2 .plotpanel { margin:0; max-width:none; min-width:0; }
.plots2 h3 { margin:0 0 2px; font-size:16px; }
.plots2 .cap { margin:0 0 6px; }
h3.scr { margin:18px 0 8px; font-size:17px; }
a.qa { color:inherit; text-decoration:none; } a.qa:hover .q { text-decoration:underline; } svg.scatter a { cursor:pointer; }
.uinote { margin:4px 0 0; }
table.scr { max-width:1000px; } table.scr td:nth-child(2), table.scr td:nth-child(3) { white-space:nowrap; }
.cap2 { margin:0 0 8px; max-width:none; }
table.rec { font-size:12.5px; } table.rec th { position:static; text-align:center; padding:5px 6px; font-size:12px; line-height:1.25; }
table.rec th .u { font-weight:400; color:var(--muted); font-size:11px; }
table.rec td { padding:2px 5px; text-align:center; white-space:nowrap; vertical-align:middle; }
table.rec th.qcol, table.rec td.qcol { text-align:left; padding-left:9px; }
table.rec .g0 { border-left:1px solid var(--line-strong); }
table.rec tr.top td { border-bottom:0; padding-top:5px; } table.rec tr.bot td { padding-bottom:5px; }
table.rec td.rn { font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:11px; }
table.rec td.rn a { color:var(--muted); } table.rec td.rn a:hover { color:var(--accent-ink); }
table.rec tfoot th, table.rec tfoot td { background:var(--surface-2); }
.rc { display:inline-block; min-width:50px; padding:1px 5px; border-radius:4px; font-family:"JetBrains Mono",Menlo,Consolas,monospace;
  font-size:11.5px; font-variant-numeric:tabular-nums; cursor:help; }
.rc.ok { background:var(--q-ok-bg); color:var(--q-ok); }
.rc.bad { background:var(--q-bad-bg); color:var(--q-bad); font-weight:600; }
.rc.bias { background:repeating-linear-gradient(135deg, var(--q-bad-bg) 0 4px, var(--surface) 4px 7px); color:var(--q-bad);
  font-weight:600; box-shadow:inset 0 0 0 1px var(--q-bad-bg); }
.rc.scr { background:var(--q-warn-bg); color:var(--q-warn); }
.rc.none { color:var(--muted); }
.rc.info { color:var(--ink-2); }
.rc.cnt { color:var(--ink); min-width:36px; cursor:default; }
.rlegend { margin:4px 0 8px; } .rlegend .rc { min-width:0; }
"""


RUNS_DIR = Path.home() / "qab-runs"
# tinycal's local run viewer (tinycal ui, default port): a qubit links to its run's page there
UI = "http://127.0.0.1:8765/#/run/{run_id}/{q}"


@dataclass(frozen=True)
class Night:
    """Everything the page says that belongs to the night it summarises. A field typed ``str | Callable`` may be a function of
    rep.night_stats() for that night, for text that quotes the data."""
    label: str  # "n12": the work dirs are ~/qab-runs/<label>-<backend>-<stamp>
    stamp: str
    out: Path
    title: str  # the page's <title>
    description: str  # the meta description
    eyebrow: str  # between "qua-agents benchmark ·" and "· generated <time>"
    h1: str
    context: str | Callable[[dict], str]  # the line under the h1
    chip_sub: str | Callable[[dict], str]  # under the night's chip in the table header
    rb_note: str | Callable[[dict], str]  # the small note in the valid-RB row
    pull: str  # where the lab's reference values come from, in the gate-error caption
    pull_short: str  # the same, in the readout caption
    recovery_intro: str  # the recovery table's first sentence (HTML)
    recovery_sort: tuple[str, ...]  # the nights whose counts order the recovery rows, first key first
    recovery_sort_text: str  # the caption's last sentence, saying how the rows are ordered
    spec: str = "f47053f145f74c8c"  # the scramble spec hash every cell of the night started from
    scramble_note: str = ""  # HTML after the scramble table's last sentence


def _text(v, s: dict) -> str:
    return v(s) if callable(v) else v


OUTCOMES = (("ok", "completed with meaningful results"), ("warn", "completed, results not meaningful"),
            ("bad", "not completed: escalated or failed"))


def legend(rows, night: Night) -> str:
    """What a qubit tag's colour means, each shown on one of this night's own qubits (reason on hover); an outcome no run of
    the night had says so instead of showing another night's qubit."""
    parts = []
    for cls, text in OUTCOMES:
        hits = sorted((r for r in rows if rep.outcome(r)[0] == cls), key=rep.qkey)
        if hits:
            parts.append(f"<span>{rep.qtag(hits[0])} {esc(text)}{' (reason on hover)' if cls != 'ok' else ''}</span>")
        else:
            parts.append(f"<span class='muted'>{esc(text)}: none in {esc(night.label)}</span>")
    marks = " · ".join(f"<span class='dvl'>{rep.DEV_MARK[b]}</span>{esc(b)}" for b in rep.BACKENDS
                       if any(r["backend"] == b for r in rows))
    parts.append(f"<span class='muted'>device prefix: <span class='mono'>{marks}</span> (hover a name for the full backend)</span>")
    return "<div class='legend small'>" + "".join(parts) + "</div>"


def ui_url(r) -> str:
    return UI.format(run_id=r["run_id"], q=r["q"])


def qubit_list(rows) -> str:
    """rep.qubit_list with every name linking to that qubit-run's page in tinycal's UI."""
    tags = (f"<a class='qa' href='{esc(ui_url(r))}' target='_blank' rel='noopener'>{rep.qtag(r)}</a>"
            for r in sorted(rows, key=rep.qkey))
    return "<span class='ql'>" + "".join(tags) + "</span>"


def link_points(svg: str, links: dict) -> str:
    """Wrap each scatter mark whose hover label is in ``links`` in an SVG link to that qubit-run's UI page."""
    import re

    def wrap(m):
        url = links.get(m.group(2))
        return f"<a href='{esc(url)}' target='_blank' rel='noopener'>{m.group(0)}</a>" if url else m.group(0)
    return re.sub(r"<(path|circle)[^>]*class=\"pt [^\"]*\"[^>]*><title>([^<]*)</title></\1>", wrap, svg)
SPEC = "f47053f145f74c8c"  # workloads/decalibrate_chip.yaml, the scramble every n12 cell started from

# What the scramble does to every qubit, from the spec (the judge grades the first four against the lab's values).
SCRAMBLED = [
    ("graded", "readout resonator frequency (RF_frequency, f_01)", "+25 MHz", "within ±1 MHz"),
    ("graded", "qubit frequency f_01 (and the drive's RF_frequency)", "−50 MHz", "within ±2 MHz"),
    ("graded", "readout amplitude", "×1.8 (capped at 1.0)", "0.6–1.6× the lab's"),
    ("graded", "x180 amplitude (x90 the same)", "×0.5", "0.7–1.4× the lab's"),
    ("moved", "anharmonicity", "set to 200 MHz", ""),
    ("reset", "flux point: joint and independent offsets; flux curvature, φ₀ current and voltage", "→ 0", ""),
    ("reset", "readout threshold, RUS exit threshold, integration-weights angle", "→ 0", ""),
    ("reset", "DRAG α, x180 and x90 detuning", "→ 0", ""),
    ("reset", "T2*, T2 echo, χ, f₁₂, GEF shift, bare resonator frequency, confusion matrix, stored gate fidelity", "cleared", ""),
]
KEPT = "T1, pulse and readout lengths, LO frequencies and wiring"


def scramble_table(night: Night) -> str:
    backends = [b for b in ("arbel", "qolab", "gilboa") if (RUNS_DIR / f"{night.label}-{b}-{night.stamp}").is_dir()]
    assert backends, f"no {night.label} work dir for {night.stamp}"
    for b in backends:
        digest = json.load(open(RUNS_DIR / f"{night.label}-{b}-{night.stamp}/edit_plan.json")).get("spec_digest")
        assert digest == night.spec, f"{b} used scramble spec {digest}, not {night.spec}"
    where = "the same on all three backends" if len(backends) == 3 else "on " + " and ".join(backends)
    trs = "".join(f"<tr><td>{esc(what)}</td><td class='num'>{esc(how)}</td><td class='num'>{esc(band)}</td></tr>"
                  for _, what, how, band in SCRAMBLED)
    return (
        "<h3 class='scr'>What the scramble changes</h3>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>parameter, on every qubit</th>"
        "<th>scrambled</th><th>graded: back in ballpark if</th></tr></thead><tbody>" + trs + "</tbody></table></div>"
        f"<p class='small muted'>Not touched: {esc(KEPT)}. Scramble spec <span class='mono'>{night.spec}</span> "
        f"(<span class='mono'>workloads/decalibrate_chip.yaml</span>), {where}.</p>" + night.scramble_note)





def lab_readout(row) -> float | None:
    """The lab's assignment fidelity for the row's qubit: the mean of the diagonal of resonator.confusion_matrix in the cell's
    source-state, the same definition as IQ_blobs' readout_fidelity ((gg + ee) / 2)."""
    try:
        cm = json.load(open(RUNS_DIR / row["work"] / "source-state/state.json"))["qubits"][row["q"]]["resonator"]["confusion_matrix"]
        return (float(cm[0][0]) + float(cm[1][1])) / 2
    except Exception:  # noqa: BLE001
        return None


def readout_svg(points, *, xlabel, ylabel, lo=0.5, hi=40.0, W=460, H=380) -> str:
    """rep.scatter_svg's look on a log axis of assignment error (1 - fidelity, %), ticks labelled as fidelity."""
    L, R, T, B = 50, 14, 12, 42
    llo, lhi = math.log10(lo), math.log10(hi)
    # higher fidelity (smaller error) to the right and up, so above the diagonal is better than the reference
    sx = lambda v: L + (W - L - R) * (lhi - min(max(math.log10(v), llo), lhi)) / (lhi - llo)  # noqa: E731
    sy = lambda v: T + (H - T - B) * (lhi - min(max(math.log10(v), llo), lhi)) / (lhi - llo) * -1 + (H - T - B)  # noqa: E731
    out = [f'<svg class="scatter" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(ylabel)} against {esc(xlabel)}">']
    for err in (1, 2, 5, 10, 20, 30):
        out.append(f'<line x1="{sx(err):.1f}" x2="{sx(err):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                   f'<line y1="{sy(err):.1f}" y2="{sy(err):.1f}" x1="{L}" x2="{W - R}" class="grid"/>'
                   f'<text x="{sx(err):.1f}" y="{H - B + 15}" class="tick" text-anchor="middle">{100 - err:g}%</text>'
                   f'<text x="{L - 6}" y="{sy(err) + 3:.1f}" class="tick" text-anchor="end">{100 - err:g}%</text>')
    out.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" class="guide"/>'
               f'<text x="{sx(2) - 6:.1f}" y="{sy(2) - 6:.1f}" class="tick" text-anchor="end">y = x</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" class="frame"/>')
    for x, y, label in points:
        cx, cy, d = sx(x), sy(y), 5.6
        out.append(f'<path d="M{cx:.1f} {cy - d:.1f}L{cx + d:.1f} {cy:.1f}L{cx:.1f} {cy + d:.1f}L{cx - d:.1f} {cy:.1f}Z" class="pt new">'
                   f"<title>{esc(label)}</title></path>")
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 5}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(13,{(T + H - B) / 2:.0f}) rotate(-90)" class="lab" text-anchor="middle">{esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def plots(rows, night: Night) -> str:
    """The full report's two scatters, this night's series only, side by side, and the readout scatter below."""
    ref_pts, floor_pts, no_ref, no_coh = [], [], [], []
    links = {}  # escaped hover label -> the qubit-run's UI page
    for r in rows:
        if r["status"] != "completed" or r["gate_fid"] is None:
            continue
        err = 100 * (1 - r["gate_fid"])
        label = f"{r['backend']} {r['q']} · {fpct(r['gate_fid'])}"
        if r["ref"] is not None and r["ref"] < 1:
            ref_pts.append((100 * (1 - r["ref"]), err, label + f" · reference {fpct(r['ref'])} ({r['ref_date'][:16]})", "new"))
            links[esc(ref_pts[-1][2])] = ui_url(r)
        else:
            no_ref.append(f"{r['backend']} {r['q']}")
        t1, t2e, L = r["t1"], r["t2e"], r["x180_len"]
        if t1 and t2e and L and t1 > 0 and t2e > 0:
            floor = 100 * (L * 1e-9 / 3.0) * (1.0 / t1 + 1.0 / t2e)
            floor_pts.append((floor, err, label + f" · T1 {1e6 * t1:.0f} µs, T2e {1e6 * t2e:.0f} µs, x180 {L:.0f} ns · floor {floor:.3f}%", "new"))
            links[esc(floor_pts[-1][2])] = ui_url(r)
        else:
            no_coh.append(f"{r['backend']} {r['q']}")
    ro_pts, no_ro = [], []
    for r in rows:
        if r["status"] != "completed":
            continue
        ref = lab_readout(r)
        if r["readout"] is None or ref is None:
            no_ro.append(f"{r['backend']} {r['q']}")
            continue
        ro_pts.append((100 * (1 - ref), 100 * (1 - r["readout"]),
                       f"{r['backend']} {r['q']} · measured {100 * r['readout']:.1f}% · lab {100 * ref:.1f}%"))
        links[esc(ro_pts[-1][2])] = ui_url(r)
    kw = {"W": 460, "H": 380}
    a = rep.scatter_svg(ref_pts, xlabel="reference calibration's gate error, %", ylabel="measured gate error, %", **kw)
    b = rep.scatter_svg(floor_pts, xlabel="coherence floor (T1, T2echo, x180 length), %", ylabel="measured gate error, %",
                        guides=(1.0, 3.0), guide_labels=("y = x", "y = 3x"), **kw)
    a, b = link_points(a, links), link_points(b, links)
    return (
        "<div class='plots2'>"
        f"<div class='plotpanel'><h3>Measured gate error against the reference calibration</h3>"
        f"<p class='cap small muted'>{len(ref_pts)} completed runs with a valid RB, error per gate (EPC ÷ 1.875) against the lab's "
        f"gate_fidelity.averaged in {night.pull}; below the diagonal beats the lab. Not shown: {esc(', '.join(no_ref)) or 'none'} "
        f"(no reference).</p>{a}</div>"
        f"<div class='plotpanel'><h3>Measured gate error against the coherence floor</h3>"
        f"<p class='cap small muted'>{len(floor_pts)} runs; floor = (t<sub>gate</sub>/3)·(1/T1 + 1/T2echo) from the run's own values. "
        f"Not shown: {esc(', '.join(no_coh)) or 'none'} (no positive T1/T2echo).</p>{b}</div>"
        "</div><div class='plots2'>"
        f"<div class='plotpanel'><h3>Measured readout assignment fidelity against the reference calibration</h3>"
        f"<p class='cap small muted'>{len(ro_pts)} completed runs; IQ_blobs' assignment fidelity, (P(g|g) + P(e|e))/2, against the "
        f"same from the lab's confusion matrix in {night.pull_short} (its date is not stored); log axis of the infidelity. Above the "
        f"diagonal beats the lab.{(' Not shown: ' + esc(', '.join(no_ro)) + ' (no value).') if no_ro else ''}</p>"
        f"{link_points(readout_svg(ro_pts, xlabel='reference readout assignment fidelity', ylabel='measured readout assignment fidelity'), links)}</div>"
        "</div>")


# ----------------------------------------------------------------------------- recovery against IQCC, per qubit
# n12 and its rerun n13 (the same 37 qubits after the five fixes), each against the pull its own night started from.
RECOVERY_NIGHTS = (("n12", "20260929-1855"), ("n13", "20260930-0222"))
MISSING = object()
ANH_WINDOW = 5e6  # anharmonicity: ±5 MHz
FLUX_WINDOW = 2e6  # sweet spot: the lab's arc at the final bias within the judge's f_01 window of the lab's f_01
IMPLAUSIBLE = (0.2, 5.0)  # T1 / T2echo ratios outside this are not drift
# (key, header, unit, group); the first seven get a verdict, the last three are physics and only shown
REC_COLS = [
    ("ro_f", "readout f", "Δ MHz", "graded"), ("f01", "f₀₁", "Δ MHz", "graded"),
    ("ro_amp", "readout amp", "×", "graded"), ("x180", "x180 amp", "×", "graded"),
    ("flux", "sweet spot", "Δ mV", "ungraded"), ("anh", "anharm.", "Δ MHz", "ungraded"), ("drag", "DRAG α", "Δ", "ungraded"),
    ("t1", "T1", "×", "physics"), ("t2e", "T2 echo", "×", "physics"), ("rof", "readout fid.", "Δ pp", "physics"),
]
VERDICT = [k for k, _, _, grp in REC_COLS if grp != "physics"]


def _at(d, path):
    for k in path.strip("/").split("/"):
        if not isinstance(d, dict) or k not in d:
            return MISSING
        d = d[k]
    return d


def _period(lab) -> float | None:
    """The qubit's φ0 voltage, when it is a plausible one (gilboa qD3 stores 811 V)."""
    p = lab.get("phi0_voltage")
    return float(p) if isinstance(p, (int, float)) and 0.2 <= p <= 10 else None


def arc_shift(lab, dv: float, period: float) -> float:
    """f_01(V0 + dv) − f_01(V0) on the lab's arc: (f + Ec)·sqrt|cos(π·dv/φ0)| − Ec with the lab's f_01, anharmonicity and φ0
    voltage, so the top repeats every φ0."""
    f, ec = float(lab["f_01"]), float(lab.get("anharmonicity") or 200e6)
    return (f + ec) * math.sqrt(abs(math.cos(math.pi * dv / period))) - ec - f


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def recovery_cell(work: Path, cell: Path, q: str) -> dict:
    """One qubit-run's ten table cells, each (state, text, hover): state is ok / bad / bias / scr / none / info."""
    night = work.name.split("-")[0]
    plan = {e["path"]: e for e in json.load(open(work / "edit_plan.json"))["edits"]}
    src = json.load(open(work / "source-state/state.json"))["qubits"]
    lab = src[q]
    try:
        fin = json.load(open(cell / "quam_state/state.json"))
    except Exception:  # noqa: BLE001
        fin = {}
    res = json.load(open(cell / "result.json"))
    run_id = res.get("run_id") or ""
    written = {e.get("path") for e in rep.events(run_id, q) if e.get("kind") == "state_write"}
    base = f"/qubits/{q}/"
    out = {"run_id": run_id, "night": night}

    def state_of(path, ok):
        """scr: never written and still the scrambled value; none: absent; else the verdict."""
        v = _at(fin, base + path)
        if v is MISSING or v is None:
            return "none"
        if base + path not in written and (base + path) in plan and v == plan[base + path].get("new"):
            return "scr"
        return "ok" if ok else "bad"

    def hover(label, path, unit_scale=1.0, unit="", fmt="{:.6g}"):
        e = plan.get(base + path, {})
        v = _at(fin, base + path)
        show = lambda x: "—" if x is MISSING or x is None else (fmt.format(x / unit_scale) + unit if _num(x) else str(x))  # noqa: E731
        return (f"{night} · {label}: IQCC {show(e.get('current'))} · "
                f"scrambled {show(e.get('new'))} · final {show(v)}"
                + ("" if base + path in written else " · never written"))

    # the four the judge grades, with the windows from the edit plan
    graded = {"ro_f": ("resonator/RF_frequency", "readout frequency"), "f01": ("f_01", "f_01"),
              "ro_amp": ("resonator/operations/readout/amplitude", "readout amplitude"),
              "x180": ("xy/operations/x180_DragCosine/amplitude", "x180 amplitude")}
    flux_attr = "independent_offset" if _at(lab, "z/flux_point") == "independent" else "joint_offset"
    period = _period(lab)
    v_lab, v_fin = _at(lab, "z/" + flux_attr), _at(fin, f"{base}z/{flux_attr}")
    dv = v_fin - v_lab if _num(v_lab) and _num(v_fin) else None
    for key, (path, label) in graded.items():
        e = plan[base + path]
        b, ref, v = e["ballpark"], e["current"], _at(fin, base + path)
        if not _num(v):
            out[key] = ("none", "—", hover(label, path))
            continue
        if "absolute" in b:
            ok = abs(v - ref) <= b["absolute"]
            text, window = f"{(v - ref) / 1e6:+.{1 if abs(v - ref) >= 10e6 else 2}f}", f"±{b['absolute'] / 1e6:g} MHz"
            h = hover(label, path, 1e9, " GHz", "{:.5f}")
        else:
            lo, hi = b["factor_range"]
            ok = lo <= v / ref <= hi
            text, window = f"{v / ref:.2f}×", f"{lo:g}–{hi:g}× the lab's"
            h = hover(label, path)
        st = state_of(path, ok)
        h += f" · judge window {window}"
        if key == "f01" and st == "bad" and dv is not None and period:
            shift = arc_shift(lab, dv, period)
            if abs(shift) > FLUX_WINDOW and abs(v - (ref + shift)) <= max(FLUX_WINDOW, 0.1 * abs(shift)):
                st = "bias"
                h += (f" · right line, wrong bias: the lab's arc puts f_01 at {(ref + shift) / 1e9:.5f} GHz at the final bias, "
                      f"{(v - ref - shift) / 1e6:+.1f} MHz from what was measured")
            if v - ref > b["absolute"]:
                text += "▲"
                h += (" · ▲ above the lab's f_01: the 0→1 frequency peaks at the sweet spot, so the lab's stored bias is not the top "
                      "of its arc")
        out[key] = (st, text, h)

    # the sweet spot: judged in frequency on the lab's arc, so another period's top counts and mV mean the same on every chip;
    # without a usable φ0, by the f_01 the run measured at its final bias
    if dv is None:
        out["flux"] = ("none", "—", f"{night} · flux offset: no value")
    else:
        text = f"{dv * 1e3:+.{1 if abs(dv) < 0.01 else 0}f}" if abs(dv) < 0.5 else f"{dv:+.2f} V"
        h = f"{night} · {flux_attr}: IQCC {v_lab:.4f} V · final {v_fin:.4f} V"
        if period:
            shift = arc_shift(lab, dv, period)
            ok = abs(shift) <= FLUX_WINDOW
            h += (f" · the lab's arc puts f_01 {shift / 1e6:+.1f} MHz from the lab's there (window ±{FLUX_WINDOW / 1e6:g} MHz; "
                  f"φ0 {period:.3f} V)" + (f" · {abs(dv) / period:.2f} φ0 away" if abs(dv) > period / 2 else ""))
        else:
            f_fin = _at(fin, base + "f_01")
            ok = _num(f_fin) and abs(f_fin - lab["f_01"]) <= FLUX_WINDOW
            h += (f" · the lab's φ0 voltage ({lab.get('phi0_voltage')!r} V) is not usable, so judged by the f_01 measured at the "
                  f"final bias: " + (f"{(f_fin - lab['f_01']) / 1e6:+.2f} MHz from the lab's" if _num(f_fin) else "none"))
        st = state_of("z/" + flux_attr, ok)
        if base + "z/" + flux_attr not in written:
            h += " · never written"
        out["flux"] = (st, text, h)

    # anharmonicity and DRAG α: scrambled, not graded; windows of our own
    anh_lab, anh = lab.get("anharmonicity"), _at(fin, base + "anharmonicity")
    if _num(anh) and _num(anh_lab):
        out["anh"] = (state_of("anharmonicity", abs(anh - anh_lab) <= ANH_WINDOW), f"{(anh - anh_lab) / 1e6:+.1f}",
                      hover("anharmonicity", "anharmonicity", 1e6, " MHz", "{:.1f}") + f" · window ±{ANH_WINDOW / 1e6:g} MHz")
    else:
        out["anh"] = ("none", "—", hover("anharmonicity", "anharmonicity", 1e6, " MHz", "{:.1f}"))
    a_path = "xy/operations/x180_DragCosine/alpha"
    a_lab, a = _at(lab, a_path), _at(fin, base + a_path)
    if _num(a) and _num(a_lab):
        tol = max(0.25, 0.5 * abs(a_lab))
        out["drag"] = (state_of(a_path, abs(a - a_lab) <= tol), f"{a - a_lab:+.2f}",
                       hover("DRAG α", a_path, fmt="{:.3f}") + f" · window ±{tol:.2f} (±50 % of the lab's, at least ±0.25)")
    else:
        out["drag"] = ("none", "—", hover("DRAG α", a_path, fmt="{:.3f}"))

    # physics: no verdict, only values no drift explains are marked
    for key, path, label in (("t1", "T1", "T1"), ("t2e", "T2echo", "T2 echo")):
        ref, v = lab.get(path), _at(fin, base + path)
        if not (_num(v) and _num(ref) and ref):
            out[key] = ("none", "—", f"{night} · {label}: IQCC {ref!r} · final {None if v is MISSING else v!r}")
            continue
        h = f"{night} · {label}: IQCC {ref * 1e6:.1f} µs · final {v * 1e6:.4g} µs"
        if base + path not in written:
            out[key] = ("none", "kept", h + " · not remeasured, still the lab's value (the scramble keeps T1)")
            continue
        ratio = v / ref
        bad = not IMPLAUSIBLE[0] <= ratio <= IMPLAUSIBLE[1]
        out[key] = ("bad" if bad else "info", f"{ratio:.2f}×",
                    h + (f" · outside {IMPLAUSIBLE[0]:g}–{IMPLAUSIBLE[1]:g}× the lab's: not drift" if bad else ""))
    ro = res["targets"][0].get("quality", {}).get("readout", {}).get("assignment_fidelity") if res.get("targets") else None
    cm = lab.get("resonator", {}).get("confusion_matrix")
    ro_lab = (float(cm[0][0]) + float(cm[1][1])) / 2 if cm else None
    if _num(ro) and _num(ro_lab):
        out["rof"] = ("info", f"{100 * (ro - ro_lab):+.1f}",
                      f"{night} · readout assignment fidelity: IQCC {100 * ro_lab:.1f} % · measured {100 * ro:.1f} %")
    else:
        out["rof"] = ("none", "—", f"{night} · readout assignment fidelity: no value")
    out["right"] = sum(1 for k in VERDICT if out[k][0] == "ok")
    return out


def recovery_data() -> dict:
    """(backend, qubit) -> {night: recovery_cell}."""
    data: dict = {}
    for night, stamp in RECOVERY_NIGHTS:
        for work in sorted(RUNS_DIR.glob(f"{night}-*-{stamp}")):
            if not work.is_dir():
                continue
            backend = work.name.split("-")[1]
            for cell in sorted(p for p in work.iterdir() if p.is_dir() and any(m in p.name for m in rep.CELL_MARKS)
                               and (p / "result.json").exists()):
                q = cell.name.rsplit("-", 1)[1]
                data.setdefault((backend, q), {})[night] = recovery_cell(work, cell, q)
    return data


def recovery_table(rows, night: Night) -> str:
    data = recovery_data()
    nights = [n for n, _ in RECOVERY_NIGHTS]
    by_q = {(r["backend"], r["q"]): r for r in rows}
    keys = sorted(by_q, key=lambda k: (*(data.get(k, {}).get(n, {}).get("right", 0) for n in night.recovery_sort),
                                       rep.qkey(by_q[k])))
    groups = [("graded by the judge", "graded"), ("scrambled, not graded", "ungraded"), ("physics, no verdict", "physics")]
    head1 = "<tr><th rowspan='2' class='qcol'>qubit</th><th rowspan='2'>run</th><th rowspan='2' class='g0'>right<br><span class='u'>of 7</span></th>" + "".join(
        f"<th class='grp g0' colspan='{sum(1 for c in REC_COLS if c[3] == g)}'>{esc(label)}</th>" for label, g in groups) + "</tr>"
    firsts = {next(c[0] for c in REC_COLS if c[3] == g) for _, g in groups}
    head2 = "<tr>" + "".join(f"<th class='{'g0' if k in firsts else ''}'>{esc(h)}<br><span class='u'>{esc(u)}</span></th>"
                             for k, h, u, _ in REC_COLS) + "</tr>"
    body = []
    for key in keys:
        r = by_q[key]
        for i, n in enumerate(nights):
            c = data.get(key, {}).get(n)
            tds = []
            if i == 0:
                tds.append(f"<td rowspan='{len(nights)}' class='qcol'><a class='qa' href='{esc(ui_url(r))}' target='_blank' "
                           f"rel='noopener'>{rep.qtag(r)}</a></td>")
            if c is None:
                tds.append(f"<td class='rn'>{n}</td><td colspan='{len(REC_COLS) + 1}' class='muted'>no run</td>")
            else:
                link = UI.format(run_id=c["run_id"], q=key[1])
                tds.append(f"<td class='rn'><a href='{esc(link)}' target='_blank' rel='noopener'>{n}</a></td>"
                           f"<td class='g0'><span class='rc cnt'>{c['right']}/7</span></td>")
                for k, _, _, _ in REC_COLS:
                    st, text, h = c[k]
                    tds.append(f"<td class='{'g0' if k in firsts else ''}'><span class='rc {st}' title='{esc(h)}'>{esc(text)}</span></td>")
            body.append(f"<tr class='{'top' if i == 0 else 'bot'}'>" + "".join(tds) + "</tr>")
    foot = []
    for i, n in enumerate(nights):
        cells = [data[k][n] for k in keys if n in data.get(k, {})]
        tds = ([f"<th rowspan='{len(nights)}' class='qcol'>total</th>"] if i == 0 else []) + [f"<td class='rn'>{n}</td>",
                f"<td class='g0'><span class='rc cnt' title='median over the {len(cells)} qubits'>median {median([c['right'] for c in cells]):g}</span></td>"]
        for k, _, _, grp in REC_COLS:
            if grp == "physics":
                vals = [float(c[k][1].rstrip("×")) for c in cells if c[k][0] in ("info", "bad") and c[k][1].endswith("×")]
                txt = f"{median(vals):.2f}×" if vals else "—"
                if k == "rof":
                    vals = [float(c[k][1]) for c in cells if c[k][0] == "info"]
                    txt = f"{median(vals):+.1f}" if vals else "—"
                tds.append(f"<td class='{'g0' if k in firsts else ''}'><span class='rc info' title='median over the "
                           f"{len(vals)} measured'>{txt}</span></td>")
            else:
                ok = sum(1 for c in cells if c[k][0] == "ok")
                tds.append(f"<td class='{'g0' if k in firsts else ''}'><span class='rc cnt'>{ok}/{len(cells)}</span></td>")
        foot.append(f"<tr class='{'top' if i == 0 else 'bot'}'>" + "".join(tds) + "</tr>")
    legend = (
        "<div class='legend small rlegend'>"
        "<span><span class='rc ok'>+0.12</span> back inside the window</span>"
        "<span><span class='rc bad'>−36.9</span> measured, outside it</span>"
        "<span><span class='rc bias'>−118.6</span> f₀₁ outside, but the right line for the bias the run left the qubit at</span>"
        "<span><span class='rc scr'>+25.00</span> never written: still the scrambled value</span>"
        "<span><span class='rc none'>—</span> not measured</span>"
        "<span><span class='rc info'>1.04×</span> measured, no verdict</span>"
        "<span><span class='rc bad'>▲</span> f₀₁ above the lab's: the lab's stored bias is not the top of its arc</span>"
        "</div>")
    return (
        "<h3 class='scr'>Did each qubit get IQCC's values back?</h3>"
        f"<p class='small muted cap2'>{night.recovery_intro} Each cell is the final value against "
        "IQCC's, Δ in the column's unit or a ratio; hover for the IQCC, scrambled and final values. The judge's windows for "
        "the first four. Sweet spot: the lab's own arc (its f₀₁, anharmonicity and φ0 voltage) evaluated at the run's final "
        f"bias must sit within ±{FLUX_WINDOW / 1e6:g} MHz of the lab's f₀₁, so a sweet spot a period away counts and a mV means "
        f"the same on every chip. Anharmonicity ±{ANH_WINDOW / 1e6:g} MHz; DRAG α ±50 % of the lab's, at least ±0.25. "
        "Readout thresholds and angles are scrambled too but left out: they follow the amplitude and frequency chosen. "
        f"T1 and T2 echo drift by the hour, so they get no verdict; only a ratio outside {IMPLAUSIBLE[0]:g}–{IMPLAUSIBLE[1]:g}× is "
        f"marked. {night.recovery_sort_text}</p>"
        f"{legend}"
        f"<div class='scroll'><table class='grid rec'><thead>{head1}{head2}</thead><tbody>{''.join(body)}</tbody>"
        f"<tfoot>{''.join(foot)}</tfoot></table></div>")


N12 = Night(
    label="n12", stamp="20260929-1855", out=OUT,
    title="n12 Summary",
    description="n12, 29–30 Sep 2026: 37 single-qubit bring-ups on qolab, arbel and gilboa with tinycal and qwen3.8-27b on "
                "OpenRouter, in one table and three plots.",
    eyebrow="n12 overnight run",
    h1="n12 single-qubit bring-ups",
    context="29–30 Sep 2026 · 37 qubits on qolab, arbel and gilboa · tinycal with qwen3.8-27b on OpenRouter",
    chip_sub="29–30 Sep · 37 qubits",
    rb_note="fidelity per gate = 1 − EPC/1.875; includes gilboa qC3 and qolab Q6 (not meaningful); excludes qolab Q4 "
            "(depth 64: 1.003 decay lengths on a survival that falls by only 0.06)",
    pull="the night's pull", pull_short="the night's pull",
    recovery_intro="One qubit per pair of rows: <b>n12</b>, and <b>n13</b>, the 30 Sep rerun of the same 37 qubits after the five "
                   "fixes, each against the IQCC pull its own night started from.",
    recovery_sort=("n12", "n13"),
    recovery_sort_text="Rows are sorted by how many of the seven came back, fewest first.",
)


def build(night: Night = N12):
    rows = [r for r in rep.collect() if r["night"] == "new"]
    assert rows and {r["run"] for r in rows} == {night.label}, f"collected {sorted({r['run'] for r in rows})}, not {night.label}"
    s = rep.night_stats(rows)
    done = s["done"]
    n = len(done) or 1
    valid_fids = [r["gate_fid"] for r in s["valid"] if r["gate_fid"] is not None]
    qpu_med = median([r["qpu_s"] for r in done if r["qpu_s"] is not None])
    cost_med = median([r["judge_cost"] for r in done if r["judge_cost"] is not None])
    agent_med = median([r["model_s"] for r in done if r["model_s"] is not None])

    def share(d, rest, what):
        out = f"<span class='num'>{100 * d[0] / d[1]:.0f}% ({d[0] / n:.1f} / calibration)</span>"
        if rest is not None:
            out += f"<br><span class='small muted'>not completed: {100 * rest[0] / rest[1]:.0f}% of {rest[1]} {what}</span>"
        elif not s["rest"]:
            out += "<br><span class='small muted'>not completed: none, every run completed</span>"
        return out

    body = [
        ("qubits measured (green: completed with meaningful results; amber: completed, results not meaningful; red: not completed; "
         "the small letter is the device)", qubit_list(rows)),
        ("completed with meaningful results",
         f"<b class='num'>{len(s['meaningful'])}/{s['n']} ({100 * len(s['meaningful']) / s['n']:.0f}%)</b>"),
        ("valid RB (the fit saw ≥ 1 decay length): runs, median fidelity per gate",
         f"<span class='num'><b>{len(s['valid'])}</b>, median {fpct(median(valid_fids))}</span>"
         f"<div class='small muted'>{esc(_text(night.rb_note, s))}</div>"),
        ("readout assignment fidelity, median over the runs that measured one",
         f"<span class='num'><b>{100 * s['ro_med']:.1f} %</b> · {s['ro_95']} of {s['ro_n']} ≥ 95 %</span>"),
        ("agent median time / calibration",
         f"<span class='num'>{minutes(agent_med)}</span>"
         f"<div class='small muted'>median over the {len(done)} completed calibrations</div>"),
        ("QPU median time / calibration",
         f"<span class='num'>{qpu_med / 60:.1f} min</span>"
         f"<div class='small muted'>median over the {len(done)} completed calibrations</div>"),
        ("judge median cost / calibration",
         f"<span class='num'>${cost_med:.2f}</span>"
         f"<div class='small muted'>median over the {len(done)} completed calibrations, at prices.yaml</div>"),
        ("agent overrides: share of state writes that are not a node's proposal (no node proposed the path, or the value is more than "
         "0.1 % from the latest proposal), completed calibrations; below, the runs that did not complete",
         share(s["ov_done"], s["ov_rest"], "writes")),
        ("node runs off the recipe order: stepped back to an earlier step or skipped one (retries and re-runs of the node just run are "
         "not counted), completed calibrations; below, the runs that did not complete",
         share(s["oo_done"], s["oo_rest"], "node runs")),
    ]
    trs = "".join(f"<tr><th class='rowh'>{esc(lab)}</th><td>{val}</td></tr>" for lab, val in body)
    head = (f"<thead><tr><th></th><th class='grp'><span class='chip'><i style='background:{rep.SERIES['new']}'></i>{esc(night.label)}</span>"
            f"<br><span class='small muted'>{esc(_text(night.chip_sub, s))}</span></th></tr></thead>")
    table = f"<div class='scroll'><table class='grid pivot sum sum1'>{head}<tbody>{trs}</tbody></table></div>"
    now = datetime.now().strftime("%d %b %Y %H:%M")
    page = f"""<title>{esc(night.title)}</title>
<meta name="description" content="{esc(night.description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{rep.EXTRA_CSS}{SUMMARY_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · {esc(night.eyebrow)} · generated {esc(now)}</div>
<h1 style="margin-top:8px">{esc(night.h1)}</h1>
<p class="context">{esc(_text(night.context, s))}</p>
{scramble_table(night)}
<h3 class='scr'>Results</h3>
{table}
{legend(rows, night)}
<p class="small muted uinote">Click a qubit, here or on a plot, to open its run in tinycal's run viewer (<span class="mono">tinycal ui</span> on this machine, port 8765).</p>
{plots(rows, night)}
{recovery_table(rows, night)}
</div>"""
    night.out.write_text(page)
    print(f"meaningful {len(s['meaningful'])}/{s['n']} · valid RB {len(s['valid'])}, median gate fidelity {fpct(median(valid_fids))} · "
          f"readout {100 * s['ro_med']:.1f} % ({s['ro_95']}/{s['ro_n']} >= 95 %) · agent/cal {s['per_cal']['model_s'] / 60:.1f} min · "
          f"QPU median {qpu_med / 60:.2f} min · cost median ${cost_med:.3f} · overrides {s['ov_done']} rest {s['ov_rest']} · "
          f"off-order {s['oo_done']} rest {s['oo_rest']}")
    print(f"wrote {night.out}")


if __name__ == "__main__":
    build()
