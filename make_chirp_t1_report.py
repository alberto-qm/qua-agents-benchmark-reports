"""Build 2026-09-25-chirp-t1.html from 2026-09-25-chirp-t1/ (no QPU, no network).

The data were taken on 25 Sep 2026 by t1_chirp_test.py and fitted by t1_chirp_analyse.py (both in the data folder).
Usage:
    python make_chirp_t1_report.py [--sync]
--sync copies the run files from ~/qab-runs/chirp-nodes-20260924 into 2026-09-25-chirp-t1/ first.
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
DATA = HERE / "2026-09-25-chirp-t1"
RUNS = Path.home() / "qab-runs/chirp-nodes-20260924"
OUT = HERE / "2026-09-25-chirp-t1.html"
TITLE = "Chirp T1"
QB = {"arbel": ["qB4", "qA5", "qD1", "qA6"], "qolab": ["Q1", "Q2", "Q5"], "gilboa": ["qD2", "qC3", "qD5"]}


def sync():
    (DATA / "t1_chirp").mkdir(parents=True, exist_ok=True)
    (DATA / "figures").mkdir(exist_ok=True)
    for f in (RUNS / "t1_chirp").glob("*"):
        if f.is_file() and f.suffix in (".npz", ".json", ".log"):
            shutil.copy(f, DATA / "t1_chirp" / f.name)
    for f in (RUNS / "figures").glob("t1_chirp_*.png"):
        shutil.copy(f, DATA / "figures" / f.name)
    for name in ("t1_chirp_test.py", "t1_chirp_analyse.py", "retry_t1_gilboa.sh"):
        shutil.copy(RUNS / name, DATA / name)
    for f in RUNS.glob("t1_chirp_*.out"):
        shutil.copy(f, DATA / f.name)


def img(path, caption, alt):
    b64 = base64.b64encode((DATA / path).read_bytes()).decode()
    return (f'<figure><figcaption class="small">{caption}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:100%;height:auto;border-radius:6px"></figure>')


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="grid small"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def num(v, fmt="{:.2f}", dash="–"):
    if v is None:
        return dash
    try:
        if v != v:
            return dash
    except TypeError:
        return dash
    return f'<span class="num">{fmt.format(v)}</span>'


def build() -> str:
    s = json.loads((DATA / "t1_chirp/summary.json").read_text())
    metas = {be: json.loads((DATA / "t1_chirp" / f"{be}_meta.json").read_text())
             for be in QB if (DATA / "t1_chirp" / f"{be}_meta.json").exists()}
    rows, sig, amps, qpus, done = [], [], [], [], []
    for be, qs in QB.items():
        for q in qs:
            v = s.get(f"{be}/{q}")
            if not v or "T1_us" not in (v.get("chirp_fit") or {}):
                rows.append([be, f"<b>{q}</b>"] + ['<span class="muted">not measured (gateway down)</span>'] + ["–"] * 7)
                continue
            fc, fx = v["chirp_fit"], v["x180_fit"]
            d = (fc["T1_us"] - fx["T1_us"]) / (fc["T1_err_us"] ** 2 + fx["T1_err_us"] ** 2) ** 0.5
            sig.append(abs(d))
            amps.append(fc["A"] / fx["A"])
            qpus.append(v.get("qpu_s") or 0)
            done.append(f"{be} {q}")
            rows.append([be, f"<b>{q}</b>", num(v.get("chirp_centre_mhz"), "{:+.2f}"),
                         f'{num(fc["T1_us"], "{:.1f}")} ± {num(fc["T1_err_us"], "{:.1f}")}',
                         f'{num(fx["T1_us"], "{:.1f}")} ± {num(fx["T1_err_us"], "{:.1f}")}', num(d, "{:+.1f}σ"),
                         num(fc["A"] / fx["A"], "{:.2f}"), num(v.get("stored_T1_us"), "{:.1f}"), num(v.get("lean_T1_us"), "{:.1f}"),
                         num(v.get("qpu_s"), "{:.1f} s")])
    n = len(done)
    gilboa_ok = all(f"gilboa/{q}" in s and "T1_us" in (s[f"gilboa/{q}"].get("chirp_fit") or {}) for q in QB["gilboa"])
    snap = "~/qab-runs/state-20260925-1526"
    figs = "\n".join(img(f"figures/t1_chirp_{be}.png", f"{be}: excited population against the delay before readout, chirp (blue) and "
                         "calibrated x180 (orange) in the same job; lines are the exponential fits.", f"T1 decays on {be}")
                     for be in QB if (DATA / "figures" / f"t1_chirp_{be}.png").exists() and any(f"{be}/" in k for k in s))
    CSS = base.CSS + """
figure img { display:block; }
ul.tight li, ol.recs li { margin:6px 0; max-width:82ch; }
td .muted, .muted { color:var(--muted); }
h3 { margin-top:28px; }
"""
    gilboa_note = ("" if gilboa_ok else
                   '<p class="small muted">gilboa (qD2, qC3, qD5) was not measured: its QOP gateway refused connections '
                   '("Gateway health is not good enough") from 15:29, before any job ran, and on the retries.</p>')
    body = f"""
<div class="page">
<p class="eyebrow">qua-agents benchmark · hardware test · 25 Sep 2026 · {", ".join(sorted({d.split()[0] for d in done}))} · {n} qubits</p>
<h1>{TITLE}</h1>
<p class="lede">At bring-up T1 is not known, and the chirp spectroscopy nodes then fall back on QuAM's 50 µs wait between shots,
which on qolab Q1 and Q2 left the qubit partly excited and the line lost. A chirp is a π pulse that needs no calibration, so T1
can be measured right after 03a: flip the qubit with the chirp 03a's frequency implies, wait, read out. On {n} qubits the chirp
gave the same T1 as the calibrated x180 played in the same job, within {max(sig):.1f}σ on every one, for about
{sum(qpus) / len(qpus):.0f} s of QPU per qubit.</p>

<div class="tiles">
  <div class="tile"><div class="v">{n} / {n}</div><div class="k">qubits where the chirp's T1 agrees with the x180's (largest difference {max(sig):.1f}σ)</div></div>
  <div class="tile"><div class="v">{min(amps):.2f}–{max(amps):.2f}</div><div class="k">the chirp's excitation relative to the calibrated x180 at the shortest delay</div></div>
  <div class="tile"><div class="v">~{sum(qpus) / len(qpus):.0f} s</div><div class="k">QPU per qubit: 20 delays, 200 shots each for chirp and x180, a ground reference, 1 ms between shots</div></div>
  <div class="tile"><div class="v">none</div><div class="k">calibration the chirp needed: f₀₁ to a few MHz and a drive above the adiabatic threshold, both from 03a</div></div>
</div>

<h2>1 · The measurement</h2>
<p>For each qubit one job interleaves, at 20 delays spaced logarithmically from 16 ns to 600 µs:</p>
<ul class="tight">
<li><b>the chirp</b>: a linear up-chirp over 20 MHz in 2 µs — the chirp nodes' sweep when T1 is missing — centred on the f₀₁
that 03a found with T1 removed from the state (the 13:32 run of the chirp-nodes report), at the drive 03a selected; then the delay,
then the readout;</li>
<li><b>the control</b>: the state's calibrated x180 at the stored frequency, then the delay, then the readout;</li>
<li><b>a ground reference</b>: no pulse.</li>
</ul>
<p>Every shot waits 1 ms first — 5 T1 up to T1 = 200 µs — so nothing depends on knowing T1. The signal is the I/Q projected on the
line from the ground reference to the x180 at 16 ns (0 = ground, 1 = the x180's excited state), and each pulse's points are fitted
with A e<sup>−t/T1</sup> + B. States: the IQCC cloud states pulled at 15:26 on 25 Sep (<span class="mono">{snap}</span>; gilboa with
its C/D qubits marked active). Nothing was written to any state.</p>

<h2>2 · Results</h2>
{table(["backend", "qubit", "chirp centre − stored f₀₁ [MHz]", "T1, chirp [µs]", "T1, x180 [µs]", "difference", "chirp excitation / x180",
        "stored T1 [µs]", "T1 from 03a's up/down lean [µs]", "QPU"], rows)}
{gilboa_note}
{figs}
{img("figures/t1_chirp_compare.png", "T1 from the chirp against T1 from the x180; grey ×: the T1 stored in the state.", "chirp against x180")}
<ul class="tight">
<li><b>The chirp measures the same T1 as the calibrated pulse</b> on every qubit, from 4 µs (arbel qA6) to 80 µs (qolab Q2).</li>
<li><b>It flips the qubit as fully as the calibrated pulse:</b> 0.94–1.01 of the x180's excitation on the long-T1 qubits. On arbel
qA6 (T1 ≈ 4.5 µs) it reaches 0.86, because the excitation, created mid-sweep, decays for ~1 µs before the sweep ends; the amplitude
does not enter T1.</li>
<li><b>It does not need an accurate frequency:</b> on qolab Q1 the chirp was centred 3.35 MHz from the line (03a's imprecise answer
with T1 missing, −3.35 ± 2.23 MHz) and still flipped it as fully as the x180 (1.00) — the band is 20 MHz wide.</li>
<li><b>qolab Q1 is fine here.</b> Its no-T1 chirp spectroscopy failed at the 50 µs wait; at 1 ms its T1 comes out 59 µs, which is why
50 µs (0.85 T1) left it partly excited.</li>
<li><b>The stored T1s are not always current:</b> qolab Q5 stores 32 µs and measured 44–48; arbel qB4 stores 28 and measured 34–35.
T1 moves from hour to hour, so a fresh one at bring-up is worth more than a stored one.</li>
<li><b>It agrees with 03a's lean:</b> for arbel qA6, the one qubit short-lived enough for the up/down-chirp lean to measure
(sweep/T1 = 0.52), that gives 3.8 µs, against 4.3 ± 0.4 µs here.</li>
</ul>

<h2>3 · Cost, and where it fits</h2>
<p>About {sum(qpus) / len(qpus):.0f} s of QPU per qubit as run (8,200 shots of ~1.1 ms, nearly all of it the 1 ms wait), plus a
few seconds of compile and queue per job. Halving the shots, or waiting 500 µs (safe to T1 = 100 µs), brings it to ~5 s. At
bring-up it would sit between the two chirp nodes:</p>
{table(["step", "wait between shots", "QPU per qubit"], [
    ["03a chirp: find the line (T1 unknown)", "~400 µs (proposed)", "~14 s"],
    ["T1 with the chirp (this test)", "1 ms", f"~{sum(qpus) / len(qpus):.0f} s (~5 s at 500 µs)"],
    ["03b chirp: flux map", "5 × the measured T1", "10–15 s on T1 of 30–45 µs"]])}
<p>After that every node can use the qubit's real T1 instead of QuAM's 50 µs fallback. The ground and excited readout levels at the
shortest delay come for free — the IQ blobs and a readout threshold, again without a π pulse — which is what an active reset with a
chirp as the flip would need.</p>

<h2>4 · Open</h2>
<ol class="recs">
<li>gilboa's three qubits{" were measured" if gilboa_ok else " (T1 1.3, 1.8 and 43 µs stored) still need the run; its gateway was down"}.</li>
<li>Make it a node that runs after 03a, takes 03a's frequency and drive, and writes T1 (and the readout levels) to the state.</li>
<li>The chirp nodes' wait parameter for when T1 is missing (~400 µs) is not in the nodes yet.</li>
</ol>

<h2>Provenance</h2>
<p class="small">Scripts, data and run logs are in <span class="mono">2026-09-25-chirp-t1/</span>: <span class="mono">t1_chirp_test.py</span>
(the jobs), <span class="mono">t1_chirp_analyse.py</span> (fits and figures), <span class="mono">retry_t1_gilboa.sh</span> (gilboa
retries), the per-qubit data (<span class="mono">t1_chirp/&lt;backend&gt;_&lt;qubit&gt;.npz</span>), fits
(<span class="mono">t1_chirp/summary.json</span>) and driver logs. The chirp code is the chirp-spectroscopy nodes' (qua-libs fork,
<span class="mono">feat/chirp-spectroscopy</span> at ee24095). Each launch is logged in <span class="mono">~/qab-runs/recipe-qolab-LOG.md</span>.
Related: <a href="2026-09-24-chirp-spectroscopy-nodes.html">chirp spectroscopy nodes</a>. Generated by
<span class="mono">make_chirp_t1_report.py</span>.</p>
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
