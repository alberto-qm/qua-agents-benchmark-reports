"""Build 2026-09-23-spectroscopy-chirp-vs-saturation.html from 2026-09-23-spectroscopy/ (no QPU, no network).

The measurements were taken on 23 Sep 2026 on qolab Q1, arbel qB4 and gilboa qD2 by the scripts in that folder;
their data, figures and run logs sit beside them. The night-4 node timing comes from night4_node_qpu.json, computed
from the tinycal event logs of the 20-22 Sep OpenRouter runs. Usage:
    python make_spectroscopy_report.py [--artifact PATH]   # PATH: body-only copy for publishing
"""
from __future__ import annotations

import base64
import json
import sys
from html import escape as esc
from pathlib import Path

import make_fwcmp2_report as base

HERE = Path(__file__).parent
DATA = HERE / "2026-09-23-spectroscopy"
OUT = HERE / "2026-09-23-spectroscopy-chirp-vs-saturation.html"
TITLE = "Chirp vs saturation spectroscopy"


def fig(png, caption, alt):
    b64 = base64.b64encode((DATA / png).read_bytes()).decode()
    return (f'<figure><figcaption class="small">{caption}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:100%;height:auto;border-radius:6px"></figure>')


def table(head, rows, cls="grid small"):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def n(v, fmt="{:.0f}"):
    return f'<span class="num">{fmt.format(v)}</span>'


def node_table():
    t = json.load(open(DATA / "night4_node_qpu.json"))
    rows = []
    for x in t["nodes"]:
        rows.append([f'<span class="mono">{esc(x["node"])}</span>', n(x["execs"]), n(x["median"], "{:.1f} s"), n(x["mean"], "{:.1f} s"),
                     n(x["p90"], "{:.0f} s"), n(x["max"], "{:.0f} s"),
                     f'<b>{n(100 * x["over60"], "{:.0f} %")}</b>' if x["over60"] >= 0.3 else n(100 * x["over60"], "{:.0f} %"),
                     n(100 * x["share"], "{:.1f} %")])
    devs = [[d["dev"], n(d["runs"]), n(d["execs"]), n(d["qpu_min"], "{:.0f} min"), n(d["qpu_per_run"], "{:.1f} min"),
             n(d["med_qpu"], "{:.1f} s"), n(d["med_wait"], "{:.1f} s"), n(100 * d["qpu_share"], "{:.0f} %")] for d in t["devices"]]
    return t, rows, devs


def build() -> str:
    t, node_rows, dev_rows = node_table()
    CSS = base.CSS + """
figure img { display:block; }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; font-size:14px; }
ol.recs li, ul.tight li { margin:6px 0; max-width:80ch; }
.corr { border-left:3px solid var(--warn); background:var(--warn-bg); padding:10px 14px; border-radius:6px; margin:14px 0; max-width:86ch; }
.corr li { margin:5px 0; }
h2 .eyebrow { display:block; margin-bottom:4px; }
"""
    body = f"""
<div class="page">
<p class="eyebrow">qua-agents benchmark · hardware investigation · 23 Sep 2026 · qolab Q1, arbel qB4, gilboa qD2</p>
<h1>{TITLE}</h1>
<p class="lede">IQCC now kills any job longer than 60 s. Working out what that means for the calibration graph led from
per-node QPU time, to what a spectroscopy shot actually spends its time on, to replacing the saturation pulse with a frequency
chirp. On three chips the chirp gave twice the signal, tolerated a wrong drive amplitude, and ignored the 0→2 two-photon line
that sent the arbel runs astray. Along the way: node 03b plays its pulses 4× longer than the state says, and on qolab its
flux-map timing produces no usable arc at all.</p>

<div class="tiles">
  <div class="tile"><div class="v">22 %</div><div class="k">of night-4 node executions ran over 60 s of QPU, holding 61 % of all chip time</div></div>
  <div class="tile"><div class="v">80 µs</div><div class="k">what 03b's "20 µs" saturation pulse really plays: <span class="mono">length * u.ns</span> passed as clock cycles</div></div>
  <div class="tile"><div class="v">0.71 / 0.66</div><div class="k">arbel qB4, node's saturation: real line vs 0→2 line. Chirp: 1.05 vs 0.09</div></div>
  <div class="tile"><div class="v">2.5×</div><div class="k">per-column precision of a chirped flux map over saturation, same QPU time</div></div>
  <div class="tile"><div class="v">250 ns</div><div class="k">chirp that still reaches 0.86 on gilboa qD2, T1 = 1.3 µs</div></div>
</div>

<div class="note"><b>Bottom line.</b> A chirp (an adiabatic passage: a 4 µs tone swept over 20 MHz) is a better spectroscopy
pulse than the long saturation pulse for both qubit spectroscopy and the qubit flux map: ~0.95 signal instead of at most ½, a
flat response over a 4× range of drive where a π pulse loses 20 % at ±30 %, and a first-order-only response that leaves the
second-order 0→2 line at a few percent. It needs a clean reset every shot, a drive in a window (set it with a short amplitude
ladder, not from a stored amplitude), and a sweep much shorter than T1. Independently of the pulse choice, three node defects
are worth fixing now: 03b's pulse units, 03b's flux timing on qolab, and the stored drive amplitudes that make arbel's default
drive 23 MHz.</div>

<h2><span class="eyebrow">1 · the 60 s cap</span>Where the chip time goes</h2>
<p>Every <span class="mono">run_node</span> event of the night-4 qwen3.8 OpenRouter runs ({n(t['n_execs'])} executions,
{n(t['n_runs'])} qubit-runs, 20–22 Sep) carries the executor's QPU counter. {n(t['over60'])} executions ({n(100 * t['over60'] / t['n_execs'], "{:.0f} %")})
exceeded 60 s, and they hold {n(t['qpu_min_in_long_jobs'], "{:.0f} min")} of the {n(t['total_qpu_min'], "{:.0f} min")} of chip time. Splitting every long job at 55 s would
add {n(t['extra_jobs_at_55s'])} jobs (1.35× the count). Nodes sorted by median QPU per execution:</p>
{table(["node", "execs", "median", "mean", "p90", "max", "over 60 s", "share of QPU"], node_rows)}
<p>Per device, what surrounds the QPU time matters more than the QPU time itself:</p>
{table(["device", "qubit-runs", "execs", "QPU total", "QPU per run", "median QPU", "median queue", "QPU share of wall"], dev_rows)}
<p>arbel's median queue wait is 61 s against 3–9 s elsewhere, so splitting jobs there costs a full queue wait per piece
(+62 min over its 9 runs at night-4 sizes). Because every node puts <span class="mono">n_avg</span> in the outermost loop and averages
in the stream processor, a job cut short by the new cap loses <i>averages</i>, not sweep points: the fetched data are noisier, not
truncated in frequency. The completed-average counter <span class="mono">n</span> that every node saves is the thing to check
against <span class="mono">n_avg</span>.</p>

<h3>One shot, taken apart (qolab Q1, T1 47.7 µs, 100 000 shots per step)</h3>
{table(["step", "adds", "µs per shot", "cost of the step"], [
    ["1", "loop and <span class='mono'>save(n)</span>", n(1.08, "{:.2f}"), n(1.08, "{:.2f}")],
    ["2", "readout and saving I, Q", n(11.50, "{:.2f}"), n(10.43, "{:.2f}")],
    ["3", "three <span class='mono'>align()</span>", n(11.58, "{:.2f}"), n(0.08, "{:.2f}")],
    ["4", "real-time <span class='mono'>update_frequency</span>", n(11.62, "{:.2f}"), n(0.04, "{:.2f}")],
    ["5", "saturation drive", n(23.30, "{:.2f}"), "11.67 — not trustworthy, see below"],
    ["6", "5×T1 thermal reset", n(261.95, "{:.2f}"), f"<b>{n(238.65, '{:.2f}')}</b> (= 5 × 47.7)"]])}
<p>Timed directly in a separate program, the saturation drive costs what its duration says: no duration argument (the stored
20 µs) +20.6 µs; <span class="mono">duration=5000</span> +20.1 µs; <span class="mono">duration=20000</span> +80.2 µs; the node's own
expression <span class="mono">length * u.ns</span> = 20000 → <b>+79.7 µs</b>. The ladder script's step 5 under-counted the same call
for a reason not found; its other rows agree with independent measurements. With the node's real 80 µs drive, a qolab Q1 shot is
~330 µs, of which the 5×T1 wait is ~72 % and the drive ~24 %.</p>

<h3>The pulse-units bug in node 03b</h3>
<p>QuAM's <span class="mono">play(duration=…)</span> counts 4 ns clock cycles; <span class="mono">unit(coerce_to_integer=True).ns</span> is 1.
03b and 03c pass <span class="mono">length * u.ns</span>, so the saturation pulse and the flux step run 4× the stored length. 03a
divides by 4 correctly. History: the expression is upstream's (original node, May 2025, still on qua-platform main
<span class="mono">1e15c4e</span> today, also in 03c). The fork fixed it on 19 Sep (<span class="mono">4ca26fb</span>), reverted it on
20 Sep (<span class="mono">553b9ea</span>) when gilboa maps looked wrong, found the real cause 70 minutes later (gilboa's state listed
only qC2 as active, parking every other flux line at 0 V) and noted "restore later"; it was never restored. In 09a the upstream bug
was fixed upstream on 24 Jul (<span class="mono">bce70ac</span>, #534), after the fork's last sync (<span class="mono">8007896</span>, 16 Jul).</p>
<p>Effect on night 4: 60 µs per flux-map shot, 44 of 223 flux-map QPU minutes (20 %); maps over 60 s would drop from 102 of 124 to
88. Restoring the <span class="mono">// 4</span> is not free, though: at 20 µs the saturated qubit has not reached steady state, and the
on-resonance signal depends on the unknown Rabi angle — measured on Q1 between 0.18 and 0.80 across drives (section 4). The fix
belongs together with a pulse change: a longer saturation (≥ ~2/γ ≈ 60–80 µs here) or a chirp.</p>

<h3>Active reset instead of the 5×T1 wait</h3>
<p>Same loop on qolab Q1 with <span class="mono">reset_qubit_active</span>: one measure-and-conditional-π costs 10.2 µs against
238.65 µs of thermal wait. The default 15-attempt loop averaged 5.67 attempts because <span class="mono">rus_exit_threshold</span>
(0.000184) sits inside today's |0⟩ blob (mean 0.000225, σ 0.000071; 70 % of thermally reset shots read above it); assignment
fidelity at the stored threshold is 0.92 (0.94 at the best one). Active reset only applies after readout is calibrated, so it
helps DRAG, Ramsey, error amplification and T1-type nodes, not the spectroscopy nodes that are the worst offenders.</p>

<h2><span class="eyebrow">2 · carry-over between shots</span>Does a short reset leave a copy of the arc?</h2>
<p>The concern: with the flux loop innermost, excitation left from a shot on the arc could appear in the next flux column. Pairs of
shots on qolab Q1 (pump on or 10 MHz off resonance, readout, wait k×T1, 80 µs probe 10 MHz off, readout) found <b>no carry-over</b>:
{n(-15.2, "{:+.1f}")} ± 1.0, {n(5.7, "{:+.1f}")}, {n(-0.2, "{:+.1f}")} ± 0.9, {n(-2.1, "{:+.1f}")}, {n(-1.9, "{:+.1f}")}, {n(-0.7, "{:+.1f}")} %
at k = 0, 0.5, 1, 2, 3, 5 (predicted 66 → 0.4 %). A follow-up with 20 µs pulses showed T1 intact (x180, 20 µs idle: 0.706) and the
readout non-destructive (the same with a readout in the middle: 0.710), while a detuned drive fired right after a readout pulled
the qubit to ~0.3 from either state. A detuned drive with no readout before it excites only ~0.02 (section 3, panel B), so the
likely cause is photons still ringing down in the resonator, Stark-shifting the qubit through the drive frequency — an accidental
adiabatic passage. That explains k = 0; the null result at k ≥ 1 is measured but not explained.</p>

<h2><span class="eyebrow">3 · qolab Q1</span>Chirp against saturation, one qubit, one frequency</h2>
{fig("chirp_vs_saturation.png", "qolab Q1, 36 s of QPU. A: population vs drive amplitude — the chirp plateaus at 0.97 from ½× to 2× the intended amplitude (T1 costs ~3 % over the 4 µs sweep) and follows Landau–Zener with the amplitude-to-Rabi scale taken from the calibrated π pulse (0.12 / 0.37 / 0.82 measured against 0.12 / 0.39 / 0.86 predicted); the π pulse drops to 0.80 at ±30 %. B: saturation line (FWHM 5.1 MHz, peak 0.55) against the chirp's response vs band centre. C: on f₀₁, saturation reads 0.52 at 20 µs and 0.47 at 80 µs, the chirp 0.97. D: taken where the stored α — a placeholder — puts the 0→2 line; nothing there (section 4 finds it 40 MHz away).", "Four panels comparing chirp and saturation on qolab Q1")}
<p><b>Resolution comes from the edges.</b> The chirp's response to band centre is a flat-topped box. Fitting an erf-edged box
(width 17.9 MHz, edges σ 1.07 MHz) places the line at {n(-0.05, "{:+.2f}")} ± 0.03 MHz from the stored f₀₁, twice as precisely as the
Lorentzian fit to the saturation line ({n(-0.29, "{:+.2f}")} ± 0.06 MHz, χ²/dof 6.9 against 2.2) from coarser steps (3 against 2 MHz).</p>

<h2><span class="eyebrow">4 · the trap</span>The 0→2 two-photon line</h2>
<p>A drive at f₀₁ − |α|/2 excites |2⟩ through two photons. Its coupling is second order, √2·Ω²/|α| (43 kHz at 3 MHz Rabi on Q1,
70× weaker than the drive), so its line is narrow. Saturation parks on each frequency for 20–80 µs and gives even a weak coupling
time to act; a chirp sweeps across that narrow resonance in ~10 ns, too fast for it to respond (Landau–Zener: transfer grows as
Ω⁴/(α²·sweep rate) for the 0→2 line, as Ω²/sweep rate for the real one).</p>
<p>qolab stores α = 215.5 MHz for all six qubits — a placeholder. A strong-drive search found Q1's 0→2 line at f₀₁ − 148 MHz:
<b>α ≈ 296 MHz</b>.</p>
{fig("twophoton_suppression.png", "qolab Q1, 37 s of QPU. A: the 0→2 line, found 40 MHz below where the placeholder α puts it. B: saturation shows it as a sharp line from 2 MHz Rabi on. C: at the same drive, saturation on the 0→2 line reads 0.28 / 0.40 / 0.64 / 0.83 at 2 / 4 / 8 / 16 MHz; the chirp reads 0.00 / 0.00 / 0.10 / 0.85, Landau–Zener with the measured α predicts 0.00 / 0.006 / 0.088 / 0.77; the chirp on f₀₁ stays at 0.93–0.95. Saturation on f₀₁ (20 µs) wanders between 0.18 and 0.80 with drive — the unknown Rabi angle at 20 µs.", "Search for and drive ladder on Q1's 0→2 line")}
{fig("trap_qB4.png", "arbel qB4, 25 s of QPU on the 22 Sep lab snapshot. The stored saturation amplitude is 1.0 (full scale), so the node's default drive is 23 MHz Rabi. A: at that drive the real line (0.71) and the 0→2 line at f₀₁ − 98 MHz (0.66, α ≈ 196 MHz) have the same height — the picture the agents misread four times; the chirp over the same 150 MHz shows the real line at 1.05 and the 0→2 line at 0.09 (baseline 0.04). B: chirp on f₀₁ ≥ 0.90 from 2 MHz Rabi, chirp on the 0→2 line ≤ 0.03 up to 4 MHz, 0.24 at 8 and 0.88 at 16 MHz (Landau–Zener 0.19 and 0.97). A gentle saturation drive (1–2 MHz) also leaves the 0→2 line at 0.07: the failures come from the stored drive amplitude.", "Saturation and chirp on arbel qB4")}
<p>The window where the chirp inverts the real line and ignores the 0→2 line is ~1.5–7 MHz Rabi on Q1 and ~1.5–5.6 MHz on
qB4 for a 4 µs, 20 MHz sweep (strict: ≥ 99 % and &lt; 5 %). In practice the chirp keeps the real line ≥ 30× above the 0→2 line from 1 to
4 MHz on both chips (still 10× on Q1 and 4× on qB4 at 8 MHz), while saturation's two lines are within 2× of each other from 2 MHz upward and trade places at 8 MHz on Q1.
The drive that sets this window is not known before power Rabi: the stored π amplitudes imply 10 to 222 MHz of Rabi per unit
amplitude across qolab Q1 and arbel qB4, qC2, qC3. A 3–4-step amplitude ladder finds the window and separates the lines in one
pass.</p>

<h2><span class="eyebrow">5 · short T1</span>Where the chirp breaks down</h2>
{fig("shortT1_qD2.png", "gilboa qD2, T1 1.34 µs, 7 s of QPU on a local copy of the 22 Sep snapshot with the C/D qubits marked active (the cloud state lists only qC2, which parks qD2's flux at 0 V). A: a 4 µs sweep decays before it finishes (0.02–0.41); 1 µs reaches 0.45–0.71; 250 ns reaches 0.86 at 8 MHz Rabi. B: the real line sits 4.75 MHz below the stored f₀₁ — saturation at the node's weak default drive (0.5 MHz Rabi) parked on the stored f₀₁ saw nothing (−0.01); the chirp band covered the offset (plateau 0.80–0.90 for band centres −10 … 0 MHz). Short sweeps soften the box edges and narrow the amplitude window to ~4–10 MHz.", "Chirp durations against saturation on gilboa qD2")}

<h2><span class="eyebrow">6 · qubit flux map</span>Chirped map against saturation, and 03b's timing</h2>
{fig("fluxmap2_chirp_vs_saturation.png", "qolab Q1, 15 pulsed flux offsets over ±94 mV, ~17.5 s of QPU per map. A: saturation with 03b's timing — drive and flux step start and stop together — shows an idle-frequency line in every column and only a faint arc in the outer columns; the fit finds no arc. B: the same drive placed inside the flux step (5 µs margins): clean arc, sweet spot +0.13 ± 0.14 mV. C: chirp with the flux step leading by 5 µs: sweet spot +0.30 ± 0.06 mV. D: the two good maps agree (curvature −2.62 and −2.60 kHz/mV²); the chirp's column centres are ±69 kHz against ±173 kHz. Both are 16 % deeper than the stored DC curvature (−2.25 kHz/mV²).", "Three flux maps and the extracted arcs")}
<p><b>03b's timing fails on qolab.</b> The flux step holds its value (a lead-time scan gave f₀₁ − 17.0 MHz at ±81 mV for leads from
0.5 to 40 µs), and 03b's flux scaling is correct (<span class="mono">dc / const_amp</span>: −11.8 MHz at +67 mV against −12.1 for the
reference). What differs between panels A and B is only whether the drive overlaps the flux step's edges — a latency between the
xy and z lines or the step's rise leaves the qubit near its idle frequency for part of the drive, and 80 µs of drive is plenty to
saturate it there. Which of the two it is has not been separated. The fix is small: start the flux step before the drive and end it
after. This may be behind the "shallow arc" notes in the qolab recipe log; arbel and gilboa were not checked.</p>
<p>The chirped map needs ~20 band centres instead of ~70 rows and gives each column's f₀₁ from a box fit rather than a peak; the
downstream cosine fit is unchanged.</p>

<h2><span class="eyebrow">7 · what to do</span>Recommendations</h2>
<ol class="recs">
<li><b>03b: give the flux step margins around the drive</b> (5 µs before and after). On qolab Q1 this is the difference between no
arc and a clean one. Check arbel and gilboa the same way.</li>
<li><b>Replace saturation with a chirp in 03a and 03b.</b> 03a: a chirp amplitude ladder over ~20 MHz bands, a box fit, then the
fine spectroscopy node inside the found band (the 0→2 line is α/2 away, outside its window). 03b: the chirped map with flux margins
and per-column box fits. Shorten the sweep for short T1 (250 ns at T1 = 1.3 µs); keep saturation as a fallback below ~1 µs.</li>
<li><b>Fix the 03b/03c pulse units</b> (restore <span class="mono">4ca26fb</span>, apply it to 03c, take upstream's 09a line) together
with item 2 or a ≥ 60 µs saturation — a bare 20 µs saturation trades 20 % of QPU for an unpredictable signal. Report 03b/03c
upstream; they are still broken on qua-platform main.</li>
<li><b>Stop taking the spectroscopy drive from the stored saturation amplitude.</b> arbel stores 1.0 (full scale) on every qubit,
which is what put qB4 and qC2 in the two-photon regime. Set it from a measured line width or the chirp ladder.</li>
<li><b>Treat qolab's anharmonicities as unknown</b>: 215.5 MHz on all six qubits; Q1's is ~296 MHz. Anything that uses α on qolab
(two-photon checks, DRAG, leakage estimates) inherits an 80 MHz error.</li>
<li><b>For the 60 s cap</b>: per-node timeouts from <span class="mono">job_preflight</span>'s estimate instead of one global value;
active reset for the post-readout nodes after recalibrating <span class="mono">rus_exit_threshold</span>; split jobs by
<span class="mono">n_avg</span> only where the sweep can't shrink, preferring large pieces on arbel.</li>
</ol>
<p><b>Not yet shown:</b> the chirp on more qubits (arbel qC2, the other two-photon case, and gilboa's C/D row), why 03b's timing fails
(latency or rise), the null carry-over at waits ≥ 1×T1, and a chirp node inside the agents' loop.</p>

<h2><span class="eyebrow">8 · corrections</span>What was said and then corrected during the day</h2>
<div class="corr"><ul class="tight">
<li>"A fixed ~65–85 µs per shot is unexplained overhead" → partly retracted, then explained: on qolab the non-reset, non-drive cost is
~12 µs; arbel's extra ~87 µs was the 80 µs drive.</li>
<li>"The detuned probe drive erases the previous shot's state" → wrong for waits ≥ 1×T1; the erasure seen at k = 0 needed a readout
just before the drive.</li>
<li>"The f₀₁ line at the node's drive is tens of MHz wide" → 5.1 MHz on Q1; the wide estimate came from the same readout-preceded
measurement.</li>
<li>"No 0→2 line on Q1" → there is one, 40 MHz from where the placeholder α put the search.</li>
<li>First chirped and saturation flux maps: my program wrote <span class="mono">dc * (1/0.1)</span>, putting 10.0 into QUA's fixed type
([−8, 8)), which wrapped to −6 → both maps at −0.6× flux. Data kept in the folder as <span class="mono">fluxmap_*</span>; the
figures above are the corrected <span class="mono">fluxmap2</span> run.</li>
<li>"A chirp needs 1.5–7 MHz Rabi" is the strict window; the practical range where it gives the right answer by a clear margin is
~0.5–10 MHz.</li>
</ul></div>

<h2><span class="eyebrow">9 · provenance</span>Runs, scripts and data</h2>
<p>All scripts, data (<span class="mono">.npz</span> / <span class="mono">.json</span>), figures and run logs are in
<a href="2026-09-23-spectroscopy/">2026-09-23-spectroscopy/</a>; the first profiling scripts are under
<span class="mono">scratch-profiling/</span>. Every run started from the lab snapshots pulled on 22 Sep
(<span class="mono">~/qab-runs/reference-state-20260922/</span>), used <span class="mono">initialize_qpu</span>, injected its pulses into
the generated config only and wrote nothing to any state. Each launch is logged in <span class="mono">~/qab-runs/recipe-qolab-LOG.md</span>.</p>
{table(["test", "backend · qubit", "QPU", "script"], [
    ["shot ladder, pulse timing, active reset, readout check", "qolab Q1", "~80 s", "<span class='mono'>scratch-profiling/profile_*.py, duration_types.py</span>"],
    ["carry-over test and diagnostic", "qolab Q1", "~115 s", "<span class='mono'>scratch-profiling/ghost_test.py, ghost_diag.py</span>"],
    ["chirp vs saturation (section 3)", "qolab Q1", "36 s", "<span class='mono'>chirp_test.py</span>"],
    ["two-photon search and ladders (section 4)", "qolab Q1", "37 s", "<span class='mono'>twophoton_test.py</span>"],
    ["flux maps v1 (overflowed flux), lead-time scan", "qolab Q1", "40 s", "<span class='mono'>fluxmap_test.py, fluxlead_test.py</span>"],
    ["flux maps v2 and 03b expression check (section 6)", "qolab Q1", "56 s", "<span class='mono'>fluxmap2_test.py</span>"],
    ["trap test (section 4)", "arbel qB4", "25 s", "<span class='mono'>trap_arbel_qB4.py</span>"],
    ["short-T1 test (section 5)", "gilboa qD2", "7 s", "<span class='mono'>shortT1_gilboa_qD2.py</span>"]])}
<p class="small muted">Code: qua-libs fork <span class="mono">feat/qualibrate-ai</span> at <span class="mono">65b4755</span> (03b with the
4× duration), upstream qua-platform/qua-libs main <span class="mono">1e15c4e</span>; tinycal's IQCC wrapper for execution. Night-4
node timing: <span class="mono">night4_node_qpu.json</span> from {esc(t['source'])}. Generated by
<span class="mono">make_spectroscopy_report.py</span>.</p>
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
