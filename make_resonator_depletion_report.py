"""Build 2026-09-25-resonator-depletion.html from 2026-09-25-resonator-depletion/ (no QPU, no network).

The data were taken on 25 Sep 2026 by run_02b.py, run_02a.py, kappa.py and kappa_long.py and analysed by ab_analyse.py,
kappa_analyse.py and long_analyse.py (all in the data folder).
Usage:
    python make_resonator_depletion_report.py [--sync]
--sync copies the run files from ~/qab-runs/depletion-20260925 into 2026-09-25-resonator-depletion/ first (everything but
the 02b/02a netCDF maps, ~200 MB, which stay in the run folder).
"""
from __future__ import annotations

import base64
import json
import math
import shutil
import sys
from html import escape as esc
from pathlib import Path

import make_fwcmp2_report as base

HERE = Path(__file__).parent
DATA = HERE / "2026-09-25-resonator-depletion"
RUNS = Path.home() / "qab-runs/depletion-20260925"
OUT = HERE / "2026-09-25-resonator-depletion.html"
TITLE = "Resonator depletion"
QB = [("arbel", "qB4"), ("arbel", "qA5"), ("arbel", "qD1"), ("qolab", "Q1"), ("qolab", "Q2"), ("qolab", "Q5")]
QB_02B = QB[1:]          # qB4: the node finds no resonance in its window under any condition


def sync():
    for sub in ("ab", "kappa", "figures"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    for be in ("arbel", "qolab", "gilboa"):
        for kind, pats in (("ab", ("*.json", "*.log", "*.jsonl")), ("kappa", ("*.npz", "*.json", "*.log"))):
            src = RUNS / kind / be
            if not src.exists():
                continue
            (DATA / kind / be).mkdir(parents=True, exist_ok=True)
            for pat in pats:
                for f in src.glob(pat):
                    shutil.copy(f, DATA / kind / be / f.name)
    for f in ("ab/summary.json", "kappa/summary.json", "kappa/long_summary.json"):
        shutil.copy(RUNS / f, DATA / f)
    for f in (RUNS / "figures").glob("*.png"):
        shutil.copy(f, DATA / "figures" / f.name)
    for f in list(RUNS.glob("*.py")) + list(RUNS.glob("*.sh")):
        shutil.copy(f, DATA / f.name)


def img(path, caption, alt):
    b64 = base64.b64encode((DATA / path).read_bytes()).decode()
    return (f'<figure><figcaption class="small">{caption}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" style="width:100%;height:auto;border-radius:6px"></figure>')


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="grid small"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def ok(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def num(v, fmt="{:.2f}", dash="–"):
    return f'<span class="num">{fmt.format(v)}</span>' if ok(v) else dash


def mean(vs):
    vs = [v for v in vs if ok(v)]
    return sum(vs) / len(vs) if vs else float("nan")


def build() -> str:
    ab = json.loads((DATA / "ab/summary.json").read_text())
    ks = json.loads((DATA / "kappa/summary.json").read_text())
    ls = json.loads((DATA / "kappa/long_summary.json").read_text())

    def r02b(be, q, c, rnd):
        return ab.get(f"{be}/{q}/{c}/r{rnd}") or {}

    # --- 02b table -----------------------------------------------------------------------------------------------
    rows_02b, d_c, d_d, sc_a, sc_c, qpu = [], [], [], [], [], {c: [] for c in "ABCD"}
    for be, q in QB_02B:
        a = [r02b(be, q, "A", r).get("optimal_power") for r in (1, 2)]
        fa = [r02b(be, q, "A", r).get("frequency_shift") for r in (1, 2)]
        row = [be, f"<b>{q}</b>"]
        for c in "ABCD":
            p = [r02b(be, q, c, r).get("optimal_power") for r in (1, 2)]
            f = [r02b(be, q, c, r).get("frequency_shift") for r in (1, 2)]
            qpu[c] += [r02b(be, q, c, r).get("qpu_s") for r in (1, 2)]
            bad = c == "B" and (min(abs((x or 0) - mean(fa)) for x in f) > 0.5e6 or abs(mean(p) - mean(a)) > 3)
            cell = (f'{num(p[0], "{:.1f}")} / {num(p[1], "{:.1f}")} dBm<br>'
                    f'{num(f[0] / 1e6 if ok(f[0]) else None, "{:+.2f}")} / {num(f[1] / 1e6 if ok(f[1]) else None, "{:+.2f}")} MHz')
            row.append(f'<span class="warn">{cell}</span>' if bad else cell)
        d_c.append(mean([r02b(be, q, "C", r).get("optimal_power") for r in (1, 2)]) - mean(a))
        d_d.append(mean([r02b(be, q, "D", r).get("optimal_power") for r in (1, 2)]) - mean(a))
        sc_a.append(abs(a[0] - a[1]) if ok(a[0]) and ok(a[1]) else float("nan"))
        c2 = [r02b(be, q, "C", r).get("optimal_power") for r in (1, 2)]
        sc_c.append(abs(c2[0] - c2[1]) if ok(c2[0]) and ok(c2[1]) else float("nan"))
        rows_02b.append(row)
    qpu_m = {c: mean(v) for c, v in qpu.items()}

    # --- low rows ------------------------------------------------------------------------------------------------
    rows_low = []
    for be, q in QB_02B:
        row = [be, f"<b>{q}</b>"]
        for c in "ABC":
            lr = [(r02b(be, q, c, r).get("low_rows") or {}) for r in (1, 2)]
            dl = mean([x.get("depth_low_at_ref") for x in lr])
            dr = mean([x.get("depth_ref") for x in lr])
            pl = mean([x.get("pos_low") for x in lr])
            row.append(f'{num(dl / dr if ok(dl) and ok(dr) and dr else None, "{:.2f}")} · {num(pl / 1e6 if ok(pl) else None, "{:+.2f}")} MHz')
        rows_low.append(row)

    # --- pump-probe ----------------------------------------------------------------------------------------------
    rows_pump = []
    for be, q in QB:
        ex = {}
        for r in (1, 2):
            p = (ks.get(f"{be}/{q}/r{r}") or {}).get("pump")
            if p:
                for t, v in zip(p["taus_ns"], p["excited"]):
                    ex.setdefault(t, []).append(v)
        meta = (ks.get(f"{be}/{q}/r1") or {}).get("meta") or {}
        dp = meta.get("top_dbm", float("nan")) - meta.get("committed_power_dbm", float("nan"))
        rows_pump.append([be, f"<b>{q}</b>", num(dp, "+{:.0f} dB")] +
                         [num(mean(ex.get(t, [])), "{:.2f}") for t in (200, 3000, 12000, 40000, 80000)] +
                         [num((meta.get("T1") or float("nan")) * 1e6, "{:.0f} µs")])

    # --- kappa ---------------------------------------------------------------------------------------------------
    rows_k, rows_dep, ratios_short, ratios_long, ratios_node = [], [], [], [], []
    ratios_02b, ratios_02a, stored_k, left3 = [], [], [], []
    for be, q in QB:
        kd = [((ks.get(f"{be}/{q}/r{r}") or {}).get("ringdown") or {}).get(lv, {}).get("kappa_down") for r in (1, 2)
              for lv in ("low", "ro")]
        ku = [((ks.get(f"{be}/{q}/r{r}") or {}).get("ringdown") or {}).get(lv, {}).get("kappa_up") for r in (1, 2)
              for lv in ("low", "ro")]
        kd = [k / 2 / math.pi for k in kd if ok(k)]
        ku = [k / 2 / math.pi for k in ku if ok(k)]
        kr = mean(kd)
        sd = (sum((k - kr) ** 2 for k in kd) / max(1, len(kd) - 1)) ** 0.5 if len(kd) > 1 else float("nan")
        L = ls.get(f"{be}/{q}") or {}
        ll, lc = L.get("long_low", {}).get("lorentz"), L.get("long_low", {}).get("circle")
        sl, sc = L.get("short_ro", {}).get("lorentz"), L.get("short_ro", {}).get("circle")
        nb = mean([r02b(be, q, "A", r).get("linewidth") for r in (1, 2)])
        na = []
        for r in (1, 2):
            f = DATA / "ab" / be / f"02a-r{r}-{q}.json"
            if f.exists():
                na.append(((json.loads(f.read_text()).get("fit_results") or {}).get(q) or {}).get("fwhm"))
        na = mean(na)
        for v, lst in ((ll, ratios_long), (lc, ratios_long), (sl, ratios_short), (sc, ratios_short), (nb, ratios_node)):
            if ok(v):
                lst.append(v / kr)
        if ok(na) and na < 5e6:
            ratios_node.append(na / kr)
            ratios_02a.append(na / kr)
        if ok(nb):
            ratios_02b.append(nb / kr)
        rl = (L.get("readout_len") or 0)
        rows_k.append([be, f"<b>{q}</b>", f'{num(kr / 1e6, "{:.3f}")} ± {num(sd / 1e6, "{:.3f}")}', num(mean(ku) / 1e6, "{:.3f}"),
                       num(ll / 1e6 if ok(ll) else None, "{:.3f}"), num(lc / 1e6 if ok(lc) else None, "{:.3f}"),
                       f'{rl} ns', num(sl / 1e6 if ok(sl) else None, "{:.2f}"), num(sc / 1e6 if ok(sc) else None, "{:.2f}"),
                       num(nb / 1e6 if ok(nb) else None, "{:.2f}"),
                       num(na / 1e6 if ok(na) else None, "{:.2f}") + ('<span class="warn"> (wrong feature)</span>' if ok(na) and na > 5e6 else "")])
        kappa = 2 * math.pi * kr
        stored = ((ks.get(f"{be}/{q}/r1") or {}).get("meta") or {}).get("depletion_time")
        if stored:
            stored_k.append(stored * 1e-9 * kappa)
        left3.append(math.exp(-kappa * 3e-6 / 2))
        rows_dep.append([be, f"<b>{q}</b>", num(kr / 1e6, "{:.2f}"), num(1e9 / kappa, "{:.0f} ns"),
                         num(9.21 / kappa * 1e6, "{:.1f} µs"),
                         num(math.exp(-kappa * 1e-6 / 2), "{:.2f}"), num(math.exp(-kappa * 3e-6 / 2), "{:.3f}"),
                         num(math.exp(-kappa * 12e-6 / 2), "{:.0e}"), num(stored * 1e-3 * kappa * 1e-6 if stored else None, "{:.1f} / κ")])

    CSS = base.CSS + """
figure img { display:block; }
ul.tight li, ol.recs li { margin:6px 0; max-width:82ch; }
td .muted, .muted { color:var(--muted); }
.warn { color:#c0392b; }
h3 { margin-top:28px; }
"""
    rmin_s, rmax_s = min(ratios_short), max(ratios_short)
    rmin_l, rmax_l = min(ratios_long), max(ratios_long)
    rmin_n, rmax_n = min(ratios_node), max(ratios_node)
    body = f"""
<div class="page">
<p class="eyebrow">qua-agents benchmark · hardware test · 25 Sep 2026 · arbel, qolab · 6 qubits</p>
<h1>{TITLE}</h1>
<p class="lede">Two thirds of 02b's QPU time (resonator spectroscopy vs power) is a wait four times longer than intended: <span class="mono">depletion_time</span> is in ns and is passed unconverted to <span class="mono">wait()</span>,
which counts 4 ns cycles, so the state's 3 µs waits 12 µs. Fixing that alone breaks the node: the top powers of each
frequency column leave the qubit excited for tens of µs, and with points 4.5 µs apart that excitation fills the lowest rows, the
ones the fit takes its reference from — on qolab Q1 and Q2 it proposed the resonator frequency for an excited qubit, 1.1–1.25 MHz
off, and a power 7–10 dB too low. With power as the outer loop and one reset per round the node is three times faster
({qpu_m['A']:.0f} → {qpu_m['C']:.0f} s of QPU per qubit) and proposes what it proposes today, within today's own round-to-round
scatter, from clean reference rows. Separately, the linewidth the nodes report is not κ: the readout pulse is shorter than the
resonator's fill time, and the width comes out {rmin_n:.1f}–{rmax_n:.1f}× the κ measured in the time domain. A long-pulse sweep at
low power gets κ within {100 * (rmax_l - 1):.0f} %.</p>

<div class="tiles">
  <div class="tile"><div class="v">{qpu_m['A']:.0f} → {qpu_m['C']:.0f} s</div><div class="k">QPU per qubit for 02b's default grid, today against the fix (power outer, 3 µs); {qpu_m['D']:.0f} s at 1 µs</div></div>
  <div class="tile"><div class="v">−1.2 MHz</div><div class="k">where the unit fix alone puts qolab Q1's and Q2's resonator: the line of an excited qubit</div></div>
  <div class="tile"><div class="v">0.7–1.0</div><div class="k">of the way to |e⟩, the next readout 3 µs after a top-of-sweep readout; still 0.25–0.8 after 80 µs</div></div>
  <div class="tile"><div class="v">{rmin_n:.1f}–{rmax_n:.1f}×</div><div class="k">the nodes' linewidths over the ring-down κ (κ/2π = 0.25–0.59 MHz here)</div></div>
</div>

<h2>1 · What was tested</h2>
<p>Six qubits, three per backend: arbel qB4, qA5, qD1 and qolab Q1, Q2, Q5 (gilboa's QOP was down all evening; its three qubits are
not measured). States: the IQCC cloud states pulled at 18:03 (<span class="mono">~/qab-runs/state-20260925-1803</span>). Nothing was
written to any state. Everything ran twice, the second round in reverse order.</p>
<ul class="tight">
<li><b>02b under four conditions</b>, one job per qubit, the node's default grid (15 MHz at 50 kHz × 100 powers × 100 shots). The power
window is 35 dB wide, ending at min(−10 dBm, committed power + 20 dB), because committed powers here span −43 to −8 dBm and the
default −50…−25 dBm misses most onsets.
<b>A</b> today's node: power the inner loop, the 3 µs depletion time waiting 12 µs.
<b>B</b> today's node with <span class="mono">depletion_time</span> ÷ 4 in a local state copy: the unit fix alone, 3 µs.
<b>C</b> the fix (qua-libs branch <span class="mono">fix/resonator-depletion-wait</span> 9ae8a32): the wait converted to cycles,
power the outer loop and frequency the inner, and a 1 ms wait once per round before the sweep climbs the powers again.
<b>D</b> as C with a 1 µs depletion time.</li>
<li><b>κ in the time domain</b>: a 2 µs drive at the readout frequency followed by 2.5 µs of silence in one measurement window,
demodulated in 40 ns slices on the FPGA. After the drive stops only the field leaking out of the resonator is left; it decays
as e<sup>−κt/2</sup>. Fitted at two powers (committed − 10 dB and committed) for the decay and for the ring-up.</li>
<li><b>κ from lineshapes</b>: the same ±10 MHz sweep with the qubit's readout pulse (circle fit, Lorentzian on |S21|², the half-depth
width of |S21|); a ±3 MHz sweep with a 10 µs pulse integrated over its last 4 µs, once the resonator has settled; and what 02a and
02b themselves report.</li>
<li><b>Pump–probe</b>: a readout at the top of 02b's sweep, a delay τ, then the committed readout, against ground and x180
references — what the next point of a 02b sweep inherits from the last one.</li>
</ul>

<h2>2 · 02b under the four conditions</h2>
<p>Proposed power and frequency shift, round 1 / round 2 (red: the unit fix alone moving the answer by more than 3 dB or 0.5 MHz).
arbel qB4 is left out: its line sits on a broad background feature and the node finds no resonance under any condition.</p>
{table(["backend", "qubit", "A · today (12 µs, power inner)", "B · 3 µs, power inner", "C · 3 µs, power outer", "D · 1 µs, power outer"], rows_02b)}
{img("figures/summary_02b.png", "Each condition's proposal against A's first round. Circles round 1, squares round 2. Right: QPU per job.", "02b summary")}
<ul class="tight">
<li><b>C reproduces today's answers</b>: its power is {min(d_c):+.1f} to {max(d_c):+.1f} dB from A's (mean of two rounds),
where A itself moved by up to {max(sc_a):.1f} dB between rounds (C: up to {max(sc_c):.1f} dB); the frequency agrees within 0.1 MHz on
every qubit.</li>
<li><b>B does not</b>: on qolab Q1 and Q2 it proposes −1.2 and −1.1 MHz — the resonator with the qubit excited — and 7–10 dB less power;
on Q5 (round 2) −0.9 MHz and −18 dB; on arbel qA5 3.7 dB more (qD1 stays within A's own scatter).</li>
<li><b>D mostly agrees with C</b> ({min(d_d):+.1f} to {max(d_d):+.1f} dB from A), but arbel qA5 comes out 4.5 dB lower in both rounds, so
1 µs is at the edge for a 0.31 MHz resonator. 3 µs showed no such shift anywhere.</li>
<li><b>QPU per qubit</b>: A {qpu_m['A']:.0f} s, B {qpu_m['B']:.0f} s, C {qpu_m['C']:.0f} s, D {qpu_m['D']:.0f} s. The 1 ms reset per round costs
0.1 s.</li>
</ul>
{img("figures/ab_qolab_r1.png", "qolab, round 1: |S21| normalised per power row. Green line and marker: the proposed power and frequency. Shaded: the lowest 10 rows. In A a second line ~1.2 MHz below the resonator fades up from the bottom; in B it dominates the reference block and the node fits it; C and D show one line bending smoothly towards the bare cavity.", "qolab maps")}
{img("figures/ab_arbel_r1.png", "arbel, round 1 (D in the report data folder).", "arbel maps")}

<h2>3 · Why the unit fix alone breaks it</h2>
<p>A readout 20 dB above the committed power leaves the qubit excited, or leaked beyond |e⟩ (values above 1, and a component across
the g–e line), and it comes back on the scale of T1 or slower. The next readout 3 µs later sits 0.7–1.0 of the way to |e⟩:</p>
{table(["backend", "qubit", "top of 02b vs committed", "0.2 µs", "3 µs", "12 µs", "40 µs", "80 µs", "stored T1"], rows_pump)}
{img("figures/summary_pump.png", "The committed readout's shift towards |e⟩, τ after a readout at the top of 02b's sweep (mean of two rounds). arbel qB4's top is only 4 dB above its committed power and does nothing.", "pump-probe")}
<p>With power the inner loop, every frequency column ends on those powers and the next one starts at the bottom. 80 µs is six rows
at A's 13.5 µs per row and eighteen at B's 4.5 µs, and the excitation is still 0.25–0.8 then: in B it covers most of the block the
fit takes the dressed frequency, linewidth and plateau contrast from. The lowest 10 rows show it directly (dip depth at the reference minimum,
relative to rows 10–24, and where the lowest rows' minimum sits):</p>
{table(["backend", "qubit", "A · depth ratio · minimum", "B", "C"], rows_low)}
<p>Today's node is already affected, just less: the ratio is 0.4–0.6, and on qolab Q1 and Q2 the lowest rows' minimum already sits
at −1.1 to −1.2 MHz. With power the outer loop the top powers come once per round, at the end, followed by a 1 ms wait, and the ratio
is 0.9–1.1 with the minimum where the reference has it.</p>

<h2>4 · Is the linewidth κ?</h2>
{table(["backend", "qubit", "ring-down κ/2π [MHz]", "ring-up", "10 µs pulse, low power: |S21|² Lorentzian", "circle fit",
        "readout pulse", "readout pulse: |S21|² Lorentzian", "circle fit", "02b linewidth", "02a FWHM"], rows_k)}
{img("figures/summary_kappa.png", "Every lineshape estimate over the ring-down κ. Green: a 10 µs pulse integrated once settled, at committed − 10 dB. Red: the qubit's readout pulse. Purple, orange: the nodes.", "kappa comparison")}
<ul class="tight">
<li><b>The ring-down is consistent</b>: the decay at two powers and in two rounds agrees within ~10 % on every resonator (the ±
column). κ/2π is 0.25–0.32 MHz on five of them and 0.59 MHz on arbel qD1. The ring-up, with the steady state a free parameter
and the drive's own transient in it, scatters more and is shown for comparison only.</li>
<li><b>A steady-state sweep gives κ</b>: with the 10 µs pulse at low power the |S21|² Lorentzian and the circle fit land at
{rmin_l:.2f}–{rmax_l:.2f}× the ring-down value.</li>
<li><b>The readout pulse does not</b>: {rmin_s:.1f}–{rmax_s:.1f}× with the same fits. The readout pulses here are 0.8–2 µs; the
resonator's field builds up with a time constant 2/κ = 0.5–1.3 µs, so it never settles within the pulse, and the dip takes the
pulse's own spectral width, ~1/T.</li>
<li><b>The nodes' widths</b>: 02b's linewidth is {min(ratios_02b):.1f}–{max(ratios_02b):.1f}× κ; 02a at its defaults
{min(ratios_02a):.1f}–{max(ratios_02a):.1f}× where it finds the line, and ~10 MHz on arbel qA5 and qD1, where its fit took a
background feature.</li>
</ul>
{img("figures/long_qolab.png", "qolab: |S21|² with the 10 µs pulse (blue, green) and the readout pulse (red), and a Lorentzian whose width is the ring-down κ (dashed).", "long pulse qolab")}
{img("figures/long_arbel.png", "arbel, the same.", "long pulse arbel")}
{img("figures/kappa_qolab_r1.png", "qolab, round 1: the sliced traces (top; the drive stops at 2000 ns), the tail in the IQ plane, the ±10 MHz sweep with the readout pulse and its circle fit, and the pump–probe.", "ring-down qolab")}
{img("figures/kappa_arbel_r1.png", "arbel, round 1, the same.", "ring-down arbel")}

<h2>5 · How long a depletion wait needs to be</h2>
<p>The field left after a wait w is e<sup>−κw/2</sup> of what it was. For it to fall to 1 % the wait is 2 ln 100 / κ = 9.2 / κ:</p>
{table(["backend", "qubit", "κ/2π [MHz]", "1/κ", "wait to 1 % of the field", "field left after 1 µs", "after 3 µs", "after 12 µs",
        "stored depletion_time"], rows_dep)}
<p>The stored 3 µs is {min(stored_k):.1f}–{max(stored_k):.0f} / κ on these resonators and leaves up to
{100 * max(left3):.0f} % of the field on the narrowest (qolab Q2). In 02b that does
not matter much, because the next point is 50 kHz away at the same power and carries almost the same field (C and D agree with A);
it would matter more where consecutive measurements differ, and there κ from a ring-down or a steady-state sweep gives the wait
directly.</p>

<h2>6 · Recommendations</h2>
<ol class="recs">
<li><b>Take the fix as a whole</b> (<span class="mono">fix/resonator-depletion-wait</span> 9ae8a32): the wait in cycles, power the outer
loop, a reset per round. It is three times faster and cleaner than today's node. Do not fix the unit alone.</li>
<li><b>Keep the depletion wait at the stored 3 µs</b> for 02b; 1 µs is at the edge (qA5). Better, set it per resonator from κ,
~9 / κ.</li>
<li><b>Measure κ, do not read it off the nodes' linewidths.</b> A ring-down (one level, ~2 s of QPU) or a steady-state sweep with a
long pulse at low power (~2 s) both agree with each other within ~15 %; 02a/02b's widths are 1.5–4 × too wide.</li>
<li><b>The same unit slip is in 02a, 02c, 02d, 02e</b> (and in 01a/b, 03a, 03d, 07, 12, 14, 15). The resonator sweeps pay for it the
same way; each would need the same check that no strong readout precedes the reference points before the wait is shortened.</li>
</ol>

<h2>7 · Open</h2>
<ol class="recs">
<li>gilboa (qD2, qC3, qD5): not measured, QOP down from 18:13.</li>
<li>Whether the committed readout itself leaves the qubit excited (a pump at the committed power) was not measured; it matters for
back-to-back readouts and active reset.</li>
<li>02a's fit at its defaults took a 10 MHz background feature for the line on arbel qA5 and qD1.</li>
</ol>

<h2>Provenance</h2>
<p class="small">Scripts, per-job records, κ data and driver logs are in <span class="mono">2026-09-25-resonator-depletion/</span>:
<span class="mono">run_02b.py</span> (the four 02b conditions), <span class="mono">run_02a.py</span>, <span class="mono">kappa.py</span>
(ring-down, sweep, pump–probe), <span class="mono">kappa_long.py</span> (the 10 µs pulse), <span class="mono">chain.sh</span>,
<span class="mono">chain2.sh</span>, <span class="mono">retry.sh</span> (five round-2 jobs lost to cloud 503s and timeouts, rerun),
the analyses (<span class="mono">ab_analyse.py</span>, <span class="mono">kappa_analyse.py</span>, <span class="mono">long_analyse.py</span>) and
plots. The 02b/02a netCDF maps (~200 MB) stay in <span class="mono">~/qab-runs/depletion-20260925/ab/</span>. Each launch is logged
in <span class="mono">~/qab-runs/recipe-qolab-LOG.md</span>. Generated by <span class="mono">make_resonator_depletion_report.py</span>.</p>
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
