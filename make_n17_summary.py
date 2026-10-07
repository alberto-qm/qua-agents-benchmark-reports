#!/usr/bin/env python3
"""The n17 summary page: make_e2_summary.py's page for n17, the 6 Oct 2026 run of tinycal with qwen3.8-27b on n16's 37
single-qubit bring-ups through the chirp bring-up graph with the FPGA-adaptive pi pulse and DRAG
(FluxTunableTransmon_ChirpBringUp, graph 84), beside n16 (the same agent and model on graph 80, 2 Oct) on the same qubits.

    python3 make_n17_summary.py

It can run while n17 is still going: only the cells with a result.json are collected, and n16's column covers the same
qubits, so every comparison is on one set. The layout and every number come from make_n12_summary.py and
make_n12_report.py's collection (via make_e2_summary.py's two-night table and plots), pointed at the n17 and n16 work dirs.
~/qab-runs/n17-LOG.md has the run's libraries, the smoke tests and the user's decisions (3 cells per device, x180 length
writable). The output is page content for a claude.ai artifact.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.dont_write_bytecode = True
import make_e2_summary as e2s  # noqa: E402
import make_n12_summary as summ  # noqa: E402
import make_n13_summary as n13summ  # noqa: E402
from make_fwcmp2_report import CSS, esc, median, minutes  # noqa: E402
from make_fwcmp2_report import pct as fpct  # noqa: E402

rep = summ.rep
QAB = Path.home() / "qab-runs"
OUT = Path(__file__).with_name("2026-10-06-n17-summary.html")
STAMP = (QAB / "n17-stamp.txt").read_text().strip()
N16_STAMP = (QAB / "n16-stamp.txt").read_text().strip()
LABEL = {"old": "n16", "new": "n17"}
N_QUBITS = 37  # n16's set: qolab Q1-Q6, arbel all 21, gilboa qC1-qC5 qD1-qD5

# Graph 84's order (tinycal recipe flux_tunable_1q_chirp_adaptive.md): the power sweep repeats at the sweet spot (step 4),
# T1_chirp precedes the flux map, and T1/T2echo share a step.
RECIPE_84 = [("resonator_identification",), ("resonator_spectroscopy_vs_power",), ("resonator_spectroscopy_vs_flux",),
             ("resonator_spectroscopy_vs_power",), ("qubit_spectroscopy_chirp",), ("T1_chirp",),
             ("qubit_spectroscopy_vs_flux_chirp",), ("readout_frequency_chirp",), ("readout_power_photons_chirp",),
             ("IQ_blobs_chirp",), ("pi_pulse_adaptive",), ("drag_adaptive",), ("T1", "T2echo"),
             ("Randomized_benchmarking",)]


def configure() -> None:
    graph80 = rep.RECIPE["new"]
    rep.NIGHTS = {"old": ("n16",), "new": ("n17",)}
    rep.NIGHT_LABEL = {"old": "n16 (graph 80)", "new": "n17 (graph 84)"}
    rep.CELL_MARKS = ("-tinycal-",)
    rep.RECIPE["old"], rep.RECIPE["new"] = graph80, RECIPE_84
    rep.NOT_MEANINGFUL = {}
    rep.NOT_COMPLETED = {}
    summ.RECOVERY_NIGHTS = (("n16", N16_STAMP), ("n17", STAMP))


def not_completed(rows: list) -> None:
    """The escalation reason tinycal recorded, as the hover text of each n17 cell that did not complete."""
    for r in rows:
        if r["night"] == "new" and r["status"] != "completed":
            doc = json.load(open(QAB / r["work"] / r["cell"] / "result.json"))
            why = (doc["targets"][0].get("escalation_reason") or r["status"]).strip().splitlines()[0][:300]
            rep.NOT_COMPLETED[("new", r["backend"], r["q"])] = why


def context(s: dict, done: int) -> str:
    a, _ = (n13summ._minute(t) for t in s["span"])
    state = (f"interim: {done} of {N_QUBITS} qubits finished at {datetime.now():%H:%M}, the rest still running"
             if done < N_QUBITS else f"all {N_QUBITS} qubits")
    return (f"{a:%-d %b %Y}, from {a:%H:%M} CEST · {state} · tinycal with qwen3.8-27b on the chirp bring-up graph "
            "(chirp spectroscopy, T1 and readout, then the FPGA-adaptive pi pulse and DRAG) · beside n16, the same agent and "
            "model on the usual bring-up graph, 2 Oct")


N17 = summ.Night(
    label="n17", stamp=STAMP, out=OUT,
    title="n17 Summary",
    description="n17, 6 Oct 2026: tinycal and qwen3.8-27b on n16's single-qubit bring-ups through the chirp bring-up graph "
                "with the FPGA-adaptive pi pulse and DRAG, beside n16 on the same qubits: one table, four plots and a "
                "per-qubit check against IQCC.",
    eyebrow="n17 chirp + adaptive graph",
    h1="n17: chirp bring-up with the adaptive pi pulse",
    context=lambda s: "",
    chip_sub=lambda s: f"6 Oct · {s['n']} qubits · graph 84, qwen3.8-27b",
    rb_note=n13summ.rb_note,
    pull="the pull each run started from",
    pull_short="the pull each run started from",
    recovery_intro="One qubit per pair of rows: <b>n16</b>, the 2 Oct run on graph 80, and <b>n17</b>, graph 84 on 6 Oct, "
                   "each against the IQCC pull its own run started from. n17's adaptive pi pulse may lengthen the x180 "
                   "(the profile lets it write the length), and an x180 amplitude at another length is not comparable "
                   "with the lab's: the results table counts those qubits.",
    recovery_sort=("n17", "n16"),
    recovery_sort_text="Rows are sorted by how many of the seven came back in n17, fewest first, then by n16's count.",
    spec="252e28f8ab313894",
    scramble_note="<p class='small muted'>The same spec as n16, with its state-field allowlist; n17's states were pulled and "
                  "scrambled afresh on 6 Oct 12:36.</p>",
)


# ----------------------------------------------------------------------------- results table, two runs
def _lab_x180_len(r) -> float | None:
    try:
        lab = json.load(open(QAB / r["work"] / "source-state/state.json"))["qubits"][r["q"]]
        return float(lab["xy"]["operations"]["x180_DragCosine"]["length"])
    except Exception:  # noqa: BLE001
        return None


def _adaptive(r) -> tuple[int, int]:
    """(pi_pulse_adaptive runs, of them failed or refused) in one n17 cell."""
    runs = [e for e in rep.events(r["run_id"], r["q"]) if e.get("kind") == "tool_call" and e.get("tool") == "run_node"
            and e.get("node") == "pi_pulse_adaptive"]
    return len(runs), sum(1 for e in runs if not (e.get("status") == "completed" and e.get("outcome") == "successful"))


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
        return f"<span class='num'>{a}/{b} ({100 * a / b:.0f}%)</span>" if b else "<span class='muted'>—</span>"

    def adaptive(k, rs, s):
        if k == "old":
            return ("<span class='muted'>— (power Rabi, error amplification, Ramsey and the DRAG sweep)</span>")
        counts = [_adaptive(r) for r in rs]
        n, bad = sum(c[0] for c in counts), sum(c[1] for c in counts)
        once = sum(1 for c in counts if c[0] == 1 and c[1] == 0)
        return (f"<span class='num'>{n} runs · {bad} failed</span><div class='small muted'>median "
                f"{median([c[0] for c in counts]):.0f} per qubit; right first time on {once} of {len(rs)}</div>")

    def lengths(k, rs, s):
        if k == "old":
            return "<span class='muted'>— (read-only)</span>"
        moved = []
        for r in s["done"]:
            lab = _lab_x180_len(r)
            if lab is not None and r["x180_len"] is not None and abs(r["x180_len"] - lab) > 0.5:
                moved.append(f"{r['backend']} {r['q']} {lab:.0f}→{r['x180_len']:.0f} ns")
        return (f"<span class='num'>{len(moved)} of {len(s['done'])}</span>"
                + (f"<div class='small muted'>{esc(', '.join(moved))}</div>" if moved else ""))

    def model_time(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>{minutes(median([r['model_s'] for r in done if r['model_s'] is not None]))}</span>"
                f"<div class='small muted'>median over the {len(done)} completed calibrations</div>")

    def qpu(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>{median([r['qpu_s'] for r in done if r['qpu_s'] is not None]) / 60:.1f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · total {s['qpu_tot'] / 60:.0f} min</div>")

    def wall(k, rs, s):
        done = s["done"]
        w = [r["wall_s"] for r in done if r["wall_s"] is not None]
        par = "3 in flight in all" if k == "old" else "3 in flight per device"
        return (f"<span class='num'>{median(w) / 60:.0f} min</span>"
                f"<div class='small muted'>median over the {len(done)} completed · longest {max(w) / 60:.0f} min · "
                f"{par}</div>")

    def cost(k, rs, s):
        done = s["done"]
        return (f"<span class='num'>${median([r['judge_cost'] for r in done if r['judge_cost'] is not None]):.2f}</span>"
                f"<div class='small muted'>median over the {len(done)} completed, at prices.yaml · total "
                f"${s['judge_cost']:.2f}</div>")

    def runs(k, rs, s):
        return (f"<span class='num'>{s['runs']} · {s['not_ok']} failed or refused</span>"
                f"<div class='small muted'>median {s['runs_med']:.0f} per qubit</div>")

    def overrides(k, rs, s):
        return e2s._share(s["ov_done"], s["ov_rest"], len(s["done"]) or 1, "writes")

    def off_order(k, rs, s):
        return e2s._share(s["oo_done"], s["oo_rest"], len(s["done"]) or 1, "node runs")

    body = [
        ("qubits measured (green: completed with meaningful results; red: not completed, reason on hover; the small letter "
         "is the device)", cells(qubits)),
        ("completed with meaningful results", cells(meaningful)),
        ("valid RB (the fit saw ≥ 1 decay length): runs, median fidelity per gate", cells(valid)),
        ("readout assignment fidelity, median over the runs that measured one", cells(readout)),
        ("judge: scrambled parameters back in the ballpark (readout frequency and amplitude, f₀₁, x180 amplitude)",
         cells(judge)),
        ("adaptive pi pulse (pi_pulse_adaptive): runs, all qubits", cells(adaptive)),
        ("x180 length changed from the lab's (the adaptive pi pulse chose a longer pulse)", cells(lengths)),
        ("model time / calibration", cells(model_time)),
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
    subs = {"old": f"2 Oct · the same {S['old']['n']} qubits · graph 80, qwen3.8-27b", "new": N17.chip_sub(S["new"])}
    head = "<thead><tr><th></th>" + "".join(
        f"<th class='grp'><span class='chip'><i style='background:{rep.SERIES[k]}'></i>{esc(LABEL[k])}</span>"
        f"<br><span class='small muted'>{esc(subs[k])}</span></th>" for k in ("old", "new")) + "</tr></thead>"
    return f"<div class='scroll'><table class='grid pivot sum sum2'>{head}<tbody>{trs}</tbody></table></div>"


# ----------------------------------------------------------------------------- the adaptive pi pulse's readout model
def _first_04f_readout(r) -> tuple[float | None, int, int]:
    """(IQ blobs' readout fidelity before the first pi_pulse_adaptive, its runs, of them failed) in one n17 cell."""
    readout, runs, bad = None, 0, 0
    for e in rep.events(r["run_id"], r["q"]):
        if e.get("kind") != "tool_call" or e.get("tool") != "run_node":
            continue
        if e.get("node") == "IQ_blobs_chirp" and e.get("outcome") == "successful" and readout is None and not runs:
            v = (e.get("numerics") or {}).get("readout_fidelity")
            readout = v.get("value") if isinstance(v, dict) else v
        if e.get("node") == "pi_pulse_adaptive":
            runs += 1
            bad += 0 if (e.get("status") == "completed" and e.get("outcome") == "successful") else 1
    return readout, runs, bad


def readout_model_section(rows: dict) -> str:
    """Why the adaptive pi pulse failed on some qubits: it ran with a perfect-readout model."""
    cells = [(r, *_first_04f_readout(r)) for r in rows["new"]]
    cells = [c for c in cells if c[2]]
    lo = [c for c in cells if c[1] is not None and c[1] < 92]
    hi = [c for c in cells if c[1] is not None and c[1] >= 92]

    def line(cs):
        first = sum(1 for c in cs if c[3] == 0)
        return f"{first} of {len(cs)} right first time, {sum(c[3] for c in cs)} failed runs"

    n16 = {(r["backend"], r["q"]): r for r in rows["old"]}

    def qpu(cs, src):
        v = [(src.get((c[0]["backend"], c[0]["q"])) or {}).get("qpu_s") if src else c[0]["qpu_s"] for c in cs]
        v = [x for x in v if x is not None]
        return f"{median(v) / 60:.1f} min" if v else "—"
    clean = [c for c in cells if c[2] == 1 and c[3] == 0]
    retried = [c for c in cells if c not in clean]
    return (
        "<h3 class='scr'>The adaptive pi pulse ran with a perfect-readout model</h3>"
        "<p class='small muted cap2'>04f and 04g weigh every shot by the readout's confusion matrix "
        "(<span class='mono'>resonator.confusion_matrix</span>) and take a missing one as perfect readout. In these runs it "
        "was missing on every qubit: the scramble clears it, and the matrix IQ blobs measures never reached the state "
        "(qualibrate_ai on the local integration branch records no list values, and tinycal writes only scalars; qua-agents "
        "main records small lists by element since 6a2df30, but not where the state holds null, as it does after the "
        "scramble). With readout below ~92 % the filter grows certain on the wrong Ramsey fringe and its own likelihood "
        "check rejects the run; the agent retries. A fix (tinycal writes list elements, the scramble keeps the matrix's "
        "shape, 04f/04g refuse without a matrix) is pending; these numbers are with the defect.</p>"
        "<div class='scroll'><table class='grid small scr'><thead><tr><th>IQ blobs' readout fidelity before 04f</th>"
        "<th>qubits</th><th>pi_pulse_adaptive</th></tr></thead><tbody>"
        f"<tr><td>≥ 92 %</td><td class='num'>{len(hi)}</td><td>{line(hi)}</td></tr>"
        f"<tr><td>&lt; 92 %</td><td class='num'>{len(lo)}</td><td>{line(lo)}</td></tr>"
        "</tbody></table></div>"
        f"<p class='small muted'>QPU median per qubit: {qpu(clean, None)} on the {len(clean)} qubits where 04f was right "
        f"first time (n16 on the same qubits {qpu(clean, n16)}), {qpu(retried, None)} on the {len(retried)} that retried "
        f"(n16 {qpu(retried, n16)}). A clean 04f run costs ~30 s of QPU, most of it the program's compilation (~25 s on "
        "QOP 3.8, from the node's README): the estimator runs on the FPGA. "
        + _q3_note(rows) + "</p>")


def _q3_note(rows: dict) -> str:
    q3 = next((r for r in rows["new"] if (r["backend"], r["q"]) == ("qolab", "Q3")), None)
    if q3 is None:
        return ""
    old = next((r for r in rows["old"] if (r["backend"], r["q"]) == ("qolab", "Q3")), None)
    runs = _first_04f_readout(q3)[1]
    vs = f" against n16's {100 * old['rb']:.3f} %" if old and old["rb"] else ""
    return (f"qolab Q3 ran 04f {runs} times and ended on a 5π pulse (0.105 at 192 ns, 20 V·ns against the lab's 4.1), "
            f"which IQ blobs prepared by that pulse cannot tell from π: RB {100 * q3['rb']:.3f} %{vs}.")


# ----------------------------------------------------------------------------- plots: n17 alone, and n17 against n16
def plots(rows: list) -> str:
    """make_e2_summary.plots with n17 in e2's place."""
    saved = e2s.E2, e2s.LABEL
    e2s.LABEL = LABEL
    try:
        html = e2s.plots(rows)
    finally:
        e2s.E2, e2s.LABEL = saved
    # e2's captions name e2 and its pull; the series is n17 here
    return (html.replace("e2's completed runs", "n17's completed runs")
            .replace("the pull e2 started from (all three backends pulled 6 Oct 03:53, arbel and gilboa on square readout "
                     "with ≥ 3000 ns depletion)",
                     "the pull n17 started from (all three backends pulled 6 Oct 12:36, arbel and gilboa on square readout "
                     "with ≥ 3000 ns depletion)")
            .replace("the pull e2 started from", "the pull n17 started from")
            .replace("e2's error per gate against n16's on the same qubit (2 Oct, tinycal with qwen3.8-27b); below the "
                     "diagonal e2 is better", "n17's error per gate against n16's on the same qubit (2 Oct, graph 80); "
                     "below the diagonal n17 is better")
            .replace("e2's measured gate error", "n17's measured gate error")
            .replace("e2&#x27;s measured gate error", "n17&#x27;s measured gate error")
            .replace("(stuck in e2)", "(not completed in n17)").replace("(no valid RB in e2)", "(no valid RB in n17)")
            .replace("· e2 ", "· n17 "))


def build() -> None:
    configure()
    rows = rep.collect()
    new = [r for r in rows if r["night"] == "new"]
    keys = {(r["backend"], r["q"]) for r in new}
    rows = new + [r for r in rows if r["night"] == "old" and (r["backend"], r["q"]) in keys]
    not_completed(rows)
    by = {k: [r for r in rows if r["night"] == k] for k in ("old", "new")}
    assert {r["run"] for r in by["old"]} == {"n16"} and {r["run"] for r in by["new"]} == {"n17"}, "wrong work dirs collected"
    S = {k: rep.night_stats(v) for k, v in by.items()}
    now = datetime.now().strftime("%d %b %Y %H:%M")
    page = f"""<title>{esc(N17.title)}</title>
<meta name="description" content="{esc(N17.description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}{rep.EXTRA_CSS}{summ.SUMMARY_CSS}{e2s.SUMMARY2_CSS}</style>
<div class="page">
<div class="eyebrow">qua-agents benchmark · {esc(N17.eyebrow)} · generated {esc(now)}</div>
<h1 style="margin-top:8px">{esc(N17.h1)}</h1>
<p class="context">{esc(context(S['new'], len(new)))}</p>
{summ.scramble_table(N17)}
<h3 class='scr'>Results</h3>
{results_table(by, S)}
{summ.legend(by['new'], N17)}
<p class="small muted uinote">Click a qubit, here or on a plot, to open its run in tinycal's run viewer (<span class="mono">tinycal ui</span> on this machine, port 8765).</p>
{readout_model_section(by)}
{plots(rows)}
{summ.recovery_table(by['new'], N17)}
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
