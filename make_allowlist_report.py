#!/usr/bin/env python3
"""The state-field allowlist report: the benchmark scramble's state_fields rules as adopted on 2 Oct 2026, the leaks
they close, what they did to the n16 prep, and when IQCC's schema changed.

    ~/code/QM/tinycal/.venv/bin/python make_allowlist_report.py      # needs PyYAML

Reads the rules from qua-agents-benchmark's workloads/decalibrate_chip.yaml (state_fields), the state_fields.json each
n16 prep wrote beside its edit_plan.json, and every pulled state archived under ~/qab-runs (for the dates fields
appeared). Classification here mirrors qua_agents_benchmark.scramble.state_fields: first match wins, `*` one segment,
`**` any number.
"""
from __future__ import annotations

import datetime
import glob
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CSS = '\n:root {\n  --ground:#f4f6f5; --surface:#ffffff; --ink:#1b2422; --ink-2:#3c4a46; --muted:#66756f; --line:#d6dedb; --line-2:#e7ecea;\n  --accent:#1d6b5f; --accent-ink:#165548;\n  --keep:#55635f; --keep-bg:#e6ebe9; --scr:#2c5aa0; --scr-bg:#e3ebf7; --res:#8a5a0b; --res-bg:#f6ecd8; --rem:#a23a2a; --rem-bg:#f6e2de;\n  --flag:#7a4bb0; --flag-bg:#efe6f8; --ok:#1d6b5f; --bad:#a23a2a;\n}\n@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {\n  color-scheme:dark; --ground:#111615; --surface:#171e1c; --ink:#e2e9e6; --ink-2:#c3cfcb; --muted:#8c9b96; --line:#2c3633; --line-2:#222b28;\n  --accent:#5fc0ad; --accent-ink:#7fd3c2;\n  --keep:#aab6b2; --keep-bg:#25302d; --scr:#8fb2ec; --scr-bg:#1d2a40; --res:#e2b45e; --res-bg:#3a2e17; --rem:#ef8f7f; --rem-bg:#3d221d;\n  --flag:#c6a4ef; --flag-bg:#2f2440; --ok:#5fc0ad; --bad:#ef8f7f; } }\n:root[data-theme="dark"] {\n  color-scheme:dark; --ground:#111615; --surface:#171e1c; --ink:#e2e9e6; --ink-2:#c3cfcb; --muted:#8c9b96; --line:#2c3633; --line-2:#222b28;\n  --accent:#5fc0ad; --accent-ink:#7fd3c2;\n  --keep:#aab6b2; --keep-bg:#25302d; --scr:#8fb2ec; --scr-bg:#1d2a40; --res:#e2b45e; --res-bg:#3a2e17; --rem:#ef8f7f; --rem-bg:#3d221d;\n  --flag:#c6a4ef; --flag-bg:#2f2440; --ok:#5fc0ad; --bad:#ef8f7f; }\nbody { background:var(--ground); color:var(--ink); font:15px/1.55 "IBM Plex Sans", system-ui, sans-serif; padding-inline:16px; padding-block:28px 64px; }\n.page { max-width:1120px; margin:0 auto; display:flex; flex-direction:column; gap:30px; }\nh1, h2 { font-family:"IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif; font-weight:600; text-wrap:balance; margin:0; }\nh1 { font-size:30px; letter-spacing:-0.01em; }\nh2 { font-size:20px; margin-bottom:8px; }\np { margin:0 0 10px; max-width:72ch; }\n.eyebrow { font:500 12px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }\n.lede { color:var(--ink-2); font-size:16px; }\n.mono { font-family:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; }\n.small { font-size:12.5px; color:var(--muted); }\n.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }\n.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }\n.panel { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:18px 20px; }\n.grid2 { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:16px; }\n@media (max-width:820px) { .grid2 { grid-template-columns:1fr; } }\nul.tight { margin:6px 0 0; padding-left:18px; } ul.tight li { margin:4px 0; }\n.chip { display:inline-block; font:500 11.5px/1.6 "IBM Plex Mono", ui-monospace, monospace; padding:0 7px; border-radius:4px; }\n.chip.keep { color:var(--keep); background:var(--keep-bg); } .chip.scramble { color:var(--scr); background:var(--scr-bg); }\n.chip.reset { color:var(--res); background:var(--res-bg); } .chip.remove { color:var(--rem); background:var(--rem-bg); }\n.chip.refuse { color:var(--bad); background:var(--rem-bg); }\n.flag { display:inline-block; margin-left:6px; font:500 11px/1.6 "IBM Plex Sans", sans-serif; color:var(--flag); background:var(--flag-bg); padding:0 6px; border-radius:4px; }\n.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--surface); }\ntable { border-collapse:collapse; width:100%; }\nth, td { text-align:left; vertical-align:top; padding:7px 10px; border-bottom:1px solid var(--line-2); }\nthead th { font-size:12px; font-weight:600; color:var(--muted); background:var(--surface); border-bottom:1px solid var(--line); }\ntr.sec th { background:var(--ground); font-size:13px; font-weight:600; color:var(--ink-2); border-top:1px solid var(--line); }\ntr.sec .n { font-weight:400; color:var(--muted); margin-left:6px; }\ntr.rv td { background:color-mix(in srgb, var(--flag-bg) 45%, transparent); }\ntd.path { min-width:300px; word-break:break-all; } td.to { min-width:90px; } td.why { min-width:300px; font-size:13.5px; color:var(--ink-2); }\ntable.rules { min-width:980px; } table.tl { min-width:760px; } table.cur { min-width:720px; }\n.decide { display:flex; flex-direction:column; gap:12px; }\n.decide > div { border-left:3px solid var(--flag); padding:2px 0 2px 12px; }\n.decide b { display:block; margin-bottom:2px; }\nblockquote { margin:8px 0; padding:8px 12px; border-left:3px solid var(--line); color:var(--ink-2); font-size:14px; background:var(--ground); }\na { color:var(--accent-ink); }\n:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }\n'

HERE = Path(__file__).parent
OUT = HERE / "2026-10-02-state-field-allowlist.html"
BENCH = Path.home() / "code/QM/qua-agents-benchmark"
RUNS = Path.home() / "qab-runs"
SPEC_FILE = BENCH / "workloads/decalibrate_chip.yaml"
SPEC = yaml.safe_load(SPEC_FILE.read_text())
RULES = SPEC["state_fields"]["rules"]
N16 = (RUNS / "n16-stamp.txt").read_text().strip()
esc = html.escape


def leaves(x, p=""):
    if isinstance(x, dict):
        if not x:
            yield p, {}
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list) and any(isinstance(i, (dict, list)) for i in x):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}/[{i}]")
    else:
        yield p, x


def _match(pat, segs):
    if not pat:
        return not segs
    if pat[0] == "**":
        return any(_match(pat[1:], segs[i:]) for i in range(len(segs) + 1))
    return bool(segs) and pat[0] in ("*", segs[0]) and _match(pat[1:], segs[1:])


COMPILED = [(i, r["match"].strip("/").split("/")) for i, r in enumerate(RULES)]


def classify(path):
    segs = path.strip("/").split("/")
    for i, pat in COMPILED:
        if _match(pat, segs):
            return i
    return None


def pattern(path):
    s = path.strip("/").split("/")
    for i in range(len(s)):
        if i > 0 and s[i - 1] in ("qubits", "qubit_pairs", "twpas") and (i == 1 or s[0] == "wiring"):
            s[i] = "<name>"
        elif re.fullmatch(r"\[\d+\]", s[i]):
            s[i] = "[i]"
    if s and s[0] == "ports":
        s[2:5] = ["*"] * len(s[2:5])
    return "/" + "/".join(s)


def load(d):
    return {**json.load(open(d / "state.json")), **(json.load(open(d / "wiring.json")) if (d / "wiring.json").exists() else {})}


# ---------------------------------------------------------------- the n16 prep: what the rules did in production
prep_rows, hits = [], Counter()
for be in ("qolab", "arbel", "gilboa"):
    work = RUNS / f"n16-{be}-{N16}"
    sf = json.load(open(work / "state_fields.json"))
    src = load(work / "source-state")
    unmatched = sum(1 for p, _ in leaves(src) if classify(p) is None)
    for p, _ in leaves(src):
        i = classify(p)
        if i is not None:
            hits[i] += 1
    scr = json.dumps(json.load(open(work / "scrambled-state/state.json")))
    left = sum(scr.count(k) for k in ("target_RF_frequency", "f_01_ground", "f_01_excited"))
    c = sf["counts"]
    prep_rows.append(
        f"<tr><td>{be}</td>" + "".join(f"<td class='num'>{c.get(k, 0):,}</td>" for k in ("keep", "scramble", "reset", "remove"))
        + f"<td class='num'>{len(sf['removed'])}</td><td class='num {'ok' if not unmatched else 'bad'}'>{unmatched}</td>"
        f"<td class='num {'ok' if not left else 'bad'}'>{left}</td></tr>")

# ---------------------------------------------------------------- the archive: when IQCC added fields
dirs = set()
for p in glob.glob(str(RUNS / "*/pulled-raw/state.json")) + glob.glob(str(RUNS / "*/source-state/state.json")) + \
        glob.glob(str(RUNS / "state-*/*/state.json")):
    if Path(p).parent.joinpath("wiring.json").exists():
        dirs.add(Path(p).parent)
appeared, n_states = {}, 0
for d in sorted(dirs, key=lambda x: os.path.getmtime(x / "state.json")):
    try:
        doc = load(d)
    except Exception:  # noqa: BLE001
        continue
    n_states += 1
    when = datetime.datetime.fromtimestamp(os.path.getmtime(d / "state.json")).strftime("%Y-%m-%d")
    names = set(doc.get("qubits") or {})
    be = "qolab" if "Q1" in names else "arbel" if "qA6" in names else "lucy" if "qA" in names else "gilboa" if "qC1" in names else "?"
    for path, _ in leaves(doc):
        appeared.setdefault((be, pattern(path)), when)
starts = {}
for (be, _), when in appeared.items():
    starts[be] = min(starts.get(be, when), when)
by_day = defaultdict(lambda: defaultdict(list))
for (be, pat), when in appeared.items():
    if be != "?" and when > starts[be]:
        by_day[when][be].append(pat)


def chip(c):
    return f"<span class='chip {c}'>{esc(c)}</span>"


def rule_class(pat):
    i = classify(pat.replace("<name>", "qX").replace("[i]", "[0]"))
    return RULES[i]["class"] if i is not None else "refuse"


tl = []
for day in sorted(by_day):
    for be, pats in sorted(by_day[day].items()):
        classes = Counter(rule_class(p) for p in pats)
        ex = sorted(pats, key=len)[:3]
        tl.append(f"<tr><td class='mono'>{day}</td><td>{esc(be)}</td><td class='num'>{len(pats)}</td>"
                  f"<td class='mono small'>{'<br>'.join(esc(e) for e in ex)}{'<br>…' if len(pats) > 3 else ''}</td>"
                  f"<td>{' '.join(chip(c) + f'<span class=small> ×{n}</span>' for c, n in sorted(classes.items()))}</td></tr>")

# ---------------------------------------------------------------- the rules
SECTIONS = ["Document, wiring and hardware", "Qubits: scrambled and graded by the edit table", "Qubits: reset by the edit table",
            "Qubits: new resets, removals and scalings", "Qubits: structure, design values, pulse shapes", "Pairs (2Q gates)"]


def section_of(r):
    m = r["match"]
    if not m.startswith(("/qubits", "/qubit_pairs")):
        return 0
    if m.startswith("/qubit_pairs"):
        return 5
    if r["class"] == "scramble" and "spec" in r:
        return 1
    if r["class"] == "reset" and "spec" in r:
        return 2
    return 3 if r["class"] in ("reset", "remove", "scramble") else 4


def target(r):
    if r["class"] == "reset":
        to = r["to"]
        return esc("{}" if to == {} else "null" if to is None else str(to))
    if r["class"] == "scramble":
        return f"× {r['factor']}" if "factor" in r else "edit table"
    return ""


def why(r):
    if "spec" in r and "why" not in r:
        return f"edit table: <span class='mono'>{esc(r['spec'])}</span>"
    return esc(str(r.get("why", "")))


rows_by = defaultdict(list)
for i, r in enumerate(RULES):
    rows_by[section_of(r)].append((i, r))
rule_rows = []
for si, title in enumerate(SECTIONS):
    rs = rows_by.get(si, [])
    if not rs:
        continue
    rule_rows.append(f"<tr class='sec'><th colspan='5'>{esc(title)} <span class='n'>{len(rs)} rules</span></th></tr>")
    for i, r in rs:
        flag = "<span class='flag' title='a judgement call: see Still open'>open call</span>" if r.get("review") else ""
        rule_rows.append(f"<tr{' class=rv' if r.get('review') else ''}><td class='mono path'>{esc(r['match'])}</td>"
                         f"<td>{chip(r['class'])}{flag}</td><td class='mono to'>{target(r)}</td>"
                         f"<td class='num'>{hits.get(i, 0):,}</td><td class='why'>{why(r)}</td></tr>")

counts = Counter(r["class"] for r in RULES)
n_review = sum(1 for r in RULES if r.get("review"))
spec_digest = json.load(open(RUNS / f"n16-arbel-{N16}/state_fields.json"))["spec_digest"]
n_dates = len(by_day)

page = f"""<title>State Field Allowlist</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>__CSS__</style>
<div class="page">
<header>
  <div class="eyebrow">qua-agents benchmark · scramble · adopted 2 Oct 2026 · benchmark 600b096 · spec {spec_digest}</div>
  <h1>State Field Allowlist</h1>
  <p class="lede">From 2 Oct the benchmark's scramble accounts for every field of a pulled IQCC state: each one is kept,
  scrambled, reset or removed by a rule in <span class="mono">workloads/decalibrate_chip.yaml</span>, and a field no rule covers
  stops the scramble. Before, every field the edit table did not name reached the agent with the lab's value in it.</p>
</header>

<section class="grid2">
  <div class="panel">
    <h2>What leaked, and who read it</h2>
    <ul class="tight">
      <li><span class="mono">extras.target_RF_frequency</span>: the lab's f₀₁ within 0.3 MHz on arbel qA6, qB3, qD1, unscrambled
        since 14 Sep. IQCC's own flux node treats it as “park this qubit here”
        (<span class="mono">iqcc-calibration · 09_ramsey_vs_flux_calibration</span>), and their 6 Sep state flagged the same three
        <span class="mono">at_sweet_spot: False</span>. Read in two n14 and two n15 runs.</li>
      <li><span class="mono">resonator.extras.f_01_ground / f_01_excited</span>: the lab's readout-resonator frequency, which the
        judge grades to ±1 MHz, on 19–20 arbel qubits since 30 Sep. Read in one n14 and three n15 runs.</li>
    </ul>
    <p style="margin-top:10px">They misled as well as leaked. n15 qC5 took the resonator value for the qubit:</p>
    <blockquote>“the resonator extras in the state show f_01_ground: 7.661 GHz and f_01_excited: 7.660 GHz, confirming the qubit IS
    at ~7.66 GHz, right next to the resonator”</blockquote>
    <p>n15 qD1 centred its search on <span class="mono">target_RF_frequency</span>, 368 MHz below the sweet spot it had just biased
    to, and never found its line.</p>
  </div>
  <div class="panel">
    <h2>What the scramble does now</h2>
    <ul class="tight">
      <li>Every leaf of <span class="mono">state.json</span> and <span class="mono">wiring.json</span> must match one of
        {len(RULES)} rules ({counts['keep']} keep, {counts['scramble']} scramble, {counts['reset']} reset, {counts['remove']} remove);
        the first match decides. An unmatched field refuses the scramble and is named, in <span class="mono">--dry-run</span>
        too.</li>
      <li>The rules and the edit table are checked against each other: every field the table edits must be named by a rule, and
        no rule may name a field the table does not edit.</li>
      <li>Removals and resets are applied before Quam loads the state; a removal that would leave a reference dangling is
        refused. <span class="mono">state_fields.json</span>, written beside <span class="mono">edit_plan.json</span>, lists every
        node removed, reset and scaled.</li>
      <li>The spec hash moved from <span class="mono">f47053f145f74c8c</span> to <span class="mono">{spec_digest}</span>, so
        the judge refuses to compare a run scrambled before with one scrambled after. Earlier arbel runs started with the leaks
        above in their state.</li>
    </ul>
  </div>
</section>

<section>
  <h2>First use: the n16 prep, 2 Oct 16:32</h2>
  <p>All three backends pulled fresh and scrambled under the new spec. Counts are fields (leaf values) by class; “removed” is
  nodes dropped; the last two columns are fields no rule matched and copies of the three leaked fields left in the
  scrambled state.</p>
  <div class="scroll"><table class="cur">
    <thead><tr><th>backend</th><th class="num">keep</th><th class="num">scramble</th><th class="num">reset</th><th class="num">remove</th><th class="num">nodes removed</th><th class="num">unmatched</th><th class="num">leaks left</th></tr></thead>
    <tbody>{''.join(prep_rows)}</tbody></table></div>
</section>

<section class="panel">
  <h2>Still open</h2>
  <p>{n_review} rules are judgement calls, marked <span class="flag">open call</span> below. Two of them bear on what is
  measured:</p>
  <div class="decide">
    <div><b>The off-sweet-spot qubits.</b> Removing <span class="mono">target_RF_frequency</span> stops the leak, but the recipe
      still sends qA6, qB3 and qD1 to their sweet spots while the judge grades them at the frequency IQCC parks them at. A correct
      calibration fails the f₀₁ window on all three. That is a judge decision for those three qubits, which the allowlist leaves
      open.</div>
    <div><b>Pairs and CZ pulses kept wholesale.</b> <span class="mono">/qubit_pairs/**</span> and
      <span class="mono">/qubits/*/z/operations/**</span> pass whole, apart from fidelity and confusion records. Nothing in the 1Q
      bring-up reads or grades them, but a <span class="mono">**</span> rule admits new fields there unreviewed: gilboa's 100 CZ
      pulse fields of 9 Sep would have passed.</div>
    <div><b>What no allowlist catches:</b> a field whose meaning changes under the same name. n14's arbel
      <span class="mono">depletion_time</span>, sized for the self-depleting Drachma pulse and used with the square one, was that
      kind; it is handled by the square-readout prep.</div>
  </div>
</section>

<section>
  <h2>When IQCC changed its state</h2>
  <p>Fields that appeared after each backend's first archived pull, over {n_states} pulls since 6 Sep, and how the adopted rules
  classify them. With the allowlist in place, each such date stops the next prep until someone classifies the new field.</p>
  <div class="scroll"><table class="tl">
    <thead><tr><th>first seen</th><th>backend</th><th class="num">fields</th><th>examples</th><th>rule today</th></tr></thead>
    <tbody>{''.join(tl)}</tbody></table></div>
  <p class="small" style="margin-top:8px"><span class="chip refuse">refuse</span> is deliberate: the Drachma readout sitting at
  <span class="mono">operations.readout</span> must stop the scramble until the square-readout prep has run.</p>
</section>

<section>
  <h2>The rules</h2>
  <p>As committed in <span class="mono">workloads/decalibrate_chip.yaml</span>. “Fields” counts the leaf values each rule
  matched in the three n16 sources.</p>
  <div class="scroll"><table class="rules">
    <thead><tr><th>path</th><th>class</th><th>to</th><th class="num">fields</th><th>why</th></tr></thead>
    <tbody>{''.join(rule_rows)}</tbody></table></div>
</section>

<footer class="small">Code: qua-agents-benchmark 600b096, <span class="mono">src/qua_agents_benchmark/scramble/state_fields.py</span>
  and the <span class="mono">state_fields</span> section of <span class="mono">workloads/decalibrate_chip.yaml</span>. Generated by
  <span class="mono">make_allowlist_report.py</span>.</footer>
</div>
"""
OUT.write_text(page.replace("__CSS__", CSS))
print("wrote", OUT, f"{len(RULES)} rules, {n_states} archived states, {n_dates} dates with new fields")
