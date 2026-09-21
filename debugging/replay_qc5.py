"""Replay the gilboa qC5 flux-map commit turn with and without a recipe note.

Rebuilds the exact context the model saw (system prompt, tools, messages incl. the
figures still in the window at that turn) and asks the model again N times per variant.
"""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tinycal.client import image_block, text_block
from tinycal.loop import make_client
from tinycal.profile import ModelConfig
from tinycal.tools import TOOLS

RUNS = Path.home() / "code/QM/tinycal/runs"
CONTEXTS = {
    "original (turn 12)": (RUNS / "n4_gilboa_openrouter_20260920-1712/qC5", 23),
    "rerun (turn 14)": (RUNS / "n4_gilboa_openrouter-R_20260920-1712/qC5", 27),
}
NOTE = (
    "\nEvery node that fits the data draws its fit and the point it proposes (a sweet spot,\n"
    "a resonance, an optimum) on its figure. Before committing a proposal, look at the\n"
    "figure and check that the fit actually follows the data and that the marked point\n"
    "sits where physics says it should - the fit can be wrong, and a proposal from a poor\n"
    "fit is not evidence. If the marked point does not make physical sense, do not\n"
    "commit it; re-measure so that the feature is unambiguous instead.\n"
)
ANCHOR = "When a node fails or its result contradicts an earlier one"
POINTER = re.compile(r"^\[plot (\S+) removed from context; saved at (.+)\]$")


def restore(block):
    if block["type"] == "text":
        m = POINTER.match(block["text"])
        if m:
            return image_block(Path(m.group(2)).read_bytes(), m.group(1), m.group(2))
        return block
    if block["type"] == "tool_result":
        return {**block, "content": [restore(b) for b in block["content"]]}
    return block


def load_context(target_dir, n_messages):
    payload = json.loads((target_dir / "transcript.json").read_text())[:n_messages]
    messages = [{"role": m["role"], "content": [restore(b) for b in m["content"]]} for m in payload]
    system = (target_dir / "system_prompt.md").read_text()
    model = json.loads((target_dir / "events.jsonl").open().readline())["model"]
    return system, messages, ModelConfig(**model)


def classify(reply):
    calls = reply.tool_calls
    if not calls:
        return "no tool call", reply.text[:200]
    c = calls[0]
    if c.name == "write_state":
        paths = {u["path"]: u["value"] for u in c.arguments.get("updates", [])}
        jo = [v for p, v in paths.items() if p.endswith("joint_offset")]
        if jo:
            return f"commits apex {jo[0]} V", c.arguments.get("note", "")[:300]
        return "commits without joint_offset", c.arguments.get("note", "")[:300]
    if c.name == "run_node":
        return f"re-measures {c.arguments.get('node')} {json.dumps(c.arguments.get('parameters', {}))}", c.arguments.get("note", "")[:300]
    return f"{c.name} {json.dumps(c.arguments)[:120]}", reply.text[:200]


def one(system, messages, config, tag):
    client = make_client(config)
    t0 = time.time()
    reply = client.complete(system, TOOLS, messages)
    verdict, note = classify(reply)
    return {"tag": tag, "verdict": verdict, "note": note, "text": reply.text[:600], "stop": reply.stop_reason,
            "in": reply.usage.input_tokens, "out": reply.usage.output_tokens, "s": round(time.time() - t0, 1)}


def main(n=5, workers=4):
    jobs = []
    for cname, (tdir, nmsg) in CONTEXTS.items():
        system, messages, config = load_context(tdir, nmsg)
        assert ANCHOR in system
        with_note = system.replace(ANCHOR, NOTE + "\n" + ANCHOR, 1)
        for variant, sysm in (("without note", system), ("with note", with_note)):
            for i in range(n):
                jobs.append((sysm, messages, config, f"{cname} | {variant} | #{i+1}"))
    out = Path.home() / "qab-runs/replay_qc5_results.jsonl"
    with ThreadPoolExecutor(workers) as ex, out.open("a") as fh:
        for r in ex.map(lambda j: one(*j), jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{r['tag']:45s} {r['verdict']}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
