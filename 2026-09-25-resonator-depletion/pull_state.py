"""Pull each backend's latest IQCC cloud state (state + wiring) into ~/qab-runs/state-<stamp>/<backend>/ (read-only).

usage: pull_state.py <stamp> [backend ...]   Writes pulled.json with the document ids and timestamps.
"""
import json
import sys
from pathlib import Path

from iqcc_cloud_client import IQCC_Cloud

stamp = sys.argv[1]
backends = sys.argv[2:] or ["arbel", "qolab", "gilboa"]
for be in backends:
    out = Path.home() / f"qab-runs/state-{stamp}/{be}"
    out.mkdir(parents=True, exist_ok=True)
    cloud = IQCC_Cloud(quantum_computer_backend=be)
    ids = {}
    for name in ("wiring", "state"):
        doc = cloud.state.get_latest(name)
        (out / f"{name}.json").write_text(json.dumps(doc.data, indent=4) + "\n")
        ids[name] = {"id": str(getattr(doc, "id", "")), "timestamp": str(getattr(doc, "timestamp", ""))}
    (out / "pulled.json").write_text(json.dumps(ids, indent=1))
    print(be, ids)
