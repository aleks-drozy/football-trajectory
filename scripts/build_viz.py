"""Build the self-contained explorer page from the template + result JSONs.

viz/template.html is the source of truth for markup/CSS/JS; this injects
results/trajectories.json and results/legend_scenarios.json into it and writes
viz/trajectory_explorer.html. The built file is what gets served and published —
never edit it by hand.

Usage:
    python scripts/build_viz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def inject(html: str, placeholder: str, payload_path: Path) -> str:
    payload = payload_path.read_text(encoding="utf-8")
    json.loads(payload)  # fail loudly on a corrupt results file
    # '</' inside a JSON string would end the <script> block early.
    return html.replace(placeholder, payload.replace("</", "<\\/"))


def main() -> int:
    template = (ROOT / "viz" / "template.html").read_text(encoding="utf-8")
    for ph in ("__TRAJECTORIES__", "__LEGENDS__"):
        if ph not in template:
            raise SystemExit(f"template is missing placeholder {ph}")

    html = inject(template, "__TRAJECTORIES__", ROOT / "results" / "trajectories.json")
    html = inject(html, "__LEGENDS__", ROOT / "results" / "legend_scenarios.json")

    out = ROOT / "viz" / "trajectory_explorer.html"
    out.write_text(html, encoding="utf-8")
    print(f"-> {out} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
