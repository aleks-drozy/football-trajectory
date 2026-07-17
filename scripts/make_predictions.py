"""Freeze the model's 2026-27 predictions into a public, dated ledger.

The point of this file is the git timestamp on its output. A model that only
backtests can always be accused of hindsight; this one commits its next-season
calls before the season is played and will be scored in public when it ends
(scripts/score_predictions.py — the scoring rules are fixed now, not after).

Reads results/trajectories.json (the v2 simulation) and writes:
- results/predictions_2026-27.json  (machine-readable, scored next year)
- PREDICTIONS.md                    (the human ledger)

Usage:
    python scripts/make_predictions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FROZEN_ON = "2026-07-17"
SEASON = "2026-27"


def main() -> int:
    traj = json.loads((RESULTS / "trajectories.json").read_text(encoding="utf-8"))

    preds = []
    for p in traj["players"]:
        b = p["bands"]
        preds.append(
            {
                "player": p["player"],
                "born": p["born"],
                "age_in_2026_27": p["age"] + 1,
                "club_when_frozen": p["team"],
                "npg": {
                    "median": b["npg"]["p50"][0],
                    "p10": b["npg"]["p10"][0],
                    "p90": b["npg"]["p90"][0],
                },
                "assists": {
                    "median": b["assists"]["p50"][0],
                    "p10": b["assists"]["p10"][0],
                    "p90": b["assists"]["p90"][0],
                },
                "minutes_median": b["minutes"]["p50"][0],
            }
        )

    yamal = next(p for p in preds if p["player"] == "Lamine Yamal")

    out = {
        "frozen_on": FROZEN_ON,
        "season": SEASON,
        "model_version": traj["model_version"],
        "n_sims": traj["n_sims"],
        "scope": "Big-5 domestic league only; a player outside the Big 5 scores 0 by definition",
        "scoring_protocol": {
            "when": "after the 2026-27 Big-5 season ends (fetch the season, rebuild the panel, run scripts/score_predictions.py)",
            "point_score": "MAE of median npG across all players, compared against the persistence baseline (player repeats his 2025-26 npG)",
            "interval_score": "share of actual npG inside [p10, p90]; nominal 80%",
            "headline_claim": "Yamal scores FEWER non-penalty league goals in 2026-27 than his 13 of 2025-26",
            "no_exclusions": "every player below is scored; transfers out of the Big 5 count as 0, injuries count as what happened",
        },
        "predictions": preds,
    }
    (RESULTS / "predictions_2026-27.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# Predictions ledger — 2026-27 season",
        "",
        f"**Frozen {FROZEN_ON}**, before the 2026-27 season. The git history of this",
        "file is the proof. Scored publicly when the season ends, by the rules below,",
        "written down now so they cannot drift after the fact.",
        "",
        "The model behind these numbers failed its own calibration gate (its 80%",
        "intervals are honestly too wide — see [WRITEUP.md](WRITEUP.md)), so the",
        "intervals here are the test as much as the medians: nominal 80% coverage is",
        "itself a prediction being scored.",
        "",
        "## The headline claim",
        "",
        "**Lamine Yamal scores fewer non-penalty league goals in 2026-27 than the 13",
        "he scored in 2025-26.** Median call: "
        f"**{yamal['npg']['median']:.0f}** [{yamal['npg']['p10']:.0f}-{yamal['npg']['p90']:.0f}].",
        "Not because he isn't brilliant, but because every hot teenage season in the",
        "data regressed, and the model refuses to make him the exception in advance.",
        "",
        "## All sixteen calls (non-penalty goals, Big-5 league only)",
        "",
        "| Player | Age | Club when frozen | npG median | 80% interval | Assists | Minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in preds:
        lines.append(
            f"| {p['player']} | {p['age_in_2026_27']} | {p['club_when_frozen']} "
            f"| **{p['npg']['median']:.0f}** | {p['npg']['p10']:.0f}-{p['npg']['p90']:.0f} "
            f"| {p['assists']['median']:.0f} [{p['assists']['p10']:.0f}-{p['assists']['p90']:.0f}] "
            f"| {p['minutes_median']:,.0f} |"
        )
    lines += [
        "",
        "## Scoring rules (fixed now)",
        "",
        "1. **Point skill** — MAE of the medians vs what happens, against the",
        "   persistence baseline (each player repeats his 2025-26 npG). The model",
        "   must beat it.",
        "2. **Interval honesty** — how many of the 16 land inside their 80% band.",
        "   Nominal is 80%; the v1/v2 backtests say the true figure will run high.",
        "3. **The headline claim** — right or wrong, no partial credit.",
        "4. **No exclusions.** A transfer out of the Big 5 scores 0 (that is the",
        "   model's stated scope); an injury counts as what happened. Every row is",
        "   scored.",
        "",
        "Run `python scripts/score_predictions.py` after the 2026-27 fetch to grade.",
    ]
    (ROOT / "PREDICTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"frozen {len(preds)} player predictions for {SEASON}")
    print(f"headline: Yamal npG median {yamal['npg']['median']:.0f} "
          f"[{yamal['npg']['p10']:.0f}-{yamal['npg']['p90']:.0f}] vs 13 in 2025-26")
    print(f"-> {RESULTS / 'predictions_2026-27.json'}")
    print(f"-> {ROOT / 'PREDICTIONS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
