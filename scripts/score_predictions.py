"""Score the frozen 2026-27 predictions against what actually happened.

Run AFTER the 2026-27 season ends and the panel has been refreshed
(scripts/fetch_data.py with the 2026-2027 season added, then rebuild
data/panel.parquet). The scoring rules live in the frozen ledger itself
(results/predictions_2026-27.json, "scoring_protocol") and are not re-decided
here — this script just executes them.

Usage:
    python scripts/score_predictions.py    # -> results/predictions_score.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import schema as S

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TARGET_SEASON = 2026  # season start year for 2026-27
BASE_SEASON = 2025


def main() -> int:
    frozen = json.loads(
        (RESULTS / "predictions_2026-27.json").read_text(encoding="utf-8")
    )
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    if TARGET_SEASON not in set(panel[S.SEASON].unique()):
        print(
            f"panel has no {TARGET_SEASON} season yet - refetch first "
            "(scripts/fetch_data.py with the new season added)"
        )
        return 1

    rows = []
    for p in frozen["predictions"]:
        actual_rows = panel[
            (panel[S.PLAYER] == p["player"])
            & (panel[S.BORN] == p["born"])
            & (panel[S.SEASON] == TARGET_SEASON)
        ]
        # Absent from the Big 5 = 0, per the frozen no-exclusions rule.
        actual = float(actual_rows[S.NPG].sum()) if len(actual_rows) else 0.0
        base_rows = panel[
            (panel[S.PLAYER] == p["player"])
            & (panel[S.BORN] == p["born"])
            & (panel[S.SEASON] == BASE_SEASON)
        ]
        persistence = float(base_rows[S.NPG].sum()) if len(base_rows) else 0.0
        rows.append(
            {
                "player": p["player"],
                "predicted_median": p["npg"]["median"],
                "interval": [p["npg"]["p10"], p["npg"]["p90"]],
                "actual_npg": actual,
                "persistence_baseline": persistence,
                "abs_error_model": abs(p["npg"]["median"] - actual),
                "abs_error_persistence": abs(persistence - actual),
                "in_interval": bool(p["npg"]["p10"] <= actual <= p["npg"]["p90"]),
            }
        )

    mae_model = float(np.mean([r["abs_error_model"] for r in rows]))
    mae_persist = float(np.mean([r["abs_error_persistence"] for r in rows]))
    coverage = float(np.mean([r["in_interval"] for r in rows]))
    yamal = next(r for r in rows if r["player"] == "Lamine Yamal")
    headline_correct = bool(yamal["actual_npg"] < 13)

    out = {
        "scored_season": frozen["season"],
        "frozen_on": frozen["frozen_on"],
        "n_players": len(rows),
        "mae_model": round(mae_model, 3),
        "mae_persistence": round(mae_persist, 3),
        "model_beats_persistence": bool(mae_model < mae_persist),
        "interval_coverage": round(coverage, 3),
        "nominal_coverage": 0.80,
        "headline_claim_correct": headline_correct,
        "per_player": rows,
    }
    (RESULTS / "predictions_score.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print(f"scored {len(rows)} players for {frozen['season']}")
    print(f"  MAE  model {mae_model:.2f}  vs persistence {mae_persist:.2f}  "
          f"-> model {'BEATS' if mae_model < mae_persist else 'LOSES TO'} the baseline")
    print(f"  interval coverage {coverage:.0%} (nominal 80%)")
    print(f"  headline (Yamal < 13 npG): {'CORRECT' if headline_correct else 'WRONG'} "
          f"(actual {yamal['actual_npg']:.0f})")
    print(f"-> {RESULTS / 'predictions_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
