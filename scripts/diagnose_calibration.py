"""Locate the calibration failure instead of just reporting it.

The gate says the 80% interval covers ~91% — the model is *under*-confident,
which is the opposite of this genre's usual failure. That is a finding, not an
excuse, and it needs a cause. Two candidate explanations:

1. **Discreteness + zero-inflation.** For a centre-back whose predictive
   distribution is mostly 0, the empirical 10th/90th percentiles are 0 and ~1,
   and an actual of 0 is trivially "covered". Coverage then overshoots for
   reasons that have nothing to do with the model being cautious.
2. **Genuinely inflated spread.** The simulation stacks four uncertainty
   sources; if one is double-counted the intervals are simply too wide for
   everyone, high scorers included.

These make opposite predictions. Under (1) the overshoot concentrates in
low-expectation players and shrinks among high scorers. Under (2) it is roughly
flat across the board. So coverage is broken out by predicted level and by
position — where they disagree is the answer.

Usage:
    python scripts/diagnose_calibration.py    # -> results/calibration_diagnostic.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and cannot encode many real player
# names; without this a print can kill the run after all the work is done.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import schema as S
from validation.backtest import actuals_at, build_cohort, fit_system, project_cohort

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TRAIN = [2017, 2018, 2019, 2020, 2021]
BASE = 2021
HORIZON = 3
N_SIMS = 1500


def coverage(sims: np.ndarray, actual: np.ndarray) -> tuple[float, np.ndarray]:
    lo = np.percentile(sims, 10, axis=1)
    hi = np.percentile(sims, 90, axis=1)
    cov = (actual >= lo) & (actual <= hi)
    return float(cov.mean()), cov


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    gate = json.loads((RESULTS / "gate.json").read_text())
    variant = gate["variant"]

    print(f"refitting (variant={variant}) to reproduce the gate's sims...")
    system = fit_system(panel, TRAIN, variant=variant, n_boot=30)
    cohort = build_cohort(panel, BASE)
    sims = project_cohort(panel, system, cohort, HORIZON, n_sims=N_SIMS, seed=0)

    out: dict = {"base_season": BASE, "n_sims": N_SIMS, "by_horizon": {}}

    for h in (1, 2, 3):
        g = sims["goals"][:, :, h - 1]
        actual = actuals_at(panel, cohort, BASE + h, S.NPG)
        overall, covered = coverage(g, actual)

        med = np.median(g, axis=1)
        width = np.percentile(g, 90, axis=1) - np.percentile(g, 10, axis=1)

        # Split by how much the model expects, which is what hypothesis (1)
        # says should matter.
        buckets = {
            "median 0 goals": med == 0,
            "median 1-2": (med >= 1) & (med <= 2),
            "median 3-5": (med >= 3) & (med <= 5),
            "median 6+": med >= 6,
        }
        by_level = {}
        for name, mask in buckets.items():
            if mask.sum() < 20:
                continue
            by_level[name] = {
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_interval_width": float(width[mask].mean()),
                "mean_actual": float(actual[mask].mean()),
            }

        by_pos = {}
        for pos in sorted(cohort[S.POS].unique()):
            mask = (cohort[S.POS] == pos).to_numpy()
            if mask.sum() < 20:
                continue
            by_pos[pos] = {
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_interval_width": float(width[mask].mean()),
            }

        # How often is the interval so degenerate that covering is automatic?
        degenerate = float(np.mean(np.percentile(g, 10, axis=1) == 0))

        out["by_horizon"][f"h+{h}"] = {
            "overall_coverage": overall,
            "share_with_zero_lower_bound": degenerate,
            "by_predicted_level": by_level,
            "by_position": by_pos,
        }

        print(f"\n[goals h+{h}] overall coverage {overall:.1%}")
        print(f"  share whose 10th pct is 0 (covering 0 is automatic): {degenerate:.1%}")
        for name, d in by_level.items():
            print(
                f"  {name:16s} n={d['n']:5d}  coverage {d['coverage']:6.1%}  "
                f"width {d['mean_interval_width']:5.1f}  mean actual {d['mean_actual']:5.2f}"
            )
        for pos, d in by_pos.items():
            print(f"  {pos:4s} n={d['n']:5d}  coverage {d['coverage']:6.1%}  width {d['mean_interval_width']:5.1f}")

    (RESULTS / "calibration_diagnostic.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {RESULTS / 'calibration_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
