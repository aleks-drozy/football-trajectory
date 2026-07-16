"""Run the pre-registered gate (DESIGN.md §7).

Season budget (9 seasons, 2017-2025) is split three ways so that nothing used
to *choose* the model is also used to *judge* it:

    2017-2019  inner-train    fit both aging-curve variants
    2020-2021  inner-select   pick the variant  <- selection happens here
    2017-2021  final-train    refit everything with the chosen variant
    2022-2024  TEST           the gate  <- touched exactly once

DESIGN.md §5.3 says validation chooses between the survivors and dropouts
variants. Choosing it on the test split would be selection on the test set — the
model would get two attempts at the gate and report the better one. So the
choice is made on an inner split carved out of training data, and the test
seasons are not touched until the chosen variant is fixed.

Usage:
    python run_validate.py            # -> results/gate.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and cannot encode many real player
# names; without this a print can kill the run after all the work is done.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data import schema as S
from data.loader import audit_identity, build_panel
from validation.backtest import (
    actuals_at,
    build_cohort,
    fit_system,
    project_cohort,
)
from validation.baselines import cohort_mean, fit_cohort_means, persistence
from validation.gate import evaluate_gate

RESULTS = Path(__file__).parent / "results"
PANEL_CACHE = Path(__file__).parent / "data" / "panel.parquet"

INNER_TRAIN = [2017, 2018, 2019]
INNER_BASE = 2019
INNER_HORIZONS = 2

FINAL_TRAIN = [2017, 2018, 2019, 2020, 2021]
FINAL_BASE = 2021
FINAL_HORIZONS = 3

N_SIMS = 1500
N_BOOT_CURVES = 30
METRIC_EVENTS = {"goals": S.NPG, "assists": S.ASSISTS}


def load_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        return pd.read_parquet(PANEL_CACHE)
    panel = build_panel()
    panel.to_parquet(PANEL_CACHE, index=False)
    return panel


def _baselines_for(
    panel: pd.DataFrame,
    cohort: pd.DataFrame,
    train_seasons: list[int],
    events_col: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    cells, overall = fit_cohort_means(panel, train_seasons, events_col)
    return {
        "persistence": persistence(cohort, events_col),
        "cohort_mean": cohort_mean(cohort, cells, overall, horizon),
    }


def select_variant(panel: pd.DataFrame) -> tuple[str, dict]:
    """Pick the aging-curve variant on an inner split, never on the test."""
    cohort = build_cohort(panel, INNER_BASE)
    scores: dict[str, float] = {}
    detail: dict[str, dict] = {}

    for variant in ("survivors", "dropouts"):
        system = fit_system(panel, INNER_TRAIN, variant=variant, n_boot=10)
        sims = project_cohort(panel, system, cohort, INNER_HORIZONS, n_sims=600, seed=1)
        maes = []
        for h in range(1, INNER_HORIZONS + 1):
            actual = actuals_at(panel, cohort, INNER_BASE + h, S.NPG)
            point = np.median(sims["goals"][:, :, h - 1], axis=1)
            maes.append(float(np.mean(np.abs(point - actual))))
        scores[variant] = float(np.mean(maes))
        detail[variant] = {"mae_by_horizon": maes, "mean_mae": scores[variant]}

    chosen = min(scores, key=scores.get)
    return chosen, {"scores": detail, "chosen": chosen, "split": "inner (2020-2021)"}


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()
    audit = audit_identity(panel)
    print(f"panel: {panel.shape}")
    print(f"identity: {audit.summary()}\n")

    print("selecting aging-curve variant on the inner split...")
    variant, selection = select_variant(panel)
    for v, d in selection["scores"].items():
        print(f"  {v:10s} inner mean MAE = {d['mean_mae']:.4f}")
    print(f"  -> chosen: {variant}\n")

    print(f"fitting final system on {FINAL_TRAIN} (variant={variant})...")
    system = fit_system(panel, FINAL_TRAIN, variant=variant, n_boot=N_BOOT_CURVES)
    print(f"  fitted K (npg)     = {system.K_npg}")
    print(f"  fitted K (assists) = {system.K_ast}")
    print(f"  bootstrap curves   = {len(system.npg_curves)} npg / {len(system.ast_curves)} ast\n")

    cohort = build_cohort(panel, FINAL_BASE)
    print(f"test cohort: {len(cohort)} players from {FINAL_BASE}")
    print(f"projecting {FINAL_HORIZONS} seasons x {N_SIMS} sims...\n")
    sims = project_cohort(panel, system, cohort, FINAL_HORIZONS, n_sims=N_SIMS, seed=0)

    groups = {
        "league": cohort[S.LEAGUE].to_numpy(),
        "pos": cohort[S.POS].to_numpy(),
    }

    out: dict = {
        "panel_rows": int(len(panel)),
        "identity_audit": audit.summary(),
        "variant_selection": selection,
        "variant": variant,
        "train_seasons": FINAL_TRAIN,
        "base_season": FINAL_BASE,
        "test_seasons": [FINAL_BASE + h for h in range(1, FINAL_HORIZONS + 1)],
        "cohort_n": int(len(cohort)),
        "n_sims": N_SIMS,
        "fitted_K": {"npg": system.K_npg, "assists": system.K_ast},
        "K_scores": {
            "npg": {str(k): v for k, v in system.K_scores_npg.items()},
            "assists": {str(k): v for k, v in system.K_scores_ast.items()},
        },
        "gates": {},
    }

    for metric, events_col in METRIC_EVENTS.items():
        out["gates"][metric] = []
        for h in range(1, FINAL_HORIZONS + 1):
            actual = actuals_at(panel, cohort, FINAL_BASE + h, events_col)
            baselines = _baselines_for(panel, cohort, FINAL_TRAIN, events_col, h)
            res = evaluate_gate(
                horizon=h,
                metric=metric,
                sims=sims[metric][:, :, h - 1],
                actual=actual,
                baselines=baselines,
                groups=groups,
            )
            out["gates"][metric].append(res.to_dict())

            c, s = res.calibration, res.skill
            print(f"[{metric} h+{h}]  {res.verdict}")
            print(
                f"   calibration: {c.coverage:.1%} in the 80% interval "
                f"(need {c.lo_bound:.0%}-{c.hi_bound:.0%}) -> {'PASS' if c.passed else 'FAIL'}"
            )
            print(
                f"   skill: MAE {s.model_mae:.3f} vs {s.best_baseline} {s.best_baseline_mae:.3f} "
                f"| diff {s.diff:+.3f} [{s.ci_lo:+.3f}, {s.ci_hi:+.3f}] "
                f"-> {'PASS' if s.passed else 'FAIL'}"
            )
            n_bad = sum(1 for v in res.concentration.groups.values() if not v["passed"])
            print(
                f"   concentration: {n_bad}/{len(res.concentration.groups)} groups fail "
                f"-> {'PASS' if res.concentration.passed else 'FAIL'}\n"
            )

    (RESULTS / "gate.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {RESULTS / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
