"""The pre-registered gate (DESIGN.md §7), evaluated per horizon.

1. Calibration — the nominal 80% interval must cover 72%-88% of actuals.
2. Skill — MAE must beat the better baseline, bootstrap 95% CI excluding zero.
3. No concentration — the skill must survive leave-one-league-out and
   leave-one-position-out.

All three must hold, or the horizon is NOT PROVEN. These thresholds were fixed
before any result existed and are read from this module, not re-argued here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

NOMINAL = 0.80
COVERAGE_LO = 0.72
COVERAGE_HI = 0.88
N_BOOT = 2000


@dataclass
class CalibrationResult:
    nominal: float
    coverage: float
    lo_bound: float
    hi_bound: float
    n: int
    passed: bool


@dataclass
class SkillResult:
    model_mae: float
    baseline_maes: dict[str, float]
    best_baseline: str
    best_baseline_mae: float
    diff: float  # model - best baseline; negative means the model is better
    ci_lo: float
    ci_hi: float
    passed: bool


@dataclass
class ConcentrationResult:
    groups: dict[str, dict[str, Any]]
    passed: bool


@dataclass
class GateResult:
    horizon: int
    metric: str
    calibration: CalibrationResult
    skill: SkillResult
    concentration: ConcentrationResult

    @property
    def passed(self) -> bool:
        return self.calibration.passed and self.skill.passed and self.concentration.passed

    @property
    def verdict(self) -> str:
        return "PROVEN" if self.passed else "NOT PROVEN"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        d["passed"] = self.passed
        return d


def interval_coverage(sims: np.ndarray, actual: np.ndarray, nominal: float = NOMINAL) -> CalibrationResult:
    """Share of actuals inside the central `nominal` predictive interval.

    `sims` is [n_players, n_sims]. Counts are discrete and zero-inflated, so
    the empirical percentiles are lumpy and coverage tends to land conservative;
    that is a property of the target, reported rather than smoothed away.
    """
    tail = (1.0 - nominal) / 2.0
    lo = np.percentile(sims, 100 * tail, axis=1)
    hi = np.percentile(sims, 100 * (1 - tail), axis=1)
    covered = (actual >= lo) & (actual <= hi)
    cov = float(covered.mean())
    return CalibrationResult(
        nominal=nominal,
        coverage=cov,
        lo_bound=COVERAGE_LO,
        hi_bound=COVERAGE_HI,
        n=int(actual.size),
        passed=bool(COVERAGE_LO <= cov <= COVERAGE_HI),
    )


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def skill_vs_baselines(
    model_point: np.ndarray,
    baselines: dict[str, np.ndarray],
    actual: np.ndarray,
    seed: int = 0,
    n_boot: int = N_BOOT,
) -> SkillResult:
    """Compare model MAE with the *best* baseline, bootstrapping over players.

    Bootstrapping the paired error differences (rather than each MAE
    separately) respects that both methods are scored on the same players.
    """
    model_mae = _mae(model_point, actual)
    base_maes = {k: _mae(v, actual) for k, v in baselines.items()}
    best = min(base_maes, key=base_maes.get)
    best_mae = base_maes[best]

    model_err = np.abs(model_point - actual)
    base_err = np.abs(baselines[best] - actual)
    paired = model_err - base_err

    rng = np.random.default_rng(seed)
    n = paired.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = paired[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    return SkillResult(
        model_mae=model_mae,
        baseline_maes=base_maes,
        best_baseline=best,
        best_baseline_mae=best_mae,
        diff=float(paired.mean()),
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        # The model must be better (negative diff) with the whole CI below zero.
        passed=bool(ci_hi < 0.0),
    )


def concentration_check(
    model_point: np.ndarray,
    baselines: dict[str, np.ndarray],
    actual: np.ndarray,
    groups: dict[str, np.ndarray],
    seed: int = 0,
) -> ConcentrationResult:
    """Leave-one-group-out: does any single league or position carry the skill?

    Each group is dropped in turn and the skill test re-run on the remainder.
    If dropping one group flips the verdict, the edge is concentrated.
    """
    results: dict[str, dict[str, Any]] = {}
    all_pass = True
    for label, values in groups.items():
        # Nulls would make this a silent hole in the check — the affected rows
        # could never be left out, so an edge concentrated in them would pass
        # unnoticed. (np.unique also raises on mixed str/float.) Fail loudly.
        if pd.isna(pd.Series(values)).any():
            raise ValueError(
                f"group '{label}' contains nulls; leave-one-group-out cannot "
                "check a level it cannot name"
            )
        for level in np.unique(values):
            mask = values != level
            if mask.sum() < 50:
                continue
            sub = skill_vs_baselines(
                model_point[mask],
                {k: v[mask] for k, v in baselines.items()},
                actual[mask],
                seed=seed,
            )
            key = f"{label}!={level}"
            results[key] = {
                "n": int(mask.sum()),
                "diff": sub.diff,
                "ci_lo": sub.ci_lo,
                "ci_hi": sub.ci_hi,
                "passed": sub.passed,
            }
            all_pass = all_pass and sub.passed
    return ConcentrationResult(groups=results, passed=bool(all_pass and results))


def evaluate_gate(
    horizon: int,
    metric: str,
    sims: np.ndarray,
    actual: np.ndarray,
    baselines: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    seed: int = 0,
) -> GateResult:
    """Run all three tests for one horizon.

    The model's point prediction is the predictive **median**, which is the
    MAE-optimal summary of a distribution. Using the mean here would hand the
    model a worse score under its own metric; using a tuned quantile would be
    gaming it.
    """
    model_point = np.median(sims, axis=1)
    return GateResult(
        horizon=horizon,
        metric=metric,
        calibration=interval_coverage(sims, actual),
        skill=skill_vs_baselines(model_point, baselines, actual, seed=seed),
        concentration=concentration_check(model_point, baselines, actual, groups, seed=seed),
    )
