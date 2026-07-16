"""Gate tests.

The gate is the project's verdict machinery, so it has to fail when it should.
These tests feed it constructions with known answers — a perfectly calibrated
predictor, a badly overconfident one, a useless model, and an edge that lives in
a single league — and check the verdict each time.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.gate import (
    COVERAGE_HI,
    COVERAGE_LO,
    concentration_check,
    evaluate_gate,
    interval_coverage,
    skill_vs_baselines,
)


def test_calibrated_predictor_passes_coverage():
    """Sims drawn from the same law as the actuals must cover ~80%."""
    rng = np.random.default_rng(0)
    n, n_sims = 3000, 4000
    lam = rng.uniform(1, 12, size=n)
    sims = rng.poisson(lam[:, None], size=(n, n_sims)).astype(float)
    actual = rng.poisson(lam).astype(float)
    res = interval_coverage(sims, actual)
    assert res.passed
    assert COVERAGE_LO <= res.coverage <= COVERAGE_HI


def test_overconfident_predictor_fails_coverage():
    """Intervals far too narrow must be caught, not tolerated."""
    rng = np.random.default_rng(1)
    n, n_sims = 2000, 2000
    lam = rng.uniform(1, 12, size=n)
    # sims cling to the mean; actuals have real Poisson spread
    sims = (lam[:, None] + rng.normal(0, 0.01, size=(n, n_sims))).astype(float)
    actual = rng.poisson(lam).astype(float)
    res = interval_coverage(sims, actual)
    assert not res.passed
    assert res.coverage < COVERAGE_LO


def test_underconfident_predictor_fails_coverage():
    """Absurdly wide intervals are also a failure, not a safe default."""
    rng = np.random.default_rng(2)
    n, n_sims = 2000, 2000
    lam = rng.uniform(1, 12, size=n)
    sims = rng.poisson(lam[:, None], size=(n, n_sims)) + rng.normal(0, 40, size=(n, n_sims))
    actual = rng.poisson(lam).astype(float)
    res = interval_coverage(sims, actual)
    assert not res.passed
    assert res.coverage > COVERAGE_HI


def test_skill_passes_only_when_the_whole_ci_is_below_zero():
    rng = np.random.default_rng(3)
    n = 4000
    actual = rng.poisson(6, size=n).astype(float)
    good = actual + rng.normal(0, 0.5, size=n)  # nearly right
    bad = actual + rng.normal(0, 6.0, size=n)  # noisy baseline
    res = skill_vs_baselines(good, {"noisy": bad}, actual)
    assert res.passed
    assert res.ci_hi < 0
    assert res.diff < 0


def test_no_skill_fails():
    """A model no better than the baseline must not pass."""
    rng = np.random.default_rng(4)
    n = 3000
    actual = rng.poisson(6, size=n).astype(float)
    a = actual + rng.normal(0, 3.0, size=n)
    b = actual + rng.normal(0, 3.0, size=n)
    res = skill_vs_baselines(a, {"twin": b}, actual)
    assert not res.passed
    assert res.ci_lo < 0 < res.ci_hi


def test_worse_than_baseline_fails():
    rng = np.random.default_rng(5)
    n = 3000
    actual = rng.poisson(6, size=n).astype(float)
    worse = actual + rng.normal(0, 8.0, size=n)
    better = actual + rng.normal(0, 1.0, size=n)
    res = skill_vs_baselines(worse, {"better": better}, actual)
    assert not res.passed
    assert res.diff > 0


def test_best_baseline_is_the_one_the_model_must_beat():
    """Skill is measured against the strongest baseline, not a convenient one."""
    rng = np.random.default_rng(6)
    n = 2000
    actual = rng.poisson(6, size=n).astype(float)
    model = actual + rng.normal(0, 2.0, size=n)
    strong = actual + rng.normal(0, 1.0, size=n)
    weak = actual + rng.normal(0, 9.0, size=n)
    res = skill_vs_baselines(model, {"strong": strong, "weak": weak}, actual)
    assert res.best_baseline == "strong"
    assert not res.passed  # model is worse than the strong baseline


def test_concentration_catches_an_edge_from_a_single_league():
    """A model that is only good in one league must fail the LOGO check."""
    rng = np.random.default_rng(7)
    n = 4000
    league = rng.choice(["A", "B", "C", "D"], size=n)
    actual = rng.poisson(6, size=n).astype(float)
    baseline = actual + rng.normal(0, 3.0, size=n)
    model = baseline.copy()
    only = league == "A"
    model[only] = actual[only] + rng.normal(0, 0.2, size=only.sum())  # all the edge here

    res = concentration_check(model, {"base": baseline}, actual, {"league": league})
    assert not res.passed
    assert res.groups["league!=A"]["passed"] is False


def test_concentration_passes_for_a_broad_edge():
    rng = np.random.default_rng(8)
    n = 4000
    league = rng.choice(["A", "B", "C", "D"], size=n)
    actual = rng.poisson(6, size=n).astype(float)
    baseline = actual + rng.normal(0, 4.0, size=n)
    model = actual + rng.normal(0, 1.0, size=n)  # good everywhere
    res = concentration_check(model, {"base": baseline}, actual, {"league": league})
    assert res.passed


def test_gate_requires_all_three_tests():
    """Skill without calibration is still NOT PROVEN — the point of the gate."""
    rng = np.random.default_rng(9)
    n, n_sims = 2000, 1500
    lam = rng.uniform(2, 10, size=n)
    actual = rng.poisson(lam).astype(float)
    # accurate median, but essentially zero spread -> skilful yet overconfident
    sims = np.repeat(lam[:, None], n_sims, axis=1) + rng.normal(0, 0.01, size=(n, n_sims))
    baseline = actual + rng.normal(0, 6.0, size=n)
    groups = {"pos": rng.choice(["FW", "MF"], size=n)}
    res = evaluate_gate(1, "goals", sims, actual, {"noisy": baseline}, groups)
    assert res.skill.passed
    assert not res.calibration.passed
    assert not res.passed
    assert res.verdict == "NOT PROVEN"


def test_gate_uses_the_median_as_the_point_prediction():
    rng = np.random.default_rng(10)
    n, n_sims = 500, 999
    sims = rng.poisson(5, size=(n, n_sims)).astype(float)
    actual = rng.poisson(5, size=n).astype(float)
    groups = {"pos": np.array(["FW"] * n)}
    res = evaluate_gate(1, "goals", sims, actual, {"b": actual + 1.0}, groups)
    expected_mae = float(np.mean(np.abs(np.median(sims, axis=1) - actual)))
    assert res.skill.model_mae == pytest.approx(expected_mae)


def test_verdict_serialises_for_the_results_file():
    rng = np.random.default_rng(11)
    n, n_sims = 300, 400
    sims = rng.poisson(4, size=(n, n_sims)).astype(float)
    actual = rng.poisson(4, size=n).astype(float)
    groups = {"pos": rng.choice(["FW", "MF"], size=n)}
    res = evaluate_gate(2, "goals", sims, actual, {"b": actual + 2.0}, groups)
    d = res.to_dict()
    assert d["verdict"] in ("PROVEN", "NOT PROVEN")
    assert d["horizon"] == 2
    assert "calibration" in d and "skill" in d and "concentration" in d
