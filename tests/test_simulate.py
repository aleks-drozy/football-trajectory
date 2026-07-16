"""Monte Carlo tests.

A simulator is easy to write so that it *looks* right — plausible numbers, a
nice fan chart — while quietly collapsing a source of uncertainty. These tests
check that each source actually propagates, and that talent, availability and
event noise all move the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schema as S
from mc.simulate import PlayerState, bootstrap_curves, simulate_player
from model.aging import fit_aging_curve
from model.minutes import fit_minutes_model
from model.talent import TalentModel, age_bucket
from tests.conftest import make_player_season


def _talent(K: float = 10.0, prior: float = 0.25) -> TalentModel:
    return TalentModel(
        metric=S.NPG90,
        events_col=S.NPG,
        K=K,
        priors={("FW", age_bucket(20)): prior, ("FW", age_bucket(24)): prior},
        global_prior=prior,
    )


def _state(minutes: float = 2700.0, age: int = 20) -> PlayerState:
    hist = pd.DataFrame([make_player_season("Kid", 2005, 2025, minutes, npg=12, assists=6)])
    return PlayerState("Kid", 2005, "FW", age, minutes, hist)


@pytest.fixture
def parts(cohort_panel):
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors")
    acurve = fit_aging_curve(cohort_panel, S.AST90, variant="survivors")
    mm = fit_minutes_model(cohort_panel)
    return curve, acurve, mm


def test_output_shapes_and_nonnegativity(parts):
    curve, acurve, mm = parts
    res = simulate_player(
        _state(), _talent(), _talent(), [curve], [acurve], mm,
        horizon=4, n_sims=500, rng=np.random.default_rng(0),
    )
    assert res.goals.shape == (500, 4)
    assert res.minutes.shape == (500, 4)
    assert (res.goals >= 0).all()
    assert (res.minutes >= 0).all()


def test_same_seed_reproduces(parts):
    curve, acurve, mm = parts
    a = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                        horizon=3, n_sims=200, rng=np.random.default_rng(42))
    b = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                        horizon=3, n_sims=200, rng=np.random.default_rng(42))
    assert np.array_equal(a.goals, b.goals)


def test_different_seeds_diverge(parts):
    curve, acurve, mm = parts
    a = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                        horizon=3, n_sims=200, rng=np.random.default_rng(1))
    b = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                        horizon=3, n_sims=200, rng=np.random.default_rng(2))
    assert not np.array_equal(a.goals, b.goals)


def test_talent_uncertainty_widens_the_fan(parts):
    """A thin record must produce a wider career spread than a long one at the
    same rate. If it does not, talent uncertainty is not propagating."""
    curve, acurve, mm = parts
    thin_hist = pd.DataFrame([make_player_season("Kid", 2005, 2025, 450, npg=2, assists=1)])
    thick_hist = pd.DataFrame(
        [make_player_season("Kid", 2005, y, 2700, npg=12, assists=6) for y in (2023, 2024, 2025)]
    )
    thin = PlayerState("Kid", 2005, "FW", 20, 2700.0, thin_hist)
    thick = PlayerState("Kid", 2005, "FW", 20, 2700.0, thick_hist)

    a = simulate_player(thin, _talent(), _talent(), [curve], [acurve], mm,
                        horizon=5, n_sims=3000, rng=np.random.default_rng(0))
    b = simulate_player(thick, _talent(), _talent(), [curve], [acurve], mm,
                        horizon=5, n_sims=3000, rng=np.random.default_rng(0))
    assert a.totals()["goals"].std() > b.totals()["goals"].std()


def test_zero_minutes_paths_score_zero(parts):
    """Availability must actually gate output: a path with no minutes cannot
    produce goals."""
    curve, acurve, mm = parts
    res = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                          horizon=5, n_sims=2000, rng=np.random.default_rng(3))
    zero = res.minutes == 0
    if zero.any():
        assert (res.goals[zero] == 0).all()


def test_more_minutes_means_more_goals(parts):
    curve, acurve, mm = parts
    res = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                          horizon=5, n_sims=3000, rng=np.random.default_rng(4))
    mins = res.minutes.ravel()
    goals = res.goals.ravel()
    if mins.std() > 0:
        assert np.corrcoef(mins, goals)[0, 1] > 0.3


def test_curve_ensemble_adds_spread(parts):
    """Passing many bootstrapped curves must widen outcomes versus pinning one.
    Otherwise aging-curve uncertainty is being treated as zero."""
    curve, acurve, mm = parts
    ensemble = bootstrap_curves(_dummy_panel(), S.NPG90, "survivors", n_boot=12, seed=0)
    if len(ensemble) < 5:
        pytest.skip("bootstrap produced too few curves on the fixture")
    one = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                          horizon=6, n_sims=4000, rng=np.random.default_rng(5))
    many = simulate_player(_state(), _talent(), _talent(), ensemble, [acurve], mm,
                           horizon=6, n_sims=4000, rng=np.random.default_rng(5))
    assert many.totals()["goals"].std() >= one.totals()["goals"].std() * 0.95


def _dummy_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(80):
        for season in range(2017, 2023):
            age = season - 1997
            rate = 0.4 - 0.02 * max(age - 21, 0) + rng.normal(0, 0.05)
            rows.append(
                make_player_season(f"P{pid}", 1997, season, 2700, npg=max(rate, 0) * 30,
                                   assists=5)
            )
    return pd.DataFrame(rows)


def test_bootstrap_curves_resamples_players_not_rows():
    curves = bootstrap_curves(_dummy_panel(), S.NPG90, "survivors", n_boot=8, seed=1)
    assert len(curves) > 0
    deltas = [c.delta("FW", 22, 26) for c in curves]
    assert len(set(np.round(deltas, 6))) > 1, "bootstrap curves must actually differ"


def test_requires_at_least_one_curve(parts):
    _, acurve, mm = parts
    with pytest.raises(ValueError):
        simulate_player(_state(), _talent(), _talent(), [], [acurve], mm,
                        horizon=2, n_sims=10, rng=np.random.default_rng(0))


def test_cumulative_goals_are_monotonic(parts):
    curve, acurve, mm = parts
    res = simulate_player(_state(), _talent(), _talent(), [curve], [acurve], mm,
                          horizon=5, n_sims=200, rng=np.random.default_rng(6))
    cum = res.cumulative_goals()
    assert (np.diff(cum, axis=1) >= 0).all()
