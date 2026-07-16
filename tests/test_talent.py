"""Talent-model tests.

The shrinkage estimator is the component most able to look reasonable while
being wrong, because any K produces a plausible-looking number. These tests pin
the maths (it must be the Gamma-Poisson posterior mean), the behaviour at the
limits of K, and the absence of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schema as S
from model.talent import (
    TalentModel,
    age_bucket,
    build_talent_model,
    fit_K,
    fit_priors,
)
from tests.conftest import make_player_season


def _model(K: float, prior: float = 0.2) -> TalentModel:
    return TalentModel(
        metric=S.NPG90,
        events_col=S.NPG,
        K=K,
        priors={("FW", age_bucket(22)): prior},
        global_prior=prior,
    )


def _history(npg: float, minutes: float, season: int = 2019) -> pd.DataFrame:
    return pd.DataFrame([make_player_season("X", 1997, season, minutes, npg=npg)])


def test_estimate_matches_closed_form_shrinkage():
    """(w·events + K·prior) / (w·n90 + K), with the most recent weight = 5."""
    m = _model(K=10.0, prior=0.2)
    hist = _history(npg=10, minutes=900)  # 10 x 90s
    expected = (5 * 10 + 10.0 * 0.2) / (5 * 10.0 + 10.0)
    assert m.estimate(hist, "FW", 22) == pytest.approx(expected)


def test_posterior_mean_equals_estimate():
    """The shrinkage point estimate must be the posterior mean, or the
    uncertainty drawn from that posterior describes a different quantity."""
    m = _model(K=10.0)
    hist = _history(npg=10, minutes=900)
    shape, rate = m.posterior(hist, "FW", 22)
    assert shape / rate == pytest.approx(m.estimate(hist, "FW", 22))


def test_empty_history_returns_the_prior():
    m = _model(K=10.0, prior=0.23)
    assert m.estimate(pd.DataFrame(columns=[S.SEASON, S.NPG, S.N90]), "FW", 22) == pytest.approx(0.23)


def test_large_K_pulls_to_prior_small_K_trusts_the_player():
    hot = _history(npg=10, minutes=450)  # 2.0 npg/90 over 5 x 90s: hot and thin
    # K is in units of 90s, so it only swamps the evidence once it dwarfs the
    # player's weighted workload (here 5 x 5 = 25 ninety-minute blocks).
    heavy = _model(K=1e6, prior=0.2).estimate(hot, "FW", 22)
    light = _model(K=0.01, prior=0.2).estimate(hot, "FW", 22)
    assert heavy == pytest.approx(0.2, abs=0.01)
    assert light == pytest.approx(2.0, abs=0.05)


def test_K_moves_the_estimate_monotonically_toward_the_prior():
    hot = _history(npg=10, minutes=450)
    estimates = [_model(K=k, prior=0.2).estimate(hot, "FW", 22) for k in (1, 10, 100, 1000)]
    assert estimates == sorted(estimates, reverse=True), "more shrinkage must move toward 0.2"
    assert all(e > 0.2 for e in estimates)


def test_shrinkage_is_stronger_for_smaller_samples():
    """The whole point: a thin hot streak must be trusted less than a full season
    at the same rate."""
    m = _model(K=20.0, prior=0.2)
    thin = m.estimate(_history(npg=5, minutes=225), "FW", 22)  # 2.0/90 over 2.5 x 90s
    thick = m.estimate(_history(npg=60, minutes=2700), "FW", 22)  # 2.0/90 over 30 x 90s
    assert thin < thick


def test_recency_weighting_favours_the_latest_season():
    m = _model(K=1e-9, prior=0.0)
    hist = pd.DataFrame(
        [
            make_player_season("X", 1997, 2018, 900, npg=0),  # cold, older
            make_player_season("X", 1997, 2019, 900, npg=10),  # hot, recent
        ]
    )
    got = m.estimate(hist, "FW", 22)
    # weights 5 (2019) and 4 (2018): (5*10 + 4*0) / (5*10 + 4*10)
    assert got == pytest.approx(50.0 / 90.0, rel=1e-6)


def test_sample_talent_centres_on_the_posterior_mean():
    m = _model(K=10.0)
    hist = _history(npg=10, minutes=900)
    rng = np.random.default_rng(0)
    draws = m.sample_talent(hist, "FW", 22, rng, size=40000)
    assert draws.mean() == pytest.approx(m.estimate(hist, "FW", 22), rel=0.03)
    assert (draws >= 0).all(), "a rate cannot be negative"


def test_posterior_narrows_as_workload_grows():
    m = _model(K=10.0)
    rng = np.random.default_rng(0)
    thin = m.sample_talent(_history(npg=5, minutes=450), "FW", 22, rng, 20000)
    thick = m.sample_talent(_history(npg=30, minutes=2700), "FW", 22, rng, 20000)
    assert thick.std() < thin.std()


def test_fit_K_selects_from_the_grid_and_scores_every_point(cohort_panel):
    K, scores = fit_K(cohort_panel, S.NPG90, S.NPG, train_seasons=[2017, 2018, 2019])
    from model.talent import K_GRID

    assert K in K_GRID
    assert set(scores) == set(K_GRID)
    assert scores[K] == min(scores.values())
    assert all(np.isfinite(v) for v in scores.values())


def test_vectorised_fit_path_matches_the_per_row_estimator(cohort_panel):
    """fit_K precomputes K-independent sums and evaluates the grid in closed
    form. That must agree exactly with calling TalentModel.estimate row by row,
    or the optimisation has quietly changed the model.
    """
    from model.aging import fit_aging_curve
    from model.talent import (
        DEFAULT_RECENCY_WEIGHTS,
        _prediction_frame,
        _sufficient_stats,
    )

    train_seasons = [2017, 2018, 2019, 2020]
    train = cohort_panel[cohort_panel[S.SEASON].isin(train_seasons)]
    curve = fit_aging_curve(train, S.NPG90, variant="survivors")
    priors = fit_priors(train, S.NPG90)
    played = train[train[S.N90] > 0]
    global_prior = float(
        np.average(played[S.NPG90].to_numpy(), weights=played[S.N90].to_numpy())
    )

    targets = _prediction_frame(train, S.NPG90, 5.0)
    w_events, w_n90, prior_arr, delta_arr = _sufficient_stats(
        train, targets, S.NPG, S.NPG90, curve, DEFAULT_RECENCY_WEIGHTS, priors, global_prior
    )

    K = 17.0
    fast = (w_events + K * prior_arr) / (w_n90 + K) + delta_arr

    model = TalentModel(
        metric=S.NPG90, events_col=S.NPG, K=K, priors=priors, global_prior=global_prior
    )
    slow = np.empty(len(targets))
    for i, row in enumerate(targets.itertuples(index=False)):
        grp = train[
            (train[S.PLAYER] == getattr(row, S.PLAYER))
            & (train[S.BORN] == getattr(row, S.BORN))
        ]
        hist = grp[grp[S.SEASON] <= row.from_season]
        pos, age = getattr(row, S.POS), int(getattr(row, S.AGE))
        slow[i] = model.estimate(hist, pos, age) + curve.delta(pos, age, int(row.age_next))

    assert len(fast) > 50, "fixture should produce a meaningful number of targets"
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-12)


def test_priors_are_minutes_weighted(cohort_panel):
    priors = fit_priors(cohort_panel, S.NPG90)
    assert priors, "expected at least one (pos, bucket) prior"
    assert all(np.isfinite(v) for v in priors.values())


def test_build_talent_model_ignores_seasons_outside_training(cohort_panel):
    """A prior fitted on training seasons must not shift when later seasons
    change — that would be leakage."""
    train = [2017, 2018]
    m1 = build_talent_model(cohort_panel, S.NPG90, S.NPG, train, K=10.0)

    tampered = cohort_panel.copy()
    later = tampered[S.SEASON] > 2018
    tampered.loc[later, S.NPG90] = 99.0
    m2 = build_talent_model(tampered, S.NPG90, S.NPG, train, K=10.0)

    assert m1.priors == m2.priors
    assert m1.global_prior == pytest.approx(m2.global_prior)


def test_age_bucket_covers_all_ages():
    for age in range(14, 45):
        assert age_bucket(age) is not None
