"""Aging-curve tests.

The censoring guard and the survivors/dropouts distinction are the two places
this module can be silently wrong in a way that still produces a plausible
curve, so both are pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schema as S
from model.aging import (
    build_pairs,
    fit_aging_curve,
    replacement_rates,
)
from tests.conftest import make_player_season


def test_pairs_link_consecutive_seasons_only(tiny_panel):
    pairs = build_pairs(tiny_panel, S.NPG90, variant="survivors")
    steady = pairs[pairs[S.PLAYER] == "Steady"]
    # 2017->2018 and 2018->2019 qualify; 2019->2020 is outside the data.
    assert sorted(steady[S.SEASON]) == [2017, 2018]


def test_last_season_does_not_manufacture_dropouts(tiny_panel):
    """Everyone is absent after the final season; that is censoring, not decline.

    Without the guard, every player in the panel would appear to retire at once
    and the dropouts curve would show a cliff that is purely an artefact of the
    dataset ending.
    """
    pairs = build_pairs(tiny_panel, S.NPG90, variant="dropouts")
    assert 2019 not in set(pairs[S.SEASON]), "final season must not pair forward"


def test_dropout_variant_keeps_the_player_who_vanishes():
    """A player present at a, absent at a+1, must still contribute a delta."""
    rows = [
        make_player_season("Gone", 1990, 2017, 1800, npg=9),
        make_player_season("Gone", 1990, 2018, 0.0, npg=0),  # placeholder, filtered
        make_player_season("Stays", 1990, 2017, 1800, npg=9),
        make_player_season("Stays", 1990, 2018, 1800, npg=9),
    ]
    panel = pd.DataFrame(rows)
    panel = panel[panel[S.MINUTES] > 0]  # "Gone" simply has no 2018 row

    surv = build_pairs(panel, S.NPG90, variant="survivors")
    drop = build_pairs(panel, S.NPG90, variant="dropouts")

    assert set(surv[S.PLAYER]) == {"Stays"}
    assert set(drop[S.PLAYER]) == {"Gone", "Stays"}


def test_dropout_variant_is_not_more_optimistic(dropout_panel):
    """Correcting survivor bias must not make aging look kinder.

    In `dropout_panel` every surviving player holds 0.50 npg/90 exactly, so the
    survivors-only curve sees zero decline. The players who fell out of the
    league were equally good the year before, and ignoring them is precisely
    the bias being corrected — so the dropout-aware deltas must be worse.
    """
    surv = build_pairs(dropout_panel, S.NPG90, variant="survivors")
    drop = build_pairs(dropout_panel, S.NPG90, variant="dropouts")
    assert surv["delta"].mean() == pytest.approx(0.0, abs=1e-9)
    assert drop["delta"].mean() < surv["delta"].mean()


def test_replacement_rate_is_genuinely_weak(dropout_panel):
    """Replacement must sit at the poor end, or imputing it to a vanished
    player would claim disappearing made him better."""
    rates = replacement_rates(dropout_panel, S.NPG90)
    median_fw = dropout_panel[dropout_panel[S.N90] >= 5][S.NPG90].median()
    assert rates["FW"] < median_fw


def test_weight_is_harmonic_mean_of_workloads():
    rows = [
        make_player_season("A", 1995, 2017, 900, npg=5),  # 10 x 90s
        make_player_season("A", 1995, 2018, 2700, npg=15),  # 30 x 90s
    ]
    pairs = build_pairs(pd.DataFrame(rows), S.NPG90, variant="survivors")
    expected = 2 * 10.0 * 30.0 / (10.0 + 30.0)  # 15.0
    assert pairs["weight"].iloc[0] == pytest.approx(expected)


def test_curve_recovers_a_known_linear_decline(cohort_panel):
    """A synthetic league losing exactly 0.02 npg/90 per year after 21."""
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors", reference_age=21)
    fw = curve.curve["FW"]
    assert fw[21] == pytest.approx(0.0, abs=1e-9)  # anchored at the reference
    assert fw[22] == pytest.approx(-0.02, abs=5e-3)
    assert fw[24] == pytest.approx(-0.06, abs=1e-2)
    assert curve.delta("FW", 22, 25) == pytest.approx(-0.06, abs=1e-2)


def test_curve_anchors_to_nearest_age_when_reference_is_unobserved(cohort_panel):
    """Reference age 30 is outside the fixture's range (20-26). The curve must
    still build and stay usable, because `delta` is anchor-invariant."""
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors", reference_age=30)
    assert curve.curve["FW"], "curve must not come back empty"
    assert curve.delta("FW", 22, 25) == pytest.approx(-0.06, abs=1e-2)


def test_offset_clamps_instead_of_extrapolating(cohort_panel):
    """Beyond the estimated range the curve must flatten, not run off."""
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors")
    ages = sorted(curve.curve["FW"])
    assert curve.offset("FW", ages[-1] + 15) == curve.offset("FW", ages[-1])
    assert curve.offset("FW", ages[0] - 15) == curve.offset("FW", ages[0])


def test_delta_is_difference_of_offsets(cohort_panel):
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors")
    got = curve.delta("FW", 22, 24)
    assert got == pytest.approx(curve.offset("FW", 24) - curve.offset("FW", 22))


def test_unknown_position_is_neutral_not_an_error(cohort_panel):
    curve = fit_aging_curve(cohort_panel, S.NPG90, variant="survivors")
    assert curve.delta("NOPE", 20, 30) == 0.0


def test_unknown_variant_rejected(tiny_panel):
    with pytest.raises(ValueError):
        build_pairs(tiny_panel, S.NPG90, variant="wishful")


def test_replacement_rate_uses_low_minute_players(tiny_panel):
    rates = replacement_rates(tiny_panel, S.NPG90)
    assert "FW" in rates
    assert np.isfinite(rates["FW"])


def test_max_season_excludes_later_data(cohort_panel):
    """Training must not see seasons past the split."""
    pairs = build_pairs(cohort_panel, S.NPG90, variant="survivors", max_season=2019)
    assert pairs[S.SEASON].max() <= 2018  # 2019 can only pair into 2020, excluded
