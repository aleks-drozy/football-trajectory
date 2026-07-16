"""Minutes-model tests.

Two failure modes matter here and neither is visible in the output distribution:
absences must be materialised as real zeros (or dropout is invisible), and the
final season must not be treated as everyone retiring at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schema as S
from model.minutes import (
    age_bucket_idx,
    fit_minutes_model,
    materialise_absences,
    minute_bucket,
)
from tests.conftest import make_player_season


def test_gap_season_becomes_an_explicit_zero():
    """Present 2017, absent 2018, back in 2019 -> a real 0-minute 2018 row,
    so both the fall to zero and the return are observable transitions."""
    panel = pd.DataFrame(
        [
            make_player_season("Injured", 1995, 2017, 2700, npg=10),
            make_player_season("Injured", 1995, 2019, 2700, npg=10),
        ]
    )
    full = materialise_absences(panel)
    row = full[(full[S.PLAYER] == "Injured") & (full[S.SEASON] == 2018)]
    assert len(row) == 1
    assert row[S.MINUTES].iloc[0] == 0.0
    assert row[S.AGE].iloc[0] == 2018 - 1995


def test_materialisation_starts_at_first_appearance():
    """A player who debuts in 2019 was not "a zero" in 2017; he was not yet
    observed, and inventing those rows would fabricate dropout history."""
    panel = pd.DataFrame(
        [
            make_player_season("Old", 1990, 2017, 2700, npg=5),
            make_player_season("Kid", 2003, 2019, 900, npg=2),
        ]
    )
    full = materialise_absences(panel)
    kid = full[full[S.PLAYER] == "Kid"]
    assert set(kid[S.SEASON]) == {2019}


def test_absent_after_debut_is_filled_to_the_end_of_data():
    panel = pd.DataFrame(
        [
            make_player_season("Old", 1990, 2017, 2700, npg=5),
            make_player_season("Old", 1990, 2019, 2700, npg=5),
            make_player_season("Kid", 2003, 2018, 900, npg=2),
        ]
    )
    full = materialise_absences(panel)
    kid = full[full[S.PLAYER] == "Kid"].sort_values(S.SEASON)
    assert list(kid[S.SEASON]) == [2018, 2019]
    assert list(kid[S.MINUTES]) == [900.0, 0.0]


def test_final_season_does_not_pair_forward():
    """Everyone is absent after the last season. If that counted as a
    transition, the model would learn that all players retire simultaneously
    and every projection would collapse to zero minutes."""
    panel = pd.DataFrame(
        [
            make_player_season("A", 1995, 2017, 2700, npg=9),
            make_player_season("A", 1995, 2018, 2700, npg=9),
        ]
    )
    model = fit_minutes_model(panel)
    # only the 2017->2018 transition exists, and it is emphatically not zero
    assert (model.overall == 2700.0).all()


def test_zero_transitions_are_retained_not_dropped():
    panel = pd.DataFrame(
        [
            make_player_season("Gone", 1995, 2017, 2700, npg=9),
            make_player_season("Stay", 1995, 2017, 2700, npg=9),
            make_player_season("Stay", 1995, 2018, 2700, npg=9),
        ]
    )
    model = fit_minutes_model(panel)
    assert (model.overall == 0.0).any(), "the player who vanished must appear as a zero"


def test_sample_returns_values_from_the_observed_pool():
    panel = pd.DataFrame(
        [
            make_player_season(f"P{i}", 1995, 2017, 2700, npg=9) for i in range(60)
        ]
        + [make_player_season(f"P{i}", 1995, 2018, 1800, npg=6) for i in range(60)]
    )
    model = fit_minutes_model(panel)
    rng = np.random.default_rng(0)
    draws = model.sample(2700, 23, rng, size=200)
    assert set(np.unique(draws)) <= {1800.0}


def test_sample_backs_off_when_a_cell_is_thin():
    """An unseen (minutes, age) combination must still produce a draw rather
    than raising or returning nothing."""
    panel = pd.DataFrame(
        [make_player_season(f"P{i}", 1995, 2017, 2700, npg=9) for i in range(40)]
        + [make_player_season(f"P{i}", 1995, 2018, 1800, npg=6) for i in range(40)]
    )
    model = fit_minutes_model(panel)
    rng = np.random.default_rng(0)
    draws = model.sample(500, 41, rng, size=10)  # never observed
    assert draws.shape == (10,)
    assert np.isfinite(draws).all()


def test_minute_buckets_are_ordered_and_cover_the_range():
    assert minute_bucket(0.0) < minute_bucket(600.0) < minute_bucket(2800.0)
    assert minute_bucket(0.0) == 0
    assert minute_bucket(99999.0) == len([e for e in (0, 1, 500, 1000, 1500, 2000, 2500, 3000)]) - 1


def test_age_buckets_are_monotonic():
    assert age_bucket_idx(17) <= age_bucket_idx(23) <= age_bucket_idx(35)


def test_max_season_censors_training_data():
    panel = pd.DataFrame(
        [
            make_player_season("A", 1995, 2017, 2700, npg=9),
            make_player_season("A", 1995, 2018, 900, npg=2),
            make_player_season("A", 1995, 2019, 100, npg=0),
        ]
    )
    model = fit_minutes_model(panel, max_season=2018)
    # only 2017->2018 is in scope; the 2019 collapse must be invisible
    assert set(np.unique(model.overall)) == {900.0}
