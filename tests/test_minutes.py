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
    TALENT_MIN_N90,
    UNKNOWN_TERCILE,
    age_bucket_idx,
    attach_carried_talent,
    fit_minutes_model,
    fit_talent_cuts,
    materialise_absences,
    minute_bucket,
    talent_tercile,
    talent_terciles_vec,
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


# --- talent conditioning (v2) -------------------------------------------------


def _talent_panel() -> pd.DataFrame:
    """Strong and weak players at identical minutes, with different fates.

    Every player starts on 2,700 minutes at 24. The strong ones keep playing;
    the weak ones fall out. Minutes and age alone cannot tell them apart — only
    talent can — so this is the fixture that proves conditioning works.
    """
    rows = []
    for i in range(60):  # strong: 0.8/90 attacking output, keep their place
        rows.append(make_player_season(f"Strong{i}", 1993, 2017, 2700, npg=15, assists=9))
        rows.append(make_player_season(f"Strong{i}", 1993, 2018, 2700, npg=15, assists=9))
    for i in range(60):  # weak: 0.033/90, gone next season
        rows.append(make_player_season(f"Weak{i}", 1993, 2017, 2700, npg=1, assists=0))
        # no 2018 row -> a real zero
    for i in range(60):  # padding so the weak cohort isn't the whole bottom
        rows.append(make_player_season(f"Mid{i}", 1993, 2017, 2700, npg=6, assists=3))
        rows.append(make_player_season(f"Mid{i}", 1993, 2018, 2700, npg=6, assists=3))
    return pd.DataFrame(rows)


def test_talent_cuts_are_ordered_and_per_position():
    panel = _talent_panel()
    cuts = fit_talent_cuts(panel)
    assert "FW" in cuts
    lo, hi = cuts["FW"]
    assert lo < hi


def test_talent_tercile_bands():
    cuts = {"FW": (0.2, 0.6)}
    assert talent_tercile(0.05, "FW", cuts) == 0
    assert talent_tercile(0.4, "FW", cuts) == 1
    assert talent_tercile(0.9, "FW", cuts) == 2


def test_unknown_talent_is_neutral_not_an_error():
    cuts = {"FW": (0.2, 0.6)}
    assert talent_tercile(float("nan"), "FW", cuts) == UNKNOWN_TERCILE
    assert talent_tercile(None, "FW", cuts) == UNKNOWN_TERCILE
    assert talent_tercile(0.4, "NOPE", cuts) == UNKNOWN_TERCILE  # unseen position


def test_vectorised_terciles_match_the_scalar_version():
    cuts = {"FW": (0.2, 0.6)}
    rates = np.array([0.05, 0.4, 0.9, np.nan])
    got = talent_terciles_vec(rates, "FW", cuts)
    expected = [talent_tercile(float(r), "FW", cuts) for r in rates]
    assert list(got) == expected


def test_talent_is_carried_forward_into_absent_seasons():
    """A player at zero minutes keeps the talent he last demonstrated — that is
    what decides whether he comes back, and a zero-minute row has no rate."""
    panel = pd.DataFrame(
        [
            make_player_season("Star", 1995, 2017, 2700, npg=20, assists=10),
            make_player_season("Star", 1995, 2019, 2700, npg=20, assists=10),
        ]
    )
    cuts = fit_talent_cuts(panel)
    full = attach_carried_talent(materialise_absences(panel), cuts)
    gap = full[(full[S.PLAYER] == "Star") & (full[S.SEASON] == 2018)]
    played = full[(full[S.PLAYER] == "Star") & (full[S.SEASON] == 2017)]
    assert gap["_tt"].iloc[0] == played["_tt"].iloc[0]


def test_talent_never_read_from_the_future():
    """Entering his first season a player has no track record, so his tercile
    must be the neutral default — not his own later form."""
    panel = pd.DataFrame(
        [
            make_player_season("Rookie", 2000, 2017, 200, npg=0, assists=0),  # below floor
            make_player_season("Rookie", 2000, 2018, 2700, npg=25, assists=10),  # elite
        ]
    )
    cuts = fit_talent_cuts(panel)
    full = attach_carried_talent(materialise_absences(panel), cuts)
    first = full[(full[S.PLAYER] == "Rookie") & (full[S.SEASON] == 2017)]
    assert first["_tt"].iloc[0] == UNKNOWN_TERCILE


def test_thin_season_does_not_overwrite_a_known_talent_level():
    panel = pd.DataFrame(
        [
            make_player_season("Vet", 1990, 2017, 2700, npg=20, assists=10),
            make_player_season("Vet", 1990, 2018, 90, npg=0, assists=0),  # one cameo
        ]
    )
    cuts = fit_talent_cuts(panel)
    full = attach_carried_talent(materialise_absences(panel), cuts)
    rows = full[full[S.PLAYER] == "Vet"].sort_values(S.SEASON)
    assert rows["_tt"].iloc[1] == rows["_tt"].iloc[0], "a cameo must not redefine talent"


def test_talent_conditioning_separates_identical_minutes():
    """The point of the change: same minutes, same age, different fate."""
    panel = _talent_panel()
    model = fit_minutes_model(panel)
    rng = np.random.default_rng(0)

    strong = model.sample(2700, 24, rng, size=4000, attack_rate=0.80, pos="FW")
    weak = model.sample(2700, 24, rng, size=4000, attack_rate=0.033, pos="FW")

    assert np.mean(weak == 0) > np.mean(strong == 0), (
        "the weak cohort dropped out and the strong one did not; conditioning on "
        "talent must reproduce that"
    )
    assert np.mean(strong == 0) < 0.1


def test_omitting_talent_falls_back_rather_than_assuming():
    """No talent supplied must mean 'unconditioned', not 'middling'."""
    panel = _talent_panel()
    model = fit_minutes_model(panel)
    rng = np.random.default_rng(0)
    pooled = model.sample(2700, 24, rng, size=4000)
    strong = model.sample(2700, 24, rng, size=4000, attack_rate=0.80, pos="FW")
    weak = model.sample(2700, 24, rng, size=4000, attack_rate=0.033, pos="FW")
    p_pooled, p_strong, p_weak = (np.mean(x == 0) for x in (pooled, strong, weak))
    assert p_strong <= p_pooled <= p_weak


def test_talent_cuts_respect_the_training_cutoff():
    """Cut points must not be computed from seasons past the split."""
    panel = _talent_panel()
    tampered = panel.copy()
    extra = pd.DataFrame(
        [make_player_season(f"Future{i}", 1993, 2019, 2700, npg=40, assists=20) for i in range(50)]
    )
    tampered = pd.concat([tampered, extra], ignore_index=True)

    a = fit_minutes_model(panel, max_season=2018)
    b = fit_minutes_model(tampered, max_season=2018)
    assert a.talent_cuts == b.talent_cuts


def test_cell_support_is_reported():
    model = fit_minutes_model(_talent_panel())
    assert model.cell_support(), "expected some (minutes, age, talent) cells"
    assert 0.0 <= model.talent_conditioned_share() <= 1.0


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
