"""Synthetic fixtures with known ground truth.

Tests here assert on hand-computable answers rather than on whatever the code
currently prints, so a regression shows up as a failure instead of a new
"expected" value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schema as S


def make_player_season(
    player: str,
    born: int,
    season: int,
    minutes: float,
    npg: float = 0.0,
    assists: float = 0.0,
    pos: str = "FW",
    league: str = "ENG-Premier League",
    team: str = "Team",
    shots: float = 0.0,
) -> dict:
    n90 = minutes / 90.0
    return {
        S.PLAYER: player,
        S.BORN: born,
        S.SEASON: season,
        S.AGE: season - born,
        S.LEAGUE: league,
        S.TEAM: team,
        S.POS: pos,
        S.MINUTES: minutes,
        S.N90: n90,
        S.MATCHES: max(int(minutes // 90), 0),
        S.STARTS: max(int(minutes // 90), 0),
        S.GOALS: npg,
        S.PK: 0.0,
        S.NPG: npg,
        S.ASSISTS: assists,
        S.SHOTS: shots,
        S.SOT: shots / 2.0,
        S.NPG90: npg / n90 if n90 > 0 else 0.0,
        S.AST90: assists / n90 if n90 > 0 else 0.0,
        S.SH90: shots / n90 if n90 > 0 else 0.0,
    }


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    """Two players, three seasons, all values chosen to be checkable by hand."""
    rows = [
        # steady scorer: 0.5 npg/90 every season
        make_player_season("Steady", 1998, 2017, 1800, npg=10, assists=5),
        make_player_season("Steady", 1998, 2018, 1800, npg=10, assists=5),
        make_player_season("Steady", 1998, 2019, 1800, npg=10, assists=5),
        # decliner: drops out after 2018
        make_player_season("Decliner", 1990, 2017, 1800, npg=10, assists=4),
        make_player_season("Decliner", 1990, 2018, 900, npg=2, assists=1),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def cohort_panel() -> pd.DataFrame:
    """A larger synthetic league with a known aging signal.

    True npg/90 is 0.40 at age 21 and falls by exactly 0.02 per year after,
    with no noise, so an aging curve fitted to it has a known right answer.

    The age span deliberately straddles the reference age of 21 (ages 20-26);
    a fixture starting at 22 could not test the curve's anchoring at all.
    """
    rng = np.random.default_rng(7)
    rows = []
    for pid in range(120):
        born = 1997
        base_minutes = float(rng.choice([1800.0, 2700.0]))
        for season in range(2017, 2024):  # ages 20..26
            age = season - born
            true_rate = 0.40 - 0.02 * max(age - 21, 0)
            npg = true_rate * (base_minutes / 90.0)
            rows.append(
                make_player_season(
                    f"P{pid}", born, season, base_minutes, npg=npg, assists=npg / 2
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def dropout_panel() -> pd.DataFrame:
    """Strong players who vanish — the case the dropouts variant exists for.

    Every player scores at 0.50 npg/90 at age 25. Half of them disappear at 26.
    Survivors-only sees no decline whatsoever; a dropout-aware curve must see
    the players who fell out of the league.
    """
    rows = []
    for pid in range(60):
        born = 1992
        rows.append(make_player_season(f"S{pid}", born, 2017, 1800, npg=10))  # 0.5/90
        rows.append(make_player_season(f"S{pid}", born, 2018, 1800, npg=10))
    for pid in range(60):
        born = 1992
        rows.append(make_player_season(f"G{pid}", born, 2017, 1800, npg=10))
        # no 2018 row: gone from the Big-5
    # a weak tail so the replacement quantile has genuinely poor regulars in it
    for pid in range(40):
        born = 1992
        rows.append(make_player_season(f"W{pid}", born, 2017, 1800, npg=1))  # 0.05/90
        rows.append(make_player_season(f"W{pid}", born, 2018, 1800, npg=1))
    return pd.DataFrame(rows)
