"""Loader tests against synthetic raw frames shaped like FBref's output."""

from __future__ import annotations

import pandas as pd
import pytest

from data import schema as S
from data.loader import (
    _primary_position,
    _season_start_year,
    audit_identity,
    repair_missing_league,
)
from tests.conftest import make_player_season


def test_season_code_maps_to_start_year():
    assert _season_start_year("1718") == 2017
    assert _season_start_year("2526") == 2025
    assert _season_start_year(1920) == 2019


def test_primary_position_takes_the_first_listed():
    assert _primary_position("MF,FW") == "MF"
    assert _primary_position("FW") == "FW"


def test_primary_position_handles_junk():
    assert _primary_position(None) == "UNK"
    assert _primary_position("") == "UNK"
    assert _primary_position("WIZARD") == "UNK"


def test_missing_league_label_is_inferred():
    """soccerdata nulls the Bundesliga label; four labels present means the
    unlabelled rows can only be the fifth."""
    raw = pd.DataFrame(
        {
            "league": [
                "ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "ITA-Serie A", None
            ],
            "team": ["Arsenal", "Barcelona", "PSG", "Milan", "Bayern Munich"],
        }
    )
    out = repair_missing_league(raw)
    assert out["league"].isna().sum() == 0
    assert out.loc[4, "league"] == "GER-Bundesliga"


def test_league_repair_is_a_noop_when_nothing_is_missing():
    raw = pd.DataFrame({"league": ["ENG-Premier League", "ESP-La Liga"]})
    assert repair_missing_league(raw)["league"].tolist() == [
        "ENG-Premier League",
        "ESP-La Liga",
    ]


def test_league_repair_refuses_an_ambiguous_fill():
    """Two labels missing means null rows could belong to either, so the
    inference is unsafe and must raise instead of mislabelling the data."""
    raw = pd.DataFrame(
        {"league": ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", None]}
    )
    with pytest.raises(ValueError, match="exactly one"):
        repair_missing_league(raw)


def test_league_repair_rejects_unexpected_leagues():
    raw = pd.DataFrame({"league": ["ENG-Premier League", "POR-Primeira Liga", None]})
    with pytest.raises(ValueError, match="unexpected leagues"):
        repair_missing_league(raw)


def test_identity_audit_counts_reused_names():
    """Two different players sharing a name are separated by birth year — the
    key works, and the audit should say so rather than flagging a problem."""
    panel = pd.DataFrame(
        [
            make_player_season("Rafinha", 1985, 2017, 1800, npg=2),
            make_player_season("Rafinha", 1993, 2017, 1800, npg=3),
        ]
    )
    audit = audit_identity(panel)
    assert audit.names_with_multiple_born == 1
    assert audit.n_players == 2
    assert audit.suspect_merges == 0


def test_identity_audit_flags_impossible_minutes():
    """Two distinct players sharing name *and* birth year would merge silently.
    The observable signature is a season no single human could play."""
    panel = pd.DataFrame([make_player_season("Ghost", 1995, 2017, 6000, npg=4)])
    audit = audit_identity(panel)
    assert audit.suspect_merges == 1
    assert audit.suspect_examples[0][0] == "Ghost"


def test_identity_audit_accepts_a_full_season():
    """A genuine ever-present must not be flagged as an identity merge."""
    panel = pd.DataFrame([make_player_season("Ironman", 1995, 2017, 3420, npg=10)])
    assert audit_identity(panel).suspect_merges == 0


def test_audit_summary_is_reportable():
    panel = pd.DataFrame([make_player_season("A", 1995, 2017, 1800, npg=5)])
    assert "panel rows=1" in audit_identity(panel).summary()
