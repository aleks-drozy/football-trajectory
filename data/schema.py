"""Canonical column names and constants for the player-season panel."""

from __future__ import annotations

# Identity
PLAYER = "player"
BORN = "born"
SEASON = "season"  # season start year, e.g. 2023 for 2023-2024
AGE = "age"  # season_start_year - born_year
LEAGUE = "league"
TEAM = "team"
POS = "pos"  # primary position group: GK / DF / MF / FW

# Volume
MINUTES = "minutes"
N90 = "n90"  # minutes / 90
MATCHES = "matches"
STARTS = "starts"

# Events
GOALS = "goals"  # all goals incl. penalties
PK = "pk"  # penalties scored
NPG = "npg"  # non-penalty goals
ASSISTS = "assists"
SHOTS = "shots"
SOT = "sot"  # shots on target

# Rates (per 90)
NPG90 = "npg90"
AST90 = "ast90"
SH90 = "sh90"

PANEL_COLUMNS = [
    PLAYER, BORN, SEASON, AGE, LEAGUE, TEAM, POS,
    MINUTES, N90, MATCHES, STARTS,
    GOALS, PK, NPG, ASSISTS, SHOTS, SOT,
    NPG90, AST90, SH90,
]

POSITION_GROUPS = ["GK", "DF", "MF", "FW"]

# A single domestic Big-5 league season is at most 38 matches (34 in the
# Bundesliga). 38 * 90 = 3420; allow headroom for stoppage-time inflation in
# FBref's minute totals. A (player, born) key exceeding this in one season is
# implausible for one human and flags a probable identity merge of two players
# sharing both name and birth year.
MAX_PLAUSIBLE_SEASON_MINUTES = 4200
