"""Record thresholds and career-total probabilities.

Every threshold here carries its source. The comparison is only meaningful if
the units match, and they nearly never do by default:

- Published records count **all** goals including penalties, so record
  comparisons use total goals (`gls90`), not the model's non-penalty unit.
- Published records are **domestic league only** for a single competition
  (Messi's 474 is La Liga; Shearer's 260 is the Premier League). They exclude
  Champions League and domestic cups — which suits this project, whose data is
  Big-5 domestic league only (DESIGN.md §3).
- A career total = goals already scored (observed in the panel) + goals still
  to come (simulated). That only works for players whose whole career falls
  inside the data window, i.e. those who debuted in 2017-18 or later. For anyone
  older the observed part is truncated and the total would be understated.

The last point is a hard gate, not a caveat: `career_totals` refuses to compute
a total for a player whose career predates the panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import schema as S

PANEL_FIRST_SEASON = 2017


@dataclass(frozen=True)
class Record:
    name: str
    competition: str
    holder: str
    goals: int
    source: str
    includes_penalties: bool = True


RECORDS = (
    Record(
        name="La Liga all-time",
        competition="ESP-La Liga",
        holder="Lionel Messi",
        goals=474,
        source="https://en.wikipedia.org/wiki/List_of_La_Liga_top_scorers",
    ),
    Record(
        name="Premier League all-time",
        competition="ENG-Premier League",
        holder="Alan Shearer",
        goals=260,
        source="https://en.wikipedia.org/wiki/List_of_footballers_with_100_or_more_Premier_League_goals",
    ),
    Record(
        name="Serie A all-time",
        competition="ITA-Serie A",
        holder="Silvio Piola",
        goals=274,
        source="https://en.wikipedia.org/wiki/Silvio_Piola",
    ),
    Record(
        name="Bundesliga all-time",
        competition="GER-Bundesliga",
        holder="Gerd Müller",
        goals=365,
        source="https://en.wikipedia.org/wiki/List_of_Bundesliga_top_scorers",
    ),
    Record(
        name="Ligue 1 all-time",
        competition="FRA-Ligue 1",
        holder="Delio Onnis",
        goals=299,
        source="https://en.wikipedia.org/wiki/List_of_Ligue_1_top_scorers",
    ),
)

# Round-number milestones, useful when a player's league has no listed record
# or when the record is far out of reach.
MILESTONES = (50, 100, 150, 200, 250, 300)


def career_is_fully_observed(panel: pd.DataFrame, player: str, born: int) -> bool:
    """True only if the player's first Big-5 season is inside the data window.

    A player who debuted before 2017-18 has goals the panel never saw, so any
    "career total" built from it would silently understate him.
    """
    rows = panel[(panel[S.PLAYER] == player) & (panel[S.BORN] == born)]
    if rows.empty:
        return False
    return int(rows[S.SEASON].min()) > PANEL_FIRST_SEASON


def goals_so_far(panel: pd.DataFrame, player: str, born: int, up_to_season: int) -> int:
    rows = panel[
        (panel[S.PLAYER] == player)
        & (panel[S.BORN] == born)
        & (panel[S.SEASON] <= up_to_season)
    ]
    return int(rows[S.GOALS].sum())


def career_totals(
    panel: pd.DataFrame,
    player: str,
    born: int,
    up_to_season: int,
    simulated_future_goals: np.ndarray,
) -> np.ndarray:
    """Per-path career totals = observed goals + simulated future goals."""
    if not career_is_fully_observed(panel, player, born):
        raise ValueError(
            f"{player} ({born}) debuted at or before {PANEL_FIRST_SEASON}; the panel "
            "does not contain his full career, so a career total would understate him"
        )
    return goals_so_far(panel, player, born, up_to_season) + simulated_future_goals


def probability_of(totals: np.ndarray, threshold: float) -> float:
    return float(np.mean(totals >= threshold))


def record_report(totals: np.ndarray, league: str | None = None) -> dict:
    """P(reaching) each record and milestone, from simulated career totals."""
    out: dict = {
        "median": float(np.median(totals)),
        "p10": float(np.percentile(totals, 10)),
        "p90": float(np.percentile(totals, 90)),
        "milestones": {str(m): probability_of(totals, m) for m in MILESTONES},
        "records": {},
    }
    for rec in RECORDS:
        out["records"][rec.name] = {
            "holder": rec.holder,
            "goals": rec.goals,
            "competition": rec.competition,
            "source": rec.source,
            "same_competition_as_player": (league == rec.competition),
            "probability": probability_of(totals, rec.goals),
        }
    return out
