"""Availability model: next-season minutes, sampled non-parametrically.

Minutes are not a tidy distribution. They are strongly bimodal (a regular
starter lives near 2,700; a squad player near 800), truncated at both ends, and
shaped by injury, rotation, loan and transfer. Fitting a parametric shape to
that would impose structure the data does not have.

Instead, the empirical conditional distribution of next-season minutes is
sampled directly, given (current minutes bucket, age bucket). Injuries, benching
and departures need no separate model — they are already in the observed
transitions.

Zero is a **real, retained outcome**. A player absent from the Big-5 next season
is a genuine zero for this project's scope (DESIGN.md §3), so absences are
materialised as zero-minute rows rather than dropped. Two guards keep that
honest:

- Materialisation starts at a player's first appearance — a 16-year-old not yet
  in the Big-5 is not "a zero", he is not yet observed.
- A transition only counts when the follow-up season exists in the data.
  Otherwise the final season of the panel would invent a league-wide phantom
  retirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import schema as S

MINUTE_BUCKET_EDGES = (0.0, 1.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, np.inf)
AGE_BUCKET_EDGES = (0, 20, 22, 24, 27, 30, 33, 99)

# Below this many observed transitions a cell is too thin to sample from and
# the model falls back to a coarser conditioning.
MIN_CELL = 25


def minute_bucket(minutes: float) -> int:
    return int(np.searchsorted(MINUTE_BUCKET_EDGES, minutes, side="right") - 1)


def age_bucket_idx(age: int) -> int:
    return int(np.searchsorted(AGE_BUCKET_EDGES, age, side="right") - 1)


def materialise_absences(panel: pd.DataFrame, max_season: int | None = None) -> pd.DataFrame:
    """Fill each player's gap seasons with explicit zero-minute rows.

    Spans a player's first appearance through the last season in the data, so
    that "left the Big-5" and "returned after a year out" are both observable
    transitions rather than silent holes.
    """
    df = panel if max_season is None else panel[panel[S.SEASON] <= max_season]
    last_season = int(df[S.SEASON].max())

    rows: list[dict] = []
    for (player, born), grp in df.groupby([S.PLAYER, S.BORN], sort=False):
        seen = {int(s): r for s, r in zip(grp[S.SEASON], grp.to_dict("records"))}
        first = min(seen)
        pos_default = grp.loc[grp[S.MINUTES].idxmax(), S.POS]
        for season in range(first, last_season + 1):
            if season in seen:
                rows.append(seen[season])
            else:
                rows.append(
                    {
                        S.PLAYER: player,
                        S.BORN: born,
                        S.SEASON: season,
                        S.AGE: season - born,
                        S.POS: pos_default,
                        S.MINUTES: 0.0,
                        S.N90: 0.0,
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class MinutesModel:
    """Empirical conditional sampler for next-season minutes."""

    # (minute_bucket, age_bucket) -> observed next-season minutes
    cells: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # minute_bucket -> pooled fallback across ages
    by_minutes: dict[int, np.ndarray] = field(default_factory=dict)
    # age_bucket -> pooled fallback across minutes
    by_age: dict[int, np.ndarray] = field(default_factory=dict)
    overall: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def sample(self, minutes: float, age: int, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw next-season minutes, backing off when a cell is too thin."""
        mb, ab = minute_bucket(minutes), age_bucket_idx(age)
        pool = self.cells.get((mb, ab))
        if pool is None or len(pool) < MIN_CELL:
            pool = self.by_minutes.get(mb)
        if pool is None or len(pool) < MIN_CELL:
            pool = self.by_age.get(ab)
        if pool is None or len(pool) == 0:
            pool = self.overall
        return rng.choice(pool, size=size, replace=True)

    def cell_support(self) -> dict[tuple[int, int], int]:
        return {k: len(v) for k, v in self.cells.items()}


def fit_minutes_model(panel: pd.DataFrame, max_season: int | None = None) -> MinutesModel:
    full = materialise_absences(panel, max_season=max_season)
    last_season = int(full[S.SEASON].max())

    cur = full[[S.PLAYER, S.BORN, S.SEASON, S.AGE, S.MINUTES]].copy()
    cur["_next_season"] = cur[S.SEASON] + 1
    nxt = full[[S.PLAYER, S.BORN, S.SEASON, S.MINUTES]].rename(
        columns={S.SEASON: "_next_season", S.MINUTES: "minutes_next"}
    )
    pairs = cur.merge(nxt, on=[S.PLAYER, S.BORN, "_next_season"], how="left")

    # Only score transitions whose follow-up season is actually in the data.
    pairs = pairs[pairs["_next_season"] <= last_season]
    # A player still inside his span but absent next season is a true zero.
    pairs["minutes_next"] = pairs["minutes_next"].fillna(0.0)

    pairs["_mb"] = pairs[S.MINUTES].map(minute_bucket)
    pairs["_ab"] = pairs[S.AGE].map(age_bucket_idx)

    model = MinutesModel()
    for (mb, ab), grp in pairs.groupby(["_mb", "_ab"]):
        model.cells[(int(mb), int(ab))] = grp["minutes_next"].to_numpy(dtype=float)
    for mb, grp in pairs.groupby("_mb"):
        model.by_minutes[int(mb)] = grp["minutes_next"].to_numpy(dtype=float)
    for ab, grp in pairs.groupby("_ab"):
        model.by_age[int(ab)] = grp["minutes_next"].to_numpy(dtype=float)
    model.overall = pairs["minutes_next"].to_numpy(dtype=float)
    return model
