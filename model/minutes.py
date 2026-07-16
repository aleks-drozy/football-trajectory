"""Availability model: next-season minutes, sampled non-parametrically.

Minutes are not a tidy distribution. They are strongly bimodal (a regular
starter lives near 2,700; a squad player near 800), truncated at both ends, and
shaped by injury, rotation, loan and transfer. Fitting a parametric shape to
that would impose structure the data does not have.

Instead, the empirical conditional distribution of next-season minutes is
sampled directly. Injuries, benching and departures need no separate model —
they are already in the observed transitions.

Zero is a **real, retained outcome**. A player absent from the Big-5 next season
is a genuine zero for this project's scope (DESIGN.md §3), so absences are
materialised as zero-minute rows rather than dropped. Two guards keep that
honest:

- Materialisation starts at a player's first appearance — a 16-year-old not yet
  in the Big-5 is not "a zero", he is not yet observed.
- A transition only counts when the follow-up season exists in the data.
  Otherwise the final season of the panel would invent a league-wide phantom
  retirement.

**Conditioning on talent (v2).** v1 conditioned on (minutes, age) alone, which
handed a generational teenager the dropout hazard of the *average* teenager with
similar minutes — the single largest known flaw in the v1 results (WRITEUP §4).
Good players keep their place; that is most of what availability is. So
transitions are now also conditioned on a **talent tercile**.

Three choices make that work without leaking or collapsing:

- **The measure is attacking output per 90** (non-penalty goals + assists),
  which is the only quality signal this dataset carries. It is informative for
  forwards and midfielders and near-meaningless for defenders and keepers — for
  them the terciles simply carry little signal, which costs nothing.
- **Cut points are absolute within position, not age-relative.** Whether a player
  keeps his place depends on how good he is, not on how good he is *for his
  age*: a 0.20/90 striker is squeezed out at 19 and at 29 alike. Age-relative
  cuts would say a teenager stops being good the moment he turns 24.
- **Talent is carried forward from the last qualifying season**, never taken from
  the future. A player sitting at zero minutes keeps the talent he last
  demonstrated, which is exactly what decides whether he comes back.
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

# A season below this is too noisy to read a talent level from.
TALENT_MIN_N90 = 5.0

# Tercile boundaries. Terciles (not deciles) because the extra conditioning
# dimension multiplies the cell count by 3 — finer splits would push most cells
# under MIN_CELL and back off to the unconditioned pool anyway.
TALENT_QUANTILES = (1 / 3, 2 / 3)

# Used when a player has no qualifying season yet — neutral rather than an
# assumption in either direction.
UNKNOWN_TERCILE = 1


def minute_bucket(minutes: float) -> int:
    return int(np.searchsorted(MINUTE_BUCKET_EDGES, minutes, side="right") - 1)


def age_bucket_idx(age: int) -> int:
    return int(np.searchsorted(AGE_BUCKET_EDGES, age, side="right") - 1)


def attacking_rate(npg: float, assists: float, n90: float) -> float:
    return (npg + assists) / n90 if n90 > 0 else float("nan")


def fit_talent_cuts(
    panel: pd.DataFrame, min_n90: float = TALENT_MIN_N90
) -> dict[str, tuple[float, float]]:
    """Tercile boundaries of attacking output per 90, per position.

    Only seasons clearing the workload floor contribute, so the cuts describe
    regulars rather than cameo per-90 rates.
    """
    qualified = panel[panel[S.N90] >= min_n90].copy()
    qualified["_attack"] = (qualified[S.NPG] + qualified[S.ASSISTS]) / qualified[S.N90]
    cuts: dict[str, tuple[float, float]] = {}
    for pos, grp in qualified.groupby(S.POS):
        lo, hi = grp["_attack"].quantile(list(TALENT_QUANTILES))
        cuts[pos] = (float(lo), float(hi))
    return cuts


def talent_tercile(rate: float, pos: str, cuts: dict[str, tuple[float, float]]) -> int:
    """0 = weak, 1 = middling, 2 = strong, within the player's own position."""
    if rate is None or not np.isfinite(rate):
        return UNKNOWN_TERCILE
    bounds = cuts.get(pos)
    if bounds is None:
        return UNKNOWN_TERCILE
    lo, hi = bounds
    if rate < lo:
        return 0
    return 1 if rate < hi else 2


def talent_terciles_vec(
    rates: np.ndarray, pos: str, cuts: dict[str, tuple[float, float]]
) -> np.ndarray:
    """Vectorised `talent_tercile` for a whole set of simulated paths."""
    bounds = cuts.get(pos)
    if bounds is None:
        return np.full(rates.shape, UNKNOWN_TERCILE, dtype=int)
    lo, hi = bounds
    out = np.full(rates.shape, UNKNOWN_TERCILE, dtype=int)
    finite = np.isfinite(rates)
    out[finite & (rates < lo)] = 0
    out[finite & (rates >= lo) & (rates < hi)] = 1
    out[finite & (rates >= hi)] = 2
    return out


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
                        S.NPG: 0.0,
                        S.ASSISTS: 0.0,
                    }
                )
    return pd.DataFrame(rows)


def attach_carried_talent(
    full: pd.DataFrame, cuts: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Add the talent tercile a player had *entering* each season.

    The rate is read only from seasons clearing the workload floor and then
    carried forward, so a zero-minute year keeps the talent last demonstrated —
    which is what actually decides whether he comes back — and nothing is ever
    read from the future.
    """
    df = full.sort_values([S.PLAYER, S.BORN, S.SEASON]).copy()
    qualifies = df[S.N90] >= TALENT_MIN_N90
    rate = (df[S.NPG] + df[S.ASSISTS]) / df[S.N90].where(df[S.N90] > 0)
    df["_rate_obs"] = rate.where(qualifies)
    df["_rate_known"] = df.groupby([S.PLAYER, S.BORN])["_rate_obs"].ffill()
    df["_tt"] = [
        talent_tercile(r, p, cuts) for r, p in zip(df["_rate_known"], df[S.POS])
    ]
    return df


@dataclass
class MinutesModel:
    """Empirical conditional sampler for next-season minutes."""

    # (minute_bucket, age_bucket, talent_tercile) -> observed next-season minutes
    cells: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    # (minute_bucket, age_bucket) -> pooled across talent
    by_minutes_age: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # minute_bucket -> pooled fallback across ages
    by_minutes: dict[int, np.ndarray] = field(default_factory=dict)
    # age_bucket -> pooled fallback across minutes
    by_age: dict[int, np.ndarray] = field(default_factory=dict)
    overall: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    talent_cuts: dict[str, tuple[float, float]] = field(default_factory=dict)

    def pool_for(self, mb: int, ab: int, tt: int | None) -> np.ndarray:
        """The most specific pool with enough support, backing off in order."""
        pool = self.cells.get((mb, ab, tt)) if tt is not None else None
        if pool is None or len(pool) < MIN_CELL:
            pool = self.by_minutes_age.get((mb, ab))
        if pool is None or len(pool) < MIN_CELL:
            pool = self.by_minutes.get(mb)
        if pool is None or len(pool) < MIN_CELL:
            pool = self.by_age.get(ab)
        if pool is None or len(pool) == 0:
            pool = self.overall
        return pool

    def sample(
        self,
        minutes: float,
        age: int,
        rng: np.random.Generator,
        size: int = 1,
        attack_rate: float | None = None,
        pos: str | None = None,
    ) -> np.ndarray:
        """Draw next-season minutes.

        `attack_rate`/`pos` are optional: omit them and the draw falls back to
        the unconditioned (minutes, age) pool rather than silently assuming a
        talent level.
        """
        mb, ab = minute_bucket(minutes), age_bucket_idx(age)
        tt = (
            talent_tercile(attack_rate, pos, self.talent_cuts)
            if attack_rate is not None and pos is not None
            else None
        )
        return rng.choice(self.pool_for(mb, ab, tt), size=size, replace=True)

    def cell_support(self) -> dict[tuple[int, int, int], int]:
        return {k: len(v) for k, v in self.cells.items()}

    def talent_conditioned_share(self) -> float:
        """Share of (minutes, age, talent) cells with enough support to be used.

        Reported rather than assumed: if this is low the extra dimension is
        mostly backing off and buying nothing.
        """
        support = self.cell_support()
        if not support:
            return 0.0
        return float(np.mean([n >= MIN_CELL for n in support.values()]))


def fit_minutes_model(panel: pd.DataFrame, max_season: int | None = None) -> MinutesModel:
    scope = panel if max_season is None else panel[panel[S.SEASON] <= max_season]
    cuts = fit_talent_cuts(scope)

    full = materialise_absences(panel, max_season=max_season)
    full = attach_carried_talent(full, cuts)
    last_season = int(full[S.SEASON].max())

    cur = full[[S.PLAYER, S.BORN, S.SEASON, S.AGE, S.MINUTES, "_tt"]].copy()
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

    model = MinutesModel(talent_cuts=cuts)
    for (mb, ab, tt), grp in pairs.groupby(["_mb", "_ab", "_tt"]):
        model.cells[(int(mb), int(ab), int(tt))] = grp["minutes_next"].to_numpy(dtype=float)
    for (mb, ab), grp in pairs.groupby(["_mb", "_ab"]):
        model.by_minutes_age[(int(mb), int(ab))] = grp["minutes_next"].to_numpy(dtype=float)
    for mb, grp in pairs.groupby("_mb"):
        model.by_minutes[int(mb)] = grp["minutes_next"].to_numpy(dtype=float)
    for ab, grp in pairs.groupby("_ab"):
        model.by_age[int(ab)] = grp["minutes_next"].to_numpy(dtype=float)
    model.overall = pairs["minutes_next"].to_numpy(dtype=float)
    return model
