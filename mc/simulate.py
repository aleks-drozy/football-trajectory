"""Monte Carlo career simulation.

One simulated path per draw:

1. Draw true talent from its **Gamma posterior** (not the point estimate) — a
   teenager's talent is genuinely uncertain and that uncertainty must propagate.
2. For each future season: age the rate along the aging curve, sample minutes
   from the empirical availability model, then draw events.

Three distinct sources of uncertainty are kept separate rather than merged into
one hand-tuned spread:

- **talent** — Gamma posterior (narrows with career workload)
- **availability** — empirical minutes transitions (includes zero)
- **event noise** — Poisson draws given rate and workload

A fourth, **aging-curve uncertainty**, enters by passing a bootstrap ensemble of
curves; each path is assigned one curve. Without it the simulation would treat
the aging curve as known exactly, which it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import schema as S
from model.aging import AgingCurve
from model.minutes import MinutesModel, age_bucket_idx, minute_bucket
from model.talent import TalentModel


@dataclass(frozen=True)
class PlayerState:
    """Everything known about a player at the moment of projection."""

    player: str
    born: int
    pos: str
    age: int
    minutes: float
    history: pd.DataFrame  # seasons at or before the projection point only


@dataclass(frozen=True)
class SimResult:
    """Per-path, per-season simulated output. Shape [n_sims, horizon]."""

    player: str
    minutes: np.ndarray
    goals: np.ndarray
    assists: np.ndarray
    base_age: int

    @property
    def n_sims(self) -> int:
        return self.minutes.shape[0]

    @property
    def horizon(self) -> int:
        return self.minutes.shape[1]

    def cumulative_goals(self) -> np.ndarray:
        return self.goals.cumsum(axis=1)

    def totals(self) -> dict[str, np.ndarray]:
        return {
            "goals": self.goals.sum(axis=1),
            "assists": self.assists.sum(axis=1),
            "minutes": self.minutes.sum(axis=1),
        }


def _sample_minutes_vec(
    model: MinutesModel, prev_minutes: np.ndarray, age: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample next-season minutes for every path at once.

    Paths are grouped by minute bucket so each empirical pool is drawn from in
    one vectorised call rather than per path.
    """
    out = np.empty(prev_minutes.shape[0], dtype=float)
    ab = age_bucket_idx(age)
    buckets = np.array([minute_bucket(m) for m in prev_minutes])
    for mb in np.unique(buckets):
        idx = np.flatnonzero(buckets == mb)
        pool = model.cells.get((int(mb), ab))
        from model.minutes import MIN_CELL

        if pool is None or len(pool) < MIN_CELL:
            pool = model.by_minutes.get(int(mb))
        if pool is None or len(pool) < MIN_CELL:
            pool = model.by_age.get(ab)
        if pool is None or len(pool) == 0:
            pool = model.overall
        out[idx] = rng.choice(pool, size=idx.size, replace=True)
    return out


def simulate_player(
    state: PlayerState,
    npg_model: TalentModel,
    ast_model: TalentModel,
    npg_curves: list[AgingCurve],
    ast_curves: list[AgingCurve],
    minutes_model: MinutesModel,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
) -> SimResult:
    """Simulate `n_sims` futures of `horizon` seasons for one player."""
    if not npg_curves or not ast_curves:
        raise ValueError("at least one aging curve required per metric")

    talent_npg = npg_model.sample_talent(state.history, state.pos, state.age, rng, n_sims)
    talent_ast = ast_model.sample_talent(state.history, state.pos, state.age, rng, n_sims)

    # Each path commits to one draw of the aging curve, so curve uncertainty
    # shows up as spread across paths instead of vanishing. The two ensembles
    # are indexed separately: they are fitted from independent bootstraps and
    # can differ in length, because degenerate resamples are skipped.
    npg_idx = rng.integers(0, len(npg_curves), size=n_sims)
    ast_idx = rng.integers(0, len(ast_curves), size=n_sims)

    minutes = np.full(n_sims, float(state.minutes))
    out_min = np.zeros((n_sims, horizon))
    out_g = np.zeros((n_sims, horizon))
    out_a = np.zeros((n_sims, horizon))

    for h in range(horizon):
        target_age = state.age + h + 1
        minutes = _sample_minutes_vec(minutes_model, minutes, target_age, rng)
        n90 = minutes / 90.0

        d_npg = np.array(
            [c.delta(state.pos, state.age, target_age) for c in npg_curves]
        )[npg_idx]
        d_ast = np.array(
            [c.delta(state.pos, state.age, target_age) for c in ast_curves]
        )[ast_idx]

        # Rates are counts per 90 and cannot be negative; an aging offset that
        # would push a low-rate player below zero floors instead of inverting.
        rate_npg = np.maximum(talent_npg + d_npg, 0.0)
        rate_ast = np.maximum(talent_ast + d_ast, 0.0)

        out_min[:, h] = minutes
        out_g[:, h] = rng.poisson(rate_npg * n90)
        out_a[:, h] = rng.poisson(rate_ast * n90)

    return SimResult(
        player=state.player,
        minutes=out_min,
        goals=out_g,
        assists=out_a,
        base_age=state.age,
    )


def bootstrap_curves(
    panel: pd.DataFrame,
    metric: str,
    variant: str,
    n_boot: int,
    seed: int = 0,
    max_season: int | None = None,
    reference_age: int = 21,
) -> list[AgingCurve]:
    """Resample players (not rows) to build an ensemble of aging curves.

    Players are the independent unit — resampling rows would break the
    within-player correlation across seasons and understate curve uncertainty.
    """
    from model.aging import fit_aging_curve

    rng = np.random.default_rng(seed)
    keys = panel[[S.PLAYER, S.BORN]].drop_duplicates().to_numpy()
    curves: list[AgingCurve] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), size=len(keys))
        picked = pd.DataFrame(keys[idx], columns=[S.PLAYER, S.BORN])
        # merge (not isin) so a player drawn twice contributes twice
        picked["_replica"] = np.arange(len(picked))
        boot = picked.merge(panel, on=[S.PLAYER, S.BORN], how="left")
        # Each replica must become its own identity. Downstream, build_pairs
        # self-merges the panel on (player, born, season) to link consecutive
        # seasons — so a player drawn k times would cross-join against his own
        # copies and yield k^2 pairs instead of k, weighting duplicated players
        # quadratically and inflating the data by orders of magnitude.
        boot[S.PLAYER] = boot[S.PLAYER] + "#r" + boot["_replica"].astype(str)
        boot = boot.drop(columns=["_replica"])
        try:
            curves.append(
                fit_aging_curve(
                    boot, metric, variant=variant, max_season=max_season,
                    reference_age=reference_age,
                )
            )
        except Exception:  # noqa: BLE001 - a degenerate resample is skipped
            continue
    return curves
