"""Out-of-sample backtest of the whole projection system.

Fit everything — aging curves, shrinkage constant K, priors, minutes
transitions — on training seasons only, then project held-out players forward
and compare against what actually happened.

The leakage rule is absolute: nothing fitted may touch a season used for
scoring. `fit_system` takes `train_seasons` and passes `max_season` down into
every component that pairs consecutive seasons.

Absence is scored as zero, not as missing. A player who left the Big-5 really
did score zero Big-5 goals (DESIGN.md §3), and the model is expected to predict
that through the minutes path. Dropping those rows would quietly grade the model
only on the players who stuck around — the same survivor bias the aging curves
are built to resist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import schema as S
from mc.simulate import PlayerState, bootstrap_curves, simulate_player
from model.aging import AgingCurve, fit_aging_curve
from model.minutes import MinutesModel, fit_minutes_model
from model.talent import TalentModel, build_talent_model, fit_K


@dataclass
class FittedSystem:
    variant: str
    train_seasons: list[int]
    npg_model: TalentModel
    ast_model: TalentModel
    npg_curves: list[AgingCurve]
    ast_curves: list[AgingCurve]
    minutes_model: MinutesModel
    K_npg: float
    K_ast: float
    K_scores_npg: dict[float, float]
    K_scores_ast: dict[float, float]


def fit_system(
    panel: pd.DataFrame,
    train_seasons: list[int],
    variant: str = "survivors",
    n_boot: int = 30,
    seed: int = 0,
) -> FittedSystem:
    """Fit every component on `train_seasons` only."""
    max_season = max(train_seasons)
    train = panel[panel[S.SEASON].isin(train_seasons)]

    npg_curve = fit_aging_curve(train, S.NPG90, variant=variant, max_season=max_season)
    ast_curve = fit_aging_curve(train, S.AST90, variant=variant, max_season=max_season)

    K_npg, scores_npg = fit_K(panel, S.NPG90, S.NPG, train_seasons, curve=npg_curve)
    K_ast, scores_ast = fit_K(panel, S.AST90, S.ASSISTS, train_seasons, curve=ast_curve)

    npg_model = build_talent_model(panel, S.NPG90, S.NPG, train_seasons, K_npg)
    ast_model = build_talent_model(panel, S.AST90, S.ASSISTS, train_seasons, K_ast)

    npg_curves = bootstrap_curves(
        train, S.NPG90, variant, n_boot=n_boot, seed=seed, max_season=max_season
    ) or [npg_curve]
    ast_curves = bootstrap_curves(
        train, S.AST90, variant, n_boot=n_boot, seed=seed + 1, max_season=max_season
    ) or [ast_curve]

    return FittedSystem(
        variant=variant,
        train_seasons=sorted(train_seasons),
        npg_model=npg_model,
        ast_model=ast_model,
        npg_curves=npg_curves,
        ast_curves=ast_curves,
        minutes_model=fit_minutes_model(train, max_season=max_season),
        K_npg=K_npg,
        K_ast=K_ast,
        K_scores_npg=scores_npg,
        K_scores_ast=scores_ast,
    )


def build_cohort(panel: pd.DataFrame, base_season: int, min_n90: float = 5.0) -> pd.DataFrame:
    """Players with enough base-season workload to project from."""
    base = panel[(panel[S.SEASON] == base_season) & (panel[S.N90] >= min_n90)]
    return base.reset_index(drop=True)


def actuals_at(
    panel: pd.DataFrame, cohort: pd.DataFrame, target_season: int, events_col: str
) -> np.ndarray:
    """Actual counts in `target_season`; absence from the Big-5 counts as zero."""
    tgt = panel[panel[S.SEASON] == target_season][[S.PLAYER, S.BORN, events_col]]
    merged = cohort[[S.PLAYER, S.BORN]].merge(tgt, on=[S.PLAYER, S.BORN], how="left")
    return merged[events_col].fillna(0.0).to_numpy(dtype=float)


def project_cohort(
    panel: pd.DataFrame,
    system: FittedSystem,
    cohort: pd.DataFrame,
    horizon: int,
    n_sims: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate every cohort player forward, returning per-season sim draws.

    Returns arrays shaped [n_players, n_sims, horizon] for goals and assists.
    History is sliced to the base season so no future information leaks in.
    """
    rng = np.random.default_rng(seed)
    base_season = int(cohort[S.SEASON].iloc[0])
    hist_all = panel[panel[S.SEASON] <= base_season]
    hist_by_player = {k: g for k, g in hist_all.groupby([S.PLAYER, S.BORN], sort=False)}

    goals = np.zeros((len(cohort), n_sims, horizon))
    assists = np.zeros((len(cohort), n_sims, horizon))
    minutes = np.zeros((len(cohort), n_sims, horizon))

    for i, row in enumerate(cohort.itertuples(index=False)):
        key = (getattr(row, S.PLAYER), getattr(row, S.BORN))
        state = PlayerState(
            player=key[0],
            born=key[1],
            pos=getattr(row, S.POS),
            age=int(getattr(row, S.AGE)),
            minutes=float(getattr(row, S.MINUTES)),
            history=hist_by_player.get(key, hist_all.iloc[0:0]),
        )
        res = simulate_player(
            state,
            system.npg_model,
            system.ast_model,
            system.npg_curves,
            system.ast_curves,
            system.minutes_model,
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
        )
        goals[i] = res.goals
        assists[i] = res.assists
        minutes[i] = res.minutes

    return {"goals": goals, "assists": assists, "minutes": minutes}
