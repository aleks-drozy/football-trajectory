"""True-talent estimation by empirical-Bayes shrinkage (Marcel-style).

A teenager's observed goal rate is mostly noise: a striker with 4 goals in 700
minutes has a 0.51/90 rate and almost no evidence for it. Shrinking toward a
cohort prior in proportion to workload is what turns that into an estimate:

    talent = (Σ_s w_s·events_s + K·prior) / (Σ_s w_s·n90_s + K)

`K` is the shrinkage constant in units of 90s — the workload at which observed
play and the prior carry equal weight. **`K` is fitted from training data**, not
chosen. Picking `K` by eye is the quiet cheat in this genre, because it is the
single knob that decides whether a hot teenage sample is believed.

Finishing regresses harder than shot volume, so shot rate is used as a
co-signal. FBref's Big-5 Combined tables expose no xG (DESIGN.md §A1), so
shots/90 stands in for npxG/90: it plays the same statistical role of a stable
underlying signal behind noisy conversion. The blend weight λ is fitted too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import schema as S
from model.aging import AgingCurve

# Most recent season first. Marcel-style geometric-ish decay: recent play is
# more informative about current talent than three-year-old play.
DEFAULT_RECENCY_WEIGHTS = (5.0, 4.0, 3.0)

AGE_BUCKETS = ((0, 19), (20, 21), (22, 24), (25, 28), (29, 31), (32, 99))

K_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 25.0, 35.0, 50.0, 75.0, 110.0, 160.0)
LAMBDA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def age_bucket(age: int) -> tuple[int, int]:
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return (lo, hi)
    return AGE_BUCKETS[-1]


def fit_priors(panel: pd.DataFrame, metric: str) -> dict[tuple[str, tuple[int, int]], float]:
    """Minutes-weighted cohort mean rate per (position, age bucket)."""
    df = panel[panel[S.N90] > 0].copy()
    df["_bucket"] = df[S.AGE].map(age_bucket)
    priors: dict[tuple[str, tuple[int, int]], float] = {}
    for (pos, bucket), grp in df.groupby([S.POS, "_bucket"]):
        w = grp[S.N90].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        priors[(pos, bucket)] = float(np.average(grp[metric].to_numpy(), weights=w))
    return priors


@dataclass(frozen=True)
class TalentModel:
    """Shrinkage estimator for one metric."""

    metric: str
    events_col: str
    K: float
    priors: dict[tuple[str, tuple[int, int]], float]
    global_prior: float
    recency_weights: tuple[float, ...] = DEFAULT_RECENCY_WEIGHTS

    def prior_for(self, pos: str, age: int) -> float:
        return self.priors.get(
            (pos, age_bucket(age)),
            self.priors.get((pos, AGE_BUCKETS[2]), self.global_prior),
        )

    def posterior(self, history: pd.DataFrame, pos: str, age: int) -> tuple[float, float]:
        """Gamma posterior (shape, rate) over the player's true rate.

        The shrinkage formula is not an ad-hoc weighted average — it is exactly
        the posterior mean of a Gamma-Poisson conjugate pair with prior
        Gamma(K·prior, K) and Poisson(rate · n90) counts. So the same `K` that
        controls shrinkage also delivers principled *uncertainty*, and the
        Monte Carlo can draw talent from a real posterior instead of bolting on
        an invented spread.

        `history` must contain only seasons at or before the estimation point;
        passing future seasons in would leak.
        """
        prior = self.prior_for(pos, age)
        shape = self.K * prior
        rate = self.K
        if not history.empty:
            hist = history.sort_values(S.SEASON, ascending=False)
            for i, (_, row) in enumerate(hist.iterrows()):
                if i >= len(self.recency_weights):
                    break
                w = self.recency_weights[i]
                shape += w * float(row[self.events_col])
                rate += w * float(row[S.N90])
        return shape, rate

    def estimate(self, history: pd.DataFrame, pos: str, age: int) -> float:
        """Posterior-mean shrunk talent (see `posterior`)."""
        shape, rate = self.posterior(history, pos, age)
        return shape / rate

    def sample_talent(
        self, history: pd.DataFrame, pos: str, age: int, rng, size: int = 1
    ):
        """Draw from the talent posterior rather than fixing it at the mean."""
        shape, rate = self.posterior(history, pos, age)
        shape = max(shape, 1e-6)
        return rng.gamma(shape=shape, scale=1.0 / rate, size=size)


def _prediction_frame(
    panel: pd.DataFrame, metric: str, min_n90_target: float
) -> pd.DataFrame:
    """One row per (player, season) whose *next* season is observed with enough
    workload to score a prediction against."""
    nxt = panel[[S.PLAYER, S.BORN, S.SEASON, S.N90, S.AGE, metric]].copy()
    # NB: no leading underscore. DataFrame.itertuples silently renames columns
    # that aren't valid Python identifiers to positional names (_1, _2...), so
    # a column called "_from_season" is unreachable by attribute downstream.
    nxt = nxt.rename(
        columns={
            S.SEASON: "from_season",
            S.N90: "n90_next",
            S.AGE: "age_next",
            metric: "actual_next",
        }
    )
    nxt["from_season"] = nxt["from_season"] - 1
    cur = panel[[S.PLAYER, S.BORN, S.SEASON, S.AGE, S.POS, S.N90]].copy()
    cur = cur.rename(columns={S.SEASON: "from_season"})
    out = cur.merge(nxt, on=[S.PLAYER, S.BORN, "from_season"], how="inner")
    return out[out["n90_next"] >= min_n90_target].reset_index(drop=True)


def _predict_series(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    model: TalentModel,
    curve: AgingCurve | None,
) -> np.ndarray:
    hist_by_player = {
        key: grp for key, grp in panel.groupby([S.PLAYER, S.BORN], sort=False)
    }
    preds = np.empty(len(targets), dtype=float)
    for i, row in enumerate(targets.itertuples(index=False)):
        player = getattr(row, S.PLAYER)
        born = getattr(row, S.BORN)
        from_season = row.from_season
        pos = getattr(row, S.POS)
        age = getattr(row, S.AGE)
        grp = hist_by_player.get((player, born))
        hist = grp[grp[S.SEASON] <= from_season] if grp is not None else panel.iloc[0:0]
        talent = model.estimate(hist, pos, age)
        if curve is not None:
            talent += curve.delta(pos, int(age), int(row.age_next))
        preds[i] = talent
    return preds


def fit_K(
    panel: pd.DataFrame,
    metric: str,
    events_col: str,
    train_seasons: list[int],
    curve: AgingCurve | None = None,
    min_n90_target: float = 5.0,
) -> tuple[float, dict[float, float]]:
    """Grid-search the shrinkage constant on training seasons only.

    Scored by workload-weighted MAE of the next-season prediction, so that a
    player with a full season counts for more than a cameo.
    """
    train = panel[panel[S.SEASON].isin(train_seasons)]
    priors = fit_priors(train, metric)
    global_prior = float(
        np.average(
            train[train[S.N90] > 0][metric].to_numpy(),
            weights=train[train[S.N90] > 0][S.N90].to_numpy(),
        )
    )
    targets = _prediction_frame(train, metric, min_n90_target)
    if targets.empty:
        raise ValueError("no training pairs available to fit K")

    scores: dict[float, float] = {}
    for K in K_GRID:
        model = TalentModel(
            metric=metric,
            events_col=events_col,
            K=K,
            priors=priors,
            global_prior=global_prior,
        )
        preds = _predict_series(train, targets, model, curve)
        err = np.abs(preds - targets["actual_next"].to_numpy())
        scores[K] = float(np.average(err, weights=targets["n90_next"].to_numpy()))
    best = min(scores, key=scores.get)
    return best, scores


def build_talent_model(
    panel: pd.DataFrame,
    metric: str,
    events_col: str,
    train_seasons: list[int],
    K: float,
) -> TalentModel:
    train = panel[panel[S.SEASON].isin(train_seasons)]
    played = train[train[S.N90] > 0]
    return TalentModel(
        metric=metric,
        events_col=events_col,
        K=K,
        priors=fit_priors(train, metric),
        global_prior=float(
            np.average(played[metric].to_numpy(), weights=played[S.N90].to_numpy())
        ),
    )
