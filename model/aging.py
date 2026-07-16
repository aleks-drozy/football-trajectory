"""Aging curves by the delta method, in two variants.

The delta method pairs a player's consecutive seasons, averages the change in a
rate at each age, and chains those changes into a curve. Its well-known flaw is
**survivor bias**: a player who declines loses his place and never supplies his
next delta, so the surviving pairs make aging look kinder than it is.

Two variants are built here and the choice between them is deferred to
out-of-sample validation (DESIGN.md §5.3, §7) rather than to taste:

``survivors``
    Only players meeting the minutes floor in *both* seasons. This estimates
    aging **conditional on still playing**. Paired with a minutes model that can
    independently send a player to zero, that conditional is coherent — but the
    players who keep their place are the ones who held up, so it still skews
    optimistic.

``dropouts``
    Every player meeting the floor at age *a* is retained. If he is missing at
    *a+1* he enters at replacement level instead of vanishing, so decline that
    manifests as disappearance is not silently discarded.

Rates differ by an order of magnitude across positions (a full-back's goal rate
is not a striker's), so curves are estimated **per position group** and applied
additively within it. Sparse (position, age) cells are smoothed across adjacent
ages, and every cell's sample size is retained for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import schema as S

# A season below this is too noisy to anchor a delta on.
DEFAULT_MIN_N90 = 5.0

# Replacement level is the weak end of actual observed play rather than an
# invented constant: a low quantile of the *rate* distribution among players
# with real workload at that position.
#
# It is deliberately a rate quantile, not "the mean rate of low-minute players".
# The latter is not reliably weak — low-minute players include hot young
# substitutes with inflated per-90 rates — and imputing an above-average rate to
# a player who vanished would assert that disappearing improved him, inverting
# the very bias this variant exists to correct.
REPLACEMENT_RATE_QUANTILE = 0.20


@dataclass(frozen=True)
class AgingCurve:
    """Chained additive aging curve for one metric, per position group."""

    metric: str
    variant: str
    # {pos: {age: cumulative offset relative to REFERENCE_AGE}}
    curve: dict[str, dict[int, float]]
    # {pos: {age: n paired observations behind the delta into that age}}
    support: dict[str, dict[int, int]] = field(default_factory=dict)
    reference_age: int = 21

    def offset(self, pos: str, age: int) -> float:
        """Cumulative aging offset at `age`, relative to the reference age.

        Ages beyond the estimated range clamp to the nearest estimated age
        rather than extrapolating a trend the data never supported.
        """
        by_age = self.curve.get(pos)
        if not by_age:
            return 0.0
        if age in by_age:
            return by_age[age]
        ages = sorted(by_age)
        if age < ages[0]:
            return by_age[ages[0]]
        return by_age[ages[-1]]

    def delta(self, pos: str, from_age: int, to_age: int) -> float:
        """Expected additive change in the metric moving between two ages."""
        return self.offset(pos, to_age) - self.offset(pos, from_age)


def replacement_rates(
    panel: pd.DataFrame, metric: str, min_n90: float = DEFAULT_MIN_N90
) -> dict[str, float]:
    """Low quantile of the rate distribution per position — the weak end.

    Only players clearing the workload floor count, so the quantile describes
    genuine regulars performing poorly rather than cameo appearances.
    """
    out: dict[str, float] = {}
    qualified = panel[panel[S.N90] >= min_n90]
    for pos, grp in qualified.groupby(S.POS):
        if grp.empty:
            continue
        out[pos] = float(grp[metric].quantile(REPLACEMENT_RATE_QUANTILE))
    return out


def build_pairs(
    panel: pd.DataFrame,
    metric: str,
    variant: str = "survivors",
    min_n90: float = DEFAULT_MIN_N90,
    max_season: int | None = None,
) -> pd.DataFrame:
    """Consecutive-season pairs with the metric delta and its weight.

    `max_season` censors the panel: a player absent at a+1 only counts as a
    dropout if season a+1 exists in the data at all. Without that guard the
    final season of the dataset would manufacture a league-wide phantom
    retirement.
    """
    if variant not in ("survivors", "dropouts"):
        raise ValueError(f"unknown variant {variant!r}")

    df = panel if max_season is None else panel[panel[S.SEASON] <= max_season]
    last_season = int(df[S.SEASON].max())

    cur = df[df[S.N90] >= min_n90][
        [S.PLAYER, S.BORN, S.SEASON, S.AGE, S.POS, S.N90, metric]
    ].copy()
    cur["_next_season"] = cur[S.SEASON] + 1

    nxt = df[[S.PLAYER, S.BORN, S.SEASON, S.N90, metric]].copy()
    nxt = nxt.rename(
        columns={S.SEASON: "_next_season", S.N90: "n90_next", metric: "metric_next"}
    )

    pairs = cur.merge(nxt, on=[S.PLAYER, S.BORN, "_next_season"], how="left")

    # A pair whose follow-up season lies outside the data is censored, not a
    # dropout, and cannot inform aging either way.
    pairs = pairs[pairs["_next_season"] <= last_season]

    present = pairs["n90_next"].notna()
    if variant == "survivors":
        pairs = pairs[present & (pairs["n90_next"] >= min_n90)].copy()
        weight_next = pairs["n90_next"]
    else:
        repl = replacement_rates(df, metric, min_n90)
        pairs = pairs.copy()
        imputed = pairs[S.POS].map(repl).astype(float)
        pairs["metric_next"] = pairs["metric_next"].where(present, imputed)
        # A dropout is credited the minimum qualifying workload so his decline
        # carries real weight instead of being rounded away.
        pairs["n90_next"] = pairs["n90_next"].where(present, min_n90)
        weight_next = pairs["n90_next"]

    pairs["delta"] = pairs["metric_next"] - pairs[metric]
    # Harmonic mean of the two workloads: a delta is only as trustworthy as the
    # smaller of the seasons behind it.
    a = pairs[S.N90].to_numpy(dtype=float)
    b = weight_next.to_numpy(dtype=float)
    pairs["weight"] = np.where((a + b) > 0, 2 * a * b / (a + b), 0.0)
    return pairs[pairs["weight"] > 0].reset_index(drop=True)


def _smooth(values: dict[int, float], support: dict[int, int], min_support: int) -> dict[int, float]:
    """Weighted 3-point smooth across adjacent ages, for thin cells only."""
    ages = sorted(values)
    out: dict[int, float] = {}
    for age in ages:
        if support.get(age, 0) >= min_support:
            out[age] = values[age]
            continue
        num = 0.0
        den = 0.0
        for neighbour in (age - 1, age, age + 1):
            if neighbour in values:
                w = float(support.get(neighbour, 0)) + 1.0
                num += values[neighbour] * w
                den += w
        out[age] = num / den if den else values[age]
    return out


def fit_aging_curve(
    panel: pd.DataFrame,
    metric: str,
    variant: str = "survivors",
    min_n90: float = DEFAULT_MIN_N90,
    reference_age: int = 21,
    min_cell: int = 20,
    max_season: int | None = None,
) -> AgingCurve:
    """Estimate the chained aging curve for one metric."""
    pairs = build_pairs(panel, metric, variant=variant, min_n90=min_n90, max_season=max_season)

    curve: dict[str, dict[int, float]] = {}
    support: dict[str, dict[int, int]] = {}

    for pos, grp in pairs.groupby(S.POS):
        if pos == "GK" and metric in (S.NPG90, S.SH90):
            continue  # goalkeepers do not have a meaningful scoring curve
        # Mean delta for the step age -> age+1, weighted by workload.
        step: dict[int, float] = {}
        cell_n: dict[int, int] = {}
        for age, cell in grp.groupby(S.AGE):
            w = cell["weight"].to_numpy()
            if w.sum() <= 0:
                continue
            step[int(age)] = float(np.average(cell["delta"].to_numpy(), weights=w))
            cell_n[int(age)] = len(cell)

        if not step:
            continue
        step = _smooth(step, cell_n, min_cell)

        # Chain the per-age steps into a cumulative curve by walking the
        # observed age range contiguously from its lowest age. Walking outward
        # from `reference_age` instead would break the chain whenever the
        # reference age is absent from the data or a step is missing, silently
        # leaving neighbouring ages unset.
        ages = sorted(step)
        lo, hi = min(ages), max(ages) + 1
        cum: dict[int, float] = {lo: 0.0}
        for age in range(lo, hi):
            # A missing step means no evidence of change at that age, so the
            # curve is held flat across it rather than severed.
            cum[age + 1] = cum[age] + step.get(age, 0.0)

        # Re-anchor so the reference age sits at zero. If the reference age was
        # never observed, anchor at the nearest age that was — the curve is only
        # ever used through `delta`, which is invariant to the anchor.
        anchor = reference_age if reference_age in cum else min(
            cum, key=lambda a: abs(a - reference_age)
        )
        base = cum[anchor]
        curve[pos] = {age: value - base for age, value in sorted(cum.items())}
        support[pos] = cell_n

    return AgingCurve(
        metric=metric,
        variant=variant,
        curve=curve,
        support=support,
        reference_age=reference_age,
    )
