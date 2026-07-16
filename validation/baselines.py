"""The baselines the model has to beat to be worth anything.

A projection system that cannot outperform "he'll do what he did last year" is
an expensive way to draw a fan chart. Both baselines are fitted on training
seasons only, exactly like the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import schema as S


def persistence(base_rows: pd.DataFrame, events_col: str) -> np.ndarray:
    """"Next season looks like last season" — the base season's raw count."""
    return base_rows[events_col].to_numpy(dtype=float)


def fit_cohort_means(
    panel: pd.DataFrame, train_seasons: list[int], events_col: str
) -> tuple[dict[tuple[str, int], float], float]:
    """Mean season count per (position, age), from training seasons only.

    Includes zero-minute absences? No — this is the mean over players who
    appear, which is what "an average player of his age and position" means.
    """
    train = panel[panel[S.SEASON].isin(train_seasons)]
    cells: dict[tuple[str, int], float] = {}
    for (pos, age), grp in train.groupby([S.POS, S.AGE]):
        cells[(pos, int(age))] = float(grp[events_col].mean())
    overall = float(train[events_col].mean())
    return cells, overall


def cohort_mean(
    base_rows: pd.DataFrame,
    cells: dict[tuple[str, int], float],
    overall: float,
    horizon: int,
) -> np.ndarray:
    """"Becomes an average player of his age and position" at the target age."""
    out = np.empty(len(base_rows), dtype=float)
    for i, row in enumerate(base_rows.itertuples(index=False)):
        pos = getattr(row, S.POS)
        target_age = int(getattr(row, S.AGE)) + horizon
        out[i] = cells.get((pos, target_age), overall)
    return out
