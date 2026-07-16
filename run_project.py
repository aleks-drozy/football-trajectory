"""Project young players' careers, labelled with the trust the gate earned.

Run `run_validate.py` first: this script reads `results/gate.json` and stamps
every projection with the verdict. Per DESIGN.md §8, if the model failed
calibration the output is published as a **demonstration of that failure**, not
as a forecast. A well-drawn interval that is not calibrated is a lie with error
bars, and this genre of project fails precisely by shipping one.

Two units are reported:

- **non-penalty goals** — the model's honest unit, and the one the gate scored.
- **total goals** — needed only because published records count penalties.
  Total goals runs through the same machinery but was **not separately gated**,
  so it is at best as trustworthy as the non-penalty result and is labelled so.

Usage:
    python run_project.py            # -> results/projections.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data import schema as S
from data.loader import build_panel
from mc.records import career_is_fully_observed, career_totals, goals_so_far, record_report
from mc.simulate import PlayerState, bootstrap_curves, simulate_player
from model.aging import fit_aging_curve
from model.minutes import fit_minutes_model
from model.talent import build_talent_model, fit_K

RESULTS = Path(__file__).parent / "results"
PANEL_CACHE = Path(__file__).parent / "data" / "panel.parquet"

BASE_SEASON = 2025
RETIREMENT_AGE = 38
N_SIMS = 6000
N_BOOT = 30

# Showcase selection (DESIGN.md §5.6): young attackers, chosen from the data.
# Ranking by minutes alone would return goalkeepers and full-backs, for whom a
# goal projection is meaningless, so the rank is on attacking output among
# forwards and midfielders. This selection shapes *who is shown*; it has no
# bearing on the gate, which runs on the whole unbiased cohort.
SHOWCASE_MAX_AGE = 21
SHOWCASE_MIN_MINUTES = 1200
SHOWCASE_POSITIONS = ("FW", "MF")
SHOWCASE_N = 10
ALWAYS_INCLUDE = [("Lamine Yamal", 2007)]


def load_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        panel = pd.read_parquet(PANEL_CACHE)
        if S.GLS90 in panel.columns:
            return panel
    panel = build_panel()
    panel.to_parquet(PANEL_CACHE, index=False)
    return panel


def select_showcase(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel[S.SEASON] == BASE_SEASON]
    pool = base[
        (base[S.AGE] <= SHOWCASE_MAX_AGE)
        & (base[S.MINUTES] >= SHOWCASE_MIN_MINUTES)
        & (base[S.POS].isin(SHOWCASE_POSITIONS))
    ].copy()
    pool["_output"] = pool[S.NPG] + pool[S.ASSISTS]
    picked = pool.nlargest(SHOWCASE_N, "_output")

    for name, born in ALWAYS_INCLUDE:
        if not ((picked[S.PLAYER] == name) & (picked[S.BORN] == born)).any():
            extra = base[(base[S.PLAYER] == name) & (base[S.BORN] == born)]
            picked = pd.concat([picked, extra], ignore_index=True)
    return picked.drop(columns=["_output"], errors="ignore").reset_index(drop=True)


def load_verdict() -> dict:
    path = RESULTS / "gate.json"
    if not path.exists():
        raise FileNotFoundError("run run_validate.py first — projections need a verdict")
    gate = json.loads(path.read_text())
    passed = {
        m: [g["passed"] for g in gs] for m, gs in gate["gates"].items()
    }
    any_pass = any(any(v) for v in passed.values())
    calib = {
        m: [g["calibration"]["passed"] for g in gs] for m, gs in gate["gates"].items()
    }
    return {
        "variant": gate["variant"],
        "passed_by_metric_horizon": passed,
        "calibration_by_metric_horizon": calib,
        "any_horizon_proven": any_pass,
        "trust_label": (
            "USABLE — at least one horizon passed the pre-registered gate"
            if any_pass
            else "NOT TRUSTWORTHY — no horizon passed the pre-registered gate; "
            "these intervals are shown to demonstrate the failure, not to forecast"
        ),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()
    verdict = load_verdict()
    variant = verdict["variant"]

    print(f"gate verdict: {verdict['trust_label']}\n")

    all_seasons = sorted(panel[S.SEASON].unique().tolist())
    print(f"fitting on all seasons {all_seasons[0]}-{all_seasons[-1]} (variant={variant})")

    npg_curve = fit_aging_curve(panel, S.NPG90, variant=variant)
    ast_curve = fit_aging_curve(panel, S.AST90, variant=variant)
    gls_curve = fit_aging_curve(panel, S.GLS90, variant=variant)

    K_npg, _ = fit_K(panel, S.NPG90, S.NPG, all_seasons, curve=npg_curve)
    K_ast, _ = fit_K(panel, S.AST90, S.ASSISTS, all_seasons, curve=ast_curve)
    K_gls, _ = fit_K(panel, S.GLS90, S.GOALS, all_seasons, curve=gls_curve)
    print(f"  fitted K: npg={K_npg} ast={K_ast} gls={K_gls}")

    npg_model = build_talent_model(panel, S.NPG90, S.NPG, all_seasons, K_npg)
    ast_model = build_talent_model(panel, S.AST90, S.ASSISTS, all_seasons, K_ast)
    gls_model = build_talent_model(panel, S.GLS90, S.GOALS, all_seasons, K_gls)

    npg_curves = bootstrap_curves(panel, S.NPG90, variant, n_boot=N_BOOT, seed=0) or [npg_curve]
    ast_curves = bootstrap_curves(panel, S.AST90, variant, n_boot=N_BOOT, seed=1) or [ast_curve]
    gls_curves = bootstrap_curves(panel, S.GLS90, variant, n_boot=N_BOOT, seed=2) or [gls_curve]
    minutes_model = fit_minutes_model(panel)
    print(f"  bootstrap curves: {len(npg_curves)}/{len(ast_curves)}/{len(gls_curves)}\n")

    showcase = select_showcase(panel)
    print(f"showcase: {len(showcase)} players\n")

    out: dict = {
        "base_season": BASE_SEASON,
        "retirement_age": RETIREMENT_AGE,
        "n_sims": N_SIMS,
        "variant": variant,
        "verdict": verdict,
        "fitted_K": {"npg": K_npg, "assists": K_ast, "total_goals": K_gls},
        "units": {
            "non_penalty_goals": "the model's honest unit; this is what the gate scored",
            "total_goals": "includes penalties, for record comparison only; "
            "NOT separately gated",
        },
        "players": [],
    }

    hist_all = panel[panel[S.SEASON] <= BASE_SEASON]
    for row in showcase.itertuples(index=False):
        player, born = getattr(row, S.PLAYER), int(getattr(row, S.BORN))
        age = int(getattr(row, S.AGE))
        horizon = max(RETIREMENT_AGE - age, 1)
        hist = hist_all[(hist_all[S.PLAYER] == player) & (hist_all[S.BORN] == born)]
        state = PlayerState(
            player=player,
            born=born,
            pos=getattr(row, S.POS),
            age=age,
            minutes=float(getattr(row, S.MINUTES)),
            history=hist,
        )

        npg_sim = simulate_player(
            state, npg_model, ast_model, npg_curves, ast_curves, minutes_model,
            horizon=horizon, n_sims=N_SIMS, rng=np.random.default_rng(abs(hash(player)) % 2**31),
        )
        gls_sim = simulate_player(
            state, gls_model, ast_model, gls_curves, ast_curves, minutes_model,
            horizon=horizon, n_sims=N_SIMS, rng=np.random.default_rng((abs(hash(player)) + 7) % 2**31),
        )

        npg_tot = npg_sim.totals()
        entry: dict = {
            "player": player,
            "born": born,
            "age": age,
            "pos": getattr(row, S.POS),
            "team": getattr(row, S.TEAM),
            "league": getattr(row, S.LEAGUE),
            "base_season_minutes": float(getattr(row, S.MINUTES)),
            "base_season_npg": float(getattr(row, S.NPG)),
            "base_season_assists": float(getattr(row, S.ASSISTS)),
            "seasons_observed": int(len(hist)),
            "future_non_penalty_goals": {
                "median": float(np.median(npg_tot["goals"])),
                "p10": float(np.percentile(npg_tot["goals"], 10)),
                "p90": float(np.percentile(npg_tot["goals"], 90)),
            },
            "future_assists": {
                "median": float(np.median(npg_tot["assists"])),
                "p10": float(np.percentile(npg_tot["assists"], 10)),
                "p90": float(np.percentile(npg_tot["assists"], 90)),
            },
            "next_season_non_penalty_goals": {
                "median": float(np.median(npg_sim.goals[:, 0])),
                "p10": float(np.percentile(npg_sim.goals[:, 0], 10)),
                "p90": float(np.percentile(npg_sim.goals[:, 0], 90)),
            },
        }

        if career_is_fully_observed(panel, player, born):
            totals = career_totals(
                panel, player, born, BASE_SEASON, gls_sim.totals()["goals"]
            )
            entry["goals_so_far"] = goals_so_far(panel, player, born, BASE_SEASON)
            entry["career_total_goals"] = record_report(
                totals, league=getattr(row, S.LEAGUE)
            )
        else:
            entry["career_total_goals"] = {
                "unavailable": "debuted at or before 2017-18; the panel does not hold "
                "his full career, so a career total would understate him"
            }

        out["players"].append(entry)

        g = entry["future_non_penalty_goals"]
        print(
            f"  {player:24s} age {age}  future npG median {g['median']:6.0f} "
            f"[{g['p10']:.0f}-{g['p90']:.0f}]"
        )

    (RESULTS / "projections.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {RESULTS / 'projections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
