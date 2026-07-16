"""Export per-season trajectory bands for the fan-chart artifact.

run_project.py stores only career totals; a fan chart needs the whole path. This
re-runs the same v2 simulation and, for each showcase player, writes:

- observed history (age -> actual npg / assists / minutes), the solid past
- per-future-season percentile bands (p10/p25/p50/p75/p90) for npg, assists,
  minutes, and *cumulative* career npg — the widening fan
- the same career-total and record summary run_project.py already reports

Everything is stamped with the gate verdict so the front-end can carry the
NOT TRUSTWORTHY label the pre-registration requires. Output is
results/trajectories.json.

Usage:
    python scripts/export_trajectories.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import schema as S
from mc.records import career_is_fully_observed, career_totals, goals_so_far, record_report
from mc.simulate import PlayerState, bootstrap_curves, simulate_player
from model.aging import fit_aging_curve
from model.minutes import fit_minutes_model
from model.talent import build_talent_model, fit_K

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASE_SEASON = 2025
RETIREMENT_AGE = 38
N_SIMS = 6000
N_BOOT = 30
PCTLS = [10, 25, 50, 75, 90]

# The same showcase run_project.py selects, re-listed so this script stands
# alone. Chosen from the data (young attackers), plus Yamal by name.
SHOWCASE_MAX_AGE = 21
SHOWCASE_MIN_MINUTES = 1200
SHOWCASE_POSITIONS = ("FW", "MF")
SHOWCASE_N = 10
ALWAYS_INCLUDE = [("Lamine Yamal", 2007)]


def player_seed(player: str, salt: int = 0) -> int:
    import zlib

    return (zlib.crc32(player.encode("utf-8")) + salt) % (2**31)


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
            picked = pd.concat(
                [picked, base[(base[S.PLAYER] == name) & (base[S.BORN] == born)]],
                ignore_index=True,
            )
    return picked.reset_index(drop=True)


def bands(arr: np.ndarray) -> dict[str, list[float]]:
    """Percentile bands over paths (axis 0) for each season (axis 1)."""
    p = {f"p{q}": np.percentile(arr, q, axis=0) for q in PCTLS}
    return {k: [float(x) for x in v] for k, v in p.items()}


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    gate = json.loads((RESULTS / "gate.json").read_text(encoding="utf-8"))
    fresh = json.loads((RESULTS / "gate_fresh.json").read_text(encoding="utf-8"))
    variant = gate["variant"]

    all_seasons = sorted(panel[S.SEASON].unique().tolist())
    print(f"fitting v2 on {all_seasons[0]}-{all_seasons[-1]} (variant={variant})")

    npg_curve = fit_aging_curve(panel, S.NPG90, variant=variant)
    ast_curve = fit_aging_curve(panel, S.AST90, variant=variant)
    gls_curve = fit_aging_curve(panel, S.GLS90, variant=variant)
    K_npg, _ = fit_K(panel, S.NPG90, S.NPG, all_seasons, curve=npg_curve)
    K_ast, _ = fit_K(panel, S.AST90, S.ASSISTS, all_seasons, curve=ast_curve)
    K_gls, _ = fit_K(panel, S.GLS90, S.GOALS, all_seasons, curve=gls_curve)

    npg_model = build_talent_model(panel, S.NPG90, S.NPG, all_seasons, K_npg)
    ast_model = build_talent_model(panel, S.AST90, S.ASSISTS, all_seasons, K_ast)
    gls_model = build_talent_model(panel, S.GLS90, S.GOALS, all_seasons, K_gls)
    npg_curves = bootstrap_curves(panel, S.NPG90, variant, n_boot=N_BOOT, seed=0) or [npg_curve]
    ast_curves = bootstrap_curves(panel, S.AST90, variant, n_boot=N_BOOT, seed=1) or [ast_curve]
    gls_curves = bootstrap_curves(panel, S.GLS90, variant, n_boot=N_BOOT, seed=2) or [gls_curve]
    minutes_model = fit_minutes_model(panel)

    showcase = select_showcase(panel)
    print(f"showcase: {len(showcase)} players\n")

    hist_all = panel[panel[S.SEASON] <= BASE_SEASON]
    players_out = []

    for row in showcase.itertuples(index=False):
        player, born = getattr(row, S.PLAYER), int(getattr(row, S.BORN))
        age = int(getattr(row, S.AGE))
        horizon = max(RETIREMENT_AGE - age, 1)
        hist = hist_all[(hist_all[S.PLAYER] == player) & (hist_all[S.BORN] == born)].sort_values(S.SEASON)

        state = PlayerState(
            player=player, born=born, pos=getattr(row, S.POS), age=age,
            minutes=float(getattr(row, S.MINUTES)), history=hist,
        )
        npg_sim = simulate_player(
            state, npg_model, ast_model, npg_curves, ast_curves, minutes_model,
            horizon=horizon, n_sims=N_SIMS, rng=np.random.default_rng(player_seed(player)),
        )
        gls_sim = simulate_player(
            state, gls_model, ast_model, gls_curves, ast_curves, minutes_model,
            horizon=horizon, n_sims=N_SIMS, rng=np.random.default_rng(player_seed(player, 7)),
        )

        future_ages = [age + h + 1 for h in range(horizon)]
        observed = [
            {
                "age": int(r[S.AGE]),
                "season": int(r[S.SEASON]),
                "npg": float(r[S.NPG]),
                "goals": float(r[S.GOALS]),  # total incl. penalties, for the career view
                "assists": float(r[S.ASSISTS]),
                "minutes": float(r[S.MINUTES]),
            }
            for _, r in hist.iterrows()
        ]

        # Cumulative career goals path: goals already scored + running future.
        so_far = goals_so_far(panel, player, born, BASE_SEASON) if career_is_fully_observed(panel, player, born) else None
        cum_future = gls_sim.goals.cumsum(axis=1)
        cum_path = bands(cum_future + (so_far or 0)) if so_far is not None else None

        entry = {
            "player": player,
            "born": born,
            "age": age,
            "pos": getattr(row, S.POS),
            "team": getattr(row, S.TEAM),
            "league": getattr(row, S.LEAGUE),
            "seasons_observed": len(observed),
            "observed": observed,
            "future_ages": future_ages,
            "bands": {
                "npg": bands(npg_sim.goals),
                "assists": bands(npg_sim.assists),
                "minutes": bands(npg_sim.minutes),
                "cumulative_goals": cum_path,
            },
            "totals": {
                "future_npg": {
                    "median": float(np.median(npg_sim.totals()["goals"])),
                    "p10": float(np.percentile(npg_sim.totals()["goals"], 10)),
                    "p90": float(np.percentile(npg_sim.totals()["goals"], 90)),
                },
                "future_assists": {
                    "median": float(np.median(npg_sim.totals()["assists"])),
                    "p10": float(np.percentile(npg_sim.totals()["assists"], 10)),
                    "p90": float(np.percentile(npg_sim.totals()["assists"], 90)),
                },
            },
            "goals_so_far": so_far,
        }
        if so_far is not None:
            totals = career_totals(panel, player, born, BASE_SEASON, gls_sim.totals()["goals"])
            entry["career_total_goals"] = record_report(totals, league=getattr(row, S.LEAGUE))

        players_out.append(entry)
        g = entry["totals"]["future_npg"]
        print(f"  {player:22s} age {age}  npG {g['median']:5.0f} [{g['p10']:.0f}-{g['p90']:.0f}]  horizon {horizon}")

    out = {
        "base_season": BASE_SEASON,
        "retirement_age": RETIREMENT_AGE,
        "n_sims": N_SIMS,
        "model_version": "v2 (availability conditioned on talent)",
        "percentiles": PCTLS,
        "verdict": {
            "overall": "NOT PROVEN at every horizon",
            "trust_label": "NOT TRUSTWORTHY",
            "headline": (
                "Skill is real and replicates on unseen data, but the intervals are "
                "too wide — the model is under-confident. Point projections are useful; "
                "the bands are shown to demonstrate that failure, not to forecast."
            ),
            "gate_2022_2024": {
                "status": gate.get("test_status", ""),
                "goals_plus3_mae": gate["gates"]["goals"][2]["skill"]["model_mae"],
                "goals_plus3_baseline": gate["gates"]["goals"][2]["skill"]["best_baseline_mae"],
                "goals_plus1_calibration": gate["gates"]["goals"][0]["calibration"]["coverage"],
            },
            "gate_2025_fresh": {
                "status": fresh.get("test_status", ""),
                "goals_plus1_mae": fresh["gates"]["goals"][0]["skill"]["model_mae"],
                "goals_plus1_baseline": fresh["gates"]["goals"][0]["skill"]["best_baseline_mae"],
                "goals_plus1_calibration": fresh["gates"]["goals"][0]["calibration"]["coverage"],
            },
        },
        "players": players_out,
    }
    (RESULTS / "trajectories.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {RESULTS / 'trajectories.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
