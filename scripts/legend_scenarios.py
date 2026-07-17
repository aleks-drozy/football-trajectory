"""Legend scenarios: every showcase youngster's career, if it followed the arc
of an all-time great.

Generalises the Messi scenario (scripts/messi_scenario.py) to any (youngster,
legend) pair. These are SCENARIOS, not forecasts — no probability attached,
because conditioning on a legend's arc is conditioning on one of the best
careers ever played. The trajectory model elsewhere in this repo says what
usually happens; this says what following a specific legendary path would look
like.

Two modes, because "like Messi" is ambiguous and the two readings behave very
differently:

**at_rate — "if he performed at the legend's rate"** (the primary mode).
From the youngster's current age onward he simply inherits the legend's actual
per-90 rate and minutes. Career total = goals already scored + the legend's
remaining league goals. Always well defined.

**improving_like — "if he improved by the same factor the legend did"**
(secondary). Scales the youngster's *current* rate by the legend's growth
multiplier M(a) = legend_rate(a) / legend_rate(A).

The second mode has a domain of validity and it is enforced, not assumed. The
multiplier's denominator is the legend's rate at the youngster's current age,
so a legend who was *not yet* a scorer at that age produces an explosive factor:
Cristiano Ronaldo at 18 was a 0.232/90 winger and peaked at 1.394 — a **6x**
multiplier. Applying 6x to a youngster who is already elite (Yamal is at
0.637/90 at 18) yields ~3.8 goals/90, which is not a scenario but an artefact.
Messi is the exception that hid this: he was already scoring at 0.591/90 at 18,
so his multiplier is a sane 2.3x.

So `improving_like` is emitted only when it survives **two** guards: the anchor
ratio youngster_rate(A) / legend_rate(A) must sit inside SCALE_BAND, *and* the
implied peak rate must stay under a plausibility ceiling computed from the data.
The ratio guard alone is not enough — a legend with a low baseline carries a
huge internal multiplier (Ronaldo's peak is 7.5x his age-19 rate), so even a
comparable youngster gets blown past anything ever recorded. Failing either
guard returns null with the reason attached, and the UI does not offer the mode.

Legends (league seasons only, all series verified against FBref):
- Lionel Messi     — La Liga 2004-2020, ages 17-33 (data/messi_seasons.csv +
                     the table in messi_scenario.py, sums to the 474 record)
- Cristiano Ronaldo — PL/La Liga/Serie A 2003-2020, ages 18-35
- Karim Benzema     — Ligue 1/La Liga 2004-2022, ages 17-35
                     (both from data/legend_seasons.csv, scripts/fetch_legends.py)

A legend covers a youngster only if the legend has a season at the youngster's
current age; all three arcs start at 17-18, so every showcase player (18-21) is
covered by Messi and Benzema, and everyone 18+ by Ronaldo.

Usage:
    python scripts/legend_scenarios.py   # -> results/legend_scenarios.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import schema as S
from mc.records import RECORDS
from scripts.messi_scenario import MESSI as MESSI_TABLE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASE_SEASON = 2025
RECORD_BY_LEAGUE = {r.competition: r for r in RECORDS}

# Two guards on the "improving like X" mode, because one is not enough.
#
# 1. SCALE_BAND — the youngster and the legend must have been at a comparable
#    level at the same age, or the multiplier measures how *low* the legend's
#    baseline was rather than how much he improved.
# 2. A plausibility ceiling on the implied peak rate. The band alone is not
#    sufficient: a legend whose anchor-age rate was low has a huge internal
#    multiplier (Ronaldo's peak is 7.5x his age-19 rate), so even a youngster
#    who passes the band gets blown past anything ever observed.
#
#    The reference set matters. Taking the ceiling from the panel alone (2017+)
#    gives Lewandowski's 1.50/90 — but that window excludes the legends' own
#    primes, and Messi's 2012-13 (46 goals in 2,646 min) is 1.565/90, *above*
#    it. A ceiling that outlaws Messi's real season is the wrong ceiling. So it
#    is computed at runtime over the panel AND the legend arcs, with a 25%
#    tolerance: a scenario may exceed the best season ever played by a little
#    (that is what "he is better than the best" means) but not by half again.
SCALE_BAND = (0.5, 2.0)
CEILING_TOLERANCE = 1.25
MIN_MINUTES_FOR_PEAK = 1800  # a peak rate needs a real season behind it


def load_legends(panel: pd.DataFrame) -> dict[str, dict]:
    """Return {legend: {age: (minutes, goals)}} with provenance.

    Seasons from 2017-18 onward live in the project's own panel (the original
    Big-5 fetch), so any season the single-league scrape missed — e.g.
    Ronaldo's Juventus years, where FBref's single-league Serie A endpoint
    errors — is backfilled from the panel rather than re-scraped.
    """
    legends: dict[str, dict] = {
        "Lionel Messi": {
            "arc": {age: (mins, gls) for _, age, mins, gls in MESSI_TABLE},
            "source": "statmuse + FBref cross-check (sums to the 474 record)",
            "born": 1987,
        }
    }
    csv = pd.read_csv(ROOT / "data" / "legend_seasons.csv")
    borns = {
        "Cristiano Ronaldo": 1985,
        "Karim Benzema": 1987,
        "Kylian Mbappé": 1998,
        "Erling Haaland": 2000,
        "Wayne Rooney": 1985,
        "Sergio Agüero": 1988,
    }
    for name, grp in csv.groupby("player"):
        # A legend's age-17 cameo (a handful of minutes) would make a wild
        # multiplier denominator; require a real season to anchor on.
        arc = {
            int(r.age): (int(r.Min), int(r.Gls))
            for _, r in grp.iterrows()
            if r.Min >= 400
        }
        filled = []
        rows = panel[(panel[S.PLAYER] == name) & (panel[S.BORN] == borns[name])]
        for _, r in rows.iterrows():
            age = int(r[S.AGE])
            if age not in arc and r[S.MINUTES] >= 400:
                arc[age] = (int(r[S.MINUTES]), int(r[S.GOALS]))
                filled.append(age)
        if filled:
            print(f"  [panel backfill] {name}: ages {sorted(filled)}")
        legends[name] = {
            "arc": arc,
            "source": "FBref via scripts/fetch_legends.py"
            + (" + panel backfill for 2017+ seasons" if filled else ""),
            "born": borns[name],
        }

    # Presentation order: the two all-timers, then the two active heirs, then
    # the rest. Insertion order survives the JSON round-trip into the UI.
    preferred = [
        "Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland",
        "Karim Benzema", "Sergio Agüero", "Wayne Rooney",
    ]
    return {n: legends[n] for n in preferred if n in legends} | {
        n: d for n, d in legends.items() if n not in preferred
    }


def _finish(youngster: dict, path: list[dict], extra: dict) -> dict:
    rec = RECORD_BY_LEAGUE.get(youngster["league"])
    cum = path[-1]["cumulative"]
    out = {
        "career_total": round(cum, 0),
        "ends_at_age": path[-1]["age"],
        "peak_season": max(path, key=lambda p: p["goals"]),
        "record": (
            {
                "name": rec.name,
                "holder": rec.holder,
                "goals": rec.goals,
                "beats": bool(cum >= rec.goals),
                "broken_at_age": next(
                    (p["age"] for p in path if p["cumulative"] >= rec.goals), None
                ),
            }
            if rec
            else None
        ),
        "path": path,
    }
    out.update(extra)
    return out


def peak_rate_ceiling(panel: pd.DataFrame, legends: dict[str, dict]) -> float:
    """Highest goals/90 ever seen here, plus tolerance — computed, not guessed.

    Spans the panel *and* the legend arcs, because the legends' primes predate
    the panel window and a ceiling that outlaws Messi's real 2012-13 season
    would be indefensible.
    """
    best = float(panel[panel[S.N90] >= MIN_MINUTES_FOR_PEAK / 90.0][S.GLS90].max())
    for d in legends.values():
        for mins, gls in d["arc"].values():
            if mins >= MIN_MINUTES_FOR_PEAK:
                best = max(best, gls * 90.0 / mins)
    return best * CEILING_TOLERANCE


def build_scenarios(
    youngster: dict, arc: dict[int, tuple[int, int]], ceiling: float
) -> dict:
    """Both modes for one (youngster, legend) pair."""
    A = youngster["age"]
    out: dict = {"at_rate": None, "improving_like": None}
    if A not in arc:
        out["unavailable"] = f"the legend has no season at age {A} to anchor on"
        return out

    a_min, a_gls = arc[A]
    legend_rate_A = a_gls * 90.0 / a_min
    later = [a for a in sorted(arc) if a > A]
    if not later:
        out["unavailable"] = f"the legend's Big-5 career ends at {A}"
        return out

    # --- mode 1: perform at the legend's actual rate -------------------------
    cum = float(youngster["goals_so_far"])
    path = []
    for age in later:
        mins, gls = arc[age]
        cum += gls
        path.append(
            {
                "age": age,
                "minutes": mins,
                "rate_per90": round(gls * 90.0 / mins, 3),
                "goals": float(gls),
                "cumulative": round(cum, 1),
            }
        )
    out["at_rate"] = _finish(youngster, path, {})

    # --- mode 2: improve by the legend's growth factor -----------------------
    if legend_rate_A <= 0:
        out["improving_like"] = None
        out["improving_like_reason"] = (
            f"the legend did not score at age {A}, so the growth factor is undefined"
        )
        return out

    scale = youngster["rate"] / legend_rate_A
    if not (SCALE_BAND[0] <= scale <= SCALE_BAND[1]):
        out["improving_like"] = None
        out["improving_like_reason"] = (
            f"not comparable at age {A}: the youngster is {scale:.1f}x the legend's "
            f"{legend_rate_A:.2f}/90, outside the {SCALE_BAND[0]}-{SCALE_BAND[1]}x band "
            "where a growth multiplier still means something"
        )
        return out

    cum = float(youngster["goals_so_far"])
    path = []
    for age in later:
        mins, gls = arc[age]
        rate = youngster["rate"] * ((gls * 90.0 / mins) / legend_rate_A)
        goals = rate * mins / 90.0
        cum += goals
        path.append(
            {
                "age": age,
                "minutes": mins,
                "rate_per90": round(rate, 3),
                "goals": round(goals, 1),
                "cumulative": round(cum, 1),
            }
        )

    peak_rate = max(p["rate_per90"] for p in path)
    if peak_rate > ceiling:
        out["improving_like"] = None
        out["improving_like_reason"] = (
            f"implies a peak of {peak_rate:.2f} goals/90, past the plausibility ceiling "
            f"of {ceiling:.2f} (the best season ever played here, plus tolerance); the "
            f"legend's own growth factor from age {A} is too steep to transplant"
        )
        return out

    out["improving_like"] = _finish(youngster, path, {"anchor_scale": round(scale, 2)})
    return out


def main() -> int:
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    traj = json.loads((RESULTS / "trajectories.json").read_text(encoding="utf-8"))
    legends = load_legends(panel)
    ceiling = peak_rate_ceiling(panel, legends)
    print(f"  plausibility ceiling: {ceiling:.2f} goals/90")

    out: dict = {
        "note": "Scenarios, not forecasts. at_rate = he performs at the legend's actual per-90 rate; improving_like = his current rate scaled by the legend's growth factor (suppressed where that factor cannot be transplanted).",
        "peak_rate_ceiling": round(ceiling, 3),
        "legends": {
            name: {
                "ages": [min(d["arc"]), max(d["arc"])],
                "league_goals": sum(g for _, g in d["arc"].values()),
                "source": d["source"],
                # An arc whose last season is the current one belongs to a
                # player still writing his career; the UI must not present it
                # as a finished life's work.
                "arc_complete": d["born"] + max(d["arc"]) < BASE_SEASON,
            }
            for name, d in legends.items()
        },
        "records": {
            r.competition: {"name": r.name, "holder": r.holder, "goals": r.goals, "source": r.source}
            for r in RECORDS
        },
        "players": {},
    }

    for pl in traj["players"]:
        row = panel[
            (panel[S.PLAYER] == pl["player"])
            & (panel[S.BORN] == pl["born"])
            & (panel[S.SEASON] == BASE_SEASON)
        ].iloc[0]
        youngster = {
            "age": pl["age"],
            "league": pl["league"],
            "rate": float(row[S.GOALS]) * 90.0 / float(row[S.MINUTES]),
            "goals_so_far": pl["goals_so_far"] if pl["goals_so_far"] is not None else 0,
        }
        entry = {"anchor_rate_per90": round(youngster["rate"], 3), "scenarios": {}}
        for name, d in legends.items():
            entry["scenarios"][name] = build_scenarios(youngster, d["arc"], ceiling)
        out["players"][pl["player"]] = entry
        bits = []
        for n, s in entry["scenarios"].items():
            short = n.split()[-1]
            if s["at_rate"]:
                bits.append(f"{short}:{s['at_rate']['career_total']:.0f}")
                if s["improving_like"]:
                    bits[-1] += f"/{s['improving_like']['career_total']:.0f}*"
        print(f"  {pl['player']:24s} age {pl['age']}  " + "  ".join(bits))

    (RESULTS / "legend_scenarios.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\n-> {RESULTS / 'legend_scenarios.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
