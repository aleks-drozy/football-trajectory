"""The Messi scenario: what if Yamal's career now follows Messi's arc?

This is a SCENARIO, not a forecast. The model's job elsewhere in this repo is to
say what usually happens; this script answers a different, explicitly
conditional question — "if Yamal improved and lasted the way Messi actually
did, where would he land?" No probability is attached, because conditioning on
a career like Messi's is conditioning on roughly the best case in the sport's
history.

Construction (all of it visible below):

- Messi's real La Liga seasons, age 17-33 (statmuse.com/fc/ask/messi-la-liga-seasons,
  cross-checked: goals sum to exactly the known 474; minutes imply ~1.01 g/90
  career, consistent with the record). Goals INCLUDE penalties, so the scenario
  is in total-goals units, same as every published record.
- Growth multiplier at each age: M(a) = messi_rate(a) / messi_rate(18) — how
  much Messi's per-90 scoring grew (or fell) relative to his own age-18 level.
- Yamal's anchor: his real age-18 season (16 goals in 2,262 min = 0.637/90 —
  for the record, slightly ahead of Messi's 0.591 at the same age).
- Scenario season at age a: rate = 0.637 x M(a), minutes = Messi's actual
  minutes at age a (the durability is part of "like Messi"), goals = rate x
  minutes / 90.
- Career total = the 30 league goals he has already scored + the scenario
  seasons from 19 to 33 (the age Messi's La Liga run ended).

A useful identity falls out: because the multiplier is built from Messi's own
rates and the minutes are Messi's own, the scenario total is exactly

    30 + (yamal_rate18 / messi_rate18) x (Messi's goals from age 19 on)

i.e. "Messi's remaining career, scaled by how much better Yamal's age-18 was."

Usage:
    python scripts/messi_scenario.py     # -> results/messi_scenario.json
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Messi, La Liga, league only, goals incl. penalties.
# Source: statmuse.com/fc/ask/messi-la-liga-seasons (fetched 2026-07-16).
# Verified: goals sum to 474 == the documented La Liga record total.
# Age convention matches the panel: season start year minus birth year (1987).
MESSI = [
    # (season_start, age, minutes, goals)
    (2004, 17,   77,  1),
    (2005, 18,  914,  6),
    (2006, 19, 1996, 14),
    (2007, 20, 2006, 10),
    (2008, 21, 2516, 23),
    (2009, 22, 2840, 34),
    (2010, 23, 2857, 31),
    (2011, 24, 3269, 50),
    (2012, 25, 2646, 46),
    (2013, 26, 2503, 28),
    (2014, 27, 3375, 43),
    (2015, 28, 2728, 26),
    (2016, 29, 2832, 37),
    (2017, 30, 2996, 34),
    (2018, 31, 2710, 36),
    (2019, 32, 2880, 25),
    (2020, 33, 3023, 30),
]

RECORD = 474  # Messi, La Liga all-time


def main() -> int:
    total_goals = sum(g for *_, g in MESSI)
    assert total_goals == RECORD, f"source table corrupt: {total_goals} != {RECORD}"

    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    y18 = panel[(panel[S.PLAYER] == "Lamine Yamal") & (panel[S.SEASON] == 2025)].iloc[0]
    yamal_rate18 = float(y18[S.GOALS]) * 90.0 / float(y18[S.MINUTES])
    goals_so_far = int(
        panel[(panel[S.PLAYER] == "Lamine Yamal") & (panel[S.SEASON] <= 2025)][S.GOALS].sum()
    )

    messi_rate = {a: g * 90.0 / m for _, a, m, g in MESSI}
    scale = yamal_rate18 / messi_rate[18]

    path = []
    cum = float(goals_so_far)
    for _, age, minutes, _ in MESSI:
        if age <= 18:
            continue
        rate = yamal_rate18 * (messi_rate[age] / messi_rate[18])
        goals = rate * minutes / 90.0
        cum += goals
        path.append(
            {
                "age": age,
                "season": 2025 + (age - 18),
                "minutes": minutes,
                "rate_per90": round(rate, 3),
                "goals": round(goals, 1),
                "cumulative": round(cum, 1),
                "messi_multiplier": round(messi_rate[age] / messi_rate[18], 2),
            }
        )

    out = {
        "label": "If he ages like Messi (scenario, not a forecast)",
        "anchor": {
            "yamal_age18_goals": float(y18[S.GOALS]),
            "yamal_age18_minutes": float(y18[S.MINUTES]),
            "yamal_rate18": round(yamal_rate18, 3),
            "messi_rate18": round(messi_rate[18], 3),
            "scale": round(scale, 3),
            "goals_already_scored": goals_so_far,
        },
        "source": "statmuse.com/fc/ask/messi-la-liga-seasons; goals cross-checked to the 474 record",
        "career_total": round(cum, 0),
        "record": RECORD,
        "beats_record_by": round(cum - RECORD, 0),
        "record_broken_at_age": next(
            (p["age"] for p in path if p["cumulative"] >= RECORD), None
        ),
        "peak_season": max(path, key=lambda p: p["goals"]),
        "path": path,
        "model_probability_note": (
            "The trajectory model assigns a path like this less than 1-in-6000 "
            "probability; that is what conditioning on the best career in "
            "history means."
        ),
    }
    (RESULTS / "messi_scenario.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print(f"anchor: Yamal 18 = {yamal_rate18:.3f}/90 vs Messi 18 = {messi_rate[18]:.3f}/90 (scale {scale:.2f}x)")
    print(f"career total under the Messi arc: {cum:.0f} league goals (record {RECORD})")
    print(f"record broken at age: {out['record_broken_at_age']}")
    pk = out["peak_season"]
    print(f"peak season: age {pk['age']}, {pk['goals']:.0f} goals ({pk['rate_per90']:.2f}/90)")
    print(f"-> {RESULTS / 'messi_scenario.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
