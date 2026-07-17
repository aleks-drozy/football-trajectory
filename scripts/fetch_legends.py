"""Fetch full Big-5 league career arcs for the scenario legends.

Legends must have (a) a teenage start, so their arc covers every showcase
youngster's anchor age, and (b) a complete Big-5 career, so the scenario runs to
a natural end. Messi is already fetched (data/messi_seasons.csv, verified).
This adds:

- Cristiano Ronaldo — Premier League 2003-2008, La Liga 2009-2017, Serie A
  2018-2020 (ages 18-35; born Feb 1985)
- Karim Benzema — Ligue 1 2004-2008, La Liga 2009-2022 (ages 17-35; born Dec 1987)

The La Liga 2009-2020 season tables are already in soccerdata's cache from the
Messi fetch, so most of this run is cache hits; only the PL, Serie A, Ligue 1
and 2021-22 La Liga seasons hit the network.

Usage:
    python scripts/fetch_legends.py    # -> data/legend_seasons.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LEGENDS = {
    "Cristiano Ronaldo": {
        "born": 1985,
        "runs": [
            ("ENG-Premier League", range(2003, 2009)),
            ("ESP-La Liga", range(2009, 2018)),
            ("ITA-Serie A", range(2018, 2021)),
        ],
    },
    "Karim Benzema": {
        "born": 1987,
        "runs": [
            ("FRA-Ligue 1", range(2004, 2009)),
            ("ESP-La Liga", range(2009, 2023)),
        ],
    },
}


def main() -> int:
    import soccerdata as sd

    rows = []
    for name, spec in LEGENDS.items():
        for league, years in spec["runs"]:
            for y in years:
                season = f"{y}-{y+1}"
                try:
                    fb = sd.FBref(leagues=league, seasons=season)
                    df = fb.read_player_season_stats(stat_type="standard").reset_index()
                    m = df[df["player"].str.contains(name, na=False)]
                    if m.empty:
                        print(f"[warn] {name} {season} {league}: not found", flush=True)
                        continue
                    r = m.iloc[0]
                    rows.append(
                        {
                            "player": name,
                            "born": spec["born"],
                            "league": league,
                            "season_start": y,
                            "age": y - spec["born"],
                            "MP": int(r[("Playing Time", "MP")]),
                            "Min": int(r[("Playing Time", "Min")]),
                            "Gls": int(r[("Performance", "Gls")]),
                            "Ast": int(r[("Performance", "Ast")]),
                        }
                    )
                    print(f"[ok] {name} {season} {league}: {rows[-1]['Gls']} gls / {rows[-1]['Min']} min", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[FAIL] {name} {season}: {type(e).__name__}: {e}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "legend_seasons.csv", index=False)
    print(f"\nsaved {len(out)} legend seasons")
    for name, grp in out.groupby("player"):
        print(f"  {name}: ages {grp.age.min()}-{grp.age.max()}, {grp.Gls.sum()} league goals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
