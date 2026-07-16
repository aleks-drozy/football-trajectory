"""Pull Big-5 player-season stats from FBref (via soccerdata) into parquet.

Each season is fetched and written independently so a failure on one season
doesn't lose the others. soccerdata keeps its own scrape cache, so re-runs of
an already-fetched season are cheap.

Usage:
    python scripts/fetch_data.py                 # all seasons in SEASONS
    python scripts/fetch_data.py 2023-2024       # one season
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# xG/xAG exist on FBref from 2017-2018 onward for the Big 5; earlier seasons
# would silently lose the finishing-stable signal the talent model relies on.
SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

STAT_TYPES = ["standard", "shooting", "passing", "playing_time"]


def fetch_season(season: str) -> None:
    import soccerdata as sd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fb = sd.FBref(leagues="Big 5 European Leagues Combined", seasons=season)

    for stat_type in STAT_TYPES:
        out = RAW_DIR / f"{season}__{stat_type}.parquet"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        for attempt in (1, 2, 3):
            try:
                df = fb.read_player_season_stats(stat_type=stat_type)
                # Flatten the MultiIndex columns so parquet round-trips cleanly.
                df.columns = [
                    "__".join(c).strip("_") if isinstance(c, tuple) else str(c)
                    for c in df.columns
                ]
                df.reset_index().to_parquet(out, index=False)
                print(f"[ok]   {out.name}  shape={df.shape}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(
                    f"[warn] {season}/{stat_type} attempt {attempt} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == 3:
                    print(f"[FAIL] {season}/{stat_type} giving up", flush=True)
                    traceback.print_exc()
                else:
                    time.sleep(5 * attempt)


def main() -> int:
    seasons = sys.argv[1:] or SEASONS
    for season in seasons:
        print(f"=== {season} ===", flush=True)
        try:
            fetch_season(season)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] season {season}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print("=== fetch complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
