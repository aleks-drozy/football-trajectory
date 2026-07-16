"""Build the tidy player-season panel from raw FBref parquet dumps.

Two facts about the raw data drive this module:

1. A player who moves mid-season appears once per team, so rows must be
   aggregated to one row per (player, born, season).
2. FBref's season tables carry no stable player ID, so identity is
   ``(player_name, born_year)``. That key correctly separates the common
   mononym collisions (Marcelo, Rafinha, Rafael...), but it would silently
   merge two players sharing both name and birth year. ``audit_identity``
   measures how often that is plausibly happening rather than assuming it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import schema as S

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

BIG5_LEAGUES = frozenset(
    {
        "ENG-Premier League",
        "ESP-La Liga",
        "FRA-Ligue 1",
        "ITA-Serie A",
        "GER-Bundesliga",
    }
)


def repair_missing_league(raw: pd.DataFrame) -> pd.DataFrame:
    """Fill in the league label soccerdata drops for one of the Big 5.

    soccerdata's "Big 5 European Leagues Combined" view returns a null `league`
    for every Bundesliga row (~18% of the panel) while labelling the other four
    correctly. The rows themselves are fine — Bayern, Dortmund and Leverkusen
    are all present — only the label is missing.

    The repair is inferred, so it verifies its own premise instead of trusting
    it: the fetch requests exactly five leagues, so if precisely one label is
    absent the unlabelled rows can only belong to it. If that ever stops
    holding — two labels missing, or an unexpected league appearing — this
    raises rather than silently mislabelling a fifth of the data.
    """
    if not raw["league"].isna().any():
        return raw

    present = set(raw["league"].dropna().unique())
    unexpected = present - BIG5_LEAGUES
    if unexpected:
        raise ValueError(f"unexpected leagues in a Big-5 fetch: {sorted(unexpected)}")

    missing = sorted(BIG5_LEAGUES - present)
    if len(missing) != 1:
        raise ValueError(
            "cannot infer the missing league label: expected exactly one of the "
            f"Big 5 to be absent, found {len(missing)} ({missing}). "
            "Null league rows can no longer be attributed unambiguously."
        )

    raw = raw.copy()
    raw["league"] = raw["league"].fillna(missing[0])
    return raw


def _season_start_year(season_code: str | int) -> int:
    """FBref/soccerdata season codes look like '1718' -> 2017, '2526' -> 2025."""
    code = str(season_code).zfill(4)
    start = int(code[:2])
    # Big-5 data here runs 2017-18 onward; two-digit years are unambiguous
    # within 2000-2099 for this project's range.
    return 2000 + start


def _primary_position(pos: str | float) -> str:
    """FBref encodes multi-position players as 'MF,FW'. Take the first listed,
    which FBref orders by primary usage."""
    if not isinstance(pos, str) or not pos:
        return "UNK"
    first = pos.split(",")[0].strip().upper()
    return first if first in S.POSITION_GROUPS else "UNK"


def load_raw_season(season: str) -> pd.DataFrame:
    """Merge the standard + shooting tables for one season into raw rows."""
    std_path = RAW_DIR / f"{season}__standard.parquet"
    if not std_path.exists():
        raise FileNotFoundError(f"missing {std_path}")
    std = pd.read_parquet(std_path)

    keys = ["league", "season", "team", "player", "born"]
    out = std.copy()

    shoot_path = RAW_DIR / f"{season}__shooting.parquet"
    if shoot_path.exists():
        shoot = pd.read_parquet(shoot_path)
        shoot_cols = [c for c in ("Standard__Sh", "Standard__SoT") if c in shoot.columns]
        if shoot_cols:
            out = out.merge(
                shoot[keys + shoot_cols].drop_duplicates(subset=keys),
                on=keys,
                how="left",
            )
    return out


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def build_panel(seasons: list[str] | None = None) -> pd.DataFrame:
    """Return one row per (player, born, season) with minutes-weighted context."""
    if seasons is None:
        seasons = sorted({p.name.split("__")[0] for p in RAW_DIR.glob("*__standard.parquet")})
    if not seasons:
        raise FileNotFoundError(f"no raw standard parquets in {RAW_DIR}")

    frames = [load_raw_season(s) for s in seasons]
    raw = pd.concat(frames, ignore_index=True)
    raw = repair_missing_league(raw)

    raw = raw[raw["born"].notna()].copy()
    raw["_season_start"] = raw["season"].map(_season_start_year)
    raw["_pos"] = raw["pos"].map(_primary_position)
    raw["_minutes"] = _to_num(raw, "Playing Time__Min")
    raw["_matches"] = _to_num(raw, "Playing Time__MP")
    raw["_starts"] = _to_num(raw, "Playing Time__Starts")
    raw["_goals"] = _to_num(raw, "Performance__Gls")
    raw["_pk"] = _to_num(raw, "Performance__PK")
    raw["_assists"] = _to_num(raw, "Performance__Ast")
    raw["_shots"] = _to_num(raw, "Standard__Sh")
    raw["_sot"] = _to_num(raw, "Standard__SoT")

    grouped = raw.groupby(["player", "born", "_season_start"], as_index=False)

    agg = grouped.agg(
        minutes=("_minutes", "sum"),
        matches=("_matches", "sum"),
        starts=("_starts", "sum"),
        goals=("_goals", "sum"),
        pk=("_pk", "sum"),
        assists=("_assists", "sum"),
        shots=("_shots", "sum"),
        sot=("_sot", "sum"),
    )

    # Context (team/league/pos) is taken from wherever the player played most.
    def _dominant(frame: pd.DataFrame) -> pd.Series:
        top = frame.loc[frame["_minutes"].idxmax()]
        return pd.Series(
            {"league": top["league"], "team": top["team"], "pos": top["_pos"]}
        )

    context = (
        raw.groupby(["player", "born", "_season_start"])
        .apply(_dominant, include_groups=False)
        .reset_index()
    )

    panel = agg.merge(context, on=["player", "born", "_season_start"])
    panel = panel.rename(columns={"_season_start": S.SEASON})
    panel[S.BORN] = panel[S.BORN].astype(int)
    panel[S.AGE] = panel[S.SEASON] - panel[S.BORN]
    panel[S.N90] = panel[S.MINUTES] / 90.0
    panel[S.NPG] = panel[S.GOALS] - panel[S.PK]

    for rate, events in (
        (S.NPG90, S.NPG),
        (S.AST90, S.ASSISTS),
        (S.SH90, S.SHOTS),
        (S.GLS90, S.GOALS),
    ):
        panel[rate] = (panel[events] / panel[S.N90]).where(panel[S.N90] > 0, 0.0)

    panel = panel[S.PANEL_COLUMNS].sort_values([S.PLAYER, S.BORN, S.SEASON])
    return panel.reset_index(drop=True)


@dataclass(frozen=True)
class IdentityAudit:
    """Measured evidence about the (name, born) identity key."""

    n_rows: int
    n_players: int
    names_with_multiple_born: int
    suspect_merges: int
    suspect_examples: list[tuple[str, int, int, float]]

    def summary(self) -> str:
        pct = 100.0 * self.suspect_merges / max(self.n_rows, 1)
        return (
            f"panel rows={self.n_rows}, distinct (name,born) players={self.n_players}; "
            f"names reused across birth years={self.names_with_multiple_born}; "
            f"implausible-minute player-seasons={self.suspect_merges} ({pct:.3f}%)"
        )


def audit_identity(panel: pd.DataFrame) -> IdentityAudit:
    """Quantify the identity risk the (name, born) key carries.

    A merge of two distinct same-name same-birth-year players shows up as a
    single player-season with more minutes than one human could physically
    play. That is the observable signature, so it is what gets counted.
    """
    names_multi = int((panel.groupby(S.PLAYER)[S.BORN].nunique() > 1).sum())
    suspects = panel[panel[S.MINUTES] > S.MAX_PLAUSIBLE_SEASON_MINUTES]
    examples = [
        (r[S.PLAYER], int(r[S.BORN]), int(r[S.SEASON]), float(r[S.MINUTES]))
        for _, r in suspects.head(10).iterrows()
    ]
    return IdentityAudit(
        n_rows=len(panel),
        n_players=int(panel.groupby([S.PLAYER, S.BORN]).ngroups),
        names_with_multiple_born=names_multi,
        suspect_merges=len(suspects),
        suspect_examples=examples,
    )
