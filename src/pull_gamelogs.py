"""Pull team game logs (one row per team per game) for Q1.

Each season is cached to data/raw/gamelog_<season>_<type>.parquet; cached
seasons are skipped on rerun. The combined table is written to
data/processed/team_games.parquet.
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SLEEP_SECONDS = 1.5
MAX_RETRIES = 3


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    tag = season_type.lower().replace(" ", "_")
    path = RAW / f"gamelog_{season}_{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                timeout=60,
            ).get_data_frames()[0]
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SECONDS * 4 * attempt
            print(f"  {season} {season_type}: {exc!r}, retrying in {wait:.0f}s")
            time.sleep(wait)

    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    df.to_parquet(path)
    time.sleep(SLEEP_SECONDS)
    return df


def add_opponent_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's box score to each team-game row."""
    stat_cols = ["TEAM_ID", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                 "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
    opp = games[["GAME_ID"] + stat_cols].rename(
        columns={c: f"OPP_{c}" for c in stat_cols}
    )
    merged = games.merge(opp, on="GAME_ID")
    return merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].reset_index(drop=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            df = fetch_season(season, season_type)
            print(f"{season} {season_type}: {len(df)} team-game rows")
            frames.append(df)

    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)
    games["WIN"] = (games["WL"] == "W").astype(int)
    games["FG2A"] = games["FGA"] - games["FG3A"]
    games["FG2M"] = games["FGM"] - games["FG3M"]
    games = add_opponent_columns(games)
    games["MARGIN"] = games["PTS"] - games["OPP_PTS"]
    # Standard possession estimate, averaged over both teams.
    team_poss = games["FGA"] + 0.44 * games["FTA"] - games["OREB"] + games["TOV"]
    opp_poss = (games["OPP_FGA"] + 0.44 * games["OPP_FTA"]
                - games["OPP_OREB"] + games["OPP_TOV"])
    games["POSS"] = (team_poss + opp_poss) / 2

    out = PROCESSED / "team_games.parquet"
    games.to_parquet(out)
    print(f"\nWrote {len(games)} rows to {out}")
    print(games.groupby(["SEASON", "SEASON_TYPE"]).size().unstack())


if __name__ == "__main__":
    main()
