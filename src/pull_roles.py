"""Pull player season stats and five-man lineup stats for the roles analysis (Q3).

Per season (regular season only), cached to data/raw/roles/:
  players_base_<season>.parquet      LeagueDashPlayerStats, Base, Totals
  players_adv_<season>.parquet       LeagueDashPlayerStats, Advanced, Totals
  bio_<season>.parquet               LeagueDashPlayerBioStats (height)
  lineups_adv_<season>_<team_id>.parquet  LeagueDashLineups, 5-man, Advanced, Totals, one team
                                     (the league-wide request is silently cut to 2,000 rows)

Cached files are skipped on rerun. Usage: python src/pull_roles.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashLineups,
    LeagueDashPlayerBioStats,
    LeagueDashPlayerStats,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "roles"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]
TEAM_IDS = list(range(1610612737, 1610612767))
SLEEP = 1.0
RETRIES = 3


def fetch(path: Path, make):
    if path.exists():
        return
    for attempt in range(1, RETRIES + 1):
        try:
            df = make().get_data_frames()[0]
            df.to_parquet(path)
            print(f"saved {path.name}: {len(df)} rows", flush=True)
            time.sleep(SLEEP)
            return
        except Exception as exc:  # network drops, rate limits
            print(f"  retry {attempt} for {path.name}: {exc}", flush=True)
            time.sleep(5 * attempt)
    print(f"FAILED {path.name}", flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        fetch(RAW / f"players_base_{s}.parquet", lambda: LeagueDashPlayerStats(
            season=s, per_mode_detailed="Totals", measure_type_detailed_defense="Base", timeout=60))
        fetch(RAW / f"players_adv_{s}.parquet", lambda: LeagueDashPlayerStats(
            season=s, per_mode_detailed="Totals", measure_type_detailed_defense="Advanced", timeout=60))
        fetch(RAW / f"bio_{s}.parquet", lambda: LeagueDashPlayerBioStats(season=s, timeout=60))
        for tid in TEAM_IDS:
            fetch(RAW / f"lineups_adv_{s}_{tid}.parquet", lambda: LeagueDashLineups(
                season=s, group_quantity="5", per_mode_detailed="Totals",
                measure_type_detailed_defense="Advanced", team_id_nullable=str(tid), timeout=60))


if __name__ == "__main__":
    main()
