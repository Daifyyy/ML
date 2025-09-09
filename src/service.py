from __future__ import annotations
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

from config import AppConfig, get_api_key
from client import APIFootballClient
from storage import ParquetStore, SQLiteStore
from fetchers import FixturesFetcher, StandingsFetcher, TeamStatsFetcher, InjuriesFetcher, H2HFetcher

class DataService:
    """
    Orchestrates fetching and storage.
    - Full snapshot: iterate leagues/seasons; fetch everything selected in config.
    - Incremental: for fixtures, you can restrict to recent days; for injuries, use today's date.
    """
    def __init__(self, cfg: AppConfig):
        key = get_api_key()
        self.client = APIFootballClient(
            api_key=key,
            api_base=cfg.api_base,
            min_ms_between_calls=cfg.min_ms_between_calls,
            max_requests_per_minute=cfg.max_requests_per_minute,
        )
        self.cfg = cfg
        self.parquet = ParquetStore(cfg.parquet_dir)
        self.sqlite = SQLiteStore(cfg.sqlite_path)

        # fetchers
        self.fx = FixturesFetcher(self.client)
        self.std = StandingsFetcher(self.client)
        self.tst = TeamStatsFetcher(self.client)
        self.inj = InjuriesFetcher(self.client)
        self.h2h = H2HFetcher(self.client)

    def _unique_team_ids_from_fixtures(self, fixture_rows: List[Dict[str, Any]]) -> Set[int]:
        teams = set()
        for r in fixture_rows:
            teams.add(r["home_id"]); teams.add(r["away_id"])
        return teams

    def full_snapshot(self, leagues: List[int], seasons: List[int]) -> None:
        for league in leagues:
            for season in seasons:
                print(f"Snapshot: league={league}, season={season}")

                # Fixtures (finished)
                if self.cfg.include_fixtures_finished:
                    fx_ft = self.fx.fetch(league, season, status="FT")
                    self.sqlite.upsert_fixtures(fx_ft)
                    self.parquet.write_partition("fixtures_finished", league, season, pd.DataFrame(fx_ft))

                # Fixtures (scheduled)
                if self.cfg.include_fixtures_scheduled:
                    fx_ns = self.fx.fetch(league, season, status="NS")
                    self.sqlite.upsert_fixtures(fx_ns)
                    self.parquet.write_partition("fixtures_scheduled", league, season, pd.DataFrame(fx_ns))

                # Standings
                if self.cfg.include_standings:
                    st = self.std.fetch(league, season)
                    self.sqlite.upsert_standings(st)
                    self.parquet.write_partition("standings", league, season, pd.DataFrame(st))

                # Team stats: derive team ids from fixtures or standings
                if self.cfg.include_team_statistics:
                    fx_for_teams = fx_ft if self.cfg.include_fixtures_finished else []
                    team_ids = self._unique_team_ids_from_fixtures(fx_for_teams) or {r["team_id"] for r in st}
                    rows = []
                    for tid in tqdm(sorted(team_ids), desc=f"team_stats l{league} s{season}"):
                        rows.append(self.tst.fetch(league, season, tid))
                    self.sqlite.upsert_team_stats(rows)
                    self.parquet.write_partition("team_stats", league, season, pd.DataFrame(rows))

                # Injuries: snapshot by "today". You can also use per-fixture dates if desired.
                if self.cfg.include_injuries:
                    today = datetime.utcnow().date().isoformat()
                    rows = []
                    team_ids = self._unique_team_ids_from_fixtures(fx_ft)
                    for tid in tqdm(sorted(team_ids), desc=f"injuries l{league} s{season}"):
                        rows.extend(self.inj.fetch(league, season, tid, today))
                    self.sqlite.upsert_injuries(rows)
                    self.parquet.write_partition("injuries", league, season, pd.DataFrame(rows))

                # H2H: optional – cross product of teams in that league/season (bounded by h2h_last)
                if self.cfg.include_h2h:
                    team_ids = sorted(self._unique_team_ids_from_fixtures(fx_ft))
                    h2h_rows = []
                    for i in range(len(team_ids)):
                        for j in range(i+1, len(team_ids)):
                            h2h_rows.extend(self.h2h.fetch(team_ids[i], team_ids[j], last=self.cfg.h2h_last))
                    self.sqlite.upsert_h2h(h2h_rows)
                    self.parquet.write_partition("h2h", league, season, pd.DataFrame(h2h_rows))

    def incremental(self, leagues: List[int], seasons: List[int], days_back: int = 14) -> None:
        """
        Pull only recently finished fixtures and scheduled updates,
        update standings, team stats for involved teams, and today's injuries.
        """
        since = datetime.utcnow() - timedelta(days=days_back)
        since_iso = since.isoformat()

        for league in leagues:
            for season in seasons:
                print(f"Incremental: league={league}, season={season}, since={since_iso}")

                # Finished recently
                fx_ft = self.fx.fetch(league, season, status="FT")
                fx_ft_recent = [r for r in fx_ft if r["date"] >= since_iso]
                self.sqlite.upsert_fixtures(fx_ft_recent)
                self.parquet.write_partition("fixtures_finished_inc", league, season, pd.DataFrame(fx_ft_recent))

                # Upcoming / NS
                fx_ns = self.fx.fetch(league, season, status="NS")
                self.sqlite.upsert_fixtures(fx_ns)
                self.parquet.write_partition("fixtures_scheduled_inc", league, season, pd.DataFrame(fx_ns))

                # Standings
                st = self.std.fetch(league, season)
                self.sqlite.upsert_standings(st)
                self.parquet.write_partition("standings_inc", league, season, pd.DataFrame(st))

                # Team stats for involved teams only
                teams_changed = self._unique_team_ids_from_fixtures(fx_ft_recent or fx_ns)
                rows = []
                for tid in sorted(teams_changed):
                    rows.append(self.tst.fetch(league, season, tid))
                self.sqlite.upsert_team_stats(rows)
                self.parquet.write_partition("team_stats_inc", league, season, pd.DataFrame(rows))

                # Injuries today for involved teams
                today = datetime.utcnow().date().isoformat()
                inj_rows = []
                for tid in sorted(teams_changed):
                    inj_rows.extend(self.inj.fetch(league, season, tid, today))
                self.sqlite.upsert_injuries(inj_rows)
                self.parquet.write_partition("injuries_inc", league, season, pd.DataFrame(inj_rows))
