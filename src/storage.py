from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os, json, sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

@dataclass
class StoragePaths:
    parquet_dir: str
    sqlite_path: str

class ParquetStore:
    """
    Writes normalized tables to partitioned Parquet (league/season), one dataset per logical entity.
    """
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, dataset: str, league: int, season: int) -> str:
        p = os.path.join(self.base_dir, dataset, f"league={league}", f"season={season}")
        os.makedirs(p, exist_ok=True)
        return os.path.join(p, "part.parquet")

    def write_partition(self, dataset: str, league: int, season: int, df: pd.DataFrame) -> None:
        path = self._path(dataset, league, season)
        if not len(df):
            return
        df.to_parquet(path, index=False)

class SQLiteStore:
    """
    Upsert-friendly mirror for querying.
    Tables:
      - fixtures (pk: fixture_id)
      - standings (pk: league_id, season, team_id)
      - team_stats (pk: league_id, season, team_id)
      - injuries (pk: league_id, season, team_id, date, player_id?)
      - h2h (pk: fixture_id)
    """
    def __init__(self, sqlite_path: str) -> None:
        self.engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
        self._init_schema()

    def _init_schema(self):
        with self.engine.begin() as con:
            con.execute(text("""
            CREATE TABLE IF NOT EXISTS fixtures (
              fixture_id INTEGER PRIMARY KEY,
              date TEXT, status TEXT,
              league_id INTEGER, season INTEGER,
              home_id INTEGER, home_name TEXT,
              away_id INTEGER, away_name TEXT,
              goals_home INTEGER, goals_away INTEGER,
              raw_json TEXT
            );
            """))
            con.execute(text("""
            CREATE TABLE IF NOT EXISTS standings (
              league_id INTEGER, season INTEGER, team_id INTEGER,
              rank INTEGER, points INTEGER, goalsDiff INTEGER,
              played INTEGER, win INTEGER, draw INTEGER, lose INTEGER,
              gf INTEGER, ga INTEGER,
              home_pts_pg REAL, away_pts_pg REAL,
              form TEXT,
              PRIMARY KEY (league_id, season, team_id)
            );
            """))
            con.execute(text("""
            CREATE TABLE IF NOT EXISTS team_stats (
              league_id INTEGER, season INTEGER, team_id INTEGER,
              raw_json TEXT,
              PRIMARY KEY (league_id, season, team_id)
            );
            """))
            con.execute(text("""
            CREATE TABLE IF NOT EXISTS injuries (
              league_id INTEGER, season INTEGER, team_id INTEGER,
              date TEXT, player_name TEXT, player_id INTEGER,
              reason TEXT, position TEXT,
              PRIMARY KEY (league_id, season, team_id, date, player_id)
            );
            """))
            con.execute(text("""
            CREATE TABLE IF NOT EXISTS h2h (
              fixture_id INTEGER PRIMARY KEY,
              league_id INTEGER, season INTEGER,
              home_id INTEGER, away_id INTEGER,
              goals_home INTEGER, goals_away INTEGER,
              date TEXT, raw_json TEXT
            );
            """))

    def upsert_fixtures(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as con:
            for r in rows:
                con.execute(text("""
                INSERT INTO fixtures (fixture_id, date, status, league_id, season, home_id, home_name, away_id, away_name, goals_home, goals_away, raw_json)
                VALUES (:fixture_id, :date, :status, :league_id, :season, :home_id, :home_name, :away_id, :away_name, :goals_home, :goals_away, :raw_json)
                ON CONFLICT(fixture_id) DO UPDATE SET
                  date=excluded.date, status=excluded.status, league_id=excluded.league_id,
                  season=excluded.season, home_id=excluded.home_id, home_name=excluded.home_name,
                  away_id=excluded.away_id, away_name=excluded.away_name,
                  goals_home=excluded.goals_home, goals_away=excluded.goals_away,
                  raw_json=excluded.raw_json;
                """), r)

    def upsert_standings(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as con:
            for r in rows:
                con.execute(text("""
                INSERT INTO standings (league_id, season, team_id, rank, points, goalsDiff, played, win, draw, lose, gf, ga, home_pts_pg, away_pts_pg, form)
                VALUES (:league_id, :season, :team_id, :rank, :points, :goalsDiff, :played, :win, :draw, :lose, :gf, :ga, :home_pts_pg, :away_pts_pg, :form)
                ON CONFLICT(league_id, season, team_id) DO UPDATE SET
                  rank=excluded.rank, points=excluded.points, goalsDiff=excluded.goalsDiff,
                  played=excluded.played, win=excluded.win, draw=excluded.draw, lose=excluded.lose,
                  gf=excluded.gf, ga=excluded.ga, home_pts_pg=excluded.home_pts_pg,
                  away_pts_pg=excluded.away_pts_pg, form=excluded.form;
                """), r)

    def upsert_team_stats(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as con:
            for r in rows:
                con.execute(text("""
                INSERT INTO team_stats (league_id, season, team_id, raw_json)
                VALUES (:league_id, :season, :team_id, :raw_json)
                ON CONFLICT(league_id, season, team_id) DO UPDATE SET
                  raw_json=excluded.raw_json;
                """), r)

    def upsert_injuries(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as con:
            for r in rows:
                con.execute(text("""
                INSERT INTO injuries (league_id, season, team_id, date, player_name, player_id, reason, position)
                VALUES (:league_id, :season, :team_id, :date, :player_name, :player_id, :reason, :position)
                ON CONFLICT(league_id, season, team_id, date, player_id) DO UPDATE SET
                  player_name=excluded.player_name, reason=excluded.reason, position=excluded.position;
                """), r)

    def upsert_h2h(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as con:
            for r in rows:
                con.execute(text("""
                INSERT INTO h2h (fixture_id, league_id, season, home_id, away_id, goals_home, goals_away, date, raw_json)
                VALUES (:fixture_id, :league_id, :season, :home_id, :away_id, :goals_home, :goals_away, :date, :raw_json)
                ON CONFLICT(fixture_id) DO UPDATE SET
                  league_id=excluded.league_id, season=excluded.season, home_id=excluded.home_id,
                  away_id=excluded.away_id, goals_home=excluded.goals_home, goals_away=excluded.goals_away,
                  date=excluded.date, raw_json=excluded.raw_json;
                """), r)
