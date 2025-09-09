from __future__ import annotations
from typing import Dict, Any, List, Iterable
from dataclasses import dataclass
import json
import pandas as pd
from .client import APIFootballClient

@dataclass
class FixtureCore:
    fixture_id: int
    date: str
    status: str
    league_id: int
    season: int
    home_id: int
    home_name: str
    away_id: int
    away_name: str
    goals_home: int | None
    goals_away: int | None

def parse_fixture(item: Dict[str, Any]) -> FixtureCore:
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    return FixtureCore(
        fixture_id=fixture.get("id"),
        date=fixture.get("date"),
        status=fixture.get("status", {}).get("short"),
        league_id=league.get("id"),
        season=league.get("season"),
        home_id=teams.get("home", {}).get("id"),
        home_name=teams.get("home", {}).get("name"),
        away_id=teams.get("away", {}).get("id"),
        away_name=teams.get("away", {}).get("name"),
        goals_home=goals.get("home"),
        goals_away=goals.get("away"),
    )

class FixturesFetcher:
    def __init__(self, client: APIFootballClient):
        self.client = client

    def fetch(self, league: int, season: int, status: str) -> List[Dict[str, Any]]:
        """
        status: 'FT' for finished, 'NS' for not started (scheduled)
        """
        items = list(self.client.get_all_pages("/fixtures", {"league": league, "season": season, "status": status}))
        rows = []
        for it in items:
            f = parse_fixture(it)
            rows.append({
                "fixture_id": f.fixture_id,
                "date": f.date,
                "status": f.status,
                "league_id": f.league_id,
                "season": f.season,
                "home_id": f.home_id,
                "home_name": f.home_name,
                "away_id": f.away_id,
                "away_name": f.away_name,
                "goals_home": f.goals_home,
                "goals_away": f.goals_away,
                "raw_json": json.dumps(it, ensure_ascii=False)
            })
        return rows

class StandingsFetcher:
    def __init__(self, client: APIFootballClient):
        self.client = client

    def fetch(self, league: int, season: int) -> List[Dict[str, Any]]:
        items = list(self.client.get_all_pages("/standings", {"league": league, "season": season}))
        out = []
        for it in items:
            tables = it.get("league", {}).get("standings", [])
            for table in tables:
                for row in table:
                    team = row.get("team", {})
                    tid = team.get("id")
                    out.append({
                        "league_id": league, "season": season, "team_id": tid,
                        "rank": row.get("rank"),
                        "points": row.get("points"),
                        "goalsDiff": row.get("goalsDiff"),
                        "played": row.get("all", {}).get("played"),
                        "win": row.get("all", {}).get("win"),
                        "draw": row.get("all", {}).get("draw"),
                        "lose": row.get("all", {}).get("lose"),
                        "gf": row.get("all", {}).get("goals", {}).get("for"),
                        "ga": row.get("all", {}).get("goals", {}).get("against"),
                        "home_pts_pg": self._ppg(row.get("home", {})),
                        "away_pts_pg": self._ppg(row.get("away", {})),
                        "form": row.get("form"),
                    })
        return out

    @staticmethod
    def _ppg(block: Dict[str, Any]) -> float | None:
        pts = block.get("points")
        played = block.get("played") or 0
        if not played:
            return None
        return round(pts / played, 3) if pts is not None else None

class TeamStatsFetcher:
    def __init__(self, client: APIFootballClient):
        self.client = client

    def fetch(self, league: int, season: int, team_id: int) -> Dict[str, Any]:
        # Single call per team; you can collect team_ids from standings or fixtures
        data = self.client.get_page("/teams/statistics", {"league": league, "season": season, "team": team_id})
        resp = data.get("response", {})
        return {
            "league_id": league,
            "season": season,
            "team_id": team_id,
            "raw_json": json.dumps(resp, ensure_ascii=False)
        }

class InjuriesFetcher:
    def __init__(self, client: APIFootballClient):
        self.client = client

    def fetch(self, league: int, season: int, team_id: int, date_iso: str) -> List[Dict[str, Any]]:
        items = list(self.client.get_all_pages("/injuries", {"league": league, "season": season, "team": team_id, "date": date_iso}))
        out = []
        for it in items:
            player = it.get("player", {}) or {}
            out.append({
                "league_id": league, "season": season, "team_id": team_id,
                "date": date_iso,
                "player_name": player.get("name"),
                "player_id": player.get("id"),
                "reason": it.get("player", {}).get("reason"),
                "position": player.get("position"),
            })
        return out

class H2HFetcher:
    def __init__(self, client: APIFootballClient):
        self.client = client

    def fetch(self, teamA: int, teamB: int, last: int = 10) -> List[Dict[str, Any]]:
        items = list(self.client.get_all_pages("/fixtures/headtohead", {"h2h": f"{teamA}-{teamB}", "last": last}))
        out = []
        for it in items:
            fc = parse_fixture(it)
            out.append({
                "fixture_id": fc.fixture_id,
                "league_id": fc.league_id,
                "season": fc.season,
                "home_id": fc.home_id,
                "away_id": fc.away_id,
                "goals_home": fc.goals_home,
                "goals_away": fc.goals_away,
                "date": fc.date,
                "raw_json": json.dumps(it, ensure_ascii=False)
            })
        return out
