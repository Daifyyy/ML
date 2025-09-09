from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os, yaml

@dataclass(frozen=True)
class AppConfig:
    api_base: str
    parquet_dir: str
    sqlite_path: str
    data_dir: str
    leagues: List[int]
    seasons: List[int]
    max_requests_per_minute: int
    min_ms_between_calls: int
    include_fixtures_finished: bool
    include_fixtures_scheduled: bool
    include_standings: bool
    include_team_statistics: bool
    include_injuries: bool
    include_h2h: bool
    h2h_last: int

    @staticmethod
    def load(path: str = "config.yaml") -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        inc = cfg.get("include", {})
        return AppConfig(
            api_base=cfg["api_base"],
            data_dir=cfg["data_dir"],
            parquet_dir=cfg["parquet_dir"],
            sqlite_path=cfg["sqlite_path"],
            leagues=cfg["leagues"],
            seasons=cfg["seasons"],
            max_requests_per_minute=cfg.get("max_requests_per_minute", 55),
            min_ms_between_calls=cfg.get("min_ms_between_calls", 100),
            include_fixtures_finished=inc.get("fixtures_finished", True),
            include_fixtures_scheduled=inc.get("fixtures_scheduled", True),
            include_standings=inc.get("standings", True),
            include_team_statistics=inc.get("team_statistics", True),
            include_injuries=inc.get("injuries", True),
            include_h2h=inc.get("h2h", True),
            h2h_last=cfg.get("h2h_last", 10),
        )

def get_api_key() -> str:
    """
    Reads API key from env var. In GitHub Actions, define:
    - Secret: API_FOOTBALL_KEY
    - (Optional) Repository Variable: API_FOOTBALL_ENV="prod"
    """
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("Missing API_FOOTBALL_KEY environment variable.")
    return key
