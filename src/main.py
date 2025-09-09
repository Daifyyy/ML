from __future__ import annotations
import argparse
from config import AppConfig
from service import DataService

def parse_args():
    p = argparse.ArgumentParser(description="API-Football data acquisition to Parquet/SQLite (offline).")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--mode", choices=["snapshot", "incremental"], default="snapshot")
    p.add_argument("--leagues", nargs="*", type=int, help="Override league ids")
    p.add_argument("--seasons", nargs="*", type=int, help="Override seasons (e.g., 2023 2024)")
    p.add_argument("--days-back", type=int, default=14, help="Incremental look-back days")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = AppConfig.load(args.config)
    leagues = args.leagues or cfg.leagues
    seasons = args.seasons or cfg.seasons

    svc = DataService(cfg)

    if args.mode == "snapshot":
        svc.full_snapshot(leagues, seasons)
    else:
        svc.incremental(leagues, seasons, days_back=args.days_back)

if __name__ == "__main__":
    main()
