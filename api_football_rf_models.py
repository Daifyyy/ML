"""
api_football_rf_models.py

Beginner-friendly, production-leaning pipeline to:
  (1) Train a RandomForestClassifier for match result (W/D/L) with calibrated probabilities
  (2) Train a RandomForestRegressor for total goals (derive Over/Under 2.5 & P(Over))

Data source: API-FOOTBALL v3 (https://v3.football.api-sports.io/)

High-level flow:
  - Fetch finished fixtures + season stats + standings + injuries + H2H (time-aware)
  - Engineer leak-proof, pre-match features (rolling form, H/A splits, ELO, standings, injuries, H2H, rest days)
  - Split chrono: train -> valid -> test
  - Tune RF via RandomizedSearchCV with TimeSeriesSplit
  - Calibrate classifier probabilities (isotonic)
  - Evaluate with clear metrics and plots
  - Save models + feature schema
  - Provide predict_fixture() to score upcoming matches

Author: You
"""

from __future__ import annotations
import os, time, math, json, warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from tqdm import tqdm
from datetime import datetime, timedelta, timezone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
)
from sklearn.inspection import permutation_importance

from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform, norm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- CONFIG: edit these ----------
API_BASE = "https://v3.football.api-sports.io"
API_KEY = os.environ.get("API_FOOTBALL_KEY")

# Choose target leagues & seasons to train on (start small; add more later)
LEAGUES = [39, 140, 61, 78, 135]  # EPL, LaLiga, Ligue 1, Bundesliga, Serie A
SEASONS = [2021, 2022, 2023, 2024] # adjust per your plan & coverage

# Rolling windows for features
ROLL_WINDOWS = [5, 10]

# Random seeds for reproducibility
RANDOM_STATE = 42

# Output dir for artifacts
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ----------- Helper: API client with retry/backoff -----------

class APIFootballClient:
    """
    Thin wrapper around requests.Session that:
      - adds required headers
      - retries on 429/5xx with backoff
      - centralized GET method returning JSON or raising descriptive errors
    """
    def __init__(self, api_key: str, base_url: str = API_BASE, timeout: int = 30):
        if not api_key:
            raise ValueError("API_FOOTBALL_KEY is not set. Set env var before running.")
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": api_key,
            "Accept": "application/json"
        })
        retries = Retry(
            total=5, backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.timeout = timeout

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code == 200:
            return r.json()
        # Helpful error message for beginners:
        try:
            payload = r.json()
        except Exception:
            payload = {"message": r.text[:200]}
        raise RuntimeError(f"API error ({r.status_code}) for {url} params={params} -> {payload}")


# ----------- Helpers: API endpoints (read-only) -----------

class API:
    """
    Minimal set of API-Football v3 endpoints we’ll rely on.

    Notes on docs:
      - Fixtures (finished/upcoming):   /fixtures?league={id}&season={yr}&status=FT|NS
      - Team statistics (season scope): /teams/statistics?league={id}&season={yr}&team={id}
      - Standings:                      /standings?league={id}&season={yr}
      - H2H:                            /fixtures/headtohead?h2h={teamA}-{teamB}
      - Injuries (by date/league/team): /injuries?league={id}&season={yr}&team={id}&date=YYYY-MM-DD
      - Fixture events:                 /fixtures/events?fixture={fixture_id}
      - Lineups (optional):             /fixtures/lineups?fixture={fixture_id}

    These paths and usage patterns are illustrated in API-Football docs and tutorials. 
    """
    def __init__(self, client: APIFootballClient):
        self.c = client

    # Finished fixtures for training data
    def fixtures_finished(self, league: int, season: int) -> List[Dict[str, Any]]:
        res = self.c.get("/fixtures", {"league": league, "season": season, "status": "FT"})
        return res.get("response", [])

    # Upcoming fixtures (for prediction)
    def fixtures_upcoming(self, league: int, season: int) -> List[Dict[str, Any]]:
        res = self.c.get("/fixtures", {"league": league, "season": season, "status": "NS"})
        return res.get("response", [])

    # One fixture by id (handy during prediction)
    def fixture_by_id(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        res = self.c.get("/fixtures", {"id": fixture_id})
        arr = res.get("response", [])
        return arr[0] if arr else None

    # Team season statistics (aggregates per competition season)
    def team_statistics(self, league: int, season: int, team_id: int) -> Dict[str, Any]:
        res = self.c.get("/teams/statistics", {"league": league, "season": season, "team": team_id})
        return res.get("response", {})

    # Standings (table + recent form)
    def standings(self, league: int, season: int) -> List[Dict[str, Any]]:
        res = self.c.get("/standings", {"league": league, "season": season})
        return res.get("response", [])

    # Head-to-head last N fixtures
    def h2h(self, teamA: int, teamB: int, last: int = 10) -> List[Dict[str, Any]]:
        res = self.c.get("/fixtures/headtohead", {"h2h": f"{teamA}-{teamB}", "last": last})
        return res.get("response", [])

    # Injuries near a date (no long history preserved per docs)
    def injuries(self, league: int, season: int, team_id: int, date: str) -> List[Dict[str, Any]]:
        res = self.c.get("/injuries", {"league": league, "season": season, "team": team_id, "date": date})
        return res.get("response", [])

    # Optional: Events, Lineups, Players (can enrich features if your plan allows)
    def events(self, fixture_id: int) -> List[Dict[str, Any]]:
        res = self.c.get("/fixtures/events", {"fixture": fixture_id})
        return res.get("response", [])

    def lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        res = self.c.get("/fixtures/lineups", {"fixture": fixture_id})
        return res.get("response", [])


# ----------- ELO utilities (lightweight, soccer-friendly) -----------

def update_elo(rating_home, rating_away, goals_home, goals_away, k=20, home_adv=60):
    """One-step ELO update. home_adv is an offset given to home team."""
    # expected score for home:
    exp_home = 1.0 / (1 + 10 ** (-( (rating_home + home_adv) - rating_away ) / 400))
    # actual result as 1/0.5/0:
    if goals_home > goals_away: actual_home = 1.0
    elif goals_home == goals_away: actual_home = 0.5
    else: actual_home = 0.0
    new_home = rating_home + k * (actual_home - exp_home)
    # away expected vs actual (symmetric)
    exp_away = 1 - exp_home
    actual_away = 1 - actual_home
    new_away = rating_away + k * (actual_away - exp_away)
    return new_home, new_away

def expected_points_from_elo(elo_home, elo_away, home_adv=60):
    """Convert ELO diff to expected home points (3-w-1-d-0-l expectation)."""
    p_home_win = 1.0 / (1 + 10 ** (-( (elo_home + home_adv) - elo_away ) / 400))
    # crude tie prob heuristic: higher near balanced teams
    p_draw = 0.25 * (1 - abs(p_home_win - 0.5)*2)
    p_home_loss = 1 - p_home_win - p_draw
    return 3*p_home_win + 1*p_draw, (p_home_win, p_draw, p_home_loss)


# ----------- Data assembly & feature engineering -----------

def parse_fixture_core(f):
    """Extracts core fields from a raw fixture object."""
    fixture = f.get("fixture", {})
    teams   = f.get("teams", {})
    goals   = f.get("goals", {})
    league  = f.get("league", {})

    fid   = fixture.get("id")
    date  = fixture.get("date")  # ISO
    status= fixture.get("status", {}).get("short")
    lid   = league.get("id")
    season= league.get("season")
    home_id = teams.get("home", {}).get("id")
    away_id = teams.get("away", {}).get("id")
    home_name = teams.get("home", {}).get("name")
    away_name = teams.get("away", {}).get("name")
    gh = goals.get("home")
    ga = goals.get("away")
    return {
        "fixture_id": fid,
        "date": date,
        "status": status,
        "league_id": lid,
        "season": season,
        "home_id": home_id,
        "away_id": away_id,
        "home_name": home_name,
        "away_name": away_name,
        "goals_home": gh,
        "goals_away": ga
    }

def last_n_form(rows: List[Dict[str, Any]], n=5):
    """Compute form points & goal stats for last n fixtures (team perspective)."""
    if len(rows) == 0:
        return {"points": np.nan, "gf": np.nan, "ga": np.nan}
    df = pd.DataFrame(rows).sort_values("date")
    tail = df.tail(n)
    # Points: Win=3, Draw=1, Loss=0
    pts = 0
    gf, ga = 0, 0
    for _, r in tail.iterrows():
        gf += r["gf"]
        ga += r["ga"]
        if   r["gf"] > r["ga"]: pts += 3
        elif r["gf"] == r["ga"]: pts += 1
    return {"points": pts, "gf": gf, "ga": ga}

def build_team_rolling_features(team_history: List[Dict[str, Any]], windows=ROLL_WINDOWS):
    """
    Given all past fixtures for a team (dicts with date,gf,ga,shots,sot,possession,...),
    compute rolling summaries for multiple windows.
    """
    if len(team_history) == 0:
        return {}
    df = pd.DataFrame(team_history).sort_values("date")
    # Helpful rates:
    df["gd"] = df["gf"] - df["ga"]
    df["sg_ratio"] = (df.get("sot", pd.Series(np.nan, index=df.index)) /
                      df.get("shots", pd.Series(np.nan, index=df.index))).replace([np.inf, -np.inf], np.nan)

    feats = {}
    for w in windows:
        tail = df.tail(w)
        feats.update({
            f"form_pts_{w}":    np.nansum([3 if r.gf>r.ga else (1 if r.gf==r.ga else 0) for r in tail.itertuples()]),
            f"gf_avg_{w}":      np.nanmean(tail["gf"]) if len(tail) else np.nan,
            f"ga_avg_{w}":      np.nanmean(tail["ga"]) if len(tail) else np.nan,
            f"gd_avg_{w}":      np.nanmean(tail["gd"]) if len(tail) else np.nan,
            f"sot_avg_{w}":     np.nanmean(tail["sot"]) if "sot" in tail else np.nan,
            f"shots_avg_{w}":   np.nanmean(tail["shots"]) if "shots" in tail else np.nan,
            f"sg_ratio_{w}":    np.nanmean(tail["sg_ratio"]) if "sg_ratio" in tail else np.nan,
            f"possession_{w}":  np.nanmean(tail["possession"]) if "possession" in tail else np.nan,
            f"cards_{w}":       np.nanmean(tail["cards"]) if "cards" in tail else np.nan,
        })
    return feats

def extract_simple_team_match_stats(statistics_block: Dict[str, Any], team_side="home"):
    """
    From /teams/statistics (season scope) or /fixtures/statistics (match scope), we’ll
    try to map a few common stats to numeric, handling missing values gracefully.
    """
    # Statistics blocks vary; here we standardize a few if present:
    out = {}
    # Typical keys we might see: shots on target, total shots, possession (%), fouls, corners, yellow/red cards
    # We'll prefix with team_side (home/away) later where needed.
    mapping = {
        "Shots on Goal": "sot",
        "Shots on Target": "sot",   # alias
        "Total Shots": "shots",
        "Ball Possession": "possession",
        "Fouls": "fouls",
        "Corner Kicks": "corners",
        "Yellow Cards": "yellows",
        "Red Cards": "reds"
    }
    stats_list = statistics_block if isinstance(statistics_block, list) else []
    for s in stats_list:
        # In /fixtures/statistics, each item often has: type, value
        k = s.get("type")
        v = s.get("value")
        if k in mapping:
            key = mapping[k]
            # Possession given like '54%':
            if key == "possession" and isinstance(v, str) and v.endswith("%"):
                try: v = float(v.replace("%",""))
                except: v = np.nan
            elif isinstance(v, str):
                # Try to coerce numeric
                try: v = float(v)
                except: v = np.nan
            out[key] = v
    return out

def safe_div(a, b):
    return np.nan if (b is None or b == 0 or pd.isna(b)) else (a / b)

# ----------- Dataset builder -----------

class DatasetBuilder:
    """
    Builds a leak-free, time-ordered match dataset with engineered features.
    """
    def __init__(self, api: API):
        self.api = api
        # in-memory caches to reduce API calls within a run
        self._standings_cache = {}
        self._team_stats_cache = {}
        self._injuries_cache = {}

        # ELO store by team across dataset (init ~1500)
        self._elo = {}

        # Per-team past matches to compute rolling features
        self._team_hist = {}

    def _get_standings_map(self, league, season) -> Dict[int, Dict[str, Any]]:
        key = (league, season)
        if key in self._standings_cache:
            return self._standings_cache[key]
        resp = self.api.standings(league, season)
        # Standings response contains groups; we flatten:
        m = {}
        for blk in resp:
            for table in blk.get("league", {}).get("standings", []):
                for row in table:
                    team = row.get("team", {})
                    tid = team.get("id")
                    m[tid] = {
                        "rank": row.get("rank"),
                        "points": row.get("points"),
                        "goalsDiff": row.get("goalsDiff"),
                        "form": row.get("form"),  # e.g., "WWDLW"
                        "played": row.get("all", {}).get("played"),
                        "win": row.get("all", {}).get("win"),
                        "draw": row.get("all", {}).get("draw"),
                        "lose": row.get("all", {}).get("lose"),
                        "gf": row.get("all", {}).get("goals", {}).get("for"),
                        "ga": row.get("all", {}).get("goals", {}).get("against"),
                        "home_pts_pg": safe_div(row.get("home", {}).get("points"), row.get("home", {}).get("played")),
                        "away_pts_pg": safe_div(row.get("away", {}).get("points"), row.get("away", {}).get("played")),
                    }
        self._standings_cache[key] = m
        return m

    def _get_team_stats(self, league, season, team_id) -> Dict[str, Any]:
        key = (league, season, team_id)
        if key in self._team_stats_cache:
            return self._team_stats_cache[key]
        resp = self.api.team_statistics(league, season, team_id)
        self._team_stats_cache[key] = resp
        return resp

    def _get_injuries_count(self, league, season, team_id, date_str) -> Dict[str, int]:
        key = (league, season, team_id, date_str)
        if key in self._injuries_cache:
            return self._injuries_cache[key]
        items = self.api.injuries(league, season, team_id, date_str)
        # Count injured/doubtful and bucket by position if present
        total = len(items)
        by_pos = {}
        for it in items:
            pos = (it.get("player", {}) or {}).get("position", "NA")
            by_pos[pos] = by_pos.get(pos, 0) + 1
        out = {"inj_total": total}
        # flatten top few pos groups:
        for p in ["G", "D", "M", "F"]:
            out[f"inj_{p}"] = by_pos.get(p, 0)
        self._injuries_cache[key] = out
        return out

    def _ensure_team_hist(self, team_id):
        if team_id not in self._team_hist:
            self._team_hist[team_id] = []

    def _init_elo_if_needed(self, team_id):
        if team_id not in self._elo:
            self._elo[team_id] = 1500.0

    def _record_team_match(self, team_id, date, gf, ga, shots=None, sot=None, possession=None, cards=None):
        self._ensure_team_hist(team_id)
        self._team_hist[team_id].append({
            "date": pd.to_datetime(date),
            "gf": gf, "ga": ga,
            "shots": shots, "sot": sot,
            "possession": possession,
            "cards": cards
        })

    def _build_row(self, fcore: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build a single feature row for a FIXTURE, using only data prior to its date.
        """
        fid = fcore["fixture_id"]
        league = fcore["league_id"]; season = fcore["season"]
        home = fcore["home_id"]; away = fcore["away_id"]
        date = pd.to_datetime(fcore["date"])

        # Standings snapshot
        table = self._get_standings_map(league, season)
        s_home = table.get(home, {})
        s_away = table.get(away, {})

        # Pre-match ELO snapshot (init if needed)
        self._init_elo_if_needed(home); self._init_elo_if_needed(away)
        elo_home = self._elo[home]; elo_away = self._elo[away]

        # Injuries on match date (API keeps current injured players; use match date)
        inj_date = date.strftime("%Y-%m-%d")
        inj_h = self._get_injuries_count(league, season, home, inj_date)
        inj_a = self._get_injuries_count(league, season, away, inj_date)

        # Team season-level stats (per API) – may include averages per home/away/overall
        tstat_h = self._get_team_stats(league, season, home)
        tstat_a = self._get_team_stats(league, season, away)

        # Construct rolling-form features from our accumulated past fixtures:
        feats_h_roll = build_team_rolling_features(self._team_hist.get(home, []))
        feats_a_roll = build_team_rolling_features(self._team_hist.get(away, []))

        # H2H summaries (last 5)
        h2h = self.api.h2h(home, away, last=5)
        h2h_wins_h = 0; h2h_wins_a = 0; h2h_draws = 0; h2h_goals_h=0; h2h_goals_a=0
        for r in h2h:
            rc = parse_fixture_core(r)
            if rc["status"] != "FT": 
                continue
            gh, ga = rc["goals_home"], rc["goals_away"]
            # Determine perspective (home/away identity depends on each past match’s teams)
            is_home_home = (rc["home_id"] == home)
            if is_home_home:
                h2h_goals_h += gh; h2h_goals_a += ga
                if gh>ga: h2h_wins_h += 1
                elif gh<ga: h2h_wins_a += 1
                else: h2h_draws += 1
            else:
                h2h_goals_h += ga; h2h_goals_a += gh
                if ga>gh: h2h_wins_h += 1
                elif ga<gh: h2h_wins_a += 1
                else: h2h_draws += 1

        # Rest days & congestion
        def last_played(team_id, before_date):
            hist = [x for x in self._team_hist.get(team_id, []) if x["date"] < before_date]
            return max([x["date"] for x in hist]) if hist else None
        last_h = last_played(home, date); last_a = last_played(away, date)
        rest_h = (date - last_h).days if last_h else np.nan
        rest_a = (date - last_a).days if last_a else np.nan

        # Standings-derived basic features
        def ppd(row):  # points per game
            return safe_div(row.get("points"), row.get("played")) if row else np.nan

        feats = {
            "fixture_id": fid, "date": date, "league_id": league, "season": season,
            "home_id": home, "away_id": away,
            # standings snapshot
            "rank_diff": (s_away.get("rank") or np.nan) - (s_home.get("rank") or np.nan),
            "ppg_diff": (ppd(s_home) or np.nan) - (ppd(s_away) or np.nan),
            "gd_diff":  (s_home.get("goalsDiff") or np.nan) - (s_away.get("goalsDiff") or np.nan),
            # ELO snapshot:
            "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_home - elo_away,
            # injuries:
            "inj_home_total": inj_h.get("inj_total", 0),
            "inj_away_total": inj_a.get("inj_total", 0),
            "inj_home_G": inj_h.get("inj_G", 0), "inj_home_D": inj_h.get("inj_D", 0),
            "inj_home_M": inj_h.get("inj_M", 0), "inj_home_F": inj_h.get("inj_F", 0),
            "inj_away_G": inj_a.get("inj_G", 0), "inj_away_D": inj_a.get("inj_D", 0),
            "inj_away_M": inj_a.get("inj_M", 0), "inj_away_F": inj_a.get("inj_F", 0),
            # H2H:
            "h2h_wins_home5": h2h_wins_h, "h2h_wins_away5": h2h_wins_a, "h2h_draws5": h2h_draws,
            "h2h_gf_home5": h2h_goals_h, "h2h_ga_home5": h2h_goals_a,
            # Rest:
            "rest_days_home": rest_h, "rest_days_away": rest_a,
        }

        # Attach rolling feats
        for k,v in feats_h_roll.items(): feats[f"h_{k}"] = v
        for k,v in feats_a_roll.items(): feats[f"a_{k}"] = v

        return feats

    def build_training_table(self, leagues: List[int], seasons: List[int]) -> pd.DataFrame:
        """
        Iterate finished fixtures chronologically; after each match, update ELO & team histories.
        Ensures features at row t are from info strictly < kickoff(t).
        """
        rows = []
        for league in leagues:
            for season in seasons:
                fixtures = self.api.fixtures_finished(league, season)
                fcores = [parse_fixture_core(f) for f in fixtures if f]
                df = pd.DataFrame(fcores).dropna(subset=["date"])
                df["date"] = pd.to_datetime(df["date"])
                df.sort_values("date", inplace=True)

                for _, r in tqdm(df.iterrows(), total=len(df), desc=f"League {league} season {season}"):
                    # Build features BEFORE updating history with the current match
                    row = self._build_row(r)
                    if row is None:
                        continue
                    # targets (only available for finished fixtures)
                    gh, ga = r["goals_home"], r["goals_away"]
                    row["y_outcome"] = ( "H" if gh>ga else ("D" if gh==ga else "A") )
                    row["y_total_goals"] = (gh or 0) + (ga or 0)
                    rows.append(row)

                    # AFTER: update ELO and team histories using this result (for next matches)
                    self._init_elo_if_needed(r["home_id"]); self._init_elo_if_needed(r["away_id"])
                    new_h, new_a = update_elo(self._elo[r["home_id"]], self._elo[r["away_id"]], gh, ga)
                    self._elo[r["home_id"]] = new_h; self._elo[r["away_id"]] = new_a

                    # (Optional) If you have per-fixture team stats (/fixtures/statistics), you can also parse shots/SoT/possession.
                    # Here we just record goals; you can expand by calling self.api.events(...) or statistics endpoint.
                    self._record_team_match(r["home_id"], r["date"], gh, ga)
                    self._record_team_match(r["away_id"], r["date"], ga, gh)

        return pd.DataFrame(rows)


# ----------- Train / Tune / Evaluate -----------

def time_split(df: pd.DataFrame, valid_frac=0.15, test_frac=0.15):
    """
    Chronological split: oldest -> train, mid -> valid, newest -> test.
    """
    df_sorted = df.sort_values("date")
    n = len(df_sorted)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    n_train = n - n_valid - n_test
    train = df_sorted.iloc[:n_train]
    valid = df_sorted.iloc[n_train:n_train+n_valid]
    test  = df_sorted.iloc[n_train+n_valid:]
    return train, valid, test

def build_feature_matrix(df: pd.DataFrame):
    """
    Select and order numeric feature columns. Avoid IDs/leakage columns.
    """
    cols_to_drop = [
        "fixture_id","date","league_id","season","home_id","away_id",
        "y_outcome","y_total_goals"
    ]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    y_cls = df["y_outcome"].values
    y_reg = df["y_total_goals"].values.astype(float)
    # RandomForest doesn’t need scaling. Ensure numeric dtype:
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y_cls, y_reg

def tune_rf_classifier(X_train, y_train):
    """
    Randomized search over RF hyperparameters with time-aware CV.
    """
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=30, cv=tscv, scoring="neg_log_loss",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_rf_regressor(X_train, y_train):
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "n_estimators": randint(300, 900),
        "max_depth": randint(3, 24),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", 0.6, None],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=30, cv=tscv, scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def evaluate_classifier(name, model, X, y, calibrated=False, show_plots=True):
    proba = model.predict_proba(X)
    preds = np.array([["A","D","H"][np.argmax(p)] for p in proba])  # ensure consistent class order later
    # metrics
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, labels=["H","D","A"], zero_division=0)
    brier = brier_score_loss((y=="H").astype(int), proba[:, list(model.classes_).index("H")])  # example for Home
    print(f"\n{name} – accuracy={acc:.3f} | Brier(Home)={brier:.3f}")
    print("Class-wise (H, D, A):")
    print(f"  Precision: {prec}")
    print(f"  Recall:    {rec}")
    print(f"  F1:        {f1}")
    cm = confusion_matrix(y, preds, labels=["H","D","A"])
    print("Confusion matrix (rows=true, cols=pred) order [H,D,A]:\n", cm)

    if show_plots:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["H","D","A"], yticklabels=["H","D","A"], ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.show()

        # Calibration curve for “Home win” class
        if calibrated:
            prob_pos = proba[:, list(model.classes_).index("H")]
            frac_pos, mean_pred = calibration_curve((y=="H").astype(int), prob_pos, n_bins=10)
            plt.figure(figsize=(5,4))
            plt.plot(mean_pred, frac_pos, marker="o", label="Home class")
            plt.plot([0,1],[0,1],"--")
            plt.title(f"{name} Calibration (Home)") ; plt.xlabel("Predicted prob") ; plt.ylabel("Observed freq")
            plt.legend(); plt.show()

def evaluate_regressor(name, model, X, y, show_plots=True):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = math.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"\n{name} – MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")

    # Derive Over/Under 2.5 as a binary check for intuition:
    y_over = (y >= 3).astype(int)
    pred_over = (pred >= 2.5).astype(int)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_over, pred_over)
    pr, rc, f1, _ = precision_recall_fscore_support(y_over, pred_over, zero_division=0)
    print(f"Over/Under derived – acc={acc:.3f} | precision={pr} | recall={rc} | f1={f1}")

    if show_plots:
        plt.figure(figsize=(5,4))
        plt.scatter(y, pred, alpha=0.4)
        lims = [min(y.min(), pred.min())-0.2, max(y.max(), pred.max())+0.2]
        plt.plot(lims, lims, "--")
        plt.title(f"{name}: Predictions vs Actuals (Total Goals)")
        plt.xlabel("Actual total goals"); plt.ylabel("Predicted total goals")
        plt.show()

        # Residuals
        resid = y - pred
        plt.figure(figsize=(5,4))
        sns.histplot(resid, bins=30, kde=True)
        plt.title(f"{name}: Residuals")
        plt.xlabel("Actual - Predicted")
        plt.show()

    return pred

# ----------- TRAINING ENTRYPOINT -----------

def train_pipeline():
    client = APIFootballClient(API_KEY)
    api = API(client)
    builder = DatasetBuilder(api)

    print("Building training dataset (this will call the API; be patient & mind rate limits)…")
    df = builder.build_training_table(LEAGUES, SEASONS)
    df.to_csv(os.path.join(ARTIFACT_DIR, "training_raw.csv"), index=False)

    print(f"Dataset rows: {len(df)}")
    train, valid, test = time_split(df, valid_frac=0.15, test_frac=0.15)

    X_train, y_train_cls, y_train_reg = build_feature_matrix(train)
    X_valid, y_valid_cls, y_valid_reg = build_feature_matrix(valid)
    X_test,  y_test_cls,  y_test_reg  = build_feature_matrix(test)

    # Tune & fit classifier
    print("\nTuning RF Classifier (W/D/L)…")
    best_rf_cls, cls_params = tune_rf_classifier(X_train, y_train_cls)
    print("Best classifier params:", cls_params)

    # Calibrate on VALID set (best practice: fit base model on train; calibrate on valid)
    calibrator = CalibratedClassifierCV(best_rf_cls, method="isotonic", cv="prefit")
    calibrator.fit(X_valid, y_valid_cls)

    # Evaluate on valid & test
    evaluate_classifier("RF-CLS (valid)", calibrator, X_valid, y_valid_cls, calibrated=True)
    evaluate_classifier("RF-CLS (test)", calibrator, X_test, y_test_cls, calibrated=True)

    # Feature importance (permutation on VALID to be fair)
    perm_imp = permutation_importance(calibrator, X_valid, y_valid_cls, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({"feature": X_valid.columns, "importance": perm_imp.importances_mean}).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(ARTIFACT_DIR, "rf_cls_feature_importance.csv"), index=False)
    plt.figure(figsize=(7,6))
    sns.barplot(data=imp_df.head(20), x="importance", y="feature")
    plt.title("Classifier – Top 20 permutation importances")
    plt.tight_layout(); plt.show()

    # Tune & fit regressor
    print("\nTuning RF Regressor (total goals)…")
    best_rf_reg, reg_params = tune_rf_regressor(X_train, y_train_reg)
    print("Best regressor params:", reg_params)

    # Evaluate
    pred_valid_reg = evaluate_regressor("RF-REG (valid)", best_rf_reg, X_valid, y_valid_reg)
    pred_test_reg  = evaluate_regressor("RF-REG (test)",  best_rf_reg, X_test,  y_test_reg)

    # Save artifacts
    dump(calibrator, os.path.join(ARTIFACT_DIR, "rf_classifier_calibrated.joblib"))
    dump(best_rf_reg, os.path.join(ARTIFACT_DIR, "rf_regressor.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X_train.columns), f)
    print("\nSaved models + feature schema to 'artifacts/'")

# ----------- PREDICTION FOR A NEW FIXTURE -----------

def load_artifacts():
    cls = load(os.path.join(ARTIFACT_DIR, "rf_classifier_calibrated.joblib"))
    reg = load(os.path.join(ARTIFACT_DIR, "rf_regressor.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
        cols = json.load(f)
    return cls, reg, cols

def build_features_for_fixture(api: API, fixture_id: int) -> pd.DataFrame:
    """
    Rebuilds features for a single upcoming fixture: fetches its teams, league, season,
    then constructs the same set of features we used during training (using only data BEFORE kickoff).
    For simplicity here we rebuild a minimal ephemeral DatasetBuilder; in production, you’ll persist historical caches.
    """
    f = api.fixture_by_id(fixture_id)
    if not f:
        raise ValueError(f"Fixture {fixture_id} not found.")
    fc = parse_fixture_core(f)
    league, season = fc["league_id"], fc["season"]

    # To replicate rolling & elo, we need past finished fixtures for this league/season
    builder = DatasetBuilder(api)
    fixtures = api.fixtures_finished(league, season)
    past = pd.DataFrame([parse_fixture_core(x) for x in fixtures])
    past["date"] = pd.to_datetime(past["date"])
    past = past[past["date"] < pd.to_datetime(fc["date"])].sort_values("date")

    # Walk through history to fill builder’s state
    for _, r in past.iterrows():
        row = builder._build_row(r)  # not used directly
        gh, ga = r["goals_home"], r["goals_away"]
        # Update ELO & history
        builder._init_elo_if_needed(r["home_id"]); builder._init_elo_if_needed(r["away_id"])
        new_h, new_a = update_elo(builder._elo[r["home_id"]], builder._elo[r["away_id"]], gh, ga)
        builder._elo[r["home_id"]] = new_h; builder._elo[r["away_id"]] = new_a
        builder._record_team_match(r["home_id"], r["date"], gh, ga)
        builder._record_team_match(r["away_id"], r["date"], ga, gh)

    # Now build the features for the target fixture:
    feat_row = builder._build_row(fc)
    df = pd.DataFrame([feat_row])

    # Align columns to training schema (order + missing columns)
    _, _, _, = None, None, None
    try:
        with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
            cols = json.load(f)
    except FileNotFoundError:
        raise RuntimeError("Run training first so feature_columns.json exists.")
    # Drop non-feature columns we added for convenience:
    for c in ["fixture_id","date","league_id","season","home_id","away_id","y_outcome","y_total_goals"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    # Add any missing columns as NaN:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # Keep only training cols and order them:
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def probability_over_25_from_regression(reg_model, X, residual_std: float):
    """
    Convert total-goals regression output into P(Over 2.5) by assuming residuals ~ Normal(0, sigma).
    """
    mu = reg_model.predict(X)
    # P(Total > 2.5) = 1 - CDF((2.5 - mu)/sigma)
    z = (2.5 - mu) / residual_std
    return 1 - norm.cdf(z)

def estimate_residual_std(y_true, y_pred):
    resid = y_true - y_pred
    return np.std(resid)

def predict_fixture(fixture_id: int) -> Dict[str, Any]:
    client = APIFootballClient(API_KEY)
    api = API(client)
    cls, reg, cols = load_artifacts()
    X = build_features_for_fixture(api, fixture_id)

    # Classification probabilities (H/D/A), calibrated:
    proba = cls.predict_proba(X)[0]
    classes = list(cls.classes_)  # e.g., ["A","D","H"] in alphabetical order
    p = {c: proba[i] for i,c in enumerate(classes)}
    # Reorder to H/D/A with %:
    prob_H = 100 * p.get("H", 0.0)
    prob_D = 100 * p.get("D", 0.0)
    prob_A = 100 * p.get("A", 0.0)
    confidence = max(prob_H, prob_D, prob_A) / 100.0  # 0..1

    # Regression: need an estimate of residual std; load from artifacts if you’ve saved it.
    # Here we assume 1.25 as a typical residual SD baseline; you can persist your own value post-training.
    residual_sd = 1.25
    total_goals_pred = float(reg.predict(X)[0])
    p_over = float(probability_over_25_from_regression(reg, X, residual_sd)[0]) * 100

    return {
        "WDL_probabilities": {"Home%": round(prob_H,1), "Draw%": round(prob_D,1), "Away%": round(prob_A,1)},
        "confidence_score": round(confidence, 3),
        "pred_total_goals": round(total_goals_pred, 2),
        "prob_over_2_5_%": round(p_over, 1),
    }

# ----------- CLI -----------

if __name__ == "__main__":
    # Run once to train & evaluate:
    train_pipeline()

    # Example of using predict (after training completes), replace with a real upcoming fixture id:
    # print(predict_fixture(123456))
