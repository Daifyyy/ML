from __future__ import annotations
from typing import Any, Dict, Optional, Iterable
import time, math
import requests
from requests.adapters import HTTPAdapter, Retry

class APIFootballClient:
    """
    Resilient API client:
    - adds headers
    - handles retries/backoff
    - respects pagination via response['paging']
    - simple rate limiting (sleep between calls)
    """
    def __init__(
        self,
        api_key: str,
        api_base: str,
        min_ms_between_calls: int = 100,
        max_requests_per_minute: int = 55,
        timeout: int = 30,
    ) -> None:
        self.base = api_base.rstrip("/")
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
        self.min_interval = max(0.0, min_ms_between_calls / 1000.0)
        self.max_rpm = max_requests_per_minute
        self._last_call_ts = 0.0

    def _respect_rate_limit(self):
        now = time.monotonic()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_ts = time.monotonic()

    def get_page(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._respect_rate_limit()
        url = f"{self.base}/{path.lstrip('/')}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            try:
                payload = r.json()
            except Exception:
                payload = {"message": r.text[:200]}
            raise RuntimeError(f"API error {r.status_code} for {url} {params} -> {payload}")
        return r.json()

    def get_all_pages(self, path: str, params: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        """
        Iterates over pages until done. API-Football returns:
          { 'paging': {'current': 1, 'total': 3}, 'response': [ ... ] }
        """
        params = dict(params or {})
        page = 1
        while True:
            params["page"] = page
            data = self.get_page(path, params)
            response = data.get("response", [])
            for item in response:
                yield item
            paging = data.get("paging", {})
            cur, tot = int(paging.get("current", page)), int(paging.get("total", page))
            if cur >= tot or tot == 0:
                break
            page += 1
