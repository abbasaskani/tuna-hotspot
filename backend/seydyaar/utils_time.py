from __future__ import annotations
import datetime as dt
import json
from typing import Tuple, Optional
import requests

def trusted_utc_now(timeout_s: float = 3.5) -> Tuple[dt.datetime, str]:
    """
    Tries to fetch a trusted UTC time from public time APIs; falls back to local system UTC.

    Returns:
      (utc_dt, source)
    """
    # Keep it robust: try 2 sources quickly, then fallback.
    sources = [
        ("worldtimeapi", "https://worldtimeapi.org/api/timezone/Etc/UTC"),
        ("timeapi", "https://timeapi.io/api/Time/current/zone?timeZone=UTC"),
    ]
    for name, url in sources:
        try:
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            if name == "worldtimeapi":
                iso = data.get("datetime")
            else:
                # timeapi returns: {"dateTime":"2026-02-10T...","timeZone":"UTC",...}
                iso = data.get("dateTime")
            if iso:
                # normalize Z
                iso = iso.replace("Z", "+00:00")
                return dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc), name
        except Exception:
            continue

    return dt.datetime.now(dt.timezone.utc), "local_system_fallback"
