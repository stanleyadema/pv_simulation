"""
ghi_fetcher_local.py

Fetch hourly Global Horizontal Irradiance (GHI) from NASA POWER, then convert
timestamps to the *real local timezone* of the chosen coordinates (e.g. Asia/Jakarta,
America/New_York). This avoids the common confusion where UTC data appears to
have sunlight at "night".

Key points
----------
- No API key required (NASA POWER).
- Input can be address/place name or "lat,lon".
- We fetch in UTC from the API, then convert to the IANA timezone at that lat/lon.
- Output DataFrame columns: timestamp, ALLSKY_SFC_SW_DWN (W/m^2).
- CLI included.

Install
-------
pip install pandas requests timezonefinder pytz

Example
-------
from ghi_fetcher_local import get_hourly_ghi_local, save_ghi_csv

df = get_hourly_ghi_local(
    location="New York, USA",   # or "-6.2,106.8" or (-6.2, 106.8)
    start_date="2024-01-01",
    end_date="2024-01-03"
)
save_ghi_csv(df, "nyc_ghi.csv")
"""

import re
import requests
import pandas as pd
from datetime import datetime
from typing import Tuple, Union, Optional

# Optional deps for timezone detection
try:
    from timezonefinder import TimezoneFinder  # pip install timezonefinder
    _HAS_TZ_FINDER = True
except Exception:
    TimezoneFinder = None
    _HAS_TZ_FINDER = False

try:
    import pytz  # pip install pytz
    _HAS_PYTZ = True
except Exception:
    pytz = None
    _HAS_PYTZ = False

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAM = "ALLSKY_SFC_SW_DWN"  # GHI (W/m^2)


def _parse_latlon(s: str):
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _geocode(address: str, user_agent: str = "ghi-fetcher-local") -> Tuple[float, float]:
    headers = {"User-Agent": user_agent}
    params = {"q": address, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError(f"Could not geocode address: {address}")
    return float(js[0]["lat"]), float(js[0]["lon"])


def _fmt_date(d: Union[str, datetime]) -> str:
    if isinstance(d, str):
        dt = datetime.strptime(d, "%Y-%m-%d")
    else:
        dt = d
    return dt.strftime("%Y%m%d")


def _power_url(lat: float, lon: float, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> str:
    # IMPORTANT: request UTC from POWER, we will convert to local zone ourselves
    start = _fmt_date(start_date)
    end = _fmt_date(end_date)
    return (
        f"{POWER_URL}?parameters={PARAM}"
        f"&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}"
        f"&time-standard=UTC"
        f"&format=JSON"
    )


def _detect_timezone_name(lat: float, lon: float) -> Optional[str]:
    if not _HAS_TZ_FINDER:
        return None
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    return tz_name


def _convert_utc_to_local(df: pd.DataFrame, tz_name: Optional[str]) -> pd.DataFrame:
    if tz_name and _HAS_PYTZ:
        tz = pytz.timezone(tz_name)
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"])
            .dt.tz_localize("UTC")
            .dt.tz_convert(tz)
            .dt.tz_localize(None)
        )
        return df
    else:
        # Fallback: keep UTC if timezone cannot be determined
        return df


def get_hourly_ghi_local(
    location: Union[str, Tuple[float, float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    user_agent: str = "ghi-fetcher-local"
) -> pd.DataFrame:
    """
    Fetch hourly GHI in UTC from NASA POWER, then convert timestamps to the
    detected local timezone for the given coordinates. If timezone detection
    libraries are missing, the timestamps remain in UTC.
    """
    if isinstance(location, tuple) and len(location) == 2:
        lat, lon = float(location[0]), float(location[1])
    elif isinstance(location, str):
        maybe = _parse_latlon(location)
        if maybe is not None:
            lat, lon = maybe
        else:
            lat, lon = _geocode(location, user_agent=user_agent)
    else:
        raise ValueError("`location` must be an address string or a (lat, lon) tuple.")

    url = _power_url(lat, lon, start_date, end_date)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()

    param = payload.get("properties", {}).get("parameter", {}).get(PARAM, {})
    if not param:
        raise ValueError("No GHI data returned by NASA POWER for the requested period/location.")

    rows = []
    for k, v in sorted(param.items()):
        if not re.match(r"^\d{10}$", k):
            continue
        ts = datetime.strptime(k, "%Y%m%d%H")  # naive UTC
        rows.append({"timestamp": ts, PARAM: v})

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    tz_name = _detect_timezone_name(lat, lon)
    df = _convert_utc_to_local(df, tz_name)

    df.attrs["lat"] = lat
    df.attrs["lon"] = lon
    df.attrs["timezone"] = tz_name or "UTC"

    return df


def save_ghi_csv(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch hourly GHI and convert to local timezone of the location.")
    parser.add_argument("--location", required=True, help='Address/place or "lat,lon" (e.g., "-6.2,106.8")')
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", default="ghi_local.csv", help="Output CSV path")
    args = parser.parse_args()

    df = get_hourly_ghi_local(args.location, args.start, args.end)
    save_ghi_csv(df, args.out)
    print(df.head())
