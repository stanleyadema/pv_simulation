# app.py — Solar Week Overview (aligned charts + summary + clean inputs)

from __future__ import annotations

import io
from dataclasses import dataclass
import base64
from datetime import datetime, time, timedelta, date
from typing import List, Optional, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ghi_fetcher_local import get_hourly_ghi_local


# ---------- Page & constants ----------
st.set_page_config(page_title="Simulation", layout="wide", page_icon="☀️")

BASE_HEIGHT   = 150
ENERGY_HEIGHT = BASE_HEIGHT * 2         # Source/Enjoy is 2x
SOC_HEIGHT    = BASE_HEIGHT
SPILL_HEIGHT  = BASE_HEIGHT

BAR_WIDTH = 0.85                         # same thickness across all charts
BAR_GAP   = 0.15
Y_ZOOM_PCT = 120

DAY_WIDTH = 220                          # fixed width per day to keep alignment
FIG_WIDTH = 7 * DAY_WIDTH

# Colors
COLOR_PV     = "#28a745"
COLOR_BATT   = "#1f77b4"
COLOR_DIESEL = "#000000"
COLOR_LOAD   = "#d62728"
COLOR_SPILL  = "#7cd992"
COLOR_CHART_BLACK = "#000000"
OVERVIEW_WEEK_HEIGHT = 220
OVERVIEW_WEEK_HEIGHT_OTHER = 120

def _render_week_row(layout_ratio=None):
    ratio = layout_ratio or [0.8, 4, 1]
    return st.columns(ratio, gap="small")
NASA_SOURCE  = "NASA GHI"
IRRADIANCE_COL = "irradiance_wm2"
GENERATOR_PALETTE = [
    "#000000", "#1f77b4", "#bcbd22", "#ff7f0e",
    "#9467bd", "#2ca02c", "#d2b48c", "#17becf"
]
OVERVIEW_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def generator_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        colors.append(GENERATOR_PALETTE[i % len(GENERATOR_PALETTE)])
    return colors
VALUE_COLS = ["load_kWh", "harvest_kWh", "harvest_used_kWh",
              "batt_discharge_kWh", "batt_charge_kWh", "diesel_kWh", "pv_spill_kWh"]

SAMPLE_DL_CSS = """
<style>
.sample-download {
    font-size: 12px;
    text-decoration: underline;
    color: #0F75BD;
    cursor: pointer;
    display: inline-block;
    margin-top: -2px;
}
.upload-title, .upload-title-bold {
    font-weight: 400;
    margin-bottom: 0px;
    margin-top: 8px;
}
.upload-title-bold {
    font-weight: 600;
    margin-top: 4px;
}
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0 0 4px 0;
    }
</style>
"""

OVERVIEW_CSS = """
<style>
.pv-overview-week-wrap {
    width: 70px;
    display: flex;
    justify-content: flex-end;
    padding-right: 8px;
}
.pv-overview-week-wrap.chart {
    height: 220px;
    align-items: center;
}
.pv-overview-week-wrap.chart-small {
    height: 120px;
    align-items: center;
}
.pv-overview-week-wrap.avg {
    min-height: 80px;
    align-items: center;
}
.pv-overview-week-label {
    font-size: 20px;
    font-weight: 700;
    text-align: right;
    margin: 0;
}
.pv-overview-week-label-block {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}
.pv-overview-week-avg {
    font-size: 16px;
    font-weight: 600;
    color: #1b8e3e;
}
.pv-overview-week-avg.muted {
    color: #999;
}
.pv-overview-day-strip {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    font-weight: 600;
    color: #777;
    margin-bottom: 4px;
    text-align: center;
}
.pv-overview-day-values {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    margin-top: 4px;
}
.pv-overview-day-value {
    text-align: left;
    padding-left: 4px;
}
.pv-overview-day-header {
    text-align: center;
    font-weight: 600;
    margin-bottom: 4px;
}
.pv-overview-day-top {
    text-align: center;
    color: #1b8e3e;
    font-weight: 600;
    font-size: 15px;
    margin-bottom: -8px;
}
.pv-overview-day-top.muted {
    color: #999999;
}
.pv-overview-day-bottom {
    text-align: center;
    color: #b71c1c;
    font-weight: 600;
    font-size: 15px;
    margin-top: -6px;
}
.pv-overview-week-summary {
    text-align: left;
    font-size: 16px;
    line-height: 1.25;
}
.pv-overview-week-summary.week {
    height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.pv-overview-week-summary .pv-spill-line {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    min-height: 0;
    margin-bottom: 4px;
}
.pv-overview-week-summary.week-small {
    height: 120px;
}
.pv-overview-week-summary .pv-green {
    color: #1b8e3e;
    font-weight: 700;
}
.pv-overview-week-summary .pv-yellow {
    color: #f6c343;
    font-weight: 700;
}
.pv-overview-week-summary .pv-red {
    color: #b71c1c;
    font-weight: 700;
}
.pv-green {
    color: #1b8e3e;
    font-weight: 700;
}
.pv-red {
    color: #b71c1c;
    font-weight: 700;
}
.week-total-text {
    font-size: 20px;
    font-weight: 700;
}
.pv-overview-week-summary small {
    display: block;
    font-weight: 500;
    color: #555;
}
.pv-overview-divider {
    border-top: 1px solid #e0e0e0;
    margin: 18px 0 12px 0;
}
.pv-overview-avg-label {
    font-weight: 700;
    text-align: right;
    font-size: 18px;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
}
.pv-overview-avg-pct {
    font-size: 12px;
    display: block;
}
</style>
"""

def _upload_control(label: str, state_key: str, *, types: Optional[Iterable[str]] = None) -> Optional[dict]:
    """Persist uploaded file bytes in session state and hide uploader once loaded."""
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    file_state = st.session_state[state_key]

    if not file_state:
        file = st.file_uploader(label, type=types, key=f"{state_key}_uploader")
        if file is not None:
            st.session_state[state_key] = {"name": file.name, "data": file.getvalue()}
            st.rerun()
    else:
        st.success(f"{label} loaded: {file_state['name']}")
        if st.button(f"Remove {label}", key=f"{state_key}_remove"):
            st.session_state[state_key] = None
            st.rerun()
    return st.session_state[state_key]


def _small_download_link(label: str, data_bytes: bytes, file_name: str, key: str):
    b64 = base64.b64encode(data_bytes).decode()
    st.markdown(
        f'<a class="sample-download" id="{key}" download="{file_name}" href="data:application/octet-stream;base64,{b64}">{label}</a>',
        unsafe_allow_html=True
    )


@st.cache_data
def _sample_load_excel() -> bytes:
    dt = pd.date_range("2024-01-01 00:00", periods=5, freq="H")
    df = pd.DataFrame({"Timestamp": dt, "Total": [320, 310, 305, 315, 300]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Overview")
    return buf.getvalue()


@st.cache_data
def _sample_harvest_excel() -> bytes:
    dt = pd.date_range("2024-01-01 00:00", periods=5, freq="H")
    df = pd.DataFrame({"Timestamp": dt, "Harvest": [210, 220, 250, 200, 180]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


@st.cache_data
def _sample_detailed_excel() -> bytes:
    dt = pd.date_range("2024-01-01 00:00", periods=5, freq="H")
    df = pd.DataFrame({
        "Timestamp": dt,
        "G1": [120, 110, 115, 118, 112],
        "G2": [80, 85, 82, 78, 81],
        "G3": [60, 62, 58, 64, 59],
        "Total": [260, 257, 255, 260, 252],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ---------- Helpers ----------
def detect_ts_column(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "time", "datetime", "date", "dt", "waktu"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    return df.columns[0]


def detect_named_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return df.columns[0]


def nav_buttons(key_prefix: str, prev_label: str, next_label: str,
                prev_disabled: bool, next_disabled: bool) -> tuple[bool, bool]:
    spacer_left, prev_col, next_col, spacer_right = st.columns([2, 1, 1, 2])
    with prev_col:
        prev_clicked = st.button(prev_label, use_container_width=True,
                                 disabled=prev_disabled, key=f"{key_prefix}_prev")
    with next_col:
        next_clicked = st.button(next_label, use_container_width=True,
                                 disabled=next_disabled, key=f"{key_prefix}_next")
    return prev_clicked, next_clicked


def add_calendar_columns(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = out.sort_values(ts_col).reset_index(drop=True)
    iso = out[ts_col].dt.isocalendar()
    out["year"] = iso["year"].astype(int)
    out["week"] = iso["week"].astype(int)
    out["month"] = out[ts_col].dt.month
    out["date"] = out[ts_col].dt.date
    out["hod"]  = out[ts_col].dt.hour
    out["dow"]  = out[ts_col].dt.dayofweek
    return out


def figure_out_total(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() == "total":
            return c
    gcols = [c for c in df.columns if c.strip().lower() in ("g1","g2","g3","g4","g5")]
    if gcols:
        df["Total"] = df[gcols].sum(axis=1)
        return "Total"
    raise ValueError("No 'Total' or G1..G5 columns found in the Overview sheet.")


def week_list(df: pd.DataFrame) -> List[tuple[int,int]]:
    uniq = df.drop_duplicates(["year","week"]).sort_values(["year","week"])
    return [(int(y), int(w)) for y, w in uniq[["year","week"]].to_records(index=False)]


def month_list(df: pd.DataFrame) -> List[tuple[int,int]]:
    uniq = df.drop_duplicates(["year","month"]).sort_values(["year","month"])
    return [(int(y), int(m)) for y, m in uniq[["year","month"]].to_records(index=False)]


def day_list(df: pd.DataFrame, ts_col: str) -> List[datetime.date]:
    days = pd.to_datetime(df[ts_col]).dt.date
    uniq = pd.Series(days).drop_duplicates().sort_values()
    return uniq.tolist()


def year_list(df: pd.DataFrame) -> List[int]:
    years = df["year"].dropna().astype(int).sort_values().unique()
    return years.tolist()


def iso_week_monday(iso_year:int, iso_week:int) -> date:
    jan4 = date(iso_year, 1, 4)
    delta = timedelta(days=jan4.isoweekday()-1)
    week1_monday = jan4 - delta
    return week1_monday + timedelta(weeks=iso_week-1)


def build_week_index(iso_year:int, iso_week:int) -> pd.DatetimeIndex:
    monday = iso_week_monday(iso_year, iso_week)
    start = datetime.combine(monday, time(0,0))
    end = start + timedelta(days=7, hours=-1)      # inclusive Sun 23:00
    return pd.date_range(start, end, freq="H")


def build_month_index(year:int, month:int) -> pd.DatetimeIndex:
    start = datetime(year, month, 1, 0, 0)
    if month == 12:
        next_month = datetime(year + 1, 1, 1, 0, 0)
    else:
        next_month = datetime(year, month + 1, 1, 0, 0)
    end = next_month - timedelta(hours=1)
    return pd.date_range(start, end, freq="H")


def build_day_index(day:date) -> pd.DatetimeIndex:
    start = datetime.combine(day, time(0,0))
    end = start + timedelta(hours=23)
    return pd.date_range(start, end, freq="H")


def pv_weights_24_cosine(start_h: float, end_h: float) -> np.ndarray:
    w = np.zeros(24, dtype=float)
    s = int(np.floor(start_h)); e = int(np.ceil(end_h))
    e = max(e, s)
    idx = np.arange(s, e+1)
    if len(idx) == 0:
        return w
    x = np.linspace(0, np.pi, len(idx))
    vals = np.sin(x) ** 2
    if vals.sum() > 0:
        w[idx] = vals / vals.sum()
    return w


@st.cache_data(show_spinner=False)
def read_overview_from_excel(file_bytes: bytes) -> pd.DataFrame:
    with io.BytesIO(file_bytes) as bio:
        xls = pd.ExcelFile(bio)
        if not xls.sheet_names:
            raise ValueError("The provided file does not contain any sheets.")
        overview_sheet = None
        for sheet in xls.sheet_names:
            if sheet.strip().lower() == "overview":
                overview_sheet = sheet
                break
        sheet_to_use = overview_sheet or xls.sheet_names[0]
        df = xls.parse(sheet_to_use)
    return df


@st.cache_data(show_spinner=False)
def read_harvest_from_excel(file_bytes: bytes) -> pd.DataFrame:
    with io.BytesIO(file_bytes) as bio:
        df = pd.read_excel(bio)
    return df


@dataclass
class Inputs:
    pv_kwp: float
    pv_hr: float
    sunrise_h: float
    sunset_h: float
    batt_mwh: float
    soc_max_pct: int
    soc_min_pct: int
    init_soc_pct: int
    rt_eff: float


def to_hour_float(t: time) -> float:
    return t.hour + t.minute/60.0


def simulate_dispatch(overview: pd.DataFrame, ts_col: str, inp: Inputs,
                      harvest_override: Optional[np.ndarray] = None) -> tuple[pd.DataFrame, float]:
    df = overview.copy()
    total_col = figure_out_total(df)
    if harvest_override is not None and len(harvest_override) != len(df):
        raise ValueError("Harvest override length must match overview length.")

    weights24 = pv_weights_24_cosine(inp.sunrise_h, inp.sunset_h)
    pv_per_day_kwh = inp.pv_kwp * inp.pv_hr
    if harvest_override is not None:
        harvest = harvest_override.astype(float)
    else:
        harvest = pv_per_day_kwh * np.array([weights24[h] for h in df["hod"].to_numpy()])

    cap_kwh = max(0.0, inp.batt_mwh) * 1000.0
    soc_max = (inp.soc_max_pct/100.0) * cap_kwh
    soc_min = (inp.soc_min_pct/100.0) * cap_kwh
    soc_min = min(soc_min, soc_max)

    eta_rt = float(np.clip(inp.rt_eff, 0, 1))
    eta_c = eta_d = np.sqrt(eta_rt) if eta_rt > 0 else 0.0

    init_soc = (inp.init_soc_pct/100.0) * cap_kwh
    soc = min(max(init_soc, soc_min), soc_max) if cap_kwh > 0 else 0.0

    total = df[total_col].astype(float).to_numpy()
    harvest_used = np.zeros_like(total, dtype=float)
    batt_dis_to_load = np.zeros_like(total, dtype=float)
    batt_charge = np.zeros_like(total, dtype=float)
    diesel = np.zeros_like(total, dtype=float)
    pv_spill = np.zeros_like(total, dtype=float)
    soc_path = np.zeros_like(total, dtype=float)

    for i in range(len(df)):
        L = float(total[i])
        H = float(harvest[i])

        used = min(H, L)
        harvest_used[i] = used
        rem = L - used
        excess = H - used

        if rem > 0 and cap_kwh > 0 and soc > soc_min and eta_d > 0:
            deliverable = (soc - soc_min) * eta_d
            d_used = min(rem, deliverable)
            batt_dis_to_load[i] = d_used
            soc -= d_used / eta_d
            rem -= d_used

        if rem > 0:
            diesel[i] = rem
            rem = 0.0

        if excess > 0 and cap_kwh > 0 and soc < soc_max and eta_c > 0:
            space = soc_max - soc
            pv_to_batt = min(excess, space / eta_c)
            stored = pv_to_batt * eta_c
            batt_charge[i] = stored
            soc += stored
            pv_spill[i] = excess - pv_to_batt
        else:
            pv_spill[i] = excess

        soc_path[i] = soc

    out = df.copy()
    out["load_kWh"]            = np.round(total, 1)
    out["harvest_kWh"]         = np.round(harvest, 1)
    out["harvest_used_kWh"]    = np.round(harvest_used, 1)
    out["batt_discharge_kWh"]  = np.round(batt_dis_to_load, 1)
    out["batt_charge_kWh"]     = np.round(batt_charge, 1)
    out["diesel_kWh"]          = np.round(diesel, 1)
    out["pv_spill_kWh"]        = np.round(pv_spill, 1)
    out["soc_kWh"]             = np.round(soc_path, 1)
    out["soc_pct"]             = np.round(100.0 * out["soc_kWh"] / cap_kwh, 1) if cap_kwh > 0 else 0.0
    return out, cap_kwh


def compute_energy_ymax(sim_w: pd.DataFrame) -> float:
    pos_peak = float((sim_w["harvest_used_kWh"] + sim_w["batt_discharge_kWh"] + sim_w["diesel_kWh"]).max())
    neg_peak = float(sim_w["load_kWh"].max())
    auto_y = max(100.0, np.ceil(max(pos_peak, neg_peak) / 50.0) * 50.0)
    return auto_y * (Y_ZOOM_PCT / 100.0)


def _day_boundaries_and_labels(ts: pd.Series) -> tuple[list[int], list[str]]:
    ts = pd.to_datetime(ts).reset_index(drop=True)
    norm_days = ts.dt.normalize()
    starts = norm_days.drop_duplicates(keep="first").index.to_list()
    labels = ts.dt.strftime("%a").iloc[starts].tolist()
    return starts, labels


def unified_energy_figure(sim_w: pd.DataFrame, ts_col: str, ymax: float,
                          *, show_day_lines: bool = True, show_day_labels: bool = False,
                          hover_style: str = "auto",  # retained for backward compatibility
                          day_hr_map: Optional[dict[pd.Timestamp, float]] = None,
                          hr_values: Optional[np.ndarray] = None,
                          extra_hr: Optional[np.ndarray] = None,
                          show_night: bool = False,
                          include_time: bool = True) -> go.Figure:
    fig = go.Figure()
    n = len(sim_w)
    x = np.arange(n)

    ts = pd.to_datetime(sim_w[ts_col]).reset_index(drop=True)
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    pv_vals     = sim_w["harvest_used_kWh"].to_numpy()
    batt_vals   = sim_w["batt_discharge_kWh"].to_numpy()
    diesel_vals = sim_w["diesel_kWh"].to_numpy()
    load_vals   = sim_w["load_kWh"].to_numpy()

    pv_hover     = np.rint(pv_vals).astype(int)
    batt_hover   = np.rint(batt_vals).astype(int)
    diesel_hover = np.rint(diesel_vals).astype(int)
    load_hover   = np.rint(np.abs(load_vals)).astype(int)

    hr_display = None
    if hr_values is not None:
        hr_display = np.array([f"{val:.1f}" for val in np.round(hr_values, 1)])
    elif extra_hr is not None:
        hr_display = np.array([f"{val:.1f}" for val in np.round(extra_hr, 1)])

    def build_custom(values):
        cols = [values]
        if include_time:
            cols.extend([time_str, date_str])
        else:
            cols.append(date_str)
        if hr_display is not None:
            cols.append(hr_display)
        return np.stack(cols, axis=1)

    if include_time:
        hover_template = "%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}"
        hr_idx = 3
    else:
        hover_template = "%{customdata[0]} kWh, %{customdata[1]}"
        hr_idx = 2
    if hr_display is not None:
        hover_template += f", %{{customdata[{hr_idx}]}} HR"
    hover_template += "<extra></extra>"

    fig.add_bar(x=x, y=pv_vals,   width=BAR_WIDTH, marker_color=COLOR_PV,     showlegend=False,
                customdata=build_custom(pv_hover),   hovertemplate=hover_template)
    fig.add_bar(x=x, y=batt_vals, width=BAR_WIDTH, marker_color=COLOR_BATT,   showlegend=False,
                customdata=build_custom(batt_hover), hovertemplate=hover_template)
    fig.add_bar(x=x, y=diesel_vals, width=BAR_WIDTH, marker_color=COLOR_DIESEL, showlegend=False,
                customdata=build_custom(diesel_hover), hovertemplate=hover_template)
    # plot negative but show positive value
    fig.add_bar(x=x, y=-load_vals, width=BAR_WIDTH, marker_color=COLOR_LOAD,  showlegend=False,
                customdata=build_custom(load_hover), hovertemplate=hover_template)

    day_starts = day_labels = None
    if show_day_lines or show_day_labels:
        day_starts, day_labels = _day_boundaries_and_labels(sim_w[ts_col])

    if show_day_lines and day_starts is not None:
        for i, start_idx in enumerate(day_starts):
            if i > 0:
                pos = start_idx - 0.5
                fig.add_vline(x=pos, line_width=1, line_dash="dash", line_color="lightgray")

    if show_day_labels and day_starts is not None and day_labels is not None:
        for i, start_idx in enumerate(day_starts):
            next_idx = day_starts[i+1] if i+1 < len(day_starts) else len(sim_w)
            mid_x = start_idx + (next_idx - start_idx) / 2
            fig.add_annotation(
                x=mid_x,
                y=ymax * 0.96,
                text=f"<b>{day_labels[i]}</b>",
                showarrow=False,
                font=dict(size=12, color="#444"),
                xref="x",
                yref="y"
            )

    if day_hr_map and day_starts is not None and day_labels is not None:
        day_dates = ts.iloc[day_starts].dt.normalize()
        for i, start_idx in enumerate(day_starts):
            day_key = day_dates.iloc[i].strftime("%Y-%m-%d")
            if day_key not in day_hr_map:
                continue
            next_idx = day_starts[i+1] if i+1 < len(day_starts) else len(sim_w)
            mid_x = start_idx + (next_idx - start_idx) / 2
            hr_val = day_hr_map[day_key]
            fig.add_annotation(
                x=mid_x,
                y=ymax * 0.85,
                text=f"{hr_val:.1f} HR",
                showarrow=False,
                font=dict(size=11, color="#111"),
                xref="x",
                yref="y",
            )

    if show_night:
        for start_idx, end_idx in _night_segments(ts):
            x0 = start_idx - 0.5
            x1 = end_idx + 0.5
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(0,0,0,0.25)",
                opacity=0.35,
                line_width=0,
                layer="below",
                yref="y",
                y0=-ymax,
                y1=ymax,
            )

    fig.update_layout(
        barmode="relative",
        height=ENERGY_HEIGHT,
        width=FIG_WIDTH,
        margin=dict(l=10, r=10, t=64, b=20),
        bargap=BAR_GAP, bargroupgap=0.02, showlegend=False,
    )
    fig.update_yaxes(range=[-ymax, ymax], showticklabels=False, zeroline=True,
                     zerolinecolor="lightgray", zerolinewidth=1)
    fig.update_xaxes(showticklabels=False)

    return fig


def _format_mwh(value_kwh: float) -> str:
    return f"{value_kwh / 1000.0:.2f}"


def _week_dataframe(sim_all: pd.DataFrame, ts_col: str, year_sel: int, week_sel: int) -> pd.DataFrame:
    full_index = build_week_index(year_sel, week_sel)
    week_df = sim_all[(sim_all["year"] == year_sel) & (sim_all["week"] == week_sel)].copy()
    if ts_col not in week_df.columns:
        week_df[ts_col] = pd.to_datetime(week_df.index)
    week_df = week_df.set_index(pd.to_datetime(week_df[ts_col])).sort_index()
    week_full = pd.DataFrame(index=full_index)
    for col in VALUE_COLS:
        if col in week_df.columns:
            week_full[col] = week_df[col].reindex(full_index).fillna(0.0)
        else:
            week_full[col] = 0.0
    if IRRADIANCE_COL in week_df.columns:
        week_full[IRRADIANCE_COL] = week_df[IRRADIANCE_COL].reindex(full_index).fillna(0.0)
    if "soc_pct" in week_df.columns:
        week_full["soc_pct"] = week_df["soc_pct"].reindex(full_index).ffill().fillna(0.0)
    else:
        week_full["soc_pct"] = 0.0
    week_full[ts_col] = week_full.index
    return week_full


def _overview_day_chart_summary(day_df: pd.DataFrame, ts_index: pd.DatetimeIndex, include_pv: bool, ymax: float) -> go.Figure:
    fig = go.Figure()
    x = np.arange(len(day_df))
    load_vals = day_df["load_kWh"].to_numpy()
    time_str = pd.Index(ts_index).strftime("%H:%M")
    date_str = pd.Index(ts_index).strftime("%d %b")

    def tooltip_data(values: np.ndarray) -> np.ndarray:
        clean_vals = np.rint(np.abs(values)).astype(int)
        return np.stack([clean_vals, time_str, date_str], axis=1)

    positive_base = load_vals.copy()
    if include_pv:
        pv_vals = np.minimum(day_df["harvest_used_kWh"].to_numpy(), load_vals)
        remainder = np.clip(load_vals - pv_vals, 0.0, None)
        fig.add_bar(
            x=x,
            y=pv_vals,
            width=0.9,
            marker_color=COLOR_PV,
            showlegend=False,
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
            customdata=tooltip_data(pv_vals),
            name="PV Direct",
        )
        positive_base = remainder
    fig.add_bar(
        x=x,
        y=positive_base,
        width=0.9,
        marker_color=COLOR_CHART_BLACK,
        showlegend=False,
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
        customdata=tooltip_data(positive_base),
        name="Load fulfilled",
    )
    neg_vals = -load_vals
    fig.add_bar(
        x=x,
        y=neg_vals,
        width=0.9,
        marker_color=COLOR_LOAD,
        showlegend=False,
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
        customdata=tooltip_data(load_vals),
        name="Load demand",
    )
    fig.update_layout(
        barmode="relative",
        height=120,
        margin=dict(l=2, r=2, t=2, b=2),
        bargap=0.01,
        showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(range=[-ymax, ymax], visible=False)
    return fig


def _overview_ymax(week_df: pd.DataFrame, include_pv: bool) -> float:
    load_peak = float(week_df["load_kWh"].max()) if "load_kWh" in week_df else 0.0
    base = max(load_peak, 1.0)
    return np.ceil(base / 25.0) * 25.0


def _overview_irradiance_ymax(df: pd.DataFrame) -> float:
    if IRRADIANCE_COL not in df.columns:
        return 100.0
    vals = df[IRRADIANCE_COL].to_numpy(dtype=float)
    peak = float(np.nanmax(vals)) if len(vals) else 100.0
    if not np.isfinite(peak) or peak <= 0:
        peak = 100.0
    return np.ceil(peak / 100.0) * 100.0


def soc_bar_figure(sim_w: pd.DataFrame, ts_col: str, soc_min: int, soc_max: int) -> go.Figure:
    ts = pd.to_datetime(sim_w[ts_col]).reset_index(drop=True)
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    vals = sim_w["soc_pct"].to_numpy()
    hover = "%{customdata[0]} %, %{customdata[1]}, %{customdata[2]}<extra></extra>"
    fig = go.Figure()
    fig.add_bar(x=np.arange(len(sim_w)), y=vals, width=BAR_WIDTH, marker_color=COLOR_BATT, showlegend=False,
                customdata=np.stack([vals, time_str, date_str], axis=1), hovertemplate=hover)
    fig.add_hline(y=soc_max, line=dict(color="green", width=1, dash="dash"))
    fig.add_hline(y=soc_min, line=dict(color="red", width=1, dash="dash"))
    fig.update_layout(height=SOC_HEIGHT, width=FIG_WIDTH, margin=dict(l=10, r=10, t=10, b=20), showlegend=False)
    fig.update_yaxes(range=[0, 100], showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig


def spill_bar_figure(sim_w: pd.DataFrame, ts_col: str, ymax_from_energy: float, *,
                     timeframe: str | None = None) -> go.Figure:
    ts = pd.to_datetime(sim_w[ts_col]).reset_index(drop=True)
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    day_str = ts.dt.strftime("%a")
    vals = sim_w["pv_spill_kWh"].to_numpy()
    hover_vals = np.rint(vals).astype(int)
    if timeframe in ("Month", "Week"):
        hover = "%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>"
        custom = np.stack([hover_vals, day_str, date_str], axis=1)
    elif timeframe == "Year":
        month_str = ts.dt.strftime("%b")
        hover = "%{customdata[0]} kWh, %{customdata[1]}<extra></extra>"
        custom = np.stack([hover_vals, month_str], axis=1)
    else:
        hover = "%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>"
        custom = np.stack([hover_vals, day_str, ", " + date_str], axis=1)
    fig = go.Figure()
    fig.add_bar(x=np.arange(len(sim_w)), y=vals, width=BAR_WIDTH, marker_color=COLOR_SPILL, showlegend=False,
                customdata=custom, hovertemplate=hover)
    fig.update_layout(height=SPILL_HEIGHT, width=FIG_WIDTH, margin=dict(l=10, r=10, t=10, b=20), showlegend=False)
    # Same Y scale as the source chart
    fig.update_yaxes(range=[0, ymax_from_energy], showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig


def compute_detail_ymax(detail_df: pd.DataFrame, g_cols: list[str], total_col: str) -> float:
    if not g_cols:
        peak = float(detail_df[total_col].max())
    else:
        peak_sources = float(detail_df[g_cols].sum(axis=1).max())
        peak_total = float(detail_df[total_col].max())
        peak = max(peak_sources, peak_total)
    peak = max(peak, 1.0)
    auto_y = max(100.0, np.ceil(peak / 50.0) * 50.0)
    return auto_y * (Y_ZOOM_PCT / 100.0)


def detail_energy_figure(detail_df: pd.DataFrame, ts_col: str, g_cols: list[str], total_col: str,
                         use_day_hover: bool = False) -> go.Figure:
    ymax = compute_detail_ymax(detail_df, g_cols, total_col)
    fig = go.Figure()
    ts = pd.to_datetime(detail_df[ts_col]).reset_index(drop=True)
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    day_str = ts.dt.strftime("%a")
    month_str = ts.dt.strftime("%b")
    hover_primary = month_str if use_day_hover == "month" else day_str if use_day_hover else time_str
    if use_day_hover == "month":
        hover_primary = month_str
        hover_secondary = np.full(len(detail_df), "", dtype=object)
    elif use_day_hover:
        hover_primary = day_str + ","
        hover_secondary = " " + date_str
    else:
        hover_secondary = ", " + date_str
    x = np.arange(len(detail_df))
    colors = generator_colors(len(g_cols))
    for idx, col in enumerate(g_cols):
        vals = detail_df[col].to_numpy()
        hover_vals = np.rint(vals).astype(int)
        color = colors[idx]
        fig.add_bar(
            x=x,
            y=vals,
            width=BAR_WIDTH,
            marker_color=color,
            name=col,
            customdata=np.stack([hover_vals, hover_primary, hover_secondary], axis=1),
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}%{customdata[2]}<extra></extra>"
        )

    total_vals = -detail_df[total_col].to_numpy()
    total_hover = np.rint(np.abs(total_vals)).astype(int)
    fig.add_bar(
        x=x,
        y=total_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_LOAD,
        name="Total",
        customdata=np.stack([total_hover, hover_primary, hover_secondary], axis=1),
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}%{customdata[2]}<extra></extra>"
    )

    starts, labels = _day_boundaries_and_labels(detail_df[ts_col])
    date_range_span = (pd.to_datetime(detail_df[ts_col]).max() - pd.to_datetime(detail_df[ts_col]).min()).days + 1
    show_day_labels = date_range_span <= 14
    for i, start_idx in enumerate(starts):
        if show_day_labels and i > 0:
            fig.add_vline(x=start_idx, line_width=1, line_dash="dash", line_color="lightgray")
        if show_day_labels:
            next_idx = starts[i+1] if i+1 < len(starts) else len(detail_df)
            mid_x = start_idx + (next_idx - start_idx) / 2
            fig.add_annotation(
                x=mid_x,
                y=ymax * 0.92,
                text=labels[i],
                showarrow=False,
                font=dict(size=12, color="#444"),
                xref="x",
                yref="y"
            )

    fig.update_layout(
        barmode="relative",
        height=ENERGY_HEIGHT,
        width=FIG_WIDTH,
        margin=dict(l=10, r=10, t=64, b=20),
        bargap=BAR_GAP,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(range=[-ymax, ymax], showticklabels=False, zeroline=True,
                     zerolinecolor="lightgray", zerolinewidth=1)
    fig.update_xaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.15)", griddash="dash")
    return fig


def _overview_day_chart_load(day_df: pd.DataFrame, ts_index: pd.DatetimeIndex, ymax: float) -> go.Figure:
    fig = go.Figure()
    loads = day_df["load_kWh"].to_numpy()
    x = np.arange(len(day_df))
    time_str = pd.Index(ts_index).strftime("%H:%M")
    date_str = pd.Index(ts_index).strftime("%d %b")
    custom = np.stack([np.rint(loads).astype(int), time_str, date_str], axis=1)
    fig.add_bar(
        x=x,
        y=loads,
        width=0.9,
        marker_color=COLOR_CHART_BLACK,
        showlegend=False,
        customdata=custom,
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
    )
    fig.update_layout(
        height=120,
        margin=dict(l=2, r=2, t=2, b=2),
        bargap=0.01,
        showlegend=False,
    )
    fig.update_xaxes(visible=False, showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.15)", griddash="dash")
    fig.update_yaxes(range=[0, ymax], visible=False)
    return fig


def _overview_day_chart_irradiance(day_df: pd.DataFrame, ts_index: pd.DatetimeIndex, ymax: float) -> go.Figure:
    fig = go.Figure()
    irr_vals = day_df.get(IRRADIANCE_COL, pd.Series(np.nan, index=day_df.index)).to_numpy(dtype=float)
    x = np.arange(len(day_df))
    time_str = pd.Index(ts_index).strftime("%H:%M")
    date_str = pd.Index(ts_index).strftime("%d %b")
    custom = np.stack([np.rint(np.nan_to_num(irr_vals, nan=0.0)).astype(int), time_str, date_str], axis=1)
    mask = ~np.isnan(irr_vals)
    if mask.any():
        fig.add_scatter(
            x=x,
            y=np.where(mask, irr_vals, None),
            mode="lines",
            line=dict(color=COLOR_PV, width=2),
            showlegend=False,
            customdata=custom,
            hovertemplate="%{customdata[0]} W/m², %{customdata[1]}, %{customdata[2]}<extra></extra>",
        )
    fig.update_layout(
        height=120,
        margin=dict(l=2, r=2, t=2, b=2),
        showlegend=False,
    )
    fig.update_xaxes(visible=False, showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.2)", griddash="dash")
    fig.update_yaxes(range=[0, ymax], visible=False)
    return fig


def _night_segments(ts: pd.Series) -> list[tuple[int, int]]:
    ts = pd.to_datetime(ts).reset_index(drop=True)
    hours = ts.dt.hour.to_numpy()
    segments: list[tuple[int, int]] = []
    start_idx: Optional[int] = None
    for idx, hour in enumerate(hours):
        is_night = (hour >= 18) or (hour < 5)
        if is_night:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                segments.append((start_idx, idx - 1))
                start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(hours) - 1))
    return segments


def _overview_week_chart(week_df: pd.DataFrame, ts_col: str, data_mode: str,
                         show_pv: bool, show_ess: bool = False, show_night: bool = False,
                         irr_ymax: Optional[float] = None) -> go.Figure:
    if data_mode == "Summary":
        return _overview_week_chart_summary(week_df, ts_col, show_pv, show_ess, show_night)
    if data_mode == "Load":
        return _overview_week_chart_load(week_df, ts_col, show_night)
    return _overview_week_chart_irradiance(week_df, ts_col, ymax=irr_ymax)


def _overview_week_chart_load(week_df: pd.DataFrame, ts_col: str, show_night: bool = False) -> go.Figure:
    fig = go.Figure()
    ts = pd.to_datetime(week_df[ts_col]).reset_index(drop=True)
    load_vals = week_df["load_kWh"].to_numpy()
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    custom = np.stack([np.rint(load_vals).astype(int), time_str, date_str], axis=1)
    fig.add_bar(
        x=np.arange(len(load_vals)),
        y=load_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_CHART_BLACK,
        customdata=custom,
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
        showlegend=False,
    )
    ymax = float(np.nanmax(load_vals)) if load_vals.size else 0.0
    ymax = max(ymax, 1.0)
    day_starts, _ = _day_boundaries_and_labels(ts)
    for idx in range(len(day_starts)):
        if idx == 0:
            continue
        pos = day_starts[idx] - 0.5
        fig.add_vline(x=pos, line_width=1, line_dash="dash", line_color="lightgray")
    if show_night:
        for start_idx, end_idx in _night_segments(ts):
            x0 = start_idx - 0.5
            x1 = end_idx + 0.5
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(0,0,0,0.25)",
                opacity=0.35,
                line_width=0,
                layer="below",
                yref="y",
                y0=0,
                y1=ymax,
            )
    fig.update_layout(
        height=OVERVIEW_WEEK_HEIGHT_OTHER,
        margin=dict(l=0, r=0, t=4, b=0),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=[0, ymax], showticklabels=False)
    return fig


def _overview_week_chart_summary(week_df: pd.DataFrame, ts_col: str,
                                 show_pv: bool, show_ess: bool, show_night: bool) -> go.Figure:
    fig = go.Figure()
    ts = pd.to_datetime(week_df[ts_col]).reset_index(drop=True)
    load_vals = week_df["load_kWh"].to_numpy()
    x = np.arange(len(week_df))
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")

    def tooltip(values: np.ndarray) -> np.ndarray:
        cleaned = np.rint(np.abs(values)).astype(int)
        return np.stack([cleaned, time_str, date_str], axis=1)

    remaining_load = load_vals.copy()
    if show_pv:
        pv_vals = np.minimum(week_df["harvest_used_kWh"].to_numpy(), remaining_load)
        remaining_load = np.clip(remaining_load - pv_vals, 0.0, None)
        fig.add_bar(
            x=x,
            y=pv_vals,
            width=BAR_WIDTH,
            marker_color=COLOR_PV,
            showlegend=False,
            customdata=tooltip(pv_vals),
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
        )

    if show_ess and "batt_discharge_kWh" in week_df:
        ess_source = np.maximum(week_df["batt_discharge_kWh"].to_numpy(), 0.0)
        ess_vals = np.minimum(ess_source, remaining_load)
        remaining_load = np.clip(remaining_load - ess_vals, 0.0, None)
        fig.add_bar(
            x=x,
            y=ess_vals,
            width=BAR_WIDTH,
            marker_color=COLOR_BATT,
            showlegend=False,
            customdata=tooltip(ess_vals),
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
        )

    positive_base = remaining_load
    fig.add_bar(
        x=x,
        y=positive_base,
        width=BAR_WIDTH,
        marker_color=COLOR_CHART_BLACK,
        showlegend=False,
        customdata=tooltip(positive_base),
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
    )
    fig.add_bar(
        x=x,
        y=-load_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_LOAD,
        showlegend=False,
        customdata=tooltip(load_vals),
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}, %{customdata[2]}<extra></extra>",
    )

    ymax = _overview_ymax(week_df, show_pv)
    day_starts, _ = _day_boundaries_and_labels(ts)
    for idx in range(len(day_starts)):
        if idx == 0:
            continue
        pos = day_starts[idx] - 0.5
        fig.add_vline(x=pos, line_width=1, line_dash="dash", line_color="lightgray")

    if show_night:
        for start_idx, end_idx in _night_segments(ts):
            x0 = start_idx - 0.5
            x1 = end_idx + 0.5
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(0,0,0,0.25)",
                opacity=0.35,
                line_width=0,
                layer="below",
                yref="y",
                y0=-ymax,
                y1=ymax,
            )

    fig.update_layout(
        barmode="relative",
        height=OVERVIEW_WEEK_HEIGHT,
        margin=dict(l=0, r=0, t=8, b=2),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=[-ymax, ymax], showticklabels=False)
    return fig


def _overview_period_chart(period_df: pd.DataFrame, ts_col: str, show_pv: bool,
                           show_ess: bool, freq: str, label_fmt: str) -> go.Figure:
    df = period_df.copy()
    timestamps = pd.to_datetime(df[ts_col])
    df["_period"] = timestamps.dt.to_period(freq)
    agg_cols = ["load_kWh", "harvest_used_kWh", "batt_discharge_kWh"]
    grouped = df.groupby("_period")[agg_cols].sum() if agg_cols[0] in df else None
    if grouped is None or grouped.empty:
        return go.Figure()

    load_vals = grouped["load_kWh"].to_numpy(dtype=float)
    remaining = load_vals.copy()

    pv_vals = np.zeros_like(remaining)
    if show_pv and "harvest_used_kWh" in grouped:
        pv_source = grouped["harvest_used_kWh"].to_numpy(dtype=float)
        pv_vals = np.minimum(pv_source, remaining)
        remaining = np.clip(remaining - pv_vals, 0.0, None)

    ess_vals = np.zeros_like(remaining)
    if show_ess and "batt_discharge_kWh" in grouped:
        ess_source = np.maximum(grouped["batt_discharge_kWh"].to_numpy(dtype=float), 0.0)
        ess_vals = np.minimum(ess_source, remaining)
        remaining = np.clip(remaining - ess_vals, 0.0, None)

    labels = grouped.index.to_timestamp()
    if label_fmt == "day":
        x_labels = labels.strftime("%d")
    elif label_fmt == "month":
        x_labels = labels.strftime("%b")
    else:
        x_labels = labels.strftime("%Y-%m-%d")

    fig = go.Figure()
    idx = np.arange(len(grouped))
    tooltip = lambda vals: np.stack([np.rint(np.abs(vals)).astype(int),
                                     labels.strftime("%d %b %Y")], axis=1)
    if show_pv:
        fig.add_bar(
            x=idx,
            y=pv_vals,
            width=BAR_WIDTH,
            marker_color=COLOR_PV,
            showlegend=False,
            customdata=tooltip(pv_vals),
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}<extra></extra>",
        )
    if show_ess:
        fig.add_bar(
            x=idx,
            y=ess_vals,
            width=BAR_WIDTH,
            marker_color=COLOR_BATT,
            showlegend=False,
            customdata=tooltip(ess_vals),
            hovertemplate="%{customdata[0]} kWh, %{customdata[1]}<extra></extra>",
        )
    fig.add_bar(
        x=idx,
        y=remaining,
        width=BAR_WIDTH,
        marker_color=COLOR_CHART_BLACK,
        showlegend=False,
        customdata=tooltip(remaining),
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}<extra></extra>",
    )
    fig.add_bar(
        x=idx,
        y=-load_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_LOAD,
        showlegend=False,
        customdata=tooltip(load_vals),
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}<extra></extra>",
    )
    ymax = max(_overview_ymax(pd.DataFrame({"load_kWh": load_vals}), False), 1.0)
    fig.update_layout(
        barmode="relative",
        height=OVERVIEW_WEEK_HEIGHT,
        margin=dict(l=0, r=0, t=8, b=20),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=idx,
        ticktext=x_labels,
        tickangle=0,
        showgrid=False,
    )
    fig.update_yaxes(range=[-ymax, ymax], showticklabels=False)
    return fig


def _overview_period_chart_load(period_df: pd.DataFrame, ts_col: str,
                                freq: str, label_fmt: str) -> go.Figure:
    df = period_df.copy()
    timestamps = pd.to_datetime(df[ts_col])
    df["_period"] = timestamps.dt.to_period(freq)
    grouped = df.groupby("_period")["load_kWh"].sum()
    if grouped.empty:
        return go.Figure()
    load_vals = grouped.to_numpy(dtype=float)
    idx = np.arange(len(grouped))
    labels = grouped.index.to_timestamp()
    if label_fmt == "day":
        x_labels = labels.strftime("%d")
    elif label_fmt == "month":
        x_labels = labels.strftime("%b")
    else:
        x_labels = labels.strftime("%Y-%m-%d")

    time_labels = labels.strftime("%d %b %Y")
    hover_data = np.stack([np.rint(np.abs(load_vals)).astype(int), time_labels], axis=1)
    fig = go.Figure()
    fig.add_bar(
        x=idx,
        y=load_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_CHART_BLACK,
        customdata=hover_data,
        hovertemplate="%{customdata[0]} kWh, %{customdata[1]}<extra></extra>",
        showlegend=False,
    )
    ymax = max(np.max(load_vals), 1.0)
    ymax = np.ceil(ymax / 25.0) * 25.0
    fig.update_layout(
        barmode="relative",
        height=OVERVIEW_WEEK_HEIGHT_OTHER,
        margin=dict(l=0, r=0, t=8, b=20),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=idx,
        ticktext=x_labels,
        tickangle=0,
        showgrid=False,
    )
    fig.update_yaxes(range=[0, ymax], showticklabels=False)
    return fig


def _overview_period_chart_irradiance(period_df: pd.DataFrame, ts_col: str,
                                      freq: str, label_fmt: str) -> go.Figure:
    if IRRADIANCE_COL not in period_df.columns:
        return go.Figure()
    df = period_df[[ts_col, IRRADIANCE_COL]].copy()
    ts = pd.to_datetime(df[ts_col])
    df["_period"] = ts.dt.to_period(freq)
    grouped = df.groupby("_period")[IRRADIANCE_COL].sum()
    if grouped.empty:
        return go.Figure()
    vals = grouped.to_numpy(dtype=float)
    idx = np.arange(len(grouped))
    labels = grouped.index.to_timestamp()
    if label_fmt == "day":
        x_labels = labels.strftime("%d")
    else:
        x_labels = labels.strftime("%b")
    display_labels = labels.strftime("%d %b %Y")
    hover = np.stack([np.rint(vals).astype(int), display_labels], axis=1)
    fig = go.Figure()
    fig.add_bar(
        x=idx,
        y=vals,
        width=BAR_WIDTH,
        marker_color=COLOR_PV,
        showlegend=False,
        customdata=hover,
        hovertemplate="%{customdata[0]} Wh/m², %{customdata[1]}<extra></extra>",
    )
    ymax = max(np.max(vals), 1.0)
    ymax = np.ceil(ymax / 100.0) * 100.0
    fig.update_layout(
        height=OVERVIEW_WEEK_HEIGHT_OTHER,
        margin=dict(l=0, r=0, t=8, b=20),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=idx,
        ticktext=x_labels,
        showgrid=False,
    )
    fig.update_yaxes(range=[0, ymax], showticklabels=False)
    return fig


def _overview_week_chart_irradiance(week_df: pd.DataFrame, ts_col: str,
                                    ymax: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    if IRRADIANCE_COL not in week_df:
        fig.update_layout(
            height=OVERVIEW_WEEK_HEIGHT_OTHER,
            margin=dict(l=0, r=0, t=4, b=0),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    ts = pd.to_datetime(week_df[ts_col]).reset_index(drop=True)
    irr_vals = week_df[IRRADIANCE_COL].to_numpy(dtype=float)
    irr_vals = np.where(np.isfinite(irr_vals), irr_vals, np.nan)
    x = np.arange(len(irr_vals))
    time_str = ts.dt.strftime("%H:%M")
    date_str = ts.dt.strftime("%d %b")
    custom = np.stack([np.rint(np.nan_to_num(irr_vals, nan=0.0)).astype(int), time_str, date_str], axis=1)
    bar_vals = np.where(np.isnan(irr_vals), None, irr_vals)
    fig.add_bar(
        x=x,
        y=bar_vals,
        width=BAR_WIDTH,
        marker_color=COLOR_PV,
        customdata=custom,
        hovertemplate="%{customdata[0]} W/m², %{customdata[1]}, %{customdata[2]}<extra></extra>",
        showlegend=False,
    )
    day_starts, _ = _day_boundaries_and_labels(ts)
    for idx in range(len(day_starts)):
        if idx == 0:
            continue
        fig.add_vline(x=day_starts[idx], line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.25)")

    chart_cap = ymax if (ymax is not None and ymax > 0) else float(np.nanmax(np.nan_to_num(irr_vals, nan=0.0)))
    if chart_cap <= 0:
        chart_cap = 1.0
    if day_starts:
        for i, start_idx in enumerate(day_starts):
            end_idx = day_starts[i + 1] if i + 1 < len(day_starts) else len(irr_vals)
            day_slice = irr_vals[start_idx:end_idx]
            if day_slice.size == 0:
                continue
            day_total = np.nansum(day_slice) / 1000.0
            if not np.isfinite(day_total) or day_total <= 0:
                continue
            mid_x = start_idx + max((end_idx - start_idx - 1), 0) / 2.0
            fig.add_annotation(
                x=mid_x,
                y=chart_cap * 0.85,
                text=f"{day_total:.1f} HR",
                showarrow=False,
                font=dict(size=12, color="#1b8e3e"),
            )
    fig.update_layout(
        height=OVERVIEW_WEEK_HEIGHT_OTHER,
        margin=dict(l=0, r=0, t=2, b=2),
        bargap=BAR_GAP,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    if ymax is not None and ymax > 0:
        fig.update_yaxes(range=[0, ymax], showticklabels=False)
    else:
        fig.update_yaxes(showticklabels=False)
    return fig


def _overview_week_summary(week_df: pd.DataFrame, ts_col: str, data_mode: str,
                           show_pv: bool, show_ess: bool = False, timeframe: str = "Week") -> str:
    if data_mode == "Summary":
        week_load = float(week_df["load_kWh"].sum())
        week_pv = float(week_df["harvest_used_kWh"].sum()) if show_pv else 0.0
        week_spill = float(week_df.get("pv_spill_kWh", 0.0).sum()) if "pv_spill_kWh" in week_df else 0.0
        week_ess = float(week_df.get("batt_discharge_kWh", 0.0).sum()) if "batt_discharge_kWh" in week_df else 0.0
        pct = 0.0 if week_load <= 0 or not show_pv else (week_pv / week_load) * 100.0
        pv_label = f"{_format_mwh(week_pv)} MWh" if show_pv else "—"
        pct_markup = f"<span class='pv-green' style='font-size:16px;margin-left:6px;'>{pct:.0f}%</span>" if show_pv and week_load > 0 else ""
        ess_label = f"{_format_mwh(week_ess)} MWh" if show_ess else "—"
        ess_pct = 0.0 if week_load <= 0 or not show_ess else (week_ess / week_load) * 100.0
        ess_pct_markup = f"<span style='color:{COLOR_BATT};font-size:16px;margin-left:6px;'>{ess_pct:.0f}%</span>" if show_ess and week_load > 0 else ""
        html = "<div class='pv-overview-week-summary week'>"
        spill_label = f"{_format_mwh(week_spill)} MWh" if show_pv else "—"
        html += f"<div class='pv-spill-line'><span class='pv-yellow' style='font-size:20px;'>{spill_label}</span></div>"
        html += (
            f"<div><span style='color:{COLOR_BATT};font-size:20px;font-weight:700;'>{ess_label}</span>"
            f"{ess_pct_markup}</div>"
        )
        html += f"<div><span class='pv-green' style='font-size:20px;'>{pv_label}</span>{pct_markup}</div>"
        html += f"<div><span class='pv-red' style='font-size:20px;'>{_format_mwh(week_load)} MWh</span></div>"
        html += "</div>"
        return html
    if data_mode == "Load":
        week_load = float(week_df["load_kWh"].sum())
        html = "<div class='pv-overview-week-summary week week-small'>"
        html += "<div><span class='pv-green' style='font-size:20px;'>&nbsp;</span></div>"
        html += f"<div><span class='pv-red' style='font-size:20px;'>{_format_mwh(week_load)} MWh</span></div>"
        html += "</div>"
        return html
    if IRRADIANCE_COL not in week_df:
        return "<div class='pv-overview-week-summary week week-small'>&nbsp;</div>"
    avg_hr = _period_irradiance_avg_hr(week_df, ts_col)
    avg_label = f"{avg_hr:.1f} HR" if avg_hr > 0 else "—"
    title = "Week Avg" if timeframe == "Week" else ("Month Avg" if timeframe == "Month" else "Year Avg")
    html = "<div class='pv-overview-week-summary week week-small'>"
    html += f"<div style='text-align:center; width:100%;'>"
    html += f"<div style='font-size:14px;color:#777;'>{title}</div>"
    html += f"<div style='font-size:20px;font-weight:700;color:{COLOR_PV};'>{avg_label}</div>"
    html += "</div></div>"
    return html


def _period_irradiance_avg_hr(period_df: pd.DataFrame, ts_col: str) -> float:
    if IRRADIANCE_COL not in period_df or period_df.empty:
        return 0.0
    data = period_df[[ts_col, IRRADIANCE_COL]].copy()
    data["_date"] = pd.to_datetime(data[ts_col]).dt.date
    daily = data.groupby("_date")[IRRADIANCE_COL].sum(min_count=1)
    if daily.empty:
        return 0.0
    avg_kwh_m2 = float(daily.mean()) / 1000.0
    return max(avg_kwh_m2, 0.0)


def _overview_label_markup(label: str, data_mode: str) -> str:
    classes = ["pv-overview-week-wrap", "chart" if data_mode == "Summary" else "chart-small"]
    html = (
        f"<div class='{' '.join(classes)}'>"
        "<div class='pv-overview-week-label-block'>"
        f"<span class='pv-overview-week-label'>{label}</span>"
        "</div>"
        "</div>"
    )
    return html


def render_detail_summary(container, df: pd.DataFrame, g_cols: list[str]):
    totals = {col: float(df[col].sum()) for col in g_cols}
    total_sum = sum(totals.values())
    rows = ""
    for col in g_cols:
        val = totals[col] / 1000.0
        pct = 0 if total_sum <= 0 else int(round(100 * totals[col] / total_sum))
        rows += (
            "<div class='ess-line detail-row'>"
            f"<span class='detail-label'>{col}</span>"
            f"<span class='detail-value'>{_fmt_trim(val,2)}&nbsp;MWh&nbsp;&nbsp;{pct}%</span>"
            "</div>"
        )
    css = """
    <style>
    .detail-sum { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding-left: 8px; }
    .detail-title { font-size: 26px; font-weight: 700; margin: 0 0 6px 0; text-align: left; }
    .detail-row  { font-size: 18px; display: flex; gap: 24px; margin-bottom: 6px; }
    .detail-label { min-width: 40px; }
    .detail-value { flex: 1; }
    </style>
    """
    html = f"""
    {css}
    <div class="detail-sum">
      <div class="detail-title">Summary</div>
      {rows}
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


def render_generator_overview(detail: pd.DataFrame, ts_col: str, g_cols: list[str], total_col: str):
    if detail.empty:
        st.info("No generator data to summarize.")
        return
    totals = detail[g_cols].sum().sort_values(ascending=False)
    if totals.sum() <= 0:
        st.info("Generator columns contain no energy to summarize.")
        return

    colors = generator_colors(len(g_cols))
    color_map = {col: colors[idx] for idx, col in enumerate(g_cols)}
    totals_mwh = totals / 1000.0
    text_labels = [f"{_fmt_trim(val,2)} MWh" for val in totals_mwh]
    bar_colors = [color_map.get(col, "#777777") for col in totals_mwh.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=totals_mwh.index.tolist(),
                y=totals_mwh.values.tolist(),
                marker_color=bar_colors,
                text=text_labels,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=ENERGY_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=40),
        yaxis_title="MWh",
        xaxis_title="Generator",
        showlegend=False,
    )

    ts = pd.to_datetime(detail[ts_col])
    coverage = f"{ts.min():%d %b %Y} → {ts.max():%d %b %Y}"
    left, right = st.columns([4, 1], gap="large")
    with left:
        st.subheader("Generator Output Overview")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Coverage: {coverage}")
    with right:
        render_detail_summary(st, detail, g_cols)

    share = totals / totals.sum() * 100.0
    overview_table = pd.DataFrame({
        "Generator": totals_mwh.index,
        "Energy (MWh)": totals_mwh.values,
        "Share (%)": np.round(share.loc[totals_mwh.index].values, 1),
    })
    overview_table["Energy (MWh)"] = overview_table["Energy (MWh)"].round(2)
    st.dataframe(overview_table, use_container_width=True, hide_index=True)


def _month_dataframe(sim_all: pd.DataFrame, ts_col: str, year_sel: int, month_sel: int) -> pd.DataFrame:
    full_index = build_month_index(year_sel, month_sel)
    month_df = sim_all[(sim_all["year"] == year_sel) & (sim_all["month"] == month_sel)].copy()
    if ts_col not in month_df.columns:
        month_df[ts_col] = pd.to_datetime(month_df.index)
    month_df = month_df.set_index(pd.to_datetime(month_df[ts_col])).sort_index()
    month_full = pd.DataFrame(index=full_index)
    for col in VALUE_COLS:
        month_full[col] = month_df[col].reindex(full_index).fillna(0.0)
    if IRRADIANCE_COL in month_df.columns:
        month_full[IRRADIANCE_COL] = month_df[IRRADIANCE_COL].reindex(full_index).fillna(0.0)
    if "soc_pct" in month_df.columns:
        month_full["soc_pct"] = month_df["soc_pct"].reindex(full_index).ffill().fillna(0.0)
    else:
        month_full["soc_pct"] = 0.0
    month_full[ts_col] = month_full.index
    return month_full


def _year_dataframe(sim_all: pd.DataFrame, ts_col: str, year_sel: int) -> pd.DataFrame:
    year_df = sim_all[sim_all["year"] == year_sel].copy()
    agg_spec = {col: "sum" for col in VALUE_COLS}
    if IRRADIANCE_COL in year_df.columns:
        agg_spec[IRRADIANCE_COL] = "sum"
    agg_spec["date"] = "nunique"
    monthly = year_df.groupby("month").agg(agg_spec)
    monthly = monthly.reindex(range(1, 13), fill_value=0.0)
    monthly = monthly.reset_index().rename(columns={"month": "month_num", "date": "days_present"})
    monthly["days_present"] = monthly["days_present"].fillna(0).astype(int)
    month_dates = pd.to_datetime({"year": year_sel, "month": monthly["month_num"], "day": 1})
    monthly[ts_col] = month_dates
    return monthly


def render_pv_overview(sim_all: pd.DataFrame, ts_col: str, weeks: List[tuple[int, int]],
                       data_mode: str, irradiance_enabled: bool, timeframe: str = "Week"):
    timeframe = timeframe if timeframe in {"Week", "Month", "Year"} else "Week"
    is_summary = data_mode == "Summary"
    is_load = data_mode == "Load"
    is_irr = data_mode == "Irradiance"

    if is_irr and timeframe not in {"Week", "Month", "Year"}:
        timeframe = "Week"
    allow_night = (timeframe == "Week")

    if is_irr and not irradiance_enabled:
        st.info("Irradiance view is available only when NASA GHI data is selected.")
        return
    if data_mode not in {"Summary", "Load", "Irradiance"}:
        st.info("Selected data mode is not available.")
        return

    st.markdown(
        OVERVIEW_CSS + """
<style>
.pv-row > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stToggle"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    padding-top: 0 !important;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stToggle"]) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stToggle"]) + div[data-testid="stVerticalBlock"]:has(> div[data-testid="stToggle"]) {
    margin-top: -14px !important;
}
</style>""",
        unsafe_allow_html=True
    )
    st.session_state.setdefault("pv_overview_show_pv", False)
    st.session_state.setdefault("pv_overview_show_ess", False)
    st.session_state.setdefault("pv_overview_show_night", False)

    show_pv = st.session_state["pv_overview_show_pv"]
    show_ess = st.session_state["pv_overview_show_ess"]
    show_night = st.session_state["pv_overview_show_night"]
    if not show_pv and show_ess:
        st.session_state["pv_overview_show_ess"] = False
        show_ess = False
    if is_summary:
        if allow_night:
            show_night = st.toggle("Show Night", key="pv_overview_show_night")
        else:
            show_night = False
            st.session_state["pv_overview_show_night"] = False
        show_pv = st.toggle("PV Direct", key="pv_overview_show_pv")
        show_ess = st.toggle("ESS", disabled=not show_pv, key="pv_overview_show_ess")
    elif is_load:
        show_pv = False
        show_ess = False
        if allow_night:
            show_night = st.toggle("Show Night", key="pv_overview_show_night")
        else:
            show_night = False
            st.session_state["pv_overview_show_night"] = False
    else:
        show_pv = st.session_state["pv_overview_show_pv"]
        show_ess = st.session_state["pv_overview_show_ess"]
        show_night = st.session_state["pv_overview_show_night"] if allow_night else False

    period_label = "Week" if timeframe == "Week" else timeframe
    header_cols = st.columns([0.35, 5.65, 1], gap="small")
    header_cols[0].markdown("&nbsp;")
    if timeframe == "Week":
        header_cols[1].markdown(
            "<div class='pv-overview-day-strip'>"
            + "".join(f"<span>{label}</span>" for label in OVERVIEW_DAY_LABELS)
            + "</div>",
            unsafe_allow_html=True
        )
    else:
        header_cols[1].markdown("&nbsp;")
    if is_summary or is_load:
        header_cols[2].markdown(
            f"<div class='pv-overview-day-header' style='text-align:center;'>{period_label} Total</div>",
            unsafe_allow_html=True
        )
    elif is_irr:
        header_cols[2].markdown(
            f"<div class='pv-overview-day-header' style='text-align:center;'>{period_label} Avg</div>",
            unsafe_allow_html=True
        )
    else:
        header_cols[2].markdown("&nbsp;")

    allow_night = (timeframe == "Week")
    if not allow_night:
        show_night = False

    period_records: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    global_irr_max = 0.0
    if timeframe == "Week":
        if not weeks:
            st.info("No week data available for the overview.")
            return
        for year_sel, week_sel in weeks:
            week_full = _week_dataframe(sim_all, ts_col, year_sel, week_sel)
            label = f"W{week_sel:02d} {year_sel}"
            period_records.append((label, week_full, week_full))
            if is_irr and IRRADIANCE_COL in week_full:
                irr_vals = week_full[IRRADIANCE_COL].to_numpy(dtype=float)
                finite_vals = irr_vals[np.isfinite(irr_vals)]
                if finite_vals.size:
                    local_max = float(finite_vals.max())
                    if local_max > global_irr_max:
                        global_irr_max = local_max
    elif timeframe == "Month":
        months = month_list(sim_all)
        if not months:
            st.info("No month data available for the overview.")
            return
        for year_sel, month_sel in months:
            month_full = _month_dataframe(sim_all, ts_col, year_sel, month_sel)
            label = datetime(year_sel, month_sel, 1).strftime("%b %Y")
            period_records.append((label, month_full, month_full))
            if is_irr and IRRADIANCE_COL in month_full:
                irr_vals = month_full[IRRADIANCE_COL].to_numpy(dtype=float)
                finite_vals = irr_vals[np.isfinite(irr_vals)]
                if finite_vals.size:
                    local_max = float(finite_vals.max())
                    if local_max > global_irr_max:
                        global_irr_max = local_max
    else:  # Year
        years = year_list(sim_all)
        if not years:
            st.info("No year data available for the overview.")
            return
        for year_sel in years:
            year_full = _year_dataframe(sim_all, ts_col, year_sel)
            year_raw = sim_all[sim_all["year"] == year_sel].copy()
            label = str(year_sel)
            period_records.append((label, year_full, year_raw))
            if is_irr and IRRADIANCE_COL in year_full:
                irr_vals = year_full[IRRADIANCE_COL].to_numpy(dtype=float)
                finite_vals = irr_vals[np.isfinite(irr_vals)]
                if finite_vals.size:
                    local_max = float(finite_vals.max())
                    if local_max > global_irr_max:
                        global_irr_max = local_max
    irr_ymax = None
    if is_irr:
        irr_ymax = global_irr_max if global_irr_max > 0 else None

    for label, period_df, summary_df in period_records:
        if timeframe == "Week":
            chart = _overview_week_chart(period_df, ts_col, data_mode, show_pv,
                                         show_ess, show_night=(show_night if allow_night else False),
                                         irr_ymax=irr_ymax)
        else:
            freq = "D" if timeframe == "Month" else "M"
            label_fmt = "day" if timeframe == "Month" else "month"
            if data_mode == "Summary":
                chart = _overview_period_chart(period_df, ts_col, show_pv, show_ess, freq, label_fmt)
            elif data_mode == "Load":
                chart = _overview_period_chart_load(period_df, ts_col, freq, label_fmt)
            elif data_mode == "Irradiance":
                chart = _overview_period_chart_irradiance(period_df, ts_col, freq, label_fmt)
            else:
                chart = _overview_week_chart(period_df, ts_col, data_mode, show_pv,
                                             show_ess, show_night=False, irr_ymax=irr_ymax)
        st.markdown("<div class='pv-row'>", unsafe_allow_html=True)
        row = st.columns([0.35, 5.65, 1], gap="small")
        with row[0]:
            st.markdown(
                _overview_label_markup(label, data_mode),
                unsafe_allow_html=True
            )
        with row[1]:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        with row[2]:
            st.markdown(_overview_week_summary(summary_df, ts_col, data_mode, show_pv, show_ess, timeframe), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if is_summary or is_load:
        st.markdown("<div class='pv-overview-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pv-row'>", unsafe_allow_html=True)
        avg_cols = st.columns([0.35, 5.65, 1], gap="small")
        avg_cols[0].markdown("<div class='pv-overview-week-wrap avg'><span class='pv-overview-week-label'>Avg</span></div>", unsafe_allow_html=True)
        avg_html = ""
        sim_copy = sim_all.copy()
        sim_copy["_date"] = pd.to_datetime(sim_copy[ts_col]).dt.date
        daily = sim_copy.groupby("_date")[["load_kWh", "harvest_kWh", "harvest_used_kWh"]].sum()
        daily["dow"] = pd.to_datetime(daily.index).dayofweek
        dow_avg = daily.groupby("dow").mean().reindex(range(7)).fillna(0.0)
        for day_idx, label in enumerate(OVERVIEW_DAY_LABELS):
            stats = dow_avg.iloc[day_idx] if not dow_avg.empty else pd.Series()
            if is_summary:
                avg_load = float(stats.get("load_kWh", 0.0))
                avg_harvest = float(stats.get("harvest_kWh", 0.0))
                avg_pv = float(stats.get("harvest_used_kWh", 0.0)) if show_pv else 0.0
                pct = 0.0 if avg_harvest <= 0 or not show_pv else (avg_pv / avg_harvest) * 100.0
                pv_text = f"{_format_mwh(avg_pv)} MWh" if show_pv else "—"
                pct_text = f"{pct:.0f}% Used" if show_pv and avg_harvest > 0 else ""
                pct_markup = f"<span class='pv-green' style='font-size:14px;margin-left:6px;'>{pct_text}</span>" if pct_text else ""
                pv_markup = f"<span class='pv-green week-total-text'>{pv_text}</span>{pct_markup}"
                load_markup = f"<span class='pv-red week-total-text'>{_format_mwh(avg_load)} MWh</span>"
            else:
                avg_load = float(stats.get("load_kWh", 0.0))
                pv_markup = "<span class='pv-green week-total-text'>&nbsp;</span>"
                load_markup = f"<span class='pv-red week-total-text'>{_format_mwh(avg_load)} MWh</span>"
            avg_html += f"<div class='pv-overview-day-value'><div>{pv_markup}</div><div>{load_markup}</div></div>"
        avg_cols[1].markdown(
            f"<div class='pv-overview-day-values'>{avg_html}</div>",
            unsafe_allow_html=True
        )
        avg_cols[2].markdown("&nbsp;")
        st.markdown("</div>", unsafe_allow_html=True)


def _fmt_trim(v: float, decimals=1) -> str:
    # Custom rule:
    # If value < 1 -> keep 2 decimals max
    # If value >= 1 -> keep 1 decimal max
    try:
        if abs(v) < 1:
            return f"{v:.2f}".rstrip('0').rstrip('.')
        else:
            return f"{v:.1f}".rstrip('0').rstrip('.')
    except:
        return str(v)




def _fmt_pv_capacity(pv_kwp: float) -> str:
    # Remove incorrect division; just format as kWp or MWp properly
    if pv_kwp >= 1000:
        return f"{_fmt_trim(pv_kwp/1000,1)} MWp"
    return f"{_fmt_trim(pv_kwp,1)} kWp"


def _fmt_summary_mwh(val_mwh: float) -> str:
    if abs(val_mwh) >= 100:
        return f"{int(round(val_mwh))}"
    return _fmt_trim(val_mwh, 1)


def render_week_summary(container, batt_mwh: float, pv_kwp: float, week_df: pd.DataFrame, avg_day_mwh: float | None = None):
    total_load = float(week_df["load_kWh"].sum())
    pv_direct  = float(week_df["harvest_used_kWh"].sum())
    ess_used   = float(week_df["batt_discharge_kWh"].sum())
    genset     = float(week_df["diesel_kWh"].sum())

    pct = (lambda v: 0.0 if total_load <= 0 else (100.0 * v / total_load))
    pv_pct, ess_pct, gen_pct = pct(pv_direct), pct(ess_used), pct(genset)

    css = """
    <style>
    .ess-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding-left: 8px; }
    .ess-title { font-size: 30px; font-weight: 700; margin: 0 0 6px 0; text-align: left; }
    .ess-title + .ess-title { margin-top: -4px; }
    .ess-line  { font-size: 18px; display: flex; justify-content: space-between; width: 260px; }
    .ess-hr    { border-top: 1px solid #e5e5e5; margin: 8px 0; }
    </style>
    """
    html = f"""
    {css}
    <div class="ess-wrap">
      <div class="ess-title">PV {_fmt_pv_capacity(pv_kwp)}</div>
      <div class="ess-title">ESS { _fmt_trim(batt_mwh,1) } MWh</div>
      <div style="height:6px"></div>
      <div class="ess-line"><span>PV Direct</span><span>{_fmt_summary_mwh(pv_direct/1000)}&nbsp;MWh&nbsp;&nbsp;{int(round(pv_pct))}%</span></div>
      <div class="ess-line"><span>ESS</span><span>{_fmt_summary_mwh(ess_used/1000)}&nbsp;MWh&nbsp;&nbsp;{int(round(ess_pct))}%</span></div>
      <div class="ess-line"><span>Grid</span><span>{_fmt_summary_mwh(genset/1000)}&nbsp;MWh&nbsp;&nbsp;{int(round(gen_pct))}%</span></div>
      <div class="ess-hr"></div>
      <div class="ess-line"><span><b>Load Total</b></span><span><b>{_fmt_summary_mwh(total_load/1000)}&nbsp;MWh</b></span></div>
      {f'<div class="ess-line"><span>Avg / day</span><span>{_fmt_summary_mwh(avg_day_mwh)}&nbsp;MWh</span></div>' if avg_day_mwh is not None else ''}
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


def _render_value_summary(container, label: str, value_mwh: float, suffix: str | None = None):
    css = """
    <style>
    .pv-summary-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding-left: 8px;}
    .pv-summary-line  { font-size: 18px; display: flex; justify-content: space-between; width: 260px; }
    </style>
    """
    value_text = f"{_fmt_summary_mwh(value_mwh)} MWh"
    if suffix:
        value_text = f"{value_text} {suffix}"
    html = f"""
    {css}
    <div class="pv-summary-wrap">
      <div class="pv-summary-line"><span>{label}</span><span>{value_text}</span></div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


def render_spill_summary(container, period_df: pd.DataFrame | None):
    if period_df is None or "pv_spill_kWh" not in period_df:
        return
    spill = float(period_df["pv_spill_kWh"].sum())
    harvest_total = float(period_df["harvest_kWh"].sum()) if "harvest_kWh" in period_df else 0.0
    suffix = None
    if harvest_total > 0:
        spill_pct = (spill / harvest_total) * 100.0
        suffix = f"{spill_pct:.0f}%"
    _render_value_summary(container, "PV Spill", spill / 1000.0, suffix=suffix)


def render_harvest_summary(container, period_df: pd.DataFrame | None):
    if period_df is None or "harvest_kWh" not in period_df:
        return
    harvest_total = float(period_df["harvest_kWh"].sum())
    _render_value_summary(container, "Harvest", harvest_total / 1000.0)


def _peak_mask(timestamps: pd.Series) -> np.ndarray:
    hours = pd.to_datetime(timestamps).dt.hour.to_numpy(dtype=int)
    return (hours >= 18) & (hours < 22)


def compute_cost_comparison(
    cost_df: pd.DataFrame,
    ts_col: str,
    *,
    pricing_mode: str,
    flat_price: float,
    peak_price: float,
    offpeak_price: float,
    pv_discount_pct: float,
    ess_discount_pct: float,
    pv_override: float | None,
    ess_override: float | None,
) -> pd.DataFrame | None:
    if cost_df is None or cost_df.empty:
        return None
    required = ["load_kWh", "harvest_used_kWh", "batt_discharge_kWh"]
    for col in required:
        if col not in cost_df.columns:
            return None
    df = cost_df[[ts_col] + required].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    load = df["load_kWh"].to_numpy(dtype=float)
    pv_direct = df["harvest_used_kWh"].to_numpy(dtype=float)
    ess_used = df["batt_discharge_kWh"].to_numpy(dtype=float)
    if load.size == 0:
        return None
    pv_discount = max(0.0, min(100.0, pv_discount_pct)) / 100.0
    ess_discount = max(0.0, min(100.0, ess_discount_pct)) / 100.0

    if pricing_mode == "Flat Price":
        base_rate = np.full(load.shape, max(flat_price, 0.0), dtype=float)
    else:
        peak_mask = _peak_mask(df[ts_col])
        peak_vals = np.where(peak_mask, max(peak_price, 0.0), max(offpeak_price, 0.0))
        base_rate = peak_vals.astype(float)

    baseline_step = load * base_rate
    grid_portion = np.clip(load - pv_direct - ess_used, 0.0, None)

    def _override_or_discount(override_value: float | None, discount: float) -> np.ndarray:
        if override_value is not None and override_value > 0:
            return np.full(load.shape, override_value, dtype=float)
        return base_rate * (1.0 - discount)

    pv_rate = _override_or_discount(pv_override, pv_discount)
    ess_rate = _override_or_discount(ess_override, ess_discount)
    pv_rate = np.clip(pv_rate, 0.0, None)
    ess_rate = np.clip(ess_rate, 0.0, None)

    pv_cost = pv_direct * pv_rate
    ess_cost = ess_used * ess_rate
    grid_cost = grid_portion * base_rate
    hybrid_step = pv_cost + ess_cost + grid_cost

    result = pd.DataFrame({
        ts_col: df[ts_col],
        "grid_cost": baseline_step,
        "pv_ess_cost": hybrid_step,
    })
    return result



def render_energy_cost_section(cost_df: pd.DataFrame | None, ts_col: str):
    if cost_df is None or cost_df.empty:
        st.info("No data available for the selected timeframe.")
        return

    pricing_defaults = st.session_state.setdefault("pricing_inputs", {
        "mode": "Peak / Off-Peak",
        "detail": "Simple",
        "flat_price": 1200.0,
        "peak_price": 1500.0,
        "offpeak_price": 1000.0,
        "pv_discount": 15.0,
        "ess_discount": 20.0,
        "pv_price": 800.0,
        "ess_price": 900.0,
    })
    pricing_mode = pricing_defaults.get("mode", "Peak / Off-Peak")
    detail_mode = pricing_defaults.get("detail", "Simple")
    flat_price = float(pricing_defaults.get("flat_price", 1200.0))
    peak_price = float(pricing_defaults.get("peak_price", 1500.0))
    offpeak_price = float(pricing_defaults.get("offpeak_price", 1000.0))
    if pricing_mode == "Flat Price":
        peak_price = offpeak_price = flat_price
    pv_discount = float(pricing_defaults.get("pv_discount", 15.0))
    ess_discount = float(pricing_defaults.get("ess_discount", 20.0))
    pv_override = float(pricing_defaults.get("pv_price", 800.0)) if detail_mode == "Advanced" else None
    ess_override = float(pricing_defaults.get("ess_price", 900.0)) if detail_mode == "Advanced" else None

    chart_col, summary_col = st.columns([4, 1], gap="large")
    summary_box = summary_col.container()

    cost_series = compute_cost_comparison(
        cost_df,
        ts_col,
        pricing_mode=pricing_mode,
        flat_price=flat_price,
        peak_price=peak_price,
        offpeak_price=offpeak_price,
        pv_discount_pct=pv_discount,
        ess_discount_pct=ess_discount,
        pv_override=pv_override,
        ess_override=ess_override,
    )

    if cost_series is None or cost_series.empty:
        st.info("Unable to compute cost comparison for the selected period.")
        return

    pv_cost = float(cost_series["pv_ess_cost"].sum())
    grid_cost = float(cost_series["grid_cost"].sum())
    savings = grid_cost - pv_cost
    savings_pct = (savings / grid_cost * 100) if grid_cost > 0 else 0.0

    with summary_box:
        st.metric("Savings", f"IDR {_fmt_comma(savings, 0)}", f"{savings_pct:.1f}%")
        st.metric("PV + ESS Cost", f"IDR {_fmt_comma(pv_cost, 0)}")
        st.metric("Grid Only Cost", f"IDR {_fmt_comma(grid_cost, 0)}")

    is_week = (st.session_state.timeframe == "Week")
    timeframe = st.session_state.timeframe
    hover = "IDR %{y:,.0f}<br>%{x|%d %b %Y %H:%M}<extra></extra>"

    def render_cost_chart(series_df: pd.DataFrame, *, cumulative: bool = False, show_night: bool = False):
        chart = go.Figure()
        data = series_df.copy()
        if cumulative and timeframe in ("Day", "Week"):
            data["grid_cost"] = data["grid_cost"].cumsum()
            data["pv_ess_cost"] = data["pv_ess_cost"].cumsum()

        def add_bar_trace(x_vals, y_vals, name, color):
            chart.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                name=name,
                marker_color=color,
                width=BAR_WIDTH,
                opacity=0.7,
                hovertemplate="IDR %{y:,.0f}<br>%{x}<extra></extra>"
            ))

        if timeframe in ("Day", "Week"):
            hover_tpl = hover if not cumulative else "IDR %{y:,.0f}<br>%{x|%d %b %Y %H:%M}<extra></extra>"
            if show_night:
                ts_vals = pd.to_datetime(data[ts_col])
                hours = ts_vals.dt.hour
                night_mask = (hours < 6) | (hours >= 18)
                current_start = None
                for idx, is_night in enumerate(night_mask):
                    if is_night and current_start is None:
                        current_start = ts_vals.iloc[idx]
                    elif not is_night and current_start is not None:
                        chart.add_vrect(
                            x0=current_start,
                            x1=ts_vals.iloc[idx-1],
                            fillcolor="rgba(80,80,80,0.12)",
                            line_width=0,
                            layer="below"
                        )
                        current_start = None
                if current_start is not None:
                    chart.add_vrect(
                        x0=current_start,
                        x1=ts_vals.iloc[-1],
                        fillcolor="rgba(80,80,80,0.12)",
                        line_width=0,
                        layer="below"
                    )
            chart.add_trace(go.Scatter(
                x=data[ts_col],
                y=data["grid_cost"],
                mode="lines",
                name="Grid Only",
                line=dict(color=COLOR_CHART_BLACK, width=3),
                hovertemplate=hover_tpl
            ))
            chart.add_trace(go.Scatter(
                x=data[ts_col],
                y=data["pv_ess_cost"],
                mode="lines",
                name="PV + ESS",
                line=dict(color=COLOR_PV, width=3),
                hovertemplate=hover_tpl
            ))
        elif timeframe == "Month":
            data["_day"] = pd.to_datetime(data[ts_col]).dt.floor("D")
            grouped = data.groupby("_day")[["grid_cost", "pv_ess_cost"]].sum().reset_index()
            if cumulative:
                grouped["grid_cost"] = grouped["grid_cost"].cumsum()
                grouped["pv_ess_cost"] = grouped["pv_ess_cost"].cumsum()
            labels = grouped["_day"].dt.strftime("%d %b (%a)")
            add_bar_trace(labels, grouped["grid_cost"], "Grid Only", COLOR_CHART_BLACK)
            add_bar_trace(labels, grouped["pv_ess_cost"], "PV + ESS", COLOR_PV)
        else:  # Year
            data["_month"] = pd.to_datetime(data[ts_col]).dt.to_period("M").dt.to_timestamp()
            grouped = data.groupby("_month")[["grid_cost", "pv_ess_cost"]].sum().reset_index()
            if cumulative:
                grouped["grid_cost"] = grouped["grid_cost"].cumsum()
                grouped["pv_ess_cost"] = grouped["pv_ess_cost"].cumsum()
            labels = grouped["_month"].dt.strftime("%b %Y")
            add_bar_trace(labels, grouped["grid_cost"], "Grid Only", COLOR_CHART_BLACK)
            add_bar_trace(labels, grouped["pv_ess_cost"], "PV + ESS", COLOR_PV)

        barmode = "overlay" if timeframe in ("Month", "Year") else None
        chart.update_layout(
            height=BASE_HEIGHT * 2,
            margin=dict(l=0, r=10, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(showticklabels=False, title=None),
            xaxis=dict(showticklabels=False),
            barmode=barmode,
        )
        chart.update_yaxes(showgrid=True, gridcolor="#eee")
        return chart

    show_night = bool(st.session_state.get("energy_show_night_week", False))
    if st.session_state.timeframe == "Day":
        show_night = bool(st.session_state.get("energy_show_night_day", False))

    cost_fig = render_cost_chart(cost_series, cumulative=False, show_night=show_night)
    with chart_col:
        st.plotly_chart(cost_fig, use_container_width=True, config={"displayModeBar": True})

    if timeframe in ("Day", "Week", "Month", "Year"):
        cum_fig = render_cost_chart(cost_series, cumulative=True, show_night=False)
        with chart_col:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(cum_fig, use_container_width=True, config={"displayModeBar": True})


def _fmt_comma(val: float, decimals: int = 0, unit: str = "") -> str:
    if val is None or np.isnan(val):
        return "-"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(val) + unit


def _fmt_percent(val: float) -> str:
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.0f}%"


def render_finance_view(sim_all: pd.DataFrame, pv_inputs: dict):
    finance_defaults = st.session_state.setdefault("finance_inputs", {
        "pcs_kw": 1000.0,
        "pv_price_usd": 250.0,
        "ess_price_usd": 350.0,
        "pcs_price_usd": 75.0,
        "usd_idr": 16600.0,
        "pv_direct_price_idr": 1250.0,
        "ess_price_idr": 3600.0,
        "rental_years": 5,
        "pv_month": "Avg",
        "ess_month": "Avg",
        "pv_min_offtake": 36000.0,
        "ess_min_offtake": 20000.0,
    })

    monthly = (
        sim_all.groupby(["year", "month"])[["harvest_used_kWh", "batt_discharge_kWh", "load_kWh"]]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )
    day_counts = (
        sim_all.groupby(["year", "month"])["date"]
        .nunique()
        .reset_index()
        .rename(columns={"date": "days_present"})
    )
    monthly = monthly.merge(day_counts, on=["year", "month"], how="left")
    monthly["days_present"] = monthly["days_present"].fillna(0).astype(int)
    monthly["label"] = monthly.apply(
        lambda r: datetime(int(r["year"]), int(r["month"]), 1).strftime("%b %Y"), axis=1
    ) if not monthly.empty else []

    if monthly.empty:
        st.warning("Monthly data unavailable for finance view. Upload more data first.")
        return

    month_options = monthly["label"].tolist()
    if finance_defaults["pv_month"] not in month_options:
        finance_defaults["pv_month"] = month_options[0]
    if finance_defaults["ess_month"] not in month_options:
        finance_defaults["ess_month"] = month_options[0]

    label_map = dict(zip(monthly["label"], zip(monthly["year"], monthly["month"])))
    pv_data_source = st.session_state.get("pv_data_source_selection", NASA_SOURCE)

    def month_value(label: str, column: str) -> float:
        row = monthly[monthly["label"] == label]
        if row.empty:
            return 0.0
        return float(row[column].iloc[0])

    def reference_values(label: str) -> tuple[float, float, float, float]:
        if label == "Avg Month":
            if monthly.empty:
                return 0.0, 0.0, 0.0, 0.0
            return (
                float(monthly["load_kWh"].mean()),
                float(monthly["harvest_used_kWh"].mean()),
                float(monthly["batt_discharge_kWh"].mean()),
                float(monthly["days_present"].mean()),
            )
        row = monthly[monthly["label"] == label]
        if row.empty:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(row["load_kWh"].iloc[0]),
            float(row["harvest_used_kWh"].iloc[0]),
            float(row["batt_discharge_kWh"].iloc[0]),
            float(row["days_present"].iloc[0]),
        )

    savings_state = st.session_state.setdefault(
        "finance_savings",
        {
            "price_idr": 1500.0,
            "reference": "Avg Month",
            "options": [
                {"pct": 20.0},
                {"pct": 40.0},
                {"pct": 60.0},
            ],
        },
    )
    while len(savings_state["options"]) < 3:
        savings_state["options"].append({"pct": 20.0})

    st.subheader("Savings Targets")
    with st.expander("Calculation Model", expanded=True):
        calc_mode = st.radio(
            "Calculation Mode",
            ["Simple", "Advanced"],
            horizontal=True,
            key="finance_calc_mode"
        )
    price_col, ref_col, system_col = st.columns([1, 1, 1], gap="small")
    with price_col:
        savings_state["price_idr"] = st.number_input(
            "Electricity Price (IDR/kWh)",
            min_value=0.0,
            value=float(savings_state["price_idr"]),
            step=100.0,
            format="%g",
            key="finance_savings_price",
        )
    reference_choices = ["Avg Month"] + month_options
    if savings_state["reference"] not in reference_choices:
        savings_state["reference"] = reference_choices[0]
    prev_reference = savings_state.get("last_reference", savings_state["reference"])
    with ref_col:
        savings_state["reference"] = st.selectbox(
            "Reference Month",
            reference_choices,
            index=reference_choices.index(savings_state["reference"]),
            key="finance_savings_reference",
        )
        ref_load_tmp, _, _, _ = reference_values(savings_state["reference"])
        if ref_load_tmp > 0:
            estimated_bill = ref_load_tmp * savings_state["price_idr"]
            st.caption(f"Monthly bill estimate: {_fmt_comma(estimated_bill, 0)} IDR")
    reference_changed = savings_state["reference"] != prev_reference
    savings_state["last_reference"] = savings_state["reference"]
    system_choices = ["PV", "PV + ESS"]
    if savings_state.get("system") not in system_choices:
        savings_state["system"] = "PV + ESS"
    ref_load, ref_pv_energy, ref_ess_energy, ref_days = reference_values(savings_state["reference"])
    ts_col = detect_ts_column(sim_all)
    irr_col = IRRADIANCE_COL if IRRADIANCE_COL in sim_all.columns else None
    sunrise_time = pv_inputs.get("sunrise", time(6, 0))
    sunset_time = pv_inputs.get("sunset", time(18, 0))

    def daylight_ratio_pct(label: str) -> float:
        df = sim_all[[ts_col, "load_kWh"]].copy()
        df["_ts"] = pd.to_datetime(df[ts_col])
        if label != "Avg Month" and label in label_map:
            year, month = label_map[label]
            df = df[(df["_ts"].dt.year == year) & (df["_ts"].dt.month == month)]
        if df.empty:
            return 0.0
        load_total = float(df["load_kWh"].sum())
        if load_total <= 0:
            return 0.0
        if pv_data_source == NASA_SOURCE and irr_col is not None:
            irr = sim_all.loc[df.index, irr_col]
            day_mask = irr > 0
        else:
            hours = df["_ts"].dt.hour + df["_ts"].dt.minute / 60.0 + df["_ts"].dt.second / 3600.0
            sunrise_hr = sunrise_time.hour + sunrise_time.minute / 60.0 + sunrise_time.second / 3600.0
            sunset_hr = sunset_time.hour + sunset_time.minute / 60.0 + sunset_time.second / 3600.0
            if sunrise_hr >= sunset_hr:
                day_mask = (hours >= sunrise_hr) | (hours < sunset_hr)
            else:
                day_mask = (hours >= sunrise_hr) & (hours < sunset_hr)
        load_day = float(df.loc[day_mask, "load_kWh"].sum())
        return 0.0 if load_day <= 0 else (load_day / load_total) * 100.0

    pv_only_max_pct = daylight_ratio_pct(savings_state["reference"])
    prev_system = savings_state.get("last_system", savings_state["system"])
    with system_col:
        savings_state["system"] = st.selectbox(
            "System",
            system_choices,
            index=system_choices.index(savings_state["system"]),
            key="finance_savings_system",
        )
        if savings_state["system"] == "PV" and ref_load > 0:
            st.caption(f"Max achievable savings with PV-only: {pv_only_max_pct:.0f}%")
        else:
            st.caption("&nbsp;", unsafe_allow_html=True)
    system_changed = savings_state["system"] != prev_system
    savings_state["last_system"] = savings_state["system"]
    prev_mode = savings_state.get("last_calc_mode", "Simple")
    if reference_changed or system_changed or prev_mode != calc_mode:
        for opt in savings_state["options"]:
            opt["computed_state"] = None
    savings_state["last_calc_mode"] = calc_mode
    def pv_energy_per_kwp(label: str, days_count: float) -> float:
        if pv_data_source == NASA_SOURCE:
            aligned = st.session_state.get("nasa_harvest_aligned")
            ts_col_aligned = st.session_state.get("nasa_aligned_ts_col")
            pr_val = float(st.session_state.get("nasa_fetch", {}).get("pr_pct", 80.0)) / 100.0
            if aligned is None or ts_col_aligned is None:
                return 0.0
            df = aligned.copy()
            df["_ts"] = pd.to_datetime(df[ts_col_aligned])
            if label != "Avg Month" and label in label_map:
                year, month = label_map[label]
                df = df[(df["_ts"].dt.year == year) & (df["_ts"].dt.month == month)]
            if df.empty:
                return 0.0
            return float(df["harvest_kWh_base"].sum()) * pr_val
        else:
            pv_hr = max(float(pv_inputs.get("pv_hr", 0.0)), 0.0)
            return pv_hr * max(days_count, 1.0)

    if ref_load <= 0:
        st.info("Reference month has no load data. Savings calculator disabled.")
    else:
        per_pv_energy = pv_energy_per_kwp(savings_state["reference"], ref_days)
        daylight_fraction = pv_only_max_pct / 100.0
        daylight_energy_cap = ref_load * daylight_fraction

        def simple_savings_target(pct: float, *, extended: bool = False) -> tuple[bool, dict[str, float] | str]:
            if pct <= 0:
                return False, "Target % must be positive."
            if pct > 100:
                return False, "Savings target cannot exceed 100% of the monthly bill."
            if savings_state["system"] == "PV" and pct > pv_only_max_pct + 0.5:
                return False, f"PV-only system can save at most {pv_only_max_pct:.0f}% for this month."
            target_energy = ref_load * (pct / 100.0)
            if per_pv_energy <= 0:
                return False, "Reference month has no PV energy to scale from."
            pv_kwp = target_energy / per_pv_energy
            if savings_state["system"] == "PV":
                ess_mwh = None
            else:
                ess_energy_month = max(0.0, target_energy - min(target_energy, daylight_energy_cap))
                ess_mwh = (ess_energy_month / max(ref_days, 1.0)) / 1000.0 if ess_energy_month > 0 else 0.0
            pcs_kw = pv_kwp
            savings_idr = target_energy * savings_state["price_idr"]
            result = {
                "pv_kwp": pv_kwp,
                "ess_mwh": ess_mwh,
                "pcs_kw": pcs_kw,
                "savings_idr": savings_idr,
                "target_energy": target_energy,
            }
            if extended:
                result["per_pv_energy"] = per_pv_energy
                result["daylight_energy_cap"] = daylight_energy_cap
            return True, result

        def advanced_savings_target(pct: float) -> tuple[bool, dict[str, float] | str]:
            overview_df = st.session_state.get("finance_overview_df")
            base_ts = st.session_state.get("finance_ts_col")
            if overview_df is None or base_ts is None:
                return False, "Run a simulation first before using Advanced mode."
            simple_ok, base_plan = simple_savings_target(pct, extended=True)
            if not simple_ok:
                return simple_ok, base_plan
            harvest_series = st.session_state.get("nasa_harvest_series")
            pr_val = float(st.session_state.get("nasa_fetch", {}).get("pr_pct", 80.0)) / 100.0
            target_fraction = pct / 100.0
            pv_candidate = max(base_plan["pv_kwp"], 1.0)
            if savings_state["system"] == "PV + ESS":
                ess_candidate = base_plan["ess_mwh"] if base_plan["ess_mwh"] not in (None, 0.0) else 0.1
            else:
                ess_candidate = 0.0
            pcs_candidate = max(base_plan["pcs_kw"], pv_candidate)

            def run_once(pv_kwp: float, ess_mwh: float) -> float:
                inp = Inputs(
                    pv_kwp=pv_kwp,
                    pv_hr=float(pv_inputs["pv_hr"]),
                    sunrise_h=to_hour_float(pv_inputs["sunrise"]),
                    sunset_h=to_hour_float(pv_inputs["sunset"]),
                    batt_mwh=ess_mwh if savings_state["system"] == "PV + ESS" else 0.0,
                    soc_max_pct=int(pv_inputs["soc_max_pct"]),
                    soc_min_pct=int(pv_inputs["soc_min_pct"]),
                    init_soc_pct=int(pv_inputs["init_soc_pct"]),
                    rt_eff=float(pv_inputs["rt_eff_pct"]) / 100.0,
                )
                override = None
                if pv_data_source == NASA_SOURCE and harvest_series is not None:
                    override = pv_kwp * pr_val * harvest_series
                sim_df, _ = simulate_dispatch(overview_df, base_ts, inp, harvest_override=override)
                load_total = float(sim_df["load_kWh"].sum())
                clean = float(sim_df["harvest_used_kWh"].sum() + sim_df["batt_discharge_kWh"].sum())
                return clean / load_total if load_total > 0 else 0.0

            best = None
            with st.spinner("Running advanced solver..."):
                for _ in range(8):
                    achieved = run_once(pv_candidate, ess_candidate)
                    best = (pv_candidate, ess_candidate, achieved)
                    if achieved >= target_fraction - 0.01:
                        break
                    scale = target_fraction / max(achieved, 1e-3)
                    scale = min(max(scale, 1.05), 1.8)
                    pv_candidate *= scale
                    if savings_state["system"] == "PV + ESS":
                        ess_candidate = max(ess_candidate * scale, 0.1)
            if best is None:
                return False, "Unable to evaluate advanced solver."
            pv_final, ess_final, achieved = best
            result = {
                "pv_kwp": pv_final,
                "ess_mwh": ess_final if savings_state["system"] == "PV + ESS" else None,
                "pcs_kw": max(pv_final, pcs_candidate),
                "savings_idr": ref_load * achieved * savings_state["price_idr"],
                "achieved_pct": achieved * 100.0,
            }
            return True, result

        pv_sidebar_defaults = st.session_state.get("pv_sidebar_defaults", {})

        for idx, option in enumerate(savings_state["options"]):
            option.setdefault("computed_state", None)
            option.setdefault("achieved_pct", None)
            cols = st.columns([1.2, 0.8, 1.1, 1.1, 1.1, 0.8, 0.9], vertical_alignment="bottom")
            with cols[0]:
                option["pct"] = st.number_input(
                    f"Target {idx+1} (%)",
                    min_value=0.0,
                    value=float(option.get("pct", 0.0)),
                    step=5.0,
                    format="%.0f",
                    key=f"finance_savings_pct_{idx}",
                )
            with cols[1]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                calc_state = (calc_mode, option["pct"])
                calc_disabled = option.get("computed_state") == calc_state
                calc_clicked = st.button("Calculate", key=f"finance_savings_calc_{idx}", disabled=calc_disabled)
                if calc_clicked:
                    if calc_mode == "Advanced":
                        ok, result = advanced_savings_target(option["pct"])
                    else:
                        ok, result = simple_savings_target(option["pct"])
                    if ok:
                        option.update(result)
                        option["computed_state"] = calc_state
                        option["achieved_pct"] = result.get("achieved_pct") if calc_mode == "Advanced" else None
                        option["error"] = ""
                    else:
                        option["error"] = result
                    st.rerun()
            pv_val = option.get("pv_kwp")
            ess_val = option.get("ess_mwh")
            pcs_val = option.get("pcs_kw")
            save_val = option.get("savings_idr")
            with cols[2]:
                st.write("")
                pv_display = _fmt_trim(round(pv_val), 0) if pv_val else "—"
                st.markdown(f"PV (kWp)<br><b>{pv_display}</b>", unsafe_allow_html=True)
            with cols[3]:
                st.write("")
                ess_color = "#9ea3b0" if savings_state["system"] == "PV" else COLOR_BATT
                if savings_state["system"] == "PV + ESS" and ess_val is not None:
                    ess_display = _fmt_trim(round(ess_val * 1000.0), 0)
                else:
                    ess_display = "—"
                st.markdown(f"ESS (kWh)<br><b><span style='color:{ess_color};'>{ess_display}</span></b>", unsafe_allow_html=True)
            with cols[4]:
                st.write("")
                pcs_display = _fmt_trim(round(pcs_val), 0) if pcs_val else "—"
                st.markdown(f"PCS (kW)<br><b>{pcs_display}</b>", unsafe_allow_html=True)
            with cols[5]:
                st.write("")
                st.markdown(f"Savings (IDR)<br><b>{_fmt_comma(save_val,0) if save_val else '—'}</b>", unsafe_allow_html=True)
                if calc_mode == "Advanced" and option.get("achieved_pct") is not None:
                    st.caption(f"Achieved: {option['achieved_pct']:.1f}%")
            if option.get("error"):
                st.caption(f":warning: {option['error']}")
            use_disabled = not (pv_val and pcs_val)
            with cols[6]:
                st.write("")
                if st.button("Use Output", key=f"finance_savings_use_{idx}", disabled=use_disabled):
                    if not use_disabled:
                        st.session_state["finance_apply_output"] = {
                            "pv_kwp": round(pv_val),
                            "ess_mwh": round(ess_val, 3) if ess_val is not None else None,
                            "pcs_kw": round(pcs_val),
                        }
                        st.rerun()

    st.session_state["finance_savings"] = savings_state

    st.subheader("Investment Cost")
    inv_inputs, inv_table = st.columns([1, 2])
    with inv_inputs:
        finance_defaults["pv_price_usd"] = st.number_input("PV Price (USD/kWp)", min_value=0.0, value=float(finance_defaults["pv_price_usd"]), step=10.0, format="%g", key="pv_price_usd_input")
        finance_defaults["ess_price_usd"] = st.number_input("ESS Price (USD/kWh)", min_value=0.0, value=float(finance_defaults["ess_price_usd"]), step=10.0, format="%g", key="ess_price_usd_input")
        finance_defaults["pcs_price_usd"] = st.number_input("PCS Price (USD/kW)", min_value=0.0, value=float(finance_defaults["pcs_price_usd"]), step=5.0, format="%g", key="pcs_price_usd_input")
        pcs_kw = st.number_input("PCS Capacity (kW)", min_value=0.0, value=float(finance_defaults["pcs_kw"]), step=50.0, format="%g", key="pcs_kw_input")
        finance_defaults["pcs_kw"] = pcs_kw
        finance_defaults["usd_idr"] = st.number_input("USD → IDR", min_value=1.0, value=float(finance_defaults["usd_idr"]), step=100.0, format="%g", key="usd_idr_input")
    with inv_table:
        pv_kwp = float(pv_inputs["pv_kwp"])
        ess_kwh = float(pv_inputs["batt_mwh"]) * 1000.0
        pv_amt_usd = pv_kwp * finance_defaults["pv_price_usd"]
        ess_amt_usd = ess_kwh * finance_defaults["ess_price_usd"]
        pcs_amt_usd = pcs_kw * finance_defaults["pcs_price_usd"]
        total_usd = pv_amt_usd + ess_amt_usd + pcs_amt_usd
        total_idr = total_usd * finance_defaults["usd_idr"]

        invest_df = pd.DataFrame([
            {"Item": "PV Capacity",  "Qty": _fmt_trim(pv_kwp,1),  "UoM": "kWp", "Price": _fmt_comma(finance_defaults["pv_price_usd"],0," USD/kWp"), "Amount (USD)": _fmt_comma(pv_amt_usd,0)},
            {"Item": "ESS Capacity", "Qty": _fmt_trim(ess_kwh,0), "UoM": "kWh", "Price": _fmt_comma(finance_defaults["ess_price_usd"],0," USD/kWh"), "Amount (USD)": _fmt_comma(ess_amt_usd,0)},
            {"Item": "PCS Capacity", "Qty": _fmt_trim(pcs_kw,0),  "UoM": "kW",  "Price": _fmt_comma(finance_defaults["pcs_price_usd"],0," USD/kW"), "Amount (USD)": _fmt_comma(pcs_amt_usd,0)},
        ])
        invest_df.loc[len(invest_df)] = {"Item": "Total", "Qty": "", "UoM": "", "Price": "", "Amount (USD)": _fmt_comma(total_usd,0)}
        display_df = invest_df.copy()
        index_labels = [str(i) for i in range(1, len(invest_df))] + [""]
        display_df.index = index_labels
        display_df.index.name = ""
        st.table(display_df)
        spacer_total, metric_total = st.columns([1.5, 1])
        with metric_total:
            st.metric("Total (IDR)", _fmt_comma(total_idr,0))

    st.subheader("Billing to Client")
    bill_left, bill_right = st.columns([1, 2])
    with bill_left:
        finance_defaults["pv_direct_price_idr"] = st.number_input("PV Direct Price (IDR/kWh)", min_value=0.0, value=float(finance_defaults["pv_direct_price_idr"]), step=50.0, format="%g", key="pv_direct_price_input")
        finance_defaults["ess_price_idr"] = st.number_input("ESS Output Price (IDR/kWh)", min_value=0.0, value=float(finance_defaults["ess_price_idr"]), step=50.0, format="%g", key="ess_price_input")
        pv_period = st.selectbox("PV Consumed Month", month_options, index=month_options.index(finance_defaults["pv_month"]), key="pv_period_select")
        finance_defaults["pv_month"] = pv_period
        pv_consumed = month_value(pv_period, "harvest_used_kWh")
        ess_period = st.selectbox("ESS Consumed Month", month_options, index=month_options.index(finance_defaults["ess_month"]), key="ess_period_select")
        finance_defaults["ess_month"] = ess_period
        ess_consumed = month_value(ess_period, "batt_discharge_kWh")
        finance_defaults["rental_years"] = st.number_input("Rental Tenor (years)", min_value=1, max_value=25, value=int(finance_defaults["rental_years"]), step=1, key="rental_years_input")
    rental_years = max(1, int(finance_defaults["rental_years"]))

    pv_amount = pv_consumed * finance_defaults["pv_direct_price_idr"]
    ess_amount = ess_consumed * finance_defaults["ess_price_idr"]
    rental_amount = total_idr / rental_years / 12.0
    revenue = pv_amount + ess_amount + rental_amount
    total_consumed = pv_consumed + ess_consumed
    consumed_amount = pv_amount + ess_amount
    payback_years = total_idr / revenue / 12.0 if revenue > 0 else 0.0

    billing_rows = [
        {
            "Item": "PV Direct Price",
            "Price (IDR/kWh)": _fmt_comma(finance_defaults["pv_direct_price_idr"],0),
            "Consumed / month (kWh)": _fmt_comma(pv_consumed,0),
            "Amount (IDR)": _fmt_comma(pv_amount,0),
            "Share": _fmt_percent(100 * pv_amount / revenue) if revenue > 0 else "-"
        },
        {
            "Item": "ESS Output Price",
            "Price (IDR/kWh)": _fmt_comma(finance_defaults["ess_price_idr"],0),
            "Consumed / month (kWh)": _fmt_comma(ess_consumed,0),
            "Amount (IDR)": _fmt_comma(ess_amount,0),
            "Share": _fmt_percent(100 * ess_amount / revenue) if revenue > 0 else "-"
        },
        {
            "Item": "System monthly Rental Price",
            "Price (IDR/kWh)": "-",
            "Consumed / month (kWh)": _fmt_comma(total_consumed,0),
            "Amount (IDR)": _fmt_comma(rental_amount,0),
            "Share": _fmt_percent(100 * rental_amount / revenue) if revenue > 0 else "-"
        },
        {
            "Item": "Revenue",
            "Price (IDR/kWh)": "-",
            "Consumed / month (kWh)": "-",
            "Amount (IDR)": _fmt_comma(revenue,0),
            "Share": "-"
        },
    ]
    billing_df = pd.DataFrame(billing_rows)
    billing_df.index = [str(i) for i in range(1, len(billing_df))] + [""]
    billing_df.index.name = ""
    with bill_right:
        st.table(billing_df)
        spacer, metric_col = st.columns([3, 1])
        with metric_col:
            st.metric("Pay Back Time (years)", _fmt_trim(payback_years,2))

    st.subheader("Minimum Offtake")
    min_left, min_right = st.columns([1, 2])
    with min_left:
        finance_defaults["pv_min_offtake"] = st.number_input("PV Direct Min Offtake (kWh)", min_value=0.0, value=float(finance_defaults["pv_min_offtake"]), step=1000.0, format="%g", key="pv_min_offtake_input")
        finance_defaults["ess_min_offtake"] = st.number_input("ESS Min Offtake (kWh)", min_value=0.0, value=float(finance_defaults["ess_min_offtake"]), step=1000.0, format="%g", key="ess_min_offtake_input")

    pv_min_amt = finance_defaults["pv_min_offtake"] * finance_defaults["pv_direct_price_idr"]
    ess_min_amt = finance_defaults["ess_min_offtake"] * finance_defaults["ess_price_idr"]
    min_revenue = pv_min_amt + ess_min_amt + rental_amount
    min_payback = total_idr / min_revenue / 12.0 if min_revenue > 0 else 0.0

    min_rows = [
        {"Item": "PV Direct Min Offtake", "Qty (kWh)": _fmt_comma(finance_defaults["pv_min_offtake"],0), "Amount (IDR)": _fmt_comma(pv_min_amt,0)},
        {"Item": "ESS Min Offtake", "Qty (kWh)": _fmt_comma(finance_defaults["ess_min_offtake"],0), "Amount (IDR)": _fmt_comma(ess_min_amt,0)},
        {"Item": "System monthly Rental Price", "Qty (kWh)": "-", "Amount (IDR)": _fmt_comma(rental_amount,0)},
        {"Item": "Revenue", "Qty (kWh)": "-", "Amount (IDR)": _fmt_comma(min_revenue,0)},
    ]
    min_df = pd.DataFrame(min_rows)
    min_df.index = [str(i) for i in range(1, len(min_df))] + [""]
    min_df.index.name = ""
    with min_right:
        st.table(min_df)
        spacer, metric_col = st.columns([3, 1])
        with metric_col:
            st.metric("Pay Back Time (years)", _fmt_trim(min_payback,2))


def run_pv_mode(load_file, harvest_file, data_source, pv_inputs):
    if load_file is None:
        st.info("Upload your Load Excel to begin.")
        st.stop()

    overview_raw = read_overview_from_excel(load_file["data"])
    ts_col = detect_ts_column(overview_raw)
    overview = add_calendar_columns(overview_raw, ts_col)
    st.session_state["finance_overview_df"] = overview
    st.session_state["finance_ts_col"] = ts_col
    if IRRADIANCE_COL in overview.columns:
        overview = overview.drop(columns=[IRRADIANCE_COL])
    harvest_override = None
    if data_source == "Excel Data":
        if harvest_file is None:
            st.warning("Upload Harvest Excel when using 'Excel Data' source.")
            st.stop()
        harvest_raw = read_harvest_from_excel(harvest_file["data"])
        h_ts_col = detect_ts_column(harvest_raw)
        harvest_col = detect_named_column(harvest_raw, ["harvest"])
        harvest_raw[h_ts_col] = pd.to_datetime(harvest_raw[h_ts_col])
        overview_ts = pd.to_datetime(overview[ts_col])
        align_df = pd.DataFrame({ts_col: overview_ts})
        align_df["_order"] = np.arange(len(align_df))
        merged = pd.merge(
            align_df,
            harvest_raw[[h_ts_col, harvest_col]],
            left_on=ts_col,
            right_on=h_ts_col,
            how="left"
        ).sort_values("_order")
        missing = merged[harvest_col].isna().sum()
        if missing > 0:
            st.warning(f"{missing} harvest timestamps missing; filled with 0.")
        harvest_override = merged[harvest_col].fillna(0.0).to_numpy()
    elif data_source == NASA_SOURCE:
        overview_ts = pd.to_datetime(overview[ts_col])
        load_start = overview_ts.min().date()
        load_end = overview_ts.max().date()
        nasa_start = load_start - timedelta(days=365)
        nasa_end = load_end - timedelta(days=365)
        location = nasa_defaults["location"]
        nasa_df = st.session_state.get("nasa_ghi")
        meta = st.session_state.get("nasa_meta")
        needs_fetch = True
        if meta and nasa_df is not None and not nasa_df.empty:
            if (meta.get("location") == location and
                meta.get("start") <= nasa_start and
                meta.get("end") >= nasa_end):
                needs_fetch = False
        if needs_fetch:
            try:
                with st.spinner("Fetching NASA GHI for this load period..."):
                    nasa_df = get_hourly_ghi_local(
                        location,
                        nasa_start.strftime("%Y-%m-%d"),
                        nasa_end.strftime("%Y-%m-%d")
                    )
                st.session_state["nasa_ghi"] = nasa_df
                st.session_state["nasa_meta"] = {
                    "location": location,
                    "start": nasa_start,
                    "end": nasa_end,
                    "tz": nasa_df.attrs.get("timezone", "UTC"),
                    "rows": len(nasa_df)
                }
                meta = st.session_state["nasa_meta"]
            except Exception as exc:
                st.error(f"NASA GHI fetch failed: {exc}")
                nasa_df = None
        if nasa_df is None or nasa_df.empty:
            st.warning("NASA GHI data unavailable for this load range.")
            harvest_override = None
        else:
            nasa_copy = nasa_df.copy()
            nasa_copy["timestamp"] = pd.to_datetime(nasa_copy["timestamp"])
            nasa_copy["timestamp_shifted"] = nasa_copy["timestamp"] + pd.DateOffset(years=1)
            ghi_cols = [c for c in nasa_copy.columns if c not in ("timestamp", "timestamp_shifted")]
            if not ghi_cols:
                st.error("NASA GHI data missing irradiance column.")
                st.stop()
            ghi_col = ghi_cols[0]
            nasa_copy[ghi_col] = nasa_copy[ghi_col].astype(float)
            nasa_copy.loc[nasa_copy[ghi_col] < 0, ghi_col] = 0.0  # -999 flag
            nasa_copy[IRRADIANCE_COL] = nasa_copy[ghi_col]
            nasa_copy["harvest_kWh"] = (nasa_copy[ghi_col] / 1000.0)
            align_df = pd.DataFrame({ts_col: overview_ts})
            align_df["_order"] = np.arange(len(align_df))
            merged = pd.merge(
                align_df,
                nasa_copy[["timestamp_shifted", "harvest_kWh", IRRADIANCE_COL]],
                left_on=ts_col,
                right_on="timestamp_shifted",
                how="left"
            ).sort_values("_order")
            missing = merged["harvest_kWh"].isna().sum()
            if missing > 0:
                st.session_state["nasa_missing_warning"] = f"{missing} NASA GHI timestamps missing; filled with 0."
            harvest_kwh = merged["harvest_kWh"].fillna(0.0)
            irradiance_vals = merged[IRRADIANCE_COL].fillna(0.0)
            pr = float(nasa_defaults.get("pr_pct", 80.0)) / 100.0
            harvest_override = (float(pv_inputs["pv_kwp"]) * pr * harvest_kwh).to_numpy(dtype=float)
            overview[IRRADIANCE_COL] = irradiance_vals.to_numpy(dtype=float)
            aligned_nasa = align_df[[ts_col]].copy()
            aligned_nasa["harvest_kWh_base"] = harvest_kwh
            st.session_state["nasa_harvest_aligned"] = aligned_nasa
            st.session_state["nasa_aligned_ts_col"] = ts_col
            st.session_state["nasa_harvest_series"] = harvest_kwh.to_numpy(dtype=float)
    else:
        st.session_state["nasa_harvest_aligned"] = None
        st.session_state["nasa_aligned_ts_col"] = None
        st.session_state["nasa_harvest_series"] = None
    inp = Inputs(
        pv_kwp=float(pv_inputs["pv_kwp"]),
        pv_hr=float(pv_inputs["pv_hr"]),
        sunrise_h=to_hour_float(pv_inputs["sunrise"]),
        sunset_h=to_hour_float(pv_inputs["sunset"]),
        batt_mwh=float(pv_inputs["batt_mwh"]),
        soc_max_pct=int(pv_inputs["soc_max_pct"]),
        soc_min_pct=int(pv_inputs["soc_min_pct"]),
        init_soc_pct=int(pv_inputs["init_soc_pct"]),
        rt_eff=float(pv_inputs["rt_eff_pct"]) / 100.0,
    )

    sim_all, _ = simulate_dispatch(overview, ts_col, inp, harvest_override=harvest_override)
    st.session_state["sim_run_pv_kwp"] = inp.pv_kwp
    st.session_state["sim_run_ess_mwh"] = inp.batt_mwh

    weeks = week_list(sim_all)
    months = month_list(sim_all)
    years = year_list(sim_all)
    days = day_list(sim_all, ts_col)
    if not weeks:
        st.warning("No week data found in 'Overview'.")
        st.stop()
    if not years:
        st.warning("No year data found in 'Overview'.")
        st.stop()

    if "day_idx" not in st.session_state:
        st.session_state.day_idx = 0
    if "week_idx" not in st.session_state:
        st.session_state.week_idx = 0
    if "month_idx" not in st.session_state:
        st.session_state.month_idx = 0
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "Week"
    if "year_idx" not in st.session_state:
        st.session_state.year_idx = 0
    VIEW_OPTIONS = ["Overview", "Energy Chart", "Energy Cost Comparison", "Finance"]
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = VIEW_OPTIONS[0]

    timeframe_options = ["Day", "Week", "Month", "Year"]
    if days:
        st.session_state.day_idx = min(st.session_state.day_idx, len(days)-1)
    else:
        st.session_state.day_idx = 0
    day_options = [datetime.combine(d, time()).strftime("%d %b %Y") for d in days]
    week_options = [f"W{w} ({y})" for y, w in weeks]
    month_options = [datetime(y, m, 1).strftime("%b %Y") for y, m in months]
    year_options = [str(y) for y in years]

    view_col, tf_col, period_col = st.columns([1, 1, 2])
    with view_col:
        view_selected = st.selectbox(
            "View",
            VIEW_OPTIONS,
            index=VIEW_OPTIONS.index(st.session_state.view_mode)
        )
    if view_selected != st.session_state.view_mode:
        st.session_state.view_mode = view_selected
        st.rerun()

    view_mode = st.session_state.view_mode
    energy_week_show_night = bool(st.session_state.get("energy_show_night_week", False))
    energy_day_show_night = bool(st.session_state.get("energy_show_night_day", False))
    overview_timeframes = ["Week", "Month", "Year"]
    if "overview_timeframe" not in st.session_state:
        st.session_state["overview_timeframe"] = "Week"

    if view_mode == "Overview":
        is_nasa_source = data_source == NASA_SOURCE
        data_options = ["Summary", "Load", "Irradiance"]
        current_data_mode = st.session_state.get("pv_overview_data_mode", "Summary")
        if current_data_mode not in data_options or (current_data_mode == "Irradiance" and not is_nasa_source):
            current_data_mode = "Summary"
            st.session_state.pv_overview_data_mode = current_data_mode
        data_key = "pv_overview_data_select"
        if data_key not in st.session_state or st.session_state[data_key] not in data_options:
            st.session_state[data_key] = current_data_mode
        if not is_nasa_source and st.session_state[data_key] == "Irradiance":
            st.session_state[data_key] = "Summary"
        with tf_col:
            data_selected = st.selectbox(
                "Data",
                data_options,
                format_func=lambda opt: f"{opt} (NASA only)" if (opt == "Irradiance" and not is_nasa_source) else opt,
                key=data_key
            )
        with period_col:
            overview_tf = st.selectbox(
                "Timeframe",
                overview_timeframes,
                index=overview_timeframes.index(st.session_state["overview_timeframe"]),
                key="overview_timeframe_select"
            )
            if overview_tf != st.session_state["overview_timeframe"]:
                st.session_state["overview_timeframe"] = overview_tf
                st.rerun()
        if data_selected == "Irradiance" and not is_nasa_source:
            st.info("Irradiance view is available only when NASA GHI data is selected.")
            st.session_state[data_key] = "Summary"
            st.session_state.pv_overview_data_mode = "Summary"
            st.rerun()
        st.session_state.pv_overview_data_mode = data_selected
        render_pv_overview(sim_all, ts_col, weeks, data_selected, is_nasa_source, timeframe=st.session_state["overview_timeframe"])
        return

    if view_mode in ("Energy Chart", "Energy Cost Comparison"):
        with tf_col:
            timeframe_selected = st.selectbox(
                "Timeframe",
                timeframe_options,
                index=timeframe_options.index(st.session_state.timeframe)
            )

        if timeframe_selected != st.session_state.timeframe:
                st.session_state.timeframe = timeframe_selected
                st.rerun()

        with period_col:
            if st.session_state.timeframe == "Day":
                if not day_options:
                    st.warning("No day data found in 'Overview'.")
                    st.stop()
                day_label = st.selectbox(
                    "Choose day",
                    day_options,
                    index=int(st.session_state.day_idx)
                )
                sel_idx = day_options.index(day_label)
                if sel_idx != st.session_state.day_idx:
                    st.session_state.day_idx = sel_idx
                    st.rerun()
            elif st.session_state.timeframe == "Week":
                selected = st.selectbox(
                    "Choose week",
                    week_options,
                    index=int(st.session_state.week_idx)
                )
                sel_idx = week_options.index(selected)
                if sel_idx != st.session_state.week_idx:
                    st.session_state.week_idx = sel_idx
                    st.rerun()
            elif st.session_state.timeframe == "Month":
                if not month_options:
                    st.warning("No month data found in 'Overview'.")
                    st.stop()
                month_label = st.selectbox(
                    "Choose month",
                    month_options,
                    index=int(st.session_state.month_idx)
                )
                sel_idx = month_options.index(month_label)
                if sel_idx != st.session_state.month_idx:
                    st.session_state.month_idx = sel_idx
                    st.rerun()
            elif st.session_state.timeframe == "Year":
                if not year_options:
                    st.warning("No year data found in 'Overview'.")
                    st.stop()
                year_label = st.selectbox(
                    "Choose year",
                    year_options,
                    index=int(st.session_state.year_idx)
                )
                sel_idx = year_options.index(year_label)
                if sel_idx != st.session_state.year_idx:
                    st.session_state.year_idx = sel_idx
                    st.rerun()
            else:
                st.selectbox("Choose period", ["Coming soon"], index=0)

    else:
        tf_col.empty()
        period_col.empty()

    if view_mode == "Finance":
        render_finance_view(sim_all, pv_inputs)
        return

    period_full: pd.DataFrame | None = None
    cost_df: pd.DataFrame | None = None
    avg_day_mwh: float | None = None
    week_avg_hr: float | None = None
    energy_show_night = False

    if st.session_state.timeframe == "Day":
        prev_clicked, next_clicked = nav_buttons(
            "nav_day",
            "◀ Prev day",
            "Next day ▶",
            st.session_state.day_idx <= 0,
            st.session_state.day_idx >= len(days)-1,
        )
        if prev_clicked:
            st.session_state.day_idx = max(0, st.session_state.day_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.day_idx = min(len(days)-1, st.session_state.day_idx + 1)
            st.rerun()

        capacity = max(float(pv_inputs["pv_kwp"]), 1e-6)
        energy_day_show_night = st.toggle(
            "Show Night",
            value=energy_day_show_night,
            key="energy_show_night_day"
        )
        energy_show_night = energy_day_show_night

        day_selected = days[st.session_state.day_idx]
        full_index = build_day_index(day_selected)
        mask = pd.to_datetime(sim_all[ts_col]).dt.date == day_selected
        day_df = sim_all[mask].copy()
        day_df = day_df.set_index(pd.to_datetime(day_df[ts_col])).sort_index()

        day_full = pd.DataFrame(index=full_index)
        for col in VALUE_COLS:
            day_full[col] = day_df[col].reindex(full_index).fillna(0.0)
        if IRRADIANCE_COL in day_df.columns:
            day_full[IRRADIANCE_COL] = day_df[IRRADIANCE_COL].reindex(full_index).fillna(0.0)
        day_full["soc_pct"] = day_df["soc_pct"].reindex(full_index).ffill().fillna(0.0)
        day_full[ts_col] = day_full.index
        period_full = day_full
        cost_df = day_full
    elif st.session_state.timeframe == "Week":
        prev_clicked, next_clicked = nav_buttons(
            "nav_week",
            "◀ Prev week",
            "Next week ▶",
            st.session_state.week_idx <= 0,
            st.session_state.week_idx >= len(weeks)-1,
        )
        if prev_clicked:
            st.session_state.week_idx = max(0, st.session_state.week_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.week_idx = min(len(weeks)-1, st.session_state.week_idx + 1)
            st.rerun()

        capacity = max(float(pv_inputs["pv_kwp"]), 1e-6)
        energy_week_show_night = st.toggle(
            "Show Night",
            value=energy_week_show_night,
            key="energy_show_night_week"
        )
        energy_show_night = energy_week_show_night

        year_sel, week_sel = weeks[st.session_state.week_idx]
        full_index = build_week_index(year_sel, week_sel)
        week_df = sim_all[(sim_all["year"]==year_sel) & (sim_all["week"]==week_sel)].copy()
        week_df = week_df.set_index(pd.to_datetime(week_df[ts_col])).sort_index()

        week_full = pd.DataFrame(index=full_index)
        for col in VALUE_COLS:
            week_full[col] = week_df[col].reindex(full_index).fillna(0.0)
        if IRRADIANCE_COL in week_df.columns:
            week_full[IRRADIANCE_COL] = week_df[IRRADIANCE_COL].reindex(full_index).fillna(0.0)
        week_full["soc_pct"] = week_df["soc_pct"].reindex(full_index).ffill().fillna(0.0)
        week_full[ts_col] = week_full.index
        period_full = week_full
        cost_df = week_full
    elif st.session_state.timeframe == "Month":
        prev_clicked, next_clicked = nav_buttons(
            "nav_month",
            "◀ Prev month",
            "Next month ▶",
            st.session_state.month_idx <= 0,
            st.session_state.month_idx >= len(months)-1,
        )
        if prev_clicked:
            st.session_state.month_idx = max(0, st.session_state.month_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.month_idx = min(len(months)-1, st.session_state.month_idx + 1)
            st.rerun()

        year_sel, month_sel = months[st.session_state.month_idx]
        full_index = build_month_index(year_sel, month_sel)
        month_df = sim_all[(sim_all["year"]==year_sel) & (sim_all["month"]==month_sel)].copy()
        month_df = month_df.set_index(pd.to_datetime(month_df[ts_col])).sort_index()

        month_full = pd.DataFrame(index=full_index)
        for col in VALUE_COLS:
            month_full[col] = month_df[col].reindex(full_index).fillna(0.0)
        if IRRADIANCE_COL in month_df.columns:
            month_full[IRRADIANCE_COL] = month_df[IRRADIANCE_COL].reindex(full_index).fillna(0.0)
        month_full["soc_pct"] = month_df["soc_pct"].reindex(full_index).ffill().fillna(0.0)
        month_full[ts_col] = month_full.index
        period_full = month_full
        cost_df = month_full
        active_days = month_df.index.normalize().unique()
        if len(active_days) > 0:
            avg_day_mwh = (month_df["load_kWh"].sum() / 1000.0) / len(active_days)
    elif st.session_state.timeframe == "Year":
        prev_clicked, next_clicked = nav_buttons(
            "nav_year",
            "◀ Prev year",
            "Next year ▶",
            st.session_state.year_idx <= 0,
            st.session_state.year_idx >= len(years)-1,
        )
        if prev_clicked:
            st.session_state.year_idx = max(0, st.session_state.year_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.year_idx = min(len(years)-1, st.session_state.year_idx + 1)
            st.rerun()

        year_sel = years[st.session_state.year_idx]
        year_df = sim_all[(sim_all["year"]==year_sel)].copy()
        agg_spec = {col: "sum" for col in VALUE_COLS}
        if IRRADIANCE_COL in year_df.columns:
            agg_spec[IRRADIANCE_COL] = "sum"
        agg_spec["date"] = "nunique"
        monthly = year_df.groupby("month").agg(agg_spec)
        monthly = monthly.reindex(range(1, 13), fill_value=0.0)
        monthly = monthly.reset_index().rename(columns={"month": "month_num", "date": "days_present"})
        monthly["days_present"] = monthly["days_present"].fillna(0).astype(int)
        month_dates = pd.to_datetime({"year": year_sel, "month": monthly["month_num"], "day": 1})
        monthly[ts_col] = month_dates
        period_cols = [ts_col] + VALUE_COLS
        if IRRADIANCE_COL in monthly.columns:
            period_cols.append(IRRADIANCE_COL)
        if "days_present" in monthly.columns:
            period_cols.append("days_present")
        period_full = monthly[period_cols].copy()
        cost_df = year_df
    else:
        st.info("Year view coming soon.")
        st.stop()

    if period_full is None or period_full.empty:
        st.warning("No data available for the selected timeframe.")
        st.stop()

    chart_df = period_full.copy()
    if st.session_state.timeframe == "Month":
        daily = chart_df.copy()
        daily["_day"] = pd.to_datetime(daily[ts_col]).dt.normalize()
        agg_spec = {col: "sum" for col in VALUE_COLS}
        if IRRADIANCE_COL in daily.columns:
            agg_spec[IRRADIANCE_COL] = "sum"
        chart_df = (
            daily.groupby("_day").agg(agg_spec)
            .reset_index()
            .rename(columns={"_day": ts_col})
        )
    elif st.session_state.timeframe == "Year":
        chart_df = chart_df.sort_values(ts_col).reset_index(drop=True)

    ymax = compute_energy_ymax(chart_df)

    left, right = st.columns([4, 1], gap="large")
    with left:
        day_hr_map = None
        hr_values = None
        extra_hr = None
        hover_mode = "auto"
        capacity = max(float(pv_inputs["pv_kwp"]), 1e-6)
        if st.session_state.timeframe == "Week" and capacity > 0:
            week_ts = pd.to_datetime(chart_df[ts_col])
            grouped = chart_df.groupby(week_ts.dt.normalize())["harvest_kWh"].sum()
            day_hr_map = {pd.to_datetime(idx).strftime("%Y-%m-%d"): round(val / capacity, 1) for idx, val in grouped.items()}
            hover_mode = "week"
            valid_vals = [val for val in day_hr_map.values() if np.isfinite(val)]
            week_avg_hr = float(np.mean(valid_vals)) if valid_vals else 0.0
        elif st.session_state.timeframe == "Month":
            if capacity > 0:
                extra_hr = chart_df["harvest_kWh"].to_numpy(dtype=float) / capacity
                extra_hr = np.clip(extra_hr, 0.0, None)
            hover_mode = "month_hr"
        elif st.session_state.timeframe == "Year":
            if "days_present" in chart_df.columns:
                days_present = chart_df["days_present"].replace(0, np.nan).to_numpy(dtype=float)
                hr_values = (chart_df["harvest_kWh"].to_numpy(dtype=float) / capacity) / days_present
                hr_values = np.where(np.isfinite(hr_values), hr_values, 0.0)
            else:
                hr_values = chart_df["harvest_kWh"].to_numpy(dtype=float) / capacity
            hover_mode = "year_hr"
        is_week_tf = (st.session_state.timeframe == "Week")
        is_day_tf = (st.session_state.timeframe == "Day")
        energy_fig = unified_energy_figure(
            chart_df,
            ts_col,
            ymax,
            show_day_lines=is_week_tf,
            show_day_labels=is_week_tf,
            hover_style=hover_mode,
            day_hr_map=day_hr_map,
            hr_values=hr_values,
            extra_hr=extra_hr,
            show_night=energy_show_night,
            include_time=(is_week_tf or is_day_tf)
        )
        st.plotly_chart(energy_fig, use_container_width=True, config={"displayModeBar": True})
    with right:
        render_week_summary(st, batt_mwh=float(pv_inputs["batt_mwh"]), pv_kwp=float(pv_inputs["pv_kwp"]),
                            week_df=period_full, avg_day_mwh=avg_day_mwh)

    if view_mode == "Energy Cost Comparison":
        render_energy_cost_section(cost_df, ts_col)
        return

    if st.session_state.timeframe == "Month":
        day_ts = pd.to_datetime(chart_df[ts_col]).reset_index(drop=True)
        if capacity > 0:
            hr_daily = np.clip(chart_df["harvest_kWh"].to_numpy(dtype=float) / capacity, 0.0, None)
        else:
            hr_daily = np.zeros(len(chart_df), dtype=float)
        hr_hover = np.round(hr_daily, 2)
        day_names = day_ts.dt.strftime("%a")
        date_str = day_ts.dt.strftime("%d %b")
        hr_fig = go.Figure()
        hr_fig.add_bar(
            x=np.arange(len(hr_daily)),
            y=hr_daily,
            width=BAR_WIDTH,
            marker_color=COLOR_PV,
            showlegend=False,
            customdata=np.stack([hr_hover, day_names, date_str], axis=1),
            hovertemplate="%{customdata[0]} HR, %{customdata[1]}, %{customdata[2]}<extra></extra>"
        )
        hr_fig.update_layout(
            height=SOC_HEIGHT,
            width=FIG_WIDTH,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
        )
        hr_fig.update_yaxes(showgrid=False, showticklabels=False)
        hr_fig.update_xaxes(showticklabels=False)
        valid_mask = np.isfinite(hr_daily)
        avg_hr_month = float(hr_daily[valid_mask].mean()) if valid_mask.any() else 0.0
        irr_col, irr_info = st.columns([4, 1], gap="large")
        with irr_col:
            st.plotly_chart(hr_fig, use_container_width=True, config={"displayModeBar": False})
        with irr_info:
            render_harvest_summary(irr_info, period_full)
            avg_label = f"{avg_hr_month:.1f} HR" if valid_mask.any() else "—"
            irr_info.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            irr_info.markdown(
                f"""
                <div style="text-align:center;padding-top:4px;">
                    <div style="font-size:16px;font-weight:600;color:#555;">Avg HR</div>
                    <div style="font-size:30px;font-weight:700;color:{COLOR_PV};">{avg_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    elif st.session_state.timeframe == "Year":
        year_ts = pd.to_datetime(chart_df[ts_col]).reset_index(drop=True)
        if hr_values is not None and len(hr_values) == len(chart_df):
            hr_year = np.clip(hr_values, 0.0, None)
        elif capacity > 0:
            hr_year = np.clip(chart_df["harvest_kWh"].to_numpy(dtype=float) / capacity, 0.0, None)
        else:
            hr_year = np.zeros(len(chart_df), dtype=float)
        hr_hover = np.round(hr_year, 2)
        month_labels = year_ts.dt.strftime("%b")
        year_labels = year_ts.dt.strftime("%Y")
        hr_fig = go.Figure()
        hr_fig.add_bar(
            x=np.arange(len(hr_year)),
            y=hr_year,
            width=BAR_WIDTH,
            marker_color=COLOR_PV,
            showlegend=False,
            customdata=np.stack([hr_hover, month_labels, year_labels], axis=1),
            hovertemplate="%{customdata[0]} HR, %{customdata[1]} %{customdata[2]}<extra></extra>"
        )
        hr_fig.update_layout(
            height=SOC_HEIGHT,
            width=FIG_WIDTH,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
        )
        hr_fig.update_yaxes(showgrid=False, showticklabels=False)
        hr_fig.update_xaxes(showticklabels=False)
        irr_col, irr_info = st.columns([4, 1], gap="large")
        with irr_col:
            st.plotly_chart(hr_fig, use_container_width=True, config={"displayModeBar": False})
        with irr_info:
            render_harvest_summary(irr_info, period_full)
    elif data_source == NASA_SOURCE:
        irr_ts = pd.to_datetime(chart_df[ts_col]).reset_index(drop=True)
        if IRRADIANCE_COL in chart_df.columns:
            irr_vals = chart_df[IRRADIANCE_COL].to_numpy()
            unit_label = "Wh/m²"
            hover_vals = np.rint(np.abs(irr_vals)).astype(int)
        else:
            irr_vals = chart_df["harvest_kWh"].to_numpy()
            unit_label = "Wh/m²"
            hover_vals = np.rint(np.abs(irr_vals) * 1000).astype(int)
        irr_x = np.arange(len(irr_vals))
        day_str = irr_ts.dt.strftime("%a")
        date_str = irr_ts.dt.strftime("%d %b")
        time_str = irr_ts.dt.strftime("%H:%M")
        month_str = irr_ts.dt.strftime("%b")
        timeframe = st.session_state.timeframe
        if timeframe == "Week":
            hover_primary = time_str
            hover_secondary = date_str
            hover = f"%{{customdata[0]}} {unit_label}, %{{customdata[1]}}, %{{customdata[2]}}<extra></extra>"
            custom = np.stack([hover_vals, hover_primary, hover_secondary], axis=1)
        elif timeframe == "Month":
            hover_primary = day_str
            hover_secondary = date_str
            hover = f"%{{customdata[0]}} {unit_label}, %{{customdata[1]}}, %{{customdata[2]}}<extra></extra>"
            custom = np.stack([hover_vals, hover_primary, hover_secondary], axis=1)
        elif timeframe == "Year":
            hover_primary = month_str
            hover = f"%{{customdata[0]}} {unit_label}, %{{customdata[1]}}<extra></extra>"
            custom = np.stack([hover_vals, hover_primary], axis=1)
        else:
            hover_primary = day_str
            hover_secondary = ", " + date_str
            hover = f"%{{customdata[0]}} {unit_label}, %{{customdata[1]}}, %{{customdata[2]}}<extra></extra>"
            custom = np.stack([hover_vals, hover_primary, hover_secondary], axis=1)

        irradiance_fig = go.Figure()
        is_month_tf = (timeframe == "Month")
        trace_name = "Irradiance (Wh/m²)"
        is_year_tf = (timeframe == "Year")
        if is_month_tf or is_year_tf:
            irradiance_fig.add_trace(go.Bar(
                x=irr_x,
                y=irr_vals,
                name=trace_name,
                marker_color="#ffbf00",
                hovertemplate=hover,
                customdata=custom,
            ))
        else:
            irradiance_fig.add_trace(go.Scatter(
                x=irr_x,
                y=irr_vals,
                mode="lines",
                name=trace_name,
                line=dict(color="#ffbf00"),
                hovertemplate=hover,
                customdata=custom,
            ))
        irradiance_fig.update_layout(
            height=SOC_HEIGHT,
            width=FIG_WIDTH,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
        )
        irradiance_fig.update_yaxes(showticklabels=False, visible=False)
        irradiance_fig.update_xaxes(showticklabels=False)
        irr_col, irr_info = st.columns([4, 1], gap="large")
        with irr_col:
            st.plotly_chart(irradiance_fig, use_container_width=True, config={"displayModeBar": False})
        with irr_info:
            render_harvest_summary(irr_info, period_full)
            if st.session_state.timeframe == "Week":
                avg_label = "—"
                if week_avg_hr is not None and week_avg_hr > 0:
                    avg_label = f"{week_avg_hr:.1f} HR"
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="text-align:center;margin-top:4px;display:flex;flex-direction:column;justify-content:center;height:100%;">
                        <div style="font-size:16px;font-weight:600;color:#555;">Avg HR</div>
                        <div style="font-size:30px;font-weight:700;color:{COLOR_PV};">{avg_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    if st.session_state.timeframe in ("Week", "Day"):
        left2, right2 = st.columns([4, 1], gap="large")
        with left2:
            soc_fig = soc_bar_figure(period_full, ts_col, soc_min=int(pv_inputs["soc_min_pct"]), soc_max=int(pv_inputs["soc_max_pct"]))
            st.plotly_chart(soc_fig, use_container_width=True, config={"displayModeBar": False})
        with right2:
            st.empty()
    else:
        st.markdown(" ")

    left3, right3 = st.columns([4, 1], gap="large")
    with left3:
        spill_fig = spill_bar_figure(chart_df, ts_col, ymax_from_energy=ymax,
                                     timeframe=st.session_state.timeframe)
        st.plotly_chart(spill_fig, use_container_width=True, config={"displayModeBar": False})
    with right3:
        render_spill_summary(st, period_full)

    st.subheader("Export Data")
    export_source = period_full if st.session_state.timeframe in ("Week", "Day") else chart_df
    export_df = pd.DataFrame({
        "Timestamp": pd.to_datetime(export_source[ts_col]),
        "load": export_source["load_kWh"],
        "harvest": export_source["harvest_kWh"],
        "harvest used": export_source["harvest_used_kWh"],
        "batt discharge": export_source["batt_discharge_kWh"],
        "batt charge": export_source["batt_charge_kWh"],
        "diesel": export_source["diesel_kWh"],
        "pv spill": export_source["pv_spill_kWh"],
    })
    if st.session_state.timeframe == "Week" and "soc_pct" in export_source:
        export_df["soc"] = export_source["soc_pct"]

    excel_buf = io.BytesIO()
    export_df.to_excel(excel_buf, index=False)
    export_name = f"{st.session_state.timeframe.lower()}_overview.xlsx"
    st.download_button("Download Excel (.xlsx)", data=excel_buf.getvalue(),
                       file_name=export_name)


def run_detail_mode(detail_file):
    if detail_file is None:
        st.info("Upload your Detailed Analysis Excel to begin.")
        st.stop()

    detail_raw = read_harvest_from_excel(detail_file["data"])
    ts_col = detect_ts_column(detail_raw)
    detail = add_calendar_columns(detail_raw, ts_col)
    total_col = figure_out_total(detail)
    g_cols = [c for c in detail.columns if c.lower().startswith("g")]
    g_cols = sorted(g_cols, key=lambda c: c.lower())
    if not g_cols:
        st.error("No generator columns (G1, G2, ...) found in the uploaded file.")
        st.stop()

    days = day_list(detail, ts_col)
    weeks = week_list(detail)
    months = month_list(detail)
    years = year_list(detail)
    if not days:
        st.warning("No data found.")
        st.stop()

    if "detail_timeframe" not in st.session_state:
        st.session_state.detail_timeframe = "Week"
    if "detail_day_idx" not in st.session_state:
        st.session_state.detail_day_idx = 0
    if "detail_week_idx" not in st.session_state:
        st.session_state.detail_week_idx = 0
    if "detail_month_idx" not in st.session_state:
        st.session_state.detail_month_idx = 0
    if "detail_year_idx" not in st.session_state:
        st.session_state.detail_year_idx = 0
    if "detail_view_mode" not in st.session_state:
        st.session_state.detail_view_mode = "Overview"

    st.session_state.detail_day_idx = min(st.session_state.detail_day_idx, len(days)-1)
    if weeks:
        st.session_state.detail_week_idx = min(st.session_state.detail_week_idx, len(weeks)-1)
    if months:
        st.session_state.detail_month_idx = min(st.session_state.detail_month_idx, len(months)-1)
    if years:
        st.session_state.detail_year_idx = min(st.session_state.detail_year_idx, len(years)-1)

    detail_view_options = ["Overview", "Detail"]
    timeframe_options = ["Day","Week","Month","Year"]
    current_view = st.session_state.detail_view_mode
    if current_view not in detail_view_options:
        current_view = detail_view_options[0]
        st.session_state.detail_view_mode = current_view
    view_col, tf_col, period_col = st.columns([1, 1, 2])
    with view_col:
        view_selected = st.selectbox(
            "View",
            detail_view_options,
            index=detail_view_options.index(current_view)
        )
    if view_selected != st.session_state.detail_view_mode:
        st.session_state.detail_view_mode = view_selected
        st.rerun()

    if st.session_state.detail_view_mode == "Overview":
        render_generator_overview(detail, ts_col, g_cols, total_col)
        st.subheader("Export Data")
        export_df = detail[[ts_col] + g_cols + [total_col]].rename(columns={ts_col: "Timestamp", total_col: "Total"})
        excel_buf = io.BytesIO()
        export_df.to_excel(excel_buf, index=False)
        st.download_button("Download Excel (.xlsx)", data=excel_buf.getvalue(),
                           file_name="generator_overview.xlsx")
        return

    with tf_col:
        timeframe_selected = st.selectbox(
            "Timeframe",
            timeframe_options,
            index=timeframe_options.index(st.session_state.detail_timeframe)
        )
    if timeframe_selected != st.session_state.detail_timeframe:
        st.session_state.detail_timeframe = timeframe_selected
        st.rerun()

    period_df = None

    if st.session_state.detail_timeframe == "Day":
        day_labels = [d.strftime("%d %b %Y") for d in days]
        with period_col:
            selected = st.selectbox("Choose day", day_labels, index=int(st.session_state.detail_day_idx))
        sel_idx = day_labels.index(selected)
        if sel_idx != st.session_state.detail_day_idx:
            st.session_state.detail_day_idx = sel_idx
            st.rerun()
        prev_clicked, next_clicked = nav_buttons(
            "detail_day",
            "◀ Prev day",
            "Next day ▶",
            st.session_state.detail_day_idx <= 0,
            st.session_state.detail_day_idx >= len(days)-1,
        )
        if prev_clicked:
            st.session_state.detail_day_idx = max(0, st.session_state.detail_day_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.detail_day_idx = min(len(days)-1, st.session_state.detail_day_idx + 1)
            st.rerun()
        day_sel = days[st.session_state.detail_day_idx]
        full_index = build_day_index(day_sel)
        day_df = detail[pd.to_datetime(detail[ts_col]).dt.date == day_sel].copy()
        day_df = day_df.set_index(pd.to_datetime(day_df[ts_col])).sort_index()
        day_full = pd.DataFrame(index=full_index)
        for col in g_cols + [total_col]:
            day_full[col] = day_df[col].reindex(full_index).fillna(0.0)
        day_full[ts_col] = day_full.index
        period_df = day_full.reset_index(drop=True)

    elif st.session_state.detail_timeframe == "Week":
        if not weeks:
            st.warning("No week data found in the file.")
            st.stop()
        options = [f"W{w} ({y})" for y, w in weeks]
        with period_col:
            selected = st.selectbox("Choose week", options, index=int(st.session_state.detail_week_idx))
        sel_idx = options.index(selected)
        if sel_idx != st.session_state.detail_week_idx:
            st.session_state.detail_week_idx = sel_idx
            st.rerun()
        prev_clicked, next_clicked = nav_buttons(
            "detail_week",
            "◀ Prev week",
            "Next week ▶",
            st.session_state.detail_week_idx <= 0,
            st.session_state.detail_week_idx >= len(weeks)-1,
        )
        if prev_clicked:
            st.session_state.detail_week_idx = max(0, st.session_state.detail_week_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.detail_week_idx = min(len(weeks)-1, st.session_state.detail_week_idx + 1)
            st.rerun()
        year_sel, week_sel = weeks[st.session_state.detail_week_idx]
        full_index = build_week_index(year_sel, week_sel)
        week_df = detail[(detail["year"]==year_sel) & (detail["week"]==week_sel)].copy()
        week_df = week_df.set_index(pd.to_datetime(week_df[ts_col])).sort_index()
        week_full = pd.DataFrame(index=full_index)
        for col in g_cols + [total_col]:
            week_full[col] = week_df[col].reindex(full_index).fillna(0.0)
        week_full[ts_col] = week_full.index
        period_df = week_full.reset_index(drop=True)

    elif st.session_state.detail_timeframe == "Month":
        if not months:
            st.warning("No month data found in the file.")
            st.stop()
        month_labels = [f"{datetime(y,m,1):%b %Y}" for y,m in months]
        with period_col:
            selected = st.selectbox("Choose month", month_labels, index=int(st.session_state.detail_month_idx))
        sel_idx = month_labels.index(selected)
        if sel_idx != st.session_state.detail_month_idx:
            st.session_state.detail_month_idx = sel_idx
            st.rerun()
        prev_clicked, next_clicked = nav_buttons(
            "detail_month",
            "◀ Prev month",
            "Next month ▶",
            st.session_state.detail_month_idx <= 0,
            st.session_state.detail_month_idx >= len(months)-1,
        )
        if prev_clicked:
            st.session_state.detail_month_idx = max(0, st.session_state.detail_month_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.detail_month_idx = min(len(months)-1, st.session_state.detail_month_idx + 1)
            st.rerun()
        year_sel, month_sel = months[st.session_state.detail_month_idx]
        month_df = detail[(detail["year"]==year_sel) & (detail["month"]==month_sel)].copy()
        month_df["date_only"] = pd.to_datetime(month_df[ts_col]).dt.normalize()
        start = datetime(year_sel, month_sel, 1)
        end = (start + pd.offsets.MonthEnd(0)).to_pydatetime()
        idx = pd.date_range(start, end, freq="D")
        agg = month_df.groupby("date_only")[g_cols + [total_col]].sum().reindex(idx, fill_value=0).reset_index()
        agg = agg.rename(columns={"index": ts_col, "date_only": ts_col})
        agg[ts_col] = pd.to_datetime(agg[ts_col])
        period_df = agg

    else:  # Year
        if not years:
            st.warning("No year data found in the file.")
            st.stop()
        year_labels = [str(y) for y in years]
        with period_col:
            selected = st.selectbox("Choose year", year_labels, index=int(st.session_state.detail_year_idx))
        sel_idx = year_labels.index(selected)
        if sel_idx != st.session_state.detail_year_idx:
            st.session_state.detail_year_idx = sel_idx
            st.rerun()
        prev_clicked, next_clicked = nav_buttons(
            "detail_year",
            "◀ Prev year",
            "Next year ▶",
            st.session_state.detail_year_idx <= 0,
            st.session_state.detail_year_idx >= len(years)-1,
        )
        if prev_clicked:
            st.session_state.detail_year_idx = max(0, st.session_state.detail_year_idx - 1)
            st.rerun()
        if next_clicked:
            st.session_state.detail_year_idx = min(len(years)-1, st.session_state.detail_year_idx + 1)
            st.rerun()
        year_sel = years[st.session_state.detail_year_idx]
        year_df = detail[detail["year"]==year_sel].copy()
        monthly = (
            year_df.groupby("month")[g_cols + [total_col]].sum()
            .reindex(range(1,13), fill_value=0).reset_index()
        )
        month_dates = pd.to_datetime({"year": year_sel, "month": monthly["month"], "day": 1})
        monthly[ts_col] = month_dates
        period_df = monthly[[ts_col] + g_cols + [total_col]]

    if period_df is None or period_df.empty:
        st.warning("No data for the selected timeframe.")
        st.stop()

    hover_mode = "month" if st.session_state.detail_timeframe == "Year" else (True if st.session_state.detail_timeframe == "Month" else False)
    energy_fig = detail_energy_figure(period_df.reset_index(drop=True), ts_col, g_cols, total_col,
                                      use_day_hover=hover_mode)
    left, right = st.columns([4,1], gap="large")
    with left:
        st.plotly_chart(energy_fig, use_container_width=True, config={"displayModeBar": True})
    with right:
        render_detail_summary(st, period_df, g_cols)

    st.subheader("Export Data")
    export_df = period_df[[ts_col] + g_cols + [total_col]].rename(columns={ts_col: "Timestamp", total_col: "Total"})
    excel_buf = io.BytesIO()
    export_df.to_excel(excel_buf, index=False)
    st.download_button("Download Excel (.xlsx)", data=excel_buf.getvalue(),
                       file_name="detailed_export.xlsx")


# ---------- Sidebar (clean inputs) ----------
MODE_OPTIONS = ["PV Simulation", "Generator Analysis"]
load_file = harvest_file = detail_file = None
pv_inputs = {}
pv_sidebar_defaults = st.session_state.setdefault("pv_sidebar_defaults", {
    "pv_kwp": 500.0,
    "pv_hr": 4.5,
    "sunrise": time(6, 0),
    "sunset": time(18, 0),
    "batt_mwh": 1.0,
    "soc_max": 90,
    "soc_min": 20,
    "init_soc": 20,
    "rt_eff": 90,
})
pending_finance_apply = st.session_state.pop("finance_apply_output", None)
if pending_finance_apply:
    if "pv_kwp" in pending_finance_apply:
        pv_sidebar_defaults["pv_kwp"] = pending_finance_apply["pv_kwp"]
        st.session_state["pv_kwp_input"] = pending_finance_apply["pv_kwp"]
    if "ess_mwh" in pending_finance_apply:
        ess_value = pending_finance_apply["ess_mwh"] or 0.0
        pv_sidebar_defaults["batt_mwh"] = ess_value
        st.session_state["batt_mwh_input"] = ess_value
    finance_state = st.session_state.get("finance_inputs")
    if finance_state and "pcs_kw" in pending_finance_apply:
        finance_state["pcs_kw"] = pending_finance_apply["pcs_kw"]
        st.session_state["pcs_kw_input"] = pending_finance_apply["pcs_kw"]
st.session_state.setdefault("pv_data_source_selection", NASA_SOURCE)
st.session_state.setdefault("pv_overview_data_mode", "Summary")
nasa_defaults = st.session_state.setdefault("nasa_fetch", {
    "location": "Jakarta, Indonesia",
    "start": date.today() - timedelta(days=7),
    "end": date.today(),
    "pr_pct": 80.0,
})
if "nasa_ghi" not in st.session_state:
    st.session_state["nasa_ghi"] = None
if "nasa_meta" not in st.session_state:
    st.session_state["nasa_meta"] = None

with st.sidebar:
    mode = st.selectbox("Mode", MODE_OPTIONS, key="mode_select")
    st.markdown(SAMPLE_DL_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-top:-8px;">Upload</div>', unsafe_allow_html=True)

    if mode == "PV Simulation":
        st.markdown('<div class="upload-title" style="margin-top:4px;">Load</div>', unsafe_allow_html=True)
        _small_download_link("Download Sample", _sample_load_excel(), "sample_load.xlsx", "sample_load_dl")
        load_file = _upload_control("Load Excel (.xlsx)", "load_upload", types=["xlsx","xlsm","xls"])

        st.subheader("PV")
        data_source_options = [NASA_SOURCE, "Simple Calculation", "Excel Data"]
        current_source = st.session_state.get("pv_data_source_selection", NASA_SOURCE)
        if current_source == "Calculation":
            current_source = "Simple Calculation"
            st.session_state["pv_data_source_selection"] = current_source
        if current_source not in data_source_options:
            current_source = NASA_SOURCE
            st.session_state["pv_data_source_selection"] = current_source
        data_source = st.selectbox(
            "Data Source",
            data_source_options,
            key="pv_data_source",
            index=data_source_options.index(current_source)
        )
        st.session_state["pv_data_source_selection"] = data_source

        if data_source == "Excel Data":
            st.markdown('<div class="upload-title" style="margin-top:6px;">Harvest</div>', unsafe_allow_html=True)
            _small_download_link("Download Sample", _sample_harvest_excel(), "sample_harvest.xlsx", "sample_harvest_dl")
            harvest_file = _upload_control("Harvest Excel (.xlsx)", "harvest_upload", types=["xlsx","xlsm","xls"])
        else:
            st.session_state["harvest_upload"] = None

        pv_kwp_val = st.number_input("PV Capacity (kWp)", min_value=0.0, value=float(pv_sidebar_defaults["pv_kwp"]), step=100.0, format="%g", key="pv_kwp_input")
        pv_sidebar_defaults["pv_kwp"] = pv_kwp_val
        pv_inputs["pv_kwp"] = pv_kwp_val
        if data_source == NASA_SOURCE:
            # NASA mode ignores manual harvest ratio and daylight window
            pv_inputs["pv_hr"] = 0.0
            pv_inputs["sunrise"] = time(6, 0)
            pv_inputs["sunset"] = time(18, 0)
        else:
            pv_hr_val = st.number_input(
                "Harvest Ratio (hr/day)",
                min_value=0.0,
                value=float(pv_sidebar_defaults["pv_hr"]),
                step=0.1,
                format="%.1f",
                key="pv_hr_input",
            )
            pv_sidebar_defaults["pv_hr"] = pv_hr_val
            pv_inputs["pv_hr"] = pv_hr_val
            sunrise_val = st.time_input("Sunrise", value=pv_sidebar_defaults["sunrise"], key="sunrise_input")
            sunset_val  = st.time_input("Sunset",  value=pv_sidebar_defaults["sunset"], key="sunset_input")
            pv_sidebar_defaults["sunrise"] = sunrise_val
            pv_sidebar_defaults["sunset"] = sunset_val
            pv_inputs["sunrise"] = sunrise_val
            pv_inputs["sunset"] = sunset_val

        if data_source == NASA_SOURCE:
            st.subheader("NASA GHI", help="Data is 1 year behind.")
            missing_msg = st.session_state.get("nasa_missing_warning")
            if missing_msg:
                st.info(missing_msg)
            nasa_location = st.text_input(
                "Location / Lat,Lon",
                value=nasa_defaults["location"],
                help="Enter an address/place name or coordinates like '-6.2,106.8'."
            )
            nasa_defaults["location"] = nasa_location
            nasa_pr = st.number_input(
                "Performance Ratio (%)",
                min_value=0,
                max_value=100,
                value=int(nasa_defaults.get("pr_pct", 80.0)),
                step=1,
                format="%d",
                key="nasa_pr_input"
            )
            nasa_defaults["pr_pct"] = float(nasa_pr)
            meta = st.session_state.get("nasa_meta")
            if meta:
                st.caption(
                    f"Cached NASA GHI: {meta['location']} · {meta['start']} → {meta['end']} · {meta['tz']} · {meta['rows']} pts"
                )
                nasa_df_cached = st.session_state.get("nasa_ghi")
                if nasa_df_cached is not None and not nasa_df_cached.empty:
                    excel_buf = io.BytesIO()
                    nasa_df_cached.to_excel(excel_buf, index=False)
                    st.download_button(
                        "Download NASA GHI (.xlsx)",
                        data=excel_buf.getvalue(),
                        file_name=f"nasa_ghi_{meta['start']}_{meta['end']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="nasa_excel_download"
                    )

        st.subheader("Battery")
        batt_val = st.number_input("Battery Capacity (MWh)", min_value=0.0, value=float(pv_sidebar_defaults["batt_mwh"]), step=0.1, format="%.1f", key="batt_mwh_input")
        pv_sidebar_defaults["batt_mwh"] = batt_val
        pv_inputs["batt_mwh"] = batt_val
        soc_max_val = st.number_input("Max SoC (%)", min_value=0, max_value=100, value=int(pv_sidebar_defaults["soc_max"]), step=1, format="%d", key="soc_max_input")
        pv_sidebar_defaults["soc_max"] = soc_max_val
        pv_inputs["soc_max_pct"] = soc_max_val
        soc_min_val = st.number_input("Min SoC / Reserve (%)", min_value=0, max_value=100, value=int(pv_sidebar_defaults["soc_min"]), step=1, format="%d", key="soc_min_input")
        pv_sidebar_defaults["soc_min"] = soc_min_val
        pv_inputs["soc_min_pct"] = soc_min_val
        init_soc_val = st.number_input("Initial SoC at file start (%)", min_value=0, max_value=100, value=int(pv_sidebar_defaults["init_soc"]), step=1, format="%d", key="init_soc_input")
        pv_sidebar_defaults["init_soc"] = init_soc_val
        pv_inputs["init_soc_pct"] = init_soc_val
        rt_eff_val = st.number_input("Round-trip Efficiency (%)", min_value=50, max_value=100, value=int(pv_sidebar_defaults["rt_eff"]), step=1, format="%d", key="rt_eff_input")
        pv_sidebar_defaults["rt_eff"] = rt_eff_val
        pv_inputs["rt_eff_pct"] = rt_eff_val
        st.subheader("Energy Cost Pricing", help="Only works for Energy Cost Comparison View")
        pricing_defaults = st.session_state.setdefault("pricing_inputs", {
            "mode": "Peak / Off-Peak",
            "detail": "Simple",
            "flat_price": 1200.0,
            "peak_price": 1500.0,
            "offpeak_price": 1000.0,
            "pv_discount": 15.0,
            "ess_discount": 20.0,
            "pv_price": 800.0,
            "ess_price": 900.0,
        })
        pricing_mode = st.radio("Tariff Structure", ["Flat Price", "Peak / Off-Peak"],
                                index=0 if pricing_defaults["mode"] == "Flat Price" else 1,
                                key="pricing_sidebar_mode")
        pricing_defaults["mode"] = pricing_mode
        detail_mode = st.radio("Pricing Detail", ["Simple", "Advanced"],
                               index=0 if pricing_defaults["detail"] == "Simple" else 1,
                               key="pricing_sidebar_detail")
        pricing_defaults["detail"] = detail_mode
        if pricing_mode == "Flat Price":
            flat_price = st.number_input("Flat Price (IDR/kWh)", min_value=0.0,
                                          value=float(pricing_defaults["flat_price"]),
                                          step=50.0, format="%0.0f", key="pricing_flat_price")
            pricing_defaults["flat_price"] = flat_price
            pricing_defaults["peak_price"] = flat_price
            pricing_defaults["offpeak_price"] = flat_price
        else:
            peak_price = st.number_input("Peak Price (IDR/kWh)", min_value=0.0,
                                          value=float(pricing_defaults["peak_price"]),
                                          step=50.0, format="%0.0f", key="pricing_peak_price")
            offpeak_price = st.number_input("Off-Peak Price (IDR/kWh)", min_value=0.0,
                                             value=float(pricing_defaults["offpeak_price"]),
                                             step=50.0, format="%0.0f", key="pricing_offpeak_price")
            pricing_defaults["peak_price"] = peak_price
            pricing_defaults["offpeak_price"] = offpeak_price
        pv_discount = st.number_input("PV Discount (%)", min_value=0.0, max_value=100.0,
                                      value=float(pricing_defaults["pv_discount"]),
                                      step=1.0, format="%0.0f", key="pricing_pv_discount")
        ess_discount = st.number_input("ESS Discount (%)", min_value=0.0, max_value=100.0,
                                       value=float(pricing_defaults["ess_discount"]),
                                       step=1.0, format="%0.0f", key="pricing_ess_discount")
        pricing_defaults["pv_discount"] = pv_discount
        pricing_defaults["ess_discount"] = ess_discount
        if detail_mode == "Advanced":
            pv_price = st.number_input("PV Direct Price (IDR/kWh)", min_value=0.0,
                                       value=float(pricing_defaults["pv_price"]),
                                       step=50.0, format="%0.0f", key="pricing_pv_price")
            ess_price = st.number_input("ESS Price (IDR/kWh)", min_value=0.0,
                                        value=float(pricing_defaults["ess_price"]),
                                        step=50.0, format="%0.0f", key="pricing_ess_price")
            pricing_defaults["pv_price"] = pv_price
            pricing_defaults["ess_price"] = ess_price
        st.session_state["pricing_inputs"] = pricing_defaults

    else:
        st.markdown('<div class="upload-title" style="margin-top:4px;">Load</div>', unsafe_allow_html=True)
        _small_download_link("Download Sample", _sample_detailed_excel(), "sample_detailed.xlsx", "sample_detail_dl")
        detail_file = _upload_control("Generator Excel (.xlsx)", "detail_upload", types=["xlsx","xlsm","xls"])


# ---------- Main ----------
if mode == "PV Simulation":
    run_pv_mode(load_file, harvest_file, data_source, pv_inputs)
else:
    run_detail_mode(detail_file)
