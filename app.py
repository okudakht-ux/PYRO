"""
PYRO — AI-Powered Wildfire Situational Awareness
IBM AI Builders Challenge August 2026 · Advance Space Exploration with AI

Transforms real NASA FIRMS satellite thermal-anomaly detections into clear,
explainable wildfire intelligence.
"""

import os
import math
import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PYRO · Wildfire Situational Awareness",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — polished dark-accent theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- global ---- */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* ---- headings ---- */
    h1, h2, h3, h4 { color: #e6edf3; }
    /* ---- metric cards ---- */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label { color: #8b949e; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ff6b35; font-size: 26px; font-weight: 700;
    }
    /* ---- info / warning boxes ---- */
    .pyro-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .pyro-card-accent {
        border-left: 4px solid #ff6b35;
    }
    .pyro-card-blue {
        border-left: 4px solid #1f6feb;
    }
    .pyro-card-yellow {
        border-left: 4px solid #d29922;
    }
    .pyro-card-red {
        border-left: 4px solid #da3633;
    }
    .pyro-card-green {
        border-left: 4px solid #3fb950;
    }
    .pyro-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .pyro-value {
        font-size: 14px;
        color: #e6edf3;
        line-height: 1.6;
    }
    /* ---- disclaimer ---- */
    .disclaimer {
        background: #1c1107;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 12px;
        color: #d29922;
        margin-top: 8px;
    }
    /* ---- source badge ---- */
    .source-badge {
        background: #0d419d22;
        border: 1px solid #1f6feb;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 12px;
        color: #58a6ff;
        display: inline-block;
        margin-bottom: 8px;
    }
    /* ---- table ---- */
    [data-testid="stDataFrame"] { font-size: 13px; }
    /* ---- divider ---- */
    hr { border-color: #30363d; }
    /* ---- selectbox / slider labels ---- */
    label[data-testid="stWidgetLabel"] { color: #8b949e; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "nasa_firms.csv")

# ── Demo regions ─────────────────────────────────────────────────────────────
# Each entry: (display_label, centre_lat, centre_lon, zoom, lat_min, lat_max, lon_min, lon_max)
# bbox is used to spatially pre-filter detections for the selected region.
DEMO_REGIONS: dict[str, dict] = {
    "United States": {
        "label": "United States",
        "lat": 39.5, "lon": -98.35, "zoom": 4,
        "bbox": (17.0, 72.0, -168.0, -64.0),   # (lat_min, lat_max, lon_min, lon_max)
    },
    "California": {
        "label": "California",
        "lat": 37.5, "lon": -119.5, "zoom": 6,
        "bbox": (32.5, 42.0, -124.5, -114.0),
    },
    "Sonoma County / Timber Cove": {
        "label": "Sonoma County / Timber Cove",
        "lat": 38.55, "lon": -123.30, "zoom": 10,
        "bbox": (38.1, 39.0, -123.9, -122.4),
    },
}

# Confidence normalisation: VIIRS uses "low"/"nominal"/"high" strings;
# MODIS uses integer 0-100.
CONFIDENCE_ORDER = {"low": 0, "nominal": 50, "high": 100}

CONFIDENCE_COLOUR = {
    # numeric bucket
    "high": "#da3633",
    "nominal": "#ff6b35",
    "low": "#d29922",
    "unknown": "#8b949e",
}

# Standard FIRMS column aliases — maps possible column names → canonical name
# Supports both MODIS (brightness / bright_t31) and VIIRS (bright_ti4 / bright_ti5)
COL_ALIASES: dict[str, str] = {
    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "acq_date": "acq_date",
    "acquisition_date": "acq_date",
    "acq_time": "acq_time",
    "acquisition_time": "acq_time",
    "satellite": "satellite",
    "instrument": "instrument",
    "confidence": "confidence",
    # MODIS band names
    "brightness": "brightness",
    "bright_t31": "bright_t31",
    # VIIRS 375m band names (NOAA-20, SNPP)
    "bright_ti4": "bright_ti4",
    "bright_ti5": "bright_ti5",
    "frp": "frp",
    "daynight": "daynight",
    "scan": "scan",
    "track": "track",
    "version": "version",
    "type": "fire_type",
}

# Human-readable satellite labels
SATELLITE_LABELS: dict[str, str] = {
    "N20": "NOAA-20 (VIIRS)",
    "N21": "NOAA-21 (VIIRS)",
    "SNPP": "Suomi NPP (VIIRS)",
    "Terra": "Terra (MODIS)",
    "Aqua": "Aqua (MODIS)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _na(value) -> str:
    """Return 'Not available' for missing/NaN values, otherwise str(value)."""
    if value is None:
        return "Not available"
    try:
        if math.isnan(float(value)):
            return "Not available"
    except (ValueError, TypeError):
        pass
    v = str(value).strip()
    return "Not available" if v in ("", "nan", "NaN", "None", "none") else v


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and apply canonical aliases."""
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {k: v for k, v in COL_ALIASES.items() if k in df.columns}
    return df.rename(columns=rename_map)


def confidence_to_bucket(val) -> str:
    """Convert any confidence representation to high/nominal/low/unknown."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "unknown"
    s = str(val).strip().lower()
    if s in ("high", "nominal", "low"):
        return s
    try:
        n = float(s)
        if n >= 80:
            return "high"
        if n >= 30:
            return "nominal"
        return "low"
    except ValueError:
        return "unknown"


def confidence_colour(bucket: str) -> str:
    return CONFIDENCE_COLOUR.get(bucket, CONFIDENCE_COLOUR["unknown"])


def frp_severity(frp_val) -> str:
    """Return a plain-English severity label for Fire Radiative Power."""
    try:
        frp = float(frp_val)
    except (ValueError, TypeError):
        return "undetermined"
    if frp >= 500:
        return "extreme (≥ 500 MW)"
    if frp >= 100:
        return "very high (100–499 MW)"
    if frp >= 50:
        return "high (50–99 MW)"
    if frp >= 10:
        return "moderate (10–49 MW)"
    return "low (< 10 MW)"


def format_time(raw_time) -> str:
    """Convert HHMM integer/string to HH:MM UTC."""
    s = _na(raw_time)
    if s == "Not available":
        return s
    t = str(s).zfill(4)
    if len(t) >= 4:
        return f"{t[:2]}:{t[2:4]} UTC"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)
    # Ensure required columns exist
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            st.error(
                f"❌ The CSV file is missing a required column: `{col}`. "
                "Please check that this is a valid NASA FIRMS export."
            )
            st.stop()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    # Confidence bucket
    if "confidence" in df.columns:
        df["conf_bucket"] = df["confidence"].apply(confidence_to_bucket)
    else:
        df["conf_bucket"] = "unknown"
    # Parse date if present
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Explainable Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def build_intelligence(row: pd.Series) -> dict[str, str]:
    """
    Build the explainable intelligence narrative for a single detection row.
    Never invents missing values — uses _na() for every field access.
    """
    lat = _na(row.get("latitude"))
    lon = _na(row.get("longitude"))
    sat_raw = _na(row.get("satellite"))
    sat = SATELLITE_LABELS.get(sat_raw, sat_raw)  # expand N20 → NOAA-20 (VIIRS) etc.
    inst = _na(row.get("instrument"))
    date = _na(row.get("acq_date"))
    time_raw = row.get("acq_time")
    acq_time = format_time(time_raw)
    conf_raw = _na(row.get("confidence"))
    conf_bucket = row.get("conf_bucket", "unknown")
    frp_raw = _na(row.get("frp"))
    # Support both MODIS (brightness/bright_t31) and VIIRS (bright_ti4/bright_ti5).
    # Use _na() to test: if the MODIS column exists but is NaN, fall back to VIIRS column.
    _b_mod = row.get("brightness")
    _b_vii = row.get("bright_ti4")
    _bg_mod = row.get("bright_t31")
    _bg_vii = row.get("bright_ti5")
    brightness = _na(_b_mod) if _na(_b_mod) != "Not available" else _na(_b_vii)
    bright_t31 = _na(_bg_mod) if _na(_bg_mod) != "Not available" else _na(_bg_vii)
    # Determine band labels based on which columns are present in the row
    is_viirs = _na(_b_vii) != "Not available" or ("bright_ti4" in row.index and _na(_b_mod) == "Not available")
    band_fire_label = "I4 (~3.74 µm, fire)" if is_viirs else "Band 21/22"
    band_bg_label = "I5 (~11.45 µm, background)" if is_viirs else "Band 31"
    daynight = _na(row.get("daynight"))
    scan = _na(row.get("scan"))
    track = _na(row.get("track"))

    # ── What was detected ────────────────────────────────────────────────────
    sensor_str = (
        f"{inst} aboard {sat}"
        if sat != "Not available" and inst != "Not available"
        else sat if sat != "Not available"
        else inst if inst != "Not available"
        else "A NASA satellite"
    )
    dn_str = ""
    if daynight not in ("Not available", ""):
        dn_str = " during a **daytime**" if daynight.upper() == "D" else " during a **nighttime**"
    detected_text = (
        f"{sensor_str} recorded a thermal anomaly{dn_str} pass at "
        f"**{lat}°, {lon}°** on **{date}** at **{acq_time}**. "
        "This detection indicates a significant surface temperature elevation "
        "consistent with an active fire or heat-generating event."
    )

    # ── Why it may matter ────────────────────────────────────────────────────
    frp_label = frp_severity(frp_raw) if frp_raw != "Not available" else None
    conf_map = {"high": "high", "nominal": "moderate", "low": "low", "unknown": "undetermined"}
    conf_label = conf_map.get(conf_bucket, "undetermined")

    matter_parts = []
    if frp_label and frp_label != "undetermined":
        matter_parts.append(
            f"The Fire Radiative Power reading is **{frp_label}**, "
            "indicating the energy intensity of the fire front at the time of overpass."
        )
    if conf_bucket in ("high", "nominal"):
        matter_parts.append(
            f"Algorithm confidence is **{conf_label}**, suggesting this detection "
            "is unlikely to be a false alarm under standard conditions."
        )
    elif conf_bucket == "low":
        matter_parts.append(
            "Algorithm confidence is **low** — this detection warrants verification "
            "against additional sources before operational decisions are made."
        )
    if not matter_parts:
        matter_parts.append(
            "This detection represents a thermal anomaly identified during satellite overpass. "
            "Ground-truth verification is recommended."
        )
    matter_text = " ".join(matter_parts)

    # ── What the data tells us ────────────────────────────────────────────────
    data_parts = []
    if brightness != "Not available":
        data_parts.append(f"Brightness temperature ({band_fire_label}): **{brightness} K**")
    if bright_t31 != "Not available":
        data_parts.append(f"Background brightness temperature ({band_bg_label}): **{bright_t31} K**")
    if frp_raw != "Not available":
        data_parts.append(f"Fire Radiative Power: **{frp_raw} MW**")
    if scan != "Not available" and track != "Not available":
        data_parts.append(
            f"Pixel spatial resolution: **{scan} km × {track} km** "
            "(larger pixels occur at wider scan angles)"
        )
    if conf_raw != "Not available":
        data_parts.append(f"Raw confidence value: **{conf_raw}**")
    if not data_parts:
        data_parts.append(
            "Geospatial coordinates have been confirmed. "
            "Additional radiometric fields are not available in this dataset."
        )
    data_text = "\n\n".join(f"- {p}" for p in data_parts)

    # ── What remains uncertain ────────────────────────────────────────────────
    uncertain_parts = [
        "Satellite thermal sensors detect radiant heat at pixel scale "
        f"({scan + ' km × ' + track + ' km' if scan != 'Not available' and track != 'Not available' else 'multiple km²'}). "
        "Actual fire extent and behaviour cannot be confirmed from this detection alone.",
        "Cloud cover, smoke, or atmospheric interference may have attenuated the signal, "
        "causing the true intensity to be higher or lower than reported.",
        "The detection represents a single overpass snapshot. Fire behaviour — spread rate, "
        "direction, and intensity — may have changed significantly since acquisition.",
    ]
    if conf_bucket == "low":
        uncertain_parts.append(
            "Low-confidence detections have elevated false-positive rates and may represent "
            "industrial heat sources, gas flares, volcanic activity, or sensor artefacts."
        )
    uncertain_text = "\n\n".join(f"- {p}" for p in uncertain_parts)

    # ── Environmental factors to monitor ─────────────────────────────────────
    env_text = (
        "- **Wind speed and direction** — drives fire spread and ember transport\n\n"
        "- **Relative humidity** — values below 20% significantly increase fire intensity\n\n"
        "- **Temperature** — elevated ambient temperatures accelerate fuel drying\n\n"
        "- **Fuel moisture content** — dry vegetation dramatically lowers ignition threshold\n\n"
        "- **Terrain slope** — upslope fires spread faster and are harder to suppress\n\n"
        "- **Proximity to structures, roads, and water sources** — critical for resource dispatch\n\n"
        "- **Repeat satellite passes** — monitor subsequent overpasses to assess fire progression"
    )

    return {
        "detected": detected_text,
        "matter": matter_text,
        "data": data_text,
        "uncertain": uncertain_text,
        "env": env_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Map builder
# ─────────────────────────────────────────────────────────────────────────────

def build_map(
    df: pd.DataFrame,
    selected_idx=None,
    centre_lat: float | None = None,
    centre_lon: float | None = None,
    zoom: int = 4,
) -> folium.Map:
    # Use caller-supplied centre/zoom (from demo region), else derive from data
    if centre_lat is None:
        centre_lat = float(df["latitude"].mean()) if not df.empty else 39.5
    if centre_lon is None:
        centre_lon = float(df["longitude"].mean()) if not df.empty else -98.35

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",   # free, no API key required
        control_scale=True,
    )

    # Cluster / individual markers
    for idx, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        bucket = row.get("conf_bucket", "unknown")
        colour = confidence_colour(bucket)
        radius = 6

        frp_val = _na(row.get("frp"))
        if frp_val != "Not available":
            try:
                radius = min(18, max(5, int(float(frp_val) / 30) + 5))
            except (ValueError, TypeError):
                pass

        is_selected = selected_idx is not None and idx == selected_idx
        fill_opacity = 0.95 if is_selected else 0.75
        stroke_weight = 3 if is_selected else 1

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;min-width:180px'>"
            f"<b style='color:#ff6b35'>Hotspot #{idx}</b><br>"
            f"<b>Lat / Lon:</b> {lat:.4f}, {lon:.4f}<br>"
            f"<b>Date:</b> {_na(row.get('acq_date'))}<br>"
            f"<b>Satellite:</b> {_na(row.get('satellite'))}<br>"
            f"<b>Instrument:</b> {_na(row.get('instrument'))}<br>"
            f"<b>Confidence:</b> {_na(row.get('confidence'))}<br>"
            f"<b>FRP:</b> {_na(row.get('frp'))} MW<br>"
            f"</div>"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#ffffff" if is_selected else colour,
            weight=stroke_weight,
            fill=True,
            fill_color=colour,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"#{idx} · conf: {bucket} · FRP: {frp_val} MW",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: #161b22cc; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px;
        font-family: sans-serif; font-size: 12px; color: #e6edf3;
    ">
        <b style="color:#ff6b35">PYRO</b> · Confidence<br>
        <span style="color:#da3633">●</span> High &nbsp;
        <span style="color:#ff6b35">●</span> Nominal &nbsp;
        <span style="color:#d29922">●</span> Low &nbsp;
        <span style="color:#8b949e">●</span> Unknown
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown(
        "<div style='padding:8px 0 16px'>"
        "<span style='font-size:28px'>🔥</span> "
        "<span style='font-size:20px;font-weight:700;color:#e6edf3'>PYRO</span><br>"
        "<span style='font-size:11px;color:#8b949e;letter-spacing:0.05em'>"
        "WILDFIRE SITUATIONAL AWARENESS</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='source-badge'>🛰 Source: NASA FIRMS — Fire Information for Resource Management System</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#3fb950;margin-bottom:4px;margin-top:2px'>"
        "✅ Real satellite observations — no simulated fire detections."
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    filters: dict = {}

    # ── Demo region selector ──────────────────────────────────────────────────
    st.sidebar.markdown("### 📍 Demo Region")
    region_names = list(DEMO_REGIONS.keys())
    selected_region = st.sidebar.selectbox(
        "Focus map on", region_names, index=0, key="demo_region"
    )
    filters["demo_region"] = selected_region

    st.sidebar.markdown("### Filters")

    # ── Satellite / Instrument ────────────────────────────────────────────────
    if "satellite" in df.columns:
        sats = sorted(df["satellite"].dropna().unique().tolist())
        if sats:
            selected_sats = st.sidebar.multiselect(
                "Satellite", sats, default=sats, key="filter_sat"
            )
            filters["satellite"] = selected_sats

    if "instrument" in df.columns:
        insts = sorted(df["instrument"].dropna().unique().tolist())
        if insts:
            selected_insts = st.sidebar.multiselect(
                "Instrument", insts, default=insts, key="filter_inst"
            )
            filters["instrument"] = selected_insts

    # ── Confidence ────────────────────────────────────────────────────────────
    buckets_present = sorted(
        df["conf_bucket"].unique().tolist(),
        key=lambda x: {"high": 0, "nominal": 1, "low": 2, "unknown": 3}.get(x, 4),
    )
    if len(buckets_present) > 1:
        selected_buckets = st.sidebar.multiselect(
            "Confidence level",
            buckets_present,
            default=buckets_present,
            key="filter_conf",
        )
        filters["conf_bucket"] = selected_buckets

    # ── Date range ────────────────────────────────────────────────────────────
    if "acq_date" in df.columns and df["acq_date"].notna().any():
        min_date = df["acq_date"].min().date()
        max_date = df["acq_date"].max().date()
        if min_date != max_date:
            date_range = st.sidebar.date_input(
                "Acquisition date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="filter_date",
            )
            filters["date_range"] = date_range

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='disclaimer'>"
        "⚠️ <b>DISCLAIMER</b><br>"
        "PYRO is a research and situational-awareness prototype. "
        "Satellite thermal detections contain uncertainty. "
        "PYRO is <b>not</b> an emergency alert system and must not replace "
        "official emergency information or evacuation instructions."
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#484f58;margin-top:16px;text-align:center'>"
        "IBM AI Builders Challenge · August 2026<br>"
        "Advance Space Exploration with AI"
        "</div>",
        unsafe_allow_html=True,
    )

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    dff = df.copy()
    if "satellite" in filters and filters["satellite"]:
        dff = dff[dff["satellite"].isin(filters["satellite"])]
    if "instrument" in filters and filters["instrument"]:
        dff = dff[dff["instrument"].isin(filters["instrument"])]
    if "conf_bucket" in filters and filters["conf_bucket"]:
        dff = dff[dff["conf_bucket"].isin(filters["conf_bucket"])]
    if "date_range" in filters:
        dr = filters["date_range"]
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            start, end = dr
            mask = (dff["acq_date"].dt.date >= start) & (dff["acq_date"].dt.date <= end)
            dff = dff[mask]
    # ── Spatial filter for demo region ────────────────────────────────────────
    region_name = filters.get("demo_region", "United States")
    region = DEMO_REGIONS.get(region_name, DEMO_REGIONS["United States"])
    lat_min, lat_max, lon_min, lon_max = region["bbox"]
    dff = dff[
        (dff["latitude"] >= lat_min) & (dff["latitude"] <= lat_max) &
        (dff["longitude"] >= lon_min) & (dff["longitude"] <= lon_max)
    ]
    return dff


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Panel renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_intelligence_panel(row: pd.Series, idx: int):
    intel = build_intelligence(row)
    st.markdown(
        f"<div style='margin-bottom:6px'>"
        f"<span style='font-size:18px;font-weight:700;color:#e6edf3'>"
        f"🔥 Hotspot #{idx} — Explainable Intelligence"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    cards = [
        ("blue", "🛰 What the Satellite Detected", "detected"),
        ("accent", "⚡ Why This Detection May Matter", "matter"),
        ("green", "📊 What the Available Data Tells Us", "data"),
        ("yellow", "❓ What Remains Uncertain", "uncertain"),
        ("red", "🌿 Environmental Factors to Monitor", "env"),
    ]
    for colour_key, label, key in cards:
        st.markdown(
            f"<div class='pyro-card pyro-card-{colour_key}'>"
            f"<div class='pyro-label'>{label}</div>"
            f"<div class='pyro-value'>"
            + intel[key].replace("\n\n", "<br><br>")
            + "</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Raw data fields panel
# ─────────────────────────────────────────────────────────────────────────────

def render_data_fields(row: pd.Series):
    # Build field list dynamically based on what is actually present in the row.
    # VIIRS datasets (N20, SNPP) use bright_ti4/bright_ti5; MODIS uses brightness/bright_t31.
    bright_fire_key = "bright_ti4" if "bright_ti4" in row.index else "brightness"
    bright_bg_key = "bright_ti5" if "bright_ti5" in row.index else "bright_t31"
    bright_fire_label = (
        "Brightness Temp I4 — Fire (K)" if bright_fire_key == "bright_ti4"
        else "Brightness Temp Band 21/22 (K)"
    )
    bright_bg_label = (
        "Brightness Temp I5 — Background (K)" if bright_bg_key == "bright_ti5"
        else "Background Bright. Temp Band 31 (K)"
    )

    # Satellite display with friendly name
    sat_raw = _na(row.get("satellite"))
    sat_display = SATELLITE_LABELS.get(sat_raw, sat_raw)

    field_defs = [
        ("latitude", "Latitude"),
        ("longitude", "Longitude"),
        ("acq_date", "Acquisition Date"),
        ("acq_time", "Acquisition Time (UTC)"),
        ("_satellite_display", "Satellite", sat_display),
        ("instrument", "Instrument"),
        ("confidence", "Confidence"),
        (bright_fire_key, bright_fire_label),
        (bright_bg_key, bright_bg_label),
        ("frp", "Fire Radiative Power (MW)"),
        ("daynight", "Day / Night"),
        ("scan", "Scan (km)"),
        ("track", "Track (km)"),
        ("version", "Algorithm Version"),
        ("fire_type", "Fire Type"),
    ]

    st.markdown(
        "<div class='pyro-card'><div class='pyro-label'>📋 NASA FIRMS Detection Fields</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, entry in enumerate(field_defs):
        # entry may be (col_key, label) or (col_key, label, precomputed_value)
        if len(entry) == 3:
            col_key, label, val = entry
        else:
            col_key, label = entry
            raw = row.get(col_key)
            if col_key == "acq_time":
                val = format_time(raw)
            elif col_key == "acq_date":
                val = _na(raw)
                if val != "Not available":
                    try:
                        val = pd.Timestamp(val).strftime("%Y-%m-%d")
                    except Exception:
                        pass
            else:
                val = _na(raw)

        with cols[i % 2]:
            st.markdown(
                f"<div style='margin-bottom:10px'>"
                f"<div class='pyro-label' style='margin-bottom:2px'>{label}</div>"
                f"<div class='pyro-value'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.markdown(
            "<div style='font-size:56px;line-height:1;padding-top:8px'>🔥</div>",
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown(
            "<h1 style='margin:0;font-size:36px;font-weight:800;color:#e6edf3'>"
            "PYRO"
            "<span style='font-size:16px;font-weight:400;color:#8b949e;margin-left:12px'>"
            "AI-Powered Wildfire Situational Awareness"
            "</span></h1>"
            "<div style='font-size:12px;color:#484f58;margin-top:4px'>"
            "IBM AI Builders Challenge · August 2026 · Advance Space Exploration with AI"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Data loading guard ────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        st.warning(
            "### 📂 No data file found\n\n"
            f"PYRO is ready — place your NASA FIRMS CSV at **`{DATA_PATH}`** and reload.\n\n"
            "**How to get the data:**\n"
            "1. Visit [https://firms.modaps.eosdis.nasa.gov/active_fire/]"
            "(https://firms.modaps.eosdis.nasa.gov/active_fire/)\n"
            "2. Choose your region and time range (24 h, 48 h, or 7 days)\n"
            "3. Download any satellite CSV (MODIS, VIIRS SNPP, VIIRS NOAA-20, etc.)\n"
            "4. Save as `data/nasa_firms.csv`\n\n"
            "> PYRO never generates synthetic detections. Only real NASA FIRMS data is accepted."
        )
        st.markdown(
            "<div class='disclaimer'>"
            "⚠️ <b>DISCLAIMER:</b> PYRO is a research and situational-awareness prototype. "
            "Satellite thermal detections contain uncertainty. "
            "PYRO is not an emergency alert system and must not replace official emergency "
            "information or evacuation instructions."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Loading NASA FIRMS data …"):
        df = load_data(DATA_PATH)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    filters = render_sidebar(df)
    dff = apply_filters(df, filters)

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Detections", f"{len(df):,}")
    with m2:
        st.metric("Filtered Detections", f"{len(dff):,}")
    with m3:
        high_count = int((dff["conf_bucket"] == "high").sum())
        st.metric("High Confidence", f"{high_count:,}")
    with m4:
        if "frp" in dff.columns:
            max_frp = dff["frp"].apply(pd.to_numeric, errors="coerce").max()
            st.metric("Max FRP (MW)", f"{max_frp:,.1f}" if not math.isnan(max_frp) else "N/A")
        else:
            st.metric("Max FRP (MW)", "N/A")
    with m5:
        sats = (
            dff["satellite"].nunique() if "satellite" in dff.columns else 0
        )
        st.metric("Satellites", f"{sats}")

    st.markdown("---")

    # ── Resolve demo region for map centre/zoom ───────────────────────────────
    region_name = filters.get("demo_region", "United States")
    region = DEMO_REGIONS.get(region_name, DEMO_REGIONS["United States"])

    if dff.empty:
        # Still render the map centred on the selected region with no markers,
        # and show an appropriate info message.
        st.info(
            f"No detections found for **{region['label']}** with the current filters. "
            "The map is centred on the selected region. "
            "PYRO does not fabricate detections — try adjusting the confidence or date filters."
        )
        empty_map = build_map(
            pd.DataFrame(columns=df.columns),
            centre_lat=region["lat"],
            centre_lon=region["lon"],
            zoom=region["zoom"],
        )
        st_folium(empty_map, height=400, use_container_width=True, returned_objects=[])
        return

    # ── Main layout: map + inspector ─────────────────────────────────────────
    map_col, inspect_col = st.columns([6, 4], gap="large")

    with map_col:
        st.markdown(
            f"<h3 style='margin-top:0'>🗺 Active Fire Detections — {region['label']}</h3>"
            "<div style='font-size:12px;color:#8b949e;margin-bottom:8px'>"
            "Click any marker for a quick summary · Use the inspector on the right for full analysis"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='source-badge'>"
            "🛰 Data source: NASA FIRMS — VIIRS NOAA-20 &nbsp;|&nbsp; "
            "Real satellite observations — no simulated fire detections."
            "</div>",
            unsafe_allow_html=True,
        )

        # Warn when filtered set is very large (performance)
        if len(dff) > 5000:
            st.warning(
                f"⚡ {len(dff):,} detections in view. "
                "Rendering may take a moment. Use filters to narrow down the dataset."
            )

        # Sonoma / Timber Cove: show info if no detections exist in that small bbox
        if region_name == "Sonoma County / Timber Cove" and dff.empty:
            st.info(
                "🔍 No NASA FIRMS detections found in the Sonoma County / Timber Cove region "
                "within the loaded dataset. The map is centred on Timber Cove (38.55°N, 123.30°W). "
                "PYRO does not fabricate detections — expand the date range or load a broader dataset."
            )

        selected_idx = st.session_state.get("selected_idx")
        fire_map = build_map(
            dff.head(5000),
            selected_idx=selected_idx,
            centre_lat=region["lat"],
            centre_lon=region["lon"],
            zoom=region["zoom"],
        )
        st_folium(fire_map, height=520, use_container_width=True, returned_objects=[])

    with inspect_col:
        st.markdown(
            "<h3 style='margin-top:0'>🔍 Hotspot Inspector</h3>",
            unsafe_allow_html=True,
        )
        # Row selector
        max_idx = len(dff) - 1
        display_indices = dff.index.tolist()

        # Provide a simple numeric selector (0 … N-1 position in filtered frame)
        position = st.number_input(
            f"Select detection (0 – {max_idx})",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
            key="position_input",
        )
        selected_row_idx = display_indices[position]
        st.session_state["selected_idx"] = selected_row_idx
        selected_row = dff.loc[selected_row_idx]

        render_data_fields(selected_row)

        st.markdown("---")
        render_intelligence_panel(selected_row, selected_row_idx)

    st.markdown("---")

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 View all filtered detections (table)", expanded=False):
        display_cols = [
            c for c in [
                "latitude", "longitude", "acq_date", "acq_time",
                "satellite", "instrument", "confidence",
                "brightness", "bright_t31",   # MODIS
                "bright_ti4", "bright_ti5",    # VIIRS
                "frp", "daynight", "version", "conf_bucket",
            ]
            if c in dff.columns
        ]
        st.dataframe(
            dff[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=320,
        )
        st.caption(f"{len(dff):,} detections shown · Source: NASA FIRMS")

    # ── Confidence distribution chart ─────────────────────────────────────────
    with st.expander("📊 Confidence distribution", expanded=False):
        bucket_counts = (
            dff["conf_bucket"]
            .value_counts()
            .reindex(["high", "nominal", "low", "unknown"])
            .fillna(0)
            .astype(int)
        )
        bar_colours = {
            "high": "#da3633",
            "nominal": "#ff6b35",
            "low": "#d29922",
            "unknown": "#8b949e",
        }
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=bucket_counts.index.tolist(),
                y=bucket_counts.values.tolist(),
                marker_color=[bar_colours[b] for b in bucket_counts.index],
                text=bucket_counts.values.tolist(),
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#e6edf3",
            xaxis=dict(gridcolor="#30363d", title="Confidence Level"),
            yaxis=dict(gridcolor="#30363d", title="Detections"),
            margin=dict(t=20, b=20, l=0, r=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;margin-top:32px;padding-top:16px;"
        "border-top:1px solid #30363d;font-size:12px;color:#484f58'>"
        "PYRO · IBM AI Builders Challenge August 2026 · Advance Space Exploration with AI<br>"
        "Data source: <a href='https://firms.modaps.eosdis.nasa.gov' style='color:#58a6ff'>"
        "NASA FIRMS — VIIRS NOAA-20 · Fire Information for Resource Management System</a>"
        " &nbsp;|&nbsp; Real satellite observations — no simulated fire detections.<br><br>"
        "<b style='color:#d29922'>⚠️ DISCLAIMER:</b> "
        "<span style='color:#8b949e'>"
        "PYRO is a research and situational-awareness prototype. "
        "Satellite thermal detections contain uncertainty. "
        "PYRO is not an emergency alert system and must not replace official emergency "
        "information or evacuation instructions."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
