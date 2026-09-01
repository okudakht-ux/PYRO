# 🔥 PYRO — AI-Powered Wildfire Situational Awareness

**IBM AI Builders Challenge August 2026 · Advance Space Exploration with AI**

PYRO transforms real NASA FIRMS satellite observations into clear, explainable wildfire intelligence for emergency planners, researchers, and situational-awareness teams.

---

## What It Does

- **Real NASA Data** — loads live satellite thermal-anomaly detections directly from NASA FIRMS CSV exports
- **Interactive Map** — all detected hotspots plotted on a zoomable global map, colour-coded by confidence
- **Explainable Intelligence Panel** — for every hotspot: what was detected, why it may matter, what the data tells us, what remains uncertain, and which environmental factors to monitor
- **Smart Filters** — filter by satellite/instrument, confidence level, and acquisition date
- **No Fabricated Data** — "Not available" is displayed when a value is absent; nothing is invented

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NASA FIRMS data

1. Visit [https://firms.modaps.eosdis.nasa.gov/active_fire/](https://firms.modaps.eosdis.nasa.gov/active_fire/)
2. Select your region and time range (last 24 h / 48 h / 7 days)
3. Download the **CSV** for any satellite (MODIS, VIIRS SNPP, VIIRS NOAA-20, etc.)
4. Save the file as **`data/nasa_firms.csv`**

> FIRMS CSV files are named things like `MODIS_C6_1_Global_24h.csv`,  
> `VNP14IMGTDL_NRT_Global_24h.csv`, or similar. Rename to `nasa_firms.csv`.

### 3. Run PYRO

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## NASA FIRMS CSV Column Reference

PYRO auto-detects all standard FIRMS columns. Typical columns include:

| Column | Description |
|---|---|
| `latitude` | Detection latitude (decimal degrees) |
| `longitude` | Detection longitude (decimal degrees) |
| `acq_date` | Acquisition date (YYYY-MM-DD) |
| `acq_time` | Acquisition time (HHMM UTC) |
| `satellite` | Satellite name (e.g. Terra, Aqua, SNPP, NOAA-20) |
| `instrument` | Instrument name (e.g. MODIS, VIIRS) |
| `confidence` | Detection confidence (0–100 or low/nominal/high) |
| `bright_t31` / `brightness` | Brightness temperature (K) |
| `frp` | Fire Radiative Power (MW) |
| `daynight` | Day (D) or Night (N) detection |
| `scan` / `track` | Spatial resolution of the pixel (km) |
| `version` | FIRMS algorithm version |

---

## Project Structure

```
PYRO/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── data/
    └── nasa_firms.csv   # ← Place your NASA FIRMS download here
```

---

## Disclaimer

> **PYRO is a research and situational-awareness prototype.**  
> Satellite thermal detections contain uncertainty. PYRO is **not** an emergency alert system and **must not** replace official emergency information or evacuation instructions.

---

## Data Source

**NASA FIRMS — Fire Information for Resource Management System**  
[https://firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov)  
Operated by NASA's Earth Science Data and Information System (ESDIS).

---

## IBM AI Builders Challenge

PYRO demonstrates how AI and space-based Earth observation can work together to deliver faster, clearer, and more actionable wildfire intelligence — a direct application of NASA satellite infrastructure in service of public safety and planetary stewardship.
