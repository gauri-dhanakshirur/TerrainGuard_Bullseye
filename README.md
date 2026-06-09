# 🛡️ TerrainGuard
### Controlled Flight Into Terrain (CFIT) Prevention System

> A hackathon-grade avionics + GIS pipeline that fuses satellite terrain data with real-time emergency triage to predict and prevent CFIT events.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [System Architecture](#system-architecture)
- [Mathematical Framework](#mathematical-framework)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Deep-Dive](#pipeline-deep-dive)
- [Web GIS Dashboard](#web-gis-dashboard)
- [Sample Outputs](#sample-outputs)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Extending TerrainGuard](#extending-terrainguard)
- [License & Attribution](#license--attribution)

---

## Overview

TerrainGuard is a two-engine avionics decision-support system:

| Engine | Trigger | Output |
|---|---|---|
| **Pre-Flight Engine** | Loaded before departure | Terrain Topography Complexity Index (TTCI) raster |
| **Emergency Triage Engine** | Fuel/engine emergency | Forced Landing Suitability Index (FLSI) + Top-3 safe landing vectors |

Both engines operate on aligned 500×500 WGS84 raster grids synthesised from SRTM/Copernicus DEM data and ESA WorldCover LULC classification. All computation runs in pure Python with no proprietary dependencies.

---

## The Problem

Controlled Flight Into Terrain (CFIT) accounts for approximately **25% of all fatal commercial aviation accidents** worldwide. It occurs when a fully airworthy aircraft is inadvertently flown into terrain, water, or an obstacle — typically during poor visibility, crew distraction, or navigation error.

Existing solutions (GPWS/TAWS) are **reactive** — they alert only when the aircraft is already in a collision trajectory. TerrainGuard is **proactive**: it pre-computes terrain complexity along the entire flight path and activates a multi-factor triage engine the moment an emergency is declared, giving the crew ranked, coordinate-precise forced-landing options within their unpowered glide envelope.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        pipeline.py                               │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Data        │    │ Feature      │    │ TTCI Engine        │  │
│  │ Generation  │───▶│ Extraction   │───▶│ (Pre-Flight)       │  │
│  │             │    │              │    │                    │  │
│  │ terrain_dem │    │ Slope (Horn) │    │ TTCI = 0.4·S_n     │  │
│  │ land_cover  │    │ TRI (Riley)  │    │      + 0.4·T_n     │  │
│  └─────────────┘    │ Variance     │    │      + 0.2·V_n     │  │
│                     └──────────────┘    └────────────────────┘  │
│                                                  │               │
│                                                  ▼               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Emergency Triage Engine  (FLSI)                         │    │
│  │                                                         │    │
│  │  Glide Cone Mask → Slope Gate (>15°=0) → LULC Scoring  │    │
│  │  → ndimage.label clustering → Rank components          │    │
│  │  → Dynamic MSA → avionics_output.json                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │ terrain_analysis_        │  │ survivability_           │    │
│  │ dashboard.png            │  │ correlation_chart.png    │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                        index.html                                │
│                                                                  │
│  Leaflet.js Map  ·  TTCI Canvas Overlay  ·  LULC Canvas Overlay  │
│  Flight Router (10 NM buffer)  ·  MAYDAY Triage Widget           │
│  Glide Cone  ·  Top-3 Landing Vectors  ·  Sector Radar Table     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Framework

### Feature Extraction

**Slope** (Horn 1981 — same algorithm used in ESRI ArcGIS and GDAL):

```
Slope = arctan(√((∂z/∂x)² + (∂z/∂y)²)) × 180°/π
```

where `∂z/∂x` and `∂z/∂y` are computed via Sobel kernels divided by 8 × cell size in metres.

**Terrain Ruggedness Index** (Riley et al. 1999):

```
TRI = √(Σᵢ₌₁⁸ (zᵢ - z_centre)²)
```

Applied over a 3×3 window via `scipy.ndimage.generic_filter`.

**Elevation Variance** (5×5 moving standard deviation):

```
Variance = √(E[z²] - E[z]²)
```

Computed efficiently using `scipy.ndimage.uniform_filter` on `z` and `z²`.

---

### Terrain Topography Complexity Index (TTCI)

All three features are Min-Max normalised to [0, 1]:

```
f_norm = (f - f_min) / (f_max - f_min)
```

Then composited with empirically weighted combination:

```
TTCI = 0.4 × Slope_norm + 0.4 × TRI_norm + 0.2 × Variance_norm
```

| TTCI Range | Risk Class | Colour |
|---|---|---|
| 0.0 – 0.3 | Low | 🟢 Green |
| 0.3 – 0.7 | Moderate | 🟡 Yellow |
| ≥ 0.7 | High | 🔴 Red |

---

### Forced Landing Suitability Index (FLSI)

**Step 1 — Glide Cone Mask:**

```
cone(r,c) = [(r - r_ac)² + (c - c_ac)²] ≤ R_px²
```

where `R_px = ⌊GlideRange_NM × 1852 / m_per_pixel⌋`

**Step 2 — Slope Gate:**

```
candidate(r,c) = cone(r,c) ∩ (Slope(r,c) ≤ 15°)
```

**Step 3 — LULC Suitability Scoring:**

| ESA WorldCover Class | Code | Score |
|---|---|---|
| Airport Runway | 50 | **1.0** |
| Grassland | 30 | 0.8 |
| Water | 40 | 0.3 |
| Forest | 20 | 0.1 |
| Urban | 10 | **0.0** |

```
FLSI(r,c) = candidate(r,c) × LULC_score(r,c)
```

**Step 4 — Connected Component Analysis:**

`scipy.ndimage.label` identifies contiguous clearings (FLSI > 0.5). Each component is scored and ranked:

```
score_k = mean_FLSI_k × { 1.2  if dom. class = Runway
                         { 1.0  otherwise
```

---

### Dynamic Minimum Safe Altitude (MSA)

```
MSA_ft = MaxElev_m × 3.281 + 1000 × (1 + mean_TTCI_cone)
```

The mean TTCI term increases the buffer dynamically over rougher terrain, providing conservative clearance in areas where a standard fixed 1,000 ft buffer would be inadequate.

**Computed for this scene:** `8,807 ft` at (77.22°E, 31.225°N) with 8 NM glide envelope.

---

## Project Structure

```
terrainguard/
│
├── pipeline.py                      # Full computational pipeline
├── index.html                       # Standalone Web GIS dashboard
├── README.md                        # This file
│
├── terrain_dem.tif                  # ← generated: 500×500 DEM (WGS84)
├── land_cover.tif                   # ← generated: 500×500 LULC raster
├── terrain_ttci.tif                 # ← generated: TTCI output raster
│
├── terrain_analysis_dashboard.png   # ← generated: 2×2 analysis grid
├── survivability_correlation_chart.png  # ← generated: FLSI vs TTCI scatter
└── avionics_output.json             # ← generated: FLSI triage telemetry
```

Files marked `← generated` are produced by running `pipeline.py`.

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/gauri-dhanakshirur/TerrainGuard_Bullseye.git
cd TerrainGuard_Bullseye

pip install rasterio numpy scipy matplotlib seaborn
```

> **Python version:** 3.10+ recommended.

### 2. Run the pipeline

```bash
python pipeline.py
```

Expected output:

```
============================================================
  TerrainGuard  –  CFIT Prevention Computational Pipeline
============================================================

[1/6] Generating synthetic DEM …
      DEM range: 1132.7 m – 2413.2 m
[2/6] Generating LULC raster …
      Class  10 (Urban           ) :   1,710 px
      Class  20 (Forest          ) : 209,645 px
      Class  30 (Grassland       ) :  31,901 px
      Class  40 (Water           ) :   1,848 px
      Class  50 (Airport Runway  ) :   4,896 px
[3/6] Computing TTCI …
      TTCI range: 0.0000 – 0.9999  (mean 0.0924)
[4/6] Running Emergency Triage (FLSI) …
      Aircraft @ (77.22°E, 31.225°N), glide cone 8.0 NM
      Dynamic MSA : 8807 ft
      Candidate px: 82,219
      Top-3 vectors:
        #1  Grassland        FLSI=0.837  Slope=3.7°  Survival=HIGH
        #2  Grassland        FLSI=0.800  Slope=2.2°  Survival=HIGH
        #3  Grassland        FLSI=0.800  Slope=5.4°  Survival=HIGH
[5/6] Generating terrain analysis dashboard …
[✓] Saved terrain_analysis_dashboard.png
[6/6] Generating survivability correlation chart …
[✓] Saved survivability_correlation_chart.png

============================================================
  All outputs written successfully.
============================================================
```

### 3. Open the dashboard

Simply open `index.html` in any modern browser — **no server required**.

```bash
# macOS
open index.html

# Linux
xdg-open index.html

# Windows
start index.html
```

Press the **⚠ MAYDAY** button in the top-right to activate the emergency triage overlay.

---

## Pipeline Deep-Dive

### `generate_dem()` — Synthetic DEM Construction

The DEM is built from stacked Gaussian-smoothed noise octaves at five spatial frequencies, producing geologically plausible mountain topography:

| Octave | Sigma | Amplitude |
|---|---|---|
| 1 (macro ridge) | 80 | 600 m |
| 2 (valley scale) | 40 | 300 m |
| 3 (slope scale) | 20 | 150 m |
| 4 (rock scale) | 10 | 75 m |
| 5 (roughness) | 5 | 40 m |

Three flat clearings are carved via Gaussian-blended compositing:
- **Main valley** (rows 200–300, cols 100–400) at ~1,200 m elevation
- **NW clearing** (rows 120–170, cols 50–140) at ~1,250 m elevation
- **SE clearing** (rows 330–380, cols 300–400) at ~1,280 m elevation

A central ridge is injected (cols 220–280) with a Gaussian cross-section.

**Scene bounds:** 77.0–77.45°E, 31.0–31.45°N (Shimla region, Himachal Pradesh, India). Pixel resolution: ~90 m GSD (SRTM-equivalent).

### `generate_lulc()` — LULC Classification

| Region | Rows | Cols | Class |
|---|---|---|---|
| High terrain (elev > 1600 m) | — | — | Forest (20) |
| Valley floor | 200–300 | 100–400 | Grassland (30) |
| NW clearing | 120–170 | 50–140 | Grassland (30) |
| SE clearing | 330–380 | 300–400 | Grassland (30) |
| Urban cluster | 285–315 | 108–165 | Urban (10) |
| Water body | 210–252 | 348–392 | Water (40) |
| Airport runway | 233–267 | 178–322 | Runway (50) |

### `compute_flsi()` — FLSI & Triage Output

The function accepts `(lon, lat, glide_radius_nm, dem, lulc, slope, ttci)` and returns a structured JSON dict:

```json
{
  "aircraft_pos":        {"lon": 77.22, "lat": 31.225},
  "glide_radius_nm":     8.0,
  "dynamic_msa_ft":      8807.0,
  "n_candidate_pixels":  82219,
  "top3_vectors": [
    {
      "comp_id": 2,
      "lon": 77.21794, "lat": 31.22608,
      "mean_flsi": 0.837,
      "mean_slope_deg": 3.7,
      "mean_ttci": 0.0403,
      "size_pixels": 26441,
      "dom_lulc_code": 30,
      "dom_lulc_name": "Grassland",
      "survival_class": "HIGH"
    },
    ...
  ],
  "sector_metrics": [
    {"sector": "N", "mean_flsi": 0.1441, "max_elev_m": 2353.6, ...},
    {"sector": "W", "mean_flsi": 0.5991, "max_elev_m": 1524.8, ...},
    ...
  ]
}
```

**Survival Classification Thresholds:**

| FLSI | Survival Class |
|---|---|
| ≥ 0.75 | HIGH |
| 0.40 – 0.74 | MODERATE |
| 0.15 – 0.39 | LOW |
| < 0.15 | CATASTROPHIC |

---

## Web GIS Dashboard

`index.html` is a fully standalone, zero-dependency (beyond CDN) single-page application.

### Map Layers

| Layer | Toggle | Description |
|---|---|---|
| Satellite Base | ✅ default on | Esri World Imagery |
| TTCI Overlay | ✅ default on | Canvas-rendered green/yellow/red complexity grid |
| LULC Overlay | ☐ off | Canvas-rendered ESA WorldCover categorical fills |
| Flight Route | ✅ default on | Dashed cyan polyline + 10 NM buffer corridor |

### MAYDAY Workflow

When the **⚠ MAYDAY** button is pressed:

1. Button enters pulsing red `ACTIVE` state
2. Status pill changes to `TRIAGE ACTIVE`
3. Animated 8 NM glide cone circle renders around the aircraft position
4. Three colour-coded vector lines draw from the aircraft to the Top-3 landing coordinates:
   - 🟢 **Priority 1** — solid green line (highest FLSI)
   - 🔵 **Priority 2** — dashed ice-blue line
   - 🟡 **Priority 3** — dashed amber line
5. Priority 1 landing popup auto-opens showing: surface type, FLSI %, slope angle, TTCI, and survival class
6. Right panel populates with triage cards, Dynamic MSA value, and sector analysis table

### Interactive Flight Router

Click **✚ Add Waypoint** to enter drawing mode (cursor changes to crosshair). Each click adds a waypoint; the 10 NM buffer corridor redraws automatically. Click **✕ Clear** to reset.

---

## Sample Outputs

### `terrain_analysis_dashboard.png`

2×2 dark-theme subplot grid:

| Panel | Content |
|---|---|
| Top-left | DEM elevation (m) — terrain colormap |
| Top-right | Slope in degrees — hot_r colormap |
| Bottom-left | Terrain Ruggedness Index — plasma colormap |
| Bottom-right | **TTCI** — custom green→yellow→red colormap |

### `survivability_correlation_chart.png`

Scatter plot of 4,000 randomly sampled pixels showing TTCI (x-axis) vs FLSI (y-axis), coloured by LULC class, with an OLS regression trend line demonstrating the inverse survivability–complexity relationship.

### `avionics_output.json`

Machine-readable telemetry export. The schema is designed to be ARINC 429 label mapping-ready for integration with existing avionics data buses.

---

## Data Sources

| Data | Source | Resolution | Licence |
|---|---|---|---|
| Digital Elevation Model | [NASA SRTM](https://www2.jpl.nasa.gov/srtm/) / [Copernicus GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) | 30–90 m | Public Domain / CC-BY 4.0 |
| Land Use / Land Cover | [ESA WorldCover 2021 v200](https://esa-worldcover.org/) | 10 m | CC-BY 4.0 |
| Map Tiles | [Esri World Imagery](https://www.esri.com/en-us/arcgis/products/arcgis-living-atlas/overview) | Variable | Esri ToU |
| Routing Library | [Leaflet.js 1.9.4](https://leafletjs.com/) | — | BSD-2-Clause |

The synthetic data generated by `pipeline.py` is designed to match the statistical properties of the Himachal Pradesh terrain region, with ESA WorldCover class codes used verbatim.

---

## Dependencies

```
rasterio>=1.3       # GeoTIFF I/O and CRS handling
numpy>=1.24         # Array computation
scipy>=1.10         # ndimage filtering and labelling
matplotlib>=3.7     # Visualisation
seaborn>=0.12       # Scatter aesthetics (used in correlation chart)
```

Install all at once:

```bash
pip install rasterio numpy scipy matplotlib seaborn
```

No GDAL system package installation is required — `rasterio` wheels ship with GDAL bundled on all major platforms.

The dashboard (`index.html`) loads two external CDN resources:

```
https://unpkg.com/leaflet@1.9.4/dist/leaflet.css
https://unpkg.com/leaflet@1.9.4/dist/leaflet.js
```

An internet connection is required to load map tiles and fonts. The TTCI and LULC overlays are generated entirely client-side via Canvas API and require no server.

---

## Extending TerrainGuard

### Swap in real DEM data

Replace `generate_dem()` with a `rasterio` read from a real SRTM or Copernicus tile:

```python
with rasterio.open("path/to/srtm_tile.tif") as src:
    dem = src.read(1).astype(np.float32)
    # Update WEST, SOUTH, EAST, NORTH from src.bounds
```

### Live aircraft position via ADS-B

Replace the mock `AC_POS` in `index.html` with a fetch to the [OpenSky Network REST API](https://openskynetwork.github.io/opensky-api/rest.html):

```javascript
const res = await fetch("https://opensky-network.org/api/states/all?icao24=<ICAO>");
const { states } = await res.json();
const [lon, lat, alt] = [states[0][5], states[0][6], states[0][7]];
```

### Adjust TTCI weights for aircraft category

The three weights in `pipeline.py` can be tuned per aircraft type:

```python
# Example: helicopter (roughness matters more than slope)
W_SLOPE    = 0.25
W_TRI      = 0.55
W_VARIANCE = 0.20
```

### Export to avionics bus

`avionics_output.json` maps directly to ARINC 429 word labels. Each `top3_vectors` entry corresponds to one discrete navigation fix that can be loaded into an FMS as a user waypoint.

---

## Demo

[Watch the Demo Video](https://github.com/gauri-dhanakshirur/TerrainGuard_Bullseye/raw/main/demo.webm)

<video src="https://github.com/gauri-dhanakshirur/TerrainGuard_Bullseye/raw/main/demo.webm" width="100%" controls></video>

*Press ⚠ MAYDAY to activate the emergency triage engine and see the glide cone, landing vectors, and sector analysis.*

---

## License & Attribution

This project is released under the **MIT License**.

If you use TerrainGuard's methodology in published work, please cite the underlying algorithms:

- Horn, B.K.P. (1981). *Hill shading and the reflectance map.* Proceedings of the IEEE, 69(1), 14–47.
- Riley, S.J., DeGloria, S.D., & Elliot, R. (1999). *A terrain ruggedness index that quantifies topographic heterogeneity.* Intermountain Journal of Sciences, 5(1–4), 23–27.
- ESA WorldCover 10 m 2021 v200. Zanaga et al. (2022). [doi:10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221)

---

*Built for the CFIT Prevention Hackathon · Open-source avionics safety tooling*
