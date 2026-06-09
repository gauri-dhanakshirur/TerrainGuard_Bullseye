#!/usr/bin/env python3
"""
TerrainGuard – CFIT Prevention Computational Pipeline
=====================================================

Generates synthetic terrain data and runs the full TTCI + FLSI analysis
pipeline for Controlled Flight Into Terrain prevention.

Outputs:
    terrain_dem.tif                  – 500×500 DEM raster (WGS84)
    land_cover.tif                   – 500×500 LULC raster (WGS84)
    terrain_ttci.tif                 – TTCI output raster
    terrain_analysis_dashboard.png   – 2×2 analysis grid
    survivability_correlation_chart.png – FLSI vs TTCI scatter
    avionics_output.json             – FLSI triage telemetry
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import scipy.ndimage as ndimage
import seaborn as sns

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ──────────────────────────── CONSTANTS ────────────────────────────

ROWS, COLS = 500, 500
WEST, SOUTH, EAST, NORTH = 77.0, 31.0, 77.45, 31.45  # Shimla region
CRS_WGS84 = CRS.from_epsg(4326)
TRANSFORM = from_bounds(WEST, SOUTH, EAST, NORTH, COLS, ROWS)

# Approx metres per pixel at ~31°N
M_PER_PIXEL = (EAST - WEST) * 111_320 * np.cos(np.radians(31.225)) / COLS  # ~85 m

# TTCI weights
W_SLOPE = 0.4
W_TRI = 0.4
W_VARIANCE = 0.2

# Aircraft mock position
AC_LON, AC_LAT = 77.22, 31.225
GLIDE_RADIUS_NM = 8.0

# LULC suitability scores (ESA WorldCover codes)
LULC_SCORES = {10: 0.0, 20: 0.1, 30: 0.8, 40: 0.3, 50: 1.0}
LULC_NAMES = {
    10: "Urban",
    20: "Forest",
    30: "Grassland",
    40: "Water",
    50: "Airport Runway",
}

# Output directory (same as script location)
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) or "."


# ──────────────────────────── HELPERS ────────────────────────────


def _minmax(arr):
    """Min-Max normalise array to [0, 1]."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo == 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _write_tif(path, data, dtype="float32"):
    """Write a single-band GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "height": ROWS,
        "width": COLS,
        "count": 1,
        "dtype": dtype,
        "crs": CRS_WGS84,
        "transform": TRANSFORM,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


# ──────────────────────── 1. DEM GENERATION ──────────────────────


def generate_dem(seed=42):
    """Build a synthetic DEM from stacked Gaussian-smoothed noise octaves."""
    rng = np.random.default_rng(seed)

    # Octave table: (sigma, amplitude)
    octaves = [(80, 600), (40, 300), (20, 150), (10, 75), (5, 40)]
    dem = np.zeros((ROWS, COLS), dtype=np.float64)

    for sigma, amp in octaves:
        noise = rng.standard_normal((ROWS, COLS))
        dem += ndimage.gaussian_filter(noise, sigma=sigma) * amp

    # Shift to realistic elevation range (~1100–2300 m)
    dem -= dem.min()
    dem = dem / dem.max() * 1150 + 1130

    # ── Carve a flat valley (rows 200–300, cols 100–400) ──
    valley_elev = 1200.0
    valley_mask = np.zeros_like(dem)
    valley_mask[200:300, 100:400] = 1.0
    valley_mask = ndimage.gaussian_filter(valley_mask, sigma=12)
    dem = dem * (1 - valley_mask) + valley_elev * valley_mask

    # ── Carve a NW clearing (rows 120–170, cols 50–140) ──
    nw_mask = np.zeros_like(dem)
    nw_mask[120:170, 50:140] = 1.0
    nw_mask = ndimage.gaussian_filter(nw_mask, sigma=8)
    dem = dem * (1 - nw_mask) + 1250.0 * nw_mask

    # ── Carve a SE clearing (rows 330–380, cols 300–400) ──
    se_mask = np.zeros_like(dem)
    se_mask[330:380, 300:400] = 1.0
    se_mask = ndimage.gaussian_filter(se_mask, sigma=8)
    dem = dem * (1 - se_mask) + 1280.0 * se_mask

    # ── Inject a central ridge (cols 220–280) ──
    ridge = np.zeros((ROWS, COLS), dtype=np.float64)
    col_centre = 250
    for c in range(220, 280):
        ridge[:, c] = 400 * np.exp(-0.5 * ((c - col_centre) / 8) ** 2)
    ridge = ndimage.gaussian_filter(ridge, sigma=6)
    # Only add ridge outside valley
    ridge[195:305, 95:405] = 0
    dem += ridge

    path = os.path.join(OUT_DIR, "terrain_dem.tif")
    _write_tif(path, dem)
    return dem, path


# ──────────────────────── 2. LULC GENERATION ──────────────────────


def generate_lulc(dem):
    """Generate a synthetic ESA WorldCover-style LULC raster."""
    lulc = np.full((ROWS, COLS), 20, dtype=np.int16)  # default forest

    # High terrain → Forest (already set)
    # Valley floor → Grassland
    lulc[200:300, 100:400] = 30

    # NW grassland clearing
    lulc[120:170, 50:140] = 30

    # SE grassland clearing
    lulc[330:380, 300:400] = 30

    # Urban cluster
    lulc[285:315, 108:165] = 10

    # Water body
    lulc[210:252, 348:392] = 40

    # Airport runway (wide strip in the valley)
    lulc[233:267, 178:322] = 50

    path = os.path.join(OUT_DIR, "land_cover.tif")
    _write_tif(path, lulc, dtype="int16")

    # Stats
    counts = {}
    for code, name in LULC_NAMES.items():
        counts[code] = int(np.sum(lulc == code))

    return lulc, path, counts


# ──────────────────── 3. FEATURE EXTRACTION ──────────────────────


def compute_slope(dem):
    """Horn (1981) slope in degrees."""
    cell = M_PER_PIXEL
    # Sobel gradients
    dzdx = ndimage.sobel(dem, axis=1) / (8 * cell)
    dzdy = ndimage.sobel(dem, axis=0) / (8 * cell)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)


def compute_tri(dem):
    """Riley et al. (1999) Terrain Ruggedness Index via 3×3 window."""

    def _tri_kernel(values):
        centre = values[4]  # centre of 3×3 flat array
        return np.sqrt(np.sum((values - centre) ** 2))

    return ndimage.generic_filter(dem, _tri_kernel, size=3)


def compute_variance(dem):
    """5×5 moving standard deviation of elevation."""
    k = 5
    mean = ndimage.uniform_filter(dem, size=k)
    mean_sq = ndimage.uniform_filter(dem**2, size=k)
    var = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    return var


def compute_ttci(slope, tri, variance):
    """Compute Terrain Topography Complexity Index."""
    s_n = _minmax(slope)
    t_n = _minmax(tri)
    v_n = _minmax(variance)
    ttci = W_SLOPE * s_n + W_TRI * t_n + W_VARIANCE * v_n
    return ttci


# ──────────────────── 4. EMERGENCY TRIAGE (FLSI) ─────────────────


def compute_flsi(lon, lat, glide_nm, dem, lulc, slope, ttci):
    """
    Compute Forced Landing Suitability Index and return structured triage data.
    """
    # Convert aircraft lat/lon to pixel row/col
    col_ac = int((lon - WEST) / (EAST - WEST) * COLS)
    row_ac = int((NORTH - lat) / (NORTH - SOUTH) * ROWS)

    # Glide cone radius in pixels
    glide_m = glide_nm * 1852
    r_px = int(glide_m / M_PER_PIXEL)

    # Step 1 — Glide Cone Mask
    rr, cc = np.ogrid[:ROWS, :COLS]
    cone = ((rr - row_ac) ** 2 + (cc - col_ac) ** 2) <= r_px**2

    # Step 2 — Slope Gate (≤15°)
    candidate = cone & (slope <= 15.0)

    # Step 3 — LULC Suitability Scoring
    score_map = np.zeros_like(dem)
    for code, score in LULC_SCORES.items():
        score_map[lulc == code] = score

    flsi = np.where(candidate, score_map, 0.0)

    # Step 4 — Connected Component Analysis (FLSI > 0.5)
    labels, n_features = ndimage.label(flsi > 0.5)
    components = []

    for comp_id in range(1, n_features + 1):
        mask = labels == comp_id
        size = int(np.sum(mask))
        if size < 50:
            continue

        mean_flsi = float(np.mean(flsi[mask]))
        mean_slope = float(np.mean(slope[mask]))
        mean_ttci_val = float(np.mean(ttci[mask]))

        # Dominant LULC class
        lulc_vals = lulc[mask]
        dom_code = int(np.bincount(lulc_vals.astype(int)).argmax())

        # Score with runway bonus
        comp_score = mean_flsi * (1.2 if dom_code == 50 else 1.0)

        # Centroid → lon/lat
        rows_idx, cols_idx = np.where(mask)
        cr = float(np.mean(rows_idx))
        cc_val = float(np.mean(cols_idx))
        comp_lon = WEST + (cc_val / COLS) * (EAST - WEST)
        comp_lat = NORTH - (cr / ROWS) * (NORTH - SOUTH)

        # Survival classification
        if mean_flsi >= 0.75:
            surv = "HIGH"
        elif mean_flsi >= 0.40:
            surv = "MODERATE"
        elif mean_flsi >= 0.15:
            surv = "LOW"
        else:
            surv = "CATASTROPHIC"

        components.append(
            {
                "comp_id": comp_id,
                "lon": round(comp_lon, 5),
                "lat": round(comp_lat, 5),
                "mean_flsi": round(mean_flsi, 4),
                "mean_slope_deg": round(mean_slope, 2),
                "mean_ttci": round(mean_ttci_val, 4),
                "size_pixels": size,
                "dom_lulc_code": dom_code,
                "dom_lulc_name": LULC_NAMES.get(dom_code, "Unknown"),
                "survival_class": surv,
                "score": round(comp_score, 4),
            }
        )

    # Rank by score descending
    components.sort(key=lambda c: c["score"], reverse=True)
    top3 = components[:3]

    # Remove internal score key
    for v in top3:
        v.pop("score", None)

    # Dynamic MSA
    cone_dem = dem[cone]
    cone_ttci = ttci[cone]
    max_elev_m = float(np.max(cone_dem))
    mean_ttci_cone = float(np.mean(cone_ttci))
    msa_ft = max_elev_m * 3.281 + 1000 * (1 + mean_ttci_cone)

    # Sector metrics (8 compass sectors)
    sector_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector_metrics = []
    for i, name in enumerate(sector_names):
        angle_lo = i * 45 - 22.5
        angle_hi = i * 45 + 22.5
        # Compute angle from aircraft for each pixel in cone
        rr_full, cc_full = np.where(cone)
        dy = -(rr_full - row_ac)  # negative because row increases downward
        dx = cc_full - col_ac
        angles = np.degrees(np.arctan2(dx, dy)) % 360

        if angle_lo < 0:
            sec_mask = (angles >= angle_lo + 360) | (angles < angle_hi)
        else:
            sec_mask = (angles >= angle_lo) & (angles < angle_hi)

        if np.sum(sec_mask) == 0:
            sector_metrics.append(
                {
                    "sector": name,
                    "mean_flsi": 0.0,
                    "max_elev_m": 0.0,
                    "mean_ttci": 0.0,
                    "n_pixels": 0,
                }
            )
            continue

        sec_rows = rr_full[sec_mask]
        sec_cols = cc_full[sec_mask]
        sector_metrics.append(
            {
                "sector": name,
                "mean_flsi": round(float(np.mean(flsi[sec_rows, sec_cols])), 4),
                "max_elev_m": round(float(np.max(dem[sec_rows, sec_cols])), 1),
                "mean_ttci": round(float(np.mean(ttci[sec_rows, sec_cols])), 4),
                "n_pixels": int(np.sum(sec_mask)),
            }
        )

    result = {
        "aircraft_pos": {"lon": lon, "lat": lat},
        "glide_radius_nm": glide_nm,
        "dynamic_msa_ft": round(msa_ft, 1),
        "n_candidate_pixels": int(np.sum(candidate)),
        "top3_vectors": top3,
        "sector_metrics": sector_metrics,
    }

    return result, flsi


# ──────────────────── 5. VISUALISATION ─────────────────────────


def generate_dashboard(dem, slope, tri, ttci):
    """Generate 2×2 dark-theme terrain analysis dashboard."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "TerrainGuard – Terrain Analysis Dashboard",
        fontsize=18,
        fontweight="bold",
        color="#00ffcc",
        y=0.98,
    )

    # Panel 1: DEM
    im1 = axes[0, 0].imshow(dem, cmap="terrain", aspect="auto")
    axes[0, 0].set_title("Digital Elevation Model (m)", fontsize=13, color="#cccccc")
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label="Elevation (m)")

    # Panel 2: Slope
    im2 = axes[0, 1].imshow(slope, cmap="hot_r", aspect="auto", vmin=0, vmax=60)
    axes[0, 1].set_title("Slope (degrees)", fontsize=13, color="#cccccc")
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label="Slope (°)")

    # Panel 3: TRI
    im3 = axes[1, 0].imshow(tri, cmap="plasma", aspect="auto")
    axes[1, 0].set_title(
        "Terrain Ruggedness Index (Riley)", fontsize=13, color="#cccccc"
    )
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8, label="TRI")

    # Panel 4: TTCI with custom colormap
    ttci_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ttci", ["#00ff00", "#ffff00", "#ff0000"]
    )
    im4 = axes[1, 1].imshow(ttci, cmap=ttci_cmap, aspect="auto", vmin=0, vmax=1)
    axes[1, 1].set_title("TTCI (Complexity Index)", fontsize=13, color="#cccccc")
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8, label="TTCI")

    for ax in axes.flat:
        ax.set_xlabel("Column", fontsize=10, color="#999999")
        ax.set_ylabel("Row", fontsize=10, color="#999999")
        ax.tick_params(colors="#888888")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "terrain_analysis_dashboard.png")
    fig.savefig(path, dpi=150, facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return path


def generate_correlation_chart(ttci, flsi, lulc):
    """Generate FLSI vs TTCI scatter coloured by LULC class."""
    plt.style.use("dark_background")

    # Sample 4000 random points where flsi > 0
    valid = flsi > 0
    if np.sum(valid) < 10:
        print("      [!] Not enough valid FLSI pixels for scatter.")
        return None

    n_sample = min(4000, int(np.sum(valid)))
    rng = np.random.default_rng(123)
    idx = rng.choice(np.sum(valid), size=n_sample, replace=False)

    ttci_flat = ttci[valid].ravel()[idx]
    flsi_flat = flsi[valid].ravel()[idx]
    lulc_flat = lulc[valid].ravel()[idx]

    # Map LULC codes to names
    lulc_labels = np.array([LULC_NAMES.get(int(c), "Other") for c in lulc_flat])

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        "TerrainGuard – Survivability Correlation",
        fontsize=16,
        fontweight="bold",
        color="#00ffcc",
        y=0.97,
    )

    palette = {
        "Urban": "#ff4444",
        "Forest": "#228b22",
        "Grassland": "#88cc44",
        "Water": "#4488ff",
        "Airport Runway": "#ffaa00",
        "Other": "#888888",
    }

    sns.scatterplot(
        x=ttci_flat,
        y=flsi_flat,
        hue=lulc_labels,
        palette=palette,
        alpha=0.6,
        s=18,
        edgecolor="none",
        ax=ax,
    )

    # OLS regression trend line
    coeffs = np.polyfit(ttci_flat, flsi_flat, 1)
    x_line = np.linspace(0, 1, 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, "--", color="#ffffff", linewidth=2, alpha=0.7, label="OLS trend")

    ax.set_xlabel("TTCI (Terrain Complexity)", fontsize=13, color="#cccccc")
    ax.set_ylabel("FLSI (Landing Suitability)", fontsize=13, color="#cccccc")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10, framealpha=0.3, loc="upper right")
    ax.tick_params(colors="#888888")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "survivability_correlation_chart.png")
    fig.savefig(path, dpi=150, facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return path


# ──────────────────────── MAIN PIPELINE ──────────────────────────


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TerrainGuard  –  CFIT Prevention Computational Pipeline")
    print(f"{sep}\n")

    # ── Step 1: DEM ──
    print("[1/6] Generating synthetic DEM …")
    dem, dem_path = generate_dem()
    print(f"      DEM range: {dem.min():.1f} m – {dem.max():.1f} m")

    # ── Step 2: LULC ──
    print("[2/6] Generating LULC raster …")
    lulc, lulc_path, lulc_counts = generate_lulc(dem)
    for code, name in LULC_NAMES.items():
        print(f"      Class  {code:2d} ({name:16s}) : {lulc_counts[code]:>7,d} px")

    # ── Step 3: TTCI ──
    print("[3/6] Computing TTCI …")
    slope = compute_slope(dem)
    tri = compute_tri(dem)
    variance = compute_variance(dem)
    ttci = compute_ttci(slope, tri, variance)
    ttci_path = os.path.join(OUT_DIR, "terrain_ttci.tif")
    _write_tif(ttci_path, ttci)
    print(f"      TTCI range: {ttci.min():.4f} – {ttci.max():.4f}  (mean {ttci.mean():.4f})")

    # ── Step 4: FLSI Triage ──
    print("[4/6] Running Emergency Triage (FLSI) …")
    print(f"      Aircraft @ ({AC_LON}°E, {AC_LAT}°N), glide cone {GLIDE_RADIUS_NM} NM")
    triage, flsi = compute_flsi(AC_LON, AC_LAT, GLIDE_RADIUS_NM, dem, lulc, slope, ttci)
    print(f"      Dynamic MSA : {triage['dynamic_msa_ft']:.0f} ft")
    print(f"      Candidate px: {triage['n_candidate_pixels']:,d}")
    print(f"      Top-3 vectors:")
    for i, v in enumerate(triage["top3_vectors"]):
        print(
            f"        #{i + 1}  {v['dom_lulc_name']:16s} FLSI={v['mean_flsi']:.3f}  "
            f"Slope={v['mean_slope_deg']:.1f}°  Survival={v['survival_class']}"
        )

    # Write avionics JSON
    json_path = os.path.join(OUT_DIR, "avionics_output.json")
    with open(json_path, "w") as f:
        json.dump(triage, f, indent=2)

    # ── Step 5: Dashboard ──
    print("[5/6] Generating terrain analysis dashboard …")
    dash_path = generate_dashboard(dem, slope, tri, ttci)
    print(f"[✓] Saved {os.path.basename(dash_path)}")

    # ── Step 6: Correlation Chart ──
    print("[6/6] Generating survivability correlation chart …")
    corr_path = generate_correlation_chart(ttci, flsi, lulc)
    if corr_path:
        print(f"[✓] Saved {os.path.basename(corr_path)}")

    print(f"\n{sep}")
    print("  All outputs written successfully.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
