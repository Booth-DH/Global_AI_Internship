# Chronic Disease Burden & Social Determinants of Health
**Population Health Management | Python · K-Means Clustering · Geospatial Mapping**

This project identifies at-risk U.S. counties by merging the CDC PLACES dataset with the CDC Social Vulnerability Index (SVI), then applying unsupervised clustering to segment counties into four actionable risk tiers.

---

## Project Overview

| Step | Description |
|------|-------------|
| Data Merge | CDC PLACES (2023) + SVI (2022) joined on FIPS county code |
| Correlation Analysis | Pearson r between SDOH factors and diabetes/hypertension rates (2019–2023) |
| Choropleth Map | Interactive map of diabetes prevalence overlaid with SVI scores |
| K-Means Clustering | Counties segmented into Low / Moderate / High / Critical risk tiers |

---

## Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| CDC PLACES | CDC Socrata API | ~3,000 U.S. counties, 45 health measures, 2019–2023 |
| CDC Social Vulnerability Index (SVI) | CDC/ATSDR Socrata API | ~3,143 counties, 158 variables, 2020 & 2022 snapshots |

Raw data is downloaded automatically by the notebook on first run and cached in `raw_data/`.

---

## Repository Structure

```
.
├── assignment.ipynb        # Main analysis notebook
├── raw_data/               # Downloaded source files (PLACES + SVI CSVs)
├── processed_data/         # Cleaned & merged data
│   ├── panel_county_health.csv   # 5-year panel (2019–2023), ~15k rows
│   └── us_counties.geojson       # County boundary geometries
└── results/                # All outputs
    ├── clustered_counties_2023.csv   # County-level cluster assignments
    ├── fig1_sdoh_trend.png           # SDOH correlation trends 2019–2023
    ├── fig2_2019_vs_2023.png         # Endpoint comparison
    ├── fig3_heatmap_2023.png         # SDOH × disease correlation heatmap
    ├── fig4_top_correlates_2023.png  # Ranked SDOH factors (2023)
    ├── fig5_diabetes_svi_map.html    # Interactive choropleth (diabetes + SVI)
    ├── fig6_kmeans_validation.png    # Elbow & silhouette plots
    ├── fig7_cluster_profiles.png     # Risk tier feature heatmap
    ├── fig8_cluster_distributions.png# Box plots by risk tier
    └── fig9_risk_tier_map.html       # Interactive county risk tier map
```

---

## Key Findings

- **Food insecurity, housing insecurity, and lack of transportation** are the strongest SDOH correlates of diabetes (r ≈ 0.87–0.89) at the county level.
- **Physical inactivity** is the strongest correlate available across all years (r ≈ 0.84–0.88).
- K-Means (K=4) identifies four well-separated risk tiers:

| Risk Tier | Counties | Diabetes Rate | SVI Score |
|-----------|----------|---------------|-----------|
| Low | 765 (33%) | 9.1% | 0.19 |
| Moderate | 788 (34%) | 10.8% | 0.46 |
| High | 508 (22%) | 12.6% | 0.75 |
| Critical | 238 (10%) | 15.8% | 0.90 |

---

## Setup & Usage

### Requirements
```
pip install pandas numpy matplotlib seaborn folium scikit-learn scipy requests branca
```

### Running the Notebook
Open `assignment.ipynb` in Jupyter and run all cells top to bottom. The notebook will:
1. Download and cache raw data (skip if already present)
2. Clean, merge, and build the 5-year panel
3. Generate all figures and maps into `results/`

> The two interactive HTML maps (`fig5`, `fig9`) can be opened directly in any browser.

---

## Skills Demonstrated
- Data wrangling across multi-year, multi-schema public health datasets
- Ecological correlation analysis (Pearson r, 5-year trend)
- Interactive geospatial visualization with Folium
- Unsupervised machine learning (K-Means, elbow method, silhouette scoring)
- Population health risk stratification for SDOH-informed intervention
