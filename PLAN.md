# Chicago Road Congestion Prediction — Implementation Plan

**Stack:** Python · osmnx · GeoPandas · PostGIS · scikit-learn  
**Goal:** Predict a congestion score (or binary bottleneck label) for road segments in Chicago using OSM road network features + city traffic count data.

---

## Milestone 1 — Environment & Project Structure

- [x] Create `requirements.txt` / `environment.yml` with all dependencies
  - `osmnx`, `geopandas`, `shapely`, `psycopg2-binary`, `sqlalchemy`, `geoalchemy2`
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `folium`
- [x] Spin up PostGIS via Docker (`postgis/postgis` image), write `docker-compose.yml`
- [x] Scaffold project directory layout:
  ```
  congestion-predictions/
  ├── data/           # raw downloads
  ├── notebooks/      # exploratory notebooks
  ├── src/
  │   ├── db.py       # shared DB engine + PostGIS helpers
  │   ├── ingest.py   # data loading & PostGIS writes
  │   ├── features.py # feature engineering
  │   ├── model.py    # training & evaluation
  │   └── visualize.py
  ├── docker-compose.yml
  ├── requirements.txt
  └── PLAN.md
  ```
- [x] Verify PostGIS connection and `postgis` extension enabled

---

## Milestone 2 — Data Acquisition

- [x] Fetch Chicago road network with `osmnx.graph_from_place("Chicago, Illinois, USA")`
  - Convert to GeoDataFrame of edges (road segments) and nodes (intersections)
  - Inspect key attributes: `highway`, `lanes`, `maxspeed`, `length`, `oneway`
- [x] Download Chicago Traffic Count dataset from the city open data portal
  - Source: [Chicago Data Portal — Average Daily Traffic Counts](https://data.cityofchicago.org/Transportation/Average-Daily-Traffic-Counts/pfsx-4HKf)
  - Columns of interest: location point, total volume, street name, direction
- [x] Exploratory notebook: distributions, CRS checks, missing value audit

---

## Milestone 3 — PostGIS Integration

- [x] Write OSM edges GeoDataFrame to PostGIS table `road_segments` (geometry: `LINESTRING`, SRID 4326)
- [x] Write traffic count points to PostGIS table `traffic_counts` (geometry: `POINT`, SRID 4326)
- [x] Spatial join: for each traffic count point, snap it to the nearest road segment
  - Use PostGIS `ST_DWithin` + `ST_Distance` to find nearest edge within a tolerance
  - Result: `traffic_counts` rows enriched with `segment_id`
- [x] Aggregate counts per segment (avg/max volume where multiple counts map to one segment)
- [x] Verify join quality: % of traffic points successfully matched, visual spot-check

---

## Milestone 4 — Feature Engineering

- [x] **Road type** — one-hot encode `highway` (motorway, trunk, primary, secondary, residential, …)
- [x] **Lane count** — integer, fill missing with median per highway type
- [x] **Speed limit** — parse `maxspeed`, fill missing with OSM defaults per type
- [x] **Segment length** — from edge geometry (meters)
- [x] **Intersection density** — count of OSM nodes (degree ≥ 3) within 100 m buffer of each segment, using PostGIS `ST_Buffer` + spatial join
- [x] **One-way flag** — binary from `oneway` attribute
- [x] **Target variable** — normalize traffic volume to a 0–1 congestion score (or threshold top-25% as binary bottleneck)
- [x] Write final feature matrix to PostGIS table `segment_features`; also export as `data/features.parquet`

---

## Milestone 5 — Model Training & Evaluation

- [x] Load `features.parquet`, define `X` (features) and `y` (congestion score / label)
- [x] Train/test split (80/20), stratify if binary classification
- [x] Baseline: predict mean volume (regression) or majority class (classification)
- [x] Train Random Forest — tune `n_estimators`, `max_depth` via cross-validation
- [x] Evaluate:
  - Regression: RMSE, MAE, R²
  - Classification (if binary): accuracy, F1, ROC-AUC, confusion matrix
- [x] Feature importance plot — identify which road attributes drive predictions most
- [x] Save trained model to `data/model.joblib`

---

## Milestone 6 — Visualization & Analysis

- [x] Merge predictions back onto road segment geometries
- [x] Static map: color road segments by predicted congestion score using `geopandas` + `matplotlib`
- [x] Interactive map: `folium` choropleth overlay, tooltips showing segment name + predicted score
- [x] Highlight top-10 predicted bottleneck segments on the map
- [x] Export final map as `data/congestion_map.html`

---

## Milestone 6.5 — Public Web App & Deployment

- [x] Add `src/export.py` to bake a slim, DB-free dataset from PostGIS into `webapp/assets/` (major roads, simplified geometry, precomputed colors, bottleneck points, `metrics.json`)
- [x] Build Streamlit app `webapp/app.py` (reads `webapp/assets/`, no DB/model/GDAL at runtime) with map, bottleneck markers, metrics, and data-source links
- [x] App-only `webapp/requirements.txt` + `.streamlit/config.toml`
- [x] Verify app runs locally against the baked assets with PostGIS stopped
- [ ] Deploy to Streamlit Community Cloud and add the public URL to the top of `README.md` *(manual: requires GitHub push under Geo-AI-Course org)*

---

## Milestone 7 — Future Bottleneck Prediction via Population Growth

- [ ] **Population data acquisition**
  - Download Chicago Community Area population estimates from the [Chicago Data Portal — Community Area Profiles](https://data.cityofchicago.org/Health-Human-Services/Chicago-Community-Area-Profiles/3ekn-bfbs)
  - Download or derive growth projections: CMAP (Chicago Metropolitan Agency for Planning) 2050 population forecasts by community area
  - Write both to PostGIS tables `community_areas` (geometry: `MULTIPOLYGON`) and `population_projections`
- [ ] **Spatial join: road segments → community areas**
  - For each road segment centroid, assign it to its containing community area using PostGIS `ST_Within` / `ST_Contains`
  - Store `community_area_id` on `road_segments`
- [ ] **Growth-adjusted feature matrix**
  - Compute **growth multiplier** per community area: `projected_population / current_population`
  - Add `growth_multiplier` and `current_population_density` as features in the segment feature table
  - Scale current `avg_volume` by the growth multiplier to produce `projected_volume`
- [ ] **Scenario model**
  - Train a second Random Forest (or reuse M5 model) using `projected_volume` as the target
  - Compare predicted congestion scores under current vs. projected-growth scenarios
  - Identify segments whose predicted score increases by more than a threshold (e.g., +0.15) — these are **emerging bottlenecks**
- [ ] **Visualisation**
  - Choropleth layer: community areas shaded by population growth rate
  - Road layer: segments colored by score delta (current → projected)
  - Highlight top-10 emerging bottleneck segments with distinct styling
  - Export as `data/future_congestion_map.html`

---

## Notes

- All spatial data stored in **SRID 4326** (WGS84); reproject to **EPSG:26916** (UTM zone 16N, Illinois) for distance/area calculations.
- Keep raw downloads in `data/` unmodified; all transforms happen in code.
- Each `src/` script should be runnable standalone (`python src/ingest.py`, etc.) as well as importable.
