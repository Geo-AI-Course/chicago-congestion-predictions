# Chicago Road Congestion Predictions

**Created by Ronen Gelmanovich and Shaked Ram**

### ▶ Live app: **https://REPLACE-WITH-STREAMLIT-URL.streamlit.app**

> Replace the link above with the public URL you get after deploying `webapp/app.py` to [Streamlit Community Cloud](https://share.streamlit.io). See [Deploy the app](#deploy-the-app).

Predict congestion scores on Chicago road segments using OSM road network features and city traffic count data, visualized as both static and interactive maps.

**Stack:** Python · osmnx · GeoPandas · PostGIS (Docker) · scikit-learn · folium · Streamlit

![Chicago congestion map](webapp/assets/congestion_static.png)

---

## Project Summary

**Problem.** Identify *where* Chicago's road network chokes relative to its design capacity, and *why* — predicting a congestion (volume/capacity) score for every road segment, not just the ~1.6% with measured counts, and pinpointing the structural cause of each genuine bottleneck.

**Why this is a GeoAI project.** The entire pipeline is spatial. Three GIS layers (road network, traffic signals, traffic counts) are stored as PostGIS geometries (SRID 4326) and joined spatially: traffic counts are snapped to their nearest road segment with a 150 m `ST_DWithin` lateral join over a GIST index; intersection, signal, and ramp density are computed by proximity (`ST_DWithin` in metres via geography casts); distances/areas use UTM EPSG:26916 (`ST_Transform`); and **network betweenness centrality** captures each segment's structural role in the graph. The model then predicts a phenomenon (congestion) continuously across space.

**Data** (counts from the latest run — see `webapp/assets/metrics.json`):

| Layer | Source | Records |
|---|---|---|
| Road segments | OpenStreetMap via `osmnx` | **77,544** |
| Intersection nodes | OSM (degree ≥ 3) | **26,478** |
| Traffic signals | OSM (`highway=traffic_signals`) | **4,204** |
| Traffic counts | [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Average-Daily-Traffic-Counts/pfsx-4HKf) | **1,279** (1,271 snapped to a segment) |
| Labeled segments (≥1 matched count) | — | **1,250 (1.6%)** — the training set |

**Data preparation.** List-valued OSM fields collapsed; `lanes` imputed by per-highway-type median then global median; `maxspeed` strings parsed to numeric (`"30 mph"` → `30`); highway type one-hot encoded; 13 bottleneck features engineered via PostGIS spatial joins; target = volume/capacity ratio normalised to its 99th percentile and clipped to [0, 1]. Detail in [Stage 2](#stage-2--feature-engineering-srcfeaturespy).

**Machine learning.** **Supervised regression** — a `RandomForestRegressor` tuned with 5-fold `GridSearchCV`, trained only on labeled segments and used to score all 77,544. Held-out test metrics from the latest run: **RMSE 0.126 · MAE 0.101 · R² 0.45** (vs. a mean-prediction baseline).

**Stage reached.** **Descriptive + Predictive** — the app describes the current network (scores + attributed bottleneck causes) and predicts scores for unlabeled segments. A *prescriptive* extension (forecasting emerging bottlenecks from population growth) is scoped as future work in [PLAN.md](PLAN.md) M7.

---

## Methodology — Features, Cleaning & Indexes

### What may cause congestion (the features)

We don't measure traffic on most roads, so the model learns congestion from the *structural* characteristics that create it. The engineered features group into four mechanisms:

**1. Capacity & geometry — how much traffic the road can physically carry**
- `lanes`, `speed_limit`, `length` — raw throughput capacity of the segment.
- `lane_drop_downstream` (1/0) and `downstream_capacity_ratio` — flag and quantify where the road *narrows ahead*. A 3-lane road feeding a 2-lane one is a classic choke point; the ratio captures how severe the drop is.
- `curvature_ratio` (`length ÷ straight-line distance`) — curves force drivers to slow, reducing effective capacity. 1.0 = straight; higher = more curved.
- `oneway` — one-way streets behave differently from two-way under load.

**2. Network structure — the segment's role in the wider graph**
- `betweenness` — approximate edge betweenness centrality (NetworkX, k=500 sampled sources): how often a segment lies on the shortest path between any two points. High betweenness = a structural funnel the whole city routes through.
- `fan_in_count` — number of segments whose end node is this segment's start node, i.e. how many roads *merge into* it. Many roads converging = merge congestion.
- `intersection_density` — intersection nodes (degree ≥ 3) within 100 m of the segment centroid; denser grids back up more easily.

**3. Proximity hazards — nearby features that create stop-and-go flow**
- `is_near_ramp` (1/0) — a `motorway_link` (on/off ramp) within 150 m. Ramp weaving zones are textbook bottlenecks.
- `traffic_signal_count` — signals within 100 m. Each signal injects stop-go cycles that queue traffic upstream.

**4. Demand context & road class**
- `neighbor_avg_volume` — measured volume on directly adjacent (upstream + downstream) segments, propagating demand signal to unlabeled roads.
- `hw_*` — one-hot road classification (motorway, trunk, primary, …), since a primary arterial and a residential street congest very differently.

**Target — `congestion_score`.** Not raw volume but a **volume/capacity (V/C) ratio**: `(avg_volume ÷ (lanes × speed_limit)) ÷ p99`, clipped to [0, 1]. This measures whether a road is *overloaded relative to its own design capacity* — a busy 6-lane highway may be fine, while a saturated 2-lane arterial is the real problem.

### How the data was cleaned

| Issue | Handling |
|---|---|
| OSM returns **list-valued fields** (e.g. multiple `lanes`/`maxspeed` per edge) | Collapsed to a single value (first / parsed token) |
| **Missing `lanes`** | Imputed with the median for that highway type, then the global median as a fallback |
| **`maxspeed` as free text** (`"30 mph"`, `"30;40"`) | Regex-parsed to a number; missing filled with the global median → `speed_limit` |
| **Categorical `highway`** | One-hot encoded (`pd.get_dummies`, `hw_*` columns) |
| **Traffic counts not on any road** (GPS noise) | Points with no segment within 150 m are left unmatched and excluded from volume aggregation |
| **Multiple counts per segment** | Aggregated to `avg_volume` / `max_volume` per segment |
| **Target outliers / undefined V/C** | Normalised to the 99th percentile and clipped to [0, 1]; segments with no measured volume get score 0 and are excluded from *training* (but still scored at inference) |

### Database indexes

Indexes are created in `src/ingest.py` immediately after each table is written, because every feature query is a spatial or topological join that would otherwise table-scan ~77k segments:

| Index | Table / column | Purpose |
|---|---|---|
| `road_segments_geom_idx` | `road_segments USING GIST(geometry)` | Powers the 150 m `ST_DWithin` snap and `<->` nearest-neighbour search, plus all proximity joins (signals, ramps, intersections) |
| `intersection_nodes_geom_idx` | `intersection_nodes USING GIST(geometry)` | Intersection-density proximity counts |
| `traffic_signals_geom_idx` | `traffic_signals USING GIST(geometry)` | Signal-density proximity counts |
| `traffic_counts_geom_idx` | `traffic_counts USING GIST(geometry)` | Fast spatial lookups when snapping counts |
| `road_segments_u_idx` | `road_segments(u)` (B-tree) | Topology self-joins on the **start** node |
| `road_segments_v_idx` | `road_segments(v)` (B-tree) | Topology self-joins on the **end** node — together `u`/`v` drive the fan-in, lane-drop, and neighbour-volume joins (`rs2.u = rs1.v`) |

---

## How It Works

The pipeline has four stages: data ingestion, feature engineering, model training, and visualization.

### Stage 1 — Data Ingestion (`src/ingest.py`)

Two independent data sources are fetched and loaded into PostGIS:

**Road network** — `osmnx` queries OpenStreetMap for Chicago's entire drivable road network. The result is a directed graph; edges become road segments and nodes become intersections. Each segment carries the attributes that will become model features: `highway` (road classification), `lanes`, `maxspeed`, `length` (meters), and `oneway`. List-valued fields (OSM sometimes returns multiple values per edge) are collapsed to a single value. A GIST spatial index is created on the geometry column for fast proximity queries later.

Intersection nodes are extracted separately: a node is counted as an intersection if its undirected degree is ≥ 3, meaning three or more roads meet there. These are stored in a separate `intersection_nodes` table and used in feature engineering.

While the graph is loaded, **approximate edge betweenness centrality** is computed via NetworkX (k=500 sampled sources). This measures how often each road segment appears on the shortest path between any two points in the network and is stored directly as a column on `road_segments`.

**Traffic signals** — OSM nodes tagged `highway=traffic_signals` are fetched via `osmnx.features_from_place` and stored in a `traffic_signals` table. These are used in feature engineering to count signalised intersections near each road segment.

**Traffic counts** — ~2,000 count records are fetched as GeoJSON from the Chicago Data Portal. Each record is a point geometry with a `total_passing_vehicle_volume` reading at that location. At this stage `segment_id` is null; snapping happens in the next step.

---

### Stage 2 — Feature Engineering (`src/features.py`)

**Snapping traffic counts to road segments**

Each traffic count point is assigned to its nearest road segment using a PostGIS lateral join:

```sql
CROSS JOIN LATERAL (
    SELECT rs.segment_id
    FROM   road_segments rs
    WHERE  ST_DWithin(tc.geometry::geography, rs.geometry::geography, 150)
    ORDER  BY tc.geometry <-> rs.geometry
    LIMIT  1
) AS closest
```

`ST_DWithin` restricts candidates to within 150 metres (in geographic units, so true metres on the ground), and the `<->` operator picks the single nearest one. Points with no segment within 150 m are left unmatched and excluded from volume aggregation.

**Intersection density**

For each road segment, the number of intersection nodes within 100 m of the segment's centroid is counted via `ST_DWithin`. This captures how connected (and therefore congestion-prone) the surrounding street network is.

**Feature matrix construction**

The full feature matrix is assembled by joining `road_segments` with aggregated traffic volumes per segment:

| Feature | Construction |
|---|---|
| `hw_*` (one-hot) | `pd.get_dummies` on the `highway` column — one column per road type |
| `lanes` | Integer; missing values filled with the median for that highway type, then the global median as a fallback |
| `speed_limit` | Parsed from `maxspeed` strings (e.g. `"30 mph"` → `30.0`); missing filled with global median |
| `length` | Segment length in metres from OSM edge geometry |
| `oneway` | Boolean cast to 0/1 |
| `betweenness` | Approximate edge betweenness centrality (k=500 sampled sources via NetworkX) — how often this segment appears on shortest paths through the network; high betweenness = structural bottleneck |
| `intersection_density` | Count of intersection nodes within 100 m of segment centroid |
| `fan_in_count` | Number of road segments whose end node equals this segment's start node — high count means many roads merging in |
| `lane_drop_downstream` | 1 if any immediately downstream segment has fewer lanes — flags where road capacity narrows |
| `downstream_capacity_ratio` | `self.lanes / avg(downstream.lanes)` — quantifies severity of a lane drop; > 1 means this segment is wider than what follows |
| `curvature_ratio` | `segment.length / straight_line_distance(start, end)` — 1 = straight road; higher values indicate curves that reduce effective capacity |
| `is_near_ramp` | 1 if any `motorway_link` segment is within 150 m — on/off ramp weaving zones are classic bottlenecks |
| `traffic_signal_count` | Count of traffic signal nodes within 100 m — signals create stop-go cycles that back up upstream traffic |
| `neighbor_avg_volume` | Average traffic volume of directly adjacent (upstream + downstream) segments — captures intersection-level stress |
| `congestion_score` (target) | `(avg_volume / (lanes × speed_limit)) / p99` clipped to [0, 1] — Volume/Capacity ratio normalised to the 99th percentile; measures whether a road is overloaded relative to its design capacity, not just how busy it is in absolute terms |

---

### Stage 3 — Model Training (`src/model.py`)

Only road segments that have at least one matched traffic count (i.e. `congestion_score > 0`) are used for training. Segments with no volume data are excluded from the training set but still receive a predicted score at inference time based on their road features.

The labeled subset is split 80/20 into train and test sets (random seed 42).

A **mean-prediction baseline** (always predict the training mean) is evaluated first to establish a lower bound.

A **Random Forest regressor** is then tuned with 5-fold cross-validation over:
- `n_estimators`: 100, 200, 300
- `max_depth`: 5, 10, unconstrained

`GridSearchCV` selects the combination with the best mean R². The winning model is evaluated on the held-out test set (RMSE, MAE, R²) and saved to `data/model.joblib`. A feature importance bar chart is saved to `data/feature_importance.png`.

---

### Stage 4 — Visualization (`src/visualize.py`)

The saved model runs `predict` on the full feature matrix to produce a `predicted_score` for every road segment. These scores are merged back onto the PostGIS geometries.

**Static map** — road segments are reprojected to EPSG:26916 (UTM, Illinois) for correct aspect ratio and plotted with a `RdYlGn_r` colormap (green = low congestion, red = high).

**Interactive map** — a Folium map (CartoDB Positron tiles) renders each segment as a `GeoJson` layer. Congestion level is expressed through both color (`RdYlGn_r`) and stroke weight (1–4 px), so heavily loaded roads are visually thicker. The color scale is normalised to the 99th percentile of predicted scores so a single outlier cannot wash out contrast across the rest of the map. Hovering over any segment shows its name, V/C score, and all active bottleneck indicators.

**Bottleneck markers** — circle dots are overlaid at the centroid of genuine bottleneck locations. A segment qualifies only if it satisfies *both* gates:

1. `predicted_score` is in the top `BOTTLENECK_SCORE_PCT` (default 5%) of all segments
2. At least one structural cause is present (lane drop, 4+ roads merging, ramp within 150 m, 2+ signals within 100 m, or top-5% betweenness centrality)

Markers are then capped at `MAX_BOTTLENECK_MARKERS` (default 400), keeping the most severe. Color indicates the primary cause:

| Color | Cause |
|---|---|
| Red | Lane drop ahead |
| Orange | 4+ roads merging in |
| Blue | On/off ramp weaving zone |
| Purple | 2+ traffic signals nearby |
| Dark red | Top-5% network centrality |

Both thresholds are tunable constants at the top of `src/visualize.py`.

---

### Stage 5 — Web Export (`src/export.py`)

PostGIS is the analytical backbone, but a hosted app can't query a local database. This step **extracts a slim, cleaned dataset from PostGIS** for the public app: predictions are merged onto geometries, filtered to the major-road network (motorway → tertiary + links), simplified (10 m Douglas-Peucker in UTM), coordinate-rounded, and baked into GeoJSON with precomputed colors/weights. The 127 MB local folium map becomes a **~6 MB** `webapp/assets/segments.geojson` the browser can load. Bottleneck points and a `metrics.json` (record counts + model metrics) are written alongside. Tunable via `MAJOR_HIGHWAYS` and `SIMPLIFY_TOLERANCE_M` at the top of `src/export.py`.

---

## The App

`webapp/app.py` is a [Streamlit](https://streamlit.io) app that reads only the `webapp/assets/*` files baked by `src/export.py` — **no database, model, or GDAL at runtime** — and renders the interactive congestion map, bottleneck markers, model metrics, and data-source links.

**Run locally:**

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py        # opens http://localhost:8501
```

### Deploy the app

1. Push this repo (including the committed `webapp/assets/`) to GitHub under the **Geo-AI-Course** org.
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at **`webapp/app.py`** (Streamlit auto-installs `webapp/requirements.txt`).
3. Copy the resulting public URL into the **Live app** link at the top of this README.

To refresh the deployed data, re-run the pipeline + `python src/export.py`, then commit the updated `webapp/assets/`.

---

## Quickstart

### 1. Start PostGIS

```bash
docker compose up -d
```

> **Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and ensure it is running before this step. WSL2 backend is recommended.

### 2. Set up Python environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **PowerShell note:** If you see an error about execution policy, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.

### 3. Configure credentials

Copy `.env.example` to `.env` (or create it) with the Docker defaults:

```
DB_URL=postgresql://congestion:congestion@localhost:5432/congestion
```

### 4. Verify the database connection

```bash
python db_check.py
```

### 5. Run the pipeline

```bash
python src/ingest.py      # fetch OSM + traffic data → PostGIS
python src/features.py    # spatial join + feature matrix → data/features.parquet
python src/model.py       # train Random Forest → data/model.joblib
python src/visualize.py   # generate local maps → data/congestion_map.html + data/congestion_static.png
python src/export.py      # bake slim web dataset → webapp/assets/  (for the Streamlit app)
```

---

## Outputs

| File | Description |
|---|---|
| `data/features.parquet` | Feature matrix used for training |
| `data/model.joblib` | Serialized Random Forest model |
| `data/feature_importance.png` | Bar chart of feature importances |
| `data/congestion_map.html` | Interactive folium map with segment tooltips (local, full network) |
| `data/congestion_static.png` | Static matplotlib choropleth map |
| `webapp/assets/segments.geojson` | Slim major-road map data for the deployed app (committed) |
| `webapp/assets/bottlenecks.geojson` | Bottleneck points + causes for the app (committed) |
| `webapp/assets/metrics.json` | Record counts + model test metrics (committed) |

---

## Project Structure

```
congestion-predictions/
├── data/                  # raw downloads and outputs (gitignored)
├── notebooks/
│   └── 01_eda.ipynb       # exploratory analysis: distributions, CRS checks, missing values
├── src/
│   ├── db.py              # shared DB engine + PostGIS helpers
│   ├── ingest.py          # load OSM road network + city traffic counts
│   ├── features.py        # spatial join, feature engineering
│   ├── model.py           # Random Forest training and evaluation
│   ├── visualize.py       # static + interactive folium maps
│   └── export.py          # bake slim web dataset from PostGIS → webapp/assets/
├── webapp/
│   ├── app.py             # public Streamlit app (reads webapp/assets/, no DB)
│   ├── requirements.txt   # app-only runtime deps
│   └── assets/            # committed GeoJSON + metrics the app renders
├── .streamlit/
│   └── config.toml        # Streamlit theme/server config
├── docker-compose.yml     # PostGIS container (port 5432)
├── requirements.txt
├── .env                   # DB credentials (gitignored)
├── db_check.py            # PostGIS smoke test
└── PLAN.md                # milestone-by-milestone implementation plan
```

---

## Data Sources

| Dataset | Source |
|---|---|
| Chicago road network | OpenStreetMap via `osmnx` |
| Average Daily Traffic Counts | [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Average-Daily-Traffic-Counts/pfsx-4HKf) |

---

## CRS Convention

- **Storage / exchange:** EPSG:4326 (WGS84) — all PostGIS tables
- **Distance / area calculations:** EPSG:26916 (UTM zone 16N, Illinois) — reprojected in-query via `ST_Transform`

---

## Roadmap

**Milestone 7 — Future Bottleneck Prediction (planned)**

Extend the model with CMAP 2050 population growth projections by community area to predict *emerging* bottlenecks: road segments whose congestion score is expected to increase significantly as the city grows.
