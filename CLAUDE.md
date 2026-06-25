# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progress Tracking

After completing any task that maps to a checklist item in `PLAN.md`, immediately update that item from `- [ ]` to `- [x]` in the same response. Do not wait to be asked.

## Infrastructure

Start/stop the PostGIS container (required before any pipeline step):

```bash
docker compose up -d       # start PostGIS on localhost:5432
docker compose down        # stop
```

Verify DB connection and PostGIS extension:

```bash
python db_check.py
```

## Pipeline Execution Order

Each `src/` script is a standalone entrypoint with a `__main__` block. Run in order:

```bash
python src/ingest.py      # fetch OSM + traffic data + traffic signals → PostGIS (computes betweenness)
python src/features.py    # spatial join (150 m snap), bottleneck features, V/C target → PostGIS + data/features.parquet
python src/model.py       # train RandomForest → data/model.joblib + data/feature_importance.png
python src/visualize.py   # generate maps → data/congestion_map.html + data/congestion_static.png
python src/export.py      # bake slim web dataset from PostGIS → webapp/assets/ (for the Streamlit app)
```

Scripts are also importable as modules; they share `src/db.py` for the DB connection.

## Web App

`webapp/app.py` is a Streamlit app that is the project's public deliverable. It reads ONLY the committed `webapp/assets/*` files produced by `src/export.py` — **no PostGIS, model, osmnx, or GDAL at runtime** — so it deploys cleanly to Streamlit Community Cloud (which can't host the database).

```bash
pip install -r webapp/requirements.txt   # app-only deps, separate from root requirements.txt
streamlit run webapp/app.py
```

To refresh deployed data, re-run the pipeline then `python src/export.py` and commit the updated `webapp/assets/`. The major-road filter and simplification tolerance are tunable at the top of `src/export.py` (`MAJOR_HIGHWAYS`, `SIMPLIFY_TOLERANCE_M`). Keep `webapp/assets/` committed — it lives outside the gitignored `data/`.

## Architecture

**Data flow:**

```
osmnx (Chicago OSM)  →  road_segments (PostGIS LINESTRING, SRID 4326)
                     →  intersection_nodes (PostGIS POINT, SRID 4326)
                     →  traffic_signals (PostGIS POINT, SRID 4326)
Chicago Data Portal  →  traffic_counts (PostGIS POINT, SRID 4326)
                              ↓
                    ST_DWithin snap (150 m tolerance)
                              ↓
                    segment_features (PostGIS + data/features.parquet)
                    [bottleneck features: lane_drop, fan_in, curvature,
                     betweenness, ramp proximity, signal density, V/C target]
                              ↓
                    RandomForestRegressor trained on labeled segments only
                              ↓
                    folium interactive map + matplotlib static map
                    + bottleneck circle markers (top 5% score AND structural cause)
```

**Module responsibilities:**
- `src/db.py` — `get_engine()` reads `DB_URL` from `.env`; `ensure_postgis()` enables the extension
- `src/ingest.py` — fetches OSM road network (with edge betweenness centrality via NetworkX), traffic counts, and traffic signal locations; writes `road_segments`, `intersection_nodes`, `traffic_signals`, `traffic_counts`
- `src/features.py` — PostGIS spatial join (150 m snap), bottleneck feature engineering, V/C ratio target; outputs `segment_features` table and `data/features.parquet`
- `src/model.py` — filters to labeled segments only, trains/evaluates Random Forest, serializes to `data/model.joblib`
- `src/visualize.py` — loads model + PostGIS geometries, produces static map and interactive map with bottleneck markers

**PostGIS tables:**
| Table | Geometry | Key columns |
|---|---|---|
| `road_segments` | LINESTRING 4326 | highway, lanes, maxspeed, length, oneway, betweenness, u, v |
| `intersection_nodes` | POINT 4326 | osmid, degree |
| `traffic_signals` | POINT 4326 | osmid |
| `traffic_counts` | POINT 4326 | total_passing_vehicle_volume, segment_id (after snap) |
| `segment_features` | — | hw_* (one-hot), lanes, speed_limit, length, oneway, betweenness, fan_in_count, lane_drop_downstream, downstream_capacity_ratio, curvature_ratio, is_near_ramp, traffic_signal_count, neighbor_avg_volume, intersection_density, congestion_score (V/C ratio target) |

**Bottleneck feature notes:**
- `congestion_score` target is now a V/C ratio: `(avg_volume / (lanes × speed_limit)) / p99`, clipped to [0, 1]. Higher = road is overloaded relative to its design capacity.
- Model is trained only on the ~1–5% of segments with actual traffic count data; it predicts scores for all segments at inference time.
- Bottleneck markers in the interactive map require BOTH high predicted score (top 5% by default) AND a structural cause. Tune `BOTTLENECK_SCORE_PCT` and `MAX_BOTTLENECK_MARKERS` in `src/visualize.py`.

## CRS Convention

- **Storage / exchange:** EPSG:4326 (WGS84) — all PostGIS tables
- **Distance / area calculations:** EPSG:26916 (UTM zone 16N, Illinois) — reproject in-query via `ST_Transform`

## Environment

Credentials live in `.env` (gitignored). The Docker defaults are:

```
DB_URL=postgresql://congestion:congestion@localhost:5432/congestion
```

Python 3.12, virtual environment at `.venv/`. Activate with `source .venv/bin/activate`.
