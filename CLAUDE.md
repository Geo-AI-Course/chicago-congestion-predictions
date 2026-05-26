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
python src/ingest.py      # fetch OSM + Chicago traffic data → PostGIS
python src/features.py    # spatial join, feature engineering → PostGIS + data/features.parquet
python src/model.py       # train RandomForest → data/model.joblib + data/feature_importance.png
python src/visualize.py   # generate maps → data/congestion_map.html + data/congestion_static.png
```

Scripts are also importable as modules; they share `src/db.py` for the DB connection.

## Architecture

**Data flow:**

```
osmnx (Chicago OSM)  →  road_segments (PostGIS LINESTRING, SRID 4326)
Chicago Data Portal  →  traffic_counts (PostGIS POINT, SRID 4326)
                              ↓
                    ST_DWithin snap (50 m tolerance)
                              ↓
                    segment_features (PostGIS + data/features.parquet)
                              ↓
                    RandomForestRegressor (scikit-learn)
                              ↓
                    folium interactive map + matplotlib static map
```

**Module responsibilities:**
- `src/db.py` — `get_engine()` reads `DB_URL` from `.env`; `ensure_postgis()` enables the extension
- `src/ingest.py` — fetches external data and writes raw tables (`road_segments`, `traffic_counts`)
- `src/features.py` — PostGIS spatial join, feature matrix construction, normalization; outputs `segment_features` table and `data/features.parquet`
- `src/model.py` — loads parquet, trains/evaluates model, serializes to `data/model.joblib`
- `src/visualize.py` — loads model + PostGIS geometries, produces both map outputs

**PostGIS tables:**
| Table | Geometry | Key columns |
|---|---|---|
| `road_segments` | LINESTRING 4326 | highway, lanes, maxspeed, length, oneway |
| `traffic_counts` | POINT 4326 | total_volume, segment_id (after snap) |
| `segment_features` | — | highway_* (one-hot), lanes, speed_limit, length, oneway, avg_volume, congestion_score |

## CRS Convention

- **Storage / exchange:** EPSG:4326 (WGS84) — all PostGIS tables
- **Distance / area calculations:** EPSG:26916 (UTM zone 16N, Illinois) — reproject in-query via `ST_Transform`

## Environment

Credentials live in `.env` (gitignored). The Docker defaults are:

```
DB_URL=postgresql://congestion:congestion@localhost:5432/congestion
```

Python 3.12, virtual environment at `.venv/`. Activate with `source .venv/bin/activate`.
