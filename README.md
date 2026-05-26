# Chicago Road Congestion Predictions

Predict congestion scores on Chicago road segments using OSM road network features and city traffic count data.

**Stack:** Python · osmnx · GeoPandas · PostGIS (Docker) · scikit-learn · folium

---

## Quickstart

### 1. Start PostGIS

```bash
docker compose up -d
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the database connection

```bash
python db_check.py
```

### 4. Run the pipeline

```bash
python src/ingest.py      # fetch OSM + traffic data → PostGIS
python src/features.py    # spatial join + feature matrix → data/features.parquet
python src/model.py       # train Random Forest → data/model.joblib
python src/visualize.py   # generate maps → data/congestion_map.html
```

---

## Project Structure

```
congestion-predictions/
├── data/                  # raw downloads and outputs (gitignored)
├── notebooks/             # exploratory notebooks
├── src/
│   ├── db.py              # shared DB engine + PostGIS helpers
│   ├── ingest.py          # load OSM road network + city traffic counts
│   ├── features.py        # spatial join, feature engineering
│   ├── model.py           # Random Forest training and evaluation
│   └── visualize.py       # static + interactive folium maps
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

- Storage: **EPSG:4326** (WGS84)
- Distance / area calculations: **EPSG:26916** (UTM zone 16N, Illinois)
