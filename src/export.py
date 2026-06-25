"""Extract a slim, deploy-ready web dataset from PostGIS.

PostGIS is the source of truth and the analytical backbone (spatial joins, GIST
index, UTM distance calcs). But Streamlit Community Cloud cannot host a database,
so this step bakes geometry + model predictions + bottleneck flags into small
GeoJSON files that the Streamlit app reads directly — the deployed app needs no DB.

Outputs (committed, read by webapp/app.py):
    webapp/assets/segments.geojson     — major roads, simplified, colored by score
    webapp/assets/bottlenecks.geojson  — genuine bottleneck points + causes
    webapp/assets/metrics.json         — dataset counts + model test metrics
    webapp/assets/congestion_static.png — copy of the static overview map

Usage:
    python src/export.py
"""
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from db import get_engine
from model import FEATURE_PATH, MODEL_PATH, load_data, evaluate
from visualize import (
    load_predictions,
    select_bottlenecks,
    _bottleneck_tooltip,
    _primary_cause,
)

ASSET_DIR = Path("webapp/assets")
STATIC_SRC = "data/congestion_static.png"

# Highway classes worth drawing — keeps the web payload small and the map legible.
# Residential/service streets dominate the segment count but carry little signal.
MAJOR_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}
SIMPLIFY_TOLERANCE_M = 10   # metres; Douglas-Peucker tolerance in UTM 26916
COORD_PRECISION = 5         # decimal degrees (~1.1 m) — trims GeoJSON file size


def _highway_map(engine) -> pd.DataFrame:
    sql = text("SELECT segment_id, highway FROM road_segments")
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def build_web_gdf(engine):
    """Predictions + geometry, filtered to major roads and simplified for the web."""
    gdf = load_predictions(engine)               # geometry (4326), name, score, causes
    gdf = gdf.merge(_highway_map(engine), on="segment_id", how="left")
    gdf = gdf[gdf["highway"].isin(MAJOR_HIGHWAYS)].copy()

    # simplify in metric CRS, then back to WGS84 for the web
    proj = gdf.to_crs(epsg=26916)
    proj["geometry"] = proj.geometry.simplify(SIMPLIFY_TOLERANCE_M)
    gdf = proj.to_crs(epsg=4326)
    print(f"  {len(gdf)} major-road segments kept for the web map")
    return gdf


def _style(gdf):
    """RdYlGn_r color + line weight per segment — mirrors visualize.interactive_map."""
    vmax = gdf["predicted_score"].quantile(0.99)
    norm = mcolors.Normalize(vmin=0, vmax=max(vmax, 1e-6))
    cmap = matplotlib.colormaps["RdYlGn_r"]

    def fn(score):
        n = norm(score)
        return mcolors.to_hex(cmap(n)), round(1 + 3 * n, 2)
    return fn


def _coords(geom, p):
    """Rounded coordinate lists for a (Multi)LineString."""
    def r(c):
        return [round(c[0], p), round(c[1], p)]
    if geom.geom_type == "LineString":
        return {"type": "LineString", "coordinates": [r(c) for c in geom.coords]}
    if geom.geom_type == "MultiLineString":
        return {"type": "MultiLineString",
                "coordinates": [[r(c) for c in g.coords] for g in geom.geoms]}
    from shapely.geometry import mapping
    return mapping(geom)


def write_segments(gdf):
    style = _style(gdf)
    features = []
    for _, row in gdf.iterrows():
        color, weight = style(row["predicted_score"])
        features.append({
            "type": "Feature",
            "geometry": _coords(row.geometry, COORD_PRECISION),
            "properties": {
                "name": row.get("name") or "unnamed",
                "score": round(float(row["predicted_score"]), 4),
                "color": color,
                "weight": weight,
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    out = ASSET_DIR / "segments.geojson"
    out.write_text(json.dumps(fc))
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(features)} features)")


def write_bottlenecks(gdf):
    bn = select_bottlenecks(gdf)
    features = []
    for _, row in bn.iterrows():
        centroid = row.geometry.centroid
        label, color = _primary_cause(row)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(centroid.x, COORD_PRECISION),
                                         round(centroid.y, COORD_PRECISION)]},
            "properties": {
                "label": label,
                "color": color,
                "tooltip": f"<b>Bottleneck: {label}</b><br>{_bottleneck_tooltip(row)}",
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    out = ASSET_DIR / "bottlenecks.geojson"
    out.write_text(json.dumps(fc))
    print(f"  wrote {out} ({len(features)} bottleneck points)")


def write_metrics(engine):
    """Dataset counts + reproducible model test metrics for the README and app."""
    df = pd.read_parquet(FEATURE_PATH)
    total = len(df)
    labeled = int((df["congestion_score"] > 0).sum())

    # reproduce model.py's held-out evaluation against the saved model
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(MODEL_PATH)
    m = evaluate(y_test, model.predict(X_test), label="RandomForest (test)")

    with engine.connect() as conn:
        tc_total = conn.execute(text("SELECT COUNT(*) FROM traffic_counts")).scalar()
        tc_matched = conn.execute(text("SELECT COUNT(segment_id) FROM traffic_counts")).scalar()
        signals = conn.execute(text("SELECT COUNT(*) FROM traffic_signals")).scalar()
        nodes = conn.execute(text("SELECT COUNT(*) FROM intersection_nodes")).scalar()

    metrics = {
        "total_segments": total,
        "labeled_segments": labeled,
        "labeled_pct": round(100.0 * labeled / total, 2) if total else 0.0,
        "traffic_count_records": int(tc_total),
        "traffic_counts_matched": int(tc_matched),
        "traffic_signals": int(signals),
        "intersection_nodes": int(nodes),
        "model": {k: round(float(v), 4) for k, v in m.items()},
    }
    out = ASSET_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {out}: {metrics}")
    return metrics


if __name__ == "__main__":
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("Building web GeoDataFrame …")
    gdf = build_web_gdf(engine)

    print("Writing segments …")
    write_segments(gdf)

    print("Writing bottlenecks …")
    write_bottlenecks(gdf)

    print("Writing metrics …")
    write_metrics(engine)

    if Path(STATIC_SRC).exists():
        shutil.copy(STATIC_SRC, ASSET_DIR / "congestion_static.png")
        print(f"  copied {STATIC_SRC} → {ASSET_DIR}")

    print("Export complete.")
