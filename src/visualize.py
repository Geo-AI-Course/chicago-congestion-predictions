"""Merge predictions onto road geometries and produce maps.

Usage:
    python src/visualize.py
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sqlalchemy import text
from db import get_engine

MODEL_PATH = "data/model.joblib"
FEATURE_PATH = "data/features.parquet"
MAP_OUT = "data/congestion_map.html"
TOP_N = 10


def load_predictions(engine=None) -> gpd.GeoDataFrame:
    engine = engine or get_engine()
    features = pd.read_parquet(FEATURE_PATH)
    model = joblib.load(MODEL_PATH)
    drop_cols = ["segment_id", "avg_volume", "max_volume", "congestion_score"]
    feature_cols = [c for c in features.columns if c not in drop_cols]
    features["predicted_score"] = model.predict(features[feature_cols])

    sql = text("SELECT segment_id, name, geometry FROM road_segments")
    with engine.connect() as conn:
        geoms = gpd.read_postgis(sql, conn, geom_col="geometry")

    gdf = geoms.merge(features[["segment_id", "predicted_score"]], on="segment_id")
    return gdf


def static_map(gdf: gpd.GeoDataFrame, out: str = "data/congestion_static.png"):
    gdf_proj = gdf.to_crs(epsg=26916)
    fig, ax = plt.subplots(figsize=(14, 14))
    gdf_proj.plot(ax=ax, column="predicted_score", cmap="RdYlGn_r",
                  linewidth=0.5, legend=True,
                  legend_kwds={"label": "Predicted congestion score"})
    ax.set_title("Chicago Road Congestion — Predicted Score", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Static map saved to {out}")


def interactive_map(gdf: gpd.GeoDataFrame, out: str = MAP_OUT):
    gdf_wgs = gdf.to_crs(epsg=4326)
    center = [gdf_wgs.geometry.centroid.y.mean(),
              gdf_wgs.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    norm = mcolors.Normalize(vmin=0, vmax=1)
    colormap = cm.get_cmap("RdYlGn_r")

    top_segments = gdf_wgs.nlargest(TOP_N, "predicted_score")["segment_id"]

    for _, row in gdf_wgs.iterrows():
        rgba = colormap(norm(row["predicted_score"]))
        color = mcolors.to_hex(rgba)
        weight = 5 if row["segment_id"] in top_segments.values else 2
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _, c=color, w=weight: {
                "color": c, "weight": w, "opacity": 0.8
            },
            tooltip=folium.Tooltip(
                f"{row.get('name', 'unnamed')} — score: {row['predicted_score']:.3f}"
            ),
        ).add_to(m)

    m.save(out)
    print(f"Interactive map saved to {out}")


if __name__ == "__main__":
    engine = get_engine()
    gdf = load_predictions(engine)
    static_map(gdf)
    interactive_map(gdf)
    print("Visualisation complete.")
