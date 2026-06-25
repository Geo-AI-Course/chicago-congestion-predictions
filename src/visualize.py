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
import matplotlib
import matplotlib.colors as mcolors
from sqlalchemy import text
from db import get_engine

MODEL_PATH = "data/model.joblib"
FEATURE_PATH = "data/features.parquet"
MAP_OUT = "data/congestion_map.html"

# Bottleneck marker tuning
BOTTLENECK_SCORE_PCT = 0.95   # a road must be in the top 5% of congestion to qualify
MAX_BOTTLENECK_MARKERS = 400   # hard cap on points shown, most severe first

# Bottleneck columns passed through to the map for tooltip display
BOTTLENECK_COLS = [
    "fan_in_count",
    "lane_drop_downstream",
    "downstream_capacity_ratio",
    "curvature_ratio",
    "is_near_ramp",
    "traffic_signal_count",
    "neighbor_avg_volume",
    "betweenness",
]


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

    keep = ["segment_id", "predicted_score"] + [
        c for c in BOTTLENECK_COLS if c in features.columns
    ]
    gdf = geoms.merge(features[keep], on="segment_id")
    return gdf


def _bottleneck_tooltip(row) -> str:
    """Build an HTML tooltip summarising congestion score and bottleneck causes."""
    name = row.get("name") or "unnamed"
    score = row["predicted_score"]

    lines = []

    if row.get("lane_drop_downstream", 0):
        ratio = row.get("downstream_capacity_ratio", 1.0)
        lines.append(f"Lane drop ahead (capacity ratio {ratio:.1f}x)")

    if row.get("is_near_ramp", 0):
        lines.append("Near on/off ramp weaving zone")

    sig = int(row.get("traffic_signal_count", 0))
    if sig > 0:
        lines.append(f"{sig} traffic signal{'s' if sig > 1 else ''} within 100 m")

    fan = int(row.get("fan_in_count", 0))
    if fan > 2:
        lines.append(f"{fan} roads merging at entry node")

    bc = row.get("betweenness", 0.0)
    if bc > 0.001:
        lines.append(f"High network centrality ({bc:.4f})")

    curv = row.get("curvature_ratio", 1.0)
    if curv > 1.3:
        lines.append(f"Curved road (ratio {curv:.2f})")

    cause_html = (
        "<br>".join(f"&bull; {l}" for l in lines)
        if lines
        else "No specific bottleneck indicators"
    )

    return (
        f"<b>{name}</b><br>"
        f"Congestion score: <b>{score:.3f}</b>"
        f"<hr style='margin:4px 0;border-color:#ccc'>"
        f"{cause_html}"
    )


def _primary_cause(row) -> tuple:
    """Return (label, hex_color) for the dominant bottleneck cause."""
    if row.get("lane_drop_downstream", 0):
        ratio = row.get("downstream_capacity_ratio", 1.0)
        return f"Lane drop (capacity ratio {ratio:.1f}x)", "#e74c3c"
    fan = int(row.get("fan_in_count", 0))
    if fan >= 4:
        return f"{fan} roads merging in", "#e67e22"
    if row.get("is_near_ramp", 0):
        return "On/off ramp weaving zone", "#2980b9"
    sig = int(row.get("traffic_signal_count", 0))
    if sig >= 2:
        return f"{sig} traffic signals nearby", "#8e44ad"
    bc = row.get("betweenness", 0.0)
    if bc > 0:
        return f"High network centrality ({bc:.4f})", "#c0392b"
    return "High congestion", "#c0392b"


def select_bottlenecks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the genuine bottleneck segments, most severe first.

    A bottleneck requires BOTH high predicted congestion AND a structural cause —
    a lane drop on an empty street is not a bottleneck. The result is capped to
    the most severe MAX_BOTTLENECK_MARKERS so downstream maps stay readable.
    """
    cols = set(gdf.columns)

    # --- gate 1: only roads that are actually congested ---
    score_gate = gdf["predicted_score"].quantile(BOTTLENECK_SCORE_PCT)
    congested = gdf["predicted_score"] >= score_gate

    # --- gate 2: a structural reason for the choke must be present ---
    bc_thresh = (
        gdf["betweenness"].quantile(0.95)
        if "betweenness" in cols else float("inf")
    )
    has_cause = pd.Series(False, index=gdf.index)
    if "lane_drop_downstream"  in cols: has_cause |= gdf["lane_drop_downstream"] == 1
    if "fan_in_count"          in cols: has_cause |= gdf["fan_in_count"] >= 4
    if "is_near_ramp"          in cols: has_cause |= gdf["is_near_ramp"] == 1
    if "traffic_signal_count"  in cols: has_cause |= gdf["traffic_signal_count"] >= 2
    if "betweenness"           in cols: has_cause |= gdf["betweenness"] >= bc_thresh

    # bottleneck = congested AND has a cause, then keep only the most severe
    return (
        gdf[congested & has_cause]
        .sort_values("predicted_score", ascending=False)
        .head(MAX_BOTTLENECK_MARKERS)
    )


def _add_bottleneck_markers(m: folium.Map, gdf: gpd.GeoDataFrame):
    """Overlay circle markers at genuine bottleneck pinpoints."""
    bottlenecks = select_bottlenecks(gdf)
    print(f"  Marking {len(bottlenecks)} bottleneck points "
          f"(capped at {MAX_BOTTLENECK_MARKERS}) …")

    for _, row in bottlenecks.iterrows():
        centroid = row.geometry.centroid
        label, color = _primary_cause(row)
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=4,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b>Bottleneck: {label}</b><br>{_bottleneck_tooltip(row)}",
                sticky=False,
            ),
        ).add_to(m)


def static_map(gdf: gpd.GeoDataFrame, out: str = "data/congestion_static.png"):
    gdf_proj = gdf.to_crs(epsg=26916)
    fig, ax = plt.subplots(figsize=(14, 14))
    gdf_proj.plot(ax=ax, column="predicted_score", cmap="RdYlGn_r",
                  linewidth=0.5, legend=True,
                  legend_kwds={"label": "Predicted congestion score (V/C ratio)"})
    ax.set_title("Chicago Road Congestion — Predicted V/C Score", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Static map saved to {out}")


def interactive_map(gdf: gpd.GeoDataFrame, out: str = MAP_OUT):
    gdf_wgs = gdf.to_crs(epsg=4326)
    center = [gdf_wgs.geometry.centroid.y.mean(),
              gdf_wgs.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    vmax = gdf_wgs["predicted_score"].quantile(0.99)
    norm = mcolors.Normalize(vmin=0, vmax=max(vmax, 1e-6))
    colormap = matplotlib.colormaps["RdYlGn_r"]

    for _, row in gdf_wgs.iterrows():
        normed = norm(row["predicted_score"])
        color  = mcolors.to_hex(colormap(normed))
        weight = 1 + 3 * normed          # 1 (low) → 4 (high congestion)

        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _, c=color, w=weight: {
                "color": c, "weight": w, "opacity": 0.8
            },
            tooltip=folium.Tooltip(
                _bottleneck_tooltip(row),
                sticky=False,
            ),
        ).add_to(m)

    _add_bottleneck_markers(m, gdf_wgs)
    m.save(out)
    print(f"Interactive map saved to {out}")


if __name__ == "__main__":
    engine = get_engine()
    gdf = load_predictions(engine)
    static_map(gdf)
    interactive_map(gdf)
    print("Visualisation complete.")
