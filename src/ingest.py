"""Data loading and PostGIS writes.

Usage:
    python src/ingest.py
"""
import pandas as pd
import osmnx as ox
import geopandas as gpd
import networkx as nx
from sqlalchemy import text
from db import get_engine, ensure_postgis

CHICAGO = "Chicago, Illinois, USA"
TRAFFIC_URL = (
    "https://data.cityofchicago.org/resource/pfsx-4n4m.geojson"
    "?$limit=2000"
)


def fetch_road_network(place: str = CHICAGO):
    """Returns (edges GeoDataFrame with segment_id, intersection nodes GeoDataFrame)."""
    G = ox.graph_from_place(place, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)

    # Intersection nodes: degree >= 3 in the undirected sense
    undirected = G.to_undirected()
    degree = dict(undirected.degree())
    nodes = nodes.reset_index()
    nodes["degree"] = nodes["osmid"].map(degree)
    intersections = nodes[nodes["degree"] >= 3][["osmid", "geometry"]].copy()

    edges = edges.reset_index()
    edges = edges[["u", "v", "key", "highway", "lanes", "maxspeed",
                   "length", "oneway", "name", "geometry"]].copy()

    edges["highway"] = edges["highway"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    edges["lanes"] = edges["lanes"].apply(_parse_lanes)

    # approximate edge betweenness centrality (k=500 sampled sources)
    print("  Computing edge betweenness centrality (k=500 approx) …")
    k = min(500, len(G))
    bc = nx.edge_betweenness_centrality(G, k=k, normalized=True, seed=42)
    def _bc(u, v, key):
        return bc.get((u, v, key), bc.get((u, v), 0.0))
    edges["betweenness"] = edges.apply(
        lambda r: _bc(r["u"], r["v"], r["key"]), axis=1
    )

    # sequential 1-based segment_id
    edges.insert(0, "segment_id", range(1, len(edges) + 1))

    return edges, intersections


def _parse_lanes(val):
    if pd.isna(val) if not isinstance(val, list) else False:
        return None
    s = val[0] if isinstance(val, list) else val
    try:
        return int(str(s).split(";")[0].strip())
    except (ValueError, AttributeError):
        return None


def fetch_traffic_counts(url: str = TRAFFIC_URL) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(url)
    print(f"Traffic count columns: {list(gdf.columns)}")
    gdf = gdf.reset_index(drop=True)
    gdf.insert(0, "tc_id", range(1, len(gdf) + 1))
    gdf["segment_id"] = pd.array([pd.NA] * len(gdf), dtype=pd.Int64Dtype())
    return gdf


def write_road_segments(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("road_segments", engine, if_exists="replace",
                   index=False, chunksize=500)
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS road_segments_geom_idx "
            "ON road_segments USING GIST(geometry)"
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS road_segments_u_idx ON road_segments(u)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS road_segments_v_idx ON road_segments(v)"))
        conn.commit()
    print(f"Wrote {len(gdf)} road segments to PostGIS.")


def write_intersection_nodes(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("intersection_nodes", engine, if_exists="replace",
                   index=False, chunksize=500)
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS intersection_nodes_geom_idx "
            "ON intersection_nodes USING GIST(geometry)"
        ))
        conn.commit()
    print(f"Wrote {len(gdf)} intersection nodes to PostGIS.")


def fetch_traffic_signals(place: str = CHICAGO) -> gpd.GeoDataFrame:
    """Fetch OSM nodes tagged as traffic signals."""
    signals = ox.features_from_place(place, tags={"highway": "traffic_signals"})
    pts = signals[signals.geometry.type == "Point"].reset_index()
    pts = pts[["osmid", "geometry"]].copy().reset_index(drop=True)
    print(f"Fetched {len(pts)} traffic signal locations.")
    return pts


def write_traffic_signals(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("traffic_signals", engine, if_exists="replace",
                   index=False, chunksize=500)
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS traffic_signals_geom_idx "
            "ON traffic_signals USING GIST(geometry)"
        ))
        conn.commit()
    print(f"Wrote {len(gdf)} traffic signal locations to PostGIS.")


def write_traffic_counts(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("traffic_counts", engine, if_exists="replace",
                   index=False, chunksize=500)
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS traffic_counts_geom_idx "
            "ON traffic_counts USING GIST(geometry)"
        ))
        conn.commit()
    print(f"Wrote {len(gdf)} traffic count records to PostGIS.")


if __name__ == "__main__":
    engine = get_engine()
    ensure_postgis(engine)

    print("Fetching road network …")
    edges, intersections = fetch_road_network()
    write_road_segments(edges, engine)
    write_intersection_nodes(intersections, engine)

    print("Fetching traffic counts …")
    counts = fetch_traffic_counts()
    write_traffic_counts(counts, engine)

    print("Fetching traffic signals …")
    signals = fetch_traffic_signals()
    write_traffic_signals(signals, engine)

    print("Ingest complete.")
