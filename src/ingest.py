"""Data loading and PostGIS writes.

Usage:
    python src/ingest.py
"""
import osmnx as ox
import geopandas as gpd
import pandas as pd
from db import get_engine, ensure_postgis

CHICAGO = "Chicago, Illinois, USA"
TRAFFIC_URL = (
    "https://data.cityofchicago.org/api/geospatial/pfsx-4HKf"
    "?method=export&type=geojson"
)


def fetch_road_network(place: str = CHICAGO):
    G = ox.graph_from_place(place, network_type="drive")
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()
    edges = edges[["u", "v", "key", "highway", "lanes", "maxspeed",
                   "length", "oneway", "name", "geometry"]]
    edges["highway"] = edges["highway"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    return edges


def fetch_traffic_counts(url: str = TRAFFIC_URL) -> gpd.GeoDataFrame:
    return gpd.read_file(url)


def write_road_segments(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("road_segments", engine, if_exists="replace",
                   index=False, chunksize=500)
    print(f"Wrote {len(gdf)} road segments to PostGIS.")


def write_traffic_counts(gdf: gpd.GeoDataFrame, engine=None):
    engine = engine or get_engine()
    gdf.to_postgis("traffic_counts", engine, if_exists="replace",
                   index=False, chunksize=500)
    print(f"Wrote {len(gdf)} traffic count records to PostGIS.")


if __name__ == "__main__":
    engine = get_engine()
    ensure_postgis(engine)

    print("Fetching road network …")
    edges = fetch_road_network()
    write_road_segments(edges, engine)

    print("Fetching traffic counts …")
    counts = fetch_traffic_counts()
    write_traffic_counts(counts, engine)

    print("Ingest complete.")
