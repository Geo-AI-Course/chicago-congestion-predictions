"""Feature engineering — spatial joins and derived attributes.

Usage:
    python src/features.py
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from db import get_engine

SNAP_TOLERANCE_M = 50  # metres; snapping tolerance for traffic → segment join


def snap_traffic_to_segments(engine=None):
    """Spatial join: traffic count points → nearest road segment."""
    engine = engine or get_engine()
    sql = text("""
        UPDATE traffic_counts tc
        SET    segment_id = rs.segment_id
        FROM   road_segments rs
        WHERE  ST_DWithin(
                   tc.geometry::geography,
                   rs.geometry::geography,
                   :tol
               )
          AND  tc.segment_id IS NULL
        -- pick the single nearest segment
        AND rs.segment_id = (
            SELECT rs2.segment_id
            FROM   road_segments rs2
            ORDER  BY tc.geometry <-> rs2.geometry
            LIMIT  1
        )
    """)
    with engine.connect() as conn:
        conn.execute(sql, {"tol": SNAP_TOLERANCE_M})
        conn.commit()


def build_feature_matrix(engine=None) -> pd.DataFrame:
    """Join road attributes with aggregated traffic volume."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            rs.segment_id,
            rs.highway,
            COALESCE(rs.lanes::float, NULL)          AS lanes,
            rs.maxspeed,
            rs.length,
            rs.oneway::int                            AS oneway,
            AVG(tc.total_passing_vehicle_volume::float) AS avg_volume,
            MAX(tc.total_passing_vehicle_volume::float) AS max_volume
        FROM   road_segments rs
        LEFT JOIN traffic_counts tc USING (segment_id)
        GROUP  BY rs.segment_id, rs.highway, rs.lanes,
                  rs.maxspeed, rs.length, rs.oneway
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    # --- one-hot encode highway type ---
    highway_dummies = pd.get_dummies(df["highway"], prefix="hw")
    df = pd.concat([df.drop(columns="highway"), highway_dummies], axis=1)

    # --- fill missing lanes with median per highway type (already dropped) ---
    df["lanes"] = df["lanes"].fillna(df["lanes"].median())

    # --- parse maxspeed (e.g. "30 mph" → 30) ---
    df["speed_limit"] = (
        df["maxspeed"]
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .fillna(df["maxspeed"].str.extract(r"(\d+)")[0].astype(float).median())
    )
    df = df.drop(columns="maxspeed")

    # --- target: normalised congestion score [0, 1] ---
    df["congestion_score"] = (
        df["avg_volume"] / df["avg_volume"].max()
    ).fillna(0)

    return df


def save_features(df: pd.DataFrame, engine=None):
    engine = engine or get_engine()
    df.to_sql("segment_features", engine, if_exists="replace", index=False)
    df.to_parquet("data/features.parquet", index=False)
    print(f"Saved feature matrix: {df.shape}")


if __name__ == "__main__":
    engine = get_engine()
    print("Snapping traffic counts to road segments …")
    snap_traffic_to_segments(engine)
    print("Building feature matrix …")
    df = build_feature_matrix(engine)
    save_features(df, engine)
    print("Feature engineering complete.")
