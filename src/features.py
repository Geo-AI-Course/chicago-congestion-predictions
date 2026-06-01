"""Feature engineering — spatial joins and derived attributes.

Usage:
    python src/features.py
"""
import pandas as pd
from sqlalchemy import text
from db import get_engine

SNAP_TOLERANCE_M = 50  # metres; snapping tolerance for traffic → segment join


def snap_traffic_to_segments(engine=None):
    """Snap each traffic count point to its nearest road segment within tolerance."""
    engine = engine or get_engine()
    sql = text("""
        WITH snapped AS (
            SELECT tc.tc_id,
                   closest.segment_id
            FROM   traffic_counts tc
            CROSS JOIN LATERAL (
                SELECT rs.segment_id
                FROM   road_segments rs
                WHERE  ST_DWithin(
                           tc.geometry::geography,
                           rs.geometry::geography,
                           :tol
                       )
                ORDER  BY tc.geometry <-> rs.geometry
                LIMIT  1
            ) AS closest
            WHERE tc.segment_id IS NULL
        )
        UPDATE traffic_counts tc
        SET    segment_id = s.segment_id
        FROM   snapped s
        WHERE  tc.tc_id = s.tc_id
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"tol": SNAP_TOLERANCE_M})
        conn.commit()
        print(f"Snapped {result.rowcount} traffic count points to road segments.")


def _intersection_density(engine) -> pd.DataFrame:
    """Count intersection nodes within 100 m of each segment centroid."""
    sql = text("""
        SELECT rs.segment_id,
               COUNT(n.osmid) AS intersection_density
        FROM   road_segments rs
        LEFT JOIN intersection_nodes n
            ON ST_DWithin(
                   ST_Centroid(rs.geometry)::geography,
                   n.geometry::geography,
                   100
               )
        GROUP BY rs.segment_id
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def verify_join_quality(engine=None) -> float:
    """Report what fraction of traffic count points were snapped to a segment."""
    engine = engine or get_engine()
    sql = text("""
        SELECT COUNT(*) AS total, COUNT(segment_id) AS matched
        FROM   traffic_counts
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
    total, matched = row.total, row.matched
    pct = 100.0 * matched / total if total else 0.0
    print(f"Join quality: {matched}/{total} traffic points matched ({pct:.1f}%)")
    return pct


def build_feature_matrix(engine=None) -> pd.DataFrame:
    """Join road attributes with aggregated traffic volume."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            rs.segment_id,
            rs.highway,
            rs.lanes::float                              AS lanes,
            rs.maxspeed,
            rs.length,
            rs.oneway::int                               AS oneway,
            AVG(tc.total_passing_vehicle_volume::float)  AS avg_volume,
            MAX(tc.total_passing_vehicle_volume::float)  AS max_volume
        FROM   road_segments rs
        LEFT JOIN traffic_counts tc USING (segment_id)
        GROUP  BY rs.segment_id, rs.highway, rs.lanes,
                  rs.maxspeed, rs.length, rs.oneway
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    # --- fill missing lanes with median per highway type, then global fallback ---
    df["lanes"] = df.groupby("highway")["lanes"].transform(
        lambda x: x.fillna(x.median())
    )
    df["lanes"] = df["lanes"].fillna(df["lanes"].median())

    # --- one-hot encode highway type ---
    highway_dummies = pd.get_dummies(df["highway"], prefix="hw")
    df = pd.concat([df.drop(columns="highway"), highway_dummies], axis=1)

    # --- parse maxspeed (e.g. "30 mph" → 30.0) ---
    extracted = df["maxspeed"].str.extract(r"(\d+)")[0].astype(float)
    df["speed_limit"] = extracted.fillna(extracted.median())
    df = df.drop(columns="maxspeed")

    # --- intersection density ---
    density = _intersection_density(engine)
    df = df.merge(density, on="segment_id", how="left")
    df["intersection_density"] = df["intersection_density"].fillna(0).astype(int)

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
    verify_join_quality(engine)
    print("Building feature matrix …")
    df = build_feature_matrix(engine)
    save_features(df, engine)
    print("Feature engineering complete.")
