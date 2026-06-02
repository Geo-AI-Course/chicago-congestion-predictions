"""Feature engineering — spatial joins and derived attributes.

Usage:
    python src/features.py
"""
import pandas as pd
from sqlalchemy import text
from db import get_engine

SNAP_TOLERANCE_M = 150  # metres; snapping tolerance for traffic → segment join


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


def _topology_features(engine) -> pd.DataFrame:
    """Fan-in count, lane-drop flag, and downstream capacity ratio."""
    fan_in_sql = text("""
        SELECT rs1.segment_id,
               COUNT(rs2.segment_id) AS fan_in_count
        FROM   road_segments rs1
        LEFT JOIN road_segments rs2 ON rs2.v = rs1.u
        GROUP BY rs1.segment_id
    """)
    lane_sql = text("""
        SELECT rs1.segment_id,
               COALESCE(
                   BOOL_OR(
                       rs2.lanes IS NOT NULL AND rs1.lanes IS NOT NULL
                       AND rs2.lanes::int < rs1.lanes::int
                   ),
                   FALSE
               )::int AS lane_drop_downstream,
               COALESCE(
                   CASE
                       WHEN rs1.lanes IS NOT NULL AND AVG(rs2.lanes::float) > 0
                       THEN rs1.lanes::float / AVG(rs2.lanes::float)
                       ELSE 1.0
                   END,
                   1.0
               )      AS downstream_capacity_ratio
        FROM   road_segments rs1
        LEFT JOIN road_segments rs2
               ON  rs2.u = rs1.v
               AND rs2.lanes IS NOT NULL
               AND rs1.lanes IS NOT NULL
        GROUP BY rs1.segment_id, rs1.lanes
    """)
    with engine.connect() as conn:
        fan_in = pd.read_sql(fan_in_sql, conn)
        lanes  = pd.read_sql(lane_sql,   conn)
    return fan_in.merge(lanes, on="segment_id")


def _geometry_features(engine) -> pd.DataFrame:
    """Curvature ratio: actual length divided by straight-line endpoint distance."""
    sql = text("""
        SELECT segment_id,
               CASE
                   WHEN ST_Distance(
                            ST_StartPoint(geometry)::geography,
                            ST_EndPoint(geometry)::geography
                        ) > 1
                   THEN length / ST_Distance(
                            ST_StartPoint(geometry)::geography,
                            ST_EndPoint(geometry)::geography
                        )
                   ELSE 1.0
               END AS curvature_ratio
        FROM road_segments
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def _proximity_features(engine) -> pd.DataFrame:
    """On/off ramp adjacency and traffic signal density per segment."""
    ramp_sql = text("""
        SELECT rs.segment_id,
               (COUNT(ramp.segment_id) > 0)::int AS is_near_ramp
        FROM   road_segments rs
        LEFT JOIN road_segments ramp
               ON  ramp.highway = 'motorway_link'
               AND ST_DWithin(rs.geometry::geography, ramp.geometry::geography, 150)
               AND ramp.segment_id != rs.segment_id
        GROUP BY rs.segment_id
    """)
    signal_sql = text("""
        SELECT rs.segment_id,
               COUNT(ts.osmid) AS traffic_signal_count
        FROM   road_segments rs
        LEFT JOIN traffic_signals ts
               ON ST_DWithin(rs.geometry::geography, ts.geometry::geography, 100)
        GROUP BY rs.segment_id
    """)
    with engine.connect() as conn:
        ramps   = pd.read_sql(ramp_sql,   conn)
        signals = pd.read_sql(signal_sql, conn)
    return ramps.merge(signals, on="segment_id")


def _neighbor_volume(engine) -> pd.DataFrame:
    """Average traffic volume of directly adjacent (upstream + downstream) segments."""
    sql = text("""
        WITH adj AS (
            SELECT rs1.segment_id AS seg, rs2.segment_id AS neighbor
            FROM   road_segments rs1
            JOIN   road_segments rs2 ON rs2.u = rs1.v
            WHERE  rs2.segment_id != rs1.segment_id
            UNION
            SELECT rs1.segment_id AS seg, rs2.segment_id AS neighbor
            FROM   road_segments rs1
            JOIN   road_segments rs2 ON rs2.v = rs1.u
            WHERE  rs2.segment_id != rs1.segment_id
        ),
        vol AS (
            SELECT segment_id,
                   AVG(total_passing_vehicle_volume::float) AS avg_vol
            FROM   traffic_counts
            WHERE  segment_id IS NOT NULL
            GROUP  BY segment_id
        )
        SELECT rs.segment_id,
               COALESCE(AVG(vol.avg_vol), 0.0) AS neighbor_avg_volume
        FROM   road_segments rs
        LEFT JOIN adj ON adj.seg          = rs.segment_id
        LEFT JOIN vol ON vol.segment_id   = adj.neighbor
        GROUP BY rs.segment_id
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def build_feature_matrix(engine=None) -> pd.DataFrame:
    """Join road attributes with aggregated traffic volume and all bottleneck features."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            rs.segment_id,
            rs.highway,
            rs.lanes::float                              AS lanes,
            rs.maxspeed,
            rs.length,
            rs.oneway::int                               AS oneway,
            rs.betweenness,
            AVG(tc.total_passing_vehicle_volume::float)  AS avg_volume,
            MAX(tc.total_passing_vehicle_volume::float)  AS max_volume
        FROM   road_segments rs
        LEFT JOIN traffic_counts tc USING (segment_id)
        GROUP  BY rs.segment_id, rs.highway, rs.lanes,
                  rs.maxspeed, rs.length, rs.oneway, rs.betweenness
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    # fill missing lanes with median per highway type, then global fallback
    df["lanes"] = df.groupby("highway")["lanes"].transform(
        lambda x: x.fillna(x.median())
    )
    df["lanes"] = df["lanes"].fillna(df["lanes"].median())

    # parse maxspeed (e.g. "30 mph" → 30.0)
    extracted = df["maxspeed"].str.extract(r"(\d+)")[0].astype(float)
    df["speed_limit"] = extracted.fillna(extracted.median())
    df = df.drop(columns="maxspeed")

    # V/C ratio target: volume relative to road capacity (lanes × speed limit)
    capacity = df["lanes"] * df["speed_limit"]
    vc_raw = df["avg_volume"] / capacity.replace(0, float("nan"))
    labeled = vc_raw[df["avg_volume"].notna() & (df["avg_volume"] > 0)]
    p99 = labeled.quantile(0.99) if len(labeled) > 0 else 1.0
    if pd.isna(p99) or p99 == 0:
        p99 = 1.0
    df["congestion_score"] = (vc_raw / p99).clip(0, 1).fillna(0)

    # one-hot encode highway type
    highway_dummies = pd.get_dummies(df["highway"], prefix="hw")
    df = pd.concat([df.drop(columns="highway"), highway_dummies], axis=1)

    print("  Computing intersection density …")
    density = _intersection_density(engine)
    df = df.merge(density, on="segment_id", how="left")
    df["intersection_density"] = df["intersection_density"].fillna(0).astype(int)

    print("  Computing topology features …")
    topo = _topology_features(engine)
    df = df.merge(topo, on="segment_id", how="left")
    df["fan_in_count"]              = df["fan_in_count"].fillna(0).astype(int)
    df["lane_drop_downstream"]      = df["lane_drop_downstream"].fillna(0).astype(int)
    df["downstream_capacity_ratio"] = df["downstream_capacity_ratio"].fillna(1.0)

    print("  Computing geometry features …")
    geom = _geometry_features(engine)
    df = df.merge(geom, on="segment_id", how="left")
    df["curvature_ratio"] = df["curvature_ratio"].fillna(1.0)

    print("  Computing proximity features …")
    prox = _proximity_features(engine)
    df = df.merge(prox, on="segment_id", how="left")
    df["is_near_ramp"]          = df["is_near_ramp"].fillna(0).astype(int)
    df["traffic_signal_count"]  = df["traffic_signal_count"].fillna(0).astype(int)

    print("  Computing neighbor volume …")
    nbr = _neighbor_volume(engine)
    df = df.merge(nbr, on="segment_id", how="left")
    df["neighbor_avg_volume"] = df["neighbor_avg_volume"].fillna(0.0)

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
