"""Verify PostGIS is reachable and the postgis extension is enabled.

Usage:
    python db_check.py
"""
from sqlalchemy import text
from src.db import get_engine, ensure_postgis


def main():
    engine = get_engine()
    ensure_postgis(engine)

    with engine.connect() as conn:
        version = conn.execute(text("SELECT postgis_full_version()")).scalar()
        print("PostGIS OK")
        print(version)


if __name__ == "__main__":
    main()
