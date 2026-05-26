"""Shared database connection helpers."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = os.environ["DB_URL"]


def get_engine():
    return create_engine(DB_URL)


def ensure_postgis(engine=None):
    """Create the postgis extension if it doesn't exist yet."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()
