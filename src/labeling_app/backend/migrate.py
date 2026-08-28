"""One-off schema migration: add columns introduced after the table was first
created (SQLAlchemy's Base.metadata.create_all() only creates missing tables,
it never alters existing ones). Safe to re-run -- uses IF NOT EXISTS.

Usage: DATABASE_URL="..." python migrate.py
"""
from database import engine
from sqlalchemy import text

STATEMENTS = [
    "ALTER TABLE samples ADD COLUMN IF NOT EXISTS mt_en TEXT",
    "ALTER TABLE samples ADD COLUMN IF NOT EXISTS final_mt_text TEXT",
]

if __name__ == "__main__":
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Đang chạy: {stmt}")
            conn.execute(text(stmt))
    print("Xong migration.")
