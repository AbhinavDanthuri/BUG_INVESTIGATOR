"""Generate MySQL DDL from the SQLAlchemy models -> sql/schema.sql.

Run from backend/:  python -m scripts.export_schema
"""
from pathlib import Path

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.db.models import Base

OUT = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def main() -> None:
    dialect = mysql.dialect()
    parts = ["-- Generated from app/db/models.py by scripts/export_schema.py. Do not edit by hand.\n"]
    for table in Base.metadata.sorted_tables:
        parts.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";\n")
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            parts.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
