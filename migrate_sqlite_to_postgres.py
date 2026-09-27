"""Copy an existing V11 SQLite database into PostgreSQL.

Usage:
    python migrate_sqlite_to_postgres.py \
      --sqlite sqlite:///./platform_data.db \
      --postgres postgresql+psycopg://user:password@localhost/cv_analyzer

The target schema is created before data is copied. Existing target rows with
matching primary keys are replaced so the operation can be re-run safely.
"""
from __future__ import annotations

import argparse

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from database import create_db_engine
from models import Application, Base, Candidate, Job, User


def copy_rows(source: Session, target: Session) -> None:
    for model in (User, Job, Candidate, Application):
        target.execute(delete(model))
        target.commit()
        rows = source.scalars(select(model)).all()
        if rows:
            for row in rows:
                data = {column.name: getattr(row, column.name) for column in model.__table__.columns}
                target.add(model(**data))
            target.commit()
        print(f"Copied {len(rows):>5} rows: {model.__tablename__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, help="SQLite SQLAlchemy URL, e.g. sqlite:///./platform_data.db")
    parser.add_argument("--postgres", required=True, help="PostgreSQL SQLAlchemy URL")
    args = parser.parse_args()

    source_engine = create_db_engine(args.sqlite)
    target_engine = create_db_engine(args.postgres)
    Base.metadata.create_all(target_engine)

    Source = sessionmaker(bind=source_engine, expire_on_commit=False)
    Target = sessionmaker(bind=target_engine, expire_on_commit=False)

    try:
        with Source() as source, Target() as target:
            copy_rows(source, target)
    finally:
        source_engine.dispose()
        target_engine.dispose()

    print("SQLite → PostgreSQL migration completed.")


if __name__ == "__main__":
    main()
