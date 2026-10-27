"""Load the pinned MovieTweetings snapshot into the local MySQL database."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHUNK_SIZE = 2_000


def read_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(
        DATA / "movies.dat",
        sep=r"::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "string"},
        encoding="utf-8",
    )
    ratings = pd.read_csv(
        DATA / "ratings.dat",
        sep=r"::",
        engine="python",
        names=["user_id", "movie_id", "rating", "rating_timestamp"],
        dtype={"user_id": "int64", "movie_id": "string"},
        encoding="utf-8",
    )
    movies["movie_id"] = movies["movie_id"].str.strip()
    movies["title"] = movies["title"].fillna("").str.strip()
    movies["genres"] = movies["genres"].fillna("").str.strip()
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["release_year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    ratings["movie_id"] = ratings["movie_id"].str.strip()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    ratings["rating_timestamp"] = pd.to_numeric(
        ratings["rating_timestamp"], errors="coerce"
    ).astype("Int64")
    return movies, ratings


def chunks(records: list[dict], size: int):
    for start in range(0, len(records), size):
        yield records[start : start + size]


def upsert(engine, query: str, records: list[dict], table_name: str) -> None:
    for index, batch in enumerate(chunks(records, CHUNK_SIZE), start=1):
        with engine.begin() as connection:
            connection.execute(text(query), batch)
        if index % 10 == 0 or index == 1:
            print(f"{table_name}: {min(index * CHUNK_SIZE, len(records)):,}/{len(records):,}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is missing. Copy .env.example to .env and configure MySQL."
        )

    movies, ratings = read_source()
    unknown_movies = set(ratings["movie_id"]) - set(movies["movie_id"])
    if unknown_movies:
        raise ValueError(f"Ratings reference {len(unknown_movies)} unknown movies.")

    engine = create_engine(database_url, pool_pre_ping=True)
    user_ids = sorted(ratings["user_id"].unique().tolist())
    user_records = [{"user_id": int(user_id)} for user_id in user_ids]
    movie_records = [
        {
            "movie_id": str(row.movie_id),
            "title": str(row.title)[:255],
            "release_year": None if pd.isna(row.release_year) else int(row.release_year),
            "genres": str(row.genres)[:255],
        }
        for row in movies.itertuples(index=False)
    ]
    rating_records = [
        {
            "user_id": int(row.user_id),
            "movie_id": str(row.movie_id),
            "rating": float(row.rating),
            "rating_timestamp": (
                None if pd.isna(row.rating_timestamp) else int(row.rating_timestamp)
            ),
            "source": "MovieTweetings",
        }
        for row in ratings.itertuples(index=False)
    ]

    upsert(
        engine,
        "INSERT IGNORE INTO users (user_id) VALUES (:user_id)",
        user_records,
        "users",
    )
    upsert(
        engine,
        """INSERT INTO movies (movie_id, title, release_year, genres)
           VALUES (:movie_id, :title, :release_year, :genres)
           ON DUPLICATE KEY UPDATE title=VALUES(title),
             release_year=VALUES(release_year), genres=VALUES(genres)""",
        movie_records,
        "movies",
    )
    upsert(
        engine,
        """INSERT INTO ratings
             (user_id, movie_id, rating, rating_timestamp, source)
           VALUES
             (:user_id, :movie_id, :rating, :rating_timestamp, :source)
           ON DUPLICATE KEY UPDATE rating=VALUES(rating),
             rating_timestamp=VALUES(rating_timestamp), source=VALUES(source)""",
        rating_records,
        "ratings",
    )

    with engine.connect() as connection:
        counts = connection.execute(
            text(
                "SELECT 'users' AS table_name, COUNT(*) AS row_count FROM users "
                "UNION ALL SELECT 'movies', COUNT(*) FROM movies "
                "UNION ALL SELECT 'ratings', COUNT(*) FROM ratings"
            )
        ).all()
    print("\nImported row counts:")
    for table_name, row_count in counts:
        print(f"  {table_name}: {row_count:,}")


if __name__ == "__main__":
    main()
