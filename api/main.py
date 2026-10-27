"""FastAPI layer for the movie recommender."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "models" / "recommender_bundle.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. Run the notebook first."
    )

MODEL = joblib.load(MODEL_PATH)
DATABASE_URL = os.getenv("DATABASE_URL")
ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
MOVIES = MODEL["movies"].copy()
MOVIES["movie_id"] = MOVIES["movie_id"].astype(str)
MOVIE_LOOKUP = MOVIES.set_index("movie_id", drop=False)
POPULAR = MODEL["popularity"].copy()
POPULAR["movie_id"] = POPULAR["movie_id"].astype(str)
POPULAR_SCORE = dict(
    zip(POPULAR["movie_id"], POPULAR["bayesian_score"].astype(float))
)

app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "Учебный API рекомендательной системы на MovieTweetings. "
        "Модели обучены на закреплённом снимке данных."
    ),
    version="1.0.0",
)


class RatingInput(BaseModel):
    user_id: int = Field(gt=0, description="Обезличенный ID пользователя")
    movie_id: str = Field(min_length=1, max_length=10, description="ID фильма")
    rating: float = Field(ge=0, le=10, description="Оценка от 0 до 10")


def movie_payload(movie_id: str, score: float | None = None) -> dict:
    row = MOVIE_LOOKUP.loc[movie_id]
    result = {
        "movie_id": str(row["movie_id"]),
        "title": str(row["title"]),
        "genres": str(row["genres"]),
        "release_year": (
            None if pd.isna(row["release_year"]) else int(row["release_year"])
        ),
    }
    if score is not None and np.isfinite(score):
        result["score"] = round(float(score), 4)
    return result


def db_profile(user_id: int) -> list[tuple[str, float]]:
    if ENGINE is None:
        return []
    try:
        with ENGINE.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT movie_id, rating FROM ratings "
                    "WHERE user_id=:user_id ORDER BY rating_timestamp"
                ),
                {"user_id": user_id},
            ).all()
        return [(str(movie_id), float(rating)) for movie_id, rating in rows]
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"MySQL unavailable: {exc}") from exc


def fallback_profile(user_id: int) -> list[tuple[str, float]]:
    user_to_idx = MODEL["user_to_idx"]
    if user_id not in user_to_idx:
        return []
    row = MODEL["ratings"].getrow(user_to_idx[user_id])
    movie_ids = MODEL["model_movie_ids"]
    return [
        (str(movie_ids[index]), float(rating))
        for index, rating in zip(row.indices, row.data)
    ]


def user_cf_scores(user_id: int) -> dict[str, float]:
    user_to_idx = MODEL["user_to_idx"]
    if user_id not in user_to_idx:
        return {}
    user_index = user_to_idx[user_id]
    similarities = MODEL["user_similarity"].getrow(user_index)
    if similarities.nnz == 0:
        return {}
    order = np.argsort(similarities.data)[::-1][:MODEL["user_neighbors"]]
    neighbor_ids = similarities.indices[order]
    weights = similarities.data[order].astype(np.float32)
    keep = weights > 0
    neighbor_ids, weights = neighbor_ids[keep], weights[keep]
    if len(neighbor_ids) == 0:
        return {}
    weight_row = csr_matrix(
        (weights, (np.zeros(len(weights), dtype=int), neighbor_ids)),
        shape=(1, MODEL["ratings"].shape[0]),
    )
    numerator = weight_row @ MODEL["centered_ratings"]
    denominator = weight_row @ MODEL["rating_presence"]
    numerator = numerator.toarray().ravel()
    denominator = denominator.toarray().ravel()
    scores = np.full(len(MODEL["model_movie_ids"]), np.nan, dtype=np.float32)
    valid = denominator > 0
    scores[valid] = MODEL["user_mean"][user_index] + (
        numerator[valid] / denominator[valid]
    )
    return {
        str(movie_id): float(scores[index])
        for index, movie_id in enumerate(MODEL["model_movie_ids"])
        if np.isfinite(scores[index])
    }


def item_cf_scores(profile: list[tuple[str, float]]) -> dict[str, float]:
    movie_to_idx = MODEL["movie_to_idx"]
    known = [(movie_to_idx[movie_id], rating) for movie_id, rating in profile if movie_id in movie_to_idx]
    if not known:
        return {}
    movie_indices = np.array([item[0] for item in known], dtype=np.int32)
    values = np.array([item[1] for item in known], dtype=np.float32)
    baseline = float(values.mean())
    residuals = values - baseline
    row = csr_matrix(
        (residuals, (np.zeros(len(movie_indices), dtype=int), movie_indices)),
        shape=(1, len(MODEL["model_movie_ids"])),
    )
    presence = csr_matrix(
        (np.ones(len(movie_indices), dtype=np.float32), (np.zeros(len(movie_indices), dtype=int), movie_indices)),
        shape=row.shape,
    )
    numerator = (row @ MODEL["item_similarity"]).toarray().ravel()
    denominator = (presence @ MODEL["item_similarity"]).toarray().ravel()
    scores = np.full(len(MODEL["model_movie_ids"]), np.nan, dtype=np.float32)
    valid = denominator > 0
    scores[valid] = baseline + numerator[valid] / denominator[valid]
    return {
        str(movie_id): float(scores[index])
        for index, movie_id in enumerate(MODEL["model_movie_ids"])
        if np.isfinite(scores[index])
    }


def content_scores(profile: list[tuple[str, float]]) -> dict[str, float]:
    liked = [
        (MODEL["content_to_idx"][movie_id], rating)
        for movie_id, rating in profile
        if rating >= MODEL["relevance_threshold"]
        and movie_id in MODEL["content_to_idx"]
    ]
    if not liked:
        return {}
    indices = np.array([item[0] for item in liked], dtype=np.int32)
    weights = np.array(
        [max(rating - MODEL["relevance_threshold"], 0.5) for _, rating in liked],
        dtype=np.float32,
    )
    profile_vector = MODEL["content_matrix"][indices].multiply(
        weights[:, None]
    ).sum(axis=0)
    profile_vector = np.asarray(profile_vector).ravel()
    scores = MODEL["content_matrix"] @ profile_vector
    if hasattr(scores, "toarray"):
        scores = scores.toarray().ravel()
    return {
        str(movie_id): float(scores[index])
        for index, movie_id in enumerate(MODEL["content_movie_ids"])
        if np.isfinite(scores[index]) and scores[index] > 0
    }


def percentile_map(scores: dict[str, float], candidates: list[str]) -> dict[str, float]:
    available = [(movie_id, scores[movie_id]) for movie_id in candidates if movie_id in scores]
    if not available:
        return {}
    values = pd.Series([value for _, value in available], dtype=float)
    ranks = values.rank(method="average", pct=True).to_numpy()
    return {movie_id: float(rank) for (movie_id, _), rank in zip(available, ranks)}


def bpr_scores(user_id: int) -> dict[str, float]:
    model = MODEL.get("bpr_model")
    if not model or user_id not in model["user_to_idx"]:
        return {}
    user_index = model["user_to_idx"][user_id]
    scores = model["P"][user_index] @ model["Q"].T + model["item_bias"]
    return {
        str(movie_id): float(score)
        for movie_id, score in zip(model["movie_ids"], scores)
    }


def personalized(user_id: int, method: str, limit: int) -> list[dict]:
    profile = db_profile(user_id)
    if not profile:
        profile = fallback_profile(user_id)
    seen = {movie_id for movie_id, _ in profile}
    candidates = [
        str(movie_id)
        for movie_id in MOVIES["movie_id"]
        if str(movie_id) not in seen
    ]

    if method == "user":
        user_scores = user_cf_scores(user_id)
        scores = {key: value for key, value in user_scores.items() if key in candidates}
    elif method == "item":
        item_scores = item_cf_scores(profile)
        scores = {key: value for key, value in item_scores.items() if key in candidates}
    elif method == "content":
        text_scores = content_scores(profile)
        scores = {key: value for key, value in text_scores.items() if key in candidates}
    elif method == "bpr":
        pairwise_scores = bpr_scores(user_id)
        scores = {key: value for key, value in pairwise_scores.items() if key in candidates}
        if not scores:
            # BPR has no learned vector for a new user; use the metadata profile.
            text_scores = content_scores(profile)
            text_rank = percentile_map(text_scores, candidates)
            pop_rank = percentile_map(POPULAR_SCORE, candidates)
            scores = {
                movie_id: 0.5 * text_rank.get(movie_id, 0.0)
                + 0.5 * pop_rank.get(movie_id, 0.0)
                for movie_id in candidates
            }
    else:
        text_scores = content_scores(profile)
        content_rank = percentile_map(text_scores, candidates)
        pop_rank = percentile_map(POPULAR_SCORE, candidates)
        bpr_rank = percentile_map(bpr_scores(user_id), candidates)
        weights = MODEL.get(
            "hybrid_weights",
            {"bpr": 0.0, "popularity": 0.70, "content": 0.30},
        )
        scores = {}
        for movie_id in candidates:
            components = [
                (float(weights.get("bpr", 0.0)), bpr_rank, movie_id),
                (float(weights.get("popularity", 0.0)), pop_rank, movie_id),
                (float(weights.get("content", 0.0)), content_rank, movie_id),
            ]
            available = [(weight, values[key]) for weight, values, key in components if key in values and weight > 0]
            weight_sum = sum(weight for weight, _ in available)
            scores[movie_id] = (
                sum(weight * value for weight, value in available) / weight_sum
                if weight_sum
                else POPULAR_SCORE.get(movie_id, 0.0)
            )

    if not scores:
        scores = {
            movie_id: POPULAR_SCORE.get(movie_id, 0.0)
            for movie_id in candidates
            if movie_id in POPULAR_SCORE
        }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [movie_payload(movie_id, score) for movie_id, score in ranked]


@app.get("/health")
def health() -> dict:
    database_ok = False
    if ENGINE is not None:
        try:
            with ENGINE.connect() as connection:
                connection.execute(text("SELECT 1"))
            database_ok = True
        except SQLAlchemyError:
            database_ok = False
    return {
        "status": "ok" if database_ok else "model_ready_database_unavailable",
        "model_loaded": True,
        "database_connected": database_ok,
    }


@app.get("/help")
def help_page() -> dict:
    return {
        "description": "Команды API для приложения рекомендательной системы.",
        "endpoints": [
            {"method": "GET", "path": "/recommendations/popular?limit=10", "purpose": "Топ популярных фильмов."},
            {"method": "GET", "path": "/recommendations/movie/{movie_id}?method=collaborative|content&limit=10", "purpose": "Фильмы, похожие на выбранный."},
            {"method": "GET", "path": "/recommendations/user/{user_id}?method=hybrid|bpr|user|item|content&limit=10", "purpose": "Персональная выдача."},
            {"method": "GET", "path": "/movies?q=matrix&limit=20", "purpose": "Поиск фильма по названию."},
            {"method": "GET", "path": "/users?limit=20", "purpose": "Примеры обезличенных ID пользователей."},
            {"method": "POST", "path": "/ratings", "purpose": "Сохранить или обновить оценку в MySQL.", "body": {"user_id": 1, "movie_id": "0133093", "rating": 9}},
            {"method": "GET", "path": "/health", "purpose": "Проверить модель и соединение с БД."},
            {"method": "GET", "path": "/docs", "purpose": "Интерактивная документация OpenAPI."},
        ],
    }


@app.get("/movies")
def search_movies(
    q: str = Query(default="", max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    frame = MOVIES
    if q.strip():
        frame = frame[frame["title"].str.contains(q.strip(), case=False, regex=False)]
    return [movie_payload(str(row.movie_id)) for row in frame.head(limit).itertuples()]


@app.get("/users")
def list_users(limit: int = Query(default=20, ge=1, le=100)) -> list[int]:
    return [int(user_id) for user_id in MODEL["model_user_ids"][:limit]]


@app.get("/recommendations/popular")
def popular(limit: int = Query(default=10, ge=1, le=50)) -> list[dict]:
    result = []
    for row in POPULAR.head(limit).itertuples(index=False):
        result.append(
            {
                "movie_id": str(row.movie_id),
                "title": str(row.title),
                "genres": str(row.genres),
                "rating_count": int(row.rating_count),
                "mean_rating": round(float(row.mean_rating), 3),
                "score": round(float(row.bayesian_score), 4),
            }
        )
    return result


@app.get("/recommendations/movie/{movie_id}")
def similar_movies(
    movie_id: str,
    method: Literal["collaborative", "content"] = "collaborative",
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    if movie_id not in MOVIE_LOOKUP.index:
        raise HTTPException(status_code=404, detail="Movie not found.")
    if method == "collaborative":
        index = MODEL["movie_to_idx"].get(movie_id)
        if index is None:
            raise HTTPException(
                status_code=404,
                detail="This movie has no collaborative model profile; use method=content.",
            )
        row = MODEL["item_similarity"].getrow(index)
        order = np.argsort(row.data)[::-1]
        pairs = [
            (MODEL["model_movie_ids"][row.indices[i]], float(row.data[i]))
            for i in order
            if MODEL["model_movie_ids"][row.indices[i]] != movie_id
        ][:limit]
    else:
        index = MODEL["content_to_idx"].get(movie_id)
        if index is None:
            raise HTTPException(status_code=404, detail="Movie content vector unavailable.")
        row = MODEL["content_matrix"][index] @ MODEL["content_matrix"].T
        scores = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
        order = np.argsort(scores)[::-1]
        pairs = [
            (MODEL["content_movie_ids"][i], float(scores[i]))
            for i in order
            if MODEL["content_movie_ids"][i] != movie_id and scores[i] > 0
        ][:limit]
    return [movie_payload(str(key), score) for key, score in pairs]


@app.get("/recommendations/user/{user_id}")
def user_recommendations(
    user_id: int,
    method: Literal["hybrid", "bpr", "user", "item", "content"] = "hybrid",
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    if user_id <= 0:
        raise HTTPException(status_code=422, detail="user_id must be positive.")
    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured; add a .env file and restart the API.",
        )
    return personalized(user_id, method, limit)


@app.post("/ratings")
def save_rating(payload: RatingInput) -> dict:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured.")
    if payload.movie_id not in MOVIE_LOOKUP.index:
        raise HTTPException(status_code=404, detail="Movie not found.")
    try:
        with ENGINE.begin() as connection:
            connection.execute(
                text("INSERT IGNORE INTO users (user_id) VALUES (:user_id)"),
                {"user_id": payload.user_id},
            )
            connection.execute(
                text(
                    """INSERT INTO ratings
                         (user_id, movie_id, rating, rating_timestamp, source)
                       VALUES (:user_id, :movie_id, :rating, :timestamp, 'app')
                       ON DUPLICATE KEY UPDATE
                         rating=VALUES(rating), rating_timestamp=VALUES(rating_timestamp),
                         source='app'"""
                ),
                {
                    "user_id": payload.user_id,
                    "movie_id": payload.movie_id,
                    "rating": payload.rating,
                    "timestamp": int(time.time()),
                },
            )
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Could not save rating: {exc}") from exc
    return {
        "saved": True,
        "user_id": payload.user_id,
        "movie_id": payload.movie_id,
        "rating": payload.rating,
    }
