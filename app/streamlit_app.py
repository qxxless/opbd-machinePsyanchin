"""Small Streamlit client for the Movie Recommendation API."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Кино-рекомендации", page_icon="🎬", layout="wide")
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def api_get(path: str, **params):
    response = requests.get(f"{API}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def show_movies(items):
    if not items:
        st.info("Подходящих фильмов пока нет.")
        return
    frame = pd.DataFrame(items)
    columns = [col for col in ["title", "release_year", "genres", "rating_count", "mean_rating", "score"] if col in frame]
    frame = frame[columns].rename(
        columns={
            "title": "Фильм",
            "release_year": "Год",
            "genres": "Жанры",
            "rating_count": "Число оценок",
            "mean_rating": "Средняя оценка",
            "score": "Оценка модели",
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


st.title("🎬 Рекомендательная система фильмов")
st.caption("MovieTweetings · коллаборативная фильтрация · контентные рекомендации · MySQL")

with st.sidebar:
    st.subheader("Пользователь")
    example_users = []
    try:
        example_users = api_get("/users", limit=50)
    except requests.RequestException as exc:
        st.warning(f"API пока недоступен: {exc}")
    default_user = int(example_users[0]) if example_users else 1
    user_id = st.number_input("Обезличенный ID", min_value=1, value=default_user, step=1)
    st.markdown("Оценка сохраняется в MySQL через API.")

tab_popular, tab_movie, tab_personal, tab_rate, tab_help = st.tabs(
    ["Популярное", "Похожие фильмы", "Персонально", "Оценить фильм", "Справка"]
)

with tab_popular:
    st.subheader("Популярные фильмы")
    limit_popular = st.slider("Количество", 5, 30, 10, key="popular_limit")
    if st.button("Показать популярное", key="popular_button"):
        try:
            show_movies(api_get("/recommendations/popular", limit=limit_popular))
        except requests.RequestException as exc:
            st.error(f"Не удалось получить рекомендации: {exc}")

with tab_movie:
    st.subheader("Рекомендации по выбранному фильму")
    query = st.text_input("Найти фильм", value="Matrix", key="movie_search")
    method_movie = st.selectbox("Метод", ["collaborative", "content"], format_func=lambda x: {
        "collaborative": "Коллаборативный (по фильмам)",
        "content": "Контентный (название и жанры)",
    }[x])
    try:
        found = api_get("/movies", q=query, limit=30)
    except requests.RequestException:
        found = []
    if found:
        labels = {f'{m["title"]} [{m["movie_id"]}]': m["movie_id"] for m in found}
        selected_label = st.selectbox("Выберите фильм", list(labels))
        if st.button("Подобрать похожие", key="similar_button"):
            try:
                items = api_get(
                    f'/recommendations/movie/{labels[selected_label]}',
                    method=method_movie,
                    limit=10,
                )
                show_movies(items)
            except requests.RequestException as exc:
                st.error(f"Не удалось получить рекомендации: {exc}")
    else:
        st.info("Введите часть названия; для поиска нужен запущенный API.")

with tab_personal:
    st.subheader("Персональная выдача")
    method_user = st.selectbox(
        "Модель",
        ["hybrid", "bpr", "user", "item", "content"],
        format_func=lambda x: {
            "hybrid": "Гибридная",
            "bpr": "Персональная BPR",
            "user": "По похожим пользователям",
            "item": "По похожим фильмам",
            "content": "По жанрам и названию",
        }[x],
    )
    if st.button("Получить рекомендации", key="personal_button"):
        try:
            show_movies(
                api_get(
                    f"/recommendations/user/{int(user_id)}",
                    method=method_user,
                    limit=10,
                )
            )
        except requests.RequestException as exc:
            st.error(f"Не удалось получить рекомендации: {exc}")

with tab_rate:
    st.subheader("Оценить фильм")
    rate_query = st.text_input("Название фильма", value="", key="rate_search")
    try:
        rate_found = api_get("/movies", q=rate_query, limit=30) if rate_query else []
    except requests.RequestException:
        rate_found = []
    if rate_found:
        rate_labels = {f'{m["title"]} [{m["movie_id"]}]': m["movie_id"] for m in rate_found}
        rate_label = st.selectbox("Фильм", list(rate_labels), key="rate_movie")
        score = st.slider("Ваша оценка (0–10)", 0, 10, 7, 1)
        if st.button("Сохранить оценку", key="save_rating"):
            try:
                response = requests.post(
                    f"{API}/ratings",
                    json={
                        "user_id": int(user_id),
                        "movie_id": rate_labels[rate_label],
                        "rating": score,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                st.success("Оценка сохранена в базе данных.")
            except requests.RequestException as exc:
                st.error(f"Не удалось сохранить оценку: {exc}")
    else:
        st.info("Введите название фильма для поиска.")

with tab_help:
    st.subheader("Справка по API")
    try:
        help_data = api_get("/help")
        for endpoint in help_data["endpoints"]:
            st.markdown(f'**{endpoint["method"]} {endpoint["path"]}** — {endpoint["purpose"]}')
        st.markdown(f"Интерактивная документация: {API}/docs")
    except requests.RequestException as exc:
        st.error(f"Справка API недоступна: {exc}")
