from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
DATA_PATH = BASE_DIR / "real_estate_data_ru_FINAL2.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR.parent / "real_estate_data_ru_FINAL2.csv"
TEST_MAE = 107361.0

st.set_page_config(
    page_title="Real Estate",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Загрузка модели и данных
# -----------------------------
@st.cache_resource

def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data

def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)
    df["цена"] = pd.to_numeric(df.get("цена"), errors="coerce")
    df["площадь_м2"] = pd.to_numeric(df.get("площадь_м2"), errors="coerce")

    # Адрес хранится как Город/Район/Микрорайон.
    address = df["адрес"].fillna("").astype(str).str.split("/", expand=True)
    df["город"] = address[0].replace("", np.nan)
    df["район"] = address[1].replace("", np.nan) if address.shape[1] > 1 else np.nan
    df["микрорайон"] = address[2].replace("", np.nan) if address.shape[1] > 2 else np.nan

    # Для аналитики берём только корректные цены в TRY и два типа объявлений.
    analytics = df[
        (df["валюта"] == "Турецкая лира")
        & (df["тип_объявления"].isin(["Продажа", "Аренда"]))
        & (df["цена"] > 0)
    ].copy()
    return df, analytics


model = load_model()
df, analytics = load_data()

# Категории, реально поддерживаемые сохранённым Pipeline.
prep = model.named_steps["prep"]
cat_cols = list(prep.transformers_[1][2])
enc = prep.transformers_[1][1].named_steps["encoder"]
model_categories = {c: [str(x) for x in cats] for c, cats in zip(cat_cols, enc.categories_)}


def floor_to_number(value):
    s = str(value).strip()
    special = {
        "Уровень -1": -1,
        "Уровень -2": -2,
        "Уровень -3": -3,
        "Уровень -4": -4,
        "Подвальный этаж": -1,
        "Цокольный/нулевой этаж": 0,
        "Входной этаж": 0,
        "Садовый этаж": 0,
        "Высокий первый этаж": 1,
    }
    if s in special:
        return float(special[s])
    if "и более" in s:
        m = re.search(r"\d+", s)
        return float(m.group()) if m else np.nan
    if re.fullmatch(r"-?\d+(?:\.0)?", s):
        return float(s)
    return np.nan


def money(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):,.0f}".replace(",", " ") + " TRY"


def money_short(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f} млн"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f} тыс."
    return f"{x:.0f}"


def style_fig(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=65, b=40),
        hovermode="x unified",
        legend_title_text="",
    )
    return fig


# -----------------------------
# Боковая навигация
# -----------------------------
st.sidebar.title("Real Estate")
st.sidebar.caption("Анализ и прогноз стоимости недвижимости")
page = st.sidebar.radio("Раздел", ["Прогноз", "Дашборд", "Справка"])
st.sidebar.divider()
st.sidebar.caption("Финальная модель: LightGBM")

# -----------------------------
# ПРОГНОЗ
# -----------------------------
if page == "Прогноз":
    st.title("Прогноз стоимости недвижимости")
    st.write("Введите характеристики объекта. Модель вернёт ориентировочную стоимость в турецких лирах.")

    left, right = st.columns(2, gap="large")
    with left:
        property_type = st.selectbox("Тип недвижимости", model_categories["тип_недвижимости"])
        subtype = st.selectbox("Подтип", model_categories["подтип"])
        ad_type = st.selectbox("Тип объявления", model_categories["тип_объявления"])
        area = st.number_input("Площадь, м²", min_value=1.0, max_value=100000.0, value=60.0, step=1.0)
        days = st.number_input("Дней на рынке", min_value=0.0, max_value=5000.0, value=30.0, step=1.0)
        heating = st.selectbox("Тип отопления", model_categories["тип_отопления"])
        age = st.number_input("Возраст здания, лет", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

    with right:
        total_floors = st.number_input("Всего этажей", min_value=1.0, max_value=200.0, value=10.0, step=1.0)
        floor_category = st.selectbox("Этаж / категория этажа", model_categories["этаж_категория"])
        floor_number = floor_to_number(floor_category)
        if np.isnan(floor_number):
            floor_number = float(st.number_input("Числовой этаж", min_value=-10.0, max_value=200.0, value=1.0, step=1.0))
        rooms = st.number_input("Количество комнат", min_value=0.0, max_value=30.0, value=2.0, step=1.0)
        year = st.number_input("Год начала объявления", min_value=2000, max_value=2100, value=2019, step=1)
        month = st.number_input("Месяц начала объявления", min_value=1, max_value=12, value=1, step=1)

        cities = model_categories["город"]
        city = st.selectbox("Город", cities)

        # Район подбирается по выбранному городу. Это предотвращает нелогичные
        # сочетания и делает форму понятнее.
        district_source = df[df["город"] == city]["район"].dropna().astype(str).unique().tolist()
        allowed_districts = [d for d in district_source if d in model_categories["район"]]
        if not allowed_districts:
            allowed_districts = model_categories["район"]
        district = st.selectbox("Район", sorted(allowed_districts))

    st.divider()
    if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
        if area <= 1:
            st.error("Площадь должна быть больше 1 м².")
        elif days < 0:
            st.error("Количество дней не может быть отрицательным.")
        elif total_floors < 1:
            st.error("Количество этажей должно быть не меньше 1.")
        else:
            row = pd.DataFrame([{
                "тип_недвижимости": property_type,
                "подтип": subtype,
                "тип_объявления": ad_type,
                "дней_на_рынке": days,
                "площадь_м2": area,
                "тип_отопления": heating,
                "возраст_число": age,
                "этажей_число": total_floors,
                "этаж_число": floor_number,
                "этаж_категория": floor_category,
                "комнаты_число": rooms,
                "год_начала": year,
                "месяц_начала": month,
                "город": city,
                "район": district,
            }])
            try:
                pred_log = float(model.predict(row)[0])
                price = max(0.0, float(np.expm1(pred_log)))
                low = max(0.0, price - TEST_MAE)
                high = price + TEST_MAE

                st.success(f"Ориентировочная стоимость: **{money(price)}**")
                a, b, c = st.columns(3)
                a.metric("Прогноз", money(price))
                b.metric("Нижняя граница", money(low), delta=f"− {money(TEST_MAE)}")
                c.metric("Верхняя граница", money(high), delta=f"+ {money(TEST_MAE)}")
                st.caption(
                    f"Ориентир рассчитан как прогноз ± MAE тестовой выборки ({money(TEST_MAE)}). "
                    "Это не доверительный интервал и не гарантированная рыночная цена."
                )
            except Exception as exc:
                st.error(f"Не удалось выполнить прогноз: {exc}")

# -----------------------------
# ДАШБОРД
# -----------------------------
elif page == "Дашборд":
    st.title("Аналитический дашборд")
    st.caption("Интерактивный анализ объявлений с корректной ценой в турецких лирах.")

    # Фильтры
    with st.expander("Фильтры", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sale_type = st.selectbox("Тип объявления", ["Все", "Продажа", "Аренда"])
        with f2:
            property_filter = st.selectbox("Подтип", ["Все"] + sorted(analytics["подтип"].dropna().astype(str).unique().tolist()))
        with f3:
            city_filter = st.selectbox("Город", ["Все"] + sorted(analytics["город"].dropna().astype(str).unique().tolist()))
        with f4:
            max_price = float(analytics["цена"].quantile(0.99)) if len(analytics) else 0
            price_limit = st.number_input("Максимальная цена для графиков, TRY", min_value=1.0, value=max_price, step=10000.0)

    d = analytics.copy()
    if sale_type != "Все":
        d = d[d["тип_объявления"] == sale_type]
    if property_filter != "Все":
        d = d[d["подтип"].astype(str) == property_filter]
    if city_filter != "Все":
        d = d[d["город"].astype(str) == city_filter]

    d_graph = d[d["цена"] <= price_limit].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Объектов", f"{len(d):,}".replace(",", " "))
    k2.metric("Медианная цена", money(d["цена"].median()))
    k3.metric("Средняя площадь", f"{d['площадь_м2'].mean():,.1f} м²".replace(",", " "))
    k4.metric("Городов", f"{d['город'].nunique()}")

    st.divider()
    tab_overview, tab_cities, tab_types, tab_area, tab_model = st.tabs(
        ["Обзор", "Города", "Типы", "Площадь", "Модель"]
    )

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            hist = px.histogram(
                d_graph,
                x="цена",
                nbins=50,
                title="Распределение цен",
                labels={"цена": "Цена, TRY", "count": "Количество объектов"},
            )
            hist.update_xaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(hist), use_container_width=True)
        with c2:
            by_ad = d.groupby("тип_объявления", as_index=False).agg(
                медианная_цена=("цена", "median"),
                количество=("цена", "size"),
            )
            fig = px.bar(
                by_ad,
                x="тип_объявления",
                y="медианная_цена",
                text="медианная_цена",
                title="Медианная цена: продажа и аренда",
                labels={"тип_объявления": "Тип", "медианная_цена": "Медианная цена, TRY"},
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("Ключевые статистики")
        stats = pd.DataFrame({
            "Показатель": ["Минимум", "25%", "Медиана", "75%", "Максимум", "Среднее"],
            "Значение": [
                d["цена"].min(), d["цена"].quantile(.25), d["цена"].median(),
                d["цена"].quantile(.75), d["цена"].max(), d["цена"].mean()
            ]
        })
        stats["Значение"] = stats["Значение"].map(money)
        st.dataframe(stats, use_container_width=True, hide_index=True)

    with tab_cities:
        st.subheader("Сравнение городов")
        city_sale = analytics[analytics["тип_объявления"] == "Продажа"].groupby("город", as_index=False).agg(
            медианная_цена=("цена", "median"), количество=("цена", "size")
        ).sort_values("медианная_цена", ascending=False)
        city_rent = analytics[analytics["тип_объявления"] == "Аренда"].groupby("город", as_index=False).agg(
            медианная_цена=("цена", "median"), количество=("цена", "size")
        ).sort_values("медианная_цена", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                city_sale.head(15).sort_values("медианная_цена"),
                x="медианная_цена", y="город", orientation="h",
                title="Продажа — топ-15 городов по медианной цене",
                labels={"медианная_цена": "Медианная цена, TRY", "город": "Город"},
                text="медианная_цена",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
        with c2:
            fig = px.bar(
                city_rent.head(15).sort_values("медианная_цена"),
                x="медианная_цена", y="город", orientation="h",
                title="Аренда — топ-15 городов по медианной цене",
                labels={"медианная_цена": "Медианная цена, TRY", "город": "Город"},
                text="медианная_цена",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)

        st.info("Продажа и аренда показаны отдельно: это разные типы цены, поэтому объединять их в один рейтинг некорректно.")

    with tab_types:
        type_stats = d.groupby("подтип", as_index=False).agg(
            количество=("цена", "size"), медианная_цена=("цена", "median")
        ).sort_values("количество", ascending=False).head(15)
        fig = px.bar(
            type_stats.sort_values("количество"), x="количество", y="подтип", orientation="h",
            title="Количество объявлений по типам недвижимости — топ-15",
            labels={"количество": "Количество", "подтип": "Тип недвижимости"},
            text="количество",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                type_stats.sort_values("медианная_цена"), x="медианная_цена", y="подтип", orientation="h",
                title="Медианная цена по типам",
                labels={"медианная_цена": "TRY", "подтип": "Тип"}, text="медианная_цена",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
        with c2:
            ad_counts = d["тип_объявления"].value_counts().reset_index()
            ad_counts.columns = ["тип_объявления", "количество"]
            fig = px.pie(ad_counts, names="тип_объявления", values="количество", hole=.45, title="Продажа и аренда")
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    with tab_area:
        area_df = d[["площадь_м2", "цена", "подтип", "город", "тип_объявления"]].dropna().copy()
        area_df = area_df[(area_df["площадь_м2"] > 0) & (area_df["площадь_м2"] <= area_df["площадь_м2"].quantile(.99))]
        area_df = area_df[area_df["цена"] <= price_limit]
        if len(area_df) > 3000:
            area_df = area_df.sample(3000, random_state=42)
        fig = px.scatter(
            area_df, x="площадь_м2", y="цена", color="тип_объявления",
            hover_data=["подтип", "город"], opacity=.6,
            title="Зависимость цены от площади",
            labels={"площадь_м2": "Площадь, м²", "цена": "Цена, TRY"},
        )
        fig.update_yaxes(tickformat=",.0f")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

        corr = area_df[["площадь_м2", "цена"]].corr().iloc[0, 1] if len(area_df) > 1 else np.nan
        st.metric("Корреляция площади и цены", f"{corr:.3f}" if not pd.isna(corr) else "—")
        st.caption("Корреляция показывает статистическую связь двух признаков, но сама по себе не доказывает причинность.")

    with tab_model:
        st.subheader("Сравнение моделей")
        metrics = pd.DataFrame([
            {"Модель": "Linear Regression", "MAE": 493640.92, "RMSE": 6758841.00, "MAPE": 40.05, "R²": 0.184},
            {"Модель": "Decision Tree", "MAE": 478680.41, "RMSE": 6618101.00, "MAPE": 60.05, "R²": 0.217},
            {"Модель": "Random Forest", "MAE": 478058.08, "RMSE": 6872606.00, "MAPE": 35.80, "R²": 0.156},
            {"Модель": "LightGBM", "MAE": 107361.00, "RMSE": np.nan, "MAPE": np.nan, "R²": 0.4265},
        ])
        st.dataframe(
            metrics.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.2f}", "R²": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(metrics.dropna(subset=["MAE"]), x="Модель", y="MAE", text="MAE", title="MAE — средняя абсолютная ошибка")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fig = px.bar(metrics.dropna(subset=["R²"]), x="Модель", y="R²", text="R²", title="R² — доля объяснённой вариации")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("Важность признаков LightGBM")
        try:
            feature_names = prep.get_feature_names_out()
            importances = model.named_steps["model"].feature_importances_
            imp = pd.DataFrame({"Признак": feature_names, "Важность": importances})
            imp = imp.sort_values("Важность", ascending=False).head(15).sort_values("Важность")
            fig = px.bar(imp, x="Важность", y="Признак", orientation="h", title="Топ-15 признаков")
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
        except Exception as exc:
            st.warning(f"Не удалось получить важность признаков: {exc}")

        st.info("LightGBM используется как финальная модель приложения. Метрики сравнения относятся к сохранённым результатам оценки моделей проекта.")

# -----------------------------
# СПРАВКА
# -----------------------------
else:
    st.title("Справка")
    st.markdown("""
### Назначение
Приложение предназначено для анализа объявлений о недвижимости и прогнозирования ориентировочной стоимости объекта.

### Как пользоваться прогнозом
1. Откройте раздел **Прогноз**.
2. Выберите тип и характеристики объекта.
3. Укажите город и район.
4. Нажмите **Рассчитать стоимость**.
5. Получите прогноз и ориентировочный диапазон.

### Что означает каждое поле
- **Тип недвижимости / подтип** — категория объекта.
- **Тип объявления** — продажа или аренда.
- **Площадь, м²** — площадь объекта.
- **Дней на рынке** — длительность размещения объявления.
- **Возраст здания** — возраст в годах.
- **Всего этажей** — этажность здания.
- **Этаж / категория этажа** — этаж объекта или специальная категория.
- **Количество комнат** — числовое значение комнат.
- **Год / месяц начала объявления** — дата публикации.
- **Город / район** — местоположение объекта.
- **Тип отопления** — вариант отопления из обучающей выборки.

### Что происходит внутри модели
Сначала исходные признаки приводятся к формату, который понимает алгоритм: числовые признаки очищаются и заполняются при необходимости, категориальные признаки кодируются. Для обучения цена преобразуется через `log(1 + цена)`, а после прогноза выполняется обратное преобразование.

Финальная модель — **LightGBM Regressor**. Она была сохранена вместе с preprocessing в одном Pipeline, поэтому приложение использует ту же обработку признаков, что и при обучении.

### Дашборд
На вкладке **Дашборд** можно:
- фильтровать объявления по продаже/аренде, типу недвижимости и городу;
- смотреть распределение цен;
- сравнивать города отдельно для продажи и аренды;
- анализировать популярность типов недвижимости;
- исследовать связь площади и цены;
- смотреть метрики моделей и важность признаков.

### Ограничения
- Модель обучалась на исторических объявлениях и не отражает автоматически текущий рынок.
- Результат является прогнозом, а не гарантированной ценой сделки.
- Диапазон на странице прогноза построен как `прогноз ± MAE` и не является статистическим доверительным интервалом.
- Аналитика дашборда для цен использует объявления с валютой **турецкая лира** и типами **продажа/аренда**.

**Финальная модель:** LightGBM  
**Интерфейс:** Streamlit
""")
