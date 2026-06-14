from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================================
# Контракт признаков модели LightGBM (GeoATM-LightGBM-PRD / lightgbm_best.txt)
#
# Порядок и состав строго соответствуют обучению (notebooks/LightGBM_mlflow.ipynb)
# и проверены против lgb.Booster('models/lightgbm_best.txt').feature_name().
# =========================================================================

BINARY_FEATURES: List[str] = [
    "is_24_7", "contactless_tech", "qr_codes", "usd_available", "eur_available",
    "cash_in", "cash_out", "cashless_pay", "account_statement", "access_for_disabled",
    "transfer_p2p", "transfer_a2a", "loan_payments",
    "is_federal_city", "is_federal_district_capital", "is_city_center", "has_subway_nearby",
]

DISTANCE_FEATURES: List[str] = [
    "nearest_malls_dist_m", "nearest_supermarkets_dist_m",
    "nearest_pharmacies_hospitals_dist_m", "nearest_cafes_dist_m",
    "nearest_restaurants_dist_m", "nearest_public_transport_dist_m",
    "nearest_parking_dist_m", "nearest_education_dist_m", "nearest_subway_dist_m",
    "nearest_post_offices_dist_m", "nearest_offices_dist_m",
    "nearest_shops_food_small_dist_m", "nearest_fitness_sport_dist_m",
    "nearest_hotels_hostels_dist_m", "land_use_dist_m", "city_center_dist_m",
    "nearest_residential_landuse_dist_m",
]

COUNT_FEATURES: List[str] = [
    "count_malls_300m", "count_supermarkets_300m", "count_pharmacies_hospitals_300m",
    "count_banks_atms_300m", "count_cafes_300m", "count_restaurants_300m",
    "count_public_transport_300m", "count_parking_300m", "count_education_300m",
    "count_post_offices_300m", "count_offices_300m", "count_payment_terminals_300m",
    "count_money_transfer_300m", "count_shops_food_small_300m", "count_hypermarkets_300m",
    "count_markets_300m", "count_fitness_sport_300m", "count_hotels_hostels_300m",
    "count_railway_stations_300m", "count_residential_buildings_300m",
    "count_residential_landuse_300m", "count_fuel_300m",
    "count_highway_pedestrian_300m", "count_landuse_mix_300m", "count_footway_100m_100m",
]

ECONOMIC_FEATURES: List[str] = [
    "avg_salary_oct_2025_rub", "grp_per_capita_2023_rub",
    "avg_income_q3_2025_rub", "population_density_per_km2",
]

CATEGORICAL_FEATURES: List[str] = [
    "federal_district", "city_size_category", "land_use_type", "atm_group",
]

ALL_FEATURES: List[str] = (
    BINARY_FEATURES + DISTANCE_FEATURES + COUNT_FEATURES
    + ECONOMIC_FEATURES + CATEGORICAL_FEATURES
)

# Числовые признаки, к которым в обучении применялся SimpleImputer(strategy='median').
# Медианы считались в пространстве ПОСЛЕ log1p для расстояний (см. ноутбук).
NUMERIC_FEATURES: List[str] = (
    BINARY_FEATURES + DISTANCE_FEATURES + COUNT_FEATURES + ECONOMIC_FEATURES
)

DEFAULT_MEDIANS_PATH = PROJECT_ROOT / "models" / "feature_medians.json"


def load_medians(path: Optional[Path] = None) -> Dict[str, float]:
    """
    Загружает медианы числовых признаков, посчитанные на train-сплите обучения.

    Используются для импьютинга пропусков ровно так же, как при обучении модели.
    Возвращает пустой словарь, если файл отсутствует (тогда импьютинг будет пропущен).
    """
    p = path or DEFAULT_MEDIANS_PATH
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return {}


def preprocess(df: pd.DataFrame, medians: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Приводит сырые признаки к виду, который ожидает LightGBM-модель.

    Повторяет препроцессинг из обучения (notebooks/LightGBM_mlflow.ipynb):
    1. недостающие колонки добавляются как NaN;
    2. бинарные bool -> int;
    3. расстояния -> log1p(clip>=0);
    4. числовые пропуски -> медиана (как SimpleImputer на train);
    5. категориальные -> fillna('unknown').astype('category');
    6. финальный порядок колонок = ALL_FEATURES.
    """
    medians = medians if medians is not None else {}

    X = df.copy()

    # 1. Добиваем недостающие колонки
    for col in ALL_FEATURES:
        if col not in X.columns:
            X[col] = np.nan

    # 2. Бинарные bool -> int (числовые/строковые приводим к числу)
    for col in BINARY_FEATURES:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 3. Расстояния -> log1p (с обрезкой отрицательных)
    for col in DISTANCE_FEATURES:
        vals = pd.to_numeric(X[col], errors="coerce")
        X[col] = np.log1p(vals.clip(lower=0))

    # COUNT / ECONOMIC -> числовой тип (без трансформации)
    for col in COUNT_FEATURES + ECONOMIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 4. Импьютинг медианой (как при обучении). Медианы расстояний — в log1p-пространстве.
    for col in NUMERIC_FEATURES:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
        else:
            # fallback: оставляем NaN — LightGBM умеет с ними работать нативно
            pass
        X[col] = X[col].astype(float)

    # 5. Категориальные -> 'unknown' + category
    # Значения категорий (включая числовой atm_group) сохраняются как есть —
    # маппинг кодов берётся из pandas_categorical, зашитого в booster при обучении.
    for col in CATEGORICAL_FEATURES:
        s = X[col]
        mask_empty = s.astype("string").str.strip().eq("").fillna(False) | s.isna()
        X[col] = s.mask(mask_empty, "unknown").astype("category")

    # 6. Строгий порядок колонок
    return X[ALL_FEATURES]
