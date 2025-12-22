from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeaturesValidator:
    """
    Валидатор входных признаков для ATM ML-модели.

    Назначение:
    - привести произвольный вход (из FeaturesBuilder / тестов / API)
      к СТРОГОМУ контракту модели;
    - удалить лишние признаки;
    - добавить отсутствующие;
    - заполнить пропуски;
    - привести типы;
    - гарантировать корректный порядок колонок.

    Валидатор НЕ:
    - знает про sklearn;
    - знает про pipeline;
    - создаёт новые признаки;
    - содержит бизнес-логику.
    """

    strict_no_nan: bool = True

    def __post_init__(self) -> None:
        # 🔒 ЭТАЛОННЫЙ СПИСОК ПРИЗНАКОВ МОДЕЛИ (КОНТРАКТ)
        self.expected_features: List[str] = [
            "id",
            "atm_group",
            "geo_lon",
            "geo_lat",
            "population_density_per_km2",
            "is_24_7",
            "contactless_tech",
            "qr_codes",
            "usd_available",
            "eur_available",
            "cash_in",
            "cash_out",
            "cashless_pay",
            "account_statement",
            "access_for_disabled",
            "transfer_p2p",
            "transfer_a2a",
            "loan_payments",
            "nearest_malls_dist_m",
            "count_malls_300m",
            "nearest_supermarkets_dist_m",
            "count_supermarkets_300m",
            "nearest_pharmacies_hospitals_dist_m",
            "count_pharmacies_hospitals_300m",
            "count_banks_atms_300m",
            "nearest_cafes_dist_m",
            "count_cafes_300m",
            "nearest_restaurants_dist_m",
            "count_restaurants_300m",
            "nearest_public_transport_dist_m",
            "count_public_transport_300m",
            "nearest_parking_dist_m",
            "count_parking_300m",
            "nearest_education_dist_m",
            "count_education_300m",
            "nearest_subway_dist_m",
            "nearest_post_offices_dist_m",
            "count_post_offices_300m",
            "has_subway_nearby",
        ]

    # -----------------------------------------------------

    def validate(self, raw_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Приводит входной DataFrame к формату модели.

        Parameters
        ----------
        raw_features : pd.DataFrame
            Сырые признаки (1 строка).

        Returns
        -------
        X_valid : pd.DataFrame
            Валидированный DataFrame (1 строка, строгий порядок).
        warnings : list[str]
            Предупреждения о внесённых изменениях.
        """
        if not isinstance(raw_features, pd.DataFrame):
            raise ValueError("raw_features должен быть pandas DataFrame")

        if raw_features.shape[0] == 0:
            raise ValueError("raw_features пуст")

        warnings: List[str] = []
        X = raw_features.copy()

        # 1️⃣ Удаляем лишние признаки
        extra_cols = [c for c in X.columns if c not in self.expected_features]
        if extra_cols:
            X.drop(columns=extra_cols, inplace=True)
            warnings.append(f"Удалены лишние признаки: {extra_cols}")

        # 2️⃣ Добавляем отсутствующие признаки
        missing_cols = [c for c in self.expected_features if c not in X.columns]
        for c in missing_cols:
            X[c] = np.nan
        if missing_cols:
            warnings.append(f"Добавлены отсутствующие признаки: {missing_cols}")

        # 3️⃣ Приведение типов + заполнение пропусков
        for c in self.expected_features:
            if X[c].dtype == bool:
                X[c] = X[c].fillna(False).astype(float)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

        # 4️⃣ ЖЁСТКИЙ ПОРЯДОК КОЛОНОК
        X = X[self.expected_features]

        # 5️⃣ Контроль NaN
        if self.strict_no_nan and X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"После валидации остались NaN в колонках: {nan_cols}")

        # 6️⃣ Финальное приведение типов
        X = X.astype(float)

        return X, warnings
