import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _load_bin_features_from_env() -> list[str]:
    """
    Загружает список бинарных признаков из переменной окружения.

    Используется как fallback, если признаки не переданы явно.
    """
    raw = os.getenv("ATM_BIN_FEATURES", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


class RegionGrouper(BaseEstimator, TransformerMixin):
    """
    Группирует редкие значения региона в категорию __OTHER__.

    Используется для снижения размерности категориального признака region.
    """

    def __init__(self, rare_threshold: float = 0.01):
        self.rare_threshold = rare_threshold
        self.rare_regions_ = None

    def fit(self, X, y=None):
        """
        Определяет редкие регионы по частотному порогу.
        """
        df = pd.DataFrame(X)
        self.rare_regions_ = set()

        if "region" in df.columns and len(df) > 0:
            cnt = (
                df["region"]
                .astype("string")
                .fillna("__NA__")
                .value_counts(dropna=False)
            )
            thr = max(1, int(np.ceil(self.rare_threshold * len(df))))
            self.rare_regions_ = set(cnt[cnt < thr].index)

        return self

    def transform(self, X):
        """
        Заменяет редкие регионы на __OTHER__ и удаляет исходную колонку region.
        """
        df = pd.DataFrame(X).copy()

        if "region" in df.columns:
            s = df["region"].astype("string").fillna("__NA__")
            df["region_grouped"] = s.where(~s.isin(self.rare_regions_), "__OTHER__")
            df.drop(columns=["region"], inplace=True, errors="ignore")

        return df


class BinaryToFloat(BaseEstimator, TransformerMixin):
    """
    Преобразует бинарные признаки в float.

    Список признаков может задаваться явно или через переменную окружения.
    """

    def __init__(self, bin_features: Optional[list[str]] = None):
        self.bin_features = bin_features

    def _resolve_bin_features(self) -> list[str]:
        """
        Определяет итоговый список бинарных признаков.
        """
        feats = getattr(self, "bin_features", None)

        if not feats:
            feats = _load_bin_features_from_env()

        return list(feats)

    def fit(self, X, y=None):
        """
        Фиксирует список бинарных признаков.
        """
        self.bin_features = self._resolve_bin_features()
        return self

    def transform(self, X):
        """
        Приводит бинарные признаки к float с заполнением пропусков нулями.
        """
        df = pd.DataFrame(X).copy()

        bin_features = self._resolve_bin_features()

        for col in bin_features:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0)
                    .astype(float)
                )

        return df


class ATMLocationFeatures(BaseEstimator, TransformerMixin):
    """
    Генерирует дополнительные фичи из геопризнаков.

    Добавляет логарифмы расстояний и агрегированную плотность POI.
    """

    def fit(self, X, y=None):
        """
        Fit без состояния (совместим с sklearn Pipeline).
        """
        return self

    def transform(self, X):
        """
        Строит производные геофичи:
        - log1p для расстояний,
        - суммарную плотность POI в радиусе 300м.
        """
        df = pd.DataFrame(X).copy()

        # Логарифмы расстояний
        for c in [c for c in df.columns if "dist_m" in c]:
            df[f"{c}_log"] = np.log1p(
                pd.to_numeric(df[c], errors="coerce").fillna(0)
            )

        # Агрегированная плотность POI
        count_cols = [c for c in df.columns if c.startswith("count_")]
        if count_cols:
            df["poi_density_300m"] = (
                df[count_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum(axis=1)
            )
        else:
            df["poi_density_300m"] = 0.0

        return df
