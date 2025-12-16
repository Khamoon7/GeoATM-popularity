# features_validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib


@dataclass
class FeaturesValidator:
    """
    Класс для проверки и приведения признаков к формату ML-модели.

    Назначение:
        Обеспечивает полное соответствие входных признаков формату,
        использованному при обучении модели.

    Зона ответственности:
        - удаление лишних признаков;
        - добавление отсутствующих признаков;
        - приведение типов;
        - заполнение пропусков;
        - контроль качества результата.

    ВАЖНО:
        Модуль НЕ занимается бизнес-логикой (география, адреса, Россия/не Россия)
        и НЕ выполняет инференс модели.
    """

    # Путь к файлу со списком признаков модели (в правильном порядке)
    features_path: str

    # Путь к файлу со статистиками признаков (медианы и т.п.)
    stats_path: str

    # Если True — выбрасывает ошибку, если после валидации остались NaN
    strict_no_nan: bool = True

    def __post_init__(self) -> None:
        """
        Загружает артефакты модели:
        - список ожидаемых признаков;
        - медианные значения признаков.
        """
        self.expected_features: List[str] = self._load_expected_features(self.features_path)
        self.feature_medians: Dict[str, float] = self._load_feature_medians(self.stats_path)

        if not self.expected_features or not isinstance(self.expected_features, list):
            raise ValueError("Ожидался список фичей, но его нет или файл некорректный.")

        if len(set(self.expected_features)) != len(self.expected_features):
            raise ValueError("Список фичек содержит дубликаты.")

    # загрузка атрибутов

    @staticmethod
    def _load_expected_features(path: str) -> List[str]:
        """
        Загружает список признаков модели из .pkl файла.

        Допустимые форматы:
        - list[str]
        - pandas.Index
        """
        obj = joblib.load(path)

        if isinstance(obj, pd.Index):
            return obj.tolist()

        if isinstance(obj, (list, tuple)):
            return list(obj)

        raise ValueError(
            f"model_features.pkl должен содердать List[str] или pandas.Index, получено {type(obj)}"
        )

    @staticmethod
    def _load_feature_medians(path: str) -> Dict[str, float]:
        """
        Загружает медианы признаков из .pkl файла.

        Поддерживаемые форматы:
        - {"feature": 0.5}
        - {"feature": {"median": 0.5}}
        """
        obj = joblib.load(path)

        if not isinstance(obj, dict):
            raise ValueError(f"feature_stats.pkl должен содерать dict, получено {type(obj)}")

        medians: Dict[str, float] = {}

        for k, v in obj.items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                medians[str(k)] = float(v)
            elif (
                isinstance(v, dict)
                and "median" in v
                and isinstance(v["median"], (int, float, np.number))
                and np.isfinite(v["median"])
            ):
                medians[str(k)] = float(v["median"])

        return medians

    #  Вспомогательные методы 

    @staticmethod
    def _is_boolean_like(series: pd.Series) -> bool:
        """
        Определяет, можно ли считать признак булевым по данным.

        Булевой признак:
        - dtype == bool
        - либо все непустые значения принадлежат {0, 1}
        """
        if series.dtype == bool:
            return True

        s = series.dropna()
        if s.empty:
            return False

        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if s_num.empty:
            return False

        return set(pd.unique(s_num)).issubset({0, 1})

    # ---------- основной метод ----------

    def validate(self, raw_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Проверяет и приводит входные признаки к формату модели.

        Parameters
        ----------
        raw_features : pd.DataFrame
            Сырые признаки после FeaturesBuilder.

        Returns
        -------
        X_valid : pd.DataFrame
            Валидированный набор признаков (1 строка, правильный порядок).
        warnings : List[str]
            Предупреждения о внесённых изменениях.
        """
        if raw_features is None or not isinstance(raw_features, pd.DataFrame):
            raise ValueError("raw_features должен быть DataFrame.")

        if raw_features.shape[0] == 0:
            raise ValueError("raw_features пуст (0 строк).")

        warnings: List[str] = []
        X = raw_features.copy()

        # 1. Удаляем лишние признаки
        extra_cols = [c for c in X.columns if c not in self.expected_features]
        if extra_cols:
            X = X.drop(columns=extra_cols, errors="ignore")
            warnings.append(f"Удалены следующие колонки: {extra_cols}")

        # 2. Добавляем отсутствующие признаки
        missing_cols = [c for c in self.expected_features if c not in X.columns]
        if missing_cols:
            for c in missing_cols:
                X[c] = np.nan
            warnings.append(f"Добавлены необходимые колонки с N/A значениями: {missing_cols}")

        # 3. Приводим признаки к числовому типу
        for c in self.expected_features:
            before_na = X[c].isna().sum()

            if X[c].dtype != bool:
                X[c] = pd.to_numeric(X[c], errors="coerce")

            after_na = X[c].isna().sum()
            if after_na > before_na:
                warnings.append(
                    f"Столбец '{c}': при приведении к числовому типу появилось {after_na - before_na} новых пропусков (NaN)."
                )

        # 4. Заполняем пропуски
        for c in self.expected_features:
            if not X[c].isna().any():
                continue

            if self._is_boolean_like(X[c]):
                fill_val = self.feature_medians.get(c, 0.0)
                X[c] = X[c].fillna(fill_val)
                warnings.append(
                    f"Заполнены пропуски в признаках с булевыми значениями '{c}' медианой={fill_val}."
                )
            else:
                X[c] = X[c].fillna(0.0)
                warnings.append(f"Заполнены пропуски в '{c}'нулями.")

        # 5. Приводим порядок колонок
        X = X[self.expected_features]

        # 6. Финальный контроль NaN
        if self.strict_no_nan:
            nan_cols = [c for c in self.expected_features if X[c].isna().any()]
            if nan_cols:
                nan_counts = {c: int(X[c].isna().sum()) for c in nan_cols}
                raise ValueError(
                    f"Ошибка валидации - остались пропуски после заполнения: {nan_counts}"
                )

        # 7. Приведение типов
        X = X.astype(float)

        return X, warnings
