from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from app.core.preprocess import load_medians, preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Источник модели по умолчанию — локальный LightGBM booster.
# Можно переопределить через переменную окружения MODEL_URI:
#   - локальный файл:  models/lightgbm_best.txt  (или абсолютный путь)
#   - MLflow Registry: models:/GeoATM-LightGBM-PRD/1  (или runs:/<run_id>/model)
DEFAULT_MODEL_URI = "models/lightgbm_best.txt"


class ATMModelService:
    """
    Сервис работы с ML-моделью популярности банкоматов (LightGBM Optuna, v2).

    Загружает модель из локального booster-файла или из MLflow Model Registry,
    применяет препроцессинг (как при обучении) и выполняет инференс.
    """

    def __init__(self, model_uri: Union[str, None] = None) -> None:
        """
        Инициализирует сервис модели.

        Источник модели берётся из аргумента, иначе из MODEL_URI,
        иначе — локальный файл models/lightgbm_best.txt.
        """
        self.model_uri = model_uri or os.getenv("MODEL_URI") or DEFAULT_MODEL_URI
        self.medians = load_medians()
        self.model, self.backend = self._load_model(self.model_uri)

    def _load_model(self, uri: str):
        """
        Загружает модель: MLflow pyfunc (models:/, runs:/) или локальный LightGBM booster.
        """
        if uri.startswith("models:/") or uri.startswith("runs:/"):
            return self._load_from_mlflow(uri), "mlflow"
        return self._load_local_booster(uri), "lightgbm"

    def _load_from_mlflow(self, uri: str):
        """
        Загружает модель из MLflow Model Registry через mlflow.pyfunc.

        Требует поднятый MLflow + MinIO (docker-compose) и переменные окружения
        MLFLOW_TRACKING_URI / AWS_* / MLFLOW_S3_ENDPOINT_URL.
        """
        import mlflow  # импорт здесь, чтобы не тянуть mlflow при локальном пути

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        return mlflow.pyfunc.load_model(uri)

    def _load_local_booster(self, uri: str):
        """
        Загружает локальный LightGBM booster из текстового файла.
        """
        import lightgbm as lgb

        p = Path(uri)
        path = p if p.is_absolute() else (PROJECT_ROOT / p)
        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        return lgb.Booster(model_file=str(path))

    def predict_popularity(self, raw_features: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Применяет препроцессинг и инференс модели.

        Возвращает предсказание и список предупреждений.
        """
        if not isinstance(raw_features, pd.DataFrame) or raw_features.shape[0] == 0:
            raise ValueError("raw_features должен быть непустым pandas DataFrame")

        warnings: List[str] = []
        X = preprocess(raw_features, self.medians)

        y_pred = self.model.predict(X)

        # mlflow.pyfunc возвращает numpy/Series/DataFrame — нормализуем к float
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        return float(y_pred[0]), warnings
