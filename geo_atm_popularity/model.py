# app/core/model.py

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


class ATMModelService:
    """

    - загружает финальный Pipeline (preprocessor + модель),
    - принимает сырые фичи банкомата в виде dict,
    - возвращает числовой индекс популярности (float).
    """

    def __init__(
        self,
        model_path: Path | str = Path("models") / "atm_popularity_model.pkl",
    ) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model()

    def load_model(self):
        """
        Загружает обученный пайплайн из .pkl.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = joblib.load(self.model_path)
        if not hasattr(model, "predict"):
            raise TypeError(
                f"Loaded object does not have predict(): {type(model)}"
            )
        return model

    def predict_popularity(self, features: Dict[str, Any]) -> float:
        # Оборачиваем в DataFrame с одной строкой
        df = pd.DataFrame([features])
        y_pred = self.model.predict(df)
        return float(y_pred[0])

atm_model_service = ATMModelService()
