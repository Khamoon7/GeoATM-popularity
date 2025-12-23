from pathlib import Path
from typing import Tuple, List, Union
import sys

import joblib
import pandas as pd

from app.services.features_validator import FeaturesValidator


class ATMModelService:
    def __init__(
        self,
        model_path: Union[Path, str, None] = None,
        strict_no_nan: bool = True,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

        default_path = project_root / "models" / "final_atm_pipeline.pkl"

        if model_path is None:
            self.model_path = default_path
        else:
            p = Path(model_path)
            self.model_path = p if p.is_absolute() else (project_root / p)

        self.model = self._load_model()

        self.validator = FeaturesValidator(strict_no_nan=strict_no_nan)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        model = joblib.load(self.model_path)

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Загруженный объект не поддерживает predict(): {type(model)}"
            )

        return model

    def predict_popularity(
        self,
        raw_features: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        X_valid, warnings = self.validator.validate(raw_features)

        y_pred = self.model.predict(X_valid)

        return float(y_pred[0]), warnings
