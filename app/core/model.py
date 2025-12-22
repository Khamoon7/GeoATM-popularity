from pathlib import Path
from typing import Tuple, List, Union

import joblib
import pandas as pd

from .features_validator import FeaturesValidator


class ATMModelService:
    """
    Сервис работы с ML-моделью популярности банкоматов.

    Ответственность:
    - загрузка sklearn Pipeline (preprocessor + модель);
    - валидация и очистка входных признаков;
    - инференс модели.

    ВАЖНО:
    - модель ожидает DataFrame с признаками;
    - FeaturesValidator НЕ меняет набор колонок,
      а только чистит и заполняет значения.
    """

    def __init__(
        self,
        model_path: Union[Path, str] = "final_atm_pipeline.pkl",
        strict_no_nan: bool = True,
    ) -> None:
        # путь к pickle с pipeline
        self.model_path = Path(model_path)

        # загружаем sklearn pipeline
        self.model = self._load_model()

        # инициализируем валидатор
        self.validator = FeaturesValidator(strict_no_nan=strict_no_nan)

    def _load_model(self):
        """
        Загружает обученный sklearn Pipeline (preprocessor + модель).
        """
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
        """
        Выполняет инференс модели.

        Parameters
        ----------
        raw_features : pd.DataFrame
            DataFrame из одной строки после FeaturesBuilder.

        Returns
        -------
        prediction : float
            Индекс популярности банкомата.
        warnings : List[str]
            Предупреждения, полученные на этапе валидации.
        """

        # 1️⃣ Валидация и очистка признаков
        X_valid, warnings = self.validator.validate(raw_features)

        # 2️⃣ Инференс через sklearn pipeline
        y_pred = self.model.predict(X_valid)

        return float(y_pred[0]), warnings
