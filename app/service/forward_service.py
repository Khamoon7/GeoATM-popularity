from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# твои компоненты
from app.service.location_reader import LocationReader, LocationResult
from app.service.features_builder import FeaturesBuilder
from app.service.features_validator import FeaturesValidator
from app.core.model import ATMModelService


class ForwardService:
    """
    Пайплайн для forward

    Последовательно:
      1) LocationReader.read() -> LocationResult
      2) FeaturesBuilder.build() -> DataFrame(features)
      3) FeaturesValidator.validate() -> DataFrame(features_clean), warnings
      4) ModelService.predict() -> popularity_index
      5) Формирует единый JSON-ответ
    """

    def __init__(
        self,
        location_reader: LocationReader,
        features_builder: FeaturesBuilder,
        features_validator: FeaturesValidator,
        model: ATMModelService,
    ) -> None:
        self.location_reader = location_reader
        self.features_builder = features_builder
        self.features_validator = features_validator
        self.model = model

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload ожидается в виде словаря из pydantic:
          - либо {"address": "...", ...}
          - либо {"lat": ..., "lon": ..., ...}
          - плюс поля банкомата: operations, bank_type, etc.

        Возвращает JSON-словарь для ответа /forward.
        """
        address = payload.get("address")
        lat = payload.get("lat")
        lon = payload.get("lon")

        # Геокодинг
        loc: LocationResult = self.location_reader.read(address=address, lat=lat, lon=lon)

        if not loc.ok:
            return {
                "ok": False,
                "error": loc.error,
                "warnings": [],
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        if loc.lat is None or loc.lon is None:
            return {
                "ok": False,
                "error": "LocationReader вернул ok=True, но координаты отсутствуют",
                "warnings": [],
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        # Собираем параметры банкомата + результат геокодинга
        atm_params = self._build_atm_params(payload=payload, loc=loc)

        # Генерация признаков
        features_df = await self.features_builder.build(
            lat=loc.lat,
            lon=loc.lon,
            atm_params=atm_params,
        )

        # Валидация/пост-обработка признаков
        ##features_df, warnings = self.features_validator.validate(features_df)

        # Инференс модели
        popularity_index = self.model.predict_popularity(features_df)

        # Единый ответ
        return {
            "ok": True,
            "normalized_address": loc.normalized_address,
            "lat": float(loc.lat),
            "lon": float(loc.lon),
            "popularity_index": float(popularity_index),
            "warnings": warnings,
        }

    def _build_atm_params(self, payload: Dict[str, Any], loc: LocationResult) -> Dict[str, Any]:
        """
        Собирает словарь параметров для FeaturesBuilder:
          - адресные поля из LocationResult
          - исходные поля банкомата из payload
        """
        atm_only: Dict[str, Any] = dict(payload)
        atm_only.pop("address", None)
        atm_only.pop("lat", None)
        atm_only.pop("lon", None)

        loc_dict = self._location_to_dict(loc)

        merged = {
            "input_type": loc.input_type,
            "normalized_address": loc.normalized_address,
            "is_russia": loc.is_russia,
            "country": loc.country,
            "province": loc.province,
            "area": loc.area,
            "locality": loc.locality,
            "street": loc.street,
            "house": loc.house,
        }
        merged.update(atm_only)
        return merged

    @staticmethod
    def _location_to_dict(loc: LocationResult) -> Dict[str, Any]:
        """
        Универсальная конвертация LocationResult в dict.
        """
        if is_dataclass(loc):
            return asdict(loc)
        return dict(loc.__dict__)