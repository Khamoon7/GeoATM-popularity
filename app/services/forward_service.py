from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import pandas as pd

from app.services.location_reader import LocationReader, LocationResult
from app.services.features_builder import FeaturesBuilder
from app.services.features_validator import FeaturesValidator
from app.core.model import ATMModelService


class ForwardService:
    def __init__(
        self,
        location_reader: LocationReader,
        features_builder: FeaturesBuilder,
        model: ATMModelService,
        features_validator: FeaturesValidator,
    ) -> None:
        self.location_reader = location_reader
        self.features_builder = features_builder
        self.model = model
        self.features_validator = features_validator

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        address = payload.get("address")
        lat = payload.get("lat")
        lon = payload.get("lon")

        loc: LocationResult = self.location_reader.read(
            address=address,
            lat=lat,
            lon=lon,
        )

        if not loc.ok:
            return {
                "ok": False,
                "error": loc.error,
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        if loc.lat is None or loc.lon is None:
            return {
                "ok": False,
                "error": "LocationReader вернул ok=True, но координаты отсутствуют",
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        atm_params = self._build_atm_params(payload=payload, loc=loc)

        features_df = await self.features_builder.build(
            lat=loc.lat,
            lon=loc.lon,
            atm_params=atm_params,
        )

        features_df, fv_warnings = self.features_validator.validate(features_df)

        pred, model_warnings = self.model.predict_popularity(features_df)

        expected = getattr(
            getattr(self.model, "model", None),
            "feature_names_in_",
            None,
        )

        warnings = (fv_warnings or []) + (model_warnings or [])

        return {
            "ok": True,
            "normalized_address": loc.normalized_address,
            "lat": float(loc.lat),
            "lon": float(loc.lon),
            "popularity_index": float(pred),
            "warnings": warnings,
        }

    def _build_atm_params(
        self,
        payload: Dict[str, Any],
        loc: LocationResult,
    ) -> Dict[str, Any]:
        atm_only: Dict[str, Any] = dict(payload)
        atm_only.pop("address", None)
        atm_only.pop("lat", None)
        atm_only.pop("lon", None)

        merged = {
            "input_type": loc.input_type,
            "normalized_address": loc.normalized_address,
            "is_russia": loc.is_russia,
            "country": loc.country,
            "province": loc.province,
            "region": loc.province,
            "area": loc.area,
            "locality": loc.locality,
            "city": loc.locality,
            "street": loc.street,
            "house": loc.house,
        }
        merged.update(atm_only)
        return merged

    @staticmethod
    def _location_to_dict(loc: LocationResult) -> Dict[str, Any]:
        if is_dataclass(loc):
            return asdict(loc)
        return dict(loc.__dict__)
