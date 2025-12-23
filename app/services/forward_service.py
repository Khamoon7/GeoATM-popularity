from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.services.location_reader import LocationReader, LocationResult
from app.services.features_builder import FeaturesBuilder
from app.core.model import ATMModelService


class ForwardService:
    def __init__(
        self,
        location_reader: LocationReader,
        features_builder: FeaturesBuilder,
        model: ATMModelService,
    ) -> None:
        self.location_reader = location_reader
        self.features_builder = features_builder
        self.model = model

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
            raise ValueError(f"bad request: {loc.error}")

        if loc.lat is None or loc.lon is None:
            raise RuntimeError("LocationReader вернул ok=True, но координаты отсутствуют")

        atm_params = self._build_atm_params(payload=payload, loc=loc)

        build_payload = dict(atm_params)
        build_payload["geo_lat"] = float(loc.lat)
        build_payload["geo_lon"] = float(loc.lon)

        try:
            features_df = await self.features_builder.build(payload=build_payload)
        except ValueError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            # падение overpass/сети/любая сборка фичей — это 403
            raise RuntimeError(f"FeaturesBuilder failed: {type(e).__name__}: {e}") from e

        try:
            pred, model_warnings = self.model.predict_popularity(features_df)
        except ValueError as e:
            raise RuntimeError(f"Model validation/predict failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Model predict failed: {type(e).__name__}: {e}") from e

        warnings = model_warnings or []

        return {
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

        merged: Dict[str, Any] = {
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