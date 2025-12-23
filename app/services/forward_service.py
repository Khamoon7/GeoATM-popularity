from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import pandas as pd

from app.services.location_reader import LocationReader, LocationResult
from app.services.features_builder import FeaturesBuilder
from app.services.features_validator import FeaturesValidator
from app.core.model import ATMModelService

from app.core.logging import get_logger

logger = get_logger("forward.service")

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

        logger.info(
            "RUN start: address=%s lat=%s lon=%s payload_keys=%s",
            address,
            lat,
            lon,
            sorted(payload.keys()),
        )

        loc: LocationResult = self.location_reader.read(address=address, lat=lat, lon=lon)

        logger.info(
            "LocationReader: ok=%s error=%s normalized_address=%s country=%s province=%s locality=%s lat=%s lon=%s",
            getattr(loc, "ok", None),
            getattr(loc, "error", None),
            getattr(loc, "normalized_address", None),
            getattr(loc, "country", None),
            getattr(loc, "province", None),
            getattr(loc, "locality", None),
            getattr(loc, "lat", None),
            getattr(loc, "lon", None),
        )

        if not loc.ok:
            logger.warning("Location invalid: error=%s", loc.error)
            return {
                "ok": False,
                "error": loc.error,
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        if loc.lat is None or loc.lon is None:
            logger.error("Location ok=True but coords missing: lat=%s lon=%s", loc.lat, loc.lon)
            return {
                "ok": False,
                "error": "LocationReader вернул ok=True, но координаты отсутствуют",
                "normalized_address": loc.normalized_address,
                "lat": loc.lat,
                "lon": loc.lon,
            }

        atm_params = self._build_atm_params(payload=payload, loc=loc)
        logger.info(
            "ATM params built: keys=%s city(locality)=%s region=%s atm_group=%s bank_name=%s",
            sorted(atm_params.keys()),
            atm_params.get("locality"),
            atm_params.get("region"),
            atm_params.get("atm_group"),
            atm_params.get("bank_name"),
        )
        features_df = await self.features_builder.build(
            lat=loc.lat,
            lon=loc.lon,
            atm_params=atm_params,
        )
        logger.info(
            "Features built: shape=%s cols=%s",
            features_df.shape,
            list(features_df.columns),
        )
        features_df, fv_warnings  = self.features_validator.validate(features_df)

        cols_after = list(features_df.columns)
        logger.info(
            "After validator: shape=%s n_cols=%d cols=%s",
            features_df.shape,
            len(cols_after),
            cols_after,
        )

        if fv_warnings:
            logger.warning("Validator warnings (%d): %s", len(fv_warnings), fv_warnings)
        else:
            logger.info("Validator warnings: none")

        pred, model_warnings = self.model.predict_popularity(features_df)

        expected = getattr(getattr(self.model, "model", None), "feature_names_in_", None)

        if expected is not None:
            expected = list(expected)
            cols = list(features_df.columns)

            missing = sorted(set(expected) - set(cols))
            extra = sorted(set(cols) - set(expected))

            if missing:
                logger.error("Model missing columns: %s", missing)
            else:
                logger.info("Model missing columns: none")

            if extra:
                logger.warning("Model extra columns: %s", extra)
        else:
            logger.warning("Model has no feature_names_in_")

        warnings = (fv_warnings or []) + (model_warnings or [])

        return {
            "ok": True,
            "normalized_address": loc.normalized_address,
            "lat": float(loc.lat),
            "lon": float(loc.lon),
            "popularity_index": float(pred),
            "warnings": warnings,
        }

    def _build_atm_params(self, payload: Dict[str, Any], loc: LocationResult) -> Dict[str, Any]:
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
            "region": loc.province,
            "area": loc.area,
            "locality": loc.locality,
            "city": loc.locality,  # <-- ДОБАВИЛИ
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