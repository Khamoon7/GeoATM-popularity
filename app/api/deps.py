from __future__ import annotations

from functools import lru_cache

from app.services.location_reader import LocationReader
from app.services.features_builder import FeaturesBuilder
from app.core.model import ATMModelService
from app.services.forward_service import ForwardService
from app.services.features_validator import FeaturesValidator

@lru_cache
def get_forward_service() -> ForwardService:
    return ForwardService(
        location_reader=LocationReader(),
        features_builder=FeaturesBuilder(),
        model=ATMModelService(),
        features_validator=FeaturesValidator(),
    )