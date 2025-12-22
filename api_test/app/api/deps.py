from __future__ import annotations

from functools import lru_cache

from app.service.location_reader import LocationReader
from app.service.features_builder import FeaturesBuilder
from app.core.model import ATMModelService
from app.service.forward_service import ForwardService


@lru_cache
def get_forward_service() -> ForwardService:
    return ForwardService(
        location_reader=LocationReader(),
        features_builder=FeaturesBuilder(),
        model=ATMModelService(),
    )