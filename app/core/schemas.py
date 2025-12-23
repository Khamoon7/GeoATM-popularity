from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Coords(BaseModel):
    """
    Географические координаты.
    """
    lat: float = Field(..., ge=-90.0, le=90.0, description="Широта")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Долгота")


class ATMPredictRequest(BaseModel):
    """
    Входная схема /forward.

    Поддерживает два взаимоисключающих способа задания локации:
    - address
    - lat/lon

    Также умеет вычислять atm_group по bank_name, если atm_group не задан явно.
    """

    # Запрещаем лишние поля в запросе
    model_config = ConfigDict(extra="forbid")

    atm_id: Optional[str] = Field(default=None, description="ID банкомата (опционально)")

    address: Optional[str] = Field(
        default=None,
        description="Адрес (если ввод адресом)",
        examples=["Москва, Тверская 1"]
    )
    lat: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Широта (если ввод координатами)",
        examples=[55.7558]
    )
    lon: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Долгота (если ввод координатами)",
        examples=[37.6173]
    )

    bank_name: Optional[str] = Field(default=None, description="Название банка для авто-вычисления atm_group")
    atm_group: Optional[float] = Field(default=None, description="Числовой код группы банка")

    is_24_7: bool = Field(default=False)
    contactless_tech: bool = Field(default=False)
    qr_codes: bool = Field(default=False)
    usd_available: bool = Field(default=False)
    eur_available: bool = Field(default=False)
    cash_in: bool = Field(default=False)
    cash_out: bool = Field(default=False)
    cashless_pay: bool = Field(default=False)
    account_statement: bool = Field(default=False)
    access_for_disabled: bool = Field(default=False)
    transfer_p2p: bool = Field(default=False)
    transfer_a2a: bool = Field(default=False)
    loan_payments: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_location_and_set_group(self) -> "ATMPredictRequest":
        """
        Валидирует способ задания локации и при необходимости заполняет atm_group.

        Правила:
        - address и lat/lon взаимоисключающие,
        - lat/lon должны быть заданы парой,
        - если atm_group не задан, он вычисляется из bank_name (fallback = 32.0).
        """
        has_address = self.address is not None and self.address.strip() != ""
        has_lat = self.lat is not None
        has_lon = self.lon is not None
        has_coords = has_lat and has_lon

        if has_address and has_coords:
            raise ValueError("Укажите либо address, либо lat/lon, но не оба варианта одновременно")
        if not has_address and not has_coords:
            raise ValueError("Необходимо указать address или координаты lat/lon")
        if (has_lat and not has_lon) or (has_lon and not has_lat):
            raise ValueError("Координаты должны быть заданы парой: и lat, и lon")

        # Авто-определение группы банка, если не передана явно
        if self.atm_group is None:
            name = (self.bank_name or "").strip().lower()
            if not name:
                self.atm_group = 32.0
            elif "втб" in name or "уралсиб" in name:
                self.atm_group = 5478.0
            elif "альфа" in name:
                self.atm_group = 1942.0
            elif "росбанк" in name:
                self.atm_group = 8083.0
            elif "россельхоз" in name:
                self.atm_group = 496.5
            elif "газпром" in name:
                self.atm_group = 3185.5
            elif "ак барс" in name or "акбарс" in name:
                self.atm_group = 1022.0
            else:
                self.atm_group = 32.0

        return self


class ATMPredictResponse(BaseModel):
    """
    Выходная схема /forward.
    """

    atm_id: Optional[str] = Field(default=None, description="ID банкомата")
    popularity_index: float = Field(..., description="Индекс популярности")
    segment: Optional[str] = Field(default=None, description="Сегмент")
    coords: Coords = Field(..., description="Финальные координаты")
