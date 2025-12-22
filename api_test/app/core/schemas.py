from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Coords(BaseModel):
    """
    Координаты точки.
    """
    lat: float = Field(..., ge=-90.0, le=90.0, description="Широта")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Долгота")


class ATMPredictRequest(BaseModel):
    """
    ## ATMPredictRequest

    Схема входа для `/forward` (валидация инпута).

    ### Входные данные
    Пользователь должен передать:
    - либо `address`
    - либо `lat` и `lon`

    А также фичи банкомата (bool) и `atm_group`.

    ### atm_group
    Можно передать напрямую (`atm_group`) **или** передать `bank_name`,
    тогда `atm_group` будет вычислен по правилу:
    - 5478  → ВТБ (Уралсиб)
    - 1942  → Альфа Банк
    - 8083  → Росбанк
    - 496.5 → Россельхозбанк
    - 3185.5→ Газпромбанк
    - 1022  → Ак Барс Банк
    - 32    → Прочее
    """

    atm_id: Optional[str] = Field(default=None, description="ID банкомата (опционально)")

    # Локация: либо address, либо coords
    address: Optional[str] = Field(default=None, description="Адрес (если ввод адресом)")
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Широта (если ввод координатами)")
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Долгота (если ввод координатами)")

    # Для вычисления atm_group (если atm_group не пришёл)
    bank_name: Optional[str] = Field(default=None, description="Название банка для авто-вычисления atm_group")
    atm_group: Optional[float] = Field(default=None, description="Числовой код группы банка")

    # Булевы фичи банкомата
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
        ## _validate_location_and_set_group

        ### Проверки
        - Нельзя одновременно передать `address` и `lat/lon`
        - Нужно передать либо `address`, либо оба `lat` и `lon`

        ### atm_group
        - Если `atm_group` не передан, пытаемся вычислить из `bank_name`
        - Если `bank_name` не задан или не распознан — ставим `32`
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

        # atm_group правило
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
    ## ATMPredictResponse

    Ответ `/forward`.
    """
    atm_id: Optional[str] = Field(default=None, description="ID банкомата")
    popularity_index: float = Field(..., description="Индекс популярности")
    segment: Optional[str] = Field(default=None, description="Сегмент")
    coords: Coords = Field(..., description="Финальные координаты")