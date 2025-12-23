from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import requests


@dataclass(frozen=True)
class LocationResult:
    ok: bool
    input_type: str
    normalized_address: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    is_russia: Optional[bool]
    country: Optional[str]
    province: Optional[str]
    area: Optional[str]
    locality: Optional[str]
    street: Optional[str]
    house: Optional[str]
    raw: Optional[dict]
    error: Optional[str]


class LocationReader:
    YANDEX_GEOCODER_URL = "https://geocode-maps.yandex.ru/v1/"
    DEFAULT_TIMEOUT_SEC = 10

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        lang: str = "ru_RU",
        results: int = 1,
    ) -> None:
        self.api_key = api_key or os.getenv("YANDEX_GEOCODER_API_KEY")
        self.timeout_sec = timeout_sec
        self.lang = lang
        self.results = results

    def read(
        self,
        *,
        address: Optional[str] = None,
        lat: Optional[Union[float, str]] = None,
        lon: Optional[Union[float, str]] = None,
    ) -> LocationResult:
        input_type = self._detect_input_type(address=address, lat=lat, lon=lon)

        if not self.api_key:
            raise RuntimeError("Не задан YANDEX_GEOCODER_API_KEY")

        if input_type == "address":
            address_clean = self._validate_address(address)
            geo = self._geocode_address(address_clean)
            return self._to_result(input_type="address", geo=geo)

        # input_type == "coords"
        lat_f, lon_f = self._validate_coords(lat, lon)
        geo = self._reverse_geocode(lat_f, lon_f)
        return self._to_result(input_type="coords", geo=geo)

    def _detect_input_type(
        self,
        *,
        address: Optional[str],
        lat: Optional[Union[float, str]],
        lon: Optional[Union[float, str]],
    ) -> str:
        has_address = address is not None and str(address).strip() != ""

        lat_provided = lat is not None and str(lat).strip() != ""
        lon_provided = lon is not None and str(lon).strip() != ""
        has_coords = lat_provided and lon_provided

        if has_address and has_coords:
            raise ValueError("Укажите либо адрес, либо координаты (lat, lon), но не оба варианта одновременно")
        if not has_address and not has_coords:
            raise ValueError("Необходимо указать адрес или координаты (lat, lon)")
        if (lat_provided and not lon_provided) or (lon_provided and not lat_provided):
            raise ValueError("Если указываете координаты, нужно передать и lat, и lon")

        return "address" if has_address else "coords"

    def _validate_address(self, address: Optional[str]) -> str:
        if address is None or str(address).strip() == "":
            raise ValueError("Адрес пустой")
        addr = str(address).strip()
        addr = re.sub(r"\s+", " ", addr)
        if len(addr) < 5:
            raise ValueError("Адрес слишком короткий")
        return addr

    def _validate_coords(
        self,
        lat: Optional[Union[float, str]],
        lon: Optional[Union[float, str]],
    ) -> Tuple[float, float]:
        if lat is None or lon is None:
            raise ValueError("Необходимо указать координаты lat и lon")

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception as e:
            raise ValueError("Координаты lat и lon должны быть числовыми") from e

        if not (-90.0 <= lat_f <= 90.0):
            raise ValueError("Широта должна быть в диапазоне [-90, 90]")
        if not (-180.0 <= lon_f <= 180.0):
            raise ValueError("Долгота должна быть в диапазоне [-180, 180]")

        return lat_f, lon_f


    def _geocode_address(self, address: str) -> dict:
        query = address
        if "россия" not in address.lower():
            query = f"{address}, Россия"

        params = {
            "apikey": self.api_key,
            "geocode": query,
            "format": "json",
            "lang": self.lang,
            "results": self.results,
        }
        data = self._request(params)
        return self._extract_geo(data)

    def _reverse_geocode(self, lat: float, lon: float) -> dict:

        params = {
            "apikey": self.api_key,
            "geocode": f"{lon},{lat}",
            "format": "json",
            "lang": self.lang,
            "results": self.results,
        }
        data = self._request(params)
        return self._extract_geo(data)

    def _request(self, params: Dict[str, Any]) -> dict:
        try:
            r = requests.get(self.YANDEX_GEOCODER_URL, params=params, timeout=self.timeout_sec)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Yandex Geocoder request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Yandex Geocoder unexpected error: {e}") from e

    def _extract_geo(self, data: dict) -> dict:
        collection = data.get("response", {}).get("GeoObjectCollection", {})
        members = collection.get("featureMember", [])
        if not members:
            raise ValueError("Геокодер не вернул результатов")

        geo_obj = members[0].get("GeoObject", {})
        meta = geo_obj.get("metaDataProperty", {}).get("GeocoderMetaData", {})
        text = meta.get("text")

        pos = geo_obj.get("Point", {}).get("pos")
        if not pos:
            raise RuntimeError("Результат геокодирования не содержит координат")

        try:
            lon_s, lat_s = pos.split()
            lon = float(lon_s)
            lat = float(lat_s)
        except Exception as e:
            raise RuntimeError(f"Некорректный формат координат в ответе геокодера: {e}") from e

        comps = meta.get("Address", {}).get("Components", []) or []

        return {
            "text": text,
            "lat": lat,
            "lon": lon,
            "components": comps,
            "raw": data,
        }

    def _to_result(self, *, input_type: str, geo: dict) -> LocationResult:
        comp_map = self._components_to_map(geo.get("components", []))

        country = comp_map.get("country")
        province = comp_map.get("province")
        area = comp_map.get("area")
        locality = comp_map.get("locality")
        street = comp_map.get("street")
        house = comp_map.get("house")

        is_ru = self._is_russia(country=country)

        return LocationResult(
            ok=True,
            input_type=input_type,
            normalized_address=geo.get("text"),
            lat=geo.get("lat"),
            lon=geo.get("lon"),
            is_russia=is_ru,
            country=country,
            province=province,
            area=area,
            locality=locality,
            street=street,
            house=house,
            raw=geo.get("raw"),
            error=None,
        )

    def _components_to_map(self, components: list) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for item in components:
            kind = item.get("kind")
            name = item.get("name")
            if kind and name and kind not in out:
                out[kind] = name
        return out

    def _is_russia(self, *, country: Optional[str]) -> bool:
        if not country:
            return False
        return country == "Россия"
