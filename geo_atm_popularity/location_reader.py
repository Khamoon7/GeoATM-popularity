# Импорт необходимых библиотек
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import requests


@dataclass(frozen=True)
class LocationResult:
    """
    Нормализованный результат обработки пользовательского ввода
    (адреса или географических координат).

    Используется как единый формат передачи данных между слоями сервиса:
    от модуля геокодинга к модулю генерации признаков и инференса модели.
    """
    ok: bool
    input_type: str  # "address" | "coords"
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
    """
    Компонент сервиса, отвечающий за приём и нормализацию геолокационных данных.

    Класс принимает адрес или географические координаты пользователя,
    выполняет валидацию входных данных, прямое или обратное геокодирование
    с использованием Yandex Geocoder и возвращает результат в виде
    нормализованного объекта LocationResult.
    """

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

    # Публичный API
    def read(
        self,
        *,
        address: Optional[str] = None,
        lat: Optional[Union[float, str]] = None,
        lon: Optional[Union[float, str]] = None,
    ) -> LocationResult:
        """
        Публичная точка входа для обработки пользовательского ввода.

        Принимает либо текстовый адрес, либо пару координат (lat, lon)
        и возвращает нормализованный результат в формате LocationResult.
        """
        try:
            input_type = self._detect_input_type(address=address, lat=lat, lon=lon)
        except ValueError as e:
            return LocationResult(
                ok=False,
                input_type="unknown",
                normalized_address=None,
                lat=None,
                lon=None,
                is_russia=None,
                country=None,
                province=None,
                area=None,
                locality=None,
                street=None,
                house=None,
                raw=None,
                error=str(e),
            )

        if not self.api_key:
            return LocationResult(
                ok=False,
                input_type=input_type,
                normalized_address=None,
                lat=None,
                lon=None,
                is_russia=None,
                country=None,
                province=None,
                area=None,
                locality=None,
                street=None,
                house=None,
                raw=None,
                error="Не задан YANDEX_GEOCODER_API_KEY",
            )

        try:
            if input_type == "address":
                address_clean = self._validate_address(address)
                geo = self._geocode_address(address_clean)
                return self._to_result(input_type="address", geo=geo)

            # input_type == "coords"
            lat_f, lon_f = self._validate_coords(lat, lon)
            geo = self._reverse_geocode(lat_f, lon_f)
            return self._to_result(input_type="coords", geo=geo)

        except Exception as e:
            return LocationResult(
                ok=False,
                input_type=input_type,
                normalized_address=None,
                lat=float(lat) if self._can_float(lat) else None,
                lon=float(lon) if self._can_float(lon) else None,
                is_russia=None,
                country=None,
                province=None,
                area=None,
                locality=None,
                street=None,
                house=None,
                raw=None,
                error=f"LocationReader failed: {e}",
            )

    # Валидация
    def _detect_input_type(
        self,
        *,
        address: Optional[str],
        lat: Optional[Union[float, str]],
        lon: Optional[Union[float, str]],
    ) -> str:
        """
        Определяет тип пользовательского ввода и проверяет его корректность.

        Пользователь должен передать либо текстовый адрес, либо пару
        координат (lat, lon). Одновременная передача обоих вариантов
        или отсутствие входных данных считается ошибкой.
        """
        has_address = address is not None and str(address).strip() != ""

        lat_ok = bool(lat and str(lat).strip())
        lon_ok = bool(lon and str(lon).strip())
        has_coords = lat_ok and lon_ok

        if has_address and has_coords:
            raise ValueError("Укажите либо адрес, либо координаты (lat, lon), но не оба варианта одновременно")
        if not has_address and not has_coords:
            raise ValueError("Необходимо указать адрес или координаты (lat, lon)")
        return "address" if has_address else "coords"

    def _validate_address(self, address: Optional[str]) -> str:
        """
        Проверяет корректность текстового адреса и выполняет его базовую очистку.

        Удаляет лишние пробелы, проверяет, что адрес не пустой и имеет
        минимально допустимую длину для передачи в геокодер.
        """
        if address is None or str(address).strip() == "":
            raise ValueError("Адрес пустой")
        addr = str(address).strip()
        # мягкая чистка, чтобы не ломать адреса
        addr = re.sub(r"\s+", " ", addr)
        if len(addr) < 5:
            raise ValueError("Адрес слишком короткий")
        return addr

    def _validate_coords(
        self,
        lat: Optional[Union[float, str]],
        lon: Optional[Union[float, str]],
    ) -> Tuple[float, float]:
        """
        Проверяет корректность географических координат.

        Приводит широту и долготу к типу float и проверяет,
        что значения находятся в допустимых диапазонах:
        - широта: от -90 до 90
        - долгота: от -180 до 180
        """
        if lat is None or lon is None:
            raise ValueError("Необходимо указать координаты lat и lon")

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            raise ValueError("Координаты lat и lon должны быть числовыми")

        if not (-90.0 <= lat_f <= 90.0):
            raise ValueError("Широта должна быть в диапазоне [-90, 90]")
        if not (-180.0 <= lon_f <= 180.0):
            raise ValueError("Долгота должна быть в диапазоне [-180, 180]")

        return lat_f, lon_f

    def _can_float(self, x: Any) -> bool:
        """
        Проверяет, можно ли безопасно привести значение к типу float.
        """
        try:
            float(x)
            return True
        except Exception:
            return False

    # Yandex API
    def _geocode_address(self, address: str) -> dict:
        """
        Выполняет прямое геокодирование адреса через Yandex Geocoder
        и возвращает нормализованный результат геокодинга.
        """
        params = {
            "apikey": self.api_key,
            "geocode": f"{address}, Россия",
            "format": "json",
            "lang": self.lang,
            "results": self.results,
        }
        data = self._request(params)
        return self._extract_geo(data)

    def _reverse_geocode(self, lat: float, lon: float) -> dict:
        """
        Выполняет обратное геокодирование координат через Yandex Geocoder
        и возвращает нормализованный результат геокодинга.
        """
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
        """
        Выполняет HTTP-запрос к API Yandex Geocoder и возвращает JSON-ответ.
        """
        r = requests.get(self.YANDEX_GEOCODER_URL, params=params, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def _extract_geo(self, data: dict) -> dict:
        """
        Извлекает координаты и адресные компоненты из ответа Yandex Geocoder.

        Разбирает JSON-ответ геокодера, извлекает координаты (lat, lon),
        текстовое представление адреса и список структурированных
        адресных компонентов.
        """
        collection = data.get("response", {}).get("GeoObjectCollection", {})
        members = collection.get("featureMember", [])
        if not members:
            raise ValueError("Геокодер не вернул результатов")

        geo_obj = members[0].get("GeoObject", {})
        meta = geo_obj.get("metaDataProperty", {}).get("GeocoderMetaData", {})
        text = meta.get("text")

        # Координаты
        pos = geo_obj.get("Point", {}).get("pos")
        if not pos:
            raise ValueError("Результат геокодирования не содержит координат")
        lon_s, lat_s = pos.split()
        lon = float(lon_s)
        lat = float(lat_s)

        # Компоненты адреса
        comps = meta.get("Address", {}).get("Components", []) or []

        return {
            "text": text,
            "lat": lat,
            "lon": lon,
            "components": comps,
            "raw": data,
        }

    # Приведение результата к формату LocationResult
    def _to_result(self, *, input_type: str, geo: dict) -> LocationResult:
        """
        Формирует объект LocationResult из нормализованных данных геокодинга.

        Преобразует ответ геокодера в единый формат LocationResult,
        извлекая адресные компоненты, координаты и служебные поля.
        """
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
        """
        Преобразует список адресных компонентов геокодера
        в словарь формата {kind: name}.
        """
        out: Dict[str, str] = {}
        for item in components:
            kind = item.get("kind")
            name = item.get("name")
            if kind and name and kind not in out:
                out[kind] = name
        return out

    def _is_russia(self, *, country: Optional[str]) -> bool:
        """
        Определяет, относится ли локация к Российской Федерации
        на основе значения поля country, возвращаемого геокодером.
        """
        if not country:
            return False

        return country == "Россия"
