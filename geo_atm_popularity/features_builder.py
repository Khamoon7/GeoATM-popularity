from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import math
import random

import httpx
import pandas as pd


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Считает расстояние между двумя точками по формуле гаверсинусов (в метрах).
    """
    R = 6371000  # радиус Земли в метрах
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class FeaturesBuilder:
    """
    ## FeaturesBuilder

    Класс генерации признаков для одного банкомата/точки.

    ### Вход
    - координаты: `lat`, `lon`
    - дополнительные поля банкомата: адрес, операции, и т.д. (через `atm_params` и/или `**kwargs`)

    ### OSM (Overpass)
    - `count_<group>_300m`: количество объектов в радиусе 300м
    - `nearest_<group>_dist_m`: расстояние до ближайшего объекта (поиск кандидатов в радиусе 1000м)
    - ошибки OSM: NaN/NA проставляется только для тех признаков, которые не удалось получить (по группам)

    ### Плотность населения в регионе
    - `population_density_per_km2`: берётся из CSV `data/regions_population_density_area.csv`
      по `region` (вход) -> `"Субъект РФ"` (CSV) -> `"Плотность населения, чел/км²"`.

    ### Выход
    - `pandas.DataFrame` из одной строки с исходными параметрами + OSM признаками.
    """

    overpass_url: str = "https://overpass-api.de/api/interpreter"

    count_radius_m: int = 300
    nearest_search_radius_m: int = 1000

    timeout_s: float = 60.0  # сетевой timeout httpx
    overpass_timeout_s: int = 60  # timeout на стороне Overpass
    retries: int = 3

    sleep_on_504_s: float = 5.0
    sleep_on_exc_s: float = 3.0
    sleep_on_429_s: float = 8.0
    jitter_s: Tuple[float, float] = (0.2, 0.5)

    # ограничение параллелизма запросов по группам (чтобы меньше ловить 429)
    concurrency: int = 3

    regions_density_csv: str = "data/regions_population_density_area.csv"
    csv_region_col: str = "Субъект РФ"
    csv_density_col: str = "Плотность населения, чел/км²"

    # Фильтры для OpenStreetMap
    poi_filters: Dict[str, List[str]] = field(default_factory=lambda: {
        "malls": ['"shop"="mall"'],
        "supermarkets": ['"shop"="supermarket"', '"shop"="alcohol"'],
        "pharmacies_hospitals": ['"amenity"="pharmacy"', '"amenity"="hospital"', '"amenity"="clinic"'],
        "banks_atms": ['"amenity"="bank"', '"amenity"="atm"'],
        "cafes": ['"amenity"="cafe"', '"amenity"="coffee_shop"'],
        "restaurants": ['"amenity"="restaurant"', '"amenity"="fast_food"'],
        "public_transport": ['"public_transport"="stop_position"', '"highway"="bus_stop"'],
        "parking": ['"amenity"="parking"'],

        "education": [
            '"amenity"="school"',
            '"amenity"="university"',
            '"name"~"моу|сош|гимназия|лицей|институт|университет|академия|ниу|school|academy|camp",i',
        ],
        "subway": [
            '"railway"="subway"',
            '"railway"="station"',
            '"name"~"метро|subway",i',
        ],
        "post_offices": [
            '"amenity"="post_office"',
            '"name"~"почта|отделение связи|post",i',
        ],
    })

    _region_to_density: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        ## __post_init__

        Хук dataclass, выполняется сразу после инициализации.

        ### Что делает
        - Загружает справочник плотности населения из CSV и строит кэш-маппинг:
          `normalize("Субъект РФ") -> density`.
        """
        self._region_to_density = self._load_region_density_map_safely()

    async def build(
        self,
        lat: float,
        lon: float,
        atm_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        
        """
        ## build

        Главный метод генерации признаков для одного банкомата/точки.

        ### Параметры
        - `lat`, `lon`: координаты банкомата
        - `atm_params`: словарь признаков/полей банкомата (адрес и т.д.)
        - `**kwargs`: дополнительные поля (перекрывают `atm_params`, если ключи совпадают)

        ### Логика
        1. Собирает "сырой" df.row из входных данных (координаты + доступные операции).
        2. Добавляет `population_density_per_km2` по `region` из справочных данных.
        3. Для каждой группы POI:
           - делает запрос к OSM в радиусе 1000м
           - считает `count_*_300m` и `nearest_*_dist_m`
           - если OSM недоступен для группы — заполняет **только её признаки** NA/NaN.
        4. Добавляет `has_subway_nearby` (nullable boolean).

        ### Возвращает
        - `pd.DataFrame`: одна строка с признаками
        """
        
        lat = float(lat)
        lon = float(lon)

        params: Dict[str, Any] = {}
        if atm_params:
            params.update(atm_params)
        params.update(kwargs)

        row: Dict[str, Any] = {
            "atm_lat": lat,
            "atm_lon": lon,
            **params,
        }

        # плотность населения
        row["population_density_per_km2"] = self._get_population_density(params.get("region"))

        # один клиент на весь build
        headers = {"User-Agent": "features-builder/1.0 (contact: you@example.com)"}
        timeout = httpx.Timeout(self.timeout_s)

        sem = asyncio.Semaphore(max(1, int(self.concurrency)))

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:

            async def run_group(group_name: str, filters: List[str]) -> Dict[str, Any]:
                async with sem:
                    return await self._features_for_group(client, lat, lon, group_name, filters)

            tasks = [run_group(g, f) for g, f in self.poi_filters.items()]
            results = await asyncio.gather(*tasks)

        osm_feats: Dict[str, Any] = {}
        for feats in results:
            osm_feats.update(feats)

        # has_subway_nearby (nullable boolean)
        # - pd.NA, если ошибка OSM по группе subway (nearest = pd.NA)
        # - False, если данных нет (nearest = NaN или missing)
        # - True, если nearest число
        val = osm_feats.get("nearest_subway_dist_m", pd.NA)
        if val is pd.NA:
            osm_feats["has_subway_nearby"] = pd.NA
        elif isinstance(val, float) and math.isnan(val):
            osm_feats["has_subway_nearby"] = False
        else:
            osm_feats["has_subway_nearby"] = True

        row.update(osm_feats)

        df = pd.DataFrame([row])

        # counts => nullable Int64
        for c in df.columns:
            if c.startswith("count_") and c.endswith("m"):
                df[c] = df[c].astype("Int64")

        # nearest => nullable Float64, чтобы поддерживать pd.NA (ошибка OSM)
        for c in df.columns:
            if c.startswith("nearest_") and c.endswith("_dist_m"):
                df[c] = df[c].astype("Float64")

        # has_subway_nearby => nullable boolean
        df["has_subway_nearby"] = df["has_subway_nearby"].astype("boolean")

        return df

    @staticmethod
    def _normalize_region_name(s: Any) -> str:
        """
        ## _normalize_region_name
        Нормализует название региона для сопоставления со справочником.
        """
        return " ".join(str(s).strip().lower().split())

    def _load_region_density_map_safely(self) -> Dict[str, float]:
        """
        ## _load_region_density_map_safely

        Загружает CSV `data/regions_population_density_area.csv` и строит маппинг:
        `normalize("Субъект РФ") -> float("Плотность населения, чел/км²")`.

        ### Поведение при ошибках
        - Если файл не найден / не читается / нет нужных колонок — возвращает `{}`.

        ### Возвращает
        - `dict[str, float]`: словарь регион → плотность
        """
        candidates: List[Path] = [Path(self.regions_density_csv)]

        try:
            here = Path(__file__).resolve()
            candidates.append(here.parent / self.regions_density_csv)
            candidates.append(here.parent.parent / self.regions_density_csv)
            candidates.append(here.parent.parent.parent / self.regions_density_csv)
        except NameError:
            pass

        csv_path = next((p.resolve() for p in candidates if p.exists()), None)
        if csv_path is None:
            return {}

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            return {}

        if self.csv_region_col not in df.columns or self.csv_density_col not in df.columns:
            return {}

        mapping: Dict[str, float] = {}
        sub = df[[self.csv_region_col, self.csv_density_col]].dropna(subset=[self.csv_region_col])

        for _, r in sub.iterrows():
            key = self._normalize_region_name(r[self.csv_region_col])
            raw_val = r[self.csv_density_col]
            try:
                val = float(str(raw_val).replace(",", ".").strip())
            except Exception:
                continue
            mapping[key] = val

        return mapping

    def _get_population_density(self, region: Any) -> float:
        """
        ## _get_population_density

        Возвращает `population_density_per_km2` по входному `region`
        (из твоего `LocationResult.region`).

        ### Источник
        CSV: `data/regions_population_density_area.csv`

        - поиск по колонке `"Субъект РФ"`
        - значение берём из колонки `"Плотность населения, чел/км²"`

        ### Поведение при ошибках/неполадках
        - если `region` пустой или не найден в маппинге → `NaN`

        ### Параметры
        - `region`: название региона (строка)

        ### Возвращает
        - `float`: плотность населения или `NaN`
        """
        if region is None or (isinstance(region, str) and not region.strip()):
            return float("nan")
        key = self._normalize_region_name(region)
        val = self._region_to_density.get(key)
        return float(val) if val is not None else float("nan")

    async def osm_queries(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        filters: List[str],
        *,
        radius_m: int,
    ) -> Optional[List[Dict[str, Any]]]:
        
        """
        ## osm_queries

        Выполняет запрос к Overpass API для набора фильтров.

        ### Как формируется запрос
        Для каждого фильтра `f` добавляются подзапросы:
        - `node[f](around:R,lat,lon)`
        - `way[f](around:R,lat,lon)`
        - `relation[f](around:R,lat,lon)`

        Все объединяются в один Overpass-запрос:
        - `out center;` чтобы у `way/relation` был центр геометрии.

        ### Обработка ошибок и ретраи
        - `200`: парсим JSON и возвращаем `elements`
        - `504`: ретрай с паузой `sleep_on_504_s`
        - `429`: ретрай с учётом `Retry-After` (если есть), иначе `sleep_on_429_s`
        - другие коды: считаем ошибкой получения и возвращаем `None`
        - исключения (timeout/transport/json-ошибки): ретрай с `sleep_on_exc_s`

        ### Важно
        - `None` означает ошибка OSM (нужно проставлять NA/NaN только для соответствующих признаков).
        - `[]` означает OSM ответил, но объектов по фильтрам нет.

        ### Параметры
        - `lat`, `lon`: координаты
        - `filters`: список фильтров вида `shop="mall"` или `name~"метро|subway"`
        - `radius_m`: радиус поиска для Overpass

        ### Возвращает
        - `list[dict]`: список элементов OSM, если успешно
        - `None`: если не удалось получить данные
        """
        objects: List[str] = []
        for f in filters:
            objects.append(f"node[{f}](around:{radius_m},{lat},{lon});")
            objects.append(f"way[{f}](around:{radius_m},{lat},{lon});")
            objects.append(f"relation[{f}](around:{radius_m},{lat},{lon});")

        query = f"[out:json][timeout:{int(self.overpass_timeout_s)}];({''.join(objects)});out center;"

        for _ in range(self.retries):
            try:
                r = await client.get(self.overpass_url, params={"data": query})

                if r.status_code == 200:
                    data = r.json() or {}
                    elements = data.get("elements", [])
                    # На случай если объект попал в результаты несколько раз из-за пересечения фильтров
                    uniq: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
                    for el in elements:
                        k = (el.get("type"), el.get("id"))
                        uniq[k] = el
                    return list(uniq.values())

                if r.status_code == 504:
                    await asyncio.sleep(self.sleep_on_504_s + random.uniform(*self.jitter_s))
                    continue

                if r.status_code == 429:
                    ra = r.headers.get("retry-after")
                    if ra is not None:
                        try:
                            wait_s = float(ra)
                        except Exception:
                            wait_s = self.sleep_on_429_s
                    else:
                        wait_s = self.sleep_on_429_s

                    await asyncio.sleep(wait_s + random.uniform(*self.jitter_s))
                    continue

                return None

            except Exception:
                await asyncio.sleep(self.sleep_on_exc_s + random.uniform(*self.jitter_s))

        return None

    @staticmethod
    def _get_latlon(el: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        ## _get_latlon

        Извлекает координаты элемента Overpass.

        ### Поддерживаемые случаи
        - `node`: координаты лежат в `el["lat"]`, `el["lon"]`
        - `way/relation`: координаты лежат в `el["center"]["lat/lon"]` (если использован `out center`)

        ### Параметры
        - `el`: элемент из `elements` ответа Overpass

        ### Возвращает
        - `(lat, lon)` или `None`, если координаты недоступны
        """
        if "lat" in el and "lon" in el:
            return float(el["lat"]), float(el["lon"])
        center = el.get("center")
        if center and "lat" in center and "lon" in center:
            return float(center["lat"]), float(center["lon"])
        return None

    def _dists(self, lat: float, lon: float, elements: List[Dict[str, Any]]) -> List[float]:
        """
        ## _dists

        Преобразует список элементов OSM в список расстояний до них от точки (`lat`, `lon`).

        ### Параметры
        - `lat`, `lon`: координаты банкомата
        - `elements`: список OSM-элементов (node/way/relation)

        ### Возвращает
        - `list[float]`: расстояния в метрах до всех элементов, у которых удалось достать координаты
        """
        out: List[float] = []
        for el in elements:
            ll = self._get_latlon(el)
            if not ll:
                continue
            out.append(distance_m(lat, lon, ll[0], ll[1]))
        return out

    async def _features_for_group(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        group: str,
        filters: List[str],
    ) -> Dict[str, Any]:
        
        """
        ## _features_for_group

        Считает признаки для одной группы POI.

        ### Что считает
        - `count_<group>_300m`: количество объектов в радиусе `count_radius_m` (обычно 300м)
        - `nearest_<group>_dist_m`: расстояние до ближайшего объекта (кандидаты ищутся в `nearest_search_radius_m`, обычно 1000м)

        ### Параметры
        - `lat`, `lon`: координаты банкомата
        - `group`: имя группы (ключ словаря `poi_filters`)
        - `filters`: список OSM-фильтров для группы

        ### Возвращает
        - `dict`: признаки по группе
        """
        elements = await self.osm_queries(client, lat, lon, filters, radius_m=self.nearest_search_radius_m)

        if elements is None:
            return self._group_na(group)

        dists = self._dists(lat, lon, elements)
        cnt_300 = int(sum(d <= self.count_radius_m for d in dists)) if dists else 0

        feats: Dict[str, Any] = {}
        feats[f"count_{group}_{self.count_radius_m}m"] = cnt_300

        # для banks_atms nearest не создаём
        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = float(min(dists)) if dists else float("nan")

        return feats

    def _group_na(self, group: str) -> Dict[str, Any]:
        """
        ## _group_na

        Возвращает словарь признаков для группы в случае ошибки OSM.

        ### Правила заполнения
        - `count_*` → `pd.NA` (nullable Int64)
        - `nearest_*_dist_m` → `NaN` (float)
        - для `banks_atms` nearest не создаём (как в основной логике)

        ### Параметры
        - `group`: имя группы

        ### Возвращает
        - `dict`: NA/NaN-значения только по этой группе
        """
        feats: Dict[str, Any] = {f"count_{group}_{self.count_radius_m}m": pd.NA}
        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = pd.NA
        return feats