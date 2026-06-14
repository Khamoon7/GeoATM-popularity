from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import math
import os
import random

import httpx
import pandas as pd

from app.services.region_stats import RegionStats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Вычисляет расстояние между двумя точками на сфере в метрах.

    Используется для расчёта расстояний до POI.
    """
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class FeaturesBuilder:
    """
    Сервис построения признаков по координатам банкомата.

    Выполняет запросы к Overpass API, агрегирует POI,
    добавляет региональные и геопризнаки.
    """

    # Endpoint и параметры можно переопределить через env (для самохоста/зеркала).
    # Дефолт — российское зеркало mail.ru: быстрое и со свежими данными по РФ.
    overpass_url: str = field(
        default_factory=lambda: os.getenv(
            "OVERPASS_URL", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
        )
    )

    count_radius_m: int = 300
    nearest_search_radius_m: int = 1000

    timeout_s: float = field(default_factory=lambda: float(os.getenv("OVERPASS_HTTP_TIMEOUT_S", "120")))
    overpass_timeout_s: int = field(default_factory=lambda: int(os.getenv("OVERPASS_QUERY_TIMEOUT_S", "90")))
    retries: int = 3

    sleep_on_504_s: float = 5.0
    sleep_on_exc_s: float = 3.0
    sleep_on_429_s: float = 8.0
    jitter_s: Tuple[float, float] = (0.2, 0.5)

    concurrency: int = field(default_factory=lambda: int(os.getenv("OVERPASS_CONCURRENCY", "4")))

    regions_density_csv: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "regions_population_density_area.csv"
    )
    csv_region_col: str = "Субъект РФ"
    csv_density_col: str = "Плотность населения, чел/км²"

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
        # --- Новые группы v2 (из data/new_poi_agg/async_parse_OSM.ipynb) ---
        "offices": ['"office"', '"building"="office"'],
        "payment_terminals": ['"amenity"="payment_terminal"'],
        "money_transfer": ['"amenity"="money_transfer"'],
        "shops_food_small": ['"shop"="convenience"', '"shop"="discount"'],
        "hypermarkets": ['"shop"="hypermarket"'],
        "markets": ['"amenity"="marketplace"', '"shop"="market"'],
        "fitness_sport": ['"leisure"="fitness_centre"', '"leisure"="sports_centre"'],
        "hotels_hostels": ['"tourism"="hotel"', '"tourism"="hostel"'],
        "railway_stations": ['"railway"="station"'],
        "residential_buildings": ['"building"="residential"'],
        "residential_landuse": ['"landuse"="residential"'],
        "fuel": ['"amenity"="fuel"'],
        "highway_pedestrian": ['"highway"="pedestrian"'],
        "landuse_mix": ['"landuse"~"commercial|retail|industrial|residential"'],
        # footway считается в радиусе 100м -> count_footway_100m_100m
        "footway_100m": ['"highway"="footway"'],
    })

    # Группы, для которых count/поиск идёт не в стандартном радиусе
    count_radius_overrides: Dict[str, int] = field(default_factory=lambda: {"footway_100m": 100})
    # Для групп, где модели нужен только count_*_300m (без nearest_*), ищем сразу в 300м,
    # а не в 1000м — это резко уменьшает объём ответа Overpass на тяжёлых слоях.
    search_radius_overrides: Dict[str, int] = field(default_factory=lambda: {
        "footway_100m": 100,
        "banks_atms": 300,
        "payment_terminals": 300,
        "money_transfer": 300,
        "hypermarkets": 300,
        "markets": 300,
        "railway_stations": 300,
        "residential_buildings": 300,
        "fuel": 300,
        "highway_pedestrian": 300,
        "landuse_mix": 300,
    })

    # Поиск центра города (place=city) и тип землепользования
    city_search_radius_m: int = 50_000
    city_center_threshold_m: int = 1500
    landuse_lookup_radius_m: int = 500

    _region_to_density: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _region_stats: RegionStats = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._region_to_density = self._load_region_density_map_safely()
        self._region_stats = RegionStats()

    async def build(
        self,
        payload: Optional[Dict[str, Any]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        atm_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Строит DataFrame признаков для одного банкомата.

        Принимает координаты напрямую или через payload/atm_params.
        """
        params: Dict[str, Any] = {}
        if payload:
            params.update(payload)
        if atm_params:
            params.update(atm_params)
        params.update(kwargs)

        lat_val = params.get("geo_lat", lat)
        lon_val = params.get("geo_lon", lon)
        if lat_val is None or lon_val is None:
            raise ValueError("Не заданы координаты (geo_lat/geo_lon или lat/lon)")

        lat_f = float(lat_val)
        lon_f = float(lon_val)

        row: Dict[str, Any] = {
            "geo_lat": lat_f,
            "geo_lon": lon_f,
            **params,
        }

        # Региональная/городская статистика: экономика, гео-флаги, плотность населения
        rstats = self._region_stats.lookup(params.get("region"), params.get("city"))
        rstats.pop("_warnings", None)
        row.update(rstats)

        # Fallback по плотности населения из старого CSV, если в реестре пусто
        dens = row.get("population_density_per_km2")
        if dens is None or (isinstance(dens, float) and math.isnan(dens)):
            row["population_density_per_km2"] = self._get_population_density(params.get("region"))

        # Убираем возможные дубли координат
        row.pop("atm_lat", None)
        row.pop("atm_lon", None)
        row.pop("lat", None)
        row.pop("lon", None)

        headers = {"User-Agent": "features-builder/1.0"}
        timeout = httpx.Timeout(self.timeout_s)

        sem = asyncio.Semaphore(max(1, int(self.concurrency)))

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:

            async def run_group(group_name: str, filters: List[str]) -> Dict[str, Any]:
                async with sem:
                    return await self._features_for_group(client, lat_f, lon_f, group_name, filters)

            async def run_city() -> Dict[str, Any]:
                async with sem:
                    return await self._city_center_features(client, lat_f, lon_f)

            async def run_landuse() -> Dict[str, Any]:
                async with sem:
                    return await self._land_use_features(client, lat_f, lon_f)

            tasks = [run_group(g, f) for g, f in self.poi_filters.items()]
            results = await asyncio.gather(*tasks, run_city(), run_landuse())

        osm_feats: Dict[str, Any] = {}
        for feats in results:
            osm_feats.update(feats)

        # Флаг наличия метро поблизости
        val = osm_feats.get("nearest_subway_dist_m", pd.NA)
        if val is pd.NA:
            osm_feats["has_subway_nearby"] = pd.NA
        elif isinstance(val, float) and math.isnan(val):
            osm_feats["has_subway_nearby"] = False
        else:
            osm_feats["has_subway_nearby"] = True

        row.update(osm_feats)

        df = pd.DataFrame([row])

        # Приведение типов счётчиков
        for c in df.columns:
            if c.startswith("count_") and c.endswith("m"):
                df[c] = df[c].astype("Int64")

        # Приведение типов расстояний
        for c in df.columns:
            if c.startswith("nearest_") and c.endswith("_dist_m"):
                df[c] = df[c].astype("Float64")

        df["has_subway_nearby"] = (
            df["has_subway_nearby"]
            .map({True: 1.0, False: 0.0})
            .astype("Float64")
        )

        return df

    @staticmethod
    def _normalize_region_name(s: Any) -> str:
        """
        Нормализует название региона для сопоставления.
        """
        return " ".join(str(s).strip().lower().split())

    def _load_region_density_map_safely(self) -> Dict[str, float]:
        """
        Загружает CSV с плотностью населения, если файл доступен.

        Возвращает пустой словарь при любой ошибке.
        """
        if not self.regions_density_csv.exists():
            return {}

        try:
            df = pd.read_csv(self.regions_density_csv, encoding="utf-8")
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
        Возвращает плотность населения для региона или NaN.
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
        Выполняет Overpass-запрос для набора фильтров.

        Возвращает список уникальных OSM-объектов или None при ошибке.
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
        Извлекает координаты объекта OSM.
        """
        if "lat" in el and "lon" in el:
            return float(el["lat"]), float(el["lon"])
        center = el.get("center")
        if center and "lat" in center and "lon" in center:
            return float(center["lat"]), float(center["lon"])
        return None

    def _dists(self, lat: float, lon: float, elements: List[Dict[str, Any]]) -> List[float]:
        """
        Считает расстояния от точки до списка OSM-объектов.
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
        Строит признаки для одной группы POI.
        """
        count_radius = self.count_radius_overrides.get(group, self.count_radius_m)
        search_radius = self.search_radius_overrides.get(group, self.nearest_search_radius_m)

        elements = await self.osm_queries(client, lat, lon, filters, radius_m=search_radius)

        if elements is None:
            return self._group_na(group)

        dists = self._dists(lat, lon, elements)
        cnt = int(sum(d <= count_radius for d in dists)) if dists else 0

        feats: Dict[str, Any] = {}
        feats[f"count_{group}_{count_radius}m"] = cnt

        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = float(min(dists)) if dists else float("nan")

        return feats

    def _group_na(self, group: str) -> Dict[str, Any]:
        """
        Возвращает NA-признаки для группы POI при ошибке запроса.
        """
        count_radius = self.count_radius_overrides.get(group, self.count_radius_m)
        feats: Dict[str, Any] = {f"count_{group}_{count_radius}m": pd.NA}
        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = pd.NA
        return feats

    async def _city_center_features(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Dict[str, Any]:
        """
        Ищет ближайший центр города (place=city) и считает is_city_center.
        """
        elements = await self.osm_queries(
            client, lat, lon, ['"place"="city"'], radius_m=self.city_search_radius_m
        )
        dists = self._dists(lat, lon, elements) if elements else []
        if not dists:
            return {"city_center_dist_m": float("nan"), "is_city_center": 0}
        d = float(min(dists))
        return {
            "city_center_dist_m": d,
            "is_city_center": 1 if d <= self.city_center_threshold_m else 0,
        }

    async def _land_use_features(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Dict[str, Any]:
        """
        Определяет тип землепользования и расстояние до него.

        Stage 1: ближайший landuse (commercial/retail/industrial/residential).
        Stage 2 (fallback): building=residential / office как прокси.
        """
        elements = await self.osm_queries(
            client, lat, lon,
            ['"landuse"~"commercial|retail|industrial|residential"'],
            radius_m=self.landuse_lookup_radius_m,
        )
        best = self._nearest_with_tag(lat, lon, elements, tag_key="landuse")
        if best is not None:
            return {"land_use_type": self._normalize_landuse(best[1]), "land_use_dist_m": round(best[0], 1)}

        proxy = await self.osm_queries(
            client, lat, lon,
            ['"building"="residential"', '"office"', '"building"="office"'],
            radius_m=self.landuse_lookup_radius_m,
        )
        best2 = self._nearest_proxy_landuse(lat, lon, proxy)
        if best2 is not None:
            return {"land_use_type": best2[1], "land_use_dist_m": round(best2[0], 1)}

        return {"land_use_type": "unknown", "land_use_dist_m": float("nan")}

    @staticmethod
    def _normalize_landuse(val: Any) -> str:
        """
        Нормализует тег landuse: retail -> commercial, остальное как есть.
        """
        v = (str(val) if val is not None else "").strip().lower()
        if v in ("retail", "commercial"):
            return "commercial"
        if v in ("industrial", "residential"):
            return v
        return v or "unknown"

    def _nearest_with_tag(
        self, lat: float, lon: float, elements: Optional[List[Dict[str, Any]]], tag_key: str
    ) -> Optional[Tuple[float, str]]:
        """
        Находит ближайший объект с заданным тегом, возвращает (расстояние, значение тега).
        """
        if not elements:
            return None
        best: Optional[Tuple[float, str]] = None
        for el in elements:
            ll = self._get_latlon(el)
            if not ll:
                continue
            val = (el.get("tags") or {}).get(tag_key)
            if not val:
                continue
            d = distance_m(lat, lon, ll[0], ll[1])
            if best is None or d < best[0]:
                best = (d, val)
        return best

    def _nearest_proxy_landuse(
        self, lat: float, lon: float, elements: Optional[List[Dict[str, Any]]]
    ) -> Optional[Tuple[float, str]]:
        """
        Прокси-определение землепользования по зданиям: residential / office.
        """
        if not elements:
            return None
        best: Optional[Tuple[float, str]] = None
        for el in elements:
            ll = self._get_latlon(el)
            if not ll:
                continue
            tags = el.get("tags") or {}
            if tags.get("building") == "residential":
                kind = "residential"
            elif tags.get("office") or tags.get("building") == "office":
                kind = "commercial"
            else:
                continue
            d = distance_m(lat, lon, ll[0], ll[1])
            if best is None or d < best[0]:
                best = (d, kind)
        return best
