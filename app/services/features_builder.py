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

    overpass_url: str = "https://overpass-api.de/api/interpreter"

    count_radius_m: int = 300
    nearest_search_radius_m: int = 1000

    timeout_s: float = 60.0
    overpass_timeout_s: int = 60
    retries: int = 3

    sleep_on_504_s: float = 5.0
    sleep_on_exc_s: float = 3.0
    sleep_on_429_s: float = 8.0
    jitter_s: Tuple[float, float] = (0.2, 0.5)

    concurrency: int = 3

    regions_density_csv: str = "data/regions_population_density_area.csv"
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
    })

    _region_to_density: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._region_to_density = self._load_region_density_map_safely()

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
            "population_density_per_km2": self._get_population_density(params.get("region")),
        }

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

            tasks = [run_group(g, f) for g, f in self.poi_filters.items()]
            results = await asyncio.gather(*tasks)

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

        df["has_subway_nearby"] = df["has_subway_nearby"].astype("boolean")

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
        elements = await self.osm_queries(client, lat, lon, filters, radius_m=self.nearest_search_radius_m)

        if elements is None:
            return self._group_na(group)

        dists = self._dists(lat, lon, elements)
        cnt_300 = int(sum(d <= self.count_radius_m for d in dists)) if dists else 0

        feats: Dict[str, Any] = {}
        feats[f"count_{group}_{self.count_radius_m}m"] = cnt_300

        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = float(min(dists)) if dists else float("nan")

        return feats

    def _group_na(self, group: str) -> Dict[str, Any]:
        """
        Возвращает NA-признаки для группы POI при ошибке запроса.
        """
        feats: Dict[str, Any] = {f"count_{group}_{self.count_radius_m}m": pd.NA}
        if group != "banks_atms":
            feats[f"nearest_{group}_dist_m"] = pd.NA
        return feats
