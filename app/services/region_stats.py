from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_STATS_DIR = PROJECT_ROOT / "data" / "new_stats"

# Колонки, которые отдаём в признаки модели
ECON_COLS = ["grp_per_capita_2023_rub", "avg_salary_oct_2025_rub", "avg_income_q3_2025_rub"]


def _norm(s: Any) -> str:
    """
    Нормализует название региона/города для сопоставления:
    нижний регистр, схлопывание пробелов, срез префиксов 'г.', 'город', 'ё'->'е'.
    """
    if s is None:
        return ""
    t = str(s).strip().lower().replace("ё", "е")
    for pref in ("г. ", "г.", "город ", "пос. ", "пгт "):
        if t.startswith(pref):
            t = t[len(pref):]
    return " ".join(t.split())


@dataclass
class RegionStats:
    """
    Джойн региональной/городской статистики из data/new_stats/.

    По названию города (locality) и региона (province) из геокодера возвращает
    экономические признаки, гео-флаги и категории для модели v2.
    """

    cities_csv: Path = field(default_factory=lambda: NEW_STATS_DIR / "cities_statistics_2025.csv")
    regions_csv: Path = field(default_factory=lambda: NEW_STATS_DIR / "regions_registry_2025.csv")

    _cities: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _regions: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cities = self._index_csv(self.cities_csv, "city_name")
        self._regions = self._index_csv(self.regions_csv, "region_name")

    @staticmethod
    def _index_csv(path: Path, key_col: str) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        if key_col not in df.columns:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for rec in df.to_dict(orient="records"):
            out[_norm(rec.get(key_col))] = rec
        return out

    def lookup(self, region: Optional[str], city: Optional[str]) -> Dict[str, Any]:
        """
        Возвращает признаки модели по региону/городу.

        Приоритет: экономику и city_size_category/federal_district берём из города,
        флаги (is_federal_city/...) и плотность населения — из реестра регионов.
        При промахе подставляются дефолты ('unknown' / False / NaN).
        """
        warnings: List[str] = []

        city_rec = self._cities.get(_norm(city)) if city else None
        region_key = _norm(region)
        region_rec = self._regions.get(region_key) if region else None

        # Если регион не нашёлся напрямую, пробуем регион из записи города
        if region_rec is None and city_rec is not None:
            region_rec = self._regions.get(_norm(city_rec.get("region_name")))

        if city and city_rec is None:
            warnings.append(f"Город не найден в реестре: {city}")
        if region and region_rec is None:
            warnings.append(f"Регион не найден в реестре: {region}")

        out: Dict[str, Any] = {
            "federal_district": "unknown",
            "city_size_category": "unknown",
            "is_federal_city": False,
            "is_federal_district_capital": False,
            "population_density_per_km2": float("nan"),
            "grp_per_capita_2023_rub": float("nan"),
            "avg_salary_oct_2025_rub": float("nan"),
            "avg_income_q3_2025_rub": float("nan"),
        }

        if region_rec is not None:
            out["federal_district"] = region_rec.get("federal_district", "unknown")
            out["is_federal_city"] = bool(region_rec.get("is_federal_city", False))
            out["is_federal_district_capital"] = bool(region_rec.get("is_federal_district_capital", False))
            out["population_density_per_km2"] = self._num(region_rec.get("population_density"))
            for c in ECON_COLS:
                out[c] = self._num(region_rec.get(c))

        if city_rec is not None:
            # городские значения приоритетнее региональных для этих полей
            out["city_size_category"] = city_rec.get("city_size_category", "unknown")
            if city_rec.get("federal_district"):
                out["federal_district"] = city_rec.get("federal_district")
            for c in ECON_COLS:
                val = self._num(city_rec.get(c))
                if val == val:  # not NaN
                    out[c] = val

        out["_warnings"] = warnings
        return out

    @staticmethod
    def _num(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
