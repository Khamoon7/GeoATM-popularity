import asyncio
import pandas as pd

from geo_atm_popularity.features_builder import FeaturesBuilder


async def main():
    builder = FeaturesBuilder(
        regions_density_csv="data/regions_population_density_area.csv"
    )

    # Пример из трейнаы
    lat, lon = 54.270443, 48.300652

    df = await builder.build(
        lat=lat,
        lon=lon,
        atm_params={
            "normalized_address": "Москва, Красная площадь, 1",
            "province": "Ульяновская область",              # должно матчиться с "Субъект РФ" в CSV (после нормализации)
            "operations": ["withdraw", "deposit"],
            "bank_type": "sber",
            "cash_in": True,
            "is_247": True,
        },
    )

    print(df.T)      
    print("\nDtypes:\n", df.dtypes)

    # sanity-check по ключевым колонкам
    must_have = [
        "atm_lat", "atm_lon", "normalized_address", "province", "operations",
        "population_density_per_km2",
        "count_cafes_300m", "nearest_cafes_dist_m",
        "count_subway_300m", "nearest_subway_dist_m", "has_subway_nearby",
        "count_post_offices_300m", "nearest_post_offices_dist_m",
    ]
    missing = [c for c in must_have if c not in df.columns]
    print("\nMissing columns:", missing)


if __name__ == "__main__":
    asyncio.run(main())