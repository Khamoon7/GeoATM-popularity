import numpy as np
import pandas as pd

from app.core.model import ATMModelService


def test_random_prediction():
    service = ATMModelService(
        model_path="app/models/final_atm_pipeline.pkl"
    )

    # Минимально допустимый вход
    # raw_df = pd.DataFrame([{
    #     "id": 5.0,
    #     "atm_group": 496.5,
    #     "geo_lon": 44.260605,
    #     "geo_lat": 46.318231,
    #     "population_density_per_km2": 3.52,
    #     "is_24_7": False,
    #     "contactless_tech": False,
    #     "qr_codes": False,
    #     "usd_available": False,
    #     "eur_available": False,
    #     "cash_in": True,
    #     "cash_out": True,
    #     "cashless_pay": False,
    #     "account_statement": True,
    #     "access_for_disabled": True,
    #     "transfer_p2p": True,
    #     "transfer_a2a": False,
    #     "loan_payments": False,
    #     "nearest_malls_dist_m": 928.7,
    #     "count_malls_300m": 0,
    #     "nearest_supermarkets_dist_m": 364.3,
    #     "count_supermarkets_300m": 0,
    #     "nearest_pharmacies_hospitals_dist_m": 242.6,
    #     "count_pharmacies_hospitals_300m": 2,
    #     "count_banks_atms_300m": 2,
    #     "nearest_cafes_dist_m": 124.5,
    #     "count_cafes_300m": 1,
    #     "nearest_restaurants_dist_m": 317.9,
    #     "count_restaurants_300m": 0,
    #     "nearest_public_transport_dist_m": 93.6,
    #     "count_public_transport_300m": 5,
    #     "nearest_parking_dist_m": 143.7,
    #     "count_parking_300m": 3,
    #     "nearest_education_dist_m": 247.3,
    #     "count_education_300m": 1,
    #     "nearest_subway_dist_m": 0.0,
    #     "nearest_post_offices_dist_m": 0.0,
    #     "count_post_offices_300m": 0,
    #     "has_subway_nearby": False,
    # }])

    # raw_df = pd.DataFrame([{
    # "atm_lat": 55.75,
    # "atm_lon": 37.61,
    # "cash_in": True,
    # }])

    # raw_df = pd.DataFrame([{
    #     "atm_lat": 55.75,
    #     "atm_lon": 37.61,
    #     "cash_in": True,
    #     "cash_out": False,
    #     "random_trash_feature": "HELLO",
    #     "another_one": 12345,
    # }])

    # raw_df = pd.DataFrame([{
    #     "cash_in": "YES",
    #     "cash_out": "NO",
    #     "nearest_malls_dist_m": "far",
    # }])

    prediction, warnings = service.predict(raw_df)

    print("=== INTEGRATION TEST ===")
    print("Prediction:", prediction)
    print("Warnings:")
    for w in warnings:
        print("-", w)

    assert isinstance(prediction, float)


if __name__ == "__main__":
    test_random_prediction()
