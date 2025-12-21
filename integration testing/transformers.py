import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


EXCLUDE_COLS = [
    "id", "address_raw", "address_geocoded", "street", "house", 
    "geo_lon", "geo_lat", "municipality", "country", "atm_group" # ПОКА ВЫКИНУЛА atm_group из обучения 
]

BIN_FEATURES = [
    "is_24_7", "contactless_tech", "qr_codes", "usd_available", "eur_available",
    "cash_in", "cash_out", "cashless_pay", "account_statement", 
    "access_for_disabled", "transfer_p2p", "transfer_a2a", "loan_payments", 
    "has_subway_nearby"
]

LOCATION_FEATURES = [
    "population_density_per_km2", "nearest_malls_dist_m", "count_malls_300m",
    "nearest_supermarkets_dist_m", "nearest_pharmacies_hospitals_dist_m",
    "count_pharmacies_hospitals_300m", "count_banks_atms_300m",
    "nearest_cafes_dist_m", "count_cafes_300m", "nearest_restaurants_dist_m",
    "count_restaurants_300m", "nearest_public_transport_dist_m",
    "count_public_transport_300m", "nearest_parking_dist_m", "count_parking_300m",
    "nearest_education_dist_m", "count_education_300m", "nearest_subway_dist_m",
    "nearest_post_offices_dist_m"
]

CATEGORICAL_FEATURES = ["city"]
TARGET = "target"




class RegionGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, rare_threshold=0.01):
        self.rare_threshold = rare_threshold
        self.rare_regions_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if 'region' in df.columns:
            cnt = df['region'].value_counts()
            self.rare_regions_ = set(
                cnt[cnt < self.rare_threshold * len(df)].index
            )
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        if 'region' in df.columns and self.rare_regions_ is not None:
            df['region_grouped'] = df['region'].apply(
                lambda x: '__OTHER__' if pd.isna(x) or x in self.rare_regions_ else str(x)
            )
            df.drop(columns=['region'], inplace=True, errors='ignore')
        return df


class BinaryToFloat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)

        for col in BIN_FEATURES:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0)
                    .astype(float)
                )
            else:
                df[col] = 0.0

        return df



class ATMLocationFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)

        dist_cols = [c for c in df.columns if "dist_m" in c]
        for c in dist_cols:
            df[f"{c}_log"] = np.log1p(df[c].fillna(0))

        count_cols = [c for c in df.columns if c.startswith("count_")]
        if count_cols:
            df["poi_density_300m"] = df[count_cols].sum(axis=1)

        return df
    
class DropNonNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.select_dtypes(include=["number"])