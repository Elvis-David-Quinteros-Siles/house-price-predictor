# src/features/engineer.py
"""
Variables del modelo y pipeline completo (variables -> preprocesado -> modelo).

Todo vive en un único Pipeline de scikit-learn que se ajusta solo con los datos
de entrenamiento y se guarda en un .pkl: la API carga el mismo objeto, así que
entrenamiento e inferencia calculan las variables con el mismo código.
La API importa este archivo como módulo `engineer` (el .pkl guarda esa ruta).
"""
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

EARTH_RADIUS_KM = 6371.0

# Entradas que recibe el pipeline (las mismas que la API).
INPUT_FEATURES = ['sqft', 'sqft_lot', 'bedrooms', 'bathrooms', 'year_built', 'condition',
                  'grade', 'view', 'waterfront', 'lat', 'long']
# Variables que añade HouseFeatures.
NEIGHBOUR_FEATURES = ['nb_log_price', 'nb_log_price_sqft', 'nb_sqft', 'nb_dist_km']
CATEGORICAL_FEATURES = ['condition']
# lat/long sustituyen al zipcode: en validación cruzada dan R² 0.895 frente
# a 0.871 con zipcode, y la interfaz las obtiene de un clic en el mapa.
NUMERICAL_FEATURES = ['sqft', 'sqft_lot', 'bedrooms', 'bathrooms', 'year_built', 'bed_bath_ratio',
                      'grade', 'view', 'waterfront', 'lat', 'long', 'sale_month'] + NEIGHBOUR_FEATURES


def create_features(df):
    """Create the features that depend only on the house itself."""
    df_featured = df.copy()

    # No se crea house_age (año actual - year_built): los precios son de
    # 2014-2015 y la edad cambiaría con la fecha del build. Los árboles usan
    # year_built directamente.
    # Tampoco price_per_sqft: se deriva del precio (target) y filtraría la respuesta.
    ratio = df_featured['bedrooms'] / df_featured['bathrooms']
    df_featured['bed_bath_ratio'] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_featured


class HouseFeatures(BaseEstimator, TransformerMixin):
    """
    Adds create_features(), the sale month and the k nearest sales in training.

    - sale_month (meses desde enero de 2014) recoge la subida de precios entre
      2014 y 2015. Al predecir no hay fecha de venta: se usa el último mes de
      entrenamiento, es decir, el nivel de precios más reciente.
    - nb_*: media de log(precio), log(precio/sqft) y superficie de las k ventas
      más cercanas, y su distancia media. Describen el barrio mejor que lat/long
      solas. y llega ya en log (TransformedTargetRegressor).
      Al ajustar, cada casa se excluye de sus propios vecinos: si no, su precio
      se filtraría en sus variables.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self._fit(X, y)
        return self

    def transform(self, X):
        distances, idx = self.tree_.query(np.radians(X[['lat', 'long']].to_numpy()), k=self.k)
        return self._add(X, idx, distances)

    def fit_transform(self, X, y=None):
        coords = self._fit(X, y)
        distances, idx = self.tree_.query(coords, k=self.k + 1)
        # Quita la propia casa; con coordenadas repetidas puede no salir la primera.
        not_self = idx != np.arange(len(X))[:, None]
        not_self[not_self.all(axis=1), -1] = False
        idx = idx[not_self].reshape(len(X), self.k)
        distances = distances[not_self].reshape(len(X), self.k)
        return self._add(X, idx, distances)

    def _fit(self, X, y):
        if y is None:
            raise ValueError("HouseFeatures needs the target to compute neighbour prices")
        coords = np.radians(X[['lat', 'long']].to_numpy())
        self.tree_ = BallTree(coords, metric='haversine')
        self.log_price_ = np.asarray(y, dtype=float)
        self.log_price_sqft_ = self.log_price_ - np.log(X['sqft'].to_numpy())
        self.sqft_ = X['sqft'].to_numpy(dtype=float)
        self.latest_sale_month_ = int(X['sale_month'].max())
        return coords

    def _add(self, X, idx, distances):
        X = create_features(X)
        if 'sale_month' not in X:
            X['sale_month'] = self.latest_sale_month_
        X['nb_log_price'] = self.log_price_[idx].mean(axis=1)
        X['nb_log_price_sqft'] = self.log_price_sqft_[idx].mean(axis=1)
        X['nb_sqft'] = self.sqft_[idx].mean(axis=1)
        X['nb_dist_km'] = distances.mean(axis=1) * EARTH_RADIUS_KM
        return X


def create_preprocessor():
    """Imputation and one-hot encoding of the columns the model uses."""
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def build_pipeline(model, neighbours=10):
    """Full pipeline: raw inputs (INPUT_FEATURES + sale_month) -> price model."""
    return Pipeline(steps=[
        ('features', HouseFeatures(k=neighbours)),
        ('preprocessor', create_preprocessor()),
        ('model', model),
    ])
