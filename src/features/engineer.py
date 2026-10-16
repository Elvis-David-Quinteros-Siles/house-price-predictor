# src/features/engineer.py
import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data."""
    logger.info("Creating new features")
    
    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # No se crea house_age (año actual - year_built): los precios son de
    # 2014-2015 y la edad cambiaría con la fecha del build. Los árboles usan
    # year_built directamente.
    # Tampoco price_per_sqft: se deriva del precio (target) y filtraría la respuesta.

    # Bedroom to bathroom ratio
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    # Handle division by zero
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan)
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].fillna(0)
    logger.info("Created 'bed_bath_ratio' feature")
    
    # Do NOT one-hot encode categorical variables here; let the preprocessor handle it
    return df_featured

def create_preprocessor():
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessor pipeline")
    
    # Define feature groups
    categorical_features = ['condition']
    # lat/long sustituyen al zipcode: en validación cruzada dan R² 0.895 frente
    # a 0.871 con zipcode, y la interfaz las obtiene de un clic en el mapa.
    numerical_features = ['sqft', 'sqft_lot', 'bedrooms', 'bathrooms', 'year_built', 'bed_bath_ratio',
                          'grade', 'view', 'waterfront', 'lat', 'long']
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        # El CSV de features necesita una matriz densa.
        sparse_threshold=0
    )
    
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full feature engineering pipeline."""
    # Load cleaned data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Create features
    df_featured = create_features(df)
    logger.info(f"Created featured dataset with shape: {df_featured.shape}")
    
    # Create and fit the preprocessor
    preprocessor = create_preprocessor()
    X = df_featured.drop(columns=['price'], errors='ignore')  # Features only
    y = df_featured['price'] if 'price' in df_featured.columns else None  # Target column (if available)
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")
    
    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    # Coordenadas de las casas vendidas: la API rechaza ubicaciones lejos de
    # cualquier venta (agua, fuera del condado), donde el modelo no sabe nada.
    locations_file = Path(preprocessor_file).with_name('sold_locations.npy')
    np.save(locations_file, df[['lat', 'long']].to_numpy())
    logger.info(f"Saved {len(df)} sold locations to {locations_file}")
    
    # Save fully preprocessed data
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")
    
    return df_transformed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering for housing data.')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file (engineered features)')
    parser.add_argument('--preprocessor', required=True, help='Path for saving the preprocessor')
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output, args.preprocessor)
