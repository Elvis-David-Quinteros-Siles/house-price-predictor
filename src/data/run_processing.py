# src/data/run_processing.py
import argparse
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')

# Escala oficial de King County para "condition" (1-5).
CONDITION_LABELS = {1: "Poor", 2: "Fair", 3: "Average", 4: "Good", 5: "Very Good"}

def load_data(file_path):
    """Load the King County house sales CSV."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path, dtype={"id": str, "zipcode": str})

def to_project_schema(df):
    """Map the King County columns to the fields the API and the UI use."""
    return pd.DataFrame({
        "price": df["price"],
        "sqft": df["sqft_living"],
        "bedrooms": df["bedrooms"],
        "bathrooms": df["bathrooms"],
        "location": df["zipcode"],
        "year_built": df["yr_built"],
        "condition": df["condition"].map(CONDITION_LABELS),
    })

def clean_data(df):
    """Drop repeated sales and rows that are data-entry errors."""
    logger.info("Cleaning dataset")

    # 177 casas se vendieron dos veces: se conserva la venta más reciente para
    # que la misma casa no termine a la vez en train y en test.
    df_cleaned = df.sort_values("date").drop_duplicates("id", keep="last")
    logger.info(f"Removed {len(df) - len(df_cleaned)} repeated sales of the same house")

    # 0 dormitorios / 0 baños / 33 dormitorios son errores de captura. No se
    # filtra por precio: las casas caras son reales y el modelo debe cubrirlas.
    valid = df_cleaned["bedrooms"].between(1, 10) & (df_cleaned["bathrooms"] > 0)
    logger.info(f"Removed {(~valid).sum()} rows with invalid bedrooms/bathrooms")
    df_cleaned = df_cleaned[valid]

    df_cleaned = to_project_schema(df_cleaned)
    if df_cleaned.isnull().any().any():
        raise ValueError("Unexpected missing values after mapping the King County schema")

    return df_cleaned

def process_data(input_file, output_file):
    """Full data processing pipeline."""
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Clean data
    df_cleaned = clean_data(df)
    logger.info(f"Cleaned dataset shape: {df_cleaned.shape}")

    # Save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the King County house sales dataset.")
    parser.add_argument("--input", default="data/raw/kc_house_data.csv", help="Raw King County CSV")
    parser.add_argument("--output", default="data/processed/cleaned_house_data.csv", help="Cleaned CSV")
    args = parser.parse_args()
    process_data(input_file=args.input, output_file=args.output)
