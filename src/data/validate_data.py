# src/data/validate_data.py
"""
Validación del dataset de King County antes de entrenar.

run_processing.py limpia los errores de captura conocidos (dormitorios y baños
imposibles, ventas repetidas de la misma casa); esto comprueba que el archivo
cumple el contrato que dan por supuesto la limpieza, el modelo y la API, y
para el ciclo si no: un modelo entrenado con datos rotos puede pasar la
puerta de calidad (su test está igual de roto) y llegar a producción.

  - columnas, nulos, formato de fecha y tipos;
  - rangos (los mismos que acepta la API) de cada variable;
  - cuánto descartaría la limpieza;
  - ventas duplicadas (un mes publicado dos veces);
  - meses seguidos, con un mínimo de ventas cada uno;
  - precio mediano del último mes frente al de los anteriores (un cambio de
    unidades o de moneda salta aquí).

Uso:
  python src/data/validate_data.py --input data/live/kc_house_data.csv
Salida 0 si es válido, 5 si no (con cada problema en la salida).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

INVALID = 5
REQUIRED = ['id', 'date', 'price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'yr_built',
            'condition', 'grade', 'view', 'waterfront', 'lat', 'long']
DATE_FORMAT = r'^\d{8}T000000$'  # 20141013T000000


def load_rules(config_path="configs/model_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["data_validation"]


def sale_months(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "-" + dates.str[4:6]


def validate(df: pd.DataFrame, rules: dict) -> list[str]:
    """Problemas del dataset (vacío si es válido). Se lee con dtype=str."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"]
    if df.empty:
        return ["no rows"]

    problems = []
    nulls = df[REQUIRED].isnull().sum()
    for column, count in nulls[nulls > 0].items():
        problems.append(f"{column}: {count} empty values")

    bad_dates = ~df["date"].fillna("").str.match(DATE_FORMAT)
    if bad_dates.any():
        problems.append(f"date: {bad_dates.sum()} values not in YYYYMMDDT000000 format "
                        f"(e.g. {df.loc[bad_dates, 'date'].iloc[0]!r})")

    numeric = df[REQUIRED[2:]].apply(pd.to_numeric, errors="coerce")
    not_numbers = numeric.isnull() & df[REQUIRED[2:]].notnull()
    for column, count in not_numbers.sum()[lambda s: s > 0].items():
        problems.append(f"{column}: {count} non-numeric values")

    for column, (low, high) in rules["ranges"].items():
        outside = ~numeric[column].between(low, high) & numeric[column].notnull()
        if outside.any():
            values = numeric.loc[outside, column]
            problems.append(f"{column}: {outside.sum()} values outside [{low:g}, {high:g}] "
                            f"(min {values.min():g}, max {values.max():g})")

    # Lo que run_processing.clean_data descarta por error de captura.
    dropped = ~(numeric["bedrooms"].between(1, 10) & (numeric["bathrooms"] > 0))
    if dropped.mean() > rules["max_dropped_share"]:
        problems.append(f"cleaning would drop {dropped.mean():.1%} of the rows as data-entry errors "
                        f"(max {rules['max_dropped_share']:.0%})")

    duplicated = df.duplicated(["id", "date"])
    if duplicated.any():
        problems.append(f"{duplicated.sum()} repeated sales (same house and date): a month published twice?")

    if bad_dates.any():
        return problems
    months = sale_months(df["date"])
    sales = months.value_counts().sort_index()
    expected = pd.period_range(sales.index[0], sales.index[-1], freq="M").strftime("%Y-%m")
    gaps = sorted(set(expected) - set(sales.index))
    if gaps:
        problems.append(f"months without sales: {', '.join(gaps)}")
    few = sales[sales < rules["min_month_sales"]]
    if not few.empty:
        problems.append("months with fewer than {} sales: {}".format(
            rules["min_month_sales"], ", ".join(f"{m} ({n})" for m, n in few.items())))

    latest = sales.index[-1]
    if len(sales) > 1:
        median_latest = numeric.loc[months == latest, "price"].median()
        median_before = numeric.loc[months < latest, "price"].median()
        shift = median_latest / median_before - 1
        if abs(shift) > rules["max_median_price_shift"]:
            problems.append(f"median price of {latest} is {shift:+.0%} against the previous months "
                            f"(${median_latest:,.0f} vs ${median_before:,.0f}, "
                            f"max ±{rules['max_median_price_shift']:.0%})")
    return problems


def summary(df: pd.DataFrame) -> str:
    months = sale_months(df["date"].dropna())
    return f"{len(df)} sales, {months.min()} to {months.max()}" if len(months) else f"{len(df)} rows"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raw/kc_house_data.csv", help="Raw King County CSV")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    problems = validate(df, load_rules(args.config))
    print(f"{Path(args.input).name}: {summary(df)}")
    if problems:
        for problem in problems:
            print(f"INVALID  {problem}")
        print(f"Data validation failed: {len(problems)} problems. Not training on this dataset.")
        return INVALID
    print("Data validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
