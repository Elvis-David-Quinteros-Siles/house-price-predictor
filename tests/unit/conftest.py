import sys
from pathlib import Path

import pandas as pd
import pytest

# Los scripts del pipeline no son un paquete: se importan desde sus carpetas.
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "data"), str(ROOT / "src" / "features"), str(ROOT / "src" / "monitoring"),
                str(ROOT / "src" / "api"), str(ROOT / "scripts")]


@pytest.fixture
def raw_sales():
    """Few King County rows covering the cleaning cases."""
    base = dict(price=500000, bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000,
                condition=3, grade=7, view=0, waterfront=0, yr_built=1975,
                lat=47.66, long=-122.33, zipcode="98103")
    rows = [
        dict(base, id="1", date="20140601T000000", price=400000),
        dict(base, id="1", date="20150301T000000", price=450000),  # misma casa, venta posterior
        dict(base, id="2", date="20140701T000000", bedrooms=0),     # error de captura
        dict(base, id="3", date="20140801T000000", bedrooms=33),    # error de captura
        dict(base, id="4", date="20140901T000000", bathrooms=0),    # error de captura
        dict(base, id="5", date="20141001T000000", condition=5, grade=10, waterfront=1),
    ]
    return pd.DataFrame(rows)
