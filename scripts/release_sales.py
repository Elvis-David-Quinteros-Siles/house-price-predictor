"""
Fuente de datos simulada para el entrenamiento continuo.

El dataset de King County es fijo (mayo 2014 - mayo 2015). Para que haya ventas
nuevas con precio real, el archivo completo hace de "mundo" y las ventas se
publican mes a mes en el dataset vivo (un volumen en el despliegue), con el que
entrena el modelo.

Uso:
  python scripts/release_sales.py --init 2015-02   # dataset vivo con las ventas hasta febrero de 2015
  python scripts/release_sales.py                  # publica el mes siguiente
"""
import argparse
from pathlib import Path

import pandas as pd

ARCHIVE = "data/raw/kc_house_data.csv"
LIVE = "data/live/kc_house_data.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", default=ARCHIVE, help="Full King County CSV")
    parser.add_argument("--live", default=LIVE, help="Dataset the trainer uses")
    parser.add_argument("--init", metavar="YYYY-MM", help="Start the live dataset with the sales up to this month")
    return parser.parse_args()


def sale_month(df):
    """Mes de cada venta (AAAA-MM) a partir de la fecha de King County (20141013T000000)."""
    return df["date"].str[:4] + "-" + df["date"].str[4:6]


def release(archive, live, init=None):
    """Escribe el dataset vivo y devuelve el mes publicado, o None si no quedan ventas."""
    world = pd.read_csv(archive, dtype=str)
    months = sale_month(world)
    if init:
        until = init
    elif not Path(live).exists():
        raise SystemExit(f"{live} does not exist yet: start it with --init YYYY-MM")
    else:
        published = sale_month(pd.read_csv(live, dtype=str)).max()
        later = months[months > published]
        if later.empty:
            return None
        until = later.min()
    Path(live).parent.mkdir(parents=True, exist_ok=True)
    # Se reescribe con las filas del archivo en su orden: el mismo corte da
    # siempre el mismo archivo y el mismo hash.
    world[months <= until].to_csv(live, index=False)
    return until


def main(args):
    until = release(args.archive, args.live, args.init)
    if until is None:
        print("No more sales to release: the live dataset already has the whole archive.")
        return
    rows = len(pd.read_csv(args.live, usecols=["date"]))
    print(f"Live dataset: sales up to {until} ({rows} rows) in {args.live}")


if __name__ == "__main__":
    main(parse_args())
