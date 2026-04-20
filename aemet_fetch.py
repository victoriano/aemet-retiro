# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "httpx", "pandas"]
# ///
"""
Descarga/actualiza los datos diarios de una estación AEMET desde la API oficial.

Uso:
    uv run aemet_fetch.py                        # actualiza 3195 (Retiro) incrementalmente
    uv run aemet_fetch.py --station 0076         # Barcelona El Prat
    uv run aemet_fetch.py --station 3195 --from 2024-01-01
    uv run aemet_fetch.py --station 3195 --refresh   # rebaja el caché y baja todo

Salida: `data/{estacion}.parquet` con columnas (fecha DATE, tmax DOUBLE, tmin DOUBLE).

API: https://opendata.aemet.es  ·  key en ~/.config/aemet/api_key o $AEMET_API_KEY.
Límites: ~50 req/min, máximo 5 años por petición. Cada llamada devuelve un
puntero (campo `datos`) y hacemos una segunda GET al S3-like para el payload.
"""
import argparse
import asyncio
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

API_BASE = "https://opendata.aemet.es/opendata/api"
DAILY_URL = (
    API_BASE
    + "/valores/climatologicos/diarios/datos/"
    + "fechaini/{fi}/fechafin/{ff}/estacion/{sta}"
)
STATIONS_URL = API_BASE + "/valores/climatologicos/inventarioestaciones/todasestaciones"

MAX_DAYS_PER_CHUNK = 180  # AEMET limita datos diarios a 6 meses por petición
REQ_INTERVAL = 60.0 / 30  # ~30 req/min, margen cómodo sobre el límite de 50


def api_key() -> str:
    tok = os.environ.get("AEMET_API_KEY")
    if tok:
        return tok
    p = Path("~/.config/aemet/api_key").expanduser()
    if p.exists():
        return p.read_text().strip()
    raise SystemExit(
        "No AEMET API key. Set $AEMET_API_KEY or write ~/.config/aemet/api_key"
    )


def _loose_json(text: str) -> dict | list | None:
    """AEMET a veces devuelve JSON con caracteres Latin-1 (acentos)."""
    for enc_try in ("utf-8", "latin-1"):
        try:
            return json.loads(text.encode("latin-1").decode(enc_try))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def aemet_get(client: httpx.AsyncClient, url: str, key: str, *, tries: int = 6) -> list | dict:
    """Dos pasos: /api/... devuelve {estado, datos}; luego GET a `datos`.
    AEMET puede devolver HTTP 200 con `estado: 429` cuando se satura, y el
    fichero en `datos` se prepara en background (HTML de error si no listo)."""
    delay = 3.0
    env: dict | None = None
    for attempt in range(tries):
        r = await client.get(url, headers={"api_key": key, "Accept": "application/json"})
        if r.status_code == 404:
            return []
        # tolerar encodings raros en los mensajes de error
        env = _loose_json(r.text)
        if env is None:
            await asyncio.sleep(delay)
            delay = min(delay * 1.7, 45)
            continue
        estado = env.get("estado")
        if estado in (429, 500, 502, 503, 504):
            print(f"  AEMET {estado} → reintento en {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.7, 45)
            continue
        break
    else:
        raise RuntimeError(f"AEMET sigue saturada tras {tries} intentos: {env}")

    estado = env.get("estado")
    if estado == 404:
        return []
    if estado != 200:
        raise RuntimeError(f"AEMET estado={estado}: {env.get('descripcion')}  ({url})")

    # Fichero de datos: puede estar preparándose todavía.
    datos_url = env["datos"]
    backoff = 2.5
    for attempt in range(tries + 2):
        r2 = await client.get(datos_url, timeout=60.0)
        ct = r2.headers.get("content-type", "")
        text = r2.text
        if r2.status_code == 200 and text.lstrip().startswith(("[", "{")):
            parsed = _loose_json(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # AEMET a veces envuelve lista en dict — ver si hay error
                if parsed.get("estado") and parsed["estado"] != 200:
                    print(f"  datos envelope error: {parsed}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.7, 30)
                    continue
                return parsed
        await asyncio.sleep(backoff)
        backoff = min(backoff * 1.7, 30)
    raise RuntimeError(f"AEMET datos nunca listos: {datos_url}")


def chunk_date_range(start: date, end: date, days: int = MAX_DAYS_PER_CHUNK):
    """Divide [start, end] en trozos contiguos de como máximo `days` días."""
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def iso(d: date, end_of_day: bool = False) -> str:
    t = "23:59:59" if end_of_day else "00:00:00"
    return f"{d:%Y-%m-%d}T{t}UTC"


def parse_num(v) -> float | None:
    if v in (None, "", "Ip"):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def normalize(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        try:
            d = datetime.strptime(r["fecha"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        records.append({"fecha": d, "tmax": parse_num(r.get("tmax")), "tmin": parse_num(r.get("tmin"))})
    return pd.DataFrame(records, columns=["fecha", "tmax", "tmin"])


async def fetch_station_range(station: str, start: date, end: date, key: str) -> pd.DataFrame:
    print(f"Descargando {station}: {start} → {end}  ({(end - start).days + 1} días)")
    rows: list[dict] = []
    async with httpx.AsyncClient(http2=False) as client:
        first = True
        for s, e in chunk_date_range(start, end):
            if not first:
                await asyncio.sleep(REQ_INTERVAL)
            first = False
            url = DAILY_URL.format(fi=iso(s), ff=iso(e, end_of_day=True), sta=station)
            chunk = await aemet_get(client, url, key)
            print(f"  {s} → {e}: {len(chunk)} filas")
            rows.extend(chunk)
    return normalize(rows)


def load_existing(parquet: Path) -> pd.DataFrame | None:
    if not parquet.exists():
        return None
    return duckdb.sql(f"SELECT * FROM '{parquet}'").df()


def save_parquet(df: pd.DataFrame, parquet: Path) -> None:
    con = duckdb.connect()
    con.register("merged", df)
    con.execute(
        f"COPY (SELECT fecha, tmax, tmin FROM merged ORDER BY fecha) "
        f"TO '{parquet}' (FORMAT PARQUET)"
    )


async def update_station(station: str, *, refresh: bool, from_date: date | None,
                         to_date: date | None) -> Path:
    key = api_key()
    parquet = DATA_DIR / f"{station}.parquet"

    existing = None if refresh else load_existing(parquet)

    # Rango a descargar
    if from_date is not None:
        start = from_date
    elif existing is not None and not existing.empty:
        last = pd.to_datetime(existing["fecha"]).max().date()
        start = last + timedelta(days=1)
    else:
        start = date(1920, 1, 1)
    end = to_date or (date.today() - timedelta(days=1))  # ayer, AEMET no suele tener hoy

    if start > end:
        print(f"{station}: caché al día (última fecha {existing['fecha'].max() if existing is not None else '—'}).")
        return parquet

    fresh = await fetch_station_range(station, start, end, key)
    print(f"Recibidas {len(fresh)} filas nuevas")

    def to_ts(df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
        return df

    if existing is not None and not existing.empty and not fresh.empty:
        merged = pd.concat([to_ts(existing), to_ts(fresh)], ignore_index=True)
        merged = merged.drop_duplicates(subset="fecha", keep="last").sort_values("fecha")
    elif existing is not None and fresh.empty:
        merged = to_ts(existing)
    else:
        merged = to_ts(fresh)

    save_parquet(merged, parquet)
    print(f"Parquet → {parquet}  ({len(merged)} filas totales, "
          f"{merged['fecha'].min()} → {merged['fecha'].max()})")
    return parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="3195", help="Indicativo AEMET (default: 3195 Madrid Retiro)")
    ap.add_argument("--from", dest="from_date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    help="Fecha inicio YYYY-MM-DD (sobreescribe el incremental)")
    ap.add_argument("--to", dest="to_date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    help="Fecha fin YYYY-MM-DD (default: ayer)")
    ap.add_argument("--refresh", action="store_true", help="Reescribe desde cero aunque haya caché")
    args = ap.parse_args()

    asyncio.run(update_station(
        args.station, refresh=args.refresh,
        from_date=args.from_date, to_date=args.to_date,
    ))


if __name__ == "__main__":
    main()
