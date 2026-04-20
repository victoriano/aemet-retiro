# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "httpx[http2]"]
# ///
"""
Días "primaverales" en Madrid Retiro (estación AEMET 3195) por año.

Definición configurable por CLI:
    --tmax-min    temperatura máxima mínima (default 20)
    --tmax-max    temperatura máxima máxima (default 25)
    --tmin-min    temperatura mínima mínima (default 14)

Dataset: https://huggingface.co/datasets/datania/aemet  (JSON diarios por estación)
Se descargan en remoto solo los JSON y se filtra la estación 3195. La primera
ejecución cachea los datos crudos en `retiro_raw.parquet`; en ejecuciones
posteriores con otros umbrales no se vuelve a bajar nada.
"""
import argparse
import asyncio
import duckdb
import json
import os
import subprocess
import time
from pathlib import Path
import httpx

OUT_DIR = Path(__file__).parent
RAW_PARQUET = OUT_DIR / "retiro_raw.parquet"
CSV_OUT = OUT_DIR / "dias_primaverales_retiro.csv"
CHART_OUT = OUT_DIR / "dias_primaverales_retiro.png"

STATION = "3195"

REPO = "datania/aemet"
ROOT = "valores-climatologicos"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{ROOT}"

CONCURRENCY = 12
MAX_RETRIES = 5


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else None


async def fetch_json(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> list[dict]:
    delay = 1.0
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                r = await client.get(url, timeout=30.0)
                if r.status_code == 429:
                    retry = float(r.headers.get("retry-after", delay))
                    await asyncio.sleep(retry)
                    delay = min(delay * 2, 30)
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  ! fallo {url}: {e}")
                    return []
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
    return []


async def list_year_files(client: httpx.AsyncClient, year: int) -> list[str]:
    r = await client.get(f"{API}/{year}?recursive=true", timeout=30.0)
    r.raise_for_status()
    return [
        f"{RESOLVE}/{x['path']}"
        for x in r.json()
        if x["type"] == "file" and x["path"].endswith(".json")
    ]


async def fetch_station_rows() -> list[dict]:
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, http2=True) as client:
        r = await client.get(API, timeout=30.0)
        r.raise_for_status()
        years = sorted(int(x["path"].split("/")[-1]) for x in r.json() if x["type"] == "directory")
        print(f"Años disponibles: {years[0]}–{years[-1]} ({len(years)})")
        sem = asyncio.Semaphore(CONCURRENCY)
        rows: list[dict] = []
        t0 = time.time()
        for y in years:
            urls = await list_year_files(client, y)
            if not urls:
                continue
            tasks = [fetch_json(client, sem, u) for u in urls]
            results = await asyncio.gather(*tasks)
            year_rows = [rec for daily in results for rec in daily if rec.get("indicativo") == STATION]
            rows.extend(year_rows)
            elapsed = time.time() - t0
            print(f"  {y}: {len(urls):3d} ficheros, {len(year_rows):3d} filas de Retiro  [{elapsed:5.1f}s]")
        return rows


def build_raw_cache(con: duckdb.DuckDBPyConnection) -> None:
    """Descarga (1 sola vez) los datos crudos de Retiro y los guarda en parquet."""
    rows = asyncio.run(fetch_station_rows())
    print(f"\nTotal filas Retiro recolectadas: {len(rows)}")

    def parse(v):
        if not v:
            return None
        try:
            return float(v.replace(",", "."))
        except (ValueError, AttributeError):
            return None

    records = [(r["fecha"], parse(r.get("tmax")), parse(r.get("tmin"))) for r in rows]
    con.execute("CREATE OR REPLACE TABLE retiro(fecha DATE, tmax DOUBLE, tmin DOUBLE);")
    con.executemany("INSERT INTO retiro VALUES (?, ?, ?)", records)
    con.execute(f"COPY retiro TO '{RAW_PARQUET}' (FORMAT PARQUET)")
    print(f"Parquet crudo cacheado → {RAW_PARQUET}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax-min", type=float, default=20.0)
    ap.add_argument("--tmax-max", type=float, default=25.0)
    ap.add_argument("--tmin-min", type=float, default=14.0)
    ap.add_argument("--refetch", action="store_true", help="Forzar re-descarga aunque exista caché")
    args = ap.parse_args()

    con = duckdb.connect()

    if args.refetch or not RAW_PARQUET.exists():
        build_raw_cache(con)
    else:
        print(f"Usando caché: {RAW_PARQUET}")
        con.execute(f"CREATE OR REPLACE TABLE retiro AS SELECT * FROM '{RAW_PARQUET}'")

    label = f"{args.tmax_min:g}°C ≤ Tmax ≤ {args.tmax_max:g}°C · Tmin ≥ {args.tmin_min:g}°C"
    print(f"\nCriterio: {label}")

    result = con.execute("""
        SELECT
            EXTRACT(YEAR FROM fecha)::INTEGER AS anio,
            COUNT(*) FILTER (WHERE tmax IS NOT NULL AND tmin IS NOT NULL) AS dias_con_datos,
            COUNT(*) FILTER (
                WHERE tmax BETWEEN ? AND ?
                  AND tmin >= ?
            ) AS dias_primaverales
        FROM retiro
        GROUP BY anio
        ORDER BY anio
    """, [args.tmax_min, args.tmax_max, args.tmin_min]).fetchall()

    with open(CSV_OUT, "w") as f:
        f.write("anio,dias_con_datos,dias_primaverales\n")
        for anio, dias, prim in result:
            f.write(f"{anio},{dias},{prim}\n")
    print(f"CSV → {CSV_OUT}")

    if result:
        top = sorted(result, key=lambda r: r[2], reverse=True)[:5]
        print("\nTop 5 años con más días primaverales:")
        for anio, dias, prim in top:
            print(f"  {anio}: {prim} días (de {dias} observados)")

    filtered = [(a, p) for a, d, p in result if d >= 300]
    chart_data = {"labels": [str(a) for a, _ in filtered], "values": [p for _, p in filtered]}
    chart_script = "/Users/victoriano/.claude/skills/charts/scripts/chart.py"
    cmd = [
        "uv", "run", chart_script,
        "--type", "bar",
        "--title", f"Días primaverales al año en Madrid Retiro ({label})",
        "--subtitle", f"Fuente: AEMET · Estación {STATION} · años con ≥300 días observados",
        "--data", json.dumps(chart_data),
        "--xlabel", "Año",
        "--ylabel", "Días",
        "--output", str(CHART_OUT),
        "--width", "18",
        "--height", "7",
        "--color-mode", "gradient",
    ]
    print("\nGenerando gráfico…")
    subprocess.run(cmd, check=True)
    print(f"PNG → {CHART_OUT}")


if __name__ == "__main__":
    main()
