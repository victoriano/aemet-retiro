# aemet-retiro

Análisis de la serie climatológica diaria de **Madrid Retiro** (estación AEMET
3195) y de cualquier otra estación española, usando la **API oficial de
AEMET OpenData** y DuckDB. Genera heatmaps día×año, versión filtrada de días
de entretiempo (👕) y bar chart de evolución anual con tendencia.

## Estructura

```
aemet_fetch.py              # descarga/actualiza AEMET API → data/{estacion}.parquet
dias_primaverales.py        # fallback histórico desde HuggingFace (datania/aemet)
heatmap_primaverales.py     # genera los 3 charts (--station <id>)
data/{estacion}.parquet     # caché por estación (fecha, tmax, tmin)
charts/{estacion}_*.png     # charts de estaciones != 3195
```

Para Retiro (estación por defecto) los PNGs se guardan en la raíz
(`heatmap_primaverales.png`, `heatmap_solo_primaverales.png`, `barras_primaverales.png`).

## Uso

### 1. Conseguir la API key de AEMET

Registro gratuito en https://opendata.aemet.es/centrodedescargas/altaUsuario.
Llega por mail un JWT. Guárdala en `~/.config/aemet/api_key` (chmod 600) o
exporta `AEMET_API_KEY=…`.

### 2. Descargar / actualizar datos

```bash
# Update incremental de Retiro hasta ayer
uv run aemet_fetch.py

# Otra estación: Barcelona (0076)
uv run aemet_fetch.py --station 0076

# Rango concreto (para rellenar huecos)
uv run aemet_fetch.py --station 3195 --from 2025-01-01 --to 2025-12-31

# Rebajar caché
uv run aemet_fetch.py --station 3195 --refresh
```

La API permite 6 meses por petición y ~50 req/min, así que descargar un
backfill histórico de 50 años tarda ~2-3 minutos. El fetch incremental (1-7
días desde la última fila cacheada) es casi instantáneo.

### 3. Generar charts

```bash
uv run heatmap_primaverales.py                     # los 3 charts (Retiro)
uv run heatmap_primaverales.py --station 0076      # Barcelona
uv run heatmap_primaverales.py --bars              # solo bar chart
uv run heatmap_primaverales.py --only-entretiempo  # solo heatmap verde
uv run heatmap_primaverales.py --full              # solo heatmap completo
```

## Categorías térmicas

| Emoji | Nombre       | Rango                          |
|-------|--------------|--------------------------------|
| 🧣    | Muy frío     | Tmax < 10°C                    |
| 🧥    | Frío         | 10 ≤ Tmax < 16°C               |
| 🧶    | Fresco       | 16 ≤ Tmax < 20°C               |
| 👕    | Entretiempo  | 20 ≤ Tmax ≤ 27°C · Tmin ≥ 10°C |
| 🩳    | Cálido       | 27 < Tmax ≤ 32°C               |
| 🥵    | Caluroso     | 32 < Tmax ≤ 37°C               |
| 🩱    | Sofocante    | Tmax > 37°C                    |

## Estaciones conocidas

| Código | Nombre                |
|--------|-----------------------|
| 3195   | Madrid Retiro         |
| 3129   | Madrid Barajas        |
| 0076   | Barcelona El Prat     |
| 5783   | Sevilla Aeropuerto    |
| 1024E  | Bilbao Aeropuerto     |
| 8414A  | Valencia Aeropuerto   |
| 6155A  | Málaga Aeropuerto     |

Inventario completo: endpoint `/api/valores/climatologicos/inventarioestaciones/todasestaciones`.

## Dependencias

Python 3.11+ con [`uv`](https://github.com/astral-sh/uv). Deps inline PEP 723:
`duckdb`, `httpx`, `pandas`, `matplotlib`, `numpy`, `pillow`.

## Fuentes

- **API**: https://opendata.aemet.es (docs: https://opendata.aemet.es/dist/index.html)
- **Fallback histórico**: https://huggingface.co/datasets/datania/aemet
  (sin actualización desde 2025, útil solo para backfill puntual)
- Estación 3195: Madrid, Retiro (40.41°N, 3.68°W, 667 m)
