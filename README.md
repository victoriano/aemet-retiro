# aemet-retiro

Scripts que analizan la serie diaria de la estación AEMET **Madrid Retiro (3195)**
usando el dataset [`datania/aemet`](https://huggingface.co/datasets/datania/aemet)
en HuggingFace. Los datos se leen en remoto (sin clonar el dataset completo:
~20k JSON diarios con cientos de estaciones cada uno), se filtra sólo Retiro y
se agrega con DuckDB.

## Qué hay

- `dias_primaverales.py` — descarga async los JSON de Retiro desde HF, los vuelca
  a `retiro_raw.parquet` (~19k filas, cacheado en disco) y calcula días
  "primaverales" con umbrales configurables:
  ```bash
  uv run dias_primaverales.py --tmax-min 20 --tmax-max 27 --tmin-min 10
  uv run dias_primaverales.py --refetch  # refresca el caché
  ```
- `heatmap_primaverales.py` — usa el parquet local y genera 3 charts:
  - `heatmap_primaverales.png` — heatmap día×año coloreado por categoría
    térmica (muy frío → sofocante), con emojis de ropa.
  - `heatmap_solo_primaverales.png` — sólo los días de "entretiempo".
  - `barras_primaverales.png` — total anual de días de entretiempo con media
    móvil 5 años, tendencia lineal y color por temperatura mediana anual.
  ```bash
  uv run heatmap_primaverales.py           # los 3
  uv run heatmap_primaverales.py --bars    # solo barras
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

## Cobertura del dataset

`datania/aemet` sólo publica **1920-1922** y **1974-presente** para Retiro. El
bar chart se limita a 1975-2024 para mostrar una serie continua sin huecos. Años
con <300 días observados se excluyen automáticamente.

## Dependencias

Python 3.11+ con `uv`. Las dependencias se declaran inline con PEP 723; `uv run`
las resuelve automáticamente (`duckdb`, `httpx[http2]`, `matplotlib`, `numpy`,
`pillow`).

## Fuente

- Dataset: https://huggingface.co/datasets/datania/aemet (por @datania)
- Estación AEMET 3195: Madrid, Retiro (40.41°N, 3.68°W, 667 m)
