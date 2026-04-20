# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "matplotlib", "numpy", "pillow"]
# ///
"""
Heatmap de clima diario en Madrid Retiro + bar chart de evolución.

Outputs (siempre):
  heatmap_primaverales.png        todos los días coloreados por sensación térmica
  heatmap_solo_primaverales.png   solo los días de entretiempo (resto atenuado)
  barras_primaverales.png         total de días de entretiempo por año

Flags:
  --only-entretiempo    genera únicamente el heatmap de sólo-entretiempo
  --full                genera únicamente el heatmap completo
  --bars                genera únicamente el bar chart
  (sin flags: genera los 3)

Categorías:
    🧣 Muy frío      Tmax < 10°C
    🧥 Frío          10 ≤ Tmax < 16°C
    🧶 Fresco        16 ≤ Tmax < 20°C  o  Tmin < 10 con Tmax moderada
    👕 Entretiempo   20 ≤ Tmax ≤ 27  y  Tmin ≥ 10
    🩳 Cálido        27 < Tmax ≤ 32
    🥵 Caluroso      32 < Tmax ≤ 37
    🩱 Sofocante     Tmax > 37
"""
import argparse
import io
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Nombres por defecto (estación 3195 Retiro) se mantienen en raíz para compat.
LEGACY_STATION = "3195"
LEGACY_OUT_FULL = ROOT / "heatmap_primaverales.png"
LEGACY_OUT_ONLY = ROOT / "heatmap_solo_primaverales.png"
LEGACY_OUT_BARS = ROOT / "barras_primaverales.png"

STATION_NAMES = {
    "3195": "Madrid Retiro",
    "3129": "Madrid Barajas",
    "0076": "Barcelona El Prat",
    "5783": "Sevilla Aeropuerto",
    "1024E": "Bilbao Aeropuerto",
    "8414A": "Valencia Aeropuerto",
    "6155A": "Málaga Aeropuerto",
}


def raw_parquet_for(station: str) -> Path:
    cand = DATA_DIR / f"{station}.parquet"
    if cand.exists():
        return cand
    # Backward compat: antiguo retiro_raw.parquet en raíz
    legacy = ROOT / "retiro_raw.parquet"
    if station == LEGACY_STATION and legacy.exists():
        return legacy
    raise FileNotFoundError(f"No hay parquet para la estación {station} ({cand})")


def outputs_for(station: str) -> tuple[Path, Path, Path]:
    if station == LEGACY_STATION:
        return LEGACY_OUT_FULL, LEGACY_OUT_ONLY, LEGACY_OUT_BARS
    return (
        CHARTS_DIR / f"{station}_heatmap.png",
        CHARTS_DIR / f"{station}_solo_entretiempo.png",
        CHARTS_DIR / f"{station}_barras.png",
    )

CATS = [
    ("Sin datos",   "#eeeeee", "",   ""),
    ("Muy frío",    "#3a0ca3", "🧣", "Tmax <10°"),
    ("Frío",        "#4361ee", "🧥", "10–16°"),
    ("Fresco",      "#8ecae6", "🧶", "16–20°"),
    ("Entretiempo", "#2e8b57", "👕", "20–27° · Tmin≥10°"),
    ("Cálido",      "#ffd60a", "🩳", "27–32°"),
    ("Caluroso",    "#f77f00", "🥵", "32–37°"),
    ("Sofocante",   "#d00000", "🩱", ">37°"),
]
ENT_IDX = 4

EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_SIZE = 160  # único tamaño bitmap disponible en macOS


CLASSIFY_SQL = """
CASE
    WHEN tmax IS NULL AND tmin IS NULL            THEN 0
    WHEN tmax < 10                                THEN 1
    WHEN tmax < 16                                THEN 2
    WHEN tmax < 20                                THEN 3
    WHEN tmax BETWEEN 20 AND 27 AND tmin >= 10    THEN 4
    WHEN tmax BETWEEN 20 AND 27 AND tmin <  10    THEN 3
    WHEN tmax <= 32                               THEN 5
    WHEN tmax <= 37                               THEN 6
    ELSE                                               7
END
"""


def load_grid(parquet: Path):
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT EXTRACT(YEAR FROM fecha)::INT AS anio,
               EXTRACT(DOY  FROM fecha)::INT AS doy,
               {CLASSIFY_SQL} AS cat
        FROM '{parquet}'
        ORDER BY anio, doy
    """).fetchall()
    years = sorted({r[0] for r in rows})
    y_idx = {y: i for i, y in enumerate(years)}
    grid = np.zeros((len(years), 366), dtype=int)
    for anio, doy, cat in rows:
        if 1 <= doy <= 366:
            grid[y_idx[anio], doy - 1] = cat
    return grid, years


def load_median_temp(parquet: Path) -> dict[int, float]:
    """Temperatura media anual mediana: median(tmax+tmin)/2 por año."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT EXTRACT(YEAR FROM fecha)::INT AS anio,
               MEDIAN((tmax + tmin) / 2.0) AS tmed
        FROM '{parquet}'
        WHERE tmax IS NOT NULL AND tmin IS NOT NULL
        GROUP BY anio
        ORDER BY anio
    """).fetchall()
    return {y: float(t) for y, t in rows}


def emoji_image(ch: str, target_h: int, font) -> Image.Image:
    if not ch:
        return Image.new("RGBA", (0, target_h), (0, 0, 0, 0))
    em = Image.new("RGBA", (EMOJI_SIZE + 40, EMOJI_SIZE + 40), (0, 0, 0, 0))
    ImageDraw.Draw(em).text((0, 0), ch, font=font, embedded_color=True)
    bbox = em.getbbox()
    if bbox:
        em = em.crop(bbox)
    ratio = target_h / em.height
    return em.resize((max(1, int(em.width * ratio)), target_h), Image.LANCZOS)


def build_legend(width: int, cats: list, scale: float = 1.0) -> Image.Image:
    """Leyenda horizontal grande: swatch + emoji + nombre + rango."""
    pad_y = int(26 * scale)
    swatch_w = int(44 * scale)
    swatch_h = int(44 * scale)
    gap_sw = int(14 * scale)
    gap_em = int(14 * scale)
    col_gap = int(46 * scale)
    emoji_h = int(72 * scale)
    name_size = int(30 * scale)
    range_size = int(22 * scale)
    name_h = int(name_size * 1.25)
    range_h = int(range_size * 1.25)
    block_h = max(emoji_h, name_h + range_h + int(6 * scale))

    name_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", name_size)
    range_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", range_size)
    emoji_font = ImageFont.truetype(EMOJI_FONT, EMOJI_SIZE)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    items = []
    for name, color, emoji, rng in cats:
        em = emoji_image(emoji, emoji_h, emoji_font)
        label_w = max(
            probe.textlength(name, font=name_font),
            probe.textlength(rng, font=range_font),
        )
        w = swatch_w + gap_sw + em.width + (gap_em if em.width else 0) + int(label_w)
        items.append((name, color, em, rng, w))

    total_w = sum(w for *_, w in items) + col_gap * (len(items) - 1)
    # Si se pasa del ancho disponible, reducimos el gap entre columnas
    if total_w > width - 40:
        col_gap = max(18, (width - 40 - sum(w for *_, w in items)) // max(1, len(items) - 1))
        total_w = sum(w for *_, w in items) + col_gap * (len(items) - 1)
    x0 = max(10, (width - total_w) // 2)

    canvas = Image.new("RGBA", (width, block_h + pad_y * 2), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    top = pad_y
    x = x0
    for name, color, em, rng, w in items:
        sy = top + (block_h - swatch_h) // 2
        draw.rectangle(
            [x, sy, x + swatch_w, sy + swatch_h],
            fill=color,
            outline="#999" if name == "Sin datos" else None,
            width=2,
        )
        cx = x + swatch_w + gap_sw
        if em.width:
            canvas.alpha_composite(em, (cx, top + (block_h - em.height) // 2))
            cx += em.width + gap_em
        draw.text((cx, top + int(4 * scale)), name, font=name_font, fill="#1a1a1a")
        if rng:
            draw.text((cx, top + int(4 * scale) + name_h), rng, font=range_font, fill="#555")
        x += w + col_gap
    return canvas


def render_heatmap(grid: np.ndarray, years: list[int], title: str, *, only_ent: bool) -> Image.Image:
    if only_ent:
        display = np.where(grid == ENT_IDX, 1, 0)
        cmap = mcolors.ListedColormap(["#f2f2f2", CATS[ENT_IDX][1]])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    else:
        display = grid
        cmap = mcolors.ListedColormap([c for _, c, *_ in CATS])
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(CATS) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(22, 0.32 * len(years) + 2.5))
    ax.imshow(display, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=10)
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    ax.set_xticks([d - 1 for d in starts])
    ax.set_xticklabels(labels, fontsize=13)
    for d in starts[1:]:
        ax.axvline(d - 1, color="#ffffff", lw=0.5, alpha=0.6)
    ax.set_title(title, fontsize=18, pad=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def compose(heatmap: Image.Image, cats: list, path: Path, scale: float = 1.6) -> None:
    legend = build_legend(heatmap.width, cats, scale=scale)
    final = Image.new("RGBA", (heatmap.width, heatmap.height + legend.height), (255, 255, 255, 255))
    final.paste(heatmap, (0, 0))
    final.paste(legend, (0, heatmap.height))
    final.convert("RGB").save(path, "PNG", optimize=True)
    print(f"PNG → {path}")


def render_bars(grid: np.ndarray, years: list[int], parquet: Path,
                 out_path: Path, station_name: str, start_year: int = 1975) -> None:
    """Bar chart: días de entretiempo por año, coloreado por temperatura mediana anual."""
    ent_per_year = (grid == ENT_IDX).sum(axis=1)
    obs_per_year = (grid > 0).sum(axis=1)
    years_arr = np.array(years)
    mask = (obs_per_year >= 300) & (years_arr >= start_year)
    years_f = years_arr[mask]
    ent_f = ent_per_year[mask]
    obs_f = obs_per_year[mask]

    med_temp = load_median_temp(parquet)
    temps_f = np.array([med_temp.get(int(y), np.nan) for y in years_f])

    window = 5
    if len(ent_f) >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(ent_f, kernel, mode="same")
    else:
        rolling = ent_f
    coef = np.polyfit(years_f, ent_f, 1) if len(years_f) >= 2 else None

    fig, ax = plt.subplots(figsize=(24, 11))
    fig.subplots_adjust(left=0.06, right=0.93, top=0.86, bottom=0.14)

    # Color por temperatura mediana (azul frío → rojo cálido)
    cmap = plt.get_cmap("RdYlBu_r")
    t_min, t_max = np.nanmin(temps_f), np.nanmax(temps_f)
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
    colors = [cmap(norm(t)) for t in temps_f]

    bars = ax.bar(years_f, ent_f, color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    ax.plot(years_f, rolling, color="#222", lw=2.2,
            label=f"Media móvil ({window} años)", zorder=4)
    if coef is not None:
        trend = np.poly1d(coef)(years_f)
        slope_per_decade = coef[0] * 10
        ax.plot(years_f, trend, color="#d00000", lw=2, ls="--",
                label=f"Tendencia lineal ({slope_per_decade:+.1f} días/década)", zorder=4)

    # Etiqueta por barra: "YYYY · Nd" vertical encima de la barra.
    # Añade "*" cuando la cobertura es parcial (<330 días).
    for bar, y, d, t, obs in zip(bars, years_f, ent_f, temps_f, obs_f):
        x = bar.get_x() + bar.get_width() / 2
        suffix = "*" if obs < 330 else ""
        ax.text(
            x, d + 1.5,
            f"{int(y)} · {int(d)}d{suffix}",
            rotation=90, ha="center", va="bottom",
            fontsize=9, color="#333", zorder=5,
        )

    ax.set_title(
        f"Días de entretiempo por año en {station_name} · {years_f.min()}-{years_f.max()} "
        "(20°C ≤ Tmax ≤ 27°C · Tmin ≥ 10°C)",
        fontsize=16, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Año", fontsize=13)
    ax.set_ylabel("Días de entretiempo", fontsize=13)
    partial_years = [int(y) for y, o in zip(years_f, obs_f) if o < 330]
    if partial_years:
        ax.text(
            0.99, 0.97,
            f"* año con cobertura parcial ({', '.join(map(str, partial_years))})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#777", style="italic",
        )
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    ax.set_xlim(years_f.min() - 0.8, years_f.max() + 0.8)
    # Margen arriba para las etiquetas verticales
    ax.set_ylim(0, max(ent_f) * 1.30)

    # Colorbar con la temperatura mediana
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.015, fraction=0.025, aspect=30)
    cbar.set_label("Temperatura mediana anual (°C)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.8,
                facecolor="white")
    plt.close(fig)
    print(f"PNG → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="3195",
                    help="Indicativo AEMET (default: 3195 Madrid Retiro)")
    ap.add_argument("--only-entretiempo", "--only-primaveral", dest="only_ent",
                    action="store_true", help="Solo heatmap de días de entretiempo")
    ap.add_argument("--full", action="store_true", help="Solo heatmap completo")
    ap.add_argument("--bars", action="store_true", help="Solo bar chart")
    args = ap.parse_args()

    do_full = args.full or not (args.only_ent or args.bars)
    do_only = args.only_ent or not (args.full or args.bars)
    do_bars = args.bars or not (args.full or args.only_ent)

    parquet = raw_parquet_for(args.station)
    out_full, out_only, out_bars = outputs_for(args.station)
    station_name = STATION_NAMES.get(args.station, f"estación {args.station}")

    grid, years = load_grid(parquet)

    if do_full:
        hm = render_heatmap(grid, years,
                            title=f"Clima diario en {station_name} (AEMET {args.station})",
                            only_ent=False)
        compose(hm, CATS, out_full)

    if do_only:
        hm = render_heatmap(grid, years,
                            title=f"Días de entretiempo en {station_name} (20°C ≤ Tmax ≤ 27°C · Tmin ≥ 10°C)",
                            only_ent=True)
        ent_cats = [CATS[ENT_IDX], ("Otros días", "#f2f2f2", "", "Resto del año")]
        compose(hm, ent_cats, out_only)

    if do_bars:
        render_bars(grid, years, parquet, out_bars, station_name)


if __name__ == "__main__":
    main()
