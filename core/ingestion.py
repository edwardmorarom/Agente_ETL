"""
ingestion.py

Módulo de ingesta y estandarización de datos.

Responsabilidad única: recibir un archivo en cualquier formato soportado,
inferir el tipo real de cada columna, estandarizar la representación de
los valores faltantes, y entregar SIEMPRE un CSV limpio + un perfil JSON.

Los scripts R del pipeline (diagnostico.R, mice_imputer.R,
regresion_imputer.R) nunca leen el archivo original del usuario. Siempre
leen el CSV que este módulo produce. Así, R no necesita saber si el
usuario subió un .xlsx, un .sav o un .json.

Uso:
    python ingestion.py --input datos.xlsx \
                         --output_csv datos_limpio.csv \
                         --output_profile_json perfil.json

Formatos soportados: .csv, .tsv, .json, .xlsx, .xls, .sav, .dta, .parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# -----------------------------------------------------------------------
# Tokens de texto que representan "faltante" en distintas fuentes de datos.
# Solo se incluyen marcadores de TEXTO, nunca números como -999 o 9999,
# porque esos son sentinelas específicos de cada dataset y estandarizarlos
# a ciegas puede borrar información válida sin que el usuario lo note.
# -----------------------------------------------------------------------
DEFAULT_NA_TOKENS = [
    "NA", "N/A", "n/a", "na",
    "NULL", "null", "Null",
    "None", "none",
    "NaN", "nan",
    "", " ", ".", "..", "?", "-",
]

SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".xlsx", ".xls", ".sav", ".dta", ".parquet",
}


class IngestionError(Exception):
    """Error controlado de ingesta, para diferenciarlo de bugs internos."""


# -----------------------------------------------------------------------
# Carga de archivos
# -----------------------------------------------------------------------
def load_file(
    path: Path,
    extra_na_tokens: list[str] | None = None,
    sheet: str | int | None = None,
) -> pd.DataFrame:
    """Carga cualquier formato soportado a un DataFrame de pandas."""
    if not path.exists():
        raise IngestionError(f"No existe el archivo de entrada: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Formato no soportado: '{ext}'. "
            f"Formatos válidos: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    na_tokens = list(DEFAULT_NA_TOKENS)
    if extra_na_tokens:
        na_tokens.extend(extra_na_tokens)

    try:
        if ext == ".csv":
            df = pd.read_csv(path, na_values=na_tokens, keep_default_na=True)
        elif ext == ".tsv":
            df = pd.read_csv(
                path, sep="\t", na_values=na_tokens, keep_default_na=True
            )
        elif ext == ".json":
            df = _load_json(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(
                path,
                sheet_name=sheet if sheet is not None else 0,
                na_values=na_tokens,
                keep_default_na=True,
            )
        elif ext == ".sav":
            df = _load_spss(path)
        elif ext == ".dta":
            df = pd.read_stata(path, convert_categoricals=False)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:  # pragma: no cover - cubierto por la validación de arriba
            raise IngestionError(f"Formato no manejado: {ext}")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"No fue posible leer '{path.name}' como {ext}: {exc}"
        ) from exc

    if df.empty:
        raise IngestionError("El archivo no contiene filas de datos.")
    if df.shape[1] == 0:
        raise IngestionError("El archivo no contiene columnas.")

    # Aplicar tokens de NA también sobre archivos que no pasan por
    # na_values de pandas (json, sav, dta, parquet).
    df = _apply_na_tokens(df, na_tokens)

    return df


def _load_json(path: Path) -> pd.DataFrame:
    """Soporta tanto una lista de registros como un dict de columnas."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return pd.json_normalize(raw)
    if isinstance(raw, dict):
        # Caso {"columna1": [...], "columna2": [...]}
        if all(isinstance(v, list) for v in raw.values()):
            return pd.DataFrame(raw)
        # Caso de un solo registro anidado -> se normaliza como 1 fila
        return pd.json_normalize([raw])

    raise IngestionError(
        "El JSON debe ser una lista de registros o un objeto de columnas."
    )


def _load_spss(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError as exc:
        raise IngestionError(
            "Falta la librería 'pyreadstat' para leer archivos .sav. "
            "Instálala con: pip install pyreadstat"
        ) from exc

    df, _meta = pyreadstat.read_sav(str(path))
    return df


def _apply_na_tokens(df: pd.DataFrame, na_tokens: list[str]) -> pd.DataFrame:
    """
    Convierte tokens de texto a NaN real, columna por columna.

    No se filtra por dtype == object: dependiendo de la fuente (json,
    sav, dta) pandas puede representar texto con dtypes distintos
    (object clásico, StringDtype, etc.). Se revisa cada valor
    individualmente y solo se actúa sobre los que sí son str.
    """
    token_set = set(na_tokens)

    def _clean(v):
        if isinstance(v, str) and v.strip() in token_set:
            return pd.NA
        return v

    for col in df.columns:
        df[col] = df[col].apply(_clean)
    return df


# -----------------------------------------------------------------------
# Inferencia de tipos por columna
# -----------------------------------------------------------------------
def infer_column_type(series: pd.Series) -> str:
    """
    Devuelve uno de: 'numeric', 'integer', 'boolean', 'datetime',
    'categorical', 'text'.

    La inferencia se hace sobre los valores NO faltantes, para no dejar
    que la cantidad de NA afecte la detección del tipo real.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "text"  # no hay evidencia suficiente; se marca por defecto

    # Booleanos explícitos: agrupa por equivalencia semántica
    # ("true"/"1" -> True, "false"/"0" -> False) sin importar cuántas
    # representaciones distintas de cada estado aparezcan.
    true_tokens = {"true", "1", "verdadero", "si", "sí", "yes"}
    false_tokens = {"false", "0", "falso", "no"}
    unique_vals = set(non_null.astype(str).str.lower().str.strip().unique())
    if unique_vals and unique_vals.issubset(true_tokens | false_tokens):
        return "boolean"

    # Numérico (ya viene numérico desde pandas, o convertible)
    if pd.api.types.is_numeric_dtype(non_null):
        if (non_null.dropna() % 1 == 0).all():
            return "integer"
        return "numeric"

    coerced = pd.to_numeric(non_null, errors="coerce")
    if coerced.notna().mean() > 0.95:
        if (coerced.dropna() % 1 == 0).all():
            return "integer"
        return "numeric"

    # Fechas (se ignora el warning de formato mixto; solo se usa para
    # decidir el tipo, la conversión real y definitiva ocurre en
    # standardize())
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coerced_dates = pd.to_datetime(non_null, errors="coerce")
    if coerced_dates.notna().mean() > 0.95:
        return "datetime"

    # Categórico vs texto libre: pocas categorías repetidas -> categórico
    n_unique = non_null.nunique()
    if n_unique <= max(20, int(len(non_null) * 0.05)):
        return "categorical"

    return "text"


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Genera el perfil de composición de la base, columna por columna."""
    n_rows = len(df)
    columns_profile = {}

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        inferred = infer_column_type(series)

        entry: dict[str, Any] = {
            "inferred_type": inferred,
            "pandas_dtype": str(series.dtype),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / n_rows * 100, 2) if n_rows else 0.0,
            "n_unique": int(series.nunique(dropna=True)),
        }

        if inferred in ("numeric", "integer"):
            numeric_series = pd.to_numeric(series, errors="coerce")
            entry["min"] = _safe_float(numeric_series.min())
            entry["max"] = _safe_float(numeric_series.max())
            entry["mean"] = _safe_float(numeric_series.mean())
        elif inferred == "categorical":
            entry["categories"] = (
                series.dropna().astype(str).value_counts().head(10).to_dict()
            )

        columns_profile[col] = entry

    return {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "total_missing_cells": int(df.isna().sum().sum()),
        "pct_missing_overall": round(
            df.isna().sum().sum() / (n_rows * df.shape[1]) * 100, 2
        ) if n_rows and df.shape[1] else 0.0,
        "columns": columns_profile,
    }


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


# -----------------------------------------------------------------------
# Estandarización final antes de escribir el CSV
# -----------------------------------------------------------------------
def standardize(df: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    """
    Convierte cada columna al tipo pandas correspondiente según lo
    inferido, para que el CSV de salida ya llegue "tipado" y R no tenga
    que volver a adivinar (read.csv interpretará cada columna de forma
    consistente con lo que este perfil dice).
    """
    result = df.copy()

    for col, meta in profile["columns"].items():
        inferred = meta["inferred_type"]
        try:
            if inferred in ("numeric", "integer"):
                result[col] = pd.to_numeric(result[col], errors="coerce")
            elif inferred == "boolean":
                result[col] = (
                    result[col]
                    .astype(str)
                    .str.lower()
                    .map({"true": True, "1": True, "false": False, "0": False})
                )
            elif inferred == "datetime":
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result[col] = pd.to_datetime(result[col], errors="coerce")
            # categorical y text se dejan como están (string), R los lee
            # como character/factor sin problema.
        except Exception:
            # Si algo falla en la conversión, se deja la columna original
            # en vez de romper todo el pipeline por una sola columna.
            continue

    return result


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingesta y estandarización de bases de datos "
        "antes de pasarlas al pipeline de imputación."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_profile_json", required=True, type=Path)
    parser.add_argument(
        "--extra_na_values",
        default=None,
        help="Tokens adicionales de NA separados por coma, ej: '999,-1'",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Nombre u número de hoja para archivos Excel (default: la primera)",
    )
    args = parser.parse_args(argv)

    extra_tokens = None
    if args.extra_na_values:
        extra_tokens = [t.strip() for t in args.extra_na_values.split(",") if t.strip()]

    sheet = args.sheet
    if sheet is not None and sheet.isdigit():
        sheet = int(sheet)

    try:
        df = load_file(args.input, extra_na_tokens=extra_tokens, sheet=sheet)
        profile = build_profile(df)
        df_standardized = standardize(df, profile)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_profile_json.parent.mkdir(parents=True, exist_ok=True)

        df_standardized.to_csv(args.output_csv, index=False)
        with open(args.output_profile_json, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    except IngestionError as exc:
        print(f"Error de ingesta: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.input.name} -> {args.output_csv.name} "
          f"({profile['n_rows']} filas, {profile['n_columns']} columnas, "
          f"{profile['pct_missing_overall']}% faltantes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
