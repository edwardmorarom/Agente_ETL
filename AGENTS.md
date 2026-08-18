# AGENTS.md

Reglas de arquitectura del proyecto. Cualquier módulo nuevo debe
seguir esto sin excepción. Si algo no está claro, preguntar antes de
improvisar.

## Estado actual (ya existe, NO modificar la lógica interna)

- `core/ingestion.py` — lee cualquier formato (csv, json, xlsx, sav,
  dta, parquet), estandariza NA y tipos, entrega CSV limpio + perfil.
- `r_scripts/diagnostico.R` — diagnóstico de faltantes (md.pattern,
  test de Little manual y vía naniar, EM manual y vía mvnmle).
  Recibe --input, --output_json, --output_plots_dir.
- `r_scripts/mice_imputer.R` — imputación MICE con pooling de Rubin
  (univariado o multivariado según --vars). Recibe --input,
  --output_data, --output_json, --vars, --m, --maxit, --seed.
- `r_scripts/regresion_imputer.R` — imputación por media estocástica
  o regresión (determinística/estocástica). Recibe --input, --output,
  --method, --models, --seed.
- `r_scripts/raw/` — los 6 scripts R originales, de respaldo. NUNCA
  se ejecutan desde el pipeline, solo quedan como referencia.

## Arquitectura general

DOS capas desacopladas:

1. **Core determinístico** (`core/`, `imputers/`) — Python que
   orquesta, más R que hace el cálculo estadístico pesado vía
   subprocess. Debe poder correr completo SIN el LLM.
2. **Capa de razonamiento** (`llm/`) — llama a un LLM (Gemini o
   DeepSeek, intercambiable) solo para interpretar resultados o
   generar explicaciones. Nunca es indispensable para que el
   pipeline funcione.

## Patrón obligatorio: Python llama a R, nunca lo reemplaza

Los scripts en `r_scripts/` NO se reescriben en Python. Cada
imputador en `imputers/` es una clase Python que hereda de
`BaseImputer` y por debajo ejecuta el script R correspondiente vía
`subprocess`, usando archivos temporales para pasar datos.

```python
class BaseImputer(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
```

- Firma obligatoria: `fit_transform(df: pd.DataFrame) -> pd.DataFrame`
- No inventar nombres alternativos (`.run()`, `.process()`, etc.)
- Todo imputador vive en `/imputers` y hereda de `BaseImputer`
  (`/imputers/base.py`)

## Convenciones de código

- Type hints obligatorios en todas las funciones Python.
- Nombres de archivo y funciones: `snake_case`, sin tildes ni espacios.
- Cada módulo nuevo en `/imputers` requiere su test en `/tests`.
- Sin comentarios redundantes que solo repitan lo que el código dice.

## Manejo de API keys y secretos

- Nunca hardcodear API keys en código ni en la imagen de Docker.
- Siempre desde variables de entorno (`os.environ`).
- `.env.example` lista las variables necesarias SIN valores reales.

## Stack técnico

- Python: pandas, subprocess (built-in).
- R: mice, naniar, VIM, DataExplorer, inspectdf, dlookr, mvnmle,
  optparse (ya están fijos en los 3 scripts de `r_scripts/`).
- LLM: API de Gemini o DeepSeek (configurable por variable de entorno).

## Docker

- Una sola imagen con Python Y R instalados (no se usa Ollama).
- No reescribir en Python nada que ya funcione en R.

## Al agregar un módulo nuevo

1. Si es un imputador: heredar de `BaseImputer`, envolver el script R
   correspondiente vía subprocess, agregar su test en `/tests`.
2. Si es diagnóstico: seguir el mismo patrón, envolviendo
   `r_scripts/diagnostico.R`.
3. No introducir una librería nueva para resolver algo que ya se
   resuelve con el stack existente, salvo que se justifique aquí.
