# Agente ETL para imputacion de datos faltantes

Este proyecto implementa un pipeline reproducible para limpiar datos, diagnosticar
patrones de valores faltantes, elegir un metodo de imputacion y generar un reporte
interpretativo. La arquitectura separa el nucleo estadistico deterministico de la
capa opcional de IA: el pipeline puede correr completo sin LLM.

## Objetivo

El agente ayuda a responder tres preguntas:

- Que tan grave y estructurado es el problema de datos faltantes.
- Que metodo de imputacion conviene usar segun el objetivo del analisis.
- Que implican los resultados, supuestos y advertencias para las conclusiones.

## Estructura del proyecto

- `core/ingestion.py`: carga archivos `csv`, `json`, `xlsx`, `sav`, `dta` y
  `parquet`; estandariza valores faltantes y tipos; detecta columnas ID; elimina
  filas sin informacion suficiente; genera perfil JSON.
- `core/pipeline.py`: orquesta la decision entre MICE y regresion estocastica
  usando porcentaje maximo de faltantes, ratio filas/columnas y objetivo
  (`inference` o `prediction`).
- `core/comparison_plots.py`: genera graficos interactivos Plotly antes/despues
  de imputar: boxplot, histograma y QQ-plot embebibles en HTML.
- `core/report_builder.py`: consolida diagnostico, decision, imputacion,
  advertencias, resumen descriptivo y graficos comparativos en un reporte JSON.
- `imputers/base.py`: contrato base de imputadores y errores controlados de
  ejecucion.
- `imputers/mice_imputer.py`: wrapper Python sobre `r_scripts/mice_imputer.R`.
  Ejecuta MICE via `Rscript`, guarda datos imputados y reporte de Rubin.
- `imputers/regresion_imputer.py`: wrapper Python sobre
  `r_scripts/regresion_imputer.R` para imputacion por regresion estocastica.
- `imputers/diagnostico_runner.py`: ejecuta `r_scripts/diagnostico.R`, copia PNGs
  de diagnostico a una carpeta persistente y devuelve el JSON estadistico.
- `r_scripts/diagnostico.R`: diagnostico de faltantes con `md.pattern`, resumenes
  de faltantes, test de Little, EM y graficos de diagnostico.
- `r_scripts/mice_imputer.R`: imputacion MICE, pooling de Rubin, severidad,
  validacion de supuestos y ajuste opcional de distribucion Beta con `--beta_vars`.
- `r_scripts/regresion_imputer.R`: imputacion por regresion estocastica y reporte
  opcional de supuestos sobre residuos.
- `llm/client.py`: clientes intercambiables para Gemini, DeepSeek y Ollama, con
  timeouts, errores explicativos y reintentos en `429`/`503`.
- `llm/explainer.py`: interpreta el reporte consolidado con reglas anti
  alucinacion; sanea datos grandes o irrelevantes antes de enviarlos al LLM; puede
  responder preguntas de seguimiento.
- `scripts/run_pipeline.py`: CLI principal. Ejecuta ingestion, diagnostico,
  imputacion, graficos comparativos, reporte JSON, explicacion IA opcional y
  dashboard HTML.
- `scripts/generate_html_report.py`: construye el dashboard HTML interactivo,
  incluyendo interpretacion IA por seccion, graficos Plotly y formulas KaTeX.
- `assets/katex/`: recursos locales para renderizar notacion matematica en el
  reporte HTML sin depender de CDN.
- `tests/`: pruebas unitarias e integracion real con R/Ollama cuando estan
  disponibles.

## Requisitos

Instala dependencias de Python:

```powershell
python -m pip install -r requirements.txt
```

Tambien necesitas R con los paquetes usados por los scripts en `r_scripts/`,
incluyendo `mice`, `naniar`, `VIM`, `DataExplorer`, `inspectdf`, `dlookr`,
`mvnmle`, `optparse`, `kSamples`, `nortest`, `lmtest` y `MASS`.

## Configuracion de IA

Copia `.env.example` a `.env` y configura el proveedor deseado:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=300
```

Tambien se soportan Gemini y DeepSeek mediante variables de entorno. No guardes
claves reales fuera de `.env`; este archivo esta ignorado por Git.

## Uso principal

Ejecutar el pipeline con IA:

```powershell
python scripts/run_pipeline.py --input "WH2023.xlsx" --goal inference --dataset_name WH2023 --domain_context "Datos del World Happiness Report 2023 con indicadores sociales y economicos por pais."
```

Ejecutar sin IA:

```powershell
python scripts/run_pipeline.py --input "WH2023.xlsx" --goal inference --dataset_name WH2023 --skip_ai
```

Seleccionar variables para pooling MICE:

```powershell
python scripts/run_pipeline.py --input "WH2023.xlsx" --goal inference --dataset_name WH2023 --vars "Ladder,LGDP,Social_support"
```

Ajustar distribucion Beta para variables proporcionales o porcentuales:

```powershell
python scripts/run_pipeline.py --input "CoreHouseholdIndicators.xlsx" --goal inference --dataset_name CoreHouse --vars "Ind_Mobile" --beta_vars "Ind_Mobile"
```

Los resultados quedan en `outputs/<dataset>_<timestamp>/`:

- `datos_limpios.csv`
- `perfil.json`
- `datos_imputados.csv`
- `reporte.json`
- `reporte.html`
- `plots/`

## Decision automatica del metodo

- `goal="inference"`: siempre usa MICE. Si hay muchos faltantes o bajo ratio
  filas/columnas, usa mas imputaciones.
- `goal="prediction"`: con pocos faltantes usa regresion estocastica; con faltantes
  moderados o altos usa MICE.

El reporte registra los criterios usados (`max_pct`, `ratio`, `goal`) para que la
decision sea auditable.

## Reporte HTML

El dashboard incluye:

- interpretacion IA general y notas por seccion;
- exploracion previa de faltantes;
- graficos de diagnostico;
- resumen descriptivo antes/despues;
- decision del metodo;
- resultados de imputacion y severidad;
- ajuste Beta opcional;
- cumplimiento de supuestos;
- advertencias.

## Pruebas

Ejecuta la suite completa:

```powershell
python -m pytest -q
```

Algunos tests de integracion se saltan automaticamente si `Rscript` u Ollama no
estan disponibles.
