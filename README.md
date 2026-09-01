# Agente de Imputacion de Datos Faltantes

Este proyecto es un agente estadistico que limpia bases de datos, diagnostica
patrones de valores faltantes, elige un metodo de imputacion, ejecuta el calculo
con R y genera un dashboard HTML offline con interpretacion asistida por IA.

El nucleo del pipeline funciona sin IA. La capa LLM se usa para explicar los
resultados en lenguaje claro y responder preguntas de seguimiento.

## Requisitos

- Python 3.12.
- R instalado y disponible en terminal como `Rscript`.
- Paquetes principales de R: `mice`, `naniar`, `VIM`, `kSamples`, `nortest`,
  `lmtest`, `MASS` y `mvnmle`.
- Una opcion de IA:
  - Ollama local.
  - API key de Gemini.
  - API key de DeepSeek.

## Instalacion rapida

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Luego abre `.env` y configura el proveedor de IA que quieras usar. Si vas a
correr sin IA, puedes usar `--skip_ai`.

## Uso basico - modo conversacional recomendado

Comando minimo:

```powershell
python -m scripts.run_pipeline --input tu_archivo.xlsx
```

El agente te preguntara lo demas en lenguaje simple: de que se tratan los datos,
si buscas conclusiones estadisticas confiables o solo completar el archivo, y si
aplica algun analisis Beta. No necesitas conocer de antemano todos los parametros
estadisticos para empezar.

## Uso avanzado - argumentos directos

Argumentos disponibles:

- `--input`: ruta del archivo de entrada. Obligatorio.
- `--goal`: objetivo del analisis. Usa `inference` para conclusiones
  estadisticas o `prediction` para completar datos rapidamente.
- `--domain_context`: descripcion breve del dataset para mejorar la
  interpretacion IA.
- `--vars`: columnas separadas por coma para el pooling MICE.
- `--beta_vars`: columnas separadas por coma para ajustar distribucion Beta.
- `--dataset_name`: nombre usado para la carpeta de salida.
- `--skip_ai`: omite la interpretacion con IA.
- `--no_auto_retry`: desactiva el reintento automatico cuando MICE detecta
  severidad alta.
- `--no_interactive`: no hace preguntas por consola; usa defaults seguros para
  lo que falte.

Ejemplo completo:

```powershell
python -m scripts.run_pipeline `
  --input "WH2023.xlsx" `
  --goal inference `
  --dataset_name WH2023 `
  --domain_context "Datos del World Happiness Report 2023 por pais." `
  --vars "Ladder,LGDP,Social_support" `
  --beta_vars "Ladder" `
  --no_auto_retry
```

## Que hace el pipeline

1. Limpia y estandariza los datos.
2. Diagnostica el patron de faltantes.
3. Decide el metodo de imputacion.
4. Imputa e informa supuestos.
5. Genera interpretacion con IA si esta habilitada.
6. Arma el dashboard HTML offline.

## Configurar el proveedor de IA

El proveedor se controla con `LLM_PROVIDER` en `.env`.

- `LLM_PROVIDER=ollama`: usa Ollama local. Es gratis y privado, pero suele ser
  mas lento y menos preciso que servicios externos.
- `LLM_PROVIDER=gemini`: usa la API de Gemini. Puede tener capa gratuita con
  limites de uso.
- `LLM_PROVIDER=deepseek`: usa la API de DeepSeek. Es de pago, con buen balance
  entre costo y calidad.

Variables comunes en `.env.example`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
GEMINI_MODEL=gemini-3.6-flash
DEEPSEEK_MODEL=deepseek-chat
```

## Resultados

Cada corrida crea una carpeta nueva:

```text
outputs/<nombre>_<fecha>/
```

Dentro encontraras:

- `datos_limpios.csv`: datos estandarizados antes de imputar.
- `datos_imputados.csv`: datos finales despues de imputar.
- `perfil.json`: perfil de ingesta y filas eliminadas por falta de informacion.
- `reporte.json`: reporte consolidado para lectura y auditoria.
- `reporte.html`: dashboard HTML offline.
- `plots/`: graficos de diagnostico generados por R.
