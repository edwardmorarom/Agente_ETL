#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(VIM)
})

option_list <- list(
  make_option("--input", type = "character", default = NULL,
              help = "Ruta CSV de entrada"),
  make_option("--output", type = "character", default = NULL,
              help = "Ruta CSV de salida"),
  make_option("--method", type = "character", default = "stochastic_regression",
              help = "stochastic_mean, regression o stochastic_regression [default: %default]"),
  make_option("--models", type = "character", default = NULL,
              help = "Ruta opcional a archivo JSON o texto con las fórmulas"),
  make_option("--seed", type = "integer", default = 123,
              help = "Semilla [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

fail <- function(...) stop(paste0(...), call. = FALSE)

if (is.null(opt$input) || !nzchar(opt$input)) fail("Debe especificar --input.")
if (is.null(opt$output) || !nzchar(opt$output)) fail("Debe especificar --output.")
if (!file.exists(opt$input)) fail("Archivo de entrada no encontrado: ", opt$input)
if (!is.null(opt$models) && nzchar(opt$models) && !file.exists(opt$models)) {
  fail("Archivo de modelos no encontrado: ", opt$models)
}

valid_methods <- c("stochastic_mean", "regression", "stochastic_regression")
if (!(opt$method %in% valid_methods)) {
  fail("--method debe ser uno de: ", paste(valid_methods, collapse = ", "), ".")
}

set.seed(opt$seed)

datos <- tryCatch(
  read.csv(opt$input, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE),
  error = function(e) fail("No fue posible leer el CSV: ", conditionMessage(e))
)

if (nrow(datos) == 0L) fail("El dataset no contiene filas.")
if (ncol(datos) == 0L) fail("El dataset no contiene columnas.")

quote_col <- function(x) {
  paste0("`", gsub("`", "\\\\`", x, fixed = TRUE), "`")
}

make_model <- function(target, predictors) {
  if (length(predictors) == 0L) {
    fail("No hay predictores numéricos suficientes para la variable '", target, "'.")
  }

  as.formula(
    paste(
      quote_col(target), "~",
      paste(vapply(predictors, quote_col, character(1)), collapse = " + ")
    )
  )
}

parse_models_file <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  txt <- paste(lines, collapse = "\n")
  txt_trim <- trimws(txt)

  if (!nzchar(txt_trim)) fail("El archivo de modelos está vacío: ", path)

  # JSON: extrae strings que contengan fórmulas.
  if (substr(txt_trim, 1L, 1L) %in% c("{", "[")) {
    quoted <- regmatches(
      txt,
      gregexpr('"([^"\\\\]|\\\\.)*"', txt, perl = TRUE)
    )[[1]]

    if (length(quoted) > 0L && !identical(quoted, character(0))) {
      strings <- substring(quoted, 2L, nchar(quoted) - 1L)
      formulas_txt <- strings[grepl("~", strings, fixed = TRUE)]

      if (length(formulas_txt) > 0L) {
        models <- lapply(formulas_txt, function(x) {
          tryCatch(as.formula(x), error = function(e) {
            fail("Fórmula inválida en archivo de modelos: ", x)
          })
        })
        return(models)
      }
    }
  }

  # Texto/R: admite el formato original list(mod1 = y ~ x1 + x2, ...)
  # y también una fórmula por línea.
  chunks <- unlist(strsplit(txt, "[,\\n]"))
  chunks <- trimws(chunks)
  chunks <- chunks[nzchar(chunks)]
  chunks <- chunks[grepl("~", chunks, fixed = TRUE)]

  formulas_txt <- vapply(chunks, function(x) {
    x <- sub("^[[:space:]]*(models[[:space:]]*(<-|=)[[:space:]]*)?list[[:space:]]*\\(", "", x)
    x <- sub("^[^=]+=[[:space:]]*", "", x)
    x <- sub("[[:space:]]*\\)+[[:space:]]*$", "", x)
    trimws(x)
  }, character(1))

  formulas_txt <- formulas_txt[grepl("~", formulas_txt, fixed = TRUE)]
  if (length(formulas_txt) == 0L) {
    fail(
      "No se encontraron fórmulas válidas en '", path,
      "'. Use JSON con fórmulas como strings o el formato list(mod1 = y ~ x1 + x2, ...)."
    )
  }

  lapply(formulas_txt, function(x) {
    tryCatch(as.formula(x), error = function(e) {
      fail("Fórmula inválida en archivo de modelos: ", x)
    })
  })
}

build_auto_models <- function(data) {
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  targets <- numeric_cols[vapply(data[numeric_cols], anyNA, logical(1))]

  if (length(targets) == 0L) return(list())

  if (length(numeric_cols) < 2L) {
    fail(
      "Columnas numéricas insuficientes: se requieren al menos dos para ",
      "generar automáticamente los modelos de regresión."
    )
  }

  lapply(targets, function(target) {
    make_model(target, setdiff(numeric_cols, target))
  })
}

validate_models <- function(models, data) {
  if (length(models) == 0L) fail("La lista de modelos está vacía.")

  for (k in seq_along(models)) {
    f <- models[[k]]

    if (!inherits(f, "formula") || length(f) < 3L || !is.symbol(f[[2]])) {
      fail("El modelo ", k, " debe tener una sola columna como variable respuesta.")
    }

    vars <- all.vars(f)
    missing_vars <- setdiff(vars, names(data))
    if (length(missing_vars) > 0L) {
      fail(
        "El modelo ", k, " contiene variables que no existen en el dataset: ",
        paste(missing_vars, collapse = ", "), "."
      )
    }

    target <- as.character(f[[2]])
    if (!is.numeric(data[[target]])) {
      fail("La variable respuesta '", target, "' del modelo ", k, " debe ser numérica.")
    }

    if (length(setdiff(vars, target)) == 0L) {
      fail("El modelo ", k, " no contiene predictores.")
    }
  }
}

stochastic_mean_impute <- function(data) {
  result <- data
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]

  for (j in numeric_cols) {
    if (!anyNA(data[[j]])) next

    observed <- data[[j]][!is.na(data[[j]])]
    if (length(observed) < 2L) {
      fail(
        "La variable '", j,
        "' no tiene suficientes valores observados para calcular media y desviación estándar."
      )
    }

    muj <- mean(data[[j]], na.rm = TRUE)
    sj <- sd(data[[j]], na.rm = TRUE)

    for (i in seq_len(nrow(data))) {
      if (is.na(data[i, j])) {
        result[i, j] <- muj + rnorm(1, 0, sj)
      }
    }
  }

  result
}

regression_impute <- function(data, models, stochastic = FALSE) {
  datos.reg <- data
  datos.sreg <- data

  for (k in seq_along(models)) {
    model <- models[[k]]
    target <- as.character(model[[2]])

    fit <- tryCatch(
      lm(model, data = datos.reg),
      error = function(e) fail("Error al ajustar el modelo ", k, ": ", conditionMessage(e))
    )

    sig <- sigma(fit)
    if (!is.finite(sig)) {
      fail(
        "No fue posible estimar sigma para el modelo ", k,
        ". Verifique que existan suficientes casos completos."
      )
    }

    miss <- is.na(datos.reg[, target])

    datos.reg <- tryCatch(
      regressionImp(model, datos.reg, family = gaussian, robust = FALSE),
      error = function(e) fail("Error en regressionImp() para el modelo ", k, ": ", conditionMessage(e))
    )

    if (stochastic) {
      datos.sreg[, target] <- datos.reg[, target] +
        rnorm(nrow(datos.reg), 0, sig) * miss
    }
  }

  if (stochastic) datos.sreg else datos.reg
}

if (opt$method == "stochastic_mean") {
  resultado <- stochastic_mean_impute(datos)
} else {
  models <- if (!is.null(opt$models) && nzchar(opt$models)) {
    parse_models_file(opt$models)
  } else {
    build_auto_models(datos)
  }

  if (length(models) == 0L) {
    resultado <- datos
  } else {
    validate_models(models, datos)

    resultado <- regression_impute(
      datos,
      models,
      stochastic = identical(opt$method, "stochastic_regression")
    )
  }
}

output_dir <- dirname(opt$output)
if (!dir.exists(output_dir)) {
  ok <- dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  if (!ok && !dir.exists(output_dir)) fail("No fue posible crear la carpeta de salida: ", output_dir)
}

tryCatch(
  write.csv(resultado, opt$output, row.names = FALSE, na = ""),
  error = function(e) fail("No fue posible escribir el CSV de salida: ", conditionMessage(e))
)
