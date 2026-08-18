#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(mice)
  library(optparse)
})

options(digits = 3)

# -----------------------------
# Command-line arguments
# -----------------------------
option_list <- list(
  make_option(
    c("--input"),
    type = "character",
    help = "Ruta CSV de entrada"
  ),
  make_option(
    c("--output_data"),
    type = "character",
    help = "Ruta CSV para guardar complete(mi, action = 1)"
  ),
  make_option(
    c("--output_json"),
    type = "character",
    help = "Ruta JSON para guardar el reporte de pooling"
  ),
  make_option(
    c("--vars"),
    type = "character",
    default = NULL,
    help = "Columnas separadas por coma para el pooling"
  ),
  make_option(
    c("--m"),
    type = "integer",
    default = 5,
    help = "Numero de imputaciones [default %default]"
  ),
  make_option(
    c("--maxit"),
    type = "integer",
    default = 5,
    help = "Numero maximo de iteraciones [default %default]"
  ),
  make_option(
    c("--seed"),
    type = "integer",
    default = 123,
    help = "Semilla [default %default]"
  )
)

parser <- OptionParser(option_list = option_list)
opt <- parse_args(parser)

required_args <- c("input", "output_data", "output_json")
missing_args <- required_args[
  vapply(required_args, function(x) {
    is.null(opt[[x]]) || !nzchar(opt[[x]])
  }, logical(1))
]

if (length(missing_args) > 0) {
  stop(
    paste0(
      "Faltan argumentos obligatorios: ",
      paste(paste0("--", missing_args), collapse = ", ")
    ),
    call. = FALSE
  )
}

if (!file.exists(opt$input)) {
  stop(paste0("No existe el archivo de entrada: ", opt$input), call. = FALSE)
}

# -----------------------------
# Input data
# -----------------------------
data <- read.csv(
  opt$input,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (is.null(opt$vars) || !nzchar(trimws(opt$vars))) {
  vars <- names(data)[vapply(data, is.numeric, logical(1))]
} else {
  vars <- trimws(strsplit(opt$vars, ",", fixed = TRUE)[[1]])
  vars <- vars[nzchar(vars)]
}

if (length(vars) == 0) {
  stop("No hay variables numericas disponibles para calcular el pooling.", call. = FALSE)
}

not_found <- setdiff(vars, names(data))
if (length(not_found) > 0) {
  stop(
    paste0(
      "Variables no encontradas en el dataset: ",
      paste(not_found, collapse = ", ")
    ),
    call. = FALSE
  )
}

non_numeric <- vars[!vapply(data[vars], is.numeric, logical(1))]
if (length(non_numeric) > 0) {
  stop(
    paste0(
      "Las variables usadas para el pooling deben ser numericas: ",
      paste(non_numeric, collapse = ", ")
    ),
    call. = FALSE
  )
}

q <- length(vars)
n <- nrow(data)
m <- opt$m
maxit <- opt$maxit
seed <- opt$seed

if (m < 2) {
  stop("--m debe ser al menos 2 para aplicar las Reglas de Rubin.", call. = FALSE)
}

# -----------------------------
# MICE
# -----------------------------
mi <- mice(data, m = m, maxit = maxit, seed = seed)

completed_data <- complete(mi, action = 1)

output_data_dir <- dirname(opt$output_data)
if (!dir.exists(output_data_dir)) {
  dir.create(output_data_dir, recursive = TRUE, showWarnings = FALSE)
}

write.csv(completed_data, opt$output_data, row.names = FALSE)

# Complete all imputations for pooling
completed_sets <- lapply(seq_len(m), function(i) {
  complete(mi, action = i)
})

# -----------------------------
# Estimates of means
# -----------------------------
mean.m <- lapply(completed_sets, function(dat) {
  sapply(vars, function(v) mean(dat[[v]]))
})

mean.all <- do.call(rbind, mean.m)
colnames(mean.all) <- vars

mean.est <- colMeans(mean.all)

# -----------------------------
# JSON helpers (base R only)
# -----------------------------
json_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub("\"", "\\\\\"", x)
  x <- gsub("\b", "\\\\b", x)
  x <- gsub("\f", "\\\\f", x)
  x <- gsub("\n", "\\\\n", x)
  x <- gsub("\r", "\\\\r", x)
  x <- gsub("\t", "\\\\t", x)
  x
}

json_number <- function(x) {
  if (length(x) != 1 || is.na(x) || !is.finite(x)) {
    return("null")
  }
  sprintf("%.15g", x)
}

to_json <- function(x, indent = 0) {
  pad <- paste(rep(" ", indent), collapse = "")
  pad2 <- paste(rep(" ", indent + 2), collapse = "")

  if (is.null(x)) {
    return("null")
  }

  if (is.list(x)) {
    nms <- names(x)
    is_object <- !is.null(nms) && length(nms) == length(x) && all(nzchar(nms))

    if (length(x) == 0) {
      return(if (is_object) "{}" else "[]")
    }

    if (is_object) {
      pieces <- vapply(seq_along(x), function(i) {
        paste0(
          pad2,
          "\"", json_escape(nms[i]), "\": ",
          to_json(x[[i]], indent + 2)
        )
      }, character(1))
      return(paste0("{\n", paste(pieces, collapse = ",\n"), "\n", pad, "}"))
    }

    pieces <- vapply(x, function(item) {
      paste0(pad2, to_json(item, indent + 2))
    }, character(1))
    return(paste0("[\n", paste(pieces, collapse = ",\n"), "\n", pad, "]"))
  }

  if (is.numeric(x)) {
    if (length(x) == 1) {
      return(json_number(x))
    }
    return(paste0("[", paste(vapply(x, json_number, character(1)), collapse = ", "), "]"))
  }

  if (is.logical(x)) {
    if (length(x) == 1) {
      if (is.na(x)) return("null")
      return(if (x) "true" else "false")
    }
    vals <- vapply(x, function(v) {
      if (is.na(v)) "null" else if (v) "true" else "false"
    }, character(1))
    return(paste0("[", paste(vals, collapse = ", "), "]"))
  }

  if (is.character(x)) {
    if (length(x) == 1) {
      if (is.na(x)) return("null")
      return(paste0("\"", json_escape(x), "\""))
    }
    vals <- vapply(x, function(v) {
      if (is.na(v)) "null" else paste0("\"", json_escape(v), "\"")
    }, character(1))
    return(paste0("[", paste(vals, collapse = ", "), "]"))
  }

  stop("Tipo no soportado para serializacion JSON.", call. = FALSE)
}

named_numeric_list <- function(x) {
  out <- as.list(as.numeric(x))
  names(out) <- names(x)
  out
}

matrix_to_nested_list <- function(mat) {
  out <- lapply(seq_len(nrow(mat)), function(i) {
    row_values <- as.list(as.numeric(mat[i, ]))
    names(row_values) <- colnames(mat)
    row_values
  })
  names(out) <- rownames(mat)
  out
}

# -----------------------------
# Pooling
# -----------------------------
if (q == 1) {

  # ==========================================
  # UNIVARIATE: scalar Rubin pooling
  # ==========================================

  vm <- function(x) var(x) / length(x)

  var.m <- lapply(completed_sets, function(dat) {
    sapply(vars, function(v) vm(dat[[v]]))
  })

  var.all <- do.call(rbind, var.m)
  colnames(var.all) <- vars

  # Combining estimates
  mean.est <- colMeans(mean.all)
  S <- colMeans(var.all)
  B <- apply(mean.all, 2, var)
  var.est <- S + (1 + 1 / m) * B

  # Severity (%) of the missing data problem

  # Proportion of variance due to missing data
  lam <- (1 + 1 / m) * B / var.est

  # Relative increase in variance due to missing data
  r <- (1 + 1 / m) * B / S

  # Fraction of missing information about Mean due to missing data
  v <- n * (n - 1) / (n + 2) * (1 - lam)
  gam <- 1 / (1 + r) * (r + 2 / (v + 3))

  report <- list(
    case = "univariate",
    n_imputations = m,
    point_estimate = named_numeric_list(mean.est),
    covariance_estimate = named_numeric_list(var.est),
    severity = list(
      lambda = as.numeric(lam),
      r = as.numeric(r),
      df = as.numeric(v),
      gamma = as.numeric(gam)
    )
  )

} else {

  # ==========================================
  # MULTIVARIATE: covariance matrix pooling
  # ==========================================

  # using vec notation
  cov.m <- lapply(completed_sets, function(dat) {
    cov_matrix <- cov(dat[, vars, drop = FALSE]) / n
    cov_matrix[lower.tri(cov_matrix, diag = TRUE)]
  })

  cov.all <- do.call(rbind, cov.m)

  # Combining estimates
  mean.est <- colMeans(mean.all)

  # using vec notation
  S <- colMeans(cov.all)

  # matrix notation
  Sigma <- matrix(0L, nrow = q, ncol = q)
  Sigma[lower.tri(Sigma, diag = TRUE)] <- S
  Sigma <- t(Sigma)
  Sigma[lower.tri(Sigma, diag = TRUE)] <- S
  dimnames(Sigma) <- list(vars, vars)

  B <- cov(mean.all)
  dimnames(B) <- list(vars, vars)

  cov.est <- Sigma + (1 + 1 / m) * B
  dimnames(cov.est) <- list(vars, vars)

  # Severity (%) of the missing data problem

  # Proportion of variance due to missing data
  lam <- (1 + 1 / m) / q * sum(diag(B %*% solve(cov.est)))

  # Relative increase in variance due to missing data
  r <- (1 + 1 / m) / q * sum(diag(B %*% solve(Sigma)))

  # Fraction of missing information about Mean due to missing data
  v <- (n - q + 1) * (n - q) / (n - q + 3) * (1 - lam)
  gam <- 1 / (1 + r) * (r + 2 / (v + 3))

  report <- list(
    case = "multivariate",
    n_imputations = m,
    point_estimate = named_numeric_list(mean.est),
    covariance_estimate = matrix_to_nested_list(cov.est),
    severity = list(
      lambda = as.numeric(lam),
      r = as.numeric(r),
      df = as.numeric(v),
      gamma = as.numeric(gam)
    )
  )
}

# -----------------------------
# Write JSON
# -----------------------------
output_json_dir <- dirname(opt$output_json)
if (!dir.exists(output_json_dir)) {
  dir.create(output_json_dir, recursive = TRUE, showWarnings = FALSE)
}

writeLines(
  to_json(report),
  con = opt$output_json,
  useBytes = TRUE
)
