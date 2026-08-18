#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(mice)
  library(naniar)
  library(VIM)
  library(DataExplorer)
  library(inspectdf)
  library(dlookr)
  library(mvnmle)
})

# -----------------------------------------------------------------------------
# Interfaz de línea de comandos
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("--input"),
              type = "character",
              default = NULL,
              help = "Ruta al archivo CSV de entrada"),
  make_option(c("--output_json"),
              type = "character",
              default = NULL,
              help = "Ruta del archivo JSON de salida"),
  make_option(c("--output_plots_dir"),
              type = "character",
              default = NULL,
              help = "Carpeta donde se guardarán los gráficos PNG")
)

parser <- OptionParser(option_list = option_list)
opt <- parse_args(parser)

if (is.null(opt$input) || is.null(opt$output_json) || is.null(opt$output_plots_dir)) {
  print_help(parser)
  stop("Debe especificar --input, --output_json y --output_plots_dir.", call. = FALSE)
}

if (!file.exists(opt$input)) {
  stop(paste0("No existe el archivo de entrada: ", opt$input), call. = FALSE)
}

dir.create(opt$output_plots_dir, recursive = TRUE, showWarnings = FALSE)
output_json_dir <- dirname(opt$output_json)
if (!identical(output_json_dir, ".")) {
  dir.create(output_json_dir, recursive = TRUE, showWarnings = FALSE)
}

# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
Y <- read.csv(opt$input, check.names = FALSE, stringsAsFactors = FALSE)

if (nrow(Y) == 0 || ncol(Y) == 0) {
  stop("El CSV de entrada no contiene datos analizables.", call. = FALSE)
}

non_numeric <- names(Y)[!vapply(Y, is.numeric, logical(1))]
if (length(non_numeric) > 0) {
  stop(
    paste0(
      "El algoritmo EM y el test de Little de los scripts originales requieren variables numéricas. ",
      "Columnas no numéricas encontradas: ",
      paste(non_numeric, collapse = ", ")
    ),
    call. = FALSE
  )
}

if (!anyNA(Y)) {
  stop("El CSV de entrada no contiene datos faltantes (NA).", call. = FALSE)
}

n = nrow(Y) ; p = ncol(Y)

# -----------------------------------------------------------------------------
# Utilidades para guardar gráficos
# -----------------------------------------------------------------------------
plots_generated <- character(0)

save_png <- function(filename, expr, width = 1600, height = 1000, res = 150) {
  path <- file.path(opt$output_plots_dir, filename)
  png(filename = path, width = width, height = height, res = res)
  tryCatch(
    {
      force(expr)
    },
    finally = {
      dev.off()
    }
  )
  plots_generated <<- c(plots_generated, filename)
  invisible(path)
}

# -----------------------------------------------------------------------------
# a. Patrón de datos faltantes: md.pattern(), md.pairs()
# -----------------------------------------------------------------------------
md_pattern_result <- md.pattern(Y, plot = FALSE)
md_pairs_result <- md.pairs(Y)

# -----------------------------------------------------------------------------
# b. Resumen de faltantes: miss_var_summary(), inspect_na(), diagnose()
# -----------------------------------------------------------------------------
miss_var_summary_result <- naniar::miss_var_summary(Y)
inspect_na_result <- inspectdf::inspect_na(Y)
diagnose_result <- dlookr::diagnose(Y)

# -----------------------------------------------------------------------------
# c. Gráficos de diagnóstico de faltantes
# -----------------------------------------------------------------------------
save_png(
  "gg_miss_var.png",
  print(naniar::gg_miss_var(Y))
)

save_png(
  "gg_miss_upset.png",
  print(naniar::gg_miss_upset(Y, nsets = min(4, ncol(Y))))
)

save_png(
  "aggr.png",
  VIM::aggr(Y)
)

save_png(
  "plot_missing.png",
  print(
    DataExplorer::plot_missing(
      Y,
      missing_only = FALSE,
      group = list(Bien = 0.05, Regular = 0.3, Mal = 0.5, Remover = 1),
      group_color = list(
        Bien = "#1B9E77",
        Regular = "#E6AB02",
        Mal = "#D95F02",
        Remover = "#E41A1C"
      )
    )
  )
)

save_png(
  "histMiss.png",
  VIM::histMiss(Y, pos = 1)
)

# -----------------------------------------------------------------------------
# d. Test de Little vía naniar
# -----------------------------------------------------------------------------
mcar_test_result <- naniar::mcar_test(Y)

mcar_statistic <- as.numeric(mcar_test_result$statistic[1])
mcar_p_value <- as.numeric(mcar_test_result$p.value[1])

mcar_conclusion <- if (is.na(mcar_p_value)) {
  "No fue posible determinar la conclusión del test MCAR."
} else if (mcar_p_value < 0.05) {
  "Se rechaza H0: los datos no son compatibles con MCAR."
} else {
  "No se rechaza H0: los datos son compatibles con MCAR."
}

# -----------------------------------------------------------------------------
# e. Algoritmo EM manual del segundo script
#    Se conserva la lógica original; únicamente Y0 toma el CSV de entrada.
# -----------------------------------------------------------------------------
Y0 = Y
ind.miss = ici(Y0) ; ind.miss # indicador de fila con NA
R = is.na(Y0) ; R # matriz indicadora de faltantes

for(t in 0:100){
  if(t == 0){
    mu = colMeans(Y0, na.rm = TRUE)
    S = cov(Y0, use = "pairwise.complete.obs")
    mu_1 = mu ; S_1 = S
  } else{
    mu_1 = mu ; S_1 = S
  }
  
  St = 0
  for(i in  1:nrow(Y0)){
    if(ind.miss[i]){
      EYi = mu[R[i,]] + S[R[i,],!R[i,], drop = FALSE]%*%solve(S[!R[i,],!R[i,], drop = FALSE])%*%(t(Y0[i,!R[i,]])-mu[!R[i,]])
      VYi = S[R[i,],R[i,]] - S[R[i,],!R[i,], drop = FALSE]%*%solve(S[!R[i,],!R[i,]], drop = FALSE)%*%S[!R[i,],R[i,], drop = FALSE] 
      Y0[i,R[i,]] = as.vector(EYi)
      Vi = 0*S ; Vi[R[i,],R[i,]] = VYi 
      St = 1/n * Vi + St
    }
  }
  
  mu = colMeans(Y0)           
  S = St + (n-1)/n*cov(Y0)  
  
  if(sum((mu-mu_1)^2) < 10^(-6) &
     sum((S-S_1)^2) < 10^(-6)) break
}

Y0
mu
S

mu_manual <- mu
S_manual <- S

# -----------------------------------------------------------------------------
# e. Test de Little manual del segundo script
# -----------------------------------------------------------------------------
# Datos observados dentro de los patrones de faltantes
Q = !R
pat = apply(Q, 1, paste0, collapse = "") ; pat # etiqueta del patrón por fila
pats = unique(pat) ; pats

X2 = 0
df_sum = 0

# sumar contribuciones por patrón j
for(pj in pats){
  idx = which(pat == pj)
  nj = length(idx)
  
  # Variables observadas en ese patrón
  Oj = which(Q[idx[1], ])
  kj = length(Oj)
  if(kj == 0) next # al menos una variable debe estar observada
  
  # promedio del patrón (solo en variables observadas)
  ybarj = colMeans(Y[idx, Oj, drop = FALSE])
  
  # Subvector/submatriz según Oj
  muj = mu[Oj]
  Sj  = S[Oj, Oj, drop = FALSE]
  
  # Contribución chi-cuadrado del patrón
  d = matrix(ybarj - muj, ncol = 1)
  X2 = X2 + as.numeric(nj * t(d) %*% solve(Sj) %*% d)
  
  # Para los g.l. (acumulado)
  df_sum = df_sum + kj
}

# Grados de libertad y p-valor
df = df_sum - p
p_value = 1 - pchisq(X2, df)

# Resultado
X2
df
p_value

# -----------------------------------------------------------------------------
# f. EM vía mvnmle para comparar contra el manual
# -----------------------------------------------------------------------------
EM = mvnmle::mlest(Y, iterlim = 100)
EM$muhat
EM$sigmahat

mu_mvnmle <- EM$muhat
S_mvnmle <- EM$sigmahat

if (is.null(names(mu_mvnmle))) {
  names(mu_mvnmle) <- colnames(Y)
}
if (is.null(rownames(S_mvnmle))) {
  rownames(S_mvnmle) <- colnames(Y)
}
if (is.null(colnames(S_mvnmle))) {
  colnames(S_mvnmle) <- colnames(Y)
}

# -----------------------------------------------------------------------------
# g. Del script Ej_IQ_Intro_e_Imp_por_reg: SOLO diagnóstico
#    profile_missing() y barMiss(). Se excluye lm/regressionImp por completo.
# -----------------------------------------------------------------------------
profile_missing_result <- DataExplorer::profile_missing(Y)

save_png(
  "barMiss.png",
  {
    par(mfrow = c(2, 3))
    for(j in seq_len(min(5, ncol(Y)))) {
      VIM::barMiss(Y, pos = j, col = c(gray(.7), 2), border = gray(.6))
    }
    par(mfrow = c(1, 1))
  },
  width = 1800,
  height = 1200,
  res = 150
)

# -----------------------------------------------------------------------------
# Conversión a estructuras serializables sin agregar una librería JSON nueva
# -----------------------------------------------------------------------------
matrix_payload <- function(x) {
  x <- as.matrix(x)
  list(
    row_names = if (is.null(rownames(x))) NULL else rownames(x),
    column_names = if (is.null(colnames(x))) NULL else colnames(x),
    values = lapply(seq_len(nrow(x)), function(i) unname(as.vector(x[i, ])))
  )
}

data_frame_records <- function(x) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  if (nrow(x) == 0) return(list())
  lapply(seq_len(nrow(x)), function(i) {
    row <- lapply(x, function(col) col[[i]])
    names(row) <- names(x)
    row
  })
}

named_vector_payload <- function(x, fallback_names) {
  vals <- as.numeric(x)
  nms <- names(x)
  if (is.null(nms) || any(!nzchar(nms))) nms <- fallback_names
  setNames(as.list(vals), nms)
}

named_matrix_payload <- function(x, fallback_names) {
  x <- as.matrix(x)
  rn <- rownames(x)
  cn <- colnames(x)
  if (is.null(rn)) rn <- fallback_names
  if (is.null(cn)) cn <- fallback_names
  out <- lapply(seq_len(nrow(x)), function(i) {
    setNames(as.list(as.numeric(x[i, ])), cn)
  })
  names(out) <- rn
  out
}

md_pairs_payload <- lapply(md_pairs_result, matrix_payload)

result <- list(
  md_pattern_summary = list(
    md_pattern = matrix_payload(md_pattern_result),
    md_pairs = md_pairs_payload,
    miss_var_summary = data_frame_records(miss_var_summary_result),
    inspect_na = data_frame_records(inspect_na_result),
    diagnose = data_frame_records(diagnose_result),
    profile_missing = data_frame_records(profile_missing_result)
  ),
  mcar_test_naniar = list(
    statistic = mcar_statistic,
    p_value = mcar_p_value,
    conclusion = mcar_conclusion
  ),
  little_test_manual = list(
    X2 = as.numeric(X2),
    df = as.numeric(df),
    p_value = as.numeric(p_value)
  ),
  em_estimates_manual = list(
    mu = named_vector_payload(mu_manual, colnames(Y)),
    sigma = named_matrix_payload(S_manual, colnames(Y))
  ),
  em_estimates_mvnmle = list(
    mu = named_vector_payload(mu_mvnmle, colnames(Y)),
    sigma = named_matrix_payload(S_mvnmle, colnames(Y))
  ),
  plots_generated = as.list(plots_generated)
)

# -----------------------------------------------------------------------------
# Serializador JSON en base R para no añadir jsonlite ni otra dependencia.
# -----------------------------------------------------------------------------
json_escape <- function(x) {
  x <- enc2utf8(as.character(x))
  x <- gsub("\\", "\\\\", x, fixed = TRUE)
  x <- gsub("\"", "\\\"", x, fixed = TRUE)
  x <- gsub("\b", "\\b", x, fixed = TRUE)
  x <- gsub("\f", "\\f", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x <- gsub("\r", "\\r", x, fixed = TRUE)
  x <- gsub("\t", "\\t", x, fixed = TRUE)
  paste0("\"", x, "\"")
}

to_json <- function(x, indent = 0) {
  pad <- paste(rep(" ", indent), collapse = "")
  child_pad <- paste(rep(" ", indent + 2), collapse = "")

  if (is.null(x)) return("null")

  if (is.factor(x)) x <- as.character(x)

  if (length(x) == 1 && !is.list(x)) {
    if (is.na(x)) return("null")
    if (is.logical(x)) return(if (x) "true" else "false")
    if (is.numeric(x)) {
      if (!is.finite(x)) return("null")
      return(sprintf("%.17g", as.numeric(x)))
    }
    return(json_escape(x))
  }

  if (!is.list(x)) {
    x <- as.list(x)
  }

  if (length(x) == 0) return("[]")

  nms <- names(x)
  is_object <- !is.null(nms) && length(nms) == length(x) && all(nzchar(nms))

  if (is_object) {
    parts <- vapply(seq_along(x), function(i) {
      paste0(
        child_pad,
        json_escape(nms[i]),
        ": ",
        to_json(x[[i]], indent + 2)
      )
    }, character(1))
    return(paste0("{\n", paste(parts, collapse = ",\n"), "\n", pad, "}"))
  }

  parts <- vapply(seq_along(x), function(i) {
    paste0(child_pad, to_json(x[[i]], indent + 2))
  }, character(1))
  paste0("[\n", paste(parts, collapse = ",\n"), "\n", pad, "]")
}

json_text <- to_json(result)
writeLines(json_text, con = opt$output_json, useBytes = TRUE)
