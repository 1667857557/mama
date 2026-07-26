#' Run the MAMA core estimator
#'
#' The arrays use the same convention as the Python implementation: `betas`
#' is M by P and `omega` and `sigma` are M by P by P.  Work is split into
#' chunks before crossing the native boundary.  Inside Rust, only the matrices
#' for one SNP and one target population are allocated.
#'
#' @param betas Numeric M by P matrix.
#' @param omega Numeric M by P by P array.
#' @param sigma Numeric M by P by P array.
#' @param chunk_size Maximum variants passed to Rust per call.
#' @return A list with M by P matrices `beta` and `se`.
#' @export
mama_core <- function(betas, omega, sigma, chunk_size = 4096L) {
  if (!is.matrix(betas) || !is.numeric(betas)) stop("betas must be a numeric matrix")
  m <- nrow(betas)
  p <- ncol(betas)
  if (!identical(dim(omega), c(m, p, p)) || !is.numeric(omega))
    stop("omega must be a numeric M x P x P array")
  if (!identical(dim(sigma), c(m, p, p)) || !is.numeric(sigma))
    stop("sigma must be a numeric M x P x P array")
  chunk_size <- as.integer(chunk_size)
  if (length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1L)
    stop("chunk_size must be a positive integer")

  ans_beta <- matrix(NA_real_, m, p)
  ans_se <- matrix(NA_real_, m, p)
  if (m == 0L) return(list(beta = ans_beta, se = ans_se))

  for (first in seq.int(1L, m, by = chunk_size)) {
    last <- min(m, first + chunk_size - 1L)
    rows <- first:last
    # drop=FALSE retains the three-dimensional layout for one-row chunks.
    value <- .Call(C_mamars_core, betas[rows, , drop = FALSE],
                   omega[rows, , , drop = FALSE], sigma[rows, , , drop = FALSE])
    ans_beta[rows, ] <- value[[1L]]
    ans_se[rows, ] <- value[[2L]]
  }
  list(beta = ans_beta, se = ans_se)
}

#' Run matrix construction, QC, and the MAMA estimator
#'
#' This is the memory-friendly counterpart of the numerical portion of
#' `mama_pipeline`: it constructs Omega and Sigma per SNP, applies the original
#' Omega tweak and Sigma positive-definiteness checks, substitutes identity
#' matrices for rejected SNPs, and evaluates the original estimator. Regression
#' coefficients are explicit inputs so a fitted analysis is reproducible.
#'
#' @param betas, ses Numeric M by P matrices.
#' @param ldscores Numeric M by P by P LD-score array.
#' @param ld_coef,const_coef,se2_coef Numeric P by P coefficient matrices.
#' @return A list containing `beta`, `se`, logical `keep`, and logical `tweaked`.
#' @export
mama_analysis <- function(betas, ses, ldscores, ld_coef, const_coef, se2_coef) {
  if (!is.matrix(betas) || !is.double(betas)) stop("betas must be a double matrix")
  m <- nrow(betas); p <- ncol(betas)
  if (!identical(dim(ses), c(m, p)) || !is.double(ses))
    stop("ses must be a double M x P matrix")
  if (!identical(dim(ldscores), c(m, p, p)) || !is.double(ldscores))
    stop("ldscores must be a double M x P x P array")
  for (value in list(ld_coef = ld_coef, const_coef = const_coef, se2_coef = se2_coef))
    if (!identical(dim(value), c(p, p)) || !is.double(value))
      stop("all coefficient inputs must be double P x P matrices")
  .Call(C_mamars_analysis, betas, ses, ldscores, ld_coef, const_coef, se2_coef)
}

#' Fit LD-score regressions and run MAMA
#'
#' This is the main numerical entry point. `NA_real_` values in a fixed matrix
#' are fitted; numeric values constrain the corresponding coefficient exactly.
#'
#' @param betas,ses Numeric M by P matrices after QC and allele harmonisation.
#' @param ldscores Numeric M by P by P array in the same SNP/population order.
#' @param ld_fixed,const_fixed,se_fixed Optional P by P constraint matrices.
#' @return Analysis results and the three fitted coefficient matrices.
#' @export
mama_fit <- function(betas, ses, ldscores,
                     ld_fixed = NULL, const_fixed = NULL, se_fixed = NULL) {
  if (!is.matrix(betas) || !is.double(betas)) stop("betas must be a double matrix")
  m <- nrow(betas); p <- ncol(betas)
  if (!identical(dim(ses), c(m, p)) || !is.double(ses)) stop("invalid ses")
  if (!identical(dim(ldscores), c(m, p, p)) || !is.double(ldscores)) stop("invalid ldscores")
  free <- matrix(NA_real_, p, p)
  values <- list(ld_fixed %||% free, const_fixed %||% free, se_fixed %||% free)
  if (any(vapply(values, function(x) !is.double(x) || !identical(dim(x), c(p, p)), logical(1))))
    stop("fixed coefficients must be double P x P matrices")
  .Call(C_mamars_fit, betas, ses, ldscores, values[[1]], values[[2]], values[[3]])
}

`%||%` <- function(x, y) if (is.null(x)) y else x
