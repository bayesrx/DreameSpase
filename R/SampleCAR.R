#' Samples from standard CAR process.
#' @param W Weight/adjacency matrix (n x n).
#' @param tau "Variance" parameter; must be strictly positive.
#' @param rho "Correlation" parameter. Must be in \[-1, 1\]
#' @return numeric vector of length n of samples
SampleCAR = function(W, tau, rho){
  D_w = diag(rowSums(W))
  # Marginal covariance of the spatial effects (from Banerjee, p 81)
  cov_mat = solve((1/tau) * (D_w - rho * W))
  return(as.numeric(rmvnorm_chol(1, rep(0, nrow(W)), cov_mat)))
}
