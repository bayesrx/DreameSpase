#' Computes covariance matrix for CAR process
#' @param W Weight/adjacency matrix (n x n).
#' @param tau "Variance" parameter; must be strictly positive.
#' @param rho "Correlation" parameter. Must be in \[-1, 1\]
#' @return covariance matrix
CAR_cov = function(W, tau, rho){
  D_w = diag(rowSums(W))
  # Marginal covariance of the spatial effects (from Banerjee, p 81)
  cov_mat = solve((1 / tau) * (D_w - rho * W))
  return(cov_mat)
}