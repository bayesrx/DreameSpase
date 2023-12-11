#' Simulate data from the model on a rectangular lattice.
#' @param W Adjacency matrix
#' @param r Number of rows in the lattice
#' @param alpha p x 1 vector of mean effects of covariates
#' @param sigma_2 variance of means conditional upon alpha
#' @param X p x 1 vector of covariate values
#' @param psi_2 Variance component of spatial effects CAR (eta)
#' @param phi Correlation component of spatial effects CAR (eta)
#' @param tau_2 Variance component of spatial error CAR (delta)
#' @param rho Correlation component of spatial error CAR (delta)
#' @param nu_2 Variance of pure error
#' @return List
#' @import MASS
#' @import purrr
SimulateData = function(W, r, alpha, X, psi_2, Sigma, tau_2, rho, nu_2){
  n = nrow(W)
  p = length(X)
  Y = matrix(0, nrow = n)
  # generate mean
  mu = as.numeric(t(X) %*% alpha)
  # mean component
  Y = Y + mu

  # eta
  eta = do.call(cbind, purrr::map(1:p,
                                  ~ MASS::mvrnorm(mu = rep(0, n), Sigma = Sigma)))

  Y = Y + eta %*% (X * sqrt(psi_2))

  # delta
  delta = SampleCAR(W, tau_2, rho)
  Y = Y + delta

  # epsilon
  epsilon = rnorm(n, mean = 0, sd = sqrt(nu_2))

  Y = Y + epsilon
  Y_mat = matrix(Y, nrow = r)
  ret_obj = list()
  ret_obj$Y = Y_mat
  ret_obj$eta = map(1:p, ~ matrix(sqrt(psi_2[.x]) * eta[,.x], nrow = r))
  ret_obj$delta = matrix(delta, nrow = r)
  ret_obj$epsilon = matrix(epsilon, nrow = r)
  ret_obj$X = X
  ret_obj$mu = mu
  ret_obj$W = W
  return(ret_obj)
}
