#' Samples from a multivariate normal distribution using cholesky decomposition.
#' @param n Number of samples from specified distribution.
#' @param mu Mean vector of distribution (p x 1)
#' @param Sigma Covariance matrix of distribution (p x p).
#' @return n x p matrix of samples
rmvnorm_chol <- function(n,mu,Sigma){
  p = length(mu)
  Z = matrix(rnorm(n*p),nrow=p,ncol=n)
  U = chol(Sigma)
  X = mu + crossprod(U, Z)
  return(t(X))
}
