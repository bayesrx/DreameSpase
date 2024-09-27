#' Simulates N sets of data - returns list of simulated data objects.
#' @return List of length N, each entry of which is a list with the following
#' attributes: `data` is the final simulated lattice; `mu` is the mean component;
#' `alpha` is the effect of the covariates on the lattice;
#' `delta` is the spatial error; `epsilon` is the pure error; `X` is the
#' simulated covariates; `max_val` and `min_val` are the maximum and minimum
#' values simulated on the lattice, respectively; `W` is the adjacency matrix;
#' `D_w` is a diagonal matrix whose entries are the row sums of the
#' corresponding rows of `W` (this is used in sampling from CARs on the lattice).
#' @import purrr
#' @export
SimulateDataSetX = function(N, m, n, X,
                           alpha,
                           tau_2, rho,
                           psi_2,
                           nu_2,
                           phi) {

    W = RectangularAdjMat(m, n)

    Sigma = solve((diag(x = rowSums(W), nrow = nrow(W)) - phi * W))

    return(map(1:N,
             ~ SimulateData(W = W, r = m,
                               alpha = alpha,
                               X = t(t(X[.x, ])),
                               tau_2 = tau_2, rho = rho,
                               psi_2 = psi_2, Sigma = Sigma,
                               nu_2 = nu_2)))
}
