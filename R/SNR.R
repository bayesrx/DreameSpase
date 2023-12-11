SNR = function(W, p, alpha, psi_2, phi, sigma_2_x, nu_2, tau_2, rho){

  base_eigen = unlist(
    lapply(1:p,
           FUN = function(i){
             return(sum(diag(CAR_cov(W, 1, phi))))
           })
  )

  psi_component = sum(base_eigen * sigma_2_x * psi_2)

  alpha_component = sum(sigma_2_x * alpha)

  denominator = nu_2 + sum(diag(CAR_cov(W, tau_2, rho)))

  SNR = (alpha_component + psi_component) / denominator


  return(SNR)
}
