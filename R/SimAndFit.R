#' @export
SimAndFit = function(
  alpha,
  alpha_prior,
  X_sigma_2,
  N, m, n,
  tau_2, rho,
  nu_2,
  psi_2,
  phi,
  initial_values,
  prior_values,
  burnin, samples,
  spike_and_slab = TRUE
){

  ifelse = function(test, t, f){
    if(test) return(t)
    else return(f)
  }

  p = length(alpha)
  raw_data = SimulateDataSet(
    N = N,
    m = m,
    n = n,
    X_mu = rep(0, p),
    X_sigma = diag(x = X_sigma_2, nrow = p),
    alpha = alpha,
    tau_2 = tau_2,
    rho = rho,
    psi_2 = psi_2,
    phi = phi,
    nu_2 = nu_2
  )

  #############
  # Data
  #############
  y_l = map(raw_data, ~ as.numeric(.x$Y))
  y = unlist(y_l)
  X = do.call(rbind, map(raw_data, ~ .x$X))
  y_counts = map_dbl(y_l, ~ length(.x))
  W = map(raw_data, ~ .x$W)

  #################
  # Initial Values
  #################

  nu_2_init = ifelse(
    !is.null(initial_values[["nu_2_init"]]),
    initial_values[["nu_2_init"]],
    0.1
  )

  alpha_init = ifelse(
    !is.null(initial_values[["alpha_init"]]),
    initial_values[["alpha_init"]],
    rep(0, p)
  )

  delta_init = unlist(map(y_counts, ~ rep(0, .x)))

  rho_init = ifelse(
    !is.null(initial_values[["rho_init"]]),
    initial_values[["rho_init"]],
    5
  )

  tau_2_delta_init = ifelse(
    !is.null(initial_values[["tau_2_delta_init"]]),
    initial_values[["tau_2_delta_init"]],
    1
  )

  psi_2_init = ifelse(
    !is.null(initial_values[["psi_2_init"]]),
    initial_values[["psi_2_init"]],
    rep(1,N)
  )

  eta_init = map(y_counts, ~ matrix(0, nrow = .x, ncol = p))

  d_init = rep(1, p)


  #################
  # Priors
  #################

  nu_2_alpha = ifelse(
    !is.null(prior_values[["nu_2_alpha"]]),
    prior_values[["nu_2_alpha"]],
    0.001
  )

  nu_2_beta = ifelse(
    !is.null(prior_values[["nu_2_beta"]]),
    prior_values[["nu_2_beta"]],
    0.001
  )

  lambda_alpha = ifelse(
    !is.null(prior_values[["lambda_alpha"]]),
    prior_values[["lambda_alpha"]],
    rep(1, p)
  )

  nu_j = ifelse(
    !is.null(prior_values[["nu_j"]]),
    prior_values[["nu_j"]],
    rep(1, p)
  )

  xi = ifelse(
    !is.null(prior_values[["xi"]]),
    prior_values[["xi"]],
    1
  )

  tau_s_alpha = ifelse(
    !is.null(prior_values[["tau_s_alpha"]]),
    prior_values[["tau_s_alpha"]],
    ifelse(alpha_prior == "ridge" || alpha_prior == "Ridge",
           1000000, 1)
  )

  grid_prior_values = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  rho_prior_probs = rep(1 / length(grid_prior_values), length(grid_prior_values))

  tau_2_delta_alpha = ifelse(
    !is.null(prior_values[["tau_2_delta_alpha"]]),
    prior_values[["tau_2_delta_alpha"]],
    0.001
  )

  tau_2_delta_beta = ifelse(
    !is.null(prior_values[["tau_2_delta_beta"]]),
    prior_values[["tau_2_delta_beta"]],
    0.001
  )

  psi_2_alpha = ifelse(!is.null(prior_values[["psi_2_alpha"]]),
                        prior_values[["psi_2_alpha"]],
                        0.01)

  psi_2_beta = ifelse(!is.null(prior_values[["psi_2_beta"]]),
                        prior_values[["psi_2_beta"]],
                        0.01)

  sigma_2_j = ifelse(!is.null(prior_values[["sigma_2_j"]]),
                     prior_values[["sigma_2_j"]],
                     100)

  sigma_2_j_s = ifelse(!is.null(prior_values[["sigma_2_j_s"]]),
                       prior_values[["sigma_2_j_s"]],
                       0.01)

  prior_d = ifelse(!is.null(prior_values[["prior_d"]]),
                     prior_values[["prior_d"]],
                     0.5)

  if(!spike_and_slab){
    model_output = FitModel(
      y = y,             # Data
      y_counts = y_counts,
      X = X,
      method = "gibbs",
      alpha_prior = alpha_prior,
      W = W,
      phi = phi,
      correction = correction,
      burnin = burnin,     # Settings
      samples = samples,
      thinning = 1,
      verbose = TRUE,
      nu_2_init = nu_2_init, # Initial Values
      alpha_init = alpha_init,
      delta_init = delta_init,
      rho_init = rho_init,
      tau_2_delta_init = tau_2_delta_init,
      psi_2_init = psi_2_init,
      eta_init = eta_init,
      nu_2_alpha = nu_2_alpha,
      nu_2_beta = nu_2_beta,
      tau_s_alpha = tau_s_alpha,
      lambda_alpha = lambda_alpha,
      grid_prior_values = grid_prior_values,
      rho_prior_probs = rho_prior_probs,
      tau_2_delta_alpha = tau_2_delta_alpha,
      tau_2_delta_beta = tau_2_delta_beta,
      psi_2_alpha = psi_2_alpha,
      psi_2_beta = psi_2_beta
    )
  }else{
    model_output = FitSSModel(
      y = y,             # Data
      y_counts = y_counts,
      X = X,
      method = "gibbs",
      alpha_prior = "horseshoe",
      W = W,
      phi = phi,
      correction = correction,
      burnin = burnin,     # Settings
      samples = samples,
      thinning = 1,
      verbose = TRUE,
      nu_2_init = nu_2_init, # Initial Values
      alpha_init = alpha_init,
      delta_init = delta_init,
      rho_init = rho_init,
      tau_2_delta_init = tau_2_delta_init,
      psi_2_init = psi_2_init,
      eta_init = eta_init,
      d_init = d_init,
      nu_2_alpha = nu_2_alpha,  # prior values
      nu_2_beta = nu_2_beta,
      tau_s_alpha = tau_s_alpha,
      lambda_alpha = lambda_alpha,
      nu_j = nu_j,
      xi = xi,
      grid_prior_values = grid_prior_values,
      rho_prior_probs = rho_prior_probs,
      tau_2_delta_alpha = tau_2_delta_alpha,
      tau_2_delta_beta = tau_2_delta_beta,
      psi_2_alpha = psi_2_alpha,
      psi_2_beta = psi_2_beta,
      sigma_2_j = sigma_2_j,
      sigma_2_j_s = sigma_2_j_s,
      prior_d = prior_d
    )
  }


  ret_obj = list(
    raw_data = raw_data,
    model_output = model_output
  )

  return(ret_obj)

}
