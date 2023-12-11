#' Fits DreameSpase model on given data
#' @param data_list List of vectors of outcomes. List entries represent regions,
#' and vector entries represent outcomes at sub-region levels.
#' @param X N x p matrix of covariates, where N is number of regions (i.e. length
#' of data_list), and p is number of covariates
#' @param W List of adjacency matrices. Entry i represents the adjacency matrix
#' for region i, i.e. length(data_list[i]) = nrow(W[[i]]) = ncol(W[[i]]). These
#' matrices should be symmetric, and should have 1 in entry i,j if sub-regions
#' i and j are adjacent, and zeroes otherwise (including the diagonal).
#' @param n_samples Number of samples to take after burnin
#' @param n_burnin Number of burnin samples
#' @param thinning How much if at all to thin the posterior samples. Defaults
#' to 1, i.e. no thinning
#' @param settings Additional settings, including priors. See vignette for
#' overview of most important settings.
#' @import magrittr
#' @export
DreameSpase = function(data_list, X, W, n_samples, n_burnin, thinning = 1,
                       settings = NULL){
  setting_default = setd = function(s, d){
    if(is.null(s)) return(d)
    else return(s)
  }

  s = settings
  seed = setd(s$seed, 24601)
  set.seed(seed)

  #################
  # Model settings
  #################
  update_spike_every = setd(s$update_spike_every, 10)
  progress_every = setd(s$progress_every, 1000)
  spike_optim = setd(s$spike_optim, TRUE)
  sample_delta = setd(s$sample_delta, TRUE)
  sample_gamma = setd(s$sample_gamma, TRUE)
  sample_d = setd(s$sample_d, TRUE)

  #############
  # Data
  #############
  intercept = setd(s$intercept, FALSE)

  y = unlist(data_list)
  center_scale_outcome = setd(s$center_scale_outcome, FALSE)
  if(center_scale_outcome){
    y = (y - mean(y))/sd(y)
  }
  p = ncol(X)
  y_counts = map_dbl(data_list, ~ length(.x))


  #################
  # Main Effects
  #################
  # Prior
  sigma_2_alpha_spike = setd(s$sigma_2_alpha_spike, 0.01)
  sigma_2_alpha_slab = setd(s$sigma_2_alpha_slab, 100)
  gamma_prior = rep(0.5, p)
  P_gamma = setd(s$P_gamma, 0.5)
  a_gamma = setd(s$a_gamma, 1)
  b_gamma = setd(s$b_gamma, 1)

  # Initial values
  reg_tib = map2(1:nrow(X), y_counts, \(r, n){
    matrix(rep(X[r,], n), nrow = n, byrow = TRUE)
  }) %>%
    do.call(rbind, .) %>%
    set_colnames(paste0("V", 1:ncol(X))) %>%
    as_tibble()

  simple_lm = lm(y ~ 0 + ., reg_tib)

  alpha_init = coef(simple_lm) %>% unname()
  gamma_init = setd(s$gamma_init, rep(1, p))

  #####################
  # Covariate CAR (eta)
  #####################
  # Initial Values
  total_var = var(y)
  if(!is.null(s$warm_start) && s$warm_start){

    biopsy_sd = data_list %>% map_dbl(sd)

    regression_tib = abs(X) %>%
      set_colnames(paste0("V", 1:ncol(X))) %>%
      as_tibble() %>%
      mutate(
        sd = biopsy_sd
      )

    univ_lms = map(paste0("V", 1:(ncol(X))), \(v){
      lm(as.formula(paste("sd ~ ", v)), data = regression_tib)
    })

    univ_p = map_dbl(univ_lms, ~ summary(.x)$coefficients[2,4])

    rand_start_selected = which(univ_p < 0.1)
  }else{
    rand_start_selected = 1:p
  }

  n_rand_components_start = length(rand_start_selected) + 1 + sample_delta
  var_init = total_var / n_rand_components_start
  phi = setd(s$phi, 0.3)

  # Priors
  psi_2_j_slab = setd(s$psi_2_j_slab, 100)
  psi_2_j_spike = setd(s$psi_2_j_spike, 0.1)
  prior_d = 0.5
  P_d = setd(s$P_d, 0.5)
  a_d = setd(s$a_d, 1)
  b_d = setd(s$b_d, 1)

  # Initial Values
  eta_init = map(y_counts, ~ matrix(0, nrow = .x, ncol = p - intercept))
  psi_2_init = abs(rnorm(p - intercept, mean = 0, sd = psi_2_j_spike))
  psi_2_init[rand_start_selected] = var_init
  d_init = setd(s$d_init, rep(0, p - intercept))
  d_init[rand_start_selected] = 1

  #####################
  # Global CAR (delta)
  #####################
  # Initial values
  sample_delta = setd(s$sample_delta, TRUE)
  delta_init = unlist(map(y_counts, ~ rep(0, .x)))
  rho_init = setd(s$rho_init, 0)
  tau_2_delta_init = ifelse(sample_delta, var_init, 0)

  # Priors
  tau_2_delta_alpha = 0.001
  tau_2_delta_beta = 0.001
  grid_prior_values = setd(s$grid_prior_values, seq(0.1, 0.9, 0.1))
  rho_prior_probs = setd(s$grid_prior_probs,
                         rep(1 / length(grid_prior_values),
                             length(grid_prior_values)))

  #####################
  # Pure error
  #####################
  # Prior
  nu_2_alpha = 0.001
  nu_2_beta = 0.001

  # Initial value
  nu_2_init = var_init

  #####################
  # Outdated/to remove
  #####################
  groups = c(0)
  n_groups = 0
  d_groups_init = c(0)
  gamma_groups_init = c(0)
  psi_2_alpha = -1
  psi_2_beta = -1
  tau_s_alpha = 1
  xi = 1
  nu_j = rep(1, p)
  lambda_alpha = rep(1, p)

  model_output = FitSSModel(
    y = y,             # Data
    y_counts = y_counts,
    X = X,
    groups = groups, # groups,
    n_groups = 0, # n_groups,
    intercept = intercept,
    method = "gibbs",
    alpha_prior = "ss",
    W = W,
    phi = phi,
    correction = 1,
    burnin = n_burnin,     # Settings
    samples = n_samples,
    thinning = thinning,
    verbose = TRUE,
    spike_optim = spike_optim,
    progress_every = progress_every,
    update_spike_every = update_spike_every,
    sample_delta = sample_delta,
    sample_gamma = sample_gamma,
    sample_d = sample_d,
    nu_2_init = nu_2_init, # Initial Values
    alpha_init = alpha_init,
    delta_init = delta_init,
    rho_init = rho_init,
    tau_2_delta_init = tau_2_delta_init,
    psi_2_init = psi_2_init,
    eta_init = eta_init,
    d_init = d_init,
    gamma_init = gamma_init,
    d_groups_init = d_groups_init,
    gamma_groups_init = gamma_groups_init,
    nu_2_alpha = nu_2_alpha,  # prior values
    nu_2_beta = nu_2_beta,
    tau_s_alpha = tau_s_alpha,
    sigma_2_alpha_spike = sigma_2_alpha_spike,
    sigma_2_alpha_slab = sigma_2_alpha_slab,
    nu_j = nu_j,
    xi = xi,
    lambda_alpha = lambda_alpha,
    grid_prior_values = grid_prior_values,
    rho_prior_probs = rho_prior_probs,
    tau_2_delta_alpha = tau_2_delta_alpha,
    tau_2_delta_beta = tau_2_delta_beta,
    psi_2_alpha = psi_2_alpha,
    psi_2_beta = psi_2_beta,
    sigma_2_j = psi_2_j_slab,
    sigma_2_j_s = psi_2_j_spike,
    prior_d = prior_d,
    gamma_prior = gamma_prior,
    P_gamma = P_gamma,
    a_gamma = a_gamma,
    b_gamma = b_gamma,
    P_d = P_d,
    a_d = a_d,
    b_d = b_d
  )
  return(model_output)
}
