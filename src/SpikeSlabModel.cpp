#include "RcppArmadillo.h"
#include "vector"
#include "math.h"
#include <cmath>
#include "gig.h"
#include "utility_functions.h"
#include <random>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

class SpikeSlabModel{
private:
  struct SpatialModelData{
    struct XMats{
      arma::mat X; // N x p covariate matrix
      arma::mat X_T_R_inv_X; // X^T * (diagonal matrix of squared tile counts) * X
      arma::mat X_T_R_inv; // X^T * (diagonal matrix of squared tile counts)
    } X_mats;

    struct Outcome{
      LongVector Y; // Tile level outcomes
      arma::vec y_sums; // Vector of length N of outcome sums by biopsy
      arma::vec tile_counts; // Vector of length N of number of tiles by biopsy
    } outcome;

    struct DataDim{
      int S; // Total # tiles across all biopsies
      int p; // # covariates
      int N; // # biopsies
      int intercept; // 1 if intercept, 0 if no intercept
    } data_dim;

    // Vector of adjacency matrices
    std::vector<arma::mat> W;

    // groups (if doing group selection)
    struct GroupingInfo{
      std::vector<int> groups;
      int n_groups;
    } groups;

    // Psi is the covariance matrix of the global CAR process, i.e. delta;
    // Psi_inv is the precision matrix. These are the precision matrices
    // for various values of rho (which is possible due to the grid prior)
    // as well as select decompositions to speed up the computation of
    // quadratic forms and sampling
    struct PsiDecompositions{
      std::vector<std::vector<arma::mat>> Psi_inv;
      std::vector<std::vector<arma::mat>> Psi;
      std::vector<std::vector<arma::mat>> Psi_inv_chol_inv;
      std::vector<std::vector<arma::mat>> Psi_chol_lower;
      std::vector<std::vector<arma::mat>> Psi_eigen_mats;
      std::vector<std::vector<arma::vec>> Psi_eigen_vals;
    } Psi_decompositions;

    // Ditto, but for Sigma, the "core" of the random effects covariance matrix
    struct SigmaDecompositions{
      std::vector<arma::mat> Sigma;
      std::vector<arma::mat> Sigma_inv;
      std::vector<arma::mat> Sigma_inv_chol_inv;
      std::vector<arma::mat> Sigma_chol_lower;
      std::vector<arma::vec> Sigma_eigen_vals;
      std::vector<arma::mat> Sigma_eigen_mats;
      arma::vec Sigma_log_dets;
    } Sigma_decompositions;

    // matrix of determinants of PSI, so we only need to compute once
    arma::mat PSI_log_det;
    arma::mat PSI_det;
  } dat;

  struct RandSampler{
    Random rand = Random(1);
  }rand_samp;

  struct AlgorithmControl{
    int method;
    int alpha_prior;
    int sample;
    int burnin;
    int thinning;
    bool verbose;
    bool spike_optim;
    int update_spike_every;
    bool sample_delta;
    bool sample_gamma;
    bool sample_d;
  } ctrl;

  // Parameters that are sampled/updated
  struct SampledParas{
    //////////////////////////////////
    // alpha parameters (main effects)
    //////////////////////////////////
    struct Alpha{
      arma::vec alpha; // main effects
      arma::vec gamma; // selection indicators for main effects
      arma::vec gamma_groups; // group selection indicators
      double P_gamma; // Prior probability of selection of main effects
      arma::vec lambda_alpha; // Horseshoe pseudo-probabilities
    } alpha;

    /////////////////////////////////////////
    // delta parameters (global CAR process)
    /////////////////////////////////////////
    struct Delta{
      LongVector delta; // Global CAR process
      int rho; // Global CAR "correlation" param
      double tau_2_delta; // Global CAR "variance" param
    } delta;

    //////////////////////////////////////////////
    // eta parameters (random effects CAR process)
    //////////////////////////////////////////////
    struct Eta{
      std::vector<arma::mat> eta; // Biopsy specific CAR processes
      arma::vec psi_2; // Bipsy specific CAR "variance" parameter
      arma::vec d; // Selection indicators
      arma::vec d_groups; // group selection indicators
      double P_d; // Prior probability of selection of main effects
    } eta;

    //////////////
    // Pure error
    /////////////
    double nu_2; // Variance of residuals

    //////////////////
    // log-likelihood
    /////////////////
    double loglik; // yeah
  }cur_paras, prop_paras;

  // Values that are "expensive" to compute
  struct PreComp{
    LongVector residuals;
    // pre-computed values for GIG - just chi and psi, because everything else is "cheap"
    arma::vec chi;
    arma::vec psi;
    arma::mat Psi_QFs;
    std::vector<bool> d_change;
  } prop_vars;

  struct Prior{
    ////////////////
    // NU
    ///////////////
    struct Nu_2{
      double nu_2_alpha; // alpha parameter for inverse gamma prior
      double nu_2_beta; // beta parameter for IG prior
    } nu_2;

    ////////////////
    // ALPHA
    ///////////////
    struct Alpha{
      // Shrinkage param for ridge regression
      // Global shrinkage for horseshoe
      double tau_s_alpha;
      // Prior mean of each alpha is assumed to be 0
      // Horseshoe stuff
      arma::vec nu_j;
      double xi;
      // Prior for variable selection
      double sigma_2_alpha_spike; // standard deviation of alpha spike
      double sigma_2_alpha_slab; // standard deviation of alpha slab
      // beta parameters for prior on P_gamma
      double a_gamma; // a parameter
      double b_gamma; // b parameter
    } alpha;

    //////////////
    // DELTA
    /////////////
    // Grid prior for rho parameter
    struct Delta{
      struct GridPrior {
        arma::vec probabilities;
        std::vector<double> values;
      } rho_prior;
      double tau_2_delta_alpha; // alpha parameter for IG prior on tau_2
      double tau_2_delta_beta; // beta parameter for IG prior on tau_2
    } delta;

    //////////////
    // eta
    /////////////
    struct Eta{
      // psi
      double sigma_2_j_slab; // slab sd
      double sigma_2_j_spike; // spike sd
      // Parameters for beta prior on P_d
      double a_d; // a parameter
      double b_d; // b parameter
    } eta;
  } prior;

  struct Samples{
    int iter;
    arma::mat mu;
    arma::vec nu_2;
    arma::mat alpha;
    arma::vec rho;
    arma::vec tau_2_delta;
    arma::mat psi_2;
    arma::mat d;
    arma::mat gamma;
    arma::vec P_gamma;
    arma::vec P_d;
    arma::vec loglik;
  } samples;

public:
  void set_method(String);
  void set_alpha_prior(String);
  void load_data(LongVector, // y
                 vec, // y_counts/n
                 arma::mat, // X
                 std::vector<int>, // groups
                 int, // n_groups
                 bool, // intercept
                 std::vector<arma::mat> // W
                );
  void set_control(int, int, int, bool, bool, int, bool, bool, bool);
  void samples_init();
  void sampler_init();
  void prior_init(double, // alpha_nu
                  double, // beta_nu
                  double, // tau_s_alpha
                  double, // sigma_2_alpha_spike
                  double, // sigma_2_alpha_slab
                  arma::vec, // gamma
                  double, // P_gamma
                  double, // a_gamma
                  double, // b_gamma
                  arma::vec, // nu_j
                  double, // xi
                  std::vector<double>, // grid_prior_values
                  arma::vec, // rho_prior_probs
                  double, // tau_2_delta_alpha
                  double, // tau_2_delta_beta
                  double, // psi_eta_alpha
                  double, // psi_eta_beta
                  double, // sigma_2_j_slab
                  double, // sigma_2_j_spike
                  double, // prior_d
                  double, // P_d
                  double, // a_d
                  double, // b_d
                  double, // phi
                  double, // correction
                  std::vector<arma::mat>, // W
                  int // N
                );

  void initial_values_init(double, // nu_2
                           arma::vec, // alpha
                           arma::vec, // lambda_alpha
                           LongVector, // delta
                           std::vector<arma::mat>, // eta
                           int, // rho
                           double, // tau_2_delta
                           arma::vec, // psi
                           arma::vec, // d
                           arma::vec, // gamma
                           arma::vec, // d_groups_init
                           arma::vec // gamma_groups_init
                           );

  void d_change_init();
  void residuals_init();
  void chi_psi_init();
  int  get_iter();
  void augment_iter();
  void update_proposal_paras();
  void update_proposal_nu();
  void update_proposal_alpha();
  void update_proposal_gamma();
  void update_proposal_P_gamma();
  void update_proposal_residuals_alpha(arma::vec, arma::vec);
  void update_proposal_lambdas();
  void update_proposal_nu_j();
  void update_proposal_tau_alpha();
  void update_proposal_xi();
  double lambda_log_lik(
    arma::vec, // lambda
    arma::vec // alpha
  );
  void update_proposal_delta();
  void update_proposal_rho();
  void update_proposal_tau_2();
  void update_proposal_residuals_delta(LongVector, LongVector);
  void update_proposal_eta();
  void update_proposal_residuals_eta(std::vector<arma::mat>&, // old eta
                                     std::vector<arma::mat>&, // new eta
                                     int, // biopsy index
                                     int // variable index
                                     );
  // TODO: add prior for tau_s_alpha
  // void update_proposal_tau_s_alpha();
  void update_proposal_d();
  void update_proposal_P_d();
  void update_proposal_psi_2();
  void update_current_paras();
  void update_proposal_loglik();
  void save_sample();
  double get_rmse_resid_diff();
  double get_max_resid_diff();
  List get_samples();
};

double SpikeSlabModel::get_rmse_resid_diff(){
  LongVector resids = dat.outcome.Y;
  for(int i = 0; i < resids.get_N(); i++){
    // TODO: UPDATE AS YOU ADD OTHER PARAMETERS
    resids[i] -= (as_scalar(dat.X_mats.X.row(i) * prop_paras.alpha.alpha) + prop_paras.delta.delta[i] +
                  prop_paras.eta.eta[i] *
                  dat.X_mats.X.submat(i, 0, i, dat.data_dim.p - dat.data_dim.intercept - 1).as_col());
  }

  double rmse_resids = arma::mean((resids.full_vec() - prop_vars.residuals.full_vec()) %
                            (resids.full_vec() - prop_vars.residuals.full_vec()));
  rmse_resids = sqrt(rmse_resids);
  return rmse_resids;
}

double SpikeSlabModel::get_max_resid_diff(){
  LongVector resids = dat.outcome.Y;
  for(int i = 0; i < resids.get_N(); i++){
    // TODO: UPDATE AS YOU ADD OTHER PARAMETERS
    resids[i] -= (as_scalar(dat.X_mats.X.row(i) * prop_paras.alpha.alpha) + prop_paras.delta.delta[i] +
                  prop_paras.eta.eta[i] *
                  dat.X_mats.X.submat(i, 0, i, dat.data_dim.p - dat.data_dim.intercept - 1).as_col());
  }

  double max_diff = arma::max(resids.full_vec() - prop_vars.residuals.full_vec());
  return max_diff;
}

void SpikeSlabModel::set_method(String method){
  if(method == "Gibbs" || method == "gibbs"){
    std::cout << "Method: Gibbs sampler" << std::endl;
    ctrl.method = 0;
  }else{
    std::invalid_argument("currently, only gibbs sampler method is supported");
  }
}

void SpikeSlabModel::set_alpha_prior(String prior_type){
  if(prior_type == "ridge" || prior_type == "Ridge"){
    ctrl.alpha_prior = 0;
  }else if(prior_type == "horseshoe" || prior_type == "Horseshoe"){
    ctrl.alpha_prior = 1;
  }else if(prior_type == "ss" || prior_type == "spike and slab"){
    ctrl.alpha_prior = 2;
  }else{
    std::invalid_argument("please select either 'horseshoe' or 'ridge' prior for alpha");
  }
}

// Each "biopsy"/region can have different numbers of tiles, hence
// why y is stored as a vector of vec objects. However, we are assuming
// common covariates across biopsies.
void SpikeSlabModel::load_data(LongVector y, arma::vec y_counts,
                                     arma::mat X,
                                     std::vector<int> groups,
                                     int n_groups,
                                     bool intercept,
                                     std::vector<arma::mat> W){
  if(ctrl.verbose){
    std::cout << "Loading data..." << std::endl;
  }
  dat.X_mats.X = X;
  dat.groups.groups = groups;
  dat.groups.n_groups = n_groups;
  if(intercept)
    dat.data_dim.intercept = 1;
  else
    dat.data_dim.intercept = 0;
  dat.X_mats.X_T_R_inv_X = X.t() * arma::diagmat(y_counts % y_counts) * X;
  dat.X_mats.X_T_R_inv = X.t() * arma::diagmat(y_counts % y_counts);
  dat.outcome.tile_counts = y_counts;
  dat.data_dim.N = y_counts.n_elem;
  dat.data_dim.p = X.n_cols;
  dat.W = W;
  dat.PSI_log_det = arma::mat(dat.data_dim.N, prior.delta.rho_prior.values.size(), arma::fill::zeros);
  dat.PSI_det = arma::mat(dat.data_dim.N, prior.delta.rho_prior.values.size(), arma::fill::zeros);
  for(int i = 0; i < dat.data_dim.N; i++){
    std::vector<arma::mat> M1;
    dat.Psi_decompositions.Psi_inv.push_back(M1);
    std::vector<arma::mat> M2;
    dat.Psi_decompositions.Psi.push_back(M2);
    std::vector<arma::mat> M3;
    dat.Psi_decompositions.Psi_inv_chol_inv.push_back(M3);
    std::vector<arma::mat> M4;
    dat.Psi_decompositions.Psi_chol_lower.push_back(M4);
    std::vector<arma::mat> M5;
    dat.Psi_decompositions.Psi_eigen_mats.push_back(M5);
    std::vector<arma::vec> M6;
    dat.Psi_decompositions.Psi_eigen_vals.push_back(M6);
    for(int j = 0; j < prior.delta.rho_prior.values.size(); j++){
      dat.Psi_decompositions
         .Psi_inv[i]
         .push_back((arma::diagmat(arma::sum(W[i], 1)) - prior.delta.rho_prior.values[j] * W[i]));
      dat.Psi_decompositions
         .Psi[i]
         .push_back(arma::inv(dat.Psi_decompositions.Psi_inv[i][j]));
      dat.Psi_decompositions
         .Psi_chol_lower[i]
         .push_back(arma::trimatl(arma::chol(dat.Psi_decompositions.Psi[i][j], "lower")));
      // Inverse of cholesky decomposition of precision matrix i j, stored as upper triangular
      // matrix for efficiency
      dat.Psi_decompositions
         .Psi_inv_chol_inv[i]
         .push_back(arma::trimatu(arma::inv(arma::chol(dat.Psi_decompositions.Psi_inv[i][j]))));

      dat.PSI_det(i, j) = arma::det(dat.Psi_decompositions.Psi[i][j]);
      dat.PSI_log_det(i, j) = log(dat.PSI_det(i, j));

      arma::mat Psi_cur_eigen_mat;
      arma::vec Psi_cur_eigen_vals;
      arma::eig_sym(Psi_cur_eigen_vals, Psi_cur_eigen_mat, dat.Psi_decompositions.Psi[i][j]);
      dat.Psi_decompositions.Psi_eigen_vals[i].push_back(Psi_cur_eigen_vals);
      dat.Psi_decompositions.Psi_eigen_mats[i].push_back(Psi_cur_eigen_mat);
    }
  }
  dat.outcome.Y = y;
  dat.outcome.y_sums = arma::vec(dat.outcome.Y.get_N(), arma::fill::zeros);
  for(int i = 0; i < dat.data_dim.N; i++){
    dat.outcome.y_sums(i) = y.get_sums(i);
  }
  dat.data_dim.S = arma::accu(y_counts);
  prop_vars.Psi_QFs = arma::mat(dat.data_dim.N, prior.delta.rho_prior.probabilities.n_elem, arma::fill::zeros);
}

void SpikeSlabModel::set_control(int sample, int burnin, int thinning,
                                       bool verbose, bool spike_optim,
                                       int update_spike_every,
                                       bool sample_delta, bool sample_gamma,
                                       bool sample_d){
  ctrl.sample = sample;
  ctrl.burnin = burnin;
  ctrl.thinning = thinning;
  ctrl.verbose = verbose;
  ctrl.spike_optim = spike_optim;
  ctrl.update_spike_every = update_spike_every;
  ctrl.sample_delta = sample_delta;
  ctrl.sample_gamma = sample_gamma;
  ctrl.sample_d = sample_d;
}

void SpikeSlabModel::samples_init(){
  samples.iter = 0;
  samples.mu = arma::mat(dat.X_mats.X.n_rows, ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.nu_2 = vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.alpha = arma::mat(dat.data_dim.p, ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.tau_2_delta = arma::vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.rho = arma::vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.psi_2 = arma::mat(dat.data_dim.p - dat.data_dim.intercept, ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.d = arma::mat(dat.data_dim.p - dat.data_dim.intercept, ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.gamma = arma::mat(dat.data_dim.p, ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.P_gamma = arma::vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.P_d = arma::vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
  samples.loglik = arma::vec(ctrl.sample / ctrl.thinning, arma::fill::zeros);
}

void SpikeSlabModel::sampler_init(){
    rand_samp.rand = Random(1);
    // std::cout << rand_samp.rand.gig(2.1, 0.1, 1.0) << std::endl;
}

void SpikeSlabModel::prior_init(double nu_2_alpha, double nu_2_beta,
                                      double tau_s_alpha,
                                      double sigma_2_alpha_spike,
                                      double sigma_2_alpha_slab,
                                      arma::vec gamma_prior,
                                      double P_gamma,
                                      double a_gamma,
                                      double b_gamma,
                                      arma::vec nu_j, double xi,
                                      std::vector<double> grid_prior_values,
                                      arma::vec rho_prior_probs,
                                      double tau_2_delta_alpha,
                                      double tau_2_delta_beta,
                                      double psi_2_eta_alpha, double psi_2_eta_beta,
                                      double sigma_2_j_slab, double sigma_2_j_spike,
                                      double prior_d,
                                      double P_d,
                                      double a_d,
                                      double b_d,
                                      double phi, double correction,
                                      std::vector<arma::mat> W, int N){
  // Nu_2
  prior.nu_2.nu_2_alpha = nu_2_alpha;
  prior.nu_2.nu_2_beta = nu_2_beta;

  // alpha
  prior.alpha.tau_s_alpha = tau_s_alpha;
  prop_paras.alpha.P_gamma = P_gamma;
  cur_paras.alpha.P_gamma = P_gamma;
  prior.alpha.a_gamma = a_gamma;
  prior.alpha.b_gamma = b_gamma;
  prior.alpha.sigma_2_alpha_spike = sigma_2_alpha_spike;
  prior.alpha.sigma_2_alpha_slab = sigma_2_alpha_slab;
  prior.alpha.nu_j = nu_j;
  prior.alpha.xi = xi;

  // delta
  prior.delta.rho_prior.values = grid_prior_values;
  prior.delta.rho_prior.probabilities = rho_prior_probs;
  prior.delta.tau_2_delta_alpha = tau_2_delta_alpha;
  prior.delta.tau_2_delta_beta = tau_2_delta_beta;

  // eta
  prior.eta.sigma_2_j_slab = sigma_2_j_slab;
  prior.eta.sigma_2_j_spike = sigma_2_j_spike;
  prop_paras.eta.P_d = P_d;
  cur_paras.eta.P_d = P_d;
  prior.eta.a_d = a_d;
  prior.eta.b_d = b_d;

  // Sigma decompositions
  dat.Sigma_decompositions.Sigma_log_dets = arma::vec(N, arma::fill::zeros);
  for(int i = 0; i < N; i++){
    arma::mat Sigma_inv_cur = (1.0/correction)*(arma::diagmat(arma::sum(W[i], 1)) - phi * W[i]);
    arma::mat Sigma_cur = arma::inv(Sigma_inv_cur);
    dat.Sigma_decompositions
       .Sigma
       .push_back(Sigma_cur);
    dat.Sigma_decompositions
       .Sigma_inv
       .push_back(Sigma_inv_cur);
    dat.Sigma_decompositions
       .Sigma_inv_chol_inv
       .push_back(arma::trimatu(arma::inv(arma::chol(dat.Sigma_decompositions.Sigma_inv[i]))));
    dat.Sigma_decompositions
       .Sigma_chol_lower.push_back(arma::trimatl(arma::chol(dat.Sigma_decompositions.Sigma[i], "lower")));
    arma::vec Sigma_eigen_vals_cur;
    arma::mat Sigma_eigen_mat_cur;
    arma::eig_sym(Sigma_eigen_vals_cur, Sigma_eigen_mat_cur, Sigma_cur);
    dat.Sigma_decompositions
       .Sigma_eigen_vals
       .push_back(Sigma_eigen_vals_cur);
    dat.Sigma_decompositions
       .Sigma_eigen_mats
       .push_back(Sigma_eigen_mat_cur);

    dat.Sigma_decompositions.Sigma_log_dets(i) = log(arma::det(Sigma_cur));
  }
}

void SpikeSlabModel::initial_values_init(double nu_2,
                                               arma::vec alpha,
                                               arma::vec lambda_alpha,
                                               LongVector delta,
                                               std::vector<arma::mat> eta,
                                               int rho,
                                               double tau_2_delta,
                                               arma::vec psi_2,
                                               arma::vec d,
                                               arma::vec gamma,
                                               arma::vec d_groups_init,
                                               arma::vec gamma_groups_init){

  cur_paras.nu_2 = nu_2;
  prop_paras.nu_2 = nu_2;

  cur_paras.alpha.alpha = alpha;
  prop_paras.alpha.alpha = alpha;

  cur_paras.alpha.lambda_alpha = lambda_alpha;
  prop_paras.alpha.lambda_alpha = lambda_alpha;

  cur_paras.delta.delta = delta;
  prop_paras.delta.delta = delta;

  cur_paras.eta.eta = eta;
  prop_paras.eta.eta = eta;

  cur_paras.delta.rho = rho;
  prop_paras.delta.rho = rho;

  cur_paras.delta.tau_2_delta = tau_2_delta;
  prop_paras.delta.tau_2_delta = tau_2_delta;

  cur_paras.eta.psi_2 = psi_2;
  prop_paras.eta.psi_2 = psi_2;

  cur_paras.eta.d = d;
  prop_paras.eta.d = d;

  cur_paras.alpha.gamma = gamma;
  prop_paras.alpha.gamma = gamma;

  cur_paras.eta.d_groups = d_groups_init;
  prop_paras.eta.d_groups = d_groups_init;

  cur_paras.alpha.gamma_groups = gamma_groups_init;
  prop_paras.alpha.gamma_groups = gamma_groups_init;
}

void SpikeSlabModel::residuals_init(){
  prop_vars.residuals = dat.outcome.Y;
  for(int i = 0; i < prop_vars.residuals.get_N(); i++){
    prop_vars.residuals[i] -= (as_scalar(dat.X_mats.X.row(i) * prop_paras.alpha.alpha) + prop_paras.delta.delta[i] +
                              prop_paras.eta.eta[i] *
                              dat.X_mats.X.submat(i, 0, i, dat.data_dim.p - dat.data_dim.intercept - 1).as_col());
  }
}

void SpikeSlabModel::chi_psi_init(){
  prop_vars.chi = arma::vec(dat.data_dim.p - dat.data_dim.intercept, arma::fill::zeros);
  prop_vars.psi = arma::vec(dat.data_dim.p - dat.data_dim.intercept, arma::fill::zeros);
}

int SpikeSlabModel::get_iter(){
  return samples.iter;
}

void SpikeSlabModel::augment_iter(){
  samples.iter++;
}

void SpikeSlabModel::update_proposal_paras(){
  update_proposal_nu();
  update_proposal_alpha();
  if(ctrl.sample_delta)
    update_proposal_delta();
  update_proposal_eta();
  update_proposal_loglik();
}

void SpikeSlabModel::update_proposal_nu(){
  double total_square_sums = accu(prop_vars.residuals.full_vec() %
                                  prop_vars.residuals.full_vec());

  double n_obs = dat.data_dim.S;
  if(ctrl.method == 1){
    total_square_sums += arma::as_scalar(1.0/prior.alpha.tau_s_alpha *
                          prop_paras.alpha.alpha.t() *
                          ((1.0/prop_paras.alpha.lambda_alpha) % prop_paras.alpha.alpha));
    n_obs += dat.data_dim.p;
  }
  prop_paras.nu_2 = InverseGamma(prior.nu_2.nu_2_alpha + (n_obs / 2.0),
                                 prior.nu_2.nu_2_beta + (total_square_sums / 2.0));
}

void SpikeSlabModel::update_proposal_alpha(){
  arma::vec old_alpha = prop_paras.alpha.alpha;
  LongVector y_star = prop_vars.residuals;
  for(int i = 0; i < dat.data_dim.N; i++){
    y_star[i] += as_scalar(dat.X_mats.X.row(i) * prop_paras.alpha.alpha);
  }

  arma::vec y_bar = (1.0/dat.outcome.tile_counts) % y_star.get_sums();

  if(ctrl.alpha_prior == 0){
    arma::mat fc_cov = arma::inv((1.0 / prop_paras.nu_2) * dat.X_mats.X_T_R_inv_X +
                       arma::diagmat(arma::vec(dat.data_dim.p,
                                               arma::fill::value(1.0/prior.alpha.tau_s_alpha))));

    arma::vec fc_mean = fc_cov * ((1.0/prop_paras.nu_2) * dat.X_mats.X_T_R_inv * y_bar);

    prop_paras.alpha.alpha = arma::mvnrnd(fc_mean, fc_cov);
  }else if(ctrl.alpha_prior == 1){
    arma::mat fc_cov = arma::inv((1.0 / prop_paras.nu_2) * dat.X_mats.X_T_R_inv_X +
                                  arma::diagmat((1.0/prop_paras.alpha.lambda_alpha)* (1.0/prior.alpha.tau_s_alpha)));

    arma::vec fc_mean = fc_cov * ((1.0/prop_paras.nu_2) * dat.X_mats.X_T_R_inv * y_bar);

    prop_paras.alpha.alpha = arma::mvnrnd(fc_mean, fc_cov);

    update_proposal_lambdas();
    update_proposal_tau_alpha();
    update_proposal_nu_j();
    update_proposal_xi();
  }else if(ctrl.alpha_prior == 2){
    arma::uvec alpha_non_zero = arma::find(prop_paras.alpha.gamma != 0);
    arma::uvec alpha_zero = arma::find(prop_paras.alpha.gamma == 0);

    arma::vec alpha_prior_var(dat.data_dim.p, arma::fill::zeros);

    alpha_prior_var.elem(alpha_zero).fill(prior.alpha.sigma_2_alpha_spike);
    alpha_prior_var.elem(alpha_zero).fill(prior.alpha.sigma_2_alpha_slab);

    arma::mat fc_cov = arma::inv((1.0 / prop_paras.nu_2) * dat.X_mats.X_T_R_inv_X +
                              arma::diagmat(alpha_prior_var));

    arma::vec fc_mean = fc_cov * ((1.0/prop_paras.nu_2) * dat.X_mats.X_T_R_inv * y_bar);

    prop_paras.alpha.alpha = arma::mvnrnd(fc_mean, fc_cov);
    if(ctrl.sample_gamma){
      update_proposal_gamma();
    }
  }else{
    std::invalid_argument("please select a valid prior for alpha");
  }
  update_proposal_residuals_alpha(old_alpha, prop_paras.alpha.alpha);
}

void SpikeSlabModel::update_proposal_gamma(){
  if(dat.groups.n_groups == 0){
    for(int j = 0; j < dat.data_dim.p; j++){
      double log_p_gamma_1 = arma::log_normpdf(prop_paras.alpha.alpha(j), 0.0,
                                              prior.alpha.sigma_2_alpha_slab) +
                                              log(prop_paras.alpha.P_gamma);

      double log_p_gamma_0 = arma::log_normpdf(prop_paras.alpha.alpha(j), 0.0,
                                              prior.alpha.sigma_2_alpha_spike) +
                                              log(1 - prop_paras.alpha.P_gamma);

      double normed_log_p_1 = log_p_gamma_1 - log(exp(log_p_gamma_1) + exp(log_p_gamma_0));

      if(log(arma::randu<double>()) < normed_log_p_1){
          prop_paras.alpha.gamma(j) = 1;
      }else{
          prop_paras.alpha.gamma(j) = 0;
      }
    }
  }else{
      arma::vec group_log_p1(dat.groups.n_groups, arma::fill::zeros);
      arma::vec group_log_p0(dat.groups.n_groups, arma::fill::zeros);
      // get the group contributions from each alpha
      for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
          double log_p_1 = arma::log_normpdf(prop_paras.alpha.alpha(j), 0.0, prior.alpha.sigma_2_alpha_slab);

          double log_p_0 = arma::log_normpdf(prop_paras.alpha.alpha(j), 0.0, prior.alpha.sigma_2_alpha_spike);

          group_log_p1(dat.groups.groups[j]) += log_p_1;
          group_log_p0(dat.groups.groups[j]) += log_p_0;
      }
      group_log_p1 += log(prop_paras.alpha.P_gamma);
      group_log_p0 += log(1 - prop_paras.alpha.P_gamma);
      arma::vec normed_log_group_p1 = group_log_p1 - arma::log(arma::exp(group_log_p1) + arma::exp(group_log_p0));
      // draw selection for each group
      for(int j = 0; j < normed_log_group_p1.n_elem; j++){
          if(log(arma::randu<double>()) < normed_log_group_p1(j)){
              prop_paras.alpha.gamma_groups(j) = 1;
          }else{
              prop_paras.alpha.gamma_groups(j) = 0;
          }
      }
      // set specific indicators
      for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
        prop_paras.alpha.gamma(j) = prop_paras.alpha.gamma_groups(dat.groups.groups[j]);
      }
    }

  update_proposal_P_gamma();
}

void SpikeSlabModel::update_proposal_P_gamma(){
  if(dat.groups.n_groups == 0){
    double posterior_a = prior.alpha.a_gamma + arma::accu(prop_paras.alpha.gamma);
    double posterior_b = prior.alpha.b_gamma + (prop_paras.alpha.gamma.n_elem - arma::accu(prop_paras.alpha.gamma));
    prop_paras.alpha.P_gamma = SampleBeta(posterior_a, posterior_b);
  }else{
    double posterior_a = prior.eta.a_d + arma::accu(prop_paras.alpha.gamma_groups);
    double posterior_b = prior.eta.b_d + (prop_paras.alpha.gamma_groups.n_elem -
                                          arma::accu(prop_paras.alpha.gamma_groups));
    prop_paras.alpha.P_gamma = SampleBeta(posterior_a, posterior_b);
  }
}

void SpikeSlabModel::update_proposal_residuals_alpha(arma::vec old_alpha, arma::vec new_alpha){
  for(int i = 0; i < dat.data_dim.N; i++){
    prop_vars.residuals[i] += as_scalar(dat.X_mats.X.row(i) * (old_alpha - new_alpha));
  }
}

void SpikeSlabModel::update_proposal_lambdas(){
  for(int j = 0; j < dat.data_dim.p; j++){
    prop_paras.alpha.lambda_alpha(j) = InverseGamma(1, 1.0/prior.alpha.nu_j(j) +
                                  (prop_paras.alpha.alpha(j) * prop_paras.alpha.alpha(j))/
                                  (2 * prior.alpha.tau_s_alpha * prop_paras.nu_2));
  }
}

void SpikeSlabModel::update_proposal_tau_alpha(){
  prior.alpha.tau_s_alpha = InverseGamma((dat.data_dim.p + 1) / 2.0,
                                    1.0/prior.alpha.xi +
                                    1.0/(2 * prop_paras.nu_2) *
                                    accu(prop_paras.alpha.alpha %
                                          prop_paras.alpha.alpha %
                                          (1.0/prop_paras.alpha.lambda_alpha)));
}

void SpikeSlabModel::update_proposal_nu_j(){
  for(int j = 0; j < dat.data_dim.p; j++){
    prior.alpha.nu_j(j) = InverseGamma(1, 1 + 1.0/prop_paras.alpha.lambda_alpha(j));
  }
}

void SpikeSlabModel::update_proposal_xi(){
  prior.alpha.xi = InverseGamma(1, 1 + 1.0/prior.alpha.tau_s_alpha);
}

double SpikeSlabModel::lambda_log_lik(arma::vec lambda, arma::vec alpha){
  arma::vec log_lik = -1*arma::log(lambda) -
                    0.5 * prior.alpha.tau_s_alpha * ((1.0/(lambda % lambda)) % (alpha % alpha)) -
                    - arma::log(1 + lambda % lambda);
  return arma::accu(log_lik);
}

void SpikeSlabModel::update_proposal_delta(){
  LongVector old_delta = prop_paras.delta.delta;
  for(int i = 0; i < dat.data_dim.N; i++){
    arma::vec y_star_i = prop_vars.residuals[i] + prop_paras.delta.delta[i];

    prop_paras.delta.delta[i] = QSMPS_decomp_2(dat.Psi_decompositions.Psi[i][prop_paras.delta.rho],
                                          prop_paras.nu_2,
                                          prop_paras.delta.tau_2_delta,
                                          dat.Psi_decompositions.Psi_chol_lower[i][prop_paras.delta.rho],
                                          dat.Psi_decompositions.Psi_eigen_vals[i][prop_paras.delta.rho],
                                          dat.Psi_decompositions.Psi_eigen_mats[i][prop_paras.delta.rho],
                                          y_star_i);
  }

  update_proposal_residuals_delta(old_delta, prop_paras.delta.delta);

  update_proposal_tau_2();
  update_proposal_rho();
}

void SpikeSlabModel::update_proposal_residuals_delta(LongVector old_delta, LongVector new_delta){
  prop_vars.residuals.full_vec() += (old_delta.full_vec() - new_delta.full_vec());
}

void SpikeSlabModel::update_proposal_rho() {
  arma::vec log_post_rho_probs(prior.delta.rho_prior.probabilities.size(), arma::fill::zeros);
  for(int j = 0; j < log_post_rho_probs.n_elem; j++){
    double quad_form_sum = 0;
    for(int i = 0; i < dat.data_dim.N; i++){
      double cur_qf = CholQF(dat.Psi_decompositions.Psi_inv_chol_inv[i][j], prop_paras.delta.delta[i]);
      prop_vars.Psi_QFs(i, j) = cur_qf;
      quad_form_sum += cur_qf;
    }
    // start on exponential scale to compute normalizing constant
    log_post_rho_probs(j) = -0.5 * accu(dat.PSI_log_det.col(j)) +
                            (-1.0/(2.0 * prop_paras.delta.tau_2_delta)) * quad_form_sum +
                            log(prior.delta.rho_prior.probabilities[j]);
  }
  prop_paras.delta.rho = (int) GumbelMax(log_post_rho_probs);
}

void SpikeSlabModel::update_proposal_tau_2() {
  double quad_form_sum = 0;
  // double qfs_debug = 0;
  for(int i = 0; i < dat.data_dim.N; i++){
    quad_form_sum += CholQF(dat.Psi_decompositions.Psi_inv_chol_inv[i][prop_paras.delta.rho], prop_paras.delta.delta[i]);
  }
  prop_paras.delta.tau_2_delta = InverseGamma(prior.delta.tau_2_delta_alpha + dat.data_dim.S / 2.0,
                                              prior.delta.tau_2_delta_beta + quad_form_sum / 2.0);
}


void SpikeSlabModel::update_proposal_eta(){
    std::vector<arma::mat> old_eta = prop_paras.eta.eta;
    for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
        for(int i = 0; i < dat.data_dim.N; i++){
            // TODO: think about case where X_ij = 0
            // Whether or not to sample this eta_ij
            bool sample_eta_ij = (prop_paras.eta.d(j) == 1 ||
                                  (!ctrl.spike_optim) ||
                                  ((samples.iter % ctrl.update_spike_every) == 0 )
                                  // prop_vars.d_change[j]
                                  ) &&
                                  dat.X_mats.X(i,j) != 0;
            if(sample_eta_ij){
                double X_ij = dat.X_mats.X(i,j);
                arma::vec y_ij_star = prop_vars.residuals[i] + X_ij * prop_paras.eta.eta[i].col(j);
                y_ij_star /= X_ij;

                prop_paras.eta.eta[i].col(j) = QSMPS_decomp_2(dat.Sigma_decompositions.Sigma[i],
                                                          prop_paras.nu_2 / (X_ij * X_ij),
                                                          prop_paras.eta.psi_2(j),
                                                          dat.Sigma_decompositions.Sigma_chol_lower[i],
                                                          dat.Sigma_decompositions.Sigma_eigen_vals[i],
                                                          dat.Sigma_decompositions.Sigma_eigen_mats[i],
                                                          y_ij_star);
            }else{
                // prop_paras.eta.eta[i].col(j) = arma::vec(dat.outcome.tile_counts(i), arma::fill::zeros);
                prop_paras.eta.eta[i].col(j) = cur_paras.eta.eta[i].col(j);
            }

            update_proposal_residuals_eta(old_eta, prop_paras.eta.eta, i, j);
        }
    }
    if(ctrl.sample_d){
      update_proposal_d();
    }
    update_proposal_psi_2();
}

void SpikeSlabModel::update_proposal_residuals_eta(std::vector<arma::mat>& old_eta,
                                                         std::vector<arma::mat>& new_eta,
                                                         int i, int j){
  prop_vars.residuals[i] += ((old_eta[i].col(j) - new_eta[i].col(j)) * dat.X_mats.X(i,j));
}

void SpikeSlabModel::update_proposal_d(){
    if(dat.groups.n_groups == 0){
      for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
          // double log_p_1 = arma::log_normpdf(prop_paras.eta.psi_2(j), 0.0, prior.eta.sigma_2_j_slab) +
          //                   log(prop_paras.eta.P_d);

          // double log_p_0 = arma::log_normpdf(prop_paras.eta.psi_2(j), 0.0, prior.eta.sigma_2_j_spike) +
          //                   log(1 - prop_paras.eta.P_d);
          double log_p_1 = arma::log_normpdf(sqrt(prop_paras.eta.psi_2(j)), 0.0, prior.eta.sigma_2_j_slab) +
                            log(prop_paras.eta.P_d);

          double log_p_0 = arma::log_normpdf(sqrt(prop_paras.eta.psi_2(j)), 0.0, prior.eta.sigma_2_j_spike) +
                            log(1 - prop_paras.eta.P_d);
          double normed_log_p_1 = log_p_1 - log(exp(log_p_1) + exp(log_p_0));


          if(log(arma::randu<double>()) < normed_log_p_1){
              prop_paras.eta.d(j) = 1;
          }else{
              prop_paras.eta.d(j) = 0;
          }

          prop_vars.d_change[j] = !(prop_paras.eta.d(j) == cur_paras.eta.d(j));
      }
    }else{
      arma::vec group_log_p1(dat.groups.n_groups, arma::fill::zeros);
      arma::vec group_log_p0(dat.groups.n_groups, arma::fill::zeros);
      // get the group contributions from each psi_j^2
      for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
          double log_p_1 = arma::log_normpdf(prop_paras.eta.psi_2(j), 0.0, prior.eta.sigma_2_j_slab);

          double log_p_0 = arma::log_normpdf(prop_paras.eta.psi_2(j), 0.0, prior.eta.sigma_2_j_spike);

          group_log_p1(dat.groups.groups[j]) += log_p_1;
          group_log_p0(dat.groups.groups[j]) += log_p_0;
      }
      // TODO
      // add in the prior probability of inclusion
      group_log_p1 += log(prop_paras.eta.P_d);
      group_log_p0 += log(1 - prop_paras.eta.P_d);
      arma::vec normed_log_group_p1 = group_log_p1 - arma::log(arma::exp(group_log_p1) + arma::exp(group_log_p0));
      // draw selection for each group
      for(int j = 0; j < normed_log_group_p1.n_elem; j++){
          if(log(arma::randu<double>()) < normed_log_group_p1(j)){
              prop_paras.eta.d_groups(j) = 1;
          }else{
              prop_paras.eta.d_groups(j) = 0;
          }
      }
      // set specific indicators
      for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
        prop_paras.eta.d(j) = prop_paras.eta.d_groups(dat.groups.groups[j]);
      }
    }

    update_proposal_P_d();
}

void SpikeSlabModel::update_proposal_P_d(){
  if(dat.groups.n_groups == 0){
    double posterior_a = prior.eta.a_d + arma::accu(prop_paras.eta.d);
    double posterior_b = prior.eta.b_d + (prop_paras.eta.d.n_elem - arma::accu(prop_paras.eta.d));
    prop_paras.eta.P_d = SampleBeta(posterior_a, posterior_b);
  }else{
    double posterior_a = prior.eta.a_d + arma::accu(prop_paras.eta.d_groups);
    double posterior_b = prior.eta.b_d + (prop_paras.eta.d_groups.n_elem - arma::accu(prop_paras.eta.d_groups));
    prop_paras.eta.P_d = SampleBeta(posterior_a, posterior_b);
  }
}

void SpikeSlabModel::update_proposal_psi_2(){
  for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
    double lambda = -(dat.data_dim.S)/2 + 1;
    // double lambda = -1 * (dat.data_dim.S/4) + 0.5;
    double chi;
    double psi;
    // if((samples.iter % ctrl.update_spike_every) != 0)
    if(prop_paras.eta.d(j) == 0 && ctrl.spike_optim && (samples.iter % ctrl.update_spike_every) != 0){
        chi = prop_vars.chi(j);
    }else{
        double quad_sum = 0;
        for(int i = 0; i < dat.data_dim.N; i++){
            quad_sum += CholQF(dat.Sigma_decompositions.Sigma_inv_chol_inv[i], prop_paras.eta.eta[i].col(j));
        }
        chi = quad_sum;
    }

    if(prop_paras.eta.d(j) == 0){
        psi = 1.0/prior.eta.sigma_2_j_spike;
    }else{
        psi = 1.0/prior.eta.sigma_2_j_slab;
    }

    prop_vars.chi(j) = chi;
    prop_vars.psi(j) = psi;
    prop_paras.eta.psi_2(j) = rand_samp.rand.gig(lambda, chi, psi);
  }
}

void SpikeSlabModel::update_proposal_loglik(){
  // TODO: add mvn density function
  double y_component = arma::accu(arma::log_normpdf(prop_vars.residuals.full_vec(), 0, sqrt(prop_paras.nu_2)));

  double eta_component = 0;
  if(ctrl.alpha_prior == 2){
    eta_component += (-0.5) * arma::accu(prop_vars.chi);
    for(int j = 0; j < dat.data_dim.p - dat.data_dim.intercept; j++){
        eta_component += -0.5 * arma::accu((dat.outcome.tile_counts * log(prop_paras.eta.psi_2(j))) +
                          dat.Sigma_decompositions.Sigma_log_dets);
    }
  }

  double psi_component = arma::accu(arma::log_normpdf(arma::sqrt(prop_paras.eta.psi_2),
                                                arma::vec(dat.data_dim.p - dat.data_dim.intercept, arma::fill::zeros),
                                                (prop_paras.eta.d * prior.eta.sigma_2_j_slab) +
                                                ((1 - prop_paras.eta.d) * prior.eta.sigma_2_j_spike)));

  double d_component = arma::accu(log(prop_paras.eta.P_d * prop_paras.eta.d +
                            (1 - prop_paras.eta.P_d) * (1 - prop_paras.eta.d)));

  double delta_component = 0;
  if(ctrl.sample_delta){
    delta_component += (-0.5) * arma::accu(prop_vars.Psi_QFs.col(prop_paras.delta.rho));
    delta_component += -0.5 * arma::accu((dat.outcome.tile_counts * prop_paras.delta.tau_2_delta) +
                                dat.PSI_log_det.col(prop_paras.delta.rho));
  }

  double tau_2_component = 0;
  if(ctrl.sample_delta){
      tau_2_component = LogInverseGammaDensity(prop_paras.delta.tau_2_delta,
                                                  prior.delta.tau_2_delta_alpha,
                                                  prior.delta.tau_2_delta_beta);
  }

  double rho_component = 0;
  if(ctrl.sample_delta){
    rho_component = log(prior.delta.rho_prior.probabilities[prop_paras.delta.rho]);
  }

  double nu_2_component = LogInverseGammaDensity(prop_paras.nu_2,
                                                 prior.nu_2.nu_2_alpha,
                                                 prior.nu_2.nu_2_beta);

  double alpha_component = 0;

  double gamma_component = 0;

  if(ctrl.alpha_prior == 0){
    alpha_component =  arma::accu(arma::log_normpdf(prop_paras.alpha.alpha,
                            arma::vec(dat.data_dim.p, arma::fill::zeros),
                            prop_paras.nu_2 * prop_paras.alpha.lambda_alpha * prior.alpha.tau_s_alpha));
  }else if(ctrl.alpha_prior == 1){
    // TODO: implement likelihood for horseshoe
  }else{
    alpha_component = arma::accu(arma::log_normpdf(prop_paras.alpha.alpha,
                                                   arma::vec(dat.data_dim.p, arma::fill::zeros),
                                                   (prop_paras.alpha.gamma * prior.alpha.sigma_2_alpha_slab) +
                                                   ((1 - prop_paras.alpha.gamma) * prior.alpha.sigma_2_alpha_spike)));

    gamma_component = arma::accu(log(prop_paras.alpha.P_gamma * prop_paras.alpha.gamma +
                                    (1 - prop_paras.alpha.P_gamma) * (1 - prop_paras.alpha.gamma)));
  }


  double lambda_component = 0;
  double lambda_and_xi_component = 0;
  if(ctrl.alpha_prior == 1){
    for(int j = 0; j < dat.data_dim.p; j++){
      lambda_component += LogInverseGammaDensity(prop_paras.alpha.lambda_alpha(j),
                                               0.5, 1.0/prior.alpha.nu_j(j));
    }
    lambda_and_xi_component = LogInverseGammaDensity(prior.alpha.xi, 0.5, 1);
    for(int j = 0; j < dat.data_dim.p; j++){
      lambda_and_xi_component += LogInverseGammaDensity(prior.alpha.nu_j(j), 0.5, 1);
    }
  }

  double tau_2_alpha_component = 0;
  if(ctrl.alpha_prior == 0){
    tau_2_alpha_component += LogInverseGammaDensity(prior.alpha.tau_s_alpha, 0.5, 1.0/prior.alpha.xi);
  }

  prop_paras.loglik = y_component + eta_component
                      + psi_component + d_component + delta_component
                      + tau_2_component + rho_component + nu_2_component +
                      alpha_component + lambda_component + tau_2_alpha_component +
                      lambda_and_xi_component + gamma_component;
}

void SpikeSlabModel::update_current_paras(){
  if(ctrl.method == 0){
    cur_paras.nu_2 = prop_paras.nu_2;
    cur_paras.alpha = prop_paras.alpha;
    cur_paras.alpha.lambda_alpha = prop_paras.alpha.lambda_alpha;
    cur_paras.delta = prop_paras.delta;
    cur_paras.delta.rho = prop_paras.delta.rho;
    cur_paras.delta.tau_2_delta = prop_paras.delta.tau_2_delta;
    cur_paras.eta = prop_paras.eta;
    cur_paras.eta.psi_2 = prop_paras.eta.psi_2;
    cur_paras.eta.d = prop_paras.eta.d;
    cur_paras.alpha.gamma = prop_paras.alpha.gamma;
    cur_paras.eta.P_d = prop_paras.eta.P_d;
    cur_paras.alpha.P_gamma = prop_paras.alpha.P_gamma;
    cur_paras.eta.d_groups = prop_paras.eta.d_groups;
    cur_paras.alpha.gamma_groups = prop_paras.alpha.gamma_groups;
    cur_paras.loglik = prop_paras.loglik;
  }else{
    std::invalid_argument("currently, only gibbs sampler method is supported");
  }
}

void SpikeSlabModel::save_sample(){
  int n = (samples.iter - ctrl.burnin) / ctrl.thinning;
  samples.nu_2(n) = cur_paras.nu_2;
  samples.alpha.col(n) = cur_paras.alpha.alpha;
  samples.rho(n) = prior.delta.rho_prior.values[cur_paras.delta.rho];
  samples.tau_2_delta(n) = cur_paras.delta.tau_2_delta;
  samples.psi_2.col(n) = cur_paras.eta.psi_2;
  samples.d.col(n) = cur_paras.eta.d;
  samples.gamma.col(n) = cur_paras.alpha.gamma;
  samples.P_gamma(n) = cur_paras.alpha.P_gamma;
  samples.P_d(n) = cur_paras.eta.P_d;
  samples.loglik(n) = cur_paras.loglik;
}

List SpikeSlabModel::get_samples(){
  return List::create(
    Named("nu_2") = samples.nu_2,
    Named("alpha") = samples.alpha,
    Named("rho") = samples.rho,
    Named("tau_2_delta") = samples.tau_2_delta,
    Named("psi") = samples.psi_2,
    Named("d") = samples.d,
    Named("gamma") = samples.gamma,
    Named("P_gamma") = samples.P_gamma,
    Named("P_d") = samples.P_d,
    Named("loglik") = samples.loglik
  );
}

void SpikeSlabModel::d_change_init(){
  std::vector<bool> d_c(dat.data_dim.p, false);
  prop_vars.d_change = d_c;
}

//' @export
// [[Rcpp::export]]
List FitSSModel(arma::vec y, arma::vec y_counts,            // Data/input /////
              arma::mat X,
              std::vector<int> groups,
              int n_groups,
              String method,
              String alpha_prior,
              std::vector<arma::mat> W,
              double phi,
              double correction,
              bool intercept,
              int burnin, int samples, int thinning,        // Settings ///////
              bool verbose,
              int progress_every,
              bool spike_optim,
              int update_spike_every,
              bool sample_delta,
              bool sample_gamma,
              bool sample_d,
              double nu_2_init,                             // Initial values /
              arma::vec alpha_init, arma::vec delta_init,
              int rho_init,
              double tau_2_delta_init,
              std::vector<double> psi_2_init,
              std::vector<arma::mat> eta_init,
              arma::vec d_init, arma::vec gamma_init,
              arma::vec d_groups_init, arma::vec gamma_groups_init,
              double P_gamma,
              double P_d,
              double nu_2_alpha, double nu_2_beta,          // Prior params ///
              double tau_s_alpha,
              double sigma_2_alpha_spike,
              double sigma_2_alpha_slab,
              arma::vec gamma_prior,
              double a_gamma, double b_gamma,
              double a_d, double b_d,
              arma::vec nu_j, double xi,
              arma::vec lambda_alpha,
              std::vector<double> grid_prior_values,
              arma::vec rho_prior_probs,
              double tau_2_delta_alpha, double tau_2_delta_beta,
              double psi_2_alpha, double psi_2_beta,
              double sigma_2_j, double sigma_2_j_s,
              double prior_d){

  std::vector<std::vector<int>> indices;
  int cur_total = 0;
  for(int i = 0; i < y_counts.n_elem; i++){
    std::vector<int> cur_inds = {cur_total, cur_total + ((int)y_counts(i)) - 1};
    indices.push_back(cur_inds);
    cur_total += y_counts(i);
  }

  LongVector Y(y, indices, y_counts.size());

  LongVector delta_init_long(delta_init, indices, y_counts.size());

  std::vector<std::vector<int>> v_indices;
  for(int i = 0; i < y_counts.n_elem; i++){
    std::vector<int> cur_inds = {i* ((int) alpha_init.n_elem), (i+1)*((int) alpha_init.n_elem) - 1};
    v_indices.push_back(cur_inds);
    cur_total += y_counts(i);
  }

  // TODO: add non-default constructor that pre-allocates vectors of matrices
  SpikeSlabModel mod;

  mod.set_method(method);

  mod.set_alpha_prior(alpha_prior);

  mod.set_control(samples, burnin, thinning, verbose, spike_optim, update_spike_every,
                  sample_delta, sample_gamma, sample_d);

  mod.prior_init(nu_2_alpha, nu_2_beta,
                 tau_s_alpha,
                 sigma_2_alpha_spike,
                 sigma_2_alpha_slab,
                 gamma_prior,
                 P_gamma,
                 a_gamma,
                 b_gamma,
                 nu_j, xi,
                 grid_prior_values,
                 rho_prior_probs,
                 tau_2_delta_alpha, tau_2_delta_beta,
                 psi_2_alpha, psi_2_beta,
                 sigma_2_j, sigma_2_j_s,
                 prior_d,
                 P_d,
                 a_d,
                 b_d,
                 phi, correction,
                 W, y_counts.size());

  mod.initial_values_init(nu_2_init, alpha_init,
                          lambda_alpha,
                          delta_init_long,
                          eta_init,
                          rho_init, tau_2_delta_init,
                          psi_2_init,
                          d_init, gamma_init,
                          d_groups_init, gamma_groups_init);

  mod.load_data(Y, y_counts, X, groups, n_groups, intercept, W);

  mod.residuals_init();


  mod.chi_psi_init();

  mod.samples_init();

  mod.sampler_init();

  mod.d_change_init();

  std::vector<double> sec_per_ten((burnin + samples) / 10);

  double prev_time{};

  wall_clock timer;
  timer.tic();
  if(verbose){
    std::cout << "Sampling..." << std::endl;
  }
  while(mod.get_iter() < burnin + samples){
    if((mod.get_iter() + 1) % 10 == 0){
      sec_per_ten[(mod.get_iter() + 1) / 10 - 1] = timer.toc();
    }
    if(verbose && (mod.get_iter() + 1) % progress_every == 0){
      std::cout << "Iteration " << mod.get_iter() + 1 <<
        " (" << timer.toc() - prev_time << " seconds)" << std::endl;
      // std::cout << "Residual difference rmse: " << mod.get_rmse_resid_diff() << std::endl;
      // std::cout << "Residual difference max: " << mod.get_max_resid_diff() << std::endl;
      prev_time = timer.toc();
    }
    mod.update_proposal_paras();
    mod.update_current_paras();
    if(mod.get_iter() >= burnin && (mod.get_iter() % thinning) == 0){
      mod.save_sample();
    }
    mod.augment_iter();
  }
  double elapsed = timer.toc();

  return List::create(
    Named("samples") = mod.get_samples(),
    Named("method") = method,
    Named("time") = elapsed,
    Named("sec_per_ten") = sec_per_ten
  );
}
