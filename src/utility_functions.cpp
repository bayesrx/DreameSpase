#include "RcppArmadillo.h"
#include "vector"
#include "math.h"
#include "gig.h"
#include <random>
// #include <boost/math/special_functions/bessel.hpp>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

class LongVector{
private:
  std::vector<std::vector<int>> indices;
  int N;
  arma::vec Y;
public:
  LongVector(){
    Y = {0};
    indices = {{0}};
    N = 0;
  }

  LongVector(arma::vec Y_v, std::vector<std::vector<int>> indices_v, int N_v){
    Y = Y_v;
    indices = indices_v;
    N = N_v;
  }

  arma::vec& full_vec(){
    return Y;
  }

  arma::subview_col<double> operator [](int i) {
    if(i < 0 || i >= N){
      std::cerr << "Error: index out of bounds on LongVector" << std::endl;
    }
    return Y.subvec(indices[i][0], indices[i][1]);
  }

  arma::vec get_sums(){
    arma::vec sum_vec(N, arma::fill::zeros);
    for(int i = 0; i < N; i++){
      sum_vec(i) = accu(Y.subvec(indices[i][0], indices[i][1]));
    }
    return sum_vec;
  }

  double get_sums(int c){
    return accu(Y.subvec(indices[c][0], indices[c][1]));
  }

  int get_N(){
    return N;
  }
};

class BDSparseMatrix {
  private:
    std::vector<std::vector<int>> indices;
    int N;
    arma::sp_mat M;
  public:
    // Default constructor
    BDSparseMatrix(){
      M = sp_mat();
      indices = {{0}};
      N = 0;
    }

    // 
    BDSparseMatrix(std::vector<arma::mat> M_v, std::vector<std::vector<int>> indices_v, int N, int sq_sum){
      arma::mat locations(2, sq_sum, arma::fill::zeros);

    }
};

void PrintVec(arma::vec x){
  for(int i = 0; i < x.n_elem; i++){
    std::cout << x(i) << " ";
  }
  std::cout << std::endl;
}

void PrintVec(arma::uvec x){
  for(int i = 0; i < x.n_elem; i++){
    std::cout << x(i) << " ";
  }
  std::cout << std::endl;
}

void PrintVecSum(arma::vec x){
  double s = accu(x);
  std::cout << s << std::endl;
}

void PrintVec(arma::subview_col<double> x){
  for(int i = 0; i < x.n_elem; i++){
    std::cout << x(i) << " ";
  }
  std::cout << std::endl;
}

void PrintFullVec(LongVector v){
  for(int i = 0; i < v.full_vec().n_elem; i++){
    std::cout << v.full_vec()(i) << " ";
  }
  std::cout << std::endl;
}

void PrintVecSum(LongVector v){
  std::cout << accu(v.full_vec()) << std::endl;
}

void PrintSqFullVec(LongVector v){
  for(int i = 0; i < v.full_vec().n_elem; i++){
    std::cout << v.full_vec()(i) * v.full_vec()(i) << " ";
  }
  std::cout << std::endl;
}

void PrintSqSumFullVec(LongVector v){
  std::cout << accu(v.full_vec() % v.full_vec()) << std::endl;
}

void CheckHasNaN(LongVector v){
  std::cout << v.full_vec().has_nan() << std::endl;
}

void WhichAreNaN(LongVector v){
  for(int i = 0; i < v.full_vec().n_elem; i++){
    if(isnan(v.full_vec()(i))){
      std::cout << i << ": " << v.full_vec()(i) << " -- ";
    }
  }
  std::cout << std::endl;
}

void PrintMatCol(arma::mat x, int c){
  arma::vec v = x.col(c);
  PrintVec(v);
}

void GetMatColSum(arma::mat x, int c){
  double sum = arma::accu(x.col(c));
  std::cout << sum << std::endl;
}

double InverseGamma(double alpha, double beta){
  return 1.0/randg(arma::distr_param(alpha, 1.0/beta));
}

double SampleBeta(double a, double b){
  double X = arma::randg(arma::distr_param(a, 1.0));
  double Y = arma::randg(arma::distr_param(b, 1.0));
  return X / (X + Y);
}

double LogInverseGammaDensity(double x, double alpha, double beta){
  return alpha * log(beta) - log(tgamma(alpha)) + (-1 * alpha - 1) * log(x) - beta / x;
}

double LogMVNDensity(arma::vec x, arma::vec mu, arma::mat Sigma){
  arma::vec x_minus_mu = x - mu;
  double val;
  double sign;
  bool ok = arma::log_det(val, sign, Sigma);
  double log_det_Sigma_inv = exp(val) * sign;
  return (-x.n_elem / 2.0) * log(6.28) -  0.5*log_det_Sigma_inv - 
          - 0.5 * as_scalar(x_minus_mu.t() * arma::inv(Sigma) * x_minus_mu);
}

double det_debug(arma::mat M){
  return arma::det(M);
}

arma::mat HPTMVN_0(arma::vec& mu, arma::mat& Sigma){
  // Initial sample
  arma::vec y = mvnrnd(mu, Sigma);
  // Because of special case, G Sigma G^T is just sum of elements of Sigma
  double G_Sigma_G_T = accu(Sigma);
  arma::vec Sigma_G_T = arma::sum(Sigma, 1);

  double alpha = ((-1) * accu(y)) / G_Sigma_G_T;
  y += (Sigma_G_T * alpha);

  return y;
}

arma::vec QSMVN_1(arma::mat& Omega, arma::vec& mu){
  arma::mat R_T = chol(Omega, "lower");
  arma::vec b = arma::solve(R_T, mu);
  arma::vec z = randn(mu.n_elem);
  arma::vec beta = arma::solve(R_T.t(), b + z);
  return beta;
}

arma::vec chol_sample(arma::mat& R_upper, arma::vec& mu){
  arma::vec b = arma::solve(R_upper.t(), mu);
  arma::vec z = randn(mu.n_elem);
  arma::vec beta = arma::solve(R_upper, b + z);
  return beta;
}

arma::vec chol_sample(arma::mat& R_upper){
  arma::vec mu(R_upper.n_rows, arma::fill::zeros);
  arma::vec b = arma::solve(R_upper.t(), mu);
  arma::vec z = randn(mu.n_elem);
  arma::vec beta = arma::solve(R_upper, b + z);
  return beta;
}

arma::vec chol_sample_zero_mean(arma::mat& cov_chol_lower){
  arma::vec norm_samp = arma::randn(cov_chol_lower.n_rows);
  return cov_chol_lower * norm_samp;
}

// Lecture 9, slide 7, method 2
arma::vec QuickSampleMVNPosterior(arma::mat& A_inv, arma::mat& Phi, arma::mat& Omega_inv, arma::vec& mu){
  arma::vec eta_1 = mvnrnd(arma::vec(A_inv.n_rows, arma::fill::zeros), A_inv);
  arma::vec eta_2 = mvnrnd(arma::vec(Omega_inv.n_rows, arma::fill::zeros), Omega_inv);

  arma::mat A_inv_Phi_T = A_inv * Phi.t();

  arma::mat M_1 = Omega_inv + Phi*A_inv_Phi_T;
  arma::vec V_1 = mu - Phi*eta_1 - eta_2;

  arma::vec alpha = arma::solve(M_1, V_1);

  arma::vec final_sample = eta_1 + A_inv_Phi_T * alpha;

  return final_sample;
}

// Lecture 9, slide 7, method 2
// Overloaded for case where Phi = I
arma::vec QuickSampleMVNPosteriorSimple(arma::mat& A_inv, arma::mat& Omega_inv, arma::vec& mu){
  arma::vec eta_1 = mvnrnd(arma::vec(A_inv.n_rows, arma::fill::zeros), A_inv);
  arma::vec eta_2 = mvnrnd(arma::vec(Omega_inv.n_rows, arma::fill::zeros), Omega_inv);

  arma::mat M_1 = Omega_inv + A_inv;
  arma::vec V_1 = mu - eta_1 - eta_2;

  arma::vec alpha = arma::solve(M_1, V_1, arma::solve_opts::fast + arma::solve_opts::likely_sympd);

  arma::vec final_sample = eta_1 + A_inv * alpha;

  return final_sample;
}

arma::vec QSMPS_decomp(arma::mat& A_inv, double nu_2, double tau_2, 
                         arma::mat& A_inv_chol_lower,  arma::vec& mu){
  
  // arma::vec eta_1 = mvnrnd(arma::vec(A_inv.n_rows, arma::fill::zeros), A_inv);
  arma::vec eta_1 = sqrt(tau_2) * chol_sample_zero_mean(A_inv_chol_lower);
  arma::vec eta_2 = sqrt(nu_2) * arma::randn(A_inv.n_rows);

  arma::mat M_1 = tau_2 * A_inv;

  M_1.diag() += nu_2;

  arma::vec V_1 = mu - eta_1 - eta_2;

  arma::vec alpha = arma::solve(M_1, V_1, arma::solve_opts::fast + arma::solve_opts::likely_sympd);

  arma::vec final_sample = eta_1 + tau_2 * A_inv * alpha;

  return final_sample;
}

arma::vec QSMPS_decomp_2(arma::mat& A_inv, double nu_2, double tau_2, 
                         arma::mat& A_inv_chol_lower,  
                         arma::vec& A_inv_eigen_vals,
                         arma::mat& A_inv_eigen_mat,
                         arma::vec& mu){
  
  // arma::vec eta_1 = mvnrnd(arma::vec(A_inv.n_rows, arma::fill::zeros), A_inv);
  arma::vec eta_1 = sqrt(tau_2) * chol_sample_zero_mean(A_inv_chol_lower);
  arma::vec eta_2 = sqrt(nu_2) * arma::randn(A_inv.n_rows);

  // arma::mat M_1 = tau_2 * A_inv;

  // M_1.diag() += nu_2;

  arma::vec V_1 = mu - eta_1 - eta_2;

  // arma::vec alpha = arma::solve(M_1, V_1, arma::solve_opts::fast + arma::solve_opts::likely_sympd);

  arma::vec alpha = A_inv_eigen_mat * ((1.0/(tau_2 * A_inv_eigen_vals + nu_2)) % (A_inv_eigen_mat.t() * V_1));

  arma::vec final_sample = eta_1 + tau_2 * A_inv * alpha;

  return final_sample;
}

// Sample from Gumbel with location parameter 0
arma::vec Gumbel(int n){
  arma::vec unif_sample(n, arma::fill::randu);
  arma::vec gumbel_sample = -log(-log(unif_sample));
  return gumbel_sample;
}

// Implementation of the Gumbel-Max trick:
// https://laurent-dinh.github.io/2016/11/22/gumbel-max.html
uword GumbelMax(arma::vec non_norm_log_prob){
  arma::vec shifted_gumbel_sample = Gumbel(non_norm_log_prob.n_elem) + non_norm_log_prob;
  return shifted_gumbel_sample.index_max();
}

double CholQF(arma::mat const& R_inv, arma::vec const& v){
  arma::vec q = arma::solve(R_inv, v);

  return arma::accu(q % q);
}