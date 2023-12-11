#include "RcppArmadillo.h"
#include "vector"
#include "math.h"
#include "gig.h"
#include <random>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

void PrintVec(arma::vec x);

void PrintVecSum(arma::vec x);

void PrintVec(arma::subview_col<double> x);

void PrintVec(arma::uvec x);

void PrintMatCol(arma::mat x, int c);

void GetMatColSum(arma::mat x, int c);

double InverseGamma(double alpha, double beta);

double SampleBeta(double a, double b);

double LogInverseGammaDensity(double x, double alpha, double beta);

double LogMVNDensity(arma::vec x, arma::vec mu, arma::mat Sigma);

double det_debug(arma::mat M);

// double GIG_density(double x, double lambda, double chi, double psi);

// double log_GIG_density(double x, double lambda, double chi, double psi);

arma::mat HPTMVN_0(arma::vec& mu, arma::mat& Sigma);

arma::vec QSMVN_1(arma::mat& Omega, arma::vec& mu);

arma::vec chol_sample(arma::mat& R_upper, arma::vec& mu);

arma::vec chol_sample(arma::mat& R_upper);

arma::vec chol_sample_zero_mean(arma::mat& cov_chol_lower);

// Lecture 9, slide 7, method 2
arma::vec QuickSampleMVNPosterior(arma::mat& A_inv, arma::mat& Phi, arma::mat& Omega_inv, arma::vec& mu);

// Lecture 9, slide 7, method 2
// Overloaded for case where Phi = I
arma::vec QuickSampleMVNPosteriorSimple(arma::mat& A_inv, arma::mat& Omega_inv, arma::vec& mu);

arma::vec QSMPS_decomp(arma::mat& A_inv, double nu_2, double tau_2,
                       arma::mat& A_inv_chol_lower,  arma::vec& mu);

arma::vec QSMPS_decomp_2(arma::mat& A_inv, double nu_2, double tau_2, 
                         arma::mat& A_inv_chol_lower,  
                         arma::vec& A_inv_eigen_vals,
                         arma::mat& A_inv_eigen_mat,
                         arma::vec& mu);

// Sample from Gumbel with location parameter 0
arma::vec Gumbel(int n);

double CholQF(arma::mat const& R_inv, arma::vec const& v);

// Implementation of the Gumbel-Max trick:
// https://laurent-dinh.github.io/2016/11/22/gumbel-max.html
uword GumbelMax(arma::vec non_norm_log_prob);

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