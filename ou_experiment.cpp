#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

/*
 * Simulate N independent OU paths of dimension d over time [0, T] with step delta
 * Returns:
 *   A : true drift matrix (d x d)
 *   X : simulated paths (d x (steps+1) x N) as arma::cube
 */
// [[Rcpp::export]]
List simulate_OU(int d, int N, double T, double delta) {
  int steps = static_cast<int>(T / delta);
  
  arma::mat A(d, d, arma::fill::zeros);              // Drift matrix
  arma::cube X(d, steps + 1, N, arma::fill::zeros);  // Simulated paths
  
  
  // Construct sparse random drift matrix A
  for (int i = 0; i < d; ++i) {
    A(i, i) = R::runif(-1.0, 1.0);
    for (int j = 0; j < d; ++j) {
      if (i != j) {
        A(i, j) = (R::runif(0.0, 1.0) < 0.8) ? 0.0 : R::runif(-0.5, 0.5);
      }
    }
  }
  
  // Simulate N independent OU paths
  for (int n = 0; n < N; ++n) {
    arma::mat path(d, steps + 1, arma::fill::zeros);  // Each path: (d × T+1)
    
    for (int t = 0; t < steps; ++t) {
      arma::vec xt = path.col(t);
      arma::vec drift = A * xt;
      arma::vec noise = arma::randn<arma::vec>(d) * std::sqrt(delta);
      path.col(t + 1) = xt + delta * drift + noise;
    }
    
    X.slice(n) = path;
  }
  
  return List::create(
    Named("A") = A,
    Named("X") = X
  );
}


/*
 * Compute empirical negative log-likelihood (risk) from paths and estimated A
 *   A     : estimated drift matrix (d x d)
 *   X     : paths (d x (T+1) x N) as arma::cube
 *   delta : time step size
 */
// [[Rcpp::export]]
double empirical_risk(const arma::mat& A, const arma::cube& X, double delta) {
  int d = X.n_rows;
  int T = X.n_cols - 1;
  int N = X.n_slices;
  
  double loss = 0.0;
  
  for (int n = 0; n < N; ++n) {
    const arma::mat& path = X.slice(n);  // d × (T+1)
    
    for (int t = 0; t < T; ++t) {
      arma::vec x_t   = path.col(t);
      arma::vec x_tp1 = path.col(t + 1);
      arma::vec dx    = x_tp1 - x_t;
      arma::vec Ax    = A * x_t;
      
      loss += -arma::dot(Ax, dx) + 0.5 * arma::dot(Ax, Ax) * delta;
    }
  }
  
  return loss / N;
}
