library(Rcpp)
library(SLOPE)
library(progress)

# Load gradient-step simulator
sourceCpp("ou_experiment.cpp")

# Helper: compute distances
compute_dists <- function(A_hat, A_true) {
  diff <- A_hat - A_true
  list(
    l2 = sqrt(sum(diff^2)),
    l1 = sum(abs(diff))
  )
}

# Build pseudo-data
build_pseudo_data <- function(x_train, delta) {
  dims <- dim(x_train)
  d <- dims[1]
  T_steps <- dims[2] - 1
  N <- dims[3]
  
  NT <- N * T_steps
  y_pseudo <- numeric(NT * d)
  X_pseudo <- matrix(0, nrow=NT*d, ncol=d*d)
  
  row_idx <- 1
  for (n in 1:N) {
    for (t in 1:T_steps) {
      xt0  <- x_train[,t,n]
      xt1  <- x_train[,t+1,n]
      dxt  <- xt1 - xt0
      y_pseudo[row_idx:(row_idx+d-1)] <- dxt
      X_pseudo[row_idx:(row_idx+d-1), ] <- delta * kronecker( diag(d), t(xt0))
      row_idx <- row_idx + d
    }
  }
  list(y=y_pseudo, X=X_pseudo)
}
# Unified coefficient extraction from SLOPE
get_A_from_SLOPE <- function(X, y, lambda_vec, d) {
  coef_raw <- SLOPE(X, y, lambda = lambda_vec,
                    center = FALSE, scale = FALSE, intercept = FALSE, alpha=1)$coefficients[[1]]
  coef_vec <- as.character(coef_raw)
  coef_num <- suppressWarnings(as.numeric(coef_vec))
  coef_num[is.na(coef_num)] <- 0
  matrix(coef_num, d, d, byrow = TRUE)
}

# Main function using only the SLOPE package
fit_ou_estimators <- function(x_train, A_true, delta, lambda0_grid, x_valid) {
  pseudo <- build_pseudo_data(x_train, delta)
  y_pseudo <- pseudo$y
  X_pseudo <- pseudo$X
  d <- dim(x_train)[1]
  
  # Slope weights
  weights_slope <- sqrt(log(2 * d^2 / seq_len(d*d)))
  
  ## MLE: lambda = 0 vector
  A_mle <- get_A_from_SLOPE(X_pseudo, y_pseudo, rep(0, d*d), d)
  loss_mle <- empirical_risk(A_mle, x_valid, delta)
  dists_mle <- compute_dists(A_mle, A_true)
  
  ## Lasso: grid search over uniform lambda0
  lasso_losses <- numeric(length(lambda0_grid))
  lasso_As <- vector("list", length(lambda0_grid))
  
  for (i in seq_along(lambda0_grid)) {
    lambda_vec <- lambda0_grid[i] * rep(1, d*d)
    A_lasso_tmp <- get_A_from_SLOPE(X_pseudo, y_pseudo, lambda_vec, d)
    lasso_As[[i]] <- A_lasso_tmp
    lasso_losses[i] <- empirical_risk(A_lasso_tmp, x_valid, delta)
  }
  idx_lasso_best <- which.min(lasso_losses)
  lambda_lasso_best <- lambda0_grid[idx_lasso_best]
  A_lasso <- lasso_As[[idx_lasso_best]]
  dists_lasso <- compute_dists(A_lasso, A_true)
  
  ## Slope: grid search over lambda0 * weights_slope
  slope_losses <- numeric(length(lambda0_grid))
  slope_As <- vector("list", length(lambda0_grid))
  
  for (i in seq_along(lambda0_grid)) {
    lambda_vec <- lambda0_grid[i] * weights_slope
    A_slope_tmp <- get_A_from_SLOPE(X_pseudo, y_pseudo, lambda_vec, d)
    slope_As[[i]] <- A_slope_tmp
    slope_losses[i] <- empirical_risk(A_slope_tmp, x_valid, delta)
  }
  idx_slope_best <- which.min(slope_losses)
  lambda_slope_best <- lambda0_grid[idx_slope_best]
  A_slope <- slope_As[[idx_slope_best]]
  dists_slope <- compute_dists(A_slope, A_true)
  # 
  ## Return unified result
  list(
    mle = list(loss = loss_mle, l2 = dists_mle$l2, l1 = dists_mle$l1),
    lasso = list(loss = lasso_losses[idx_lasso_best], lambda = lambda_lasso_best,
                 l2 = dists_lasso$l2, l1 = dists_lasso$l1),
    slope = list(loss = slope_losses[idx_slope_best], lambda = lambda_slope_best,
                 l2 = dists_slope$l2, l1 = dists_slope$l1)
  )
}



# Run all experiments
delta <- 0.01
T <- 1.0
N_train <- 400
N_valid <- 100
lambda_grid <- 10^seq(-8, -6, length.out=9)
dim_grid <- 5:25

results <- list()
pb <- progress_bar$new(format=":current/:total [:bar] :percent :elapsed", total=length(dim_grid)*10)

for (d in dim_grid) {
  cat(sprintf("\n=== d = %d ===\n", d))
  records_d <- list()
  
  for (iter in 1:10) {
    sim <- simulate_OU(d, N_train+N_valid, T, delta)
    A_true <- sim$A
    X <- sim$X
    
    
    x_train <- X[,,1:N_train]
    x_valid <- X[,,(N_train+1):(N_train+N_valid)]
    
    
    record <- fit_ou_estimators(x_train, A_true, delta, lambda_grid, x_valid)
    records_d[[iter]] <- record
    pb$tick()
  }
  results[[as.character(d)]] <- records_d
}

# Without Time Stamp
saveRDS(results, "ou_experiment.rds")
cat("\nResults saved\n")

# # With Time Stamp
# timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
# saveRDS(results, file = paste0("ou_experiment_", timestamp, ".rds"))
# cat("\nResults saved\n")
