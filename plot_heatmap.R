# --- Visualization: Heatmap comparison for d = 15 ---

library(ggplot2)
library(SLOPE)
library(tidyr)
library(dplyr)
library(Rcpp)

sourceCpp("ou_experiment.cpp")
# Helper: reshape matrix to long data frame
as_heat_df <- function(mat, name) {
  as.data.frame(mat) %>%
    mutate(row = n():1) %>%
    pivot_longer(-row, names_to = "col", values_to = "value") %>%
    mutate(
      col = as.integer(gsub("V", "", col)),
      name = name
    )
}
# Simulate one example for d = 15
set.seed(42)
d <- 15
T = 1
delta = 0.01
N_train <- 400
N_valid <- 100
sim <- simulate_OU(d, N_train + N_valid, T, delta)
A_true <- sim$A
X <- sim$X
x_train <- X[,,1:N_train]
x_valid <- X[,,(N_train+1):(N_train+N_valid)]

# Fit all estimators
pseudo <- build_pseudo_data(x_train, delta)
y_pseudo <- pseudo$y
X_pseudo <- pseudo$X
weights_slope <- sqrt(log(2 * d^2 / seq_len(d*d)))

# MLE
A_mle <- get_A_from_SLOPE(X_pseudo, y_pseudo, rep(0, d*d), d)

# Lasso (best λ)
lasso_losses <- sapply(lambda_grid, function(lambda0) {
  empirical_risk(get_A_from_SLOPE(X_pseudo, y_pseudo, rep(lambda0, d*d), d), x_valid, delta)
})
A_lasso <- get_A_from_SLOPE(X_pseudo, y_pseudo, rep(lambda_grid[which.min(lasso_losses)], d*d), d)

# Slope (best λ)
slope_losses <- sapply(lambda_grid, function(lambda0) {
  empirical_risk(get_A_from_SLOPE(X_pseudo, y_pseudo, lambda0 * weights_slope, d), x_valid, delta)
})
A_slope <- get_A_from_SLOPE(X_pseudo, y_pseudo, lambda_grid[which.min(slope_losses)] * weights_slope, d)

# Combine into one data frame
df_heat <- bind_rows(
  as_heat_df(A_true, "True A"),
  as_heat_df(A_mle, "MLE"),
  as_heat_df(A_lasso, "Lasso"),
  as_heat_df(A_slope, "Slope")
)

# Shared color scale
vmax <- max(abs(df_heat$value))

# Plot
# Set factor level order for facet layout
df_heat$name <- factor(df_heat$name, levels = c("True A", "MLE", "Lasso", "Slope"))

# Shared color limits
vmax <- max(abs(df_heat$value))
library(scales)

# Final heatmap 2x2
p <- ggplot(df_heat, aes(x = col, y = row, fill = value)) +
  geom_tile(color = "gray50", linewidth = 0.2) +  # add borders to each tile
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-vmax, vmax),
                       trans = modulus_trans(p = 0.001),   # expands small |z|, keeps sign
                       oob = squish) +
  coord_fixed() +
  facet_wrap(~name, nrow = 2) +
  labs(
    # title = sprintf("Heatmap Comparison of Drift Matrix Estimates (d = %d)", d),
    x = NULL, y = NULL, fill = "Value"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1, "lines")
  )
p

ggsave(sprintf("heatmap_comparison_d%d_2x2_clean.eps",d), p, width = 8, height = 6)
eprint(p)

