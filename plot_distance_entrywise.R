library(dplyr)
library(tidyr)
library(ggplot2)

library(RColorBrewer)
library(colorspace)  # for lighten()

# Load results
# results <- readRDS("ou_experiment.rds")

# Collect values including entrywise-normalized errors
df_all <- list()
for (d_str in names(results)) {
  d <- as.integer(d_str)
  recs <- results[[d_str]]
  for (iter in seq_along(recs)) {
    r <- recs[[iter]]
    for (method in c("mle", "lasso", "slope")) {
      df_all[[length(df_all) + 1]] <- data.frame(
        d = d,
        iter = iter,
        method = method,
        l2_entry = (r[[method]]$l2)^2 / (d^2),  # squared Frobenius per entry
        l1_entry = r[[method]]$l1 / (d^2)       # ℓ₁ per entry
      )
    }
  }
}
# --- Reshape and relabel ---

df <- bind_rows(df_all)
df_long <- df %>%
  pivot_longer(cols = c(l2_entry, l1_entry),
               names_to = "metric", values_to = "value")

df_long$method <- factor(df_long$method,
                         levels = c("mle", "lasso", "slope"),
                         labels = c("MLE", "Lasso", "Slope"))
df_long$metric <- factor(df_long$metric,
                         levels = c("l2_entry", "l1_entry"),
                         labels = c("Entrywise L2", "Entrywise L1"))

# --- Compute summaries ---

df_summary <- df_long %>%
  group_by(d, method, metric) %>%
  summarise(mean = mean(value), sd = sd(value), .groups = "drop")

# --- Plot : Entrywise Distances (2×3) ---


cols  <- brewer.pal(max(3, length(unique(df_summary$method))), "Set1")
fills <- lighten(cols, amount = 0.8)  # lighter, but still opaque (EPS-safe)

p_dist_2x3 <- ggplot(df_summary,
                     aes(x = d, y = mean, colour = method)) +
  geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = method),
              colour = NA) +          # <- remove alpha; default is fully opaque
  geom_line(linewidth = 1) +
  facet_grid(rows = vars(metric), cols = vars(method), scales = "free_y") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values = fills) +
  scale_y_continuous(limits = c(0, NA))

ggsave("plot_entrywise_distances_2x3.eps", p_dist_2x3,
       device = cairo_ps, width = 12, height = 6.5)
print(p_dist_2x3)
