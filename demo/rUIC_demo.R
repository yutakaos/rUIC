####
#### Unified information-theoretic causality (UIC)
#### Yutaka Osada, Masayuki Ushio, Michio Kondoh
####
#### Demonstration
####

# Load library
library(rUIC); packageVersion("rUIC") # 0.1.2
library(tidyverse); packageVersion("tidyverse") # 1.3.0
library(ggplot2); packageVersion("ggplot2") # 3.3.0
library(cowplot); packageVersion("cowplot"); theme_set(theme_cowplot()) # 1.0.0

# Create output directory
dir.create("demo_figures")

# Simulated logistic map
tl <- 400  # time length
x <- y <- rep(NA, tl)
x[1] <- 0.4
y[1] <- 0.2
for (t in 1:(tl - 1)) {  # causality : x -> y
  x[t+1] = x[t] * (3.8 - 3.8 * x[t] - 0.0 * y[t])
  y[t+1] = y[t] * (3.5 - 3.5 * y[t] - 0.1 * x[t])
}
block <- data.frame(t = 1:tl, x = x, y = y)

block_melted <- block %>% pivot_longer(col = c(x,y), names_to = "variable", values_to = "value")
ts <- ggplot(block_melted, aes(x = t, y = value, color = variable)) + geom_line() +
  scale_color_manual(values = c("red3", "royalblue")) + 
  ggtitle("Model time series") + ylab("Value") + xlab("Time") + xlim(100,200)

# No.1: Determine the optimal embedding dimension using simplex projection
## Univariate UIC-version simplex projection
simp_x <- rUIC::simplex(block, lib_var = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
simp_y <- rUIC::simplex(block, lib_var = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)

## Multivariate UIC-version simplex projection
simp_xy <- rUIC::simplex(block, lib_var = "x", cond_var = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
simp_yx <- rUIC::simplex(block, lib_var = "y", cond_var = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)

# Select the optimal embedding dimension (RMSE or UIC?)
Exy <- simp_xy[which.min(simp_xy[simp_xy$pval < 0.05,]$rmse), "E"]
Eyx <- simp_yx[which.min(simp_yx[simp_yx$pval < 0.05,]$rmse), "E"]

# No.2: Cross-map
xmap_xy <- rUIC::xmap(block, lib_var = "x", tar_var = "y", E = Exy, tau = 1, tp = -1)
xmap_yx <- rUIC::xmap(block, lib_var = "y", tar_var = "x", E = Eyx, tau = 1, tp = -1)

# No.3: Compute UIC
uic_xy <- rUIC::uic(block, lib_var = "x", tar_var = "y", E = Exy + 1, tau = 1, tp = -4:5, n_boot = 2000)
uic_yx <- rUIC::uic(block, lib_var = "y", tar_var = "x", E = Eyx + 1, tau = 1, tp = -4:5, n_boot = 2000)


# ------------------------- Visualize results -------------------------#
# Visualize Simplex
g1 <- ggplot(simp_x, aes(x = E, y = rmse)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) +
  ylab("RMSE") + theme(legend.position = "none") +
  ggtitle(expression("{"~X[t]~","~X[t-1]~", ...}"))
g2 <- ggplot(simp_y, aes(x = E, y = rmse)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) +
  ylab("RMSE") + theme(legend.position = "none") +
  ggtitle(expression("{"~Y[t]~","~Y[t-1]~", ...}"))
g3 <- ggplot(simp_xy, aes(x = E, y = rmse)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) +
  ylab("RMSE") + theme(legend.position = "none") +
  ggtitle(expression("{"~X[t]~","~X[t-1]~", ..."~Y[t]~"}"))
g4 <- ggplot(simp_yx, aes(x = E, y = rmse)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) +
  ylab("RMSE") + theme(legend.position = "none") +
  ggtitle(expression("{"~Y[t]~","~Y[t-1]~", ..."~X[t]~"}"))

# Visualize cross-map
g5 <- ggplot(xmap_xy$model_output, aes(x = data, y = pred)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
  xlim(0.19, 0.9) + ylim(0.19, 0.9) +
  geom_point(alpha = 0.7) + xlab("Observed") + ylab("Predicted") +
  ggtitle(expression("X cross-map Y (Y cause X?)"))
g6 <- ggplot(xmap_yx$model_output, aes(x = data, y = pred)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
  xlim(0.15, 1) + ylim(0.15, 1) + 
  geom_point(alpha = 0.7) + xlab("Observed") + ylab("Predicted") +
  ggtitle(expression("Y cross-map X (X cause Y?)"))

# Visualize UIC
g7 <- ggplot(uic_xy, aes(x = tp, y = te)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) + xlab("tp") + ylab("UIC") + 
  scale_color_manual(values = c("black", "red3")) +
  scale_x_continuous(breaks = -4:5) +
  ylim(-0.01, 0.4) + ggtitle(expression("UIC X" %<-% "Y (Y cause X?)")) + theme(legend.position = "none")
g8 <- ggplot(uic_yx, aes(x = tp, y = te)) + geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
  geom_line() + scale_x_continuous(breaks = -4:5) +
  geom_point(aes(color = pval < 0.05), size = 2) + xlab("tp") + ylab("UIC") +
  scale_color_manual(values = c("black", "red3")) +
  ylim(-0.01, 0.4) + ggtitle(expression("UIC Y" %<-% "X (X cause Y?)")) + theme(legend.position = "none")
  
# Save figures
ggsave("demo_figures/time_series.png",
       plot = ts,
       width = 8, height = 4)

ggsave("demo_figures/simplex_rmse.png",
       plot = plot_grid(g1, g2, g3, g4, nrow = 2, align = "hv"),
       width = 7, height = 6)

ggsave("demo_figures/xmap.png",
       plot_grid(g5, g6, nrow = 1, align = "hv"),
       width = 8, height = 4)

ggsave("demo_figures/uic.png",
       plot_grid(g7, g8, nrow = 1, align = "hv"),
       width = 8, height = 4)
