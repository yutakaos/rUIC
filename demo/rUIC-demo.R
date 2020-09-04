#------------------------------------------------------------------------------------------#
# Unified information-theoretic causality (UIC)
# Yutaka Osada, Masayuki Ushio, Michio Kondoh
#
# Demonstration:
#     rUIC::simplex()
#     rUIC::xmap()
#     rUIC::uic()
#     rUIC::marginal_uic()
#------------------------------------------------------------------------------------------#

# Load library
library(rUIC);    packageVersion("rUIC")    # 0.1.4
library(ggplot2); packageVersion("ggplot2") # 3.3.2
library(cowplot); packageVersion("cowplot") # 1.0.0
theme_set(theme_cowplot())

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

# No.1: Determine the optimal embedding dimension using simplex projection
## Univariate UIC-version simplex projection
simp_x <- rUIC::simplex(block, lib_var = "x", E = 1:8, tau = 1, tp = 1)
simp_y <- rUIC::simplex(block, lib_var = "y", E = 1:8, tau = 1, tp = 1)

## Multivariate UIC-version simplex projection
simp_xy <- rUIC::simplex(block, lib_var = "x", cond_var = "y", E = 1:8, tau = 1, tp = 1, Enull = "adaptive")
simp_yx <- rUIC::simplex(block, lib_var = "y", cond_var = "x", E = 1:8, tau = 1, tp = 1, Enull = "adaptive")

# Select the optimal embedding dimension
Exy <- with(simp_xy, max(c(0, E[pval < 0.05])))
Eyx <- with(simp_yx, max(c(0, E[pval < 0.05])))

# No.2: Cross-map
xmap_xy <- rUIC::xmap(block, lib_var = "x", tar_var = "y", E = Exy + 1, tau = 1, tp = -1)
xmap_yx <- rUIC::xmap(block, lib_var = "y", tar_var = "x", E = Eyx + 1, tau = 1, tp = -1)

# No.3: Compute UIC
uic_xy <- rUIC::uic(block, lib_var = "x", tar_var = "y", E = Exy + 1, tau = 1, tp = -4:4)
uic_yx <- rUIC::uic(block, lib_var = "y", tar_var = "x", E = Eyx + 1, tau = 1, tp = -4:4)

# No.4: Wrapper functions for computing UIC
## compute UIC using optimal embedding dimension (the same results of No.3)
uic_opt_xy <- rUIC::uic.optimal(block, lib_var = "x", tar_var = "y", E = 1:10, tau = 1, tp = -4:4)
uic_opt_yx <- rUIC::uic.optimal(block, lib_var = "y", tar_var = "x", E = 1:10, tau = 1, tp = -4:4)
## compute UIC marginalizing embedding dimension
uic_mar_xy <- rUIC::uic.marginal(block, lib_var = "x", tar_var = "y", E = 1:10, tau = 1, tp = -4:4)
uic_mar_yx <- rUIC::uic.marginal(block, lib_var = "y", tar_var = "x", E = 1:10, tau = 1, tp = -4:4)

# ------------------------- Visualize results -------------------------#
# Visualize time series
ts <- ggplot(block[100:200,]) +
    geom_line(aes(x = t, y = x), col = "red3") +
    geom_line(aes(x = t, y = y), col = "royalblue") +
    xlab("Time") + ylab("Value")

# Visualize Simplex
g1 <- ggplot(simp_x, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ylab("RMSE") + theme(legend.position = "none") +
    ggtitle(expression("{"~x[t]~","~x[t-1]~", ...,"~x[t-E]~"}"))
g2 <- ggplot(simp_y, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ylab("RMSE") + theme(legend.position = "none") +
    ggtitle(expression("{"~y[t]~","~y[t-1]~", ...,"~y[t-E]~"}"))
g3 <- ggplot(simp_xy, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ylab("RMSE") + theme(legend.position = "none") +
    ggtitle(expression("{"~x[t]~","~x[t-1]~", ..."~x[t-E]~","~y[t]~"}"))
g4 <- ggplot(simp_yx, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ylab("RMSE") + theme(legend.position = "none") +
    ggtitle(expression("{"~y[t]~","~y[t-1]~", ..."~y[t-E]~","~x[t]~"}"))

# Visualize cross-map
g5 <- ggplot(xmap_xy$model_output, aes(x = data, y = pred)) +
    geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
    xlim(0.19, 0.9) + ylim(0.19, 0.9) +
    geom_point(alpha = 0.7) + xlab("Observed") + ylab("Predicted") +
    ggtitle(expression("x cross-map y (y cause x?)"))
g6 <- ggplot(xmap_yx$model_output, aes(x = data, y = pred)) +
    geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
    xlim(0.15, 1) + ylim(0.15, 1) + 
    geom_point(alpha = 0.7) + xlab("Observed") + ylab("Predicted") +
    ggtitle(expression("y cross-map x (x cause y?)"))

# Visualize UIC
g7 <- ggplot(uic_xy, aes(x = tp, y = te)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 1.6) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("UIC (y cause x?)")) +
    theme(legend.position = "none")
g8 <- ggplot(uic_yx, aes(x = tp, y = te)) +
    geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
    geom_line() + geom_point(aes(color = pval >= 0.05), size = 2) +
    scale_color_manual(values = c("red3", "black")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 1.6) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("UIC (x cause y?)")) +
    theme(legend.position = "none")

# Visualize UIC computed by wrapper functions
g9 <- ggplot(uic_opt_xy, aes(x = tp, y = te)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 1.6) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("optimal UIC (y cause x?)")) +
    theme(legend.position = "none")
g10 <- ggplot(uic_opt_yx, aes(x = tp, y = te)) +
    geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
    geom_line() + geom_point(aes(color = pval >= 0.05), size = 2) +
    scale_color_manual(values = c("red3", "black")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 1.6) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("optimal UIC (x cause y?)")) +
    theme(legend.position = "none")
g11 <- ggplot(uic_mar_xy, aes(x = tp, y = te)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 0.4) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("marginal UIC (y cause x?)")) +
    theme(legend.position = "none")
g12 <- ggplot(uic_mar_yx, aes(x = tp, y = te)) +
    geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.03, 0.4) +
    xlab("tp") + ylab("UIC") + ggtitle(expression("marginal UIC (x cause y?)")) +
    theme(legend.position = "none")

# Save figures
ggsave("demo_figures/time_series.png",
       plot = ts,
       width = 8, height = 3)

ggsave("demo_figures/simplex_rmse.png",
       plot = plot_grid(g1, g2, g3, g4, nrow = 2, align = "hv"),
       width = 7, height = 6)

ggsave("demo_figures/xmap.png",
       plot_grid(g5, g6, nrow = 1, align = "hv"),
       width = 8, height = 4)

ggsave("demo_figures/uic.png",
       plot_grid(g7, g8, nrow = 1, align = "hv"),
       width = 8, height = 3.5)

ggsave("demo_figures/uic_wrapper.png",
       plot_grid(g9, g10, g11, g12, nrow = 2, align = "hv"),
       width = 7, height = 6)
