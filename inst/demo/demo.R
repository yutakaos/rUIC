#------------------------------------------------------------------------------------------#
# Unified information-theoretic causality (UIC)
# Yutaka Osada, Masayuki Ushio
#
# Demonstration:
#     rUIC::simplex()
#     rUIC::xmap()
#     rUIC::uic()
#     rUIC::uic.optimal()
#------------------------------------------------------------------------------------------#

# Install package
library(remotes)
remotes::install_github("yutakaos/rUIC")

# Load library
library(rUIC);    packageVersion("rUIC")    # 0.9.14
library(ggplot2); packageVersion("ggplot2") # 3.5.1
library(cowplot); packageVersion("cowplot") # 1.1.3

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
block <- data.frame(t=1:tl, x=x, y=y)

# No.1: Determine the optimal embedding dimension using simplex projection
# Univariate UIC-version simplex projection
simp_x <- simplex(block, lib_var="x", E=0:8, tau=1, tp=1, alpha=0.05)
simp_y <- simplex(block, lib_var="y", E=0:8, tau=1, tp=1, alpha=0.05)

# Multivariate UIC-version simplex projection
simp_xy <- simplex(block, lib_var="x", cond_var="y", E=0:8, tau=1, tp=1, alpha=0.05)
simp_yx <- simplex(block, lib_var="y", cond_var="x", E=0:8, tau=1, tp=1, alpha=0.05)

# Select the optimal embedding dimension
Exy <- with(simp_xy, max(c(0, E[pval < 0.05]))) + 1
Eyx <- with(simp_yx, max(c(0, E[pval < 0.05]))) + 1

# No.2: Cross-map
xmap_xy <- xmap(block, lib_var="x", tar_var="y", E=Exy, tau=1, tp=-1)
xmap_yx <- xmap(block, lib_var="y", tar_var="x", E=Eyx, tau=1, tp=-1)

# No.3: Compute UIC
uic_xy <- uic(block, lib_var="x", tar_var="y", E=Exy, tau=1, tp=-4:4)
uic_yx <- uic(block, lib_var="y", tar_var="x", E=Eyx, tau=1, tp=-4:4)

# No.4: Wrapper functions for computing UIC
# compute UIC using optimal embedding dimension (the same results as No.3)
uic_opt_xy <- uic.optimal(block, lib_var="x", tar_var="y", E=0:8, tau=1, tp=-4:4, sequential_test=TRUE)
uic_opt_yx <- uic.optimal(block, lib_var="y", tar_var="x", E=0:8, tau=1, tp=-4:4, sequential_test=TRUE)

# ------------------------- Visualize results -------------------------#
# Visualize time series
ts <- ggplot(block[100:200,]) +
    geom_line(aes(x=t, y=x), col="red3") +
    geom_line(aes(x=t, y=y), col="royalblue") +
    labs(x = "Time", y = "Value") + theme_classic()

# Visualize Simplex
g1_1 <- ggplot(simp_x, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ggtitle(expression("{"~x[t]~","~x[t-1]~", ...,"~x[t-(E-1)]~"}")) +
    labs(y ="RMSE") + theme_classic() + theme(legend.position = "none")
g1_2 <- ggplot(simp_y, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ggtitle(expression("{"~y[t]~","~y[t-1]~", ...,"~y[t-(E-1)]~"}")) +
    labs(y ="RMSE") + theme_classic() + theme(legend.position = "none")
g1_3 <- ggplot(simp_xy, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ggtitle(expression("{"~x[t]~","~x[t-1]~", ..."~x[t-(E-1)]~","~y[t]~"}")) +
    labs(y ="RMSE") + theme_classic() + theme(legend.position = "none")
g1_4 <- ggplot(simp_yx, aes(x = E, y = rmse)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    ggtitle(expression("{"~y[t]~","~y[t-1]~", ..."~y[t-(E-1)]~","~x[t]~"}")) +
    labs(y ="RMSE") + theme_classic() + theme(legend.position = "none")

g1 = plot_grid(g1_1, g1_2, g1_3, g1_4, nrow = 2, align = "hv"); g1

# Visualize cross-map
g2_1 <- ggplot(xmap_xy, aes(x = data, y = pred)) +
    xlim(0.2, 0.9) + ylim(0.2, 0.9) +
    geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
    geom_point(alpha = 0.7) +
    ggtitle(expression("x cross-map y (y cause x?)")) +
    labs(x = "Observed", y = "Predicted") + theme_classic()
g2_2 <- ggplot(xmap_yx, aes(x = data, y = pred)) +
    xlim(0.18, 1) + ylim(0.18, 1) + 
    geom_abline(intercept = 0, slope = 1, linetype = 2, color = "red3") +
    geom_point(alpha = 0.7) +
    ggtitle(expression("y cross-map x (x cause y?)")) +
    labs(x = "Observed", y = "Predicted") + theme_classic()

g2 = plot_grid(g2_1, g2_2, nrow = 1, align = "hv"); g2

# Visualize UIC
g3_1 <- ggplot(uic_xy, aes(x = tp, y = ete)) +
    geom_line() + geom_point(aes(color = pval < 0.05), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.021, 1.50) +
    ggtitle(expression("UIC (y cause x?)")) +
    labs(x = "tp", y = "Effective TE") +
    theme_classic() + theme(legend.position = "none")
g3_2 <- ggplot(uic_yx, aes(x = tp, y = ete)) +
    geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
    geom_line() + geom_point(aes(color = pval >= 0.05), size = 2) +
    scale_color_manual(values = c("red3", "black")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.021, 1.50) +
    ggtitle(expression("UIC (x cause y?)")) +
    labs(x = "tp", y = "Effective TE") +
    theme_classic() + theme(legend.position = "none")

g3 = plot_grid(g3_1, g3_2, nrow = 1, align = "hv"); g3

# Visualize UIC computed by wrapper functions
g4_1 <- ggplot(uic_opt_xy, aes(x = tp, y = ete)) +
    geom_line() + geom_point(aes(color = seq_test <= 0), size = 2) +
    scale_color_manual(values = c("black", "red3")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.021, 1.50) +
    ggtitle(expression("optimal UIC (y cause x?)")) +
    labs(x = "tp", y = "Effective TE") +
    theme_classic() + theme(legend.position = "none")
g4_2 <- ggplot(uic_opt_yx, aes(x = tp, y = ete)) +
    geom_vline(xintercept = -1, size = 3, alpha = 0.2) +
    geom_line() + geom_point(aes(color = seq_test <= 0), size = 2) +
    scale_color_manual(values = c("red3", "black")) +
    scale_x_continuous(breaks = -4:5) + ylim(-0.021, 1.50) +
    ggtitle(expression("optimal UIC (x cause y?)")) +
    labs(x = "tp", y = "Effective TE") +
    theme_classic() + theme(legend.position = "none")

g4 = plot_grid(g4_1, g4_2, nrow = 1, align = "hv"); g4

# Save figures
ggsave("demo_figures/time_series.png" , plot = ts, width = 8, height = 3)
ggsave("demo_figures/simplex_rmse.png", plot = g1, width = 7, height = 6)
ggsave("demo_figures/xmap.png"        , plot = g2, width = 8, height = 4)
ggsave("demo_figures/uic.png"         , plot = g3, width = 8, height = 3.5)
ggsave("demo_figures/uic_wrapper.png" , plot = g4, width = 7, height = 3)

# End