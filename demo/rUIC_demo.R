####
#### Unified information-theoretic causality (UIC)
#### Yutaka Osada, Masayuki Ushio, Michio Kondoh
####
#### Demonstration
####

# Install rUIC library
#library(devtools)
#devtools::install(pkg = 'ruic-master', reload = TRUE, quick = FALSE)

# Load library
library(ruic); packageVersion("ruic") # 0.1, 2020.4.12
library(ggplot2); packageVersion("ggplot2") # 3.3.0, 2020.4.12
library(cowplot); packageVersion("cowplot"); theme_set(theme_cowplot()) # 1.0.0, 2020.4.12

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
block = data.frame(t = 1:tl, x = x, y = y)

# Visualize time series
plot(block$x, type = "l", col = "royalblue")
lines(block$y, col = "red3")

# Step.1: Determine the optimal embedding dimension using simplex projection
## Univariate UIC-version simplex projection
simp_x <- ruic::simplex(block, x_column = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
simp_y <- ruic::simplex(block, x_column = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)

## Multivariate UIC-version simplex projection
simp_xy <- ruic::simplex(block, x_column = "x", y_column = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
simp_yx <- ruic::simplex(block, x_column = "y", y_column = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)

## Visualize results
g1 <- ggplot(simp_x, aes(x = E, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylab("UIC") + ggtitle("X") + theme(legend.position = "none")
g2 <- ggplot(simp_y, aes(x = E, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylab("UIC") + ggtitle("Y") + theme(legend.position = "none")
g3 <- ggplot(simp_xy, aes(x = E, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylab("UIC") + ggtitle("XY") + theme(legend.position = "none")
g4 <- ggplot(simp_yx, aes(x = E, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) +
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylab("UIC") + ggtitle("YX") + theme(legend.position = "none")

ggsave("demo_figures/simplex_uic.pdf",
       plot = plot_grid(g1, g2, g3, g4, nrow = 2),
       width = 7, height = 6)

# Select the optimal embedding dimension (RMSE or UIC?)
Exy <- simp_xy[which.min(simp_xy[simp_xy$pval < 0.05,]$rmse), "E"]
Eyx <- simp_yx[which.min(simp_yx[simp_yx$pval < 0.05,]$rmse), "E"]

# Cross-map
xmap_xy <- ruic::xmap(block, x_column = "x", y_column = "y", E = Exy, tau = 1, tp = -1)
xmap_yx <- ruic::xmap(block, x_column = "y", y_column = "x", E = Eyx, tau = 1, tp = -1)

# Visualize prediction
g5 <- ggplot(xmap_xy$model_output, aes(x = data, y = pred)) +
  geom_point() + xlab("Observed") + ylab("Predicted") +
  ggtitle("X cross-map Y (Y cause X?)")
g6 <- ggplot(xmap_yx$model_output, aes(x = data, y = pred)) +
  geom_point() + xlab("Observed") + ylab("Predicted") +
  ggtitle("Y cross-map X (X cause Y?)")

ggsave("demo_figures/xmap.pdf",
       plot_grid(g5, g6, nrow = 1),
       width = 8, height = 4)


# Compute UIC
uic_xy <- ruic::uic(block, x_column = "x", y_column = "y", E = Exy + 1, tau = 1, tp = -4:5, n_boot = 2000)
uic_yx <- ruic::uic(block, x_column = "y", y_column = "x", E = Eyx + 1, tau = 1, tp = -4:5, n_boot = 2000)

# Visualize prediction
g7 <- ggplot(uic_xy, aes(x = tp, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) + xlab("tp") + ylab("UIC") + 
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylim(-0.01, 0.4) + ggtitle("UIC X <- Y (Y cause X?)") + theme(legend.position = "none")
g8 <- ggplot(uic_yx, aes(x = tp, y = uic)) + geom_line() +
  geom_point(aes(color = pval < 0.05), size = 2) + xlab("tp") + ylab("UIC") +
  scale_color_manual(values = c("black", "red3")) + #ylim(0, 0.16) +
  ylim(-0.01, 0.4) + ggtitle("UIC Y <- X (X cause Y?)") + theme(legend.position = "none")

ggsave("demo_figures/uic.pdf",
       plot_grid(g7, g8, nrow = 1),
       width = 8, height = 4)
