# rUIC : Unified Information-theoretic Causality for R

Tentative manual of rUIC package.

## Installation

プライベートレポジトリを使用しているので、次のようにインストールします。
(インターネットを介する方式だと、personal access tokens を生成する必要があるため)

1. 「clone or download」をクリックして、zipファイルをダウンロード
2. 作業ディレクトリにzipファイルを解凍する（ruic-masterというフォルダができると思います）
3. ライブラリのインストール
``` r
library(devtools)
devtools::install(pkg = 'ruic-master', reload = TRUE, quick = FALSE)
``` 
4. 具体例の実践

``` r
## simulate logistic map
tl <- 400  # time length
x <- y <- rep(NA, tl)
x[1] <- 0.4
y[1] <- 0.2
for (t in 1:(tl - 1)) {  # causality : x -> y
    x[t+1] = x[t] * (3.8 - 3.8 * x[t] - 0.0 * y[t])
    y[t+1] = y[t] * (3.5 - 3.5 * y[t] - 0.1 * x[t])
}
block = data.frame(t = 1:tl, x = x, y = y)

## xmap
par(mfrow = c(2, 2))
op0 = xmap(block, x_column = "x", y_column = "y", E = 2, tau = 1, tp = -1)
op1 = xmap(block, x_column = "y", y_column = "x", E = 2, tau = 1, tp = -1)
with(op0$model_output, plot(data, pred)); op0$stats
with(op1$model_output, plot(data, pred)); op1$stats

## simplex projection
op0 = simplex(block, x_column = "x", y_column = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
op1 = simplex(block, x_column = "y", y_column = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
with(op0, plot(E, uic, type = "l"))
with(op0[op0$pval < 0.05,], points(E, uic, pch = 16, col = "red"))
with(op1, plot(E, uic, type = "l"))
with(op1[op1$pval < 0.05,], points(E, uic, pch = 16, col = "red"))

## UIC
op0 = uic(block, x_column = "x", y_column = "y", E = 3, tau = 1, tp = -4:0, n_boot = 2000)
op1 = uic(block, x_column = "y", y_column = "x", E = 3, tau = 1, tp = -4:0, n_boot = 2000)
par(mfrow = c(2, 1))
with(op0, plot(tp, uic, type = "l"))
with(op0[op0$pval < 0.05,], points(tp, uic, pch = 16, col = "red"))
with(op1, plot(tp, uic, type = "l"))
with(op1[op1$pval < 0.05,], points(tp, uic, pch = 16, col = "red"))
``` 

## Functions implemented in rUIC package

- `xmap`
　Perform cross-mapping and return model predictions and statistics.
    - `E`, `tau`, `tp`, and `nn` accept a scalar value only.
    - Potential causal variable should be specified by `y_column` augument.
    - Specify `z_column` augument for the multivariate version of `xmap`.

- `simplex`
　Perform simplex projection and return statistics only.
    - `E`, `tau`, `tp`, and `nn` accept vectors. All possible combinations of  `E`, `tau`, and `tp` are used.
    - Potential causal variable should be specified by `y_column` augument.
    - Return _p_ value if `n_boot > 1`.
    - _p_ value indicates "Probability of the improvements of prediction compared with when one less embedding dimension is used" as specified in the following inequality:
    **_p(x<sub>t+tp</sub> | y<sub>t</sub>, x<sub>t</sub>, x<sub>t-&tau;</sub>, ... x<sub>t-(E-1)&tau;</sub>) > p(x<sub>t+tp</sub> | y<sub>t</sub>, x<sub>t</sub>, x<sub>t-&tau;</sub>, ... x<sub>t-(E-2)&tau;</sub>)_**

- `uic`
　Perform uic and return statistics only.
    - `E` should be an optimal embedding dimension (estimated by `simplex`) + 1.
    - `E`, `tau`, `tp`, and `nn` accept vectors. All possible combinations of  `E`, `tau`, and `tp` are used.
    - Potential causal variable should be specified by `y_column` augument.
    - Return _p_ value if `n_boot > 1`.
    - _p_ value indicates "Probability that y causes x in the sense of transfer entropy" as specified in the following inequality:
    **_p(x<sub>t+tp</sub> | x<sub>t+1</sub>, x<sub>t</sub>, x<sub>t-&tau;</sub>, ... x<sub>t-(E-2)&tau;</sub>) > p(x<sub>t+tp</sub> |  x<sub>t</sub>, x<sub>t-&tau;</sub>, ... x<sub>t-(E-2)&tau;</sub>)_**

## Arguments in rUIC package

Arguments identical with those used in rEDM package are currently not explained. For arguments used in rEDM package, please see the rEDM tutorial (https://ha0ye.github.io/rEDM/index.html).

- `x_column` : the name or column index of library data
    - A variable to be used for time-delay embedding.

- `y_column` : the name or column index of target data
    - A variable to be predicted (`target_column` argument in rEDM).

- `z_column` : the name or column index of condition data
    - A third variable to be used for multivariate prediction, or detection of indirect interactions.

- `nn` : the number of neighbors used for prediction
    - `num_neighbors` argument in rEDM.
    - `"e+1"` may be used. If a scalar value is specified, nn = rep(nn, length(E+1)).
    - If vector is specified and if `length(E) != length(nn)`, error will be returned.

- `n_boot` :  the number of bootstrap to be used for computing p-value  
    - The number of bootstrap to calculate p value.

- `scaling` : the local scaling (neighbor, velocity, no_scale)
    - **This argument is experimental. May be changed in near future.**
    - Method for local scaling of distance matrix. Implemented to improve noise-robustness.

- `is_naive` : whether rEDM-style estimator is used
    - **This argument is experimental. May be changed in near future.**
    - Whether to return not-corrected RMSE （naive estimator） (estimator that is not corrected using neighbors)
    - If `TRUE`, the result will be similar to Convergent Cross Mapping (CCM)

