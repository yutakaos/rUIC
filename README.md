# rUIC : Unified Information-theoretic Causality for R

ひとまずパッケージとして試用するためのサイトをつくりました！

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
block <- data.frame(t = 1:tl, x = x, y = y)

## xmap
par(mfrow = c(2, 2))
op0 <- xmap(block, lib_var = "x", tar_var = "y", E = 2, tau = 1, tp = -1)
op1 <- xmap(block, lib_var = "y", tar_var = "x", E = 2, tau = 1, tp = -1)
with(op0$model_output, plot(data, pred)); op0$stats
with(op1$model_output, plot(data, pred)); op1$stats

## simplex projection
op0 <- simplex(block, lib_var = "x", cond_var = "y", E = 1:8, tau = 1, tp = 1, n_boot = 2000)
op1 <- simplex(block, lib_var = "y", cond_var = "x", E = 1:8, tau = 1, tp = 1, n_boot = 2000)
with(op0, plot(E, te, type = "l"))
with(op0[op0$pval < 0.05,], points(E, te, pch = 16, col = "red"))
with(op1, plot(E, te, type = "l"))
with(op1[op1$pval < 0.05,], points(E, te, pch = 16, col = "red"))

## UIC
op0 <- uic(block, lib_var = "x", tar_var = "y", E = 3, tau = 1, tp = -4:0, n_boot = 2000)
op1 <- uic(block, lib_var = "y", tar_var = "x", E = 3, tau = 1, tp = -4:0, n_boot = 2000)
par(mfrow = c(2, 1))
with(op0, plot(tp, te, type = "l"))
with(op0[op0$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
with(op1, plot(tp, te, type = "l"))
with(op1[op1$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
``` 

## ruic で実装している関数

- `xmap`
  - モデルの予測結果と統計量を返します。予測結果がほしいときに使用.
  - `E`, `tau`, `tp`, `nn` はスカラーのみに対応.
  - `cond_var` を使うことで多変量にも対応.

- `simplex`
  - 統計量のみを返します.
  - `E`, `tau`, `tp` はベクターに対応し、ベクターの場合 `E`, `tau`, `tp` の全組み合わせを効率的に計算する.
  - UIC における `E` を推定するときは `cond_var` に原因となる変数を指定する.
  - `n_boot` > 1 以上で $p$ 値を返す.
  - `te` は次の数式で表される. $x$ は `lib_var`, $z$ は `cond_var`.
  $$latex
  \sum_{t} log p(x_{t+tp} | x_{t}, x_{t-&tau}, \ldots, x_{t-(E-1)*&tau}, z_{t}) -
           log p(x_{t+tp} | x_{t}, x_{t-&tau}, \ldots, x_{t-(E-2)*&tau}, z_{t})
  $$
  - $p$ 値の帰無仮説は te <= 0.

- `uic`
  - 統計量のみを返します.
  - `E`, `tau`, `tp` はベクターに対応し、ベクターの場合 `E`, `tau`, `tp` の全組み合わせを効率的に計算する.
  - `E` には simplex projection で得られた E に 1 を追加したものを指定.
  - `n_boot` > 1 以上で $p$ 値を返す.
  - `te` は次の数式で表される. $x$ は `lib_var`, $z$ は `cond_var`.
  $$latex
  \sum_{t} log p(y_{t+tp} | x_{t}, x_{t-&tau}, \ldots, x_{t-(E-1)*&tau}, z_{t}) -
           log p(y_{t+tp} |        x_{t-&tau}, \ldots, x_{t-(E-1)*&tau}, z_{t})
  $$
  - $p$ 値の帰無仮説は te <= 0.

## ruic で使われる引数

rEDM と同じ名前・同じ使い方の引数の説明は省略します.

- `lib_var` : 埋め込みに使われる変数, rEDM における lib_column に対応.
- `tar_var` : 予測に使われる変数, rEDM における target_column に対応.
- `cond_var` :  条件付きに使われる変数, 多変量予測や間接因果の推定に使う.
- `nn` : 埋め込みに使われる近傍数, rEDM における num_neighbors に対応.
  - "e+1" を使用可, スカラーの場合は nn = rep(nn, length(E)).
  - ベクトルの場合は length(E) == length(nn) でないとエラーを返す.
- `n_boot` : $p$値を計算するために必要なブートストラップ回数
- `scaling` : 距離行列の局所スケーリング手法.
  - ノイズ頑健になるといわれているため実装.
  - 検証した結果次第で、default は変更するかもしれない？
- `is_naive` : 近傍数によるの補正を行わない RMSE（ナイーブな推定量）を返すかどうか.
  - TRUE にすると CCM の結果に近いものになる.
  - 補正が必要なことが確かめられたら、将来的には引数から削除する予定？
