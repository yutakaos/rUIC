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

## ruic で実装している関数

__xmap__  
　モデルの予測結果と統計量を返します。予測結果がほしいときに使用する。  
　E, tau, tp, nn はスカラーのみに対応。  
　z_column 引数を使うことで多変量にも対応。

__simplex__  
　統計量のみを返します。  
　E (+ nn), tau, tp はベクターに対応し、ベクターの場合 E, tau, tp の全組み合わせを計算ロスがないよう計算する。  
　y_column に原因となる変数（uic と同じ）を指定する。  
　n_boot > 1 以上で p 値を返す  
　p 値は次の不等式が成り立つ確率で,「埋め込み次元をひとつ減らした場合に比べて予測力が改善した確率」を表します。  
　　　p(x[t+tp] | y[t], x[t], x[t-tau], ... x[t-(E-1)*tau]) >  
　　　p(x[t+tp] | y[t], x[t], x[t-tau], ... x[t-(E-2)*tau])

$p(x[t+tp] | y[t], x[t], x[t-\tau], ... x[t-(E-1) \times \tau] > p(x[t+tp] | y[t], x[t], x[t-\tau], ... x[t-(E-2) \times \tau])$


　simplex projection において y_column と z_column は同じ役割を果たすため、現在 z_column は省略している  

__uic__  
　統計量のみを返します  
　E には simplex projection で得られた E に 1 を追加したものを指定  
　E (+ nn), tau, tp はベクターに対応し、ベクターの場合 E, tau, tp の全組み合わせを計算ロスがないよう計算する  
　n_boot > 1 以上で p 値を返す  
　p 値は次の不等式が成り立つ確率で,「Transfer Entropy の意味で y -> x の因果がある確率」を表します。  
　　　p(y[t+tp] | x[t+1], x[t], x[t-tau], ... x[t-(E-2)*tau]) >  
　　　p(y[t+tp] |         x[t], x[t-tau], ... x[t-(E-2)*tau])

## ruic で使われる引数

rEDM と同じ名前・同じ使い方の引数の説明は省略します。  

__x_column__ : the name or column index of library data  
　　⇒ 埋め込みに使われる変数, rEDM における lib_column に対応

__y_column__ : the name or column index of target data  
　　⇒ 予測に使われる変数, rEDM における target_column に対応

__z_column__ : the name or column index of condition data  
　　⇒ 条件付きに使われる変数, 多変量予測や間接因果の推定に使う

__nn__ : the number of neighbors  
　　⇒ rEDM における num_neighbors に対応  
　　⇒ "e+1" を使用可, スカラーの場合は nn = rep(nn, length(E+1))  
　　⇒ ベクトルの場合は length(E) == length(nn) でないとエラーを返す

__n_boot__ :  the number of bootstrap to be used for computing p-value  
　　⇒ p値を計算するために必要なブートストラップ回数

__scaling__ : the local scaling (neighbor, velocity, no_scale)  
　　⇒ 距離行列の局所スケーリング, ノイズ頑健になるといわれているため実装している  
　　⇒ 検証した結果次第で、default は変更するかもしれない？

__is_naive__ : whether rEDM-style estimator is used  
　　⇒ 近傍数によるの補正を行わない RMSE（ナイーブな推定量）を返すかどうか  
　　⇒ TRUE にすると CCM の結果に近いものになる  
　　⇒ 補正が必要なことが確かめられたら、将来的には引数から削除する予定？

