#' Compute unified information-theoretic causality
#' 
#' \code{uic} returns model statistics computed from given multiple time series
#' based on cross mapping. This function simultaneously performs unified
#' information-theoretic causality computations for all possible combinations of
#' \code{E}, \code{tau} and \code{tp}.
#' 
#' @inheritParams xmap
#' @param E
#' the embedding dimension to use for time-delay embedding.
#' @param tau
#' the time-lag to use for time-delay embedding.
#' @param tp
#' the time index to predict.
#' @param nn
#' the number of nearest neighbors to use.
#' If \code{nn = "e+1"} (or \code{nn = -1}), \code{nn} is set to \code{E} + 1.
#' If \code{nn = 0}, \code{nn} is set to the number of all data.
#' Output \code{nn} is sometimes different from the specified (see Details section in \code{xmap}).
#' 
#' @details
#' Transfer entropy is computed as follows:
#' \deqn{
#' \sum_{t} log p(y_{t+tp} | x_{t}     , x_{t- \tau}, \ldots, x_{t-(E-1)\tau}, z_{t}) -
#'          log p(y_{t+tp} | x_{t-\tau}, x_{t-2\tau}, \ldots, x_{t-(E-1)\tau}, z_{t})
#' }
#' where \eqn{x} is library, \eqn{y} is target and \eqn{z} is condition.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E}      \tab \code{:} embedding dimension \cr
#' \code{tau}    \tab \code{:} time-lag \cr
#' \code{tp}     \tab \code{:} time prediction horizon \cr
#' \code{nn}     \tab \code{:} number of nearest neighbors \cr
#' \code{E_R}    \tab \code{:} embedding dimension of reference model \cr
#' \code{nn_R}   \tab \code{:} number of nearest neighbors of reference model \cr
#' \code{n_lib}  \tab \code{:} number of time indices used for attractor reconstruction \cr
#' \code{n_pred} \tab \code{:} number of time indices used for model predictions \cr
#' \code{rmse}   \tab \code{:} root mean squared error \cr
#' \code{te}     \tab \code{:} transfer entropy \cr
#' \code{pval}   \tab \code{:} p-value to test alternative hypothesis, te > 0 \cr
#' }
#' Note that the mumimum value of p-value is 0.001.
#' 
#' @seealso \link{xmap}, \link{simplex}
#' 
#' @examples
#' ## simulate logistic map
#' tl <- 400  # time length
#' x <- y <- rep(NA, tl)
#' x[1] <- 0.4
#' y[1] <- 0.2
#' for (t in 1:(tl - 1)) {  # causality : x -> y
#'     x[t+1] = x[t] * (3.8 - 3.8 * x[t] - 0.0 * y[t])
#'     y[t+1] = y[t] * (3.5 - 3.5 * y[t] - 0.1 * x[t])
#' }
#' block <- data.frame(t = 1:tl, x = x, y = y)
#' 
#' ## UIC
#' op0 <- uic(block, lib_var = "x", tar_var = "y", E = 2, tau = 1, tp = -4:4)
#' op1 <- uic(block, lib_var = "y", tar_var = "x", E = 2, tau = 1, tp = -4:4)
#' par(mfrow = c(2, 1))
#' with(op0, plot(tp, te, type = "l"))
#' with(op0[op0$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' with(op1, plot(tp, te, type = "l"))
#' with(op1[op1$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' 
uic = function (
    block, lib = c(1, NROW(block)), pred = lib,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1",
    scaling = c("no_scale", "neighbor", "velocity"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(tar_var) != 1) stop("The length of 'tar_var' must be 1.")
    lib  = rbind(lib)
    pred = rbind(pred)
    
    p = pmax(0, norm)
    NORM = 3  # Lp norm
    if      (norm == 2) NORM = 0  # L2 norm
    else if (norm == 1) NORM = 1  # L1 norm
    else if (norm <= 0) NORM = 2  # Max norm
    
    nn  = set_nn(nn, E)
    idx = order(E)
    E   = E [idx]
    nn  = nn[idx]
    if (is.null(exclusion_radius)) exclusion_radius = 0;
    if (is.null(epsilon)) epsilon = -1
    LS = switch(match.arg(scaling), "no_scale" = 0, "neighbor" = 1, "velocity" = 2)
    
    x = as.matrix(block[,lib_var])  # data.frame to matrix
    y = as.matrix(block[,tar_var])
    z = matrix()
    if (!is.null(cond_var)) z = as.matrix(block[,cond_var])
    
    uic = new(rUIC)
    uic$set_norm(NORM, LS, p, exclusion_radius, epsilon)
    uic$set_estimator(is_naive)
    op = uic$xmap_seq(x, y, z, lib, pred, E , nn, tau, tp)
    op[,which(!colnames(op) %in% "rmse_R")]
}

# End