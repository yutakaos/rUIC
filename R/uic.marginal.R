#' Wrapper function for computing marginal unified information-theoretic causality
#' 
#' \code{uic.marginal} returns model statistics computed from given multiple time series
#' based on simplex projection and cross mapping. This function computes UIC using a model
#' averaging technique (i.e., marginalizing \code{E}). Thus, the users do not have to determine
#' the optimal \code{E} by themselves.
#' 
#' @details
#' \code{scaling} specifies the methods for local scaling of distance matrix.
#' The following distances can be used as local scaling factors:
#' the mean distances to nearest neighbors of the embedding space (\code{scaling = neighbor}),
#' the mean distances to nearest time indices (\code{scaling = velocity}) and
#' the constant distance (\code{scaling = no_scale}).
#' 
#' @inheritParams uic
#' @param nn
#' the number of nearest neighbors to use. Must be an integer or "e+1".
#' If \code{nn = "e+1"}, \code{nn} is set as \code{E} + 1.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E}      \tab \code{:} model-averaged embedding dimension \cr
#' \code{tau}    \tab \code{:} time-lag \cr
#' \code{tp}     \tab \code{:} time prediction horizon \cr
#' \code{nn}     \tab \code{:} model-averaged number of nearest neighbors \cr
#' \code{n_lib}  \tab \code{:} model-averaged number of time indices used for attractor reconstruction \cr
#' \code{n_pred} \tab \code{:} model-averaged number of time indices used for model predictions \cr
#' \code{rmse}   \tab \code{:} model-averaged root mean squared error \cr
#' \code{te}     \tab \code{:} model-averaged transfer entropy \cr
#' \code{pval}   \tab \code{:} bootstrap p-value to test alternative hypothesis, te > 0 \cr
#' }
#' 
#' @seealso \link{simplex}, \link{uic}
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
#' op0 <- uic.marginal(block, lib_var = "x", tar_var = "y", E = 0:8, tau = 1, tp = -4:4)
#' op1 <- uic.marginal(block, lib_var = "y", tar_var = "x", E = 0:8, tau = 1, tp = -4:4)
#' par(mfrow = c(2, 1))
#' with(op0, plot(tp, te, type = "l"))
#' with(op0[op0$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' with(op1, plot(tp, te, type = "l"))
#' with(op1[op1$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' 
uic.marginal = function (
    block, lib = c(1, NROW(block)), pred = lib,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 1, E = 1, tau = 1, tp = 0, nn = "e+1", n_boot = 2000,
    scaling = c("neighbor", "velocity", "no_scale"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(nn) != 1)
        stop("nn must be spcified as an integer or \"e+1\".")
    
    op_simplex = lapply(tau, function (x) {
        op = simplex(
            block, lib, pred, lib_var, c(tar_var, cond_var),
            norm, E, tau = x, tp = x, nn, 0, Enull = "e-1", 0.05,
            scaling, exclusion_radius, epsilon, is_naive)
        op$weight = with(op, exp(-log(rmse) / 2 - n_lib / n_pred / 2))
        op$weight = with(op, weight / sum(weight))
        op
    })
    
    op_uic = lapply(op_simplex, function (opS)
    {
        opU = uic(
            block, lib, pred, lib_var, tar_var, cond_var,
            norm, E + 1, opS$tau[1], tp, nn, n_boot,
            scaling, exclusion_radius, epsilon, is_naive)
        
        opM = lapply(tp, function (x) {
            op = apply(subset(opU, tp == x) * opS$weight, 2, sum)
            op[c("tau", "tp")] = c(opS$tau[1], x)
            op
        })
        data.frame(do.call(rbind, opM))
    })
    do.call(rbind, op_uic)
}

# End