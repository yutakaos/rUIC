#' Wrapper function for computing unified information-theoretic causality
#' for the optimal embedding dimension.
#' 
#' \code{uic.optimal} returns model statistics computed from given multiple time series
#' based on simplex projection and cross mapping. This function computes UIC after exploring
#' the optimal \code{E} based on simplex projection. Thus, the users do not have to determine
#' the optimal \code{E} by themselves.
#' 
#' @inheritParams uic
#' @param alpha
#' the significant level to use for the "adaptive" simplex method. Default is 0.05.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' See the details in Value section of \code{uic}.
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
#' op0 <- uic.optimal(block, lib_var = "x", tar_var = "y", E = 0:8, tau = 1, tp = -4:4)
#' op1 <- uic.optimal(block, lib_var = "y", tar_var = "x", E = 0:8, tau = 1, tp = -4:4)
#' par(mfrow = c(2, 1))
#' with(op0, plot(tp, te, type = "l"))
#' with(op0[op0$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' with(op1, plot(tp, te, type = "l"))
#' with(op1[op1$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' 
uic.optimal = function (
    block, lib = c(1, NROW(block)), pred = lib,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", alpha = 0.05,
    scaling = c("no_scale", "neighbor", "velocity"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(nn) != 1) stop("'nn' must be spcified as an integer or \"e+1\".")
    
    op_simplex = lapply(tau, function (x)
        simplex(
            block, lib, pred, lib_var, c(tar_var, cond_var),
            norm, E, tau = x, tp = x, nn, Enull = "adaptive", alpha,
            scaling, exclusion_radius, epsilon, is_naive)
    )
    
    op_uic = lapply(op_simplex, function (op)
    {
        optE = with(op, max(c(0, E[pval < alpha]))) + 1
        uic(
            block, lib, pred, lib_var, tar_var, cond_var,
            norm, E = optE, op$tau[1], tp, nn,
            scaling, exclusion_radius, epsilon, is_naive)
    })
    do.call(rbind, op_uic)
}

# End