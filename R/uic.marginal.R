#' Wrapper function for computing unified information-theoretic causality
#' for the marginal embedding dimension.
#' 
#' \code{uic.marginal} returns model statistics computed from given multiple time series
#' based on simplex projection and cross mapping. This function computes UICs using a model
#' averaging technique (i.e., marginalizing \code{E}). Thus, the users do not have to determine
#' the optimal \code{E} by themselves.
#' 
#' @inheritParams uic
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
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1",
    scaling = c("no_scale", "neighbor", "velocity"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(nn) != 1) stop("'nn' must be spcified as an integer or \"e+1\".")
    
    op_simplex = lapply(tau, function (x) {
        op = simplex(
            block, lib, pred, lib_var, c(tar_var, cond_var),
            norm, E, tau = x, tp = x, nn, Enull = "e-1", 0.05,
            scaling, exclusion_radius, epsilon, is_naive)
        op$weight = with(op, exp(-log(rmse) / 2 - n_lib / n_pred / 2))
        op$weight = with(op, weight / sum(weight))
        op
    })
    
    op_uic = lapply(op_simplex, function (opS)
    {
        opU = uic(
            block, lib, pred, lib_var, tar_var, cond_var,
            norm, E + 1, opS$tau[1], tp, nn,
            scaling, exclusion_radius, epsilon, is_naive)
        
        opM = lapply(tp, function (x) {
            op = apply(subset(opU, tp == x) * opS$weight, 2, sum)
            op[c("tau", "tp")] = c(opS$tau[1], x)
            op
        })
        data.frame(do.call(rbind, opM))
    })
    op_uic = do.call(rbind, op_uic)
    
    int_par = c("E","nn","E_R","nn_R","n_lib","n_pred")
    op_uic[,int_par] = round(op_uic[,int_par], 1)
    op_uic
}

# End