#' Perform simplex projection and compute marginal unified information-theoretic causality
#' 
#' \code{marginal_uic} returns model statistics computed from given multiple time series
#' based on simplex projection and cross mapping. This function computes UIC using a model
#' averaging technique (and marginalizing \code{E} and \code{tau}). Thus, the users do not
#' have to determine the optimal \code{E} and \code{tau} by themselves.
#' 
#' @inheritParams uic
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E_opt}   \tab \code{:} embedding dimension, which minimize RMSE for simplex projection \cr
#' \code{tau_opt} \tab \code{:} time-lag, which minimize RMSE for simplex projection \cr
#' \code{tp}      \tab \code{:} time prediction horizon \cr
#' \code{nn_opt}  \tab \code{:} number of nearest neighbors, which minimize RMSE for simplex projection \cr
#' \code{te}      \tab \code{:} transfer entropy \cr
#' \code{pval}    \tab \code{:} bootstrap p-value for te > 0 \cr
#' }
#' 
#' \code{te} is transfer entropy based on the unified information-theoretic causality test:
#' \deqn{
#' \sum_{t} log p(y_{t+tp} | x_{t}     , x_{t- \tau}, \ldots, x_{t-(E-1)\tau}, z_{t}) -
#'          log p(y_{t+tp} | x_{t-\tau}, x_{t-2\tau}, \ldots, x_{t-(E-1)\tau}, z_{t})
#' }
#' where \eqn{x} is library, \eqn{y} is target and \eqn{z} is condition.
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
#' op0 = marginal_uic(block, lib_var = "x", tar_var = "y", E = 1:10, tau = 1:3, tp = -4:4,
#'                    n_boot = 2000)
#' op1 = marginal_uic(block, lib_var = "y", tar_var = "x", E = 1:10, tau = 1:3, tp = -4:4,
#'                    n_boot = 2000)
#' par(mfrow = c(2, 1))
#' with(op0, plot(tp, te, type = "l"))
#' with(op0[op0$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' with(op1, plot(tp, te, type = "l"))
#' with(op1[op1$pval < 0.05,], points(tp, te, pch = 16, col = "red"))
#' 
marginal_uic = function (
    block, lib = c(1, NROW(block)), pred = lib,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", n_boot = 2000,
    scaling = c("neighbor", "velocity", "no_scale"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(tar_var) != 1)
    {
        stop("Only a target variable (tar_var) must be specifed.")
    }
    if (any(E <= 0))
    {
        stop("Should be E > 0.")
    }
    lib  = rbind(lib)
    pred = rbind(pred)
    
    p = pmax(0, norm)
    NORM = 2  # Lp norm
    if      (norm == 2) NORM = 0  # L2 norm
    else if (norm == 1) NORM = 1  # L1 norm
    else if (norm <= 0) NORM = 3  # Max norm
    
    if (nn == "e+1") nn = E + 1
    else if (length(nn) == 1) nn = rep(nn, length(E))
    if (is.null(exclusion_radius)) exclusion_radius = 0;
    if (is.null(epsilon)) epsilon = -1
    LS = match.arg(scaling)
    LS = switch(LS, "no_scale" = 0, "neighbor" = 1, "velocity" = 2)
    
    x = cbind(block[,lib_var])
    y = cbind(block[,tar_var])
    w = cbind(block[,c(tar_var, cond_var)])
    z = matrix()
    if (!is.null(cond_var)) z = cbind(block[,cond_var])
    
    uic = new(rUIC)
    uic$set_norm(NORM, LS, p, exclusion_radius, epsilon)
    uic$set_estimator(is_naive)
    
    simplex = lapply(tau, function (tau_)
        uic$simplex_seq(1, x, w, lib, pred, E - 1, nn, tau_, tau_)
    )
    simplex = do.call(rbind, simplex)
    simplex = simplex[order(simplex$E),]
    simplex$weight = with(simplex, exp(-log(rmse) / 2 - n_lib / n_pred / 2))
    simplex$weight = with(simplex, weight / sum(weight))
    
    E_id   = with(simplex, which.max(tapply(weight, E, sum)))
    tau_id = with(simplex, which.max(tapply(weight, tau, sum)))
    
    xmap = uic$xmap_seq(n_boot, x, y, z, lib, pred, E , nn, tau, tp)
    op = lapply(tp, function (tp_) {
        data.frame(
            E_opt   = E[E_id],
            tau_opt = tau[tau_id],
            tp      = tp_,
            nn_opt  = nn[E_id],
            te   = sum(xmap[xmap$tp == tp_, "te"  ] * simplex$weight),
            pval = sum(xmap[xmap$tp == tp_, "pval"] * simplex$weight)
        )
    })
    op = do.call(rbind, op)
    op
}

# End