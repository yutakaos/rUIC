#' Perform simplex projection
#' 
#' \code{simplex} returns model statistics computed from given multiple time series
#' based on simplex projection. This function simultaneously performs simplex projections
#' for all possible combinations of \code{E}, \code{tau} and \code{tp}.
#' 
#' @inheritParams uic
#' @param lib_var
#' the name or column index of a library (and target) variable.
#' The specified variable is used as a response variable and its time-delay variables are
#' used as explanatory variables.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E}      \tab \code{:} embedding dimension \cr
#' \code{tau}    \tab \code{:} time-lag \cr
#' \code{tp}     \tab \code{:} time prediction horizon \cr
#' \code{nn}     \tab \code{:} number of nearest neighbors \cr
#' \code{n_lib}  \tab \code{:} number of time indices used for attractor reconstruction \cr
#' \code{n_pred} \tab \code{:} number of time indices used for model predictions \cr
#' \code{rmse}   \tab \code{:} unbiased root mean squared error \cr
#' \code{te}     \tab \code{:} transfer entropy \cr
#' \code{pval}   \tab \code{:} bootstrap p-value for te > 0 \cr
#' }
#' 
#' \code{nn} can be different between argument specification and output results
#' when some nearest neighbors have tied distances.
#' 
#' \code{rmse} is the unbiased root mean squared error computed from model predictions.
#' If \code{is_naive = TRUE}, the raw root mean squared error is returned.
#' 
#' \code{te} is transfer entropy based on the difference of two simplex projection:
#' \deqn{
#' \sum_{t} log p(x_{t+tp} | x_{t}, x_{t-\tau}, \ldots, x_{t-(E-1)\tau}, z_{t}) -
#'          log p(x_{t+tp} | x_{t}, x_{t-\tau}, \ldots, x_{t-(E-2)\tau}, z_{t})
#' }
#' where \eqn{x} is library and \eqn{z} is condition.
#' 
#' @seealso \link{xmap}, \link{uic}
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
#' ## simplex projecton
#' op0 <- simplex(block, lib_var = "x", cond_var = "y", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
#' op1 <- simplex(block, lib_var = "y", cond_var = "x", E = 1:8, tau = 1, tp = -1, n_boot = 2000)
#' par(mfrow = c(2, 1))
#' with(op0, plot(E, te, type = "l"))
#' with(op0[op0$pval < 0.05,], points(E, te, pch = 16, col = "red"))
#' with(op1, plot(E, te, type = "l"))
#' with(op1[op1$pval < 0.05,], points(E, te, pch = 16, col = "red"))
#' 
simplex = function (
    block, lib = c(1, NROW(block)), pred = lib,
    lib_var = 1, cond_var = 2,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", n_boot = 2000,
    scaling = c("neighbor", "velocity", "no_scale"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(lib_var) != 1)
    {
        stop("Only a target variable (tar_var) must be specifed.")
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
    z = cbind(block[,cond_var])
    
    uic = new(rUIC)
    uic$set_norm(NORM, LS, p, exclusion_radius, epsilon)
    uic$set_estimator(is_naive)
    op = uic$simplex_seq(n_boot, x, z, lib, pred, E, nn, tau, tp)
    op[,which(colnames(op) != "rmse_R")]
}

# End