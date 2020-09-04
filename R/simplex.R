#' Perform simplex projection
#' 
#' \code{simplex} returns model statistics computed from given multiple time series
#' based on simplex projection. This function simultaneously performs simplex projections
#' for all possible combinations of \code{E}, \code{tau} and \code{tp}.
#' 
#' @details
#' \code{scaling} specifies the methods for local scaling of distance matrix. This argument is
#' experimental. The following distances can be used as local scaling factors:
#' the mean distances to nearest neighbors of the embedding space (\code{scaling = neighbor}),
#' the mean distances to nearest time indices (\code{scaling = velocity}) and
#' the constant distance (\code{scaling = no_scale}).
#' 
#' @inheritParams uic
#' @param lib_var
#' the name or column index of a library (and target) variable.
#' The specified variable is used as a response variable and its time-delay variables are
#' used as explanatory variables.
#' @param Enull
#' specifies the method to determine the embedding dimension of null model. If "e-1" is used,
#' Enull is E - 1. If "adaptive" is used, Enull is the largest E, which satisfied E < Enull
#' and p-value < alpha. 
#' @param alpha
#' the significant level to use if Enull = "adaptive". Default is 0.05.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E}      \tab \code{:} embedding dimension \cr
#' \code{tau}    \tab \code{:} time-lag \cr
#' \code{tp}     \tab \code{:} time prediction horizon \cr
#' \code{nn}     \tab \code{:} number of nearest neighbors \cr
#' \code{Enull}  \tab \code{:} embedding dimension of null model \cr
#' \code{n_lib}  \tab \code{:} number of time indices used for attractor reconstruction \cr
#' \code{n_pred} \tab \code{:} number of time indices used for model predictions \cr
#' \code{rmse}   \tab \code{:} root mean squared error \cr
#' \code{te}     \tab \code{:} transfer entropy \cr
#' \code{pval}   \tab \code{:} bootstrap p-value for te > 0 \cr
#' }
#' 
#' \code{nn} may be different between argument specification and output results
#' when some nearest neighbors have tied distances.
#' 
#' \code{rmse} is the unbiased root mean squared error computed from model predictions.
#' If \code{is_naive = TRUE}, the raw root mean squared error is returned.
#' 
#' \code{te} is transfer entropy based on the difference of two simplex projection:
#' \deqn{
#' \sum_{t} log p(x_{t+tp} | x_{t}, x_{t-\tau}, \ldots, x_{t-(E -1)\tau}, z_{t}) -
#'          log p(x_{t+tp} | x_{t}, x_{t-\tau}, \ldots, x_{t-(Enull-1)\tau}, z_{t})
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
#' op0 <- simplex(block, lib_var = "x", cond_var = "y", E = 1:8, tau = 1, tp = 1)
#' op1 <- simplex(block, lib_var = "y", cond_var = "x", E = 1:8, tau = 1, tp = 1)
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
    Enull = c("e-1", "adaptive"), alpha = 0.05,
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
    
    nn = set_nn(nn, E)
    ord = order(E)
    E  = E [ord]
    nn = nn[ord]
    if (is.null(exclusion_radius)) exclusion_radius = 0;
    if (is.null(epsilon)) epsilon = -1
    LS = switch(match.arg(scaling), "no_scale" = 0, "neighbor" = 1, "velocity" = 2)
    
    x = as.matrix(block[,lib_var])
    z = as.matrix(block[,cond_var])
    
    uic = new(rUIC)
    uic$set_norm(NORM, LS, p, exclusion_radius, epsilon)
    uic$set_estimator(is_naive)
    
    Enull = sapply(strsplit(tolower(Enull), " "), paste, collapse = "")
    Enull = match.arg(Enull)
    if (Enull == "e-1")
    {
        op = uic$simplex_seq(n_boot, x, z, lib, pred, E, nn, tau, tp)
        op = op[order(op$tau),]
        op = op[order(op$tp),]
    }
    else if (Enull == "adaptive")
    {
        op = NULL
        for (tau_ in tau)
        {
            Enull = rep(0, length(tp))
            for (i in seq_along(E))
            {
                op_i = uic$simplex(n_boot, x, z, lib, pred, E[i], nn[i], tau_, tp, Enull)
                Enull[op_i$pval < alpha] = with(op_i, E[pval < alpha]) 
                op = rbind(op, op_i)
            }
        }
        op = op[order(op$tp),]
    }
    op = op[,which(colnames(op) != "rmse_R")]
    rownames(op) = NULL
    op
}

# End