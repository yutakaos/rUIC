#' Perform cross mapping
#' 
#' \code{xmap} returns model predictions and their statistics computed from
#' given multiple time series based on cross mapping.
#' 
#' @details
#' \code{scaling} specifies the methods for local scaling of distance matrix. This argument is
#' experimental. The following distances can be used as local scaling factors:
#' the mean distances to nearest neighbors of the embedding space (\code{scaling = neighbor}),
#' the mean distances to nearest time indices (\code{scaling = velocity}) and
#' the constant distance (\code{scaling = no_scale}).
#' 
#' @param block
#' a data.frame or matrix where each column is a time series.
#' @param lib
#' a two-column matrix (or two-element vector) where each row specifies
#' the first and last indices of time series to use for attractor reconstruction.
#' @param pred
#' a two-column matrix (or two-element vector) where each row specifies
#' the first and last indices of time series to use for model predictions.
#' @param lib_var
#' the names or column indices of library variables.
#' The specified variables are used as explanatory variables with time-delay embedding.
#' @param tar_var
#' the name or column index of a target variable.
#' The specified variable is used as response variables.
#' @param cond_var
#' the names or column indeices of condition data.
#' The specified variables are used as explanatory variables (without time-delay embedding).
#' @param norm
#' the power of Lp distance to use.
#' If \code{norm} \eqn{\le} 0, Maximum distance is used as the special case of Lp distance.
#' @param E
#' the embedding dimension to use for time-delay embedding. Must be an integer.
#' @param tau
#' the time-lag to use for time-delay embedding. Must be an integer.
#' @param tp
#' the time index to predict. Must be an integer.
#' @param nn
#' the number of nearest neighbors to use. Must be an integer or "e+1".
#' If \code{nn = "e+1"}, \code{nn} is set as \code{E} + 1.
#' @param n_boot
#' the number of bootstraps to use for computing p-value.
#' @param scaling
#' the method for local scaling of distance matrix. See Details.
#' @param exclusion_radius
#' the filtering to exclude nearest neighbors if their time index is within exclusion radius.
#' @param epsilon
#' the filtering to exclude nearest neighbors if epsilon is farther than their distances.
#' @param is_naive
#' specifies whether the rEDM-style estimator is used.
#' 
#' @return
#' A list with model predictions and its statistics.
#' 
#' \code{model_output} is a data.frame where each column is:
#' \tabular{ll}{
#' \code{time} \tab \code{:} time indices \cr
#' \code{data} \tab \code{:} data values used for model prediction \cr
#' \code{pred} \tab \code{:} predicted values \cr
#' \code{enn}  \tab \code{:} effective number of nearest neighbors \cr
#' }
#' 
#' \code{stats} is a data.frame where each row represents model statistics computed from a parameter set:
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
#' If \code{is_naive = TRUE}, the raw root mean sqaured error is returned.
#' 
#' \code{te} is transfer entropy based on the unified information-theoretic causality test:
#' \deqn{
#' \sum_{t} log p(y_{t+tp} | x_{t}, x_{t- \tau}, \ldots, x_{t-(E-1)\tau}, z_{t}) -
#'          log p(y_{t+tp} | z_{t})
#' }
#' where \eqn{x} is library, \eqn{y} is target and \eqn{z} is condition.
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
#' ## cross mapping
#' op0 <- xmap(block, lib_var = "x", tar_var = "y", E = 2, tau = 1, tp = -1)
#' op1 <- xmap(block, lib_var = "y", tar_var = "x", E = 2, tau = 1, tp = -1)
#' par(mfrow = c(2, 1))
#' with(op0$model_output, plot(data, pred)); op0$stats
#' with(op1$model_output, plot(data, pred)); op1$stats
#' 
xmap = function(
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
    if (length(E)  != 1) stop("E must be an integer.")
    if (length(tau)!= 1) stop("tau must be an integer.")
    if (length(tp) != 1) stop("tp must be an integer.")
    lib  = rbind(lib)
    pred = rbind(pred)
    
    p = pmax(0, norm)
    NORM = 3  # Lp norm
    if      (norm == 2) NORM = 0  # L2 norm
    else if (norm == 1) NORM = 1  # L1 norm
    else if (norm <= 0) NORM = 2  # Max norm
    
    nn = set_nn(nn, E)
    if (is.null(exclusion_radius)) exclusion_radius = 0;
    if (is.null(epsilon)) epsilon = -1
    LS = switch(match.arg(scaling), "no_scale" = 0, "neighbor" = 1, "velocity" = 2)
    
    x = as.matrix(block[,lib_var])
    y = as.matrix(block[,tar_var])
    z = matrix()
    if (!is.null(cond_var)) z = as.matrix(block[,cond_var])
    
    uic = new(rUIC)
    uic$set_norm(NORM, LS, p, exclusion_radius, epsilon)
    uic$set_estimator(is_naive)
    op = uic$xmap(n_boot, x, y, z, lib, pred, E[1], nn[1], tau[1], tp[1])
    op$stats = op$stats[,which(colnames(op$stats) != "rmse_R")]
    op$model_output$time = op$model_output$time + 1
    op
}

# End