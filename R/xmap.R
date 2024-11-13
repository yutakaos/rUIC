#' Cross Mapping Prediction
#' 
#' \code{xmap} returns model predictions from given multiple time series using
#' cross mapping.
#' 
#' @param block
#' a data.frame or matrix where each column is a time series.
#' @param lib
#' a two-column matrix (or two-element vector) where each row specifies the
#' first and last indices of time series to use for attractor reconstruction.
#' @param pred
#' a two-column matrix (or two-element vector) where each row specifies the
#' first and last indices of time series to use for model predictions.
#' @param group
#' the name or column index of a group variable, which specifies different
#' groups of time series data.
#' @param lib_var
#' the names or column indices of library variables.
#' @param tar_var
#' the names or column indices of target variables.
#' @param cond_var
#' the names or column indeices of condition variables, which are used as
#' library variables without time-delay embedding.
#' @param norm
#' the power of Lp distance. Maximum distance is used if \code{norm} == Inf.
#' @param E
#' the embedding dimension used for time-delay embedding. Must be an integer.
#' @param tau
#' the time-lag used for time-delay embedding. Must be an integer.
#' @param tp
#' the time index to predict. Must be an integer.
#' @param nn
#' the number of nearest neighbors. Must be an integer or "e+1".
#' If \code{nn = "e+1"} or \code{nn = -1}, \code{nn} is set to \code{E} + 1.
#' If \code{nn = 0}, \code{nn} is set to the number of all data.
#' @param exclusion_radius
#' the filtering to exclude nearest neighbors if their time index is within
#' exclusion radius.
#' @param epsilon
#' the filtering to exclude nearest neighbors if their distance is farther
#' away than epsilon.
#' @param is_naive
#' specifies whether naive estimator is used or not.
#' @param knn_method
#' the method to search nearest neighbors.
#' The KD-tree ("KD") or brute-force ("BF") search can be used.
#' 
#' @return
#' A list or data.frame with model predictions, where each column is:
#' \tabular{ll}{
#' \code{data} \tab \code{:} data values used for model prediction \cr
#' \code{pred} \tab \code{:} predicted values \cr
#' \code{enn}  \tab \code{:} effective number of nearest neighbors \cr
#' \code{sqe}  \tab \code{:} squared errors \cr
#' }
#' 
#' @examples
#' # simulate logistic map
#' tl <- 400  # time length
#' x <- y <- rep(NA, tl)
#' x[1] <- 0.4
#' y[1] <- 0.2
#' for (t in 1:(tl - 1)) {  # causality : x -> y
#'     x[t+1] = x[t] * (3.8 - 3.8 * x[t] - 0.0 * y[t])
#'     y[t+1] = y[t] * (3.5 - 3.5 * y[t] - 0.1 * x[t])
#' }
#' block <- data.frame(t=1:tl, x=x, y=y)
#' 
#' # cross mapping
#' out0 <- xmap(block, lib_var="x", tar_var="y", E=2, tau=1, tp=-1)
#' out1 <- xmap(block, lib_var="y", tar_var="x", E=2, tau=1, tp=-1)
#' par(mfrow=c(2, 1))
#' with(out0, plot(data, pred))
#' with(out1, plot(data, pred))
#' 
xmap = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, tar_var = lib_var, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1",
    exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    if (E  [1] < 0) stop("E must be non-negative.")
    if (tau[1] < 0) stop("tau must be non-negative.")
    if (norm   < 1) stop("norm must be >= 1.")
    if (!is.numeric(nn) & tolower(nn) != "e+1") stop('nn must be an integer or "e+1".')
    lib  <- rbind(lib)
    pred <- rbind(pred)
    if (length(group) == 0) Group <- rep(1, nrow(block))
    if (length(group) != 0) Group <- as.numeric(as.factor(block[,group[1]]))
    
    tp <- tp[1]
    p  <- ifelse(is.finite(norm), norm, 0)
    KNN <- switch(match.arg(knn_method), "KD"=0, "BF"=1)
    if (tolower(nn) == "e+1") nn <- -1
    if (is.null(exclusion_radius)) exclusion_radius <- 0
    if (is.null(epsilon)) epsilon <- -1
    
    X <- as.matrix(block[ lib_var])
    Y <- as.matrix(block[ tar_var])
    Z <- as.matrix(block[cond_var])
    out <- .Call(`_rUIC_xmap_predict_R`,
        X, Y, Z, Group, lib, pred, E, tau, tp, nn, p,
        exclusion_radius, epsilon, is_naive, KNN)
    for (k in 1:ncol(Y)) out[[k]]$data <- Y[,k]
    if (length(out) == 1) out <- out[[1]]
    return(out)
}

# End