#' Unfied Information-Theoretic Causality
#' 
#' \code{uic} returns model statistics computed from given multiple time series
#' using cross mapping. This function computes perform cross mappings for all
#' possible combinations of \code{E}, \code{tau} and \code{tp}.
#' 
#' @inheritParams xmap
#' @param E
#' the embedding dimensions used for time-delay embedding.
#' @param tau
#' the time-lags used for time-delay embedding.
#' @param tp
#' the time indices to predict.
#' @param num_surr
#' the number of surrogate data generated to compute p-value.
#' 
#' @details
#' Transfer entropy is computed as follows:
#' \deqn{
#' \sum_{t} log p(Y_{t+tp} | X_{t}     , X_{t- \tau}, \ldots, X_{t-(E-1)\tau}, Z_{t}) -
#'          log p(Y_{t+tp} | X_{t-\tau}, X_{t-2\tau}, \ldots, X_{t-(E-1)\tau}, Z_{t})
#' }
#' where \eqn{X}, \eqn{Y} and \eqn{Z} are library, target and condition variables,
#' respectively.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' \tabular{ll}{
#' \code{E}      \tab \code{:} embedding dimension \cr
#' \code{E0}     \tab \code{:} embedding dimension of reference model \cr
#' \code{tau}    \tab \code{:} time-lag \cr
#' \code{tp}     \tab \code{:} time prediction horizon \cr
#' \code{nn}     \tab \code{:} number of nearest neighbors \cr
#' \code{nn0}    \tab \code{:} number of nearest neighbors of reference model \cr
#' \code{n_lib}  \tab \code{:} number of time indices used for attractor reconstruction \cr
#' \code{n_pred} \tab \code{:} number of time indices used for model predictions \cr
#' \code{rmse}   \tab \code{:} root mean squared error \cr
#' \code{te}     \tab \code{:} transfer entropy \cr
#' \code{ete}    \tab \code{:} effective transfer entropy \cr
#' \code{pval}   \tab \code{:} p-value to test alternative hypothesis, te > 0 \cr
#' \code{n_surr} \tab \code{:} number of surrogate data \cr
#' }
#' 
#' @seealso \link{xmap}, \link{simplex}
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
#' # UIC
#' out0 <- uic(block, lib_var="x", tar_var="y", E=2, tau=1, tp=-4:0)
#' out1 <- uic(block, lib_var="y", tar_var="x", E=2, tau=1, tp=-4:0)
#' par(mfrow=c(2, 1))
#' with(out0, plot(tp, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out1, plot(tp, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' 
uic = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", num_surr = 1000,
    exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    if (norm < 1) stop("norm must be >= 1.")
    if (!is.numeric(nn) & tolower(nn) != "e+1") stop('nn must be an integer or "e+1".')
    lib  <- rbind(lib)
    pred <- rbind(pred)
    if (length(group) == 0) Group <- rep(1, nrow(block))
    if (length(group) != 0) Group <- as.numeric(as.factor(block[,group[1]]))
    
    E   <- sort(unique(pmax(1, E)))
    tau <- unique(pmax(1, tau))
    tp  <- unique(tp)
    p   <- ifelse(is.finite(norm), norm, 0)
    num_surr <- pmax(0, num_surr)
    KNN <- switch(match.arg(knn_method), "KD"=0, "BF"=1)
    if (tolower(nn) == "e+1") nn <- -1
    if (is.null(exclusion_radius)) exclusion_radius <- 0
    if (is.null(epsilon)) epsilon <- -1
    
    X <- as.matrix(block[ lib_var])
    Y <- as.matrix(block[ tar_var])
    Z <- as.matrix(block[cond_var])
    out <- .Call(`_rUIC_xmap_fit_R`,
        X, Y, Z, Group, lib, pred, E, E-1, tau, tp, nn, p, num_surr,
        exclusion_radius, epsilon, is_naive, 1, KNN)
    return(out)
}

# End