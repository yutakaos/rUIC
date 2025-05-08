#' Simplex Projection
#' 
#' \code{simplex} returns model statistics computed from given multiple time series
#' using simplex projection. This function performs simplex projections for all
#' possible combinations of \code{E}, \code{tau} and \code{tp}.
#' 
#' @inheritParams uic
#' @param lib_var
#' the name or column index of a library (and target) variable.
#' The specified variable is used as a response variable and its time-delay variables are
#' used as explanatory variables.
#' @param alpha
#' the significant level to determine the embedding dimension of reference model
#' (i.e., E0). If \code{alpha = NULL}, E0 is set to E - 1. If \code{0 <= alpha <= 1}, 
#' E0 depends on the model results with lower embedding dimensions.
#' 
#' @details
#' Transfer entropy is computed as follows:
#' \deqn{
#' \sum_{t} log p(X_{t+tp} | X_{t}, X_{t-\tau}, \ldots, X_{t-(E -1)\tau}, Z_{t}) -
#'          log p(X_{t+tp} | X_{t}, X_{t-\tau}, \ldots, X_{t-(E0-1)\tau}, Z_{t})
#' }
#' where \eqn{X} and \eqn{Z} are library and condition variables, respectively.
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
#' @seealso \link{xmap}, \link{uic}
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
#' # simplex projecton
#' out0 <- simplex(block, lib_var="x", cond_var="y", E=0:8, tau=1, tp=1)
#' out1 <- simplex(block, lib_var="y", cond_var="x", E=0:8, tau=1, tp=1)
#' par(mfrow=c(2, 1))
#' with(out0, plot(E, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out1, plot(E, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' 
simplex = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 1, nn = "e+1", num_surr = 1000, alpha = NULL,
    exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    if (norm < 1) stop("norm must be >= 1.")
    if (!is.numeric(nn) & tolower(nn) != "e+1") stop('nn must be an integer or "e+1".')
    lib  <- rbind(lib)
    pred <- rbind(pred)
    if (length(group) == 0) Group <- rep(1, nrow(block))
    if (length(group) != 0) Group <- as.numeric(as.factor(block[,group[1]]))
    
    E   <- sort(unique(pmax(0, E)))
    tau <- unique(pmax(1, tau))
    tp  <- unique(tp)
    p   <- ifelse(is.finite(norm), norm, 0)
    num_surr <- pmax(0, num_surr)
    KNN <- switch(match.arg(knn_method), "KD"=0, "BF"=1)
    if (tolower(nn) == "e+1") nn <- -1
    if (is.null(exclusion_radius)) exclusion_radius <- 0
    if (is.null(epsilon)) epsilon <- -1
    
    X <- as.matrix(block[ lib_var])
    Z <- as.matrix(block[cond_var])
    if (is.null(alpha) || num_surr == 0) {
        out <- .Call(`_rUIC_xmap_fit_R`,
            X, X, Z, Group, lib, pred, E, E-1, tau, tp, nn, p, num_surr,
            exclusion_radius, epsilon, is_naive, 0, KNN)
    }
    else {
        out <- NULL
        for (tpi in tp) for (taui in tau) {
            E0 <- 0
            for (Ei in E) {
                outi <- .Call(`_rUIC_xmap_fit_R`,
                    X, X, Z, Group, lib, pred, Ei, E0, taui, tpi, nn, p,
                    num_surr, exclusion_radius, epsilon, is_naive, 0, KNN)
                if(outi$pval < alpha) E0 <- Ei
                out <- rbind(out, outi)
            }
        }
    }
    return(out)
}

# End