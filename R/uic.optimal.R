#' Wrapper function for computing unified information-theoretic causality
#' for the optimal embedding dimension.
#' 
#' \code{uic.optimal} returns model statistics computed from given multiple time
#' series using simplex projection and cross mapping. This function computes UICs
#' after exploring the optimal \code{E} based on simplex projection. The users
#' do not have to determine the optimal \code{E} by themselves.
#' 
#' @inheritParams uic
#' @param alpha
#' the significant level to determine the embedding dimension of reference model
#' (i.e., E0). If \code{alpha = NULL}, E0 is set to E - 1. If \code{0 < alpha < 1} 
#' E0 depends on the model results with lower embedding dimensions. Default is
#' 0.05.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' See the details in Value section of \code{uic}.
#' 
#' @seealso \link{simplex}, \link{uic}
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
#' out0 <- uic.optimal(block, lib_var="x", tar_var="y", E=0:8, tau=1, tp=-4:0)
#' out1 <- uic.optimal(block, lib_var="y", tar_var="x", E=0:8, tau=1, tp=-4:0)
#' par(mfrow=c(2, 1))
#' with(out0, plot(tp, te, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out1, plot(tp, te, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' 
uic.optimal = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", num_surr = 1000, alpha = 0.05, 
    exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    if (num_surr <= 0) stop("num_surr must be a positive integer.")
    out <- lapply(tau, function (x)
    {
        # exploring optimal E using simplex projection
        simp <- simplex(
            block, lib, pred, group, lib_var, c(tar_var, cond_var), norm,
            E=E-1, tau=x, tp=x, nn, num_surr, alpha, exclusion_radius, epsilon,
            is_naive, knn_method)
        # compute UICs
        E <- with(simp, max(0, E[pval < alpha])) + 1
        uic(
            block, lib, pred, group, lib_var, tar_var, cond_var, norm,
            E, tau=x, tp, nn, num_surr, exclusion_radius, epsilon,
            is_naive, knn_method)
    })
    do.call(rbind, out)
}

# End