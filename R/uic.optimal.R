#' Wrapper function for computing unified information-theoretic causality
#' for the optimal embedding dimension.
#' 
#' \code{uic.optimal} returns model statistics computed from given multiple time
#' series using simplex projection and cross mapping. This function computes UICs
#' after exploring the optimal \code{E} based on simplex projection. The users do
#' not have to determine the optimal \code{E} by themselves.
#' 
#' @inheritParams uic
#' @param tau
#' the time-lag used for time-delay embedding. Must be an integer.
#' @param alpha
#' the significant level to determine the embedding dimension of reference model
#' (i.e., E0). If \code{alpha = NULL}, E0 is set to E - 1. If \code{0 < alpha < 1} 
#' E0 depends on the model results with lower embedding dimensions. Default is
#' 0.05.
#' @param sequential_test
#' if \code{sequential_test = TRUE}, the function explores the optimal \code{tp} based
#' on sequential conditional test.
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' See the details in Value section of \code{uic}.
#' If \code{sequential_test = TRUE}, the data.frame includes an additional column
#' \code{seq_test}. The \code{tp} with \code{seq_test} = i > 0 were selected as optimal
#' by the i-th sequential test. The causal effect of \code{tp} with \code{seq_test} = -i
#' < 0 is blocked by \code{tp} with \code{seq_test} = i.
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
#' out0 <- uic.optimal(block, lib_var="x", tar_var="y", E=0:8, tau=1, tp=-4:4, sequential_test=TRUE)
#' out1 <- uic.optimal(block, lib_var="y", tar_var="x", E=0:8, tau=1, tp=-4:4, sequential_test=TRUE)
#' par(mfrow=c(2,2))
#' with(out0, plot(tp, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out1, plot(tp, ete, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out0, plot(tp, ete, type="b", pch=c(1,16)[1+(seq_test>0)]))
#' with(out1, plot(tp, ete, type="b", pch=c(1,16)[1+(seq_test>0)]))
#' 
uic.optimal = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", num_surr = 1000, alpha = 0.05,
    sequential_test = FALSE, exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    if (length(tau) != 1) stop("tau must be an integer.")
    if (num_surr <= 0) stop("num_surr must be a positive integer.")
    
    # Basic uic.optimal function
    if (!sequential_test) {
        simp <- simplex(
            block, lib, pred, group, lib_var, c(tar_var, cond_var), norm,
            E=E-1, tau, tp=tau, nn, num_surr, alpha, exclusion_radius, epsilon,
            is_naive, knn_method)
        E <- with(simp, max(0, E[pval < alpha])) + 1
        out <- uic(
            block, lib, pred, group, lib_var, tar_var, cond_var, norm,
            E, tau, tp, nn, num_surr, exclusion_radius, epsilon,
            is_naive, knn_method)
        return(out)
    }
    
    # Sequential test
    if (is.null(alpha)) stop("Must be 0 <= alpha <= 1 if sequential_test = TRUE.")
    out <- Recall(
        block, lib, pred, group, lib_var, tar_var, cond_var,
        norm, E, tau, tp, nn, num_surr, alpha, FALSE,
        exclusion_radius, epsilon, is_naive, knn_method)
    seq_test <- rep(0, nrow(out))
    idx <- which(out$pval < alpha)
    seq_test[idx[which.max(out$ete[idx])]] <- 1
    lag <- block[0]
    while (1) {
        idx <- which(out$pval<alpha & seq_test==0)
        if (length(idx) == 0) break
        lag.tp <- out$tp[which.max(seq_test)]
        lag <- cbind(lag, make_block(block, tar_var, -lag.tp, group))
        out.cond <- Recall(
            cbind(block,lag), lib, pred, group,
            lib_var, tar_var, c(cond_var,colnames(lag)),
            norm, E, tau, tp[idx], nn, num_surr, alpha, FALSE,
            exclusion_radius, epsilon, is_naive, knn_method)
        r <- max(seq_test)
        seq_test[idx[which.max(out.cond$ete)]] <-  r + 1
        seq_test[idx[out.cond$pval>=alpha]]    <- -r
    }
    return(cbind(out, seq_test=seq_test))
}

# End