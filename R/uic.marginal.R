#' Wrapper function for computing unified information-theoretic causality
#' for the marginal embedding dimension.
#' 
#' \code{uic.marginal} returns model statistics computed from given multiple time
#' series using simplex projection and cross mapping. This function computes UICs
#' by a model averaging technique (i.e., marginalizing \code{E}). The users
#' do not have to determine the optimal \code{E} by themselves.
#' 
#' @inheritParams uic
#' 
#' @return
#' A data.frame where each row represents model statistics computed from a parameter set.
#' See the details in Value section of \code{uic}.
#' 
#' @seealso \link{simplex}, \link{uic}, \link{uic.optimal}
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
#' out0 <- uic.marginal(block, lib_var="x", tar_var="y", E=0:8, tau=1, tp=-4:0)
#' out1 <- uic.marginal(block, lib_var="y", tar_var="x", E=0:8, tau=1, tp=-4:0)
#' par(mfrow=c(2, 1))
#' with(out0, plot(tp, te, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' with(out1, plot(tp, te, type="b", pch=c(1,16)[1+(pval<0.05)]))
#' 
uic.marginal = function (
    block, lib = c(1, NROW(block)), pred = lib, group = NULL,
    lib_var = 1, tar_var = 2, cond_var = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", num_surr = 1000,
    exclusion_radius = NULL, epsilon = NULL,
    is_naive = FALSE, knn_method = c("KD","BF"))
{
    out <- lapply(tau, function (x) {
        # compute weights for specified dimensions
        simp <- simplex(
            block, lib, pred, group, lib_var, c(tar_var, cond_var), norm,
            E=E-1, tau=x, tp=x, nn, 0, NULL, exclusion_radius, epsilon,
            is_naive, knn_method)
        simp$weight = with(simp, exp(-log(rmse) - n_lib / n_pred))
        simp$weight = with(simp, weight / sum(weight))
        # compute UICs
        outx <- lapply(tp, function (y) {
            model <- uic(
                block, lib, pred, group, lib_var, tar_var, cond_var, norm,
                simp$E+1, tau=x, tp=y, nn, num_surr, exclusion_radius, epsilon,
                is_naive, knn_method)
            data.frame(rbind(apply(model * simp$weight, 2, sum)))
        })
        do.call(rbind, outx)
    })
    out <- do.call(rbind, out)
    par_int <- c("E","E0","tau","tp","nn","n_lib","n_pred","n_surr")
    out[,par_int] <- round(out[,par_int], 2)
    out
}

# End