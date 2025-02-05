#' Making time-lagged block
#' 
#' \code{make_block} makes time-lagged block.
#' 
#' @param X
#' a data.frame or matrix where each column is a time series.
#' @param lib
#' a two-column matrix (or two-element vector) where each row specifies the
#' first and last indices of time series to use for attractor reconstruction.
#' @param lag
#' the time-lag. Must be an integer.
#' @param group
#' the name or column index of a group variable, which specifies different
#' groups of time series data.
#' 
#' @return
#' A data.frame where each column represents a time-lagged variable.
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
#' head(make_block(block[-1], lag= 0))
#' head(make_block(block[-1], lag= 1))
#' head(make_block(block[-1], lag=-1))
#' 
make_block = function (X, lib = c(1,NROW(X)), lag = 0, group=NULL)
{
    lib <- rbind(lib)
    if (is.null(group)) {
        X <- as.matrix(X)
        naF <- matrix(NA, nrow=pmax(0, lag), ncol=ncol(X))
        naB <- matrix(NA, nrow=pmax(0,-lag), ncol=ncol(X))
        nNA <- nrow(naB)
        out <- lapply(1:nrow(lib), function(i) {
            L1 <- lib[i,1]
            L2 <- lib[i,2]
            dc <- rbind(naF, X[L1:L2,,drop=FALSE], naB)
            dc [nNA+1:(L2-L1+1),,drop=FALSE]
        })
        out <- data.frame(do.call(rbind, out))
        suffix <- paste0("_lag.", ifelse(lag>0,"B","F"), abs(lag))
        if (lag==0) suffix <- ""
        colnames(out) <- paste0(colnames(X), suffix)
        return(out)
    }
    lib <- lapply(1:nrow(lib), function(i) {
        L <- lapply(unique(group), function(g) range((lib[i,1]:lib[i,2])[group==g]))
        L <- do.call(rbind, L)
    })
    lib <- do.call(rbind, lib)
    Recall(X, lib, lag)
}

# End