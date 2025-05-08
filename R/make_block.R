#' Making time-lagged block
#' 
#' \code{make_block} makes time-lagged block.
#' 
#' @param block
#' a data.frame or matrix where each column is a time series.
#' @param lib_var
#' the names or column indices of library variables.
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
#' head(make_block(block, lib_var=c("x","y"), lag= 0))
#' head(make_block(block, lib_var=c("x","y"), lag= 1))
#' head(make_block(block, lib_var=c("x","y"), lag=-1))
#' 
make_block = function (block, lib_var = 1, lag = 0, group = NULL)
{
    if (length(group) == 0) Group <- rep(1, nrow(block))
    if (length(group) != 0) Group <- as.numeric(as.factor(block[,group[1]]))
    Group[is.na(Group)] <- 0
    
    not_same_group <- Group[-1] != Group[-nrow(block)]
    lib <- data.frame(
        first = c(1, which(not_same_group)+1),
        last  = c(which(not_same_group), nrow(block)) )
    lib$group <- Group[lib$first]
    
    X <- as.matrix(block[lib_var])
    naF <- matrix(NA, nrow=pmax(0, lag), ncol=ncol(X))
    naB <- matrix(NA, nrow=pmax(0,-lag), ncol=ncol(X))
    nNA <- nrow(naB)
    out <- lapply(1:nrow(lib), function(i) {
        L1 <- lib[i,1]
        L2 <- lib[i,2]
        if (lib[i,3]==0) return(matrix(NA, L2-L1+1, ncol(X)))
        dc <- rbind(naF, X[L1:L2,,drop=FALSE], naB)
        dc [nNA+1:(L2-L1+1),,drop=FALSE]
    })
    out <- data.frame(do.call(rbind, out))
    suffix <- paste0("_lag.", ifelse(lag>0,"B","F"), abs(lag))
    if (lag==0) suffix <- ""
    colnames(out) <- paste0(colnames(X), suffix)
    return(out)
}

# End