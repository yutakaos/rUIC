#' Perform cross mapping (return model output and statistics)
#' 
#' @param block    a data.frame or matrix where each column is a time series
#' @param lib      the time range to be used for attractor reconstruction
#' @param pred     the time range to be used for prediction forecast
#' @param x_column the name or column index of library data
#' @param y_column the name or column index of target data
#' @param z_column the name or column index of condition data
#' @param norm     the power of Lp norm (if p < 0, max norm is used)
#' @param E        the embedding dimension
#' @param tau      the time-lag for delay embedding
#' @param tp       the time index to predict
#' @param nn       the number of neighbors
#' @param scaling  the local scaling (neighbor, velocity, no_scale)
#' @param exclusion_radius the norm filtering (time difference < exclusion_radius)
#' @param epsilon  the norm filtering (d < epsilon)
#' @param is_naive whether rEDM-style estimator is used
#' 
#' @return A data.frame with model parameters, RMSE, TE and p-value
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
#' block = data.frame(t = 1:tl, x = x, y = y)
#' 
#' ## cross mapping
#' op0 <- xmap(block, x_column = "x", y_column = "y", E = 2, tau = 1, tp = -1)
#' op1 <- xmap(block, x_column = "y", y_column = "x", E = 2, tau = 1, tp = -1)
#' par(mfrow = c(2, 1))
#' with(op0$model_output, plot(data, pred)); op0$stats
#' with(op1$model_output, plot(data, pred)); op1$stats
#' 
xmap = function (
    block, lib = c(1, NROW(block)), pred = lib, x_column = 1, y_column = 2, z_column = NULL,
    norm = 2, E = 1, tau = 1, tp = 0, nn = "e+1", scaling = c("neighbor", "velocity", "no_scale"),
    exclusion_radius = NULL, epsilon = NULL, is_naive = FALSE)
{
    if (length(y_column) != 1)
    {
        stop("Target column (y_column) must be a scalar.")
    }
    lib  = rbind(lib)
    pred = rbind(pred)
    
    p = pmax(0, norm)
    NORM = 2  # Lp norm
    if      (norm == 2) NORM = 0  # L2 norm
    else if (norm == 1) NORM = 1  # L1 norm
    else if (norm <= 0) NORM = 3  # Max norm
    
    if (nn == "e+1") nn = E + 1
    if (is.null(exclusion_radius)) exclusion_radius = 0;
    if (is.null(epsilon)) epsilon = -1
    LS = match.arg(scaling)
    LS = switch(LS, "no_scale" = 0, "neighbor" = 1, "velocity" = 2)
    
    x = cbind(block[,x_column])
    y = cbind(block[,y_column])
    z = matrix()
    if (!is.null(z_column)) z = cbind(block[,z_column])
    
    uic = new(rUIC)
    uic$set_norm (NORM, LS, p, exclusion_radius, epsilon)
    uic$style_ccm(is_naive)
    op = uic$xmap(x, y, z, lib, pred, E[1], nn[1], tau[1], tp[1])
    op$stats = op$stats[,-which(colnames(op$stats) == "pval")]
    op$model_output$time = op$model_output$time + 1
    op
}

# End