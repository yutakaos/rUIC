#' Set nn for xmap, simplex and uic functions in rUIC package
#' 
#' \code{set_nn} returns an integer vector, which is needed for xmap, simplex and uic
#' functions in rUIC package.
#' 
#' @param E
#' the embedding dimension to use for time-delay embedding.
#' @param nn
#' the number of nearest neighbors to use.
#' If \code{nn = "e+1"}, \code{nn} is set as \code{E} + 1.
#' 
#' @return
#' An integer vector.
#' 
#' @examples
#' set_nn("e+1", 1:4)  # 2 3 4 5
#' set_nn(5, 1:4)      # 5 5 5 5
#' set_nn(3:6, 1:4)    # 3 4 5 6
#' 
set_nn = function (nn, E)
{
    if (length(nn) == 1)
    {
        if (is.character(nn))
        {
            nn = paste(strsplit(tolower(nn), " ")[[1]], collapse = "")
            if (nn == "e+1") return(E + 1)
        }
        else if (as.integer(nn))
        {
            nn = rep(nn, length(E))
            return(nn)
        }
    }
    else if (length(nn) == length(E))
    {
        if (is.numeric(nn))
        {
            return(as.integer(nn))
        }
    }
    else stop("The length of nn was not equal to length(E).")
    stop("nn must be an integer, a vector of length(E) or \"e+1\".")
}

# End