#' @name rUIC
#' @title Unified Information-theoretic Causality for R.
#' @author
#'     \strong{Maintainer}  : Yutaka Osada
#'     \strong{Contributors}: Masayuki Ushio
#' @description
#'     The \pkg{rUIC} package is an implementation of UIC algorithms.
#'     UIC is based on the unified theory of convergent cross mapping (CCM) and
#'     transfer entropy (TE). The functions of this package have similar interface
#'     to \pkg{rEDM}.
#' @details
#'     This package has three basic functions (\code{xmap}, \code{simplex} and \code{uic})
#'     and two wrapper functions (\code{uic.optimal} and \code{uic.optimal2}) to perform UIC.
#' 
#' \strong{Main Functions}: 
#'     \itemize{
#'     \item \code{\link{xmap}}    - cross mapping
#'     \item \code{\link{simplex}} - simplex projection
#'     \item \code{\link{uic}}     - unified information-theoretic causality (UIC)
#'     \item \code{\link{uic.optimal}}  - UIC using the optimal embedding dimension
#'     \item \code{\link{make_block}} - make time-lagged variables
#'     }
#' @keywords package
"_PACKAGE"
