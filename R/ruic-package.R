#' @name rUIC
#' @docType package
#' @title Unified Information-theoretic Causality for R.
#' @author
#'     \strong{Maintainer}  : Yutaka Osada
#'     \strong{Contributors}: Masayuki Ushio
#' @description
#'     The \pkg{rUIC} package is a experimental implementation of UIC algorithms.
#'     UIC is based on the unified theory of convergent cross mapping (CCM) and
#'     transfer entropy (TE). The functions of this package have similar interface
#'     to \pkg{rEDM} because the UIC algorithm was developed based on that of CCM.
#' @details
#'     This package has three basic functions (\code{xmap}, \code{simplex} and \code{uic})
#'     and two wrapper functions (\code{uic.optimal} and \code{uic.marginal}) to perform UIC.
#' 
#' \strong{Main Functions}: 
#'     \itemize{
#'         \item \code{\link{xmap}}    - cross mapping
#'         \item \code{\link{simplex}} - simplex projection
#'         \item \code{\link{uic}}     - unified information-theoretic causality (UIC)
#'         \item \code{\link{uic.optimal}}  - UIC using the optimal embedding dimension
#'         \item \code{\link{uic.marginal}} - UIC marginalizing embedding dimensions
#'     }
#' @keywords package
NULL
