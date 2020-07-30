#' @name rUIC
#' @docType package
#' @title Unified Information-theoretic Causality for R.
#' @author
#'   \strong{Maintainer}  : Yutaka Osada
#'   \strong{Contributors}: Masayuki Ushio
#' @description
#'   The \pkg{rUIC} package is a experimental implementation of UIC algorithms.
#'   UIC is based on the unified theory of convergent cross mapping (CCM) and
#'   transfer entropy (TE). The functions of this package have similar interface
#'   to \pkg{rEDM} because the UIC algorithm was developed based on that of CCM.
#' @details
#'   This package is divided into a set of main functions to perform UIC.
#' \strong{Main Functions}: 
#'   \itemize{
#'     \item \code{\link{xmap}        } - cross mapping
#'     \item \code{\link{simplex}     } - simplex projection
#'     \item \code{\link{uic}         } - unified information-theoretic causality
#'     \item \code{\link{marginal_uic}} - marginal unified information-theoretic causality
#'   }
#' @keywords package
NULL

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
#' @useDynLib rUIC, .registration = TRUE
#' @importFrom methods new
#' @export xmap
#' @export simplex
#' @export uic
#' @export marginal_uic
## usethis namespace: end
"_PACKAGE"
