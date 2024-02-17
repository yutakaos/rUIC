// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// xmap_fit_R
Rcpp::List xmap_fit_R(const Rcpp::NumericMatrix& xR, const Rcpp::NumericMatrix& yR, const Rcpp::NumericMatrix& zR, Rcpp::IntegerVector& groupR, Rcpp::IntegerMatrix& range_lib, Rcpp::IntegerMatrix& range_prd, Rcpp::IntegerVector& E, Rcpp::IntegerVector& E0, Rcpp::IntegerVector& tau, Rcpp::IntegerVector& tp, int nn, double p, int n_surr, int exclusion_radius, double epsilon, const bool is_naive, const bool uic_type, const bool knn_type);
RcppExport SEXP _rUIC_xmap_fit_R(SEXP xRSEXP, SEXP yRSEXP, SEXP zRSEXP, SEXP groupRSEXP, SEXP range_libSEXP, SEXP range_prdSEXP, SEXP ESEXP, SEXP E0SEXP, SEXP tauSEXP, SEXP tpSEXP, SEXP nnSEXP, SEXP pSEXP, SEXP n_surrSEXP, SEXP exclusion_radiusSEXP, SEXP epsilonSEXP, SEXP is_naiveSEXP, SEXP uic_typeSEXP, SEXP knn_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type yR(yRSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type zR(zRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type groupR(groupRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix& >::type range_lib(range_libSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix& >::type range_prd(range_prdSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type E(ESEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type E0(E0SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type tp(tpSEXP);
    Rcpp::traits::input_parameter< int >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type n_surr(n_surrSEXP);
    Rcpp::traits::input_parameter< int >::type exclusion_radius(exclusion_radiusSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< const bool >::type is_naive(is_naiveSEXP);
    Rcpp::traits::input_parameter< const bool >::type uic_type(uic_typeSEXP);
    Rcpp::traits::input_parameter< const bool >::type knn_type(knn_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(xmap_fit_R(xR, yR, zR, groupR, range_lib, range_prd, E, E0, tau, tp, nn, p, n_surr, exclusion_radius, epsilon, is_naive, uic_type, knn_type));
    return rcpp_result_gen;
END_RCPP
}
// xmap_predict_R
Rcpp::List xmap_predict_R(const Rcpp::NumericMatrix& xR, const Rcpp::NumericMatrix& yR, const Rcpp::NumericMatrix& zR, Rcpp::IntegerVector& groupR, Rcpp::IntegerMatrix& range_lib, Rcpp::IntegerMatrix& range_prd, int E, int tau, int tp, int nn, double p, int exclusion_radius, double epsilon, const bool is_naive, const bool knn_type);
RcppExport SEXP _rUIC_xmap_predict_R(SEXP xRSEXP, SEXP yRSEXP, SEXP zRSEXP, SEXP groupRSEXP, SEXP range_libSEXP, SEXP range_prdSEXP, SEXP ESEXP, SEXP tauSEXP, SEXP tpSEXP, SEXP nnSEXP, SEXP pSEXP, SEXP exclusion_radiusSEXP, SEXP epsilonSEXP, SEXP is_naiveSEXP, SEXP knn_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type yR(yRSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type zR(zRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector& >::type groupR(groupRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix& >::type range_lib(range_libSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix& >::type range_prd(range_prdSEXP);
    Rcpp::traits::input_parameter< int >::type E(ESEXP);
    Rcpp::traits::input_parameter< int >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int >::type tp(tpSEXP);
    Rcpp::traits::input_parameter< int >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type exclusion_radius(exclusion_radiusSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< const bool >::type is_naive(is_naiveSEXP);
    Rcpp::traits::input_parameter< const bool >::type knn_type(knn_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(xmap_predict_R(xR, yR, zR, groupR, range_lib, range_prd, E, tau, tp, nn, p, exclusion_radius, epsilon, is_naive, knn_type));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rUIC_xmap_fit_R", (DL_FUNC) &_rUIC_xmap_fit_R, 18},
    {"_rUIC_xmap_predict_R", (DL_FUNC) &_rUIC_xmap_predict_R, 15},
    {NULL, NULL, 0}
};

RcppExport void R_init_rUIC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
