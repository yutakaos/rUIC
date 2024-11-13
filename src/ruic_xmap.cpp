/*****************************************************************************
 * R interface for xmap algorithm
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

/**------------------------------------------------------------------------
 * Note
 * ------------------------------------------------------------------------
 * p  : power of norm 
 *    p <= 0   | Max norm
 *    p >= 1   | Lp  norm
 * nn : number of neighbors
 *    nn < 0   | E - nn
 *    nn = 0   | all data
 *    nn > 0   | nn
 * uic_type : simplex or xmap
 *    false(0) | simplex
 *    true     | xmap
 * knn_type : nearest neighbor search method 
 *    false(0) | KD-tree
 *    true     | brute-force
 * ------------------------------------------------------------------------
 */
#ifndef _ruic_xmap_cpp_
#define _ruic_xmap_cpp_

// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> // std::vector

#include "R_xmap.hpp"
#include "R_utils.hpp"

typedef double num_t;


// [[Rcpp::export]]
Rcpp::List xmap_fit_R (
    const Rcpp::NumericMatrix &xR, // library
    const Rcpp::NumericMatrix &yR, // target
    const Rcpp::NumericMatrix &zR, // condition
    Rcpp::IntegerVector &groupR,
    Rcpp::IntegerMatrix &range_lib,
    Rcpp::IntegerMatrix &range_prd,
    Rcpp::IntegerVector &E,
    Rcpp::IntegerVector &E0,
    Rcpp::IntegerVector &tau,
    Rcpp::IntegerVector &tp,
    int    nn = 0,
    double p = 2,
    int    n_surr = 100,
    int    exclusion_radius = -1,
    double epsilon = -1,
    const bool is_naive = false,
    const bool uic_type = false,
    const bool knn_type = false)
{
    int dim = E.size();
    int nt  = groupR.size();
    if (E0.size() != dim) Rcpp::stop("length(E0) must be length(E).");
    if (xR.rows() != nt)  Rcpp::stop("length(group) must be nrow(block_x).");
    if (yR.rows() != nt)  Rcpp::stop("length(group) must be nrow(block_y).");
    if (zR.rows() != nt)  Rcpp::stop("length(group) must be nrow(block_z).");
    if (0 < p && p < 1 )  Rcpp::stop("p must be >= 1 (Lp norm) or == 0 (Max norm).");
    
    /* convert data format from R to cpp */
    std::vector<std::vector<num_t>> x, y, z;
    std::vector<int> group;
    UIC::as_cpp(&x, xR);
    UIC::as_cpp(&y, yR);
    UIC::as_cpp(&z, zR);
    UIC::as_cpp(&group, groupR);
    
    /* reformat time ranges (to binary) */
    std::vector<bool> lib, prd;
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* main */
    struct Output {
        Rcpp::IntegerVector E, E0, tau, tp, nn, nn0, nl, np, ns;
        Rcpp::NumericVector rmse, te, ete, pval;
    } out;
    
    for (int tpi : tp)
    {
        for (int taui : tau)
        for (int i = 0; i < dim; ++i)
        {
            /* make datasets */
            UIC::DataSet<num_t> Data(
                x, y, z, group, lib, prd, E[i], taui, tpi, nn, p, exclusion_radius,
                epsilon, UIC::KNN_TYPE(knn_type), uic_type);
            
            /* KNN regresion */
            int df = E0[i] > 0 ? E[i]-E0[i] : E[i];
            UIC::xmap::ResultSet<num_t> result;
            UIC::xmap::Model<num_t> xmap(Data, is_naive);
            xmap.compute(&result, df, n_surr);
            
            /* output */
            out.E   .push_back(E[i]);
            out.E0  .push_back(E[i]-df);
            out.tau .push_back(taui);
            out.tp  .push_back(tpi);
            out.nn  .push_back(result.nn);
            out.nn0 .push_back(result.nn0);
            out.nl  .push_back(result.n_lib);
            out.np  .push_back(result.n_prd);
            out.rmse.push_back(result.rmse);
            out.te  .push_back(result.te);
            out.ete .push_back(result.ete);
            out.pval.push_back(result.pval);
            out.ns  .push_back(result.n_surr);
        }
    }
    return Rcpp::DataFrame::create(
        Rcpp::Named("E"     ) = out.E,
        Rcpp::Named("E0"    ) = out.E0,
        Rcpp::Named("tau"   ) = out.tau,
        Rcpp::Named("tp"    ) = out.tp,
        Rcpp::Named("nn"    ) = out.nn,
        Rcpp::Named("nn0"   ) = out.nn0,
        Rcpp::Named("n_lib" ) = out.nl,
        Rcpp::Named("n_pred") = out.np,
        Rcpp::Named("rmse"  ) = out.rmse,
        Rcpp::Named("te"    ) = out.te,
        Rcpp::Named("ete"   ) = out.ete,
        Rcpp::Named("pval"  ) = out.pval,
        Rcpp::Named("n_surr") = out.ns
    );
}


// [[Rcpp::export]]
Rcpp::List xmap_predict_R (
    const Rcpp::NumericMatrix &xR, // library
    const Rcpp::NumericMatrix &yR, // target
    const Rcpp::NumericMatrix &zR, // condition
    Rcpp::IntegerVector &groupR,
    Rcpp::IntegerMatrix &range_lib,
    Rcpp::IntegerMatrix &range_prd,
    int    E,
    int    tau,
    int    tp,
    int    nn = 0,
    double p = 2,
    int    exclusion_radius = -1,
    double epsilon = -1,
    const bool is_naive = false,
    const bool knn_type = false)
{
    int nt = groupR.size();
    if (xR.rows() != nt) Rcpp::stop("length(group) must be nrow(block_x).");
    if (yR.rows() != nt) Rcpp::stop("length(group) must be nrow(block_y).");
    if (zR.rows() != nt) Rcpp::stop("length(group) must be nrow(block_z).");
    if (0 < p && p < 1 ) Rcpp::stop("p must be >= 1 (Lp norm) or == 0 (Max norm).");
    
    /* convert data format from R to cpp */
    std::vector<std::vector<num_t>> x, y, z;
    std::vector<int> group;
    UIC::as_cpp(&x, xR);
    UIC::as_cpp(&y, yR);
    UIC::as_cpp(&z, zR);
    UIC::as_cpp(&group, groupR);
    
    /* reformat time ranges (to binary) */
    std::vector<bool> lib, prd;
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* make datasets */
    UIC::DataSet<num_t> Data(
        x, y, z, group, lib, prd, E, tau, tp, nn, p, exclusion_radius,
        epsilon, UIC::KNN_TYPE(knn_type));
    
    /* KNN regresion */
    UIC::xmap::Model<num_t> xmap(Data, is_naive);
    
    /* output */
    Rcpp::List out;
    for (auto x : xmap.model_output())
    {
        out.push_back(
        Rcpp::DataFrame::create(
            Rcpp::Named("group") = groupR,
            Rcpp::Named("data") = x.data,
            Rcpp::Named("pred") = x.pred,
            Rcpp::Named("enn" ) = x.enn,
            Rcpp::Named("sqe" ) = x.sqe)
        );
    }
    return out;
}

#endif
// End