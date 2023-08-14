/*****************************************************************************
 * R interface for UIC algorithm
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

/**------------------------------------------------------------------------
 * Note
 * ------------------------------------------------------------------------
 * p  : power of norm 
 *    p <= 0 | Max norm 
 *    p >  0 | Lp  norm
 * nn : number of neighbors
 *    nn <  0 | all data
 *    nn == 0 | E + 1
 *    nn >  0 | nn
 * uic_type : simplex or xmap
 *    false(0) | simplex
 *    true     | xmap
 * knn_type : nearest neighbor search method 
 *    false(0) | KD-tree
 *    true     | brute-force
 * ------------------------------------------------------------------------
 */
#ifndef _uic_R_cpp_
#define _uic_R_cpp_

// [[Rcpp::plugins("cpp11")]]
#include <Rcpp.h>
#include <random> // std::mt19937
#include <vector> // std::vector

#include "R_xmap.hpp"

typedef double num_t;


// [[Rcpp::export]]
Rcpp::List npmodel_R (
    const Rcpp::NumericMatrix &block_x, // library
    const Rcpp::NumericMatrix &block_y, // target
    const Rcpp::NumericMatrix &block_z, // condition
    Rcpp::IntegerVector &group,
    Rcpp::IntegerMatrix &range_lib,
    Rcpp::IntegerMatrix &range_prd,
    Rcpp::IntegerVector &E,
    Rcpp::IntegerVector &E0,
    Rcpp::IntegerVector &tau,
    Rcpp::IntegerVector &tp,
    size_t nn = 0,
    float p = 2,
    size_t n_surr = 100,
    int exclusion_radius = -1,
    float epsilon = -1,
    const bool is_naive = false,
    const bool uic_type = false,
    const bool knn_type = false)
{
    if (E.size() != E0.size()) 
        Rcpp::stop("length(E) and length(E0) must be same.");
    
    /* set RNG engine (seed is generated from R) */
    std::mt19937 engine;
    engine.seed(int(Rcpp::runif(1,1,INT32_MAX)(0)));
    
    /* reformated time ranges (to binary) */
    std::vector<bool> lib, prd;
    size_t nt = block_x.rows();
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* main */
    Rcpp::DataFrame out = UIC::xmapR<num_t>(
        engine, block_x, block_y, block_z, group, lib, prd,
        E, E0, tau, tp, nn, p, n_surr, exclusion_radius, epsilon,
        is_naive, uic_type, knn_type
    );
    return out;
}


// [[Rcpp::export]]
Rcpp::List predict_R (
    const Rcpp::NumericMatrix &block_x, // library
    const Rcpp::NumericMatrix &block_y, // target
    const Rcpp::NumericMatrix &block_z, // condition
    Rcpp::IntegerVector &group,
    Rcpp::IntegerMatrix &range_lib,
    Rcpp::IntegerMatrix &range_prd,
    size_t E,
    size_t tau,
    int tp,
    size_t nn = 0,
    float p = 2,
    int exclusion_radius = -1,
    float epsilon = -1,
    const bool is_naive = false,
    const bool knn_type = false)
{
    if (tau < 1) tau = 1;
    
    /* reformated time ranges (to binary) */
    std::vector<bool> lib, prd;
    size_t nt = block_x.rows();
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* main */
    Rcpp::List out = UIC::predict_xmapR<num_t>(
        block_x, block_y, block_z, group, lib, prd,
        E, tau, tp, nn, p, exclusion_radius, epsilon,
        is_naive, knn_type
    );
    return out;
}

#endif
// End