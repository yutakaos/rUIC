/*****************************************************************************
 * Simplex and Cross Map
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _simplex_map_cpp_
#define _simplex_map_cpp_

// [[Rcpp::plugins("cpp11")]]

//*** Header(s) ***/
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector> // std::vectpr
#include "make_delay_block.hpp"
#include "knnregr.hpp"

typedef float num_t;

template <typename num_t>
struct OUTPUT
{
    Rcpp::IntegerMatrix E, E0, tau, tp, nn, nl, np, ns;
    Rcpp::NumericMatrix rmse, te, pval;
    
    void save (
        const size_t E_, const size_t E0_, const size_t tau_, const int tp_,
        const UIC::ResultSetKR<num_t> &RES)
    {
        E   .push_back(E_);
        E0  .push_back(E0_);
        tau .push_back(tau_);
        tp  .push_back(tp_);
        nn  .push_back(RES.nn);
        nl  .push_back(RES.n_lib);
        np  .push_back(RES.n_prd);
        rmse.push_back(RES.rmse);
        te  .push_back(RES.te);
        pval.push_back(RES.pval);
        ns  .push_back(RES.n_surr);
    }
    
    Rcpp::DataFrame get ()
    {
        return Rcpp::DataFrame::create(
            Rcpp::Named("E"     ) = E,
            Rcpp::Named("E0"    ) = E0,
            Rcpp::Named("tau"   ) = tau,
            Rcpp::Named("tp"    ) = tp,
            Rcpp::Named("nn"    ) = nn,
            Rcpp::Named("n_lib" ) = nl,
            Rcpp::Named("n_pred") = np,
            Rcpp::Named("rmse"  ) = rmse,
            Rcpp::Named("te"    ) = te,
            Rcpp::Named("pval"  ) = pval,
            Rcpp::Named("n_surr") = ns
        );
    }
    
};


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

// [[Rcpp::export]]
Rcpp::List simplex_map (
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
    if (E.size() != E0.size()) Rcpp::stop("Different length between E and E0.");
    
    /* variables */
    const size_t nt = block_x.rows();
    
    /* reformated time ranges (to binary) */
    std::vector<bool> lib, prd;
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* main */
    OUTPUT<num_t> out;
    for (int tpi : tp)
    {
        /* make target data for KNN regression */
        Rcpp::NumericMatrix Data_y;
        UIC::make_delay_block(Data_y, block_y, group, {-tpi}, 1);
        
        for (size_t taui : tau)
        for (int i = 0; i < E.size(); ++i)
        {
            size_t Ei = E [i];
            size_t df = E0[i] > 0 ? Ei - E0[i] : Ei;
            if (taui < 1) taui = 1;
            
            /* make library data for KNN regression */
            Rcpp::NumericMatrix Data_x = block_z;
            std::vector<int> dtx = UIC::make_delay_time(Ei, uic_type);
            UIC::make_delay_block(Data_x, block_x, group, dtx, taui);
            
            /* check whther Data_x and Data_y have NAs */
            Rcpp::IntegerVector groupi = Rcpp::clone(group);
            size_t nx = Data_x.cols();
            size_t ny = Data_y.cols();
            for (size_t t = 0; t < nt; ++t)
            {
                bool complete = true;
                for (size_t k = 0; k < nx && complete; ++k)
                    if (std::isnan(Data_x(t,k))) complete = false;
                for (size_t k = 0; k < ny && complete; ++k)
                    if (std::isnan(Data_y(t,k))) complete = false;
                if (!complete) groupi(t) = /*NA*/-2147483648;
            }
            
            /* KNN regresion */
            UIC::ResultSetKR<num_t> result;
            UIC::KnnRegr<num_t> REGR(Data_x, Data_y, groupi, lib, prd, knn_type, is_naive);
            REGR.compute(&result, df, nn, p, exclusion_radius, epsilon, n_surr);
            out.save(Ei, Ei - df, taui, tpi, result);
        }
    }
    return out.get();
}

// [[Rcpp::export]]
Rcpp::List predict_simplex (
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
    /* variables */
    const size_t nt = block_x.rows();
    if (tau < 1) tau = 1;
    
    /* reformated time ranges (to binary) */
    std::vector<bool> lib, prd;
    UIC::reformat_range(&lib, range_lib, nt);
    UIC::reformat_range(&prd, range_prd, nt);
    
    /* make library data for KNN regression */
    Rcpp::NumericMatrix Data_x = block_z;
    std::vector<int> dtx = UIC::make_delay_time(E);
    UIC::make_delay_block(Data_x, block_x, group, dtx, tau);
    
    /* make target data for KNN regression */
    Rcpp::NumericMatrix Data_y;
    UIC::make_delay_block(Data_y, block_y, group, {-tp}, 1);
    
    /* check whther Data_x and Data_y have NAs */
    size_t nx = Data_x.cols();
    size_t ny = Data_y.cols();
    for (size_t t = 0; t < nt; ++t)
    {
        bool complete = true;
        for (size_t k = 0; k < nx && complete; ++k)
            if (std::isnan(Data_x(t,k))) complete = false;
        for (size_t k = 0; k < ny && complete; ++k)
            if (std::isnan(Data_y(t,k))) complete = false;
        if (!complete) group(t) = /*NA*/-2147483648;
    }
    
    /* KNN regresion */
    UIC::KnnRegr<num_t> REGR(Data_x, Data_y, group, lib, prd, knn_type, is_naive);
    return REGR.model_output(tp, nn, p, exclusion_radius, epsilon);
}

#endif
// End