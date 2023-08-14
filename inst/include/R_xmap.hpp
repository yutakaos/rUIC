/*****************************************************************************
 * KNN Regresion for R
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _R_xmap_hpp_
#define _R_xmap_hpp_

#include <Rcpp.h>

#include "R_utils.hpp"
#include "R_xmap/model.hpp"


namespace UIC {
namespace R_xmap {

template <typename num_t>
struct Output
{
    Rcpp::IntegerMatrix E, E0, tau, tp, nn, nl, np, ns;
    Rcpp::NumericMatrix rmse, te, ete, pval;
    
    void save (
        const size_t E_,
        const size_t E0_,
        const size_t tau_,
        const int tp_,
        const ResultSet<num_t> &RES)
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
        ete .push_back(RES.ete);
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
            Rcpp::Named("ete"   ) = ete,
            Rcpp::Named("pval"  ) = pval,
            Rcpp::Named("n_surr") = ns
        );
    }
};

} //namespace R_xmap


template <typename num_t = double>
inline Rcpp::DataFrame xmapR (
    std::mt19937 &engine,
    const Rcpp::NumericMatrix &block_x, // library
    const Rcpp::NumericMatrix &block_y, // target
    const Rcpp::NumericMatrix &block_z, // condition
    const Rcpp::IntegerVector &group,
    const std::vector<bool> &lib,
    const std::vector<bool> &prd,
    const Rcpp::IntegerVector &E,
    const Rcpp::IntegerVector &E0,
    const Rcpp::IntegerVector &tau,
    const Rcpp::IntegerVector &tp,
    const size_t nn = 0,
    const float p = 2,
    const size_t n_surr = 100,
    const int exclusion_radius = -1,
    const float epsilon = -1,
    const bool is_naive = false,
    const bool uic_type = false,
    const bool knn_type = false)
{
    R_xmap::Output<num_t> out;
    for (int tpi : tp)
    {
        /* make target data for KNN regression */
        Rcpp::NumericMatrix Data_y;
        UIC::make_delay_block(Data_y, block_y, group, {-tpi}, 1);
        
        for (size_t taui : tau)
        for (int i = 0; i < E.size(); ++i)
        {
            size_t Ei = E [i];
            size_t df = E0[i] > 0 ? Ei-E0[i] : Ei;
            if (taui < 1) taui = 1;
            
            /* make library data for KNN regression */
            Rcpp::NumericMatrix Data_x = block_z;
            std::vector<int> dtx = UIC::make_delay_time(Ei, uic_type);
            UIC::make_delay_block(Data_x, block_x, group, dtx, taui);
            
            /* check whther data have NAs */
            Rcpp::IntegerVector groupi = Rcpp::clone(group);
            UIC::check_NAs_group(groupi, Data_x, Data_y);
            
            /* KNN regresion */
            R_xmap::ResultSet<num_t> result;
            R_xmap::Model<num_t> xmap(
                engine, Data_x, Data_y, groupi, lib, prd, knn_type, is_naive);
            xmap.compute(&result, df, nn, p, exclusion_radius, epsilon, n_surr);
            out.save(Ei, Ei-df, taui, tpi, result);
        }
    }
    return out.get();
}


template <typename num_t = double>
Rcpp::List predict_xmapR (
    const Rcpp::NumericMatrix &block_x, // library
    const Rcpp::NumericMatrix &block_y, // target
    const Rcpp::NumericMatrix &block_z, // condition
    const Rcpp::IntegerVector &group,
    const std::vector<bool> &lib,
    const std::vector<bool> &prd,
    const size_t E,
    const size_t tau,
    const int tp,
    const size_t nn = 0,
    const float p = 2,
    const int exclusion_radius = -1,
    const float epsilon = -1,
    const bool is_naive = false,
    const bool knn_type = false)
{
    /* make library data for KNN regression */
    Rcpp::NumericMatrix Data_x = block_z;
    std::vector<int> dtx = UIC::make_delay_time(E);
    UIC::make_delay_block(Data_x, block_x, group, dtx, tau);
    
    /* make target data for KNN regression */
    Rcpp::NumericMatrix Data_y;
    UIC::make_delay_block(Data_y, block_y, group, {-tp}, 1);
    
    /* check whther data have NAs */
    Rcpp::IntegerVector group0 = Rcpp::clone(group);
    UIC::check_NAs_group(group0, Data_x, Data_y);
    
    /* KNN regresion */
    std::mt19937 engine;
    UIC::R_xmap::Model<num_t> xmap(
        engine, Data_x, Data_y, group0, lib, prd, knn_type, is_naive);
    return xmap.model_output(tp, nn, p, exclusion_radius, epsilon);
}

} //namespace UIC

#endif
//* End */