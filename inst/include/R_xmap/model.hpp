/*****************************************************************************
 * KNN Regresion for R
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _R_xmap_model_hpp_
#define _R_xmap_model_hpp_

#include <Rcpp.h>
#include <random> // std::mt19937
#include <vector> // std::vector

#include "model/knnregr.hpp"
#include "R_xmap/dataset.hpp"
#include "R_xmap/surrogate.hpp"


namespace UIC {
namespace R_xmap {

template <typename num_t>
struct ResultSet
{
    size_t E, E0, nn;
    size_t n_lib, n_prd, n_surr;
    num_t rmse, te, ete, pval;
};


template <typename num_t>
class Model
{
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    
    std::mt19937 &engine; // RNG engine
    KNN_TYPE knn_type;    // NN search method
    bool is_naive;
    
    size_t num_data, dim_X, dim_Y;
    R_xmap::DataSet<num_t> DataX, DataY;
    R_xmap::surrogate<num_t> surrogate;
    
public:
    
    inline Model (
        std::mt19937 &engine,
        const Rcpp::NumericMatrix &X,
        const Rcpp::NumericMatrix &Y,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &prd,
        const bool knn_t = false,  // kdtree = false(0), brute-force = true
        const bool is_naive = false)
        : engine(engine), knn_type(KNN_TYPE(knn_t)), is_naive(is_naive),
          surrogate(engine, X, Y, group, lib, prd, knn_type, is_naive)
    {
        num_data = X.rows();
        dim_X = X.cols();
        dim_Y = Y.cols();
        
        /* make datasets */
        DataX.set(X, group, lib, prd);
        DataY.set(Y, group, lib, prd);
    }
    
    inline void compute (
        R_xmap::ResultSet<num_t> *result,
        const size_t df = 1, const size_t nn = 0, const num_t p = 2,
        int exclusion_radius = -1, const num_t epsilon = -1,
        const size_t n_surr = 0)
    {
        if (exclusion_radius < 0) exclusion_radius = 0;
        size_t dim_comp = dim_X - df;
        size_t dim_full = dim_X;
        
        /* result */
        (*result).E  = dim_full;
        (*result).E0 = dim_comp;
        (*result).nn = nn;
        (*result).n_lib = DataX.trn_data.size();
        (*result).n_prd = DataX.val_data.size();
        (*result).rmse = qnan;
        (*result).te   = qnan;
        (*result).ete  = qnan;
        (*result).pval = qnan;
        (*result).n_surr = n_surr;
        if (DataX.val_data.size()==0) return;
        
        /* set knn-regresions */
        model::knnregr<num_t> regr_comp(
            dim_comp, DataX, nn, p, exclusion_radius, epsilon,
            knn_type, is_naive);
        model::knnregr<num_t> regr_full(
            dim_full, DataX, nn, p, exclusion_radius, epsilon,
            knn_type, is_naive);
        
        /* compute log-predictives */
        num_t lp_comp = calc_lp(regr_comp, DataY.trn_data, DataY.val_data);
        num_t lp_full = calc_lp(regr_full, DataY.trn_data, DataY.val_data);
        num_t TE = lp_comp - lp_full;
        (*result).rmse = std::exp(lp_full);
        (*result).te   = TE;
        if (n_surr == 0) return;
        
        /* compute ETE and p-value using surrogate data */
        (*result).ete  = 0.0;
        (*result).pval = 1.0;
        if (TE <= 0) return;
        surrogate.set(dim_comp, nn, p, exclusion_radius, epsilon);
        num_t count = 0.0;
        num_t TEsur = 0.0;
        for (size_t k = 0; k < n_surr; ++k)
        {
            /* generate surrogate data */
            std::vector<std::vector<num_t>> trn_data;
            std::vector<std::vector<num_t>> val_data;
            surrogate.generate(&trn_data, &val_data);
            /* compute TE */
            num_t lp_comp0 = calc_lp(regr_comp, trn_data, val_data);
            num_t lp_full0 = calc_lp(regr_full, trn_data, val_data);
            num_t TE0 = lp_comp0 - lp_full0;
            TEsur += (TE0 < 0 ? 0 : TE0);
            if(TE0 >= TE) ++count;
        }
        num_t ETE = TE - TEsur / num_t(n_surr);
        (*result).ete  = (ETE < 0 ? 0 : ETE);
        (*result).pval = count / num_t(n_surr);
    }
    
    Rcpp::List model_output (
        const int tp, const size_t nn = 0, const num_t p = 2,
        int exclusion_radius = -1, const num_t epsilon = -1)
    {
        if (exclusion_radius < 0) exclusion_radius = 0;
        size_t num_val = DataX.val_data.size();
        
        /* set knn-regresions */
        model::knnregr<num_t> regr(
            dim_X, DataX, nn, p, exclusion_radius, epsilon,
            knn_type, is_naive);
        
        /* prediction */
        std::vector<std::vector<num_t>> yp;
        regr.predict(yp, DataY.trn_data);
        
        /* effective number of neighbors */
        std::vector<num_t> enn;
        regr.calc_enn(enn);
        
        /* outputs for R */
        Rcpp::List out;
        for (size_t k = 0; k < dim_Y; ++k)
        {
            Rcpp::NumericVector data(num_data, qnan);
            Rcpp::NumericVector pred(num_data, qnan);
            Rcpp::NumericVector enn_(num_data, qnan);
            Rcpp::NumericVector sqe_(num_data, qnan);
            for (size_t i = 0; i < num_val; ++i)
            {
                size_t t = DataY.val_idx[i] + tp;
                /* data and predicts */
                data(t) = DataY.val_data[i][k];
                pred(t) = yp[i][k];
                /* effective number of neighbors */
                enn_(t) = enn[i];
                /* squared errors */
                num_t w = 1.0;
                if (!is_naive) w = 1.0 / (1.0 + 1.0/enn[i]);
                sqe_(t) = w * (data(t) - pred(t)) * (data(t) - pred(t));
            }
            out.push_back(
                Rcpp::DataFrame::create(
                    Rcpp::Named("data") = data,
                    Rcpp::Named("pred") = pred,
                    Rcpp::Named("enn" ) = enn_,
                    Rcpp::Named("sqe" ) = sqe_)
            );
        }
        return out;
    }
    
private:
    
    inline num_t calc_lp (
        model::knnregr<num_t> &regr,
        const std::vector<std::vector<num_t>> &trn_data,
        const std::vector<std::vector<num_t>> &val_data)
    {
        std::vector<std::vector<num_t>> pred;
        regr.predict(pred, trn_data);
        return regr.calc_lp(pred, val_data);
    }
};

} //namespace R_xmap
} //namespace UIC

#endif
//* End */