/*****************************************************************************
 * KNN Regresion
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _knnregr_hpp_
#define _knnregr_hpp_

//*** Header(s) ***/
//#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <vector> // std::vectpr
#include "dataset.hpp"
#include "find_neighbors.hpp"
#include "surrogate_xmap.hpp"


namespace UIC
{

/*** Reformat range ***/
inline void reformat_range (
    std::vector<bool> *binary,
    const Rcpp::IntegerMatrix &range,
    const int num_time)
{
    std::vector<bool>().swap(*binary);
    (*binary).resize(num_time, false);
    for (int i = 0; i < range.rows(); ++i)
    {
        int t0 = range(i,0) - 1;
        int t1 = range(i,1);
        for (int t = t0; t < t1; ++ t) (*binary)[t] = true;
    }
}

/*** KNN regression ***/
template <typename num_t>
struct ResultSetKR
{
    size_t E, E0, nn;
    size_t n_lib, n_prd, n_surr;
    num_t rmse, te, pval;
    //size_t tau ,tp; /* unused in KnnRegr */
};

template <typename num_t>
class KnnRegr
{
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    const UIC::DataSet<num_t> DataX, DataY;
    
    size_t num_data, num_lib, num_prd, dim_X, dim_Y;
    bool is_naive;
    UIC::KNN_TYPE knn_type;
    UIC::SurrogateXmap<num_t> Surrogate;
    
public:

    inline KnnRegr (
        const Rcpp::NumericMatrix &X,
        const Rcpp::NumericMatrix &Y,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &prd,
        const bool knn_type = false, // kdtree = false(0), brute-force = true
        const bool is_naive = false)
        : DataX(X, group, lib, prd), DataY(Y, group, lib, prd),
          Surrogate(X, Y, group, lib, prd, knn_type, is_naive)
    {
        num_data = X.rows();
        num_lib = DataX.lib_data.size();
        num_prd = DataX.prd_data.size();
        dim_X = X.cols();
        dim_Y = Y.cols();
        this->is_naive = is_naive;
        this->knn_type = UIC::KNN_TYPE(knn_type);
    }
    
    inline void compute (
        ResultSetKR<num_t> *result,
        const size_t df = 1, const size_t nn = 0, const num_t p = 2,
        int exclusion_radius = -1, const num_t epsilon = -1,
        const size_t n_surr = 0)
    {
        if (exclusion_radius < 0) exclusion_radius = 0;
        
        /* find neighbors */
        size_t dim_comp = dim_X - df;
        size_t dim_full = dim_X;
        std::vector<std::vector<size_t>> idxs_full, idxs_comp;
        std::vector<std::vector<num_t>>  weis_full, weis_comp;
        UIC::find_neighbors(
            dim_comp, idxs_comp, weis_comp, DataX, nn, p,
            exclusion_radius, epsilon, true, knn_type);
        UIC::find_neighbors(
            dim_full, idxs_full, weis_full, DataX, nn, p,
            exclusion_radius, epsilon, true, knn_type);
        
        /* compute log-liklihoods */
        std::vector<std::vector<num_t>> yp_full, yp_comp;
        predict(yp_comp, DataY.lib_data, idxs_comp, weis_comp);
        predict(yp_full, DataY.lib_data, idxs_full, weis_full);
        num_t logL_comp = loglik(DataY.prd_data, yp_comp, weis_comp);
        num_t logL_full = loglik(DataY.prd_data, yp_full, weis_full);
        num_t TE = logL_comp - logL_full;
        
        /* result */
        (*result).E  = dim_full;
        (*result).E0 = dim_comp;
        (*result).nn = nn;
        (*result).n_lib = DataX.lib_data.size();
        (*result).n_prd = DataX.prd_data.size();
        (*result).rmse = std::exp(logL_full);
        (*result).te   = TE;
        (*result).pval = 1.0;
        (*result).n_surr = n_surr;
        if (TE <= 0) return;
        
        /* compute p-value using surrogate data */
        if (n_surr == 0){
            (*result).pval = qnan;
            return;
        }
        Surrogate.set(dim_comp, nn, p, exclusion_radius, epsilon);
        num_t pval = 0.0;
        for (size_t k = 0; k < n_surr; ++k)
        {
            /* generate surrogate data */
            std::vector<std::vector<num_t>> lib_data;
            std::vector<std::vector<num_t>> prd_data;
            Surrogate.generate(&lib_data, &prd_data);
            /* compute TE */
            std::vector<std::vector<num_t>> yp_comp0;
            std::vector<std::vector<num_t>> yp_full0;
            predict(yp_comp0, lib_data, idxs_comp, weis_comp);
            predict(yp_full0, lib_data, idxs_full, weis_full);
            num_t logL_comp0 = loglik(prd_data, yp_comp0, weis_comp);
            num_t logL_full0 = loglik(prd_data, yp_full0, weis_full);
            if(logL_comp0 - logL_full0 >= TE) ++pval;
        }
        pval /= num_t(n_surr);
        (*result).pval = pval;
    }
    
    Rcpp::List model_output (
        const int tp, const size_t nn = 0, const num_t p = 2,
        int exclusion_radius = -1, const num_t epsilon = -1)
    {
        if (exclusion_radius < 0) exclusion_radius = 0;
        
        /* find neighbors */
        std::vector<std::vector<size_t>> idxs;
        std::vector<std::vector<num_t>>  weis;
        UIC::find_neighbors(
            dim_X, idxs, weis, DataX, nn, p, exclusion_radius, epsilon,
            true, knn_type);
        
        /* prediction */
        std::vector<std::vector<num_t>> yp;
        predict(yp, DataY.lib_data, idxs, weis);
        
        /* outputs for R */
        Rcpp::List out;
        for (size_t k = 0; k < dim_Y; ++k)
        {
            Rcpp::NumericVector data(num_data, qnan);
            Rcpp::NumericVector pred(num_data, qnan);
            Rcpp::NumericVector enn (num_data, qnan);
            Rcpp::NumericVector sqe (num_data, qnan);
            for (size_t i = 0; i < num_prd; ++i)
            {
                size_t t = DataY.prd_idx[i] + tp;
                /* data and predicts */
                data(t) = DataY.prd_data[i][k];
                pred(t) = yp[i][k];
                /* effective number of neighbors */
                num_t w2 = 0;
                for (auto w : weis[i]) w2 += w * w;
                enn(t) = 1.0 / w2;
                /* squared errors */
                num_t c = 1.0;
                if (!is_naive) c = 1.0 / (1.0 + w2);
                sqe(t) = c * (data(t) - pred(t)) * (data(t) - pred(t));
            }
            out.push_back(
                Rcpp::DataFrame::create(
                    Rcpp::Named("data") = data,
                    Rcpp::Named("pred") = pred,
                    Rcpp::Named("enn" ) = enn,
                    Rcpp::Named("sqe" ) = sqe)
            );
        }
        return out;
    }
    
private:
    
    inline void predict (
        std::vector<std::vector<num_t>> &yp,
        const std::vector<std::vector<num_t>>  &y,
        const std::vector<std::vector<size_t>> &idxs,
        const std::vector<std::vector<num_t>>  &weis)
    {
        std::vector<std::vector<num_t>>().swap(yp);
        yp.resize(num_prd);
        for (size_t i = 0; i < num_prd; ++i)
        {
            yp[i].resize(dim_Y, 0);
            size_t nn = idxs[i].size();
            for (size_t j = 0; j < dim_Y; ++j)
            {
                for (size_t k = 0; k < nn; ++k)
                {
                    yp[i][j] += weis[i][k] * y[idxs[i][k]][j];
                }
            }
        }
    }
    
    inline num_t loglik (
        const std::vector<std::vector<num_t>> &y,
        const std::vector<std::vector<num_t>> &yp,
        const std::vector<std::vector<num_t>> &weis)
    {
        arma::mat Sm = arma::zeros(dim_Y, dim_Y);
        for (size_t t = 0; t < num_prd; ++t)
        {
            num_t c = 1.0;
            if (!is_naive)
            {
                num_t w2 = 0;
                for (auto w : weis[t]) w2 += w * w;
                c = 1.0 / (1.0 + w2);
            }
            arma::mat S = arma::mat(dim_Y, dim_Y);
            for (size_t i = 0; i < dim_Y; ++i)
            {
                num_t ei = y[t][i] - yp[t][i];
                S(i,i) = c * ei * ei;
                for (size_t j = i + 1; j < dim_Y; ++j)
                {
                    num_t ej = y[t][j] - yp[t][j];
                    S(i,j) = c * ei * ej;
                    S(j,i) = S(i, j);
                }
            }
            Sm += S;
        }
        Sm /= num_t(num_prd);
        return 0.5 * num_t(log_det(Sm).real());
    }
};

}

#endif
//* End */