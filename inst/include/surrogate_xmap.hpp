/*****************************************************************************
 * Generate Surrogate Data Using Cross Map
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _surrogate_xmap_hpp_
#define _surrogate_xmap_hpp_

//*** Header(s) ***/
#include <RcppArmadillo.h>
#include <vector> // std::vectpr
#include "dataset.hpp"
#include "find_neighbors.hpp"


namespace UIC
{

/*** Surrogate ***/
template <typename num_t>
class SurrogateXmap
{
    size_t dim_X, dim_Y, num_data;
    bool is_naive;
    
    UIC::DataSet<num_t> DataX, DataY;
    std::vector<size_t> lib_idx;
    std::vector<size_t> prd_idx;
    UIC::KNN_TYPE knn_type;
    
    std::vector<std::vector<num_t>> mu;
    arma::mat Sigma;
    std::vector<arma::mat> sigma;
    
public:
    
    inline SurrogateXmap (
        const Rcpp::NumericMatrix &X,
        const Rcpp::NumericMatrix &Y,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &prd,
        const bool knn_type = false, // kdtree = false(0), brute-force = true
        const bool is_naive = false)
    {
        dim_X = X.cols();
        dim_Y = Y.cols();
        num_data = X.rows();
        this->is_naive = is_naive;
        this->knn_type = UIC::KNN_TYPE(knn_type);
        
        /* make datasets */
        std::vector<bool> all = lib;
        for (size_t i = 0; i < num_data; ++i)
        {
            if (prd[i]) all[i] = true;
        }
        DataX.set(X, group, lib, all);
        DataY.set(Y, group, lib, all);
        
        /* indices */
        UIC::DataSet<num_t> Temp(Y, group, lib, prd);
        lib_idx = Temp.lib_idx;
        prd_idx = Temp.prd_idx;
    }
    
    inline void set (
        const size_t dim_comp,
        const size_t nn = 0,
        const num_t p = 2,
        const int exclusion_radius = 0,
        const num_t epsilon = -1)
    {
        size_t num_all = DataY.prd_data.size();
        
        /* find neighbors */
        std::vector<std::vector<size_t>> idxs;
        std::vector<std::vector<num_t>>  weis;
        UIC::find_neighbors(
            dim_comp, idxs, weis, DataX, nn, p, exclusion_radius, epsilon,
            true, knn_type);
        
        /* mean */
        std::vector<std::vector<num_t>>().swap(mu);
        mu.resize(num_all);
        for (size_t t = 0; t < num_all; ++t)
        {
            mu[t].resize(dim_Y, 0);
            for (size_t k = 0; k < idxs[t].size(); ++k)
                for (size_t i = 0; i < dim_Y; ++i)
                {
                    mu[t][i] += weis[t][k] * DataY.lib_data[idxs[t][k]][i];
                }
        }
        
        /* correction coefficients */
        std::vector<num_t> cc(num_all, 1);
        if (!is_naive)
        {
            for (size_t t = 0; t < num_all; ++t)
            {
                num_t w2 = 0;
                for (auto w : weis[t]) w2 += w * w;
                cc[t] = 1.0 / (1.0 + w2);
            }
        }
        
        /* variance */
        std::vector<arma::mat>().swap(sigma);
        sigma.resize(num_all);
        Sigma = arma::zeros(dim_Y, dim_Y);
        for (size_t t = 0; t < num_all; ++t)
        {
            arma::mat &S = sigma[t];
            S = arma::mat(dim_Y, dim_Y);
            for (size_t i = 0; i < dim_Y; ++i)
            {
                num_t ei = DataY.prd_data[t][i] - mu[t][i];
                S(i,i) = cc[t] * ei * ei;
                for (size_t j = i + 1; j < dim_Y; ++j)
                {
                    num_t ej = DataY.prd_data[t][j] - mu[t][j];
                    S(i,j) = cc[t] * ei * ej;
                    S(j,i) = S(i,j);
                }
            }
            Sigma += S;
        }
        Sigma /= num_t(num_all);
    }
    
    inline void generate (
        std::vector<std::vector<num_t>> *lib_data,
        std::vector<std::vector<num_t>> *prd_data)
    {
        std::vector<std::vector<num_t>> data_all(num_data);
        for (size_t t = 0; t < mu.size(); ++t)
        {
            arma::mat &S = sigma[t];
            //arma::mat &S = Sigma;
            std::vector<num_t> &data = data_all[DataY.prd_idx[t]];
            data.resize(dim_Y);
            arma::vec mean = arma::vec  (dim_Y);
            arma::vec rand = arma::randn(dim_Y);
            data[0] = mu[t][0] + std::sqrt(S(0,0)) * rand(0);
            for (size_t i = 1; i < dim_Y; ++i)
            {
                size_t k = i - 1;
                mean(k) = data[k] - mu[t][k];
                arma::mat X = S.submat(0, 0, k, k);
                arma::mat Y = S.submat(0, i, k, i);
                arma::mat Z = arma::solve(X, Y).t();
                arma::mat m = mu[t][i] + Z * mean.subvec(0, k);
                arma::mat v = S(i,i) - Z * Y;
                data[i] = m(0,0) + std::sqrt(v(0,0)) * rand(i);
            }
        }
        /* return */
        for (size_t i : lib_idx) (*lib_data).push_back(data_all[i]);
        for (size_t i : prd_idx) (*prd_data).push_back(data_all[i]);
    }
};

}

#endif
//* End */