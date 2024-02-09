/*****************************************************************************
 * Generate Surrogate Data Using Cross Map
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _R_xmap_surrogate_hpp_
#define _R_xmap_surrogate_hpp_

#include <Rcpp.h>
#include <random> // std::mt19937
#include <vector> // std::vector
//#include <Eigen/Dense>

#include "model/knnregr.hpp"
#include "random/mvnormal.hpp"
#include "R_xmap/dataset.hpp"


namespace UIC {
namespace R_xmap {

template <typename num_t>
class surrogate
{
    using EigenVector = Eigen::Matrix<num_t,Eigen::Dynamic,1>;
    using EigenMatrix = Eigen::Matrix<num_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    std::mt19937 &engine;   // RNG engine
    UIC::KNN_TYPE knn_type; // NN search method
    bool is_naive;
    
    /* data */
    size_t dim_Y, num_data;
    R_xmap::DataSet<num_t> DataX, DataY;
    std::vector<size_t> trn_idx;
    std::vector<size_t> val_idx;
    /* parameters */
    std::vector<EigenVector> mu;
    std::vector<EigenMatrix> sigma;
    
public:
    
    inline surrogate (
        std::mt19937 &engine,
        const Rcpp::NumericMatrix &X,
        const Rcpp::NumericMatrix &Y,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &prd,
        const UIC::KNN_TYPE knn_type = UIC::KNN_TYPE::KD,
        const bool is_naive = false)
        : engine(engine), knn_type(knn_type), is_naive(is_naive)
    {
        num_data = X.rows();
        dim_Y = Y.cols();
        
        /* make datasets */
        std::vector<bool> all = lib;
        for (size_t i = 0; i < num_data; ++i) if (prd[i]) all[i] = true;
        DataX.set(X, group, lib, all);
        DataY.set(Y, group, lib, all);
        
        /* indices for training and validation data */
        R_xmap::DataSet<num_t> tmp(Y, group, lib, prd);
        trn_idx = tmp.trn_idx;
        val_idx = tmp.val_idx;
    }
    
    inline void set (
        const size_t dim_comp,
        const size_t nn = 0,
        const num_t p = 2,
        const int exclusion_radius = 0,
        const num_t epsilon = -1)
    {
        /* set knn-regresions */
        model::knnregr<num_t> regr(
            dim_comp, DataX, nn, p, exclusion_radius, epsilon,
            knn_type, is_naive);
        
        /* predictions */
        regr.predict(       mu, DataY.trn_data);
        regr.calc_S2(sigma, mu, DataY.val_data);
        //regr.calc_E2(sigma, mu, DataY.val_data);
    }
    
    inline void generate (
        std::vector<std::vector<num_t>> *trn_data,
        std::vector<std::vector<num_t>> *val_data)
    {
        std::vector<std::vector<num_t>> data_all(num_data);
        for (size_t t = 0; t < mu.size(); ++t)
        {
            std::vector<num_t> &data = data_all[DataY.val_idx[t]];
            data.resize(dim_Y);
            Rand_Eigen::mvnormal<num_t> rand_(mu[t], sigma[t]);
            EigenVector x = rand_(engine);
            for (size_t i = 0; i < dim_Y; ++i) data[i] = x(i);
        }
        /* return */
        for (size_t i : trn_idx) (*trn_data).push_back(data_all[i]);
        for (size_t i : val_idx) (*val_data).push_back(data_all[i]);
    }
};

} //namespace R_xmap
} //namespace UIC

#endif
//* End */