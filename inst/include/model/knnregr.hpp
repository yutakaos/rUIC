/*****************************************************************************
 * KNN Regresion
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _model_knnregr_hpp_
#define _model_knnregr_hpp_

#include <vector>     // std::vector
#include <functional> // std::function
#include <ext/Eigen/Cholesky>

#include "nns/find_neighbors.hpp"


namespace UIC {
namespace model {

template <typename num_t>
class knnregr
{
    using EigenMatrix = Eigen::Matrix<num_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    size_t num_val;
    size_t dim;
    std::vector<std::vector<size_t>> idxs;
    std::vector<std::vector<num_t >> weis;
    std::function<num_t(size_t)> calc_cc;
    
public:
    
    /*-----------------------------------------------------------------------*
     | DataX : dataset, which has the following member variables:
     |    - DataX.trn_data : training data
     |    - DataX.val_data : validation data
     |    - DataX.trn_time : time indices of training data
     |    - DataX.val_time : time indices of validation data
     *-----------------------------------------------------------------------*/
    template <typename DataSet_t>
    inline knnregr (
        const size_t dim,
        const DataSet_t &DataX,
        const size_t nn = 0,
        const num_t p = 2,
        const int exclusion_radius = -1,
        const num_t epsilon = -1,
        const UIC::KNN_TYPE knn_type = UIC::KNN_TYPE::KD,
        const bool is_naive = false)
    {
        num_val = DataX.val_data.size();
        
        /* define function for corrections */
        if (is_naive)
            calc_cc = [](size_t) { return 1.0; };
        else {
            calc_cc = [this](size_t t)
            {
                num_t w2 = 0;
                for (auto w : weis[t]) w2 += w * w;
                return 1.0 / (1.0 + w2);
            };
        }
        
        /* find neighbors */
        UIC::find_neighbors(
            dim, idxs, weis, DataX, nn, p, exclusion_radius, epsilon,
            true, knn_type);
    }
    
    // vec_t = std::vector<num_t> or Eigen::VectorXd
    template <typename vec_t>
    inline void predict (
        std::vector<vec_t> &yhat,
        const std::vector<std::vector<num_t>> &ytrn)
    {
        size_t dim = ytrn[0].size();
        std::vector<vec_t>().swap(yhat);
        yhat.resize(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            yhat[t].resize(dim);
            size_t nn = idxs[t].size();
            for (size_t i = 0; i < dim; ++i)
            {
                yhat[t][i] = 0;
                for (size_t k = 0; k < nn; ++k)
                {
                    yhat[t][i] += weis[t][k] * ytrn[idxs[t][k]][i];
                }
            }
        }
    }
    
    /* calculate squared errors
    template <typename vec_t>
    inline void calc_E2 (
        std::vector<EigenMatrix> &Sm,
        const std::vector<vec_t> &yhat,
        const std::vector<std::vector<num_t>> &yval)
    {
        dim = yval[0].size();
        std::vector<EigenMatrix>().swap(Sm);
        Sm.resize(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            Sm[t] = calc_e2(yval[t], yhat[t], w);
        }
    } //*/
    
    template <typename vec_t>
    inline void calc_S2 (
        std::vector<EigenMatrix> &S2,
        const std::vector<vec_t> &yhat,
        const std::vector<std::vector<num_t>> &yval)
    {
        dim = yval[0].size();
        std::vector<std::vector<num_t>> e2d(num_val);
        std::vector<std::vector<num_t>> s2d;
        std::vector<num_t> enn(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            e2d[t] = calc_e2d(yval[t], yhat[t], w);
            enn[t] = w / (1.0 - w);
        }
        predict(s2d, e2d);
        
        std::vector<EigenMatrix>().swap(S2);
        S2.resize(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            S2[t] = calc_S2t(e2d[t], s2d[t], enn[t]);
        }
    }
    
    /* calculate log-predictive each time
    template <typename vec_t>
    inline num_t calc_lp (
        const std::vector<vec_t> &yhat,
        const std::vector<std::vector<num_t>> &yval)
    {
        dim = yval[0].size();
        std::vector<std::vector<num_t>> e2d(num_val);
        std::vector<std::vector<num_t>> s2d;
        std::vector<num_t> enn(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            e2d[t] = calc_e2d(yval[t], yhat[t], w);
            enn[t] = w / (1.0 - w);
        }
        predict(s2d, e2d);
        
        num_t lp = 0.0;
        for (size_t t = 0; t < num_val; ++t)
        {
            EigenMatrix S2t = calc_S2t(e2d[t], s2d[t], enn[t]);
            lp += logdet(S2t);
        }
        return 0.5 * lp / num_t(num_val);
    } //*/
    
    //* calculate log-predictives (temporally homogeneous)
    template <typename vec_t>
    inline num_t calc_lp (
        const std::vector<vec_t> &yhat,
        const std::vector<std::vector<num_t>> &yval)  // validation
    {
        dim = yval[0].size();
        EigenMatrix Sm = EigenMatrix::Zero(dim,dim);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            Sm += calc_e2(yval[t], yhat[t], w);
        }
        Sm /= num_t(num_val);
        return 0.5 * logdet(Sm);
    } //*/
    
    inline void calc_enn (std::vector<num_t> &enn)
    {
        std::vector<num_t>().swap(enn);
        enn.resize(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w2 = 0;
            for (auto w : weis[t]) w2 += w * w;
            enn[t] = 1.0 / w2;
        }
    }
    
private:
    
    template <typename vec_t>
    inline EigenMatrix calc_e2 (
        const std::vector<num_t> &yval, const vec_t &yhat,
        const num_t w)
    {
        EigenMatrix e2(dim,dim);
        for (size_t i = 0; i < dim; ++i)
        {
            num_t ei = yval[i] - yhat[i];
            e2(i,i) = w * ei * ei;
            for (size_t j = i + 1; j < dim; ++j)
            {
                num_t ej = yval[j] - yhat[j];
                e2(i,j) = w * ei * ej;
                e2(j,i) = e2(i,j);
            }
        }
        return e2;
    }
    
    template <typename vec_t>
    inline std::vector<num_t> calc_e2d (
        const std::vector<num_t> &yval, const vec_t &yhat,
        const num_t w)
    {
        size_t K = dim * (dim + 1) / 2;
        std::vector<num_t> e2d(K);
        for (size_t i = 0, k = 0; i < dim; ++i)
        {
            num_t ei = yval[i] - yhat[i];
            e2d[k++] = w * ei * ei;
            for (size_t j = i + 1; j < dim; ++j)
            {
                num_t ej = yval[j] - yhat[j];
                e2d[k++] = w * ei * ej;
            }
        }
        return e2d;
    }
    
    inline EigenMatrix calc_S2t (
        const std::vector<num_t> &e2d,
        const std::vector<num_t> &s2d,
        const num_t enn)
    {
        EigenMatrix S2(dim,dim);
        for (size_t i = 0, k = 0; i < dim; ++i)
        {
            S2(i,i) = (enn*s2d[k] + e2d[k]) / (enn + 1.0);
            ++k;
            for (size_t j = i + 1; j < dim; ++j, ++k)
            {
                S2(i,j) = (enn*s2d[k] + e2d[k]) / (enn + 1.0);
                S2(j,i) = S2(i,j);
            }
        }
        return S2;
    }
    
    inline num_t logdet (const EigenMatrix &Mat)
    {
        num_t L_logdet = 
            Eigen::LLT<EigenMatrix>(Mat).matrixL().toDenseMatrix()
            .diagonal().array().log().sum();
        return 2.0 * L_logdet;
    }
};

} //namespace model
} //namespace UIC

#endif
//* End */