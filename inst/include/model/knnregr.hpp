/*****************************************************************************
 * KNN Regresion
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _model_knnregr_hpp_
#define _model_knnregr_hpp_

#include <vector>     // std::vector
#include <functional> // std::function
//#include <Eigen/Cholesky>

#include "./find_neighbors.hpp"


namespace UIC {
namespace model {

/*---------------------------------------------------------------------------*
 | KNN regression
 *---------------------------------------------------------------------------*/
template <typename num_t>
class knnregr
{
    using EigenMatrix = Eigen::Matrix<num_t,Eigen::Dynamic,Eigen::Dynamic>;
    using data_t = std::vector<num_t>;
    
    const int num_val, dim_y;
    std::function<num_t(size_t)> calc_cc;
    std::vector<std::vector<size_t>> idxs;
    std::vector<std::vector<num_t >> weis;
    
public:
    
    inline knnregr (
        const int dim,
        const int nn,
        const DataSet<num_t>   &Data,
        const std::vector<int> &idx_val,
        const bool is_naive = false)
        : num_val(idx_val.size()), dim_y(Data.dim_y)
    {
        /* define function for corrections */
        set_calc_cc(is_naive);
        
        /* find neighbors */
        UIC::find_neighbors(dim, nn, idxs, weis, Data, idx_val);
        
        /* convert distances to weights for simplex projection */
        for (auto &ds : weis)
        {
            if (ds[0] == 0.0)
                for (auto &d : ds) d = (d == 0.0 ? 1.0 : 0.0); 
            else {
                num_t mind = ds[0];
                for (auto &d : ds) d = std::exp(-d / mind);
            }
            num_t sumw = 0;
            for (auto  w : ds) sumw += w;
            for (auto &w : ds) w /= sumw;
        }
    }
    
    /* predictions */
    template <typename vec_t>  // vec_t = std::vector<num_t> or Eigen::VectorXd
    inline void predict (
        std::vector<vec_t> &pred,
        const std::vector<data_t> &trn_data)
    {
        std::vector<vec_t>().swap(pred);
        pred.resize(num_val);
        for (int t = 0; t < num_val; ++t) pred[t] = mean<vec_t>(t, trn_data);
    }
    
    /* calculate log-predictives */
    inline num_t calc_lp (
        const std::vector<data_t> &trn_data,
        const std::vector<data_t> &val_data)
    {
        std::vector<data_t> pred;
        predict(pred, trn_data);
        return LP(pred, val_data);
    }
    
    /* calculate effective number of neighbors */
    inline void calc_enn (std::vector<num_t> &enn)
    {
        std::vector<num_t>().swap(enn);
        enn.resize(num_val);
        for (int t = 0; t < num_val; ++t)
        {
            num_t w2 = 0;
            for (auto w : weis[t]) w2 += w * w;
            enn[t] = 1.0 / w2;
        }
    }
    
    /* calculate covariance matrix */
    template <typename vec_t>
    inline void calc_SL (
        std::vector<EigenMatrix>  &SL,
        const std::vector<vec_t>  &pred,
        const std::vector<data_t> &val_data)
    {
        std::vector<data_t> e2d(num_val);
        std::vector<num_t>  enn(num_val);
        for (int t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            e2d[t] = calc_e2d(pred[t], val_data[t], w);
            enn[t] = w / (1.0 - w);
        }
        std::vector<data_t> s2d(num_val);
        for (int t = 0; t < num_val; ++t) s2d[t] = mean<data_t>(t, e2d);
        
        std::vector<EigenMatrix>().swap(SL);
        SL.resize(num_val);
        for (int t = 0; t < num_val; ++t)
        {
            SL[t] = calc_S2t(e2d[t], s2d[t], enn[t]).llt().matrixL();
        }
    }
    
private:
    
    /* set function for corrections */
    inline void set_calc_cc (const bool is_naive)
    {
        if (is_naive)
            calc_cc = [](size_t) { return 1.0; };
        else
            calc_cc = [this](size_t t)
            {
                num_t w2 = 0;
                for (auto w : weis[t]) w2 += w * w;
                return 1.0 / (1.0 + w2);
            };
    }
    
    /* weighted mean */
    template <typename vec_t>
    inline vec_t mean (int t, const std::vector<data_t> &data)
    {
        int nn  = idxs[t].size();
        int dim = data[0].size();
        vec_t pred(dim);
        for (int i = 0; i < dim; ++i)
        {
            pred[i] = 0;
            for (int k = 0; k < nn; ++k)
                pred[i] += weis[t][k] * data[idxs[t][k]][i];
        }
        return pred;
    }
    
    /* calculate log-predictive (temporally homogeneous) */
    inline num_t LP (
        const std::vector<data_t> &yhat,
        const std::vector<data_t> &yval)
    {
        EigenMatrix Sm = EigenMatrix::Zero(dim_y,dim_y);
        for (int t = 0; t < num_val; ++t)
        {
            Sm += calc_e2(yhat[t], yval[t], calc_cc(t));
        }
        Sm /= num_t(num_val);
        return 0.5 * logdet(Sm);
    }
    
    /* calculate log-predictive (temporally heterogeneous)
    inline num_t LP2 (
        const std::vector<data_t> &yhat,
        const std::vector<data_t> &yval)
    {
        std::vector<data_t> e2d(num_val);
        std::vector<num_t>  enn(num_val);
        for (size_t t = 0; t < num_val; ++t)
        {
            num_t w = calc_cc(t);
            e2d[t] = calc_e2d(yval[t], yhat[t], w);
            enn[t] = w / (1.0 - w);
        }
        std::vector<data_t> s2d(num_val);
        for (int t = 0; t < num_val; ++t) s2d[t] = mean<data_t>(t, e2d);
        
        num_t lp = 0.0;
        for (size_t t = 0; t < num_val; ++t)
        {
            EigenMatrix S2t = calc_S2t(e2d[t], s2d[t], enn[t]);
            lp += logdet(S2t);
        }
        return 0.5 * lp / num_t(num_val);
    } //*/
    
    inline num_t logdet (const EigenMatrix &Mat)
    {
        num_t L_logdet = 
            Eigen::LLT<EigenMatrix>(Mat).matrixL().toDenseMatrix()
            .diagonal().array().log().sum();
        return 2.0 * L_logdet;
    }
    
    template <typename vec_t>
    inline EigenMatrix calc_e2 (
        const vec_t &yhat, const data_t &yval, const num_t w)
    {
        EigenMatrix e2(dim_y,dim_y);
        for (int i = 0; i < dim_y; ++i)
        {
            num_t ei = yval[i] - yhat[i];
            e2(i,i) = w * ei * ei;
            for (int j = i + 1; j < dim_y; ++j)
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
        const vec_t &yhat, const data_t &yval, const num_t w)
    {
        size_t K = dim_y * (dim_y + 1) / 2;
        std::vector<num_t> e2d(K);
        for (int i = 0, k = 0; i < dim_y; ++i)
        {
            num_t ei = yval[i] - yhat[i];
            e2d[k++] = w * ei * ei;
            for (int j = i + 1; j < dim_y; ++j)
            {
                num_t ej = yval[j] - yhat[j];
                e2d[k++] = w * ei * ej;
            }
        }
        return e2d;
    }
    
    inline EigenMatrix calc_S2t (
        const data_t &e2d, const data_t &s2d, const num_t enn)
    {
        EigenMatrix S2(dim_y, dim_y);
        for (int i = 0, k = 0; i < dim_y; ++i)
        {
            S2(i,i) = (enn*s2d[k] + e2d[k]) / (enn + 1.0);
            ++k;
            for (int j = i + 1; j < dim_y; ++j, ++k)
            {
                S2(i,j) = (enn*s2d[k] + e2d[k]) / (enn + 1.0);
                S2(j,i) = S2(i,j);
            }
        }
        return S2;
    }
};

} //namespace model
} //namespace UIC

#endif
//* End */