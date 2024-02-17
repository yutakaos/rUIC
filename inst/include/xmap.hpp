/*****************************************************************************
 * KNN Regresion for R
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _model_xmap_hpp_
#define _model_xmap_hpp_

#include <random> // std::mt19937
#include <vector> // std::vector
//#include <Eigen/Dense>

#include "model/knnregr.hpp"
#include "random/mvnormal.hpp"


namespace UIC {
namespace xmap {

/*---------------------------------------------------------------------------*
 | Result sets
 *---------------------------------------------------------------------------*/
template <typename num_t>
struct ResultSet
{
    int E, E0, nn, n_lib, n_prd, n_surr;
    num_t rmse, te, ete, pval;
};

template <typename num_t>
struct Output
{
    std::vector<num_t> data, pred, enn, sqe;
};


/*---------------------------------------------------------------------------*
 | Xmap surrogate
 *---------------------------------------------------------------------------*/
template <typename num_t>
class Surrogate
{
    using EigenVector = Eigen::Matrix<num_t,Eigen::Dynamic,1>;
    using EigenMatrix = Eigen::Matrix<num_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    std::mt19937 &engine;
    const DataSet<num_t> &Data;
    const bool is_naive;
    
    /* parameters */
    std::vector<EigenVector> mu;
    std::vector<EigenMatrix> sigma;
    
public:
    
    inline Surrogate (
        std::mt19937 &engine,
        const DataSet<num_t> &Data,
        const bool is_naive = false)
        : engine(engine), Data(Data), is_naive(is_naive)
    {}
    
    inline void set (const size_t dim_comp)
    {
        /* set knn-regresions */
        model::knnregr<num_t> regr(dim_comp, Data, Data.idx_all, is_naive);
        /* predictions */
        std::vector<std::vector<num_t>> Y_val; // for surrogate data
        for (auto t : Data.idx_all) Y_val.push_back(Data.Y(t));
        regr.predict(mu,   Data.Y_trn);
        regr.calc_S2(sigma, mu, Y_val);
    }
    
    inline void generate (
        std::vector<std::vector<num_t>> *trn_data,
        std::vector<std::vector<num_t>> *val_data)
    {
        std::vector<std::vector<num_t>> data_all(Data.num_data);
        for (size_t t = 0; t < mu.size(); ++t)
        {
            std::vector<num_t> &data = data_all[Data.idx_all[t]];
            data.resize(Data.dim_y);
            Rand_Eigen::mvnormal<num_t> rand_(mu[t], sigma[t]);
            EigenVector x = rand_(engine);
            for (int i = 0; i < Data.dim_y; ++i) data[i] = x(i);
        }
        /* return */
        for (size_t i : Data.idx_trn) (*trn_data).push_back(data_all[i]);
        for (size_t i : Data.idx_val) (*val_data).push_back(data_all[i]);
    }
};


template <typename num_t>
class Model
{
    using data_t = std::vector<num_t>;
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    
    std::mt19937 &engine; // RNG engine
    const DataSet<num_t> &Data;
    const bool is_naive;
    xmap::Surrogate<num_t> surrogate;
    
public:
    
    inline Model (
        std::mt19937 &engine, const DataSet<num_t> &Data,
        const bool is_naive = false)
        : engine(engine), Data(Data), is_naive(is_naive),
          surrogate(engine, Data, is_naive)
    {}
    
    inline void compute (
        xmap::ResultSet<num_t> *result,
        const int df = 1,
        const int n_surr = 0)
    {
        int dim_comp = Data.dim_x - df;
        int dim_full = Data.dim_x;
        
        /* result */
        (*result).E  = dim_full;
        (*result).E0 = dim_comp;
        (*result).nn = Data.nn;
        (*result).n_lib = Data.num_trn;
        (*result).n_prd = Data.num_val;
        (*result).rmse = qnan;
        (*result).te   = qnan;
        (*result).ete  = qnan;
        (*result).pval = qnan;
        (*result).n_surr = n_surr;
        if (Data.num_val == 0) return;
        
        /* set knn-regresions */
        model::knnregr<num_t> regr_comp(dim_comp, Data, Data.idx_val, is_naive);
        model::knnregr<num_t> regr_full(dim_full, Data, Data.idx_val, is_naive);
        
        /* compute log-predictives */
        num_t lp_comp = regr_comp.calc_lp(Data.Y_trn, Data.Y_val);
        num_t lp_full = regr_full.calc_lp(Data.Y_trn, Data.Y_val);
        num_t TE = lp_comp - lp_full;
        (*result).rmse = std::exp(lp_full);
        (*result).te   = TE;
        if (n_surr == 0) return;
        
        /* compute ETE and p-value using surrogate data */
        (*result).ete  = 0.0;
        (*result).pval = 1.0;
        if (TE <= 0) return;
        
        surrogate.set(dim_comp);
        num_t count = 0.0;
        num_t TEsur = 0.0;
        for (int k = 0; k < n_surr; ++k)
        {
            /* generate surrogate data */
            std::vector<data_t> Y_trn, Y_val;
            surrogate.generate(&Y_trn, &Y_val);
            /* compute TE */
            num_t lp_comp0 = regr_comp.calc_lp(Y_trn, Y_val);
            num_t lp_full0 = regr_full.calc_lp(Y_trn, Y_val);
            num_t TE0 = lp_comp0 - lp_full0;
            TEsur += (TE0 < 0 ? 0 : TE0);
            if(TE0 >= TE) ++count;
        }
        num_t ETE = TE - TEsur / num_t(n_surr);
        (*result).ete  = (ETE < 0 ? 0 : ETE);
        (*result).pval = count / num_t(n_surr);
    }
    
    inline std::vector<Output<num_t>> model_output()
    {
        /* prediction by knn-regresion */
        std::vector<data_t> yp; // prediction
        std::vector<num_t> enn; // effective number of neighbors
        model::knnregr<num_t> regr(Data.dim_x, Data, Data.idx_val, is_naive);
        regr.predict(yp, Data.Y_trn);
        regr.calc_enn(enn);
        
        /* output */
        std::vector<Output<num_t>> out(Data.dim_y);
        for (int k = 0; k < Data.dim_y; ++k)
        {
            out[k].data.resize(Data.num_data, qnan);
            out[k].pred.resize(Data.num_data, qnan);
            out[k].enn .resize(Data.num_data, qnan);
            out[k].sqe .resize(Data.num_data, qnan);
            for (int i = 0; i < Data.num_val; ++i)
            {
                size_t t = Data.idx_val[i] + Data.tp;
                num_t  w = is_naive ? 1.0 : 1.0 / (1.0 + 1.0/enn[i]);
                num_t  e = Data.Y_val[i][k] - yp[i][k];
                out[k].data[t] = Data.Y_val[i][k];
                out[k].pred[t] = yp[i][k];
                out[k].enn [t] = enn[i];
                out[k].sqe [t] = w * e * e;
            }
        }
        return out;
    }
};

} //namespace R_xmap
} //namespace UIC

#endif
//* End */