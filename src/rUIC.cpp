/*------------------------------------------------------------------------------------------#
 * R implementation of Unified information-theoretic causality
 *------------------------------------------------------------------------------------------#
 */

#ifndef _ruic_cpp_
#define _ruic_cpp_

/* Header(s) */
#include <Rcpp.h>
#include <vector> // std::vector
#include "uic_method.hpp"
#include "as_cpp.hpp"

using namespace Rcpp;
//typedef double num_t;
typedef float num_t;


class rUIC : protected UIC::UIC_METHOD <num_t>
{
private:
    
    std::vector<UIC::ResultSet<num_t>> output;
    
public:
    
    rUIC ()
    {
        set_norm_params_R();
        set_estimator_R();
    }
    
    void set_norm_params_R (
        size_t norm_type = 1, size_t scale_type = 1, double p = 0.5,
        int exclusion_radius = 0, double epsilon = -1)
    {
        set_norm_params(
            nanoflann::NORM(norm_type), UIC::SCALE(scale_type),
            p, std::max(0, exclusion_radius), epsilon
        );
    }
    
    void set_estimator_R (bool is_naive = false)
    {
        set_estimator(is_naive);
    }
    
    List xmap_R (
        NumericMatrix time_series_lib,
        NumericVector time_series_tar,
        NumericMatrix time_series_cond,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        int E, int nn, int tau, int tp)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        
        xmap(
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_tar),
            as_cpp<num_t>(time_series_cond),
            as_cpp_range(range_lib),
            as_cpp_range(range_prd),
            {E}, {nn}, {tau}, {tp}, {0}
        );
        return List::create(
            Named("model_output") = model_output(),
            Named("stats") = model_statistics()
        );
    }
    
    DataFrame xmap_seq_R (
        NumericMatrix time_series_lib,
        NumericVector time_series_tar,
        NumericMatrix time_series_cond,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        NumericVector vector_E,
        NumericVector vector_nn,
        NumericVector vector_tau,
        NumericVector vector_tp)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        if (vector_E.size() != vector_nn.size()) Rcpp::stop("length(E) != length(nn)");
        
        xmap_seq(
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_tar),
            as_cpp<num_t>(time_series_cond),
            as_cpp_range(range_lib),
            as_cpp_range(range_prd),
            as_cpp<int>(vector_E),
            as_cpp<int>(vector_nn),
            as_cpp<int>(vector_tau),
            as_cpp<int>(vector_tp)
        );
        return model_statistics();
    }
    
    DataFrame simplex_R (
        NumericVector time_series_lib,
        NumericMatrix time_series_cond,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        int E, int nn, int tau,
        NumericVector vector_tp,
        NumericVector vector_Enull)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        if (vector_tp.size() != vector_Enull.size()) Rcpp::stop("length(tp) != length(Enull)");
        
        xmap(
            as_cpp<num_t>(Rcpp::NumericMatrix(time_series_lib)),
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_cond),
            as_cpp_range(range_lib),
            as_cpp_range(range_prd),
            {E}, {nn}, {tau},
            as_cpp<int>(vector_tp),
            as_cpp<int>(vector_Enull),
            false
        );
        return model_statistics();
    }
    
    DataFrame simplex_seq_R (
        NumericVector time_series_lib,
        NumericMatrix time_series_cond,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        NumericVector vector_E,
        NumericVector vector_nn,
        NumericVector vector_tau,
        NumericVector vector_tp)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        if (vector_E.size() != vector_nn.size()) Rcpp::stop("length(E) != length(nn)");
        
        xmap_seq(
            as_cpp<num_t>(Rcpp::NumericMatrix(time_series_lib)),
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_cond),
            as_cpp_range(range_lib),
            as_cpp_range(range_prd),
            as_cpp<int>(vector_E),
            as_cpp<int>(vector_nn),
            as_cpp<int>(vector_tau),
            as_cpp<int>(vector_tp),
            false
        );
        return model_statistics();
    }
    
private:
    
    void xmap (
        std::vector<std::vector<num_t>> time_series_lib,
        std::vector<num_t> time_series_tar,
        std::vector<std::vector<num_t>> time_series_cond,
        std::vector<std::pair<int, int>> range_lib,
        std::vector<std::pair<int, int>> range_prd,
        int E, int nn, int tau,
        std::vector<int> tp,
        std::vector<int> Enull,
        bool is_uic = true)
    {
        UIC::clear_and_resize(output);
        
        set_time_indices(range_lib, range_prd);
        set_E_and_nn(E, nn);
        set_tau(tau);
        set_lib(time_series_lib, time_series_cond, is_uic);
        set_valid_indices();
        make_dist_lib(true);
        for (size_t i = 0; i < tp.size(); ++i)
        {
            make_dist_lib(false, E - Enull[i]);
            set_tp(tp[i]);
            set_tar(time_series_tar);
            set_neighbors(true);
            set_neighbors(false);
            primitive_simplex_map(true);
            primitive_simplex_map(false);
            compute_uic();
            output.push_back(result);
        }
    }
    
    void xmap_seq (
        std::vector<std::vector<num_t>> time_series_lib,
        std::vector<num_t> time_series_tar,
        std::vector<std::vector<num_t>> time_series_cond,
        std::vector<std::pair<int, int>> range_lib,
        std::vector<std::pair<int, int>> range_prd,
        std::vector<int> E,
        std::vector<int> nn,
        std::vector<int> tau_ip,
        std::vector<int> tp_ip,
        bool is_uic = true)
    {
        UIC::clear_and_resize(output);
        
        set_time_indices(range_lib, range_prd);
        for (size_t i = 0; i < E.size(); ++i)
        {
            set_E_and_nn(E[i], nn[i]);
            for (auto tau: tau_ip)
            {
                set_tau(tau);
                set_lib(time_series_lib, time_series_cond, is_uic);
                set_valid_indices();
                make_dist_lib(true);
                make_dist_lib(false, 1);
                for (auto tp: tp_ip)
                {
                    set_tp(tp);
                    set_tar(time_series_tar);
                    set_neighbors(true);
                    set_neighbors(false);
                    primitive_simplex_map(true);
                    primitive_simplex_map(false);
                    compute_uic();
                    output.push_back(result);
                }
            }
        }
    }
    
    DataFrame model_output ()
    {
        std::vector<int> time_indices;
        for (auto x : time_prd) time_indices.push_back(x.first);
        return DataFrame::create(
            Named("time") = wrap(time_indices),
            Named("data") = wrap(y_prd),
            Named("pred") = wrap(model_full.pred),
            Named("enn" ) = wrap(model_full.nenn)
        );
    }
    
    DataFrame model_statistics ()
    {
        IntegerVector E, nn, tau, tp, ER, nnR;
        NumericVector n_lib, n_pred, rmse, rmseR, uic, pval;
        for (auto op : output)
        {
            size_t nn_ = op.nn < op.n_lib ? op.nn : op.n_lib;
            E     .push_back(op.E);
            nn    .push_back(nn_);
            tau   .push_back(op.tau);
            tp    .push_back(op.tp);
            ER    .push_back(op.E0);
            nnR   .push_back(op.nn0);
            n_lib .push_back(op.n_lib);
            n_pred.push_back(op.n_pred);
            rmse  .push_back(op.rmse);
            rmseR .push_back(op.rmse0);
            uic   .push_back(op.uic);
            pval  .push_back(op.pval);
        }
        return DataFrame::create(
            Named("E"  ) = E,
            Named("tau") = tau,
            Named("tp" ) = tp,
            Named("nn" ) = nn,
            Named("E_R" ) = ER,
            Named("nn_R") = nnR,
            Named("n_lib" ) = n_lib,
            Named("n_pred") = n_pred,
            Named("rmse"  ) = rmse,
            Named("rmse_R") = rmseR,
            Named("te"  ) = uic,
            Named("pval") = pval
        );
    }
};

// *** RCPP_MODULE *** //
RCPP_MODULE (rUIC)
{
    class_<rUIC> ("rUIC")
    .constructor()
    .method("set_norm"     , &rUIC::set_norm_params_R)
    .method("set_estimator", &rUIC::set_estimator_R)
    .method("xmap"         , &rUIC::xmap_R)
    .method("xmap_seq"     , &rUIC::xmap_seq_R)
    .method("simplex"      , &rUIC::simplex_R)
    .method("simplex_seq"  , &rUIC::simplex_seq_R);
}

#endif

// End