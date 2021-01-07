/*------------------------------------------------------------------------------------------#
 * R implementation of Unified information-theoretic causality
 *------------------------------------------------------------------------------------------#
 */


#ifndef _ruic_cpp_
#define _ruic_cpp_

/* Header(s) */
#include <Rcpp.h>
#include <vector> // std::vector
#include <uic_method.hpp>
#include <as_cpp.hpp>

using namespace Rcpp;
typedef double num_t;


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
    
    void set_seed_R (size_t seed)
    {
        std::srand(seed);
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
    
    void set_estimator_R (bool rmse_is_naive = false)
    {
        set_estimator(rmse_is_naive);
    }
    
    List xmap_R (
        int n_boot,
        NumericMatrix time_series_lib,
        NumericVector time_series_tar,
        NumericMatrix time_series_mvs,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        int E, int nn, int tau, int tp)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        
        xmap(
            n_boot,
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_tar),
            as_cpp<num_t>(time_series_mvs),
            as_cpp_range(range_lib),
            as_cpp_range(range_prd),
            {E}, {nn}, {tau}, {tp}, {0}
        );
        return List::create(
            Named("model_output") = model_output(time_series_tar),
            Named("stats") = model_statistics()
        );
    }
    
    DataFrame xmap_seq_R (
        int n_boot,
        NumericMatrix time_series_lib,
        NumericVector time_series_tar,
        NumericMatrix time_series_mvs,
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
            n_boot,
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_tar),
            as_cpp<num_t>(time_series_mvs),
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
        int n_boot,
        NumericVector time_series_lib,
        NumericMatrix time_series_mvs,
        IntegerMatrix range_lib,
        IntegerMatrix range_prd,
        int E, int nn, int tau,
        NumericVector vector_tp,
        NumericVector vector_Enull)
    {
        if (range_lib.ncol() != 2) Rcpp::stop("ncol(lib) != 2.");
        if (range_prd.ncol() != 2) Rcpp::stop("ncol(pred) != 2.");
        if (vector_tp.size() != vector_Enull.size()) Rcpp::stop("length(tp) != length(Enull)");
        
        size_t n_time = time_series_lib.size();
        std::vector<std::vector<num_t>> ts_lib(n_time);
        for (size_t i = 0; i < n_time; ++i) ts_lib[i] = { time_series_lib(i) };
        
        xmap(
            n_boot, ts_lib,
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_mvs),
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
        int n_boot,
        NumericVector time_series_lib,
        NumericMatrix time_series_mvs,
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
        
        size_t n_time = time_series_lib.size();
        std::vector<std::vector<num_t>> ts_lib(n_time);
        for (size_t i = 0; i < n_time; ++i) ts_lib[i] = { time_series_lib(i) };
        
        xmap_seq(
            n_boot, ts_lib,
            as_cpp<num_t>(time_series_lib),
            as_cpp<num_t>(time_series_mvs),
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
        int n_boot,
        std::vector<std::vector<num_t>> time_series_lib,
        std::vector<num_t> time_series_tar,
        std::vector<std::vector<num_t>> time_series_mvs,
        std::vector<std::pair<int, int>> range_lib,
        std::vector<std::pair<int, int>> range_prd,
        int E,
        int nn,
        int tau,
        std::vector<int> tp,
        std::vector<int> ER,
        bool is_uic = true)
    {
        std::vector<UIC::ResultSet<num_t>>().swap(output); 
        
        if (n_boot < 0) n_boot = 0;
        std::vector<int> seed(n_boot);
        for (int r = 0; r < n_boot; ++r) seed[r] = std::rand();
        
        set_time_indices(range_lib, range_prd);
        set_E_and_nn(E, nn);
        set_tau(tau);
        set_data_lib(time_series_lib, is_uic);
        set_data_mvs(time_series_mvs);
        set_valid_indices();
        make_dist_lib(true);
        for (size_t i = 0; i < tp.size(); ++i)
        {
            make_dist_lib(false, E - ER[i]);
            set_tp(tp[i]);
            set_data_tar(time_series_tar);
            set_neighbors(true);
            set_neighbors(false);
            primitive_simplex_map(true);
            primitive_simplex_map(false);
            compute_uic();
            bootstrap_pval(seed);
            output.push_back(result);
        }
    }
    
    void xmap_seq (
        int n_boot,
        std::vector<std::vector<num_t>> time_series_lib,
        std::vector<num_t> time_series_tar,
        std::vector<std::vector<num_t>> time_series_mvs,
        std::vector<std::pair<int, int>> range_lib,
        std::vector<std::pair<int, int>> range_prd,
        std::vector<int> E,
        std::vector<int> nn,
        std::vector<int> tau_ip,
        std::vector<int> tp_ip,
        bool is_uic = true)
    {
        std::vector<UIC::ResultSet<num_t>>().swap(output); 
        
        if (n_boot < 0) n_boot = 0;
        std::vector<int> seed(n_boot);
        for (int r = 0; r < n_boot; ++r) seed[r] = std::rand();
        
        set_time_indices(range_lib, range_prd);
        for (size_t i = 0; i < E.size(); ++i)
        {
            set_E_and_nn(E[i], nn[i]);
            for (auto tau: tau_ip)
            {
                set_tau(tau);
                set_data_lib(time_series_lib, is_uic);
                set_data_mvs(time_series_mvs);
                set_valid_indices();
                make_dist_lib(true);
                make_dist_lib(false, 1);
                for (auto tp: tp_ip)
                {
                    set_tp(tp);
                    set_data_tar(time_series_tar);
                    set_neighbors(true);
                    set_neighbors(false);
                    primitive_simplex_map(true);
                    primitive_simplex_map(false);
                    compute_uic();
                    bootstrap_pval(seed);
                    output.push_back(result);
                }
            }
        }
    }
    
    DataFrame model_output (NumericVector data)
    {
        size_t nt = time_prd.size();
        std::vector<int> time_indices(nt);
        for (size_t t = 0; t < nt; ++t) time_indices[t] = time_prd[t].first;
        
        return DataFrame::create(
            Named("time") = wrap(time_indices),
            Named("data") = wrap(y_prd),
            Named("pred") = wrap(pred_full),
            Named("enn" ) = wrap(nenn_full)
        );
    }
    
    DataFrame model_statistics ()
    {
        IntegerVector E, nn, tau, tp, ER;
        NumericVector n_lib, n_pred, rmseF, rmseR, uic, pval;
        for (auto op : output)
        {
            size_t nn_ = op.nn < op.n_lib ? op.nn : op.n_lib;
            E     .push_back(op.E);
            nn    .push_back(nn_);
            tau   .push_back(op.tau);
            tp    .push_back(op.tp);
            ER    .push_back(op.ER);
            n_lib .push_back(op.n_lib);
            n_pred.push_back(op.n_pred);
            rmseF .push_back(op.rmseF);
            rmseR .push_back(op.rmseR);
            uic   .push_back(op.uic);
            pval  .push_back(op.pval);
        }
        return DataFrame::create(
            Named("E"  ) = E,
            Named("tau") = tau,
            Named("tp" ) = tp,
            Named("nn" ) = nn,
            Named("Enull" ) = ER,
            Named("n_lib" ) = n_lib,
            Named("n_pred") = n_pred,
            Named("rmse"  ) = rmseF,
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
    .method("set_seed"     , &rUIC::set_seed_R)
    .method("set_norm"     , &rUIC::set_norm_params_R)
    .method("set_estimator", &rUIC::set_estimator_R)
    .method("xmap"         , &rUIC::xmap_R)
    .method("xmap_seq"     , &rUIC::xmap_seq_R)
    .method("simplex"      , &rUIC::simplex_R)
    .method("simplex_seq"  , &rUIC::simplex_seq_R);
}

#endif

// End