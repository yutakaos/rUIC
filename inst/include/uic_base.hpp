/*------------------------------------------------------------------------------------------#
 * Class UIC_BASE
 *------------------------------------------------------------------------------------------#
 */

#ifndef _uic_base_hpp_
#define _uic_base_hpp_

//* Header(s) */
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include "helper.hpp"


namespace UIC
{
    //* Result set to save */
    template <typename num_t>
    struct ResultSet
    {
        int E , nn;   // E and nn of full model
        int E0, nn0;  // E and nn of reference model
        int tau, tp, n_lib, n_pred;
        num_t rmse, rmse0, uic, pval;
    };
    
    //* Model parameter set to compute UIC */
    template <typename num_t>
    struct ModelSet
    {
        //* Flags */
        bool PRED;
        //* Parameters */
        int E, nn;  // embedding dimension and number of nearest neighbors
        num_t sse;  // sum squared error
        std::vector<std::vector<num_t>> dist;  // distance
        std::vector<std::vector<int>>   neis;  // neighbors
        std::vector<num_t> se;    // squared errors
        std::vector<num_t> pred;  // predictions
        std::vector<num_t> nenn;  // number of effective nearest neighbors
    };
    
    //* Main */
    template <typename num_t>
    class UIC_BASE
    {
        typedef std::vector<std::pair<int, int>> range_t;
        typedef std::vector<std::vector<num_t>> matrix_t;
        
    protected:
        
        int E, nn, tau, tp, ndim_x, ndim_z;
        
        ResultSet<num_t> result;
        std::vector<num_t> y_lib;
        std::vector<num_t> y_prd;
        std::vector<std::vector<num_t>> x_lib; 
        std::vector<std::vector<num_t>> x_prd;
        range_t time_lib;  // [time, id]
        range_t time_prd;  // [time, id]
        range_t range_lib; // [time, time]
        range_t range_prd; // [time, time]
        
    private:
        
        int n_time; // input data length
        const int nnmax = std::numeric_limits<int>::max();
        
    public:
        
        UIC_BASE ()
        {}
        
        void set_E_and_nn (const int E_ip, const int nn_ip)
        {
            E  = E_ip;
            nn = nn_ip == 0 ? nnmax : nn_ip;
        }
        
        void set_tau (const int tau_ip)
        {
            tau = result.tau = tau_ip;
        }
        
        void set_tp (const int tp_ip)
        {
            tp = result.tp = tp_ip;
        }
        
        void set_time_indices (const range_t &range_lib_ip, const range_t &range_prd_ip)
        {
            if (range_lib_ip.size() == 0) Rcpp::stop("No time ranges for library.");
            if (range_prd_ip.size() == 0) Rcpp::stop("No time ranges for prediction.");
            
            clear_and_resize(range_lib);
            clear_and_resize(range_prd);
            range_lib = range_lib_ip;
            range_prd = range_prd_ip;
            UIC::format_time_range(&range_lib);
            UIC::format_time_range(&range_prd);
            
            range_t range_all = range_lib;
            UIC::bind(&range_all, range_prd);
            UIC::format_time_range(&range_all);
            
            UIC::set_time_indices(&time_lib, range_lib, range_all);
            UIC::set_time_indices(&time_prd, range_prd, range_all);
            if (time_lib.front().first < 0) Rcpp::stop("Negative time indices for library.");
            if (time_prd.front().first < 0) Rcpp::stop("Negative time indices for target.");
        }
        
        void set_lib (
            const std::vector<std::vector<num_t>> &X,
            std::vector<std::vector<num_t>> &Z = matrix_t(),
            bool uic = true)
        {
            n_time = X.size();
            ndim_x = X[0].size();
            if (time_lib.back().first >= n_time) Rcpp::stop("Too large time indices for library.");
            if (time_prd.back().first >= n_time) Rcpp::stop("Too large time indices for target.");
            
            if (Z.size() == 0) Z.resize(n_time);
            if (int(Z.size()) != n_time) Rcpp::stop("Different time length between X and Z.");
            ndim_z = Z[0].size();
            UIC::set_data_lib(&x_lib, Z, range_lib);
            UIC::set_data_lib(&x_prd, Z, range_prd);
            
            const bool forward = !uic;
            std::vector<std::vector<num_t>> x_lib_add, x_prd_add;
            UIC::set_data_lib(&x_lib_add, X, range_lib, E, tau, forward);
            UIC::set_data_lib(&x_prd_add, X, range_prd, E, tau, forward);
            for (size_t i = 0; i < x_lib.size(); ++i) UIC::bind(&x_lib[i], x_lib_add[i]);
            for (size_t i = 0; i < x_prd.size(); ++i) UIC::bind(&x_prd[i], x_prd_add[i]);
        }
        
        void set_tar (const std::vector<num_t> &Y)
        {
            if (int(Y.size()) != n_time) Rcpp::stop("Different time length between X and Y.");
            UIC::set_data_tar(&y_lib, Y, range_lib, tp);
            UIC::set_data_tar(&y_prd, Y, range_prd, tp);
        }
    };
}

#endif

// End