/*------------------------------------------------------------------------------------------#
 * Class UIC_BASE
 *------------------------------------------------------------------------------------------#
 */


#ifndef _ruic_base_hpp_
#define _ruic_base_hpp_

//* Header(s) */
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include <helper.hpp>


namespace UIC
{
    //* Result set to save */
    template <typename num_t>
    struct ResultSet
    {
        int E, nn, tau, tp, ER;
        int n_lib, n_pred;
        num_t rmseF, rmseR, uic, pval;
    };
    
    //* Main */
    template <typename num_t>
    class UIC_BASE
    {
        typedef std::vector<std::pair<int, int>> time_range_t;
        
    protected:
        
        int E, nn, tau, tp, n_xd;
        
        ResultSet<num_t> result;
        std::vector<num_t> y_lib;
        std::vector<num_t> y_prd;
        std::vector<std::vector<num_t>> x_lib; 
        std::vector<std::vector<num_t>> x_prd;
        std::vector<std::pair<int, int>> time_lib; // [time, id]
        std::vector<std::pair<int, int>> time_prd; // [time, id]
        
    private:
        
        const int nnmax = std::numeric_limits<int>::max();
        
        int n_time; // input data length
        time_range_t range_lib;
        time_range_t range_prd;
        
    public:
        
        UIC_BASE ()
        {}
        
        void set_E_and_nn (const int E_ip, const int nn_ip)
        {
            E  = result.E  = E_ip;
            nn = result.nn = nn_ip < 1 ? nnmax : nn_ip;
        }
        
        void set_tau (const int tau_ip)
        {
            tau = result.tau = tau_ip;
        }
        
        void set_tp (const int tp_ip)
        {
            tp = result.tp = tp_ip;
        }
        
        void set_time_indices (time_range_t &range_lib_ip, time_range_t &range_prd_ip)
        {
            if (range_lib_ip.size() == 0) Rcpp::stop("No time ranges for library.");
            if (range_prd_ip.size() == 0) Rcpp::stop("No time ranges for prediction.");
            
            time_range_t().swap(range_lib);
            time_range_t().swap(range_prd);
            range_lib = range_lib_ip;
            range_prd = range_prd_ip;
            UIC::realign_time_range(&range_lib);
            UIC::realign_time_range(&range_prd);
            
            time_range_t range_all = range_lib;
            for (auto tr : range_prd) range_all.push_back(tr);
            UIC::realign_time_range(&range_all);
            
            UIC::set_time_indices(&time_lib, range_lib, range_all);
            UIC::set_time_indices(&time_prd, range_prd, range_all);
        }
        
        void set_data_lib (const std::vector<std::vector<num_t>> &data, bool uic = true)
        {
            n_time = data.size();
            if (time_lib.front().first < 0) Rcpp::stop("Negative time indices for library.");
            if (time_prd.front().first < 0) Rcpp::stop("Negative time indices for target.");
            if (time_lib.back().first >= n_time) Rcpp::stop("Too large time indices for library.");
            if (time_prd.back().first >= n_time) Rcpp::stop("Too large time indices for target.");
            
            n_xd = data[0].size();
            UIC::set_data_lib(&x_lib, data, range_lib, E, tau, !uic);
            UIC::set_data_lib(&x_prd, data, range_prd, E, tau, !uic);
        }
        
        void set_data_mvs (const std::vector<std::vector<num_t>> data)
        {
            if (data.size() == 0) return;
            if (data.size() != size_t(n_time)) Rcpp::stop("Different time length between x and z.");
            
            size_t n_ed = x_lib[0].size();
            UIC::add_data_lib(&x_lib, data, range_lib);
            UIC::add_data_lib(&x_prd, data, range_prd);
            if (nn != nnmax) nn += (x_lib[0].size() - n_ed);
        }
        
        void set_data_tar (const std::vector<num_t> data)
        {
            if (data.size() != size_t(n_time)) Rcpp::stop("Different time length between x and y.");
            
            UIC::set_data_tar(&y_lib, data, range_lib, tp);
            UIC::set_data_tar(&y_prd, data, range_prd, tp);
        }
    };
}

#endif

// End