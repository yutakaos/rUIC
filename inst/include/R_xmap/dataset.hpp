/*****************************************************************************
 * Dataset class (with R interface)
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _R_xmap_dataset_hpp_
#define _R_xmap_dataset_hpp_

#include <Rcpp.h>
#include <limits> // std::numeric_limits
#include <vector> // std::vector


namespace UIC {
namespace R_xmap {

/*---------------------------------------------------------------------------*
 | Dataset class (with R interface)
 *---------------------------------------------------------------------------*/
template <typename num_t>
struct DataSet
{
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    
    /* data block */
    std::vector<std::vector<num_t>> trn_data; // training
    std::vector<std::vector<num_t>> val_data; // validation
    /* time and group indices */
    std::vector<std::pair<int,int>> trn_time;
    std::vector<std::pair<int,int>> val_time;
    /* data indices */
    std::vector<size_t> trn_idx;
    std::vector<size_t> val_idx;
    
    DataSet ()
    {}
    
    DataSet (
        const Rcpp::NumericMatrix &block,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &trn,
        const std::vector<bool> &val)
    {
        set(block, group, trn, val);
    }
    
    inline void set (
        const Rcpp::NumericMatrix &block,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &trn,
        const std::vector<bool> &val)
    {
        const int nd = block.cols();
        const int nt = block.rows();
        if (group.size() != nt) Rcpp::stop("length(group) must be nrow(block).");
        
        std::vector<num_t> data(nd);
        for (int t = 0, i = 0, k = 0, g = group(0); t < nt; ++t)
        {
            if (group(t) != /*NA*/-2147483648) //INT32_MIN
            {
                if (g != group(t)) { ++k; i = 0; g = group(t); }
                for (int k = 0; k < nd; ++k)
                {
                    num_t val = block(t,k);
                    data[k] = std::isnan(val) ? qnan : val;
                }
                if (complete_case(data))
                {
                    if (trn[t]) {
                        trn_data.push_back(data);
                        trn_time.push_back({i, k});
                        trn_idx .push_back(t);
                    }
                    if (val[t]) {
                        val_data.push_back(data);
                        val_time.push_back({i, k});
                        val_idx .push_back(t);
                    }
                }
            }
            ++i;
        }
    }
    
private:
    
    inline bool complete_case (std::vector<num_t> &x) 
    { 
        for (auto xi : x) if (std::isnan(xi)) return false;
        return true; // check NAs
    }
};

} //namespace R_xmap
} //namespace UIC

#endif
//* End */