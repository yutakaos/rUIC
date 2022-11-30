/*****************************************************************************
 * Dataset class
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _dataset_hpp_
#define _dataset_hpp_

//*** Header(s) ***/
#include <Rcpp.h>
#include <limits> // std::numeric_limits
#include <vector> // std::vector


namespace UIC
{

/*** Dataset class ***/
template <typename num_t>
struct DataSet
{
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    
    /* data block */
    std::vector<std::vector<num_t>> lib_data; // library
    std::vector<std::vector<num_t>> prd_data; // prediction
    /* time and group indices */
    std::vector<std::pair<int,int>> lib_time;
    std::vector<std::pair<int,int>> prd_time;
    /* data indices */
    std::vector<size_t> lib_idx;
    std::vector<size_t> prd_idx;
    
    DataSet ()
    {}
    
    DataSet (
        const Rcpp::NumericMatrix &block,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &tar)
    {
        set(block, group, lib, tar);
    }
    
    inline void set (
        const Rcpp::NumericMatrix &block,
        const Rcpp::IntegerVector &group,
        const std::vector<bool> &lib,
        const std::vector<bool> &tar)
    {
        const int nd = block.cols();
        const int nt = block.rows();
        if (group.size() != nt) Rcpp::stop("Invalid length of group.");
        
        std::vector<num_t> data(nd);
        for (int t = 0, i = 0, k = 0, g = group(0); t < nt; ++t)
        {
            if (group(t) != /*NA*/-2147483648)
            {
                if (g != group(t)) { ++k; i = 0; g = group(t); }
                for (int k = 0; k < nd; ++k)
                {
                    num_t val = block(t,k);
                    data[k] = std::isnan(val) ? qnan : val;
                }
                if (complete_case(data))
                {
                    if (lib[t]) {
                        lib_data.push_back(data);
                        lib_time.push_back({i, k});
                        lib_idx .push_back(t);
                    }
                    if (tar[t]) {
                        prd_data.push_back(data);
                        prd_time.push_back({i, k});
                        prd_idx .push_back(t);
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

}

#endif
//* End */