/*****************************************************************************
 * Utility Functions and Class for UIC
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _utils_hpp_
#define _utils_hpp_

//*** Header(s) ***/
#include <Rcpp.h>
#include <limits> // std::numeric_limits
#include <vector> // std::vector


namespace UIC
{

/*** Make delay time ***/
inline std::vector<int> make_delay_time (const int E, const bool reverse = false)
{
    std::vector<int> dt;
    if (reverse)
        for (int i = E; i > 0; --i) dt.push_back(i - 1);
    else
        for (int i = 0; i < E; ++i) dt.push_back(i);
    return dt;
}

/*** Make delay block ***/
inline void make_delay_block (
    Rcpp::NumericMatrix &block_to_add_dc,
    const Rcpp::NumericMatrix &block,
    const Rcpp::IntegerVector &group,
    const std::vector<int> &delay_time,
    const size_t tau)
{
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const int nt = block.rows();
    const int nd = block.cols();
    
    /* make group range */
    std::vector<std::vector<int>> range_grp;
    range_grp.push_back({0, 0});
    for (int t = 0, k = 0, g = group(0); t < nt; ++t)
    {
        if (group(t) != /*NA*/-2147483648)
        {
            if (g != group(t))
            {
                range_grp[k][1] = t - 1;
                range_grp.push_back({t, 0});
                g = group(t); ++k;
            }
        }
    }
    range_grp.back()[1] = nt - 1;
    
    /* add delay block to original block */
    Rcpp::NumericMatrix &B = block_to_add_dc;
    for (int dim = 0; dim < nd; ++dim)
    {
        for (int dt : delay_time)
        {
            for (auto range : range_grp)
            {
                int dtt = dt * tau;
                int L = range[0] - (dtt < 0 ? dtt : 0);
                int R = range[1] - (dtt > 0 ? dtt : 0);
                /* forward NAs */
                for (int t = 0; t < dtt ; ++t) B.push_back(qnan);
                /* input data */
                for (int t = L; t <= R  ; ++t) B.push_back(block(t,dim));
                /* backward NAs */
                for (int i = 0; i < -dtt; ++i) B.push_back(qnan);
            }
        }
    }
    B.attr("dim") = Rcpp::Dimension(nt, B.size()/nt);
    B = Rcpp::clone(B);
}

}

#endif
//* End */