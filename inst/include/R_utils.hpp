/*****************************************************************************
 * Utility Functions
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _R_utils_hpp_
#define _R_utils_hpp_

#include <Rcpp.h>
#include <limits> // std::numeric_limits
#include <vector> // std::vector


namespace UIC {

/*---------------------------------------------------------------------------*
 | reformat time ranges (from Matrix to Binary)
 *---------------------------------------------------------------------------
 |    input : range_mat = [[2,4],[6,9]], num_time = 10
 |    output: range_bin = [F,T,T,T,F,T,T,T,T,F]
 *---------------------------------------------------------------------------*/
inline void reformat_range (
    std::vector<bool> *range_bin,
    const Rcpp::IntegerMatrix &range_mat,
    const int num_time)
{
    std::vector<bool>().swap(*range_bin);
    (*range_bin).resize(num_time, false);
    for (int i = 0; i < range_mat.rows(); ++i)
    {
        int t0 = range_mat(i,0) - 1;
        int t1 = range_mat(i,1);
        for (int t = t0; t < t1; ++ t) (*range_bin)[t] = true;
    }
}


/*---------------------------------------------------------------------------*
 | make time-delayed indicies
 *---------------------------------------------------------------------------*/
inline std::vector<int> make_delay_time (
    const int E,
    const bool reverse = false)
{
    std::vector<int> dt;
    if (reverse)
        for (int i = E; i > 0; --i) dt.push_back(i - 1);
    else
        for (int i = 0; i < E; ++i) dt.push_back(i);
    return dt;
}


/*---------------------------------------------------------------------------*
 | make time-delayed block
 *---------------------------------------------------------------------------*/
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
    range_grp.push_back({0,0});
    for (int t = 0, k = 0, g = group(0); t < nt; ++t)
    {
        if (group(t) != /*NA*/-2147483648) //INT32_MIN
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
    
    /* add time-delayed block to original block */
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

/*---------------------------------------------------------------------------*
 | check whether data hve NAs, reflecting 'group' vector
 *---------------------------------------------------------------------------*/
inline void check_NAs_group (
    Rcpp::IntegerVector &group,
    const Rcpp::NumericMatrix &Data_x,
    const Rcpp::NumericMatrix &Data_y)
{
    size_t nt = Data_x.rows();
    size_t nx = Data_x.cols();
    size_t ny = Data_y.cols();
    for (size_t t = 0; t < nt; ++t)
    {
        bool complete = true;
        for (size_t k = 0; k < nx && complete; ++k)
            if (std::isnan(Data_x(t,k))) complete = false;
        for (size_t k = 0; k < ny && complete; ++k)
            if (std::isnan(Data_y(t,k))) complete = false;
        if (!complete) group(t) = /*NA*/-2147483648; //INT32_MIN
    }
}

} //namespace UIC

#endif
//* End */