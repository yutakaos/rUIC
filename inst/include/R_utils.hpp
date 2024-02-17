/*****************************************************************************
 * Utility Functions
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
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
 | convert data format from R to cpp
 *---------------------------------------------------------------------------*/
template <typename num_t>
inline void as_cpp (
    std::vector<std::vector<num_t>> *matC,
    const Rcpp::NumericMatrix &matR)
{
    std::vector<std::vector<num_t>>().swap(*matC);
    size_t nrow = matR.rows();
    size_t ncol = matR.cols();
    
    (*matC).resize(nrow);
    for (size_t i = 0; i < nrow; ++i)
    {
        (*matC)[i].resize(ncol);
        for (size_t j = 0; j < ncol; ++j)
        {
            (*matC)[i][j] = num_t(matR(i,j));
        }
    }
}

template <typename num_t=int>
inline void as_cpp (
    std::vector<num_t> *vecC,
    const Rcpp::IntegerVector &vecR)
{
    std::vector<num_t>().swap(*vecC);
    size_t size = vecR.size();
    
    num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    (*vecC).resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        (*vecC)[i] = (vecR[i] == -2147483648) ? qnan : num_t(vecR(i));
        // NA is -2147483648 (INT32_MIN) for R::Integer
    }
}

} //namespace UIC

#endif
//* End */