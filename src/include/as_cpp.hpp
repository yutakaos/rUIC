/*------------------------------------------------------------------------------------------#
 * Convert Rcpp vectors into std::vector
 *------------------------------------------------------------------------------------------#
 */


#ifndef _ruic_as_cpp_hpp_
#define _ruic_as_cpp_hpp_

//* Header(s) */
#include <Rcpp.h>
#include <vector> // std::vector


template <typename num_t>
std::vector<std::vector<num_t>> as_cpp (const Rcpp::NumericMatrix &M)
{
    size_t nrow = M.rows();
    size_t ncol = M.cols();
    if (nrow == 1 && ncol == 1 && std::isnan(M(0, 0))) return {};
    
    num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    std::vector<std::vector<num_t>> op(nrow);
    for (size_t i = 0; i < nrow; ++i)
    {
        op[i].resize(ncol);
        for (size_t j = 0; j < ncol; ++j)
        {
            op[i][j] = std::isnan(M(i, j)) ? qnan : num_t(M(i, j));
        }
    }
    return op;
}

template <typename num_t>
std::vector<num_t> as_cpp (const Rcpp::NumericVector &V)
{
    size_t size = V.size();
    if (size == 1 && std::isnan(V(0))) return {};
    
    num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    std::vector<num_t> op(size);
    for (size_t i = 0; i < size; ++i)
    {
        op[i] = std::isnan(V(i)) ? qnan : num_t(V(i));
    }
    return op;
}

std::vector<std::pair<int, int>> as_cpp_range (Rcpp::IntegerMatrix &range)
{
    size_t nrow = range.rows();
    std::vector<std::pair<int, int>> op(nrow);
    for (size_t i = 0; i < nrow; ++i)
    {
        op[i].first  = range(i, 0) - 1;
        op[i].second = range(i, 1) - 1;
    }
    return op;
}

#endif

// End