/*------------------------------------------------------------------------------------------#
 * Helper functions for UIC
 *------------------------------------------------------------------------------------------#
 */

#ifndef _uic_helper_hpp_
#define _uic_helper_hpp_

//* Header(s) */
#include <algorithm> // std::sort
#include <limits>    // std::numeric_limits
#include <vector>    // std::vector


namespace UIC
{
    //* Clear and resize std::vector */
    template <typename num_t, typename int_t = int>
    inline void clear_and_resize (std::vector<num_t> &vector, int_t size = 0)
    {
        std::vector<num_t>().swap(vector);
        vector.resize(size);
    }
    
    //* Combine two vecotrs */
    template <typename num_t>
    inline void bind (std::vector<num_t> *vector, std::vector<num_t> &vector_to_add)
    {
        int n = (*vector).size();
        int m = vector_to_add.size();
        (*vector).resize(n + m);
        for (int i = 0; i < m; ++i) (*vector)[n + i] = vector_to_add[i];
    }
    
    //* format time-ranges */
    inline void format_time_range (std::vector<std::pair<int, int>> *time_range)
    {
        typedef std::pair<int, int> pair_t;
        
        //* assure (begining time) <= (end time) */
        for (auto &tr : *time_range)
        {
            if (tr.first > tr.second) std::swap(tr.first, tr.second);
        }
        
        //* sort time-ranges along begining time */
        std::sort(
            (*time_range).begin(), (*time_range).end(),
            [](pair_t x, pair_t y) { return x.first < y.first; }
        );
        
        //* combine consecutive time-ranges */
        for (int i = (*time_range).size() - 1; i > 0; --i)
        {
            int i_begin = (*time_range)[i].first;
            int i_end   = (*time_range)[i].second;
            for (int j = i - 1; j >= 0; --j)
            {
                int &j_end = (*time_range)[j].second;
                if (i_begin <= j_end)
                {
                    if (j_end < i_end) j_end = i_end;
                    (*time_range).erase((*time_range).begin() + i);
                    break;
                }
            }
        }
    }
    
    //* Set time indices */
    inline void set_time_indices (
        std::vector<std::pair<int, int>> *indices, 
        const std::vector<std::pair<int, int>> &time_range, 
        const std::vector<std::pair<int, int>> &time_series)
    {
        clear_and_resize(*indices);
        int k = 0;
        for (size_t i = 0; i < time_range.size(); ++i)
        {
            if (time_series[k].second < time_range[i].first) ++k;
            int n = (*indices).size();
            int m = time_range[i].second - time_range[i].first + 1;
            (*indices).resize(n + m);
            for (int j = 0; j < m; ++j)
            {
                (*indices)[n + j].first  = time_range[i].first + j;
                (*indices)[n + j].second = k;
            }
        }
    }
    
    //* Construct time-delay embedding coordinates */
    template <typename num_t>
    inline void delay_embedding (
        std::vector<std::vector<num_t>> *delay_coord,
        const std::vector<std::vector<num_t>> &X,
        const std::pair<int, int> &time_range,
        const int E,
        const int tau,
        const bool forward = true)
    {
        const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
        int nd = X[0].size();
        int t0 = time_range.first;
        int t1 = time_range.second + 1;
        int nt = t1 - t0;
        
        clear_and_resize(*delay_coord, nt);
        for (int t = 0; t < nt; ++t)
        {
            (*delay_coord)[t].resize(E * nd);
            for (int i = 0; i < E; ++i)
            {
                int T = i * tau;
                int L = (forward ? i : E - i - 1) * nd;
                for (int k = 0; k < nd; ++k)
                {
                    (*delay_coord)[t][L + k] = t < T ? qnan : X[t0 + t - T][k];
                }
            }
        }
    }
    
    //* Set library data */
    template <typename num_t>
    inline void set_data_lib (
        std::vector<std::vector<num_t>> *output,
        const std::vector<std::vector<num_t>>  &data,
        const std::vector<std::pair<int, int>> &time_range,
        const int E = 1,
        const int tau = 1,
        const bool forward = true)
    {
        clear_and_resize(*output, 0);
        for (auto &tr : time_range)
        {
            std::vector<std::vector<num_t>> delay_coord;
            delay_embedding(&delay_coord, data, tr, E, tau, forward);
            bind(output, delay_coord);
        }
    }
    
    //* Set target data */
    template <typename num_t>
    inline void set_data_tar (
        std::vector<num_t> *output,
        const std::vector<num_t> &data,
        const std::vector<std::pair<int, int>> &time_range,
        const int tp)
    {
        const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
        clear_and_resize(*output);
        for (auto &tr : time_range)
        {
            for (int i = tr.first + tp; i <= tr.second + tp; ++i)
            {
                bool invalid = i < tr.first || tr.second < i;
                (*output).push_back(invalid ? qnan : data[i]);
            }
        }
    }
    
    //* Set valid library index */
    template <typename num_t>
    inline size_t set_valid_indices (
        std::vector<int> *valid_index,
        const std::vector<std::vector<num_t>> &data)
    {
        clear_and_resize(*valid_index);
        
        int n_data = data.size();
        for (int i = 0; i < n_data; ++i)
        {
            bool has_nan = false;
            for (auto &x : data[i])
            {
                if (std::isnan(x))
                {
                    has_nan = true;
                    break;
                }
            }
            if (!has_nan) (*valid_index).push_back(i);
        }
        return (*valid_index).size();
    }
    
    //* Find (nn)th nearest neighbors */
    template <typename num_t>
    inline void find_neighbors (
        std::vector<int> *neis,
        const int nn,
        const std::vector<num_t> &dist,
        std::vector<int> idx,
        const num_t epsilon)
    {
        //* sort */
        std::sort(
            idx.begin(), idx.end(),
            [&dist](size_t i, size_t j) { return dist[i] < dist[j]; }
        );
        
        clear_and_resize(*neis);
        if (dist[idx.back()] == 0)  // all distance == 0
        {
            (*neis) = idx;
            return;
        }
        
        //* count effective nearest neighbors */
        int n_idx = idx.size();
        int n_enn = std::min(nn, n_idx);
        for (int i = n_enn; i < n_idx; ++i)  // add ties
        {
            if (dist[idx[n_enn - 1]] != dist[idx[i]]) break;
            ++n_enn;
        }
        if (0.0 < epsilon) // filter by dist < epsilon
        {
            for (int i = n_enn - 1; 0 <= i; --i)
            {
                if (dist[idx[i]] < epsilon) break;
                --n_enn;
            }
        }
        
        //* copy results */
        (*neis).resize(n_enn);
        for (int i = 0; i < n_enn; ++i) (*neis)[i] = idx[i];
    }
}

#endif

// End