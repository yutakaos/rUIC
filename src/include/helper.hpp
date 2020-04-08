/*------------------------------------------------------------------------------------------#
 * Helper functions for ruic
 *------------------------------------------------------------------------------------------#
 */


#ifndef _ruic_helper_hpp_
#define _ruic_helper_hpp_

//* Header(s) */
#include <algorithm> // std::sort
#include <limits>    // std::numeric_limits
#include <vector>    // std::vector


namespace UIC
{
    //* Realign time-ranges */
    inline void realign_time_range (std::vector<std::pair<int, int>> *time_range)
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
        std::vector<std::pair<int, int>> *idx_time, 
        const std::vector<std::pair<int, int>> &time_range, 
        const std::vector<std::pair<int, int>> &time_series)
    {
        std::vector<std::pair<int, int>>().swap(*idx_time);
        int k = 0;
        for (size_t i = 0; i < time_range.size(); ++i)
        {
            if (time_series[k].second < time_range[i].first) ++k;
            int n = (*idx_time).size();
            int m = time_range[i].second - time_range[i].first + 1;
            (*idx_time).resize(n + m);
            for (int j = 0; j < m; ++j)
            {
                (*idx_time)[n + j].first  = time_range[i].first + j;
                (*idx_time)[n + j].second = k;
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
        
        std::vector<std::vector<num_t>>().swap(*delay_coord);
        (*delay_coord).resize(nt);
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
        const int E,
        const int tau,
        const bool forward = true)
    {
        std::vector<std::vector<num_t>>().swap(*output);
        for (auto &tr : time_range)
        {
            std::vector<std::vector<num_t>> tlc;
            delay_embedding(&tlc, data, tr, E, tau, forward);
            int n = (*output).size();
            int m = tlc.size();
            (*output).resize(n + m);
            for (int i = 0; i < m; ++i) (*output)[n + i] = tlc[i];
        }
    }
    
    //* Add library data */
    template <typename num_t>
    inline void add_data_lib (
        std::vector<std::vector<num_t>> *output,
        const std::vector<std::vector<num_t>>  &data,
        const std::vector<std::pair<int, int>> &time_range,
        const int E   = 1,
        const int tau = 1)
    {
        std::vector<std::vector<num_t>> x;
        set_data_lib(&x, data, time_range, E, tau, true);
        
        int nt = (*output).size();
        int nd = (*output)[0].size();
        int nx = x[0].size();
        for (int i = 0; i < nt; ++i)
        {
            (*output)[i].resize(nd + nx);
            for (int j = nd - 1; j >= 0; --j) (*output)[i][nx + j] = (*output)[i][j];
            for (int j = 0; j < nx; ++j) (*output)[i][j] = x[i][j];
        }
    }
    
    //* Set valid library index */
    template <typename num_t>
    inline size_t set_valid_indices (
        std::vector<int> *valid_index,
        const std::vector<std::vector<num_t>> &data)
    {
        std::vector<int>().swap(*valid_index);
        
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
    
    //* Set target data */
    template <typename num_t>
    inline void set_data_tar (
        std::vector<num_t> *output,
        const std::vector<num_t> &data,
        const std::vector<std::pair<int, int>> &time_range,
        const int tp)
    {
        const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
        std::vector<num_t>().swap(*output);
        for (auto &tr : time_range)
        {
            for (int i = tr.first + tp; i <= tr.second + tp; ++i)
            {
                (*output).push_back(i < tr.first || tr.second < i ? qnan : data[i]);
            }
        }
    }
    
    //* Find (nn)-th nearest neighbors */
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
        
        std::vector<int>().swap(*neis);
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
    
    //* Compute weights */
    template <typename num_t>
    inline void compute_weights (
        std::vector<num_t> *weight,
        const std::vector<num_t> &dist,
        const std::vector<int>   &neis)
    {
        size_t nn = neis.size();
        std::vector<num_t>().swap(*weight);
        (*weight).resize(nn);
        
        std::function<num_t (num_t, num_t)> calc_w;
        if (dist[neis[0]] == 0)
            calc_w = [](num_t d, num_t    ) { return d == 0; };
        else
            calc_w = [](num_t d, num_t ref) { return std::exp(-d / ref); };
        
        num_t sum_w = 0;
        for (size_t i = 0; i < nn; ++i)
        {
            (*weight)[i] = calc_w(dist[neis[i]], dist[neis[0]]);
            sum_w += (*weight)[i];
        }
        for (auto &x : *weight) x /= sum_w;
    }
}

#endif

// End