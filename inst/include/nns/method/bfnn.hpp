/*****************************************************************************
 * Brute-force Nearest Neighbor Search
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _bfnn_hpp_
#define _bfnn_hpp_

#include <algorithm> // std::sort
#include <vector>    // std::vector


namespace UIC {

template <typename num_t>
class bfnn
{
    using data_t = std::vector<num_t>;
    using time_t = std::pair<int,int>;
    
    std::vector<data_t> data;
    std::vector<time_t> time;
    size_t nt;  // number of times
    size_t nd;  // data dimensions
    size_t dim; // data dimensions to compute distances (<= nd)
    int exclusion_radius;
    num_t p, epsilon;
    
    std::function<num_t(num_t, num_t)> Sum;
    std::function<num_t(num_t, num_t)> Pow;
    std::function<num_t(num_t)> Root;
    
public:
    
    bfnn (num_t p = 2, int exclusion_radius = -1, num_t epsilon = -1)
    {
        set_norm(p);
        this->exclusion_radius = exclusion_radius;
        this->epsilon = epsilon;
    }
    
    inline void set_norm (num_t p = 2)
    {
        this->p = p;
        if      (p == 2.0) set_L2 ();
        else if (p == 1.0) set_L1 ();
        else if (p <= 0.0) set_Max();
        else set_Lp();
    }
    
    inline void set_data (
        const size_t dim,
        const std::vector<data_t> &data,
        const std::vector<time_t> &time)
    {
        std::vector<data_t>().swap(this->data);
        std::vector<time_t>().swap(this->time);
        this->data = data;
        this->time = time;
        nt = data.size();
        nd = data[0].size();
        this->dim = dim < nd ? dim : nd;
    }
    
    inline size_t search (
        std::vector<size_t> *index,
        std::vector<num_t>  *dist,
        const data_t &query,
        const time_t &qtime, // query time indices
        size_t num_nearest,
        bool tied = true) const
    {
        /* compute distances */
        std::vector<num_t> ds(nt);
        for (size_t i = 0; i < nt; ++i) ds[i] = eval_norm(query, i, dim);
        
        /* sort indices */
        std::vector<size_t> idx;
        for (size_t t = 0; t < nt; ++t) idx.push_back(t);
        std::sort(
            idx.begin(), idx.end(),
            [&ds](size_t i, size_t j) { return ds[i] < ds[j]; }
        );
        
        /* search neighbors */
        size_t n = 0;
        for (size_t i = 0; i < nt; ++i)
        {
            size_t k = idx[i];
            if (time_exclusion(k, qtime)) continue;
            num_t d = Root(ds[k]);
            if (d <= epsilon) continue;
            (*index).push_back(k);
            (*dist ).push_back(d);
            if (++n == num_nearest)
            {
                if (tied)
                {
                    for (++i; i < nt; ++i)
                    {
                        k = idx[i];
                        if (time_exclusion(k, qtime)) continue;
                        if (d != Root(ds[k])) break;
                        (*index).push_back(k);
                        (*dist ).push_back(d);
                        ++n;
                    }
                }
                break;
            }
        }
        return n;
    }
    
private:
    
    inline bool time_exclusion (size_t i, const time_t &q) const
    {
        if (time[i].second != q.second) return false;
        if (std::abs(time[i].first - q.first) > exclusion_radius) return false;
        return true;
    }
    
    inline num_t eval_norm (const data_t &query, size_t i, size_t size) const
    {
        num_t d = num_t();
        for (size_t k = 0; k < size; ++k)
        {
            d = Sum(d, Pow(query[k], data[i][k]));
        }
        return d;
    }
    
    void set_L2 ()
    {
        Sum  = [](num_t a, num_t b) { return a + b; };
        Pow  = [](num_t a, num_t b) { return (a - b) * (a - b); };
        Root = [](num_t a) { return std::sqrt(a); };
    }
    
    void set_L1 ()
    {
        Sum  = [](num_t a, num_t b) { return a + b; };
        Pow  = [](num_t a, num_t b) { return std::abs(a - b); };
        Root = [](num_t a) { return a; };
    }
    
    void set_Max ()
    {
        Sum  = [](num_t a, num_t b) { return a > b ? a : b; };
        Pow  = [](num_t a, num_t b) { return std::abs(a - b); };
        Root = [](num_t a) { return a; };
    }
    
    void set_Lp ()
    {
        Sum  = [    ](num_t a, num_t b) { return a + b; };
        Pow  = [this](num_t a, num_t b) { return std::pow(std::abs(a - b), p); };
        Root = [this](num_t a) { return std::pow(a, 1.0 / p); };
    }
};

} //namespace UIC

#endif
//* End */