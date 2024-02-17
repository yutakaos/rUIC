/*****************************************************************************
 * Find Nearest Neighbors
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _find_neighbors_hpp_
#define _find_neighbors_hpp_

#include <vector> // std::vector

#include "./dataset.hpp"
#include "./method/bfnn.hpp"
#include "./method/nanoflann.hpp"


namespace UIC {

/*---------------------------------------------------------------------------*
 | Data class (point cloud)
 *---------------------------------------------------------------------------*/
template <typename num_t>
class DataCloud
{
    using data_t = std::vector<num_t>;
    
    const DataSet<num_t> &Data;
    const std::vector<int> &group;
    const std::vector<int> &time;
    int   exclusion_radius;
    num_t exclusion_dist;
    
    inline num_t DATA (int i, int j) const
    {
        int t = Data.idx_trn[i];
        if (j < Data.nz) return Data.z[t][j];
        j -= Data.nz;
        return Data.x[t-Data.dtx[j/Data.nx]][j%Data.nx];
    }
    
public:
    
    const nanoflann::NormAdaptor<num_t> norm;
    
    DataCloud (const DataSet<num_t> &Data)
        : Data(Data), group(Data.group), time(Data.idx_trn),
          exclusion_radius(Data.exclusion_radius), norm(Data.p)
    {
        exclusion_dist = Data.epsilon < 0 ? -1 : norm.pow(Data.epsilon, 0);
    }
    
    /* for KD-TREE */
    inline bool time_exclusion (const size_t idx, const int &qtime) const
    {
        if (group[time[idx]] != group[qtime]) return false;
        int radius = std::abs(time[idx] - qtime);
        if (radius > exclusion_radius) return false;
        return true;
    }
    
    inline bool dist_exclusion (const num_t dist) const
    {
        return dist <= exclusion_dist;
    }
    
    inline num_t eval_norm (
        const data_t &query, size_t idx, size_t size, num_t worst_dist = -1) const
    {
        num_t d = num_t();
        size_t dim = 0;
        while (dim + 3 < size)
        {
            d = norm.sum(d, norm.pow(query[dim], DATA(idx,dim))); ++dim;
            d = norm.sum(d, norm.pow(query[dim], DATA(idx,dim))); ++dim;
            d = norm.sum(d, norm.pow(query[dim], DATA(idx,dim))); ++dim;
            d = norm.sum(d, norm.pow(query[dim], DATA(idx,dim))); ++dim;
            if (worst_dist > 0 && d > worst_dist) return d;
        }
        while (dim < size)
        {
            d = norm.sum(d, norm.pow(query[dim], DATA(idx,dim))); ++dim;
        }
        return d;
    }
    
    inline size_t get_size () const { return Data.num_trn; }
    inline size_t get_dim  () const { return Data.dim_x; }
    inline num_t  get_pt (const size_t idx, const size_t dim) const
        { return DATA(idx,dim); }
};


/*---------------------------------------------------------------------------*
 | Find neighbors 
 |---------------------------------------------------------------------------*
 |    dim   : data dimension to compute distances (<= data.cols)
 |    index : k-NN indices
 |    dist  : k-NN distances
 *---------------------------------------------------------------------------*/
template <typename num_t>
inline void find_neighbors (
    const size_t dim,
    std::vector<std::vector<size_t>> &index,    
    std::vector<std::vector<num_t >> &dist,
    const DataSet<num_t>   &Data,
    const std::vector<int> &idx_val)
{
    size_t n_query = idx_val.size();
    std::vector<std::vector<size_t>>().swap(index);
    std::vector<std::vector<num_t >>().swap(dist);
    index.resize(n_query);
    dist .resize(n_query);
    
    int nn = (Data.nn == 0) ? dim + 1 : Data.nn;
    if (Data.knn_type == KNN_TYPE::KD) // KD-tree nearest neighbor search
    {
        DataCloud<num_t> cloud(Data);
        nanoflann::KDTreeInterface<num_t, DataCloud<num_t>> kdnn(dim, cloud);
        for (size_t i = 0; i < n_query; ++i)
        {
            kdnn.search(&index[i], &dist[i], Data.X(idx_val[i]), idx_val[i], nn);
        }
    }
    else // brute-force nearest neighbor search
    {
        std::vector<std::vector<num_t>> trn_data;
        std::vector<std::pair<int,int>> trn_time;
        for (auto t : Data.idx_trn)
        {
            trn_data.push_back(Data.X(t));
            trn_time.push_back({t, Data.group[t]});
        }
        bfnn<num_t> bfnn(Data.p, Data.exclusion_radius, Data.epsilon);
        bfnn.set_data(dim, trn_data, trn_time);
        for (size_t i = 0; i < n_query; ++i)
        {
            int t = idx_val[i];
            std::vector<num_t> val_data = Data.X(t);
            std::pair<int,int> val_time = {t, Data.group[t]};
            bfnn.search(&index[i], &dist[i], val_data, val_time, nn);
        }
    }
}

} //namespace UIC

#endif
//* End */